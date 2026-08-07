"""One offline pass that turns the raw corpus into everything training needs.

Reads the raw JSONL files, infers a positive snippet per question, fits the BPE
vocabulary, and records what it did in a manifest.

Every number the report quotes about the *data* -- how many rows survived, how
many had no trustworthy positive, how the answerable classes fell -- comes from
that manifest rather than from a notebook cell that no longer exists.

    train file --> train.jsonl  (+ the text sample the tokenizer is fitted on)
    val file   --> val.jsonl
    test file  --> test.jsonl

The upstream corpus ships a product-disjoint test file, so each split is simply
its own pass. When `data.test_path` is null the older fallback applies instead:
validation is divided by hashed asin, which keeps each product whole but costs
half the validation rows.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any, TextIO

from qar.config import RunConfig
from qar.data import select as _selectors  # noqa: F401  (registers the selectors)
from qar.data.schema import BASELINE_FIELDS, RawRow, parse_row
from qar.data.split import assign_split
from qar.data.text import normalise
from qar.data.tokenizer import train_tokenizer
from qar.registry import build
from qar.utils.logging import get_logger

log = get_logger(__name__)

_LOG_EVERY = 50_000


@dataclass
class SplitStats:
    """Row-level accounting for one output split. Serialised into the manifest."""

    rows_read: int = 0
    no_positive: int = 0
    kept: int = 0
    answerable: int = 0
    snippets: int = 0
    asins: set[str] = field(default_factory=set)
    question_groups: set[str] = field(default_factory=set)
    score_sum: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "no_positive": self.no_positive,
            "kept": self.kept,
            "answerable": self.answerable,
            "answerable_frac": round(self.answerable / self.kept, 4) if self.kept else 0.0,
            "unique_asins": len(self.asins),
            "unique_questions": len(self.question_groups),
            "snippets": self.snippets,
            "snippets_per_row": round(self.snippets / self.kept, 2) if self.kept else 0.0,
            "mean_positive_score": round(self.score_sum / self.kept, 4) if self.kept else 0.0,
        }


def question_group(question: str) -> str:
    """Short stable id for a normalised question string.

    Two rows sharing one lets the in-batch sampler avoid making each row's positive
    the other's negative -- the false-negative guard `data.dedup_questions_in_batch`
    turns on. Computed here so the sampler never has to re-read the corpus.
    """
    return blake2b(normalise(question).encode("utf-8"), digest_size=8).hexdigest()


def prepare(cfg: RunConfig) -> dict[str, Any]:
    """Build the processed corpus described by `cfg`. Returns the manifest."""
    started = time.time()
    raw_train, raw_val = Path(cfg.data.train_path), Path(cfg.data.val_path)
    for path in (raw_train, raw_val):
        if not path.exists():
            raise FileNotFoundError(f"raw corpus not found: {path.resolve()}")

    # A configured-but-missing test file is an error, not a silent downgrade to the
    # carve-from-val fallback: the two produce different corpora, and discovering
    # that from a manifest after a 20-minute run is too late.
    raw_test = Path(cfg.data.test_path) if cfg.data.test_path else None
    if raw_test is not None and not raw_test.exists():
        raise FileNotFoundError(
            f"data.test_path is set but not found: {raw_test.resolve()}\n"
            "Set data.test_path=null to carve a test split out of validation instead."
        )

    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = _resolve_selector(cfg.prepare.selector)

    # -- pass 1: train ----------------------------------------------------- #
    sample = _Reservoir(cfg.prepare.tokenizer_sample_docs, seed=cfg.prepare.split_seed)
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        train_split, train_malformed = _write_split(
            raw_train, {"train": handle}, cfg, selector,
            route=lambda row: "train", sample=sample,
        )
    train_stats = train_split["train"]

    # -- vocabulary, fitted on the train split only ------------------------ #
    log.info("fitting BPE (vocab=%d) on %d sampled texts", cfg.model.vocab_size, len(sample))
    train_tokenizer(sample.items, cfg.model.vocab_size, out_dir / "tokenizer.json")

    # -- pass 2: evaluation splits ----------------------------------------- #
    if raw_test is None:
        # Fallback: no upstream test file, so carve one out of validation. Hashing
        # the asin keeps every product whole on one side.
        with (out_dir / "val.jsonl").open("w", encoding="utf-8") as val_handle, \
             (out_dir / "test.jsonl").open("w", encoding="utf-8") as test_handle:
            eval_stats, val_malformed = _write_split(
                raw_val, {"val": val_handle, "test": test_handle}, cfg, selector,
                route=lambda row: assign_split(
                    row.asin, cfg.prepare.test_fraction, cfg.prepare.split_seed
                ),
            )
        val_stats, test_stats = eval_stats["val"], eval_stats["test"]
        test_malformed = 0
    else:
        # The upstream test file is already product-disjoint from both train and
        # validation, so validation is kept whole and each file is simply copied
        # through its own pass.
        with (out_dir / "val.jsonl").open("w", encoding="utf-8") as handle:
            stats, val_malformed = _write_split(
                raw_val, {"val": handle}, cfg, selector, route=lambda row: "val",
            )
        val_stats = stats["val"]
        with (out_dir / "test.jsonl").open("w", encoding="utf-8") as handle:
            stats, test_malformed = _write_split(
                raw_test, {"test": handle}, cfg, selector, route=lambda row: "test",
            )
        test_stats = stats["test"]

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - started, 1),
        "sources": {
            "train": str(raw_train),
            "val": str(raw_val),
            "test": str(raw_test) if raw_test is not None else None,
        },
        # Which of the two split regimes produced test.jsonl. Corpora built under
        # different regimes have different val sizes and are not comparable.
        "test_source": "upstream_file" if raw_test is not None else "carved_from_val",
        "prepare": {
            "selector": cfg.prepare.selector,
            "min_positive_score": cfg.prepare.min_positive_score,
            "min_snippet_tokens": cfg.prepare.min_snippet_tokens,
            "max_snippets": cfg.prepare.max_snippets,
            # Meaningless under the upstream regime; recorded as null so a manifest
            # cannot suggest a carve that never happened.
            "test_fraction": cfg.prepare.test_fraction if raw_test is None else None,
            "split_seed": cfg.prepare.split_seed,
            "max_rows": cfg.prepare.max_rows,
        },
        "vocab_size": cfg.model.vocab_size,
        "ignored_raw_fields": list(BASELINE_FIELDS),
        # Counted per source file, not per split: a row that will not parse has no
        # asin, so there is no split it could honestly be attributed to.
        "malformed_rows": {
            "train": train_malformed, "val": val_malformed, "test": test_malformed
        },
        "splits": {
            "train": train_stats.to_dict(),
            "val": val_stats.to_dict(),
            "test": test_stats.to_dict(),
        },
        "leakage": _leakage_report(train_stats, val_stats, test_stats),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


# --------------------------------------------------------------------------- #


def _resolve_selector(name: str):
    """Bind the selector once so a bad name fails before the corpus is read."""

    def run(snippets: list[str], answers: list[str]) -> tuple[int, float]:
        return build("selector", name, snippets, answers)

    run(["probe text long enough"], ["probe"])  # fails now, not 20 minutes in
    return run


def _write_split(
    source: Path,
    handles: dict[str, TextIO],
    cfg: RunConfig,
    selector,
    route,
    sample: _Reservoir | None = None,
) -> tuple[dict[str, SplitStats], int]:
    """Stream `source`, route each surviving row to a handle, and tally as we go.

    Returns the per-split tallies and the number of rows that never parsed.
    """
    stats = {name: SplitStats() for name in handles}
    malformed = 0
    log.info("reading %s", source)

    for index, raw in enumerate(_iter_json(source, cfg.prepare.max_rows), start=1):
        if index % _LOG_EVERY == 0:
            log.info("  %s: %d rows", source.name, index)

        row = parse_row(
            raw,
            min_snippet_tokens=cfg.prepare.min_snippet_tokens,
            max_snippets=cfg.prepare.max_snippets,
        )
        if row is None:
            malformed += 1
            continue

        target = route(row)
        tally = stats[target]
        tally.rows_read += 1

        positive_idx, score = selector(row.snippets, row.answers)
        if positive_idx < 0 or score < cfg.prepare.min_positive_score:
            tally.no_positive += 1
            continue

        record = _record(row, positive_idx, score)
        handles[target].write(json.dumps(record, ensure_ascii=False) + "\n")

        tally.kept += 1
        tally.answerable += row.is_answerable
        tally.snippets += len(row.snippets)
        tally.asins.add(row.asin)
        tally.question_groups.add(record["qgroup"])
        tally.score_sum += score

        if sample is not None:
            sample.offer(row.question)
            for snippet in row.snippets:
                sample.offer(snippet)

    for name, tally in stats.items():
        log.info("  -> %s: kept %d of %d rows", name, tally.kept, tally.rows_read)
    if malformed:
        log.info("  -> %s: %d unparseable rows skipped", source.name, malformed)
    return stats, malformed


def _record(row: RawRow, positive_idx: int, score: float) -> dict[str, Any]:
    """The processed schema. `snippets` is the retrieval pool; the positive is an
    index into it rather than a copy, so the two can never drift apart."""
    return {
        "qid": row.qid,
        "asin": row.asin,
        "category": row.category,
        "question": row.question,
        "question_type": row.question_type,
        "is_answerable": row.is_answerable,
        "qgroup": question_group(row.question),
        "positive_idx": positive_idx,
        "positive_score": round(score, 4),
        "snippets": row.snippets,
    }


def _iter_json(path: Path, max_rows: int | None) -> Iterator[dict[str, Any]]:
    """Yield parsed lines, skipping any that are not valid JSON objects."""
    with path.open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            if max_rows is not None and count > max_rows:
                return
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _leakage_report(train: SplitStats, val: SplitStats, test: SplitStats) -> dict[str, Any]:
    """The check the whole split design exists to pass.

    Product overlap must be ~0 between every pair. Under the carve-from-val regime
    `asin_overlap_val_test` is exactly 0 by construction; under the upstream regime
    it is small but non-zero (15 products in the raw files), inherited from how the
    corpus authors built their splits rather than introduced here.

    Question overlap is expected and is **not** leakage: generic phrasing ("does it
    fit?") recurs across products whose reviews are disjoint.
    """
    return {
        "asin_overlap_train_val": len(train.asins & val.asins),
        "asin_overlap_train_test": len(train.asins & test.asins),
        "asin_overlap_val_test": len(val.asins & test.asins),
        "question_overlap_train_val": len(train.question_groups & val.question_groups),
        "question_overlap_val_test": len(val.question_groups & test.question_groups),
    }


class _Reservoir:
    """Uniform sample of a stream of unknown length, in one pass (Vitter's R).

    Taking the first N texts instead would be cheaper and wrong: the corpus is
    grouped by product, so the head of the file is a handful of categories and the
    vocabulary would be fitted on cameras.
    """

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = max(0, capacity)
        self.items: list[str] = []
        self.seen = 0
        self._rng = random.Random(seed)

    def offer(self, text: str) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(text)
            return
        if self.capacity:
            slot = self._rng.randrange(self.seen)
            if slot < self.capacity:
                self.items[slot] = text

    def __len__(self) -> int:
        return len(self.items)
