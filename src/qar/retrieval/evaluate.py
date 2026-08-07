"""Run a retriever over a prepared split and produce the baseline table.

Two details decide whether the resulting numbers are honest.

**Pools are shuffled before scoring.** Lexical scorers return 0.0 for every snippet
when the question shares no words with any of them, and `argsort` then falls back
to position -- silently handing those rows to snippet 0. Since the pool may carry
an upstream ordering, that would quietly blend the `first` baseline into every
other row of the table. Shuffling per row makes an unscored row land at chance,
which is what "the method had nothing to say" should look like.

**The whole split is scored into one padded matrix.** 44k rows by at most 32
candidates is 6 MB, so there is no reason to chunk, and one matrix means the
metrics come from `qar.eval.metrics` unchanged rather than from a re-implementation
that averages chunks slightly differently.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from qar.config import RunConfig
from qar.data.dataset import PairDataset
from qar.eval.metrics import ranking_metrics
from qar.utils.logging import get_logger

log = get_logger(__name__)


def evaluate_retriever(cfg: RunConfig, retriever, dataset: PairDataset) -> dict[str, Any]:
    """Rank every row's pool and return metrics, with per-slice breakdowns."""
    rng = random.Random(cfg.retrieval.tie_break_seed)
    rows: list[list[float]] = []
    targets: list[int] = []
    pool_sizes: list[int] = []
    question_types: list[str] = []
    answerable: list[int] = []

    limit = min(cfg.retrieval.max_rows or len(dataset), len(dataset))
    chunk = max(1, cfg.retrieval.batch_rows)

    # Rows are shuffled and grouped here, then handed to the retriever in chunks.
    # The per-row semantics are identical -- `score_batch` defaults to the same loop
    # -- but a neural retriever can encode a whole chunk in one forward pass.
    for start in range(0, limit, chunk):
        queries, pools = [], []
        for index in range(start, min(start + chunk, limit)):
            record = dataset[index]
            pool = record["snippets"]

            order = list(range(len(pool)))
            rng.shuffle(order)

            queries.append(record["question"])
            pools.append([pool[i] for i in order])
            targets.append(order.index(record["positive_idx"]))
            pool_sizes.append(len(pool))
            question_types.append(record["question_type"])
            answerable.append(record["is_answerable"])

        rows.extend(retriever.score_batch(queries, pools))

    if not rows:
        raise ValueError("split is empty; nothing to evaluate")

    matrix = _pad(rows)
    target = torch.tensor(targets, dtype=torch.long)
    ks = tuple(cfg.retrieval.ks)

    result: dict[str, Any] = {
        "rows": len(rows),
        "mean_pool": round(sum(pool_sizes) / len(pool_sizes), 2),
        "overall": _round(ranking_metrics(matrix, target, ks=ks)),
        "by_question_type": {},
        "by_answerable": {},
    }

    for group, name, mask in _slices(question_types, answerable):
        if mask.any():
            result[group][name] = {
                "rows": int(mask.sum()),
                **_round(ranking_metrics(matrix[mask], target[mask], ks=ks)),
            }
    return result


def _pad(rows: list[list[float]]) -> torch.Tensor:
    """Rectangular score matrix; absent candidates get -inf so they always rank last."""
    if not rows:
        return torch.zeros((0, 1))
    width = max(len(row) for row in rows)
    matrix = torch.full((len(rows), width), float("-inf"))
    for index, row in enumerate(rows):
        if row:
            matrix[index, : len(row)] = torch.tensor(row, dtype=torch.float32)
    return matrix


def _slices(question_types: list[str], answerable: list[int]):
    """(group, name, row mask) for every breakdown worth reporting separately.

    Yes-no questions are 15% of the corpus and behave differently from descriptive
    ones -- a lexical method has much less to grip on in "does it fit?". Unanswerable
    rows still carry an inferred positive, and a retriever scoring *well* on them is
    suspicious rather than impressive: there was no supporting evidence to find.
    """
    for name in sorted(set(question_types)):
        yield "by_question_type", name, torch.tensor([t == name for t in question_types])

    yield "by_answerable", "answerable", torch.tensor([a == 1 for a in answerable])
    yield "by_answerable", "unanswerable", torch.tensor([a == 0 for a in answerable])


def _round(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 4) for key, value in metrics.items()}


def merge_results(path: Path, fresh: dict[str, Any]) -> dict[str, Any]:
    """Fold fresh rows into whatever `path` already holds.

    The results file is named after the split alone, so before this a run scoring a
    single retriever replaced the entire table: evaluating a checkpoint with
    `retrieval.baselines=[dense]` silently deleted six lexical rows that had cost
    minutes to produce. Merging makes "measure the baselines once, add the trained
    model later" the normal workflow instead of a footgun.

    Rows measured over a **different number of rows** are dropped rather than kept.
    They came from another corpus or from a `max_rows` probe, and a table putting
    those beside a full-split number, with nothing on the line to say so, is worse
    than a table missing them.
    """
    if not path.exists():
        return fresh

    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("results", {})
    except (json.JSONDecodeError, OSError):
        log.warning("%s is unreadable; starting a fresh table", path.name)
        return fresh

    rows = next(iter(fresh.values()))["rows"]
    kept, dropped = {}, []
    for name, result in previous.items():
        if name in fresh:
            continue  # superseded by this run
        if result.get("rows") == rows:
            kept[name] = result
        else:
            dropped.append(f"{name} ({result.get('rows')} rows)")

    if dropped:
        log.warning(
            "dropped %d stale row(s) measured over a different split size: %s",
            len(dropped), ", ".join(dropped),
        )
    if kept:
        log.info("kept %d existing row(s): %s", len(kept), ", ".join(kept))

    return {**kept, **fresh}  # fresh last, so the table ends with this run


def markdown_table(results: dict[str, dict[str, Any]], ks: list[int]) -> str:
    """The baseline table, paste-ready for the report."""
    columns = [f"recall@{k}" for k in ks] + ["mrr", "mean_rank"]
    header = "| retriever | " + " | ".join(columns) + " |"
    rule = "| --- | " + " | ".join("---" for _ in columns) + " |"

    lines = [header, rule]
    for name, result in results.items():
        overall = result["overall"]
        cells = [f"{overall.get(column, float('nan')):.4f}" for column in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
