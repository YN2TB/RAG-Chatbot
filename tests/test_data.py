"""Corpus preparation, on a synthetic corpus shaped like the real one.

Everything here runs on a handful of rows written into tmp_path. The real files
are 3.4 GB and must never be touched by the suite -- but the schema below is the
one measured off `amazonqa_validation.jsonl`, including the four precomputed
baseline fields the parser is expected to ignore.
"""

from __future__ import annotations

import json

import pytest

from qar.config import load_config
from qar.data.dataset import PairCollator, PairDataset
from qar.data.prepare import prepare, question_group
from qar.data.schema import parse_row
from qar.data.split import assign_split
from qar.data.text import token_f1, token_recall
from qar.data.tokenizer import load_tokenizer, pad_id
from qar.registry import available, build


def _raw_row(qid, asin, question, snippets, answers, answerable=1):
    """A raw row carrying the baseline fields the pipeline must drop."""
    return {
        "qid": qid,
        "asin": asin,
        "category": "Electronics",
        "questionText": question,
        "questionType": "descriptive",
        "review_snippets": snippets,
        "answers": [{"answerText": a, "answerType": "NA", "helpful": [1, 1]} for a in answers],
        "is_answerable": answerable,
        "random_sentence": ["ignored"],
        "top_sentences_IR": ["ignored"],
        "top_review_wilson": ["ignored"],
        "top_review_helpful": ["ignored"],
    }


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _corpus(tmp_path, n_products=12):
    """Train and validation files with disjoint products, as the real corpus has."""
    train, val = [], []
    for p in range(n_products):
        for q in range(3):
            train.append(_raw_row(
                f"t{p}-{q}", f"TRAIN{p:03d}",
                f"how long does the battery of model {p} last in cold weather",
                [
                    f"the case of model {p} is made of moulded plastic and feels sturdy enough",
                    f"battery of model {p} lasts about nine hours even in cold weather outdoors",
                    f"shipping was quick and the box for model {p} arrived without any damage",
                ],
                [f"battery lasts about nine hours in cold weather on model {p}"],
            ))
        val.append(_raw_row(
            f"v{p}", f"VALID{p:03d}",
            f"is the strap of unit {p} removable without any tools",
            [
                f"the strap on unit {p} detaches by hand without tools which is convenient",
                f"colour of unit {p} is slightly darker than the product photograph shows",
            ],
            [f"yes the strap on unit {p} comes off by hand"],
        ))
    return _write(tmp_path / "raw_train.jsonl", train), _write(tmp_path / "raw_val.jsonl", val)


def _cfg(tmp_path, **overrides):
    train, val = _corpus(tmp_path)
    base = [
        f"data.train_path={train.as_posix()}",
        f"data.val_path={val.as_posix()}",
        f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
        "model.vocab_size=800",
        "prepare.tokenizer_sample_docs=500",
    ]
    return load_config("configs/base.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


# -- text ------------------------------------------------------------------- #


def test_normalisation_ignores_articles_and_punctuation():
    assert token_f1("The battery lasts!", "battery lasts") == pytest.approx(1.0)


def test_recall_ignores_candidate_length_but_f1_does_not():
    long_snippet = "battery lasts " + "unrelated filler words here " * 5
    assert token_recall(long_snippet, "battery lasts") == pytest.approx(1.0)
    assert token_f1(long_snippet, "battery lasts") < 0.4


# -- schema ----------------------------------------------------------------- #


def test_parse_row_drops_short_snippets_and_baseline_fields():
    raw = _raw_row("q1", "A1", "does it fit", ["too short", "this snippet is long enough here"],
                   ["it fits"])
    row = parse_row(raw, min_snippet_tokens=5, max_snippets=32)
    assert row.snippets == ["this snippet is long enough here"]
    assert not hasattr(row, "top_sentences_IR")


def test_parse_row_rejects_rows_with_no_usable_snippet():
    raw = _raw_row("q1", "A1", "does it fit", ["short"], ["yes"])
    assert parse_row(raw, min_snippet_tokens=5, max_snippets=32) is None


def test_parse_row_caps_the_pool():
    raw = _raw_row("q1", "A1", "does it fit",
                   [f"snippet number {i} is long enough to survive" for i in range(50)], ["yes"])
    row = parse_row(raw, min_snippet_tokens=5, max_snippets=8)
    assert len(row.snippets) == 8


# -- selectors -------------------------------------------------------------- #


def test_selectors_are_registered():
    assert {"answer_overlap", "answer_recall", "first"} <= set(available("selector"))


def test_answer_overlap_picks_the_supporting_snippet():
    snippets = ["the box arrived on time and undamaged", "the battery lasts about nine hours"]
    index, score = build("selector", "answer_overlap", snippets, ["battery lasts nine hours"])
    assert index == 1 and score > 0.5


def test_first_selector_is_a_null_control():
    snippets = ["the box arrived on time", "the battery lasts about nine hours"]
    assert build("selector", "first", snippets, ["battery lasts nine hours"]) == (0, 1.0)


# -- split ------------------------------------------------------------------ #


def test_split_is_deterministic_per_product():
    assert all(
        assign_split("B0001", 0.5, 7) == assign_split("B0001", 0.5, 7) for _ in range(5)
    )


def test_split_fractions_are_exhaustive():
    asins = [f"P{i:05d}" for i in range(2000)]
    assert all(assign_split(a, 0.0, 1) == "val" for a in asins)
    assert all(assign_split(a, 1.0, 1) == "test" for a in asins)
    share = sum(assign_split(a, 0.5, 1) == "test" for a in asins) / len(asins)
    assert 0.45 < share < 0.55, f"asin hashing is not uniform: {share:.3f}"


def test_split_rejects_impossible_fraction():
    with pytest.raises(ValueError):
        assign_split("B0001", 1.5, 1)


# -- prepare, end to end ---------------------------------------------------- #


def test_prepare_writes_every_artifact(tmp_path):
    cfg = _cfg(tmp_path)
    manifest = prepare(cfg)
    out = tmp_path / "processed"

    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "tokenizer.json", "manifest.json"):
        assert (out / name).exists(), f"{name} was not written"

    assert manifest["splits"]["train"]["kept"] == 36
    assert manifest["vocab_size"] == 800
    assert manifest["ignored_raw_fields"]


def test_prepare_keeps_products_disjoint(tmp_path):
    manifest = prepare(_cfg(tmp_path))
    leakage = manifest["leakage"]
    assert leakage["asin_overlap_train_val"] == 0
    assert leakage["asin_overlap_train_test"] == 0
    assert leakage["asin_overlap_val_test"] == 0, "a product landed in both val and test"


def test_prepare_routes_validation_into_both_splits(tmp_path):
    manifest = prepare(_cfg(tmp_path))
    val, test = manifest["splits"]["val"]["kept"], manifest["splits"]["test"]["kept"]
    assert val > 0 and test > 0, f"degenerate split: val={val} test={test}"
    assert val + test == 12


def test_prepare_selects_the_supporting_snippet(tmp_path):
    prepare(_cfg(tmp_path))
    records = [
        json.loads(line)
        for line in (tmp_path / "processed" / "train.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all("battery" in r["snippets"][r["positive_idx"]] for r in records)
    assert all(r["positive_score"] > 0.5 for r in records)


def test_prepare_drops_rows_below_the_score_threshold(tmp_path):
    manifest = prepare(_cfg(tmp_path, **{"prepare.min_positive_score": 0.99}))
    assert manifest["splits"]["train"]["kept"] == 0
    assert manifest["splits"]["train"]["no_positive"] == 36


def test_prepare_survives_a_malformed_line(tmp_path):
    cfg = _cfg(tmp_path)
    path = tmp_path / "raw_train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    assert prepare(cfg)["splits"]["train"]["kept"] == 36


def test_unparseable_rows_are_counted_once_per_source(tmp_path):
    """A row with no asin cannot be attributed to a split, so it must not be
    tallied into every split's `rows_read`."""
    cfg = _cfg(tmp_path)
    val = tmp_path / "raw_val.jsonl"
    val.write_text(
        val.read_text(encoding="utf-8") + json.dumps({"questionText": "orphan"}) + "\n",
        encoding="utf-8",
    )
    manifest = prepare(cfg)

    assert manifest["malformed_rows"] == {"train": 0, "val": 1}
    splits = manifest["splits"]
    assert splits["val"]["rows_read"] + splits["test"]["rows_read"] == 12


def test_max_rows_bounds_the_pass(tmp_path):
    manifest = prepare(_cfg(tmp_path, **{"prepare.max_rows": 5}))
    assert manifest["splits"]["train"]["kept"] == 5


def test_question_group_matches_on_normalised_text():
    assert question_group("Does IT fit?") == question_group("does it fit")
    assert question_group("does it fit") != question_group("does it ship")


# -- dataset and collator --------------------------------------------------- #


def test_dataset_indexes_every_record(tmp_path):
    prepare(_cfg(tmp_path))
    dataset = PairDataset(tmp_path / "processed" / "train.jsonl")
    assert len(dataset) == 36
    assert {dataset[i]["qid"] for i in range(len(dataset))} == {
        f"t{p}-{q}" for p in range(12) for q in range(3)
    }, "byte-offset indexing returned the wrong records"


def test_dataset_subset_truncates(tmp_path):
    prepare(_cfg(tmp_path))
    assert len(PairDataset(tmp_path / "processed" / "train.jsonl", subset=4)) == 4


def test_collator_pads_and_masks(tmp_path):
    cfg = _cfg(tmp_path)
    prepare(cfg)
    out = tmp_path / "processed"
    tokenizer = load_tokenizer(out / "tokenizer.json")
    dataset = PairDataset(out / "train.jsonl")
    collate = PairCollator(tokenizer, cfg.data.max_query_len, cfg.data.max_doc_len)

    batch = collate([dataset[i] for i in range(4)])
    assert batch["query_ids"].shape == batch["query_mask"].shape
    assert batch["doc_ids"].shape[0] == 4
    assert batch["is_answerable"].tolist() == [1.0, 1.0, 1.0, 1.0]
    # Padding must be masked out and carry the pad id, or mean pooling averages noise.
    padded = ~batch["doc_mask"]
    assert (batch["doc_ids"][padded] == pad_id(tokenizer)).all()
    assert batch["doc_mask"].any(dim=1).all(), "a row has no real tokens"


def test_collator_truncates_to_the_configured_length(tmp_path):
    cfg = _cfg(tmp_path, **{"data.max_doc_len": 6})
    prepare(cfg)
    out = tmp_path / "processed"
    dataset = PairDataset(out / "train.jsonl")
    collate = PairCollator(load_tokenizer(out / "tokenizer.json"), cfg.data.max_query_len, 6)
    assert collate([dataset[i] for i in range(4)])["doc_ids"].shape[1] == 6
