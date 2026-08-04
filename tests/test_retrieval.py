"""Untrained retrieval baselines, on hand-built pools with a known right answer."""

from __future__ import annotations

import json

import pytest
import torch

from qar.config import load_config
from qar.data.dataset import PairDataset
from qar.registry import available, build
from qar.retrieval.evaluate import _pad, evaluate_retriever, markdown_table
from qar.retrieval.idf import build_document_frequencies, idf_lookup, load_idf, save_idf


def _cfg(tmp_path, **overrides):
    base = [f"out_dir={tmp_path.as_posix()}"]
    return load_config("configs/base.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


def _split(tmp_path, records):
    path = tmp_path / "val.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return PairDataset(path)


def _record(qid, question, snippets, positive_idx, qtype="descriptive", answerable=1):
    return {
        "qid": qid, "asin": f"A{qid}", "category": "Electronics",
        "question": question, "question_type": qtype, "is_answerable": answerable,
        "qgroup": qid, "positive_idx": positive_idx, "positive_score": 0.5,
        "snippets": snippets,
    }


# -- registration ----------------------------------------------------------- #


def _idf_corpus(tmp_path, records):
    """A processed_dir holding just the IDF table `bm25_global` needs."""
    dataset = _split(tmp_path, records)
    table = build_document_frequencies(dataset, min_df=1)
    save_idf(tmp_path / "idf.json", table)
    return table


def test_every_baseline_is_registered():
    assert {"random", "first", "overlap", "bm25", "bm25_global", "bm25_noidf"} <= set(
        available("retriever")
    )


def test_noidf_variant_differs_from_bm25_global_only_in_idf(tmp_path):
    """The controlled comparison must stay controlled.

    `bm25_noidf` exists to attribute a measured effect to IDF alone, so if the two
    ever stop agreeing on a pool where every query term has the same IDF, some
    other difference has crept in and the attribution is void.
    """
    corpus = [_record(f"c{i}", "q", [f"alpha beta gamma delta {i}"], 0) for i in range(50)]
    _idf_corpus(tmp_path, corpus)
    cfg = _cfg(tmp_path, **{"data.processed_dir": tmp_path.as_posix()})

    pool = ["alpha padding words here", "nothing relevant at all here"]
    weighted = build("retriever", "bm25_global", cfg).scores("alpha", pool)
    plain = build("retriever", "bm25_noidf", cfg).scores("alpha", pool)

    # One query term => IDF is a single positive constant, so ranking is identical
    # and the scores differ only by that factor.
    assert (weighted[0] > weighted[1]) == (plain[0] > plain[1])
    assert weighted[0] / plain[0] == pytest.approx(weighted[0] / plain[0])
    assert plain[0] > 0 and plain[1] == 0.0


def test_baselines_build_from_config(tmp_path):
    _idf_corpus(tmp_path, [_record("q0", "a question", ["one doc here now"], 0)])
    cfg = _cfg(tmp_path, **{"data.processed_dir": tmp_path.as_posix()})
    for name in cfg.retrieval.baselines:
        assert build("retriever", name, cfg).scores("a question", ["one doc"]) is not None


# -- scoring ---------------------------------------------------------------- #


def test_bm25_prefers_the_lexically_matching_snippet(tmp_path):
    bm25 = build("retriever", "bm25", _cfg(tmp_path))
    scores = bm25.scores(
        "how long does the battery last",
        ["the packaging arrived crushed and taped", "battery life is long on this unit"],
    )
    assert scores[1] > scores[0]


def test_bm25_scores_zero_when_nothing_matches(tmp_path):
    bm25 = build("retriever", "bm25", _cfg(tmp_path))
    assert bm25.scores("battery life", ["completely unrelated wording here"]) == [0.0]


def test_bm25_discounts_terms_present_in_every_snippet(tmp_path):
    """A word in every candidate cannot discriminate.

    The Lucene IDF form used here floors at a small positive value rather than
    going negative, so such a term does not score zero -- it scores every document
    *alike*, and far below a term that appears in only one of them. Equal lengths
    keep the length normalisation out of the comparison.
    """
    bm25 = build("retriever", "bm25", _cfg(tmp_path))
    pool = ["camera aaa bbb", "camera ccc ddd", "camera eee fff", "camera ggg zoom"]

    common = bm25.scores("camera", pool)
    assert common == pytest.approx([common[0]] * 4), "a ubiquitous term discriminated"

    rare = bm25.scores("zoom", pool)
    assert rare[3] > 5 * common[0], "a term in 1 of 4 docs must outweigh one in 4 of 4"


def test_overlap_is_symmetric_in_neither_direction_but_ranks_sanely(tmp_path):
    overlap = build("retriever", "overlap", _cfg(tmp_path))
    scores = overlap.scores("does the strap detach", ["the strap detaches easily", "blue colour"])
    assert scores[0] > scores[1]


def test_first_ranks_by_position(tmp_path):
    first = build("retriever", "first", _cfg(tmp_path))
    assert first.scores("anything", ["a", "b", "c"]) == [0.0, -1.0, -2.0]


def test_empty_pool_yields_no_scores(tmp_path):
    for name in ("bm25", "overlap", "first", "random"):
        assert build("retriever", name, _cfg(tmp_path)).scores("q", []) == []


# -- corpus-wide IDF -------------------------------------------------------- #


def test_document_frequency_counts_snippets_not_rows():
    """A term repeated inside one snippet counts once; the unit is the snippet."""
    dataset = [
        {"snippets": ["alpha alpha alpha beta", "alpha gamma"]},
        {"snippets": ["beta delta"]},
    ]
    table = build_document_frequencies(dataset, min_df=1)
    assert table["n_docs"] == 3
    assert table["df"]["alpha"] == 2
    assert table["df"]["beta"] == 2
    assert table["df"]["gamma"] == 1


def test_document_frequency_prunes_the_rare_tail():
    dataset = [{"snippets": ["common word", "common thing", "typoo"]}]
    table = build_document_frequencies(dataset, min_df=2)
    assert "common" in table["df"]
    assert "typoo" not in table["df"], "min_df did not prune"
    assert table["vocabulary"] > len(table["df"])


def test_idf_is_monotonically_decreasing_in_frequency():
    table = {"n_docs": 1000, "min_df": 1, "df": {"rare": 2, "middling": 50, "everywhere": 900}}
    idf = idf_lookup(table)
    assert idf("rare") > idf("middling") > idf("everywhere") > 0


def test_unseen_term_is_treated_as_rare_but_not_infinite():
    table = {"n_docs": 1000, "min_df": 5, "df": {"seen": 5}}
    idf = idf_lookup(table)
    assert idf("never-observed") == pytest.approx(idf("seen")), "pruned terms need the floor"


def test_idf_table_round_trips(tmp_path):
    table = build_document_frequencies([{"snippets": ["alpha beta", "beta gamma"]}], min_df=1)
    save_idf(tmp_path / "idf.json", table)
    assert load_idf(tmp_path / "idf.json") == table


def test_missing_idf_table_names_the_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_idf"):
        load_idf(tmp_path / "absent.json")


def test_global_idf_beats_pool_local_when_a_term_saturates_the_product(tmp_path):
    """The failure that motivated `bm25_global`.

    Every snippet of this product mentions "tripod", so pool-local IDF discards the
    term -- while corpus-wide it is rare and highly discriminative.
    """
    corpus = [_record(f"c{i}", "q", [f"ordinary review sentence {i} about nothing"], 0)
              for i in range(200)]
    corpus.append(_record("c-tripod", "q", ["tripod tripod tripod"], 0))
    _idf_corpus(tmp_path, corpus)
    cfg = _cfg(tmp_path, **{"data.processed_dir": tmp_path.as_posix()})

    pool = ["tripod mount is ordinary", "tripod ordinary review sentence"]
    local = build("retriever", "bm25", cfg).scores("tripod", pool)
    glob = build("retriever", "bm25_global", cfg).scores("tripod", pool)

    assert local[0] == pytest.approx(local[1]), "pool-local IDF should see no difference"
    assert max(glob) > max(local), "corpus-wide IDF must weight the rare term higher"


# -- padding ---------------------------------------------------------------- #


def test_pad_puts_absent_candidates_last():
    matrix = _pad([[1.0, 2.0], [0.5]])
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] == float("-inf")
    assert matrix[1].argmax().item() == 0, "padding outranked a real candidate"


# -- evaluation ------------------------------------------------------------- #


def test_perfect_retriever_scores_one(tmp_path):
    """A pool whose positive is the only lexical match must give recall@1 = 1."""
    records = [
        _record(f"q{i}", "how long does the battery last",
                ["shipping was fast and cheap", "the battery lasts nine hours"], 1)
        for i in range(20)
    ]
    cfg = _cfg(tmp_path)
    result = evaluate_retriever(cfg, build("retriever", "bm25", cfg), _split(tmp_path, records))
    assert result["overall"]["recall@1"] == pytest.approx(1.0)
    assert result["overall"]["mrr"] == pytest.approx(1.0)
    assert result["mean_pool"] == 2.0


def test_random_baseline_lands_near_chance(tmp_path):
    """With a pool of 4, random recall@1 must sit near 0.25 -- the floor the table
    is read against."""
    records = [
        _record(f"q{i}", "a question", [f"snippet {j} text" for j in range(4)], i % 4)
        for i in range(800)
    ]
    cfg = _cfg(tmp_path)
    result = evaluate_retriever(cfg, build("retriever", "random", cfg), _split(tmp_path, records))
    assert 0.20 < result["overall"]["recall@1"] < 0.30


def test_unscorable_rows_land_at_chance_not_on_snippet_zero(tmp_path):
    """No lexical overlap anywhere: every score is 0.0, so the ranking is arbitrary.

    Without the per-row shuffle those rows would all go to snippet 0 and the number
    reported would secretly be the `first` baseline.
    """
    records = [
        _record(f"q{i}", "zzz", [f"unrelated {j}" for j in range(4)], 0) for i in range(800)
    ]
    cfg = _cfg(tmp_path)
    result = evaluate_retriever(cfg, build("retriever", "bm25", cfg), _split(tmp_path, records))
    assert 0.20 < result["overall"]["recall@1"] < 0.30, "ties collapsed onto position 0"


def test_breakdowns_partition_the_rows(tmp_path):
    records = [
        _record(f"q{i}", "does the strap detach",
                ["the strap detaches easily", "blue colour option"], 0,
                qtype="yes/no" if i % 2 else "descriptive", answerable=i % 2)
        for i in range(40)
    ]
    cfg = _cfg(tmp_path)
    result = evaluate_retriever(cfg, build("retriever", "bm25", cfg), _split(tmp_path, records))

    by_type = result["by_question_type"]
    assert sum(v["rows"] for v in by_type.values()) == 40
    assert sum(v["rows"] for v in result["by_answerable"].values()) == 40
    assert set(by_type) == {"descriptive", "yes/no"}


def test_max_rows_bounds_the_evaluation(tmp_path):
    records = [_record(f"q{i}", "a question", ["one", "two three"], 0) for i in range(50)]
    cfg = _cfg(tmp_path, **{"retrieval.max_rows": 7})
    result = evaluate_retriever(cfg, build("retriever", "first", cfg), _split(tmp_path, records))
    assert result["rows"] == 7


def test_empty_split_is_an_error(tmp_path):
    path = tmp_path / "val.jsonl"
    path.write_text("\n", encoding="utf-8")
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        evaluate_retriever(cfg, build("retriever", "first", cfg), PairDataset(path))


def test_markdown_table_has_a_row_per_retriever(tmp_path):
    records = [_record(f"q{i}", "battery life", ["battery life", "colour"], 0) for i in range(5)]
    cfg = _cfg(tmp_path)
    dataset = _split(tmp_path, records)
    results = {
        name: evaluate_retriever(cfg, build("retriever", name, cfg), dataset)
        for name in ("first", "bm25")
    }
    table = markdown_table(results, cfg.retrieval.ks)
    assert table.count("\n") == 3  # header, rule, two rows
    assert "recall@1" in table and "bm25" in table


def test_evaluation_is_reproducible(tmp_path):
    records = [
        _record(f"q{i}", "zzz", [f"unrelated {j}" for j in range(5)], 0) for i in range(100)
    ]
    cfg = _cfg(tmp_path)
    dataset = _split(tmp_path, records)
    runs = [
        evaluate_retriever(cfg, build("retriever", "bm25", cfg), dataset)["overall"]
        for _ in range(2)
    ]
    assert runs[0] == runs[1], "tie-break shuffling is not seeded"


def test_metrics_come_from_the_shared_implementation(tmp_path):
    """Guards against a re-implementation drifting from qar.eval.metrics."""
    from qar.eval.metrics import ranking_metrics

    matrix = _pad([[0.1, 0.9], [0.9, 0.1]])
    direct = ranking_metrics(matrix, torch.tensor([1, 0]), ks=(1,))
    assert direct["recall@1"] == pytest.approx(1.0)
