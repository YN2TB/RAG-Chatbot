"""The retriever task end to end, on a synthetic corpus it can actually learn.

The overfit check is the point. `dev_toy` proves the *harness* works; this proves
the real path works -- prepared corpus, BPE, collator, two towers, InfoNCE, ranking
metrics, checkpointing -- before a single GPU hour is spent on the real thing.

The synthetic corpus is learnable by construction: each product has a distinctive
topic word that appears in the question and in exactly one of its snippets. A model
that cannot drive this loss down has a bug, not a research problem.
"""

from __future__ import annotations

import json

import pytest
import torch

from qar import tasks  # noqa: F401  (registers the retriever task)
from qar.config import load_config
from qar.data.prepare import prepare
from qar.data.sampler import QGroupBatchSampler
from qar.registry import build
from qar.training.trainer import Trainer
from qar.utils.seed import seed_everything

TOPICS = [
    "battery", "strap", "zoom", "waterproof", "bluetooth", "tripod",
    "shutter", "lens", "charger", "viewfinder", "memory", "flash",
]


def _raw(qid, asin, topic, index):
    """One row whose answer, question and a single snippet share a topic word.

    Questions are mostly distinct, as in the real corpus (704k rows, 638k questions).
    Every fourth row deliberately repeats a question so the de-duplicating batch
    sampler has something to de-duplicate.
    """
    phrasing = "on this particular unit" if index % 4 == 0 else f"on unit number {index}"
    return {
        "qid": qid,
        "asin": asin,
        "category": "Electronics",
        "questionText": f"how good is the {topic} {phrasing}",
        "questionType": "descriptive",
        "review_snippets": [
            f"the {topic} on this unit performs very well in daily use",
            f"packaging arrived intact and delivery was prompt number {index}",
            f"colour is darker than the listing photograph suggests here {index}",
        ],
        "answers": [{"answerText": f"the {topic} works very well",
                     "answerType": "NA", "helpful": [1, 1]}],
        "is_answerable": index % 2,
        "random_sentence": ["ignored"],
        "top_sentences_IR": ["ignored"],
        "top_review_wilson": ["ignored"],
        "top_review_helpful": ["ignored"],
    }


def _corpus(tmp_path, per_topic=24):
    train, val = [], []
    for topic_index, topic in enumerate(TOPICS):
        for i in range(per_topic):
            train.append(_raw(f"t{topic_index}-{i}", f"TRAIN{topic_index:02d}{i:03d}", topic, i))
        for i in range(6):
            val.append(_raw(f"v{topic_index}-{i}", f"VAL{topic_index:02d}{i:03d}", topic, i))

    for name, rows in (("raw_train", train), ("raw_val", val)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
    return tmp_path / "raw_train.jsonl", tmp_path / "raw_val.jsonl"


def _cfg(tmp_path, **overrides):
    raw_train, raw_val = _corpus(tmp_path)
    base = [
        f"data.train_path={raw_train.as_posix()}",
        f"data.val_path={raw_val.as_posix()}",
        # Without this, prepare falls through to the repo default and reads the real
        # 751 MB test-qar_all.jsonl from the repo root -- a test suite that touches
        # the corpus, and minutes per test.
        "data.test_path=null",
        f"data.processed_dir={(tmp_path / 'processed').as_posix()}",
        f"out_dir={tmp_path.as_posix()}",
        "device=cpu", "train.amp=off", "data.num_workers=0",
        "model.vocab_size=1200", "prepare.tokenizer_sample_docs=2000",
        "model.d_model=64", "model.n_layers=2", "model.n_heads=4", "model.d_ff=128",
        "model.max_len=64", "model.dropout=0.0",
        "data.max_query_len=32", "data.max_doc_len=48",
        "data.batch_size=16", "data.eval_batch_size=16",
    ]
    return load_config("configs/retriever.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


def _prepared(tmp_path, **overrides):
    cfg = _cfg(tmp_path, **overrides)
    prepare(cfg)
    return cfg


def test_task_is_registered(tmp_path):
    cfg = _prepared(tmp_path)
    assert build("task", "retriever", cfg) is not None


def test_batch_reaches_the_model_with_the_expected_keys(tmp_path):
    cfg = _prepared(tmp_path)
    task = build("task", cfg.task, cfg)
    batch = next(iter(task.train_loader()))

    assert set(batch) == {"query_ids", "query_mask", "doc_ids", "doc_mask", "is_answerable"}
    assert batch["query_ids"].shape[0] == cfg.data.batch_size
    assert batch["query_ids"].shape[1] <= cfg.data.max_query_len
    assert batch["doc_ids"].shape[1] <= cfg.data.max_doc_len


def test_training_step_returns_loss_and_plain_floats(tmp_path):
    cfg = _prepared(tmp_path)
    task = build("task", cfg.task, cfg)
    model = task.build_model()

    loss, metrics = task.training_step(model, next(iter(task.train_loader())))
    assert loss.requires_grad and loss.ndim == 0
    assert all(isinstance(v, float) for v in metrics.values())
    assert {"nce", "acc"} <= set(metrics)


def test_model_can_overfit_the_synthetic_corpus(tmp_path):
    """Loss must fall and in-batch retrieval must beat chance (1/16 = 0.0625)."""
    cfg = _prepared(tmp_path, name="overfit",
                    **{"train.max_steps": 120, "train.eval_every": 120,
                       "train.log_every": 20, "optim.lr": 1e-3})
    seed_everything(cfg.seed)
    results = Trainer(cfg, build("task", cfg.task, cfg)).train()

    records = [
        json.loads(line)
        for line in (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    losses = [r["train/loss"] for r in records if "train/loss" in r]
    assert len(losses) >= 3
    assert losses[-1] < losses[0] * 0.8, f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert results["val/recall@1"] > 0.25, "in-batch retrieval no better than chance"


def test_answerability_head_adds_its_own_metrics(tmp_path):
    cfg = _prepared(tmp_path, name="multitask",
                    **{"loss.answerable_weight": 0.5, "train.max_steps": 20,
                       "train.eval_every": 20, "train.log_every": 20})
    seed_everything(cfg.seed)
    results = Trainer(cfg, build("task", cfg.task, cfg)).train()

    assert "val/ans_bce" in results
    assert "val/ans_f1" in results, "binary metrics did not reach the log"


def test_zero_weight_leaves_the_auxiliary_loss_out(tmp_path):
    cfg = _prepared(tmp_path, **{"loss.answerable_weight": 0.0})
    task = build("task", cfg.task, cfg)
    _, metrics = task.training_step(task.build_model(), next(iter(task.train_loader())))
    assert "ans_bce" not in metrics


def test_oversized_tokenizer_is_rejected(tmp_path):
    """An embedding table smaller than the vocabulary would crash mid-run."""
    cfg = _prepared(tmp_path)
    cfg.model.vocab_size = 4
    with pytest.raises(ValueError, match="cannot represent"):
        build("task", "retriever", cfg)


def test_val_subset_bounds_the_evaluation_pass(tmp_path):
    """A full val pass costs more than the training between passes.

    Regression: without a bound, a smoke run with eval_every=60 spent minutes of
    every evaluation walking all 43k validation rows.
    """
    cfg = _prepared(tmp_path, **{"data.val_subset": 32, "data.eval_batch_size": 16})
    task = build("task", cfg.task, cfg)
    assert len(task.val_loader().dataset) == 32

    unbounded = _cfg(tmp_path, **{"data.val_subset": "null"})
    assert len(build("task", cfg.task, unbounded).val_loader().dataset) > 32


def test_train_loader_drops_the_short_final_batch(tmp_path):
    """InfoNCE labels are arange(batch); a ragged last batch would quietly change
    how many negatives that step sees."""
    cfg = _prepared(tmp_path)
    task = build("task", cfg.task, cfg)
    sizes = {batch["query_ids"].shape[0] for batch in task.train_loader()}
    assert sizes == {cfg.data.batch_size}


def test_dedup_sampler_is_used_for_training_only(tmp_path):
    cfg = _prepared(tmp_path, **{"data.dedup_questions_in_batch": "true"})
    task = build("task", cfg.task, cfg)
    assert isinstance(task.train_loader().batch_sampler, QGroupBatchSampler)
    assert not isinstance(task.val_loader().batch_sampler, QGroupBatchSampler)


def test_dedup_can_be_switched_off(tmp_path):
    cfg = _prepared(tmp_path, **{"data.dedup_questions_in_batch": "false"})
    task = build("task", cfg.task, cfg)
    assert not isinstance(task.train_loader().batch_sampler, QGroupBatchSampler)


def test_batch_larger_than_the_distinct_questions_fails_loudly(tmp_path):
    """Regression: this used to yield zero batches, which the trainer's endless
    cycle turned into a silent infinite loop rather than an error."""
    cfg = _prepared(tmp_path, **{"data.batch_size": 4096})
    task = build("task", cfg.task, cfg)
    with pytest.raises(ValueError, match="distinct question groups"):
        task.train_loader()


def test_hard_negatives_widen_the_score_matrix(tmp_path):
    """`[B, B]` in-batch becomes `[B, B + n]`, the extra columns being that row's own
    product's snippets."""
    cfg = _prepared(tmp_path, **{"loss.hard_negatives": 2})
    task = build("task", cfg.task, cfg)
    batch = next(iter(task.train_loader()))

    assert batch["neg_ids"].shape[:2] == (cfg.data.batch_size, 2)
    assert batch["neg_valid"].shape == (cfg.data.batch_size, 2)

    scores, target, _ = task._scores(task.build_model(), batch)
    assert scores.shape == (cfg.data.batch_size, cfg.data.batch_size + 2)
    assert target.tolist() == list(range(cfg.data.batch_size))


def test_hard_negatives_come_from_the_same_product_and_are_not_the_positive(tmp_path):
    cfg = _prepared(tmp_path, **{"loss.hard_negatives": 1})
    task = build("task", cfg.task, cfg)
    dataset = task.train_loader().dataset

    for index in range(20):
        record = dataset[index]
        pool = record["snippets"]
        batch = task.collate([record])
        text = task.tokenizer.decode(
            [i for i in batch["neg_ids"][0, 0].tolist() if i != task.pad_id]
        ).strip()
        others = [s for j, s in enumerate(pool) if j != record["positive_idx"]]
        assert any(text[:40].lower() in s.lower() for s in others), \
            "hard negative was not one of this product's other snippets"


def test_unfillable_negative_slots_are_masked_out(tmp_path):
    """A single-snippet pool has no same-product negative; the slot must leave the
    softmax rather than pretend to be a candidate."""
    cfg = _prepared(tmp_path, **{"loss.hard_negatives": 1})
    task = build("task", cfg.task, cfg)
    record = {"question": "q", "snippets": ["only one snippet here"],
              "positive_idx": 0, "is_answerable": 1}

    batch = task.collate([record])
    assert batch["neg_valid"].tolist() == [[False]]

    scores, _, _ = task._scores(task.build_model(), batch)
    assert torch.isinf(scores[0, -1]) and scores[0, -1] < 0


def test_zero_hard_negatives_leaves_the_batch_unchanged(tmp_path):
    cfg = _prepared(tmp_path, **{"loss.hard_negatives": 0})
    task = build("task", cfg.task, cfg)
    batch = next(iter(task.train_loader()))
    assert "neg_ids" not in batch

    scores, _, _ = task._scores(task.build_model(), batch)
    assert scores.shape == (cfg.data.batch_size, cfg.data.batch_size)


def test_trainer_refuses_an_empty_loader(tmp_path):
    """The other half of the same regression, guarded at the trainer."""
    cfg = _prepared(tmp_path, **{"train.max_steps": 2})
    trainer = Trainer(cfg, build("task", cfg.task, cfg))
    with pytest.raises(RuntimeError, match="no batches"):
        next(trainer._endless([]))
