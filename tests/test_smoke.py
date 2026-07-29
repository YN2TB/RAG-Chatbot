"""End-to-end harness checks on the synthetic task.

The overfit test is the one that matters: if a model cannot drive the loss down on
a task that is learnable by construction, the bug is in the harness, not in your
research idea. Run this before every real experiment.
"""

from __future__ import annotations

import json

import pytest

from qar import tasks  # noqa: F401  (registers dev_toy)
from qar.config import load_config
from qar.registry import available, build
from qar.training.trainer import Trainer
from qar.utils.seed import seed_everything


def _cfg(tmp_path, **overrides):
    base = [
        f"out_dir={tmp_path.as_posix()}",
        "device=cpu",
        "train.amp=off",
        "data.num_workers=0",
    ]
    return load_config("configs/dev.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


def test_dev_toy_is_registered():
    assert "dev_toy" in available("task")


def test_model_can_overfit(tmp_path):
    """Loss must fall substantially and ranking must beat chance."""
    cfg = _cfg(
        tmp_path,
        name="overfit",
        **{"train.max_steps": 150, "train.eval_every": 150, "data.batch_size": 32},
    )
    seed_everything(cfg.seed)
    trainer = Trainer(cfg, build("task", cfg.task, cfg))
    results = trainer.train()

    records = [
        json.loads(line)
        for line in (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    losses = [r["train/loss"] for r in records if "train/loss" in r]
    assert len(losses) >= 2, "expected several logged training points"
    assert losses[-1] < losses[0] * 0.7, f"loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert results["val/recall@1"] > 0.5, "retrieval no better than chance"


def test_checkpoint_resume_restores_step(tmp_path):
    cfg = _cfg(tmp_path, name="resume", **{"train.max_steps": 100, "train.save_every": 50})
    seed_everything(cfg.seed)
    Trainer(cfg, build("task", cfg.task, cfg)).train()

    revived = Trainer(cfg, build("task", cfg.task, cfg))
    assert revived.step == 0
    revived.maybe_resume()
    assert revived.step == 100, "resume did not restore the optimisation step"


def test_grad_accum_matches_large_batch(tmp_path):
    """batch=32 x accum=2 should track batch=64 x accum=1 closely."""
    shared = {"train.max_steps": 60, "train.eval_every": 60, "train.log_every": 60}

    big = _cfg(tmp_path, name="big", **{"data.batch_size": 64, **shared})
    seed_everything(big.seed)
    a = Trainer(big, build("task", big.task, big)).train()

    split = _cfg(
        tmp_path, name="split", **{"data.batch_size": 32, "train.grad_accum": 2, **shared}
    )
    seed_everything(split.seed)
    b = Trainer(split, build("task", split.task, split)).train()

    assert a["val/loss"] == pytest.approx(b["val/loss"], rel=0.35)


def test_metrics_are_valid_jsonl(tmp_path):
    cfg = _cfg(tmp_path, name="jsonl", **{"train.max_steps": 60, "train.eval_every": 60})
    seed_everything(cfg.seed)
    Trainer(cfg, build("task", cfg.task, cfg)).train()

    for line in (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "step" in record and "wall_s" in record
    assert (cfg.run_dir / "config.yaml").exists(), "run config was not snapshotted"
