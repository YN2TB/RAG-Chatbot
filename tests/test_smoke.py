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
from qar.utils.logging import read_series
from qar.utils.seed import seed_everything


def _cfg(tmp_path, **overrides):
    base = [
        f"out_dir={tmp_path.as_posix()}",
        "device=cpu",
        "train.amp=off",
        "data.num_workers=0",
    ]
    return load_config("configs/dev.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


def _records(cfg, prefix="train/"):
    lines = (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if any(k.startswith(prefix) for k in r)]


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


def test_train_scalars_are_not_scaled_by_grad_accum(tmp_path):
    """`train/*` must be a mean over micro-batches, not a sum.

    Regression: the running totals were accumulated once per micro-batch but
    divided by the number of optimisation steps, so every `train/*` scalar read
    `grad_accum` times too high. Both runs here take one step from the same seed,
    so micro-batch 1 is identical and only the averaging can differ.
    """
    shared = {"data.batch_size": 32, "train.max_steps": 1, "train.log_every": 1}

    plain = _cfg(tmp_path, name="accum1", **{"train.grad_accum": 1, **shared})
    seed_everything(plain.seed)
    Trainer(plain, build("task", plain.task, plain)).train()

    accum = _cfg(tmp_path, name="accum2", **{"train.grad_accum": 2, **shared})
    seed_everything(accum.seed)
    Trainer(accum, build("task", accum.task, accum)).train()

    one, two = _records(plain)[0], _records(accum)[0]
    assert two["train/loss"] == pytest.approx(one["train/loss"], rel=0.2)
    assert 0.0 <= two["train/acc"] <= 1.0, "accuracy is a mean of an indicator"


def test_metrics_are_valid_jsonl(tmp_path):
    cfg = _cfg(tmp_path, name="jsonl", **{"train.max_steps": 60, "train.eval_every": 60})
    seed_everything(cfg.seed)
    Trainer(cfg, build("task", cfg.task, cfg)).train()

    for line in (cfg.run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert "step" in record and "wall_s" in record
    assert (cfg.run_dir / "config.yaml").exists(), "run config was not snapshotted"


def test_read_series_resolves_a_resume_rewind(tmp_path):
    """A resumed run rewinds; the abandoned tail must not win.

    metrics.jsonl is append-only, so resuming from step 200 leaves the interrupted
    run's records for 300 and 400 in the file. Later records are the surviving
    trajectory and must supersede them.
    """
    path = tmp_path / "metrics.jsonl"
    records = [
        {"step": 100, "val/recall@1": 0.10},
        {"step": 200, "val/recall@1": 0.20},
        {"step": 300, "val/recall@1": 0.95},   # abandoned trajectory
        {"step": 400, "val/recall@1": 0.99},   # abandoned trajectory
        {"step": 200, "event": "resume"},      # rewind marker, carries no metric
        {"step": 300, "val/recall@1": 0.30},   # surviving trajectory
        {"step": 400, "val/recall@1": 0.40},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    series = read_series(path, "val/recall@1")
    assert series == [(100, 0.10), (200, 0.20), (300, 0.30), (400, 0.40)]
    assert max(v for _, v in series) == 0.40, "a discarded trajectory won"


def test_read_series_ignores_other_metrics_and_bad_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        json.dumps({"step": 1, "train/loss": 2.0}) + "\n"
        + "{ not json\n"
        + json.dumps({"step": 2, "val/loss": 1.0}) + "\n",
        encoding="utf-8",
    )
    assert read_series(path, "train/loss") == [(1, 2.0)]
    assert read_series(path, "val/loss") == [(2, 1.0)]
    assert read_series(path, "absent") == []


def test_resume_writes_a_rewind_marker(tmp_path):
    """The log must say a rewind happened, or the duplicate steps are unexplained."""
    cfg = _cfg(tmp_path, name="marker", **{"train.max_steps": 100, "train.save_every": 50})
    seed_everything(cfg.seed)
    Trainer(cfg, build("task", cfg.task, cfg)).train()

    revived = Trainer(cfg, build("task", cfg.task, cfg))
    revived.maybe_resume()

    events = [
        r for r in map(json.loads, (cfg.run_dir / "metrics.jsonl").read_text(
            encoding="utf-8").splitlines())
        if r.get("event") == "resume"
    ]
    assert len(events) == 1, "resume did not record a rewind marker"
    assert events[0]["step"] == 100
    assert events[0]["from_checkpoint"].endswith(".pt")
