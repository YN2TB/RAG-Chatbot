"""Checkpoint save/load with rotation and exact resume.

A checkpoint carries optimizer, scheduler and scaler state as well as weights, so
a run killed at step 6000 resumes at step 6000 rather than restarting -- which
matters when a sweep runs overnight on a laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qar.utils.logging import get_logger

log = get_logger(__name__)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    step: int = 0,
    best: float | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best": best,
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leave a half-written checkpoint behind


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    log.info("resumed from %s at step %d", path.name, payload.get("step", 0))
    return payload


def rotate(ckpt_dir: Path, keep: int, pattern: str = "step_*.pt") -> None:
    """Delete all but the `keep` newest step checkpoints. `best.pt` is never touched."""
    if keep <= 0:
        return
    files = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    for stale in files[:-keep]:
        stale.unlink(missing_ok=True)


def find_latest(ckpt_dir: Path, pattern: str = "step_*.pt") -> Path | None:
    files = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None
