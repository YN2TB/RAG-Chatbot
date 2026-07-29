"""The contract between the trainer and whatever you are training.

The Trainer knows nothing about bi-encoders, InfoNCE or retrieval. A task owns the
model, the data and the loss; the trainer owns the loop, precision, checkpointing
and logging. Adding the real retriever later means writing one Task subclass --
no trainer changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader

from qar.config import RunConfig


class Task(ABC):
    """Base class for anything trainable in this project."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    # -- components the trainer drives ------------------------------------- #

    @abstractmethod
    def build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def train_loader(self) -> DataLoader: ...

    @abstractmethod
    def val_loader(self) -> DataLoader: ...

    # -- the two steps ------------------------------------------------------ #

    @abstractmethod
    def training_step(
        self, model: torch.nn.Module, batch: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (loss to backprop, scalars to log)."""

    @torch.no_grad()
    def validation_step(
        self, model: torch.nn.Module, batch: Any
    ) -> dict[str, float]:
        """Per-batch validation scalars; averaged by the trainer.

        Override `validate` instead when the metric is not a batch mean -- ranking
        metrics over a full index are the obvious case.
        """
        loss, metrics = self.training_step(model, batch)
        return {"loss": float(loss.detach()), **metrics}

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, loader: DataLoader, device: Any) -> dict[str, float]:
        """Full validation pass. Default: mean of `validation_step` over batches."""
        model.eval()
        totals: dict[str, float] = {}
        n = 0
        for batch in loader:
            batch = move_to(batch, device.device)
            with device.autocast():
                out = self.validation_step(model, batch)
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n += 1
        model.train()
        if n == 0:
            return {}
        return {f"val/{k}": v / n for k, v in totals.items()}


def move_to(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to `device`, leaving other types alone."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [move_to(v, device) for v in batch]
        return type(batch)(moved) if not isinstance(batch, tuple) else tuple(moved)
    return batch
