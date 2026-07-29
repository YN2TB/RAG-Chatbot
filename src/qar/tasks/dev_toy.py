"""A synthetic contrastive task, shaped exactly like the real retriever.

Its only job is to prove the harness works -- loop, precision, accumulation,
scheduling, checkpointing, resume, metrics -- before the real data pipeline
exists. Two towers, InfoNCE over in-batch negatives, ranking metrics: swap the
synthetic pairs for (question, snippet) pairs and the trainer does not change.

The task is learnable by construction, so `tests/test_smoke.py` can assert that
the loss actually falls -- the standard "can it overfit?" sanity check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from qar.config import RunConfig
from qar.eval.metrics import ranking_metrics
from qar.registry import register
from qar.training.task import Task


class SyntheticPairs(Dataset):
    """Query/document pairs that share a latent concept, observed through noise."""

    def __init__(self, n_items: int, n_concepts: int, dim: int, noise: float, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.concepts = torch.randn(n_concepts, dim, generator=g)
        self.ids = torch.randint(0, n_concepts, (n_items,), generator=g)
        self.noise = noise
        self.dim = dim
        self._g = g

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        latent = self.concepts[self.ids[idx]]
        return {
            "query": latent + self.noise * torch.randn(self.dim),
            "doc": latent + self.noise * torch.randn(self.dim),
        }


class TwoTower(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.query_tower = self._tower(dim, hidden, dropout)
        self.doc_tower = self._tower(dim, hidden, dropout)

    @staticmethod
    def _tower(dim: int, hidden: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, query: torch.Tensor, doc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # L2-normalised, so the dot product below is a cosine similarity.
        return (
            F.normalize(self.query_tower(query), dim=-1),
            F.normalize(self.doc_tower(doc), dim=-1),
        )


@register("task", "dev_toy")
class DevToyTask(Task):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)
        self.dim = 64
        self.n_concepts = 512

    def build_model(self) -> nn.Module:
        return TwoTower(self.dim, self.cfg.model.d_model, self.cfg.model.dropout)

    def _loader(self, n: int, seed: int, batch_size: int, shuffle: bool) -> DataLoader:
        ds = SyntheticPairs(n, self.n_concepts, self.dim, noise=0.4, seed=seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,  # InfoNCE labels assume a full, square score matrix
            num_workers=self.cfg.data.num_workers,
        )

    def train_loader(self) -> DataLoader:
        return self._loader(8192, self.cfg.seed, self.cfg.data.batch_size, shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._loader(1024, self.cfg.seed + 1, self.cfg.data.eval_batch_size, shuffle=False)

    def training_step(self, model: nn.Module, batch) -> tuple[torch.Tensor, dict[str, float]]:
        q, d = model(batch["query"], batch["doc"])
        scores = q @ d.T / self.cfg.loss.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        loss = F.cross_entropy(scores, target)
        return loss, {"acc": float((scores.argmax(1) == target).float().mean())}

    @torch.no_grad()
    def validation_step(self, model: nn.Module, batch) -> dict[str, float]:
        q, d = model(batch["query"], batch["doc"])
        scores = q @ d.T / self.cfg.loss.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        return {
            "loss": float(F.cross_entropy(scores, target)),
            **ranking_metrics(scores, target),
        }
