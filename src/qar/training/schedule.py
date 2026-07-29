"""Learning-rate schedules with linear warmup.

Warmup is not optional for a from-scratch transformer: without it the first few
hundred steps of Adam on a randomly initialised attention stack routinely diverge.
The warmup ratio is a config field precisely so you can show that in an ablation.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    total_steps: int,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    warmup = max(0, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if name == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            factor = 1.0 - progress
        elif name == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"scheduler must be cosine|linear|constant, got {name!r}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * factor

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, cfg) -> Optimizer:
    """AdamW with weight decay applied only to matrices.

    Decaying biases, LayerNorm gains and embeddings measurably hurts transformer
    training; this split is the standard fix and worth a line in the report.
    """
    import torch

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.name.lower() != "adamw":
        raise ValueError(f"unsupported optimizer {cfg.name!r} (only adamw is wired up)")
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
