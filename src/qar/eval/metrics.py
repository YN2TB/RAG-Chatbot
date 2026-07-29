"""Metrics for both halves of the project.

Ranking metrics serve the DL report (retriever quality); the classification
metrics serve the answerability head. Generation metrics land here later for the
NLP report.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ranking_metrics(scores: torch.Tensor, target: torch.Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """Recall@k and MRR from a (queries x candidates) score matrix.

    `target[i]` is the column index of the correct candidate for query i -- which
    for in-batch contrastive training is just `arange(batch)`.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {tuple(scores.shape)}")
    n, c = scores.shape
    order = scores.argsort(dim=1, descending=True)
    rank_of_target = (order == target.unsqueeze(1)).float().argmax(dim=1)  # 0-indexed

    out = {"mrr": float((1.0 / (rank_of_target + 1)).mean())}
    for k in ks:
        if k <= c:
            out[f"recall@{k}"] = float((rank_of_target < k).float().mean())
    out["mean_rank"] = float((rank_of_target + 1).float().mean())
    return out


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.0) -> dict[str, float]:
    """Accuracy / precision / recall / F1 for the answerability head.

    `logits` are raw (pre-sigmoid); the 62/38 class split makes F1 the metric to
    report and accuracy the one to distrust.
    """
    pred = (logits.squeeze(-1) > threshold).long()
    labels = labels.long()

    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "acc": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
