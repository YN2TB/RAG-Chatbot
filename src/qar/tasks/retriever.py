"""The real task: contrastive retrieval over AmazonQA.

InfoNCE over in-batch negatives. Each question's positive snippet is the answer;
every *other* row's positive in the same batch is a negative. That is the whole
objective, and it has one consequence worth internalising before reading any
ablation: **the batch size is part of the loss.** Batch 32 asks the model to pick
one of 32; batch 256 asks it to pick one of 256, a materially harder problem that
produces a sharper representation. On 8 GB the achievable batch is a research
constraint, not a footnote.

Validation reports ranking metrics over the same in-batch matrix, so `val/recall@1`
here is recall within `data.eval_batch_size` candidates drawn from *different*
products. It is a training signal, not the headline number -- the honest comparison
against `runs/_baselines/` is within-product ranking over the real pool, which
`scripts/evaluate_retrieval.py` measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from qar import models  # noqa: F401  (registers every model this task can name)
from qar.config import RunConfig
from qar.data.dataset import PairCollator, PairDataset, question_groups
from qar.data.sampler import QGroupBatchSampler
from qar.data.tokenizer import load_tokenizer, pad_id
from qar.eval.metrics import binary_metrics, ranking_metrics
from qar.registry import build, register
from qar.training.task import Task
from qar.utils.logging import get_logger
from qar.utils.seed import worker_init_fn

log = get_logger(__name__)


@register("task", "retriever")
class RetrieverTask(Task):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)
        self.processed = Path(cfg.data.processed_dir)
        self.tokenizer = load_tokenizer(self.processed / "tokenizer.json")
        self.pad_id = pad_id(self.tokenizer)
        self.collate = PairCollator(
            self.tokenizer, cfg.data.max_query_len, cfg.data.max_doc_len,
            hard_negatives=cfg.loss.hard_negatives, seed=cfg.seed,
        )
        # The embedding table must cover every id the tokenizer can emit. Fewer tokens
        # than requested is legitimate -- BPE stops early on a small corpus -- but a
        # larger tokenizer would index past the embedding and crash mid-run.
        built = self.tokenizer.get_vocab_size()
        if built > cfg.model.vocab_size:
            raise ValueError(
                f"tokenizer has {built} tokens but model.vocab_size={cfg.model.vocab_size}; "
                "the embedding table cannot represent the prepared corpus"
            )
        if built < cfg.model.vocab_size:
            log.warning(
                "tokenizer built %d of %d requested tokens; %d embedding rows will never "
                "be used", built, cfg.model.vocab_size, cfg.model.vocab_size - built
            )

    # -- components the trainer drives ------------------------------------- #

    def build_model(self) -> nn.Module:
        return build("model", self.cfg.model.name, self.cfg, self.pad_id)

    def train_loader(self) -> DataLoader:
        return self._loader("train", self.cfg.data.batch_size, shuffle=True,
                            subset=self.cfg.data.train_subset)

    def val_loader(self) -> DataLoader:
        return self._loader("val", self.cfg.data.eval_batch_size, shuffle=False,
                            subset=self.cfg.data.val_subset)

    def _loader(self, split: str, batch_size: int, shuffle: bool,
                subset: int | None = None) -> DataLoader:
        path = self.processed / f"{split}.jsonl"
        dataset = PairDataset(path, subset=subset)
        common = {
            "collate_fn": self.collate,
            "num_workers": self.cfg.data.num_workers,
            "worker_init_fn": worker_init_fn if self.cfg.data.num_workers else None,
        }

        # Only training needs the guard: validation computes no contrastive gradient,
        # so a duplicated question there costs nothing but a slightly harder metric.
        if shuffle and self.cfg.data.dedup_questions_in_batch:
            groups = question_groups(path)[: len(dataset)]
            sampler = QGroupBatchSampler(groups, batch_size, seed=self.cfg.seed)
            log.info("dedup sampler on: %d rows, %d distinct questions",
                     len(groups), len(set(groups.tolist())))
            return DataLoader(dataset, batch_sampler=sampler, **common)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # InfoNCE labels are arange(batch); a short final batch would silently
            # change how many negatives the last step of every epoch sees.
            drop_last=True,
            **common,
        )

    # -- the two steps ------------------------------------------------------ #

    def _scores(self, model: nn.Module, batch) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Similarity matrix, its target column, and the answerability logits.

        Without hard negatives the matrix is the familiar `[B, B]` in-batch one.
        With them it grows to `[B, B + n]`, where the trailing `n` columns are that
        row's **own** product's snippets -- scored per row rather than shared, so one
        row's hard negative can never become another row's false negative.
        """
        query, doc, answerable = model(batch)
        scores = query @ doc.T

        if "neg_ids" in batch:
            rows, n, length = batch["neg_ids"].shape
            flat = model.encode_doc(
                batch["neg_ids"].view(rows * n, length),
                batch["neg_mask"].view(rows * n, length),
            ).view(rows, n, -1)
            hard = torch.einsum("bd,bnd->bn", query, flat)
            # Slots no pool could fill drop out of the softmax entirely.
            hard = hard.masked_fill(~batch["neg_valid"], float("-inf"))
            scores = torch.cat([scores, hard], dim=1)

        scores = scores / self.cfg.loss.temperature
        target = torch.arange(query.size(0), device=scores.device)
        return scores, target, answerable

    def training_step(self, model: nn.Module, batch) -> tuple[torch.Tensor, dict[str, float]]:
        scores, target, answerable = self._scores(model, batch)

        contrastive = F.cross_entropy(scores, target)
        metrics = {"nce": float(contrastive.detach()),
                   "acc": float((scores.argmax(1) == target).float().mean())}

        loss = contrastive
        weight = self.cfg.loss.answerable_weight
        if answerable is not None and weight > 0:
            auxiliary = F.binary_cross_entropy_with_logits(answerable, batch["is_answerable"])
            loss = contrastive + weight * auxiliary
            metrics["ans_bce"] = float(auxiliary.detach())
        return loss, metrics

    @torch.no_grad()
    def validation_step(self, model: nn.Module, batch) -> dict[str, float]:
        scores, target, answerable = self._scores(model, batch)
        metrics = {
            "loss": float(F.cross_entropy(scores, target)),
            **ranking_metrics(scores, target),
        }
        if answerable is not None:
            labels = batch["is_answerable"]
            metrics["ans_bce"] = float(F.binary_cross_entropy_with_logits(answerable, labels))
            metrics.update({f"ans_{k}": v for k, v in binary_metrics(answerable, labels).items()})
        return metrics
