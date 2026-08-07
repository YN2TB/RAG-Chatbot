"""The trained bi-encoder, scored the same way the baselines are.

Everything else in this folder is untrained by construction. This one loads a
checkpoint, which is what finally makes a training run comparable with the table
it has to beat: the same rows, the same within-product pools, the same shuffle,
the same metric code.

That comparability is the entire point. `val/recall@1` printed during training
ranks a row against `eval_batch_size` candidates drawn from **other products** --
an easier problem with a different candidate count -- so it can never be quoted
against `overlap`. The number that can is the one this class produces.

    python scripts/evaluate_retrieval.py configs/retriever.yaml \\
        --set retrieval.baselines=[dense] retrieval.checkpoint=runs/<name>/checkpoints/best.pt
"""

from __future__ import annotations

from pathlib import Path

import torch

from qar import models as _models  # noqa: F401  (registers the architectures)
from qar.config import RunConfig, config_from_dict
from qar.data.tokenizer import load_tokenizer, pad_id
from qar.registry import build, register
from qar.retrieval.base import Retriever
from qar.utils.device import resolve_device
from qar.utils.logging import get_logger

log = get_logger(__name__)


@register("retriever", "dense")
class DenseRetriever(Retriever):
    """Cosine similarity between the trained query and document towers."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)
        path = cfg.retrieval.checkpoint
        if not path:
            raise ValueError(
                "retrieval.checkpoint is required for the 'dense' retriever, e.g. "
                "--set retrieval.checkpoint=runs/retriever/checkpoints/best.pt"
            )
        checkpoint = Path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint.resolve()}")

        processed = Path(cfg.data.processed_dir)
        self.tokenizer = load_tokenizer(processed / "tokenizer.json")
        self.dev = resolve_device(cfg.device, cfg.train.amp)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # The architecture must come from the checkpoint's own config, not from
        # whichever config file is driving this evaluation. Rebuilding a 6-layer
        # model for 4-layer weights fails loudly; rebuilding with a different
        # pooling or vocab would load cleanly and score nonsense.
        model_cfg = config_from_dict(payload["config"]) if payload.get("config") else cfg

        self.model = build("model", model_cfg.model.name, model_cfg, pad_id(self.tokenizer))
        self.model.load_state_dict(payload["model"])
        self.model.to(self.dev.device).eval()

        self.max_query_len = model_cfg.data.max_query_len
        self.max_doc_len = model_cfg.data.max_doc_len
        # Memory knob, read from the *evaluating* config: it describes this machine,
        # not the run that produced the checkpoint.
        self.encode_batch = max(1, cfg.retrieval.encode_batch)
        log.info(
            "dense retriever: %s step %s, %s",
            checkpoint.name, payload.get("step", "?"), self.dev.describe(),
        )

    def _encode(self, texts: list[str], max_len: int, tower: str) -> torch.Tensor:
        """Encode in length-sorted, bounded sub-batches, then restore input order.

        Both details are load-bearing, and the naive version was measurably worse
        than encoding one row at a time:

        **Sorted.** Padding is to the longest text in a sub-batch. Snippets average
        72 tokens against a 128 cap, so padding 2,400 mixed-length documents to one
        common width wasted ~1.8x the compute. Grouping similar lengths together
        pads to a local maximum instead.

        **Bounded.** One forward over every document of 256 rows pushed allocation
        to 7.7 GiB of 8.1, and this card does not raise on overcommit -- it spills to
        system RAM and crawls. `retrieval.encode_batch` caps the sub-batch so memory
        stays flat regardless of `batch_rows`.
        """
        encode = self.model.encode_query if tower == "query" else self.model.encode_doc
        pad = pad_id(self.tokenizer)
        encoded = [self.tokenizer.encode(text).ids[:max_len] or [pad] for text in texts]

        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        out: list[torch.Tensor | None] = [None] * len(encoded)

        for start in range(0, len(order), self.encode_batch):
            group = order[start : start + self.encode_batch]
            width = max(len(encoded[i]) for i in group)

            ids = torch.full((len(group), width), pad, dtype=torch.long)
            mask = torch.zeros((len(group), width), dtype=torch.bool)
            for slot, index in enumerate(group):
                row = encoded[index]
                ids[slot, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[slot, : len(row)] = True  # True marks REAL tokens throughout qar

            vectors = encode(ids.to(self.dev.device), mask.to(self.dev.device))
            for slot, index in enumerate(group):
                out[index] = vectors[slot]

        return torch.stack(out)  # back in the caller's order

    @torch.no_grad()
    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        with self.dev.autocast():
            q = self._encode([query], self.max_query_len, "query")
            d = self._encode(documents, self.max_doc_len, "doc")
            # Both towers L2-normalise their output, so this dot product is a cosine.
            # No temperature: it is a positive constant and cannot change a ranking.
            similarity = (q @ d.T).squeeze(0)
        return similarity.float().tolist()

    @torch.no_grad()
    def score_batch(self, queries: list[str], pools: list[list[str]]) -> list[list[float]]:
        """Encode every query and every pooled document in two forward passes.

        Scoring one row at a time is dominated by kernel-launch overhead at this
        model size, not by arithmetic -- whole validation took 18.5 min that way,
        which would cost more than the training it is meant to evaluate once an
        ablation grid runs. Documents from all rows are flattened into one batch and
        split back afterwards.
        """
        if not queries:
            return []

        flat, spans = [], []
        for pool in pools:
            spans.append((len(flat), len(pool)))
            flat.extend(pool)

        if not flat:  # every pool empty
            return [[] for _ in pools]

        with self.dev.autocast():
            q = self._encode(queries, self.max_query_len, "query")
            d = self._encode(flat, self.max_doc_len, "doc")
            # Per row: its own query against its own slice. Rows never see each
            # other's documents, so this is exactly the per-row computation.
            out = []
            for index, (start, size) in enumerate(spans):
                if size == 0:
                    out.append([])
                    continue
                out.append((q[index] @ d[start : start + size].T).float().tolist())
        return out
