"""Two towers, one contrastive space.

A bi-encoder embeds question and snippet **independently**, which is the property
that makes retrieval possible at all: snippets can be encoded once, ahead of time,
and a question then needs one forward pass rather than one per candidate. A
cross-encoder scores better and cannot be indexed; that trade is why production
retrieval is bi-encoder first, cross-encoder second.

Outputs are L2-normalised, so the dot product in the loss is a cosine similarity
and `loss.temperature` alone controls how sharp the softmax over candidates is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qar.models.encoder import TextEncoder
from qar.registry import register


@register("model", "biencoder")
class BiEncoder(nn.Module):
    """Query tower, document tower, and an optional answerability head.

    `model.share_encoder` decides whether the towers are one module or two. Two is
    the DPR default and lets each side specialise -- questions and review prose are
    genuinely different registers. One halves the parameters and gives every
    gradient twice the data, which on 700k pairs from scratch may well win. It is
    an ablation, which is why it is a config field rather than a decision.
    """

    def __init__(self, cfg, pad_id: int = 0) -> None:
        super().__init__()
        model = cfg.model
        self.share_encoder = model.share_encoder

        def tower() -> TextEncoder:
            return TextEncoder(
                vocab_size=model.vocab_size,
                d_model=model.d_model,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_ff=model.d_ff,
                dropout=model.dropout,
                max_len=model.max_len,
                pooling=model.pooling,
                pad_id=pad_id,
            )

        self.query_encoder = tower()
        self.doc_encoder = self.query_encoder if model.share_encoder else tower()

        # Built only when it is used: an unused head would still be checkpointed and
        # would make `loss.answerable_weight=0` runs incomparable with older ones.
        self.answerable_head = (
            nn.Linear(model.d_model, 1) if cfg.loss.answerable_weight > 0 else None
        )

    def encode_query(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(ids, mask), dim=-1)

    def encode_doc(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.doc_encoder(ids, mask), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]):
        """Returns (query vectors, document vectors, answerability logits or None).

        The answerability head reads the **pooled question only**, deliberately. It
        could be given the positive document as well, but that positive was itself
        inferred by distant supervision -- letting the head see it would teach it to
        predict the selector rather than answerability. Question-only keeps the
        auxiliary task honest, and its job here is to regularise the query tower.
        """
        pooled_query = self.query_encoder(batch["query_ids"], batch["query_mask"])
        query = F.normalize(pooled_query, dim=-1)
        doc = self.encode_doc(batch["doc_ids"], batch["doc_mask"])

        logits = None
        if self.answerable_head is not None:
            logits = self.answerable_head(pooled_query).squeeze(-1)
        return query, doc, logits
