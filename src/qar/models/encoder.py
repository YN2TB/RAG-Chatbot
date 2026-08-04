"""A transformer text encoder, built here rather than downloaded.

The DL report's question is whether *this training setup* learns a good
representation, so the encoder has to be ours: randomly initialised, trained only
by the contrastive objective, with every architectural choice visible and
ablatable. Pretrained encoders appear in the NLP half as a baseline row, never as
the thing being studied.

Pre-norm blocks, not post-norm. A randomly initialised post-norm stack needs
careful warmup to train at all; pre-norm keeps the residual path clean and makes
the from-scratch run reproducible on a laptop rather than a coin flip. Warmup is
still on (`optim.warmup_ratio`), but it is now insurance rather than a load-bearing
trick.
"""

from __future__ import annotations

import torch
from torch import nn


class EncoderBlock(nn.Module):
    """Pre-norm self-attention + feed-forward, the standard modern arrangement."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(x)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=pad_mask, need_weights=False
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


class TextEncoder(nn.Module):
    """Token ids in, one pooled vector out.

    `mask` is True for real tokens -- the opposite of the convention
    `nn.MultiheadAttention` wants, so it is inverted exactly once, here, rather
    than at every call site.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pooling: str = "mean",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"pooling must be mean|cls, got {pooling!r}")
        self.pooling = pooling
        self.max_len = max_len

        self.tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positions = nn.Embedding(max_len, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small normal init, the GPT-2 recipe.

        PyTorch's default `nn.Linear` init is tuned for shallow nets and gives a
        deep stack activations large enough to stall early training. `padding_idx`
        is re-zeroed because `normal_` overwrites it.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, length = ids.shape
        if length > self.max_len:
            raise ValueError(
                f"sequence of {length} exceeds model.max_len={self.max_len}; "
                "raise it or lower data.max_query_len / data.max_doc_len"
            )

        positions = torch.arange(length, device=ids.device).unsqueeze(0)
        x = self.embed_dropout(self.embed_norm(self.tokens(ids) + self.positions(positions)))

        pad_mask = ~mask  # True marks positions attention must ignore
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.final_norm(x)

        return x[:, 0] if self.pooling == "cls" else _masked_mean(x, mask)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over real tokens only.

    Padding is excluded rather than averaged in: a batch pads to its longest
    sequence, so including it would make a short question's representation depend
    on which other questions happened to share its batch. The denominator is
    clamped so an all-padding row returns zeros instead of NaN.
    """
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
