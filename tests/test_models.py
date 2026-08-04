"""The from-scratch encoder and the bi-encoder built on it.

Shape and invariant checks only -- whether the architecture *learns* is
`test_retriever.py`'s job, and whether it learns anything useful is the DL report's.
"""

from __future__ import annotations

import pytest
import torch

from qar.config import load_config
from qar.models.biencoder import BiEncoder
from qar.models.encoder import TextEncoder, _masked_mean
from qar.registry import available, build

VOCAB, PAD = 64, 0


def _encoder(**overrides):
    kwargs = {
        "vocab_size": VOCAB, "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64,
        "dropout": 0.0, "max_len": 16, "pooling": "mean", "pad_id": PAD,
    }
    return TextEncoder(**{**kwargs, **overrides})


def _batch(rows=3, length=8):
    ids = torch.randint(1, VOCAB, (rows, length))
    mask = torch.ones((rows, length), dtype=torch.bool)
    return ids, mask


def _cfg(tmp_path, **overrides):
    base = [
        f"out_dir={tmp_path.as_posix()}",
        "model.name=biencoder", "model.vocab_size=64", "model.d_model=32",
        "model.n_layers=2", "model.n_heads=4", "model.d_ff=64", "model.max_len=16",
        "model.dropout=0.0",
    ]
    return load_config("configs/base.yaml", base + [f"{k}={v}" for k, v in overrides.items()])


# -- encoder ---------------------------------------------------------------- #


def test_encoder_pools_to_one_vector_per_row():
    ids, mask = _batch(rows=5, length=9)
    assert _encoder()(ids, mask).shape == (5, 32)


def test_padding_cannot_change_the_representation():
    """The core masking guarantee: a batch pads to its longest row, so if padding
    leaked into the pooled vector a question's embedding would depend on which
    other questions shared its batch."""
    encoder = _encoder().eval()
    ids = torch.randint(1, VOCAB, (1, 5))
    mask = torch.ones((1, 5), dtype=torch.bool)

    padded_ids = torch.cat([ids, torch.full((1, 6), PAD)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros((1, 6), dtype=torch.bool)], dim=1)

    with torch.no_grad():
        assert torch.allclose(encoder(ids, mask), encoder(padded_ids, padded_mask), atol=1e-5)


def test_masked_mean_ignores_padded_positions():
    x = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])
    mask = torch.tensor([[True, False]])
    assert torch.allclose(_masked_mean(x, mask), torch.tensor([[1.0, 1.0]]))


def test_masked_mean_survives_an_all_padding_row():
    x = torch.ones((1, 3, 2))
    out = _masked_mean(x, torch.zeros((1, 3), dtype=torch.bool))
    assert torch.isfinite(out).all() and out.abs().sum() == 0.0


def test_cls_pooling_takes_the_first_position():
    ids, mask = _batch(rows=2, length=6)
    assert _encoder(pooling="cls")(ids, mask).shape == (2, 32)


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="mean|cls"):
        _encoder(pooling="sum")


def test_sequence_longer_than_max_len_is_a_clear_error():
    ids, mask = _batch(rows=1, length=17)
    with pytest.raises(ValueError, match="max_len"):
        _encoder()(ids, mask)


def test_padding_embedding_stays_zero_after_init():
    """`normal_` overwrites what `padding_idx` zeroed, so init has to restore it."""
    assert _encoder().tokens.weight[PAD].abs().sum() == 0.0


def test_gradients_reach_every_parameter():
    encoder = _encoder()
    ids, mask = _batch()
    encoder(ids, mask).sum().backward()
    missing = [n for n, p in encoder.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached {missing}"


# -- bi-encoder ------------------------------------------------------------- #


def test_biencoder_is_registered():
    assert "biencoder" in available("model")


def test_outputs_are_unit_vectors(tmp_path):
    model = build("model", "biencoder", _cfg(tmp_path), PAD).eval()
    ids, mask = _batch(rows=4)
    batch = {"query_ids": ids, "query_mask": mask, "doc_ids": ids, "doc_mask": mask}

    with torch.no_grad():
        query, doc, _ = model(batch)
    for vectors in (query, doc):
        assert torch.allclose(vectors.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_separate_towers_do_not_share_parameters(tmp_path):
    model = BiEncoder(_cfg(tmp_path, **{"model.share_encoder": "false"}), PAD)
    assert model.query_encoder is not model.doc_encoder


def test_shared_tower_halves_the_parameters(tmp_path):
    separate = BiEncoder(_cfg(tmp_path, **{"model.share_encoder": "false"}), PAD)
    shared = BiEncoder(_cfg(tmp_path, **{"model.share_encoder": "true"}), PAD)

    assert shared.query_encoder is shared.doc_encoder
    n_separate = sum(p.numel() for p in separate.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    assert n_shared == pytest.approx(n_separate / 2, rel=0.01)


def test_answerability_head_exists_only_when_weighted(tmp_path):
    """An unused head would still be checkpointed and would make a zero-weight run
    incomparable with the runs it is supposed to be a control for."""
    assert BiEncoder(_cfg(tmp_path, **{"loss.answerable_weight": 0.0}), PAD).answerable_head is None
    assert BiEncoder(_cfg(tmp_path, **{"loss.answerable_weight": 0.5}), PAD).answerable_head


def test_answerability_logits_are_one_per_row(tmp_path):
    model = BiEncoder(_cfg(tmp_path, **{"loss.answerable_weight": 0.5}), PAD).eval()
    ids, mask = _batch(rows=6)
    batch = {"query_ids": ids, "query_mask": mask, "doc_ids": ids, "doc_mask": mask}
    with torch.no_grad():
        _, _, logits = model(batch)
    assert logits.shape == (6,)
