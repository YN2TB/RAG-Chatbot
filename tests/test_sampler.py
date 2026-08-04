"""The false-negative guard: no batch may contain two rows asking the same question."""

from __future__ import annotations

import numpy as np
import pytest

from qar.data.sampler import QGroupBatchSampler


def _sampler(groups, batch_size=4, seed=0):
    return QGroupBatchSampler(np.asarray(groups, dtype=np.uint64), batch_size, seed=seed)


def test_no_batch_repeats_a_question_group():
    """The property the whole class exists for."""
    groups = [i % 25 for i in range(400)]  # every question asked 16 times
    for batch in _sampler(groups, batch_size=8):
        assert len({groups[i] for i in batch}) == len(batch)


def test_every_batch_is_exactly_batch_size():
    """InfoNCE labels are arange(batch) and batch size is the negative count, so a
    short batch would make one step an easier problem than its neighbours."""
    for batch in _sampler([i % 30 for i in range(300)], batch_size=7):
        assert len(batch) == 7


def test_indices_are_never_repeated_within_an_epoch():
    groups = [i % 40 for i in range(400)]
    seen = [index for batch in _sampler(groups, batch_size=8) for index in batch]
    assert len(seen) == len(set(seen)), "a row was sampled twice in one epoch"


def test_all_distinct_groups_place_every_row():
    """With no collisions possible, nothing may be dropped."""
    sampler = _sampler(list(range(200)), batch_size=10)
    placed = sum(len(batch) for batch in sampler)
    assert placed == 200 and sampler.dropped == 0


def test_realistic_collision_rate_loses_almost_nothing():
    """Train is ~9% duplicated questions; the guard must not cost real data."""
    rng = np.random.default_rng(0)
    groups = np.arange(5000)
    groups[rng.choice(5000, 450, replace=False)] = 7  # force a 9% pile-up
    sampler = _sampler(groups, batch_size=32)

    placed = sum(len(batch) for batch in sampler)
    assert placed > 4500, f"guard dropped too much: kept {placed} of 5000"


def test_fewer_distinct_questions_than_batch_size_is_rejected():
    """No valid batch can exist, so yielding nothing would read to the trainer as an
    empty epoch forever. It has to fail where the cause is visible."""
    with pytest.raises(ValueError, match="distinct question groups"):
        _sampler([1] * 100, batch_size=4)


def test_a_skewed_split_terminates_and_reports_what_it_dropped():
    """Four distinct questions but one of them asked 100 times: only a handful of
    batches are constructible. It must give up after bounded passes, not loop."""
    groups = [0] * 100 + [1, 2, 3]
    sampler = _sampler(groups, batch_size=4)

    batches = list(sampler)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) <= 1, "cannot build more batches than the rarest group allows"
    assert sampler.dropped > 90, "most rows were unplaceable and should be reported"


def test_epochs_differ_but_a_seed_reproduces_them():
    groups = [i % 50 for i in range(200)]
    a, b = _sampler(groups, seed=3), _sampler(groups, seed=3)
    assert list(a) == list(b), "same seed gave a different first epoch"

    first = list(_sampler(groups, seed=3))
    sampler = _sampler(groups, seed=3)
    list(sampler)
    assert list(sampler) != first, "epoch 2 repeated epoch 1's order"


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        _sampler([1, 2, 3], batch_size=0)
