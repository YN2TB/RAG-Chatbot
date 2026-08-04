"""Baselines that read nothing, and exist to make the real numbers interpretable.

`random` is the floor. Every other number in the table is only meaningful as a
distance above it, and with a variable pool size that floor cannot be quoted from
theory -- a pool of 9 gives recall@1 of 1/9, a pool of 3 gives 1/3, and the corpus
mixes them. Measure it.

`first` asks a question that is easy to forget to ask: **is the snippet pool
already ordered?** If taking snippet 0 every time beats BM25, the pool carries a
relevance ordering from upstream, and any retriever that does not beat `first` has
learned nothing worth reporting.
"""

from __future__ import annotations

import random

from qar.registry import register
from qar.retrieval.base import Retriever


@register("retriever", "random")
class RandomRetriever(Retriever):
    """Uniform random scores. Seeded per instance so a run is reproducible."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._rng = random.Random(cfg.retrieval.tie_break_seed)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [self._rng.random() for _ in documents]


@register("retriever", "first")
class FirstRetriever(Retriever):
    """Position prior: rank by the order the pool arrived in."""

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [float(-index) for index in range(len(documents))]
