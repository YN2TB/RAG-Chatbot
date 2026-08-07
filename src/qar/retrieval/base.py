"""The retriever interface, and what "retrieval" means in this project.

Every retriever here ranks **the snippets of one product** against one question.
That scope is a deliberate choice, not a shortcut: a user asking about a camera is
never served a sentence from someone else's blender review, so the realistic
candidate pool is the product's own reviews -- about nine snippets per question.

The consequence has to be stated wherever these numbers appear: recall@1 over a
pool of nine is not recall@1 over a 6.5M-snippet index, and the two must never
share a column in the report. A global-index evaluation is a separate, harder
measurement that the bi-encoder will need; this interface deliberately cannot
express it, so nobody can conflate them by accident.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from qar.config import RunConfig


class Retriever(ABC):
    """Scores a pool of documents against one query. Higher is better.

    Untrained by construction: these are the baselines the learned retriever has to
    beat, so none of them may see `positive_idx`.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def scores(self, query: str, documents: list[str]) -> list[float]:
        """One score per document, in the order the documents were given."""

    def score_batch(self, queries: list[str], pools: list[list[str]]) -> list[list[float]]:
        """Score several rows at once. Default: the per-row loop, unchanged.

        Only worth overriding where per-call overhead dominates. For the lexical
        baselines it does not -- they are pure Python over a handful of strings, and
        this default keeps them exactly as they were. For a neural retriever it is
        the difference between a launch-bound crawl and a useful evaluation.

        An override MUST return the same values the per-row path would; padding
        inside a batch must not change a score. `TextEncoder` masks padding out of
        its mean pooling, and `test_padding_cannot_change_the_representation` is
        what makes that safe to rely on here.
        """
        return [self.scores(query, pool) for query, pool in zip(queries, pools)]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
