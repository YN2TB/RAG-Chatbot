"""Choosing which snippet counts as the positive.

AmazonQA labels answers, not evidence: a row knows what the answer was but not
which review snippet supports it. Contrastive training needs a positive, so one
has to be inferred -- distant supervision, and the weakest link in the whole
pipeline.

Because it is the weakest link it is a registered, swappable component rather
than a hardcoded rule. `answer_overlap` is the default; `answer_recall` shifts the
length bias the other way; `first` is the null control that answers "does the
selector matter at all?" -- a row the DL report should contain.
"""

from __future__ import annotations

from qar.data.text import counts, f1_from_counts, recall_from_counts
from qar.registry import register


@register("selector", "answer_overlap")
def answer_overlap(snippets: list[str], answers: list[str]) -> tuple[int, float]:
    """Snippet with the highest token-F1 against any reference answer."""
    return _argmax(snippets, answers, f1_from_counts)


@register("selector", "answer_recall")
def answer_recall(snippets: list[str], answers: list[str]) -> tuple[int, float]:
    """Snippet covering the most answer tokens, with no penalty for being long."""
    return _argmax(snippets, answers, recall_from_counts)


@register("selector", "first")
def first(snippets: list[str], answers: list[str]) -> tuple[int, float]:
    """Null control: take the first snippet and claim nothing about it.

    Scores 1.0 so `min_positive_score` cannot silently discard rows under this
    selector -- the control has to see the same rows as the real thing for the
    comparison to mean anything.
    """
    return (0, 1.0) if snippets else (-1, 0.0)


def _argmax(snippets, answers, score_fn) -> tuple[int, float]:
    """Best (index, score) over the cartesian product of snippets and answers.

    Each text is tokenised once, not once per comparison: a row with 9 snippets and
    4 answers costs 13 tokenisations instead of 72.
    """
    if not snippets or not answers:
        return -1, 0.0

    answer_counts = [counts(answer) for answer in answers]
    best_idx, best = -1, 0.0
    for index, snippet in enumerate(snippets):
        snippet_counts = counts(snippet)
        for answer in answer_counts:
            score = score_fn(snippet_counts, answer)
            if score > best:
                best_idx, best = index, score
    return (best_idx, best) if best_idx >= 0 else (-1, 0.0)
