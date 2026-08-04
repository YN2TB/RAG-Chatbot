"""Text normalisation shared by positive selection and question grouping.

One normaliser, used everywhere, so that "does this snippet overlap the answer?"
and "are these two rows asking the same question?" cannot silently disagree about
what counts as the same token.

The rules are SQuAD's official normalisation -- lowercase, strip punctuation, drop
articles, collapse whitespace. Borrowing a published definition rather than
inventing one means the overlap scores in the report are comparable to numbers
other people quote.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    text = _ARTICLES.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    normalised = normalise(text)
    return normalised.split() if normalised else []


class TokenCounts(NamedTuple):
    """A text's token multiset and how many tokens it holds."""

    bag: Counter
    total: int


def counts(text: str) -> TokenCounts:
    """Count a text once so it can be scored many times.

    Selection compares every snippet against every answer, so a row with 9 snippets
    and 4 answers would otherwise tokenise 72 times instead of 13. Over 738k rows
    that is the difference between minutes and an hour.
    """
    bag = Counter(tokens(text))
    return TokenCounts(bag, sum(bag.values()))


def f1_from_counts(candidate: TokenCounts, reference: TokenCounts) -> float:
    """Harmonic mean of token precision and recall (SQuAD F1).

    Punishes a snippet for being much longer than the answer as well as for
    missing its content. That length penalty is a real modelling choice: it biases
    selection towards focused snippets. Use recall instead for the opposite bias.
    """
    overlap = _overlap(candidate, reference)
    if not overlap:
        return 0.0
    precision = overlap / candidate.total
    recall = overlap / reference.total
    return 2 * precision * recall / (precision + recall)


def recall_from_counts(candidate: TokenCounts, reference: TokenCounts) -> float:
    """Fraction of the reference's tokens present in the candidate.

    No length penalty, so it favours long snippets -- which is sometimes right
    (a long snippet really may contain the answer) and sometimes just lazy.
    """
    return _overlap(candidate, reference) / reference.total if reference.total else 0.0


def token_f1(candidate: str, reference: str) -> float:
    """String-level convenience wrapper. Prefer the counts form in hot loops."""
    return f1_from_counts(counts(candidate), counts(reference))


def token_recall(candidate: str, reference: str) -> float:
    """String-level convenience wrapper. Prefer the counts form in hot loops."""
    return recall_from_counts(counts(candidate), counts(reference))


def _overlap(candidate: TokenCounts, reference: TokenCounts) -> int:
    if not candidate.total or not reference.total:
        return 0
    return sum((candidate.bag & reference.bag).values())
