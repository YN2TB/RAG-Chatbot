"""Raw AmazonQA rows in, validated records out.

The raw corpus carries four precomputed baseline fields the harness never trains
on -- `top_sentences_IR`, `top_review_wilson`, `top_review_helpful` and
`random_sentence`. They are the dataset authors' own baselines and belong in the
results table, not in the training pipeline, so this parser drops them
deliberately rather than by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BASELINE_FIELDS = (
    "top_sentences_IR",
    "top_review_wilson",
    "top_review_helpful",
    "random_sentence",
)


@dataclass(slots=True)
class RawRow:
    qid: str
    asin: str
    category: str
    question: str
    question_type: str
    is_answerable: int
    snippets: list[str]
    answers: list[str]


def parse_row(raw: dict[str, Any], *, min_snippet_tokens: int, max_snippets: int) -> RawRow | None:
    """Validate one raw row, or return None with the reason recorded by the caller.

    Returning None rather than raising is deliberate: a 738k-row corpus will always
    contain a few malformed rows, and a single bad line must not abort a 20-minute
    preparation pass. The counts land in the manifest so the loss is visible.
    """
    question = _clean(raw.get("questionText"))
    asin = _clean(raw.get("asin"))
    if not question or not asin:
        return None

    snippets = [s for s in (_clean(x) for x in _as_list(raw.get("review_snippets")))
                if len(s.split()) >= min_snippet_tokens]
    if not snippets:
        return None
    snippets = snippets[:max_snippets]

    answers = [
        text
        for text in (_clean(a.get("answerText")) for a in _as_list(raw.get("answers"))
                     if isinstance(a, dict))
        if text
    ]

    return RawRow(
        qid=str(raw.get("qid", "")),
        asin=asin,
        category=_clean(raw.get("category")) or "unknown",
        question=question,
        question_type=_clean(raw.get("questionType")) or "unknown",
        is_answerable=int(bool(raw.get("is_answerable", 0))),
        snippets=snippets,
        answers=answers,
    )


def _clean(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
