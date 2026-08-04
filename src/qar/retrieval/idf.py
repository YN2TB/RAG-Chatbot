"""Corpus-wide document frequencies, because pool-local IDF is worse than useless.

Measured on the validation split, BM25 with IDF estimated from the ~9 snippets of
one product scores recall@1 0.177 -- **below** plain token F1 at 0.215. The reason
is not noise but sign: within one product, the terms that recur across its reviews
are the product's own features, exactly the terms a question about it turns on.
Pool-local IDF down-weights them precisely because they recur.

Estimating document frequency over all 6.5M training snippets restores the
intended meaning: rare *in the corpus* rather than rare *in this product*.

A document here is one snippet, not one product, because a snippet is the unit
being ranked.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from qar.data.text import tokens
from qar.utils.logging import get_logger

log = get_logger(__name__)

_LOG_EVERY = 100_000


def build_document_frequencies(
    dataset, min_df: int = 5, max_rows: int | None = None
) -> dict[str, Any]:
    """Count how many snippets each term appears in, over a prepared split.

    Terms below `min_df` are dropped: the tail is dominated by typos and one-off
    model numbers, and keeping it triples the file for terms no question will ever
    contain. A term absent from the table is treated as maximally rare at lookup.
    """
    frequencies: Counter[str] = Counter()
    n_docs = 0
    total_len = 0

    limit = min(max_rows or len(dataset), len(dataset))
    for index in range(limit):
        if index and index % _LOG_EVERY == 0:
            log.info("  %d rows, %d snippets, %d terms", index, n_docs, len(frequencies))
        for snippet in dataset[index]["snippets"]:
            terms = tokens(snippet)
            n_docs += 1
            total_len += len(terms)
            frequencies.update(set(terms))

    pruned = {term: count for term, count in frequencies.items() if count >= min_df}
    log.info("%d snippets, %d terms kept of %d", n_docs, len(pruned), len(frequencies))
    return {
        "n_docs": n_docs,
        "avg_len": round(total_len / n_docs, 4) if n_docs else 0.0,
        "min_df": min_df,
        "vocabulary": len(frequencies),
        "df": pruned,
    }


def save_idf(path: str | Path, table: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table), encoding="utf-8")


def load_idf(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no IDF table at {path}; run scripts/build_idf.py before using bm25_global"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def idf_lookup(table: dict[str, Any]):
    """Return a term -> IDF function with a floor for unseen terms.

    An unseen term was pruned or never occurred, so it is at least as rare as the
    `min_df` cutoff; scoring it as `min_df` is the conservative reading and stops a
    typo in a question from dominating the whole score.
    """
    n_docs = table["n_docs"]
    df = table["df"]
    floor = max(1, table.get("min_df", 1))

    def idf(term: str) -> float:
        count = df.get(term, floor)
        return math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))

    return idf
