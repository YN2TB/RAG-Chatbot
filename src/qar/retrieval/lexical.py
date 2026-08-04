"""Lexical baselines: BM25 and plain token overlap.

Both tokenise with `qar.data.text`, the same normaliser that chose the positives.
That is on purpose -- it keeps "what counts as a token" identical across the whole
pipeline, so a gap between these baselines and the learned retriever is a gap in
*modelling* rather than in preprocessing.

There is no circularity in scoring this way. Positives were selected by overlap
with the **answer**; these baselines rank by overlap with the **question**. A
snippet can easily support the answer while sharing few words with the question --
that gap is precisely the part of the task a lexical method cannot do.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from qar.data.text import tokens
from qar.registry import register
from qar.retrieval.base import Retriever
from qar.retrieval.idf import idf_lookup, load_idf


@register("retriever", "bm25")
class BM25(Retriever):
    """Okapi BM25 over the product's own snippets.

    Document frequencies come from the pool itself, which is all a within-product
    retriever has. With ~9 documents the IDF term is coarse -- it can tell a word
    appearing in one snippet from one appearing in all nine, and little else. A
    corpus-wide IDF table would be sharper and is the obvious next improvement;
    this version is self-contained and needs no extra pass over 6.5M snippets.
    """

    def scores(self, query: str, documents: list[str]) -> list[float]:
        k1 = self.cfg.retrieval.bm25_k1
        b = self.cfg.retrieval.bm25_b

        doc_tokens = [tokens(document) for document in documents]
        lengths = [len(d) for d in doc_tokens]
        n_docs = len(doc_tokens)
        if not n_docs:
            return []
        avg_len = sum(lengths) / n_docs or 1.0

        frequencies = [Counter(d) for d in doc_tokens]
        document_freq = Counter()
        for counts in frequencies:
            document_freq.update(counts.keys())

        query_terms = set(tokens(query))
        results = []
        for counts, length in zip(frequencies, lengths):
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(
                    1.0 + (n_docs - document_freq[term] + 0.5) / (document_freq[term] + 0.5)
                )
                norm = freq + k1 * (1.0 - b + b * length / avg_len)
                score += idf * freq * (k1 + 1.0) / norm
            results.append(score)
        return results


@register("retriever", "bm25_global")
class BM25Global(BM25):
    """BM25 with document frequencies estimated over the whole training corpus.

    The only difference from `bm25` is where IDF comes from, which is the whole
    point: it isolates a single design decision, so the gap between the two rows in
    the results table measures exactly that decision and nothing else.

    Length normalisation also uses the corpus average snippet length rather than
    the pool's, for the same reason.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        table = load_idf(Path(cfg.data.processed_dir) / cfg.retrieval.idf_file)
        self._idf = idf_lookup(table)
        self._avg_len = table["avg_len"] or 1.0

    def scores(self, query: str, documents: list[str]) -> list[float]:
        k1, b = self.cfg.retrieval.bm25_k1, self.cfg.retrieval.bm25_b
        query_terms = set(tokens(query))
        if not query_terms:
            return [0.0] * len(documents)

        results = []
        for document in documents:
            document_tokens = tokens(document)
            counts = Counter(document_tokens)
            length = len(document_tokens)
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                norm = freq + k1 * (1.0 - b + b * length / self._avg_len)
                score += self._idf(term) * freq * (k1 + 1.0) / norm
            results.append(score)
        return results


@register("retriever", "bm25_noidf")
class BM25NoIDF(BM25Global):
    """`bm25_global` with every IDF set to 1.

    The controlled comparison. `bm25_global` and this differ in exactly one term,
    so the gap between their rows measures what IDF weighting is worth on this task
    and nothing else -- which matters, because the raw table says a lexical method
    without IDF (`overlap`) beats one with it, and that claim needs an isolated
    variable behind it rather than two implementations that differ in several ways.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._idf = lambda term: 1.0


@register("retriever", "overlap")
class TokenOverlap(Retriever):
    """Question-snippet token F1. BM25 stripped of IDF and length saturation.

    Worth running beside BM25: if the two score the same, BM25's weighting is
    buying nothing at this pool size and the report should say so rather than
    imply a tuned baseline.
    """

    def scores(self, query: str, documents: list[str]) -> list[float]:
        query_counts = Counter(tokens(query))
        query_total = sum(query_counts.values())
        if not query_total:
            return [0.0] * len(documents)

        results = []
        for document in documents:
            counts = Counter(tokens(document))
            total = sum(counts.values())
            overlap = sum((counts & query_counts).values())
            if not overlap or not total:
                results.append(0.0)
                continue
            precision, recall = overlap / total, overlap / query_total
            results.append(2 * precision * recall / (precision + recall))
        return results
