"""Untrained retrieval baselines.

Importing this package registers every baseline under the "retriever" registry key,
so `retrieval.baselines` in a config resolves without any direct import.
"""

from qar.retrieval import lexical, trivial  # noqa: F401  (registration side effect)
from qar.retrieval.base import Retriever
from qar.retrieval.evaluate import evaluate_retriever, markdown_table

__all__ = ["Retriever", "evaluate_retriever", "markdown_table"]
