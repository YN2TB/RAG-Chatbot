"""Within-product retrieval: the untrained baselines and the trained bi-encoder.

Importing this package registers every retriever under the "retriever" registry
key, so `retrieval.baselines` in a config resolves without any direct import.

Everything here except `dense` is untrained by construction; `dense` loads a
checkpoint and exists so a training run can be scored on exactly the rows, pools
and metric code the baseline table used.
"""

from qar.retrieval import dense, lexical, trivial  # noqa: F401  (registration side effect)
from qar.retrieval.base import Retriever
from qar.retrieval.evaluate import evaluate_retriever, markdown_table, merge_results

__all__ = ["Retriever", "evaluate_retriever", "markdown_table", "merge_results"]
