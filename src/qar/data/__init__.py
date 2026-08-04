"""Corpus preparation and loading.

Importing this package registers every positive selector under the "selector"
registry key, so `prepare.selector` in a config resolves without any direct import.
"""

from qar.data import select  # noqa: F401  (import for registration side effect)
from qar.data.dataset import PairCollator, PairDataset, line_offsets
from qar.data.prepare import prepare, question_group
from qar.data.split import assign_split
from qar.data.tokenizer import load_tokenizer, pad_id, train_tokenizer

__all__ = [
    "PairCollator",
    "PairDataset",
    "assign_split",
    "line_offsets",
    "load_tokenizer",
    "pad_id",
    "prepare",
    "question_group",
    "train_tokenizer",
]
