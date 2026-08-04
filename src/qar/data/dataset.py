"""Reading the processed corpus without loading it into memory.

`train.jsonl` carries every snippet of every product, which is the whole raw
corpus again -- gigabytes. Holding it as Python objects would cost more RAM than
the machine has once torch is also resident, so the dataset indexes the file by
byte offset and seeks per item. The offset table for 700k rows is a few MB and is
cached next to the file.

The file is opened in binary mode on purpose: Windows text mode translates
newlines, which makes `tell()` return values that `seek()` cannot use.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from qar.data.tokenizer import pad_id


def line_offsets(path: str | Path) -> np.ndarray:
    """Byte offset of every non-empty line, cached beside the file."""
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".offsets.npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)

    offsets, position = [], 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)

    table = np.asarray(offsets, dtype=np.int64)
    np.save(cache, table)
    return table


def question_groups(path: str | Path) -> np.ndarray:
    """Question-group id per row, as uint64, cached beside the file.

    `qgroup` is a 64-bit blake2b digest of the normalised question, written at
    prepare time. Loading it as integers rather than strings keeps 700k rows to
    5.6 MB and lets the batch sampler compare with array operations.
    """
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".qgroups.npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.load(cache)

    groups = []
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                groups.append(int(json.loads(line)["qgroup"], 16))

    table = np.asarray(groups, dtype=np.uint64)
    np.save(cache, table)
    return table


class PairDataset(Dataset):
    """One processed record per index: question, snippet pool, positive index."""

    def __init__(self, path: str | Path, subset: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} is missing; run scripts/prepare_data.py first"
            )
        self.offsets = line_offsets(self.path)
        if subset is not None:
            self.offsets = self.offsets[:subset]
        self._handle = None  # opened lazily: a file object cannot be pickled to a worker

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        handle = self._open()
        handle.seek(int(self.offsets[index]))
        return json.loads(handle.readline())

    def _open(self):
        if self._handle is None:
            self._handle = self.path.open("rb")
        return self._handle

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_handle": None}


class PairCollator:
    """Records in, padded tensors out.

    The query and its positive are always emitted; the negatives for plain InfoNCE
    are the other rows' positives in the same batch. Keys are unprefixed and flat so
    the batch passes through `move_to` unchanged.

    With `hard_negatives > 0` each row also carries that many snippets drawn from
    **its own product's pool**. That is the negative that matters. An in-batch
    negative comes from a different product, so telling it apart only requires
    recognising the topic -- a model can score well on it while learning nothing
    about whether a snippet *answers the question*. A snippet from the same product
    shares all the topic vocabulary and differs only in relevance.

    They are kept row-private (never shared across the batch) on purpose: two rows in
    one batch can concern the same product, and a shared hard negative could then be
    another row's positive -- a false negative reintroduced by the very mechanism
    meant to sharpen the loss.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_query_len: int,
        max_doc_len: int,
        hard_negatives: int = 0,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.hard_negatives = hard_negatives
        self.pad = pad_id(tokenizer)
        # With num_workers > 0 each worker copies this stream, so the negative draws
        # repeat across workers. Harmless (it costs a little sampling diversity, not
        # correctness) and num_workers is 0 by default on Windows.
        self._rng = random.Random(seed)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [record["question"] for record in records]
        positives = [record["snippets"][record["positive_idx"]] for record in records]

        query_ids, query_mask = self._encode(queries, self.max_query_len)
        doc_ids, doc_mask = self._encode(positives, self.max_doc_len)
        batch = {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "doc_ids": doc_ids,
            "doc_mask": doc_mask,
            "is_answerable": torch.tensor(
                [float(record["is_answerable"]) for record in records], dtype=torch.float32
            ),
        }
        if self.hard_negatives:
            batch.update(self._hard_negatives(records))
        return batch

    def _hard_negatives(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Sample same-product snippets, flagging the slots no pool could fill."""
        n = self.hard_negatives
        texts: list[str] = []
        valid: list[bool] = []

        for record in records:
            pool = [
                snippet
                for index, snippet in enumerate(record["snippets"])
                if index != record["positive_idx"]
            ]
            for slot in range(n):
                if pool:
                    texts.append(pool[self._rng.randrange(len(pool))] if len(pool) > 1
                                 else pool[0])
                    valid.append(True)
                else:
                    # A single-snippet pool has no same-product negative. Emit a
                    # placeholder and mask it out of the softmax rather than pretend.
                    texts.append("")
                    valid.append(False)

        ids, mask = self._encode(texts, self.max_doc_len)
        rows = len(records)
        return {
            "neg_ids": ids.view(rows, n, -1),
            "neg_mask": mask.view(rows, n, -1),
            "neg_valid": torch.tensor(valid, dtype=torch.bool).view(rows, n),
        }

    def _encode(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Truncate to `max_len`, pad to the longest in the batch.

        Padding to the batch maximum rather than to `max_len` is what makes short
        questions cheap; the mask is what stops the padding reaching the loss.
        """
        sequences = [encoding.ids[:max_len] for encoding in self.tokenizer.encode_batch(texts)]
        width = max(1, max(len(sequence) for sequence in sequences))

        ids = torch.full((len(sequences), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            if sequence:
                ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                mask[row, : len(sequence)] = True
        return ids, mask
