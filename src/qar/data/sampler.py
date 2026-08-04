"""Batching that does not hand InfoNCE a false negative.

Train has 704,201 rows but only 638,306 distinct questions: roughly 66,000 rows
share their question text with another row. In-batch contrastive training treats
every other row's positive as a negative, so when two of those land in the same
batch the loss punishes the model for scoring a *correct* snippet highly. The
gradient says "this right answer is wrong".

`data.dedup_questions_in_batch` turns this sampler on. The ablation against it
belongs in the DL report -- it is a rare case where the bug and the control are the
same experiment.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

# A row deferred this many times is dropped for the epoch rather than chased
# further; with ~9% collisions the tail is tiny and chasing it costs more than it
# is worth.
_MAX_PASSES = 3


class QGroupBatchSampler(Sampler[list[int]]):
    """Yields batches whose rows all have distinct question groups.

    Batches are always exactly `batch_size` long. That is not tidiness: InfoNCE
    labels are `arange(batch)` and the batch size *is* the number of negatives, so
    a short batch would quietly make one step an easier problem than its neighbours.
    """

    def __init__(
        self, groups: np.ndarray, batch_size: int, seed: int = 0, epoch: int = 0
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # With fewer distinct questions than the batch size, no valid batch can ever
        # be built and this sampler would yield nothing at all -- which reads to the
        # trainer as an empty epoch, forever. Fail here, where the cause is visible.
        distinct = len(np.unique(groups))
        if distinct < batch_size:
            raise ValueError(
                f"data.batch_size={batch_size} exceeds the {distinct} distinct question "
                "groups in this split, so no de-duplicated batch exists. Lower the batch "
                "size or set data.dedup_questions_in_batch=false"
            )

        self.groups = groups
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.dropped = 0  # rows this sampler could not place; read after an epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pending = rng.permutation(len(self.groups))
        self.epoch += 1
        self.dropped = 0

        for _ in range(_MAX_PASSES):
            if len(pending) == 0:
                break
            batches, pending = self._pack(pending)
            yield from batches

        self.dropped = len(pending)

    def _pack(self, indices: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        """One greedy pass: fill batches with distinct groups, defer the clashes."""
        batches: list[list[int]] = []
        deferred: list[int] = []
        batch: list[int] = []
        seen: set[int] = set()

        for index in indices:
            group = int(self.groups[index])
            if group in seen:
                deferred.append(int(index))
                continue
            batch.append(int(index))
            seen.add(group)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch, seen = [], set()

        # An incomplete trailing batch goes back into the pool rather than being
        # yielded short.
        deferred.extend(batch)
        return batches, np.asarray(deferred, dtype=np.int64)

    def __len__(self) -> int:
        """Upper bound. The real count is slightly lower once clashes are dropped,
        and the trainer is step-based so nothing depends on this being exact."""
        return len(self.groups) // self.batch_size
