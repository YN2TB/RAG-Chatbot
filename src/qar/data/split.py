"""Deterministic, product-disjoint splitting.

The corpus ships a train file and a validation file that are already
product-disjoint, but no test file. Carving test out of validation has to keep
that property: a product whose reviews appear in val must not also appear in
test, or the final number measures memorisation of a product rather than
generalisation to a new one.

Hashing the asin rather than shuffling row indices is what guarantees it. Every
row of a product hashes identically, so the product lands whole on one side, and
the assignment is reproducible without storing a manifest of ids.
"""

from __future__ import annotations

from hashlib import blake2b

_DIGEST_BYTES = 8
_SCALE = float(1 << (8 * _DIGEST_BYTES))


def asin_bucket(asin: str, seed: int) -> float:
    """Stable pseudo-random number in [0, 1) for a product id."""
    digest = blake2b(asin.encode("utf-8"), digest_size=_DIGEST_BYTES,
                     key=str(seed).encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / _SCALE


def assign_split(asin: str, test_fraction: float, seed: int) -> str:
    """`"test"` for the chosen fraction of products, `"val"` for the rest.

    `test_fraction=0.0` puts everything in val, `1.0` everything in test; both are
    legitimate for a run that only needs one of the two.
    """
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}")
    return "test" if asin_bucket(asin, seed) < test_fraction else "val"
