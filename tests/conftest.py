"""Make `src/` importable, pin the working directory, and fence off the corpus.

Tests refer to `configs/*.yaml` by repo-relative path, so they must run from the
project root regardless of where pytest was invoked.
"""

import builtins
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# The raw AmazonQA files, which no test may read. A single pass over one of them
# costs minutes, so a test that reaches for one does not fail -- it hangs, and the
# suite looks broken rather than wrong.
_CORPUS = {"train-qar.jsonl", "val-qar.jsonl", "test-qar_all.jsonl"}


@pytest.fixture(autouse=True)
def _no_corpus_reads(monkeypatch):
    """Fail loudly if a test opens a raw corpus file.

    `tests/CLAUDE.md` has always said tests must not touch the 3.4 GB corpus, but
    prose does not enforce anything: when `data.test_path` gained a real default,
    every prepare test that forgot to null it silently started reading the 751 MB
    test file. This turns that into an immediate, named failure.
    """
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).name in _CORPUS:
            raise AssertionError(
                f"test opened the raw corpus file {Path(file).name!r}. Point the "
                f"config at a fixture under tmp_path (see the _cfg helpers), or "
                f"pass data.test_path=null."
            )
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
