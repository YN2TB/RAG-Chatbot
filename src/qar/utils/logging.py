"""Console logging plus a machine-readable metric stream.

Every scalar goes to `<run_dir>/metrics.jsonl` as one JSON object per record, so
the ablation table and the training curves in your report are a pandas read_json
away and never have to be transcribed by hand.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger once, optionally teeing to `<run_dir>/train.log`."""
    root = logging.getLogger()
    if root.handlers:  # idempotent: repeated calls in tests must not duplicate output
        return
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(console)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        file.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(file)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class JsonlLogger:
    """Append-only scalar sink. One JSON object per line, flushed as it goes.

    Append-only matters for resumed runs: a crash at step 8000 leaves the first
    8000 steps of curve intact, and the resumed run continues the same file.
    """

    def __init__(self, path: Path, run_name: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._start = time.time()
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, step: int, metrics: dict[str, Any], **extra: Any) -> None:
        record = {
            "run": self.run_name,
            "step": step,
            "wall_s": round(time.time() - self._start, 3),
            **{k: _plain(v) for k, v in metrics.items()},
            **extra,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _plain(value: Any) -> Any:
    """Unwrap tensors/numpy scalars so the record stays JSON-serialisable."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return str(value)
    return value


def read_series(path: Path, key: str) -> list[tuple[int, float]]:
    """Every `(step, value)` for one metric, with rewinds resolved, step-ordered.

    `metrics.jsonl` is append-only, which is what lets a resumed run keep the curve
    from before the crash. The cost is that resuming rewinds to the checkpoint's
    step while the records from the abandoned tail stay in the file, so a step can
    carry two different values from two different trajectories.

    The later record wins: appends are chronological, so the last value written for
    a step is the one the surviving run actually produced. Reading the raw file
    instead gives a curve that jumps backwards, and a "best" that may come from a
    trajectory that was thrown away.
    """
    latest: dict[int, float] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in record and "step" in record:
                latest[int(record["step"])] = record[key]
    return sorted(latest.items())


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        v = _plain(v)
        parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)
