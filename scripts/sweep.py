"""Run an ablation grid and collect the results into one table.

    python scripts/sweep.py sweeps/temperature.yaml

A sweep file names a base config and a grid of overrides:

    config: configs/dev.yaml
    prefix: temp
    grid:
      loss.temperature: [0.01, 0.05, 0.1]
      optim.lr: [1e-4, 3e-4]

Each cell runs as its own subprocess, so one diverging configuration cannot take
down the rest of an overnight sweep, and GPU memory is fully released between
runs. Results land in `runs/<prefix>__<key>/` plus a summary CSV and a Markdown
table you can paste straight into the report.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qar.utils.logging import read_series


def slug(pairs: list[tuple[str, object]]) -> str:
    """Compact, filesystem-safe run name from the overridden values."""
    parts = [f"{key.split('.')[-1]}{value}" for key, value in pairs]
    return "_".join(parts).replace(".", "p").replace("-", "m")


def best_from(run_dir: Path, monitor: str, mode: str) -> dict[str, object]:
    """Best monitored value and the step it occurred at, read back from the log."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {"status": "no-metrics"}

    # read_series resolves resume rewinds: a step logged twice keeps the later
    # value, so "best" can never come from a trajectory that was abandoned.
    series = read_series(path, monitor)
    if not series:
        return {"status": "monitor-missing"}

    pick = min if mode == "min" else max
    at_step, best = pick(series, key=lambda pair: pair[1])
    return {
        "status": "ok",
        "best": round(float(best), 5),
        "best_step": at_step,
        "final_step": series[-1][0],
    }


def within_product(name: str, config: str, options: dict) -> dict[str, object]:
    """Score this cell's checkpoint the way the baseline table is scored.

    `best_from` reads `metrics.jsonl`, whose `val/recall@1` ranks a row against
    `eval_batch_size` candidates from *other products*. A grid ranked on that picks
    its winner on the easy problem: the first trained run scored 0.4998 there and
    0.1707 within-product, below plain token overlap. So a sweep that reports only
    the in-batch number would hand the report a table of the wrong metric.
    """
    checkpoint = ROOT / "runs" / name / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        return {"within_status": "no-checkpoint"}

    overrides = [
        "retrieval.baselines=[dense]",
        f"retrieval.checkpoint={checkpoint.as_posix()}",
        f"retrieval.out_name={name}",  # per cell, or every cell overwrites "dense"
        f"retrieval.split={options.get('split', 'val')}",
    ]
    if options.get("max_rows"):
        overrides.append(f"retrieval.max_rows={options['max_rows']}")

    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_retrieval.py"), config,
           "--set", *overrides]
    if subprocess.run(cmd, cwd=ROOT, check=False).returncode != 0:
        return {"within_status": "eval-failed"}

    path = ROOT / "runs" / "_baselines" / f"{name}.json"
    try:
        overall = json.loads(path.read_text(encoding="utf-8"))["results"]["dense"]["overall"]
    except (json.JSONDecodeError, KeyError, OSError):
        return {"within_status": "eval-unreadable"}

    # Prefixed so no column can be mistaken for the in-batch number beside it.
    return {f"within_{k}": v for k, v in overall.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the runs and exit")
    args = parser.parse_args()

    spec = yaml.safe_load(args.sweep.read_text(encoding="utf-8"))
    base_config = spec["config"]
    prefix = spec.get("prefix", args.sweep.stem)
    grid: dict[str, list] = spec["grid"]
    extra: list[str] = spec.get("fixed", [])
    monitor = spec.get("monitor", "val/recall@1")
    mode = spec.get("monitor_mode", "max")
    evaluate = spec.get("evaluate")

    keys = list(grid)
    combos = [list(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    print(f"{len(combos)} run(s) from {args.sweep}\n")

    rows = []
    for i, pairs in enumerate(combos, 1):
        name = f"{prefix}__{slug(pairs)}"
        overrides = [f"{k}={v}" for k, v in pairs] + [f"name={name}", *extra]
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), base_config, "--set", *overrides]

        print(f"[{i}/{len(combos)}] {name}")
        if args.dry_run:
            print("   ", " ".join(cmd))
            continue

        completed = subprocess.run(cmd, cwd=ROOT, check=False)
        result = {"run": name, **dict(pairs)}
        if completed.returncode != 0:
            result["status"] = f"failed(rc={completed.returncode})"
        else:
            result.update(best_from(ROOT / "runs" / name, monitor, mode))
            if evaluate:
                result.update(within_product(name, base_config, evaluate))
        rows.append(result)

    if args.dry_run or not rows:
        return

    out_dir = ROOT / "runs" / "_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list({k: None for row in rows for k in row})

    csv_path = out_dir / f"{prefix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    md += ["| " + " | ".join(str(row.get(f, "")) for f in fields) + " |" for row in rows]
    md_path = out_dir / f"{prefix}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n{csv_path}\n{md_path}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
