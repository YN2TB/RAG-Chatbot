"""Run the untrained retrieval baselines and write the results table.

    python scripts/evaluate_retrieval.py configs/base.yaml
    python scripts/evaluate_retrieval.py configs/base.yaml --set retrieval.split=test
    python scripts/evaluate_retrieval.py configs/base.yaml --set retrieval.max_rows=2000

Reads `data.processed_dir`, so `scripts/prepare_data.py` has to have run first.
Writes `runs/_baselines/<split>.json` and `.md` -- the Markdown is the table the DL
report needs before any trained number means anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qar import retrieval  # noqa: F401  (registers every baseline)
from qar.config import load_config
from qar.data.dataset import PairDataset
from qar.registry import build
from qar.retrieval.evaluate import evaluate_retriever, markdown_table
from qar.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted config overrides, e.g. retrieval.split=test",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging()
    log = get_logger("baselines")

    split_path = Path(cfg.data.processed_dir) / f"{cfg.retrieval.split}.jsonl"
    dataset = PairDataset(split_path)
    log.info("%s: %d rows", split_path, len(dataset))

    results = {}
    for name in cfg.retrieval.baselines:
        retriever = build("retriever", name, cfg)
        results[name] = evaluate_retriever(cfg, retriever, dataset)
        log.info("%-10s %s", name, results[name]["overall"])

    out_dir = Path(cfg.out_dir) / "_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": cfg.retrieval.split,
        "processed_dir": cfg.data.processed_dir,
        "bm25": {"k1": cfg.retrieval.bm25_k1, "b": cfg.retrieval.bm25_b},
        "tie_break_seed": cfg.retrieval.tie_break_seed,
        "results": results,
    }
    (out_dir / f"{cfg.retrieval.split}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    table = markdown_table(results, cfg.retrieval.ks)
    (out_dir / f"{cfg.retrieval.split}.md").write_text(table + "\n", encoding="utf-8")

    print(f"\npool size: {next(iter(results.values()))['mean_pool']} candidates on average\n")
    print(table)


if __name__ == "__main__":
    main()
