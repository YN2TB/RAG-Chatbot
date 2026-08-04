"""Count corpus-wide document frequencies for the `bm25_global` baseline.

    python scripts/build_idf.py configs/base.yaml
    python scripts/build_idf.py configs/base.yaml --set retrieval.idf_min_df=10

One pass over every snippet in `train.jsonl`, writing
`<processed_dir>/<retrieval.idf_file>`. Built from **train only**: fitting term
statistics on val or test would leak their vocabulary distribution into a baseline
those splits are then used to score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qar.config import load_config
from qar.data.dataset import PairDataset
from qar.retrieval.idf import build_document_frequencies, save_idf
from qar.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted config overrides, e.g. retrieval.idf_min_df=10",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging()
    log = get_logger("idf")

    processed = Path(cfg.data.processed_dir)
    dataset = PairDataset(processed / "train.jsonl")
    log.info("counting over %d training rows", len(dataset))

    table = build_document_frequencies(dataset, min_df=cfg.retrieval.idf_min_df)
    out = processed / cfg.retrieval.idf_file
    save_idf(out, table)

    log.info(
        "wrote %s: %d snippets, %d terms of %d kept at min_df=%d, avg_len=%.1f",
        out, table["n_docs"], len(table["df"]), table["vocabulary"],
        table["min_df"], table["avg_len"],
    )


if __name__ == "__main__":
    main()
