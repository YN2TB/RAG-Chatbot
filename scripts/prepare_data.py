"""Build the processed corpus from the raw AmazonQA files.

    python scripts/prepare_data.py configs/dev.yaml
    python scripts/prepare_data.py configs/base.yaml --set prepare.max_rows=5000
    python scripts/prepare_data.py configs/base.yaml --set prepare.selector=first \
        data.processed_dir=data/processed_first

Run once per prepare configuration. Changing `prepare.selector` or
`prepare.min_positive_score` changes the training data, so point
`data.processed_dir` somewhere new rather than overwriting -- the two corpora are
an ablation pair, not successive versions of one thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qar.config import load_config
from qar.data.prepare import prepare
from qar.utils.logging import get_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted config overrides, e.g. prepare.max_rows=5000",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging()
    log = get_logger("prepare")

    manifest = prepare(cfg)

    log.info("wrote %s", Path(cfg.data.processed_dir).resolve())
    print(json.dumps(manifest["splits"], indent=2))
    print(json.dumps(manifest["leakage"], indent=2))


if __name__ == "__main__":
    main()
