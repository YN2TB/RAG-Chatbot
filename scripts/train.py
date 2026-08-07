"""Single training run.

    python scripts/train.py configs/dev.yaml
    python scripts/train.py configs/dev.yaml --set optim.lr=1e-4 loss.temperature=0.02
    python scripts/train.py configs/dev.yaml --set name=lr1e-4 --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qar import tasks  # noqa: F401  (registers every task)
from qar.config import load_config
from qar.registry import build
from qar.training.trainer import Trainer
from qar.utils.logging import get_logger, setup_logging
from qar.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
        help="dotted config overrides, e.g. optim.lr=1e-4",
    )
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(cfg.run_dir)
    log = get_logger("train")
    log.info("config: %s%s", args.config, f" + {args.overrides}" if args.overrides else "")

    seed_everything(cfg.seed, cfg.deterministic)
    task = build("task", cfg.task, cfg)
    trainer = Trainer(cfg, task)
    if args.resume:
        trainer.maybe_resume()

    results = trainer.train()
    log.info("done: %s", results)


if __name__ == "__main__":
    main()
