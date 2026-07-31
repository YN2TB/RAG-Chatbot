"""The training loop.

Step-based rather than epoch-based, because ablations must be compared at equal
optimisation steps even when the data subset size differs -- which it will, once
you run the data-scaling curve.

Responsibilities: precision, gradient accumulation and clipping, LR scheduling,
evaluation cadence, checkpoint rotation, early stopping, resume, and metric
logging. Everything task-specific lives behind the `Task` interface.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from qar.config import RunConfig
from qar.training.checkpoint import find_latest, load_checkpoint, rotate, save_checkpoint
from qar.training.schedule import build_optimizer, build_scheduler
from qar.training.task import Task, move_to
from qar.utils.device import memory_summary, resolve_device
from qar.utils.logging import JsonlLogger, format_metrics, get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, cfg: RunConfig, task: Task) -> None:
        self.cfg = cfg
        self.task = task
        self.dev = resolve_device(cfg.device, cfg.train.amp)

        self.run_dir = cfg.run_dir
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.model = task.build_model().to(self.dev.device)
        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.scheduler = build_scheduler(
            self.optimizer,
            cfg.optim.scheduler,
            total_steps=cfg.train.max_steps,
            warmup_ratio=cfg.optim.warmup_ratio,
        )
        self.scaler = torch.amp.GradScaler(
            self.dev.device.type, enabled=self.dev.use_scaler
        )

        if cfg.train.grad_checkpoint and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if cfg.train.compile:
            self.model = torch.compile(self.model)

        self.step = 0
        self.best: float | None = None
        self.last_val: dict[str, float] = {}
        self._last_eval_step = -1
        self._since_improved = 0
        self.metrics = JsonlLogger(self.run_dir / "metrics.jsonl", cfg.name)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("run '%s' | %s", cfg.name, self.dev.describe())
        log.info("trainable parameters: %.2fM", n_params / 1e6)

    # ------------------------------------------------------------------ #

    def _better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.cfg.train.monitor_mode == "min" else value > self.best

    def _endless(self, loader) -> Iterator[Any]:
        """Cycle the loader so the loop is bounded by steps, not epochs."""
        while True:
            for batch in loader:
                yield batch

    def maybe_resume(self) -> None:
        latest = find_latest(self.ckpt_dir)
        if latest is None:
            return
        payload = load_checkpoint(
            latest,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.dev.use_scaler else None,
            map_location=self.dev.device,
        )
        self.step = int(payload.get("step", 0))
        self.best = payload.get("best")

    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, float]:
        cfg = self.cfg.train
        self.cfg.save(self.run_dir / "config.yaml")

        train_loader = self.task.train_loader()
        val_loader = self.task.val_loader()
        batches = self._endless(train_loader)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        seen = 0
        tick = time.time()

        while self.step < cfg.max_steps:
            # ---- accumulate ------------------------------------------- #
            for _ in range(cfg.grad_accum):
                batch = move_to(next(batches), self.dev.device)
                with self.dev.autocast():
                    loss, extra = self.task.training_step(self.model, batch)
                    loss = loss / cfg.grad_accum
                self.scaler.scale(loss).backward()

                # Both are averaged over `seen` optimisation steps below, so each
                # micro-batch may contribute only its 1/grad_accum share. `loss` is
                # already divided; the task's scalars are raw per-micro-batch means.
                running["loss"] = running.get("loss", 0.0) + float(loss.detach())
                for k, v in extra.items():
                    running[k] = running.get(k, 0.0) + float(v) / cfg.grad_accum
            seen += 1

            # ---- step -------------------------------------------------- #
            if self.cfg.optim.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip
                )
                running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # ---- log ---------------------------------------------------- #
            if self.step % cfg.log_every == 0:
                elapsed = time.time() - tick
                scalars = {k: v / seen for k, v in running.items()}
                scalars["lr"] = self.scheduler.get_last_lr()[0]
                scalars["steps_per_s"] = seen / max(elapsed, 1e-9)
                scalars.update(memory_summary(self.dev.device))
                self.metrics.log(self.step, {f"train/{k}": v for k, v in scalars.items()})
                log.info("step %6d | %s", self.step, format_metrics(scalars))
                running, seen, tick = {}, 0, time.time()

            # ---- evaluate ----------------------------------------------- #
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                if self._evaluate(val_loader):
                    break

            if cfg.save_every and self.step % cfg.save_every == 0:
                self._save(self.ckpt_dir / f"step_{self.step:07d}.pt")
                rotate(self.ckpt_dir, cfg.keep_last)

        if self._last_eval_step != self.step:  # the cadence may already have covered it
            self._evaluate(val_loader, final=True)
        self._save(self.ckpt_dir / f"step_{self.step:07d}.pt")
        self.metrics.close()
        return {"best": self.best, "steps": self.step, **self.last_val}

    # ------------------------------------------------------------------ #

    def _evaluate(self, val_loader, final: bool = False) -> bool:
        results = self.task.validate(self.model, val_loader, self.dev)
        self._last_eval_step = self.step
        if not results:
            return False

        self.last_val = results
        self.metrics.log(self.step, results, split="val")
        log.info("step %6d | EVAL %s", self.step, format_metrics(results))

        monitor = self.cfg.train.monitor
        if monitor in results:
            value = results[monitor]
            if self._better(value):
                self.best = value
                self._since_improved = 0
                self._save(self.ckpt_dir / "best.pt")
                log.info("new best %s=%.4f", monitor, value)
            else:
                self._since_improved += 1

            patience = self.cfg.train.early_stop_patience
            if patience and self._since_improved >= patience and not final:
                log.info("early stop: no %s improvement in %d evals", monitor, patience)
                return True
        elif not final:
            log.warning("monitor '%s' not in metrics %s", monitor, sorted(results))
        return False

    def _save(self, path: Path) -> None:
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.dev.use_scaler else None,
            step=self.step,
            best=self.best,
            config=self.cfg.to_dict(),
        )
