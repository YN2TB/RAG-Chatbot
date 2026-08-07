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
        self._last_save = time.time()
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
        """Cycle the loader so the loop is bounded by steps, not epochs.

        A loader that yields nothing would otherwise spin here forever, burning CPU
        and looking exactly like a slow first step. A batch sampler that cannot
        satisfy its constraint is the realistic way to get one.
        """
        while True:
            empty = True
            for batch in loader:
                empty = False
                yield batch
            if empty:
                raise RuntimeError(
                    "train loader produced no batches; check data.batch_size against "
                    "the split size and data.dedup_questions_in_batch"
                )

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

        # The log is append-only, so rewinding to the checkpoint's step leaves the
        # records from the abandoned tail still in the file: steps between here and
        # wherever the interrupted run got to will appear twice, with different
        # values. Mark the rewind so a reader can tell the two trajectories apart
        # instead of plotting a curve that jumps backwards.
        self.metrics.log(self.step, {"event": "resume", "from_checkpoint": latest.name})

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

            # Two independent cadences, whichever comes first. `save_every` alone is
            # a step count, so the real-time gap between checkpoints swings with
            # throughput: 2000 steps is 7 minutes at 4.5 steps/s and 19 at 1.75. The
            # wall-clock bound is what actually caps how much work an interruption
            # can destroy.
            due_by_step = cfg.save_every and self.step % cfg.save_every == 0
            due_by_time = (
                cfg.save_every_minutes > 0
                and time.time() - self._last_save >= cfg.save_every_minutes * 60
            )
            if due_by_step or due_by_time:
                self._rotating_save()

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

    def _rotating_save(self) -> None:
        """Write a step checkpoint, rotate, and restart the wall-clock timer.

        The timer restarts on every save however it was triggered, so a step-driven
        save also defers the next time-driven one — the two cadences cannot stack up
        and write twice in a row.
        """
        self._save(self.ckpt_dir / f"step_{self.step:07d}.pt")
        rotate(self.ckpt_dir, self.cfg.train.keep_last)
        self._last_save = time.time()

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
