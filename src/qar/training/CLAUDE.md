# CLAUDE.md — src/qar/training/

> **Before touching this folder:** read `.claude/` (agent roster, permission
> rules) and the root `CLAUDE.md`. Record the change in `CHANGELOG.md` in the
> same commit — what was learned, not what the diff shows.

The loop and everything generic around it: precision, gradient accumulation and
clipping, LR scheduling, evaluation cadence, checkpoint rotation, early stopping,
resume, metric logging.

```
trainer.py    the loop
task.py       the Task interface + move_to
schedule.py   optimizer and LR schedule construction
checkpoint.py atomic save / load / rotate / find_latest
```

## Contract

**In** — a `RunConfig` and a `Task` instance. Nothing else; the trainer never reads
a file, an argument or an environment variable.

**Out** — `runs/<name>/{config.yaml, metrics.jsonl, checkpoints/}` plus a returned
summary `{"best": ..., "steps": ..., **last_val}`.

**The trainer knows nothing task-specific.** No bi-encoder, no InfoNCE, no
retrieval, and never an `if cfg.task == ...`. Adding the real retriever means
writing one `Task` subclass and changing nothing here. That invariant is the
reason this folder exists.

## trainer.py

`__init__` has side effects: creates `run_dir`, builds the model onto the resolved
device, builds optimizer and scheduler, creates the `GradScaler` (enabled only for
fp16), applies `grad_checkpoint` / `compile`, opens the `JsonlLogger`.

`train()`:

1. snapshots `config.yaml` **before** step 1
2. cycles the train loader endlessly — the loop is bounded by `max_steps`, never
   by epochs, so ablations stay comparable when the data subset size changes
3. per step: `grad_accum` micro-batches → clip → `scaler.step` → `scheduler.step`
4. `log_every` → `train/*` scalars + `lr` + `steps_per_s` + CUDA peak memory
5. `eval_every` → `task.validate` → `val/*`; a better `train.monitor` writes
   `best.pt`
6. `save_every` **or** `save_every_minutes` → `step_*.pt`, then `rotate(keep_last)`
7. on exit: a final eval **unless the cadence already covered this step**, a final
   checkpoint, `metrics.close()`

### Two checkpoint cadences

`save_every` is a step count, so the *real time* it represents swings with
throughput — 2000 steps is 7 minutes at 4.5 steps/s and 19 minutes at 1.75. What
bounds how much work an interruption destroys is wall-clock time, not steps, so
`train.save_every_minutes` (default 10.0, 0 disables) runs alongside it and whichever
comes first wins.

`_rotating_save` restarts the timer on **every** save however it was triggered, so a
step-driven checkpoint also defers the next time-driven one and the two cadences
cannot stack into two writes in a row.

Both go through the same atomic write and `rotate(keep_last)`, so a time-driven save
is not a second class of checkpoint — it resumes identically.

Early stopping counts evaluations without improvement against
`train.early_stop_patience` (0 disables) and never fires on the final eval.
A `monitor` key missing from the validation results only logs a warning — check
your key against what the task actually returns.

## task.py — the interface

Required: `build_model`, `train_loader`, `val_loader`, `training_step`.

```python
def training_step(self, model, batch) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (loss to backprop, scalars to log)."""
```

- Return the **full** loss. The trainer divides by `grad_accum` itself — dividing
  inside the task double-counts.
- The scalar dict must hold plain floats, not tensors.
- Metric keys are unprefixed. `Task.validate` adds `val/`, the trainer adds
  `train/`.
- The default `validate` is the mean of `validation_step` across batches. Override
  `validate` when the metric is not a batch mean — ranking over a full index is the
  case that will come up with the real retriever.

`move_to` walks dicts, lists and tuples, moves tensors with `non_blocking=True`,
and leaves everything else untouched.

## checkpoint.py

Payload keys: `model`, `optimizer`, `scheduler`, `scaler`, `step`, `best`,
`config`. Enough for exact resume — a run killed at step 6000 continues at 6000
rather than restarting, which matters when a sweep runs overnight on a laptop.

- Writes to `<path>.tmp` and `replace()`s it. **Never remove the atomic write**;
  a half-written checkpoint from an interrupted overnight run is unrecoverable.
- `rotate` deletes all but the `keep` newest `step_*.pt` by mtime. `best.pt` is
  never matched by the pattern and never deleted.
- `load_checkpoint` uses `weights_only=False` because it restores optimizer and
  scheduler state. Only ever load checkpoints this project produced.

## schedule.py

`build_scheduler(name in cosine|linear|constant)` with linear warmup over
`int(total_steps * warmup_ratio)` steps. Warmup is not optional for a from-scratch
transformer — Adam on a randomly initialised attention stack routinely diverges in
the first few hundred steps without it.

`build_optimizer` is AdamW only (anything else raises) and applies weight decay
only to parameters with `ndim >= 2` that are not biases or norm gains. Decaying
biases, LayerNorm gains and embeddings measurably hurts transformer training.

The scheduler advances once per optimizer step, not per micro-batch.

## Gotchas

- **`train/*` averaging under accumulation**, already fixed: `loss` and the task's
  extra scalars were accumulated once per *micro-batch* but divided by the number
  of *optimizer steps*, so every `train/*` value read `grad_accum`× too high. It
  was invisible at the default `grad_accum=1`. Each micro-batch now contributes
  only its `1/grad_accum` share; `grad_norm` is still per step and divided by
  `seen` alone. Any curve logged before this fix from a run with `grad_accum > 1`
  is wrong and must be regenerated. Guarded by
  `tests/test_smoke.py::test_train_scalars_are_not_scaled_by_grad_accum`.
- **Duplicate final eval**, already fixed: when `max_steps` was a multiple of
  `eval_every` the last step logged twice. `_last_eval_step` guards it — keep it.
- **Resume duplicates steps, by design.** Append-only is what lets a resumed run keep
  the curve from before the crash, but rewinding to the checkpoint's step leaves the
  interrupted run's later records in the file, so those steps hold two values from two
  trajectories. `maybe_resume` logs an `{"event": "resume"}` marker; readers must go
  through `qar.utils.logging.read_series`, which keeps the later record per step.
  Do not "fix" this by truncating the log on resume — the pre-crash curve is the
  reason the file is append-only in the first place.
- An unknown scheduler name raises from inside `lr_lambda`, so with a non-zero
  warmup the error surfaces at the end of warmup rather than at construction.
- `grad_clip > 0` calls `scaler.unscale_` first; with bf16 the scaler is disabled
  and that is a no-op, which is correct — bf16 must never get a scaler.

## Rules

- Nothing task-specific, ever. If the loop needs to know what it is training,
  the knowledge belongs behind `Task`.
- Every scalar goes through `JsonlLogger`. No `print`.
- New behaviour arrives as a config field with a default that preserves today's
  numbers, so old runs stay comparable.
