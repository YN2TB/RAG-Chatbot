# CLAUDE.md — src/qar/utils/

Cross-cutting infrastructure: reproducibility, device/precision resolution,
logging. Used by the trainer and the scripts, depended on by everything.

```
seed.py      seed_everything, worker_init_fn
device.py    resolve_device -> DeviceInfo, memory_summary
logging.py   setup_logging, get_logger, JsonlLogger, format_metrics
```

## Contract

**In** — plain scalars and paths. Never a `RunConfig`, never a model, never a
`Task`.

**Out** — a `DeviceInfo`, a logger, or a side effect on global state (RNG seeds,
root logger handlers, files on disk).

This folder is a **leaf**: it imports `torch`/`numpy` and nothing from `qar`.
`qar.training` imports it, so any import back the other way is a cycle. Everything
here must be importable without a GPU.

## seed.py

`seed_everything(seed, deterministic=False)` seeds `PYTHONHASHSEED`, `random`,
`numpy`, `torch` and all CUDA devices. **Call it before the model is built** —
`scripts/train.py` does, and weight init is not reproducible otherwise.

`deterministic=True` additionally pins cuDNN to reproducible kernels, sets
`CUBLAS_WORKSPACE_CONFIG` and enables `use_deterministic_algorithms(warn_only=True)`.
It costs roughly 10–20% throughput: use it for the runs that go in the report,
leave it off while iterating. `deterministic=False` turns `cudnn.benchmark` on.

`worker_init_fn` gives each DataLoader worker a distinct, run-reproducible seed.
Pass it whenever `data.num_workers > 0`, or the workers draw identical noise.

An ablation table is only evidence if the runs differ by the thing you changed and
not by the RNG. Report the seed in the results table.

## device.py

`resolve_device(spec="auto", amp="bf16") -> DeviceInfo(device, amp_dtype, use_scaler)`

| Input | Result |
|---|---|
| `spec="auto"` | cuda if available, else cpu |
| `spec="cuda"`, no CUDA | `RuntimeError` — explicit request, loud failure |
| cpu, or `amp="off"` | `amp_dtype=None`, autocast disabled |
| `amp="bf16"` on a bf16 card | `bfloat16`, **no scaler** |
| `amp="bf16"` without bf16 support | silently degrades to `float16` + scaler |
| `amp="fp16"` | `float16` + scaler |
| anything else | `ValueError` |

`use_scaler` is True for fp16 only. The target machine (RTX 5060 Laptop, Blackwell
sm_120) is native bf16, so the default path never constructs a live scaler — bf16
with a GradScaler is wrong, not merely redundant.

`DeviceInfo.autocast()` is the context manager the trainer wraps the forward pass
in; `describe()` is the one-line startup banner.

`memory_summary(device)` returns `{}` on CPU, otherwise `mem/alloc_mib` and
`mem/reserved_mib`. These are **peaks since process start** — nothing resets them,
so they answer "did this config fit in 8 GB?", not "what is it using now".

## logging.py

`setup_logging(run_dir)` configures the root logger once and tees to
`<run_dir>/train.log`. It is **idempotent by early return** if the root logger
already has handlers — that is what keeps pytest from duplicating every line.
Consequence: the first caller wins, so call it once, early, from the entry script.
Library modules use `get_logger(__name__)` and never configure anything.

`JsonlLogger(path, run_name)` is the machine-readable stream. One JSON object per
line, flushed on every write, opened in **append** mode — a run that crashes at
step 8000 keeps the first 8000 steps of curve, and the resumed run continues the
same file. Every record carries `run`, `step` and `wall_s` before the metrics.

That schema is a contract, not an implementation detail: `scripts/sweep.py` reads
`step` and the monitored key back out of `metrics.jsonl`, and
`tests/test_metrics_are_valid_jsonl` asserts `step` and `wall_s` are present.
Curves and ablation tables in both reports are read from this file — never
transcribed by hand from `train.log`.

`_plain` unwraps anything with `.item()` so tensors stay JSON-serialisable.
`format_metrics` is console-only; never parse its output.

## Rules

- No `qar` imports. Ever.
- Nothing here may require CUDA at import time.
- `train.log` is for humans, `metrics.jsonl` is for the report. Anything you will
  read twice goes to the latter.
- Changing a `JsonlLogger` field name invalidates every `metrics.jsonl` already on
  disk. Add fields; do not rename them.
