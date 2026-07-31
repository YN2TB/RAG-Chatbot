# CLAUDE.md — scripts/

Command-line entry points. These are the only files a human runs directly, and they
are deliberately thin: parse arguments, wire components together, hand off to
`src/qar/`. No training logic, no metric definitions, no hyperparameter defaults
live here.

## Contract

**In** — a config path plus explicit overrides. Nothing else. No environment
variables, no interactive prompts, no implicit state.

**Out** — files under `runs/`. Everything a report needs is read back from disk,
never scraped from stdout.

## train.py

```
python scripts/train.py <config.yaml> [--set KEY=VALUE ...] [--resume]
```

| | |
|---|---|
| `config` | positional, path to a YAML config (may use `_base_:`) |
| `--set` | zero or more `dotted.key=value` overrides, e.g. `optim.lr=1e-4` |
| `--resume` | continue from the newest `step_*.pt` in the run's checkpoint dir |

Writes to `runs/<cfg.name>/`:

| Path | Contents |
|---|---|
| `config.yaml` | the fully resolved config, snapshotted before step 1 |
| `metrics.jsonl` | append-only, one JSON record per log/eval point |
| `train.log` | human-readable console tee |
| `checkpoints/step_*.pt` | rotated, `train.keep_last` newest kept |
| `checkpoints/best.pt` | best `train.monitor` so far, never rotated |

Exit code 0 on success. A non-zero exit is what `sweep.py` reads as a failed cell.

Order of operations is load-bearing: `load_config` → `setup_logging` →
`seed_everything` → build task → `Trainer` → optional `maybe_resume` → `train`.
Seeding happens **before** the model is constructed, otherwise weight init is not
reproducible.

## sweep.py

```
python scripts/sweep.py <sweep.yaml> [--dry-run]
```

Sweep file keys:

| Key | Required | Default | Meaning |
|---|---|---|---|
| `config` | yes | — | base config every cell starts from |
| `grid` | yes | — | `dotted.key: [v1, v2, ...]`; the cartesian product is run |
| `prefix` | no | sweep filename stem | run-name prefix |
| `fixed` | no | `[]` | extra `key=value` overrides applied to every cell |
| `monitor` | no | `val/recall@1` | metric key read back out of `metrics.jsonl` |
| `monitor_mode` | no | `max` | `min` or `max` |

Note `monitor` here defaults to `val/recall@1`/`max`, which is **not** the same as
`train.monitor` (`val/loss`/`min`) that decides `best.pt`. The sweep re-reads
`metrics.jsonl` itself and does not look at the checkpoint.

Outputs:

- `runs/<prefix>__<slug>/` — one full run directory per grid cell
- `runs/_sweeps/<prefix>.csv` — one row per cell
- `runs/_sweeps/<prefix>.md` — the same table, paste-ready for the report

`slug` is the last segment of each key glued to its value, with `.` → `p` and
`-` → `m` (`loss.temperature=0.05` → `temperature0p05`). Row `status` is one of
`ok`, `failed(rc=N)`, `no-metrics`, `monitor-missing`.

Each cell runs as its own `subprocess` with `cwd=ROOT`. That is the point: a
diverging configuration cannot take down an overnight sweep, and VRAM is fully
released between cells. Do not "optimise" this into an in-process loop.

## Rules

- **Never hardcode a hyperparameter here.** If a run needs a knob, it becomes a
  field in `src/qar/config.py` with a default, and reaches the run via the config
  file or `--set`.
- **Never import a task module directly.** `import qar.tasks` for the registration
  side effect, then `build("task", cfg.task, cfg)`. A script that imports
  `dev_toy` by name defeats the registry.
- The `sys.path.insert(...)` before the `qar` imports is intentional — it lets the
  scripts run without installing the package. Keep the `# noqa: E402` markers if
  you add a script.
- Anything worth reading twice goes to `metrics.jsonl`, not to `print`.
- A new script belongs here only if a human invokes it. Library code goes in
  `src/qar/`.
