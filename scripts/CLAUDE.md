# CLAUDE.md — scripts/

> **Before touching this folder:** read `.claude/` (agent roster, permission
> rules) and the root `CLAUDE.md`. Record the change in `CHANGELOG.md` in the
> same commit — what was learned, not what the diff shows.

Command-line entry points. These are the only files a human runs directly, and they
are deliberately thin: parse arguments, wire components together, hand off to
`src/qar/`. No training logic, no metric definitions, no hyperparameter defaults
live here.

## Contract

**In** — a config path plus explicit overrides. Nothing else. No environment
variables, no interactive prompts, no implicit state.

**Out** — files under `runs/`. Everything a report needs is read back from disk,
never scraped from stdout.

## prepare_data.py

```
python scripts/prepare_data.py <config.yaml> [--set KEY=VALUE ...]
```

Offline, run once per prepare configuration. Reads `data.train_path`,
`data.val_path` and `data.test_path`, writes the processed corpus, the BPE and a
manifest into `data.processed_dir`. Everything it does is driven by `cfg.prepare` —
see [src/qar/data/CLAUDE.md](../src/qar/data/CLAUDE.md) for the contract.

`data.test_path` names the corpus's own test file (the default). Set it to null to
carve a test split out of validation instead — an older regime that costs half the
validation rows. A configured-but-missing file raises rather than downgrading
silently, and `manifest.test_source` records which regime ran.

Changing a `prepare.*` field produces a *different training corpus*. Point
`data.processed_dir` somewhere new rather than overwriting; the two are an ablation
pair, not successive versions of one thing.

`prepare.max_rows=N` bounds the pass for a smoke run.

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
| `evaluate` | no | off | score each cell's `best.pt` within-product; keys `split`, `max_rows` |

Note `monitor` here defaults to `val/recall@1`/`max`, which is **not** the same as
`train.monitor` (`val/loss`/`min`) that decides `best.pt`. The sweep re-reads
`metrics.jsonl` itself and does not look at the checkpoint.

### Rank cells on `within_*`, not on `monitor`

`monitor` can only name a key that exists in `metrics.jsonl`, and `val/recall@1`
there ranks a row against `eval_batch_size` candidates from **other products**. The
first trained run scored 0.4998 on it and **0.1707** within-product — below plain
token overlap. A grid ranked on the logged metric therefore selects its winner on the
easy problem.

`evaluate:` closes that hole. After each cell it runs `evaluate_retrieval.py` on that
cell's `best.pt` with `retrieval.out_name=<run name>`, and folds the result in as
`within_recall@1`, `within_mrr` and so on. Keep `monitor` as a training-health signal;
read `within_*` when deciding what the ablation showed. The prefix exists so the two
can never be mistaken for each other in one row.

`max_rows` under `evaluate:` trades precision for grid time. Drop it for any number
that goes in the report.

Outputs:

- `runs/<prefix>__<slug>/` — one full run directory per grid cell
- `runs/_sweeps/<prefix>.csv` — one row per cell
- `runs/_sweeps/<prefix>.md` — the same table, paste-ready for the report

- `runs/_baselines/<prefix>__<slug>.{json,md}` — per-cell within-product scores,
  written only when `evaluate:` is set

`slug` is the last segment of each key glued to its value, with `.` → `p` and
`-` → `m` (`loss.temperature=0.05` → `temperature0p05`). Row `status` is one of
`ok`, `failed(rc=N)`, `no-metrics`, `monitor-missing`; `within_status` is one of
`no-checkpoint`, `eval-failed`, `eval-unreadable`, or absent when the evaluation
succeeded.

### The grids that exist

| Sweep | Cells | Question |
|---|---|---|
| `negatives` | 3 | do hard negatives close the in-batch/within-product gap? |
| `temperature` | 3 | how sharply must the softmax separate candidates? |
| `pooling` | 4 | mean vs cls pooling, one tower vs two |
| `data_scaling` | 4 | more pairs, or better ones? |
| `multitask` | 3 | does the answerability head help retrieval or compete with it? |
| `dedup` | 2 | is the false-negative guard worth its cost? |

`negatives` is the one to run first — it tests the finding every other grid is
conditioned on. Note its `fixed:` block pins `hard_negatives=2` in the other grids,
which presumes that result; revisit them if it comes out otherwise.

Each cell runs as its own `subprocess` with `cwd=ROOT`. That is the point: a
diverging configuration cannot take down an overnight sweep, and VRAM is fully
released between cells. Do not "optimise" this into an in-process loop.

## build_idf.py

```
python scripts/build_idf.py <config.yaml> [--set KEY=VALUE ...]
```

One pass over every snippet in `train.jsonl`, writing
`<processed_dir>/<retrieval.idf_file>`. **Train only** — fitting document
frequencies on val or test would leak their term distribution into a scorer that is
then evaluated on them. Must run before `bm25_global`, which is in the default
baseline set, so `evaluate_retrieval.py` fails without it.

Measured 2026-08-06: ~3 min over 6,551,161 snippets, keeping 158,812 of 489,543
terms at `idf_min_df=5`, mean snippet length 48.9 tokens.

## evaluate_retrieval.py

```
python scripts/evaluate_retrieval.py <config.yaml> [--set KEY=VALUE ...]
```

Ranks `retrieval.split` with each name in `retrieval.baselines` and writes
`runs/_baselines/<retrieval.out_name or split>.{json,md}`. The Markdown is the table
the DL report needs before any trained number means anything. Scoring is **within one
product's pool** (~9 candidates) — never comparable with a global-index number, and
never with `val/recall@1` from a training log.

**Results merge, they do not replace.** Fresh rows fold into whatever the file holds,
so "measure the baselines once, add the trained model later" works. Rows measured
over a different row count are dropped loudly rather than left to sit unlabelled
beside a full-split number. `retrieval.out_name` gives a run its own file, which is
what stops twenty sweep cells — all named `dense` — from overwriting each other.

Scoring a trained checkpoint uses the same entry point, which is the whole reason
`dense` exists:

```
python scripts/evaluate_retrieval.py configs/retriever.yaml \
    --set retrieval.baselines=[dense] \
          retrieval.checkpoint=runs/<name>/checkpoints/best.pt
```

Cost scales with the split: whole validation is 87,475 rows, roughly double the
older half-validation figure.

**Select on val, report on test.** `retrieval.split=test` exists for the final
number only; a parameter chosen on a split cannot also be scored on it.

## Rules

- **Never hardcode a hyperparameter here.** If a run needs a knob, it becomes a
  field in `src/qar/config.py` with a default, and reaches the run via the config
  file or `--set`.
- **Never import a task module directly.** `import qar.tasks` for the registration
  side effect, then `build("task", cfg.task, cfg)`. A script that imports
  `dev_toy` by name defeats the registry.
- The `sys.path.insert(...)` before the `qar` imports is intentional — it lets the
  scripts run without installing the package. It needs **no** `# noqa: E402`: ruff
  does not treat a path insert before imports as a violation, and verifying that
  (`ruff check --extend-select E402 --ignore-noqa` reports zero) is what retired the
  markers this file used to tell you to add. `# noqa: F401` on a
  registration-side-effect import is still required and load-bearing.
- Anything worth reading twice goes to `metrics.jsonl`, not to `print`.
- A new script belongs here only if a human invokes it. Library code goes in
  `src/qar/`.
