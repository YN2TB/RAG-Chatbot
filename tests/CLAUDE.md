# CLAUDE.md — tests/

Harness correctness, not research results. 16 tests, all CPU, all a few seconds.
Run before every experiment: if these fail, any number the experiment produces is
meaningless.

```bash
uv run pytest
```

## Contract

**In** — the repo's real config files (`configs/base.yaml`, `configs/dev.yaml`) and
nothing else. No AmazonQA corpus, no GPU, no network, no `runs/` directory.

**Out** — pass/fail only. Every byte a test writes goes to pytest's `tmp_path` via
an `out_dir=` override. A test that writes into `runs/` is a bug.

`conftest.py` puts `src/` on `sys.path` and `os.chdir(ROOT)`, so repo-relative
paths like `"configs/dev.yaml"` resolve no matter where pytest was invoked from.

## Files

**test_config.py** (10) — config loading in isolation. `_base_:` inheritance
overriding only named fields, CLI overrides landing as typed values rather than
strings, `Optional` fields accepting `null`, unknown keys raising, and
save→reload round-tripping to an identical dict.

**test_smoke.py** (6) — the whole loop on `dev_toy`:

| Test | Asserts |
|---|---|
| `test_dev_toy_is_registered` | import side effect populated the registry |
| `test_model_can_overfit` | final `train/loss` < 0.7 × first, `val/recall@1` > 0.5 |
| `test_checkpoint_resume_restores_step` | a fresh `Trainer` resumes at step 100 |
| `test_grad_accum_matches_large_batch` | 32×2 tracks 64×1 within `rel=0.35` |
| `test_train_scalars_are_not_scaled_by_grad_accum` | `train/*` is a mean over micro-batches, not a sum |
| `test_metrics_are_valid_jsonl` | every line parses, has `step` and `wall_s`; `config.yaml` was snapshotted |

The last two are complementary: the first checks `val/*` (computed by
`Task.validate`, never touched by the accumulation bug), the second checks the
`train/*` running averages the trainer maintains itself.

The overfit test is the important one. `dev_toy` is learnable by construction, so
a loss that will not fall means the harness is broken, not the idea.

## Coupling you must not break silently

These tests read the real configs, so `configs/*.yaml` is part of the test
contract:

- `configs/base.yaml`: `optim.scheduler == "cosine"`, `optim.lr == 3e-4`,
  `data.train_path` ends with `train-qar.jsonl`
- `configs/dev.yaml`: `name == "dev"`, `train.max_steps == 300`, and it must not
  override `optim.lr` (the inheritance test depends on it being inherited)

Changing a default in those files without updating the assertion produces a red
suite that looks like a code regression. Change both, in the same commit.

`test_metrics_are_valid_jsonl` also pins the `metrics.jsonl` schema that
`scripts/sweep.py` parses — `step` and `wall_s` on every record.

## Rules

- **CPU only.** Every smoke test forces `device=cpu`, `train.amp=off`,
  `data.num_workers=0`. Nothing here may require CUDA or Windows-specific
  behaviour.
- **No corpus.** Tests must never touch the 3.4 GB `*.jsonl` files. Synthetic data
  from `dev_toy` is the only data source.
- **`out_dir` is always `tmp_path`.** Use the `_cfg` helper in `test_smoke.py`.
- The numeric thresholds (0.7, 0.5, `rel=0.35`) are loose on purpose — they catch
  a broken loop without going flaky on a seed. Tighten only with evidence, and
  never "fix" a failure by relaxing one.
- The YAML 1.1 boolean tests are regression guards for a bug that already shipped
  once (`train.amp=off` silently became `"False"`). They stay.
- A new component in `src/qar/` arrives with a test here. New task → extend the
  smoke checks; new config field with logic → extend `test_config.py`.
