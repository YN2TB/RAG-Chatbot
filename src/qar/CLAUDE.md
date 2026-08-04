# CLAUDE.md — src/qar/

The package root holds the two things everything else depends on: the typed config
and the component registry. Both are the **base of the dependency graph** — they
import nothing from `qar`, and nothing in this folder may import `qar.training`,
`qar.tasks`, `qar.data` or `qar.eval`.

```
config.py    YAML + CLI overrides -> RunConfig
registry.py  name -> factory lookup
__init__.py  version only; keep it import-free and cheap
```

## config.py

**In** — one YAML path (optionally chained through `_base_:`) plus a list of
`dotted.key=value` strings.
**Out** — a `RunConfig`: nested dataclasses, every field typed and defaulted.

```python
cfg = load_config("configs/dev.yaml", ["optim.lr=1e-4", "loss.temperature=0.02"])
```

Resolution order, in this order and no other:

1. `_read_yaml` — resolve the `_base_:` chain relative to the *file*, deep-merged
   so a child overrides only the keys it names.
2. `apply_overrides` — `--set` strings patched into the raw dict, creating nested
   dicts as needed.
3. `_build` — recursive dataclass construction. **Unknown keys raise `KeyError`.**
4. `_coerce` — each scalar cast to its annotated field type, `Optional` unwrapped.

Public surface: `RunConfig`, the section dataclasses (`DataConfig`, `PrepareConfig`,
`ModelConfig`, `LossConfig`, `OptimConfig`, `TrainConfig`), `load_config`,
`apply_overrides`.

`PrepareConfig` describes the *offline* corpus build and is read only by
`scripts/prepare_data.py`. It lives in `RunConfig` anyway so it is typed, defaulted
and overridable like everything else — but a training run's `config.yaml` snapshot
proves nothing about how its data was built. `data/processed/manifest.json` is the
record that does.
`cfg.run_dir` is derived (`out_dir / name`), never stored. `cfg.save(path)` writes
the resolved snapshot.

### Rules

- **A new hyperparameter is a new dataclass field with a default.** Not a dict
  entry, not a kwarg, not an environment variable. The default *is* the
  documentation — `configs/*.yaml` only needs to state what it changes.
- **Never add a free-form `extra: dict` escape hatch.** The unknown-key `KeyError`
  is the whole point: a silently ignored typo in an ablation config produces a run
  that looks like it tested something and did not.
- **Keep this module torch-free.** It currently imports only `dataclasses`,
  `pathlib`, `typing` and `yaml`. Config loading must stay importable and instant
  without CUDA.
- No I/O beyond reading the given YAML and writing the snapshot. No logging, no
  `sys.argv`, no `os.environ`.
- `save` → `load_config` must round-trip to an identical dict. Guarded by
  `tests/test_config.py::test_roundtrip_through_yaml`.

### Gotcha: YAML 1.1 booleans

`off`, `no`, `yes`, `on`, `y`, `n` parse as booleans. That once turned
`train.amp=off` into the string `"False"` — a run that reported AMP disabled while
training in bf16. `_parse_scalar` keeps only the canonical `true`/`false`
spellings boolean and lets the rest survive as strings, so the target field's own
type still coerces `train.compile=off` to `False`. Do not "simplify" it back to a
plain `yaml.safe_load`; two regression tests cover this.

## registry.py

**In** — `kind` and `name` strings.
**Out** — whatever the registered factory returns.

```python
@register("task", "biencoder")
class BiEncoderTask(Task): ...

task = build("task", cfg.task, cfg)   # KeyError lists what *is* registered
available("task")                     # sorted names
```

Registering the same `(kind, name)` twice raises — a duplicate name means one of
two ablations silently ran the wrong component.

### Rules

- **Registration happens by import side effect**, collected in
  `qar/tasks/__init__.py`. Nothing here imports a concrete component.
- The harness resolves components by name from the config only. If a module
  reaches for a class directly, the config stops describing the run.
- Adding a `kind` costs nothing (`_REGISTRIES` is created on demand); use one for
  models, encoders or retrievers when the retriever lands.

Kinds in use today: `task` (populated by importing `qar.tasks`) and `selector`
(populated by importing `qar.data`).
