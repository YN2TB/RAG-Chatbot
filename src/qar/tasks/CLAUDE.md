# CLAUDE.md — src/qar/tasks/

Everything trainable. A task owns the model, the data and the loss; the trainer
owns the loop. This is the only folder that needs to change when the research
question changes.

```
__init__.py   imports every task module for its registration side effect
dev_toy.py    synthetic contrastive task — the harness's proof of life
```

## Contract

**In** — a `RunConfig`. That is the entire input. A task reads its hyperparameters
from `cfg` and nothing else: no globals, no CLI parsing, no environment.

**Out** — the `Task` interface in `qar/training/task.py`:

| Method | Returns |
|---|---|
| `build_model()` | `nn.Module`, on CPU (the trainer moves it) |
| `train_loader()` / `val_loader()` | `DataLoader` |
| `training_step(model, batch)` | `(loss_tensor, {str: float})` — full loss, unprefixed keys |
| `validation_step(model, batch)` | `{str: float}`, averaged by the trainer |
| `validate(model, loader, device)` | override only when the metric is not a batch mean |

**A task is invisible until it is imported in `__init__.py`.** Registration is an
import side effect; `configs/*.yaml` names the task via `task: <name>` and the
harness resolves it through the registry. Forgetting the import produces
`KeyError: unknown task '...'` with a list of what did register.

## Adding a task

```python
@register("task", "biencoder")
class BiEncoderTask(Task):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__(cfg)
```

1. Subclass `Task`, decorate with `@register("task", "<name>")`.
2. Add `from qar.tasks import <module>  # noqa: F401` to `__init__.py`.
3. Add any new knob to `src/qar/config.py` as a defaulted field — never hardcode a
   hyperparameter in the task body.
4. Extend `tests/test_smoke.py`.
5. Change nothing in `qar/training/`. If you think you need to, the abstraction is
   wrong — fix it there deliberately, not with a special case.

Contrastive loaders need `drop_last=True`: in-batch InfoNCE labels are
`arange(batch)` and assume a square score matrix. Use `cfg.data.num_workers`
(0 on Windows by default — raising it is a measured decision, not a default).

## dev_toy.py

Query/document pairs sharing a latent concept observed through noise, scored by
two MLP towers with L2-normalised outputs, trained with InfoNCE over in-batch
negatives. Shaped exactly like the real retriever: swap the synthetic pairs for
(question, review-snippet) pairs and the trainer does not change.

Fixed structure: `dim=64`, `n_concepts=512`, `noise=0.4`, 8192 train / 1024 val
items, val seeded at `cfg.seed + 1`. From config: `model.d_model` (tower hidden),
`model.dropout`, `data.batch_size` / `eval_batch_size`, `loss.temperature`, `seed`.

It exists so `tests/test_smoke.py` can assert the loss actually falls — the
standard "can it overfit?" check on a task learnable by construction. **Its
difficulty is a test fixture.** `test_model_can_overfit` asserts final loss < 0.7 ×
first and `val/recall@1` > 0.5 after 150 CPU steps; raising `noise`, cutting
`n_concepts` or shrinking the towers turns a green suite red for reasons that have
nothing to do with the harness.

Keep it fast and CPU-only. It is not a baseline and no number from it goes in
either report.

## Rules

- One task per module, named after the task.
- Everything configurable comes from `cfg`. If a value wants tuning, it is a
  config field; if it is structural, it is a constant in `__init__` with a comment
  saying why.
- Metric keys returned from the steps are unprefixed — the trainer and
  `Task.validate` add `train/` and `val/`.
- Reuse `qar.eval.metrics`; never define a metric inline here.
- No file I/O in `__init__`. Building a task must stay cheap enough that the
  registry can construct one in a test.
