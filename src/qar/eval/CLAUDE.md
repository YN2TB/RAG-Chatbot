# CLAUDE.md — src/qar/eval/

Metric definitions. Pure functions, no state, no config, no I/O.

```
metrics.py   ranking_metrics (retriever), binary_metrics (answerability head)
```

## Contract

**In** — torch tensors only. Never a model, never a config, never a path.

**Out** — `dict[str, float]` with **plain Python floats and unprefixed keys**. The
caller adds `train/` or `val/`; `JsonlLogger` serialises the floats without any
conversion step. Returning tensors here would leak device memory into the log
record.

This folder is a **leaf**: it imports `torch` and nothing from `qar`. It must never
import `qar.training` or `qar.tasks`.

## ranking_metrics(scores, target, ks=(1, 5, 10))

`scores` is `(queries × candidates)`; `target[i]` is the column index of the
correct candidate for query `i` — for in-batch contrastive training that is just
`arange(batch)`. Raises `ValueError` if `scores` is not 2-D.

Returns `mrr`, `recall@k` for each `k <= n_candidates`, and `mean_rank`. Ranks are
0-indexed internally; `mean_rank` is reported 1-indexed, so a perfect retriever
gives `mean_rank == 1.0`, not 0.

`recall@k` is silently omitted when `k` exceeds the number of candidates — with
`eval_batch_size=8` there is no `recall@10` column. Do not build a report table
that assumes a fixed key set; read the keys that are there.

**These numbers are batch-local.** In-batch recall@1 over 128 candidates is not
recall@1 over a 6.8M-snippet index, and the two must never share a column in the
DL report. Full-index evaluation means overriding `Task.validate`.

## binary_metrics(logits, labels, threshold=0.0)

For the multi-task answerability head. `logits` are **raw, pre-sigmoid** — the
default `threshold=0.0` is the logit-space equivalent of p=0.5. Returns `acc`,
`precision`, `recall`, `f1`, all zero-division-safe (0.0 rather than NaN).

The corpus is 62% answerable / 38% not, so a constant "yes" predictor scores 0.62
accuracy. **Report F1; distrust accuracy.**

## Rules

- Every function is `@torch.no_grad()` and stateless. A metric that needs to
  accumulate across batches is the task's job, not this folder's.
- Plain floats out. No tensors, no numpy scalars.
- Unprefixed keys, and stable ones — a renamed key silently breaks
  `train.monitor`, `sweeps/*.yaml` `monitor:`, and any curve already plotted from
  an old `metrics.jsonl`.
- New metrics land here and get exported in `__init__.py`, even one-liners. A
  metric defined inline in a task cannot be tested or reused, and the two reports
  need the same definitions.
- Generation and grounding metrics for the NLP report belong here too, in their
  own module (`generation.py`), following the same tensors-in/floats-out shape.
