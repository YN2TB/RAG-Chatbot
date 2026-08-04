# CLAUDE.md — src/qar/models/

Architectures. Randomly initialised, trained only by the objective in the task.

```
encoder.py    TextEncoder — pre-norm transformer, masked mean/cls pooling
biencoder.py  BiEncoder — two towers + optional answerability head
```

## Contract

**In** — a `RunConfig` and the tokenizer's `pad_id`. A model reads `cfg.model.*`
(and `cfg.loss.answerable_weight`, to decide whether the head exists at all). It
never reads a file, a dataset, or a path.

**Out** — `nn.Module`. `BiEncoder.forward(batch)` returns
`(query, doc, answerable_logits | None)`, with query and doc **L2-normalised** so
the dot product in the loss is a cosine and `loss.temperature` alone controls how
sharp the softmax is.

Models register under kind `model`; `model.name` in a config selects one. Nothing
here imports a task — the dependency runs task → model, never back.

## Choices that are choices, not conventions

**Pre-norm blocks.** A randomly initialised post-norm stack needs careful warmup to
train at all. Pre-norm keeps the residual path clean and makes a from-scratch run
reproducible on a laptop instead of a coin flip. `optim.warmup_ratio` is still on,
but as insurance rather than a load-bearing trick.

**GPT-2 style init (normal, std 0.02).** PyTorch's default `nn.Linear` init is tuned
for shallow networks; on a deep stack it gives activations large enough to stall the
first few hundred steps. `padding_idx` is re-zeroed afterwards because `normal_`
overwrites what the embedding constructor zeroed.

**Masked mean pooling, denominator clamped.** Padding is excluded rather than
averaged in. A batch pads to its longest row, so if padding leaked into the pooled
vector a short question's embedding would depend on which other questions shared its
batch — a silent, batch-order-dependent bug. `test_padding_cannot_change_the_representation`
locks it.

**`model.share_encoder` is an ablation, not a default.** Two towers let each side
specialise; questions and review prose are genuinely different registers. One tower
halves the parameters and gives every gradient twice the data, which on 700k pairs
from scratch may well win. Nobody knows yet — that is why it is a config field.

**The answerability head reads the pooled question only.** It could be given the
positive document too, but that positive was itself inferred by distant supervision
(mean overlap 0.26), so a head that sees it learns to predict *the selector* rather
than answerability. Question-only keeps the auxiliary task honest; its job is to
regularise the query tower.

**The head is not built when `loss.answerable_weight == 0`.** An unused head would
still be checkpointed and would make a zero-weight run structurally different from
the runs it is meant to be the control for.

## Rules

- No file or dataset access. A model takes tensors and returns tensors.
- No task-specific knowledge: the loss lives in the task, not here.
- `mask` is True for **real** tokens throughout this project. `nn.MultiheadAttention`
  wants the opposite, so it is inverted exactly once, inside `TextEncoder.forward`.
  Do not invert it at a call site.
- A sequence longer than `model.max_len` raises rather than truncating silently —
  truncation belongs to the collator, where it is configured and visible.
- New architectures register under kind `model` and get shape tests in
  `tests/test_models.py`, including the padding-invariance check.

## Size

Defaults (`d_model=384`, 6 layers, `d_ff=1536`, vocab 32k) give ~23M parameters per
tower, ~46M with two. With AdamW state that is roughly 0.75 GB before activations —
comfortable inside 8 GB, which is what leaves room for the batch size that actually
matters to InfoNCE.
