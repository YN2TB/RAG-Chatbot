# CLAUDE.md

Working context for this project. Loaded automatically each session.

## What this is

A review-grounded product QA system over AmazonQA, built as **coursework for two
university courses**. One codebase, two reports, deliberately different research questions:

- **Deep Learning report** — the retriever. From-scratch dual-encoder, InfoNCE contrastive
  objective, multi-task answerability head, ablations. Question: *does this training setup
  learn a good representation?*
- **NLP report** — the system. Retrieval + generation, abstention, grounding and
  hallucination evaluation, error analysis. Question: *does it answer correctly?*

Pretrained models live in the **NLP half only**. The DL half stays from-scratch, with
pretrained encoders appearing solely as a baseline row. This is deliberate: the DL
lecturer's stance on pretrained weights is not yet confirmed, and this split means the DL
report survives either answer.

Owner is a final-year Data Science student. Prefers direct answers; says so when they want
the short version.

## Hardware and environment

- **RTX 5060 Laptop, 8 GB VRAM**, Blackwell sm_120. bf16 is native — it is the default AMP
  mode and needs no GradScaler.
- **Python 3.12** in `.venv` (3.14 is the machine default, but `tokenizers`/`faiss` lag there).
- **torch 2.13.0+cu130**, pinned to the PyTorch index in `pyproject.toml`. The default PyPI
  Windows wheel is CPU-only — never install torch without that index.
- Windows + PowerShell. `data.num_workers` defaults to 0; raising it is a measured decision.

```bash
uv run pytest                                          # 15 tests, run before any experiment
uv run python scripts/train.py configs/dev.yaml        # single run
uv run python scripts/sweep.py sweeps/temperature.yaml # ablation grid -> CSV + Markdown
```

## Dataset facts (measured, not assumed)

`train-qar.jsonl` 2.67 GB / `val-qar.jsonl` 747 MB, in the repo root, gitignored.

- 738,776 train rows / 92,183 val rows
- 124,416 / 15,592 unique products; 684,703 / 89,336 unique questions
- ~9.3 review snippets and ~3.9 answers per question (~6.8M snippets in train)
- 17 categories (Electronics 23%, Home & Kitchen 15%); 85% descriptive / 15% yes-no
- 62% answerable / 38% not

Fields: `asin`, `category`, `questionText`, `questionType`, `review_snippets[]`,
`answers[{answerText, answerType, helpful}]`, `is_answerable`, `qid`.

**The split is product-disjoint** — only 103 of 15,592 val products (0.7%) appear in train.
Evaluation therefore measures generalisation to unseen products. Question-text overlap is
7.5% but is *not* leakage: it is generic phrasing ("what's the weight limit?") asked about
different products whose reviews do not overlap.

**In-batch negatives need a dedup guard.** 738,776 rows but 684,703 unique questions. Two
rows sharing a question string in one batch make each one's positive the other's negative.
`data.dedup_questions_in_batch` controls this; the ablation against it belongs in the DL report.

## Conventions

- **Every run is a config.** No hyperparameter is ever hardcoded or passed ad hoc — it goes
  in `configs/`, and ablations override single fields via `--set key=value`. The resolved
  config is snapshotted to `runs/<name>/config.yaml`.
- **Unknown config keys raise.** A silently ignored typo means a run that looks like it
  tested something and did not.
- **Step-based, not epoch-based.** Runs must stay comparable at equal optimisation steps
  even when the data subset size changes — which it will, for the scaling curve.
- **Metrics go to `runs/<name>/metrics.jsonl`**, append-only. Curves and ablation tables are
  read from there, never transcribed by hand.
- **The trainer knows nothing task-specific.** New models subclass `Task`
  (`src/qar/training/task.py`) and register themselves; the loop does not change.
- **Overfit test first.** If a model cannot drive loss down on `dev_toy` (learnable by
  construction), the bug is the harness, not the idea.

## Layout

```
configs/          run configs; ablations inherit via `_base_:`
sweeps/           ablation grids
scripts/          train.py (single run), sweep.py (grid + results table)
src/qar/
  config.py       typed config, YAML inheritance, CLI overrides
  registry.py     name -> component lookup
  tasks/          trainable tasks (dev_toy today; retriever next)
  training/       trainer.py is the loop; task.py is the interface
  eval/           ranking + classification metrics
  utils/          seeding, device/AMP, logging
tests/            harness correctness
runs/             per-run outputs (gitignored)
```

## Gotchas already hit

- **YAML 1.1 booleans.** `off`/`no`/`yes`/`on` parse as booleans and used to corrupt string
  fields (`train.amp=off` became `"False"` — a run claiming AMP was off while training in
  bf16). Fixed in `_parse_scalar`; keep the regression tests.
- **Duplicate final eval.** When `max_steps` was a multiple of `eval_every`, the last step
  logged twice. Fixed via `_last_eval_step`.

## Status

- [x] Harness: config, seeding, AMP, trainer, checkpoint/resume, logging, sweeps, 15 tests
- [x] Split validated for leakage
- [ ] Data pipeline: pair construction, BPE tokenizer, asin-based val/test split
- [ ] BM25 baseline
- [ ] From-scratch bi-encoder + InfoNCE
- [ ] Hard negative mining, multi-task answerability head
- [ ] Ablations (temperature, batch/negatives, lambda, pooling, init, data scaling) → DL report
- [ ] Generation, grounding evaluation, abstention → NLP report

## Open questions

- Confirm with both lecturers that one codebase with two distinct reports is acceptable.
  Most departments allow it, some require a declaration, a few forbid it.
- Confirm whether the DL course permits pretrained weights. The current split makes this
  non-blocking, but it decides how much of the pretrained comparison goes in which report.

## Git

Initialized, **nothing committed yet**. `.gitignore` excludes `*.jsonl`, `runs/`, `.venv/` —
verified that the 3.4 GB corpus stays out. `git add -A` is safe.
