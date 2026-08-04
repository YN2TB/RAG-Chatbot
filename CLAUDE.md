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
uv sync --extra dev                                    # once; pytest lives in the dev extra
uv run pytest                                          # 103 tests, run before any experiment

uv run python scripts/prepare_data.py configs/base.yaml      # raw corpus -> data/processed
uv run python scripts/build_idf.py configs/base.yaml         # corpus IDF for bm25_global
uv run python scripts/evaluate_retrieval.py configs/base.yaml # baseline table
uv run python scripts/train.py configs/retriever.yaml        # the DL-report run
uv run python scripts/sweep.py sweeps/temperature.yaml       # ablation grid -> CSV + Markdown
```

The first three are offline and run once; `prepare_data` takes ~13 min, `build_idf`
~3 min.

`uv run` installs the default dependencies but not extras, so a fresh environment
needs the `uv sync --extra dev` above before `pytest` exists. It does not remove the
extra afterwards.

## Dataset facts (measured, not assumed)

`amazonqa_train.jsonl` 2.67 GB / `amazonqa_validation.jsonl` 747 MB, in the repo root,
gitignored. (`data.ipynb` downloads them from the S3 bucket under those names; the
upstream files are called `train-qar.jsonl` / `val-qar.jsonl`.)

- 738,776 train rows / 92,183 val rows
- 124,416 / 15,592 unique products; 684,703 / 89,336 unique questions
- ~9.3 review snippets and ~3.9 answers per question (~6.8M snippets in train)
- 17 categories (Electronics 23%, Home & Kitchen 15%); 85% descriptive / 15% yes-no
- 62% answerable / 38% not

Fields: `asin`, `category`, `questionText`, `questionType`, `review_snippets[]`,
`answers[{answerText, answerType, helpful}]`, `is_answerable`, `qid`.

**Four more fields are the dataset authors' own baselines**, and the pipeline drops
them on purpose: `top_sentences_IR` (~9.8 sentences from an IR baseline),
`top_review_wilson`, `top_review_helpful` (1 review each), `random_sentence`. Keeping
them out of the training pipeline avoids leaking a competing system's output into the
positives.

**They are not free rows for the retrieval table.** Measured over 3,000 validation
rows, only 7.2% of `top_sentences_IR` sentences appear in `review_snippets` at all,
and 54% of rows share none — it is a *different extraction* from the review text, not
a reranking of our pool. Scoring it as a ranking over our candidates would be
meaningless. Where it is genuinely useful is the **NLP report**: as an alternative
evidence set handed to the generator, answer quality given the authors' IR selection
versus given ours is a fair comparison.

**There is no snippet-level relevance label.** A row knows its answers but not which
snippet supports them, so every training positive is inferred by distant supervision
(`prepare.selector`). That inference is the weakest link in the whole system and
belongs in the report as a limitation, not a footnote.

**The split is product-disjoint** — only 103 of 15,592 val products (0.7%) appear in train.
Evaluation therefore measures generalisation to unseen products. Question-text overlap is
7.5% but is *not* leakage: it is generic phrasing ("what's the weight limit?") asked about
different products whose reviews do not overlap.

**In-batch negatives need a dedup guard.** 738,776 rows but 684,703 unique questions. Two
rows sharing a question string in one batch make each one's positive the other's negative.
`data.dedup_questions_in_batch` controls this; the ablation against it belongs in the DL report.

## Prepared corpus (data/processed/, built 2026-08-04)

`scripts/prepare_data.py configs/base.yaml`, 13.5 min, 2.2 GB.

| | train | val | test |
|---|---|---|---|
| kept | 704,201 | 43,801 | 43,674 |
| dropped: no trustworthy positive | 33,013 (4.5%) | 2,214 | 2,348 |
| unique products | 123,616 | 7,739 | 7,757 |
| answerable | 62.8% | 65.0% | 65.1% |
| mean positive score | 0.262 | 0.260 | 0.260 |

Rows read plus 1,708 unparseable rows reconcile exactly to 738,776 and 92,183.
`asin_overlap_val_test` is **0**; the 49 + 54 = 103 products shared with train are the
overlap the upstream split already had, not one this pipeline introduced.

**Mean positive score 0.26 is the project's headline limitation** — the inferred
positive shares about a quarter of its tokens with the reference answer. The retriever
is trained towards a noisy target and the report must say so.

Loading runs at ~5,400 pairs/s single-threaded, so `num_workers=0` is not a bottleneck.

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
- **A `prepare.*` change is a new corpus, not a new version.** Point
  `data.processed_dir` somewhere new; the two are an ablation pair. `data/processed/manifest.json`
  records which settings produced the corpus a run trained on.

## Layout

Each code folder carries its own `CLAUDE.md` stating that folder's inputs, outputs
and boundary rules. Read it before changing anything inside.

```
configs/          run configs; ablations inherit via `_base_:`
sweeps/           ablation grids
scripts/          prepare_data, build_idf, evaluate_retrieval (offline); train, sweep
src/qar/
  config.py       typed config, YAML inheritance, CLI overrides
  registry.py     name -> component lookup (kinds: task, model, selector, retriever)
  data/           corpus preparation, BPE, splitting, dataset, collator, dedup sampler
  models/         TextEncoder (from scratch) and the BiEncoder built on it
  retrieval/      untrained baselines + the within-product evaluation harness
  tasks/          trainable tasks (dev_toy, retriever)
  training/       trainer.py is the loop; task.py is the interface
  eval/           ranking + classification metrics
  utils/          seeding, device/AMP, logging
tests/            harness correctness
data/processed/   prepared corpus + tokenizer + IDF + manifest (gitignored)
runs/             per-run outputs and _baselines/ (gitignored)
```

## Gotchas already hit

- **YAML 1.1 booleans.** `off`/`no`/`yes`/`on` parse as booleans and used to corrupt string
  fields (`train.amp=off` became `"False"` — a run claiming AMP was off while training in
  bf16). Fixed in `_parse_scalar`; keep the regression tests.
- **Duplicate final eval.** When `max_steps` was a multiple of `eval_every`, the last step
  logged twice. Fixed via `_last_eval_step`.
- **`train/*` inflated by `grad_accum`.** The running totals were summed once per
  micro-batch but divided by the number of optimisation steps, so every `train/*` scalar
  read `grad_accum`× too high — invisible at the default `grad_accum=1`, and `val/*` was
  never affected. Fixed in `trainer.py`; keep the regression test. Any accumulated run
  logged before this needs its curve regenerated.

## Status

- [x] Harness: config, seeding, AMP, trainer, checkpoint/resume, logging, sweeps, 103 tests
- [x] Split validated for leakage
- [x] Data pipeline: distant-supervised pair construction, byte-level BPE, asin-hashed
      val/test split, offset-indexed dataset + padding collator, manifest with a
      leakage report. `scripts/prepare_data.py`
- [x] Lexical baselines: random / first / overlap / bm25 / bm25_global / bm25_noidf,
      within-product ranking on the real pool. `scripts/evaluate_retrieval.py`
- [x] From-scratch bi-encoder + InfoNCE: pre-norm encoder, two towers, in-batch
      negatives, de-duplicating batch sampler, optional answerability head.
      `configs/retriever.yaml`. **Never trained on a GPU yet** — see below.
- [x] Hard negatives from the same product (`loss.hard_negatives`), row-private, with
      unfillable slots masked out of the softmax. No mining pass needed — the pool is
      already in every record.
- [ ] A real training run (blocked: no CUDA in the dev shell)
- [ ] Model-based hard-negative mining (round-2 negatives from the retriever itself)
- [ ] `top_sentences_IR` as an *evidence-set* comparison for the NLP report (it is
      not a reranking of our pool — see the dataset section)
- [ ] Ablations (temperature, batch/negatives, lambda, pooling, init, data scaling) → DL report
- [ ] Generation, grounding evaluation, abstention → NLP report

## Baselines to beat (val, within-product, mean pool 9.32)

| retriever | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| random (the floor) | 0.1272 | 0.3483 | 0.5458 | 0.3237 |
| first (position prior) | 0.1294 | 0.3487 | 0.5471 | 0.3252 |
| bm25 (pool-local IDF) | 0.1768 | 0.4236 | 0.6210 | 0.3759 |
| bm25_global (corpus IDF) | 0.1861 | 0.4417 | 0.6406 | 0.3872 |
| bm25_noidf | 0.1940 | 0.4561 | 0.6532 | 0.3961 |
| **overlap** (token F1) | **0.2145** | **0.4906** | **0.6868** | **0.4181** |

`first ≈ random`, so the snippet pool carries no ordering — nothing here is quietly
benefiting from position. **BM25 loses to plain token overlap**, and the rows
decompose why: corpus IDF over pool-local IDF is worth +0.009, removing IDF entirely
another +0.008, and the F1 form over summed saturated tf a further +0.021. Within one
product the discriminative terms are the *common* ones — the product's own features —
and IDF suppresses exactly those. Details in `src/qar/retrieval/CLAUDE.md`.

**`val/recall@1` printed during training is a different number** — it ranks within
`eval_batch_size` candidates from *different* products, a far easier problem. Never
put it in the same table as the above.

## Blocked

`torch.cuda.is_available()` is False in the dev shell and `nvidia-smi` is
unreachable, so the retriever has only ever run on CPU (0.32 steps/s at batch 32,
45.97M parameters). The wiring is verified end to end on the real corpus; the
training itself is not. A real run needs a shell that can see the 5060.

## Open questions

- Confirm with both lecturers that one codebase with two distinct reports is acceptable.
  Most departments allow it, some require a declaration, a few forbid it.
- Confirm whether the DL course permits pretrained weights. The current split makes this
  non-blocking, but it decides how much of the pretrained comparison goes in which report.

## Git

Initialized, **nothing committed yet**. `.gitignore` excludes `*.jsonl`, `runs/`, `.venv/` —
verified that the 3.4 GB corpus stays out. `git add -A` is safe.
