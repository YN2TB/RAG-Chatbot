# CLAUDE.md

Working context for this project. Loaded automatically each session.

## Before you write code, or anything else

**1. Read `.claude/` first.** `.claude/agents/*.md` defines six specialist roles
(`qar-explorer`, `qar-planner`, `qar-data-engineer`, `qar-ml-scientist`,
`qar-validator`, `qar-reviewer`) and `.claude/settings.json` defines what may run
without asking and what must never be read. Read them before starting work, not
after — they carry the project's standing constraints, and the `deny` rules exist
because a single `Read` of a 2.5 GB corpus file ends the session.

`.claude/` is checked in, so it is there in every clone — there is no excuse for
skipping it. It is also shared: a permission rule you add applies to everyone
working on this repo, so `allow` entries stay read-only and `deny` keeps the
multi-gigabyte corpus files out of any agent's context.

Then read the `CLAUDE.md` of every folder you are about to touch. Those files are
the specification — they state each folder's inputs, outputs and boundary rules,
and a change that violates one is a defect even when the tests pass.

**2. Record every change in `CHANGELOG.md`**, in the same commit as the change.
Not a restatement of the diff — git already has that. Record what was *learned*:
a measurement that came out against expectation, an assumption shown false, a
default that moved and why, and what the change invalidates. A measured number
goes in with the split it came from; `recall@1 0.2145` means nothing without
"val, within-product, mean pool 9.32". `CHANGELOG.md` explains the entry format.

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

- **GPU changed 2026-08-08.** Was an RTX 5060 Laptop (8 GiB, Blackwell sm_120); now an
  RTX 3070 Ti (reported 16 GiB, Ampere sm_86). **Every memory and throughput figure in
  this repo was measured on the 8 GiB card and needs re-probing** — see the migration
  section at the top. bf16 stays correct on Ampere and still needs no GradScaler.
- **Python 3.12** in `.venv` (3.14 was the old machine's default, but `tokenizers`/`faiss`
  lag there; check what the new one ships before assuming).
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

`train-qar.jsonl` 2.67 GB / `val-qar.jsonl` 747 MB / `test-qar_all.jsonl` 751 MB, in
the repo root, gitignored. `data.ipynb` downloads all three from the S3 bucket under
their upstream names, which is what `configs/base.yaml` points at.

- 738,776 train rows / 92,183 val rows / 92,726 test rows
- 124,416 / 15,592 / 15,599 unique products; 684,703 / 89,336 unique train/val questions
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

**The corpus ships its own test split** (`test-qar_all.jsonl`), and it is
product-disjoint from both train (116 shared products, 0.74%) and validation (15,
0.10%) — the same profile the train/validation pair already had. The pipeline used to
carve test out of validation by hashed asin, which cost half the validation rows for
no gain; `data.test_path` now selects between the two regimes and the manifest
records which one ran. Measured 2026-08-06.

**The splits are product-disjoint** — only 103 of 15,592 val products (0.7%) appear in
train. Evaluation therefore measures generalisation to unseen products. Question-text
overlap is 7.5% (train/val) and 7.8% (train/test) but is *not* leakage: it is generic
phrasing ("what's the weight limit?") asked about different products whose reviews do
not overlap.

**In-batch negatives need a dedup guard.** 738,776 rows but 684,703 unique questions. Two
rows sharing a question string in one batch make each one's positive the other's negative.
`data.dedup_questions_in_batch` controls this; the ablation against it belongs in the DL report.

## Prepared corpus (data/processed/, rebuilt 2026-08-06)

`scripts/prepare_data.py configs/base.yaml`, 10.8 min. Built under
`test_source: upstream_file` — validation whole, test from `test-qar_all.jsonl`.

| | train | val | test |
|---|---|---|---|
| kept | 704,201 | 87,475 | 88,182 |
| dropped: no trustworthy positive | 33,013 (4.5%) | 4,562 | 4,429 |
| unique products | 123,616 | 15,496 | 15,509 |
| unique questions | 638,306 | 83,667 | 84,287 |
| answerable | 62.8% | 65.1% | 65.3% |
| mean positive score | 0.262 | 0.260 | 0.261 |

Every split reconciles exactly against the raw row counts: 737,214 + 1,562
malformed = 738,776; 92,037 + 146 = 92,183; 92,611 + 115 = 92,726.

Leakage is inherited, never introduced: 103 products shared train↔val, 116
train↔test, 15 val↔test — identical to what the raw files already had.

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
- **Equal steps is not enough: `max_steps` must also match.** The cosine schedule is
  defined over `train.max_steps`, so step 12000 of a 12k run has annealed the LR to
  zero while step 12000 of a 20k run is still mid-decay. Two checkpoints at the same
  step number from different `max_steps` are *not* a controlled comparison. Every
  sweep pins `max_steps` in its `fixed:` block for exactly this reason; comparisons
  *across* grids with different lengths need a matched control run.
- **Metrics go to `runs/<name>/metrics.jsonl`**, append-only. Curves and ablation tables are
  read from there, never transcribed by hand — and read via
  `qar.utils.logging.read_series`, which resolves the duplicate steps a resumed run
  leaves behind. A bare `json.loads` loop over the file is not safe on a resumed run.
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
- **Resuming duplicates steps in `metrics.jsonl`.** The log is append-only, which is
  what lets a resumed run keep the curve from before the crash — but resuming rewinds
  to the checkpoint's step while the interrupted run's later records stay in the file.
  Those steps then carry two values from two trajectories. Read curves with
  `qar.utils.logging.read_series`, which keeps the later record per step; reading the
  raw file gives a curve that jumps backwards and a "best" that may come from a
  trajectory that was thrown away. `Trainer.maybe_resume` now writes an
  `{"event": "resume"}` marker so the rewind is visible.
  **`runs/retriever_b128_hn2` has duplicates at steps 2100/2200/2300 and no marker** —
  it was resumed before the marker existed. `read_series` still resolves it.
- **`train/*` inflated by `grad_accum`.** The running totals were summed once per
  micro-batch but divided by the number of optimisation steps, so every `train/*` scalar
  read `grad_accum`× too high — invisible at the default `grad_accum=1`, and `val/*` was
  never affected. Fixed in `trainer.py`; keep the regression test. Any accumulated run
  logged before this needs its curve regenerated.

## Status

- [x] Harness: config, seeding, AMP, trainer, checkpoint/resume, logging, sweeps, 124 tests
- [x] Split validated for leakage
- [x] Data pipeline: distant-supervised pair construction, byte-level BPE, asin-hashed
      val/test split, offset-indexed dataset + padding collator, manifest with a
      leakage report. `scripts/prepare_data.py`
- [x] Lexical baselines: random / first / overlap / bm25 / bm25_global / bm25_noidf,
      within-product ranking on the real pool. `scripts/evaluate_retrieval.py`
- [x] From-scratch bi-encoder + InfoNCE: pre-norm encoder, two towers, in-batch
      negatives, de-duplicating batch sampler, optional answerability head.
      `configs/retriever.yaml`.
- [x] Hard negatives from the same product (`loss.hard_negatives`), row-private, with
      unfillable slots masked out of the softmax. No mining pass needed — the pool is
      already in every record.
- [x] `dense` retriever: scores a checkpoint through the *same* within-product path
      as the baselines, so a run finally produces a number comparable to `overlap`.
      `retrieval.checkpoint` selects the checkpoint; architecture comes from its own
      snapshot, not from the evaluating config
- [x] A real training run — `runs/retriever_b128`, 20k steps at batch 128, 64.7 min,
      5.35 steps/s, 3,184 MiB of 8,151. Loss 4.85 (chance) → 1.43, train acc 0.62,
      in-batch `val/recall@1` → 0.4998. **Still rising at 20k steps**, so the run is
      cut short by `max_steps`, not converged. Batch 256 also fits (5,730 MiB) but
      runs at 1.6 steps/s → 3.5 h, so more negatives is a planned ablation
Everything still open is in **Upcoming objectives** below, in the order it should
happen.

## READ FIRST — moving to a new machine (written 2026-08-08)

The project is moving from an **RTX 5060 Laptop, 8 GiB, Blackwell sm_120** to an
**RTX 3070 Ti, reported 16 GiB, Ampere sm_86**. Everything below assumes you are
starting on the new machine with a fresh clone.

### 1. What did NOT travel

`/runs/` is gitignored apart from `_baselines/`, and the corpus is gitignored
entirely. So the new clone has **no raw corpus, no processed corpus, and no
checkpoints** — only the code, the docs and the results tables.

Rebuild, in this order (~30 min plus the download):

```bash
uv venv --python 3.12 && uv sync --extra dev
# then run data.ipynb to fetch the three raw .jsonl files (3.4 GB total)
uv run pytest                                                 # 130 tests, all CPU
uv run python scripts/prepare_data.py configs/base.yaml       # ~11 min
uv run python scripts/build_idf.py configs/base.yaml          # ~3 min
```

**Do not re-run the lexical baselines to "check".** They are committed in
`runs/_baselines/val.json` and cost nothing to trust; regenerating them is only
correct if `prepare.*` changed, in which case the corpus is a different one and
every trained number is invalidated too.

**All trained checkpoints are gone.** Any run you want to *evaluate* again has to be
*re-trained* first. Everything already measured is recorded below and in
`CHANGELOG.md`, so nothing needs re-running just to recover a number.

### 2. Every hardware figure in this file is 8 GiB / Blackwell — re-probe

These were the binding constraints on the old card and are now **wrong**:

| Recorded on the 5060 (8 GiB) | Status on the 3070 Ti |
|---|---|
| `hard_negatives=4` overcommits (9,757 MiB) and thrashes at 0.16 steps/s | Probably fits — **re-test before believing it is ruled out** |
| batch 256 fits at 5,730 MiB, 1.6 steps/s | Should be comfortable; larger batches may now be reachable |
| hn=2 runs at 1.75 steps/s, 6,556 MiB | Ampere is a different architecture; expect different throughput |
| `retrieval.encode_batch=256`, ~3.5 min per full-val dense evaluation | Can likely go higher |

Re-probe before committing to a long run — 30 steps is enough:

```bash
uv run python scripts/train.py configs/retriever.yaml \
    --set name=probe data.batch_size=256 loss.hard_negatives=4 \
          train.max_steps=30 train.log_every=10 train.eval_every=0 train.save_every=0
```

Then read `train/mem/alloc_mib` and `train/steps_per_s` from
`runs/probe/metrics.jsonl`. **Allocation above physical VRAM does not raise on
Windows** — it spills to system RAM and collapses throughput, which is how
`hard_negatives=4` wasted an hour before. A plausible-looking run at 0.16 steps/s is
that failure, not a slow model.

Confirm the reported 16 GiB with the probe rather than trusting the spec: every
batch-size and hard-negative decision in this project is memory-bound, and the whole
`sweeps/` design was shaped by an 8 GiB ceiling that may no longer apply.

**bf16 is fine on Ampere** (sm_80 and up), so `train.amp=bf16` stays correct and
still needs no GradScaler. `torch 2.13.0+cu130` supports Ampere; the cu130 pin in
`pyproject.toml` was for Blackwell but does no harm here — keep it rather than
risking the CPU-only default PyPI wheel.

### 3. What was running when the old machine stopped

**Nothing.** `runs/retriever_b128_hn1` completed before the move and its result is
committed in `runs/_baselines/val_hn1.json`, so the negatives curve is finished and
needs no re-running. The command below is kept only as the pattern for the next
single-cell run:

```bash
uv run python scripts/train.py configs/retriever.yaml \
    --set name=retriever_b128_hn1 data.batch_size=128 loss.hard_negatives=1 \
          train.max_steps=20000
uv run python scripts/evaluate_retrieval.py configs/retriever.yaml \
    --set retrieval.baselines=[dense] \
          retrieval.checkpoint=runs/retriever_b128_hn1/checkpoints/best.pt \
          retrieval.out_name=val_hn1
```

`max_steps` must match whatever the run is being compared against — the cosine
schedule is defined over it, so a 12k cell is not comparable with a 20k one. See the
convention under Conventions.

**The first thing worth running on the new card** is `hard_negatives=4`, which the
8 GiB ceiling made impossible. Expect ~0.197-0.201 from the saturation trend, i.e.
confirmation rather than a win. If it lands far above that, the saturation reading
is wrong and more negatives are back on the table.

## Upcoming objectives

Ordered by value, not by ease. Costs are measured on the 5060 where a comparable
run exists. Nothing below needs new infrastructure unless it says so.

### 1. ~~Hard negatives~~ — DONE 2026-08-08, curve complete

0.1707 → 0.1871 → 0.1940 for n = 0, 1, 2, all at 20k steps. **The effect saturates:**
+0.0164 then +0.0069, while cost rises 5.35 → 4.52 → 1.75 steps/s. Full table in the
results section above.

**What is left of it:** `hard_negatives=4` was impossible on the 8 GiB card (9,757
MiB, no exception raised, 0.16 steps/s) and may fit on the new one. Worth one run to
confirm the saturation, not as a route to beating `overlap` — the extrapolation lands
near 0.197–0.201, still short of 0.2149.

### 2. ~~Train to convergence~~ — ANSWERED 2026-08-07: 12,000 steps

**With hard negatives the run converges by step 12,000.** `retriever_b128_hn2`
peaked in-batch at 12000 and stayed flat to 20000 (0.281–0.289); scored
within-product, step 12000 gives 0.1940 and step 20000 gives 0.1941. The last 8,000
steps — ~1.3 h — bought 0.0001 recall@1.

`retriever_b128` (hn=0) was still rising at 20000, so the two configurations do not
converge at the same rate: a more informative gradient per step gets there sooner.

**Use `train.max_steps=12000` as the standard ablation length**, which cuts ~40% off
every cell. Three caveats worth stating in the report:

- The earliest checkpoint that survived `keep_last=2` was step 12000, so convergence
  may be earlier still and this is an upper bound, not the point itself.
- It is measured for hn=2 only. hn=0 was still rising at 20000, so the two
  configurations demonstrably do not converge at the same rate and an hn=0 arm may
  need longer to be fairly represented.
- **A 12k schedule costs ~0.0016 in absolute terms.** `retriever_hn2_12k` reaches
  0.1924 where the 20k run reaches 0.1940/0.1941. The shorter cosine anneals faster
  and lands slightly lower. That is fine for *relative* comparisons inside a grid,
  where every cell shares the schedule — but a headline number for the report should
  come from a 20k run, not a 12k ablation cell.

**Still open:** batch 256 fits (5,730 MiB) at 1.6 steps/s. Batch size is the number
of negatives, so this is a real axis, not a speed knob — budget it deliberately.

### 3. The DL ablation grid

Six grids exist in `sweeps/`, 19 cells total: `negatives`, `temperature`, `pooling`,
`data_scaling`, `multitask`, `dedup`. Run `negatives` first — every other grid pins
`hard_negatives=2` in its `fixed:` block and so presumes that result.

Both blockers are cleared: `score_batch` batches `dense` evaluation, and `sweep.py`'s
`evaluate:` block scores each cell within-product so the table ranks on the metric
that matters. **Read `within_recall@1`, never the logged `val/recall@1`** — twice
today the in-batch number would have supported a wrong conclusion.

### 4. ~~The answerability head~~ — DONE 2026-08-08, grid complete

Three cells, all hn=2, all `max_steps=12000`, so the comparison is controlled.
Retrieval is within-product on whole val; head metrics are from `metrics.jsonl`.

| `answerable_weight` | recall@1 | mrr | `ans_recall` | `ans_acc` | `ans_f1` |
|---|---|---|---|---|---|
| 0.0 (`retriever_hn2_12k`, control) | 0.1924 | 0.3950 | — | — | — |
| 0.1 (`retriever_hn2_mt01`) | 0.1934 | 0.3962 | 0.901 | 0.700 | 0.796 |
| 0.5 (`retriever_hn2_mt05`) | 0.1939 | 0.3964 | 0.887 | 0.708 | 0.798 |
| *untrained head, for reference* | — | — | 0.996 | 0.655 | 0.790 |

**The head learns.** It stops being a constant function — `ans_recall` falls from
0.996 to 0.887, and accuracy reaches 0.708 against the 0.651 class prior. F1 moves
barely at all because it is dominated by the majority class, which is exactly why
`ans_recall` is the honest first read and `ans_acc` alone would mislead.

**It does not compete with retrieval, even at 0.5.** Recall@1 rises monotonically
across the three cells (0.1924 → 0.1934 → 0.1939). The total spread is 0.0015, small
enough that the honest claim is *no measurable cost*, not *a gain* — but the
direction is consistent and nothing suggests a trade-off yet. A weight above 0.5 is
untested.

**The control mattered, and this is why.** Before it existed, the head was compared
against step 12000 of the 20k run and appeared to *cost* 0.0006. Against the matched
12k control it *gains* 0.0010. The confound was large enough to flip the sign of the
effect — see the `max_steps` convention above.

### 5. First numbers on test

Nothing has touched `test.jsonl` yet — correctly. Select every hyperparameter on val,
then report once on test, and say which split produced which number. 88,182 rows are
sitting there for exactly one measurement.

### 6. Model-based hard-negative mining

Round two: train, retrieve, take highly-ranked non-positives as negatives. Only worth
starting once (1) shows that row-private hard negatives help.

### 7. The NLP half

Generation over retrieved snippets, abstention driven by `is_answerable`, grounding
and hallucination evaluation, error analysis by category and question type. Pretrained
models are allowed here and nowhere else.

`top_sentences_IR` belongs in this half as an alternative **evidence set** for the
generator — it is a different extraction, not a reranking of our pool, so it can never
be a row in the retrieval table.

## Baselines to beat (val, within-product, 87,475 rows, mean pool 9.33)

Regenerated 2026-08-06 on the full validation split. The earlier table was measured
over 43,801 rows — half of validation, back when test was carved out of it.
**Every value moved by less than 0.005 and the ordering is identical**, which is
worth knowing: it says the baseline ranking was never an artefact of the smaller
sample, and it is the evidence that doubling validation changed the precision of
these numbers rather than their meaning.

| retriever | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| first (position prior) | 0.1282 | 0.3458 | 0.5455 | 0.3239 |
| random (the floor) | 0.1305 | 0.3514 | 0.5486 | 0.3262 |
| **dense** (trained, 20k steps, in-batch negatives only) | **0.1707** | 0.4232 | 0.6276 | 0.3728 |
| bm25 (pool-local IDF) | 0.1771 | 0.4244 | 0.6215 | 0.3763 |
| bm25_global (corpus IDF) | 0.1870 | 0.4427 | 0.6412 | 0.3878 |
| bm25_noidf | 0.1942 | 0.4564 | 0.6535 | 0.3961 |
| **overlap** (token F1) | **0.2149** | **0.4903** | **0.6855** | **0.4184** |

## Hard negatives: the DL report's central experiment (2026-08-07)

`runs/retriever_b128_hn2` — identical to `retriever_b128` except
`loss.hard_negatives=2`. Both 20k steps, batch 128. Scored within-product on whole
validation, 87,475 rows:

| model | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| `dense` hn=0 (in-batch negatives only) | 0.1707 | 0.4232 | 0.6276 | 0.3728 |
| **`dense` hn=2 (row-private hard negatives)** | **0.1940** | **0.4591** | **0.6557** | **0.3967** |
| — for reference, `bm25_noidf` | 0.1942 | 0.4564 | 0.6535 | 0.3961 |
| — for reference, `overlap` | 0.2149 | 0.4903 | 0.6855 | 0.4184 |

**+0.0233 recall@1, a 13.6% relative gain, from a single config field.** The
diagnosis was right: the model trained only on other-product negatives had learned
topic, and showing it same-product negatives taught it relevance.

### The effect saturates (completed 2026-08-08)

All three cells at batch 128, `max_steps=20000`, so the schedules match:

| `hard_negatives` | recall@1 | Δ vs previous | mrr | steps/s | VRAM |
|---|---|---|---|---|---|
| 0 | 0.1707 | — | 0.3728 | 5.35 | 3,184 MiB |
| 1 | 0.1871 | **+0.0164** | 0.3907 | 4.52 | 4,830 MiB |
| 2 | 0.1940 | **+0.0069** | 0.3967 | 1.75 | 6,556 MiB |

**The gain roughly halves with each negative added** while the cost rises steeply —
the step from 1 to 2 negatives is 2.6x slower for 42% of the improvement the first
one bought. Naive extrapolation puts `hard_negatives=4` near 0.197–0.201, still short
of `overlap` 0.2149.

That is the important negative result for the DL report: **row-private hard negatives
alone will not close the gap to token overlap.** Testing 4 on the new GPU is worth
doing to confirm the shape, but it should be framed as confirming saturation rather
than as a route to beating the baseline. The remaining gap more plausibly belongs to
the architecture, the from-scratch vocabulary, or the 0.26 mean positive score the
whole model is trained against.

**It still does not beat `overlap`.** hn=2 pulls level with `bm25_noidf` (0.1940 vs
0.1942) and remains 0.021 behind plain token F1. So the honest headline is *hard
negatives close most of the gap to the lexical baselines and none of the gap to the
best one* — which is a far more interesting result for the report than a win, and it
points at the next question: whether the ceiling is the architecture, the training
length, or the 0.26 mean positive score the whole thing is trained against.

**Cost:** 20k steps at 1.75 steps/s ≈ 3.2 h and 6,556 MiB, against 5.35 steps/s and
3,184 MiB at hn=0. Hard negatives triple the document encoding, so they are the most
expensive knob in the project as well as the most valuable.

### `best.pt` is chosen by the wrong metric

`train.monitor` is `val/recall@1` — the in-batch number. For this run it peaked at
step 12000 (0.2886) on a curve that is flat from 12000 to 20000 (0.281–0.289), so
`best.pt` is the step-12000 model, not the final one.

Measured both: step 12000 gives within-product 0.1940, step 20000 gives 0.1941.
**No harm here, but that is luck rather than design** — checkpoint selection is
running on a metric this project has twice shown to be misleading. Any run whose
in-batch curve is not flat should have its final checkpoint scored too.

## The earlier result this replaced

**The retriever trained on in-batch negatives alone loses to plain token overlap.**
`dense` at 0.1707 ranked fifth of seven — above only `random` and `first`, and below
every BM25 variant.

The same checkpoint scores **in-batch `val/recall@1` 0.4998**. Both numbers are
correct; they measure different problems, and the gap between them *is* the finding:

- **in-batch** ranks a snippet against candidates from *other products*. Winning
  needs only topical discrimination — "is this about a camera or a blender?"
- **within-product** ranks it against ~9 snippets from the *same* product, which all
  share the product's vocabulary. Winning needs relevance, not topic.

Trained on in-batch negatives alone, the model learned exactly the easier task, and
`loss.hard_negatives=0` in this run means it was never shown the harder one. This is
the empirical case for hard negatives, and it belongs in the DL report as the
motivating result rather than as a footnote — a 0.4998 quoted without the 0.1707
would be a serious misreport.

**That prediction was then tested and held** (see the section above): hn=2 lifted
within-product recall@1 to 0.1940. The diagnosis → intervention → measurement chain
is the spine of the DL report's retriever section.

`first` now sits marginally *below* `random` (0.1282 vs 0.1305), which sharpens the
original reading rather than changing it: the snippet pool carries no usable
ordering at all.

**BM25 loses to plain token overlap**, and the rows decompose why: corpus IDF over
pool-local IDF is worth +0.010, removing IDF entirely another +0.007, and the F1 form
over summed saturated tf a further +0.021. Within one product the discriminative
terms are the *common* ones — the product's own features — and IDF suppresses exactly
those. Details in `src/qar/retrieval/CLAUDE.md`.

**`val/recall@1` printed during training is a different number** — it ranks within
`eval_batch_size` candidates from *different* products, a far easier problem. Never
put it in the same table as the above.

## GPU

**Unblocked as of 2026-08-06.** `torch.cuda.is_available()` is now True in the dev
shell — `torch 2.13.0+cu130` sees the RTX 5060 Laptop (8 GiB, sm_120) and bf16
autocast runs. The earlier CPU-only measurement (0.32 steps/s at batch 32,
45.97M parameters) was a property of the shell, not of the code.

## Open questions

- Confirm with both lecturers that one codebase with two distinct reports is acceptable.
  Most departments allow it, some require a declaration, a few forbid it.
- Confirm whether the DL course permits pretrained weights. The current split makes this
  non-blocking, but it decides how much of the pretrained comparison goes in which report.

## Git

`.gitignore` excludes `*.jsonl`, `/data/`, `/runs/`, `.venv/` — verified that the
3.4 GB corpus and the processed corpus stay out, so `git add -A` is safe. The leading
slashes matter: an unanchored `data/` also matched `src/qar/data/` and silently
excluded the whole corpus-preparation package.

**Uncommitted work as of 2026-08-06**: the upstream test split, the `dense`
retriever, `merge_results`, the tests/conftest corpus guard, and every doc updated
with the regenerated numbers. `CHANGELOG.md` entries for all of it are written.
