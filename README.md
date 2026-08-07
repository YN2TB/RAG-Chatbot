# Review-Grounded Product Question Answering (AmazonQA)

A retrieval-augmented QA system over Amazon product reviews. The project supports two
coursework deliverables from one codebase:

- **Deep Learning report** — a from-scratch dual-encoder retriever trained with a
  contrastive (InfoNCE) objective and a multi-task answerability head, plus ablations.
- **NLP report** — the end-to-end grounded chatbot built on that retriever: generation,
  abstention, grounding and hallucination evaluation, error analysis.

## Dataset

`train-qar.jsonl` (2.67 GB), `val-qar.jsonl` (747 MB) and `test-qar_all.jsonl`
(751 MB), downloaded by `data.ipynb` and not tracked in git.

| | |
|---|---|
| Rows | 738,776 train / 92,183 val / 92,726 test |
| Unique products | 124,416 train / 15,592 val / 15,599 test |
| Unique questions | 684,703 train / 89,336 val |
| Snippets / answers | ~9.3 and ~3.9 per question (~6.8M snippets in train) |
| Categories | 17 (Electronics 23%, Home & Kitchen 15%, …) |
| Question type | 85% descriptive / 15% yes-no |
| Answerable | 62% / 38% |

Each row: `asin`, `category`, `questionText`, `questionType`, `review_snippets[]`,
`answers[{answerText, answerType, helpful}]`, `is_answerable`, `qid`. Four further
fields — `top_sentences_IR`, `top_review_wilson`, `top_review_helpful`,
`random_sentence` — are the dataset authors' precomputed baselines, which the pipeline
ignores so a competing system's output cannot leak into the positives. They are *not*
comparable rows for the retrieval table: only 7.2% of `top_sentences_IR` sentences
appear in `review_snippets`, so it ranks a different candidate set entirely.

### Three measured facts that shape the design

**The splits are product-disjoint.** Only 103 of 15,592 val products (0.7%) and 116
of 15,599 test products (0.74%) appear in train, so evaluation measures generalisation
to unseen products. The corpus ships its own test file, so nothing has to be carved
out of validation. Question-text overlap is higher (7.5% train↔val, 7.8% train↔test)
but is not leakage — it is generic phrasing (*"what's the weight limit?"*, *"Does it
swivel?"*) asked about different products with disjoint reviews.

**In-batch negatives need a dedup guard.** Train has 738,776 rows but 684,703 unique
questions. If two rows sharing a question string land in the same batch, InfoNCE treats
each one's positive as the other's negative. `data.dedup_questions_in_batch` controls
the fix; the ablation against it belongs in the DL report.

**There is no snippet-level relevance label.** A row lists its answers but never says
which review snippet supports them, so every training positive has to be inferred.
`prepare.selector` does that by token overlap against the reference answers — distant
supervision, and the weakest assumption in the system. It is a registered, swappable
component precisely so the report can measure how much it matters.

## Setup

Requires Python 3.12 (the NLP ecosystem still lags on 3.14) and a CUDA GPU.
Torch is pinned to the `cu130` index in `pyproject.toml`, which covers Blackwell
(sm_120) cards such as the RTX 5060.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"          # add ".[nlp]" for the generation half
```

Verify the harness end to end:

```bash
uv run pytest                        # config + data pipeline + overfit + resume checks
uv run python scripts/train.py configs/dev.yaml
```

`pytest` comes from the `dev` extra, which the setup step above installs. `uv run`
alone never installs an extra, so in a fresh environment run `uv sync --extra dev`
first.

## Preparing the corpus

One offline pass turns the three raw files into everything training reads:

```bash
uv run python scripts/prepare_data.py configs/base.yaml
```

| Output in `data/processed/` | Contents |
|---|---|
| `train.jsonl` | one record per usable train question |
| `val.jsonl` / `test.jsonl` | the validation and test files, one pass each |
| `tokenizer.json` | byte-level BPE fitted on the train split only |
| `manifest.json` | row accounting, the settings used, and a leakage report |

Each record carries the question, the full snippet pool, and `positive_idx` into that
pool. Keeping the pool is what later allows within-product evaluation and mining hard
negatives from the same product.

Changing any `prepare.*` field produces a different training corpus, so write it
somewhere new rather than overwriting:

```bash
python scripts/prepare_data.py configs/base.yaml \
    --set prepare.selector=first data.processed_dir=data/processed_first
```

## Baselines

```bash
python scripts/build_idf.py configs/base.yaml          # corpus IDF, ~3 min
python scripts/evaluate_retrieval.py configs/base.yaml # -> runs/_baselines/val.{json,md}
```

Ranking is scoped to **one product's snippet pool** — about nine candidates — because
that is the pool a real system would search. On val (87,475 rows, mean pool 9.33):

| retriever | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| random | 0.1305 | 0.3514 | 0.5486 | 0.3262 |
| first | 0.1282 | 0.3458 | 0.5455 | 0.3239 |
| bm25 | 0.1771 | 0.4244 | 0.6215 | 0.3763 |
| bm25_global | 0.1870 | 0.4427 | 0.6412 | 0.3878 |
| bm25_noidf | 0.1942 | 0.4564 | 0.6535 | 0.3961 |
| **overlap** | **0.2149** | **0.4903** | **0.6855** | **0.4184** |

Two results worth stating plainly. `first` does not beat `random`, so the snippet
pool carries no ordering to exploit. And **BM25 loses to plain token overlap** —
within a single product the discriminative terms are the common ones (the product's
own features), and IDF suppresses exactly those. The `bm25_global` and `bm25_noidf`
rows isolate that effect one variable at a time.

## Training the retriever

```bash
python scripts/train.py configs/retriever.yaml
```

A from-scratch dual encoder (~46M parameters) trained with InfoNCE over in-batch
negatives. Two things to keep in mind:

- **The batch size is part of the loss.** It is the number of negatives, so batch 256
  poses a harder problem than batch 32 and yields a sharper representation.
  `grad_accum` does *not* substitute — accumulation adds gradient steps, not
  candidates to the softmax.
- **`val/recall@1` in the training log is not the baseline number.** It ranks within
  `eval_batch_size` candidates drawn from different products, which is much easier
  than picking one snippet out of a single product's nine. The comparable figure
  comes from `scripts/evaluate_retrieval.py`.

Scoring a trained checkpoint against the table above uses the same entry point as
the baselines, which is the point of the `dense` retriever:

```bash
python scripts/evaluate_retrieval.py configs/retriever.yaml \
    --set retrieval.baselines=[dense] \
          retrieval.checkpoint=runs/retriever_b128_hn2/checkpoints/best.pt
```

The architecture is rebuilt from the checkpoint's own snapshotted config, so the
config driving the evaluation cannot silently redefine the model being scored.
Results merge into `runs/_baselines/<split>.json` rather than replacing it.

**Measured first run** (`runs/retriever_b128`, 20k steps, batch 128, 64.7 min on an
RTX 5060): loss 4.85 → 1.43, in-batch `val/recall@1` → 0.4998, still rising at the
step limit rather than converged.

Scored properly, it reaches **recall@1 0.1707** within-product — *below* every BM25
variant and well below `overlap`'s 0.2149. The two numbers measure different things:
in-batch ranking only needs to tell a camera review from a blender review, while
within-product ranking has to pick the right snippet out of nine that all discuss the
same product. This run used in-batch negatives only (`loss.hard_negatives=0`), so the
model was never trained on the harder distinction. That gap is the motivating result
for hard negatives, and quoting 0.4998 beside the baseline table would misreport it.

## Layout

```
configs/          run configs; ablations inherit via `_base_:`
sweeps/           ablation grids for scripts/sweep.py
scripts/          prepare_data.py (offline), train.py (single run), sweep.py (grid)
src/qar/
  config.py       typed config, YAML inheritance, CLI overrides
  registry.py     name -> component lookup, so configs can swap parts
  data/           corpus preparation, BPE, splitting, dataset + collator
  tasks/          trainable tasks (dev_toy, retriever)
  training/       trainer, task interface, schedules, checkpointing
  eval/           ranking and classification metrics
  utils/          seeding, device/AMP, logging
tests/            harness correctness
runs/             per-run outputs (gitignored)
```

## Running experiments

Every run is fully described by a config plus explicit overrides, and the resolved
config is snapshotted to `runs/<name>/config.yaml`.

```bash
# single run
python scripts/train.py configs/dev.yaml --set optim.lr=1e-4 name=lr1e-4

# ablation grid -> runs/_sweeps/temp.csv and .md
python scripts/sweep.py sweeps/temperature.yaml
```

Each run directory contains:

| File | Contents |
|---|---|
| `config.yaml` | resolved config for the run |
| `metrics.jsonl` | one JSON record per log/eval point — training curves for the report |
| `train.log` | human-readable console log |
| `checkpoints/` | rotated `step_*.pt` plus `best.pt` |

`metrics.jsonl` is append-only, so a resumed run keeps the curve from before the crash.

## Status

- [x] Harness: config, seeding, AMP, trainer, checkpoint/resume, logging, sweeps, tests
- [x] Split validated (leakage checked)
- [x] Data pipeline: pair construction, BPE tokenizer, upstream train/val/test splits
- [x] Lexical baselines (random, first, overlap, three BM25 variants)
- [x] From-scratch bi-encoder + InfoNCE, de-duplicating batch sampler, answerability head
- [x] `dense` retriever: scores a checkpoint through the same within-product path
      as the baselines, so a run produces a directly comparable number
- [x] A real GPU training run (`runs/retriever_b128`, 20k steps, 64.7 min)
- [ ] A longer run — the curve was still rising when `max_steps` cut it off
- [ ] Hard negative mining from the retriever itself (row-private negatives already exist)
- [ ] Ablations → DL report
- [ ] Generation, grounding evaluation, abstention → NLP report
