# Review-Grounded Product Question Answering (AmazonQA)

A retrieval-augmented QA system over Amazon product reviews. The project supports two
coursework deliverables from one codebase:

- **Deep Learning report** — a from-scratch dual-encoder retriever trained with a
  contrastive (InfoNCE) objective and a multi-task answerability head, plus ablations.
- **NLP report** — the end-to-end grounded chatbot built on that retriever: generation,
  abstention, grounding and hallucination evaluation, error analysis.

## Dataset

`train-qar.jsonl` (2.67 GB) and `val-qar.jsonl` (747 MB), not tracked in git.

| | |
|---|---|
| Rows | 738,776 train / 92,183 val |
| Unique products | 124,416 train / 15,592 val |
| Unique questions | 684,703 train / 89,336 val |
| Snippets / answers | ~9.3 and ~3.9 per question (~6.8M snippets in train) |
| Categories | 17 (Electronics 23%, Home & Kitchen 15%, …) |
| Question type | 85% descriptive / 15% yes-no |
| Answerable | 62% / 38% |

Each row: `asin`, `category`, `questionText`, `questionType`, `review_snippets[]`,
`answers[{answerText, answerType, helpful}]`, `is_answerable`, `qid`.

### Two measured facts that shape the design

**The split is product-disjoint.** Only 103 of 15,592 val products (0.7%) appear in
train, so evaluation measures generalisation to unseen products. Question-text overlap
is higher (7.5%) but is not leakage — it is generic phrasing (*"what's the weight
limit?"*, *"Does it swivel?"*) asked about different products with disjoint reviews.

**In-batch negatives need a dedup guard.** Train has 738,776 rows but 684,703 unique
questions. If two rows sharing a question string land in the same batch, InfoNCE treats
each one's positive as the other's negative. `data.dedup_questions_in_batch` controls
the fix; the ablation against it belongs in the DL report.

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
uv run pytest                        # config + overfit + resume + grad-accum checks
uv run python scripts/train.py configs/dev.yaml
```

## Layout

```
configs/          run configs; ablations inherit via `_base_:`
sweeps/           ablation grids for scripts/sweep.py
scripts/          train.py (single run), sweep.py (grid + results table)
src/qar/
  config.py       typed config, YAML inheritance, CLI overrides
  registry.py     name -> component lookup, so configs can swap parts
  tasks/          trainable tasks (dev_toy today; retriever next)
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
- [ ] Data pipeline: pair construction, BPE tokenizer, asin-based val/test split
- [ ] BM25 baseline
- [ ] From-scratch bi-encoder + InfoNCE
- [ ] Hard negative mining, multi-task answerability head
- [ ] Ablations → DL report
- [ ] Generation, grounding evaluation, abstention → NLP report
