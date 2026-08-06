# CLAUDE.md — src/qar/data/

> **Before touching this folder:** read `.claude/` (agent roster, permission
> rules) and the root `CLAUDE.md`. Record the change in `CHANGELOG.md` in the
> same commit — what was learned, not what the diff shows.

Everything between the raw corpus and a batch of tensors.

```
text.py       normalisation + token overlap (SQuAD rules)
schema.py     raw row -> validated RawRow, or None
select.py     which snippet is the positive (registered, swappable)
split.py      deterministic asin -> val/test
tokenizer.py  byte-level BPE, fitted on the train split only
prepare.py    the one offline pass; writes the processed corpus + manifest
dataset.py    byte-offset indexed reading, padding collator
sampler.py    batches with distinct questions — the false-negative guard
```

## Contract

**In** — the two raw JSONL files named by `data.train_path` / `data.val_path`, plus
`cfg.prepare` and `cfg.model.vocab_size`. Nothing else: no environment, no network,
no hidden resource files.

**Out** — `data.processed_dir` containing:

| File | Contents |
|---|---|
| `train.jsonl` | one record per usable train question |
| `val.jsonl` / `test.jsonl` | the validation file, split by hashed asin |
| `tokenizer.json` | self-contained byte-level BPE |
| `manifest.json` | row accounting, prepare settings, leakage report |
| `*.offsets.npy` | byte-offset index, written lazily by `dataset.py` |

Processed record schema — **this is the interface the retriever task consumes**:

```
qid, asin, category, question, question_type, is_answerable (0/1),
qgroup, positive_idx, positive_score, snippets[]
```

`positive_idx` indexes into `snippets`; the positive is never stored as a second
copy, so the two cannot drift apart. `snippets` is the full retrieval pool, kept
for within-product evaluation and for mining hard negatives from the same product.

## The one decision that matters

**AmazonQA has no snippet-level relevance label.** A row knows its answers but not
which review snippet supports them. Every positive in `train.jsonl` is inferred —
distant supervision, and the weakest link in the pipeline.

That is why `select.py` is a registry, not a function: `prepare.selector` is an
ablation axis for the DL report, not an implementation detail.

| Selector | Bias |
|---|---|
| `answer_overlap` (default) | token F1 — penalises snippets much longer than the answer |
| `answer_recall` | answer-token coverage — no length penalty, favours long snippets |
| `first` | null control: does the selector matter at all? |

`first` returns score 1.0 on purpose, so `min_positive_score` cannot silently feed
it a different row population than the real selectors see. A control that trains on
different data is not a control.

**Measured on the full corpus** (`answer_overlap`, `min_positive_score=0.10`, 13.5 min):

| | train | val | test |
|---|---|---|---|
| rows read | 737,214 | 46,015 | 46,022 |
| no trustworthy positive | 33,013 (4.5%) | 2,214 | 2,348 |
| kept | 704,201 | 43,801 | 43,674 |
| unique products | 123,616 | 7,739 | 7,757 |
| snippets / row | 9.30 | 9.32 | 9.33 |
| answerable | 62.8% | 65.0% | 65.1% |
| mean positive score | 0.262 | 0.260 | 0.260 |

Plus 1,562 train and 146 validation rows that never parsed. Read + malformed comes
to exactly 738,776 and 92,183 — the corpus row counts — so nothing is silently lost.

**A mean positive score of 0.26 is the headline limitation of this project.** It says
the average inferred positive shares roughly a quarter of its tokens with the
reference answer. Distant supervision at that strength is normal and usable, but the
DL report has to state it: the retriever is being trained towards a noisy target, and
its ceiling is set by that noise, not only by the architecture.

## Leakage, measured

`manifest.json` carries the check the split design exists to pass:

```
asin_overlap_val_test    0      <- the one this pipeline controls
asin_overlap_train_val   49  }  103 total, inherited from the upstream
asin_overlap_train_test  54  }  train/validation split, not introduced here
```

The 103 matches the count already recorded for the raw corpus exactly, which is the
evidence that asin hashing kept every product whole: it moved the pre-existing
overlap around between val and test without creating any of its own.

Question-text overlap (3,887 between train and val) is expected and is **not**
leakage — generic phrasing recurring across products whose reviews are disjoint.

## Rules

- **The four baseline fields are dropped on purpose.** `top_sentences_IR`,
  `top_review_wilson`, `top_review_helpful` and `random_sentence` are the dataset
  authors' own baselines, and keeping them out of the pipeline stops a competing
  system's output leaking into the positives. `schema.BASELINE_FIELDS` names them and
  the manifest records that they were ignored.

  **Do not mistake them for extra rows in the retrieval table.** Measured over 3,000
  validation rows, only 7.2% of `top_sentences_IR` sentences occur in
  `review_snippets`, and 54% of rows share none at all: it is a *different extraction*
  from the review text (mean 9.8 sentences against our 9.3), not a reranking of the
  same candidates. Scoring it against `positive_idx` would compare two different
  candidate sets and mean nothing. Its honest use is as an alternative **evidence
  set** for the NLP report's generator.
- **The tokenizer is fitted on the train split only.** Fitting on val or test would
  leak their token distribution into the model's input representation.
- **`prepare.split_seed` is not `seed`.** Changing a run's seed must never reshuffle
  the data split, or two runs stop being comparable. Keep them separate.
- **A malformed row is counted, not fatal.** A 738k-row corpus always has a few; the
  manifest reports how many so the loss stays visible.
- **Changing a `prepare.*` field changes the training data.** Write to a new
  `data.processed_dir` rather than overwriting — the two corpora are an ablation
  pair, not successive versions of one thing.
- Prepare is offline and one-shot. Nothing here may be called from the training
  loop.

## Gotchas

- **`dataset.py` opens files in binary mode.** Windows text mode translates
  newlines, which makes `tell()` return values `seek()` cannot use — the offset
  index would silently return the wrong records. Do not "simplify" it to text mode.
- **The file handle is opened lazily and dropped from `__getstate__`.** Windows
  spawns DataLoader workers rather than forking, so the dataset is pickled; a live
  file handle cannot be.
- **`_Reservoir`, not the first N texts.** The corpus is grouped by product, so the
  head of the file is a handful of categories and a vocabulary fitted on it would
  be a vocabulary of cameras.
- Offsets are cached beside the file and invalidated by mtime. Rewriting a split
  without touching its mtime would serve stale offsets — prepare always rewrites,
  so this only bites if you hand-edit a processed file.

## Measured cost

Preparing the full corpus takes ~13.5 minutes and writes 2.2 GB
(`train.jsonl` 1.94 GB, `val`/`test` 146 MB each, `tokenizer.json` 2.1 MB).
Indexing `train.jsonl` by byte offset takes 3 s on first open and is cached after.

Loading runs at **~5,400 pairs/s single-threaded** — about 85 batches of 64 per
second, far above what any GPU step will consume. `data.num_workers=0` is therefore
the right default on Windows and the loader will not be the bottleneck; raising it
should be justified by a measurement, not assumed.

## The false-negative guard (sampler.py)

704,201 rows, 638,306 distinct questions — about 66,000 rows share their question
text with another row. In-batch InfoNCE treats every other row's positive as a
negative, so when two of those meet in one batch the gradient says *"this correct
snippet is wrong"*. `data.dedup_questions_in_batch` turns `QGroupBatchSampler` on;
it packs batches greedily from a shuffled order, deferring clashes for up to three
passes and reporting what it could not place in `.dropped`.

Batches are always exactly `batch_size` long. Not tidiness: the batch size **is** the
number of negatives, so a short batch would quietly make one step an easier problem
than its neighbours.

**It raises if the split has fewer distinct questions than `batch_size`**, because no
valid batch could ever be built and yielding nothing reads to the trainer as an empty
epoch, forever. That combination used to hang the suite; `Trainer._endless` now
refuses an empty loader as well, so both halves are guarded.

`qgroup` is read via `question_groups()`, which caches a uint64 array beside the
split — 5.6 MB for 704k rows, ~6 s to build once.

## Hard negatives, for free

`PairCollator(hard_negatives=n)` draws `n` extra snippets per row **from that row's
own product pool** — which is why `snippets` is kept in the processed record. No
mining pass is needed for this first form. Unfillable slots (a pool of one) are
emitted as placeholders with `neg_valid=False` so the task can mask them out of the
softmax instead of pretending they are candidates.

Model-based mining — train once, retrieve, take the highly-ranked non-positives — is
the refinement that would come after, and it is not built.

## Still missing

- Model-based hard-negative mining (round-2 negatives from the retriever itself).
