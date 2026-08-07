# CLAUDE.md — src/qar/retrieval/

> **Before touching this folder:** read `.claude/` (agent roster, permission
> rules) and the root `CLAUDE.md`. Record the change in `CHANGELOG.md` in the
> same commit — what was learned, not what the diff shows.

The baselines, the trained bi-encoder scored against them, and the harness that
turns both into the results table the DL report is built on.

```
base.py      the Retriever interface — scope is one product's pool
trivial.py   random (the floor), first (the position prior)
lexical.py   bm25, bm25_global, bm25_noidf, overlap
dense.py     the trained bi-encoder, loaded from a checkpoint
idf.py       corpus-wide document frequencies
evaluate.py  ranking over a prepared split + breakdowns + Markdown table
```

## `dense` — the only entry that is not a baseline

Everything else here is untrained by construction. `dense` loads a checkpoint and
scores the **same rows, pools, shuffle and metric code** as the baselines, which is
what makes a training run comparable with the table it has to beat:

```bash
python scripts/evaluate_retrieval.py configs/retriever.yaml \
    --set retrieval.baselines=[dense] \
          retrieval.checkpoint=runs/<name>/checkpoints/best.pt
```

**The architecture is rebuilt from the checkpoint's own snapshotted config**, not
from the config driving the evaluation. A width or layer-count mismatch would fail
loudly, but a different `pooling` or `max_query_len` would load cleanly and rank
nonsense — so the evaluating config does not get a vote.

Two things it must never be confused with:

- **`val/recall@1` in the training log is a different number.** It ranks within
  `eval_batch_size` candidates drawn from *other* products — an easier problem over
  a different candidate count. Only `dense` produces a figure comparable to
  `overlap`.
- **This is still within-product**, ~9 candidates. A global-index evaluation over
  6.5M snippets is a separate, harder measurement that needs its own entry point.

No temperature is applied: both towers L2-normalise, so the scores are cosines and a
positive constant cannot change a ranking.

**Cost.** Whole validation took 18.5 min per checkpoint scoring one row at a time —
more wall time than training, once a 19-cell grid is involved. `score_batch` is the
fix: `DenseRetriever` overrides it to encode a chunk of queries and all their pooled
documents in two forward passes, chunk size `retrieval.batch_rows` (default 256).

`retrieval.max_rows` gives a fast estimate, but a number quoted beside the baseline
table must come from the whole split, since those did.

## `score_batch` — batching that may not change a score

`Retriever.score_batch` defaults to the per-row loop, so every lexical baseline is
untouched; only `dense` overrides it. The contract is strict: **an override must
return exactly what the per-row path would.**

Batching pads documents from different rows to a common length, so the guarantee
rests on `TextEncoder` masking padding out of its mean pooling. That invariant has
its own test (`test_padding_cannot_change_the_representation`); do not weaken it and
expect this to keep working.

Two tests hold the line here: `test_dense_batched_matches_per_row` compares the two
paths at `abs=1e-4`, and `test_batch_rows_does_not_change_the_metrics` runs a full
evaluation at chunk 1, 3 and 256 and requires identical metrics. A performance knob
that changes a result is not a performance knob.

## Contract

**In** — a `RunConfig` and a `PairDataset` over a prepared split. Retrievers see the
question and the snippet pool. **None of them may see `positive_idx`** — that is the
answer, and a baseline that peeks is not a baseline.

**Out** — `dict[str, Any]` of metrics, and `runs/_baselines/<split>.{json,md}`.
Metrics come from `qar.eval.metrics.ranking_metrics` unchanged; nothing here
re-implements a metric.

## Scope: one product, ~9 candidates

Every retriever ranks the snippets of **one product**. A user asking about a camera
is never served a sentence from a blender review, so that is the realistic pool.

**This must be stated wherever these numbers appear.** recall@1 over 9 candidates is
not recall@1 over a 6.5M-snippet index, and the two must never share a table column.
The interface deliberately cannot express a global index, so the two cannot be
conflated by accident — evaluating the bi-encoder globally will need a different
entry point, not a flag on this one.

## Measured on val (87,475 rows, mean pool 9.33, k1=1.5 b=0.75)

Regenerated 2026-08-06 on whole validation. The previous run of this table used
43,801 rows, from when test was carved out of validation. **No value moved by more
than 0.005 and the ordering is unchanged** — the findings below were not an artefact
of the smaller sample.

| retriever | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| first | 0.1282 | 0.3458 | 0.5455 | 0.3239 |
| random | 0.1305 | 0.3514 | 0.5486 | 0.3262 |
| dense (trained, 20k steps, in-batch negatives only) | 0.1707 | 0.4232 | 0.6276 | 0.3728 |
| bm25 (pool-local IDF) | 0.1771 | 0.4244 | 0.6215 | 0.3763 |
| bm25_global (corpus IDF) | 0.1870 | 0.4427 | 0.6412 | 0.3878 |
| bm25_noidf (IDF = 1) | 0.1942 | 0.4564 | 0.6535 | 0.3961 |
| **overlap** (token F1) | **0.2149** | **0.4903** | **0.6855** | **0.4184** |

Five findings the report should carry.

**The trained retriever does not beat the lexical baselines.** `dense` sits fifth of
seven, below every BM25 variant. The same checkpoint scores in-batch
`val/recall@1` 0.4998 — the gap is not a bug in either number but the difference
between discriminating topic (in-batch, other products) and discriminating relevance
(within-product, same vocabulary). Trained with `loss.hard_negatives=0`, the model
only ever saw the former. Full reasoning in the root `CLAUDE.md`.

**The pool is not pre-ordered.** `first` (0.1282) does not beat `random` (0.1305) —
on the larger sample it sits a hair *below* it. Snippet position carries no relevance
signal, so no baseline is quietly benefiting from it. Worth establishing before
trusting any other number here.

**Chance is 0.131, not 1/9.33 = 0.107.** Pool sizes vary and the mean of `1/pool`
exceeds `1/mean(pool)`, so the floor has to be measured rather than derived.

**BM25 loses to plain token overlap on this task** — the opposite of what a
retrieval baseline table normally shows. The rows above decompose the 0.0378 gap
into three separate causes, each isolated to one changed term:

| change | Δ recall@1 |
|---|---|
| pool-local IDF → corpus-wide IDF | +0.0099 |
| corpus-wide IDF → no IDF at all | +0.0072 |
| summed saturated tf → token F1 | +0.0207 |

**IDF weighting is actively harmful here, and length normalisation is what BM25 was
getting wrong.** Within one product, the terms a question turns on are the common
ones — "fit", "last", "battery", the product's own features recurring across its
reviews. IDF promotes rare tokens instead: model numbers, proper nouns and typos,
which are poor evidence that a snippet answers anything. A `b` sweep on 15k rows
confirms the length half, monotonically:

| b (at k1=1.5) | 0.4 | 0.75 | 1.0 |
|---|---|---|---|
| bm25_global recall@1 | 0.1749 | 0.1862 | 0.1936 |

**Methodology note.** That sweep was run on val, so `b=1.0` must not be quoted as a
val result — parameters chosen on a split cannot also be scored on it. The table
above therefore uses the standard `k1=1.5, b=0.75`; the final report table should be
produced on **test** with whatever parameters val selected.

## Rules

- **Baselines never see the label.** No `positive_idx`, no answers.
- **Pools are shuffled per row before scoring.** Lexical scorers return 0.0 for
  every candidate when the question shares no words with the pool; `argsort` then
  falls back to position and hands those rows to snippet 0, silently blending the
  `first` baseline into every other row. Shuffling makes an unscorable row land at
  chance, which is what "the method had nothing to say" should look like. A test
  locks this: `test_unscorable_rows_land_at_chance_not_on_snippet_zero`.
- **`bm25` and `bm25_global` differ in one thing only** — where IDF comes from. That
  is what makes the gap between their rows a measurement of that decision rather
  than of two unrelated implementations.
- **IDF is built from train only.** Fitting term statistics on val or test leaks
  their vocabulary into a baseline those splits then score.
- New baselines register under kind `retriever` and get a row in the table. A
  baseline nobody can reproduce from a config is not a baseline.

## Gotchas

- `bm25_global` needs `scripts/build_idf.py` to have run; it is in the default
  `retrieval.baselines`, so the whole table fails without it. The error names the
  script.
- A term missing from the IDF table is scored as if it had `min_df` occurrences,
  not zero. Treating it as unseen would give it near-infinite weight and let one
  typo in a question dominate the ranking.
- The whole split is scored into one padded matrix (44k × 32 floats ≈ 6 MB).
  Padding is `-inf` so absent candidates always rank last. Do not "optimise" this
  into chunks — one matrix is why the metrics come from `eval/` unchanged.
- `recall@k` for `k` near the pool size is close to meaningless (recall@10 over a
  pool of 9 is 1.0 by construction). `retrieval.ks` defaults to 1/3/5 for that
  reason.

## Still missing

- Global-index retrieval. Everything here is within-product.
- Hard-negative-aware evaluation once mining exists.

## Not coming: a `top_sentences_IR` row

The dataset ships the authors' own IR baseline, and it looks like a free extra row
until you check. Over 3,000 validation rows only **7.2%** of its sentences appear in
`review_snippets`, and **54%** of rows share none — it selects sentences from the
review text by a different extraction, so it ranks a different candidate set. Scoring
it against our `positive_idx` would compare two different pools and produce a number
that means nothing. It belongs in the NLP report as an alternative evidence set for
the generator, not in this table.
