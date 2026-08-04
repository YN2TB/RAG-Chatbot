# CLAUDE.md — src/qar/retrieval/

Untrained baselines, and the harness that turns them into the results table the DL
report needs before any trained number means anything.

```
base.py      the Retriever interface — scope is one product's pool
trivial.py   random (the floor), first (the position prior)
lexical.py   bm25, bm25_global, overlap
idf.py       corpus-wide document frequencies
evaluate.py  ranking over a prepared split + breakdowns + Markdown table
```

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

## Measured on val (43,801 rows, mean pool 9.32, k1=1.5 b=0.75)

| retriever | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| random | 0.1272 | 0.3483 | 0.5458 | 0.3237 |
| first | 0.1294 | 0.3487 | 0.5471 | 0.3252 |
| bm25 (pool-local IDF) | 0.1768 | 0.4236 | 0.6210 | 0.3759 |
| bm25_global (corpus IDF) | 0.1861 | 0.4417 | 0.6406 | 0.3872 |
| bm25_noidf (IDF = 1) | 0.1940 | 0.4561 | 0.6532 | 0.3961 |
| **overlap** (token F1) | **0.2145** | **0.4906** | **0.6868** | **0.4181** |

Four findings the report should carry.

**The pool is not pre-ordered.** `first` ≈ `random` to three decimal places, so
snippet position carries no relevance signal and no baseline is quietly benefiting
from it. Worth establishing before trusting any other number here.

**Chance is 0.127, not 1/9.32 = 0.107.** Pool sizes vary and the mean of `1/pool`
exceeds `1/mean(pool)`, so the floor has to be measured rather than derived.

**BM25 loses to plain token overlap on this task** — the opposite of what a
retrieval baseline table normally shows. The rows above decompose the 0.0377 gap
into three separate causes, each isolated to one changed term:

| change | Δ recall@1 |
|---|---|
| pool-local IDF → corpus-wide IDF | +0.0093 |
| corpus-wide IDF → no IDF at all | +0.0079 |
| summed saturated tf → token F1 | +0.0205 |

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
