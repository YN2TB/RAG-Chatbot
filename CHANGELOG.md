# Changelog

Every change to this project gets an entry here, in the same commit as the change.

This is not a duplicate of `git log`. Git records *what* the code became; this
records **what was learned and what was decided** — a measurement that came out
against expectation, an assumption that turned out false, a default that changed
and why. Those are the things the two reports have to explain, and they are
invisible in a diff.

## How to write an entry

Newest first. One `##` heading per date, `###` per change.

```markdown
## YYYY-MM-DD

### Short imperative title
**What changed.** Files or behaviour, one or two lines.
**Why.** The reason, not the restatement.
**Evidence.** The number, the command, or the test that settles it.
**Consequences.** What is now invalidated, incomparable, or newly possible.
```

Omit a field that has nothing to say. Never write "various fixes" — an entry that
does not name a consequence is not worth the line it occupies.

**A measured number belongs here with the split it came from.** `recall@1 0.2145`
means nothing without "val, within-product, mean pool 9.32".

---

## 2026-08-08

### Fault tolerance was throughput-dependent; now it has a wall-clock floor
**What changed.** `train.save_every_minutes` (default 10.0, 0 disables) runs
alongside `save_every`; whichever cadence comes first writes the checkpoint.
`_rotating_save` restarts the timer on every save however triggered, so the two
cannot stack into consecutive writes.
**Why.** `save_every` counts steps, so the real time it protects varies with
throughput. At `save_every=2000` that is ~7 minutes at hn=1's 4.5 steps/s but **~19
minutes** at hn=2's 1.75 — the slowest and most expensive runs had the *weakest*
protection, which is exactly backwards.
**Evidence.** Two tests: one disables the step cadence entirely and asserts the timer
alone still produces a checkpoint; the other sets the interval to 0 and asserts only
the final checkpoint exists.
**Consequences.** Worst-case loss from an interruption is now bounded by wall time
rather than by whatever `steps/s` a configuration happens to run at — which matters
most for the hard-negative and large-batch runs, the ones that cost hours. Cost is
one extra ~550 MB write per 10 minutes, absorbed by `keep_last=2`.

### The hard-negative curve saturates
**What changed.** `runs/retriever_b128_hn1` — the missing middle point. All three
cells now exist at batch 128 and `max_steps=20000`, so the schedules match and the
curve is a controlled comparison.
**Evidence.** val, within-product, 87,475 rows:

| n | recall@1 | Δ | mrr | steps/s | VRAM |
|---|---|---|---|---|---|
| 0 | 0.1707 | — | 0.3728 | 5.35 | 3,184 MiB |
| 1 | 0.1871 | +0.0164 | 0.3907 | 4.52 | 4,830 MiB |
| 2 | 0.1940 | +0.0069 | 0.3967 | 1.75 | 6,556 MiB |

**Reading it.** The gain roughly halves per negative added while cost climbs
steeply: going from 1 to 2 costs 2.6x the wall time for 42% of the improvement the
first negative bought.
**Consequences.** This is the DL report's important *negative* result: row-private
hard negatives alone will not close the gap to `overlap` 0.2149. Extrapolating the
halving puts n=4 near 0.197–0.201. Testing 4 on the new GPU is worth doing to confirm
the shape, but it should be framed as confirming saturation, not as a route to
beating the baseline. The residual gap more plausibly belongs to the architecture,
the from-scratch vocabulary, or the 0.26 mean positive score the model is trained
against — which makes the selector ablation (`prepare.selector`) the more promising
next lead than more negatives.

### Multi-task grid complete: the head is free, and the control flipped the sign
**What changed.** `runs/retriever_hn2_12k` (the matched control) and
`runs/retriever_hn2_mt05`. All three cells now share hn=2 and `max_steps=12000`.
**Evidence.** val, within-product, 87,475 rows:

| weight | recall@1 | mrr | ans_recall | ans_acc | ans_f1 |
|---|---|---|---|---|---|
| 0.0 | 0.1924 | 0.3950 | — | — | — |
| 0.1 | 0.1934 | 0.3962 | 0.901 | 0.700 | 0.796 |
| 0.5 | 0.1939 | 0.3964 | 0.887 | 0.708 | 0.798 |
| untrained | — | — | 0.996 | 0.655 | 0.790 |

**Reading it.** The head learns: `ans_recall` 0.996 → 0.887 and accuracy 0.708
against a 0.651 class prior. Retrieval rises monotonically across the cells but the
whole spread is 0.0015, so the claim is *no measurable cost*, not *a gain*. Nothing
suggests a trade-off even at 0.5; above 0.5 is untested.
**Consequences, and the point of the exercise.** Against the unmatched 20k
checkpoint the head appeared to **cost** 0.0006; against the matched 12k control it
**gains** 0.0010. The confound was large enough to flip the sign of the effect. Any
cross-grid comparison in this project needs a control at the same `max_steps`.

### A 12k schedule lands ~0.0016 below a 20k one
**Evidence.** hn=2, no head: the 12k schedule reaches within-product recall@1
0.1924; the 20k run reaches 0.1940 at step 12000 and 0.1941 at step 20000.
**Why.** The cosine decay is defined over `max_steps`, so a 12k run anneals faster
and settles slightly lower. This is a property of the schedule, not evidence that
the model was still learning.
**Consequences.** 12k stays the right ablation length — every cell in a grid shares
it and only relative differences are read. But a **headline** number for the report
should come from a 20k run: quoting 0.1924 beside `overlap` 0.2149 would understate
the retriever by roughly a sixth of the remaining gap.

## 2026-08-07

### The multi-task head learns, and costs retrieval nothing
**What changed.** `runs/retriever_hn2_mt01` — the first run in the project's history
with `loss.answerable_weight > 0`, so the first time the answerability head has
existed at runtime at all.
**Evidence.** Head, val: `ans_recall` 0.9957 → **0.901**, `ans_acc` 0.6555 →
**0.700**, `ans_f1` 0.7904 → 0.796. Retrieval, val within-product: recall@1
**0.1934** against 0.1940 for hn=2 without the head.
**Reading it.** Accuracy is +0.049 over the 0.651 class prior and recall has fallen
well off 1.0, so the head is a real classifier rather than a constant function. F1
moves barely at all because it is dominated by the majority class — `ans_recall` is
the honest first read, and `ans_acc` alone would also have been misleading.
**Consequences.** The auxiliary task does not compete with the retrieval objective
at weight 0.1, so the "multi-task" half of the DL report now has a result behind it
rather than a claim about code. Whether it can be pushed harder is what
`answerable_weight=0.5` would answer.
**Caveat.** The comparison is a 12k run against step 12000 of a 20k run, and those
had different LR trajectories (below). `runs/retriever_hn2_12k` is the matched
control.
**Also:** the run was killed at step 11700 of 12000 by something outside the
harness. It was already converged — `val/recall@1` flat at 0.281/0.281/0.280 across
9k/10k/11k — so `best.pt` (step 10000) was scored rather than spending 20 minutes
finishing.

### `max_steps` is part of the comparison, not just the stopping rule
**What changed.** Documented as a convention in the root `CLAUDE.md`.
**Why.** `build_scheduler` takes `total_steps=cfg.train.max_steps`, so the cosine
decay is defined over the configured length. Step 12000 of a 12k run has annealed
the LR to zero; step 12000 of a 20k run is still mid-decay. Two checkpoints at the
same step number from runs with different `max_steps` are **not** a controlled
comparison, even though the project convention "compare at equal steps" is satisfied.
**Consequences.** Every sweep already pins `max_steps` in its `fixed:` block, so
grids are internally clean. Comparisons *across* grids of different lengths need a
matched control — which is why `retriever_hn2_12k` is being run rather than reusing
the 20k run's step-12000 checkpoint.

### The answerability head's baseline is F1 0.79, not zero
**What changed.** Documented in `src/qar/tasks/CLAUDE.md` and at the top of
`sweeps/multitask.yaml`. No code change — the metrics were always right, the
reading of them was the trap.
**Why.** Validation is 65.1% answerable, so a head that always predicts "yes"
scores precision 0.651, recall 1.000, F1 **0.789**, accuracy 0.651.
**Evidence.** Measured at step 60 of a real run, before the head could have learned
anything: `ans_acc` 0.6555, `ans_precision` 0.6563, `ans_recall` **0.9957**,
`ans_f1` 0.7904 — the majority-class baseline almost exactly.
**Consequences.** The head has done something only if F1 clears ~0.79 *and*
`ans_recall` falls meaningfully below 1.0. `ans_recall` is the honest first read:
while it sits near 1.0 the head is a constant function whatever F1 says. `ans_acc`
should not be quoted at all — it cannot beat 0.651 without the model taking a real
risk. Caught before any run produced a number, which is the only reason it will not
appear in the report as a success.

### Convergence measured: 12,000 steps with hard negatives
**Evidence.** `retriever_b128_hn2` peaked in-batch at step 12000 (0.2886) and stayed
flat to 20000 (0.281–0.289). Scored within-product, step 12000 gives recall@1 0.1940
and step 20000 gives 0.1941 — the last 8,000 steps, ~1.3 h, bought 0.0001.
`retriever_b128` (hn=0) was still rising at 20000.
**Consequences.** `train.max_steps=12000` becomes the standard ablation length,
cutting ~40% off every cell of a 19-cell grid. Two caveats for the report: the
earliest surviving checkpoint was step 12000 (`keep_last=2` deleted the rest), so
this is an upper bound on convergence rather than the point itself; and it is
measured for hn=2 only — the two configurations demonstrably do not converge at the
same rate, so an hn=0 arm may need longer to be fairly represented.

### Hard negatives lift within-product recall@1 from 0.1707 to 0.1940
**What changed.** `runs/retriever_b128_hn2` — the same recipe as `retriever_b128`
with `loss.hard_negatives=2`. No code change; one config field.
**Evidence.** val, within-product, 87,475 rows, mean pool 9.33, 20k steps both:
hn=0 recall@1 **0.1707** / mrr 0.3728, hn=2 recall@1 **0.1940** / mrr 0.3967.
**+0.0233 recall@1, +13.6% relative.**
**Why it matters.** It confirms the diagnosis the previous run only implied. A model
trained on other-product negatives learns topic; shown same-product negatives it
learns relevance. Diagnosis → intervention → measurement, all on the same split.
**Consequences.** hn=2 draws level with `bm25_noidf` (0.1940 vs 0.1942) and is still
0.021 behind `overlap` 0.2149. The honest claim is that hard negatives close most of
the gap to the lexical baselines and none of the gap to the best one. What remains
unresolved is whether the ceiling is the architecture, the training length, or the
0.26 mean positive score the model is trained against — that is the DL report's
discussion section.
**Cost.** 1.75 steps/s and 6,556 MiB against 5.35 steps/s and 3,184 MiB at hn=0:
~3.2 h for 20k steps. Hard negatives triple document encoding, making them the most
expensive knob in the project as well as the most valuable.

### `best.pt` is selected on the metric this project distrusts
**Evidence.** `train.monitor` is the in-batch `val/recall@1`. For the hn=2 run it
peaked at step 12000 (0.2886) on a curve flat from 12000 to 20000 (0.281–0.289), so
`best.pt` is the step-12000 model. Scored within-product: step 12000 → 0.1940,
step 20000 → 0.1941.
**Consequences.** No harm in this run, but only because the model had plateaued —
that is luck, not design. Any run whose in-batch curve is still moving should have
its final checkpoint scored alongside `best.pt`. Changing `train.monitor` is not a
fix either: the within-product metric cannot be computed inside the training loop,
which is the whole reason `dense` exists as a separate evaluation.

### Batched dense evaluation, done wrong then done right
**What changed.** `DenseRetriever._encode` now sorts by token length and encodes in
sub-batches bounded by the new `retrieval.encode_batch` (default 256), restoring the
caller's order afterwards.
**Why.** The first attempt encoded every document of a 256-row chunk in one pass.
That was **slower than the per-row code it replaced**: >50 min against 18.5 min for
whole validation, and it was killed rather than allowed to finish.
**Evidence.** Two causes, both measured. Padding: snippets average 72 tokens against
a 128 cap, so padding 2,400 mixed-length documents to one common width wasted ~1.78x
the compute. Memory: allocation reached 7.7 GiB of 8.1, and this card does not raise
on overcommit — it spills to system RAM and crawls, the same silent failure as
`hard_negatives=4`. After the fix: 15 s per 4,000 rows, **~3.5 min for whole
validation** (measured 215 s), against 18.5 min per-row.
**Consequences.** A 19-cell ablation grid now costs ~1 h of evaluation rather than
~6 h. `test_encode_batch_does_not_change_scores` pins the invariant across
`encode_batch` of 1, 2, 3 and 64 — length sorting reorders the encoder's input, so
this had to be proven, not assumed. The general lesson is worth keeping: an
"optimisation" that was never timed against the thing it replaced was a 3x
regression hiding behind a plausible story.

## 2026-08-06

### Resuming a run made its own curve unreadable
**What changed.** `qar.utils.logging.read_series(path, key)` returns a step-ordered
series with rewinds resolved — later record per step wins. `Trainer.maybe_resume`
writes an `{"event": "resume", "from_checkpoint": ...}` marker. `sweep.py`'s
`best_from` reads through `read_series` instead of scanning raw records.
**Why.** `metrics.jsonl` is append-only so a resumed run keeps the curve from before
the crash. The unexamined consequence: resuming rewinds to the checkpoint's step
while the interrupted run's later records stay in the file, so those steps carry two
values from two different trajectories.
**Evidence.** `runs/retriever_b128_hn2` was interrupted at step 2300 and resumed from
step 2000. Its log now holds two records each for 2100, 2200 and 2300. Before this,
`best_from` scanned every record, so a "best" could be reported from a trajectory
that was abandoned — and any plotted curve would jump backwards.
**Consequences.** That run predates the marker, so its duplicates are unlabelled;
`read_series` still resolves them correctly. The project convention "metrics are read
from `metrics.jsonl`, never transcribed by hand" now needs the second half: read them
*through `read_series`*, not with a bare `json.loads` loop.

### `val/recall@1` cannot compare two `hard_negatives` settings
**What changed.** Documented in `src/qar/tasks/CLAUDE.md` and at the top of
`sweeps/negatives.yaml`. No code change — the behaviour is correct, the reading of it
was the trap.
**Why.** `validation_step` scores the same matrix training does, so a run with
`hard_negatives=n` validates over `B + n` candidates and the extra `n` are
same-product distractors. That is a harder question, not the same question.
**Evidence.** Step 1000, batch 128: `hard_negatives=0` logged `val/recall@1` 0.1611,
`hard_negatives=2` logged 0.1145. Taken at face value that says hard negatives hurt,
which is very likely backwards.
**Consequences.** The negatives ablation is only interpretable through
`evaluate_retrieval.py`, where every cell meets the same ~9-candidate pool however it
was trained. This is the second time today the in-batch metric would have produced a
wrong conclusion; it should be treated as a training-health signal and nothing more.

### Ablation grids would have ranked cells on the wrong metric
**What changed.** `sweep.py` gained an optional `evaluate:` block that scores each
cell's `best.pt` through `evaluate_retrieval.py` and adds `within_*` columns to the
results table. New config field `retrieval.out_name` so twenty cells all scored as
`dense` write to separate files instead of overwriting one another. Six real grids
added: `negatives`, `temperature`, `dedup`, `pooling`, `data_scaling`, `multitask`
(19 cells).
**Why.** `best_from` reads `metrics.jsonl`, whose `val/recall@1` is the *in-batch*
number. Today's run scored 0.4998 there and 0.1707 within-product — so a grid ranked
on the logged metric would have selected its winner on the easy problem and handed
the DL report a table of the wrong measurement.
**Consequences.** `within_` prefixes keep the two metrics visually separable in one
row. The in-batch monitor is retained as a training-health signal, not as the
ranking criterion.

### `score_batch`: dense evaluation stops being the bottleneck
**What changed.** `Retriever.score_batch` on the base class, defaulting to the
existing per-row loop so every lexical baseline is byte-for-byte unaffected.
`DenseRetriever` overrides it to encode a chunk of queries and all their pooled
documents in two forward passes. `evaluate_retriever` now groups rows by
`retrieval.batch_rows` (default 256).
**Why.** Scoring one row at a time took 18.5 min for whole validation — at 19 grid
cells that is more wall time evaluating than training.
**Evidence.** `test_dense_batched_matches_per_row` asserts the batched path returns
the per-row values to `abs=1e-4`, and `test_batch_rows_does_not_change_the_metrics`
runs the full evaluation at chunk 1, 3 and 256 and requires identical metrics.
Batching pads documents from different rows to a common length, so this also
exercises the masked-pooling invariant end to end.
**Consequences.** Speedup not yet measured on GPU — the hard-negative run holds
6,425 MiB of 8,151 and a concurrent evaluation risked OOMing a multi-hour job.
Correctness is established; the timing number is outstanding.

### Hard negatives are capped at 2 on 8 GB, and 4 fails silently
**Evidence.** Probed at batch 128, 30 steps each: `hard_negatives=2` runs at
2.32 steps/s using 6,425 MiB; `hard_negatives=4` reports **9,757 MiB against 8,151
physical**, does not raise, and collapses to 0.16 steps/s — ~36 h for 20k steps.
**Why it matters.** Windows lets the allocation spill into system RAM rather than
raising `CUDA out of memory`, so the failure mode is a run that looks healthy and
would have wasted a day. Recorded in `sweeps/negatives.yaml` so nobody adds a 4.
**Consequences.** The negatives grid is `[0, 1, 2]`. Reaching 4 means dropping the
batch size, which also changes the in-batch negative count — so it would not be a
clean comparison and needs stating if it is ever run.

### The trained retriever loses to token overlap — and the gap says why
**What changed.** Nothing in the code; this is the first honest comparison between a
trained checkpoint and the baseline table, now that `dense` makes one possible.
**Evidence.** val, within-product, 87,475 rows, mean pool 9.33, `best.pt` at step
20000: **dense recall@1 0.1707**, mrr 0.3728. That is fifth of seven — above
`random` 0.1305 and `first` 0.1282, below `bm25` 0.1771, `bm25_global` 0.1870,
`bm25_noidf` 0.1942 and `overlap` 0.2149. The same checkpoint reports in-batch
`val/recall@1` 0.4998.
**Why the two disagree.** They are different problems. In-batch ranks against
candidates from *other products*, so winning needs only topical discrimination.
Within-product ranks against ~9 snippets sharing all of the product's vocabulary, so
winning needs relevance. This run had `loss.hard_negatives=0`, meaning the model was
trained solely on the easy discrimination and never saw the hard one — and it learned
precisely what it was shown.
**Consequences.** The headline claim for the DL report is not "the retriever works".
It is that **in-batch InfoNCE alone does not learn within-product relevance on this
corpus**, with the 0.4998/0.1707 pair as the evidence. That makes hard negatives the
next experiment rather than an optional refinement — `loss.hard_negatives` already
draws them from the row's own product pool, so no mining pass is needed to test it.
Quoting 0.4998 anywhere near the baseline table would be a serious misreport.
**Cost.** 18.5 min to score 87,475 rows (~79 rows/s), faster than the 37 rows/s the
2,000-row probe suggested.

### First real training run: the retriever learns
**What changed.** `runs/retriever_b128` — the from-scratch bi-encoder trained on GPU
for the first time. 20,000 steps, batch 128, 45.97M parameters, 64.7 min.
**Why.** Everything before this was wiring verified on CPU. `torch.cuda.is_available()`
is now True in the dev shell, so the item the root `CLAUDE.md` listed as blocked no
longer is.
**Evidence.** train loss 4.85 (= ln 128, chance) → 1.4268, train acc → 0.6197.
In-batch `val/recall@1` 0.1611 (step 1k) → 0.29 (2k) → 0.4185 (5k) → 0.4805 (10k) →
0.4998 (20k), still creeping at the end rather than turning over. 5.35 steps/s,
3,184 MiB allocated of 8,151.
**Consequences.** The curve is still rising at 20k steps, so `train.max_steps` is
cutting the run short rather than the model having converged — worth an extension
before reading anything into the final number. **The 0.4998 is in-batch recall over
128 candidates from other products and must never appear beside the baseline
table**; the comparable figure comes from `dense`.
**Batch probe:** batch 128 → 4.8-5.7 steps/s at 3,184 MiB; batch 256 → 1.6 steps/s
at 5,730 MiB. 256 fits in 8 GB but needs ~3.5 h for 20k steps, so more negatives is
a deliberate later ablation, not a free upgrade.

### Evaluating one retriever used to delete the whole baseline table
**What changed.** `merge_results` in `src/qar/retrieval/evaluate.py` folds fresh rows
into the existing `runs/_baselines/<split>.json` instead of replacing it. Rows
measured over a different row count are dropped, loudly.
**Why.** The results file is named after the split alone. Running
`evaluate_retrieval.py --set retrieval.baselines=[dense]` therefore overwrote six
lexical rows with one dense row.
**Evidence.** It actually happened: a 2,000-row `dense` smoke test reduced
`runs/_baselines/val.md` to a single line, discarding a table that had just cost
minutes to compute. Five tests now cover merge, supersede, stale-drop and a corrupt
file.
**Consequences.** "Measure the baselines once, add the trained model later" is now
the normal workflow rather than a footgun. The stale-drop rule matters as much as
the merge: a `max_rows` probe sitting unlabelled beside a full-split number is worse
than a missing row.

### `dense`: the trained bi-encoder, scorable against the baseline table
**What changed.** `src/qar/retrieval/dense.py` registers a `dense` retriever that
loads a checkpoint and scores within-product pools through the same
`evaluate_retriever` path as every baseline. New config field
`retrieval.checkpoint`.
**Why.** There was no way to compare a training run with the table it exists to
beat. `val/recall@1` in the training log ranks against `eval_batch_size` candidates
from *other* products — an easier problem over a different candidate count — so it
could never be quoted against `overlap 0.2149`. Without `dense`, a finished training
run produced no reportable number at all.
**Evidence.** Five tests in `tests/test_retrieval.py`, including one that saves a
checkpoint at `d_model=32, n_layers=1` and evaluates it under a config demanding
`384/6`: the architecture must come from the checkpoint's snapshotted config, since
a mismatched `pooling` or `max_query_len` would load cleanly and rank nonsense.
**Consequences.** Found a real bug in the process: nothing on the evaluation path
imported `qar.models`, so `build("model", ...)` raised `unknown model 'biencoder'`.
`dense.py` now imports it for the registration side effect. Still within-product
(~9 candidates); a global-index evaluation over 6.5M snippets remains a separate,
harder measurement with no entry point yet.

### Baselines regenerated on whole validation; conclusions survive
**What changed.** `runs/_baselines/val.{json,md}` rebuilt over 87,475 rows, and the
tables in the root `CLAUDE.md` and `src/qar/retrieval/CLAUDE.md` updated.
**Evidence.** val, within-product, mean pool 9.33: random 0.1305, first 0.1282,
bm25 0.1771, bm25_global 0.1870, bm25_noidf 0.1942, **overlap 0.2149** recall@1.
Against the 43,801-row measurement every value moved by less than 0.005 and the
ordering is identical.
**Consequences.** The BM25-loses-to-overlap finding was not an artefact of the
smaller sample, and the decomposition holds: corpus IDF over pool-local +0.0099, no
IDF at all a further +0.0072, F1 over summed saturated tf +0.0207. `first` now sits
marginally *below* `random` (0.1282 vs 0.1305), which sharpens rather than changes
the reading: the pool carries no usable ordering whatsoever.

### Use the corpus's own test split instead of carving one out of validation
**What changed.** `data.test_path` (default `test-qar_all.jsonl`) makes each raw
file its own pass in `prepare`, so validation is no longer divided in half.
Setting it to null restores the old asin-hashed carve; `manifest.test_source`
records which regime ran, and `prepare.test_fraction` is written as null when it
did not apply. A configured-but-missing test file raises rather than silently
falling back.
**Why.** `test-qar_all.jsonl` was sitting unused in the repo root. Carving test out
of validation was a workaround for a file we already had.
**Evidence.** Measured on the raw files: test is 92,726 rows over 15,599 products,
overlapping train on 116 products (0.74%) and validation on 15 (0.10%) — the same
profile the train/validation pair already had (103 shared, 0.7%). Question-text
overlap train↔test is 7,031 (7.8%), matching the 7.5% already known for
train↔validation and generic-phrasing in origin, not leakage.
**Consequences.** Validation roughly doubles and the test set becomes the canonical
one, so results are comparable with published AmazonQA work. **Every baseline number
measured before today is superseded.** `asin_overlap_val_test` is no longer
0-by-construction but a small inherited 15; the manifest reports it rather than
asserting zero.
**Corpus rebuilt** 2026-08-06 in 10.8 min: train 704,201 kept, val 87,475 (was
43,801), test 88,182. Unique products 123,616 / 15,496 / 15,509. Mean positive score
0.262 / 0.260 / 0.261 — the distant-supervision ceiling is unchanged, as expected,
since the selector did not move. All three splits reconcile exactly against the raw
counts (737,214+1,562 = 738,776; 92,037+146 = 92,183; 92,611+115 = 92,726), and the
three leakage figures match the raw files exactly, so the pipeline introduced none.

### Raw corpus filenames: config now names the files that exist
**What changed.** `data.train_path` / `data.val_path` default to `train-qar.jsonl`
/ `val-qar.jsonl`, and `data.ipynb` downloads under those same upstream names —
streamed in 1 MiB chunks, skipping any file already present.
**Why.** The config named `amazonqa_train.jsonl` while the folder held
`train-qar.jsonl`, so `prepare_data.py` could not find its input at all. The
notebook also buffered each whole file in memory via `response.content` before
writing a byte.
**Consequences.** `tests/test_config.py` asserts on the new name; the coupling is
listed in `tests/CLAUDE.md`.

### Tests could silently read the 751 MB corpus, and now cannot
**What changed.** `tests/conftest.py` gained an autouse fixture that wraps `open`
and raises if any test touches `train-qar.jsonl`, `val-qar.jsonl` or
`test-qar_all.jsonl`. `tests/test_retriever_task.py` and `tests/test_data.py` now
pass `data.test_path=null` explicitly.
**Why.** Giving `data.test_path` a real default was enough to break the rule
`tests/CLAUDE.md` had stated in prose since the folder was written: every prepare
helper that did not override it fell through to the repo root and began reading the
real test file.
**Evidence.** The suite went from seconds to over 600 s and was killed before
finishing; with the fixture the same mistake fails immediately, naming the file and
the fix.
**Consequences.** A prose rule with no enforcement was worth nothing here. Any new
prepare helper must point at a `tmp_path` fixture or null the path.

### Project agent roster and permission rules
**What changed.** `.claude/` now holds six subagent definitions (`qar-explorer`,
`qar-planner`, `qar-data-engineer`, `qar-ml-scientist`, `qar-validator`,
`qar-reviewer`) and a `settings.json` permission allowlist. Added this changelog
and the read-first rule to all eleven `CLAUDE.md` files.
**Why.** The folder arrived holding Codex-format `.toml` agents written for an
unrelated flight-demand project. Claude Code only loads `.md` with YAML
frontmatter, so all six were inert.
**Consequences.** `.claude/` is checked in, so the agents and the permission rules
travel with the repo and the read-first rule is always satisfiable. It also means
permission rules are shared: an `allow` entry added here applies to everyone
working on the repo.

### `top_sentences_IR` is not a usable baseline row
**What changed.** Removed the claim from `CLAUDE.md`, `README.md` and two folder
contracts.
**Why.** It was assumed to be a reranking of `review_snippets`. It is not.
**Evidence.** Over 3,000 validation rows, 7.2% of its sentences appear in
`review_snippets`; 54% of rows share none.
**Consequences.** Scoring it against `positive_idx` would compare two different
candidate sets. Its honest use is as an alternative evidence set for the NLP
report's generator, not as a retrieval row.

## 2026-08-04

### From-scratch bi-encoder, InfoNCE, hard negatives (`55b8d67`)
**What changed.** Pre-norm `TextEncoder`, two-tower `BiEncoder` (~46M params),
`RetrieverTask`, same-product hard negatives, `data.val_subset`.
**Evidence.** 107 tests; end-to-end on the real corpus at 704,201 rows over
638,306 distinct questions.
**Consequences.** **Never trained on a GPU** — `torch.cuda.is_available()` is
False in the dev shell. Wiring is verified; training is not.

### Two latent hangs fixed
**What changed.** `QGroupBatchSampler` raises when distinct questions < batch
size; `Trainer._endless` raises on an empty loader.
**Why.** The combination yielded zero batches, which the endless cycle turned
into a silent infinite loop that looked like a slow first step. It hung the suite
for ten minutes before being noticed.

### BM25 loses to plain token overlap (`1f33456`)
**What changed.** Six lexical baselines and a within-product evaluation harness.
**Evidence.** val, 43,801 rows, mean pool 9.32 — `overlap` recall@1 **0.2145**
beats `bm25` **0.1768**. Decomposed one variable at a time: corpus-wide IDF over
pool-local +0.0093, removing IDF entirely +0.0079, F1 form over summed saturated
tf +0.0205.
**Consequences.** Within one product the discriminative terms are the *common*
ones — the product's own features — and IDF promotes rare tokens instead. The
number the learned retriever must beat is 0.2145, not any BM25 row. Also measured:
`first ≈ random`, so the snippet pool carries no ordering to exploit.

### Corpus preparation pipeline (`0ffbca9`)
**What changed.** Distant-supervised positives, byte-level BPE, asin-hashed
val/test split, offset-indexed dataset, de-duplicating batch sampler.
**Evidence.** 704,201 / 43,801 / 43,674 rows kept; rows read plus 1,708
unparseable reconcile exactly to 738,776 and 92,183; `asin_overlap_val_test` = 0.
**Consequences.** **Mean positive score is 0.262** — the retriever is trained
towards a noisy target, and that ceiling belongs in the DL report as a stated
limitation, not a footnote.

### `.gitignore` was hiding source code (`1563b2e`)
**What changed.** `data/` → `/data/`, `runs/` → `/runs/`.
**Why.** An unanchored `data/` also matched `src/qar/data/`, silently excluding
the entire corpus-preparation package. Nothing warned; the files simply never
appeared in `git status`.

## 2026-07-31

### `train/*` scalars were inflated by `grad_accum` (`bcfc7b2`)
**What changed.** Each micro-batch now contributes its `1/grad_accum` share.
**Why.** Totals were accumulated per micro-batch but divided by the number of
optimisation steps, so every `train/*` scalar read `grad_accum`× too high.
Invisible at the default `grad_accum=1`; `val/*` was never affected.
**Consequences.** Any curve logged before this from a run with `grad_accum > 1`
is wrong and must be regenerated.

### Per-folder `CLAUDE.md` contracts (`4c3bf00`)
**What changed.** Every code folder states its own inputs, outputs and boundary
rules.
