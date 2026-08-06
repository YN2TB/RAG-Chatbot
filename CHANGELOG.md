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

## 2026-08-06

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
