# Plan: Adversarial Variants (Thread 2) and Decomposition (Thread 3)

Context: `report/Report.pdf` §3.5–3.9 (variants, position controls, metrics, decomposition, statistics). The baseline (Thread 1) is in `plan.md`; its LLM runs are in progress on ADA. Design decisions go to `design.md`.

## Status
| Step | Status |
|---|---|
| Variant pools A2, A3, A4, A6, A1, A1-ctrl, A5 (n = 1, 2, 5, 10) | ✅ Built (`code/coldrank/adversarial/`); all invariants pass on every pool |
| Schema additions (`role`, `new_dated`, `shuffle_key`), Trap@1, renderer change | ✅ Done; baseline pools and prompts unchanged (verified byte-identical) |
| Heuristic sanity table | ✅ Matches expectations except A1-ctrl (finding, see §3) |
| `--conditions` for multi-condition jobs | ⏳ Not done (cluster convenience) |
| LLM runs on variants | ⏳ Next |
| `eval/decompose.py` (Thread 3) | ⏳ Next |

**Built counts:** A2/A3/A4 1,900 pools each; A6 1,898 (1 instance dropped); A1 family 1,111 pools per condition (39 of 1,150 old instances dropped: too few low-similarity traps).

**Goal:** build the variant pools A1–A6 and A1-ctrl from the existing Amazon Video Games pools and instances, run the same models and methods on them, and compute the recency-prior decomposition. Each variant changes **one factor** relative to its reference, so recency and relevance can be separated.

## 0. What already exists and is reused

| Piece | Where | Reused for |
|---|---|---|
| Baseline pools (950 instances × random/hard) | `data/processed/amazon_vg/split_T2022-09-01_w180/pools/baseline_*.jsonl` | A2, A3, A4 (rewrite dates), A6 (same instances) |
| Old-target instances (150 dev / 1,000 test) | `split_T…/old_{dev,test}.jsonl` | A1, A1-ctrl, A5 |
| SASRec top-100 per instance (cold *and* old instances) | `split_T…/sasrec_topk.jsonl` | Warm slots of A1 / A1-ctrl / A5 |
| Item embeddings, launch times, exclusion rules | `pools/build.py` (`Embeddings`, `profile`, `check_pool`) | Traps, A6 negatives, A5 relabel choice |
| Displayed `date` stored separately from true `t0` | pool schema | Date variants only rewrite `date` |
| Runner, heuristics, report all take `--condition` | `llm/run.py`, `baselines/heuristics.py`, `eval/report.py` | Run any variant without new code |

**Feasibility, checked on the data:**
- **A6:** every cold instance has ≥ 49 eligible new items except **1 test instance**, which is dropped from A6.
- **A1:** median 1,700 eligible new items per old instance; about **4% of old instances** (targets bought in the first days after `T`) have too few low-similarity traps and are dropped.
- **A1 gold:** in SASRec's top 40 only **4.8%** of the time, so it is **inserted**, as the gold is in the baseline.

## 1. Variant specifications

All variants keep 50 candidates, the prompt template, the reranking method and the 3 shuffles. A2–A4 change only the displayed `date`; A6, A1, A1-ctrl and A5 change pool membership.

| Condition | Built from | Change | Negative settings | Instances (dev / test) |
|---|---|---|---|---|
| `baseline` | (exists) | 40 warm + gold + 9 cold, true dates | random, hard | 150 / 800 |
| `a2_swap` | baseline pool | Displayed dates of `g` and **one random warm candidate** exchanged | random, hard | 150 / 800 |
| `a3_hide` | baseline pool | `date = None` for all candidates **and** history items (history order kept) | random, hard | 150 / 800 |
| `a4_shuffle` | baseline pool | Displayed dates permuted across all 50 candidates; history dates unchanged | random, hard | 150 / 800 |
| `a6_allcold` | cold instances | Gold + **49 cold negatives**, same eligibility rules as the baseline | random, hard | 150 / 799 |
| `a1_oldgold` | old instances | Gold (warm) + **39 SASRec warm** + **10 traps** (new, low similarity) | trap | ≈ 144 / ≈ 958 |
| `a1ctrl` | A1 pools | The 10 trap slots replaced by **SASRec warm ranks 40–49** (no new dates) | trap | same as A1 |
| `a5_n{1,2,5,10}` | A1 pools | The **n lowest-similarity warm negatives** relabeled with new dates | trap | same as A1 |

### Details and decided defaults
Change any of these before building if you disagree; each one goes into `design.md`.

1. **A2 swap partner:** a uniformly random warm candidate (seeded by pool). Only the two `date` fields change.
2. **A3** also hides history dates (the report says "candidates and history"). The renderer already drops the field and adjusts the header when `date` is `None`.
3. **A4:** one seeded permutation per pool. The multiset of displayed dates is unchanged, so 10 new-looking dates remain, now on random items.
4. **A6 negatives:**
   - same eligibility as the baseline: launched in `(T, target_ts]`, not in `H_u` or the user's window items, no near-duplicates of the gold;
   - *random* = 49 uniform; *hard* = 49 most similar to the user profile;
   - instances with fewer than 49 eligible items are dropped and the count is logged.
5. **A1 warm slots:** the gold plus SASRec's top 39 (excluding `H_u`, the user's other window items and the gold).
6. **A1 traps (Amazon):**
   - pool: items launched in `(T, target_ts]` that the user never interacted with, near-duplicates of the gold removed;
   - **low similarity:** below the **25th percentile** of similarity to the user profile among that instance's eligible items;
   - pick 10 uniformly at random (seeded);
   - instances with fewer than 10 such items are dropped.
7. **A1 negative setting:** a single setting (`trap`) instead of random/hard, since traps are low-similarity by definition. Files are named `a1_oldgold_trap_<split>.jsonl`.
8. **A1-ctrl:** SASRec warm ranks 40–49 (after the same exclusions). The candidates in the 40 shared slots are identical to A1.
9. **A5 relabeling:**
   - which items: the n warm *negatives* (never the gold) with the lowest similarity to the profile;
   - new displayed date: copied from a randomly chosen eligible cold item, so dates are realistic and fall in `(T, target_ts]`;
   - `n = 10` doubles the number of new-looking items (10 traps + 10 relabeled).
10. **A5 base:** builds on A1 (traps kept), as in the report.

## 2. Schema additions (backwards compatible)

New candidate fields (absent = old behaviour):
- `role`: one of `gold`, `warm`, `cold_neg`, `trap`, `relabeled`, `ctrl_fill`.
- `new_dated`: displayed date later than `T`. Computed when the pool is built, so Cold@1 and Trap@1 follow what the model *sees*.

New pool fields:
- `shuffle_key`: the reference pool's ID.
  - A2, A3, A4 → the baseline pool ID.
  - A1-ctrl and A5 → the A1 pool ID.
- `reference_pool_id` and `condition`.

**Why `shuffle_key`:** today the shuffle is seeded by `pool_id`. Variant pools have new IDs, so the candidate order would change *together with* the dates, and the comparison would no longer change one factor. The renderer will use `pool.get("shuffle_key", pool["pool_id"])`. Baseline pools have no `shuffle_key`, so the **runs already in progress are unaffected**.

**A6** keeps its own seed: its candidates differ from the baseline's, so there is no order to match.

## 3. Code to write

| File | What |
|---|---|
| `code/coldrank/pools/variants.py` | `make_a2/a3/a4(pool)` (pure date rewrites); `build_a6(...)`; `build_a1_family(...)` returning A1, A1-ctrl, A5 (n = 1, 2, 5, 10); `check_variant(pool, ref)` invariants; CLI writing `pools/<condition>_<neg>_<split>.jsonl` and `variant_stats.json` |
| `code/coldrank/prompts/render.py` | Shuffle by `shuffle_key` (one line) |
| `code/coldrank/eval/metrics.py` | `cold1` uses `new_dated` when present; new `trap1` (top-1 has role `trap` or `relabeled`; for A1-ctrl, role `ctrl_fill`) |
| `code/coldrank/eval/decompose.py` | Thread 3 table: ∆evidence, ∆recency, S_swap, trap effect, flood slope, Cold@1/Trap@1 with paired-bootstrap CIs and McNemar p-values (§5) |
| `code/coldrank/llm/run.py` | `--conditions a2_swap,a3_hide,…` to run several pool files in one job (the model loads once) |
| `code/scripts/submit.sh` | Stages `adv-direct`, `adv-reasoning`, `adv-test` |
| `code/tests/test_variants.py` | Invariants below, determinism, A2/A3/A4 change nothing except dates |

### Invariants (`check_variant`, asserted for every pool)
- **All:** 50 unique candidates, exactly 1 gold, none in `H_u`, every candidate available at `target_ts` (true `t0 ≤ target_ts`).
- **A2:**
  - same candidates and true `t0` as the baseline;
  - exactly two displayed dates differ: the gold's and one warm item's;
  - the gold's displayed date is ≤ `T`.
- **A3:** every candidate and history `date` is `None`; all other fields equal the baseline.
- **A4:** the multiset of displayed dates equals the baseline's; nothing else changes.
- **A6:** 50 candidates with `t0 > T`; the gold is among them.
- **A1:**
  - the gold is warm (`t0 ≤ T`);
  - exactly 10 traps with `t0 > T` and similarity below the threshold;
  - 39 other warm items.
- **A1-ctrl:** no `new_dated` candidate; 40 slots shared with A1.
- **A5:**
  - exactly 10 + n `new_dated` candidates;
  - the relabeled items are the n lowest-similarity warm negatives;
  - the gold is never relabeled.

### Manipulation checks (`variant_stats.json`, reported in the paper)
- **A1 / A5:**
  - age gap between the gold and the traps (median days);
  - content similarity to the history: gold vs traps vs warm negatives;
  - fraction of instances dropped.
- **A2:** age difference between the gold's true and displayed dates.
- **A6:** gold vs negative similarity (random and hard), instances dropped.

### Heuristics as a sanity check (run before any LLM)
The date heuristics read displayed dates, so they must react in predictable ways:

| Condition | Expected heuristic behaviour |
|---|---|
| A2 | Random-among-newest falls from ≈ 0.10 to ≈ 0 (the gold no longer looks new) |
| A3 | Every date heuristic falls to random (≈ 0.02) |
| A4 | Date heuristics fall to ≈ 0.02 |
| A6 | Date heuristics ≈ 0.02 (everything is new) |
| A1 | Date heuristics ≈ 0 (the gold is old); Trap@1 of newest-first = 1 |
| A1-ctrl | No new dates, so date heuristics fall back to hash order |

If a heuristic does not behave as in this table, the variant is built wrong. Fix it before spending GPU time.

**Result (test split):** every row matched except A1-ctrl.
- **A2:** random-among-newest and newest-first fell to 0.000.
- **A3 / A4 / A6:** date heuristics fell to about 0.02.
- **A1 / A5:** newest-first scored Recall@1 = 0 with Trap@1 = 1.0.
- **A1-ctrl:** newest-first reaches **0.413**. Among older items, the one the user buys next tends to be the most recently launched, so A1-ctrl is not recency-neutral. A1 still works as intended, because the traps are newer than the gold (median 706 days). The trap effect (A1-ctrl − A1) must be read with this in mind; report it in the paper.

## 4. Runs on ADA

Same models, prompts (frozen after baseline dev), shuffles and decoding as the baseline. One job per (model, method) loops over all its condition files, so each model loads once.

| Stage | Methods | Models | Conditions | Split |
|---|---|---|---|---|
| `adv-direct` | Direct-Rec, pointwise | 4 listwise (Direct-Rec) + 5 (pointwise) | All: A2, A3, A4, A6 (random, hard), A1, A1-ctrl, A5 × 4 | dev, then test |
| `adv-reasoning` | Base, Fast, Structural, SSC | 4 listwise | Core: A2, A3 (random, hard), A1, A1-ctrl | dev, then test |
| `adv-norecency` | Structural without the recency instruction | 4 listwise | A2 (random, hard) | dev, then test |

This follows the report: Direct-Rec and pointwise run on every condition; the reasoning methods on a subset; Structural without recency on the baseline and A2. Order of work:
1. Build and check the pools on CPU.
2. Run the heuristics.
3. Run Direct-Rec and pointwise on dev.
4. Read the outputs.
5. Run test.
6. Run the reasoning methods.

## 5. Decomposition and statistics (Thread 3)

Computed per model × method × negative setting by `eval/decompose.py`. `R` is Recall@1 on the 800 cold test instances unless stated otherwise.

| Quantity | Formula | Pairing / test |
|---|---|---|
| ∆evidence | R_hide − R_random (R_random = 1/50) | Bootstrap CI over instances |
| ∆recency | R_true − R_hide | Paired (same instances): paired bootstrap + McNemar |
| Swap sensitivity S_swap | R_true − R_swap | Paired |
| Shuffle effect | R_true − R_shuffle | Paired |
| All-cold effect | R_true − R_allcold | Paired (799 instances) |
| Trap effect | R_A1-ctrl − R_A1 | Paired (A1 instances) |
| Raw over-crediting gap | R_baseline − R_A1 | **Unpaired** bootstrap (different instances) |
| Flood slope | Slope of Recall@1 (and Trap@1) on n ∈ {0, 1, 2, 5, 10}, with n = 0 = A1 | Bootstrap CI of the slope |
| Cold@1, Trap@1 | Per condition | Bootstrap CI; Cold@1 against chance |
| Domain × condition | Logistic regression of per-instance success | After EB-NeRD exists |

**Reading the results** (report §3.8): a relevance-driven reranker shows a large ∆evidence, ∆recency ≈ 0, a small S_swap and trap effect, and a flat flood slope on Amazon. A recency-driven one shows the opposite.

## 6. Also in the report, not yet built (later)
- **Exposure check (§3.3):** ask each model for each item's release year from its title; label items as known/unknown; report results separately. One extra pointwise-style job per model.
- **Residual position effect (§3.6):** Recall@1 by the gold's position in the prompt. Computable from the existing `raw.jsonl` (`id_map`), no new runs.
- **EB-NeRD:** the same variants, with traps = in-view but not-clicked articles.

## 7. Work split (suggested)
- **Thread 2:** `variants.py`, schema and renderer change, invariants, manipulation checks, heuristic sanity table.
- **Thread 3:** `decompose.py`, statistics, the paper's decomposition table and flood-slope figure.
- **Whoever runs ADA:** add the `adv-*` stages to `submit.sh` once the pools pass the checks.
