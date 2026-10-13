# Relevance or Recency? Diagnosing the Temporal Shortcut in LLM-Based Cold-Start Reranking

IRE course project, IIIT Hyderabad: Vedavyas Bysani, Ayush Daga, Sanjith Ganapathi.

## The question

LLM rerankers are reported to do well on **cold-start (new) items**. But cold-start test sets are built with a temporal split, so the correct item is always one of the *newest* candidates. A reranker that simply prefers new items scores well without reading any content.

**When an LLM reranker succeeds on new items, how much of that is semantic relevance, and how much is a preference for recency?**

We answer this in two steps:
1. Reproduce the Netflix cold-start reranking protocol of Li et al. (WWW '26) on public data.
2. Apply adversarial variants that separate recency from relevance: an old correct item with new "trap" items, and dates that are hidden, swapped, shuffled or flooded.

The domains are Amazon Video Games (static catalogue, where recency should not matter) and EB-NeRD news (volatile, where recency is a real signal).

## Repository

| Path | What it is |
|---|---|
| [`report/`](report/) | LaTeX write-up; [`report/Report.pdf`](report/Report.pdf) is the current version |
| [`plan.md`](plan.md) | Implementation plan for the baseline, with done / pending status per phase |
| [`plan_adversarial.md`](plan_adversarial.md) | Plan for the adversarial variants (A1–A6) and the decomposition (Threads 2–3) |
| [`design.md`](design.md) | Every design decision with its reason (§11 is the implementation log) |
| [`code/`](code/) | The `coldrank` pipeline: split, warm ranker, pools, prompts, LLM inference, evaluation, tests. See [`code/README.md`](code/README.md) |
| [`readings/`](readings/) | Key papers: Li et al. (the baseline), Fang et al. (closest prior work), Dong et al. (decomposition and settings) |
| `data/`, `outputs/` | Datasets and run outputs. **Not in git**; rebuilt with the commands in `code/README.md` |

## Status

| Part | Status |
|---|---|
| Amazon Video Games data, audit and global-cutoff split | ✅ Done: cutoff 2022-09-01, 180-day window, 950 test instances (150 dev / 800 test) |
| Warm-start ranker (SASRec, stand-in for Netflix's ranker) | ✅ Done: validation NDCG@10 0.064; 3.7× popularity on older items |
| Candidate pools (40 warm + 10 cold, random and hard negatives) | ✅ Done: 1,900 pools, all invariant checks pass |
| Prompts for the five Li et al. methods + pointwise control | ✅ Implemented and tested; to be tuned on dev with the real models |
| LLM inference (vLLM + HF fallback, caching, shuffles) | ✅ Implemented; ⏳ not yet run on the GPU |
| Non-LLM reference rankers and evaluation | ✅ Done (results below) |
| LLM runs on ADA (2080 Ti) | ⏳ Next |
| Adversarial variant pools A1–A6 (+ A1-ctrl, A5 × 4) | ✅ Built and checked: 10 conditions, all invariants pass, heuristic sanity table below |
| Decomposition script (Thread 3) | ✅ `code/coldrank/eval/decompose.py`, validated on the heuristics |
| LLM runs on the variants | ⏳ Next |
| EB-NeRD, QLoRA SFT | ⏳ Later |

## First results

Recall@1 of non-LLM rankers on the 800 Amazon Video Games test pools. These rankers see the same 50 candidates the LLMs will see. 95% confidence intervals are about ±0.02 to ±0.03.

| Ranker | Random negatives | Hard negatives |
|---|---|---|
| Random | 0.029 | 0.024 |
| Random among the 10 newest | 0.111 | 0.102 |
| Newest first | 0.095 | 0.090 |
| Text similarity only | 0.041 | 0.009 |
| **10 newest, then text similarity** | **0.324** | 0.009 |
| SASRec / popularity | 0.000 | 0.000 |

- Dates alone give about 0.10, five times random.
- Restricting text similarity to the newest items raises it from 0.041 to 0.324.
- An LLM has to beat 0.32 (random negatives) before its cold-start performance shows anything beyond the date shortcut.
- No LLM results yet. A local CPU test with a tiny 135M model only checked that the pipeline runs; it says nothing about recency bias.

## Adversarial variants

Each variant changes **one factor** relative to a reference pool, so recency and relevance can be separated (report §3.5). All of the code is in [`code/coldrank/adversarial/`](code/coldrank/adversarial/). The pool files are written next to the baseline ones in `data/processed/amazon_vg/split_T2022-09-01_w180/pools/<condition>_<neg>_<split>.jsonl`.

| Condition | What changes | Reference | Built by | Pools (dev / test) |
|---|---|---|---|---|
| `baseline` | 40 warm + gold + 9 cold; true dates | — | `pools/build.py` → `build_pools()` | 150 / 800 per neg |
| `a2_swap` | Displayed dates of the gold and one random warm item exchanged | baseline | `adversarial/date_variants.py` → `make_a2_swap()` | 150 / 800 per neg |
| `a3_hide` | No dates on candidates **or** history | baseline | `date_variants.py` → `make_a3_hide()` | 150 / 800 per neg |
| `a4_shuffle` | Displayed dates permuted across all 50 candidates | baseline | `date_variants.py` → `make_a4_shuffle()` | 150 / 800 per neg |
| `a6_allcold` | Gold + 49 cold negatives (random / hard); recency no longer separates the gold | baseline instances | `adversarial/allcold.py` → `build_a6()` | 150 / 799 per neg |
| `a1_oldgold` | Old (warm) gold + 39 SASRec warm items + 10 **traps** (new, low-similarity items) | old-target instances | `adversarial/oldgold.py` → `build_oldgold()` | 144 / 967 |
| `a1ctrl` | A1 with the 10 trap slots refilled by SASRec warm ranks 40–49 | A1 | `oldgold.py` → `build_oldgold()` | 144 / 967 |
| `a5_n{1,2,5,10}` | A1 + the *n* least similar warm negatives relabeled with new dates | A1 | `oldgold.py` → `build_oldgold()` | 144 / 967 each |

*neg* = `random` and `hard` for baseline, A2–A4 and A6, and `trap` for the A1 family.

### How it fits together

| File | Function | Role |
|---|---|---|
| `adversarial/common.py` | `derive()` | Copies a reference pool as a new condition. It keeps the candidate order slot by slot and sets `shuffle_key` to the reference, so the prompt shuffles put shared candidates in the **same positions**. |
| | `annotate()` | Sets each candidate's `role` (`gold`, `warm`, `cold_neg`, `trap`, `relabeled`, `ctrl_fill`) and `new_dated` (does its *displayed* date fall after the cutoff?). |
| | `extended_embeddings()` | Reuses the baseline item embeddings unchanged and encodes only missing items into `item_embeddings_adv.npz`. |
| `adversarial/checks.py` | `check_date_variant()`, `check_a6()`, `check_a1()`, `check_a1ctrl()`, `check_a5()` | Invariants asserted on **every** pool. Examples: A2 changes exactly two dates; A3 leaves no date anywhere; A4 keeps the same set of dates; A1 traps are new and below the similarity threshold; A1-ctrl has nothing new-dated; A5 relabels exactly the *n* least similar warm items. |
| | `manipulation_stats()` | Similarity and age statistics per condition, written to `pools/variant_stats.json`. |
| `adversarial/build.py` | `main()` | Builds everything, runs the checks and writes the files: `python -m coldrank.adversarial.build` (or `--only a2_swap,a3_hide`). |
| `pools/build.py` | `eligible_cold()`, `cold_catalogue()`, `history_records()` | Eligibility rules shared with the baseline: a new negative must have launched between the cutoff and the purchase, must not be one of the user's items and must not be a near-duplicate of the gold. Extracted unchanged; the baseline pools regenerate byte-identical. |
| `prompts/render.py` | `shuffled_candidates()` | Shuffles by `shuffle_key` when present. Baseline pools have no key, so baseline prompts are unchanged. |
| `eval/metrics.py` | `pool_metrics()` | **Cold@1** counts items that *look* new (`new_dated`). **Trap@1** = the top item is a trap or relabeled item (on A1-ctrl: a fill item). |

Settings are in `code/configs/amazon_vg.yaml` under `adversarial:`. Unit tests are in `code/tests/test_adversarial.py`.

### Sanity check: non-LLM rankers on each variant (test, Recall@1)

| Condition | Random | Random among 10 newest | Newest first | Text similarity | 10 newest + text |
|---|---|---|---|---|---|
| baseline (random neg.) | 0.029 | 0.111 | 0.095 | 0.041 | 0.324 |
| A2 swap | 0.024 | **0.000** | **0.000** | 0.041 | **0.000** |
| A3 hide | 0.022 | 0.022 | 0.022 | 0.041 | 0.041 |
| A4 shuffle | 0.024 | 0.031 | 0.021 | 0.041 | 0.028 |
| A6 all-cold | 0.026 | 0.018 | 0.040 | 0.110 | 0.061 |
| A1 old-gold | 0.022 | 0.000 | 0.000 (picks a trap 100%) | 0.039 | 0.000 |
| A1-ctrl | 0.018 | 0.068 | **0.413** | 0.033 | 0.068 |
| A5 (any n) | ≈ 0.02 | 0.000 | 0.000 (picks a trap 100%) | 0.039 | 0.000 |

- Date-based rankers collapse exactly where the dates stop pointing at the gold (A2, A3, A4, A6, A1, A5), and text similarity is unaffected by date-only changes. This confirms the variants are built correctly.
- **Manipulation checks:** traps are much less similar to the user than the gold (0.30 vs 0.52) and about 2 years newer (median 706 days).
- **Finding (A1-ctrl):** among *older* items too, the item a user buys next tends to be the most recently launched one; newest-first gets it right 41% of the time. Recency therefore points to the gold even within the warm items, and the trap effect (A1-ctrl − A1) includes this.

## Getting started

```bash
# 1. Environment (cluster: Python 3.10 + CUDA; see code/requirements.txt)
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# 2. Data + pipeline: run from code/, in the order given in code/README.md
cd code
python -m coldrank.data.amazon_vg        # after downloading the raw files
python -m coldrank.data.split
python -m coldrank.warm.sasrec
python -m coldrank.pools.build
python -m coldrank.baselines.heuristics --split test
python -m coldrank.eval.report --split test
python -m coldrank.adversarial.build      # all variant pools (CPU, a few minutes)
python -m coldrank.baselines.heuristics --condition a2_swap --split test

# 3. LLM runs (GPU)
python -m coldrank.llm.run --model llama2_7b --method direct --split dev --neg random

# Tests
python -m pytest -q -m "not slow"
```

Things to know:
- Configs live in `code/configs/` (`amazon_vg.yaml` for the protocol, `models.yaml` for the pinned model checkpoints).
- Set `HF_HUB_OFFLINE=1` once models are downloaded. Model loading can otherwise hang on a stalled Hugging Face connection.
- On the dev machine, `USE_TF=0` is already set inside the package; it avoids a TensorFlow/Keras clash with `transformers`.

## Decomposition (Thread 3)

`python -m coldrank.eval.decompose --split test` reads every `preds.jsonl` (LLMs and heuristics) and writes `outputs/amazon_vg/decomposition_test/{rates.csv, decomposition.csv, summary.md}`. It is implemented in `code/coldrank/eval/decompose.py`: `load_outcomes()` → `decompose()` → `summary_md()`; the statistics are in `code/coldrank/eval/stats.py`.

| Quantity | Definition | Test |
|---|---|---|
| ∆evidence | R_hide − 1/50 | Bootstrap CI |
| ∆recency | R_true − R_hide | Paired bootstrap + McNemar |
| Swap sensitivity | R_true − R_swap | Paired |
| Shuffle / all-cold effect | R_true − R_shuffle, R_true − R_allcold | Paired |
| Trap effect | R_A1-ctrl − R_A1 | Paired |
| Flood slope | Slope of Recall@1 over n = 0 (A1), 1, 2, 5, 10 | Bootstrap over instances |
| Flood lift | How much more often a relabeled item is ranked first once it looks new (0 for date-blind rankers) | Paired |
| Over-crediting gap | R_baseline − R_A1 | Unpaired (different instances) |

Validation on the heuristics (test, random negatives): text similarity, which never reads dates, has ∆recency = 0. "10 newest + text" has ∆recency = +0.282 and swap sensitivity = +0.324 (both p < 0.05).

## Next steps

1. **Baseline LLM runs** (in progress): dev, then test, once prompts are frozen.
2. **LLM runs on the variants:** Direct-Rec and pointwise on all conditions; the reasoning methods on A1, A1-ctrl, A2 and A3. Use `--condition <name>`, and `--neg trap` for the A1 family. Then run `python -m coldrank.eval.decompose`.
3. **Remaining code:** the exposure check (known/unknown items), the position-effect analysis, and the EB-NeRD port.
