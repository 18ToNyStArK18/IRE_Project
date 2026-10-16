# Design Decisions

Decisions for *Relevance or Recency? Diagnosing the Temporal Shortcut in LLM-Based Cold-Start Reranking*, recorded with their reasons. See `plan.md` for the implementation steps and `report/report.tex` for the write-up.

## 1. Research framing
- **Question:** when an LLM reranker succeeds on new items, how much of that is semantic relevance and how much is a preference for recency?
- **Novelty claim (narrowed):** Fang et al. (2025) already show recency bias in LLM *passage* rerankers. Our new contributions are:
  - the temporal-split confound in *cold-start recommendation* benchmarks;
  - pools where recency and relevance point in opposite directions (A1/A1-ctrl/A5);
  - the decomposition into evidence and recency;
  - the news vs. video games contrast.
- **Hypotheses H1–H5:** removed from the report (team decision).

## 2. Baseline = Li et al. (2026), the Netflix paper, only
- We reproduce their **protocol and prompting methods**, not their numbers. Their data, production ranker and prompts are not public.
- **Cold-start definition: one global cutoff `T`**, as in Li et al., not per-user. Cold item = launch time `t0 > T`. Report Eq. 1 must be updated to match.
- **Test instances:** for each user, the first interaction in `[T, T+w]` whose item is cold. Users need ≥ 5 history items before `T`.
- **Choosing `T` and `w`:** ~~latest `T` with ≥ 1,000 instances and `w` ≤ 90 days~~ — not achievable on Amazon VG (max 465 at T = 2023-01, w = 90). **Decided (team, after audit): `T` = 2022-09-01, `w` = 180 days** → 950 instances (dev 150 / test 800). See §11.
- **Pool:** K = 50 = top-40 warm items from the warm-start ranker + 10 cold items (gold + 9 negatives). Items already in `H_u` are excluded.
- **Cold negatives:** two settings, both reported:
  - *random*: uniform over eligible cold items;
  - *hard*: the 9 most content-similar to `H_u` (SBERT).
- **Warm-start ranker:** SASRec trained on data before `T`, standing in for Netflix's production ranker.
  - Hyperparameters from Dong et al.: 2 layers, 2 heads, max length 50, dropout 0.2, 64 dimensions. LightGCN is the fallback.
  - Our own implementation, because Dong's pipeline expects hand-prepared CSVs and uses a different (70/10/20) split.
- **Prompt attributes:** title, launch date, category (genre); brand/platform as well for Structural Reasoning. The history is listed in time order in the same format.
- **Methods (all five from Li et al.):**
  - Direct-Rec;
  - Base-Reason (2 worked examples);
  - Fast-Reason (2× history, 1 worked example; Li et al.'s reference method);
  - Structural Reasoning;
  - Soft Self-Consistency (k = 3 paths in one generation, reconciled softly).
- **Structural Reasoning recency instruction:** the original prompt weights by "recency", so we also run a variant without it, on the baseline and A2 pools.
- **Prompts** are rebuilt from the paper's descriptions, tuned on 200 dev instances only, then frozen. Worked examples use short 5-candidate pools taken from pre-`T` data.
- **Output:** a top-10 ranking after `FINAL:`. If nothing parses, the instance scores Recall@1 = 0 and the parse-failure rate is reported.
- **Fine-tuning:** none in the main runs (prompts are zero-shot, apart from the worked examples). QLoRA SFT is a stretch goal. GRPO is out of scope (hardware).
- **Leftover pieces of Li et al. we don't use:** AnyPlay (Netflix-only). Our Recall@1 plays the role of their Discovery metric.

## 3. Our additions on top of the baseline (not part of Li et al.)
- **Pointwise control:** score = log p(Yes) − log p(No), one candidate per prompt. It has no parse failures and no order effects, and it fits Phi-2's 2k context.
- **Position control:** 3 shuffled candidate orders per pool, combined by Borda count (following Hou et al.).
- **Reference heuristics:** random (0.02), random-among-newest (0.10), newest-first, SASRec order, popularity, content cosine, and **cold-filter + content** (the main bar an LLM has to beat).
- **Metrics:** Recall@1 (primary), NDCG@10, Cold@1, Trap@1, parse-failure rate. Everything is reported as **absolute** numbers next to the date-only baselines.

## 4. Adversarial variants
| Condition | Change | Reference |
|---|---|---|
| A1 Old-gold | Gold is warm; the 10 cold slots are traps | different instances |
| A1-ctrl | A1 with the trap slots filled by warm items | A1 |
| A2 Date swap | Dates of gold and one warm item exchanged | baseline |
| A3 Date hide | No dates on candidates **or** history | baseline |
| A4 Date shuffle | Dates permuted across all 50 | baseline |
| A5 Fresh flood | A1 + n ∈ {1,2,5,10} **lowest-similarity** warm negatives relabeled as new | A1 |
| A6 All-cold | All 50 cold (random and hard negatives) | baseline |

- A1 traps: Amazon uses low-similarity items the user never interacted with. EB-NeRD uses in-view articles that weren't clicked.
- Old-target instances are saved during the Phase 1 split, so A1 can be built later without redoing it.

## 5. Decomposition and statistics
- ∆evidence = R_hide − R_random; ∆recency = R_true − R_hide. These two add up exactly to R_true − R_random.
- ∆evidence is treated as an **upper bound** on semantic evidence, because hiding dates doesn't hide recency the content itself reveals.
- Swap sensitivity S_swap = R_true − R_swap is reported separately and is not part of the decomposition.
- Trap effect = R_A1-ctrl − R_A1 (paired). The raw baseline − A1 gap is descriptive only (unpaired bootstrap).
- Flood slope = change in Recall@1 per fresh-dated trap in A5.
- Tests:
  - McNemar and paired bootstrap (10k resamples) for paired conditions;
  - a power analysis to fix N;
  - logistic regression on domain × condition for the cross-domain claim.

## 6. Data
- **Amazon Video Games first** (team decision), using Amazon Reviews 2023 raw reviews and metadata. EB-NeRD is ported afterwards.
- **Amazon launch time:** **decided: `t0 = min(Date First Available, first review time)`**, first review time when DFA is missing. See §11.
- **Amazon positives:** **decided: rating ≥ 3.0** (Dong et al.); **5-core only on SASRec's training data**, never on cold items or test users. See §11.
- **EB-NeRD:**
  - `t0` = published_time; text = title, subtitle, category, subcategory;
  - train + validation splits only (test labels aren't public);
  - short `w` (1–3 days);
  - if the small models can't handle Danish, translate once with a fixed MT model and cache it.

## 7. Models and pretraining exposure
- **Models:** Phi-2, Llama-2-7B/13B-Chat, Mistral-7B-Instruct, Qwen-7B-Chat, all released in 2023. **Kept deliberately:** newer models might have been trained on our datasets (team decision). No modern models are added.
- **Phi-2:** pointwise only (2k context).
- **Known items:** ask each model for each item's release year; results are reported separately for items it knows and items it doesn't.

## 8. Hardware and inference
- 1× RTX 2080 Ti (11 GB, sm_75): fp16 (no bf16), no FlashAttention-2.
- 7B/13B models in 4-bit AWQ/GPTQ. **One fixed quantized checkpoint per model across all conditions.**
- vLLM (check it builds for sm_75 on ADA) with a Hugging Face transformers fallback. `max_model_len` = 4096 for every model, for parity.
- Greedy decoding, with a prompt-hash cache so jobs can resume. Reasoning methods run on a subsample of instances; Direct-Rec and pointwise run on everything.
- Context budget is about 2.6k prompt tokens (English). Danish needs a shorter history, no worked examples, or translation.

## 9. Repository
- `report/` (LaTeX), `readings/` (Li, Fang, Dong + README), `plan.md`, `design.md`.
- `.gitignore` covers LaTeX build files, Python clutter, `data/` and `outputs/`.

## 10. Open items
- ~~Update report Eq. 1 / §3.1 to the global cutoff; add `T`, `w`, history length, SASRec, and quantization as a limitation.~~ Done.
- ~~Fix `T` and `w` after the Phase 0 audit.~~ Done (§11).
- ~~Settle the Amazon launch-date rule and rating/k-core filtering.~~ Done (§11).
- Optional wording change: call the non-LLM rankers "reference heuristics" to avoid confusion with the Netflix baseline.

## 11. Implementation decision log (Amazon VG baseline)
Every choice made while implementing `code/`, with the reason. Values live in `code/configs/amazon_vg.yaml` and `code/configs/models.yaml`.

### Repository and environment
- **Code layout (team decision):** all code in `code/` (package `coldrank`, configs, tests). Data in `data/` and run outputs in `outputs/` at the root, both git-ignored.
- **Local environment:** `.venv` built on the system Python with `sentence-transformers` installed without dependencies; `USE_TF=0` is set in `coldrank/__init__.py` because the system TensorFlow/Keras 3 breaks `transformers` imports.
- **Data source:** Amazon Reviews 2023 raw files from the McAuley Lab Hugging Face mirror (the UCSD server stalled). Sizes match the server byte counts: 137,269 items, 4,624,615 reviews.

### Data audit findings (`outputs/audit/amazon_vg_audit.md`)
- Date First Available parsed for 93.6% of items (after adding the `"04M 26, 2018"` format, 0 unparseable). Release date exists for only 25.5% and is sometimes wrong, so it is not used.
- DFA is *after* the first review for 8.2% of items (median 235 days): relisted products. Hence `t0 = min(DFA, first review)`; this also guarantees by construction that no cold item appears in a pre-`T` history (asserted in the code).
- **Launch date means "listed on Amazon", not "released".** Example: the most frequent target is a PS5 console listing dated 2022-10-01 (the PS5 came out in 2020). This matches the recommender meaning of a new catalogue item but means some cold items are known products; the planned known/unknown split (§7) covers this.

### Split (`coldrank/data/split.py`)
- `T` = 2022-09-01, `w` = 180 days, rating ≥ 3.0, ≥ 5 positives before `T` (team decision after the audit; alternatives and counts in the audit and chat log).
- Positives de-duplicated per (user, item), earliest kept. Items need a title and a `t0`.
- **History = positives strictly before `T`.** Interactions inside the window before the target are not added to the history: in Li et al. every history ends before the cutoff.
- **Cold target = the user's first positive in `[T, T+w)` on a cold item.** At most one cold instance per user, so dev/test are user-disjoint.
- **Old-target instances** (first window positive on a warm item not in `H_u`) are saved now for A1.
- Result: 950 cold instances (dev 150 / test 800; `n_test` = 1000 requested), 4,328 old instances, 485 distinct targets (top-10 targets cover 19% of instances). Median history 8.5 items; median 281 days between the last history item and the target; median 56 days between the target's launch and the purchase.

### Warm-start ranker (`coldrank/warm/sasrec.py`)
- Own SASRec implementation with Dong et al.'s hyperparameters (2 layers, 2 heads, 64-d, max length 50, dropout 0.2), Adam lr 1e-3, batch 256.
- **Loss: full-softmax cross-entropy over all items** (no sampled negatives), padding ignored.
- **Validation:** each training user's last item (leave-one-out), early stopping on NDCG@10, patience 10.
- **Training data:** pre-`T` positives, iterative 5-core → 598,719 interactions, 69,809 users, 19,291 items.
- **Warm candidates** = SASRec vocabulary (≥ 5 pre-`T` positives), top 40 excluding `H_u`. Test users whose history has no in-vocabulary item fall back to popularity (flagged per instance).
- **Attention mask fix:** positions may always attend to themselves; otherwise left-padded positions produce NaN that leaks into real positions (unit-tested).
- **Embedding init N(0, 0.02)** (BERT-style) instead of PyTorch's default N(0, 1). With the default, the first-epoch loss was 26.6 against ln(19,292) = 9.87 for a uniform guess (saturated softmax). With the fix, the initial loss is 9.88 and validation NDCG@10 after one epoch went from 0.003 to 0.021.
- **CPU speed-ups that leave results unchanged:** logits are computed only at real target positions; batches are bucketed by sequence length and leading all-padding columns are trimmed. Positions are counted from the right edge, so trimming gives identical outputs (unit-tested). This took an epoch from about 4 minutes to about 60 seconds on 16 CPU cores.

### Pools (`coldrank/pools/build.py`)
- **Candidate order is neutral** (sorted by item ID) in the stored pool; the prompt shuffles. A fixed warm-then-cold order would leak the cold flag.
- **Excluded from every pool:** `H_u`, and any *other* item the user bought in the window (they would be unlabelled positives).
- **Cold negatives** must have launched in `(T, target_ts]`, i.e. be new *and* already available when the user bought the target.
- **Near-duplicate filter:** cold negatives with cosine ≥ 0.9 to the gold (other editions or bundles of the same product) are dropped.
- **Hard negatives** = the 9 eligible cold items most similar to the user profile; profile = mean embedding of the last 20 history items (= the displayed history); embeddings from `all-MiniLM-L6-v2` on `title | category path`.
- Pools store the displayed `date` separately from the true `t0`, plus `sim_to_history`, `pop_preT` and `warm_rank` for the heuristics and manipulation checks. Invariants are asserted for every pool (`check_pool`).

### Prompts (`coldrank/prompts/`)
- **Display compaction (token budget):** Amazon titles run to 200 characters, so a full candidate line costs ~61 Llama tokens (3,000 for 50 candidates). Displayed titles are cut at 60 characters at a word boundary; category paths drop the "Legacy Systems"/"… Systems" grouping levels and keep ≤ 3 levels; field labels appear once in a header (`(each item: title | launch date | category)`) instead of on every line. Result ≈ 39 tokens per line, ≈ 1,950 for 50 candidates. Pools keep the full text.
- **History is trimmed from the oldest item** until the prompt leaves the method's minimum generation room (direct 60, base 400, fast 300, structural/SSC 600 tokens), never below 5 items; overflowing prompts are flagged.
- **Worked examples are hand-written** with real pre-2019 products. In both examples the correct item is *not* the newest candidate (2nd and 4th newest of 5), and the reasoning never mentions dates (unit-tested), so the demonstrations teach no recency rule in either direction.
- **Base-Reason and Fast-Reason** include one neutral guideline sentence (platform, item type, franchise, brand), standing in for Li et al.'s unpublished "detailed guidelines".
- **Structural Reasoning:** Li et al.'s weighting factors are "prominence, action type and recency". We have only one action type (purchase), so "action type" becomes "the kind of interaction". The `structural_norecency` variant removes the clause; both share an exemplar justified by prominence only. Brand is shown only for Structural Reasoning. Only candidates matching a path are scored (not all 50), to fit the output budget.
- **Soft Self-Consistency:** k = 3 perspectives in one generation, reconciled softly.
- **Direct-Rec** uses the assistant prefill `FINAL: [` to improve parse rates.
- **Parsing:** IDs after the last `FINAL:`; fallback = the last line with ≥ 2 bracketed IDs; IDs de-duplicated and range-checked; the parse mode is logged.

### Inference (`coldrank/llm/`)
- **Pinned checkpoints:** TheBloke AWQ for Llama-2 7B/13B and Mistral-7B-Instruct-v0.1, the official Qwen-7B-Chat-Int4 (GPTQ), Phi-2 in fp16, each pinned to a commit SHA in `models.yaml`. The AWQ repos are not gated.
- **Chat formats are written by hand** per family (`llama2`, `mistral`, `chatml`, `phi2`), because the pinned checkpoints do not all ship chat templates. No BOS is written in text (the tokenizers add it).
- **Pointwise score = exact log p(Yes) − log p(No)** from the full next-token distribution, summed over single-token spellings ("Yes"/" Yes"). The vLLM engine scores `prompt + answer token` via prompt logprobs (with prefix caching), avoiding vLLM's top-k logprob cut-off. The HF engine reads the logits directly. Both are verified against a manual log-softmax on a tiny model.
- **Cache:** an on-disk JSONL cache keyed by model, revision, quantization, call, prompt and parameters makes jobs resumable.
- **Aggregation:** Borda count over the 3 shuffles (position p earns 10 − p points). **All ties are broken by a stable hash of (pool, item)**, never by stored order.
- **Empty rankings** (every shuffle failed to parse) score 0 on Recall@1 and NDCG@10; Cold@1 is computed over non-empty rankings, with the empty rate reported beside it.

### Reference heuristics (`coldrank/baselines/heuristics.py`)
- Date heuristics use the **displayed** date, so under date variants they see what the LLM sees.
- **"Cold-filter + content"** = the 10 newest *displayed* candidates ranked by content similarity, then the rest. It uses only prompt-visible information rather than the true cold flag.
- SASRec order places cold items last (it cannot score them).

### Tests
- `code/tests/`: 38 tests covering the split invariants and audit/split agreement, SASRec (NaN-free masking, causality, learns a pattern), pool invariants and exclusions, rendering, parsing, metrics, heuristics, statistics, and an end-to-end HF run with SmolLM2-135M (`-m slow`).

### First results (Amazon VG test, 800 pools; `outputs/amazon_vg/baseline/report_test/summary.md`)
- **Warm ranker sanity:** SASRec validation NDCG@10 = 0.064; on old-target test users, Recall@10 = 0.033 against 0.009 for popularity. Popularity fallback for 0.6% of users.
- **Pools:** all 1,900 pass the invariants; none dropped. The gold item is about 3 years newer than the median warm candidate and is the single newest candidate in 10% of pools (chance among the 10 cold items).
- **Reference heuristics, Recall@1, random / hard negatives:**
  - random: 0.029 / 0.024;
  - random-among-newest: 0.111 / 0.102;
  - newest-first: 0.095 / 0.090;
  - content only: 0.041 / 0.009;
  - **newest-10 filter + content: 0.324 / 0.009**;
  - SASRec and popularity: 0 / 0 (they cannot rank cold items).
- **Hard negatives are adversarial to content matching**, not just harder: they are the cold items most similar to the history (mean similarity 0.73 against 0.54 for the gold), so any similarity-based ranker almost never ranks the gold first. Read hard-negative results as a stress test.
- **Local environment note:** model loading can hang on a stalled Hugging Face CDN connection even when the files are cached; set `HF_HUB_OFFLINE=1` once models are downloaded.

### Adversarial variants (`code/coldrank/adversarial/`, plan_adversarial.md)
- **Separate package** (team request), so the baseline code stays as it was. The shared eligibility logic was extracted from `pools/build.py` into `eligible_cold()`, `cold_catalogue()` and `history_records()`. Regenerating all 1,900 baseline pools in memory gave **byte-identical** output.
- **Order matching:** variant pools keep their reference's candidate order slot by slot and carry `shuffle_key` = the reference pool ID. The renderer seeds the shuffle with it, so every shared candidate appears in the same prompt position; a date variant changes only dates. Baseline pools have no key, so in-flight baseline runs are unaffected.
- **Embeddings:** baseline vectors are reused unchanged; only items missing for the old-target instances are encoded, into `item_embeddings_adv.npz`. The baseline cache is never rewritten, since re-encoding in different batches can shift vectors slightly and change hard-negative order.
- **A2:** partner = a uniformly random warm candidate, seeded by pool, so random and hard pools of the same instance may get different partners.
- **A6:** the baseline's eligibility rules; 49 negatives (random / hard); 1 test instance dropped.
- **A1:**
  - gold = the old target (warm, not in `H_u`), inserted (SASRec ranks it in its top 40 only 4.8% of the time); its true SASRec rank is kept in `warm_rank`;
  - warm slots = SASRec top 39 excluding the gold, `H_u` and the user's window items;
  - traps = 10 random items among the eligible new items at or below the 25th percentile of similarity to the user profile, which must also *display* a date after the cutoff (an item launched later on the cutoff day shows the cutoff date and would not look new);
  - 39 of 1,150 instances dropped.
- **A1-ctrl:** trap slots refilled, in slot order, by SASRec ranks 40–49.
- **A5:** the *n* warm negatives with the lowest similarity are relabeled (nested in *n*); each new date is copied from a random eligible new item that displays a date after the cutoff, seeded per (instance, item).
- **Metrics:** Cold@1 uses `new_dated` (what the model sees) when present, otherwise the true cold flag (baseline). Trap@1 counts `trap`/`relabeled` roles, or `ctrl_fill` on A1-ctrl.
- **Manipulation checks (test):**
  - trap similarity 0.30 vs gold 0.52; traps a median 706 days newer than the gold;
  - A1-ctrl fill similarity 0.535 vs warm 0.544;
  - A5 relabeled similarity 0.32 (n = 1) to 0.42 (n = 10);
  - A2 moves the gold's displayed date back a median of about 1,150 days.
- **Finding:** in A1-ctrl, newest-first reaches Recall@1 = 0.41. Among older items, the one bought next tends to be the most recently launched, so recency is informative even within the warm items. A1-ctrl is therefore not recency-neutral; this must be stated when reporting the trap effect.

### Decomposition and statistics (`code/coldrank/eval/decompose.py`, Thread 3)
- **Input:** per-instance outcomes (Recall@1, Cold@1, Trap@1, top-1 item) for every condition, model and method with a `preds.jsonl`. As in `eval/report.py`, a pool without predictions counts as an empty ranking (Recall@1 = 0). A quantity is skipped when one of its conditions has no predictions.
- **R_random = 1/50**, the analytic chance level for 50 candidates, rather than the empirical random heuristic (which only adds noise).
- **Pairing:** every comparison between conditions built from the same instances is paired on `instance_id`.
  - ∆recency (baseline − A3), swap sensitivity (baseline − A2), shuffle effect (baseline − A4), all-cold effect (baseline − A6, on the 799 shared instances) and trap effect (A1-ctrl − A1).
  - Each gets a paired bootstrap CI (10,000 resamples, seed 0) and an exact McNemar test.
- **∆evidence = R_hide − 1/50**, with the CI of R_hide shifted by 1/50. The code asserts that baseline and A3 cover the same instances, which is the condition for ∆evidence + ∆recency = R_true − 1/50 to hold exactly.
- **Raw over-crediting gap R_baseline − R_A1** uses an **unpaired** bootstrap (different instance sets), for both the random and hard baselines.
- **Flood slope:** OLS slope of the mean Recall@1 over n ∈ {0 (= A1), 1, 2, 5, 10}, on instances present at every level. The CI resamples instances with the same draw across all levels (`stats.bootstrap_slope`).
- **Trap@1 flood slope is descriptive only.** Relabeling turns warm items into "fresh-dated" items, so a ranker that ignores dates gets a positive Trap@1 slope whenever it already liked one of them. Measured on the heuristics: random +0.022 per relabeled item (the mechanical 1/50), SASRec +0.021, popularity +0.046.
- **New primary flood measure, `flood_lift_n`:** P(top-1 is one of the n relabeled items | A5_n) − P(top-1 is one of the *same* items | A1), paired with McNemar. It is exactly 0 for any ranker that ignores dates. Verified: text similarity, popularity and SASRec give 0.000; random gives −0.023 (not significant, because the hash tie-break differs between pool IDs); date-driven heuristics give +0.50 to +0.95.
- **Validated on the heuristic outputs (test):**
  - text similarity (date-blind): ∆recency = 0, swap sensitivity = 0;
  - random: no significant effect;
  - "10 newest + text": ∆recency +0.282, swap sensitivity +0.324, both p < 0.05.
- **Not built yet:** the domain × condition logistic regression (needs EB-NeRD).
- **Outputs:** `outputs/<dataset>/decomposition_<split>/{rates.csv, decomposition.csv, summary.md}`.
- **Tests:** `code/tests/test_decompose.py` covers:
  - a recency-driven model (the terms add up) and a relevance-driven one (zero recency effect, p = 1);
  - trap effect and flood slopes recovered exactly;
  - flood lift = 0 for a date-blind model;
  - skipping missing conditions;
  - the instance-set assertion.

### Running on ADA (4 × RTX 2080 Ti, qos=medium)
- **Our model is Mistral-7B-Instruct** (`mistral_7b`, the default in `scripts/llm_ada.sbatch`); a teammate runs Qwen-7B-Chat with `MODEL=qwen_7b`. Both checkpoints are downloaded to `~/IRE_Project/hf_cache`.
- **ADA facts (checked on the cluster):**
  - account `research`, partition `u22`;
  - qos `medium` allows 4 days, 4 GPUs, 40 CPUs and 125 GB per job;
  - home has 40 GB (11 GB used), `/share1/$USER` has 100 GB;
  - modules include `u22/python/3.12.4` and CUDA 9.2–12.9;
  - the login node `ada-gw1` has internet (PyPI and Hugging Face).
- **Data parallelism, not tensor parallelism:** a 4-bit 7B model fits one 11 GB card, so the job runs 4 independent vLLM workers (one per GPU), each on its own task list. Tasks are assigned longest-first by estimated cost (`scripts/make_tasks.py`), so the GPUs finish at about the same time.
- **One model load per worker:** `coldrank.llm.run --jobs-file` runs many (method, condition, neg, split) tasks in one process. Without it, vLLM startup (~2–3 min) would repeat for each of ~32 tasks.
- **Per-worker cache files:** `--cache-tag gpu<k>` writes to `outputs/cache/<model>__gpu<k>.jsonl`. Four processes appending to one file on a network filesystem can interleave writes and lose entries.
- **Resume:** a task with a `run_summary.json` is skipped, so a job killed by the time limit can be resubmitted. `--limit` runs write to `<split>_limit<N>/` and never count as done.
- **Fast pointwise scoring.** The old vLLM method scored each Yes/No token by computing logprobs over the whole prompt, with no prefix reuse; for ~800k candidate prompts that is impractical.
  - The new `VLLMEngine.yes_no` generates **one token restricted to the Yes/No ids** (`allowed_token_ids`) and reads their logprobs. Restricting the vocabulary renormalizes over those ids, which leaves log p(Yes) − log p(No) unchanged, and prefix caching reuses each pool's shared history.
  - The exact method is kept as `yes_no_exact`. The smoke stage runs `--verify-pointwise 8` and **aborts if the two disagree by more than 0.05**. This depends on vLLM applying `allowed_token_ids` before computing logprobs, which is why vLLM is pinned and the V0 engine forced.
- **vLLM 0.8.5 pinned, with `VLLM_USE_V1=0`:** the V0 engine supports Turing (sm_75) and Qwen v1 (`QWenLMHeadModel`), and applies logits processors before logprobs. It brings torch 2.6 / CUDA 12.4. Jobs run with `HF_HUB_OFFLINE=1`.
- **Where the environment lives (after three failed attempts):**
  1. venv on the login node with the system Python: the login node is **Rocky Linux 9 (Python 3.9)**, the `u22` compute nodes are **Ubuntu 22.04**, and the `u22/python/3.12.4` module exists only on the compute nodes. The venv would point to an interpreter the compute nodes lack.
  2. Setup as a job on a compute node: pip's resolver backtracked for 40 min on vLLM's dependencies, and with `uv` the compute node's slow PyPI link (~2–3 MB/s) hit uv's 30 s timeout mid-download of the 311 MB vLLM wheel.
  3. `/share1` is **not mounted on compute nodes**, so models live in `~/IRE_Project/hf_cache` (home: 40 GB quota).
  - **Final setup:** `scripts/ada_setup_env.sh` runs **on the login node** with **uv's standalone CPython 3.12** (`uv python install 3.12`, stored in `$HOME`). It is self-contained, so the venv runs on both Rocky 9 and Ubuntu 22, and home is shared by all nodes. It uses `UV_HTTP_TIMEOUT=900`.
  - Models are downloaded on the login node (Hugging Face ≈ 17 MB/s there): Mistral-7B-Instruct-v0.1-AWQ 3.9 GB and Qwen-7B-Chat-Int4 5.5 GB.
- **Compute estimate for Qwen** (rough, ±2×, from assumed per-prompt costs on one 2080 Ti: Direct-Rec 2.5 s per prompt; reasoning 10–22 s per prompt, nearly serial because Qwen-7B has no grouped-query attention and the 4k attention cache leaves room for only 1–2 sequences; pointwise 4 s per pool):
  - smoke: minutes;
  - dev-direct: 7.6 GPU-h (~2 h on 4 GPUs);
  - test-direct: 44 GPU-h (~11 h);
  - dev-reasoning: 79 GPU-h (~20 h);
  - test-reasoning: ~446 GPU-h (~112 h), **not feasible as specified**. It needs a reduction (e.g. one shuffle, random negatives only, or a test subsample); this is a team decision, pending.
- **Smoke run first:** its `run_summary.json` seconds replace the assumed per-prompt costs in `make_tasks.py`.
- **Fixes needed to run on ADA (2026-10-14):**
  - `transformers` is **pinned to 4.51.3**. Unpinned, uv installed 5.19, which vLLM 0.8.5 cannot import (`ProcessorMixin`).
  - `llm_ada.sbatch` requests **36 CPUs / 100 GB**: sbatch rejects 40 CPUs for one task, even though the nodes have 40 CPUs and 128 GB.
  - **Mistral: `disable_sliding_window: true`** (`models.yaml`). Mistral v0.1's 4,096-token sliding window cannot be combined with prefix caching in vLLM. With `max_model_len` = 4096 = the window, no token ever leaves the window, so attention is identical.
  - **Pointwise reference changed:** vLLM 0.8.5 V0 crashes on prompt logprobs when prefix caching is on (`assert len(next_token_ids) == len(query_indices)`). `yes_no_exact` now uses one unrestricted greedy step with the **top-20 logprobs over the full vocabulary** (the exact model probabilities) and skips prompts where Yes/No are not in the top 20.
- **Smoke run, Mistral-7B-Instruct (job 20870):**
  - fast vs exact pointwise: 8/8 comparable, max |diff| = 0.016 (fp16 noise, tolerance 0.05);
  - Direct-Rec: 100% parsed, ~2,600 prompt tokens, no overflow;
  - speed: ~1.6 s per Direct-Rec prompt and ~1.2 s per pointwise pool on one 2080 Ti, from a tiny batch (larger batches are faster);
  - vLLM reports up to 10 concurrent 4,096-token sequences (grouped-query attention).
  - dev-direct (job 20873) is therefore expected to take ≈ 1 h on 4 GPUs.
- **Keeping results** (team request: everything must be available for later analysis):
  - **Per task** (`outputs/<dataset>/<condition>/<neg>/<model>/<method>/<split>/`):
    - `raw.jsonl`: every raw generation per pool and shuffle, with parse mode, prompt length, history used and the prompt-ID → item map;
    - `preds.jsonl`: final rankings;
    - `run_summary.json`.
    - Prompts are not stored; they are regenerated exactly from the pool, method and shuffle seed.
  - **Per job:** worker logs; task lists (`outputs/tasks/`); the per-GPU LLM caches (`outputs/cache/<model>__gpu<k>.jsonl`, every generation keyed by prompt hash); and a **run manifest** (`outputs/logs/<model>_<stage>_<job>/manifest.json`) written by `scripts/write_manifest.py`. The manifest holds job ID, node, GPUs and driver, stage, model and pinned revision, both configs, package versions, a SHA-256 of all code, configs and scripts, the task lists, start/end times and exit status.
  - **`ada_sync.sh pull` copies everything, caches included,** and never deletes local files. ADA keeps its own copy.
  - **`scripts/collect_results.sh <split>`** (run after each stage) pulls, reruns heuristics, per-condition reports and the decomposition, and copies the small files into the **git-tracked `results/<dataset>/<split>/`**: per-condition reports, the decomposition, rates, a table of all LLM run summaries, and the manifests. `outputs/` stays git-ignored because of size.
  - Job 20873 (dev-direct) was submitted before the manifest step existed; its manifest is written after the fact.

### First LLM results (Mistral-7B, dev-direct, job 20873) and the Direct-Rec fix
- **Direct-Rec degenerated:** 446/450 dev baseline outputs were `FINAL: [1] > [2] > ... > [10]`, simply the first ten candidates in prompt order. Recall@1 = 0.007 and Cold@1 = 0.19 (chance 0.20). The parse rate was 100%, so **"parsed" does not mean "ranked"**.
  - Every run summary now reports `identity_rate` (share of outputs starting [1], [2], [3]) and `top1_mode_share` (share of the most common first ID).
  - Likely causes: the format template `[id] > [id] > … > [id]`, the `FINAL: [` prefill, and 50 candidates being a lot for a 7B model.
- **Pointwise was also at chance** (Recall@1 0.007–0.013), with Cold@1 0.03–0.10 (below chance: it prefers the older candidates). Not interpretable until the model ranks the gold above chance.
- **No recency conclusions are drawn from job 20873.** Its dev outputs are kept for the record.
- **Diagnostic stage `promptcheck`** (job 21054; 50 dev baseline pools, random negatives):
  1. `direct_v2` at K = 50: no prefill; a scrambled example (`FINAL: [12] > [3] > [17] > …`); "the candidates are listed in random order: do not simply copy that order".
  2. `direct_sw` at K = 50: a RankGPT-style **sliding window** (window 20, step 10, bottom to top; 4 calls per shuffle) with the v2 prompt. This keeps Li et al.'s 50-candidate protocol.
  3. `direct` and `direct_v2` at **K = 20** (`baseline_k20`: 16 best-SASRec warm items + gold + 3 of the 9 cold negatives, seeded; `scripts/make_k20_pools.py`). This separates the prompt effect from the pool-size effect. At K = 20, random = 0.05 and random-among-newest = 0.25.
  - **Decision rule:** prefer a fix that keeps K = 50 (v2, then sliding window). Use K = 20 only if nothing at 50 works, and report it as a deviation.
- **Shared task queue** (from job 21054 on): all workers read one `queue.txt` (longest task first) and claim tasks with an atomic `mkdir` (`llm.run --claim-dir`). The static shards of job 20873 left 3 GPUs idle for ~15 min because one GPU drew the four slowest tasks. Unit-tested: each task runs exactly once under concurrent workers.

### Repository hygiene (2026-10-16)
- **`.gitignore` bug:** the unanchored `data/` rule also matched `code/coldrank/data/`, so the data-loading package (`amazon_vg.py`, `audit.py`, `split.py`) was **never committed**. A teammate had to rewrite it from the tests (`analysis_2026-10-16.md` §2). Rules are now anchored (`/data/`, `/outputs/`); `code/coldrank/data/` must be committed.
- **ADA scripts are git-ignored** (team decision): `code/scripts/*.sbatch`, `ada_*.sh`, `setup_ada.sh`, `submit.sh`, `progress.sh`, `collect_results.sh`, `make_tasks.py`, `write_manifest.py`. The three that were already committed were untracked with `git rm --cached`; local copies are kept.

### Why the LLM rerankers fail here, and what the literature actually supports (2026-10-16)
Evidence: our Mistral runs (dev-direct job 20873, promptcheck job 21054) and the teammates' 88 runs (`analysis_2026-10-16.md`).
- **Only one LLM run beats chance:** Llama-2-7B pointwise, test Recall@1 0.072. **All of its gain is Δrecency** (+0.050*, Δevidence ≈ 0; swap sensitivity +0.044*). Every listwise method, and every other model, is at chance at K = 50.
- **The sign of the date effect is model-dependent:** Mistral, Qwen and Phi-2 *penalise* post-cutoff dates (Cold@1 below chance; −0.06 to −0.12 log-odds when an item gains a new date).
- **This is consistent with the published evidence, not a bug in our pipeline:**
  - Dong et al. (2026), with our exact domain (Amazon Video Games) and stronger 2025 models (Qwen3-8B/32B, Llama-3.3-70B): "Qwen3-8B does not win any domain … a cross-encoder wins Video Games". Scaling "does not rescue most domains"; the exception is MIND news (weak interaction signal, rich text). Pointwise yes/no beats listwise (VideoG Recall@10 .040 vs .020/.020). Their Video Games evidence effect is −0.017.
  - Li et al.'s Netflix gains come from **Qwen-2.5-32B with SFT/GRPO**, on a catalogue (movies/shows) with rich descriptive metadata, and are **reported only as relative improvements** over their own Fast-Reason baseline. They never show absolute accuracy against a date-only ranker.
  - Our domain carries little content signal: text similarity alone scores Recall@1 0.041 (random 0.029); histories are short (median 8.5) and stale (median 281 days between the last history item and the target); many targets are hardware or accessories.
- **Implication:** on Amazon Video Games, zero-shot 2023 7B/13B models do not show the cold-start competence that the recency question presupposes. The one model that "succeeds" does so through the date shortcut. That supports the project's thesis, but it is a narrow base.
