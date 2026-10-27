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

### K = 15 pools and letter codes; Llama-2-7B on the 15-candidate baseline (2026-10-16, team decision)
- **Model focus:** Llama-2-7B-Chat (`llama2_7b`, TheBloke AWQ, pinned revision), the only model with above-chance pointwise accuracy in the teammates' runs (its gain was entirely Δrecency).
- **Pool size 15**, keeping Li et al.'s 4:1 warm:cold ratio (`coldrank/adversarial/resize.py`, `python -m coldrank.adversarial.resize`). Every K = 15 pool is derived **from the corresponding 50-pool**, so the gold, history and SASRec order are unchanged and instances stay paired across conditions:
  - `k15_baseline`: the 12 best-SASRec warm items + gold + 2 cold negatives (random: 2 of the 9 sampled; hard: the 2 most similar);
  - `k15_a2_swap/a3_hide/a4_shuffle`: the same date-variant functions applied to `k15_baseline`;
  - `k15_a6_allcold`: gold + 14 of the 49 cold negatives;
  - `k15_a1_oldgold`: gold + 11 best warm + 3 of the 10 traps;
  - `k15_a1ctrl`: trap slots refilled by the next warm ranks (12–14);
  - `k15_a5_n{1,2,5,10}`: relabeled dates copied from the 50-pool's traps.
  - Pools carry `pool_size: 15`; all `adversarial.checks` invariants pass (generalised from 50 to any size; the A5 check counts the pool's traps instead of assuming 10).
  - Counts: 150 / 800 (A6 799) and 144 / 967 for the A1 family.
- **Deviation from Li et al. (40 + 10):** it makes listwise prompts feasible for 7B models at a 4k context. Measured with the Llama-2 tokenizer: 1.2–2.4k prompt tokens, 0 overflow, ~10 history items kept (5 at K = 50). Consequences:
  - chance is 1/15;
  - the date shortcut is much stronger: random-among-newest-3 ≈ 0.36, newest-3 + text ≈ 0.59 on test (random negatives).
- **Heuristic fix:** `newest_set` used a fixed 10; it now uses 1/5 of the pool (10 of 50, 3 of 15). 50-pool numbers are unchanged (0.111 / 0.324).
- **Decomposition:** `--prefix k15_` selects the family; ∆evidence uses chance = 1/K, computed from the pool files.
- **K = 15 reference numbers (test, Recall@1, random negatives):**
  - random 0.055;
  - random-among-newest 0.360;
  - newest-first 0.346;
  - text 0.098;
  - newest-3 + text 0.590.
  - Under A2, A3 and A4 the date heuristics collapse (0.000 / 0.06 / 0.06–0.09). A1: newest-first Trap@1 = 1. A1-ctrl newest-first 0.523.
  - Decomposition, "newest + text": ∆recency +0.492*, swap sensitivity +0.590*, trap effect +0.178*, flood lift (n = 10) +0.996*. For text similarity all are 0.
- **Letter codes for all listwise methods** (`prompt.id_style: codes`, the teammates' frozen `direct_codes_free`):
  - candidates get seeded two-consonant codes; worked examples get vowel codes (AE, IO, …) that never label a real candidate, so copying the example's answer cannot parse;
  - Direct-Rec has no prefill; the parser maps codes to positions.
  - Earlier runs used numbered IDs (still available with `id_style: numbers`).
- **Run (stage `k15-baseline`, `llm_ada.sbatch`):** Llama-2-7B × {direct, pointwise, base, fast, structural, structural_norecency, ssc} × {random, hard} × {dev, then test} on `k15_baseline`, 28 tasks, ~26 GPU-h estimated (≈ 6–7 h on 4 GPUs). The fast-vs-exact pointwise check now runs at the start of every job.
- **Result of job 23791 (dev, 150 pools; full write-up: `analysis_2026-10-16_k15.md`):**
  - **Pointwise is the only method above chance:** Recall@1 0.167 [0.107, 0.227] random negatives, 0.113 hard (chance 0.067). It ranks the gold first among the 3 new items 49% of the time (chance 33%) and prefers new items (Cold@1 0.33 / 0.47 vs 0.20). It stays far below newest-first (0.347) and newest + text (0.580).
  - **Direct-Rec and the 5 reasoning methods are at chance** (0.013–0.067). Reasoning outputs are usable only 25–48% of the time (placeholder/example copying, no final line). Context no longer overflows at K = 15.
  - **Decision pending:** run pointwise on the K = 15 variants (dev) to decompose; drop listwise/reasoning for this model.
- **Variants, job 23838** (Llama-2-7B pointwise + Direct-Rec on all K = 15 variants, dev; write-up: `analysis_2026-10-16_k15.md` §10–§11):
  - **Pointwise:** R_true 0.167 → R_hide 0.060, i.e. **Δrecency +0.107\***, Δevidence ≈ 0 (hard negatives: +0.073\*). Swap sensitivity +0.067\*. Direct date effect ±0.07 log-odds (same as at K = 50).
  - **Not fooled by low-similarity traps:** trap effect n.s., Trap@1 0.167 < 0.20. Flood lift +0.125\* at n = 10, but Recall@1 drops only 0.146 → 0.118.
  - **Direct-Rec:** at chance everywhere, no date effect.
  - **Stage `k15-variants`** in `make_tasks.py`. Reasoning was not run on variants (at chance on the baseline).
- **Test run, job 23920** (team decision, 2026-10-16: "run all the things we did in dev in test"; stage `k15-test` in `make_tasks.py`):
  - Exactly the dev tasks of jobs 23791 + 23838, on test: 7 methods × {random, hard} on `k15_baseline` (14 tasks), plus pointwise + Direct-Rec on all 10 K = 15 variants (28 tasks); 42 tasks.
  - Pools: 800 cold test instances (baseline, A2, A3, A4, A6); 1000 old-target test instances (A1, A1-ctrl, A5).
  - The `k15-baseline` stage also listed the test tasks, but job 23791 ran only its dev half (test skipped by request), so test needed its own stage.
  - Estimated ~8.5 h on 4 GPUs from the dev timings (`make_tasks.py` prints 11.5 h). Most of that is the 5 reasoning methods, which were at chance on dev; they are run anyway so that test mirrors dev.
  - **Result (finished 2026-10-16 23:17, 7 h 16 min, exit 0; write-up `analysis_2026-10-16_k15.md` §12):** test confirms every dev conclusion. Pointwise Recall@1 0.149 / 0.105 (random / hard); dates hidden 0.075 / 0.068 = chance; **Δrecency +0.074\*** / +0.037\*, **Δevidence +0.008** / +0.001 (n.s.); swap +0.062\* / +0.054\*; direct date effect −0.075 / +0.069 log-odds; trap effect −0.009 (n.s.); flood lift +0.121\*. Direct-Rec and the 5 reasoning methods at or below chance. **The paper uses these test numbers.**

### EB-NeRD port (2026-10-17, team decision; plan_ebnerd.md)
- **Scope:** everything run on Amazon, on EB-NeRD: split, SASRec, K = 50 pools, variants A1–A6, K = 15 resize, heuristics, Llama-2-7B (K = 15 baseline × 7 methods; pointwise + Direct-Rec on every K = 15 variant), dev then test, decomposition.
- **Data:** EB-NeRD **small** (84 MB; large is 3.2 GB and not needed). Train + validation histories and behaviors (test labels are not public): 3.05M clicks, 18,827 users, 12,042 clicked articles, 27 Apr – 1 Jun 2023, continuous across the cutoff.
- **Cutoff and window:** **T = 2023-05-25 00:00, w = 1 day** (Li et al.'s single test day). Audit: 8,676 cold-target and 5,107 old-target instances, 109 distinct golds, 123 articles published in the window.
- **Positive:** every click (rating = 1.0). **Launch time** = min(published_time, first click), the Amazon rule (525 clicks predate the listed publication time).
- **Displayed date:** **date and time**, `YYYY-MM-DD HH:MM` (`date_format` in the config; `common.ts_to_date` / `date_to_ts`). The median click is 1.4 h after publication, so with w = 1 day only the time separates the cold articles. All times are floored to the minute, so displayed = stored. Amazon keeps date only.
- **Translation:** Danish → English with **Helsinki-NLP/opus-mt-da-en**, revision `0960cff5…`, **greedy** decoding (beam search, the model default of 4, took ~3 s per subtitle on CPU, about 30× slower), every distinct string translated once and cached (`data/processed/ebnerd/translations.jsonl`). Title, subtitle and topics are translated. The 26 section names (`category_str`) use a fixed English table, because several are site-specific ("auto" = automated local-business news, "play" = the video section, "side9" = page-nine glamour). Danish originals are kept as `*_da`.
- **Split sizes as for Amazon:** cold 150 dev / 800 test (`n_test_cold`), old 150 / 1,000.
- **Feasibility filters** (EB-NeRD config only): cold instances need ≥ 9 cold articles in (T, target time] outside the user's window (`require_cold_negs`), old instances need ≥ 10 new in-view-not-clicked articles (`require_traps`). These are the pool builders' own preconditions, applied before subsampling so dev/test reach their sizes; early-morning targets are the ones removed.
- **A1 traps (news):** `adversarial.trap_source: inview`: new articles (published in (T, target time]) **shown to the user in an impression in [T, target time] and not clicked** in the window, not in the history (`split.attach_inview_not_clicked`). No similarity filter: being skipped is the irrelevance evidence. `trap_sim_threshold` is null and its check is skipped.
- **Embeddings** (hard negatives, profiles, near-duplicate filter, content heuristic): all-MiniLM-L6-v2 on the English "title. subtitle | section".
- **Prompts:** `prompt.domain: news` selects a `Domain` in `prompts/render.py`: "a Danish news website (articles translated into English)", "reading history", "read next", fields **title | published | section | summary** (subtitle cut at 100 characters; titles cut at 100), Structural Reasoning's extra attribute = **topics** (first 3; news has no brand). Two new hand-written news worked examples (`examples.py`): invented articles dated 2019, correct article 2nd / 4th newest, no dates in the reasoning. **Amazon prompts verified byte-identical** to the committed renderer (1,435 prompts: all methods, both ID styles, 3 shuffles, pointwise; baseline, A3, A1, K = 50), so Amazon caches stay valid.
- **ADA:** `CONFIG=ebnerd` in `llm_ada.sbatch` (passed to `make_tasks.py` and `coldrank.llm.run`); task/log directories get an `ebnerd_` prefix. `ada_sync.sh push` syncs the pool directories of every dataset.
- **A1 trap count (team decision, 2026-10-17):** an old-target user needs **≥ 3** new in-view-not-clicked articles (`require_traps: 3`), and the K = 50 A1 pool is **gold + 46 warm + 3 traps** (`a1_n_warm: 46`, `a1_n_traps: 3`, `ctrl_fill: 3`). Requiring Amazon's 10 left only 865 old instances (150 dev / 715 test), all heavy readers; with 3 there are 2,618. The K = 15 pools that are run are gold + 11 warm + 3 traps for both datasets.
- **Split result:** 8,676 cold / 5,107 old instances; 829 cold dropped (too few cold negatives before the target), 2,489 old dropped (fewer than 3 traps); **cold 150 dev / 800 test, old 150 / 1,000**. 1,756 cold and 11,106 warm articles.
- **A6 at K = 15 directly (forced by the data, 2026-10-17):** A6 negatives must be new articles published before the target click (no future items). With a one-day window only 68 / 265 dev / test targets have 49 of them, so EB-NeRD's A6 is built with 14 negatives (`a6_n_negs: 14`; the pool carries `pool_size: 15`; random and hard selection unchanged) and resized 1:1. A6 covers 131 / 150 dev and 650 / 800 test instances (Amazon 799); the all-cold effect is computed on these paired instances.
- **Local pipeline result:** SASRec on 2.77M pre-T clicks (5-core: 18,588 users, 4,757 articles), val NDCG@10 0.259, old-target Recall@10 0.108 (popularity 0). K = 15 heuristics on test: newest-first 0.560 / 0.515 (random / hard), against 0.346 / 0.323 on Amazon: the date shortcut is stronger in news, as expected. Content similarity alone is at chance (0.060 / 0.037).
- **ADA run:** job **25738** (`CONFIG=ebnerd`, stage `k15-all`, 84 tasks, gnode056, 11 h 26 min, exit 0). Job 25735 failed in 18 s on gnode069 (`Failed to get device handle for GPU 1`), so that node was excluded (`sbatch --exclude=gnode069`).
- **Result (test; write-up `analysis_2026-10-18_ebnerd.md`):** Llama-2-7B pointwise is **at chance on news** (Recall@1 0.066 / 0.066; dev had suggested 0.107). Δrecency +0.009 (n.s.), Δevidence −0.009 (n.s.); the Amazon − news difference in Δrecency is +0.065 (z = 4.3, p = 2×10⁻⁵). The date preference is the **same size** as on Amazon (A2: −0.070 / +0.059 log-odds; swap +0.020\*, shuffle +0.018\*, flood lift +0.076\*) although the gold–warm date gap is ~18 h instead of ~1,000 days. It does not turn into accuracy because the gold trails its best competitor by more on news (median 0.43 vs 0.24 log-odds). All-cold effect −0.031\* (better when all candidates are new). Other methods at or below chance.
- **Manifest fix (2026-10-23):** `scripts/write_manifest.py` always recorded `configs/amazon_vg.yaml` as `data_config`; it now takes `--config` (passed by `llm_ada.sbatch`). Job 25738's manifest was rebuilt with the EB-NeRD config (every other field unchanged; see its `note`). The results were never affected: `llm.run` received `--config ebnerd`.
- **Results package (2026-10-23):** `results_veda.zip` (98 MB) + `results_veda.md` (handoff: scope, status, every run, headline results, zip layout, how to continue).

### Stronger-model pilot: Qwen2.5-32B-Instruct (team decision, 2026-10-23)
- **Why:** Llama-2-7B is above chance only for Amazon pointwise; listwise and reasoning methods fail on format and context, so the question "how much of the LLM's cold-start success is recency" lacks a model with success. Qwen-2.5-32B-Instruct is **Li et al.'s own base model** for their reasoning methods (§3.2, "Reasoning without Fine-Tuning").
- **Checkpoint:** `Qwen/Qwen2.5-32B-Instruct-AWQ` @ `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` (4-bit, 19.3 GB), chatml format, **tensor parallel over the 4 GPUs** (`vllm.tensor_parallel_size: 4`; `TP=4` in `llm_ada.sbatch` runs one worker on GPUs 0–3), `max_model_len` 8192 (reasoning prompts fit without trimming history).
- **Relaxes the 2023-only rule** (Qwen2.5: Sep 2024, after both datasets). Argument (report §3.3): exposure can only shrink Δrecency (a model that knows an item's real date relies less on the displayed one), so recency findings stay conservative; it can inflate Δevidence, so the exposure check becomes required for this model.
- **Pilot scope (dev, both datasets; stage `pilot-dev`):** K = 15 baseline × 7 methods × {random, hard}, plus pointwise + Direct-Rec on A3 (dates hidden, random + hard), A1 (traps) and A1-ctrl (needed for the trap effect): 22 tasks per dataset. Preceded by `smoke-k15` (every method on 16 dev pools) to measure speed. Go / no-go for test and the remaining variants after the pilot.
- **ADA disk:** home quota 50 GB; the pip download cache (5.3 GB) was cleared to fit the model.
- **Pointwise verification tolerance (2026-10-23):** the first Qwen smoke (job 36382) stopped at the start-of-job check: fast vs exact scores differed by 0.0625 log-odds (e.g. 15.656 vs 15.594), above the fixed 0.05. The differences are multiples of 1/32, i.e. fp16 rounding of logits: Qwen's Yes − No log-odds are +13 to +16 (Llama-2: ~+5), and the rounding step grows with magnitude. The tolerance is now **0.05 + 0.5% × |score|** (`run.verify_pointwise`, `rel_tol`); Qwen passes with margin, Llama-2's earlier diffs (≤ 0.02) pass unchanged, and an actual mismatch (tenths of a log-odds or more) still aborts. Note: rounded scores can tie within a pool; ties are broken by the existing seeded hash (`eval.metrics.tie_key`).
- **Qwen smoke tests (2026-10-23):** jobs 36386 (Amazon, 9 min 22 s) and 36387 (EB-NeRD, 13 min 56 s), all 7 methods on 16 dev pools, TP = 4 on gnode055; weights load in ~6 s per GPU, KV cache allows ~6.8 concurrent 8k-token sequences. Usable rankings (Amazon / news): Base 100% / 100%, Fast 100% / 98%, SSC 94% / 79%, Structural 75% / 33%, Structural without recency 60% / 42%, Direct-Rec 83% / 75%.
- **Two parsing/output fixes found by the smoke:**
  - **Bare codes:** Qwen often writes `FINAL: RB > DG > …` without brackets. `parse_ranking` now also accepts bare codes after `FINAL:` when they label a candidate of that prompt (mode `final_bare`). Re-parsing Llama-2's saved K = 15 outputs with the same rule would rescue 1,055 / 76,992 Amazon and 142 / 76,686 news outputs (mostly Direct-Rec); those results have not been re-scored yet (pending team decision).
  - **Output cap:** Structural / SSC reasoning ran out at 1,024 tokens. Per-model override `gen_cap` in `models.yaml` (Qwen: 2,048 for structural, structural_norecency, ssc); Llama-2 keeps the 4k-context caps.
- **Pilot jobs:** 36400 (Amazon) and 36401 (EB-NeRD), stage `pilot-dev`, 22 tasks each. Estimate from the smoke timings: ≈ 3.5 h + ≈ 5.5 h, run one after the other (the model needs all 4 GPUs).
- **Pilot cancelled (2026-10-23, user request):** 36400 cancelled after 1 min 41 s (model loading, no task finished); 36401 cancelled before it started. Nothing to clean up; the smoke outputs (`dev_limit16`) and the generation cache are kept.
- **Pilot scope extended (team decision, 2026-10-23):** SSC and Structural also run on A3 (random, hard), A1 and A1-ctrl, so the reasoning methods can be decomposed too (Δrecency, trap effect). `pilot-dev` = 30 tasks per dataset (14 baseline + 4 × 4 variant tasks). Structural without recency stays baseline-only (our ablation, not one of Li et al.'s methods). Estimate ≈ 6.5 h (Amazon) and ≈ 10 h (EB-NeRD) at ADA speed.
- **Split across machines (team decision):** Amazon on ADA (4 × 2080 Ti, TP = 4); EB-NeRD on a teammate's machine (one free RTX 5000 Ada, 32 GB; TP = 1). Each dataset runs entirely on one machine, so every paired comparison stays on identical hardware; scores may differ across GPU types in the last fp16 digit, which never enters a comparison.
- **Single-machine tooling:** `run.py --tensor-parallel N` overrides the model's split (same checkpoint; outputs and cache keys unchanged); `scripts/run_local.sh` (non-SLURM counterpart of `llm_ada.sbatch`: pins `CUDA_VISIBLE_DEVICES`, refuses a GPU that already has memory in use, stable per-machine cache file, manifest); `scripts/setup_local_env.sh` (Python 3.12 venv with the ADA versions: vllm 0.8.5, torch 2.6.0, transformers 4.51.3). Teammate instructions: `scripts/README_friend_ebnerd.md`; package `ebnerd_qwen_bundle.zip` (26.6 MB: code incl. the git-ignored scripts + the 7 EB-NeRD dev pool files the pilot needs).
- **Pilot results (dev; tables in `outputs/{amazon_vg,ebnerd}/k15_{baseline,a3_hide,a1_oldgold,a1ctrl}/report_dev/summary.md`):**
  - **Runs:** Amazon job 36428 (gnode061, TP = 4, 5 h 22 min, exit 0); EB-NeRD on the teammate's RTX 5000 Ada (TP = 1, `local-20261024-134452`, ~10.5 h, status 0). Pointwise fast-vs-exact check passed on both (max excess −0.007 / −0.022).
  - **Qwen2.5-32B is at chance on both datasets and not better than Llama-2-7B.** Baseline Recall@1, random / hard negatives: Amazon pointwise 0.080 / 0.073 (Llama-2 0.167 / 0.113; paired McNemar on random p = 0.02 in Llama-2's favour), EB-NeRD pointwise 0.060 / 0.027 (Llama-2 0.107 / 0.080). Best Qwen cell: EB-NeRD SSC 0.093 vs random 0.073 (p = 0.69). Newest-first: 0.347 / 0.307 (Amazon), 0.567 / 0.540 (news).
  - **No recency preference:** Cold@1 at or below chance; hiding dates changes nothing (Amazon pointwise 0.080 → 0.053, news 0.060 → 0.087); Trap@1 0.007–0.18, below chance (0.20–0.23); Recall@1 with traps ≥ without (Amazon pointwise 0.132 vs 0.118; news 0.120 vs 0.107).
  - **Reasoning outputs:** Amazon SSC / Structural unparsable in 15–30% of outputs (3–18% of pools empty), EB-NeRD 9–18%. Not the output cap (≤ 4 per task reach 2,048 tokens): the failed outputs are short, end mid-sentence and use letter codes that label no candidate. Suspected fp16 + plain AWQ kernels (both machines use `quantization=awq`, fp16); not confirmed. Base / Fast / Direct-Rec / pointwise unaffected.
  - **Decision:** Qwen2.5-32B is not scaled to test.

### Fine-tuning experiment: does Li et al.'s fine-tuning learn the date shortcut? (team decision, 2026-10-26)
- **Question.** Zero-shot models are at chance (Qwen2.5-32B) or succeed only through dates (Llama-2-7B pointwise on Amazon: Δrecency +0.074\*, Δevidence +0.008). Li et al.'s largest gains come from fine-tuning on their own protocol, where the target is always one of the newest candidates, so "pick the newest" is the cheapest rule to learn. *(Correction, 2026-10-27: that holds for Li et al.'s **test** set only. Their SFT set is "16,218 interactions after removing cold-start items" (§3.1.2), and §3.3 explains the weak fine-tuned cold-start results by "the training data excludes cold-start items", so their training targets are not new items. Our pool-based training data is therefore not a replication of their SFT data; see "Production-style training data (impression negatives)" below.)* We fine-tune on training pools built exactly like our test pools, **dates shown** (Li et al.'s and production's setting), then run the existing diagnostics on the fine-tuned model: baseline, A2 swap, A3 hide, A1 traps / A1-ctrl. Recency learning predicts a higher baseline Recall@1 that disappears under A3/A2 (Δrecency up, Δevidence ≈ 0) and a higher Trap@1.
- **No dates-hidden training arm** (team decision: replicate the production setting; the diagnosis comes from the test-time variants, as for the zero-shot models). It stays an optional control (would rule out "the model only fails on A3 because it never saw a prompt without dates").
- **Model: Llama-2-7B-Chat** — the one model with a measured recency shortcut (so we test whether fine-tuning amplifies it), released 2023 (no exception to the 2023 rule), fp16-safe on Turing. Full-precision base `meta-llama/Llama-2-7b-chat-hf` @ `f5db02db724555f92da89c216ac04704f23d4590` (gated: needs a Hugging Face account with Meta's licence accepted); the zero-shot runs used TheBloke's AWQ copy of the same weights.
- **One model per dataset**, as Li et al. train on their own platform's data: Amazon model tested on Amazon, EB-NeRD model on EB-NeRD. Amazon first.
- **What replicates Li et al.'s SFT (§3.2.2):** QLoRA with a 4-bit base, **1 epoch**, labels from the protocol itself. **Deviation: pointwise Yes/No training instead of SFT on reasoning paths + GRPO.** Reasons: pointwise is the only method above chance for Llama-2 (its reasoning methods are at chance with 25–48% usable outputs, so there are few successful paths to imitate); GRPO needs many long generations per step, not feasible on 11 GB Turing GPUs; pointwise is also the method our decomposition is measured on.
- **Training instances: earlier cutoffs T′ < T**, built with the test-split code (`data.split.build_instances`: same positives, launch times, min_history 5, first cold positive in [T′, T′ + w)):
  - every window ends before T and windows do not overlap (no training target falls in or after the test window);
  - **all dev/test users excluded** (cold and old; 1,977 Amazon / 1,996 EB-NeRD users), so no test user's later behaviour is trained on; test golds are launched after T and cannot occur in training;
  - **Amazon:** w = 180 days, T′ ∈ {2019-09-01, 2020-03-01, 2020-09-01, 2021-03-01, 2021-09-01, 2022-03-01}: 980 / 488 / 969 / 580 / 711 / 458 = **4,186 instances** (before the pool step), 244–421 distinct golds per cutoff; all used.
  - **EB-NeRD:** w = 1 day, `require_cold_negs` 9 as for test; 6,000–8,700 feasible instances per day between 4 and 24 May. T′ ∈ {May 6, 9, 12, 15, 18, 21, 24}, **750 instances per cutoff** (seeded subsample): 5,250 instances, ~100 distinct golds per day. Seven cutoffs, not 21, because each needs its own SASRec (~40 min).
- **Pools: same procedure as test.** Per cutoff: SASRec retrained on positives before T′ with the dataset's hyperparameters (the ranker never sees the training target's future), top-100 warm items, the K = 50 pool builder (`pools.build.build_pools`, same exclusions, cold negatives launched in (T′, target], near-duplicate filter), then `adversarial.resize.shrink_baseline` to K = 15 (12 warm + gold + 2 cold). Each instance keeps one negative setting, random or hard (seeded 50/50), so no gold is trained on twice.
- **SFT examples:** per pool the gold, both cold negatives and 3 of the 12 warm items (seeded): 6 examples, 1 positive : 5 negatives (all 15 would give 1 : 14 and 2.5× the tokens). The prompt is `render_pointwise` + `format_chat` exactly as at inference (history 20, oldest items dropped until the prompt fits). Target: the single answer token "Yes" / "No"; the loss is on that token only. 5% of training users are held out for a validation loss (no early stopping: 1 epoch, as Li et al.).
- **Hyperparameters** (`configs/finetune.yaml`; standard QLoRA values, Li et al. give none beyond 4-bit and 1 epoch): NF4 with double quantization, fp16 compute (Turing has no bf16); LoRA r = 16, α = 32, dropout 0.05 on all attention and MLP projections; AdamW, lr 2e-4, cosine schedule, 3% warmup; effective batch 32 (4 GPUs × 4 × 2 accumulation); gradient checkpointing; seed 42; DDP over the 4 GPUs (one model copy per GPU, ~8 GB each).
- **Evaluation:** the adapter is merged into the fp16 base. A fair before/after needs the same weights, so zero-shot Llama-2 is re-run on the **fp16 base** (`llama2_7b_fp16`) next to the fine-tuned model (`llama2_7b_ft_<dataset>`). fp16 7B (13.5 GB) does not fit one 11 GB GPU: vLLM with TP = 2, two workers on 4 GPUs. Pointwise on k15 baseline, A2, A3, A1, A1-ctrl (dev, then test); decomposition as before.
- **Environment:** `peft`, `bitsandbytes`, `accelerate`, `sentence-transformers` added to the ADA venv with vllm 0.8.5 / torch 2.6.0 / transformers 4.51.3 pinned, so the inference stack is unchanged.
- **Implementation (2026-10-26):**
  - `coldrank/finetune/trainpools.py` (per-cutoff instances → SASRec → K = 50 pools → K = 15; `--collect` merges and re-checks: no dev/test user, every target before T, unique pool ids), `sft_data.py` (examples + user holdout), `train.py` (QLoRA with HF `Trainer`), `merge.py` (adapter → fp16 base); `tests/test_finetune.py` (9 tests; suite 90 passed).
  - **Loss at the last prompt position:** the model gets the inference prompt (BOS included, as vLLM tokenizes it), left-padded with explicit position ids, and only the last position's logits are computed (`logits_to_keep=1`); cross-entropy against the answer token. That is the distribution the pointwise scorer reads. The answer token is checked at start-up to be the token the model emits after `[/INST]` and one of the scorer's Yes/No ids.
  - **Embeddings for items new to a cutoff** are encoded with all-MiniLM-L6-v2 @ `1110a243…` and appended in a per-cutoff cache; the split's `item_embeddings.npz` is reused unchanged (`adversarial.common.extended_embeddings`).
  - **Evaluation plumbing:** a models.yaml entry with `base` + `adapter` is merged by `llm.run` on first use into node-local `/ssd_scratch/$USER/coldrank_merged/<base>__<adapter hash>` (13.5 GB per model; ADA home has ~5 GB free; nodes have 491 GB SSD and 125 GB RAM), once per node under a file lock. The generation cache identifies it by base + adapter hash (`ident_hf_id`), so cache keys do not depend on the node; keys of every existing model are unchanged.
  - **Stages** `ft-dev` / `ft-test` in `make_tasks.py`: pointwise on the K = 15 baseline and all ten K = 15 variants (as for the zero-shot Llama-2 runs). Run with `llm_ada.sbatch`, `TP=2`, for `llama2_7b_fp16` and `llama2_7b_ft_<dataset>`.
  - **ADA scripts** (git-ignored): `ft_pools.sbatch` (one cutoff per GPU, 4 at a time, then `--collect`), `ft_train.sbatch` (SFT examples if missing; `torchrun` DDP on 4 GPUs; `PILOT=1` = 60 steps into `<run>_pilot/` to measure memory and speed; GPU memory logged every 60 s). `download_models.py` honours a model's `allow_patterns` (the fp16 Llama repo also ships .bin copies).
  - **ADA environment:** `peft 0.15.2`, `bitsandbytes 0.45.5`, `accelerate 1.6.0`, `sentence-transformers 4.1.0` installed with `vllm==0.8.5 torch==2.6.0 transformers==4.51.3` pinned (dry run: additions only, nothing upgraded). The split files the pool builder needs (items, positives, instances, embeddings) were copied to ADA.
  - **Jobs:** training pools 40041 (Amazon) and 40042 (EB-NeRD), submitted 2026-10-26, pending on the account's group CPU limit.

### A1-ctrl letter codes (2026-10-25, branch `fix/a1ctrl-a3-prompts`; pending team decision)
- **Problem (code review):** `render.candidate_codes` seeds the codes with the candidates' item ids. A1-ctrl replaces A1's 3 traps with fill items, so every one of its 15 candidates got a different code from A1, although `derive` keeps every shared candidate in the same prompt position. On the EB-NeRD K = 15 dev pools, 0 of 450 A1-ctrl prompts (150 pools × 3 shuffles) shared A1's codes; 13 of 6,750 slots matched by chance. The trap effect (R_A1-ctrl − R_A1) therefore also compared two code sets. A2–A5 keep A1's or the baseline's item ids and were not affected; pointwise prompts have no codes.
- **Fix:** each A1-ctrl fill item carries `code_as` = the item id of the trap it replaces (`resize.a1ctrl_from`, and `oldgold.build_oldgold` for K = 50); `candidate_codes` uses `code_as` in place of the id, so A1-ctrl gets A1's codes slot by slot. `check_a1ctrl` asserts it; `test_oldgold_family` compares the codes (fails on the old code).
- **Effect on files:** rebuilding the EB-NeRD K = 15 pools changes only `k15_a1ctrl_trap_{dev,test}.jsonl` (byte-identical otherwise; the 150 dev pools differ only by `code_as`). Pools without `code_as` render exactly as before, so every other prompt and cache key is unchanged.
- **Rerun needed:** Direct-Rec, SSC and Structural on A1-ctrl (pointwise is unaffected). Qwen2.5-32B EB-NeRD dev: stage `a1ctrl-fix-dev` on pkgpu (3 tasks). Llama-2 K = 15 A1-ctrl Direct-Rec (Amazon and EB-NeRD) and Qwen Amazon A1-ctrl would need the same.
- **Rerun result (2026-10-25, pkgpu, 1 × RTX 5000 Ada, 1 h 15 min):** Qwen2.5-32B EB-NeRD dev A1-ctrl with A1's codes. Direct-Rec R_A1-ctrl 0.040 → **0.093**, trap effect −0.053 [−0.100, −0.007] → **+0.000 [−0.053, +0.053]**: the significant Direct-Rec trap effect of the pilot came from the code mismatch alone. SSC −0.040 → −0.047 and Structural −0.040 → −0.033 (both n.s. before and after); pointwise unchanged (−0.013). Outputs: `a1ctrl_fix_results.zip`; reports `outputs/pkgpu_a1fix_eval/`.

### A3 worked examples without dates (2026-10-26, branch `fix/a1ctrl-a3-prompts`; pending team decision)
- **Problem (code review):** A3 removes every displayed date of the task (history and candidates), but the worked examples (Base, Fast, Structural, Structural without recency) kept their dates, so an A3 prompt still showed dated items and the "launch date" / "published" field. Direct-Rec, SSC and pointwise have no worked examples and were not affected.
- **Fix:** `render._user_text` checks whether the task shows any date; if not, `_example_block` renders the example history and candidates without dates. The example reasoning never mentions dates (asserted by `test_examples_gold_is_not_newest`), so it stays valid. Test `test_hidden_dates_also_hidden_in_examples`.
- **Effect on prompts:** hashing every listwise prompt of the 16 EB-NeRD K = 15 dev pool files × 6 methods with the old and new code: only the 8 groups A3 × {Base, Fast, Structural, Structural without recency} change; the other 88 are byte-identical.
- **Rerun needed:** in `pilot-dev` only Structural ran on A3 (random, hard). Qwen2.5-32B EB-NeRD dev: stage `a3-fix-dev` on pkgpu (2 tasks). Qwen Amazon A3 Structural (ADA) would need the same; the Llama-2 K = 15 runs used only Direct-Rec and pointwise on A3 and are unaffected.
- **Rerun result (2026-10-26, pkgpu GPU 0, 1 h 10 min, status 0):** Structural R_hide went from 0.060 to 0.067 (random) and from 0.027 to 0.053 (hard). Δevidence (hard) changed from −0.040 [−0.060, −0.013], which was significant, to −0.013 [−0.047, +0.027], which is not. Δrecency changed from +0.013 to −0.013 (hard) and from +0.007 to +0.000 (random); neither is significant. So the pilot's "Structural is below chance without dates" came from the dated examples alone. Parse failures per prompt on A3 changed little, from 17–18% to 16%. Evaluation (pilot + fixed A1-ctrl + fixed A3): `outputs/pkgpu_a3fix_eval/`; outputs: `a3_fix_results.zip`.
- **Both fixes merged into `main` (2026-10-26, fast-forward to `c0e8258`) after verification:** test suite 94 passed on the branch; rebuilding the K = 15 pools (`adversarial.resize`) in a scratch copy changes only `k15_a1ctrl_trap_{dev,test}.jsonl` for both datasets (Amazon 144 / 967, EB-NeRD 150 / 1,000 pools), and those differ from the old files only by `code_as`; A1-ctrl prompts now carry A1's letter codes in 432/432, 2,901/2,901, 450/450 and 3,000/3,000 (pool × shuffle) cases (old: 0 in each). Every Llama-2 K = 15 A2/A3 prompt (pointwise and Direct-Rec, dev and test, both datasets) renders byte-identically with the old and new code: **the Llama-2 A2/A3 results (Δrecency, swap sensitivity) and every pointwise result are unaffected**. Still to do: replace the A1-ctrl pool files with the rebuilt ones and rerun Llama-2 Direct-Rec on A1-ctrl (both datasets, dev + test) and Qwen2.5-32B Amazon dev on A1-ctrl (Direct-Rec, SSC, Structural) and A3 (Structural).
- **Rebuild timing (team decision, 2026-10-26):** the K = 15 pools are rebuilt with the fix only **after** the fine-tuning experiment. Fine-tuning (training pools, training, `ft-dev` / `ft-test`) is pointwise-only, and pointwise prompts are byte-identical with the old and rebuilt A1-ctrl pools, so its results do not depend on the rebuild. The listwise A1-ctrl / A3 reruns follow the rebuild.
- **Training pools built (jobs 40041 / 40042, gnode058, 6 min / 7.5 min, exit 0):**
  - **Amazon:** 4,183 pools (3 dropped for too few cold negatives), 3,493 users, 2,007 distinct golds; 2,087 random / 2,096 hard. Per-cutoff SASRec val NDCG@10 0.065–0.075 (test split 0.064). The correct item is the newest candidate in 27–30% of pools (test pools 33–35%); the gold is ~1,010 days newer than the median warm item.
  - **EB-NeRD:** 5,246 pools (4 dropped), 4,392 users, 583 distinct golds; 2,609 random / 2,637 hard. SASRec val NDCG@10 0.254–0.298 (test 0.259). Correct item newest in 66–68% of pools (test 52–57%): the date shortcut is, if anything, stronger in the news training data; the gold is ~0.8 days newer than the median warm item.
- **ADA storage (2026-10-26, user decision):** the Qwen2.5-32B-AWQ weights (18 GB, finished with) were moved from `~/IRE_Project/hf_cache` to `/share1/jenik.gajera/hf_cache_archive/hub/` (copy, checksum verification, then removal from home), to make room for the fp16 Llama-2-7B base (13.5 GB). `/share1` is mounted on the login node only: to run Qwen again, copy it back (or set `HF_HOME` accordingly on ada-gw1).
- **Base weights (2026-10-26):** Meta access granted; `meta-llama/Llama-2-7b-chat-hf` @ `f5db02db…` downloaded on ada-gw1 (safetensors only) and checked against the official SHA-256 (`66dec18c…` / `0fd68950…` shards, `9e556afd…` tokenizer). An ungated mirror (`NousResearch/Llama-2-7b-chat-hf`) has byte-identical weights but different config/tokenizer files (an extra `<pad>` token, other JSON blobs), so it was not used. Note: the official `config.json` has `rms_norm_eps = 1e-05`, TheBloke's AWQ copy (all zero-shot Llama-2 runs) has 1e-06 — the fp16 zero-shot reference (`llama2_7b_fp16`) is therefore required for a like-for-like before/after, not only because of quantization.
- **Training pilot, job 40089 (Amazon, 60 steps, gnode075, 12 min 52 s, exit 0):** SFT data 23,778 train / 1,320 validation examples (3,309 / 184 users), positive rate 1/6, prompts 524 tokens on average (p95 909, max 1,040), none trimmed. Model load 2 min 22 s; ~5 s per optimizer step of 32 examples once warm; peak GPU memory 10.4 of 11 GB per GPU (batch 4, gradient checkpointing). Training loss fell steadily, no NaN/overflow in fp16; validation loss after 60 steps **0.356** (predicting the 1/6 base rate gives 0.451). Full Amazon run estimated at ~743 steps ≈ 1–1.5 h.
- **Loss-scaling fix (after the pilot):** with transformers 4.51 the `Trainer` does not divide the loss by the accumulation steps when the model's forward takes `**kwargs` (PEFT/Llama do), and our `compute_loss` returns a per-batch mean — so the pilot's logged training loss and its gradients were 2× (logged 27.4 → 0.85, i.e. 13.7 → 0.42 real; validation loss is unaffected). Adam is scale-invariant, but gradient clipping (0.3) acted at half its intended threshold. `train.py` now sets `model_accepts_loss_kwargs = False`, so the Trainer averages over accumulation steps. The full runs use the fixed code.
- **Full Amazon run:** job 40143 submitted 2026-10-26.
- **Amazon fine-tune done (job 40143, gnode079, 1 h 43 min, exit 0):** 743 steps, 1 epoch over 23,778 examples (~8.4 s/step on this node). Validation loss 0.322 → 0.307 → 0.274 → … → **0.240** at the end (base-rate predictor 0.451); final training loss ~0.23; no NaN. Adapter: `outputs/finetune/amazon_vg/llama2_7b/adapter/` (160 MB, on ADA).
- **Submitted 2026-10-26:** EB-NeRD fine-tune (job 40506), and Amazon `ft-dev` evaluation for the fine-tuned model (40507, `llama2_7b_ft_amazon_vg`) and the zero-shot fp16 reference (40508, `llama2_7b_fp16`), `TP=2`. The evaluation uses the current (pre-fix) A1-ctrl pools: pointwise prompts are identical either way (rebuild after fine-tuning, team decision).
- **EB-NeRD dev evaluations queued:** zero-shot fp16 (job 40526) and fine-tuned (job 40527, starts after training job 40506 succeeds). **Rebuild timing refined (team decision):** the K = 15 pools are rebuilt right after these four `ft-dev` evaluations (40507, 40508, 40526, 40527), before `ft-test` or any other run.
- **Resubmitted with smaller requests (2026-10-26):** the queued jobs requested 36 CPUs / 100 GB (copied from the inference script) and waited on the account's group CPU limit. Training needs ~12 processes (4 ranks + 2 data loaders each), evaluation 2 vLLM workers + a one-time CPU merge, so they were cancelled while pending and resubmitted with `--cpus-per-task=16 --mem=80G`; Slurm raises this to **28 CPUs** for 4 GPUs (its per-GPU minimum). `ft_train.sbatch` now sets `OMP_NUM_THREADS` = CPUs / 4 instead of a fixed 8. New ids: EB-NeRD fine-tune **40530**; `ft-dev` evaluations **40531** (Amazon fine-tuned), **40532** (Amazon zero-shot fp16), **40533** (EB-NeRD zero-shot fp16), **40534** (EB-NeRD fine-tuned, after 40530).
- **Evaluations moved to 2-GPU jobs:** still blocked by the group CPU limit, the four `ft-dev` evaluations were cancelled while pending and resubmitted with `--gres=gpu:2` (one vLLM worker, TP = 2, `NGPU=2`), which Slurm sizes at 21 CPUs instead of 28: **40537** (Amazon fine-tuned), **40538** (Amazon zero-shot fp16), **40539** (EB-NeRD zero-shot fp16), **40540** (EB-NeRD fine-tuned, after 40530). Same model, tasks and outputs; only the worker count changes. The EB-NeRD fine-tune stays a 4-GPU job (40530).
- **Failures (2026-10-26) and fixes:**
  - **Evaluations 40537–40539 failed** (exit 1 after ~45 min, all on **gnode011**, a GTX 1080 Ti node in `u22`): the job saw one GPU, so vLLM's TP = 2 fell back to Ray and timed out waiting for a second GPU (`Cannot provide a placement group … within 1800 seconds`). Pascal GPUs are not supported by vLLM anyway. All jobs now request **`--constraint=2080ti`** (45 labelled RTX 2080 Ti nodes in `u22`; every earlier successful job ran on 2080 Ti nodes).
  - **EB-NeRD fine-tune 40530 ran out of GPU memory at step 1** (gnode092): news SFT prompts average 1,243 tokens (p95 1,349, max 1,419; Amazon 524). `configs/ebnerd.yaml` sets `finetune.train_overrides: {per_device_batch: 2, grad_accum: 4}`, so the effective batch stays 32 (4 × 2 × 4) with the same optimizer schedule; `train.py` merges the override and records the settings used in `train_summary.json`. `ft_train.sbatch` also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. EB-NeRD SFT data: 30,018 train / 1,458 validation examples (4,181 / 211 users), 39.1M prompt tokens in total (~3× Amazon), so ~5 h expected.
  - **Resubmitted:** EB-NeRD fine-tune **40929**; `ft-dev` evaluations (2 GPUs each) **40930** (Amazon fine-tuned), **40931** (Amazon zero-shot fp16), **40932** (EB-NeRD zero-shot fp16), **40933** (EB-NeRD fine-tuned, after 40929).
- **Amazon dev, fine-tuned vs zero-shot (preliminary; zero-shot = AWQ runs until the fp16 reference 40931 is in):** Recall@1 0.373 / 0.460 (random / hard; zero-shot 0.167 / 0.113; newest-first 0.347 / 0.307). Dates hidden: 0.093 / 0.087 (Δrecency +0.280\* / +0.373\*, Δevidence +0.027 / +0.020 n.s.). Swap sensitivity +0.367\* / +0.453\* (zero-shot +0.067\* / +0.047\*). Trap effect +0.250\* (A1 0.139 vs A1-ctrl 0.389; Trap@1 0.528, zero-shot 0.167). Flood lift (n = 10) +0.576\* (zero-shot +0.125\*). All-new pools (A6, dates shown): 0.240 / 0.280 vs newest-first 0.087 and random 0.06–0.09; there the gold's date is uniform over the pool, the model's top-1 is not concentrated on newer items, and the most history-similar item is the gold in only 23% / 1% of pools — so the model also learned content. **Reading:** fine-tuning learned relevance plus a strong date shortcut that wins when they conflict (swap, traps, flood). A3 alone likely overstates the recency share because the model never saw date-less prompts (format shift).
- **Dates-hidden training control (team decision, 2026-10-26): deferred** — to be run later (same SFT data with all dates removed, evaluated on A3) to give a clean content-only ceiling. Until then the swap / trap / flood results are the primary evidence.
- **fp16 zero-shot reference (job 40931, gnode090, 21 min, exit 0; pointwise check max diff 0.023):** zero-shot Llama-2-7B on Meta's fp16 weights is **at chance on Amazon dev and shows no date effect**: baseline 0.080 / 0.060 (AWQ: 0.167 / 0.113; fp16 − AWQ = −0.087\* / −0.053, paired); A2 0.080 / 0.053, A3 0.073 / 0.073, A4 0.067 / 0.060, A6 0.067 / 0.067 (all Δ n.s.); A1-ctrl 0.146, A1 0.118, A5 n = 1/2/5/10 0.111 / 0.104 / 0.118 / 0.104 (n.s.). Scores are sane (mean 5.37, sd 1.20; correlation 0.93 with the AWQ scores) but crowded (median within-pool spread 0.68 log-odds; ties in 71 / 150 pools), so small numeric differences flip the top-1. **Consequence:** the zero-shot recency effect reported for Llama-2 (AWQ: Δrecency +0.074\* test) is specific to TheBloke's 4-bit AWQ checkpoint (also `rms_norm_eps` 1e-06 vs Meta's 1e-05) and does not appear on the original weights (dev; test pending in `ft-test`). Like-for-like, **fine-tuning lifts Recall@1 by +0.293\* / +0.400\*** (vs fp16 zero-shot) and the gain collapses under every date manipulation: fine-tuning creates the date shortcut.
- **EB-NeRD fp16 zero-shot (job 40932, 31 min, exit 0), dev:** unlike Amazon, fp16 matches the AWQ checkpoint: baseline 0.100 / 0.073 (AWQ 0.107 / 0.080); A2 0.060 / 0.040 (Δ −0.040\* / −0.033), A3 0.060 / 0.040 (−0.040 / −0.033), A4 0.047 / 0.027 (−0.053\* / −0.047\*), A6 0.107 / 0.084 (n.s.); A1-ctrl 0.020, A1 0.047, A5 0.027–0.047 (n.s.). A small date sensitivity (swap, shuffle) on a near-chance baseline, as with AWQ. The EB-NeRD fine-tune (40929) is waiting for a free 4-GPU 2080 Ti node; its evaluation (40933) follows.
- **Amazon pools rebuilt with the A1-ctrl fix (2026-10-26)**, after the Amazon `ft-dev` evaluations and before `ft-test` (team decision): `adversarial.resize --config amazon_vg`; only `k15_a1ctrl_trap_{dev,test}.jsonl` changed (967 test pools now carry `code_as`); both pushed to ADA; the previous K = 15 files are kept in `pools_backup_pre_a1ctrl_fix/` (local). EB-NeRD is rebuilt after its fine-tuned `ft-dev` evaluation (40933).
- **Amazon test evaluations queued:** `ft-test` (pointwise, baseline + all 10 variants; 800 cold / 967–1,000 old-target users), 2 GPUs each: **41008** (fine-tuned), **41009** (zero-shot fp16).
- **Amazon test, fine-tuned model (job 41008, gnode055, 2 h 01 min, completed 2026-10-27):** replicates dev in every row (same sign, all McNemar p < 1e-20). Baseline 0.445 / 0.546 (newest-first 0.346 / 0.323); A2 0.005 / 0.006; A3 0.101 / 0.107; A4 0.077 / 0.077; A6 0.253 / 0.292; A1-ctrl 0.384 → A1 0.152 (Trap@1 0.562); A5 n = 1/2/5/10: 0.103 / 0.084 / 0.067 / 0.060 (Trap@1 0.675 → 0.930). Re-scoring dev with the same script reproduces the dev table exactly. Table in `results.md`; the zero-shot column waits for job 41009, which is queued behind EB-NeRD fine-tune 40929 (started 2026-10-27, gnode071) because of the 40-CPU QoS limit.
- **Amazon test, zero-shot fp16 (job 41009, gnode090, 1 h 57 min, exit 0):** at chance with no meaningful date effect: baseline 0.083 / 0.072; A2 0.086 / 0.066; A3 0.068 / 0.061; A4 0.083 / 0.066; A6 0.076 / 0.053 (hard −0.020\*); A1-ctrl 0.101, A1 0.087 (−0.014\*, Trap@1 0.198 = chance), A5 0.089–0.093 (n.s.). Fine-tuned vs zero-shot baseline +0.362\* / +0.474\*. `results.md` Amazon test section replaced with the complete table.
- **EB-NeRD fine-tune (job 40929, gnode071, 5 h 01 min, exit 0):** 938 steps, final validation loss 0.200. **Dev evaluation (job 40933, 31 min):** baseline 0.433 / 0.413 (zero-shot 0.100 / 0.073; **newest-first 0.567 / 0.540, so the fine-tuned model is below the date rule**); A2 0.000 / 0.000; A3 0.007 / 0.007; A4 0.053 / 0.067; A6 0.115 / 0.084 (chance; n = 131); A1-ctrl 0.247 → A1 0.127 (−0.120\*, Trap@1 0.767); A5 n = 1/2/5/10 0.107 / 0.107 / 0.107 / 0.100 (Trap@1 0.827 → 0.887, saturated). All fine-tuned changes p < 1e-4. The fine-tuned news model learned the date rule and no usable content signal (unlike Amazon, where A6 is ~4× chance). Scored with the same per-instance McNemar script as Amazon; EB-NeRD A1-ctrl pools are still the pre-fix files (pointwise prompts identical). Table in `results.md`. Next: rebuild the EB-NeRD A1-ctrl pools, then EB-NeRD ft-test.
- **EB-NeRD pools rebuilt with the A1-ctrl fix (2026-10-27)**, after the EB-NeRD `ft-dev` evaluation and before `ft-test`: `adversarial.resize --config ebnerd`; only `k15_a1ctrl_trap_{dev,test}.jsonl` changed (sha256 1428d92d… / 72113ed3…), and the records differ from the old files only by the per-candidate `code_as` field (150 / 1,000 checked). Pushed to ADA; old K = 15 files in `pools_backup_pre_a1ctrl_fix/` (local). Both datasets are now rebuilt.
- **EB-NeRD test evaluations queued:** `ft-test`, 2 GPUs each: **41455** (fine-tuned), **41456** (zero-shot fp16).
- **Fine-tuning data audit (2026-10-27, run on ADA against the SFT and pool files the jobs used; script kept in the scratchpad, not the repo):**
  - **Prompts are identical to evaluation:** for 100 random training pools per dataset, the training prompts equal byte for byte what `llm.run.run_pointwise` (the evaluation code) sends to the model: 600/600 on Amazon and 600/600 on EB-NeRD.
  - **Dates shown:** the candidate's date appears in 25,098/25,098 (Amazon) and 31,476/31,476 (EB-NeRD) training prompts.
  - **Labels:** "Yes" exactly for the gold (Amazon 4,183 Yes / 20,915 No; EB-NeRD 5,246 / 26,230).
  - **No leakage** (UTC, as in the data code):
    - 0 users shared with any dev/test pool (Amazon 3,493 train vs 1,945 dev/test users; EB-NeRD 4,392 vs 1,996);
    - 0 training targets at or after T (latest 2022-08-26 22:33 and 2023-05-24 23:18);
    - 0 of the new dev/test golds (485 / 94) occur as training golds.
    - Old A1/A5 targets can be earlier training golds (177/905 and 60/162). This is allowed: those items existed before T, and it can only help the model on A1.
  - **Same protocol as the test pools:** 12 warm + 2 cold + gold in every pool, and the gold is always a new item. The gold is the single newest candidate in 28% of training vs 34% of test pools (Amazon), and in 66% vs 54% (EB-NeRD).
- **ADA home cleanup (2026-10-27, user request):** deleted the HF caches of Qwen-7B-Chat-Int4, Mistral-7B-Instruct-v0.1-AWQ and TheBloke/Llama-2-7B-Chat-AWQ (13 GB; home now 31 / 50 GB). Rerunning the original AWQ zero-shot Llama-2 results, or the early Qwen-7B / Mistral runs, needs a re-download (`scripts/download_models.py`). Kept: Meta's fp16 Llama-2-7B-Chat (base of both fine-tuned models, used by the running EB-NeRD test jobs) and all-MiniLM-L6-v2. Qwen2.5-32B-AWQ stays in `/share1/jenik.gajera/hf_cache_archive/`.
- **EB-NeRD test (jobs 41455 fine-tuned, gnode085, 2 h 48 min; 41456 zero-shot fp16, gnode055, 2 h 59 min; both exit 0):** replicates dev.
  - **Fine-tuned:** baseline 0.394 / 0.487 (newest-first 0.560 / 0.515); A2 0.001 / 0.000; A3 0.014 / 0.013; A4 0.035 / 0.031; A6 0.097 / 0.117 (n = 650); A1-ctrl 0.241 → A1 0.106 (Trap@1 0.828); A5 0.104–0.113 (Trap@1 0.846 → 0.888); all changes p < 1e-26.
  - **Zero-shot:** baseline 0.055 / 0.054, with small significant date effects (A2 −0.019\* / −0.015\*, A3 −0.031\* / −0.029\*, A4 −0.013\* / −0.014\*), A1 Trap@1 0.183.
  - Table in `results.md`. All four fine-tuning evaluations (Amazon, EB-NeRD × dev, test) are complete.

### Fine-tuning Qwen2.5-7B on EB-NeRD (exploratory, 2026-10-26, branch `ft/qwen25-7b`; pending team decision)
- **Scope:** Qwen2.5-7B-Instruct was released in Sept 2024, so it is **outside the 2023-only rule (§7)**. This is an exploratory second fine-tuned model next to Llama-2-7B, on pkgpu (one RTX 5000 Ada, 32 GB, bf16). It is not a study model until the team decides.
- **Models (`configs/models.yaml`):**
  - `qwen25_7b`: `Qwen/Qwen2.5-7B-Instruct` @ `a09a3545…`, full bf16 weights, ChatML, `max_model_len` 4096. It is the base and also the zero-shot reference with the same weights.
  - `qwen25_7b_ft_ebnerd`: `base: qwen25_7b`, adapter `outputs/finetune/ebnerd/qwen25_7b/adapter`, TP 1.
- **`dtype` per model** (optional): read by both engines and by `finetune.merge`. The default stays float16, so every existing model, cache key and merged Llama-2 model is unchanged. Qwen2.5 runs and merges in bfloat16, its training dtype.
- **`configs/finetune_qwen25_7b.yaml`:** the same QLoRA values as `finetune.yaml`, except bf16 compute and one GPU with batch 8 × accumulation 4. The effective batch stays at 32, as for Llama-2. `train.py` sets `fp16` / `bf16` from the compute dtype.
- **Answer token by chat format:** `train.answer_ids` used to probe with `"[/INST] Yes"`, which is Llama-2 specific. It now appends the label to the family's real prompt ending (`format_chat`):
  - ChatML ends in `"assistant\n"`, so the label is the token `"Yes"` with no space.
  - Llama-2 ends in `"[/INST]"`, so it stays `" Yes"`, the same token as before.
  - In both cases the label must be one of the scorer's Yes / No ids.
- **SFT files per model:** the prompts depend on the tokenizer, which sets the history trimming, and on the chat format. `sft_data.sft_paths` gives each model its own files: `sft_{train,val}_<model>.jsonl` and `sft_stats_<model>.json`. Llama-2 (`llama2_7b_fp16`) keeps the unsuffixed names the ADA scripts expect, so Qwen cannot overwrite Llama's data.
- **Training pools:** the EB-NeRD training pools do not depend on the model, so the ones from ADA job 40042 (`finetune/train_pools_k15.jsonl`) are reused.
- **Prerequisite on pkgpu:** install `peft 0.15.2`, `bitsandbytes 0.45.5` and `accelerate 1.6.0` into `/scratch/sriharinim/ire_venv`, with vllm 0.8.5, torch 2.6.0 and transformers 4.51.3 pinned. Set `COLDRANK_MERGED_DIR=/scratch/sriharinim/coldrank_merged`, because pkgpu has no `/ssd_scratch`.
- **Planned run order (one fully free GPU):**
  1. `python -m coldrank.finetune.sft_data --config ebnerd --model qwen25_7b`
  2. A pilot: `python -m coldrank.finetune.train --config ebnerd --ft-config finetune_qwen25_7b --max-steps 60` (memory and speed).
  3. 1 epoch.
  4. Stage `ft-dev` (pointwise on the K = 15 baseline and variants) for `qwen25_7b` and `qwen25_7b_ft_ebnerd`.
  5. The decomposition.

### Production-style training data (impression negatives), EB-NeRD, Llama-2-7B and Qwen2.5-7B (2026-10-27, user decision)
- **Why.** In the pool-based training data (`trainpools` + `sft_data`), **every warm negative is older than every cold candidate in 100% of training pools** (5,246/5,246; dev pools: also 100%), and the gold is the newest candidate in 67% (dev 54–57%). Pointwise SFT on it therefore rewards "old means No" by construction, which the Llama-2 results show (A2/A3 collapse to ~0). That is a property of the cold-start test protocol, not of how Li et al. train (see the correction above), nor of production news training, where the negatives are **impressions: articles shown to the user and not clicked**.
- **Data (`coldrank/finetune/impressions.py`):**
  - **Instances:** `trainpools.training_instances` unchanged (test split rules, `min_history` 5, `require_cold_negs` 9, all dev/test users excluded, seeded cap 750 per cutoff), so the gold is still a new article at T′ and the instance distribution matches the test set.
  - **Cutoffs:** daily **2023-05-18 … 2023-05-24**, 1-day windows (`finetune.impressions` in `configs/ebnerd.yaml`). EB-NeRD impressions (behaviors) exist only from 18 May; the clicks of 6–17 May come from the history files without impressions, so the pool cutoffs of 6–15 May cannot be used.
  - **Negatives:** the user's impression(s) at the target time whose clicks contain the gold; negatives = its in-view articles minus everything clicked there, the user's window items and history, kept if the article is known and launched by the target time; **any age**; up to **5** per instance (seeded by instance id), so 1 positive : 5 negatives as before. Instances with no such impression or no negative are dropped and counted. No SASRec warm negatives (no SASRec needed: CPU only).
  - **Examples:** gold "Yes", each negative "No", prompt = `sft_data.pointwise_prompt` (exactly the inference prompt, history trimmed to fit). Users split train / validation with `sft_data.is_holdout` (5%).
  - **Leak checks:** no dev/test user, every target before T, every negative launched by the target time.
  - **Outputs** in `<split>/finetune/impressions/`: `instances.jsonl`, `sft_{train,val}_<model>_imp.jsonl`, `sft_stats_<model>_imp.json`. The stats include **date-cue diagnostics**: share of instances where the gold is newer than all its negatives, median ages (hours at the click) of gold and negatives, share of negatives that are new at T′. Check these before training: they say how much of the date shortcut is left.
- **Training:** same QLoRA recipe; only the data changes. `sft_variant: imp` in the ft config selects these files (`sft_data.sft_dir` / `sft_paths(..., variant)`) and writes to `outputs/finetune/ebnerd/<base>_imp/` (a new directory, so no checkpoint of a pool-data run is resumed).
  - Llama-2: `configs/finetune_imp.yaml` = `finetune.yaml` + `sft_variant: imp`; EB-NeRD's `train_overrides` apply (4 GPUs × 2 × 4 = 32, as job 40929). Model `llama2_7b_ftimp_ebnerd`.
  - Qwen2.5-7B: `configs/finetune_qwen25_7b_imp.yaml`; model `qwen25_7b_ftimp_ebnerd`.
- **Batch-override precedence:** `train.train_settings` applies the dataset's `finetune.train_overrides` unless the ft config sets `dataset_train_overrides: false`. Both Qwen configs set it (their 1 GPU × 8 × 4 = 32 was sized for one 32 GB GPU; the EB-NeRD override would have made it 1 × 2 × 4 = 8). Llama-2 is unchanged.
- **Qwen pool-data run aborted (2026-10-27):** it ran on the pre-loss-fix `train.py` (gradients 4× with accumulation 4) and was stopped by the GPU watchdog at step 150 of 938 (another user on the GPU). Its `checkpoint-100` is kept aside, not resumed. Qwen restarts on impression data with the fixed code.
- **Unchanged:** Amazon (no impressions), the pool-based data and runs, the evaluation (K = 15 test pools and variants).
- **How to run (ADA, Llama-2):**
  1. `python -m coldrank.finetune.impressions --config ebnerd --model llama2_7b_fp16` (CPU; needs `data/processed/ebnerd/impressions.parquet` and the split's items / positives / instance files).
  2. `torchrun --nproc_per_node 4 -m coldrank.finetune.train --config ebnerd --ft-config finetune_imp`. Note: `ft_train.sbatch` (git-ignored) checks for `sft_train.jsonl`; for this variant it must check `finetune/impressions/sft_train_llama2_7b_fp16_imp.jsonl` and call `coldrank.finetune.impressions` instead of `sft_data`.
  3. `ft-dev` / `ft-test` pointwise for `llama2_7b_ftimp_ebnerd` (the zero-shot `llama2_7b_fp16` runs are reused).
- **Qwen (pkgpu):** the same with `--model qwen25_7b` and `--ft-config finetune_qwen25_7b_imp`, one GPU.
- **Review of the impression-negative data (2026-10-27, before any run):**
  - **Li et al. correction verified** against the PDF (`readings/li2026_…pdf`, §3.1.2: "for the SFT training set, we sampled 16,218 interactions after removing cold-start items"; §3.3: "the training data excludes cold-start items"). Our pool-based training data, where the gold is always new, is therefore **not** Li et al.'s SFT data. That data is the cold-start test protocol used as training data, and the paper should describe it that way. The earlier statements here and in chat that it replicates Li et al.'s training labels were wrong.
  - **Code:** 102 tests pass on `main` at `91f1715`. `impressions.py` is consistent with the pool pipeline: the same `training_instances`, the inference prompt, leak asserts, and `is_cold` measured against the instance's own T′.
  - **Date cue removed** (local CPU run of the instance and negative steps; no tokenizer needed):
    - the gold is newer than all its negatives in 11.3% of instances;
    - the gold is the newest of its group in 11.7%, against 18.2% expected by chance;
    - median age at the click is 1.8 h for the gold and 1.5 h for the negatives;
    - 80.5% of the negatives are themselves new at T′.

    Pool data, by comparison: the gold is newest in 66%, and every warm negative is older than every cold candidate in 100% of pools.
  - **Size:** only **1,677 of 5,250 instances** keep a negative (≈ 4.6 negatives each, so ≈ 9.4k examples against 31.5k for the pool data). The drops are real, not a matching bug. On 21 May (750 instances), the gold click has an impression at the same timestamp for 257. For the rest:
    - 352: the gold never appears in the user's logged impressions (the click comes from the history log, and EB-NeRD's behaviour logs are a sample);
    - 136: the gold is only shown, not clicked, in a logged impression;
    - 5: clicked in an impression hours away.

    Training on about a third of the data is a confound when comparing against the pool-data model.
  - **Amazon has no impressions**, so this ablation exists for EB-NeRD only.
