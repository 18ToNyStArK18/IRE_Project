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
- Update report Eq. 1 / §3.1 to the global cutoff; add `T`, `w`, history length, SASRec, and quantization as a limitation.
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
