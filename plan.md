# Plan: Reproducing the Netflix Cold-Start Reranking Baseline

Baseline paper: Li et al., *LLM Reasoning for Cold-Start Item Recommendation*, WWW '26 (`readings/li2026_llm_reasoning_cold_start_netflix.pdf`).

Goal: rebuild their **evaluation protocol and prompting methods** on public data, so that the adversarial variants (A1–A6, `report/report.tex` §3.5) can later be applied to the same pools. We reproduce the protocol, not their numbers: their data, production ranker and prompts are not public.

**Decisions made**
- **Cold-start definition:** a single **global cutoff**, as in Li et al. (*not* per-user).
- **First domain:** **Amazon Video Games**. EB-NeRD is ported after the Amazon pipeline works.
- **Hardware:** 1× **RTX 2080 Ti** (11 GB, Turing sm_75). It has no bf16 and no FlashAttention-2, so run fp16, with 4-bit weights for 7B/13B models.

---

## 1. What Li et al. do, and what we use instead

| Component | Li et al. (Netflix) | Ours (Amazon VG first) |
|---|---|---|
| Cold-start item | Launched after the latest interaction in **all** user histories, and after the LLM's training cutoff (§2.1) | Launch date `t0 > T`, where `T` is one global cutoff, chosen ≥ 2023-01 if the data allows (§3 below) |
| Cold items in training data | Never appear in train/test histories; used only as targets (§3.1.1) | Same: interactions before `T` contain no cold items by construction |
| Test set | Thousands of interactions from **one day** whose target is cold (§3.1.2) | Interactions in a window `[T, T+w]` whose target is cold; `w` chosen so there are enough instances |
| Candidate pool | Top-40 from the production ranker + 10 sampled cold titles including the target = 50 (§2.1) | Top-40 from **SASRec** trained on data before `T` + 10 cold items (target + 9 negatives, random **and** hard) |
| Item attributes in prompt | Title, launch date, genre (§2.2.1) | Title, launch date, category, plus brand/platform for Structural Reasoning |
| Base LLM | Qwen-2.5-32B-Instruct | Phi-2, Llama-2-7B/13B-Chat, Mistral-7B-Instruct, Qwen-7B-Chat (all from 2023) |
| Methods | Direct-Rec, Base-Reason (2-shot), Fast-Reason (1-shot, 2× history), Structural Reasoning, Soft Self-Consistency | All five, plus the **pointwise yes/no** control |
| Fine-tuning | SFT (QLoRA 4-bit, 1 epoch), GRPO, SFT+GRPO | None in the main runs; QLoRA SFT is a stretch goal (Phase 8); no GRPO |
| Metric | Recall@1 on Discovery (primary) and AnyPlay | Recall@1 (every target is new to the user, so it plays the role of Discovery), plus NDCG@10, Cold@1 and parse-failure rate |
| Reported as | Relative % gains only | **Absolute** numbers next to the date-only baselines |

What Li et al. do not specify, and we must decide ourselves (each choice is recorded in `configs/`):
- how the 9 cold negatives are sampled;
- the exact prompt texts;
- the history length;
- the number of reasoning paths in Soft Self-Consistency;
- the output format.

---

## 2. Repository layout

```
IRE_Project/
├── plan.md
├── report/                    # LaTeX report (exists)
├── readings/                  # papers (exists)
├── configs/
│   ├── amazon_vg.yaml         # cutoff T, window w, history length, K, seeds
│   └── models.yaml            # HF ids, quantization, context limits
├── src/
│   ├── data/
│   │   ├── amazon_vg.py       # load reviews + metadata, parse launch dates
│   │   ├── ebnerd.py          # (Phase 7)
│   │   └── split.py           # global-cutoff split, test-instance selection
│   ├── warm/
│   │   └── sasrec.py          # train/score warm ranker (reuse Dong et al. code)
│   ├── pools/
│   │   ├── build.py           # 40 warm + 10 cold pools, random/hard negatives
│   │   └── schema.py          # pool record definition (see §4.3)
│   ├── prompts/
│   │   ├── templates/         # one .txt per method (+ few-shot examples)
│   │   └── render.py          # pool + condition -> prompt string
│   ├── llm/
│   │   ├── engine.py          # vLLM wrapper (HF fallback), prompt cache
│   │   ├── listwise.py        # generate + parse ranked IDs
│   │   └── pointwise.py       # log p(Yes) - log p(No)
│   ├── baselines/
│   │   └── heuristics.py      # random, random-among-newest, newest-first, popularity, content, cold-filter+content
│   └── eval/
│       ├── metrics.py         # Recall@1, NDCG@10, Cold@1, parse-failure rate
│       └── stats.py           # paired bootstrap, McNemar
├── scripts/                   # SLURM job scripts for ADA
├── data/                      # git-ignored: raw + processed datasets
└── outputs/                   # git-ignored: pools, raw generations, metrics
```

---

## 3. Phases

### Phase 0: Environment and data audit (do this first; it decides `T` and `w`)
- [ ] Set up the conda environment: Python 3.10, PyTorch with CUDA ≥ 11.8, `vllm` (check that it supports sm_75 on the installed version), `transformers`, `autoawq`/`auto-gptq`, `sentence-transformers`, `pandas`, `pyarrow`.
- [ ] Clone `github.com/dongzhe1/rec-diag-bench` at tag `0.0.0` into `external/` and check that its Video Games preprocessing runs.
- [ ] Download Amazon Reviews 2023, Video Games: reviews and metadata (McAuley lab).
- [ ] **Audit launch dates:**
  - What fraction of items has a parseable "Date First Available" in `details`?
  - For the rest, the fallback is the first-review timestamp.
  - Plot the item launch histogram and the interaction histogram over time.
- [ ] **Choose `T` and `w`:**
  - For candidate cutoffs (monthly, 2022-06 … 2023-06), count test instances. A test instance is a user with ≥ 5 interactions before `T` who interacts with an item where `t0 > T`, within `[T, T+w]`, for `w ∈ {7, 30, 60, 90}` days.
  - Pick the latest `T` (to stay after model training cutoffs) that gives ≥ 1,000 instances with `w` ≤ 90 days.
  - Record the choice and the reason in `configs/amazon_vg.yaml`.
- [ ] Positive feedback: treat every review as an interaction, matching Dong et al. (or set a rating threshold; decide once and record it).

**Deliverable:** a data audit notebook or report with the instance counts per `(T, w)` and the launch-date coverage.

### Phase 1: Global-cutoff split
- [ ] Item launch time `t0(i)` = Date First Available, otherwise the first review timestamp. Record which source was used for each item (needed for the "age proxy" limitation).
- [ ] **History** `H_u` = the user's interactions with timestamp < `T`. Keep the last `L` items (`L` = 20 for Base-Reason, 40 for Fast-Reason, adjusted to fit the context budget in §5).
- [ ] **Cold items:** `t0 > T`. **Warm items:** `t0 ≤ T` with ≥ 1 interaction before `T`.
- [ ] **Test instances:** for each user, the **first** interaction in `[T, T+w]` whose item is cold. That item is the target `g`. Only users with `|H_u| ≥ 5`.
- [ ] Check that no cold item appears in any history (should hold by construction).
- [ ] Subsample: a dev set of 200 instances for prompt debugging and a test set of N instances, with N fixed later by a power analysis (start with 1,000). Fix the seeds.
- [ ] Also store **old-target instances** (the next interaction in the window is a warm item). A1 needs these later, so build them now while the split code is fresh.

### Phase 2: Warm-start ranker (stand-in for the production ranker)
- [ ] Train **SASRec** on interactions before `T`, using Dong et al.'s implementation and hyperparameters. Validate on the last 10% of the time before `T`.
- [ ] Sanity check: report Recall@10 / NDCG@10 on warm next-items in the window. It must clearly beat popularity, otherwise it isn't a credible stand-in.
- [ ] For each test user: take the top-40 warm items, excluding `H_u`. Warm candidates are drawn only from items with `t0 ≤ T`; cold items have no embeddings, so SASRec cannot rank them anyway.
- [ ] Fallback if SASRec is weak: LightGCN from the same codebase.

### Phase 3: Pool construction
- [ ] Pool = top-40 warm + `g` + 9 cold negatives, for 50 candidates in total.
- [ ] Cold negatives, two settings, both stored:
  - **random:** uniform over cold items that launched before the test interaction time (an item launched after the user's click cannot be a real candidate), excluding `g`;
  - **hard:** the top-9 such items by cosine similarity between the SBERT embedding of `x_c` and the mean embedding of `H_u`.
- [ ] Store every pool as one JSONL record (schema in §4.3) with the **true dates** and a `condition` field. The adversarial variants later only rewrite fields of these records.
- [ ] Manipulation stats per pool: age gap between `g` and the warm items, and content similarity of `g` and of the negatives to `H_u`.

### Phase 4: Prompts (five methods + pointwise)
Li et al. do not publish their prompts, so we rebuild them from §2.2 and §3.2.1 of the paper and freeze them before the test runs. All prompt tuning happens on the 200 dev instances only.

| Method | What the prompt does | Output |
|---|---|---|
| Direct-Rec | History + candidates → "rank the candidates" | Top-10 candidate IDs |
| Base-Reason | (i) summarize the user's interests, (ii) reason about the candidates using that summary; **2 worked examples** | Reasoning, then `FINAL: [id] > [id] > …` |
| Fast-Reason | Base-Reason with **2× history length**, shorter example reasoning, **1 worked example** | Same |
| Structural Reasoning | (1) find the key preference factors (genre, brand/publisher, platform), build one path per factor from the history; (2) score each candidate against each path; (3) weight the paths by prominence, action type and **recency**, then aggregate. Also a **no-recency** variant with "recency" removed from step 3 | Path scores, then `FINAL:` ranking |
| Soft Self-Consistency | The model writes several (k = 3) different reasoning paths in **one generation**, then reconciles them softly into one ranking (no majority vote) | `FINAL:` ranking |
| Pointwise control | History + **one** candidate → "Will the user engage with this item? Answer Yes or No." | Score = log p(Yes) − log p(No) |

- [ ] Candidates appear as `[17] Title | 2023-05-02 | Category`, with IDs 1–50 assigned after shuffling. The history uses the same format, in chronological order.
- [ ] Worked examples use **short** pools (5 candidates) so they fit in the context window. They come from training-period data, never from test users.
- [ ] Ask for the **top-10** ranking (not just the top-1), so NDCG@10 can be computed. Unlisted candidates are tied at the end.
- [ ] Parser: take the IDs after `FINAL:`, de-duplicate them, and drop out-of-range IDs. If the result is empty, count it as a parse failure: the instance gets Recall@1 = 0 and is logged. Report the parse-failure rate per model × method.
- [ ] Use the chat template for each chat model. Phi-2 uses an `Instruct:/Output:` format.

### Phase 5: Inference engine (2080 Ti)
- [ ] **Models and quantization** (fix one quantized checkpoint per model and use it in every condition, so paired comparisons stay valid):

| Model | Weights | VRAM (approx.) | Context | Runs |
|---|---|---|---|---|
| Phi-2 | fp16 | ~5.5 GB | 2k | Pointwise only |
| Llama-2-7B-Chat | AWQ/GPTQ 4-bit | ~4 GB | 4k | All |
| Llama-2-13B-Chat | AWQ/GPTQ 4-bit | ~7.5 GB | 4k | All (small batches, short KV cache) |
| Mistral-7B-Instruct-v0.1 | AWQ/GPTQ 4-bit | ~4 GB | 8k (cap at 4k for parity) | All |
| Qwen-7B-Chat | Official Int4 GPTQ | ~5 GB | 8k (cap at 4k) | All |

- [ ] Engine: vLLM with `dtype=float16`, `max_model_len=4096`, `gpu_memory_utilization≈0.9`. Fallback: HF `transformers` + bitsandbytes if vLLM won't build for sm_75 on ADA.
- [ ] Decoding: greedy (temperature 0) for every listwise method. Max new tokens: 64 for Direct-Rec, about 1,024 for the reasoning methods (cap and log truncations).
- [ ] Pointwise: one forward pass per (user, candidate). Read the next-token logprobs for "Yes"/"No" (vLLM `logprobs=20`, `max_tokens=1`). Score all 50 candidates of a pool in one batch.
- [ ] **Position controls:** 3 shuffled candidate orders per pool (fixed seeds), with Borda aggregation of the three rankings. Pointwise needs no shuffles.
- [ ] **Cache:** key = sha256(model, quantization, prompt, decoding params) → raw output in `outputs/cache/*.jsonl`. Resumable jobs.
- [ ] SLURM scripts in `scripts/`, one job per (model, method, condition shard). Log tokens/s to estimate the full grid.
- [ ] **Throughput check first:** time Direct-Rec and Base-Reason on 50 dev pools per model, then fix N and the reasoning subsample size (report §4, compute budget).

### Phase 6: Non-LLM baselines and evaluation
- [ ] Heuristics on the same pools:
  - random (0.02);
  - random-among-newest (0.10);
  - newest-first;
  - SASRec order (warm items only, so its Recall@1 on cold targets is 0 by construction; this is still worth stating);
  - popularity before `T`;
  - content cosine;
  - **cold-filter + content**.
- [ ] Metrics: Recall@1, NDCG@10, Cold@1 and parse-failure rate, per model × method × negative setting.
- [ ] Paired bootstrap CIs (10k resamples) and McNemar for method-vs-method comparisons.
- [ ] **Baseline results table:** absolute Recall@1 for every method next to the date-only and cold-filter+content rows. This is deliverable 1 of report Table 2.

### Phase 7: EB-NeRD port
- [ ] Load EB-NeRD train+validation. `t0` = `published_time`. `x_c` = title, subtitle, category, subcategory.
- [ ] Choose a global cutoff `T` inside 27 Apr – 8 Jun 2023 and a short window. News items live for hours to days, so `w` ≈ 1–3 days, which is close to Li et al.'s one-day test set.
- [ ] Warm ranker: SASRec on clicks before `T` (same code).
- [ ] Danish text: check parse rate and dev Recall@1. If the small models fail, translate titles and subtitles once with a fixed MT model and cache the result (report, Limitations).
- [ ] For A1 traps later: in-view but not-clicked articles from the impression logs.

### Phase 8 (stretch): QLoRA SFT
- [ ] Following Li et al. §3.3: collect successful reasoning traces (target ranked first) from Base-Reason and Soft Self-Consistency on **pre-`T`** training instances, then run QLoRA 4-bit for 1 epoch on Llama-2-13B-Chat.
- [ ] On an 11 GB card this needs a short max sequence length (≈ 3k), gradient checkpointing and batch size 1 with accumulation. Check feasibility with a 100-step run first.
- [ ] Question to answer: does SFT on temporally split data *increase* Cold@1 and ∆recency? (report §3.4)
- [ ] GRPO is out of scope on this hardware.

---

## 4. Interfaces

### 4.1 Config (`configs/amazon_vg.yaml`)
```yaml
dataset: amazon_vg
cutoff_T: 2023-03-01        # set in Phase 0
window_days: 30             # set in Phase 0
min_history: 5
history_len: {base: 20, fast: 40}
pool: {warm: 40, cold: 10}
cold_negatives: [random, hard]
n_test: 1000
n_dev: 200
shuffles: 3
seed: 42
```

### 4.2 Run ID
`{dataset}/{condition}/{neg_setting}/{model}/{method}/shuffle{s}`

### 4.3 Pool record (JSONL, one per instance × negative setting)
```json
{
  "instance_id": "vg_000123",
  "user_id": "...",
  "condition": "baseline",
  "neg_setting": "random",
  "cutoff_T": "2023-03-01",
  "history": [{"item_id": "...", "title": "...", "date": "2021-11-02", "category": "...", "ts": 1635811200}],
  "candidates": [{"item_id": "...", "title": "...", "date": "2023-04-10", "category": "...",
                  "brand": "...", "platform": "...", "is_cold": true, "is_gold": false,
                  "date_source": "date_first_available", "sim_to_history": 0.41}],
  "gold_item_id": "..."
}
```
The displayed `date` is kept separate from the true `t0` (stored with the item), so that A2 (swap), A3 (hide) and A4 (shuffle) only rewrite `date`.

---

## 5. Context budget (4k cap)

| Part | Tokens (approx., English) |
|---|---|
| Instructions | 250 |
| Worked examples (2 × 5-candidate) | 600 |
| History, 20 items × ~25 | 500 |
| 50 candidates × ~25 | 1,250 |
| **Prompt total** | **~2,600** |
| Room for reasoning output | ~1,400 |

- Fast-Reason (40-item history, 1 example) comes to ~2,800.
- Danish (EB-NeRD) roughly doubles the history and candidate tokens. In that case, cut the history, drop the worked examples, or translate.
- In Phase 4, log the real token counts per prompt and flag any prompt over 3,000 tokens.

---

## 6. Sanity checks before the full run
1. Every pool has exactly 50 unique candidates: one gold, 10 cold, 40 warm, none from `H_u`.
2. In the baseline pools, every cold candidate's `t0` is after `T`, and every history item is before `T`.
3. Random-among-newest gives ≈ 0.10 and random gives ≈ 0.02 empirically.
4. Parse-failure rate is < 10% for each model × method on the dev set. If it isn't, fix the prompt before running the test set.
5. Identical prompts give identical outputs across reruns (greedy decoding plus the cache).
6. SASRec clearly beats popularity on warm next-item prediction.

## 7. Report updates triggered by this plan
- Eq. 1 / §3.1 must change to the **global cutoff** definition (decided above).
- Specify the test window `w`, cutoff `T`, history length `L`, k = 3 paths for Soft Self-Consistency, and the top-10 output format.
- Name SASRec as the warm-start ranker.
- State the quantization used (4-bit AWQ/GPTQ for 7B/13B) as a limitation.

## 8. Mapping to report threads (Table 2)
- **Thread 1 (Baseline):** Phases 0–6 on Amazon, then Phase 7.
- **Thread 2 (Variants):** starts from the Phase 3 pool files. The old-target instances are already saved in Phase 1.
- **Thread 3 (Decomposition):** reuses Phase 6 `eval/`.
- **Thread 4 (Scale):** the Phase 5 model table.
