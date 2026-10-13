# Plan: Reproducing the Netflix Cold-Start Reranking Baseline

Baseline paper: Li et al., *LLM Reasoning for Cold-Start Item Recommendation*, WWW '26 (`readings/li2026_llm_reasoning_cold_start_netflix.pdf`).

Goal: rebuild their **evaluation protocol and prompting methods** on public data, so that the adversarial variants (A1–A6, `report/report.tex` §3.5) can later be applied to the same pools. We reproduce the protocol, not their numbers: their data, production ranker and prompts are not public.

Every design choice and its reason is logged in `design.md` (§11 covers the implementation). Code is in `code/` (see `code/README.md` for commands).

## Status (Amazon Video Games)

| Phase | Status |
|---|---|
| 0. Environment and data audit | ✅ Done. `T` and `w` chosen from the audit |
| 1. Global-cutoff split | ✅ Done. 950 cold instances (dev 150 / test 800), 4,328 old-target instances |
| 2. Warm-start ranker (SASRec) | ✅ Done. Validation NDCG@10 0.064; 3.7× popularity on old targets |
| 3. Pools | ✅ Done. 1,900 pools (2 negative settings), all invariants pass |
| 4. Prompts (5 methods + pointwise) | ✅ Implemented and tested. ⏳ To be tuned on dev with the real models |
| 5. Inference engine | ✅ Implemented (vLLM + HF fallback, cache, Borda). ⏳ Not yet run on the GPU |
| 6. Reference heuristics + evaluation | ✅ Done. Heuristic results on test (report Table 4) |
| — LLM runs on ADA | ⏳ Next: vLLM smoke test on the 2080 Ti, then dev runs |
| Adversarial variants + decomposition | ⏳ Next: see [`plan_adversarial.md`](plan_adversarial.md) |
| 7. EB-NeRD port | ⏳ Not started |
| 8. QLoRA SFT (stretch) | ⏳ Not started |

**Decisions made**
- **Cold-start definition:** a single **global cutoff**, as in Li et al. (*not* per-user).
- **First domain:** **Amazon Video Games**, then EB-NeRD.
- **Split:** `T` = **2022-09-01**, `w` = **180 days**, rating ≥ 3, ≥ 5 positives before `T` (after the audit; a 90-day window gives at most 465 instances).
- **Hardware:** 1× **RTX 2080 Ti** (11 GB, sm_75): fp16, 4-bit weights for 7B/13B.

---

## 1. What Li et al. do, and what we use instead

| Component | Li et al. (Netflix) | Ours (Amazon VG) |
|---|---|---|
| Cold-start item | Launched after the latest interaction in **all** user histories, and after the LLM's training cutoff (§2.1) | `t0 > T` with global `T` = 2022-09-01 (also Llama-2's documented data cutoff) |
| Launch time | Netflix launch date | `t0 = min(Date First Available, first review)` (Amazon listing date) |
| Cold items in training data | Never appear in histories; used only as targets (§3.1.1) | Same, guaranteed by construction (asserted in the code) |
| Test set | Interactions from **one day** whose target is cold (§3.1.2) | First cold positive per user in `[T, T + 180 days)`: 950 instances |
| Candidate pool | Top-40 from the production ranker + 10 sampled cold titles including the target (§2.1) | Top-40 from **SASRec** (our own implementation, Dong et al.'s hyperparameters) + gold + 9 cold negatives, **random** and **hard** |
| Item attributes in prompt | Title, launch date, genre (§2.2.1) | Title (cut to 60 chars), launch date, category path; plus brand for Structural Reasoning |
| Base LLM | Qwen-2.5-32B-Instruct | Phi-2, Llama-2-7B/13B-Chat, Mistral-7B-Instruct, Qwen-7B-Chat (all 2023) |
| Methods | Direct-Rec, Base-Reason (2-shot), Fast-Reason (1-shot, 2× history), Structural Reasoning, Soft Self-Consistency | All five + a Structural variant without the recency instruction + the **pointwise yes/no** control |
| Fine-tuning | SFT, GRPO, SFT+GRPO | None in the main runs; QLoRA SFT is a stretch goal; no GRPO |
| Metric | Recall@1 on Discovery (primary) and AnyPlay | Recall@1 (plays the role of Discovery), NDCG@10, Cold@1, parse failures |
| Reported as | Relative % gains only | **Absolute** numbers next to the date-only baselines |

---

## 2. Repository layout (as built)

```
IRE_Project/
├── plan.md, design.md
├── report/                       # LaTeX report
├── readings/                     # papers + README
├── code/
│   ├── README.md                 # pipeline commands
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── configs/
│   │   ├── amazon_vg.yaml        # T, w, filters, SASRec, pools, prompts
│   │   └── models.yaml           # pinned checkpoints, quantization, chat family
│   ├── coldrank/
│   │   ├── common.py             # paths, config, JSONL, deterministic hashing
│   │   ├── data/amazon_vg.py     # raw JSONL -> parquet
│   │   ├── data/audit.py         # Phase 0 audit
│   │   ├── data/split.py         # global-cutoff split
│   │   ├── warm/sasrec.py        # warm-start ranker
│   │   ├── pools/build.py        # 40 + 10 pools, invariants
│   │   ├── prompts/render.py     # 5 methods + pointwise, parser
│   │   ├── prompts/examples.py   # hand-written worked examples
│   │   ├── llm/chat.py           # per-family chat formats
│   │   ├── llm/engine.py         # vLLM / HF engines + cache
│   │   ├── llm/run.py            # run one model x method over a pool file
│   │   ├── baselines/heuristics.py
│   │   └── eval/{metrics,stats,report}.py
│   ├── scripts/run_llm.sbatch    # SLURM template (fill in account/partition)
│   └── tests/                    # 39 tests
├── data/                         # git-ignored: raw + processed datasets
└── outputs/                      # git-ignored: audit, predictions, reports, cache, logs
```

---

## 3. Phases

### Phase 0: Environment and data audit ✅
- [x] Local environment (CPU): `.venv` on the system Python; set `USE_TF=0` (done in the package) and `HF_HUB_OFFLINE=1` once models are cached.
- [ ] Cluster environment (Python 3.10, CUDA, `vllm`, `autoawq`, `auto-gptq`, `optimum`, `tiktoken`, …): see `code/requirements.txt`.
- [x] Downloaded Amazon Reviews 2023 Video Games from the Hugging Face mirror (137,269 items, 4,624,615 reviews).
- [x] ~~Clone Dong et al.'s code~~ → own implementation with their settings (their pipeline uses a 70/10/20 split and hand-made CSVs).
- [x] Launch-date audit: DFA parsed for 93.6% of items; DFA after the first review for 8.2% → `t0 = min(DFA, first review)`.
- [x] Instance counts per (`T`, `w`) → `T` = 2022-09-01, `w` = 180 days (`outputs/audit/amazon_vg_audit.md`).
- [x] Positives: rating ≥ 3.0 (Dong et al.).

### Phase 1: Global-cutoff split ✅
- [x] History = positives before `T`; cold = `t0 > T`; warm = `t0 ≤ T` with a pre-`T` positive.
- [x] Target = the user's first cold positive in `[T, T+w)`; users with ≥ 5 history items.
- [x] No cold item in any history (asserted).
- [x] Dev 150 / test 800 (all 950 available instances), user-disjoint, seeded.
- [x] Old-target instances saved for A1 (4,328).

### Phase 2: Warm-start ranker ✅
- [x] SASRec on pre-`T` positives with a 5-core filter (598,719 interactions, 69,809 users, 19,291 items); leave-one-out validation, early stopping.
- [x] Sanity check: Recall@10 0.033 vs 0.009 for popularity on old targets.
- [x] Top-100 warm items per instance excluding `H_u` (popularity fallback for 0.6% of users).

### Phase 3: Pools ✅
- [x] 40 warm + gold + 9 cold negatives; neutral stored order.
- [x] Cold negatives launched in `(T, target_ts]`; the user's other window items excluded; near-duplicates of the gold (cosine ≥ 0.9) removed.
- [x] Random and hard (most similar to the user profile) negatives.
- [x] Manipulation stats in `pools/pool_stats.json`.

### Phase 4: Prompts ✅ implemented / ⏳ tune on dev
- [x] Five methods + `structural_norecency` + pointwise, with compact display and history trimming to fit 4,096 tokens.
- [x] Hand-written worked examples in which the gold is not the newest item and dates are never mentioned.
- [x] Top-10 output after `FINAL:`; parser with a fallback; parse mode logged.
- [ ] Read dev outputs per model × method; adjust prompts **on dev only**; then freeze.

### Phase 5: Inference ✅ implemented / ⏳ run on GPU
- [x] Pinned checkpoints (`models.yaml`): TheBloke AWQ for Llama-2 7B/13B and Mistral, official Qwen-7B-Chat-Int4, Phi-2 fp16.
- [x] vLLM engine (fp16, `max_model_len` 4096, prefix caching) and HF fallback; exact pointwise log p(Yes) − log p(No).
- [x] 3 shuffles + Borda; hash tie-break; resumable on-disk cache.
- [x] Verified end to end on CPU with SmolLM2-135M on real pools (prompts ≈ 2,490 tokens, no overflow, 100% parsed).
- [ ] **ADA smoke test:** `--model llama2_7b --method direct --split dev --limit 2` (does vLLM run on sm_75?).
- [ ] Throughput check, then decide the reasoning-method subsample size.

### Phase 6: Reference heuristics and evaluation ✅
- [x] Random, random-among-newest, newest-first, SASRec, popularity, content, cold-filter + content.
- [x] Report script with bootstrap CIs; paired tests in `eval/stats.py`.
- [x] Test results (Recall@1, random / hard negatives): random 0.029 / 0.024; random-among-newest 0.111 / 0.102; newest-first 0.095 / 0.090; content 0.041 / 0.009; **cold-filter + content 0.324** / 0.009; SASRec and popularity 0 / 0.
- [ ] LLM rows for the baseline table.

### Phase 7: EB-NeRD port ⏳
- [ ] Load EB-NeRD train + validation; `t0` = `published_time`; text = title, subtitle, category, subcategory.
- [ ] Global `T` inside 27 Apr – 8 Jun 2023 with `w` ≈ 1–3 days (close to Li et al.'s one-day test set). Run the audit first.
- [ ] Warm ranker: SASRec on clicks before `T` (same code).
- [ ] Danish: check parse rate and dev Recall@1; translate titles and subtitles once if the models fail.
- [ ] A1 traps: in-view but not-clicked articles.

### Phase 8 (stretch): QLoRA SFT ⏳
- [ ] Successful traces from Base-Reason and Soft Self-Consistency on pre-`T` instances → QLoRA 4-bit, 1 epoch, Llama-2-13B-Chat (short max length, gradient checkpointing). Check feasibility with 100 steps.
- [ ] Question: does SFT on temporally split data increase Cold@1 and ∆recency? No GRPO.

---

## 4. Interfaces
- **Config:** `code/configs/amazon_vg.yaml` (split, SASRec, pools, prompts) and `code/configs/models.yaml`.
- **Run path:** `outputs/<dataset>/<condition>/<neg>/<model>/<method>/<split>/{raw,preds}.jsonl`, plus `run_summary.json`.
- **Pool record** (`data/processed/amazon_vg/split_T2022-09-01_w180/pools/<condition>_<neg>_<split>.jsonl`): `pool_id`, `instance_id`, `split`, `user_id`, `condition`, `neg_setting`, `cutoff_T`, `target_ts`, `gold_item_id`, `history[]`, `candidates[]`.
  - Each candidate has `item_id`, `title`, `date` (**displayed**), `t0` (**true**), `t0_source`, `category`, `brand`, `is_cold`, `is_gold`, `sim_to_history`, `pop_preT` and `warm_rank`.
  - The date variants (A2–A4) only rewrite `date`.

## 5. Context budget (4,096-token cap, measured with the Llama-2 tokenizer)

| Part | Tokens |
|---|---|
| One candidate line (compact) | ≈ 39 (was ≈ 61 with full titles and labels) |
| 50 candidates | ≈ 1,950 |
| Typical Direct-Rec prompt | ≈ 2,490 (no overflow on dev) |
| Room for generation | ≥ 60 (direct), 300–600 (reasoning); otherwise history is trimmed (minimum 5 items) |

## 6. Sanity checks
1. ✅ Every pool has 50 unique candidates: 1 gold, 10 cold, 40 warm, none from `H_u` (asserted for all 1,900 pools).
2. ✅ Cold `t0 > T`, warm `t0 ≤ T`, history before `T`, every candidate available at the target time.
3. ✅ Random ≈ 0.02–0.03 and random-among-newest ≈ 0.10–0.11 empirically.
4. ⏳ Parse-failure rate < 10% per model × method on dev (needs GPU runs).
5. ✅ Determinism: greedy decoding, seeded shuffles, hash tie-breaks, cache.
6. ✅ SASRec clearly beats popularity on warm next items.

## 7. Report updates ✅
- [x] Eq. 1 / §3.1 now uses the global cutoff; `T`, `w`, history, SASRec, launch-date rule, compaction, quantization and the new limitations (listing date, 180-day window, 4-bit) are all in the report.
- [x] Preliminary heuristic results added (report Table 4).

## 8. Mapping to report threads (Table 2)
- **Thread 1 (Baseline):** Phases 0–6 on Amazon (LLM runs pending), then Phase 7.
- **Thread 2 (Variants):** starts from the Phase 3 pool files; old-target instances are already saved.
- **Thread 3 (Decomposition):** reuses `coldrank/eval/`.
- **Thread 4 (Scale):** the model list in `models.yaml`.
