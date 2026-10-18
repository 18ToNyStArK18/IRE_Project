# Plan: EB-NeRD port (Phase 7)

**Goal:** run on EB-NeRD news exactly what was run on Amazon Video Games: the Li et al. cold-start split, SASRec warm ranker, 50-candidate pools, adversarial variants A1–A6, the K = 15 pools, the non-LLM heuristics, and Llama-2-7B (all 7 methods on the K = 15 baseline; pointwise + Direct-Rec on every K = 15 variant), on dev and then test. Then the decomposition and a write-up.

EB-NeRD is the **volatile** domain: recency is a real signal in news. Amazon is the **static** one. The paper compares the two.

All decisions are logged in `design.md` (§ EB-NeRD).

## Status

| Step | Status |
|---|---|
| E0. Download EB-NeRD small (84 MB) | ✅ `data/raw/ebnerd/small/` |
| E0. Audit: cutoff and window | ✅ T = 2023-05-25, w = 1 day: 8,676 cold / 5,107 old instances |
| E1. Parser + Danish → English translation | ✅ 20,738 articles, 38,822 strings translated |
| E2. Split (+ in-view-not-clicked lists for A1 traps) | ✅ cold 150 / 800, old 150 / 1,000 (≥ 3 traps) |
| E3. SASRec on pre-T clicks | ✅ val NDCG@10 0.259; old-target Recall@10 0.108 |
| E4. Pools (K = 50), variants, K = 15 resize, heuristics | ✅ all invariants pass; A6 131 / 650 (see below) |
| E5. News prompts (wording, worked examples), date + time display | ✅ fit 4,096 tokens, no overflow |
| E6. ADA: one job, stage `k15-all` (84 tasks: dev + test) | ✅ job 25738 (11 h 26 min; job 25735 failed on a faulty GPU node, gnode069) |
| E7. Report, decomposition, write-up | ✅ `analysis_2026-10-18_ebnerd.md`; logistic regression still to do |

## Decisions (team, 2026-10-17)

| Question | Decision | Why |
|---|---|---|
| Dataset size | **EB-NeRD small** | Already gives 9× the instances needed; large is 3.2 GB |
| Cutoff / window | **T = 2023-05-25 00:00, w = 1 day** | Li et al. use a single test day; the plan said 1–3 days |
| Displayed date | **Date and time** (`2023-05-25 14:32`) | Median click is 1.4 h after publication; with w = 1 day every cold article has the same date, so only the time separates them |
| Translation | **Helsinki-NLP/opus-mt-da-en**, pinned revision, greedy, translated once and cached | Dedicated da→en model; deterministic; every condition sees identical English text |
| LLM scope | **Current Amazon protocol**: Llama-2-7B, K = 15 | Same as jobs 23791, 23838, 23920 |

## E0. Data and audit
- **Files:** `articles.parquet` (20,738 articles), and for `train` (behaviors 18–25 May) and `validation` (25 May–1 Jun): `history.parquet` (each user's clicks in the 21 days before the behaviors) and `behaviors.parquet` (impressions: in-view and clicked lists). Test labels are not public, so only train + validation are used.
- **Clicks:** histories + behaviors of both splits, de-duplicated: **3.05M clicks, 18,827 users, 12,042 clicked articles**, 27 Apr – 1 Jun 2023, a continuous stream across T.
- **Audit (min history 5):**

| T | w | Cold instances | Distinct golds | Old instances | Articles published in window |
|---|---|---|---|---|---|
| **25 May** | **1** | **8,676** | **109** | **5,107** | **123** |
| 25 May | 2 | 11,126 | 222 | 5,641 | 252 |
| 25 May | 3 | 12,323 | 321 | 5,935 | 364 |
| 26–31 May | 1 | 6,625–8,134 | 95–131 | 3,617–4,188 | 108–139 |

- **Click age:** median 1.4 h after publication (10%: 0.3 h, 90%: 9.6 h). 525 clicks predate the listed publication time, so launch time = min(published, first click), the Amazon rule.

## E1. Parser and translation (`code/coldrank/data/ebnerd.py`, `translate.py`)
- Writes the **same schema as Amazon** (`items`, `interactions`), so split, SASRec, pools and variants run unchanged; plus `impressions.parquet` for the traps.
- **Positive:** every click (rating = 1.0).
- **Times floored to the minute**, the displayed resolution.
- **Text:** title, subtitle and topics translated with opus-mt-da-en (greedy; beam search was ~30× slower on CPU). Categories (`category_str`, 26 values) use a fixed English table, because several are site-specific ("auto" = automated local-business news, "play" = the video section, "side9" = page-nine glamour). Danish originals are kept (`*_da`).
- **No brand in news:** Structural Reasoning's extra attribute is the article's first 3 topics.

## E2. Split
- Generic `coldrank.data.split` with `configs/ebnerd.yaml`: T = 2023-05-25, w = 1, min history 5, rating ≥ 1 (all clicks).
- **Sizes match Amazon:** cold 150 dev / 800 test; old 150 / 1,000.
- **A1 traps** (design.md §4): articles **shown to the user (in-view) in [T, target time] but not clicked** in the window, that are new (published in (T, target]). Stored per old-target instance as `inview_not_clicked`. No similarity filter: being skipped is the irrelevance evidence.

## E3. SASRec
- Same code and hyperparameters as Amazon (`kcore` 5, max_seq_len 50, …), trained on pre-T clicks. Report validation NDCG@10.

## E4. Pools, variants, heuristics
- **K = 50 pools** (40 SASRec warm + gold + 9 cold negatives; random and hard), then the variants and `resize.py` → K = 15 (12 + gold + 2; A1: gold + 11 + 3 traps).
- **Cold negatives:** articles published in (T, target time]. Early-morning targets may have too few; dropped instances are counted, and the split oversamples if needed so dev/test reach 150/800.
- **Embeddings** (hard negatives, profiles, near-duplicate filter, content heuristic): all-MiniLM-L6-v2 on the **English** title + subtitle + category.
- **Heuristics:** the same set (random, random-among-newest, newest-first, SASRec, popularity, content, newest + content); dates parsed with minute resolution.
- **Manipulation checks:** the same invariants as Amazon, plus the trap age gap and similarity.

## E5. News prompts
- `prompt.domain: news` in the config selects news wording ("news site", "reading history", "read next", "published"); Amazon prompts stay **byte-identical** (cached results are keyed on prompt text).
- **Displayed fields:** title | published (date + time) | category, plus the subtitle (cut at 100 characters). Structural adds topics.
- **Worked examples:** new, hand-written English news examples, following the Amazon rules: the correct item is not the newest, and the reasoning never mentions dates.

## E6. ADA runs (Llama-2-7B, K = 15)
- `CONFIG=ebnerd` selects the dataset in `llm_ada.sbatch` / `make_tasks.py`; `ada_sync.sh` syncs both datasets' pools.
- **Dev:** `k15-baseline` dev half (7 methods × 2 negative settings) + `k15-variants` (28 tasks). Check parse rates and prompt lengths.
- **Test:** `k15-test` (42 tasks).

## E7. Analysis
- `eval.report` and `eval.decompose --prefix k15_` on dev and test; write-up `analysis_<date>_ebnerd.md`; compare with Amazon (static vs volatile).
- Later: the domain × condition logistic regression (plan_adversarial.md §5), now possible.

## Local results (2026-10-17)
- **Pools:** K = 50 baseline 950 instances, none dropped. Variants A2–A4 1,900 pools each, A1 family 1,150 per condition. **A6 is built at K = 15 directly** (14 negatives): only 68 / 265 dev / test targets have 49 new articles published before the click, 131 / 650 have 14.
- **K = 15 heuristics (test, Recall@1, random / hard negatives):** newest-first **0.560 / 0.515** (Amazon 0.346 / 0.323), random-among-newest 0.312 / 0.329, newest + content 0.443 / 0.107, content 0.060 / 0.037, random 0.071 / 0.066. Dates hidden (A3): every date heuristic at chance. A1: newest-first Trap@1 = 1, A1-ctrl newest-first 0.603.
- **Prompt lengths (Llama-2 tokens, K = 15):** Direct-Rec ~2,100, SSC ~2,150, Structural ~3,400, Fast ~3,550 (32 of 40 history items kept), Base ~3,620 (18 of 20 kept), pointwise ~1,260. No overflow.
- **ADA estimate:** `make_tasks.py` says 53.6 GPU-h, ≈ 13.4 h on 4 GPUs.
