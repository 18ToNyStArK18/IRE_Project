# Relevance or Recency? Diagnosing the Temporal Shortcut in LLM-Based Cold-Start Reranking

IRE course project, IIIT Hyderabad: Vedavyas Bysani, Ayush Daga, Sanjith Ganapathi.

## The question

LLM rerankers are reported to do well on **cold-start (new) items**. But cold-start test sets are built with a temporal split, so the correct item is always one of the *newest* candidates. A reranker that simply prefers new items scores well without reading any content.

**When an LLM reranker succeeds on new items, how much of that is relevance, and how much is a preference for recency?**

We answer this in three steps:
1. **Adapt** the Netflix cold-start reranking protocol of Li et al. (WWW '26) to public data: Amazon Video Games (a static catalogue, where recency should matter little) and EB-NeRD news (volatile, where recency is a real signal).
2. **Fine-tune** 7B LLMs on that protocol, since no zero-shot model was competent enough to diagnose.
3. **Probe** each model with adversarial variants that pull recency and relevance apart: dates swapped, hidden or shuffled; all-new pools; and an old correct item next to new-looking "trap" items.

## Headline findings

1. **No zero-shot model works in our setting.** Llama-2-7B, Qwen2.5-7B and Qwen2.5-32B (Li et al.'s own base model) sit at or below chance across all seven prompting methods. Before fine-tuning they have neither skill nor a strong date rule.
2. **Fine-tuned on the cold-start protocol, every model learns the date shortcut.** This holds for both base models (Llama-2-7B, Qwen2.5-7B) on both datasets. The models look strong on the benchmark (Llama-2 on Amazon: 0.445 / 0.546 Recall@1, about 7× chance and above newest-first). But swapping the correct item's date takes them to about 0. Hiding or shuffling dates takes them to about chance. New-looking irrelevant items take the top spot 56–89% of the time on A1 (chance 20%).
3. **The benchmark over-credits these models.** The Baseline score mostly measures "pick the newest". On news, fine-tuned models do not even beat the newest-first rule. On Amazon they beat it, but lose 43–47% of their accuracy when all candidates are new (A6).
4. **Production-style training data does not remove the shortcut.** We trained on impression negatives (shown-but-not-clicked articles) for both models on EB-NeRD. Traps still win about 90% of top spots, and accuracy on hard negatives drops.
5. **Neither does hiding dates in training.** A dates-hidden model still reacts to dates it never saw in training: traps still win 50% on A1. It also no longer beats newest-first. Hiding dates does not hide age.
6. **Mixing old correct answers into training (the "50/50 fix") is the fair test.** Traps fall from 56% to 28% on A1 (chance 20%). But swapping dates still drops the model to about chance. Its Baseline score falls below newest-first, which shows how much of the original score the shortcut supplied.
7. **Training the way Li et al. do (old correct items only, no new items at all) does not remove the date reliance either.** The cue comes from user behaviour, not from our data construction. Among old candidates, the clicked or bought item is the single newest one in about half of training pools (Amazon 51%, EB-NeRD 56%; chance 8%).
   - **Amazon:** traps take **95%** of top spots (94.7% ± 1.0% over three seeds), the strongest trap failure of any model. Qwen2.5-7B trained the same way repeats it (97%), so it is not specific to one base model. The benchmark scores this model *low* (below newest-first), which matches Li et al.'s weak cold-start numbers. Only the diagnostics show why.
   - **EB-NeRD (three seeds, test):** swapping or hiding dates still collapses it. But it is *not* drawn to the traps (15.6% ± 3.3%, below chance on every seed; with more traps, well below chance), so the shortcut takes a different form on news: among all-old candidates it picks the newest 64% of the time, yet it ranks items dated after the user's history lower. It learned "newest among old items", not "newest overall". Qwen2.5-7B trained the same way repeats this almost cell for cell (traps 19%), so the dataset, not the base model, decides the form of the shortcut.
8. **Results are stable across seeds.** The date-trained Llama-2 was trained with three seeds on both datasets. Every drop replicates on every seed. Amazon varies little (Baseline SD ≈ 0.015). EB-NeRD varies more, so we report it as mean ± spread.

**What we claim:** recency is a real and useful signal. Users do prefer new items, and the model is right to use it. The problem is that fine-tuned rerankers use it as a **shortcut** that overrides relevance, and the standard cold-start benchmark cannot tell the two apart. **What we do not claim:** that this is specific to LLMs, or that our training data replicates Netflix's.

The full numbers, with every table, McNemar test and caveat, are in [`results.md`](results.md). Every design decision and its reason is in [`design.md`](design.md).

## Experimental setup

| | Amazon Video Games | EB-NeRD (news) |
|---|---|---|
| Test cutoff and window | 2022-09-01, 180 days | 2023-05-25, 1 day |
| Users (dev / test) | 150 / 800 cold-target; 144 / 967 old-target | 150 / 800 cold-target; 150 / 1,000 old-target |
| Warm ranker (stand-in for Netflix's) | SASRec | SASRec |
| Fine-tuning data | 4,183 pools from 6 earlier cutoffs | 5,246 pools from 7 earlier cutoffs |

- **Pools:** K = 15 candidates: the correct item, 12 warm (old) items from SASRec, and 2 new negatives, either random or hard (similar to the user). Li et al. use 50; at that size our small models only copied the prompt order, so we kept their 4:1 ratio at 15 (logged in design.md).
- **Scoring:** pointwise Yes/No (TALLRec style), the only method above chance for any zero-shot model and free of parse failures and list-order effects. Listwise methods (Direct-Rec, Base, Fast, SSC, Structural) were run zero-shot only.
- **Fine-tuning:** QLoRA (4-bit NF4, LoRA r16 / α32), 1 epoch, lr 2e-4, effective batch 32, on 4× RTX 2080 Ti (IIIT ADA cluster). Training prompts are byte-identical to the test prompts. Training users never appear in dev or test.
- **Metrics:** Recall@1 (primary, as in Li et al.), reported as random / hard negatives; chance is 1/15 = 0.067. Trap@1 is how often a new-looking wrong item is ranked first: chance 0.20 on A1, rising with n on A5. Significance is exact McNemar paired by instance (\* = p < 0.05).

### Adversarial variants

Each variant changes **one factor** relative to a reference pool. Shared candidates stay in the same prompt positions.

| Condition | What changes | Reference |
|---|---|---|
| Baseline | Li et al.'s protocol: correct item is new, true dates shown | — |
| A2 Swap | The correct item's date is exchanged with an old wrong item's | Baseline |
| A3 Hide | No dates on candidates or history | Baseline |
| A4 Shuffle | Dates permuted across the candidates | Baseline |
| A6 All-new | Every candidate is new, so dates cannot single out the correct item | Baseline users |
| A1-ctrl | Correct item is **old**; nothing looks new | — |
| A1 Traps | A1-ctrl plus 3 new, irrelevant "trap" items | A1-ctrl |
| A5 Flood (n = 1, 2, 5, 10) | A1 plus *n* old wrong items relabelled with new dates | A1-ctrl |

A quick read: a model using content should barely move on A2–A4 and should ignore traps. A model using the date rule collapses on A2–A4 and picks traps.

### Training arms

| Arm | Training data | Models and datasets | Seeds |
|---|---|---|---|
| **Date-trained** (main) | The cold-start protocol itself: correct item always new, dates shown | Llama-2 and Qwen2.5-7B × Amazon and EB-NeRD | Llama-2: 3 seeds on both datasets |
| Impression negatives | Shown-but-not-clicked articles as negatives (production style) | Llama-2 and Qwen2.5-7B × EB-NeRD | 1 |
| Dates-hidden control | Date-trained data with every date removed | Llama-2 × Amazon | 1 |
| 50/50 fix | Half the pools have an old correct item | Llama-2 × Amazon | 1 |
| **Li-style** (main) | Old correct items only, no new item anywhere (Li et al.'s setup) | Llama-2 and Qwen2.5-7B × Amazon and EB-NeRD | Llama-2: 3 on both; Qwen: 1 |

## Results in brief (test set unless noted)

**Zero-shot.** Llama-2-7B scores 0.083 / 0.072 on Amazon and 0.055 / 0.054 on EB-NeRD. Qwen2.5-7B is below chance and avoids new items on content (A1 Trap@1 0.004 on Amazon). Qwen2.5-32B is at or near chance on all seven methods. One checkpoint, Llama-2 4-bit AWQ, shows a small date effect that disappears with the original fp16 weights. So the bias we report appears mainly *after* fine-tuning.

**Date-trained models.** In every cell, A2 is about 0, A3 and A4 are at or below chance, and traps win far above the 20% chance level.

| | Baseline | Newest-first | A2 Swap | A6 All-new | A1 Trap@1 |
|---|---|---|---|---|---|
| Llama-2, Amazon | 0.445 / 0.546 | 0.346 / 0.323 | 0.005 / 0.006 | 0.253 / 0.292 | 0.56 |
| Qwen2.5-7B, Amazon | 0.432 / 0.532 | 0.321 / 0.316 | 0.015 / 0.014 | 0.152 / 0.297 | 0.60 |
| Llama-2, EB-NeRD | 0.394 / 0.487 | 0.560 / 0.515 | 0.001 / 0.000 | 0.097 / 0.117 | 0.83 |
| Qwen2.5-7B, EB-NeRD | 0.492 / 0.539 | 0.560 / 0.515 | 0.000 / 0.001 | 0.214 / 0.222 | 0.86 |

**Other training arms** (Llama-2 on Amazon unless noted):

| Arm | Baseline | A2 Swap | A1 Trap@1 | Reading |
|---|---|---|---|---|
| Date-trained | 0.445 / 0.546 | 0.005 / 0.006 | 0.56 | Over-credited by the benchmark |
| Dates-hidden | 0.299 / 0.345 | 0.084 / 0.100 | 0.50 | Still follows dates it never saw in training |
| 50/50 fix | 0.296 / 0.334 | 0.068 / 0.091 | 0.28 | Traps mostly neutralised; dates still dominate |
| Li-style (3 seeds, mean) | 0.160 / 0.298 | 0.004 / 0.012 | 0.95 | Scored low by the benchmark, strongest trap failure |
| Li-style, Qwen2.5-7B | 0.149 / 0.383 | 0.000 / 0.006 | 0.97 | Same as Llama-2 Li-style |
| Impression, EB-NeRD (Llama-2 / Qwen) | 0.486 / 0.269, 0.406 / 0.194 | about 0 | 0.90, 0.89 | Production data keeps the shortcut |
| Li-style, EB-NeRD (3 seeds, mean) | 0.281 / 0.266 | 0.022 / 0.027 | 0.16 | Date-reliant within old items, but avoids new-dated traps |
| Li-style, EB-NeRD, Qwen2.5-7B | 0.305 / 0.256 | 0.021 / 0.021 | 0.19 | Same as Llama-2 Li-style on news |

## Status

| Part | Status |
|---|---|
| Data, splits, SASRec, K = 15 pools and all adversarial variants (both datasets) | ✅ Done; invariants asserted on every pool |
| Zero-shot runs (Llama-2-7B, Qwen2.5-7B, Qwen2.5-32B; 7 methods) | ✅ Done |
| Date-trained fine-tunes (Llama-2, Qwen2.5-7B × both datasets; Llama-2 seeds 43, 44) | ✅ Done, dev and test |
| Impression-negative fine-tunes (both models, EB-NeRD) | ✅ Done |
| Dates-hidden control and 50/50 fix (Llama-2, Amazon) | ✅ Done |
| Li-style, Llama-2: three seeds on both datasets | ✅ Done |
| Li-style, Qwen2.5-7B, one seed, both datasets | ✅ Done, dev and test |
| Paper write-up | ⏳ In progress ([`progress/progress.pdf`](progress/progress.pdf) is the reference document) |

Fine-tuned adapters are uploaded privately to the ColdRank organisation on Hugging Face.

## Repository

| Path | What it is |
|---|---|
| [`results.md`](results.md) | Running log of every result table, with jobs, sample sizes, tests and caveats |
| [`design.md`](design.md) | Every design decision with its reason; §11 is the implementation log |
| [`progress/progress.pdf`](progress/progress.pdf) | Reference document for writing the paper: findings, evidence index and objections |
| [`report/`](report/) | Earlier LaTeX write-up |
| [`plan.md`](plan.md) | Original implementation plan |
| [`code/`](code/) | The `coldrank` pipeline; see [`code/README.md`](code/README.md) |
| [`readings/`](readings/) | Key papers: Li et al. (the baseline), Fang et al. (recency bias in LLM reranking), Dong et al. (cold-start diagnosis) |
| `data/`, `outputs/` | Datasets and run outputs. **Not in git**; rebuilt with the commands in `code/README.md` |

Inside `code/coldrank/`:

| Module | Role |
|---|---|
| `data/`, `warm/`, `pools/` | Parsing, audit, temporal split, SASRec, K = 15 candidate pools |
| `adversarial/` | Builds and checks A1–A6, A1-ctrl and A5 |
| `prompts/`, `llm/` | Prompt rendering (7 methods, letter codes) and inference (vLLM with an HF fallback) |
| `finetune/` | Training pools at earlier cutoffs (`trainpools.py`), impression data (`impressions.py`), 50/50 and Li-style pools (`oldmix.py`), SFT data (`sft_data.py`), QLoRA training (`train.py`), adapter merge (`merge.py`) |
| `baselines/`, `eval/` | Newest-first and other heuristics, metrics (Recall@1, Trap@1), McNemar tests, decomposition |

Configs are in `code/configs/`: `amazon_vg.yaml` and `ebnerd.yaml` for the protocol, `finetune*.yaml` for one file per training arm and seed, and `models.yaml` for pinned checkpoints and adapters.

## Getting started

```bash
# Environment (Python 3.10 + CUDA)
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
cd code

# Data and pools (CPU, except SASRec); --config ebnerd for news
python -m coldrank.data.split
python -m coldrank.warm.sasrec
python -m coldrank.pools.build
python -m coldrank.adversarial.build
python -m coldrank.baselines.heuristics --split test

# Fine-tuning and evaluation on ADA (SLURM)
sbatch --export=ALL,CONFIG=amazon_vg,FT_CONFIG=finetune scripts/ft_train.sbatch
sbatch --export=ALL,STAGE=ft-test,MODEL=llama2_7b_ft_amazon_vg,CONFIG=amazon_vg,TP=2 scripts/llm_ada.sbatch

# Tests
python -m pytest -q -m "not slow"
```

The full step list, with outputs, is in [`code/README.md`](code/README.md). Things to know:
- Set `HF_HUB_OFFLINE=1` once the models are downloaded; loading can otherwise hang on a stalled Hugging Face connection.
- On ADA, use `--constraint=2080ti` and exclude the nodes listed in design.md, which had hung I/O or a full disk.
