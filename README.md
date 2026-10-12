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
| [`plan.md`](plan.md) | Implementation plan, with done / pending status per phase |
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
| Adversarial variants A1–A6, EB-NeRD, QLoRA SFT | ⏳ Later |

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

# 3. LLM runs (GPU)
python -m coldrank.llm.run --model llama2_7b --method direct --split dev --neg random

# Tests
python -m pytest -q -m "not slow"
```

Things to know:
- Configs live in `code/configs/` (`amazon_vg.yaml` for the protocol, `models.yaml` for the pinned model checkpoints).
- Set `HF_HUB_OFFLINE=1` once models are downloaded. Model loading can otherwise hang on a stalled Hugging Face connection.
- On the dev machine, `USE_TF=0` is already set inside the package; it avoids a TensorFlow/Keras clash with `transformers`.

## Next steps

1. **vLLM smoke test on ADA:** `--model llama2_7b --method direct --split dev --limit 2`.
2. **Dev runs:** every model × method on the 150 dev instances. Fix prompts on dev only, then freeze them.
3. **Test runs:** produce the baseline table with LLM rows.
4. **Adversarial variants (Thread 2):** build them from the existing pool files.
