# coldrank: Netflix-style cold-start reranking baseline

Reproduces the evaluation protocol and prompting methods of Li et al. (2026) on
public data. The plan is in `../plan.md` and every design decision is in
`../design.md` (§11 covers the implementation).

All commands run from this directory (`code/`). Data goes to `../data/` and outputs
to `../outputs/` (both git-ignored). Set `COLDRANK_DATA` / `COLDRANK_OUT` to move them.

## Pipeline (Amazon Video Games)

| Step | Command | Where | Output |
|---|---|---|---|
| 0. Download | see below | any | `data/raw/amazon_vg/*.jsonl` |
| 1. Parse | `python -m coldrank.data.amazon_vg` | CPU | `data/processed/amazon_vg/{items,interactions}.parquet` |
| 2. Audit | `python -m coldrank.data.audit` | CPU | `outputs/audit/amazon_vg_audit.md` |
| 3. Split | `python -m coldrank.data.split` | CPU | `data/processed/amazon_vg/split_T…/{cold,old}_{dev,test}.jsonl` |
| 4. SASRec | `python -m coldrank.warm.sasrec` | CPU or GPU | `split_T…/sasrec_topk.jsonl`, `sasrec_report.json` |
| 5. Pools | `python -m coldrank.pools.build` | CPU | `split_T…/pools/baseline_{random,hard}_{dev,test}.jsonl` |
| 5b. Adversarial pools | `python -m coldrank.adversarial.build` | CPU | `split_T…/pools/<condition>_<neg>_<split>.jsonl`, `variant_stats.json` |
| 6. Heuristics | `python -m coldrank.baselines.heuristics --split dev` | CPU | `outputs/amazon_vg/baseline/<neg>/heuristic/<name>/<split>/preds.jsonl` |
| 7. LLM runs | `python -m coldrank.llm.run --model M --method X --split dev --neg random` | GPU (ADA) | `outputs/amazon_vg/baseline/<neg>/<M>/<X>/<split>/{raw,preds}.jsonl` |
| 8. Report | `python -m coldrank.eval.report --split dev` | CPU | `outputs/amazon_vg/baseline/report_dev/summary.md` |

Download (Hugging Face mirror of Amazon Reviews 2023):
```bash
mkdir -p ../data/raw/amazon_vg && cd ../data/raw/amazon_vg
base=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw
curl -L -C - -O $base/meta_categories/meta_Video_Games.jsonl
curl -L -C - -O $base/review_categories/Video_Games.jsonl
```

Variants: `--condition a2_swap|a3_hide|a4_shuffle|a6_allcold` (with `--neg random|hard`) and
`a1_oldgold|a1ctrl|a5_n1|a5_n2|a5_n5|a5_n10` (with `--neg trap`). See the project README for details.

Models (`configs/models.yaml`): `phi2` (pointwise only), `llama2_7b`, `llama2_13b`,
`mistral_7b`, `qwen_7b`. Methods: `direct`, `base`, `fast`, `structural`,
`structural_norecency`, `ssc`, `pointwise`. On SLURM:
```bash
sbatch --export=ALL,MODEL=llama2_7b,METHOD=direct,SPLIT=dev,NEG=random scripts/run_llm.sbatch
```
LLM outputs are cached in `outputs/cache/<model>.jsonl`, so a killed job resumes
where it stopped.

## Order of work on the cluster
1. **Smoke test** (does vLLM run on the 2080 Ti?): `--model llama2_7b --method direct --split dev --limit 2`.
2. **Dev runs** for every model × method on `dev` (150 pools × 2 negative settings): check
   `run_summary.json` for the parse-failure (`fail`) and `overflow_rate` values, and
   read some `raw.jsonl` outputs. Fix prompts **only** on dev.
3. Freeze the prompts, then run `test`.

## Tests
```bash
python -m pytest -q -m "not slow"   # fast unit tests
python -m pytest -q                  # + end-to-end HF run with SmolLM2-135M (downloads ~270 MB)
```
