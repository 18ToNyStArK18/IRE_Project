#!/bin/bash
# Submit a whole stage of LLM jobs (one job per model x method x negative setting).
# Run from code/ after setup_ada.sh and download_models.py:
#   export HF_HOME=/path/to/hf_cache
#   bash scripts/submit.sh smoke      # 1 job: is vLLM working on the 2080 Ti?
#   bash scripts/submit.sh direct     # Direct-Rec + pointwise, all models, dev, both negative settings
#   bash scripts/submit.sh reasoning  # Base/Fast/Structural(+no-recency)/SSC, listwise models, dev
#   bash scripts/submit.sh test       # everything on test (only after prompts are frozen on dev)
# Add DRY=1 to print the sbatch commands without submitting.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ../outputs/logs   # SLURM cannot create the log directory itself

STAGE=${1:?usage: submit.sh smoke|direct|reasoning|test}
LISTWISE=(llama2_7b mistral_7b qwen_7b llama2_13b)
ALL=(llama2_7b mistral_7b qwen_7b llama2_13b phi2)
REASONING=(base fast structural structural_norecency ssc)
NEGS=(random hard)

submit() {  # MODEL METHOD SPLIT NEG [LIMIT]
    local exports="ALL,MODEL=$1,METHOD=$2,SPLIT=$3,NEG=$4${5:+,LIMIT=$5}"
    local cmd=(sbatch --job-name="cr-$1-$2-$3-$4" --export="$exports" scripts/run_llm.sbatch)
    if [[ -n "${DRY:-}" ]]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

case "$STAGE" in
    smoke)
        submit llama2_7b direct dev random 2
        ;;
    direct)
        for neg in "${NEGS[@]}"; do
            for m in "${LISTWISE[@]}"; do submit "$m" direct dev "$neg"; done
            for m in "${ALL[@]}"; do submit "$m" pointwise dev "$neg"; done
        done
        ;;
    reasoning)
        for neg in "${NEGS[@]}"; do
            for m in "${LISTWISE[@]}"; do
                for meth in "${REASONING[@]}"; do submit "$m" "$meth" dev "$neg"; done
            done
        done
        ;;
    test)
        for neg in "${NEGS[@]}"; do
            for m in "${LISTWISE[@]}"; do
                for meth in direct "${REASONING[@]}"; do submit "$m" "$meth" test "$neg"; done
            done
            for m in "${ALL[@]}"; do submit "$m" pointwise test "$neg"; done
        done
        ;;
    *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
