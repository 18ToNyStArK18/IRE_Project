#!/bin/bash
# One (model, method, split, neg) LLM run on one GPU.
#   sbatch --export=ALL,MODEL=llama2_7b,METHOD=direct,SPLIT=dev,NEG=random scripts/run_llm.sbatch
# Fill in the account/partition lines for ADA before use.
#SBATCH --job-name=coldrank
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=../outputs/logs/%x-%j.out
##SBATCH -A <account>
##SBATCH --partition=<partition>
##SBATCH --qos=<low-qos>

set -euo pipefail
: "${MODEL:?set MODEL}" "${METHOD:?set METHOD}"
SPLIT=${SPLIT:-dev}
NEG=${NEG:-random}
BACKEND=${BACKEND:-vllm}
CONDITION=${CONDITION:-baseline}

cd "$SLURM_SUBMIT_DIR"
source "${VENV:-../.venv}/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m coldrank.llm.run --model "$MODEL" --method "$METHOD" --split "$SPLIT" --neg "$NEG" \
    --condition "$CONDITION" --backend "$BACKEND" ${LIMIT:+--limit $LIMIT}
