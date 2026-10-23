#!/bin/bash
# Run one stage of LLM tasks on a single machine without SLURM (e.g. a lab GPU box).
# The local counterpart of scripts/llm_ada.sbatch: same task lists (make_tasks.py), same
# runner (coldrank.llm.run), same outputs, cache and manifest.
#
# Run from code/:
#   CONFIG=ebnerd MODEL=qwen25_32b STAGE=smoke-k15 GPU=0 TP=1 bash scripts/run_local.sh   # ~15 min
#   CONFIG=ebnerd MODEL=qwen25_32b STAGE=pilot-dev GPU=0 TP=1 nohup bash scripts/run_local.sh > run.log 2>&1 &
# GPU: comma-separated device ids this run may use (default 0); TP: GPUs per model (default 1).
# Finished tasks are skipped on a rerun, and the generation cache (one file per machine) keeps every
# model output already produced, so an interrupted run can simply be started again.
set -euo pipefail
CONFIG=${CONFIG:-ebnerd}
MODEL=${MODEL:-qwen25_32b}
STAGE=${STAGE:-smoke-k15}
GPU=${GPU:-0}
TP=${TP:-1}
MAX_USED_MIB=${MAX_USED_MIB:-1000}   # refuse to start if a requested GPU already has more memory in use

cd "$(dirname "$0")/.."               # code/
source "${VENV:-../.venv}/bin/activate"
export HF_HOME=${HF_HOME:-$PWD/../hf_cache}
export HF_HUB_OFFLINE=1               # the model is downloaded beforehand (scripts/download_models.py)
export VLLM_USE_V1=0                  # same engine as on ADA
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=$GPU      # never touch the other GPUs of a shared machine

# the requested GPUs must be idle (other users' jobs may be running on the same machine)
for g in ${GPU//,/ }; do
    used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used > MAX_USED_MIB )); then
        echo "GPU $g already has $used MiB in use: not starting (set GPU=... to another free GPU)"; exit 1
    fi
done

START_TS=$(date +%Y-%m-%dT%H:%M:%S)
RUN_ID=$(date +%Y%m%d-%H%M%S)
TAG=${MODEL}_${STAGE}_local-${RUN_ID}
[[ "$CONFIG" != "amazon_vg" ]] && TAG=${CONFIG}_${TAG}
TASKS=../outputs/tasks/${TAG}
LOGS=../outputs/logs/${TAG}
mkdir -p "$TASKS" "$LOGS"

echo "host=$(hostname) stage=$STAGE model=$MODEL config=$CONFIG gpus=$GPU tp=$TP"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)"

python scripts/make_tasks.py --config "$CONFIG" --model "$MODEL" --stage "$STAGE" --shards 1 --out "$TASKS"
echo "queue: $(grep -c . "$TASKS/queue.txt") tasks -> log $LOGS/gpu0.log"

status=0
python -u -m coldrank.llm.run --config "$CONFIG" --model "$MODEL" --jobs-file "$TASKS/queue.txt" \
    --claim-dir "$TASKS/claims" --cache-tag "local-$(hostname -s)" --tensor-parallel "$TP" \
    --verify-pointwise 8 > "$LOGS/gpu0.log" 2>&1 || status=$?
grep -h '"task"' "$LOGS/gpu0.log" || true
SLURM_JOB_ID="local-${RUN_ID}" python scripts/write_manifest.py --config "$CONFIG" --model "$MODEL" \
    --stage "$STAGE" --tasks "$TASKS" --logs "$LOGS" --start "$START_TS" --status "$status" \
    || echo "manifest could not be written"
echo "stage $STAGE finished with status $status (log: $LOGS/gpu0.log)"
exit $status
