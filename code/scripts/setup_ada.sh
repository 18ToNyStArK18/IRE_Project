#!/bin/bash
# One-time environment setup on ADA (run on the login node, which has internet).
#   bash scripts/setup_ada.sh
# Creates a conda env "coldrank" with vLLM and the pipeline's dependencies.
set -euo pipefail

ENV_NAME=${ENV_NAME:-coldrank}

# Load conda if the cluster provides it as a module; otherwise use your own Miniconda.
if ! command -v conda >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || {
        echo "conda not found: install Miniconda in your home/share directory first"; exit 1; }
fi
eval "$(conda shell.bash hook)"

conda create -y -n "$ENV_NAME" python=3.10
conda activate "$ENV_NAME"

# vLLM brings a matching CUDA build of PyTorch and transformers.
pip install --upgrade pip
pip install vllm
# Pipeline dependencies (pandas etc.) and Qwen-7B-Chat's remote-code dependencies.
pip install pandas pyarrow pyyaml scipy tqdm pytest tiktoken einops transformers_stream_generator

python - <<'EOF'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
EOF
echo "Done. Activate with: conda activate $ENV_NAME"
