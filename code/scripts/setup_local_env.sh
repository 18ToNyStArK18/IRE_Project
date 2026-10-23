#!/bin/bash
# Create the inference environment in ../.venv (Python 3.12), with the package versions
# used on ADA for every run so far (vllm 0.8.5, torch 2.6.0, transformers 4.51.3).
# Does not touch any existing conda / system environment.
#
# Run from code/:   bash scripts/setup_local_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."               # code/
VENV=${VENV:-../.venv}

if ! command -v uv >/dev/null; then
    echo "installing uv (user-level, ~/.local/bin)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" \
    vllm==0.8.5 torch==2.6.0 transformers==4.51.3 tokenizers==0.21.4 xformers==0.0.29.post2 \
    "huggingface_hub<1.0" "numpy<2.3" pandas pyarrow scipy pyyaml tqdm
"$VENV/bin/python" -c "import vllm, torch, transformers; print('ok: vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
