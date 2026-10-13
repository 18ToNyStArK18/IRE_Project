"""Download the pinned model checkpoints (configs/models.yaml) into HF_HOME.

Run on the ADA login node (compute nodes may have no internet), then run jobs
with HF_HUB_OFFLINE=1 so they never touch the network.

    export HF_HOME=/path/with/30GB/free/hf_cache
    python scripts/download_models.py                 # all study models (~27 GB)
    python scripts/download_models.py llama2_7b       # just one
"""

import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs" / "models.yaml"))
STUDY = ["llama2_7b", "mistral_7b", "qwen_7b", "llama2_13b", "phi2"]

for key in sys.argv[1:] or STUDY:
    m = CFG[key]
    path = snapshot_download(m["hf_id"], revision=m["revision"])
    print(f"{key:12s} {m['hf_id']} @ {m['revision'][:10]} -> {path}")
