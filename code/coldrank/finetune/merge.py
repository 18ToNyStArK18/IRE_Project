"""Merge a LoRA adapter into its base (in the base's dtype, fp16 by default) so vLLM loads the
fine-tuned model like any other.

A models.yaml entry with `base` and `adapter` is resolved here (called by llm.run): the merged
weights go to node-local disk (13.5 GB per model; $COLDRANK_MERGED_DIR, default
/ssd_scratch/$USER/coldrank_merged, or /scratch if the SSD has too little room: job 49867 stopped on
a full /ssd_scratch), one directory per adapter content hash, built once per node under a file lock. The generation cache identifies the model by base + adapter hash, never by
the local path.

Usage (optional: llm.run merges on first use):
    python -m coldrank.finetune.merge --model llama2_7b_ft_amazon_vg
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
from pathlib import Path

from coldrank.common import PROJECT_DIR, load_config

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
MIN_FREE_BYTES = 20 * 1024**3  # one merged model plus margin
LOCAL_DISKS = ("/ssd_scratch", "/scratch")


def merged_root() -> Path:
    if os.environ.get("COLDRANK_MERGED_DIR"):
        return Path(os.environ["COLDRANK_MERGED_DIR"])
    for disk in LOCAL_DISKS:
        if Path(disk).is_dir() and shutil.disk_usage(disk).free >= MIN_FREE_BYTES:
            return Path(disk) / os.environ.get("USER", "user") / "coldrank_merged"
    return PROJECT_DIR / "merged"


def adapter_sha(adapter_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ADAPTER_FILES:
        h.update(name.encode())
        h.update((adapter_dir / name).read_bytes())
    return h.hexdigest()


def merge(base: dict, adapter_dir: Path, target: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tmp = target.with_name(target.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model = AutoModelForCausalLM.from_pretrained(base["hf_id"], revision=base["revision"],
                                                 torch_dtype=getattr(torch, base.get("dtype", "float16")))
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(base["hf_id"], revision=base["revision"]).save_pretrained(tmp)
    tmp.rename(target)


def resolve_adapter(mcfg: dict) -> dict:
    """models.yaml entry with `base` + `adapter` -> engine config pointing at the merged weights."""
    base = load_config("models")[mcfg["base"]]
    adapter_dir = PROJECT_DIR / mcfg["adapter"]
    sha = adapter_sha(adapter_dir)
    root = merged_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{mcfg['base']}__{sha[:16]}"
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several workers on one node: merge once
        if not (target / "config.json").exists():
            print(f"[merge] {adapter_dir} into {base['hf_id']} -> {target}", flush=True)
            merge(base, adapter_dir, target)
    # quantization: the fine-tuned entry's own load-time setting (e.g. bitsandbytes on one 11 GB GPU)
    return {**mcfg, "hf_id": str(target), "revision": None, "quantization": mcfg.get("quantization"), "adapter_sha": sha,
            "ident_hf_id": f"{base['hf_id']}@{base['revision']}+lora:{sha}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="models.yaml key with `base` and `adapter`")
    args = ap.parse_args()
    print(resolve_adapter(load_config("models")[args.model])["hf_id"])


if __name__ == "__main__":
    main()
