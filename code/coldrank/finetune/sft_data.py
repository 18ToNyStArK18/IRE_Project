"""Pointwise SFT examples from the training pools.

Per pool: the gold ("Yes"), both cold negatives and `finetune.warm_per_pool` of the 12 warm
candidates (seeded) ("No"). The prompt is built exactly as at inference (`llm.run.run_pointwise`):
`render_pointwise` with the pointwise history length, `format_chat`, and the oldest history
items dropped until the prompt fits the context with one answer token. Users are split into
train / validation (`finetune.holdout_frac`, seeded by user id).

Outputs (in <split dir>/finetune/): sft_train.jsonl, sft_val.jsonl, sft_stats.json for Llama-2
(`llama2_7b_fp16`); sft_{train,val}_<model>.jsonl, sft_stats_<model>.json for any other model

`--variant nodates` (dates-hidden control, design.md): the same pools, candidates and users, with
every displayed date removed as in A3; files sft_{train,val}_<model>_nodates.jsonl.

`--variant oldmix` (fix experiment, coldrank.finetune.oldmix): half of each cutoff's pools have an old
correct item; files sft_{train,val}_<model>_oldmix.jsonl.

`--variant oldonly` (Li-style control): every pool has an old correct item and no new item (gold + 12 warm);
the two cold-negative slots go to warm candidates, so each pool still gives gold + 5 wrong examples;
files sft_{train,val}_<model>_oldonly.jsonl.

Usage:
    python -m coldrank.finetune.sft_data --config amazon_vg --model llama2_7b_fp16
    python -m coldrank.finetune.sft_data --config amazon_vg --model llama2_7b_fp16 --variant nodates
    python -m coldrank.finetune.sft_data --config amazon_vg --model llama2_7b_fp16 --variant oldmix
    python -m coldrank.finetune.sft_data --config amazon_vg --model llama2_7b_fp16 --variant oldonly
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coldrank.adversarial.common import COLD_NEG, GOLD, WARM
from coldrank.adversarial.resize import N_COLD_NEG
from coldrank.common import load_config, read_jsonl, rng_for, write_jsonl
from coldrank.finetune.trainpools import ft_dir
from coldrank.llm.chat import format_chat
from coldrank.llm.run import domain_of
from coldrank.prompts.render import render_pointwise


LEGACY_MODEL = "llama2_7b_fp16"  # its files keep the unsuffixed names the ADA scripts expect


def sft_paths(fd: Path, model: str, variant: str = "") -> tuple[Path, Path, Path]:
    """(train, val, stats) files of a model's SFT examples; the prompts depend on the model's
    tokenizer and chat format, so every model has its own files. A data variant (e.g. "imp",
    coldrank.finetune.impressions) always gets the model and variant suffix."""
    sfx = "" if model == LEGACY_MODEL and not variant else f"_{model}" + (f"_{variant}" if variant else "")
    return fd / f"sft_train{sfx}.jsonl", fd / f"sft_val{sfx}.jsonl", fd / f"sft_stats{sfx}.json"


def sft_dir(cfg: dict, variant: str = "") -> Path:
    """Directory of the SFT files: <split>/finetune/ (also "nodates", "oldmix", "oldonly"), or <split>/finetune/impressions/ for "imp"."""
    if variant not in ("", "imp", "nodates", "oldmix", "oldonly"):
        raise ValueError(f"unknown sft_variant {variant!r}")
    return ft_dir(cfg) / "impressions" if variant == "imp" else ft_dir(cfg)


def hide_dates(pool: dict) -> dict:
    """The pool with every displayed date removed, as make_a3_hide does at evaluation. pool_id and
    candidate order are kept, so the same candidates and users are selected as in the dated data."""
    return {**pool, "history": [{**h, "date": None} for h in pool["history"]],
            "candidates": [{**c, "date": None} for c in pool["candidates"]]}


def select_candidates(pool: dict, n_warm: int) -> list[dict]:
    """Gold + every cold negative + n_warm warm candidates (seeded by pool id), in pool order."""
    warm = [c for c in pool["candidates"] if c["role"] == WARM]
    keep = {c["item_id"] for c in pool["candidates"] if c["role"] in (GOLD, COLD_NEG)}
    keep |= {c["item_id"] for c in rng_for("ft_warm", pool["pool_id"]).sample(warm, min(n_warm, len(warm)))}
    return [c for c in pool["candidates"] if c["item_id"] in keep]


def is_holdout(user_id: str, frac: float) -> bool:
    return rng_for("ft_holdout", user_id).random() < frac


def pointwise_prompt(pool: dict, cand: dict, cfg: dict, family: str, count_tokens, max_len: int) -> tuple[str, bool]:
    """The inference prompt (llm.run.run_pointwise) and whether history items were dropped to fit."""
    hist_len = cfg["prompt"]["history_len"]["pointwise"]
    h = hist_len
    while True:
        sys_, user = render_pointwise(pool, cand, h, domain_of(cfg))
        prompt = format_chat(family, sys_, user)
        if count_tokens(prompt) < max_len or h <= 1:
            return prompt, h < min(hist_len, len(pool["history"]))
        h -= 1


def build_examples(pools: list[dict], cfg: dict, family: str, count_tokens, max_len: int, n_warm: int | None = None) -> list[dict]:
    n_warm = int(cfg["finetune"]["warm_per_pool"]) if n_warm is None else n_warm
    rows = []
    for p in pools:
        for c in select_candidates(p, n_warm):
            prompt, trimmed = pointwise_prompt(p, c, cfg, family, count_tokens, max_len)
            rows.append({"pool_id": p["pool_id"], "user_id": p["user_id"], "item_id": c["item_id"],
                         "role": c["role"], "label": "Yes" if c["is_gold"] else "No",
                         "trimmed": trimmed, "prompt": prompt})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--model", default="llama2_7b_fp16", help="models.yaml key: tokenizer and chat format")
    ap.add_argument("--variant", default="", choices=["", "nodates", "oldmix", "oldonly"],
                    help="nodates: dates-hidden control; oldmix: half the pools have an old correct item; "
                         "oldonly: every correct item old, no new items (Li-style)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = load_config("models")[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    count = lambda s: len(tok(s)["input_ids"])  # noqa: E731  (with BOS, as vLLM counts)

    fd = ft_dir(cfg)
    pool_file = {"oldmix": "train_pools_oldmix_k15.jsonl", "oldonly": "train_pools_oldonly_k15.jsonl"}.get(args.variant, "train_pools_k15.jsonl")
    pools = list(read_jsonl(fd / pool_file))
    if args.variant == "nodates":
        pools = [hide_dates(p) for p in pools]
    n_warm = int(cfg["finetune"]["warm_per_pool"])
    if args.variant == "oldonly":
        assert not any(c["role"] == COLD_NEG or c["is_cold"] for p in pools for c in p["candidates"]), "oldonly pools contain a new item"
        n_warm += N_COLD_NEG  # same examples per pool as the other variants
    rows = build_examples(pools, cfg, mcfg["family"], count, mcfg["max_model_len"], n_warm)
    frac = float(cfg["finetune"]["holdout_frac"])
    val = [r for r in rows if is_holdout(r["user_id"], frac)]
    train = [r for r in rows if not is_holdout(r["user_id"], frac)]
    train_path, val_path, stats_path = sft_paths(fd, args.model, args.variant)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    lens = np.array([count(r["prompt"]) for r in rows])
    stats = {
        "model": args.model, "variant": args.variant, "n_pools": len(pools), "n_train": len(train), "n_val": len(val),
        "n_users_train": len({r["user_id"] for r in train}), "n_users_val": len({r["user_id"] for r in val}),
        "positive_rate": float(np.mean([r["label"] == "Yes" for r in rows])),
        "roles": {k: sum(r["role"] == k for r in rows) for k in (GOLD, COLD_NEG, WARM)},
        "prompt_tokens": {"mean": float(lens.mean()), "p50": float(np.percentile(lens, 50)),
                          "p95": float(np.percentile(lens, 95)), "max": int(lens.max()), "total": int(lens.sum())},
        "history_trimmed": sum(r["trimmed"] for r in rows),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
