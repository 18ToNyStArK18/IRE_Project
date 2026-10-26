"""Pointwise SFT examples from the training pools.

Per pool: the gold ("Yes"), both cold negatives and `finetune.warm_per_pool` of the 12 warm
candidates (seeded) ("No"). The prompt is built exactly as at inference (`llm.run.run_pointwise`):
`render_pointwise` with the pointwise history length, `format_chat`, and the oldest history
items dropped until the prompt fits the context with one answer token. Users are split into
train / validation (`finetune.holdout_frac`, seeded by user id).

Outputs (in <split dir>/finetune/): sft_train.jsonl, sft_val.jsonl, sft_stats.json

Usage:
    python -m coldrank.finetune.sft_data --config amazon_vg --model llama2_7b_fp16
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from coldrank.adversarial.common import COLD_NEG, GOLD, WARM
from coldrank.common import load_config, read_jsonl, rng_for, write_jsonl
from coldrank.finetune.trainpools import ft_dir
from coldrank.llm.chat import format_chat
from coldrank.llm.run import domain_of
from coldrank.prompts.render import render_pointwise


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


def build_examples(pools: list[dict], cfg: dict, family: str, count_tokens, max_len: int) -> list[dict]:
    fc = cfg["finetune"]
    rows = []
    for p in pools:
        for c in select_candidates(p, int(fc["warm_per_pool"])):
            prompt, trimmed = pointwise_prompt(p, c, cfg, family, count_tokens, max_len)
            rows.append({"pool_id": p["pool_id"], "user_id": p["user_id"], "item_id": c["item_id"],
                         "role": c["role"], "label": "Yes" if c["is_gold"] else "No",
                         "trimmed": trimmed, "prompt": prompt})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--model", default="llama2_7b_fp16", help="models.yaml key: tokenizer and chat format")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = load_config("models")[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    count = lambda s: len(tok(s)["input_ids"])  # noqa: E731  (with BOS, as vLLM counts)

    fd = ft_dir(cfg)
    pools = list(read_jsonl(fd / "train_pools_k15.jsonl"))
    rows = build_examples(pools, cfg, mcfg["family"], count, mcfg["max_model_len"])
    frac = float(cfg["finetune"]["holdout_frac"])
    val = [r for r in rows if is_holdout(r["user_id"], frac)]
    train = [r for r in rows if not is_holdout(r["user_id"], frac)]
    write_jsonl(fd / "sft_train.jsonl", train)
    write_jsonl(fd / "sft_val.jsonl", val)
    lens = np.array([count(r["prompt"]) for r in rows])
    stats = {
        "model": args.model, "n_pools": len(pools), "n_train": len(train), "n_val": len(val),
        "n_users_train": len({r["user_id"] for r in train}), "n_users_val": len({r["user_id"] for r in val}),
        "positive_rate": float(np.mean([r["label"] == "Yes" for r in rows])),
        "roles": {k: sum(r["role"] == k for r in rows) for k in (GOLD, COLD_NEG, WARM)},
        "prompt_tokens": {"mean": float(lens.mean()), "p50": float(np.percentile(lens, 50)),
                          "p95": float(np.percentile(lens, 95)), "max": int(lens.max()), "total": int(lens.sum())},
        "history_trimmed": sum(r["trimmed"] for r in rows),
    }
    (fd / "sft_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
