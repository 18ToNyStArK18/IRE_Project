"""Training pools for fine-tuning: the test protocol, replayed at earlier cutoffs T' < T.

Per cutoff T' (config `finetune.cutoffs`), with the code that built the test pools:
  1. instances: `data.split.build_instances` at (T', window_days); dev/test users of every
     kind are removed; `require_cold_negs` as for test; optional seeded cap per cutoff;
  2. warm ranker: SASRec retrained on the positives before T' (dataset hyperparameters);
  3. K = 50 pools (`pools.build.build_pools`), shrunk to K = 15 (`adversarial.resize.shrink_baseline`);
     each instance keeps one negative setting, random or hard (seeded 50/50).
Every window ends before T, so no training target lies in or after the test window.

Outputs (in <split dir>/finetune/):
  cutoff_<T'>/pools_k15.jsonl, stats.json, sasrec_report.json, item_embeddings.npz
  train_pools_k15.jsonl (after --collect: all cutoffs)

Usage:
    python -m coldrank.finetune.trainpools --config amazon_vg --cutoff 2019-09-01   # one cutoff (GPU)
    python -m coldrank.finetune.trainpools --config amazon_vg --collect             # merge the cutoffs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.adversarial.common import extended_embeddings
from coldrank.adversarial.resize import N_COLD_NEG, N_WARM, shrink_baseline
from coldrank.common import date_to_ts, load_config, read_jsonl, rng_for, ts_to_date, write_jsonl
from coldrank.data.split import build_instances, require_cold_negs, split_dir
from coldrank.pools.build import build_pools, check_pool, pool_summary

EVAL_FILES = ("cold_dev", "cold_test", "old_dev", "old_test")


def ft_dir(cfg: dict) -> Path:
    return split_dir(cfg) / "finetune"


def cutoff_tag(cutoff: str) -> str:
    return str(cutoff).replace(" ", "_").replace(":", "")


def eval_users(sd: Path) -> set:
    """Users of every dev/test instance file: never used for training."""
    users = set()
    for name in EVAL_FILES:
        users |= {r["user_id"] for r in read_jsonl(sd / f"{name}.jsonl")}
    return users


def check_cutoffs(cfg: dict) -> list[str]:
    """Cutoffs in order; each window must end by the next cutoff and by T."""
    cuts = [str(c) for c in cfg["finetune"]["cutoffs"]]
    ts = [date_to_ts(c) for c in cuts]
    w = int(cfg["window_days"]) * 86400
    ends = ts[1:] + [date_to_ts(str(cfg["cutoff_T"]))]
    if ts != sorted(ts) or any(t + w > e for t, e in zip(ts, ends)):
        raise ValueError(f"training windows overlap each other or the test window: {cuts}")
    return cuts


def training_instances(pos: pd.DataFrame, items: pd.DataFrame, cfg: dict, cutoff: str,
                       excluded: set) -> tuple[list[dict], dict]:
    """Cold-target instances at T' = cutoff (test rules), without dev/test users, capped."""
    T = date_to_ts(cutoff)
    res = build_instances(pos, items, T, int(cfg["window_days"]), int(cfg["min_history"]))
    inst = [r for r in res.cold_instances if r["user_id"] not in excluded]
    stats = {"cutoff": cutoff, "n_cold_instances": len(res.cold_instances), "n_not_eval_users": len(inst)}
    if cfg.get("require_cold_negs"):
        inst, n_drop = require_cold_negs(inst, items, T, int(cfg["require_cold_negs"]))
        stats["dropped_too_few_cold_negs"] = n_drop
    cap = cfg["finetune"].get("max_instances_per_cutoff")
    inst = sorted(inst, key=lambda r: r["user_id"])  # canonical order, then seeded shuffle
    rng_for("ft_subsample", cfg["dataset"], cutoff, cfg["seed"]).shuffle(inst)
    if cap:
        inst = inst[:cap]
    tag = cutoff_tag(cutoff)
    for k, r in enumerate(inst):
        r["instance_id"] = f"train_{tag}_{k:05d}"
        r["split"] = "train"
    stats["n_instances"] = len(inst)
    return inst, stats


def neg_setting(instance_id: str) -> str:
    """One negative setting per training instance (seeded 50/50)."""
    return "random" if rng_for("ft_neg", instance_id).random() < 0.5 else "hard"


def k15_pools(pools50: list[dict]) -> list[dict]:
    out = []
    for p in pools50:
        if p["neg_setting"] != neg_setting(p["instance_id"]):
            continue
        q = shrink_baseline(p)
        q["condition"] = "k15_train"
        check_pool(q, N_WARM, N_COLD_NEG + 1)
        out.append(q)
    return out


def rank_and_pool(cfg: dict, inst: list[dict], items: pd.DataFrame, pos: pd.DataFrame, T: int, out: Path, device: str,
                  cutoff: str, topn: int = 100) -> tuple[list[dict], dict, float]:
    """Steps 2-3 up to the K = 50 pools for a list of instances: writes sasrec_report.json and the
    embeddings to `out`; returns (K = 50 pools, pool step stats, SASRec best validation NDCG@10)."""
    from coldrank.warm.sasrec import build_sequences, kcore, topk_for_instances, train

    sd = split_dir(cfg)
    hp = cfg["sasrec"]
    pre = pos[pos["ts"] < T]
    train_df = kcore(pre, hp["kcore"])
    vocab, seqs = build_sequences(train_df)
    print(f"[ft {cutoff}] {len(inst)} instances; SASRec on {len(train_df):,} interactions, "
          f"{len(seqs):,} users, {len(vocab):,} items; device={device}", flush=True)
    model, curve = train(seqs, len(vocab), hp, device)
    popularity = pre.groupby("item_id").size().sort_values(ascending=False, kind="mergesort")
    topk = topk_for_instances(model, vocab, inst, topn, hp["max_seq_len"], device, popularity)
    (out / "sasrec_report.json").write_text(json.dumps({
        "n_train_interactions": len(train_df), "n_users": len(seqs), "n_items": len(vocab),
        "best_val_ndcg10": max(c["val_ndcg10"] for c in curve), "curve": curve,
        "fallback_rate": float(np.mean([r["source"] != "sasrec" for r in topk])) if topk else 0.0,
    }, indent=2))

    t0 = items.set_index("item_id")["t0"].astype("int64")
    w_end = T + int(cfg["window_days"]) * 86400
    need = set(t0.index[(t0 > T) & (t0 < w_end)])
    for r, tk in zip(inst, topk):
        need |= {h[0] for h in r["history"]} | set(tk["items"]) | {r["target_item"]}
    emb = extended_embeddings(items, need, cfg["pool"]["embed_model"], sd / "item_embeddings.npz",
                              out / "item_embeddings.npz")

    pop = pre.groupby("item_id").size().to_dict()
    pools50, pstats = build_pools(inst, {r["instance_id"]: r["items"] for r in topk}, items, emb, pop, T, cfg)
    return pools50, pstats, max(c["val_ndcg10"] for c in curve)


def build_cutoff(cfg: dict, cutoff: str, device: str, topn: int = 100) -> dict:
    """Steps 1-3 for one cutoff; writes cutoff_<T'>/ and returns its stats."""
    sd = split_dir(cfg)
    out = ft_dir(cfg) / f"cutoff_{cutoff_tag(cutoff)}"
    out.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(sd / "items.parquet")
    pos = pd.read_parquet(sd / "positives.parquet")
    T = date_to_ts(cutoff)

    inst, stats = training_instances(pos, items, cfg, cutoff, eval_users(sd))
    pools50, pstats, best = rank_and_pool(cfg, inst, items, pos, T, out, device, cutoff, topn)
    for p in pools50:
        check_pool(p, cfg["pool"]["n_warm"], cfg["pool"]["n_cold"])
    pools = k15_pools(pools50)
    write_jsonl(out / "pools_k15.jsonl", pools)
    stats.update(cutoff_T=ts_to_date(T), pool_step=pstats, n_pools=len(pools),
                 n_random=sum(p["neg_setting"] == "random" for p in pools),
                 sasrec_best_val_ndcg10=best,
                 summary=pool_summary(pools) if pools else {})
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "summary"}), flush=True)
    return stats


def collect(cfg: dict) -> dict:
    """Concatenate the per-cutoff pools; check that no dev/test user or window leaked in."""
    sd, fd = split_dir(cfg), ft_dir(cfg)
    excluded, T = eval_users(sd), date_to_ts(str(cfg["cutoff_T"]))
    pools, per_cut = [], {}
    for c in check_cutoffs(cfg):
        f = fd / f"cutoff_{cutoff_tag(c)}" / "pools_k15.jsonl"
        if not f.exists():
            raise SystemExit(f"missing {f}: run --cutoff {c} first")
        ps = list(read_jsonl(f))
        per_cut[c] = len(ps)
        pools += ps
    for p in pools:
        assert p["user_id"] not in excluded, p["pool_id"]
        assert p["target_ts"] < T and all(c["t0"] <= p["target_ts"] for c in p["candidates"]), p["pool_id"]
    assert len({p["pool_id"] for p in pools}) == len(pools)
    n = write_jsonl(fd / "train_pools_k15.jsonl", pools)
    stats = {"n_pools": n, "per_cutoff": per_cut, "n_users": len({p["user_id"] for p in pools}),
             "n_distinct_golds": len({p["gold_item_id"] for p in pools}), "summary": pool_summary(pools)}
    (fd / "train_pools_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--cutoff", help="build one cutoff (one of finetune.cutoffs)")
    ap.add_argument("--collect", action="store_true", help="merge the per-cutoff pools")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cuts = check_cutoffs(cfg)
    if args.cutoff:
        if args.cutoff not in cuts:
            raise SystemExit(f"{args.cutoff} is not in finetune.cutoffs {cuts}")
        import torch

        build_cutoff(cfg, args.cutoff, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.collect:
        collect(cfg)
    if not (args.cutoff or args.collect):
        raise SystemExit("give --cutoff or --collect")


if __name__ == "__main__":
    main()
