"""Old-gold training mix (the fix experiment, design.md): half of each cutoff's training pools have
an old correct item, so a new date no longer nearly guarantees the answer.

Per cutoff T' (config `finetune.cutoffs`), with the code that built the cold-target training pools:
  1. old-target instances: `data.split.build_instances` at (T', window_days), without dev/test users;
     users who also have a cold-target training pool at T' come first, then the others, each in a
     seeded order;
  2. warm ranker: SASRec retrained on the positives before T' (`trainpools.rank_and_pool`);
  3. K = 15 pools shaped like the cold ones: gold (old) + 12 warm + 2 new negatives.
--collect keeps the per-cutoff pool count: floor(n/2) old-target pools replace cold-target pools of the
same users where possible; the cold-target pools to keep are a seeded choice among the other users.

Outputs (in <split dir>/finetune/):
  cutoff_<T'>/old/pools_k15.jsonl, stats.json, sasrec_report.json, item_embeddings.npz
  train_pools_oldmix_k15.jsonl, train_pools_oldmix_stats.json (after --collect)

Usage:
    python -m coldrank.finetune.oldmix --config amazon_vg --cutoff 2019-09-01   # one cutoff (GPU)
    python -m coldrank.finetune.oldmix --config amazon_vg --collect             # merge the cutoffs
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from coldrank.adversarial.common import COLD_NEG, GOLD, WARM
from coldrank.adversarial.resize import N_COLD_NEG, N_WARM, shrink_baseline
from coldrank.common import date_to_ts, load_config, read_jsonl, rng_for, write_jsonl
from coldrank.data.split import build_instances, split_dir
from coldrank.finetune.trainpools import check_cutoffs, cutoff_tag, eval_users, ft_dir, neg_setting, rank_and_pool

POOLS_FILE = "train_pools_oldmix_k15.jsonl"


def n_old_for(n_pools: int) -> int:
    return n_pools // 2


def old_training_instances(old: list[dict], cold_users: set, excluded: set, cfg: dict, cutoff: str) -> tuple[list[dict], dict]:
    """Old-target instances at T' in preference order (users with a cold-target pool at T' first).
    Twice the needed number of other users is kept as a margin for the pool step's drops."""
    inst = sorted((r for r in old if r["user_id"] not in excluded), key=lambda r: r["user_id"])
    rng_for("ft_old_subsample", cfg["dataset"], cutoff, cfg["seed"]).shuffle(inst)
    same = [r for r in inst if r["user_id"] in cold_users]
    others = [r for r in inst if r["user_id"] not in cold_users]
    margin = 2 * n_old_for(len(cold_users))
    chosen = same + others[:margin]
    tag = cutoff_tag(cutoff)
    for k, r in enumerate(chosen):
        r["instance_id"] = f"trainold_{tag}_{k:05d}"
        r["split"] = "train"
    return chosen, {"cutoff": cutoff, "n_old_instances": len(old), "n_not_eval_users": len(inst),
                    "n_same_users": len(same), "n_chosen": len(chosen)}


def check_old_pool(p: dict) -> None:
    c = p["candidates"]
    ids = [x["item_id"] for x in c]
    T = date_to_ts(p["cutoff_T"])
    gold = [x for x in c if x["is_gold"]]
    assert len(ids) == N_WARM + N_COLD_NEG + 1 == len(set(ids)), p["pool_id"]
    assert len(gold) == 1 and gold[0]["item_id"] == p["gold_item_id"] and gold[0]["role"] == GOLD
    assert gold[0]["t0"] <= T and not gold[0]["is_cold"], p["pool_id"]
    assert sum(x["role"] == COLD_NEG for x in c) == N_COLD_NEG and sum(x["role"] == WARM for x in c) == N_WARM
    assert all(x["t0"] > T for x in c if x["role"] == COLD_NEG) and all(x["t0"] <= T for x in c if x["role"] == WARM)
    assert not set(ids) & {h["item_id"] for h in p["history"]}, p["pool_id"]
    assert all(h["ts"] < T for h in p["history"]) and all(x["t0"] <= p["target_ts"] for x in c), p["pool_id"]


def k15_old_pools(pools50: list[dict]) -> list[dict]:
    """One negative setting per instance, shrunk as for cold targets. build_pools marks the gold slot
    is_cold, which is what shrink_baseline expects; the record then gets the true value."""
    out = []
    for p in pools50:
        if p["neg_setting"] != neg_setting(p["instance_id"]):
            continue
        q = shrink_baseline(p)
        q["condition"] = "k15_train_old"
        next(c for c in q["candidates"] if c["is_gold"])["is_cold"] = False
        check_old_pool(q)
        out.append(q)
    return out


def mix_cutoff(cold: list[dict], old: list[dict], cutoff: str) -> tuple[list[dict], dict]:
    """Same pool count as `cold`: floor(n/2) old pools (in preference order) + cold pools of the other users."""
    n_old = n_old_for(len(cold))
    if len(old) < n_old:
        raise ValueError(f"{cutoff}: {len(old)} old-target pools, {n_old} needed")
    chosen = old[:n_old]
    users = {p["user_id"] for p in chosen}
    rest = sorted((p for p in cold if p["user_id"] not in users), key=lambda p: p["pool_id"])
    rng_for("ft_old_keep", cutoff).shuffle(rest)
    kept = rest[:len(cold) - n_old]
    return kept + chosen, {"n_cold_kept": len(kept), "n_old": n_old,
                           "n_old_same_user": len(users & {p["user_id"] for p in cold})}


def build_cutoff(cfg: dict, cutoff: str, device: str) -> dict:
    sd = split_dir(cfg)
    cut_dir = ft_dir(cfg) / f"cutoff_{cutoff_tag(cutoff)}"
    out = cut_dir / "old"
    out.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(sd / "items.parquet")
    pos = pd.read_parquet(sd / "positives.parquet")
    T = date_to_ts(cutoff)
    res = build_instances(pos, items, T, int(cfg["window_days"]), int(cfg["min_history"]))
    cold_users = {p["user_id"] for p in read_jsonl(cut_dir / "pools_k15.jsonl")}
    inst, stats = old_training_instances(res.old_instances, cold_users, eval_users(sd), cfg, cutoff)
    pools50, pstats, best = rank_and_pool(cfg, inst, items, pos, T, out, device, cutoff)
    pools = k15_old_pools(pools50)
    write_jsonl(out / "pools_k15.jsonl", pools)
    stats.update(pool_step=pstats, n_pools=len(pools), sasrec_best_val_ndcg10=best)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)
    return stats


def collect(cfg: dict) -> dict:
    """Mix every cutoff; check that no dev/test user or window leaked in."""
    sd, fd = split_dir(cfg), ft_dir(cfg)
    excluded, T = eval_users(sd), date_to_ts(str(cfg["cutoff_T"]))
    pools, per_cut = [], {}
    for c in check_cutoffs(cfg):
        cut = fd / f"cutoff_{cutoff_tag(c)}"
        if not (cut / "old" / "pools_k15.jsonl").exists():
            raise SystemExit(f"missing old-target pools for {c}: run --cutoff {c} first")
        mixed, st = mix_cutoff(list(read_jsonl(cut / "pools_k15.jsonl")), list(read_jsonl(cut / "old" / "pools_k15.jsonl")), c)
        per_cut[c] = st
        pools += mixed
    for p in pools:
        assert p["user_id"] not in excluded, p["pool_id"]
        assert p["target_ts"] < T and all(c["t0"] <= p["target_ts"] for c in p["candidates"]), p["pool_id"]
    assert len({p["pool_id"] for p in pools}) == len(pools)
    n = write_jsonl(fd / POOLS_FILE, pools)
    stats = {"n_pools": n, "per_cutoff": per_cut, "n_users": len({p["user_id"] for p in pools}),
             "n_old": sum(s["n_old"] for s in per_cut.values()),
             "n_old_same_user": sum(s["n_old_same_user"] for s in per_cut.values())}
    (fd / "train_pools_oldmix_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--cutoff", help="build one cutoff (one of finetune.cutoffs)")
    ap.add_argument("--collect", action="store_true", help="mix the per-cutoff pools")
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
