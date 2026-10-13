"""Build every adversarial variant pool, check it, and write it next to the baseline.

Usage:
    python -m coldrank.adversarial.build --config amazon_vg
    python -m coldrank.adversarial.build --only a2_swap,a3_hide

Writes pools/<condition>_<neg>_<split>.jsonl and pools/variant_stats.json. The
baseline pool files and the baseline embedding cache are only read.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from coldrank.adversarial.allcold import build_a6
from coldrank.adversarial.checks import (
    check_a1, check_a1ctrl, check_a5, check_a6, check_date_variant, manipulation_stats,
)
from coldrank.adversarial.common import extended_embeddings
from coldrank.adversarial.date_variants import DATE_VARIANTS
from coldrank.adversarial.oldgold import NEG as A1_NEG, build_oldgold
from coldrank.common import date_to_ts, load_config, read_jsonl, write_jsonl
from coldrank.data.split import split_dir

SPLITS = ("dev", "test")


def all_conditions(cfg: dict) -> list[str]:
    return ["a2_swap", "a3_hide", "a4_shuffle", "a6_allcold", "a1_oldgold", "a1ctrl"] + [
        f"a5_n{n}" for n in cfg["adversarial"]["flood_ns"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--only", default=None, help="comma-separated subset of conditions")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sd = split_dir(cfg)
    pool_dir = sd / "pools"
    T = date_to_ts(str(cfg["cutoff_T"]))
    wanted = set(args.only.split(",")) if args.only else set(all_conditions(cfg))
    unknown = wanted - set(all_conditions(cfg))
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")

    stats_path = pool_dir / "variant_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    written: dict[str, list[dict]] = {}

    def write(cond: str, neg: str, pools: list[dict]) -> None:
        for split in SPLITS:
            rows = [p for p in pools if p["split"] == split]
            write_jsonl(pool_dir / f"{cond}_{neg}_{split}.jsonl", rows)
            stats.setdefault(cond, {})[f"{neg}_{split}"] = manipulation_stats(cond, rows)
        written.setdefault(cond, []).extend(pools)

    # ---- A2 / A3 / A4: rewrite dates of the baseline pools ------------------ #
    for cond in [c for c in DATE_VARIANTS if c in wanted]:
        for neg in cfg["pool"]["neg_settings"]:
            pools = []
            for split in SPLITS:
                for base in read_jsonl(pool_dir / f"baseline_{neg}_{split}.jsonl"):
                    q = DATE_VARIANTS[cond](base)
                    check_date_variant(q, base)
                    pools.append(q)
            write(cond, neg, pools)

    # ---- pools that need new candidates: A6 and the A1 family -------------- #
    need_a6 = "a6_allcold" in wanted
    need_a1 = bool(wanted & ({"a1_oldgold", "a1ctrl"} | {f"a5_n{n}" for n in cfg["adversarial"]["flood_ns"]}))
    if need_a6 or need_a1:
        items = pd.read_parquet(sd / "items.parquet")
        pos = pd.read_parquet(sd / "positives.parquet")
        pop = pos[pos["ts"] < T].groupby("item_id").size().to_dict()
        topk = {r["instance_id"]: r["items"] for r in read_jsonl(sd / "sasrec_topk.jsonl")}
        cold_inst = [r for s in SPLITS for r in read_jsonl(sd / f"cold_{s}.jsonl")]
        old_inst = [r for s in SPLITS for r in read_jsonl(sd / f"old_{s}.jsonl")]
        t0 = items.set_index("item_id")["t0"].astype("int64")
        w_end = T + int(cfg["window_days"]) * 86400
        need = set(t0.index[(t0 > T) & (t0 < w_end)])
        for r in cold_inst + old_inst:
            need |= {h[0] for h in r["history"]} | set(topk[r["instance_id"]]) | {r["target_item"]}
        emb = extended_embeddings(items, need, cfg["pool"]["embed_model"], sd / "item_embeddings.npz",
                                  sd / "item_embeddings_adv.npz")

        if need_a6:
            pools, st = build_a6(cold_inst, items, emb, pop, T, cfg)
            for p in pools:
                check_a6(p, cfg["adversarial"]["a6_n_negs"])
            for neg in cfg["pool"]["neg_settings"]:
                write("a6_allcold", neg, [p for p in pools if p["neg_setting"] == neg])
            stats["a6_allcold"]["build"] = st

        if need_a1:
            ac = cfg["adversarial"]
            fam, st = build_oldgold(old_inst, topk, items, emb, pop, T, cfg)
            a1_by_id = {p["pool_id"]: p for p in fam["a1_oldgold"]}
            for p in fam["a1_oldgold"]:
                check_a1(p, ac["a1_n_warm"], ac["a1_n_traps"])
            for p in fam["a1ctrl"]:
                check_a1ctrl(p, a1_by_id[p["reference_pool_id"]], ac["ctrl_fill"])
            for n in ac["flood_ns"]:
                for p in fam[f"a5_n{n}"]:
                    check_a5(p, a1_by_id[p["reference_pool_id"]], n)
            for cond, pools in fam.items():
                if cond in wanted:
                    write(cond, A1_NEG, pools)
            stats.setdefault("a1_oldgold", {})["build"] = st

    stats_path.write_text(json.dumps(stats, indent=2))
    for cond, pools in written.items():
        print(f"[{cond}] {len(pools)} pools written, all invariants passed")
    print(json.dumps({c: stats[c] for c in written}, indent=2))


if __name__ == "__main__":
    main()
