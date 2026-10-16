"""Smaller candidate pools (team decision: K = 15) for the baseline and every variant.

Built FROM the 50-candidate pools, so the gold, history and SASRec order are unchanged,
and keeping the 4:1 warm:cold ratio of Li et al.'s 40 + 10:

  k15_baseline   12 best-SASRec warm + gold + 2 cold negatives
                   random negs: 2 of the 9 sampled (seeded); hard negs: the 2 most similar
  k15_a2_swap / k15_a3_hide / k15_a4_shuffle
                 the date variants re-applied to k15_baseline (same functions as at 50)
  k15_a6_allcold gold + 14 cold negatives from the a6 pool (random: sampled; hard: most similar)
  k15_a1_oldgold gold (old) + 11 best-SASRec warm + 3 of the 10 traps (sampled)
  k15_a1ctrl     k15_a1 with the 3 trap slots refilled by the next SASRec warm ranks (12-14)
  k15_a5_n{1,2,5,10}  k15_a1 + the n least similar warm negatives relabeled with new dates
                 (dates copied from the 50-pool's traps, all displayed after the cutoff)

Pool ids get a `__k15` suffix, conditions a `k15_` prefix, and every pool carries
pool_size = 15. All invariants of adversarial.checks are asserted.

Usage:
    python -m coldrank.adversarial.resize --config amazon_vg
"""

from __future__ import annotations

import argparse
import copy
import json

from coldrank.adversarial.checks import (
    check_a1, check_a1ctrl, check_a5, check_a6, check_common, check_date_variant, manipulation_stats,
)
from coldrank.adversarial.common import CTRL_FILL, GOLD, RELABELED, TRAP, WARM, annotate, derive
from coldrank.adversarial.date_variants import DATE_VARIANTS
from coldrank.common import load_config, pool_files, read_jsonl, rng_for, write_jsonl
from coldrank.data.split import split_dir

K = 15
N_WARM, N_COLD_NEG = 12, 2           # baseline: 12 + gold + 2 = 15
A1_WARM, A1_TRAPS = 11, 3            # A1 family: gold + 11 + 3 = 15
SPLITS = ("dev", "test")


def _role(c: dict) -> str:
    return c.get("role") or (GOLD if c["is_gold"] else ("cold_neg" if c["is_cold"] else WARM))


def _finish(q: dict, cands: list[dict], condition: str, ref: dict) -> dict:
    q["candidates"] = sorted(cands, key=lambda c: c["item_id"])  # neutral order
    q["condition"] = condition
    q["reference_pool_id"] = ref["pool_id"]
    q["pool_id"] = f"{ref['pool_id']}__k15"
    q["pool_size"] = K
    q.pop("shuffle_key", None)  # a fresh pool: shuffled by its own id
    return annotate(q)


def _pick_negs(negs: list[dict], n: int, setting: str, key: str) -> list[dict]:
    negs = sorted(negs, key=lambda c: c["item_id"])
    if setting == "hard":  # the most similar to the user, as in the 50-pool hard setting
        return sorted(negs, key=lambda c: (-c["sim_to_history"], c["item_id"]))[:n]
    return rng_for("k15_negs", key).sample(negs, n)


def shrink_baseline(p: dict) -> dict:
    q = copy.deepcopy(p)
    cands = q["candidates"]
    warm = sorted((c for c in cands if not c["is_cold"]), key=lambda c: c["warm_rank"])[:N_WARM]
    gold = [c for c in cands if c["is_gold"]]
    negs = _pick_negs([c for c in cands if c["is_cold"] and not c["is_gold"]], N_COLD_NEG, p["neg_setting"], p["pool_id"])
    return _finish(q, warm + gold + negs, "k15_baseline", p)


def shrink_a6(p: dict) -> dict:
    q = copy.deepcopy(p)
    gold = [c for c in q["candidates"] if c["is_gold"]]
    negs = _pick_negs([c for c in q["candidates"] if not c["is_gold"]], K - 1, p["neg_setting"], p["pool_id"])
    return _finish(q, gold + negs, "k15_a6_allcold", p)


def shrink_a1(p: dict) -> tuple[dict, list[dict], list[str]]:
    """k15 A1 pool, the next 3 warm items (A1-ctrl's fill) and the 50-pool's trap dates (A5)."""
    q = copy.deepcopy(p)
    cands = q["candidates"]
    warm_sorted = sorted((c for c in cands if _role(c) == WARM),
                         key=lambda c: (c["warm_rank"] if c["warm_rank"] is not None else 10**9, c["item_id"]))
    warm, fill = warm_sorted[:A1_WARM], warm_sorted[A1_WARM:A1_WARM + A1_TRAPS]
    gold = [c for c in cands if c["is_gold"]]
    traps = rng_for("k15_traps", p["pool_id"]).sample(sorted((c for c in cands if _role(c) == TRAP),
                                                             key=lambda c: c["item_id"]), A1_TRAPS)
    out = _finish(q, gold + warm + traps, "k15_a1_oldgold", p)
    out["trap_sim_threshold"] = p["trap_sim_threshold"]
    return out, fill, [c["date"] for c in cands if _role(c) == TRAP]


def a1ctrl_from(a1: dict, fill: list[dict]) -> dict:
    ctrl = derive(a1, "k15_a1ctrl")
    it = iter(sorted(fill, key=lambda c: (c["warm_rank"], c["item_id"])))
    new = []
    for c in ctrl["candidates"]:
        if c["role"] == TRAP:
            f = copy.deepcopy(next(it))
            f["role"] = CTRL_FILL
            new.append(f)
        else:
            new.append(c)
    ctrl["candidates"] = new
    ctrl["pool_id"] = a1["pool_id"].replace("__k15", "").replace("a1_oldgold", "a1ctrl") + "__k15"
    return annotate(ctrl)


def a5_from(a1: dict, n: int, trap_dates: list[str]) -> dict:
    q = derive(a1, f"k15_a5_n{n}")
    q["pool_id"] = a1["pool_id"].replace("__k15", "").replace("a1_oldgold", f"a5_n{n}") + "__k15"
    order = sorted((c for c in a1["candidates"] if c["role"] == WARM), key=lambda c: (c["sim_to_history"], c["item_id"]))
    relabel = {c["item_id"] for c in order[:n]}
    for c in q["candidates"]:
        if c["item_id"] in relabel:
            c["date"] = rng_for("k15_a5_date", a1["instance_id"], c["item_id"]).choice(sorted(trap_dates))
            c["role"] = RELABELED
    return annotate(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pool_dir = split_dir(cfg) / "pools"
    flood_ns = cfg["adversarial"]["flood_ns"]
    stats: dict = {}

    def write(cond: str, neg: str, split: str, pools: list[dict]) -> None:
        write_jsonl(pool_dir / f"{cond}_{neg}_{split}.jsonl", pools)
        stats.setdefault(cond, {})[f"{neg}_{split}"] = manipulation_stats(cond, pools)

    for split in SPLITS:
        for neg, path in pool_files(pool_dir, "baseline", split).items():
            base = [shrink_baseline(p) for p in read_jsonl(path)]
            for p in base:
                check_common(p)
                assert sum(c["is_cold"] for c in p["candidates"]) == N_COLD_NEG + 1
            write("k15_baseline", neg, split, base)
            for cond, fn in DATE_VARIANTS.items():
                var = []
                for b in base:
                    q = fn(b)
                    check_date_variant(q, b)
                    q["condition"] = f"k15_{cond}"
                    var.append(q)
                write(f"k15_{cond}", neg, split, var)
        for neg, path in pool_files(pool_dir, "a6_allcold", split).items():
            a6 = [shrink_a6(p) for p in read_jsonl(path)]
            for p in a6:
                check_a6(p, K - 1)
            write("k15_a6_allcold", neg, split, a6)
        for neg, path in pool_files(pool_dir, "a1_oldgold", split).items():
            a1s, ctrls, a5s = [], [], {n: [] for n in flood_ns}
            for p in read_jsonl(path):
                a1, fill, trap_dates = shrink_a1(p)
                check_a1(a1, A1_WARM, A1_TRAPS)
                ctrl = a1ctrl_from(a1, fill)
                check_a1ctrl(ctrl, a1, A1_TRAPS)
                a1s.append(a1)
                ctrls.append(ctrl)
                for n in flood_ns:
                    q = a5_from(a1, n, trap_dates)
                    check_a5(q, a1, n)
                    a5s[n].append(q)
            write("k15_a1_oldgold", neg, split, a1s)
            write("k15_a1ctrl", neg, split, ctrls)
            for n in flood_ns:
                write(f"k15_a5_n{n}", neg, split, a5s[n])
    (pool_dir / "k15_stats.json").write_text(json.dumps(stats, indent=2))
    for cond in stats:
        print(f"[{cond}] " + ", ".join(f"{k}: {v['n_pools']}" for k, v in stats[cond].items()))


if __name__ == "__main__":
    main()
