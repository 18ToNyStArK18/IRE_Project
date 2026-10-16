"""Diagnostic: smaller pools (K = 20: 16 warm + gold + 3 cold, the same 4:1 ratio as 40 + 10).

Built from the baseline pools so the gold, history and warm order are unchanged: the 16
warm items with the best SASRec rank, the gold, and 3 of the 9 cold negatives (seeded).
Writes pools/baseline_k20_<neg>_<split>.jsonl (condition "baseline_k20").

    python scripts/make_k20_pools.py --neg random --split dev
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coldrank.common import load_config, read_jsonl, rng_for, write_jsonl  # noqa: E402
from coldrank.data.split import split_dir  # noqa: E402

N_WARM, N_COLD_NEG = 16, 3


def shrink(pool: dict) -> dict:
    q = copy.deepcopy(pool)
    warm = sorted((c for c in q["candidates"] if not c["is_cold"]), key=lambda c: c["warm_rank"])[:N_WARM]
    gold = [c for c in q["candidates"] if c["is_gold"]]
    negs = sorted((c for c in q["candidates"] if c["is_cold"] and not c["is_gold"]), key=lambda c: c["item_id"])
    negs = rng_for("k20", pool["pool_id"]).sample(negs, N_COLD_NEG)
    q["candidates"] = sorted(warm + gold + negs, key=lambda c: c["item_id"])
    q["condition"] = "baseline_k20"
    q["reference_pool_id"] = pool["pool_id"]
    q["pool_id"] = f"{pool['pool_id']}__k20"
    assert len(q["candidates"]) == N_WARM + 1 + N_COLD_NEG and sum(c["is_gold"] for c in q["candidates"]) == 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--neg", default="random")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    pool_dir = split_dir(load_config(args.config)) / "pools"
    pools = [shrink(p) for p in read_jsonl(pool_dir / f"baseline_{args.neg}_{args.split}.jsonl")]
    out = pool_dir / f"baseline_k20_{args.neg}_{args.split}.jsonl"
    write_jsonl(out, pools)
    print(f"{len(pools)} pools of {len(pools[0]['candidates'])} candidates -> {out}")


if __name__ == "__main__":
    main()
