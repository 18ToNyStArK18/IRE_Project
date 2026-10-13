"""Reference heuristics (non-LLM) on the same pools (design.md Sec. 3).

They read only what the pool exposes. Date-based heuristics use the DISPLAYED date
(`date`), so under date variants they see exactly what the LLM sees; with hidden
dates (None) they fall back to the hash order.

Usage:
    python -m coldrank.baselines.heuristics --config amazon_vg --split dev
"""

from __future__ import annotations

import argparse

from coldrank.common import OUT_DIR, date_to_ts, load_config, pool_files, read_jsonl, write_jsonl
from coldrank.data.split import split_dir
from coldrank.eval.metrics import sort_by_score, tie_key

N_NEWEST = 10


def _date_score(c: dict) -> float:
    return float(date_to_ts(c["date"])) if c.get("date") else float("-inf")


def newest_set(pool: dict, n: int = N_NEWEST) -> set[str]:
    cs = [c for c in pool["candidates"] if c.get("date")]
    order = sorted(cs, key=lambda c: (-_date_score(c), tie_key(pool["pool_id"], c["item_id"])))
    return {c["item_id"] for c in order[:n]}


def rank_random(pool: dict) -> list[str]:
    return sort_by_score(pool["pool_id"], {c["item_id"]: 0.0 for c in pool["candidates"]})


def rank_random_among_newest(pool: dict) -> list[str]:
    new = newest_set(pool)
    return sort_by_score(pool["pool_id"], {c["item_id"]: float(c["item_id"] in new) for c in pool["candidates"]})


def rank_newest_first(pool: dict) -> list[str]:
    return sort_by_score(pool["pool_id"], {c["item_id"]: _date_score(c) for c in pool["candidates"]})


def rank_sasrec(pool: dict) -> list[str]:
    # warm items in SASRec order, then cold items (SASRec cannot score them) in hash order
    return sort_by_score(pool["pool_id"], {
        c["item_id"]: (-c["warm_rank"] if c.get("warm_rank") else float("-inf")) for c in pool["candidates"]
    })


def rank_popularity(pool: dict) -> list[str]:
    return sort_by_score(pool["pool_id"], {c["item_id"]: float(c["pop_preT"]) for c in pool["candidates"]})


def rank_content(pool: dict) -> list[str]:
    return sort_by_score(pool["pool_id"], {c["item_id"]: c["sim_to_history"] for c in pool["candidates"]})


def rank_newest_filter_content(pool: dict) -> list[str]:
    """Cold-filter + content: the 10 newest (displayed) candidates by content
    similarity, then the rest by content similarity."""
    new = newest_set(pool)
    return sort_by_score(pool["pool_id"], {
        c["item_id"]: (1.0 if c["item_id"] in new else 0.0) * 10 + c["sim_to_history"] for c in pool["candidates"]
    })


HEURISTICS = {
    "random": rank_random,
    "random_among_newest": rank_random_among_newest,
    "newest_first": rank_newest_first,
    "sasrec": rank_sasrec,
    "popularity": rank_popularity,
    "content": rank_content,
    "newest_filter_content": rank_newest_filter_content,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--condition", default="baseline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    files = pool_files(split_dir(cfg) / "pools", args.condition, args.split)
    if not files:
        raise SystemExit(f"no pool files for condition={args.condition} split={args.split}")
    for neg, path in files.items():
        pools = list(read_jsonl(path))
        for name, fn in HEURISTICS.items():
            preds = [{"pool_id": p["pool_id"], "ranking": fn(p)} for p in pools]
            out = OUT_DIR / cfg["dataset"] / args.condition / neg / "heuristic" / name / args.split / "preds.jsonl"
            write_jsonl(out, preds)
        print(f"[heuristics] {neg}/{args.split}: {len(pools)} pools x {len(HEURISTICS)} heuristics")


if __name__ == "__main__":
    main()
