"""Ranking aggregation and per-pool metrics.

Conventions (design.md):
  * A ranking is a list of item_ids, best first. It may be partial (a listwise
    model lists its top-N only) or empty (all shuffles failed to parse).
  * Recall@1 / NDCG@10 are computed over ALL pools; an empty ranking scores 0.
  * Cold@1 is computed over pools with a non-empty ranking (it describes what the
    model picks when it does pick); the parse-failure rate is reported beside it.
  * Ties are broken by a stable hash of (pool_id, item_id): pseudo-random but
    reproducible, and independent of stored candidate order.
"""

from __future__ import annotations

import math
from collections import defaultdict

from coldrank.common import stable_hash


def tie_key(pool_id: str, item_id: str) -> str:
    return stable_hash("tie", pool_id, item_id)


def borda(pool_id: str, rankings: list[list[str]], top_n: int) -> list[str]:
    """Combine several (partial) rankings. An item at 0-based position p of a list
    earns (top_n - p) points; unlisted items earn 0. Items with 0 points are left
    out, so a pool whose rankings are all empty yields an empty ranking."""
    pts: dict[str, float] = defaultdict(float)
    for r in rankings:
        for p, item in enumerate(r[:top_n]):
            pts[item] += top_n - p
    return sorted(pts, key=lambda i: (-pts[i], tie_key(pool_id, i)))


def sort_by_score(pool_id: str, scores: dict[str, float]) -> list[str]:
    """Full ranking from scores (higher is better), hash tie-break."""
    return sorted(scores, key=lambda i: (-scores[i], tie_key(pool_id, i)))


def pool_metrics(pool: dict, ranking: list[str], k: int = 10) -> dict:
    gold = pool["gold_item_id"]
    cold = {c["item_id"] for c in pool["candidates"] if c["is_cold"]}
    rank = ranking.index(gold) + 1 if gold in ranking else None
    return {
        "pool_id": pool["pool_id"],
        "gold_rank": rank,
        "recall1": float(rank == 1),
        "ndcg10": 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0,
        "empty": not ranking,
        "cold1": (ranking[0] in cold) if ranking else None,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    nonempty = [r for r in rows if not r["empty"]]
    return {
        "n": n,
        "recall1": sum(r["recall1"] for r in rows) / n if n else float("nan"),
        "ndcg10": sum(r["ndcg10"] for r in rows) / n if n else float("nan"),
        "cold1": sum(r["cold1"] for r in nonempty) / len(nonempty) if nonempty else float("nan"),
        "empty_rate": 1 - len(nonempty) / n if n else float("nan"),
    }
