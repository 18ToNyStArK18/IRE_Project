"""A2 date swap, A3 date hide, A4 date shuffle: rewrite displayed dates of a
baseline pool. Candidates, true launch times, history and order are unchanged."""

from __future__ import annotations

from coldrank.adversarial.common import GOLD, WARM, annotate, derive
from coldrank.common import rng_for


def make_a2_swap(base: dict) -> dict:
    """Exchange the displayed dates of the gold and one uniformly random warm item."""
    q = derive(base, "a2_swap")
    warm = sorted(c["item_id"] for c in q["candidates"] if c["role"] == WARM)
    partner = rng_for("a2_partner", base["pool_id"]).choice(warm)
    by_id = {c["item_id"]: c for c in q["candidates"]}
    gold = next(c for c in q["candidates"] if c["role"] == GOLD)
    gold["date"], by_id[partner]["date"] = by_id[partner]["date"], gold["date"]
    q["swap_partner"] = partner
    return annotate(q)


def make_a3_hide(base: dict) -> dict:
    """Remove displayed dates from every candidate and every history item."""
    q = derive(base, "a3_hide")
    for c in q["candidates"]:
        c["date"] = None
    for h in q["history"]:
        h["date"] = None
    return annotate(q)


def make_a4_shuffle(base: dict) -> dict:
    """Randomly permute the displayed dates across all candidates (history unchanged)."""
    q = derive(base, "a4_shuffle")
    dates = [c["date"] for c in q["candidates"]]
    rng_for("a4_perm", base["pool_id"]).shuffle(dates)
    for c, d in zip(q["candidates"], dates):
        c["date"] = d
    return annotate(q)


DATE_VARIANTS = {"a2_swap": make_a2_swap, "a3_hide": make_a3_hide, "a4_shuffle": make_a4_shuffle}
