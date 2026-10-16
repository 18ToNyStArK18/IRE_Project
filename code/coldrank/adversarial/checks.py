"""Invariants for every variant pool (plan_adversarial.md Sec. 3) and the
manipulation-check statistics reported in the paper."""

from __future__ import annotations

import numpy as np

from coldrank.adversarial.common import CTRL_FILL, GOLD, RELABELED, TRAP, WARM
from coldrank.common import date_to_ts

N_CANDS = 50


def _ids(p: dict) -> list[str]:
    return [c["item_id"] for c in p["candidates"]]


def _strip(c: dict, *keys: str) -> dict:
    return {k: v for k, v in c.items() if k not in keys}


def check_common(p: dict) -> None:
    ids = _ids(p)
    T = date_to_ts(p["cutoff_T"])
    assert len(ids) == p.get("pool_size", N_CANDS) == len(set(ids)), p["pool_id"]  # 50, or 15 for k15_* pools
    golds = [c for c in p["candidates"] if c["is_gold"]]
    assert len(golds) == 1 and golds[0]["item_id"] == p["gold_item_id"] and golds[0]["role"] == GOLD, p["pool_id"]
    assert not set(ids) & {h["item_id"] for h in p["history"]}, p["pool_id"]
    assert all(c["t0"] <= p["target_ts"] for c in p["candidates"]), p["pool_id"]
    assert all(h["ts"] < T for h in p["history"]), p["pool_id"]
    for c in p["candidates"]:
        assert c["new_dated"] == (c["date"] is not None and date_to_ts(c["date"]) > T), p["pool_id"]


def check_date_variant(p: dict, ref: dict) -> None:
    """A2/A3/A4 against their baseline pool: same candidates, order, true t0 and every
    non-date field; only displayed dates (and the derived new_dated flag) change."""
    check_common(p)
    T = date_to_ts(p["cutoff_T"])
    assert p["shuffle_key"] == ref["pool_id"] and _ids(p) == _ids(ref), p["pool_id"]
    for a, b in zip(p["candidates"], ref["candidates"]):
        assert _strip(a, "date", "new_dated", "role") == _strip(b, "date", "new_dated", "role"), p["pool_id"]
    cond = p["condition"]
    if cond == "a2_swap":
        changed = {a["item_id"] for a, b in zip(p["candidates"], ref["candidates"]) if a["date"] != b["date"]}
        assert changed == {p["gold_item_id"], p["swap_partner"]}, p["pool_id"]
        gold = next(c for c in p["candidates"] if c["is_gold"])
        partner = next(c for c in p["candidates"] if c["item_id"] == p["swap_partner"])
        ref_gold = next(c for c in ref["candidates"] if c["is_gold"])
        assert partner["role"] == WARM and partner["date"] == ref_gold["date"], p["pool_id"]
        assert date_to_ts(gold["date"]) <= T and partner["new_dated"], p["pool_id"]
    elif cond == "a3_hide":
        assert all(c["date"] is None for c in p["candidates"]), p["pool_id"]
        assert all(h["date"] is None for h in p["history"]), p["pool_id"]
        assert [_strip(h, "date") for h in p["history"]] == [_strip(h, "date") for h in ref["history"]]
    elif cond == "a4_shuffle":
        assert sorted(c["date"] for c in p["candidates"]) == sorted(c["date"] for c in ref["candidates"])
        assert p["history"] == ref["history"], p["pool_id"]
    else:
        raise ValueError(cond)


def check_a6(p: dict, n_negs: int) -> None:
    check_common(p)
    T = date_to_ts(p["cutoff_T"])
    assert all(c["t0"] > T and c["is_cold"] for c in p["candidates"]), p["pool_id"]
    assert sum(c["role"] == "cold_neg" for c in p["candidates"]) == n_negs, p["pool_id"]


def check_a1(p: dict, n_warm: int, n_traps: int) -> None:
    check_common(p)
    T = date_to_ts(p["cutoff_T"])
    gold = next(c for c in p["candidates"] if c["is_gold"])
    assert gold["t0"] <= T and not gold["new_dated"], p["pool_id"]
    traps = [c for c in p["candidates"] if c["role"] == TRAP]
    warm = [c for c in p["candidates"] if c["role"] == WARM]
    assert len(traps) == n_traps and len(warm) == n_warm, p["pool_id"]
    assert all(c["t0"] > T and c["new_dated"] for c in traps), p["pool_id"]
    assert all(c["sim_to_history"] <= p["trap_sim_threshold"] + 1e-4 for c in traps), p["pool_id"]
    assert all(c["t0"] <= T and not c["new_dated"] for c in warm), p["pool_id"]


def check_a1ctrl(p: dict, a1: dict, n_fill: int) -> None:
    check_common(p)
    assert p["shuffle_key"] == a1["pool_id"], p["pool_id"]
    assert not any(c["new_dated"] for c in p["candidates"]), p["pool_id"]
    assert sum(c["role"] == CTRL_FILL for c in p["candidates"]) == n_fill, p["pool_id"]
    for a, b in zip(p["candidates"], a1["candidates"]):
        if b["role"] == TRAP:
            assert a["role"] == CTRL_FILL, p["pool_id"]
        else:
            assert a == b, p["pool_id"]


def check_a5(p: dict, a1: dict, n: int) -> None:
    check_common(p)
    T = date_to_ts(p["cutoff_T"])
    assert p["shuffle_key"] == a1["pool_id"] and _ids(p) == _ids(a1), p["pool_id"]
    relabeled = [c for c in p["candidates"] if c["role"] == RELABELED]
    n_traps = sum(c["role"] == TRAP for c in a1["candidates"])
    assert len(relabeled) == n and sum(c["new_dated"] for c in p["candidates"]) == n_traps + n, p["pool_id"]
    least_similar = sorted((c for c in a1["candidates"] if c["role"] == WARM),
                           key=lambda c: (c["sim_to_history"], c["item_id"]))[:n]
    assert {c["item_id"] for c in relabeled} == {c["item_id"] for c in least_similar}, p["pool_id"]
    assert all(T < date_to_ts(c["date"]) <= p["target_ts"] and not c["is_cold"] for c in relabeled), p["pool_id"]
    for a, b in zip(p["candidates"], a1["candidates"]):
        if a["role"] != RELABELED:
            assert a == b, p["pool_id"]


# --------------------------------------------------------------------------- #
# Manipulation checks
# --------------------------------------------------------------------------- #
def manipulation_stats(cond: str, pools: list[dict]) -> dict:
    out: dict = {"n_pools": len(pools)}
    if not pools:
        return out
    gold = [next(c for c in p["candidates"] if c["is_gold"]) for p in pools]
    out["sim_gold_mean"] = float(np.mean([g["sim_to_history"] for g in gold]))
    out["new_dated_per_pool"] = float(np.mean([sum(c["new_dated"] for c in p["candidates"]) for p in pools]))
    out["gold_new_dated_frac"] = float(np.mean([g["new_dated"] for g in gold]))

    def role_sim(role: str) -> float:
        v = [c["sim_to_history"] for p in pools for c in p["candidates"] if c["role"] == role]
        return float(np.mean(v)) if v else float("nan")

    for role in ("warm", "cold_neg", TRAP, CTRL_FILL, RELABELED):
        s = role_sim(role)
        if s == s:
            out[f"sim_{role}_mean"] = s
    if cond == "a2_swap":
        out["gold_displayed_minus_true_days_median"] = float(np.median(
            [(date_to_ts(g["date"]) - g["t0"]) / 86400 for g in gold]))
    if cond in ("a1_oldgold",) or cond.startswith("a5_"):
        gaps = [np.median([c["t0"] for c in p["candidates"] if c["role"] == TRAP]) - g["t0"]
                for p, g in zip(pools, gold)]
        out["trap_minus_gold_age_days_median"] = float(np.median(gaps) / 86400)
    return out
