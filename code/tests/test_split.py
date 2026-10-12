import numpy as np
import pandas as pd
import pytest

from coldrank.data.amazon_vg import category_path, parse_amazon_date
from coldrank.data.audit import count_instances
from coldrank.data.split import (
    assign_ids_and_subsample,
    attach_launch_times,
    build_instances,
    positives,
    usable_items,
)

DAY = 86400
T = 1_000 * DAY


def _items(rows):
    df = pd.DataFrame(rows, columns=["item_id", "title", "dfa_ts"])
    df["dfa_ts"] = df["dfa_ts"].astype("Int64")
    return df


def _inter(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "ts"])


def test_parse_amazon_date():
    assert parse_amazon_date("October 2, 2001") == int(pd.Timestamp("2001-10-02", tz="UTC").timestamp())
    assert parse_amazon_date("Oct 2, 2001") == parse_amazon_date("October 2, 2001")
    assert parse_amazon_date("04M 26, 2018") == parse_amazon_date("April 26, 2018")
    assert parse_amazon_date("garbage") is None
    assert parse_amazon_date(None) is None


def test_category_path():
    assert category_path(["Video Games", "PlayStation 4", "Games"]) == "PlayStation 4 > Games"
    assert category_path([]) == ""
    assert category_path(None) == ""


def test_launch_time_is_min_of_dfa_and_first_review():
    items = _items([("a", "A", 50 * DAY), ("b", "B", 500 * DAY), ("c", "C", None), ("d", "D", 7 * DAY)])
    inter = _inter([("u", "a", 5, 100 * DAY), ("u", "b", 5, 200 * DAY), ("u", "c", 1, 300 * DAY)])
    out = attach_launch_times(items, inter).set_index("item_id")
    assert out.loc["a", "t0"] == 50 * DAY and out.loc["a", "t0_source"] == "dfa"
    assert out.loc["b", "t0"] == 200 * DAY and out.loc["b", "t0_source"] == "first_review"  # DFA after review
    assert out.loc["c", "t0"] == 300 * DAY and out.loc["c", "t0_source"] == "first_review"  # no DFA; low rating still counts
    assert out.loc["d", "t0"] == 7 * DAY and out.loc["d", "t0_source"] == "dfa"  # never reviewed


def test_positives_threshold_and_dedup():
    items = usable_items(attach_launch_times(_items([("a", "A", 1), ("b", "B", 1), ("x", "", 1)]), _inter([])))
    inter = _inter([
        ("u", "a", 5, 10), ("u", "a", 4, 5),   # duplicate: keep earliest (ts=5)
        ("u", "b", 2, 7),                       # below threshold
        ("u", "x", 5, 8),                       # untitled item -> unusable
    ])
    p = positives(inter, items, 3.0)
    assert p[["item_id", "ts"]].values.tolist() == [["a", 5]]


def _world():
    """Synthetic world around cutoff T."""
    # warm items w0..w9 launched long before T; cold c0..c4 launched after T
    rows = [(f"w{i}", f"Warm {i}", 10 * DAY) for i in range(10)]
    rows += [(f"c{i}", f"Cold {i}", T + (i + 1) * DAY) for i in range(5)]
    items = _items(rows)
    inter = []
    # u1: 6 pre-T positives, then warm-new w9 on day +2, cold c0 on day +3, cold c1 on day +4
    inter += [("u1", f"w{i}", 5, T - (10 - i) * DAY) for i in range(6)]
    inter += [("u1", "w9", 5, T + 2 * DAY), ("u1", "c0", 5, T + 3 * DAY), ("u1", "c1", 5, T + 4 * DAY)]
    # u2: only 3 pre-T positives -> not eligible
    inter += [("u2", f"w{i}", 5, T - 5 * DAY + i) for i in range(3)] + [("u2", "c2", 5, T + 5 * DAY)]
    # u3: 5 pre-T positives; cold c3 only on day +40 (outside w=30); low-rated cold c4 inside
    inter += [("u3", f"w{i}", 5, T - 20 * DAY + i) for i in range(5)]
    inter += [("u3", "c4", 1, T + 6 * DAY), ("u3", "c3", 5, T + 40 * DAY)]
    # u4: re-buys a history item after T (not an old target), then nothing else
    inter += [("u4", f"w{i}", 5, T - 30 * DAY + i) for i in range(5)] + [("u4", "w0", 5, T + 1 * DAY)]
    # other reviewers making w9 warm (pre-T positive by someone)
    inter += [("u5", "w9", 5, T - DAY)]
    inter = _inter(inter)
    items = usable_items(attach_launch_times(items, inter))
    return items, positives(inter, items, 3.0)


def test_build_instances_world():
    items, pos = _world()
    res = build_instances(pos, items, T, 30, 5)
    cold = {r["user_id"]: r for r in res.cold_instances}
    old = {r["user_id"]: r for r in res.old_instances}
    assert set(cold) == {"u1"}                      # u2 ineligible, u3 outside window / low rating
    assert cold["u1"]["target_item"] == "c0"        # first cold, not c1
    assert [h[0] for h in cold["u1"]["history"]] == [f"w{i}" for i in range(6)]  # chronological, pre-T only
    assert cold["u1"]["window_items"] == ["w9", "c0", "c1"]
    assert old["u1"]["target_item"] == "w9"         # warm, not in history
    assert "u4" not in old                           # re-buy of a history item is not an old target
    for r in res.cold_instances:
        assert all(ts < T for _, ts in r["history"])
        assert r["target_ts"] >= T


def test_audit_counts_match_split():
    items, pos = _world()
    for w in (3, 4, 30, 60):
        n_split = build_instances(pos, items, T, w, 5).stats["n_cold_instances"]
        n_audit = count_instances(pos, items, T, [w], 5)[0]["cold_instances"]
        assert n_split == n_audit, w


def test_audit_counts_match_split_random():
    rng = np.random.default_rng(0)
    n_items, n_users = 60, 300
    dfa = rng.integers(0, 2 * T, n_items)
    items = _items([(f"i{k}", f"I{k}", int(dfa[k]) if rng.random() < 0.8 else None) for k in range(n_items)])
    inter = _inter([
        (f"u{rng.integers(n_users)}", f"i{rng.integers(n_items)}", int(rng.integers(1, 6)), int(rng.integers(0, 2 * T)))
        for _ in range(6000)
    ])
    items = usable_items(attach_launch_times(items, inter))
    pos = positives(inter, items, 3.0)
    for w in (5, 30, 200):
        res = build_instances(pos, items, T, w, 5)
        assert res.stats["n_cold_instances"] == count_instances(pos, items, T, [w], 5)[0]["cold_instances"]
        t0 = items.set_index("item_id")["t0"]
        for r in res.cold_instances:
            assert t0[r["target_item"]] > T
            assert all(t0[i] <= T for i, _ in r["history"])


def test_subsample_disjoint_and_deterministic():
    inst = [{"user_id": f"u{i}"} for i in range(50)]
    d1, t1 = assign_ids_and_subsample([dict(r) for r in inst], "cold", 10, 20, 42)
    d2, t2 = assign_ids_and_subsample([dict(r) for r in inst], "cold", 10, 20, 42)
    assert [r["user_id"] for r in d1] == [r["user_id"] for r in d2]
    assert not ({r["user_id"] for r in d1} & {r["user_id"] for r in t1})
    assert len(d1) == 10 and len(t1) == 20
    assert t1[0]["instance_id"] == "cold_test_00000"
