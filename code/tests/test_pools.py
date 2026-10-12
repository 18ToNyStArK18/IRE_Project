import numpy as np
import pandas as pd
import pytest

from coldrank.pools.build import Embeddings, build_pools, check_pool, pool_summary

DAY = 86400
T = 1_000 * DAY
CFG = {"pool": {"n_warm": 4, "n_cold": 3, "neg_settings": ["random", "hard"], "dup_threshold": 0.9, "profile_len": 20}}


def _setup():
    ids, t0s, vecs = [], [], []
    rng = np.random.default_rng(0)
    for k in range(12):  # warm w00..w11
        ids.append(f"w{k:02d}"); t0s.append(T - (k + 1) * DAY); vecs.append(rng.normal(size=8))
    for k in range(8):   # cold c0..c7 launched T+1..T+8 days
        ids.append(f"c{k}"); t0s.append(T + (k + 1) * DAY); vecs.append(rng.normal(size=8))
    vecs = np.array(vecs)
    vecs[ids.index("c6")] = vecs[ids.index("c5")] + 1e-3  # c6 is a near-duplicate of gold c5
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    items = pd.DataFrame({"item_id": ids, "title": [f"T {i}" for i in ids], "t0": t0s,
                          "t0_source": "dfa", "category_path": "PS4 > Games", "brand": "B"})
    emb = Embeddings(ids, vecs)
    inst = {
        "instance_id": "cold_test_00000", "split": "test", "user_id": "u",
        "target_item": "c5", "target_ts": T + 7 * DAY,  # c7 (launched T+8) is not yet available
        "history": [["w00", T - 100], ["w01", T - 50]],
        "window_items": ["w05", "c5", "c2"],  # w05 and c2 are other window positives
    }
    topk = {"cold_test_00000": ["w02", "w05", "w03", "w04", "w06", "w07", "w08"]}
    return items, emb, inst, topk


def test_pool_invariants_and_exclusions():
    items, emb, inst, topk = _setup()
    pools, stats = build_pools([inst], topk, items, emb, {}, T, CFG)
    assert len(pools) == 2
    for p in pools:
        check_pool(p, 4, 3)
        ids = {c["item_id"] for c in p["candidates"]}
        assert {"w02", "w03", "w04", "w06"} <= ids          # w05 skipped (window positive)
        assert not ids & {"w05", "c2", "c6", "c7"}           # window item, near-dup of gold, not yet launched
        assert [c["item_id"] for c in p["candidates"]] == sorted(ids)  # neutral order
    hard = next(p for p in pools if p["neg_setting"] == "hard")
    elig = ["c0", "c1", "c3", "c4"]
    prof = emb[["w00", "w01"]].mean(0); prof /= np.linalg.norm(prof)
    expect = sorted(elig, key=lambda i: -(emb[[i]][0] @ prof))[:2]
    got = sorted(c["item_id"] for c in hard["candidates"] if c["is_cold"] and not c["is_gold"])
    assert got == sorted(expect)
    assert pool_summary(pools)["hard"]["n_pools"] == 1


def test_random_negatives_deterministic():
    items, emb, inst, topk = _setup()
    a, _ = build_pools([inst], topk, items, emb, {}, T, CFG)
    b, _ = build_pools([inst], topk, items, emb, {}, T, CFG)
    assert a == b


def test_drops_when_too_few_cold():
    items, emb, inst, topk = _setup()
    inst = dict(inst, target_ts=T + 2 * DAY, target_item="c1")  # only c0 available -> < 2 negatives
    pools, stats = build_pools([inst], topk, items, emb, {}, T, CFG)
    assert pools == [] and stats["dropped"]["too_few_cold"] == 1


def test_check_pool_catches_leak():
    items, emb, inst, topk = _setup()
    p = build_pools([inst], topk, items, emb, {}, T, CFG)[0][0]
    p["history"].append({"item_id": p["candidates"][0]["item_id"], "ts": T - 1})
    with pytest.raises(AssertionError):
        check_pool(p, 4, 3)
