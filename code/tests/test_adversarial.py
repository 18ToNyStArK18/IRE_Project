import copy

import numpy as np
import pandas as pd
import pytest

from coldrank.adversarial.allcold import build_a6
from coldrank.adversarial.checks import (
    check_a1, check_a1ctrl, check_a5, check_a6, check_date_variant, manipulation_stats,
)
from coldrank.adversarial.date_variants import DATE_VARIANTS, make_a2_swap
from coldrank.adversarial.oldgold import build_oldgold
from coldrank.baselines.heuristics import HEURISTICS
from coldrank.common import pool_files, ts_to_date
from coldrank.eval.metrics import pool_metrics
from coldrank.pools.build import Embeddings, build_pools
from coldrank.prompts.render import render_listwise, shuffled_candidates

DAY = 86400
T = 1_000 * DAY
CFG = {
    "pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.9, "profile_len": 20},
    "adversarial": {"a6_n_negs": 49, "a1_n_warm": 39, "a1_n_traps": 10, "trap_quantile": 0.25,
                    "ctrl_fill": 10, "flood_ns": [1, 2, 5, 10]},
}


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    ids, t0s = [], []
    for k in range(100):                       # warm w000..w099, launched before T
        ids.append(f"w{k:03d}"); t0s.append(T - (k + 1) * DAY)
    for k in range(120):                       # cold c000..c119, launched after T
        ids.append(f"c{k:03d}"); t0s.append(T + (k + 1) * DAY)
    vecs = rng.normal(size=(len(ids), 16))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    items = pd.DataFrame({"item_id": ids, "title": [f"Title {i}" for i in ids], "t0": t0s, "t0_source": "dfa",
                          "category_path": "PS4 > Games", "brand": "B"})
    emb = Embeddings(ids, vecs)
    hist = [[f"w{k:03d}", T - (200 - k) * DAY] for k in range(90, 96)]
    cold_inst = {"instance_id": "cold_test_00000", "split": "test", "user_id": "u1", "target_item": "c050",
                 "target_ts": T + 115 * DAY, "history": hist, "window_items": ["w080", "c050", "c007"]}
    old_inst = {"instance_id": "old_test_00000", "split": "test", "user_id": "u2", "target_item": "w010",
                "target_ts": T + 115 * DAY, "history": hist, "window_items": ["w010", "c003"]}
    order = [f"w{k:03d}" for k in range(100)]
    topk = {"cold_test_00000": order, "old_test_00000": order}
    base, _ = build_pools([cold_inst], topk, items, emb, {}, T, CFG)
    return dict(items=items, emb=emb, cold=cold_inst, old=old_inst, topk=topk, base=base)


def test_date_variants_change_only_dates(world):
    for base in world["base"]:
        for cond, fn in DATE_VARIANTS.items():
            q = fn(base)
            check_date_variant(q, base)                  # raises on any violation
            assert q == fn(copy.deepcopy(base))          # deterministic
            assert base.get("condition") == "baseline" and "role" not in base["candidates"][0]  # ref untouched
            # same prompt positions as the baseline for every shuffle
            for s in range(3):
                assert [c["item_id"] for c in shuffled_candidates(q, s)] == \
                       [c["item_id"] for c in shuffled_candidates(base, s)]


def test_a3_prompt_has_no_dates(world):
    q = DATE_VARIANTS["a3_hide"](world["base"][0])
    r = render_listwise(q, "direct", 0, 20, 10)
    assert "launch date" not in r.user and ts_to_date(T + 51 * DAY) not in r.user


def test_a6(world):
    pools, st = build_a6([world["cold"]], world["items"], world["emb"], {}, T, CFG)
    assert st["dropped_too_few_cold"] == 0 and {p["neg_setting"] for p in pools} == {"random", "hard"}
    for p in pools:
        check_a6(p, 49)
        assert p["reference_pool_id"] == f"cold_test_00000__{p['neg_setting']}"
        assert not {"c007", "w080"} & {c["item_id"] for c in p["candidates"]}  # the user's other window items
        # only items available at the target time (launched by T+115 days)
        assert all(c["t0"] <= world["cold"]["target_ts"] for c in p["candidates"])



def test_oldgold_family(world):
    fam, st = build_oldgold([world["old"]], world["topk"], world["items"], world["emb"], {}, T, CFG)
    assert st["dropped"] == {"too_few_warm": 0, "too_few_traps": 0}
    a1 = fam["a1_oldgold"][0]
    check_a1(a1, 39, 10)
    check_a1ctrl(fam["a1ctrl"][0], a1, 10)
    prev = set()
    for n in (1, 2, 5, 10):
        q = fam[f"a5_n{n}"][0]
        check_a5(q, a1, n)
        rel = {c["item_id"] for c in q["candidates"] if c["role"] == "relabeled"}
        assert prev <= rel                     # nested in n
        prev = rel
    # gold is the warm target, inserted although SASRec ranks it 11th
    gold = next(c for c in a1["candidates"] if c["is_gold"])
    assert gold["item_id"] == "w010" and gold["warm_rank"] == 11
    # A1-ctrl / A5 keep A1's prompt positions for shared items
    for s in range(3):
        pos_a1 = {c["item_id"]: k for k, c in enumerate(shuffled_candidates(a1, s))}
        for q in (fam["a1ctrl"][0], fam["a5_n5"][0]):
            for k, c in enumerate(shuffled_candidates(q, s)):
                if c["item_id"] in pos_a1:
                    assert pos_a1[c["item_id"]] == k
    st = manipulation_stats("a1_oldgold", fam["a1_oldgold"])
    assert st["trap_minus_gold_age_days_median"] > 0 and st["new_dated_per_pool"] == 10


def test_metrics_cold_and_trap(world):
    fam, _ = build_oldgold([world["old"]], world["topk"], world["items"], world["emb"], {}, T, CFG)
    a1, ctrl = fam["a1_oldgold"][0], fam["a1ctrl"][0]
    trap = next(c["item_id"] for c in a1["candidates"] if c["role"] == "trap")
    m = pool_metrics(a1, [trap, a1["gold_item_id"]])
    assert m["trap1"] is True and m["cold1"] is True and m["recall1"] == 0
    m = pool_metrics(a1, [a1["gold_item_id"]])
    assert m["trap1"] is False and m["cold1"] is False and m["recall1"] == 1
    fill = next(c["item_id"] for c in ctrl["candidates"] if c["role"] == "ctrl_fill")
    m = pool_metrics(ctrl, [fill])
    assert m["trap1"] is True and m["cold1"] is False
    base = world["base"][0]
    assert pool_metrics(base, [base["gold_item_id"]])["trap1"] is None


def test_heuristics_react_to_variants(world):
    base = world["base"][0]
    gold = base["gold_item_id"]
    q = make_a2_swap(base)
    assert HEURISTICS["newest_first"](q)[0] == q["swap_partner"] or \
        HEURISTICS["newest_first"](q)[0] != gold          # the gold never looks newest after the swap
    from coldrank.baselines.heuristics import newest_set
    assert gold in newest_set(base) and gold not in newest_set(q) and q["swap_partner"] in newest_set(q)
    a3 = DATE_VARIANTS["a3_hide"](base)
    assert newest_set(a3) == set()


def test_pool_files_prefix(tmp_path):
    for name in ("a5_n1_trap_dev", "a5_n10_trap_dev", "baseline_random_dev", "baseline_hard_dev"):
        (tmp_path / f"{name}.jsonl").write_text("")
    assert set(pool_files(tmp_path, "a5_n1", "dev")) == {"trap"}
    assert set(pool_files(tmp_path, "baseline", "dev")) == {"random", "hard"}
