"""Fine-tuning experiment: training cutoffs, training instances and K = 15 pools, SFT examples, batching."""

import numpy as np
import pandas as pd
import pytest

from coldrank.adversarial.common import COLD_NEG, GOLD, WARM
from coldrank.finetune.sft_data import is_holdout, pointwise_prompt, select_candidates
from coldrank.finetune.trainpools import check_cutoffs, k15_pools, neg_setting, training_instances
from coldrank.llm.chat import format_chat
from coldrank.pools.build import Embeddings, build_pools, check_pool
from coldrank.prompts.render import render_pointwise
from tests.test_split import T, _world

DAY = 86400


def test_check_cutoffs():
    cfg = {"cutoff_T": "2022-09-01", "window_days": 180,
           "finetune": {"cutoffs": ["2021-09-01", "2022-03-01"]}}
    assert check_cutoffs(cfg) == ["2021-09-01", "2022-03-01"]
    for bad in (["2022-03-01", "2021-09-01"],   # not in order
                ["2021-09-01", "2021-12-01"],   # windows overlap
                ["2022-03-10"]):                # window runs into the test window
        cfg["finetune"]["cutoffs"] = bad
        with pytest.raises(ValueError):
            check_cutoffs(cfg)


def test_configured_cutoffs_are_valid():
    from coldrank.common import load_config

    for ds in ("amazon_vg", "ebnerd"):
        cfg = load_config(ds)
        assert len(check_cutoffs(cfg)) == len(cfg["finetune"]["cutoffs"])
    load_config("amazon_vg")  # restore the date-only display format


def test_training_instances_exclude_eval_users_and_cap():
    items, pos = _world()
    cfg = {"dataset": "x", "seed": 42, "window_days": 30, "min_history": 5,
           "finetune": {"max_instances_per_cutoff": None}}
    from coldrank.common import ts_to_date

    cut = ts_to_date(T)
    inst, stats = training_instances(pos, items, cfg, cut, excluded=set())
    assert [r["user_id"] for r in inst] == ["u1"] and inst[0]["split"] == "train"
    assert inst[0]["instance_id"].startswith("train_")
    inst, stats = training_instances(pos, items, cfg, cut, excluded={"u1"})
    assert inst == [] and stats["n_cold_instances"] == 1 and stats["n_not_eval_users"] == 0


def _world50(n_inst=6):
    """A K = 50 world: 70 warm items, 30 cold items, n_inst cold-target users."""
    rng = np.random.default_rng(0)
    ids = [f"w{k:02d}" for k in range(70)] + [f"c{k:02d}" for k in range(30)]
    t0s = [T - (k + 1) * DAY for k in range(70)] + [T + (k + 1) * 3600 for k in range(30)]
    vecs = rng.normal(size=(100, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    items = pd.DataFrame({"item_id": ids, "title": [f"T {i}" for i in ids], "t0": t0s, "t0_source": "dfa",
                          "category_path": "PS4 > Games", "brand": "B"})
    inst, topk = [], {}
    for u in range(n_inst):
        iid = f"train_x_{u:05d}"
        inst.append({"instance_id": iid, "split": "train", "user_id": f"u{u}", "target_item": "c20",
                     "target_ts": T + 25 * 3600, "history": [[f"w{60 + k}", T - 100 + k] for k in range(5)],
                     "window_items": ["c20"]})
        topk[iid] = [f"w{k:02d}" for k in range(55)]
    return items, Embeddings(ids, vecs), inst, topk


def test_k15_pools_one_setting_per_instance():
    items, emb, inst, topk = _world50()
    cfg = {"pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.99,
                    "profile_len": 20}}
    pools50, _ = build_pools(inst, topk, items, emb, {}, T, cfg)
    assert len(pools50) == 2 * len(inst)
    pools = k15_pools(pools50)
    assert len(pools) == len(inst)
    for p in pools:
        check_pool(p, 12, 3)
        assert p["neg_setting"] == neg_setting(p["instance_id"]) and p["condition"] == "k15_train"
        assert sorted(c["warm_rank"] for c in p["candidates"] if not c["is_cold"]) == list(range(1, 13))
    assert k15_pools(pools50) == pools  # deterministic


def test_neg_setting_is_balanced():
    share = np.mean([neg_setting(f"train_x_{k:05d}") == "random" for k in range(2000)])
    assert 0.45 < share < 0.55


def test_select_candidates():
    items, emb, inst, topk = _world50(1)
    cfg = {"pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.99,
                    "profile_len": 20}}
    pool = k15_pools(build_pools(inst, topk, items, emb, {}, T, cfg)[0])[0]
    sel = select_candidates(pool, 3)
    roles = sorted(c["role"] for c in sel)
    assert roles == sorted([GOLD, COLD_NEG, COLD_NEG, WARM, WARM, WARM])
    assert select_candidates(pool, 3) == sel
    assert sum(c["is_gold"] for c in sel) == 1


def test_holdout_fraction():
    share = np.mean([is_holdout(f"user{k}", 0.05) for k in range(4000)])
    assert 0.035 < share < 0.065


def test_pointwise_prompt_matches_inference_and_trims():
    items, emb, inst, topk = _world50(1)
    cfg_pool = {"pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.99,
                         "profile_len": 20}}
    pool = k15_pools(build_pools(inst, topk, items, emb, {}, T, cfg_pool)[0])[0]
    cand = pool["candidates"][0]
    cfg = {"prompt": {"history_len": {"pointwise": 20}}}
    prompt, trimmed = pointwise_prompt(pool, cand, cfg, "llama2", len, 10**6)
    assert prompt == format_chat("llama2", *render_pointwise(pool, cand, 20)) and not trimmed
    full = len(prompt)
    prompt2, trimmed2 = pointwise_prompt(pool, cand, cfg, "llama2", len, full - 5)
    assert trimmed2 and len(prompt2) < full - 5


def test_collate_left_pads_with_real_positions():
    torch = pytest.importorskip("torch")
    from coldrank.finetune.train import collate

    b = collate([{"input_ids": [1, 5, 6], "label_id": 9}, {"input_ids": [1, 7], "label_id": 8}], pad_id=0)
    assert b["input_ids"].tolist() == [[1, 5, 6], [0, 1, 7]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert b["position_ids"].tolist() == [[0, 1, 2], [0, 0, 1]]
    assert b["labels"].tolist() == [9, 8]
    assert torch.is_tensor(b["labels"])
