"""Fine-tuning experiment: training cutoffs, training instances and K = 15 pools, SFT examples, batching."""

import numpy as np
import pandas as pd
import pytest

from coldrank.adversarial.common import COLD_NEG, GOLD, WARM
from coldrank.adversarial.date_variants import make_a3_hide
from coldrank.common import load_config
from coldrank.finetune.impressions import IMP_NEG, attach_negatives, date_cue, imp_cfg
from coldrank.finetune.impressions import build_examples as build_imp_examples
from coldrank.finetune import merge as merge_mod
from coldrank.finetune.oldmix import check_old_pool, k15_old_pools, mix_cutoff, old_training_instances, oldonly_cutoff, oldonly_quotas, strip_new
from coldrank.finetune.sft_data import hide_dates, is_holdout, pointwise_prompt, select_candidates, sft_dir, sft_paths
from coldrank.finetune.trainpools import check_cutoffs, k15_pools, neg_setting, training_instances
from coldrank.llm.chat import format_chat
from coldrank.pools.build import Embeddings, build_pools, candidate_record, check_pool, history_records
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


def test_answer_text_follows_the_chat_format():
    pytest.importorskip("torch")
    from coldrank.finetune.train import answer_text

    assert [answer_text("llama2", lab) for lab in ("Yes", "No")] == [" Yes", " No"]   # after "[/INST]"
    assert [answer_text("chatml", lab) for lab in ("Yes", "No")] == ["Yes", "No"]     # after "assistant\n"


def test_sft_paths_per_model(tmp_path):
    assert [p.name for p in sft_paths(tmp_path, "llama2_7b_fp16")] == ["sft_train.jsonl", "sft_val.jsonl", "sft_stats.json"]
    qwen = sft_paths(tmp_path, "qwen25_7b")
    assert [p.name for p in qwen] == ["sft_train_qwen25_7b.jsonl", "sft_val_qwen25_7b.jsonl", "sft_stats_qwen25_7b.json"]
    assert not set(qwen) & set(sft_paths(tmp_path, "llama2_7b_fp16"))


def test_qwen_finetune_configs_consistent():
    models, fcfg = load_config("models"), load_config("finetune_qwen25_ada")
    base, ft = models[fcfg["base_model"]], models["qwen25_7b_ft_ebnerd"]
    assert fcfg["base_model"] == ft["base"] == "qwen25_7b_fp16"
    # fp16 weights and compute (2080 Ti), evaluated 4-bit like the Amazon Qwen run
    assert fcfg["quant"]["compute_dtype"] == "float16" and fcfg["quant"]["keep_embeddings_fp16"]
    assert "dtype" not in base and "dtype" not in ft
    assert base["quantization"] == ft["quantization"] == models["qwen25_7b_ft_amazon_vg"]["quantization"] == "bitsandbytes"
    assert base["family"] == ft["family"] == "chatml"
    # train.py writes outputs/finetune/<dataset>/<base without _fp16>/adapter
    assert ft["adapter"] == f"outputs/finetune/{load_config('ebnerd')['dataset']}/qwen25_7b/adapter"
    # same hyperparameters as the Amazon Qwen run and as Llama-2; only the batch split differs
    one = load_config("finetune_qwen25_1gpu")
    for k in ("epochs", "lr", "scheduler", "warmup_ratio", "weight_decay", "max_grad_norm", "seed"):
        assert fcfg["train"][k] == one["train"][k] == load_config("finetune")["train"][k], k
    assert fcfg["lora"] == one["lora"] and fcfg["quant"] == one["quant"]
    # the Llama-2 fine-tuning config is unchanged
    assert load_config("finetune")["base_model"] == "llama2_7b_fp16"
    assert load_config("finetune")["quant"]["compute_dtype"] == "float16"


def test_sft_paths_variant_never_collides(tmp_path):
    imp = sft_paths(tmp_path, "llama2_7b_fp16", "imp")
    assert [p.name for p in imp] == ["sft_train_llama2_7b_fp16_imp.jsonl", "sft_val_llama2_7b_fp16_imp.jsonl",
                                     "sft_stats_llama2_7b_fp16_imp.json"]
    assert not set(imp) & set(sft_paths(tmp_path, "llama2_7b_fp16"))
    assert sft_paths(tmp_path, "qwen25_7b", "imp")[0].name == "sft_train_qwen25_7b_imp.jsonl"
    with pytest.raises(ValueError):
        sft_dir({"dataset": "x", "cutoff_T": "2023-05-25", "window_days": 1}, "bogus")


def _imp_world():
    """_world50 plus impressions: the gold c20 clicked at the target time next to shown articles."""
    items, emb, inst, topk = _world50(3)
    r = inst[0]
    ts = r["target_ts"]
    imp = pd.DataFrame([
        # the gold's impression: c21 also clicked (a positive), w60 in the history, zz unknown,
        # c25 launched after the target time (T + 26 h > T + 25 h)
        {"user_id": "u0", "ts": ts, "inview": ["c20", "c21", "c22", "c25", "w00", "w60", "zz"], "clicked": ["c20", "c21"]},
        # another impression of u0 at another time: never a source of negatives
        {"user_id": "u0", "ts": ts - 600, "inview": ["c23", "c24"], "clicked": []},
        # u1 clicked something else at the target time: no gold impression
        {"user_id": "u1", "ts": ts, "inview": ["c20", "c26"], "clicked": ["c26"]},
        # u2: the gold's impression shows nothing else
        {"user_id": "u2", "ts": ts, "inview": ["c20"], "clicked": ["c20"]},
    ])
    return items, inst, imp


def test_impression_negatives_come_from_the_gold_impression_only():
    items, inst, imp = _imp_world()
    t0 = items.set_index("item_id")["t0"].astype("int64").to_dict()
    kept, drops = attach_negatives(inst, imp, t0, n_neg=5, min_neg=1)
    assert [r["user_id"] for r in kept] == ["u0"]
    assert drops == {"no_gold_impression": 1, "no_negatives": 1, "too_few_negatives": 0}
    # c20 gold, c21 clicked, w60 history, zz unknown, c25 launched after the target time,
    # c23/c24 only in another impression: all excluded
    assert t0["c25"] > inst[0]["target_ts"] >= t0["c22"]
    assert sorted(kept[0]["imp_negatives"]) == ["c22", "w00"]
    # capped and seeded
    small, _ = attach_negatives(inst, imp, t0, n_neg=1, min_neg=1)
    assert len(small[0]["imp_negatives"]) == 1 and small == attach_negatives(inst, imp, t0, n_neg=1, min_neg=1)[0]
    assert attach_negatives(inst, imp, t0, n_neg=5, min_neg=3)[1]["too_few_negatives"] == 1


def test_impression_cap_applies_after_the_impression_filter(monkeypatch):
    import coldrank.finetune.impressions as impressions

    items, inst, imp = _imp_world()
    t0 = items.set_index("item_id")["t0"].astype("int64").to_dict()
    seen = {}

    def fake_instances(pos, items_, cfg, cutoff, excluded):  # seeded order: u1, u2 (no negative), u0
        seen["cap"] = cfg["finetune"]["max_instances_per_cutoff"]
        return [dict(r) for r in (inst[1], inst[2], inst[0])], {"n_instances": 3}

    monkeypatch.setattr(impressions, "training_instances", fake_instances)
    icfg = {"finetune": {"max_instances_per_cutoff": 1, "n_neg": 5, "min_neg": 1}}
    kept, st = impressions.cutoff_instances(None, items, imp, t0, icfg, "2023-05-18", set())
    # capping first would have kept u1 and then dropped it
    assert seen["cap"] is None
    assert [(r["user_id"], r["cutoff_T"]) for r in kept] == [("u0", "2023-05-18")]
    assert (st["n_instances"], st["n_with_impression"], st["n_kept"]) == (3, 1, 1)
    assert st["no_gold_impression"] == 1 and st["no_negatives"] == 1


def test_impression_examples_match_inference_prompts():
    from coldrank.common import ts_to_date

    items, inst, imp = _imp_world()
    t0 = items.set_index("item_id")["t0"].astype("int64").to_dict()
    kept, _ = attach_negatives(inst, imp, t0, n_neg=5, min_neg=1)
    kept[0]["cutoff_T"] = ts_to_date(T)
    cfg = {"prompt": {"history_len": {"pointwise": 20}}}
    it = items.set_index("item_id")
    rows = build_imp_examples(kept, it, cfg, "chatml", len, 10**6)
    assert [(r["item_id"], r["role"], r["label"]) for r in rows] == \
        [("c20", GOLD, "Yes")] + [(i, IMP_NEG, "No") for i in kept[0]["imp_negatives"]]
    pool = {"history": history_records(kept[0], it)}
    for r in rows:
        cand = candidate_record(r["item_id"], it, is_cold=t0[r["item_id"]] > T, is_gold=r["label"] == "Yes",
                                sim=0.0, pop=0, warm_rank=None)
        assert r["prompt"] == format_chat("chatml", *render_pointwise(pool, cand, 20)) and not r["trimmed"]
    cue = date_cue(kept, t0)
    # c22 (T + 23 h) is newer than the gold c20 (T + 21 h); w00 is old
    assert cue["gold_newer_than_all_negatives"] == 0.0 and cue["neg_cold_frac"] == 0.5


def test_train_settings_override_precedence():
    pytest.importorskip("torch")
    from coldrank.finetune.train import train_settings

    eb = load_config("ebnerd")
    llama, qwen = load_config("finetune"), load_config("finetune_qwen25_7b")
    # Llama-2: EB-NeRD's batch shape applies (4 GPUs x 2 x 4 = 32), as in job 40929
    assert {k: train_settings(eb, llama)[k] for k in ("per_device_batch", "grad_accum")} == {"per_device_batch": 2, "grad_accum": 4}
    # Qwen on one GPU keeps its own batch split (effective 32)
    for name, split in (("finetune_qwen25_7b", (8, 4)), ("finetune_qwen25_7b_imp", (2, 16))):
        tc = train_settings(eb, load_config(name))
        assert (tc["per_device_batch"], tc["grad_accum"]) == split
    assert train_settings(load_config("amazon_vg"), llama) == llama["train"]
    assert qwen["dataset_train_overrides"] is False


def test_impression_finetune_configs_consistent():
    models, eb = load_config("models"), load_config("ebnerd")
    for ft_name, model in (("finetune_imp", "llama2_7b_ftimp_ebnerd"), ("finetune_qwen25_7b_imp", "qwen25_7b_ftimp_ebnerd")):
        fcfg = load_config(ft_name)
        # Qwen: the 1-GPU recipe of the Qwen pool-data runs (Amazon 40320; EB-NeRD on 4 GPUs, 45076)
        pool_cfg = load_config("finetune" if ft_name == "finetune_imp" else "finetune_qwen25_1gpu")
        assert fcfg["sft_variant"] == "imp" and models[model]["base"] == fcfg["base_model"]
        # identical training recipe to the pool-data config, apart from the data variant (and, for
        # Qwen on one GPU, ignoring EB-NeRD's 4-GPU batch override)
        assert {k: v for k, v in fcfg.items() if k not in ("sft_variant", "dataset_train_overrides")} == pool_cfg
        # train.py writes outputs/finetune/<dataset>/<base without _fp16>_<variant>/adapter
        assert models[model]["adapter"] == \
            f"outputs/finetune/{eb['dataset']}/{fcfg['base_model'].replace('_fp16', '')}_imp/adapter"
    # ADA (4 GPUs): the Qwen EB-NeRD pool-data config (job 45076) plus the data variant; same output dir
    ada = load_config("finetune_qwen25_ada_imp")
    assert {k: v for k, v in ada.items() if k != "sft_variant"} == load_config("finetune_qwen25_ada")
    assert ada["sft_variant"] == "imp" and "dataset_train_overrides" not in ada  # EB-NeRD's 2 x 4 x 4 GPUs = 32
    assert ada["base_model"] == load_config("finetune_qwen25_7b_imp")["base_model"]
    # evaluated as the Qwen pool-data model (4-bit at load, same tolerance)
    ft, pool = models["qwen25_7b_ftimp_ebnerd"], models["qwen25_7b_ft_ebnerd"]
    assert {k: v for k, v in ft.items() if k != "adapter"} == {k: v for k, v in pool.items() if k != "adapter"}
    icfg = imp_cfg(eb)
    cuts = check_cutoffs(icfg)
    assert cuts[0] == "2023-05-18" and len(cuts) == 7 and icfg["finetune"]["max_instances_per_cutoff"] == 750


def test_nodates_prompts_match_a3_inference():
    items, emb, inst, topk = _world50(1)
    cfg_pool = {"pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.99,
                         "profile_len": 20}}
    pool = k15_pools(build_pools(inst, topk, items, emb, {}, T, cfg_pool)[0])[0]
    cfg = {"prompt": {"history_len": {"pointwise": 20}}}
    hidden, a3 = hide_dates(pool), make_a3_hide(pool)
    a3_cands = {c["item_id"]: c for c in a3["candidates"]}
    assert pool["candidates"][0]["date"] and pool["history"][0]["date"]  # the input pool is left alone
    assert hidden["pool_id"] == pool["pool_id"]
    # the same candidates are selected as in the dated data
    assert [c["item_id"] for c in select_candidates(hidden, 3)] == [c["item_id"] for c in select_candidates(pool, 3)]
    for c in hidden["candidates"]:
        prompt, _ = pointwise_prompt(hidden, c, cfg, "llama2", len, 10**6)
        assert prompt == format_chat("llama2", *render_pointwise(a3, a3_cands[c["item_id"]], 20))
        assert "launch date" not in prompt


def test_nodates_finetune_config_consistent(tmp_path):
    fcfg, models = load_config("finetune_nodates"), load_config("models")
    assert fcfg["sft_variant"] == "nodates"
    assert {k: v for k, v in fcfg.items() if k != "sft_variant"} == load_config("finetune")
    ctrl, main = models["llama2_7b_ftnodates_amazon_vg"], models["llama2_7b_ft_amazon_vg"]
    assert {k: v for k, v in ctrl.items() if k != "adapter"} == {k: v for k, v in main.items() if k != "adapter"}
    assert ctrl["adapter"] == "outputs/finetune/amazon_vg/llama2_7b_nodates/adapter"
    assert not set(sft_paths(tmp_path, "llama2_7b_fp16", "nodates")) & set(sft_paths(tmp_path, "llama2_7b_fp16"))
    world = {"dataset": "x", "cutoff_T": "2023-05-25", "window_days": 1}
    assert sft_dir(world, "nodates") == sft_dir(world)


def _old_pools(n=6):
    """Old-target K = 15 pools: as _world50, with a warm item outside the top-k as the target."""
    items, emb, inst, topk = _world50(n)
    for r in inst:
        r["instance_id"] = r["instance_id"].replace("train_", "trainold_")
        r["target_item"], r["window_items"], r["kind"] = "w58", ["w58"], "old_target"
        topk[r["instance_id"]] = [f"w{k:02d}" for k in range(55)]
    cfg = {"pool": {"n_warm": 40, "n_cold": 10, "neg_settings": ["random", "hard"], "dup_threshold": 0.99,
                    "profile_len": 20}}
    return k15_old_pools(build_pools(inst, topk, items, emb, {}, T, cfg)[0])


def test_old_pools_have_the_shape_of_the_cold_pools():
    pools = _old_pools()
    assert len(pools) == 6 and {p["condition"] for p in pools} == {"k15_train_old"}
    for p in pools:
        check_old_pool(p)
        gold = next(c for c in p["candidates"] if c["is_gold"])
        assert gold["item_id"] == "w58" and not gold["new_dated"] and not gold["is_cold"]
        roles = [c["role"] for c in select_candidates(p, 3)]
        assert sorted(roles) == sorted([GOLD, COLD_NEG, COLD_NEG, WARM, WARM, WARM])
        assert all(c["new_dated"] for c in p["candidates"] if c["role"] == COLD_NEG)
        assert not any(c["new_dated"] for c in p["candidates"] if c["role"] == WARM)


def test_old_instances_prefer_cold_users_and_exclude_eval_users():
    old = [{"user_id": f"u{k}", "target_item": "w1"} for k in range(20)]
    cfg = {"dataset": "x", "seed": 1}
    cold_users = {"u0", "u1", "u2", "u3", "u19"}
    inst, st = old_training_instances(old, cold_users, {"u19", "u5"}, cfg, "2020-01-01")
    assert {r["user_id"] for r in inst[:4]} == {"u0", "u1", "u2", "u3"}
    assert not {r["user_id"] for r in inst} & {"u19", "u5"}
    assert st["n_same_users"] == 4 and len(inst) == 4 + 2 * 2  # margin: 2 * floor(5 / 2)
    assert len({r["instance_id"] for r in inst}) == len(inst)


def test_mix_keeps_the_count_and_one_pool_per_user():
    cold = [{"pool_id": f"c{k}", "user_id": f"u{k}"} for k in range(11)]
    old = [{"pool_id": f"o{k}", "user_id": f"u{k}"} for k in range(3)] + [{"pool_id": f"o{k}", "user_id": f"x{k}"} for k in range(9)]
    mixed, st = mix_cutoff(cold, old, "2020-01-01")
    users = [p["user_id"] for p in mixed]
    assert len(mixed) == 11 and len(set(users)) == 11
    assert st == {"n_cold_kept": 6, "n_old": 5, "n_old_same_user": 3}
    assert [p["pool_id"] for p in mixed[-5:]] == ["o0", "o1", "o2", "o0", "o1"]  # preference order
    assert mix_cutoff(cold, old, "2020-01-01")[0] == mixed
    with pytest.raises(ValueError):
        mix_cutoff(cold, old[:4], "2020-01-01")


def test_oldmix_and_seed_configs_consistent(tmp_path):
    base, models = load_config("finetune"), load_config("models")
    mix = load_config("finetune_oldmix")
    assert mix["sft_variant"] == "oldmix" and "run_tag" not in mix
    same = lambda c: {**{k: v for k, v in c.items() if k not in ("sft_variant", "run_tag", "train")},
                      "train": {k: v for k, v in c["train"].items() if k not in ("seed", "save_total_limit")}}
    base_view = same(base)
    assert same(mix) == base_view and mix["train"]["seed"] == base["train"]["seed"]
    world = {"dataset": "x", "cutoff_T": "2023-05-25", "window_days": 1}
    assert sft_dir(world, "oldmix") == sft_dir(world)
    assert not set(sft_paths(tmp_path, "llama2_7b_fp16", "oldmix")) & set(sft_paths(tmp_path, "llama2_7b_fp16"))
    main = models["llama2_7b_ft_amazon_vg"]
    assert {k: v for k, v in models["llama2_7b_ftoldmix_amazon_vg"].items() if k != "adapter"} == {k: v for k, v in main.items() if k != "adapter"}
    for sd in (43, 44):
        f = load_config(f"finetune_s{sd}")
        assert same(f) == base_view and f["train"]["seed"] == sd and f["run_tag"] == f"s{sd}" and "sft_variant" not in f
        for ds in ("amazon_vg", "ebnerd"):
            m = models[f"llama2_7b_ft_{ds}_s{sd}"]
            assert m["adapter"] == f"outputs/finetune/{ds}/llama2_7b_s{sd}/adapter"
            assert {k: v for k, v in m.items() if k != "adapter"} == {k: v for k, v in main.items() if k != "adapter"}


def test_oldonly_pools_have_no_new_item_and_six_examples():
    pools = _old_pools()
    cold = [{"pool_id": f"c{k}", "user_id": p["user_id"]} for k, p in enumerate(pools[:4])]
    chosen, st = oldonly_cutoff(cold, pools, "2020-01-01")
    assert len(chosen) == 4 and len({p["user_id"] for p in chosen}) == 4
    assert st["n_old"] == 4 and st["n_old_same_user"] == 4 and 0 <= st["n_gold_newest"] <= 4
    for p in chosen:
        assert p["condition"] == "k15_train_oldonly" and len(p["candidates"]) == 13
        assert not any(c["role"] == COLD_NEG or c["is_cold"] or c["new_dated"] for c in p["candidates"])
        roles = [c["role"] for c in select_candidates(p, 3 + 2)]
        assert sorted(roles) == sorted([GOLD] + [WARM] * 5)
    assert strip_new(pools[0])["candidates"] == [c for c in pools[0]["candidates"] if c["role"] != COLD_NEG]
    with pytest.raises(ValueError):
        oldonly_cutoff(cold + [{"pool_id": "c9", "user_id": "zz"}] * 10, pools, "2020-01-01")


def test_oldonly_takes_one_pool_per_user_in_preference_order():
    pools = _old_pools()
    dup = [pools[0], {**pools[0], "pool_id": "dup"}] + pools[1:]
    chosen, _ = oldonly_cutoff([{"user_id": "a"}, {"user_id": "b"}, {"user_id": "c"}], dup, "2020-01-01")
    assert [p["pool_id"] for p in chosen] == [pools[k]["pool_id"] for k in range(3)]


def test_oldonly_quotas_top_up_shortfall_from_other_cutoffs():
    assert oldonly_quotas([5, 5, 5], [9, 9, 9]) == [5, 5, 5]
    assert oldonly_quotas([5, 5, 5], [6, 3, 9]) == [6, 3, 6]
    assert sum(oldonly_quotas([746, 750], [643, 900])) == 1496
    with pytest.raises(ValueError):
        oldonly_quotas([5, 5], [4, 5])


def test_oldonly_cutoff_takes_n_pools_when_given():
    pools = _old_pools()
    cold = [{"pool_id": "c0", "user_id": pools[0]["user_id"]}]
    chosen, st = oldonly_cutoff(cold, pools, "2020-01-01", n=3)
    assert [p["pool_id"] for p in chosen] == [p["pool_id"] for p in pools[:3]]
    assert st["n_needed"] == 1 and st["n_old"] == 3 and st["n_old_same_user"] == 1


def test_oldonly_config_and_model_consistent(tmp_path):
    base, models = load_config("finetune"), load_config("models")
    f = load_config("finetune_oldonly")
    assert f["sft_variant"] == "oldonly" and "run_tag" not in f
    same = lambda c: {**{k: v for k, v in c.items() if k not in ("sft_variant", "run_tag", "train")},
                      "train": {k: v for k, v in c["train"].items() if k not in ("save_total_limit",)}}
    assert same(f) == same(base)
    world = {"dataset": "x", "cutoff_T": "2023-05-25", "window_days": 1}
    assert sft_dir(world, "oldonly") == sft_dir(world)
    assert not set(sft_paths(tmp_path, "llama2_7b_fp16", "oldonly")) & (set(sft_paths(tmp_path, "llama2_7b_fp16"))
                                                                         | set(sft_paths(tmp_path, "llama2_7b_fp16", "oldmix")))
    main = models["llama2_7b_ft_amazon_vg"]
    m = models["llama2_7b_ftoldonly_amazon_vg"]
    assert m["adapter"] == "outputs/finetune/amazon_vg/llama2_7b_oldonly/adapter"
    assert {k: v for k, v in m.items() if k != "adapter"} == {k: v for k, v in main.items() if k != "adapter"}


def test_oldmix_seed_configs_consistent():
    models, mix = load_config("models"), load_config("finetune_oldmix")
    drop = lambda c: {**{k: v for k, v in c.items() if k != "run_tag"}, "train": {k: v for k, v in c["train"].items() if k != "seed"}}
    for sd in (43, 44):
        f = load_config(f"finetune_oldmix_s{sd}")
        assert f["run_tag"] == f"s{sd}" and f["train"]["seed"] == sd and drop(f) == drop(mix)
        m = models[f"llama2_7b_ftoldmix_amazon_vg_s{sd}"]
        assert m["adapter"] == f"outputs/finetune/amazon_vg/llama2_7b_oldmix_s{sd}/adapter"
        assert {k: v for k, v in m.items() if k != "adapter"} == {k: v for k, v in models["llama2_7b_ftoldmix_amazon_vg"].items() if k != "adapter"}


def test_oldonly_seed_configs_and_models_consistent():
    models, li = load_config("models"), load_config("finetune_oldonly")
    drop = lambda c: {**{k: v for k, v in c.items() if k != "run_tag"}, "train": {k: v for k, v in c["train"].items() if k != "seed"}}
    ref = {k: v for k, v in models["llama2_7b_ftoldonly_amazon_vg"].items() if k != "adapter"}
    assert models["llama2_7b_ftoldonly_ebnerd"]["adapter"] == "outputs/finetune/ebnerd/llama2_7b_oldonly/adapter"
    for sd in (43, 44):
        f = load_config(f"finetune_oldonly_s{sd}")
        assert f["run_tag"] == f"s{sd}" and f["train"]["seed"] == sd and drop(f) == drop(li)
        for ds in ("amazon_vg", "ebnerd"):
            m = models[f"llama2_7b_ftoldonly_{ds}_s{sd}"]
            assert m["adapter"] == f"outputs/finetune/{ds}/llama2_7b_oldonly_s{sd}/adapter"
            assert {k: v for k, v in m.items() if k != "adapter"} == ref


def test_qwen_oldonly_config_and_models_consistent():
    models, ada, f = load_config("models"), load_config("finetune_qwen25_ada"), load_config("finetune_qwen25_ada_oldonly")
    assert f["sft_variant"] == "oldonly" and "run_tag" not in f
    assert {k: v for k, v in f.items() if k not in ("sft_variant", "train")} == {k: v for k, v in ada.items() if k != "train"}
    assert {k: v for k, v in f["train"].items() if k != "save_total_limit"} == ada["train"]
    ref = {k: v for k, v in models["qwen25_7b_ft_ebnerd"].items() if k != "adapter"}
    for ds in ("amazon_vg", "ebnerd"):
        m = models[f"qwen25_7b_ftoldonly_{ds}"]
        assert m["adapter"] == f"outputs/finetune/{ds}/qwen25_7b_oldonly/adapter"
        assert {k: v for k, v in m.items() if k != "adapter"} == ref


def test_merged_root_skips_a_full_disk(tmp_path, monkeypatch):
    ssd, hdd = tmp_path / "ssd", tmp_path / "hdd"
    ssd.mkdir(), hdd.mkdir()
    free = {str(ssd): 138 * 1024**2, str(hdd): 975 * 1024**3}
    monkeypatch.delenv("COLDRANK_MERGED_DIR", raising=False)
    monkeypatch.setenv("USER", "me")
    monkeypatch.setattr(merge_mod, "LOCAL_DISKS", (str(ssd), str(hdd)))
    monkeypatch.setattr(merge_mod.shutil, "disk_usage", lambda d: type("U", (), {"free": free[str(d)]}))
    assert merge_mod.merged_root() == hdd / "me" / "coldrank_merged"
    free[str(ssd)] = 500 * 1024**3
    assert merge_mod.merged_root() == ssd / "me" / "coldrank_merged"
    free[str(ssd)] = free[str(hdd)] = 0
    assert merge_mod.merged_root() == merge_mod.PROJECT_DIR / "merged"
