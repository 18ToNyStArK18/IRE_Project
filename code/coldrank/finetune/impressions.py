"""Production-style SFT examples for EB-NeRD: negatives from the impression of the gold click.

The pool-based data (trainpools + sft_data) has the test pools' shape: every warm negative is
older than the gold, so pointwise training rewards "old means No". Here, as in news production
logs, the negatives are the articles shown to the user together with the gold and not clicked
(design.md "Production-style training data").

Per cutoff T' (config `finetune.impressions.cutoffs`; EB-NeRD impressions start 2023-05-18):
  1. instances: `trainpools.training_instances` (test split rules, no dev/test user,
     `require_cold_negs`, seeded cap `max_instances_per_cutoff`);
  2. the gold's impression(s): the user's impressions at the target time whose clicks include
     the gold; negatives = in view, not clicked there, not in the user's window or history,
     a known article launched by the target time; up to `n_neg` (seeded); an instance with
     fewer than `min_neg` is dropped;
  3. examples: the gold ("Yes") and each negative ("No"), with the inference prompt
     (`sft_data.pointwise_prompt`). Users are split into train / validation as in sft_data.

Outputs (in <split dir>/finetune/impressions/): instances.jsonl, and per model
sft_{train,val}_<model>_imp.jsonl, sft_stats_<model>_imp.json (incl. date-cue diagnostics).

Usage:
    python -m coldrank.finetune.impressions --config ebnerd --model llama2_7b_fp16
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from coldrank.adversarial.common import GOLD
from coldrank.common import DATA_DIR, date_to_ts, load_config, rng_for, write_jsonl
from coldrank.data.split import split_dir
from coldrank.finetune.sft_data import is_holdout, pointwise_prompt, sft_dir, sft_paths
from coldrank.finetune.trainpools import check_cutoffs, eval_users, training_instances
from coldrank.pools.build import candidate_record, history_records

IMP_NEG = "imp_neg"
VARIANT = "imp"


def imp_cfg(cfg: dict) -> dict:
    """The dataset config with `finetune.impressions` laid over `finetune`, so the cutoff check
    and `training_instances` read the impression cutoffs and cap unchanged."""
    return {**cfg, "finetune": {**cfg["finetune"], **cfg["finetune"]["impressions"]}}


def impression_negatives(r: dict, imps: pd.DataFrame, t0: dict, n_neg: int) -> tuple[list[str], str | None]:
    """(negatives, drop reason) of instance r; `imps` holds the user's impressions (ts, inview, clicked)."""
    at = imps[imps["ts"] == r["target_ts"]]
    rows = [k for k, c in enumerate(at["clicked"]) if r["target_item"] in c]
    if not rows:
        return [], "no_gold_impression"
    shown, clicked = set(), set()
    for k in rows:
        shown.update(at["inview"].iloc[k])
        clicked.update(at["clicked"].iloc[k])
    excl = clicked | set(r["window_items"]) | {h[0] for h in r["history"]}
    elig = sorted(i for i in shown - excl if i in t0 and t0[i] <= r["target_ts"])
    if not elig:
        return [], "no_negatives"
    return rng_for("ft_imp_negs", r["instance_id"]).sample(elig, min(n_neg, len(elig))), None


def attach_negatives(inst: list[dict], imp: pd.DataFrame, t0: dict, n_neg: int, min_neg: int) -> tuple[list[dict], dict]:
    """Instances with their impression negatives (`imp_negatives`), and the drop counts."""
    by_user = {u: g for u, g in imp[imp["user_id"].isin({r["user_id"] for r in inst})].groupby("user_id", sort=False)}
    empty = imp.iloc[:0]
    kept, drops = [], {"no_gold_impression": 0, "no_negatives": 0, "too_few_negatives": 0}
    for r in inst:
        negs, why = impression_negatives(r, by_user.get(r["user_id"], empty), t0, n_neg)
        if why is None and len(negs) < min_neg:
            why = "too_few_negatives"
        if why:
            drops[why] += 1
            continue
        kept.append({**r, "imp_negatives": negs})
    return kept, drops


def build_examples(inst: list[dict], it: pd.DataFrame, cfg: dict, family: str, count_tokens, max_len: int) -> list[dict]:
    """Pointwise examples: gold "Yes", each impression negative "No"; prompts as at inference."""
    rows = []
    for r in inst:
        T = date_to_ts(r["cutoff_T"])
        pool = {"pool_id": f"imp_{r['instance_id']}", "user_id": r["user_id"], "history": history_records(r, it)}
        for item, is_gold in [(r["target_item"], True)] + [(i, False) for i in r["imp_negatives"]]:
            cand = candidate_record(item, it, is_cold=int(it.loc[item, "t0"]) > T, is_gold=is_gold, sim=0.0, pop=0,
                                    warm_rank=None)
            del cand["sim_to_history"], cand["pop_preT"]  # not computed here (never rendered)
            prompt, trimmed = pointwise_prompt(pool, cand, cfg, family, count_tokens, max_len)
            rows.append({"pool_id": pool["pool_id"], "user_id": r["user_id"], "item_id": item,
                         "role": GOLD if is_gold else IMP_NEG, "label": "Yes" if is_gold else "No",
                         "trimmed": trimmed, "prompt": prompt})
    return rows


def date_cue(inst: list[dict], t0: dict) -> dict:
    """How much the launch time alone separates the gold from its negatives."""
    newest, gold_age, neg_age, neg_cold = [], [], [], []
    for r in inst:
        T, g = date_to_ts(r["cutoff_T"]), t0[r["target_item"]]
        newest.append(all(g > t0[i] for i in r["imp_negatives"]))
        gold_age.append((r["target_ts"] - g) / 3600)
        neg_age += [(r["target_ts"] - t0[i]) / 3600 for i in r["imp_negatives"]]
        neg_cold += [t0[i] > T for i in r["imp_negatives"]]
    return {"gold_newer_than_all_negatives": float(np.mean(newest)),
            "gold_age_hours_median": float(np.median(gold_age)), "neg_age_hours_median": float(np.median(neg_age)),
            "neg_cold_frac": float(np.mean(neg_cold))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ebnerd")
    ap.add_argument("--model", default="llama2_7b_fp16", help="models.yaml key: tokenizer and chat format")
    args = ap.parse_args()
    cfg = load_config(args.config)
    icfg = imp_cfg(cfg)
    ic = icfg["finetune"]
    mcfg = load_config("models")[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=mcfg["revision"])
    count = lambda s: len(tok(s)["input_ids"])  # noqa: E731  (with BOS, as vLLM counts)

    sd = split_dir(cfg)
    items = pd.read_parquet(sd / "items.parquet")
    pos = pd.read_parquet(sd / "positives.parquet")
    imp = pd.read_parquet(DATA_DIR / "processed" / cfg["dataset"] / "impressions.parquet")
    imp["inview"] = imp["inview"].map(list)
    imp["clicked"] = imp["clicked"].map(list)
    t0 = items.set_index("item_id")["t0"].astype("int64").to_dict()
    it = items.set_index("item_id")
    excluded, T = eval_users(sd), date_to_ts(str(cfg["cutoff_T"]))

    inst, per_cut = [], {}
    for c in check_cutoffs(icfg):
        ci, st = training_instances(pos, items, icfg, c, excluded)
        ci, drops = attach_negatives(ci, imp, t0, int(ic["n_neg"]), int(ic["min_neg"]))
        for r in ci:
            r["cutoff_T"] = c
        per_cut[c] = {**st, **drops, "n_kept": len(ci)}
        inst += ci
    for r in inst:  # leak checks, as in trainpools.collect
        assert r["user_id"] not in excluded and r["target_ts"] < T, r["instance_id"]
        assert all(t0[i] <= r["target_ts"] for i in r["imp_negatives"]), r["instance_id"]

    out = sft_dir(cfg, VARIANT)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "instances.jsonl", inst)
    rows = build_examples(inst, it, cfg, mcfg["family"], count, mcfg["max_model_len"])
    frac = float(cfg["finetune"]["holdout_frac"])
    train_path, val_path, stats_path = sft_paths(out, args.model, VARIANT)
    write_jsonl(train_path, [r for r in rows if not is_holdout(r["user_id"], frac)])
    write_jsonl(val_path, [r for r in rows if is_holdout(r["user_id"], frac)])
    lens = np.array([count(r["prompt"]) for r in rows])
    n_val = sum(is_holdout(r["user_id"], frac) for r in rows)
    stats = {
        "model": args.model, "variant": VARIANT, "per_cutoff": per_cut, "n_instances": len(inst),
        "n_users": len({r["user_id"] for r in inst}), "n_distinct_golds": len({r["target_item"] for r in inst}),
        "negatives_per_instance": float(np.mean([len(r["imp_negatives"]) for r in inst])),
        "n_train": len(rows) - n_val, "n_val": n_val,
        "positive_rate": float(np.mean([r["label"] == "Yes" for r in rows])),
        "prompt_tokens": {"mean": float(lens.mean()), "p50": float(np.percentile(lens, 50)),
                          "p95": float(np.percentile(lens, 95)), "max": int(lens.max()), "total": int(lens.sum())},
        "history_trimmed": sum(r["trimmed"] for r in rows),
        "date_cue": date_cue(inst, t0),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
