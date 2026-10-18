"""Baseline candidate pools: 40 warm (SASRec) + 10 cold (gold + 9 negatives).

Per cold-target instance and per negative setting ("random", "hard") we write one
pool record (schema: plan.md Sec. 4.3). Candidates are stored in a neutral order
(sorted by item_id); the prompt renderer shuffles them. Displayed dates live in
`date`, true launch times in `t0`, so date variants only rewrite `date`.

Exclusions (design.md):
  * no candidate is in H_u;
  * no negative (warm or cold) is another item the user interacted with in the
    test window (those would be unlabelled positives);
  * cold negatives launched in (T, target_ts]: cold for the user and already
    available at decision time;
  * cold negatives whose text embedding has cosine >= dup_threshold with the gold
    (other editions / bundles of the same product) are dropped.

Usage:
    python -m coldrank.pools.build --config amazon_vg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.common import date_to_ts, load_config, read_jsonl, rng_for, ts_to_date, write_jsonl
from coldrank.data.split import split_dir


OPTIONAL_FIELDS = ("subtitle", "topics")  # EB-NeRD only; Amazon items have neither, so its records are unchanged


def item_text(row) -> str:
    text = row["title"]
    if isinstance(row.get("subtitle"), str) and row["subtitle"]:
        text = f"{text}. {row['subtitle']}"  # news: the subtitle carries most of the content
    return f"{text} | {row['category_path']}" if row["category_path"] else text


def optional_fields(r) -> dict:
    return {k: r[k] for k in OPTIONAL_FIELDS if k in r.index}


class Embeddings:
    """Normalized sentence embeddings for a set of items, cached on disk."""

    def __init__(self, ids: list[str], vecs: np.ndarray):
        self.index = {i: k for k, i in enumerate(ids)}
        self.vecs = vecs.astype(np.float32)

    def __getitem__(self, ids: list[str]) -> np.ndarray:
        return self.vecs[[self.index[i] for i in ids]]

    @classmethod
    def build(cls, items: pd.DataFrame, ids: list[str], model_name: str, cache: Path) -> "Embeddings":
        ids = sorted(set(ids))
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            cached = dict(zip(z["ids"].tolist(), range(len(z["ids"]))))
            if all(i in cached for i in ids):
                return cls(z["ids"].tolist(), z["vecs"])
        from sentence_transformers import SentenceTransformer

        texts = items.set_index("item_id").loc[ids].apply(item_text, axis=1).tolist()
        model = SentenceTransformer(model_name)
        vecs = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
        np.savez(cache, ids=np.array(ids), vecs=vecs)
        return cls(ids, vecs)


def profile(emb: Embeddings, hist_items: list[str]) -> np.ndarray:
    v = emb[hist_items].mean(0)
    return v / (np.linalg.norm(v) + 1e-12)


def candidate_record(
    item_id: str, it: pd.DataFrame, is_cold: bool, is_gold: bool, sim: float, pop: int, warm_rank: int | None
) -> dict:
    r = it.loc[item_id]
    return {
        "item_id": item_id,
        "title": r["title"],
        "date": ts_to_date(int(r["t0"])),  # displayed date (variants rewrite this)
        "t0": int(r["t0"]),                 # true launch time (never rewritten)
        "t0_source": r["t0_source"],
        "category": r["category_path"],
        "brand": r["brand"],
        "is_cold": is_cold,
        "is_gold": is_gold,
        "sim_to_history": round(float(sim), 4),
        "pop_preT": int(pop),
        "warm_rank": warm_rank,             # 1-based SASRec rank among warm candidates; None for cold
        **optional_fields(r),
    }


def cold_catalogue(items: pd.DataFrame, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Cold items (t0 > T) sorted by launch time: (item_ids, t0s)."""
    t0 = items.set_index("item_id")["t0"].astype("int64")
    cold_ids = t0[t0 > T].sort_values(kind="mergesort")  # ascending launch time
    return cold_ids.index.to_numpy(), cold_ids.to_numpy()


def eligible_cold(
    r: dict, gold: str, excl: set, cold_index: np.ndarray, cold_t0: np.ndarray, emb: "Embeddings", dup_thr: float
) -> tuple[list[str], int | None]:
    """Cold items usable as negatives for instance r: launched in (T, target_ts], not in
    `excl`, and not a near-duplicate of the gold (cosine >= dup_thr). Returns the list
    (in launch-time order) and the number of near-duplicates removed (None if no items)."""
    hi = np.searchsorted(cold_t0, r["target_ts"], side="right")
    elig = [i for i in cold_index[:hi] if i not in excl]
    if not elig:
        return elig, None
    sim_gold = emb[elig] @ emb[[gold]][0]
    keep = sim_gold < dup_thr
    return [i for i, k in zip(elig, keep) if k], int((~keep).sum())


def history_records(r: dict, it: pd.DataFrame) -> list[dict]:
    """Displayed history records for an instance (dates = item launch dates)."""
    out = []
    for h_item, h_ts in r["history"]:
        h = it.loc[h_item]
        out.append({"item_id": h_item, "title": h["title"], "date": ts_to_date(int(h["t0"])),
                    "t0": int(h["t0"]), "category": h["category_path"], "brand": h["brand"],
                    "ts": int(h_ts), **optional_fields(h)})
    return out


def build_pools(
    instances: list[dict],
    topk: dict[str, list[str]],
    items: pd.DataFrame,
    emb: Embeddings,
    pop: dict[str, int],
    T: int,
    cfg: dict,
) -> tuple[list[dict], dict]:
    pc = cfg["pool"]
    n_warm, n_neg = pc["n_warm"], pc["n_cold"] - 1
    dup_thr = pc["dup_threshold"]
    prof_len = pc["profile_len"]
    it = items.set_index("item_id")
    cold_index, cold_t0 = cold_catalogue(items, T)

    pools, dropped = [], {"too_few_warm": 0, "too_few_cold": 0}
    dup_removed = []
    for r in instances:
        hist = [h[0] for h in r["history"]]
        excl = set(hist) | set(r["window_items"])
        gold = r["target_item"]
        warm = [i for i in topk[r["instance_id"]] if i not in excl][:n_warm]
        if len(warm) < n_warm:
            dropped["too_few_warm"] += 1
            continue
        # cold negatives: launched in (T, target_ts], not the user's, not near-duplicates of gold
        elig, n_dup = eligible_cold(r, gold, excl, cold_index, cold_t0, emb, dup_thr)
        if n_dup is not None:
            dup_removed.append(n_dup)
        if len(elig) < n_neg:
            dropped["too_few_cold"] += 1
            continue
        prof = profile(emb, hist[-prof_len:])
        sims_elig = emb[elig] @ prof
        for setting in pc["neg_settings"]:
            if setting == "random":
                rng = rng_for("cold_negs", r["instance_id"], setting)
                negs = rng.sample(elig, n_neg)
            elif setting == "hard":
                order = np.lexsort((np.array(elig), -sims_elig))  # sim desc, ties by id
                negs = [elig[k] for k in order[:n_neg]]
            else:
                raise ValueError(setting)
            cands = warm + [gold] + negs
            sims = emb[cands] @ prof
            cand_recs = [
                candidate_record(c, it, is_cold=(k >= n_warm), is_gold=(c == gold), sim=s, pop=pop.get(c, 0),
                                 warm_rank=(k + 1 if k < n_warm else None))
                for k, (c, s) in enumerate(zip(cands, sims))
            ]
            cand_recs.sort(key=lambda c: c["item_id"])  # neutral order; shuffled at prompt time
            hist_recs = history_records(r, it)
            pools.append({
                "pool_id": f"{r['instance_id']}__{setting}",
                "instance_id": r["instance_id"],
                "split": r["split"],
                "user_id": r["user_id"],
                "condition": "baseline",
                "neg_setting": setting,
                "cutoff_T": ts_to_date(T),
                "target_ts": r["target_ts"],
                "gold_item_id": gold,
                "history": hist_recs,
                "candidates": cand_recs,
            })
    stats = {
        "n_instances_in": len(instances),
        "dropped": dropped,
        "mean_dup_removed_per_instance": float(np.mean(dup_removed)) if dup_removed else 0.0,
    }
    return pools, stats


def check_pool(p: dict, n_warm: int, n_cold: int) -> None:
    """Hard invariants for a baseline pool (plan.md Sec. 6)."""
    c = p["candidates"]
    ids = [x["item_id"] for x in c]
    T = date_to_ts(p["cutoff_T"])
    assert len(ids) == n_warm + n_cold == len(set(ids)), p["pool_id"]
    assert sum(x["is_gold"] for x in c) == 1 and next(x for x in c if x["is_gold"])["item_id"] == p["gold_item_id"]
    assert sum(x["is_cold"] for x in c) == n_cold
    assert next(x for x in c if x["is_gold"])["is_cold"]
    assert not set(ids) & {h["item_id"] for h in p["history"]}
    assert all(x["t0"] > T for x in c if x["is_cold"]) and all(x["t0"] <= T for x in c if not x["is_cold"])
    assert all(h["ts"] < T for h in p["history"])
    assert all(x["t0"] <= p["target_ts"] for x in c)


def pool_summary(pools: list[dict]) -> dict:
    out = {}
    for setting in sorted({p["neg_setting"] for p in pools}):
        ps = [p for p in pools if p["neg_setting"] == setting]
        g_sim, n_sim, w_sim, age_gap, gold_newest = [], [], [], [], 0
        for p in ps:
            gold = next(c for c in p["candidates"] if c["is_gold"])
            g_sim.append(gold["sim_to_history"])
            n_sim += [c["sim_to_history"] for c in p["candidates"] if c["is_cold"] and not c["is_gold"]]
            w_sim += [c["sim_to_history"] for c in p["candidates"] if not c["is_cold"]]
            age_gap.append((gold["t0"] - np.median([c["t0"] for c in p["candidates"] if not c["is_cold"]])) / 86400)
            gold_newest += gold["t0"] == max(c["t0"] for c in p["candidates"])
        out[setting] = {
            "n_pools": len(ps),
            "sim_gold_mean": float(np.mean(g_sim)),
            "sim_cold_neg_mean": float(np.mean(n_sim)),
            "sim_warm_mean": float(np.mean(w_sim)),
            "gold_minus_median_warm_age_days": float(np.median(age_gap)),
            "gold_is_newest_frac": gold_newest / len(ps),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sd = split_dir(cfg)
    T = date_to_ts(str(cfg["cutoff_T"]))

    items = pd.read_parquet(sd / "items.parquet")
    pos = pd.read_parquet(sd / "positives.parquet")
    pop = pos[pos["ts"] < T].groupby("item_id").size().to_dict()
    topk = {r["instance_id"]: r["items"] for r in read_jsonl(sd / "sasrec_topk.jsonl")}
    instances = list(read_jsonl(sd / "cold_dev.jsonl")) + list(read_jsonl(sd / "cold_test.jsonl"))

    t0 = items.set_index("item_id")["t0"].astype("int64")
    w_end = T + int(cfg["window_days"]) * 86400
    need = set(t0.index[(t0 > T) & (t0 < w_end)])
    for r in instances:
        need |= {h[0] for h in r["history"]} | set(topk[r["instance_id"]]) | {r["target_item"]}
    emb = Embeddings.build(items, sorted(need), cfg["pool"]["embed_model"], sd / "item_embeddings.npz")

    pools, stats = build_pools(instances, topk, items, emb, pop, T, cfg)
    for p in pools:
        check_pool(p, cfg["pool"]["n_warm"], cfg["pool"]["n_cold"])
    out = sd / "pools"
    out.mkdir(exist_ok=True)
    for split in ("dev", "test"):
        for setting in cfg["pool"]["neg_settings"]:
            rows = [p for p in pools if p["split"] == split and p["neg_setting"] == setting]
            write_jsonl(out / f"baseline_{setting}_{split}.jsonl", rows)
    stats["summary"] = pool_summary(pools)
    (out / "pool_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
