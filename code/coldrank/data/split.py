"""Global-cutoff cold-start split (Li et al. 2026, Sec. 2.1 / 3.1).

Definitions (see design.md):
  * Launch time  t0(i) = min(Date First Available, first review time); the first
    review time alone when DFA is missing. An item cannot be reviewed before it is
    available, so taking the min also guarantees that no cold item occurs in any
    pre-cutoff history.
  * Positive interaction: rating >= cfg.rating_threshold, de-duplicated per
    (user, item) keeping the earliest.
  * Global cutoff T. History H_u = u's positives with ts < T (chronological).
  * Cold item: t0 > T.  Warm item: t0 <= T and >= 1 positive before T.
  * Cold-target instance: u with |H_u| >= min_history; target = u's first positive
    in [T, T + w) whose item is cold.
  * Old-target instance (for A1 later): u's first positive in [T, T + w) whose item
    is warm and not in H_u.

Usage:
    python -m coldrank.data.split --config amazon_vg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.common import DATA_DIR, date_to_ts, load_config, rng_for, ts_to_date, write_jsonl


# --------------------------------------------------------------------------- #
# Step 1: items with launch times, positive interactions
# --------------------------------------------------------------------------- #
def attach_launch_times(items: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    """Add t0 / t0_source / first_review_ts to items. Uses ALL reviews (any rating):
    any review proves the item was available by then."""
    first = inter.groupby("item_id", sort=False)["ts"].min().rename("first_review_ts")
    out = items.merge(first, left_on="item_id", right_index=True, how="left")
    dfa = out["dfa_ts"].astype("float64")
    fr = out["first_review_ts"].astype("float64")
    t0 = np.fmin(dfa, fr)  # fmin ignores NaN on either side
    out["t0"] = pd.array(np.where(np.isnan(t0), pd.NA, t0), dtype="Int64")
    src = np.where(
        dfa.isna() & fr.isna(),
        "none",
        np.where(fr.isna() | (dfa <= fr), "dfa", "first_review"),
    )
    out["t0_source"] = src
    return out


def usable_items(items: pd.DataFrame) -> pd.DataFrame:
    """Items we can show to an LLM: non-empty title and a launch time."""
    return items[(items["title"].str.len() > 0) & items["t0"].notna()].copy()


def positives(inter: pd.DataFrame, items: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    """Positive interactions on usable items, earliest per (user, item), sorted."""
    p = inter[(inter["rating"] >= rating_threshold) & inter["item_id"].isin(items["item_id"])]
    p = p.sort_values(["user_id", "ts", "item_id"], kind="mergesort")
    p = p.drop_duplicates(["user_id", "item_id"], keep="first")
    return p.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Step 2: instances for one (T, w)
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    cold_instances: list[dict]
    old_instances: list[dict]
    stats: dict


def build_instances(
    pos: pd.DataFrame,
    items: pd.DataFrame,
    T: int,
    w_days: int,
    min_history: int,
) -> SplitResult:
    t0 = items.set_index("item_id")["t0"].astype("int64")
    cold_items = set(t0.index[t0 > T])

    pre = pos[pos["ts"] < T]
    warm_items = set(pre["item_id"].unique()) - cold_items  # t0 <= T is implied, see below
    # Sanity: an item with a pre-T positive has a review before T, so t0 <= first review < T.
    assert not (set(pre["item_id"].unique()) & cold_items), "cold item found in a pre-T history"

    hist_len = pre.groupby("user_id", sort=False).size()
    eligible = set(hist_len.index[hist_len >= min_history])

    w_end = T + w_days * 86400
    post = pos[(pos["ts"] >= T) & (pos["ts"] < w_end) & pos["user_id"].isin(eligible)]
    post = post.sort_values(["user_id", "ts", "item_id"], kind="mergesort")

    pre_e = pre[pre["user_id"].isin(set(post["user_id"].unique()))]
    histories = {
        u: list(zip(g["item_id"].tolist(), g["ts"].tolist()))
        for u, g in pre_e.groupby("user_id", sort=False)
    }

    cold, old = [], []
    for u, g in post.groupby("user_id", sort=False):
        hist = histories[u]
        hist_set = {i for i, _ in hist}
        window_items = g["item_id"].tolist()
        got_cold = got_old = False
        for item, ts in zip(g["item_id"].tolist(), g["ts"].tolist()):
            rec = None
            if not got_cold and item in cold_items:
                rec, got_cold = "cold_target", True
            elif not got_old and item in warm_items and item not in hist_set:
                rec, got_old = "old_target", True
            if rec:
                (cold if rec == "cold_target" else old).append(
                    {
                        "user_id": u,
                        "kind": rec,
                        "target_item": item,
                        "target_ts": int(ts),
                        "history": [[i, int(t)] for i, t in hist],
                        "window_items": window_items,
                    }
                )
            if got_cold and got_old:
                break

    stats = {
        "T": ts_to_date(T),
        "w_days": w_days,
        "min_history": min_history,
        "n_eligible_users": len(eligible),
        "n_cold_items": len(cold_items),
        "n_warm_items": len(warm_items),
        "n_cold_instances": len(cold),
        "n_old_instances": len(old),
        "n_cold_target_items_distinct": len({r["target_item"] for r in cold}),
    }
    return SplitResult(cold, old, stats)


def assign_ids_and_subsample(
    instances: list[dict], prefix: str, n_dev: int, n_test: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle by user, then dev = first n_dev, test = next n_test.
    Dev and test are user-disjoint (each user has at most one instance per kind)."""
    rng = rng_for("subsample", prefix, seed)
    order = sorted(instances, key=lambda r: r["user_id"])  # canonical order first
    rng.shuffle(order)
    if len(order) < n_dev + n_test:
        print(f"[warn] {prefix}: only {len(order)} instances for dev={n_dev} + test={n_test}")
    dev, test = order[:n_dev], order[n_dev : n_dev + n_test]
    for split, rows in (("dev", dev), ("test", test)):
        for k, r in enumerate(rows):
            r["instance_id"] = f"{prefix}_{split}_{k:05d}"
            r["split"] = split
    return dev, test


# --------------------------------------------------------------------------- #
# Optional steps (EB-NeRD; config keys absent for Amazon)
# --------------------------------------------------------------------------- #
def new_between(items: pd.DataFrame, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Cold items (t0 > T) sorted by launch time: (item_ids, t0s)."""
    t0 = items.set_index("item_id")["t0"].astype("int64")
    c = t0[t0 > T].sort_values(kind="mergesort")
    return c.index.to_numpy(), c.to_numpy()


def require_cold_negs(instances: list[dict], items: pd.DataFrame, T: int, k: int) -> tuple[list[dict], int]:
    """Keep instances with >= k cold items launched in (T, target_ts] that are not in the
    user's window: the pool builder's precondition for its k cold negatives."""
    ids, t0s = new_between(items, T)
    keep = []
    for r in instances:
        hi = np.searchsorted(t0s, r["target_ts"], side="right")
        excl = set(r["window_items"])
        if sum(1 for i in ids[:hi] if i not in excl) >= k:
            keep.append(r)
    return keep, len(instances) - len(keep)


def attach_inview_not_clicked(instances: list[dict], imp: pd.DataFrame, items: pd.DataFrame, T: int) -> None:
    """A1 traps for news (design.md §4): new articles (launched in (T, target_ts]) shown to
    the user in an impression in [T, target_ts] that the user did not click in the window
    and that are not in the history. Stored sorted as `inview_not_clicked`."""
    t0 = items.set_index("item_id")["t0"].astype("int64").to_dict()
    users = {r["user_id"] for r in instances}
    imp = imp[(imp["ts"] >= T) & imp["user_id"].isin(users)]
    by_user = {u: g for u, g in imp.groupby("user_id", sort=False)}
    for r in instances:
        g = by_user.get(r["user_id"])
        seen = set()
        if g is not None:
            for v in g.loc[g["ts"] <= r["target_ts"], "inview"]:
                seen.update(v)
        excl = set(r["window_items"]) | {h[0] for h in r["history"]}
        r["inview_not_clicked"] = sorted(i for i in seen - excl if i in t0 and T < t0[i] <= r["target_ts"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_processed(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = DATA_DIR / "processed" / dataset
    return pd.read_parquet(proc / "items.parquet"), pd.read_parquet(proc / "interactions.parquet")


def split_dir(cfg: dict) -> Path:
    return DATA_DIR / "processed" / cfg["dataset"] / f"split_T{cfg['cutoff_T']}_w{cfg['window_days']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("cutoff_T") in (None, "TBD") or cfg.get("window_days") in (None, "TBD"):
        raise SystemExit("cutoff_T / window_days not set: run coldrank.data.audit first")

    items_raw, inter = load_processed(cfg["dataset"])
    items = usable_items(attach_launch_times(items_raw, inter))
    pos = positives(inter, items, cfg["rating_threshold"])
    T = date_to_ts(str(cfg["cutoff_T"]))
    res = build_instances(pos, items, T, int(cfg["window_days"]), int(cfg["min_history"]))

    cold_inst, old_inst = res.cold_instances, res.old_instances
    if cfg.get("require_cold_negs"):
        cold_inst, n_drop = require_cold_negs(cold_inst, items, T, int(cfg["require_cold_negs"]))
        res.stats["cold_dropped_too_few_cold_negs"] = n_drop
    if (cfg.get("adversarial") or {}).get("trap_source") == "inview":
        imp = pd.read_parquet(DATA_DIR / "processed" / cfg["dataset"] / "impressions.parquet")
        attach_inview_not_clicked(old_inst, imp, items, T)
        if cfg.get("require_traps"):
            k = int(cfg["require_traps"])
            n0 = len(old_inst)
            old_inst = [r for r in old_inst if len(r["inview_not_clicked"]) >= k]
            res.stats["old_dropped_too_few_traps"] = n0 - len(old_inst)

    out = split_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    n_test_cold = cfg.get("n_test_cold", cfg["n_test"])
    c_dev, c_test = assign_ids_and_subsample(cold_inst, "cold", cfg["n_dev"], n_test_cold, seed)
    o_dev, o_test = assign_ids_and_subsample(old_inst, "old", cfg["n_dev"], cfg["n_test"], seed)
    write_jsonl(out / "cold_dev.jsonl", c_dev)
    write_jsonl(out / "cold_test.jsonl", c_test)
    write_jsonl(out / "old_dev.jsonl", o_dev)
    write_jsonl(out / "old_test.jsonl", o_test)

    items.to_parquet(out / "items.parquet", index=False)
    pos.to_parquet(out / "positives.parquet", index=False)
    res.stats.update(
        n_cold_dev=len(c_dev), n_cold_test=len(c_test), n_old_dev=len(o_dev), n_old_test=len(o_test),
        rating_threshold=cfg["rating_threshold"],
    )
    (out / "split_stats.json").write_text(json.dumps(res.stats, indent=2))
    print(json.dumps(res.stats, indent=2))


if __name__ == "__main__":
    main()
