"""Phase 0 data audit: launch-date coverage and instance counts per (T, w).

The counts here must agree with coldrank.data.split.build_instances (checked in
tests/test_split.py), so they can be used to choose cutoff_T and window_days.

Usage:
    python -m coldrank.data.audit --config amazon_vg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.common import OUT_DIR, load_config, ts_to_date
from coldrank.data.split import attach_launch_times, load_processed, positives, usable_items


def count_instances(
    pos: pd.DataFrame, items: pd.DataFrame, T: int, windows: list[int], min_history: int
) -> list[dict]:
    t0 = items.set_index("item_id")["t0"].astype("int64")
    is_cold = pos["item_id"].map(t0 > T).fillna(False).to_numpy()
    pre = pos[pos["ts"].to_numpy() < T]
    hist_len = pre.groupby("user_id", sort=False).size()
    eligible = hist_len.index[hist_len >= min_history]

    max_end = T + max(windows) * 86400
    m = (pos["ts"].to_numpy() >= T) & (pos["ts"].to_numpy() < max_end) & is_cold
    post_cold = pos[m & pos["user_id"].isin(eligible).to_numpy()]
    first_cold = post_cold.groupby("user_id", sort=False)["ts"].min()

    rows = []
    for w in windows:
        sel = first_cold[first_cold < T + w * 86400]
        tgt = post_cold[post_cold["user_id"].isin(sel.index)]
        tgt = tgt.sort_values(["user_id", "ts", "item_id"], kind="mergesort").drop_duplicates("user_id")
        rows.append(
            {
                "T": ts_to_date(T),
                "w_days": w,
                "eligible_users": int(len(eligible)),
                "cold_items": int((t0 > T).sum()),
                "cold_instances": int(len(sel)),
                "distinct_targets": int(tgt["item_id"].nunique()),
                "median_hist_len": float(hist_len.loc[sel.index].median()) if len(sel) else float("nan"),
            }
        )
    return rows


def month_starts(first: str, last: str) -> list[int]:
    return [int(d.timestamp()) for d in pd.date_range(first, last, freq="MS", tz="UTC")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--first", default="2021-01-01")
    ap.add_argument("--last", default="2023-08-01")
    ap.add_argument("--windows", default="7,14,30,60,90")
    args = ap.parse_args()
    cfg = load_config(args.config)
    windows = [int(x) for x in args.windows.split(",")]

    items_raw, inter = load_processed(cfg["dataset"])
    items_all = attach_launch_times(items_raw, inter)
    items = usable_items(items_all)
    pos = positives(inter, items, cfg["rating_threshold"])

    out_dir = OUT_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Data audit: {cfg['dataset']}\n"]

    # ---- launch dates ---------------------------------------------------- #
    n = len(items_all)
    has_title = (items_all["title"].str.len() > 0).sum()
    has_dfa = items_all["dfa_ts"].notna().sum()
    has_rel = items_all["release_ts"].notna().sum()
    has_rev = items_all["first_review_ts"].notna().sum()
    both = items_all["dfa_ts"].notna() & items_all["first_review_ts"].notna()
    dfa_after = (items_all.loc[both, "dfa_ts"] > items_all.loc[both, "first_review_ts"])
    gap_days = (
        (items_all.loc[both, "dfa_ts"] - items_all.loc[both, "first_review_ts"]).astype("float64") / 86400
    )[dfa_after]
    unparsed = items_all["dfa_raw"].notna() & items_all["dfa_ts"].isna()
    lines += [
        "## Launch dates\n",
        f"- items in metadata: {n:,}; with title: {has_title:,}",
        f"- Date First Available parsed: {has_dfa:,} ({has_dfa / n:.1%}); present but unparseable: {unparsed.sum():,}",
        f"- Release date parsed: {has_rel:,} ({has_rel / n:.1%})",
        f"- items with >= 1 review: {has_rev:,}",
        f"- DFA *after* first review (DFA wrong or relisted): {dfa_after.sum():,} of {both.sum():,} "
        f"({dfa_after.mean():.1%}); median gap {gap_days.median():.0f} days",
        f"- t0 source among usable items: {items['t0_source'].value_counts().to_dict()}",
        f"- usable items (title + t0): {len(items):,}",
        f"- reviews: {len(inter):,}; positives (rating >= {cfg['rating_threshold']}, dedup): {len(pos):,}; "
        f"users: {pos['user_id'].nunique():,}",
        f"- review time range: {ts_to_date(inter['ts'].min())} .. {ts_to_date(inter['ts'].max())}\n",
    ]
    if len(unparsed):
        ex = items_all.loc[unparsed, "dfa_raw"].head(10).tolist()
        lines.append(f"- unparseable DFA examples: {ex}\n")

    # ---- activity over time --------------------------------------------- #
    by_year_rev = pd.to_datetime(pos["ts"], unit="s").dt.year.value_counts().sort_index()
    by_year_launch = pd.to_datetime(items["t0"].astype("int64"), unit="s").dt.year.value_counts().sort_index()
    lines += ["## Positives and launches per year\n", "| year | positives | launches |", "|---|---|---|"]
    for y in sorted(set(by_year_rev.index) | set(by_year_launch.index)):
        if y >= 2010:
            lines.append(f"| {y} | {by_year_rev.get(y, 0):,} | {by_year_launch.get(y, 0):,} |")
    lines.append("")

    # ---- instance counts ------------------------------------------------- #
    rows = []
    for T in month_starts(args.first, args.last):
        rows += count_instances(pos, items, T, windows, int(cfg["min_history"]))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{cfg['dataset']}_instances.csv", index=False)
    piv = df.pivot(index="T", columns="w_days", values="cold_instances")
    lines += [
        f"## Cold-target instances (min_history = {cfg['min_history']})\n",
        "| T | " + " | ".join(f"w={w}" for w in piv.columns) + " | eligible users | cold items |",
        "|---|" + "---|" * (len(piv.columns) + 2),
    ]
    meta = df.drop_duplicates("T").set_index("T")
    for T, r in piv.iterrows():
        lines.append(
            f"| {T} | " + " | ".join(f"{int(v):,}" for v in r.values)
            + f" | {meta.loc[T, 'eligible_users']:,} | {meta.loc[T, 'cold_items']:,} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / f"{cfg['dataset']}_audit.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
