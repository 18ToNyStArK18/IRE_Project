"""Recency-prior decomposition and the variant comparisons (report Sec. 3.8-3.9).

For every (model, method) with predictions, from per-instance outcomes:

  cold-target instances, per negative setting (random, hard)
    R_true, R_hide, R_swap, R_shuffle, R_allcold   Recall@1 per condition
    d_evidence      = R_hide - R_random            (R_random = 1/50; CI from R_hide)
    d_recency       = R_true - R_hide              paired (same instances), McNemar
    swap_sens       = R_true - R_swap              paired, McNemar
    shuffle_effect  = R_true - R_shuffle           paired, McNemar
    allcold_effect  = R_true - R_allcold           paired on the shared instances, McNemar
    d_evidence + d_recency == R_true - R_random exactly (same instance set, asserted)
  old-target instances (A1 family, negative setting "trap")
    trap_effect     = R_A1ctrl - R_A1              paired, McNemar
    flood_slope     = slope of Recall@1 over n in {0 (=A1), 1, 2, 5, 10}   bootstrap over instances
    flood_slope_trap= slope of Trap@1 over the same levels (DESCRIPTIVE ONLY: partly
                      mechanical, since relabeling turns items a ranker already liked into
                      "traps"; a date-blind random ranker gets 1/50 per relabeled item)
    flood_lift_n    = P(top-1 is one of the n relabeled items | A5_n)
                      - P(top-1 is one of the SAME items | A1)   paired, McNemar.
                      The date effect of the flood: exactly 0 for a ranker that ignores dates.
  across instance sets
    overcredit_gap  = R_baseline(neg) - R_A1       UNPAIRED bootstrap (different instances)

Missing predictions for a pool count as an empty ranking (Recall@1 = 0), as in
eval/report.py; a quantity is skipped when one of its conditions has no predictions.

Outputs (outputs/<dataset>/decomposition_<split>/):
  rates.csv          Recall@1 / Cold@1 / Trap@1 with 95% CIs per model x method x condition x neg
  decomposition.csv  every quantity above (value, lo, hi, p, n)
  summary.md         readable tables

Usage:
    python -m coldrank.eval.decompose --config amazon_vg --split test
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from coldrank.common import OUT_DIR, load_config, pool_files, read_jsonl
from coldrank.data.split import split_dir
from coldrank.eval.metrics import pool_metrics
from coldrank.eval.stats import (
    bootstrap_ci, bootstrap_slope, mcnemar_exact, paired_bootstrap_diff, unpaired_bootstrap_diff,
)

N_CANDIDATES = 50
R_RANDOM = 1 / N_CANDIDATES
COLD_FAMILY = ["baseline", "a3_hide", "a2_swap", "a4_shuffle", "a6_allcold"]
PAIRED_COLD = [  # (quantity, condition subtracted from baseline)
    ("d_recency", "a3_hide"),
    ("swap_sens", "a2_swap"),
    ("shuffle_effect", "a4_shuffle"),
    ("allcold_effect", "a6_allcold"),
]


def conditions(cfg: dict) -> list[str]:
    return COLD_FAMILY + ["a1_oldgold", "a1ctrl"] + [f"a5_n{n}" for n in cfg["adversarial"]["flood_ns"]]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_outcomes(cfg: dict, split: str) -> pd.DataFrame:
    """One row per (condition, neg, model, method, instance) with recall1 / cold1 / trap1."""
    rows = []
    pool_dir = split_dir(cfg) / "pools"
    for cond in conditions(cfg):
        for neg, path in pool_files(pool_dir, cond, split).items():
            pools = {p["pool_id"]: p for p in read_jsonl(path)}
            pred_root = OUT_DIR / cfg["dataset"] / cond / neg
            for f in sorted(pred_root.glob(f"*/*/{split}/preds.jsonl")):
                model, method = f.parts[-4], f.parts[-3]
                preds = {r["pool_id"]: r["ranking"] for r in read_jsonl(f)}
                for pid, p in pools.items():
                    ranking = preds.get(pid, [])
                    m = pool_metrics(p, ranking)
                    relabeled = tuple(sorted(c["item_id"] for c in p["candidates"] if c.get("role") == "relabeled"))
                    rows.append({
                        "condition": cond, "neg": neg, "model": model, "method": method,
                        "instance_id": p["instance_id"], "recall1": m["recall1"],
                        "cold1": np.nan if m["cold1"] is None else float(m["cold1"]),
                        "trap1": np.nan if m["trap1"] is None else float(m["trap1"]),
                        "empty": m["empty"],
                        "top_item": ranking[0] if ranking else None,
                        "relabeled": relabeled,
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Computation (pure functions on the outcome table)
# --------------------------------------------------------------------------- #
def _series(df: pd.DataFrame, cond: str, neg: str, col: str = "recall1") -> pd.Series:
    s = df[(df["condition"] == cond) & (df["neg"] == neg)].set_index("instance_id")[col]
    assert s.index.is_unique, f"duplicate instances in {cond}/{neg}"
    return s


def _paired_row(name: str, a: pd.Series, b: pd.Series, **key) -> dict:
    """a - b on the instances both have."""
    common = a.index.intersection(b.index)
    av, bv = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    d = paired_bootstrap_diff(av, bv)
    t = mcnemar_exact(av, bv)
    return dict(key, quantity=name, value=d["diff"], lo=d["lo"], hi=d["hi"], p=t["p"], n=len(common),
                test="paired bootstrap + McNemar")


def rates(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, method, cond, neg), g in df.groupby(["model", "method", "condition", "neg"], sort=True):
        r, lo, hi = bootstrap_ci(g["recall1"].to_numpy())
        cold = g["cold1"].dropna().to_numpy()
        trap = g["trap1"].dropna().to_numpy()
        out.append({
            "model": model, "method": method, "condition": cond, "neg": neg, "n": len(g),
            "recall1": r, "r1_lo": lo, "r1_hi": hi,
            "cold1": cold.mean() if len(cold) else np.nan,
            "trap1": trap.mean() if len(trap) else np.nan,
            "empty_rate": g["empty"].mean(),
        })
    return pd.DataFrame(out)


def decompose(df: pd.DataFrame, flood_ns: list[int]) -> pd.DataFrame:
    rows = []
    for (model, method), g in df.groupby(["model", "method"], sort=True):
        key = {"model": model, "method": method}
        have = set(zip(g["condition"], g["neg"]))

        # ---- cold-target instances ---------------------------------------- #
        for neg in ("random", "hard"):
            if ("baseline", neg) not in have:
                continue
            base = _series(g, "baseline", neg)
            k = dict(key, neg=neg)
            if ("a3_hide", neg) in have:
                hide = _series(g, "a3_hide", neg)
                r, lo, hi = bootstrap_ci(hide.to_numpy())
                rows.append(dict(k, quantity="d_evidence", value=r - R_RANDOM, lo=lo - R_RANDOM, hi=hi - R_RANDOM,
                                 p=np.nan, n=len(hide), test="bootstrap CI of R_hide, minus 1/50"))
                # d_evidence + d_recency = R_true - R_random holds exactly only if both
                # terms are computed on the same instance set
                assert set(hide.index) == set(base.index), "baseline and A3 must share instances"
            for name, cond in PAIRED_COLD:
                if (cond, neg) in have:
                    rows.append(_paired_row(name, base, _series(g, cond, neg), **k))

        # ---- old-target instances (A1 family) ----------------------------- #
        k = dict(key, neg="trap")
        if ("a1_oldgold", "trap") in have:
            a1 = _series(g, "a1_oldgold", "trap")
            if ("a1ctrl", "trap") in have:
                rows.append(_paired_row("trap_effect", _series(g, "a1ctrl", "trap"), a1, **k))
            levels = [0] + [n for n in flood_ns if (f"a5_n{n}", "trap") in have]
            if len(levels) > 1:
                for col, name in (("recall1", "flood_slope"), ("trap1", "flood_slope_trap")):
                    cols = [_series(g, "a1_oldgold", "trap", col)] + \
                           [_series(g, f"a5_n{n}", "trap", col) for n in levels[1:]]
                    Y = pd.concat(cols, axis=1, join="inner").dropna()
                    if Y.empty:  # no outcomes for this metric (e.g. Trap@1 not defined)
                        continue
                    s = bootstrap_slope(Y.to_numpy(), levels)
                    rows.append(dict(k, quantity=name, value=s["slope"], lo=s["lo"], hi=s["hi"], p=np.nan,
                                     n=len(Y), test=f"bootstrap over instances; n levels {levels}"))
            a1_top = g[(g["condition"] == "a1_oldgold") & (g["neg"] == "trap")].set_index("instance_id")["top_item"]
            for n in levels[1:]:
                a5 = g[(g["condition"] == f"a5_n{n}") & (g["neg"] == "trap")].set_index("instance_id")
                common = a5.index.intersection(a1_top.index)
                rel = a5.loc[common, "relabeled"]
                in_a5 = pd.Series([t in r for t, r in zip(a5.loc[common, "top_item"], rel)], index=common, dtype=float)
                in_a1 = pd.Series([t in r for t, r in zip(a1_top.loc[common], rel)], index=common, dtype=float)
                row = _paired_row(f"flood_lift_n{n}", in_a5, in_a1, **k)
                rows.append(row)
            for neg in ("random", "hard"):
                if ("baseline", neg) in have:
                    d = unpaired_bootstrap_diff(_series(g, "baseline", neg).to_numpy(), a1.to_numpy())
                    rows.append(dict(key, neg=f"{neg}-vs-trap", quantity="overcredit_gap", value=d["diff"],
                                     lo=d["lo"], hi=d["hi"], p=np.nan, n=len(a1),
                                     test="UNPAIRED bootstrap (different instances)"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(r) -> str:
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "–"
    return f"{r.value:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" + ("" if math.isnan(r.p) else (" *" if r.p < 0.05 else ""))


def summary_md(dec: pd.DataFrame, rt: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}\n",
             "Values are differences in Recall@1 with 95% bootstrap CIs; * = McNemar p < 0.05 (paired quantities).\n"]
    for neg in ("random", "hard"):
        sub = dec[dec["neg"] == neg]
        if sub.empty:
            continue
        lines += [f"## Cold-target instances, {neg} negatives\n",
                  "| model | method | R_true | R_hide | Δevidence | Δrecency | swap sens. | shuffle eff. | all-cold eff. |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for (model, method), g in sub.groupby(["model", "method"]):
            q = {r.quantity: r for r in g.itertuples()}
            rr = rt[(rt.model == model) & (rt.method == method) & (rt.neg == neg)].set_index("condition")["recall1"]
            lines.append(f"| {model} | {method} | {rr.get('baseline', float('nan')):.3f} | "
                         f"{rr.get('a3_hide', float('nan')):.3f} | " + " | ".join(
                             _fmt(q.get(n)) for n in ("d_evidence", "d_recency", "swap_sens", "shuffle_effect",
                                                      "allcold_effect")) + " |")
        lines.append("")
    sub = dec[dec["neg"] == "trap"]
    lifts = [int(q[len("flood_lift_n"):]) for q in sub["quantity"].unique() if q.startswith("flood_lift_n")]
    max_n = max(lifts) if lifts else 10
    if not sub.empty:
        lines += ["## Old-target instances (A1 family)\n",
                  "| model | method | R_A1 | R_A1-ctrl | trap effect | flood slope (R@1 per trap) "
                  f"| flood lift, n = {max_n} | flood slope (Trap@1, descriptive) |",
                  "|---|---|---|---|---|---|---|---|"]
        for (model, method), g in sub.groupby(["model", "method"]):
            q = {r.quantity: r for r in g.itertuples()}
            rr = rt[(rt.model == model) & (rt.method == method) & (rt.neg == "trap")].set_index("condition")["recall1"]
            lines.append(f"| {model} | {method} | {rr.get('a1_oldgold', float('nan')):.3f} | "
                         f"{rr.get('a1ctrl', float('nan')):.3f} | " + " | ".join(
                             _fmt(q.get(n)) for n in ("trap_effect", "flood_slope", f"flood_lift_n{max_n}",
                                                      "flood_slope_trap")) + " |")
        lines.append("")
    gap = dec[dec["quantity"] == "overcredit_gap"]
    if not gap.empty:
        lines += ["## Over-crediting gap R_baseline − R_A1 (different instances, unpaired)\n",
                  "| model | method | baseline neg | gap |", "|---|---|---|---|"]
        for r in gap.itertuples():
            lines.append(f"| {r.model} | {r.method} | {r.neg.split('-')[0]} | {_fmt(r)} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = load_outcomes(cfg, args.split)
    if df.empty:
        raise SystemExit("no predictions found")
    rt = rates(df)
    dec = decompose(df, cfg["adversarial"]["flood_ns"])
    out = OUT_DIR / cfg["dataset"] / f"decomposition_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    rt.to_csv(out / "rates.csv", index=False)
    dec.to_csv(out / "decomposition.csv", index=False)
    md = summary_md(dec, rt, f"{cfg['dataset']} / {args.split}: recency-prior decomposition")
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
