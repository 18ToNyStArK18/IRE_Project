"""Evaluate every preds.jsonl for one (condition, split): a table with 95% CIs.

Usage:
    python -m coldrank.eval.report --config amazon_vg --split dev
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from coldrank.common import OUT_DIR, load_config, pool_files, read_jsonl
from coldrank.data.split import split_dir
from coldrank.eval.metrics import pool_metrics, summarize
from coldrank.eval.stats import bootstrap_ci


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--condition", default="baseline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = OUT_DIR / cfg["dataset"] / args.condition

    rows, per_pool = [], []
    for neg, path in pool_files(split_dir(cfg) / "pools", args.condition, args.split).items():
        pools = {p["pool_id"]: p for p in read_jsonl(path)}
        for f in sorted((root / neg).glob(f"*/*/{args.split}/preds.jsonl")):
            model, method = f.parts[-4], f.parts[-3]
            preds = {r["pool_id"]: r["ranking"] for r in read_jsonl(f)}
            missing = set(pools) - set(preds)
            if missing:
                print(f"[warn] {neg}/{model}/{method}: {len(missing)} pools without predictions (scored as empty)")
            m = [pool_metrics(pools[pid], preds.get(pid, [])) for pid in sorted(pools)]
            s = summarize(m)
            r1, lo, hi = bootstrap_ci([x["recall1"] for x in m])
            rows.append({"neg": neg, "model": model, "method": method, "n": s["n"],
                         "recall1": r1, "r1_lo": lo, "r1_hi": hi, "ndcg10": s["ndcg10"],
                         "cold1": s["cold1"], "trap1": s["trap1"], "empty_rate": s["empty_rate"]})
            per_pool += [dict(x, neg=neg, model=model, method=method) for x in m]

    if not rows:
        raise SystemExit(f"no predictions found under {root}")
    df = pd.DataFrame(rows).sort_values(["neg", "model", "method"])
    out = root / f"report_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_pool).to_csv(out / "per_pool.csv", index=False)
    lines = [f"# {cfg['dataset']} / {args.condition} / {args.split}\n"]
    for neg, g in df.groupby("neg"):
        lines += [f"## negatives: {neg}\n",
                  "| model | method | n | Recall@1 [95% CI] | NDCG@10 | Cold@1 | Trap@1 | empty |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            trap = "–" if r.trap1 != r.trap1 else f"{r.trap1:.3f}"
            lines.append(f"| {r.model} | {r.method} | {r.n} | {r.recall1:.3f} [{r.r1_lo:.3f}, {r.r1_hi:.3f}] | "
                         f"{r.ndcg10:.3f} | {r.cold1:.3f} | {trap} | {r.empty_rate:.1%} |")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
