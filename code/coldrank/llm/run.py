"""Run one (model, method) over one pool file.

Listwise methods: each pool is ranked under `shuffles` candidate orders; each
output is parsed and the parsed rankings are combined by Borda count.
Pointwise: every candidate is scored by log p(Yes) - log p(No); no shuffles needed.

Outputs (outputs/<dataset>/<condition>/<neg>/<model>/<method>/<split>/):
  raw.jsonl    one row per (pool, shuffle) or per pool (pointwise)
  preds.jsonl  {"pool_id", "ranking"} - the input to coldrank.eval.report

Usage:
    python -m coldrank.llm.run --model llama2_7b --method direct --split dev --neg random --backend vllm
"""

from __future__ import annotations

import argparse
import json
import time

from coldrank.common import OUT_DIR, load_config, read_jsonl, write_jsonl
from coldrank.data.split import split_dir
from coldrank.eval.metrics import borda, sort_by_score
from coldrank.llm.chat import format_chat
from coldrank.llm.engine import make_engine
from coldrank.prompts.render import METHODS, parse_ranking, render_listwise, render_pointwise


def run_listwise(engine, family: str, pools: list[dict], method: str, cfg: dict, max_len: int) -> tuple[list, list]:
    pc = cfg["prompt"]
    hist_len = pc["history_len"]["structural" if method.startswith("structural") else method]
    count = lambda s, u, pf: engine.count_tokens(format_chat(family, s, u, pf))  # noqa: E731

    jobs = []
    for p in pools:
        for s in range(pc["shuffles"]):
            r = render_listwise(p, method, s, hist_len, pc["top_n"], count, max_len)
            jobs.append((p, s, r))
    runnable = [j for j in jobs if j[2].max_new_tokens > 0]
    texts = engine.generate(
        [format_chat(family, r.system, r.user, r.prefill) for _, _, r in runnable],
        [r.max_new_tokens for _, _, r in runnable],
    )
    out_text = {id(r): t for (_, _, r), t in zip(runnable, texts)}

    raw, per_pool = [], {}
    for p, s, r in jobs:
        text = out_text.get(id(r))
        full = (r.prefill + text) if text is not None else ""
        ids, mode = parse_ranking(full, len(r.id_map)) if text is not None else ([], "skipped_overflow")
        ranking = [r.id_map[i] for i in ids]
        raw.append({
            "pool_id": p["pool_id"], "shuffle": s, "output": full, "parse_mode": mode, "ranking": ranking,
            "prompt_tokens": r.prompt_tokens, "max_new_tokens": r.max_new_tokens, "n_history": r.n_history,
            "overflow": r.overflow, "id_map": r.id_map,
        })
        per_pool.setdefault(p["pool_id"], []).append(ranking)
    preds = [{"pool_id": pid, "ranking": borda(pid, rs, pc["top_n"]),
              "n_parsed": sum(bool(x) for x in rs)} for pid, rs in per_pool.items()]
    return raw, preds


def run_pointwise(engine, family: str, pools: list[dict], cfg: dict, max_len: int) -> tuple[list, list]:
    hist_len = cfg["prompt"]["history_len"]["pointwise"]
    prompts, index, n_trimmed = [], [], 0
    for p in pools:
        for c in p["candidates"]:
            h = hist_len
            while True:  # drop oldest history items until the prompt fits (needs 1 answer token)
                sys_, user = render_pointwise(p, c, h)
                prompt = format_chat(family, sys_, user)
                if engine.count_tokens(prompt) < max_len or h <= 1:
                    break
                h -= 1
            n_trimmed += h < min(hist_len, len(p["history"]))
            prompts.append(prompt)
            index.append((p["pool_id"], c["item_id"]))
    if n_trimmed:
        print(f"[pointwise] history trimmed for {n_trimmed}/{len(prompts)} prompts to fit {max_len} tokens")
    scores = engine.yes_no(prompts)
    by_pool: dict[str, dict[str, float]] = {}
    for (pid, iid), sc in zip(index, scores):
        by_pool.setdefault(pid, {})[iid] = sc
    raw = [{"pool_id": pid, "scores": sc} for pid, sc in by_pool.items()]
    preds = [{"pool_id": pid, "ranking": sort_by_score(pid, sc), "n_parsed": 1} for pid, sc in by_pool.items()]
    return raw, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True, choices=list(METHODS) + ["pointwise"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--neg", default="random")
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None, help="first N pools only (debugging)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = load_config("models")[args.model]
    if args.method != "pointwise" and not mcfg["listwise"]:
        raise SystemExit(f"{args.model} is pointwise-only (context {mcfg['max_model_len']})")
    pools = list(read_jsonl(split_dir(cfg) / "pools" / f"{args.condition}_{args.neg}_{args.split}.jsonl"))
    if args.limit:
        pools = pools[: args.limit]

    engine = make_engine(mcfg, args.backend, OUT_DIR / "cache", args.model)
    t = time.time()
    if args.method == "pointwise":
        raw, preds = run_pointwise(engine, mcfg["family"], pools, cfg, mcfg["max_model_len"])
    else:
        raw, preds = run_listwise(engine, mcfg["family"], pools, args.method, cfg, mcfg["max_model_len"])
    out = OUT_DIR / cfg["dataset"] / args.condition / args.neg / args.model / args.method / args.split
    write_jsonl(out / "raw.jsonl", raw)
    write_jsonl(out / "preds.jsonl", preds)
    summary = {"pools": len(pools), "seconds": round(time.time() - t, 1)}
    if args.method != "pointwise":
        modes = [r["parse_mode"] for r in raw]
        summary.update({m: modes.count(m) / len(modes) for m in sorted(set(modes))})
        summary["overflow_rate"] = sum(r["overflow"] for r in raw) / len(raw)
        summary["mean_prompt_tokens"] = sum(r["prompt_tokens"] for r in raw) / len(raw)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
