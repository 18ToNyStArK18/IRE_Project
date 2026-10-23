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
import os
import time

from coldrank.common import OUT_DIR, load_config, read_jsonl, write_jsonl
from coldrank.data.split import split_dir
from coldrank.eval.metrics import borda, sort_by_score
from coldrank.llm.chat import format_chat
from coldrank.llm.engine import make_engine
from coldrank.prompts.render import METHODS, parse_ranking, render_listwise, render_pointwise, shuffled_candidates


def domain_of(cfg: dict) -> str:
    """Prompt wording for the dataset (prompts.render.DOMAINS); Amazon is the default."""
    return cfg["prompt"].get("domain", "video_games")


def run_listwise(engine, family: str, pools: list[dict], method: str, cfg: dict, max_len: int,
                 mcfg: dict | None = None) -> tuple[list, list]:
    mcfg = mcfg or {}
    pc = cfg["prompt"]
    hist_len = pc["history_len"][history_key(method)]
    count = lambda s, u, pf: engine.count_tokens(format_chat(family, s, u, pf))  # noqa: E731

    jobs = []
    for p in pools:
        for s in range(pc["shuffles"]):
            r = render_listwise(p, method, s, hist_len, pc["top_n"], count, max_len,
                                id_style=pc.get("id_style", "numbers"), domain=domain_of(cfg),
                                gen_cap=(mcfg.get("gen_cap") or {}).get(method))
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
        ids, mode = parse_ranking(full, len(r.id_map), r.code_map) if text is not None else ([], "skipped_overflow")
        ranking = [r.id_map[i] for i in ids]
        raw.append({
            "pool_id": p["pool_id"], "shuffle": s, "output": full, "parse_mode": mode, "ranking": ranking,
            "ids": ids, "prompt_tokens": r.prompt_tokens, "max_new_tokens": r.max_new_tokens,
            "n_history": r.n_history, "overflow": r.overflow, "id_map": r.id_map,
        })
        per_pool.setdefault(p["pool_id"], []).append(ranking)
    preds = [{"pool_id": pid, "ranking": borda(pid, rs, pc["top_n"]),
              "n_parsed": sum(bool(x) for x in rs)} for pid, rs in per_pool.items()]
    return raw, preds


def history_key(method: str) -> str:
    if method.startswith("structural"):
        return "structural"
    if method.startswith("direct"):
        return "direct"
    return method


def run_sliding(engine, family: str, pools: list[dict], cfg: dict, max_len: int,
                window: int = 20, step: int = 10) -> tuple[list, list]:
    """RankGPT-style sliding window over the full pool. Per shuffle, windows of `window`
    candidates are reranked from the bottom of the list to the top, moving by `step`; the
    model returns its top-N of each window, which move to the front of that window (unlisted
    items keep their relative order). The final top-N of the list is the shuffle's ranking.
    A shuffle whose windows all fail to parse yields an empty ranking."""
    pc = cfg["prompt"]
    hist_len = pc["history_len"]["direct"]
    count = lambda s, u, pf: engine.count_tokens(format_chat(family, s, u, pf))  # noqa: E731
    states = []
    for p in pools:
        for s in range(pc["shuffles"]):
            states.append({"pool": p, "shuffle": s, "order": shuffled_candidates(p, s), "parsed": 0})
    n_c = len(pools[0]["candidates"]) if pools else 0
    starts = list(range(max(n_c - window, 0), -1, -step))
    if starts and starts[-1] != 0:
        starts.append(0)
    raw = []
    for rnd, a in enumerate(starts):
        b = min(a + window, n_c)
        rendered = [render_listwise(st["pool"], "direct_sw", st["shuffle"], hist_len, pc["top_n"], count,
                                    max_len, cands=st["order"][a:b], id_style=pc.get("id_style", "numbers"),
                                    domain=domain_of(cfg))
                    for st in states]
        texts = engine.generate([format_chat(family, r.system, r.user, r.prefill) for r in rendered],
                                [r.max_new_tokens for r in rendered])
        for st, r, text in zip(states, rendered, texts):
            ids, mode = parse_ranking(text, len(r.id_map), r.code_map)
            win = st["order"][a:b]
            if ids:
                st["parsed"] += 1
                chosen = [win[i - 1] for i in ids]
                rest = [c for c in win if c["item_id"] not in {x["item_id"] for x in chosen}]
                st["order"][a:b] = chosen + rest
            raw.append({"pool_id": st["pool"]["pool_id"], "shuffle": st["shuffle"], "round": rnd,
                        "window": [a, b], "output": text, "parse_mode": mode, "ids": ids,
                        "ranking": [r.id_map[i] for i in ids], "prompt_tokens": r.prompt_tokens,
                        "max_new_tokens": r.max_new_tokens, "n_history": r.n_history, "overflow": r.overflow,
                        "id_map": r.id_map})
    per_pool = {}
    for st in states:
        final = [c["item_id"] for c in st["order"][: pc["top_n"]]] if st["parsed"] else []
        per_pool.setdefault(st["pool"]["pool_id"], []).append(final)
    preds = [{"pool_id": pid, "ranking": borda(pid, rs, pc["top_n"]),
              "n_parsed": sum(bool(x) for x in rs)} for pid, rs in per_pool.items()]
    return raw, preds


def degeneracy(raw: list[dict]) -> dict:
    """How much of the output is just the listing order? identity_rate: share of outputs
    whose first three IDs are [1], [2], [3]; top1_mode_share: share of the most common
    first ID (1/20 or 1/50 would be expected for a real ranking)."""
    outs = [r["ids"] for r in raw if r.get("ids")]
    if not outs:
        return {}
    firsts = [o[0] for o in outs]
    mode = max(set(firsts), key=firsts.count)
    return {"identity_rate": sum(o[:3] == [1, 2, 3] for o in outs) / len(outs),
            "top1_mode_id": mode, "top1_mode_share": firsts.count(mode) / len(outs)}


def run_pointwise(engine, family: str, pools: list[dict], cfg: dict, max_len: int) -> tuple[list, list]:
    hist_len = cfg["prompt"]["history_len"]["pointwise"]
    prompts, index, n_trimmed = [], [], 0
    for p in pools:
        for c in p["candidates"]:
            h = hist_len
            while True:  # drop oldest history items until the prompt fits (needs 1 answer token)
                sys_, user = render_pointwise(p, c, h, domain_of(cfg))
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


def run_task(engine, cfg: dict, mcfg: dict, model: str, method: str, condition: str, neg: str, split: str,
             limit: int | None = None, force: bool = False) -> dict:
    """Run one (method, condition, neg, split) and write raw / preds / run_summary.
    Skips the task if its run_summary.json exists (resume after a time-limit kill)."""
    out = OUT_DIR / cfg["dataset"] / condition / neg / model / method / split
    if (out / "run_summary.json").exists() and not force and not limit:
        print(f"[skip] {condition}/{neg}/{method}/{split}: already done")
        return json.loads((out / "run_summary.json").read_text())
    if method != "pointwise" and not mcfg["listwise"]:
        raise SystemExit(f"{model} is pointwise-only (context {mcfg['max_model_len']})")
    pools = list(read_jsonl(split_dir(cfg) / "pools" / f"{condition}_{neg}_{split}.jsonl"))
    if limit:
        pools = pools[:limit]
        out = out.parent / f"{split}_limit{limit}"  # never mistake a debug run for a full one
    t = time.time()
    if method == "pointwise":
        raw, preds = run_pointwise(engine, mcfg["family"], pools, cfg, mcfg["max_model_len"])
    elif method == "direct_sw":
        raw, preds = run_sliding(engine, mcfg["family"], pools, cfg, mcfg["max_model_len"])
    else:
        raw, preds = run_listwise(engine, mcfg["family"], pools, method, cfg, mcfg["max_model_len"], mcfg)
    write_jsonl(out / "raw.jsonl", raw)
    write_jsonl(out / "preds.jsonl", preds)
    summary = {"task": f"{condition}/{neg}/{method}/{split}", "pools": len(pools),
               "seconds": round(time.time() - t, 1)}
    if method != "pointwise" and raw:
        modes = [r["parse_mode"] for r in raw]
        summary.update({m: modes.count(m) / len(modes) for m in sorted(set(modes))})
        summary["overflow_rate"] = sum(r["overflow"] for r in raw) / len(raw)
        summary["mean_prompt_tokens"] = sum(r["prompt_tokens"] for r in raw) / len(raw)
        summary.update(degeneracy(raw))
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


def verify_pointwise(engine, family: str, cfg: dict, n: int, tol: float = 0.05, rel_tol: float = 0.005) -> float:
    """Compare the engine's fast yes_no() with yes_no_exact() on n real prompts from the
    first dev baseline pool; abort if they disagree."""
    pool = next(iter(read_jsonl(split_dir(cfg) / "pools" / "baseline_random_dev.jsonl")))
    hist_len = cfg["prompt"]["history_len"]["pointwise"]
    prompts = [format_chat(family, *render_pointwise(pool, c, hist_len, domain_of(cfg))) for c in pool["candidates"][:n]]
    fast, exact = engine.engine.yes_no(prompts), engine.engine.yes_no_exact(prompts)
    pairs = [(a, b) for a, b in zip(fast, exact) if b == b]  # exact is NaN if Yes/No not in top-20
    print(json.dumps({"verify_pointwise_n": n, "comparable": len(pairs), "fast": fast[:4], "exact": exact[:4]}))
    if len(pairs) < max(1, n // 2):
        raise SystemExit(f"only {len(pairs)}/{n} prompts had both Yes and No in the top-500: cannot verify")
    diff = max(abs(a - b) for a, b in pairs)
    # fp16 logits are rounded to steps that grow with their size (1/32 near 16), so the tolerance is
    # absolute + relative: Llama-2's log-odds are ~+5 (diffs <= 0.02), Qwen2.5-32B's ~+13..+16 (0.0625)
    excess = max(abs(a - b) - (tol + rel_tol * abs(b)) for a, b in pairs)
    print(json.dumps({"verify_pointwise_max_abs_diff": diff, "max_excess_over_tolerance": excess}))
    if excess > 0:
        raise SystemExit(f"fast pointwise scores disagree with exact ones (max diff {diff:.4f}; "
                         f"tolerance {tol} + {rel_tol} x |score|)")
    return diff


def claim(claim_dir: str, job: dict) -> bool:
    """Atomically claim a task for this worker. mkdir is atomic (also on NFS), so exactly
    one worker wins; the others skip the task."""
    name = "__".join(str(job[k]) for k in ("method", "condition", "neg", "split")) + (
        f"__limit{job['limit']}" if job.get("limit") else "")
    os.makedirs(claim_dir, exist_ok=True)
    try:
        os.mkdir(os.path.join(claim_dir, name))
        return True
    except FileExistsError:
        return False


def read_jobs(path: str) -> list[dict]:
    """Lines 'method condition neg split [limit]'; '#' comments and blank lines ignored."""
    jobs = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            f = line.split()
            jobs.append({"method": f[0], "condition": f[1], "neg": f[2], "split": f[3],
                         "limit": int(f[4]) if len(f) > 4 else None})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="amazon_vg")
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", choices=list(METHODS) + ["pointwise"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--neg", default="random")
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None, help="first N pools only (debugging)")
    ap.add_argument("--jobs-file", default=None,
                    help="run every task in this file (one model load); lines: method condition neg split [limit]")
    ap.add_argument("--cache-tag", default=None, help="separate cache file, e.g. per GPU worker")
    ap.add_argument("--claim-dir", default=None,
                    help="shared task queue: several workers read the SAME jobs file and each task is "
                         "run by whichever worker claims it first (atomic mkdir in this directory)")
    ap.add_argument("--verify-pointwise", type=int, default=0,
                    help="before pointwise tasks, check fast vs exact scores on N prompts")
    ap.add_argument("--force", action="store_true", help="rerun tasks that already have a run_summary.json")
    ap.add_argument("--tensor-parallel", type=int, default=None,
                    help="override the model's vllm.tensor_parallel_size (e.g. 1 for qwen25_32b on one 32 GB GPU)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = load_config("models")[args.model]
    if args.tensor_parallel:  # same checkpoint, different GPU split: outputs and cache keys are unchanged
        mcfg["vllm"] = {**(mcfg.get("vllm") or {}), "tensor_parallel_size": args.tensor_parallel}
    if args.jobs_file:
        jobs = read_jobs(args.jobs_file)
    elif args.method:
        jobs = [{"method": args.method, "condition": args.condition, "neg": args.neg, "split": args.split,
                 "limit": args.limit}]
    else:
        raise SystemExit("give --method or --jobs-file")

    engine = make_engine(mcfg, args.backend, OUT_DIR / "cache", args.model, args.cache_tag)
    if args.verify_pointwise and any(j["method"] == "pointwise" for j in jobs):
        verify_pointwise(engine, mcfg["family"], cfg, args.verify_pointwise)
    t0 = time.time()
    for k, j in enumerate(jobs, 1):
        if args.claim_dir and not claim(args.claim_dir, j):
            continue  # another worker has it
        print(f"[task {k}/{len(jobs)}] {j}", flush=True)
        run_task(engine, cfg, mcfg, args.model, j["method"], j["condition"], j["neg"], j["split"],
                 j["limit"], args.force)
    print(f"[done] {len(jobs)} tasks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
