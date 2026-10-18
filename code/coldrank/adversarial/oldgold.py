"""A1 old-gold with traps, A1-ctrl, and A5 fresh flood (old-target instances).

A1      gold (warm, bought in the window, not in H_u) + 39 SASRec warm items
        + 10 traps: new items (launched in (T, target_ts]) the user never interacted
        with, with similarity to the user profile <= the `trap_quantile` quantile of
        that instance's eligible new items (Amazon), or, with `trap_source: inview`
        (EB-NeRD), new articles shown to the user before the target and not clicked.
        Recency and relevance point in opposite directions.
A1-ctrl the same pool with the 10 trap slots refilled, slot by slot, by SASRec warm
        ranks 40-49: no candidate looks new.
A5_n    the A1 pool with the n warm negatives least similar to the user relabeled
        with a new displayed date (copied from a random eligible new item whose
        displayed date is after the cutoff, so it lies in (T, target_ts]). Nested in n.

A1-ctrl and A5 keep A1's candidate order and shuffle key, so shared candidates sit
in the same prompt positions. Instances with too few warm items or traps are
dropped (counted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldrank.adversarial.common import CTRL_FILL, GOLD, RELABELED, TRAP, WARM, annotate, derive
from coldrank.common import date_to_ts, rng_for, ts_to_date
from coldrank.pools.build import Embeddings, candidate_record, cold_catalogue, eligible_cold, history_records, profile

NEG = "trap"  # the single negative setting of the A1 family


def build_oldgold(instances: list[dict], topk: dict[str, list[str]], items: pd.DataFrame, emb: Embeddings,
                  pop: dict[str, int], T: int, cfg: dict) -> tuple[dict[str, list[dict]], dict]:
    ac, pc = cfg["adversarial"], cfg["pool"]
    n_warm, n_traps, n_fill = ac["a1_n_warm"], ac["a1_n_traps"], ac["ctrl_fill"]
    it = items.set_index("item_id")
    t0 = it["t0"].astype("int64")
    cold_index, cold_t0 = cold_catalogue(items, T)
    out: dict[str, list[dict]] = {"a1_oldgold": [], "a1ctrl": []}
    for n in ac["flood_ns"]:
        out[f"a5_n{n}"] = []
    dropped = {"too_few_warm": 0, "too_few_traps": 0}

    for r in instances:
        iid, gold = r["instance_id"], r["target_item"]
        hist = [h[0] for h in r["history"]]
        excl = set(hist) | set(r["window_items"])  # includes the gold
        # SASRec order over warm items, gold included (to give it its true rank)
        ranked = [i for i in topk[iid] if i not in excl or i == gold]
        rank_of = {i: k + 1 for k, i in enumerate(ranked)}
        others = [i for i in ranked if i != gold]
        if len(others) < n_warm + n_fill:
            dropped["too_few_warm"] += 1
            continue
        warm, fill = others[:n_warm], others[n_warm:n_warm + n_fill]

        elig, _ = eligible_cold(r, gold, excl, cold_index, cold_t0, emb, pc["dup_threshold"])
        prof = profile(emb, hist[-pc["profile_len"]:])
        if len(elig) < n_traps:
            dropped["too_few_traps"] += 1
            continue
        sims_elig = emb[elig] @ prof
        # Items launched later on the cutoff day itself display the cutoff date and would
        # not LOOK new; traps and relabel dates must display a date after the cutoff.
        shows_new = [i for i in elig if date_to_ts(ts_to_date(int(t0[i]))) > T]
        shows_new_set = set(shows_new)
        if ac.get("trap_source") == "inview":
            # news: new articles the user was shown and did not click (split.attach_inview_not_clicked);
            # being skipped is the irrelevance evidence, so no similarity filter
            thr = None
            seen = set(r["inview_not_clicked"])
            low = [i for i in elig if i in seen and i in shows_new_set]
        else:
            thr = float(np.quantile(sims_elig, ac["trap_quantile"]))
            low = [i for i, s in zip(elig, sims_elig) if s <= thr and i in shows_new_set]
        if len(low) < n_traps:
            dropped["too_few_traps"] += 1
            continue
        traps = rng_for("a1_traps", iid).sample(low, n_traps)

        def rec(item: str, role: str) -> dict:
            x = candidate_record(item, it, is_cold=bool(t0[item] > T), is_gold=(item == gold),
                                 sim=float(emb[[item]][0] @ prof), pop=pop.get(item, 0),
                                 warm_rank=rank_of.get(item))
            x["role"] = role
            return x

        cands = [rec(gold, GOLD)] + [rec(i, WARM) for i in warm] + [rec(i, TRAP) for i in traps]
        cands.sort(key=lambda c: c["item_id"])  # neutral order; defines the slots shared by A1-ctrl / A5
        a1 = annotate({
            "pool_id": f"{iid}__{NEG}__a1_oldgold",
            "instance_id": iid,
            "split": r["split"],
            "user_id": r["user_id"],
            "condition": "a1_oldgold",
            "neg_setting": NEG,
            "cutoff_T": ts_to_date(T),
            "target_ts": r["target_ts"],
            "gold_item_id": gold,
            "trap_sim_threshold": None if thr is None else round(thr, 4),
            "history": history_records(r, it),
            "candidates": cands,
        })
        out["a1_oldgold"].append(a1)

        # A1-ctrl: trap slots (in slot order) <- fill items (in SASRec order)
        ctrl = derive(a1, "a1ctrl")
        ctrl["pool_id"] = f"{iid}__{NEG}__a1ctrl"
        fill_iter = iter(fill)
        ctrl["candidates"] = [rec(next(fill_iter), CTRL_FILL) if c["role"] == TRAP else c
                              for c in ctrl["candidates"]]
        out["a1ctrl"].append(annotate(ctrl))

        # A5: relabel the n least similar warm negatives (nested in n)
        order = sorted((c for c in a1["candidates"] if c["role"] == WARM),
                       key=lambda c: (c["sim_to_history"], c["item_id"]))
        for n in ac["flood_ns"]:
            q = derive(a1, f"a5_n{n}")
            q["pool_id"] = f"{iid}__{NEG}__a5_n{n}"
            relabel = {c["item_id"] for c in order[:n]}
            for c in q["candidates"]:
                if c["item_id"] in relabel:
                    donor = rng_for("a5_date", iid, c["item_id"]).choice(shows_new)
                    c["date"] = ts_to_date(int(t0[donor]))
                    c["role"] = RELABELED
            out[f"a5_n{n}"].append(annotate(q))

    return out, {"n_instances_in": len(instances), "dropped": dropped}
