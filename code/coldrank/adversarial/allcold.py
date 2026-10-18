"""A6 all-cold: gold + 49 cold negatives for the baseline (cold-target) instances.

Recency no longer separates the gold from the negatives. Negatives follow the
baseline's eligibility rules exactly (pools.build.eligible_cold): launched in
(T, target_ts], not in H_u or the user's window items, no near-duplicates of the
gold. Random and hard settings as in the baseline. Instances with fewer than 49
eligible items are dropped (counted)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coldrank.adversarial.common import COLD_NEG, GOLD, annotate
from coldrank.common import rng_for, ts_to_date
from coldrank.pools.build import Embeddings, candidate_record, cold_catalogue, eligible_cold, history_records, profile


def build_a6(instances: list[dict], items: pd.DataFrame, emb: Embeddings, pop: dict[str, int], T: int,
             cfg: dict) -> tuple[list[dict], dict]:
    n_neg = cfg["adversarial"]["a6_n_negs"]
    pc = cfg["pool"]
    it = items.set_index("item_id")
    cold_index, cold_t0 = cold_catalogue(items, T)
    pools, dropped = [], 0
    for r in instances:
        gold = r["target_item"]
        hist = [h[0] for h in r["history"]]
        excl = set(hist) | set(r["window_items"])
        elig, _ = eligible_cold(r, gold, excl, cold_index, cold_t0, emb, pc["dup_threshold"])
        if len(elig) < n_neg:
            dropped += 1
            continue
        prof = profile(emb, hist[-pc["profile_len"]:])
        sims_elig = emb[elig] @ prof
        for setting in pc["neg_settings"]:
            if setting == "random":
                negs = rng_for("a6_negs", r["instance_id"], setting).sample(elig, n_neg)
            else:  # hard
                order = np.lexsort((np.array(elig), -sims_elig))
                negs = [elig[k] for k in order[:n_neg]]
            cands = [gold] + negs
            sims = emb[cands] @ prof
            recs = []
            for c, s in zip(cands, sims):
                rec = candidate_record(c, it, is_cold=True, is_gold=(c == gold), sim=s, pop=pop.get(c, 0),
                                       warm_rank=None)
                rec["role"] = GOLD if c == gold else COLD_NEG
                recs.append(rec)
            recs.sort(key=lambda c: c["item_id"])
            base_id = f"{r['instance_id']}__{setting}"
            pool = {
                "pool_id": f"{base_id}__a6_allcold",
                "instance_id": r["instance_id"],
                "split": r["split"],
                "user_id": r["user_id"],
                "condition": "a6_allcold",
                "neg_setting": setting,
                "reference_pool_id": base_id,  # the baseline pool of the same instance (pairing)
                "cutoff_T": ts_to_date(T),
                "target_ts": r["target_ts"],
                "gold_item_id": gold,
                "history": history_records(r, it),
                "candidates": recs,
            }
            if n_neg + 1 != 50:  # EB-NeRD builds A6 at 15 directly (too few new articles for 49 negatives)
                pool["pool_size"] = n_neg + 1
            pools.append(annotate(pool))
    return pools, {"n_instances_in": len(instances), "dropped_too_few_cold": dropped}
