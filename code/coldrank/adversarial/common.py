"""Shared pieces for the adversarial variants: roles, the new-dated flag, deriving a
variant pool from a reference pool, and an embedding cache that extends the
baseline one without altering it."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.common import date_to_ts
from coldrank.pools.build import Embeddings, item_text

# Candidate roles
GOLD, WARM, COLD_NEG = "gold", "warm", "cold_neg"
TRAP, RELABELED, CTRL_FILL = "trap", "relabeled", "ctrl_fill"


def is_new_dated(c: dict, T: int) -> bool:
    """Does the candidate LOOK new, i.e. is its displayed date after the cutoff?"""
    return c.get("date") is not None and date_to_ts(c["date"]) > T


def annotate(pool: dict) -> dict:
    """Fill `role` (if missing, from the baseline flags) and `new_dated` for every
    candidate. Called on every variant pool after its dates are final."""
    T = date_to_ts(pool["cutoff_T"])
    for c in pool["candidates"]:
        if "role" not in c:
            c["role"] = GOLD if c["is_gold"] else (COLD_NEG if c["is_cold"] else WARM)
        c["new_dated"] = is_new_dated(c, T)
    return pool


def derive(ref: dict, condition: str) -> dict:
    """Deep copy of a reference pool as a new condition.

    The candidate list keeps the reference's order slot by slot and `shuffle_key`
    points at the reference's key, so the prompt shuffles put every shared candidate
    in the same position: a date variant changes the dates and nothing else."""
    q = copy.deepcopy(ref)
    q["condition"] = condition
    q["reference_pool_id"] = ref["pool_id"]
    q["shuffle_key"] = ref.get("shuffle_key", ref["pool_id"])
    q["pool_id"] = f"{ref['pool_id']}__{condition}"
    for c in q["candidates"]:
        if "role" not in c:
            c["role"] = GOLD if c["is_gold"] else (COLD_NEG if c["is_cold"] else WARM)
    return q


def extended_embeddings(items: pd.DataFrame, need: set[str], model_name: str, base_cache: Path,
                        adv_cache: Path) -> Embeddings:
    """Embeddings for `need`, reusing the baseline vectors unchanged and encoding only
    the missing items. The baseline cache is never rewritten, so baseline pools stay
    reproducible bit for bit."""
    if adv_cache.exists():
        z = np.load(adv_cache, allow_pickle=False)
        ids = z["ids"].tolist()
        if need <= set(ids):
            return Embeddings(ids, z["vecs"])
    z = np.load(base_cache, allow_pickle=False)
    ids, vecs = z["ids"].tolist(), z["vecs"]
    missing = sorted(need - set(ids))
    if missing:
        from sentence_transformers import SentenceTransformer

        texts = items.set_index("item_id").loc[missing].apply(item_text, axis=1).tolist()
        new = SentenceTransformer(model_name).encode(texts, batch_size=256, normalize_embeddings=True,
                                                     show_progress_bar=True)
        ids, vecs = ids + missing, np.vstack([vecs, new.astype(vecs.dtype)])
    np.savez(adv_cache, ids=np.array(ids), vecs=vecs)
    return Embeddings(ids, vecs)
