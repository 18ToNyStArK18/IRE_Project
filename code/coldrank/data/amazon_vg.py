"""Amazon Reviews 2023, Video Games: raw JSONL -> items / interactions parquet.

No filtering happens here (beyond dropping unparseable rows); every filtering
decision lives in the split (coldrank.data.split) and its config, so it is logged
in one place.

Usage:
    python -m coldrank.data.amazon_vg            # uses data/raw/amazon_vg/*.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from coldrank.common import DATA_DIR

RAW_DIR = DATA_DIR / "raw" / "amazon_vg"
PROC_DIR = DATA_DIR / "processed" / "amazon_vg"

_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%mM %d, %Y", "%Y-%m-%d", "%d %B %Y", "%B %Y")  # "%mM": e.g. "04M 26, 2018"


def parse_amazon_date(s: object) -> int | None:
    """'October 2, 2001' -> Unix seconds (UTC midnight). None if unparseable."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def category_path(categories: object) -> str:
    """['Video Games', 'PlayStation 4', 'Games'] -> 'PlayStation 4 > Games'.

    The leading 'Video Games' is the same for every item and is dropped. The path
    carries both the platform and the item type, and plays the role of 'genre'.
    """
    if not isinstance(categories, list):
        return ""
    parts = [str(c).strip() for c in categories if str(c).strip()]
    if parts and parts[0] == "Video Games":
        parts = parts[1:]
    return " > ".join(parts)


def load_items(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in tqdm(f, desc="meta", unit=" items"):
            r = json.loads(line)
            details = r.get("details") or {}
            if not isinstance(details, dict):
                details = {}
            title = (r.get("title") or "").strip()
            store = (r.get("store") or "").strip() if isinstance(r.get("store"), str) else ""
            brand = details.get("Brand") if isinstance(details.get("Brand"), str) else ""
            rows.append(
                {
                    "item_id": r["parent_asin"],
                    "title": " ".join(title.split()),  # collapse internal whitespace/newlines
                    "store": store,
                    "brand": (brand or store).strip(),
                    "main_category": r.get("main_category") or "",
                    "category_path": category_path(r.get("categories")),
                    "dfa_ts": parse_amazon_date(details.get("Date First Available")),
                    "release_ts": parse_amazon_date(details.get("Release date")),
                    "dfa_raw": details.get("Date First Available"),
                }
            )
    df = pd.DataFrame(rows)
    n_dup = df["item_id"].duplicated().sum()
    if n_dup:
        print(f"[items] dropping {n_dup} duplicate parent_asin rows (keeping first)")
        df = df.drop_duplicates("item_id", keep="first")
    for c in ("dfa_ts", "release_ts"):
        df[c] = df[c].astype("Int64")
    return df.reset_index(drop=True)


def load_interactions(path: Path) -> pd.DataFrame:
    users, items, ratings, tss = [], [], [], []
    with open(path) as f:
        for line in tqdm(f, desc="reviews", unit=" reviews"):
            r = json.loads(line)
            users.append(r["user_id"])
            items.append(r["parent_asin"])
            ratings.append(float(r["rating"]))
            tss.append(int(r["timestamp"]) // 1000)  # ms -> s
    return pd.DataFrame(
        {
            "user_id": pd.Series(users, dtype="string"),
            "item_id": pd.Series(items, dtype="string"),
            "rating": pd.Series(ratings, dtype="float32"),
            "ts": pd.Series(tss, dtype="int64"),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROC_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.raw_dir / "meta_Video_Games.jsonl")
    items.to_parquet(args.out_dir / "items.parquet", index=False)
    print(f"[items] {len(items):,} items -> {args.out_dir / 'items.parquet'}")

    inter = load_interactions(args.raw_dir / "Video_Games.jsonl")
    inter.to_parquet(args.out_dir / "interactions.parquet", index=False)
    print(f"[interactions] {len(inter):,} reviews -> {args.out_dir / 'interactions.parquet'}")


if __name__ == "__main__":
    main()
