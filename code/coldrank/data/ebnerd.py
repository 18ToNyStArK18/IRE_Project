"""EB-NeRD (Ekstra Bladet news): raw parquet -> items / interactions / impressions parquet.

Same output schema as coldrank.data.amazon_vg, so the split, SASRec, pools, variants and
LLM code run unchanged:
  items.parquet         item_id, title, subtitle, category_path, brand, topics, dfa_ts, ...
                        (English text; the Danish originals are kept as *_da)
  interactions.parquet  user_id, item_id, rating (1.0 = click), ts
  impressions.parquet   user_id, ts, inview, clicked (for the A1 traps: shown, not clicked)

Choices (design.md, EB-NeRD):
  * Clicks from the train and validation histories and behaviors (test labels are not
    public). Every click is a positive (rating 1.0).
  * Launch time = published_time; the split then takes min(published, first click), the
    Amazon rule, because 525 clicks predate the listed publication time.
  * All times are floored to the minute, the displayed resolution (`date_format`), so a
    displayed time never differs from the stored one.
  * Title, subtitle and topics are translated Danish -> English once
    (coldrank.data.translate); categories use the fixed English names below.

Usage:
    python -m coldrank.data.ebnerd            # uses data/raw/ebnerd/small/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coldrank.common import DATA_DIR
from coldrank.data.translate import translate

RAW_DIR = DATA_DIR / "raw" / "ebnerd" / "small"
PROC_DIR = DATA_DIR / "processed" / "ebnerd"

# Ekstra Bladet section names (category_str) in English. Several are site-specific, so a
# fixed table is clearer than machine-translating single words.
CATEGORY_EN = {
    "nyheder": "news", "sport": "sport", "underholdning": "entertainment", "krimi": "crime",
    "side9": "page nine (glamour)", "auto": "local business", "forbrug": "consumer",
    "sex_og_samliv": "sex and relationships", "nationen": "readers' section", "musik": "music",
    "ferie": "travel", "play": "video", "biler": "cars", "penge": "money", "opinionen": "opinion",
    "haandvaerkeren": "DIY", "incoming": "miscellaneous", "vin": "wine", "plus": "premium",
    "horoskoper": "horoscopes", "om_ekstra_bladet": "about Ekstra Bladet", "video": "video",
    "dagsorden": "agenda", "bibliotek": "archive", "services": "services", "podcast": "podcast",
}
N_TOPICS = 3  # topics shown per article (Structural Reasoning's extra attribute)


def minute(ts: pd.Series) -> np.ndarray:
    """datetime64 -> Unix seconds, floored to the minute. EB-NeRD times are naive; they are
    all in the same (local) zone, so they are treated as UTC consistently."""
    s = pd.to_datetime(ts).astype("datetime64[s]").astype("int64")
    return (s // 60) * 60


def load_clicks(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clicks, imps = [], []
    for split in ("train", "validation"):
        h = pd.read_parquet(raw / split / "history.parquet",
                            columns=["user_id", "article_id_fixed", "impression_time_fixed"])
        e = h.explode(["article_id_fixed", "impression_time_fixed"]).dropna()
        clicks.append(pd.DataFrame({"user_id": e["user_id"].astype(str).values,
                                    "item_id": e["article_id_fixed"].astype("int64").astype(str).values,
                                    "ts": minute(e["impression_time_fixed"])}))
        b = pd.read_parquet(raw / split / "behaviors.parquet",
                            columns=["user_id", "impression_time", "article_ids_inview", "article_ids_clicked"])
        imps.append(pd.DataFrame({"user_id": b["user_id"].astype(str).values,
                                  "ts": minute(b["impression_time"]),
                                  "inview": [[str(x) for x in v] for v in b["article_ids_inview"]],
                                  "clicked": [[str(x) for x in v] for v in b["article_ids_clicked"]]}))
        e = b[["user_id", "impression_time", "article_ids_clicked"]].explode("article_ids_clicked").dropna()
        clicks.append(pd.DataFrame({"user_id": e["user_id"].astype(str).values,
                                    "item_id": e["article_ids_clicked"].astype("int64").astype(str).values,
                                    "ts": minute(e["impression_time"])}))
    c = pd.concat(clicks, ignore_index=True).drop_duplicates()
    c = c.sort_values(["user_id", "ts", "item_id"], kind="mergesort").reset_index(drop=True)
    inter = pd.DataFrame({"user_id": c["user_id"].astype("string"), "item_id": c["item_id"].astype("string"),
                          "rating": np.float32(1.0), "ts": c["ts"].astype("int64")})
    imp = pd.concat(imps, ignore_index=True).sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)
    return inter, imp


def load_items(raw: Path, cache: Path) -> pd.DataFrame:
    a = pd.read_parquet(raw / "articles.parquet",
                        columns=["article_id", "title", "subtitle", "category_str", "topics", "published_time",
                                 "article_type"])
    clean = lambda s: " ".join(str(s or "").split())  # noqa: E731
    title_da = a["title"].map(clean)
    sub_da = a["subtitle"].map(clean)
    topics_da = [[clean(t) for t in (v if v is not None else [])][:N_TOPICS] for v in a["topics"]]
    unknown = set(a["category_str"].dropna()) - set(CATEGORY_EN)
    if unknown:
        raise SystemExit(f"categories without an English name: {sorted(unknown)}")
    # short strings first, so titles and topics are done before the long subtitles
    strings = sorted({t for ts in topics_da for t in ts}) + sorted(set(title_da)) + sorted(set(sub_da))
    en = translate(strings, cache)
    return pd.DataFrame({
        "item_id": a["article_id"].astype("int64").astype(str).astype("string"),
        "title": title_da.map(en),
        "subtitle": sub_da.map(en),
        "category_path": a["category_str"].map(CATEGORY_EN).fillna(""),
        "brand": "",                      # no brand in news; Structural Reasoning shows topics instead
        "topics": [", ".join(en[t] for t in ts) for ts in topics_da],
        "dfa_ts": pd.array(minute(a["published_time"]), dtype="Int64"),  # launch time = publication
        "release_ts": pd.array([pd.NA] * len(a), dtype="Int64"),
        "title_da": title_da,
        "subtitle_da": sub_da,
        "category_da": a["category_str"].fillna(""),
        "article_type": a["article_type"],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROC_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inter, imp = load_clicks(args.raw_dir)
    inter.to_parquet(args.out_dir / "interactions.parquet", index=False)
    imp.to_parquet(args.out_dir / "impressions.parquet", index=False)
    print(f"[interactions] {len(inter):,} clicks, {inter['user_id'].nunique():,} users; "
          f"[impressions] {len(imp):,}")

    items = load_items(args.raw_dir, args.out_dir / "translations.jsonl")
    items.to_parquet(args.out_dir / "items.parquet", index=False)
    print(f"[items] {len(items):,} articles -> {args.out_dir / 'items.parquet'}")


if __name__ == "__main__":
    main()
