"""Paths, config loading, JSONL helpers, and deterministic seeding."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("COLDRANK_DATA", PROJECT_DIR / "data"))
OUT_DIR = Path(os.environ.get("COLDRANK_OUT", PROJECT_DIR / "outputs"))
CONFIG_DIR = CODE_DIR / "configs"


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by name (looked up in code/configs) or by path. A dataset config
    also sets the displayed date format (`date_format`, default date only)."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "dataset" in cfg:
        set_date_format(cfg.get("date_format", DATE_ONLY))
    return cfg


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)  # atomic: never leave a half-written file behind
    return n


def stable_hash(*parts: Any) -> str:
    """Deterministic hash (unlike Python's salted hash())."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x1f")
    return h.hexdigest()


def rng_for(*parts: Any) -> random.Random:
    """A random.Random seeded deterministically from arbitrary keys."""
    return random.Random(int(stable_hash(*parts)[:16], 16))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


DATE_ONLY = "%Y-%m-%d"           # Amazon: launch dates
DATE_TIME = "%Y-%m-%d %H:%M"     # EB-NeRD: news ages in hours, so the time is shown too
_date_format = DATE_ONLY


def set_date_format(fmt: str) -> None:
    """Displayed date format for ts_to_date (set by load_config from the dataset config)."""
    global _date_format
    if fmt not in (DATE_ONLY, DATE_TIME):
        raise ValueError(f"unsupported date_format {fmt!r}")
    _date_format = fmt


def ts_to_date(ts: int | float) -> str:
    """Unix seconds -> displayed date (UTC): 'YYYY-MM-DD', or 'YYYY-MM-DD HH:MM' with DATE_TIME."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(_date_format)


def date_to_ts(d: str) -> int:
    """'YYYY-MM-DD' (00:00) or 'YYYY-MM-DD HH:MM' -> Unix seconds, UTC."""
    fmt = DATE_TIME if len(d) > 10 else DATE_ONLY
    return int(datetime.strptime(d, fmt).replace(tzinfo=timezone.utc).timestamp())


def pool_files(pool_dir: str | Path, condition: str, split: str) -> dict[str, Path]:
    """{neg_setting: path} for every pool file `<condition>_<neg>_<split>.jsonl`.
    Baseline / A2-A4 / A6 have `random` and `hard`; the A1 family has `trap`."""
    out = {}
    for f in sorted(Path(pool_dir).glob(f"{condition}_*_{split}.jsonl")):
        neg = f.name[len(condition) + 1 : -len(f"_{split}.jsonl")]
        if neg and "_" not in neg:  # guards against prefix clashes, e.g. a5_n1 vs a5_n10
            out[neg] = f
    return out
