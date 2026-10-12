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
    """Load a YAML config by name (looked up in code/configs) or by path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


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


def ts_to_date(ts: int | float) -> str:
    """Unix seconds -> 'YYYY-MM-DD' (UTC)."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def date_to_ts(d: str) -> int:
    """'YYYY-MM-DD' -> Unix seconds at 00:00 UTC."""
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
