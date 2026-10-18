"""Danish -> English translation of EB-NeRD article text, done once and cached.

Every distinct string is translated exactly once with one fixed model and revision, so
every condition, method and model sees identical English text (design.md, EB-NeRD).
The cache is a JSONL file of {"da": ..., "en": ...} lines, appended batch by batch, so
an interrupted run resumes where it stopped.

Usage (normally called by coldrank.data.ebnerd):
    python -m coldrank.data.translate --strings strings.txt --cache translations.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

MT_MODEL = "Helsinki-NLP/opus-mt-da-en"
MT_REVISION = "0960cff506758772cf23186b941305e0afeb43c5"
NUM_BEAMS = 1          # greedy: beam search (the model default, 4) took ~3 s per subtitle on CPU, ~30x slower
MAX_NEW_TOKENS = 256   # the longest subtitles are ~500 characters
BATCH = 32


def load_cache(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line of a killed run
                out[r["da"]] = r["en"]
    return out


def translate(strings: list[str], cache_path: Path, batch: int = BATCH) -> dict[str, str]:
    """{danish: english} for every string (empty strings map to themselves)."""
    cache = load_cache(cache_path)
    todo = sorted({s for s in strings if s.strip() and s not in cache}, key=len)  # similar lengths per batch
    if todo:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        torch.manual_seed(0)
        tok = MarianTokenizer.from_pretrained(MT_MODEL, revision=MT_REVISION)
        model = MarianMTModel.from_pretrained(MT_MODEL, revision=MT_REVISION).eval()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f, torch.inference_mode():
            for s in tqdm(range(0, len(todo), batch), desc="translate", unit=" batches"):
                chunk = todo[s : s + batch]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
                gen = model.generate(**enc, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                for da, en in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                    cache[da] = en.strip()
                    f.write(json.dumps({"da": da, "en": cache[da]}, ensure_ascii=False) + "\n")
                f.flush()
    return {s: (cache.get(s, s) if s.strip() else s) for s in strings}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strings", type=Path, required=True, help="one Danish string per line")
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    strings = [line.rstrip("\n") for line in open(args.strings)]
    out = translate(strings, args.cache)
    print(f"[translate] {len(out)} strings, cache {args.cache}")


if __name__ == "__main__":
    main()
