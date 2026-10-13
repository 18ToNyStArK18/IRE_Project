"""Prompt rendering for the five Li et al. methods and the pointwise control.

Li et al. do not publish their prompts; these are rebuilt from Sec. 2.2 / 3.2.1 of
the paper (design.md). Every method sees the same history/candidate formatting,
except that Structural Reasoning also shows the brand (Li et al. Sec. 2.2.1 give
it extra attributes for factor matching).

A displayed date of None means "hidden" (condition A3): the date field is omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from coldrank.common import rng_for
from coldrank.prompts.examples import EXAMPLE_1, EXAMPLE_2

METHODS = ("direct", "base", "fast", "structural", "structural_norecency", "ssc")

SYSTEM = (
    "You are an expert recommender system for an online video game store. "
    "You recommend items to a user based on their purchase history."
)

# Max new tokens per method, and the minimum we accept before trimming history.
GEN_CAP = {"direct": 80, "base": 768, "fast": 512, "structural": 1024, "structural_norecency": 1024, "ssc": 1024}
GEN_MIN = {"direct": 60, "base": 400, "fast": 300, "structural": 600, "structural_norecency": 600, "ssc": 600}
MIN_HISTORY = 5


TITLE_CHARS = 60  # Amazon titles run to 200 chars; display cut keeps 50 candidates within 4k tokens


def display_title(title: str) -> str:
    """Cut long titles at a word boundary (display only; pools keep full titles)."""
    title = " ".join(title.split())
    if len(title) <= TITLE_CHARS:
        return title
    cut = title[:TITLE_CHARS].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-|/(") or title[:TITLE_CHARS]


def display_category(cat: str) -> str:
    """'Legacy Systems > Nintendo Systems > Wii > Games' -> 'Wii > Games'.
    Drops the grouping levels ('Legacy Systems', '... Systems') and keeps <= 3 levels."""
    parts = [p for p in (cat or "").split(" > ") if p and p != "Legacy Systems" and not p.endswith(" Systems")]
    return " > ".join(parts[:3])


def _fields(it: dict, brand: bool) -> list[tuple[str, str]]:
    f = [("title", display_title(it["title"]))]
    if it.get("date"):
        f.append(("launch date", it["date"]))
    f.append(("category", display_category(it.get("category", "")) or "unknown"))
    if brand:
        f.append(("brand", it.get("brand") or "unknown"))
    return f


def item_line(it: dict, brand: bool) -> str:
    return " | ".join(v for _, v in _fields(it, brand))


def format_header(items: list[dict], brand: bool) -> str:
    names = [n for n, _ in _fields(items[0], brand)] if items else ["title", "category"]
    return f"(each item: {' | '.join(names)})"


def history_block(hist: list[dict], brand: bool) -> str:
    lines = "\n".join(f"- {item_line(h, brand)}" for h in hist)
    return f"The user's purchase history, from oldest to most recent {format_header(hist, brand)}:\n{lines}"


def candidate_block(cands: list[dict], brand: bool) -> str:
    lines = "\n".join(f"[{k}] {item_line(c, brand)}" for k, c in enumerate(cands, 1))
    return f"Candidate items {format_header(cands, brand)}:\n{lines}"


def final_line(ids: list[int]) -> str:
    return "FINAL: " + " > ".join(f"[{i}]" for i in ids)


def format_instruction(n: int) -> str:
    return (
        "End your answer with one line in exactly this format, listing the IDs of the "
        f"{n} candidates the user is most likely to purchase next, best first:\n"
        "FINAL: [id] > [id] > ... > [id]"
    )


GUIDELINES = (
    "Consider what the history says about the platforms the user owns, the kinds of items they buy, "
    "and the franchises and brands they like."
)

SR_STEPS = (
    "Work in three steps.\n"
    "Step 1: Identify the three most important factors behind the user's preferences (for example the "
    "platform, the type of game or item, the brand or franchise). For each factor, build a reasoning path "
    "by listing the history items that support it.\n"
    "Step 2: For each candidate that matches at least one path, give a match score from 0 to 10 for each "
    "path. Candidates you do not list score 0.\n"
    "Step 3: Give each path an importance weight, considering how prominent the factor is in the "
    "history{recency}. Compute each candidate's overall score as the weighted sum of its match scores, "
    "and rank the candidates by overall score."
)

SSC_STEPS = (
    "Think about the user's preferences from three different perspectives of your choice. For each "
    "perspective, write a short reasoning path and name the candidates it favours. Then combine the three "
    "paths into one overall judgement, reconciling them as you see fit rather than taking a majority vote."
)


def _example_block(ex: dict, kind: str, brand: bool) -> str:
    reasoning = ex[kind]
    final = ex[f"{kind}_final"]
    return (
        f"{history_block(ex['history'], brand)}\n\n{candidate_block(ex['candidates'], brand)}\n\n"
        f"Answer:\n{reasoning}\n{final_line(final)}"
    )


def _user_text(method: str, hist: list[dict], cands: list[dict], n: int) -> tuple[str, str]:
    """(user message, assistant prefill) for a listwise method."""
    brand = method.startswith("structural")
    task = f"{history_block(hist, brand)}\n\n{candidate_block(cands, brand)}"
    if method == "direct":
        return (
            f"{task}\n\nRank the candidates by how likely the user is to purchase them next. Do not explain. "
            f"Output only one line in exactly this format, listing the IDs of the {n} most likely candidates, "
            "best first:\nFINAL: [id] > [id] > ... > [id]",
            "FINAL: [",
        )
    if method in ("base", "fast"):
        exs = [EXAMPLE_1, EXAMPLE_2] if method == "base" else [EXAMPLE_1]
        kind = "base" if method == "base" else "fast"
        demo = "\n\n".join(f"Example {k}:\n{_example_block(ex, kind, False)}" for k, ex in enumerate(exs, 1))
        how = (
            "First, summarize the user's interests in two or three sentences. Then reason about which "
            "candidates fit these interests best."
            if method == "base"
            else "Briefly state the user's main interest and which candidate fits it best."
        )
        return (
            f"Here {'are worked examples' if len(exs) > 1 else 'is a worked example'} of the task.\n\n{demo}\n\n"
            f"Now the actual task.\n\n{task}\n\n{GUIDELINES} {how} {format_instruction(n)}",
            "",
        )
    if method in ("structural", "structural_norecency"):
        recency = ", the kind of interaction, and how recent it is" if method == "structural" else ""
        steps = SR_STEPS.format(recency=recency)
        demo = _example_block(EXAMPLE_1, "structural", True)
        return (
            f"Here is a worked example of the task.\n\nExample:\n{demo}\n\nNow the actual task.\n\n{task}\n\n"
            f"{steps} {format_instruction(n)}",
            "",
        )
    if method == "ssc":
        return f"{task}\n\n{SSC_STEPS} {format_instruction(n)}", ""
    raise ValueError(method)


@dataclass
class Rendered:
    system: str
    user: str
    prefill: str
    id_map: dict[int, str]          # prompt ID -> item_id
    n_history: int
    max_new_tokens: int = 0
    prompt_tokens: int = 0
    overflow: bool = False
    meta: dict = field(default_factory=dict)


def shuffled_candidates(pool: dict, shuffle: int) -> list[dict]:
    """Candidates in the order of one shuffle. Seeded by `shuffle_key` when present
    (variant pools point it at their reference pool), so a variant keeps every shared
    candidate in the same prompt position; baseline pools have no key and use pool_id."""
    cands = list(pool["candidates"])
    rng_for("shuffle", pool.get("shuffle_key", pool["pool_id"]), shuffle).shuffle(cands)
    return cands


def render_listwise(
    pool: dict,
    method: str,
    shuffle: int,
    history_len: int,
    top_n: int,
    count_tokens: Callable[[str, str, str], int] | None = None,
    max_model_len: int = 4096,
) -> Rendered:
    """Render one listwise prompt. `count_tokens(system, user, prefill)` returns the
    token length of the fully formatted prompt; if given, the oldest history items
    are dropped until the prompt leaves at least GEN_MIN[method] tokens to generate."""
    cands = shuffled_candidates(pool, shuffle)
    id_map = {k: c["item_id"] for k, c in enumerate(cands, 1)}
    hist_all = pool["history"][-history_len:]
    n_hist = len(hist_all)
    while True:
        hist = hist_all[len(hist_all) - n_hist:]
        user, prefill = _user_text(method, hist, cands, top_n)
        r = Rendered(SYSTEM, user, prefill, id_map, n_hist)
        if count_tokens is None:
            r.max_new_tokens = GEN_CAP[method]
            return r
        r.prompt_tokens = count_tokens(SYSTEM, user, prefill)
        room = max_model_len - r.prompt_tokens
        if room >= GEN_MIN[method] or n_hist <= MIN_HISTORY:
            r.max_new_tokens = max(0, min(GEN_CAP[method], room))
            r.overflow = room < GEN_MIN[method]
            return r
        n_hist -= 1


def render_pointwise(pool: dict, cand: dict, history_len: int) -> tuple[str, str]:
    """(system, user) for one candidate. Answer token is read from the next position."""
    hist = pool["history"][-history_len:]
    user = (
        f"{history_block(hist, False)}\n\nCandidate item {format_header([cand], False)}:\n{item_line(cand, False)}\n\n"
        "Will the user purchase this candidate item next? Answer Yes or No."
    )
    return SYSTEM, user


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_BRACKET = re.compile(r"\[\s*(\d{1,3})\s*\]")


def parse_ranking(text: str, n_cands: int) -> tuple[list[int], str]:
    """Extract candidate IDs (1..n_cands), de-duplicated, in order.

    Primary: the bracketed IDs after the LAST 'FINAL:'. Fallback: the last line with
    at least two bracketed IDs. Returns (ids, mode) with mode in
    {'final', 'fallback_line', 'fail'}.
    """
    def ids_from(s: str) -> list[int]:
        out, seen = [], set()
        for m in _BRACKET.finditer(s):
            k = int(m.group(1))
            if 1 <= k <= n_cands and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    pos = text.upper().rfind("FINAL:")
    if pos >= 0:
        tail = text[pos + len("FINAL:"):].split("\n\n")[0]
        ids = ids_from(tail)
        if ids:
            return ids, "final"
    for line in reversed(text.splitlines()):
        if len(_BRACKET.findall(line)) >= 2:
            ids = ids_from(line)
            if ids:
                return ids, "fallback_line"
    return [], "fail"
