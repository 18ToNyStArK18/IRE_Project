"""Prompt rendering for the five Li et al. methods and the pointwise control.

Li et al. do not publish their prompts; these are rebuilt from Sec. 2.2 / 3.2.1 of
the paper (design.md). Every method sees the same history/candidate formatting,
except that Structural Reasoning also shows the brand (Li et al. Sec. 2.2.1 give
it extra attributes for factor matching).

A displayed date of None means "hidden" (condition A3): the date field is omitted.

All domain wording (system prompt, "purchase history" vs "reading history", field labels,
guidelines, worked examples) lives in a `Domain`; config `prompt.domain` selects it.
VIDEO_GAMES (the default) reproduces the Amazon prompts byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from coldrank.common import rng_for
from coldrank.prompts.examples import EXAMPLE_1, EXAMPLE_2, NEWS_EXAMPLE_1, NEWS_EXAMPLE_2

METHODS = ("direct", "direct_v2", "direct_sw", "base", "fast", "structural", "structural_norecency", "ssc")

SYSTEM = (
    "You are an expert recommender system for an online video game store. "
    "You recommend items to a user based on their purchase history."
)

# Max new tokens per method, and the minimum we accept before trimming history.
GEN_CAP = {"direct": 80, "direct_v2": 80, "direct_sw": 80, "base": 768, "fast": 512, "structural": 1024, "structural_norecency": 1024, "ssc": 1024}
GEN_MIN = {"direct": 60, "direct_v2": 60, "direct_sw": 60, "base": 400, "fast": 300, "structural": 600, "structural_norecency": 600, "ssc": 600}
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


def cut(text: str, n: int) -> str:
    """Cut text at a word boundary to at most n characters (display only)."""
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    c = text[:n].rsplit(" ", 1)[0]
    return c.rstrip(" ,;:-|/(") or text[:n]


def _fields(it: dict, brand: bool, dom: "Domain") -> list[tuple[str, str]]:
    f = [("title", display_title(it["title"]) if dom.title_chars == TITLE_CHARS else cut(it["title"], dom.title_chars))]
    if it.get("date"):
        f.append((dom.date_label, it["date"]))
    f.append((dom.category_label, display_category(it.get("category", "")) or "unknown"))
    if dom.summary_chars:
        f.append(("summary", cut(it.get("subtitle", ""), dom.summary_chars) or "none"))
    if brand:
        f.append((dom.extra_label, it.get(dom.extra_key) or "unknown"))
    return f


def item_line(it: dict, brand: bool, dom: "Domain") -> str:
    return " | ".join(v for _, v in _fields(it, brand, dom))


def format_header(items: list[dict], brand: bool, dom: "Domain") -> str:
    names = [n for n, _ in _fields(items[0], brand, dom)] if items else ["title", dom.category_label]
    return f"(each {dom.noun}: {' | '.join(names)})"


def history_block(hist: list[dict], brand: bool, dom: "Domain") -> str:
    lines = "\n".join(f"- {item_line(h, brand, dom)}" for h in hist)
    return f"The user's {dom.history}, from oldest to most recent {format_header(hist, brand, dom)}:\n{lines}"


# Letter codes (prompt.id_style: codes). Numbered IDs made models copy the prompt order
# ("[1] > [2] > ..." in 90-100% of outputs); two-letter codes cut this sharply (teammates'
# prompt experiment). Candidates get consonant pairs; worked examples get vowel pairs, which
# never label a real candidate, so a copied example answer cannot parse as a ranking.
CODE_LETTERS = "BCDFGHJKLMNPQRSTVWXZ"
EXAMPLE_CODES = ["AE", "IO", "UA", "EO", "OI"]


def candidate_codes(key: str, shuffle: int, cands: list[dict]) -> list[str]:
    pairs = [a + b for a in CODE_LETTERS for b in CODE_LETTERS if a != b]
    return rng_for("codes", key, shuffle, [c["item_id"] for c in cands]).sample(pairs, len(cands))


def candidate_block(cands: list[dict], brand: bool, dom: "Domain", labels: list[str] | None = None) -> str:
    labels = labels or [str(k) for k in range(1, len(cands) + 1)]
    lines = "\n".join(f"[{lab}] {item_line(c, brand, dom)}" for lab, c in zip(labels, cands))
    return f"Candidate {dom.nouns} {format_header(cands, brand, dom)}:\n{lines}"


def final_line(ids: list[int], labels: list[str] | None = None) -> str:
    return "FINAL: " + " > ".join(f"[{labels[i - 1] if labels else i}]" for i in ids)


def format_instruction(n: int, dom: "Domain", codes: bool = False) -> str:
    tok = "code" if codes else "id"
    return (
        f"End your answer with one line in exactly this format, listing the {'codes' if codes else 'IDs'} of the "
        f"{n} candidates the user is most likely to {dom.verb} next, best first:\n"
        f"FINAL: [{tok}] > [{tok}] > ... > [{tok}]"
    )


GUIDELINES = (
    "Consider what the history says about the platforms the user owns, the kinds of items they buy, "
    "and the franchises and brands they like."
)

SR_STEPS = (
    "Work in three steps.\n"
    "Step 1: Identify the three most important factors behind the user's preferences (for example {factors}). "
    "For each factor, build a reasoning path "
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


@dataclass(frozen=True)
class Domain:
    """Everything in a prompt that depends on the dataset."""
    system: str
    noun: str                 # "item" / "article"
    nouns: str
    history: str              # "purchase history" / "reading history"
    verb: str                 # "purchase" / "read" (next)
    date_label: str
    category_label: str
    extra_label: str          # Structural Reasoning's extra attribute
    extra_key: str
    title_chars: int
    summary_chars: int        # 0: no summary field
    guidelines: str
    sr_factors: str           # the factor examples in Structural Reasoning's step 1
    examples: tuple           # (example 1, example 2)


VIDEO_GAMES = Domain(
    system=SYSTEM, noun="item", nouns="items", history="purchase history", verb="purchase",
    date_label="launch date", category_label="category", extra_label="brand", extra_key="brand",
    title_chars=TITLE_CHARS, summary_chars=0, guidelines=GUIDELINES,
    sr_factors="the platform, the type of game or item, the brand or franchise",
    examples=(EXAMPLE_1, EXAMPLE_2),
)

NEWS = Domain(
    system=(
        "You are an expert recommender system for a Danish news website. The articles have been translated "
        "into English. You recommend articles to a user based on their reading history."
    ),
    noun="article", nouns="articles", history="reading history", verb="read",
    date_label="published", category_label="section", extra_label="topics", extra_key="topics",
    title_chars=100, summary_chars=100,
    guidelines=(
        "Consider what the history says about the sections and topics the user reads, the stories and "
        "people they follow, and the kind of articles they open."
    ),
    sr_factors="the section, the topic, the story or people involved",
    examples=(NEWS_EXAMPLE_1, NEWS_EXAMPLE_2),
)

DOMAINS = {"video_games": VIDEO_GAMES, "news": NEWS}


def _example_block(ex: dict, kind: str, brand: bool, dom: Domain, codes: bool = False) -> str:
    reasoning = ex[kind]
    final = ex[f"{kind}_final"]
    labels = EXAMPLE_CODES[: len(ex["candidates"])] if codes else None
    if codes:  # the example reasoning refers to candidates as [k]; rewrite to the example codes
        for k, lab in enumerate(labels, 1):
            reasoning = reasoning.replace(f"[{k}]", f"[{lab}]")
    return (
        f"{history_block(ex['history'], brand, dom)}\n\n{candidate_block(ex['candidates'], brand, dom, labels)}\n\n"
        f"Answer:\n{reasoning}\n{final_line(final, labels)}"
    )


def _user_text(method: str, hist: list[dict], cands: list[dict], n: int,
               labels: list[str] | None = None, dom: Domain = VIDEO_GAMES) -> tuple[str, str]:
    """(user message, assistant prefill) for a listwise method. `labels` = letter codes
    (id_style: codes) or None for numbered IDs."""
    brand = method.startswith("structural")
    codes = labels is not None
    task = f"{history_block(hist, brand, dom)}\n\n{candidate_block(cands, brand, dom, labels)}"
    if codes and method in ("direct", "direct_v2", "direct_sw"):
        # teammates' frozen `direct_codes_free`: letter codes, no prefill
        return (
            f"{task}\n\nRank the candidates by how likely the user is to {dom.verb} them next. Do not explain. "
            f"Output only one line in exactly this format, listing the codes of the {n} most likely candidates, "
            "best first:\nFINAL: [code] > [code] > ... > [code]",
            "",
        )
    if method == "direct":
        return (
            f"{task}\n\nRank the candidates by how likely the user is to {dom.verb} them next. Do not explain. "
            f"Output only one line in exactly this format, listing the IDs of the {n} most likely candidates, "
            "best first:\nFINAL: [id] > [id] > ... > [id]",
            "FINAL: [",
        )
    if method in ("direct_v2", "direct_sw"):
        # v2: no prefill and a scrambled example. With the [id] template and the "FINAL: ["
        # prefill, Mistral-7B answered "[1] > [2] > ... > [10]" for 446/450 dev prompts.
        ex = " > ".join(f"[{i}]" for i in (12, 3, 17, 8, 5)[: min(n, 5)])
        return (
            f"{task}\n\nRank the candidates by how likely the user is to {dom.verb} them next, judging each "
            f"candidate by how well it fits the user's {dom.history}. The candidates are listed in random "
            "order: do not simply copy that order. Do not explain. Output only one line with the IDs of the "
            f"{n} most likely candidates, best first, for example:\nFINAL: {ex} > ...",
            "",
        )
    if method in ("base", "fast"):
        exs = list(dom.examples) if method == "base" else [dom.examples[0]]
        kind = "base" if method == "base" else "fast"
        demo = "\n\n".join(f"Example {k}:\n{_example_block(ex, kind, False, dom, codes)}" for k, ex in enumerate(exs, 1))
        how = (
            "First, summarize the user's interests in two or three sentences. Then reason about which "
            "candidates fit these interests best."
            if method == "base"
            else "Briefly state the user's main interest and which candidate fits it best."
        )
        return (
            f"Here {'are worked examples' if len(exs) > 1 else 'is a worked example'} of the task.\n\n{demo}\n\n"
            f"Now the actual task.\n\n{task}\n\n{dom.guidelines} {how} {format_instruction(n, dom, codes)}",
            "",
        )
    if method in ("structural", "structural_norecency"):
        recency = ", the kind of interaction, and how recent it is" if method == "structural" else ""
        steps = SR_STEPS.format(recency=recency, factors=dom.sr_factors)
        demo = _example_block(dom.examples[0], "structural", True, dom, codes)
        return (
            f"Here is a worked example of the task.\n\nExample:\n{demo}\n\nNow the actual task.\n\n{task}\n\n"
            f"{steps} {format_instruction(n, dom, codes)}",
            "",
        )
    if method == "ssc":
        return f"{task}\n\n{SSC_STEPS} {format_instruction(n, dom, codes)}", ""
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
    code_map: dict[str, int] | None = None   # letter code -> prompt position (id_style: codes)


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
    cands: list[dict] | None = None,
    id_style: str = "numbers",
    domain: str = "video_games",
    gen_cap: int | None = None,
) -> Rendered:
    """Render one listwise prompt. `count_tokens(system, user, prefill)` returns the
    token length of the fully formatted prompt; if given, the oldest history items
    are dropped until the prompt leaves at least GEN_MIN[method] tokens to generate."""
    if cands is None:  # sliding windows pass their own (already ordered) candidate subset
        cands = shuffled_candidates(pool, shuffle)
    id_map = {k: c["item_id"] for k, c in enumerate(cands, 1)}
    labels = (candidate_codes(pool.get("shuffle_key", pool["pool_id"]), shuffle, cands)
              if id_style == "codes" else None)
    code_map = {lab: k for k, lab in enumerate(labels, 1)} if labels else None
    dom = DOMAINS[domain]
    hist_all = pool["history"][-history_len:]
    n_hist = len(hist_all)
    while True:
        hist = hist_all[len(hist_all) - n_hist:]
        user, prefill = _user_text(method, hist, cands, top_n, labels, dom)
        r = Rendered(dom.system, user, prefill, id_map, n_hist, code_map=code_map)
        if count_tokens is None:
            r.max_new_tokens = gen_cap or GEN_CAP[method]
            return r
        r.prompt_tokens = count_tokens(dom.system, user, prefill)
        room = max_model_len - r.prompt_tokens
        if room >= GEN_MIN[method] or n_hist <= MIN_HISTORY:
            r.max_new_tokens = max(0, min(gen_cap or GEN_CAP[method], room))
            r.overflow = room < GEN_MIN[method]
            return r
        n_hist -= 1


def render_pointwise(pool: dict, cand: dict, history_len: int, domain: str = "video_games") -> tuple[str, str]:
    """(system, user) for one candidate. Answer token is read from the next position."""
    dom = DOMAINS[domain]
    hist = pool["history"][-history_len:]
    user = (
        f"{history_block(hist, False, dom)}\n\nCandidate {dom.noun} {format_header([cand], False, dom)}:\n"
        f"{item_line(cand, False, dom)}\n\n"
        f"Will the user {dom.verb} this candidate {dom.noun} next? Answer Yes or No."
    )
    return dom.system, user


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #
_BRACKET = re.compile(r"\[\s*(\d{1,3})\s*\]")


_CODE = re.compile(r"\[\s*([A-Za-z]{2})\s*\]")
_BARE_CODE = re.compile(r"\b([A-Z]{2})\b")  # only accepted after FINAL:, and only if it labels a candidate


def parse_ranking(text: str, n_cands: int, code_map: dict[str, int] | None = None) -> tuple[list[int], str]:
    """Extract candidate positions (1..n_cands), de-duplicated, in order.

    With `code_map` (letter codes), bracketed codes are mapped to positions; codes that
    label no candidate (e.g. copied from the worked example) are ignored.
    Primary: the bracketed IDs after the LAST 'FINAL:' (with letter codes, bare codes
    there too: mode 'final_bare'). Fallback: the last line with at least two bracketed
    IDs. Returns (ids, mode) with mode in {'final', 'final_bare', 'fallback_line', 'fail'}.
    """
    pattern = _CODE if code_map else _BRACKET

    def ids_from(s: str) -> list[int]:
        out, seen = [], set()
        for m in pattern.finditer(s):
            k = code_map.get(m.group(1).upper()) if code_map else int(m.group(1))
            if k is not None and 1 <= k <= n_cands and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    pos = text.upper().rfind("FINAL:")
    if pos >= 0:
        tail = text[pos + len("FINAL:"):].split("\n\n")[0]
        ids = ids_from(tail)
        if ids:
            return ids, "final"
        if code_map:  # codes written without brackets ("FINAL: RB > DG > ..."; Qwen2.5-32B)
            out, seen = [], set()
            for m in _BARE_CODE.finditer(tail):
                k = code_map.get(m.group(1).upper())
                if k is not None and k <= n_cands and k not in seen:
                    seen.add(k)
                    out.append(k)
            if out:
                return out, "final_bare"
    for line in reversed(text.splitlines()):
        if len(pattern.findall(line)) >= 2:
            ids = ids_from(line)
            if ids:
                return ids, "fallback_line"
    return [], "fail"
