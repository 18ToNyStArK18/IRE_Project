"""Adversarial variants of the baseline pools (report Sec. 3.5; plan_adversarial.md).

  date_variants  A2 date swap, A3 date hide, A4 date shuffle (rewrite displayed dates
                 of the baseline pools; nothing else changes)
  allcold        A6 all-cold pools (gold + 49 cold negatives, same instances)
  oldgold        A1 old-gold pools with traps, A1-ctrl, A5 fresh flood (old-target instances)
  checks         invariants per condition + manipulation-check statistics
  build          CLI: python -m coldrank.adversarial.build --config amazon_vg

Variant pools are written next to the baseline pools as
`pools/<condition>_<neg>_<split>.jsonl`, so the existing runner, heuristics and
report work with `--condition <condition>`.
"""
