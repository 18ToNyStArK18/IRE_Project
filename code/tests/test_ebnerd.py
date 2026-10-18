"""EB-NeRD port: date + time display, news prompt wording, in-view-not-clicked traps."""

import pandas as pd
import pytest

from coldrank import common
from coldrank.common import DATE_ONLY, DATE_TIME, date_to_ts, set_date_format, ts_to_date
from coldrank.data.split import attach_inview_not_clicked, require_cold_negs
from coldrank.prompts.render import METHODS, render_listwise, render_pointwise
from tests.test_prompts_eval import make_pool

T = date_to_ts("2023-05-25")
H = 3600


@pytest.fixture
def date_time():
    set_date_format(DATE_TIME)
    yield
    set_date_format(DATE_ONLY)


def test_date_time_round_trip(date_time):
    ts = T + 14 * H + 32 * 60
    assert ts_to_date(ts) == "2023-05-25 14:32"
    assert date_to_ts(ts_to_date(ts)) == ts
    assert date_to_ts("2023-05-25") == date_to_ts("2023-05-25 00:00") == T


def test_date_only_is_the_default():
    assert common._date_format == DATE_ONLY and ts_to_date(T + 5 * H) == "2023-05-25"


def test_load_config_sets_format():
    common.load_config("ebnerd")
    try:
        assert ts_to_date(T + 90 * 60) == "2023-05-25 01:30"
    finally:
        common.load_config("amazon_vg")
    assert ts_to_date(T + 90 * 60) == "2023-05-25"


def news_pool():
    p = make_pool(n_warm=12, n_cold=3, n_hist=10)
    for x in p["candidates"] + p["history"]:
        x.update(subtitle="A long subtitle " * 20, topics="Crime, Gangs", category="crime")
    return p


@pytest.mark.parametrize("method", METHODS)
def test_news_wording(method):
    r = render_listwise(news_pool(), method, 0, 20, 10, id_style="codes", domain="news")
    text = r.system + r.user
    assert "purchase" not in text and "video game" not in text
    assert "reading history" in text and "read" in r.user and "(each article: title | published | section | summary" in r.user
    assert ("| summary | topics)" in r.user) == method.startswith("structural")  # Structural's extra attribute
    assert "A long subtitle A long" in r.user and "A long subtitle " * 7 not in r.user  # cut at 100 chars


def test_news_pointwise():
    p = news_pool()
    sys_, user = render_pointwise(p, p["candidates"][0], 20, "news")
    assert user.endswith("Will the user read this candidate article next? Answer Yes or No.")
    assert "Candidate article (each article:" in user and "news website" in sys_


def test_inview_not_clicked():
    items = pd.DataFrame({"item_id": ["new1", "new2", "new3", "late", "old", "hist"],
                          "t0": [T + H, T + 2 * H, T + 3 * H, T + 9 * H, T - 50 * H, T - 60 * H]})
    r = {"user_id": "u", "target_ts": T + 5 * H, "window_items": ["new2", "old"], "history": [["hist", T - H]]}
    imp = pd.DataFrame({"user_id": ["u", "u", "u", "v"], "ts": [T - H, T + 4 * H, T + 6 * H, T + 4 * H],
                        "inview": [["new3"], ["new1", "new2", "old", "hist", "late"], ["new3"], ["new3"]],
                        "clicked": [[], ["new2"], [], []]})
    attach_inview_not_clicked([r], imp, items, T)
    # new1 only: new2 clicked, old/hist not new, late published after the target, new3 shown
    # only before T, after the target, or to another user
    assert r["inview_not_clicked"] == ["new1"]


def test_require_cold_negs():
    items = pd.DataFrame({"item_id": ["a", "b", "c"], "t0": [T + H, T + 2 * H, T + 8 * H]})
    early = {"target_ts": T + 90 * 60, "window_items": []}
    late = {"target_ts": T + 9 * H, "window_items": ["b"]}
    keep, dropped = require_cold_negs([early, late], items, T, 2)
    assert keep == [late] and dropped == 1
