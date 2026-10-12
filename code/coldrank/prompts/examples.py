"""Hand-written worked examples (few-shot demonstrations).

Li et al. give Base-Reason two worked examples, Fast-Reason one (shorter), and
Structural Reasoning one exemplar of reasoning paths; their texts are not public.

Design constraints (design.md):
  * Items are real products launched long before any cutoff we use (<= 2018), so
    they cannot leak test items.
  * The correct item is NOT the newest candidate (example 1: 2nd newest of 5;
    example 2: 4th newest of 5) and the reasoning never mentions dates, so the
    demonstrations do not teach a recency rule in either direction.
  * Pools are short (5 candidates) to fit the context budget.
  * The same exemplar is used for Structural Reasoning with and without the
    recency instruction; its weights are justified by prominence only.
"""

EXAMPLE_1 = {
    "history": [
        {"title": "The Legend of Zelda: Breath of the Wild - Nintendo Switch", "date": "2017-03-03",
         "category": "Nintendo Switch > Games", "brand": "Nintendo"},
        {"title": "Nintendo Switch Pro Controller", "date": "2017-03-03",
         "category": "Nintendo Switch > Accessories > Controllers", "brand": "Nintendo"},
        {"title": "Mario Kart 8 Deluxe - Nintendo Switch", "date": "2017-04-28",
         "category": "Nintendo Switch > Games", "brand": "Nintendo"},
        {"title": "Splatoon 2 - Nintendo Switch", "date": "2017-07-21",
         "category": "Nintendo Switch > Games", "brand": "Nintendo"},
        {"title": "Super Mario Odyssey - Nintendo Switch", "date": "2017-10-27",
         "category": "Nintendo Switch > Games", "brand": "Nintendo"},
    ],
    "candidates": [
        {"title": "Halo 5: Guardians - Xbox One", "date": "2015-10-27",
         "category": "Xbox One > Games", "brand": "Microsoft"},
        {"title": "Donkey Kong Country: Tropical Freeze - Nintendo Switch", "date": "2018-05-04",
         "category": "Nintendo Switch > Games", "brand": "Nintendo"},
        {"title": "FIFA 19 - PlayStation 4", "date": "2018-09-28",
         "category": "PlayStation 4 > Games", "brand": "Electronic Arts"},
        {"title": "Logitech G502 Proteus Spectrum Gaming Mouse", "date": "2016-01-25",
         "category": "PC > Accessories > Gaming Mice", "brand": "Logitech"},
        {"title": "Xbox One Wireless Controller", "date": "2016-08-02",
         "category": "Xbox One > Accessories > Controllers", "brand": "Microsoft"},
    ],
    "base": (
        "Summary: The user plays on a Nintendo Switch and buys Nintendo's own games: Zelda, Mario Kart, "
        "Splatoon and Super Mario Odyssey, plus a Switch controller. They like Nintendo's action, platform "
        "and adventure games.\n"
        "Reasoning: [2] Donkey Kong Country: Tropical Freeze is a Nintendo platform game for the Switch, so it "
        "matches both the user's platform and the kind of game they buy. [3] FIFA 19 and [1] Halo 5 are popular "
        "games, but for PlayStation 4 and Xbox One, which the user has shown no sign of owning. [5] is an Xbox "
        "controller and [4] is a PC mouse; neither is useful to a Switch player."
    ),
    "base_final": [2, 3, 1, 5, 4],
    "fast": "The user buys Nintendo Switch games, mostly Nintendo's own. [2] is the only Nintendo Switch game among the candidates.",
    "fast_final": [2, 3, 1, 5, 4],
    "structural": (
        "Step 1 - reasoning paths:\n"
        "Path A (platform: Nintendo Switch): Zelda: Breath of the Wild, Pro Controller, Mario Kart 8 Deluxe, Splatoon 2, Super Mario Odyssey.\n"
        "Path B (brand: Nintendo): Zelda: Breath of the Wild, Mario Kart 8 Deluxe, Splatoon 2, Super Mario Odyssey.\n"
        "Path C (type: action and platform games): Zelda: Breath of the Wild, Splatoon 2, Super Mario Odyssey.\n"
        "Step 2 - match scores (A, B, C) for candidates matching at least one path:\n"
        "[2] A=10, B=9, C=9\n"
        "[1] A=0, B=0, C=5\n"
        "[3] A=0, B=0, C=1\n"
        "All other candidates score 0.\n"
        "Step 3 - weights: A=0.40 (every history item is for the Switch), B=0.35 (four of five items are Nintendo games), "
        "C=0.25 (three items are action or platform games).\n"
        "Overall: [2]=9.40, [1]=1.25, [3]=0.25, [4]=0, [5]=0."
    ),
    "structural_final": [2, 1, 3, 5, 4],
}

EXAMPLE_2 = {
    "history": [
        {"title": "Razer BlackWidow Chroma Mechanical Gaming Keyboard", "date": "2014-09-19",
         "category": "PC > Accessories > Gaming Keyboards", "brand": "Razer"},
        {"title": "XCOM 2 - PC", "date": "2016-02-05", "category": "PC > Games", "brand": "2K"},
        {"title": "Stellaris - PC", "date": "2016-05-09", "category": "PC > Games", "brand": "Paradox Interactive"},
        {"title": "Total War: Warhammer - PC", "date": "2016-05-24", "category": "PC > Games", "brand": "SEGA"},
        {"title": "Sid Meier's Civilization VI - PC", "date": "2016-10-21", "category": "PC > Games", "brand": "2K"},
    ],
    "candidates": [
        {"title": "Forza Horizon 3 - Xbox One", "date": "2016-09-27",
         "category": "Xbox One > Games", "brand": "Microsoft"},
        {"title": "Europa Universalis IV - PC", "date": "2013-08-13",
         "category": "PC > Games", "brand": "Paradox Interactive"},
        {"title": "Mario Party 10 - Wii U", "date": "2015-03-20",
         "category": "Legacy Systems > Nintendo Systems > Wii U > Games", "brand": "Nintendo"},
        {"title": "Xbox 360 Kinect Sensor", "date": "2010-11-04",
         "category": "Legacy Systems > Xbox Systems > Xbox 360 > Accessories", "brand": "Microsoft"},
        {"title": "Just Dance 2014 - Wii", "date": "2013-10-08",
         "category": "Legacy Systems > Nintendo Systems > Wii > Games", "brand": "Ubisoft"},
    ],
    "base": (
        "Summary: The user is a PC gamer who buys strategy games: XCOM 2, Stellaris, Total War: Warhammer and "
        "Civilization VI. Stellaris is from Paradox Interactive. They also own a gaming keyboard.\n"
        "Reasoning: [2] Europa Universalis IV is a PC grand-strategy game from Paradox Interactive, the publisher "
        "of Stellaris, so it fits the user's platform, genre and a publisher they already like. [1] Forza "
        "Horizon 3 is a well-made game but a racing game for Xbox One. [3] Mario Party 10 and [5] Just Dance 2014 "
        "are party games for Nintendo consoles, far from the user's taste. [4] is a motion sensor for the Xbox 360, "
        "which the user does not appear to own."
    ),
    "base_final": [2, 1, 3, 5, 4],
}
