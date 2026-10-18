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


# --------------------------------------------------------------------------- #
# News (EB-NeRD, prompt.domain: news). Same constraints as above: invented articles dated
# 2019, years before the 2023 data, so they cannot leak test items; the correct article
# is not the newest (example 1: 2nd newest of 5; example 2: 4th newest of 5); the
# reasoning never mentions dates.
# --------------------------------------------------------------------------- #
NEWS_EXAMPLE_1 = {
    "history": [
        {"title": "FC Copenhagen sign Swedish striker on a four-year deal", "date": "2019-08-02 09:14",
         "category": "sport", "subtitle": "The Superliga champions have paid a club record fee for the 22-year-old forward",
         "topics": "Sport, Football"},
        {"title": "Brøndby fire their coach after a third straight defeat", "date": "2019-08-05 18:40",
         "category": "sport", "subtitle": "The club's board acted the morning after the 0-3 loss to Midtjylland",
         "topics": "Sport, Football"},
        {"title": "Superliga: FCK go top after a late winner in Aarhus", "date": "2019-08-11 21:05",
         "category": "sport", "subtitle": "The new Swedish signing scored in the 89th minute of his debut",
         "topics": "Sport, Football"},
        {"title": "National team squad: three new names for the Ireland match", "date": "2019-08-29 12:30",
         "category": "sport", "subtitle": "The national coach has picked two FCK players and a goalkeeper from Brøndby",
         "topics": "Sport, Football, National team"},
        {"title": "Christian Eriksen: I will decide about my future soon", "date": "2019-09-02 07:55",
         "category": "sport", "subtitle": "The Danish playmaker's contract with Tottenham runs out next summer",
         "topics": "Sport, Football, Celebrities"},
    ],
    "candidates": [
        {"title": "Electricity prices will rise sharply this winter", "date": "2019-09-03 06:10",
         "category": "consumer", "subtitle": "Households can expect to pay several hundred kroner more a month",
         "topics": "Economy, Energy"},
        {"title": "FCK coach on the injury crisis: The young players must step up", "date": "2019-09-04 08:20",
         "category": "sport", "subtitle": "Five first-team players are out before Sunday's top match against Brøndby",
         "topics": "Sport, Football"},
        {"title": "The royal couple celebrate their anniversary at Fredensborg", "date": "2019-09-03 14:45",
         "category": "entertainment", "subtitle": "Hundreds of well-wishers had gathered outside the palace",
         "topics": "Royals, Celebrities"},
        {"title": "Woman robbed at knifepoint in Odense", "date": "2019-09-02 22:30",
         "category": "crime", "subtitle": "The police are looking for two men who fled on foot",
         "topics": "Crime, Robbery"},
        {"title": "Recipe: the best apple cake of the autumn", "date": "2019-09-04 10:05",
         "category": "consumer", "subtitle": "Our food editor shares her grandmother's recipe with crumble and cream",
         "topics": "Food, Lifestyle"},
    ],
    "base": (
        "Summary: The user reads football news: transfers and results from FC Copenhagen and Brøndby, the "
        "Superliga, the national team and Christian Eriksen. Every article they read is in the sport section.\n"
        "Reasoning: [2] is about FC Copenhagen's injury crisis before a match against Brøndby, the two clubs the "
        "user follows most closely. [3] is about celebrities, which only touches the user's interest in famous "
        "players. [4] is a crime story, [1] is about electricity prices and [5] is a recipe; none of them matches "
        "what the user reads."
    ),
    "base_final": [2, 3, 4, 1, 5],
    "fast": "The user reads football news about FC Copenhagen and Brøndby. [2] is the only football article among the candidates.",
    "fast_final": [2, 3, 4, 1, 5],
    "structural": (
        "Step 1 - reasoning paths:\n"
        "Path A (section: sport): all five history articles.\n"
        "Path B (topic: Danish club football): FCK signing, Brøndby coach fired, Superliga result.\n"
        "Path C (people: famous players): national team squad, Christian Eriksen.\n"
        "Step 2 - match scores (A, B, C) for candidates matching at least one path:\n"
        "[2] A=10, B=9, C=4\n"
        "[3] A=0, B=0, C=2\n"
        "[4] A=0, B=0, C=1\n"
        "All other candidates score 0.\n"
        "Step 3 - weights: A=0.40 (every history article is in the sport section), B=0.40 (three of five are about "
        "Danish clubs), C=0.20 (two are about individual players).\n"
        "Overall: [2]=8.40, [3]=0.40, [4]=0.20, [1]=0, [5]=0."
    ),
    "structural_final": [2, 3, 4, 1, 5],
}

NEWS_EXAMPLE_2 = {
    "history": [
        {"title": "Man charged with the murder of a 24-year-old in Aalborg", "date": "2019-02-11 10:02",
         "category": "crime", "subtitle": "The 31-year-old denies the charge and has been remanded in custody for four weeks",
         "topics": "Crime, Murder"},
        {"title": "Gang conflict: police set up a stop-and-search zone in Copenhagen", "date": "2019-02-14 16:20",
         "category": "crime", "subtitle": "The zone covers Nørrebro and Nordvest after two shootings in one week",
         "topics": "Crime, Gangs"},
        {"title": "Court: ten years in prison for armed robbery of a jeweller", "date": "2019-02-20 13:45",
         "category": "crime", "subtitle": "The 27-year-old threatened the staff with a pistol and fled with watches",
         "topics": "Crime, Courts"},
        {"title": "Police warn of fake bank calls to the elderly", "date": "2019-02-25 08:30",
         "category": "crime", "subtitle": "Fraudsters pretend to be from the bank and ask for card codes",
         "topics": "Crime, Fraud"},
        {"title": "Biker leader arrested at Copenhagen Airport", "date": "2019-03-01 19:10",
         "category": "crime", "subtitle": "The 45-year-old was wanted for his role in a large drugs case",
         "topics": "Crime, Gangs"},
    ],
    "candidates": [
        {"title": "The new iPhone sold out within hours", "date": "2019-03-06 09:00",
         "category": "consumer", "subtitle": "Customers queued outside the shops in central Copenhagen from early morning",
         "topics": "Technology, Shopping"},
        {"title": "Police: young man shot in the leg in Nørrebro", "date": "2019-03-04 23:40",
         "category": "crime", "subtitle": "The police are investigating whether the shooting is linked to the gang conflict",
         "topics": "Crime, Gangs"},
        {"title": "Reality star reveals she is pregnant", "date": "2019-03-05 20:15",
         "category": "entertainment", "subtitle": "The 26-year-old shared the news with her followers on Instagram",
         "topics": "Celebrities, Reality TV"},
        {"title": "Snow and frost on the way this weekend", "date": "2019-03-06 11:30",
         "category": "news", "subtitle": "The Meteorological Institute expects down to minus eight degrees at night",
         "topics": "Weather"},
        {"title": "Handball: Denmark beat Norway in a thriller", "date": "2019-03-03 07:45",
         "category": "sport", "subtitle": "The winning goal came with four seconds left on the clock",
         "topics": "Sport, Handball"},
    ],
    "base": (
        "Summary: The user reads crime news: murder and robbery cases, court verdicts, fraud warnings and above "
        "all the gang conflict in Copenhagen, including the stop-and-search zone in Nørrebro.\n"
        "Reasoning: [2] is a shooting in Nørrebro that may be linked to the gang conflict, which the user has "
        "followed in several articles, so it fits both their section and their main story. [4] is general news "
        "about the weather. [5] is sport, [3] is celebrity news and [1] is about a phone launch; the user reads "
        "none of these kinds of articles."
    ),
    "base_final": [2, 4, 5, 3, 1],
}
