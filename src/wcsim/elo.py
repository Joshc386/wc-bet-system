"""World Football Elo engine (eloratings.net formula replica — see ADR-0001).

Rating change = K * G * (W - We), where K depends on match importance,
G on the winning goal margin, W is the result and We the expected result
including +100 home advantage for non-neutral matches.
"""

from typing import Iterable

HOME_ADVANTAGE = 100.0

# K by match importance, following eloratings.net tiers.
_K_WORLD_CUP = 60
_K_CONTINENTAL_FINALS = 50
_K_QUALIFIERS_MAJOR = 40
_K_OTHER = 30
_K_FRIENDLY = 20

_CONTINENTAL_FINALS = {
    "Copa América",
    "UEFA Euro",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Confederations Cup",
    "Oceania Nations Cup",
    "OFC Nations Cup",
}


def k_factor(tournament: str) -> int:
    """Importance weight K for a tournament name as it appears in results.csv."""
    if tournament == "FIFA World Cup":
        return _K_WORLD_CUP
    if tournament in _CONTINENTAL_FINALS:
        return _K_CONTINENTAL_FINALS
    if "qualification" in tournament or tournament == "UEFA Nations League":
        return _K_QUALIFIERS_MAJOR
    if tournament == "Friendly":
        return _K_FRIENDLY
    return _K_OTHER


def goal_multiplier(margin: int) -> float:
    """G: 1 for <=1 goal margin, 1.5 for 2, 1.75 + (N-3)/8 for N>=3."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8.0


def expected_score(home_elo: float, away_elo: float, neutral: bool) -> float:
    """Expected result for the home team, with +100 Elo if truly at home."""
    dr = home_elo - away_elo + (0.0 if neutral else HOME_ADVANTAGE)
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def update(
    home_elo: float,
    away_elo: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
) -> tuple[float, float]:
    """Rating deltas (home, away) for one match. Zero-sum by construction."""
    we = expected_score(home_elo, away_elo, neutral)
    w = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
    g = goal_multiplier(abs(home_score - away_score))
    delta = k_factor(tournament) * g * (w - we)
    return delta, -delta


def run_history(
    matches: Iterable[tuple],
    initial: float = 1500.0,
) -> tuple[list[tuple], dict[str, float]]:
    """Replay matches chronologically, recording pre-match Elo for both sides.

    `matches` rows: (date, home, away, home_score, away_score, tournament, neutral),
    already sorted by date. Returns (rows + (home_elo_pre, away_elo_pre), final ratings).
    """
    ratings: dict[str, float] = {}
    enriched: list[tuple] = []
    for date, home, away, hs, as_, tournament, neutral in matches:
        he = ratings.setdefault(home, initial)
        ae = ratings.setdefault(away, initial)
        enriched.append((date, home, away, hs, as_, tournament, neutral, he, ae))
        dh, da = update(he, ae, hs, as_, tournament, neutral)
        ratings[home] = he + dh
        ratings[away] = ae + da
    return enriched, ratings
