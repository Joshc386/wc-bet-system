"""Per-fixture group-stage odds: exactness, symmetry, and regression lock."""

import numpy as np
import pytest

from wcsim.match_odds import fixture_odds, group_match_odds
from wcsim.tournament import load_inputs

INPUTS = load_inputs()
P = INPUTS["params"]


def test_probabilities_sum_to_one():
    o = fixture_odds(elo_diff=150.0, params=P)
    assert o["p_home"] + o["p_draw"] + o["p_away"] == pytest.approx(1.0, abs=1e-6)


def test_equal_teams_symmetric():
    o = fixture_odds(elo_diff=0.0, params=P)
    assert o["p_home"] == pytest.approx(o["p_away"], abs=1e-12)
    assert o["xg_home"] == pytest.approx(o["xg_away"], abs=1e-12)


def test_mirrored_fixture_mirrors_odds():
    a = fixture_odds(elo_diff=220.0, params=P)
    b = fixture_odds(elo_diff=-220.0, params=P)
    assert a["p_home"] == pytest.approx(b["p_away"], abs=1e-12)
    assert a["xg_home"] == pytest.approx(b["xg_away"], abs=1e-12)


def test_all_72_fixtures_with_valid_odds():
    df = group_match_odds(INPUTS)
    assert len(df) == 72
    s = df["p_home"] + df["p_draw"] + df["p_away"]
    assert np.allclose(s, 1.0, atol=1e-6)
    assert (df["xg_home"] > 0).all() and (df["xg_away"] > 0).all()


def test_host_bonus_flows_into_fixture_odds():
    df = group_match_odds(INPUTS).set_index(["home_team", "away_team"])
    # Mexico open at the Azteca: heavy favourites incl. +100 host bonus
    assert df.loc[("Mexico", "South Africa"), "p_home"] > 0.75


def test_regression_lock_opening_match():
    df = group_match_odds(INPUTS).set_index(["home_team", "away_team"])
    row = df.loc[("Mexico", "South Africa")]
    assert row["p_home"] == pytest.approx(0.82201, abs=1e-4)
    assert row["p_draw"] == pytest.approx(0.12522, abs=1e-4)
