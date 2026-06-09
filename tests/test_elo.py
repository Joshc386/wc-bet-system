"""World Football Elo engine tests (red first, per /tdd)."""

import pytest

from wcsim.elo import expected_score, goal_multiplier, k_factor, update, run_history


def test_expected_score_equal_ratings_neutral():
    assert expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_score_400_point_gap():
    # Elo definition: +400 difference -> 10/11 expected
    assert expected_score(1900, 1500, neutral=True) == pytest.approx(10 / 11)


def test_home_advantage_only_when_not_neutral():
    neutral = expected_score(1500, 1500, neutral=True)
    at_home = expected_score(1500, 1500, neutral=False)
    # +100 home advantage: dr=100 -> 1/(10^(-100/400)+1)
    assert neutral == pytest.approx(0.5)
    assert at_home == pytest.approx(1 / (10 ** (-100 / 400) + 1))


@pytest.mark.parametrize(
    "tournament,k",
    [
        ("FIFA World Cup", 60),
        ("Copa América", 50),
        ("UEFA Euro", 50),
        ("African Cup of Nations", 50),
        ("AFC Asian Cup", 50),
        ("Gold Cup", 50),
        ("Confederations Cup", 50),
        ("FIFA World Cup qualification", 40),
        ("UEFA Euro qualification", 40),
        ("UEFA Nations League", 40),
        ("King's Cup", 30),
        ("Friendly", 20),
    ],
)
def test_k_factor(tournament, k):
    assert k_factor(tournament) == k


@pytest.mark.parametrize(
    "margin,mult",
    [(0, 1.0), (1, 1.0), (2, 1.5), (3, 1.75), (4, 1.875), (5, 2.0), (7, 2.25)],
)
def test_goal_multiplier(margin, mult):
    assert goal_multiplier(margin) == pytest.approx(mult)


def test_update_zero_sum():
    dh, da = update(1700, 1500, 2, 0, "Friendly", neutral=True)
    assert dh == pytest.approx(-da)


def test_update_hand_computed():
    # Friendly, neutral, equal ratings, home wins 1-0:
    # K=20, G=1, We=0.5 -> home +10
    dh, da = update(1500, 1500, 1, 0, "Friendly", neutral=True)
    assert dh == pytest.approx(10.0)
    assert da == pytest.approx(-10.0)


def test_update_draw_moves_points_to_weaker_team():
    dh, da = update(1800, 1500, 1, 1, "Friendly", neutral=True)
    assert dh < 0 < da


def test_run_history_tracks_pre_match_elo_and_final_ratings():
    matches = [
        # date, home, away, hs, as, tournament, neutral
        ("2020-01-01", "Aaa", "Bbb", 1, 0, "Friendly", True),
        ("2020-01-02", "Aaa", "Bbb", 1, 0, "Friendly", True),
    ]
    enriched, ratings = run_history(matches, initial=1500.0)
    # both pre-match Elos recorded before any update
    assert enriched[0][-2:] == (1500.0, 1500.0)
    # second match sees first match's update: Aaa 1510, Bbb 1490
    assert enriched[1][-2:] == (1510.0, 1490.0)
    # second win: We = 1/(10**(-20/400)+1) ≈ 0.5288, gain = 20*(1-We)
    expected_gain = 20 * (1 - 1 / (10 ** (-20 / 400) + 1))
    assert ratings["Aaa"] == pytest.approx(1510 + expected_gain)
    assert ratings["Aaa"] + ratings["Bbb"] == pytest.approx(3000.0)
