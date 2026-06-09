"""Match simulator tests: distributional checks at fixed seed + regression lock."""

import numpy as np
import pytest

from wcsim.match_sim import sample_dc, knockout_winner
from wcsim.scores import dc_score_matrix

RHO = -0.046
N = 200_000


def test_equal_teams_symmetric_and_draw_rate_matches_theory():
    rng = np.random.default_rng(123)
    lam = np.full(N, 1.28)
    h, a = sample_dc(lam, lam, RHO, rng)
    m = dc_score_matrix(1.28, 1.28, RHO)
    assert (h > a).mean() == pytest.approx((h < a).mean(), abs=0.005)
    assert (h == a).mean() == pytest.approx(np.trace(m), abs=0.005)
    assert (h == 0).mean() * 0 + ((h == 0) & (a == 0)).mean() == pytest.approx(
        m[0, 0], abs=0.004
    )


def test_sampled_cells_match_dc_matrix():
    rng = np.random.default_rng(99)
    lh_, la_ = 1.9, 0.8
    h, a = sample_dc(np.full(N, lh_), np.full(N, la_), RHO, rng)
    m = dc_score_matrix(lh_, la_, RHO)
    for i, j in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        assert ((h == i) & (a == j)).mean() == pytest.approx(m[i, j], abs=0.004)


def test_stronger_team_scores_more():
    rng = np.random.default_rng(5)
    h, a = sample_dc(np.full(N, 2.2), np.full(N, 0.6), RHO, rng)
    assert h.mean() == pytest.approx(2.2, abs=0.02)
    assert a.mean() == pytest.approx(0.6, abs=0.02)


def test_knockout_no_draws_and_fair_when_equal():
    rng = np.random.default_rng(11)
    lam = np.full(N, 1.28)
    first_wins = knockout_winner(lam, lam, RHO, rng)
    assert first_wins.dtype == bool
    assert first_wins.mean() == pytest.approx(0.5, abs=0.005)


def test_knockout_stronger_team_advances_more():
    rng = np.random.default_rng(12)
    first_wins = knockout_winner(np.full(N, 2.0), np.full(N, 0.7), RHO, rng)
    assert 0.75 < first_wins.mean() < 0.95


def test_fixed_seed_regression():
    rng = np.random.default_rng(2026)
    h, a = sample_dc(np.full(10, 1.5), np.full(10, 1.0), RHO, rng)
    # locked output — a refactor that changes sampling silently must fail here
    assert h.tolist() == [0, 2, 3, 1, 4, 2, 1, 2, 0, 1]
    assert a.tolist() == [0, 1, 0, 0, 3, 2, 1, 1, 0, 1]
