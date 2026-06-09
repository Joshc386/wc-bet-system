"""Calibration tests: parameter recovery on synthetic data (fixed seed),
DC correction properties, and plausibility bounds on real fitted params."""

import json
from pathlib import Path

import numpy as np
import pytest

from wcsim.calibrate import fit_goal_model, fit_rho
from wcsim.scores import dc_score_matrix, match_lambdas

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(alpha: float, beta: float, n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 150, n)  # effective Elo differences
    gh = rng.poisson(np.exp(alpha + beta * d))
    ga = rng.poisson(np.exp(alpha - beta * d))
    return d, gh, ga


def test_goal_model_recovers_synthetic_params():
    alpha, beta = 0.26, 0.0025
    d, gh, ga = _synthetic(alpha, beta, 20_000, seed=42)
    a_hat, b_hat = fit_goal_model(d, gh, ga)
    assert a_hat == pytest.approx(alpha, abs=0.02)
    assert b_hat == pytest.approx(beta, rel=0.05)


def test_rho_zero_on_independent_poisson_data():
    d, gh, ga = _synthetic(0.26, 0.0025, 20_000, seed=7)
    rho = fit_rho(d, gh, ga, alpha=0.26, beta=0.0025)
    assert abs(rho) < 0.02


def test_dc_matrix_is_normalized_probability():
    m = dc_score_matrix(lam_home=1.6, lam_away=1.1, rho=-0.12, max_goals=12)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)
    assert (m >= 0).all()


def test_negative_rho_boosts_draws():
    base = dc_score_matrix(1.3, 1.3, rho=0.0, max_goals=12)
    dc = dc_score_matrix(1.3, 1.3, rho=-0.12, max_goals=12)
    assert np.trace(dc) > np.trace(base)
    # and only the four low-score cells differ
    diff = np.abs(dc - base)
    diff[:2, :2] = 0
    assert diff.max() < 1e-12


def test_match_lambdas_positive_at_extreme_gap():
    lh, la = match_lambdas(elo_diff=800, alpha=0.26, beta=0.0025)
    assert lh > la > 0


def test_match_lambdas_symmetry():
    lh, la = match_lambdas(elo_diff=200, alpha=0.26, beta=0.0025)
    lh2, la2 = match_lambdas(elo_diff=-200, alpha=0.26, beta=0.0025)
    assert lh == pytest.approx(la2) and la == pytest.approx(lh2)


def test_fitted_params_regression_lock():
    """Known-output lock: refits on the same data must reproduce these exactly.

    If this fails after a deliberate data refresh, refit and update the
    constants consciously. If it fails after a refactor, the refactor
    changed model behaviour — that's a bug.
    """
    p = json.loads((ROOT / "data/processed/params.json").read_text())
    assert p["n_calibration_matches"] == 15790
    assert p["alpha"] == pytest.approx(0.1762518, abs=1e-4)
    assert p["beta"] == pytest.approx(0.00185531, abs=1e-6)
    assert p["rho"] == pytest.approx(-0.0458185, abs=1e-4)
    assert p["wc_baseline"] == pytest.approx(2.56640625, abs=1e-9)


def test_fitted_params_plausible():
    """Real-data fit must exist and be in defensible ranges (run build_params first)."""
    p = json.loads((ROOT / "data/processed/params.json").read_text())
    # expected GD at 100-Elo gap, at tournament baseline
    lh, la = match_lambdas(100, np.log(p["wc_baseline"] / 2), p["beta"])
    assert 0.15 < lh - la < 0.8
    assert -0.3 < p["rho"] < 0.0
    assert 2.2 < p["wc_baseline"] < 3.0
