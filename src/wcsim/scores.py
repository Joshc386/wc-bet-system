"""Scoreline model: Poisson rates from Elo difference + Dixon-Coles correction.

λ_i = exp(α ± β·d) where d is the effective Elo difference (including any
home/host +100). The DC τ adjustment reweights only the 0-0, 1-0, 0-1, 1-1
cells and preserves total probability exactly (see ADR-0002, CONTEXT.md).
"""

import numpy as np
from scipy.stats import poisson


def match_lambdas(elo_diff: float, alpha: float, beta: float) -> tuple[float, float]:
    """Poisson rates (home-side, away-side) for an effective Elo difference."""
    return float(np.exp(alpha + beta * elo_diff)), float(np.exp(alpha - beta * elo_diff))


def dc_score_matrix(
    lam_home: float, lam_away: float, rho: float, max_goals: int = 12
) -> np.ndarray:
    """(max_goals+1)² matrix of P(home=i, away=j) with the DC low-score correction."""
    i = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(i, lam_home), poisson.pmf(i, lam_away))
    m[0, 0] *= 1 - lam_home * lam_away * rho
    m[0, 1] *= 1 + lam_home * rho
    m[1, 0] *= 1 + lam_away * rho
    m[1, 1] *= 1 - rho
    return m
