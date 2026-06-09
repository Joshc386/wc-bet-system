"""Fit model parameters on historical internationals (2010+, our own Elo).

Two-step MLE, standard Dixon-Coles practice:
  1. Poisson log-link goal model: log λ = α ± β·d  (d = effective Elo diff)
  2. ρ fitted on the four low-score cells with λs fixed from step 1.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize


def fit_goal_model(d: np.ndarray, goals_home: np.ndarray, goals_away: np.ndarray) -> tuple[float, float]:
    """MLE for (α, β). Each match contributes both sides as observations."""
    x = np.concatenate([d, -d])
    y = np.concatenate([goals_home, goals_away])

    def nll(params: np.ndarray) -> float:
        log_lam = params[0] + params[1] * x
        return float(np.sum(np.exp(log_lam) - y * log_lam))

    res = minimize(nll, x0=np.array([0.25, 0.002]), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8})
    return float(res.x[0]), float(res.x[1])


def fit_rho(d: np.ndarray, goals_home: np.ndarray, goals_away: np.ndarray,
            alpha: float, beta: float) -> float:
    """Profile MLE for the DC ρ given fixed (α, β)."""
    lh = np.exp(alpha + beta * d)
    la = np.exp(alpha - beta * d)
    h, a = goals_home, goals_away

    def nll(rho: float) -> float:
        tau = np.ones_like(lh)
        m00 = (h == 0) & (a == 0)
        m01 = (h == 0) & (a == 1)
        m10 = (h == 1) & (a == 0)
        m11 = (h == 1) & (a == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        if (tau <= 0).any():
            return np.inf
        return -float(np.sum(np.log(tau)))

    res = minimize_scalar(nll, bounds=(-0.5, 0.5), method="bounded",
                          options={"xatol": 1e-6})
    return float(res.x)
