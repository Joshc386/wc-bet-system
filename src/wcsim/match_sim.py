"""Vectorized match sampling: DC-corrected scorelines and knockout resolution.

DC sampling is exact via rejection: draw independent Poisson pairs, accept
with probability τ(h,a)/τ_max. With ρ ≈ -0.05 acceptance is ~95%, so the
resample loop runs 1-2 rounds over a shrinking remainder.
"""

import numpy as np


def _tau(h: np.ndarray, a: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lh, dtype=float)
    m = (h == 0) & (a == 0)
    tau[m] = 1 - lh[m] * la[m] * rho
    m = (h == 0) & (a == 1)
    tau[m] = 1 + lh[m] * rho
    m = (h == 1) & (a == 0)
    tau[m] = 1 + la[m] * rho
    m = (h == 1) & (a == 1)
    tau[m] = 1 - rho
    return tau


def sample_dc(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (home, away) goals per element with the DC low-score correction."""
    lh = np.asarray(lam_home, dtype=float)
    la = np.asarray(lam_away, dtype=float)
    tau_max = np.maximum.reduce(
        [np.ones_like(lh), 1 - lh * la * rho, 1 + lh * rho, 1 + la * rho,
         np.full_like(lh, 1 - rho)]
    )
    h = np.empty(lh.shape, dtype=np.int64)
    a = np.empty(lh.shape, dtype=np.int64)
    pending = np.arange(lh.size)
    while pending.size:
        hh = rng.poisson(lh[pending])
        aa = rng.poisson(la[pending])
        accept = rng.random(pending.size) * tau_max[pending] < _tau(
            hh, aa, lh[pending], la[pending], rho
        )
        h[pending[accept]] = hh[accept]
        a[pending[accept]] = aa[accept]
        pending = pending[~accept]
    return h, a


def knockout_winner(
    lam_home: np.ndarray, lam_away: np.ndarray, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """True where the first team advances: 90' DC sample, then ET at λ/3
    (independent Poisson), then a fair coin (see CONTEXT.md: Knockout Resolution)."""
    h, a = sample_dc(lam_home, lam_away, rho, rng)
    level = h == a
    if level.any():
        eth = rng.poisson(lam_home[level] / 3.0)
        eta = rng.poisson(lam_away[level] / 3.0)
        h = h.astype(float)
        h[level] += eth
        a = a.astype(float)
        a[level] += eta
        still = h == a
        coin = rng.random(int(still.sum())) < 0.5
        h[still] += np.where(coin, 0.5, -0.5)
    return h > a
