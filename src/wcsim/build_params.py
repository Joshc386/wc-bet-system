"""Fit (α, β, ρ) on all internationals 2010+ and the WC baseline; write params.json."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from wcsim.calibrate import fit_goal_model, fit_rho
from wcsim.elo import HOME_ADVANTAGE

ROOT = Path(__file__).resolve().parents[2]


def build() -> None:
    df = pd.read_csv(ROOT / "data/processed/matches_with_elo.csv")
    cal = df[df["date"] >= "2010-01-01"].copy()
    d = (
        cal["home_elo_pre"] - cal["away_elo_pre"]
        + np.where(cal["neutral"], 0.0, HOME_ADVANTAGE)
    ).to_numpy()
    gh = cal["home_score"].to_numpy(dtype=int)
    ga = cal["away_score"].to_numpy(dtype=int)

    alpha, beta = fit_goal_model(d, gh, ga)
    rho = fit_rho(d, gh, ga, alpha, beta)

    wc = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2010-01-01")
        & (df["date"] < "2023-01-01")
    ]
    wc_baseline = float((wc["home_score"] + wc["away_score"]).mean())

    params = {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "wc_baseline": wc_baseline,
        "n_calibration_matches": int(len(cal)),
        "n_wc_baseline_matches": int(len(wc)),
    }
    out = ROOT / "data/processed/params.json"
    out.write_text(json.dumps(params, indent=2))
    print(json.dumps(params, indent=2))
    # implied expected GD at a 100-Elo gap, at tournament baseline
    lh = wc_baseline / 2 * np.exp(beta * 100)
    la = wc_baseline / 2 * np.exp(-beta * 100)
    print(f"implied GD at 100 Elo gap (WC baseline): {lh - la:.3f}")


if __name__ == "__main__":
    build()
