"""Exact per-fixture 1X2 probabilities for the 72 group matches.

No Monte Carlo: with both Elos known, the DC scoreline matrix is the model's
closed-form distribution for a single match — win/draw/loss probabilities are
cell sums. These are model prices for manual comparison only (see CONTEXT.md:
odds comparison and staking are out of scope).

    python -m wcsim.match_odds   # writes data/processed/group_match_odds.csv
"""

import numpy as np
import pandas as pd

from wcsim.scores import dc_score_matrix, match_lambdas
from wcsim.tournament import ROOT, _effective_diff, load_inputs


def fixture_odds(elo_diff: float, params: dict) -> dict:
    """Exact 1X2 probabilities, expected goals and modal scoreline for one
    fixture, from the effective Elo difference (host bonus included)."""
    alpha = float(np.log(params.get("baseline", params.get("wc_baseline")) / 2.0))
    lh, la = match_lambdas(elo_diff, alpha, params["beta"])
    # max_goals=20: truncation tail < 1e-9 even at this tournament's λ≈3.7 extremes
    m = dc_score_matrix(lh, la, params["rho"], max_goals=20)
    i, j = np.unravel_index(int(m.argmax()), m.shape)
    return {
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "xg_home": lh,
        "xg_away": la,
        "top_score": f"{i}-{j}",
        "p_top_score": float(m[i, j]),
    }


def group_match_odds(inputs: dict) -> pd.DataFrame:
    rows = []
    for _, r in inputs["schedule"].iterrows():
        d = _effective_diff(
            r["home_team"], r["away_team"], r["host_country"],
            inputs["elo"], inputs["team_idx"],
        )
        rows.append({
            "date": r["date"],
            "group": r["match_group"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            **fixture_odds(d, inputs["params"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = group_match_odds(load_inputs())
    out = df.copy()
    for c in ["p_home", "p_draw", "p_away", "p_top_score"]:
        out[c] = out[c].round(4)
    for c in ["xg_home", "xg_away"]:
        out[c] = out[c].round(2)
    out.to_csv(ROOT / "data/processed/group_match_odds.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
