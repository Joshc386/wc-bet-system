"""Frozen-Elo backtests of the pipeline on the 2018 and 2022 World Cups.

Everything is out-of-sample: Elo is replayed only up to kickoff, and
(α, β, ρ, baseline) are refitted on data strictly before the tournament.
32-team format: 8 groups, top two advance, standard FIFA R16 template.
"""

import numpy as np
import pandas as pd

from wcsim.build_ratings import ROOT
from wcsim.calibrate import fit_goal_model, fit_rho
from wcsim.elo import HOME_ADVANTAGE, run_history
from wcsim.match_sim import knockout_winner, sample_dc
from wcsim.tournament import _rank_groups

GROUPS_2018 = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}
GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}
# Standard 32-team R16 template (same for 2018 and 2022).
R16_TEMPLATE = [
    ("W:A", "R:B"), ("W:C", "R:D"), ("W:B", "R:A"), ("W:D", "R:C"),
    ("W:E", "R:F"), ("W:G", "R:H"), ("W:F", "R:E"), ("W:H", "R:G"),
]
QF_TEMPLATE = [(0, 1), (4, 5), (2, 3), (6, 7)]  # indices into R16 winners
SF_TEMPLATE = [(0, 1), (2, 3)]  # indices into QF winners

EDITIONS = {
    2018: {"groups": GROUPS_2018, "cutoff": "2018-06-14", "host": "Russia"},
    2022: {"groups": GROUPS_2022, "cutoff": "2022-11-20", "host": "Qatar"},
}


def _kickoff_inputs(year: int) -> dict:
    cfg = EDITIONS[year]
    df = pd.read_csv(ROOT / "data/raw/results.csv")
    played = df.dropna(subset=["home_score", "away_score"]).sort_values("date", kind="stable")
    pre = played[played["date"] < cfg["cutoff"]]
    matches = list(zip(pre["date"], pre["home_team"], pre["away_team"],
                       pre["home_score"].astype(int), pre["away_score"].astype(int),
                       pre["tournament"], pre["neutral"].astype(bool)))
    enriched, ratings = run_history(matches)

    cal = pd.DataFrame(enriched, columns=["date", "home_team", "away_team", "home_score",
                                          "away_score", "tournament", "neutral",
                                          "home_elo_pre", "away_elo_pre"])
    cal = cal[cal["date"] >= "2010-01-01"]
    d = (cal["home_elo_pre"] - cal["away_elo_pre"]
         + np.where(cal["neutral"], 0.0, HOME_ADVANTAGE)).to_numpy()
    gh = cal["home_score"].to_numpy(dtype=int)
    ga = cal["away_score"].to_numpy(dtype=int)
    alpha, beta = fit_goal_model(d, gh, ga)
    rho = fit_rho(d, gh, ga, alpha, beta)
    wc = cal[cal["tournament"] == "FIFA World Cup"]
    baseline = float((wc["home_score"] + wc["away_score"]).mean())

    # the year's actual group fixtures (group stage = first 48 WC matches)
    wc_year = played[(played["tournament"] == "FIFA World Cup")
                     & (played["date"] >= cfg["cutoff"])
                     & (played["date"] < f"{year + 1}-01-01")]
    group_fix = wc_year.head(48).rename(columns={"country": "host_country"})

    groups = cfg["groups"]
    teams = [t for g in sorted(groups) for t in groups[g]]
    fixture_teams = set(group_fix["home_team"]) | set(group_fix["away_team"])
    assert fixture_teams == set(teams), fixture_teams ^ set(teams)

    return {
        "groups": groups,
        "schedule": group_fix.reset_index(drop=True),
        "teams": teams,
        "team_idx": {t: i for i, t in enumerate(teams)},
        "elo": np.array([ratings[t] for t in teams]),
        "params": {"alpha": alpha, "beta": beta, "rho": rho, "wc_baseline": baseline},
        "host": cfg["host"],
        "actual": wc_year,
    }


def run_backtest(year: int, n_sims: int = 100_000, seed: int = 2026) -> pd.DataFrame:
    inputs = _kickoff_inputs(year)
    rng = np.random.default_rng(seed)
    p = inputs["params"]
    alpha_wc = float(np.log(p["wc_baseline"] / 2.0))
    beta, rho = p["beta"], p["rho"]
    elo, idx, teams, host = inputs["elo"], inputs["team_idx"], inputs["teams"], inputs["host"]
    n = n_sims
    sched = inputs["schedule"]

    diffs = np.array([
        float(elo[idx[r["home_team"]]] - elo[idx[r["away_team"]]])
        + HOME_ADVANTAGE * (r["home_team"] == host)
        - HOME_ADVANTAGE * (r["away_team"] == host)
        for _, r in sched.iterrows()
    ])
    lh = np.repeat(np.exp(alpha_wc + beta * diffs), n)
    la = np.repeat(np.exp(alpha_wc - beta * diffs), n)
    h, a = sample_dc(lh, la, rho, rng)
    gh, ga = h.reshape(len(sched), n), a.reshape(len(sched), n)
    winner, runner, _, _, _ = _rank_groups(inputs, gh, ga, rng, n)

    group_no = {chr(65 + i): i for i in range(8)}
    sims = np.arange(n)
    n_teams = len(teams)
    reached = {st: np.zeros((n_teams, n), dtype=bool)
               for st in ["R16", "QF", "SF", "Final", "Champion"]}

    def play(home: np.ndarray, away: np.ndarray, stage: str) -> np.ndarray:
        d = elo[home] - elo[away]
        if host in idx:  # host bonus in all knockout matches (played in host country)
            d = d + HOME_ADVANTAGE * (home == idx[host]) - HOME_ADVANTAGE * (away == idx[host])
        w = np.where(knockout_winner(np.exp(alpha_wc + beta * d),
                                     np.exp(alpha_wc - beta * d), rho, rng), home, away)
        reached[stage][w, sims] = True
        return w

    r16 = []
    for hr, ar in R16_TEMPLATE:
        home = winner[group_no[hr[2]]] if hr[0] == "W" else runner[group_no[hr[2]]]
        away = runner[group_no[ar[2]]] if ar[0] == "R" else winner[group_no[ar[2]]]
        reached["R16"][home, sims] = True
        reached["R16"][away, sims] = True
        r16.append(play(home, away, "QF"))
    qf = [play(r16[i], r16[j], "SF") for i, j in QF_TEMPLATE]
    sf = [play(qf[i], qf[j], "Final") for i, j in SF_TEMPLATE]
    play(sf[0], sf[1], "Champion")

    out = pd.DataFrame({st: reached[st].mean(axis=1) for st in reached},
                       index=pd.Index(teams, name="team"))
    return out.sort_values("Champion", ascending=False)


if __name__ == "__main__":
    for year in (2018, 2022):
        probs = run_backtest(year)
        probs.to_csv(ROOT / f"data/processed/backtest_{year}.csv")
        print(f"\n=== {year} (top 10 by predicted champion prob) ===")
        print((probs.head(10) * 100).round(1).to_string())
