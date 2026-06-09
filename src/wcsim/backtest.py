"""Frozen-Elo backtests on nine past tournaments (WC, Euro, Copa América).

Everything is out-of-sample per edition: Elo replayed only up to kickoff,
(α, β, ρ) refitted on internationals 2010 → kickoff, and the baseline
scoring rate taken from prior editions of the same competition only.

Group fixtures come from the actual played matches in results.csv (which
also yields venue countries for the uniform host-bonus rule). Knockout
brackets are encoded below in the tournament.py slot grammar, verified
against the actual R16/QF/SF lineups of each edition.
"""

import numpy as np
import pandas as pd

from wcsim.build_ratings import ROOT
from wcsim.calibrate import fit_goal_model, fit_rho
from wcsim.elo import HOME_ADVANTAGE, run_history
from wcsim.tournament import run_tournament, third_slots


def _rounds(host: str, *rounds: tuple[str, list[tuple[str, str]]]) -> dict:
    """Build a bracket dict from (round_name, [(home_slot, away_slot), ...]),
    numbering matches sequentially. Single-host editions: every knockout
    venue is in `host`. For multi-host editions pass per-match venues via
    a (home, away, venue) triple instead."""
    bracket: dict = {}
    no = 1
    for name, matches in rounds:
        rnd = []
        for m in matches:
            venue = m[2] if len(m) == 3 else host
            rnd.append({"match": no, "home": m[0], "away": m[1], "host_country": venue})
            no += 1
        bracket[name] = rnd
    return bracket


def _wc32(host: str) -> dict:
    return _rounds(
        host,
        ("r16", [("W:A", "R:B"), ("W:C", "R:D"), ("W:B", "R:A"), ("W:D", "R:C"),
                 ("W:E", "R:F"), ("W:G", "R:H"), ("W:F", "R:E"), ("W:H", "R:G")]),
        ("qf", [("M:1", "M:2"), ("M:5", "M:6"), ("M:3", "M:4"), ("M:7", "M:8")]),
        ("sf", [("M:9", "M:10"), ("M:11", "M:12")]),
        ("final", [("M:13", "M:14")]),
    )


def _euro2016() -> dict:
    return _rounds(
        "France",
        ("r16", [("R:A", "R:C"), ("W:D", "T:BEF"), ("W:B", "T:ACD"), ("W:F", "R:E"),
                 ("W:C", "T:ABF"), ("W:E", "R:D"), ("W:A", "T:CDE"), ("R:B", "R:F")]),
        ("qf", [("M:1", "M:2"), ("M:3", "M:4"), ("M:5", "M:6"), ("M:7", "M:8")]),
        ("sf", [("M:9", "M:10"), ("M:11", "M:12")]),
        ("final", [("M:13", "M:14")]),
    )


def _euro_2020s(venues: list[str] | None = None) -> dict:
    """Shared Euro 2020/2024 template. `venues`: 15 knockout venue countries
    in match order for the multi-host 2020 edition; None = all Germany (2024)."""
    v = venues or ["Germany"] * 15
    return _rounds(
        "",
        ("r16", [("R:A", "R:B", v[0]), ("W:A", "R:C", v[1]), ("W:C", "T:DEF", v[2]),
                 ("W:B", "T:ADEF", v[3]), ("R:D", "R:E", v[4]), ("W:F", "T:ABC", v[5]),
                 ("W:D", "R:F", v[6]), ("W:E", "T:ABCD", v[7])]),
        ("qf", [("M:1", "M:3", v[8]), ("M:2", "M:4", v[9]),
                ("M:5", "M:6", v[10]), ("M:7", "M:8", v[11])]),
        ("sf", [("M:10", "M:11", v[12]), ("M:12", "M:9", v[13])]),
        ("final", [("M:13", "M:14", v[14])]),
    )


def _copa16(host: str) -> dict:
    return _rounds(
        host,
        ("qf", [("W:A", "R:B"), ("W:B", "R:A"), ("W:C", "R:D"), ("W:D", "R:C")]),
        ("sf", [("M:1", "M:2"), ("M:3", "M:4")]),
        ("final", [("M:5", "M:6")]),
    )


def _copa2019() -> dict:
    return _rounds(
        "Brazil",
        ("qf", [("W:A", "T:BC"), ("R:A", "R:B"), ("W:B", "R:C"), ("W:C", "T:AB")]),
        ("sf", [("M:1", "M:2"), ("M:3", "M:4")]),
        ("final", [("M:5", "M:6")]),
    )


def _copa2021() -> dict:
    return _rounds(
        "Brazil",
        ("qf", [("W:A", "P:4:B"), ("R:A", "P:3:B"), ("W:B", "P:4:A"), ("R:B", "P:3:A")]),
        ("sf", [("M:1", "M:2"), ("M:3", "M:4")]),
        ("final", [("M:5", "M:6")]),
    )


EURO_2020_KO_VENUES = [
    "Netherlands", "England", "Hungary", "Spain", "Denmark", "Romania",
    "England", "Scotland",                       # R16
    "Azerbaijan", "Germany", "Russia", "Italy",  # QF
    "England", "England", "England",             # SF, final
]

WC_STAGES = ["R16", "QF", "SF", "Final", "Champion"]
EURO_STAGES = ["R16", "QF", "SF", "Final", "Champion"]
COPA_STAGES = ["QF", "SF", "Final", "Champion"]

EDITIONS: dict[str, dict] = {
    "wc2018": {
        "competition": "FIFA World Cup", "cutoff": "2018-06-14", "end": "2019-01-01",
        "champion": "France", "bronze": True, "stages": WC_STAGES,
        "baseline": ("FIFA World Cup", "2010-01-01"), "bracket": _wc32("Russia"),
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
    "wc2022": {
        "competition": "FIFA World Cup", "cutoff": "2022-11-20", "end": "2023-01-01",
        "champion": "Argentina", "bronze": True, "stages": WC_STAGES,
        "baseline": ("FIFA World Cup", "2010-01-01"), "bracket": _wc32("Qatar"),
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
    "euro2016": {
        "competition": "UEFA Euro", "cutoff": "2016-06-10", "end": "2016-08-01",
        "champion": "Portugal", "bronze": False, "stages": EURO_STAGES,
        "baseline": ("UEFA Euro", "2000-01-01"), "bracket": _euro2016(),
        "groups": {
            "A": ["France", "Romania", "Albania", "Switzerland"],
            "B": ["England", "Russia", "Wales", "Slovakia"],
            "C": ["Germany", "Ukraine", "Poland", "Northern Ireland"],
            "D": ["Spain", "Czech Republic", "Turkey", "Croatia"],
            "E": ["Belgium", "Italy", "Republic of Ireland", "Sweden"],
            "F": ["Portugal", "Iceland", "Austria", "Hungary"],
        },
    },
    "euro2020": {
        "competition": "UEFA Euro", "cutoff": "2021-06-11", "end": "2021-08-01",
        "champion": "Italy", "bronze": False, "stages": EURO_STAGES,
        "baseline": ("UEFA Euro", "2000-01-01"),
        "bracket": _euro_2020s(EURO_2020_KO_VENUES),
        "groups": {
            "A": ["Italy", "Switzerland", "Turkey", "Wales"],
            "B": ["Belgium", "Denmark", "Finland", "Russia"],
            "C": ["Netherlands", "Ukraine", "Austria", "North Macedonia"],
            "D": ["England", "Croatia", "Czech Republic", "Scotland"],
            "E": ["Spain", "Sweden", "Poland", "Slovakia"],
            "F": ["France", "Germany", "Portugal", "Hungary"],
        },
    },
    "euro2024": {
        "competition": "UEFA Euro", "cutoff": "2024-06-14", "end": "2024-08-01",
        "champion": "Spain", "bronze": False, "stages": EURO_STAGES,
        "baseline": ("UEFA Euro", "2000-01-01"), "bracket": _euro_2020s(),
        "groups": {
            "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
            "B": ["Spain", "Croatia", "Italy", "Albania"],
            "C": ["England", "Denmark", "Slovenia", "Serbia"],
            "D": ["Poland", "Netherlands", "Austria", "France"],
            "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
            "F": ["Portugal", "Czech Republic", "Georgia", "Turkey"],
        },
    },
    "copa2016": {
        "competition": "Copa América", "cutoff": "2016-06-03", "end": "2016-07-01",
        "champion": "Chile", "bronze": True, "stages": COPA_STAGES,
        "baseline": ("Copa América", "2000-01-01"),
        "bracket": _copa16("United States"),
        "groups": {
            "A": ["United States", "Colombia", "Costa Rica", "Paraguay"],
            "B": ["Brazil", "Ecuador", "Haiti", "Peru"],
            "C": ["Mexico", "Uruguay", "Jamaica", "Venezuela"],
            "D": ["Argentina", "Chile", "Panama", "Bolivia"],
        },
    },
    "copa2019": {
        "competition": "Copa América", "cutoff": "2019-06-14", "end": "2019-08-01",
        "champion": "Brazil", "bronze": True, "stages": COPA_STAGES,
        "baseline": ("Copa América", "2000-01-01"), "bracket": _copa2019(),
        "groups": {
            "A": ["Brazil", "Bolivia", "Venezuela", "Peru"],
            "B": ["Argentina", "Colombia", "Paraguay", "Qatar"],
            "C": ["Uruguay", "Ecuador", "Japan", "Chile"],
        },
    },
    "copa2021": {
        "competition": "Copa América", "cutoff": "2021-06-13", "end": "2021-08-01",
        "champion": "Argentina", "bronze": True, "stages": COPA_STAGES,
        "baseline": ("Copa América", "2000-01-01"), "bracket": _copa2021(),
        "groups": {
            "A": ["Argentina", "Bolivia", "Uruguay", "Chile", "Paraguay"],
            "B": ["Brazil", "Colombia", "Venezuela", "Ecuador", "Peru"],
        },
    },
    "copa2024": {
        "competition": "Copa América", "cutoff": "2024-06-20", "end": "2024-08-01",
        "champion": "Argentina", "bronze": True, "stages": COPA_STAGES,
        "baseline": ("Copa América", "2000-01-01"),
        "bracket": _copa16("United States"),
        "groups": {
            "A": ["Argentina", "Peru", "Chile", "Canada"],
            "B": ["Mexico", "Ecuador", "Venezuela", "Jamaica"],
            "C": ["United States", "Uruguay", "Panama", "Bolivia"],
            "D": ["Brazil", "Colombia", "Paraguay", "Costa Rica"],
        },
    },
}


def _n_group_matches(groups: dict) -> int:
    return sum(len(ts) * (len(ts) - 1) // 2 for ts in groups.values())


def build_inputs(edition: str) -> dict:
    """Kickoff-frozen, out-of-sample inputs dict for run_tournament."""
    cfg = EDITIONS[edition]
    df = pd.read_csv(ROOT / "data/raw/results.csv")
    played = df.dropna(subset=["home_score", "away_score"]).sort_values("date", kind="stable")
    pre = played[played["date"] < cfg["cutoff"]]
    enriched, ratings = run_history(list(zip(
        pre["date"], pre["home_team"], pre["away_team"],
        pre["home_score"].astype(int), pre["away_score"].astype(int),
        pre["tournament"], pre["neutral"].astype(bool),
    )))

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

    comp, base_start = cfg["baseline"]
    base = pre[(pre["tournament"] == comp) & (pre["date"] >= base_start)]
    assert len(base) >= 50, f"{edition}: only {len(base)} baseline matches"
    baseline = float((base["home_score"] + base["away_score"]).mean())

    comp_matches = played[
        (played["tournament"] == cfg["competition"])
        & (played["date"] >= cfg["cutoff"]) & (played["date"] < cfg["end"])
    ]
    n_group = _n_group_matches(cfg["groups"])
    sched = comp_matches.head(n_group).rename(columns={"country": "host_country"})

    groups = cfg["groups"]
    teams = [t for g in sorted(groups) for t in groups[g]]
    team_group = {t: g for g, ts in groups.items() for t in ts}
    for _, r in sched.iterrows():  # cross-validate groups vs actual fixtures
        assert team_group[r["home_team"]] == team_group[r["away_team"]], (
            edition, r["home_team"], r["away_team"])

    return {
        "groups": groups,
        "schedule": sched.reset_index(drop=True),
        "bracket": cfg["bracket"],
        "teams": teams,
        "team_idx": {t: i for i, t in enumerate(teams)},
        "elo": np.array([ratings[t] for t in teams]),
        "params": {"alpha": alpha, "beta": beta, "rho": rho, "baseline": baseline},
        "third_slot_groups": third_slots(cfg["bracket"]),
        "stages": cfg["stages"],
        "_actual_knockout": comp_matches.iloc[n_group:],
    }


def actual_stage_sets(edition: str, inputs: dict) -> dict[str, set]:
    """Which teams actually reached each stage, from the played knockout
    matches (bronze match excluded from Final membership)."""
    cfg = EDITIONS[edition]
    ko = inputs["_actual_knockout"]
    if cfg["bronze"]:
        ko = pd.concat([ko.iloc[:-2], ko.iloc[[-1]]])  # drop the bronze match
    stages = cfg["stages"]
    sizes = {"R16": 8, "QF": 4, "SF": 2, "Final": 1}
    out: dict[str, set] = {}
    pos = 0
    for st in stages[:-1]:
        block = ko.iloc[pos:pos + sizes[st]]
        out[st] = set(block["home_team"]) | set(block["away_team"])
        pos += sizes[st]
    out["Champion"] = {cfg["champion"]}
    return out


def evaluate(edition: str, probs: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    """Per-stage Brier vs the base-rate forecast, plus champion rank."""
    cfg = EDITIONS[edition]
    actual = actual_stage_sets(edition, inputs)
    n_teams = len(inputs["teams"])
    rows = []
    for st in cfg["stages"][:-1]:
        y = probs.index.isin(list(actual[st])).astype(float)
        slots = 2 * {"R16": 8, "QF": 4, "SF": 2, "Final": 1}[st]
        base = slots / n_teams
        rows.append({
            "edition": edition, "stage": st,
            "model_brier": ((probs[st] - y) ** 2).mean(),
            "base_brier": base * (1 - base),
        })
    champ = cfg["champion"]
    rows.append({
        "edition": edition, "stage": "Champion",
        "model_brier": ((probs["Champion"] - probs.index.isin([champ])) ** 2).mean(),
        "base_brier": (1 / n_teams) * (1 - 1 / n_teams),
    })
    out = pd.DataFrame(rows)
    out["skill_%"] = (1 - out["model_brier"] / out["base_brier"]) * 100
    return out


def main() -> None:
    summaries = []
    for edition in EDITIONS:
        inputs = build_inputs(edition)
        probs = run_tournament(inputs, n_sims=100_000, seed=2026)
        probs.to_csv(ROOT / f"data/processed/backtest_{edition}.csv")
        ev = evaluate(edition, probs, inputs)
        summaries.append(ev)
        champ = EDITIONS[edition]["champion"]
        rank = list(probs.index).index(champ) + 1
        print(f"{edition}: champion {champ} predicted "
              f"{probs.loc[champ, 'Champion'] * 100:.1f}% (rank {rank}/{len(probs)})")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(ROOT / "data/processed/backtest_summary.csv", index=False)
    agg = summary.groupby("stage", sort=False)[["model_brier", "base_brier"]].mean()
    agg["skill_%"] = (1 - agg["model_brier"] / agg["base_brier"]) * 100
    print("\n=== mean Brier by stage across all 9 editions ===")
    print(agg.round(4).to_string())
    overall = (1 - summary["model_brier"].sum() / summary["base_brier"].sum()) * 100
    print(f"\noverall Brier skill vs base rate: {overall:.1f}%")


if __name__ == "__main__":
    main()
