"""Monte Carlo simulation of the 2026 World Cup.

All stages are vectorized across simulations. Group ranking uses an integer
sort key (points, GD, GF) with a pairwise head-to-head bonus layer that only
activates between teams tied on all three — for multi-way ties this reduces
to FIFA's "points among tied teams" criterion — then a random component
standing in for fair play/lots (see CONTEXT.md: Tiebreak Cascade).

Third-place allocation: the 8 best thirds are matched to the bracket's
T-slots by deterministic backtracking against the eligibility lists in
bracket.json, memoized per qualified-combination (ADR-0003).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from wcsim.match_sim import knockout_winner, sample_dc

ROOT = Path(__file__).resolve().parents[2]
HOST_BONUS = 100.0
HOSTS = {"United States", "Mexico", "Canada"}

# sort-key strata: points, GD (+100 offset), GF, h2h bonus, random
_K_PTS, _K_GD, _K_GF, _K_H2H, _K_RND = 10**13, 10**10, 10**7, 10**4, 10**2


def load_inputs(data_dir: Path | None = None) -> dict:
    d = data_dir or ROOT / "data"
    groups = json.loads((d / "static/groups.json").read_text(encoding="utf-8"))
    schedule = pd.read_csv(d / "static/group_schedule.csv")
    bracket = json.loads((d / "static/bracket.json").read_text(encoding="utf-8"))
    ratings = pd.read_csv(d / "processed/current_ratings.csv", index_col="team")["elo"]
    params = json.loads((d / "processed/params.json").read_text())
    teams = [t for g in sorted(groups) for t in groups[g]]
    third_slot_groups = [
        s["away"][2:] for s in bracket["round_of_32"] if s["away"].startswith("T:")
    ]
    return {
        "groups": groups,
        "schedule": schedule,
        "bracket": bracket,
        "elo": np.array([ratings[t] for t in teams]),
        "teams": teams,
        "team_idx": {t: i for i, t in enumerate(teams)},
        "params": params,
        "third_slot_groups": third_slot_groups,
    }


def _effective_diff(home: str, away: str, host_country: str, elo: np.ndarray, idx: dict) -> float:
    d = elo[idx[home]] - elo[idx[away]]
    if home in HOSTS and home == host_country:
        d += HOST_BONUS
    if away in HOSTS and away == host_country:
        d -= HOST_BONUS
    return float(d)


def allocation_for_mask(qualified: frozenset, third_slot_groups: list[str]) -> tuple:
    """Deterministically assign 8 qualified third-place groups (ints 0-11) to
    the 8 T-slots, respecting eligibility. Most-constrained slot first."""
    eligible = [
        sorted(ord(c) - 65 for c in s if (ord(c) - 65) in qualified)
        for s in third_slot_groups
    ]
    order = sorted(range(8), key=lambda i: len(eligible[i]))
    assignment: dict[int, int] = {}

    def backtrack(k: int, used: set) -> bool:
        if k == 8:
            return True
        slot = order[k]
        for g in eligible[slot]:
            if g not in used:
                assignment[slot] = g
                if backtrack(k + 1, used | {g}):
                    return True
                del assignment[slot]
        return False

    if not backtrack(0, set()):
        raise ValueError(f"No valid third-place allocation for {sorted(qualified)}")
    return tuple(assignment[i] for i in range(8))


def _rank_groups(
    inputs: dict, gh: np.ndarray, ga: np.ndarray, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (winner, runner, third) team-index arrays of shape (12, n) and
    the thirds' (pts, gd, gf) base key for cross-group ranking."""
    sched = inputs["schedule"]
    idx = inputs["team_idx"]
    n_teams = len(inputs["teams"])
    pts = np.zeros((n_teams, n), dtype=np.int64)
    gd = np.zeros((n_teams, n), dtype=np.int64)
    gf = np.zeros((n_teams, n), dtype=np.int64)
    for f, row in sched.iterrows():
        h, a = idx[row["home_team"]], idx[row["away_team"]]
        hg, ag = gh[f], ga[f]
        pts[h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gd[h] += hg - ag
        gd[a] += ag - hg
        gf[h] += hg
        gf[a] += ag

    group_letters = sorted(inputs["groups"])
    n_groups = len(group_letters)
    winner = np.empty((n_groups, n), dtype=np.int64)
    runner = np.empty((n_groups, n), dtype=np.int64)
    third = np.empty((n_groups, n), dtype=np.int64)
    third_base = np.empty((n_groups, n), dtype=np.int64)
    for gi, letter in enumerate(group_letters):
        members = [idx[t] for t in inputs["groups"][letter]]
        base = np.stack(
            [pts[t] * _K_PTS + (gd[t] + 100) * _K_GD + gf[t] * _K_GF for t in members]
        )
        # head-to-head bonus among teams tied on (pts, GD, GF)
        h2h = np.zeros((4, n), dtype=np.int64)
        gmask = (
            (sched["home_team"].map(idx).isin(members))
            & (sched["away_team"].map(idx).isin(members))
        )
        for f in sched.index[gmask]:
            h, a = idx[sched.at[f, "home_team"]], idx[sched.at[f, "away_team"]]
            i, j = members.index(h), members.index(a)
            tied = base[i] == base[j]
            hg, ag = gh[f], ga[f]
            h2h[i] += tied * np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            h2h[j] += tied * np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        key = base + h2h * _K_H2H + rng.integers(0, 100, size=(4, n)) * _K_RND
        order = np.argsort(-key, axis=0, kind="stable")
        marr = np.array(members)
        winner[gi] = marr[order[0]]
        runner[gi] = marr[order[1]]
        third[gi] = marr[order[2]]
        third_base[gi] = np.take_along_axis(base, order[2][None, :], axis=0)[0]
    return winner, runner, third, third_base, pts


def _qualify_thirds(
    third_base: np.ndarray, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Boolean (12, n): which groups' thirds are among the best 8."""
    key = third_base + rng.integers(0, 100, size=(12, n)) * _K_RND
    order = np.argsort(-key, axis=0, kind="stable")
    qualified = np.zeros((12, n), dtype=bool)
    np.put_along_axis(qualified, order[:8], True, axis=0)
    return qualified


def run_tournament(inputs: dict, n_sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = inputs["params"]
    alpha_wc = float(np.log(p["wc_baseline"] / 2.0))
    beta, rho = p["beta"], p["rho"]
    elo, idx, teams = inputs["elo"], inputs["team_idx"], inputs["teams"]
    n = n_sims

    # --- group stage: 72 fixtures, fixed λs per fixture ---
    sched = inputs["schedule"]
    diffs = np.array(
        [
            _effective_diff(r["home_team"], r["away_team"], r["host_country"], elo, idx)
            for _, r in sched.iterrows()
        ]
    )
    lh = np.repeat(np.exp(alpha_wc + beta * diffs), n)
    la = np.repeat(np.exp(alpha_wc - beta * diffs), n)
    h, a = sample_dc(lh, la, rho, rng)
    gh = h.reshape(len(sched), n)
    ga = a.reshape(len(sched), n)

    winner, runner, third, third_base, _ = _rank_groups(inputs, gh, ga, rng, n)
    qualified = _qualify_thirds(third_base, rng, n)

    # --- third-place slot allocation, memoized over the ≤495 combinations ---
    slot_groups = inputs["third_slot_groups"]
    masks = (qualified * (1 << np.arange(12))[:, None]).sum(axis=0)
    cached = lru_cache(maxsize=None)(
        lambda m: allocation_for_mask(
            frozenset(g for g in range(12) if m >> g & 1), slot_groups
        )
    )
    slot_third = np.empty((8, n), dtype=np.int64)  # group index per T-slot per sim
    for m in np.unique(masks):
        sel = masks == m
        for s, g in enumerate(cached(int(m))):
            slot_third[s, sel] = g

    # --- knockout bracket ---
    group_no = {chr(65 + i): i for i in range(12)}
    match_winner: dict[int, np.ndarray] = {}
    t_slot_counter = [0]

    def resolve(ref: str) -> np.ndarray:
        kind, val = ref.split(":")
        if kind == "W":
            return winner[group_no[val]]
        if kind == "R":
            return runner[group_no[val]]
        if kind == "M":
            return match_winner[int(val)]
        s = t_slot_counter[0]
        t_slot_counter[0] += 1
        return third[slot_third[s], np.arange(n)]

    reached = {st: np.zeros((48, n), dtype=bool) for st in
               ["R32", "R16", "QF", "SF", "Final", "Champion"]}
    rounds = [
        ("round_of_32", "R16"), ("round_of_16", "QF"), ("quarter_finals", "SF"),
        ("semi_finals", "Final"), ("final", "Champion"),
    ]
    sims = np.arange(n)
    for round_key, win_stage in rounds:
        for m in inputs["bracket"][round_key]:
            home, away = resolve(m["home"]), resolve(m["away"])
            if round_key == "round_of_32":
                reached["R32"][home, sims] = True
                reached["R32"][away, sims] = True
            d = elo[home] - elo[away]
            hc = m["host_country"]
            for t in HOSTS & set(teams):
                if t == hc:
                    d = d + HOST_BONUS * (home == idx[t]) - HOST_BONUS * (away == idx[t])
            lh = np.exp(alpha_wc + beta * d)
            la = np.exp(alpha_wc - beta * d)
            first_wins = knockout_winner(lh, la, rho, rng)
            w = np.where(first_wins, home, away)
            match_winner[m["match"]] = w
            reached[win_stage][w, sims] = True

    out = pd.DataFrame(
        {st: reached[st].mean(axis=1) for st in reached}, index=pd.Index(teams, name="team")
    )
    return out.sort_values("Champion", ascending=False)
