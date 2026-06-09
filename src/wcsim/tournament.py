"""Format-driven Monte Carlo tournament simulator (2026 World Cup + backtests).

All stages are vectorized across simulations. Group ranking uses an integer
sort key (points, GD, GF) with a pairwise head-to-head bonus layer that only
activates between teams tied on all three — for multi-way ties this reduces
to FIFA's "points among tied teams" criterion — then a random component
standing in for fair play/lots (see CONTEXT.md: Tiebreak Cascade).

Best-third allocation: qualified thirds are matched to the bracket's
T-slots by deterministic backtracking against eligibility lists, memoized
per qualified-combination (ADR-0003).

Slot grammar in brackets: "W:g" group winner, "R:g" runner-up,
"P:k:g" k-th place of group g (direct, no qualification), "T:<groups>"
best-third slot with eligible groups, "M:n" winner of match n.
Host bonus rule (uniform across editions, incl. multi-host Euro 2020):
+100 Elo iff the team's name equals the match venue's country.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from wcsim.match_sim import knockout_winner, sample_dc

ROOT = Path(__file__).resolve().parents[2]
HOST_BONUS = 100.0
DEFAULT_STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]

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
    return {
        "groups": groups,
        "schedule": schedule,
        "bracket": bracket,
        "elo": np.array([ratings[t] for t in teams]),
        "teams": teams,
        "team_idx": {t: i for i, t in enumerate(teams)},
        "params": params,
        "third_slot_groups": third_slots(bracket),
        "stages": DEFAULT_STAGES,
    }


def third_slots(bracket: dict) -> list[str]:
    """Eligible-group strings of every T-slot, in bracket order."""
    return [
        side[2:]
        for rnd in bracket.values()
        if isinstance(rnd, list)
        for m in rnd
        for side in (m["home"], m["away"])
        if side.startswith("T:")
    ]


def _effective_diff(home: str, away: str, host_country: str, elo: np.ndarray, idx: dict) -> float:
    d = elo[idx[home]] - elo[idx[away]]
    if home == host_country:
        d += HOST_BONUS
    if away == host_country:
        d -= HOST_BONUS
    return float(d)


def allocation_for_mask(qualified: frozenset, third_slot_groups: list[str]) -> tuple:
    """Deterministically assign the qualified third-place groups (ints) to
    the T-slots, respecting eligibility. Most-constrained slot first."""
    n_slots = len(third_slot_groups)
    eligible = [
        sorted(ord(c) - 65 for c in s if (ord(c) - 65) in qualified)
        for s in third_slot_groups
    ]
    order = sorted(range(n_slots), key=lambda i: len(eligible[i]))
    assignment: dict[int, int] = {}

    def backtrack(k: int, used: set) -> bool:
        if k == n_slots:
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
    return tuple(assignment[i] for i in range(n_slots))


def _rank_groups(
    inputs: dict, gh: np.ndarray, ga: np.ndarray, rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ranked, pos_base): ranked[g, k] is the team index at position
    k of group g per sim, shape (n_groups, group_size, n); pos_base is the
    (pts, GD, GF) base key at each position, for cross-group third ranking."""
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
    size = len(inputs["groups"][group_letters[0]])
    ranked = np.empty((n_groups, size, n), dtype=np.int64)
    pos_base = np.empty((n_groups, size, n), dtype=np.int64)
    for gi, letter in enumerate(group_letters):
        members = [idx[t] for t in inputs["groups"][letter]]
        base = np.stack(
            [pts[t] * _K_PTS + (gd[t] + 100) * _K_GD + gf[t] * _K_GF for t in members]
        )
        # head-to-head bonus among teams tied on (pts, GD, GF)
        h2h = np.zeros((size, n), dtype=np.int64)
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
        key = base + h2h * _K_H2H + rng.integers(0, 100, size=(size, n)) * _K_RND
        order = np.argsort(-key, axis=0, kind="stable")
        ranked[gi] = np.array(members)[order]
        pos_base[gi] = np.take_along_axis(base, order, axis=0)
    return ranked, pos_base


def _qualify_thirds(
    third_base: np.ndarray, n_qualify: int, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Boolean (n_groups, n): which groups' thirds are among the best n_qualify."""
    key = third_base + rng.integers(0, 100, size=third_base.shape) * _K_RND
    order = np.argsort(-key, axis=0, kind="stable")
    qualified = np.zeros(third_base.shape, dtype=bool)
    np.put_along_axis(qualified, order[:n_qualify], True, axis=0)
    return qualified


def run_tournament(inputs: dict, n_sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = inputs["params"]
    alpha_t = float(np.log(p.get("baseline", p.get("wc_baseline")) / 2.0))
    beta, rho = p["beta"], p["rho"]
    elo, idx, teams = inputs["elo"], inputs["team_idx"], inputs["teams"]
    stages = inputs["stages"]
    n = n_sims

    # --- group stage: fixed λs per fixture ---
    sched = inputs["schedule"]
    diffs = np.array(
        [
            _effective_diff(r["home_team"], r["away_team"], r["host_country"], elo, idx)
            for _, r in sched.iterrows()
        ]
    )
    lh = np.repeat(np.exp(alpha_t + beta * diffs), n)
    la = np.repeat(np.exp(alpha_t - beta * diffs), n)
    h, a = sample_dc(lh, la, rho, rng)
    gh = h.reshape(len(sched), n)
    ga = a.reshape(len(sched), n)

    ranked, pos_base = _rank_groups(inputs, gh, ga, rng, n)
    n_groups = ranked.shape[0]

    # --- best-third qualification + slot allocation (if format has T-slots) ---
    slot_groups = inputs["third_slot_groups"]
    if slot_groups:
        qualified = _qualify_thirds(pos_base[:, 2], len(slot_groups), rng, n)
        masks = (qualified * (1 << np.arange(n_groups))[:, None]).sum(axis=0)
        cached = lru_cache(maxsize=None)(
            lambda m: allocation_for_mask(
                frozenset(g for g in range(n_groups) if m >> g & 1), slot_groups
            )
        )
        slot_third = np.empty((len(slot_groups), n), dtype=np.int64)
        for m in np.unique(masks):
            sel = masks == m
            for s, g in enumerate(cached(int(m))):
                slot_third[s, sel] = g

    # --- knockout bracket ---
    group_no = {chr(65 + i): i for i in range(n_groups)}
    match_winner: dict[int, np.ndarray] = {}
    t_slot_counter = [0]
    sims = np.arange(n)

    def resolve(ref: str) -> np.ndarray:
        parts = ref.split(":")
        kind = parts[0]
        if kind == "W":
            return ranked[group_no[parts[1]], 0]
        if kind == "R":
            return ranked[group_no[parts[1]], 1]
        if kind == "P":
            return ranked[group_no[parts[2]], int(parts[1]) - 1]
        if kind == "M":
            return match_winner[int(parts[1])]
        s = t_slot_counter[0]
        t_slot_counter[0] += 1
        return ranked[slot_third[s], 2, sims]

    reached = {st: np.zeros((len(teams), n), dtype=bool) for st in stages}
    round_keys = [k for k, v in inputs["bracket"].items() if isinstance(v, list)]
    for round_key, win_stage in zip(round_keys, stages[1:]):
        for m in inputs["bracket"][round_key]:
            home, away = resolve(m["home"]), resolve(m["away"])
            if round_key == round_keys[0]:
                reached[stages[0]][home, sims] = True
                reached[stages[0]][away, sims] = True
            d = elo[home] - elo[away]
            hc = m["host_country"]
            if hc in idx:
                d = d + HOST_BONUS * (home == idx[hc]) - HOST_BONUS * (away == idx[hc])
            lh = np.exp(alpha_t + beta * d)
            la = np.exp(alpha_t - beta * d)
            first_wins = knockout_winner(lh, la, rho, rng)
            w = np.where(first_wins, home, away)
            match_winner[m["match"]] = w
            reached[win_stage][w, sims] = True

    out = pd.DataFrame(
        {st: reached[st].mean(axis=1) for st in reached}, index=pd.Index(teams, name="team")
    )
    return out.sort_values(stages[-1], ascending=False)
