"""Tournament simulator tests: structural invariants on real data (small n,
fixed seed), allocation constraint checks, and a regression lock."""

import numpy as np
import pandas as pd
import pytest

from wcsim.tournament import load_inputs, run_tournament, allocation_for_mask

INPUTS = load_inputs()


@pytest.fixture(scope="module")
def probs() -> pd.DataFrame:
    return run_tournament(INPUTS, n_sims=4000, seed=1)


STAGE_SLOTS = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}


def test_stage_probabilities_conserve_slots(probs):
    for stage, slots in STAGE_SLOTS.items():
        assert probs[stage].sum() == pytest.approx(slots, abs=1e-9), stage


def test_stage_probabilities_monotone_per_team(probs):
    cols = list(STAGE_SLOTS)
    for a, b in zip(cols, cols[1:]):
        assert (probs[a] >= probs[b] - 1e-12).all(), (a, b)


def test_all_48_teams_present_with_valid_probabilities(probs):
    assert len(probs) == 48
    assert ((probs >= 0) & (probs <= 1)).all().all()
    # nobody is mathematically eliminated before a ball is kicked
    assert (probs["R32"] > 0).all()


def test_favourites_outrank_minnows(probs):
    assert probs.loc["Spain", "Champion"] > probs.loc["Haiti", "Champion"]
    assert probs.loc["Argentina", "R16"] > probs.loc["Jordan", "R16"]


def test_allocation_respects_eligibility_and_uniqueness():
    rng = np.random.default_rng(3)
    eligibility = [set(s) for s in INPUTS["third_slot_groups"]]
    for _ in range(60):
        qualified = frozenset(rng.choice(12, size=8, replace=False).tolist())
        assignment = allocation_for_mask(qualified, INPUTS["third_slot_groups"])
        assert sorted(assignment) == sorted(qualified)  # each third used once
        for slot, grp in enumerate(assignment):
            assert grp in {ord(c) - 65 for c in INPUTS["third_slot_groups"][slot]}


def test_fixed_seed_regression(probs):
    # locked: champion prob of the Elo favourite at n=4000, seed=1
    assert probs.loc["Spain", "Champion"] == pytest.approx(0.25925, abs=1e-9)
