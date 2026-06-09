"""Structural invariants for the edition-config backtests, exercising the
generic engine paths the 2026 forecast doesn't: 4-slot thirds (Euro),
groups of five + direct P:k:g slots (Copa 2021)."""

import pandas as pd
import pytest

from wcsim.backtest import EDITIONS, build_inputs
from wcsim.tournament import run_tournament


@pytest.fixture(scope="module")
def euro2020():
    inputs = build_inputs("euro2020")
    return inputs, run_tournament(inputs, n_sims=1500, seed=1)


@pytest.fixture(scope="module")
def copa2021():
    inputs = build_inputs("copa2021")
    return inputs, run_tournament(inputs, n_sims=1500, seed=1)


def test_euro_slot_conservation(euro2020):
    _, probs = euro2020
    for stage, slots in {"R16": 16, "QF": 8, "SF": 4, "Final": 2, "Champion": 1}.items():
        assert probs[stage].sum() == pytest.approx(slots, abs=1e-9), stage


def test_copa21_slot_conservation(copa2021):
    _, probs = copa2021
    for stage, slots in {"QF": 8, "SF": 4, "Final": 2, "Champion": 1}.items():
        assert probs[stage].sum() == pytest.approx(slots, abs=1e-9), stage


def test_monotone_stage_probabilities(euro2020, copa2021):
    for _, probs in (euro2020, copa2021):
        cols = list(probs.columns)
        for a, b in zip(cols, cols[1:]):
            assert (probs[a] >= probs[b] - 1e-12).all(), (a, b)


def test_copa21_all_teams_present_groups_of_five(copa2021):
    inputs, probs = copa2021
    assert len(probs) == 10
    assert all(len(ts) == 5 for ts in inputs["groups"].values())
    # 8 of 10 reach the QF — every team has a realistic shot
    assert (probs["QF"] > 0.3).all()


def test_euro2020_thirds_machinery(euro2020):
    inputs, _ = euro2020
    assert len(inputs["third_slot_groups"]) == 4
    # multi-host venues: England appears as a knockout venue country
    venues = {m["host_country"] for r in inputs["bracket"].values() for m in r}
    assert "England" in venues and "Germany" in venues


def test_all_editions_group_fixture_counts():
    expected = {"wc2018": 48, "wc2022": 48, "euro2016": 36, "euro2020": 36,
                "euro2024": 36, "copa2016": 24, "copa2019": 18, "copa2021": 20,
                "copa2024": 24}
    for ed, n in expected.items():
        groups = EDITIONS[ed]["groups"]
        assert sum(len(t) * (len(t) - 1) // 2 for t in groups.values()) == n, ed
