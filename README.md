# WC 2026 Tournament Forecaster

Monte Carlo simulation of the 2026 FIFA World Cup (48 teams, 12 groups, best
8 third-placed teams advance to a Round of 32). Produces calibrated per-team
probabilities of reaching each stage (R32, R16, QF, SF, Final, Champion).

**Scope**: probability table only. No odds ingestion, no staking — comparing
against bookmaker prices is a manual, human activity. See `CONTEXT.md` for the
domain language and `docs/adr/` for the key decisions.

## Model

1. **Team Elo** — own replica of the World Football Elo formula over the full
   international results history, 1872 → June 2026, friendlies included
   (ADR-0001). Elo differences, not absolute levels, drive everything.
2. **Elo → goals** — Poisson rates `λ = (baseline/2)·exp(±β·Δelo)`, β fitted by
   MLE on ~15.8k internationals 2010+ (ADR-0002). Baseline = mean total goals
   at World Cups 2010–2022 (≈2.57).
3. **Scorelines** — independent Poisson + Dixon-Coles τ correction on the
   0-0/1-0/0-1/1-1 cells (ρ ≈ −0.046, fitted on the same set).
4. **Host bonus** — +100 Elo for USA/Mexico/Canada only in matches physically
   played in their own country (venue-conditional).
5. **Knockouts** — draws go to extra time (Poisson at λ/3), then a fair-coin
   shootout. Group ties: points → GD → GF → head-to-head → random. Third-place
   allocation: constraint-satisfying matching per FIFA eligibility (ADR-0003).

## Quick start

```
python -m venv .venv
.venv\Scripts\pip install -e .
```

Run the forecast (optional args: n_sims, seed — defaults 100000, 2026):

```
.venv\Scripts\python -m wcsim.forecast        # writes FORECAST.md + stage_probabilities.csv
```

Rebuild the pipeline from scratch (results CSV → ratings → params → forecast):

```
curl -sL -o data/raw/results.csv https://raw.githubusercontent.com/martj42/international_results/master/results.csv
.venv\Scripts\python -m wcsim.build_ratings
.venv\Scripts\python -m wcsim.build_params
.venv\Scripts\python -m wcsim.forecast
.venv\Scripts\python -m wcsim.backtest        # optional: 2018/2022 frozen-Elo validation
```

Tests (the model core is TDD with fixed-seed regression locks):

```
.venv\Scripts\python -m pytest tests/ -q
```

## Validation

- Replica Elo vs eloratings.net: Spearman rank correlation 0.979 across the 48
  qualified teams (absolute levels differ by a near-affine offset that the
  calibrated slope absorbs — see ADR-0001).
- Frozen-Elo backtests, fully out-of-sample: beats the base-rate Brier score on
  6 of 8 stage/year combinations (2018 + 2022), average improvement +12.4%.
  2022 champion Argentina was ranked 2nd pre-tournament; 2018 champion France
  6th. Known caveat: pure-Elo models concentrate more probability on top
  favourites than the market does (no rating-uncertainty shrinkage).

## Layout

```
data/static/      groups.json, group_schedule.csv, bracket.json (fixed public facts)
data/raw/         results.csv (downloaded, gitignored)
data/processed/   ratings, params.json, stage_probabilities.csv, backtests
src/wcsim/        elo, calibrate, scores, match_sim, tournament, backtest, forecast
tests/            pytest suite incl. fixed-seed regression locks
docs/adr/         decision records
```
