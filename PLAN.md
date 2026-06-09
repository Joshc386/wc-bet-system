# WC 2026 Tournament Forecaster — Build Plan

Status: awaiting approval. No code until signed off.
Decisions below were resolved in the 2026-06-09 grilling session; vocabulary in `CONTEXT.md`, key trade-offs in `docs/adr/`.

## What we're building

A Monte Carlo simulator of the 2026 World Cup (48 teams, 12 groups, best 8 thirds → Round of 32) that outputs one table: for every team, the probability of reaching the R32, R16, QF, SF, Final, and winning. Probability table only — no odds ingestion, no staking.

## Model summary (as agreed)

1. **Team Elo** — self-computed World Football Elo over the full international results history (ADR-0001), friendlies included at standard K. Sanity-checked vs eloratings.net (±25 pts on top teams).
2. **Strength → goals** — λ_i = (Baseline/2)·exp(±β·Elodiff), β calibrated on all internationals 2010+ (ADR-0002: multiplicative, not the linear split — avoids negative λ at this tournament's 600+ Elo gaps). Baseline = mean total goals at World Cups 2010–2022.
3. **Scoreline** — independent Poisson per team + Dixon-Coles τ on the four low-score cells; τ by MLE on the 2010+ set. (Full bivariate Poisson rejected — τ alone fixes the draw deficit.)
4. **Host Bonus** — +100 Elo for USA/Mexico/Canada only in matches physically in their own country (venue-conditional).
5. **Knockouts** — drawn after 90' → ET as Poisson at λ/3 → penalties as a fair coin flip.
6. **Groups** — full FIFA tiebreak cascade (points → GD → GF → head-to-head → random); same cascade ranks the 12 thirds; FIFA allocation table maps the 8 best thirds into the bracket.
7. **Runs** — 100,000 simulated tournaments per run.

## Phases

### Phase 1 — Data (no model code)
- Download the martj42 Kaggle results CSV; **verify it covers the March 2026 playoffs and June 2026 friendlies**; top up manually if not.
- Build static fixture files from web research: `groups.json` (12 groups × 4 teams), `schedule.csv` (104 matches, dates, venues, host country per venue), `bracket.json` (R32 template + third-place allocation table).
- Eyeball-verification checklist: 48 teams, each in exactly one group; 72 group matches + 32 knockout slots; allocation table covers all 495 third-place combinations (or FIFA's published template form).

### Phase 2 — Elo engine (TDD)
- Replicate World Football Elo: K by match importance, goal-margin multiplier, +100 home advantage, over the full results history.
- Outputs: Elo-at-match-time for every historical match; current rating per 2026 qualifier.
- Acceptance: top-10 current ratings within ±25 of eloratings.net.

### Phase 3 — Calibration (TDD, fixed-seed regression tests)
- Fit β (log-link goal model vs Elo diff) on all internationals 2010+ using Elo-at-match-time from Phase 2.
- Fit Dixon-Coles τ by MLE on the same set.
- Estimate Baseline from WC 2010–2022 matches.
- Acceptance: parameters within plausible published ranges (β ≈ such that 100 Elo ≈ 0.6–0.7 expected GD at baseline; τ < 0 small; baseline ≈ 2.5–2.7). Known-output regression test locks all three.

### Phase 4 — Match simulator (TDD)
- Elo diff (+ Host Bonus) → λs → DC-corrected scoreline sample; knockout mode adds ET/pens resolution.
- Tests: symmetry (equal Elos → mirror-image probabilities), draw rate vs τ=0 control, no negative λ at 800-point gaps, fixed-seed regression.

### Phase 5 — Tournament simulator (TDD)
- Group stage from `schedule.csv`, tiebreak cascade, third-place ranking + allocation, knockout bracket with venue-conditional Host Bonus, 100k runs.
- Tests: probabilities sum correctly (exactly 1 champion; 32 R32 teams per sim), every team's stage probabilities are monotone non-increasing across stages, fixed-seed regression.

### Phase 6 — Backtest + outputs
- Freeze Elo at 2018 and 2022 kickoff; simulate those tournaments (32-team format variant of the sim); compare predicted stage probabilities to actual outcomes (winners' predicted ranks, calibration of "reached KO stage" buckets).
- Final outputs: `stage_probabilities.csv` (team × stage) + sorted markdown table; README with run instructions.

## Explicitly out of scope
- Odds ingestion, value detection, staking (manual, human).
- Player/squad-level inputs, injury adjustments, pedigree factors.
- Travel, altitude, rest-day effects.
- Databases, APIs, n8n — plain Python package + CSVs, one CLI entrypoint.
