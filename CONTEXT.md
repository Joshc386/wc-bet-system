# WC 2026 Tournament Forecaster

A Monte Carlo simulation of the 2026 FIFA World Cup (48 teams) that produces calibrated per-team probabilities of reaching each tournament stage. Output is a probability table for human consumption — no odds ingestion, no staking logic, no live capital.

## Language

**Stage Probability**:
The probability that a team reaches a given tournament stage (Round of 32, Round of 16, Quarter-final, Semi-final, Final, Champion), estimated by Monte Carlo simulation.
_Avoid_: odds, price (those are bookmaker concepts — out of scope)

**Team Elo**:
The single team-strength input: a World Football Elo rating per national team, computed by our own engine over the full historical results dataset (see ADR-0001).
_Avoid_: player Elo, squad rating, individual ratings (no such input exists in this system)

**Supremacy**:
Expected goal difference between two teams, implied by their Elo difference. Realised multiplicatively: λ_i = (Baseline/2)·exp(±β·Elodiff), with β calibrated on all internationals 2010+ (see ADR-0002).
_Avoid_: edge, advantage (ambiguous); the linear (baseline ± supremacy)/2 split (rejected — goes negative at this tournament's Elo gaps)

**Baseline**:
The expected total goals per match under equal teams, estimated from World Cup matches 2010–2022 (≈2.5–2.7). Not hardcoded.

**Scoreline Model**:
Independent Poisson goals per team (λ from Baseline and Supremacy), with the Dixon-Coles τ correction applied to the 0-0, 1-0, 0-1 and 1-1 cells; τ fitted by MLE on the same 2010+ calibration set.
_Avoid_: bivariate Poisson (considered and rejected — τ alone fixes the draw deficit; a shared λ₃ component would double-correct)

**Host Bonus**:
+100 Elo applied to a host nation (USA, Mexico, Canada) only in matches physically played in that team's own country, per the known venue schedule.
_Avoid_: home advantage "for hosts" unqualified — it is venue-conditional, not team-conditional

**Knockout Resolution**:
Drawn after 90' → extra time simulated as Poisson at one-third of each side's λ → if still level, penalties as a fair coin flip.

**Backtest**:
Rerunning the full pipeline with Elo frozen at a past tournament's kickoff and comparing predicted Stage Probabilities to actual outcomes — fully out-of-sample (parameters refit on pre-tournament data, baseline from prior editions of the same competition). Nine editions: WC 2018/2022, Euro 2016/2020/2024, Copa América 2016/2019/2021/2024. The Euro editions matter most: their best-thirds advancement is the closest analogue to 2026's format. History is never a model input (Elo already encodes it).
_Avoid_: pedigree, tournament experience (rejected as model inputs)

**Tiebreak Cascade**:
Points → goal difference → goals scored → head-to-head (points, GD, goals among tied teams) → random. Used both within groups and to rank the 12 third-placed teams for the 8 advancing spots.

## Relationships

- **Team Elo** difference (plus any **Host Bonus**) → **Supremacy** via the calibrated slope
- **Supremacy** + baseline scoring rate → per-team λ → **Scoreline Model** → match result
- Match results + **Tiebreak Cascade** + the FIFA third-place allocation table → bracket → **Stage Probability** per team over many simulated tournaments

## Data sources

- **Results history** (1872 → June 2026, incl. friendlies): the martj42 Kaggle "International football results" CSV, manually topped up if it lags the June 2026 friendlies. Feeds the Elo engine and all calibration.
- **Tournament structure** (12 groups, 104-match schedule with venues, R32 bracket template, FIFA third-place allocation table): static JSON/CSV files checked into the repo, populated once from web research. No runtime APIs.

## Example dialogue

> **Dev:** "Does Brazil's **Team Elo** get a boost from their five titles?"
> **Domain expert:** "No — past World Cups only enter through the **Backtest**. Elo already encodes every historical result; adding pedigree would double-count."
> **Dev:** "And when Mexico plays a semi-final in Dallas, do they get the **Host Bonus**?"
> **Domain expert:** "No — the bonus is venue-conditional. Dallas isn't Mexico, so that match is neutral for them."

## Flagged ambiguities

- "Bet System" (folder name) vs actual scope — resolved 2026-06-09: the deliverable is the **probability table only**. Comparing against bookmaker odds and staking are manual, human activities outside this system.
- "individual elo ratings" in the original brief — resolved 2026-06-09: there is no individual-player input. **Team Elo** is the only strength input. Player/squad-level adjustments are explicitly out of scope.
