"""CLI entrypoint: run the full 2026 forecast and write the deliverables.

    python -m wcsim.forecast [n_sims] [seed]

Writes data/processed/stage_probabilities.csv and FORECAST.md, and prints
the full sorted table.
"""

import sys
from pathlib import Path

from wcsim.tournament import ROOT, load_inputs, run_tournament


def main() -> None:
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    probs = run_tournament(load_inputs(), n_sims=n_sims, seed=seed)
    probs.to_csv(ROOT / "data/processed/stage_probabilities.csv")

    pct = (probs * 100).round(1)
    md = [
        "# World Cup 2026 — Stage Probabilities",
        "",
        f"{n_sims:,} Monte Carlo simulations, seed {seed}. "
        "Probabilities (%) of reaching each stage. Model: own-Elo + "
        "calibrated Poisson/Dixon-Coles (see CONTEXT.md, docs/adr/).",
        "",
        pct.to_markdown(),
        "",
    ]
    (ROOT / "FORECAST.md").write_text("\n".join(md), encoding="utf-8")
    print(pct.to_string())


if __name__ == "__main__":
    main()
