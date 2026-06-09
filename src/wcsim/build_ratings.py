"""Replay the full results history through the Elo engine.

Outputs:
  data/processed/matches_with_elo.csv  — every played match + pre-match Elo for both sides
  data/processed/current_ratings.csv   — final rating per team
"""

from pathlib import Path

import pandas as pd

from wcsim.elo import run_history

ROOT = Path(__file__).resolve().parents[2]


def build() -> None:
    df = pd.read_csv(ROOT / "data/raw/results.csv")
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played = played.sort_values("date", kind="stable")
    matches = list(
        zip(
            played["date"],
            played["home_team"],
            played["away_team"],
            played["home_score"].astype(int),
            played["away_score"].astype(int),
            played["tournament"],
            played["neutral"].astype(bool),
        )
    )
    enriched, ratings = run_history(matches)
    out = pd.DataFrame(
        enriched,
        columns=[
            "date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral", "home_elo_pre", "away_elo_pre",
        ],
    )
    processed = ROOT / "data/processed"
    processed.mkdir(exist_ok=True)
    out.to_csv(processed / "matches_with_elo.csv", index=False)
    pd.Series(ratings, name="elo").rename_axis("team").sort_values(
        ascending=False
    ).to_csv(processed / "current_ratings.csv")
    print(f"{len(out)} matches replayed, {len(ratings)} teams rated")


if __name__ == "__main__":
    build()
