"""
Clean and transform raw NHL standings JSON into processed CSVs.
Reads from data/raw/, writes to data/processed/.

Each row = one team's full regular-season stat line for that season.
"""

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "20172018",
    "20182019",
    "20192020",
    "20202021",
    "20212022",
    "20222023",
    "20232024",
]


def parse_standings(season: str) -> pd.DataFrame:
    path = RAW_DIR / f"standings_{season}.json"
    with open(path) as f:
        data = json.load(f)

    records = []
    for team in data.get("standings", []):
        gp = team.get("gamesPlayed") or 1  # avoid div/0
        gf = team.get("goalFor") or 0
        ga = team.get("goalAgainst") or 0
        wins = team.get("wins") or 0
        losses = team.get("losses") or 0
        ot_losses = team.get("otLosses") or 0
        points = team.get("points") or 0

        records.append({
            "season": season,
            "team": team.get("teamAbbrev", {}).get("default"),
            "conference": team.get("conferenceName"),
            "division": team.get("divisionName"),
            "games_played": gp,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "points": points,
            "goals_for": gf,
            "goals_against": ga,
            # Derived
            "goal_diff": gf - ga,
            "points_pct": points / (gp * 2),
            "goals_for_pg": gf / gp,
            "goals_against_pg": ga / gp,
            "win_pct": wins / gp,
        })

    return pd.DataFrame(records)


def build_all_seasons() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        path = RAW_DIR / f"standings_{season}.json"
        if not path.exists():
            print(f"  Missing: {path} — skipping")
            continue
        df = parse_standings(season)
        frames.append(df)
        print(f"  Parsed {season}: {len(df)} teams")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("Parsing standings for all seasons...")
    all_standings = build_all_seasons()

    out_path = PROCESSED_DIR / "standings_all.csv"
    all_standings.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(all_standings)} rows)")
    print(all_standings[["season", "team", "points", "goal_diff", "points_pct"]].head(10).to_string(index=False))
