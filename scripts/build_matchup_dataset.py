"""
Build a per-series matchup dataset.

For each playoff series across all seasons, join both teams' regular-season
stats into a single row and record which team won the series.

Output columns (per row = one playoff series):
  season, round, series_letter,
  team_hi, team_lo          <- higher/lower seed by regular-season points
  {stat}_hi, {stat}_lo      <- each team's regular-season stat
  {stat}_diff               <- hi - lo  (positive = hi team was "better")
  winner                    <- abbreviation of the team that won
  hi_won                    <- 1 if the higher-points team won, else 0
"""

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "20102011",
    "20112012",
    "20122013",
    "20132014",
    "20142015",
    "20152016",
    "20162017",
    "20172018",
    "20182019",
    "20192020",
    "20202021",
    "20212022",
    "20222023",
    "20232024",
    "20242025",
]

# Stats we want to compare between the two teams in each series
STATS = [
    "points",
    "wins",
    "win_pct",
    "goal_diff",
    "points_pct",
    "goals_for_pg",
    "goals_against_pg",  # lower is better — handled separately in analysis
]

# Stats where LOWER value is better (used in predictor_analysis.py)
LOWER_IS_BETTER = {"goals_against_pg"}


def load_standings(season: str) -> pd.DataFrame:
    path = PROCESSED_DIR / "standings_all.csv"
    df = pd.read_csv(path, dtype={"season": str})
    return df[df["season"] == season].set_index("team")


def parse_bracket(season: str) -> list[dict]:
    """
    Parse raw playoff bracket JSON into a flat list of series.
    Returns list of dicts: {round, series_letter, team_a, team_b, wins_a, wins_b}

    NHL API v1 actual structure:
    { "series": [ { "playoffRound": 1, "seriesLetter": "A",
                    "topSeedWins": 4, "bottomSeedWins": 1,
                    "topSeedTeam": {"abbrev": "FLA", ...},
                    "bottomSeedTeam": {"abbrev": "TBL", ...} } ] }
    """
    path = RAW_DIR / f"playoff_bracket_{season}.json"
    with open(path) as f:
        data = json.load(f)

    series_list = []

    for s in data.get("series", []):
        top = s.get("topSeedTeam") or {}
        bot = s.get("bottomSeedTeam") or {}

        team_a = top.get("abbrev") or top.get("triCode")
        team_b = bot.get("abbrev") or bot.get("triCode")

        # Wins are at the series level, not inside the team object
        wins_a = s.get("topSeedWins") or 0
        wins_b = s.get("bottomSeedWins") or 0

        if not team_a or not team_b:
            continue

        series_list.append({
            "round": s.get("playoffRound", "?"),
            "series_letter": s.get("seriesLetter", ""),
            "team_a": team_a,
            "team_b": team_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
        })

    return series_list


def build_matchup_row(season: str, s: dict, standings: pd.DataFrame) -> "dict | None":
    """
    Combine one series entry with both teams' regular-season stats.
    Returns None if either team is missing from standings or series isn't complete.
    """
    team_a, team_b = s["team_a"], s["team_b"]
    wins_a, wins_b = s["wins_a"], s["wins_b"]

    # Series must be finished (one team reached 4 wins)
    if wins_a != 4 and wins_b != 4:
        return None

    winner = team_a if wins_a == 4 else team_b

    if team_a not in standings.index or team_b not in standings.index:
        return None

    row_a = standings.loc[team_a]
    row_b = standings.loc[team_b]

    # Designate hi/lo by regular-season points so differentials are consistent
    if row_a["points"] >= row_b["points"]:
        hi, lo, row_hi, row_lo = team_a, team_b, row_a, row_b
    else:
        hi, lo, row_hi, row_lo = team_b, team_a, row_b, row_a

    record = {
        "season": season,
        "round": s["round"],
        "series_letter": s["series_letter"],
        "team_hi": hi,
        "team_lo": lo,
        "winner": winner,
        "hi_won": int(winner == hi),
        "series_result": f"{wins_a}-{wins_b}" if wins_a == 4 else f"{wins_b}-{wins_a}",
    }

    for stat in STATS:
        if stat not in row_hi or stat not in row_lo:
            continue
        record[f"{stat}_hi"] = row_hi[stat]
        record[f"{stat}_lo"] = row_lo[stat]
        record[f"{stat}_diff"] = row_hi[stat] - row_lo[stat]

    return record


def build_dataset() -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        standings_path = PROCESSED_DIR / "standings_all.csv"
        bracket_path = RAW_DIR / f"playoff_bracket_{season}.json"

        if not standings_path.exists() or not bracket_path.exists():
            print(f"  Skipping {season} — missing data files")
            continue

        standings = load_standings(season)
        series_list = parse_bracket(season)

        season_rows = 0
        for s in series_list:
            row = build_matchup_row(season, s, standings)
            if row:
                rows.append(row)
                season_rows += 1

        print(f"  {season}: {season_rows} completed series parsed")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building matchup dataset...")
    df = build_dataset()

    if df.empty:
        print("No data — run fetch_data.py and preprocess.py first.")
    else:
        out_path = PROCESSED_DIR / "playoff_matchups.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}  ({len(df)} series across {df['season'].nunique()} seasons)")
        print(df[["season", "round", "team_hi", "team_lo", "winner", "hi_won", "points_diff"]].to_string(index=False))
