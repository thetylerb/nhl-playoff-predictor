"""
Download and process MoneyPuck team-level expected goals data.

Sources:
  - Historical (2008-2024): peter-tanner.com bulk zip (free, no Referer needed)
  - Current (2025):         moneypuck.com direct CSV (requires Referer header)

Output: data/processed/moneypuck_xg.csv
  One row per team per season (first year of season, e.g. 2024 = 2024-25).
  Filtered to situation == "all" (all game states combined).
  Key columns:
    team, season, games_played,
    xgf_pg  — scoreVenueAdjustedxGoalsFor  / games_played
    xga_pg  — scoreVenueAdjustedxGoalsAgainst / games_played
    gf_pg   — raw goalsFor  / games_played  (kept for comparison)
    ga_pg   — raw goalsAgainst / games_played
"""

import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HIST_URL = (
    "https://peter-tanner.com/moneypuck/downloads/"
    "historicalOneRowPerSeason/teams_2008_to_2024.zip"
)
CURRENT_YEAR = 2025
CURRENT_URL = (
    f"https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    f"{CURRENT_YEAR}/regular/teams.csv"
)
REFERER = "https://moneypuck.com"

KEEP_COLS = [
    "team", "season", "games_played",
    "scoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsAgainst",
    "goalsFor",
    "goalsAgainst",
]

# MoneyPuck uses legacy multi-char abbreviations; normalize to NHL API standard.
# PHX kept as-is: NHL API also used PHX for the Coyotes pre-Arizona rebranding.
ABBREV_MAP = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["situation"] == "all"].copy()
    df = df[KEEP_COLS].copy()
    df["team"] = df["team"].replace(ABBREV_MAP)
    df["season"] = df["season"].astype(int)
    df["xgf_pg"] = df["scoreVenueAdjustedxGoalsFor"]  / df["games_played"]
    df["xga_pg"] = df["scoreVenueAdjustedxGoalsAgainst"] / df["games_played"]
    df["gf_pg"]  = df["goalsFor"]    / df["games_played"]
    df["ga_pg"]  = df["goalsAgainst"] / df["games_played"]
    return df[["team", "season", "games_played", "xgf_pg", "xga_pg", "gf_pg", "ga_pg"]]


def fetch_historical() -> pd.DataFrame:
    print("Fetching historical MoneyPuck data (2008-2024)...")
    r = requests.get(HIST_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"))
    return _clean(df)


def fetch_current() -> pd.DataFrame:
    print(f"Fetching current MoneyPuck data (season {CURRENT_YEAR})...")
    r = requests.get(CURRENT_URL, headers={"Referer": REFERER}, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return _clean(df)


if __name__ == "__main__":
    hist = fetch_historical()
    curr = fetch_current()

    combined = pd.concat([hist, curr], ignore_index=True)
    combined = combined.sort_values(["season", "team"]).reset_index(drop=True)

    out = PROCESSED_DIR / "moneypuck_xg.csv"
    combined.to_csv(out, index=False)
    print(f"\nSaved {len(combined)} rows to {out}")
    print(f"Seasons: {sorted(combined['season'].unique())}")
    print(f"\nSample (2025 season):")
    print(combined[combined["season"] == 2025].to_string(index=False))
