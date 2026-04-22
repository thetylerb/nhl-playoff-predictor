"""
Download and process MoneyPuck team-level and goalie-level data.

Sources:
  - Historical (2008-2024): peter-tanner.com bulk zips (no Referer needed)
  - Current (2025):         moneypuck.com direct CSVs (Referer header required)

Outputs:
  data/processed/moneypuck_xg.csv
    One row per team per season. Situation == "all".
    team, season, games_played,
    xgf_pg  - scoreVenueAdjustedxGoalsFor / games_played
    xga_pg  - scoreVenueAdjustedxGoalsAgainst / games_played
    gf_pg, ga_pg  - raw goals (kept for comparison)

  data/processed/moneypuck_goalies.csv
    One row per team per season (primary goalie = most games played).
    team, season, goalie_name, goalie_gp, gsax, gsax_pg
    gsax = xGoals - goals (goals saved above expected; positive = better)
    gsax_pg = gsax / goalie_gp
"""

import io
import zipfile
import requests
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REFERER = "https://moneypuck.com"
CURRENT_YEAR = 2025

TEAM_HIST_URL = (
    "https://peter-tanner.com/moneypuck/downloads/"
    "historicalOneRowPerSeason/teams_2008_to_2024.zip"
)
GOALIE_HIST_URL = (
    "https://peter-tanner.com/moneypuck/downloads/"
    "historicalOneRowPerSeason/goalies_2008_to_2024.zip"
)
TEAM_CURR_URL = (
    f"https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    f"{CURRENT_YEAR}/regular/teams.csv"
)
GOALIE_CURR_URL = (
    f"https://moneypuck.com/moneypuck/playerData/seasonSummary/"
    f"{CURRENT_YEAR}/regular/goalies.csv"
)

# MoneyPuck uses legacy multi-char abbreviations; normalize to NHL API standard.
# PHX kept as-is: NHL API also used PHX for the Coyotes pre-Arizona rebranding.
ABBREV_MAP = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}

TEAM_KEEP_COLS = [
    "team", "season", "games_played",
    "scoreVenueAdjustedxGoalsFor",
    "scoreVenueAdjustedxGoalsAgainst",
    "goalsFor",
    "goalsAgainst",
]


# ---------------------------------------------------------------------------
# Team xG
# ---------------------------------------------------------------------------

def _clean_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["situation"] == "all"].copy()
    df = df[TEAM_KEEP_COLS].copy()
    df["team"] = df["team"].replace(ABBREV_MAP)
    df["season"] = df["season"].astype(int)
    df["xgf_pg"] = df["scoreVenueAdjustedxGoalsFor"]   / df["games_played"]
    df["xga_pg"] = df["scoreVenueAdjustedxGoalsAgainst"] / df["games_played"]
    df["gf_pg"]  = df["goalsFor"]    / df["games_played"]
    df["ga_pg"]  = df["goalsAgainst"] / df["games_played"]
    return df[["team", "season", "games_played", "xgf_pg", "xga_pg", "gf_pg", "ga_pg"]]


def fetch_teams_historical() -> pd.DataFrame:
    print("Fetching historical team xG data (2008-2024)...")
    r = requests.get(TEAM_HIST_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"))
    return _clean_teams(df)


def fetch_teams_current() -> pd.DataFrame:
    print(f"Fetching current team xG data (season {CURRENT_YEAR})...")
    r = requests.get(TEAM_CURR_URL, headers={"Referer": REFERER}, timeout=30)
    r.raise_for_status()
    return _clean_teams(pd.read_csv(io.StringIO(r.text)))


# ---------------------------------------------------------------------------
# Goalie GSAX
# ---------------------------------------------------------------------------

def _clean_goalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["situation"] == "all"].copy()
    df["team"] = df["team"].replace(ABBREV_MAP)
    df["season"] = df["season"].astype(int)
    df["gsax"] = df["xGoals"] - df["goals"]   # goals saved above expected
    df["gsax_pg"] = df["gsax"] / df["games_played"]
    # Primary goalie per team = most games played
    primary = (
        df.sort_values("games_played", ascending=False)
        .groupby(["team", "season"])
        .first()
        .reset_index()
    )
    return primary[["team", "season", "name", "games_played", "gsax", "gsax_pg"]].rename(
        columns={"name": "goalie_name", "games_played": "goalie_gp"}
    )


def fetch_goalies_historical() -> pd.DataFrame:
    print("Fetching historical goalie GSAX data (2008-2024)...")
    r = requests.get(GOALIE_HIST_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"))
    return _clean_goalies(df)


def fetch_goalies_current() -> pd.DataFrame:
    print(f"Fetching current goalie GSAX data (season {CURRENT_YEAR})...")
    r = requests.get(GOALIE_CURR_URL, headers={"Referer": REFERER}, timeout=30)
    r.raise_for_status()
    return _clean_goalies(pd.read_csv(io.StringIO(r.text)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Team xG
    teams = pd.concat(
        [fetch_teams_historical(), fetch_teams_current()], ignore_index=True
    ).sort_values(["season", "team"]).reset_index(drop=True)
    teams.to_csv(PROCESSED_DIR / "moneypuck_xg.csv", index=False)
    print(f"Saved {len(teams)} rows -> moneypuck_xg.csv")

    # Goalie GSAX
    goalies = pd.concat(
        [fetch_goalies_historical(), fetch_goalies_current()], ignore_index=True
    ).sort_values(["season", "team"]).reset_index(drop=True)
    goalies.to_csv(PROCESSED_DIR / "moneypuck_goalies.csv", index=False)
    print(f"Saved {len(goalies)} rows -> moneypuck_goalies.csv")

    print("\n2025 goalie GSAX (playoff teams):")
    playoff_teams = {
        "BUF","BOS","TBL","MTL","CAR","OTT","PIT","PHI",
        "COL","LAK","DAL","MIN","VGK","UTA","EDM","ANA",
    }
    sample = goalies[
        (goalies["season"] == CURRENT_YEAR) & (goalies["team"].isin(playoff_teams))
    ].sort_values("gsax", ascending=False)
    print(sample.to_string(index=False))
