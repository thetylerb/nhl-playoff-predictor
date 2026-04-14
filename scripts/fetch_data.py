"""
Fetch NHL stats data from the NHL Stats API.
Data is saved as raw JSON in data/raw/.

Standings endpoint requires a specific date (use the last day of the regular season).
Playoff bracket endpoint uses just the 4-digit end year (e.g. 2024 for 2023-24 season).
"""

import requests
import json
import time
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api-web.nhle.com/v1"

# Maps season ID (e.g. "20232024") -> last day of regular season
# Used to pull final standings before playoffs begin
SEASON_END_DATES = {
    "20172018": "2018-04-08",
    "20182019": "2019-04-06",
    "20192020": "2020-03-11",  # Season paused/restarted as bubble; standings reflect pre-pause
    "20202021": "2021-05-19",
    "20212022": "2022-04-29",
    "20222023": "2023-04-14",
    "20232024": "2024-04-18",
}


def get_standings(date: str) -> dict:
    """Fetch standings as of a specific date (YYYY-MM-DD)."""
    url = f"{BASE_URL}/standings/{date}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def get_playoff_bracket(year: int) -> dict:
    """
    Fetch playoff bracket for a given end-year (e.g. 2024 for the 2023-24 season).
    Returns round/series data including wins per team.
    """
    url = f"{BASE_URL}/playoff-bracket/{year}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.json()


def save_json(data: dict, filename: str) -> None:
    path = RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def fetch_season(season: str) -> None:
    end_date = SEASON_END_DATES[season]
    end_year = int(season[4:])  # e.g. 20232024 -> 2024

    print(f"\n--- Season {season} ---")

    print(f"  Fetching standings ({end_date})...")
    try:
        standings = get_standings(end_date)
        save_json(standings, f"standings_{season}.json")
    except Exception as e:
        print(f"  ERROR fetching standings: {e}")

    print(f"  Fetching playoff bracket ({end_year})...")
    try:
        bracket = get_playoff_bracket(end_year)
        save_json(bracket, f"playoff_bracket_{season}.json")
    except Exception as e:
        print(f"  ERROR fetching bracket: {e}")

    time.sleep(0.5)  # be polite to the API


if __name__ == "__main__":
    seasons = list(SEASON_END_DATES.keys())
    print(f"Fetching data for {len(seasons)} seasons: {seasons}")
    for season in seasons:
        fetch_season(season)
    print("\nDone.")
