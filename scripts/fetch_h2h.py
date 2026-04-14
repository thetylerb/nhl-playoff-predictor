"""
Fetch regular-season head-to-head records for every playoff matchup pair.

For each (season, team) that appears in playoff_matchups.csv, fetches the
team's full season schedule from the NHL API and caches it to data/raw/.
Then computes H2H wins/losses between each playoff pair and appends those
columns to data/processed/playoff_matchups.csv.

New columns added:
  h2h_gp          - regular season games played between the two teams
  h2h_wins_hi     - wins by the higher-seed team in those games
  h2h_wins_lo     - wins by the lower-seed team
  h2h_wins_diff   - h2h_wins_hi minus h2h_wins_lo (positive = hi team dominated H2H)
"""

import json
import time
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
BASE_URL = "https://api-web.nhle.com/v1"


# ---------------------------------------------------------------------------
# Fetching / caching
# ---------------------------------------------------------------------------

def fetch_team_schedule(team: str, season: str) -> list[dict]:
    """
    Fetch all games for a team in a season from the NHL API.
    Returns list of game dicts. Caches result to data/raw/.
    """
    cache_path = RAW_DIR / f"schedule_{team}_{season}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"{BASE_URL}/club-schedule-season/{team}/{season}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    games = resp.json().get("games", [])

    with open(cache_path, "w") as f:
        json.dump(games, f)

    time.sleep(0.3)
    return games


# ---------------------------------------------------------------------------
# H2H computation
# ---------------------------------------------------------------------------

def h2h_record(games: list[dict], focus_team: str, opponent: str) -> dict:
    """
    Given a team's full game list, return their regular-season H2H record
    against a specific opponent.

    Returns {"gp": int, "wins": int, "losses": int, "otl": int}
    """
    wins = losses = otl = 0

    for g in games:
        if g.get("gameType") != 2:          # regular season only
            continue
        if g.get("gameState") not in ("OFF", "FINAL"):
            continue

        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        home_abbrev = home.get("abbrev", "")
        away_abbrev = away.get("abbrev", "")

        # Must be a game between our two teams
        teams_in_game = {home_abbrev, away_abbrev}
        if focus_team not in teams_in_game or opponent not in teams_in_game:
            continue

        home_score = home.get("score", 0) or 0
        away_score = away.get("score", 0) or 0

        # Determine if this game went to OT/SO (score difference of 1 in period 3+)
        # The API marks this via periodDescriptor or gameOutcome
        went_to_ot = _went_to_ot(g)

        if focus_team == home_abbrev:
            if home_score > away_score:
                wins += 1
            elif went_to_ot:
                otl += 1
            else:
                losses += 1
        else:  # focus_team is away
            if away_score > home_score:
                wins += 1
            elif went_to_ot:
                otl += 1
            else:
                losses += 1

    return {"gp": wins + losses + otl, "wins": wins, "losses": losses, "otl": otl}


def _went_to_ot(game: dict) -> bool:
    """Return True if the losing team got an OTL (game went past regulation)."""
    outcome = game.get("gameOutcome", {})
    if outcome:
        last_period = outcome.get("lastPeriodType", "")
        return last_period in ("OT", "SO")
    # Fallback: check periodDescriptor
    pd_ = game.get("periodDescriptor", {})
    return pd_.get("periodType") in ("OT", "SO")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_h2h_to_matchups() -> None:
    matchups_path = PROCESSED_DIR / "playoff_matchups.csv"
    df = pd.read_csv(matchups_path, dtype={"season": str})

    # Collect all unique (team, season) pairs we need schedules for
    teams_needed = set()
    for _, row in df.iterrows():
        teams_needed.add((row["team_hi"], row["season"]))
        teams_needed.add((row["team_lo"], row["season"]))

    print(f"Fetching schedules for {len(teams_needed)} team-seasons...")
    schedule_cache: dict[tuple, list] = {}
    for i, (team, season) in enumerate(sorted(teams_needed), 1):
        cached = (RAW_DIR / f"schedule_{team}_{season}.json").exists()
        print(f"  [{i}/{len(teams_needed)}] {team} {season}{'  (cached)' if cached else ''}")
        schedule_cache[(team, season)] = fetch_team_schedule(team, season)

    print("\nComputing H2H records for each matchup...")
    h2h_gp, h2h_wins_hi, h2h_wins_lo = [], [], []

    for _, row in df.iterrows():
        hi, lo, season = row["team_hi"], row["team_lo"], row["season"]
        games = schedule_cache.get((hi, season), [])
        rec = h2h_record(games, hi, lo)

        h2h_gp.append(rec["gp"])
        h2h_wins_hi.append(rec["wins"])
        # losses + otl from hi's perspective = wins for lo
        h2h_wins_lo.append(rec["losses"] + rec["otl"])

    df["h2h_gp"]       = h2h_gp
    df["h2h_wins_hi"]  = h2h_wins_hi
    df["h2h_wins_lo"]  = h2h_wins_lo
    df["h2h_wins_diff"] = df["h2h_wins_hi"] - df["h2h_wins_lo"]

    df.to_csv(matchups_path, index=False)
    print(f"\nSaved updated matchups: {matchups_path}")

    # Quick sanity check
    print("\nSample H2H records:")
    print(df[["season", "round", "team_hi", "team_lo", "h2h_gp",
              "h2h_wins_hi", "h2h_wins_lo", "h2h_wins_diff", "winner", "hi_won"]]
          .head(20).to_string(index=False))


if __name__ == "__main__":
    add_h2h_to_matchups()
