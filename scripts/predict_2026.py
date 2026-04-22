"""
Predict 2025-26 NHL first-round playoff matchups using a Poisson/Skellam model.

--- Model overview ---

Goals For and Goals Against per game are modelled as independent Poisson
processes.  To account for opponent quality, each team's effective scoring rate
is computed as the geometric mean of the team's own GF/G and the opponent's GA/G:

    λ_A = sqrt(team_A GF/G  *  opponent GA/G)
    λ_B = sqrt(team_B GF/G  *  team_A  GA/G)

This blends each team's offensive output with the defence it actually faces,
avoiding the inflation that arises from using raw GF/G alone (which ignores
defensive quality entirely).

Single-game win probability uses the Skellam distribution (the exact PMF of the
difference of two independent Poisson variables), with overtime handled via the
goal-scoring rate proportion:

    P(A wins game) = P(Skellam(λ_A, λ_B) > 0)
                   + P(Skellam = 0) * λ_A / (λ_A + λ_B)

--- Parameter choices ---

HOME_ADV = 1.10  (±10% attack-rate multiplier)
  Calibrated against both raw-GF and geometric-mean variants.  The 5% default
  produced near-uniform spreads (most series 54–59%); 10% better reflects the
  historical home-ice effect in playoff hockey (~54–57% single-game win rate for
  the home team, per NHL data) while keeping series probabilities credible.
  Raw GF/G without opponent adjustment was rejected because it produced a
  69.5% series win probability for PIT over PHI — an artifact of ignoring
  Pittsburgh's significantly worse defensive rate (3.27 GA/G vs 2.96 for PHI).

Series probability uses closed-form best-of-7 math.  Top seed has home ice for
games 1, 2, 5, 7 (4 home / 3 away).  A weighted average p is formed and fed into:

    P(A wins series) = Σ_{g=4}^{7} C(g-1, 3) · p^4 · (1-p)^(g-4)

Data source: NHL Stats API (api-web.nhle.com/v1), 2025-26 final regular-season
standings as of 2026-04-15.

Output: printed table + outputs/predictions_2026.png
"""

import math
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skellam as skellam_dist

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api-web.nhle.com/v1"

CURRENT_SEASON_DATE = "2026-04-15"
BRACKET_YEAR = 2026

HOME_ADV = 1.10  # ±10% attack-rate multiplier; see module docstring for calibration rationale


# ---------------------------------------------------------------------------
# Fetch 2025-26 standings
# ---------------------------------------------------------------------------

def fetch_current_standings() -> pd.DataFrame:
    """Fetch and parse the 2025-26 final regular-season standings."""
    r = requests.get(f"{BASE_URL}/standings/{CURRENT_SEASON_DATE}", timeout=15)
    r.raise_for_status()
    data = r.json()

    records = []
    for team in data.get("standings", []):
        gp = team.get("gamesPlayed") or 1
        gf = team.get("goalFor") or 0
        ga = team.get("goalAgainst") or 0
        wins = team.get("wins") or 0
        losses = team.get("losses") or 0
        ot_losses = team.get("otLosses") or 0
        points = team.get("points") or 0

        records.append({
            "team": team.get("teamAbbrev", {}).get("default"),
            "games_played": gp,
            "wins": wins,
            "losses": losses,
            "ot_losses": ot_losses,
            "points": points,
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gf - ga,
            "goals_for_pg": gf / gp,
            "goals_against_pg": ga / gp,
        })

    return pd.DataFrame(records).set_index("team")


# ---------------------------------------------------------------------------
# Fetch 2025-26 first-round matchups from bracket
# ---------------------------------------------------------------------------

def fetch_first_round_matchups() -> list[dict]:
    """Return list of first-round series from the 2025-26 bracket."""
    r = requests.get(f"{BASE_URL}/playoff-bracket/{BRACKET_YEAR}", timeout=15)
    r.raise_for_status()
    data = r.json()

    matchups = []
    for s in data.get("series", []):
        if s.get("playoffRound") != 1:
            continue
        top = s.get("topSeedTeam", {})
        bot = s.get("bottomSeedTeam", {})
        matchups.append({
            "series_letter": s.get("seriesLetter", ""),
            "top_seed": top.get("abbrev"),
            "top_seed_name": top.get("name", {}).get("default", ""),
            "bottom_seed": bot.get("abbrev"),
            "bottom_seed_name": bot.get("name", {}).get("default", ""),
            "top_seed_rank": s.get("topSeedRankAbbrev", ""),
            "bottom_seed_rank": s.get("bottomSeedRankAbbrev", ""),
        })

    return sorted(matchups, key=lambda x: x["series_letter"])


# ---------------------------------------------------------------------------
# Poisson/Skellam model
# ---------------------------------------------------------------------------

def _game_win_prob(lam_a: float, lam_b: float) -> float:
    """
    P(team A wins a single game) given Poisson rates lam_a (A) and lam_b (B).
    Overtime resolved by scoring-rate proportion.
    """
    p_reg_win = float(sum(skellam_dist.pmf(k, lam_a, lam_b) for k in range(1, 25)))
    p_tie = float(skellam_dist.pmf(0, lam_a, lam_b))
    p_ot_win = lam_a / (lam_a + lam_b) if (lam_a + lam_b) > 0 else 0.5
    return p_reg_win + p_tie * p_ot_win


def _series_win_prob(p: float) -> float:
    """
    P(team wins best-of-7) given constant single-game win probability p.
    Closed-form: Σ_{g=4}^{7} C(g-1, 3) · p^4 · (1-p)^(g-4)
    """
    return sum(
        math.comb(g - 1, 3) * (p ** 4) * ((1 - p) ** (g - 4))
        for g in range(4, 8)
    )


def compute_series_prob(top_gf: float, top_ga: float,
                        bot_gf: float, bot_ga: float) -> float:
    """
    P(top seed wins the series).

    Rates are blended with geometric mean (opponent-adjusted), then
    home-ice advantage is applied per game. Top seed has home ice for
    games 1, 2, 5, 7 (4 of 7).
    """
    # Geometric-mean blending: team attack vs opponent defense
    lam_top = math.sqrt(top_gf * bot_ga)
    lam_bot = math.sqrt(bot_gf * top_ga)

    # Home games for top seed
    p_home = _game_win_prob(lam_top * HOME_ADV, lam_bot / HOME_ADV)
    # Away games for top seed
    p_away = _game_win_prob(lam_top / HOME_ADV, lam_bot * HOME_ADV)

    # Weighted average p across the series (4 home, 3 away games for top seed)
    p_avg = (4 * p_home + 3 * p_away) / 7

    return _series_win_prob(p_avg)


def predict_matchups(matchups_raw: list[dict], standings: pd.DataFrame) -> pd.DataFrame:
    """Compute Poisson/Skellam win probabilities for each first-round series."""
    rows = []
    for m in matchups_raw:
        top, bot = m["top_seed"], m["bottom_seed"]
        if top not in standings.index or bot not in standings.index:
            print(f"  WARNING: {top} or {bot} not found in standings — skipping")
            continue

        r_top = standings.loc[top]
        r_bot = standings.loc[bot]

        p_top = compute_series_prob(
            r_top["goals_for_pg"], r_top["goals_against_pg"],
            r_bot["goals_for_pg"], r_bot["goals_against_pg"],
        )
        p_bot = 1.0 - p_top

        predicted_winner = top if p_top >= 0.5 else bot
        predicted_winner_name = (
            m["top_seed_name"] if p_top >= 0.5 else m["bottom_seed_name"]
        )

        rows.append({
            "series": m["series_letter"],
            "top_seed_abbrev": top,
            "top_seed_name": m["top_seed_name"],
            "top_seed_rank": m["top_seed_rank"],
            "bottom_seed_abbrev": bot,
            "bottom_seed_name": m["bottom_seed_name"],
            "bottom_seed_rank": m["bottom_seed_rank"],
            "predicted_winner_abbrev": predicted_winner,
            "predicted_winner_name": predicted_winner_name,
            "p_top_seed_wins": round(p_top, 3),
            "p_bottom_seed_wins": round(p_bot, 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Team colors
# ---------------------------------------------------------------------------

TEAM_COLORS = {
    # Series A
    "BUF": "#002654",   # Sabres navy
    "BOS": "#FFB81C",   # Bruins gold
    # Series B
    "TBL": "#002868",   # Lightning blue
    "MTL": "#AF1E2D",   # Canadiens red
    # Series C — both teams are red; use OTT gold as secondary to contrast
    "CAR": "#CC0000",   # Hurricanes red
    "OTT": "#C69214",   # Senators gold
    # Series D
    "PIT": "#000000",   # Penguins black
    "PHI": "#F74902",   # Flyers orange
    # Series E
    "COL": "#6F263D",   # Avalanche burgundy
    "LAK": "#A2AAAD",   # Kings silver
    # Series F — both teams are green; use MIN red as secondary to contrast
    "DAL": "#006847",   # Stars green
    "MIN": "#AF1E2D",   # Wild red
    # Series G
    "VGK": "#B4975A",   # Golden Knights gold
    "UTA": "#6CACE4",   # Utah blue
    # Series H — both teams are orange; use ANA black as secondary to contrast
    "EDM": "#FF4C00",   # Oilers orange
    "ANA": "#111111",   # Ducks black
}


def _text_color(hex_color: str) -> str:
    """Return black or white depending on the luminance of the background."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.45 else "white"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions(predictions: pd.DataFrame) -> None:
    """
    Stacked horizontal probability chart with team colors.
    Each row = one matchup. Left portion = top seed win probability (team color),
    right portion = bottom seed win probability (team color).
    """
    BG = "#F7F7F7"

    y_pos = {"A": 8.5, "B": 7.5, "C": 6.5, "D": 5.5,
              "E": 4.0, "F": 3.0, "G": 2.0, "H": 1.0}

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for _, row in predictions.iterrows():
        s      = row["series"]
        y      = y_pos[s]
        top    = row["top_seed_abbrev"]
        bot    = row["bottom_seed_abbrev"]
        r_top  = row["top_seed_rank"]
        r_bot  = row["bottom_seed_rank"]
        p_top  = row["p_top_seed_wins"]
        p_bot  = row["p_bottom_seed_wins"]
        c_top  = TEAM_COLORS.get(top, "#888888")
        c_bot  = TEAM_COLORS.get(bot, "#888888")

        # Subtle full-width track
        ax.barh(y, 1.0, height=0.68, color="#E0E0E0", zorder=1)

        # Team probability bars
        ax.barh(y, p_top,        height=0.68, color=c_top, zorder=2)
        ax.barh(y, p_bot, left=p_top, height=0.68, color=c_bot, zorder=2)

        # Thin white divider at the join point
        ax.plot([p_top, p_top], [y - 0.38, y + 0.38], color="white",
                linewidth=2, zorder=3)

        top_label = f"({r_top}) {top}  {p_top:.0%}"
        bot_label = f"{p_bot:.0%}  {bot} ({r_bot})"

        if p_top >= 0.18:
            ax.text(p_top / 2, y, top_label,
                    ha="center", va="center", fontsize=10.5,
                    color=_text_color(c_top), fontweight="bold", zorder=4)
        else:
            ax.text(p_top - 0.01, y, top_label,
                    ha="right", va="center", fontsize=9,
                    color="#444444", zorder=4)

        if p_bot >= 0.18:
            ax.text(p_top + p_bot / 2, y, bot_label,
                    ha="center", va="center", fontsize=10.5,
                    color=_text_color(c_bot), fontweight="bold", zorder=4)
        else:
            ax.text(p_top + p_bot + 0.01, y, bot_label,
                    ha="left", va="center", fontsize=9,
                    color="#444444", zorder=4)

    # 50% reference line
    ax.axvline(0.5, color="#999999", linewidth=1, linestyle="--", zorder=1)

    # Conference section labels
    for label, y_label, y_line in [
        ("EASTERN CONFERENCE", 9.15, 5.05),
        ("WESTERN CONFERENCE",  4.65, 0.55),
    ]:
        ax.text(0.5, y_label, label, ha="center", va="bottom", fontsize=10,
                color="#555555", fontweight="bold", fontstyle="italic",
                transform=ax.get_yaxis_transform())
        ax.axhline(y_line, color="#CCCCCC", linewidth=0.8, zorder=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, 9.45)
    ax.set_xticks([0.25, 0.50, 0.75])
    ax.set_xticklabels(["25%", "50%", "75%"], color="#888888", fontsize=9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title("2025–26 NHL First Round Predictions", fontsize=15,
                 fontweight="bold", color="#222222", pad=14)
    fig.text(0.5, 0.01,
             "Poisson/Skellam Model · geometric-mean opponent adjustment · 10% home-ice rate multiplier · live 2025-26 standings",
             ha="center", fontsize=8, color="#AAAAAA")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUTPUTS_DIR / "predictions_2026.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nChart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("2025-26 NHL Playoff First-Round Predictions")
    print("Model: Poisson/Skellam (opponent-adjusted rates + home ice)")
    print("=" * 60)

    # 1. Fetch current season data
    print("\nFetching 2025-26 standings...")
    standings_2026 = fetch_current_standings()

    print("Fetching 2025-26 playoff bracket...")
    matchups_raw = fetch_first_round_matchups()
    print(f"  {len(matchups_raw)} first-round matchups found")

    # 2. Compute Poisson/Skellam predictions
    predictions = predict_matchups(matchups_raw, standings_2026)

    # 3. Print results
    print("\n" + "=" * 60)
    print("PREDICTIONS — 2025-26 First Round")
    print("=" * 60)

    conf_groups = {
        "EASTERN CONFERENCE": ["A", "B", "C", "D"],
        "WESTERN CONFERENCE": ["E", "F", "G", "H"],
    }

    for conf, letters in conf_groups.items():
        print(f"\n  {conf}")
        print(f"  {'Series':<8} {'Matchup':<35} {'Predicted Winner':<22} {'Win Prob'}")
        print(f"  {'-'*8} {'-'*35} {'-'*22} {'-'*10}")
        for letter in letters:
            row = predictions[predictions["series"] == letter]
            if row.empty:
                continue
            row = row.iloc[0]
            matchup = (
                f"({row['top_seed_rank']}) {row['top_seed_abbrev']} "
                f"vs ({row['bottom_seed_rank']}) {row['bottom_seed_abbrev']}"
            )
            winner = row["predicted_winner_abbrev"]
            p_win = max(row["p_top_seed_wins"], row["p_bottom_seed_wins"])
            print(f"  {letter:<8} {matchup:<35} {winner:<22} {p_win:.1%}")

    print("\nFull probability breakdown:")
    print(f"\n  {'Ser':<5} {'Top Seed':<24} {'Bottom Seed':<24} {'P(top)':<9} {'P(bot)'}")
    print(f"  {'-'*5} {'-'*24} {'-'*24} {'-'*9} {'-'*9}")
    for _, row in predictions.iterrows():
        top_name = f"({row['top_seed_rank']}) {row['top_seed_abbrev']}"
        bot_name = f"({row['bottom_seed_rank']}) {row['bottom_seed_abbrev']}"
        print(
            f"  {row['series']:<5} {top_name:<24} {bot_name:<24} "
            f"{row['p_top_seed_wins']:<9.1%} {row['p_bottom_seed_wins']:.1%}"
        )

    # 4. Save predictions CSV
    out_csv = PROCESSED_DIR / "predictions_2026.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"\nPredictions saved: {out_csv}")

    # 5. Plot
    plot_predictions(predictions)
