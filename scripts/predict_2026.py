"""
Predict 2025-26 NHL first-round playoff matchups using historical data.

Trains a logistic regression and random forest on 15 seasons of completed
playoff series (2010-11 through 2024-25) — 225 series total — then applies
both models to each of the 8 first-round matchups of the 2025-26 playoffs.

Features: regular-season stat differentials between the two teams.
Target: did the higher-points team win the series? (hi_won)

Output: printed table of matchups with predicted winner + win probability.
Also saves outputs/predictions_2026.png — a visual summary chart.
"""

import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api-web.nhle.com/v1"

# Features to use for prediction — excludes H2H which was shown to be weakest
FEATURES = [
    "goal_diff_diff",
    "points_diff",
    "points_pct_diff",
    "win_pct_diff",
    "goals_for_pg_diff",
    "goals_against_pg_diff",
]

CURRENT_SEASON = "20252026"
CURRENT_SEASON_DATE = "2026-04-15"
BRACKET_YEAR = 2026


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
            "points_pct": points / (gp * 2),
            "goals_for_pg": gf / gp,
            "goals_against_pg": ga / gp,
            "win_pct": wins / gp,
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
# Build feature rows for current matchups
# ---------------------------------------------------------------------------

def build_matchup_features(matchups: list[dict], standings: pd.DataFrame) -> pd.DataFrame:
    """
    For each first-round matchup, compute the same stat differentials
    used in the training data. Designates hi/lo by regular-season points.
    """
    rows = []
    for m in matchups:
        t1, t2 = m["top_seed"], m["bottom_seed"]
        if t1 not in standings.index or t2 not in standings.index:
            print(f"  WARNING: {t1} or {t2} not found in standings — skipping")
            continue

        r1 = standings.loc[t1]
        r2 = standings.loc[t2]

        # Designate hi/lo by points — consistent with training data labeling
        if r1["points"] >= r2["points"]:
            hi, lo, row_hi, row_lo = t1, t2, r1, r2
            hi_is_top_seed = True
        else:
            hi, lo, row_hi, row_lo = t2, t1, r2, r1
            hi_is_top_seed = False

        row = {
            "series": m["series_letter"],
            "top_seed": t1,
            "top_seed_name": m["top_seed_name"],
            "bottom_seed": t2,
            "bottom_seed_name": m["bottom_seed_name"],
            "top_seed_rank": m["top_seed_rank"],
            "bottom_seed_rank": m["bottom_seed_rank"],
            "team_hi": hi,
            "team_lo": lo,
            "hi_is_top_seed": hi_is_top_seed,
            "points_hi": row_hi["points"],
            "points_lo": row_lo["points"],
            # Differentials (hi minus lo — same convention as training data)
            "goal_diff_diff":       row_hi["goal_diff"]       - row_lo["goal_diff"],
            "points_diff":          row_hi["points"]          - row_lo["points"],
            "points_pct_diff":      row_hi["points_pct"]      - row_lo["points_pct"],
            "win_pct_diff":         row_hi["win_pct"]         - row_lo["win_pct"],
            "goals_for_pg_diff":    row_hi["goals_for_pg"]    - row_lo["goals_for_pg"],
            "goals_against_pg_diff":row_hi["goals_against_pg"]- row_lo["goals_against_pg"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Train models & cross-validate
# ---------------------------------------------------------------------------

def train_and_evaluate(df_train: pd.DataFrame):
    """
    Train logistic regression and random forest on historical matchup data.
    Uses leave-one-season-out CV for honest accuracy estimates.
    Returns fitted models and scaler.
    """
    X = df_train[FEATURES].fillna(0).values
    y = df_train["hi_won"].values
    groups = df_train["season"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logo = LeaveOneGroupOut()

    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)

    lr_scores = cross_val_score(lr, X_scaled, y, cv=logo, groups=groups, scoring="accuracy")
    rf_scores = cross_val_score(rf, X_scaled, y, cv=logo, groups=groups, scoring="accuracy")

    print(f"\nLeave-one-season-out CV accuracy ({len(df_train)} series, 15 seasons):")
    print(f"  Logistic Regression: {lr_scores.mean():.3f} ± {lr_scores.std():.3f}")
    print(f"  Random Forest:       {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # Fit on full dataset for predictions
    lr.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Feature importance from logistic regression coefficients
    print("\nLogistic Regression coefficients (higher = more predictive):")
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coef": lr.coef_[0]
    }).sort_values("coef", ascending=False)
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:30s}  {row['coef']:+.3f}")

    print("\nRandom Forest feature importances:")
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:30s}  {row['importance']:.3f}")

    return lr, rf, scaler


# ---------------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------------

def predict_matchups(matchup_df: pd.DataFrame, lr, rf, scaler) -> pd.DataFrame:
    """
    Apply trained models to 2025-26 first-round matchups.
    Returns DataFrame with predicted winner and win probability from each model.
    """
    X = matchup_df[FEATURES].fillna(0).values
    X_scaled = scaler.transform(X)

    lr_proba = lr.predict_proba(X_scaled)[:, 1]   # P(hi_team wins)
    rf_proba = rf.predict_proba(X_scaled)[:, 1]
    ensemble_proba = (lr_proba + rf_proba) / 2

    results = []
    for i, row in matchup_df.iterrows():
        hi = row["team_hi"]
        lo = row["team_lo"]
        p_hi = ensemble_proba[matchup_df.index.get_loc(i)]
        p_lo = 1 - p_hi

        # Map back to top/bottom seed framing for output
        top = row["top_seed"]
        p_top = p_hi if hi == top else p_lo
        predicted_winner = top if p_top >= 0.5 else row["bottom_seed"]
        predicted_winner_name = row["top_seed_name"] if p_top >= 0.5 else row["bottom_seed_name"]

        results.append({
            "series": row["series"],
            "top_seed_abbrev": top,
            "top_seed_name": row["top_seed_name"],
            "top_seed_rank": row["top_seed_rank"],
            "bottom_seed_abbrev": row["bottom_seed"],
            "bottom_seed_name": row["bottom_seed_name"],
            "bottom_seed_rank": row["bottom_seed_rank"],
            "predicted_winner_abbrev": predicted_winner,
            "predicted_winner_name": predicted_winner_name,
            "p_top_seed_wins": round(p_top, 3),
            "p_bottom_seed_wins": round(1 - p_top, 3),
            "lr_p_hi": round(lr_proba[matchup_df.index.get_loc(i)], 3),
            "rf_p_hi": round(rf_proba[matchup_df.index.get_loc(i)], 3),
            "points_hi": row["points_hi"],
            "points_lo": row["points_lo"],
            "team_hi": hi,
            "team_lo": lo,
            "goal_diff_diff": row["goal_diff_diff"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_predictions(predictions: pd.DataFrame) -> None:
    """Generate a bar chart showing predicted win probability for each series."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Build matchup labels and probabilities
    labels = []
    top_probs = []
    bot_probs = []
    bar_colors_top = []
    bar_colors_bot = []

    WIN_COLOR = "#2e7d32"
    LOSE_COLOR = "#c62828"
    GRAY = "#9e9e9e"

    for _, row in predictions.iterrows():
        label = (
            f"({row['top_seed_rank']}) {row['top_seed_abbrev']}\nvs\n"
            f"({row['bottom_seed_rank']}) {row['bottom_seed_abbrev']}"
        )
        labels.append(label)
        p_top = row["p_top_seed_wins"]
        p_bot = row["p_bottom_seed_wins"]
        top_probs.append(p_top)
        bot_probs.append(p_bot)

        top_wins = row["predicted_winner_abbrev"] == row["top_seed_abbrev"]
        bar_colors_top.append(WIN_COLOR if top_wins else LOSE_COLOR)
        bar_colors_bot.append(LOSE_COLOR if top_wins else WIN_COLOR)

    x = np.arange(len(labels))
    width = 0.38

    bars_top = ax.bar(x - width / 2, top_probs, width, color=bar_colors_top, alpha=0.85)
    bars_bot = ax.bar(x + width / 2, bot_probs, width, color=bar_colors_bot, alpha=0.85)

    # Annotate bars with probability and predicted winner label
    for bar, prob, row in zip(bars_top, top_probs, predictions.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{prob:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, prob in zip(bars_bot, bot_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{prob:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Win Probability (Ensemble Model)")
    ax.set_title(
        "2025–26 NHL First-Round Playoff Predictions\n"
        "(Logistic Regression + Random Forest, trained on 225 series / 15 seasons)"
    )

    win_patch = mpatches.Patch(color=WIN_COLOR, alpha=0.85, label="Predicted winner")
    lose_patch = mpatches.Patch(color=LOSE_COLOR, alpha=0.85, label="Predicted loser")
    ax.legend(handles=[win_patch, lose_patch], loc="upper right")

    plt.tight_layout()
    out_path = OUTPUTS_DIR / "predictions_2026.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nChart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("2025-26 NHL Playoff First-Round Predictions")
    print("=" * 60)

    # 1. Load training data
    df_train = pd.read_csv(PROCESSED_DIR / "playoff_matchups.csv", dtype={"season": str})
    print(f"\nTraining data: {len(df_train)} series across {df_train['season'].nunique()} seasons")

    # 2. Train models
    lr, rf, scaler = train_and_evaluate(df_train)

    # 3. Fetch current season data
    print("\nFetching 2025-26 standings...")
    standings_2026 = fetch_current_standings()

    print("\nFetching 2025-26 playoff bracket...")
    matchups_raw = fetch_first_round_matchups()
    print(f"  {len(matchups_raw)} first-round matchups found")

    # 4. Build feature matrix for 2026 matchups
    matchup_df = build_matchup_features(matchups_raw, standings_2026)

    # 5. Predict
    predictions = predict_matchups(matchup_df, lr, rf, scaler)

    # 6. Print results
    print("\n" + "=" * 60)
    print("PREDICTIONS — 2025-26 First Round")
    print("=" * 60)

    conf_groups = {
        "EASTERN CONFERENCE": ["A", "B", "C", "D"],
        "WESTERN CONFERENCE": ["E", "F", "G", "H"],
    }

    for conf, letters in conf_groups.items():
        print(f"\n  {conf}")
        print(f"  {'Series':<8} {'Matchup':<35} {'Predicted Winner':<22} {'Confidence'}")
        print(f"  {'-'*8} {'-'*35} {'-'*22} {'-'*12}")
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
    print(f"\n  {'Series':<6} {'Top Seed':<24} {'Bottom Seed':<24} {'P(top)':<9} {'P(bot)':<9} {'Predicted'}")
    print(f"  {'-'*6} {'-'*24} {'-'*24} {'-'*9} {'-'*9} {'-'*15}")
    for _, row in predictions.iterrows():
        top_name = f"({row['top_seed_rank']}) {row['top_seed_abbrev']}"
        bot_name = f"({row['bottom_seed_rank']}) {row['bottom_seed_abbrev']}"
        print(
            f"  {row['series']:<6} {top_name:<24} {bot_name:<24} "
            f"{row['p_top_seed_wins']:<9.1%} {row['p_bottom_seed_wins']:<9.1%} "
            f"{row['predicted_winner_abbrev']}"
        )

    # 7. Save predictions CSV
    out_csv = PROCESSED_DIR / "predictions_2026.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"\nPredictions saved: {out_csv}")

    # 8. Plot
    plot_predictions(predictions)
