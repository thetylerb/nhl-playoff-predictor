"""
Predictor Analysis — how well does each regular-season stat predict playoff series wins?

For each stat we calculate:
  - Accuracy     : % of series where the team with the better stat won
  - Correlation  : point-biserial r between stat_diff and series outcome (hi_won)
  - AUC          : logistic regression leave-one-out AUC

Results are ranked and printed as a leaderboard, then saved as a bar chart.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Stats where a LOWER value is actually better (flip the diff sign before measuring)
LOWER_IS_BETTER = {"goals_against_pg"}

STAT_LABELS = {
    "points":          "Points",
    "wins":            "Wins",
    "win_pct":         "Win %",
    "goal_diff":       "Goal Differential",
    "points_pct":      "Points %",
    "goals_for_pg":    "Goals For / Game",
    "goals_against_pg": "Goals Against / Game",
}


def load_matchups() -> pd.DataFrame:
    path = PROCESSED_DIR / "playoff_matchups.csv"
    if not path.exists():
        raise FileNotFoundError(
            "playoff_matchups.csv not found — run build_matchup_dataset.py first."
        )
    return pd.read_csv(path)


def stat_accuracy(df: pd.DataFrame, stat: str) -> float:
    """
    % of series where the team that was 'better' by this stat won.
    For lower-is-better stats, better means lower (so we flip the diff).
    """
    diff_col = f"{stat}_diff"
    if diff_col not in df.columns:
        return float("nan")

    diff = df[diff_col].copy()
    if stat in LOWER_IS_BETTER:
        diff = -diff  # negative diff now means hi team is better

    # Ties (diff == 0) don't count as a correct prediction
    valid = df[diff != 0].copy()
    valid_diff = diff[diff != 0]

    if len(valid) == 0:
        return float("nan")

    # "Better" team won = diff > 0 and hi_won == 1, or diff < 0 and hi_won == 0
    better_team_won = ((valid_diff > 0) & (valid["hi_won"] == 1)) | \
                      ((valid_diff < 0) & (valid["hi_won"] == 0))
    return better_team_won.mean()


def stat_correlation(df: pd.DataFrame, stat: str) -> float:
    """Point-biserial correlation between stat_diff and series outcome."""
    diff_col = f"{stat}_diff"
    if diff_col not in df.columns:
        return float("nan")
    diff = df[diff_col].fillna(0)
    if stat in LOWER_IS_BETTER:
        diff = -diff
    r, _ = scipy_stats.pointbiserialr(df["hi_won"], diff)
    return r


def stat_auc(df: pd.DataFrame, stat: str) -> float:
    """Logistic regression LOO AUC using just the stat differential as feature."""
    diff_col = f"{stat}_diff"
    if diff_col not in df.columns:
        return float("nan")

    sub = df[[diff_col, "hi_won"]].dropna()
    if len(sub) < 10:
        return float("nan")

    X = sub[[diff_col]].values
    y = sub["hi_won"].values

    if stat in LOWER_IS_BETTER:
        X = -X

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    cv_folds = min(5, len(sub))
    scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring="roc_auc")
    return scores.mean()


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    stats = [col.removesuffix("_diff") for col in df.columns if col.endswith("_diff")]

    results = []
    for stat in stats:
        results.append({
            "stat": stat,
            "label": STAT_LABELS.get(stat, stat),
            "accuracy": stat_accuracy(df, stat),
            "correlation": stat_correlation(df, stat),
            "auc": stat_auc(df, stat),
            "n_series": df[f"{stat}_diff"].notna().sum(),
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)


def print_leaderboard(results: pd.DataFrame, df: pd.DataFrame) -> None:
    n_seasons = df["season"].nunique()
    n_series = len(df)

    print(f"\n{'='*65}")
    print(f"  NHL PLAYOFF SERIES PREDICTOR LEADERBOARD")
    print(f"  {n_seasons} seasons | {n_series} total series")
    print(f"{'='*65}")
    print(f"  {'Stat':<22} {'Accuracy':>9} {'Corr (r)':>10} {'AUC':>8}  {'N':>5}")
    print(f"  {'-'*22} {'-'*9} {'-'*10} {'-'*8}  {'-'*5}")

    for _, row in results.iterrows():
        marker = "  <-- BEST" if row.name == 0 else ""
        print(
            f"  {row['label']:<22} {row['accuracy']:>8.1%} "
            f"{row['correlation']:>10.3f} {row['auc']:>8.3f}  "
            f"{int(row['n_series']):>5}{marker}"
        )

    print(f"{'='*65}")

    best = results.iloc[0]
    print(f"\n  Best predictor: {best['label']}")
    print(f"  The team with the better {best['label'].lower()} during the")
    print(f"  regular season won the playoff series {best['accuracy']:.1%} of the time.")
    print(f"  (AUC = {best['auc']:.3f}, r = {best['correlation']:.3f})\n")


def plot_results(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Regular-Season Stat → Playoff Series Win Prediction", fontsize=13, fontweight="bold")

    colors = ["#1a6faf" if i == 0 else "#6baed6" for i in range(len(results))]

    # Left: Accuracy
    ax = axes[0]
    bars = ax.barh(results["label"][::-1], results["accuracy"][::-1], color=colors[::-1])
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Coin flip (50%)")
    ax.set_xlabel("Accuracy (% series where better-stat team won)")
    ax.set_title("Prediction Accuracy by Stat")
    ax.set_xlim(0.3, 0.85)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, results["accuracy"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)

    # Right: AUC
    ax2 = axes[1]
    bars2 = ax2.barh(results["label"][::-1], results["auc"][::-1], color=colors[::-1])
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    ax2.set_xlabel("Logistic Regression AUC (cross-validated)")
    ax2.set_title("Predictive AUC by Stat")
    ax2.set_xlim(0.3, 0.85)
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, results["auc"][::-1]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out_path = OUTPUTS_DIR / "predictor_leaderboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved: {out_path}")


def per_round_accuracy(df: pd.DataFrame, best_stat: str) -> None:
    """Break down prediction accuracy by playoff round for the best stat."""
    diff_col = f"{best_stat}_diff"
    if diff_col not in df.columns:
        return

    diff = df[diff_col].copy()
    if best_stat in LOWER_IS_BETTER:
        diff = -diff

    df = df.copy()
    df["_better_won"] = ((diff > 0) & (df["hi_won"] == 1)) | \
                        ((diff < 0) & (df["hi_won"] == 0))
    df = df[diff != 0]

    print(f"\n  Accuracy by round — {STAT_LABELS.get(best_stat, best_stat)}:")
    for rnd, grp in df.groupby("round"):
        acc = grp["_better_won"].mean()
        print(f"    Round {rnd}: {acc:.1%}  ({len(grp)} series)")


if __name__ == "__main__":
    print("Loading matchup data...")
    df = load_matchups()
    print(f"  {len(df)} series, {df['season'].nunique()} seasons\n")

    print("Running predictor analysis...")
    results = analyze(df)

    print_leaderboard(results, df)

    best_stat = results.iloc[0]["stat"]
    per_round_accuracy(df, best_stat)

    print("\nGenerating chart...")
    plot_results(results)

    # Save results table
    out_csv = PROCESSED_DIR / "predictor_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"  Results table saved: {out_csv}")
