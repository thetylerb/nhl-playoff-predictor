"""
Backtest: xG-based Poisson/Skellam model vs raw GF/GA on 225 historical playoff series.

Methodology: same geometric-mean Skellam model used in predict_2026.py.
The only variable is whether Poisson rates come from scoreVenueAdjustedxGoals
(MoneyPuck) or raw goals (NHL API standings).

Output: accuracy table by season and overall, printed to stdout.
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skellam as skellam_dist

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

HOME_ADV = 1.10


def _game_win_prob(lam_a: float, lam_b: float) -> float:
    p_win = float(sum(skellam_dist.pmf(k, lam_a, lam_b) for k in range(1, 25)))
    p_tie = float(skellam_dist.pmf(0, lam_a, lam_b))
    return p_win + p_tie * lam_a / (lam_a + lam_b)


def _series_win_prob(p: float) -> float:
    return sum(
        math.comb(g - 1, 3) * (p ** 4) * ((1 - p) ** (g - 4))
        for g in range(4, 8)
    )


def predict_series(gf_hi: float, ga_hi: float,
                   gf_lo: float, ga_lo: float) -> float:
    """P(hi-seed wins series) via geometric-mean Skellam model."""
    lam_hi = math.sqrt(gf_hi * ga_lo)
    lam_lo = math.sqrt(gf_lo * ga_hi)
    p_home = _game_win_prob(lam_hi * HOME_ADV, lam_lo / HOME_ADV)
    p_away = _game_win_prob(lam_hi / HOME_ADV, lam_lo * HOME_ADV)
    p_avg = (4 * p_home + 3 * p_away) / 7
    return _series_win_prob(p_avg)


def run_backtest():
    matchups = pd.read_csv(PROCESSED_DIR / "playoff_matchups.csv", dtype={"season": str})
    xg = pd.read_csv(PROCESSED_DIR / "moneypuck_xg.csv")

    # season key: "20102011" → 2010
    matchups["mp_season"] = matchups["season"].str[:4].astype(int)

    # Build xG lookup: (team, season) → (xgf_pg, xga_pg, gf_pg, ga_pg)
    xg_lookup = xg.set_index(["team", "season"])

    results = []
    skipped = 0

    for _, row in matchups.iterrows():
        hi, lo = row["team_hi"], row["team_lo"]
        yr = row["mp_season"]

        # Skip if either team missing from xG data (e.g. relocated franchises)
        if (hi, yr) not in xg_lookup.index or (lo, yr) not in xg_lookup.index:
            skipped += 1
            continue

        hi_xg = xg_lookup.loc[(hi, yr)]
        lo_xg = xg_lookup.loc[(lo, yr)]

        # xG model
        p_xg = predict_series(
            hi_xg["xgf_pg"], hi_xg["xga_pg"],
            lo_xg["xgf_pg"], lo_xg["xga_pg"],
        )

        # Raw GF/GA model (from pre-computed matchup columns)
        p_raw = predict_series(
            row["goals_for_pg_hi"],  row["goals_against_pg_hi"],
            row["goals_for_pg_lo"],  row["goals_against_pg_lo"],
        )

        results.append({
            "season":    row["season"],
            "hi":        hi,
            "lo":        lo,
            "hi_won":    row["hi_won"],
            "p_xg":      p_xg,
            "p_raw":     p_raw,
            "pred_xg":   int(p_xg >= 0.5),
            "pred_raw":  int(p_raw >= 0.5),
        })

    df = pd.DataFrame(results)
    n = len(df)

    # ---- Overall accuracy ----
    acc_xg  = (df["pred_xg"]  == df["hi_won"]).mean()
    acc_raw = (df["pred_raw"] == df["hi_won"]).mean()

    # Brier scores (lower = better calibration)
    brier_xg  = ((df["p_xg"]  - df["hi_won"]) ** 2).mean()
    brier_raw = ((df["p_raw"] - df["hi_won"]) ** 2).mean()

    print(f"Backtest: {n} series across {df['season'].nunique()} seasons  "
          f"({skipped} skipped — team not in xG data)\n")
    print(f"{'Metric':<28} {'xG model':>10} {'Raw GF/GA':>10}  {'diff':>8}")
    print("-" * 62)
    print(f"{'Accuracy':<28} {acc_xg:>10.3f} {acc_raw:>10.3f}  {acc_xg - acc_raw:>+8.3f}")
    print(f"{'Brier score (lower=better)':<28} {brier_xg:>10.4f} {brier_raw:>10.4f}  {brier_xg - brier_raw:>+8.4f}")

    # ---- Per-season breakdown ----
    print("\nPer-season accuracy:")
    print(f"  {'Season':<12} {'n':>4}  {'xG':>7}  {'Raw':>7}  {'diff':>7}")
    print(f"  {'-'*12} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}")
    for season, grp in df.groupby("season"):
        a_xg  = (grp["pred_xg"]  == grp["hi_won"]).mean()
        a_raw = (grp["pred_raw"] == grp["hi_won"]).mean()
        print(f"  {season:<12} {len(grp):>4}  {a_xg:>7.1%}  {a_raw:>7.1%}  {a_xg - a_raw:>+7.1%}")

    # ---- Confidence distribution ----
    print("\nPredictions by confidence bucket (xG model):")
    df["conf"] = df["p_xg"].apply(lambda p: max(p, 1 - p))
    for lo_b, hi_b in [(0.5, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]:
        bucket = df[(df["conf"] >= lo_b) & (df["conf"] < hi_b)]
        if bucket.empty:
            continue
        a = (bucket["pred_xg"] == bucket["hi_won"]).mean()
        print(f"  {lo_b:.0%}–{min(hi_b, 1.0):.0%} confidence: "
              f"{len(bucket):>3} series, {a:.1%} correct")

    return df


if __name__ == "__main__":
    run_backtest()
