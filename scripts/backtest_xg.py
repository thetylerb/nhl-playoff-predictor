"""
Backtest: three-way comparison on 221 historical playoff series (2010-11 through 2024-25).

  Model A: raw GF/GA as Poisson rates
  Model B: MoneyPuck xG rates (score/venue adjusted)
  Model C: MoneyPuck xG rates + goalie GSAX adjustment

GSAX adjustment: each team's xGA per game is reduced by the primary goalie's
goals-saved-above-expected per game.  A goalie with positive GSAX suppresses the
opponent's effective scoring rate; a negative GSAX goalie inflates it.

    adj_xGA = max(0.5, xGA_pg - goalie_gsax_pg)

The floor of 0.5 prevents degenerate lambda values in extreme cases.
"""

import math
import pandas as pd
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
    lam_hi = math.sqrt(gf_hi * ga_lo)
    lam_lo = math.sqrt(gf_lo * ga_hi)
    p_home = _game_win_prob(lam_hi * HOME_ADV, lam_lo / HOME_ADV)
    p_away = _game_win_prob(lam_hi / HOME_ADV, lam_lo * HOME_ADV)
    return _series_win_prob((4 * p_home + 3 * p_away) / 7)


def run_backtest():
    matchups = pd.read_csv(PROCESSED_DIR / "playoff_matchups.csv", dtype={"season": str})
    xg       = pd.read_csv(PROCESSED_DIR / "moneypuck_xg.csv").set_index(["team", "season"])
    goalies  = pd.read_csv(PROCESSED_DIR / "moneypuck_goalies.csv").set_index(["team", "season"])

    matchups["mp_season"] = matchups["season"].str[:4].astype(int)

    results = []
    skipped = 0

    for _, row in matchups.iterrows():
        hi, lo = row["team_hi"], row["team_lo"]
        yr = row["mp_season"]

        if (hi, yr) not in xg.index or (lo, yr) not in xg.index:
            skipped += 1
            continue

        hi_xg = xg.loc[(hi, yr)]
        lo_xg = xg.loc[(lo, yr)]

        # GSAX per game (0.0 if goalie data missing for that team/season)
        gsax_hi = goalies.loc[(hi, yr), "gsax_pg"] if (hi, yr) in goalies.index else 0.0
        gsax_lo = goalies.loc[(lo, yr), "gsax_pg"] if (lo, yr) in goalies.index else 0.0

        # Goalie-adjusted xGA: floor at 0.5 to keep lambdas sensible
        adj_xga_hi = max(0.5, hi_xg["xga_pg"] - gsax_hi)
        adj_xga_lo = max(0.5, lo_xg["xga_pg"] - gsax_lo)

        results.append({
            "season":      row["season"],
            "hi":          hi,
            "lo":          lo,
            "hi_won":      row["hi_won"],
            # Model A: raw GF/GA
            "p_raw":       predict_series(
                               row["goals_for_pg_hi"], row["goals_against_pg_hi"],
                               row["goals_for_pg_lo"], row["goals_against_pg_lo"],
                           ),
            # Model B: xG only
            "p_xg":        predict_series(
                               hi_xg["xgf_pg"], hi_xg["xga_pg"],
                               lo_xg["xgf_pg"], lo_xg["xga_pg"],
                           ),
            # Model C: xG + goalie GSAX
            "p_xg_gsax":   predict_series(
                               hi_xg["xgf_pg"], adj_xga_hi,
                               lo_xg["xgf_pg"], adj_xga_lo,
                           ),
        })

    df = pd.DataFrame(results)
    n = len(df)

    for col in ("p_raw", "p_xg", "p_xg_gsax"):
        df[f"pred_{col}"] = (df[col] >= 0.5).astype(int)

    def _acc(col):  return (df[f"pred_{col}"] == df["hi_won"]).mean()
    def _brier(col): return ((df[col] - df["hi_won"]) ** 2).mean()

    print(f"Backtest: {n} series, {df['season'].nunique()} seasons  "
          f"({skipped} skipped)\n")
    print(f"{'Model':<28} {'Accuracy':>9} {'Brier':>8}")
    print("-" * 48)
    for label, col in [
        ("A: Raw GF/GA",         "p_raw"),
        ("B: xG only",           "p_xg"),
        ("C: xG + goalie GSAX",  "p_xg_gsax"),
    ]:
        print(f"  {label:<26} {_acc(col):>9.3f} {_brier(col):>8.4f}")

    print("\nPer-season accuracy:")
    print(f"  {'Season':<12} {'n':>4}  {'Raw':>7}  {'xG':>7}  {'xG+GSAX':>9}")
    print(f"  {'-'*12} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*9}")
    for season, grp in df.groupby("season"):
        a_raw  = (grp["pred_p_raw"]      == grp["hi_won"]).mean()
        a_xg   = (grp["pred_p_xg"]       == grp["hi_won"]).mean()
        a_gsax = (grp["pred_p_xg_gsax"]  == grp["hi_won"]).mean()
        print(f"  {season:<12} {len(grp):>4}  {a_raw:>7.1%}  {a_xg:>7.1%}  {a_gsax:>9.1%}")

    print("\nConfidence buckets (Model C: xG + GSAX):")
    df["conf_c"] = df["p_xg_gsax"].apply(lambda p: max(p, 1 - p))
    for lo_b, hi_b in [(0.5, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]:
        bucket = df[(df["conf_c"] >= lo_b) & (df["conf_c"] < hi_b)]
        if bucket.empty:
            continue
        a = (bucket["pred_p_xg_gsax"] == bucket["hi_won"]).mean()
        label = f"{lo_b:.0%}-{min(hi_b, 1.0):.0%}"
        print(f"  {label} confidence: {len(bucket):>3} series, {a:.1%} correct")

    return df


if __name__ == "__main__":
    run_backtest()
