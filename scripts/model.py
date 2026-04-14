"""
Train and evaluate a playoff prediction model using team regular-season stats.
Reads processed standings from data/processed/, saves model to models/.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["wins", "losses", "ot_losses", "points", "goal_diff", "points_pct"]


def load_data(season: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"standings_{season}.csv"
    return pd.read_csv(path)


def label_playoff_teams(df: pd.DataFrame, top_n: int = 16) -> pd.DataFrame:
    """Label the top N teams by points as playoff qualifiers (binary target)."""
    df = df.copy()
    threshold = df["points"].nlargest(top_n).min()
    df["made_playoffs"] = (df["points"] >= threshold).astype(int)
    return df


def train(df: pd.DataFrame):
    df = label_playoff_teams(df)
    X = df[FEATURES].fillna(0)
    y = df["made_playoffs"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    clf.fit(X_scaled, y)
    return clf, scaler


if __name__ == "__main__":
    season = "20232024"
    df = load_data(season)
    clf, scaler = train(df)

    model_path = MODELS_DIR / f"playoff_rf_{season}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler}, f)
    print(f"Model saved: {model_path}")
