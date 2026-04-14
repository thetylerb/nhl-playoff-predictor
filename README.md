# NHL Playoff Predictor

How well does a team's regular-season record predict whether they win a playoff series — and which stat is the best predictor?

This project pulls data from the NHL Stats API across 7 seasons (2017–18 through 2023–24), builds a dataset of 105 playoff series matchups, and ranks regular-season stats by their ability to predict which team wins.

---

## Key Finding

![Predictor Leaderboard](outputs/predictor_leaderboard.png)

**Goal differential is the strongest single predictor**, with the team that had the better goal differential during the regular season winning the playoff series **61.5%** of the time. All of the top stats (points, wins, win %, goal differential) cluster tightly between 60–62% — they're largely capturing the same thing: overall team quality.

The most interesting result is **head-to-head regular season record**, which ranks last with a *negative* correlation (`r = -0.056`). Teams that dominated their eventual playoff opponent during the regular season win the series slightly *less* often — suggesting that familiarity and film study may help the team that lost those matchups prepare better for the playoffs.

No stat clears 65% accuracy, which reflects a genuine truth about the NHL playoffs: regular season performance is a modest signal, not a reliable forecast.

---

## Results

| Stat | Accuracy | r | AUC |
|---|---|---|---|
| Goal Differential | 61.5% | 0.123 | 0.590 |
| Wins | 61.2% | 0.124 | 0.644 |
| Win % | 61.2% | 0.096 | 0.631 |
| Points | 60.8% | 0.108 | 0.630 |
| Points % | 60.8% | 0.088 | 0.610 |
| Goals Against / Game | 56.7% | 0.099 | 0.467 |
| H2H Wins | 55.0% | -0.056 | 0.520 |
| Goals For / Game | 54.3% | 0.013 | 0.425 |

---

## How to Run

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt

# 2. Fetch raw data from the NHL API
python scripts/fetch_data.py

# 3. Clean and process standings
python scripts/preprocess.py

# 4. Build per-series matchup dataset
python scripts/build_matchup_dataset.py

# 5. Fetch head-to-head regular season records
python scripts/fetch_h2h.py

# 6. Run predictor analysis and generate chart
python scripts/predictor_analysis.py

# 7. Explore interactively
jupyter notebook
```

---

## Project Structure

```
nhl_data/
├── data/
│   ├── raw/               # Raw JSON from NHL API (gitignored)
│   └── processed/         # Cleaned CSVs
├── notebooks/
│   ├── 01_exploration.ipynb       # Standings EDA
│   └── 02_predictor_analysis.ipynb  # Full predictor analysis
├── outputs/
│   └── predictor_leaderboard.png  # Results chart
├── scripts/
│   ├── fetch_data.py         # Fetch standings + playoff brackets
│   ├── preprocess.py         # Parse standings into CSV
│   ├── build_matchup_dataset.py  # Join series matchups with stats
│   ├── fetch_h2h.py          # Fetch H2H regular season records
│   └── predictor_analysis.py # Rank stats, generate chart
└── requirements.txt
```

---

## Data Source

All data is fetched from the public [NHL Stats API](https://api-web.nhle.com/v1). No API key required.
