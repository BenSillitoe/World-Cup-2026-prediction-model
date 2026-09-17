# World Cup 2026 Prediction Model

An end-to-end forecasting system for the 2026 FIFA World Cup — combining a Python simulation engine (XGBoost, Monte Carlo simulation, and Elo ratings) with a live SwiftUI iOS app that visualises the predictions and pulls real match data as the tournament unfolds.

## Overview

Most match-prediction projects stop at a notebook. This one goes further: the Python model doesn't just produce a static prediction — it feeds a companion iOS app that fetches live results and keeps the simulation grounded in what's actually happening in the tournament, through the group stage and into the knockout rounds.

The project has two connected parts:

- **`/` (root)** — the Python modelling pipeline: data cleaning, feature engineering, Elo rating calculation, model training, and Monte Carlo tournament simulation
- **`/ios-app`** — a SwiftUI app that displays simulation results and fetches live match data via the [football-data.org](https://www.football-data.org/) API

## How it works

1. **Elo ratings** (`elo.py`) — each national team is assigned a dynamically updated Elo rating based on historical match results, giving a continuously-updated measure of relative team strength.
2. **Feature engineering** (`features.py`, `clean_data.py`) — historical match data is cleaned and transformed into features usable by the model (team strength differentials, home/away/neutral venue, stadium/venue conditions, etc.).
3. **Model training** (`train_model.py`) — an XGBoost classifier is trained to predict match outcomes from these features.
4. **Tournament simulation** (`simulate.py`, `knockout_simulate.py`) — the group stage and knockout rounds are simulated using **Monte Carlo methods**: each match is simulated many times based on the model's predicted probabilities, and results are aggregated to produce tournament-wide outcome probabilities (e.g. "Team X has a 14% chance of reaching the final") rather than a single deterministic prediction.
5. **Evaluation** (`evaluate.py`) — model predictions are validated against historical tournament data to check calibration and accuracy.
6. **Export & app integration** (`export_for_app.py`, `WorldCupData.py`) — simulation outputs are exported in a format the iOS app consumes, and the app itself fetches live match results to keep predictions current as real results come in.

## iOS App

Built in SwiftUI, the app displays:
- Simulated tournament outcomes and probabilities
- Stadium and venue information (`stadiums.py` on the Python side supplies venue/condition data used in modelling)
- Live match results, fetched from the football-data.org API

### Running the app locally

The app requires your own API key, which is intentionally **not** included in this repo:

1. Get a free API key from [football-data.org](https://www.football-data.org/)
2. Inside `ios-app/`, duplicate `Secrets.example.swift`, rename the copy to `Secrets.swift`, and add your key:
```swift
   enum Secrets {
       static let footballDataAPIKey = "YOUR_API_KEY_HERE"
   }
```
3. Open the project in Xcode and run.

`Secrets.swift` is git-ignored on purpose — never commit real API keys.

## Project structure
## Tech stack

**Modelling**: Python · pandas · XGBoost · Monte Carlo simulation · Elo rating system
**App**: Swift · SwiftUI · football-data.org API

## Running the Python pipeline

```bash
pip install install #pandas, xgboost, numpy individually
python Main.py
```

## Possible next steps

- Incorporate player-level injury/availability data
- Backtest against previous World Cup tournaments for calibration
- Expand the iOS app with a bracket visualisation of simulated outcomes

