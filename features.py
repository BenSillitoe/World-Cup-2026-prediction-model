import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

results = pd.read_csv(os.path.join(DATA_DIR, 'results_clean.csv'), parse_dates=['date'])
results = results[results['date'] >= '1990-01-01'].copy()
results = results.sort_values('date').reset_index(drop=True)

# --- Load Elo by match (calculated in elo.py) ---
elo_by_match = pd.read_csv(os.path.join(DATA_DIR, 'elo_by_match.csv'), parse_dates=['date'])

results = results.merge(elo_by_match, on=['date', 'home_team', 'away_team'], how='left')
results['elo_diff'] = results['home_elo'] - results['away_elo']

# --- Home advantage ---
# 1 if home team is actually playing at home, 0 for neutral venues
# Also give host nation advantage for USA, Canada, Mexico at WC2026
host_nations = ['United States', 'Canada', 'Mexico']

def get_home_advantage(row):
    if not row['neutral']:
        return 1
    # Host nation advantage at WC2026
    if row['home_team'] in host_nations and row['tournament'] == 'FIFA World Cup':
        return 1
    return 0

results['home_advantage'] = results.apply(get_home_advantage, axis=1)

# --- Rolling form (last 10 matches) ---
def get_team_results(df):
    home = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    home.columns = ['date', 'team', 'scored', 'conceded']
    home['points'] = home.apply(
        lambda r: 3 if r['scored'] > r['conceded']
        else 1 if r['scored'] == r['conceded'] else 0, axis=1
    )

    away = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    away.columns = ['date', 'team', 'scored', 'conceded']
    away['points'] = away.apply(
        lambda r: 3 if r['scored'] > r['conceded']
        else 1 if r['scored'] == r['conceded'] else 0, axis=1
    )

    all_results = pd.concat([home, away]).sort_values('date').reset_index(drop=True)
    return all_results

team_results = get_team_results(results)

team_results['form_points'] = team_results.groupby('team')['points'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)
team_results['avg_scored'] = team_results.groupby('team')['scored'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)
team_results['avg_conceded'] = team_results.groupby('team')['conceded'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean()
)

home_form = team_results[['date', 'team', 'form_points', 'avg_scored', 'avg_conceded']].copy()
home_form.columns = ['date', 'home_team', 'home_form', 'home_avg_scored', 'home_avg_conceded']
results = results.merge(home_form.drop_duplicates(['date', 'home_team']),
                        on=['date', 'home_team'], how='left')

away_form = team_results[['date', 'team', 'form_points', 'avg_scored', 'avg_conceded']].copy()
away_form.columns = ['date', 'away_team', 'away_form', 'away_avg_scored', 'away_avg_conceded']
results = results.merge(away_form.drop_duplicates(['date', 'away_team']),
                        on=['date', 'away_team'], how='left')

# --- Derived features ---
results['form_diff']         = results['home_form'] - results['away_form']
results['avg_scored_diff']   = results['home_avg_scored'] - results['away_avg_scored']
results['avg_conceded_diff'] = results['home_avg_conceded'] - results['away_avg_conceded']
results['is_competitive']    = (results['tournament'] != 'Friendly').astype(int)
results['is_world_cup']      = results['tournament'].str.contains(
                                    'FIFA World Cup', na=False).astype(int)

# --- Target variable ---
def get_outcome(row):
    if pd.isna(row['home_score']) or pd.isna(row['away_score']):
        return np.nan
    if row['home_score'] > row['away_score']:
        return 2
    elif row['home_score'] == row['away_score']:
        return 1
    else:
        return 0

results['outcome'] = results.apply(get_outcome, axis=1)

# --- Drop rows with NaN features ---
feature_cols = [
    'elo_diff', 'home_elo', 'away_elo',
    'form_diff', 'home_form', 'away_form',
    'avg_scored_diff', 'home_avg_scored', 'away_avg_scored',
    'avg_conceded_diff', 'home_avg_conceded', 'away_avg_conceded',
    'is_competitive', 'is_world_cup', 'home_advantage'
]

results_clean = results.dropna(subset=feature_cols).copy()

results_clean.to_csv(os.path.join(DATA_DIR, 'features.csv'), index=False)

print(f"✅ Features built successfully")
print(f"   Total matches with full features: {len(results_clean):,}")
print(f"   World Cup matches in dataset:     {results_clean['is_world_cup'].sum()}")
print(f"   Date range: {results_clean['date'].min().date()} to {results_clean['date'].max().date()}")
print(f"\n--- Feature sample (last 5 matches) ---")
print(results_clean[['date', 'home_team', 'away_team', 'elo_diff',
                      'home_form', 'away_form', 'home_advantage', 'outcome']].tail(5).to_string(index=False))