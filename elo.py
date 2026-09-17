import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

results = pd.read_csv(os.path.join(DATA_DIR, 'results_clean.csv'), parse_dates=['date'])
results = results[results['date'] >= '1990-01-01'].copy()
results = results.sort_values('date').reset_index(drop=True)

STARTING_ELO = 1500

def get_k(tournament):
    tournament = str(tournament).lower()
    if 'fifa world cup' in tournament:
        return 48
    elif 'friendly' in tournament:
        return 20
    else:
        return 32

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def get_goal_diff_multiplier(goal_diff):
    """
    Scale K factor by margin of victory.
    A 5-0 win should move ratings more than a 1-0 win.
    Formula used by World Football Elo Ratings.
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    elif goal_diff == 3:
        return 1.75
    else:
        return 1.75 + (goal_diff - 3) / 8

def update_elo(elo_a, elo_b, result, k, goal_diff):
    multiplier = get_goal_diff_multiplier(goal_diff)
    k_adjusted = k * multiplier

    expected_a = expected_score(elo_a, elo_b)
    new_elo_a = elo_a + k_adjusted * (result - expected_a)
    new_elo_b = elo_b + k_adjusted * ((1 - result) - (1 - expected_a))
    return new_elo_a, new_elo_b

# --- Calculate Elo ratings ---
elo_ratings = {}
home_elos = []
away_elos = []

for _, row in results.iterrows():
    home = row['home_team']
    away = row['away_team']

    elo_ratings.setdefault(home, STARTING_ELO)
    elo_ratings.setdefault(away, STARTING_ELO)

    # Record Elo BEFORE the match
    home_elos.append(elo_ratings[home])
    away_elos.append(elo_ratings[away])

    # Skip matches with no scores (future fixtures)
    if pd.isna(row['home_score']) or pd.isna(row['away_score']):
        continue

    if row['home_score'] > row['away_score']:
        result = 1
    elif row['home_score'] < row['away_score']:
        result = 0
    else:
        result = 0.5

    goal_diff = abs(row['home_score'] - row['away_score'])
    k = get_k(row['tournament'])

    elo_ratings[home], elo_ratings[away] = update_elo(
        elo_ratings[home], elo_ratings[away], result, k, goal_diff
    )

# --- Save Elo ratings per match (used by features.py) ---
results['home_elo'] = home_elos
results['away_elo'] = away_elos
results[['date', 'home_team', 'away_team', 'home_elo', 'away_elo']].to_csv(
    os.path.join(DATA_DIR, 'elo_by_match.csv'), index=False
)

# --- Save final Elo ratings per team ---
elo_df = pd.DataFrame([
    {'team': team, 'elo': round(elo, 1)}
    for team, elo in elo_ratings.items()
]).sort_values('elo', ascending=False).reset_index(drop=True)

elo_df['rank'] = elo_df.index + 1
elo_df.to_csv(os.path.join(DATA_DIR, 'elo_ratings.csv'), index=False)

print("=== TOP 20 TEAMS BY ELO (Goal-Diff Adjusted) ===")
print(elo_df.head(20).to_string(index=False))

wc2026_teams = [
    'Mexico', 'South Africa', 'South Korea', 'Czechia',
    'Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland',
    'Brazil', 'Haiti', 'Morocco', 'Scotland',
    'Australia', 'Paraguay', 'Turkey', 'United States',
    'Curaçao', 'Ecuador', 'Germany', 'Ivory Coast',
    'Japan', 'Netherlands', 'Sweden', 'Tunisia',
    'Belgium', 'Egypt', 'Iran', 'New Zealand',
    'Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay',
    'France', 'Iraq', 'Norway', 'Senegal',
    'Algeria', 'Argentina', 'Austria', 'Jordan',
    'Colombia', 'DR Congo', 'Portugal', 'Uzbekistan',
    'Croatia', 'England', 'Ghana', 'Panama'
]

print("\n=== WC2026 TEAMS ELO RATINGS (Goal-Diff Adjusted) ===")
wc_elo = elo_df[elo_df['team'].isin(wc2026_teams)].copy()
print(wc_elo.to_string(index=False))
print(f"\n{len(wc_elo)}/48 WC2026 teams found")