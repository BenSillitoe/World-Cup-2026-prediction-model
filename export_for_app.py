import pandas as pd
import numpy as np
import os
import json
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

model        = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
features     = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

wc2026 = features[
    (features['date'] >= '2026-06-11') &
    (features['is_world_cup'] == 1)
].copy()

probs = model.predict_proba(wc2026[feature_cols])

matches = []
for i, (_, row) in enumerate(wc2026.iterrows()):
    matches.append({
        'home_team':  row['home_team'],
        'away_team':  row['away_team'],
        'p_home_win': round(float(probs[i, 2]), 4),
        'p_draw':     round(float(probs[i, 1]), 4),
        'p_away_win': round(float(probs[i, 0]), 4),
    })

groups = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Haiti', 'Morocco', 'Scotland'],
    'D': ['Australia', 'Paraguay', 'Turkey', 'United States'],
    'E': ['Curaçao', 'Ecuador', 'Germany', 'Ivory Coast'],
    'F': ['Japan', 'Netherlands', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Cape Verde', 'Saudi Arabia', 'Spain', 'Uruguay'],
    'I': ['France', 'Iraq', 'Norway', 'Senegal'],
    'J': ['Algeria', 'Argentina', 'Austria', 'Jordan'],
    'K': ['Colombia', 'DR Congo', 'Portugal', 'Uzbekistan'],
    'L': ['Croatia', 'England', 'Ghana', 'Panama'],
}

elo_df  = pd.read_csv(os.path.join(DATA_DIR, 'elo_ratings.csv'))
elo_dict = dict(zip(elo_df['team'], elo_df['elo']))

round_of_16 = [
    ('Canada',      'Morocco'),
    ('Paraguay',    'France'),
    ('Brazil',      'Norway'),
    ('Mexico',      'England'),
    ('Portugal',    'Spain'),
    ('United States', 'Belgium'),
    ('Argentina',   'Egypt'),
    ('Switzerland', 'Colombia'),
]

def get_knockout_probs(team_a, team_b, elo_dict):
    """
    For knockout rounds there are no draws.
    """
    elo_a = elo_dict.get(team_a, 1500)
    elo_b = elo_dict.get(team_b, 1500)

    # Base win probability from Elo
    p_a_wins = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    p_b_wins = 1 - p_a_wins

    return round(p_a_wins, 4), round(p_b_wins, 4)

knockout_probs = []
for team_a, team_b in round_of_16:
    p_a, p_b = get_knockout_probs(team_a, team_b, elo_dict)
    knockout_probs.append({
        'team_a':   team_a,
        'team_b':   team_b,
        'p_a_wins': p_a,
        'p_b_wins': p_b,
        'round':    'R16'
    })

export = {
    'matches':         matches,
    'groups':          groups,
    'elo_ratings':     elo_dict,
    'knockout_probs':  knockout_probs,
    'round_of_16':     [{'team_a': a, 'team_b': b} for a, b in round_of_16],
}

out_path = os.path.join(DATA_DIR, 'app_data.json')
with open(out_path, 'w') as f:
    json.dump(export, f, indent=2)

print(f"✅ Exported {len(matches)} group matches")
print(f"✅ Exported {len(knockout_probs)} knockout fixtures")
print(f"   Copy data/app_data.json into your Xcode project")