import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

results = pd.read_csv(os.path.join(DATA_DIR, 'results_clean.csv'))

# All 48 World Cup 2026 teams
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

all_teams = set(results['home_team']).union(set(results['away_team']))

missing = [team for team in wc2026_teams if team not in all_teams]

if missing:
    print("⚠️  Missing teams - need name fixes:")
    for t in missing:
        print(f"  {t}")
else:
    print("✅ All 48 WC2026 teams found in dataset")