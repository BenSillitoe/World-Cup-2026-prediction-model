import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

features = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

wc2026 = features[
    (features['date'] >= '2026-01-01') &
    (features['is_world_cup'] == 1)
].copy()

print(f"2026 World Cup matches in dataset: {len(wc2026)}")
print(f"Date range: {wc2026['date'].min().date()} to {wc2026['date'].max().date()}")
print()
print(wc2026[['date', 'home_team', 'away_team', 'outcome']].to_string(index=False))