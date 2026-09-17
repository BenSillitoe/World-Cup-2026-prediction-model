import pandas as pd
import os

#Build a path relative to data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

#Load all files
results = pd.read_csv(os.path.join(DATA_DIR, 'results.csv'), parse_dates=['date'])
rankings = pd.read_csv(os.path.join(DATA_DIR, 'fifa_ranking-2024-06-20.csv'), parse_dates=['rank_date'])
goalscorers = pd.read_csv(os.path.join(DATA_DIR, 'goalscorers.csv'), parse_dates=['date'])
shootouts = pd.read_csv(os.path.join(DATA_DIR, 'shootouts.csv'), parse_dates=['date'])
former_names = pd.read_csv(os.path.join(DATA_DIR, 'former_names.csv'))

#Checks to see if the data is loaded
print(f"Results: {len(results):,} rows | {results['date'].min().year} - {results['date'].max().year}")
print(f"Rankings: {len(rankings):,} rows")
print(f"Goalscorers: {len(goalscorers):,} rows")
print(f"Shootouts: {len(shootouts):,} rows")
print(f"Former names: {len(former_names):,} rows")

print("\n--- Results sample ---")
print(results.head())

print("\n--- Rankings sample ---")
print(rankings.head())

print(rankings['rank_date'].max())