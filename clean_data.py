import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load the files we need
results = pd.read_csv(os.path.join(DATA_DIR, 'results.csv'), parse_dates=['date'])
former_names = pd.read_csv(os.path.join(DATA_DIR, 'former_names.csv'), parse_dates=['start_date', 'end_date'])

# Build a mapping from every former name to its current name
name_mapping = dict(zip(former_names['former'], former_names['current']))

print("Name mappings found:")
for old, new in name_mapping.items():
    print(f"  {old} -> {new}")

#Renaming teams to how they appear on an english table
name_mapping['Czech Republic'] = 'Czechia'
name_mapping['Türkiye'] = 'Turkey'

# Apply the mapping to both home and away team columns
results['home_team'] = results['home_team'].replace(name_mapping)
results['away_team'] = results['away_team'].replace(name_mapping)

# Verify it worked - check a specific case
print("\n--- Checking DR Congo fix ---")
print(results[results['home_team'] == 'DR Congo'].head(3))
print(results[results['home_team'] == 'Congo-Kinshasa'].head(3))

# Save the cleaned version to a new file
results.to_csv(os.path.join(DATA_DIR, 'results_clean.csv'), index=False)
print(f'Total rows; {len(results):,}')