import pandas as pd
import numpy as np
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load model and features
model = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
features = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

# Get 2026 World Cup group stage fixtures
wc2026 = features[
    (features['date'] >= '2026-06-11') &
    (features['is_world_cup'] == 1)
].copy()

print(f"Fixtures loaded: {len(wc2026)}")

# --- Define groups ---
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

# --- Get match probabilities from model ---
X_pred = wc2026[feature_cols]
probabilities = model.predict_proba(X_pred)

# probabilities columns: [P(away win), P(draw), P(home win)]
wc2026['p_away_win'] = probabilities[:, 0]
wc2026['p_draw']     = probabilities[:, 1]
wc2026['p_home_win'] = probabilities[:, 2]

# Build a lookup dict: (home_team, away_team) -> (p_home_win, p_draw, p_away_win)
match_probs = {}
for _, row in wc2026.iterrows():
    match_probs[(row['home_team'], row['away_team'])] = (
        row['p_home_win'], row['p_draw'], row['p_away_win']
    )

print(f"Match probabilities generated: {len(match_probs)}")

# --- Print predicted probabilities for each match ---
print("\n=== MATCH PROBABILITIES ===")
for _, row in wc2026.iterrows():
    home = row['home_team']
    away = row['away_team']
    print(f"  {home:>25} vs {away:<25}  "
          f"W:{row['p_home_win']:.1%}  D:{row['p_draw']:.1%}  L:{row['p_away_win']:.1%}")

# --- Monte Carlo simulation ---
N_SIMULATIONS = 10000

# Track results
team_qualify_count = {team: 0 for group in groups.values() for team in group}
team_group_winner  = {team: 0 for group in groups.values() for team in group}
team_points_total  = {team: 0 for group in groups.values() for team in group}

fallback_count = 0

for sim in range(N_SIMULATIONS):
    all_third_place = []

    for group_name, teams in groups.items():
        points = {team: 0 for team in teams}
        gd     = {team: 0 for team in teams}
        gf     = {team: 0 for team in teams}

        for i, team_a in enumerate(teams):
            for team_b in teams[i+1:]:

                # Get probabilities always from team_a's perspective
                if (team_a, team_b) in match_probs:
                    p_a_win, p_draw, p_b_win = match_probs[(team_a, team_b)]
                elif (team_b, team_a) in match_probs:
                    p_b_win, p_draw, p_a_win = match_probs[(team_b, team_a)]
                else:
                    fallback_count += 1
                    if fallback_count == 1:
                        print(f"FALLBACK: {team_a} vs {team_b} not found in match_probs")
                    p_a_win, p_draw, p_b_win = 0.4, 0.2, 0.4

                # Normalise probabilities
                total = p_a_win + p_draw + p_b_win
                p_a_win /= total
                p_draw  /= total
                p_b_win /= total

                # Simulate outcome
                outcome = np.random.choice(
                    ['a_win', 'draw', 'b_win'],
                    p=[p_a_win, p_draw, p_b_win]
                )

                # Simulate goals for goal difference tiebreaker
                if outcome == 'a_win':
                    a_goals = np.random.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
                    b_goals = np.random.choice([0, 1],    p=[0.65, 0.35])
                elif outcome == 'draw':
                    g = np.random.choice([0, 1, 2], p=[0.30, 0.45, 0.25])
                    a_goals = b_goals = g
                else:
                    b_goals = np.random.choice([1, 2, 3], p=[0.45, 0.35, 0.20])
                    a_goals = np.random.choice([0, 1],    p=[0.65, 0.35])

                # Award points using team_a and team_b consistently
                if outcome == 'a_win':
                    points[team_a] += 3
                elif outcome == 'draw':
                    points[team_a] += 1
                    points[team_b] += 1
                else:
                    points[team_b] += 3

                # Update goal difference
                gd[team_a] += a_goals - b_goals
                gd[team_b] += b_goals - a_goals
                gf[team_a] += a_goals
                gf[team_b] += b_goals

        # Rank teams: points, then goal difference, then goals scored
        standings = sorted(teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True)

        # 1st and 2nd qualify directly
        team_qualify_count[standings[0]] += 1
        team_qualify_count[standings[1]] += 1
        team_group_winner[standings[0]] += 1

        for t in teams:
            team_points_total[t] += points[t]

        third = standings[2]
        all_third_place.append((points[third], gd[third], gf[third], third))

    # Best 8 third-place teams also qualify
    all_third_place.sort(reverse=True)
    for _, _, _, team in all_third_place[:8]:
        team_qualify_count[team] += 1

print(f"Total fallbacks across all simulations: {fallback_count}")

# --- Build results table ---
results_list = []
for group_name, teams in groups.items():
    for team in teams:
        results_list.append({
            'group': group_name,
            'team': team,
            'qualify_prob': team_qualify_count[team] / N_SIMULATIONS * 100,
            'group_winner_prob': team_group_winner[team] / N_SIMULATIONS * 100,
            'avg_points': team_points_total[team] / N_SIMULATIONS,
        })

sim_results = pd.DataFrame(results_list)

# --- Print results by group ---
print(f"\n{'='*65}")
print(f"  WORLD CUP 2026 GROUP STAGE SIMULATION ({N_SIMULATIONS:,} runs)")
print(f"{'='*65}")

for group_name in sorted(groups.keys()):
    group_df = sim_results[sim_results['group'] == group_name].sort_values(
        'qualify_prob', ascending=False
    )
    print(f"\n  GROUP {group_name}")
    print(f"  {'Team':<28} {'Qualify':>8} {'Win Group':>10} {'Avg Pts':>8}")
    print(f"  {'-'*56}")
    for _, row in group_df.iterrows():
        bar = '█' * int(row['qualify_prob'] / 5)
        print(f"  {row['team']:<28} {row['qualify_prob']:>7.1f}% "
              f"{row['group_winner_prob']:>9.1f}% "
              f"{row['avg_points']:>7.1f}  {bar}")

# --- Overall rankings ---
print(f"\n{'='*65}")
print(f"  OVERALL QUALIFICATION PROBABILITY")
print(f"{'='*65}")
overall = sim_results.sort_values('qualify_prob', ascending=False)
for i, (_, row) in enumerate(overall.iterrows(), 1):
    bar = '█' * int(row['qualify_prob'] / 5)
    print(f"  {i:>2}. {row['team']:<28} {row['qualify_prob']:>6.1f}%  {bar}")

# --- Save results ---
sim_results.to_csv(os.path.join(DATA_DIR, 'simulation_results.csv'), index=False)
print(f"\n✅ Results saved to data/simulation_results.csv")

