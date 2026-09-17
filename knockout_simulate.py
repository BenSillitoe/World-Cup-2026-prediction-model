import pandas as pd
import numpy as np
import os
import json
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

elo_df   = pd.read_csv(os.path.join(DATA_DIR, 'elo_ratings.csv'))
elo_dict = dict(zip(elo_df['team'], elo_df['elo']))

from stadiums import STADIUMS, adjusted_win_prob

# Round of 16 with venues — check FIFA site for actual assignments
round_of_16 = [
    ('Canada',        'Morocco',   'BC Place'),
    ('Paraguay',      'France',    'AT&T Stadium'),
    ('Brazil',        'Norway',    'Hard Rock Stadium'),
    ('Mexico',        'England',   'Estadio Azteca'),
    ('Portugal',      'Spain',     'MetLife Stadium'),
    ('United States', 'Belgium',   'SoFi Stadium'),
    ('Argentina',     'Egypt',     'NRG Stadium'),
    ('Switzerland',   'Colombia',  'Estadio BBVA'),
]

def get_win_prob(team_a, team_b, stadium=None):
    if stadium:
        return adjusted_win_prob(team_a, team_b, stadium, elo_dict)
    elo_a = elo_dict.get(team_a, 1500)
    elo_b = elo_dict.get(team_b, 1500)
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


N = 10000
advance_count = {}
qf_count  = {}
sf_count  = {}
final_count = {}
winner_count = {}

for team_a, team_b, stadium in round_of_16:
    advance_count[team_a] = 0
    advance_count[team_b] = 0

for _ in range(N):
    # Round of 16 — uses actual venue for each match
    r16_winners = []
    for team_a, team_b, stadium in round_of_16:
        p = get_win_prob(team_a, team_b, stadium)
        winner = team_a if np.random.random() < p else team_b
        r16_winners.append(winner)
        advance_count[winner] = advance_count.get(winner, 0) + 1

    # Quarter finals — no fixed venue assigned yet, use neutral Elo
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        if i+1 < len(r16_winners):
            ta, tb = r16_winners[i], r16_winners[i+1]
            p = get_win_prob(ta, tb)
            winner = ta if np.random.random() < p else tb
            qf_winners.append(winner)
            qf_count[winner] = qf_count.get(winner, 0) + 1

    # Semi finals
    sf_winners = []
    for i in range(0, len(qf_winners), 2):
        if i+1 < len(qf_winners):
            ta, tb = qf_winners[i], qf_winners[i+1]
            p = get_win_prob(ta, tb)
            winner = ta if np.random.random() < p else tb
            sf_winners.append(winner)
            sf_count[winner] = sf_count.get(winner, 0) + 1

    # Final
    if len(sf_winners) == 2:
        ta, tb = sf_winners[0], sf_winners[1]
        p = get_win_prob(ta, tb)
        champion = ta if np.random.random() < p else tb
        final_count[ta]    = final_count.get(ta, 0) + 1
        final_count[tb]    = final_count.get(tb, 0) + 1
        winner_count[champion] = winner_count.get(champion, 0) + 1

# --- Print results ---
print(f"\n{'='*65}")
print(f"  KNOCKOUT STAGE SIMULATION ({N:,} runs)")
print(f"{'='*65}")

print(f"\n  ROUND OF 16 — MATCH PREDICTIONS (with stadium conditions)")
print(f"  {'-'*55}")
for team_a, team_b, stadium in round_of_16:
    p_a = get_win_prob(team_a, team_b, stadium)
    p_b = 1 - p_a
    predicted = team_a if p_a > p_b else team_b
    print(f"  {team_a:<22} {p_a:.0%}  vs  {p_b:.0%}  {team_b:<22}  [{stadium}] → {predicted}")

all_teams = list(advance_count.keys())

print(f"\n  TOURNAMENT PROGRESSION PROBABILITIES")
print(f"  {'Team':<25} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Win':>6}")
print(f"  {'-'*60}")

sorted_teams = sorted(all_teams, key=lambda t: winner_count.get(t, 0), reverse=True)
for team in sorted_teams:
    r16  = advance_count.get(team, 0) / N * 100
    qf   = qf_count.get(team, 0)     / N * 100
    sf   = sf_count.get(team, 0)     / N * 100
    fin  = final_count.get(team, 0)  / N * 100
    win  = winner_count.get(team, 0) / N * 100
    print(f"  {team:<25} {r16:>5.1f}% {qf:>5.1f}% {sf:>5.1f}% {fin:>5.1f}% {win:>5.1f}%")

print(f"\n{'='*65}")
print(f"  TOP 5 WORLD CUP WINNER PREDICTIONS")
print(f"{'='*65}")
top5 = sorted(winner_count.items(), key=lambda x: x[1], reverse=True)[:5]
for team, count in top5:
    pct = count / N * 100
    bar = '█' * int(pct / 2)
    print(f"  {team:<25} {pct:>5.1f}%  {bar}")

# Save
output = {
    'round_of_16_probs': [
        {
            'team_a': a, 'team_b': b, 'stadium': s,
            'p_a_wins': round(get_win_prob(a, b, s), 4),
            'p_b_wins': round(1 - get_win_prob(a, b, s), 4),
        }
        for a, b, s in round_of_16
    ],
    'tournament_probs': [
        {
            'team': t,
            'p_r16':   round(advance_count.get(t, 0) / N, 4),
            'p_qf':    round(qf_count.get(t, 0)      / N, 4),
            'p_sf':    round(sf_count.get(t, 0)       / N, 4),
            'p_final': round(final_count.get(t, 0)    / N, 4),
            'p_win':   round(winner_count.get(t, 0)   / N, 4),
        }
        for t in sorted_teams
    ]
}

with open(os.path.join(DATA_DIR, 'knockout_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to data/knockout_results.json")