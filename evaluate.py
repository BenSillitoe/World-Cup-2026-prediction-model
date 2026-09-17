import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load model and features
model       = joblib.load(os.path.join(DATA_DIR, 'model.pkl'))
feature_cols = joblib.load(os.path.join(DATA_DIR, 'feature_cols.pkl'))
features    = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

# --- Actual 2026 World Cup results ---
actual_results = []

# --- MATCHDAY 1 --- (add scores as they come in)
actual_results.append(('South Korea',   'Czechia',                2, 1))
actual_results.append(('Mexico',        'South Africa',           2, 0))

# --- MATCHDAY 2 ---
actual_results.append(('Canada',        'Bosnia and Herzegovina', 1, 1))
actual_results.append(('United States', 'Paraguay',               4, 1))

# --- MATCHDAY 3 ---
actual_results.append(('Qatar',         'Switzerland',            1, 1))

# Convert to DataFrame
results_df = pd.DataFrame(actual_results,
    columns=['home_team', 'away_team', 'home_score', 'away_score'])

# Determine actual outcomes
def get_outcome(row):
    if row['home_score'] > row['away_score']:
        return 2   # home win
    elif row['home_score'] == row['away_score']:
        return 1   # draw
    else:
        return 0   # away win

results_df['actual_outcome'] = results_df.apply(get_outcome, axis=1)

# --- Smart prediction with draw logic ---
def smart_predict(p_away_win, p_draw, p_home_win):
    """
    Predict draw when:
    - Draw probability is above 30%, AND
    - The gap between the two win probabilities is less than 15%
    """
    gap = abs(p_home_win - p_away_win)

    if p_draw > 0.30 and gap < 0.15:
        return 1  # draw
    elif p_home_win > p_away_win:
        return 2  # home win
    else:
        return 0  # away win

# --- Get model predictions for these matches ---
wc2026 = features[
    (features['date'] >= '2026-06-11') &
    (features['is_world_cup'] == 1)
].copy()

# Join actual results onto features
eval_df = results_df.merge(
    wc2026[['home_team', 'away_team'] + feature_cols],
    on=['home_team', 'away_team'],
    how='left'
)

# Check for any unmatched rows
missing = eval_df[eval_df[feature_cols[0]].isna()]
if len(missing) > 0:
    print("⚠️  Could not find features for:")
    print(missing[['home_team', 'away_team']].to_string(index=False))

eval_df = eval_df.dropna(subset=feature_cols)

# Get probabilities
X_eval = eval_df[feature_cols]
probs  = model.predict_proba(X_eval)

eval_df['p_away_win'] = probs[:, 0]
eval_df['p_draw']     = probs[:, 1]
eval_df['p_home_win'] = probs[:, 2]

# Apply smart prediction instead of raw argmax
eval_df['predicted_outcome'] = eval_df.apply(
    lambda r: smart_predict(r['p_away_win'], r['p_draw'], r['p_home_win']),
    axis=1
)

eval_df['correct'] = (eval_df['predicted_outcome'] == eval_df['actual_outcome']).astype(int)

# Outcome labels for display
outcome_labels = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

# --- Print match by match results ---
print(f"\n{'='*75}")
print(f"  MATCH BY MATCH PREDICTION ACCURACY")
print(f"{'='*75}")
print(f"  {'Match':<40} {'Predicted':>12} {'Actual':>12} {'Correct':>8}")
print(f"  {'-'*72}")

for _, row in eval_df.iterrows():
    match = f"{row['home_team']} vs {row['away_team']}"
    predicted = outcome_labels[row['predicted_outcome']]
    actual    = outcome_labels[row['actual_outcome']]
    correct   = '✅' if row['correct'] else '❌'
    print(f"  {match:<40} {predicted:>12} {actual:>12} {correct:>8}")

# --- Overall metrics ---
accuracy = eval_df['correct'].mean()
n_played = len(eval_df)

y_true = eval_df['actual_outcome'].values
y_prob = probs[:len(eval_df)]

try:
    ll = log_loss(y_true, y_prob)
    ll_str = f"{ll:.3f}"
except Exception:
    ll_str = "N/A"

print(f"\n{'='*75}")
print(f"  OVERALL PERFORMANCE ({n_played} matches played)")
print(f"{'='*75}")
print(f"  Accuracy:   {accuracy:.1%}  ({eval_df['correct'].sum()}/{n_played} correct)")
print(f"  Log Loss:   {ll_str}")

# Breakdown by outcome type
print(f"\n  ACCURACY BY OUTCOME TYPE")
print(f"  {'-'*40}")
for outcome_val, label in outcome_labels.items():
    subset = eval_df[eval_df['actual_outcome'] == outcome_val]
    if len(subset) > 0:
        acc = subset['correct'].mean()
        print(f"  {label:<12}  {acc:.1%}  ({subset['correct'].sum()}/{len(subset)} correct)")

# --- Probability calibration ---
print(f"\n  PROBABILITY CALIBRATION")
print(f"  (When model says X% for predicted outcome, how often is it right?)")
print(f"  {'-'*50}")

eval_df['predicted_prob'] = eval_df.apply(
    lambda r: r['p_home_win'] if r['predicted_outcome'] == 2
    else r['p_draw'] if r['predicted_outcome'] == 1
    else r['p_away_win'], axis=1
)

bins = [(0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 1.0)]
for low, high in bins:
    subset = eval_df[
        (eval_df['predicted_prob'] >= low) &
        (eval_df['predicted_prob'] < high)
    ]
    if len(subset) > 0:
        actual_acc = subset['correct'].mean()
        print(f"  Predicted {low:.0%}-{high:.0%}:  actual {actual_acc:.0%}  ({len(subset)} matches)")

# --- Biggest upsets (model was most wrong) ---
print(f"\n  BIGGEST UPSETS (model most confident, got it wrong)")
print(f"  {'-'*60}")
wrong = eval_df[eval_df['correct'] == 0].copy()
wrong['confidence'] = wrong['predicted_prob']
wrong = wrong.sort_values('confidence', ascending=False).head(5)

for _, row in wrong.iterrows():
    match = f"{row['home_team']} vs {row['away_team']}"
    print(f"  {match:<35} predicted {outcome_labels[row['predicted_outcome']]} "
          f"({row['confidence']:.0%}) → actual {outcome_labels[row['actual_outcome']]}")

# --- Save evaluation ---
eval_df.to_csv(os.path.join(DATA_DIR, 'evaluation.csv'), index=False)
print(f"\n✅ Evaluation saved to data/evaluation.csv")