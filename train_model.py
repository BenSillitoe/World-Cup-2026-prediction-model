import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

features = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['date'])

# --- Split data ---
train_data = features[
    (features['date'] < '2026-06-11') &
    (features['outcome'].notna())
].copy()

wc2026_fixtures = features[
    (features['date'] >= '2026-06-11') &
    (features['is_world_cup'] == 1)
].copy()

print(f"Training rows:       {len(train_data):,}")
print(f"Fixtures to predict: {len(wc2026_fixtures)}")

# --- Features and target ---
feature_cols = [
    'elo_diff', 'home_elo', 'away_elo',
    'form_diff', 'home_form', 'away_form',
    'avg_scored_diff', 'home_avg_scored', 'away_avg_scored',
    'avg_conceded_diff', 'home_avg_conceded', 'away_avg_conceded',
    'is_competitive', 'is_world_cup', 'home_advantage'
]

X = train_data[feature_cols]
y = train_data['outcome'].astype(int)

# --- Sample weights — recent matches matter more ---
# Exponential decay with ~7 year half life
max_date = train_data['date'].max()
train_data['days_ago'] = (max_date - train_data['date']).dt.days
train_data['weight'] = np.exp(-train_data['days_ago'] / 2555)

# Also upweight World Cup matches
train_data['weight'] = np.where(
    train_data['is_world_cup'] == 1,
    train_data['weight'] * 2.0,
    train_data['weight']
)

# --- Time-based train/test split ---
split_idx = int(len(train_data) * 0.8)
X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]
w_train = train_data['weight'].iloc[:split_idx]

print(f"\nTrain set: {len(X_train):,} matches")
print(f"Test set:  {len(X_test):,} matches")

# --- Train XGBoost ---
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=w_train)

# --- Evaluate ---
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
logloss  = log_loss(y_test, y_pred_proba)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"Accuracy:  {accuracy:.3f}  ({accuracy*100:.1f}%)")
print(f"Log Loss:  {logloss:.3f}")

baseline_acc = (y_test == y_test.value_counts().idxmax()).mean()
print(f"Baseline:  {baseline_acc:.3f}  ({baseline_acc*100:.1f}%)")
print(f"Improvement over baseline: +{(accuracy - baseline_acc)*100:.1f}%")

# --- Feature importance ---
print(f"\n=== FEATURE IMPORTANCE ===")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"  {row['feature']:<25} {bar} {row['importance']:.3f}")

# --- Save ---
joblib.dump(model, os.path.join(DATA_DIR, 'model.pkl'))
joblib.dump(feature_cols, os.path.join(DATA_DIR, 'feature_cols.pkl'))
print(f"\n✅ Model saved to data/model.pkl")