import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

features_df = pd.read_csv("data/mlb_features.csv")

played_games_df = features_df[features_df["target"] != -1].copy()

x = played_games_df.drop(columns=["target", "home_team", "away_team", "home_starter", "away_starter", "date", "home_score", "away_score"])
x = x.fillna(0.5)
y = played_games_df["target"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Optional: scale features (helps some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train model
model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.025,
    max_depth=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump((scaler, model), "models/logreg_with_scaler.pkl")

import matplotlib.pyplot as plt

# Get importance scores
importance = model.feature_importances_

# Map to feature names
feature_names = x.columns
feat_imp = dict(zip(feature_names, importance))

# Sort by importance
feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

# Print
for f, score in feat_imp.items():
    print(f"{f}: {score:.3f}")

plt.figure(figsize=(10,6))
plt.barh(list(feat_imp.keys()), list(feat_imp.values()))
plt.gca().invert_yaxis()  # largest on top
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.show()

