import joblib
import argparse
import pandas as pd
from datetime import datetime

# Color formatting strings
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
YELLOW = "\033[93m"

def prob_to_american(prob: float) -> int:
    if prob == 0:
        return 99999  # avoid division by zero
    if prob == 1:
        return -99999
    if prob >= 0.5:
        return int(- (prob / (1 - prob)) * 100)
    else:
        return int(((1 - prob) / prob) * 100)

# Argument parser
parser = argparse.ArgumentParser(description="Predict all MLB games on a day")
parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Date in format YYYY-MM-DD")
args = parser.parse_args()

DATE = args.date
DATE_FORMATTED = datetime.strptime(DATE, "%Y-%m-%d").strftime("%A, %B %d, %Y").replace(" 0", " ")

# Load model & features
scaler, model = joblib.load("models/logreg_with_scaler.pkl")
features_df = pd.read_csv("data/mlb_features.csv")

# Filter today's games
today = features_df[features_df["date"] == DATE].copy().reset_index(drop=True)

if today.empty:
    print(f"No games found on {DATE_FORMATTED}.")
else:
    X = today.drop(columns=["home_team", "away_team", "home_starter", "away_starter", "target", "date", "home_score", "away_score"])
    X_scaled = scaler.transform(X)

    col_width = [25, 25, 25, 7, 7]
    total_width = sum(col_width) + len(col_width) + 1  # accounting for separators

    # Print header with box-drawing characters
    print(f"\nMLB Predictions for {BOLD}{DATE_FORMATTED}{RESET}")
    print("╔" + "═"*total_width + "╗")
    print(f"║ {BLUE + BOLD}{'Home Team':{col_width[0]}} {YELLOW + BOLD}{'Away Team':{col_width[1]}}{RESET} {GREEN + BOLD}{'Predicted Winner':{col_width[2]}}{RESET} {MAGENTA + BOLD}{'Win %':>{col_width[3]}}{RESET} {MAGENTA + BOLD}{'Odds':>{col_width[4]}}{RESET} ║")
    print("╠" + "═"*total_width + "╣")

    # Print rows
    for idx, game in today.iterrows():
        home_team = today["home_team"][idx]
        away_team = today["away_team"][idx]

        prob = model.predict_proba(X_scaled)[:, -1][idx]
        winner = home_team if prob >= 0.5 else away_team
        win_prob = prob if winner == home_team else 1 - prob

        print(f"║ {home_team:{col_width[0]}} {away_team:{col_width[1]}} {winner:{col_width[2]}} {win_prob*100:{col_width[3]-1}.1f}% {prob_to_american(win_prob):{col_width[4]}} ║")

    print("╚" + "═"*total_width + "╝")
