import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm

print("Running prediction...")

# Load model and scaler
scaler, model = joblib.load("models/logreg_with_scaler.pkl")

# Load features
features_df = pd.read_csv("data/mlb_features.csv")

# Keep only scheduled games
upcoming = features_df[features_df["target"] == -1].copy()
completed = features_df[features_df["target"] != -1]

if upcoming.empty:
    print("No upcoming games found in this feature file.")
else:
    # Predict upcoming games
    X_upcoming = upcoming.drop(columns=["target", "home_team", "away_team", "home_starter", "away_starter", "date", "home_score", "away_score"])
    X_scaled = scaler.transform(X_upcoming)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    upcoming["prediction"] = predictions
    upcoming["home_win_prob"] = probabilities.round(3)

    # Compute predicted final standings
    teams = pd.concat([features_df["home_team"], features_df["away_team"]]).unique()
    total_wins = {team: [] for team in teams}
    team_ranks = {team: [] for team in teams}
    n_simulations = 1000

    for sim in tqdm(range(n_simulations), desc="Simulating seasons", colour="cyan"):
        sim_wins = {}

        # Count actual wins from completed games
        for team in teams:
            home_wins = completed[completed["home_team"] == team]["target"].sum()
            away_wins = completed[completed["away_team"] == team]["target"].apply(lambda x: 0 if x == 1 else 1).sum()
            sim_wins[team] = home_wins + away_wins

        # Add predicted wins from upcoming games
        for idx, row in upcoming.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            prob = row["home_win_prob"]
            # Sample outcome: 1 = home wins, 0 = away wins
            home_win = np.random.binomial(1, prob)
            if home_win:
                sim_wins[home] += 1
            else:
                sim_wins[away] += 1

        # Sort teams by wins
        sim_rank_df = pd.DataFrame({
            "team": list(sim_wins.keys()),
            "wins": list(sim_wins.values())
        }).sort_values("wins", ascending=False).reset_index(drop=True)

        # Add 1 to avg rank to account for zero-based indexing
        for idx, row in sim_rank_df.iterrows():
            team = row["team"]
            team_ranks[team].append(idx + 1)

        for team, wins in sim_wins.items():
            total_wins[team].append(wins)

    # Create standings DataFrame
    final_standings = pd.DataFrame({
        "team": teams,
        "predicted_wins": [round(np.mean(total_wins[team]), 2) for team in teams],
        "avg_rank": [round(np.mean(team_ranks[team]), 2) for team in teams]
    })
    final_standings = final_standings.sort_values("predicted_wins", ascending=False)
    final_standings["rank"] = range(1, len(final_standings) + 1)

    # Organize for output
    final_standings = final_standings[["rank", "team", "predicted_wins", "avg_rank"]]

    print("\n\033[1;35mPredicted Final Standings:\033[0m")
    print(final_standings.to_string(index=False))
