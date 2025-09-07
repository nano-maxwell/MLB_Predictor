import pandas as pd
from tqdm import tqdm
from get_stats import get_pitcher_stat, get_game_stats, get_pitcher_stat_cached

print("Running feature engineering...")

# Load CSV
df = pd.read_csv("data/mlb-2025-asplayed.csv")

# Convert dates to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

# Add home_win column for completed games
df["home_win"] = (df["Home Score"] > df["Away Score"]).astype(int)

# Containers for team stats and features
team_stats = {}  # Running stats per team
features = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing games", colour="cyan"):
    date = row["Date"]
    home = row["Home"]
    away = row["Away"]
    status = row["Status"]
    home_score = row.get("Home Score", 0)
    away_score = row.get("Away Score", 0)
    away_starter = row.get("Away Starter", 0)
    home_starter = row.get("Home Starter", 0)
    home_win = row.get("home_win", 0)

    # Initialize team stats if first occurrence
    for team in [home, away]:
        if team not in team_stats:
            team_stats[team] = {
                "games": 0,
                "wins": 0,
                "runs_scored": 0,
                "runs_allowed": 0,
                "last10": []
            }

    home_record = team_stats[home]
    away_record = team_stats[away]

    # Compute features based on historical stats
    feat = {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_win_pct": home_record["wins"] / home_record["games"] if home_record["games"] > 0 else 0.5,
        "away_win_pct": away_record["wins"] / away_record["games"] if away_record["games"] > 0 else 0.5,
        "home_last10_win_pct": sum(home_record["last10"][-10:]) / len(home_record["last10"][-10:]) if home_record["last10"] else 0.5,
        "away_last10_win_pct": sum(away_record["last10"][-10:]) / len(away_record["last10"][-10:]) if away_record["last10"] else 0.5,
        "home_runs_pg": home_record["runs_scored"] / home_record["games"] if home_record["games"] > 0 else 4.5,
        "away_runs_pg": away_record["runs_scored"] / away_record["games"] if away_record["games"] > 0 else 4.5,
        "home_runs_allowed_pg": home_record["runs_allowed"] / home_record["games"] if home_record["games"] > 0 else 4.5,
        "away_runs_allowed_pg": away_record["runs_allowed"] / away_record["games"] if away_record["games"] > 0 else 4.5,
        "home_pitcher_era": get_pitcher_stat_cached(player_name=home_starter, stat="era") if pd.notna(home_starter) else 4.25,
        "away_pitcher_era": get_pitcher_stat_cached(player_name=away_starter, stat="era") if pd.notna(away_starter) else 4.25,
        "home_pitcher_whip": get_pitcher_stat_cached(player_name=home_starter, stat="whip") if pd.notna(home_starter) else 1.35,
        "away_pitcher_whip": get_pitcher_stat_cached(player_name=away_starter, stat="whip") if pd.notna(away_starter) else 1.35,
        "home_score": home_score,
        "away_score": away_score,
        # Target: 1/0 for played games, -1 for scheduled
        "target": home_win if status != "Scheduled" else -1
    }

    features.append(feat)

    # Update stats only for completed games
    if status != "Scheduled":
        team_stats[home]["games"] += 1
        team_stats[home]["wins"] += 1 if home_win else 0
        team_stats[home]["runs_scored"] += home_score
        team_stats[home]["runs_allowed"] += away_score
        team_stats[home]["last10"].append(1 if home_win else 0)
        if len(team_stats[home]["last10"]) > 10:
            team_stats[home]["last10"].pop(0)

        team_stats[away]["games"] += 1
        team_stats[away]["wins"] += 0 if home_win else 1
        team_stats[away]["runs_scored"] += away_score
        team_stats[away]["runs_allowed"] += home_score
        team_stats[away]["last10"].append(0 if home_win else 1)
        if len(team_stats[away]["last10"]) > 10:
            team_stats[away]["last10"].pop(0)

# Save features
features_df = pd.DataFrame(features)
# Select numeric columns only
numeric_cols = features_df.select_dtypes(include='number').columns

# Fill NaNs only for numeric columns
features_df[numeric_cols] = features_df[numeric_cols].fillna(0.5)

features_df.to_csv("data/mlb_features.csv", index=False)

print("Feature engineering complete. Features saved to data/mlb_features.csv")
