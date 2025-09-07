import joblib
import argparse
import pandas as pd

TEAM_NAME_MAP = {
    "diamondbacks": "Arizona Diamondbacks",
    "braves": "Atlanta Braves",
    "orioles": "Baltimore Orioles",
    "redsox": "Boston Red Sox",
    "whitesox": "Chicago White Sox",
    "cubs": "Chicago Cubs",
    "reds": "Cincinnati Reds",
    "guardians": "Cleveland Guardians",
    "rockies": "Colorado Rockies",
    "tigers": "Detroit Tigers",
    "astros": "Houston Astros",
    "royals": "Kansas City Royals",
    "angels": "Los Angeles Angels",
    "dodgers": "Los Angeles Dodgers",
    "marlins": "Miami Marlins",
    "brewers": "Milwaukee Brewers",
    "twins": "Minnesota Twins",
    "yankees": "New York Yankees",
    "mets": "New York Mets",
    "athletics": "Las Vegas Athletics",
    "phillies": "Philadelphia Phillies",
    "pirates": "Pittsburgh Pirates",
    "padres": "San Diego Padres",
    "giants": "San Francisco Giants",
    "mariners": "Seattle Mariners",
    "cardinals": "St. Louis Cardinals",
    "rays": "Tampa Bay Rays",
    "rangers": "Texas Rangers",
    "bluejays": "Toronto Blue Jays",
    "nationals": "Washington Nationals"
}


def resolve_alias(name: str) -> str:
    # Return the standard name from alias
    key = name.strip().lower().replace(" ", "")

    return TEAM_NAME_MAP[key]

def prob_to_american(prob: float) -> int:
    if prob == 0:
        return 99999  # avoid division by zero
    if prob == 1:
        return -99999

    if prob >= 0.5:  # favorite
        return int(- (prob / (1 - prob)) * 100)
    else:  # underdog
        return int(((1 - prob) / prob) * 100)


# Parse arguments
parser = argparse.ArgumentParser(description="Predict outcome of a single MLB game")
parser.add_argument("--home", required=True, type=str, help="Home team name")
parser.add_argument("--away", required=True, type=str, help="Away team name")
args = parser.parse_args()

home_team = args.home
away_team = args.away

home_team = resolve_alias(home_team)
away_team = resolve_alias(away_team)

print(f"Predicting {home_team} vs {away_team}...")

features_df = pd.read_csv("data/mlb_features.csv")

features_df["date"] = pd.to_datetime(features_df["date"])

upcoming = features_df[features_df["target"] == -1].copy()
games = upcoming[((upcoming["home_team"] == home_team) & (upcoming["away_team"] == away_team))]

scaler, model = joblib.load("models/logreg_with_scaler.pkl")

if games.empty:
    print("No scheduled games for those teams.")
else:
    X = games.drop(columns=['home_team', 'away_team', 'home_starter', 'away_starter', 'target', 'date', 'home_score', 'away_score'])

    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[:, -1][0]
    winner = home_team if prob >= 0.5 else away_team
    loser = away_team if prob >= 0.5 else home_team
    win_prob = prob if winner == home_team else 1 - prob
    fair_odds = prob_to_american(win_prob)
    
    game_date = games.iloc[0]["date"].strftime("%A, %B %d, %Y").replace(" 0", " ")
    home_starter = games.iloc[0]["home_starter"]
    away_starter = games.iloc[0]["away_starter"]

    print(f"\n\033[1;35mPredicted winner:\033[0m {winner}")
    print(f"\nThe {winner} have a \033[1;35m{win_prob * 100:.1f}%\033[0m chance to win against the {loser} on {game_date}.")
    print(f"Pitching Matchup: {home_starter} ({home_team}) vs {away_starter} ({away_team}).")
    print(f"Fair odds for the {winner}: \033[1;35m{fair_odds:+d}\033[0m")