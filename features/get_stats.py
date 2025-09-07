import statsapi
import json
import os

DATA_DIR = "data"
CACHE_FILE = os.path.join(DATA_DIR, "pitcher_stats_cache.json")

# Load cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        pitcher_cache = json.load(f)
else:
    pitcher_cache = {}

def get_pitcher_stat_cached(player_name, stat, season="2025"):
    # Check cache first
    if player_name in pitcher_cache and stat in pitcher_cache[player_name]:
        return pitcher_cache[player_name][stat]

    try:
        # Fetch from API
        player_id = statsapi.lookup_player(player_name)[0]['id']
        stats_list = statsapi.player_stat_data(personId=player_id)['stats']
        value = next(
            (float(s['stats'][stat]) for s in stats_list
             if s.get('group') == 'pitching' and s.get('type') == 'season' and s.get('season') == season),
            None
        )

        # Initialize dictionary if player not in cache
        if player_name not in pitcher_cache:
            pitcher_cache[player_name] = {}

        # Add/Update stat
        pitcher_cache[player_name][stat] = value

        # Save updated cache
        with open(CACHE_FILE, "w") as f:
            json.dump(pitcher_cache, f)

        return value
    except Exception:
        return None



def get_pitcher_stat(player_name, stat, season="2025", default=0.0):
    try:
        player_id = statsapi.lookup_player(player_name)[0]["id"]
        stats_list = statsapi.player_stat_data(personId=player_id)["stats"]
        return float(next(
            (s["stats"][stat] for s in stats_list
             if s.get("group") == "pitching" and s.get("type") == "season" and s.get("season") == season),
            default
        ))
    except Exception:
        return default
    
def get_game_stats(date, team):
    team_id = get_team_id(team)
    return statsapi.schedule(date=date, team=team_id)

def get_team_id(team):
    team_info = statsapi.lookup_team(team)
    team_id = team_info[0]["id"]
    return team_id