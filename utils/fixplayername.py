import statsapi
import unicodedata

def strip_accents(text: str) -> str:
    """Normalize string by removing accents."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

# Cache dictionary to reduce API calls
fixed_names_cache = {}

def fix_player_name(team_name, player_name):
    """
    Fix pitcher names that contain � by looking up the correct name
    from the team roster. Uses a cache to avoid repeated API calls.
    """
    # If already fixed, return cached result
    if player_name in fixed_names_cache:
        return fixed_names_cache[player_name]

    # If no issue, just return the name
    if "�" not in player_name:
        fixed_names_cache[player_name] = player_name
        return player_name

    try:
        # Get team ID
        team_id = statsapi.lookup_team(team_name)[0]["id"]

        # Get roster for the team
        roster = statsapi.roster(team_id, rosterType="pitching")

        # Try to find match
        for player in roster:
            if player_name.replace("�", "").lower() in player["person"]["fullName"].lower().replace("é", "e").replace("í", "i").replace("ó", "o").replace("á", "a"):
                fixed_names_cache[player_name] = player["person"]["fullName"]
                return player["person"]["fullName"]

        # If no match found, just return original
        fixed_names_cache[player_name] = player_name
        return player_name

    except Exception as e:
        print(f"Error fixing name {player_name}: {e}")
        fixed_names_cache[player_name] = player_name
        return player_name
