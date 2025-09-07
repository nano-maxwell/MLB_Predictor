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
    "athletics": "Oakland Athletics",
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