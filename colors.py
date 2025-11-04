"""Team colors for Spanish football leagues"""

TEAM_COLORS = {
    # La Liga (SP1)
    'Real Madrid': '#004996',           # Blue (not white for visibility)
    'Barcelona': '#A60042',             # Blaugrana maroon
    'Atlético Madrid': '#CF321F',       # Red
    'Athletic Bilbao': '#EE2523',       # Red-orange (distinct from Atlético)
    'Real Sociedad': '#003DA5',         # Blue
    'Villarreal': '#F5D547',            # Yellow
    'Valencia': '#FF7900',              # Orange
    'Real Betis': '#00954C',            # Green
    'Osasuna': '#E31919',               # Dark red (distinct from Athletic)
    'Celta Vigo': '#87CEEB',            # Sky blue
    'Sevilla': '#C60C30',               # Red
    'Rayo Vallecano': '#FFFFFF',        # White with black border (fallback)
    # La Liga (SP2)
    'Santander': '#009900'
}

def get_team_color(team_name):
    """
    Get color for a team, fallback to generated color if not found.
    
    Args:
        team_name: Team name (must match CSV exactly)
    
    Returns:
        HEX color code string
    """
    return TEAM_COLORS.get(team_name, generate_fallback_color(team_name))

def generate_fallback_color(team_name, saturation=0.7, lightness=0.5):
    """
    Generate a fallback color based on team name hash for unknown teams.
    Ensures unknown teams still get distinct colors.
    """
    import colorsys
    hash_value = hash(team_name) % 360  # Get value 0-359
    hue = hash_value / 360
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return 'rgb({},{},{})'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_team_colors_dict(team_names):
    """
    Get color mapping for a list of teams.
    
    Args:
        team_names: List of team names
    
    Returns:
        Dict mapping team name -> color
    """
    return {team: get_team_color(team) for team in team_names}
