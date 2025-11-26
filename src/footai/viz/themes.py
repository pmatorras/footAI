"""Team colors for Spanish football leagues"""


import json
from footai.utils.config import COLOR_DIR

def load_team_colors_from_json(country='SP'):
    """
    Load colors from the country-specific JSON file.
    
    Args:
        country: Country code (e.g. 'SP', 'EN')
        
    Returns:
        Dict mapping team names to list of colors
    """
    json_path = COLOR_DIR / f'{country}_colors.json'
    
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'colors' in data:
                    return data['colors']
                return data
        except Exception as e:
            print(f"Warning: Could not load colors for {country}: {e}")
    return {}


def is_visible_on_white(hex_color, threshold=230):
    """
    Check if a color is visible on a white background.
    
    Args:
        hex_color: Hex color string
        threshold: Luminance threshold (0-255)
        
    Returns:
        True if visible, False otherwise
    """
    if not hex_color or not hex_color.startswith('#'):
        return True
        
    try:
        h = hex_color.lstrip('#')
        if len(h) == 3: h = ''.join([c*2 for c in h])
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return luminance < threshold
    except:
        return True


def pick_best_color(colors):
    """
    Pick the best visible color from a list of options.
    
    Args:
        colors: List of hex strings or single string
        
    Returns:
        Selected hex color string
    """
    if isinstance(colors, str):
        return colors
        
    if not colors:
        return None

    for color in colors:
        if is_visible_on_white(color):
            return color
            
    return "#333333"

def pick_best_color(colors):
    """
    Pick the best visible color from a list of options.
    
    Args:
        colors: List of hex strings or single string
        
    Returns:
        Selected hex color string
    """
    if isinstance(colors, str): return colors
        
    if not colors: return None
    if is_visible_on_white(colors[0]):
        return colors[0]
    
    for color in colors[1:]:
        if is_visible_on_white(color):
            return color
            
    return "#333333"


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

OVERRIDES = {
    'Real Madrid': "#940F77", # Use Black instead of Blue to distinguish from Barca
}


def get_team_color(team_name, loaded_colors=None):
    """
    Get color for a team, fallback to generated color if not found.
    
    Args:
        team_name: Team name (must match CSV exactly)
        loaded_colors: Optional dict of loaded colors

    
    Returns:
        HEX color code string
    """
    if team_name in OVERRIDES:
        return OVERRIDES[team_name]
    if loaded_colors and team_name in loaded_colors:
        raw_color = loaded_colors[team_name]
        
        if raw_color == "#CCCCCC" or (isinstance(raw_color, list) and not raw_color):
            return generate_fallback_color(team_name)
            
        return pick_best_color(raw_color)

    return generate_fallback_color(team_name)

def get_team_colors_dict(team_names, country='SP'):
    """
    Get color mapping for a list of teams.
    
    Args:
        team_names: List of team names
    
    Returns:
        Dict mapping team name -> color
    """
    loaded_colors = load_team_colors_from_json(country)

    return {team: get_team_color(team, loaded_colors) for team in team_names}