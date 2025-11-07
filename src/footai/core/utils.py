"""Utility functions for season handling."""
from footai.core.config import year_to_season_code

def parse_start_years(years_str):
    """
    Parse season start years and convert to season codes.
    
    Takes comma-separated years and converts each to a compact season format.
    Supports both 4-digit (2024) and 2-digit (24) formats.
    
    Args:
        seasons_str: Comma-separated years (e.g., "2024,2025" or "23,24")
    
    Returns:
        List of season codes (e.g., ['2425', '2526'])
    
    Examples:
        >>> parse_season_codes("2024,2025")
        ['2425', '2526']
        >>> parse_season_codes("23,24")
        ['2324', '2425']
    """
    years = []
    for year_str in years_str.split(','):
        year_str = year_str.strip()
        year = int(year_str)
        
        # If 2-digit year, expand to 20xx
        if year < 100:
            year = 2000 + year
        
        years.append(year_to_season_code(year))
    
    return years
