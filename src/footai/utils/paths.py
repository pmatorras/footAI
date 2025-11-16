"""Utility functions for season handling."""
from pathlib import Path

def year_to_season_code(season):
    """Convert starting year to compact season format (e.g., 2024 -> '2425' for 2024-25)"""
    if isinstance(season, int) and season > 1000:
        next_year = (season % 100) + 1
        season = f"{season %100}{next_year:02d}"
    else:
        season = str(season)
    return season

def get_season_paths(season, division, dirs, args):
    """
    Get file paths for a season/division combination.
    
    Returns a dict with keys: 'season', 'raw', 'elo', 'fig'
    """
    
    if args.multi_season:
        suffix = '_transfer' if args.elo_transfer else '_multi'  
    else:
        suffix = ''

    raw_path = get_data_loc(season, division, args.country, dirs['raw'], verbose=args.verbose)
    elo_path = get_data_loc(season, division, args.country, dirs['proc'], file_type='elo', suffix=suffix, verbose=args.verbose)
    feat_path = get_data_loc(season, division, args.country, dirs['feat'], suffix='_feat', verbose=args.verbose)
    fig_path = get_data_loc(season, division, args.country, dirs['fig'], file_type='fig', suffix=suffix, verbose=args.verbose)
    
    return {
        'raw' : raw_path,
        'proc': elo_path,
        'feat': feat_path,
        'fig' : fig_path
    }

def get_promotion_relegation_file(dirs, country, season):
    '''A common promotion_relegation file path'''
    promo_dir = Path(dirs['proc']) / "promotion"
    promo_dir.mkdir(parents=True, exist_ok=True)

    return  promo_dir / f"{country}_{season}_promotion_relegation.csv"



def get_multiseason_path(multiseason_dir, division, season_start, season_end, args=None):
    suffix = '_transfer' if args.elo_transfer else '_multi'
    multiseason_dir.mkdir(parents=True, exist_ok=True)
    return multiseason_dir / f'{division}_{season_start}_to_{season_end}{suffix}.csv'

def get_data_loc(season, division, country, file_dir = None, file_type='', suffix='', verbose=False):
    """
    Generate file path for data storage.
    
    Returns path as: {country}_{season}_{division}[_elo][_multi][.csv|.html]
    """
    file_dir = Path(file_dir) if file_dir is not None else Path('.')
    file_dir.mkdir(parents=True, exist_ok=True)

    elo_suffix = '_elo' if 'elo' in file_type else ''
    file_type = '.html' if 'fig' in file_type else '.csv'
    # Create filename
    filename = f"{country.upper()}_{season}_{division}{elo_suffix}{suffix}{file_type}"
    path = file_dir / filename
    if verbose: print(f"Chosen file: {path}")
    return path

def get_previous_season(season_str):
    """Convert season string to previous season (e.g., '2324' -> '2223')."""
    year_start = int(season_str[:2])
    year_end = int(season_str[2:])
    
    prev_start = year_start - 1
    prev_end = year_end - 1
    
    return f"{prev_start:02d}{prev_end:02d}"

def parse_start_years(years_str):
    """
    Parse season start years and convert to season codes.
    
    Supports:
    - Ranges: '15-24' → ['15','16',...,'24']  (NEW!)
    - Individual: '22,23,24' → ['22','23','24']
    - Mixed: '15-20,23,24' → ['15','16',...,'20','23','24']  (NEW!)
    
    Also handles 4-digit years (2024 → '2425') for backward compatibility.
    
    Args:
        years_str: Comma-separated years, with optional ranges
                   (e.g., "2024,2025" or "22-24" or "15-24")
    
    Returns:
        List of season codes (e.g., ['2223', '2324', '2425'])
    """
    seasons = []
    
    for part in years_str.split(','):
        part = part.strip()
        
        if not part:
            continue
        
        # Handle range (e.g., '22-24')
        if '-' in part:
            try:
                start, end = part.split('-')
                start_year = int(start)
                end_year = int(end)
            except ValueError:
                raise ValueError(
                    f"Invalid season range format: '{part}'. "
                    f"Expected format: '15-24' or '2015-2024'"
                )
            
            # Validate range
            if start_year > end_year:
                raise ValueError(
                    f"Invalid season range: '{part}'. "
                    f"Start year must be <= end year"
                )
            
            # Handle both 2-digit and 4-digit years
            if start_year < 100:
                start_year = 2000 + start_year
            if end_year < 100:
                end_year = 2000 + end_year
            
            # Expand range
            for year in range(start_year, end_year + 1):
                seasons.append(year_to_season_code(year))
        
        else:
            # Single year (existing logic - backward compatible)
            year = int(part)
            
            # If 2-digit year, expand to 20xx
            if year < 100:
                year = 2000 + year
            
            seasons.append(year_to_season_code(year))
    
    if not seasons:
        raise ValueError(
            "No seasons provided. Expected format: '22-24' or '22,23,24'"
        )
    
    return seasons


# At the bottom of src/footai/utils/paths.py

def format_season_list(seasons):
    """
    Format a list of season codes for readable display.
    
    Converts sequential seasons to ranges:
    - ['2223','2324','2425'] → '2022-2025'
    - ['2223','2324','2526'] → '2022-2024, 2025-2026'
    
    Args:
        seasons (list[str]): List of season codes (e.g., ['2223', '2324'])
    
    Returns:
        str: Compact string representation (e.g., '2022-2025')
    
    Example:
        >>> format_season_list(['2223','2324','2425'])
        '2022-2025'
        
        >>> format_season_list(['2223','2324','2526'])
        '2022-2024, 2025-2026'
    """
    if not seasons:
        return ''
    
    # Convert season codes to start years (2223 → 2022)
    years = sorted([2000 + int(s[:2]) for s in seasons])
    
    # Group consecutive years into ranges
    ranges = []
    start = years[0]
    prev = years[0]
    
    for curr in years[1:] + [None]:
        if curr is None or curr != prev + 1:
            # End of range found
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev+1}")  # End year is prev+1 (season end)
            
            if curr is not None:
                start = curr
        
        if curr is not None:
            prev = curr
    
    return ', '.join(ranges)
