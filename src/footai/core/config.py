from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[3] #to be changed if this goes to /src/
DATA_DIR      = ROOT_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR  = DATA_DIR / 'features'
FIG_DIR       = ROOT_DIR / 'figures'

COUNTRIES = {
    "SP" : {
        "name" : "Spain",
        "divisions": {
            "SP1" : "La Liga",
            "SP2" : "Segunda"
            },
        },
    "EN" : {
        "name" : "England",
        "divisions": {
            "E0" : "Premier League",
            "E1" : "Championship",
            "E2" : "EFL League 1",
            "E3" : "EFL League 2",
            "EC" : "National League"
            },
        },
    "IT" : {
        "name" : "Italy",
        "divisions": {
            "I1" : "Serie A",
            "I2" : "Serie B"
            },
        },
    "DE" : {
        "name" : "Germany",
        "divisions": {
            "D1" : "Bundesliga",
            "D2" : "2. Bundesliga"
            },
        },
    "FR" : {
        "name" : "France",
        "divisions": {
        "FR1" : "Ligue 1",
        "FR2" : "Ligue 2"
        }
    }
}

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
