"""
Project configuration and directory structure definitions.

This module defines the static root directories (raw, processed, features) and 
contains the master dictionary of supported countries and divisions. It acts as 
the single source of truth for the project's file organization and scope.
"""

from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[3]
DATA_DIR      = ROOT_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR  = DATA_DIR / 'features'
COLOR_DIR     = DATA_DIR / 'colors'
FIG_DIR       = ROOT_DIR / 'figures'

def setup_directories(args):
    '''Create dictionary with directories and ensure they exist'''
    dirs = {}
    verbose   = getattr(args, 'verbose', False)
    raw_dir_base       = getattr(args, 'raw_dir', RAW_DIR)
    processed_dir_base = getattr(args, 'processed_dir', PROCESSED_DIR)
    features_dir_base  = getattr(args, 'features_dir', FEATURES_DIR)
    for country in args.countries:
        dirs[country] = {
            'raw'  : raw_dir_base / country,
            'proc' : processed_dir_base / country,
            'feat' : features_dir_base / country,
            'col'  : COLOR_DIR,
            'fig'  : FIG_DIR

        }
        for dir in dirs[country].keys():
            if verbose: print("creating", dirs[country][dir])
            Path(dirs[country][dir]).mkdir(parents=True, exist_ok=True)
    return dirs

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
            #"E2" : "EFL League 1",
            #"E3" : "EFL League 2",
            #"EC" : "National League"
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
        "F1" : "Ligue 1",
        "F2" : "Ligue 2"
        }
    }
}




from footai.ml.feature_engineering.definitions import FEATURE_SETS

def select_features(df, feature_set="baseline"):
    """
    Select features for training.

    Args:
        df: DataFrame with features
        feature_set: Which feature set to use:

    Returns:
        List of feature column names
    """
    # Define columns to exclude (metadata, raw data, target)
    exclude_cols = [
        'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
        'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC',
        'HY', 'AY', 'HR', 'AR', 'Division', 'Season'  # Add any other metadata columns
    ]
    if feature_set == 'all':
            return [col for col in df.columns 
                    if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    return [col for col in FEATURE_SETS[feature_set] if col in df.columns]

def parse_countries(country_arg):
    """
    Parse country argument into list of countries.
    
    Args:
        country_arg: String like 'SP' or 'SP,IT' or 'SP IT'
    
    Returns:
        List of country codes (e.g., ['SP'] or ['SP', 'IT'])
        
    Examples:
        >>> parse_countries('SP')
        ['SP']
        >>> parse_countries('SP,IT')
        ['SP', 'IT']
        >>> parse_countries('SP IT')
        ['SP', 'IT']
    """

    # Handle comma-separated or space-separated
    if ',' in country_arg:
        countries = [c.strip().upper() for c in country_arg.split(',')]
    else:
        countries = [c.strip().upper() for c in country_arg.split()]
    
    # Validate all countries exist
    for c in countries:
        if c not in COUNTRIES.keys():
            raise ValueError(f"Unknown country code: {c}. Valid: {list(COUNTRIES.keys())}")
    
    return countries


def get_default_divisions(countries):
    """
    Get first two divisions for a country (Tier 1 + Tier 2).
    
    Args:
        countries: Single country code string or list of country codes (e.g., 'SP' or ['SP', 'IT'])    
    Returns:
        Dictionary mapping country -> list of division codes
        Example: 
            {'SP': ['SP1', 'SP2']} for Spain
            {'SP': ['SP1', 'SP2'], 'IT': ['I1', 'I2']} for Spain+Italy
    """
    
    country_divisions = {}
    for country in countries:
        country_divs = list(COUNTRIES[country]['divisions'].keys())
        country_divisions[country] = country_divs[:2]  # First two tiers per country
    
    return country_divisions

def get_divisions_for_countries(countries, divisions):
    """
    Map specified divisions to their countries.
    
    Args:
        countries: List of country codes (e.g., ['SP', 'IT'])
        divisions: List of division codes (e.g., ['SP1', 'I1'])
    
    Returns:
        Dictionary mapping country -> list of divisions
        Example: {'SP': ['SP1'], 'IT': ['I1']}
    
    Raises:
        ValueError: If a division doesn't belong to any specified country
    """
    country_divisions = {country: [] for country in countries}
    for country in countries:
        for division in divisions[country]:
            found = False
            if division in COUNTRIES[country]['divisions']:
                country_divisions[country].append(division)
                found = True
        
        if not found:
            raise ValueError(
                f"Division '{division}' doesn't belong to any specified country: {countries}"
            )
    
    return country_divisions
