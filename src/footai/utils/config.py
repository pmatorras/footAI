from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parents[3] #to be changed if this goes to /src/
DATA_DIR      = ROOT_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR  = DATA_DIR / 'features'
FIG_DIR       = ROOT_DIR / 'figures'

def setup_directories(args):
    '''Create dictionary with directories and ensure they exist'''
    dirs = {}
    for country in args.countries:
        dirs[country] = {
            'raw'  : args.raw_dir / country,
            'proc' : args.processed_dir / country,
            'feat' : args.features_dir / country,
            'fig'  : FIG_DIR

        }
        for dir in dirs[country].keys():
            if args.verbose: print("creating", dirs[country][dir])
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

BASELINE_FEATURES = [
    'HomeElo', 'AwayElo', 'elo_diff',
    'home_goals_scored_L5', 'away_goals_scored_L5',
    'home_goals_conceded_L5', 'away_goals_conceded_L5',
    'home_ppg_L5', 'away_ppg_L5', 'form_diff_L5',
    'odds_home_prob_norm', 'odds_away_prob_norm'
]

EXTENDED_FEATURES = BASELINE_FEATURES + [
    'home_shots_L5', 'away_shots_L5',
    'home_shot_accuracy_L5', 'away_shot_accuracy_L5',
    'home_gd_L5', 'away_gd_L5'
]
DRAW_FEATURES = EXTENDED_FEATURES + [
    'draw_prob_consensus', 'draw_prob_dispersion', 'under_2_5_prob', 'under_2_5_zscore',
    'abs_elo_diff', 'elo_diff_sq', 'low_elo_diff', 'medium_elo_diff', 'abs_odds_prob_diff',
    'abs_ahh', 'ahh_zero', 'ahh_flat', 'min_shots_l5', 'min_shot_acc_l5', 'min_goals_scored_l5',
    'home_draw_rate_l10', 'away_draw_rate_l10',
    'league_draw_bias'] 
TOP_DRAW_FEATURES = EXTENDED_FEATURES + [
    'draw_prob_consensus', 'abs_odds_prob_diff', 'under_2_5_prob', 
    'draw_prob_dispersion', 'abs_elo_diff', 'under_2_5_zscore', 'elo_diff_sq',
    'min_shot_acc_l5', 'min_shots_l5', 'abs_ahh'  
]
EXTENDED_LITE_FEATURES = BASELINE_FEATURES + [
    'draw_prob_consensus',
    'under_2_5_prob',
    'asian_handicap_diff',
    'draw_prob_dispersion',
    'odds_draw_prob_norm',
]
FAULTS_FEATURES = ["home_fouls_L5", "away_fouls_L5", "foul_diff_L5"]

DRAW_FAULT_FEATURES = EXTENDED_LITE_FEATURES + FAULTS_FEATURES
# Registry for cleaner lookup
FEATURE_SETS = {
    'baseline': BASELINE_FEATURES,
    'extended': EXTENDED_FEATURES,
    'draw_features': DRAW_FEATURES,
    'draw_optimized': TOP_DRAW_FEATURES,
    'draw_lite': EXTENDED_LITE_FEATURES,
    'draw_faults' : DRAW_FAULT_FEATURES
}

EXCLUDE_COLS = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 
                'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']

def select_features(df, feature_set="baseline"):
    """
    Select features for training.

    Args:
        df: DataFrame with features
        feature_set: Which feature set to use:

    Returns:
        List of feature column names
    """

    if feature_set == 'all':
            return [col for col in df.columns 
                    if col not in EXCLUDE_COLS and df[col].dtype in ['float64', 'int64']]
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
