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
    #'home_draw_rate_l10', 'away_draw_rate_l10', #Currently broken
    'league_draw_bias'] 
TOP_DRAW_FEATURES = EXTENDED_FEATURES + [
    'draw_prob_consensus', 'abs_odds_prob_diff', 'under_2_5_prob', 
    'draw_prob_dispersion', 'abs_elo_diff', 'under_2_5_zscore', 'elo_diff_sq',
    'min_shot_acc_l5', 'min_shots_l5', 'abs_ahh'  
]

# Registry for cleaner lookup
FEATURE_SETS = {
    'baseline': BASELINE_FEATURES,
    'extended': EXTENDED_FEATURES,
    'draw_features': DRAW_FEATURES,
    'draw_optimized': TOP_DRAW_FEATURES,
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

def setup_directories(args):
    '''Create dictionary with directories and ensure they exist'''
    dirs = {
        'raw'  : args.raw_dir / args.country,
        'proc' : args.processed_dir / args.country,
        'feat' : args.features_dir / args.country,
        'fig'  : FIG_DIR

    }
    for dir in dirs.keys():
        if args.verbose: print("creating", dirs[dir])
        Path(dirs[dir]).mkdir(parents=True, exist_ok=True)
    return dirs




