'''Command-line interface setup.'''
import argparse
from footai.utils.config import RAW_DIR, PROCESSED_DIR, FEATURES_DIR, FEATURE_SETS
from footai.ml.models import MODEL_METADATA
from footai.utils.validators import ValidateDivisionAction, validate_decay_factor

def create_parser():
    '''Create and configure the argument parser.'''
    parser = argparse.ArgumentParser(
        prog='footAI',
        description='FootAI - Football analytics via Elo ratings', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            Examples:
            python download_football.py 2024                    # La Liga 2024-25, Division 1
            python download_football.py 2024 --division 2       # La Liga 2024-25, Division 2
            python download_football.py 2021 --country SP       # La Liga 2021-22
            python download_football.py 2024 --data-dir my_data # Save to custom directory
            '''
        )

    sub = parser.add_subparsers(dest='cmd', required=True)

    p_down = sub.add_parser('download', help='Download new data')
    p_promo = sub.add_parser('promotion-relegation', help='Identify promoted/relegated teams between seasons')
    p_elo = sub.add_parser('elo', help='Calculate ELO rankings')
    p_feat = sub.add_parser('features', help='Calculate feature analysis varialbes')
    p_plot = sub.add_parser('plot', help='Plot ELO rankings')
    p_train = sub.add_parser('train', help='Plot ELO rankings')
    p_train.add_argument('--nostats', action='store_true', help='Remove printout of relevant statistics.')

    for sp in (p_down, p_elo, p_feat, p_plot, p_promo,p_train):
        sp.add_argument( '--season-start', type=str, help='Season year (e.g., 2024 for 2024-25 season)', default='2024')
        sp.add_argument( '--division', '-div', action=ValidateDivisionAction, default=None, help='League division (default: First two tiers for given country)')
        sp.add_argument( '--countries', '--country', dest='countries', type=str, default='SP', help='Country code(s). Can take single entry: eg (default: SP for Spain/La Liga) or multiple ones (eg SP,IT or SP IT for both Spanish and italian data)')
        sp.add_argument( '--raw-dir', type=str, default=RAW_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--processed-dir', type=str, default=PROCESSED_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--features-dir', type=str, default=FEATURES_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--features-set', type=str, default='baseline', help='Set of features to train on. default baseline', choices=FEATURE_SETS.keys()) 
        sp.add_argument( '--model', type=str, default='rf', help='Model to run', choices=MODEL_METADATA.keys())       
        sp.add_argument('-ms', '--multi-season', action='store_true', help='Calculate over multiple seasons')
        sp.add_argument('-v', '--verbose', action='store_true', help='Verbose additional info')
        sp.add_argument( '--decay-factor', '-df', type=validate_decay_factor, help='Decay factor', default=0.95)
        sp.add_argument('--elo-transfer', action='store_true', help='Transfer ELO ratings from relegated to promoted teams')
        sp.add_argument('-md', '--multi-division', action='store_true', help='Train on multiple divisions (e.g., SP1+SP2).')
        sp.add_argument('-mc', '--multi-countries', action='store_true', help='Train on multiple countries (Eg SP+EN).')

    return parser

