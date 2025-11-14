"""Command-line interface setup."""
import argparse
from footai.utils.config import COUNTRIES, RAW_DIR, PROCESSED_DIR, FEATURES_DIR
from footai.utils.validators import ValidateDivisionAction, validate_decay_factor

def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="footAI",
        description='FootAI - Football analytics via Elo ratings', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python download_football.py 2024                    # La Liga 2024-25, Division 1
            python download_football.py 2024 --division 2       # La Liga 2024-25, Division 2
            python download_football.py 2021 --country SP       # La Liga 2021-22
            python download_football.py 2024 --data-dir my_data # Save to custom directory
            """
        )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_down = sub.add_parser("download", help="Download new data")
    p_promo = sub.add_parser('promotion-relegation', help="Identify promoted/relegated teams between seasons")
    p_elo = sub.add_parser('elo', help="Calculate ELO rankings")
    p_feat = sub.add_parser('features', help="Calculate feature analysis varialbes")
    p_plot = sub.add_parser("plot", help="Plot ELO rankings")
    p_train = sub.add_parser("train", help="Plot ELO rankings")
    p_train.add_argument('--nostats', action='store_true', help='Remove printout of relevant statistics.')

    for sp in (p_down, p_elo, p_feat, p_plot, p_promo,p_train):
        sp.add_argument( '--season-start', type=str, help='Season year (e.g., 2024 for 2024-25 season)', default="2024")
        sp.add_argument( '--division', '-div', action=ValidateDivisionAction, default=["SP1"], help='League division (default: SP1)')
        sp.add_argument( '--country', type=str, default='SP', help='Country code (default: SP for Spain/La Liga)', choices=COUNTRIES.keys())
        sp.add_argument( '--raw-dir', type=str, default=RAW_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--processed-dir', type=str, default=PROCESSED_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--features-dir', type=str, default=FEATURES_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--features-set', type=str, default='baseline', help='Set of features to train on. default baseline', choices=["baseline", "extended", "draw_optimized", "all"]) 
        sp.add_argument( '--model', type=str, default='rf', help='Model to run', choices=["rf", "rf_cal", "gb"])       
        sp.add_argument("-m", "--multi-season", action="store_true", help="Calculate over multiple seasons")
        sp.add_argument("-v", "--verbose", action="store_true", help="Verbose additional info")
        sp.add_argument( '--decay-factor', '-df', type=validate_decay_factor, help='Decay factor', default=0.95)
        sp.add_argument("--elo-transfer", action="store_true", help="Transfer ELO ratings from relegated to promoted teams")
        sp.add_argument('--multi-division', action='store_true', help='Train on multiple divisions (e.g., SP1+SP2). Requires --div to specify both divisions.')

    return parser

