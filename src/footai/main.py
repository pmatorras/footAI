"""Main execution logic for footAI commands."""
from footai.cli.parser import create_parser
from footai.utils.config import setup_directories

from footai.utils.paths import parse_start_years
from footai.cli import download, promotion, elo, features, train, plot

def main():
    #Parse common parameters
    parser = create_parser()
    args = parser.parse_args()
    if args.elo_transfer or args.multi_division: args.multi_season=True
    if args.verbose: print("Running the code with args:", args)

    divisions = args.division
    seasons = parse_start_years(args.season_start)
    dirs = setup_directories(args)

        # Command registry
    commands = {
        'download': download.execute,
        'promotion-relegation': promotion.execute,
        'elo': elo.execute,
        'features': features.execute,
        'train': train.execute,
        'plot': plot.execute,
    }
    
    handler = commands.get(args.cmd)
    if handler:
        handler(seasons, divisions, args, dirs)
    else:
        print(f"Unknown command: {args.cmd}")
        return 1

    if args.verbose: print('Code finished running')
