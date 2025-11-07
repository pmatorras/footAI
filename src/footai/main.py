import pandas as pd
import argparse
from pathlib import Path
from footai.core.elo import calculate_elo_season, calculate_elo_multiseason
from footai.data.downloader import download_football_data
from footai.core.team_movements import identify_promotions_relegations_for_season, save_promotion_relegation
from footai.core.config import get_season_paths, year_to_season_code, get_previous_season, FIG_DIR, RAW_DIR, PROCESSED_DIR, COUNTRIES
from footai.viz.plotter import plot_elo_rankings

from footai.core.utils import parse_start_years
from footai.core.validators import ValidateDivisionAction, validate_decay_factor




def main():
    parser = argparse.ArgumentParser(
        description='Download football data and calculate elo rankings', 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python download_football.py 2024                    # La Liga 2024-25, Division 1
            python download_football.py 2024 --division 2       # La Liga 2024-25, Division 2
            python download_football.py 2021 --country SP       # La Liga 2021-22
            python download_football.py 2024 --data-dir my_data # Save to custom directory
            """
        )

    parser = argparse.ArgumentParser(prog="footAI", description="FootAI pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_down = sub.add_parser("download", help="Download new data")
    p_promo = sub.add_parser('promotion-relegation', help="Identify promoted/relegated teams between seasons")
    p_elo = sub.add_parser('elo', help="Calculate ELO rankings")
    p_plot = sub.add_parser("plot", help="Plot ELO rankings")


    for sp in (p_down, p_elo, p_plot, p_promo):
        sp.add_argument( '--season-start', type=str, help='Season year (e.g., 2024 for 2024-25 season)', default="2024")
        sp.add_argument( '--division', '-div', action=ValidateDivisionAction, default=["SP1"], help='League division (default: SP1)')
        sp.add_argument( '--country', type=str, default='SP', help='Country code (default: SP for Spain/La Liga)', choices=COUNTRIES.keys())
        sp.add_argument( '--raw-dir', type=str, default=RAW_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--processed-dir', type=str, default=PROCESSED_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument("-m", "--multiseason", action="store_true", help="Calculate over multiple seasons")
        sp.add_argument("-v", "--verbose", action="store_true", help="Verbose additional info")
        sp.add_argument( '--decay-factor', '-df', type=validate_decay_factor, help='Decay factor', default=0.95)
        sp.add_argument("--elo-transfer", action="store_true", help="Transfer ELO ratings from relegated to promoted teams")
    args = parser.parse_args()
    print("Running the code with args:", args)
    divisions = args.division
    seasons = parse_start_years(args.season_start)

    #Create dictionary with directories and ensure they exist
    dirs = {
        'raw'  : args.raw_dir,
        'proc' : args.processed_dir,
        'fig'  : FIG_DIR
    }
    for dir in dirs.keys():
        if args.verbose: print("creating", dirs[dir])
        Path(dirs[dir]).mkdir(parents=True, exist_ok=True)


    if args.cmd == "download":
        for season in seasons:
            for division in divisions:
                paths = get_season_paths(season, division, dirs, args)
                download_football_data(season, division, paths['raw'])

    elif args.cmd == "promotion-relegation":        
        for season in seasons:
            prev_season = get_previous_season(season)
            results = identify_promotions_relegations_for_season(season, args.country, prev_season, dirs, args)
            save_promotion_relegation(results, season, args.country, dirs)
            print(f"Saved promotion/relegation data for {prev_season} -> {season}") 
             
    elif args.cmd == 'elo':
        if args.multiseason:
            print("calculating multi season")
            calculate_elo_multiseason(seasons, divisions, args.country, dirs, decay_factor=args.decay_factor, initial_elo=1500, k_factor=32, args=args)
        else:
            for season in seasons:
                for division in divisions:
                    paths = get_season_paths(season, division, dirs, args)
                    df = pd.read_csv(paths['raw'])
                    df_with_elos = calculate_elo_season(df)
                    df_with_elos.to_csv(paths['proc'], index=False)
                    print(f"{season} / {division} saved to {paths['proc']}")

    elif args.cmd == "plot":
        for season in seasons:
            for division in divisions:
                paths = get_season_paths(season, division, dirs, args)
                fig = plot_elo_rankings(paths['proc'], division=division, custom_title=f"for {COUNTRIES[args.country]['divisions'][division]} ({COUNTRIES[args.country]["name"]}, season {season})")
                fig.write_html(paths['fig'])
                print(f"{season} / {division} saved to {paths['fig']}")

    print('Code finished running')
if __name__ == "__main__":
    main()