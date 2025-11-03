import pandas as pd
import argparse
from calculate_elo import calculate_elo_ratings
from download_data import download_football_data
from common import get_data_loc, season_to_season_str, DATA_DIR, RAW_DIR, PROCESSED_DIR, COUNTRIES
from plot_elo import plot_elo_rankings

class ValidateDivisionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        country = namespace.country
        print(COUNTRIES.keys())
        if values not in COUNTRIES[country]["divisions"]:
            valid = ', '.join(COUNTRIES[country]["divisions"].keys())
            parser.error(f"Invalid division '{values}' for {country}. Choose from: {valid}")
        setattr(namespace, self.dest, values)


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

    parser = argparse.ArgumentParser(prog="financial_ml", description="S&P 500 data pipeline: fetch, fundamentals, train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_down = sub.add_parser("download", help="Download new data")
    p_elo = sub.add_parser("elo", help="Calculate ELO rankings")
    p_plot = sub.add_parser("plot", help="Plot ELO rankings")


    for sp in (p_down, p_elo, p_plot):
        sp.add_argument( '--season', type=int, help='Season year (e.g., 2024 for 2024-25 season)', default=2024)
        sp.add_argument( '--division', '-div', action=ValidateDivisionAction, default="SP1", help='League division (default: SP1)')
        sp.add_argument( '--country', type=str, default='SP', help='Country code (default: SP for Spain/La Liga)', choices=COUNTRIES.keys())
        sp.add_argument( '--raw-dir', type=str, default=RAW_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument( '--processed-dir', type=str, default=PROCESSED_DIR, help='Directory to save CSV files (default: football_data)')
        sp.add_argument("-v", "--verbose", action="store_true", help="Verbose additional info")

    args = parser.parse_args()
    print("Running the code with args:", args)

    season_str = season_to_season_str(args.season, args.division, args.country)
    data_path = get_data_loc(season_str, args.division, args.country, args.raw_dir)
    elo_path = get_data_loc(season_str, args.division, args.country, args.processed_dir, elo=True)
    print(season_str, args.division, args.country)
    if args.cmd == "download":
        success, message = download_football_data(season_str=season_str, division=args.division, filepath=data_path)
        print(message)
    if args.cmd =="elo":
        # Load Input CSV and calculate elo
        df = pd.read_csv(data_path)
        team_history, df_with_elos = calculate_elo_ratings(df, initial_elo=1500, k_factor=32)

        # Save for training/plotting
        df_with_elos.to_csv(elo_path, index=False)

        if args.verbose:
            # View final standings by ELO
            final_elos = {team: history[-1][1] for team, history in team_history.items()}
            print("Final elos:")
            for team, elo in sorted(final_elos.items(), key=lambda x: x[1], reverse=True):
                print(f"{team}: {elo:.1f}")
    if args.cmd =="plot":
        print(args.country, args.division)
        fig = plot_elo_rankings(elo_path, division=args.division, custom_title=f"for {COUNTRIES[args.country]["divisions"][args.division]} ({COUNTRIES[args.country]["name"]})")
        fig.show()

if __name__ == "__main__":
    main()