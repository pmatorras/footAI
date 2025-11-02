import pandas as pd
import argparse
from calculate_elo import calculate_elo_ratings
from download_data import download_football_data
from common import get_data_loc, season_to_season_str
from plot_elo import plot_elo_rankings

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
        sp.add_argument( '--division', type=int, default=1, choices=[1, 2], help='League division (default: 1)')
        sp.add_argument( '--country', type=str, default='SP', help='Country code (default: SP for Spain/La Liga)')
        sp.add_argument( '--data-dir', type=str, default='football_data', help='Directory to save CSV files (default: football_data)')
        
    args = parser.parse_args()
    print("Running the code with args:", args)
    elo_csv = 'laliga_with_elo.csv'
    if args.cmd == "download":
        filepath, success, message = download_football_data(season=args.season, division=args.division, country=args.country, data_dir=args.data_dir)
        print(message)
        if success:
            return 0
        else:
            return 1
    if args.cmd =="elo":
        # Load your downloaded CSV
        season_str = season_to_season_str(args.season, args.division, args.country)
        print(season_str)
        data_path = get_data_loc(season_str, args.division, args.country, args.data_dir)
        df = pd.read_csv(data_path)

        # Calculate ELO
        team_history, df_with_elos = calculate_elo_ratings(df, initial_elo=1500, k_factor=32)

        # Save for training
        df_with_elos.to_csv(elo_csv, index=False)

        # View final standings by ELO
        final_elos = {team: history[-1][1] for team, history in team_history.items()}
        for team, elo in sorted(final_elos.items(), key=lambda x: x[1], reverse=True):
            print(f"{team}: {elo:.1f}")
    if args.cmd =="plot":
        fig = plot_elo_rankings(elo_csv, division='SP1')
        fig.show()

if __name__ == "__main__":
    main()