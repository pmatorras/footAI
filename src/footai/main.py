"""Main execution logic for footAI commands."""
import pandas as pd
from pathlib import Path
from footai.cli import create_parser
from footai.core.elo import calculate_elo_season, calculate_elo_multiseason
from footai.data.downloader import download_football_data
from footai.core.team_movements import identify_promotions_relegations_for_season, save_promotion_relegation
from footai.core.config import get_season_paths, get_multiseason_path, get_previous_season, FIG_DIR, COUNTRIES
from footai.viz.plotter import plot_elo_rankings

from footai.core.utils import parse_start_years
from footai.ml.training import train_baseline_model
from footai.ml.evaluation import print_results_summary
from footai.ml.feature_engineering import engineer_features, save_features

def setup_directories(args):
    '''Create dictionary with directories and ensure they exist'''
    dirs = {
        'raw'  : args.raw_dir,
        'proc' : args.processed_dir,
        'feat' : args.features_dir,
        'fig'  : FIG_DIR

    }
    for dir in dirs.keys():
        if args.verbose: print("creating", dirs[dir])
        Path(dirs[dir]).mkdir(parents=True, exist_ok=True)
    return dirs

def main():
    
    parser = create_parser()
    args = parser.parse_args()
    if args.elo_transfer: args.multiseason=True
    if args.verbose: print("Running the code with args:", args)

    divisions = args.division
    seasons = parse_start_years(args.season_start)
    dirs = setup_directories(args)

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

    elif args.cmd == 'features':
        if args.multiseason:
            for division in divisions:
                elo_dir = get_multiseason_path(dirs['proc'], division, seasons[0], seasons[-1], args)
                df = pd.read_csv(elo_dir)
                enriched_df = engineer_features(df, window_sizes=[3, 5], verbose=True)
                proc_dir = get_multiseason_path(dirs['feat'], division, seasons[0], seasons[-1], args)
                save_features(enriched_df, proc_dir, verbose=True)

        else:
            for season in seasons:
                for division in divisions:
                    paths = get_season_paths(season, division, dirs, args)
                    # Load your elo-enriched data
                    df = pd.read_csv(paths['proc'])

                    # Engineer features
                    enriched_df = engineer_features(df, window_sizes=[3, 5], verbose=True)
                    # Save
                    save_features(enriched_df, paths['feat'], verbose=True)

    elif args.cmd == "train":
        if args.verbose: print("Training...")
        all_results={}
        if args.multiseason:
            for division in divisions:
                features_csv = get_multiseason_path(dirs['feat'], division, seasons[0], seasons[-1], args)
                print("\n" + "="*70)
                print(f"TRAINING for {division} ({seasons})")
                print("="*70)
                # Train baseline model
                results = train_baseline_model( features_csv, feature_set=args.features_set, test_size=0.2, save_model=f"models/{seasons[0]}_to{seasons[-1]}_{division}_{args.features_set}_rf.pkl",args=args)
                all_results[division] = results['accuracy']

                print(f"\nFinal accuracy ({args.features_set}): {results['accuracy']*100:.1f}%")
                print(f"Features used: {len(results['feature_names'])}")
                print("="*70)         
        else:
            for season in seasons:
                all_results[season] = {}
                for division in divisions:
                    paths = get_season_paths(season, division, dirs, args)
                    features_csv = paths['feat']
                    print("\n" + "="*70)
                    print(f"TRAINING for {division} ({season})")
                    print("="*70)
                    # Train baseline model
                    results = train_baseline_model( features_csv, feature_set="baseline", test_size=0.2, save_model=f"models/{season}_{division}_baseline_rf.pkl",args=args)
                    all_results[season][division] = results['accuracy']

                    print(f"\nFinal accuracy: {results['accuracy']*100:.1f}%")
                    print(f"Features used: {len(results['feature_names'])}")
                    print("="*70)
            print(divisions)
            print_results_summary(all_results, divisions)

    elif args.cmd == "plot":
        for season in seasons:
            for division in divisions:
                paths = get_season_paths(season, division, dirs, args)
                fig = plot_elo_rankings(paths['proc'], division=division, custom_title=f"for {COUNTRIES[args.country]['divisions'][division]} ({COUNTRIES[args.country]["name"]}, season {season})")
                fig.write_html(paths['fig'])
                print(f"{season} / {division} saved to {paths['fig']}")

    if args.verbose: print('Code finished running')
if __name__ == "__main__":
    main()