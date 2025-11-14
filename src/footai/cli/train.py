"""
Training command handler for footAI.

Orchestrates model training across three modes:
- Multi-division: train one model on combined division data
- Multi-season: train per division across multiple seasons
- Single-season: train per season per division

Handles result logging, JSON export, and aggregate summaries.
"""


from footai.utils.paths import get_multiseason_path, get_season_paths
from footai.ml.training import train_baseline_model
from footai.ml.feature_engineering import combine_divisions_features
from footai.ml.evaluation import print_results_summary

def execute(seasons, divisions, args, dirs):    
    args.stats = False if args.nostats else True
    if args.verbose: print("Training...")
    all_results={}
    if args.multi_division:  
        combined_features = combine_divisions_features(args.country, divisions, seasons, dirs, args)
        print("\n" + "="*70)
        print(f"TRAINING for {divisions} ({seasons})")
        print("="*70)          
        results = train_baseline_model(combined_features, feature_set=args.features_set, test_size=0.2, args=args)            
    elif args.multi_season:
        for division in divisions:
            features_csv = get_multiseason_path(dirs['feat'], division, seasons[0], seasons[-1], args)
            print("\n" + "="*70)
            print(f"TRAINING for {division} ({seasons})")
            print("="*70)
            # Train baseline model
            results = train_baseline_model( features_csv, feature_set=args.features_set, test_size=0.2, save_model=f"models/{seasons[0]}_to{seasons[-1]}_{division}_{args.features_set}_rf.pkl",args=args)
            all_results[division] = results['accuracy']

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

        print_results_summary(all_results, divisions)
    """
    Execute the train command.
    
    Args:
        args: Parsed CLI arguments (Namespace)
        dirs: Dictionary of directory paths (raw, processed, features, models)
    """