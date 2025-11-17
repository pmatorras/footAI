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
from footai.ml.evaluation import print_results_summary, write_metrics_json
from footai.utils.logger import log_training_run

def execute(seasons, divisions, args, dirs):    
    args.stats = False if args.nostats else True
    if args.verbose: print("Training...")
    all_results={}
    args.multicountries=False
    if args.multicountries:
        print("no functionality yet")
    else:
        for country in args.countries:
            all_results[country] = {}
            if args.multi_division:  
                combined_features = combine_divisions_features(country, divisions[country], seasons, dirs[country], args)
                with log_training_run(country, divisions[country], args.features_set, seasons, args.model, multidiv=True) as json_path:   
                    results = train_baseline_model(combined_features, feature_set=args.features_set, test_size=0.2, save_model=f"models/{country}/{country}_multidiv_{seasons[0]}_to{seasons[-1]}_{args.features_set}.pkl", args=args)
                    write_metrics_json(json_path, country, divisions[country], args.features_set, results, seasons)

            elif args.multi_season:
                for division in divisions[country]:
                    features_csv = get_multiseason_path(dirs[country]['feat'], division, seasons[0], seasons[-1], args)
                    with log_training_run(country, division, args.features_set, seasons, args.model) as json_path:
                        # Train baseline model
                        results = train_baseline_model( features_csv, feature_set=args.features_set, test_size=0.2, save_model=f"models/{country}/{args.model}_{division}_{seasons[0]}_to{seasons[-1]}_{args.features_set}.pkl",args=args)
                        all_results[country][division] = results['accuracy']
                        print("="*70)
                        write_metrics_json(json_path, country, [division], args.features_set, results, seasons)

            else:
                for season in seasons:
                    all_results[country][season] = {}
                    for division in divisions[country]:
                        paths = get_season_paths(season, division, dirs[country], args)
                        with log_training_run(country, [division], args.features_set, [season]) as json_path:
                            # Train baseline model
                            results = train_baseline_model( paths['feat'], feature_set="baseline", test_size=0.2, save_model=f"models/{country}/{season}_{division}_baseline_rf.pkl",args=args)
                            all_results[country][season][division] = results['accuracy']
                            write_metrics_json(json_path, country, division, args.features_set, results, season)
                print_results_summary(all_results[country], divisions[country])
