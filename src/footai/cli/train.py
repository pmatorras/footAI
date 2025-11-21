"""
Training command handler for footAI.

Orchestrates model training across three modes:
- Multi-division: train one model on combined division data
- Multi-season: train per division across multiple seasons
- Single-season: train per season per division

Handles result logging, JSON export, and aggregate summaries.
"""


from footai.utils.paths import get_multiseason_path, get_season_paths, get_multicountry_model_path
from footai.data.loader import load_combined_features
from footai.ml.training import train_model
from footai.ml.evaluation import print_results_summary, write_metrics_json
from footai.utils.logger import log_training_run

def execute(countries, seasons, divisions, args, dirs):    
    args.stats = False if args.nostats else True
    if args.verbose: print("Training...")
    all_results={}
    if args.multi_countries:
        """Execute training with optional multi-country support."""

        features_csv = load_combined_features(countries, divisions, seasons, dirs, args)
        
        # Train single model on combined data
        multicountry_path = get_multicountry_model_path(countries, seasons, divisions, args.model, tier=args.tier)
        with log_training_run(countries, divisions, args.features_set, seasons, args.model, multidiv=args.multi_division, multicountry=True, tier=args.tier) as json_path:   
            results = train_model(features_csv, feature_set=args.features_set,save_model=multicountry_path,args=args)
            write_metrics_json(json_path, args.countries, divisions, args.features_set, results, seasons)

    else:
        for country in countries:
            all_results[country] = {}
            if args.multi_division:  
                combined_features = load_combined_features(country, divisions, seasons, dirs, args)
                with log_training_run(country, divisions[country], args.features_set, seasons, args.model, multidiv=True) as json_path:   
                    results = train_model(combined_features, feature_set=args.features_set, save_model=f"models/{country}/{country}_multidiv_{seasons[0]}_to{seasons[-1]}_{args.features_set}.pkl", args=args)
                    write_metrics_json(json_path, country, divisions[country], args.features_set, results, seasons)

            elif args.multi_season:
                for division in divisions[country]:
                    features_csv = get_multiseason_path(dirs[country]['feat'], division, seasons[0], seasons[-1], args)
                    with log_training_run(country, division, args.features_set, seasons, args.model) as json_path:
                        # Train baseline model
                        results = train_model( features_csv, feature_set=args.features_set, save_model=f"models/{country}/{args.model}_{division}_{seasons[0]}_to{seasons[-1]}_{args.features_set}.pkl",args=args)
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
                            results = train_model( paths['feat'], feature_set=args.feature_set, save_model=f"models/{country}/{season}_{division}_{args.feature_set}_rf.pkl",args=args)
                            all_results[country][season][division] = results['accuracy']
                            write_metrics_json(json_path, country, division, args.features_set, results, season)
                print_results_summary(all_results[country], divisions[country])
