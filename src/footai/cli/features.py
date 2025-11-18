"""Command handler to for the footAI feature engineering."""


import pandas as pd
from footai.utils.paths import get_season_paths, get_multiseason_path
from footai.ml.feature_engineering import engineer_features, save_features


def execute(countries, seasons, divisions, args, dirs):
    for country in countries:
        if args.multi_season:
            for division in divisions[country]:
                elo_dir = get_multiseason_path(dirs[country]['proc'], division, seasons[0], seasons[-1], args)
                df = pd.read_csv(elo_dir)
                enriched_df = engineer_features(df, window_sizes=[3, 5], verbose=True)
                proc_dir = get_multiseason_path(dirs[country]['feat'], division, seasons[0], seasons[-1], args)
                save_features(enriched_df, proc_dir, verbose=True)

        else:
            for season in seasons:
                for division in divisions[country]:
                    paths = get_season_paths(country, season, division, dirs, args)
                    # Load your elo-enriched data
                    df = pd.read_csv(paths['proc'])

                    # Engineer features
                    enriched_df = engineer_features(df, window_sizes=[3, 5], verbose=True)
                    # Save
                    save_features(enriched_df, paths['feat'], verbose=True)