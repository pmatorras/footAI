"""elo calculation command handler for footAI."""

import pandas as pd
from footai.utils.paths import get_season_paths
from footai.core.elo import calculate_elo_season, calculate_elo_multiseason

def execute(seasons, divisions, args, dirs):
    for country in args.countries:
        if args.multi_season:
            print("calculating multi season")
            calculate_elo_multiseason(seasons, divisions[country], country, dirs, decay_factor=args.decay_factor, initial_elo=1500, k_factor=32, args=args)
        else:
            for season in seasons:
                for division in divisions[country]:
                    paths = get_season_paths(country, season, division, dirs, args)
                    df = pd.read_csv(paths['raw'])
                    df = df[
                        df['HomeTeam'].notna() &
                        df['AwayTeam'].notna() &
                        (df['HomeTeam'].astype(str).str.strip().str.lower() != 'nan') &
                        (df['AwayTeam'].astype(str).str.strip().str.lower() != 'nan') &
                        (df['HomeTeam'].astype(str).str.strip() != '') &
                        (df['AwayTeam'].astype(str).str.strip() != '')
                    ].copy()
                    df_with_elos = calculate_elo_season(df)
                    df_with_elos.to_csv(paths['proc'], index=False)
                    print(f"{season} / {division} saved to {paths['proc']}")