# data/loader.py
import pandas as pd
from pathlib import Path
from footai.utils.paths import get_multiseason_path, get_season_paths

def load_combined_features(countries, divisions, seasons, dirs, args):
    """
    Load and combine features from multiple divisions/countries.
    
    Args:
        countries: List of country codes (e.g., ['SP'] or ['SP', 'IT'])
        divisions: Dict mapping country -> list of divisions
                  e.g., {'SP': ['SP1', 'SP2'], 'IT': ['I1', 'I2']}
        seasons: List of season codes
        dirs: Directory structure
        args: Command-line arguments
    
    Returns:
        Path to combined CSV file (or in-memory DataFrame)
    """
    dfs = []
    if isinstance(countries, str): countries = [countries]
    for country in countries:
        country_divisions = divisions.get(country, [])
        
        if not country_divisions:
            print(f"Warning: No divisions specified for {country}, skipping")
            continue
        
        for division in country_divisions:
            # Load features for this division
            if args.multi_season:
                feat_path = get_multiseason_path(
                    dirs[country]['feat'], 
                    division, 
                    seasons[0], 
                    seasons[-1], 
                    args
                )
            else:
                # Concatenate all seasons for this division
                season_dfs = []
                for season in seasons:
                    paths = get_season_paths(country, season, division, dirs, args)
                    if Path(paths['feat']).exists():
                        season_dfs.append(pd.read_csv(paths['feat']))
                
                if not season_dfs:
                    print(f"Warning: No season data for {country}/{division}")
                    continue
                
                feat_df = pd.concat(season_dfs, ignore_index=True)
                # Save temporary combined file
                temp_dir = Path(f"data/features/{country}")
                temp_dir.mkdir(parents=True, exist_ok=True)
                feat_path = temp_dir / f"{division}_{seasons[0]}_to_{seasons[-1]}_combined.csv"
                feat_df.to_csv(feat_path, index=False)
            
            if not Path(feat_path).exists():
                print(f"Warning: {feat_path} not found, skipping {country}/{division}")
                continue
            
            df = pd.read_csv(feat_path)
            
            # Add metadata columns
            df['Country'] = country
            if 'Division' not in df.columns:
                df['Division'] = division
            
            dfs.append(df)
            print(f"Loaded {len(df)} matches from {country}/{division}")
    
    if not dfs:
        raise ValueError(f"No feature files found for specified countries/divisions")
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by date to maintain temporal order for CV
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\n  Combined dataset: {len(combined_df)} matches from {len(countries)} countries")
    print(f"  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"  Divisions: {combined_df['Division'].unique().tolist()}")
    
    # Save to temporary file
    is_multicountry = len(countries) > 1
    
    if is_multicountry:
        # Multi-country: SP_IT_multicountry_1516_to_2526.csv
        country_str = '_'.join(countries)
        temp_path = Path(f"data/temp/{country_str}_multicountry_{seasons[0]}_to_{seasons[-1]}_{args.features_set}.csv")
    else:
        # Multi-division (single country): SP_multidiv_1516_to_2526.csv
        country = countries[0]
        temp_path = Path(f"data/temp/{country}_multidiv_{seasons[0]}_to_{seasons[-1]}_{args.features_set}.csv")
    
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(temp_path, index=False)
    
    return str(temp_path)
