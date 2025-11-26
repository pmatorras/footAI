import os
import pandas as pd
from footai.utils.config import COUNTRIES
from footai.utils.paths import get_season_paths, get_promotion_relegation_file

def identify_promotions_relegations_for_season(season, country, prev_season, dirs, args):
    """
    Identify promoted and relegated teams by comparing rosters between consecutive seasons.
    
    Args:
        season: current season (e.g., '23-24')
        prev_season: previous season (e.g., '22-23')
        country: country code (e.g., 'SP')
    
    Returns:
        pd.DataFrame with columns: [season, tier, team, status]
    """
    divisions = list(COUNTRIES[country]['divisions'].keys())
    tier1_div = divisions[0]  # e.g., 'SP1' for Spain
    tier2_div = divisions[1]  # e.g., 'SP2' for Spain
    
    # Load previous season data
    paths_prev_tier1 = get_season_paths(country, prev_season, tier1_div, dirs, args)
    paths_prev_tier2 = get_season_paths(country, prev_season, tier2_div, dirs, args)
    
    # Check if previous season files exist
    if not os.path.exists(paths_prev_tier1['raw']):
        print(f" WARNING: Previous season data not found: {paths_prev_tier1['raw']}")
        print(f"   Skipping promotion-relegation for season {season}")
        print(f"   To identify promotions/relegations, download season {prev_season} first:")
        print(f"   footai download --country {country} --div {tier1_div},{tier2_div} --season-start {prev_season}")
        return None
    
    if not os.path.exists(paths_prev_tier2['raw']):
        print(f" WARNING: Previous season data not found: {paths_prev_tier2['raw']}")
        print(f"   Skipping promotion-relegation for season {season}")
        return None
    
    # Load previous season data if exists
    df_prev_tier1 = pd.read_csv(paths_prev_tier1['raw'])
    df_prev_tier2 = pd.read_csv(paths_prev_tier2['raw'])
    
    teams_prev_tier1 = set(df_prev_tier1['HomeTeam'].dropna().unique())
    teams_prev_tier2 = set(df_prev_tier2['HomeTeam'].dropna().unique())
    
    # Load current season data
    paths_curr_tier1 = get_season_paths(country, season, tier1_div, dirs, args)
    paths_curr_tier2 = get_season_paths(country, season, tier2_div, dirs, args)
    
    df_curr_tier1 = pd.read_csv(paths_curr_tier1['raw'])
    df_curr_tier2 = pd.read_csv(paths_curr_tier2['raw'])
    
    teams_curr_tier1 = set(df_curr_tier1['HomeTeam'].dropna().unique())
    teams_curr_tier2 = set(df_curr_tier2['HomeTeam'].dropna().unique())
    
    # Identify promotions and relegations
    relegated_from_tier1 = teams_prev_tier1 - teams_curr_tier1
    promoted_to_tier1 = teams_curr_tier1 - teams_prev_tier1
    
    relegated_from_tier2 = teams_prev_tier2 - teams_curr_tier2
    promoted_to_tier2 = teams_curr_tier2 - teams_prev_tier2
    
    # Build results DataFrame
    results = []
    
    for team in relegated_from_tier1:
        results.append({
            'season': f"{prev_season}_{season}",
            'tier': 'tier1',
            'team': team,
            'status': 'relegated'
        })
    
    for team in promoted_to_tier1:
        results.append({
            'season': f"{prev_season}_{season}",
            'tier': 'tier1',
            'team': team,
            'status': 'promoted'
        })
    for team in relegated_from_tier2:
        results.append({
            'season': f"{prev_season}_{season}",
            'tier': 'tier2',
            'team': team,
            'status': 'relegated'
        })
    
    for team in promoted_to_tier2:
        results.append({
            'season': f"{prev_season}_{season}",
            'tier': 'tier2',
            'team': team,
            'status': 'promoted'
        })
       
    
    results_df = pd.DataFrame(results)
    
    # Validation
    num_relegated = len(relegated_from_tier1)
    num_promoted = len(promoted_to_tier1)
    
    if num_relegated != num_promoted:
        print(f"Warning: {num_relegated} team relegated but {num_promoted} promoted from tier1!")
    if args.verbose:
        print(f"Identified {num_relegated} relegated and {num_promoted} promoted teams\n")
        print(f"Relegated: {relegated_from_tier1}")
        print(f"Promoted: {promoted_to_tier1}\n")
    
    return results_df



def load_promotion_relegation(season, country, dirs):
    """Load previously computed promotion/relegation data."""
    path = get_promotion_relegation_file(dirs, country, season)
    if not path.exists():
        print(f"  File not found: {path}")
        return None
    df = pd.read_csv(path)
    return df.dropna(subset=['team'])

def save_promotion_relegation(results_df, season, country, dirs):
    """Save promotion/relegation data to CSV."""
    output_path = get_promotion_relegation_file(dirs, country, season)
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}\n")
    return output_path