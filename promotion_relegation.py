
import pandas as pd
from pathlib import Path
from common import get_season_paths, COUNTRIES

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
    paths_prev_tier1 = get_season_paths(prev_season, tier1_div, dirs, args)
    paths_prev_tier2 = get_season_paths(prev_season, tier2_div, dirs, args)
    
    df_prev_tier1 = pd.read_csv(paths_prev_tier1['raw'])
    df_prev_tier2 = pd.read_csv(paths_prev_tier2['raw'])
    
    teams_prev_tier1 = set(df_prev_tier1['HomeTeam'].unique())
    teams_prev_tier2 = set(df_prev_tier2['HomeTeam'].unique())
    
    # Load current season data
    paths_curr_tier1 = get_season_paths(season, tier1_div, dirs, args)
    paths_curr_tier2 = get_season_paths(season, tier2_div, dirs, args)
    
    df_curr_tier1 = pd.read_csv(paths_curr_tier1['raw'])
    df_curr_tier2 = pd.read_csv(paths_curr_tier2['raw'])
    
    teams_curr_tier1 = set(df_curr_tier1['HomeTeam'].unique())
    teams_curr_tier2 = set(df_curr_tier2['HomeTeam'].unique())
    
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
    
    results_df = pd.DataFrame(results)
    
    # Validation
    num_relegated = len(relegated_from_tier1)
    num_promoted = len(promoted_to_tier1)
    
    if num_relegated != num_promoted:
        print(f"Warning: {num_relegated} team relegated but {num_promoted} promoted from tier1!")
    
    print(f"Identified {num_relegated} relegated and {num_promoted} promoted teams\n")
    print(f"Relegated: {relegated_from_tier1}")
    print(f"Promoted: {promoted_to_tier1}\n")
    
    return results_df


def save_promotion_relegation(results_df, season, country, dirs):
    """Save promotion/relegation data to CSV."""
    output_path = dirs['proc'] / f"{country}_{season}_promotion_relegation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}\n")
    return output_path


def load_promotion_relegation(season, country, dirs):
    """Load previously computed promotion/relegation data."""
    path = dirs['proc'] / f"{country}_{season}_promotion_relegation.csv"
    if not path.exists():
        print(f"  File not found: {path}")
        return None
    return pd.read_csv(path)
