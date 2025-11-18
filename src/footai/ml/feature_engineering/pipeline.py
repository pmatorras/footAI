"""
Feature Engineering Pipeline
=============================

Main entry point for feature generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import  List
from footai.utils.paths import get_multiseason_path
from footai.ml.feature_engineering.rolling import calculate_team_rolling_features
from footai.ml.feature_engineering.builders import (
    add_match_features, 
    add_odds_features, 
    add_draw_features, 
    add_league_features
)


def engineer_features(df: pd.DataFrame, window_sizes: List[int] = [3, 5], verbose: bool = False) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Args:
        df: DataFrame with match data (must have Date column and Elo ratings)
        window_sizes: Rolling window sizes for features (default: [3, 5])
        verbose: Whether to print progress

    Returns:
        DataFrame enriched with all engineered features
    """
    if verbose:
        print("Starting feature engineering...")

    # Prepare data
    df = df[
        df['HomeTeam'].notna() &
        df['AwayTeam'].notna() &
        (df['HomeTeam'].astype(str).str.strip().str.lower() != 'nan') &
        (df['AwayTeam'].astype(str).str.strip().str.lower() != 'nan') &
        (df['HomeTeam'].astype(str).str.strip() != '') &
        (df['AwayTeam'].astype(str).str.strip() != '')
    ].copy()
    enriched_df = df.copy()
    enriched_df['Date'] = pd.to_datetime(enriched_df['Date'])
    enriched_df = enriched_df.sort_values('Date').reset_index(drop=True)

    # Cache for team features
    team_cache = {}

    # For each window size, add rolling features
    for window in window_sizes:
        if verbose:
            print(f"Processing window size: {window}")

        # Initialize columns
        feature_cols = [
            f'home_goals_scored_L{window}',
            f'home_goals_conceded_L{window}',
            f'home_ppg_L{window}',
            f'home_shots_L{window}',
            f'home_shot_accuracy_L{window}',
            f'home_fouls_L{window}',
            f'away_goals_scored_L{window}',
            f'away_goals_conceded_L{window}',
            f'away_ppg_L{window}',
            f'away_shots_L{window}',
            f'away_shot_accuracy_L{window}',
            f'away_fouls_L{window}',
        ]

        for col in feature_cols:
            enriched_df[col] = np.nan

        # Process each match
        for idx, row in enriched_df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['Date']

            # Get features for both teams
            home_features = calculate_team_rolling_features(enriched_df, home_team, window, team_cache)
            away_features = calculate_team_rolling_features(enriched_df, away_team, window, team_cache)

            # Get features up to this match
            if match_date in home_features:
                for feat_name, feat_value in home_features[match_date].items():
                    enriched_df.at[idx, f'home_{feat_name}'] = feat_value

            if match_date in away_features:
                for feat_name, feat_value in away_features[match_date].items():
                    enriched_df.at[idx, f'away_{feat_name}'] = feat_value

    # Add match-level features
    if verbose:
        print("Adding match-level features...")
    enriched_df = add_match_features(enriched_df)

    # Add betting odds features
    if verbose:
        print("Adding betting market features...")
    enriched_df = add_odds_features(enriched_df)

    # Add draw-optimized features
    if verbose:
        print("Adding draw-optimized features...")
    enriched_df = add_draw_features(enriched_df)


    # Add league specific features
    if verbose:
        print("Adding league specific features...")
    enriched_df = add_league_features(enriched_df)

    if verbose:
        print(f"Feature engineering complete!")
        print(f"Total columns: {len(enriched_df.columns)}")

    return enriched_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of engineered feature columns (excluding metadata and raw odds).

    Args:
        df: DataFrame with engineered features

    Returns:
        List of feature column names
    """
    # Define columns to exclude
    exclude_cols = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 
                   'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']

    # Get all columns that aren't metadata or raw betting odds
    feature_cols = [col for col in df.columns if col not in exclude_cols 
                   and not col.startswith('B365') 
                   and not col.startswith('BW')
                   and not col.startswith('BF')
                   and not col.startswith('PS')
                   and not col.startswith('WH')
                   and not col.startswith('1XB')
                   and not col.startswith('Max')
                   and not col.startswith('Avg')
                   and not col.startswith('AH')
                   and not col.startswith('P>')
                   and not col.startswith('P<')]

    # Add back our calculated odds features
    feature_cols.extend([col for col in df.columns if col.startswith('odds_')])

    # Remove duplicates
    feature_cols = list(dict.fromkeys(feature_cols))

    return feature_cols


def save_features(df: pd.DataFrame, output_path: str, verbose: bool = False):
    """
    Save engineered features to CSV.

    Args:
        output_path: Path to save the CSV file
        verbose: Whether to print progress
    """
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved features to: {output_path}")
        print(f"Shape: {df.shape}")



# src/footai/ml/feature_engineering.py

def combine_divisions_features(country, divisions, seasons, dirs, args):
    """
    Combine feature files from multiple divisions into single training dataset.
    
    Args:
        country: Country code (e.g., 'SP')
        divisions: List of divisions (e.g., ['SP1', 'SP2'])
        seasons: List of season codes
        dirs: Directory paths dict
        args: Command line arguments
    
    Returns:
        Path to combined features file
    """
    
    all_dfs = []
    
    print(f"\nCombining features from {len(divisions)} divisions...")
    print("="*70)
    
    for division in divisions:
        feature_file = get_multiseason_path(dirs['feat'], division, seasons[0], seasons[-1], args)
        
        if not Path(feature_file).exists():
            print(f"Warning: {division} features not found at {feature_file}")
            print(f"Run: footai features --country {country} --div {division} --season-start {','.join(seasons)} --multiseason")
            continue
        
        df = pd.read_csv(feature_file)
        df['Division'] = division
        all_dfs.append(df)
        
        print(f"Loaded {len(df)} matches from {division}")
    if not all_dfs:
        raise ValueError(f"No feature files found for divisions: {divisions}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    
    # Add division features
    combined['is_tier1'] = (combined['Division'] == divisions[0]).astype(int)
    combined['division_tier'] = combined['Division'].map({
        div: idx + 1 for idx, div in enumerate(divisions)
    })
    
    # Save combined file
    suffix = '_transfer' if args.elo_transfer else '_multi'
    output_file = dirs['feat'] / f"{country}_multidiv_{seasons[0]}_to_{seasons[-1]}{suffix}.csv"
    combined.to_csv(output_file, index=False)
    
    print(f"\nCombined dataset:")
    print(f"  Total matches: {len(combined)}")
    print(f"  Divisions: {divisions}")
    print(f"  Output: {output_file}")
    
    return output_file
