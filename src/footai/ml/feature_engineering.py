"""
Feature Engineering Module for Football Match Prediction
=========================================================

This module creates rolling features from historical match data while 
maintaining temporal correctness (no data leakage).

Called from main.py via the 'feature-engineer' command.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def team_matches_rows(df, team_name):
    team_matches = []
    for idx, row in df.iterrows():
        if row['HomeTeam'] == team_name:
            team_matches.append({
                'date': row['Date'],
                'goals_scored': row['FTHG'],
                'goals_conceded': row['FTAG'],
                'shots': row['HS'],
                'shots_on_target': row['HST'],
                'result': row['FTR'],  # H/D/A
                'is_home': True
            })
        elif row['AwayTeam'] == team_name:
            team_matches.append({
                'date': row['Date'],
                'goals_scored': row['FTAG'],
                'goals_conceded': row['FTHG'],
                'shots': row['AS'],
                'shots_on_target': row['AST'],
                'result': 'W' if row['FTR'] == 'A' else ('D' if row['FTR'] == 'D' else 'L'),
                'is_home': False
            })
    return team_matches

def calculate_team_rolling_features(df: pd.DataFrame, team_name: str, window: int, cache: Dict, initial_history: Dict = None) -> Dict:
    """
    Calculate rolling features for a specific team.

    Args:
        df: DataFrame with all match data (sorted by date)
        team_name: Name of the team
        window: Rolling window size (e.g., 3 or 5 matches)
        cache: Cache dictionary to store computed features

    Returns:
        Dict mapping match_date -> feature dict
    """
    cache_key = f"{team_name}_{window}"
    if cache_key in cache:
        return cache[cache_key]

    # Extract team's matches
    team_matches = team_matches_rows(df, team_name=team_name)
    if initial_history and team_name in initial_history:
        team_matches = initial_history[team_name] + team_matches
    team_df = pd.DataFrame(team_matches).sort_values('date').reset_index(drop=True)

    # Calculate rolling features
    features = {}
    for i in range(len(team_df)):
        match_date = team_df.iloc[i]['date']

        if i < 1:
            # First match - no history
            features[match_date] = {
                f'goals_scored_L{window}': np.nan,
                f'goals_conceded_L{window}': np.nan,
                f'ppg_L{window}': np.nan,
                f'shots_L{window}': np.nan,
                f'shot_accuracy_L{window}': np.nan,
            }
        else:
            # Use only PREVIOUS matches (i-window to i-1)
            prev_matches = team_df.iloc[max(0, i-window):i]

            # Calculate points
            points = prev_matches['result'].apply(
                lambda x: 3 if x == 'W' else (1 if x == 'D' else 0)
            )

            # Shot accuracy
            total_shots = prev_matches['shots'].sum()
            shot_acc = (prev_matches['shots_on_target'].sum() / total_shots * 100) if total_shots > 0 else 0

            features[match_date] = {
                f'goals_scored_L{window}': prev_matches['goals_scored'].mean(),
                f'goals_conceded_L{window}': prev_matches['goals_conceded'].mean(),
                f'ppg_L{window}': points.mean(),
                f'shots_L{window}': prev_matches['shots'].mean(),
                f'shot_accuracy_L{window}': shot_acc,
            }

    cache[cache_key] = features
    return features


def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add match-level derived features.

    Args:
        df: DataFrame with team-level features

    Returns:
        DataFrame with additional match-level features
    """
    # Elo difference
    df['elo_diff'] = df['HomeElo'] - df['AwayElo']

    # Form difference (using last 5 matches)
    if 'home_ppg_L5' in df.columns and 'away_ppg_L5' in df.columns:
        df['form_diff_L5'] = df['home_ppg_L5'] - df['away_ppg_L5']

    # Goal difference trends
    if 'home_goals_scored_L5' in df.columns:
        df['home_gd_L5'] = df['home_goals_scored_L5'] - df['home_goals_conceded_L5']
        df['away_gd_L5'] = df['away_goals_scored_L5'] - df['away_goals_conceded_L5']

    # Home advantage indicator (always 1 for home team)
    df['is_home'] = 1

    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert betting odds to implied probabilities.

    Args:
        df: DataFrame with betting odds

    Returns:
        DataFrame with odds-based features
    """
    # Bet365 odds (H/D/A)
    if 'B365H' in df.columns and 'B365D' in df.columns and 'B365A' in df.columns:
        # Convert odds to implied probability (1/odds)
        df['odds_home_prob'] = 1 / df['B365H']
        df['odds_draw_prob'] = 1 / df['B365D']
        df['odds_away_prob'] = 1 / df['B365A']

        # Normalize probabilities (remove bookmaker margin)
        total_prob = df['odds_home_prob'] + df['odds_draw_prob'] + df['odds_away_prob']
        df['odds_home_prob_norm'] = df['odds_home_prob'] / total_prob
        df['odds_away_prob_norm'] = df['odds_away_prob'] / total_prob

        # Odds vs Elo disagreement (if Elo expected exists)
        if 'HomeExpected' in df.columns:
            df['odds_elo_diff'] = df['odds_home_prob_norm'] - df['HomeExpected']

    return df


def engineer_features(df: pd.DataFrame, window_sizes: List[int] = [3, 5], team_initial_history=None, verbose: bool = False) -> pd.DataFrame:
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
            f'away_goals_scored_L{window}',
            f'away_goals_conceded_L{window}',
            f'away_ppg_L{window}',
            f'away_shots_L{window}',
            f'away_shot_accuracy_L{window}',
        ]

        for col in feature_cols:
            enriched_df[col] = np.nan

        # Process each match
        for idx, row in enriched_df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['Date']

            # Get features for both teams
            home_features = calculate_team_rolling_features(enriched_df, home_team, window, team_cache, initial_history=team_initial_history)
            away_features = calculate_team_rolling_features(enriched_df, away_team, window, team_cache, initial_history=team_initial_history)

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



def extract_final_match_histories(df: pd.DataFrame, window_sizes: List[int]) -> Dict:
    """
    Extract the last N matches for each team to seed next season's rolling features.
    
    Returns:
        Dict[team_name, List[match_dicts]] with last max(window_sizes) matches per team
    """
    max_window = max(window_sizes)
    team_histories = {}
    
    for team in df['HomeTeam'].unique():
        # Get all matches for this team
        team_matches = team_matches_rows(df, team_name=team)
        team_histories[team] = team_matches[-max_window:] if len(team_matches) >= max_window else team_matches
    
    return team_histories

def engineer_multiseason_features( seasons: List[str], divisions: List[str], 
    dirs: Dict, window_sizes: List[int] = [3, 5], args = None) -> None:
    """
    Engineer features across multiple seasons with temporal continuity.
    
    Args:
        seasons: List of season codes
        divisions: List of division names
        carry_stats: Whether to carry team rolling stats between seasons
        ...
    """
    
    """
    Engineer features across multiple seasons and save one combined file per division.
    
    This processes seasons sequentially with rolling state carry-over between them,
    then concatenates all seasons into a single training-ready DataFrame.
    """
    from footai.core.config import get_season_paths
    import pandas as pd
    
    for division in divisions:
        all_season_dfs = []
        team_match_history = {}  # carry match history between seasons
        
        for season_idx, season in enumerate(seasons):
            paths = get_season_paths(season, division, dirs, args)
            df = pd.read_csv(paths['proc'])
            
            if args.verbose:
                print(f"Processing {season}/{division}...")
            
            # Engineer features (with history from previous season if available)
            enriched_df = engineer_features( df, window_sizes=window_sizes, team_initial_history=team_match_history if season_idx > 0 else None,verbose=args.verbose)
            
            # Add season identifier
            enriched_df['Season'] = season
            all_season_dfs.append(enriched_df)
            
            # Extract final match history for each team to carry into next season
            team_match_history = extract_final_match_histories( enriched_df, window_sizes=window_sizes)
        
        # Concatenate all seasons
        combined_df = pd.concat(all_season_dfs, ignore_index=True)
        
        # Save single multi-season file
        output_path = dirs['feat'] / f"{division}_{seasons[0]}_to_{seasons[-1]}.csv"
        combined_df.to_csv(output_path, index=False)
        
        if args.verbose:
            print(f"Saved {len(seasons)} seasons to {output_path}")
            print(f"Total matches: {len(combined_df)}")
