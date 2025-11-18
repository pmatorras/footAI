"""
Baseline Feature Engineering
============================

Core features: Add match, odds, draw and  features 
"""

import numpy as np
import pandas as pd

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

    if 'home_fouls_L3' in df.columns and 'away_fouls_L3' in df.columns:
        df['foul_diff_L3'] = df['home_fouls_L3'] - df['away_fouls_L3']
    
    if 'home_fouls_L5' in df.columns and 'away_fouls_L5' in df.columns:
        df['foul_diff_L5'] = df['home_fouls_L5'] - df['away_fouls_L5']
        
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

def add_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add draw-optimized features: odds consensus/dispersion, totals probs,
    parity indicators, low-event composites, and rolling draw rates.
    
    Args:
        df: DataFrame with odds and L5 features (post-engineer_features).
    
    Returns:
        DataFrame with added draw features.
    """
    # Odds-derived: Consensus draw prob and dispersion (across books; assumes odds are present)
    draw_odds_cols = ['B365D', 'BWD', 'WH D', 'IWD', 'PSD']  # Adjust if exact col names vary; use available D odds
    available_draw_odds = [col for col in draw_odds_cols if col in df.columns]
    if available_draw_odds:
        df['draw_prob_consensus'] = df[available_draw_odds].apply(lambda x: np.mean(1 / x), axis=1)
        df['draw_prob_dispersion'] = df[available_draw_odds].apply(lambda x: np.std(1 / x), axis=1)
    else:
        df['draw_prob_consensus'] = 1 / df['B365D'] if 'B365D' in df.columns else np.nan
        df['draw_prob_dispersion'] = np.nan
    
    # Under 2.5 prob (implied from totals odds; use B365 as primary)
    if 'B365>2.5' in df.columns and 'B365<2.5' in df.columns:
        df['under_2_5_prob'] = 1 / (1 + df['B365>2.5'] / df['B365<2.5'])
        under_mean = df['under_2_5_prob'].mean()
        under_std = df['under_2_5_prob'].std()
        df['under_2_5_zscore'] = (df['under_2_5_prob'] - under_mean) / under_std if under_std > 0 else 0
    else:
        df['under_2_5_prob'] = np.nan
        df['under_2_5_zscore'] = np.nan
    
    # Parity: Elo and odds closeness (using existing elo_diff and odds norms)
    df['abs_elo_diff'] = np.abs(df['elo_diff'])  # Assumes elo_diff already added
    df['elo_diff_sq'] = df['elo_diff'] ** 2
    df['low_elo_diff'] = (np.abs(df['elo_diff']) < 25).astype(int)
    df['medium_elo_diff'] = ((np.abs(df['elo_diff']) >= 25) & (np.abs(df['elo_diff']) < 50)).astype(int)
    df['abs_odds_prob_diff'] = np.abs(df['odds_home_prob_norm'] - df['odds_away_prob_norm'])
    
    # Asian handicap parity (using AHh if present)
    if 'AHh' in df.columns:
        df['abs_ahh'] = np.abs(df['AHh'])
        df['ahh_zero'] = (np.abs(df['AHh']) < 0.25).astype(int)
        df['ahh_flat'] = (df['AHh'] == 0).astype(int)
    else:
        df['abs_ahh'] = np.nan
        df['ahh_zero'] = np.nan
        df['ahh_flat'] = np.nan
    
    # Low-scoring composites (min of L5; assumes L5 cols exist from engineer_features)
    if all(col in df.columns for col in ['home_shots_L5', 'away_shots_L5']):
        df['min_shots_l5'] = np.minimum(df['home_shots_L5'], df['away_shots_L5'])
    else:
        df['min_shots_l5'] = np.nan
    if all(col in df.columns for col in ['home_shot_accuracy_L5', 'away_shot_accuracy_L5']):
        df['min_shot_acc_l5'] = np.minimum(df['home_shot_accuracy_L5'], df['away_shot_accuracy_L5'])
    else:
        df['min_shot_acc_l5'] = np.nan
    if all(col in df.columns for col in ['home_goals_scored_L5', 'away_goals_scored_L5']):
        df['min_goals_scored_l5'] = np.minimum(df['home_goals_scored_L5'], df['away_goals_scored_L5'])
    else:
        df['min_goals_scored_l5'] = np.nan
    
    # Team/League draw priors (rolling draw rate L10; compute post-sort for temporality)
    # Add row_id to track original positions
    df_sorted = df.sort_values('Date').copy()
    df_sorted['row_id'] = df.index  # Matches original df index for merging back

    # Map FTR to numeric: 1 if 'D', 0 otherwise (binary for draw rate); check if FTR exists/is string
    if 'FTR' not in df_sorted.columns:
        print("Warning: 'FTR' column missing; skipping draw rates.")
        df['home_draw_rate_l10'] = np.nan
        df['away_draw_rate_l10'] = np.nan
    else:
        # Binary: 1 if draw, 0 otherwise
        is_draw = (df_sorted['FTR'] == 'D').astype(float)
        
        # Calculate draw rates for each team type
        df_sorted['home_draw_rate_l10'] = np.nan
        df_sorted['away_draw_rate_l10'] = np.nan

 
        # Process home teams
        if 'HomeTeam' in df_sorted.columns:
            for team in df_sorted['HomeTeam'].unique():
                team_mask = df_sorted['HomeTeam'] == team
                team_indices = df_sorted[team_mask].index
                
                # Calculate rolling mean (past 10 matches only - excluding current)
                team_draws = is_draw[team_mask]
                rolling_rate = team_draws.shift(1).rolling(window=10, min_periods=3).mean()
                
                # Assign back to dataframe
                df_sorted.loc[team_indices, 'home_draw_rate_l10'] = rolling_rate.values
        
        # Process away teams
        if 'AwayTeam' in df_sorted.columns:
            for team in df_sorted['AwayTeam'].unique():
                team_mask = df_sorted['AwayTeam'] == team
                team_indices = df_sorted[team_mask].index
                
                # Calculate rolling mean (past 10 matches only - excluding current)
                team_draws = is_draw[team_mask]
                rolling_rate = team_draws.shift(1).rolling(window=10, min_periods=3).mean()
                
                # Assign back to dataframe
                df_sorted.loc[team_indices, 'away_draw_rate_l10'] = rolling_rate.values
        
        # Merge back to original df (preserve original order)
        df = df.merge(
            df_sorted[['Date', 'HomeTeam', 'AwayTeam', 'home_draw_rate_l10', 'away_draw_rate_l10']],
            on=['Date', 'HomeTeam', 'AwayTeam'],
            how='left',
            suffixes=('', '_new')
        )
        
        # Use the new columns (drop old if exists)
        if 'home_draw_rate_l10_new' in df.columns:
            df['home_draw_rate_l10'] = df['home_draw_rate_l10_new']
            df = df.drop('home_draw_rate_l10_new', axis=1)
        if 'away_draw_rate_l10_new' in df.columns:
            df['away_draw_rate_l10'] = df['away_draw_rate_l10_new']
            df = df.drop('away_draw_rate_l10_new', axis=1)
    
    # League draw bias (per-division if Division column exists, else global)
    if 'FTR' in df.columns:
        if 'Division' in df.columns and df['Division'].nunique() > 1:
            # Multi-league: per-division draw rate
            league_draw_rates = df.groupby('Division')['FTR'].apply(lambda x: (x == 'D').mean())
            df['league_draw_bias'] = df['Division'].map(league_draw_rates)
        else:
            # Single-league: global draw rate (same for all rows)
            df['league_draw_bias'] = float((df['FTR'] == 'D').mean())
    else:
        df['league_draw_bias'] = np.nan
    
    return df

def add_league_features(df):
    """Add league-specific contextual features for pooled models."""
    
    # League-level aggregates (historical draw rates, etc.)
    league_stats = df.groupby('Division').agg({
        'FTR': lambda x: (x == 'D').mean(),  # draw_rate
        'FTHG': 'mean',  # avg_goals_home
        'FTAG': 'mean',  # avg_goals_away
    }).rename(columns={'FTR': 'league_draw_rate', 
                       'FTHG': 'league_avg_goals_home',
                       'FTAG': 'league_avg_goals_away'})
    
    df = df.merge(league_stats, left_on='Division', right_index=True, how='left')
    
    # Home advantage by league
    home_wins = df.groupby('Division')['FTR'].apply(lambda x: (x == 'H').mean())
    df['league_home_advantage'] = df['Division'].map(home_wins)
    
    # League identity (one-hot encoding for model to learn league-specific patterns)
    # Optional: Only if you want explicit league indicators
    # df = pd.get_dummies(df, columns=['Division'], prefix='is_league', drop_first=True)
    
    return df