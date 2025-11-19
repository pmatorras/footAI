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

    # Check if both opening (Avg) and closing (AvgC) odds exist
    if all(col in df.columns for col in ['AvgH', 'AvgD', 'AvgA', 'AvgCH', 'AvgCD', 'AvgCA']):
        # Calculate drift for each outcome (positive = odds lengthened)
        df['draw_odds_drift'] = (df['AvgCD'] - df['AvgD']) / df['AvgD']
        df['home_odds_drift'] = (df['AvgCH'] - df['AvgH']) / df['AvgH']
        df['away_odds_drift'] = (df['AvgCA'] - df['AvgA']) / df['AvgA']
        
        # Sharp money indicator: draw odds shortened > 2% (negative drift)
        df['sharp_money_on_draw'] = (df['draw_odds_drift'] < -0.02).astype(int)
        
        # Market uncertainty: magnitude of draw odds movement
        df['odds_movement_magnitude'] = np.abs(df['draw_odds_drift'])
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
    league_stats = df.groupby('Div').agg({
        'FTR': lambda x: (x == 'D').mean(),  # draw_rate
        'FTHG': 'mean',  # avg_goals_home
        'FTAG': 'mean',  # avg_goals_away
    }).rename(columns={'FTR': 'league_draw_rate', 
                       'FTHG': 'league_avg_goals_home',
                       'FTAG': 'league_avg_goals_away'})
    
    df = df.merge(league_stats, left_on='Div', right_index=True, how='left')
    
    # Home advantage by league
    home_wins = df.groupby('Div')['FTR'].apply(lambda x: (x == 'H').mean())
    df['league_home_advantage'] = df['Div'].map(home_wins)
    
    # League identity (one-hot encoding for model to learn league-specific patterns)
    # Optional: Only if you want explicit league indicators
    # df = pd.get_dummies(df, columns=['Division'], prefix='is_league', drop_first=True)
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum/trajectory features using rolling slope calculations.
    
    Calculates linear trend over last 5 matches for goals scored and PPG.
    Positive slope = improving form, negative slope = declining form.
    
    Args:
        df: DataFrame with L5 rolling features already computed
        
    Returns:
        DataFrame with new momentum features:
    
    Note:
        Requires that rolling features (goals_scored_L5, ppg_L5) have been
        computed first via calculate_team_rolling_features().
    """
    from footai.ml.feature_engineering.rolling import calculate_slope
    
    # Sort by date to ensure temporal correctness
    df = df.sort_values('Date').copy()
    
    # Initialize momentum columns
    df['home_goals_trend_L5'] = np.nan
    df['away_goals_trend_L5'] = np.nan
    df['home_ppg_trend_L5'] = np.nan
    df['away_ppg_trend_L5'] = np.nan
    
    # Process each team's home matches
    if 'HomeTeam' in df.columns and 'home_goals_scored_L5' in df.columns:
        for team in df['HomeTeam'].unique():
            team_mask = df['HomeTeam'] == team
            team_indices = df[team_mask].index
            
            # Get team's historical goals (sorted by date)
            team_goals = df.loc[team_mask, 'home_goals_scored_L5']
            team_ppg = df.loc[team_mask, 'home_ppg_L5']
            
            # Calculate slopes using rolling window
            goals_slopes = team_goals.rolling(window=5, min_periods=5).apply(
                calculate_slope, raw=False
            )
            ppg_slopes = team_ppg.rolling(window=5, min_periods=5).apply(
                calculate_slope, raw=False
            )
            
            # Assign back to dataframe
            df.loc[team_indices, 'home_goals_trend_L5'] = goals_slopes.values
            df.loc[team_indices, 'home_ppg_trend_L5'] = ppg_slopes.values
    
    # Process each team's away matches
    if 'AwayTeam' in df.columns and 'away_goals_scored_L5' in df.columns:
        for team in df['AwayTeam'].unique():
            team_mask = df['AwayTeam'] == team
            team_indices = df[team_mask].index
            
            # Get team's historical goals (sorted by date)
            team_goals = df.loc[team_mask, 'away_goals_scored_L5']
            team_ppg = df.loc[team_mask, 'away_ppg_L5']
            
            # Calculate slopes using rolling window
            goals_slopes = team_goals.rolling(window=5, min_periods=5).apply(
                calculate_slope, raw=False
            )
            ppg_slopes = team_ppg.rolling(window=5, min_periods=5).apply(
                calculate_slope, raw=False
            )
            
            # Assign back to dataframe
            df.loc[team_indices, 'away_goals_trend_L5'] = goals_slopes.values
            df.loc[team_indices, 'away_ppg_trend_L5'] = ppg_slopes.values
    
    # Calculate momentum differential
    df['momentum_diff'] = df['home_ppg_trend_L5'] - df['away_ppg_trend_L5']
    
    return df


def add_corners_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add corners-based features.
    
    High corners + low goals = defensive draw signal.
    """
    if 'home_corners_L5' in df.columns and 'away_corners_L5' in df.columns:
        # Corners ratio (parity indicator)
        df['corners_ratio'] = np.where(
            df['away_corners_L5'] > 0,
            df['home_corners_L5'] / df['away_corners_L5'],
            np.nan
        )
        
        # Defensive draw signal: high corners x low-scoring expectation
        if 'under_2_5_prob' in df.columns:
            avg_corners = (df['home_corners_L5'] + df['away_corners_L5']) / 2
            df['defensive_draw_signal'] = avg_corners * df['under_2_5_prob']
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features (products of existing features).
    
    Explicit interactions help gradient boosting models capture
    non-linear relationships between features.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with 4 new interaction features
    """
    # Elo x Odds: Strength agreement between systems
    if 'HomeElo' in df.columns and 'AwayElo' in df.columns and 'odds_home_prob_norm' in df.columns and 'odds_away_prob_norm' in df.columns:
        elo_strength = (df['HomeElo'] - df['AwayElo']) / 400  # Normalize Elo diff
        odds_strength = df['odds_home_prob_norm'] - df['odds_away_prob_norm']
        df['elo_odds_agreement'] = elo_strength * odds_strength
    
    # Form x Odds: Recent form weighted by market confidence
    if 'form_diff_L5' in df.columns and 'abs_odds_prob_diff' in df.columns:
        df['form_odds_weighted'] = df['form_diff_L5'] * df['abs_odds_prob_diff']
    
    # Parity x Market uncertainty: Draw signal amplifier
    if 'abs_elo_diff' in df.columns and 'draw_prob_dispersion' in df.columns:
        df['parity_uncertainty'] = (1 / (1 + df['abs_elo_diff'])) * df['draw_prob_dispersion']
    
    # Odds movement x Parity: Late information on balanced matches
    if 'draw_odds_drift' in df.columns and 'abs_odds_prob_diff' in df.columns:
        df['movement_parity_signal'] = df['draw_odds_drift'] * (1 - df['abs_odds_prob_diff'])
    
    return df
