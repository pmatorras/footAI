"""
Rolling Statistics Computation
==============================

Temporal feature computation with no data leakage.
"""
import pandas as pd
import numpy as np
from typing import Dict
def team_matches_rows(df, team_name):
    """Extract chronological match history for a team."""
    team_matches = []
    for idx, row in df.iterrows():
        if row['HomeTeam'] == team_name:
            team_matches.append({
                'date': row['Date'],
                'goals_scored': row['FTHG'],
                'goals_conceded': row['FTAG'],
                'result': row['FTR'],  # H/D/A
                'is_home': True,
                # Shot data (might not exist in older seasons)
                'shots': row.get('HS', None),
                'shots_on_target': row.get('HST', None),
                'fouls': row.get('HF', None),
            })
        elif row['AwayTeam'] == team_name:
            team_matches.append({
                'date': row['Date'],
                'goals_scored': row['FTAG'],
                'goals_conceded': row['FTHG'],
                'result': 'W' if row['FTR'] == 'A' else ('D' if row['FTR'] == 'D' else 'L'),
                'is_home': False,
                # Shot data (might not exist in older seasons)
                'shots_conceded': row.get('AS', None),
                'shots_on_target_conceded': row.get('AST', None),
                'fouls_conceded': row.get('AF', None),
            })
    return team_matches

def calculate_team_rolling_features(df: pd.DataFrame, team_name: str, window: int, cache: Dict) -> Dict:
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
    date_col = 'Date' if 'Date' in team_matches[0] else 'date'
    team_df = pd.DataFrame(team_matches).sort_values(date_col).reset_index(drop=True)

    # Calculate rolling features
    features = {}
    for i in range(len(team_df)):
        match_date = team_df.iloc[i][date_col]

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

            # Foul aggregations
            fouls_data = prev_matches['fouls'].dropna()
            avg_fouls = fouls_data.mean() if len(fouls_data) > 0 else np.nan
            
            features[match_date] = {
                f'goals_scored_L{window}': prev_matches['goals_scored'].mean(),
                f'goals_conceded_L{window}': prev_matches['goals_conceded'].mean(),
                f'ppg_L{window}': points.mean(),
                f'shots_L{window}': prev_matches['shots'].mean(),
                f'shot_accuracy_L{window}': shot_acc,
                f'fouls_L{window}': avg_fouls,  
            }

    cache[cache_key] = features
    return features