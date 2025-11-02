import pandas as pd
from collections import defaultdict


def expected_score(elo_a, elo_b):
    """Expected probability that team A beats team B"""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def new_elo(old_elo, expected, actual, k_factor=32):
    """Calculate new ELO rating after a match"""
    return old_elo + k_factor * (actual - expected)


def calculate_elo_ratings(matches_df, initial_elo=1500, k_factor=32):
    """
    Calculate ELO ratings for all teams based on match results.
    
    Parameters:
    - matches_df: DataFrame with columns: Date, HomeTeam, AwayTeam, FTHG (Full time home goals), FTAG (full time away goals)
    - initial_elo: Starting ELO for all teams (default: 1500)
    - k_factor: Rating volatility (default: 32, higher = more change per match)
    
    Returns:
    - dict: Team name -> list of (date, elo) tuples
    - DataFrame: Original data with added ELO columns
    """
    
    # Initialize ELO ratings
    team_elos = defaultdict(lambda: initial_elo)
    team_history = defaultdict(list)
    
    # Sort chronologically
    matches_df = matches_df.sort_values('Date').reset_index(drop=True)
    
    # Create output with ELO columns
    output_df = matches_df.copy()
    output_df['HomeElo'] = 0.0
    output_df['AwayElo'] = 0.0
    output_df['HomeExpected'] = 0.0
    output_df['AwayExpected'] = 0.0
    
    # Process each match
    for idx, match in output_df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Pre-match ELO values
        home_elo = team_elos[home_team]
        away_elo = team_elos[away_team]
        
        # Expected probabilities
        home_expected = expected_score(home_elo, away_elo)
        away_expected = expected_score(away_elo, home_elo)
        
        # Store pre-match values
        output_df.loc[idx, 'HomeElo'] = home_elo
        output_df.loc[idx, 'AwayElo'] = away_elo
        output_df.loc[idx, 'HomeExpected'] = home_expected
        output_df.loc[idx, 'AwayExpected'] = away_expected
        
        # Determine actual result
        home_goals = match['FTHG']
        away_goals = match['FTAG']
        
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Calculate new ELO ratings
        new_home_elo = new_elo(home_elo, home_expected, home_actual)
        new_away_elo = new_elo(away_elo, away_expected, away_actual)
        
        # Update for next match
        team_elos[home_team] = new_home_elo
        team_elos[away_team] = new_away_elo
        
        # Store history
        team_history[home_team].append((match['Date'], new_home_elo))
        team_history[away_team].append((match['Date'], new_away_elo))
    
    return dict(team_history), output_df
