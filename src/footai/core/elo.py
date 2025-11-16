import pandas as pd
import warnings
from collections import defaultdict
from footai.core.team_movements import load_promotion_relegation
from footai.utils.paths import get_season_paths, get_multiseason_path
warnings.filterwarnings('ignore', message='Could not infer format')
def expected_score(elo_a, elo_b):
    """
    Expected probability that team A beats team B using official Elo formula.
    
    Reference: https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    
    Formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def new_elo(old_elo, expected, actual, k_factor=32):
    """Calculate new ELO rating after a match"""
    return old_elo + k_factor * (actual - expected)


def calculate_elo_season(matches_df, initial_elo=1500, k_factor=32, team_starting_elos=None):
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
    if team_starting_elos:
        team_elos.update(team_starting_elos)
    team_history = defaultdict(list)
    
    # Sort chronologically
    # Convert Date to datetime (handle format properly)
    matches_df = matches_df.copy()
    matches_df['Date'] = pd.to_datetime(matches_df['Date'], format='%d/%m/%Y', errors='coerce')
    
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

    return output_df

def calculate_elo_multiseason(seasons, divisions, country, dirs, decay_factor=0.95, initial_elo=1500, k_factor=32, args=None):
    """
    Process Elo ratings across multiple seasons with continuity and decay.
    
    Carries forward team Elo between seasons with decay factor to reflect
    that team strength degrades slightly during off-season.
    """
    
    regression_point = initial_elo
    team_elos_carry = {div: {} for div in divisions}  # Track per division

    all_season_dfs = {div: [] for div in divisions}

    print("ELO TRANSFER MODE:", args.elo_transfer)
    tier1_final_elos = None
    tier2_final_elos = None
    for season_idx, season in enumerate(seasons):
        tier1_final_elos_this_season = None
        tier2_final_elos_this_season = None

         # if we have previous season data and the flag is on, do ELO transfer
        if args.elo_transfer and season_idx > 0 and len(divisions) == 2:
            promo_relego_df = load_promotion_relegation(season, country, dirs)
            if args.verbose: print(promo_relego_df)
            if promo_relego_df is not None:
                relegated = promo_relego_df[promo_relego_df['status'] == 'relegated']['team'].tolist()
                promoted = promo_relego_df[promo_relego_df['status'] == 'promoted']['team'].tolist()
                if relegated and promoted:
                    # Use last season's ELOs from tier1 and tier2
                    if tier1_final_elos is None or tier2_final_elos is None:
                        print("Skipping ELO transfer because previous season data missing")
                    else:
                        #define last season elos, pahts, and dfs
                        last_eason_promoted_elos = [tier2_final_elos.get(team, initial_elo) for team in promoted]
                        last_eason_relegated_elos = [tier1_final_elos.get(team, initial_elo) for team in relegated]
                        
                        # Promoted teams to top division get ELOs from relegated teams in lower division
                        for promoted_team, rel_elo in zip(promoted, last_eason_relegated_elos):
                            team_elos_carry[divisions[0]][promoted_team] = rel_elo
                            if args.verbose: print(f"{divisions[0]}({season}): Transferred {rel_elo:.1f} to {promoted_team}")
                        # Complementary, transfer the old ELO from SP2 to the newly demoted teams
                        for relegated_team, promo_elo in zip(relegated, last_eason_promoted_elos):
                            team_elos_carry[divisions[1]][relegated_team] = promo_elo
                            if args.verbose: print(f"{divisions[1]}({season}): Transferred {promo_elo:.1f} to {relegated_team}")


        for division in divisions:
            paths = get_season_paths(season, division, dirs, args)
            
            df = pd.read_csv(paths['raw'])
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

            df_with_elos = calculate_elo_season(df, initial_elo=initial_elo,k_factor=k_factor, team_starting_elos=team_elos_carry[division])
            df_with_elos['Season'] = season
            dates = df_with_elos['Date']
            if args.verbose: print(f"{season},  - {dates.min()} to {dates.max()} ({len(df)} matches)")
            all_season_dfs[division].append(df_with_elos)
            # Extract and decay
            final_elos = {
                team: df_with_elos[df_with_elos['HomeTeam'] == team].iloc[-1]['HomeElo']
                for team in df_with_elos['HomeTeam'].dropna().unique()
            }
            team_elos_carry[division]= {
                team: regression_point + (elo - regression_point) * decay_factor
                for team, elo in final_elos.items()
            }
            if division == divisions[0]:
                tier1_final_elos_this_season = final_elos.copy()
            else:
                tier2_final_elos_this_season = final_elos.copy()
                
        # Update previous season final elos for next iteration
        tier1_final_elos = tier1_final_elos_this_season
        tier2_final_elos = tier2_final_elos_this_season


    for division in divisions:
        if all_season_dfs[division]:
            combined_df = pd.concat(all_season_dfs[division], ignore_index=True)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            # Save with multi-season naming
            multi_season_file = get_multiseason_path(dirs['proc'], division, seasons[0],seasons[-1], args)
            combined_df.to_csv(multi_season_file, index=False)
            
            print(f"\nCombined multi-season file: {multi_season_file}")
            print(f"  Total matches: {len(combined_df)}")
            print(f"  Seasons: {seasons[0]} to {seasons[-1]}")
       
