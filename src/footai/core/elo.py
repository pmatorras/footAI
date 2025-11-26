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

def apply_rating_transfer(target_teams_list, source_teams_data, target_division, team_elos_carry, decay, regression_point=1500, verbose=False):
    """
    Transfers ratings from source teams to target teams, applying decay.
    Handles two modes:
    1. Merit-based Sort (T1<->T2): If target_teams contains (Team, Elo) tuples, both lists are sorted by Elo so the best incoming teams inherit the best available ratings.
    2. Blind Assignment (T3->T2): If target_teams is just a list of names (strings), we assign ratings from the best source teams to the targets in their provided list order.
    Args:
        target_teams: List of team names OR List of (Team, Rating) tuples.
        source_teams_data: List of (Team, Rating) tuples from the source pool.
        target_division: The division key (e.g. 'SP1') to assign into.
        team_elos_carry: The main dictionary to update.
        decay: Decay factor to apply during transfer.
    """
    # Sort Source Data (Best ratings first)
    source_teams_data.sort(key=lambda x: x[1], reverse=True)
    
    # Handle Target Data
    if target_teams_list and isinstance(target_teams_list[0], tuple):
        target_teams_list.sort(key=lambda x: x[1], reverse=True)
        target_names = [t[0] for t in target_teams_list]
    else:
        # Just use names as-is if elo can't be retrieved
        target_names = target_teams_list

    # 3. Transfer Loop
    for target_team, (source_team, source_elo) in zip(target_names, source_teams_data):
        decayed_elo = regression_point + (source_elo - regression_point) * decay
        team_elos_carry[target_division][target_team] = decayed_elo
        
        if verbose:
            print(f" {target_division}: {target_team} inherits {source_elo:.1f} -> {decayed_elo:.1f} from {source_team}")


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

def calculate_elo_multiseason(seasons, divisions, country, dirs, decay_factors={'tier1':0.95, 'tier2': 0.95}, initial_elo=1500, k_factor=32, args=None):
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
                if tier1_final_elos is None or tier2_final_elos is None:
                    print("Skipping ELO transfer because previous season data missing")
                else:
                    relegated_t1 = promo_relego_df[(promo_relego_df['status']=='relegated') & (promo_relego_df['tier']=='tier1')]['team'].tolist()
                    promoted_t1 = promo_relego_df[(promo_relego_df['status']=='promoted') & (promo_relego_df['tier']=='tier1')]['team'].tolist()
                    
                    relegated_t2 = promo_relego_df[(promo_relego_df['status']=='relegated') & (promo_relego_df['tier']=='tier2')]['team'].tolist()
                    promoted_t2 = promo_relego_df[(promo_relego_df['status']=='promoted') & (promo_relego_df['tier']=='tier2')]['team'].tolist()
                    relegated_data_t1 = []

                    for team in relegated_t1:
                        elo = tier1_final_elos.get(team, initial_elo)
                        relegated_data_t1.append((team, elo))
                        
                    # T1 Promoted
                    promoted_data_t1 = []
                    for team in promoted_t1:
                        elo = tier2_final_elos.get(team, initial_elo)
                        promoted_data_t1.append((team, elo))

                    # T2 Relegated
                    relegated_data_t2 = []
                    for team in relegated_t2:
                        elo = tier2_final_elos.get(team, initial_elo)
                        relegated_data_t2.append((team, elo))
                    # Transfer Relegated -> Promoted (Entering T1)
                        apply_rating_transfer(
                            target_teams_list=promoted_data_t1,
                            source_teams_data=relegated_data_t1,
                            target_division=divisions[0],
                            team_elos_carry=team_elos_carry,
                            decay=decay_factors['tier1'],
                            verbose=args.verbose
                        )
                        
                        # Transfer Promoted -> Relegated (Entering T2)
                        apply_rating_transfer(
                            target_teams_list=relegated_data_t1,
                            source_teams_data=promoted_data_t1,
                            target_division=divisions[1],
                            team_elos_carry=team_elos_carry,
                            decay=decay_factors['tier2'],
                            verbose=args.verbose
                        )
                            
                        apply_rating_transfer(
                            target_teams_list=promoted_t2,
                            source_teams_data=relegated_data_t2,
                            target_division=divisions[1],
                            team_elos_carry=team_elos_carry,
                            decay=decay_factors['tier2'],
                            verbose=args.verbose
                        )

        for div_idx, division in enumerate(divisions):
            tier_key = f'tier{div_idx + 1}'
            decay_factor = decay_factors.get(tier_key, 0.95)
            paths = get_season_paths(country, season, division, dirs, args)
            
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
            multi_season_file = get_multiseason_path(dirs[country]['proc'], division, seasons[0],seasons[-1], args)
            combined_df.to_csv(multi_season_file, index=False)
            
            print(f"\nCombined multi-season file: {multi_season_file}")
            print(f"  Total matches: {len(combined_df)}")
            print(f"  Seasons: {seasons[0]} to {seasons[-1]}")
       
