"""
Feature Engineering Validation Script

Validates that features were calculated correctly by:
1. Checking temporal correctness (no data leakage)
2. Verifying rolling window calculations
3. Testing derived features logic
4. Ensuring no unexpected NaN values

Usage:
    python validate_features.py <path_to_features_csv>
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def validate_temporal_correctness(df: pd.DataFrame) -> dict:
    """
    Verify that rolling features only use past data.
    Check that first matches have NaN and later matches have values.
    """
    results = {'passed': True, 'issues': []}

    # Sort by date
    df = df.sort_values('Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Check that first matches have NaN for rolling features
    first_10 = df.head(10)
    rolling_cols = [col for col in df.columns if '_L3' in col or '_L5' in col]

    nan_count = first_10[rolling_cols].isna().sum().sum()
    if nan_count == 0:
        results['passed'] = False
        results['issues'].append("ERROR: First matches should have NaN values (no history available)")
    else:
        results['issues'].append(f" First matches correctly have {nan_count} NaN values")

    # Check that later matches have values
    later_matches = df.iloc[50:60]
    non_nan_count = later_matches[rolling_cols].notna().sum().sum()
    if non_nan_count < len(rolling_cols) * 5:
        results['passed'] = False
        results['issues'].append("WARNING: Later matches missing expected feature values")
    else:
        results['issues'].append(f" Later matches have {non_nan_count} non-NaN values")

    return results


def validate_rolling_calculations(df: pd.DataFrame) -> dict:
    """
    Manually verify rolling calculations for a sample team.
    """
    results = {'passed': True, 'issues': []}

    df = df.sort_values('Date').copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # Pick a team and verify their rolling features
    teams = df['HomeTeam'].unique()
    if len(teams) == 0:
        results['passed'] = False
        results['issues'].append("ERROR: No teams found")
        return results

    sample_team = teams[0]
    results['issues'].append(f"\n--- Validating calculations for: {sample_team} ---")

    # Get all matches for this team
    team_home = df[df['HomeTeam'] == sample_team].copy()
    team_away = df[df['AwayTeam'] == sample_team].copy()

    team_home['is_home'] = True
    team_away['is_home'] = False

    team_home['goals_for'] = team_home['FTHG']
    team_home['goals_against'] = team_home['FTAG']
    team_away['goals_for'] = team_away['FTAG']
    team_away['goals_against'] = team_away['FTHG']

    team_matches = pd.concat([team_home, team_away]).sort_values('Date')

    if len(team_matches) < 6:
        results['issues'].append(f"  INFO: Only {len(team_matches)} matches, skipping detailed validation")
        return results

    # Check 6th match (should have 5-match history)
    match_6 = team_matches.iloc[5]
    prev_5 = team_matches.iloc[0:5]

    # Calculate expected goals_scored_L5
    expected_goals = prev_5['goals_for'].mean()

    # Get actual value from feature
    if match_6['is_home']:
        actual_goals = match_6.get('home_goals_scored_L5', np.nan)
    else:
        actual_goals = match_6.get('away_goals_scored_L5', np.nan)

    if pd.notna(actual_goals):
        diff = abs(expected_goals - actual_goals)
        if diff < 0.01:
            results['issues'].append(f"   goals_scored_L5 correct: {actual_goals:.2f}")
        else:
            results['passed'] = False
            results['issues'].append(f"  ERROR: goals_scored_L5 mismatch!")
            results['issues'].append(f"    Expected: {expected_goals:.2f}, Got: {actual_goals:.2f}")
    else:
        results['issues'].append(f"  WARNING: goals_scored_L5 is NaN at match 6")

    return results


def validate_derived_features(df: pd.DataFrame) -> dict:
    """
    Check that derived features are calculated correctly.
    """
    results = {'passed': True, 'issues': []}

    # Check elo_diff
    if 'elo_diff' in df.columns and 'HomeElo' in df.columns and 'AwayElo' in df.columns:
        sample = df.dropna(subset=['elo_diff', 'HomeElo', 'AwayElo']).head(10)
        expected_diff = sample['HomeElo'] - sample['AwayElo']
        actual_diff = sample['elo_diff']

        if (abs(expected_diff - actual_diff) < 0.01).all():
            results['issues'].append(" elo_diff calculated correctly")
        else:
            results['passed'] = False
            results['issues'].append("ERROR: elo_diff calculation incorrect")

    # Check form_diff_L5
    if all(col in df.columns for col in ['form_diff_L5', 'home_ppg_L5', 'away_ppg_L5']):
        sample = df.dropna(subset=['form_diff_L5', 'home_ppg_L5', 'away_ppg_L5']).head(10)
        expected_form = sample['home_ppg_L5'] - sample['away_ppg_L5']
        actual_form = sample['form_diff_L5']

        if (abs(expected_form - actual_form) < 0.01).all():
            results['issues'].append(" form_diff_L5 calculated correctly")
        else:
            results['passed'] = False
            results['issues'].append("ERROR: form_diff_L5 calculation incorrect")

    # Check odds probabilities sum to ~1 (with bookmaker margin)
    if all(col in df.columns for col in ['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']):
        sample = df.dropna(subset=['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']).head(10)
        prob_sum = sample['odds_home_prob'] + sample['odds_draw_prob'] + sample['odds_away_prob']

        if (prob_sum > 1.0).all() and (prob_sum < 1.15).all():
            results['issues'].append(f" Betting odds probabilities valid (sum ~{prob_sum.mean():.3f})")
        else:
            results['passed'] = False
            results['issues'].append(f"ERROR: Betting odds probabilities sum incorrect: {prob_sum.mean():.3f}")

    return results


def validate_data_integrity(df: pd.DataFrame) -> dict:
    """
    Check for data quality issues.
    """
    results = {'passed': True, 'issues': []}

    # Check for completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        results['issues'].append(f"WARNING: {len(empty_cols)} columns are completely empty: {empty_cols[:5]}")

    # Check for unexpected NaN patterns
    rolling_cols = [col for col in df.columns if '_L5' in col]
    if rolling_cols:
        # After match 20, should have very few NaNs
        later_matches = df.iloc[20:]
        nan_pct = (later_matches[rolling_cols].isna().sum().sum() / 
                   (len(later_matches) * len(rolling_cols)) * 100)

        if nan_pct > 10:
            results['passed'] = False
            results['issues'].append(f"ERROR: High NaN percentage in later matches: {nan_pct:.1f}%")
        else:
            results['issues'].append(f" NaN percentage in later matches acceptable: {nan_pct:.1f}%")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        results['passed'] = False
        results['issues'].append(f"ERROR: Found {inf_count} infinite values")
    else:
        results['issues'].append(" No infinite values found")

    return results


def print_feature_summary(df: pd.DataFrame):
    """Print summary statistics of engineered features."""
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)

    # Identify feature groups
    rolling_features = [col for col in df.columns if '_L3' in col or '_L5' in col]
    match_features = [col for col in df.columns if col in ['elo_diff', 'form_diff_L5', 'home_gd_L5', 'away_gd_L5']]
    odds_features = [col for col in df.columns if col.startswith('odds_')]

    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Rolling features (L3, L5): {len(rolling_features)}")
    print(f"Match-level features: {len(match_features)}")
    print(f"Odds features: {len(odds_features)}")

    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Sample feature statistics
    if rolling_features:
        print(f"\nSample rolling feature stats (home_goals_scored_L5):")
        if 'home_goals_scored_L5' in df.columns:
            print(df['home_goals_scored_L5'].describe())


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_features.py <path_to_features_csv>")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    print("="*60)
    print("FEATURE ENGINEERING VALIDATION")
    print("="*60)
    print(f"\nLoading: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} matches")

    # Run validation tests
    all_passed = True

    print("\n" + "="*60)
    print("TEST 1: Temporal Correctness")
    print("="*60)
    result1 = validate_temporal_correctness(df)
    for issue in result1['issues']:
        print(issue)
    all_passed = all_passed and result1['passed']

    print("\n" + "="*60)
    print("TEST 2: Rolling Calculations")
    print("="*60)
    result2 = validate_rolling_calculations(df)
    for issue in result2['issues']:
        print(issue)
    all_passed = all_passed and result2['passed']

    print("\n" + "="*60)
    print("TEST 3: Derived Features")
    print("="*60)
    result3 = validate_derived_features(df)
    for issue in result3['issues']:
        print(issue)
    all_passed = all_passed and result3['passed']

    print("\n" + "="*60)
    print("TEST 4: Data Integrity")
    print("="*60)
    result4 = validate_data_integrity(df)
    for issue in result4['issues']:
        print(issue)
    all_passed = all_passed and result4['passed']

    # Summary
    print_feature_summary(df)

    print("\n" + "="*60)
    print("VALIDATION RESULT")
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED")
        print("\nFeatures are correctly calculated and ready for ML!")
    else:
        print("SOME TESTS FAILED")
        print("\nPlease review the errors above and fix the feature engineering code.")
    print("="*60)


if __name__ == "__main__":
    main()