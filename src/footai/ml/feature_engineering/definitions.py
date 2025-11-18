"""
Feature Set Definitions
=======================

All tested feature combinations for match prediction.
"""

# --------------------------------------------------------------------------
# Foundation Features
# --------------------------------------------------------------------------

BASELINE_FEATURES = [
    # Elo Ratings (team strength indicators)
    'HomeElo',              # Home team Elo rating
    'AwayElo',              # Away team Elo rating
    'elo_diff',             # HomeElo - AwayElo (strength differential)
    
    # Recent Form (last 5 matches)
    'home_goals_scored_L5',     # Goals scored by home team 
    'away_goals_scored_L5',     # Goals scored by away team 
    'home_goals_conceded_L5',   # Goals conceded by home team 
    'away_goals_conceded_L5',   # Goals conceded by away team 
    'home_ppg_L5',              # Points per game (home)
    'away_ppg_L5',              # Points per game (away)
    'form_diff_L5',             # home_ppg_L5 - away_ppg_L5 (form differential)
    
    # Betting Market Signals
    'odds_home_prob_norm',  # Normalized home win probability from odds
    'odds_away_prob_norm',  # Normalized away win probability from odds
] 

EXTENDED_FEATURES = BASELINE_FEATURES + [
    # Shot Metrics (attacking quality indicators)
    'home_shots_L5',            # Average shots per match (home)
    'away_shots_L5',            # Average shots per match (away)
    'home_shot_accuracy_L5',    # Shots on target % (home)
    'away_shot_accuracy_L5',    # Shots on target % (away)
    
    # Goal Differentials (net performance over last 5)
    'home_gd_L5',              # Goals scored - conceded (home)
    'away_gd_L5',              # Goals scored - conceded (away)
] 

# --------------------------------------------------------------------------
# Draw Feature Components
# --------------------------------------------------------------------------

DRAW_CORE_LEGACY = [
    'draw_prob_consensus',      # Consensus draw prob across bookmakers
    'under_2_5_prob',           # Low-scoring indicator
    'draw_prob_dispersion',     # Market disagreement on draws
    'odds_draw_prob_norm',      # Normalized draw probability
    'asian_handicap_diff',      # Handicap-based parity (if exists)
] 
# Data-driven top 5 (from baseline_draw_full analysis)
DRAW_CORE = [
    'abs_odds_prob_diff',      # Odds parity indicator
    'draw_prob_consensus',     # Market draw signal
    'elo_diff_sq',             # Non-linear parity
    'abs_elo_diff',            # Strength gap magnitude
    'draw_prob_dispersion',    # Market uncertainty
]

DRAW_FULL_CLEAN = DRAW_CORE + [
    'under_2_5_prob',          # Low-scoring indicator
    'min_shot_acc_l5',         # Defensive quality
    'abs_ahh',                 # Asian handicap parity
]

DRAW_EXTENDED = DRAW_CORE_LEGACY + [
    # Parity indicators
    'abs_odds_prob_diff',       # How close are home/away odds?
    'abs_elo_diff',             # Elo parity
    'elo_diff_sq',              # Non-linear Elo parity
    'under_2_5_zscore',         # Standardized under prob
    
    # Low-scoring composites
    'min_shots_l5',             # Min(home_shots, away_shots)
    'min_shot_acc_l5',          # Min shot accuracy
    'min_goals_scored_l5',      # Min goals scored
    
    # Asian handicap signals
    'abs_ahh',                  # Absolute handicap
    'ahh_zero',                 # Zero handicap indicator
    'ahh_flat',                 # Flat handicap
    
    # Medium/low Elo diff bins
    'low_elo_diff',             # |elo_diff| < 25
    'medium_elo_diff',          # 25 <= |elo_diff| < 50
] 

# --------------------------------------------------------------------------
# League draw and fault components (For reference only)
# --------------------------------------------------------------------------
LEAGUE_FEATURES = [
    'league_draw_bias',        # Historical draw rate per league (e.g., Serie A: 28.9%, EPL: 24.5%)
    'home_draw_rate_l10',      # Team's rolling draw rate at home (last 10 matches)
    'away_draw_rate_l10',      # Team's rolling draw rate away (last 10 matches)
]

FAULTS_FEATURES = [
    'home_fouls_L5',           # Average fouls committed (home, last 5)
    'away_fouls_L5',           # Average fouls committed (away, last 5)
    'foul_diff_L5',            # home_fouls_L5 - away_fouls_L5 (aggression differential)
]

# --------------------------------------------------------------------------
# Combined Feature Sets 
# --------------------------------------------------------------------------

# Baseline variations
BASELINE_DRAW_LITE = BASELINE_FEATURES + DRAW_CORE
BASELINE_DRAW_FULL = BASELINE_FEATURES + DRAW_EXTENDED
BASELINE_DRAW_OPTIMIZED = BASELINE_FEATURES + DRAW_FULL_CLEAN  
BASELINE_DRAW_LITE_LEGACY = BASELINE_FEATURES + DRAW_CORE_LEGACY
BASELINE_LEAGUE_ADAPTIVE = BASELINE_DRAW_OPTIMIZED + LEAGUE_FEATURES
BASELINE_FAULTS = BASELINE_DRAW_OPTIMIZED + FAULTS_FEATURES
# Extended variations
EXTENDED_DRAW_LITE = EXTENDED_FEATURES + DRAW_CORE
EXTENDED_DRAW_FULL = EXTENDED_FEATURES + DRAW_EXTENDED
EXTENDED_DRAW_OPTIMIZED = EXTENDED_FEATURES + DRAW_FULL_CLEAN  
EXTENDED_DRAW_LITE_LEGACY = EXTENDED_FEATURES + DRAW_CORE_LEGACY


# --------------------------------------------------------------------------
# Feature set registry
# --------------------------------------------------------------------------

FEATURE_SETS = {
    # Foundation
    'baseline': BASELINE_FEATURES,
    'extended': EXTENDED_FEATURES,
    
    # Draw options
    'baseline_draw_lite': BASELINE_DRAW_LITE,
    'baseline_draw_full': BASELINE_DRAW_FULL,
    'extended_draw_lite': EXTENDED_DRAW_LITE,
    'extended_draw_full': EXTENDED_DRAW_FULL,
    'baseline_draw_optimized': BASELINE_DRAW_OPTIMIZED,
    'extended_draw_optimized': EXTENDED_DRAW_OPTIMIZED,
    'baseline_draw_lite_legacy': BASELINE_DRAW_LITE_LEGACY,
    'extended_draw_lite_legacy': EXTENDED_DRAW_LITE_LEGACY,
    # League specifics
    'league_adaptive' : BASELINE_LEAGUE_ADAPTIVE,
    'faults' : BASELINE_FAULTS,
}


