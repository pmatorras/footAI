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
# Momentum features
# --------------------------------------------------------------------------

MOMENTUM_FEATURES = [
    'home_goals_trend_L5',     # Goals trajectory (home)
    'away_goals_trend_L5',     # Goals trajectory (away)
    'home_ppg_trend_L5',       # Form trajectory (home)
    'away_ppg_trend_L5',       # Form trajectory (away)
    'momentum_diff',           # Differential
]

# --------------------------------------------------------------------------
# Odds Movement Features
# --------------------------------------------------------------------------

ODDS_MOVEMENT_FEATURES = [
    'draw_odds_drift',          # (ClosingD - OpeningD) / OpeningD
    'home_odds_drift',          # (ClosingH - OpeningH) / OpeningH
    'away_odds_drift',          # (ClosingA - OpeningA) / OpeningA
    'sharp_money_on_draw',      # Binary: draw odds shortened > 2%
    'odds_movement_magnitude',  # abs(draw_odds_drift)
]

# --------------------------------------------------------------------------
# Corners Features
# --------------------------------------------------------------------------

CORNERS_FEATURES = [
    'corners_ratio',            # home_corners_L5 / away_corners_L5
    'defensive_draw_signal',    # avg_corners * under_2_5_prob
]

# --------------------------------------------------------------------------
# Interaction Features
# --------------------------------------------------------------------------

INTERACTION_FEATURES = [
    'elo_odds_agreement',       # (elo_diff/400) * (odds_H - odds_A)
    'form_odds_weighted',       # form_diff * abs_odds_prob_diff
    'parity_uncertainty',       # (1/(1+abs_elo)) * draw_dispersion
    'movement_parity_signal',   # draw_drift * (1 - abs_odds_diff)
]

# --------------------------------------------------------------------------
# Combined Feature Sets 
# --------------------------------------------------------------------------

# Baseline variations
BASELINE_DRAW_LITE = BASELINE_FEATURES + DRAW_CORE
BASELINE_DRAW_FULL = BASELINE_FEATURES + DRAW_EXTENDED
BASELINE_DRAW_OPTIMIZED = BASELINE_FEATURES + DRAW_FULL_CLEAN  
BASELINE_DRAW_LITE_LEGACY = BASELINE_FEATURES + DRAW_CORE_LEGACY

#League and faults
BASELINE_LEAGUE_ADAPTIVE_LITE = BASELINE_DRAW_LITE + LEAGUE_FEATURES
BASELINE_LEAGUE_ADAPTIVE_OPTIMIZED = BASELINE_DRAW_OPTIMIZED + LEAGUE_FEATURES

BASELINE_FAULTS = BASELINE_DRAW_OPTIMIZED + FAULTS_FEATURES
# Extended variations
EXTENDED_DRAW_LITE = EXTENDED_FEATURES + DRAW_CORE
EXTENDED_DRAW_FULL = EXTENDED_FEATURES + DRAW_EXTENDED
EXTENDED_DRAW_OPTIMIZED = EXTENDED_FEATURES + DRAW_FULL_CLEAN  
EXTENDED_DRAW_LITE_LEGACY = EXTENDED_FEATURES + DRAW_CORE_LEGACY


# Momentum variations
BASELINE_MOMENTUM_LITE = BASELINE_LEAGUE_ADAPTIVE_LITE + MOMENTUM_FEATURES
BASELINE_MOMENTUM_OPTIMIZED = BASELINE_LEAGUE_ADAPTIVE_OPTIMIZED + MOMENTUM_FEATURES

# Odds movement
BASELINE_ODDS_LITE = BASELINE_LEAGUE_ADAPTIVE_LITE + ODDS_MOVEMENT_FEATURES
BASELINE_ODDS_OPTIMIZED = BASELINE_LEAGUE_ADAPTIVE_OPTIMIZED + ODDS_MOVEMENT_FEATURES

# Corners
CORNERS_LITE = BASELINE_ODDS_LITE + CORNERS_FEATURES
CORNERS_OPTIMIZED = BASELINE_ODDS_OPTIMIZED + CORNERS_FEATURES

#Interactions

INTERACTIONS_LITE = BASELINE_ODDS_LITE + INTERACTION_FEATURES
INTERACTIONS_OPTIMIZED = BASELINE_ODDS_OPTIMIZED + INTERACTION_FEATURES

# Current contenders
BASELINE_LITE = BASELINE_ODDS_LITE
BASELINE_OPTIMIZED = BASELINE_ODDS_OPTIMIZED

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
    'league_adaptive' : BASELINE_LEAGUE_ADAPTIVE_OPTIMIZED,
    'league_lite' : BASELINE_LEAGUE_ADAPTIVE_LITE,

    'faults' : BASELINE_FAULTS,
    #momentum
    'momentum_lite' : BASELINE_MOMENTUM_LITE,
    'momentum_optimized' : BASELINE_MOMENTUM_OPTIMIZED,
    #odds movement
    'odds_lite' : BASELINE_ODDS_LITE,
    'odds_optimized' : BASELINE_ODDS_OPTIMIZED,
    #corners
    'corners_lite' : CORNERS_LITE,
    'corners_optimized' : CORNERS_OPTIMIZED,
    #interactions
    'interactions_lite' : INTERACTIONS_LITE,
    'interactions_optimized' : INTERACTIONS_OPTIMIZED,
}


