# Feature Engineering Documentation

Generated: 2025-11-19  

---

## Table of Contents
1. [Overview](#overview)
2. [Production Feature Sets](#production-feature-sets)
3. [Feature Categories](#feature-categories)
4. [Elo-Based Features](#elo-based-features)
5. [Rolling Form Features](#rolling-form-features)
6. [Match-Level Features](#match-level-features)
7. [Betting Market Features](#betting-market-features)
8. [League Context Features](#league-context-features)
9. [Technical Details](#technical-details)
10. [ML Model Expectations](#ml-model-expectations)


---

## Overview

This document describes all engineered features used for match outcome prediction (Home/Draw/Away). Features are calculated with temporal correctness - for any match N, features only use data from matches 1 to N-1, to prevent data leakage.

### Feature Count Summary
- **Elo features**: 5 (HomeElo, AwayElo, HomeExpected, AwayExpected, elo_diff)
- **Rolling form features**: 20 (10 per window size: L3 or L5)
- **Match-level features**: 13 (form_diff_L5, home_gd_L5, away_gd_L5, is_home, odds_elo_diff, draw parity signals)
- **Betting market features**: 11 (probabilities, normalized values, draw consensus, odds movement)
- **League context features**: 3 (league draw bias, rolling draw rates)
- **Total engineered features**: 52 features

##**Key results**:
- Baseline: 12 features, 34.35% draw recall
- Production: 25-28 features, 38.08-38.17% draw recall
- Total improvement: **+3.73-3.82% draw recall**

See [feature_configuration.md](./feature_configuration.md) for full experiment details.

## Production Feature Sets
Two models have been shortlisted after the systematic testing, with similar accuracy and draw recall (DR):
### a) odds_lite (25 features) 

**Composition**: 12 baseline + 5 draw_lite + 3 league + 5 odds_movement

**Performance**:
- Accuracy: 50.42%
- Draw Recall: 38.08%
- Model: Random Forest


### b) odds_optimized (28 features)

**Composition**: 12 baseline + 8 draw_optimized + 3 league + 5 odds_movement

**Performance**:
- Accuracy: 50.57%
- Draw Recall: 38.17%
- Model: Random Forest
---

## Feature Categories

### Elo-Based Features

#### `HomeElo`
- **Type**: Float
- **Calculation**: Team's current Elo rating (updated after each match)
- **Purpose**: Captures long-term team strength
- **Range**: 1200-1800 (typically 1400-1600)
- **Initial value**: 1500 for all teams at the first season (see the command options in [**README.md**](../README.md#command-options))
> **Example**: Real Madrid = 1650

#### `AwayElo`
- **Type**: Float
- **Calculation**: Same as HomeElo, for away team
- **Range**: 1200-1800 (typically 1400-1600)
> **Example**: Barcelona = 1625

#### `HomeExpected`
- **Type**: Float (0-1)
- **Calculation**: Expected **points** for home team using Elo formula
  ```
  HomeExpected = 1 / (1 + 10^((AwayElo - HomeElo) / 400))
  ```
- **Range**: 0.0-1.0 (typically 0.3-0.7)
- **Interpretation**: They **don't** represent pure win probability. Instead, it represents expected points where:
  - Win = 1.0 points
  - Draw = 0.5 points
  - Loss = 0.0 points
- **Average value**: 0.496
- `HomeExpected + AwayExpected` has to always be `= 1.0`, with draw probability "split" 0.5/0.5 between both teams:
    - HomeExpected = P(Home Win) + 0.5 × P(Draw)
    - AwayExpected = P(Away Win) + 0.5 × P(Draw)

> **Example**: 0.60 means expected to score 0.6 points (could be 40% win + 40% draw, or 50% win + 20% draw, etc.)
#### `AwayExpected`
- **Type**: Float (0-1)
- **Calculation**: Same formula as HomeExpected, for away team
- **Average value**: 0.504 (slightly favors away team in this dataset)

#### `elo_diff`
- **Type**: Float
- **Calculation**: `HomeElo - AwayElo`
- **Range**: -400 to +400 (typically -200 to +200)
- **Purpose**: Direct measure of team strength difference
- **Interpretation**:
  - Positive: Home team stronger
  - Zero: Equal strength
  - Negative: Away team stronger
> **Example**: 25 -> (home team slightly stronger)

---

### Rolling Form Features

All rolling features use **variable-length windows** for early matches:
- Match 1: NaN (no history)
- Match 2: Average of 1 match
- Match 3: Average of 2 matches
- Match 6+: Full window (5 matches for L5)

#### Window Sizes
- **L3**: Last 3 matches (recent form emphasis)
- **L5**: Last 5 matches (balanced view)

#### Per Team (Home and Away)

##### `home_goals_scored_L5` / `away_goals_scored_L5`
- **Type**: Float
- **Calculation**: Average goals scored per match in last 5 games
- **Range**: 0.0-5.0 (typically 0.5-3.0)
- **Purpose**: Recent attacking effectiveness
> **Example**: Real Madrid `home_goals_scored_L5` = 2.4

##### `home_goals_conceded_L5` / `away_goals_conceded_L5`
- **Type**: Float
- **Calculation**: Average goals conceded per match in last 5 games
- **Purpose**: Recent defensive solidity
- **Range**: 0.0-4.0 (typically 0.5-2.5)
- **Purpose**: Recent defensive solidity
> **Example**: Getafe away_goals_conceded_L5 = 0.8

##### `home_ppg_L5` / `away_ppg_L5`
- **Type**: Float
- **Calculation**: Points per game in last 5 matches
  - Win = 3 points
  - Draw = 1 point
  - Loss = 0 points
- **Range**: 0.0-3.0 (typically 0.5-2.5)
- **Purpose**: Overall recent form quality
- **Interpretation**:
  - 3.0 = All wins
  - 1.5 = Mix of wins/draws/losses
  - 0.0 = All losses
> **Example**: Barcelona home_ppg_L5 = 2.2

##### `home_shots_L5` / `away_shots_L5`
- **Type**: Float
- **Calculation**: Average total shots per match in last 5 games
- **Range**: 5.0-25.0 (typically 10-18)
- **Purpose**: Attacking intensity and possession dominance
> **Example**: Santander away_shots_L5 = 17.2

##### `home_shot_accuracy_L5` / `away_shot_accuracy_L5`
- **Type**: Float (percentage)
- **Calculation**: (Total shots on target / Total shots) × 100 over last 5 matches
- **Range**: 20%-60% (typically 30-45%)
- **Purpose**: Attacking quality and finishing ability
> **Example**: Atletico home_shot_accuracy_L5 = 38.5%

---

### Match-Level Features

These features combine information from both teams.

#### `form_diff_L5`
- **Type**: Float
- **Calculation**: `home_ppg_L5 - away_ppg_L5`
- **Range**: -3.0 to +3.0 (typically -2.0 to +2.0)
- **Purpose**: Direct form comparison
- **Interpretation**:
  - Positive: Home team has better recent form
  - Negative: Away team has better recent form
  - Near zero: Similar form
> **Example**: +0.8 (home team in better recent form than away team)

#### `home_gd_L5` / `away_gd_L5`
- **Type**: Float
- **Calculation**: `goals_scored_L5 - goals_conceded_L5`
- **Range**: -4.0 to +4.0 (typically -2.0 to +3.0)
- **Purpose**: Balance of attack vs defense
> **Example**: `home_gd_L5` = +1.6 (scoring more than conceding at home).
> Equivalent if `away_gd_L5` = +3.2

#### `is_home`
- **Type**: Integer (binary)
- **Calculation**: Always 1 for the home team
- **Range**: 1
- **Purpose**: Captures home advantage effect
- **Note**: May seem redundant but helps model explicitly learn home advantage

#### `odds_elo_diff`
- **Type**: Float
- **Calculation**: `odds_home_prob_norm - HomeExpected`
- **Range**: -0.3 to +0.3 (typically -0.1 to +0.1)
- **Purpose**: **Market disagreement signal**
- **Interpretation**:
  - Positive: Betting market more optimistic about home team than Elo
  - Negative: Elo more optimistic about home team than market
  - High absolute value: Potential value bet (market-Elo mismatch)
> **Example**: -0.05 -> (betting market rates home team slightly lower than Elo)

#### `abs_odds_prob_diff`
- **Type**: Float
- **Calculation**: `|odds_home_prob_norm - odds_away_prob_norm|`
- **Range**: 0.0-0.8 (typically 0.0-0.4)
- **Purpose**: Match parity indicator - lower values suggest closer match, higher draw probability
- **Interpretation**:
  - Near 0: Very close match (high draw chance)
  - \> 0.3: Clear favorite (lower draw chance)
> **Example**: 0.05 (very close match, high draw probability)

#### `elo_diff_sq`
- **Type**: Float
- **Calculation**: `(elo_diff)`$^2$
- **Range**: 0-160000 (typically 0-40000)
- **Purpose**: Non-linear parity capture - amplifies large differences while compressing small ones
> **Example**: elo_diff=50 → 2500 (moderate difference)

#### `abs_elo_diff`
- **Type**: Float
- **Calculation**: `|elo_diff|`
- **Range**: 0-400 (typically 0-200)
- **Purpose**: Team parity indicator (independent of home/away)
> **Example**: 75 (moderate strength difference)

#### `min_shot_acc_l5`
- **Type**: Float (percentage)
- **Calculation**: `min(home_shot_accuracy_L5, away_shot_accuracy_L5)`
- **Range**: 20%-60% (typically 30-45%)
- **Purpose**: Mutual attacking inefficiency indicator - low values suggest both teams struggle to finish, leading to low-scoring draws
> **Example**: 28% (both teams inefficient, potential draw)
---

### Betting Market Features
#### Draw Consensus Features

##### `draw_prob_consensus`
- **Type**: Float (0-1)
- **Calculation**: Mean draw probability across multiple bookmakers  
`draw_prob_consensus = mean([B365D, BWD, IWD, LBD, PSD, WHD, VCD])`
- **Range**: 0.15-0.40 (typically 0.23-0.32)
- **Purpose**: Market consensus on draw likelihood - removes single-bookmaker noise
> **Example**: 0.28 (28% draw probability per market consensus) `draw_prob_dispersion = std([B365D, BWD, IWD, LBD, PSD, WHD, VCD])

##### `draw_prob_dispersion`
- **Type**: Float
- **Calculation**: Standard deviation of draw probabilities across bookmakers
`draw_prob_dispersion = std([B365D, BWD, IWD, LBD, PSD, WHD, VCD])`
- **Range**: 0.01-0.10 (typically 0.02-0.05)
- **Purpose**: Market uncertainty indicator - high dispersion suggests disagreement about match difficulty
- **Interpretation**:
- Low (<0.03): High bookmaker agreement
- High (>0.05): Bookmaker disagreement, potential value
> **Example**: 0.04 (moderate market disagreement)

##### `under_2_5_prob`
- **Type**: Float (0-1)
- **Calculation**: Implied probability of under 2.5 total goals from betting odds
`under_2_5_prob = 1 / under_2_5_decimal_odds`
- **Range**: 0.30-0.70 (typically 0.40-0.60)
- **Purpose**: Low-scoring match indicator - correlates with draw outcomes
> **Example**: 0.55 (55% chance of under 2.5 goals - defensive match)

#### Odds Movement Features

##### `draw_odds_drift`
- **Type**: Float
- **Calculation**: `(ClosingDrawOdds - OpeningDrawOdds) / OpeningDrawOdds`
- **Range**: -0.30 to +0.30 (typically -0.10 to +0.10)
- **Purpose**: Captures late information arrival (injuries, lineup news, weather, sharp money)
- **Interpretation**:
- Negative: Draw odds shortened (money came in on draw)
- Positive: Draw odds lengthened (money came off draw)
- Large magnitude: Significant late information
> **Example**: -0.08 (draw odds shortened 8%, suggests late draw value)

##### `home_odds_drift` / `away_odds_drift`
- **Type**: Float
- **Calculation**: `(ClosingOdds - OpeningOdds) / OpeningOdds` for home/away outcomes
- **Range**: -0.40 to +0.40 (typically -0.15 to +0.15)
- **Purpose**: Track market sentiment changes for home/away outcomes
> **Example**: home_odds_drift = +0.12 (home odds lengthened, less confidence in home win)

##### `sharp_money_on_draw`
- **Type**: Integer (binary)
- **Calculation**: `1 if draw_odds_drift < -0.02 else 0`
- **Range**: 0 or 1
- **Purpose**: Binary indicator for significant draw odds shortening (sharp money detection)
- **Interpretation**:
- 1: Draw odds shortened >2% (smart money on draw)
- 0: No significant draw odds movement
> **Example**: 1 (sharp money detected on draw)

##### `odds_movement_magnitude`
- **Type**: Float
- **Calculation**: `|draw_odds_drift|`
- **Range**: 0.0-0.30 (typically 0.0-0.10)
- **Purpose**: Uncertainty signal - large movements indicate information asymmetry
> **Example**: 0.09 (significant odds movement, high uncertainty)

#### Raw Odds Probabilities

##### `odds_home_prob` / `odds_draw_prob` / `odds_away_prob`
- **Type**: Float (0-1)
- **Calculation**: Implied probability from betting odds
  ```
  odds_home_prob = 1 / decimal_odds_home
  ```
- **Purpose**: Captures market assessment
- **Average values** (your dataset):
  - Home: 0.471
  - Draw: 0.275
  - Away: 0.311
  - **Sum**: 1.0566 (105.66%)
> **Example**: If odds are 2.10, probability = 1/2.10 = 0.476

> Sum > 1.0 is the **bookmaker's overround** (profit margin). In your data, the 5.66% margin is typical for major European leagues. This represents the bookmaker's guaranteed profit edge.

#### Normalized Probabilities

##### `odds_home_prob_norm` / `odds_away_prob_norm`
- **Type**: Float (0-1)
- **Calculation**: Remove bookmaker margin
  ```
  total = odds_home_prob + odds_draw_prob + odds_away_prob
  odds_home_prob_norm = odds_home_prob / total
  ```
- **Purpose**: True market-estimated probabilities without bookmaker margin, so that one can compare directly with Elo Expected values
- **Sum**: Normalized probabilities sum to exactly 1.0
- **Use case**: 
> **Example**: Raw: 0.471-> Normalized: 0.471 / 1.0566 = 0.446
---

### 5. League Context Features

These features capture cross-league differences in playing styles and draw rates.

#### `league_draw_bias`
- **Type**: Float
- **Calculation**: Historical draw rate for the league over previous seasons
- **Range**: 0.23-0.31 (typically 0.24-0.29)
- **Purpose**: Captures inherent league characteristics (tactical, attacking, defensive styles)
- **League profiles**:
  - Serie A: 0.289 (defensive, tactical)
  - La Liga: 0.273 (technical, balanced)
  - Bundesliga: 0.262 (attacking)
  - Ligue 1: 0.259 (varied)
  - Premier League: 0.245 (high-tempo, attacking)
> **Example**: 0.289 for Serie A match (higher baseline draw expectation)

#### `home_draw_rate_l10` / `away_draw_rate_l10`
- **Type**: Float
- **Calculation**: Team's rolling draw rate over last 10 matches (home or away)
`home_draw_rate_l10 = (draws in last 10 home matches) / 10`
- **Range**: 0.0-0.7 (typically 0.1-0.5)
- **Purpose**: Team-specific draw tendency (some teams draw more frequently)
- **Interpretation**:
- High (>0.4): Team frequently draws (defensive style, mid-table mentality)
- Low (<0.2): Team rarely draws (attacking style, win-or-lose mentality)
> **Example**: 0.35 (team draws 35% of home matches - defensive approach)


---
## Technical Details

### Temporal Correctness

Features for match N use only data from matches 1 to N-1. Example timeline for Real Madrid:
```
Match 1 (Aug 18): home_goals_scored_L5 = NaN (no history)
Match 2 (Aug 25): home_goals_scored_L5 = avg([Match 1])
Match 3 (Aug 29): home_goals_scored_L5 = avg([Match 1, 2])
Match 6 (Sep 14): home_goals_scored_L5 = avg([Match 1-5])
Match 10 (Oct 5): home_goals_scored_L5 = avg([Match 5-9])
```

This feature is designed to **data leakage** so that the model never sees future information when making predictions.


### Odds Movement Data

Odds movement features require both opening and closing odds:
- **Opening odds**: First published odds (typically 3-7 days before match)
- **Closing odds**: Final odds just before kickoff (captures all late information)
- **Data source**: football-data.co.uk provides both opening (Avg*) and closing (AvgC*) odds

**Timeline example**:


### Feature Interactions

Some features are **derived** from others:
- `elo_diff = HomeElo - AwayElo`
- `form_diff_L5 = home_ppg_L5 - away_ppg_L5`
- `home_gd_L5 = home_goals_scored_L5 - home_goals_conceded_L5`

These "difference" features help the model learn comparisons more easily than raw values.

### Multicollinearity Concerns

Some features are **correlated by design**:
- `HomeExpected` and `elo_diff` (both derived from Elo ratings)
- `home_ppg_L5` and `home_goals_scored_L5` (winning teams score more)
- `odds_home_prob_norm` and `HomeExpected` (avg diff ~0.02, very close!)

**Impact**: Tree-based models (Random Forest, XGBoost) handle multicollinearity well. Feature importance analysis will show which version the model prefers.



## Feature Quality Validation

### Quick Sanity Checks

1. **Elo ratings correlation**:
   - `HomeExpected` vs `odds_home_prob_norm`: ~0.02 difference
   - Shows Elo is well-calibrated with market

2. **Rolling features reasonableness**:
   - `home_goals_scored_L5`: Mean ~1.5, $\sigma$ ~0.8
   - `home_ppg_L5`: Mean ~1.3, $\sigma$ ~0.9
   - Within expected football statistics

3. **Bookmaker margin**:
   - Average overround: 5.66%
   - Typical for major leagues (5-8%)

4. **No data leakage**:
   - First matches have NaN (correct)
   - Features update chronologically (verified)

### Known Limitations

1. **Elo Expected values**: Don't explicitly model draw probability (implicitly split 50/50)
   - Mitigation: Use betting odds features which explicitly model draws

2. **Rolling windows**: Incomplete for first 5 matches per team
   - Mitigation: Most matches (96%+) have sufficient history
   - Include multiple seasons in data

3. **Missing features**: No injury info, head-to-head history, rest days
   - Future enhancement: -> Can be added if baseline model works


---

## Next Steps

### Immediate Actions
1. ✅ Feature engineering complete
2. ✅ Documentation written
3. ✅ Build baseline Random Forest model
4. ✅ Evaluate feature importance
5. ⬜ Model optimisation
6. ⬜ Parameter tuning 
5. ⬜ Analyze prediction errors


---

## Appendix: Feature Calculation Examples

### Example 1: Real Madrid vs Barcelona

**Match date**: 2024-10-26

**Elo features**:
- HomeElo: 1650 (Real Madrid)
- AwayElo: 1625 (Barcelona)
- elo_diff: +25
- HomeExpected: 0.536
- AwayExpected: 0.464

**Rolling form** (Real Madrid - home, Barcelona - away):
- home_goals_scored_L5: 2.4
- away_goals_scored_L5: 2.2
- home_ppg_L5: 2.0
- away_ppg_L5: 2.2
- form_diff_L5: -0.2 (Barcelona slightly better form)

**Match features**:
- home_gd_L5: +1.2 (Real Madrid)
- away_gd_L5: +1.0 (Barcelona)

**Betting odds**:
- odds_home_prob: 0.45
- odds_draw_prob: 0.28
- odds_away_prob: 0.33
- odds_home_prob_norm: 0.425 (remove 5.7% margin)
- odds_elo_diff: 0.425 - 0.536 = -0.111 (Elo more optimistic about Real Madrid)

> Elo favors Real Madrid, but betting market and recent form favor Barcelona. Close match with high draw probability.

---

## Contact & Updates

For questions or suggestions about feature engineering:
- GitHub Issues: [footAI repository]
- This doc will be updated with changes

**Last updated**: 2025-11-08  
**Version**: 0.2.0  
**Author**: [@pmatorras](https://github.com/pmatorras)
