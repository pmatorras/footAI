# Feature Engineering Documentation
**footAI v0.2 - ML Predictions**

Generated: 2025-11-08  
Dataset: Spanish La Liga (2024-25 Season)

---

## Table of Contents
1. [Overview](#overview)
2. [Feature Categories](#feature-categories)
3. [Elo-Based Features](#elo-based-features)
4. [Rolling Form Features](#rolling-form-features)
5. [Match-Level Features](#match-level-features)
6. [Betting Market Features](#betting-market-features)
7. [Technical Details](#technical-details)
8. [ML Model Expectations](#ml-model-expectations)

---

## Overview

This document describes all engineered features used for match outcome prediction (Home/Draw/Away). Features are calculated with **temporal correctness** - for any match N, features only use data from matches 1 to N-1, to prevent data leakage.

### Feature Count Summary
- **Elo features**: 5 (HomeElo, AwayElo, HomeExpected, AwayExpected, elo_diff)
- **Rolling form features**: 20 (10 per window size: 3 mathces (L3), or five matches (L5))
- **Match-level features**: 5 (form_diff_L5, home_gd_L5, away_gd_L5, is_home, odds_elo_diff)
- **Betting market features**: 6 (probabilities and normalized values)
- **Total engineered features**: ~36 core features

---

## Feature Categories

### 1. Elo-Based Features

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

### 2. Rolling Form Features

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

### 3. Match-Level Features

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

---

### 4. Betting Market Features

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

## Technical Details

### Temporal Correctness

**Critical principle**: Features for match N use only data from matches 1 to N-1.

Example timeline for Real Madrid:
```
Match 1 (Aug 18): home_goals_scored_L5 = NaN (no history)
Match 2 (Aug 25): home_goals_scored_L5 = avg([Match 1])
Match 3 (Aug 29): home_goals_scored_L5 = avg([Match 1, 2])
Match 6 (Sep 14): home_goals_scored_L5 = avg([Match 1-5])
Match 10 (Oct 5): home_goals_scored_L5 = avg([Match 5-9])
```

This feature is designed to **data leakage** so that the model never sees future information when making predictions.

### Handling Missing Values

**First match strategy**: Set rolling features to NaN (no history available)

**Trade-offs**:
1. **Current approach**: Calculate features from match 2+ using available history
   - Pros: Retains 96%+ of data
   - Cons: Early matches have incomplete windows (1-4 games instead of 5)

2. **Alternative**: Require full window (drop first 5 matches per team)
   - Pros: All features have complete 5-match history
   - Cons: Loses ~25% of training data

3. **Alternative**: Carry over from previous league data
   - Pros: No NaN values
   - Cons: Tricky with promotions and relegations


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

---

## ML Model Expectations

### Performance Benchmarks

#### Baseline Accuracies
- **Random guessing**: 33.3% (equal probability H/D/A)
- **Always predict home win**: ~45% (reflects home advantage)
- **Follow betting market**: ~50-53% (market is efficient)

#### Target Accuracies
- **Phase 1 (Baseline)**: 50%+ (validates approach)
- **Phase 2 (Competitive)**: 52-54% (matches market)
- **Phase 3 (Profitable)**: 55%+ (potential betting edge)

### Profitability Context

**To beat the bookmaker margin** (5.66% in one season of laliga):
- Need to identify **value bets** where model probability > market probability by 25%+
- Even 52% overall accuracy can be profitable if you only bet on high-confidence matches
- Most successful strategies bet on <10% of matches


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

4. **Single season**: Limited training data (~350 matches)
   - Future enhancement: Multi-season features in Phase 2

---

## Next Steps

### Immediate Actions
1. ✅ Feature engineering complete
2. ✅ Documentation written
3. ⬜ Build baseline Random Forest model
4. ⬜ Evaluate feature importance
5. ⬜ Analyze prediction errors

### Future Enhancements (Phase 2)
- Multi-season feature engineering
- Home/away performance splits
- Head-to-head history (last 3 meetings)
- Rest days between matches
- Season progress (early vs late season effects)

### Possible questions to Investigate
- Do longer rolling windows (L7, L10) improve predictions?
- Is shot accuracy more predictive than total shots?
- Can we identify specific match types where model excels/fails?
- Which features are redundant and can be removed?

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
