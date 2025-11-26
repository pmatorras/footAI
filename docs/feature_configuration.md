# Feature Configuration & Model Selection Documentation

**Project:** footAI - Football Match Prediction  
**Date:** November 18, 2025  
**Author:** Pablo Matorras  
**Purpose:** Document systematic feature engineering experiments and model selection rationale

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Phase 1: Foundation Analysis](#phase-1-foundation-analysis)
- [Phase 2: Draw Feature Optimization](#phase-2-draw-feature-optimization)
- [Phase 3: Fouls and League specific features](#phase-3-failed-experiments)
- [Phase 4: Temporal features (momentum)](#phase-4-temporal-features-momentum)
- [Phase 5: Market dynamics](#phase-5-market-dynamics)
- [Future work](#future-work-feature-engineering-on-gradient-boosting-models)
- [Version history](#version-history)
---




## Executive Summary

Systematic feature engineering experiments across 5 phases to optimize the accuracy and draw prediction across the top 5 European leagues (2015-2025).

Two possible **Final Production Models** (Random Forest):
- `odds_lite`: 25 features, 50.42% acc, 38.08% DR
- `odds_optimized`: 28 features, 50.57% acc, 38.17% DR

**Total improvement**: +3.73-3.82% DR from baseline (34.35% → 38.08-38.17%)

>**Update v1.1 (Elo Logic Refinement)**:
>The underlying calculation for Elo features (`HomeElo`, `AwayElo`, `abs_elo_diff`, etc.) was refined in v1.1 to include continuous rating transfer for promoted/relegated teams (with 0.95 decay). \
>**Decision**: The feature sets defined below (`odds_optimized`, `odds_lite`) remain valid and unchanged. The definitions of the features persist, but the values fed into the model have been improved. Feature selection was not re-run as the fundamental predictive power of these categories is unchanged

The strategy was divided in different  phases:
```
Phase 1: Foundation Selection
├─ baseline (12 features)
└─ extended (18 features) → adds shots/GD

Phase 2: Draw Features on Winner
├─ draw_lite (5 features)
├─ draw_optimized (8 features)
└─ draw_full (15 features)

Phase 3: Context Features
├─ 3A: Fouls (failed)
└─ 3B: League (success)

Phase 4: Temporal Features
└─ 4A: Momentum (marginal)

Phase 5: Market Dynamics
├─ 5A: Odds Movement (success)
├─ 5B: Corners (marginal)
└─ 5C: Interactions (failed on RF)
```

**Status**: Feature engineering complete for RF.

---

## Phase 1: Foundation Analysis

> **Hypothesis**: Extended features (shots, shot accuracy, goal differentials) should improve performance by capturing attacking quality beyond raw goals scored.

### Models Tested

| Model | Features | Accuracy | Draw Recall | 
|-------|----------|----------|-------------|
| **baseline** | 12 | 50.18% | 34.35% | 
| **extended** | 18 | 50.16% | 35.03% |

**Extended adds:**
- `home_shots_L5`, `away_shots_L5`
- `home_shot_accuracy_L5`, `away_shot_accuracy_L5`  
- `home_gd_L5`, `away_gd_L5`

### Results

**Foundation Effect:** extended → baseline
- Accuracy: -0.02% (marginally worse)
- Draw Recall: +0.68% (marginally better)
- Feature Importance: 6 new features contribute only 9.4% combined

### Decision

**Extended foundation rejected, take baseline (12 features)**: Baseline is simpler and performs equivalently

---

## Phase 2: Draw Feature Optimization

> **Hypothesis**: Draw-specific features (parity indicators, market signals, low-scoring indicators) should improve draw prediction when added to the baseline foundation.

### Initial Approach: Legacy draw_lite

The original `draw_lite_legacy` was optimized for single-league (SP1) training in a previous iteration.

**Features** (exact composition unknown, but likely included):
- `draw_prob_consensus`-  Consensus draw prob across bookmakers
- `under_2_5_prob`-  Low-scoring indicator
- `draw_prob_dispersion`-  Market disagreement on draws
- `odds_draw_prob_norm`-  Normalized draw probability (B365)
- `asian_handicap_diff`-  Handicap-based parity

To try and improve the results of `draw_lite_legacy`, several experiments were made:
### Experiment 1: Full Draw Features
First, we tested the **complete set of engineered draw features** to establish the performance ceiling:

**Parity Indicators:**
- `abs_odds_prob_diff` - |odds_home - odds_away| (match closeness)
- `abs_elo_diff` - |elo_diff| (strength gap magnitude)
- `elo_diff_sq` - elo_diff² (non-linear parity effect)
- `low_elo_diff` - Binary: |elo_diff| < 25
- `medium_elo_diff` - Binary: 25 ≤ |elo_diff| < 50

**Market Signals:**
- `draw_prob_consensus` - Mean draw probability across bookmakers
- `draw_prob_dispersion` - Std dev of draw probabilities (market uncertainty)
- `under_2_5_prob` - Probability of under 2.5 goals
- `under_2_5_zscore` - Z-score of under 2.5 probability

**Low-Scoring Composites:**
- `min_shots_l5` - min(home_shots_L5, away_shots_L5)
- `min_shot_acc_l5` - min(home_shot_acc_L5, away_shot_acc_L5)
- `min_goals_scored_l5` - min(home_goals_scored_L5, away_goals_scored_L5)

**Asian Handicap Signals:**
- `abs_ahh` - |asian_handicap| (alternative parity measure)
- `ahh_zero` - Binary: handicap near 0 (±0.25)
- `ahh_flat` - Binary: handicap exactly 0

**Result:** `baseline_draw_full` (27 features)
- Accuracy: 50.36%
- Draw Recall: 37.15%
- **Performance ceiling established**

The next step was to try and find the minimal subset that captures most of the performance., for which the feature importance from `baseline_draw_full` was studied to identify high-impact features:


```
1. abs_odds_prob_diff      7.48%  ← NEW (not in legacy)
2. draw_prob_consensus     6.58%  ← In legacy
3. elo_diff_sq             3.84%  ← NEW (not in legacy)
4. abs_elo_diff            3.65%  ← NEW (not in legacy)
5. draw_prob_dispersion    1.01%  ← In legacy
6. min_shot_acc_l5         0.95%  ← NEW (not in legacy)
7. abs_ahh                 0.69%  ← NEW (related to asian_handicap_diff)
8. min_shots_l5            0.67%  ← NEW (not in legacy)
9. under_2_5_prob          0.58%  ← In legacy
10. under_2_5_zscore       0.51%  ← Drop (redundant)
11. min_goals_scored_l5    0.36%  ← NEW (not in legacy)
12. low_elo_diff           0.11%  ← Drop (redundant)
13. medium_elo_diff        0.04%  ← Drop (redundant)
14. ahh_zero               0.07%  ← Drop
15. ahh_flat               0.01%  ← Drop (noise)

Legacy features not in top list:

- odds_draw_prob_norm      ~0.3%  ← Likely redundant with draw_prob_consensus
- asian_handicap_diff      ~0.5%  ← Replaced by abs_ahh
```
### Experiment 2: Feature Sets including draw

Compared two subsets of `baseline_draw_full` to the full version:

**DRAW_CORE** - **replacement for legacy**

 - `abs_odds_prob_diff`      -  NEW - Odds parity (7.48%)
 - `draw_prob_consensus`     -  From legacy (6.58%)
 - `elo_diff_sq`             -  NEW - Non-linear parity (3.84%)
 - `abs_elo_diff`            -  NEW - Elo parity (3.65%)
 - `draw_prob_dispersion`    -  From legacy (1.01%)

**Changes from legacy:**
- Kept: `draw_prob_consensus`, `draw_prob_dispersion` (top market signals)
- Dropped: `under_2_5_prob`, `odds_draw_prob_norm`, `asian_handicap_diff`
- Added: `abs_odds_prob_diff`, `elo_diff_sq`, `abs_elo_diff` (parity indicators)

**Rationale:** Legacy focused on market signals but missed strength-gap parity features that are crucial for draw prediction.

**DRAW_OPTIMIZED**: defined as `DRAW_CORE` but including three aditional features:

 - `min_shot_acc_l5`         -  Defensive quality (0.95%)
 - `abs_ahh`                 -  Asian handicap parity (0.69%)
 - `under_2_5_prob`          -  From legacy (0.58%)



### Results

| Configuration | Features | Accuracy | Draw Recall | vs Legacy | Notes |
|--------------|----------|----------|-------------|-----------|-------|
| **baseline** (control) | 12 | 50.18% | 34.35% | - | No draw features |
| **baseline_draw_lite_legacy** | 17 | 50.36% | 36.39% | - | SP1-optimized |
| **baseline_draw_lite** | 17 | 50.49% | 36.39% | +0.13% acc | **Based on multiple countries** |
| **baseline_draw_optimized** | 20 | 50.36% | 36.90% | +0.51% DR | +3 features over lite |
| **baseline_draw_full** | 27 | 50.36% | 37.15% | +0.76% DR | Performance ceiling |

**Key findings:**

1. **Legacy was good**: +2.04% increase in draw recalls over baseline (36.39% vs 34.35%)
2. **New lite improves accuracy**: Same 5 features, data-driven selection → +0.13% accuracy
3. **Parity features matter**: Top 3 new features (abs_odds_prob_diff, elo_diff_sq, abs_elo_diff) contribute 14.97% importance

### Decision

Systematic testing identified two strong candidates:

1. **baseline_draw_lite** (17 features): 50.49% acc, 36.39% DR
2. **baseline_draw_optimized** (20 features): 50.36% acc, 36.90% DR

**Trade-off**:
- draw_lite: Best accuracy, simplest (17 features)
- draw_optimized: Best draw recall, +3 features

> **Verdict**: Both models outperform legacy. Final selection deferred to Phase 4 (after testing temporal features).

---

## Phase 3: Failed Experiments

### A. Fouls/Faults Features

> **Hypothesis**: Foul patterns indicate league style and match physicality. Physical matches with high fouls lead to more disruption, cards, and tactical draws.

**Rationale**: Different leagues have different officiating styles:
- Premier League: High-tempo, physical, more fouls tolerated
- La Liga: Technical, fewer fouls, stricter officiating
- Serie A: Tactical fouls as defensive strategy

**Expected benefit**: Capturing physicality/disruption as a draw predictor.

#### Features Tested

- home_fouls_L5 -  Average fouls committed (home, last 5)
- away_fouls_L5 - Average fouls committed (away, last 5)
- foul_diff_L5 -  Aggression differential

### Performance

| Metric | baseline_draw_optimized | faults | Change |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.36% | 50.03% | **-0.32%**  |
| **Draw Recall** | 36.90% | 36.30% | **-0.59%** |
| **Features** | 20 | 23 | +3 |

### Feature Importance

- `foul_diff_L5`: 0.31% (rank 21/23)
- `home_fouls_L5`: 0.18% (rank 22/23)
- `away_fouls_L5`: 0.17% (rank 23/23)
- **Combined**: 0.66% (bottom 3 features)

> **Verdict**: They show a weak signal, and its and no improvement (probably due to its information of match style described by other features, like elo ratings) -> rejected



### B. League-Specific Features


> **Hypothesis**:  Adding league-level aggregates improves cross-league generalization.

**Features tested:**
To test this, the `baseline_draw_optimized` was used, adding:
- `league_draw_bias` - Historical league draw rate
- `home_draw_rate_l10` - Team's rolling draw rate
- `away_draw_rate_l10` - Opponent's rolling draw rate

### Results

| Metric | baseline_draw_optimized | league_adaptive | Change |
|--------|------------------------|-----------------|---------|
| **Overall Accuracy** | 50.36% | 50.33% | **-0.02%** |
| **Overall Draw Recall** | 36.90% | **37.57%** | **+0.68%** |

**Feature Importance**:
- `league_draw_bias`: 0.34% (rank 21/23)
- `home_draw_rate_l10`: 0.29% (rank 22/23)
- `away_draw_rate_l10`: 0.30% (rank 23/23)
- **Combined**: 0.95% (bottom 3 features)




> **Verdict**: The inclusion of league specific features showed a significant improvement on average the draw recall -> included into the features.


## Phase 4: Temporal Features (Momentum)

> **Hypothesis**: Recent trajectory (improving/declining form) predicts outcomes better than static averages. Two teams with same average form (1.5 goals/match) behave differently:
- Team A: 1.0 → 1.5 → 2.0 (improving, confidence high)
- Team B: 2.0 → 1.5 → 1.0 (declining, morale low)

### Rationale

Current rolling averages (`goals_scored_L5`, `ppg_L5`) capture magnitude but not direction. Linear slope over last 5 matches captures momentum.

### Features Tested

 - `home_goals_trend_L5` -  Goals trajectory (linear slope, last 5)
 - `away_goals_trend_L5` -  Goals trajectory (away)
 - `home_ppg_trend_L5` -  Form trajectory (home)
 - `away_ppg_trend_L5` -  Form trajectory (away)
 - `momentum_diff` -  home_ppg_trend - away_ppg_trend


 **Test configurations**:
- `momentum_lite` = `draw_lite`+`league_adaptive` + `momentum` (25 features)
- `momentum_optimized` = `draw_optimized`+ `league_adaptive` + `momentum` (28 features)

### Results

| Metric | league_lite | momentum_lite | Change |
|--------|-------------|---------------|--------|
| **Accuracy** | 50.33% | 50.01% | -0.32% |
| **Draw Recall** | 36.90% | 36.30% | **-0.59%** |
| **Features** | 20 | 25 | +5 |

| Metric | league_adaptive | momentum_optimized | Change |
|--------|-----------------|-------------------|---------|
| **Accuracy** | 50.33% | 50.27% | -0.06% |
| **Draw Recall** | 37.57% | 37.91% | **+0.34%** |
| **Features** | 23 | 28 | +5 |

**Feature Importance** (momentum_optimized):
- `away_goals_trend_L5`: 0.77% (rank 17/28)
- `home_ppg_trend_L5`: 0.54% (rank 21/28)
- `away_ppg_trend_L5`: 0.54% (rank 22/28)
- `home_goals_trend_L5`: 0.54% (rank 23/28)
- `momentum_diff`: 0.53% (rank 24/28)
- **Combined**: 2.92% (bottom third of features)

**Per-league** (momentum_optimized vs league_adaptive):
| League | $\Delta$ Draw Recall |
|--------|--------------|
| D1 (Bundesliga) | +0.99%  |
| E0 (Premier League) | -0.44%  |
| FR1 (Ligue 1) | -1.55% |
| I1 (Serie A) | +1.37%  |
| SP1 (La Liga) | +0.75%  |

**League wins**: 3/5

> **Verdict**:  Not recomended, as the increase of 0.34 Draw Rate with no accuracy improvement isn't significant enough to justify the inclusion of 5 extra features

---

## Phase 5: Market Dynamics
### 5A: Odds Movement Features


> **Hypothesis**: Odds movement (opening → closing) captures late information arrival: injuries, weather, sharp money positioning. Odds shortening on draw outcome indicates smart money sees value.

### Features Tested

 - ` draw_odds_drift` -  (ClosingD - OpeningD) / OpeningD
 - ` home_odds_drift` -  (ClosingH - OpeningH) / OpeningH
 - ` away_odds_drift` -  (ClosingA - OpeningA) / OpeningA
 - ` sharp_money_on_draw` -  Binary: draw odds shortened > 2%
 - ` odds_movement_magnitude` -  abs(draw_odds_drift)

 **Test configurations**:
- `odds_lite` = `draw_lite`+`league_adaptive` + `odds_movement ` (25 features)
- `odds_optimized` = `draw_optimized`+ `league_adaptive` + `odds_movement ` (28 features)

### Results

| Metric | league_lite | odds_lite | Change |
|--------|-------------|-----------|--------|
| **Accuracy** | 50.33% | 50.42% | +0.09% |
| **Draw Recall** | 36.90% | 38.08% | **+1.19%** |
| **Features** | 20 | 25 | +5 |

| Metric | league_adaptive | odds_optimized | Change |
|--------|-----------------|----------------|---------|
| **Accuracy** | 50.33% | 50.57% | +0.24% |
| **Draw Recall** | 37.57% | 38.17% | **+0.59%** |
| **Features** | 23 | 28 | +5 |

**Feature Importance**: 1.78-2.16% combined (low individual, high impact via interactions)

**Per-league** (odds_lite): 5/5 leagues improved  
**Per-league** (odds_optimized): 4/5 leagues improved

> **Verdict** The increase in accuracy and draw recall is high enough to justify the additional features.

### Phase 5B: Corners Features (Mixed Results)

> **Hypothesis**: High corners with low goals indicates defensive matches and draws. Corners capture attacking intent without finishing quality.

### Features Tested

- `corners_ratio` - home_corners_L5 / away_corners_L5 (parity)
- `defensive_draw_signal` - avg_corners * under_2_5_prob


**Test configurations**:

- `corners_lite` = odds_lite + corners (25 + 2 = 27 features)
- `corners_optimized` = odds_optimized + corners (28 + 2 = 30 features)


### Results

| Metric | odds_lite | corners_lite | Change |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.42% | 50.40% | -0.02% |
| **Draw Recall** | 38.08% | 37.66% | **-0.42%** |
| **Features** | 25 | 27 | +2 |

| Metric | odds_optimized | corners_optimized | Change |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.57% | 50.44% | -0.13% |
| **Draw Recall** | 38.17% | 38.68% | **+0.51%** |
| **Features** | 28 | 30 | +2 |

**Feature Importance**:

- `corners_lite`: `corners_ratio` 2.43% (rank 10/27), `defensive_draw_signal` not used
- `corners_optimized`: `corners_ratio` 3.66% (rank 8/30), `defensive_draw_signal` not used

**Per-league** (`corners_optimized`): 2/5 wins (E0: +2.65%, SP1: +0.75%)

> **Verdict**: Marginal improvement only for the optimized but regresses on lite; and not consistent among leagues


***

## Phase 5C: Interaction Features
> **Hypothesis**: Explicit feature interactions help gradient boosting models and capture non-linear relationships. Random Forest creates these implicitly, but XGBoost/LightGBM benefit from explicit terms.

### Features to Test

**INTERACTION_FEATURES** (4 features):


- `elo_odds_agreement` - (elo_diff/400) * (odds_H - odds_A)
- `form_odds_weighted` - form_diff * abs_odds_prob_diff
- `parity_uncertainty` - (1/(1+abs_elo)) * draw_dispersion
- `movement_parity_signal` - draw_drift * (1 - abs_odds_diff)


**Test configurations**:

- `interactions_lite` = odds_lite + interactions (29 features)
- `interactions_optimized` = odds_optimized + interactions (32 features)

### Results

| Metric | odds_lite | interactions_lite | Change |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.42% | 50.36% | -0.06% |
| **Draw Recall** | 38.08% | 37.66% | **-0.42%** |
| **Features** | 25 | 29 | +4 |

| Metric | odds_optimized | interactions_optimized | Change |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.57% | 50.18% | -0.39% |
| **Draw Recall** | 38.17% | 36.81% | **-1.36%** |
| **Features** | 28 | 32 | +4 |

**Feature Importance**:

- interactions_lite: 9.33% combined (8th-27th rank)
- interactions_optimized: 8.75% combined (8th-31st rank)

**Individual features** (interactions_optimized):

- `elo_odds_agreement`: 4.16% (rank 8)
- `form_odds_weighted`: 3.20% (rank 9)
- `parity_uncertainty`: 1.17% (rank 13)
- `movement_parity_signal`: 0.21% (rank 31)

**Per-league**:

- interactions_lite: 3/5 wins (but overall -0.42% DR)
- interactions_optimized: 0/5 wins, -1.36% DR

> **Verdict**: It doesn't improve performance, since the random forest already captures this interactions implicitly, resulting in redundancy and overvitting

> Might be a useful feature for gradient boosting (XGBoost/LightGBM) 

---
## Final Production Model

### Choice: **odds_lite vs odds_optimized** (TIE)

Both models perform identically in draw recall (38.08% vs 38.17% = 0.08% difference).


| Metric | odds_lite | odds_optimized | Difference |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 50.42% | 50.57% | +0.15% |
| **Draw Recall** | 38.08% | 38.17% | +0.08% |
| **Features** | 25 | 28 | +3 |

**Per-league**: odds_optimized wins 2/5 DR, 3/5 accuracy (mixed)


**Decision criteria**:

- **Prefer simplicity**: Use `odds_lite`
- **Prefer completeness**: Use `odds_optimized`
- **Switching models later**: Use `odds_optimized` (more features for gradient boosting)

**Total improvement from baseline**: +3.73-3.82% DR (34.35% → 38.08-38.17%)

**Rejected features**: Corners (marginal), Interactions (regression on RF), Momentum (marginal), Fouls (regression)


## Future Work: Feature engineering on Gradient Boosting Models


Interaction features failed on Random Forest (-1.36% DR) due to RF's implicit interaction handling. However, gradient boosting models (XGBoost/LightGBM) require explicit interactions. When testing model optimisation, the interaction features will be considered to assess whether they produce an improvement, and will be included if the boosted model is found to improve the current RF results.


## Version History

| Date | Version | Model | Performance | Notes |
|------|---------|-------|-------------|-------|
| 2025-11-18 | v1.0 | baseline_draw_optimized | 50.36% / 36.90% | Initial production model |
| 2025-11-19 | v2.0 | odds_lite | 50.42% / 38.08% | Simpler model |
| 2025-11-19 | v2.0 | odds_optimized | 50.57% / 38.17% | Complete model |

---

**Document maintained by:** Pablo Matorras  
**Last updated:** November 19, 2025  
**Next review:** After model optimisation
