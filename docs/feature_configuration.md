# Feature Configuration & Model Selection Documentation

**Project:** footAI - Football Match Prediction  
**Date:** November 18, 2025  
**Author:** Pablo Matorras  
**Purpose:** Document systematic feature engineering experiments and model selection rationale

---

## Table of Contents

- [Executive Summary](#executive-summary)
-  [Phase 1: Foundation Analysis](#phase-1-foundation-analysis)
-  [Phase 2: Draw Feature Optimization](#phase-2-draw-feature-optimization)
-  [Phase 3: Fouls and League specific features](#phase-3-failed-experiments)
-  [Current status and future steps](#current-status--next-steps)



---

## Executive Summary

Systematic feature engineering experiments to optimize the accuracy and draw prediction across the top 5 European leagues (2015-2025).

The strategy was divided in different  phases:
```
Phase 1: Foundation Selection
├─ baseline (12 features)
└─ extended (18 features) → adds shots/GD

Phase 2: Draw Features on Winner
├─ draw_lite (5 features)
├─ draw_optimized (8 features)
└─ draw_full (15 features)

Phase 3: League-Specific Features
└─ league_adaptive (3 features)
```

**Status**: Phase 1-3 complete. baseline_draw_lite and baseline_draw_optimized are leading candidates.

**Next**: Test temporal features (momentum, rest) to determine final production model.

---

## Phase 1: Foundation Analysis

### Hypothesis
Extended features (shots, shot accuracy, goal differentials) should improve performance by capturing attacking quality beyond raw goals scored.

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

### Hypothesis
Draw-specific features (parity indicators, market signals, low-scoring indicators) should improve draw prediction when added to the baseline foundation.

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
4. **Sweet spot**: Optimized (8 features) captures 94% of full performance



### Decision

**Phase 2 Complete**: Systematic testing identified two strong candidates:

1. **baseline_draw_lite** (17 features): 50.49% acc, 36.39% DR
2. **baseline_draw_optimized** (20 features): 50.36% acc, 36.90% DR

**Trade-off**:
- draw_lite: Best accuracy, simplest (17 features)
- draw_optimized: Best draw recall, +3 features

**Status**: Both models outperform legacy. Final selection deferred to Phase 4 (after testing temporal features).

---

## Phase 3: Failed Experiments

### A. Fouls/Faults Features

**Hypothesis**: Foul patterns indicate league style and match physicality. Physical matches with high fouls lead to more disruption, cards, and tactical draws.

**Rationale**: Different leagues have different officiating styles:
- Premier League: High-tempo, physical, more fouls tolerated
- La Liga: Technical, fewer fouls, stricter officiating
- Serie A: Tactical fouls as defensive strategy

**Expected benefit**: Capturing physicality/disruption as a draw predictor.

#### Features Tested

**FAULTS_FEATURES** (3 features):

- home_fouls_L5' -  Average fouls committed (home, last 5)
- away_fouls_L5' - Average fouls committed (away, last 5)
- foul_diff_L5' -  Aggression differential

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

#### Verdict: **REJECTED**

**Why it failed:**

**Why we thought it would work**:
- League style indicators (physical EPL vs technical La Liga)
- Disruption hypothesis (high fouls → broken play → draws)
- Featured in some literature on match outcomes

**Why it actually didn't**:
- **Weak signal**: Fouls don't predict results, they describe match style
- **Redundant**: League style already in odds and Elo
- **Wrong granularity**: Team fouls don't capture referee strictness (varies by official)

### B. League-Specific Features


**Hypothesis**: Adding league-level aggregates improves cross-league generalization.

**Features tested:**
To test this, the `baseline_draw_optimized` was used, adding:
- `league_draw_bias` - Historical league draw rate
- `home_draw_rate_l10` - Team's rolling draw rate
- `away_draw_rate_l10` - Opponent's rolling draw rate

### Results

| Metric | baseline_draw_optimized | league_adaptive | Change |
|--------|------------------------|-----------------|---------|
| **Overall Accuracy** | 50.36% | 50.29% | **-0.06%** |
| **Overall Draw Recall** | 36.90% | 36.98% | **+0.08%** |

**Feature Importance**:
- `away_draw_rate_l10`: 0.30% (rank 21/23)
- `home_draw_rate_l10`: 0.29% (rank 22/23)
- `league_draw_bias`: 0.22% (rank 23/23)
- **Combined**: 0.82% (bottom 3 features)

**Per-League Performance**:
| League | Accuracy $\Delta$ | Draw Recall $\Delta$ |
|--------|-----------|--------------|
| D1 (Bundesliga) | +0.63%  | +1.98%  |
| E0 (Premier League) | +0.50%  | -0.44%  |
| FR1 (Ligue 1) | -0.60%  | -1.04%  |
| I1 (Serie A) | -1.00%  | -1.37%  |
| SP1 (La Liga) | +0.20%  | +1.51%  |


**Verdict**: the adidtional complexity didn't result in a significant increase on the accuracy or recall, nor in any of the leagues.

---
## Current Status & Next Steps

### Leading Candidates (After Phase 1-3)

| Model | Features | Accuracy | Draw Recall | Trade-off |
|-------|----------|----------|-------------|-----------|
| `baseline_draw_lite` | 17 | 50.49% | 36.39% | Best accuracy, simplest |
| `baseline_draw_optimized` | 20 | 50.36% | 36.90% | +0.51% DR, -0.13% acc |

**Observations**:
- Accuracy peaks at `baseline_draw_lite` (simplest model)
- Draw recall improves with features, but diminishing returns

**Decision deferred**: Final model selection will be made after Phase 4 (temporal features).

### Future steps

Test temporal features on **both** `baseline_draw_lite` and `baseline_draw_optimized`:
1. Momentum features 
2. Rest/fixture density features
3. Compare final results to choose production model

**Why test both baselines?**
- draw_lite: Better starting accuracy, simpler
- draw_optimized: Better starting DR, proven draw features
- Unknown which will benefit more from temporal features

## Version History

| Date | Version | Model | Performance | Notes |
|------|---------|-------|-------------|-------|
| 2025-11-18 | v1.0 | baseline_draw_optimized | 50.36% / 36.90% | Initial production model |

---

**Document maintained by:** Pablo Matorras  
**Last updated:** November 18, 2025  
**Next review:** After (momentum/rest features)
