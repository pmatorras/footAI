# Model Architecture Decisions

This document captures key lessons, best practices, and pitfalls discovered during the development of the football match prediction models.

---

## Why Binary Classification Should Not Be Used for Football Outcome Prediction

### Problem Statement

Football matches naturally result in three outcomes: Home win (H), Draw (D), and Away win (A). It may seem appealing to simplify this to a binary classification problem (H vs A only) by filtering out draws from the training data.

**This approach is not recommended** for the following reasons:

### 1. Distribution Shift

Removing draws (~25% of matches) creates an artificial dataset that doesn't represent real-world conditions. The model trains on only decisive outcomes, learning patterns from a biased subset of matches. When deployed, it encounters draw-prone matches it was never trained to recognize.

### 2. Loss of Draw Awareness

Matches that end in draws often have feature patterns similar to close H or A outcomes (balanced Elo ratings, tight odds, low-scoring expectations). A model trained without exposure to these patterns:
- Cannot recognize when a match is likely to be a draw
- Is forced to choose H or A with false confidence
- Produces poorly calibrated predictions on borderline cases

### 3. Inflated Performance Metrics

Accuracy measured on filtered test sets (excluding draws) appears artificially high because:
- The model is tested on the same simplified distribution it trained on
- It never faces the harder task of distinguishing draws from decisive outcomes
- Real-world performance degrades when draws are present

### 4. Model Bias

Binary models trained on filtered data often develop **home bias**, over-predicting home wins because:
- They optimize for the easier H vs A distinction
- Without draw outcomes to moderate predictions, they skew toward the majority class (home wins)
- This leads to systematic errors, especially on away predictions

### Recommended Approach

**Always train on the full three-class distribution (H/D/A):**
- Use proper class weighting (`class_weight='balanced'`) to handle class imbalance
- Include draw-specific features (odds, goal expectations, team parity indicators)
- Evaluate performance on all three classes to ensure robustness

**If binary H/A predictions are needed:**
- Train the full 3-class model
- At evaluation time, filter to only non-draw outcomes: `mask = (y_test != 'D')`
- Calculate H/A metrics only on matches where actual outcome was H or A
- This preserves proper training while allowing binary evaluation



## Model Specialization Strategy: Per-Tier vs Combined Training

### Context

Football leagues vary in quality and predictability. We investigated whether models should specialize by:

- Individual league (5 separate models for tier1: Bundesliga, Premier League, Serie A, La Liga, Ligue 1)
- Tier level (tier1 vs tier2)
- Combined across all leagues


### Experimental Results
The averaged results were found to be:
| Strategy | Tier1 Accuracy | Tier1 Draw Recall | Tier2 Accuracy | Tier2 Draw Recall |
| :-- | :-- | :-- | :-- | :-- |
| Per-league specialists | 50.9%  | 31.4% | 43.2% | 21.2% |
| Tier1 specialist | 50.3% | **37.7%** | - | - |
| Tier2 specialist | - | - | 43.7% | 19.5% |
| Multi-country (tier1+tier2) | 51.9% | 24.7% | 42.8% | **39.6%** |

### Key Findings

**1. Per-League Specialization Not Worth It**

- Per-league specialists: 50.9% average accuracy
- Tier1 combined: 50.4% accuracy (only -0.5%)
- **Verdict**: 5 models vs 1 model not justified for 0.5% gain

**2. Tier2 Has Inherently Lower Predictability**

Tier2 leagues are fundamentally harder to predict than tier1:
 - Smaller skill gaps: Differences between best and worst teams are smaller
 - Higher competitive balance: More evenly matched teams lead to closer matches
 - **Result**: Both lower overall accuracy and worse draw recall

Performance comparison:

- Tier1 combined: 50.3% accuracy, 37.7% draw recall
- Tier2 combined: 43.7% accuracy, 19.5% draw recall

**3. Critical Tier2 Class Imbalance Problem**

- Tier2 specialist has only 19.5% draw recall
- Misses 4 out of 5 draws (~25% of matches)
- Multi-country model improves draw recall to 39.6% (+20%)
- Small accuracy cost (-0.9%) is acceptable for doubling draw detection

**4. Tier1 vs Multi-Country Trade-Off**

- Multi-country improves tier1 accuracy (+1.6%)
- BUT destroys tier1 draw recall (-13%)
- Tier1 specialist already has balanced performance (37.7% draw recall)


### Recommended Architecture: Asymmetric Routing

**Deploy 2 models with tier-based routing:**

```python
def predict(match):
    if match['division'] in ['tier1', 'D1', 'E0', 'I1', 'SP1', 'FR1']:
        return tier1_specialist.predict(match)
    else:
        return multi_country_model.predict(match)
```

**Rationale:**

- **Tier1**: Use specialist to preserve balanced class performance (50.3% acc, 37.7% draw recall)
- **Tier2**: Use multi-country to fix critical draw recall problem (42.8% acc, 39.6% draw recall)


### Alternative Considered: Nested Routing for Tier1

**Proposal**: Route tier1 matches by draw probability

- High P(Draw) → tier1 specialist (better draw recall)
- Low P(Draw) → multi-country (better accuracy)

**Analysis:**

- Estimated gain: ~1% accuracy
- Cost: 2 models for tier1, routing calibration, complex debugging
- **Verdict**: Not worth operational complexity for marginal benefit

**Decision**: Keep simple tier-based routing. Focus on hyperparameter tuning for larger gains.