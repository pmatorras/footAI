# Model Selection Study
**footAI v0.3 - Phase 6 Results**

Date: 2025-11-20  
Dataset: Top 5 European Leagues (2015-2025, ~8000 matches)  
Objective: Maximize draw recall while maintaining reasonable accuracy

---

## Executive Summary

**Winner**: Random Forest with `class_weight='balanced'`  
**Draw Recall**: 38.17% (vs. 34.35% baseline)  
**Key Finding**: All gradient boosting variants fail catastrophically on minority class prediction

---

## Tested Models

### 1. Random Forest: Current PRODUCTION MODEL

**Configuration**:
```
RandomForestClassifier(
n_estimators=100,
max_depth=10,
min_samples_split=20,
min_samples_leaf=5,
class_weight='balanced',
random_state=42
)
```

**Results**:
| Feature Set | Accuracy | Draw Recall | Verdict |
|-------------|----------|-------------|---------|
| baseline (12) | 50.30% | 35.50% | Baseline |
| odds_lite (**25**) | 50.42% | 38.08% | Almost as good, less complexity |
| odds_optimized (28) | 50.57% | **38.17%** | Best performance |
| interactions (32) | 50.36% | 36.81% | Regression |

### 2. Gradient Boosting (sklearn)

**Configurations Tested**:

**Shallow** (conservative):

```
GradientBoostingClassifier(
n_estimators=50,
max_depth=3,
learning_rate=0.01,
subsample=0.5
)
```

**Deep** (aggressive):
```
GradientBoostingClassifier(
n_estimators=100,
max_depth=10,
learning_rate=0.05,
subsample=0.8
)
```

**Results**:
| Config | Accuracy | Draw Recall | Predicted Draws |
|--------|----------|-------------|-----------------|
| Shallow | 52.49% | **0.00%** | 0 / 1179 |
| Deep | 52.60% | 11.54% | 422 / 1179 |

**Analysis**:
- **Shallow GB**: Predicts ZERO draws (catastrophic failure)
- **Deep GB**: Predicts only 36% of expected draws (422 vs 1179 actual)
- Higher accuracy comes from avoiding draws, not better modeling

---

### 3. XGBoost

**Configuration**:
```
XGBClassifier(
n_estimators=100,
max_depth=10,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
reg_alpha=0.1,
reg_lambda=1.0
)
```

**Results**:
| Feature Set | Accuracy | Draw Recall | Predicted Draws |
|-------------|----------|-------------|-----------------|
| odds_lite | 53.18% | 8.82% | 336 / 1179 |
| interactions | 52.77% | 8.82% | Same |

**Analysis**:
- Predicts only 28% of expected draws (336 vs 1179)
- Interactions have **ZERO impact** (designed for XGB, but useless)
- -29.26% DR vs Random Forest
---

### 4. LightGBM

**Configuration**:
```
LGBMClassifier(
n_estimators=100,
max_depth=10,
learning_rate=0.05,
subsample=0.8,
num_leaves=15,
min_child_samples=20
)
```

**Results**:
| Feature Set | Accuracy | Draw Recall | Verdict |
|-------------|----------|-------------|---------|
| odds_lite | 53.57% | 4.58% | Worst performer |
| interactions | 54.07% | 5.34% | Still terrible |

**Analysis**:
- **Worst draw recall** of all models tested
- Highest accuracy (54%) but useless for draw prediction
- -33.50% DR vs Random Forest

---

## Why Gradient Boosting & Neural Networks Failed

### Fundamental Algorithm Problem

1. **Sequential optimization**: Each iteration/epoch minimizes global loss
2. **Minority class neglect**: Draws (~25% of data) contribute less to overall loss
3. **Greedy learning**: Models learn "skip draws, maximize H/A accuracy"
4. **Sample weights insufficient**: Even division weights + class balancing can't overcome algorithmic bias
5. **Risk aversion**: Draws are inherently uncertain → models penalized heavily for wrong D predictions → safer to predict H/A


```
Model Performance (Draw Recall):
Random Forest:     ████████████████████████████████████████ 38.2%
Sklearn GB (deep): ███████████                              11.5%
XGBoost:           █████████                                 8.8%
Neural Network:    ██████                                    6.3%
LightGBM:          ████                                      4.6%
Sklearn GB:        [empty]                                   0.0%
```

**Pattern**: All loss-optimizing models (GB/XGB/LGBM/NN) predict <12% DR vs RF's 38%

### Confusion Matrix Comparison

**Random Forest (odds_lite)**:
```
Actual draws: 1179
Predicted draws: 1473
Correct draws: 449
DR: 38.08%
```

**XGBoost (odds_lite)**:
```
Actual draws: 1179
Predicted draws: 336  ← Only 28% of expected!
Correct draws: 104
DR: 8.82%
```

**Neural Network (odds_lite)**:
```
Actual draws: 1179
Predicted draws: 207 ← Only 18% of expected!
Correct draws: 74
DR: 6.28%
```
---

## Portfolio Strategy Considerations

### Two-Model Approach

The hypothesis is that match prediction could benefit from a **portfolio strategy**:

1. **Generalist model** (for draw prediction) → **RF confirmed best** (38.17% DR)
2. **Specialist model** (for H/A prediction when draw probability is low) → **under evaluation**

### Observed H/A Performance

When evaluated on H/A matches only (excluding draws from test set), the models showed:


| Model | H/A Accuracy (on H/A subset) | Overall Accuracy | Draw Recall |
| :-- | :-- | :-- | :-- |
| Random Forest | 77.6% | 50.42% | 38.08% |
| XGBoost | 73.7% | 53.18% | 8.82% |
| LightGBM | 73.0% | 53.57% | 4.58% |
| Neural Network | 72.9% | 53.89% | 6.28% |

### Specialist Model Candidates

Three candidates for H/A specialist role:


| Candidate | H/A Accuracy | Status |
| :-- | :-- | :-- |
| **Random Forest** (existing) | **77.6%** | Best performer, already trained |
| **LightGBM** (existing) | 73.0% | -4.6% vs RF |
| **XGBoost** (existing) | 73.7% | -3.9% vs RF |
| Neural Network | 72.9% | -4.7% vs RF |
| RF Specialist (H/A only training) | Unknown | Requires additional training and testing |

### Next Steps Required

**Phase 7**: Specialist Model Evaluation

1. Determine if portfolio approach provides meaningful improvement over single model
2. Test candidate models for specialist role:
    - Use existing models (RF/LGBM/XGB) with adaptive routing
    - Optionally train dedicated RF specialist on H/A-only dataset
3. Evaluate portfolio performance vs. single generalist model
4. Decision: single model vs. portfolio based on empirical results

> Check whether the observed H/A accuracy differences (4-5% advantage for RF) justify a portfolio approach, or if a single RF model is sufficient for production.

***

## Conclusion

### Phase 6 Summary: Model Selection Complete

**Draw Prediction Winner**: Random Forest (38.17% DR)

- All gradient boosting variants (sklearn GB, XGBoost, LightGBM) failed at picking draws (<12% DR)
- Neural networks showed same minority class problem (6.28% DR)

**H/A Specialist Candidates**: Under evaluation

- Random Forest: 77.6% H/A accuracy (current leader)
- LightGBM: 73.0% H/A accuracy
- XGBoost: 73.7% H/A accuracy
- Neural Network: 72.9% H/A accuracy
- RF Specialist (H/A-only training): Not yet tested

**Portfolio Strategy**: Requires Phase 7 testing

- Hypothesis: Route predictions based on draw probability
- High draw prob → Use generalist (RF) for H/D/A prediction
- Low draw prob → Use specialist for H/A prediction
- Performance gain vs. single model approach: **To be determined empirically**


### Production Recommendation (Current State)

**Confirmed Production Model** for draw prediction:

```python
Model: RandomForestClassifier(class_weight='balanced')
Features: odds_optimized (28 features)
Performance: 50.57% accuracy, 38.17% draw recall
```

**Next Phase**: Specialist model evaluation and portfolio testing before final production decision.

***

**Status**: Phase 6 (Model Selection) COMPLETE
**Next**: Phase 7 (Specialist Evaluation - Optional)
**Production Ready**: RF generalist model confirmed, portfolio approach under consideration
