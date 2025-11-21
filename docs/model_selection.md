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
```python
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

```python
GradientBoostingClassifier(
n_estimators=50,
max_depth=3,
learning_rate=0.01,
subsample=0.5
)
```

**Deep** (aggressive):
```python
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
```python
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
```python
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
## Hyperparameter Tuning

### Methodology

Hyperparameter tuning was performed using `RandomizedSearchCV` with:
- **Cross-validation**: TimeSeriesSplit (3 folds) to respect temporal ordering
- **Scoring metric**: Custom draw-weighted scorer combining balanced accuracy and draw recall
- **Iterations**: 100 parameter combinations tested
- **Dataset**: Tier1 multi-country (SP, IT, EN, DE, FR), 2015-16 to 2025-26

### Scoring Function

To address poor draw prediction (minority class ~25%), we used a custom scorer:

```python
score = 0.5 * balanced_accuracy + 0.5 * draw_recall
```


This balances overall performance with explicit draw class optimization.

### Search Space
```python
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 8, 10, 15, 20, None],
        'min_samples_split': [0.01, 0.02, 0.05, 2, 5, 10, 20],
        'min_samples_leaf': [0.005, 0.01, 0.02, 1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False],
    }
```


### Results
#### 1. Tier1 multi-country (SP, IT, EN, DE, FR) 

Best configuration (odds_optimized features):
- `n_estimators`: 100
- `max_depth`: 5
- `min_samples_split`: 20
- `min_samples_leaf`: 1
- `max_features`: 'sqrt'
- `class_weight`: 'balanced_subsample'
- `bootstrap`: True

Performance:
- Average Accuracy: 50.5%
- Average CV Draw Recall: 36.2%
- Fold 3 (most recent): 50.8% acc, 38.3% draw recall

Key Findings:

- Max_depth=5 outperformed deeper trees (15-30), suggesting draws require simple decision boundaries to avoid overfitting the minority class.
- Split=20 (vs 2) provides better regularization for draw prediction.
- Limiting feature subsets to sqrt(n_features) prevents overfitting.
- Standard balanced_accuracy optimized to 51.3% acc but only 22.5% draw recall. Custom scorer achieved 50.5% acc with 38.3% draw recall.


#### Tier1+Tier2 Multi-Country Model

To address poor tier2 draw recall found by training tier2 only models, tier2 predictions are taken from a multi-country model trained on combined tier1+tier2. This model was also optimized:


Best configuration (odds_optimized features):

- `n_estimators`: 200
- `max_depth`: 5
- `min_samples_split`: 10
- `min_samples_leaf`: 8
- `max_features`: 'sqrt'
- `class_weight`: 'balanced'
- `bootstrap`: True

Overall Performance:

- Test Accuracy: 46.5%
- CV Draw Recall: 36.3% ± 1.4%


Tier2 Performance Comparison (Baseline vs Tuned):


| Division | Baseline Acc | Tuned Acc | Baseline DR | Tuned DR | DR Improvement |
| :-- | :-- | :-- | :-- | :-- | :-- |
| D2 | 42.0% | 43.1% | 18.9% | 24.4% | **+5.5%** |
| E1 | 41.9% | 40.8% | 30.1% | 31.4% | +1.3% |
| FR2 | 41.8% | 41.0% | 43.3% | 45.3% | +2.0% |
| I2 | 44.1% | 44.1% | 59.0% | 59.3% | +0.3% |
| SP2 | 42.4% | 42.7% | 53.1% | 55.0% | +1.9% |
| **Average** | **42.4%** | **42.3%** | **40.9%** | **43.1%** | **+2.2%** |

**Key Findings:**

- Tuning improved tier2 draw recall by **+2.2%** on average while maintaining accuracy
- D2 (Bundesliga 2) saw the largest improvement (+5.5% DR), fixing its worst-case baseline of 18.9%
- More trees (200 vs 100) and higher regularization (split=10, leaf=8) helped tier2 generalization
- Model successfully balances tier1 and tier2 performance for multi-division deployment

## Production Model Architecture

### Tier-Based Model Strategy

After evaluating various approaches, the production system uses **two separate models** optimized for different league tiers:

**1. Tier1 Model** (for top-division predictions)

- Trained on: Tier1 data only (SP1, EN, IT1, DE1, FR1)
- Features: odds_optimized (28 features)
- Performance: 50.8% accuracy, 36.1% draw recall
- Hyperparameters: n=100, depth=5, split=20, leaf=1

**2. Multi-Country Model** (for tier2 predictions)

- Trained on: Tier1 + Tier2 data (all 10 divisions)
- Features: odds_optimized (28 features)
- Performance: 42.3% accuracy, 43.1% avg tier2 draw recall
- Hyperparameters: n=200, depth=5, split=10, leaf=8


### Rationale

**Why not a single model for all tiers?**

- Tier1-only data provides optimal performance for tier1 leagues (most predictable, highest quality data)
- Tier2 suffered from poor draw recall (19.5% baseline) when trained on tier2-only data
- Adding tier1 data to tier2 training improved tier2 draw recall by +22% (19.5% → 43.1%)

**Why not specialist H/A models?**

- Initial hypothesis: route predictions based on draw probability (high prob → generalist, low prob → specialist)
- Reality: Random Forest already achieves 77.6% H/A accuracy as a generalist, with no viable specialist candidates showing meaningful improvement
- Added complexity not justified by empirical results

For detailed discussion of architecture decisions, see `model_architecture_decisions.md`.

### Deployment Strategy

**Production routing logic:**

```python
if division in ['SP1', 'EN', 'IT1', 'DE1', 'FR1']:
    model = tier1_model  # Tier1-only trained
else:
    model = multicountry_model  # Tier1+Tier2 trained
```

This maximizes performance for both tier groups while avoiding overfitting and data scarcity issues.


## Conclusion

### Phase 6 Summary: Model Selection Complete

From the models tested, the best performant was the random Forest:

- All gradient boosting variants (sklearn GB, XGBoost, LightGBM) failed at picking draws (<12% DR)
- Neural networks showed the same minority class problem (6.28% DR)
- Optimization lead to shallow depth (`depth=5`)

**Production Deployment:**

The production system uses a **tier-based architecture** with two optimized Random Forest models:


| Model | Training Data | Hyperparameters | Performance |
| :-- | :-- | :-- | :-- |
| **Tier1** | SP1, EN, IT1, DE1, FR1 only | n=100, depth=5, split=20, leaf=1 | 50.5% acc, 38.3% DR |
| **Multi-Country (Tier2)** | All tier1+tier2 divisions | n=200, depth=5, split=10, leaf=8 | 42.3% acc, 43.1% DR |

Both models use:

- **Features**: odds_optimized (28 features)
- **Tuning**: RandomizedSearchCV with 100 iterations, custom draw-weighted scorer (50/50 balanced_accuracy + draw_recall)
- **Key insight**: Shallow trees (depth=5) generalize better for minority class prediction

**Why tier-based?**

- Tier1-only training maximizes top-division performance
- Tier2 benefits from tier1 data (+22% draw recall: 19.5% → 43.1%)
- Specialist H/A models rejected (no empirical improvement over RF's 77.6% H/A accuracy)


- See [model_architecture_decisions.md](model_architecture_decisions.md) for a detaile rationale.


***

**Status**: Phase 6 (Model Selection) COMPLETE
**Next**: Phase 7 (Specialist Evaluation - Optional)
**Production Ready**: RF generalist model confirmed, portfolio approach under consideration
