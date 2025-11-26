# Model Selection Study
**footAI v0.4**

Date: 2025-11-26
Dataset: Top 5 European Leagues (2015-2025, ~8000 matches)  
Objective: Maximize draw recall while maintaining reasonable accuracy

---

## Executive Summary (v0.4)

This report details the selection of the production model for footAI v0.4. The focus of this iteration was stabilizing **Tier 2 predictions** through rigorous **Elo rating transfer** (incorporating promotion/relegation fluidly between tiers) and adopting a **Multi-Country training strategy**.

**Selected Architecture:**

* **Model**: Random Forest.
* **Features**: `odds_optimized` set (28 features) including updated Elo dynamics.
* **Strategy**: A single unified model predicts for all leagues, leveraging Tier 1 patterns to fix Tier 2 "blind spots."

**Why Random Forest?**
Random Forest was chosen over Gradient Boosting for its variance reduction. In noisy betting markets, boosting methods tended to overfit, chasing marginal accuracy gains at the cost of draw recall. Random Forest's bagging provided stable, generalized probabilities essential for long-term value.

**Key Results:**

1. **Tier 1**: While the unified model improved overall Tier 1 accuracy by ~1.0%, it reduced Draw Recall by ~10% compared to specialized Tier 1 models. This is an accepted trade-off to ensure global system stability and robust lower-division coverage.
2. **Tier 2**: The unified model improved average Tier 2 Draw Recall by +22.9% (from 22.7% baseline to 45.6%).


**Recommendation**: Deploy a **Hybrid Expert-Routing Strategy** as detailed in *Model Architecture Decisions*.

- **Tier 1**: Use the specialized **Tier 1 Model** (50.5% Acc, 37.4% DR) to maximize draw detection in elite leagues.
- **Tier 2**: Use the **Multi-Country Model** (42.4% Acc, 45.6% DR) to leverage cross-tier signal and ensure robust draw coverage in lower divisions.

> Alternatively, implement a **Gating Architecture** with tier-specific experts (as discussed in [Model Architecture Decisions](model_architecture_decisions.md#probabilistic-routing-mixture-of-experts)):
> - **Tier 1 Experts**: Tier 1 Specialist (stronger draws) and Multi-Country Model (stronger H/A).
> - **Tier 2 Experts**: Tier 2 Specialist (stronger H/A) and Multi-Country Model (stronger draws).\


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

> **Disclaimer**: v0.4 introduces rigorous Elo transfer logic (T1-T2 and T2-T3) with decay. This makes the Elo feature distribution smoother and more realistic compared to v0.3. While Tier 1 metrics dropped slightly (likely due to reduced overfitting on "lucky" static ratings), the Multi-Country model saw a significant boost in Draw Recall (+1.6%), validating the new logic's robustness for lower tiers.

#### 1. Tier 1 multi-country (SP, IT, EN, DE, FR) 

Best configuration (odds_optimized features):
- `n_estimators`: 200
- `max_depth`: 5
- `min_samples_split`: 5
- `min_samples_leaf`: 8
- `max_features`: 'log2'
- `class_weight`: 'balanced'
- `bootstrap`: True

Performance:
- Average Accuracy: 50.5%
- Average CV Draw Recall: 36.0%
- Fold 3 (most recent): 50.5% acc, 37.4% draw recall

Key Findings:

- Max_depth=5 outperformed deeper trees (15-30), suggesting draws require simple decision boundaries to avoid overfitting the minority class.
- Tighter constraints (`min_samples_leaf=8`) were preferred over v0.3, suggesting the new Elo features are cleaner but require more evidence per leaf.
- Limiting feature subsets to sqrt(n_features) prevents overfitting.
- Standard balanced_accuracy optimized to 51.3% acc but only 22.5% draw recall. Custom scorer achieved 50.5% acc with 38.3% draw recall.

#### Tier 2 multi-country (SP, IT, EN, DE, FR) 


We attempted to train a specialized model exclusively on Tier 2 data to see if it could capture lower-division nuances better than a general model.

**Best Configuration (odds_optimized features):**

- `n_estimators`: 50 (Lower due to smaller dataset)
- `max_depth`: 5
- `min_samples_split`: 0.05 (Very high regularization to prevent overfitting noise)
- `min_samples_leaf`: 1
- `class_weight`: 'balanced'

**Performance:**

- **Accuracy**: 42.9%
- **Draw Recall**: 24.3%
- **Precision (Draw)**: 32.3%

#### Tier1+Tier2 Multi-Country Model

To address poor tier2 draw recall found by training tier2 only models, tier2 predictions are taken from a multi-country model trained on combined tier1+tier2. This model was also optimized:


Best configuration (odds_optimized features):

- `n_estimators`: 200
- `max_depth`: 5
- `min_samples_split`: 10
- `min_samples_leaf`: 8
- `max_features`: 'log2'
- `class_weight`: 'balanced'
- `bootstrap`: True

Overall Performance:

- Test Accuracy: 46.6%
- CV Draw Recall: 37.9% ± 1.4%

#### Tier 1 Performance Comparison (Specialized vs Unified)

| Division | Tier 1 Model Acc | Multi-Tier Model Acc | Tier 1 Model DR | Multi-Tier Model DR | DR Change |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **D1** | 49.7% | 51.1% | 39.6% | 27.3% | **-12.3%** |
| **E0** | 50.4% | 53.6% | 29.2% | 20.3% | **-8.9%** |
| **F1** | 48.7% | 49.1% | 33.2% | 20.9% | **-12.3%** |
| **I1** | 51.7% | 51.1% | 39.6% | 30.6% | **-9.0%** |
| **SP1** | 51.6% | 52.7% | 43.2% | 33.7% | **-9.5%** |
| **Average** | **50.5%** | **51.5%** | **37.4%** | **26.6%** | **-10.8%** |

**Key Findings:**

- Accuracy Boost: The unified model actually improves accuracy by +1.0% on average across elite leagues, likely by reducing false-positive draw predictions.
- Draw Recall Trade-off: There is a significant drop in Draw Recall (-10.8%) for Tier 1. The unified model is more conservative, predicting fewer draws in elite leagues compared to the specialized model.

#### Tier2 Performance Comparison (Tier 2 vs Multi-tier):


| Division | Tier 2 Acc | Multi-tier Acc | Tier 2 DR | Multi-tier DR | DR Improvement |
| :-- | :-- | :-- | :-- | :-- | :-- |
| D2 | 45.3% | 43.7% | 4.9% | 33.3% | **+28.4%** |
| E1 | 42.9% | 40.5% | 15.4% | 36.0% | **+20.6%** |
| F2 | 41.7% | 40.3% | 22.0% | 48.4% | **+26.4%** |
| I2 | 41.7% | 44.6% | 32.6% | 55.9% | **+23.3%** |
| SP2 | 43.0% | 42.7% | 38.5% | 54.2% | **+15.7%** |
| **Average** | **42.9%** | **42.4%** | **22.7%** | **45.6%** | **+22.9%** |

**Key Findings:**

- Tuning (and using the multi-country model) improved tier2 draw recall by a massive **+22.9%** on average.
- **D2 (Bundesliga 2)** saw the most critical fix, jumping from a broken 4.9% recall to a usable 33.3%.
- The "Tier 2 Only" model (Baseline) collapsed in predicting draw rates, making the Multi-Country approach **mandatory** for viable Tier 2 predictions.
- Accuracy trade-off is minimal (~0.5% drop) for gaining the ability to actually predict draws.

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
- Future phases may investigate conditional routing (mixture‑of‑experts) based on draw probability, building on the two‑model tier1 / tier2 architecture described here.
- See [model_architecture_decisions.md](model_architecture_decisions.md) for a detaile drationale.


***

**Production Ready**: RF generalist model confirmed, portfolio approach to be considered
