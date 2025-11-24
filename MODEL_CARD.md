# MODEL CARD: footAI Match Outcome Predictor

## Model Details
**Model Type**: RandomForestClassifier (scikit-learn)  
**Version**: v1.0  
**Training Date**: 2025-11-21  
**Author**: Pablo Matorras  
**License**: MIT

## Intended Use
**Primary**: Predict football match outcomes (Home/Draw/Away) for top European leagues  
**Use Cases**: 
- Sports analytics research
- Educational demonstration of temporal feature engineering
- Proof-of-concept for multi-league ML modeling

**Out-of-Scope**:
- Real-money betting (model not calibrated for betting markets)
- Cup matches or international games (only trained on league data)
- Leagues outside top 5 European (SP/IT/EN/DE/FR)

## Two Production Models

### Model 1: Tier1 (Top Divisions Only)
**Use When**: Predicting matches in SP1, I1, E0, D1, FR1

**Training Data**:
- Seasons: 2015-16 to 2025-26 (11 seasons)
- Matches: ~4,600 training samples
- Leagues: La Liga, Serie A, Premier League, Bundesliga, Ligue 1

**Performance** (test set):
- Overall Accuracy: **50.53%**
- Draw Recall: **38.34%** (vs 34.35% baseline = +3.99% improvement)
- Home Win Recall: 55.13%
- Away Win Recall: 54.05%

**Per-League Performance**:
| League | Accuracy | Draw Recall | Test Matches |
|--------|----------|-------------|--------------|
| Bundesliga (D1) | 49.81% | 41.09% | 795 |
| Premier League (E0) | 50.25% | 29.20% | 993 |
| Ligue 1 (FR1) | 48.81% | 33.68% | 838 |
| Serie A (I1) | 51.41% | 40.61% | 996 |
| La Liga (SP1) | 51.92% | **44.91%** | 1015 |

### Model 2: Multi-Country (Tier1 + Tier2)
**Use When**: Predicting matches in second divisions (SP2, I2, E1, D2, FR2)

**Training Data**:
- Same seasons (2015-26)
- Matches: ~10,000 training samples  
- Leagues: All tier1 + tier2 divisions

**Performance** (test set):
- Overall Accuracy: **46.48%**
- Draw Recall: **36.23%**
- **Trade-off**: Lower accuracy but covers more divisions

**Per-League Performance**:
| League | Accuracy | Draw Recall | Test Matches |
| :-- | :-- | :-- | :-- |
| Serie B (I2) | 44.08% | **59.27%** | 980 |
| Segunda (SP2) | 42.70% | **55.00%** | 1199 |
| Ligue 2 (FR2) | 41.03% | 45.34% | 931 |
| Championship (E1) | 40.76% | 31.38% | 1418 |
| 2. Bundesliga (D2) | 43.10% | 24.38% | 819 |


**Why Use This Model**:
- Tier2 leagues have less predictable outcomes (more variance in team quality)
- Training on tier1+tier2 together provides more data but dilutes signal
- Use when no tier1-only model exists for target division

## Hyperparameters Tuning Results

**Grid Search Configuration**:
- `max_depth`: [5, 8, 10, 15, 20, `None`]
- `min_samples_split`: [0.01, 0.02, 0.05, 5, 10, 20]
- `min_samples_leaf`: [1, 5, 8, 15]
- `n_estimators`: [100, 200, 300]
- `max_features`: ['sqrt', 'log2', 0.3, 0.5]

**Optimal Parameters (Both Models)**:
| Hyperparameter | Tier1 Model | Multicountry Model | Notes |
|----------------|-------------|---------------------|-------|
| `max_depth` | 5 | 5 | Shallow trees prevent overfitting |
| `min_samples_leaf` | 8 | 8 | Critical for draw class stability |
| `n_estimators` | 200 | 200 | Diminishing returns beyond 200 |
| `max_features` | log2 (~5) | sqrt (~5) | Comparable feature sampling |
| `min_samples_split` | 5 | 10 | Tier1 allows more aggressive splits |

**Key Finding**: Hyperparameter convergence across models validates feature engineering quality and demonstrates robust tuning methodology. The primary performance difference (50.5% vs 46.5% accuracy) stems from **data characteristics** (tier2 noise), not model configuration.

## Features (28 Total)
**Feature Set**: `odds_optimized` (selected via 5-phase systematic testing)

**Categories**:
1. **Baseline (12)**: Elo ratings, rolling form (goals, PPG, shots), match context
2. **Draw-optimized (8)**: Parity indicators, market consensus, low-scoring signals
3. **League context (3)**: League draw bias, team draw rates
4. **Odds movement (5)**: Opening→closing drift, sharp money indicators

**Top 5 Features by Importance**:
1. `odds_away_prob_norm` (22.23%) - Market assessment of away win
2. `odds_home_prob_norm` (20.30%) - Market assessment of home win  
3. `elo_diff` (12.65%) - Team strength gap
4. `abs_odds_prob_diff` (7.26%) - Match parity (closeness)
5. `AwayElo` (6.41%) - Away team strength

## Training Methodology
**Temporal Cross-Validation** (3-fold):
- Prevents data leakage (no future info used)
- Fold 1: Train 2015-19 → Test 2020-21
- Fold 2: Train 2015-21 → Test 2022-23  
- Fold 3: Train 2015-23 → Test 2024-25

**CV Results** (Tier1 model):
- Accuracy: 50.44% ± 0.06% (very stable)
- Draw Recall: 36.23% ± 2.76% (higher variance expected for minority class)

## Known Limitations

### Data Limitations
1. **No injury/suspension data** - Model can't account for missing key players
2. **No head-to-head history** - Team-specific matchups not captured
3. **No rest days** - Fixture congestion effects ignored
4. **Odds-dependent** - Requires opening/closing odds (not available for all matches)

### Performance Limitations  
1. **Draw prediction is hard** - 38% recall means 62% of draws are missed (but this is 11% better than baseline)
2. **Tier2 degradation** - Multi-country model accuracy drops 4% on second divisions
3. **Premier League weakness** - E0 draw recall only 29.20% (vs 44.91% for La Liga)
- Hypothesis: PL's high tempo → fewer draws → less predictable draw patterns
4. **Newly promoted teams** - First 5 matches have incomplete rolling features (degraded predictions)

### Use Case Limitations
1. **Not calibrated for betting** - Predicted probabilities need isotonic calibration for Kelly criterion
2. **No confidence intervals** - Point predictions only (no uncertainty quantification)
3. **League-specific bias** - Model learns league styles but may not generalize to new leagues

## Bias Analysis
**Home Advantage Bias** (Tier1 model):
- Home Win Precision: 64.94% (model overconfident on home wins)
- Draw Precision: 30.58% (underconfident on draws)
- Away Win Precision: 53.79%

**Interpretation**: Model inherits historical home advantage (~45% of matches are home wins) but betting markets already price this in. Not a flaw, but users should be aware.

**League Bias**:
- La Liga: Best draw recall (44.91%) - technical, low-tempo
- Premier League: Worst draw recall (29.20%) - high-tempo, more decisive

## Ethical Considerations
**Responsible Gambling**:
- This model is for **educational/research purposes only**
- **No guarantee of profitability** - 50.5% accuracy ≠ profitable betting (requires 52-55% after vig)
- **Variance**: Even with 55% accuracy, 1000-bet samples can show losses due to randomness
- **Problem gambling resources**: [BeGambleAware.org](https://www.begambleaware.org/)

**Fairness**:
- Model trained on public bookmaker odds (no proprietary insider data)
- Equal treatment across leagues (no league-specific hyperparameters)
- Open source - anyone can audit/improve methodology

## Model Performance Visualization
[Include confusion matrix heatmap here - can generate from your JSON]

## Reproducibility
**Code**: [github.com/pmatorras/footAI](https://github.com/pmatorras/footAI)  
**Data**: football-data.co.uk (public, free)  
**Trained Models**: Available in `models/` directory  
**Results JSON**: Complete metrics in `results/tier1_1516_to_2526_odds_optimized_rf__20251121.json`

**Reproduce Training**:
```

footai train --country SP IT EN DE FR --div tier1 \
--season-start 15-25 --features-set odds_optimized --elo-transfer

```
## Future work
- Experiment with a mixture‑of‑experts routing scheme conditioned on predicted draw probability (e.g. choose between tier1 vs multicountry or tier2‑only vs multicountry per match).
- Calibrate predicted probabilities (e.g. for draws) and evaluate whether conditional routing improves expected betting profitability on a held‑out period.
- See [docs/model_architecture_decicions.md](docs/model_architecture_decisions.md#alternative-considered-probabilistic-routing-mixtureofexperts) for a detailed proposal.

## Contact
**Author**: Pablo Matorras ([@pmatorras](https://github.com/pmatorras))  
**Issues**: [GitHub Issues](https://github.com/pmatorras/footAI/issues)  
**Last Updated**: 2025-11-24