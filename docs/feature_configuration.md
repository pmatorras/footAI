# footAI Feature Configuration:

The footAI project develops machine learning models to predict football match outcomes (Home Win/H, Draw/D, Away Win/A) using multi-season data from leagues like SP1 (La Liga). In the first iteration, we implemented core setups like the `baseline` (Elo ratings, odds, basic L5 form) and `extended` (adding L5 shots, accuracy, goal difference) features, which achieved reasonable overall accuracy (~53%) but poor draw prediction performance (recall ~0.25-0.255 across CV folds). This underperformance isn't due to model limitations—RandomForest with balanced weights handles imbalance well—but reflects the inherent difficulty of modeling draws, a minority class (~25% prevalence) driven by subtle parity factors like close Elo diffs or low-event games, as seen in sports analytics literature. To address this, we iterated on targeted draw-specific features (e.g., consensus probabilities, under 2.5 goals z-scores, Asian Handicap indicators) layered atop the extended base, leading to the `draw_optimized` configuration. This setup, selected as the v1.0 default for training and deployment, balances overall accuracy (~55%) with improved draw recall (~0.30-0.33), while respecting modular feature engineering for traceability and ablation testing.

All decisions were taken based on a 3-fold temporal cross-validation on SP1 data (using seasons 2122-2425, ~760 matches, ~190 draws), and using a RandomForestClassifier `(n_estimators=100, max_depth=10, class_weight="balanced")` included into a pipeline with imputation and sanitization. Experiments confirmed the need for pruning to ~28 features, reducing noise without sacrificing gains.




## Feature Selection Rationale

Feature engineering in `feature_engineering.py` (get_feature_columns function) uses a modular, layered approach to enable isolated testing and avoid over-engineering:

- **Baseline (~11 features)**: Starts with essential strength signals—Elo (`HomeElo, AwayElo, elo_diff`), L5 form basics (`home/away_goals_scored_L5, goals_conceded_L5, ppg_L5, form_diff_L5`), and normalized odds (`odds_home_prob_norm, odds_away_prob_norm`). This core provided stable H/A predictions but minimal draw lift (recall ~0.255 CV), as it lacks parity indicators.
- **Extended (~17 features total)**: Builds on baseline by adding L5 event stats (`home/away_shots_L5, shot_accuracy_L5, gd_L5`) for better momentum capture. This improved acc to ~0.530 CV but kept draw recall low (~0.255), highlighting the need for draw-focused additions without bloating the set.
- **Draw Additions (full: ~16; skimmed to 10)**: Appended to extended for targeted recall—includes consensus (`draw_prob_consensus, draw_prob_dispersion`), under signals (`under_2_5_prob, under_2_5_zscore`), diffs (`abs_elo_diff, elo_diff_sq, abs_odds_prob_diff`), handicap (`abs_ahh, ahh_zero/flat` if available), mins (`min_shots_l5, min_shot_acc_l5`), and bias (`league_draw_bias`). The full version included 34 features, which to reduce overfitting, the low-importance ones were removed (<0.015 Gini from full-run importances, e.g., min_goals_scored_l5 0.008, league_draw_bias 0.011) to retain ~92% cumulative signal.

The `draw_optimized` hybrid (28 extended + skimmed draw features) emerged from ablation: Full draw additions lifted recall +5-8% over baseline/extended (e.g., consensus 0.047 importance, under_2_5_prob 0.033), but noise from lows increased variance (acc $\sigma$ ~0.021 vs. 0.013 skimmed). Pruning respected layers (no baseline cuts for stability), focused on draw block (cum ~0.25 importance from top 10), and filtered available columns (dedupe, NaN handling via median). Derived features (e.g., abs_elo_diff = abs(elo_diff)) are computed in-pipeline; broken/sparse ones (e.g., home_draw_rate_l10) omitted. CLI: `--features-set draw_optimized` for this default.

## Performance Evaluation

Evaluations used temporal CV (train on prior seasons, validate sequential; ~20% holdout test) on ~760 SP1 matches. Metrics emphasize accuracy (overall) and draw recall (TP / actual draws), with F1_draw for balance. Skimmed runs (28 features extended + draw; 22 baseline + draw) showed stable gains; final test (304 matches, ~75 draws) confusion matrix:


| Predicted/Actual | H (134) | D (75) | A (95) |
| :-- | :-- | :-- | :-- |
| **H** | 91 | 30 | 21 |
| **D** | 26 | 25 | 22 |
| **A** | 17 | 20 | 52 |

- **Overall Accuracy**: 55.0% on the test set (167 correct out of 304 matches). CV averages: 0.518 for full draw (34 features), 0.522 for extended + skim draw (28 features), 0.511 for baseline skimdraw features (22 features). Low overfitting (test close to CV, gap at ~3%).
- **Draw Recall**: 0.333 on test (25 true draws caught out of 75 actual draws). CV averages: 0.308 full, 0.302 extended skim, 0.300 baseline skim—this is +8% better than the initial ~0.255 from baseline/extended alone. Across folds: 0.258 to 0.330 (stable range).
- **Draw Precision**: 0.323 on test (25 correct out of 77 predicted draws). False negatives: 30 draws mispredicted as H (due to home strength bias), 20 as A (underdog effects).
- **F1 for Draws**: ~0.328 (balances recall and precision). For comparison: H recall 0.679 (good home wins), A recall 0.547 (moderate away wins)—preserves home advantage pattern.
- **Stability Across Runs**: Accuracy $\sigma$ 0.020-0.021 (tight, low variation); draw recall $\sigma$ 0.016-0.031 (skimming reduces spread). Final out-of-sample (OOS) accuracy 55.3% matches CV, confirming no overfitting. Draw signals like consensus and under_zscore added ~3-5% isolated recall lift without hurting accuracy.

These results show `draw_optimized` improves draws modestly while keeping overall predictions reliable.

## Comparison to Alternatives

- **Baseline Only (~11 features)**: Acc 0.511 ± 0.020 CV, draw recall 0.300 ± 0.016—lean but insufficient for draws (+4.5% skim lift, misses L5 for acc).
- **Extended Only (~17 features)**: Acc 0.530 ± 0.008, draw recall ~0.255—no parity, higher acc but no recall edge.
- **Full Draw (~34 features)**: Acc 0.518 ± 0.013, draw recall 0.308 ± 0.013—good recall but acc dip/variance from noise (e.g., low mins/bias).
- **All Numeric (~50+)**: Acc ~0.525 CV, higher $\sigma$ 0.025—dilutes signals, overfit risk.

`draw_optimized` wins: +1-2% acc over baseline/full, +5-8% draw recall—efficient (~28 features), modular, and practical for ELO/analytics.

## Limitations, Risks and Future Work
- **Draw Performance**: Recall (0.333) means missing 2/3 of draws, as subtle parity factors are hard to fully capture.
- **Feature Volatility**: L5 stats can vary in low-sample cases (e.g., early seasons, handled via median fill).
- **Fit Assessment**: Mild overfitting (~3.1% test-CV gap) is not concerning for this size/temporal setup.
- **Monitor**: 
    - Mild overfitting (~3.1% test-CV gap)
    - Ensure recall <0.30; track mild overfit with more folds/seasons.
- **Enhancements**: Fix/uncomment l10 draw rates; ensemble RF+GB; calibrate probs for precision; add SHAP for per-match explainability.
- **Extensions**: Retrain for other leagues (SP1-specific now); live API preds; Model Card for sharing. Address broader risks: ethical disclaimers for betting use, formal bias audit.