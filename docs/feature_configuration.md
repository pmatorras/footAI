# footAI Feature Configuration:

The footAI project develops machine learning models to predict football match outcomes (Home Win/H, Draw/D, Away Win/A) using multi-season data from leagues like SP1 (La Liga). In the first iteration, we implemented core setups like the `baseline` (Elo ratings, odds, basic L5 form) and `extended` (adding L5 shots, accuracy, goal difference) features, which achieved reasonable overall accuracy (~53%) but poor draw prediction performance (recall ~0.25-0.255 across CV folds). This underperformance isn't due to model limitations—RandomForest with balanced weights handles imbalance well—but reflects the inherent difficulty of modeling draws, a minority class (~25% prevalence) driven by subtle parity factors like close Elo diffs or low-event games, as seen in sports analytics literature. To address this, we iterated on targeted draw-specific features (e.g., consensus probabilities, under 2.5 goals z-scores, Asian Handicap indicators) layered atop the extended base, leading to the `draw_optimized` configuration. This setup, selected as the v1.0 default for training and deployment, balances overall accuracy (~55%) with improved draw recall (~0.30-0.33), while respecting modular feature engineering for traceability and ablation testing.

All decisions were taken based on a 3-fold temporal cross-validation on SP1 data (using seasons 2122-2425, ~760 matches, ~190 draws), and using a RandomForestClassifier `(n_estimators=100, max_depth=10, class_weight="balanced")` included into a pipeline with imputation and sanitization. Experiments confirmed the need for pruning to ~28 features, reducing noise without sacrificing gains.




## Feature Selection Rationale

Feature engineering in `feature_engineering.py` (get_feature_columns function) uses a modular, layered approach to enable isolated testing and avoid over-engineering:

- **Baseline (12 features)**: Starts with essential strength signals—Elo (`HomeElo, AwayElo, elo_diff`), L5 form basics (`home/away_goals_scored_L5, goals_conceded_L5, ppg_L5, form_diff_L5`), and normalized odds (`odds_home_prob_norm, odds_away_prob_norm`). This core provided stable H/A predictions but minimal draw lift (recall ~0.255 CV), as it lacks parity indicators.
- **Extended (18 features total)**: Builds on baseline by adding L5 event stats (`home/away_shots_L5, shot_accuracy_L5, gd_L5`) for better momentum capture. This improved acc to ~0.530 CV but kept draw recall low (~0.255), highlighting the need for draw-focused additions without bloating the set.
- **Draw Additions (full: 16; skimmed to 10)**: Appended to extended for targeted recall—includes consensus (`draw_prob_consensus, draw_prob_dispersion`), under signals (`under_2_5_prob, under_2_5_zscore`), diffs (`abs_elo_diff, elo_diff_sq, abs_odds_prob_diff`), handicap (`abs_ahh, ahh_zero/flat` if available), mins (`min_shots_l5, min_shot_acc_l5`), and bias (`league_draw_bias`). The full version included 34 features, which to reduce overfitting, the low-importance ones were removed (<0.015 Gini from full-run importances, e.g., min_goals_scored_l5 0.008, league_draw_bias 0.011) to retain ~92% cumulative signal.

The `draw_optimized` hybrid (28 extended + skimmed draw features) emerged from ablation: Full draw additions lifted recall +5-8% over baseline/extended (e.g., consensus 0.047 importance, under_2_5_prob 0.033), but noise from lows increased variance (acc $\sigma$ ~0.021 vs. 0.013 skimmed). Pruning respected layers (no baseline cuts for stability), focused on draw block (cum ~0.25 importance from top 10), and filtered available columns (dedupe, NaN handling via median). Derived features (e.g., abs_elo_diff = abs(elo_diff)) are computed in-pipeline; broken/sparse ones (e.g., home_draw_rate_l10) omitted. CLI: `--features-set draw_optimized` for this default.

## Performance Evaluation
 WIP