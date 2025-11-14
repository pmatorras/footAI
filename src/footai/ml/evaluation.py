import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


def print_results_summary(all_results,divisions):
    '''
        Print formatted summary of model results across seasons and divisions.
        all_results: dict of season -> {division: accuracy_float}
        divisions: list/tuple of two division labels; if None, auto-detect top two.
    '''
    if not all_results:
        print("No results to summarize.")
        return

    # Detect divisions if not provided
    if not divisions:
        # union of all division keys seen across seasons, then pick two deterministically
        keys = sorted({k for s in all_results.values() for k in s.keys()})
        if len(keys) < 2:
            print("Need two divisions to summarize.")
            return
        divisions = keys[:2]

    
    print("\n" + "="*70)
    print("MODEL ACCURACY BY SEASON AND DIVISION")
    print("="*70)
    print(f"{'Season':<10} {divisions[0]:<10} {divisions[1]:<10}")
    print("-"*70)
    season_rows = []
    print(all_results)
    for season in sorted(all_results.keys()):
        tier1 = all_results[season].get(divisions[0], 0)
        tier2 = all_results[season].get(divisions[1], 0)
        season_rows.append((tier1, tier2))
        print(f"{season:<10} {tier1*100:>6.1f}%    {tier2*100:>6.1f}%")

    print("-"*70)
    if season_rows:
        # Tier1 average
        tier1_values = [all_results[s][divisions[0]] for s in all_results if divisions[0] in all_results[s]]
        tier1_avg = sum(tier1_values) / len(tier1_values)

        # Tier2 average
        tier2_values = [all_results[s][divisions[1]] for s in all_results if divisions[1] in all_results[s]]
        tier2_avg = sum(tier2_values) / len(tier2_values)

        print(f"{'Average':<10} {tier1_avg*100:>6.1f}%    {tier2_avg*100:>6.1f}%")

        # Overall average
        all_values = tier1_values + tier2_values
        if all_values:
            overall_avg = sum(all_values) / len(all_values)
            var = sum((x - overall_avg)**2 for x in all_values) / len(all_values)
            std = var ** 0.5
            print("\n" + "="*70)
            print(f"Overall Average: {overall_avg*100:.1f}%")
            print(f"Total runs: {len(all_values)}")
            print(f"Range: {min(all_values)*100:.1f}% - {max(all_values)*100:.1f}%")
            print(f"Std Dev: {std*100:.1f}%")
            print("="*70)



def print_confusion(y_test, y_pred, feature_set, cm):
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(y_test, y_pred, labels=['H','D','A'], zero_division=0))

    print("-"*70)
    print(f"Confusion Matrix ({feature_set}):")
    print("-"*70)
    print("         Predicted")
    print("         H    D    A")
    print(f"Actual H {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
    print(f"       D {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
    print(f"       A {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

def print_per_division(df_test, pred_col='Prediction'):
    print("\nPer-division performance:")
    for division in sorted(df_test['Division'].unique()):
        div_data = df_test[df_test['Division'] == division]
        
        # Calculate accuracy
        div_acc = (div_data['Prediction'] == div_data['FTR']).mean()
        
        # Calculate draw recall
        div_draws = div_data['FTR'] == 'D'
        if div_draws.sum() > 0:
            div_draw_recall = (div_data.loc[div_draws, 'Prediction'] == 'D').sum() / div_draws.sum()
        else:
            div_draw_recall = 0.0
        
        n_matches = len(div_data)
        print(f"  {division}: {div_acc:.1%} acc, {div_draw_recall:.1%} draw recall ({n_matches} matches)")

        cm = confusion_matrix(div_data['FTR'], div_data[pred_col], labels=['H','D','A'])
        print(f"\nConfusion Matrix ({division}):")
        print("         Predicted")
        print("         H    D    A")
        print(f"Actual H {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
        print(f"       D {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
        print(f"       A {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

def print_feature_importance(model, feature_cols, feature_set):
    clf = model.named_steps['clf']
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        print("Warning: Classifier has no importances/coef_; skipping.")
        return
    n_importances = len(importances); n_features_list = len(feature_cols)
    if n_importances != n_features_list:
        print(f"Warning: Length mismatch! Aligning feature_cols to importances length ({n_importances}).")
        aligned = feature_cols[:n_importances]
    else:
        aligned = feature_cols
    importance_df = pd.DataFrame({'feature': aligned, 'importance': importances}).sort_values('importance', ascending=False)
    print("\n" + "-"*70)
    print(f"Feature Importance ({feature_set}, Top 10):")
    print("-"*70)
    print(importance_df.head(30).to_string(index=False))
