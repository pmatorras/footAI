import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix

def print_cv_strategy(df, n_splits=3):
    """Print expanding-window CV strategy for documentation."""
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print("\n" + "="*70)
    print("EXPANDING-WINDOW TIME SERIES CROSS-VALIDATION")
    print("="*70)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_dates = df.iloc[train_idx]['Date'].dropna()  # NEW: Drop NaT
        test_dates = df.iloc[test_idx]['Date'].dropna()    # NEW: Drop NaT
        
        # Check if we have valid dates
        if len(train_dates) == 0 or len(test_dates) == 0:
            print(f"Fold {fold+1}: Skipping (missing dates)")
            continue
        
        train_start = train_dates.min().strftime('%Y-%m')
        train_end = train_dates.max().strftime('%Y-%m')
        test_start = test_dates.min().strftime('%Y-%m')
        test_end = test_dates.max().strftime('%Y-%m')
        
        print(f"Fold {fold+1}: Train {train_start} to {train_end} ({len(train_idx):>4} matches) "
              f"→ Test {test_start} to {test_end} ({len(test_idx):>3} matches)")
    
    print("="*70 + "\n")



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


def print_classification(class_report):
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(class_report)
    print("-"*70)

def print_confusion(feature_set, cm):
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

def print_feature_importance(model, feature_cols, feature_set, stats=False):
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
    if stats:
        print("\n" + "-"*70)
        print(f"Feature Importance ({feature_set}):")
        print("-"*70)
        print(importance_df.head(30).to_string(index=False))
    return importance_df

def write_metrics_json(json_path, country, divisions, feature_set, results, seasons, model='rf', cv_folds=None):
    """Write structured training metrics to JSON."""
    metrics = {
        'metadata': {
            'country': country,
            'divisions': divisions,
            'seasons': seasons,
            'feature_set': feature_set,
            'model': model,
            'n_splits' : len(cv_folds) if cv_folds else 3,
            'timestamp': datetime.now().isoformat()
        },
        'overall_test':{
            'accuracy': float(results['accuracy']),
            'n_samples': int(results.get('n_samples', 0)),
            # Per-class metrics
            'home': {
                'precision': float(results.get('home_precision', 0)),
                'recall': float(results.get('home_recall', 0)),
                'f1': float(results.get('home_f1', 0)),
            },
            'draw': {
                'precision': float(results.get('draw_precision', 0)),
                'recall': float(results.get('draw_recall', 0)),
                'f1': float(results.get('draw_f1', 0)),
            },
            'away':{
                'precision': float(results.get('away_precision', 0)),
                'recall': float(results.get('away_recall', 0)),
                'f1': float(results.get('away_f1', 0)),
            }
        },
        'confusion_matrix' : results.get('confusion_matrix', None),
        'cv_results' : {
            'summary' : {
                'cv_accuracy_mean' : float(results['cv_accuracy_mean']),
                'cv_accuracy_std' : float(results['cv_accuracy_std']),
                'cv_draw_recall_mean' : float(results.get('cv_draw_recall_mean', 0)),
                'cv_draw_recall_std' : float(results.get('cv_draw_recall_std', 0)),
            }
        },
        'per_division' :  results.get('per_division', None),
        'feature_importance': results.get('feature_importance', None)
    }

    

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)