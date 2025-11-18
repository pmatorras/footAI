"""
ML Model Training for Football Match Prediction
================================================

Simple baseline implementation for footAI v0.2
"""
import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

from footai.ml.models import get_models
from footai.utils.config import select_features
from footai.ml.evaluation import print_cv_strategy, print_classification, print_confusion, print_per_division, print_feature_importance




def train_baseline_model(features_csv, feature_set="baseline", test_size=0.2, 
                         save_model=None, args=None):
    """
    Train baseline Random Forest model for match outcome prediction.

    Uses expanding-window time series cross-validation (3 folds) to handle 
    tactical evolution and prevent lookahead bias:
    - Fold 1: Train on earliest ~33% → Test on next ~22%
    - Fold 2: Train on earliest ~56% → Test on next ~22%
    - Fold 3: Train on earliest ~78% → Test on final ~22%
    
    This mimics production deployment where the model learns from all 
    historical data and naturally downweights old tactical patterns.
    
    Example for 10 seasons (2015-2025):
        Fold 1: Train 2015-2018 → Test 2018-2020
        Fold 2: Train 2015-2020 → Test 2020-2022
        Fold 3: Train 2015-2022 → Test 2022-2025

    Args:
        features_csv: Path to features CSV (output from feature_engineering)
        feature_set: Which features to use ("baseline", "extended", "all")
        test_size: Fraction of data for testing (default: 0.2)
        save_model: Path to save trained model (optional)
        verbose: Print progress and results

    Returns:
        dict with:
            - model: Trained sklearn Pipeline
            - X_test: Test features
            - y_test: Test labels
            - y_pred: Test predictions
            - feature_names: List of features used
            - accuracy: Test accuracy
    """
    verbose = getattr(args, 'verbose', False)  # Default to False if args is None
    stats = getattr(args, 'stats', False)  # Default to False if args is None

    # Load features
    if verbose:
        print(f"Loading features from: {features_csv}")

    df = pd.read_csv(features_csv, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if verbose:
        print(f"Loaded {len(df)} matches")

    print_cv_strategy(df, n_splits=3)

    # Select features
    feature_cols = select_features(df, feature_set)
    print(f"\nFeature check:")
    for idx, col in enumerate(feature_cols):
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct> 30 or args.verbose: print(f"  [{idx}] {col}: {missing_pct:.1f}% missing")
        if missing_pct == 100:
            print(f"      ^^^^ COMPLETELY EMPTY!")
        # Suppress known sklearn imputation warning
        # under_2_5_prob has missing values in older seasons due to incomplete odds data
        # Random Forest handles NaN values natively, so this is safe
        warnings.filterwarnings('ignore', 
            message='Skipping features without any observed values.*',
            category=UserWarning,
            module='sklearn')
    if 'is_tier1' in df.columns:
        if verbose: print(f"  Adding division features: is_tier1, division_tier")
        feature_cols = feature_cols + ['is_tier1', 'division_tier']
    if verbose:
        print(f"\nUsing {len(feature_cols)} features ({feature_set} set)")

    feature_cols = [c for c in feature_cols if c != 'division_tier']
    feature_cols = [c for c in feature_cols if c != 'is_tier1']

    # Prepare X, y
    X = df[feature_cols]
    y = df['FTR']  # Home/Draw/Away

    assert not X.empty, "Feature DataFrame is empty!"
    assert len(X) == len(y), "Mismatch in features and targets length"


    if verbose:
        print(f"\nTarget distribution (full dataset):")
        print(y.value_counts().to_string())
        print(f"\nTarget distribution (normalized):")
        print(y.value_counts(normalize=True).to_string())

    # Create and train model
    models = get_models(args)
    model = models[args.model] #baseline for first attemps

    if verbose:
        print(f"\nTraining {model}...")


    #check CV via timeseries
    tscv = TimeSeriesSplit(n_splits=3)
    cv_acc = []; cv_draw_recall = []
    label_map = {'H': 0, 'D': 1, 'A': 2}  

    use_div_weights = ('Division' in df.columns) and (df['Division'].nunique() > 1)
    final_x_test = None
    final_y_test = None    
    final_y_pred = None
    print("-"*70)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_t, X_v = X.iloc[train_idx], X.iloc[test_idx]
        y_t, y_v_raw = y.iloc[train_idx], y.iloc[test_idx]
        
        if use_div_weights:
            # Build per-fold division weights
            div_t = df.iloc[train_idx]['Division']
            div_counts = div_t.value_counts()
            w_div_map = (div_counts.sum() / (len(div_counts) * div_counts)).to_dict()
            w_t = div_t.map(w_div_map).astype(float).values
            w_t = np.nan_to_num(w_t, nan=1.0, posinf=1.0, neginf=1.0)
            w_t = w_t / w_t.mean()
            model.fit(X_t, y_t, clf__sample_weight=w_t)
        else:
            model.fit(X_t, y_t)
        y_p_raw = model.predict(X_v)  # Predictions as strings
        
        # Map to numeric for metrics
        if y_v_raw.dtype == 'object':
            y_v = y_v_raw.map(label_map)
            y_p_series = pd.Series(y_p_raw)  # Convert array to Series for map
            y_p = y_p_series.map(label_map)
        else:
            y_v = y_v_raw
            y_p = y_p_raw
        if fold == 2:  # Final fold (0-indexed)
            final_x_test = X_v
            final_y_test = y_v_raw
            final_y_pred = y_p_raw
        
        fold_acc = accuracy_score(y_v, y_p)
        fold_draw_recall = recall_score(y_v, y_p, labels=[1], average=None, zero_division=0)[0]
        
        cv_acc.append(fold_acc)
        cv_draw_recall.append(fold_draw_recall)
        print(f"Fold {fold+1} Acc: {fold_acc:.3f}, Draw Recall: {fold_draw_recall:.3f}")

    print(f"CV Acc average({feature_set}): {np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
    print(f"CV Draw Recall: {np.mean(cv_draw_recall):.3f} ± {np.std(cv_draw_recall):.3f}")

    y_test = final_y_test
    y_pred = final_y_pred
    y_pred_proba = model.predict_proba(X.iloc[test_idx])  # From last fold's test_idx

    accuracy = cv_acc[-1]
    cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])
    # Calculate per-class metrics from confusion matrix
    # cm[i,j] = true label i, predicted label j
    home_precision = cm[0,0] / cm[:,0].sum() if cm[:,0].sum() > 0 else 0
    home_recall = cm[0,0] / cm[0,:].sum() if cm[0,:].sum() > 0 else 0
    draw_precision = cm[1,1] / cm[:,1].sum() if cm[:,1].sum() > 0 else 0
    draw_recall = cm[1,1] / cm[1,:].sum() if cm[1,:].sum() > 0 else 0
    away_precision = cm[2,2] / cm[:,2].sum() if cm[:,2].sum() > 0 else 0
    away_recall = cm[2,2] / cm[2,:].sum() if cm[2,:].sum() > 0 else 0

    report = classification_report(y_test, y_pred, labels=['H','D','A'], zero_division=0)

    print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

    if stats:
        print(f"Features used: {feature_cols if verbose else (len(feature_cols))}")
        print_classification(class_report=report)
        print_confusion(feature_set=feature_set, cm=cm)

    # Around line 186-190 in training.py, after print_per_division()
    if 'Division' in df.columns:
        df_test = df.iloc[-len(y_test):].copy()
        df_test['Prediction'] = y_pred
        if stats: print_per_division(df_test, pred_col='Prediction')
        
        # NEW: Capture per-division metrics for JSON
        per_division_results = {}
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
            
            # Get confusion matrix
            div_cm = confusion_matrix(div_data['FTR'], div_data['Prediction'], labels=['H','D','A'])
            
            per_division_results[division] = {
                'accuracy': float(div_acc),
                'draw_recall': float(div_draw_recall),
                'n_matches': int(n_matches),
                'confusion_matrix': {
                    'labels': ['H', 'D', 'A'],
                    'matrix': div_cm.tolist()
                }
            }
    else:
        per_division_results = None

    feature_importance = print_feature_importance(model, feature_cols, feature_set, stats)
    results = {
        'model': model,
        'X_test': final_x_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_cols,
        'accuracy': accuracy,
        'feature_importance': feature_importance.to_dict('records'),
        'confusion_matrix': {
            'labels': ['H', 'D', 'A'],
            'matrix': cm.tolist()
        },
        # CV stats
        'cv_accuracy_mean': np.mean(cv_acc),
        'cv_accuracy_std': np.std(cv_acc),
        'cv_draw_recall_mean': np.mean(cv_draw_recall),
        'cv_draw_recall_std': np.std(cv_draw_recall),

        #Per class metrics
        'home_precision': home_precision,
        'home_recall': home_recall,
        'draw_precision': draw_precision,
        'draw_recall': draw_recall,
        'away_precision': away_precision,
        'away_recall': away_recall,
        # NEW: Per-division breakdown
        'per_division': per_division_results  
    }
    # Save model if requested
    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {args.model} Model to:", save_path)
        joblib.dump(model, save_path)
        if verbose:
            print(f"\nModel saved to: {save_path}")

    return results


def predict_match(model, home_features, away_features, feature_names):
    """
    Predict outcome for a single match.

    Args:
        model: Trained model (from train_baseline_model)
        home_features: dict with home team features
        away_features: dict with away team features
        feature_names: List of features (from train_baseline_model)

    Returns:
        dict with prediction and probabilities
    """
    # Construct feature vector
    features = {}
    for feat in feature_names:
        if feat.startswith('home_'):
            features[feat] = home_features.get(feat, np.nan)
        elif feat.startswith('away_'):
            features[feat] = away_features.get(feat, np.nan)
        else:
            # Match-level features like elo_diff
            features[feat] = home_features.get(feat, away_features.get(feat, np.nan))

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Get class labels
    classes = model.named_steps['clf'].classes_

    return {
        'prediction': prediction,
        'probabilities': dict(zip(classes, probabilities)),
        'confidence': max(probabilities)
    }