"""
ML Model Training for Football Match Prediction
================================================

Simple baseline implementation for footAI v0.2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import joblib
from pathlib import Path
from footai.ml.models import get_models, select_features





def train_baseline_model(features_csv, feature_set="baseline", test_size=0.2, 
                         save_model=None, args=None):
    """
    Train baseline Random Forest model for match outcome prediction.

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
    # Load features
    if args.verbose:
        print(f"Loading features from: {features_csv}")

    df = pd.read_csv(features_csv)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if args.verbose:
        print(f"Loaded {len(df)} matches")

    # Select features
    feature_cols = select_features(df, feature_set)

    if args.verbose:
        print(f"\nUsing {len(feature_cols)} features ({feature_set} set)")


    # Prepare X, y
    X = df[feature_cols]
    y = df['FTR']  # Home/Draw/Away

    assert not X.empty, "Feature DataFrame is empty!"
    assert len(X) == len(y), "Mismatch in features and targets length"

    # Temporal split (train on earlier matches)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for temporal order
    )

    if args.verbose:
        print(f"\nTrain: {len(X_train)} matches")
        print(f"Test:  {len(X_test)} matches")
        print(f"\nTarget distribution (full dataset):")
        print(y.value_counts().to_string())
        print(f"\nTarget distribution (normalized):")
        print(y.value_counts(normalize=True).to_string())

    # Create and train model
    models = get_models(args)
    model = models['rf'] #baseline for first attemps

    if args.verbose:
        print("\nTraining Random Forest...")


    #check CV via timeseries
    tscv = TimeSeriesSplit(n_splits=3)
    cv_acc = []; cv_draw_recall = []
    label_map = {'H': 0, 'D': 1, 'A': 2}  # Standard for FTR

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        X_t, X_v = df.iloc[train_idx][feature_cols], df.iloc[test_idx][feature_cols]
        y_t, y_v_raw = df.iloc[train_idx]['FTR'], df.iloc[test_idx]['FTR']  # Use raw FTR (strings)
        
        model.fit(X_t, y_t)  # Fit on raw (model handles strings)
        y_p_raw = model.predict(X_v)  # Predictions as strings
        
        # Map to numeric for metrics
        if y_v_raw.dtype == 'object':
            y_v = y_v_raw.map(label_map)
            y_p_series = pd.Series(y_p_raw)  # Convert array to Series for map
            y_p = y_p_series.map(label_map)
        else:
            y_v = y_v_raw
            y_p = y_p_raw
        
        fold_acc = accuracy_score(y_v, y_p)
        fold_draw_recall = recall_score(y_v, y_p, labels=[1], average=None, zero_division=0)[0]
        
        cv_acc.append(fold_acc)
        cv_draw_recall.append(fold_draw_recall)
        
        print(f"Fold {fold+1} Acc: {fold_acc:.3f}, Draw Recall: {fold_draw_recall:.3f}")

    print(f"CV Acc ({feature_set}): {np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
    print(f"CV Draw Recall: {np.mean(cv_draw_recall):.3f} ± {np.std(cv_draw_recall):.3f}")



    #Full train/test
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    if args.verbose:
        print(f"\n{'='*70}")
        print("BASELINE MODEL RESULTS")
        print('='*70)
        print(f"\nTest Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print("\nBenchmarks:")
        print("  Random guess: 33.3%")
        print("  Always home:  ~45%")
        print("  Market odds:  ~50-53%")

        print("\n" + "-"*70)
        print("Classification Report:")
        print("-"*70)
        print(classification_report(y_test, y_pred))

        print("-"*70)
        print(f"Confusion Matrix ({feature_set}):")
        print("-"*70)
        cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])
        print("         Predicted")
        print("         H    D    A")
        print(f"Actual H {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
        print(f"       D {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
        print(f"       A {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

        print("\n" + "-"*70)
        print(f"Feature Importance ({feature_set}, Top 10):")
        print("-"*70)
        # Get feature importance from the classifier in pipeline
        clf = model.named_steps['clf']


        # Handle different classifiers (RF/GB use feature_importances_; LogReg uses coef_)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).mean(axis=0)  # Mean abs coef for multi-class
        else:
            print("Warning: Classifier has no importances/coef_; skipping.")
            return { ... }  # Or raise/continue without importance

        n_importances = len(importances)
        n_features_list = len(feature_cols)

        print(f"Debug: Importances length: {n_importances}, Feature list length: {n_features_list}")  # Remove after fix

        if n_importances != n_features_list:
            print(f"Warning: Length mismatch! Aligning feature_cols to importances length ({n_importances}).")
            # Slice to match (assumes pipeline uses first N features in order)
            aligned_features = feature_cols[:n_importances]
            # If you want exact names post-pipeline, add: aligned_features = clf.feature_names_in_ if hasattr(clf, 'feature_names_in_') else aligned_features
        else:
            aligned_features = feature_cols

        importance_df = pd.DataFrame({
            'feature': aligned_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(importance_df.head(30).to_string(index=False))

    # Save model if requested
    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        if args.verbose:
            print(f"\nModel saved to: {save_path}")

    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_cols,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])
    }


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