"""
ML Model Training for Football Match Prediction
================================================

Simple baseline implementation for footAI v0.2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

    # Drop rows with NaN features (first matches)
    df_clean = df.dropna(subset=feature_cols + ['FTR'])

    if args.verbose:
        print(f"After dropping NaN: {len(df_clean)} matches ({len(df_clean)/len(df)*100:.1f}%)")

    # Prepare X, y
    X = df_clean[feature_cols]
    y = df_clean['FTR']  # Home/Draw/Away

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
        print("Confusion Matrix:")
        print("-"*70)
        cm = confusion_matrix(y_test, y_pred, labels=['H', 'D', 'A'])
        print("         Predicted")
        print("         H    D    A")
        print(f"Actual H {cm[0][0]:4d} {cm[0][1]:4d} {cm[0][2]:4d}")
        print(f"       D {cm[1][0]:4d} {cm[1][1]:4d} {cm[1][2]:4d}")
        print(f"       A {cm[2][0]:4d} {cm[2][1]:4d} {cm[2][2]:4d}")

        print("\n" + "-"*70)
        print("Feature Importance (Top 10):")
        print("-"*70)
        # Get feature importance from the classifier in pipeline
        clf = model.named_steps['clf']
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python models.py <features_csv>")
        print("Example: python models.py data/processed/SP_2425_SP1_features.csv")
        sys.exit(1)

    features_csv = sys.argv[1]

    # Train baseline model
    results = train_baseline_model(
        features_csv,
        feature_set="baseline",
        test_size=0.2,
        save_model="models/baseline_rf.pkl",
        verbose=True
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nFinal accuracy: {results['accuracy']*100:.1f}%")
    print(f"Features used: {len(results['feature_names'])}")
    print(f"\nNext steps:")
    print("  1. Analyze feature importance above")
    print("  2. Check confusion matrix for prediction patterns")
    print("  3. Try 'extended' or 'all' feature sets")
    print("  4. Compare with betting odds accuracy")