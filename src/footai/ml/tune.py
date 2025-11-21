def tune_rf_hyperparameters(X, y, label_encoder, n_iter=30, verbose=True):
    """
    Quick hyperparameter tuning for Random Forest.
    
    Args:
        X: Feature matrix
        y: Target labels (encoded)
        n_iter: Number of parameter combinations to try
        verbose: Print progress
    
    Returns:
        dict: Best parameters found
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer,balanced_accuracy_score, recall_score

    draw_label = label_encoder.transform(['D'])[0]

    print(f"\n{'='*70}")
    print("HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    print(f"Label encoder classes: {label_encoder.classes_}")  # ← What does it know?
    print(f"Draw label encoded as: {draw_label}")
    print(f"Iterations: {n_iter}")
    print(f"CV folds: 3 (TimeSeriesSplit)")
    print(f"Scoring: 50% balanced_acc + 50% draw_recall")
    
    # Define scorer inside where it can access draw_label
    def draw_focused_scorer(y_true, y_pred):
        """Score that emphasizes draw recall."""
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        # DEBUG: See what's in y_true and y_pred
        unique_true = sorted(set(y_true))
        unique_pred = sorted(set(y_pred))
        # Draw recall for the specific draw label
        draw_recall = recall_score(y_true, y_pred, labels=[draw_label], average='macro', zero_division=0)
        
        score = 0.5 * bal_acc + 0.5 * draw_recall
        
        import random
        if random.random() < 0.05:
            print(f"  [Scorer] y_true classes: {unique_true}, y_pred classes: {unique_pred}")
            print(f"  [Scorer] looking for draw_label={draw_label}")
            print(f"  [Scorer] bal_acc={bal_acc:.3f}, draw_recall={draw_recall:.3f}, score={score:.3f}")
        
        return score
    
    # Parameter distributions
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 8, 10, 15, 20, None],
        'min_samples_split': [0.01, 0.02, 0.05, 2, 5, 10, 20],
        'min_samples_leaf': [0.005, 0.01, 0.02, 1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False],
    }


    # Base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Randomized search
    scorer = make_scorer(draw_focused_scorer)

    search = RandomizedSearchCV( rf, param_dist, n_iter=n_iter, cv=tscv, scoring=scorer, n_jobs=-1, verbose=1 if verbose else 0, random_state=42)
    print("Starting search...")
    
    search.fit(X, y)

    print(f"\nBest CV score: {search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    return search.best_params_
