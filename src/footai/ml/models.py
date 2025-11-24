#From financial-ml
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
# Suppress LightGBM feature name warnings (sklearn 1.0+ issue)
warnings.filterwarnings(
    'ignore', 
    message='X does not have valid feature names',
    category=UserWarning,
    module='sklearn'
)
MODEL_METADATA = {
    "logreg_l1": {
        "name": "Logistic Regression (L1)",
        "short_name": "L1 Logreg",
        "description": "L1-regularized logistic regression with feature selection",
        "color": "#1f77b4",  # For future implemmentation
        "marker": "o"
    },
    "logreg_l2": {
        "name": "Logistic Regression (L2)",
        "short_name": "L2 Logreg",
        "description": "L2-regularized logistic regression (ridge)",
        "color": "#ff7f0e",
        "marker": "s"
    },
    "rf": {
        "name": "Random Forest",
        "short_name": "RF",
        "description": "Random forest classifier with balanced class weights",
        "color": "#2ca02c",
        "marker": "^"
    },
    "xgb": {
        "name": "XGBoost",
        "short_name": "XGB",
        "description": "Gradient boosting with histogram-based learning",
        "color": "#9467bd",
        "marker": "D"
    },
    "lgbm": {
        "name": "LightGBM",
        "short_name": "LGBM",
        "description": "Fast gradient boosting with leaf-wise growth",
        "color": "#8c564b",
        "marker": "v"
    },
    "gb": {
        "name": "Gradient Boosting",
        "short_name": "GB",
        "description": "Sklearn gradient boosting classifier",
        "color": "#e377c2",
        "marker": "p"
    },
    "gb_deep": {
        "name": "Gradient Boosting (extra layers)",
        "short_name": "GB_deep",
        "description": "Sklearn gradient boosting classifier",
        "color": "#e377c2",
        "marker": "p"
    },
    "nn": {
        "name": "Neural network sklearn test",
        "short_name": "GB_deep",
        "description": "Sklearn neural network",
        "color": "#3b3439",
        "marker": "p"
    }

}
TUNED_RF_PARAMS = {
    'tier1': {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 8,
        'max_features': 'log2',
        'class_weight': 'balanced'
    },
    'multicountry': {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 8,
        'max_features': 'sqrt',
        'class_weight': 'balanced'
    }
}

def get_model_name (model_key, short_name=False):
    """
    Get the display name for a model
    Args:
        model_key: Model identifier (has to be one of MODEL_METADATA keys)
        short_name: Display the short name or not 
    Returns:
        Display name/short name for the model
    """
    key = "short_name" if short_name else "name"
    return MODEL_METADATA.get(model_key, {}).get(key, model_key)

# Define function at module level (not inside another function)
def _sanitize_infinities(X):
    """Replace ±inf with NaN for sklearn imputation."""
    return np.where(np.isfinite(X), X, np.nan)


def build_sanitize():
    """Create transformer to replace infinite values with NaN."""
    return FunctionTransformer(_sanitize_infinities, validate=False)

def get_models(args):
    use_multicountry = getattr(args, 'multi_countries', False) if args else False
    rf_config = TUNED_RF_PARAMS['multicountry'] if use_multicountry else TUNED_RF_PARAMS['tier1']

    #Include defaults if args not defined 
    n_estimators = getattr(args, 'tree_nestimators', None) if args else None #50
    n_estimators = n_estimators if n_estimators else rf_config['n_estimators']

    max_depth = getattr(args, 'tree_max_depth', None) if args else None #10
    max_depth = max_depth if max_depth else rf_config['max_depth']

    max_samples = getattr(args, 'tree_max_samples', None) if args else None
    
    max_features = getattr(args, 'tree_max_features', 'log2') if args else 'log2'
    max_features = max_features if max_features else rf_config['max_features']

    colsample = getattr(args, 'tree_colsample', 0.8) if args else 0.8

    sanitize = build_sanitize()
    models = {
        "logreg_l1": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", penalty="l1",
                                    C=0.5, max_iter=5000, class_weight="balanced"))
        ])
        ,
        "logreg_l2": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
        ]),

        "rf": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scaler", "passthrough"),  # trees don"t need scaling
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=rf_config['min_samples_split'], #0.02
                min_samples_leaf=rf_config['min_samples_leaf'], #0.01
                max_samples=max_samples,
                max_features=max_features,
                random_state=42,
                n_jobs=-1, 
                class_weight="balanced"
            ))
        ]),

        "rf_cal": Pipeline([
            ("sanitize", sanitize),                    # replace ±inf with NaN
            ("impute", SimpleImputer(strategy="median")),  # handle NaN
            ("scaler", "passthrough"),  # trees don"t need scaling
            ("clf", CalibratedClassifierCV(
                estimator=RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=0.02,
                    min_samples_leaf=0.01,
                    max_samples=max_samples,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                ),
                method='sigmoid',
                cv=3
            ))
        ]),
        'gb': Pipeline([
        ("sanitize", sanitize),
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", "passthrough"),
        ("clf", GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.01,
            subsample=0.5,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42
        ))
        ]),
        'gb_deep': Pipeline([
        ("sanitize", sanitize),
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", "passthrough"),
        ("clf", GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=20,
            min_samples_leaf=5,
            random_state=42
        ))
        ]),
        # XGBoost - same pipeline structure
        "xgb": Pipeline([
            ("sanitize", sanitize),
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", "passthrough"), 
            ("clf", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05, #0.01
                subsample=0.8, 
                colsample_bytree=colsample,  
                reg_alpha=0.1, #0.01 
                reg_lambda=1.0, #2.0
                #scale_pos_weight=1, 
                random_state=42,
                n_jobs=-1,
                tree_method='hist', 
                enable_categorical=False
            ))
        ]),
        
        # LightGBM - same pipeline structure
        "lgbm": Pipeline([
            ("sanitize", sanitize),
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", "passthrough"), 
            ("clf", LGBMClassifier(
                force_col_wise=True,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05, #0.01
                subsample=0.8,
                subsample_freq=1,  
                colsample_bytree=colsample,
                reg_alpha=0.5, #0.01 
                reg_lambda=1.0, #2.0
                num_leaves=15, #8  
                min_child_samples=20, #50  
                #class_weight='balanced',  
                random_state=42,
                n_jobs=-1,
                verbose=-1  # Suppress training logs
            ))
        ]),
        'nn' : Pipeline([
            ("sanitize", sanitize),
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),  # Critical for NNs
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            ))
        ])

    }
    return models