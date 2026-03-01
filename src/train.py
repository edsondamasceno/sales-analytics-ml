# src/train.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_models(X_train, y_train, preprocessor):

    models = {}

    # =====================
    # Random Forest
    # =====================
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42))
        ]
    )

    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20]
    }

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    models["RandomForest"] = rf_grid.best_estimator_

    print("\nRandomForest")
    print("Best Params:", rf_grid.best_params_)

    # =====================
    # XGBoost
    # =====================
    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            ))
        ]
    )

    xgb_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.05, 0.1]
    }

    xgb_grid = GridSearchCV(
        xgb_pipeline,
        xgb_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    xgb_grid.fit(X_train, y_train)

    models["XGBoost"] = xgb_grid.best_estimator_

    print("\nXGBoost")
    print("Best Params:", xgb_grid.best_params_)

    return models