"""Machine learning utilities for DataPilot Analytics."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)


def _prepare_features(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    id_like_keywords = ["id", "name", "date", "postal", "zip", "code"]
    drop_cols = [c for c in X.columns if any(k in c.lower() for k in id_like_keywords)]

    for col in X.select_dtypes(include="object").columns:
        if col not in drop_cols and X[col].nunique() > 50:
            drop_cols.append(col)

    X = X.drop(columns=drop_cols, errors="ignore")

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.select_dtypes(include=np.number)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def train_regression(df: pd.DataFrame, target: str):
    df = df.copy()

    q1, q3 = df[target].quantile(0.25), df[target].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        df = df[(df[target] >= lower) & (df[target] <= upper)]

    X, y = _prepare_features(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "r2": round(r2_score(y_test, preds), 3),
            "mae": round(mean_absolute_error(y_test, preds), 2),
            "model": model,
        }

    best_name = max(results, key=lambda n: results[n]["r2"])
    feature_importance = None
    if hasattr(results["Random Forest"]["model"], "feature_importances_"):
        feature_importance = pd.Series(
            results["Random Forest"]["model"].feature_importances_, index=X.columns
        ).sort_values(ascending=False)

    return results, best_name, feature_importance


def train_classification(df: pd.DataFrame, target: str):
    X, y = _prepare_features(df, target)

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 3),
            "recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 3),
            "f1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 3),
            "confusion_matrix": confusion_matrix(y_test, preds),
            "model": model,
        }

    best_name = max(results, key=lambda n: results[n]["f1"])
    feature_importance = None
    if hasattr(results["Random Forest"]["model"], "feature_importances_"):
        feature_importance = pd.Series(
            results["Random Forest"]["model"].feature_importances_, index=X.columns
        ).sort_values(ascending=False)

    return results, best_name, feature_importance