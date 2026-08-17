from collections.abc import Callable

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    return ([name for name in NUMERIC_FEATURES if name in frame], [name for name in CATEGORICAL_FEATURES if name in frame])


def model_templates(frame: pd.DataFrame, random_state: int) -> dict[str, Pipeline]:
    numeric, categorical = feature_columns(frame)
    preprocess = ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False))]), categorical)])
    return {
        "logistic_regression": Pipeline([("preprocessor", clone(preprocess)), ("model", LogisticRegression(max_iter=2500, class_weight="balanced", C=.7, random_state=random_state))]),
        "random_forest": Pipeline([("preprocessor", clone(preprocess)), ("model", RandomForestClassifier(n_estimators=300, max_depth=9, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=random_state))]),
        "gradient_boosting": Pipeline([("preprocessor", clone(preprocess)), ("model", GradientBoostingClassifier(n_estimators=180, learning_rate=.05, max_depth=3, min_samples_leaf=6, subsample=.85, random_state=random_state))]),
    }


def fit_estimator(template: Pipeline, features: pd.DataFrame, labels: pd.Series) -> Pipeline:
    estimator = clone(template)
    if isinstance(estimator.named_steps["model"], GradientBoostingClassifier): estimator.fit(features, labels, model__sample_weight=compute_sample_weight(class_weight="balanced", y=labels))
    else: estimator.fit(features, labels)
    return estimator


def metrics(labels, predictions) -> dict[str, float]:
    return {"accuracy": accuracy_score(labels, predictions), "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0), "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0), "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0)}


def select_model(templates: dict[str, Pipeline], train_x: pd.DataFrame, train_y: pd.Series, validation_x: pd.DataFrame, validation_y: pd.Series) -> tuple[str, Pipeline, pd.DataFrame]:
    rows, fitted = [], {}
    for name, template in templates.items():
        fitted[name] = fit_estimator(template, train_x, train_y); rows.append({"model": name, **metrics(validation_y, fitted[name].predict(validation_x))})
    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False); best = results.iloc[0]["model"]; return best, fitted[best], results
