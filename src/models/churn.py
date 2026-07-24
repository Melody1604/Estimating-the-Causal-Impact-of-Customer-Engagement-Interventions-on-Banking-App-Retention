"""Predictive churn model. This module predicts risk; it does not estimate causality."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class ChurnResult:
    metrics: dict
    feature_importance: pd.DataFrame
    scored_customers: pd.DataFrame


def train_churn_model(df: pd.DataFrame, features: list[str], seed: int = 42) -> ChurnResult:
    X = df[features]
    y = df["churned_30d"]
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["customer_id"], test_size=0.25, stratify=y, random_state=seed
    )

    model = XGBClassifier(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        min_child_weight=4,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, pred)),
        "average_precision": float(average_precision_score(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, pred)),
        "test_customers": int(len(y_test)),
        "churn_rate_test": float(y_test.mean()),
    }

    importance = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    scored = pd.DataFrame({"customer_id": id_test.to_numpy(), "actual_churn": y_test.to_numpy(), "predicted_churn_probability": pred})

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/xgboost_churn.joblib")
    with open("models/churn_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    return ChurnResult(metrics, importance, scored)
