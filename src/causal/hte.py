"""Exploratory heterogeneous treatment effects using cross-fitted AIPW.

Customer-level CATE predictions support ranking. Pre-defined subgroup estimates
are computed directly from cross-fitted AIPW pseudo-outcomes with normal-approximation
confidence intervals. These remain observational estimates and should be validated
with a randomised holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class HTEResult:
    customer_effects: pd.DataFrame
    subgroup_effects: pd.DataFrame


def estimate_hte(df: pd.DataFrame, features: list[str], outcome: str = "retained_30d", seed: int = 42) -> HTEResult:
    X = df[features].to_numpy()
    t = df["treated"].to_numpy().astype(int)
    y = df[outcome].to_numpy().astype(float)
    n = len(df)

    e_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for train, test in kf.split(X):
        prop = Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=2000))])
        prop.fit(X[train], t[train])
        e_hat[test] = prop.predict_proba(X[test])[:, 1]

        m0 = RandomForestRegressor(n_estimators=180, min_samples_leaf=25, random_state=seed, n_jobs=-1)
        m1 = RandomForestRegressor(n_estimators=180, min_samples_leaf=25, random_state=seed + 1, n_jobs=-1)
        m0.fit(X[train][t[train] == 0], y[train][t[train] == 0])
        m1.fit(X[train][t[train] == 1], y[train][t[train] == 1])
        mu0_hat[test] = m0.predict(X[test])
        mu1_hat[test] = m1.predict(X[test])

    e_hat = np.clip(e_hat, 0.03, 0.97)
    pseudo = (mu1_hat - mu0_hat) + t * (y - mu1_hat) / e_hat - (1 - t) * (y - mu0_hat) / (1 - e_hat)

    # Cross-fit the second-stage model so customer rankings are out-of-sample.
    cate = np.zeros(n)
    kf_cate = KFold(n_splits=5, shuffle=True, random_state=seed + 11)
    for fold, (train, test) in enumerate(kf_cate.split(X)):
        cate_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=45,
            random_state=seed + 20 + fold,
            n_jobs=-1,
        )
        cate_model.fit(X[train], pseudo[train])
        cate[test] = cate_model.predict(X[test])

    effects = df[["customer_id", "baseline_sessions_30d", "tenure_months", "digital_transaction_share", "support_tickets_90d"]].copy()
    effects["estimated_cate"] = cate
    effects["aipw_pseudo_outcome"] = pseudo
    effects["effect_decile"] = pd.qcut(effects["estimated_cate"], 10, labels=False, duplicates="drop") + 1

    effects["engagement_band"] = pd.cut(
        effects["baseline_sessions_30d"],
        bins=[-np.inf, 6, 12, np.inf],
        labels=["low", "medium", "high"],
    )
    effects["tenure_band"] = pd.cut(
        effects["tenure_months"],
        bins=[-np.inf, 18, 48, np.inf],
        labels=["new", "established", "long_tenure"],
    )

    subgroup = (
        effects.groupby(["engagement_band", "tenure_band"], observed=True)
        .agg(
            customers=("customer_id", "count"),
            estimated_effect=("aipw_pseudo_outcome", "mean"),
            effect_sd=("aipw_pseudo_outcome", "std"),
        )
        .reset_index()
    )
    subgroup["standard_error"] = subgroup["effect_sd"] / np.sqrt(subgroup["customers"])
    subgroup["ci_low"] = subgroup["estimated_effect"] - 1.96 * subgroup["standard_error"]
    subgroup["ci_high"] = subgroup["estimated_effect"] + 1.96 * subgroup["standard_error"]
    subgroup = subgroup.drop(columns="effect_sd").sort_values("estimated_effect", ascending=False)
    return HTEResult(effects, subgroup)
