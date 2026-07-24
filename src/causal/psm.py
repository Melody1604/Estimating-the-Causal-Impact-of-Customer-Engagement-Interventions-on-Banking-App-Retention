"""Propensity-score matching for the ATT on 30-day retention."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PSMResult:
    att: float
    ci_low: float
    ci_high: float
    n_matched: int
    matched_pairs: pd.DataFrame
    balance: pd.DataFrame


def standardized_mean_difference(t: pd.Series, c: pd.Series) -> float:
    pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return float((t.mean() - c.mean()) / pooled) if pooled > 0 else 0.0


def run_psm(
    df: pd.DataFrame,
    features: list[str],
    outcome: str = "retained_30d",
    caliper: float = 0.2,
    n_bootstrap: int = 600,
    seed: int = 42,
) -> PSMResult:
    X = df[features]
    treatment = df["treated"].astype(int)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    model.fit(X, treatment)
    pscore = np.clip(model.predict_proba(X)[:, 1], 1e-5, 1 - 1e-5)
    logit_ps = np.log(pscore / (1 - pscore))

    treated_idx = np.flatnonzero(treatment.to_numpy() == 1)
    control_idx = np.flatnonzero(treatment.to_numpy() == 0)
    nn = NearestNeighbors(n_neighbors=1).fit(logit_ps[control_idx].reshape(-1, 1))
    distance, neighbor = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))
    threshold = caliper * np.std(logit_ps, ddof=1)
    keep = distance.ravel() <= threshold

    matched_t = treated_idx[keep]
    matched_c = control_idx[neighbor.ravel()[keep]]
    pairs = pd.DataFrame(
        {
            "treated_index": matched_t,
            "control_index": matched_c,
            "treated_customer_id": df.iloc[matched_t]["customer_id"].to_numpy(),
            "control_customer_id": df.iloc[matched_c]["customer_id"].to_numpy(),
            "treated_outcome": df.iloc[matched_t][outcome].to_numpy(),
            "control_outcome": df.iloc[matched_c][outcome].to_numpy(),
            "ps_distance": distance.ravel()[keep],
        }
    )
    pair_effects = pairs["treated_outcome"] - pairs["control_outcome"]
    att = float(pair_effects.mean())

    rng = np.random.default_rng(seed)
    boot = np.array(
        [pair_effects.iloc[rng.integers(0, len(pair_effects), len(pair_effects))].mean() for _ in range(n_bootstrap)]
    )
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    before = []
    after = []
    for feature in features:
        before.append(
            {
                "feature": feature,
                "stage": "before",
                "smd": standardized_mean_difference(df.loc[treatment == 1, feature], df.loc[treatment == 0, feature]),
            }
        )
        after.append(
            {
                "feature": feature,
                "stage": "after",
                "smd": standardized_mean_difference(df.iloc[matched_t][feature], df.iloc[matched_c][feature]),
            }
        )

    return PSMResult(att, float(ci_low), float(ci_high), len(pairs), pairs, pd.DataFrame(before + after))
