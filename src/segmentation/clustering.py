"""Customer segmentation using standardised K-means."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    assignments: pd.DataFrame
    profiles: pd.DataFrame
    silhouette: float


def run_clustering(df: pd.DataFrame, features: list[str], n_clusters: int = 4, seed: int = 42) -> ClusterResult:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)),
        ]
    )
    labels = pipe.fit_predict(df[features])
    silhouette = float(silhouette_score(pipe.named_steps["scale"].transform(df[features]), labels))

    assignments = df[["customer_id"]].copy()
    assignments["segment"] = labels
    profile_source = df[["customer_id", "retained_30d"] + features].copy()
    profile_source["segment"] = labels
    profiles = profile_source.groupby("segment").agg(
        customers=("customer_id", "count"),
        retention_rate=("retained_30d", "mean"),
        **{f"avg_{col}": (col, "mean") for col in features},
    ).reset_index()
    return ClusterResult(assignments, profiles, silhouette)
