"""End-to-end local pipeline for the banking retention causal project."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.causal.psm import run_psm
from src.causal.did import run_did
from src.causal.hte import estimate_hte
from src.data.generate_synthetic import generate_data
from src.models.churn import train_churn_model
from src.segmentation.clustering import run_clustering
from src.reporting import plot_balance, plot_event_study, plot_hte
from src.utils.io import ensure_dirs, load_data, save_json

CAUSAL_FEATURES = [
    "age",
    "tenure_months",
    "annual_income",
    "avg_balance",
    "baseline_sessions_30d",
    "digital_transaction_share",
    "support_tickets_90d",
    "product_count",
    "prior_nps",
    "missed_payment_flag",
    "metro_flag",
    "pre_avg_active_days",
]

CLUSTER_FEATURES = [
    "tenure_months",
    "avg_balance",
    "baseline_sessions_30d",
    "digital_transaction_share",
    "support_tickets_90d",
    "product_count",
    "prior_nps",
]

PREDICTIVE_FEATURES = [
    "age",
    "tenure_months",
    "annual_income",
    "avg_balance",
    "baseline_sessions_30d",
    "digital_transaction_share",
    "support_tickets_90d",
    "product_count",
    "prior_nps",
    "missed_payment_flag",
    "metro_flag",
    "pre_avg_active_days",
]


def run(customer_path: Path, panel_path: Path) -> dict:
    ensure_dirs()
    if not customer_path.exists() or not panel_path.exists():
        customer, panel = generate_data()
        customer_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        customer.to_csv(customer_path, index=False)
        panel.to_csv(panel_path, index=False)
    else:
        customer, panel = load_data(customer_path, panel_path)

    psm = run_psm(customer, CAUSAL_FEATURES)
    did = run_did(panel)
    hte = estimate_hte(customer, CAUSAL_FEATURES)
    cluster = run_clustering(customer, CLUSTER_FEATURES)
    churn = train_churn_model(customer, PREDICTIVE_FEATURES)

    psm.balance.to_csv("reports/tables/psm_balance.csv", index=False)
    psm.matched_pairs.to_csv("reports/tables/psm_matched_pairs.csv", index=False)
    did.event_study.to_csv("reports/tables/did_event_study.csv", index=False)
    hte.customer_effects.to_csv("reports/tables/customer_cate.csv", index=False)
    hte.subgroup_effects.to_csv("reports/tables/hte_subgroups.csv", index=False)
    cluster.assignments.to_csv("reports/tables/customer_segments.csv", index=False)
    cluster.profiles.to_csv("reports/tables/segment_profiles.csv", index=False)
    churn.feature_importance.to_csv("reports/tables/churn_feature_importance.csv", index=False)
    churn.scored_customers.to_csv("reports/tables/churn_scored_customers.csv", index=False)

    # Demonstrate why high churn risk is not the same as high treatment uplift.
    targeting = churn.scored_customers.merge(
        hte.customer_effects[["customer_id", "estimated_cate"]], on="customer_id", how="inner"
    )
    target_n = max(1, int(0.20 * len(targeting)))
    high_risk = targeting.nlargest(target_n, "predicted_churn_probability")
    high_uplift = targeting.nlargest(target_n, "estimated_cate")
    random_benchmark = targeting.sample(target_n, random_state=42)
    targeting_comparison = pd.DataFrame(
        [
            {
                "strategy": "highest_predicted_churn",
                "customers": len(high_risk),
                "avg_predicted_churn": high_risk["predicted_churn_probability"].mean(),
                "avg_estimated_uplift": high_risk["estimated_cate"].mean(),
                "observed_churn_rate": high_risk["actual_churn"].mean(),
            },
            {
                "strategy": "highest_estimated_uplift",
                "customers": len(high_uplift),
                "avg_predicted_churn": high_uplift["predicted_churn_probability"].mean(),
                "avg_estimated_uplift": high_uplift["estimated_cate"].mean(),
                "observed_churn_rate": high_uplift["actual_churn"].mean(),
            },
            {
                "strategy": "random_20_percent",
                "customers": len(random_benchmark),
                "avg_predicted_churn": random_benchmark["predicted_churn_probability"].mean(),
                "avg_estimated_uplift": random_benchmark["estimated_cate"].mean(),
                "observed_churn_rate": random_benchmark["actual_churn"].mean(),
            },
        ]
    )
    targeting_comparison.to_csv("reports/tables/targeting_strategy_comparison.csv", index=False)

    plot_balance(psm.balance)
    plot_event_study(did.event_study)
    plot_hte(hte.subgroup_effects)

    summary = {
        "data": {"customers": len(customer), "panel_rows": len(panel), "treatment_rate": customer["treated"].mean()},
        "psm": {"att_retention": psm.att, "ci_95": [psm.ci_low, psm.ci_high], "matched_treated": psm.n_matched},
        "did": {
            "weekly_active_days_lift": did.estimate,
            "standard_error": did.std_error,
            "ci_95": [did.ci_low, did.ci_high],
            "p_value": did.p_value,
        },
        "hte": {
            "highest_value_group": hte.subgroup_effects.iloc[0].to_dict(),
            "lowest_value_group": hte.subgroup_effects.iloc[-1].to_dict(),
        },
        "clustering": {"n_segments": 4, "silhouette_score": cluster.silhouette},
        "churn_prediction": churn.metrics,
        "targeting_comparison": targeting_comparison.set_index("strategy").to_dict(orient="index"),
        "interpretation_guardrail": "The XGBoost model predicts churn risk; it does not identify intervention effects.",
    }
    save_json(summary, "reports/metrics.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=Path, default=Path("data/raw/customers.csv"))
    parser.add_argument("--panel", type=Path, default=Path("data/raw/weekly_activity.csv"))
    args = parser.parse_args()
    summary = run(args.customers, args.panel)
    print(pd.Series(summary, dtype="object").to_string())


if __name__ == "__main__":
    main()
