"""Generate a reproducible synthetic banking customer panel.

The dataset is designed for portfolio demonstration only. It contains observed
confounding, heterogeneous treatment effects, repeated pre/post observations,
and a binary 30-day retention outcome.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit


def generate_data(n_customers: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customer_id = np.arange(1, n_customers + 1)

    # Four latent behavioural archetypes create realistic but imperfect
    # structure for the unsupervised segmentation task.
    archetype = rng.choice(4, size=n_customers, p=[0.30, 0.22, 0.20, 0.28])
    age_mu = np.array([29, 49, 38, 35])
    tenure_mu = np.array([16, 62, 31, 22])
    income_mu = np.array([76000, 138000, 72000, 59000])
    balance_mu = np.array([4500, 32000, 8000, 3000])
    sessions_mu = np.array([16, 8, 9, 4.5])
    digital_mu = np.array([0.90, 0.64, 0.59, 0.43])
    tickets_mu = np.array([0.45, 0.40, 2.25, 0.85])
    products_mu = np.array([2.2, 4.1, 2.5, 1.5])
    nps_mu = np.array([44, 34, -8, 10])

    age = np.clip(rng.normal(age_mu[archetype], 7), 18, 80).round().astype(int)
    tenure_months = np.clip(rng.normal(tenure_mu[archetype], 12), 1, 120).round(1)
    income = np.clip(rng.lognormal(np.log(income_mu[archetype]), 0.27), 22000, 320000).round(0)
    balance = np.clip(rng.lognormal(np.log(balance_mu[archetype]), 0.65), 0, 250000).round(2)
    baseline_sessions = np.clip(rng.normal(sessions_mu[archetype], 2.7), 0, 35).round(1)
    digital_share = np.clip(rng.normal(digital_mu[archetype], 0.11), 0.05, 0.99).round(3)
    support_tickets = np.clip(rng.poisson(tickets_mu[archetype]), 0, 8)
    product_count = np.clip(rng.normal(products_mu[archetype], 0.8).round(), 1, 7).astype(int)
    prior_nps = np.clip(rng.normal(nps_mu[archetype], 19), -100, 100).round(0)
    missed_payments = rng.binomial(1, expit(-2.5 + 0.42 * support_tickets - 0.000004 * income + 0.25 * (archetype == 3)))
    metro = rng.binomial(1, np.array([0.86, 0.68, 0.72, 0.62])[archetype])

    # Treatment is targeted, creating observed confounding.
    treatment_logit = (
        -1.35
        + 0.050 * baseline_sessions
        + 0.58 * digital_share
        + 0.16 * support_tickets
        - 0.005 * tenure_months
        - 0.0035 * prior_nps
        + 0.22 * missed_payments
        + rng.normal(0, 0.22, n_customers)
    )
    propensity_true = expit(treatment_logit)
    treated = rng.binomial(1, propensity_true)

    # True probability-scale treatment effect used only by the simulator.
    true_tau = (
        0.030
        + 0.045 * (baseline_sessions < 8)
        + 0.030 * (tenure_months < 18)
        - 0.025 * (support_tickets >= 3)
        + 0.018 * (digital_share > 0.75)
    )
    true_tau = np.clip(true_tau, 0.005, 0.14)

    customer = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "tenure_months": tenure_months,
            "annual_income": income,
            "avg_balance": balance,
            "baseline_sessions_30d": baseline_sessions,
            "digital_transaction_share": digital_share,
            "support_tickets_90d": support_tickets,
            "product_count": product_count,
            "prior_nps": prior_nps,
            "missed_payment_flag": missed_payments,
            "metro_flag": metro,
            "treated": treated,
            "true_propensity": propensity_true.round(5),
            "true_retention_effect": true_tau.round(5),
        }
    )

    weeks = np.arange(-8, 9)
    rows: list[pd.DataFrame] = []
    customer_random_effect = rng.normal(0, 0.55, n_customers)

    for week in weeks:
        post = int(week >= 0)
        seasonality = 0.15 * np.sin((week + 8) / 2.4)
        base_linear = (
            0.20
            + 0.070 * baseline_sessions
            + 0.34 * digital_share
            + 0.045 * product_count
            - 0.060 * support_tickets
            + 0.0035 * prior_nps
            + 0.12 * metro
            + customer_random_effect
            + seasonality
            - 0.018 * max(week, 0)
        )
        intervention_lift = treated * post * (0.34 + 2.2 * true_tau)
        active_days = np.clip(base_linear + intervention_lift + rng.normal(0, 0.62, n_customers), 0, 7)
        sessions = np.clip(active_days * rng.normal(2.1, 0.32, n_customers) + rng.poisson(1.0, n_customers), 0, 50)
        transactions = np.clip(sessions * rng.normal(0.65, 0.11, n_customers), 0, 35)

        rows.append(
            pd.DataFrame(
                {
                    "customer_id": customer_id,
                    "week": week,
                    "post": post,
                    "treated": treated,
                    "weekly_active_days": active_days.round(3),
                    "weekly_sessions": sessions.round(2),
                    "weekly_transactions": transactions.round(2),
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)

    pre = panel.loc[panel["week"] < 0].groupby("customer_id")["weekly_active_days"].mean()
    post = panel.loc[panel["week"] >= 0].groupby("customer_id")["weekly_active_days"].mean()
    post_sessions = panel.loc[panel["week"] >= 0].groupby("customer_id")["weekly_sessions"].sum()

    base_retention_logit = (
        -2.05
        + 0.34 * pre.to_numpy()
        + 0.050 * baseline_sessions
        + 0.75 * digital_share
        + 0.12 * product_count
        - 0.26 * support_tickets
        - 0.95 * missed_payments
        + 0.006 * prior_nps
        + 0.0000015 * income
        + 0.10 * metro
        + rng.normal(0, 0.28, n_customers)
    )
    p0 = expit(base_retention_logit)
    p1 = np.clip(p0 + true_tau, 0.01, 0.99)
    observed_retention_probability = np.where(treated == 1, p1, p0)
    retained_30d = rng.binomial(1, observed_retention_probability)

    customer["pre_avg_active_days"] = pre.to_numpy().round(3)
    customer["post_avg_active_days"] = post.to_numpy().round(3)
    customer["post_sessions_total"] = post_sessions.to_numpy().round(2)
    customer["retained_30d"] = retained_30d
    customer["churned_30d"] = 1 - retained_30d

    return customer, panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-customers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    customer, panel = generate_data(args.n_customers, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    customer.to_csv(args.output_dir / "customers.csv", index=False)
    panel.to_csv(args.output_dir / "weekly_activity.csv", index=False)

    # Keep copies in dbt seeds so the SQL transformation layer is runnable.
    seed_dir = Path("dbt/seeds")
    seed_dir.mkdir(parents=True, exist_ok=True)
    customer.to_csv(seed_dir / "customers.csv", index=False)
    panel.to_csv(seed_dir / "weekly_activity.csv", index=False)
    print(f"Generated {len(customer):,} customers and {len(panel):,} panel rows.")


if __name__ == "__main__":
    main()
