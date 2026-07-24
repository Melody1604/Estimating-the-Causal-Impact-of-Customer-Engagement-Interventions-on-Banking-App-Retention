select
    cast(customer_id as integer) as customer_id,
    cast(treated as integer) as treated,
    cast(retained_30d as integer) as retained_30d,
    cast(churned_30d as integer) as churned_30d,
    cast(age as integer) as age,
    cast(tenure_months as double) as tenure_months,
    cast(annual_income as double) as annual_income,
    cast(avg_balance as double) as avg_balance,
    cast(baseline_sessions_30d as double) as baseline_sessions_30d,
    cast(digital_transaction_share as double) as digital_transaction_share,
    cast(support_tickets_90d as integer) as support_tickets_90d,
    cast(product_count as integer) as product_count,
    cast(prior_nps as double) as prior_nps,
    cast(missed_payment_flag as integer) as missed_payment_flag,
    cast(metro_flag as integer) as metro_flag,
    cast(pre_avg_active_days as double) as pre_avg_active_days
from {{ ref('customers') }}
