with activity as (
    select
        customer_id,
        avg(case when post = 0 then weekly_active_days end) as pre_avg_active_days_dbt,
        avg(case when post = 1 then weekly_active_days end) as post_avg_active_days_dbt,
        sum(case when post = 1 then weekly_sessions else 0 end) as post_sessions_total_dbt
    from {{ ref('stg_weekly_activity') }}
    group by 1
)
select c.*, a.pre_avg_active_days_dbt, a.post_avg_active_days_dbt, a.post_sessions_total_dbt
from {{ ref('stg_customers') }} c
left join activity a using (customer_id)
