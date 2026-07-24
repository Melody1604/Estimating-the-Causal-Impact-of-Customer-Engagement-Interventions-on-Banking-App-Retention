select
    cast(customer_id as integer) as customer_id,
    cast(week as integer) as week,
    cast(post as integer) as post,
    cast(treated as integer) as treated,
    cast(weekly_active_days as double) as weekly_active_days,
    cast(weekly_sessions as double) as weekly_sessions,
    cast(weekly_transactions as double) as weekly_transactions
from {{ ref('weekly_activity') }}
