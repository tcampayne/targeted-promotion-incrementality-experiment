-- sql/marts/mart_user_week_experiment.sql
-- Purpose: Aggregate order-level revenue to user-week panel dataset

drop table if exists marts.mart_user_week_experiment;

create table marts.mart_user_week_experiment as
select
    user_id,
    week_index,
    pre_post_flag,
    sum(order_revenue) as revenue,
    count(*) as orders,
    case
        when count(*) > 0
        then sum(order_revenue)::numeric / count(*)
    end as aov
from core.order_revenue
group by 1,2,3;