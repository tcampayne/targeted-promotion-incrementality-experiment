-- sql/marts/mart_user_week_experiment_v2.sql
-- Purpose: analysis-ready weekly panel with experiment flags

drop table if exists marts.mart_user_week_experiment_v2;

create table marts.mart_user_week_experiment_v2 as
select
    m.user_id,
    m.week_index,
    m.pre_post_flag,
    e.high_ltv_flag,
    e.treatment_flag,
    m.revenue,
    m.orders,
    m.aov
from marts.mart_user_week_experiment m
left join core.exp_assignments e using (user_id);