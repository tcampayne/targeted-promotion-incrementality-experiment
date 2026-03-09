-- sql/marts/mart_user_week_experiment_sim.sql
-- Purpose: simulate treatment effect by applying uplift to treated high-LTV users in post period

drop table if exists marts.mart_user_week_experiment_sim;

create table marts.mart_user_week_experiment_sim as
select
    *,
    case
        when high_ltv_flag = 1
         and treatment_flag = 1
         and pre_post_flag = 'post'
        then revenue * 1.08
        else revenue
    end as revenue_sim
from marts.mart_user_week_experiment_v2;