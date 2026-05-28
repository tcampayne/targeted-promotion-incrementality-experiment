# Experiment Spec: Targeted Discount Incrementality



## Objective

Estimate whether a simulated 10% discount increases revenue for high-LTV users.



## Unit of randomization

User-level.



## Eligibility

High-LTV = top 30% of users ranked by pre-period revenue (6 weeks).



## Periods

Pre: 6 weeks  

Post: 6 calendar weeks. Users may have fewer observed post-period weeks depending on purchase activity.



## Treatment

Simulated 10% discount applied to treated users in the post period.



## Outcomes

Primary: post-period revenue per user  

Secondary: AOV, orders/user



## Analysis Plan

Balance: mean diffs, SMDs, t-tests  

ATE: diff-in-means + OLS; 95% CI  

Robustness: Difference-in-Differences (user-week panel)

Business impact: cumulative revenue lift minus a simplified 10% discount-cost estimate.



