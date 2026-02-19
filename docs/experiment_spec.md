\# Experiment Spec: Targeted Discount Incrementality



\## Objective

Measure incremental lift in revenue from offering a 10% discount to high-LTV users.



\## Unit of randomization

User-level.



\## Eligibility

High-LTV = top 30% of users ranked by pre-period revenue (6 weeks).



\## Periods

Pre: 6 weeks  

Post: 6 weeks



\## Treatment

10% discount applied to post-period orders (simulated).



\## Outcomes

Primary: post-period revenue per user  

Secondary: AOV, orders/user



\## Analysis Plan

Balance: mean diffs, SMDs, t-tests  

ATE: diff-in-means + OLS; 95% CI  

Robustness: Difference-in-Differences (user-week panel)



