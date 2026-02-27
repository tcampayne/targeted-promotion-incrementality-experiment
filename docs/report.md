# Targeted Promotion Incrementality Analysis

## 1. Objective
Estimate the causal impact of a 10% discount offered to high-LTV users on post-period revenue.

## 2. Experimental Design
- Dataset: Instacart public dataset  
- Unit of randomization: User-level  
- Eligibility: Top 30% of users ranked by pre-period revenue (6 weeks)  
- Randomization: 50/50 treatment vs control  
- Treatment: 10% discount applied to post-period revenue for treated users  
- Post-period: 6 weeks  

Revenue adjustments were simulated directly in SQL and incorporated into an experiment-ready mart table.

## 3. Validation
Pre-period revenue, orders, and AOV were statistically equivalent between treatment and control groups, supporting random assignment validity.

## 4. Results

### Difference-in-Means (ATE)
- Control Mean (Post): ~$349  
- Treatment Mean (Post): ~$378  
- Absolute Lift: ~$28.5 per user  
- Percent Lift: ~8.2%  
- 95% CI: [$23.9, $33.1]

The confidence interval does not include zero, indicating statistically significant lift.

### Difference-in-Differences
- Weekly incremental lift: ~$8.74  
- p < 0.001  

This confirms the treatment effect while controlling for time trends.

## 5. Business Impact
Estimated incremental revenue over 6 weeks:

~$28.5 × 55,102 high-LTV users ≈ $1.57M incremental revenue.

## 6. Limitations
- Offline simulation (no live deployment) 
- Assumes parallel trends in DiD 
- Simulated pricing due to lack of true product prices