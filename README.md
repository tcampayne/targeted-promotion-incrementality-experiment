# Targeted Promotion Incrementality (A/B + Causal Inference)

## Overview

This project estimates the incremental revenue impact of a **10% targeted discount** offered to high-LTV users using a user-level randomized A/B experiment.

The analysis is implemented using a reproducible SQL pipeline (**staging → core → marts**) and validated with multiple causal inference methods, including post-period ATE, clustered Difference-in-Differences (DiD), and fixed-effects panel regression.

---

## Experimental Design

- **Dataset:** Instacart public dataset  
- **Eligibility:** Top 30% of users by pre-period (6-week) revenue  
- **Randomization:** User-level, 50/50 treatment vs control  
- **Post-period:** 6 weeks  
- **Primary KPI:** Post-period revenue per user  
- **Secondary KPIs:** Orders per user, AOV  

---

## Methodology

We estimate incremental lift using:

- **Post-period Average Treatment Effect (ATE)**
- **95% Confidence Intervals**
- **Clustered Difference-in-Differences (DiD)**
  - Standard errors clustered at the user level
- **Fixed Effects DiD (within-user demeaning)**
  - Controls for time-invariant user heterogeneity

---

## Key Results

- **Post-period ATE lift:** **+$28.50 per user (+8.16%)**
- **95% CI:** **[$23.90, $33.11]**
- **Clustered DiD (weekly lift):** **+$8.74**
- **Fixed Effects DiD (weekly lift):** **+$4.53**

Even under conservative fixed-effects estimates, the promotion generates statistically significant and economically meaningful incremental revenue.

The implied 6-week lift from the FE estimate (~$4.53 × 6 ≈ $27.18) aligns closely with the ATE estimate ($28.50), strengthening causal credibility.

---

## Business Implications

- The targeted discount drives meaningful incremental revenue among high-LTV users.
- Results remain robust after controlling for time trends and user-level heterogeneity.
- Scaled to 100,000 high-LTV users, the conservative FE estimate implies **~$2.7M incremental revenue over six weeks**.

---

## Repo Structure

- `sql/staging` — raw ingestion tables  
- `sql/core` — cleaned and standardized transformations  
- `sql/marts` — experiment-ready fact tables  
- `analysis` — statistical inference and visualization  
- `docs` — experiment specifications and documentation  

---

## Reproducibility

1. Load raw data into staging tables  
2. Run SQL transformations (`staging → core → marts`)  
3. Execute `analysis/incrementality.ipynb`  

All statistical estimates are reproducible from experiment-ready mart tables.

---

## Technical Stack

- SQL (PostgreSQL)  
- Python (pandas, statsmodels, numpy, matplotlib)  
- Cluster-robust inference  
- Panel data modeling  