\# Targeted Promotion Incrementality (A/B + Causal Inference)



\## Goal

Estimate the incremental revenue impact of a targeted 10% discount using a user-level randomized A/B experiment, packaged as a reproducible SQL pipeline (staging → core → marts).



\## Design (v1)

\- Dataset: Instacart public dataset

\- Eligibility: high-LTV users = top 30% by pre-period revenue (6 weeks)

\- Randomization: user-level, 50/50 treatment vs control

\- Post-period: 6 weeks

\- Primary KPI: post-period revenue per user

\- Secondary KPIs: orders/user, AOV



\## Repo Structure

\- `sql/staging`: raw ingestion tables

\- `sql/core`: cleaned + standardized business logic

\- `sql/marts`: experiment-ready fact tables

\- `analysis`: inference + plots

\- `docs`: specs + documentation



