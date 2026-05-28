# A/B Promotion Incrementality Analysis

This project estimates whether a simulated 10% discount for high-LTV users creates incremental revenue. It uses an A/B testing workflow, a Python notebook, and a Streamlit dashboard to connect the statistical results to a clear business recommendation.

> **Data note:** Revenue is simulated. An 8% uplift is applied to treated high-LTV users in the post-period, so results should be read as a portfolio experiment simulation rather than real transaction evidence.

---

## Key Findings

- **+8.2% simulated revenue lift** — the A/B test shows **+$28.50 per user** in incremental revenue over the post-period (95% confidence interval: [$23.90, $33.11])
- **Multiple regression methods agree** — difference-in-differences (DiD) panel estimates range from +$8.41 to +$8.74 per user per week across four model specifications, consistent with the A/B test result and confirming the estimators successfully recover the simulated uplift
- **Blanket discount is not worth it** under a simplified cost model: the $28.50 revenue lift does not cover the ~$37.79 estimated discount cost per user (net impact: −$9.29/user)
- **User-level variation in lift exists** — a causal forest model suggests some users respond much more strongly than others, but this is exploratory and would need a follow-up experiment before informing any targeting decision

## Business Recommendation

- **Do not broadly roll out** the 10% blanket discount under a short-term revenue-cost objective
- **Test lower discount levels or a targeting strategy** — higher-spend users show larger absolute lift, but model-based targeting needs validation before deployment
- **Track margin, retention, and downstream behavior** — this analysis uses simplified revenue-cost assumptions with no cannibalization, margin, or retention effects

---

## Screenshots

**Overview** — key metrics at a glance

![Overview](docs/screenshots/overview.png)

**Business Impact** — cumulative lift vs. discount cost

![Business Impact](docs/screenshots/business_impact.png)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data pipeline | PostgreSQL, SQL marts (staging → core → marts) |
| Analysis | Python, pandas, statsmodels, scikit-learn, EconML (causal forests) |
| Dashboard | Streamlit |

---

## Data Flow

```
data/raw/ (Instacart public dataset)
  → PostgreSQL  (staging → core → marts)
  → analysis/02_inference.ipynb  (reads DB, exports processed CSVs)
  → app/streamlit_app.py  (reads processed CSVs, computes synthetic control live)
```

---

## How to Run

**1. Set up environment variables**

Create a `.env` file at the project root:

```
DB_NAME=...
DB_USER=...
DB_PASSWORD=...
DB_HOST=...
DB_PORT=...
```

**2. Run the notebook**

Open and run `analysis/02_inference.ipynb` end-to-end. This exports:
- `data/processed/panel_df.csv` — user-week panel (~440k rows)
- `data/processed/event_study.csv` — event-study results

**3. Launch the Streamlit app**

```powershell
.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

**Live app:** https://targeted-promotion-incrementality-experiment-ltfx4qbfgnsn5ey2x.streamlit.app

---

## Notebook

[`analysis/02_inference.ipynb`](analysis/02_inference.ipynb)

Covers:
- A/B test estimate — post-period difference in means (primary result)
- Pre-treatment trend diagnostics (event study)
- Difference-in-differences robustness checks — Naive DiD, User FE, TWFE, Weighted DiD
- Synthetic control robustness check
- Causal forest — user-level treatment effect variation (exploratory)
- Business impact and ROI breakdown
