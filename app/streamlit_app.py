import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="A/B Promo Incrementality", layout="wide")

# ============================================================
# Helpers
# ============================================================

@st.cache_data
def load_panel_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {
        "user_id",
        "treatment_flag",
        "week_index",
        "revenue_sim",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    if "post" not in df.columns:
        # Assumes weeks 1-6 are pre, 7-12 are post
        df["post"] = (df["week_index"] >= 7).astype(int)

    if "pre_post_flag" not in df.columns:
        df["pre_post_flag"] = np.where(df["post"] == 1, "post", "pre")

    if "interaction" not in df.columns:
        df["interaction"] = df["treatment_flag"] * df["post"]

    if "event_time" not in df.columns:
        # 1-6 -> -6:-1, 7-12 -> 0:5
        df["event_time"] = df["week_index"] - 7

    return df


def ci_bounds(coef: float, se: float) -> tuple[float, float]:
    return coef - 1.96 * se, coef + 1.96 * se


@st.cache_data
def compute_post_ate(df: pd.DataFrame) -> dict:
    post_df = df[df["post"] == 1].copy()

    treated = post_df.loc[post_df["treatment_flag"] == 1, "revenue_sim"]
    control = post_df.loc[post_df["treatment_flag"] == 0, "revenue_sim"]

    coef = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    ci_low, ci_high = ci_bounds(coef, se)

    return {
        "coef": float(coef),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
    }


@st.cache_data
def compute_model_table() -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "Model": ["Naive DiD", "User FE DiD", "TWFE DiD", "Weighted DiD"],
            "Lift ($)": [8.74, 4.53, -1.58, 8.61],
            "95% CI Lower": [np.nan, np.nan, np.nan, 7.76],
            "95% CI Upper": [np.nan, np.nan, np.nan, 9.47],
            "Interpretation": [
                "Baseline DiD; likely upward biased",
                "Controls for time-invariant user heterogeneity",
                "Two-way FE robustness check; estimate turns negative",
                "Reweighted robustness check; positive but model-dependent",
            ],
        }
    )
    return results


@st.cache_data
def compute_event_study_series(df: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    event_df = df.copy()

    # Make sure event_time exists and is integer
    event_df["event_time"] = event_df["event_time"].astype(int)

    # Use week -1 as the omitted reference period
    event_df["event_time_cat"] = event_df["event_time"].astype(str)
    event_df["event_time_cat"] = pd.Categorical(
        event_df["event_time_cat"],
        categories=[str(x) for x in sorted(event_df["event_time"].unique())],
        ordered=True
    )

    # TWFE event study:
    # revenue ~ event-time x treatment interactions + user FE + week FE
    model = smf.ols(
        'revenue_sim ~ C(event_time_cat, Treatment(reference="-1")):treatment_flag + C(user_id) + C(week_index)',
        data=event_df
    ).fit(cov_type="cluster", cov_kwds={"groups": event_df["user_id"]})

    rows = []

    for t in sorted(event_df["event_time"].unique()):
        if t == -1:
            rows.append({"event_time": t, "coef": 0.0, "se": 0.0})
            continue

        term = f'C(event_time_cat, Treatment(reference="-1"))[T.{t}]:treatment_flag'

        if term in model.params.index:
            rows.append(
                {
                    "event_time": t,
                    "coef": float(model.params[term]),
                    "se": float(model.bse[term]),
                }
            )

    out = pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)
    return out


@st.cache_data
def compute_quartile_hte(df: pd.DataFrame) -> pd.DataFrame:
    user_pre = (
        df[df["post"] == 0]
        .groupby("user_id", as_index=False)
        .agg(
            baseline_revenue=("revenue_sim", "mean"),
            treatment_flag=("treatment_flag", "max"),
        )
    )

    user_post = (
        df[df["post"] == 1]
        .groupby("user_id", as_index=False)
        .agg(post_revenue=("revenue_sim", "mean"))
    )

    user_df = user_pre.merge(user_post, on="user_id", how="inner")
    user_df["quartile"] = pd.qcut(
        user_df["baseline_revenue"],
        4,
        labels=["Q1 Low", "Q2", "Q3", "Q4 High"],
    )

    out = (
        user_df.groupby(["quartile", "treatment_flag"], as_index=False)["post_revenue"]
        .mean()
        .pivot(index="quartile", columns="treatment_flag", values="post_revenue")
        .reset_index()
        .rename(columns={0: "control", 1: "treated"})
    )
    out["lift"] = out["treated"] - out["control"]
    out["pct_lift"] = out["lift"] / out["control"] * 100
    return out


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("A/B Promo Incrementality")

default_path = "data/processed/panel_df.csv"
file_path = st.sidebar.text_input("CSV path", value=default_path)

st.sidebar.markdown("### App sections")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "ATE",
        "Event Study",
        "Model Comparison",
        "Business Impact",
        "HTE",
    ],
)

# ============================================================
# Load
# ============================================================

st.title("10% Discount Incrementality Analysis")
st.caption("Streamlit dashboard for the high-LTV discount causal inference project")

if not Path(file_path).exists():
    st.warning(
        f"Could not find `{file_path}`. Export your analysis dataset to CSV first, then rerun the app."
    )
    st.stop()

try:
    panel_df = load_panel_data(file_path)
except Exception as e:
    st.error(f"Failed to load panel data: {e}")
    st.stop()

ate = compute_post_ate(panel_df)
model_table = compute_model_table()
event_df = compute_event_study_series(panel_df)
hte_df = compute_quartile_hte(panel_df)

# Business impact based on experimental ATE
treated_post_mean = panel_df.loc[
    (panel_df["treatment_flag"] == 1) & (panel_df["post"] == 1),
    "revenue_sim",
].mean()

discount_rate = 0.10
discount_cost = treated_post_mean * discount_rate

ate_lift = ate["coef"]
net_impact_per_user = ate_lift - discount_cost
num_treated_users = panel_df.loc[panel_df["treatment_flag"] == 1, "user_id"].nunique()
total_impact = net_impact_per_user * num_treated_users

# ============================================================
# Overview
# ============================================================

if section == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
    "Users (Post Period)",
    f"{panel_df[panel_df['post'] == 1]['user_id'].nunique():,}"
    )
    c2.metric("Post ATE", f"${ate['coef']:.2f}")
    c3.metric("95% CI", f"[${ate['ci_low']:.2f}, ${ate['ci_high']:.2f}]")
    c4.metric("Net Impact / User", f"${net_impact_per_user:.2f}")

    st.markdown("### Executive Takeaway")
    st.write(
        "The randomized A/B test indicates a positive post-period revenue lift. "
        "However, once discount cost is included, the average lift remains smaller than the cost "
        "of a blanket 10% offer, suggesting that profitability would likely require either a lower discount "
        "or more effective targeting."
    )

    st.markdown("### What this app covers")
    st.write(
        "This dashboard summarizes the experimental post-period ATE, event-study diagnostics, "
        "model comparison across multiple DiD specifications, business impact, and heterogeneity "
        "across baseline-spend segments."
    )

# ============================================================
# ATE
# ============================================================

elif section == "ATE":
    st.header("Post-Period Average Treatment Effect")
    c1, c2 = st.columns(2)
    c1.metric("ATE", f"${ate['coef']:.2f}")
    c2.metric("95% CI", f"[${ate['ci_low']:.2f}, ${ate['ci_high']:.2f}]")

    st.write(
        "The post-period ATE compares treated and control users after the intervention. "
        "Because treatment was randomized, this is the primary causal estimate in the analysis."
    )

# ============================================================
# Event Study
# ============================================================

elif section == "Event Study":
    st.header("Event Study")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(
        event_df["event_time"],
        event_df["coef"],
        yerr=1.96 * event_df["se"],
        fmt="o",
        capsize=4
    )
    ax.axhline(0, linestyle="--")
    ax.axvline(-1, linestyle="--")
    ax.set_xlabel("Week (Event Time)")
    ax.set_ylabel("Estimated Effect")
    ax.set_title("Event Study: Dynamic Treatment Effects (Two-Way Fixed Effects)")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    st.write("""
    The event study estimates week-by-week treatment effects relative to the final pre-treatment period (week -1), while controlling for both user and time fixed effects.

    The pre-treatment coefficients (weeks -6 to -2) are positive and in several cases statistically significant. This suggests that treated users exhibited higher revenue even before the treatment was applied, indicating a violation of the parallel trends assumption.

    As a result, panel-based DiD estimates should be interpreted as robustness checks rather than primary causal estimates. The randomized A/B test provides the most credible estimate of causal impact in this setting.
    """)

# ============================================================
# Model Comparison
# ============================================================

elif section == "Model Comparison":
    st.header("Model Comparison")
    st.dataframe(model_table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(model_table["Model"], model_table["Lift ($)"])
    ax.axhline(0)
    ax.set_title("Treatment Effect Estimates Across Models")
    ax.set_ylabel("Estimated Lift ($)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    st.write("""
    Because treatment was randomly assigned, the post-period ATE provides the most credible estimate of average causal impact.

    In contrast, panel-based methods (DiD, fixed effects) rely on the parallel trends assumption, which is violated in this setting, making those estimates sensitive to modeling choices.
    """)

# ============================================================
# Business Impact
# ============================================================

elif section == "Business Impact":
    st.header("Business Impact & ROI Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("ATE Lift / User", f"${ate_lift:.2f}")
    c2.metric("Discount Cost / User", f"${discount_cost:.2f}")
    c3.metric("Net Impact / User", f"${net_impact_per_user:.2f}")

    st.metric("Estimated Total Campaign Impact", f"${total_impact:,.2f}")

    st.write("""
    While the randomized ATE indicates a positive incremental revenue effect, the average lift remains smaller than the cost of a 10% discount.

    This implies that a blanket promotion is not profitable at scale. Any viable strategy would require either a lower discount rate or more precise targeting that materially improves lift relative to cost.
    
    This analysis assumes no cannibalization or long-term retention effects.         
    """)

# ============================================================
# HTE
# ============================================================

elif section == "HTE":
    st.header("Heterogeneous Treatment Effects")
    st.dataframe(hte_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(hte_df["quartile"], hte_df["lift"], marker="o")
    ax.set_title("Lift by Baseline Spend Quartile")
    ax.set_xlabel("Baseline Spend Quartile")
    ax.set_ylabel("Lift ($)")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    st.write("""
    Treatment lift is positive across all baseline spend quartiles, with higher-spend users generating larger absolute revenue gains.

    However, percentage lift is broadly similar across segments, suggesting that higher baseline revenue — rather than stronger causal responsiveness — drives the larger dollar impact.

    The causal forest model also identifies heterogeneity in predicted treatment effects, with higher estimated lift among certain users.

    However, because earlier diagnostics reveal violations of the parallel trends assumption, these patterns should be interpreted as descriptive rather than strictly causal.

    As a result, baseline revenue alone may not be sufficient to justify a simple targeted discounting strategy, and further validation would be required before deploying model-based targeting.
    """)