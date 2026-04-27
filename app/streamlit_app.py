from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="A/B Promo Incrementality", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
PANEL_CSV = DATA_DIR / "panel_df.csv"
EVENT_CSV = DATA_DIR / "event_study.csv"
SYNTHETIC_CSV = DATA_DIR / "synthetic_control.csv"

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
def compute_weekly_post_ate(df: pd.DataFrame) -> dict:
    """Average treatment effect using post-period user-week observations."""
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
        "n_treated_obs": int(len(treated)),
        "n_control_obs": int(len(control)),
    }


@st.cache_data
def compute_cumulative_post_ate(df: pd.DataFrame) -> dict:
    """Average treatment effect using cumulative post-period revenue per user."""
    post_user = (
        df[df["post"] == 1]
        .groupby(["user_id", "treatment_flag"], as_index=False)
        .agg(post_revenue=("revenue_sim", "sum"))
    )

    treated = post_user.loc[post_user["treatment_flag"] == 1, "post_revenue"]
    control = post_user.loc[post_user["treatment_flag"] == 0, "post_revenue"]

    coef = treated.mean() - control.mean()
    pct_lift = coef / control.mean() * 100
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    ci_low, ci_high = ci_bounds(coef, se)

    return {
        "coef": float(coef),
        "pct_lift": float(pct_lift),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_treated_users": int(len(treated)),
        "n_control_users": int(len(control)),
    }


@st.cache_data
def compute_model_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["Naive DiD", "User FE DiD", "TWFE DiD", "Weighted DiD", "Synthetic Control"],
            "Lift ($)": [8.74, 4.53, -1.58, 8.61, 2.11],
            "95% CI Lower": [np.nan, np.nan, np.nan, 7.76, np.nan],
            "95% CI Upper": [np.nan, np.nan, np.nan, 9.47, np.nan],
            "Interpretation": [
                "Baseline DiD; likely sensitive to pre-period differences",
                "Controls for time-invariant user heterogeneity",
                "Two-way FE robustness check; estimate turns negative",
                "Reweighted robustness check; positive but model-dependent",
                "Synthetic-control robustness check; smaller positive estimate",
            ],
        }
    )


@st.cache_data
def load_event_study(path: str) -> pd.DataFrame:
    event_df = pd.read_csv(path)
    required_cols = {"event_time", "coef", "se"}
    missing = required_cols - set(event_df.columns)
    if missing:
        raise ValueError(f"Missing required event-study columns: {sorted(missing)}")
    return event_df.sort_values("event_time").reset_index(drop=True)


@st.cache_data
def load_synthetic_control(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None

    synth_df = pd.read_csv(path)
    required_cols = {"event_time", "treated", "synthetic_control", "effect"}
    missing = required_cols - set(synth_df.columns)
    if missing:
        raise ValueError(f"Missing required synthetic-control columns: {sorted(missing)}")
    return synth_df.sort_values("event_time").reset_index(drop=True)


@st.cache_data
def compute_synthetic_control(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight synthetic-control robustness check using baseline-spend donor cohorts.

    This recreates the notebook robustness check when the precomputed synthetic_control.csv
    is not present in the deployed app. It is intentionally aggregate/cohort-level, not a
    replacement for the randomized A/B estimate.
    """
    from scipy.optimize import minimize

    sc_df = df[["user_id", "treatment_flag", "event_time", "revenue_sim", "post"]].copy()

    baseline = (
        sc_df[sc_df["post"] == 0]
        .groupby("user_id", as_index=False)["revenue_sim"]
        .mean()
        .rename(columns={"revenue_sim": "baseline_revenue"})
    )
    sc_df = sc_df.merge(baseline, on="user_id", how="left")

    control_users = (
        sc_df[sc_df["treatment_flag"] == 0][["user_id", "baseline_revenue"]]
        .drop_duplicates()
        .dropna(subset=["baseline_revenue"])
    )
    control_users["donor_bin"] = pd.qcut(
        control_users["baseline_revenue"],
        q=10,
        labels=False,
        duplicates="drop",
    )

    sc_df = sc_df.merge(control_users[["user_id", "donor_bin"]], on="user_id", how="left")

    treated_series = (
        sc_df[sc_df["treatment_flag"] == 1]
        .groupby("event_time")["revenue_sim"]
        .mean()
        .sort_index()
    )

    donor_panel = (
        sc_df[sc_df["treatment_flag"] == 0]
        .dropna(subset=["donor_bin"])
        .groupby(["event_time", "donor_bin"])["revenue_sim"]
        .mean()
        .unstack()
        .sort_index()
    )

    common_times = treated_series.index.intersection(donor_panel.index)
    treated_series = treated_series.loc[common_times]
    donor_panel = donor_panel.loc[common_times]

    pre_periods = treated_series.index[treated_series.index < 0]
    if len(pre_periods) == 0 or donor_panel.empty:
        raise ValueError("Synthetic control requires pre-treatment periods and control donor cohorts.")

    # Keep donor cohorts with complete pre-period observations.
    donor_panel = donor_panel.dropna(axis=1, subset=pre_periods)
    if donor_panel.empty:
        raise ValueError("No donor cohorts have complete pre-period observations.")

    y_treated_pre = treated_series.loc[pre_periods].values
    y_donors_pre = donor_panel.loc[pre_periods].values
    n_donors = y_donors_pre.shape[1]

    def objective(w: np.ndarray) -> float:
        synthetic_pre = y_donors_pre @ w
        return float(np.mean((y_treated_pre - synthetic_pre) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    initial_weights = np.ones(n_donors) / n_donors

    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x if result.success else initial_weights

    synthetic_series = pd.Series(
        donor_panel.values @ weights,
        index=donor_panel.index,
        name="synthetic_control",
    )
    effect_series = treated_series.loc[synthetic_series.index] - synthetic_series

    return pd.DataFrame(
        {
            "event_time": synthetic_series.index,
            "treated": treated_series.loc[synthetic_series.index].values,
            "synthetic_control": synthetic_series.values,
            "effect": effect_series.values,
        }
    ).reset_index(drop=True)


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
        user_df.groupby(["quartile", "treatment_flag"], as_index=False, observed=False)["post_revenue"]
        .mean()
        .pivot(index="quartile", columns="treatment_flag", values="post_revenue")
        .reset_index()
        .rename(columns={0: "control", 1: "treated"})
    )
    out["lift"] = out["treated"] - out["control"]
    out["pct_lift"] = out["lift"] / out["control"] * 100
    return out


# ============================================================
# Load data
# ============================================================

st.title("10% Discount Incrementality Analysis")
st.caption("Streamlit dashboard for the high-LTV discount causal inference project")

if not PANEL_CSV.exists():
    st.error(f"Could not find `{PANEL_CSV}`. Export your analysis dataset to CSV first, then rerun the app.")
    st.stop()

if not EVENT_CSV.exists():
    st.error(f"Could not find `{EVENT_CSV}`. Export the precomputed event-study results first, then rerun the app.")
    st.stop()

try:
    panel_df = load_panel_data(str(PANEL_CSV))
    event_df = load_event_study(str(EVENT_CSV))
    synthetic_df = load_synthetic_control(str(SYNTHETIC_CSV))
    if synthetic_df is None:
        synthetic_df = compute_synthetic_control(panel_df)
except Exception as e:
    st.error(f"Failed to load app data: {e}")
    st.stop()

weekly_ate = compute_weekly_post_ate(panel_df)
cumulative_ate = compute_cumulative_post_ate(panel_df)
analysis_users = cumulative_ate["n_treated_users"] + cumulative_ate["n_control_users"]
model_table = compute_model_table()
hte_df = compute_quartile_hte(panel_df)

# Business impact based on the average post-period user-week ATE, matching the notebook ROI section.
treated_post_mean = panel_df.loc[
    (panel_df["treatment_flag"] == 1) & (panel_df["post"] == 1),
    "revenue_sim",
].mean()

discount_rate = 0.10
discount_cost = treated_post_mean * discount_rate

ate_lift = weekly_ate["coef"]
net_impact_per_user = ate_lift - discount_cost
num_treated_users = panel_df.loc[panel_df["treatment_flag"] == 1, "user_id"].nunique()
total_impact = net_impact_per_user * num_treated_users

# Synthetic-control summary from the final notebook
synthetic_effect = 2.11
synthetic_rmse = 2.38
if synthetic_df is not None:
    post_synth = synthetic_df[synthetic_df["event_time"] >= 0]
    pre_synth = synthetic_df[synthetic_df["event_time"] < 0]
    if not post_synth.empty:
        synthetic_effect = float(post_synth["effect"].mean())
    if not pre_synth.empty:
        synthetic_rmse = float(np.sqrt(np.mean(pre_synth["effect"] ** 2)))

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("A/B Promo Incrementality")

st.sidebar.markdown("### App sections")
section = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "ATE",
        "Event Study",
        "Model Comparison",
        "Synthetic Control",
        "Business Impact",
        "HTE",
    ],
)

# ============================================================
# Overview
# ============================================================

if section == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{analysis_users:,}")
    c2.metric("Avg Weekly ATE", f"${weekly_ate['coef']:.2f}")
    c3.metric("Cumulative Post ATE", f"${cumulative_ate['coef']:.2f}")
    c4.metric("Net Impact / User", f"${net_impact_per_user:.2f}")

    st.markdown("### Executive Takeaway")
    st.markdown(
        "The randomized A/B test indicates a positive revenue lift. The cumulative post-period "
        f"ATE is approximately \${cumulative_ate['coef']:.2f} per user, while the average weekly "
        f"post-period lift is approximately \${weekly_ate['coef']:.2f}. However, once discount cost "
        "is included, the average lift remains smaller than the cost of a blanket 10% offer. "
        "Profitability would likely require either a lower discount, more effective targeting, or "
        "a strategic context where short-term losses are acceptable."
    )

    st.markdown("### What this app covers")
    st.write(
        "This dashboard summarizes the experimental ATE, event-study diagnostics, model comparison "
        "across multiple DiD specifications, synthetic-control robustness, business impact, and "
        "heterogeneity across baseline-spend segments."
    )

# ============================================================
# ATE
# ============================================================

elif section == "ATE":
    st.header("Post-Period Average Treatment Effect")

    c1, c2 = st.columns(2)
    c1.metric("Cumulative Post-Period ATE", f"${cumulative_ate['coef']:.2f}")
    c2.metric("95% CI", f"[${cumulative_ate['ci_low']:.2f}, ${cumulative_ate['ci_high']:.2f}]")

    c3, c4 = st.columns(2)
    c3.metric("Average Weekly Post-Period ATE", f"${weekly_ate['coef']:.2f}")
    c4.metric("95% CI", f"[${weekly_ate['ci_low']:.2f}, ${weekly_ate['ci_high']:.2f}]")

    st.write(
        "The cumulative ATE measures the total post-period revenue difference per user, matching the "
        "primary notebook estimate. The average weekly ATE normalizes the same experiment over weekly "
        "panel observations and is used in the ROI section to compare average weekly lift against the "
        "estimated 10% discount cost."
    )

    st.write(
        "Because treatment was randomized, the post-period ATE remains the primary causal estimate. "
        "Panel-based methods such as DiD, TWFE, and synthetic control are used as robustness and "
        "diagnostic checks."
    )

# ============================================================
# Event Study
# ============================================================

elif section == "Event Study":
    st.header("Event Study")

    fig, ax = plt.subplots(figsize=(8, 5))
    if {"ci_low", "ci_high"}.issubset(event_df.columns):
        yerr = [
            event_df["coef"] - event_df["ci_low"],
            event_df["ci_high"] - event_df["coef"],
        ]
    else:
        yerr = 1.96 * event_df["se"]

    ax.errorbar(
        event_df["event_time"],
        event_df["coef"],
        yerr=yerr,
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, linestyle="--")
    ax.axvline(-1, linestyle="--")
    ax.set_xlabel("Week (Event Time)")
    ax.set_ylabel("Estimated Effect")
    ax.set_title("Event Study: Dynamic Treatment Effects (Two-Way Fixed Effects)")
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write("""
    The event study estimates week-by-week treatment effects relative to the final pre-treatment
    period (week -1), while controlling for both user and time fixed effects.

    The pre-treatment coefficients show modest differences relative to the omitted baseline period.
    Because event-study coefficients are normalized to a chosen reference period, this pattern may be
    sensitive to which pre-treatment week is used as the baseline.

    As a result, the event study suggests caution when interpreting panel-based DiD estimates, but it
    should not be read as definitive evidence of a meaningful parallel trends violation. The randomized A/B
    test remains the primary causal estimate.
    """)

# ============================================================
# Model Comparison
# ============================================================

elif section == "Model Comparison":
    st.header("Model Comparison")
    st.dataframe(model_table, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_table["Model"], model_table["Lift ($)"])
    ax.axhline(0)
    ax.set_title("Treatment Effect Estimates Across Models")
    ax.set_ylabel("Estimated Lift ($)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write("""
    Treatment effect estimates vary substantially across specifications, ranging from positive to
    negative. This divergence suggests that estimates are sensitive to modeling choices and to
    pre-treatment differences between treated and control users.

    The weighted DiD model improves comparability on observed characteristics, while the synthetic
    control check constructs a better-matched control trajectory. However, neither replaces the
    randomized post-period ATE, which remains the most reliable estimate of average causal impact.

    This instability reinforces the importance of randomized experimental estimates, which do not
    rely on parallel trends assumptions.
    """)

# ============================================================
# Synthetic Control
# ============================================================

elif section == "Synthetic Control":
    st.header("Synthetic Control Robustness Check")

    c1, c2 = st.columns(2)
    c1.metric("Synthetic-Control Effect", f"${synthetic_effect:.2f}")
    c2.metric("Pre-Period Fit RMSE", f"{synthetic_rmse:.2f}")

    if synthetic_df is not None:
        st.write(
            "The first chart shows how well the synthetic control matches the treated group before "
            "treatment, while the second chart shows the estimated treatment effect over time."
        )

        # Chart 1: pre-fit check (levels)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            synthetic_df["event_time"],
            synthetic_df["treated"],
            marker="o",
            label="Treated",
        )
        ax.plot(
            synthetic_df["event_time"],
            synthetic_df["synthetic_control"],
            marker="o",
            label="Synthetic Control",
        )
        ax.axvline(
            -1,
            linestyle="--",
            alpha=0.6,
            label="Treatment Start (t = -1)",
        )
        ax.set_title("Synthetic Control Robustness Check")
        ax.set_xlabel("Event Time")
        ax.set_ylabel("Average Revenue")
        ax.grid(alpha=0.2)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, width="content")

        # Chart 2: estimated effect over time
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            synthetic_df["event_time"],
            synthetic_df["effect"],
            marker="o",
        )
        ax.axhline(
            0,
            linestyle="--",
            alpha=0.35,
            label="Zero Effect (Baseline)",
        )
        ax.axvline(
            -1,
            linestyle="--",
            alpha=0.6,
            label="Treatment Start (t = -1)",
        )
        ax.set_title("Treated minus Synthetic Control Over Time")
        ax.set_xlabel("Event Time")
        ax.set_ylabel("Revenue Difference")
        ax.grid(alpha=0.2)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, width="content")
    else:
        st.write(
            "Synthetic-control plot data is not included in this deployed view, but the final "
            "notebook reports the synthetic-control effect and pre-period fit diagnostics below."
        )

    st.markdown("""
    The synthetic control robustness check constructs a weighted combination of control cohorts that
    more closely matches the treated group's pre-treatment revenue trajectory.

    In the final notebook, the synthetic-control estimate is approximately +\$2.11 per user, with a
    pre-period fit RMSE of 2.38. This smaller estimate highlights sensitivity to how the
    counterfactual is constructed.

    A strong pre-period fit supports the synthetic control as a useful counterfactual, but this result
    is best interpreted as a robustness check rather than the primary causal estimate.
    """)

# ============================================================
# Business Impact
# ============================================================

elif section == "Business Impact":
    st.header("Business Impact & ROI Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Weekly ATE Lift", f"${ate_lift:.2f}")
    c2.metric("Discount Cost", f"${discount_cost:.2f}")
    c3.metric("Net Impact / User", f"${net_impact_per_user:.2f}")

    st.metric("Estimated Total Campaign Impact", f"${total_impact:,.2f}")

    st.write("""
    While the randomized ATE indicates a positive incremental revenue effect, the average weekly lift
    remains smaller than the estimated cost of a 10% discount.

    This implies that a blanket promotion is not profitable under a short-term profitability objective.
    However, the business decision depends on product context. If the promotion is intended to drive
    trial for a new product, acquire customers, encourage repeat purchase, or clear slow-moving
    inventory, a short-term loss may still be strategically acceptable.

    This analysis assumes no cannibalization or long-term retention effects. Real deployment would
    require validation against observed transaction revenue, margin impact, inventory costs, and
    downstream retention outcomes.
    """)

# ============================================================
# HTE
# ============================================================

elif section == "HTE":
    st.header("Heterogeneous Treatment Effects")
    st.dataframe(hte_df, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hte_df["quartile"], hte_df["lift"], marker="o")
    ax.set_title("Lift by Baseline Spend Quartile")
    ax.set_xlabel("Baseline Spend Quartile")
    ax.set_ylabel("Lift ($)")
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.markdown("### Causal Forest Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean TE", "$8.40")
    c2.metric("Std Dev", "$5.57")
    c3.metric("Min TE", "-$13.73")
    c4.metric("Max TE", "$46.33")

    st.markdown("""
    Treatment lift is positive across all baseline spend quartiles, with higher-spend users generating
    larger absolute revenue gains.

    However, percentage lift is broadly similar across segments, suggesting that higher baseline
    revenue — rather than stronger causal responsiveness — drives much of the larger dollar impact.

    The causal forest estimates reveal meaningful variation in predicted treatment effects across
    users, with a mean estimated effect of \$8.40, standard deviation of \$5.57, minimum
    of -\$13.73, and maximum of \$46.33. Notably, some users exhibit negative predicted
    treatment effects, suggesting the discount may reduce net revenue for a subset of users.

    Because these estimates rely on model assumptions and observed covariates, they should be
    interpreted as exploratory evidence of potential heterogeneity rather than a validated targeting
    rule. A follow-up experiment would be needed before deploying model-based targeting.
    """)
