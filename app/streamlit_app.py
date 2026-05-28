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
        "control_mean": float(control.mean()),
        "treated_mean": float(treated.mean()),
        "n_treated_users": int(len(treated)),
        "n_control_users": int(len(control)),
    }


@st.cache_data
def compute_model_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": ["Synthetic Control", "Naive DiD", "Weighted DiD", "User FE DiD", "TWFE DiD"],
            "Lift ($/week)": [10.94, 8.74, 8.61, 8.41, 8.41],
            "95% CI Lower": [np.nan, 7.90, 7.76, 7.67, 7.67],
            "95% CI Upper": [np.nan, 9.59, 9.47, 9.16, 9.15],
            "Interpretation": [
                "Cohort-level robustness check; higher estimate likely reflects donor overfit with 6 pre-periods",
                "Baseline DiD; likely sensitive to pre-period differences",
                "Reweighted robustness check; positive and consistent with DiD range",
                "Controls for time-invariant user heterogeneity",
                "Two-way FE; consistent with User FE estimate",
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
def compute_synthetic_control_from_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the synthetic-control series using the same notebook logic.

    This avoids stale CSV mismatches and keeps Streamlit visuals aligned with the final notebook.
    """
    from scipy.optimize import minimize

    sc_df = df[["user_id", "treatment_flag", "event_time", "revenue_sim"]].copy()

    baseline = (
        sc_df[sc_df["event_time"] < 0]
        .groupby("user_id")["revenue_sim"]
        .mean()
        .reset_index(name="baseline_revenue")
    )
    sc_df = sc_df.merge(baseline, on="user_id", how="left")

    control_users = (
        sc_df[sc_df["treatment_flag"] == 0][["user_id", "baseline_revenue"]]
        .drop_duplicates()
        .copy()
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
        .groupby(["event_time", "donor_bin"])["revenue_sim"]
        .mean()
        .unstack()
        .sort_index()
    )

    common_times = treated_series.index.intersection(donor_panel.index)
    treated_series = treated_series.loc[common_times]
    donor_panel = donor_panel.loc[common_times]

    pre_periods = treated_series.index[treated_series.index < 0]
    y_treated_pre = treated_series.loc[pre_periods].values
    y_donors_pre = donor_panel.loc[pre_periods].values

    n_donors = y_donors_pre.shape[1]

    def objective(w):
        synthetic_pre = y_donors_pre @ w
        return np.mean((y_treated_pre - synthetic_pre) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_donors
    initial_weights = np.ones(n_donors) / n_donors

    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
    )

    weights = result.x
    synthetic_series = pd.Series(
        donor_panel.values @ weights,
        index=donor_panel.index,
        name="synthetic_control",
    )
    effect_series = treated_series - synthetic_series

    return pd.DataFrame(
        {
            "event_time": treated_series.index,
            "treated": treated_series.values,
            "synthetic_control": synthetic_series.values,
            "effect": effect_series.values,
        }
    ).sort_values("event_time").reset_index(drop=True)


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
    # Recompute synthetic-control visuals from panel data so the app matches the final notebook.
    synthetic_df = compute_synthetic_control_from_panel(panel_df)
except Exception as e:
    st.error(f"Failed to load app data: {e}")
    st.stop()

weekly_ate = compute_weekly_post_ate(panel_df)
cumulative_ate = compute_cumulative_post_ate(panel_df)
model_table = compute_model_table()
hte_df = compute_quartile_hte(panel_df)

# Business impact on cumulative basis, matching the notebook ROI section.
treated_cumulative_mean = (
    panel_df[(panel_df["treatment_flag"] == 1) & (panel_df["post"] == 1)]
    .groupby("user_id")["revenue_sim"]
    .sum()
    .mean()
)

discount_rate = 0.10
discount_cost = treated_cumulative_mean * discount_rate  # cumulative discount cost per user

cumulative_lift = cumulative_ate["coef"]  # cumulative ATE per user
net_impact_per_user = cumulative_lift - discount_cost
total_randomized_users = panel_df["user_id"].nunique()
num_treated_users = cumulative_ate["n_treated_users"]
gross_projected_lift = cumulative_lift * total_randomized_users
total_impact = net_impact_per_user * num_treated_users

# Synthetic-control summary from the recomputed notebook-matching series.
synthetic_effect = 10.94
synthetic_rmse = 2.27
if synthetic_df is not None:
    post_synth = synthetic_df[synthetic_df["event_time"] >= 0]
    pre_synth = synthetic_df[synthetic_df["event_time"] < 0]
    if not post_synth.empty:
        synthetic_effect = float(post_synth["effect"].mean())
    if not pre_synth.empty:
        synthetic_rmse = float(np.sqrt(np.mean(pre_synth["effect"] ** 2)))

# Match the notebook analysis sample: users included in the cumulative post-period ATE.
analysis_users = cumulative_ate["n_treated_users"] + cumulative_ate["n_control_users"]

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
    c1.metric("Users (Analysis Sample)", f"{analysis_users:,}")
    c2.metric("Avg Weekly ATE", f"${weekly_ate['coef']:.2f}")
    c3.metric("Cumulative Post ATE", f"${cumulative_ate['coef']:.2f}")
    c4.metric("Net Impact / User", f"${net_impact_per_user:.2f}")

    st.markdown("### Executive Takeaway")
    st.write(
        "The randomized A/B test indicates a positive revenue lift of approximately "
        f"\\${cumulative_ate['coef']:.2f} per user over observed post-weeks (+{cumulative_ate['pct_lift']:.1f}%). "
        "However, once the 10% discount cost is deducted, the net cumulative impact is approximately "
        f"\\${net_impact_per_user:.2f} per user, meaning a blanket rollout is not profitable under "
        "a simplified short-term revenue-cost objective. Profitability would likely require a lower "
        "discount rate, validated targeting, or a strategic context where short-term losses are acceptable."
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ate_yerr = np.array(
        [
            [cumulative_ate["coef"] - cumulative_ate["ci_low"]],
            [cumulative_ate["ci_high"] - cumulative_ate["coef"]],
        ]
    )
    axes[0].errorbar(
        ["ATE"],
        [cumulative_ate["coef"]],
        yerr=ate_yerr,
        fmt="o",
        capsize=5,
    )
    axes[0].axhline(0, linestyle="--", color="gray", alpha=0.7)
    axes[0].set_ylim(
        min(-2, cumulative_ate["ci_low"] - 5),
        cumulative_ate["ci_high"] + 5,
    )
    axes[0].set_title("Post-Period ATE with 95% Confidence Interval\n(Randomized Diff-in-Means)")
    axes[0].set_ylabel("Cumulative Post-Period Revenue Lift per User ($)")

    group_labels = ["Control", "Treatment"]
    group_values = [cumulative_ate["control_mean"], cumulative_ate["treated_mean"]]
    bars = axes[1].bar(group_labels, group_values, color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Post-Period Revenue by Group")
    axes[1].set_ylabel("Mean Post-Period Revenue per User ($)")
    axes[1].set_ylim(0, max(group_values) * 1.14)
    for bar, value in zip(bars, group_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"${value:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write(
        "The cumulative ATE measures the total post-period revenue difference per user, matching the "
        "primary notebook estimate. The average weekly ATE normalizes the same experiment over observed "
        "weekly panel rows and helps reconcile the panel estimates with the cumulative result."
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
        label="Estimated effect (vs. t=-1)",
    )
    ax.axhline(0, linestyle="--", color="gray", alpha=0.7, label="Zero effect")
    ax.axvline(-0.5, linestyle="--", color="#E74C3C", alpha=0.85, label="Pre/Post Boundary")
    ax.set_xlabel("Week (Event Time)")
    ax.set_ylabel("Estimated Effect on Revenue ($)")
    ax.set_title("Event Study: Weekly Treatment Effects Relative to t=-1 (TWFE)")
    ax.legend()
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
    st.caption("Synthetic Control has no confidence interval; interpret it as a robustness check.")
    st.dataframe(model_table, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(model_table["Model"], model_table["Lift ($/week)"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title("Treatment Effect Estimates Across Specifications\n(Synthetic Control has no CI)")
    ax.set_xlabel("Estimated Weekly Lift ($/week)")
    ax.set_ylabel("Model")
    ax.set_xlim(0, model_table["Lift ($/week)"].max() + 1.1)
    for i, lift in enumerate(model_table["Lift ($/week)"]):
        ax.text(lift + 0.1, i, f"${lift:.2f}/wk", va="center")
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write("""
    The panel-based robustness checks are directionally consistent: Naive DiD, User FE, TWFE, and
    Weighted DiD all estimate roughly $8.41 to $8.74 of lift per user-week. The synthetic-control
    estimate is higher, which points to sensitivity in how the counterfactual is constructed.

    The weighted DiD model improves comparability on observed characteristics, while the synthetic
    control check constructs a better-matched control trajectory. However, neither replaces the
    randomized post-period ATE, which remains the most reliable estimate of average causal impact.
    """)

# ============================================================
# Synthetic Control
# ============================================================

elif section == "Synthetic Control":
    st.header("Synthetic Control Robustness Check")

    c1, c2 = st.columns(2)
    c1.metric("Synthetic-Control Effect", f"${synthetic_effect:.2f}")
    c2.metric("Pre-Period Fit RMSE", f"{synthetic_rmse:.2f}")

    st.write(
        "The first chart checks how closely the synthetic control tracks the treated group before "
        "the treatment starts. The second chart shows the estimated treatment effect over time."
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        synthetic_df["event_time"],
        synthetic_df["treated"],
        marker="o",
        color="#DD8452",
        label="Treated",
    )
    axes[0].plot(
        synthetic_df["event_time"],
        synthetic_df["synthetic_control"],
        marker="o",
        linestyle="--",
        color="#4C72B0",
        label="Synthetic Control",
    )
    axes[0].fill_between(
        synthetic_df["event_time"],
        synthetic_df["treated"],
        synthetic_df["synthetic_control"],
        where=synthetic_df["event_time"] >= 0,
        alpha=0.15,
        color="#DD8452",
        label="Post-period gap",
    )
    axes[0].axvline(-0.5, linestyle="--", color="#E74C3C", alpha=0.75, label="Pre/Post Boundary")
    axes[0].set_title("Revenue Trajectories: Treated vs. Synthetic Control")
    axes[0].set_xlabel("Event Time (weeks)")
    axes[0].set_ylabel("Average Weekly Revenue ($)")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(
        synthetic_df["event_time"],
        synthetic_df["effect"],
        marker="o",
        color="steelblue",
        label="Gap (Treated - Synthetic)",
    )
    axes[1].fill_between(
        synthetic_df["event_time"],
        synthetic_df["effect"],
        0,
        where=synthetic_df["event_time"] >= 0,
        alpha=0.15,
        color="steelblue",
        label=f"Post-period mean: ${synthetic_effect:.2f}",
    )
    axes[1].axhline(0, linestyle="--", color="gray", alpha=0.7, label="Zero effect")
    axes[1].axvline(-0.5, linestyle="--", color="#E74C3C", alpha=0.75, label="Pre/Post Boundary")
    axes[1].set_title("Synthetic Control Gap: Treated Minus Counterfactual")
    axes[1].set_xlabel("Event Time (weeks)")
    axes[1].set_ylabel("Revenue Difference ($)")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write(
        "The synthetic control robustness check constructs a weighted combination of control cohorts "
        "that more closely matches the treated group's pre-treatment revenue trajectory."
    )

    st.write(
        f"In the final notebook, the synthetic-control estimate is approximately "
        f"\\${synthetic_effect:.2f} per user-week, with a pre-period fit RMSE of {synthetic_rmse:.2f}. "
        "Because this estimate is higher than the DiD range and uses only six pre-period weeks, it is "
        "best read as a sensitivity check that may reflect donor overfit."
    )

    st.write(
        "This result is best interpreted as a robustness check rather than the primary causal estimate."
    )

# ============================================================
# Business Impact
# ============================================================

elif section == "Business Impact":
    st.header("Business Impact & ROI Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cumulative ATE Lift / User", f"${cumulative_lift:.2f}")
    c2.metric("Estimated Discount Cost / User", f"${discount_cost:.2f}")
    c3.metric("Net Impact / User", f"${net_impact_per_user:.2f}")
    c4.metric("Projected Gross Lift", f"${gross_projected_lift / 1_000_000:.2f}M")

    st.metric("Estimated Net Campaign Impact (Observed Treated Users)", f"${total_impact:,.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Revenue Lift", "Discount Cost", "Net Impact"]
    values = [cumulative_lift, -discount_cost, net_impact_per_user]
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Simplified Business Impact per Treated User (Post-Period)")
    ax.set_ylabel("Dollars per User")
    ax.set_ylim(min(values) - 6, max(values) + 6)
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.75 if value >= 0 else -0.75
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"${value:.2f}",
            ha="center",
            va=va,
            fontweight="bold",
        )
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write("""
    While the randomized ATE indicates a positive incremental revenue effect, the cumulative lift of
    \\$28.50 per user does not cover the estimated \\$37.79 cumulative cost of a 10% discount.

    This implies that a blanket promotion is not profitable under a simplified short-term revenue-cost
    objective.
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
    st.caption("Quartile lift is observed post-period lift by baseline spend; causal forest metrics are exploratory notebook outputs.")
    st.dataframe(hte_df, width="stretch")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hte_df["quartile"], hte_df["lift"], marker="o")
    ax.axhline(0, linestyle="--", color="gray", alpha=0.7)
    ax.set_title("Observed Lift by Baseline Spend Quartile")
    ax.set_xlabel("Baseline Spend Quartile")
    ax.set_ylabel("Lift ($)")
    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.markdown("### Causal Forest Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Predicted Lift", "$8.40")
    c2.metric("Std Dev", "$5.57")
    c3.metric("Min Predicted Lift", "$-13.73")
    c4.metric("Max Predicted Lift", "$46.33")

    c5, c6 = st.columns(2)
    c5.metric("Top 20% Predicted Lift", "$16.55")
    c6.metric("Bottom 80% Predicted Lift", "$6.36")

    st.write("""
    Treatment lift is positive across all baseline spend quartiles, with higher-spend users generating
    larger absolute revenue gains.

    However, percentage lift is broadly similar across segments, suggesting that higher baseline
    revenue, rather than stronger causal responsiveness, drives much of the larger dollar impact.

    The causal forest suggests exploratory variation in predicted lift across users, with a mean
    predicted lift of \\$8.40, standard deviation of \\$5.57, minimum of -\\$13.73,
    and maximum of \\$46.33.

    The top 20% of users by predicted lift average \\$16.55, versus \\$6.36 for the remaining
    80%. Because the simulation applies a uniform multiplier, this is best interpreted as
    dollar-scale heterogeneity tied to baseline spend rather than a validated targeting rule.

    Because these estimates rely on model assumptions and observed covariates, they should be
    interpreted as exploratory evidence of potential heterogeneity rather than a validated targeting
    rule. A follow-up experiment would be needed before deploying model-based targeting.
    """)
