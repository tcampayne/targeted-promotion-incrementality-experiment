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
def compute_causal_forest_display() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Notebook causal-forest summary values for the deployed HTE view.

    EconML is used in the notebook, but Streamlit Cloud currently runs this app on
    Python 3.14, where EconML does not publish compatible wheels. The app therefore
    displays the final notebook summary and an illustrative distribution calibrated
    to those outputs.
    """
    rng = np.random.default_rng(42)
    predicted_lift = np.clip(rng.normal(loc=8.40, scale=5.57, size=55_102), -13.73, 46.33)

    cf_df = pd.DataFrame({"te_pred": predicted_lift})
    cf_summary = pd.DataFrame(
        {
            "baseline_quartile": ["Q1 Low", "Q2", "Q3", "Q4 High"],
            "te_pred": [5.31, 8.70, 9.28, 10.31],
        }
    )
    metrics = {
        "mean": 8.40,
        "std": 5.57,
        "min": -13.73,
        "max": 46.33,
        "pct_negative": 4.4,
        "top_20": 16.55,
        "bottom_80": 6.36,
    }
    return cf_df, cf_summary, metrics


# ============================================================
# Load data
# ============================================================

st.title("10% Discount Incrementality Analysis")
st.caption("Simulated randomized experiment for a high-LTV user discount strategy")

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
    st.caption(
        f"The ATE sample includes {analysis_users:,} users with post-period experiment data; "
        f"gross impact projections scale to {total_randomized_users:,} randomized users."
    )

    st.markdown("### Executive Takeaway")
    st.write(
        "The randomized test shows a positive revenue lift of approximately "
        f"\\${cumulative_ate['coef']:.2f} per user over observed post-weeks (+{cumulative_ate['pct_lift']:.1f}%). "
        "After subtracting the estimated 10% discount cost, the simplified net impact is approximately "
        f"\\${net_impact_per_user:.2f} per user, meaning a blanket rollout is not profitable under "
        "a short-term revenue-cost objective. The promotion would likely need tighter targeting, a lower "
        "discount rate, or a strategic reason to accept short-term margin loss."
    )

    st.markdown("### What this app covers")
    st.markdown(
        """
        - Primary randomized ATE and confidence interval
        - Event-study and DiD robustness checks
        - Synthetic-control sensitivity check
        - Revenue, discount cost, and net impact math
        - Exploratory HTE from the causal forest notebook output
        """
    )

# ============================================================
# ATE
# ============================================================

elif section == "ATE":
    st.header("Post-Period Average Treatment Effect")

    c1, c2 = st.columns(2)
    c1.metric("Cumulative Post-Period ATE", f"${cumulative_ate['coef']:.2f}")
    c2.metric("95% CI ($)", f"{cumulative_ate['ci_low']:.2f} to {cumulative_ate['ci_high']:.2f}")

    c3, c4 = st.columns(2)
    c3.metric("Average Weekly Post-Period ATE", f"${weekly_ate['coef']:.2f}")
    c4.metric("95% CI ($)", f"{weekly_ate['ci_low']:.2f} to {weekly_ate['ci_high']:.2f}")

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
        "The cumulative ATE is the main business-facing estimate: treated users generated about "
        f"\\${cumulative_ate['coef']:.2f} more post-period revenue than control users on average."
    )

    st.write(
        "Because assignment was randomized, this post-period difference is the primary causal estimate. "
        "The weekly ATE is included to connect the randomized result back to the panel-model diagnostics."
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
    The event study checks whether the treatment and control groups were already moving differently
    before the discount period. Pre-period estimates are close to zero, while post-period estimates
    move clearly positive after launch.

    This supports the overall direction of the result, but the randomized post-period ATE remains the
    cleaner causal estimate. The event study is best used as a diagnostic, not the headline number.
    """)

# ============================================================
# Model Comparison
# ============================================================

elif section == "Model Comparison":
    st.header("Model Comparison")
    st.caption("Synthetic Control has no confidence interval; interpret it as a robustness check.")

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
    The DiD-style models are directionally consistent, landing around $8.41 to $8.74 of lift per
    user-week. Synthetic control is higher, which makes it useful as a sensitivity check but not the
    main result.

    Overall, the robustness checks support a positive treatment effect while reinforcing that the
    randomized ATE is still the cleanest estimate of average impact.
    """)

    with st.expander("Model details"):
        st.dataframe(model_table, width="stretch")

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
        "Synthetic control builds a weighted control trajectory that tries to match the treated group "
        "before the discount starts."
    )

    st.write(
        f"The estimated post-period gap is approximately \\${synthetic_effect:.2f} per user-week, "
        f"with a pre-period fit RMSE of {synthetic_rmse:.2f}. Because this estimate is higher than the "
        "DiD range and uses only six pre-period weeks, I treat it as a sensitivity check rather than "
        "the primary causal estimate."
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
    st.caption(
        f"Gross lift scales the ATE to {total_randomized_users:,} randomized users. "
        "Net impact subtracts the estimated discount cost for observed treated users."
    )

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
    This is the key business wrinkle: the discount creates incremental revenue, but the promotion still
    has to pay for itself.

    In this simplified view, the cumulative lift of \\$28.50 per user does not cover the estimated
    \\$37.79 cumulative cost of a 10% discount. That makes the blanket promotion negative on short-term
    net impact, even though the experiment shows a real revenue lift.

    That does not make the test a failure. It means the better next step is not "ship it to everyone,"
    but refine the strategy: lower the discount, target users more selectively, or justify the loss with
    a longer-term goal such as acquisition, repeat purchase, product trial, or inventory movement.

    This analysis assumes no cannibalization or long-term retention effects. Real deployment would
    require validation against observed transaction revenue, margin impact, inventory costs, and
    downstream retention outcomes.
    """)

# ============================================================
# HTE
# ============================================================

elif section == "HTE":
    st.header("Heterogeneous Treatment Effects")
    st.caption(
        "The notebook fit the EconML causal forest; this dashboard uses notebook summary stats and "
        "a simulated distribution calibrated to those outputs."
    )
    cf_df, cf_hte_summary, cf_metrics = compute_causal_forest_display()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Predicted Lift", f"${cf_metrics['mean']:.2f}")
    c2.metric("Top 20% Predicted Lift", f"${cf_metrics['top_20']:.2f}")
    c3.metric("Bottom 80% Predicted Lift", f"${cf_metrics['bottom_80']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        cf_df["te_pred"],
        bins=30,
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
    )
    axes[0].axvline(0, linestyle="--", color="gray", label="Zero lift")
    axes[0].axvline(
        cf_metrics["mean"],
        linestyle="--",
        color="#E74C3C",
        label=f"Mean predicted lift: ${cf_metrics['mean']:.2f}",
    )
    axes[0].set_title(
        "Calibrated Distribution of User-Level Lift\n"
        f"({cf_metrics['pct_negative']:.1f}% of simulated user-level effects are negative)"
    )
    axes[0].set_xlabel("Simulated Treatment Effect ($ per user-week)")
    axes[0].set_ylabel("Number of Users")
    axes[0].legend()

    axes[1].plot(
        cf_hte_summary["baseline_quartile"],
        cf_hte_summary["te_pred"],
        marker="o",
        color="#4C72B0",
    )
    axes[1].axhline(0, linestyle="--", color="gray", alpha=0.7)
    axes[1].set_ylim(0, cf_hte_summary["te_pred"].max() + 1.5)
    axes[1].set_title("Causal Forest: Mean Predicted Lift by Baseline Spend Quartile")
    axes[1].set_xlabel("Baseline Spend Quartile")
    axes[1].set_ylabel("Mean Predicted Lift ($ per user-week)")

    plt.tight_layout()
    st.pyplot(fig, width="content")

    st.write("""
    Most predicted user-level effects are positive, but the size of the lift varies. The top predicted-response
    users average \\$16.55 per user-week, compared with \\$6.36 for the remaining users.

    Because the simulation applies a uniform treatment multiplier, I would treat this as exploratory
    dollar-scale heterogeneity tied partly to baseline spend, not a deployment-ready targeting rule.
    """)

    with st.expander("Notebook details"):
        d1, d2, d3 = st.columns(3)
        d1.metric("Std Dev", f"${cf_metrics['std']:.2f}")
        d2.metric("Min Predicted Lift", f"${cf_metrics['min']:.2f}")
        d3.metric("Max Predicted Lift", f"${cf_metrics['max']:.2f}")
        st.dataframe(cf_hte_summary, width="stretch")
