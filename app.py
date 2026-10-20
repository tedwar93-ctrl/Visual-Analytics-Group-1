
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import (
    filter_data,
    get_weekly_trends,
    identify_promotion_episodes,
    compare_before_during_after,
    get_household_behavior_summary,
    get_similar_products_summary,
)

APP_DIR = Path(__file__).resolve().parent
PROD_DATA_PATH = APP_DIR / "database" / "enriched_transactions.parquet"
SAMPLE_DATA_PATH = APP_DIR / "database" / "sample_enriched_transactions.parquet"

st.set_page_config(
    page_title="Grocery Promotion & Stockpiling Explorer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 Grocery Store Promotion & Stockpiling Explorer")
st.markdown(
    "**Client Objective:** Evaluate grocery retail promotional performance, customer stockpiling tendencies, "
    "and post-promotion household repurchase patterns across the 102-week Dunnhumby *Complete Journey* dataset."
)


@st.cache_data
def load_data(path_string):
    return pd.read_parquet(path_string)


# -------------------- Dataset Discovery & Selection --------------------

available_datasets = {}
if PROD_DATA_PATH.exists():
    available_datasets["Full Production Dataset (2,595,732 rows, Weeks 1–102)"] = PROD_DATA_PATH
if SAMPLE_DATA_PATH.exists():
    available_datasets["Prototype Sample (100,000 rows, Weeks 1–12)"] = SAMPLE_DATA_PATH

if not available_datasets:
    st.error("No prepared Parquet dataset found in `database/`.")
    st.info(
        "Run `python3 preprocess.py` from the repository root to generate the production dataset, "
        "or `python3 preprocess.py --sample-size 100000` for a sample."
    )
    st.stop()

# Sidebar dataset selector
st.sidebar.header("📁 Data Source")
if len(available_datasets) > 1:
    selected_dataset_label = st.sidebar.selectbox(
        "Select Active Dataset",
        list(available_datasets.keys()),
        index=0,
        help="Switch between the complete 102-week transaction history and the fast 12-week prototype sample.",
    )
    data_path = available_datasets[selected_dataset_label]
else:
    selected_dataset_label, data_path = next(iter(available_datasets.items()))

try:
    raw_df = load_data(str(data_path))
except Exception as exc:
    st.error(f"Could not load the selected dataset: {exc}")
    st.stop()

if raw_df.empty:
    st.warning("The prepared dataset contains no transaction rows.")
    st.stop()

# Status Banner
is_sample = "Sample" in selected_dataset_label
if is_sample:
    st.warning(
        "⚠️ **PROTOTYPE SAMPLE ACTIVE (Weeks 1–12):** This 100,000-row sample only covers fiscal Weeks 1 to 12. "
        "Because causal promotion auditing begins at Week 9, only 4 weeks of causal coverage exist (Weeks 9–12). "
        "Episodes starting at Week 9 have 0 clean pre-weeks before Week 9, and Weeks 13–16 are not present in this file. "
        "To evaluate complete 4-week Before and After windows across Weeks 9–101, select the **Full Production Dataset** in the sidebar."
    )
else:
    b1, b2, b3 = st.columns(3)
    b1.success(f"✅ **Production Data:** {len(raw_df):,} transactions loaded")
    b2.info(f"📅 **Fiscal Weeks:** {int(raw_df['WEEK_NO'].min())} to {int(raw_df['WEEK_NO'].max())} (711 Days)")
    b3.info("🏷️ **Causal Tracking Audited:** Weeks 9–101 (Displays & Mailers)")

# -------------------- Sidebar filters --------------------

st.sidebar.header("Explore the data")

# 1. Department Filter (Top of Hierarchy)
all_departments = sorted(
    raw_df["DEPARTMENT"].dropna().astype(str).unique().tolist()
)
default_dept = ["GROCERY"] if "GROCERY" in all_departments else all_departments[:1]
selected_departments = st.sidebar.multiselect(
    "1. Department",
    all_departments,
    default=default_dept,
    help="Select one or more retail departments to filter products.",
)

dept_df = raw_df[
    raw_df["DEPARTMENT"].astype(str).isin(selected_departments)
] if selected_departments else raw_df.iloc[0:0]

# 2. Commodity Filter (Cascades from Department)
available_commodities = sorted(
    dept_df["COMMODITY_DESC"].dropna().astype(str).unique().tolist()
) if not dept_df.empty else []

selected_commodities = st.sidebar.multiselect(
    "2. Commodity",
    available_commodities,
    default=[],
    help="Cascades from Department. Leave empty to include all commodities in the selected department(s).",
)

if selected_commodities:
    comm_df = dept_df[dept_df["COMMODITY_DESC"].astype(str).isin(selected_commodities)]
else:
    comm_df = dept_df

# 3. Product Filter (Cascades from Commodity & Department)
available_products_df = (
    comm_df[["PRODUCT_ID", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"]]
    .drop_duplicates(subset=["PRODUCT_ID"])
    .sort_values("PRODUCT_ID")
) if not comm_df.empty else pd.DataFrame(columns=["PRODUCT_ID", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"])

product_map = {0: "All Products in Selection"}
for _, r in available_products_df.iterrows():
    pid = int(r["PRODUCT_ID"])
    desc = str(r["SUB_COMMODITY_DESC"]).strip() if pd.notna(r["SUB_COMMODITY_DESC"]) else ""
    sz = str(r["CURR_SIZE_OF_PRODUCT"]).strip() if pd.notna(r["CURR_SIZE_OF_PRODUCT"]) else ""
    extra = f" - {desc}" if desc else ""
    if sz:
        extra += f" ({sz})"
    product_map[pid] = f"ID {pid}{extra}"

selected_product_id = st.sidebar.selectbox(
    "3. Product",
    options=list(product_map.keys()),
    format_func=lambda x: product_map[x],
    index=0,
    help="Cascades from Commodity. Choose 'All Products' or isolate an individual product item.",
)

# 4. Brand Filter (Optional secondary filter)
available_brands = ["All"] + sorted(
    comm_df["BRAND"].dropna().astype(str).unique().tolist()
) if not comm_df.empty else ["All"]
selected_brand = st.sidebar.selectbox("Brand", available_brands, index=0)

# Base DataFrame after category hierarchy
base_df = comm_df.copy()
if selected_product_id != 0:
    base_df = base_df[base_df["PRODUCT_ID"] == selected_product_id]
if selected_brand != "All":
    base_df = base_df[base_df["BRAND"].astype(str) == selected_brand]

if base_df.empty:
    st.warning("No transactions match the selected filter combination. Adjust your filters in the sidebar.")
    st.stop()

# 5. Week Range Slider
available_min = int(base_df["WEEK_NO"].min())
available_max = int(base_df["WEEK_NO"].max())

if available_min < available_max:
    week_range = st.sidebar.slider(
        "Week range",
        min_value=available_min,
        max_value=available_max,
        value=(available_min, available_max),
        help="Filter transactions by fiscal week range.",
    )
else:
    week_range = (available_min, available_max)
    st.sidebar.info(
        f"📅 **Active Week:** Week {available_min} "
        "(This item has recorded transactions in only 1 fiscal week)."
    )

# 6. Data Cleaning & Promotion Sensitivity Controls
include_returns = st.sidebar.checkbox(
    "Include returns / refunds",
    value=False,
    help="Off by default: analyses exclude rows where quantity <= 0 or sales value < 0.",
)

promo_threshold = st.sidebar.slider(
    "Promotion episode threshold (%)",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
    help="Minimum % of transactions with causal promotion evidence required to flag a week as promoted.",
)

# Apply week and return filters
filtered_df = base_df[
    base_df["WEEK_NO"].between(week_range[0], week_range[1])
].copy()

if not include_returns:
    filtered_df = filtered_df[
        (filtered_df["QUANTITY"] > 0)
        & (filtered_df["SALES_VALUE"] >= 0)
    ].copy()

if filtered_df.empty:
    st.warning("No transactions remain for the selected week range.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"**Filtered Transactions:** {len(filtered_df):,} rows")
st.sidebar.caption(f"**Active Households:** {filtered_df['household_key'].nunique():,}")
st.sidebar.caption(f"**Distinct Products:** {filtered_df['PRODUCT_ID'].nunique():,}")

show_data = st.sidebar.checkbox("Show filtered data preview", value=False)
if show_data:
    st.subheader("Filtered data preview")
    st.dataframe(filtered_df.head(500), use_container_width=True)

# -------------------- Weekly analytics --------------------

weekly = get_weekly_trends(
    filtered_df,
    include_returns=include_returns,
    min_week=week_range[0],
    max_week=week_range[1],
    promo_threshold_pct=float(promo_threshold),
    min_coverage_pct=100.0,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Weekly Trends",
        "🏷️ Promotion Evidence",
        "🔍 Before / During / After",
        "👥 Household Repurchase",
        "🔄 Explore Similar Products",
    ]
)

with tab1:
    st.subheader("Weekly Purchasing Trends Over Time")

    # High-level metric summary ribbon
    m1, m2, m3, m4, m5 = st.columns(5)
    tot_units = float(weekly["total_units"].sum()) if not weekly.empty else 0.0
    tot_sales = float(weekly["total_sales"].sum()) if not weekly.empty else 0.0
    tot_hh = int(filtered_df["household_key"].nunique())
    tot_baskets = int(weekly["total_baskets"].sum()) if not weekly.empty else 0
    promoted_wks = int(weekly["is_promoted_week"].sum()) if not weekly.empty else 0

    m1.metric("Total Units Sold", f"{tot_units:,.0f}")
    m2.metric("Total Net Sales", f"${tot_sales:,.2f}")
    m3.metric("Active Households", f"{tot_hh:,}")
    m4.metric("Shopping Trips", f"{tot_baskets:,}")
    m5.metric("Promoted Weeks", f"{promoted_wks} / {len(weekly)}")

    if weekly.empty or tot_units == 0:
        st.warning("No weekly trend data is available for these filters.")
    else:
        metric_choices = {
            "total_sales": "Total Net Sales ($)",
            "total_units": "Total Units Sold",
            "purchasing_households": "Active Purchasing Households",
            "total_baskets": "Total Shopping Baskets",
            "avg_unit_price": "Average Realized Price per Unit ($)",
            "spend_per_household": "Average Spend per Active Household ($)",
            "units_per_household": "Average Units per Active Household",
            "baskets_per_household": "Average Shopping Trips per Household",
        }

        selected_metric = st.selectbox(
            "Select purchasing metric to display over time (Y-Axis):",
            list(metric_choices.keys()),
            index=1,  # Default to 'total_units' (Purchasing Quantity)
            format_func=lambda x: metric_choices[x],
            help="Evaluate purchasing quantity, net revenue, or shopper activity across time."
        )

        # Build status string for weekly classification
        weekly_plot = weekly.copy()
        def _classify_week(row):
            if row.get("causal_coverage_status") == "Unknown" or row["WEEK_NO"] < 9 or row["WEEK_NO"] > 101:
                return "Unmonitored Causal Coverage"
            elif row.get("is_promoted_week"):
                return f"Promoted (≥ {promo_threshold}%)"
            else:
                return "Unpromoted Baseline"

        weekly_plot["promo_display_status"] = weekly_plot.apply(_classify_week, axis=1)

        # Plot trend line with promotion indicators
        trend = px.line(
            weekly_plot,
            x="WEEK_NO",
            y=selected_metric,
            markers=True,
            title=f"Visual 1: Weekly Purchasing Over Time — {metric_choices[selected_metric]}",
            labels={"WEEK_NO": "Fiscal Week (Time)", selected_metric: metric_choices[selected_metric]},
        )
        trend.update_traces(
            line=dict(color="#2980b9", width=2),
            marker=dict(size=5, color="#2980b9"),
            name="Weekly Trend",
            hovertemplate="<b>Week %{x}</b><br>" + metric_choices[selected_metric] + ": %{y:,.1f}<extra></extra>"
        )

        # Highlight verified promoted weeks with distinct crimson diamond markers
        promo_sub = weekly_plot[weekly_plot["promo_display_status"] == f"Promoted (≥ {promo_threshold}%)"]
        if not promo_sub.empty:
            trend.add_scatter(
                x=promo_sub["WEEK_NO"],
                y=promo_sub[selected_metric],
                mode="markers",
                marker=dict(size=12, color="#e74c3c", symbol="diamond", line=dict(width=1, color="#2c3e50")),
                name=f"Verified Promoted Week (≥ {promo_threshold}%)",
                hovertemplate="<b>Week %{x}</b><br>Status: <b>Promoted (≥ " + str(promo_threshold) + "%)</b><br>" + metric_choices[selected_metric] + ": %{y:,.1f}<extra></extra>"
            )

        # Shade unmonitored weeks 1-8 only if the displayed range includes weeks <= 8
        if weekly_plot["WEEK_NO"].min() <= 8:
            trend.add_vrect(
                x0=0.5, x1=8.5,
                fillcolor="#bdc3c7", opacity=0.3,
                layer="below", line_width=1, line_dash="dot", line_color="#7f8c8d",
                annotation_text="Weeks 1–8: Unmonitored Causal Coverage",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#555")
            )
        # Shade week 102 if present
        if weekly_plot["WEEK_NO"].max() >= 102:
            trend.add_vrect(
                x0=101.5, x1=102.5,
                fillcolor="#bdc3c7", opacity=0.3,
                layer="below", line_width=1, line_dash="dot", line_color="#7f8c8d",
                annotation_text="Wk 102: Tracking End",
                annotation_position="top right",
                annotation_font=dict(size=9, color="#555")
            )

        trend.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=40, l=60, r=20),
        )
        st.plotly_chart(trend, use_container_width=True)

        # Discount Depth chart with 3-color palette distinguishing unmonitored weeks
        if "avg_discount_depth_pct" in weekly_plot.columns:
            discount_fig = px.bar(
                weekly_plot,
                x="WEEK_NO",
                y="avg_discount_depth_pct",
                title="Average Recorded Discount Depth by Fiscal Week (%)",
                labels={
                    "WEEK_NO": "Fiscal Week",
                    "avg_discount_depth_pct": "Avg Discount Depth (%)",
                    "promo_display_status": "Coverage & Promotion Status",
                },
                color="promo_display_status",
                color_discrete_map={
                    f"Promoted (≥ {promo_threshold}%)": "#e74c3c",
                    "Unpromoted Baseline": "#3498db",
                    "Unmonitored Causal Coverage": "#bdc3c7",
                },
            )
            discount_fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(discount_fig, use_container_width=True)

        st.markdown("**Weekly Coverage and Promotion Status Table**")
        display_cols = [
            c for c in [
                "WEEK_NO", "total_sales", "total_units",
                "purchasing_households", "causal_promo_pct",
                "promo_evaluation_status", "causal_coverage_status",
            ] if c in weekly.columns
        ]
        table_df = weekly[display_cols].copy()
        zero_sales_mask = (table_df["total_units"] == 0) & (table_df["WEEK_NO"].between(9, 101))
        if zero_sales_mask.any():
            table_df.loc[zero_sales_mask, "promo_evaluation_status"] = "No Sales (0 units)"

        st.dataframe(
            table_df,
            column_config={
                "WEEK_NO": st.column_config.NumberColumn("Fiscal Week", format="%d"),
                "total_sales": st.column_config.NumberColumn("Net Sales ($)", format="$%.2f"),
                "total_units": st.column_config.NumberColumn("Units Sold", format="%d"),
                "purchasing_households": st.column_config.NumberColumn("Shoppers", format="%d"),
                "causal_promo_pct": st.column_config.NumberColumn("Causal Promo Rate", format="%.1f%%"),
                "promo_evaluation_status": st.column_config.TextColumn("Promotion Status"),
                "causal_coverage_status": st.column_config.TextColumn("Causal Tracking"),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "ℹ️ **Status Guide:** Weeks 9–101 are actively audited by Dunnhumby and classified as **Promoted** (≥ threshold) or **Unpromoted** (quiet baseline). "
            "Weeks 1–8 and Week 102 were not audited for store circulars/displays and are classified as **Unknown** (unmonitored causal coverage)."
        )

with tab2:
    st.subheader("Promotion evidence")

    evidence_cols = [
        c for c in [
            "has_causal_evidence", "has_discount_evidence",
            "has_display_feature", "has_mailer_feature",
            "causal_coverage_status", "promo_evidence_type",
        ] if c in filtered_df.columns
    ]

    if not evidence_cols:
        st.warning("Promotion evidence fields are not available.")
    else:
        col1, col2, col3 = st.columns(3)

        if "has_causal_evidence" in filtered_df.columns:
            col1.metric(
                "Rows with causal records",
                f"{filtered_df['has_causal_evidence'].mean() * 100:.1f}%",
            )
        if "has_discount_evidence" in filtered_df.columns:
            col2.metric(
                "Rows with discount evidence",
                f"{filtered_df['has_discount_evidence'].mean() * 100:.1f}%",
            )

        if {
            "has_causal_evidence", "has_discount_evidence"
        }.issubset(filtered_df.columns):
            overlap = (
                filtered_df["has_causal_evidence"].fillna(False).astype(bool)
                & filtered_df["has_discount_evidence"].fillna(False).astype(bool)
            ).mean() * 100
            col3.metric("Both types of evidence", f"{overlap:.1f}%")

        if "promo_evidence_type" in filtered_df.columns:
            evidence_counts = (
                filtered_df["promo_evidence_type"]
                .astype("string")
                .fillna("Unclassified")
                .value_counts()
                .rename_axis("Evidence type")
                .reset_index(name="Rows")
            )
            fig = px.bar(
                evidence_counts,
                x="Evidence type",
                y="Rows",
                title="Transaction rows by promotion evidence type",
            )
            st.plotly_chart(fig, use_container_width=True)

        if "causal_coverage_status" in filtered_df.columns:
            coverage = (
                filtered_df["causal_coverage_status"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Coverage status")
                .reset_index(name="Rows")
            )
            fig = px.pie(
                coverage,
                names="Coverage status",
                values="Rows",
                title="Causal tracking coverage",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "A causal record is evidence of a recorded display/mailer placement, "
            "not proof that the promotion caused a sales increase. Unknown "
            "coverage must not be interpreted as confirmed absence of promotion."
        )

with tab3:
    st.subheader("Before / During / After (BDA) Promotion Analysis")
    st.caption(
        "Evaluates purchasing rates across a 4-week pre-baseline, the actual promotion duration, "
        "and a 4-week post-promotion window, with automated clean window safeguards that exclude "
        "adjacent promotional events."
    )

    if selected_product_id == 0 and not selected_commodities:
        st.caption(
            "💡 **Level of Analysis Note:** You are viewing promotional episodes aggregated across all products in the selected department. "
            "To evaluate individual product sales cycles and post-promotional volume patterns, choose a specific **Commodity** or **Product** in the sidebar."
        )

    episodes = identify_promotion_episodes(
        weekly,
        min_promo_pct=float(promo_threshold),
        pre_window=4,
        post_window=4,
        enforce_causal_boundaries=True,
    )

    if episodes.empty:
        st.info(
            f"No candidate promotion episodes detected at the {promo_threshold}% causal promo threshold. "
            "Try lowering the 'Promotion episode threshold' slider in the sidebar, or selecting a broader "
            "commodity or week range."
        )
    else:
        # Categorize episodes into complete vs boundary-truncated
        complete_ids = []
        episode_labels = {}
        for _, row in episodes.iterrows():
            ep_id = str(row["episode_id"])
            s_wk = int(row["start_week"])
            e_wk = int(row["end_week"])
            dur = int(row["duration_weeks"])
            pre_w = int(row["pre_actual_weeks"])
            post_w = int(row["post_actual_weeks"])
            is_valid = bool(row.get("causal_coverage_valid", False)) and pre_w > 0 and post_w > 0

            pre_s_chk = int(row.get("pre_start_week_clean", row.get("pre_start_week_nominal", s_wk - 4)))
            post_e_chk = int(row.get("post_end_week_clean", row.get("post_end_week_nominal", e_wk + 4)))

            has_pre_trans = not filtered_df[(filtered_df["WEEK_NO"] >= pre_s_chk) & (filtered_df["WEEK_NO"] < s_wk)].empty
            has_post_trans = not filtered_df[(filtered_df["WEEK_NO"] > e_wk) & (filtered_df["WEEK_NO"] <= post_e_chk)].empty

            if is_valid and has_pre_trans and has_post_trans:
                complete_ids.append(ep_id)
                tag = "✅ Complete Windows"
            elif not has_pre_trans and not has_post_trans:
                tag = "⚠️ Zero Transactions in Pre & Post Windows"
            elif not has_pre_trans:
                tag = "⚠️ Zero Transactions in Pre-Baseline"
            elif not has_post_trans:
                tag = "⚠️ Zero Transactions in Post-Window"
            elif pre_w == 0:
                tag = "⚠️ Incomplete Baseline (0 pre-weeks before Week 9)"
            elif post_w == 0:
                tag = "⚠️ Incomplete Post-Window (0 post-weeks)"
            else:
                tag = "⚠️ Boundary Truncated"

            dur_str = f"{dur} wk{'s' if dur > 1 else ''}"
            episode_labels[ep_id] = (
                f"{ep_id}: Weeks {s_wk}–{e_wk} ({dur_str}) | Pre: {pre_w} wks | Post: {post_w} wks [{tag}]"
            )

        st.markdown(f"**Identified {len(episodes)} promotional candidate event(s)**")

        # Auto-select the first complete episode if available
        default_ep_index = 0
        if complete_ids:
            first_complete = complete_ids[0]
            all_keys = list(episode_labels.keys())
            if first_complete in all_keys:
                default_ep_index = all_keys.index(first_complete)

        selected_episode_id = st.selectbox(
            "Select a promotion episode to evaluate:",
            list(episode_labels.keys()),
            index=default_ep_index,
            format_func=lambda x: episode_labels[x],
            help="Complete episodes have verified 4-week clean pre-baselines and post-windows within audited causal tracking.",
        )
        episode = episodes[
            episodes["episode_id"] == selected_episode_id
        ].iloc[0].to_dict()

        comparison = compare_before_during_after(
            filtered_df,
            episode,
            use_clean_windows=True,
            include_returns=include_returns,
            enforce_causal_boundaries=True,
        )

        if comparison.empty:
            st.warning("The Before / During / After comparison could not be calculated.")
        else:
            result = comparison.iloc[0]

            # Episode Boundary & Coverage Notices
            start_week = int(episode["start_week"])
            end_week = int(episode["end_week"])
            pre_act_wks = int(episode.get("pre_actual_weeks", 0))
            post_act_wks = int(episode.get("post_actual_weeks", 0))
            dur_wks = int(episode["duration_weeks"])

            pre_start = int(episode.get("pre_start_week_clean", episode.get("pre_start_week_nominal", start_week - 4)))
            post_end = int(episode.get("post_end_week_clean", episode.get("post_end_week_nominal", end_week + 4)))

            pre_rows = filtered_df[
                (filtered_df["WEEK_NO"] >= pre_start) & (filtered_df["WEEK_NO"] < start_week)
            ]
            during_rows = filtered_df[
                (filtered_df["WEEK_NO"] >= start_week) & (filtered_df["WEEK_NO"] <= end_week)
            ]
            post_rows = filtered_df[
                (filtered_df["WEEK_NO"] > end_week) & (filtered_df["WEEK_NO"] <= post_end)
            ]

            if not include_returns:
                pre_rows = pre_rows[(pre_rows["QUANTITY"] > 0) & (pre_rows["SALES_VALUE"] >= 0)]
                during_rows = during_rows[(during_rows["QUANTITY"] > 0) & (during_rows["SALES_VALUE"] >= 0)]
                post_rows = post_rows[(post_rows["QUANTITY"] > 0) & (post_rows["SALES_VALUE"] >= 0)]

            # Explicit Explanation of Episode Window Status
            if pre_act_wks == 0:
                st.warning(
                    f"⚠️ **Incomplete Baseline (0 Clean Weeks):** Episode {selected_episode_id} starts at Week {start_week}. "
                    "Its 4-week nominal pre-baseline (Weeks 5–8) falls prior to Week 9, where audited causal circular tracking begins. "
                    "Because unmonitored weeks cannot be assumed to be unpromoted, 0 clean baseline weeks exist within verified causal coverage. "
                    "Promotion lift cannot be computed without a baseline."
                )
            if post_rows.empty and post_act_wks > 0:
                st.warning(
                    f"⚠️ **Missing Post-Window Transactions:** No transactions exist in the active dataset for post-promotion Weeks {end_week + 1}–{post_end}. "
                    + ("Switch to the **Full Production Dataset** in the sidebar to observe subsequent transaction weeks." if is_sample else "")
                )

            # 1. Phase Purchasing Rates (No delta arrows on dollar sales)
            st.markdown("#### 1. Normalized Weekly Purchasing Rates by Phase")
            c1, c2, c3 = st.columns(3)

            with c1:
                if pre_act_wks > 0 and not pre_rows.empty and pd.notna(result.get("pre_units_per_week")):
                    st.metric("1. Before (Pre-Clean)", f"{result['pre_units_per_week']:,.1f} units/wk")
                    st.caption(f"Sales: ${result['pre_sales_per_week']:,.2f}/wk ({pre_act_wks} clean wks)")
                else:
                    st.metric("1. Before (Pre-Clean)", "N/A")
                    st.caption(f"0 clean pre-weeks within causal data")

            with c2:
                if not during_rows.empty and pd.notna(result.get("during_units_per_week")):
                    st.metric(f"2. During Promotion ({dur_wks} wk{'s' if dur_wks > 1 else ''})", f"{result['during_units_per_week']:,.1f} units/wk")
                    st.caption(f"Sales: ${result['during_sales_per_week']:,.2f}/wk (Weeks {start_week}–{end_week})")
                else:
                    st.metric(f"2. During Promotion ({dur_wks} wks)", "N/A")
                    st.caption("0 recorded transactions during promotion")

            with c3:
                if post_act_wks > 0 and not post_rows.empty and pd.notna(result.get("post_units_per_week")):
                    st.metric("3. After (Post-Clean)", f"{result['post_units_per_week']:,.1f} units/wk")
                    st.caption(f"Sales: ${result['post_sales_per_week']:,.2f}/wk ({post_act_wks} clean wks)")
                else:
                    st.metric("3. After (Post-Clean)", "N/A")
                    st.caption("0 post transactions available in dataset")

            # 2. Lift & Post-Promotion Impact Metrics
            st.markdown("#### 2. Observed Promotion Lift & Post-Promotion Changes")
            k1, k2, k3, k4 = st.columns(4)

            has_baseline = bool(result.get("has_reliable_baseline", False)) and pre_act_wks > 0 and not pre_rows.empty and not during_rows.empty
            has_post = post_act_wks > 0 and not post_rows.empty and pd.notna(result.get("post_units_per_week"))

            unit_lift_val = f"{result['unit_lift_pct']:+.1f}%" if has_baseline and pd.notna(result.get("unit_lift_pct")) else "N/A"
            sales_lift_val = f"{result['sales_lift_pct']:+.1f}%" if has_baseline and pd.notna(result.get("sales_lift_pct")) else "N/A"
            post_vs_pre_val = f"{result['post_vs_pre_unit_pct']:+.1f}%" if has_baseline and has_post and pd.notna(result.get("post_vs_pre_unit_pct")) else "N/A"
            post_dip_val = f"{result['post_vs_during_dip_pct']:+.1f}%" if has_post and not during_rows.empty and pd.notna(result.get("post_vs_during_dip_pct")) else "N/A"

            k1.metric("Unit Volume Lift", unit_lift_val, help="Percentage increase in weekly unit volume during the promotion versus the clean pre-baseline.")
            k2.metric("Net Sales Lift", sales_lift_val, help="Percentage increase in weekly dollar revenue during the promotion versus the clean pre-baseline.")
            k3.metric("Post vs Pre Change", post_vs_pre_val, help="Normalized weekly unit volume in the clean post-period relative to the pre-promotion baseline.")
            k4.metric("Post vs During Dip", post_dip_val, help="Observed volume drop from the promotional peak to clean post-promotion weeks.")

            if not has_baseline:
                st.caption("*Note: Promotion lift percentages are reported as 'N/A' because there are 0 pre-promotion clean observation weeks within verified causal coverage.*")

            # 3. Contextual Timeline Chart
            st.markdown("#### 3. Weekly Volume Context Surrounding Episode")
            timeline_weeks = weekly[
                (weekly["WEEK_NO"] >= max(1, pre_start - 1))
                & (weekly["WEEK_NO"] <= min(102, post_end + 1))
            ].copy()

            if not timeline_weeks.empty:
                def _assign_phase(wk):
                    if pre_start <= wk < start_week:
                        return "1. Before (Pre-Clean)"
                    elif start_week <= wk <= end_week:
                        return "2. During (Promotion)"
                    elif end_week < wk <= post_end:
                        return "3. After (Post-Clean)"
                    return "Adjacent Fiscal Week"

                timeline_weeks["Episode Phase"] = timeline_weeks["WEEK_NO"].apply(_assign_phase)
                t_fig = px.bar(
                    timeline_weeks,
                    x="WEEK_NO",
                    y="total_units",
                    color="Episode Phase",
                    color_discrete_map={
                        "1. Before (Pre-Clean)": "#3498db",
                        "2. During (Promotion)": "#e74c3c",
                        "3. After (Post-Clean)": "#2ecc71",
                        "Adjacent Fiscal Week": "#bdc3c7",
                    },
                    title=f"Weekly Unit Volume Surrounding Episode {selected_episode_id} (Fiscal Weeks {pre_start}–{post_end})",
                    labels={"WEEK_NO": "Fiscal Week (WEEK_NO)", "total_units": "Total Units Sold"},
                )
                st.plotly_chart(t_fig, use_container_width=True)

            # 4. Executive Data Findings & Interpretation Box (Requirement 5)
            st.markdown("#### 4. Data Findings & Methodological Interpretation")
            with st.container():
                st.markdown(
                    f"""
                    **Key Findings for {selected_episode_id} (Weeks {start_week}–{end_week}):**
                    - **Observed Purchasing:** During the {dur_wks}-week promotional event, purchasing averaged **{result['during_units_per_week']:,.1f} units/week** (${result['during_sales_per_week']:,.2f}/week) across **{int(result.get('during_total_households', 0)):,} active households**.
                    - **Baseline Comparison:** {"A pre-promotion baseline of " + f"{result['pre_units_per_week']:,.1f} units/wk was observed, reflecting an observed lift of " + unit_lift_val if has_baseline else "No verified pre-promotion baseline exists within audited causal coverage (pre-window falls before Week 9)."}
                    - **Post-Promotion Trajectory:** {"Post-promotion purchasing averaged " + f"{result['post_units_per_week']:,.1f} units/wk, reflecting a post vs during volume dip of " + post_dip_val if has_post else "Post-promotion purchasing metrics are unavailable due to lack of recorded post-window transaction data."}
                    - **Comparison Window Integrity:** Pre-window actual clean weeks = **{pre_act_wks}** ({episode.get('pre_truncation_reason', 'None')}); Post-window actual clean weeks = **{post_act_wks}** ({episode.get('post_truncation_reason', 'None')}).
                    - **Analytical Limitation:** *Observed post-promotion volume declines describe empirical transaction timing; they do not establish proven consumer stockpiling or definitive causal effects. Post-period fluctuations may reflect baseline shopping cycle length, household pantry consumption, or unmonitored competitive promotions.*
                    """
                )

            # 5. Window Truncation & Safeguards Metadata Table
            st.markdown("**Window Truncation & Safeguards Audit Details**")
            audit_cols = [
                "episode_id", "start_week", "end_week", "duration_weeks",
                "pre_actual_weeks", "post_actual_weeks",
                "pre_is_truncated", "pre_truncation_reason",
                "post_is_truncated", "post_truncation_reason",
                "causal_coverage_valid", "is_complete_window"
            ]
            st.dataframe(
                pd.DataFrame([episode])[audit_cols],
                use_container_width=True,
                hide_index=True,
            )

with tab4:
    st.subheader("Household Repurchase & Post-Promotion Behavior")
    st.caption(
        "Tracks repeat purchasing timing, buyer cohorts (trial vs existing buyers), and shopping "
        "cycle intervals (exact calendar DAY) to evaluate customer retention after the promotional event."
    )

    if episodes.empty:
        st.info("No candidate promotion episodes available. Adjust filters to detect promotion episodes.")
    else:
        if "selected_episode_id" not in locals() or selected_episode_id not in [r["episode_id"] for _, r in episodes.iterrows()]:
            selected_episode_id = episodes.iloc[0]["episode_id"]

        household_episode = episodes[
            episodes["episode_id"] == selected_episode_id
        ].iloc[0].to_dict()

        st.markdown(f"**Analyzing Household Cohorts for Episode {household_episode['episode_id']} (Weeks {int(household_episode['start_week'])}–{int(household_episode['end_week'])})**")

        household_summary = get_household_behavior_summary(
            filtered_df,
            household_episode,
            use_clean_windows=True,
            include_returns=include_returns,
            enforce_causal_boundaries=True,
        )

        if household_summary.empty:
            st.warning("No household behavior summary could be computed for this selection.")
        else:
            h = household_summary.iloc[0]

            if h.get("coverage_warning", "None") != "None":
                st.warning(f"**Coverage Notice:** {h['coverage_warning']}")

            if not bool(h.get("causal_coverage_valid", False)):
                st.info(
                    "Coverage is incomplete for this episode. Cohort metrics are descriptive and "
                    "should not be treated as confirmed long-term retention."
                )

            # 1. Household Metric Cards (No delta arrows on percentages)
            st.markdown("#### 1. Buyer Cohort Penetration & Retention")
            hc1, hc2, hc3 = st.columns(3)
            promo_buyers = int(h.get("promo_buyers_count", 0))
            repeat_buyers = int(h.get("repeat_buyers_count", 0))
            repurchase_rate = float(h.get("post_repurchase_rate_pct", 0.0))

            trial_buyers = int(h.get("trial_buyers_count", 0))
            trial_repeat = int(h.get("trial_repeat_count", 0))
            trial_retention = float(h.get("trial_retention_rate_pct", 0.0))

            with hc1:
                st.metric("Promotion Buyers", f"{promo_buyers:,}")
                st.caption("Distinct households buying during promotion")

            with hc2:
                st.metric("Repeat Buyers (Post-Clean)", f"{repeat_buyers:,}")
                st.caption(f"{repurchase_rate:.1f}% repeat repurchase rate")

            with hc3:
                st.metric("Trial Buyers Retained", f"{trial_repeat:,} / {trial_buyers:,}")
                st.caption(f"{trial_retention:.1f}% trial retention rate")

            # 2. Timing & Shopping Cycle Metrics
            st.markdown("#### 2. Inter-Purchase Interval & Cycle Shift (Native DAY)")
            tc1, tc2, tc3 = st.columns(3)

            avg_return_days = h.get("avg_days_to_first_post_purchase")
            pre_cycle = h.get("avg_pre_interpurchase_days")
            post_cycle = h.get("avg_post_interpurchase_days")
            cycle_delta = h.get("interpurchase_day_change")

            with tc1:
                val = f"{avg_return_days:.1f} days" if pd.notna(avg_return_days) else "N/A"
                st.metric("Avg. Days to First Return", val)
                st.caption("Mean calendar days from promotion to 1st return")

            with tc2:
                val = f"{post_cycle:.1f} vs {pre_cycle:.1f} days" if pd.notna(pre_cycle) and pd.notna(post_cycle) else "N/A"
                st.metric("Post vs Pre Shopping Cycle", val)
                shift_str = f"{cycle_delta:+.1f} days shift" if pd.notna(cycle_delta) else "Cycle shift unavailable"
                st.caption(shift_str)

            with tc3:
                n_samp = int(h.get("days_to_return_sample_size", 0))
                st.metric("Cohort Sample Size", f"{n_samp:,} buyers")
                st.caption("Repeat buyers used to compute return intervals")

            # 3. Visual Cohort Breakdown Chart
            st.markdown("#### 3. Household Cohort Retention Breakdown")
            cohort_data = pd.DataFrame({
                "Buyer Cohort": [
                    "All Promotion Buyers",
                    "Repeat Buyers (Post)",
                    "Trial Buyers (New to Brand)",
                    "Retained Trial Buyers",
                ],
                "Households": [
                    promo_buyers,
                    repeat_buyers,
                    trial_buyers,
                    trial_repeat,
                ],
                "Cohort Type": ["Overall Promo Buyers", "Overall Promo Buyers", "Trial Cohort", "Trial Cohort"],
            })

            cohort_fig = px.bar(
                cohort_data,
                x="Buyer Cohort",
                y="Households",
                color="Cohort Type",
                color_discrete_map={"Overall Promo Buyers": "#3498db", "Trial Cohort": "#e67e22"},
                title=f"Household Penetration and Retention for Episode {household_episode['episode_id']}",
                text_auto=True,
            )
            st.plotly_chart(cohort_fig, use_container_width=True)

            # 4. Methodological Statement
            st.markdown(
                f"""
                **Household Repurchase Summary for {household_episode['episode_id']}:**
                - Out of **{promo_buyers:,}** households purchasing during the promotion, **{repeat_buyers:,} ({repurchase_rate:.1f}%)** returned to make a repurchase during the clean post-promotion window.
                - Among **{trial_buyers:,}** trial buyers (households with 0 purchases during the pre-promotion baseline), **{trial_repeat:,} ({trial_retention:.1f}%)** repurchased in the post-window.
                - Repeat buyers took an average of **{avg_return_days:.1f} days** to make their first return trip after purchasing during the promotion.
                """ if promo_buyers > 0 and pd.notna(avg_return_days) else ""
            )

            # Summary Table
            st.markdown("**Full Household Behavior Summary Table**")
            st.dataframe(household_summary, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Explore Similar Products in Commodity")

    if selected_product_id == 0:
        st.info(
            "👉 **Select a specific product in the sidebar (Filter 3: Product) to explore similar products within its commodity.**\n\n"
            "Currently, 'All Products in Selection' is active. Choose an individual product to compare its purchasing dynamics, "
            "promotional frequency, and weekly volume trends against comparable peer products."
        )
    else:
        # 1. Product Metadata Banner
        focal_rows = raw_df[raw_df["PRODUCT_ID"] == selected_product_id]
        focal_dept = str(focal_rows["DEPARTMENT"].iloc[0]) if not focal_rows.empty else "N/A"
        focal_comm = str(focal_rows["COMMODITY_DESC"].iloc[0]) if not focal_rows.empty else "N/A"
        focal_subcomm = str(focal_rows["SUB_COMMODITY_DESC"].iloc[0]) if not focal_rows.empty else "N/A"
        focal_brand = str(focal_rows["BRAND"].iloc[0]) if not focal_rows.empty else "N/A"
        focal_size = str(focal_rows["CURR_SIZE_OF_PRODUCT"].iloc[0]) if (not focal_rows.empty and pd.notna(focal_rows["CURR_SIZE_OF_PRODUCT"].iloc[0])) else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Focal Product:** `ID {selected_product_id}`")
        c2.markdown(f"**Commodity:** {focal_comm}")
        c3.markdown(f"**Sub-Commodity:** {focal_subcomm}")
        c4.markdown(f"**Brand & Size:** {focal_brand} ({focal_size})")

        # 2. Extract similar products summary within the same commodity
        if include_returns:
            clean_df_comm = raw_df
        else:
            clean_df_comm = raw_df[(raw_df["QUANTITY"] > 0) & (raw_df["SALES_VALUE"] >= 0)]

        similar_summary = get_similar_products_summary(
            clean_df_comm,
            focal_comm,
            current_product_id=selected_product_id,
            top_n=10,
        )

        if similar_summary.empty:
            st.warning(f"No comparable products found in commodity '{focal_comm}'.")
        else:
            st.markdown("#### 1. Peer Products Comparison in Commodity")
            st.caption(
                f"Top volume products in **{focal_comm}**, highlighting the currently selected focal product (`ID {selected_product_id}`)."
            )

            # Display table with clear indicators
            table_summary = similar_summary.copy()
            table_summary["Role"] = table_summary["is_focal"].apply(
                lambda x: "⭐ Focal Product" if x else "Peer Product"
            )
            col_order = [
                "Role", "PRODUCT_ID", "SUB_COMMODITY_DESC", "BRAND",
                "CURR_SIZE_OF_PRODUCT", "total_units", "total_sales",
                "distinct_shoppers", "avg_unit_price", "promoted_weeks_count"
            ]
            display_df = table_summary[[c for c in col_order if c in table_summary.columns]]

            st.dataframe(
                display_df,
                column_config={
                    "Role": st.column_config.TextColumn("Product Type"),
                    "PRODUCT_ID": st.column_config.NumberColumn("Product ID", format="%d"),
                    "SUB_COMMODITY_DESC": st.column_config.TextColumn("Sub-Commodity"),
                    "BRAND": st.column_config.TextColumn("Brand"),
                    "CURR_SIZE_OF_PRODUCT": st.column_config.TextColumn("Package Size"),
                    "total_units": st.column_config.NumberColumn("Units Sold", format="%d"),
                    "total_sales": st.column_config.NumberColumn("Net Sales ($)", format="$%.2f"),
                    "distinct_shoppers": st.column_config.NumberColumn("Shoppers", format="%d"),
                    "avg_unit_price": st.column_config.NumberColumn("Avg Unit Price ($)", format="$%.2f"),
                    "promoted_weeks_count": st.column_config.NumberColumn("Promoted Weeks", format="%d"),
                },
                use_container_width=True,
                hide_index=True,
            )

            # 3. Interactive Multi-Product Weekly Purchasing Comparison
            st.markdown("#### 2. Weekly Purchasing Trends: Focal vs. Peer Products")

            peer_labels = {}
            for _, r in similar_summary.iterrows():
                p_id = int(r["PRODUCT_ID"])
                star = "⭐ Focal" if r["is_focal"] else "Peer"
                peer_labels[p_id] = f"ID {p_id} - {r['BRAND']} ({r['SUB_COMMODITY_DESC']}) [{star}]"

            # Default to focal product + top 2 peers
            default_pids = [selected_product_id]
            peers = [int(p) for p in similar_summary["PRODUCT_ID"] if int(p) != selected_product_id][:2]
            default_selection = default_pids + peers

            ctrl_c1, ctrl_c2 = st.columns([3, 2])
            with ctrl_c1:
                selected_compare_pids = st.multiselect(
                    "Select products to compare over time (Max 5 recommended):",
                    options=list(peer_labels.keys()),
                    default=default_selection,
                    format_func=lambda x: peer_labels.get(x, f"ID {x}"),
                    help="Compare weekly purchasing trajectories to evaluate category-wide vs. product-specific shifts.",
                )
            with ctrl_c2:
                compare_metric_choice = st.selectbox(
                    "Comparison Metric (Y-Axis):",
                    ["total_units", "total_sales", "purchasing_households"],
                    index=0,
                    format_func=lambda x: {
                        "total_units": "Total Units Sold",
                        "total_sales": "Total Net Sales ($)",
                        "purchasing_households": "Active Purchasing Households",
                    }[x],
                    help="Metric displayed across fiscal weeks for the selected products.",
                )

            if not selected_compare_pids:
                st.info("Select at least one product above to render the comparison trend chart.")
            else:
                metric_name = {
                    "total_units": "Units Sold",
                    "total_sales": "Net Sales ($)",
                    "purchasing_households": "Active Households",
                }[compare_metric_choice]

                comparison_frames = []
                for p_id in selected_compare_pids:
                    p_trans = filter_data(clean_df_comm, product_id=p_id)
                    p_weekly = get_weekly_trends(
                        p_trans,
                        include_returns=include_returns,
                        min_week=week_range[0],
                        max_week=week_range[1],
                        promo_threshold_pct=float(promo_threshold),
                    )
                    p_weekly["Product"] = peer_labels.get(p_id, f"ID {p_id}")
                    p_weekly["is_focal_product"] = (p_id == selected_product_id)
                    comparison_frames.append(p_weekly)

                all_compare_weekly = pd.concat(comparison_frames, ignore_index=True)

                fig_compare = px.line(
                    all_compare_weekly,
                    x="WEEK_NO",
                    y=compare_metric_choice,
                    color="Product",
                    markers=True,
                    title=f"Visual 5: Weekly Purchasing Comparison — {metric_name} (Weeks {week_range[0]}–{week_range[1]})",
                    labels={
                        "WEEK_NO": "Fiscal Week (Time)",
                        compare_metric_choice: metric_name,
                        "Product": "Product Selection",
                    },
                )
                fig_compare.update_layout(
                    hovermode="x unified",
                    margin=dict(t=50, b=40, l=60, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                # Shade unmonitored weeks 1-8 if present
                if week_range[0] <= 8:
                    fig_compare.add_vrect(
                        x0=0.5, x1=8.5,
                        fillcolor="#bdc3c7", opacity=0.25,
                        layer="below", line_width=1, line_dash="dot", line_color="#7f8c8d",
                        annotation_text="Weeks 1–8: Unmonitored Causal Coverage",
                        annotation_position="top left",
                        annotation_font=dict(size=9, color="#555"),
                    )
                # Shade week 102 if present
                if week_range[1] >= 102:
                    fig_compare.add_vrect(
                        x0=101.5, x1=102.5,
                        fillcolor="#bdc3c7", opacity=0.25,
                        layer="below", line_width=1, line_dash="dot", line_color="#7f8c8d",
                        annotation_text="Wk 102: Tracking End",
                        annotation_position="top right",
                        annotation_font=dict(size=9, color="#555"),
                    )

                st.plotly_chart(fig_compare, use_container_width=True)

            # 4. Quick Peer Investigation Selector
            st.markdown("#### 3. Investigate a Related Peer Product")
            peer_candidates = [int(p) for p in similar_summary["PRODUCT_ID"] if int(p) != selected_product_id]
            if peer_candidates:
                chosen_peer = st.selectbox(
                    "Select a peer product to inspect summary details:",
                    options=peer_candidates,
                    format_func=lambda x: peer_labels.get(x, f"ID {x}"),
                    help="Inspect profile and key metrics for another product in this commodity.",
                )
                peer_row = similar_summary[similar_summary["PRODUCT_ID"] == chosen_peer].iloc[0]
                st.info(
                    f"🔍 **Peer Product Profile:** `ID {chosen_peer}` — **{peer_row['BRAND']}** ({peer_row['SUB_COMMODITY_DESC']})\n\n"
                    f"- **Package Size:** {peer_row['CURR_SIZE_OF_PRODUCT']}\n"
                    f"- **Total Units Sold:** {int(peer_row['total_units']):,} units\n"
                    f"- **Total Net Sales:** ${peer_row['total_sales']:,.2f}\n"
                    f"- **Active Purchasing Households:** {int(peer_row['distinct_shoppers']):,} households\n"
                    f"- **Average Realized Unit Price:** ${peer_row['avg_unit_price']:.2f} / unit\n"
                    f"- **Promoted Weeks Count:** {int(peer_row['promoted_weeks_count'])} weeks with audited causal activity\n\n"
                    f"👉 *To evaluate this peer product's full Before/During/After lift windows and household repurchase cycles across Tabs 1–4, "
                    f"choose **ID {chosen_peer}** in Sidebar Filter 3 (Product).*"
                )

            # 5. Methodological Explanation
            st.markdown("---")
            st.caption(
                "**Visual Analytics Methodological Principle (Cross-Product Comparison):**\n"
                "- Similar product purchasing trends allow category managers to evaluate whether promotional spikes and subsequent dips are isolated to a single brand or represent category-wide shopping dynamics.\n"
                "- Parallel volume spikes across similar products may reflect shared promotional pressure, seasonal category demand (e.g. holiday peaks), or brand substitution; they describe observed empirical purchasing patterns and do not prove actual household stockpiling or promotion causality.\n"
                "- Missing or incomplete promotion evidence (such as during Weeks 1–8 and Week 102) indicates that promotion status is unknown in the audit records, not necessarily that no promotion took place."
            )

st.divider()
st.caption(
    "**Visual Analytics Methodological Principle:** Observed purchasing patterns, post-promotion volume changes, "
    "and repurchase intervals describe empirical transactions within the audited dataset. They do not constitute "
    "proven causal claims or definitive stockpiling conclusions. Results depend on selected filters, promotion "
    "evidence thresholds, and clean comparison window availability."
)
