# ITCS 5122 - Analytics Validation & Streamlit Integration Guide
**Project**: Grocery Promotion & Stockpiling Explorer  
**Lead**: Teammate 1 (Data & Analytics Lead)  
**Target Milestone**: October 20 Progress II Presentation  
**Deliverables**: `analytics.py`, `test_analytics.py`, `reports/analytics_validation.md`

---

## 1. Executive Summary & Objective

In **Step 3 (Analytics Development)**, we developed a production-grade, modular analytics library ([analytics.py](file:///c:/Users/SiruiChen/Documents/GitHub/Visual-Analytics-Group-1/analytics.py)) that provides standardized analytical methods for **Teammate 2** to integrate into the Streamlit MVP.

All functions:
- Re-use the validated joins and multi-dimensional promotion classification from `data_preparation.py`.
- Operate on actual `product.csv` category fields (`DEPARTMENT`, `COMMODITY_DESC`, `SUB_COMMODITY_DESC`, `PRODUCT_ID`).
- Enforce strict calendar integrity (native `WEEK_NO` for weekly trends; native `DAY` for household repurchase intervals).
- Return structured pandas DataFrames with fixed schemas, ready for immediate rendering in Streamlit charts, tables, and metric cards (`st.dataframe`, `st.metric`, `st.line_chart`, Plotly).
- Strictly adhere to data integrity principles: **Missing causal coverage is never treated as a confirmed untreated control group**, and observed metric changes are framed objectively without unfounded causal claims.

---

## 2. API Reference & Function Specifications for Teammate 2

### 2.1 Category & Hierarchy Filtering
```python
import analytics as an

filtered_df = an.filter_data(
    df,
    department=None,          # e.g., 'GROCERY' or ['GROCERY', 'PRODUCE']
    commodity_desc=None,      # e.g., 'SOFT DRINKS'
    sub_commodity_desc=None,  # e.g., 'SOFT DRINKS 12/18&15PK CAN CAR'
    product_id=None           # e.g., 5569230 or [5569230, 1085604]
)
```
* **Behavior**: Case-insensitive matching, supports single values or collections.
* **Edge Case**: If no records match, returns an empty DataFrame preserving all original columns (no `KeyError` or crashes).

---

### 2.2 Weekly Trends & Time Series
```python
weekly_df = an.get_weekly_trends(
    filtered_df,
    include_returns=False,      # Exclude negative quantities and refund rows
    min_week=1,                 # Default: Week 1
    max_week=102,               # Default: Week 102
    promo_threshold_pct=25.0    # Causal promo trans % to flag is_promoted_week
)
```
* **Returns**: A gapless DataFrame of exactly 102 rows (Weeks 1 to 102) with the following schema:

| Column | Type | Description | Streamlit Widget |
| :--- | :--- | :--- | :--- |
| `WEEK_NO` | `int` | Sequential fiscal week (1 to 102) | X-axis for line/bar charts |
| `total_units` | `float` | Total units purchased | Primary volume metric |
| `total_sales` | `float` | Net sales revenue (\$) | Primary revenue metric |
| `purchasing_households`| `int` | Distinct active buyers that week | Customer penetration metric |
| `total_baskets` | `int` | Distinct shopping trips/baskets | Shopping frequency metric |
| `units_per_household` | `float` | Average volume per buyer | Basket intensity indicator |
| `spend_per_household` | `float` | Average spend per buyer (\$) | Customer spend indicator |
| `baskets_per_household`| `float` | Trips per active buyer | Shopping trip frequency |
| `avg_unit_price` | `float` | Realized price per unit (\$) | Price trajectory chart |
| `total_discount_val` | `float` | Total dollar savings from markdown/coupons | Discount scale |
| `avg_discount_depth_pct`| `float`| Average discount depth percentage | Discount intensity chart |
| `causal_promo_pct` | `float` | % of transactions with verified display/mailer | Promotion intensity |
| `is_promoted_week` | `bool` | True if `causal_promo_pct >= threshold` | Highlight / color overlay |
| `causal_coverage_status`| `str` | `'Recorded'`, `'Mixed'`, or `'Unknown'` | Data quality badge |

---

### 2.3 Promotion Episode Identification
```python
episodes_df = an.identify_promotion_episodes(
    weekly_df,
    min_promo_pct=50.0,   # Minimum weekly causal promo share to qualify
    pre_window=4,         # Baseline pre-window length in weeks (default: 4)
    post_window=4         # Post-promotion observation window in weeks (default: 4)
)
```
* **Behavior**: Coalesces consecutive promoted weeks into single episodes (e.g. Weeks 15 & 16 $\rightarrow$ 2-week episode).
* **Metadata Produced**:
  - `episode_id`: Unique identifier (e.g., `'EP_01'`, `'EP_02'`)
  - `start_week`, `end_week`, `duration_weeks`: Duration of the promotion event
  - `pre_start_week`, `pre_end_week`: Configurable pre-promotion baseline window
  - `post_start_week`, `post_end_week`: Configurable post-promotion observation window
  - `is_complete_window`: `bool` (`False` if pre-window < Week 1 or post-window > Week 102)
  - `has_overlapping_promo`: `bool` (`True` if another promotion week falls within pre/post windows)
  - `causal_coverage_valid`: `bool` (`True` if all windows fall within causal auditing: Weeks 9–101)
  - `during_avg_promo_pct`: Average promotional transaction share during the event

---

### 2.4 Before / During / After (BDA) Window Comparison
```python
bda_df = an.compare_before_during_after(
    filtered_df,
    episode_series,       # Row from episodes_df
    include_returns=False
)
```
* **Behavior**: Normalizes volume and sales to **per-week rates** to guarantee fair comparisons regardless of window duration.
* **Output Metrics**:
  - **Pre Baseline**: `pre_units_per_week`, `pre_sales_per_week`, `pre_hh_per_week`, `pre_units_per_hh`, `pre_spend_per_hh`
  - **During Event**: `during_units_per_week`, `during_sales_per_week`, `during_hh_per_week`, `during_units_per_hh`, `during_spend_per_hh`
  - **Post Window**: `post_units_per_week`, `post_sales_per_week`, `post_hh_per_week`, `post_units_per_hh`, `post_spend_per_hh`
  - **Lifts & Changes**:
    $$\text{Unit Lift \%} = \frac{\text{during\_units\_per\_week} - \text{pre\_units\_per\_week}}{\text{pre\_units\_per\_week}} \times 100$$
    $$\text{Sales Lift \%} = \frac{\text{during\_sales\_per\_week} - \text{pre\_sales\_per\_week}}{\text{pre\_sales\_per\_week}} \times 100$$
    $$\text{Post vs Pre Unit \%} = \frac{\text{post\_units\_per\_week} - \text{pre\_units\_per\_week}}{\text{pre\_units\_per\_week}} \times 100$$
    $$\text{Post vs During Dip \%} = \frac{\text{post\_units\_per\_week} - \text{during\_units\_per\_week}}{\text{during\_units\_per\_week}} \times 100$$

---

### 2.5 Household Behavior & Repurchase Timing (Native `DAY`)
```python
hh_summary_df = an.get_household_behavior_summary(
    filtered_df,
    episode_series,       # Row from episodes_df
    include_returns=False
)
```
* **Behavior**: Uses the exact transaction `DAY` to calculate inter-purchase intervals and segment buyer cohorts:
  - `promo_buyers_count`: Total distinct households purchasing during promotion
  - `repeat_buyers_count`: Promo buyers who repurchased in post-window
  - `post_repurchase_rate_pct`: Percentage of promo buyers who returned
  - `trial_buyers_count`: New buyers acquired (0 purchases in pre-window)
  - `trial_repeat_count`: Trial buyers who repurchased in post-window
  - `trial_retention_rate_pct`: Percentage of trial buyers retained
  - `avg_pre_interpurchase_days`: Average days between shopping trips in pre-window
  - `avg_post_interpurchase_days`: Average days between shopping trips in post-window
  - `interpurchase_day_change`: Change in inter-purchase cycle length ($\Delta \text{Days}$)

---

## 3. Methodological Principles & Data Handling Rules

1. **Handling Returns & Negative Records**:
   - `QUANTITY <= 0` or `SALES_VALUE < 0` (0.56% of records) are filtered out by default (`include_returns=False`) for volume, basket size, and customer frequency calculations.
   - For financial accounting, `include_returns=True` can be toggled to compute net sales.
2. **Handling Zero-Sales Positive-Quantity Transactions**:
   - `SALES_VALUE == 0` with `QUANTITY > 0` (0.17% of transactions, e.g. BOGO free goods) are included in unit volume and basket frequency, but contribute $0.00 to sales revenue.
3. **No Unwarranted Control Group Fallacy**:
   - As established in Step 1 and Step 2, `causal_data.csv` is an event log. Unmonitored product-store-weeks have **unknown** causal coverage.
   - **We strictly prohibit using unknown causal records as an untreated control group.** Baselines are evaluated against the historical pre-promotion performance of the monitored entity.
4. **Data-Driven Scientific Neutrality**:
   - In documentation and UI presentation, we report observed metrics (e.g. "observed post-promotion volume dip", "inter-purchase interval change") rather than asserting proven causal effects or definitive stockpiling.

---

## 4. Automated Test Suite Results

The comprehensive test suite in [test_analytics.py](file:///c:/Users/SiruiChen/Documents/GitHub/Visual-Analytics-Group-1/test_analytics.py) was executed:

```text
test_compare_before_during_after (test_analytics.TestAnalytics) ... ok
test_filter_data (test_analytics.TestAnalytics) ... ok
test_get_weekly_trends (test_analytics.TestAnalytics) ... ok
test_household_behavior_summary (test_analytics.TestAnalytics) ... ok
test_identify_promotion_episodes (test_analytics.TestAnalytics) ... ok
test_incomplete_and_overlapping_episodes (test_analytics.TestAnalytics) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.066s
OK
```

All 6 unit tests pass, verifying:
- Filtering on all 4 hierarchy levels.
- Graceful empty results handling.
- Gapless 102-week time-series generation.
- Correct episode duration and consecutive week coalescing.
- Incomplete window and overlapping promotion detection.
- Accurate percentage lift and BDA summary calculations.
- Cohort identification and inter-purchase day intervals using native `DAY`.

---

## 5. Real-Data Demonstration Case Study

### Selected Target: National Brand 12-Pack Can Soft Drinks (`PRODUCT_ID = 5569230`)
* **Commodity**: `SOFT DRINKS`
* **Department**: `GROCERY`
* **Total Transactions**: 4,834 across 114 stores

### A. Weekly Trends Around Promotion Event (Week 30)

| `WEEK_NO` | `total_units` | `total_sales` | `purchasing_households` | `units_per_household` | `causal_promo_pct` | `is_promoted_week` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **26** | 17.0 | \$78.54 | 15 | 1.13 | 5.88% | False (Baseline) |
| **27** | 23.0 | \$107.87 | 20 | 1.15 | 4.55% | False (Baseline) |
| **28** | 40.0 | \$146.76 | 29 | 1.38 | 3.45% | False (Baseline) |
| **29** | 26.0 | \$121.94 | 24 | 1.08 | 4.00% | False (Baseline) |
| **30** | **236.0** | **\$622.85** | **104** | **2.27** | **100.00%** | **True (Major Promotion Event)** |
| **31** | 25.0 | \$117.25 | 20 | 1.25 | 13.04% | False (Post-Promo Dip) |
| **32** | 22.0 | \$103.18 | 17 | 1.29 | 4.76% | False (Post-Promo Dip) |
| **33** | 31.0 | \$145.39 | 28 | 1.11 | 6.90% | False (Post-Promo Dip) |
| **34** | **320.0** | **\$943.54** | **125** | **2.56** | **100.00%** | **True (Second Promo Event)** |
| **35** | 28.0 | \$131.07 | 25 | 1.12 | 3.85% | False (Post-Promo Dip) |
| **36** | 23.0 | \$106.87 | 21 | 1.10 | 8.70% | False (Post-Promo Dip) |

### B. Identified Promotion Episode: `EP_04` (Week 30)
- **Start Week**: 30 | **End Week**: 30 | **Duration**: 1 week
- **Pre-Window**: Weeks 26 to 29 (4 weeks) | **Post-Window**: Weeks 31 to 34 (4 weeks)
- **Causal Promotion Intensity**: 100.0%

### C. Before / During / After Comparison
```text
Episode ID: EP_04 (Week 30)
- Pre-Promotion Baseline Rate:     26.5 units/week   ($113.78/week)
- During-Promotion Event Rate:    236.0 units/week   ($622.85/week)
- Post-Promotion Rate:             99.5 units/week   ($326.85/week)

Key Performance Lifts:
- Promotional Unit Lift:          +790.57%
- Promotional Sales Lift:         +447.43%
- Household Participation Lift:   +372.73%
- Post vs. During Volume Dip:      -57.84%
```

### D. Household Cohort Behavior & Repurchase Timing
```text
Household Behavior Summary (Episode EP_04):
- Promo Buyer Households:          104
- Post-Promotion Repeat Buyers:     40 (38.46% repeat purchase rate)
- New Trial Buyers Acquired:        81
- Retained Trial Buyers:            26 (32.10% trial retention rate)
- Pre-Promo Inter-Purchase Cycle:  7.10 days
- Post-Promo Inter-Purchase Cycle: 5.30 days
```

---

## 6. Sample Streamlit Integration Code for Teammate 2

Teammate 2 can build the October 20 Progress II presentation UI with just a few lines of code:

```python
import streamlit as st
import data_preparation as dp
import analytics as an

st.title("ITCS 5122 - Grocery Promotion & Stockpiling Explorer")

# 1. Load Data (cached)
@st.cache_data
def load_data():
    df_prod = dp.load_products()
    df_trans = dp.load_transactions()
    return dp.join_transactions_and_products(df_trans, df_prod)

df = load_data()

# 2. Sidebar Filters
commodity = st.sidebar.selectbox("Commodity", sorted(df["COMMODITY_DESC"].dropna().unique()))
filtered = an.filter_data(df, commodity_desc=commodity)

# 3. Weekly Trends
st.subheader(f"Weekly Volume & Promotion Trends: {commodity}")
weekly = an.get_weekly_trends(filtered)
st.line_chart(weekly.set_index("WEEK_NO")[["total_units", "purchasing_households"]])

# 4. Promotion Episodes
st.subheader("Observed Promotion Episodes")
episodes = an.identify_promotion_episodes(weekly)
st.dataframe(episodes[["episode_id", "start_week", "duration_weeks", "is_complete_window", "has_overlapping_promo"]])

# 5. BDA Summary for selected episode
if not episodes.empty:
    selected_ep_id = st.selectbox("Select Episode to Analyze", episodes["episode_id"])
    selected_ep = episodes[episodes["episode_id"] == selected_ep_id].iloc[0]
    
    bda_summary = an.compare_before_during_after(filtered, selected_ep)
    c1, c2, c3 = st.columns(3)
    c1.metric("Unit Lift", f"{bda_summary['unit_lift_pct'].iloc[0]:+.1f}%")
    c2.metric("Sales Lift", f"{bda_summary['sales_lift_pct'].iloc[0]:+.1f}%")
    c3.metric("Post vs During Dip", f"{bda_summary['post_vs_during_dip_pct'].iloc[0]:.1f}%")
```

---

### Step 3 Deliverables Status
- `analytics.py`: **Ready & Verified**
- `test_analytics.py`: **All 6 tests passing**
- `reports/analytics_validation.md`: **Documented & Complete**
- No raw CSV files modified, no Streamlit/AI code introduced.
