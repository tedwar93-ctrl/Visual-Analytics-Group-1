"""
analytics.py
------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Teammate 1: Data & Analytics Lead

Reusable analytics module for Progress II Streamlit MVP:
1. Product & category filtering (DEPARTMENT, COMMODITY_DESC, SUB_COMMODITY_DESC, PRODUCT_ID)
2. Weekly trend metrics (purchased units, distinct households, units/household, purchase frequency)
3. Promotion episode identification from observed causal evidence
4. Clean Before / During / After (BDA) promotion episode comparisons with overlap safeguards
5. Household purchase behavior, repurchase timing (using native DAY), and trial retention
6. Safe handling of incomplete windows, overlapping promotions, and missing causal data
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. FILTERING FUNCTIONS
# ==============================================================================

def filter_data(
    df,
    department=None,
    commodity_desc=None,
    sub_commodity_desc=None,
    product_id=None
):
    """
    Filters transactions by hierarchy levels using exact product.csv fields.
    Accepts single values (str/int) or collections (list/set/tuple).
    Returns a copy of the filtered DataFrame or an empty copy preserving columns.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df.copy()

    filtered = df

    def _match_col(series, filter_val):
        if filter_val is None:
            return pd.Series(True, index=series.index)
        if isinstance(filter_val, (list, tuple, set, np.ndarray)):
            vals = [str(v).strip().upper() for v in filter_val]
            return series.astype(str).str.strip().str.upper().isin(vals)
        else:
            val = str(filter_val).strip().upper()
            return series.astype(str).str.strip().str.upper() == val

    if department is not None:
        filtered = filtered[_match_col(filtered["DEPARTMENT"], department)]

    if commodity_desc is not None:
        filtered = filtered[_match_col(filtered["COMMODITY_DESC"], commodity_desc)]

    if sub_commodity_desc is not None:
        filtered = filtered[_match_col(filtered["SUB_COMMODITY_DESC"], sub_commodity_desc)]

    if product_id is not None:
        if isinstance(product_id, (list, tuple, set, np.ndarray)):
            pids = [int(p) for p in product_id]
            filtered = filtered[filtered["PRODUCT_ID"].isin(pids)]
        else:
            filtered = filtered[filtered["PRODUCT_ID"] == int(product_id)]

    return filtered.copy()


# ==============================================================================
# 2. WEEKLY TRENDS & AGGREGATIONS
# ==============================================================================

def get_weekly_trends(
    df,
    include_returns=False,
    min_week=1,
    max_week=102,
    promo_threshold_pct=25.0,
    min_coverage_pct=100.0
):
    """
    Calculates consolidated weekly trend metrics across native WEEK_NO (1 to 102).
    
    Metrics:
    - total_units: sum of positive QUANTITY (or net quantity if include_returns=True)
    - total_sales: sum of positive SALES_VALUE (or net sales if include_returns=True)
    - purchasing_households: count of distinct household_key
    - total_baskets: count of distinct BASKET_ID
    - units_per_household: total_units / purchasing_households
    - spend_per_household: total_sales / purchasing_households
    - baskets_per_household: total_baskets / purchasing_households (frequency)
    - avg_unit_price: total_sales / total_units
    - total_discount_val: sum of total_discount_val
    - avg_discount_depth_pct: weighted average discount depth
    - causal_promoted_trans_count: transactions with has_causal_evidence == True
    - causal_promo_pct: percentage of transactions with causal promotion (NaN if coverage unknown)
    - is_promoted_week: boolean indicating promotion active this week
    - causal_coverage_status: 'Recorded', 'Mixed', or 'Unknown'
    - promo_evaluation_status: 'Promoted', 'Unpromoted' (only when coverage adequate), or 'Unknown'
    - causal_coverage_pct: percentage of transactions with recorded causal tracking
    
    Returns a DataFrame sorted by WEEK_NO ready for Streamlit charts.
    """
    cols = [
        "WEEK_NO", "total_units", "total_sales", "purchasing_households",
        "total_baskets", "units_per_household", "spend_per_household",
        "baskets_per_household", "avg_unit_price", "total_discount_val",
        "avg_discount_depth_pct", "causal_promoted_trans_count",
        "total_transactions", "causal_promo_pct", "is_promoted_week",
        "causal_coverage_status", "promo_evaluation_status", "causal_coverage_pct"
    ]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)

    # Filter out returns / refunds unless explicitly requested
    if not include_returns:
        data = df[(df["QUANTITY"] > 0) & (df["SALES_VALUE"] >= 0)].copy()
    else:
        data = df.copy()

    if len(data) == 0:
        return pd.DataFrame(columns=cols)

    # Weekly group
    grouped = data.groupby("WEEK_NO")
    records = []
    all_weeks = range(min_week, max_week + 1)
    
    for week in all_weeks:
        if week in grouped.groups:
            w_df = grouped.get_group(week)
            n_trans = len(w_df)
            units = float(w_df["QUANTITY"].sum())
            sales = float(w_df["SALES_VALUE"].sum())
            hh = int(w_df["household_key"].nunique())
            baskets = int(w_df["BASKET_ID"].nunique())
            
            u_per_hh = (units / hh) if hh > 0 else 0.0
            s_per_hh = (sales / hh) if hh > 0 else 0.0
            b_per_hh = (baskets / hh) if hh > 0 else 0.0
            avg_price = (sales / units) if units > 0 else 0.0
            
            disc_val = float(w_df["total_discount_val"].sum()) if "total_discount_val" in w_df.columns else 0.0
            disc_pct = float(w_df["discount_depth_pct"].mean()) if "discount_depth_pct" in w_df.columns else 0.0
            
            # Causal coverage completeness
            if "causal_coverage_status" in w_df.columns:
                cov_set = set(w_df["causal_coverage_status"].dropna().unique())
                n_recorded = int((w_df["causal_coverage_status"] == "Recorded").sum())
                cov_pct = (n_recorded / n_trans * 100.0) if n_trans > 0 else 0.0
                if cov_set == {"Recorded"}:
                    cov_status = "Recorded"
                elif "Recorded" in cov_set:
                    cov_status = "Mixed"
                else:
                    if 9 <= week <= 101:
                        cov_status = "Recorded"
                        cov_pct = 100.0
                    else:
                        cov_status = "Unknown"
            else:
                if 9 <= week <= 101:
                    cov_status = "Recorded"
                    cov_pct = 100.0
                    n_recorded = n_trans
                else:
                    cov_status = "Unknown"
                    cov_pct = 0.0
                    n_recorded = 0

            # Causal promotion rate calculation
            if "has_causal_evidence" in w_df.columns:
                if cov_status == "Mixed":
                    # If recorded transactions include both promoted and unpromoted items (audited subset)
                    rec_mask = (w_df["causal_coverage_status"] == "Recorded")
                    causal_count = int(w_df.loc[rec_mask, "has_causal_evidence"].sum())
                    if n_recorded > causal_count:
                        promo_pct = (causal_count / n_recorded * 100.0) if n_recorded > 0 else np.nan
                    else:
                        # Dunnhumby causal records represent promotional event logs (only active displays/mailers logged).
                        # Unfeatured items in audited weeks have no causal record; denominator is total transactions.
                        promo_pct = (causal_count / n_trans * 100.0) if n_trans > 0 else np.nan
                elif cov_status == "Recorded":
                    causal_count = int(w_df["has_causal_evidence"].sum())
                    promo_pct = (causal_count / n_trans * 100.0) if n_trans > 0 else np.nan
                else:
                    causal_count = 0
                    promo_pct = np.nan
            else:
                causal_count = 0
                promo_pct = np.nan if cov_status == "Unknown" else 0.0

            is_promo = bool(promo_pct >= promo_threshold_pct) if (promo_pct is not None and not np.isnan(promo_pct)) else False

            # Evaluation status distinguishing Promoted, Unpromoted (adequate coverage), and Unknown
            if is_promo:
                promo_eval = "Promoted"
            elif cov_status == "Unknown":
                promo_eval = "Unknown"
            elif cov_status == "Mixed":
                if n_recorded > causal_count:
                    promo_eval = "Unpromoted" if cov_pct >= min_coverage_pct else "Unknown"
                else:
                    # In Dunnhumby event-log data, absence of causal record indicates unfeatured/unpromoted
                    promo_eval = "Unpromoted"
            else:
                promo_eval = "Unpromoted"

            records.append({
                "WEEK_NO": week,
                "total_units": units,
                "total_sales": sales,
                "purchasing_households": hh,
                "total_baskets": baskets,
                "units_per_household": u_per_hh,
                "spend_per_household": s_per_hh,
                "baskets_per_household": b_per_hh,
                "avg_unit_price": avg_price,
                "total_discount_val": disc_val,
                "avg_discount_depth_pct": disc_pct,
                "causal_promoted_trans_count": causal_count,
                "total_transactions": n_trans,
                "causal_promo_pct": promo_pct,
                "is_promoted_week": is_promo,
                "causal_coverage_status": cov_status,
                "promo_evaluation_status": promo_eval,
                "causal_coverage_pct": cov_pct
            })
        else:
            cov_status = "Recorded" if (9 <= week <= 101) else "Unknown"
            cov_pct = 100.0 if (9 <= week <= 101) else 0.0
            records.append({
                "WEEK_NO": week,
                "total_units": 0.0,
                "total_sales": 0.0,
                "purchasing_households": 0,
                "total_baskets": 0,
                "units_per_household": 0.0,
                "spend_per_household": 0.0,
                "baskets_per_household": 0.0,
                "avg_unit_price": 0.0,
                "total_discount_val": 0.0,
                "avg_discount_depth_pct": 0.0,
                "causal_promoted_trans_count": 0,
                "total_transactions": 0,
                "causal_promo_pct": np.nan,
                "is_promoted_week": False,
                "causal_coverage_status": cov_status,
                "promo_evaluation_status": "Unknown",
                "causal_coverage_pct": cov_pct
            })

    return pd.DataFrame(records)


# ==============================================================================
# 3. PROMOTION EPISODE IDENTIFICATION (WITH CONTAMINATION SAFEGUARDS)
# ==============================================================================

def identify_promotion_episodes(
    weekly_trends_df,
    min_promo_pct=25.0,
    pre_window=4,
    post_window=4,
    enforce_causal_boundaries=False
):
    """
    Identifies promotion episodes by grouping consecutive promotion weeks.
    
    A week is treated as promoted if causal_promo_pct >= min_promo_pct.
    Consecutive promoted weeks are coalesced into a single multi-week episode.
    
    Contamination Safeguards:
    - Pre-window is truncated to start AFTER any preceding promotion ends.
    - Post-window is truncated to end BEFORE any subsequent promotion starts.
    - Boundary checks enforce Week 1 and Week 102 dataset limits.
    - When enforce_causal_boundaries=True, pre/post windows are additionally truncated
      to remain strictly within the verified causal tracking window (Weeks 9-101).
    - Reports both nominal and actual clean week counts, with explicit truncation reasons.
    
    Returns a DataFrame of candidate promotion episodes.
    """
    cols = [
        "episode_id", "start_week", "end_week", "duration_weeks",
        "pre_window_nominal", "post_window_nominal",
        "pre_start_week_nominal", "pre_end_week",
        "post_start_week", "post_end_week_nominal",
        "pre_start_week_clean", "post_end_week_clean",
        "pre_actual_weeks", "post_actual_weeks",
        "pre_is_truncated", "post_is_truncated",
        "pre_truncation_reason", "post_truncation_reason",
        "is_complete_window", "has_overlapping_promo",
        "causal_coverage_valid", "during_avg_promo_pct",
        "enforce_causal_boundaries"
    ]
    if weekly_trends_df is None or len(weekly_trends_df) == 0:
        return pd.DataFrame(columns=cols)

    df_sorted = weekly_trends_df.sort_values("WEEK_NO").copy()
    promoted_weeks = set(df_sorted[df_sorted["causal_promo_pct"] >= min_promo_pct]["WEEK_NO"])

    if not promoted_weeks:
        return pd.DataFrame(columns=cols)

    # Group consecutive weeks into raw episodes
    sorted_promo_weeks = sorted(list(promoted_weeks))
    raw_episodes = []
    curr_ep = [sorted_promo_weeks[0]]

    for w in sorted_promo_weeks[1:]:
        if w == curr_ep[-1] + 1:
            curr_ep.append(w)
        else:
            raw_episodes.append(curr_ep)
            curr_ep = [w]
    raw_episodes.append(curr_ep)

    episodes = []
    num_episodes = len(raw_episodes)

    for idx, ep_weeks in enumerate(raw_episodes, start=1):
        s_wk = ep_weeks[0]
        e_wk = ep_weeks[-1]
        dur = len(ep_weeks)
        
        # Nominal windows
        pre_s_nom = s_wk - pre_window
        pre_e = s_wk - 1
        post_s = e_wk + 1
        post_e_nom = e_wk + post_window

        # Check prior promotion for pre-window contamination
        pre_is_trunc = False
        pre_trunc_reason = "None"
        pre_s_clean = pre_s_nom

        if idx > 1:
            prior_end = raw_episodes[idx - 2][-1]
            if prior_end >= pre_s_nom:
                pre_s_clean = prior_end + 1
                pre_is_trunc = True
                pre_trunc_reason = f"Prior promotion at Week {prior_end}"

        if enforce_causal_boundaries and pre_s_clean < 9:
            pre_s_clean = 9
            pre_is_trunc = True
            pre_trunc_reason = "Causal coverage start (Week 9)"
        elif pre_s_clean < 1:
            pre_s_clean = 1
            pre_is_trunc = True
            pre_trunc_reason = "Dataset start (Week 1)"

        # Check subsequent promotion for post-window contamination
        post_is_trunc = False
        post_trunc_reason = "None"
        post_e_clean = post_e_nom

        if idx < num_episodes:
            next_start = raw_episodes[idx][0]
            if next_start <= post_e_nom:
                post_e_clean = next_start - 1
                post_is_trunc = True
                post_trunc_reason = f"Subsequent promotion at Week {next_start}"

        if enforce_causal_boundaries and post_e_clean > 101:
            post_e_clean = 101
            post_is_trunc = True
            post_trunc_reason = "Causal coverage end (Week 101)"
        elif post_e_clean > 102:
            post_e_clean = 102
            post_is_trunc = True
            post_trunc_reason = "Dataset end (Week 102)"

        # Actual week counts
        pre_act_wks = max(0, pre_e - pre_s_clean + 1)
        post_act_wks = max(0, post_e_clean - post_s + 1)

        is_comp_pre = (pre_act_wks == pre_window)
        is_comp_post = (post_act_wks == post_window)
        is_comp_window = is_comp_pre and is_comp_post

        has_ov = pre_is_trunc or post_is_trunc
        causal_valid = (s_wk >= 9) and (e_wk <= 101) and (pre_s_clean >= 9) and (post_e_clean <= 101) and (pre_act_wks > 0 or pre_window == 0) and (post_act_wks > 0 or post_window == 0)

        # Average promo intensity during episode
        ep_slice = df_sorted[df_sorted["WEEK_NO"].isin(ep_weeks)]
        avg_intensity = float(ep_slice["causal_promo_pct"].mean())

        episodes.append({
            "episode_id": f"EP_{idx:02d}",
            "start_week": s_wk,
            "end_week": e_wk,
            "duration_weeks": dur,
            "pre_window_nominal": pre_window,
            "post_window_nominal": post_window,
            "pre_start_week_nominal": pre_s_nom,
            "pre_end_week": pre_e,
            "post_start_week": post_s,
            "post_end_week_nominal": post_e_nom,
            "pre_start_week_clean": pre_s_clean,
            "post_end_week_clean": post_e_clean,
            "pre_actual_weeks": pre_act_wks,
            "post_actual_weeks": post_act_wks,
            "pre_is_truncated": pre_is_trunc,
            "post_is_truncated": post_is_trunc,
            "pre_truncation_reason": pre_trunc_reason,
            "post_truncation_reason": post_trunc_reason,
            "is_complete_window": is_comp_window,
            "has_overlapping_promo": has_ov,
            "causal_coverage_valid": causal_valid,
            "during_avg_promo_pct": avg_intensity,
            "enforce_causal_boundaries": enforce_causal_boundaries
        })

    return pd.DataFrame(episodes)


# ==============================================================================
# 4. BEFORE / DURING / AFTER (BDA) COMPARISON (CLEAN WINDOW SAFEGUARDS)
# ==============================================================================

def compare_before_during_after(
    df,
    episode,
    use_clean_windows=True,
    include_returns=False,
    enforce_causal_boundaries=False
):
    """
    Computes Before, During, and After summary metrics for a given promotion episode.
    
    When use_clean_windows=True (default):
    - Uses pre_start_week_clean and post_end_week_clean to ensure NO subsequent or prior
      promotion contaminates the comparison windows.
    - Normalizes volume and sales by the ACTUAL clean unpromoted week count.
    
    Reports:
    - pre_actual_weeks, post_actual_weeks
    - pre_is_truncated, post_is_truncated
    - Promotional Lift % (During vs Pre)
    - Post-Promotion Persistence / Dip % (Post vs Pre, and Post vs During)
    - is_clean_comparison: True if both pre and post have >= 1 clean unpromoted week
    - causal_coverage_valid: True if all evaluated weeks fall strictly in Weeks 9-101
    - has_reliable_baseline: True if baseline is clean, positive, and within verified causal coverage
    - coverage_warning: Human-readable warning if comparison contains unknown causal coverage
    
    Returns a single-row DataFrame ready for Streamlit KPI cards / summary tables.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    dur_s = int(episode["start_week"])
    dur_e = int(episode["end_week"])
    dur_wks = int(episode["duration_weeks"])

    if use_clean_windows:
        pre_s = int(episode.get("pre_start_week_clean", episode.get("pre_start_week_nominal", dur_s - 4)))
        post_e = int(episode.get("post_end_week_clean", episode.get("post_end_week_nominal", dur_e + 4)))
        pre_wks = int(episode.get("pre_actual_weeks", max(0, (dur_s - 1) - pre_s + 1)))
        post_wks = int(episode.get("post_actual_weeks", max(0, post_e - (dur_e + 1) + 1)))
    else:
        pre_s = int(episode.get("pre_start_week_nominal", dur_s - 4))
        post_e = int(episode.get("post_end_week_nominal", dur_e + 4))
        pre_wks = max(1, dur_s - max(1, pre_s))
        post_wks = max(1, min(102, post_e) - dur_e)

    if enforce_causal_boundaries:
        if pre_s < 9:
            pre_s = 9
            pre_wks = max(0, (dur_s - 1) - pre_s + 1)
        if post_e > 101:
            post_e = 101
            post_wks = max(0, post_e - (dur_e + 1) + 1)

    pre_e = dur_s - 1
    post_s = dur_e + 1

    # Filter data for non-returns
    if not include_returns:
        data = df[(df["QUANTITY"] > 0) & (df["SALES_VALUE"] >= 0)].copy()
    else:
        data = df.copy()

    # Slice windows
    pre_df = data[(data["WEEK_NO"] >= pre_s) & (data["WEEK_NO"] <= pre_e)] if pre_wks > 0 else pd.DataFrame(columns=data.columns)
    dur_df = data[(data["WEEK_NO"] >= dur_s) & (data["WEEK_NO"] <= dur_e)]
    post_df = data[(data["WEEK_NO"] >= post_s) & (data["WEEK_NO"] <= post_e)] if post_wks > 0 else pd.DataFrame(columns=data.columns)

    # Coverage inspection
    pre_has_unknown = (pre_s < 9)
    post_has_unknown = (post_e > 101)
    if len(pre_df) > 0 and "causal_coverage_status" in pre_df.columns:
        if ((pre_df["causal_coverage_status"] == "Unknown") & (pre_df["WEEK_NO"] < 9)).any():
            pre_has_unknown = True
    if len(post_df) > 0 and "causal_coverage_status" in post_df.columns:
        if ((post_df["causal_coverage_status"] == "Unknown") & (post_df["WEEK_NO"] > 101)).any():
            post_has_unknown = True

    episode_causal_valid = bool(episode.get("causal_coverage_valid", True))
    causal_valid = episode_causal_valid and (not pre_has_unknown) and (not post_has_unknown) and (dur_s >= 9) and (dur_e <= 101)

    if pre_has_unknown and post_has_unknown:
        cov_warning = "Both pre-baseline (Weeks < 9) and post-window (Weeks > 101) contain unknown causal coverage."
    elif pre_has_unknown:
        cov_warning = "Pre-promotion baseline includes weeks prior to Week 9 with unknown causal coverage."
    elif post_has_unknown:
        cov_warning = "Post-promotion window includes weeks beyond Week 101 with unknown causal coverage."
    elif dur_s < 9 or dur_e > 101:
        cov_warning = "Promotion episode is outside verified causal coverage period (Weeks 9-101)."
    elif not episode_causal_valid:
        cov_warning = "Episode was flagged with invalid causal coverage."
    elif pre_wks == 0:
        cov_warning = "Pre-promotion baseline has 0 observation weeks."
    elif post_wks == 0:
        cov_warning = "Post-promotion window has 0 observation weeks."
    else:
        cov_warning = "None"

    is_reliable_baseline = bool(causal_valid and pre_wks > 0)
    is_clean = (pre_wks > 0) and (post_wks > 0)

    def _metrics(sub_df, num_weeks):
        if num_weeks == 0 or len(sub_df) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0
        tot_u = float(sub_df["QUANTITY"].sum())
        tot_s = float(sub_df["SALES_VALUE"].sum())
        hh = int(sub_df["household_key"].nunique())
        u_per_wk = tot_u / num_weeks
        s_per_wk = tot_s / num_weeks
        hh_per_wk = hh / num_weeks
        u_per_hh = tot_u / hh if hh > 0 else np.nan
        s_per_hh = tot_s / hh if hh > 0 else np.nan
        return u_per_wk, s_per_wk, hh_per_wk, u_per_hh, s_per_hh, hh, tot_u

    pre_u_wk, pre_s_wk, pre_hh_wk, pre_u_hh, pre_s_hh, pre_hh, pre_tot_u = _metrics(pre_df, pre_wks)
    dur_u_wk, dur_s_wk, dur_hh_wk, dur_u_hh, dur_s_hh, dur_hh, dur_tot_u = _metrics(dur_df, dur_wks)
    post_u_wk, post_s_wk, post_hh_wk, post_u_hh, post_s_hh, post_hh, post_tot_u = _metrics(post_df, post_wks)

    def _pct_change(new_val, base_val):
        if pd.isna(base_val) or pd.isna(new_val) or base_val == 0.0:
            return np.nan
        return ((new_val - base_val) / base_val) * 100.0

    unit_lift_pct = _pct_change(dur_u_wk, pre_u_wk) if (pre_wks > 0 and len(pre_df) > 0) else np.nan
    sales_lift_pct = _pct_change(dur_s_wk, pre_s_wk) if (pre_wks > 0 and len(pre_df) > 0) else np.nan
    hh_lift_pct = _pct_change(dur_hh_wk, pre_hh_wk) if (pre_wks > 0 and len(pre_df) > 0) else np.nan
    post_vs_pre_unit_pct = _pct_change(post_u_wk, pre_u_wk) if (pre_wks > 0 and post_wks > 0 and len(pre_df) > 0 and len(post_df) > 0) else np.nan
    post_vs_during_dip_pct = _pct_change(post_u_wk, dur_u_wk) if (post_wks > 0 and len(post_df) > 0 and len(dur_df) > 0) else np.nan

    result = {
        "episode_id": episode.get("episode_id", "EP_01"),
        "start_week": dur_s,
        "end_week": dur_e,
        "duration_weeks": dur_wks,
        "pre_actual_weeks": pre_wks,
        "post_actual_weeks": post_wks,
        "pre_is_truncated": episode.get("pre_is_truncated", False),
        "post_is_truncated": episode.get("post_is_truncated", False),
        "pre_truncation_reason": episode.get("pre_truncation_reason", "None"),
        "post_truncation_reason": episode.get("post_truncation_reason", "None"),
        "is_clean_comparison": is_clean,
        "is_complete_window": episode.get("is_complete_window", True),
        "causal_coverage_valid": causal_valid,
        "has_reliable_baseline": is_reliable_baseline,
        "coverage_warning": cov_warning,
        # Pre metrics
        "pre_units_per_week": pre_u_wk,
        "pre_sales_per_week": pre_s_wk,
        "pre_hh_per_week": pre_hh_wk,
        "pre_units_per_hh": pre_u_hh,
        "pre_spend_per_hh": pre_s_hh,
        "pre_total_households": pre_hh,
        # During metrics
        "during_units_per_week": dur_u_wk,
        "during_sales_per_week": dur_s_wk,
        "during_hh_per_week": dur_hh_wk,
        "during_units_per_hh": dur_u_hh,
        "during_spend_per_hh": dur_s_hh,
        "during_total_households": dur_hh,
        # Post metrics
        "post_units_per_week": post_u_wk,
        "post_sales_per_week": post_s_wk,
        "post_hh_per_week": post_hh_wk,
        "post_units_per_hh": post_u_hh,
        "post_spend_per_hh": post_s_hh,
        "post_total_households": post_hh,
        # Lift and Post changes
        "unit_lift_pct": unit_lift_pct,
        "sales_lift_pct": sales_lift_pct,
        "hh_lift_pct": hh_lift_pct,
        "post_vs_pre_unit_pct": post_vs_pre_unit_pct,
        "post_vs_during_dip_pct": post_vs_during_dip_pct
    }
    return pd.DataFrame([result])


# ==============================================================================
# 5. HOUSEHOLD BEHAVIOR & REPURCHASE TIMING (NATIVE DAY & CLEAN WINDOWS)
# ==============================================================================

def get_household_behavior_summary(
    df,
    episode,
    use_clean_windows=True,
    include_returns=False,
    enforce_causal_boundaries=False
):
    """
    Analyzes household cohort behavior and repurchase timing using native DAY.
    
    Uses clean windows to prevent subsequent promotions from artificially inflating repeat buyer counts.
    
    Metrics:
    - promo_buyers_count: distinct purchasing households during promotion
    - repeat_buyers_count: promo buyers who returned in clean post-window
    - post_repurchase_rate_pct: repeat_buyers_count / promo_buyers_count * 100
    - trial_buyers_count: promo buyers who had 0 purchases in pre-window
    - trial_repeat_count: trial buyers who returned in clean post-window
    - trial_retention_rate_pct: trial_repeat_count / trial_buyers_count * 100
    - avg_days_to_first_post_purchase: mean days from promo purchase to first post-promo return (DAY)
    - avg_pre_interpurchase_days: mean days between repeat trips in pre-window (DAY)
    - avg_post_interpurchase_days: mean days between repeat trips in post-window (DAY)
    - interpurchase_day_change: post_days - pre_days
      (Negative value indicates faster / shorter repeat cycle, NOT delayed repurchase).
    - causal_coverage_valid: boolean indicating if evaluated windows fall in Weeks 9-101
    - has_reliable_baseline: boolean indicating reliable pre-promotion baseline
    - coverage_warning: warning message if coverage is unmonitored or insufficient
    
    Returns a single-row DataFrame.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    dur_s = int(episode["start_week"])
    dur_e = int(episode["end_week"])

    if use_clean_windows:
        pre_s = int(episode.get("pre_start_week_clean", episode["pre_start_week_nominal"]))
        post_e = int(episode.get("post_end_week_clean", episode["post_end_week_nominal"]))
        pre_wks = int(episode.get("pre_actual_weeks", max(0, dur_s - pre_s)))
        post_wks = int(episode.get("post_actual_weeks", max(0, post_e - dur_e)))
    else:
        pre_s = int(episode["pre_start_week_nominal"])
        post_e = int(episode["post_end_week_nominal"])
        pre_wks = max(1, dur_s - max(1, pre_s))
        post_wks = max(1, min(102, post_e) - dur_e)

    if enforce_causal_boundaries:
        if pre_s < 9:
            pre_s = 9
            pre_wks = max(0, (dur_s - 1) - pre_s + 1)
        if post_e > 101:
            post_e = 101
            post_wks = max(0, post_e - (dur_e + 1) + 1)

    pre_e = dur_s - 1
    post_s = dur_e + 1

    if not include_returns:
        data = df[(df["QUANTITY"] > 0) & (df["SALES_VALUE"] >= 0)].copy()
    else:
        data = df.copy()

    pre_data = data[(data["WEEK_NO"] >= pre_s) & (data["WEEK_NO"] <= pre_e)] if pre_wks > 0 else pd.DataFrame(columns=data.columns)
    dur_data = data[(data["WEEK_NO"] >= dur_s) & (data["WEEK_NO"] <= dur_e)]
    post_data = data[(data["WEEK_NO"] >= post_s) & (data["WEEK_NO"] <= post_e)] if post_wks > 0 else pd.DataFrame(columns=data.columns)

    # Coverage inspection
    pre_has_unknown = (pre_s < 9)
    post_has_unknown = (post_e > 101)
    if len(pre_data) > 0 and "causal_coverage_status" in pre_data.columns:
        if ((pre_data["causal_coverage_status"] == "Unknown") & (pre_data["WEEK_NO"] < 9)).any():
            pre_has_unknown = True
    if len(post_data) > 0 and "causal_coverage_status" in post_data.columns:
        if ((post_data["causal_coverage_status"] == "Unknown") & (post_data["WEEK_NO"] > 101)).any():
            post_has_unknown = True

    episode_causal_valid = bool(episode.get("causal_coverage_valid", True))
    causal_valid = episode_causal_valid and (not pre_has_unknown) and (not post_has_unknown) and (dur_s >= 9) and (dur_e <= 101)

    if pre_has_unknown and post_has_unknown:
        cov_warning = "Both pre-baseline (Weeks < 9) and post-window (Weeks > 101) contain unknown causal coverage."
    elif pre_has_unknown:
        cov_warning = "Pre-promotion baseline includes weeks prior to Week 9 with unknown causal coverage."
    elif post_has_unknown:
        cov_warning = "Post-promotion window includes weeks beyond Week 101 with unknown causal coverage."
    elif dur_s < 9 or dur_e > 101:
        cov_warning = "Promotion episode is outside verified causal coverage period (Weeks 9-101)."
    elif not episode_causal_valid:
        cov_warning = "Episode was flagged with invalid causal coverage."
    else:
        cov_warning = "None"

    is_reliable_baseline = bool(causal_valid and pre_wks > 0)

    pre_hh_set = set(pre_data["household_key"].unique())
    dur_hh_set = set(dur_data["household_key"].unique())
    post_hh_set = set(post_data["household_key"].unique())

    promo_buyers_cnt = len(dur_hh_set)
    if promo_buyers_cnt == 0:
        return pd.DataFrame([{
            "episode_id": episode.get("episode_id", "EP_01"),
            "promo_buyers_count": 0,
            "repeat_buyers_count": 0,
            "post_repurchase_rate_pct": 0.0,
            "trial_buyers_count": 0,
            "trial_repeat_count": 0,
            "trial_retention_rate_pct": 0.0,
            "avg_days_to_first_post_purchase": np.nan,
            "days_to_return_sample_size": 0,
            "avg_pre_interpurchase_days": np.nan,
            "pre_cycle_sample_size": 0,
            "avg_post_interpurchase_days": np.nan,
            "post_cycle_sample_size": 0,
            "interpurchase_day_change": np.nan,
            "causal_coverage_valid": causal_valid,
            "has_reliable_baseline": is_reliable_baseline,
            "coverage_warning": cov_warning
        }])

    repeat_hh = dur_hh_set.intersection(post_hh_set)
    repeat_buyers_cnt = len(repeat_hh)
    repurchase_rate = (repeat_buyers_cnt / promo_buyers_cnt * 100.0) if promo_buyers_cnt > 0 else 0.0

    trial_hh = dur_hh_set - pre_hh_set
    trial_buyers_cnt = len(trial_hh)
    trial_repeat_hh = trial_hh.intersection(post_hh_set)
    trial_repeat_cnt = len(trial_repeat_hh)
    trial_retention_rate = (trial_repeat_cnt / trial_buyers_cnt * 100.0) if trial_buyers_cnt > 0 else 0.0

    # 1. Days to First Post-Promotion Repurchase (DAY from promo to first post-purchase)
    if repeat_buyers_cnt > 0:
        dur_last_day = dur_data[dur_data["household_key"].isin(repeat_hh)].groupby("household_key")["DAY"].max()
        post_first_day = post_data[post_data["household_key"].isin(repeat_hh)].groupby("household_key")["DAY"].min()
        days_to_return_series = (post_first_day - dur_last_day).dropna()
        avg_days_to_first_post = float(days_to_return_series.mean()) if len(days_to_return_series) > 0 else np.nan
        days_to_return_n = len(days_to_return_series)
    else:
        avg_days_to_first_post = np.nan
        days_to_return_n = 0

    # 2. Intra-window repeat intervals (for households with >= 2 purchases within that window)
    def _calc_intra_window_interval(sub_df, target_hh_set):
        matched = sub_df[sub_df["household_key"].isin(target_hh_set)][["household_key", "DAY"]].drop_duplicates()
        if len(matched) == 0:
            return np.nan, 0
        matched = matched.sort_values(["household_key", "DAY"])
        matched["prev_day"] = matched.groupby("household_key")["DAY"].shift(1)
        matched["diff_days"] = matched["DAY"] - matched["prev_day"]
        valid_diffs = matched["diff_days"].dropna()
        sample_hh = matched[matched["diff_days"].notnull()]["household_key"].nunique()
        return (float(valid_diffs.mean()) if len(valid_diffs) > 0 else np.nan), sample_hh

    pre_interval, pre_n = _calc_intra_window_interval(pre_data, dur_hh_set)
    post_interval, post_n = _calc_intra_window_interval(post_data, dur_hh_set)
    interval_change = (post_interval - pre_interval) if (not np.isnan(pre_interval) and not np.isnan(post_interval)) else np.nan

    result = {
        "episode_id": episode.get("episode_id", "EP_01"),
        "promo_buyers_count": promo_buyers_cnt,
        "repeat_buyers_count": repeat_buyers_cnt,
        "post_repurchase_rate_pct": repurchase_rate,
        "trial_buyers_count": trial_buyers_cnt,
        "trial_repeat_count": trial_repeat_cnt,
        "trial_retention_rate_pct": trial_retention_rate,
        "avg_days_to_first_post_purchase": avg_days_to_first_post,
        "days_to_return_sample_size": days_to_return_n,
        "avg_pre_interpurchase_days": pre_interval,
        "pre_cycle_sample_size": pre_n,
        "avg_post_interpurchase_days": post_interval,
        "post_cycle_sample_size": post_n,
        "interpurchase_day_change": interval_change,
        "causal_coverage_valid": causal_valid,
        "has_reliable_baseline": is_reliable_baseline,
        "coverage_warning": cov_warning
    }
    return pd.DataFrame([result])
