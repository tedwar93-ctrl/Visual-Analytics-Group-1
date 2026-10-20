"""
test_analytics.py
-----------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Unit and regression tests for analytics.py

Tests:
1. Multi-level filtering by DEPARTMENT, COMMODITY_DESC, SUB_COMMODITY_DESC, PRODUCT_ID.
2. Handling of empty filter results and empty DataFrames.
3. Weekly trend metric calculations (units, households, frequency, units/hh, pricing).
4. Promotion episode identification and consecutive week grouping.
5. Incomplete window detection, boundary checks, and truncation safeguards.
6. Clean Before/During/After metric calculations excluding subsequent and prior promotions.
7. Household behavior, Days to First Post-Purchase (DAY), and trial retention.
"""

import unittest
import numpy as np
import pandas as pd
import analytics as an


class TestAnalytics(unittest.TestCase):

    def setUp(self):
        # Create a synthetic dataset representing products and transactions over 20 weeks
        self.mock_products = pd.DataFrame({
            "PRODUCT_ID": [101, 102, 103],
            "DEPARTMENT": ["GROCERY", "GROCERY", "PRODUCE"],
            "COMMODITY_DESC": ["SOFT DRINKS", "SOFT DRINKS", "POTATOES"],
            "SUB_COMMODITY_DESC": ["COLA CAN", "DIET COLA", "RUSSET"],
            "BRAND": ["National", "National", "Private"]
        })

        # Synthetic transactions
        # Promo 1 at Week 10 (day 64)
        # Promo 2 at Week 14 (day 92)
        trans_records = []
        
        # Baseline Weeks 6-9 for HH 1, 2
        for wk in range(6, 10):
            day = (wk - 1) * 7 + 1
            trans_records.append({
                "household_key": 1, "BASKET_ID": 1000 + wk, "DAY": day,
                "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 3.00,
                "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
            trans_records.append({
                "household_key": 2, "BASKET_ID": 2000 + wk, "DAY": day + 2,
                "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 3.00,
                "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })

        # During Week 10 (Promo 1, day 64): HH 1, 2, and new trial HH 3 buy in bulk
        day = (10 - 1) * 7 + 1
        for hh_id, qty, val in [(1, 3, 6.00), (2, 2, 4.00), (3, 4, 8.00)]:
            trans_records.append({
                "household_key": hh_id, "BASKET_ID": 3000 + hh_id, "DAY": day,
                "PRODUCT_ID": 101, "QUANTITY": qty, "SALES_VALUE": val,
                "STORE_ID": 501, "WEEK_NO": 10, "has_causal_evidence": True,
                "total_discount_val": qty * 1.0, "discount_depth_pct": 33.3,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })

        # Post Weeks 11-13 (Unpromoted clean post-window):
        # HH 1 repurchases at Week 12 (day 78), HH 3 does not repurchase in clean post-window
        trans_records.append({
            "household_key": 1, "BASKET_ID": 4001, "DAY": 78,
            "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 3.00,
            "STORE_ID": 501, "WEEK_NO": 12, "has_causal_evidence": False,
            "total_discount_val": 0.0, "discount_depth_pct": 0.0,
            "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        })

        # Week 14: SUBSEQUENT PROMOTION (Promo 2, day 92)
        # Without clean window safeguards, Week 14 would contaminate Week 10 post-window!
        trans_records.append({
            "household_key": 3, "BASKET_ID": 5001, "DAY": 92,
            "PRODUCT_ID": 101, "QUANTITY": 5, "SALES_VALUE": 10.00,
            "STORE_ID": 501, "WEEK_NO": 14, "has_causal_evidence": True,
            "total_discount_val": 5.0, "discount_depth_pct": 33.3,
            "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        })

        # Add return / non-positive transaction
        trans_records.append({
            "household_key": 99, "BASKET_ID": 9999, "DAY": 10,
            "PRODUCT_ID": 101, "QUANTITY": -1, "SALES_VALUE": -3.00,
            "STORE_ID": 501, "WEEK_NO": 2, "has_causal_evidence": False,
            "total_discount_val": 0.0, "discount_depth_pct": 0.0,
            "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        })

        self.mock_df = pd.DataFrame(trans_records)

    def test_filter_data(self):
        """Tests multi-level filtering and edge-case handling."""
        res_dep = an.filter_data(self.mock_df, department="GROCERY")
        self.assertEqual(len(res_dep), len(self.mock_df))

        res_comm = an.filter_data(self.mock_df, commodity_desc="SOFT DRINKS")
        self.assertEqual(len(res_comm), len(self.mock_df))

        res_pid = an.filter_data(self.mock_df, product_id=101)
        self.assertEqual(len(res_pid), len(self.mock_df))

        res_empty = an.filter_data(self.mock_df, commodity_desc="NON_EXISTENT")
        self.assertEqual(len(res_empty), 0)
        self.assertIn("PRODUCT_ID", res_empty.columns)

    def test_get_weekly_trends(self):
        """Tests weekly trend calculation, gapless time-series, and return exclusion."""
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        self.assertEqual(len(trends), 20)

        # Week 2 had return only -> 0 units
        w2 = trends[trends["WEEK_NO"] == 2].iloc[0]
        self.assertEqual(w2["total_units"], 0.0)

        # Week 10: Promo 1 (units=9, promo_pct=100)
        w10 = trends[trends["WEEK_NO"] == 10].iloc[0]
        self.assertEqual(w10["total_units"], 9.0)
        self.assertTrue(w10["is_promoted_week"])

        # Week 14: Promo 2 (units=5, promo_pct=100)
        w14 = trends[trends["WEEK_NO"] == 14].iloc[0]
        self.assertEqual(w14["total_units"], 5.0)
        self.assertTrue(w14["is_promoted_week"])

    def test_overlapping_promotion_truncates_post_window(self):
        """
        Regression Test: Verifies that Episode 1 (Week 10) post-window does NOT include
        subsequent promotion at Week 14.
        """
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        
        self.assertEqual(len(episodes), 2)
        ep1 = episodes.iloc[0]
        self.assertEqual(ep1["start_week"], 10)
        self.assertEqual(ep1["end_week"], 10)
        
        # Nominal post-window is 11 to 14 (4 weeks)
        self.assertEqual(ep1["post_window_nominal"], 4)
        self.assertEqual(ep1["post_end_week_nominal"], 14)
        
        # Clean post-window must truncate to Week 13 (3 weeks) to avoid Week 14 promo!
        self.assertEqual(ep1["post_end_week_clean"], 13)
        self.assertEqual(ep1["post_actual_weeks"], 3)
        self.assertTrue(ep1["post_is_truncated"])
        self.assertEqual(ep1["post_truncation_reason"], "Subsequent promotion at Week 14")

    def test_clean_bda_calculation_excludes_subsequent_promo(self):
        """
        Regression Test: Verifies that BDA summary metrics for Episode 1 (Week 10)
        use clean weeks 11-13 (1 unit total / 3 weeks = 0.33 units/week),
        NOT contaminated week 14 (which had 5 promotional units!).
        """
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        ep1 = episodes.iloc[0]

        # Clean BDA comparison (default)
        bda_clean = an.compare_before_during_after(self.mock_df, ep1, use_clean_windows=True)
        row = bda_clean.iloc[0]

        self.assertEqual(row["post_actual_weeks"], 3)
        self.assertTrue(row["post_is_truncated"])
        # Total clean post units = 1 (from Week 12 only; Week 14 is excluded!)
        self.assertAlmostEqual(row["post_units_per_week"], 1.0 / 3.0)
        
        # If contaminated windows were used, post-units would be (1 + 5) / 4 = 1.5 units/week
        bda_contaminated = an.compare_before_during_after(self.mock_df, ep1, use_clean_windows=False)
        self.assertAlmostEqual(bda_contaminated.iloc[0]["post_units_per_week"], 1.5)

    def test_household_behavior_clean_windows_and_days(self):
        """
        Regression Test: Verifies that trial buyers repurchasing ONLY during subsequent
        promotions (Week 14) are NOT falsely counted as clean post-promo repeats!
        """
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        ep1 = episodes.iloc[0]

        hh_clean = an.get_household_behavior_summary(self.mock_df, ep1, use_clean_windows=True)
        row = hh_clean.iloc[0]

        # In Week 10: HH 1, 2, 3 bought
        self.assertEqual(row["promo_buyers_count"], 3)
        # In clean post (Weeks 11-13): ONLY HH 1 repurchased (Day 78)
        self.assertEqual(row["repeat_buyers_count"], 1)
        self.assertAlmostEqual(row["post_repurchase_rate_pct"], (1 / 3) * 100.0)

        # HH 3 is a trial buyer, but HH 3 only returned in Week 14 (Promo 2), so trial_repeat_count must be 0!
        self.assertEqual(row["trial_buyers_count"], 1)
        self.assertEqual(row["trial_repeat_count"], 0)
        self.assertEqual(row["trial_retention_rate_pct"], 0.0)

        # Days to first post-purchase for HH 1: Day 78 - Day 64 = 14 days
        self.assertAlmostEqual(row["avg_days_to_first_post_purchase"], 14.0)
        self.assertEqual(row["days_to_return_sample_size"], 1)

    def test_prior_promotion_truncates_pre_window(self):
        """
        Regression Test: Verifies that an episode with a prior promotion has its
        pre-window cleanly truncated to avoid prior promotion contamination.
        """
        edge_trends = pd.DataFrame([
            {"WEEK_NO": 5, "causal_promo_pct": 100.0, "is_promoted_week": True},
            {"WEEK_NO": 8, "causal_promo_pct": 100.0, "is_promoted_week": True}
        ])
        episodes = an.identify_promotion_episodes(edge_trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        self.assertEqual(len(episodes), 2)
        
        ep2 = episodes.iloc[1]
        self.assertEqual(ep2["start_week"], 8)
        # Nominal pre-window is 4 to 7 (4 weeks)
        self.assertEqual(ep2["pre_start_week_nominal"], 4)
        # Clean pre-window must start at Week 6 (after Week 5 promo ends) -> Weeks 6-7 (2 weeks)
        self.assertEqual(ep2["pre_start_week_clean"], 6)
        self.assertEqual(ep2["pre_actual_weeks"], 2)
        self.assertTrue(ep2["pre_is_truncated"])
        self.assertEqual(ep2["pre_truncation_reason"], "Prior promotion at Week 5")

    def test_unknown_causal_coverage_status(self):
        """
        Regression Test: Verifies weeks outside 9-101 are flagged as Unknown coverage,
        causal_promo_pct is NaN, and promo_evaluation_status is Unknown.
        """
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        w1 = trends[trends["WEEK_NO"] == 1].iloc[0]
        # Week 1 has no causal records -> Unknown
        self.assertEqual(w1["causal_coverage_status"], "Unknown")
        self.assertTrue(np.isnan(w1["causal_promo_pct"]))
        self.assertEqual(w1["promo_evaluation_status"], "Unknown")
        self.assertFalse(w1["is_promoted_week"])
        self.assertEqual(w1["causal_coverage_pct"], 0.0)

        # Week 10 is inside 9-101 with records -> Recorded
        w10 = trends[trends["WEEK_NO"] == 10].iloc[0]
        self.assertEqual(w10["causal_coverage_status"], "Recorded")
        self.assertAlmostEqual(w10["causal_promo_pct"], 100.0)
        self.assertEqual(w10["promo_evaluation_status"], "Promoted")
        self.assertTrue(w10["is_promoted_week"])
        self.assertEqual(w10["causal_coverage_pct"], 100.0)

    def test_product_5569230_exact_validation_case_study(self):
        """
        Regression Test: Validates exact Step 3 numbers for Product ID 5569230 (SOFT DRINKS):
        - Before (Weeks 26-29): [17, 23, 40, 26] -> average 26.5 units/week
        - During (Week 30): 236 units -> +790.6% lift
        - Clean After (Weeks 31-33): [25, 22, 31] -> average 26.0 units/week (-1.89% vs Pre)
        - Exclude Week 34 (320 units) due to subsequent promotion.
        """
        records = []
        # Pre: Weeks 26-29
        pre_quantities = {26: 17, 27: 23, 28: 40, 29: 26}
        for wk, qty in pre_quantities.items():
            records.append({
                "household_key": 1, "BASKET_ID": 1000 + wk, "DAY": (wk - 1) * 7 + 1,
                "PRODUCT_ID": 5569230, "QUANTITY": qty, "SALES_VALUE": qty * 4.0,
                "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
        # During: Week 30 (Promo 1)
        records.append({
            "household_key": 2, "BASKET_ID": 2030, "DAY": (30 - 1) * 7 + 1,
            "PRODUCT_ID": 5569230, "QUANTITY": 236, "SALES_VALUE": 236 * 2.64,
            "STORE_ID": 501, "WEEK_NO": 30, "has_causal_evidence": True,
            "total_discount_val": 236 * 1.36, "discount_depth_pct": 34.0,
            "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        })
        # Post clean: Weeks 31-33
        post_quantities = {31: 25, 32: 22, 33: 31}
        for wk, qty in post_quantities.items():
            records.append({
                "household_key": 1, "BASKET_ID": 3000 + wk, "DAY": (wk - 1) * 7 + 1,
                "PRODUCT_ID": 5569230, "QUANTITY": qty, "SALES_VALUE": qty * 4.0,
                "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
        # Subsequent Promo 2: Week 34
        records.append({
            "household_key": 3, "BASKET_ID": 4034, "DAY": (34 - 1) * 7 + 1,
            "PRODUCT_ID": 5569230, "QUANTITY": 320, "SALES_VALUE": 320 * 2.95,
            "STORE_ID": 501, "WEEK_NO": 34, "has_causal_evidence": True,
            "total_discount_val": 320 * 1.05, "discount_depth_pct": 26.2,
            "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        })
        case_df = pd.DataFrame(records)

        trends = an.get_weekly_trends(case_df, min_week=25, max_week=36)
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        
        ep_w30 = episodes[episodes["start_week"] == 30].iloc[0]
        self.assertEqual(ep_w30["pre_actual_weeks"], 4)
        self.assertEqual(ep_w30["post_actual_weeks"], 3)
        self.assertTrue(ep_w30["post_is_truncated"])
        self.assertEqual(ep_w30["post_truncation_reason"], "Subsequent promotion at Week 34")

        # Clean comparison
        bda_clean = an.compare_before_during_after(case_df, ep_w30, use_clean_windows=True).iloc[0]
        self.assertAlmostEqual(bda_clean["pre_units_per_week"], 26.5)
        self.assertAlmostEqual(bda_clean["during_units_per_week"], 236.0)
        self.assertAlmostEqual(bda_clean["post_units_per_week"], 26.0)
        self.assertAlmostEqual(bda_clean["unit_lift_pct"], 790.566037735849, places=3)
        self.assertAlmostEqual(bda_clean["post_vs_pre_unit_pct"], -1.8867924528301887, places=3)

        # Contaminated comparison (if Week 34 is not excluded: (78 + 320) / 4 = 99.5)
        bda_contam = an.compare_before_during_after(case_df, ep_w30, use_clean_windows=False).iloc[0]
        self.assertAlmostEqual(bda_contam["post_units_per_week"], 99.5)
        self.assertAlmostEqual(bda_contam["post_vs_pre_unit_pct"], 275.47169811320754, places=3)

    def test_tight_gap_subsequent_promotion(self):
        """
        Regression Test: Verifies that a subsequent promotion starting only 2 weeks
        after an episode end truncates the post-window to exactly 1 clean week.
        """
        trends = pd.DataFrame([
            {"WEEK_NO": 20, "causal_promo_pct": 100.0, "is_promoted_week": True},
            {"WEEK_NO": 22, "causal_promo_pct": 100.0, "is_promoted_week": True}
        ])
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        ep1 = episodes.iloc[0]
        self.assertEqual(ep1["start_week"], 20)
        self.assertEqual(ep1["post_end_week_clean"], 21)
        self.assertEqual(ep1["post_actual_weeks"], 1)
        self.assertTrue(ep1["post_is_truncated"])
        self.assertEqual(ep1["post_truncation_reason"], "Subsequent promotion at Week 22")

    def test_dataset_boundary_truncation(self):
        """
        Regression Test: Verifies truncation at Week 1 and Week 102 dataset boundaries.
        """
        trends = pd.DataFrame([
            {"WEEK_NO": 2, "causal_promo_pct": 100.0, "is_promoted_week": True},
            {"WEEK_NO": 101, "causal_promo_pct": 100.0, "is_promoted_week": True}
        ])
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=50.0, pre_window=4, post_window=4)
        
        ep_start = episodes.iloc[0]
        self.assertEqual(ep_start["pre_start_week_clean"], 1)
        self.assertEqual(ep_start["pre_actual_weeks"], 1)
        self.assertTrue(ep_start["pre_is_truncated"])
        self.assertEqual(ep_start["pre_truncation_reason"], "Dataset start (Week 1)")
        self.assertFalse(ep_start["causal_coverage_valid"])

        ep_end = episodes.iloc[1]
        self.assertEqual(ep_end["post_end_week_clean"], 102)
        self.assertEqual(ep_end["post_actual_weeks"], 1)
        self.assertTrue(ep_end["post_is_truncated"])
        self.assertEqual(ep_end["post_truncation_reason"], "Dataset end (Week 102)")
        self.assertFalse(ep_end["causal_coverage_valid"])

    def test_unknown_transaction_week_promo_rate_and_status(self):
        """
        Tests that active transactions in an unmonitored week (Weeks 1-8 / Unknown)
        produce causal_promo_pct = NaN, promo_evaluation_status = 'Unknown',
        and is_promoted_week = False (not confirmed unpromoted).
        """
        df_unknown = pd.DataFrame([{
            "household_key": 1, "BASKET_ID": 1001, "DAY": 3,
            "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 5.00,
            "STORE_ID": 501, "WEEK_NO": 1, "has_causal_evidence": False,
            "total_discount_val": 0.0, "discount_depth_pct": 0.0,
            "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        }])
        trends = an.get_weekly_trends(df_unknown, min_week=1, max_week=2)
        w1 = trends[trends["WEEK_NO"] == 1].iloc[0]
        self.assertEqual(w1["total_units"], 2.0)
        self.assertEqual(w1["causal_coverage_status"], "Unknown")
        self.assertTrue(np.isnan(w1["causal_promo_pct"]))
        self.assertEqual(w1["promo_evaluation_status"], "Unknown")
        self.assertFalse(w1["is_promoted_week"])
        self.assertEqual(w1["causal_coverage_pct"], 0.0)

    def test_mixed_coverage_rate_calculation(self):
        """
        Tests that mixed coverage calculates causal_promo_pct using ONLY recorded
        transactions as denominator, and surfaces causal_coverage_pct completeness.
        """
        records = []
        # Store 501: Recorded coverage, 8 promoted, 2 unpromoted (80% promo rate)
        for i in range(8):
            records.append({
                "household_key": 10 + i, "BASKET_ID": 2000 + i, "DAY": 100,
                "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 3.00,
                "STORE_ID": 501, "WEEK_NO": 15, "has_causal_evidence": True,
                "total_discount_val": 1.0, "discount_depth_pct": 25.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
        for i in range(2):
            records.append({
                "household_key": 20 + i, "BASKET_ID": 2100 + i, "DAY": 100,
                "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 4.00,
                "STORE_ID": 501, "WEEK_NO": 15, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
        # Store 502: Unknown coverage, 10 transactions
        for i in range(10):
            records.append({
                "household_key": 30 + i, "BASKET_ID": 2200 + i, "DAY": 100,
                "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 4.00,
                "STORE_ID": 502, "WEEK_NO": 15, "has_causal_evidence": False,
                "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
                "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
            })
        df_mixed = pd.DataFrame(records)
        trends = an.get_weekly_trends(df_mixed, min_week=15, max_week=15)
        w15 = trends[trends["WEEK_NO"] == 15].iloc[0]

        self.assertEqual(w15["total_transactions"], 20)
        self.assertEqual(w15["causal_coverage_status"], "Mixed")
        self.assertAlmostEqual(w15["causal_coverage_pct"], 50.0)
        self.assertEqual(w15["causal_promoted_trans_count"], 8)
        # Denominator should be 10 recorded transactions, yielding 80%, NOT 8/20 = 40%
        self.assertAlmostEqual(w15["causal_promo_pct"], 80.0)
        self.assertTrue(w15["is_promoted_week"])
        self.assertEqual(w15["promo_evaluation_status"], "Promoted")

    def test_zero_transaction_weeks(self):
        """
        Tests that zero-transaction weeks in both unmonitored (Weeks 1-8) and
        monitored (Weeks 9-101) periods retain NaN promo rate and Unknown evaluation status.
        """
        # Pass empty DataFrame across Weeks 1 to 10
        trends_empty = an.get_weekly_trends(pd.DataFrame(columns=["WEEK_NO", "QUANTITY", "SALES_VALUE"]), min_week=1, max_week=10)
        self.assertEqual(len(trends_empty), 0)

        # Pass 1 transaction in Week 5, leaving Week 1 and Week 9 as zero-transaction weeks
        df_single = pd.DataFrame([{
            "household_key": 1, "BASKET_ID": 101, "DAY": 30,
            "PRODUCT_ID": 101, "QUANTITY": 1, "SALES_VALUE": 2.00,
            "STORE_ID": 501, "WEEK_NO": 5, "has_causal_evidence": False,
            "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        }])
        trends_single = an.get_weekly_trends(df_single, min_week=1, max_week=10)

        # Week 1: 0 transactions outside causal range
        w1 = trends_single[trends_single["WEEK_NO"] == 1].iloc[0]
        self.assertEqual(w1["total_transactions"], 0)
        self.assertEqual(w1["causal_coverage_status"], "Unknown")
        self.assertTrue(np.isnan(w1["causal_promo_pct"]))
        self.assertEqual(w1["promo_evaluation_status"], "Unknown")
        self.assertFalse(w1["is_promoted_week"])

        # Week 9: 0 transactions within causal range
        w9 = trends_single[trends_single["WEEK_NO"] == 9].iloc[0]
        self.assertEqual(w9["total_transactions"], 0)
        self.assertEqual(w9["causal_coverage_status"], "Recorded")
        self.assertTrue(np.isnan(w9["causal_promo_pct"]))
        self.assertEqual(w9["promo_evaluation_status"], "Unknown")
        self.assertFalse(w9["is_promoted_week"])

    def test_early_promotion_episode_causal_boundaries(self):
        """
        Tests that promotion episodes near Week 9 enforce causal boundaries when requested,
        and flag coverage warnings in compare_before_during_after.
        """
        # Promotion at Week 10 with nominal pre-window Weeks 6-9
        trends = pd.DataFrame([
            {"WEEK_NO": 10, "causal_promo_pct": 100.0, "is_promoted_week": True}
        ])

        # 1. With enforce_causal_boundaries=True: pre-window truncates to Week 9
        episodes_enforced = an.identify_promotion_episodes(
            trends, min_promo_pct=50.0, pre_window=4, post_window=4, enforce_causal_boundaries=True
        )
        ep_enf = episodes_enforced.iloc[0]
        self.assertEqual(ep_enf["pre_start_week_clean"], 9)
        self.assertEqual(ep_enf["pre_actual_weeks"], 1)
        self.assertTrue(ep_enf["pre_is_truncated"])
        self.assertEqual(ep_enf["pre_truncation_reason"], "Causal coverage start (Week 9)")
        self.assertTrue(ep_enf["causal_coverage_valid"])

        # 2. With enforce_causal_boundaries=False: pre-window starts at Week 6, but causal_coverage_valid is False
        episodes_default = an.identify_promotion_episodes(
            trends, min_promo_pct=50.0, pre_window=4, post_window=4, enforce_causal_boundaries=False
        )
        ep_def = episodes_default.iloc[0]
        self.assertEqual(ep_def["pre_start_week_clean"], 6)
        self.assertEqual(ep_def["pre_actual_weeks"], 4)
        self.assertFalse(ep_def["causal_coverage_valid"])

        # 3. Test compare_before_during_after detects unmonitored baseline
        records = []
        for wk in range(6, 10):
            records.append({
                "household_key": 1, "BASKET_ID": 100 + wk, "DAY": wk * 7,
                "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 4.0,
                "STORE_ID": 501, "WEEK_NO": wk, "causal_coverage_status": "Unknown" if wk < 9 else "Recorded"
            })
        records.append({
            "household_key": 1, "BASKET_ID": 200, "DAY": 70,
            "PRODUCT_ID": 101, "QUANTITY": 10, "SALES_VALUE": 15.0,
            "STORE_ID": 501, "WEEK_NO": 10, "causal_coverage_status": "Recorded"
        })
        records.append({
            "household_key": 1, "BASKET_ID": 300, "DAY": 77,
            "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 4.0,
            "STORE_ID": 501, "WEEK_NO": 11, "causal_coverage_status": "Recorded"
        })
        df_trans = pd.DataFrame(records)

        bda_def = an.compare_before_during_after(df_trans, ep_def, use_clean_windows=True).iloc[0]
        self.assertFalse(bda_def["causal_coverage_valid"])
        self.assertFalse(bda_def["has_reliable_baseline"])
        self.assertIn("Pre-promotion baseline includes weeks prior to Week 9", bda_def["coverage_warning"])

        bda_enf = an.compare_before_during_after(df_trans, ep_enf, use_clean_windows=True).iloc[0]
        self.assertTrue(bda_enf["causal_coverage_valid"])
        self.assertTrue(bda_enf["has_reliable_baseline"])
        self.assertEqual(bda_enf["coverage_warning"], "None")
        self.assertEqual(bda_enf["pre_actual_weeks"], 1)

    def test_late_promotion_episode_causal_boundaries(self):
        """
        Tests that promotion episodes near Week 101 truncate post-window at Week 101
        when enforce_causal_boundaries=True and flag warnings when reaching Week 102.
        """
        trends = pd.DataFrame([
            {"WEEK_NO": 100, "causal_promo_pct": 100.0, "is_promoted_week": True}
        ])

        # Enforced: post-window truncates to Week 101 (1 week)
        episodes_enf = an.identify_promotion_episodes(
            trends, min_promo_pct=50.0, pre_window=4, post_window=4, enforce_causal_boundaries=True
        )
        ep_enf = episodes_enf.iloc[0]
        self.assertEqual(ep_enf["post_end_week_clean"], 101)
        self.assertEqual(ep_enf["post_actual_weeks"], 1)
        self.assertTrue(ep_enf["post_is_truncated"])
        self.assertEqual(ep_enf["post_truncation_reason"], "Causal coverage end (Week 101)")
        self.assertTrue(ep_enf["causal_coverage_valid"])

        # Default: post-window extends to Week 102, causal_coverage_valid is False
        episodes_def = an.identify_promotion_episodes(
            trends, min_promo_pct=50.0, pre_window=4, post_window=4, enforce_causal_boundaries=False
        )
        ep_def = episodes_def.iloc[0]
        self.assertEqual(ep_def["post_end_week_clean"], 102)
        self.assertEqual(ep_def["post_actual_weeks"], 2)
        self.assertFalse(ep_def["causal_coverage_valid"])

    def test_missing_comparison_window_returns_nan_not_zero(self):
        """
        Regression Test: Verifies that when pre or post observation windows lack transaction data,
        rates and percentage changes return np.nan rather than false 0.0 rates or false -100% dips.
        """
        # Promotion at Week 9: pre-window Weeks 5-8 is prior to causal tracking -> 0 clean pre-weeks
        # Transactions only exist for Week 9 (no post transactions)
        df_ep = pd.DataFrame([{
            "household_key": 1, "BASKET_ID": 901, "DAY": 60,
            "PRODUCT_ID": 101, "QUANTITY": 50, "SALES_VALUE": 100.0,
            "STORE_ID": 501, "WEEK_NO": 9, "has_causal_evidence": True,
            "total_discount_val": 25.0, "discount_depth_pct": 20.0,
            "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
            "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
        }])
        ep = {
            "episode_id": "EP_01", "start_week": 9, "end_week": 9, "duration_weeks": 1,
            "pre_actual_weeks": 0, "post_actual_weeks": 4,
            "pre_start_week_clean": 9, "post_end_week_clean": 13,
            "causal_coverage_valid": False
        }
        bda = an.compare_before_during_after(df_ep, ep, use_clean_windows=True).iloc[0]

        # Pre has 0 observation weeks -> pre_units_per_week must be NaN, NOT 0.0
        self.assertTrue(np.isnan(bda["pre_units_per_week"]))
        self.assertTrue(np.isnan(bda["unit_lift_pct"]))

        # Post has 0 recorded transactions -> post_units_per_week and dip must be NaN, NOT -100.0%
        self.assertTrue(np.isnan(bda["post_units_per_week"]))
        self.assertTrue(np.isnan(bda["post_vs_during_dip_pct"]))
        self.assertTrue(np.isnan(bda["post_vs_pre_unit_pct"]))

    def test_event_log_does_not_create_93_week_episode(self):
        """
        Regression Test: Verifies that when causal tracking follows an event-log pattern
        (only promoted transactions have causal_coverage_status='Recorded', while baseline
        transactions lack causal records), isolated promotions do not get coalesced into
        a single continuous 93-week episode across Weeks 9-101.
        """
        records = []
        for wk in range(9, 102):
            if wk in [10, 25]:
                # True promotion weeks: 100% promoted
                for i in range(10):
                    records.append({
                        "household_key": 100 + i, "BASKET_ID": 1000 + i, "DAY": wk * 7,
                        "PRODUCT_ID": 101, "QUANTITY": 10, "SALES_VALUE": 20.0,
                        "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": True,
                        "total_discount_val": 5.0, "discount_depth_pct": 20.0,
                        "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                        "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
                    })
            elif 11 <= wk <= 15:
                # Weeks with low background activity (1 out of 10 = 10% promo rate < 25% threshold)
                records.append({
                    "household_key": 200, "BASKET_ID": 2000, "DAY": wk * 7,
                    "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 4.0,
                    "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": True,
                    "total_discount_val": 1.0, "discount_depth_pct": 20.0,
                    "causal_coverage_status": "Recorded", "DEPARTMENT": "GROCERY",
                    "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
                })
                for i in range(1, 10):
                    records.append({
                        "household_key": 200 + i, "BASKET_ID": 2000 + i, "DAY": wk * 7,
                        "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 5.0,
                        "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                        "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                        "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
                        "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
                    })
            else:
                # Quiet baseline weeks: 0 promoted transactions
                for i in range(10):
                    records.append({
                        "household_key": 300 + i, "BASKET_ID": 3000 + i, "DAY": wk * 7,
                        "PRODUCT_ID": 101, "QUANTITY": 2, "SALES_VALUE": 5.0,
                        "STORE_ID": 501, "WEEK_NO": wk, "has_causal_evidence": False,
                        "total_discount_val": 0.0, "discount_depth_pct": 0.0,
                        "causal_coverage_status": "Unknown", "DEPARTMENT": "GROCERY",
                        "COMMODITY_DESC": "SOFT DRINKS", "SUB_COMMODITY_DESC": "COLA CAN"
                    })

        df_log = pd.DataFrame(records)
        trends = an.get_weekly_trends(df_log, min_week=9, max_week=101, promo_threshold_pct=25.0)

        # Weeks 11-15 should be 10% promo rate (<25% threshold) and NOT marked as promoted
        w12 = trends[trends["WEEK_NO"] == 12].iloc[0]
        self.assertAlmostEqual(w12["causal_promo_pct"], 10.0)
        self.assertFalse(w12["is_promoted_week"])

        # Week 10 and Week 25 should be 100% promo rate and marked promoted
        w10 = trends[trends["WEEK_NO"] == 10].iloc[0]
        self.assertAlmostEqual(w10["causal_promo_pct"], 100.0)
        self.assertTrue(w10["is_promoted_week"])

        # Episodes must identify discrete episodes (Week 10 and Week 25), NOT a single 93-week episode
        episodes = an.identify_promotion_episodes(trends, min_promo_pct=25.0)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.iloc[0]["start_week"], 10)
        self.assertEqual(episodes.iloc[0]["end_week"], 10)
        self.assertEqual(episodes.iloc[0]["duration_weeks"], 1)
        self.assertEqual(episodes.iloc[1]["start_week"], 25)
        self.assertEqual(episodes.iloc[1]["end_week"], 25)
        self.assertEqual(episodes.iloc[1]["duration_weeks"], 1)

    # --------------------------------------------------------------------------
    # Opportunity Feature: Explore Similar Products Tests
    # --------------------------------------------------------------------------

    def test_similar_products_commodity_restriction(self):
        """Verify summary returns only products belonging to requested commodity."""
        df_test = pd.DataFrame({
            "COMMODITY_DESC": ["SOFT DRINKS", "SOFT DRINKS", "POTATOES", "COLD CEREAL"],
            "PRODUCT_ID": [101, 102, 201, 301],
            "QUANTITY": [5, 10, 2, 4],
            "SALES_VALUE": [10.0, 15.0, 4.0, 12.0],
            "household_key": [1, 2, 3, 4],
            "WEEK_NO": [10, 10, 10, 10],
            "has_causal_evidence": [False, True, False, False],
            "SUB_COMMODITY_DESC": ["COLA", "DIET COLA", "RUSSET", "FLAKES"],
            "BRAND": ["National", "National", "Private", "National"],
            "CURR_SIZE_OF_PRODUCT": ["12 PK", "2 LTR", "5 LB", "18 OZ"]
        })
        summary = an.get_similar_products_summary(df_test, "SOFT DRINKS")
        self.assertEqual(len(summary), 2)
        self.assertSetEqual(set(summary["PRODUCT_ID"].tolist()), {101, 102})
        self.assertNotIn(201, summary["PRODUCT_ID"].values)
        self.assertNotIn(301, summary["PRODUCT_ID"].values)

    def test_similar_products_focal_identification_and_inclusion(self):
        """Verify focal product is marked is_focal=True and included even if ranked outside top_n."""
        df_test = pd.DataFrame({
            "COMMODITY_DESC": ["SOFT DRINKS"] * 4,
            "PRODUCT_ID": [101, 102, 103, 104],
            "QUANTITY": [100, 80, 60, 20],  # 104 ranks 4th
            "SALES_VALUE": [200.0, 160.0, 120.0, 40.0],
            "household_key": [1, 2, 3, 4],
            "WEEK_NO": [10, 10, 10, 10],
            "has_causal_evidence": [False, False, False, False],
            "SUB_COMMODITY_DESC": ["SODA"] * 4,
            "BRAND": ["National"] * 4,
            "CURR_SIZE_OF_PRODUCT": ["12 PK"] * 4
        })
        # Top 2 products requested, but focal product is 104 (ranks 4th)
        summary = an.get_similar_products_summary(df_test, "SOFT DRINKS", current_product_id=104, top_n=2)
        # Should contain top 2 (101, 102) PLUS focal product 104
        self.assertEqual(len(summary), 3)
        focal_row = summary[summary["PRODUCT_ID"] == 104]
        self.assertEqual(len(focal_row), 1)
        self.assertTrue(bool(focal_row.iloc[0]["is_focal"]))
        self.assertFalse(summary[summary["PRODUCT_ID"] == 101].iloc[0]["is_focal"])

    def test_similar_products_hand_checkable_aggregates(self):
        """Verify units, sales, and average unit price match exact calculations."""
        df_test = pd.DataFrame({
            "COMMODITY_DESC": ["SOFT DRINKS", "SOFT DRINKS"],
            "PRODUCT_ID": [101, 101],
            "QUANTITY": [3, 7],
            "SALES_VALUE": [6.00, 14.00],
            "household_key": [1, 2],
            "WEEK_NO": [10, 11],
            "has_causal_evidence": [False, False],
            "SUB_COMMODITY_DESC": ["COLA", "COLA"],
            "BRAND": ["National", "National"],
            "CURR_SIZE_OF_PRODUCT": ["12 PK", "12 PK"]
        })
        summary = an.get_similar_products_summary(df_test, "SOFT DRINKS", current_product_id=101)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["total_units"], 10.0)
        self.assertEqual(row["total_sales"], 20.0)
        self.assertEqual(row["avg_unit_price"], 2.00)

    def test_similar_products_distinct_counts_no_duplicates(self):
        """Verify distinct shoppers and promoted weeks do not double-count repeat trips or multi-store rows."""
        df_test = pd.DataFrame({
            "COMMODITY_DESC": ["SOFT DRINKS"] * 5,
            "PRODUCT_ID": [101] * 5,
            "QUANTITY": [1, 2, 1, 3, 2],
            "SALES_VALUE": [2.0, 4.0, 2.0, 6.0, 4.0],
            # Household 1 has 3 transactions, Household 2 has 2 transactions -> distinct shoppers = 2
            "household_key": [1, 1, 1, 2, 2],
            # Transactions in Week 10 (2 promo rows) and Week 12 (1 promo row) -> promoted weeks count = 2
            "WEEK_NO": [10, 10, 11, 12, 12],
            "has_causal_evidence": [True, True, False, True, False],
            "SUB_COMMODITY_DESC": ["COLA"] * 5,
            "BRAND": ["National"] * 5,
            "CURR_SIZE_OF_PRODUCT": ["12 PK"] * 5
        })
        summary = an.get_similar_products_summary(df_test, "SOFT DRINKS", current_product_id=101)
        row = summary.iloc[0]
        self.assertEqual(row["distinct_shoppers"], 2)
        self.assertEqual(row["promoted_weeks_count"], 2)

    def test_similar_products_edge_cases(self):
        """Verify empty input, missing values, single-product commodity, and fewer than top_n products."""
        # 1. Empty and None df
        cols = [
            "PRODUCT_ID", "SUB_COMMODITY_DESC", "BRAND", "CURR_SIZE_OF_PRODUCT",
            "total_units", "total_sales", "distinct_shoppers", "avg_unit_price",
            "promoted_weeks_count", "is_focal"
        ]
        res_none = an.get_similar_products_summary(None, "SOFT DRINKS")
        self.assertListEqual(list(res_none.columns), cols)
        self.assertTrue(res_none.empty)

        res_empty = an.get_similar_products_summary(pd.DataFrame(), "SOFT DRINKS")
        self.assertListEqual(list(res_empty.columns), cols)
        self.assertTrue(res_empty.empty)

        # 2. Unknown commodity
        df_single = pd.DataFrame({
            "COMMODITY_DESC": ["PASTA"],
            "PRODUCT_ID": [999],
            "QUANTITY": [2],
            "SALES_VALUE": [3.50],
            "household_key": [10],
            "WEEK_NO": [15],
            "has_causal_evidence": [False]
        })
        res_missing_comm = an.get_similar_products_summary(df_single, "NON_EXISTENT")
        self.assertTrue(res_missing_comm.empty)

        # 3. Single-product commodity with top_n=10
        res_single = an.get_similar_products_summary(df_single, "PASTA", top_n=10)
        self.assertEqual(len(res_single), 1)
        self.assertEqual(res_single.iloc[0]["PRODUCT_ID"], 999)

        # 4. Zero sales / zero units
        df_zeros = pd.DataFrame({
            "COMMODITY_DESC": ["SNACKS"],
            "PRODUCT_ID": [888],
            "QUANTITY": [0],
            "SALES_VALUE": [0.0],
            "household_key": [5],
            "WEEK_NO": [20],
            "has_causal_evidence": [False]
        })
        res_zero = an.get_similar_products_summary(df_zeros, "SNACKS")
        self.assertEqual(res_zero.iloc[0]["avg_unit_price"], 0.0)


if __name__ == "__main__":
    unittest.main()

