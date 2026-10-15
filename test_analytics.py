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
        Regression Test: Verifies weeks outside 9-101 are flagged as Unknown coverage.
        """
        trends = an.get_weekly_trends(self.mock_df, include_returns=False, min_week=1, max_week=20)
        w1 = trends[trends["WEEK_NO"] == 1].iloc[0]
        # Week 1 has no causal records -> Unknown
        self.assertEqual(w1["causal_coverage_status"], "Unknown")
        # Week 10 is inside 9-101 with records -> Recorded
        w10 = trends[trends["WEEK_NO"] == 10].iloc[0]
        self.assertEqual(w10["causal_coverage_status"], "Recorded")


if __name__ == "__main__":
    unittest.main()
