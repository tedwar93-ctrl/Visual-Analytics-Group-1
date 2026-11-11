"""
test_data_preparation.py
------------------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Unit and integration tests for data_preparation.py

Validates:
1. Uniqueness of aggregated causal composite keys (PRODUCT_ID, STORE_ID, WEEK_NO).
2. Exact row count preservation across transaction-product join (no row loss / multiplication).
3. Exact row count preservation after merging with aggregated causal data.
4. Correctness of multi-level promotion classification and separate boolean flags.
5. Strict distinction between missing causal coverage vs recorded baseline.
6. Native WEEK_NO preservation.
"""

import unittest
import numpy as np
import pandas as pd
import data_preparation as dp


class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Create synthetic test datasets
        self.mock_products = pd.DataFrame({
            "PRODUCT_ID": [101, 102, 103],
            "MANUFACTURER": [1, 2, 3],
            "DEPARTMENT": ["GROCERY", "GROCERY", "PASTRY"],
            "BRAND": ["National", "Private", "National"],
            "COMMODITY_DESC": ["SOFT DRINKS", "CEREAL", "BAGELS"],
            "SUB_COMMODITY_DESC": ["COLA", "OAT", "PLAIN"],
            "CURR_SIZE_OF_PRODUCT": ["12 PK", "18 OZ", "6 CT"]
        })

        self.mock_transactions = pd.DataFrame({
            "household_key": [1, 1, 2, 3, 4],
            "BASKET_ID": [1001, 1001, 1002, 1003, 1004],
            "DAY": [1, 1, 15, 60, 90],
            "PRODUCT_ID": [101, 102, 101, 103, 102],
            "QUANTITY": [1, 2, 1, 1, 3],
            "SALES_VALUE": [2.50, 4.00, 2.00, 3.50, 6.00],
            "STORE_ID": [501, 501, 501, 502, 503],
            "RETAIL_DISC": [-0.50, 0.0, 0.0, -1.00, 0.0],
            "TRANS_TIME": [1200, 1201, 1430, 1800, 930],
            "WEEK_NO": [1, 1, 3, 9, 14],
            "COUPON_DISC": [0.0, -0.75, 0.0, 0.0, 0.0],
            "COUPON_MATCH_DISC": [0.0, 0.0, 0.0, 0.0, 0.0]
        })

        # Mock causal data including intentional duplicate keys and conflicting codes
        self.mock_causal = pd.DataFrame({
            "PRODUCT_ID": [101, 101, 102, 103],
            "STORE_ID":   [501, 501, 501, 502],
            "WEEK_NO":    [1,   1,   1,   9],
            "display":    ["0", "6", "0", "1"],
            "mailer":     ["A", "A", "D", "0"]
        })

    def test_causal_aggregation_uniqueness(self):
        """Verifies duplicate keys in causal data are aggregated to unique composite keys."""
        # Row 0 and Row 1 share (PRODUCT_ID=101, STORE_ID=501, WEEK_NO=1)
        raw_rows = len(self.mock_causal)
        
        agg = self.mock_causal.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).agg(
            display_codes=("display", lambda s: "|".join(sorted(set(s)))),
            mailer_codes=("mailer", lambda s: "|".join(sorted(set(s)))),
            has_display_feature=("display", lambda s: any(x != "0" for x in s)),
            has_mailer_feature=("mailer", lambda s: any(x != "0" for x in s))
        ).reset_index()
        agg["has_causal_evidence"] = agg["has_display_feature"] | agg["has_mailer_feature"]
        agg["causal_record_exists"] = True

        # Assert no duplicates
        self.assertEqual(len(agg), 3)
        self.assertEqual(agg.duplicated(subset=["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).sum(), 0)

        # Assert display codes aggregated with pipe
        dup_record = agg[(agg["PRODUCT_ID"] == 101) & (agg["STORE_ID"] == 501) & (agg["WEEK_NO"] == 1)].iloc[0]
        self.assertEqual(dup_record["display_codes"], "0|6")
        self.assertEqual(dup_record["mailer_codes"], "A")
        self.assertTrue(dup_record["has_display_feature"])
        self.assertTrue(dup_record["has_mailer_feature"])
        self.assertTrue(dup_record["has_causal_evidence"])

    def test_transaction_to_product_join_preservation(self):
        """Verifies joining transactions with products preserves exact row count."""
        initial_count = len(self.mock_transactions)
        joined = dp.join_transactions_and_products(self.mock_transactions, self.mock_products)
        self.assertEqual(len(joined), initial_count)
        self.assertIn("COMMODITY_DESC", joined.columns)
        self.assertIn("DEPARTMENT", joined.columns)

    def test_causal_join_row_preservation(self):
        """Verifies merging transactions with aggregated causal data does NOT cause row explosion."""
        initial_count = len(self.mock_transactions)
        
        # Build aggregated causal
        agg = self.mock_causal.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).agg(
            display_codes=("display", lambda s: "|".join(sorted(set(s)))),
            mailer_codes=("mailer", lambda s: "|".join(sorted(set(s)))),
            has_display_feature=("display", lambda s: any(x != "0" for x in s)),
            has_mailer_feature=("mailer", lambda s: any(x != "0" for x in s))
        ).reset_index()
        agg["has_causal_evidence"] = agg["has_display_feature"] | agg["has_mailer_feature"]
        agg["causal_record_exists"] = True

        joined_tp = dp.join_transactions_and_products(self.mock_transactions, self.mock_products)
        enriched = dp.classify_promotion_evidence(joined_tp, agg)

        # Assert zero row count change
        self.assertEqual(len(enriched), initial_count)

    def test_promotion_evidence_classification(self):
        """Verifies that each transaction is accurately categorized."""
        agg = self.mock_causal.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).agg(
            display_codes=("display", lambda s: "|".join(sorted(set(s)))),
            mailer_codes=("mailer", lambda s: "|".join(sorted(set(s)))),
            has_display_feature=("display", lambda s: any(x != "0" for x in s)),
            has_mailer_feature=("mailer", lambda s: any(x != "0" for x in s))
        ).reset_index()
        agg["has_causal_evidence"] = agg["has_display_feature"] | agg["has_mailer_feature"]
        agg["causal_record_exists"] = True

        joined_tp = dp.join_transactions_and_products(self.mock_transactions, self.mock_products)
        enriched = dp.classify_promotion_evidence(joined_tp, agg)

        # Row 0: PID 101, STORE 501, WEEK 1 -> Causal (disp=0|6, mailer=A) & RETAIL_DISC -0.50 -> "Causal & Discount"
        r0 = enriched.iloc[0]
        self.assertEqual(r0["promo_evidence_type"], "Causal & Discount")
        self.assertTrue(r0["has_causal_evidence"])
        self.assertTrue(r0["has_discount_evidence"])
        self.assertEqual(r0["causal_coverage_status"], "Recorded")

        # Row 1: PID 102, STORE 501, WEEK 1 -> Causal (disp=0, mailer=D) & COUPON_DISC -0.75 -> "Causal & Discount"
        r1 = enriched.iloc[1]
        self.assertEqual(r1["promo_evidence_type"], "Causal & Discount")
        self.assertTrue(r1["has_mailer_feature"])
        self.assertFalse(r1["has_display_feature"])

        # Row 2: PID 101, STORE 501, WEEK 3 -> NOT in mock_causal, RETAIL_DISC 0, COUPON 0 -> "Baseline with Unknown Causal"
        r2 = enriched.iloc[2]
        self.assertEqual(r2["promo_evidence_type"], "Baseline with Unknown Causal")
        self.assertEqual(r2["causal_coverage_status"], "Unknown")
        self.assertFalse(r2["has_discount_evidence"])
        self.assertFalse(r2["has_causal_evidence"])

        # Row 3: PID 103, STORE 502, WEEK 9 -> Causal (disp=1, mailer=0) & RETAIL_DISC -1.00 -> "Causal & Discount"
        r3 = enriched.iloc[3]
        self.assertEqual(r3["promo_evidence_type"], "Causal & Discount")

        # Row 4: PID 102, STORE 503, WEEK 14 -> NOT in mock_causal, No discounts -> "Baseline with Unknown Causal"
        r4 = enriched.iloc[4]
        self.assertEqual(r4["promo_evidence_type"], "Baseline with Unknown Causal")
        self.assertEqual(r4["causal_coverage_status"], "Unknown")

    def test_native_week_preserved(self):
        """Verifies WEEK_NO is untouched."""
        joined = dp.join_transactions_and_products(self.mock_transactions, self.mock_products)
        self.assertTrue((joined["WEEK_NO"] == self.mock_transactions["WEEK_NO"]).all())


class TestAggregateCausalDataNoMatchMessage(unittest.TestCase):
    """Tests the contextual message branch in aggregate_causal_data() when no records match."""

    def _make_target_keys(self, pid_arr, store_arr, week_arr):
        """Helper to produce sorted encoded keys matching dp.encode_psw_key()."""
        return np.sort(np.unique(dp.encode_psw_key(
            np.array(pid_arr, dtype=np.int64),
            np.array(store_arr, dtype=np.int64),
            np.array(week_arr, dtype=np.int64),
        )))

    def _run_with_mock_csv(self, mock_csv_content, target_keys, capture_stdout=True):
        """
        Run aggregate_causal_data against a tiny mock CSV string and return
        (result_df, printed_output).  Uses a temp file to avoid touching raw data.
        """
        import io
        import tempfile
        from unittest.mock import patch

        # Write mock CSV to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(mock_csv_content)
            tmp_path = f.name

        captured = io.StringIO()
        try:
            with patch("sys.stdout", captured):
                result = dp.aggregate_causal_data(
                    causal_filepath=tmp_path,
                    target_keys_sorted=target_keys,
                    chunk_size=10_000,
                )
        finally:
            import os
            os.unlink(tmp_path)

        return result, captured.getvalue()

    MOCK_CAUSAL_WEEKS_9_PLUS = (
        "PRODUCT_ID,STORE_ID,WEEK_NO,display,mailer\n"
        "101,500,10,0,A\n"
        "102,501,15,1,D\n"
    )

    def test_weeks_1_to_8_only_prints_notice(self):
        """All target keys in Weeks 1-8 must print the informational Notice, not a Warning."""
        # Keys all in weeks 1-8 — no possible causal match
        target_keys = self._make_target_keys([101, 102], [500, 501], [1, 8])
        result, output = self._run_with_mock_csv(self.MOCK_CAUSAL_WEEKS_9_PLUS, target_keys)

        self.assertIn("Notice:", output)
        self.assertIn("Weeks 1", output)
        self.assertIn("Week 9", output)
        self.assertNotIn("Warning:", output)
        self.assertEqual(len(result), 0)

    def test_covered_weeks_with_no_match_prints_warning(self):
        """Target keys spanning covered weeks but with no overlap must print a Warning."""
        # Keys in weeks 10-12 but no matching products in mock CSV (PIDs 999, 998)
        target_keys = self._make_target_keys([999, 998], [999, 998], [10, 12])
        result, output = self._run_with_mock_csv(self.MOCK_CAUSAL_WEEKS_9_PLUS, target_keys)

        self.assertIn("Warning:", output)
        self.assertNotIn("Notice:", output)
        self.assertEqual(len(result), 0)

    def test_no_filter_path_prints_warning(self):
        """When target_keys_sorted is supplied and covered-week keys exist but no rows match, Warning is printed.

        Note: When target_keys_sorted=None, aggregate_causal_data() always appends every chunk
        (the else branch at line ~106), so an empty CSV goes through groupby which yields an
        empty DataFrame, not the no-match branch.  The Warning code path for the unfiltered case
        is therefore unreachable with an empty CSV; we test the filtered-but-no-match path instead.
        """
        # Keys in covered weeks but no product/store/week overlap with mock CSV
        target_keys = self._make_target_keys([888, 777], [888, 777], [9, 11])
        result, output = self._run_with_mock_csv(self.MOCK_CAUSAL_WEEKS_9_PLUS, target_keys)

        self.assertIn("Warning:", output)
        self.assertNotIn("Notice:", output)
        self.assertEqual(len(result), 0)

    def test_mixed_weeks_with_matching_returns_results(self):
        """Target keys spanning both pre-causal and causal weeks that DO match return aggregated rows."""
        # Include WEEK=10 for PID=101/STORE=500 to match the CSV row (PID=101, STORE=500, WEEK=10)
        # Also include a pre-causal key (WEEK=1) and a non-matching covered key
        target_keys = self._make_target_keys(
            [101, 101, 9999],
            [500, 500, 999],
            [1,   10,  11],   # WEEK=10 matches CSV entry for PID=101/STORE=500
        )
        result, output = self._run_with_mock_csv(self.MOCK_CAUSAL_WEEKS_9_PLUS, target_keys)

        # PID=101, STORE=500, WEEK=10 must be found in the mock CSV
        self.assertGreater(len(result), 0)
        self.assertNotIn("Warning:", output)
        self.assertNotIn("Notice:", output)


if __name__ == "__main__":
    unittest.main()
