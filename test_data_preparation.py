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


if __name__ == "__main__":
    unittest.main()
