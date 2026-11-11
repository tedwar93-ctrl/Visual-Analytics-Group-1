"""
test_ai.py
----------
Unit test suite for app/ai.py (AI Insight Narrator).
Validates:
1. Low-confidence safety bypass (sample size < 30 or insufficient_data=True).
2. Strict fact-bound grounding and numerical verification audit.
3. Prohibited financial terminology enforcement (profit, margin, COGS, cost, ROI).
4. Neutral behavioral framing enforcement (no causal or pantry storage claims).
5. Deterministic template fallback formatting and accuracy.
6. End-to-end generate_insight_narrative contract compliance.
"""

import math
import unittest
from unittest.mock import patch

from app.ai import (
    INSUFFICIENT_DATA_MESSAGE,
    BANNED_FINANCIAL_WORDS,
    BANNED_FINANCIAL_PATTERN,
    BANNED_CAUSAL_PATTERNS,
    is_low_confidence,
    generate_template_narrative,
    extract_numbers_from_text,
    extract_allowed_numbers_from_dict,
    is_number_allowed,
    verify_narrative,
    generate_insight_narrative,
)


class TestAILowConfidence(unittest.TestCase):
    """Tests automated low-confidence safety handling."""

    def test_insufficient_data_flag_triggers_safety_message(self):
        metrics = {
            "commodity_desc": "SOFT DRINKS",
            "sample_size_n": 150,
            "insufficient_data": True,
            "unit_lift_pct": 50.0,
        }
        self.assertTrue(is_low_confidence(metrics))
        result = generate_insight_narrative(metrics)
        self.assertEqual(result, INSUFFICIENT_DATA_MESSAGE)

    def test_sample_size_less_than_30_triggers_safety_message(self):
        metrics = {
            "commodity_desc": "SOFT DRINKS",
            "sample_size_n": 29,
            "insufficient_data": False,
            "unit_lift_pct": 50.0,
        }
        self.assertTrue(is_low_confidence(metrics))
        result = generate_insight_narrative(metrics)
        self.assertEqual(result, INSUFFICIENT_DATA_MESSAGE)

    def test_promo_buyers_count_fallback_less_than_30(self):
        metrics = {
            "commodity_desc": "SOFT DRINKS",
            "promo_buyers_count": 15,
            "insufficient_data": False,
            "unit_lift_pct": 50.0,
        }
        self.assertTrue(is_low_confidence(metrics))
        result = generate_insight_narrative(metrics)
        self.assertEqual(result, INSUFFICIENT_DATA_MESSAGE)

    def test_none_and_empty_inputs(self):
        self.assertTrue(is_low_confidence(None))
        self.assertEqual(generate_insight_narrative(None), INSUFFICIENT_DATA_MESSAGE)

        self.assertTrue(is_low_confidence({}))
        self.assertEqual(generate_insight_narrative({}), INSUFFICIENT_DATA_MESSAGE)

    def test_adequate_sample_size_passes_confidence_check(self):
        metrics = {
            "commodity_desc": "SOFT DRINKS",
            "sample_size_n": 30,
            "insufficient_data": False,
            "unit_lift_pct": 25.0,
        }
        self.assertFalse(is_low_confidence(metrics))
        result = generate_insight_narrative(metrics)
        self.assertNotEqual(result, INSUFFICIENT_DATA_MESSAGE)


class TestAIGuardrails(unittest.TestCase):
    """Tests negative constraints, banned terminology, and neutral behavioral rules."""

    def test_banned_financial_words_detected(self):
        for word in ["profit", "profits", "margin", "margins", "COGS", "cost", "costs", "ROI"]:
            text = f"- **Performance:** This promotional activity generated strong {word} for the retailer."
            self.assertIsNotNone(
                BANNED_FINANCIAL_PATTERN.search(text),
                f"Failed to detect banned word: {word}"
            )
            # Must fail narrative verification
            metrics = {"unit_lift_pct": 20.0, "sample_size_n": 50}
            self.assertFalse(verify_narrative(text, metrics))

    def test_allowed_financial_words_accepted(self):
        allowed_text = (
            "- **Promotional Performance:** Observed sales volume lift of 25.0% and unit lift of 15.0% "
            "at realized price across 50 active households."
        )
        self.assertIsNone(BANNED_FINANCIAL_PATTERN.search(allowed_text))

    def test_banned_causal_assertions_detected(self):
        causal_samples = [
            "- **Finding:** The discount caused customers to purchase 25.0% more volume.",
            "- **Finding:** Households stockpiled in their pantries following the 15.0% price drop.",
            "- **Finding:** Shoppers stored in the pantry 50 units.",
            "- **Finding:** Consumers physically stockpiled extra items.",
        ]
        metrics = {"unit_lift_pct": 25.0, "sample_size_n": 50}
        for sample in causal_samples:
            matched = any(p.search(sample) for p in BANNED_CAUSAL_PATTERNS)
            self.assertTrue(matched, f"Failed to detect banned causal claim in: {sample}")
            self.assertFalse(verify_narrative(sample, metrics))


class TestAINumericalAudit(unittest.TestCase):
    """Tests post-processing numerical extraction and cross-referencing."""

    def setUp(self):
        self.metrics = {
            "product_id": 5569230,
            "commodity_desc": "SOFT DRINKS",
            "unit_lift_pct": 125.4,
            "sales_lift_pct": 98.2,
            "avg_days_to_first_post_purchase": 17.5,
            "interpurchase_day_change": -3.2,
            "post_repurchase_rate_pct": 22.0,
            "post_vs_during_dip_pct": 34.5,
            "sample_size_n": 250,
            "insufficient_data": False,
        }

    def test_number_extraction(self):
        text = "- Observed unit lift of +125.4% and sales lift of 98.2% across 250 households with return in 17.5 days."
        nums = extract_numbers_from_text(text)
        self.assertIn(125.4, nums)
        self.assertIn(98.2, nums)
        self.assertIn(250.0, nums)
        self.assertIn(17.5, nums)

    def test_valid_numbers_pass_audit(self):
        valid_text = (
            "- **Promotional Performance:** SOFT DRINKS (Product ID 5569230) recorded an observed unit lift "
            "of +125.4% across 250 active households.\n"
            "- **Repurchase Timing:** Promotional buyers returned in an average of 17.5 days with an "
            "observed regular-price repurchase rate of 22.0%."
        )
        self.assertTrue(verify_narrative(valid_text, self.metrics))

    def test_unverified_invented_number_fails_audit(self):
        # Invented year '2024' and invented lift '55.5%' not in metrics
        hallucinated_text = (
            "- **Promotional Performance:** In 2024, SOFT DRINKS recorded an observed unit lift "
            "of +55.5% across 250 active households.\n"
            "- **Repurchase Timing:** Promotional buyers returned in an average of 17.5 days."
        )
        self.assertFalse(verify_narrative(hallucinated_text, self.metrics))

    def test_sign_and_rounding_tolerance(self):
        allowed = extract_allowed_numbers_from_dict({"interpurchase_day_change": -3.2})
        # -3.2 and 3.2 shift should both be permitted
        self.assertTrue(is_number_allowed(-3.2, allowed))
        self.assertTrue(is_number_allowed(3.2, allowed))
        # 10.0 should not be permitted
        self.assertFalse(is_number_allowed(10.0, allowed))


class TestAIDeterministicTemplate(unittest.TestCase):
    """Tests fallback template output format, clarity, and safety."""

    def setUp(self):
        self.metrics = {
            "product_id": 5569230,
            "commodity_desc": "SOFT DRINKS",
            "unit_lift_pct": 125.4,
            "sales_lift_pct": 98.2,
            "avg_days_to_first_post_purchase": 17.5,
            "interpurchase_day_change": -3.2,
            "post_repurchase_rate_pct": 22.0,
            "trial_retention_rate_pct": 14.5,
            "post_vs_during_dip_pct": 34.5,
            "post_vs_pre_unit_pct": 8.0,
            "sample_size_n": 250,
            "insufficient_data": False,
        }

    def test_template_structure_and_bullet_count(self):
        narrative = generate_template_narrative(self.metrics)
        self.assertIsInstance(narrative, str)
        lines = [l.strip() for l in narrative.split("\n") if l.strip()]
        self.assertIn(len(lines), [2, 3])
        for line in lines:
            self.assertTrue(line.startswith("- "), f"Line does not start with bullet: {line}")

    def test_template_contains_core_fields(self):
        narrative = generate_template_narrative(self.metrics)
        self.assertIn("SOFT DRINKS", narrative)
        self.assertIn("5569230", narrative)
        self.assertIn("125.4%", narrative)
        self.assertIn("17.5", narrative)
        self.assertIn("250", narrative)

    def test_template_passes_own_guardrail_audits(self):
        narrative = generate_template_narrative(self.metrics)
        self.assertTrue(verify_narrative(narrative, self.metrics))


class TestAIEndToEndExecution(unittest.TestCase):
    """Tests generate_insight_narrative complete workflow."""

    def test_fallback_when_no_credentials(self):
        metrics = {
            "commodity_desc": "FROZEN PIZZA",
            "unit_lift_pct": 45.0,
            "sales_lift_pct": 30.0,
            "avg_days_to_first_post_purchase": 14.0,
            "post_repurchase_rate_pct": 18.0,
            "sample_size_n": 100,
            "insufficient_data": False,
        }
        with patch("app.ai._get_api_credentials", return_value=(None, None)):
            result = generate_insight_narrative(metrics)
            self.assertTrue(result.startswith("- **Promotional Performance:"))
            self.assertIn("FROZEN PIZZA", result)
            self.assertIn("45.0%", result)
            self.assertTrue(verify_narrative(result, metrics))

    def test_discards_hallucinated_candidate_and_uses_template(self):
        metrics = {
            "commodity_desc": "COOKIES",
            "unit_lift_pct": 60.0,
            "avg_days_to_first_post_purchase": 10.0,
            "sample_size_n": 80,
            "insufficient_data": False,
        }
        # Simulated LLM hallucinating unverified figures and profit claim
        bad_llm_response = (
            "- **Promotional Performance:** COOKIES had a 60.0% lift in 2023 delivering higher profit margins.\n"
            "- **Timing:** Return was 10.0 days."
        )
        with patch("app.ai._get_api_credentials", return_value=("gemini", "fake-key")), \
             patch("app.ai._call_gemini_api", return_value=bad_llm_response):
            result = generate_insight_narrative(metrics)
            # Must discard the bad LLM response and fall back to template
            self.assertNotIn("2023", result)
            self.assertNotIn("profit", result.lower())
            self.assertIn("COOKIES", result)
            self.assertTrue(verify_narrative(result, metrics))

    def test_accepts_verified_candidate_from_llm(self):
        metrics = {
            "commodity_desc": "BAG SNACKS",
            "unit_lift_pct": 80.0,
            "sales_lift_pct": 70.0,
            "avg_days_to_first_post_purchase": 12.0,
            "post_repurchase_rate_pct": 25.0,
            "sample_size_n": 120,
            "insufficient_data": False,
        }
        good_llm_response = (
            "- **Promotional Performance:** BAG SNACKS exhibited an observed unit lift of +80.0% "
            "and sales volume lift of +70.0% across 120 active households.\n"
            "- **Repurchase Timing:** Shopper response patterns show promotional buyers returned "
            "in an average of 12.0 calendar days with an observed regular-price repurchase rate of 25.0%."
        )
        with patch("app.ai._get_api_credentials", return_value=("gemini", "fake-key")), \
             patch("app.ai._call_gemini_api", return_value=good_llm_response):
            result = generate_insight_narrative(metrics)
            self.assertEqual(result, good_llm_response)


if __name__ == "__main__":
    unittest.main()

