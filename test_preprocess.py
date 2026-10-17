"""
test_preprocess.py
------------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Unit tests for preprocess.py output path resolution and safety protections.

Validates:
1. Default output path for full-data runs is database/enriched_transactions.parquet.
2. Default output path for sample runs is database/sample_enriched_transactions.parquet.
3. Custom output paths are respected for both full and sample runs.
4. Attempting to direct a sample run to the production output path is strictly rejected.
5. Protection holds regardless of path format (relative, normalized, redundant slashes).
6. --force does NOT bypass the safety check blocking sample runs from overwriting production.
7. Run mode and target path are displayed prior to pipeline execution.
8. No heavy data processing is executed during safety validation or unit tests.
"""

import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import preprocess


class TestPreprocessSafety(unittest.TestCase):

    def test_default_output_path_full_run(self):
        """When --sample-size is omitted, output must default to production path."""
        resolved = preprocess.resolve_output_path(output_arg=None, sample_size=None)
        expected = os.path.abspath(preprocess.PROD_OUTPUT_PARQUET)
        self.assertEqual(resolved, expected)

    def test_default_output_path_sample_run(self):
        """When --sample-size is supplied, output must default to sample path, NOT production path."""
        resolved = preprocess.resolve_output_path(output_arg=None, sample_size=50000)
        expected = os.path.abspath(preprocess.DEFAULT_SAMPLE_OUTPUT_PARQUET)
        prod_path = os.path.abspath(preprocess.PROD_OUTPUT_PARQUET)

        self.assertEqual(resolved, expected)
        self.assertNotEqual(resolved, prod_path)

    def test_custom_output_path_full_run(self):
        """When custom --output is provided without --sample-size, resolve to absolute custom path."""
        custom_path = os.path.join("custom_folder", "custom_prod.parquet")
        resolved = preprocess.resolve_output_path(output_arg=custom_path, sample_size=None)
        self.assertEqual(resolved, os.path.abspath(custom_path))

    def test_custom_output_path_sample_run(self):
        """When custom non-production --output is provided with --sample-size, resolve correctly."""
        custom_path = os.path.join("custom_folder", "custom_sample.parquet")
        resolved = preprocess.resolve_output_path(output_arg=custom_path, sample_size=25000)
        self.assertEqual(resolved, os.path.abspath(custom_path))

    def test_validate_output_path_allows_valid_scenarios(self):
        """Standard valid scenarios should not raise any exception."""
        # Full run targeting production path
        preprocess.validate_output_path(preprocess.PROD_OUTPUT_PARQUET, sample_size=None)
        # Sample run targeting default sample path
        preprocess.validate_output_path(preprocess.DEFAULT_SAMPLE_OUTPUT_PARQUET, sample_size=50000)
        # Sample run targeting custom non-production path
        preprocess.validate_output_path(os.path.abspath("test_sample.parquet"), sample_size=50000)

    def test_validate_output_path_rejects_sample_targeting_production(self):
        """Sample runs explicitly targeting production path must raise ValueError."""
        prod_path = preprocess.PROD_OUTPUT_PARQUET
        with self.assertRaises(ValueError) as ctx:
            preprocess.validate_output_path(prod_path, sample_size=50000)

        err_msg = str(ctx.exception)
        self.assertIn("Safety Violation", err_msg)
        self.assertIn("Cannot target production data mart with a sample run", err_msg)
        self.assertIn("50,000", err_msg)

    def test_validate_output_path_rejects_path_variations(self):
        """Path variations (relative, redundant slashes, dot-slash) must also be caught."""
        variations = [
            "database/enriched_transactions.parquet",
            "./database/enriched_transactions.parquet",
            "database//enriched_transactions.parquet",
            os.path.join(".", "database", "enriched_transactions.parquet"),
        ]
        for path_var in variations:
            with self.subTest(path=path_var):
                with self.assertRaises(ValueError) as ctx:
                    preprocess.validate_output_path(path_var, sample_size=10000)
                self.assertIn("Safety Violation", str(ctx.exception))

    def test_parse_arguments_defaults_and_options(self):
        """CLI arguments must be correctly parsed with appropriate defaults."""
        # Defaults
        args = preprocess.parse_arguments([])
        self.assertIsNone(args.sample_size)
        self.assertIsNone(args.output)
        self.assertFalse(args.force)

        # Sample and force
        args = preprocess.parse_arguments(["--sample-size", "50000", "--force"])
        self.assertEqual(args.sample_size, 50000)
        self.assertTrue(args.force)

        # Custom output
        args = preprocess.parse_arguments(["--output", "test.parquet"])
        self.assertEqual(args.output, "test.parquet")

    @patch("preprocess.dp.prepare_enriched_transactions")
    def test_main_cli_sample_targeting_production_exits_without_processing(self, mock_prepare):
        """main() must exit with status 1 and NOT invoke data preparation if sample targets prod."""
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            with self.assertRaises(SystemExit) as ctx:
                preprocess.main(["--sample-size", "50000", "--output", preprocess.PROD_OUTPUT_PARQUET])

        self.assertEqual(ctx.exception.code, 1)
        mock_prepare.assert_not_called()

        output = captured_out.getvalue()
        self.assertIn("SAMPLE RUN (50,000 transaction rows)", output)
        self.assertIn("Safety Violation", output)

    @patch("preprocess.dp.prepare_enriched_transactions")
    def test_main_cli_sample_targeting_production_with_force_still_blocked(self, mock_prepare):
        """--force must NEVER override the protection against sample overwriting production."""
        captured_out = io.StringIO()
        with patch("sys.stdout", captured_out):
            with self.assertRaises(SystemExit) as ctx:
                preprocess.main([
                    "--sample-size", "50000",
                    "--output", preprocess.PROD_OUTPUT_PARQUET,
                    "--force"
                ])

        self.assertEqual(ctx.exception.code, 1)
        mock_prepare.assert_not_called()

        output = captured_out.getvalue()
        self.assertIn("Safety Violation", output)
        self.assertIn("prohibited from overwriting the production file, even when --force is supplied", output)

    def test_main_displays_run_mode_and_target_path(self):
        """Verify that run mode and target path are clearly displayed at invocation."""
        # 1. Sample Run display
        captured_sample = io.StringIO()
        with patch("sys.stdout", captured_sample), \
             patch("preprocess.verify_raw_inputs", return_value=[("Dummy", "nonexistent")]), \
             self.assertRaises(SystemExit):
            preprocess.main(["--sample-size", "25000"])

        sample_out = captured_sample.getvalue()
        self.assertIn("Run Mode:            SAMPLE RUN (25,000 transaction rows)", sample_out)
        self.assertIn(preprocess.DEFAULT_SAMPLE_OUTPUT_PARQUET, sample_out)

        # 2. Full Run display
        captured_full = io.StringIO()
        with patch("sys.stdout", captured_full), \
             patch("preprocess.verify_raw_inputs", return_value=[("Dummy", "nonexistent")]), \
             self.assertRaises(SystemExit):
            preprocess.main([])

        full_out = captured_full.getvalue()
        self.assertIn("Run Mode:            FULL DATASET RUN (All transaction rows)", full_out)
        self.assertIn(preprocess.PROD_OUTPUT_PARQUET, full_out)


if __name__ == "__main__":
    unittest.main()
