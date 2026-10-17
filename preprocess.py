"""
preprocess.py
-------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Teammate 2: Streamlit & Visualizations Lead

One-time offline preprocessing pipeline to build the compact Parquet data mart
for the Streamlit MVP application.

Reuses data_preparation.prepare_enriched_transactions() without duplicating join
or causal-aggregation logic.
"""

import os
import sys
import time
import argparse
from typing import Optional, Union
import data_preparation as dp


PROD_OUTPUT_PARQUET = os.path.join(dp.DATA_DIR, "enriched_transactions.parquet")
DEFAULT_OUTPUT_PARQUET = PROD_OUTPUT_PARQUET
DEFAULT_SAMPLE_OUTPUT_PARQUET = os.path.join(dp.DATA_DIR, "sample_enriched_transactions.parquet")


def is_production_path(path: str) -> bool:
    """Checks whether the given path targets the production Parquet data mart."""
    target_abs = os.path.abspath(os.path.normpath(path))
    prod_abs = os.path.abspath(os.path.normpath(PROD_OUTPUT_PARQUET))

    if os.path.normcase(target_abs) == os.path.normcase(prod_abs):
        return True
    if os.path.normcase(os.path.realpath(target_abs)) == os.path.normcase(os.path.realpath(prod_abs)):
        return True
    if os.path.exists(target_abs) and os.path.exists(prod_abs):
        try:
            if os.path.samefile(target_abs, prod_abs):
                return True
        except (OSError, ValueError):
            pass
    return False


def resolve_output_path(output_arg: Optional[str] = None, sample_size: Optional[int] = None) -> str:
    """Determines the target Parquet output path based on whether sample_size is provided.

    Rules:
    ------
    1. If output_arg is explicitly supplied, returns its absolute path.
    2. If output_arg is None and sample_size is supplied, defaults to
       DEFAULT_SAMPLE_OUTPUT_PARQUET (database/sample_enriched_transactions.parquet).
    3. If output_arg is None and sample_size is None, defaults to
       PROD_OUTPUT_PARQUET (database/enriched_transactions.parquet).
    """
    if output_arg:
        return os.path.abspath(output_arg)
    if sample_size is not None:
        return os.path.abspath(DEFAULT_SAMPLE_OUTPUT_PARQUET)
    return os.path.abspath(PROD_OUTPUT_PARQUET)


def validate_output_path(output_path: str, sample_size: Optional[int] = None) -> None:
    """Enforces safety rules to prevent sample datasets from overwriting the production data mart.

    Raises:
    -------
    ValueError:
        If sample_size is specified and output_path points to the production data mart.
    """
    if sample_size is not None and is_production_path(output_path):
        raise ValueError(
            f"Safety Violation: Cannot target production data mart with a sample run.\n"
            f"  Target Output:  {output_path}\n"
            f"  Production:     {os.path.abspath(PROD_OUTPUT_PARQUET)}\n"
            f"  Sample Size:    {sample_size:,} transaction rows\n\n"
            f"Overwriting the production data mart with a partial sample would corrupt downstream\n"
            f"visualizations and analytics in the Streamlit application. Sample runs are strictly\n"
            f"prohibited from overwriting the production file, even when --force is supplied.\n\n"
            f"To run a sample, please either:\n"
            f"  1. Omit --output to write to the default sample path:\n"
            f"     {DEFAULT_SAMPLE_OUTPUT_PARQUET}\n"
            f"  2. Explicitly provide a custom sample destination path, e.g.:\n"
            f"     --output database/test_sample.parquet"
        )


def parse_arguments(args=None):
    epilog_text = (
        "USAGE NOTE:\n"
        "  This preprocessing script is designed to run ONCE as an offline preparation step.\n"
        "  Because processing raw causal (~664 MB) and transaction (~135 MB) CSVs streams\n"
        "  millions of records, it takes time and should NEVER be run dynamically during\n"
        "  each Streamlit page rerun.\n"
        "  Once database/enriched_transactions.parquet is created, app.py loads it in < 1 second.\n\n"
        "SAFETY PROTECTION:\n"
        "  Sample runs (--sample-size N) automatically default to saving in:\n"
        "    database/sample_enriched_transactions.parquet\n"
        "  and are strictly prevented from writing to the production path\n"
        "    database/enriched_transactions.parquet\n"
        "  even if --force is provided.\n\n"
        "EXAMPLES:\n"
        "  # 1. Full-data production run (outputs to database/enriched_transactions.parquet):\n"
        "  python3 preprocess.py              # Safe: prompts if production parquet already exists\n"
        "  python3 preprocess.py --force      # Overwrite existing production parquet\n\n"
        "  # 2. Fast sample / dry-run (safely defaults to database/sample_enriched_transactions.parquet):\n"
        "  python3 preprocess.py --sample-size 100000          # Process 100k rows (Weeks 1-12) to sample file\n"
        "  python3 preprocess.py --sample-size 100000 --force  # Overwrite sample file safely\n"
        "  python3 preprocess.py --sample-size 100000 --output database/my_sample.parquet  # Custom sample\n"
    )

    parser = argparse.ArgumentParser(
        description=(
            "One-time offline preprocessing pipeline for Grocery Store Stockpiling / Promotion Analytics.\n"
            "Executes validated joins and causal promotion classification via data_preparation.py\n"
            "and exports a compact, high-speed Parquet data mart for the Streamlit MVP."
        ),
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing Parquet file at the target output path if it already exists."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process only the first N transaction rows (useful for fast verification / dry-runs).\n"
            "When set, defaults output to database/sample_enriched_transactions.parquet and prevents\n"
            "writing to the production file database/enriched_transactions.parquet."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            f"Custom destination path for Parquet file.\n"
            f"Defaults to '{PROD_OUTPUT_PARQUET}' for full runs, or\n"
            f"'{DEFAULT_SAMPLE_OUTPUT_PARQUET}' when --sample-size is specified.\n"
            f"Note: Sample runs cannot target the production path, even with --force."
        )
    )

    return parser.parse_args(args)


def verify_raw_inputs():
    """Verifies that all three required raw CSV files are physically present."""
    required_files = {
        "Product Catalog": dp.PRODUCT_FILE,
        "Transaction Data": dp.TRANSACTION_FILE,
        "Causal Promotion Data": dp.CAUSAL_FILE,
    }

    missing = []
    for label, path in required_files.items():
        if not os.path.exists(path):
            missing.append((label, path))

    return missing


def main(args=None):
    parsed_args = parse_arguments(args)
    output_path = resolve_output_path(parsed_args.output, parsed_args.sample_size)

    print("=" * 70)
    print("GROCERY PROMOTION & STOCKPILING EXPLORER — DATA PREPROCESSING")
    print("=" * 70)
    if parsed_args.sample_size is not None:
        print(f"Run Mode:            SAMPLE RUN ({parsed_args.sample_size:,} transaction rows)")
    else:
        print("Run Mode:            FULL DATASET RUN (All transaction rows)")
    print(f"Target Output Path:  {output_path}")

    # Enforce safety check before proceeding to file checks or pipeline execution
    try:
        validate_output_path(output_path, parsed_args.sample_size)
    except ValueError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    # 1. Verify expected raw input files exist
    missing_files = verify_raw_inputs()
    if missing_files:
        print("\n[ERROR] Missing required raw input dataset(s):")
        for label, path in missing_files:
            print(f"  - {label}: '{path}' NOT FOUND")
        print("\nPlease ensure the raw CSV files are placed in the database/ directory before running.")
        sys.exit(1)

    print("\n[CHECK] Verified required raw input files:")
    print(f"  - Product Catalog:   {dp.PRODUCT_FILE} ({os.path.getsize(dp.PRODUCT_FILE):,} bytes)")
    print(f"  - Transactions:      {dp.TRANSACTION_FILE} ({os.path.getsize(dp.TRANSACTION_FILE):,} bytes)")
    print(f"  - Causal Promotion:  {dp.CAUSAL_FILE} ({os.path.getsize(dp.CAUSAL_FILE):,} bytes)")

    # 2. Check for existing output file unless --force is specified
    if os.path.exists(output_path) and not parsed_args.force:
        file_bytes = os.path.getsize(output_path)
        file_mb = file_bytes / (1024 * 1024)
        print(f"\n[NOTICE] Output file already exists: {output_path}")
        print(f"         Existing size: {file_bytes:,} bytes (~{file_mb:.2f} MB)")
        print("\nTo re-run the preprocessing pipeline and overwrite this file, pass the --force flag:")
        if parsed_args.sample_size is not None:
            print(f"  python3 preprocess.py --sample-size {parsed_args.sample_size} --force")
        else:
            print("  python3 preprocess.py --force")
        print("\nNo changes made. Exiting safely.")
        sys.exit(0)

    # 3. Create destination directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 4. Execute the validated pipeline from data_preparation.py
    if parsed_args.sample_size:
        print(f"\n[EXECUTE] Running data_preparation pipeline with sample_size={parsed_args.sample_size:,}...")
    else:
        print("\n[EXECUTE] Running full data_preparation pipeline across all transactions...")

    start_total = time.time()
    try:
        enriched_df = dp.prepare_enriched_transactions(sample_size=parsed_args.sample_size)
    except Exception as e:
        print(f"\n[ERROR] Data preparation pipeline failed: {e}")
        sys.exit(1)

    prep_time = time.time() - start_total
    num_rows, num_cols = enriched_df.shape
    print(f"\n[SUCCESS] Enriched transactions generated in {prep_time:.2f}s: {num_rows:,} rows x {num_cols} columns.")

    # 5. Export to compressed Parquet using pyarrow engine and snappy compression
    print(f"\n[EXPORT] Writing to Parquet (engine='pyarrow', compression='snappy')...")
    start_save = time.time()
    try:
        enriched_df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to save Parquet file: {e}")
        sys.exit(1)

    save_time = time.time() - start_save
    total_time = time.time() - start_total

    file_bytes = os.path.getsize(output_path)
    file_mb = file_bytes / (1024 * 1024)

    # 6. Detailed summary report
    print("=" * 70)
    print("PREPROCESSING COMPLETE — SUMMARY REPORT")
    print("=" * 70)
    if parsed_args.sample_size is not None:
        print(f"Run Type:        Sample Run ({parsed_args.sample_size:,} rows limit)")
    else:
        print("Run Type:        Full Dataset Run")
    print(f"Output File:     {output_path}")
    print(f"Total Rows:      {num_rows:,}")
    print(f"Total Columns:   {num_cols}")
    print(f"File Size:       {file_bytes:,} bytes ({file_mb:.2f} MB)")
    print(f"Compression:     Snappy")
    print(f"Parquet Engine:  PyArrow")
    print(f"Pipeline Time:   {prep_time:.2f} seconds")
    print(f"Write Time:      {save_time:.2f} seconds")
    print(f"Total Elapsed:   {total_time:.2f} seconds")
    print("=" * 70)
    print("The Streamlit app (app.py) can now load this Parquet file directly via @st.cache_data.")


if __name__ == "__main__":
    main()
