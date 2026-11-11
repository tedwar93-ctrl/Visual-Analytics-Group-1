"""
data_preparation.py
-------------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Teammate 1: Data & Analytics Lead

Step 2: Validated Joins and Promotion Identification
- Memory-efficient causal aggregation at (PRODUCT_ID, STORE_ID, WEEK_NO) granularity
- Preserves raw display/mailer codes and duplicate observations without row multiplication
- Validated join between transactions and products (zero row loss / zero row duplication)
- Multi-dimensional promotion evidence classification with separate boolean & quantitative flags
- Strictly uses native WEEK_NO (no derived formula from DAY)
- Efficient vectorized chunked processing of causal_data.csv (~696MB)
"""

import os
import sys
import time
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "database")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TRANSACTION_FILE = os.path.join(DATA_DIR, "transaction_data.csv")
PRODUCT_FILE = os.path.join(DATA_DIR, "product.csv")
CAUSAL_FILE = os.path.join(DATA_DIR, "causal_data.csv")


def encode_psw_key(pid_arr, store_arr, week_arr):
    """
    Encodes (PRODUCT_ID, STORE_ID, WEEK_NO) into a single unique 64-bit integer.
    Safe for signed int64: (PID * 100,000,000) + (STORE * 1,000) + WEEK.
    Max PID (~18M) yields ~1.8 * 10^15 << 9.22 * 10^18.
    """
    return (pid_arr.astype(np.int64) * 100000000) + (store_arr.astype(np.int64) * 1000) + week_arr.astype(np.int64)


def load_products(filepath=PRODUCT_FILE):
    """Loads and validates product.csv."""
    df_prod = pd.read_csv(filepath)
    # Ensure types
    df_prod["PRODUCT_ID"] = df_prod["PRODUCT_ID"].astype(np.int64)
    return df_prod


def load_transactions(filepath=TRANSACTION_FILE, nrows=None):
    """Loads and validates transaction_data.csv, strictly preserving native WEEK_NO."""
    df_trans = pd.read_csv(filepath, nrows=nrows)
    df_trans["PRODUCT_ID"] = df_trans["PRODUCT_ID"].astype(np.int64)
    df_trans["STORE_ID"] = df_trans["STORE_ID"].astype(np.int64)
    df_trans["WEEK_NO"] = df_trans["WEEK_NO"].astype(np.int64)
    df_trans["household_key"] = df_trans["household_key"].astype(np.int64)
    df_trans["BASKET_ID"] = df_trans["BASKET_ID"].astype(np.int64)
    return df_trans


def aggregate_causal_data(causal_filepath=CAUSAL_FILE, target_keys_sorted=None, chunk_size=5_000_000):
    """
    Reads causal_data.csv in chunks and aggregates records to (PRODUCT_ID, STORE_ID, WEEK_NO).
    
    If target_keys_sorted (sorted array of encoded int64 keys) is provided, filters
    causal records on the fly using binary search, achieving ~15-20s runtime with
    minimal memory overhead (~35MB).
    
    Aggregations per composite key:
    - display_codes: pipe-separated sorted distinct raw string codes (e.g. '0|6' or '1')
    - mailer_codes: pipe-separated sorted distinct raw string codes (e.g. 'A|D' or '0')
    - has_display_feature: boolean (provisional assumption: any code != '0')
    - has_mailer_feature: boolean (provisional assumption: any code != '0')
    - has_causal_evidence: boolean (has_display_feature | has_mailer_feature)
    - causal_record_exists: True
    """
    matched_chunks = []
    has_filter = target_keys_sorted is not None
    if has_filter:
        n_target = len(target_keys_sorted)

    chunk_dtypes = {
        "PRODUCT_ID": np.int64,
        "STORE_ID": np.int64,
        "WEEK_NO": np.int64,
        "display": str,
        "mailer": str
    }

    print(f"Reading and aggregating causal data from {causal_filepath}...")
    chunk_idx = 0
    t0 = time.time()
    
    for chunk in pd.read_csv(causal_filepath, chunksize=chunk_size, dtype=chunk_dtypes):
        chunk_idx += 1
        if has_filter:
            c_keys = encode_psw_key(
                chunk["PRODUCT_ID"].values,
                chunk["STORE_ID"].values,
                chunk["WEEK_NO"].values
            )
            pos = np.searchsorted(target_keys_sorted, c_keys)
            pos[pos == n_target] = 0
            mask = (target_keys_sorted[pos] == c_keys)
            if mask.sum() > 0:
                matched_chunks.append(chunk[mask])
        else:
            matched_chunks.append(chunk)

    if not matched_chunks:
        if has_filter and len(target_keys_sorted) > 0:
            target_arr = np.asarray(target_keys_sorted)
            if np.all((target_arr % 1000) < 9):
                print(
                    "Notice: Target transaction keys are all in Weeks 1–8. Causal promotion data starts at Week 9; "
                    "all sample transactions will be classified with Unknown causal coverage."
                )
            else:
                print("Warning: No matching causal records found.")
        else:
            print("Warning: No matching causal records found.")
        return pd.DataFrame(columns=[
            "PRODUCT_ID", "STORE_ID", "WEEK_NO",
            "display_codes", "mailer_codes",
            "has_display_feature", "has_mailer_feature",
            "has_causal_evidence", "causal_record_exists"
        ])

    raw_matched = pd.concat(matched_chunks, ignore_index=True)
    elapsed = time.time() - t0
    print(f"  Extracted {len(raw_matched):,} raw causal records in {elapsed:.2f}s.")

    # Aggregate by (PRODUCT_ID, STORE_ID, WEEK_NO)
    agg_df = raw_matched.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).agg(
        display_codes=("display", lambda s: "|".join(sorted(set(s)))),
        mailer_codes=("mailer", lambda s: "|".join(sorted(set(s)))),
        has_display_feature=("display", lambda s: any(x != "0" for x in s)),
        has_mailer_feature=("mailer", lambda s: any(x != "0" for x in s))
    ).reset_index()

    agg_df["has_causal_evidence"] = agg_df["has_display_feature"] | agg_df["has_mailer_feature"]
    agg_df["causal_record_exists"] = True

    print(f"  Aggregated to {len(agg_df):,} unique (PRODUCT_ID, STORE_ID, WEEK_NO) composite keys.")
    return agg_df


def join_transactions_and_products(df_trans, df_prod):
    """
    Performs a validated join between transactions and products on PRODUCT_ID.
    Verifies that no rows are lost and no rows are duplicated.
    """
    initial_rows = len(df_trans)
    merged = df_trans.merge(df_prod, on="PRODUCT_ID", how="left")
    
    if len(merged) != initial_rows:
        raise ValueError(
            f"Join validation failed! Initial transaction rows: {initial_rows:,}, "
            f"Joined rows: {len(merged):,} (Difference: {len(merged) - initial_rows:,})"
        )
    return merged


def classify_promotion_evidence(df_trans_prod, df_causal_agg):
    """
    Merges aggregated causal data into transaction-product table and classifies
    promotion evidence.
    
    Distinguishes:
    - Observed causal evidence (display/mailer feature != '0')
    - Observed transaction discounts (RETAIL_DISC < 0, COUPON_DISC < 0, COUPON_MATCH_DISC < 0)
    - Recorded no-evidence cases (causal record exists with display='0' and mailer='0', and no discount)
    - Missing/unknown causal coverage (no causal record exists for that product-store-week)
    
    Maintains distinct evidence flags to avoid conflating missing coverage with 'no promotion'.
    """
    initial_rows = len(df_trans_prod)
    
    # Left join on composite key
    merged = df_trans_prod.merge(
        df_causal_agg,
        on=["PRODUCT_ID", "STORE_ID", "WEEK_NO"],
        how="left"
    )
    
    if len(merged) != initial_rows:
        raise ValueError(
            f"Causal join validation failed! Expected {initial_rows:,} rows, got {len(merged):,} rows."
        )

    # 1. Causal presence flags
    merged["causal_record_exists"] = merged["causal_record_exists"].eq(True)
    merged["causal_coverage_status"] = np.where(merged["causal_record_exists"], "Recorded", "Unknown")
    
    merged["display_codes"] = merged["display_codes"].fillna("UNKNOWN")
    merged["mailer_codes"] = merged["mailer_codes"].fillna("UNKNOWN")
    merged["has_display_feature"] = merged["has_display_feature"].eq(True)
    merged["has_mailer_feature"] = merged["has_mailer_feature"].eq(True)
    merged["has_causal_evidence"] = merged["has_causal_evidence"].eq(True)

    # 2. Transaction discount flags
    merged["has_retail_disc"] = merged["RETAIL_DISC"] < 0
    merged["has_coupon_disc"] = (merged["COUPON_DISC"] < 0) | (merged["COUPON_MATCH_DISC"] < 0)
    merged["has_discount_evidence"] = merged["has_retail_disc"] | merged["has_coupon_disc"]

    # Quantitative discount metrics
    merged["retail_disc_val"] = merged["RETAIL_DISC"].abs()
    merged["coupon_disc_val"] = merged["COUPON_DISC"].abs() + merged["COUPON_MATCH_DISC"].abs()
    merged["total_discount_val"] = merged["retail_disc_val"] + merged["coupon_disc_val"]

    # Discount depth: total_discount / (SALES_VALUE + total_discount)
    base_price = merged["SALES_VALUE"] + merged["total_discount_val"]
    merged["discount_depth_pct"] = np.where(
        base_price > 0,
        (merged["total_discount_val"] / base_price) * 100.0,
        0.0
    )

    # 3. Multi-level Evidence Classification
    # Conditions:
    cond_causal_and_disc = merged["has_causal_evidence"] & merged["has_discount_evidence"]
    cond_causal_only = merged["has_causal_evidence"] & (~merged["has_discount_evidence"])
    cond_disc_only = (~merged["has_causal_evidence"]) & merged["causal_record_exists"] & merged["has_discount_evidence"]
    cond_recorded_baseline = (~merged["has_causal_evidence"]) & merged["causal_record_exists"] & (~merged["has_discount_evidence"])
    cond_disc_unknown_causal = (~merged["causal_record_exists"]) & merged["has_discount_evidence"]
    cond_baseline_unknown_causal = (~merged["causal_record_exists"]) & (~merged["has_discount_evidence"])

    conditions = [
        cond_causal_and_disc,
        cond_causal_only,
        cond_disc_only,
        cond_recorded_baseline,
        cond_disc_unknown_causal,
        cond_baseline_unknown_causal
    ]
    choices = [
        "Causal & Discount",
        "Causal Only",
        "Discount Only",
        "Recorded Baseline",
        "Discount with Unknown Causal",
        "Baseline with Unknown Causal"
    ]
    merged["promo_evidence_type"] = pd.Categorical(
        np.select(conditions, choices, default="Other"),
        categories=choices
    )

    return merged


def prepare_enriched_transactions(sample_size=None):
    """
    Runs the complete validated join and promotion identification pipeline.
    Returns the enriched transactions DataFrame.
    """
    print("=" * 60)
    print("STEP 2: PREPARING ENRICHED TRANSACTIONS")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load products
    df_prod = load_products()
    print(f"Loaded {len(df_prod):,} products.")

    # 2. Load transactions
    df_trans = load_transactions(nrows=sample_size)
    print(f"Loaded {len(df_trans):,} transactions.")

    # 3. Create sorted target keys for fast causal filtering
    pids = df_trans["PRODUCT_ID"].values.astype(np.int64)
    stores = df_trans["STORE_ID"].values.astype(np.int64)
    weeks = df_trans["WEEK_NO"].values.astype(np.int64)
    target_keys_sorted = np.sort(np.unique(encode_psw_key(pids, stores, weeks)))
    print(f"Identified {len(target_keys_sorted):,} unique product-store-week keys in transactions.")

    # 4. Aggregate causal data
    causal_agg = aggregate_causal_data(target_keys_sorted=target_keys_sorted)

    # 5. Join transactions to products
    df_tp = join_transactions_and_products(df_trans, df_prod)
    print(f"Validated transaction-to-product join: {len(df_tp):,} rows intact.")

    # 6. Classify promotion evidence
    enriched = classify_promotion_evidence(df_tp, causal_agg)
    print(f"Promotion evidence classification complete: {len(enriched):,} rows intact.")

    elapsed = time.time() - t0
    print(f"Full data preparation completed in {elapsed:.2f} seconds.")
    return enriched


if __name__ == "__main__":
    enriched_df = prepare_enriched_transactions()
    print("\nPromotion Evidence Summary:")
    print(enriched_df["promo_evidence_type"].value_counts(dropna=False))
    print("\nSample records (first 3 rows):")
    sample_cols = [
        "BASKET_ID", "household_key", "PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC",
        "WEEK_NO", "SALES_VALUE", "total_discount_val", "discount_depth_pct",
        "causal_coverage_status", "display_codes", "mailer_codes", "promo_evidence_type"
    ]
    print(enriched_df[sample_cols].head(3).to_string())
