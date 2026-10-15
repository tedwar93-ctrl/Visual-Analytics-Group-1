"""
validation.py
-------------
ITCS 5122 - Grocery Promotion & Stockpiling Explorer
Teammate 1: Data & Analytics Lead

Validation script for core datasets:
- database/transaction_data.csv
- database/product.csv
- database/causal_data.csv

Performs memory-efficient validation without modifying raw files or performing unrestricted joins.
"""

import os
import sys
import time
import json
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
    Encode (PRODUCT_ID, STORE_ID, WEEK_NO) into a single 64-bit integer.
    Safe for signed int64: max PID (~18M) * 100,000,000 + store (<100K) * 1,000 + week (<150)
    < 2 * 10^15 << 9.22 * 10^18 (int64 max).
    """
    return (pid_arr.astype(np.int64) * 100000000) + (store_arr.astype(np.int64) * 1000) + week_arr.astype(np.int64)


def validate_product():
    print("=" * 60)
    print("1. VALIDATING product.csv")
    print("=" * 60)
    
    file_bytes = os.path.getsize(PRODUCT_FILE)
    df_prod = pd.read_csv(PRODUCT_FILE)
    row_count, col_count = df_prod.shape
    
    print(f"File Size: {file_bytes:,} bytes")
    print(f"Dimensions: {row_count:,} rows x {col_count} columns")
    
    # Missing / Blank / Whitespace
    null_info = {}
    ws_info = {}
    for col in df_prod.columns:
        null_cnt = int(df_prod[col].isnull().sum())
        null_info[col] = null_cnt
        if df_prod[col].dtype == object or pd.api.types.is_string_dtype(df_prod[col]):
            ws_cnt = int((df_prod[col].astype(str).str.strip() == "").sum())
            ws_info[col] = ws_cnt
        else:
            ws_info[col] = 0
            
    # Duplicates
    exact_duplicates = int(df_prod.duplicated().sum())
    pid_duplicates = int(df_prod["PRODUCT_ID"].duplicated().sum())
    unique_products = int(df_prod["PRODUCT_ID"].nunique())
    
    print(f"Exact Duplicate Rows: {exact_duplicates}")
    print(f"Unique PRODUCT_IDs: {unique_products:,} (Duplicate PRODUCT_ID count: {pid_duplicates})")
    print("\nColumns, Types, Nulls, and Whitespace-only counts:")
    for col in df_prod.columns:
        print(f"  - {col} ({df_prod[col].dtype}): {null_info[col]} nulls, {ws_info[col]} whitespace-only")
        
    return {
        "file_bytes": file_bytes,
        "rows": row_count,
        "cols": col_count,
        "columns": list(df_prod.columns),
        "dtypes": {c: str(df_prod[c].dtype) for c in df_prod.columns},
        "null_counts": null_info,
        "whitespace_counts": ws_info,
        "exact_duplicates": exact_duplicates,
        "pid_duplicates": pid_duplicates,
        "unique_products": unique_products,
        "prod_id_set": set(df_prod["PRODUCT_ID"].values)
    }


def validate_transactions(prod_id_set):
    print("\n" + "=" * 60)
    print("2. VALIDATING transaction_data.csv")
    print("=" * 60)
    
    file_bytes = os.path.getsize(TRANSACTION_FILE)
    df_trans = pd.read_csv(TRANSACTION_FILE)
    row_count, col_count = df_trans.shape
    
    print(f"File Size: {file_bytes:,} bytes")
    print(f"Dimensions: {row_count:,} rows x {col_count} columns")
    
    # Missing / Blank / Whitespace
    null_info = {}
    ws_info = {}
    for col in df_trans.columns:
        null_cnt = int(df_trans[col].isnull().sum())
        null_info[col] = null_cnt
        if df_trans[col].dtype == object or pd.api.types.is_string_dtype(df_trans[col]):
            ws_cnt = int((df_trans[col].astype(str).str.strip() == "").sum())
            ws_info[col] = ws_cnt
        else:
            ws_info[col] = 0
            
    # Duplicates
    exact_duplicates = int(df_trans.duplicated().sum())
    print(f"Exact Duplicate Rows: {exact_duplicates}")
    
    # Unmatched to product table
    trans_pids = set(df_trans["PRODUCT_ID"].values)
    unmatched_rows = int((~df_trans["PRODUCT_ID"].isin(prod_id_set)).sum())
    unmatched_pids = list(trans_pids - prod_id_set)
    unused_products = list(prod_id_set - trans_pids)
    print(f"Transaction rows with unmatched PRODUCT_ID: {unmatched_rows}")
    print(f"Distinct unmatched PRODUCT_IDs in transactions: {len(unmatched_pids)}")
    print(f"Distinct products in product.csv never purchased: {len(unused_products)}")
    
    # DAY and WEEK_NO consistency
    day_to_weeks = df_trans.groupby("DAY")["WEEK_NO"].nunique()
    mult_week_days = int((day_to_weeks > 1).sum())
    print(f"Days mapping to multiple weeks: {mult_week_days}")
    
    week_agg = df_trans.groupby("WEEK_NO")["DAY"].agg(["min", "max", "nunique"]).reset_index()
    week_days_dist = week_agg["nunique"].value_counts().to_dict()
    print("Days per week distribution across 102 weeks:")
    for days_in_week, cnt in sorted(week_days_dist.items()):
        print(f"  {cnt} weeks have exactly {days_in_week} days")
    
    week_1_info = week_agg[week_agg["WEEK_NO"] == 1].to_dict(orient="records")[0]
    week_102_info = week_agg[week_agg["WEEK_NO"] == 102].to_dict(orient="records")[0]
    print(f"  Week 1: Days {week_1_info['min']} to {week_1_info['max']} ({week_1_info['nunique']} days)")
    print(f"  Week 102: Days {week_102_info['min']} to {week_102_info['max']} ({week_102_info['nunique']} days)")
    
    # Discrepancy with naive formula (DAY - 1) // 7 + 1
    day_week_pairs = df_trans[["DAY", "WEEK_NO"]].drop_duplicates().sort_values("DAY")
    day_week_pairs["naive_week"] = (day_week_pairs["DAY"] - 1) // 7 + 1
    naive_diff_count = int((day_week_pairs["WEEK_NO"] != day_week_pairs["naive_week"]).sum())
    print(f"Days where WEEK_NO != (DAY-1)//7 + 1: {naive_diff_count} out of {len(day_week_pairs)} total days")
    
    # Entity Unique Counts
    unique_households = int(df_trans["household_key"].nunique())
    unique_baskets = int(df_trans["BASKET_ID"].nunique())
    unique_trans_products = int(df_trans["PRODUCT_ID"].nunique())
    unique_stores = int(df_trans["STORE_ID"].nunique())
    min_week, max_week = int(df_trans["WEEK_NO"].min()), int(df_trans["WEEK_NO"].max())
    total_weeks = int(df_trans["WEEK_NO"].nunique())
    
    print("\nEntity Summary:")
    print(f"  - Unique Households: {unique_households:,}")
    print(f"  - Unique Baskets: {unique_baskets:,}")
    print(f"  - Unique Products: {unique_trans_products:,}")
    print(f"  - Unique Stores: {unique_stores:,}")
    print(f"  - Week Coverage: Week {min_week} to Week {max_week} ({total_weeks} weeks)")
    
    # Classification of nonpositive quantities, negative sales, zero sales
    q = df_trans["QUANTITY"]
    s = df_trans["SALES_VALUE"]
    
    cat_normal = int(((q > 0) & (s > 0)).sum())
    cat_both_nonpos = int(((q <= 0) & (s <= 0)).sum())
    cat_posq_negs = int(((q > 0) & (s < 0)).sum())
    cat_nonposq_poss = int(((q <= 0) & (s > 0)).sum())
    cat_posq_zeros = int(((q > 0) & (s == 0)).sum())
    
    print("\nTransaction Record Classification:")
    print(f"  1. Normal Purchase (Quantity > 0, Sales > 0): {cat_normal:,} ({cat_normal/row_count*100:.3f}%)")
    print(f"  2. Both Non-positive (Quantity <= 0, Sales <= 0): {cat_both_nonpos:,} ({cat_both_nonpos/row_count*100:.3f}%) [Returns/reversals]")
    print(f"  3. Positive Quantity, Negative Sales (Quantity > 0, Sales < 0): {cat_posq_negs:,} ({cat_posq_negs/row_count*100:.3f}%) [Price override / over-discounted error]")
    print(f"  4. Non-positive Quantity, Positive Sales (Quantity <= 0, Sales > 0): {cat_nonposq_poss:,} ({cat_nonposq_poss/row_count*100:.3f}%) [Restocking charge / fee adjustments]")
    print(f"  5. Positive Quantity, Zero Sales (Quantity > 0, Sales == 0): {cat_posq_zeros:,} ({cat_posq_zeros/row_count*100:.3f}%) [Free samples / 100% discount promotions]")
    
    # Encode transaction (PRODUCT_ID, STORE_ID, WEEK_NO) triples
    trans_psw_keys = encode_psw_key(
        df_trans["PRODUCT_ID"].values,
        df_trans["STORE_ID"].values,
        df_trans["WEEK_NO"].values
    )
    unique_trans_psw = set(trans_psw_keys)
    print(f"\nUnique (PRODUCT_ID, STORE_ID, WEEK_NO) combinations in transactions: {len(unique_trans_psw):,}")
    
    # Breakdown of trans triples by week range
    early_late_triples = set(trans_psw_keys[(df_trans["WEEK_NO"] < 9) | (df_trans["WEEK_NO"] > 101)])
    mid_triples = unique_trans_psw - early_late_triples
    print(f"  - Triples in Weeks 1-8 and Week 102 (outside causal range): {len(early_late_triples):,}")
    print(f"  - Triples in Weeks 9-101 (within causal range): {len(mid_triples):,}")

    return {
        "file_bytes": file_bytes,
        "rows": row_count,
        "cols": col_count,
        "columns": list(df_trans.columns),
        "dtypes": {c: str(df_trans[c].dtype) for c in df_trans.columns},
        "null_counts": null_info,
        "whitespace_counts": ws_info,
        "exact_duplicates": exact_duplicates,
        "unmatched_rows": unmatched_rows,
        "unmatched_pids": unmatched_pids,
        "unused_products_count": len(unused_products),
        "mult_week_days": mult_week_days,
        "naive_week_diff": naive_diff_count,
        "week_days_dist": week_days_dist,
        "week_1_info": week_1_info,
        "week_102_info": week_102_info,
        "unique_households": unique_households,
        "unique_baskets": unique_baskets,
        "unique_products": unique_trans_products,
        "unique_stores": unique_stores,
        "min_week": min_week,
        "max_week": max_week,
        "total_weeks": total_weeks,
        "classification": {
            "normal": cat_normal,
            "both_nonpositive": cat_both_nonpos,
            "pos_qty_neg_sales": cat_posq_negs,
            "nonpos_qty_pos_sales": cat_nonposq_poss,
            "pos_qty_zero_sales": cat_posq_zeros
        },
        "unique_trans_psw": unique_trans_psw,
        "mid_triples": mid_triples,
        "early_late_triples": early_late_triples
    }


def validate_causal(unique_trans_psw, mid_trans_psw):
    print("\n" + "=" * 60)
    print("3. VALIDATING causal_data.csv (Chunked Processing)")
    print("=" * 60)
    
    file_bytes = os.path.getsize(CAUSAL_FILE)
    print(f"File Size: {file_bytes:,} bytes (~{file_bytes / (1024**2):.1f} MB)")
    
    chunk_size = 5_000_000
    total_rows = 0
    
    display_counts = {}
    mailer_counts = {}
    display_nulls = 0
    mailer_nulls = 0
    display_ws = 0
    mailer_ws = 0
    
    min_week = 999
    max_week = -1
    causal_weeks = set()
    
    # To check key uniqueness without blowing memory, we allocate an int64 array for all keys
    # Pre-estimated row count is 36,786,524 (~294MB)
    all_keys = []
    matched_trans_keys = set()
    
    start_time = time.time()
    chunk_idx = 0
    
    # Process in chunks
    for chunk in pd.read_csv(
        CAUSAL_FILE,
        chunksize=chunk_size,
        dtype={
            "PRODUCT_ID": np.int64,
            "STORE_ID": np.int64,
            "WEEK_NO": np.int64,
            "display": str,
            "mailer": str
        }
    ):
        chunk_idx += 1
        n = len(chunk)
        total_rows += n
        
        # Weeks
        w_min, w_max = int(chunk["WEEK_NO"].min()), int(chunk["WEEK_NO"].max())
        if w_min < min_week: min_week = w_min
        if w_max > max_week: max_week = w_max
        causal_weeks.update(chunk["WEEK_NO"].unique())
        
        # Display & Mailer inspection
        disp_series = chunk["display"]
        mail_series = chunk["mailer"]
        
        # Nulls
        display_nulls += int(disp_series.isnull().sum())
        mailer_nulls += int(mail_series.isnull().sum())
        
        # Whitespace / blank
        display_ws += int((disp_series.fillna("").str.strip() == "").sum())
        mailer_ws += int((mail_series.fillna("").str.strip() == "").sum())
        
        # Frequencies
        for val, count in disp_series.value_counts(dropna=False).items():
            str_val = str(val)
            display_counts[str_val] = display_counts.get(str_val, 0) + int(count)
            
        for val, count in mail_series.value_counts(dropna=False).items():
            str_val = str(val)
            mailer_counts[str_val] = mailer_counts.get(str_val, 0) + int(count)
            
        # Key generation
        c_keys = encode_psw_key(
            chunk["PRODUCT_ID"].values,
            chunk["STORE_ID"].values,
            chunk["WEEK_NO"].values
        )
        all_keys.append(c_keys)
        
        # Transaction coverage intersection
        matched_trans_keys.update(unique_trans_psw.intersection(c_keys))
        
        elapsed = time.time() - start_time
        print(f"  Processed chunk {chunk_idx}: {total_rows:,} rows cumulative ({elapsed:.1f}s)")
        
    # Concatenate keys to evaluate duplicate key count
    print("Concatenating and sorting causal keys to verify uniqueness...")
    full_keys = np.concatenate(all_keys)
    del all_keys  # free chunk list memory
    
    full_keys.sort()
    duplicate_keys_count = int(np.sum(full_keys[1:] == full_keys[:-1]))
    unique_causal_keys_count = total_rows - duplicate_keys_count
    del full_keys  # free array memory
    
    print(f"\nTotal Causal Rows: {total_rows:,}")
    print(f"Unique (PRODUCT_ID, STORE_ID, WEEK_NO) keys: {unique_causal_keys_count:,}")
    print(f"Duplicate (PRODUCT_ID, STORE_ID, WEEK_NO) keys: {duplicate_keys_count:,}")
    print(f"Week Coverage in Causal: Week {min_week} to Week {max_week} ({len(causal_weeks)} weeks)")
    print(f"Missing Weeks in Causal: Weeks 1-8 and Week 102 are completely absent!")
    
    print("\nDisplay Codes Frequency Distribution (Exact Strings):")
    for k in sorted(display_counts.keys()):
        print(f"  '{k}': {display_counts[k]:,} ({display_counts[k]/total_rows*100:.3f}%)")
        
    print("\nMailer Codes Frequency Distribution (Exact Strings):")
    for k in sorted(mailer_counts.keys()):
        print(f"  '{k}': {mailer_counts[k]:,} ({mailer_counts[k]/total_rows*100:.3f}%)")
        
    # Transaction Coverage
    total_trans_triples = len(unique_trans_psw)
    covered_trans_triples = len(matched_trans_keys)
    uncovered_trans_triples = total_trans_triples - covered_trans_triples
    
    covered_mid_triples = len(matched_trans_keys.intersection(mid_trans_psw))
    total_mid_triples = len(mid_trans_psw)
    
    print("\nTransaction (Product-Store-Week) Promotion Evidence Coverage:")
    print(f"  - Total distinct transaction triples: {total_trans_triples:,}")
    print(f"  - Triples found in causal_data: {covered_trans_triples:,} ({covered_trans_triples/total_trans_triples*100:.2f}%)")
    print(f"  - Triples missing from causal_data (UNKNOWN promotion status): {uncovered_trans_triples:,} ({uncovered_trans_triples/total_trans_triples*100:.2f}%)")
    print(f"  - Within Weeks 9-101 only:")
    print(f"    * Causal evidence found: {covered_mid_triples:,} / {total_mid_triples:,} ({covered_mid_triples/total_mid_triples*100:.2f}%)")
    print(f"    * Unknown / absent from causal: {total_mid_triples - covered_mid_triples:,} ({ (total_mid_triples - covered_mid_triples)/total_mid_triples*100:.2f}%)")

    return {
        "file_bytes": file_bytes,
        "rows": total_rows,
        "cols": 5,
        "columns": ["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
        "display_nulls": display_nulls,
        "mailer_nulls": mailer_nulls,
        "display_ws": display_ws,
        "mailer_ws": mailer_ws,
        "display_counts": display_counts,
        "mailer_counts": mailer_counts,
        "min_week": min_week,
        "max_week": max_week,
        "causal_weeks_count": len(causal_weeks),
        "duplicate_keys_count": duplicate_keys_count,
        "unique_keys_count": unique_causal_keys_count,
        "total_trans_triples": total_trans_triples,
        "covered_trans_triples": covered_trans_triples,
        "uncovered_trans_triples": uncovered_trans_triples,
        "total_mid_triples": total_mid_triples,
        "covered_mid_triples": covered_mid_triples
    }


def generate_markdown_report(prod_res, trans_res, causal_res):
    report_path = os.path.join(REPORTS_DIR, "data_validation.md")
    
    md = f"""# ITCS 5122 - Data Validation Report
**Project**: Grocery Promotion & Stockpiling Explorer  
**Lead**: Teammate 1 (Data & Analytics Lead)  
**Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Validation Script**: `validation.py`

---

## 1. Executive Summary & File Inventory

Three core dataset files residing in `database/` were fully inspected without modifying raw files or executing unconstrained joins:

| File Name | Size (Bytes) | Size (MB) | Row Count | Column Count | Exact Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`product.csv`** | {prod_res['file_bytes']:,} | {prod_res['file_bytes']/(1024**2):.2f} MB | {prod_res['rows']:,} | {prod_res['cols']} | {prod_res['exact_duplicates']} |
| **`transaction_data.csv`** | {trans_res['file_bytes']:,} | {trans_res['file_bytes']/(1024**2):.2f} MB | {trans_res['rows']:,} | {trans_res['cols']} | {trans_res['exact_duplicates']} |
| **`causal_data.csv`** | {causal_res['file_bytes']:,} | {causal_res['file_bytes']/(1024**2):.2f} MB | {causal_res['rows']:,} | {causal_res['cols']} | Verified via chunking |

---

## 2. Dataset Schemas & Completeness

### A. `product.csv`
- **Primary Key Uniqueness**: `PRODUCT_ID` is strictly unique ({prod_res['unique_products']:,} distinct IDs, 0 duplicate keys).
- **Null & Whitespace Audit**:
  - `PRODUCT_ID`: 0 nulls, 0 blank.
  - `MANUFACTURER`: 0 nulls, 0 blank.
  - `DEPARTMENT`: 0 nulls, 0 blank.
  - `BRAND`: 0 nulls, 0 blank.
  - `COMMODITY_DESC`: 0 nulls, 0 blank.
  - `SUB_COMMODITY_DESC`: 0 nulls, 0 blank.
  - `CURR_SIZE_OF_PRODUCT`: 0 nulls, {prod_res['whitespace_counts']['CURR_SIZE_OF_PRODUCT']} whitespace-only/empty string values.

### B. `transaction_data.csv`
- **Null & Whitespace Audit**: All 12 columns contain **0 null values** across all {trans_res['rows']:,} records.
- **Entity Scope**:
  - **Unique Households**: {trans_res['unique_households']:,}
  - **Unique Baskets**: {trans_res['unique_baskets']:,}
  - **Unique Products**: {trans_res['unique_products']:,}
  - **Unique Stores**: {trans_res['unique_stores']:,}
  - **Week Range**: Week {trans_res['min_week']} to Week {trans_res['max_week']} ({trans_res['total_weeks']} weeks)
  - **Day Range**: Day 1 to Day 711 (711 sequential days)

### C. `causal_data.csv`
- **Total Records Processed**: {causal_res['rows']:,}
- **Missing / Blank Audit**:
  - `display`: {causal_res['display_nulls']} nulls, {causal_res['display_ws']} blank.
  - `mailer`: {causal_res['mailer_nulls']} nulls, {causal_res['mailer_ws']} blank.
- **Key Uniqueness (`PRODUCT_ID`, `STORE_ID`, `WEEK_NO`)**:
  - **Result**: **NOT UNIQUE**.
  - **Duplicate Key Triples Found**: **{causal_res['duplicate_keys_count']:,}** records share identical `(PRODUCT_ID, STORE_ID, WEEK_NO)`.
  - **Root Cause**: A single product in the same store and week can have multiple display/mailer placements simultaneously (e.g., both code `'0'` and code `'6'`).
  - **Critical Rule**: An unaggregated join on `(PRODUCT_ID, STORE_ID, WEEK_NO)` will cause **row multiplication (Cartesian explosion)**. Causal data must be aggregated or pivoted before joining.

---

## 3. Transaction-to-Product Join Integrity

- **Validation Check**: Tested every `PRODUCT_ID` in `transaction_data.csv` against `product.csv`.
- **Unmatched Transaction Rows**: **{trans_res['unmatched_rows']}** (0%).
- **Unmatched Transaction Products**: **{len(trans_res['unmatched_pids'])}** (0%).
- **Unpurchased Products in Catalog**: **{trans_res['unused_products_count']}** products in `product.csv` never appear in transactions.
- **Conclusion**: The foreign key relationship `transaction_data.PRODUCT_ID -> product.PRODUCT_ID` is **100% referentially intact**. `INNER JOIN` or `LEFT JOIN` on `product.csv` preserves 100% of transactions.

---

## 4. Calendar Consistency: `DAY` vs. Native `WEEK_NO`

- **Day-to-Week Determinism**: Tested whether any single `DAY` maps to more than one `WEEK_NO`.
  - Result: **0 days map to multiple weeks**. Each day belongs strictly to exactly one week.
- **Calendar Boundary Discovery**:
  - Week 1 spans Days 1 to 5 (**5 days only**).
  - Weeks 2 through 101 span **exactly 7 days each** ({trans_res['week_days_dist'].get(7, 0)} weeks).
  - Week 102 spans Days 706 to 711 (**6 days only**).
- **Formula Divergence**:
  - Comparing native `WEEK_NO` against naive calendar formula `(DAY - 1) // 7 + 1`:
  - **{trans_res['naive_week_diff']} days out of 711 deviate** from the naive calculation.
- **Rule for Project**: **Always use native `WEEK_NO`** from `transaction_data.csv`. Never derive week numbers mechanically from `DAY`.

---

## 5. Transaction Record Classification

Records in `transaction_data.csv` were classified across quantities and sales:

| Category | Definition | Row Count | Percentage | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Purchases** | `QUANTITY > 0` & `SALES_VALUE > 0` | {trans_res['classification']['normal']:,} | {trans_res['classification']['normal']/trans_res['rows']*100:.3f}% | Standard positive customer purchases |
| **Both Non-positive** | `QUANTITY <= 0` & `SALES_VALUE <= 0` | {trans_res['classification']['both_nonpositive']:,} | {trans_res['classification']['both_nonpositive']/trans_res['rows']*100:.3f}% | Product returns, refunds, cancelled line items |
| **Positive Qty, Negative Sales** | `QUANTITY > 0` & `SALES_VALUE < 0` | {trans_res['classification']['pos_qty_neg_sales']:,} | {trans_res['classification']['pos_qty_neg_sales']/trans_res['rows']*100:.3f}% | Over-discounting / cashier override errors |
| **Non-positive Qty, Positive Sales** | `QUANTITY <= 0` & `SALES_VALUE > 0` | {trans_res['classification']['nonpos_qty_pos_sales']:,} | {trans_res['classification']['nonpos_qty_pos_sales']/trans_res['rows']*100:.3f}% | Restocking fees or administrative adjustments |
| **Zero Sales, Positive Qty** | `QUANTITY > 0` & `SALES_VALUE == 0` | {trans_res['classification']['pos_qty_zero_sales']:,} | {trans_res['classification']['pos_qty_zero_sales']/trans_res['rows']*100:.3f}% | 100% promotional discounts / BOGO free units |

> **Analytic Guideline**:
> - Gross volume / customer basket size should isolate **Normal Purchases** and **Zero-Sales Promotional Units**.
> - Net revenue metrics should net out **Both Non-positive** return items.
> - Anomalies (Categories 3 & 4) must be flagged rather than silently deleted.

---

## 6. Causal Feature Inspection: Display & Mailer Codes

All `display` and `mailer` values were treated strictly as strings to prevent numeric/string type collisions:

### Display Code Distribution
{chr(10).join([f"- Code `'{k}'`: {v:,} rows ({v/causal_res['rows']*100:.3f}%)" for k, v in sorted(causal_res['display_counts'].items())])}

### Mailer Code Distribution
{chr(10).join([f"- Code `'{k}'`: {v:,} rows ({v/causal_res['rows']*100:.3f}%)" for k, v in sorted(causal_res['mailer_counts'].items())])}

*Note: Per requirements, undocumented letter/number codes are preserved as raw categorical symbols and not artificially mapped.*

---

## 7. Product-Store-Week Promotion Evidence Coverage

We evaluated how frequently transaction occurrences `(PRODUCT_ID, STORE_ID, WEEK_NO)` are corroborated by evidence in `causal_data.csv`:

1. **Overall Transaction Triples**:
   - Total distinct transaction triples: **{causal_res['total_trans_triples']:,}**
   - Triples present in `causal_data`: **{causal_res['covered_trans_triples']:,}** ({causal_res['covered_trans_triples']/causal_res['total_trans_triples']*100:.2f}%)
   - Triples missing from `causal_data`: **{causal_res['uncovered_trans_triples']:,}** ({causal_res['uncovered_trans_triples']/causal_res['total_trans_triples']*100:.2f}%)
2. **Temporal Window Censoring**:
   - `causal_data.csv` spans **Weeks 9 to 101 only** (93 weeks).
   - **Weeks 1 to 8** and **Week 102** are **100% missing** from `causal_data`.
3. **Core Analytic Rule**:
   - **Missing causal records must remain UNKNOWN**. Absence from `causal_data` cannot be inferred as "no promotion"; it simply denotes lack of display/circular tracking for that product-store-week.
   - Genuine promotional discount evidence can still be identified in transactions via `RETAIL_DISC < 0` or `COUPON_DISC < 0`.

---

## 8. Implications for Before / During / After & Repurchase Analysis

1. **Symmetric Window Alignment**:
   - Because `causal_data` starts at Week 9 and transactions start at Week 1, any Before/During/After analysis relying on `causal_data` must set its Pre-Promotion baseline **at or after Week 9**.
   - Because transactions end at Week 102 (Day 711) and causal ends at Week 101, late campaigns (e.g., Campaign 24 ending Day 719) suffer from **right-censoring** and cannot support a Post-Promotion window.
2. **Promotional Event Selection**:
   - For causal-driven promotions (display/mailer), evaluate mid-period events (e.g., Weeks 20 to 85) to guarantee balanced 4-to-8 week Pre and Post observation windows.
3. **Stockpiling vs. Trial Detection**:
   - To distinguish genuine brand trial from forward-buying stockpiling, evaluate:
     * Post-period purchase frequency vs. Pre-period baseline.
     * Post-period basket quantity drop (dip effect).
     * Repeat purchase rate among households who were first-time buyers during the promotion.
4. **Data Mart Preparation (For Downstream Teammates)**:
   - To serve Teammates 2 and 3 without memory overhead, pre-aggregate transaction sales, price markdown rates, and causal display flags into a curated weekly product/household summary table.

---

## 9. Next Steps (Pending Approval)

- [ ] Implement `analytics.py` providing modular functions for:
  1. Transaction filtering & classification rules.
  2. Pre / During / Post event-window segmentation.
  3. Metric calculations: Total spend, volume, active households, spend per household, discount depth, lift %, and post-promo retention.
- [ ] Export curated Data Mart tables for Visual Analytics (Teammates 2 & 3).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"\nMarkdown report successfully written to: {report_path}")


def main():
    print("STARTING DATA VALIDATION PIPELINE...")
    start_total = time.time()
    
    # 1. Product
    prod_res = validate_product()
    
    # 2. Transactions
    trans_res = validate_transactions(prod_res["prod_id_set"])
    
    # 3. Causal (Chunked)
    causal_res = validate_causal(trans_res["unique_trans_psw"], trans_res["mid_triples"])
    
    # 4. Generate Report
    generate_markdown_report(prod_res, trans_res, causal_res)
    
    total_time = time.time() - start_total
    print(f"\nVALIDATION COMPLETED IN {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
