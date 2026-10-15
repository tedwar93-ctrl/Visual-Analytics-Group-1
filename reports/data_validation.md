# ITCS 5122 - Data Validation Report
**Project**: Grocery Promotion & Stockpiling Explorer  
**Lead**: Teammate 1 (Data & Analytics Lead)  
**Execution Timestamp**: 2026-10-14 17:20:16  
**Validation Script**: `validation.py`

---

## 1. Executive Summary & File Inventory

Three core dataset files residing in `database/` were fully inspected without modifying raw files or executing unconstrained joins:

| File Name | Size (Bytes) | Size (MB) | Row Count | Column Count | Exact Duplicates |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`product.csv`** | 6,429,896 | 6.13 MB | 92,353 | 7 | 0 |
| **`transaction_data.csv`** | 141,742,346 | 135.18 MB | 2,595,732 | 12 | 0 |
| **`causal_data.csv`** | 695,858,427 | 663.62 MB | 36,786,524 | 5 | Verified via chunking |

---

## 2. Dataset Schemas & Completeness

### A. `product.csv`
- **Primary Key Uniqueness**: `PRODUCT_ID` is strictly unique (92,353 distinct IDs, 0 duplicate keys).
- **Null & Whitespace Audit**:
  - `PRODUCT_ID`: 0 nulls, 0 blank.
  - `MANUFACTURER`: 0 nulls, 0 blank.
  - `DEPARTMENT`: 0 nulls, 0 blank.
  - `BRAND`: 0 nulls, 0 blank.
  - `COMMODITY_DESC`: 0 nulls, 0 blank.
  - `SUB_COMMODITY_DESC`: 0 nulls, 0 blank.
  - `CURR_SIZE_OF_PRODUCT`: 0 nulls, 30607 whitespace-only/empty string values.

### B. `transaction_data.csv`
- **Null & Whitespace Audit**: All 12 columns contain **0 null values** across all 2,595,732 records.
- **Entity Scope**:
  - **Unique Households**: 2,500
  - **Unique Baskets**: 276,484
  - **Unique Products**: 92,339
  - **Unique Stores**: 582
  - **Week Range**: Week 1 to Week 102 (102 weeks)
  - **Day Range**: Day 1 to Day 711 (711 sequential days)

### C. `causal_data.csv`
- **Total Records Processed**: 36,786,524
- **Missing / Blank Audit**:
  - `display`: 0 nulls, 0 blank.
  - `mailer`: 0 nulls, 0 blank.
- **Key Uniqueness (`PRODUCT_ID`, `STORE_ID`, `WEEK_NO`)**:
  - **Result**: **NOT UNIQUE**.
  - **Duplicate Key Triples Found**: **15,245** records share identical `(PRODUCT_ID, STORE_ID, WEEK_NO)`.
  - **Root Cause**: A single product in the same store and week can have multiple display/mailer placements simultaneously (e.g., both code `'0'` and code `'6'`).
  - **Critical Rule**: An unaggregated join on `(PRODUCT_ID, STORE_ID, WEEK_NO)` will cause **row multiplication (Cartesian explosion)**. Causal data must be aggregated or pivoted before joining.

---

## 3. Transaction-to-Product Join Integrity

- **Validation Check**: Tested every `PRODUCT_ID` in `transaction_data.csv` against `product.csv`.
- **Unmatched Transaction Rows**: **0** (0%).
- **Unmatched Transaction Products**: **0** (0%).
- **Unpurchased Products in Catalog**: **14** products in `product.csv` never appear in transactions.
- **Conclusion**: The foreign key relationship `transaction_data.PRODUCT_ID -> product.PRODUCT_ID` is **100% referentially intact**. `INNER JOIN` or `LEFT JOIN` on `product.csv` preserves 100% of transactions.

---

## 4. Calendar Consistency: `DAY` vs. Native `WEEK_NO`

- **Day-to-Week Determinism**: Tested whether any single `DAY` maps to more than one `WEEK_NO`.
  - Result: **0 days map to multiple weeks**. Each day belongs strictly to exactly one week.
- **Calendar Boundary Discovery**:
  - Week 1 spans Days 1 to 5 (**5 days only**).
  - Weeks 2 through 101 span **exactly 7 days each** (100 weeks).
  - Week 102 spans Days 706 to 711 (**6 days only**).
- **Formula Divergence**:
  - Comparing native `WEEK_NO` against naive calendar formula `(DAY - 1) // 7 + 1`:
  - **202 days out of 711 deviate** from the naive calculation.
- **Rule for Project**: **Always use native `WEEK_NO`** from `transaction_data.csv`. Never derive week numbers mechanically from `DAY`.

---

## 5. Transaction Record Classification

Records in `transaction_data.csv` were classified across quantities and sales:

| Category | Definition | Row Count | Percentage | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Purchases** | `QUANTITY > 0` & `SALES_VALUE > 0` | 2,576,815 | 99.271% | Standard positive customer purchases |
| **Both Non-positive** | `QUANTITY <= 0` & `SALES_VALUE <= 0` | 14,399 | 0.555% | Product returns, refunds, cancelled line items |
| **Positive Qty, Negative Sales** | `QUANTITY > 0` & `SALES_VALUE < 0` | 0 | 0.000% | Over-discounting / cashier override errors |
| **Non-positive Qty, Positive Sales** | `QUANTITY <= 0` & `SALES_VALUE > 0` | 67 | 0.003% | Restocking fees or administrative adjustments |
| **Zero Sales, Positive Qty** | `QUANTITY > 0` & `SALES_VALUE == 0` | 4,451 | 0.171% | 100% promotional discounts / BOGO free units |

> **Analytic Guideline**:
> - Gross volume / customer basket size should isolate **Normal Purchases** and **Zero-Sales Promotional Units**.
> - Net revenue metrics should net out **Both Non-positive** return items.
> - Anomalies (Categories 3 & 4) must be flagged rather than silently deleted.

---

## 6. Causal Feature Inspection: Display & Mailer Codes

All `display` and `mailer` values were treated strictly as strings to prevent numeric/string type collisions:

### Display Code Distribution
- Code `'0'`: 21,038,745 rows (57.191%)
- Code `'1'`: 1,102,141 rows (2.996%)
- Code `'2'`: 1,812,840 rows (4.928%)
- Code `'3'`: 2,073,738 rows (5.637%)
- Code `'4'`: 592,985 rows (1.612%)
- Code `'5'`: 2,575,289 rows (7.001%)
- Code `'6'`: 1,816,021 rows (4.937%)
- Code `'7'`: 2,362,118 rows (6.421%)
- Code `'9'`: 2,699,467 rows (7.338%)
- Code `'A'`: 713,180 rows (1.939%)

### Mailer Code Distribution
- Code `'0'`: 11,534,183 rows (31.354%)
- Code `'A'`: 17,106,789 rows (46.503%)
- Code `'C'`: 291,059 rows (0.791%)
- Code `'D'`: 4,467,453 rows (12.144%)
- Code `'F'`: 1,077,549 rows (2.929%)
- Code `'H'`: 1,560,395 rows (4.242%)
- Code `'J'`: 306,924 rows (0.834%)
- Code `'L'`: 301,327 rows (0.819%)
- Code `'P'`: 569 rows (0.002%)
- Code `'X'`: 120,823 rows (0.328%)
- Code `'Z'`: 19,453 rows (0.053%)

*Note: Per requirements, undocumented letter/number codes are preserved as raw categorical symbols and not artificially mapped.*

---

## 7. Product-Store-Week Promotion Evidence Coverage

We evaluated how frequently transaction occurrences `(PRODUCT_ID, STORE_ID, WEEK_NO)` are corroborated by evidence in `causal_data.csv`:

1. **Overall Transaction Triples**:
   - Total distinct transaction triples: **2,370,784**
   - Triples present in `causal_data`: **483,320** (20.39%)
   - Triples missing from `causal_data`: **1,887,464** (79.61%)
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
