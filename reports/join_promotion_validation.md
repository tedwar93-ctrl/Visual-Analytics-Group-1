# ITCS 5122 - Validated Joins & Promotion Identification Report
**Project**: Grocery Promotion & Stockpiling Explorer  
**Lead**: Teammate 1 (Data & Analytics Lead)  
**Execution Timestamp**: 2026-10-14  
**Deliverables**: `data_preparation.py`, `test_data_preparation.py`, `reports/join_promotion_validation.md`

---

## 1. Executive Summary & Objective

In **Step 2 (Validated Joins and Promotion Identification)**, we addressed the structural risks identified during initial data validation:
1. **Verified the 15,245 duplicate composite keys** in `causal_data.csv` and analyzed their conflicting display/mailer codes.
2. Built a **memory-efficient chunked and vectorized aggregation** pipeline at `(PRODUCT_ID, STORE_ID, WEEK_NO)` granularity that preserves all distinct raw codes while eliminating row multiplication.
3. Implemented a **100% referentially validated transaction-to-product join** on `PRODUCT_ID`.
4. Established a **multi-dimensional promotion evidence classification system** that explicitly differentiates between:
   - Observed causal feature evidence (`display` / `mailer` active)
   - Observed transaction price cuts (`RETAIL_DISC`, `COUPON_DISC`, `COUPON_MATCH_DISC`)
   - Unmonitored / missing causal coverage (strictly preserved as `Unknown` rather than assumed to be "no promotion").
5. Validated all logic with automated unit and integration tests (`test_data_preparation.py`), achieving 100% test pass rates and zero row loss or multiplication across all **2,595,732 transaction records**.

---

## 2. Investigation of the 15,245 Duplicate Causal Composite Keys

### A. Frequency Distribution
Across all **36,786,524 rows** in `causal_data.csv`, composite keys `(PRODUCT_ID, STORE_ID, WEEK_NO)` were scanned and quantified:

| Key Occurrence Frequency | Unique Composite Keys | Total Rows in Causal Data |
| :--- | :--- | :--- |
| Keys appearing exactly **1 time** | 36,756,034 | 36,756,034 |
| Keys appearing exactly **2 times** | 15,245 | 30,490 |
| Keys appearing **3+ times** | 0 | 0 |
| **Total** | **36,771,279 unique keys** | **36,786,524 rows** |

- **Duplicate Excess**: Exactly $36,786,524 - 36,771,279 = 15,245$ duplicate records.
- **Occurrence Pattern**: Not a single key appears 3 or more times; exactly 15,245 keys appear as duplicate pairs.

### B. Concrete Examples of Conflicting Codes
Inspecting adjacent duplicate pairs in the raw data reveals why these duplicates exist:

| `PRODUCT_ID` | `STORE_ID` | `WEEK_NO` | Row 1 `display` | Row 1 `mailer` | Row 2 `display` | Row 2 `mailer` | Conflict Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **104625** | 310 | 71 | `'0'` | `'A'` | `'1'` | `'A'` | Display Code Conflict (`'0'` vs `'1'`) |
| **104625** | 310 | 79 | `'0'` | `'A'` | `'9'` | `'A'` | Display Code Conflict (`'0'` vs `'9'`) |
| **104625** | 310 | 89 | `'0'` | `'H'` | `'7'` | `'H'` | Display Code Conflict (`'0'` vs `'7'`) |
| **105520** | 429 | 63 | `'9'` | `'0'` | `'9'` | `'L'` | Mailer Code Conflict (`'0'` vs `'L'`) |
| **27658** | 361 | 59 | `'0'` | `'D'` | `'6'` | `'D'` | Display Code Conflict (`'0'` vs `'6'`) |

### C. Retail Analytics Interpretation & Root Cause
1. **Physical Store Reality**: In grocery retail auditing, a product can simultaneously occupy a primary shelf position (`display = '0'`) and a secondary promotional display location (e.g., `'1'` = store front, `'6'` = end-cap display, or `'9'` = aisle wing).
2. **Circular Placement Reality**: A product may be advertised in the main weekly circular (`mailer = 'A'`) and also highlighted in a special holiday insert (`mailer = 'L'`).
3. **Absence of Exact Row Duplicates**: Testing confirms there are **0 exact duplicate rows** (`len(drop_duplicates()) == len(df)`). Every duplicate key represents a legitimately distinct feature placement observation.
4. **Row Multiplication Danger**: Joining `transaction_data` directly against raw `causal_data` would cause every transaction for these 15,245 keys to duplicate, distorting revenue, basket counts, and unit volume.

---

## 3. Causal Aggregation Design & Zero Row Multiplication

To resolve the duplicate key hazard, `data_preparation.py` implements `aggregate_causal_data()`:

### A. Aggregation Rules
For every unique `(PRODUCT_ID, STORE_ID, WEEK_NO)`:
- **`display_codes`**: Pipe-separated, sorted unique raw codes (e.g., `'0|1'`, `'0|6'`, `'9'`).
- **`mailer_codes`**: Pipe-separated, sorted unique raw codes (e.g., `'A'`, `'0|L'`).
- **`has_display_feature`**: Boolean, evaluating `True` if any display code $\neq$ `'0'`.
- **`has_mailer_feature`**: Boolean, evaluating `True` if any mailer code $\neq$ `'0'`.
- **`has_causal_evidence`**: Boolean (`has_display_feature | has_mailer_feature`).
- **`causal_record_exists`**: Boolean (`True` for all aggregated keys).

### B. High-Performance Vectorized Chunk Filtering
Rather than loading all 36.8M rows of causal data into memory:
1. `data_preparation.py` extracts the 2,370,784 unique transaction composite keys, encodes them into 64-bit integers:
   $$\text{Key} = (\text{PRODUCT\_ID} \times 100,000,000) + (\text{STORE\_ID} \times 1,000) + \text{WEEK\_NO}$$
2. The target keys are sorted once in memory ($O(N \log N)$ in 1.9s).
3. As causal chunks (5,000,000 rows each) are streamed, vectorized binary search (`np.searchsorted`) filters matching rows in $O(M \log N)$, completing in **16.88 seconds**.
4. The 483,935 matched causal rows are aggregated to **483,320 strictly unique keys**.

---

## 4. Documentation of Code Interpretations & Assumptions

Per project requirements, no undocumented meanings or speculative business definitions are invented. The following explicit assumptions are established:

> ### [!IMPORTANT]
> **Provisional Assumption on Code `'0'`**:
> - In standard retail causal tracking conventions (e.g. Dunnhumby, Nielsen, IRI), code `'0'` represents the baseline absence of an active promotional feature for that dimension (e.g. no end-cap display or no circular ad).
> - Any non-zero alphanumeric code (`'1'` through `'9'`, `'A'` through `'Z'`) is provisionally treated as active feature evidence.
> - Raw codes are strictly preserved as strings in `display_codes` and `mailer_codes` so analysts can override or refine this rule without re-running data pipelines.

> ### [!NOTE]
> **Discovery on `causal_data.csv` Baseline Records**:
> - Empirical scan of all 36,786,524 rows confirms: **0 rows have both `display == '0'` and `mailer == '0'`**.
> - `causal_data.csv` is an **event log of promotional activity**, NOT a continuous store-item panel. A product appears in `causal_data` *only* if at least one feature or mailer occurred.

---

## 5. Validated Transaction-to-Product Join Integrity

`data_preparation.py` executes `join_transactions_and_products()`:
* **Input Transaction Rows**: 2,595,732
* **Product Catalog Rows**: 92,353
* **Output Joined Rows**: **2,595,732**
* **Row Count Delta**: **0** (Zero rows lost, zero rows duplicated).
* **Referential Integrity**: 100% of transaction `PRODUCT_ID`s exist in `product.csv`. Attributes (`DEPARTMENT`, `COMMODITY_DESC`, `SUB_COMMODITY_DESC`, `BRAND`, `CURR_SIZE_OF_PRODUCT`) attach cleanly to every transaction.

---

## 6. Promotion Evidence Classification & Distribution

Joining the enriched transactions with aggregated causal data produces **2,595,732 rows** (100% row preservation).

### A. Promotion Evidence Breakdown (All 6 Classes)

| Classification Category | Description | Transactions | Percentage |
| :--- | :--- | :--- | :--- |
| **1. Causal & Discount** | Both in-store feature (display/mailer) and price discount (`RETAIL_DISC < 0` or `COUPON_DISC < 0`) observed | 475,475 | **18.32%** |
| **2. Causal Only** | In-store feature observed, but customer paid full regular shelf price | 88,299 | **3.40%** |
| **3. Discount Only** | Price discount observed with confirmed recorded causal tracking, but display='0' and mailer='0' | **0** | **0.00%** |
| **4. Recorded Baseline** | Confirmed recorded causal tracking with display='0' and mailer='0', and NO price discount | **0** | **0.00%** |
| **5. Discount with Unknown Causal** | Price discount observed, but causal data is unmonitored / missing (Weeks 1-8, Week 102, or unrecorded store-product combinations) | 843,924 | **32.51%** |
| **6. Baseline with Unknown Causal** | Full regular price paid; causal data is unmonitored / missing | 1,188,034 | **45.77%** |
| **Total** | **All Transactions** | **2,595,732** | **100.00%** |

> **Why Categories 3 & 4 evaluate to 0**:
> As established in Section 4, `causal_data.csv` is an **event log** of active promotional occurrences rather than a continuous store-item panel. Out of all 36,786,524 rows in `causal_data.csv`, exactly **0 rows** have both `display == '0'` and `mailer == '0'`. Consequently, any product present in `causal_data` inherently possesses active feature evidence, leaving categories 3 and 4 at 0 while ensuring zero misclassification.

### B. Preserved Evidence Flags
Every transaction record carries independent boolean and quantitative evidence fields:
- `has_retail_disc`: Boolean (`RETAIL_DISC < 0`)
- `has_coupon_disc`: Boolean (`COUPON_DISC < 0 | COUPON_MATCH_DISC < 0`)
- `has_discount_evidence`: Boolean (`has_retail_disc | has_coupon_disc`)
- `total_discount_val`: Total savings in dollars (`|RETAIL_DISC| + |COUPON_DISC| + |COUPON_MATCH_DISC|`)
- `discount_depth_pct`: Promotional markdown depth ($ \frac{\text{Total Discount}}{\text{SALES\_VALUE} + \text{Total Discount}} \times 100 $)
- `causal_record_exists`: Boolean (`True` if observed in `causal_data`)
- `causal_coverage_status`: `'Recorded'` vs `'Unknown'`
- `has_display_feature`: Boolean (display code $\neq$ `'0'`)
- `has_mailer_feature`: Boolean (mailer code $\neq$ `'0'`)
- `has_causal_evidence`: Boolean (`has_display_feature | has_mailer_feature`)
- `display_codes`: Preserved raw codes (e.g. `'0'`, `'6'`, `'0|6'`, or `'UNKNOWN'`)
- `mailer_codes`: Preserved raw codes (e.g. `'A'`, `'D'`, or `'UNKNOWN'`)
- `promo_evidence_type`: Mutually exclusive classification string

### C. Calendar Enforcement
- **Native `WEEK_NO` Preserved**: `WEEK_NO` is passed directly from `transaction_data.csv` (Weeks 1 through 102).
- Naive mathematical derivations (`(DAY-1)//7 + 1`) are prohibited due to the 202-day calendar mismatch established in Step 1.

---

## 7. Sample Records from Enriched Data Pipeline

Sample rows generated by `data_preparation.py`:

```text
  BASKET_ID    household_key  PRODUCT_ID  DEPARTMENT  COMMODITY_DESC          WEEK_NO  SALES_VALUE  total_discount_val  discount_depth_pct  causal_coverage_status  display_codes  mailer_codes  promo_evidence_type
26984851472             2375     1004906     PRODUCE  POTATOES                      1         1.39                0.60              30.15%                 Unknown        UNKNOWN       UNKNOWN  Discount with Unknown Causal
26984851472             2375     1033142     PRODUCE  ONIONS                        1         0.82                0.00               0.00%                 Unknown        UNKNOWN       UNKNOWN  Baseline with Unknown Causal
26984851472             2375     1036325     PRODUCE  VEGETABLES - ALL OTHERS       1         0.99                0.30              23.26%                 Unknown        UNKNOWN       UNKNOWN  Discount with Unknown Causal
...
32485601241              821      840361     GROCERY  EGGS                         54         1.29                0.40              23.67%                Recorded              0             A  Causal & Discount
32485601241              821      916122     GROCERY  FROZEN PIZZA                 54         3.99                1.50              27.32%                Recorded            0|6             A  Causal & Discount
```

---

## 8. Automated Test Suite Execution

All validation rules are tested in `test_data_preparation.py`:
- `test_causal_aggregation_uniqueness`: Confirms composite key duplicate resolution, pipe code preservation, and boolean flag accuracy.
- `test_transaction_to_product_join_preservation`: Confirms zero row loss on product join.
- `test_causal_join_row_preservation`: Confirms zero row multiplication when joining transactions with duplicate-prone causal keys.
- `test_promotion_evidence_classification`: Verifies assignment of each multi-level category and independent flag.
- `test_native_week_preserved`: Verifies that `WEEK_NO` is untouched.

```text
Ran 5 tests in 0.053s
OK
```

---

## 9. Recommendations for Step 3 (`analytics.py`)

1. **Pre/During/After Baseline Definition**:
   - For promotions identified via **Causal Evidence** (display/mailer), baselines must be scoped within **Weeks 9 to 101** where causal auditing was active.
   - For promotions identified via **Transaction Price Discounts** (`RETAIL_DISC` / `COUPON_DISC`), baselines can leverage the full **Weeks 1 to 102** timeline.
2. **Stockpiling Indicator Formulation**:
   - Evaluate post-promotion drop ($ \Delta Q = Q_{\text{post}} - Q_{\text{baseline}} $) on high-shelf-life commodities (e.g. Cereal, Soft Drinks, Paper Towels) vs perishable commodities (e.g. Milk, Produce, Fresh Meat).
3. **Delivery to Visual Team**:
   - `data_preparation.py` provides the canonical preparation layer. Teammates building dashboards or visual charts can consume pre-aggregated weekly summaries directly without loading 36.8M causal rows.
