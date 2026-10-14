# Spreadsheet Objective Test — Set 09

---

### 81. Question

A data analyst is auditing a shared workbook. She finds the following formulas in the same column, all intended to pull a customer's Tier from the Customers table (`Customers!$A$2:$D$51`, col 1=ID, col 2=Name, col 3=Region, col 4=Tier):

- **Row 2:** `=VLOOKUP(B2, Customers!$A$2:$D$51, 4, FALSE)` → returns "Gold"
- **Row 3:** `=VLOOKUP(B3, Customers!$A$2:$D$51, 4, FALSE)` → returns "Silver"
- **Row 4:** `=VLOOKUP(B4, Customers!A4:D53, 4, FALSE)` → returns "Gold" (suspicious)
- **Row 5:** `=VLOOKUP(B5, Customers!A5:D54, 4, FALSE)` → returns `#N/A`

Which of the following correctly explains the behavior in rows 4 and 5?

- [ ] **A.** Row 4 returns "Gold" coincidentally — the shifted range `A4:D53` happens to still contain the correct customer in its leftmost column, but it is searching a range that has drifted 2 rows down. Row 5 returns `#N/A` because the further-drifted range `A5:D54` no longer contains the lookup key.
- [ ] **B.** Row 4 and Row 5 both return errors — "Gold" is a fabricated value that VLOOKUP cannot return for a shifted range
- [ ] **C.** Rows 4 and 5 contain absolute references (`$`), so the range does not drift — the errors are from data quality issues only
- [ ] **D.** The `#N/A` in row 5 is caused by the `col_index_num` being 4, which is too high for a shifted range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Without `$` signs, `A2:D51` is a relative reference. In row 4, the formula shifts by 2 rows (relative to row 2): the range becomes `A4:D53`. The Customers table data starts at row 2, so rows 4–53 of the Customers sheet are still within the table — and the target customer's ID may still appear in that shifted range, producing an apparently correct result. In row 5, the range shifts to `A5:D54`. If the specific customer in B5 had their record in rows 2–4 of the Customers table (which are now outside the shifted range), VLOOKUP cannot find them, returning `#N/A`. This is the classic "sometimes right, sometimes wrong" behavior of un-locked range references — the subtle, dangerous kind of bug.
- **Option B (Incorrect):** A shifted range does not automatically cause errors. If the customer's row happens to be within the shifted range, VLOOKUP finds it and returns the correct value. Only when the shift pushes the customer's row outside the search range does `#N/A` appear. Row 4 returning "Gold" is plausible with a shifted but still-covering range.
- **Option C (Incorrect):** Rows 4 and 5 use `A4:D53` and `A5:D54` notation (no `$`), which are clearly relative references that have already shifted. There are no `$` signs present — this is the exact un-locked reference that causes the drift.
- **Option D (Incorrect):** `col_index_num = 4` is valid for a 4-column range (`A:D`) regardless of the starting row. Column index 4 is within bounds (≤ 4 columns). The `#N/A` is caused by the lookup key being outside the shifted search range, not by an invalid column index.

</details>

---

### 82. Question

A data analyst is reviewing a formula designed to count orders where **Amount is greater than a threshold stored in cell J1**. The formula she reviews is: `=COUNTIF(F2:F100, ">"&J1)`. Which of the following are true about this formula?

- [ ] **A.** `">"&J1` concatenates the greater-than operator string with the current value of J1, producing a dynamic criterion like `">50000"` when J1 contains 50000
- [ ] **B.** The formula produces a `#VALUE!` error because COUNTIF criteria must be hardcoded strings, not concatenated expressions
- [ ] **C.** When J1 changes (e.g., from 50000 to 75000), the formula automatically recalculates the count for the new threshold without any manual formula editing
- [ ] **D.** This technique (`"operator"&cell_reference`) works for SUMIFS and AVERAGEIFS criteria as well, not just COUNTIF

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C, D**

#### Explanations:
- **Option A (Correct):** The `&` operator is the text concatenation operator in spreadsheet formulas. `">"&J1` combines the string `">"` with the value of J1. When J1 = 50000, the effective criterion becomes `">50000"`, a valid comparison criterion string for COUNTIF/SUMIF-family functions. This is a standard pattern for dynamic thresholds.
- **Option B (Incorrect):** COUNTIF/SUMIF-family functions accept concatenated comparison criteria. `">"&J1` is not hardcoded text — it is a dynamic expression that evaluates to a valid criterion string at runtime. No `#VALUE!` error occurs. This is documented behavior in both Excel and Google Sheets.
- **Option C (Correct):** Since J1 is a cell reference (not locked with `$` in the criteria expression, but this is fine for recalculation purposes — the formula references J1 by its address), whenever J1's value changes, the spreadsheet's recalculation engine marks this formula as stale and recomputes it with the new concatenated criterion. The analyst can change J1 and immediately see the updated count.
- **Option D (Correct):** The `"operator"&cell_reference` pattern works in all SUMIF-family and COUNTIF-family functions: `SUMIF`, `SUMIFS`, `COUNTIF`, `COUNTIFS`, `AVERAGEIF`, and `AVERAGEIFS`. For example, `=SUMIFS(F2:F100, F2:F100, ">"&J1, C2:C100, "North")` sums F where F > J1 AND Region = "North".

</details>

---

### 83. Question

An analyst's colleague argues: "Since XLOOKUP can look to the left of the lookup column and doesn't require the search column to be leftmost, we should always use XLOOKUP instead of VLOOKUP." Evaluate this claim. Which statements are accurate?

- [ ] **A.** XLOOKUP is available in Excel 365/2019+ and recent versions of Google Sheets, but not in older Excel versions — so replacing all VLOOKUPs with XLOOKUP may break compatibility with colleagues using older software
- [ ] **B.** XLOOKUP removes the leftmost-column restriction of VLOOKUP, allowing the lookup column and return column to be any columns, specified independently
- [ ] **C.** XLOOKUP is always the better choice for all users in all situations — VLOOKUP should be considered obsolete and immediately replaced
- [ ] **D.** VLOOKUP still works correctly in all environments where it was previously used — migrating existing working formulas to XLOOKUP has a cost (testing, knowledge transfer) that must be weighed against XLOOKUP's benefits

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** XLOOKUP was introduced in Excel 365 (late 2019). Older Excel versions (2016, 2013, 2010) do not have XLOOKUP. A workbook using XLOOKUP opened in an older Excel version will display `#NAME?` errors for all XLOOKUP cells. In environments where colleagues use mixed Excel versions, this is a real compatibility concern.
- **Option B (Correct):** Unlike VLOOKUP, XLOOKUP's `lookup_array` and `return_array` arguments are independent ranges. The lookup array can be any column (including one to the right of the return column), removing the leftmost-column constraint entirely. This makes XLOOKUP structurally more flexible than VLOOKUP.
- **Option C (Incorrect):** "Always better" and "should be considered obsolete and immediately replaced" is an overly absolute claim. For teams using compatible software, XLOOKUP is often the better new choice. However, VLOOKUP remains functional and appropriate where: (a) backward compatibility is required; (b) existing formulas work correctly and migration cost is not justified; (c) the team is already trained in VLOOKUP. Technology decisions involve trade-off analysis, not absolute rules.
- **Option D (Correct):** Pragmatic advice: working VLOOKUP formulas don't need to be migrated just because XLOOKUP exists. The migration has real costs — formula auditing, testing, training — and benefits only materialize when VLOOKUP's limitations (leftmost column rule, error handling) are actually causing problems. A phased or need-driven migration approach is more practical.

</details>

---

### 84. Question

A data analyst is debugging why `=SUMIFS(F2:F100, C2:C100, "North", D2:D100, "Completed")` returns a value that seems too low. After investigation she finds: some Status values in column D are stored as "completed" (all lowercase) rather than "Completed". How does SUMIFS handle this?

- [ ] **A.** SUMIFS is case-sensitive, so "completed" does not match the criterion "Completed" — those rows are excluded, explaining the lower-than-expected total
- [ ] **B.** SUMIFS is case-insensitive, so "completed" matches "Completed" — the case difference is not the reason for the lower total; she should look elsewhere for the cause
- [ ] **C.** SUMIFS returns `#VALUE!` when it encounters mixed-case data in a criteria range
- [ ] **D.** SUMIFS treats "completed" and "Completed" as different values and counts both separately

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** SUMIFS is **case-insensitive** in both Excel and Google Sheets. "completed", "Completed", "COMPLETED" all match the criterion "Completed". Unlike programming languages where string equality is often case-sensitive, spreadsheet functions that accept text criteria perform case-insensitive comparisons. The lowercase values are not being excluded.
- **Option B (Correct):** Since SUMIFS treats text criteria case-insensitively, all rows with "completed" (lowercase) in column D match the criterion "Completed" just as well as rows with "Completed". The case variation is definitively not causing the lower total. The analyst should investigate other possible causes: Are some rows within the range F2:F100 but their Region or Status values have leading/trailing spaces? Are some qualifying rows below row 100 (outside the formula range)? Are some Amount values stored as text?
- **Option C (Incorrect):** SUMIFS does not error on mixed-case data. It silently treats all case variants of a text value as equivalent when evaluating criteria. `#VALUE!` from SUMIFS typically occurs from range size mismatches or invalid argument types, not from text case variations.
- **Option D (Incorrect):** SUMIFS does not "count separately" — it aggregates by summing, not counting. More importantly, it treats "completed" and "Completed" as the same match for the criterion "Completed", so both are included in the same aggregation, not separated.

</details>

---

### 85. Question

A business analyst creates a pivot table from an enriched Orders sheet with these fields: OrderID, Region, Segment, Status, Amount, CustomerName. She places:
- Region → Rows
- Segment → Columns  
- Amount → Values (Sum)
- Status → Filters

She then creates a second pivot table from the same data on a separate sheet, with just:
- Region → Rows
- Amount → Values (Count)

She references both pivot tables' results from a Dashboard sheet using direct cell references. Which of the following are TRUE about the risks and behaviors of this setup?

- [ ] **A.** The Count of Amount in the second pivot table counts the number of orders per region (non-blank Amount cells), not the sum of amounts
- [ ] **B.** Both pivot tables require a manual refresh after each Orders data update; neither updates automatically
- [ ] **C.** Cell references from the Dashboard to specific pivot cells in both tables may silently return wrong values if either pivot table's layout changes (e.g., a new segment column appears in the first table, or regions are sorted differently in the second)
- [ ] **D.** The Status filter in the first pivot table affects the second pivot table as well, since both are sourced from the same data range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C**

#### Explanations:
- **Option A (Correct):** When a numeric field like Amount is placed in the Values area and the aggregation is "Count" (not "Sum"), the pivot table counts the number of non-blank Amount cells per group — effectively counting the number of orders per region. This is different from summing the amounts. "Count of Amount" = count of orders; "Sum of Amount" = total revenue.
- **Option B (Correct):** Both pivot tables require explicit manual refresh (or are configured to refresh on workbook open) to reflect data changes. Unlike formula cells, pivot tables are not part of the live recalculation engine — they cache computation results until refreshed.
- **Option C (Correct):** Both cross-sheet references from the Dashboard to pivot table cells are hard-coded addresses. Any layout change — the first pivot table gaining a new Segment column (shifting column C's content), or the second pivot table having regions re-sorted after a refresh — changes what data occupies the referenced cells, causing the Dashboard to silently display wrong values.
- **Option D (Incorrect):** Each pivot table has its own independent filter configuration. The Status filter applied to the first pivot table does not propagate to the second pivot table. Both are independently configured. This is a common misconception — pivot tables sourced from the same data range are separate objects with separate filter states, not synchronized.

</details>

---

### 86. Question

A senior analyst has this formula that is misbehaving: `=SUMIF(C2:C100, "Enterprise", D2:D100)`. She expects a large total revenue for Enterprise orders, but the formula returns 0. Column C holds the Segment field and Column D holds the Amount field. She confirms there are Enterprise rows in the data. Which of the following could be causing the 0 result?

- [ ] **A.** The Segment values in column C are stored with extra spaces: " Enterprise" (with a leading space) rather than "Enterprise"
- [ ] **B.** The Amount values in column D are text-formatted numbers (stored as text) rather than true numeric values
- [ ] **C.** The criteria range (C2:C100) and the sum range (D2:D100) have the same number of rows, which causes SUMIF to fail
- [ ] **D.** Column C uses mixed case: some rows have "enterprise" (lowercase) which doesn't match "Enterprise"

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** Leading or trailing spaces in text values are invisible but prevent exact string matching. " Enterprise" (with a leading space) is a different text value from "Enterprise". SUMIF with criterion "Enterprise" finds zero matches when all Segment cells contain " Enterprise", returning 0. Using `TRIM()` on column C data or checking with `LEN(C2)` reveals unexpected extra characters.
- **Option B (Correct):** If column D contains text-formatted numbers (imported as strings, left-aligned, possibly with a green triangle warning in Excel), SUMIF can identify the matching rows in column C, but when it goes to sum the corresponding D values, it sums text — text contributes 0 to a numeric sum. The formula finds the correct rows but returns 0 because the "numbers" are actually stored as text strings. Converting D to numeric values with `VALUE()` or the "Convert to Number" operation fixes this.
- **Option C (Incorrect):** SUMIF requires the criteria range and sum range to be the same size (same number of rows and columns). C2:C100 and D2:D100 are both 99-row ranges — they match. Having equal-sized ranges is correct behavior, not an error. SUMIF would produce a `#VALUE!` error if the ranges were different sizes, not a 0 result.
- **Option D (Incorrect):** SUMIF is **case-insensitive**. "enterprise" (lowercase) matches the criterion "Enterprise" just as well as "Enterprise". Case variation is not a valid explanation for a 0 result in SUMIF/SUMIFS functions.

</details>

---

### 87. Question

A data analyst builds a lookup using approximate match to find the correct discount tier for an order value. The discount table on a sheet named **Discounts** is:

| Row | MinOrderValue | DiscountPct |
| :-- | :------------ | :---------- |
| 2   | 0             | 0%          |
| 3   | 1000          | 5%          |
| 4   | 5000          | 10%         |
| 5   | 10000         | 15%         |

Column A = MinOrderValue (col 1), Column B = DiscountPct (col 2). The formula is `=VLOOKUP(G2, Discounts!$A$2:$B$5, 2, TRUE)` where G2 contains 7500. What does this formula return?

- [ ] **A.** 15% — the highest available discount tier
- [ ] **B.** `#N/A` — because 7500 is not exactly in the MinOrderValue column
- [ ] **C.** 10% — the discount for the tier whose MinOrderValue (5000) is the largest value ≤ 7500
- [ ] **D.** 5% — the discount for the 1000 tier, one step below 7500

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 15% is the discount for the 10000 tier (MinOrderValue = 10000). Since 7500 < 10000, the 10000 tier has not been reached. Approximate match returns the largest value that does not exceed the lookup value — 10000 > 7500, so this tier is excluded.
- **Option B (Incorrect):** `#N/A` with approximate match only occurs when the lookup value (7500) is smaller than the smallest value in the lookup column (0). Since 7500 > 0, a valid match exists. Approximate match (`TRUE`) does not require an exact match.
- **Option C (Correct):** Approximate match (`TRUE`) requires the lookup column to be sorted in ascending order (which it is: 0, 1000, 5000, 10000). It finds the largest value in the lookup column that is ≤ the lookup value. For 7500: 0 ≤ 7500 ✓, 1000 ≤ 7500 ✓, 5000 ≤ 7500 ✓, 10000 > 7500 ✗. The largest qualifying value is **5000**, whose corresponding discount (col 2) is **10%**.
- **Option D (Incorrect):** 5% corresponds to the 1000 tier. The largest MinOrderValue ≤ 7500 is 5000 (not 1000). VLOOKUP approximate match selects the floor value — the largest that doesn't exceed the lookup value, which is 5000 not 1000.

</details>

---

### 88. Question

A data architect is reviewing these four statements about the VLOOKUP data enrichment architecture (joining an Orders "fact table" with dimension tables like Customers and Products). Which statements are accurate?

- [ ] **A.** In the VLOOKUP enrichment pattern, the Orders sheet acts as the "many" side (fact table) and the Customers sheet acts as the "one" side (dimension table) in a one-to-many relationship
- [ ] **B.** VLOOKUP enrichment is a static join — once the formula is entered, the enriched values never change even if the Customers table is updated
- [ ] **C.** The VLOOKUP enrichment pattern requires the join key (e.g., CustomerID) to be unique in the dimension table (Customers) but not necessarily unique in the fact table (Orders)
- [ ] **D.** Enriching the Orders sheet with VLOOKUP creates a denormalized flat table, similar to the result of a SQL JOIN, which can then be aggregated using SUMIFS/COUNTIFS

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C, D**

#### Explanations:
- **Option A (Correct):** In a one-to-many relationship, one customer can have many orders. The Customers table has one row per CustomerID (one side), while the Orders table has many rows per CustomerID (many side). VLOOKUP looks up from the many side (Orders) into the one side (Customers), which is the standard enrichment direction.
- **Option B (Incorrect):** VLOOKUP enrichment uses **live formulas** — they recalculate whenever the referenced cells change. If a customer's Region is updated in the Customers table, all VLOOKUP formulas that pull that customer's Region on the Orders sheet will automatically update to reflect the new Region value on the next recalculation. The enrichment is dynamic, not static.
- **Option C (Correct):** VLOOKUP's first-match behavior means that if CustomerID appears twice in the Customers table (the dimension/one side), it would always return the first match, potentially missing data. For correct enrichment, CustomerIDs must be **unique** in the Customers table. In the Orders table (many side), the same CustomerID can and should appear in multiple rows (one per order), which is expected and correct.
- **Option D (Correct):** By adding VLOOKUP-derived columns to the Orders sheet (e.g., Region, Tier, Category from respective dimension tables), the analyst creates a single wide (denormalized) table containing all attributes needed for analysis. This flattened structure mirrors the result of a SQL LEFT JOIN and enables SUMIFS/COUNTIFS to operate on all dimensions simultaneously without additional table lookups at aggregation time.

</details>

---

### 89. Question

An analyst is auditing a formula `=SUMIFS(F2:F200, C2:C200, "North", D2:D200, "Completed", E2:E200, "Enterprise")` on a 200-row Orders sheet. She wants to verify the result by manually cross-checking it against a pivot table. She sets the pivot table to: Region filter = "North", Status filter = "Completed", Segment filter = "Enterprise", and Sum of Amount in Values. After refreshing the pivot table, she sees a different number than the SUMIFS result. Which of the following could explain the discrepancy?

- [ ] **A.** The pivot table has not been refreshed after the most recent data import, while the SUMIFS formula updated automatically
- [ ] **B.** The SUMIFS formula range ends at row 200, but new orders (rows 201+) were added after the last formula update — these rows appear in the pivot table (if its source includes them) but not in SUMIFS
- [ ] **C.** The pivot table applies all three filters simultaneously (AND logic), while SUMIFS applies OR logic across its criteria
- [ ] **D.** Some rows in columns C, D, or E contain VLOOKUP errors (`#N/A`) that SUMIFS silently skips but the pivot table handles differently

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** SUMIFS formulas recalculate live with every data change. Pivot tables cache their results until manually refreshed. If new orders were added after the last pivot refresh but before this comparison, the SUMIFS result includes new data while the pivot table shows stale data — causing a discrepancy. Refreshing the pivot table should bring the values into alignment if no other issues exist.
- **Option B (Correct):** If the SUMIFS formula range is `F2:F200` and new orders were added to rows 201–250, those rows are outside the SUMIFS range and contribute 0 to the SUMIFS result. If the pivot table's source data was defined to include a larger range (e.g., F2:F500 or a named table), it would include the new rows after refresh, showing a larger total than SUMIFS.
- **Option C (Incorrect):** Both SUMIFS and pivot table filters apply AND logic across multiple criteria. SUMIFS requires all conditions to be satisfied simultaneously (AND). Similarly, a pivot table with three filters active shows data matching ALL three filter selections simultaneously (AND). There is no OR vs. AND discrepancy between the two approaches — they apply the same logical intersection.
- **Option D (Correct):** `#N/A` values in enriched columns (from failed VLOOKUP) can cause different behaviors in SUMIFS vs. pivot tables. SUMIFS skips `#N/A` cells (they don't match text criteria like "North" or "Completed") — these rows are excluded from the sum. A pivot table may group `#N/A` values differently depending on filter settings. If many rows have `#N/A` enrichment errors, the two tools may count or sum different sets of rows.

</details>

---

### 90. Question

A data analyst is building a reusable formula template that colleagues will use across different sheets. She designs the formula `=IFERROR(VLOOKUP(B2, Suppliers!$A$2:$D$100, 3, FALSE), "Not Found")`. Which of the following statements about this formula are accurate?

- [ ] **A.** `IFERROR` wraps the VLOOKUP and returns "Not Found" for any error type — including `#N/A` (key missing), `#REF!` (bad column index), and `#VALUE!` (invalid argument)
- [ ] **B.** The formula returns "Not Found" only for `#N/A` errors; other error types (`#REF!`, `#VALUE!`) still propagate through IFERROR
- [ ] **C.** Using IFERROR masks all errors, which can hide genuine formula bugs (e.g., a wrong column index or wrong table_array) — analysts should test the formula without IFERROR first before adding it in production
- [ ] **D.** `IFERROR` changes the formula's behavior when valid data is found — valid VLOOKUP results are processed differently when wrapped in IFERROR

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C**

#### Explanations:
- **Option A (Correct):** `IFERROR(value, value_if_error)` intercepts **any** error that the inner expression produces — `#N/A`, `#REF!`, `#VALUE!`, `#DIV/0!`, `#NAME?`, and others — and substitutes the fallback value ("Not Found"). It is not selective; it catches all Excel/Google Sheets error types.
- **Option B (Incorrect):** `IFERROR` is not limited to `#N/A`. It catches all error types. `IFNA` is the function that specifically handles only `#N/A` errors, leaving other error types to propagate. The analyst should use `IFNA` if she only wants to catch missing-key errors while allowing other formula errors (like `#REF!`) to surface as diagnostics.
- **Option C (Correct):** This is a critical best practice. Because `IFERROR` silences all errors, a formula like `=IFERROR(VLOOKUP(B2, Suppliers!$A$2:$D$100, 5, FALSE), "Not Found")` — where column 5 doesn't exist in a 4-column range — would silently return "Not Found" for all rows instead of the expected `#REF!` error that signals the bug. Analysts should first test the raw VLOOKUP without IFERROR to confirm it works correctly for valid data and produces expected errors for invalid data before wrapping it.
- **Option D (Incorrect):** `IFERROR` only activates when the inner expression produces an error. When VLOOKUP successfully finds the key and returns a valid result (text or number), `IFERROR` simply passes that result through unchanged. The formula's behavior on valid data is identical to the bare VLOOKUP.

</details>
