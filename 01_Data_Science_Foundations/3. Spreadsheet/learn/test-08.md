# Spreadsheet Objective Test — Set 08

---

### 71. Question

A data architect is building a Regional Sales Dashboard with three sheets: **RawOrders** (transactional data), **Customers** (reference), and **Dashboard** (summary formulas). She uses VLOOKUP to enrich RawOrders with customer attributes from the Customers table, then uses SUMIFS on the enriched Orders data to power the Dashboard. After a batch import adds 200 new orders to RawOrders, which of the following are TRUE about the Dashboard?

- [ ] **A.** The SUMIFS formulas on the Dashboard automatically recalculate to include the new orders if the new rows fall within the ranges specified in the SUMIFS arguments (e.g., if the range is `E2:E201` but new orders are in rows 202–401, they are excluded)
- [ ] **B.** The VLOOKUP formulas in RawOrders automatically enrich the new order rows with customer data without any manual action
- [ ] **C.** The pivot table on the Dashboard automatically includes the new orders without refreshing, because it shares the same source data as the SUMIFS formulas
- [ ] **D.** Extending SUMIFS ranges to cover future rows (e.g., `E2:E10000`) is a common practice to avoid needing to update formula ranges after every data import

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, D**

#### Explanations:
- **Option A (Correct):** SUMIFS formulas recalculate automatically when source cell values change. However, if the formula range is hardcoded (e.g., `E2:E201`), rows outside that range (e.g., rows 202–401) are completely excluded — no recalculation includes data outside the specified range. The analyst must either extend the range manually or use a larger pre-defined range that accommodates future growth.
- **Option B (Incorrect):** VLOOKUP formulas only exist in cells that were explicitly populated. When new rows (202–401) are added to RawOrders, they do not automatically receive VLOOKUP formulas unless the analyst copies the formula down to the new rows. Formula cells must be explicitly placed in the new rows; data added below the last formula row is not automatically enriched.
- **Option C (Incorrect):** Pivot tables do **not** automatically include new rows beyond their defined data source range, and they require a manual refresh (clicking Refresh or Data → Refresh in Excel; Refresh in Google Sheets). The pivot table is not live in the same way that formula cells are — it caches a snapshot of the data and must be explicitly refreshed to pick up new rows.
- **Option D (Correct):** Using a substantially larger range than the current data (e.g., `E2:E10000` when only 200 rows exist) is a common and practical technique. SUMIFS, COUNTIFS, and AVERAGEIF skip empty cells gracefully — they do not count or sum blank cells. This "buffer" approach prevents the need to update formula ranges after every import, at the cost of slightly more memory and processing time.

</details>

---

### 72. Question

An analyst is designing a data enrichment join using VLOOKUP. She has:
- **Orders** sheet: col B = CustomerID (key), cols G–J = enriched attributes
- **Customers** sheet: col A = CustomerID, col B = Name, col C = Region, col D = Tier

She wants column G of Orders to show the Customer **Tier** (from col D of Customers). The table has 50 customers (rows 2–51 on the Customers sheet). Her formula in G2 of Orders is:

```
=VLOOKUP(B2, Customers!$A$2:$D$51, 4, FALSE)
```

She copies it to G3 through G200. At row G75, the formula returns "Gold" correctly. At row G112, it returns `#N/A`. The CustomerID in B112 is "C044". Which of the following are plausible explanations?

- [ ] **A.** The table_array shifted because `$` was not used — now it searches the wrong rows of the Customers sheet
- [ ] **B.** "C044" exists in the Customers table but with leading zeros stripped (stored as the number 44 rather than text "C044"), causing a type mismatch
- [ ] **C.** Column D (Tier) in the Customers table is empty for customer C044, so VLOOKUP has nothing to return
- [ ] **D.** "C044" is genuinely absent from the Customers table — no record exists for this CustomerID

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** The table_array `Customers!$A$2:$D$51` uses full absolute locking (`$` on both rows and columns). It does not shift when copied down. Range shifting is not the cause of `#N/A` in this case.
- **Option B (Incorrect):** "C044" contains letters (the "C" prefix), so it is definitionally a text string, not a number. Leading zero stripping (44 vs 044) applies to purely numeric values. "C044" retains its format as text. A data-type mismatch is theoretically possible if the Customers sheet stores "C044" in a different encoding, but the specific scenario of "leading zeros stripped from a number" does not apply to a value with a text prefix.
- **Option C (Incorrect):** An empty Tier cell (col D) for a customer that IS in the Customers table would cause VLOOKUP to return an **empty string** or a blank cell — not `#N/A`. The `#N/A` error specifically indicates the lookup key was not found in the leftmost column. If the key is found but the return column is empty, VLOOKUP returns blank, not an error.
- **Option D (Correct):** `#N/A` with exact match (`FALSE`) means the lookup value was not found in the leftmost column of the table_array. The most direct and common explanation is that "C044" simply does not exist as a row in the Customers table (rows 2–51). The customer may have been missed during data entry, or the ID may be formatted differently in the Customers table. The fix is to add the missing customer record or reconcile the ID format.

</details>

---

### 73. Question

A data analyst is building a multi-table join dashboard for an e-commerce company. She has three reference tables:
- **Customers**: CustomerID → Region, Tier
- **Products**: ProductSKU → Category, UnitCost
- **Channels**: ChannelID → ChannelName, Margin

Her Orders table has: OrderID, CustomerID, ProductSKU, ChannelID, Quantity, Revenue. She enriches the Orders table using VLOOKUP for each reference table. After enrichment, she uses SUMIFS to calculate total Revenue by Region AND Tier AND Category for a dashboard. Which of the following statements about this architecture are TRUE?

- [ ] **A.** Each VLOOKUP enrichment formula (for Region, Tier, Category, etc.) must reference the leftmost column of its respective table as the search column
- [ ] **B.** The SUMIFS formula can use the enriched columns (Region, Tier, Category) as criteria ranges because they are now part of the Orders sheet as formula results, even though they were originally from separate tables
- [ ] **C.** If a CustomerID in Orders has no matching row in the Customers table, the enriched Region column will show `#N/A`, and any SUMIFS formula using that Region column as a criteria range will ignore or propagate the error for that row
- [ ] **D.** The VLOOKUP enrichment pattern described is functionally similar to a relational database JOIN operation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C, D**

#### Explanations:
- **Option A (Correct):** VLOOKUP's core rule is that the search always occurs in the **leftmost column** of `table_array`. For the Customers table, the lookup key (CustomerID) must be in column A (or whichever column is leftmost in the specified range). If the key column is not leftmost, the lookup searches the wrong column and returns incorrect or failed results.
- **Option B (Correct):** Once VLOOKUP formulas populate the enriched columns on the Orders sheet (e.g., column H = Region, column I = Tier, column J = Category), those cells contain values (text or numbers) that SUMIFS can use as criteria ranges. SUMIFS doesn't care whether the values came from direct input or from VLOOKUP formulas — it operates on cell values. The multi-table join is now "flattened" into a single enriched Orders sheet.
- **Option C (Correct):** When a CustomerID has no Customers match, VLOOKUP returns `#N/A`. SUMIFS treats `#N/A` as a non-matching value — no standard text criterion like "North" will match an `#N/A` cell, so those rows are effectively excluded from SUMIFS aggregations. However, this silent exclusion can lead to undercounting. The analyst should monitor for `#N/A` in enriched columns and resolve data quality issues before trusting aggregations.
- **Option D (Correct):** The VLOOKUP enrichment pattern — matching a foreign key in the Orders table to a primary key in the reference tables (Customers, Products, Channels) and retrieving attributes — is conceptually identical to a LEFT JOIN in SQL. The Orders table acts as the fact table; the reference tables act as dimension tables. The flattened enriched Orders sheet represents the result of the join, ready for aggregation.

</details>

---

### 74. Question

A data analyst audits a spreadsheet and finds the formula `=SUMIFS(F2:F100, C2:C100, G2, D2:D100, ">"&H2)`. Column C holds Region, column D holds Revenue, G2 holds "North", and H2 holds 50000. What does this formula compute?

- [ ] **A.** `#VALUE!` — because SUMIFS criteria must be text strings, not cell references or concatenated expressions
- [ ] **B.** The sum of F values where Region = "North" AND Revenue > 50000
- [ ] **C.** The sum of F values where Region = the text "G2" AND Revenue > "50000"
- [ ] **D.** The sum of F values where Region = "North" AND Revenue > 50000, using dynamic criterion values from cells G2 and H2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** SUMIFS criteria are not limited to hardcoded text strings. They can be: text literals (`"North"`), numeric literals (`50000`), comparison expressions (`">50000"`), cell references (`G2`), or concatenated expressions combining operators with cell references (`">"&H2`). All are valid SUMIFS criteria syntax.
- **Option B (Incorrect):** While the description of what is computed is accurate, option B ignores the dynamic nature of the criteria — the fact that "North" and 50000 come from cells (G2 and H2) rather than being hardcoded. Option D is the more precise and complete description.
- **Option C (Incorrect):** Cell references in SUMIFS criteria position are interpreted as **references to cell values**, not as literal text. `G2` in a criteria position reads the current value of G2 ("North"), not the text string "G2". Similarly, `">"&H2` concatenates the ">" operator with the value of H2 (50000), producing `">50000"` as the effective criterion.
- **Option D (Correct):** This is a dynamic SUMIFS pattern. The criterion `G2` reads the current value of G2 ("North") as the Region criterion. The criterion `">"&H2` concatenates the greater-than operator with the value of H2 (50000), creating the effective criterion `">50000"`. The formula sums F values where Region="North" AND Revenue>50000. When G2 or H2 changes, the formula automatically recalculates with the new criteria values — making this a powerful, parameterized aggregation approach.

</details>

---

### 75. Question

A finance team uses a workbook with three sheets: **Expenses** (raw data), **Budgets** (reference limits), and **Report** (dashboard). The Report sheet uses SUMIFS on Expenses data and VLOOKUP against Budgets to show spending vs. budget by department. After a major departmental restructuring, the Budgets table is updated with new department names. Which of the following statements about the dashboard are TRUE?

- [ ] **A.** SUMIFS formulas on the Report sheet that reference Expenses data recalculate automatically and correctly if the department names in Expenses are also updated
- [ ] **B.** VLOOKUP formulas that look up department budgets will return `#N/A` for any old department name that no longer exists in the updated Budgets table
- [ ] **C.** If Expenses rows still use the old department names while the Budgets table uses new names, VLOOKUP enrichment breaks but SUMIFS aggregation of the (unenriched) old names in Expenses may still work correctly for old-name criteria
- [ ] **D.** Cross-sheet SUMIFS formulas that reference cells on the Expenses sheet update automatically when Expenses data changes, without requiring any manual action

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C, D**

#### Explanations:
- **Option A (Correct):** SUMIFS formulas recalculate automatically when their source data (Expenses) changes. If department names in Expenses are updated to match the new restructuring, SUMIFS formulas that use department name criteria will correctly pick up the renamed departments without manual intervention (assuming the SUMIFS ranges encompass the updated rows).
- **Option B (Correct):** VLOOKUP with exact match (`FALSE`) returns `#N/A` for any lookup key that does not exist in the Budgets table's leftmost column. If old department names remain in the lookup values but the Budgets table now only contains new names, every old-name lookup fails with `#N/A`.
- **Option C (Correct):** If a partial update occurs — new department names in Budgets but unchanged old names still in Expenses — the two tables are misaligned. VLOOKUP (Expenses key → Budgets lookup) fails for mismatched names. However, SUMIFS on the Expenses sheet itself (using criteria like `"Marketing"` against the old Expenses data) may still produce accurate aggregations of the old data, since SUMIFS does not depend on the Budgets reference table.
- **Option D (Correct):** Cross-sheet SUMIFS references (e.g., `=SUMIFS(Expenses!F2:F500, Expenses!C2:C500, "Marketing")`) are live formulas. When any referenced cell in the Expenses sheet changes, the formula engine automatically marks the Report formulas as dirty and recalculates them. No manual refresh is required for formula-based calculations (unlike pivot tables).

</details>

---

### 76. Question

A data analyst writes the following four formulas intending to calculate the **count of cancelled Enterprise orders** from the Orders table in Question 31. Which of the following formulas are correct?

- [ ] **A.** `=COUNTIFS(D2:D11, "Cancelled", E2:E11, "Enterprise")`
- [ ] **B.** `=COUNTIF(D2:D11, "Cancelled") * COUNTIF(E2:E11, "Enterprise")`
- [ ] **C.** `=SUMPRODUCT((D2:D11="Cancelled")*(E2:E11="Enterprise"))`
- [ ] **D.** `=COUNTIF(D2:D11, "Cancelled", E2:E11, "Enterprise")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C**

#### Explanations:
- **Option A (Correct):** `=COUNTIFS(D2:D11, "Cancelled", E2:E11, "Enterprise")` is the standard multi-condition counting formula. It counts rows where Status="Cancelled" AND Segment="Enterprise". From the Orders table: O04 (Cancelled, Enterprise) = 1 matching row. Result: **1**.
- **Option B (Incorrect):** `COUNTIF(...) * COUNTIF(...)` multiplies two independent counts — this computes the product of (rows with "Cancelled") × (rows with "Enterprise"), not the count of rows satisfying both conditions simultaneously. The multiplication has no mathematical meaning as a conditional count. For the Orders table: 2 Cancelled × 4 Enterprise = 8, which is wrong.
- **Option C (Correct):** `SUMPRODUCT` with boolean arrays is an alternative to COUNTIFS. `(D2:D11="Cancelled")` produces an array of TRUE/FALSE (1/0) values. `(E2:E11="Enterprise")` produces another array. Multiplying the arrays element-wise gives 1 only where both are TRUE. SUMPRODUCT sums these products, counting rows satisfying both conditions. This is a valid advanced technique that predates COUNTIFS in older Excel versions.
- **Option D (Incorrect):** `COUNTIF` accepts exactly two arguments. Passing four arguments as shown is invalid syntax and causes a `#VALUE!` error. Multi-criteria counting must use `COUNTIFS` (with an S), not `COUNTIF`.

</details>

---

### 77. Question

A data analyst is comparing the pivot table approach vs. the SUMIFS formula approach for a dashboard. Which of the following correctly describe trade-offs between the two approaches?

- [ ] **A.** SUMIFS formulas update automatically when source data changes; pivot tables require a manual refresh to reflect new data
- [ ] **B.** Pivot tables support interactive drill-down and layout changes (drag fields, change aggregations) without modifying formulas; SUMIFS formulas require formula changes to alter what is aggregated or how
- [ ] **C.** Pivot tables can fully replace all SUMIFS/COUNTIFS use cases without exception
- [ ] **D.** SUMIFS formulas are more appropriate when a specific, always-visible calculation is needed in a fixed cell (e.g., a KPI box that shows "Total North Enterprise Revenue"); pivot tables are better for flexible, exploratory multi-dimensional summaries

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** This is a key operational difference. SUMIFS, COUNTIFS, AVERAGEIF, and VLOOKUP formulas recalculate automatically when any referenced source cell changes — they are part of Excel/Google Sheets' live recalculation engine. Pivot tables cache their computation and must be manually refreshed (or set to refresh on open) to incorporate data changes.
- **Option B (Correct):** Pivot tables provide a graphical drag-and-drop interface for reorganizing the breakdown dimensions, changing aggregation types (Sum → Average), and filtering — all without writing formulas. Changing the equivalent SUMIFS setup requires editing formula arguments, adding or removing condition pairs, or rewriting ranges.
- **Option C (Incorrect):** Pivot tables cannot fully replace conditional formulas in all cases. Some calculations are awkward or impossible in a pivot table's structure — particularly nested conditional calculations, running totals across specific filtered subsets, or when the result must appear in a specific cell of a larger layout. Conditional formulas remain necessary for many dashboard and reporting scenarios.
- **Option D (Correct):** This correctly characterizes the practical use pattern. SUMIFS excels at fixed KPI cells: a permanently placed formula that always shows one specific filtered metric. Pivot tables excel at exploratory analysis where the analyst needs to interactively slice the data by different dimensions and quickly reconfigure the layout.

</details>

---

### 78. Question

A data analyst writes `=VLOOKUP(A2, Suppliers!$B$2:$E$30, 2, FALSE)`. A senior colleague flags that the table_array starts at column **B**, not column **A**, of the Suppliers sheet. The Suppliers sheet has: col A = SupplierID, col B = CompanyName, col C = Country, col D = Category, col E = Rating. The analyst is trying to look up a SupplierID from A2 of the current sheet. What is the fundamental problem?

- [ ] **A.** The formula is correct — starting the range at column B is fine as long as the lookup key is in A2 of the active sheet
- [ ] **B.** VLOOKUP always searches the leftmost column of `table_array`; since the range starts at B, it searches CompanyName (col B) for the SupplierID — but SupplierIDs are in col A (outside the range), so the formula will search company names for a supplier ID value
- [ ] **C.** The formula should use `col_index_num = 1` instead of 2 to look at the first column of the range
- [ ] **D.** The formula will silently return a random Supplier's CompanyName instead of `#N/A`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The location of the lookup value (`A2` on the active sheet) is irrelevant to the problem. The issue is what column VLOOKUP searches within `table_array`. Since `table_array` starts at column B of the Suppliers sheet, VLOOKUP searches column B (CompanyName) for the SupplierID value — not column A where SupplierIDs actually reside.
- **Option B (Correct):** VLOOKUP always and only searches the **leftmost column of table_array**. With `table_array = Suppliers!$B$2:$E$30`, the leftmost column is B (CompanyName). VLOOKUP will search CompanyName values for whatever SupplierID is in A2 of the active sheet (e.g., "S001") — which will almost certainly not be found among company names, returning `#N/A`. The fix is to include column A: `Suppliers!$A$2:$E$30`.
- **Option C (Incorrect):** `col_index_num = 1` would return the first column of the range (CompanyName in B), not the SupplierID in A. Changing the column index doesn't fix the search column — the first fix is to extend the table_array to start from column A where SupplierIDs live.
- **Option D (Incorrect):** VLOOKUP with exact match does not return random values. When the lookup value is not found in the search column, it returns `#N/A` consistently. Random incorrect returns are a characteristic of approximate match (`TRUE`) on unsorted or inappropriate data, not of exact match (`FALSE`).

</details>

---

### 79. Question

A data analyst has a workbook with an enriched Orders sheet (columns: A=OrderID, B=CustomerID, C=Region, D=Status, E=Segment, F=Amount — where C, E are VLOOKUP-enriched from a Customers table). She builds a SUMIFS formula:

```
=SUMIFS(F2:F100, C2:C100, "North", D2:D100, "Completed", E2:E100, "Enterprise")
```

The formula returns 0, but the analyst is confident that qualifying rows exist. Which of the following should she check first?

- [ ] **A.** Whether column C (Region) or column E (Segment) contains `#N/A` values from failed VLOOKUP enrichment — SUMIFS returns 0 (not an error) when criteria ranges contain errors that don't match the criterion string
- [ ] **B.** Whether the SUMIFS range F2:F100 is too small — extending it to F2:F1000 may fix the issue
- [ ] **C.** Whether column D (Status) values have hidden trailing or leading spaces (e.g., "Completed " instead of "Completed")
- [ ] **D.** Whether the criteria strings in SUMIFS are case-sensitive — "North" might not match "north" in column C

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C**

#### Explanations:
- **Option A (Correct):** If VLOOKUP enrichment failed for any rows (returning `#N/A` in columns C or E), those rows contain the error value rather than the expected text. SUMIFS criteria like "North" will not match `#N/A` — those rows are silently skipped. More importantly, if **all** qualifying rows have `#N/A` enrichment errors, SUMIFS returns 0 even though the rows exist. The analyst should check for `#N/A` in the criteria columns using `=COUNTIF(C2:C100, "#N/A")` or conditional formatting.
- **Option B (Incorrect):** If qualifying rows are within rows 2–100, the range size is not the issue. Extending to F2:F1000 adds empty rows that contribute nothing (SUMIFS skips blank cells). The range should cover existing data, but if the data is within 2–100, extending the range won't recover 0-valued results from legitimate qualifying rows.
- **Option C (Correct):** Hidden whitespace (leading/trailing spaces) is a silent data quality issue. "Completed " (with trailing space) does not match the criterion "Completed". If Status values were imported from a system with padding, every Status cell may appear correct when viewed but fail exact-string matching. Using `TRIM()` on the data or checking with `=LEN(D2)` to count characters reveals unexpected spaces.
- **Option D (Incorrect):** SUMIFS is **case-insensitive** in both Excel and Google Sheets. "North" matches "north", "NORTH", and "North" equally. Case mismatch is not a valid explanation for SUMIFS returning 0.

</details>

---

### 80. Question

A senior analyst builds an advanced dashboard that uses VLOOKUP to join Orders to Customers (getting Region and Tier), then SUMIFS to compute revenue metrics across multiple dimensions. She presents two design options for a specific "KPI box" showing **total completed Enterprise revenue for the current quarter**:

- **Option 1:** A SUMIFS formula that references enriched columns on the Orders sheet directly
- **Option 2:** A pivot table on a separate sheet filtered to "Completed" and "Enterprise", with a cross-sheet cell reference in the KPI box

Which of the following statements about these two options are accurate?

- [ ] **A.** Option 1 (SUMIFS) updates automatically when Orders data changes; Option 2 (pivot + cross-sheet cell reference) requires manual pivot refresh to update the KPI value
- [ ] **B.** Option 2 is more robust than Option 1 when the pivot table layout changes, because cross-sheet references always track the correct labeled cell
- [ ] **C.** Option 1 is more appropriate for a fixed, always-visible KPI because the formula is stable and doesn't depend on pivot table layout
- [ ] **D.** Option 2's cross-sheet reference `=PivotSheet!C5` may silently return the wrong value if the pivot table gains new rows or columns that shift the target metric's cell address

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C, D**

#### Explanations:
- **Option A (Correct):** SUMIFS formulas are part of the live recalculation engine and update immediately when any referenced source data changes. A pivot table requires an explicit refresh action (or auto-refresh on open if configured). For a continuously updating KPI, the SUMIFS approach is more operationally reliable.
- **Option B (Incorrect):** Cross-sheet references to pivot table cells are **less** robust when the pivot layout changes, not more. The reference `=PivotSheet!C5` points to a fixed physical address that does not track labeled pivot data. If the layout shifts (new segment column inserted, new region row added, filters changed), C5 silently returns different data. This is a known fragility of this approach.
- **Option C (Correct):** A dedicated SUMIFS formula for a specific KPI (e.g., `=SUMIFS(F2:F500, D2:D500, "Completed", E2:E500, "Enterprise", G2:G500, "Q3")`) is self-contained and stable. Its result depends only on the data and the formula logic — there is no pivot layout to change, no refresh required, and no cell addressing fragility.
- **Option D (Correct):** This is the core fragility of the Option 2 approach. `=PivotSheet!C5` is a hard-coded cell address. If the pivot table adds a new region row, sorts differently, or the analyst adds a new column, the metric that was in C5 moves to a different cell. The formula silently returns whatever value now occupies C5, which may be a completely different metric. GETPIVOTDATA would be the robust alternative.

</details>
