# Spreadsheet Objective Test — Set 10

---

### 91. Question

A senior data analyst presents a capstone scenario: A company's e-commerce platform generates an Orders feed daily. The raw Orders table has: OrderID, CustomerID, ProductSKU, ChannelID, Quantity, UnitPrice, and OrderDate. She proposes enriching this table with VLOOKUP from three dimension tables: **Customers** (→Region, Tier), **Products** (→Category, Cost), and **Channels** (→ChannelName). After enrichment, she builds SUMIFS-based KPI cells and a pivot table for the monthly dashboard.

Which of the following are valid concerns about this architecture at scale?

- [ ] **A.** As the Orders table grows to tens of thousands of rows, each VLOOKUP formula must recalculate for every row on every data change — this can cause noticeable performance slowdowns in very large workbooks
- [ ] **B.** If any dimension table (Customers, Products, Channels) has a missing key for an order row, the corresponding enriched column returns `#N/A`, which can silently exclude those orders from SUMIFS aggregations
- [ ] **C.** Pivot tables sourced from the enriched Orders sheet require manual refresh after each daily data import, unlike the VLOOKUP and SUMIFS formulas which recalculate automatically
- [ ] **D.** This architecture has no scalability concerns — VLOOKUP, SUMIFS, and pivot tables scale infinitely in Excel and Google Sheets

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C**

#### Explanations:
- **Option A (Correct):** VLOOKUP is a volatile-adjacent formula — while not technically volatile (like NOW() or RAND()), VLOOKUP on tens of thousands of rows with large reference tables creates significant recalculation overhead. Each data change triggers recalculation of all formula cells, including every VLOOKUP row. At large scale (100K+ rows), this can make the workbook sluggish or unresponsive. Database tools (SQL, Python) handle this scale far better.
- **Option B (Correct):** This is a real data quality risk at scale. As the Orders table grows and is fed from multiple systems, some CustomerIDs or ProductSKUs may be missing from the dimension tables (newly added products not yet in Products table, deleted customers, etc.). Each missing key creates `#N/A` in the enriched column, and SUMIFS silently excludes those rows. At scale, many such exclusions accumulate unnoticed, leading to understated KPIs.
- **Option C (Correct):** Pivot tables cache their results and are not part of the live recalculation engine. After each daily import, pivot tables must be refreshed manually (or configured to auto-refresh on workbook open). The VLOOKUP formulas and SUMIFS formulas, however, recalculate automatically with each data change.
- **Option D (Incorrect):** Excel and Google Sheets have practical row limits (Excel: ~1M rows; Google Sheets: 10M cells total) and performance limits well before those maximums. VLOOKUP-heavy workbooks with large data are a well-known performance bottleneck. These tools do not scale infinitely — beyond certain sizes, database tools or data warehouses become necessary.

</details>

---

### 92. Question

A data analyst wants to count orders where **Status is Completed AND Amount is greater than 30000** from a 200-row Orders table (col D=Status, col F=Amount). Which formula is correct?

- [ ] **A.** `=COUNTIFS(D2:D201, "Completed", F2:F201, ">30000")`
- [ ] **B.** `=COUNTIF(D2:D201, "Completed") * COUNTIF(F2:F201, ">30000")`
- [ ] **C.** `=COUNTIF(D2:D201, "Completed" AND F2:F201 > 30000)`
- [ ] **D.** `=SUMIFS(F2:F201, D2:D201, "Completed", F2:F201, ">30000")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `=COUNTIFS(criteria_range1, criterion1, criteria_range2, criterion2)` is the exact multi-condition counting formula. It counts rows where D = "Completed" AND F > 30000 simultaneously. The numeric criterion `">30000"` is valid COUNTIFS syntax (enclosed in quotes as a comparison string).
- **Option B (Incorrect):** Multiplying two COUNTIF results computes: (count of Completed rows) × (count of rows where F > 30000). This multiplication does not produce the count of rows satisfying BOTH conditions simultaneously. For example, if 50 rows are Completed and 40 rows have F > 30000, the multiplication gives 2000 — not the count of rows satisfying both (which might be, say, 35).
- **Option C (Incorrect):** `COUNTIF` accepts only two arguments. The expression inside — `"Completed" AND F2:F201 > 30000` — is not valid spreadsheet syntax. `AND` is a logical function, not a criterion modifier. Multiple conditions in a single COUNTIF argument is invalid; COUNTIFS is required.
- **Option D (Incorrect):** `SUMIFS(F2:F201, D2:D201, "Completed", F2:F201, ">30000")` sums the Amount values for qualifying rows — it does not count them. The question asks for a count, not a total. `COUNTIFS` is the appropriate function for counting rows matching criteria.

</details>

---

### 93. Question

A data analyst has an Orders table enriched with a Segment column (E) derived from a VLOOKUP formula referencing the Customers table. She compares two approaches for a dashboard KPI showing "Count of Enterprise Orders":

- **Approach 1:** `=COUNTIF(E2:E500, "Enterprise")` where column E is VLOOKUP-populated
- **Approach 2:** A pivot table with Segment in Rows, showing Count of OrderID for the Enterprise row

Which of the following are TRUE?

- [ ] **A.** Approach 1 recalculates automatically when new orders are added to rows within E2:E500 (if those rows have VLOOKUP formulas); Approach 2 requires a manual pivot refresh
- [ ] **B.** Approach 1 could undercount if some orders have `#N/A` in column E (from failed VLOOKUP enrichment) — `#N/A` does not match "Enterprise"
- [ ] **C.** Approach 2 automatically excludes `#N/A` from the Enterprise count (displaying only valid segment values), while Approach 1 counts them
- [ ] **D.** Both approaches produce identical results when the data has no VLOOKUP errors and no orders outside the COUNTIF range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** COUNTIF formulas are part of the live recalculation engine and update automatically when any referenced cell changes. New orders that fall within E2:E500 and receive VLOOKUP-enriched Segment values are immediately counted. The pivot table requires an explicit refresh after new data is added to include those rows in its count.
- **Option B (Correct):** `#N/A` does not match the text criterion "Enterprise". COUNTIF skips error cells, returning a count that only includes rows with actual text values. If 10 orders have `#N/A` in column E, those 10 potential Enterprise or SMB orders are excluded from the COUNTIF, producing an undercount of total valid orders.
- **Option C (Incorrect):** Pivot tables also group `#N/A` as a separate item in the Segment field — they typically display it as a distinct "(blank)" or "#N/A" row in the pivot table. They do not automatically "exclude" `#N/A` from aggregation; they show it as its own category. If the analyst is looking only at the "Enterprise" row of the pivot, she would also miss orders with `#N/A` enrichment errors, similar to COUNTIF. The behavior is similar, not opposite.
- **Option D (Correct):** In the ideal scenario — no VLOOKUP errors in column E, all orders within the COUNTIF range, pivot table refreshed after the same data state — both approaches return the same count of Enterprise orders. The discrepancies only arise from data quality issues (`#N/A`) or range/refresh timing differences.

</details>

---

### 94. Question

A data analyst uses the formula `=VLOOKUP(A2, $B$2:$E$10, 1, FALSE)` where `col_index_num = 1`. The table_array has col B=Code, col C=Name, col D=Type, col E=Value. What does column index 1 return?

- [ ] **A.** The formula returns `#REF!` — column index 1 is not valid in VLOOKUP
- [ ] **B.** The formula returns the value from column B (Code) — the leftmost column of the range — because column index 1 refers to the first column of `table_array`
- [ ] **C.** The formula returns the value from column A of the worksheet, because column index 1 always means worksheet column A
- [ ] **D.** The formula returns the same value as the lookup value (A2), because index 1 returns the search column itself

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B, D**

#### Explanations:
- **Option A (Incorrect):** `col_index_num = 1` is completely valid. Index 1 refers to the first column of `table_array`. The minimum valid index is 1; indices below 1 (zero or negative) produce errors.
- **Option B (Correct):** Column index 1 always refers to the **first (leftmost) column of `table_array`**. Since `table_array = $B$2:$E$10`, column index 1 = column B = Code. VLOOKUP returns the Code value from the row where the lookup value was found.
- **Option C (Incorrect):** `col_index_num` is **relative to `table_array`**, not to the entire worksheet. If the range starts at column B, column index 1 = column B (not worksheet column A). Worksheet column letters are irrelevant to VLOOKUP's column indexing.
- **Option D (Correct):** Since VLOOKUP's search always occurs in the leftmost column of `table_array` (column B), and `col_index_num = 1` returns that same leftmost column, the formula returns the value found in the search column for the matched row. If the lookup value `A2` is found exactly in column B, then column index 1 returns that same Code value — making this formula somewhat circular but technically valid and return the matched Code.

</details>

---

### 95. Question

A business analyst is designing a dashboard where different teams need to see different subsets of the same Orders data. She considers two approaches:

- **Design A:** One pivot table with Region, Segment, Status in the Filters area — each team applies their own filter combination manually
- **Design B:** Multiple separate SUMIFS formulas, each hardcoded for a specific team's criteria (e.g., North+Enterprise+Completed for the North Enterprise team)

Which of the following accurately describe the trade-offs between these designs?

- [ ] **A.** Design A (pivot with filters) provides interactive flexibility but requires teams to manually select their filter values after each refresh — and if a team forgets to reset filters, they may view incorrect data
- [ ] **B.** Design B (hardcoded SUMIFS) is more maintenance-intensive when criteria change (new regions, new segments) — each new combination requires adding new formulas
- [ ] **C.** Design A pivot table updates automatically without refresh when any order's Status changes
- [ ] **D.** Design B SUMIFS formulas are permanently fixed and can never be updated to handle new criteria without entirely rewriting the formulas

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** A shared pivot table with Filters requires each user to set their filter selection. After a pivot refresh, filters may reset or remain from a previous session. If Team A leaves the filter set to "South" and Team B opens the file expecting "North" data, they see the wrong result without realizing it. This shared-state problem is a real operational risk in collaborative filter-based dashboards.
- **Option B (Correct):** SUMIFS with hardcoded criteria creates a separate formula for each team/metric combination. When the business adds a new region ("West"), someone must manually add West-specific SUMIFS formulas for each metric across each team's dashboard section. At scale, this becomes maintenance-heavy. Pivot tables handle new dimension values automatically after refresh without formula updates.
- **Option C (Incorrect):** Pivot tables do **not** update automatically when source data changes. They require a manual refresh (or auto-refresh on open). Design A's pivot table is not live — it reflects the data state at the last refresh. This is the opposite of SUMIFS formulas, which are live.
- **Option D (Incorrect):** SUMIFS formulas can absolutely be updated — an analyst edits the criteria strings in the formula to add new conditions, change thresholds, or add new criteria_range/criterion pairs. The claim that they "can never be updated without entirely rewriting" is false. Updating a SUMIFS formula is typically a simple criteria string change.

</details>

---

### 96. Question

A data analyst is comparing VLOOKUP with the INDEX+MATCH approach for a lookup scenario where the lookup key is in column D and the return value is in column A (to the left of the key column). Which statements about this scenario are accurate?

- [ ] **A.** VLOOKUP cannot perform this lookup because the return column (A) is to the left of the lookup column (D); VLOOKUP's leftmost-column rule requires the key to be in the first column of table_array
- [ ] **B.** `=INDEX($A$2:$A$100, MATCH(G2, $D$2:$D$100, 0))` correctly performs the left-of-key lookup by using MATCH to find the row and INDEX to return the value from any column
- [ ] **C.** To use VLOOKUP for this left-side lookup, the analyst can use a negative `col_index_num` (e.g., -3) to navigate leftward from the lookup column
- [ ] **D.** Rearranging the table columns so the key is leftmost (making it compatible with VLOOKUP) is one valid workaround, though it changes the table structure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** VLOOKUP cannot look to the left of its search column. The search column must be the leftmost column of `table_array`. If the key is in column D and the return value is in column A, VLOOKUP's architecture makes this lookup impossible without restructuring the table or the range.
- **Option B (Correct):** `INDEX(return_array, row_number)` returns a value from a specified row of a range. `MATCH(lookup_value, lookup_array, 0)` finds the row number of an exact match in any column. Combining them: `INDEX($A$2:$A$100, MATCH(G2, $D$2:$D$100, 0))` finds the row where D = G2, then returns the value from column A of that same row. This approach is not restricted to any particular column order — the lookup and return ranges are entirely independent.
- **Option C (Incorrect):** VLOOKUP does not support negative column indexes. `col_index_num` must be a positive integer (≥ 1). A negative value like -3 produces a `#VALUE!` error. VLOOKUP cannot navigate leftward by any argument manipulation.
- **Option D (Correct):** If the table can be restructured, moving the key column (D) to become the leftmost column of the range makes VLOOKUP work normally. This is a valid workaround at the data structure level. However, restructuring the physical table may not always be possible (if other formulas depend on the current column layout, or if the data is generated by an external system).

</details>

---

### 97. Question

A data analyst builds a multi-sheet workbook with the following architecture: Sheet 1 = Orders (raw + enriched), Sheet 2 = Customers (reference), Sheet 3 = Dashboard (KPI cells). She wants to present a comprehensive review of formula recalculation behavior. Which of the following are TRUE?

- [ ] **A.** `=VLOOKUP(B2, Customers!$A$2:$D$50, 3, FALSE)` in Sheet 1 automatically updates when the Customers sheet data changes
- [ ] **B.** `=SUMIFS(Sheet1!F2:F200, Sheet1!C2:C200, "North", Sheet1!D2:D200, "Completed")` in Sheet 3 automatically updates when Sheet 1 data changes
- [ ] **C.** A pivot table on Sheet 3 sourced from Sheet 1's data automatically updates when Sheet 1 data changes, without needing a manual refresh
- [ ] **D.** `=Sheet3!PivotTable1!B5` is valid syntax to reference a specific named pivot table range in another sheet

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** All formula cells (VLOOKUP, SUMIF, AVERAGEIF, etc.) are live in the recalculation engine. When the Customers sheet data changes (e.g., a customer's Region is updated), all formulas referencing that data — including VLOOKUP formulas on Sheet 1 — automatically recalculate and display the new value.
- **Option B (Correct):** Cross-sheet SUMIFS formulas reference data on Sheet 1 using standard sheet reference syntax. When Sheet 1's data changes (existing values updated or new rows within the range added), the SUMIFS formula on Sheet 3 recalculates automatically. The cross-sheet reference adds no behavioral difference — it is still a live formula.
- **Option C (Incorrect):** Pivot tables do **not** automatically recalculate when source data changes. They maintain a cached result from the last refresh. The analyst must manually click Refresh (or configure auto-refresh) to incorporate new or changed data from Sheet 1. This is a fundamental operational distinction between formula-based and pivot-based calculations.
- **Option D (Incorrect):** `=Sheet3!PivotTable1!B5` is not valid Excel or Google Sheets formula syntax. A cross-sheet reference uses only one `!` separator: `Sheet3!B5`. There is no syntax for chaining multiple `!` operators to navigate into a named object within a sheet. Pivot tables do not have a separately addressable namespace in cross-sheet references — they are just cells on a sheet, accessed by their cell address.

</details>

---

### 98. Question

A senior analyst reviews a spreadsheet dashboard designed by a junior colleague and identifies several formula design issues. She annotates four cells:

- **Cell X:** `=SUMIF(C2:C100, E2, D2:D100)` — uses a cell reference as the SUMIF criterion
- **Cell Y:** `=VLOOKUP(A2, $B$2:$E$50, 2, FALSE)` — table_array starts at column B, not column A where the lookup key column lives
- **Cell Z:** `=COUNTIFS(D2:D100, "Completed", D2:D100, "Pending")` — checks the same column D for two different values
- **Cell W:** `=AVERAGEIF(C2:C100, "North", D2:D100)` — criteria range and average range are different columns

Which cells contain genuine formula errors (not just style issues)?

- [ ] **A.** Cell Y — because table_array must always start from column A of the worksheet
- [ ] **B.** Cell Z — because COUNTIFS applies AND logic, so no row can have Status = "Completed" AND Status = "Pending" simultaneously; the result is always 0
- [ ] **C.** Cell X — because SUMIF criteria must always be a hardcoded string, not a cell reference
- [ ] **D.** Cell W — because AVERAGEIF requires the criteria_range and average_range to be the same range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This is a potential logical error (if the key column is in A and the table starts at B, VLOOKUP searches the wrong column), but it is not inherently wrong to start a table_array at column B. If the key column IS in column B of the worksheet, starting the range at B is correct. The analyst flags this for review because it is suspicious, not because it is definitively wrong in all cases. However, as described earlier in this test, if the lookup key lives in column A and the range starts at B, then it IS a logical error — just not a syntax error.
- **Option B (Correct):** `=COUNTIFS(D2:D100, "Completed", D2:D100, "Pending")` applies AND logic: it counts rows where D = "Completed" AND D = "Pending" simultaneously. A single cell cannot equal two different text values at the same time — the conditions are mutually exclusive. The formula always returns **0**, making it a genuine semantic error (the analyst likely intended OR logic, which requires `COUNTIF(...,"Completed") + COUNTIF(...,"Pending")`).
- **Option C (Incorrect):** Cell X uses a cell reference (`E2`) as the SUMIF criterion — this is perfectly valid and common practice. SUMIF criteria accept cell references, and the function reads the current value of E2 as the criterion. No error here.
- **Option D (Incorrect):** AVERAGEIF explicitly supports separate criteria_range and average_range. The syntax `=AVERAGEIF(criteria_range, criterion, average_range)` uses the criteria_range for filtering and the average_range for the values to average. These are intentionally separate ranges — the whole point is to filter by one column (Region) and average another column (some numeric field). Cell W is syntactically and logically correct.

</details>

---

### 99. Question

A data engineer builds a lookup to find the Price from a Products table for a given order's SKU. The formula is `=VLOOKUP(B2, Products!$A$2:$D$100, 4, FALSE)`. Products table: col A=SKU, col B=Name, col C=Category, col D=Price. The formula returns the correct price in most rows but returns `#N/A` specifically for SKUs that start with a leading zero (e.g., "007", "012"). Investigation shows these SKUs are imported as numbers (7, 12) in the Products table but stored as text strings ("007", "012") in the Orders sheet. Which fix addresses this root cause?

- [ ] **A.** Change `range_lookup` to `TRUE` so approximate match finds the closest numeric value
- [ ] **B.** Wrap the lookup value with `TEXT(B2, "000")` to convert the numeric SKU in Orders to the zero-padded text format matching the Products table: `=VLOOKUP(TEXT(B2, "000"), Products!$A$2:$D$100, 4, FALSE)`
- [ ] **C.** Change the `col_index_num` from 4 to 1 to look up by the SKU column itself
- [ ] **D.** Standardize the data by formatting all SKU values consistently as either all text or all numbers in both the Orders and Products tables, then run the VLOOKUP formula as-is

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B, D**

#### Explanations:
- **Option A (Incorrect):** Approximate match on unsorted text SKUs would produce unpredictable results. The SKUs are alphanumeric codes, not sorted numeric brackets. Approximate match would likely return incorrect records or `#N/A` for values below the minimum. This does not resolve the type mismatch — it introduces a worse problem.
- **Option B (Correct):** If the Products table stores SKUs as text with leading zeros ("007") but the Orders sheet stores them as numbers (7), `TEXT(B2, "000")` converts the number 7 to the text string "007" — matching the format in the Products table. The wrapped VLOOKUP then finds a successful exact match. This is a formula-level fix that addresses the type mismatch without changing the underlying data.
- **Option C (Incorrect):** `col_index_num = 1` returns the SKU value itself from the Products table (the search column). Changing the column index does not address the type mismatch between text and numeric representations of the same ID. The `#N/A` comes from the key not being found, not from the wrong column being returned.
- **Option D (Correct):** Standardizing the data at source is the cleanest, most robust fix. If all SKUs in both tables are consistently stored as the same type and format (all text "007"/"012" or all numbers 7/12), the type mismatch disappears entirely, and the original VLOOKUP formula works without any wrapper functions. Data standardization eliminates the root cause rather than working around it.

</details>

---

### 100. Question

A data analyst presents a comprehensive review of spreadsheet best practices to her team. She makes the following ten statements. Which of the following are TRUE?

- [ ] **A.** When building a VLOOKUP for a key-based join (CustomerID, ProductSKU, etc.), `range_lookup = FALSE` (exact match) should always be used to prevent silent wrong-row matches
- [ ] **B.** SUMIFS always places the sum_range as the first argument, while SUMIF places the sum_range as the last (optional) argument — mixing up these argument orders is one of the most frequent formula errors
- [ ] **C.** A pivot table cell reference like `=PivotSheet!D6` is a reliable way to embed a specific pivot metric in a KPI box, because cross-sheet references always track the labeled data regardless of pivot layout changes
- [ ] **D.** Locking the table_array with `$` signs (absolute reference) in a VLOOKUP formula is mandatory when the formula will be copied to multiple rows — without `$`, the range shifts and may produce incorrect results for rows beyond the first

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** For key-based lookups (IDs, SKUs, codes), exact match (`FALSE`) is mandatory. Approximate match (`TRUE`) on unordered or text-keyed tables silently returns wrong rows — it does not error, making the bug invisible. Exact match either returns the correct row or a clear `#N/A` error, both of which are actionable outcomes.
- **Option B (Correct):** This is one of the most common and consequential formula argument-order mistakes. **SUMIF**: `SUMIF(criteria_range, criterion, [sum_range])` — sum_range is last. **SUMIFS**: `SUMIFS(sum_range, criteria_range1, criterion1, ...)` — sum_range is first. Swapping these produces `#VALUE!` errors or incorrect results without obvious error signals. Every analyst switching between these functions should verify argument order.
- **Option C (Incorrect):** This is false — cross-sheet cell references like `=PivotSheet!D6` point to a fixed physical cell address. They do not track labeled pivot data. If the pivot table's layout changes (new rows added, new columns inserted, data re-sorted), the value at D6 changes to whatever now occupies that cell, silently producing wrong KPI data. The robust alternative is GETPIVOTDATA, which retrieves pivot data by field name and item label rather than by cell address.
- **Option D (Correct):** Without `$` on the table_array, copying VLOOKUP down shifts the range row by row (e.g., `A2:D50` becomes `A3:D51`, `A4:D52`, etc.). The search range drifts progressively outside the actual reference table. While the first few copies may accidentally produce correct results (if the customer happens to be in the shifted range), later copies find `#N/A` or wrong-row data. Using `$A$2:$D$50` locks the range absolutely for all copies.

</details>
