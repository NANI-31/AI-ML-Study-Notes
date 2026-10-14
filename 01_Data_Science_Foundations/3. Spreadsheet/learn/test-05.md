# Spreadsheet Objective Test — Set 05

---

### 41. Question

A data analyst is enriching an Orders sheet with customer information. She places this formula in column G (row 2) and copies it down to row 11:

```
=VLOOKUP(B2, Customers!$A$2:$D$6, 2, FALSE)
```

The Orders sheet has CustomerID in column B. The Customers table has: col A=CustomerID, col B=Name, col C=City, col D=Segment. After copying, the formula in G7 returns `#N/A` but the CustomerID in B7 ("C3") definitely exists in the Customers table. Which of the following could explain this `#N/A`?

- [ ] **A.** The `col_index_num` of 2 is incorrect; it should be 3 to return the Name
- [ ] **B.** The Customers table_array is locked with `$`, so it won't shift — the `#N/A` must be caused by a data mismatch: the value in B7 may contain extra whitespace or a different data type than what's in Customers column A
- [ ] **C.** VLOOKUP can only be copied down 5 rows; beyond row 7 it always fails
- [ ] **D.** The formula should use `TRUE` instead of `FALSE` for row 7 to work correctly

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Column index 2 correctly returns the Name column (B) of the Customers range. This is the right column for returning names. Changing to index 3 would return City, not Name. More importantly, a wrong column index causes the formula to return a wrong *value*, not `#N/A` — `#N/A` specifically signals that the lookup key was not found.
- **Option B (Correct):** Since the table_array is properly locked (`$A$2:$D$6`), range shifting is not the cause. The `#N/A` must come from the lookup key not being found. The most likely real-world causes when the key "appears" to exist are: (1) hidden whitespace — B7 contains "C3 " (trailing space) while Customers has "C3"; (2) data type mismatch — B7 holds the number 3 while Customers holds the text "C3"; or (3) invisible characters from data import (non-breaking spaces, etc.). Exact match (`FALSE`) requires character-for-character equality.
- **Option C (Incorrect):** There is no restriction on how many rows a VLOOKUP formula can be copied down. The function works identically in row 7, row 107, or row 10007 — the row number of the formula cell has no bearing on whether the lookup succeeds.
- **Option D (Incorrect):** Changing `FALSE` to `TRUE` would switch to approximate match, which requires sorted data and would not resolve a data quality mismatch — it would likely return the wrong customer's data for a non-matching key. `TRUE` is not a fix for lookup key mismatch issues.

</details>

---

### 42. Question

A data analyst needs the **total Amount** for orders that are **Completed AND in the East region** from the Orders table (Question 31). She proposes four formulas. Which one correctly returns this value?

- [ ] **A.** `=SUMIF(C2:C11, "East", D2:D11, "Completed", F2:F11)`
- [ ] **B.** `=SUMIFS(F2:F11, C2:C11, "East", D2:D11, "Completed")`
- [ ] **C.** `=SUMIF(C2:C11, "East") + SUMIF(D2:D11, "Completed")`
- [ ] **D.** `=COUNTIFS(C2:C11, "East", D2:D11, "Completed", F2:F11)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `SUMIF` accepts only three arguments: `(criteria_range, criterion, [sum_range])`. Passing five arguments as shown exceeds SUMIF's parameter limit and causes a `#VALUE!` error. Multi-condition summing requires `SUMIFS`.
- **Option B (Correct):** `=SUMIFS(sum_range, criteria_range1, criterion1, criteria_range2, criterion2)` — exactly the right function and argument structure. It sums F (Amount) where C (Region) = "East" AND D (Status) = "Completed". East+Completed rows: O07 (67000). The correct sum is **67000**.
- **Option C (Incorrect):** This formula adds two separate SUMIF results: total East amounts + total Completed amounts. This implements OR logic (sum of East OR Completed), and the two sums overlap — rows that are both East AND Completed are counted twice. This does not return the AND intersection the analyst needs.
- **Option D (Incorrect):** COUNTIFS with three argument pairs would count rows matching East AND Completed AND (some criterion against F). As written with F2:F11 as a range but no criterion following it, this is invalid syntax. Even if syntactically fixed, COUNTIFS counts matching rows — it does not sum amounts.

</details>

---

### 43. Question

A pivot table on the **Dashboard** sheet shows the following layout: rows = Region, columns = Segment, values = Sum of Amount. A formula on another sheet reads `=Dashboard!C3`. When the pivot table is refreshed with new data that adds a previously unseen region ("West"), the pivot table grows and West is inserted alphabetically between South and the existing rows.

After the refresh and re-layout, what does `=Dashboard!C3` return?

- [ ] **A.** The same Sum of Amount value it showed before, because pivot table data is frozen at the time of formula creation
- [ ] **B.** Whatever value now occupies cell C3 of the Dashboard sheet — which may now be a different region's data if the layout shifted
- [ ] **C.** `#REF!` — because pivot table refreshes invalidate all external references to pivot cells
- [ ] **D.** `#N/A` — because the referenced pivot table row no longer exists at position C3

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Cross-sheet cell references are live — they always return the current value of the referenced cell, not a frozen snapshot from when the formula was written. Pivot table data is not frozen in a cross-sheet reference context.
- **Option B (Correct):** The reference `=Dashboard!C3` points to a fixed physical address: column C, row 3 of the Dashboard sheet. When new data causes the pivot table to insert "West" alphabetically (pushing other regions down), the region that previously occupied row 3 may now be in row 4. Cell C3 now contains the data for whatever row 3 holds post-refresh — silently returning wrong data without any error. This is the key fragility of cross-sheet pivot references.
- **Option C (Incorrect):** Pivot table refreshes do not invalidate cross-sheet references. The cell C3 still exists; it just may contain different data. `#REF!` occurs when the referenced sheet is deleted or the range is structurally broken (e.g., entire row deletion).
- **Option D (Incorrect):** `#N/A` is a VLOOKUP-specific "not found" error. Cross-sheet cell references do not return `#N/A` — they return whatever the target cell currently contains, regardless of what that value represents.

</details>

---

### 44. Question

A junior analyst writes `=COUNTIFS(C2:C11, "North", D2:D11, "Completed", E2:E11, "SMB")` on the Orders table from Question 31. What does this formula return?

- [ ] **A.** 3 — counting all North orders
- [ ] **B.** 5 — counting all Completed orders
- [ ] **C.** 1 — counting orders that are simultaneously North AND Completed AND SMB
- [ ] **D.** 0 — because COUNTIFS cannot handle three conditions simultaneously

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Counting all North orders (regardless of status/segment) uses `=COUNTIF(C2:C11, "North")`. This COUNTIFS formula adds two more conditions, narrowing the count to only rows satisfying all three criteria simultaneously.
- **Option B (Incorrect):** Counting all Completed orders uses `=COUNTIF(D2:D11, "Completed")`. The formula here adds Region="North" and Segment="SMB" as additional AND filters.
- **Option C (Correct):** COUNTIFS applies AND logic — all three conditions must be true in the same row. Scanning the Orders table: O03 is (North, Completed, SMB) — this is the only row satisfying all three simultaneously. (O01 is North+Completed but Enterprise; O09 is North+Completed but Enterprise.) Count = **1**.
- **Option D (Incorrect):** COUNTIFS can handle any number of condition pairs (limited only by Excel/Google Sheets argument limits, which are 127 pairs). Three condition pairs is well within the allowed range. The function operates normally with three criteria.

</details>

---

### 45. Question

A data analyst examines a colleague's formula: `=SUMIFS(E2:E11, C2:C11, "South", D2:D11, "Pending", C2:C11, "East")`. What does this formula compute, given the Orders table in Question 31?

- [ ] **A.** The total amount for orders in the South OR East region that are Pending
- [ ] **B.** The total amount for orders in the South region that are Pending plus orders in the East region regardless of status
- [ ] **C.** 0 — because a single row cannot simultaneously have Region="South" AND Region="East"; the AND conditions are mutually exclusive
- [ ] **D.** 22000 — the amount of the East Pending order

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** SUMIFS applies AND logic across all criteria pairs — it does not apply OR logic. The formula does not sum South OR East orders; it requires all conditions to be satisfied in the same row.
- **Option B (Incorrect):** The formula does not mix logic types — all three conditions are ANDed together. It doesn't add separate sums; it multiplies filters.
- **Option C (Correct):** The formula requires Region="South" AND Status="Pending" AND Region="East" simultaneously. A single cell in column C can only hold one value — no row can be both "South" AND "East" at the same time. The AND logic makes these conditions mutually exclusive, so zero rows satisfy all three conditions simultaneously. SUMIFS returns **0**.
- **Option D (Incorrect):** 22000 is the amount for O10 (East, Pending), but O10's Region is "East", not "South". The formula requires Region="South" as its first condition, excluding O10. No row is South, Pending, AND East simultaneously.

</details>

---

### 46. Question

An analyst is told that `SUMIF` places its optional sum_range as the **third** argument, while `SUMIFS` places the sum_range as the **first** argument. A colleague disputes this, saying "the order doesn't matter as long as the right ranges are used." Who is correct?

- [ ] **A.** The colleague is correct — modern spreadsheet applications automatically detect which argument is the sum range
- [ ] **B.** The analyst is correct — argument order is strictly defined and fixed in both functions; mixing up the order causes wrong results or errors
- [ ] **C.** The analyst is correct only for SUMIF; SUMIFS argument order is flexible
- [ ] **D.** Both are wrong — SUMIF and SUMIFS have identical argument order

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Spreadsheet applications do not "auto-detect" argument roles. Each positional argument in a function has a fixed, defined purpose. `=SUMIFS(E2:E11, "Completed", D2:D11)` would place a criteria string where the first criteria_range is expected, causing a `#VALUE!` error. Argument position is strictly meaningful.
- **Option B (Correct):** The analyst is precisely correct. **SUMIF** syntax: `SUMIF(criteria_range, criterion, [sum_range])` — sum_range is the third (optional) argument. **SUMIFS** syntax: `SUMIFS(sum_range, criteria_range1, criterion1, [criteria_range2, criterion2], ...)` — sum_range is the **first** mandatory argument. This positional difference is one of the most common sources of confusion and errors when switching between the two functions. The argument order is rigid and non-negotiable.
- **Option C (Incorrect):** SUMIFS argument order is just as strictly defined as SUMIF's. The sum_range must be the first argument in SUMIFS. There is no flexibility in either function's argument ordering.
- **Option D (Incorrect):** SUMIF and SUMIFS deliberately have **different** argument orders. This asymmetry is a known design inconsistency in the Excel function library that frequently trips up analysts. They are not identical.

</details>

---

### 47. Question

Using the Orders table from Question 31, which formula returns the **count of SMB orders that are NOT Completed**?

- [ ] **A.** `=COUNTIFS(E2:E11, "SMB", D2:D11, "<>Completed")`
- [ ] **B.** `=COUNTIF(E2:E11, "SMB") - COUNTIFS(E2:E11, "SMB", D2:D11, "Completed")`
- [ ] **C.** `=COUNTIFS(E2:E11, "SMB") - COUNTIF(D2:D11, "Completed")`
- [ ] **D.** `=COUNTIF(D2:D11, "<>Completed", E2:E11, "SMB")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** `=COUNTIFS(E2:E11, "SMB", D2:D11, "<>Completed")` uses the not-equal operator `<>` within the criteria string to count rows where Segment="SMB" AND Status is anything except "Completed". In Excel/Google Sheets, `<>Completed` (or `"<>Completed"`) is a valid COUNTIFS criterion meaning "not equal to Completed". SMB rows: O02(Pending✓), O03(Completed✗), O06(Pending✓), O08(Cancelled✓), O10(Pending✓) → count = **4**.
- **Option B (Correct):** An alternative correct approach: (all SMB orders) − (SMB + Completed orders) = not-Completed SMB orders. All SMB: O02,O03,O06,O08,O10 = 5. SMB+Completed: O03 = 1. Result: 5 − 1 = **4**. This subtraction method is equally valid and produces the same answer.
- **Option C (Incorrect):** This subtracts the count of ALL Completed orders from ALL SMB orders. The subtraction logic is flawed — it removes rows that may not be SMB, or removes too many/few rows. For example, there are 5 Completed orders (not just 1 Completed+SMB), so 5 − 5 = 0, which is wrong.
- **Option D (Incorrect):** `COUNTIF` accepts only two arguments: a single `(range, criteria)` pair. Passing four arguments as shown is invalid syntax for COUNTIF. Multi-criteria counting requires COUNTIFS.

</details>

---

### 48. Question

A data analyst wants to use a cell reference (rather than a hardcoded string) as the criterion in AVERAGEIF. The target region is stored in cell H1. Which formula correctly averages Amount (F) for orders where Region (C) matches whatever is in H1?

- [ ] **A.** `=AVERAGEIF(C2:C11, H1, F2:F11)`
- [ ] **B.** `=AVERAGEIF(C2:C11, "H1", F2:F11)`
- [ ] **C.** `=AVERAGEIF(C2:C11, &H1, F2:F11)`
- [ ] **D.** `=AVERAGEIF(C2:C11, =H1, F2:F11)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Criteria in SUMIF/COUNTIF/AVERAGEIF can be a direct cell reference without any quotes. `=AVERAGEIF(C2:C11, H1, F2:F11)` reads the current value of H1 and uses it as the criterion. When H1 contains "North", the formula averages amounts for North orders. If H1 changes to "South", the formula automatically recalculates for South orders.
- **Option B (Incorrect):** Wrapping H1 in quotes (`"H1"`) makes it a literal text criterion — the formula searches for cells containing the text "H1" in the Region column, finding none and returning an error or 0. The cell reference H1 must be unquoted to be interpreted as a reference.
- **Option C (Incorrect):** `&H1` is not valid criteria syntax in AVERAGEIF. The `&` operator is used for text concatenation in formula strings (e.g., `">"&H1` for comparison criteria like "greater than H1"), but a bare `&H1` as a standalone argument is a syntax error.
- **Option D (Incorrect):** `=H1` is not valid inside a function's argument list as a criterion. The `=` operator within an argument position is not interpreted as an equality comparison criterion in AVERAGEIF — it is a syntax error. The correct way to reference a cell is simply the cell address: `H1`.

</details>

---

### 49. Question

A business analyst builds a pivot table from the Orders table (Question 31). She places **Amount** in the Values area. The pivot table currently shows **Sum of Amount**. She wants to change it to show **Average of Amount** instead. What is the correct way to achieve this?

- [ ] **A.** Delete Amount from Values and drag Region into Values instead
- [ ] **B.** Right-click (or click the dropdown on) the "Sum of Amount" field in the Values area and change the summarization method to "Average"
- [ ] **C.** Use `=AVERAGEIF()` in a separate cell and link the pivot table Values area to that cell
- [ ] **D.** Move Amount from the Values area to the Rows area; average calculations are only available in the Rows area

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Deleting Amount from Values removes the numeric aggregation entirely. Dragging Region into Values would count or aggregate region text labels, not average amounts. The analyst needs to keep Amount in Values but change the aggregation method.
- **Option B (Correct):** In both Excel and Google Sheets, the summarization function for a pivot table's Values field can be changed through the field's settings. Right-clicking the field (or using the dropdown in the PivotTable Field List / pivot table editor) and selecting "Summarize values by → Average" (or "Average" from the menu) switches the calculation from Sum to Average without moving or removing any fields.
- **Option C (Incorrect):** AVERAGEIF formulas live outside the pivot table and cannot directly replace the Values area's aggregation method. This approach bypasses the pivot table and creates a separate formula, defeating the purpose of the pivot table's built-in aggregation options.
- **Option D (Incorrect):** The Rows area contains categorical fields that become row headers — it does not perform or display averages. All numeric aggregations (Sum, Count, Average, Max, Min) are configured in the **Values** area. Amount must stay in Values; only the aggregation function needs to change.

</details>

---

### 50. Question

A data analyst is told that a pivot table is "live" but that cross-sheet cell references to pivot table cells are "fragile." Which of the following correctly explains the distinction?

- [ ] **A.** Both are live — pivot tables update automatically and cell references always track the correct pivot data regardless of layout changes
- [ ] **B.** The pivot table data is live (it updates when its source data changes and is refreshed), but a formula like `=PivotSheet!C5` references a fixed physical cell address — if the pivot table's layout shifts (new rows, sorted differently), that fixed address may silently point to different data than before
- [ ] **C.** Pivot tables are not live — they must be manually rebuilt each time source data changes. Cross-sheet references are the correct way to get live data from pivot results
- [ ] **D.** Cross-sheet references to pivot tables automatically track the labeled dimension (e.g., always returning "North" region data from a pivoted Region field)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Cell references do not "track" pivot data labels — they track fixed cell addresses. When the pivot layout changes, the cell at a fixed address may contain completely different data. They are not self-correcting references.
- **Option B (Correct):** This precisely describes the distinction. A pivot table's values are dynamic in the sense that refreshing it updates computed aggregates from the source data. However, a formula like `=PivotSheet!C5` is a hard-coded address pointing to row 5, column C — nothing more. If the pivot table changes its row structure (e.g., a new region is inserted, or regions are sorted differently), the data at physical address C5 changes silently. The formula does not know or care about the pivot field labels — it only reads the cell value at that address.
- **Option C (Incorrect):** Pivot tables in Excel and Google Sheets **are** live in the sense that they recalculate their aggregates when refreshed. They are not static tables that must be manually rebuilt. The refresh button or automatic refresh updates the pivot values. Cross-sheet references are not the recommended mechanism for accessing dynamic pivot data.
- **Option D (Incorrect):** Standard cross-sheet cell references do not track dimension labels. There is no built-in mechanism in standard Excel/Google Sheets formulas for `=PivotSheet!C5` to self-correct and follow the "North" row wherever it moves. Advanced techniques (like using GETPIVOTDATA function) can retrieve labeled pivot data, but standard cell references cannot.

</details>
