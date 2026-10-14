# Spreadsheet Objective Test — Set 07

---

### 61. Question

A data analyst is enriching an Orders sheet (col A=OrderID, col B=ProductSKU) with data from a Products reference table on a sheet named **Products** (col A=SKU, col B=Name, col C=Category, col D=UnitCost). She enters in cell C2 of the Orders sheet:

```
=VLOOKUP(B2, Products!$A$2:$D$50, 3, FALSE)
```

And copies it down to C100. At row C45, the formula returns "Clothing" — the correct category for the SKU in B45. At row C61, the formula returns `#N/A`. The SKU in B61 is "SKU-999". She checks and confirms that "SKU-999" is **not** in the Products table. Which action should she take?

- [ ] **A.** Change `FALSE` to `TRUE` in the formula so that it finds the nearest match for "SKU-999"
- [ ] **B.** Wrap the formula with `IFERROR` to display a meaningful label instead of `#N/A`: `=IFERROR(VLOOKUP(B2, Products!$A$2:$D$50, 3, FALSE), "Unknown Product")`
- [ ] **C.** Change `col_index_num` from 3 to 2 so the formula looks at the Name column instead
- [ ] **D.** Extend the Products table_array to include column E so VLOOKUP has a wider search area

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Switching to approximate match (`TRUE`) does not add "SKU-999" to the Products table. It changes the matching algorithm to find the nearest value in a sorted column — but since the SKU genuinely does not exist, approximate match would return data from a wrong (nearest) row, silently giving incorrect product details. This is worse than `#N/A` because the error becomes invisible.
- **Option B (Correct):** `IFERROR(formula, value_if_error)` intercepts any error (including `#N/A`) returned by the inner formula and substitutes a user-defined fallback value instead. `=IFERROR(VLOOKUP(B61, Products!$A$2:$D$50, 3, FALSE), "Unknown Product")` cleanly handles the missing SKU by displaying "Unknown Product" while preserving correct results for all valid SKUs. This is the standard production-grade approach to handling expected lookup failures.
- **Option C (Incorrect):** Changing `col_index_num` from 3 (Category) to 2 (Name) changes which column is returned for rows that DO match — it has no effect on rows that don't match. The `#N/A` for "SKU-999" occurs because the SKU is not found, not because the wrong column is requested.
- **Option D (Incorrect):** Extending the table_array to include column E gives VLOOKUP access to additional columns to return, but it does not add "SKU-999" as a row in the table. The lookup key must exist in the leftmost column of `table_array`; widening the range does not help when the key is absent.

</details>

---

### 62. Question

A finance analyst has a Payroll sheet where column A holds EmployeeID (rows 2–51). She wants to pull each employee's Department from an HR reference table on a sheet named **HR** (A=EmpID, B=Name, C=Department, D=Grade; rows 2–200). She writes the formula in column B of the Payroll sheet and copies it down. Which formula is correct in B2?

- [ ] **A.** `=VLOOKUP(A2, HR!A2:D200, 3, FALSE)`
- [ ] **B.** `=VLOOKUP($A$2, HR!$A$2:$D$200, 3, FALSE)`
- [ ] **C.** `=VLOOKUP(A2, HR!$A$2:$D$200, 3, FALSE)`
- [ ] **D.** `=VLOOKUP(A2, HR!$A$2:$D$200, "C", FALSE)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `HR!A2:D200` is fully relative. When copied down, the table_array shifts — row B3's formula uses `HR!A3:D201`, row B10 uses `HR!A10:D208`, etc. This pushes the search range progressively outside the actual HR data, producing `#N/A` for many employees.
- **Option B (Incorrect):** Locking the lookup value as `$A$2` means every formula in the column looks up the same employee (row 2's ID) regardless of which row the formula is in. Only row B2 returns the correct department; all other rows return the department for employee in A2. The lookup value must be a relative row reference (A2, A3, A4...) so each row looks up its own EmployeeID.
- **Option C (Correct):** `A2` is a relative reference that correctly shifts to `A3`, `A4`, etc. as the formula is copied down — each row looks up its own EmployeeID. `HR!$A$2:$D$200` is an absolute reference that stays fixed regardless of copy position, always searching the complete HR table. `col_index_num = 3` returns Department (column C of the range). `FALSE` ensures exact ID matching.
- **Option D (Incorrect):** `col_index_num` must be a positive integer. Passing `"C"` (a column letter string) as the third argument is invalid syntax and causes a `#VALUE!` error. VLOOKUP counts columns numerically from the start of table_array, not by worksheet column letter.

</details>

---

### 63. Question

A DevOps engineer analyzing spreadsheet formulas finds this formula: `=SUMIF(B2:B20, ">=10000")`. The column B holds numeric values. What does this formula do?

- [ ] **A.** Counts the number of rows where B is greater than or equal to 10000
- [ ] **B.** Returns `#VALUE!` because SUMIF with a numeric comparison criterion requires a third argument
- [ ] **C.** Sums the values in B2:B20 that are greater than or equal to 10000 (sum_range defaults to the criteria range itself when omitted)
- [ ] **D.** Returns 10000 — the threshold value used as the criterion

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Counting rows meeting a numeric threshold is `=COUNTIF(B2:B20, ">=10000")`. SUMIF without a third argument sums the qualifying values in the criteria range itself, not counts them.
- **Option B (Incorrect):** SUMIF's third argument (`sum_range`) is optional — omitting it is documented behavior. When omitted, SUMIF uses the `range` as both the criteria range and the sum range. No error is raised. The criterion `">=10000"` is valid syntax for a numeric comparison.
- **Option C (Correct):** When `sum_range` is omitted from SUMIF, the function evaluates each cell in `range` (B2:B20) against the criterion and sums the cells that qualify — the range and sum range are the same. `">=10000"` is a valid comparison criterion: cells with values ≥ 10000 are included in the sum. This is the documented default behavior for SUMIF with a missing third argument.
- **Option D (Incorrect):** 10000 is the threshold value written in the criterion string, not the formula's result. The formula computes a sum of qualifying data values, which is determined by the actual data in B2:B20, not by the criterion constant.

</details>

---

### 64. Question

A senior analyst compares how VLOOKUP handles **case sensitivity**. She has a lookup column containing "Widget", "WIDGET", and "widget" as three distinct rows. She uses `=VLOOKUP("widget", $A$2:$D$10, 2, FALSE)`. Which row does VLOOKUP match?

- [ ] **A.** Only the row with lowercase "widget" — VLOOKUP is case-sensitive
- [ ] **B.** All three rows — VLOOKUP returns multiple results for case-variant matches
- [ ] **C.** The first row among "Widget", "WIDGET", and "widget" (scanning top-to-bottom) that it encounters — VLOOKUP is case-insensitive
- [ ] **D.** VLOOKUP returns `#N/A` when multiple case-variant matches exist

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** VLOOKUP is **case-insensitive** in both Excel and Google Sheets. It treats "widget", "Widget", and "WIDGET" as identical strings. Searching for "widget" will match any of these three representations.
- **Option B (Incorrect):** VLOOKUP always returns a single value from a single matched row. It stops at the first match it finds (scanning top-to-bottom) and does not collect or return multiple matching rows.
- **Option C (Correct):** VLOOKUP performs case-insensitive text comparison. "widget", "Widget", and "WIDGET" are all considered equal. VLOOKUP finds the **first** occurrence scanning from the top of the search range and returns the value from column 2 of that first-matching row.
- **Option D (Incorrect):** Multiple case-variant matches do not cause `#N/A`. VLOOKUP silently returns the first match without error. This case-insensitivity is why lookup keys should ideally be unique and consistently formatted — duplicates (including case variants) can produce silently incorrect results.

</details>

---

### 65. Question

A business analyst wants to verify a SUMIFS formula by manually tracing through this small **Sales** table (columns A–D, rows 2–7):

| Row | Salesperson | Quarter | Product  | Revenue |
| :-- | :---------- | :------ | :------- | :------ |
| 2   | Alice       | Q1      | Laptop   | 50000   |
| 3   | Bob         | Q2      | Tablet   | 18000   |
| 4   | Alice       | Q2      | Laptop   | 62000   |
| 5   | Charlie     | Q1      | Tablet   | 14000   |
| 6   | Bob         | Q1      | Laptop   | 38000   |
| 7   | Alice       | Q1      | Tablet   | 9000    |

What does `=SUMIFS(D2:D7, A2:A7, "Alice", C2:C7, "Laptop")` return?

- [ ] **A.** 121000 — the total revenue for all Alice's sales
- [ ] **B.** 150000 — the total revenue for all Laptop sales
- [ ] **C.** 112000 — the total revenue for Laptop sales by Alice
- [ ] **D.** 62000 — only Alice's highest Laptop sale

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 121000 is Alice's total revenue across all products: 50000 (Q1, Laptop) + 62000 (Q2, Laptop) + 9000 (Q1, Tablet) = 121000. SUMIFS also requires Product="Laptop", excluding the Tablet row (9000).
- **Option B (Incorrect):** 150000 is total Laptop revenue: 50000 (Alice) + 62000 (Alice) + 38000 (Bob) = 150000. SUMIFS also requires Salesperson="Alice", excluding Bob's Laptop row (38000).
- **Option C (Correct):** Alice+Laptop rows: row 2 (Alice, Q1, Laptop, 50000 ✓) and row 4 (Alice, Q2, Laptop, 62000 ✓). Sum = 50000 + 62000 = **112000**.
- **Option D (Incorrect):** 62000 is Alice's single highest Laptop sale. SUMIFS sums all qualifying rows, not just the maximum. Both 50000 and 62000 qualify, so the total is 112000.

</details>

---

### 66. Question

A data analyst is building a multi-sheet dashboard. Sheet 1 contains raw Orders data; Sheet 2 contains a pivot table; Sheet 3 contains summary formulas. She writes `=Sheet2!D7` in Sheet 3 to pull a specific pivot table value. A colleague warns her that this reference is fragile. Which of the following modifications would make the dashboard more robust against pivot layout changes?

- [ ] **A.** Add `$` signs: `=Sheet2!$D$7` — this prevents the reference from shifting when copied
- [ ] **B.** Use the `GETPIVOTDATA` function to retrieve pivot data by field name and item value rather than by cell address
- [ ] **C.** Replace the pivot table with a static copy (Paste Special → Values Only) to prevent the layout from ever changing
- [ ] **D.** Add the formula to the pivot table's own sheet to reduce cross-sheet reference lag

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Adding `$` signs (making the reference `$D$7`) prevents the reference from **shifting when copied**, which is a different problem. The fragility issue is that pivot table layout changes move which cell holds the target value — locking `$D$7` still points to the same physical cell. If the pivot inserts a row above row 7, the target value moves to D8 while `$D$7` still reads the now-incorrect D7.
- **Option B (Correct):** `GETPIVOTDATA` is specifically designed to retrieve pivot table data by specifying the pivot field and item labels (e.g., "Sum of Revenue" for "North" region in "Q1" quarter) rather than by physical cell address. When the pivot layout changes, GETPIVOTDATA recalculates the correct cell address internally and returns the correct value as long as the specified field/item combination still exists. This is the recommended robust alternative to hardcoded cross-sheet cell references.
- **Option C (Incorrect):** Converting the pivot table to static values eliminates all dynamic behavior — it can never be refreshed with new data. While this does make the cell addresses stable (the layout won't change), it defeats the entire purpose of a live dashboard. The summary formulas would show stale data after each data update.
- **Option D (Incorrect):** Moving the referencing formula to the same sheet as the pivot table eliminates the "cross-sheet" aspect but not the fragility — a same-sheet direct cell reference `=D7` is just as vulnerable to pivot layout changes as a cross-sheet one. The fragility comes from referencing a fixed address, not from the cross-sheet aspect.

</details>

---

### 67. Question

A junior analyst copies a formula across columns (left to right) rather than down rows. The formula in cell E2 is `=SUMIF($A$2:$A$20, B1, $C$2:$C$20)`. She copies it from E2 to F2, G2, and H2. What happens to the formula as it is copied to the right?

- [ ] **A.** The criteria reference `B1` shifts to `C1`, `D1`, `E1` in each subsequent column — allowing each column to use a different criterion from row 1
- [ ] **B.** The criteria reference `B1` stays locked at `B1` in all columns because horizontal copying does not shift row references
- [ ] **C.** The formula fails in F2 because SUMIF cannot be copied horizontally
- [ ] **D.** The entire formula stays identical in all columns because both `$A$2:$A$20` and `$C$2:$C$20` are absolute

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `B1` is a fully relative reference (no `$`). When the formula is copied one column to the right (from E to F), both the column and row of relative references shift. Column shifts by 1: `B1` → `C1`. Copied to G2: `C1` → `D1`. Copied to H2: `D1` → `E1`. The locked ranges `$A$2:$A$20` and `$C$2:$C$20` remain fixed. This is a deliberate design pattern: the criterion shifts across columns while the data ranges stay anchored, allowing each column to filter by a different criterion read from row 1.
- **Option B (Incorrect):** Row references (the `1` in `B1`) do not shift when copying horizontally — only column letters shift. The `1` stays `1`, but the `B` shifts to `C`, `D`, `E`. The combined effect is that `B1` → `C1` → `D1` → `E1`. Row locking is controlled by `$` before the row number (e.g., `B$1` would keep it on row 1 regardless of vertical copy direction).
- **Option C (Incorrect):** SUMIF formulas can be copied in any direction — horizontally, vertically, or diagonally. The copy behavior is determined entirely by which references are relative vs. absolute, not by the direction of copying.
- **Option D (Incorrect):** While `$A$2:$A$20` and `$C$2:$C$20` are absolute and do stay fixed, the criterion reference `B1` is not absolute — it shifts with the copy direction. The formula does change in each column because of this relative criterion reference.

</details>

---

### 68. Question

A data analyst has an Orders sheet with data in columns A–G (rows 2–101). She wants to count how many orders have **Amount (column F) greater than 50000** AND **Segment (column E) equal to "Enterprise"**. Which formula is correct?

- [ ] **A.** `=COUNTIFS(F2:F101, ">50000", E2:E101, "Enterprise")`
- [ ] **B.** `=COUNTIF(F2:F101, ">50000") AND COUNTIF(E2:E101, "Enterprise")`
- [ ] **C.** `=COUNTIF(F2:F101, ">50000", E2:E101, "Enterprise")`
- [ ] **D.** `=SUMIFS(F2:F101, F2:F101, ">50000", E2:E101, "Enterprise")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `=COUNTIFS(criteria_range1, criterion1, criteria_range2, criterion2)` — exactly the right multi-condition counting formula. It counts rows where F > 50000 AND E = "Enterprise" simultaneously. The `">50000"` criterion uses a comparison operator enclosed in quotes, which is valid COUNTIFS syntax.
- **Option B (Incorrect):** `COUNTIF(...) AND COUNTIF(...)` is not valid formula syntax. `AND` is a logical function that returns TRUE or FALSE (evaluating whether two conditions are both true), not an arithmetic operator that combines two counts. Counting rows meeting multiple numeric and text criteria requires COUNTIFS, not multiple COUNTIF formulas connected by AND.
- **Option C (Incorrect):** `COUNTIF` accepts only two arguments: a single `(range, criteria)` pair. Passing four arguments as written exceeds COUNTIF's parameter limit and causes a `#VALUE!` error. Multi-criteria counting must use `COUNTIFS`.
- **Option D (Incorrect):** `SUMIFS(F2:F101, F2:F101, ">50000", E2:E101, "Enterprise")` sums the Amount values for qualifying rows (not counts them). The question asks for a count, not a sum. The result would be the total revenue of qualifying orders, not the number of qualifying orders.

</details>

---

### 69. Question

A senior data analyst runs the following two formulas on a dataset where column B holds a Region field and column C holds a Revenue field (100 rows, rows 2–101):

- **Formula X:** `=SUMIF(B2:B101, "North", C2:C101)` → returns 320000
- **Formula Y:** `=COUNTIF(B2:B101, "North")` → returns 8

What does `=AVERAGEIF(B2:B101, "North", C2:C101)` return?

- [ ] **A.** 2560000 — because AVERAGEIF multiplies the SUMIF result by COUNTIF
- [ ] **B.** 40000 — the mean revenue for North orders
- [ ] **C.** 328000 — the sum plus count
- [ ] **D.** 8 — because AVERAGEIF returns the same result as COUNTIF for text criteria

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** AVERAGEIF computes the mean by dividing the sum of matching values by the count of matching rows — it does not multiply them. 320000 × 8 = 2,560,000, which would be a nonsensical result.
- **Option B (Correct):** AVERAGEIF computes the arithmetic mean of C values where B = "North". This is equivalent to SUMIF / COUNTIF = 320000 / 8 = **40000**. AVERAGEIF produces the revenue per order for North region orders.
- **Option C (Incorrect):** 328000 = 320000 + 8 is simply the sum plus the count, which has no mathematical meaning as an average. AVERAGEIF divides, not adds.
- **Option D (Incorrect):** AVERAGEIF and COUNTIF return fundamentally different types of output. COUNTIF returns a count (number of matching rows = 8). AVERAGEIF returns an average value from a separate numeric range (= 40000). They are not equal unless the sum range happens to contain values whose mean equals the count, which is coincidental.

</details>

---

### 70. Question

A data analyst is setting up a Regional Sales Dashboard pivot table. She has the following fields available: OrderID, OrderDate, Region, Salesperson, Segment, Product, Quantity, Revenue, Status.

She wants the pivot table to show: **total Revenue and total Quantity, broken down by Region (rows), restricted to only "Completed" orders**. Which of the following field placements achieves this?

- [ ] **A.** Region → Rows; Revenue → Values (Sum); Quantity → Values (Sum); Status → Filters (select "Completed")
- [ ] **B.** Region → Columns; Revenue → Values (Sum); Quantity → Rows; Status → Filters
- [ ] **C.** Region → Values; Revenue → Rows; Quantity → Columns; Status → Filters
- [ ] **D.** Status → Rows; Revenue → Values (Sum); Quantity → Values (Sum); Region → Filters (select all regions)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** This is the exact correct layout. **Region → Rows** creates one row per region (North, South, East, etc.). **Revenue → Values (Sum)** and **Quantity → Values (Sum)** place both numeric measures in the aggregation area, each computing a sum. **Status → Filters** adds a filter control above the pivot table; selecting "Completed" restricts all displayed data to only completed orders. This produces a clean regional summary table for completed orders.
- **Option B (Incorrect):** Placing Region in Columns and Quantity in Rows creates an awkward layout where columns are broken out by region (fine) but Quantity becomes a row dimension instead of an aggregated value. Quantity is a numeric measure and belongs in the Values area for summation, not in Rows as a categorical grouping.
- **Option C (Incorrect):** Placing Region in Values would attempt to sum or count Region text labels, producing a Count of Region rather than any meaningful breakdown. Revenue should be in Values (not Rows), and Quantity should be in Values (not Columns). The entire area assignment is reversed.
- **Option D (Incorrect):** Placing Status in Rows creates one row per status (Completed, Pending, Cancelled) — useful for a status breakdown but not a regional breakdown. Region in Filters would allow filtering to a single region, losing the multi-region comparison the analyst wants. The requirement is Region in Rows and Status in Filters, not the reverse.

</details>
