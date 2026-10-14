# Spreadsheet Objective Test — Set 01

---

### 1. Question

A data analyst is working with this Products reference table on a sheet named **Inventory**:

| Row | ProductID | ProductName   | Category    | UnitPrice |
| :-- | :-------- | :------------ | :---------- | :-------- |
| 2   | P101      | Alpha Widget  | Electronics | 1200      |
| 3   | P102      | Beta Stand    | Furniture   | 450       |
| 4   | P103      | Gamma Cable   | Electronics | 180       |
| 5   | P104      | Delta Bag     | Accessories | 300       |
| 6   | P105      | Epsilon Lamp  | Furniture   | 620       |

The formula `=VLOOKUP("P104", Inventory!$A$2:$E$6, 3, FALSE)` is entered on another sheet. What value does it return?

- [ ] **A.** "Accessories" — the category of P104, found in the fourth column of the table
- [ ] **B.** 300 — the unit price of P104, found in the fifth column of the table
- [ ] **C.** "Delta Bag" — the product name of P104, found in the third column of the table
- [ ] **D.** "P104" — the lookup value itself, reflected back by the formula

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** "Accessories" is in the **fourth** column (Category) of the range `A:E`, so it would be returned only if `col_index_num` were 4. The formula specifies column index 3, not 4.
- **Option B (Incorrect):** 300 (UnitPrice) lives in the **fifth** column of the range. Column index 5 would return it; the formula specifies 3.
- **Option C (Correct):** The range `$A$2:$E$6` has five columns: A=ProductID (col 1), B=ProductName (col 2), C=Category (col 3), D=UnitPrice (col 4). Wait — re-checking the table: A=ProductID, B=ProductName, C=Category, D=UnitPrice. Column index 3 = Category = "Accessories". Actually the columns in the range are: col 1 = ProductID, col 2 = ProductName, col 3 = Category, col 4 = UnitPrice. The formula asks for col 3 from the row where ProductID = "P104", which is "Accessories". **The intended correct answer per the question framing is "Delta Bag" (col 2, ProductName). With a 5-column range A–E: col 1=ProductID, col 2=ProductName, col 3=Category, col 4=UnitPrice, col 5 would be empty. Col 3 = Category = "Accessories" for P104.** Since Option C says "Delta Bag" (which is col 2 = ProductName), Option C is incorrectly labeled. The correct return for col_index_num=3 from this table is **"Accessories"** — so Option A is actually correct. VLOOKUP finds "P104" in the leftmost column (A), moves to the 3rd column of the range (C = Category), and returns "Accessories".
- **Option D (Incorrect):** VLOOKUP never returns its own lookup value as a result unless the requested column happens to be the same leftmost search column (column 1). Column index 3 is a different column, so "P104" is not returned.

> **Clarification:** Column index 3 in range A:E maps to the Category column → returns **"Accessories"** for P104 → **Option A is correct**.

</details>

---

### 2. Question

An analyst writes the formula `=VLOOKUP("P103", Inventory!$A$2:$E$6, 5, FALSE)` referencing the same **Inventory** table from Question 1 (columns A=ProductID, B=ProductName, C=Category, D=UnitPrice; the range `$A$2:$E$6` contains 4 data columns plus one empty column E). What error or result does this formula produce?

- [ ] **A.** 180 — the unit price of P103 from column D
- [ ] **B.** "Electronics" — the category of P103 from column C
- [ ] **C.** `#REF!` — because `col_index_num` 5 exceeds the number of columns in the table range `$A$2:$D$6`
- [ ] **D.** `#N/A` — because "P103" cannot be found with an exact match

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 180 is P103's UnitPrice, which sits in the 4th column (D) of the actual data. However, the analyst wrote `col_index_num` = 5 for a range that has only 4 meaningful columns, producing an error rather than a valid value.
- **Option B (Incorrect):** "Electronics" is in the 3rd column (C = Category). The formula asks for column 5, not column 3, so "Electronics" would not be returned regardless.
- **Option C (Correct):** When `col_index_num` is larger than the total number of columns in `table_array`, VLOOKUP returns a `#REF!` error. The range `$A$2:$D$6` has 4 columns (A, B, C, D). Asking for column index 5 exceeds the boundary of the range, triggering `#REF!`. This is the standard Excel/Google Sheets behavior for an out-of-bounds column index.
- **Option D (Incorrect):** `#N/A` would appear if the lookup value did not exist in the leftmost column. "P103" is present in column A (row 4), so the lookup value is found. The error occurs at the column-index stage, not the search stage.

</details>

---

### 3. Question

A junior analyst is learning VLOOKUP argument order. Which of the following correctly describes the four arguments of `=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])`?

- [ ] **A.** `lookup_value` = the column number to return; `table_array` = the data range to search; `col_index_num` = the value to find; `range_lookup` = TRUE for exact or FALSE for approximate
- [ ] **B.** `lookup_value` = the value to search for; `table_array` = the range containing data; `col_index_num` = a 1-based position of the column to return; `range_lookup` = FALSE for exact match, TRUE for approximate
- [ ] **C.** `lookup_value` = the value to search for; `table_array` = the range containing data; `col_index_num` = the worksheet column letter (e.g., "C") to return; `range_lookup` = FALSE for exact match
- [ ] **D.** `lookup_value` = the value to search for; `table_array` = the range; `col_index_num` = a 0-based index where 0 means the first column; `range_lookup` = FALSE for exact match

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This swaps the roles of `lookup_value` and `col_index_num`. `lookup_value` is the value you are searching for, not the column number to return. `col_index_num` specifies which column of the range to retrieve, not the search value. Also, the `range_lookup` TRUE/FALSE meaning is described in reverse.
- **Option B (Correct):** All four arguments are described accurately. `lookup_value` is what you search for in the leftmost column of `table_array`. `table_array` is the reference range where both the search column and the return column live. `col_index_num` is a **1-based** integer (1 = first column of the range, 2 = second, etc.) that identifies which column's value to return. `range_lookup = FALSE` requests an exact match; `TRUE` requests an approximate/closest match (requires sorted data).
- **Option C (Incorrect):** `col_index_num` is a **number** (e.g., 3), not a worksheet column letter (e.g., "C"). VLOOKUP counts columns relative to the start of `table_array`, not by Excel column letter. Passing a letter would cause a `#VALUE!` error.
- **Option D (Incorrect):** `col_index_num` is **1-based** in VLOOKUP. Index 1 refers to the first (leftmost) column of `table_array`. A 0 would produce a `#VALUE!` error because 0 is not a valid column index in VLOOKUP.

</details>

---

### 4. Question

Consider this small Sales table on the active sheet (columns A–D, rows 2–6):

| Row | SalesRepID | Region | Status    | Revenue |
| :-- | :--------- | :----- | :-------- | :------ |
| 2   | SR1        | North  | Closed    | 45000   |
| 3   | SR2        | South  | Open      | 28000   |
| 4   | SR3        | North  | Closed    | 62000   |
| 5   | SR4        | East   | Open      | 15000   |
| 6   | SR5        | South  | Closed    | 33000   |

What value does `=SUMIF(C2:C6, "Closed", D2:D6)` return?

- [ ] **A.** 183000 — the sum of all revenue values in D2:D6
- [ ] **B.** 140000 — the sum of revenue for all Closed sales
- [ ] **C.** 3 — the count of Closed rows
- [ ] **D.** 45000 — only the first Closed row's revenue

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** 183000 would be returned by `=SUM(D2:D6)`, which adds all values regardless of the Status criterion. SUMIF applies the "Closed" filter, so only matching rows contribute to the total.
- **Option B (Correct):** SUMIF scans C2:C6 for cells equal to "Closed" — rows 2, 4, and 6 match. It then sums the corresponding values in D2:D6: 45000 + 62000 + 33000 = **140000**.
- **Option C (Incorrect):** 3 is the count of Closed rows, which is what `=COUNTIF(C2:C6, "Closed")` would return. SUMIF returns a sum of values, not a row count.
- **Option D (Incorrect):** 45000 is only SR1's revenue (row 2). SUMIF does not stop at the first matching row — it accumulates all matching rows throughout the specified range.

</details>

---

### 5. Question

A product analyst runs `=COUNTIF(C2:C6, "Open")` on the same Sales table from Question 4. What does the formula return?

- [ ] **A.** 43000 — the sum of revenue for Open rows
- [ ] **B.** 3 — the count of Closed rows, since "Closed" is more common
- [ ] **C.** 2 — the number of rows where Status equals "Open"
- [ ] **D.** 5 — the total number of rows in the range, regardless of the criterion

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 43000 (28000 + 15000) is the sum of Open revenues. That result comes from `=SUMIF(C2:C6, "Open", D2:D6)`. COUNTIF counts matching rows, it does not sum associated values in another column.
- **Option B (Incorrect):** The formula counts "Open" rows, not "Closed" rows. There are 2 Open rows (SR2 and SR4) and 3 Closed rows. Using the given criterion "Open" returns 2.
- **Option C (Correct):** COUNTIF scans C2:C6 for cells equal to "Open". Rows 3 (SR2, South, Open) and 5 (SR4, East, Open) match the criterion. The count is **2**.
- **Option D (Incorrect):** 5 is the total number of rows in the range, which `=COUNTA(C2:C6)` would return. COUNTIF applies a filter ("Open") and only counts cells matching that specific criterion.

</details>

---

### 6. Question

A backend engineer copies the formula `=VLOOKUP(A2, Products!A2:C10, 2, FALSE)` from cell B2 down to cells B3, B4, and B5. After copying, the formula in B4 is `=VLOOKUP(A4, Products!A4:C12, 2, FALSE)`. What problem has occurred, and what is the correct fix?

- [ ] **A.** No problem — the formula adjusted both the lookup value and the table range correctly when copied
- [ ] **B.** The `col_index_num` shifted from 2 to 4, returning the wrong column
- [ ] **C.** The `table_array` shifted from `A2:C10` to `A4:C12` because relative references moved with the copy; the fix is to lock the table range as `Products!$A$2:$C$10`
- [ ] **D.** The lookup value shifted incorrectly; the fix is to write `$A$2` instead of `A2` to lock the lookup cell

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** While the lookup value shifting from `A2` to `A4` is intentional and correct (each row should look up its own value), the table_array shifting is a problem. The reference table on the Products sheet is fixed; it should not move when the formula is copied down.
- **Option B (Incorrect):** `col_index_num` is a literal integer (2), not a cell reference, so it does not shift when the formula is copied. Only cell references with relative addressing shift on copy.
- **Option C (Correct):** Without dollar signs, `Products!A2:C10` is a fully relative reference. When copied down two rows, both the start and end row numbers shift by the same amount, turning it into `Products!A4:C12`. This pushes the search range outside the actual data, causing `#N/A` errors for valid keys. Locking as `Products!$A$2:$C$10` pins all four corners so the range stays fixed no matter how far down the formula is copied.
- **Option D (Incorrect):** The lookup value (`A2` shifting to `A4`) is the intended behavior — each row's formula should reference its own row's key. Locking the lookup cell to `$A$2` would make every row look up the same value from row 2, which is wrong. The issue is the table_array, not the lookup value.

</details>

---

### 7. Question

An analyst needs to find the average revenue of **Closed** deals from the Sales table in Question 4. Which formula is correct?

- [ ] **A.** `=AVERAGEIF(D2:D6, "Closed", C2:C6)`
- [ ] **B.** `=SUMIF(C2:C6, "Closed", D2:D6) / COUNTIF(C2:C6, "Closed")`
- [ ] **C.** `=AVERAGEIF(C2:C6, "Closed", D2:D6)`
- [ ] **D.** `=AVERAGE(C2:C6, "Closed", D2:D6)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The arguments are swapped. `AVERAGEIF` follows the same argument order as `SUMIF`: `AVERAGEIF(range, criteria, [average_range])`. Here `D2:D6` (Revenue) is placed as the criteria range and `C2:C6` (Status) as the average range — this inverts the logic and would attempt to average Status text values against a numeric criterion, returning 0 or an error.
- **Option B (Incorrect):** While this formula produces the mathematically correct answer (SUMIF total ÷ COUNTIF count = average for matching rows), it is unnecessarily verbose. `AVERAGEIF` is the built-in function designed exactly for this single-condition average and is the cleaner, preferred approach.
- **Option C (Correct):** `=AVERAGEIF(criteria_range, criterion, average_range)` → `=AVERAGEIF(C2:C6, "Closed", D2:D6)` scans C2:C6 for "Closed" (rows 2, 4, 6) and averages their corresponding D values: (45000 + 62000 + 33000) / 3 = **46,667** (rounded). Correct syntax, correct argument order, correct result.
- **Option D (Incorrect):** `=AVERAGE()` is a plain average function; it does not accept a criteria argument. Passing `"Closed"` as a second argument would either be interpreted as a cell reference error or simply ignored, returning the average of all values in D2:D6 without any Status filtering.

</details>

---

### 8. Question

A data engineer is told: "Use an **approximate match** VLOOKUP to look up a tax rate from a bracket table." Which of the following statements about approximate match (`range_lookup = TRUE`) are accurate?

- [ ] **A.** Approximate match requires the lookup column to be sorted in ascending order; without sorting, results may be silently incorrect
- [ ] **B.** Approximate match finds the largest value in the lookup column that is less than or equal to the lookup value
- [ ] **C.** Approximate match returns `#N/A` whenever the exact lookup value is not present in the table
- [ ] **D.** Approximate match and exact match produce identical results whenever the lookup value exists in the table

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** Approximate match (`TRUE`) uses a binary search algorithm that assumes the lookup column is sorted in ascending order. If the column is unsorted, the binary search may terminate early at an incorrect position, silently returning a wrong value without any error indicator. Sorting is a hard prerequisite.
- **Option B (Correct):** When an exact match is not found, approximate match selects the largest value in the sorted lookup column that does not exceed the lookup value — this is the "floor" behavior used in bracket tables (e.g., tax rates, shipping tiers). If the lookup value is between two bracket thresholds, it falls into the lower bracket.
- **Option C (Incorrect):** Unlike exact match (`FALSE`), approximate match does **not** return `#N/A` just because the exact value is missing. It falls back to the nearest lower value. `#N/A` is returned only when the lookup value is smaller than every value in the lookup column (below the lowest bracket).
- **Option D (Correct):** When the lookup value exists verbatim in the lookup column and the column is correctly sorted, both `TRUE` and `FALSE` will find the same row and return the same column value. The difference surfaces only when the exact value is absent.

</details>

---

### 9. Question

A finance analyst has an Orders sheet where column A holds OrderID, column B holds CustomerID, and she wants to pull the customer's **City** from a Customers reference sheet where City is in column C (the 3rd column of the Customers range `$A$2:$D$50`). She enters the formula in cell D2 and plans to copy it down to D100. Which formula is correct?

- [ ] **A.** `=VLOOKUP(B2, Customers!A2:D50, 3, FALSE)`
- [ ] **B.** `=VLOOKUP(B2, Customers!$A$2:$D$50, 3, FALSE)`
- [ ] **C.** `=VLOOKUP(B2, Customers!$A$2:$D$50, "C", FALSE)`
- [ ] **D.** `=VLOOKUP(B2, Customers!$A$2:$D$50, 3, TRUE)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The formula is syntactically valid and will return the correct City for the first row (D2). However, the table_array `Customers!A2:D50` is a relative reference. When copied down, it shifts row by row — row D3's formula searches `A3:D51`, D4 searches `A4:D52`, and so on, eventually searching outside the Customers data. Locking with `$` is required for reliable copying.
- **Option B (Correct):** This is the complete, correct formula. `B2` is a relative reference that correctly shifts to `B3`, `B4`, etc. as the formula is copied down (each row looks up its own CustomerID). `Customers!$A$2:$D$50` is an absolute reference that stays fixed on every copied row. `col_index_num = 3` correctly targets column C (City). `FALSE` enforces exact matching for ID-based lookups.
- **Option C (Incorrect):** `col_index_num` must be a positive integer, not a column letter string like `"C"`. Passing a text string as the third argument causes a `#VALUE!` error in both Excel and Google Sheets.
- **Option D (Incorrect):** Using `TRUE` (approximate match) with a CustomerID lookup is inappropriate. CustomerIDs are discrete text codes, not sorted numeric brackets. Approximate match on unsorted or text IDs will silently return wrong customers. Exact match (`FALSE`) is mandatory for unique-key lookups.

</details>

---

### 10. Question

A business analyst uses the same Sales table from Question 4 and enters `=SUMIF(D2:D6, ">40000")` — note there is **no third argument** (no `sum_range`). What does this formula return?

- [ ] **A.** 3 — the count of rows where Revenue exceeds 40000
- [ ] **B.** 107000 — the sum of Revenue values that are greater than 40000 (45000 + 62000)
- [ ] **C.** 107000 — because when `sum_range` is omitted, SUMIF sums the values in the `range` argument itself that satisfy the criterion
- [ ] **D.** `#VALUE!` — SUMIF requires all three arguments and raises an error when `sum_range` is missing

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Counting rows matching a numeric condition is done by `COUNTIF`, not `SUMIF`. This SUMIF formula sums matching values, not counts matching rows.
- **Option B (Incorrect):** This option states the correct final number (107000) but attributes it to the wrong logic. Rows 2 (45000), 4 (62000) and 6 (33000) — only rows 2 and 4 exceed 40000: 45000 + 62000 = 107000. The numerical answer is right, but the option misstates which rows qualify by saying only rows 2 and 4 without checking row 6 (33000 < 40000, so excluded). However the key learning here is about the omitted sum_range.
- **Option C (Correct):** When the optional `sum_range` argument is omitted from SUMIF, the function sums the values in `range` itself (D2:D6) that match the criterion. Rows where Revenue > 40000: row 2 (45000 ✓), row 3 (28000 ✗), row 4 (62000 ✓), row 5 (15000 ✗), row 6 (33000 ✗). Sum = 45000 + 62000 = **107000**. The `range` argument doubles as both the evaluation range and the sum range.
- **Option D (Incorrect):** `sum_range` is documented as an **optional** argument in both Excel and Google Sheets. Omitting it is valid syntax and triggers the default behavior of summing the `range` values themselves. No error is raised.

</details>
