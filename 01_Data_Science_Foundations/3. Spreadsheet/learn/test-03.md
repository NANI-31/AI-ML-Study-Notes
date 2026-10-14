# Spreadsheet Objective Test — Set 03

---

### 21. Question

A data analyst is working with the following **Inventory** table (rows 2–6, columns A–D on the active sheet):

| Row | SKU   | Category    | Warehouse | Stock |
| :-- | :---- | :---------- | :-------- | :---- |
| 2   | W101  | Widgets     | East      | 200   |
| 3   | W102  | Gadgets     | West      | 150   |
| 4   | W103  | Widgets     | North     | 75    |
| 5   | W104  | Accessories | East      | 300   |
| 6   | W105  | Gadgets     | East      | 90    |

What does `=SUMIF(B2:B6, "Widgets", D2:D6)` return?

- [ ] **A.** 815 — the sum of all stock values across all rows
- [ ] **B.** 275 — the sum of stock for all Widgets rows
- [ ] **C.** 2 — the count of Widgets rows
- [ ] **D.** 137.5 — the average stock for Widgets rows

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** 815 is the total of all stock values (200+150+75+300+90), returned by `=SUM(D2:D6)`. SUMIF with the "Widgets" criterion filters to matching rows only.
- **Option B (Correct):** SUMIF scans B2:B6 (Category) for "Widgets". Rows 2 (W101, 200) and 4 (W103, 75) match. Corresponding D values: 200 + 75 = **275**.
- **Option C (Incorrect):** 2 is the count of Widgets rows, returned by `=COUNTIF(B2:B6, "Widgets")`. SUMIF sums values, it does not count rows.
- **Option D (Incorrect):** 137.5 is the average of 200 and 75, returned by `=AVERAGEIF(B2:B6, "Widgets", D2:D6)`. SUMIF totals values, it does not average them.

</details>

---

### 22. Question

Using the same Inventory table from Question 21, what does `=COUNTIF(C2:C6, "East")` return?

- [ ] **A.** 590 — the sum of stock for East-warehouse items
- [ ] **B.** 3 — the number of items stored in the East warehouse
- [ ] **C.** 196.7 — the average stock in the East warehouse
- [ ] **D.** 300 — the maximum stock for an East-warehouse item

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** 590 (200 + 300 + 90) is the sum of stock for East items, returned by `=SUMIF(C2:C6, "East", D2:D6)`. COUNTIF counts rows, not sums values.
- **Option B (Correct):** COUNTIF scans C2:C6 (Warehouse) for cells equal to "East". Rows 2 (East), 5 (East), and 6 (East) match. Count = **3**.
- **Option C (Incorrect):** 196.7 ≈ 590/3 is the average East stock, returned by `=AVERAGEIF(C2:C6, "East", D2:D6)`. COUNTIF returns a count of matching rows, not an average.
- **Option D (Incorrect):** 300 is the maximum stock value in the East warehouse, returned by `=MAXIFS(D2:D6, C2:C6, "East")`. COUNTIF counts matching rows — it has nothing to do with maximum values.

</details>

---

### 23. Question

An analyst copies a formula from G2 down to G6 on an Orders sheet. The original formula in G2 is `=VLOOKUP(B2, $A$2:$D$10, 3, FALSE)`. What is the formula in G5 after copying?

- [ ] **A.** `=VLOOKUP(B5, $A$2:$D$10, 3, FALSE)`
- [ ] **B.** `=VLOOKUP(B5, $A$5:$D$13, 3, FALSE)`
- [ ] **C.** `=VLOOKUP(B2, $A$2:$D$10, 3, FALSE)` — unchanged because all references are absolute
- [ ] **D.** `=VLOOKUP(B5, $A$2:$D$10, 6, FALSE)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** When copying down 3 rows (from G2 to G5), only **relative** cell references shift. `B2` is a relative reference — it shifts by 3 rows to become `B5`, correctly pointing to row 5's lookup value. `$A$2:$D$10` uses absolute row and column locking (`$`), so it stays fixed as `$A$2:$D$10`. The `col_index_num` (3) is a literal number and never shifts. Result: `=VLOOKUP(B5, $A$2:$D$10, 3, FALSE)`.
- **Option B (Incorrect):** The table_array `$A$5:$D$13` would result only if the range had no `$` signs (fully relative). With `$A$2:$D$10`, the dollar signs lock all four corners of the range against shifting.
- **Option C (Incorrect):** The lookup value reference `B2` is **not** locked (no `$`), so it shifts when copied. Only the table_array is locked. If all references were absolute (`$B$2, $A$2:$D$10`), the formula would be unchanged — but in this case, `B2` is intentionally relative so each row looks up its own value.
- **Option D (Incorrect):** `col_index_num` is a hard-coded integer (3), not a cell reference. It never shifts when a formula is copied in any direction. It remains 3 in G5.

</details>

---

### 24. Question

A junior analyst is building a cross-sheet VLOOKUP. The lookup value is stored in cell C5 of the active sheet. The reference table is on a sheet named **Catalog** in the range A1:F30. She wants to retrieve the value from the 4th column with an exact match. Which formula is correctly written?

- [ ] **A.** `=VLOOKUP(C5, Catalog$A$1:$F$30, 4, FALSE)`
- [ ] **B.** `=VLOOKUP(C5, Catalog!$A$1:$F$30, 4, FALSE)`
- [ ] **C.** `=VLOOKUP(C5, "Catalog"!$A$1:$F$30, 4, FALSE)`
- [ ] **D.** `=VLOOKUP(C5, Catalog!$A$1:$F$30, 4)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The cross-sheet reference syntax requires an exclamation mark (`!`) between the sheet name and the range: `Catalog!$A$1:$F$30`. Option A is missing the `!`, writing `Catalog$A$1:$F$30` which is invalid syntax and produces an error.
- **Option B (Correct):** The cross-sheet reference is correctly formed as `SheetName!$A$1:$F$30`. The `!` separates the sheet name from the range address. The `$` signs lock the range absolutely. The other arguments are correct: `C5` as lookup value, `4` as column index, and `FALSE` for exact match.
- **Option C (Incorrect):** The sheet name should not be enclosed in double quotes in a cross-sheet range reference. Quoting it with `"Catalog"` makes it a text string rather than a sheet reference, causing a `#VALUE!` or `#REF!` error. Sheet names with spaces are enclosed in **single** quotes (e.g., `'My Catalog'!$A$1:$F$30`), not double quotes.
- **Option D (Incorrect):** While omitting `range_lookup` is syntactically allowed, the default behavior in Excel is `TRUE` (approximate match), which is unsafe for ID-based lookups. For a robust exact-match lookup, `FALSE` should always be explicitly specified. This option would potentially return wrong results for non-sorted tables.

</details>

---

### 25. Question

A senior analyst's dashboard uses the formula `=VLOOKUP(A2, 'Sales Data'!$B$2:$E$50, 3, FALSE)`. A teammate renames the sheet "Sales Data" to "Q1 Sales". What happens to the formula?

- [ ] **A.** The formula breaks immediately with a `#REF!` error and cannot be recovered
- [ ] **B.** The formula automatically updates to reference the renamed sheet: `'Q1 Sales'!$B$2:$E$50`
- [ ] **C.** The formula continues to search the old "Sales Data" sheet, which no longer exists
- [ ] **D.** The formula returns 0 because the sheet name changed and the reference is invalidated silently

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Renaming a sheet does not permanently break formula references. Both Excel and Google Sheets track sheet identity internally and automatically update all formulas referencing that sheet to use the new sheet name. The formula remains functional.
- **Option B (Correct):** When a sheet is renamed in Excel or Google Sheets, the application automatically propagates the new name into every formula that references that sheet. `'Sales Data'!$B$2:$E$50` becomes `'Q1 Sales'!$B$2:$E$50` without any manual intervention. This is a key advantage of formula-based references over hardcoded external connections.
- **Option C (Incorrect):** Once a sheet is renamed, the old name no longer exists. Formulas do not continue pointing at the old name — the application updates them automatically. If the sheet were deleted (rather than renamed), a `#REF!` error would appear.
- **Option D (Incorrect):** The formula does not silently return 0 after a rename. The application's formula recalculation engine updates the sheet reference. No value change occurs — the formula continues to return the same lookup result from the same data, now under the new sheet name.

</details>

---

### 26. Question

A data analyst needs to count the number of products in the Inventory table from Question 21 that have **more than 100 units in stock**. Which formula is correct?

- [ ] **A.** `=SUMIF(D2:D6, ">100", D2:D6)`
- [ ] **B.** `=COUNTIF(D2:D6, ">100")`
- [ ] **C.** `=COUNTIF(D2:D6, "100")`
- [ ] **D.** `=COUNT(D2:D6, ">100")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `SUMIF` with the same column as both `range` and `sum_range` would **sum** the stock values greater than 100 (200 + 150 + 300 = 650), not count the rows. The analyst needs a count, not a sum.
- **Option B (Correct):** `=COUNTIF(D2:D6, ">100")` counts the number of cells in D2:D6 where the value is greater than 100. Stock values: 200 ✓, 150 ✓, 75 ✗, 300 ✓, 90 ✗ → count = **3**.
- **Option C (Incorrect):** `=COUNTIF(D2:D6, "100")` counts cells that are **exactly equal** to 100 (no comparison operator). None of the stock values are exactly 100, so this returns 0 — not the intended result of "more than 100".
- **Option D (Incorrect):** The standard `COUNT` function counts numeric cells but does not accept a criterion as a second argument. Passing `">100"` as a second argument is not valid syntax for `COUNT` — it would treat it as an additional range reference, not a filter criterion.

</details>

---

### 27. Question

An analyst is comparing the outputs of `SUMIF` and `AVERAGEIF` on the Inventory table from Question 21. She runs:
- Formula X: `=SUMIF(B2:B6, "Gadgets", D2:D6)`
- Formula Y: `=AVERAGEIF(B2:B6, "Gadgets", D2:D6)`

What are the respective results?

- [ ] **A.** X = 240; Y = 120
- [ ] **B.** X = 240; Y = 2
- [ ] **C.** X = 2; Y = 120
- [ ] **D.** X = 150; Y = 90

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Gadgets rows: W102 (Stock=150) and W105 (Stock=90). SUMIF (X) sums them: 150 + 90 = **240**. AVERAGEIF (Y) averages them: (150 + 90) / 2 = **120**.
- **Option B (Incorrect):** Y = 2 would be the count of Gadgets rows (COUNTIF result). AVERAGEIF returns the arithmetic mean of matching values, not the count of matching rows.
- **Option C (Incorrect):** X = 2 would be COUNTIF's result. SUMIF sums numeric values, it does not count rows. The sum of 150 + 90 = 240.
- **Option D (Incorrect):** 150 and 90 are the individual stock values for the two Gadgets rows. The formulas aggregate these values (sum = 240; average = 120) rather than returning individual row values.

</details>

---

### 28. Question

A DevOps engineer asks: "What is the difference between a **relative** reference (`A2:D6`) and an **absolute** reference (`$A$2:$D$6`) in a spreadsheet formula?"

Which of the following statements are accurate?

- [ ] **A.** A relative reference shifts its row and column numbers when the formula is copied to a different cell; an absolute reference stays fixed regardless of where the formula is copied
- [ ] **B.** A relative reference is more accurate than an absolute reference for all formulas
- [ ] **C.** An absolute reference (`$A$2:$D$6`) must be used even when the formula will never be copied
- [ ] **D.** A mixed reference like `$A2` locks only the column (A stays A when copied across columns) but allows the row number to shift when copied down

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, D**

#### Explanations:
- **Option A (Correct):** This is the core definition. When a formula with a relative reference is copied to another cell, the row/column addresses shift by the same offset as the copy displacement. Absolute references (`$` on both row and column) remain fixed regardless of copy direction. This is the fundamental reason table_arrays in VLOOKUP should use `$` — the lookup table should not move as the formula is copied down.
- **Option B (Incorrect):** Neither reference type is inherently "more accurate." They serve different purposes. Relative references are correct for lookup values that should point to the current row. Absolute references are correct for fixed ranges (like reference tables) that should not shift. Choosing the wrong type introduces bugs, not the type itself being inferior.
- **Option C (Incorrect):** Absolute references are a practical necessity when formulas will be **copied** and the range must remain fixed. If a formula is entered once and never copied, using `$` is optional — the formula works identically with or without `$` when in a single cell. Mandatory usage of `$` for never-copied formulas is not required.
- **Option D (Correct):** A mixed reference uses `$` on either the column letter or the row number but not both. `$A2` locks the column A (prevents it from shifting left/right when copied across columns) but leaves the row number 2 free to shift when copied up/down. Similarly, `A$2` locks the row but not the column. Mixed references are useful for multiplication tables and two-dimensional lookup scenarios.

</details>

---

### 29. Question

A finance analyst enters `=VLOOKUP("INV-999", $A$2:$D$50, 2, FALSE)` where "INV-999" does not exist in column A of the range. What does the formula return?

- [ ] **A.** 0 — the default numeric value when no match is found
- [ ] **B.** An empty cell value (blank) indicating the match was not found
- [ ] **C.** `#N/A` — the standard error for a missing lookup key with exact match
- [ ] **D.** The value from the row closest to "INV-999" alphabetically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** VLOOKUP does not return 0 when a lookup value is missing. 0 is a valid data value and would cause confusion. The function uses a distinct error code (`#N/A`) to signal a failed lookup.
- **Option B (Incorrect):** VLOOKUP does not return an empty/blank cell on failure. A blank would also be ambiguous (is the field empty, or was the lookup unsuccessful?). The `#N/A` error is the unambiguous signal.
- **Option C (Correct):** When VLOOKUP with `range_lookup = FALSE` (exact match) cannot find the lookup value in the leftmost column, it returns the `#N/A` error. This signals "Not Available" — the value does not exist in the table. Analysts often wrap this with `IFERROR(VLOOKUP(...), "Not Found")` to display a friendlier message.
- **Option D (Incorrect):** Returning the closest alphabetical match is the behavior of **approximate match** (`range_lookup = TRUE`) on a sorted column. With exact match (`FALSE`), there is no "closest" fallback — either the value is found exactly or `#N/A` is returned.

</details>

---

### 30. Question

A product manager reviews a dashboard where a pivot table is on a sheet named **Summary** and a formula on the main sheet reads `=Summary!C5`. The pivot table currently shows "North" revenue in cell C5. A colleague then adds a new row to the pivot table's data source, causing the pivot table to insert an additional row above row 5, pushing the "North" revenue down to C6. What does `=Summary!C5` now display?

- [ ] **A.** The "North" revenue value — because pivot table cross-sheet references track labeled data, not cell addresses
- [ ] **B.** Whatever value is now in cell C5 of the Summary sheet after the pivot table re-layout, which may no longer be "North" revenue
- [ ] **C.** `#REF!` — because the pivot table invalidated the reference by moving data
- [ ] **D.** An empty string — the reference remains but returns blank until manually refreshed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This is the common dangerous misconception. Cross-sheet cell references like `=Summary!C5` point to a **fixed cell address** (row 5, column C), not to a named data label like "North". They do not track which pivot table field or label occupies a position. If the layout shifts, the reference silently points to whatever value now occupies that cell.
- **Option B (Correct):** A cross-sheet reference is a hard-coded address — `C5` means the cell at column C, row 5 of the Summary sheet, nothing more. When the pivot table inserts a new row and pushes "North" revenue to C6, cell C5 now contains whatever occupies that position (possibly a subtotal, grand total, or a different region's label). The formula returns that new value without error or warning, silently providing wrong data. This is a known fragility of linking formulas directly to pivot table cells.
- **Option C (Incorrect):** `#REF!` occurs when a referenced cell or range is deleted or when a formula's range becomes invalid (e.g., by deleting a row the reference covered). Inserting rows in the pivot table does not delete C5 — it just changes what value C5 holds. The reference itself remains valid.
- **Option D (Incorrect):** The reference does not return blank. C5 still contains a cell value after the pivot re-layout — it's just the wrong value now. Cross-sheet references return the current contents of the referenced cell, whatever those may be.

</details>
