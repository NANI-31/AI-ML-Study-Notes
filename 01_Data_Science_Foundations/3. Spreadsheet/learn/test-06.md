# Spreadsheet Objective Test — Set 06

---

### 51. Question

A business analyst is working with the following **Transactions** table on the active sheet (columns A–E, rows 2–8):

| Row | TransID | Channel  | Category    | Revenue |
| :-- | :------ | :------- | :---------- | :------ |
| 2   | T001    | Online   | Electronics | 12000   |
| 3   | T002    | In-Store | Clothing    | 4500    |
| 4   | T003    | Online   | Electronics | 8700    |
| 5   | T004    | In-Store | Electronics | 11000   |
| 6   | T005    | Online   | Clothing    | 3200    |
| 7   | T006    | In-Store | Clothing    | 6800    |
| 8   | T007    | Online   | Electronics | 5400    |

What does `=SUMIFS(E2:E8, C2:C8, "Online", D2:D8, "Electronics")` return?

- [ ] **A.** 37100 — the total revenue for all Electronics transactions
- [ ] **B.** 29300 — the total revenue for all Online transactions
- [ ] **C.** 26100 — the total revenue for Online Electronics transactions
- [ ] **D.** 3 — the count of Online Electronics transactions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 37100 is the total revenue for all Electronics transactions (Online + In-Store): 12000 + 8700 + 11000 + 5400 = 37100. SUMIFS requires BOTH Channel="Online" AND Category="Electronics", excluding the In-Store Electronics row (T004, 11000).
- **Option B (Incorrect):** 29300 is the total revenue for all Online transactions: 12000 + 8700 + 3200 + 5400 = 29300. SUMIFS also requires Category="Electronics", filtering out Online Clothing (T005, 3200).
- **Option C (Correct):** Rows satisfying BOTH Online AND Electronics: T001 (12000 ✓), T003 (8700 ✓), T007 (5400 ✓). Sum = 12000 + 8700 + 5400 = **26100**.
- **Option D (Incorrect):** 3 is the count of Online Electronics rows, returned by `=COUNTIFS(C2:C8, "Online", D2:D8, "Electronics")`. SUMIFS sums the Revenue values of matching rows, not the count of matching rows.

</details>

---

### 52. Question

Using the Transactions table from Question 51, what does `=AVERAGEIFS(E2:E8, C2:C8, "In-Store", D2:D8, "Clothing")` return?

- [ ] **A.** 5650 — the average revenue for In-Store Clothing transactions
- [ ] **B.** 11300 — the total revenue for In-Store Clothing transactions
- [ ] **C.** 2 — the count of In-Store Clothing transactions
- [ ] **D.** 6800 — the revenue of the most recent In-Store Clothing transaction

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** In-Store Clothing rows: T002 (4500 ✓) and T006 (6800 ✓). AVERAGEIFS computes the mean: (4500 + 6800) / 2 = **5650**.
- **Option B (Incorrect):** 11300 (4500 + 6800) is the sum of In-Store Clothing revenue, returned by `=SUMIFS(E2:E8, C2:C8, "In-Store", D2:D8, "Clothing")`. AVERAGEIFS returns an average, not a total.
- **Option C (Incorrect):** 2 is the count of In-Store Clothing rows, returned by `=COUNTIFS(C2:C8, "In-Store", D2:D8, "Clothing")`. AVERAGEIFS computes the mean of matching values, not a row count.
- **Option D (Incorrect):** 6800 is T006's individual revenue — just one of the two In-Store Clothing rows. AVERAGEIFS does not return individual row values; it returns the arithmetic mean across all qualifying rows.

</details>

---

### 53. Question

A data analyst has a Products reference table on a sheet named **Catalog** (columns: A=SKU, B=ProductName, C=Category, D=Price, E=LeadDays). She copies the formula `=VLOOKUP(A2, Catalog!$A$2:$E$50, 4, FALSE)` down the Orders sheet column F. Which column from the Catalog does this return?

- [ ] **A.** ProductName — the second column of the Catalog range
- [ ] **B.** Category — the third column of the Catalog range
- [ ] **C.** Price — the fourth column of the Catalog range
- [ ] **D.** LeadDays — the fifth column of the Catalog range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** ProductName is in the second column (B) of the Catalog range. Column index 2 would return ProductName; the formula specifies 4.
- **Option B (Incorrect):** Category is in the third column (C). Column index 3 would return Category; the formula specifies 4.
- **Option C (Correct):** The range `Catalog!$A$2:$E$50` maps as: col 1=SKU(A), col 2=ProductName(B), col 3=Category(C), col 4=Price(D), col 5=LeadDays(E). `col_index_num = 4` returns the **Price** column.
- **Option D (Incorrect):** LeadDays is in the fifth column (E). Column index 5 would return LeadDays; the formula specifies 4.

</details>

---

### 54. Question

A junior analyst is building a dashboard and wants to use `VLOOKUP` to find data to the **left** of the lookup column. For example, the lookup column (OrderDate) is in column D, but the value she wants to retrieve (OrderID) is in column A — to the left of the lookup column. She writes `=VLOOKUP(G2, $D$2:$D$50, -3, FALSE)`. What happens?

- [ ] **A.** The formula returns the value 3 columns to the left of column D, successfully retrieving OrderID
- [ ] **B.** The formula returns `#VALUE!` or `#REF!` because VLOOKUP cannot return values from columns to the left of the lookup column; negative column indexes are invalid
- [ ] **C.** The formula works but only in Google Sheets, not Excel
- [ ] **D.** The formula returns `#N/A` because the lookup range is too narrow (only one column)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This is a common misconception. VLOOKUP's `col_index_num` must be a **positive integer** (≥ 1) representing a column position within the `table_array` counting from left to right. Negative values are invalid. VLOOKUP also cannot look to the left of its search column by design — the search always occurs in the leftmost column of `table_array`.
- **Option B (Correct):** VLOOKUP does not support left-of-lookup retrieval. The `col_index_num` must be between 1 and the number of columns in `table_array`. A negative index (-3) is invalid and returns a `#VALUE!` error. To look to the left, analysts must use either: (a) restructure the table so the lookup key is leftmost; (b) use `INDEX`+`MATCH` which can return values from any column; or (c) use `XLOOKUP` in modern Excel/Google Sheets which supports left-of-lookup returns.
- **Option C (Incorrect):** Both Excel and Google Sheets enforce the same rule: `col_index_num` must be a positive integer. Neither application supports negative column indexes in VLOOKUP. The limitation is identical in both platforms.
- **Option D (Incorrect):** The formula would fail with a `#VALUE!` error due to the invalid negative column index, not `#N/A`. `#N/A` indicates a failed lookup (key not found), which would be a separate issue. The formula breaks at argument validation before it even searches for the lookup value.

</details>

---

### 55. Question

A data analyst compares these two formulas on the Transactions table from Question 51:

- **Formula A:** `=COUNTIF(C2:C8, "Online")`
- **Formula B:** `=COUNTIFS(C2:C8, "Online")`

Which statement accurately describes the relationship between these two formulas?

- [ ] **A.** They return different results because COUNTIFS applies additional hidden default filters
- [ ] **B.** Formula A is valid; Formula B produces a `#VALUE!` error because COUNTIFS requires at least two condition pairs
- [ ] **C.** Both formulas return the same result (4), and COUNTIFS with a single condition pair is valid — COUNTIFS is a superset of COUNTIF for cases where one or more conditions are needed
- [ ] **D.** Formula B always returns 1 regardless of data because COUNTIFS with one condition defaults to counting unique values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** COUNTIFS applies no hidden filters. With a single condition pair, COUNTIFS applies exactly one condition — identical to COUNTIF with the same criteria. Both return the count of "Online" cells in C2:C8: T001, T003, T005, T007 → **4**.
- **Option B (Incorrect):** COUNTIFS does not require two or more condition pairs. A minimum of one `(criteria_range, criterion)` pair is sufficient. The function syntax is `COUNTIFS(criteria_range1, criterion1, [criteria_range2, criterion2], ...)` — the additional pairs are optional. Formula B is valid.
- **Option C (Correct):** Both formulas return **4** (the four Online transactions). COUNTIFS with one condition behaves identically to COUNTIF with the same condition. COUNTIFS is designed to extend COUNTIF by supporting multiple AND conditions; when only one condition is provided, the result is indistinguishable from COUNTIF's result.
- **Option D (Incorrect):** COUNTIFS does not count unique values. Like COUNTIF, it counts the number of cells satisfying the specified criteria — counting all matching cells, including duplicates. There is no default "count unique" behavior in either function.

</details>

---

### 56. Question

A data analyst sees the formula `=VLOOKUP(B2, $A$1:$D$100, 2, FALSE)` in column E of the same sheet where the lookup table also lives. Row 2 of column A is a header row ("CustomerID"), not data. What is most likely wrong with this formula?

- [ ] **A.** The table_array must reference a different sheet; VLOOKUP cannot use data from the same sheet
- [ ] **B.** The table_array starts at row 1 (header row), so VLOOKUP will search the header values as potential matches, which may cause unexpected results if the header text matches a lookup value or pushes column numbering off
- [ ] **C.** Column index 2 is too low; it should be at least 3 to return meaningful data
- [ ] **D.** The formula will always return the header value "CustomerID" regardless of what B2 contains

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** VLOOKUP can search data on the same sheet as the formula. Cross-sheet references are not required. Including the table range `$A$1:$D$100` on the active sheet is perfectly valid.
- **Option B (Correct):** Including the header row (row 1) in `table_array` is a common mistake. VLOOKUP searches the leftmost column of the range, which now includes `A1` ("CustomerID") as a potential match. If the lookup value were the literal text "CustomerID", it would match the header and return whatever is in column 2 of row 1 ("Name" or another header). More subtly, including headers may also shift column index counting if the analyst forgot that col 1 is now the header row. Best practice: start the range at the first data row (`$A$2:$D$100`).
- **Option C (Incorrect):** Column index 2 is completely valid. It returns the second column of the range, regardless of what that column contains. A "too low" column index is not the primary concern here — the header inclusion in the range is.
- **Option D (Incorrect):** The formula does not always return "CustomerID". It returns "CustomerID" only if `B2` contains the exact text "CustomerID" AND the formula finds a match in `A1`. For all other values in B2, it searches the data rows below the header and returns the matching second-column value.

</details>

---

### 57. Question

A product analyst is reviewing four statements about pivot table **Values** area configuration. Which of the following are accurate?

- [ ] **A.** A field placed in the Values area can only be summarized using Sum — other aggregations (Average, Count, Max) are not available in standard pivot tables
- [ ] **B.** Placing a text field (like "Status") in the Values area results in a Count of Status values, since text cannot be numerically summed
- [ ] **C.** The same field can be added to the Values area multiple times to show, for example, both "Sum of Amount" and "Average of Amount" side by side
- [ ] **D.** A numeric field placed in the Values area defaults to Sum in Excel pivot tables

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B, C, D**

#### Explanations:
- **Option A (Incorrect):** Pivot table Values area supports multiple aggregation functions: Sum, Count, Average, Max, Min, StdDev, Var, and more. Sum is the default but is not the only option. Analysts can change the summarization method through Value Field Settings.
- **Option B (Correct):** When a text (non-numeric) field is placed in the Values area, Excel/Google Sheets cannot sum it mathematically. The application defaults to **Count**, which counts the number of non-empty cells for that field in each pivot group. This is useful for counting orders per region, for example.
- **Option C (Correct):** The same field can be added to the Values area multiple times. Each instance can be configured with a different aggregation function (one as Sum, another as Average, another as Count). This creates a multi-column comparison within the Values area, showing different perspectives on the same data side by side.
- **Option D (Correct):** In Excel pivot tables, when a numeric field is dragged to the Values area, the default summarization is **Sum**. Google Sheets also defaults to Sum for numeric fields. The analyst can change this default in the field's settings after placement.

</details>

---

### 58. Question

A data analyst has this lookup formula: `=VLOOKUP(C2, 'Product Catalog'!$B$2:$F$30, 3, FALSE)`. The sheet name "Product Catalog" contains a space. Which of the following statements about this formula's syntax are true?

- [ ] **A.** The single quotes around 'Product Catalog' are required because the sheet name contains a space — without them, the formula is a syntax error
- [ ] **B.** Double quotes should be used instead: `"Product Catalog"!$B$2:$F$30`
- [ ] **C.** Sheet names with spaces require single quotes in Excel and Google Sheets cross-sheet references
- [ ] **D.** The formula would work identically if the sheet were renamed to "ProductCatalog" (no space) and the single quotes removed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C, D**

#### Explanations:
- **Option A (Correct):** Single quotes around a sheet name in a cross-sheet reference are **mandatory** when the sheet name contains spaces, special characters, or starts with a digit. Without them, the space in "Product Catalog" causes a parse error because the application cannot determine where the sheet name ends and the range begins.
- **Option B (Incorrect):** Double quotes are NOT used to enclose sheet names in cross-sheet references. Double quotes create text strings in formulas (like `"Completed"` in SUMIF criteria). Using double quotes around the sheet name would make the reference a text string, not a range reference, causing a `#VALUE!` error.
- **Option C (Correct):** Both Excel and Google Sheets require single quotes to wrap sheet names containing spaces or special characters in cross-sheet references. This is standard spreadsheet syntax, consistent across both platforms.
- **Option D (Correct):** If the sheet is renamed to "ProductCatalog" (no space), single quotes are no longer needed. The reference becomes `ProductCatalog!$B$2:$F$30`, which is valid without quotes. Single quotes are a workaround for names that would otherwise cause parsing ambiguity.

</details>

---

### 59. Question

A finance analyst enters `=AVERAGEIF(D2:D8, "<5000", E2:E8)` on the Transactions table from Question 51 (col D = Revenue, col E would be another column). Actually re-using the table: using the Transactions table where col D = Revenue (12000, 4500, 8700, 11000, 3200, 6800, 5400), what does `=AVERAGEIF(E2:E8, "<5000", E2:E8)` return (where E2:E8 = Revenue column)?

- [ ] **A.** 3850 — the average of revenue values less than 5000
- [ ] **B.** 7657 — the average of all seven revenue values
- [ ] **C.** 7700 — the sum of values less than 5000
- [ ] **D.** 2 — the count of transactions with revenue less than 5000

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** AVERAGEIF evaluates the criteria range (E2:E8 = Revenue values: 12000, 4500, 8700, 11000, 3200, 6800, 5400) against the criterion `"<5000"`. Qualifying rows: 4500 (T002) and 3200 (T005). Average = (4500 + 3200) / 2 = **3850**.
- **Option B (Incorrect):** 7657 ≈ (12000+4500+8700+11000+3200+6800+5400)/7 ≈ 51600/7 ≈ 7371 is the average of all seven transactions. AVERAGEIF filters to only values less than 5000 before averaging.
- **Option C (Incorrect):** 7700 (4500 + 3200) would be the sum of values less than 5000, returned by `=SUMIF(E2:E8, "<5000")`. AVERAGEIF divides the sum by the count of qualifying rows.
- **Option D (Incorrect):** 2 is the count of transactions with revenue less than 5000, returned by `=COUNTIF(E2:E8, "<5000")`. AVERAGEIF returns the mean value, not a count.

</details>

---

### 60. Question

A data analyst has the Orders table from Question 31. She writes three separate formulas:

- **M:** `=SUMIF(D2:D11, "Completed", F2:F11)` → returns 245000
- **N:** `=COUNTIF(D2:D11, "Completed")` → returns 5
- **P:** `=AVERAGEIF(D2:D11, "Completed", F2:F11)` → returns 49000

She wants to verify: does M / N equal P?

- [ ] **A.** No — AVERAGEIF uses a different denominator than COUNTIF provides
- [ ] **B.** Yes — AVERAGEIF is mathematically equivalent to SUMIF divided by COUNTIF when both use the same criteria range and criterion
- [ ] **C.** No — SUMIF totals are always larger than AVERAGEIF results, making this comparison invalid
- [ ] **D.** Yes — but only when there are no blank cells in the sum range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** AVERAGEIF uses exactly the same denominator as COUNTIF in an equivalent comparison — the count of rows where the criteria match. The formula `=SUMIF(range, criterion, sum_range) / COUNTIF(range, criterion)` produces the same result as `=AVERAGEIF(range, criterion, sum_range)` for matching criteria range and criterion.
- **Option B (Correct):** AVERAGEIF computes: (sum of matching sum_range values) / (count of matching criteria_range cells). This is precisely SUMIF / COUNTIF: 245000 / 5 = **49000**, which matches the AVERAGEIF result. The three formulas are mathematically consistent, confirming the analyst's data. AVERAGEIF is syntactic sugar for this division.
- **Option C (Incorrect):** SUMIF totals are larger than AVERAGEIF results (since the average divides by the count), but this does not make the M/N = P verification invalid. The comparison M/N = P is exactly how AVERAGEIF is defined and is always valid.
- **Option D (Incorrect):** Blank cells in the sum range are excluded from both the sum (SUMIF ignores blanks) and the count (AVERAGEIF ignores blank average_range cells when computing the mean). As long as both SUMIF and AVERAGEIF handle blanks consistently (which they do), M/N = P holds whether or not blanks are present.

</details>
