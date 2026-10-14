# Spreadsheet Objective Test — Set 02

---

### 11. Question

An analyst is working with the following **Employees** reference table on a sheet named **HR**:

| Row | EmpID | Name          | Department  | Salary |
| :-- | :---- | :------------ | :---------- | :----- |
| 2   | E01   | Priya Sharma  | Engineering | 95000  |
| 3   | E02   | Arjun Mehta   | Marketing   | 72000  |
| 4   | E03   | Sara Khan     | Engineering | 88000  |
| 5   | E04   | Dev Patel     | HR          | 64000  |
| 6   | E05   | Neha Gupta    | Marketing   | 78000  |

The formula `=VLOOKUP("E03", HR!$A$2:$D$6, 4, FALSE)` is entered on an Orders sheet. What value is returned?

- [ ] **A.** "Engineering" — the department of E03, found in the third column of the range
- [ ] **B.** "Sara Khan" — the name of E03, found in the second column of the range
- [ ] **C.** 88000 — the salary of E03, found in the fourth column of the range
- [ ] **D.** `#N/A` — because "E03" uses a zero-padded number and cannot be found exactly

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** "Engineering" is in the **third** column of the range (col 1=EmpID, col 2=Name, col 3=Department, col 4=Salary). The formula requests column index 4, which is Salary, not Department.
- **Option B (Incorrect):** "Sara Khan" is in the second column (Name). Column index 2 would return it; the formula specifies index 4.
- **Option C (Correct):** The range `HR!$A$2:$D$6` has four columns: col 1=EmpID, col 2=Name, col 3=Department, col 4=Salary. VLOOKUP finds "E03" in the leftmost column (row 4 of the table) and returns the value from the 4th column of that row: **88000**.
- **Option D (Incorrect):** The `#N/A` error arises only when the lookup value is not found in the leftmost column with an exact match. "E03" is present in A4 of the HR table, so the lookup succeeds. The format "E03" (text with leading zeros) matches as long as the stored cell value is also text "E03" — no mismatch is implied here.

</details>

---

### 12. Question

An analyst enters `=COUNTIF(C2:C6, "Engineering")` on the same HR sheet from Question 11 (column C = Department). What does the formula return?

- [ ] **A.** 183000 — the sum of Engineering salaries
- [ ] **B.** 2 — the number of employees in the Engineering department
- [ ] **C.** 91500 — the average Engineering salary
- [ ] **D.** 5 — the total number of employees in the table

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** 183000 (95000 + 88000) is the sum of Engineering salaries, which `=SUMIF(C2:C6, "Engineering", D2:D6)` would return. COUNTIF counts matching rows, it does not aggregate numeric values.
- **Option B (Correct):** COUNTIF scans C2:C6 (Department) for cells equal to "Engineering". Row 2 (Engineering) and row 4 (Engineering) match. The count is **2**.
- **Option C (Incorrect):** 91500 is the average Engineering salary: (95000 + 88000) / 2. That result requires `=AVERAGEIF(C2:C6, "Engineering", D2:D6)`. COUNTIF only counts, not averages.
- **Option D (Incorrect):** 5 is the total count of all rows in the range, returned by `=COUNTA(C2:C6)`. COUNTIF with criterion "Engineering" filters to only matching rows.

</details>

---

### 13. Question

Using the HR table from Question 11, what does `=AVERAGEIF(C2:C6, "Marketing", D2:D6)` return?

- [ ] **A.** 150000 — the sum of Marketing salaries
- [ ] **B.** 78000 — only the highest Marketing salary
- [ ] **C.** 75000 — the average salary across the two Marketing employees
- [ ] **D.** 2 — the number of Marketing employees

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 150000 (72000 + 78000) is the total of Marketing salaries, returned by `=SUMIF(C2:C6, "Marketing", D2:D6)`. AVERAGEIF computes the mean, not the total.
- **Option B (Incorrect):** 78000 is Neha Gupta's salary — the higher of the two Marketing salaries. AVERAGEIF returns the mean of all matching values, not the maximum.
- **Option C (Correct):** AVERAGEIF scans C2:C6 for "Marketing" — rows 3 (E02, 72000) and 6 (E05, 78000) match. It then averages the corresponding D values: (72000 + 78000) / 2 = **75000**.
- **Option D (Incorrect):** 2 is the count of Marketing employees, which `=COUNTIF(C2:C6, "Marketing")` returns. AVERAGEIF produces a numeric average of the matched values, not a row count.

</details>

---

### 14. Question

A data analyst writes the formula `=VLOOKUP(F2, $A$2:$D$6, 2, FALSE)` in cell G2 on the **same** sheet as the HR data from Question 11, where F2 contains "E04". What does the formula return?

- [ ] **A.** `#REF!` — because the formula references data on the same sheet as itself
- [ ] **B.** "HR" — the department of E04, found in the third column of the range
- [ ] **C.** "Dev Patel" — the name of E04, found in the second column of the range
- [ ] **D.** 64000 — the salary of E04, found in the fourth column of the range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** VLOOKUP can reference data on the same sheet as the formula — this is perfectly valid and common. `#REF!` occurs when a column index exceeds the range width or a referenced range is deleted, not from same-sheet references.
- **Option B (Incorrect):** "HR" is in the **third** column of the range (Department). The formula specifies `col_index_num = 2`, which is the Name column, not Department.
- **Option C (Correct):** `$A$2:$D$6` has col 1=EmpID, col 2=Name, col 3=Department, col 4=Salary. VLOOKUP finds "E04" in column A (row 5 of data) and returns the value from column 2 (Name): **"Dev Patel"**.
- **Option D (Incorrect):** 64000 (Salary) sits in the fourth column. Column index 4 would retrieve it; the formula specifies 2.

</details>

---

### 15. Question

A product manager examines these four formula options, each intended to return the total **Revenue** for rows where Region is "North" from the Sales table in Question 4 (col A = SalesRepID, col B = Region, col C = Status, col D = Revenue, rows 2–6). Which formula is correct?

- [ ] **A.** `=SUMIF(D2:D6, "North", B2:B6)`
- [ ] **B.** `=COUNTIF(B2:B6, "North")`
- [ ] **C.** `=SUMIF(B2:B6, "North", D2:D6)`
- [ ] **D.** `=SUMIF("North", B2:B6, D2:D6)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The arguments are in the wrong order. `D2:D6` (Revenue) is placed as the criteria range and `B2:B6` (Region) as the sum range — meaning the formula searches Revenue values for the text "North" (finding none) and attempts to sum Region text values, producing 0 or an error.
- **Option B (Incorrect):** COUNTIF returns the **count** of North rows, not the sum of their revenue. The result would be 2 (rows 2 and 4 in the Sales table), not a revenue total.
- **Option C (Correct):** `SUMIF(range, criteria, sum_range)` → `SUMIF(B2:B6, "North", D2:D6)` correctly sets the criteria range as Region (B), the criterion as "North", and the sum range as Revenue (D). North rows: SR1 (45000) and SR3 (62000) → total = **107000**.
- **Option D (Incorrect):** The SUMIF argument order is `(range, criteria, [sum_range])` — the criteria string must be the second argument, not the first. Placing `"North"` as the first argument is invalid syntax and produces a `#VALUE!` error.

</details>

---

### 16. Question

A senior analyst's formula `=VLOOKUP(A3, Products!B2:E10, 1, FALSE)` is intended to find product details. A colleague flags it as structurally broken. What is the fundamental problem?

- [ ] **A.** The `col_index_num` of 1 would always return the lookup value itself, which is usually redundant but not an error
- [ ] **B.** The lookup value `A3` contains a ProductID, but the table_array starts from column B, meaning ProductID is not in the leftmost column of the range — so VLOOKUP searches the wrong column
- [ ] **C.** VLOOKUP cannot look up values that are stored in column A of the worksheet
- [ ] **D.** The `range_lookup` argument is missing, so the formula always uses approximate match by default

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** While `col_index_num = 1` does return the first column of the range (which would be B, ProductName if the range starts at B), the bigger structural issue is not which column is returned — it is what column is being searched. The search always occurs in the leftmost column of the table_array, which here is B, not the ProductID column.
- **Option B (Correct):** VLOOKUP's leftmost column rule states that the search value is always matched against the **first** column of `table_array`. By starting the range at column B (`Products!B2:E10`), the search column becomes B (ProductName or whatever lives in B), not the ProductID column in A. If A3 holds a ProductID and the ProductIDs live in column A of the Products sheet, they are outside the table_array, so the lookup fails or returns incorrect matches. The range must include column A: `Products!$A$2:$E$10`.
- **Option C (Incorrect):** VLOOKUP has no restriction on which worksheet column the lookup cell is in. `A3` on the source sheet is a perfectly valid lookup value reference. The issue is in the table_array definition, not the lookup value location.
- **Option D (Incorrect):** When `range_lookup` is omitted in Excel, it defaults to `TRUE` (approximate match). While this is a separate best-practice concern (and should be explicitly set to FALSE for ID lookups), it is not the "fundamental" structural breakage identified by the colleague. The bigger problem is the wrong starting column of the table_array.

</details>

---

### 17. Question

A data analyst wants to count the number of orders where **Status is "Pending"**. The Orders sheet has Status in column B (B2:B20). Four colleagues propose these formulas. Which one is correct?

- [ ] **A.** `=SUMIF(B2:B20, "Pending")`
- [ ] **B.** `=COUNT(B2:B20, "Pending")`
- [ ] **C.** `=COUNTIF(B2:B20, "Pending")`
- [ ] **D.** `=COUNTA(B2:B20)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `SUMIF(B2:B20, "Pending")` with no third argument would attempt to sum the values in B2:B20 that equal "Pending". Since Status is text, summing text values yields 0 — not a count of matching rows. Additionally, this is a sum function, not a count function.
- **Option B (Incorrect):** `COUNT(range, criteria)` is not the correct syntax. The standard `COUNT` function counts only numeric values in a range and does not accept a criteria argument. Passing `"Pending"` as a second argument would count an additional empty cell reference (treating "Pending" as a value), not filter by status.
- **Option C (Correct):** `=COUNTIF(range, criteria)` is exactly the right function. It counts cells in `B2:B20` that match the text "Pending". The result is the number of Pending orders — the intended output.
- **Option D (Incorrect):** `=COUNTA(B2:B20)` counts all non-empty cells in the Status column, returning the total number of orders regardless of their status. It applies no filter and would include Completed, Cancelled, and all other statuses.

</details>

---

### 18. Question

A junior analyst is confused about what happens when the leftmost column of a VLOOKUP table_array contains **duplicate** lookup values. For example, if two rows both have CustomerID "C2" and the formula uses exact match (`FALSE`), which row does VLOOKUP return?

- [ ] **A.** It returns values from both matching rows, stacking results vertically
- [ ] **B.** It returns the value from the **last** row where the lookup value matches
- [ ] **C.** It returns the value from the **first** row where the lookup value matches, scanning top-to-bottom
- [ ] **D.** It returns `#N/A` to indicate ambiguity when duplicate keys exist

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** VLOOKUP returns a single value from a single matched row — it does not return multiple results or stack rows. Functions like `FILTER` (in modern Excel/Google Sheets) can return multiple rows, but VLOOKUP is a single-value lookup function.
- **Option B (Incorrect):** VLOOKUP scans the lookup column from **top to bottom** and stops at the **first** match. It does not scan the entire column to find the last occurrence. The last-match behavior is a common misconception.
- **Option C (Correct):** When duplicate keys exist, VLOOKUP with exact match returns the column value from the **first** matching row it encounters, scanning top-to-bottom through the table_array. This means if "C2" appears in rows 3 and 7, the row 3 data is returned. This top-to-bottom, first-match-wins behavior is standard in both Excel and Google Sheets.
- **Option D (Incorrect):** VLOOKUP does not return `#N/A` for duplicate keys — it simply uses the first match without warning. This silent first-match behavior is actually why duplicates in lookup tables can create subtle data quality issues that are hard to detect.

</details>

---

### 19. Question

A product analyst is building a dashboard and writes `=SUMIF(B2:B6, ">50000", B2:B6)` on a sheet where column B contains revenue values: 45000, 28000, 62000, 15000, 33000. What does the formula return?

- [ ] **A.** 183000 — the sum of all five revenue values
- [ ] **B.** 2 — the count of rows where revenue exceeds 50000
- [ ] **C.** 62000 — the sum of revenue values greater than 50000
- [ ] **D.** 107000 — the sum of all revenue values that are at least 45000

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 183000 is the total of all five values: 45000 + 28000 + 62000 + 15000 + 33000. The formula applies a filter (`>50000`), excluding values below that threshold.
- **Option B (Incorrect):** 2 is the count of values exceeding 50000, which `=COUNTIF(B2:B6, ">50000")` returns. This SUMIF formula sums qualifying values, not counts them.
- **Option C (Correct):** The formula uses both `range` and `sum_range` pointing to the same column B. It evaluates each cell in B2:B6 against the criterion `">50000"`. Only 62000 (row 4) qualifies — 45000 does not exceed 50000 (it equals but is not greater). Sum = **62000**. (Note: `>50000` excludes values equal to 50000.)
- **Option D (Incorrect):** 107000 (45000 + 62000) would result from the criterion `">=45000"` (greater than or equal to). The formula uses strict `">"` so 45000 itself is excluded. Only 62000 passes `>50000`.

</details>

---

### 20. Question

A business analyst has a Transactions table (rows 2–101) where column E holds the transaction **Category** (text) and column F holds the **Amount** (numeric). She enters `=AVERAGEIF(E2:E101, "Retail", F2:F101)`. Which of the following descriptions most accurately explains what this formula computes?

- [ ] **A.** The total sum of all Amount values where Category equals "Retail"
- [ ] **B.** The count of transactions where Category equals "Retail"
- [ ] **C.** The arithmetic mean of Amount values for only those transactions where Category equals "Retail"
- [ ] **D.** The largest Amount value among Retail transactions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Summing matching values is `SUMIF(E2:E101, "Retail", F2:F101)`. AVERAGEIF computes a mean, not a total. Confusing SUMIF and AVERAGEIF is a very common beginner mistake.
- **Option B (Incorrect):** Counting matching rows is `COUNTIF(E2:E101, "Retail")`. AVERAGEIF does not count — it computes an average of numeric values in the specified average_range for rows that match the criterion.
- **Option C (Correct):** `AVERAGEIF(criteria_range, criterion, average_range)` computes the mean of `average_range` values (F2:F101, Amount) only for rows where `criteria_range` (E2:E101, Category) equals "Retail". This produces the average transaction amount across all Retail transactions — the precise definition of AVERAGEIF's output.
- **Option D (Incorrect):** Finding the maximum matching value requires `=MAXIFS(F2:F101, E2:E101, "Retail")` (Excel 2019+/Google Sheets). AVERAGEIF returns a mean, not a maximum.

</details>
