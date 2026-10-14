### 1 Question

A product manager is reviewing this Customers table:

| Row | CustomerID | Name             | City      | Segment    |
| :-- | :--------- | :--------------- | :-------- | :--------- |
| 2   | C1         | Meridian Traders | Mumbai    | Enterprise |
| 3   | C2         | Bright Retail    | Delhi     | SMB        |
| 4   | C3         | Coastal Foods    | Chennai   | Enterprise |
| 5   | C4         | Delta Supplies   | Bangalore | SMB        |
| 6   | C5         | Everest Mart     | Hyderabad | Enterprise |

A cell on another sheet contains `=VLOOKUP("C3", Customers!$A$2:$D$6, 3, FALSE)`. What value does this formula return?

### Options

- [ ] A) "Coastal Foods" — the customer's name, found in the second column of the table
- [ ] B) "C3" — the original lookup value, not a value returned by the formula
- [x] **C) "Chennai" — the customer's city, found in the third column of the table** _(Correct)_
- [ ] D) "Enterprise" — the customer's segment, found in the fourth column of the table

### Answer Explanation

- **A) Wrong:** "Coastal Foods" sits in the second column (Name); it would only be returned if `col_index_num` were 2 instead of 3.
- **B) Wrong:** "C3" is simply the lookup value itself, not a returned result — `VLOOKUP` never returns the value it searched for unless that value happens to also sit in the requested output column.
- **C) Correct:** The formula looks up "C3" in the leftmost column, finds row 4, and returns the value from the third column of the range, which is City — "Chennai".
- **D) Wrong:** "Enterprise" sits in the fourth column (Segment); it would only be returned if `col_index_num` were 4 instead of 3.

---

### 2 Question

As part of a broader revenue analysis, a senior data analyst is building a Regional Sales Dashboard and needs the total revenue from orders that are both Completed AND belong to the Enterprise segment, using enriched Orders columns where D holds Status, E holds Amount, and H holds Segment (rows 2 to 11). Which formula correctly returns this value?

### Options

- [x] **A) `=SUMIFS(E2:E11, D2:D11, "Completed", H2:H11, "Enterprise")` — sums Amount where Status is Completed AND Segment is Enterprise** _(Correct)_
- [ ] B) `=SUMIF(D2:D11, "Completed", E2:E11)` — sums Amount where Status is Completed, without checking Segment at all
- [ ] C) `=COUNTIFS(D2:D11, "Completed", H2:H11, "Enterprise")` — counts matching orders rather than summing their revenue
- [ ] D) `=AVERAGEIF(D2:D11, "Completed", E2:E11)` — averages Amount for completed orders, without checking Segment at all

### Answer Explanation

- **A) Correct:** SUMIFS supports multiple conditions, so it sums the Amount range only where Status equals "Completed" AND Segment equals "Enterprise" — exactly the two-condition revenue figure needed.
- **B) Wrong:** SUMIF only supports one condition, so it would sum every completed order's amount regardless of segment, mixing Enterprise and SMB revenue together.
- **C) Wrong:** COUNTIFS returns how many orders match, not their combined revenue — the wrong output type for a revenue question.
- **D) Wrong:** AVERAGEIF only supports one condition and returns an average of completed orders' amounts, not a segment-filtered sum.

---

### 3 Question

A junior ML engineer is testing a formula on this small extract of the Orders sheet. The "Row" column below shows each row's position in the sheet (like the row-number gutter on the left edge of a spreadsheet, which is separate from the lettered columns) — the actual lettered columns are OrderID in column A, Status in column B, and Amount in column C:

| Row | OrderID | Status    | Amount |
| :-- | :------ | :-------- | :----- |
| 2   | O1      | Completed | 65000  |
| 3   | O2      | Pending   | 12000  |
| 4   | O3      | Completed | 20000  |
| 5   | O4      | Cancelled | 15000  |
| 6   | O5      | Completed | 3000   |

The formula `=SUMIF(B2:B6, "Completed", C2:C6)` is entered in an empty cell. The engineer jots down four candidate statements about the formula's result before trusting it. Which statement is accurate?

### Options

- [ ] A) 65000 — only the first completed order's amount, ignoring the other matching rows
- [ ] B) 100000 — the sum of all five amounts in the range, ignoring the status criteria
- [ ] C) 3 — the count of completed rows rather than their summed amount
- [x] **D) 88000 — the sum of amounts from all rows where status equals Completed** _(Correct)_

### Answer Explanation

- **A) Wrong:** 65000 is only row 2's amount; it ignores that rows 4 and 6 also have status "Completed" and must be included in the sum.
- **B) Wrong:** 100000 adds all five amounts in the range regardless of status, which ignores the criteria argument entirely.
- **C) Wrong:** 3 is the count of completed rows, which is what COUNTIF would return, not the summed amount that SUMIF returns.
- **D) Correct:** SUMIF checks the criteria range (B2:B6) for cells equal to "Completed" — rows 2, 4, and 6 — and adds the matching values from the sum range (C2:C6): 65000 + 20000 + 3000 = 88000.

---

### 4 Question

A senior data analyst is scanning a shared spreadsheet and needs to identify which VLOOKUP formula is written correctly to fetch a customer's segment as an EXACT match against a customer ID stored in cell B2, using a locked reference to the Customers table in Customers!A2:D6 .

### Options

- [ ] A) `=VLOOKUP(B2, Customers!$A$2:$D$6, 4, TRUE)`
- [x] **B) `=VLOOKUP(B2, Customers!$A$2:$D$6, 4, FALSE)`** _(Correct)_
- [ ] C) `=VLOOKUP(B2, Customers!$A$2:$D$6, TRUE, 4)`
- [ ] D) `=VLOOKUP(4, Customers!$A$2:$D$6, B2, FALSE, TRUE)`

### Answer Explanation

- **A) Wrong:** TRUE requests an approximate match instead of an exact one. Approximate matches require a sorted lookup column and can silently return the wrong row, which is why exact matching was used to join tables by ID.
- **B) Correct:** The lookup value (B2) comes first, then the locked table range, then the column number to return (4, for Segment), then FALSE for an exact match — the standard argument order for pulling segment data by customer ID.
- **C) Wrong:** This swaps the third and fourth arguments, putting TRUE where the column number belongs and 4 where the match-type flag belongs, which breaks VLOOKUP's required argument order.
- **D) Wrong:** This swaps the lookup value and column number entirely (searching for the number 4 instead of the customer ID) and tacks on a fifth argument that VLOOKUP does not accept.

---

### 5 Question

A product manager built a Regional Sales Dashboard using VLOOKUP to enrich the Orders sheet, SUMIFS/COUNTIFS formulas for segment totals, and a pivot table sourced from Orders!A1:H11 for the dashboard analysis. After a teammate edits several rows of the underlying Orders data, which of the following statements are TRUE?

### Options

- [x] **A) Cells containing SUMIFS or COUNTIF formulas recalculate automatically as soon as the underlying Orders data changes** _(Correct)_
- [ ] B) If the teammate's edit changes which segment a CustomerID belongs to in the Customers table, the VLOOKUP-enriched Segment column in Orders will not reflect that updated segment until the workbook is closed and reopened
- [x] **C) A newly added order row whose CustomerID is missing from the Customers table will cause that row's VLOOKUP for City to return an error** _(Correct)_
- [x] **D) Locking the Customers range as `Customers!$A$2:$D$6` keeps the VLOOKUP lookup range from shifting when copied down every row** _(Correct)_

### Answer Explanation

- **A) Correct:** Formulas such as SUMIFS and COUNTIF are live and recalculate the moment their source cells change, which is why the dashboard's revenue and count cells update immediately after values are edited.
- **B) Wrong:** VLOOKUP is a formula like any other, so it recalculates live the moment its source data changes — including a change to which segment a CustomerID maps to — without needing the workbook to be closed and reopened.
- **C) Correct:** VLOOKUP with an exact match (FALSE) cannot find a lookup value that does not exist in the table's leftmost column, so it returns an error for any order whose CustomerID has no matching row in Customers.
- **D) Correct:** An absolute reference such as `$A$2:$D$6` stays fixed as the formula is copied down, keeping every row's VLOOKUP searching the same, complete Customers range instead of one that has shifted.

---

### 6 Question

A backend engineer who is new to spreadsheet-based data analysis is told that a colleague used VLOOKUP to pull customer details into an orders sheet. Which statement best describes what VLOOKUP actually does?

### Options

- [x] **A) It looks up a value in a table's leftmost column and returns a related value from a chosen column in that same row** _(Correct)_
- [ ] B) It counts how many cells in a specified range satisfy one given condition, such as a status value
- [ ] C) It adds together the values in a range of cells that satisfy one given condition, such as a status value
- [ ] D) It builds a summarized, drag-and-drop table from raw data without requiring any formula to be written

### Answer Explanation

- **A) Correct:** VLOOKUP takes a lookup value, searches for it in the leftmost column of a specified table range, and returns a value from a column chosen within that same matched row — this is exactly how a customer's city and segment were pulled into the orders sheet by matching on customer ID.
- **B) Wrong:** This describes COUNTIF, which counts matching cells rather than looking up related data across two tables.
- **C) Wrong:** This describes SUMIF, which totals matching values rather than performing a row-level lookup between tables.
- **D) Wrong:** This describes a pivot table, a separate drag-and-drop summarization tool that is not a lookup formula at all. XLOOKUP is a newer function that can achieve a similar leftmost-independent result, but VLOOKUP is the classic tool used for this pattern in spreadsheet-based data analysis.

---

### 7 Question

A senior data analyst has built a Regional Sales Dashboard combining VLOOKUP, SUMIF-family formulas, and a pivot table on a sheet renamed "PivotRegion." Which of the following statements about the dashboard are TRUE, based on how these tools were covered in this session?

### Options

- [x] **A) AVERAGEIF works exactly like SUMIF but returns an average instead of a total, so `=AVERAGEIF(D2:D11, "Completed", E2:E11)` returns the average Amount for completed orders.** _(Correct)_
- [x] **B) A cross-sheet reference such as `=PivotRegion!B4` points to a fixed cell address, so if the pivot table's layout changes, that reference may end up returning a different value than before.** _(Correct)_
- [ ] C) A pivot table can fully replace every SUMIF, COUNTIF, SUMIFS, COUNTIFS, and AVERAGEIF calculation used in the dashboard, so none of these formulas are ever needed once a pivot table exists.
- [x] **D) Locking the SUMIF/COUNTIF range arguments with `$` keeps them from shifting if the formula is later copied across multiple cells — the same principle used for VLOOKUP's table_array.** _(Correct)_

### Answer Explanation

- **A) Correct:** AVERAGEIF works exactly like SUMIF, but returns an average instead of a total, so this formula returns the average order amount across completed orders only.
- **B) Correct:** A cross-sheet reference to a pivot table cell points at a fixed cell address, not the labeled row/column it currently represents — if the pivot table's shape changes (a field is added or the data resorts), that reference may end up pointing at a different value than before.
- **C) Wrong:** A pivot table does not replicate everything a formula like SUMIF or COUNTIF can do — some conditional calculations are more naturally expressed as a formula than squeezed into a pivot table's row/column/value structure, so conditional formulas remain necessary alongside pivot tables, not replaced by them.
- **D) Correct:** SUMIF and COUNTIF ranges need the same $ -locking treatment as VLOOKUP's table_array, so the range argument stays fixed while the formula is copied across multiple cells.

---

### 8 Question

A backend engineer writes `=VLOOKUP(C2, Customers!A2:D6, 3, FALSE)` in cell G2 of the Orders sheet to pull each order's city, then copies it down through row 11. Row G2 returns the correct city, but from G4 onward the formula returns errors or clearly wrong cities. What is the most likely cause, and how should it be fixed?

### Options

- [ ] A) The col_index_num argument should be changed from 3 to 4 to point at a different column
- [x] **B) The table_array should be locked as `Customers!$A$2:$D$6` so it doesn't shift when copied down** _(Correct)_
- [ ] C) The range_lookup argument should be changed from FALSE to TRUE for an approximate match
- [ ] D) VLOOKUP formulas cannot be copied down to other rows and must be re-typed manually in every single row

### Answer Explanation

- **A) Wrong:** Column 3 correctly points to the City column no matter which row the formula sits in; the column number is not the source of this error.
- **B) Correct:** Without dollar signs, A2:D6 is a relative reference, so it shifts down one row for every row the formula is copied into (row 4's formula effectively searches A4:D8 ), eventually searching outside the actual Customers table. Locking it with `$A$2:$D$6` (the sheet name plus ! still points at the Customers sheet) keeps the table range fixed for every copied row.
- **C) Wrong:** Switching to an approximate match would not stop the range from shifting and could introduce incorrect matches on top of the existing problem.
- **D) Wrong:** VLOOKUP formulas are routinely copied down across rows; the issue here is a missing absolute reference, not an inherent limitation of the function.

---

### 9 Question

A DevOps engineer reviewing a colleague's spreadsheet finds four formulas, each with a claimed purpose. On the Orders sheet, column D holds Status, column E holds Amount, and column H holds Segment (rows 2 to 11). Which of the following formulas actually do what they claim?

### Options

- [ ] A) `=SUMIF(H2:H11, "Enterprise")` — claimed to return the total order amount for Enterprise-segment orders
- [ ] B) `=COUNTIFS(D2:D11, "Completed", H2:H11, "SMB")` — claimed to count orders that are either Completed or SMB segment
- [x] **C) `=COUNTIF(D2:D11, "Completed")` — claimed to return the number of orders with status Completed** _(Correct)_
- [x] **D) `=AVERAGEIF(D2:D11, "Completed", E2:E11)` — claimed to return the average order amount for orders with status Completed** _(Correct)_

### Answer Explanation

- **A) Wrong:** This formula omits the sum range (the Amount column) entirely — SUMIF here only has a range and a criterion, so it cannot return an order-amount total without a third argument pointing to E2:E11.
- **B) Wrong:** COUNTIFS combines its conditions with AND logic, not OR — this formula counts orders that are BOTH Completed AND SMB segment at once, not orders matching either condition.
- **C) Correct:** COUNTIF with a single range and criterion counts how many cells in D2:D11 equal "Completed," which is exactly the number of completed orders, as claimed.
- **D) Correct:** AVERAGEIF averages the values in E2:E11 only for rows where D2:D11 equals "Completed," giving the average amount of completed orders, as claimed.

---

### 10 Question

A product manager is dragging fields into a pivot table to complete a Regional Sales Dashboard that should show revenue broken down by region and segment. Which of the following field placements are appropriate?

### Options

- [x] **A) Dragging Region into the Rows area so each region gets its own row in the pivot table** _(Correct)_
- [x] **B) Dragging Segment into the Columns area so Enterprise and SMB revenue can be compared side-by-side** _(Correct)_
- [ ] C) Dragging Status into the Values area so Excel or Google Sheets can sum the order status labels
- [ ] D) Dragging Amount into the Filters area so totals can be filtered by amount range

### Answer Explanation

- **A) Correct:** Placing Region in the Rows area gives the pivot table one row per region, the standard way to break results out by region.
- **B) Correct:** Placing Segment in the Columns area creates side-by-side columns for each segment (Enterprise, SMB), letting revenue be compared across segments within each region's row.
- **C) Wrong:** Status is a text field (Completed, Pending, Cancelled) that cannot be meaningfully summed in the Values area; Status belongs in the Filters area, where it can limit the view to completed orders.
- **D) Wrong:** Amount is the numeric measure that should sit in the Values area, summarized by Sum, Count, Average, or Max as needed — putting it in Filters would prevent it from being aggregated at all.
