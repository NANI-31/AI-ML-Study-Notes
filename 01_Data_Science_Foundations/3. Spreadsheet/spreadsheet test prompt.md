Generate 100 New Objective Questions from the Reference Markdown File

Use the attached/reference Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference file:

[test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/spreadsheet/test.md)

Task

Analyze the entire reference .md file carefully and generate 100 NEW multiple-choice objective questions based on the concepts covered in it.

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create:

test-01.md
test-02.md
test-03.md
test-04.md
test-05.md
test-06.md
test-07.md
test-08.md
test-09.md
test-10.md

Each file must contain exactly 10 questions, for a total of 100 questions.

1. Use the Reference File Correctly

First, thoroughly analyze the supplied test.md.

Identify:

- Topics and subtopics covered:
  - VLOOKUP mechanics, syntax, arguments, leftmost column requirement, column index numbers, and lookup tables
  - Exact match (`FALSE` / `0`) vs. approximate match (`TRUE` / `1`), sorted-key prerequisites, and silent match errors
  - Cell references: relative (`A2:D6`), absolute (`$A$2:$D$6`), mixed (`$A2`, `A$2`), and cross-sheet references (`SheetName!$A$2:$D$6`)
  - Range shifting errors when copying formulas down rows or across columns without locking (`$`)
  - Conditional aggregation functions: `SUMIF`, `COUNTIF`, `AVERAGEIF`
  - Multi-conditional aggregation functions: `SUMIFS`, `COUNTIFS`, `AVERAGEIFS`
  - Argument ordering differences: `SUMIF(criteria_range, criterion, [sum_range])` vs. `SUMIFS(sum_range, criteria_range1, criterion1, ...)`
  - Multi-criteria logical evaluation: `AND` evaluation in `SUMIFS`/`COUNTIFS` vs. `OR` logic
  - Pivot table layout and architecture: Rows, Columns, Values, and Filters areas
  - Appropriate data types for pivot areas (numeric measures in Values; categories/dimensions in Rows/Columns; categorical status/dates in Filters)
  - Aggregation methods in pivot values: Sum, Count, Average, Max, Min
  - Dynamic behavior: live automatic recalculation of formulas vs. pivot table static cell addresses (`PivotRegion!B4`)
  - Data enrichment / relational join patterns: using `VLOOKUP` to bring parent attributes (e.g., Customer City, Segment) into transactional child rows (e.g., Orders)
  - Spreadsheet error behavior: `#N/A` on missing lookup keys, formula syntax errors, wrong argument counts
- Important definitions:
  - Lookup value, Table array, Column index number, Range lookup
  - Criteria range, Criterion, Sum range, Average range
  - Absolute reference, Relative reference, Gutter row number vs. data index
  - Pivot table areas (Rows, Columns, Values, Filters)
  - Data enrichment, Recalculation engine
- Common mistakes and misconceptions:
  - Swapping the 3rd and 4th arguments of `VLOOKUP`
  - Forgetting that `col_index_num` is 1-based relative to `table_array`, not the entire worksheet's column letter
  - Forgetting dollar signs (`$`), leading to shifting search ranges when copied down
  - Thinking `SUMIF` can take multiple criteria (confusing `SUMIF` with `SUMIFS`)
  - Confusing the argument order of `SUMIF` (sum range last) with `SUMIFS` (sum range first)
  - Believing `COUNTIFS` uses `OR` logic across multiple criteria rather than `AND` logic
  - Attempting to sum text fields in a pivot table's Values area
  - Believing pivot tables automatically eliminate the need for conditional formulas, or vice versa
  - Expecting cell references pointing to a pivot table (e.g., `=PivotRegion!B4`) to remain stable when the pivot layout changes
- Practical data analytics scenarios:
  - Regional Sales Dashboards, e-commerce Orders and Customers tables, inventory tracking, customer segmentation (Enterprise vs. SMB)
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new sample tables, numerical values, business contexts, formula variations, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference file.

Avoid:

- Copying the same question with different customer names
- Reversing the answer choices
- Changing only city names (e.g., replacing "Mumbai" with "Pune" without altering the core problem)
- Rephrasing an existing question without changing what it tests
- Reusing the exact same table repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to evaluate, interpret, trace, calculate, debug, or reason about spreadsheet mechanics.

For example, if the reference covers `VLOOKUP`, don't generate 10 questions that merely ask what the 3rd argument does.

Instead, vary the question types:

- Trace the output of a formula given a small, clearly defined data table
- Diagnose why a copied formula begins returning `#N/A` or incorrect values from row 4 onwards
- Identify syntax errors in nested or multi-argument functions
- Compare the output of `SUMIF` vs. `SUMIFS` vs. `COUNTIF` vs. `AVERAGEIF` on the same data range
- Determine which pivot table area (Rows, Columns, Values, Filters) correctly fulfills a specific reporting requirement
- Explain what happens when a lookup key is missing from the reference table
- Analyze the impact of changing source data on live formulas vs. pivot table cell links
- Differentiate between exact matching (`FALSE`) and approximate matching (`TRUE`)
- Determine whether single-condition or multi-condition formulas are required for a business scenario
- Reason about row-relative indexing vs. column-absolute referencing in formula auditing

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring a small data table or formula snippet)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, data table if applicable, and clearly phrased question]

- [ ] **A.** [Option text]
- [ ] **B.** [Option text]
- [ ] **C.** [Option text]
- [ ] **D.** [Option text]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Explain clearly and specifically why this option is wrong, identifying the common misconception.
- **Option B (Correct):** Explain clearly why this option is correct, providing formula logic or spreadsheet mechanics.
- **Option C (Incorrect):** Explain clearly why this option is wrong.
- **Option D (Incorrect):** Explain clearly why this option is wrong.

</details>

---

If multiple answers are correct (multi-select), list all correct letters in the answer key (e.g., `**Correct Answer:** **A, D**`) and explain every option individually.

4. Question Difficulty

Keep the difficulty aligned with the reference material, but gradually increase the reasoning and depth required across the 10 files:

- Sets 01–03 (Questions 1–30): Core lookup and conditional aggregation basics, VLOOKUP argument structure, absolute vs. relative reference fundamentals (`$`), single-condition `SUMIF`, `COUNTIF`, and `AVERAGEIF`.
- Sets 04–07 (Questions 31–70): Table calculations, `SUMIFS`, `COUNTIFS`, `AVERAGEIFS` multi-condition evaluation, copying formulas across sheets and rows, pivot table configuration (Rows, Columns, Values, Filters), error diagnosing (`#N/A`, shifting ranges).
- Sets 08–10 (Questions 71–100): Advanced data enrichment dashboards, multi-table joins via lookups, formula vs. pivot table architectural trade-offs, cross-sheet pivot references, edge cases (case-insensitivity, missing keys, approximate match failure modes).

Use a healthy mixture of:

- Conceptual questions
- Formula output tracing questions (with small tables)
- Syntax and error debugging questions
- Diagnostic & scenario-based questions
- Formula comparison questions (`SUMIF` vs `SUMIFS`, `COUNTIF` vs `COUNTIFS`)
- Pivot table design questions
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate formulas against tables, trace cell references, or reason through spreadsheet design decisions.

5. Coverage

Across all 100 questions, cover spreadsheet concepts comprehensively:

- VLOOKUP & Table Lookups:
  - Exact syntax: `=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])`
  - The leftmost column rule: search value must be in the first column of `table_array`
  - Column index numbers: 1-based relative to `table_array`, independent of worksheet column letters
  - Exact match (`FALSE` / `0`) vs. Approximate match (`TRUE` / `1`)
  - Why exact match is necessary for unique key lookups (IDs, SKUs, codes)
  - Error behavior: `#N/A` returned when exact match key is missing
  - Data enrichment pattern: joining dimension tables (Customers, Products) to fact tables (Orders, Transactions)
  - Modern alternatives context: `XLOOKUP` / `INDEX-MATCH` (looking to the left, independent column ranges)
- Cell Referencing & Locking (`$`):
  - Relative references (`A2:D6`): shift down/right when copied
  - Absolute references (`$A$2:$D$6`): remain locked when copied across rows and columns
  - Mixed references (`$A2` locks column; `A$2` locks row)
  - Range shifting bug diagnosis: formula searching outside target data when relative references shift
  - Cross-sheet references: syntax `SheetName!Range` (e.g., `Customers!$A$2:$D$6`, `PivotRegion!B4`)
  - Impact of renaming sheets or moving pivot table layout on hardcoded cell references
- Single-Condition Aggregations:
  - `SUMIF(range, criteria, [sum_range])`
  - `COUNTIF(range, criteria)`
  - `AVERAGEIF(range, criteria, [average_range])`
  - Criteria syntax: text (`"Completed"`), comparison operators (`">50000"`, `"<100"`), cell references (`">"&B1`)
  - Default sum_range behavior in `SUMIF` when omitted (sums the criteria range itself)
- Multi-Condition Aggregations:
  - `SUMIFS(sum_range, criteria_range1, criterion1, criteria_range2, criterion2, ...)`
  - `COUNTIFS(criteria_range1, criterion1, criteria_range2, criterion2, ...)`
  - `AVERAGEIFS(average_range, criteria_range1, criterion1, ...)`
  - Critical syntax contrast: `SUMIFS` places `sum_range` first; `SUMIF` places `sum_range` last
  - Logical `AND` behavior: all conditions must be satisfied simultaneously
  - How to handle `OR` logic (separate `SUMIF` additions or array formulas)
- Pivot Tables & Dashboard Design:
  - Four functional areas: Rows, Columns, Values, Filters
  - Placing dimensions in Rows (one row per category) vs. Columns (side-by-side comparison)
  - Appropriate placement of measures (numeric amounts in Values; non-numeric labels in Rows/Columns/Filters)
  - Summary calculations in Values: Sum, Count, Average, Max, Min
  - Filters area vs. Slicers: restricting view to specific subsets (e.g., Status = "Completed")
  - Live formula recalculation upon source edits vs. Pivot table data caching and refreshing
  - Direct cross-sheet links to pivot cells (`=PivotRegion!B4`): fragility when pivot table re-sorts or layout expands

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated spreadsheet tools (such as complex VBA macros, Google Apps Script, or Power Query M-code) merely to fill question count.

6. Numerical & Table Evaluation Questions

Include meaningful table evaluation questions across the sets.

Provide clean, markdown-formatted sample tables (3 to 6 rows) so learners can mentally trace the formula.

For example:

| Row | ProductID | Category    | UnitPrice | UnitsSold |
| :-- | :-------- | :---------- | :-------- | :-------- |
| 2   | P101      | Electronics | 1200      | 5         |
| 3   | P102      | Furniture   | 450       | 10        |
| 4   | P103      | Electronics | 800       | 4         |
| 5   | P104      | Groceries   | 50        | 40        |

Ask questions such as:

- What value is returned by `=VLOOKUP("P103", $A$2:$D$5, 3, FALSE)`?
- What value is returned by `=SUMIF(B2:B5, "Electronics", D2:D5)`?
- What value is returned by `=SUMIFS(D2:D5, B2:B5, "Electronics", C2:C5, ">1000")`?
- What value is returned by `=COUNTIFS(B2:B5, "Electronics", D2:D5, ">=4")`?
- What value is returned by `=AVERAGEIF(B2:B5, "Electronics", C2:C5)`?

Ensure all arithmetic is straightforward and verifiable so learners can calculate results without calculators.

7. Formula Syntax & Debugging Questions

Include realistic formula troubleshooting questions.

Present candidate formulas and ask:

- Which formula correctly pulls the Product Name from column 2?
- Why does `=VLOOKUP("P101", A2:D5, 5, FALSE)` produce a `#REF!` error? (Index 5 exceeds 4 columns in range)
- Why does `=SUMIF(D2:D5, "Electronics", B2:B5)` return 0? (Swapped criteria range and sum range)
- Why does row 5 of a copied lookup formula return `#N/A` when the product exists in row 2? (Missing absolute reference `$` caused range to shift down)
- Which formula correctly counts orders where Status is "Pending" AND Region is "North"?

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C & D, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, spreadsheet-standard defensible answer according to Microsoft Excel and Google Sheets behavior.

Avoid:

- Undefined row or column coordinates in formula examples
- Ambiguous criteria strings (e.g., unquoted text criteria in formulas)
- Relying on platform-specific quirks unless explicitly stated
- Conflating `COUNTIF` (count of rows) with `SUMIF` (sum of values)
- Omitting sheet names when referencing cross-sheet tables

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., which argument was violated, how the range shifted, or why the calculation yielded a different number).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files.

Each file must contain exactly 10 questions.

Use this continuous numbering across the files:

test-01.md → Questions 1–10
test-02.md → Questions 11–20
test-03.md → Questions 21–30
test-04.md → Questions 31–40
test-05.md → Questions 41–50
test-06.md → Questions 51–60
test-07.md → Questions 61–70
test-08.md → Questions 71–80
test-09.md → Questions 81–90
test-10.md → Questions 91–100

Do not restart numbering at 1 in each file.

12. File Titles

At the beginning of each file, include a clear header matching the set:

# Spreadsheet Objective Test — Set 01

Followed immediately by the 10 questions belonging to that set.

13. Quality-Control Requirements

Before completing the generation, perform a comprehensive validation against these checkpoints:

- [ ] Exactly 100 questions total.
- [ ] Exactly 10 `.md` files created.
- [ ] Exactly 10 questions per file.
- [ ] Continuous question numbering from 1 to 100.
- [ ] Every question has exactly four choices: A, B, C, D with blank checkboxes `- [ ]`.
- [ ] Every question has at least one correct answer declared in `**Correct Answer:**`.
- [ ] Every question has an HTML `<details>` and `<summary>` dropdown block.
- [ ] Every option (A, B, C, D) has an explicit explanation inside `#### Explanations:`.
- [ ] Correct answers are reasonably balanced across A/B/C/D.
- [ ] All questions are original and not verbatim copies of the reference.
- [ ] Mathematical and formula evaluations (sums, counts, averages, lookups) are verified and accurate.
- [ ] Spreadsheet syntax (`VLOOKUP`, `SUMIF`, `SUMIFS`, `COUNTIF`, `COUNTIFS`, `$`) is 100% valid in Excel / Google Sheets.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Spreadsheets, formula evaluation, data enrichment, and dashboard design.
