Generate 100 New Objective Questions from the Reference Ticks & Scales Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Ticks, Spines, Limits, and Scales (Explicit Tick Positioning, Custom Formatted Labels, Rotation & Alignment Mechanics, Coordinate Limits & Inverted Axes, Scale Transformations like `log`/`symlog`/`logit`, Major vs. Minor Locators, `FuncFormatter`/`PercentFormatter`, `tick_params` Styling, Major/Minor Grid Binding, and Precision-Recall/Learning-Rate ML Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/ticks_scales/`:

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

1. Use the Reference Material Correctly

First, thoroughly analyze the supplied `5. ticks.ipynb`, `1. basic.ipynb`, `3. plot.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Explicit Tick Positioning and Custom Labeling:
    - Pyplot interface: `plt.xticks(ticks, labels)` and `plt.yticks(ticks, labels)`
    - Object-Oriented interface: `ax.set_xticks(ticks)` and `ax.set_xticklabels(labels)`
    - Mapping numeric coordinates to domain strings:
      - Categorical names: `[0, 1, 2, 3, 4]`, `["Jan", "Feb", "Mar", "Apr", "May"]`
      - Currency formatting: `[0, 100, 200, 300]`, `["$0", "$100", "$200", "$300"]`
      - Percentage formatting: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`, `["0%", "20%", "40%", "60%", "80%", "100%"]`
  - Label Rotation, Alignment & Overlap Prevention:
    - `rotation=45`, `rotation=90`
    - Horizontal and vertical alignment: `ha='right'`, `ha='center'`, `va='top'`
    - Why `ha='right'` is essential with $45^\circ$ rotated labels to keep the text label aligned with the tick mark
  - Axis Coordinate Limits:
    - `plt.xlim(min, max)` / `plt.ylim(min, max)`
    - `ax.set_xlim(left, right)` / `ax.set_ylim(bottom, top)`
    - Axis inversion: `ax.set_ylim(1.0, 0.0)` or `ax.invert_yaxis()` (used in ranking charts, screen coordinates, or depth metrics)
    - Autoscaling control: `ax.autoscale(enable=True, axis='both', tight=True)`
  - Scale Transformations:
    - Linear: default scale
    - Logarithmic: `plt.yscale('log')`, `ax.set_yscale('log', base=10)` or `base=2`
    - Symmetrical Logarithmic (`symlog`): handling negative values, zero, and positive values with `linthresh` (linear threshold around zero)
    - Logit scale: `ax.set_yscale('logit')` for probabilities bounded in $(0, 1)$
  - Major vs. Minor Tick Locators (`matplotlib.ticker`):
    - `MultipleLocator(spacing)`: fixed interval spacing
    - `MaxNLocator(nbins)`: auto-selecting a maximum number of clean intervals
    - `AutoMinorLocator(n)`: adding minor tick subdivisions between major ticks
    - `LogLocator(base=10)`: geometric progression ticks
    - `NullLocator()`: completely suppressing ticks
    - Attaching locators: `ax.xaxis.set_major_locator(...)`, `ax.xaxis.set_minor_locator(...)`
  - Tick Label Formatters (`matplotlib.ticker`):
    - `FuncFormatter(callable)`: dynamic programmatic formatting (e.g., lambda function for scientific units or financial abbreviations)
    - `PercentFormatter(xmax=1.0)`: converting decimal fractions to clean percentages
    - `StrMethodFormatter('{x:.2f}')`: modern format string mapping
    - Attaching formatters: `ax.yaxis.set_major_formatter(...)`
  - Tick Styling & Parameter Customization (`tick_params`):
    - `ax.tick_params(axis='both', which='both'/'major'/'minor', direction='in'/'out'/'inout', length=6, width=1.5, colors='gray', pad=6, labelsize=11)`
    - Hiding tick marks while keeping labels: `bottom=False` or `tick_params(length=0)`
    - Tick marks on top and right axes: `top=True`, `right=True`, `labeltop=True`
  - Grid Line Binding to Ticks:
    - Major grid lines: `ax.grid(True, which='major', linestyle='-', alpha=0.7)`
    - Minor grid lines: `ax.grid(True, which='minor', linestyle=':', alpha=0.4)`
    - Axis-specific grids: `ax.grid(axis='x')` or `ax.grid(axis='y')`
  - Date & Time Scales (`matplotlib.dates`):
    - Formatting date axes: `DateFormatter('%Y-%m-%d')`
    - Date tick locators: `MonthLocator()`, `DayLocator(interval=7)`
    - Auto-formatting slanted date ticks: `fig.autofmt_xdate(rotation=45)`
  - Legend Placement Relative to Ticks & Axes Bounds:
    - Moving legends outside the axes box: `loc="upper left"`, `bbox_to_anchor=(1.02, 1)`
    - Preventing legends from colliding with outer tick labels and titles
  - Machine Learning & Data Science Applications:
    - Learning rate schedules plotted on logarithmic scales ($\eta = 10^{-5}$ to $10^{-1}$)
    - Training vs. validation loss convergence with minor tick gridlines
    - Confusion matrix tick labeling with categorical target class names
    - Precision-Recall curves bounded strictly in $[0, 1]$ with percentage tick formatters
- Important definitions:
  - Major Ticks, Minor Ticks, TickLocator, TickFormatter, Symlog, Linthresh, Linscale, Tick Direction, Pad, Inverted Axis, Autoscaling
- Common mistakes and misconceptions:
  - Calling `ax.set_xticklabels(labels)` without first calling `ax.set_xticks(ticks)`, triggering a `FixedFormatter` warning in modern Matplotlib
  - Passing label count that does not match tick location count, leading to misaligned or dropped labels
  - Rotating labels by $45^\circ$ without setting `ha='right'`, causing labels to visually center on the wrong tick mark
  - Trying to plot zero or negative numbers on a pure `log` scale instead of using `symlog`
  - Forgetting that `PercentFormatter(xmax=100)` expects integer percentages while `PercentFormatter(xmax=1.0)` expects decimal fractions
  - Confusing `ax.tick_params(axis='x', which='both')` with `ax.set_xticks()`, using the wrong method to adjust tick visual appearance
  - Overwriting an entire axis limit when attempting to lock only the minimum or maximum bound
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing tick values from `[10, 20, 30]` to `[20, 40, 60]` without altering the analytical demand
- Asking the same rotation question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about Matplotlib's ticks, scales, and coordinate formatting mechanisms.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, machine learning diagnostics, and dashboard engineering:

- Reading and predicting the visual output of code snippets containing locators and formatters
- Calculating tick interval subdivisions, log-scale coordinate mappings, and label offsets
- Identifying formatting bugs, FixedFormatter warnings, and scale clipping errors
- Selecting the correct ticker classes (`MultipleLocator`, `PercentFormatter`, `AutoMinorLocator`) for production plots

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Tick Mathematics: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Production Formatting: ~30 questions

4. Answer Uniformity and Quality

To eliminate test-taking bias and guessing artifacts:

- Randomize and strictly balance the correct answer options across the 100 questions:
  - Approximately 25 questions with Option A as correct
  - Approximately 25 questions with Option B as correct
  - Approximately 25 questions with Option C as correct
  - Approximately 25 questions with Option D as correct
- Ensure distractor options are realistic, plausible, and represent common programmer errors
- Keep the length, technical depth, and grammatical structure of all four choices balanced so the correct choice never stands out by length alone

5. Question Style and Content Guidelines

- Every question must feature four clear choices: A, B, C, and D
- Code blocks must follow modern Python 3 and Matplotlib conventions
- All calculations (e.g., tick intervals, log coordinate transformations, percentage conversions) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what version was FuncFormatter introduced?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_ticks=True`)
- Questions where 3 of the 4 choices are visibly absurd

7. Format Requirements for the Output

The output must be pure Markdown formatted exactly as shown below.

- Maintain continuous global numbering from Question 1 to Question 100 across the 10 files:
  - `test-01.md`: Questions 1–10
  - `test-02.md`: Questions 11–20
  - `test-03.md`: Questions 21–30
  - `test-04.md`: Questions 31–40
  - `test-05.md`: Questions 41–50
  - `test-06.md`: Questions 51–60
  - `test-07.md`: Questions 61–70
  - `test-08.md`: Questions 71–80
  - `test-09.md`: Questions 81–90
  - `test-10.md`: Questions 91–100

Format per question:

```markdown
### Question [Number]

[Question context, code block, or scenario]

- [ ] A) [Option text]
- [ ] B) [Option text]
- [ ] C) [Option text]
- [ ] D) [Option text]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** [Option Letter]) [Option text]

**Explanation:**

- **A is correct/incorrect:** [Detailed rationale]
- **B is correct/incorrect:** [Detailed rationale]
- **C is correct/incorrect:** [Detailed rationale]
- **D is correct/incorrect:** [Detailed rationale]

</details>
```

8. Detailed Explanation Requirements

The explanation must break down all four options:

- Explicitly state why the correct choice is accurate according to the Matplotlib object model
- Detail why each incorrect option fails (e.g., invalid locator, unmatched tick-to-label length, wrong rotation alignment, or scale math error)
- Include concrete references to Matplotlib mechanics (`FixedFormatter` behavior, `AutoMinorLocator` rules, `linthresh` boundaries, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): Tick Locations & Categorical Labels (`set_xticks`, `set_xticklabels`, Pyplot vs. OO methods)
- `test-02.md` (Questions 11–20): Label Aesthetics, Rotation & Alignment (`rotation=45`, `ha='right'`, padding, font styling)
- `test-03.md` (Questions 21–30): Axis Limits & Inverted Ranges (`set_xlim`, `set_ylim`, `invert_yaxis()`, autoscaling rules)
- `test-04.md` (Questions 31–40): Logarithmic & Geometric Scales (`set_yscale('log')`, `base=10`/`base=2`, non-positive value errors)
- `test-05.md` (Questions 41–50): Symmetrical Log (`symlog`) & Specialized Scales (`linthresh`, `linscale`, `logit` scale for probabilities)
- `test-06.md` (Questions 51–60): Major & Minor Tick Locators (`MultipleLocator`, `MaxNLocator`, `AutoMinorLocator`, `NullLocator`)
- `test-07.md` (Questions 61–70): Automated Tick Formatters (`FuncFormatter`, `PercentFormatter`, `StrMethodFormatter`, unit prefixes)
- `test-08.md` (Questions 71–80): `tick_params` Customization (tick direction, lengths, widths, colors, dual-axis ticks)
- `test-09.md` (Questions 81–90): Grid Line Synchronization & Outer Legends (major vs. minor grids, `bbox_to_anchor` positioning)
- `test-10.md` (Questions 91–100): Comprehensive Machine Learning Diagnostics (learning rate schedules, loss curves, confusion matrices)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/ticks_scales/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Matplotlib
8. Mathematical precision for all scales (`log`, `symlog`, percentages, multiple locators)
9. Modern Matplotlib requirement of pairing `set_xticks()` with `set_xticklabels()` is observed
10. `ha='right'` alignment necessity with rotated labels is properly emphasized
11. `MultipleLocator` vs. `MaxNLocator` behavior is accurately tested
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/ticks_scales/` following this prompt.
