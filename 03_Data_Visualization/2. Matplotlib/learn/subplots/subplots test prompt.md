Generate 100 New Objective Questions from the Reference Subplots Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Subplots (Grid Generation via `plt.subplot` vs. `plt.subplots`, Return Array Dimensions, Squeeze Mechanics, 2D Indexing & `axes.flat` Iteration, Shared Axes `sharex`/`sharey`, Spacing Control with `subplots_adjust` and `wspace`/`hspace`, Complex Layouts with `GridSpec` and `subplot_mosaic`, Unused Subplot Deletion, and Multi-Metric ML Dashboards).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/subplots/`:

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

First, thoroughly analyze the supplied `1. basic.ipynb`, `5. ticks.ipynb`, `1. Figure.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Subplot creation functions:
    - Legacy / Pyplot stateful: `plt.subplot(nrows, ncols, index)` (1-based indexing, e.g., `plt.subplot(2, 2, 1)`)
    - Object-Oriented creation: `fig, axes = plt.subplots(nrows, ncols, figsize=...)`
    - Figure-level method: `ax = fig.add_subplot(nrows, ncols, index)`
  - Subplot Return Dimensions & Tuple Unpacking:
    - Single subplot `(1, 1)` $\implies$ returns a single `AxesSubplot` instance
    - Single row `(1, N)` or single column `(N, 1)` $\implies$ 1D NumPy array of length $N$
    - Multi-row, multi-column `(M, N)` ($M, N > 1$) $\implies$ 2D NumPy array with shape `(M, N)`
    - Tuple unpacking syntax: `fig, (ax1, ax2) = plt.subplots(1, 2)` vs. `fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)`
    - The `squeeze` parameter: `squeeze=False` guaranteeing a 2D array output regardless of grid dimensions
  - Indexing and Looping Strategies:
    - 2D grid access: `axes[0, 0]`, `axes[0, 1]`, `axes[1, 0]`, `axes[1, 1]`
    - Linear iteration over all panels: `for i, ax in enumerate(axes.flat): ...` or `axes.ravel()`
    - Handling odd numbers of plots (e.g., 5 features on a 2x3 grid):
      - Deleting unused subplots: `fig.delaxes(axes[1, 2])`
      - Hiding axes: `axes[1, 2].axis('off')` or `axes[1, 2].set_visible(False)`
  - Shared Coordinate Systems (`sharex` and `sharey`):
    - Full sharing: `sharex=True`, `sharey=True`
    - Directional sharing: `sharex='col'` (shared across columns) and `sharey='row'` (shared across rows)
    - Automatic tick label suppression on interior subplots to eliminate visual redundancy
    - Interactive zoom/pan synchronization across linked panels
  - Spacing, Margins, and Padding:
    - `plt.subplots_adjust(left=..., right=..., bottom=..., top=..., wspace=..., hspace=...)`
    - Difference between `wspace` (fraction of axis width) and `hspace` (fraction of axis height)
    - `plt.tight_layout()` vs. `plt.subplots_adjust()` precedence and conflict resolution
    - `layout="constrained"` automated engine preventing title and axis label collisions
  - Advanced Layouts:
    - `GridSpec` (`from matplotlib.gridspec import GridSpec` or `fig.add_gridspec()`)
    - Multi-cell spanning with Python slicing: `ax_top = fig.add_subplot(gs[0, :])`, `ax_bottom_left = fig.add_subplot(gs[1, :2])`
    - Custom width and height ratios: `gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 2]}`
    - `subplot_mosaic`: intuitive ASCII/list layout specification (e.g., `plt.subplot_mosaic([['main', 'hist_y'], ['hist_x', 'corner']])`)
  - Figure-Wide Annotations & Shared Controls:
    - `fig.suptitle("Dashboard Title", y=0.98)` positioning
    - Common axis labels: `fig.supxlabel(...)` and `fig.supylabel(...)`
    - Shared colorbar across all subplots: `fig.colorbar(im, ax=axes.ravel().tolist())`
    - Shared single legend for multi-axis figures: extracting handles and labels from one axis and attaching to `fig.legend()`
  - Practical ML Dashboard Architectures:
    - 2x2 multi-panel training grid (Training Loss, Validation Loss, Training Accuracy, Validation Accuracy)
    - Regression residual diagnostics (Fitted vs. Actual, Residuals vs. Fitted, Residual Histogram, Q-Q Plot)
    - Classification evaluation layout (ROC Curve, Precision-Recall Curve, Confusion Matrix, Feature Importance)
- Important definitions:
  - Subplot, GridSpec, SubplotSpec, Mosaic, Squeeze, Flat, Ravel, Wspace, Hspace, Sharex, Sharey, 1-based indexing vs. 0-based indexing
- Common mistakes and misconceptions:
  - Treating `axes` as a 1D list when `plt.subplots(2, 2)` produces a 2D array, causing `TypeError` or `IndexError`
  - Unpacking `fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)` without nested parentheses, causing `ValueError: too many values to unpack`
  - Forgetting that `plt.subplot(2, 2, 1)` uses 1-based indexing while `axes[0, 0]` uses 0-based indexing
  - Manually hiding an axis with `ax.plot([], [])` instead of `ax.axis('off')` or `fig.delaxes(ax)`
  - Calling `plt.subplots_adjust()` and then immediately calling `plt.tight_layout()`, overwriting the custom spacing
  - Assuming `sharex=True` only links limits without realizing it automatically removes internal tick labels
  - Adding a separate colorbar to every single subplot instead of linking one global colorbar to the `Axes` array
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing a 2x2 grid to a 3x3 grid without altering the conceptual demand
- Asking the same indexing question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about Matplotlib's subplot system.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, machine learning diagnostics, and dashboard engineering:

- Reading and predicting the output of realistic multi-plot code snippets
- Calculating array shapes, tuple unpack structures, and slicing dimensions
- Identifying subplot geometry bugs, label overlap errors, and unpack exceptions
- Selecting the correct layout tools (`GridSpec`, `subplot_mosaic`, `subplots_adjust`) for complex scientific figures

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Grid Indexing Math: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Dashboard Engineering: ~30 questions

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
- All calculations (e.g., array dimensions, indexing slices, spacing ratios) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year were subplots introduced?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_grid=True`)
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
- Detail why each incorrect option fails (e.g., shape mismatch, unpack error, index out of bounds, or layout conflict)
- Include concrete references to Matplotlib mechanics (`squeeze` behavior, 0-based vs. 1-based indexing, `GridSpec` coordinate slices, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): Subplot Fundamentals (`plt.subplot` 1-based indexing vs. `plt.subplots` OO interface, basic syntax)
- `test-02.md` (Questions 11–20): Return Array Shapes & Unpack Mechanics (scalar, 1D, 2D arrays, `squeeze=False`, nested tuple unpacking)
- `test-03.md` (Questions 21–30): Grid Indexing & Looping Patterns (`axes[row, col]`, `axes.flat`, `axes.ravel()`, dynamic multi-plot loops)
- `test-04.md` (Questions 31–40): Shared Axes Mechanics (`sharex`, `sharey`, `'col'`, `'row'`, tick label suppression, zoom linking)
- `test-05.md` (Questions 41–50): Padding, Spacing & Alignment (`subplots_adjust`, `wspace`, `hspace`, `tight_layout`, `constrained_layout`)
- `test-06.md` (Questions 51–60): Managing Incomplete / Odd Grids (hiding axes with `axis('off')`, removing with `delaxes`, `set_visible(False)`)
- `test-07.md` (Questions 61–70): Asymmetric Grids with `GridSpec` (row/column spans, slice notation, `width_ratios`, `height_ratios`)
- `test-08.md` (Questions 71–80): Advanced Layouts with `subplot_mosaic` (visual layout strings, dictionary mapping, nested structures)
- `test-09.md` (Questions 81–90): Figure-Level Dashboard Annotations (`fig.suptitle`, `fig.supxlabel`, shared figure legends, global colorbars)
- `test-10.md` (Questions 91–100): End-to-End Machine Learning Dashboards (2x2 loss/acc monitoring, model diagnostic suites, EDA feature grids)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/subplots/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Matplotlib
8. Return array shapes and unpacking mechanics are 100% verified
9. Difference between 1-based indexing (`plt.subplot`) and 0-based indexing (`axes[i, j]`) is strictly preserved
10. `wspace` and `hspace` definitions and effects are correctly tested
11. `GridSpec` and `subplot_mosaic` slice syntax is error-free
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/subplots/` following this prompt.
