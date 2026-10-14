Generate 100 New Objective Questions from the Reference Bar Plot Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the entire reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Bar Plots (Vertical, Horizontal, Grouped, Stacked, Error Bars, Annotations, and ML Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/barplot/`:

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

First, thoroughly analyze the supplied `6.bar.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Vertical bar charts: `plt.bar(x, height, width=...)`
  - Horizontal bar charts: `plt.barh(y, width, height=...)`
  - Grouped / Clustered bar charts with numeric offsets (`np.arange(len(categories))`, `x - width/2`, `x + width/2`)
  - Multi-group clustered bars (3+ series: `x - width`, `x`, `x + width`)
  - Custom category tick labeling: `plt.xticks(x, categories)` and `plt.yticks(y, categories)`
  - Stacked bar charts: `bottom` parameter for vertical bars, `left` parameter for horizontal bars
  - Multi-layer stacked bars: chaining bottoms using cumulative sums (`bottom=np.array(s1) + np.array(s2)`)
  - Normalized 100% stacked percentage bar charts
  - Error bars on bar plots: `yerr`, `xerr`, `capsize`, `ecolor`
  - Bar container objects and automated labeling: `bars = ax.bar(...)`, `ax.bar_label(bars, fmt="%.2f", padding=3)`
  - Manual text annotations on bars: `bar.get_height()`, `bar.get_x()`, `bar.get_width()`, `ax.text()`
  - Aesthetics & Customization: `color`, list of colors per bar, `edgecolor`, `linewidth`, `alpha`, `plt.grid(axis='y', linestyle='--', alpha=0.7)`
  - Axis limits and scaling: `plt.ylim(0, 1)`, setting proper baselines
  - Data visualization ethics: why non-zero y-axis baselines distort comparisons, selecting honest ranges
  - Machine learning practical scenarios:
    - Model comparison across metrics (Accuracy, Precision, Recall, F1 for Logistic Regression, Random Forest, SVM, XGBoost)
    - Train vs. Validation performance bars (visualizing overfitting/generalization gaps)
    - Class imbalance bar charts (imbalanced target distribution in classification datasets)
    - Feature importance rankings (sorted horizontal bar charts)
- Important definitions:
  - Bar height vs. bar width, bar containers, patch artists, offsets, baseline/bottom, capsize, fliers
- Common mistakes and misconceptions:
  - Passing DataFrame directly to `plt.bar()` expecting auto-grouping without offsets
  - Swapping `width` and `height` between `plt.bar()` and `plt.barh()`
  - Forgetting the `bottom` argument in stacked bars, resulting in overlapping occluded bars
  - Calculating symmetric offsets incorrectly (e.g., offsetting by `width` instead of `width/2` for 2 series)
  - Forgetting to map numeric tick positions back to category strings with `plt.xticks(x, labels)`
  - Truncating the y-axis, making a 2% model accuracy gap appear like a 50% performance difference
  - Calling `ax.bar_label()` on raw lists instead of the `BarContainer` returned by `ax.bar()`
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different model names
- Reversing the answer choices
- Changing only category labels (e.g., replacing "Cat, Dog" with "Apple, Banana" without changing the question)
- Rephrasing an existing question without changing what it tests
- Reusing the exact same offset calculation repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about bar plot mechanics.

Vary the question types:

- Calculate the exact center coordinates for grouped bars given an integer index and bar width
- Compute the cumulative `bottom` heights needed for a 3-layer stacked bar chart
- Determine the exact value of `ax.bar_label()` formatted with `fmt="%.1f"` or `fmt="%.2f"`
- Diagnose why a stacked bar chart displays only the last plotted series (missing `bottom=...`)
- Identify why numeric labels (0, 1, 2) appear on the x-axis instead of model names
- Differentiate when to use horizontal bar charts (`barh`) over vertical ones (e.g., long feature names)
- Debug code snippets that attempt grouped bars with mismatched array lengths
- Trace how `capsize` and `yerr` affect the visual error boundaries on top of bar heights
- Analyze the ethical implications of non-zero y-axis limits in reporting model benchmarks
- Determine the correct sequence of calls to sort feature importances before plotting horizontal bars

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, numerical setups, or metric tables)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or data table if applicable, and clearly phrased question]

- [ ] **A.** [Option text]
- [ ] **B.** [Option text]
- [ ] **C.** [Option text]
- [ ] **D.** [Option text]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Explain clearly and specifically why this option is wrong, identifying the common misconception.
- **Option B (Correct):** Explain clearly why this option is correct, providing mathematical or Matplotlib API reasoning.
- **Option C (Incorrect):** Explain clearly why this option is wrong.
- **Option D (Incorrect):** Explain clearly why this option is wrong.

</details>

---

If multiple answers are correct (multi-select), list all correct letters in the answer key (e.g., `**Correct Answer:** **A, C**`) and explain every option individually.

4. Question Difficulty

Keep the difficulty aligned with the reference material, but gradually increase the reasoning and depth required across the 10 files:

- Sets 01–03 (Questions 1–30): Basic bar plot syntax, vertical vs. horizontal (`bar` vs `barh`), color styling, edge colors, line widths, axis labels, titles, gridlines, and baseline ethics (y-axis starting at 0).
- Sets 04–07 (Questions 31–70): Grouped bar offset math (`i - width/2`, 3-series offsets), tick positioning (`plt.xticks`), stacked bar mechanics (`bottom` argument), normalized 100% stacked bars, error bars (`yerr`, `capsize`), and container handling.
- Sets 08–10 (Questions 71–100): Advanced ML visualization dashboards (train vs. validation gap analysis, multi-metric grouped bars, sorted feature importance bars), `ax.bar_label` formatting, manual `ax.text` placement with `bar.get_height()`, debugging broken plot scripts, and layout optimization.

Use a healthy mixture of:

- Conceptual questions
- Mathematical / coordinate calculation questions (bar centers, stacked bottoms, percentage scaling)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (interpreting model benchmark charts)
- Comparison questions (`bar` vs `barh`, stacked vs grouped, manual offset vs Pandas `.plot(kind='bar')`)
- Edge-case questions (negative values in bar charts, zero heights, asymmetric error bars)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate coordinate positions, or analyze visual outputs.

5. Coverage

Across all 100 questions, cover Matplotlib bar plot concepts comprehensively:

- Basic Vertical Bar Charts (`plt.bar` / `ax.bar`):
  - Parameters: `x`, `height`, `width` (default 0.8), `bottom` (default 0), `align` (`'center'` vs `'edge'`)
  - Passing categorical strings vs. numeric coordinates
  - Styling: `color` (single string, RGB, Hex `#RRGGBB`, or list of colors per bar), `edgecolor`, `linewidth`, `alpha`
  - Gridlines: enabling y-axis only gridlines via `plt.grid(axis='y', linestyle='--', alpha=0.7)`
- Horizontal Bar Charts (`plt.barh` / `ax.barh`):
  - Parameters: `y`, `width` (represents bar length), `height` (represents bar thickness), `left` (baseline)
  - When to prefer horizontal over vertical: long category/feature names to avoid overlapping or 90° rotated labels
  - Inverting y-axis order (`ax.invert_yaxis()`) to display the highest value on top
- Grouped / Clustered Bar Charts:
  - Why categorical strings cannot be directly offset without numeric arrays (`x = np.arange(len(categories))`)
  - 2-group centering formula: `x - width/2` and `x + width/2`
  - 3-group centering formula: `x - width`, `x`, `x + width` (with bar `width` typically $\le 0.25$)
  - Symmetrical spacing and avoiding bar overlap
  - Relabeling the numeric x-axis: `plt.xticks(x, categories)`
  - Grouped horizontal bars: `y - height/2` and `y + height/2`, relabeling with `plt.yticks(y, categories)`
- Stacked Bar Charts:
  - Mechanism of the `bottom` argument: shifting the starting baseline of the next series
  - Multi-series stacking: `bottom=group1`, then `bottom=np.array(group1) + np.array(group2)`
  - Common failure mode: omitting `bottom`, which draws series over one another from zero
  - Stacked horizontal bars using the `left` argument: `ax.barh(..., left=group1)`
  - 100% Normalized stacked bars: computing percentage distribution per category so total bar height equals 100
- Error Bars & Statistical Variations:
  - `yerr` for vertical bars, `xerr` for horizontal bars
  - `capsize`: adding cross-ticks to the ends of error bars
  - `ecolor`: custom error bar coloring
  - Interpreting standard deviation or confidence interval overlays on model metric bars
- Direct Bar Annotations & Labels:
  - Automated labeling: `ax.bar_label(container, fmt="%.2f", padding=3, label_type='edge' | 'center')`
  - Manual text labeling: looping over bars with `for bar in bars:`
  - Geometry methods: `bar.get_height()`, `bar.get_x()`, `bar.get_width()`, `bar.get_y()`
  - Positioning text: `ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, ...)`
- Practical Machine Learning Scenarios:
  - Multi-model evaluation: comparing Accuracy, Precision, Recall, F1 for Logistic Regression, Random Forest, SVM, XGBoost
  - Generalization diagnostics: paired Training vs. Validation score bars to spot overfitting
  - Class distribution analysis: visualizing severe class imbalance in datasets
  - Feature importance ranking: sorting features by `importances = model.feature_importances_` and plotting sorted horizontal bars
- Visualization Ethics & Best Practices:
  - Truncating y-axis: why non-zero minimums exaggerate minuscule variations and deceive readers
  - When non-zero baselines are acceptable (or unacceptable) in bar plots
  - Appropriate bar width ratios (avoiding toothpick-thin bars or borderless merged blobs)

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as heatmaps, 3D plots, or pie charts) merely to fill question count.

6. Numerical & Coordinate Questions

Include meaningful coordinate and numerical calculation questions across the sets.

For example:

- Calculate the exact center positions for a category at index $i = 2$ with bar `width = 0.4` for 2 series:
  - Left bar: $2 - 0.4/2 = 1.8$
  - Right bar: $2 + 0.4/2 = 2.2$
- For 3 series with `width = 0.25` at index $i = 3$:
  - Left bar: $3 - 0.25 = 2.75$
  - Middle bar: $3.0$
  - Right bar: $3 + 0.25 = 3.25$
- In a stacked bar chart where Segment 1 has height 45, Segment 2 has height 30, and Segment 3 has height 15:
  - What is the `bottom` argument for Segment 3? ($45 + 30 = 75$)
  - What is the total top height of the bar? ($45 + 30 + 15 = 90$)
- In a 100% normalized stacked bar chart with raw counts [60, 40]:
  - Calculate the percentage heights: ($60\%$ and $40\%$)
- In a bar chart with bar height 85 and `yerr = 4.5`:
  - What are the lower and upper bounds of the error bar? ($80.5$ to $89.5$)

Ensure all calculations use small, clean numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

models = ["LR", "RF", "XGBoost"]
train_acc = [0.85, 0.98, 0.95]
val_acc = [0.83, 0.89, 0.92]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_acc, width, label="Train")
bars2 = ax.bar(x + width/2, val_acc, width, label="Val")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
```

```python
# Stacked bar chart
fig, ax = plt.subplots()
ax.bar(classes, group1, label="Group 1")
ax.bar(classes, group2, bottom=group1, label="Group 2")
```

```python
# Automated labeling
fig, ax = plt.subplots()
bars = ax.bar(models, scores)
ax.bar_label(bars, fmt="%.2f", padding=3)
```

Questions should ask the learner to determine:

- What specific coordinate or value is produced
- Which line contains a bug (e.g., passing categorical string array to arithmetic offset `models - width/2`)
- What happens if the `bottom` argument is omitted in stacked bars
- Why the x-axis displays `[0, 1, 2]` instead of the model names
- Which parameters create a horizontal bar chart with error caps
- What `bar.get_height()` returns inside an annotation loop

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Unspecified version quirks (adhere to standard modern Matplotlib 3.x API)
- Confusing bar coordinates with tick coordinates
- Ambiguous error bar interpretations

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why offsetting by the full width leaves a gap, why `bottom` is necessary, or why `barh` takes `height` for bar thickness).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/barplot/`.

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

# Matplotlib Bar Plot Objective Test — Set 01

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
- [ ] All questions are original and not verbatim copies of the reference material.
- [ ] Mathematical calculations (offsets, cumulative stack sums, error bounds) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Bar Plots, layout calculations, error bars, and model evaluation diagnostics.
