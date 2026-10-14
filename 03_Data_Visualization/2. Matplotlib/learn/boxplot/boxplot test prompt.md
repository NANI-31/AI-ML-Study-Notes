Generate 100 New Objective Questions from the Reference Box Plot Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Box Plots (Five-Number Summary, IQR Mathematics, Whiskers, Outliers/Fliers, Notched Medians, Horizontal Orientations, Model Error Distributions, and Multi-Group Comparative Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/boxplot/`:

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

First, thoroughly analyze the supplied `11. boxplot.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic box plot syntax: `plt.boxplot(data)` / `ax.boxplot(data)`
  - The Five-Number Summary:
    - Minimum (non-outlier lower bound)
    - First Quartile ($Q_1$ / 25th percentile)
    - Median ($Q_2$ / 50th percentile)
    - Third Quartile ($Q_3$ / 75th percentile)
    - Maximum (non-outlier upper bound)
  - Interquartile Range (IQR) calculations:
    - $\text{IQR} = Q_3 - Q_1$ (middle 50% of the distribution)
  - Whisker boundary formulas:
    - Lower whisker limit: $\max(\min(x), Q_1 - 1.5 \times \text{IQR})$
    - Upper whisker limit: $\min(\max(x), Q_3 + 1.5 \times \text{IQR})$
  - Outlier (flier) detection and styling:
    - Identification of points lying outside the $1.5 \times \text{IQR}$ fences
    - Customizing fliers: `flierprops=dict(marker='o', color='red', markersize=6)`, `sym='r+'`
  - Notched box plots (`notch=True`):
    - Confidence interval around the median ($\approx \text{median} \pm 1.57 \times \text{IQR} / \sqrt{n}$)
    - Visual test: non-overlapping notches indicate statistically significant difference between medians at ~95% confidence
  - Box plot orientation:
    - Default vertical vs. horizontal orientation: `vert=False` or `orientation='horizontal'`
    - When horizontal is preferred (long group/category names, aligned with metric axes)
  - Customizing box components:
    - `patch_artist=True`: enabling filled colored boxes rather than line outlines
    - Customizing sub-elements: `boxprops`, `whiskerprops`, `capprops`, `medianprops`, `meanprops`
    - Displaying the mean: `showmeans=True`, `meanline=True`
  - Multi-group comparisons:
    - Passing a list of arrays / Series: `plt.boxplot([group_a, group_b, group_c], tick_labels=['A', 'B', 'C'])`
  - Machine Learning & Data Science scenarios:
    - Model error/residual distributions: comparing error spread of Linear Regression, Random Forest, and Neural Networks
    - Feature distribution across target classes: identifying separating features
    - Cross-validation variance: box plots of validation scores across 5-fold or 10-fold splits
    - Outlier sensitivity: why median and IQR are robust to extreme outliers compared to mean and standard deviation
- Important definitions:
  - Quartiles ($Q_1, Q_2, Q_3$), IQR, whiskers, fences, fliers/outliers, notch, patch artist, robust statistics
- Common mistakes and misconceptions:
  - Confusing the median line with the mean (box plots show median by default, not mean)
  - Believing whiskers always extend to the absolute minimum and maximum data values
  - Assuming outliers are excluded from quartile calculations (they are part of the dataset used to compute $Q_1, Q_2, Q_3$)
  - Calling `patch_artist=False` and trying to set `facecolor` (results in error or ignored styling)
  - Swapping $Q_1$ and $Q_3$ or confusing IQR with total range
  - Believing overlapping notches prove medians are identical (they indicate no statistically significant difference, not exact equality)
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new sample datasets, quartile values, whisker calculations, code snippets, and diagnostic scenarios.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different salary numbers
- Reversing the answer choices
- Changing only group names without altering the problem logic
- Rephrasing an existing question without changing what it tests
- Reusing the exact same dataset array repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about box plot mechanics.

Vary the question types:

- Calculate $Q_1$, median, $Q_3$, and $\text{IQR}$ given a small sorted dataset (e.g., 9 or 11 numbers)
- Compute the exact lower and upper whisker boundaries and identify which points count as outliers
- Trace what visual elements change when `notch=True` or `patch_artist=True` is toggled
- Diagnose model performance from a multi-model residual box plot (which model has lower variance, which has skew, which has extreme outliers)
- Identify why a box plot cannot be colored with `facecolor` (missing `patch_artist=True`)
- Differentiate between horizontal (`vert=False`) and vertical box plot layouts
- Evaluate whether two groups have significantly different medians based on notch overlap
- Debug code snippets passing nested lists or mismatched label arrays to `plt.boxplot()`
- Distinguish between box plot summaries and histogram frequency distributions
- Reason about why skewed distributions cause the median line to sit off-center within the box

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, data lists, or statistical summaries)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or data setup if applicable, and clearly phrased question]

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

If multiple answers are correct (multi-select), list all correct letters in the answer key (e.g., `**Correct Answer:** **A, D**`) and explain every option individually.

4. Question Difficulty

Keep the difficulty aligned with the reference material, but gradually increase the reasoning and depth required across the 10 files:

- Sets 01–03 (Questions 1–30): Basic box plot syntax, five-number summary definitions, median vs. mean, vertical vs. horizontal orientation, basic labels, and titles.
- Sets 04–07 (Questions 31–70): Manual IQR calculations, whisker fence math ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$), identifying outliers, `notch=True` confidence intervals, `patch_artist=True` color fills, and multi-group plotting.
- Sets 08–10 (Questions 71–100): Advanced ML model evaluation diagnostics (comparing regression model error spreads, identifying asymmetric skew from median position, evaluating cross-validation stability), and debugging complex multi-axis layouts.

Use a healthy mixture of:

- Conceptual questions
- Mathematical / statistical calculation questions (quartiles, IQR, whisker bounds, outlier tests)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (interpreting model error distributions)
- Comparison questions (box plot vs histogram, mean vs median, vertical vs horizontal)
- Edge-case questions (zero IQR, all identical values, extreme single-sided outliers)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate statistical bounds, or interpret distribution plots.

5. Coverage

Across all 100 questions, cover Matplotlib box plot concepts comprehensively:

- Fundamentals of `plt.boxplot()` / `ax.boxplot()`:
  - Input: 1D array, list of arrays, or 2D NumPy array/DataFrame
  - Default orientation (vertical)
  - Return structure: dictionary of Line2D artists (`boxes`, `medians`, `whiskers`, `caps`, `fliers`, `means`)
- The Five-Number Summary & Anatomy of a Box Plot:
  - Center line: Median ($Q_2$), dividing data into top 50% and bottom 50%
  - Box bottom: First Quartile ($Q_1$ / 25th percentile)
  - Box top: Third Quartile ($Q_3$ / 75th percentile)
  - Box height: $\text{IQR} = Q_3 - Q_1$
  - Whiskers: extend to the furthest non-outlier data points within $1.5 \times \text{IQR}$ of the quartiles
  - Fliers (outliers): points plotted individually beyond the whisker ends
- Whisker & Outlier Mathematical Rules:
  - Lower fence: $Q_1 - 1.5 \times \text{IQR}$
  - Upper fence: $Q_3 + 1.5 \times \text{IQR}$
  - Extreme outlier threshold: $3.0 \times \text{IQR}$
  - Why whiskers do NOT simply stop at $Q_1 - 1.5\text{IQR}$ if there is no data point there (they stop at the actual adjacent data value)
- Customization & Advanced Arguments:
  - `notch=True`: draws a V-notch around the median representing its confidence interval
  - `vert=False`: rotates to horizontal orientation (x-axis becomes the metric value)
  - `patch_artist=True`: replaces Line2D boxes with `PathPatch` artists that accept `facecolor`
  - `showmeans=True` and `meanline=True`: overlays the mean point or mean dashed line
  - `flierprops`: dictionary customizing outlier marker shape, size, and color
  - `tick_labels`: assigning custom names to multi-group boxes
- Multi-Distribution Comparisons:
  - Passing `[data1, data2, data3]` to display multiple distributions side-by-side on a single axis
  - Color-coding individual boxes by iterating over `box_plot['boxes']`
- Machine Learning & Statistical Diagnostics:
  - Comparing residual error distributions: a good model has a compact box centered at 0 with short whiskers
  - Skewness detection:
    - If median line is closer to $Q_1$ and upper whisker is longer $\to$ right-skewed (positive skew)
    - If median line is closer to $Q_3$ and lower whisker is longer $\to$ left-skewed (negative skew)
  - Variance comparison across models: wider box indicates higher variance/instability
  - Feature inspection: detecting class separability by inspecting box overlap across classes

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as heatmaps, 3D plots, or pie charts) merely to fill question count.

6. Numerical & Statistical Questions

Include meaningful calculation questions across the sets.

For example:

- Given the sorted dataset $[10, 12, 14, 16, 18, 20, 22, 24, 26]$:
  - What is the median ($Q_2$)? ($18$)
  - What is $Q_1$? ($13$ or $14$ depending on convention, clearly specify formula)
  - If $Q_1 = 14$ and $Q_3 = 22$, calculate:
    - $\text{IQR} = 22 - 14 = 8$
    - Lower fence: $14 - 1.5 \times 8 = 14 - 12 = 2$
    - Upper fence: $22 + 1.5 \times 8 = 22 + 12 = 34$
    - If an extra point with value $38$ is added, does it count as an outlier? (Yes, $38 > 34$)
- In a box plot where $Q_1 = 50$, $Q_3 = 70$:
  - What is the maximum value a data point can have without being plotted as an individual flier? ($70 + 1.5 \times 20 = 100$)
- In a dataset where all values are identical (e.g., $[25, 25, 25, 25]$):
  - What is the $\text{IQR}$? ($0$)
  - How does the box plot render? (A flat line at $25$ with no box or whiskers)

Ensure all calculations use small, clean numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

model_a_errors = [-3, -1, 0, 1, 2, 4]
model_b_errors = [-1, 0, 0, 1, 1, 2]

fig, ax = plt.subplots()
bp = ax.boxplot(
    [model_a_errors, model_b_errors],
    tick_labels=["Model A", "Model B"],
    patch_artist=True,
    notch=True
)
bp["boxes"][0].set_facecolor("lightblue")
bp["boxes"][1].set_facecolor("lightgreen")
plt.axhline(0, color="red", linestyle="--")
plt.show()
```

```python
# Horizontal box plot
fig, ax = plt.subplots()
ax.boxplot(salaries, vert=False, flierprops=dict(marker="x", markeredgecolor="red"))
ax.set_xlabel("Salary ($)")
```

Questions should ask the learner to determine:

- What specific output, element, or coordinate bound is produced
- Which line contains a bug (e.g., trying to set `facecolor` without `patch_artist=True`)
- How to interpret overlapping vs. non-overlapping notches between two model distributions
- Which dictionary key in the returned boxplot structure controls the outlier markers (`'fliers'`)
- Why horizontal orientation is specified with `vert=False`

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Unspecified quartile interpolation methods when numbers are borderline
- Confusing sample outliers with distribution boundaries
- Conflating median notch confidence with hypothesis test p-values

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why $1.5 \times \text{IQR}$ defines the fence, why `patch_artist=True` is required for fill colors, or how skewness shifts median line position).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/boxplot/`.

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

# Matplotlib Box Plot Objective Test — Set 01

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
- [ ] Mathematical calculations (quartiles, IQR, fences, outlier bounds) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Box Plots, quartile calculations, outlier detection, notched comparisons, and model error evaluation.
