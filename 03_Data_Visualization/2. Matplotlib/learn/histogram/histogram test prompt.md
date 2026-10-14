Generate 100 New Objective Questions from the Reference Histogram Material

Use the attached/reference Jupyter Notebooks and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Histograms (Binning Strategies, Density Normalization, Cumulative Distributions, Multi-Distribution Overlays, Statistical Skewness, Reference Lines, and ML Data Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/histogram/`:

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

First, thoroughly analyze the supplied `7. hist.ipynb`, `7.1.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic histogram creation: `plt.hist(data, bins=...)`
  - Return values of `plt.hist()`: `counts, bins, patches = plt.hist(...)` (where `len(counts) == N` and `len(bins) == N + 1`)
  - Bin specification mechanics:
    - Scalar integer bin count: `bins=10`, `bins=30`, `bins=50` (automatically divides `[min, max]` into equal-width bins)
    - Explicit sequence of bin edges: `bins=[0, 10, 20, 30, 40, 50]` (allows variable or custom-boundary bins)
    - Calculation of bin width: $\text{width} = (\text{max} - \text{min}) / \text{bins}$
    - Interval conventions: half-open intervals $[a, b)$ vs. last bin inclusive $[a, b]$
  - Density normalization (`density=True`):
    - Raw frequency counts vs. probability density
    - Total area under the histogram integrates to 1: $\sum (\text{density}_i \times \text{width}_i) = 1$
    - Why individual bar heights can exceed 1.0 when bin widths are fractional ($\text{width} < 1$)
  - Cumulative distributions (`cumulative=True`):
    - Visualizing the empirical cumulative distribution function (CDF)
    - Cumulative count vs. cumulative probability density
  - Multiple distribution comparisons:
    - Transparent overlaid histograms (`alpha=0.6`, `alpha=0.7`)
    - Grouped side-by-side histogram bars: passing a list of arrays `plt.hist([group_a, group_b], bins=...)`
    - Stacked histograms: `stacked=True` or `histtype='barstacked'`
    - Step histograms: `histtype='step'` and `histtype='stepfilled'`
  - Distribution shapes & diagnostics:
    - Normal / Gaussian distribution: symmetric bell curve (mean $\approx$ median)
    - Right-skewed distribution: log-normal, exponential (long tail to the right, mean > median)
    - Left-skewed distribution: tail extending to the left (mean < median)
    - Bimodal and multi-modal distributions: two distinct peaks indicating underlying sub-populations
    - Uniform distribution: approximately flat across the range
  - Statistical reference lines:
    - Adding vertical central tendency lines: `plt.axvline(np.mean(data), color='red', linestyle='--')`, `plt.axvline(np.median(data), color='green')`
  - Outlier and anomaly visualization:
    - Isolated bars far in the tail separated by empty bins
  - Binning trade-offs:
    - Too few bins (under-smoothing): collapses data into 1–2 bars, obscures skewness and bimodality
    - Too many bins (over-smoothing): produces noisy, jagged spikes reflecting sample noise rather than population density
  - Aesthetics & customization:
    - `edgecolor="black"`, `color`, `rwidth=0.9` (adding gaps between bars), `orientation='horizontal'`
  - Reproducibility:
    - `np.random.seed(42)` guaranteeing identical pseudo-random distributions across runs
- Important definitions:
  - Bins, bin edges, bin width, frequency count, probability density, cumulative frequency, skewness, bimodality, empirical CDF, patches
- Common mistakes and misconceptions:
  - Believing changing `bins` alters the underlying data array values
  - Confusing a bar chart (categorical x-axis) with a histogram (continuous quantitative x-axis cut into bins)
  - Assuming `density=True` means the sum of bar heights equals 1 (in reality, the total _area_ equals 1)
  - Expecting `len(counts) == len(bins)` (in reality, `bins` has 1 more element than `counts` because it contains edges)
  - Assuming setting `cumulative=True` automatically normalizes to 0–1 (it only normalizes if `density=True` is also passed)
  - Believing a random seed affects plotting rather than data generation
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new distributions, sample datasets, numerical bin calculations, code snippets, and diagnostic scenarios.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different random seed values
- Reversing the answer choices
- Changing only variable names without altering the problem logic
- Rephrasing an existing question without changing what it tests
- Reusing the exact same dataset array repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about histogram mechanics.

Vary the question types:

- Calculate exact bin edges and bin width given data range $[\min, \max]$ and integer `bins`
- Compute which bin a specific data point falls into given explicit bin boundaries
- Calculate the total area under a density histogram or verify density heights for given widths
- Determine the shapes and lengths of arrays returned by `counts, bins, patches = plt.hist(...)`
- Diagnose whether a distribution is right-skewed, left-skewed, or bimodal from histogram characteristics
- Identify why a multi-series histogram overlaps invisibly and how `alpha` or `histtype='step'` fixes it
- Debug code snippets that produce distorted or confusing histograms due to inappropriate bin settings
- Distinguish between frequency histograms, probability density histograms, and cumulative CDF histograms
- Analyze the effect of adding `plt.axvline(mean)` vs `plt.axvline(median)` on skewed distributions
- Evaluate how outliers distort bin spacing when `bins` is an integer count

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, data arrays, or distribution setups)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or distribution data if applicable, and clearly phrased question]

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

- Sets 01–03 (Questions 1–30): Basic histogram syntax, scalar bins vs. sequence bins, bin width math, labels, titles, gridlines, and seed reproducibility.
- Sets 04–07 (Questions 31–70): Return values `(counts, bins, patches)`, `density=True` area mathematics, cumulative histograms (`cumulative=True`), multi-distribution comparison (`alpha`, side-by-side arrays, stacked bars), and `histtype` variations.
- Sets 08–10 (Questions 71–100): Advanced ML data diagnostics (detecting skewness, bimodality, and extreme outliers), statistical line overlays (`axvline`), binning optimization (under-smoothing vs over-smoothing), handling large sample sizes, and debugging complex plotting pipelines.

Use a healthy mixture of:

- Conceptual questions
- Mathematical / calculation questions (bin widths, bin counts, density areas, cumulative sums)
- Code execution and debugging questions
- Distribution interpretation questions (skewness, modality, tails)
- Comparison questions (`density=True` vs raw count, histogram vs bar chart, `step` vs `bar`)
- Edge-case questions (single-value datasets, bin width < 1 with density > 1, outliers creating empty stretches)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate bin limits, or analyze distribution profiles.

5. Coverage

Across all 100 questions, cover Matplotlib histogram concepts comprehensively:

- Fundamentals of `plt.hist()` / `ax.hist()`:
  - Mandatory input: 1D array-like data of continuous/discrete numeric values
  - Differences between histograms (continuous data grouped into quantitative bins) and bar plots (discrete categorical variables)
  - The return tuple `counts, bins, patches = plt.hist(...)`:
    - `counts`: array of frequency counts or density values for each bin (length $N$)
    - `bins`: array of bin edge boundaries (length $N + 1$)
    - `patches`: list of `matplotlib.patches.Rectangle` or `Polygon` objects representing the drawn bars
- Binning Strategies & Coordinate Geometry:
  - Default bin count in Matplotlib (10 bins)
  - Setting integer bins: `bins=20`, `bins=50`
  - Setting explicit bin edge lists: `bins=[0, 10, 25, 50, 100]` (unequal bin widths)
  - Computing bin boundaries: $\text{bin\_width} = (\text{max} - \text{min}) / \text{num\_bins}$
  - Bin inclusion: left-closed $[e_i, e_{i+1})$ for all bins except the final bin which is fully closed $[e_k, e_{k+1}]$
  - Custom range clipping using the `range=(min_val, max_val)` parameter to ignore extreme outliers
- Normalization & Probability Densities:
  - `density=True`: transforms counts into probability density such that $\int f(x)dx = 1$
  - Discrete formula: $\sum_{i=1}^N (\text{count}_i / (\text{total\_samples} \times \text{width}_i)) \times \text{width}_i = 1$
  - Clarifying that individual bar heights can exceed 1.0 when bin width is smaller than 1.0
  - Comparing distributions with vastly different sample sizes (e.g., $N=10,000$ vs $N=500$) using `density=True`
- Cumulative Distribution Plots:
  - `cumulative=True`: each bin represents the count (or density) of all samples $\le$ that bin's upper edge
  - Combined `cumulative=True, density=True`: displays empirical CDF climbing monotonically from 0 to 1
  - Reversed cumulative distribution (`cumulative=-1`): survival function / exceedance probability
- Multi-Series & Comparative Histograms:
  - Overlaid transparent histograms: `plt.hist(a, alpha=0.5); plt.hist(b, alpha=0.5)`
  - Side-by-side grouped bars: passing a list of arrays `plt.hist([data1, data2], label=['A', 'B'])`
  - Stacked histograms: `stacked=True`
  - Alternate rendering styles: `histtype='step'` (unfilled outline), `histtype='stepfilled'`, `histtype='barstacked'`
- Distribution Shapes & Exploratory Data Analysis:
  - Normal / Gaussian: symmetric bell curve, mean $\approx$ median $\approx$ mode
  - Right-skewed (positive skew): long tail extending to the right, mean > median (e.g., income, house prices, order sizes)
  - Left-skewed (negative skew): long tail extending to the left, mean < median
  - Bimodal: two distinct peaks indicating mixed sub-populations (e.g., customer segments)
  - Uniform: roughly flat across the range
  - Visualizing outliers: isolated bars with wide empty zero-count gaps
- Reference Annotations & Central Tendency:
  - Adding mean and median markers: `plt.axvline(data.mean(), color='red', linestyle='dashed', label='Mean')`
  - Adding standard deviation bands ($\mu \pm \sigma$)
  - Using `plt.legend()` to identify distributions and reference lines
- Formatting & Display Parameters:
  - `edgecolor="black"`: cleanly separating adjacent bars
  - `rwidth=0.85`: introducing relative spacing between bars
  - `color`: single color or list of colors for multi-series
  - `orientation='horizontal'`: drawing horizontal histogram bars

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated topics (such as 3D plots, pie charts, or advanced image filtering) merely to fill question count.

6. Numerical & Coordinate Questions

Include meaningful coordinate and numerical calculation questions across the sets.

For example:

- Given a dataset with $\min = 20$ and $\max = 120$, if `bins = 5`, calculate:
  - The bin width: $(120 - 20) / 5 = 20$
  - The 6 bin edge coordinates: $[20, 40, 60, 80, 100, 120]$
  - Which bin a data point with value $65$ lands in: bin $[60, 80)$ (the 3rd bin)
- Given explicit bin edges `bins = [0, 5, 15, 30]`:
  - What are the widths of the 3 bins? ($5, 10, 15$)
- Given counts $[10, 25, 15]$ for 3 bins of width 2:
  - Total samples: $10 + 25 + 15 = 50$
  - Calculate the normalized density height of the 2nd bin:
    $$\text{density} = \frac{\text{count}}{\text{total} \times \text{width}} = \frac{25}{50 \times 2} = \frac{25}{100} = 0.25$$
- For cumulative counts given raw counts $[4, 6, 12, 8]$:
  - What are the cumulative counts? ($[4, 10, 22, 30]$)
- If `plt.hist()` is called with `bins = 25`:
  - What is the exact length of the returned `counts` array? ($25$)
  - What is the exact length of the returned `bins` array? ($26$)

Ensure all calculations use small, clean numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(loc=50, scale=10, size=1000)

counts, bins, patches = plt.hist(data, bins=20, density=True, edgecolor="black")
mean_val = np.mean(data)
plt.axvline(mean_val, color="red", linestyle="--", label="Mean")
plt.legend()
```

```python
# Multi-distribution comparison
group_a = np.random.normal(50, 10, 500)
group_b = np.random.normal(70, 10, 500)

plt.hist([group_a, group_b], bins=15, label=["Control", "Treatment"])
plt.legend()
```

```python
# Cumulative empirical CDF
plt.hist(order_values, bins=30, density=True, cumulative=True, histtype="step")
```

Questions should ask the learner to determine:

- What specific output or array shape is produced
- Which line contains a bug (e.g., passing a 2D matrix incorrectly or misinterpreting return values)
- What visual difference occurs when switching between `histtype='bar'` and `histtype='step'`
- Why density bar heights can exceed 1.0
- Which distribution skewness is indicated by `mean > median`
- How to ensure identical random values are generated across test runs

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Ambiguity about whether bins are left-closed or right-closed
- Conflating sample statistics (sample mean) with population parameters ($\mu$)
- Unspecified default parameters (clearly note default behavior where relevant)

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the underlying statistical or graphical mechanics (e.g., why area equals 1 under density normalization, why bins has length $N+1$, or how skewness affects mean vs. median position).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/histogram/`.

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

# Matplotlib Histogram Objective Test — Set 01

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
- [ ] Mathematical calculations (bin edges, bin widths, density values, cumulative counts) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Histograms, binning mathematics, density normalization, and distribution diagnostics.
