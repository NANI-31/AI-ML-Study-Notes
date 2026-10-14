Generate 100 New Objective Questions from the Reference Scatter Plot Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Scatter Plots (Multi-Variable Encodings, Colormaps, Marker Sizing, Overplotting Mitigation, Trendline Overlays, Point Annotations, and ML Pattern Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/scatter/`:

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

First, thoroughly analyze the supplied `8. scatter.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic scatter plot creation: `plt.scatter(x, y)`
  - Multi-dimensional data encoding:
    - 1st and 2nd dimensions: Horizontal ($x$) and Vertical ($y$) spatial coordinates
    - 3rd dimension: Marker size (`s`), scaling area of circles
    - 4th dimension: Marker color (`c`), mapped through colormaps
  - Sizing mechanics (`s`):
    - Points squared ($\text{pt}^2$), proportional to marker _area_, not diameter/radius
    - Scaling continuous metrics (e.g., population, transaction value) into appropriate marker sizes
  - Color mapping & Colormaps:
    - Passing numeric arrays to `c`
    - Colormap selection (`cmap='viridis'`, `'plasma'`, `'coolwarm'`, `'Blues'`)
    - Sequential vs. Diverging vs. Qualitative colormaps
    - Adding and configuring colorbars: `cbar = plt.colorbar(scatter_handle)`, `cbar.set_label(...)`
    - Normalization ranges: `vmin` and `vmax`
  - Handling overplotting:
    - Alpha transparency (`alpha=0.5`, `alpha=0.2`)
    - Marker edge styling: `edgecolors='black'`, `linewidths=0.5`
  - Trendline and regression overlays:
    - Computing slope and intercept with NumPy: `m, b = np.polyfit(x, y, 1)`
    - Overlaying best-fit line: `plt.plot(x, m * x + b, color='red', linestyle='--')`
  - Outlier identification & Point Annotations:
    - Highlighting outlier points with secondary `plt.scatter()` calls (distinct color and enlarged `s`)
    - Text annotations with arrows: `plt.annotate("Outlier", xy=(x_target, y_target), xytext=(x_text, y_text), arrowprops=dict(arrowstyle="->"))`
    - Iterative labeling of points: looping over arrays to label coordinates
  - Multi-class / Grouped scatter plots:
    - Plotting distinct categorical classes by looping over groups (`df.groupby('class')`) or separate `plt.scatter()` calls
    - Automatic color cycling and legend generation using `label`
  - Exploratory data analysis & statistical pattern recognition:
    - Positive linear correlation, negative linear correlation, zero correlation
    - Non-linear relationships (quadratic, logarithmic)
    - Heteroscedasticity (fan-shaped variance spread as $x$ increases)
    - Clustering and classification separation boundaries
- Important definitions:
  - Scatter plot, bubble chart, marker size, colormap (`cmap`), colorbar, `PathCollection`, alpha transparency, trendline, polynomial fit, annotation anchor (`xy`) vs text position (`xytext`), arrowprops
- Common mistakes and misconceptions:
  - Assuming `s` controls marker radius rather than area (doubling $s$ doubles area, not width)
  - Passing a single color string to `c` when attempting to map a continuous numeric variable
  - Forgetting to assign the return value of `plt.scatter()` to a variable when calling `plt.colorbar()`
  - Swapping `xy` (the point being pointed to) and `xytext` (the location of the annotation text) in `plt.annotate()`
  - Using diverging colormaps (e.g., `coolwarm`) for strictly positive monotonic data instead of sequential colormaps (e.g., `viridis`)
  - Believing high correlation on a scatter plot confirms a direct causal relationship
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new sample datasets, coordinate pairs, colormap configurations, best-fit calculations, and annotation diagnostics.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different feature names
- Reversing the answer choices
- Changing only numerical coordinates without altering the reasoning required
- Rephrasing an existing question without changing what it tests
- Reusing the exact same scatter dataset repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about scatter plot mechanics.

Vary the question types:

- Calculate the predicted $y$ on a trendline given $m$, $b$, and $x$ coordinates
- Compute the visual area ratio between two markers given their `s` values
- Determine how `vmin` and `vmax` affect the color assignment of specific data points
- Diagnose why a scatter plot appears as an impenetrable solid blob and select the appropriate fix (`alpha`, marker size)
- Trace the coordinates of an arrow drawn by `plt.annotate(..., xy=..., xytext=...)`
- Identify syntax errors in binding a colorbar to a scatter plot object
- Distinguish between sequential, diverging, and categorical colormaps for a given dataset
- Evaluate which plot settings highlight clusters or distinct classification labels
- Debug code snippets that mix up continuous color scaling with discrete categorical legends
- Analyze scatter patterns indicating heteroscedasticity, non-linearity, or bivariate outliers

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, coordinate tables, or statistical setups)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or coordinate data if applicable, and clearly phrased question]

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

- Sets 01–03 (Questions 1–30): Basic scatter plot syntax, $(x, y)$ coordinate mapping, simple marker sizing, edge colors, labels, titles, gridlines, and basic correlation interpretation.
- Sets 04–07 (Questions 31–70): Multi-variable encodings (size $s$ as area, color $c$ through colormaps), colorbar binding, `vmin`/`vmax` normalization, overplotting mitigation with `alpha`, trendline fitting with `np.polyfit`, and point highlighting.
- Sets 08–10 (Questions 71–100): Advanced annotations (`xy` vs `xytext`, `arrowprops`), grouped multi-class scatter plotting, diagnosing statistical anomalies (heteroscedasticity, non-linearity, multi-collinear clusters), and debugging complex multi-layer visualization scripts.

Use a healthy mixture of:

- Conceptual questions
- Mathematical / coordinate calculation questions (trendline evaluations, marker area ratios, colormap normalization math)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (interpreting residual scatter, identifying outliers)
- Comparison questions (`plot(..., 'o')` vs `scatter()`, sequential vs diverging colormaps)
- Edge-case questions (size $s \le 0$, single point scatter, all identical color values)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate coordinate positions, or analyze bivariate patterns.

5. Coverage

Across all 100 questions, cover Matplotlib scatter plot concepts comprehensively:

- Core Syntax & Mechanics:
  - `plt.scatter(x, y)` / `ax.scatter(x, y)`
  - Differences between `plt.plot(x, y, 'o')` (fast, single uniform color/size for all points) and `plt.scatter(x, y)` (flexible, individual size and color mapping per point)
  - The return object: `PathCollection` (used for colorbar references and legend handles)
- Multi-Dimensional Visual Encoding:
  - 2D: $(x, y)$ position on Cartesian coordinates
  - 3rd dimension: Marker size $s$ (measured in points squared $\text{pt}^2$):
    - Area scaling: doubling diameter requires multiplying $s$ by 4 ($s \propto r^2$)
    - Scaling formulas for continuous data: e.g., `s = (population / max_pop) * 500 + 20`
  - 4th dimension: Marker color $c$ (numeric array or list of colors):
    - Colormap integration: `cmap='viridis'`, `'plasma'`, `'magma'`, `'cividis'`, `'coolwarm'`
    - Adding colorbars: `cbar = plt.colorbar(scatter_obj, ax=ax)`
    - Range clamping: `vmin` and `vmax` controlling the minimum and maximum data values mapped to colormap extremes
- Managing Overplotting & Large Datasets:
  - Opacity adjustment: `alpha` parameter (from 0.0 fully transparent to 1.0 fully opaque)
  - How overlapping transparent points create darker density clusters
  - Edge styling: `edgecolors='black'`, `linewidths=0.5` to separate touching points
  - Reducing marker size for dense samples
- Trendlines & Best-Fit Curves:
  - 1st-degree linear regression fit: `m, b = np.polyfit(x, y, 1)`
  - Generating trendline points: `plt.plot(x, m * x + b, color='red', label='Trendline')`
  - Polynomial fits: 2nd degree curves `np.polyfit(x, y, 2)`
- Outlier Highlighting & Annotations:
  - Overlaying a distinct scatter call for outliers: `plt.scatter(outlier_x, outlier_y, color='red', s=120, edgecolors='black')`
  - Annotating specific points:
    - `ax.annotate(text, xy=(x_pt, y_pt), xytext=(x_txt, y_txt), arrowprops=dict(arrowstyle="->", color="black"))`
    - Understanding coordinate systems: `xy` is the point to mark, `xytext` is the location of the text label
  - Looping over data points to attach text labels: `for i, txt in enumerate(labels): ax.text(x[i], y[i], txt)`
- Multi-Class & Categorical Visualizations:
  - Looping through unique categories: `for cls in np.unique(classes): ax.scatter(x[classes == cls], y[classes == cls], label=cls)`
  - Generating discrete legends using `ax.legend()`
  - Marker shapes: `'o'` (circle), `'s'` (square), `'^'` (triangle), `'*'` (star), `'x'` (cross)
- Statistical & Machine Learning Diagnostics:
  - Positive linear correlation: points rise from bottom-left to top-right
  - Negative linear correlation: points fall from top-left to bottom-right
  - Zero correlation: random dispersion cloud
  - Non-linear relationships: U-shaped (quadratic) or exponential curves where linear regression underfits
  - Heteroscedasticity: variance of $y$ expands (or contracts) across values of $x$ (cone/funnel shape)
  - Classification decision boundaries: evaluating how well two classes separate in feature space

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as 3D surface meshes, pie charts, or bar charts) merely to fill question count.

6. Numerical & Coordinate Questions

Include meaningful coordinate and numerical calculation questions across the sets.

For example:

- Given a dataset where `np.polyfit(x, y, 1)` yields slope $m = 2.5$ and intercept $b = 10$:
  - What is the predicted value on the trendline for a point with $x = 8$? ($2.5 \times 8 + 10 = 30$)
- In a bubble chart, City A has population 100,000 and City B has population 400,000. If marker area is directly proportional to population:
  - If City A has marker size `s = 50`, what should be the marker size `s` for City B? ($50 \times (400,000 / 100,000) = 200$)
  - What is the ratio of their visual diameters? ($\sqrt{4} = 2$)
- In a scatter plot with colormap `vmin = 10` and `vmax = 50`:
  - What normalized position $[0.0, 1.0]$ on the colormap does a data value of $30$ map to? ($(30 - 10) / (50 - 10) = 20 / 40 = 0.5$, the exact midpoint)
  - A value of $5$ will be clamped to which colormap extreme? (Position $0.0$, the minimum color)
- In `plt.annotate("Peak", xy=(5, 80), xytext=(3, 95), arrowprops=dict(arrowstyle="->"))`:
  - Where does the arrow tip point? (Coordinates $(5, 80)$)
  - Where is the text "Peak" rendered? (Coordinates $(3, 95)$)

Ensure all calculations use clean, friendly numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

hours = np.array([1, 2, 3, 4, 5])
scores = np.array([50, 55, 65, 72, 80])

fig, ax = plt.subplots()
scatter = ax.scatter(hours, scores, c=scores, cmap="viridis", s=100)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Exam Score")

m, b = np.polyfit(hours, scores, 1)
ax.plot(hours, m * hours + b, color="red", linestyle="--", label="Fit")
ax.legend()
```

```python
# Outlier annotation
ax.annotate(
    "Outlier",
    xy=(6, 12),
    xytext=(5, 25),
    arrowprops=dict(arrowstyle="->", color="black")
)
```

```python
# Multi-class scatter
for cls, color in zip(["A", "B"], ["blue", "orange"]):
    mask = classes == cls
    ax.scatter(x[mask], y[mask], c=color, label=f"Class {cls}", alpha=0.7)
ax.legend()
```

Questions should ask the learner to determine:

- What specific output, color, or coordinate position is produced
- Which line contains a bug (e.g., passing `plt.colorbar()` without a `PathCollection` argument)
- Why points appear visually overlapping and which parameter (`alpha`) resolves it
- Which colormap type is best suited for temperature anomalies spanning negative and positive values
- What happens if `s` is assigned a negative number (raises `ValueError`)

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Confusing marker area ($s$) with marker radius ($r$)
- Stating that correlation implies causation
- Undefined variable dimensions in code examples

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why $s$ scales quadratically with radius, how `xy` anchors the arrow tip, or why diverging colormaps require a meaningful zero center).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/scatter/`.

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

# Matplotlib Scatter Plot Objective Test — Set 01

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
- [ ] Mathematical calculations (trendline values, area ratios, colormap normalizations) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Scatter Plots, multi-variable encodings, colormaps, regression trendlines, and bivariate diagnostics.
