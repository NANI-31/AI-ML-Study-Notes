Generate 100 New Objective Questions from the Reference Area Plot Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Area Plots & Filled Curves (`fill_between`, `fill_betweenx`, Stacked Area Charts, Uncertainty Envelopes, Conditional Shading, Cumulative Trends, and ML Diagnostic Curves).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/areaplot/`:

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

First, thoroughly analyze the supplied `10 area.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic area fills: `plt.fill_between(x, y)` / `ax.fill_between(x, y)`
    - Default baseline ($y_2 = 0$)
    - Combining line curves (`plt.plot`) with area shading (`plt.fill_between`)
  - Bounded fills between two custom curves: `plt.fill_between(x, y1, y2)`
    - Visualizing the envelope or gap between two lines
  - Uncertainty envelopes & confidence intervals:
    - Creating bands around model predictions: `lower = prediction - uncertainty`, `upper = prediction + uncertainty`
    - Shading $\pm 1\sigma$ or $\pm 2\sigma$ ($95\%$ confidence) error envelopes
  - Conditional shading via the `where` parameter:
    - Shading only where specific conditions hold: `where=(y >= 0)` vs. `where=(y < 0)`
    - Profit vs. Loss coloring (e.g., green for gains, red for losses)
    - The `interpolate=True` parameter: preventing missing triangle gaps where curves cross each other
  - Horizontal area fills:
    - `plt.fill_betweenx(y, x1, x2)`: filling horizontally between vertical curves
  - Stacked Area Charts (`plt.stackplot` / `ax.stackplot`):
    - Syntax: `plt.stackplot(x, series1, series2, series3, labels=...)`
    - Visualizing parts-of-a-whole composition over time (e.g., product sales, browser shares)
    - Baseline positioning (`baseline='zero'`, `'sym'`, `'wiggle'`)
  - Cumulative Area Plots:
    - Combining `np.cumsum()` with area fills to visualize total accumulated volume over time
  - Machine Learning & System Monitoring Scenarios:
    - Training vs. Validation loss: shading the generalization gap over training epochs
    - Infrastructure monitoring: stacked CPU, Memory, and Disk usage across time
    - Model prediction uncertainty in time-series forecasting (fan charts)
  - Aesthetic & display customization:
    - Opacity: `alpha=0.3`, `alpha=0.5`
    - Layering: using `zorder` to ensure boundary lines remain sharp on top of translucent fills
    - Color choices, edge colors, and hatch patterns
  - Interpretation pitfalls & visualization ethics:
    - In stacked area charts, every series above the bottom layer rests on a fluctuating baseline, making individual trends difficult to judge accurately
    - When small multiples or line charts should be used instead of stacked area charts
- Important definitions:
  - Shaded area, baseline, boundary curves, confidence band, uncertainty envelope, conditional fill (`where`), intersection interpolation, stacked area (`stackplot`), streamgraph
- Common mistakes and misconceptions:
  - Forgetting `interpolate=True` when using `where`, leading to unshaded gaps at intersection points
  - Assuming `fill_between` automatically plots the bounding lines (it only plots the filled polygon; explicit `plt.plot()` calls are needed for crisp outlines)
  - Swapping horizontal and vertical arguments (`fill_between` vs `fill_betweenx`)
  - Passing negative values in stacked area charts with default baselines (can cause visual clipping or overlapping)
  - Misinterpreting the height of an upper layer in a stacked area chart as its absolute value rather than its relative layer thickness
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new time-series, uncertainty bounds, cumulative calculations, conditional logic, and diagnostic scenarios.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different revenue figures
- Reversing the answer choices
- Changing only variable names without altering the problem logic
- Rephrasing an existing question without changing what it tests
- Reusing the exact same trigonometric equation repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about area plot mechanics.

Vary the question types:

- Calculate the total stacked height at a specific time step $x_0$ across multiple layers
- Compute the upper and lower confidence boundaries given prediction $\hat{y}$ and standard deviation $\sigma$
- Trace which sections of a curve receive shading when conditional `where` clauses are applied
- Explain why `interpolate=True` is necessary when curves cross
- Diagnose why a stacked area chart creates optical illusions regarding individual layer trends
- Differentiate between vertical fills (`fill_between`) and horizontal fills (`fill_betweenx`)
- Debug code snippets that produce distorted area fills due to mismatched array lengths or missing parameters
- Evaluate how `zorder` and `alpha` interact when layering multiple fills and trendlines
- Analyze ML loss curves with shaded generalization gaps to detect early signs of overfitting
- Determine the cumulative sum values at specific intervals when plotting cumulative area curves

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, coordinate arrays, or ML training metrics)
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

- Sets 01–03 (Questions 1–30): Basic `fill_between` syntax, default baseline ($y=0$), `alpha` opacity, bounding two curves ($y_1, y_2$), labels, and titles.
- Sets 04–07 (Questions 31–70): Uncertainty intervals ($\mu \pm \sigma$), conditional fills (`where`), intersection gap smoothing with `interpolate=True`, horizontal fills (`fill_betweenx`), and cumulative tracking (`np.cumsum`).
- Sets 08–10 (Questions 71–100): Stacked area charts (`plt.stackplot`), parts-of-a-whole composition, layer interpretation pitfalls, ML loss gap analysis, multi-layer `zorder` management, and debugging complex monitoring scripts.

Use a healthy mixture of:

- Conceptual questions
- Mathematical calculation questions (stacked cumulative heights, confidence boundaries, area approximations)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (interpreting training vs. validation loss gaps)
- Comparison questions (`fill_between` vs `stackplot`, line chart vs stacked area)
- Edge-case questions (curves crossing at zero, negative inputs in stackplot, zero width intervals)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate boundary values, or analyze graphical trade-offs.

5. Coverage

Across all 100 questions, cover Matplotlib area plot concepts comprehensively:

- Fundamentals of `plt.fill_between()` / `ax.fill_between()`:
  - Arguments: `x`, `y1`, `y2=0`, `where=None`, `interpolate=False`, `step=None`
  - Default behavior: fills the area between $y_1$ and the horizontal line $y_2 = 0$
  - Bounded fills between two arbitrary series: filling between lower curve $y_1$ and upper curve $y_2$
  - The return object: `PolyCollection`
- Uncertainty Envelopes & Confidence Bands:
  - Formulating boundaries:
    - $\text{lower} = \hat{y} - 1.96 \times \text{SE}$
    - $\text{upper} = \hat{y} + 1.96 \times \text{SE}$
  - Visualizing Gaussian process regression, linear regression prediction intervals, and time-series forecasts
  - Setting translucent opacity (`alpha=0.2` to `0.4`) so the central prediction line remains visible
- Conditional Fills with `where`:
  - Syntax: `where=(y >= threshold)`
  - Dual conditional fills:
    - `plt.fill_between(x, y, 0, where=(y >= 0), color='green', alpha=0.5, label='Profit')`
    - `plt.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.5, label='Loss')`
  - The `interpolate=True` parameter:
    - Why it is essential: without it, Matplotlib clips at data points, leaving white unshaded triangles at zero-crossings
    - How `interpolate=True` calculates exact intersection points along line segments
- Horizontal Fills (`fill_betweenx`):
  - Parameters: `y`, `x1`, `x2=0`, `where=None`
  - When to use: filling along the vertical axis (e.g., depth profiles, horizontal distribution spreads)
- Stacked Area Charts (`plt.stackplot`):
  - Parameters: `x`, `*args` (multiple 1D series), `labels=...`, `colors=...`, `baseline='zero'`
  - How layers stack: layer $k$ sits on top of $\sum_{j=1}^{k-1} \text{layer}_j$
  - Tracking parts-of-a-whole composition over a continuous time dimension
  - Alternative baselines: `'zero'` (default), `'sym'` (symmetric around zero / streamgraph), `'wiggle'`
- Cumulative Area Plots:
  - Computing running totals: `cumulative = np.cumsum(daily_metric)`
  - Visualizing total volume accumulation (e.g., total sales, total bandwidth consumed)
- Machine Learning & System Health Diagnostics:
  - Generalization gap: shading the area between Training Loss and Validation Loss over epochs
  - Identifying overfitting: when the gap between training and validation area expands dramatically
  - Server metrics: stacking CPU, Memory, and Disk usage to monitor capacity limits
- Interpretation Pitfalls & Perceptual Ethics:
  - The "fluctuating baseline" issue: only the bottom-most layer has a flat baseline; all upper layers inherit the curves of the layers below them, making slope comparisons misleading
  - When to replace stacked area charts with individual line charts or small multiples

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as pie charts, 3D meshes, or bar plots) merely to fill question count.

6. Numerical & Boundary Questions

Include meaningful coordinate and boundary calculation questions across the sets.

For example:

- In a regression model where predicted value $\hat{y} = 25.0$ and margin of error $\text{MoE} = 4.2$:
  - What are the lower and upper bounds of `plt.fill_between(x, lower, upper)`?
  - Lower: $25.0 - 4.2 = 20.8$
  - Upper: $25.0 + 4.2 = 29.2$
- In a stacked area chart with three products:
  - At month $t=3$, Product A has sales 30, Product B has sales 45, Product C has sales 25.
  - What is the bottom baseline for Product C? ($30 + 45 = 75$)
  - What is the top boundary coordinate of Product C? ($75 + 25 = 100$)
- In a daily metric array $[10, 15, 20, 25]$:
  - What are the values plotted by `plt.fill_between(days, np.cumsum(metric))`?
  - Cumulative array: $[10, 25, 45, 70]$
- If curve $y_1 = x$ and curve $y_2 = 6 - x$ intersect:
  - At what coordinate $x$ do they cross? ($x = 6 - x \implies 2x = 6 \implies x = 3$)
  - For $x \in [0, 5]$, where does `where=(y1 >= y2)` apply? (The interval $x \in [3, 5]$)

Ensure all calculations use clean, friendly numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
prediction = np.sin(x)
lower = prediction - 0.3
upper = prediction + 0.3

plt.plot(x, prediction, color="blue", label="Prediction")
plt.fill_between(x, lower, upper, color="blue", alpha=0.25, label="Confidence Band")
plt.legend()
plt.show()
```

```python
# Conditional profit/loss fill
plt.plot(months, balance, color="black", linewidth=1.5)
plt.fill_between(months, balance, 0, where=(balance >= 0), color="green", alpha=0.4, interpolate=True)
plt.fill_between(months, balance, 0, where=(balance < 0), color="red", alpha=0.4, interpolate=True)
```

```python
# Stacked area plot
plt.stackplot(time, cpu, memory, disk, labels=["CPU", "Memory", "Disk"], alpha=0.7)
plt.legend(loc="upper left")
```

Questions should ask the learner to determine:

- What specific visual output or shaded region is produced
- Which line contains a bug (e.g., omitting `interpolate=True` or passing wrong baseline arguments)
- Why an upper layer in `stackplot` appears wavy even when its own data values are constant
- How `alpha` prevents underlying gridlines or secondary curves from being completely occluded
- Which function (`fill_between` vs `fill_betweenx`) creates a horizontal shading along the y-axis

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Ambiguity about whether boundaries are inclusive or exclusive
- Confusing individual layer values with cumulative stacked heights
- Inconsistent coordinate definitions

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why `interpolate=True` eliminates jagged intersection gaps, how cumulative heights accumulate in `stackplot`, or why fluctuating baselines distort perception).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/areaplot/`.

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

# Matplotlib Area Plot Objective Test — Set 01

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
- [ ] Mathematical calculations (confidence intervals, cumulative sums, stack baselines) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Area Plots, filled curves, confidence bands, conditional shading, stacked composition, and diagnostic interpretation.
