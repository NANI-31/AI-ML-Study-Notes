Generate 100 New Objective Questions from the Reference Line Plot Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Line Plots (Format Strings, Linestyles, Markers, Color Cycling, Multi-Series Trends, Layer Z-Order, Mathematical Curves, and Continuous vs. Discrete Visualization Ethics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/lineplot/`:

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

First, thoroughly analyze the supplied `3. plot.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic line plot syntax: `plt.plot(x, y)` / `ax.plot(x, y)`
  - Format string (`fmt`) shorthand notation: `'[color][marker][line]'`
    - Colors: `'b'` (blue), `'g'` (green), `'r'` (red), `'c'` (cyan), `'m'` (magenta), `'y'` (yellow), `'k'` (black), `'w'` (white)
    - Marker symbols: `'o'` (circle), `'s'` (square), `'^'` (triangle up), `'v'` (triangle down), `'*'` (star), `'+'` (plus), `'x'` (cross), `'D'` (diamond), `'.'` (point)
    - Line styles: `'-'` (solid), `'--'` (dashed), `'-.'` (dash-dot), `':'` (dotted), `''` or `'None'` (no line)
    - Decoding compact strings (e.g., `'ro-'` = red circle with solid line; `'k--'` = black dashed line; `'g*:'` = green star with dotted line)
  - Explicit keyword styling parameters:
    - Line properties: `linewidth` (or `lw`), `linestyle` (or `ls`), `color` (single string, Hex `#RRGGBB`, RGB tuple, CSS name)
    - Marker properties: `marker`, `markersize` (`ms`), `markerfacecolor` (`mfc`), `markeredgecolor` (`mec`), `markeredgewidth` (`mew`)
    - Transparency: `alpha` (from 0.0 invisible to 1.0 fully solid)
  - Execution order & Default Color Cycling:
    - How Matplotlib automatically assigns default colors from its color cycle (`C0`, `C1`, `C2`...) based strictly on **plotting execution sequence**, not the string text in `label`
    - Why swapping the order of `plt.plot()` calls swaps series colors
  - Layering & Z-Order (`zorder`):
    - Controlling draw hierarchy (e.g., ensuring key line traces stay above gridlines, error fills, or secondary lines)
  - Multi-series plotting techniques:
    - Multi-line in a single call: `plt.plot(x, y1, 'r-', x, y2, 'b--')`
    - Multi-line via separate calls with `label=` for automatic legend capture (`plt.legend(loc="upper left")`)
  - Plotting continuous mathematical curves:
    - Sampling with `np.linspace(start, stop, num)`
    - Sine, cosine, polynomials ($y = x^2, x^3$), sigmoid functions
    - Sampling density: why too few sample points in `np.linspace` causes visible jagged polygonal vertices rather than a smooth curve
  - Gridlines, limits & axes:
    - Enabling gridlines: `plt.grid(True, linestyle=':', alpha=0.6)`
    - Setting explicit view boundaries: `plt.xlim()`, `plt.ylim()`
    - Logarithmic scaling on lines: `plt.yscale('log')`
  - Visualization ethics & chart selection:
    - When a line chart is mathematically valid: continuous, sequential, or chronological data (time series, distance, continuous mathematical equations)
    - Why connecting discrete, unordered categories (e.g., region names: "North", "South", "East") with a line chart is an ethical / visual mistake (misleadingly implies continuous intermediate values and meaningful slope gradients; a bar chart is the appropriate choice)
- Important definitions:
  - Format string (`fmt`), line style, line width, marker, marker face, marker edge, color cycle (`prop_cycler`), z-order, linspace, continuous vs discrete domain
- Common mistakes and misconceptions:
  - Believing `linewidth` close to 10 makes a line invisible (in reality, higher values make lines thicker)
  - Believing labels determine series colors rather than plotting call order
  - Calling `plt.figure(figsize=...)` after `plt.plot()` instead of before it
  - Expecting `plt.show()` to accept a `figsize` parameter
  - Connecting discrete categories with lines, implying false continuity
  - Using too few points in `np.linspace()`, resulting in blocky or jagged curves
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new mathematical functions, business trends, sampling resolutions, format string puzzles, and visualization ethics scenarios.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different sales figures
- Reversing the answer choices
- Changing only variable names without altering the problem logic
- Rephrasing an existing question without changing what it tests
- Reusing the exact same format string repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about line plot mechanics.

Vary the question types:

- Decode compact format strings (e.g., what visual appearance does `'bs--'` or `'g^:'` produce?)
- Trace what visual attributes change when two `plt.plot()` calls are reordered in code
- Calculate the number of points generated by `np.linspace(start, stop, num)` and evaluate curve smoothness
- Diagnose why a line chart appears jagged and select the appropriate resolution adjustment
- Identify why a line plot fails to appear above a shaded region and fix it using `zorder`
- Explain why connecting discrete categories (like product categories or region names) with lines is bad practice
- Debug code snippets where `figsize` is applied too late or passed to the wrong function
- Evaluate the visual result of setting `markerfacecolor='white'` with `markeredgecolor='blue'`
- Trace multiple lines plotted inside a single `plt.plot()` call with mixed format strings
- Determine how `plt.legend(loc=...)` anchors to figure boundaries

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, format strings, or trend setups)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or format string if applicable, and clearly phrased question]

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

- Sets 01–03 (Questions 1–30): Basic line plot syntax, `fmt` shorthand strings, line width vs. opacity, marker shapes, labels, titles, gridlines, and figure creation sequence.
- Sets 04–07 (Questions 31–70): Multi-series comparisons, color cycling mechanics, legend positioning, marker face/edge styling, mathematical curve sampling resolution with `np.linspace`, and layer hierarchy with `zorder`.
- Sets 08–10 (Questions 71–100): Advanced multi-line plotting, logarithmic scale transformations, diagnosing continuous vs. discrete chart selection ethics, debugging complex multi-call scripts, and performance with large time-series.

Use a healthy mixture of:

- Conceptual questions
- Format string decoding questions (`'ro-'`, `'b--'`, `'k*:'`)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (interpreting learning curves, loss curves)
- Comparison questions (line chart vs bar chart, single call vs multiple calls)
- Edge-case questions (zero line width, empty line styles, single point lines)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, decode syntax strings, or reason through visualization design choices.

5. Coverage

Across all 100 questions, cover Matplotlib line plot concepts comprehensively:

- Core Syntax & Line Plotting:
  - `plt.plot(x, y)` / `ax.plot(x, y)`
  - What happens when only a single argument is passed: `plt.plot(y)` uses default integer indices `0, 1, ..., N-1` for $x$
  - The returned object: list of `matplotlib.lines.Line2D` objects
- Format String (`fmt`) Shorthand:
  - Structure: `'[color][marker][line]'`
  - Color codes: `'b'` (blue), `'g'` (green), `'r'` (red), `'c'` (cyan), `'m'` (magenta), `'y'` (yellow), `'k'` (black), `'w'` (white)
  - Marker codes: `'o'`, `'s'`, `'^'`, `'v'`, `'*'`, `'+'`, `'x'`, `'D'`, `'.'`, `'p'`, `'h'`
  - Line style codes: `'-'` (solid), `'--'` (dashed), `'-.'` (dash-dot), `':'` (dotted)
  - Order independence within format string: `'ro--'` is equivalent to `'r--o'` or `'--ro'`
  - Suppressing the line to show only markers: `'o'` or `linestyle='none'`
- Detailed Keyword Properties:
  - `linewidth`: float value in points (e.g., `linewidth=2.5`, higher is thicker)
  - `color`: hex codes (`'#2ca02c'`), standard names, RGB tuples
  - `alpha`: float in $[0.0, 1.0]$ controlling line opacity
  - Marker tuning: `markersize`, `markerfacecolor` (`mfc`), `markeredgecolor` (`mec`), `markeredgewidth` (`mew`)
- Execution Order & Color Cycle Mechanics:
  - The default color cycle (`tab10`: `C0`, `C1`, `C2`...)
  - Series colors are determined by the sequence of `plot()` invocations, NOT by the text passed to `label`
  - Reordering lines of code swaps the colors assigned to the series
  - Manually overriding colors to ensure consistency across charts
- Layering & Z-Order (`zorder`):
  - Default z-order values in Matplotlib (patches = 1, lines = 2, text = 3)
  - Overriding `zorder` (e.g., `zorder=5`) to ensure a benchmark line stays visible above filled areas or dense secondary plots
- Multi-Series Plotting:
  - Multi-series in a single call: `plt.plot(x, y1, 'r-', x, y2, 'b--')`
  - Multi-series via separate calls with `label=` argument
  - Legend positioning: `loc="upper left"`, `loc="lower right"`, `loc="best"`
- Continuous Curves with NumPy:
  - Using `np.linspace(start, stop, num)` to sample functions
  - Mathematical functions: `np.sin(x)`, `np.cos(x)`, polynomials, exponential growth
  - Resolution and interpolation artifacts: why low `num` (e.g., 5 points across $2\pi$) looks like an irregular polygon rather than a sine wave
- Proper Figure Lifecycle:
  - `plt.figure(figsize=(width, height), dpi=...)` must be called _before_ plotting
  - Calling `plt.figure()` after `plot()` opens a new blank figure
  - `plt.show()` does not accept `figsize`
  - Clearing figures: `plt.clf()`, `plt.close()`
- Visualization Ethics & Chart Selection:
  - Line charts inherently communicate continuous change, rate of change (slope), and interpolation between adjacent points
  - Line charts are mathematically and ethically appropriate for continuous or ordered domains (time, temperature, age, training epochs)
  - Connecting discrete, nominal, or unordered categories (e.g., department names, customer segments, product brands) with a line implies false continuity and non-existent trends; bar charts must be used instead

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as heatmaps, 3D surface meshes, or pie charts) merely to fill question count.

6. Numerical & Coordinate Questions

Include meaningful coordinate and numerical calculation questions across the sets.

For example:

- In `np.linspace(0, 10, 5)`:
  - What are the 5 exact coordinate values generated? ($[0.0, 2.5, 5.0, 7.5, 10.0]$)
  - What is the step size between adjacent points? ($2.5$)
- In `plt.figure(figsize=(10, 6), dpi=100)`:
  - What are the exact pixel dimensions of the resulting rendered image? ($1000 \times 600$ pixels)
- In a multi-line plot where Line 1 has `zorder=2` and Line 2 has `zorder=4`:
  - Which line is drawn on top when they intersect? (Line 2)
- Given `plt.plot([2, 4, 6])` (no $x$ array provided):
  - What are the automatic $x$-coordinates assigned? ($[0, 1, 2]$)
  - What is the coordinate of the second point? ($(1, 4)$)
- In `np.linspace(0, 2 * np.pi, 9)`:
  - What is the step size in radians between sample points? ($(2\pi - 0) / 8 = \pi / 4$)

Ensure all calculations use clean, friendly numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(9, 4))
plt.plot(x, y1, "ro-", label="Sine", linewidth=1.5, markersize=4)
plt.plot(x, y2, "b--", label="Cosine", linewidth=2.0)
plt.title("Trigonometric Waves", fontsize=14)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()
```

```python
# Color cycle order dependency
plt.plot(online_sales, label="online")
plt.plot(store_sales, label="store")
plt.legend()
```

```python
# Custom marker styling
plt.plot(epochs, loss, marker="s", mfc="white", mec="red", mew=2, color="red")
```

Questions should ask the learner to determine:

- What visual appearance is produced by a given format string or property bundle
- Which series receives the first default color (`C0`) when calls are reordered
- Which line contains a bug (e.g., placing `plt.figure()` after `plt.plot()`)
- Why a curve appears sharp and angular instead of smooth
- Which chart type (line vs bar) is ethically and technically appropriate for a stated dataset

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Confusing format string shorthand with keyword arguments
- Inconsistent coordinate definitions
- Overstating style behaviors (e.g., claiming tight layout "always" prevents overlap when not demonstrated)

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why plotting call order dictates color cycling, why `figsize` must precede plotting, or why discrete domains should not be connected by lines).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/lineplot/`.

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

# Matplotlib Line Plot Objective Test — Set 01

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
- [ ] Mathematical calculations (linspace intervals, pixel dimensions, z-order layering) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Line Plots, format strings, color cycling, mathematical curve sampling, and data visualization ethics.
