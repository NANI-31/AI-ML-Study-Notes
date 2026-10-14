Generate 100 New Objective Questions from the Reference Pie Chart Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebook and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Pie & Donut Charts (Wedge Geometry, Percentages & Autopct, Exploding Slices, Rotation & Starting Angles, Donut Chart Construction, Nested Rings, Multi-Panel Subplots, and Visual Perception Ethics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/piechart/`:

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

First, thoroughly analyze the supplied `9. pie.ipynb` and `test.md`.

Identify:

- Topics and subtopics covered:
  - Basic pie chart syntax: `plt.pie(values)` / `ax.pie(values)`
  - Labeling categories: `labels=['Train', 'Val', 'Test']`
  - Percentage computation and string formatting:
    - Format string notation: `autopct='%1.1f%%'`, `autopct='%1.2f%%'`, `autopct='%d%%'`
    - Dynamic formatting function: passing a custom callable `def make_autopct(values): ...` to display both percentage and raw absolute counts simultaneously
  - Slice emphasis via `explode`:
    - The `explode` tuple: specifying outward radial offset for each slice (e.g., `explode=(0.1, 0, 0)`)
    - Offset magnitude relative to circle radius
  - Orientation, direction, and starting angle:
    - Default starting angle ($0^\circ$ on positive x-axis)
    - Custom starting position: `startangle=90` (starting at 12 o'clock / positive y-axis)
    - Direction control: `counterclock=False` (rendering clockwise) vs. default counterclockwise
  - Donut Chart construction techniques:
    - Technique A (Modern `wedgeprops`): `wedgeprops=dict(width=0.4, edgecolor='white')`
    - Technique B (Center Circle Patch): creating `plt.Circle((0,0), 0.70, fc='white')` and adding with `ax.add_artist(circle)`
  - Nested / Multi-Ring Donut Charts:
    - Plotting hierarchical relationships (outer ring for high-level categories, inner ring for sub-categories)
    - Managing radius boundaries and wedge widths
  - Multi-Year / Multi-Group Subplot Comparisons:
    - Comparing class distributions side-by-side: `fig, axes = plt.subplots(1, 2)`
  - Aesthetic & edge customization:
    - Edge colors: `wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}`
    - Drop shadows: `shadow=True` (and why flat 2D is preferred over shadow distortion)
    - Aspect ratio: `ax.axis('equal')` ensuring circular rather than elliptical rendering
  - Normalization mechanics:
    - Automatic scaling: `plt.pie()` normalizes `sum(values)` to 100% if sum $> 1$
    - Partial pie phenomenon: if sum of values $< 1$, Matplotlib draws an incomplete pie sector
  - Visualization ethics & perceptual limitations:
    - Human visual perception: inability to judge subtle angle differences accurately compared to aligned linear bar charts
    - Category crowding: why pie charts should not exceed 5–7 slices (collapsing minor slices into "Other")
    - 3D pie charts: severe optical distortion exaggerating foreground slice area
- Important definitions:
  - Wedge, sector, autopct, explode, donut chart, startangle, counterclock, wedgeprops, central patch, perceptual angle distortion
- Common mistakes and misconceptions:
  - Believing values passed to `plt.pie()` must strictly sum to 100 (Matplotlib automatically normalizes)
  - Passing values summing to less than 1.0 expecting a full pie (creates an incomplete slice)
  - Swapping `startangle` behavior (starts counterclockwise from the positive x-axis by default)
  - Attempting to plot negative values in `plt.pie()` (raises `ValueError`)
  - Forgetting `ax.axis('equal')` in non-square figure sizes, resulting in squashed oval pies
  - Overusing pie charts when small, close values need precise comparative judgment
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, percentage calculations, explode offsets, donut configurations, and perceptual ethics scenarios.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Copying the same question with different class names
- Reversing the answer choices
- Changing only numerical slice percentages without altering the reasoning required
- Rephrasing an existing question without changing what it tests
- Reusing the exact same slice distribution repeatedly
- Asking the same concept in essentially the same way

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about pie chart mechanics.

Vary the question types:

- Calculate the exact angle (in degrees) occupied by a slice given raw category counts
- Compute the formatted string produced by `autopct='%1.1f%%'` for a given proportion
- Determine the visual starting position and direction when `startangle=90` and `counterclock=False` are set
- Trace how `wedgeprops={'width': 0.3}` converts a pie chart into a donut chart
- Diagnose why a pie chart renders as an elongated oval instead of a circle (`axis('equal')`)
- Identify which slice is emphasized based on an `explode` tuple
- Debug custom `autopct` functions formatting absolute counts alongside percentages
- Differentiate between hierarchical nested donut rings and side-by-side pie charts
- Evaluate whether a dataset with 15 categories should be visualized as a pie chart or a bar chart
- Analyze how partial pies render when passing fractional numbers summing to $< 1.0$

3. Maintain the Same Question Format

Follow the interactive structure and style of the study notes repository.

Each question should contain:

- A realistic scenario or context (often featuring code snippets, proportion tables, or visual setups)
- A clearly stated question
- Four answer choices: A, B, C, D (using blank `- [ ]` checkboxes)
- One or more correct answers
- An HTML `<details>` and `<summary>` collapsible block containing:
  - An explicit `**Correct Answer:** **...**` declaration
  - A clear `#### Explanations:` breakdown explaining every single option individually

Use this exact structure:

### N. Question

[Realistic scenario, code snippet or proportion table if applicable, and clearly phrased question]

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

- Sets 01–03 (Questions 1–30): Basic pie chart syntax, slice percentage calculations, labels, simple `autopct` formatting, titles, and `axis('equal')`.
- Sets 04–07 (Questions 31–70): Slice offset math (`explode`), orientation control (`startangle`, `counterclock`), custom callable `autopct` functions, donut chart construction via `wedgeprops` vs. `plt.Circle`, and edge styling.
- Sets 08–10 (Questions 71–100): Advanced hierarchical multi-ring nested donuts, multi-panel side-by-side subplot comparisons, edge cases (partial pies with sum $< 1$, zero-value slices), perceptual visualization ethics (angle vs. length perception), and debugging broken plot scripts.

Use a healthy mixture of:

- Conceptual questions
- Mathematical calculation questions (degrees of a wedge, percentage allocations, explode offsets)
- Code execution and debugging questions
- Diagnostic & scenario-based questions (choosing between pie and bar charts)
- Comparison questions (`wedgeprops` vs central circle patch, 2D pie vs 3D pie)
- Edge-case questions (negative values, sum $< 1$, 100% single slice)
- "Which statements are correct?" questions (multi-select)

Avoid making the entire set trivial recall-based questions. The learner should frequently need to evaluate code snippets, calculate sector angles, or analyze design trade-offs.

5. Coverage

Across all 100 questions, cover Matplotlib pie and donut chart concepts comprehensively:

- Core Syntax & Geometry:
  - `plt.pie(x)` / `ax.pie(x)`
  - Return tuple: `patches, texts, autotexts = ax.pie(...)`
    - `patches`: list of `matplotlib.patches.Wedge` objects
    - `texts`: list of category label text objects
    - `autotexts`: list of percentage numerical text objects (only present if `autopct` is provided)
  - Calculation of wedge sweep angle: $\theta_i = (v_i / \sum v) \times 360^\circ$
  - Equal aspect ratio: `ax.axis('equal')` or `plt.axis('equal')` to prevent oval squashing
- Percentage Representation & `autopct`:
  - Format strings: `'%1.1f%%'`, `'%1.0f%%'`, `'%.2f'`
  - Escaping the percent sign: `'%%'` in Python string formatting
  - Custom functions: `lambda pct: f"{pct:.1f}%"`
  - Formatting compound labels: displaying both raw values and percentages using custom functions
  - Styling `autotexts`: changing font color, size, and weight for visibility on dark wedges
- Slicing, Highlighting & Orientation:
  - `explode` parameter: tuple matching length of values specifying outward fractional offset (e.g., `(0, 0.1, 0)`)
  - Direction and starting angle:
    - Default: starts on the positive x-axis ($0^\circ$) and proceeds counterclockwise
    - `startangle=90`: rotates start to top ($12$ o'clock)
    - `counterclock=False`: changes progression to clockwise
  - Slices order: wedges are drawn in the exact sequence of the input list
- Donut Chart Construction & Architecture:
  - Modern implementation: `wedgeprops=dict(width=0.3, edgecolor='white')`
  - Patch overlay method: adding `plt.Circle((0, 0), radius=0.7, color='white')` to the axes
  - Adding central summary metrics or text inside the donut hole (`ax.text(0, 0, "Total\n1,000", ha='center', va='center')`)
- Nested & Concentric Multi-Ring Charts:
  - Drawing multiple pie calls on the same axis with different `radius` arguments
  - Outer ring (`radius=1.0`, `width=0.3`) and inner ring (`radius=0.7`, `width=0.3`)
  - Matching sub-category wedge widths to parent category sectors
- Multi-Panel Side-by-Side Pie Comparisons:
  - Creating subplots: `fig, (ax1, ax2) = plt.subplots(1, 2)`
  - Comparing category distributions across segments or time periods (e.g., 2024 vs 2025)
- Normalization & Special Input Behavior:
  - Automatic normalization when $\sum v > 1$ or $\sum v \ne 100$
  - Partial pie behavior when $\sum v < 1.0$ (leaves an empty wedge gap for the missing fraction)
  - Behavior with zero values (draws a $0^\circ$ wedge, label may overlap)
  - Behavior with negative values (raises `ValueError`)
- Visualization Ethics & Data Communication:
  - Perceptual limitations: Weber's Law / Cleveland-McGill experiments showing humans make more errors judging angles and areas than linear positions
  - Slice volume threshold: never exceeding 5–7 slices (grouping minor categories into "Other")
  - Misleading 3D pie charts: perspective tilt artificially inflates the visual area of the front-facing slice

Only include concepts that are supported by or reasonably derived from the reference material. Do not introduce unrelated chart types (such as heatmaps, 3D meshes, or scatter plots) merely to fill question count.

6. Numerical & Geometric Questions

Include meaningful coordinate and geometric calculation questions across the sets.

For example:

- Given category counts $[30, 45, 25]$:
  - Total count: $30 + 45 + 25 = 100$
  - What is the sweep angle in degrees of the first slice ($30$)? ($30 / 100 \times 360^\circ = 108^\circ$)
  - What is the sweep angle of the second slice ($45$)? ($45 / 100 \times 360^\circ = 162^\circ$)
- In a pie chart with `startangle=90` and `counterclock=False`:
  - If the first slice has angle $90^\circ$, what clock position does its trailing edge end at? ($90^\circ - 90^\circ = 0^\circ$, i.e., $3$ o'clock)
- In a donut chart with `radius=1.0` and `wedgeprops={'width': 0.35}`:
  - What is the radius of the inner empty hole? ($1.0 - 0.35 = 0.65$)
- Given raw values $[0.2, 0.3, 0.1]$:
  - What is their sum? ($0.6$)
  - What fraction of the full $360^\circ$ circle will remain empty? ($1.0 - 0.6 = 0.4$, i.e., $144^\circ$ empty gap)
- In `autopct='%1.2f%%'`, what is the rendered string for a fraction of $0.12345$? (`"12.35%"`)

Ensure all calculations use clean, friendly numbers so learners can verify them easily.

7. Code-Based Questions

Include realistic Python and Matplotlib code snippets.

Use code snippets such as:

```python
import matplotlib.pyplot as plt

labels = ["Train", "Val", "Test"]
sizes = [70, 15, 15]
explode = (0.05, 0, 0)

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    explode=explode,
    startangle=90,
    wedgeprops=dict(edgecolor="black")
)
ax.axis("equal")
plt.show()
```

```python
# Donut chart with central text
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.0f%%", wedgeprops=dict(width=0.4, edgecolor="w"))
ax.text(0, 0, "Dataset\nSplit", ha="center", va="center", fontsize=12, fontweight="bold")
```

```python
# Custom absolute count + percentage formatter
def custom_fmt(val, allvals):
    total = sum(allvals)
    pct = val
    count = int(round(val * total / 100.0))
    return f"{pct:.1f}%\n({count})"
```

Questions should ask the learner to determine:

- What specific visual structure or output is produced
- Which line contains a bug (e.g., negative values in `sizes`, or passing mismatched tuple length in `explode`)
- Why a donut chart hole is created by setting `width` in `wedgeprops`
- What happens when `startangle` or `counterclock` parameters are modified
- Why a dataset with 12 small categories is considered poor practice for a pie chart

8. Correct Answer Distribution

Distribute correct answers reasonably across A, B, C, and D.

- Aim for an approximately balanced distribution across all 100 questions (roughly 25% each for A, B, C, D).
- For multi-select questions, vary the combinations (e.g., A & B, B & D, A & C, etc.).
- Avoid predictable patterns (e.g., avoid consecutive identical answers or alternating cycles).

9. Avoid Ambiguity

Every question must have an unambiguous, technically defensible answer according to standard Matplotlib behavior.

Avoid:

- Vague visual descriptions without code context
- Ambiguity about angles without stating coordinate axes ($0^\circ$ at positive x-axis)
- Subjective color aesthetics unless testing contrast principles
- Conflating 2D pie charts with perspective-distorted 3D software charts

10. Explanations

Explanations are critical to learning.

For every single answer choice (A, B, C, D):

- Explain clearly why it is correct or incorrect.
- Never write shallow dismissals such as "Option A is wrong."
- Directly explain the mechanics (e.g., why angles are calculated by $(v/\sum v) \times 360^\circ$, how `explode` offsets radial coordinates, or why human perception favors bar charts over angles).
- Keep explanations clear, rigorous, and educational.

11. File Organization

Create exactly 10 Markdown files in `basics/3. Mathplotlib/piechart/`.

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

# Matplotlib Pie Chart Objective Test — Set 01

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
- [ ] Mathematical calculations (wedge angles, percentages, donut widths, explode offsets) are verified and accurate.
- [ ] Matplotlib and Python code snippets are syntax-valid and technically correct.
- [ ] No contradictory explanations or ambiguous choices exist.
- [ ] Markdown and HTML formatting are valid and clean throughout.

Most Important Requirement

Do not optimize for simply producing 100 questions. Optimize for producing 100 high-quality, non-repetitive, conceptually rigorous questions that test deep understanding and practical application of Matplotlib Pie & Donut Charts, wedge calculations, formatting functions, and perceptual visualization ethics.
