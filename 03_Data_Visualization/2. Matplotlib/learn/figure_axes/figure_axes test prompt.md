Generate 100 New Objective Questions from the Reference Figure & Axes Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Figure & Axes Architecture (Figure Hierarchy, Canvas vs. Axes, Sizing & DPI Mathematics, Manual `add_axes` Insets, Facecolor Customization, `suptitle` vs. `set_title`, Layout Engines like `tight_layout`, Export Optimization via `savefig`, and Production Memory Lifecycle).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/figure_axes/`:

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

First, thoroughly analyze the supplied `1. Figure.ipynb`, `1. basic.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Matplotlib Object Hierarchy: `Figure` (top-level container), `Canvas` (backend rendering surface), `Axes` (the actual plot coordinate space), and `Axis` (ticks and spines)
  - Figure initialization and dimensions: `plt.figure(figsize=(w, h), dpi=...)`
  - DPI and Pixel calculations: $\text{pixels} = \text{inches} \times \text{DPI}$ (e.g., $(5, 3)$ at $100\text{ DPI} = 500 \times 300\text{ px}$; $(10, 6)$ at $300\text{ DPI} = 3000 \times 1800\text{ px}$)
  - DPI methods: `fig.get_dpi()`, `fig.set_dpi(...)`
  - Facecolor and Edgecolor styling:
    - Figure background: `fig.set_facecolor("color")`, `fig.patch.set_facecolor(...)`
    - Axes plotting area background: `ax.set_facecolor("color")`, `ax.patch.set_facecolor(...)`
    - Visual hierarchy distinction: outer canvas color vs. inner plot region color
  - Title hierarchy and positioning:
    - Global super title: `fig.suptitle("Title", fontsize=..., y=..., x=...)`
    - Individual Axes title: `ax.set_title("Subplot Title", loc="left"/"center"/"right")`
    - Title overlapping and vertical spacing adjustments
  - Layout Engines & Clipping Prevention:
    - `tight_layout=True` inside `plt.figure()`
    - Post-creation layout: `plt.tight_layout()` / `fig.tight_layout(pad=..., w_pad=..., h_pad=...)`
    - Bounding box adjustments: eliminating clipped labels, overlapping titles, and cropped tick markers
    - `constrained_layout=True` as an alternative automated geometry solver
  - Manual Axes Placement:
    - `ax = fig.add_axes([left, bottom, width, height])` with normalized coordinates $[0, 1]$
    - Inset plots: placing a secondary magnified or diagnostic plot inside a parent plot (e.g., `fig.add_axes([0.6, 0.6, 0.25, 0.25])`)
    - Non-grid arbitrary multi-panel positioning
  - Object-Oriented (OO) vs. Stateful Pyplot API:
    - Pyplot implicit state machine (`plt.plot()`, `plt.title()`, `plt.xlabel()`, `plt.gca()`, `plt.gcf()`)
    - Object-Oriented explicit paradigm (`fig, ax = plt.subplots()`, `ax.plot()`, `ax.set_title()`, `ax.set_xlabel()`)
    - Why OO design is required for multithreaded apps, libraries, and complex multi-axes dashboards
  - High-Resolution Figure Export:
    - `fig.savefig("path.png", dpi=300, bbox_inches='tight', transparent=True, facecolor=fig.get_facecolor(), pad_inches=0.1)`
    - Vector formats (`.pdf`, `.svg`, `.eps`) vs. Raster formats (`.png`, `.jpg`, `.tiff`)
    - Preventing white borders and truncated legends using `bbox_inches='tight'`
  - Memory Management & Lifecycle:
    - `plt.close(fig)` vs. `plt.close('all')`
    - Figure accumulation leak: why loops calling `plt.figure()` without `plt.close()` exhaust system RAM
    - Clearing objects: `fig.clf()` (clear figure) vs. `ax.cla()` (clear axes)
    - Interaction with GUI event loops and `plt.show()`
  - Machine Learning & Scientific Visualization Use Cases:
    - High-DPI model training curves export for academic publishing
    - Embedding an inset convergence plot into an overarching ROC/AUC evaluation curve
    - Theming dark-mode/light-mode dashboards using explicit `fig.set_facecolor` and `ax.set_facecolor`
- Important definitions:
  - Figure, Canvas, AxesSubplot, Bounding Box (Bbox), DPI, Aspect Ratio, Normalized Figure Coordinates, Spines, Artist tree
- Common mistakes and misconceptions:
  - Confusing `fig.suptitle()` with `ax.set_title()`, causing overlapping text
  - Confusing `Axes` (plotting surface with coordinate system) with `Axis` (x or y ruler containing ticks and labels)
  - Passing pixel values directly to `figsize` instead of inches
  - Omitting `bbox_inches='tight'` during `savefig()`, leading to truncated axis labels in saved files
  - Using `add_axes` coordinates $> 1.0$ or $< 0.0$ and wondering why the plot renders outside the visible canvas
  - Creating figures in batch loops (e.g., 1,000 evaluation plots) without `plt.close(fig)`, causing Out-Of-Memory (OOM) crashes
  - Believing `fig.set_facecolor()` automatically changes the inner plot area background (which requires `ax.set_facecolor()`)
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing numerical coordinates from $[0.1, 0.1, 0.4, 0.8]$ to $[0.2, 0.2, 0.4, 0.8]$ without altering the analytical demand
- Asking the same syntax question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about Matplotlib's Figure and Axes system.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, machine learning diagnostics, and scientific visualization:

- Reading and predicting the output of realistic code snippets
- Calculating canvas pixel dimensions, aspect ratios, and normalized coordinate boundaries
- Identifying geometry bugs, clipped labels, and memory leaks
- Selecting the correct parameter configurations for production-grade exports and custom layouts

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Coordinate Math: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Production Best Practices: ~30 questions

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
- All calculations (e.g., DPI-to-pixel conversions, normalized bounding boxes) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year was Figure released?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_figure=True`)
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
- Detail why each incorrect option fails (e.g., invalid method, wrong coordinate space, incorrect pixel calculation, or memory leak)
- Include concrete references to Matplotlib mechanics (`Artist` hierarchy, normalized figure coordinates $[0, 1]$, raster vs. vector export, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): The Matplotlib Object Hierarchy (Figure, Canvas, Axes, Axis, Spines) & Pyplot vs. Object-Oriented paradigms
- `test-02.md` (Questions 11–20): Figure Dimensions & DPI Mathematics (inch-to-pixel conversions, `get_dpi()`, `set_dpi()`, aspect ratios)
- `test-03.md` (Questions 21–30): Color & Styling Hierarchy (`fig.set_facecolor`, `ax.set_facecolor`, patches, edges, transparency)
- `test-04.md` (Questions 31–40): Title Management (`fig.suptitle` vs. `ax.set_title`, alignment, coordinate offsets, multi-panel text hierarchy)
- `test-05.md` (Questions 41–50): Layout Optimization (`tight_layout=True`, `pad` parameters, `constrained_layout`, bounding box clipping)
- `test-06.md` (Questions 51–60): Manual Axes Placement with `add_axes` (normalized coordinate geometry, width/height bounds, non-grid designs)
- `test-07.md` (Questions 61–70): Inset Plots & Diagnostic Overlays (embedding zoom boxes, mini confusion matrices, ROC curve insets)
- `test-08.md` (Questions 71–80): High-Resolution Figure Export (`savefig`, `dpi`, `bbox_inches='tight'`, vector vs. raster formats, transparency)
- `test-09.md` (Questions 81–90): Memory Management & Object Lifecycle (`plt.close`, `fig.clf()`, `ax.cla()`, batch generation OOM prevention)
- `test-10.md` (Questions 91–100): Comprehensive Production Scenarios & Debugging (end-to-end multi-axes ML reporting pipelines)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/figure_axes/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Matplotlib
8. DPI and pixel coordinate mathematics are 100% verified
9. Distinction between `Figure`, `Axes`, and `Axis` is rigorously maintained
10. `fig.suptitle` and `ax.set_title` differences are clearly reflected
11. Normalized coordinate rules $[0, 1]$ for `add_axes` are respected
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/figure_axes/` following this prompt.
