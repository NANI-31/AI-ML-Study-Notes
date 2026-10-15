Generate 100 New Objective Questions from the Reference 3D Plots Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference file:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib 3D Data Visualization (`mplot3d` Toolkit, `projection="3d"` Axes Initialization, 3D Scatter with `depthshade`, Parametric 3D Trajectories, 3D Bar Charts `bar3d`, Meshgrid Surfaces `plot_surface` & Wireframes `plot_wireframe`, `rstride`/`cstride` Downsampling, Plane Contour Projections `zdir`, Camera Angles via `view_init`, and 3D Optimization Loss Surfaces).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/3d_plots/`:

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

First, thoroughly analyze the supplied `3d .ipynb`, `1. basic.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - 3D Projection Initialization and Setup:
    - Toolkit import: `from mpl_toolkits.mplot3d import Axes3D`
    - Subplot registration: `fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")`
    - Subplots factory syntax: `fig, ax = plt.subplots(subplot_kw={"projection": "3d"})`
    - 3D coordinate rulers: `ax.set_xlabel("X")`, `ax.set_ylabel("Y")`, `ax.set_zlabel("Z")`
    - 3D coordinate limits: `ax.set_xlim()`, `ax.set_ylim()`, `ax.set_zlim()`
  - 3D Scatter Plots (`ax.scatter`):
    - 3D coordinates $(x, y, z)$
    - Continuous scalar coloring: `c=values, cmap='viridis'`
    - Point sizing: `s=50`
    - Depth shading: `depthshade=True` (fades distant points for perspective depth) vs. `depthshade=False`
    - Adding 3D colorbars: `fig.colorbar(scatter, ax=ax, label="Magnitude")`
  - 3D Line Plots (`ax.plot`):
    - Parametric curves in 3D space: $x=\cos(t), y=\sin(t), z=t$ (spiral trajectories)
    - Optimization trajectory paths tracking gradient descent steps $(w_1, w_2, \text{Loss})$
  - 3D Bar Charts (`ax.bar3d`):
    - Position anchors: `x, y, z` coordinates of the bottom-front-left corner
    - Volume extents: `dx, dy, dz` (depth, width, and height of each rectangular bar)
    - Shading control: `shade=True/False`
    - 2D joint distribution histograms visualized as 3D bars
  - 3D Wireframe & Surface Plots:
    - Grid evaluation via `X, Y = np.meshgrid(x, y)` and $Z = f(X, Y)$
    - Wireframe lines: `ax.plot_wireframe(X, Y, Z, color='gray')`
    - Continuous shaded surfaces: `surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')`
    - Stride downsampling: `rstride=5, cstride=5` (row and column sampling frequency) to balance visual resolution and rendering performance
    - Adding colorbars to 3D surfaces: `fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)`
  - 3D Contour Projections:
    - Projecting 2D level curves onto bounding wall planes:
      `ax.contour(X, Y, Z, zdir='z', offset=min_z, cmap='coolwarm')`
      `ax.contour(X, Y, Z, zdir='x', offset=min_x, cmap='coolwarm')`
      `ax.contour(X, Y, Z, zdir='y', offset=max_y, cmap='coolwarm')`
  - Camera Viewpoint and Perspective Control:
    - View angle initialization: `ax.view_init(elev=30, azim=45, roll=0)`
    - Elevation (`elev`): angle in degrees above the $xy$-plane
    - Azimuth (`azim`): angle in degrees rotating counter-clockwise around the $z$-axis
    - Top-down birds-eye view: `elev=90, azim=-90`
    - Side elevation view: `elev=0, azim=0`
  - Machine Learning & Scientific Visualization Use Cases:
    - 3D Principal Component Analysis (PCA) cluster visualization
    - Optimization Loss Landscapes: convex quadratic bowls, non-convex multi-modal surfaces, saddle points
    - Gradient descent optimization paths traversing 3D surfaces
    - Hyperparameter response surfaces (Learning Rate vs. Batch Size vs. Validation F1)
- Important definitions:
  - Axes3D, projection="3d", depthshade, bar3d, plot_wireframe, plot_surface, rstride, cstride, zdir, view_init, elev, azim
- Common mistakes and misconceptions:
  - Calling `ax.scatter(x, y)` without $z$ coordinates on a 3D projection, or attempting to use `ax.set_zlabel()` on a standard 2D axis
  - Misunderstanding `bar3d` parameters: confusing `z` (base elevation) with `dz` (height of the bar)
  - Passing 1D arrays directly to `plot_surface` instead of 2D matrices generated via `np.meshgrid`
  - Setting `rstride` and `cstride` to very large values, producing jagged, low-fidelity polygon approximations
  - Forgetting `offset` in `ax.contour(..., zdir='z')`, causing contour lines to float ambiguously in mid-air
  - Expecting interactive 3D camera rotation in static inline Jupyter backends without switching to `%matplotlib widget` or `%matplotlib notebook`
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing coordinates from $(2, 3, 5)$ to $(3, 4, 6)$ without changing the analytical demand
- Asking the same `view_init` elevation definition repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about 3D projections, camera angles, surface strides, and multi-dimensional geometry.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in machine learning loss surface diagnostics, PCA cluster analysis, and scientific computing:

- Reading and predicting the visual output of realistic 3D plotting code snippets
- Calculating camera projection angles, grid mesh shapes, and `bar3d` spatial volumes
- Identifying mesh dimension errors, unprojected 2D axes bugs, and stride performance bottlenecks
- Selecting the correct visualization tools for loss landscapes, optimization trajectories, and 3D clusters

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & 3D Spatial Geometry: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Performance Optimization: ~30 questions

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
- Code blocks must follow modern Python 3, NumPy, and Matplotlib conventions
- All calculations (e.g., elevation/azimuth rotations, mesh dimensions, bar volumes) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year was mplot3d created?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_3d=True`)
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

- Explicitly state why the correct choice is accurate according to the Matplotlib 3D object model
- Detail why each incorrect option fails (e.g., missing projection argument, shape mismatch, parameter confusion, or camera angle error)
- Include concrete references to Matplotlib mechanics (`Axes3D` initialization, `rstride`/`cstride` subsampling, `view_init` mathematics, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): 3D Setup & Projection Architecture (`Axes3D`, `projection="3d"`, 3D axis labels and limits)
- `test-02.md` (Questions 11–20): 3D Scatter Plots (`ax.scatter`, `depthshade`, 4D scalar color mapping, colorbars)
- `test-03.md` (Questions 21–30): 3D Line Plots & Trajectories (parametric spirals, time-series curves, gradient descent paths)
- `test-04.md` (Questions 31–40): 3D Bar Charts with `bar3d` (anchor coordinates `x, y, z` vs. volume extents `dx, dy, dz`, shading)
- `test-05.md` (Questions 41–50): Meshgrid Generation & Wireframe Plots (`np.meshgrid`, `plot_wireframe`, sampling grids)
- `test-06.md` (Questions 51–60): 3D Surface Plots (`plot_surface`, colormap mapping, antialiasing, edgecolors)
- `test-07.md` (Questions 61–70): Surface Stride & Performance Optimization (`rstride`, `cstride`, polygon density control)
- `test-08.md` (Questions 71–80): 3D Contour Projections (`zdir='z'/'x'/'y'`, `offset`, wall-projected level curves)
- `test-09.md` (Questions 81–90): Camera Angles & Perspective (`view_init`, elevation `elev`, azimuth `azim`, top-down views)
- `test-10.md` (Questions 91–100): Machine Learning 3D Applications (Loss surface landscapes, saddle points, PCA cluster analysis)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/3d_plots/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / NumPy / Matplotlib
8. `bar3d` coordinate anchor and dimension math is 100% verified
9. Difference between `plot_wireframe` and `plot_surface` is rigorously tested
10. `view_init(elev, azim)` orientation angles are mathematically exact
11. `rstride` and `cstride` performance mechanics are properly covered
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/3d_plots/` following this prompt.
