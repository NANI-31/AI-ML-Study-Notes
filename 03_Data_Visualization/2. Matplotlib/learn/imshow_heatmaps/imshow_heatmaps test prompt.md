Generate 100 New Objective Questions from the Reference Imshow & Heatmaps Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib 2D Image Rendering & Heatmaps (`imshow` Mechanics, Array Dtypes & Shapes, Colormap Taxonomies, `vmin`/`vmax` Dynamic Range Scaling, Interpolation Modes, `aspect` & `origin` Orientations, Colorbar Sizing/Padding/Ticks, Cell Text Annotations, Filled Contours `contourf`, and Confusion Matrix / Activation Map ML Diagnostics).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/imshow_heatmaps/`:

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

First, thoroughly analyze the supplied `13imgshow.ipynb`, `3. plot.ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - `imshow` Array Shapes and Image Data Representation:
    - 2D scalar arrays $(M, N)$ rendered via colormap
    - 3D RGB arrays $(M, N, 3)$ with channels $[R, G, B]$
    - 3D RGBA arrays $(M, N, 4)$ with alpha transparency
    - Numerical data types and normalization: `uint8` $[0, 255]$ vs. `float32`/`float64` $[0.0, 1.0]$
    - PIL to NumPy image pipelines: `Image.open()`, `np.array(image)`, grayscale conversion via `.convert("L")`
  - Dynamic Range Scaling (`vmin` and `vmax`):
    - Default auto-scaling to `data.min()` and `data.max()`
    - Explicit range locking with `vmin` and `vmax`
    - Why matching `vmin` and `vmax` across side-by-side subplots is mandatory for honest comparative visualization
  - Colormap Classification and Selection:
    - Sequential colormaps (`viridis`, `plasma`, `inferno`, `magma`, `Blues`) for continuous unipolar metrics
    - Diverging colormaps (`coolwarm`, `bwr`, `seismic`) for zero-centered bipolar metrics (correlations, error residuals)
    - Qualitative colormaps (`tab10`, `Set1`) for categorical feature clustering
    - Perceptual uniformity: why `viridis` avoids luminance artifacts present in legacy `jet` or rainbow maps
  - Interpolation Methods:
    - Pixel-exact rendering: `'nearest'` and `'none'`
    - Smoothing algorithms: `'bilinear'`, `'bicubic'`, `'gaussian'`
    - Artifact avoidance: why `'nearest'` is mandatory for heatmaps, confusion matrices, and discrete matrices
  - Aspect Ratio and Coordinate Origin:
    - `aspect='equal'` (square pixel aspect ratio, standard for computer vision images)
    - `aspect='auto'` (stretches non-square feature-by-sample matrices to fill the subplot dimensions)
    - `origin='upper'` (default: row 0 at top-left, matrix/image convention) vs. `origin='lower'` (row 0 at bottom-left, Cartesian coordinate convention)
  - Colorbar Customization:
    - Basic colorbar: `plt.colorbar()` / `fig.colorbar(im, ax=ax)`
    - Orientation: `'vertical'` vs. `'horizontal'`
    - Geometry controls: `shrink=0.8`, `aspect=20`, `pad=0.04`
    - Tick marks and labels: `cbar.set_ticks(...)`, `cbar.set_label("Metric Name")`
    - Shared colorbar spanning multiple subplots: `fig.colorbar(im, ax=axes)`
  - Annotated Heatmaps and Confusion Matrices:
    - Drawing cell values programmatically:
      `for i in range(rows): for j in range(cols): ax.text(j, i, data[i, j], ha="center", va="center")`
    - Note the coordinate order: `ax.text(x, y)` corresponds to `ax.text(col, row)` which is `ax.text(j, i)`
    - Dynamic text contrast: thresholding cell background to flip text between white and black
  - Contour Plots (`contour` and `contourf`):
    - 2D grid evaluation via `np.meshgrid(x, y)`
    - Line contours: `plt.contour(X, Y, Z, levels=10)`
    - Filled contours: `plt.contourf(X, Y, Z, levels=20, cmap='viridis')`
    - Adding contour colorbars
  - Machine Learning Visualization Use Cases:
    - Multi-class Confusion Matrix with raw counts and normalized recall percentages
    - Feature Correlation Matrices bounded strictly in $[-1, 1]$ using `coolwarm`
    - Deep learning convolutional feature maps and hidden neuron activation matrices
    - Loss landscape contour maps with local minima and saddle points
- Important definitions:
  - imshow, vmin, vmax, cmap, aspect, origin, interpolation, colorbar, meshgrid, contour, contourf, perceptually uniform
- Common mistakes and misconceptions:
  - Swapping indices in cell text loops: calling `ax.text(i, j, val)` instead of `ax.text(j, i, val)`, transposing all labels across the diagonal
  - Passing a floating-point image with values in $[0, 255]$ without dividing by $255$, causing Matplotlib to clip nearly all pixels to pure white
  - Leaving `origin='upper'` on Cartesian contour/spatial data, vertically flipping the entire dataset
  - Forgetting `aspect='auto'` on tall $(1000 \times 10)$ feature matrices, resulting in an unreadable pencil-thin strip
  - Using a sequential colormap on a correlation matrix, obscuring negative correlations
  - Comparing two heatmaps without locking `vmin` and `vmax`, making identical values look completely different
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing matrix dimensions from $3 \times 3$ to $4 \times 4$ without changing the analytical demand
- Asking the same colormap name question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about `imshow`, colormap math, dynamic scaling, and 2D matrix rendering.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, computer vision, and machine learning diagnostics:

- Reading and predicting the output of realistic `imshow` and `contourf` code snippets
- Calculating matrix coordinate transformations, aspect scaling, and `vmin`/`vmax` color mappings
- Identifying transposed text label bugs, colormap clipping errors, and unscaled floating-point image bugs
- Selecting the correct visualization tools for confusion matrices, correlation heatmaps, and CNN activations

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Matrix Coordinate Math: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Production Heatmaps: ~30 questions

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
- All calculations (e.g., matrix indexing, coordinate mappings, dynamic range scaling) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year was viridis created?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_heatmap=True`)
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
- Detail why each incorrect option fails (e.g., transposed coordinates, clipping artifact, invalid colormap class, or interpolation blur)
- Include concrete references to Matplotlib mechanics (`ax.text(col, row)` ordering, `origin='lower'` coordinate inversion, `vmin`/`vmax` normalization, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): `imshow` Fundamentals & Image Data Shapes (2D grayscale, 3D RGB/RGBA, uint8 vs. float, PIL arrays)
- `test-02.md` (Questions 11–20): Dynamic Range Scaling (`vmin`, `vmax`, multi-panel contrast consistency, auto-scaling pitfalls)
- `test-03.md` (Questions 21–30): Colormap Taxonomies (sequential, diverging, qualitative, perceptual uniformity vs. legacy rainbow maps)
- `test-04.md` (Questions 31–40): Interpolation Modes (`'nearest'`, `'bilinear'`, `'bicubic'`, smoothing vs. pixel preservation)
- `test-05.md` (Questions 41–50): Geometry & Orientations (`aspect='equal'` vs. `'auto'`, `origin='upper'` vs. `'lower'`)
- `test-06.md` (Questions 51–60): Colorbar Engineering (vertical/horizontal orientation, shrink, pad, aspect, tick labels, multi-axis binding)
- `test-07.md` (Questions 61–70): Annotated Heatmaps & Cell Text Formatting (`ax.text(j, i)`, thresholding text contrast, numeric formatting)
- `test-08.md` (Questions 71–80): Contour & Surface Projections (`np.meshgrid`, `plt.contour`, `plt.contourf`, level lines)
- `test-09.md` (Questions 81–90): Feature Correlation & Distance Matrices (diverging $[-1, 1]$ maps, symmetric diagonals, categorical labels)
- `test-10.md` (Questions 91–100): Machine Learning & Computer Vision Applications (Confusion Matrices, CNN feature maps, activation layers)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/imshow_heatmaps/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Matplotlib
8. Mathematical coordinate mappings and cell text indexings (`j, i`) are 100% verified
9. Difference between `origin='upper'` and `origin='lower'` is strictly tested
10. `vmin` and `vmax` dynamic range mechanics are accurately explained
11. Diverging vs. sequential colormap criteria are properly tested
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/imshow_heatmaps/` following this prompt.
