Generate 100 New Objective Questions from the Reference Annotations & Legends Material

Use the attached/reference Jupyter Notebook and Markdown file as the primary source of truth for the topic, concepts, terminology, difficulty level, and question style.

Reference files:

- Reference Exam: [test.md](file:///d:/Data/GitHub/AI-ML-Study-Notes/basics/3.%20Mathplotlib/test.md)

Task

Analyze the reference notebooks and test file carefully and generate 100 NEW multiple-choice objective questions focused specifically on Matplotlib Annotations, Text, Spans, and Legends (Text Placement & Alignment, Coordinate Systems `transData` vs. `transAxes`, Point-and-Shoot Annotations with `arrowprops`, Threshold & Baseline Lines with `axhline`/`axvline`, Confidence Span Shading with `axhspan`/`axvspan`, Legend Positioning with `loc` and `bbox_to_anchor`, Multi-Column Legends, Custom Handles/Labels, Multiple Legends on a Single Axes, and ML Diagnostics Annotation).

The 100 questions must be divided into 10 separate Markdown files, with exactly 10 questions per file.

Create in `basics/3. Mathplotlib/annotations_legends/`:

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

First, thoroughly analyze the supplied `1. basic.ipynb`, `5. ticks.ipynb`, `3d .ipynb`, and `test.md`.

Identify:

- Topics and subtopics covered:
  - Text insertion and styling:
    - `plt.text(x, y, "string", ...)` / `ax.text(x, y, "string", ...)`
    - Font parameters: `fontsize`, `fontweight`, `fontfamily`, `color`, `alpha`
    - Text alignment: `ha='center'/'left'/'right'`, `va='center'/'top'/'bottom'/'baseline'`
    - Bounding boxes for text: `bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="blue", lw=2, alpha=0.5)`
  - Coordinate transformations:
    - Data coordinates (`transform=ax.transData`, default for `ax.text`): text moves with data zoom/pan
    - Normalized Axes coordinates (`transform=ax.transAxes`, $[0, 1]$): text stays fixed relative to the plot box (e.g., top-right corner stamps at $(0.95, 0.95)$)
    - Normalized Figure coordinates (`transform=fig.transFigure`, $[0, 1]$)
  - Point-and-shoot Annotations (`ax.annotate`):
    - Target point `xy=(x, y)` vs. Text location `xytext=(x, y)`
    - Coordinate system specification: `xycoords='data'`, `textcoords='offset points'`, `textcoords='axes fraction'`
    - Arrow customization (`arrowprops`):
      - Simple arrow dictionaries: `arrowprops=dict(arrowstyle="->", color="red", lw=1.5)`
      - Advanced connection styles: `connectionstyle="arc3,rad=.2"`, `angle3`
      - Classical arrows: `width`, `headwidth`, `headlength`, `shrink=0.05`
  - Reference Lines and Shading Spans:
    - Horizontal reference lines: `ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5)`
    - Vertical reference lines: `ax.axvline(x=10, color='red', linestyle=':')`
    - Horizontal shaded regions: `ax.axhspan(ymin, ymax, facecolor='green', alpha=0.2)`
    - Vertical shaded regions: `ax.axvspan(xmin, xmax, facecolor='red', alpha=0.2)`
    - Finite vs. Infinite coordinate spans: `axhline` spanning the full axes width $[0, 1]$ vs. `plot([x1, x2], [y1, y2])`
  - Legend Construction and Placement:
    - Automatic handle collection from `label` parameters: `ax.plot(x, y, label="ROC Curve"); ax.legend()`
    - Explicit handle and label passing: `ax.legend(handles=[line1, line2], labels=['Train', 'Validation'])`
    - Internal location codes and strings: `loc="upper right"`, `"best"`, `"lower left"`, `"center"`
    - External positioning with `bbox_to_anchor`:
      - `ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))` (pinning directly outside the right edge)
      - `bbox_to_anchor=(0.5, -0.15), loc="upper center"` (pinning below the plot)
  - Legend Aesthetics and Formatting:
    - Multi-column legends: `ncol=2` / `ncols=3`
    - Frame control: `frameon=True/False`, `facecolor='white'`, `edgecolor='black'`, `framealpha=0.8`, `shadow=True`, `fancybox=True`
    - Legend titles: `title="Model Families"`, `title_fontsize=12`
    - Marker scaling in legend handles: `markerscale=2.0`
  - Advanced Legend Mechanics:
    - Multiple legends on a single axes: adding the first legend manually with `ax.add_artist(first_legend)` so the second `ax.legend()` does not overwrite it
    - Sorting and filtering legend items
  - Machine Learning Visualization Use Cases:
    - Highlighting "Early Stopping Point" or "Optimal Trade-off Threshold" on Loss and Accuracy curves
    - Marking random baseline ($AUC = 0.5$) with a dashed `axhline` or diagonal reference line
    - Shading overfitting epochs with an `axvspan`
    - Multi-model evaluation charts requiring external multi-column legends to prevent data occlusion
- Important definitions:
  - transData, transAxes, transFigure, xy, xytext, xycoords, textcoords, arrowprops, bbox, axhline, axhspan, axvline, axvspan, bbox_to_anchor, loc, ncol
- Common mistakes and misconceptions:
  - Calling `ax.legend()` without supplying `label` in plot calls, resulting in "No artists with labels found to put in legend"
  - Calling `ax.legend()` a second time and wondering why the first legend disappeared (missing `ax.add_artist()`)
  - Confusing `transData` with `transAxes` in `ax.text()`, placing text far off-screen when using values like $(0.9, 0.9)$ in data space
  - Swapping `xy` (the feature pointed to) and `xytext` (where the text label sits) in `ax.annotate()`
  - Using `bbox_to_anchor` coordinates without understanding that `loc` defines which corner of the legend box anchors to those coordinates
  - Confusing `ax.axhline()` (infinite span across axes) with `ax.hlines()` (requiring explicit `xmin` and `xmax` data points)
- Question structure, option style, and explanation depth
- Correct-answer format and interactive presentation

Use these as the foundation for generating new questions.

Do not simply copy, paraphrase, or slightly modify the existing questions.

The generated questions must test the same underlying knowledge while introducing new datasets, model metrics, numerical values, code patterns, and reasoning requirements.

2. Generate Completely New Questions

All 100 questions must be meaningfully different from the questions in the reference files.

Avoid:

- Rephrasing an existing question without changing what it tests
- Simply changing the label text from "Peak" to "Highest" without changing the analytical demand
- Asking the same legend `loc` question repeatedly
- Creating trivial true/false style questions disguised as multiple choice

Instead, create questions that require the learner to apply, calculate, interpret, debug, or reason about annotations, reference spans, coordinate transforms, and legend mechanisms.

3. Maintain Real Rigor and Balance

Ensure the questions reflect authentic scenarios in data science, machine learning diagnostics, and publication visualization:

- Reading and predicting the visual placement of annotations, arrows, and external legends
- Calculating coordinate offsets, `bbox_to_anchor` alignments, and transform mappings
- Identifying legend overwrite bugs, missing artist labels, and clipped annotation arrows
- Selecting the correct parameter configurations for publication-grade reference marks and multi-panel dashboards

Difficulty Breakdown across the 100 questions:

- Conceptual / Architectural Understanding: ~30 questions
- Code Reading, Output Prediction & Coordinate Math: ~40 questions
- Diagnostic Scenarios, Bug Fixing & Production Layouts: ~30 questions

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
- All calculations (e.g., coordinate transforms, `bbox_to_anchor` alignments, span bounds) must be mathematically precise
- Code snippets should model real ML workflow tasks: training loss curves, ROC-AUC comparisons, multi-axes hyperparameter heatmaps, and publication graphics

6. Prohibited Question Types (Anti-Paraphrasing Rules)

Do not include:

- Direct rephrasing of questions from `test.md`
- Vague trivia questions (e.g., "In what year were arrowprops introduced?")
- Trivial syntax questions with obvious non-existent parameters (e.g., `make_legend=True`)
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
- Detail why each incorrect option fails (e.g., coordinate system mismatch, overwritten legend artist, wrong arrow style syntax, or clipping)
- Include concrete references to Matplotlib mechanics (`transAxes` vs. `transData`, `add_artist` requirements, `bbox_to_anchor` geometry, etc.)

9. Distribution Across the 10 Files

- `test-01.md` (Questions 1–10): Text Fundamentals (`ax.text`, font properties, horizontal and vertical alignment)
- `test-02.md` (Questions 11–20): Coordinate Transformations (`transData`, `transAxes`, `transFigure`, corner metric cards)
- `test-03.md` (Questions 21–30): Point-and-Shoot Annotations (`ax.annotate`, `xy` target vs. `xytext` label, `textcoords`)
- `test-04.md` (Questions 31–40): Arrow Styles & Geometry (`arrowprops`, `arrowstyle`, `connectionstyle`, `shrink`, curvature)
- `test-05.md` (Questions 41–50): Reference Lines (`axhline`, `axvline`, baselines, threshold indicators, dashed styles)
- `test-06.md` (Questions 51–60): Region & Span Highlighting (`axhspan`, `axvspan`, alpha shading, confidence and anomaly zones)
- `test-07.md` (Questions 61–70): Legend Construction & Internal Positioning (`loc` strings and codes, automatic vs. explicit handles)
- `test-08.md` (Questions 71–80): External Legend Pinning (`bbox_to_anchor`, `ncol`, preventing chart area occlusion)
- `test-09.md` (Questions 81–90): Advanced Legend Customization (multiple legends via `add_artist`, custom frames, marker scaling)
- `test-10.md` (Questions 91–100): Comprehensive ML Diagnostics Visualization (annotated loss curves, ROC thresholds, multi-class dashboards)

10. Quality Control Checklist (14-Point Pre-Submission Gate)

Before finalizing, verify:

1. Exact file naming: `basics/3. Mathplotlib/annotations_legends/test-01.md` through `test-10.md`
2. Exactly 10 questions per file
3. Continuous numbering 1 through 100
4. Markdown task checkboxes are blank `- [ ]`
5. HTML tags `<details>` and `<summary>` are properly opened and closed
6. Answer key distribution across 100 questions is balanced (~25% each for A, B, C, D)
7. All code snippets are syntactically valid Python 3 / Matplotlib
8. Mathematical coordinate offsets and `bbox_to_anchor` mechanics are 100% accurate
9. `transAxes` $[0, 1]$ vs. `transData` differences are strictly preserved
10. `ax.annotate` `xy` and `xytext` roles are correctly tested
11. Multiple legends requirement of `ax.add_artist` is properly enforced
12. Every option (A, B, C, D) has an individual explanation in the accordion
13. No question is a trivial paraphrase of `test.md`
14. No duplicate questions across files

15. Final Deliverables

Generate the 10 files in `basics/3. Mathplotlib/annotations_legends/` following this prompt.
