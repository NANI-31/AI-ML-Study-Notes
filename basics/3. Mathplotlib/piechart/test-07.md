# Matplotlib Pie Chart Objective Test — Set 07

---

### 61. Question

A data scientist creates a side-by-side comparison of device usage between 2023 and 2024:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.pie(share_2023, labels=labels)
ax2.pie(share_2024, labels=labels)
plt.show()
```

When displayed on a wide 12x5 inch canvas, both charts render as horizontal ovals. What code must be added to ensure **both** subplots maintain true circular geometry?

- [ ] **A.** `plt.axis('equal')` once at the very end
- [ ] **B.** `fig.set_aspect('equal')`
- [ ] **C.** `ax1.axis('equal')` and `ax2.axis('equal')` (or `set_aspect('equal')` on both Axes)
- [ ] **D.** `ax1.circular = True` and `ax2.circular = True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Calling `plt.axis('equal')` operates only on the currently active axes (`ax2`), leaving `ax1` distorted.
- **Option B (Incorrect):** `set_aspect()` is an `Axes` method, not a `Figure` method; calling it on `fig` raises an `AttributeError`.
- **Option C (Correct):** In Matplotlib's Object-Oriented interface, aspect ratio constraints must be applied explicitly to each `Axes` instance. Calling `ax1.axis('equal')` and `ax2.axis('equal')` (or `ax.set_aspect('equal')`) guarantees that both subplot panels enforce a 1:1 coordinate aspect ratio, rendering two perfect circles side-by-side.
- **Option D (Incorrect):** `circular` is not an attribute on Matplotlib Axes.

</details>

---

### 62. Question

An analyst plots customer segments across two different regional cohorts:
- In Region 1, "Enterprise" is plotted first (Color 0: blue) and "Retail" second (Color 1: orange).
- In Region 2, "Retail" is plotted first and "Enterprise" second.

Because both calls rely on default automatic color cycling, "Enterprise" is blue in Region 1 but orange in Region 2. How should the analyst fix this visual inconsistency?

- [ ] **A.** Reorder the imports at the top of the script
- [ ] **B.** Define an explicit categorical color mapping dictionary (e.g., `palette = {'Enterprise': '#2b5c8f', 'Retail': '#e66101'}`) and map slices to fixed colors across both subplots
- [ ] **C.** Increase the DPI to 300
- [ ] **D.** Remove the legend from Region 2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Python imports do not influence data color mappings.
- **Option B (Correct):** Color should maintain a consistent, stable semantic meaning across related visual panels. Inverting colors between subplots forces viewers to constantly re-read legends, causing severe cognitive dissonance. Establishing an explicit dictionary that maps each category name to a fixed color and passing `colors=[palette[k] for k in cohort_labels]` ensures that "Enterprise" is always blue and "Retail" is always orange in every panel.
- **Option C (Incorrect):** DPI adjusts image resolution, not artist colors.
- **Option D (Incorrect):** Removing the legend destroys intelligibility.

</details>

---

### 63. Question

When constructing a multi-panel figure containing two pie chart subplots, which method sets the **overarching main title** for the entire figure canvas (as opposed to individual subplot headers)?

- [ ] **A.** `fig.suptitle("Annual Performance Overview", fontsize=16)`
- [ ] **B.** `fig.title("Annual Performance Overview")`
- [ ] **C.** `plt.main_title("Annual Performance Overview")`
- [ ] **D.** `ax1.set_super_title("Annual Performance Overview")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, `fig.suptitle()` (**sup**er **title**) creates a centered title across the entire Figure window, while individual subplot panels receive localized titles via `ax1.set_title()` and `ax2.set_title()`.
- **Option B (Incorrect):** `Figure` objects do not have a `.title()` method.
- **Option C (Incorrect):** `main_title()` is not a Matplotlib Pyplot function.
- **Option D (Incorrect):** `set_super_title()` does not exist on Axes.

</details>

---

### 64. Question

A data engineer builds a two-panel dashboard comparing 2023 and 2024 market share. Both subplots share the exact same 5 categories. To prevent clutter, the engineer wants a **single unified legend** positioned horizontally below both charts. Which pattern achieves this?

- [ ] **A.** Call `ax1.legend()` and `ax2.legend()` with `merge=True`
- [ ] **B.** `plt.pie(..., global_legend=True)`
- [ ] **C.** `ax1.share_legend(ax2)`
- [ ] **D.** Capture wedge handles from the first pie and call `fig.legend(wedges1, categories, loc='lower center', ncol=5)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `legend()` does not have a `merge` keyword argument.
- **Option B (Incorrect):** `global_legend` is not a parameter of `plt.pie()`.
- **Option C (Incorrect):** There is no `share_legend` method in Matplotlib.
- **Option D (Correct):** Capturing the list of wedge artists returned by `wedges1, _ = ax1.pie(...)` and attaching them to the figure canvas via `fig.legend(wedges1, categories, loc='lower center', ncol=5)` creates a clean, shared horizontal legend docked beneath both subplots without redundant duplication.

</details>

---

### 65. Question

A product manager compares user feedback before and after a feature release:
- Before: Positive $24\%$, Neutral $45\%$, Negative $31\%$
- After: Positive $26\%$, Neutral $45Client\%$, Negative $29\%$

Both cohorts are plotted as side-by-side pie charts. Why do cognitive psychologists and visualization researchers consider side-by-side pie charts ineffective for evaluating these subtle shifts?

- [ ] **A.** Pie charts cannot render percentages that end in an odd number.
- [ ] **B.** Humans have poor visual acuity for detecting subtle angular differences (such as a $2\%$ change from $24\%$ to $26\%$) across separate rotating dials; a grouped bar chart or slopegraph enables direct linear alignment that reveals small differences instantly.
- [ ] **C.** Side-by-side subplots cause Matplotlib to crash if wedge angles differ by less than $5^\circ$.
- [ ] **D.** Neutral categories must always be plotted in a separate bar chart.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Pie charts handle arbitrary real numbers and percentages.
- **Option B (Correct):** According to the perceptual hierarchy established by Cleveland and McGill, comparing angles between two separate unaligned circular charts forces the eye to estimate rotational orientation and area, where human judgment error is high. A $2\%$ shift ($7.2^\circ$) across separate dials is virtually imperceptible to the naked eye. In contrast, grouped bars align values along a common baseline, making the $2\%$ difference immediately obvious through bar height comparison.
- **Option C (Incorrect):** Matplotlib renders arbitrary angle differences without crashing.
- **Option D (Incorrect):** Neutral categories can be visualized in any chart type.

</details>

---

### 66. Question

A financial analyst sets up a $2 \times 2$ grid of pie charts to compare market share across 4 international regions:

```python
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
```

How does the analyst access the top-right subplot axes to draw the European market pie chart?

- [ ] **A.** `axes[1]`
- [ ] **B.** `axes.top_right`
- [ ] **C.** `axes[0, 1]`
- [ ] **D.** `axes[1, 0]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** When `nrows > 1` and `ncols > 1`, `axes` is a 2D NumPy array of shape `(2, 2)`; indexing with a single integer accesses an entire row.
- **Option B (Incorrect):** `axes` has no `top_right` attribute.
- **Option C (Correct):** In a 2D array of subplots:
  - `axes[0, 0]`: Top-left
  - `axes[0, 1]`: **Top-right**
  - `axes[1, 0]`: Bottom-left
  - `axes[1, 1]`: Bottom-right
  Thus, `axes[0, 1]` addresses the top-right panel.
- **Option D (Incorrect):** `axes[1, 0]` corresponds to row index 1, column index 0, which is the **bottom-left** panel.

</details>

---

### 67. Question

An analyst saves a multi-panel figure containing an exploded pie chart:

```python
fig.savefig("market_comparison.png", bbox_inches='tight')
```

What is the primary benefit of passing `bbox_inches='tight'` when exporting pie charts with exploded slices?

- [ ] **A.** It calculates the tight bounding box around all figure elements, preventing outwardly displaced exploded wedges from being clipped or truncated by the image canvas edge.
- [ ] **B.** It compresses the file into a `.zip` archive.
- [ ] **C.** It forces the background to transparent white.
- [ ] **D.** It automatically removes all labels from the exported image.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Exploded slices translate outward beyond standard axis limits. By default, `savefig()` saves the fixed canvas box, which can slice off the outer tips of exploded wedges or category labels. Specifying `bbox_inches='tight'` instructs Matplotlib to calculate the full bounding box of all rendered artists (including exploded wedges and external titles) and adjust the canvas boundaries to fit everything cleanly.
- **Option B (Incorrect):** `bbox_inches='tight'` is a spatial cropping parameter; it does not compress into zip archives.
- **Option C (Incorrect):** Transparent backgrounds require `transparent=True`.
- **Option D (Incorrect):** Labels are preserved.

</details>

---

### 68. Question

In a side-by-side two-panel pie chart figure, the category labels of the left chart are overlapping the perimeter of the right chart. Which command increases horizontal spacing between the two subplots?

- [ ] **A.** `plt.pie(..., padding=10)`
- [ ] **B.** `fig.set_gap(50)`
- [ ] **C.** `ax1.margin_right(20)`
- [ ] **D.** `plt.subplots_adjust(wspace=0.4)` or `fig.tight_layout()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `padding` is not a parameter of `plt.pie()`.
- **Option B (Incorrect):** `set_gap` is not a `Figure` method.
- **Option C (Incorrect):** `margin_right` is not an `Axes` method.
- **Option D (Correct):** `plt.subplots_adjust(wspace=0.4)` sets the horizontal spacing (**w**idth **space**) between adjacent subplot columns as a fraction of the average axes width. Increasing `wspace` (or invoking `fig.tight_layout()` with padding) pulls the two subplots apart, preventing label collisions.

</details>

---

### 69. Question

A researcher is plotting 4 customer segments across 3 age demographics ($1 \times 3$ subplots). To maintain a minimalist aesthetic and prevent visual clutter, what is the best labeling strategy?

- [ ] **A.** Print all 4 category names and percentages on all 3 charts
- [ ] **B.** Hide all labels completely and leave viewers to guess the meaning of the colors
- [ ] **C.** Display percentages inside the wedges on all charts, but remove perimeter category names from the subplots and provide a single, shared legend at the top or bottom of the figure
- [ ] **D.** Print labels only on the middle chart and leave the outer charts unlabeled

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Repeating identical text labels across multiple side-by-side charts causes repetitive visual clutter and increases cognitive load.
- **Option B (Incorrect):** Removing all identification renders the chart unintelligible.
- **Option C (Correct):** For small multiples sharing identical categories, best practice dictates decoupling the category names from the individual circles into a single centralized legend. Each individual pie then displays only its clean numerical percentage values inside the wedges, maximizing data-ink ratio and visual elegance.
- **Option D (Incorrect):** Unlabeled charts create ambiguity.

</details>

---

### 70. Question

Consider the following statements regarding multi-panel pie chart subplots:

1. `ax.axis('equal')` must be called on each individual subplot Axes instance to prevent elliptical distortion.
2. In a $2 \times 2$ grid created via `fig, axes = plt.subplots(2, 2)`, the bottom-right axes is accessed as `axes[1, 1]`.
3. `fig.suptitle()` adds an overarching title centered above all subplot panels.
4. Using an explicit dictionary mapping categories to fixed colors prevents color-cycling mismatches when category order differs between subplots.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** Each Axes maintains independent aspect ratio settings.
- **Statement 2 is correct:** `axes[1, 1]` indexes row 1, column 1 (the bottom-right cell in zero-indexed arrays).
- **Statement 3 is correct:** `fig.suptitle()` sets the main figure-level title.
- **Statement 4 is correct:** Predefined color maps ensure consistent visual semantics across panels regardless of plotting order.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
