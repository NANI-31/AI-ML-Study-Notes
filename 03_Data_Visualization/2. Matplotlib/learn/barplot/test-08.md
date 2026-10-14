# Matplotlib Bar Plot Objective Test — Set 08

---

### 71. Question

A data scientist creates a comprehensive ML model benchmark dashboard using a multi-panel layout:

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models  = ["LR", "DT", "RF", "XGB", "SVM", "KNN"]
metrics = {
    "Accuracy":  [0.85, 0.79, 0.93, 0.94, 0.88, 0.82],
    "F1-Score":  [0.83, 0.77, 0.91, 0.92, 0.86, 0.80],
    "Precision": [0.84, 0.80, 0.92, 0.93, 0.87, 0.81],
    "Recall":    [0.82, 0.74, 0.90, 0.91, 0.85, 0.79],
}

for ax, (metric, scores) in zip(axes.flatten(), metrics.items()):
    bars = ax.bar(models, scores, color="steelblue", width=0.6)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Model Benchmark — Classification Metrics", fontsize=15, y=1.02)
plt.tight_layout()
```

What does `axes.flatten()` do, and why is it necessary here?

- [ ] **A.** It sorts the axes by their position in the figure
- [ ] **B.** It converts the 2×2 2D array of Axes objects into a 1D iterable of 4 Axes, enabling simple iteration with `zip` over the 4 metrics
- [ ] **C.** It applies a flat/linear colour gradient to all 4 axes backgrounds
- [ ] **D.** It removes spacing between subplots, making them flush against each other

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `flatten()` is a NumPy array method that reshapes a multi-dimensional array into 1D. It has no sorting functionality — the ordering after flattening is row-major (left-to-right, top-to-bottom).
- **Option B (Correct):** `plt.subplots(2, 2)` returns `axes` as a 2×2 NumPy array of Axes objects (shape (2,2)). A 2D array cannot be directly zipped with a flat iterable like `metrics.items()` (4 items). `axes.flatten()` converts the (2,2) array into a (4,) 1D array: `[ax00, ax01, ax10, ax11]`, which can then be zipped with 4 metric entries. Each iteration unpacks `(ax, (metric_name, scores))` correctly.
- **Option C (Incorrect):** `flatten()` is a data structure operation (1D conversion), not a colour/style operation. Background colours are set via `ax.set_facecolor()`.
- **Option D (Incorrect):** Subplot spacing is controlled by `plt.subplots_adjust(...)` or `plt.tight_layout()`. The `flatten()` method has no effect on figure layout or spacing.

</details>

---

### 72. Question

In the dashboard from Question 71, `plt.suptitle("...", y=1.02)` is used. What problem does `y=1.02` solve?

- [ ] **A.** It makes the super-title bold
- [ ] **B.** It positions the super-title slightly above the figure boundary (y=1.0) to prevent it from overlapping with the individual subplot titles
- [ ] **C.** It centers the super-title horizontally at 1.02 points from the left
- [ ] **D.** It increases the font size by 2%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Bolding is controlled by `fontweight="bold"` in `suptitle()`, not by `y=`.
- **Option B (Correct):** The `y` parameter of `suptitle()` specifies the y-coordinate in figure coordinates (0.0 = bottom, 1.0 = top). Setting `y=1.02` places the super-title at 2% above the top edge of the figure's subplot area. Without this offset, `suptitle` at `y=1.0` (or the default position) can overlap with the `set_title()` calls on the top-row subplots (Accuracy and F1-Score panels), since both titles compete for the same vertical space near the figure top. The 2% offset provides clearance.
- **Option C (Incorrect):** `y` is a figure-coordinate y-position (vertical), not a horizontal point measurement. Horizontal centering of `suptitle` is controlled by `x=0.5` (the default, centered).
- **Option D (Incorrect):** `y=1.02` is a position parameter, not a scale factor for font size. Font size is set by `fontsize=15` in the suptitle call.

</details>

---

### 73. Question

A developer creates a train vs. validation accuracy grouped bar chart for 5 models:

```python
models   = ["LR", "RF", "GB", "NN", "SVM"]
train_acc = [0.91, 0.98, 0.95, 0.97, 0.89]
val_acc   = [0.89, 0.86, 0.91, 0.88, 0.87]

x = np.arange(5)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b_tr = ax.bar(x - width/2, train_acc, width, label="Train",  color="#2980B9")
b_vl = ax.bar(x + width/2, val_acc,   width, label="Val",    color="#E74C3C")

ax.set_xticks(x, models)
ax.legend()
ax.set_ylim(0, 1.1)
```

Which model shows the **largest generalization gap** (train accuracy − validation accuracy) in this chart?

- [ ] **A.** LR (gap = 0.02)
- [ ] **B.** RF (gap = 0.12)
- [ ] **C.** GB (gap = 0.04)
- [ ] **D.** NN (gap = 0.09)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** LR: train=0.91, val=0.89, gap = 0.02. A small gap indicating good generalization.
- **Option B (Correct):** RF: train=0.98, val=0.86, gap = **0.12**. The largest gap — 12 percentage points between training and validation accuracy. This is a strong visual indicator of overfitting: the RF model achieved very high training accuracy but validated considerably lower, suggesting it memorized training patterns rather than generalizing.
- **Option C (Incorrect):** GB: train=0.95, val=0.91, gap = 0.04. A moderate gap, much smaller than RF's 0.12.
- **Option D (Incorrect):** NN: train=0.97, val=0.88, gap = 0.09. A sizable gap indicating some overfitting, but smaller than RF's 0.12 gap.

</details>

---

### 74. Question

A developer creates a complete ML evaluation bar chart with the following code:

```python
fig, ax = plt.subplots(figsize=(11, 5))

models = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
accuracy  = [0.871, 0.934, 0.951, 0.942]
precision = [0.863, 0.928, 0.947, 0.938]
recall    = [0.854, 0.919, 0.939, 0.930]
f1        = [0.858, 0.923, 0.943, 0.934]

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy,  width, label="Accuracy",  color="#3498DB")
bars2 = ax.bar(x - 0.5*width, precision, width, label="Precision", color="#E74C3C")
bars3 = ax.bar(x + 0.5*width, recall,    width, label="Recall",    color="#2ECC71")
bars4 = ax.bar(x + 1.5*width, f1,        width, label="F1",        color="#F39C12")

ax.set_xticks(x, models, rotation=15, ha="right")
ax.legend(loc="lower right")
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
```

A reviewer flags `ax.set_ylim(0.8, 1.0)` and notes it's appropriate here. Why is this different from the misleading truncation scenario?

- [ ] **A.** It's not different — truncating at 0.8 is equally misleading here
- [ ] **B.** This is a multi-metric grouped chart where the PURPOSE is to compare fine-grained performance differences between models; the 0.8–1.0 window helps distinguish meaningful metric differences that would be invisible in a 0–1 range, AND the reviewer should add a note acknowledging the truncation
- [ ] **C.** Truncation is acceptable for grouped bar charts but not for single-series bar charts
- [ ] **D.** The reviewer is wrong — all bar charts must start at 0 without exception

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Context matters in data visualization ethics. A truncated axis for a SINGLE model chart (where the bar length IS the primary comparison) is highly misleading. For a multi-model comparison where the RELATIVE DIFFERENCES between adjacent bars are the focus, the argument is more nuanced.
- **Option B (Correct):** This question addresses data visualization ethics with nuance. When all models perform in the 87–95% range, a 0–100% y-axis makes all bars look nearly identical (all near the top), making it impossible to perceive meaningful differences. The 0.8–1.0 window sacrifices some proportional honesty in exchange for legibility of inter-model differences. The correct approach is: (1) acknowledge the truncation explicitly (add a note or broken axis indicator), (2) justify it editorially ("zoomed view"), and (3) ensure labels show actual values (via `bar_label`). This is distinct from a single-series chart where truncation creates a false impression of large differences between nearly equal values.
- **Option C (Incorrect):** The grouped vs. single-series distinction is not the deciding factor. The deciding factors are the purpose of the visualization, the magnitude of differences, and whether the truncation is disclosed.
- **Option D (Incorrect):** As discussed throughout this test, "without exception" is too rigid. Data visualization is contextual — a zoomed axis with clear disclosure can be appropriate in specific analytical contexts.

</details>

---

### 75. Question

A developer uses `ax.bar_label()` with a custom `labels=` argument instead of `fmt=`:

```python
bars = ax.bar(models, scores)
custom_labels = ["Low", "High", "Medium", "Very High"]
ax.bar_label(bars, labels=custom_labels, padding=4)
```

What is the effect of passing `labels=custom_labels`?

- [ ] **A.** Each bar is labeled with the corresponding string from `custom_labels` instead of the numeric bar height
- [ ] **B.** The `labels` argument renames the x-axis tick labels
- [ ] **C.** Custom labels override the `fmt=` argument, which is then ignored
- [ ] **D.** Both A and C — custom labels replace bar heights, and fmt= is ignored if labels= is provided

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Correct as far as it goes):** When `labels=custom_labels` is provided, `ax.bar_label()` uses the strings in `custom_labels` directly as the text for each bar's annotation, rather than formatting the numeric bar heights. "Low", "High", "Medium", "Very High" appear above the respective bars.
- **Option B (Incorrect):** `ax.bar_label()` places text on bar tops/centers — it does NOT modify x-axis tick labels. Tick labels are controlled by `ax.set_xticklabels()` or `ax.set_xticks(positions, labels)`.
- **Option C (Correct as far as it goes):** If `labels=` is provided, the `fmt=` argument is superseded — the format string is not applied because there are no numeric values to format. The custom label strings are used as-is.
- **Option D (Correct — best answer):** Both A and C are correct simultaneously. `labels=custom_labels` causes `bar_label` to: (1) use the custom strings directly (not bar heights), AND (2) ignore any `fmt=` argument since formatting only applies to numeric values. D correctly captures both effects.

</details>

---

### 76. Question

A developer creates a feature importance chart using output from a scikit-learn `RandomForestClassifier`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

data = load_iris()
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(data.data, data.target)

importances = clf.feature_importances_
feature_names = data.feature_names

# Sort descending
idx = np.argsort(importances)[::-1]
sorted_features = np.array(feature_names)[idx]
sorted_importances = importances[idx]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(sorted_features, sorted_importances, color="darkorange")
ax.invert_yaxis()   # Most important at top
ax.bar_label(bars, fmt="%.4f", padding=3)
ax.set_xlabel("Feature Importance")
```

Why is `ax.invert_yaxis()` called AFTER `ax.barh()` and not before?

- [ ] **A.** Calling it before would fail — `invert_yaxis` only works on axes with existing artists
- [ ] **B.** The order doesn't matter — `invert_yaxis` can be called at any point before `plt.show()`
- [ ] **C.** Calling it before would create a blank chart because the y-axis limits aren't set until bars are plotted
- [ ] **D.** `invert_yaxis` must be the last statement before `plt.show()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `invert_yaxis()` can be called on an empty axes (before any artists are added). It sets an internal flag that reverses the y-axis direction. When bars are subsequently plotted, the reversed direction is applied.
- **Option B (Correct):** `ax.invert_yaxis()` sets the y-axis `inverted` property to True. This property is applied when the axes is rendered, not when the function is called. It can be called before plotting, after plotting, or even after calling `tight_layout()` — as long as it's before `plt.show()` or `fig.savefig()`. The final rendering respects the most recent state of axis properties at render time.
- **Option C (Incorrect):** `invert_yaxis()` doesn't depend on existing y-axis limits. Even on an empty axes with default limits [0,1], calling `invert_yaxis()` correctly inverts the direction. Subsequent `barh()` calls will add bars with the inverted axis already set.
- **Option D (Incorrect):** There is no requirement that `invert_yaxis()` be the last call before `plt.show()`. Many axis customization calls (`set_ylim`, `set_ylabel`, `tick_params`, etc.) can follow `invert_yaxis()` without undoing the inversion.

</details>

---

### 77. Question

A researcher creates a publication-ready bar chart and adds a legend with custom placement:

```python
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width/2, train_scores, width, label="Training", color="#2980B9")
b2 = ax.bar(x + width/2, val_scores,   width, label="Validation", color="#E74C3C")

ax.legend(loc="upper right", framealpha=0.9, edgecolor="gray",
          fontsize=10, title="Phase", title_fontsize=11)
```

What does `framealpha=0.9` control in the legend?

- [ ] **A.** The transparency of the bars behind the legend
- [ ] **B.** The opacity of the legend box background — 0.9 means 90% opaque (slightly transparent), allowing the plot behind to be faintly visible
- [ ] **C.** The width of the legend border edge
- [ ] **D.** The font opacity of the legend text entries

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `framealpha` affects the LEGEND BOX background, not the bars. Bars' transparency is controlled by the `alpha=` argument in `ax.bar()`. The two are independent.
- **Option B (Correct):** `framealpha` sets the **alpha (opacity) of the legend frame background patch**. `framealpha=0.9` makes the white legend background 90% opaque (10% transparent), allowing a very faint glimpse of whatever bars/gridlines are behind the legend box. `framealpha=1.0` makes it fully opaque (completely blocking the background), `framealpha=0.0` makes it fully transparent (no visible background box). The `edgecolor="gray"` separately controls the colour of the legend box border.
- **Option C (Incorrect):** The legend border width is controlled by `linewidth=` within the legend's patch properties (e.g., via `ax.legend().get_frame().set_linewidth(2)`), not by `framealpha`.
- **Option D (Incorrect):** Text opacity within the legend cannot be set via `framealpha`. Text colour is controlled by `labelcolor=` or Matplotlib's `text.color` rcParam.

</details>

---

### 78. Question

A developer creates a bar chart and uses a list comprehension to dynamically set bar colours based on rank:

```python
scores = np.array([0.78, 0.92, 0.85, 0.96, 0.81])
models = ["A", "B", "C", "D", "E"]

rank_colors = {1: "gold", 2: "silver", 3: "#CD7F32", 4: "steelblue", 5: "steelblue"}
ranks = scores.argsort()[::-1].argsort() + 1   # dense ranks 1–5 (1=best)
colors = [rank_colors[r] for r in ranks]

bars = ax.bar(models, scores, color=colors, width=0.6)
```

For `scores = [0.78, 0.92, 0.85, 0.96, 0.81]`, what colour does Model "C" (scores[2]=0.85) get?

- [ ] **A.** `"gold"` (rank 1)
- [ ] **B.** `"silver"` (rank 2)
- [ ] **C.** `"#CD7F32"` (bronze, rank 3)
- [ ] **D.** `"steelblue"` (rank 4 or 5)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Gold (rank 1) goes to the highest score. Scores sorted descending: 0.96 (D), 0.92 (B), 0.85 (C), 0.81 (E), 0.78 (A). D=rank 1 → gold.
- **Option B (Incorrect):** Silver (rank 2) goes to the second-highest. B=0.92 is rank 2 → silver.
- **Option C (Correct):** `scores = [0.78, 0.92, 0.85, 0.96, 0.81]`. Sorting descending: rank 1=0.96 (index 3, D), rank 2=0.92 (index 1, B), rank 3=0.85 (index 2, **C**), rank 4=0.81 (index 4, E), rank 5=0.78 (index 0, A). Model C (0.85) is rank 3 → `rank_colors[3] = "#CD7F32"` (bronze colour).
- **Option D (Incorrect):** Steelblue (rank 4 or 5) goes to Models E (0.81) and A (0.78). Model C is rank 3, not 4 or 5.

</details>

---

### 79. Question

A developer designs a bar chart report and wants to save it with maximum quality for print:

```python
plt.savefig("report_chart.pdf", bbox_inches="tight", dpi=300)
```

For saving as PDF, which statement about `dpi=300` is most accurate?

- [ ] **A.** Setting `dpi=300` makes the PDF file 300 × 300 pixels in size
- [ ] **B.** For PDF (a vector format), `dpi=300` is used for any raster elements embedded within the PDF (such as images or high-resolution background textures), but vector elements (bars, lines, text) remain resolution-independent regardless of DPI
- [ ] **C.** `dpi=300` increases the number of bars drawn in the chart to fill the resolution
- [ ] **D.** `dpi=300` is irrelevant for PDF — the format uses only points (pt) for measurement and `dpi` is silently ignored

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** PDF is a vector format — it doesn't have a fixed pixel dimension. DPI translates to physical size and raster embedded content resolution, not total pixel count for vector paths.
- **Option B (Correct):** PDF supports both vector paths (for lines, curves, text, and bar outlines) and embedded raster images. For a standard bar chart with no embedded raster images, the `dpi=300` setting has minimal practical effect on the bars and text since they are stored as PDF vector paths. However, if the chart contains any rasterized elements (e.g., complex scatter plots that Matplotlib rasterises for performance, background images, or hatching), those are embedded at `dpi=300`. Matplotlib also uses DPI internally when converting font sizes from points to PDF coordinates. The key distinction is: vector elements are resolution-independent; raster elements use the specified DPI.
- **Option C (Incorrect):** `dpi` is a rendering resolution parameter — it does not affect the number of data elements or artists. A bar chart always renders the same number of bars regardless of DPI.
- **Option D (Incorrect):** While PDF dimensions use points (1 pt = 1/72 inch), `dpi` is NOT completely ignored by Matplotlib when saving PDF. It's used for raster-to-vector conversion ratios and any embedded raster content. Saying it's "silently ignored" is incorrect.

</details>

---

### 80. Question

A developer creates a bar chart with the following complete code and identifies a subtle issue:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
categories = ["Dev", "QA", "Design", "PM"]
headcount  = [12, 8, 6, 4]

bars = ax.bar(categories, headcount, color="steelblue", width=0.7)
ax.bar_label(bars, fmt="%d", padding=5)
ax.set_ylabel("Team Members")
ax.set_title("Team Headcount by Department")
ax.set_ylim(0, 18)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("headcount.png", dpi=120, bbox_inches="tight")
plt.show()
```

A colleague suggests calling `plt.show()` BEFORE `plt.savefig()`. What is the consequence of this order change?

- [ ] **A.** No difference — `plt.show()` and `plt.savefig()` are independent and can be called in either order
- [ ] **B.** Calling `plt.show()` before `plt.savefig()` clears the current figure — `savefig` then saves a blank (empty) figure
- [ ] **C.** The saved file would be higher resolution because `show()` pre-renders the figure
- [ ] **D.** `plt.show()` locks the figure for saving only in PNG format

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The order DOES matter. `plt.show()` displays and then clears the current active figure in interactive backends. After `show()`, the figure state is reset and `savefig()` operates on a new, empty figure.
- **Option B (Correct):** In most Matplotlib backends (especially non-interactive ones like Agg, which is used for script execution), `plt.show()` displays the figure and then **clears the current figure** (`plt.clf()` equivalent). Any subsequent `plt.savefig()` call would save a blank canvas because the figure data was cleared by `show()`. The correct order is ALWAYS: (1) make the plot, (2) `plt.savefig()`, (3) `plt.show()`. This is a very common beginner mistake.
- **Option C (Incorrect):** `plt.show()` renders the figure to the screen but does not pre-cache or improve resolution for subsequent file saves. Resolution is determined entirely by `dpi=` in `savefig()`.
- **Option D (Incorrect):** `plt.show()` does not lock the figure to any format. The format is determined by the file extension or `format=` argument in `savefig()`. The real issue is the clearing of the figure, not format restriction.

</details>
