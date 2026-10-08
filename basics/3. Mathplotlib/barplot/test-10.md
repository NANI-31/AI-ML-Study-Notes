# Matplotlib Bar Plot Objective Test — Set 10

---

### 91. Question

A developer creates a comprehensive ML model report with the following capstone script:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

models = ["Logistic Regression", "Decision Tree", "Random Forest",
          "Gradient Boosting", "Neural Network"]
train_f1 = np.array([0.863, 0.941, 0.978, 0.971, 0.965])
val_f1   = np.array([0.855, 0.812, 0.891, 0.942, 0.923])

gap = train_f1 - val_f1
x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# Left: Grouped bars
b_tr = axes[0].bar(x - width/2, train_f1, width, color="#2980B9", label="Train F1")
b_vl = axes[0].bar(x + width/2, val_f1,   width, color="#E74C3C", label="Val F1")
axes[0].set_xticks(x, [m.replace(" ", "\n") for m in models], fontsize=8)
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=9)
axes[0].set_title("Train vs Validation F1-Score", fontsize=12)
axes[0].bar_label(b_tr, fmt="%.3f", padding=2, fontsize=7)
axes[0].bar_label(b_vl, fmt="%.3f", padding=2, fontsize=7)

# Right: Generalization gap horizontal bars
gap_colors = ["#E74C3C" if g > 0.05 else "#27AE60" for g in gap]
axes[1].barh(models, gap, color=gap_colors, height=0.5)
axes[1].axvline(0.05, color="black", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Generalization Gap (Train F1 − Val F1)")
axes[1].set_title("Overfitting Analysis", fontsize=12)

# Custom legend for right panel
legend_handles = [
    Patch(color="#E74C3C", label="Overfitting (gap > 5%)"),
    Patch(color="#27AE60", label="Good Generalization"),
]
axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9)

plt.suptitle("ML Model Performance Report", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("model_report.pdf", bbox_inches="tight")
```

In the right panel, `axes[1].axvline(0.05, ...)` is used instead of `axhline`. Why?

- [ ] **A.** `axvline` is always required for horizontal bar charts
- [ ] **B.** The right panel is a horizontal bar chart (`barh`) where gap values are on the x-axis; `axvline(0.05)` draws a vertical reference line at x=0.05 to mark the 5% overfitting threshold
- [ ] **C.** `axhline` draws coloured bars and would conflict with the barh bars
- [ ] **D.** `axvline` and `axhline` produce the same line in horizontal bar charts

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `axvline` is NOT always required for horizontal bar charts. The choice between `axvline` and `axhline` depends on which AXIS carries the reference value. In horizontal bar charts, either can be appropriate depending on the comparison needed.
- **Option B (Correct):** In `axes[1].barh(models, gap, ...)`, the `gap` values (ranging from ~0.01 to ~0.09) are plotted along the **x-axis** (horizontal extent of bars). The 5% threshold (0.05) is a threshold on the gap value — which lives on the x-axis. Therefore, `axvline(0.05)` draws a **vertical** line at x=0.05, crossing through all bars. Bars extending beyond this line to the right are overfitting models; bars stopping before this line generalize well. An `axhline(0.05)` would draw a horizontal line at y=0.05, which in this chart's coordinate system would bisect between the first and second model — meaningless as a threshold indicator.
- **Option C (Incorrect):** `axhline` draws a horizontal line at a specified y-value — it does NOT draw bars of any kind and does not conflict with `barh`. It is simply semantically wrong for this use case.
- **Option D (Incorrect):** `axvline` draws a VERTICAL line (constant x); `axhline` draws a HORIZONTAL line (constant y). They produce perpendicular lines and are NOT the same.

</details>

---

### 92. Question

In the capstone script from Question 91, the x-tick labels are formatted with:

```python
axes[0].set_xticks(x, [m.replace(" ", "\n") for m in models], fontsize=8)
```

What does `m.replace(" ", "\n")` do to "Gradient Boosting"?

- [ ] **A.** Removes all spaces from the string: `"GradientBoosting"`
- [ ] **B.** Replaces each space with a newline character, causing the label to wrap to multiple lines: `"Gradient\nBoosting"` displays as two lines
- [ ] **C.** Replaces the string with its first `"\n"` character count
- [ ] **D.** Adds a newline at the end only: `"Gradient Boosting\n"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `replace(" ", "\n")` replaces spaces with newline characters (`\n`), not removes them. To remove spaces, one would use `replace(" ", "")`.
- **Option B (Correct):** `"Gradient Boosting".replace(" ", "\n")` returns `"Gradient\nBoosting"`. When this string is used as a Matplotlib tick label, Matplotlib renders multi-line tick labels by splitting on `\n`: "Gradient" appears on the first line and "Boosting" on the second line, stacked vertically. This is an effective technique for reducing x-axis label width without rotation — particularly useful for bar charts where adjacent labels might otherwise overlap. "Logistic\nRegression", "Decision\nTree", etc. would similarly wrap.
- **Option C (Incorrect):** `str.replace()` is a string method that replaces occurrences of one substring with another. It does not count newline characters or return numeric results.
- **Option D (Incorrect):** `"Gradient Boosting".replace(" ", "\n")` replaces ALL spaces in the string, not just the last one. "Gradient Boosting" has one space (between the two words), so `\n` replaces that single space, producing `"Gradient\nBoosting"`. If the model name had multiple words (e.g., "Long Model Name"), ALL spaces would be replaced with newlines.

</details>

---

### 93. Question

In the capstone report, `matplotlib.patches.Patch` is used to create a custom legend:

```python
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="#E74C3C", label="Overfitting (gap > 5%)"),
    Patch(color="#27AE60", label="Good Generalization"),
]
axes[1].legend(handles=legend_handles, loc="lower right")
```

Why is a custom `Patch` legend needed here instead of the automatic legend from bar labels?

- [ ] **A.** `ax.barh()` does not automatically generate legend entries
- [ ] **B.** The bars in the right panel use dynamically assigned per-bar colours (via a list comprehension) — there is no single `label=` on the `barh()` call, so the automatic legend would have no entries; custom `Patch` objects provide legend proxies with the desired colours and labels
- [ ] **C.** `Patch` objects are required for all horizontal bar chart legends
- [ ] **D.** The `loc="lower right"` parameter is only available with custom handles

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `ax.barh()` CAN generate automatic legend entries when `label=` is passed. The issue here is not that `barh` doesn't support legends — it's that no `label=` was provided to the `barh()` call.
- **Option B (Correct):** The `barh()` call uses `color=gap_colors` (a list of colours, one per bar). Because each bar has its own colour without a shared `label=`, `ax.legend()` would produce an empty legend (no artists with labels). To create a meaningful legend explaining the colour coding (red=overfitting, green=good generalisation), the developer manually creates `Patch` proxy artists. `Patch(color=..., label=...)` creates a filled rectangle of the specified colour that serves as a legend proxy — it visually matches the bar colour and provides the descriptive label text. This is the standard technique for colour-coded legends without per-bar labels.
- **Option C (Incorrect):** Custom `Patch` objects are not required for all `barh` legends. When `label=` is used in `barh()` (e.g., for grouped horizontal bars with distinct labels per series), automatic legend generation works correctly.
- **Option D (Incorrect):** `loc="lower right"` is a universal legend location parameter available for both automatic (`ax.legend()`) and custom (`ax.legend(handles=...)`) legends. The choice of `loc=` is independent of whether handles are custom or automatic.

</details>

---

### 94. Question

A developer extends the ML report to add a **third panel** showing feature importances as a sorted horizontal bar chart. Which complete, correct code produces the correctly sorted descending horizontal bar chart with labels?

```python
feature_names = ["Avg Purchase", "Days Active", "Support Tickets",
                 "Email Opens", "Login Frequency"]
importances   = np.array([0.31, 0.24, 0.18, 0.15, 0.12])

fig, ax = plt.subplots(figsize=(8, 4))
# ... correct code goes here ...
```

- [ ] **A.** 
```python
ax.barh(feature_names, importances)
ax.invert_yaxis()
```

- [ ] **B.**
```python
idx = np.argsort(importances)
ax.barh(np.array(feature_names)[idx], importances[idx], color="darkorange")
ax.invert_yaxis()
```

- [ ] **C.**
```python
idx = np.argsort(importances)[::-1]
ax.barh(np.array(feature_names)[idx], importances[idx], color="darkorange")
```

- [ ] **D.**
```python
idx = np.argsort(importances)[::-1]
ax.barh(np.array(feature_names)[idx], importances[idx], color="darkorange")
ax.invert_yaxis()
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `np.argsort` is not called — the features are plotted in their original order (already descending since importances=[0.31, 0.24, ...]). `invert_yaxis()` reverses the bottom-to-top ordering of `barh`, placing the last item ("Login Frequency", lowest importance) at the top. The chart would show features in ASCENDING order from bottom to top — opposite of the desired descending display.
- **Option B (Incorrect):** `np.argsort(importances)` without `[::-1]` sorts ASCENDING (smallest first). After `invert_yaxis()`, the smallest feature (0.12) ends up at the top and the largest (0.31) at the bottom — this is ascending order when read top-to-bottom. Not the desired descending order.
- **Option C (Incorrect):** `np.argsort(importances)[::-1]` correctly sorts descending. `ax.barh` places the first sorted item (highest importance, 0.31, "Avg Purchase") at the BOTTOM by default. Without `invert_yaxis()`, the most important feature appears at the bottom — the opposite of convention (most important at top).
- **Option D (Correct):** Step 1: `idx = np.argsort(importances)[::-1]` → indices for descending sort: most important first. Step 2: `ax.barh(features[idx], importances[idx])` → barh places first item ("Avg Purchase", 0.31) at the BOTTOM. Step 3: `ax.invert_yaxis()` → flips the y-axis so the first item (most important) appears at the TOP. Combined effect: most important feature at top, least important at bottom — the standard convention for feature importance charts.

</details>

---

### 95. Question

A developer creates a multi-metric grouped bar chart and wants to highlight the **best-performing model per metric** by adjusting the bar's edge colour:

```python
metrics    = ["Accuracy", "F1", "Recall", "AUC"]
lr_scores  = [0.85, 0.83, 0.81, 0.87]
rf_scores  = [0.93, 0.91, 0.89, 0.94]
xgb_scores = [0.92, 0.94, 0.92, 0.93]

x = np.arange(4)
width = 0.25

all_scores = np.array([lr_scores, rf_scores, xgb_scores])   # shape (3, 4)
best_mask  = (all_scores == all_scores.max(axis=0))         # shape (3, 4)
```

What does `all_scores.max(axis=0)` compute, and what shape does `best_mask` have?

- [ ] **A.** `max(axis=0)` computes the max across all 12 values (a scalar); `best_mask` has shape (3, 4)
- [ ] **B.** `max(axis=0)` computes the max score for each model across metrics, resulting in shape (3,); `best_mask` has shape (3, 4)
- [ ] **C.** `max(axis=0)` computes the maximum score across the 3 models at each metric, resulting in shape (4,); `best_mask` has shape (3, 4) — True where a model is best for that metric
- [ ] **D.** `max(axis=0)` computes the max across metrics for each model, resulting in shape (3,); `best_mask` shape is (3,)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `max(axis=0)` does NOT compute the overall maximum. `np.ndarray.max()` without `axis=` gives a scalar. With `axis=0`, it reduces along the first axis.
- **Option B (Incorrect):** `axis=0` reduces along the FIRST dimension (rows = models). The result has shape equal to the remaining dimensions. For a (3,4) array, `max(axis=0)` gives shape **(4,)** — the max at each of the 4 column positions (metrics). Shape (3,) would result from `max(axis=1)`.
- **Option C (Correct):** `all_scores.max(axis=0)` reduces across `axis=0` (the 3 models), computing the maximum at each of the 4 column positions (metrics): `[max(0.85,0.93,0.92), max(0.83,0.91,0.94), max(0.81,0.89,0.92), max(0.87,0.94,0.93)]` = `[0.93, 0.94, 0.92, 0.94]`. Shape is **(4,)**. `best_mask = (all_scores == [0.93, 0.94, 0.92, 0.94])` broadcasts comparison across the (3,4) array, producing a Boolean (3,4) array where True marks the best model for each metric.
- **Option D (Incorrect):** `axis=0` reduces the first dimension (models), giving shape (4,) not (3,). Shape (3,) results from reducing `axis=1` (metrics). `best_mask` is always (3,4) regardless of how axis is specified because it's a comparison between (3,4) and (4,) which broadcasts.

</details>

---

### 96. Question

A developer uses `ax.containers` to programmatically access all bar containers after creating a grouped chart:

```python
fig, ax = plt.subplots()
b1 = ax.bar(x - 0.2, vals_A, 0.35, label="A")
b2 = ax.bar(x + 0.2, vals_B, 0.35, label="B")

# Later in the code:
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
```

What does iterating over `ax.containers` achieve here?

- [ ] **A.** It applies `bar_label` to every bar in both containers (A and B series) in a single loop, without needing to call `bar_label` separately for `b1` and `b2`
- [ ] **B.** It creates a new copy of each container
- [ ] **C.** `ax.containers` only contains the most recently added container
- [ ] **D.** It raises an error because `ax.containers` is not iterable

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `ax.containers` is a list of all container artists (BarContainers, ErrorbarContainers, etc.) added to the axes in the order they were created. After two `ax.bar()` calls, `ax.containers = [b1, b2]`. Iterating over `ax.containers` and calling `ax.bar_label(container, ...)` for each is equivalent to calling `ax.bar_label(b1, ...)` then `ax.bar_label(b2, ...)` separately. This is a clean, scalable pattern — if you add a third series later, the loop automatically labels it without code changes.
- **Option B (Incorrect):** Iterating over `ax.containers` does NOT copy containers. It provides references to the existing `BarContainer` objects. No duplication occurs.
- **Option C (Incorrect):** `ax.containers` accumulates ALL containers added to the axes — not just the most recent one. Each `ax.bar()` call appends a new `BarContainer` to this list.
- **Option D (Incorrect):** `ax.containers` is a standard Python list — fully iterable. It raises no error.

</details>

---

### 97. Question

A developer creates a bar chart and wants the y-axis to display a **percentage formatter** (e.g., "85%", "92%") for values stored as decimals (0.85, 0.92):

```python
import matplotlib.ticker as mticker

fig, ax = plt.subplots()
ax.bar(models, val_acc, color="steelblue")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_ylim(0, 1.1)
```

What does `mticker.PercentFormatter(xmax=1.0)` do?

- [ ] **A.** Multiplies all y-axis values by 100 and appends a "%" sign for display — 0.85 displays as "85%"
- [ ] **B.** Clips all y-axis values to the range [0, 1.0]
- [ ] **C.** Changes the y-axis unit to percentages by adding a "%" label to the axis title
- [ ] **D.** Converts stored values from floats to integers before displaying

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `mticker.PercentFormatter(xmax=1.0)` is a tick formatter that converts values to percentages for DISPLAY only. `xmax=1.0` tells the formatter that the maximum of the scale corresponds to 100%. For a value of 0.85, the formatter computes: `display_value = (0.85 / 1.0) × 100 = 85`, and appends `%` → `"85%"`. The underlying data and y-axis coordinates remain as decimals; only the tick LABELS are changed to percentage strings. This is purely a cosmetic display change.
- **Option B (Incorrect):** `PercentFormatter` is a DISPLAY formatter, not a data clipper. It does not modify data values or the axis range. Clipping would require `ax.set_ylim(0, 1.0)`.
- **Option C (Incorrect):** `PercentFormatter` modifies tick LABEL strings (from "0.85" to "85%"), not the y-axis title. Adding a "%" unit to the axis title would require `ax.set_ylabel("Accuracy (%)")`.
- **Option D (Incorrect):** The formatter does not convert floats to integers in storage. It computes and formats the display string (e.g., "85%" or "85.0%") without modifying the underlying data.

</details>

---

### 98. Question

A developer builds a complete report and wants to add **hatching patterns** to bars to make the chart accessible in black-and-white print:

```python
fig, ax = plt.subplots()
b1 = ax.bar(x - width/2, train_f1, width, hatch="//", color="lightblue",
            edgecolor="black", label="Train")
b2 = ax.bar(x + width/2, val_f1,   width, hatch="xx", color="lightsalmon",
            edgecolor="black", label="Val")
```

What does `hatch="//"` do?

- [ ] **A.** Draws two parallel diagonal lines as the bar's face colour
- [ ] **B.** Fills the bar interior with a repeating diagonal forward-slash (`/`) hatch pattern — useful for differentiating bars in black-and-white print where colour alone cannot distinguish series
- [ ] **C.** Draws the bars in the shape of forward slashes
- [ ] **D.** Creates a hash symbol (`#`) pattern inside the bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `hatch` creates a fill PATTERN drawn over the bar's face colour (set by `color=`). The pattern does not replace or redefine the face colour — it overlays diagonal lines on top of the fill.
- **Option B (Correct):** `hatch="//"` fills each bar rectangle with repeating forward-diagonal-slash lines (`/////`). The density of the pattern increases with repeated characters: `"/"` is less dense than `"//"`. Matplotlib supports hatch patterns including `/`, `\\`, `|`, `-`, `+`, `x`, `o`, `O`, `.`, `*`. Combined with `edgecolor="black"` (which also colours the hatch lines), the bars have a clearly visible pattern even when printed in black and white, making the chart accessible to colour-blind readers and usable in monochrome publications.
- **Option C (Incorrect):** `hatch` is an interior fill PATTERN — the bar retains its rectangular shape. The slash pattern is a texture inside the rectangle, not the shape of the bar itself.
- **Option D (Incorrect):** `hatch="#"` (not `"//"`) produces a hash/grid pattern. The `"//"` pattern produces diagonal lines.

</details>

---

### 99. Question

A developer creates the following code to show class-specific precision bars from a multiclass classification report. A reviewer notes a potential issue:

```python
report_dict = {
    "cat":      {"precision": 0.91, "recall": 0.88, "f1-score": 0.895},
    "dog":      {"precision": 0.87, "recall": 0.93, "f1-score": 0.899},
    "bird":     {"precision": 0.94, "recall": 0.85, "f1-score": 0.893},
    "fish":     {"precision": 0.89, "recall": 0.91, "f1-score": 0.900},
    "accuracy": {"precision": 0.90, "recall": 0.90, "f1-score": 0.90},   # Issue?
    "macro avg": {"precision": 0.903, "recall": 0.893, "f1-score": 0.897},
}

classes = list(report_dict.keys())
precision = [report_dict[c]["precision"] for c in classes]

fig, ax = plt.subplots()
ax.bar(classes, precision, color="steelblue")
```

What is the **most significant data-validity issue** with this chart?

- [ ] **A.** "accuracy" and "macro avg" should not be included as class labels — they are aggregate summary statistics, not individual classes; including them as bars alongside "cat", "dog", "bird", "fish" misleadingly implies they represent another data class
- [ ] **B.** `precision` values cannot exceed 0.9 on a bar chart
- [ ] **C.** The bars should be horizontal because there are 6 categories
- [ ] **D.** `report_dict` should use integer keys instead of strings

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When iterating over all keys in `report_dict`, "accuracy" and "macro avg" (common entries in scikit-learn's `classification_report(output_dict=True)`) are included alongside the per-class entries. However, "accuracy" represents the overall fraction of correct predictions across ALL classes — not a class-specific precision. Plotting it as a bar alongside class-specific precision bars implies it's a fifth or sixth class metric, which is misleading. The bar for "accuracy" (precision=0.90) looks identical in format to the class bars but measures a different concept. The correct fix is to filter the dictionary: `classes = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]`.
- **Option B (Incorrect):** Precision values CAN exceed 0.9. They range from 0 to 1.0. There is no bar chart constraint preventing precision values in the 0.9+ range.
- **Option C (Incorrect):** While horizontal bars (`barh`) are recommended for long category names, "cat", "dog", "bird", "fish", "accuracy", "macro avg" are relatively short. The chart type choice is a style concern, not a data validity issue. The reviewer's significant concern is the mixed metadata categories.
- **Option D (Incorrect):** String dictionary keys are valid and commonly used for category labels. Integer keys would lose the descriptive class names.

</details>

---

### 100. Question

A developer creates the final, production-ready ML benchmark bar chart. Which of the following statements about best practices are **TRUE**? (Select all that apply)

```python
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(models, f1_scores, color=colors, width=0.6,
              edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Model Benchmark — F1-Score Comparison", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=0)
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
plt.tight_layout()
plt.savefig("benchmark.pdf", bbox_inches="tight")
```

- [ ] **A.** `ax.set_ylim(0, 1.1)` sets the y-axis from 0 to 1.1 — starting at 0 ensures honest proportional bar lengths, and extending to 1.1 provides clearance above bars for `bar_label` text
- [ ] **B.** `zorder=0` on the gridlines ensures they render BEHIND the bars, preventing gridlines from appearing on top of bar fill
- [ ] **C.** Saving as PDF produces a vector-format file where bars, spines, and text are stored as mathematical paths — scalable to any resolution without pixelation
- [ ] **D.** `edgecolor="white"` with `linewidth=0.5` creates a subtle white border between adjacent bars that enhances visual separation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C, D**

#### Explanations:

- **Option A (True):** `ax.set_ylim(0, 1.1)` correctly starts at 0 (honest baseline for bar charts — proportional bar lengths reflect actual F1 scores from zero). Extending to 1.1 (10% above the maximum possible F1 of 1.0) ensures that bar labels added by `ax.bar_label()` with `padding=5` are not clipped at the top edge of the plot area.
- **Option B (True):** `zorder=0` assigns gridlines the lowest drawing order (behind all other artists whose default zorder is higher). Bars have a default zorder of ~2. By explicitly setting `zorder=0` on the grid, the horizontal dashed lines render first and bars overlay them, preventing gridlines from appearing on top of bar fill (which would make the bars look cut or meshed).
- **Option C (True):** PDF (Portable Document Format) is a vector graphics format. All Matplotlib artists — bar rectangles, spine lines, tick marks, text labels, legend elements — are stored as PDF vector path operations (Bézier curves, lines, filled polygons). The resulting file scales losslessly to any resolution — 72 DPI for screen, 600 DPI for print, any size poster — without pixelation.
- **Option D (True):** `edgecolor="white"` draws a white border around each bar. With `linewidth=0.5` (a half-point-wide stroke), this creates a thin white gap between bars. On a white background, this has minimal effect — but when bars are coloured (as they are here), the white border creates a subtle but effective visual separation between adjacent bars in a multi-colour chart, preventing the bars from appearing merged into a continuous colour block.

</details>
