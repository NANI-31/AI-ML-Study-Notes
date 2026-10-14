# Matplotlib Bar Plot Objective Test — Set 09

---

### 81. Question

A developer creates a complete model evaluation script with the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

models    = ["LR", "RF", "XGB", "NN"]
train_acc = np.array([0.862, 0.981, 0.973, 0.968])
val_acc   = np.array([0.853, 0.878, 0.941, 0.919])
gap       = train_acc - val_acc

x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: grouped bars
b_tr = axes[0].bar(x - width/2, train_acc, width, label="Train", color="#2980B9")
b_vl = axes[0].bar(x + width/2, val_acc,   width, label="Val",   color="#E74C3C")
axes[0].set_xticks(x, models)
axes[0].set_ylim(0, 1.1)
axes[0].legend()
axes[0].set_title("Train vs Validation Accuracy")
axes[0].bar_label(b_tr, fmt="%.3f", padding=2, fontsize=8)
axes[0].bar_label(b_vl, fmt="%.3f", padding=2, fontsize=8)

# Right panel: generalization gap bars
gap_colors = ["#E74C3C" if g > 0.05 else "#2ECC71" for g in gap]
axes[1].bar(models, gap, color=gap_colors, width=0.5)
axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1.5, label="5% Threshold")
axes[1].set_title("Generalization Gap (Train − Val)")
axes[1].set_ylabel("Gap")
axes[1].legend()

plt.tight_layout()
```

Which models have a generalization gap **exceeding** 0.05 and therefore appear in **red** in the right panel?

- [ ] **A.** RF only (gap = 0.103)
- [ ] **B.** RF and NN (gaps = 0.103 and 0.049)
- [ ] **C.** LR and RF (gaps = 0.009 and 0.103)
- [ ] **D.** RF and NN (gaps = 0.103 and 0.049) — NN gap is 0.049, which is below 0.05 so only RF is red

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Computing gaps: LR = 0.862−0.853 = **0.009** (green, < 0.05). RF = 0.981−0.878 = **0.103** (red, > 0.05). XGB = 0.973−0.941 = **0.032** (green, < 0.05). NN = 0.968−0.919 = **0.049** (green, < 0.05; condition is `> 0.05`, not `>= 0.05`). Only RF exceeds 0.05, so only RF's bar is red.
- **Option B (Incorrect):** NN's gap = 0.049 < 0.05, so it is NOT red. The condition `g > 0.05` requires strictly greater than, not equal to or above. NN (0.049) is below the threshold.
- **Option C (Incorrect):** LR's gap = 0.009 < 0.05 → green. RF's gap = 0.103 > 0.05 → red. LR does not exceed the threshold.
- **Option D (Incorrect):** Option D starts correctly by noting that RF gap=0.103 and NN gap=0.049, but then gives the wrong conclusion ("both RF and NN"). The correct conclusion is that ONLY RF exceeds 0.05. The last clause of option D contradicts itself.

</details>

---

### 82. Question

A developer uses `ax.bar_label()` in a multi-series chart but also wants to add a **unit suffix** to each label (e.g., "0.891 acc"):

```python
bars = ax.bar(models, accuracies, width=0.4, color="steelblue")
ax.bar_label(bars, labels=[f"{v:.3f} acc" for v in accuracies], padding=4)
```

Which statement accurately describes this technique?

- [ ] **A.** `f"{v:.3f} acc"` is invalid Python string formatting — a `TypeError` is raised
- [ ] **B.** Using `labels=` with a list of pre-formatted strings is the correct technique to add custom suffixes/units; `fmt=` cannot add suffix text, but `labels=` with f-strings can
- [ ] **C.** The suffix `" acc"` is automatically stripped by `bar_label` before rendering
- [ ] **D.** This approach only works when `label_type="center"` is used

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `f"{v:.3f} acc"` is valid Python f-string syntax. It formats `v` as a float with 3 decimal places and appends the string `" acc"`. This produces strings like `"0.891 acc"`.
- **Option B (Correct):** `fmt=` in `ax.bar_label()` is a printf-style format string applied to the numeric bar heights — it cannot add free-form text suffixes beyond what `%`-formatting allows. However, `labels=` accepts a pre-built list of strings with ANY format, including f-strings with units, currency symbols, or contextual text. This is the documented approach for fully customised bar labels. The list comprehension `[f"{v:.3f} acc" for v in accuracies]` creates the exact strings to display above each bar.
- **Option C (Incorrect):** `bar_label` renders the label strings exactly as provided in the `labels=` list. There is no stripping or modification of the provided text. What you pass is what gets rendered.
- **Option D (Incorrect):** Custom `labels=` works with both `label_type="edge"` (default, above bar top) and `label_type="center"` (inside bar midpoint). The labeling mode and the label content are independent parameters.

</details>

---

### 83. Question

A developer creates a comprehensive grouped bar chart for 4 models × 4 metrics. After running the code, the developer discovers that all bars for "Recall" (index 2) are shifted 0.25 units too far to the right. Examining the code reveals:

```python
x = np.arange(len(metrics))    # [0, 1, 2, 3]
width = 0.2

# 4 models: should be at -1.5w, -0.5w, +0.5w, +1.5w = -0.3, -0.1, +0.1, +0.3
b1 = ax.bar(x - 1.5*width, lr,   width)
b2 = ax.bar(x - 0.5*width, rf,   width)
b3 = ax.bar(x + 0.5*width, svm,  width)
b4 = ax.bar(x + 1.5*width, xgb,  width)

ax.set_xticks(x + 0.25, metrics)   # Bug: offset tick labels by +0.25
```

What causes the perceived shift and how should it be fixed?

- [ ] **A.** The `width=0.2` is too small; use `width=0.25` to align bars with ticks
- [ ] **B.** `ax.set_xticks(x + 0.25, metrics)` places tick marks at x+0.25 instead of x, making the labels appear shifted 0.25 units to the right relative to the bar groups; fix: use `ax.set_xticks(x, metrics)`
- [ ] **C.** The bug is in `x - 1.5*width` which should be `x - 2*width` for 4 bars
- [ ] **D.** The `metrics` list must be reversed to align with bar groups

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Changing `width` from 0.2 to 0.25 would change bar widths and potentially cause overlap, but would not fix the tick alignment issue. The bars themselves are correctly positioned around integer x-values [0,1,2,3].
- **Option B (Correct):** `ax.set_xticks(x + 0.25, metrics)` places tick marks at positions [0.25, 1.25, 2.25, 3.25] instead of [0, 1, 2, 3]. This makes the category labels (and tick marks) appear 0.25 units to the RIGHT of where the bar groups are centered. The fix is `ax.set_xticks(x, metrics)` — tick positions must match the bar group center positions `x = [0, 1, 2, 3]`.
- **Option C (Incorrect):** `x - 1.5*width = x - 0.3` correctly positions the leftmost (LR) bar 0.3 units left of center. For 4 equally-spaced bars with width 0.2, the offsets `{-1.5w, -0.5w, +0.5w, +1.5w}` are mathematically correct — bars are contiguous with no gaps: LR right edge = x-0.3+0.2 = x-0.1, RF left edge = x-0.5*0.2 = x-0.1 ✓.
- **Option D (Incorrect):** The `metrics` label ordering is correct — it matches the x-axis left-to-right ordering. Reversing it would misalign labels with bar groups.

</details>

---

### 84. Question

A developer creates a Matplotlib bar chart in a Jupyter Notebook and uses `%matplotlib inline`. The developer updates the bar data and calls `plt.bar(...)` again but in the SAME cell. What happens?

- [ ] **A.** The previous chart is updated with the new data in place
- [ ] **B.** A second, NEW bar chart is created below the first — both charts are displayed
- [ ] **C.** An error is raised because `plt.bar()` can only be called once per cell
- [ ] **D.** The second call adds bars to the SAME axes as the first call, creating a grouped or overlapping bar chart

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.bar()` does NOT update existing bars. It ADDS new `Rectangle` artists to the current axes. There is no in-place update mechanism via repeated calls.
- **Option B (Incorrect):** A second chart is not created automatically. `plt.bar()` continues to operate on the CURRENT active figure and axes (the one created by the first call or the implicit current axes). In `%matplotlib inline`, each cell displays the figure at the end of cell execution — but within the cell, both calls operate on the same axes.
- **Option C (Incorrect):** There is no restriction on calling `plt.bar()` multiple times. As demonstrated throughout this test, multiple `ax.bar()` calls are the standard mechanism for grouped and stacked bar charts.
- **Option D (Correct):** When `plt.bar()` is called twice in the same cell without `plt.figure()` or `plt.cla()` between them, both calls add bars to the SAME current axes. If the same x-positions are used, the second set of bars is drawn on top of (overlapping) the first — making the first bars invisible if widths are equal and positions identical. If different offsets are used (like grouped bar code), both series coexist on the same axes as intended.

</details>

---

### 85. Question

A developer creates a production-quality bar chart for a presentation and adds the following final styling:

```python
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="both", which="both", length=0)   # Remove tick marks
ax.set_facecolor("#F8F9FA")                            # Light grey background
fig.patch.set_facecolor("#FFFFFF")                     # White figure background
```

What does `ax.tick_params(axis="both", which="both", length=0)` achieve?

- [ ] **A.** It hides all tick label text on both axes
- [ ] **B.** It removes the small tick-mark lines on both x and y axes while keeping the tick label text visible
- [ ] **C.** It sets the font size of tick labels to 0 (invisible)
- [ ] **D.** It applies to both major and minor ticks of both axes, setting their line length to 0 — effectively hiding the tick marks but preserving labels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Hiding tick label TEXT requires `ax.tick_params(labelbottom=False, labelleft=False)` or `ax.set_xticklabels([])`. The `length=0` parameter controls the physical tick mark line length, not text visibility.
- **Option B (Correct as far as it goes, but incomplete):** The tick-mark lines (the short perpendicular lines extending from the axis spines at each tick position) are hidden by `length=0`. Tick labels remain visible. However, Option D gives a more complete and precise description.
- **Option C (Incorrect):** Font size zero would make text invisible, but the mechanism is `labelsize=` not `length=`. Setting `length=0` does nothing to font rendering.
- **Option D (Correct — most precise):** `axis="both"` applies to both x and y axes. `which="both"` applies to both major ticks (at labeled positions) and minor ticks (subdivisions between major ticks). `length=0` sets the tick line length to 0 points — the small perpendicular lines at each tick mark become invisible. Tick labels (numbers/strings) remain unaffected. This creates a "clean" chart style where the data space boundary spines (or none) define the chart without distracting tick marks. Combined with removing top/right spines, this is a popular modern chart style.

</details>

---

### 86. Question

A developer creates a bar chart where bar colours need to match a continuous colourmap (e.g., "viridis") based on relative bar heights:

```python
scores = np.array([0.62, 0.81, 0.74, 0.93, 0.55, 0.88])
labels = ["M1", "M2", "M3", "M4", "M5", "M6"]

import matplotlib.cm as cm
import matplotlib.colors as mcolors

norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
colors = cm.viridis(norm(scores))

fig, ax = plt.subplots()
bars = ax.bar(labels, scores, color=colors, width=0.6)
```

What does `mcolors.Normalize(vmin=scores.min(), vmax=scores.max())` accomplish?

- [ ] **A.** It scales the `scores` array to the range [0, 1] so they can index into the colourmap
- [ ] **B.** It creates a normaliser object that maps scalar values in `[scores.min(), scores.max()]` to [0, 1] — used as input to `cm.viridis()` to get RGBA colour tuples
- [ ] **C.** It clips all scores outside the range to either the minimum or maximum
- [ ] **D.** It normalises the bar heights so all bars sum to 1

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Partially correct but imprecise):** The normaliser does map values to [0,1], but calling `norm(scores)` doesn't directly rescale the scores array in-place. It creates a masked array of normalised values used for colour mapping. The description in Option B is more technically accurate.
- **Option B (Correct):** `mcolors.Normalize(vmin, vmax)` creates a callable normaliser. When called as `norm(scores)`, it maps each score from the range `[vmin, vmax]` to `[0.0, 1.0]` using linear interpolation: `normalized = (score - vmin) / (vmax - vmin)`. These normalised values [0,1] are then passed to `cm.viridis(...)`, which maps each [0,1] float to an RGBA colour tuple from the "viridis" colourmap. The lowest score (0.55) maps to viridis's purple/dark blue, and the highest (0.93) maps to viridis's yellow-green.
- **Option C (Incorrect):** Clipping (clamping values outside vmin/vmax to vmin or vmax) is the behaviour of `mcolors.Normalize` when values fall outside the range. But the primary purpose is normalisation to [0,1] — not clipping. Here, all scores are between vmin and vmax so clipping doesn't occur.
- **Option D (Incorrect):** Normalising so all bars sum to 1 would be a sum-based normalisation (proportion of total), computed as `scores / scores.sum()`. `mcolors.Normalize` is for colour mapping, not statistical normalisation of data.

</details>

---

### 87. Question

A developer creates a complete ML model comparison script and asks a colleague to review it. The colleague points out that the script is missing error handling for empty data. Which of the following code patterns would correctly guard against an empty models list?

```python
models    = []   # Empty list scenario
scores    = []
```

- [ ] **A.** `if len(models) == 0: plt.bar([], [])`
- [ ] **B.** `if not models: plt.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha="center"); return`
- [ ] **C.** `assert len(models) > 0, "No model data provided"`
- [ ] **D.** All of B and C are valid defensive patterns; B provides a user-friendly fallback visual, C raises an early error for debugging

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.bar([], [])` with empty lists creates an empty bar chart with no bars and no labels. While it doesn't crash, it provides no feedback to the user that data is missing and doesn't prevent further code from running (which might fail on subsequent operations like `bar_label`).
- **Option B (Valid):** `if not models:` catches an empty list, empty string, or None. Displaying a "No Data" message as centred text on the axes using `transform=ax.transAxes` (which uses axes coordinates 0–1) is a user-friendly fallback that prevents the chart from appearing broken. The `return` exits the function gracefully.
- **Option C (Valid):** `assert len(models) > 0` raises an `AssertionError` immediately with a descriptive message when data is missing. This is appropriate in development/debugging contexts where the programmer needs to know about invalid data immediately. However, `assert` statements can be disabled with Python's `-O` (optimize) flag, making them unreliable in production; `raise ValueError(...)` is preferred for production guard clauses.
- **Option D (Correct):** Both B and C are legitimate defensive programming patterns for different contexts. B is user-facing (graceful degradation), C is developer-facing (early failure detection). Together they represent the two standard approaches to empty data handling. D correctly identifies both as valid.

</details>

---

### 88. Question

A data scientist has a DataFrame `df` with columns `["Model", "Accuracy", "F1"]` and wants to plot a grouped bar chart. A colleague suggests:

```python
df.set_index("Model")[["Accuracy", "F1"]].plot(kind="bar", ax=ax)
```

Compared to the manual `np.arange` + `ax.bar()` + `ax.set_xticks()` approach, which statement about Pandas `.plot(kind="bar")` is most accurate?

- [ ] **A.** Pandas `plot(kind="bar")` produces identical output to the manual approach in all cases
- [ ] **B.** Pandas `plot(kind="bar")` automatically handles grouping, tick labeling, and legend — but gives less fine-grained control over bar offsets, colours per-series, and width calculation compared to the manual approach
- [ ] **C.** Pandas `plot(kind="bar")` is always preferred because it requires fewer lines of code
- [ ] **D.** The manual approach is always preferred because Pandas `plot` produces incorrect tick labels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While Pandas `.plot(kind="bar")` and the manual approach both produce grouped bar charts, the outputs are not identical. Default colours, bar widths, spacing, and styling differ. They are functionally similar but not visually identical.
- **Option B (Correct):** Pandas `.plot(kind="bar", ax=ax)` is a high-level convenience wrapper that: (1) automatically creates grouped bars for multiple columns, (2) uses the DataFrame index as x-axis labels (no `set_xticks` needed), (3) adds a legend from column names. The trade-off is reduced control — fine-tuning bar widths (`width=`), per-bar colours (colour must be a list applied to groups), asymmetric error bars, and complex multi-layer stacking require more effort or switching to the manual approach. For production-quality publication charts, the manual approach offers more precision; for quick exploratory visualization, Pandas plot is more convenient.
- **Option C (Incorrect):** "Always preferred" is too strong. Pandas plot is convenient for quick exploration, but for publication-quality charts requiring precise control, the manual approach is often necessary. The appropriate choice depends on the use case.
- **Option D (Incorrect):** Pandas `.plot(kind="bar")` handles tick labels correctly — the DataFrame index is used as category labels automatically and correctly. The tick labels are not "incorrect."

</details>

---

### 89. Question

A developer creates a function to generate a reusable bar chart component:

```python
def model_bar_chart(ax, categories, values, title,
                    color="steelblue", ylim=(0, 1),
                    show_labels=True, label_fmt="%.2f"):
    bars = ax.bar(categories, values, color=color, width=0.6,
                  edgecolor="white", linewidth=0.5)
    if show_labels:
        ax.bar_label(bars, fmt=label_fmt, padding=3, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return bars
```

A developer calls this function with `show_labels=False` and `ylim=(0, 100)`. What changes from the default behaviour?

- [ ] **A.** The bars are hidden and only the title is shown
- [ ] **B.** The numeric labels above bars are suppressed, and the y-axis spans 0–100 instead of the default 0–1
- [ ] **C.** The chart is exported as a 100 DPI PNG
- [ ] **D.** The title font size changes to 100

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `show_labels=False` suppresses the `ax.bar_label()` call only — the bars themselves are always drawn by the `ax.bar()` call which runs unconditionally. The bars remain visible.
- **Option B (Correct):** Two parameters change the default behaviour: (1) `show_labels=False` — the `if show_labels:` condition is False, so `ax.bar_label()` is NOT called, and no numeric labels appear above bars. This is useful for charts where exact values are irrelevant or where the chart is used in a small panel. (2) `ylim=(0, 100)` — `ax.set_ylim(*ylim)` unpacks to `ax.set_ylim(0, 100)`, expanding the y-axis to a 0–100 scale instead of the default 0–1. This would be appropriate when `values` are percentages (0–100 range) rather than proportions (0–1 range).
- **Option C (Incorrect):** There is no DPI or export parameter in this function. `plt.savefig()` or similar would be needed externally with explicit DPI specification.
- **Option D (Incorrect):** The title font size is hardcoded as `fontsize=12` inside the function, regardless of what `ylim=` is set to. The `100` in `ylim=(0, 100)` is the y-axis upper limit, not a font size.

</details>

---

### 90. Question

A developer creates a bar chart and formats values using `ax.bar_label` with different format strings. Given bar heights: `[0.8523, 0.9104, 0.7867, 0.9231]`, which `fmt` string produces the output `["85.2", "91.0", "78.7", "92.3"]`?

- [ ] **A.** `fmt="%.1f%%"` applied to raw proportions
- [ ] **B.** `fmt="%.1f"` applied after pre-multiplying values by 100
- [ ] **C.** `fmt="%.3f"` applied to raw proportions
- [ ] **D.** `fmt="%d"` applied after pre-multiplying values by 100

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `fmt="%.1f%%"` on raw proportions (0.8523, etc.) would produce "0.9%", "0.9%", "0.8%", "0.9%" — the `%.1f` of values like 0.8523 gives "0.9", and `%%` adds the `%` symbol. The output would show "0.9%" not "85.2".
- **Option B (Correct):** Pre-multiply: `values_pct = [0.8523*100, 0.9104*100, 0.7867*100, 0.9231*100] = [85.23, 91.04, 78.67, 92.31]`. Then `fmt="%.1f"` applied to these values: "85.2", "91.0", "78.7", "92.3". The `%.1f` rounds each to 1 decimal place. Passing these pre-multiplied values to `ax.bar()` and using `fmt="%.1f"` in `bar_label` produces the desired output.
- **Option C (Incorrect):** `fmt="%.3f"` on raw proportions gives "0.852", "0.910", "0.787", "0.923" — three decimal places of the raw proportions, not percentages.
- **Option D (Incorrect):** `fmt="%d"` on values [85.23, 91.04, 78.67, 92.31] rounds to integers: "85", "91", "79", "92" — no decimal places. The desired output requires one decimal place.

</details>
