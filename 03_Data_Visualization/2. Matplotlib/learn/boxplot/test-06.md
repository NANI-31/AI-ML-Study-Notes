# Matplotlib Box Plot Objective Test — Set 06

---

### 51. Question

A developer creates a box plot for a dataset and accesses the returned artist dictionary to find the **number of cap Line2D objects** for a 4-group plot:

```python
bp = ax.boxplot([g1, g2, g3, g4], patch_artist=True)
print(len(bp["caps"]))
```

What does this print?

- [ ] **A.** 4
- [ ] **B.** 8
- [ ] **C.** 16
- [ ] **D.** 2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 4 caps would only apply if each group had ONE cap (just the upper or just the lower). Each group has two caps — one at the lower whisker tip and one at the upper whisker tip.
- **Option B (Correct):** Each box plot group generates **2 cap Line2D objects**: one for the lower whisker end and one for the upper whisker end. With 4 groups: 4 × 2 = **8 caps**. Similarly, `bp["whiskers"]` has 8 elements and `bp["medians"]` has 4 elements. This follows the consistent pattern: per-group elements (boxes, medians) = N groups; per-whisker elements (whiskers, caps) = 2N groups.
- **Option C (Incorrect):** 16 caps would imply 4 caps per group, which doesn't correspond to any standard box plot structure.
- **Option D (Incorrect):** 2 caps would apply to a single-group box plot (one lower cap + one upper cap = 2). For 4 groups, multiply by 4.

</details>

---

### 52. Question

A researcher creates the following box plot code and asks what `bp["fliers"]` contains when there are **no outliers** in any of the 3 groups:

```python
data = [[20, 22, 24, 26, 28],
        [30, 32, 34, 36, 38],
        [40, 42, 44, 46, 48]]

bp = ax.boxplot(data)
print(len(bp["fliers"]))
for f in bp["fliers"]:
    print(f.get_xdata(), f.get_ydata())
```

What does this output?

- [ ] **A.** `len = 0` — fliers list is empty when there are no outliers
- [ ] **B.** `len = 3` — one flier Line2D per group (may contain empty data arrays when there are no outliers)
- [ ] **C.** `len = 6` — two flier objects per group (upper and lower)
- [ ] **D.** A `KeyError` — `"fliers"` key doesn't exist when no outliers are present

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Even when no outliers exist, Matplotlib still includes flier Line2D objects in `bp["fliers"]` — one per group. The list is never empty just because there are no outliers.
- **Option B (Correct):** `bp["fliers"]` always contains **one `Line2D` object per group** (3 groups → 3 elements). When no outliers exist for a group, that `Line2D` has **empty data arrays**: `get_xdata()` returns `[]` and `get_ydata()` returns `[]`. The artist exists but has no points to render, so it appears invisible. This consistent structure allows code to always access `bp["fliers"]` without checking for its existence.
- **Option C (Incorrect):** Unlike `whiskers` and `caps` (which have 2 per group), `fliers` has only **1 per group**. All outliers for a single group (whether upper or lower) are stored in a single Line2D object.
- **Option D (Incorrect):** The `"fliers"` key is always present in the returned dictionary, regardless of whether outliers exist. The dictionary always contains all keys: `'boxes'`, `'medians'`, `'whiskers'`, `'caps'`, `'fliers'`.

</details>

---

### 53. Question

A data scientist creates a horizontal box plot of model inference latencies:

```python
latencies_ms = {
    "BERT": [45, 52, 48, 51, 56, 49, 53, 55, 47, 120],
    "DistilBERT": [22, 25, 23, 26, 24, 27, 25, 23, 24, 26],
    "RoBERTa": [48, 55, 50, 53, 58, 51, 56, 49, 54, 52],
}

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(list(latencies_ms.values()),
           tick_labels=list(latencies_ms.keys()),
           vert=False,
           flierprops=dict(marker="x", markeredgecolor="crimson", markersize=8))
ax.set_xlabel("Inference Latency (ms)")
ax.set_title("Model Inference Latency Comparison")
```

In this **horizontal** box plot, which axis now represents the latency values?

- [ ] **A.** The y-axis — latency is always on the y-axis in box plots
- [ ] **B.** The x-axis — with `vert=False`, data values are mapped to the horizontal (x) axis
- [ ] **C.** Both axes equally
- [ ] **D.** Neither axis — `vert=False` creates a circular display

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In a VERTICAL box plot (`vert=True`, the default), data values map to the y-axis. But with `vert=False`, the axes are rotated: data values map to the x-axis. There is no rule that latency "always" appears on the y-axis.
- **Option B (Correct):** With `vert=False`, the box plot is rotated 90°. Data values (latency in ms) are now plotted along the **x-axis** (horizontal), and group names (BERT, DistilBERT, RoBERTa) are positioned along the **y-axis** (vertical). The boxes extend left and right instead of up and down. This is why `ax.set_xlabel("Inference Latency (ms)")` is correct — the label is placed on the x-axis where the latency values appear.
- **Option C (Incorrect):** Each axis has a distinct role: x-axis = data values (latency), y-axis = group labels (model names). They serve separate purposes.
- **Option D (Incorrect):** `vert=False` produces a standard horizontal rectangular box plot, not a circular display. Circular or radial visualizations require completely different chart types.

</details>

---

### 54. Question

A developer creates a multi-panel figure with box plots and a reference comparison:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

datasets = [train_errors, val_errors, test_errors]
titles   = ["Train", "Validation", "Test"]
colors   = ["#3498DB", "#E74C3C", "#2ECC71"]

for ax, data, title, color in zip(axes, datasets, titles, colors):
    bp = ax.boxplot(data, patch_artist=True)
    bp["boxes"][0].set_facecolor(color)
    ax.set_title(title)
    ax.axhline(0, color="black", linestyle="--", alpha=0.6)
```

What does `sharey=True` accomplish in this layout?

- [ ] **A.** All three panels share the same data — they plot identical distributions
- [ ] **B.** All three panels share the same y-axis scale — making error magnitude directly comparable across Train, Validation, and Test panels
- [ ] **C.** The y-axis label appears only on the leftmost panel
- [ ] **D.** Both B and C are correct effects of `sharey=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `sharey=True` links axis SCALES, not data. Each panel still plots its own independent dataset (`train_errors`, `val_errors`, `test_errors`).
- **Option B (Correct):** `sharey=True` links the y-axis limits of all subplots. If Train errors range from −10 to +25 and Test errors range from −5 to +15, without `sharey` each panel would auto-scale to its own data. With `sharey=True`, all panels use the same y-range (the union of all data ranges), making error magnitude directly visually comparable. This is essential for honest error distribution comparison.
- **Option C (Correct):** As a secondary effect of `sharey=True`, Matplotlib automatically suppresses y-axis tick labels on all subplots except the leftmost (to avoid redundancy). Y-axis tick marks appear on all, but numeric labels appear only on the first panel.
- **Option D (Correct):** Both B (shared y-scale for comparability) and C (suppressed redundant y-labels) are simultaneous effects of `sharey=True`. D correctly captures both effects.

</details>

---

### 55. Question

A developer inspects the following box plot description and must identify the **skewness type**:

- Box spans from **Q₁=40 to Q₃=80** (IQR=40)
- Median line is at **72** (very close to Q₃)
- Lower whisker is much longer than the upper whisker
- Two outlier points appear **below** the lower whisker

What type of skewness does this distribution exhibit?

- [ ] **A.** Right skew (positive skew) — the upper tail is longer
- [ ] **B.** Left skew (negative skew) — the lower tail is longer; median near Q₃ and long lower whisker confirm this
- [ ] **C.** Zero skew (symmetric) — the outliers on both sides cancel out
- [ ] **D.** Bimodal — two outlier points indicate two peaks in the data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Right skew (positive skew) would show: median close to Q₁ (bottom of box), longer UPPER whisker, and outliers above the upper whisker. The described pattern is the opposite.
- **Option B (Correct):** The characteristics — (1) median at 72, close to Q₃=80 (only 8 units below Q₃ but 32 units above Q₁=40), (2) longer lower whisker, (3) outliers below — are all hallmarks of **left-skew (negative skew)**. The majority of data is concentrated at higher values (cluster near Q₃), with a sparse tail extending toward lower values. The longer lower whisker and below-whisker outliers confirm the negative tail direction.
- **Option C (Incorrect):** Two outliers below (and none above) do not cancel with symmetric upper data. The outliers reinforce the left-skew — they extend the lower tail further. A symmetric distribution would need outliers on both sides, with equal whisker lengths and a centered median.
- **Option D (Incorrect):** Two outlier points in a box plot indicate extreme values in the tail — not bimodality. Bimodal distributions have two peaks in frequency, which a box plot cannot directly show (the box would appear similar to a wide, possibly symmetric distribution). Histograms or KDE plots are needed to detect bimodality.

</details>

---

### 56. Question

A developer creates a box plot for the following dataset of software bug response times (hours):

```
Sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 48, 72, 120]
```

Given: n=13, Q₁=3.5 (average of 3rd and 4th values after lower-half split), Q₃=10 (8th value), IQR=6.5.

Upper fence = Q₃ + 1.5×IQR = 10 + 9.75 = **19.75**. Which values appear as **fliers**?

- [ ] **A.** 48 and 72 only
- [ ] **B.** 48, 72, and 120 — all three exceed the upper fence of 19.75
- [ ] **C.** Only 120 — it's the single most extreme value
- [ ] **D.** No fliers — the whisker extends to 120

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 48, 72, and 120 are all greater than the upper fence of 19.75. All three values fall outside the whisker boundary and are classified as outliers. Overlooking 120 is wrong.
- **Option B (Correct):** Upper fence = 19.75. Values in the dataset: 48 > 19.75 ✓, 72 > 19.75 ✓, 120 > 19.75 ✓. All three are classified as outliers (fliers) and plotted as individual markers above the whisker cap. The upper whisker extends to the highest non-outlier value within the fence: **10** (the value at Q₃, which is also the 13th percentile boundary). Actually the highest non-outlier is 10 itself, since the next value 48 > 19.75. Lower fence = 3.5 − 9.75 = −6.25; minimum = 1 > −6.25, so no lower outliers.
- **Option C (Incorrect):** Box plot outlier detection applies the 1.5×IQR rule to ALL data points simultaneously. It doesn't select only the single most extreme value — every value beyond the fence is a flier. All three (48, 72, 120) are displayed as individual markers.
- **Option D (Incorrect):** The whisker does NOT extend to 120. Values 48, 72, and 120 are beyond the fence, so the whisker stops at 10 (the last non-outlier), and all three extreme values are plotted as separate flier markers.

</details>

---

### 57. Question

A developer wants to create a box plot that shows **both the median AND the mean** for each group:

```python
ax.boxplot([group1, group2, group3],
           showmeans=True,
           meanline=True,
           medianprops=dict(color="black", linewidth=2),
           meanprops=dict(color="red", linestyle="--", linewidth=1.5))
```

How do the median and mean appear visually with these settings?

- [ ] **A.** Both appear as identical horizontal black lines
- [ ] **B.** The median appears as a solid black horizontal line; the mean appears as a dashed red horizontal line
- [ ] **C.** The median appears as a red dashed line; the mean appears as a solid black line
- [ ] **D.** Only the mean appears — `showmeans=True` replaces the median display

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The two lines have different styles: `medianprops` specifies solid black; `meanprops` specifies dashed red. They would not appear identical.
- **Option B (Correct):** `medianprops=dict(color="black", linewidth=2)` styles the **median line** as solid black with 2-point width. `meanprops=dict(color="red", linestyle="--", linewidth=1.5)` styles the **mean line** as dashed red with 1.5-point width (used with `meanline=True`). Both lines appear inside the box simultaneously — the median as the primary statistics and the mean as a secondary overlay. When the distribution is skewed, the two lines are visually separated, making it easy to see the difference between median and mean.
- **Option C (Incorrect):** `medianprops` controls the MEDIAN (solid black per settings), and `meanprops` controls the MEAN (dashed red). Option C has them swapped.
- **Option D (Incorrect):** `showmeans=True` adds the mean display — it does NOT replace or remove the median. Both the median line and the mean indicator are shown simultaneously. This is one of the most useful aspects of box plots for comparing robust vs. non-robust location statistics.

</details>

---

### 58. Question

A developer creates a box plot and wants the boxes to be **wider** than the default. The default `widths` value for `ax.boxplot()` is approximately 0.5. Which code makes each box 80% of the spacing between adjacent boxes?

- [ ] **A.** `ax.boxplot(data, widths=8)`
- [ ] **B.** `ax.boxplot(data, width=0.8)` — singular `width`
- [ ] **C.** `ax.boxplot(data, widths=0.8)` — plural `widths`
- [ ] **D.** `ax.boxplot(data, boxwidth=0.8)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `widths=8` would set the boxes to 8 units wide in data coordinates. For a multi-group plot where groups are spaced 1 unit apart, this would make boxes 8× wider than the spacing, causing massive overlap. The value should be between 0 and 1 for non-overlapping boxes.
- **Option B (Incorrect):** `width=0.8` (singular) is NOT a valid parameter for `ax.boxplot()`. Matplotlib uses the **plural** form `widths`. Using `width=` would raise a `TypeError` for an unexpected keyword argument.
- **Option C (Correct):** `widths=0.8` is the correct parameter name and value. The `widths` parameter (plural) accepts a float or a list of floats specifying the width of each box. `widths=0.8` makes each box occupy 80% of the unit spacing between group positions, with 10% gap on each side. The default is approximately `widths=0.5`.
- **Option D (Incorrect):** `boxwidth=0.8` is not a valid parameter for `ax.boxplot()`. This would raise a `TypeError`.

</details>

---

### 59. Question

A developer compares box plots and histograms as visualization tools and asks: "What can a histogram show that a box plot **cannot**?"

- [ ] **A.** The median value of the distribution
- [ ] **B.** The presence of outliers in the data
- [ ] **C.** Bimodal distributions — two separate peaks in frequency that a box plot collapses into a single summary
- [ ] **D.** The range of the data (minimum to maximum)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Box plots explicitly show the median (center line). Histograms can estimate the median visually (the value at which the cumulative histogram reaches 50%) but don't mark it explicitly. Box plots are BETTER at showing the median than histograms.
- **Option B (Incorrect):** Box plots explicitly show outliers as individual flier markers. Histograms show the frequency distribution shape, which may include extreme bars, but outliers are not specifically identified or marked.
- **Option C (Correct):** **Bimodal distributions** (with two distinct frequency peaks) are a key example of information visible in histograms but hidden in box plots. A bimodal dataset (e.g., ages of a mixed population of children and adults) might have two clear peaks in a histogram, but its box plot would show a wide box spanning both groups, appearing similar to a uniform or platykurtic distribution. Box plots summarize distributions with five statistics and cannot capture multiple modes or complex shapes.
- **Option D (Incorrect):** Box plots explicitly show the data range through the whisker endpoints (and through flier positions for datasets with outliers). Histograms show range through the x-axis extent of non-zero bars. Both can show the approximate data range.

</details>

---

### 60. Question

A developer creates a box plot and encounters a **ValueError**. Identify the bug:

```python
group_a = [10, 12, 14, 16, 18]
group_b = [20, 22, 24, 26, 28]
group_c = [30, 32, 34, 36, 38]

fig, ax = plt.subplots()
bp = ax.boxplot([group_a, group_b, group_c],
                tick_labels=["Alpha", "Beta", "Gamma"],
                positions=[1, 2])   # Bug: only 2 positions for 3 groups
```

What is the specific bug?

- [ ] **A.** `positions` must start at 0, not 1
- [ ] **B.** `positions=[1, 2]` has 2 elements but there are 3 data groups — the number of positions must match the number of groups
- [ ] **C.** `positions` cannot be used with `tick_labels` simultaneously
- [ ] **D.** Lists cannot be used as position values — must use a NumPy array

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `positions` can start at any value, including 1. The starting value determines where the first box is placed on the x-axis. Starting at 1 is the default and perfectly valid.
- **Option B (Correct):** The `positions` array must have exactly as many elements as there are data groups. With 3 groups (`[group_a, group_b, group_c]`), `positions` must have 3 elements. `positions=[1, 2]` has only 2, causing Matplotlib to raise a `ValueError` because it cannot assign a position to every group. The fix is `positions=[1, 2, 3]` or any 3-element list.
- **Option C (Incorrect):** `positions` and `tick_labels` can be used simultaneously. `tick_labels` assigns label strings to each group; `positions` assigns x-coordinates. They are complementary parameters.
- **Option D (Incorrect):** Python lists are accepted for `positions`. NumPy arrays also work, but they are not required. The issue is length mismatch, not type mismatch.

</details>
