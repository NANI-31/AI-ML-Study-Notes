# Matplotlib Box Plot Objective Test — Set 09

---

### 81. Question

A data scientist creates a complete ML model diagnostic script comparing four models' cross-validation residuals:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

np.random.seed(7)
models = {
    "Ridge":    np.random.normal(0,    2500, 200),
    "Lasso":    np.random.normal(500,  3000, 200),
    "SVR":      np.random.normal(-200, 1500, 200),
    "XGBoost":  np.append(np.random.normal(0, 1200, 190),
                           np.random.normal(8000, 500, 10)),
}

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(list(models.values()),
                tick_labels=list(models.keys()),
                patch_artist=True,
                notch=True,
                vert=True)

palette = ["#5DADE2", "#EB984E", "#58D68D", "#AF7AC5"]
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)

for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(2)

ax.axhline(0, color="#E74C3C", linewidth=2, linestyle="--", alpha=0.8,
           label="Zero Residual (Ideal)")
ax.set_ylabel("Residual Error ($)", fontsize=12)
ax.set_title("Regression Model Residual Distribution", fontsize=14)
ax.legend(fontsize=10)
plt.tight_layout()
```

Based on the data generation code, which model has the **most symmetric and unbiased** residual distribution?

- [ ] **A.** Lasso — it has the highest standard deviation so it captures more variance
- [ ] **B.** SVR — although it has a slight negative bias (mean=−200), it has the smallest standard deviation (1500) and is closest to zero
- [ ] **C.** Ridge — mean=0 (perfectly unbiased) with moderate spread (std=2500)
- [ ] **D.** XGBoost — it has the most data points in the primary cluster (190 samples)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Lasso has `mean=500` (systematic positive bias) — it overestimates by $500 on average. High variance is not desirable for a regression model. "Unbiased" specifically requires mean residual ≈ 0, which Lasso doesn't satisfy.
- **Option B (Incorrect):** SVR has `mean=−200` (small negative bias) and the smallest standard deviation. While SVR has the most compact residuals, it is NOT perfectly unbiased (mean ≠ 0). "Most unbiased" requires the mean to be as close to zero as possible.
- **Option C (Correct):** Ridge has `mean=0` — **perfectly unbiased** in this simulation. The residuals are centered exactly at zero. While its `std=2500` is not the smallest, the zero mean ensures no systematic overestimation or underestimation. In the box plot, the median line (in white) would align closely with the red y=0 reference line. Ridge balances unbiasedness with moderate variance.
- **Option D (Incorrect):** XGBoost has 190 samples from N(0, 1200) plus 10 samples from N(8000, 500) — creating a compact main distribution with extreme high outliers. The 10 outlier points make XGBoost's mean ≈ 0 + (10×8000)/200 = 400 (non-zero bias). The question asks for "most symmetric and unbiased," which Ridge satisfies better.

</details>

---

### 82. Question

A developer adds a **custom Patch legend** to a multi-group box plot where each group uses a different fill colour:

```python
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#5DADE2", edgecolor="gray", alpha=0.82, label="Ridge"),
    Patch(facecolor="#EB984E", edgecolor="gray", alpha=0.82, label="Lasso"),
    Patch(facecolor="#58D68D", edgecolor="gray", alpha=0.82, label="SVR"),
    Patch(facecolor="#AF7AC5", edgecolor="gray", alpha=0.82, label="XGBoost"),
]
ax.legend(handles=legend_elements, loc="upper right")
```

Why are custom `Patch` objects needed here instead of using `label=` in the `boxplot()` call?

- [ ] **A.** `ax.boxplot()` does not support a `label=` parameter
- [ ] **B.** Each box is individually coloured via `set_facecolor()` in a post-hoc loop — no `label=` was passed to any box during creation, so the automatic legend has no entries; custom Patches recreate the visual style for legend display
- [ ] **C.** `Patch` objects are required for all Matplotlib legends
- [ ] **D.** The `handles=` parameter only works with Patch objects

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `ax.boxplot()` does NOT have a `label=` parameter for individual group labels (unlike `ax.bar()` or `ax.plot()`). Box plot groups are labeled via `tick_labels=`. This means there's no mechanism to create legend entries through the `boxplot()` call itself.
- **Option B (Correct):** Box plot groups don't automatically create legend entries. The colours are assigned post-hoc via `patch.set_facecolor()` — these operations don't register with the legend system. To create a legend matching the visual style of each box, the developer manually creates `Patch` proxy artists with the same colours and alpha values as the actual boxes. These proxies serve as visual representations in the legend without being actual plot artists.
- **Option C (Incorrect):** Matplotlib legends can use any artists as handles — `Line2D`, `PathPatch`, `Rectangle`, `Patch`, `Artist` subclasses. Custom `Patch` objects are not universally required for legends.
- **Option D (Incorrect):** `handles=` accepts any iterable of artists, not just `Patch` objects. `Line2D`, `PathCollection` (scatter), and other artist types can all be used as legend handles.

</details>

---

### 83. Question

A developer creates a dataset with a **very extreme outlier** and examines the box plot:

```python
data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1000]
```

Q₁=12, Q₃=18, IQR=6. Upper fence = 18 + 1.5×6 = 27. The value 1000 is an outlier.

How does the value 1000 affect **Q₁, Q₂, and Q₃** in Matplotlib's box plot computation?

- [ ] **A.** The outlier 1000 is excluded from quartile calculations — Q₁, Q₂, Q₃ are computed only from non-outlier data
- [ ] **B.** All outlier values (including 1000) are included in the dataset for computing Q₁, Q₂, Q₃, and IQR — outliers are excluded only from WHISKER ENDPOINT determination, not from statistical computation
- [ ] **C.** Matplotlib automatically winsorizes extreme values before computing quartiles
- [ ] **D.** The value 1000 shifts Q₃ significantly upward, making the box much taller

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This is the most common misconception about box plots. Outliers ARE included in the quartile calculations. Q₁, Q₂, Q₃, and IQR are computed from the full dataset including 1000. The sorting-based quartile computation doesn't exclude any values — it uses all n data points.
- **Option B (Correct):** For the 12-element dataset including 1000: sorted = [10,11,12,13,14,15,16,17,18,19,20,1000]. Q₂ (median) = average of 6th and 7th values = (15+16)/2 = **15.5**. Q₁ = median of lower half [10,11,12,13,14,15] = (12+13)/2 = **12.5**. Q₃ = median of upper half [16,17,18,19,20,1000] = (18+19)/2 = **18.5**. IQR = 6. The value 1000 is used in calculating Q₃ (it's in the upper half) — but since Q₃ is the median of the upper half, and 1000 is the 12th value (well beyond the 9th–10th positions used for Q₃ median), it doesn't drastically change Q₃. Outliers are excluded only from whisker END determination (the whisker stops at 20, not 1000).
- **Option C (Incorrect):** Matplotlib does NOT automatically winsorize (clip extreme values) before computing quartiles. Raw data is used as-is for all statistical computations.
- **Option D (Incorrect):** For this specific dataset, 1000 is in the upper half but is beyond the median of the upper half. With 6 values in the upper half [16,17,18,19,20,1000], Q₃ = average of 3rd and 4th = (18+19)/2 = 18.5 — only slightly different from the Q₃=18 computed without 1000. The box height doesn't change dramatically because Q₃ is based on percentile position, and 1000 is only the 12th (last) value.

</details>

---

### 84. Question

A developer creates the following code to produce a **styled multi-model comparison** for a research paper figure and encounters an issue:

```python
fig, ax = plt.subplots(figsize=(10, 6))

data_groups = [model1_cv, model2_cv, model3_cv]

bp = ax.boxplot(data_groups,
                tick_labels=["CNN", "RNN", "Transformer"],
                patch_artist=True,
                notch=True,
                widths=0.6)

# Attempt to colour boxes
bp["boxes"].set_facecolor(["blue", "green", "orange"])   # Bug line
```

What is the bug on the `bp["boxes"].set_facecolor(...)` line?

- [ ] **A.** `set_facecolor` doesn't accept a list — it only accepts a single colour string
- [ ] **B.** `bp["boxes"]` is a Python **list** of PathPatch objects — calling `.set_facecolor()` on a list raises an `AttributeError` because lists don't have this method
- [ ] **C.** `notch=True` disables the ability to set facecolor
- [ ] **D.** The `widths=0.6` parameter conflicts with `set_facecolor`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `set_facecolor()` CAN accept a list of colours when called on a supported artist collection. However, the issue here is not the argument type — it's that `.set_facecolor` is being called on a raw Python **list**, not an artist object.
- **Option B (Correct):** `bp["boxes"]` is a plain Python list containing 3 `PathPatch` objects. Python lists do NOT have a `set_facecolor()` method. Calling `bp["boxes"].set_facecolor(...)` raises `AttributeError: 'list' object has no attribute 'set_facecolor'`. The correct approach is to iterate: `for patch, color in zip(bp["boxes"], colors): patch.set_facecolor(color)`.
- **Option C (Incorrect):** `notch=True` only affects the visual shape of the box (adding the V-notch around the median). It has no effect on the ability to set face colours via `patch_artist=True`.
- **Option D (Incorrect):** `widths=0.6` controls the horizontal width of each box. It is completely independent of colour settings.

</details>

---

### 85. Question

A researcher creates a box plot for a dataset and observes that the **lower whisker is much shorter** than the upper whisker. The median sits near the center of the box. What does this pattern indicate?

- [ ] **A.** The data is left-skewed — the lower tail is shorter than the upper tail, indicating the data does not extend far below the median
- [ ] **B.** The data is right-skewed — a longer upper whisker indicates the upper tail extends further from the box
- [ ] **C.** The data is symmetric — the median at center means normal distribution
- [ ] **D.** The data has been incorrectly sorted before plotting

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Option A correctly identifies that the lower tail is "shorter" but incorrectly labels this as "left-skewed." Left-skew means the LEFT (lower) tail is LONGER. A shorter lower whisker means the lower tail is compact — there's little spread toward lower values.
- **Option B (Correct):** A longer upper whisker combined with a centered median indicates **right-skew (positive skew)**. The upper tail extends significantly further from Q₃ than the lower tail extends from Q₁. The median being near the box center means the middle 50% (IQR) is roughly symmetric, but the overall distribution has an asymmetric upper tail. This pattern is common in real-world data like income distributions, wait times, and response times — where most values cluster at lower levels but rare high values create an extended right tail.
- **Option C (Incorrect):** A centered median indicates symmetry WITHIN the IQR (the middle 50%), but whisker length asymmetry reveals tail behavior beyond the IQR. A truly symmetric distribution would show both equal whisker lengths AND a centered median.
- **Option D (Incorrect):** Box plots don't require pre-sorted data — `ax.boxplot()` computes quartiles internally from the raw data array regardless of input order. A short lower whisker is not a data ordering artifact.

</details>

---

### 86. Question

A developer creates a box plot for two groups and uses `showfliers=False` to hide outliers for a cleaner presentation. A reviewer asks whether this is misleading:

```python
ax.boxplot([group_A, group_B], showfliers=False)
```

Which response best explains the tradeoff?

- [ ] **A.** `showfliers=False` is always misleading and should never be used
- [ ] **B.** It is not misleading at all — outliers don't affect the box or whiskers
- [ ] **C.** `showfliers=False` can be acceptable for a cleaner view but is potentially misleading if the audience doesn't know outliers were suppressed — best practice is to note in the caption that outliers are hidden, or use `showfliers=True` (default)
- [ ] **D.** `showfliers=False` removes outliers from statistical calculations, making the chart more accurate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** "Always misleading" is too absolute. There are legitimate use cases for hiding outliers — for example, when outliers are known measurement errors, when the plot is part of a larger dashboard where outliers are discussed elsewhere, or when the audience only needs the core distribution summary. Context determines appropriateness.
- **Option B (Incorrect):** While `showfliers=False` doesn't change the box/whisker computations (they remain the same), suppressing visible outlier markers DOES hide potentially important information. If Group A has 10% of values as extreme outliers and Group B has none, hiding outliers makes both groups appear more similar than they are.
- **Option C (Correct):** `showfliers=False` is a display-level choice with a real tradeoff: it produces a cleaner, less cluttered visual at the cost of hiding extreme values that may be important. The best practice is transparency — annotating the figure caption with "Outliers not shown for clarity" (with their count) maintains honesty. Alternatively, using `showfliers=True` (the default) always shows the complete picture.
- **Option D (Incorrect):** `showfliers=False` has ZERO effect on statistical calculations. Q₁, Q₂, Q₃, IQR, whisker positions — all computed from the full dataset including outliers. Only the rendering of outlier MARKERS is suppressed.

</details>

---

### 87. Question

A developer creates a combined box plot dashboard for a complete ML training analysis. They want each subplot's box plot to use the **same y-axis limits** for fair comparison. The code uses `sharey=False`:

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
for ax, (model, data) in zip(axes, model_data.items()):
    ax.boxplot(data, patch_artist=True)
    ax.set_title(model)
    ax.set_ylim(0, 1)   # Manually set same limits
```

Could using `sharey=True` instead simplify this, and what would be the visual difference?

- [ ] **A.** `sharey=True` would be identical — `set_ylim(0, 1)` overrides `sharey` anyway
- [ ] **B.** `sharey=True` automatically synchronises y-axis limits across all subplots; if one model's data range extends beyond [0,1], `sharey=True` would expand ALL axes to include the larger range — potentially different from the manual `set_ylim(0, 1)` enforcement
- [ ] **C.** `sharey=True` only works for bar plots
- [ ] **D.** `sharey=True` removes y-axis labels from all panels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `set_ylim(0, 1)` called on individual axes AFTER `sharey=True` would actually propagate to ALL axes (since they share the y-axis). The result would be all panels showing [0, 1]. However, calling `set_ylim` on each axis individually in a `sharey=True` setup overwrites the shared limit — so they'd behave similarly, but the mechanism is different.
- **Option B (Correct):** With `sharey=True`: Matplotlib automatically determines the union of all data ranges across shared axes and applies it as the common y-limit. If one model's residuals range from −0.5 to 1.3, ALL panels would show [−0.5, 1.3]. With manual `set_ylim(0, 1)`: all panels are explicitly forced to [0, 1] regardless of actual data ranges, clipping any data below 0 or above 1. The visual difference: `sharey=True` auto-fits to data; `set_ylim(0, 1)` enforces a fixed range (which may clip data or add extra whitespace).
- **Option C (Incorrect):** `sharey=True` works for ALL Matplotlib subplot types — box plots, bar plots, line plots, scatter plots, etc. It's a general subplot configuration, not chart-type specific.
- **Option D (Partially correct but oversimplified):** `sharey=True` suppresses y-tick LABELS on non-leftmost panels (to avoid redundancy) but keeps y-axis gridlines and scale. It doesn't "remove" y-axis labels entirely — it suppresses the numeric tick labels on shared axes.

</details>

---

### 88. Question

A developer creates a box plot for a dataset where **Q₁=Q₂=Q₃** (the middle 50% of data is identical). What does the box plot look like?

- [ ] **A.** A normal-looking box with a median line at the center
- [ ] **B.** A flat horizontal line with no box body — the box collapses to the median line; whiskers extend to the non-identical extreme values
- [ ] **C.** An error is raised — box plots require Q₁ < Q₂ < Q₃
- [ ] **D.** The entire plot is blank because IQR=0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** If Q₁=Q₂=Q₃, there is no box "body" to draw — the top and bottom of the box collapse to the same value. A normal-looking box requires Q₁ < Q₃.
- **Option B (Correct):** When Q₁=Q₂=Q₃ (for example, in a dataset like [5, 10, 10, 10, 10, 10, 15, 20] where the middle half is all 10s): The box collapses to a **flat horizontal line** at the common quartile value. The whiskers still extend to the outermost non-outlier values (respecting the fence = Q±1.5×0 = Q, so anything other than Q is an outlier, or more precisely, all data = Q so fences = Q and the whiskers have zero length). The visual is a single flat line with potentially outlier markers on both sides.
- **Option C (Incorrect):** Matplotlib handles Q₁=Q₂=Q₃ gracefully without raising an error. The box simply degenerates to a flat line.
- **Option D (Incorrect):** The plot is not blank. Even with IQR=0, the degenerate box (flat line), any whiskers (extending to the fence limits, which equal Q), and outlier markers (any values ≠ Q) are still rendered.

</details>

---

### 89. Question

A developer wants to compare **10-fold cross-validation accuracy** of 5 models with a box plot and overlay individual fold scores as scatter points. The key insight is choosing between `showfliers=True` and `showfliers=False`. Which approach is better when scatter points are overlaid?

- [ ] **A.** `showfliers=True` (default) because outliers are important even with scatter overlay
- [ ] **B.** `showfliers=False` because the scatter points already show ALL individual fold values including outliers — the box plot flier markers would be redundant and cause visual cluttering
- [ ] **C.** Both show identical results — scatter points appear at different x-positions from fliers
- [ ] **D.** `showfliers=False` hides the data, making the scatter overlay unreliable

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** With individual fold scores shown as scatter points, the flier markers from the box plot would appear at the SAME y-positions as the scatter points (they represent the same data). This creates visual duplication — both the scatter points and flier markers show the same extreme values, cluttering the chart.
- **Option B (Correct):** When scatter points (or strip plots) overlay a box plot, the individual data points are already displayed — including the extreme/outlier values. Adding box plot flier markers would double-plot these extreme points, creating visual noise. `showfliers=False` suppresses the redundant flier markers, leaving the scatter points as the sole representation of individual data. The box, whiskers, and median still summarize the distribution. This combined visualization (box + scatter) is a best practice for small datasets (n<100) as it shows both the summary and the individual points.
- **Option C (Incorrect):** Scatter points and flier markers are NOT at different x-positions — they both appear at approximately x=group_position, creating overlap at the same coordinates for outlier values.
- **Option D (Incorrect):** `showfliers=False` doesn't affect the scatter overlay or its data. The scatter `ax.scatter(x_jitter, data)` call operates independently of `showfliers`. Scatter points are unaffected by box plot `showfliers` settings.

</details>

---

### 90. Question

A developer creates a box plot for a large dataset (n=10,000) and a small dataset (n=15). Both have the same Q₁=50, Q₃=70, IQR=20. They use `notch=True` for both. Which statement about the **notch widths** is correct?

- [ ] **A.** Both box plots have identical notch widths because Q₁, Q₃, and IQR are the same
- [ ] **B.** The small dataset (n=15) has a wider notch; the large dataset (n=10000) has a narrower notch — reflecting greater uncertainty in the median estimate with fewer samples
- [ ] **C.** The large dataset has a wider notch because more samples introduce more uncertainty
- [ ] **D.** Notch width depends only on IQR, not on sample size

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While Q₁, Q₃, and IQR are the same, the **sample size** (n) also appears in the notch formula: `half-width = 1.57×IQR/√n`. Different n values produce different notch widths even with identical IQR.
- **Option B (Correct):** From the formula: For n=15: half-width = 1.57×20/√15 = 31.4/3.873 = **8.11**. For n=10000: half-width = 1.57×20/√10000 = 31.4/100 = **0.314**. The small dataset (n=15) has a notch half-width of 8.11 (very wide notch) — reflecting HIGH uncertainty in the median estimate with only 15 samples. The large dataset (n=10000) has a notch half-width of 0.314 (very narrow notch) — reflecting HIGH confidence in the median estimate with 10,000 samples. Wider notch = less certainty; narrower notch = more certainty.
- **Option C (Incorrect):** More samples REDUCE uncertainty and produce NARROWER notches, not wider. The formula has n in the denominator (via √n), so increasing n decreases the half-width.
- **Option D (Incorrect):** The formula explicitly includes both IQR and n: `1.57×IQR/√n`. Notch width depends on BOTH IQR and sample size. Claiming it depends only on IQR contradicts the formula.

</details>
