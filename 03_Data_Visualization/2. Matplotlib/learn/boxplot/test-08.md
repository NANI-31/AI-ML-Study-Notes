# Matplotlib Box Plot Objective Test — Set 08

---

### 71. Question

A machine learning engineer evaluates **three regression models** on a housing price dataset by comparing their residual distributions:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
linear_res   = np.random.normal(0, 8000, 300)
gradient_res = np.append(np.random.normal(0, 3000, 285), np.random.normal(25000, 5000, 15))
neural_res   = np.random.normal(-2000, 4000, 300)

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(
    [linear_res, gradient_res, neural_res],
    tick_labels=["Linear Reg", "Gradient Boost", "Neural Net"],
    patch_artist=True,
    notch=True
)

for patch, color in zip(bp["boxes"], ["#AED6F1", "#A9DFBF", "#F9E79F"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Error")
ax.set_ylabel("Residual Error (USD)")
ax.set_title("Regression Model Residual Comparison")
ax.legend()
```

Based on the data generation code (not the plot), which model has the **highest variance in residuals**?

- [ ] **A.** Gradient Boosting — it has outliers above 25000
- [ ] **B.** Linear Regression — `std=8000` is the largest standard deviation for the primary distribution
- [ ] **C.** Neural Net — a non-zero mean indicates higher instability
- [ ] **D.** All three have equal variance because they all have 300 samples

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Gradient Boosting has `std=3000` for 285 samples and `std=5000` for the 15 outlier points. Its primary distribution is much tighter than Linear Regression's `std=8000`. The outlier cluster creates a visible group of fliers above the whisker, but the IQR-based box is narrow.
- **Option B (Correct):** Linear Regression uses `np.random.normal(0, 8000, 300)` — a standard deviation of **8000**. This is the largest single-distribution `std` among the three models. In the box plot, Linear Regression would have the widest box (largest IQR) and the longest symmetric whiskers, reflecting high but symmetric error variance. A model with wide, symmetric residuals has captured the trend but with poor precision.
- **Option C (Incorrect):** Neural Net has `std=4000` (larger than Gradient Boost's 3000 but smaller than Linear's 8000) and a mean of −2000 (slight systematic underestimation bias). A non-zero mean indicates systematic bias, not instability/variance. The mean doesn't determine the spread; the standard deviation does.
- **Option D (Incorrect):** The number of samples (300) is identical across models, but variance is a property of the DATA SPREAD, not the sample size. The three models have dramatically different standard deviations (8000, 3000/5000, 4000).

</details>

---

### 72. Question

Continuing from Question 71, the **Neural Net** model has `mean=-2000` (systematic underestimation bias). In the box plot, how would this bias appear visually?

- [ ] **A.** The box would be entirely above the y=0 reference line
- [ ] **B.** The median line would be positioned at approximately y=−2000, below the red reference line (y=0), indicating systematic negative bias
- [ ] **C.** The bias causes the box to be wider than symmetric models
- [ ] **D.** The bias is invisible — box plots cannot show distributional shift from zero

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A negative mean (−2000) means the distribution is centered BELOW zero, not above it. The box would be centered around y=−2000, mostly below the reference line.
- **Option B (Correct):** Since the Neural Net residuals are distributed as N(−2000, 4000), the **median** is approximately −2000 (for a normal distribution, mean ≈ median). In the box plot, the median line would appear at approximately y=−2000 — **below the red y=0 reference line**. The entire box would be shifted downward. This visual "gap" below zero directly communicates that the model systematically underestimates prices by about $2000 on average. This is a bias in the model, not a variance issue.
- **Option C (Incorrect):** Systematic bias (non-zero mean) shifts the entire distribution but does not widen it. The box width (IQR) is determined by the standard deviation (`std=4000`), not the mean. Bias and variance are orthogonal properties.
- **Option D (Incorrect):** Box plots explicitly show distributional shift through the median line position and box position relative to any reference lines. That's precisely why the `ax.axhline(0)` reference line is added — to visually compare whether each model's distribution is centered at zero (unbiased) or shifted (biased).

</details>

---

### 73. Question

A developer creates a box plot for cross-validation results across 5 models with 10 folds each. After examining the plot, the product manager asks: "Which model should we deploy?" 

The box plots show:

- **Model A**: Median=0.91, narrow box, no outliers
- **Model B**: Median=0.93, wide box, 3 outlier folds below 0.80
- **Model C**: Median=0.90, medium box, no outliers
- **Model D**: Median=0.88, narrow box, no outliers  
- **Model E**: Median=0.92, slightly wide box, 1 outlier fold at 0.75

Which model would a data scientist **most likely recommend** for production, and why?

- [ ] **A.** Model B — highest median accuracy (0.93) indicates superior performance
- [ ] **B.** Model A — high median (0.91) with the narrowest box and no outliers indicates consistently reliable performance
- [ ] **C.** Model E — second highest median with only one outlier
- [ ] **D.** Model C — the median is closest to the average across all models

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Model B has the highest median but a wide box and 3 outlier folds below 0.80. This means on ~30% of folds (3 out of 10), performance degraded severely. In production, this instability is dangerous — the model's performance may vary wildly on new data. High median doesn't compensate for high variance in critical applications.
- **Option B (Correct):** Model A has a high median (0.91) with the **narrowest box** (smallest IQR) and **no outliers**. This indicates the model performs consistently well across all 10 folds — the performance doesn't fluctuate significantly when the training data changes. In production, consistent, predictable performance is typically more valuable than marginally higher but variable performance. Model A represents the best balance of performance and stability.
- **Option C (Incorrect):** Model E has a slightly higher median (0.92) but shows more variance (wide box) and an outlier at 0.75 (one very bad fold). For production deployment, consistency is prioritised. One bad fold at 0.75 (vs. Model A's consistent 0.91) is a red flag.
- **Option D (Incorrect):** "Closest to average" is not a deployment criterion. Model D actually has the lowest median (0.88) with a narrow box — it's stable but underperforms. Median closeness to other models' average is statistically meaningless as a selection criterion.

</details>

---

### 74. Question

A researcher creates the following sorted dataset for a machine learning experiment and computes box plot statistics manually:

```
Data: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
n=10 (even)
Q₂ (Median) = average of 5th and 6th values = (10+12)/2 = 11
Lower half: [2, 4, 6, 8, 10] → Q₁ = 3rd value = 6
Upper half: [12, 14, 16, 18, 20] → Q₃ = 3rd value = 16
IQR = 10
```

Upper fence = 16 + 1.5×10 = **31**. Lower fence = 6 − 1.5×10 = **−9**.

If the value **35** is added, is it an outlier, and where does it appear?

- [ ] **A.** Not an outlier — 35 is just a large value
- [ ] **B.** Outlier — 35 > 31 (upper fence), plotted as a flier marker above the upper whisker cap (which ends at 20)
- [ ] **C.** Outlier — 35 is above the dataset maximum so it must be a flier
- [ ] **D.** Not an outlier — the fence recalculates with 35 included and it falls within the new fence

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 35 > upper fence of 31. Being a "large value" is not the criterion — the IQR-based fence rule determines outlier status. By this rule, 35 IS an outlier.
- **Option B (Correct):** Using the pre-computed fence (upper = 31): 35 > 31 → **outlier**. Matplotlib plots it as a flier marker above the upper whisker cap. The upper whisker extends to **20** (the highest non-outlier value ≤ 31), and 35 appears as a separate point above this cap. Note: strictly speaking, adding 35 changes the dataset and would slightly alter Q₁, Q₂, Q₃, and IQR — but the question uses the original fence values, and 35 exceeds those, making it an outlier under either calculation.
- **Option C (Incorrect):** Being the dataset maximum does NOT automatically make a value an outlier. If the maximum is within the fence (as 20 is within the original fence of [−9, 31]), it's the upper whisker endpoint. Only when the maximum exceeds the fence does it become a flier. The reasoning in C is incorrect.
- **Option D (Incorrect):** While it's true that adding 35 changes the dataset statistics, the spirit of the question uses the pre-computed fences. Even if we recompute with 35 included, the new statistics would increase IQR slightly but 35 would still likely exceed the new fence (original fence = 31, and adding 35 to an evenly-spaced dataset doesn't dramatically increase the upper fence enough to include 35).

</details>

---

### 75. Question

A developer creates a box plot styled for a dark-theme dashboard:

```python
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#1E1E2E")
ax.set_facecolor("#1E1E2E")

bp = ax.boxplot([group1, group2, group3],
                patch_artist=True,
                medianprops=dict(color="#F1C40F", linewidth=2.5),
                whiskerprops=dict(color="#ECF0F1", linewidth=1.2),
                capprops=dict(color="#ECF0F1", linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor="#E74C3C",
                                markeredgecolor="#C0392B", markersize=6))

for patch, color in zip(bp["boxes"], ["#2980B9", "#27AE60", "#8E44AD"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.tick_params(colors="#ECF0F1")
ax.spines["bottom"].set_color("#ECF0F1")
ax.spines["left"].set_color("#ECF0F1")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

What colour does the **median line** of each box appear, and why?

- [ ] **A.** White (#ECF0F1) — matching the whiskers
- [ ] **B.** Golden yellow (#F1C40F) — specified by `medianprops=dict(color="#F1C40F")`
- [ ] **C.** The same as the box fill colour (blue, green, or purple) — median inherits box colour
- [ ] **D.** Default blue — `medianprops` only works with `patch_artist=True` for patches, not for the median line

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `#ECF0F1` (near-white) is applied to whiskers and caps via `whiskerprops` and `capprops`. `medianprops` is a separate parameter that independently controls the median line.
- **Option B (Correct):** `medianprops=dict(color="#F1C40F", linewidth=2.5)` directly specifies the median line colour as **#F1C40F** (a golden yellow). The median line is thicker (2.5 pt) than the whiskers (1.2 pt), making it visually prominent against the dark background. This colour choice ensures the median is easy to distinguish from the box fill and whiskers — an important design consideration for dark-theme dashboards.
- **Option C (Incorrect):** The median line does NOT inherit the box fill colour. `medianprops` is completely independent of `patch_artist` and box colouring. Each box component is styled via its own `*props` parameter.
- **Option D (Incorrect):** `medianprops` works regardless of `patch_artist=True` or False. The median is always a `Line2D` object (never a `PathPatch`), and its colour is always controlled by `medianprops`. `patch_artist=True` only affects the box type (Line2D → PathPatch), not the median.

</details>

---

### 76. Question

A developer creates a box plot and is asked to explain the **relationship between sample size and notch width**. Given the formula:

$$\text{Notch half-width} = \frac{1.57 \times \text{IQR}}{\sqrt{n}}$$

If the IQR stays constant at 20 and sample size increases from n=25 to n=100, how does the notch half-width change?

- [ ] **A.** Doubles from 6.28 to 12.56
- [ ] **B.** Halves from 6.28 to 3.14
- [ ] **C.** Stays the same — notch width doesn't depend on sample size
- [ ] **D.** Increases by 4× because n increases by 4×

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Doubling would require the formula output to double. Since n increases (√n increases), the half-width DECREASES (it's in the denominator).
- **Option B (Correct):** At n=25: half-width = 1.57×20/√25 = 31.4/5 = **6.28**. At n=100: half-width = 1.57×20/√100 = 31.4/10 = **3.14**. Doubling n (25→100 is 4× increase): √n doubles (5→10), so half-width halves (6.28→3.14). Larger samples → narrower notches → more precise median estimate → tighter confidence interval. This demonstrates why notch box plots with small n have very wide notches (less precise median estimates) and those with large n have narrow notches.
- **Option C (Incorrect):** The formula explicitly contains `1/√n`, so notch width IS dependent on sample size. Larger samples produce narrower notches (more confidence in the median estimate).
- **Option D (Incorrect):** Increasing n by 4× increases √n by 2× (not 4×), which DECREASES the notch half-width by half. The half-width is inversely proportional to √n, not to n itself.

</details>

---

### 77. Question

A data scientist creates a box plot with a **2D NumPy array** where rows are samples and columns are features:

```python
import numpy as np

feature_matrix = np.random.randn(500, 4)  # 500 samples, 4 features

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(feature_matrix,
           tick_labels=["F1", "F2", "F3", "F4"])
```

The scientist wants to instead plot distributions **per sample** (row-wise). What modification is needed?

- [ ] **A.** Pass `rows=True` to `ax.boxplot()`
- [ ] **B.** Transpose the matrix: `ax.boxplot(feature_matrix.T)` — this makes each row a column, creating 500 boxes from 4-sample rows
- [ ] **C.** Use `ax.boxplot(feature_matrix, axis=0)` to specify row-wise processing
- [ ] **D.** Pass `feature_matrix.flatten()` to create a single box from all values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `rows=True` is not a valid parameter for `ax.boxplot()`. This would raise a `TypeError`.
- **Option B (Correct):** `ax.boxplot()` processes 2D arrays **column-wise** (each column = one box). To process row-wise (each row = one box), you must **transpose** the matrix: `feature_matrix.T` converts a (500×4) array to a (4×500) array. Now each of the 4 "columns" (which are actually original rows) becomes one box — though note: if you want 500 boxes from 500 rows, each with 4 values, `feature_matrix.T` is the right approach. This creates 4 boxes (one per original sample-row cluster) with 500 values each. For 500 individual-sample boxes from 4-value rows, you'd transpose to get the (4×500) structure where each of the 500 "columns" is a sample.
- **Option C (Incorrect):** `axis=0` is not a valid parameter for `ax.boxplot()`. The processing direction is determined by array shape, not an `axis` keyword.
- **Option D (Incorrect):** `feature_matrix.flatten()` converts the 2D array to a 1D array of 2000 values (500×4), creating a SINGLE box plot of all values mixed together — losing all feature and sample structure.

</details>

---

### 78. Question

A developer creates a box plot and wants to add **hatch patterns** to the boxes for accessibility in black-and-white printing:

```python
bp = ax.boxplot([d1, d2, d3], patch_artist=True)

hatches = ["//", "xx", ".."]
colors  = ["lightblue", "lightgreen", "lightyellow"]

for patch, hatch, color in zip(bp["boxes"], hatches, colors):
    patch.set_facecolor(color)
    patch.set_hatch(hatch)
    patch.set_edgecolor("black")
```

What does `patch.set_hatch("//")` add to the box?

- [ ] **A.** A forward-slash colour gradient
- [ ] **B.** A repeating diagonal forward-slash line pattern inside the box — distinguishing it from other boxes in monochrome
- [ ] **C.** Double the box width
- [ ] **D.** Converts the box to a Line2D object

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `hatch` creates a line PATTERN, not a colour gradient. Gradients in Matplotlib require custom normalisations or image overlays — not `set_hatch()`.
- **Option B (Correct):** `set_hatch("//")` fills the `PathPatch` box interior with a repeating **diagonal forward-slash pattern** (`/` lines). The `"//"` (doubled) creates a denser pattern than `"/"` (single). Hatch patterns are rendered in the `edgecolor` (here set to black). When combined with different patterns (`"//", "xx", ".."`) across groups, this creates visually distinct boxes even in black-and-white printing or for colour-blind audiences. Valid hatch characters include `/`, `\`, `|`, `-`, `+`, `x`, `o`, `O`, `.`, `*`.
- **Option C (Incorrect):** `set_hatch()` is a purely cosmetic styling method that affects the fill pattern. It has no effect on box dimensions.
- **Option D (Incorrect):** `patch_artist=True` already converted the box from Line2D to PathPatch. `set_hatch()` is a method of PathPatch that adds a pattern — it doesn't convert the object back to Line2D.

</details>

---

### 79. Question

A machine learning researcher creates a **combined box plot and scatter (strip plot)** overlay to show both the distribution summary and individual data points:

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
groups = [np.random.normal(loc, 2, 30) for loc in [5, 7, 9]]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(groups, positions=[1, 2, 3],
                patch_artist=True, widths=0.5)

for patch, color in zip(bp["boxes"], ["#AED6F1", "#A9DFBF", "#F9E79F"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual points
for i, group in enumerate(groups, start=1):
    x_jitter = np.random.normal(i, 0.05, len(group))
    ax.scatter(x_jitter, group, alpha=0.6, s=15, color="navy", zorder=3)
```

What does `zorder=3` on the scatter call ensure?

- [ ] **A.** The scatter points appear on top of (in front of) the box plot elements during rendering
- [ ] **B.** The scatter points are 3× larger than default
- [ ] **C.** The scatter points are drawn 3 pixels offset from their true position
- [ ] **D.** `zorder=3` limits the number of scatter points to 3

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `zorder` controls the **drawing order** (Z-order) of artists in Matplotlib. Artists with higher `zorder` values are rendered ON TOP of those with lower values. Default `zorder` for patches (like box plot boxes) is around 1–2. Setting `zorder=3` on the scatter points ensures they appear **in front of** the semi-transparent box fills, making individual data points clearly visible even when they overlap with the box area. Without `zorder=3`, scatter points might be hidden behind the box patch.
- **Option B (Incorrect):** Scatter point size is controlled by `s=15` (size in points²). `zorder` has no effect on size.
- **Option C (Incorrect):** `zorder` controls rendering layer, not spatial offset. A pixel offset would require manual `x_jitter` manipulation (which is already done separately via `np.random.normal(i, 0.05, ...)`).
- **Option D (Incorrect):** `zorder` is not a count limiter. The number of scatter points is determined by `len(group)` and the array sizes passed to `ax.scatter()`.

</details>

---

### 80. Question

A developer creates a box plot using `vert=False` for a horizontal layout and wants to add a **vertical reference line** at x=0 (representing zero error):

```python
fig, ax = plt.subplots()
ax.boxplot(residuals, vert=False)
ax.axvline(0, color="red", linestyle="--")  # Reference at zero error
ax.set_xlabel("Prediction Error")
```

Why is `ax.axvline(0)` used instead of `ax.axhline(0)` in this horizontal box plot?

- [ ] **A.** `axvline` is always required for horizontal box plots
- [ ] **B.** In a horizontal box plot, error values are plotted on the x-axis; `axvline(0)` draws a VERTICAL line at x=0, crossing through all horizontal boxes at the zero-error reference — directly analogous to `axhline(0)` in vertical box plots
- [ ] **C.** `axhline` would draw outside the plot boundary in horizontal mode
- [ ] **D.** `axvline` and `axhline` produce the same line when `vert=False`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `axvline` is NOT always required for horizontal box plots — it's required in THIS CASE because the reference value (zero) is on the x-axis (where residuals are plotted in horizontal mode). The choice depends on which axis holds the data values.
- **Option B (Correct):** With `vert=False`, the box plot is rotated: data values (residuals) are on the **x-axis**, and group positions are on the y-axis. To mark x=0 (zero error boundary) across all horizontal boxes, you draw a **vertical** line at x=0 using `axvline(0)`. This creates a vertical red dashed line that cuts through all horizontal boxes at the zero point — exactly like `axhline(0)` works in a vertical box plot (data on y-axis). The axis roles are swapped, so the reference line type must also swap.
- **Option C (Incorrect):** `axhline(0)` in a horizontal box plot would draw a horizontal line at y=0 — which would bisect between the lowest group and the x-axis. This is not meaningless (it marks the group y=0 position) but is not the intended zero-error reference for the x-axis data values.
- **Option D (Incorrect):** `axvline` draws a vertical line (constant x); `axhline` draws a horizontal line (constant y). They are always perpendicular to each other, regardless of box plot orientation.

</details>
