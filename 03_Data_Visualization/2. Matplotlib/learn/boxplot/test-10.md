# Matplotlib Box Plot Objective Test — Set 10

---

### 91. Question

A data scientist creates the following capstone ML diagnostics script and asks which model should be investigated for **data leakage or overfitting**:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

np.random.seed(21)

model_cv_scores = {
    "LogReg":    np.random.normal(0.84, 0.018, 10),
    "RandomFor": np.append(np.random.normal(0.96, 0.002, 9), [0.62]),
    "SVM":       np.random.normal(0.88, 0.012, 10),
    "XGBoost":   np.random.normal(0.91, 0.015, 10),
}

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(list(model_cv_scores.values()),
                tick_labels=list(model_cv_scores.keys()),
                patch_artist=True, notch=True)

colors = ["#5499C7", "#52BE80", "#F0B27A", "#BB8FCE"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.axhline(0.90, color="crimson", linestyle="--",
           linewidth=1.5, label="90% Target")
ax.set_ylabel("CV Accuracy")
ax.set_title("10-Fold CV Accuracy by Model")
ax.legend()
plt.tight_layout()
```

Which model's box plot would show the **most suspicious pattern** (possible data leakage)?

- [ ] **A.** LogReg — the lowest median (0.84) suggests underperformance
- [ ] **B.** RandomForest — median ≈ 0.96 but with one extreme outlier fold at 0.62 suggests the model is very sensitive and may be overfit
- [ ] **C.** SVM — medium variance across folds
- [ ] **D.** XGBoost — performs well above the 90% target

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** LogReg at median 0.84 with tight spread (std=0.018) is a stable, moderate performer. There's nothing suspicious — it simply has lower accuracy than other models, but consistently so.
- **Option B (Correct):** RandomForest shows a suspicious pattern: 9 out of 10 folds achieve ~0.96 accuracy (very high, tight cluster) while 1 fold drops to 0.62 (a single catastrophic fold). This extreme pattern — near-perfect performance on most folds with one severe failure — is a classic sign of: (1) **data leakage** where some folds accidentally expose test data during training for 9 folds but the 10th fold is correctly isolated, or (2) **severe overfitting** where the model fails to generalize to a particular fold's distribution. The very high median (0.96) combined with a single catastrophic outlier fold (0.62) warrants immediate investigation.
- **Option C (Incorrect):** SVM with std=0.012 shows normal, moderate variation across folds. No single fold is catastrophically different from others. This is expected, healthy variance in cross-validation.
- **Option D (Incorrect):** XGBoost at median≈0.91 with std=0.015 performs consistently above the 90% target. This is a well-behaved model. No suspicious patterns are present.

</details>

---

### 92. Question

A developer computes quartile statistics for two datasets and examines the **robustness** of median vs. mean when one dataset has extreme outliers:

```
Dataset A: [10, 12, 14, 16, 18, 20]          → Mean=15, Median=15
Dataset B: [10, 12, 14, 16, 18, 200]         → Mean=45, Median=15
```

A box plot is created for both datasets. Which statement is TRUE about the **median line position** in both box plots?

- [ ] **A.** Dataset B's median line is much higher than Dataset A's because of the outlier 200
- [ ] **B.** Both datasets show their median line at y=15 — the extreme outlier (200) does not shift the median
- [ ] **C.** Dataset B's box is much taller (larger IQR) due to the outlier 200
- [ ] **D.** The mean marker would also appear at y=15 for Dataset B

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The median of Dataset B: sorted=[10,12,14,16,18,200], median = average of 3rd and 4th values = (14+16)/2 = **15** — exactly the same as Dataset A. The outlier 200 does NOT change the median because the median is determined by the POSITIONAL middle value(s), not the magnitude of extreme values.
- **Option B (Correct):** **Both box plots show the median line at y=15**. This is the key insight behind median-based robust statistics: the median is completely insensitive to the magnitude of extreme values. Whether the 6th value is 20 or 200, the 3rd and 4th values (which determine the median) remain 14 and 16. The mean, however, changes dramatically: 15 vs. 45. This demonstrates why box plots use medians (robust) rather than means (non-robust) as their central measure.
- **Option C (Incorrect):** Dataset A: Q₁=(10+12)/2=11, Q₃=(18+20)/2=19, IQR=8. Dataset B: Q₁=(10+12)/2=11, Q₃=(16+18)/2=17... Wait — for Dataset B [10,12,14,16,18,200]: lower half=[10,12,14], Q₁=12; upper half=[16,18,200], Q₃=18. IQR = 18−12=6. For Dataset A: lower=[10,12,14], Q₁=12; upper=[16,18,20], Q₃=18. IQR = 18−12=6 as well. Both have the same IQR=6 — the outlier 200 doesn't change the IQR because quartile positions don't change with value magnitude, only value order.
- **Option D (Incorrect):** For Dataset B, `showmeans=True` would display a mean marker at y=45 (the arithmetic mean of Dataset B = 270/6=45), NOT at y=15. The mean is dramatically shifted by the outlier, which is exactly why the mean is non-robust.

</details>

---

### 93. Question

A developer creates a complete, production-quality box plot report and is asked to explain every visual element. Which of the following are **TRUE** statements about a standard Matplotlib box plot? (Select all that apply)

- [ ] **A.** The box spans Q₁ to Q₃, enclosing the middle 50% of the data
- [ ] **B.** The whisker caps mark the positions of Q₁−1.5×IQR and Q₃+1.5×IQR exactly
- [ ] **C.** If no outliers exist, the lower whisker extends to the dataset minimum and upper whisker extends to the dataset maximum
- [ ] **D.** The median line always divides the box into equal upper and lower halves

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C**

#### Explanations:

- **Option A (True):** The rectangular box spans from **Q₁** (bottom edge) to **Q₃** (top edge), representing the interquartile range — the middle 50% of the data distribution. This is the fundamental definition of the box in a box plot.
- **Option B (False):** This is the most common misconception. Whisker caps mark the positions of the **actual data values** (last non-outlier points), NOT the computed fence values Q₁−1.5×IQR and Q₃+1.5×IQR. The fences are BOUNDARIES used to classify outliers, but the whiskers stop at real data points within those boundaries. If no data point exists exactly at the fence, the whisker stops short.
- **Option C (True):** When no data points fall outside the 1.5×IQR fences (no outliers exist), the lower whisker extends to **min(data)** and the upper whisker extends to **max(data)** — the actual dataset extremes. This is precisely why the whisker endpoints are "the most extreme non-outlier data points": when there are no outliers, the extremes are the most extreme non-outlier points by definition.
- **Option D (False):** The median line's position within the box depends on the data distribution. In a symmetric distribution, the median is equidistant from Q₁ and Q₃ (centered). In skewed distributions, the median line is off-center: closer to Q₁ for right-skewed data, closer to Q₃ for left-skewed data. The median does NOT always bisect the box into equal halves.

</details>

---

### 94. Question

A developer creates a box plot for **model inference time** comparisons across 4 deployment environments. The developer examines the code and asks about the effect of `manage_xticks=True` (the default):

```python
bp = ax.boxplot([env1_ms, env2_ms, env3_ms, env4_ms],
                tick_labels=["Cloud A", "Cloud B", "On-Prem", "Edge"],
                manage_xticks=True)   # Default behavior
```

What does `manage_xticks=True` (the default) do?

- [ ] **A.** Automatically adds grid lines behind the boxes
- [ ] **B.** Automatically sets the x-axis tick positions and limits to neatly fit the box groups (setting `xlim` and `xticks` appropriately)
- [ ] **C.** Enables tick labels to rotate by 45° automatically
- [ ] **D.** Connects box outliers to their tick positions with lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Grid lines are controlled by `ax.grid()` or `ax.yaxis.grid()`. `manage_xticks` has no effect on grid lines.
- **Option B (Correct):** `manage_xticks=True` (the default) tells Matplotlib to automatically manage the x-axis tick positions and limits. Specifically, it sets `ax.set_xlim()` to neatly encompass the box positions with appropriate padding, and places x-ticks at the box group positions (1, 2, 3, 4, ...). This is why box plots automatically look well-spaced without manual `xlim` or `xtick` calls. Setting `manage_xticks=False` gives the developer full manual control over x-axis configuration — useful when combining box plots with other chart types at the same positions.
- **Option C (Incorrect):** Tick label rotation is never automatic — it must be explicitly set via `ax.set_xticklabels(labels, rotation=45)` or `ax.tick_params(labelrotation=45)`.
- **Option D (Incorrect):** No "connector lines" between outlier markers and tick positions exist in standard Matplotlib box plots. Outlier markers (fliers) appear at their data value y-positions directly above/below the box, aligned with the group x-position. No additional lines connect them to ticks.

</details>

---

### 95. Question

A developer creates a multi-group box plot for a time-series model evaluation across **four quarters**:

```python
Q1_errors = [0.15, 0.18, 0.12, 0.20, 0.16, 0.14, 0.19, 0.17, 0.13, 0.21]
Q2_errors = [0.22, 0.25, 0.19, 0.28, 0.24, 0.21, 0.27, 0.23, 0.20, 0.26]
Q3_errors = [0.30, 0.35, 0.28, 0.38, 0.33, 0.31, 0.36, 0.32, 0.29, 0.37]
Q4_errors = [0.18, 0.20, 0.16, 0.22, 0.19, 0.17, 0.21, 0.18, 0.15, 0.23]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([Q1_errors, Q2_errors, Q3_errors, Q4_errors],
                tick_labels=["Q1", "Q2", "Q3", "Q4"],
                patch_artist=True,
                notch=True)
```

Looking at the median values (≈0.165, 0.235, 0.325, 0.19), what trend does the box plot reveal?

- [ ] **A.** Model errors are uniformly decreasing each quarter
- [ ] **B.** Model errors increase from Q1 through Q3, then drop in Q4 — suggesting Q3 may have different data characteristics (seasonality) while Q4 returns to near-Q1 levels
- [ ] **C.** Q4 has the highest errors
- [ ] **D.** There is no meaningful trend — all medians are within 0.5 of each other

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Q1→Q2→Q3 shows INCREASING errors (0.165→0.235→0.325). Q4 drops back to 0.19. This is not uniform decrease.
- **Option B (Correct):** The trend across quarters: Q1 median≈0.165, Q2≈0.235, Q3≈0.325, Q4≈0.190. Errors rise steadily from Q1 to Q3, then drop significantly in Q4. In a time-series forecasting context, this pattern suggests that Q3's data distribution differs from other quarters (perhaps seasonal demand spikes, holidays, or weather patterns in Q3 that the model struggles to capture). Q4 errors dropping back near Q1 levels suggests the model handles Q4 similarly to Q1 conditions. This seasonality effect is a key insight from quarterly box plot analysis.
- **Option C (Incorrect):** Q3 has the highest median error (0.325), not Q4. Q4 median (0.190) is close to Q1 (0.165).
- **Option D (Incorrect):** The range 0.165 to 0.325 represents nearly a 2× increase in median error from Q1 to Q3 — this IS a meaningful trend. The claim "within 0.5" is numerically technically true (0.325−0.165=0.16) but this magnitude of variation is significant for forecasting model assessment.

</details>

---

### 96. Question

A developer examines the following code and must predict the **correct order** of rendering layers in the combined chart:

```python
fig, ax = plt.subplots()
ax.set_facecolor("#F0F0F0")       # Background

ax.grid(axis="y", zorder=0)       # Grid at back
bp = ax.boxplot([g1, g2, g3],     # Default zorder (≈2)
                patch_artist=True)
for patch in bp["boxes"]:
    patch.set_zorder(2)
    patch.set_facecolor("lightblue")
    patch.set_alpha(0.7)

ax.scatter([1,1,1], [q1_pts], zorder=5, color="red")  # Points at front
```

From back to front, what is the correct rendering order?

- [ ] **A.** Background → Boxes → Grid → Scatter points
- [ ] **B.** Grid → Background → Boxes → Scatter points
- [ ] **C.** Background → Grid (z=0) → Box plots (z=2) → Scatter points (z=5)
- [ ] **D.** Scatter points → Box plots → Grid → Background (drawn first to last)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Grid (zorder=0) renders before boxes (zorder=2). Option A incorrectly places boxes before the grid.
- **Option B (Incorrect):** The background (axes face colour) is always the rearmost element — before all drawn artists. Grid (z=0) comes after background.
- **Option C (Correct):** In Matplotlib, rendering order from back to front follows `zorder` values: (1) **Background** (axes facecolor = lowest Z, below all artists), (2) **Grid** (zorder=0), (3) **Box plot elements** (zorder≈2, default), (4) **Scatter points** (zorder=5). Higher zorder = rendered later = appears in front. The semi-transparent (alpha=0.7) box fills allow the grid lines below to show through faintly.
- **Option D (Incorrect):** In Matplotlib, lower zorder elements are drawn FIRST (background layer) and higher zorder elements are drawn LAST (foreground layer). Option D reverses this — scatter points at z=5 are drawn last, NOT first.

</details>

---

### 97. Question

A developer creates a final publication-quality box plot with complete styling for all five box plot components. Which of the following code patterns correctly applies **different line colours** to ONLY the whiskers and caps (navy) while leaving the box outline black?

- [ ] **A.**
```python
ax.boxplot(data,
           boxprops=dict(color="navy"),
           whiskerprops=dict(color="navy"),
           capprops=dict(color="navy"))
```

- [ ] **B.**
```python
ax.boxplot(data,
           boxprops=dict(color="black"),
           whiskerprops=dict(color="navy"),
           capprops=dict(color="navy"))
```

- [ ] **C.**
```python
ax.boxplot(data,
           color="navy",
           whiskers="navy", caps="navy")
```

- [ ] **D.**
```python
ax.boxplot(data,
           whisker_color="navy",
           cap_color="navy",
           box_color="black")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This sets ALL three components (box, whisker, cap) to navy. The question requires the BOX to be black, but `boxprops=dict(color="navy")` makes it navy — wrong for the box.
- **Option B (Correct):** `boxprops=dict(color="black")` makes the box outline black. `whiskerprops=dict(color="navy")` makes whisker lines navy. `capprops=dict(color="navy")` makes cap lines navy. This is the correct approach: each `*props` dictionary independently controls one component's styling.
- **Option C (Incorrect):** `color=`, `whiskers=`, and `caps=` are not valid keyword arguments for `ax.boxplot()`. These would raise a `TypeError`. The correct parameter names are `boxprops=`, `whiskerprops=`, and `capprops=`.
- **Option D (Incorrect):** `whisker_color=`, `cap_color=`, and `box_color=` are not valid parameters for `ax.boxplot()`. These would raise a `TypeError`. Matplotlib uses dictionary-based `*props` parameters, not individual color keyword shortcuts.

</details>

---

### 98. Question

A machine learning engineer creates a complete performance evaluation script comparing the **test set precision, recall, and F1-score** of three models using separate box plots for each metric:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
metrics  = ["Precision", "Recall", "F1-Score"]
all_data = [
    [lr_precision, rf_precision, xgb_precision],
    [lr_recall,    rf_recall,    xgb_recall],
    [lr_f1,        rf_f1,        xgb_f1],
]
model_names = ["LR", "RF", "XGB"]

for ax, metric, data in zip(axes, metrics, all_data):
    bp = ax.boxplot(data, tick_labels=model_names,
                    patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], ["#5DADE2","#52BE80","#AF7AC5"]):
        patch.set_facecolor(color)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.axhline(0.9, color="red", linestyle=":", alpha=0.7)

plt.suptitle("Model Performance Metrics — CV Bootstrap", fontsize=14)
plt.tight_layout()
plt.savefig("model_metrics.svg", bbox_inches="tight")
```

Why is **SVG format** chosen for `savefig` in a publication context?

- [ ] **A.** SVG files are smaller than PNG for complex figures
- [ ] **B.** SVG is a vector format — all box elements (boxes, whiskers, text, lines) are stored as resolution-independent mathematical paths, making the figure scalable to any print size without pixelation
- [ ] **C.** SVG files render faster in web browsers than PNG
- [ ] **D.** SVG is required for figures with `notch=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** SVG file size relative to PNG depends on the complexity of the figure. Simple vector figures (few paths) produce small SVGs, but complex figures with many elements can produce larger SVGs than equivalent PNG bitmaps. File size is not the primary reason for choosing SVG in publications.
- **Option B (Correct):** SVG (Scalable Vector Graphics) stores all figure elements — the box paths, whisker lines, median lines, text labels, reference lines — as mathematical vector paths. This makes the figure **completely resolution-independent**: it can be scaled to any size (from a web thumbnail to a full-page print at 600 DPI) without any pixelation or blurring. Journal publications require high-resolution figures, and SVG guarantees perfect quality at any scale. Journals like Nature, Science, and IEEE commonly accept SVG or PDF (another vector format) for figures.
- **Option C (Incorrect):** SVG and PNG rendering speed in browsers depends on browser optimisation and file complexity. SVG may actually render SLOWER for complex figures because the browser must parse and render each vector path. PNG rendering is often faster for display.
- **Option D (Incorrect):** The `notch=True` parameter has no bearing on file format requirements. Notched box plots can be saved in any format (PNG, PDF, SVG, EPS, TIFF, etc.).

</details>

---

### 99. Question

A developer creates a box plot of prediction confidence scores for a deep learning model across 5 test batches. The scores form this distribution pattern:

```
Batch 1: tight cluster around 0.92 — very few outliers
Batch 2: wide spread from 0.55 to 0.97 — many outliers below 0.70
Batch 3: tight cluster around 0.88
Batch 4: very tight cluster around 0.94 — no outliers
Batch 5: moderate spread, median at 0.85, some outliers above 0.98
```

A reviewer looks at the box plots and asks: "Which batch shows the most **distribution shift**?" What is the correct answer and reasoning?

- [ ] **A.** Batch 4 — no outliers suggests an overly constrained model
- [ ] **B.** Batch 1 — the highest median indicates potential data leakage
- [ ] **C.** Batch 2 — the widest spread (large IQR) and multiple low outliers below 0.70 indicate the model encounters data that significantly differs from training distribution in Batch 2
- [ ] **D.** Batch 5 — high outliers above 0.98 indicate overconfidence

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Batch 4's tight clustering with no outliers (very small IQR) indicates highly consistent predictions — the model performs stably and confidently on Batch 4's data. This is a GOOD sign, not evidence of distribution shift.
- **Option B (Incorrect):** Batch 1 has a high median (0.92) with few outliers — indicating consistently high confidence. A high median suggests the model is well-calibrated for Batch 1's data. Data leakage would manifest differently (e.g., suspiciously perfect accuracy on ALL batches).
- **Option C (Correct):** **Distribution shift** (covariate shift) occurs when the test data distribution differs from the training distribution. Batch 2's wide spread and many low-confidence predictions (<0.70) strongly suggest the model encountered data patterns that differ significantly from what it was trained on — causing it to be uncertain (low confidence scores) on many examples. The large IQR (wide box) and multiple low outliers are visual indicators of this uncertainty. In production ML monitoring, a batch with suddenly wide confidence score spread triggers an investigation of data quality or distribution shift.
- **Option D (Incorrect):** Batch 5's high outliers (>0.98) represent extremely high-confidence predictions on some examples. While overconfidence can be a calibration issue, it doesn't necessarily indicate distribution shift. Distribution shift typically causes LOW confidence, not high confidence.

</details>

---

### 100. Question

A developer creates a comprehensive, capstone box plot report and asks a data science colleague to review it. The colleague identifies several TRUE and FALSE statements. Which of the following are **TRUE** statements about Matplotlib box plots? (Select all that apply)

- [ ] **A.** `patch_artist=True` is required to enable `set_facecolor()` on box patches — without it, boxes are Line2D outlines and `facecolor` has no effect
- [ ] **B.** Non-overlapping notches between two box plots definitively prove the medians are statistically significantly different at p < 0.05
- [ ] **C.** The median is more robust than the mean to outliers — extreme values change the mean dramatically but leave the median unchanged as long as they don't shift the positional middle value
- [ ] **D.** `showfliers=False` removes outlier values from the dataset before computing Q₁, Q₂, Q₃, and IQR

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C**

#### Explanations:

- **Option A (True):** Without `patch_artist=True`, boxes are rendered as `Line2D` objects (rectangular outlines only, no fill capability). `Line2D` objects do not support `set_facecolor()` — calling it would either raise an `AttributeError` or be silently ignored. `patch_artist=True` converts boxes to `PathPatch` objects that support all patch properties including `set_facecolor()`, `set_alpha()`, and `set_hatch()`.
- **Option B (False):** This is a critical misconception. Non-overlapping notches provide an **approximate 95% visual significance test** — they suggest (but do not definitively prove) that medians differ at approximately 95% confidence. "Definitively prove" and "p < 0.05" imply a rigorous hypothesis test. The notch test is a graphical approximation, not a formal statistical test. The p-value analogy is approximate, not exact. Proper significance testing requires formal tests (e.g., Mann-Whitney U, Wilcoxon signed-rank) with exact p-values.
- **Option C (True):** The median's robustness is its defining property. For the example: [1, 2, 3, 4, 1000] — mean = 202, median = 3. Changing 1000 to 1,000,000 — mean = 200,001.2, but median remains 3. The median is determined purely by the position (middle value), not by the magnitude of any individual value. This makes the median ideal for skewed distributions and data with outliers.
- **Option D (False):** `showfliers=False` is purely a **display-level** parameter. It suppresses the rendering of outlier markers but has ZERO effect on statistical computations. Q₁, Q₂, Q₃, IQR, and whisker positions are all computed from the FULL dataset including all outlier values. This is fundamentally different from outlier removal or winsorisation.

</details>
