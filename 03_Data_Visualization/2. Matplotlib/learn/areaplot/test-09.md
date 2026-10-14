# Matplotlib Area Plot Objective Test — Set 09

---

### 81. Question

A data scientist wants to visualise the ROC curve with the area under the curve (AUC) highlighted:

```python
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true, y_score)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc_score:.3f}")
ax.fill_between(fpr, tpr, alpha=0.25, color="blue")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
```

What does the blue shaded area in this ROC curve plot geometrically represent?

- [ ] **A.** The area between the ROC curve and the random classifier diagonal line
- [ ] **B.** The area under the ROC curve from FPR=0 to FPR=1, bounded below by the x-axis (FPR axis)
- [ ] **C.** The area inside the ROC curve above the TPR axis
- [ ] **D.** The confidence interval around the ROC curve

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The area between the ROC curve and the diagonal represents the "gain" over random — but the code uses `fill_between(fpr, tpr)` with `y2` defaulting to 0 (the x-axis), not to the diagonal line. Filling between the curve and the diagonal would require `fill_between(fpr, tpr, fpr, ...)`.
- **Option B (Correct):** `fill_between(fpr, tpr, alpha=0.25)` fills between the ROC curve (`y1=tpr`) and `y2=0` (the x-axis, i.e., the TPR=0 line). This shaded area IS the AUC — the area under the receiver operating characteristic curve, bounded by FPR [0,1] on the x-axis and TPR=0 on the lower baseline. AUC = 0.5 represents the diagonal random classifier, while AUC = 1.0 represents a perfect classifier that fills the entire unit square.
- **Option C (Incorrect):** The fill is bounded by the x-axis (FPR), not the y-axis (TPR). There is no notion of "above the TPR axis" in this context.
- **Option D (Incorrect):** The shaded area here is not a confidence interval. A confidence interval around the ROC curve would require bootstrapped estimates and two boundary curves around the central ROC line.

</details>

---

### 82. Question

A developer plots a precision-recall curve with threshold shading:

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_true, y_score)

fig, ax = plt.subplots()
ax.plot(recall, precision, "r-", lw=2)
ax.fill_between(recall, precision, alpha=0.2, color="crimson",
                label="AUC-PR")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
```

A data scientist argues: "The precision-recall AUC is computed by integrating precision over recall using this filled area." Is this statement consistent with the code?

- [ ] **A.** Yes — `fill_between(recall, precision)` exactly computes ∫precision d(recall) which is the AUC-PR
- [ ] **B.** Yes — the shaded area visually corresponds to the AUC-PR, which is approximated by the area under the precision-recall curve using trapezoidal integration
- [ ] **C.** No — AUC-PR requires the `sklearn.metrics.average_precision_score` function; `fill_between` only shades the region and does not compute the integral
- [ ] **D.** No — precision-recall curves cannot be visualised with `fill_between` because precision and recall are not arrays of equal length

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `fill_between` is a VISUALISATION function that shades a polygon on the axes. It performs no numerical computation, integration, or area calculation. It does not compute or output AUC-PR.
- **Option B (Partially correct but misleading):** The shaded region visually REPRESENTS the area under the precision-recall curve, which corresponds conceptually to AUC-PR. However, the claim that `fill_between` "computes" this is false — it only displays the shape.
- **Option C (Correct):** `fill_between` solely renders the shaded area as a visual artist. It performs no mathematical integration. To numerically compute AUC-PR, you must use `sklearn.metrics.average_precision_score(y_true, y_score)` (which uses the trapezoidal rule or interpolated method) or `np.trapz(precision, recall)`. The `fill_between` call makes the area visible to the viewer but does not compute its numerical value.
- **Option D (Incorrect):** `precision_recall_curve` returns arrays of compatible lengths (precision and recall arrays of the same size). `fill_between(recall, precision)` is a valid call since both arrays have equal length — the x-axis is recall and y1 is precision.

</details>

---

### 83. Question

A researcher evaluates a model's accuracy on three dataset splits and fills between the min and max:

```python
epochs = np.arange(1, 31)

# Three independent experimental runs
run1_acc = 0.7 + 0.01 * epochs + np.random.normal(0, 0.02, 30)
run2_acc = 0.68 + 0.012 * epochs + np.random.normal(0, 0.02, 30)
run3_acc = 0.72 + 0.009 * epochs + np.random.normal(0, 0.02, 30)

all_runs = np.vstack([run1_acc, run2_acc, run3_acc])
mean_acc = all_runs.mean(axis=0)
min_acc  = all_runs.min(axis=0)
max_acc  = all_runs.max(axis=0)

fig, ax = plt.subplots()
ax.plot(epochs, mean_acc, "b-", lw=2, label="Mean Accuracy")
ax.fill_between(epochs, min_acc, max_acc,
                color="blue", alpha=0.2, label="Min-Max Range")
```

What does `np.vstack([run1_acc, run2_acc, run3_acc])` produce, and what does `.min(axis=0)` return?

- [ ] **A.** `vstack` concatenates the arrays end-to-end into a 90-element vector; `.min(axis=0)` returns a scalar (the global minimum)
- [ ] **B.** `vstack` creates a 3×30 matrix (3 rows, 30 columns); `.min(axis=0)` returns a 30-element array with the minimum across the 3 runs at each epoch
- [ ] **C.** `vstack` creates a 30×3 matrix; `.min(axis=0)` returns a 3-element array with the minimum across all 30 epochs for each run
- [ ] **D.** `vstack` creates a 90×1 column vector; `.min(axis=0)` returns the minimum of the first 30 values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.vstack` stacks arrays VERTICALLY (row-wise), not end-to-end. For three 30-element 1D arrays, it produces a 3×30 matrix, not a 90-element flat vector. End-to-end concatenation would use `np.concatenate` or `np.hstack`.
- **Option B (Correct):** `np.vstack([run1, run2, run3])` stacks the three 30-element arrays as rows, producing a shape-(3,30) matrix. Each row represents one experimental run; each column represents one epoch. `.min(axis=0)` computes the minimum across axis 0 (across the 3 rows) at each column position, returning a 30-element array where each element is the smallest accuracy value across the three runs at that epoch. This `min_acc` array then serves as the lower boundary of the `fill_between` envelope.
- **Option C (Incorrect):** If `axis=0` were the epoch axis, the result would be 3 elements (one per run). But `axis=0` operates on the first dimension — in a 3×30 array, axis=0 has 3 elements (the runs). Reducing across axis=0 gives a 30-element result (one per column/epoch).
- **Option D (Incorrect):** `vstack` does not produce a column vector. It produces a 2D matrix. Even if it did, `.min(axis=0)` on a column vector would return a single scalar, not 30 values.

</details>

---

### 84. Question

A machine learning engineer plots the training and validation accuracy for 5 random seed runs, fills the envelope, and adds a legend:

```python
all_val = np.vstack([run.val_acc for run in experiments])  # shape (5, 50)
mean_v = all_val.mean(axis=0)
std_v  = all_val.std(axis=0)

ax.plot(epochs, mean_v, "r-", lw=2, label="Mean Val Acc")
ax.fill_between(epochs, mean_v - std_v, mean_v + std_v,
                color="red", alpha=0.2, label="±1 Std Dev")
```

If at epoch 30 the mean validation accuracy is 0.88 and the standard deviation is 0.03, what is the shaded band at epoch 30?

- [ ] **A.** From 0.85 to 0.88 (only below the mean)
- [ ] **B.** From 0.85 to 0.91 (mean ± 1 std dev)
- [ ] **C.** From 0.82 to 0.94 (mean ± 2 std dev)
- [ ] **D.** From 0.88 to 0.94 (only above the mean)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Only shading from below (0.85 to 0.88) would require `fill_between(epochs, mean_v - std_v, mean_v)` — the upper boundary as `mean_v`. The code uses `mean_v + std_v` as the upper boundary, so the fill extends symmetrically above and below the mean.
- **Option B (Correct):** `fill_between(epochs, mean_v - std_v, mean_v + std_v)` creates a symmetric ±1σ band around the mean. At epoch 30: `lower = 0.88 - 0.03 = 0.85`, `upper = 0.88 + 0.03 = 0.91`. The shaded band spans from y=0.85 to y=0.91, representing approximately 68% of the distribution of outcomes under Gaussian assumptions.
- **Option C (Incorrect):** A ±2 std dev band would span `0.88 ± 0.06 = [0.82, 0.94]`. The code uses `std_v` (×1), not `2 * std_v`.
- **Option D (Incorrect):** Only shading above the mean would require `fill_between(epochs, mean_v, mean_v + std_v)`. The code fills both above and below by using `mean_v - std_v` as y1 and `mean_v + std_v` as y2.

</details>

---

### 85. Question

A data scientist creates an animated area plot to show how a distribution evolves:

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.linspace(-5, 5, 200)
fill = ax.fill_between(x, np.zeros(200), alpha=0.4, color="teal")

def update(frame):
    sigma = 0.5 + 0.05 * frame
    y = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (x / sigma)**2)
    # How to update the fill_between?
    pass

anim = FuncAnimation(fig, update, frames=40, interval=80)
```

What is the correct way to update the `fill_between` patch in each animation frame?

- [ ] **A.** Directly modify the existing `fill` object's data via `fill.set_data(x, y_new)`
- [ ] **B.** Clear the axes with `ax.cla()` and re-call `ax.fill_between(x, y_new)` in each frame — this is the standard way to update fills
- [ ] **C.** Call `fill.set_paths(...)` or clear and re-draw the artist using `ax.collections.clear()` and `ax.fill_between(x, y_new)` inside the update function
- [ ] **D.** `fill_between` cannot be animated — use `ax.bar` instead

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Unlike `Line2D` objects (which have `set_data(x, y)`), `PolyCollection` objects returned by `fill_between` do not have a simple `set_data` method. Direct data mutation is not straightforward.
- **Option B (Partially correct):** `ax.cla()` (clear axes) followed by re-calling `ax.fill_between` works but is computationally expensive as it destroys and recreates all artists including axis formatting each frame. This approach requires resetting titles, labels, and limits in every frame.
- **Option C (Correct):** The recommended efficient approaches for animating `fill_between` are: (1) `ax.collections.clear()` removes all `PolyCollection` artists (fills) from the axes without clearing the entire axes, then `ax.fill_between(x, y_new)` adds a fresh one; or (2) `fill.set_paths(verts)` with manually computed vertex arrays updates the polygon geometry in-place. For production animations, the `ax.collections.clear()` + `fill_between` pattern is most commonly used due to its simplicity.
- **Option D (Incorrect):** `fill_between` is fully animatable. Many Matplotlib animation examples use `fill_between` for distribution evolution, confidence band animations, and physics wave simulations.

</details>

---

### 86. Question

A developer uses `fill_between` to visualise time-series sales data but receives the following error:

```
ValueError: x and y1 must have equal lengths
```

The developer's code is:

```python
months = pd.date_range("2024-01-01", periods=12, freq="MS")   # 12 dates
sales  = np.array([100, 120, 95, 140, 115, 130, 145, 160, 135, 150, 170, 180])

plt.fill_between(months, sales)
```

What is the most likely cause and fix?

- [ ] **A.** `pd.date_range` is not compatible with `fill_between` — use `np.arange(12)` for the x-axis instead
- [ ] **B.** Matplotlib's `fill_between` requires numeric x-values; `pd.date_range` objects must be explicitly converted to matplotlib date numbers using `matplotlib.dates.date2num`
- [ ] **C.** The error is caused by a Pandas version mismatch; upgrading Pandas resolves it
- [ ] **D.** Modern versions of Matplotlib (≥3.4) directly support `pd.DatetimeIndex` in `fill_between` — the error may indicate a version incompatibility or a different array length mismatch

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Modern Matplotlib does support `pd.date_range` objects and `datetime` values in plot functions. Replacing with `np.arange(12)` would work but would lose the date-based x-axis formatting.
- **Option B (Incorrect):** Since Matplotlib ≥3.0, datetime objects including `pd.Timestamp` and `pd.DatetimeIndex` are handled automatically. Manual conversion with `date2num` is no longer required for basic usage.
- **Option C (Incorrect):** The `ValueError` is a data length mismatch error, not a Pandas version compatibility issue. Pandas version upgrades do not fix array length validation in Matplotlib.
- **Option D (Correct):** Matplotlib ≥3.4 natively supports `pd.DatetimeIndex` in plotting functions. The `ValueError: x and y1 must have equal lengths` specifically indicates that `len(months) ≠ len(sales)`. If `months` has 12 elements and `sales` has 12 elements, the code should work. The error would occur if there were an actual array length discrepancy (e.g., if `sales` had 11 elements, or if the date range produced a different count). The fix is to verify array lengths before calling `fill_between`.

</details>

---

### 87. Question

A developer creates a chart with multiple `fill_between` calls and then calls `plt.legend()`. Only some fills appear in the legend — the ones WITH `label=` appear; the ones without do not. A colleague suggests: "You should add `label="_nolegend_"` to hide specific fills from the legend." Is this correct?

- [ ] **A.** No — there is no such convention in Matplotlib; unlabelled artists never appear in the legend by default
- [ ] **B.** Yes — `label="_nolegend_"` is a Matplotlib convention where any artist with a label starting with an underscore is excluded from the automatically generated legend
- [ ] **C.** No — to exclude artists from the legend you must use `ax.legend(handles=[...])` with a manually filtered list
- [ ] **D.** Yes — but only for `Line2D` objects, not for `PolyCollection` from `fill_between`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While unlabelled artists don't appear in the legend by default, the `_nolegend_` convention is also valid and explicitly supported.
- **Option B (Correct):** Matplotlib has an official convention where any artist whose label starts with an underscore (`_`) is automatically excluded from `plt.legend()`. Specifically, `label="_nolegend_"` is the standard string used to explicitly mark an artist as "do not include in legend." This is useful when you want to add a `label=` (perhaps for programmatic access via `get_label()`) but don't want it to appear in the auto-generated legend. Artists without any `label=` argument get default labels like `"_line0"`, `"_collection0"`, etc. — all starting with underscore, which is why they don't appear in legends automatically.
- **Option C (Incorrect):** While manual filtering via `ax.legend(handles=[...])` works, `label="_nolegend_"` is a cleaner, officially supported alternative that doesn't require manual list management.
- **Option D (Incorrect):** The underscore-label convention works for ALL Matplotlib artist types — `Line2D`, `PolyCollection` (from `fill_between`), `PathCollection` (from `scatter`), `Patch` (from `bar`), and others.

</details>

---

### 88. Question

A data engineer creates an area chart for memory usage over time:

```python
time_s = np.arange(0, 3600, 10)   # 360 data points (1 hour, every 10 seconds)
memory_mb = 512 + 100 * np.sin(time_s / 600) + 50 * np.random.normal(0, 1, 360)

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(time_s, memory_mb, 512,
                where=(memory_mb > 512),
                color="red", alpha=0.3, interpolate=True, label="Above 512 MB")
ax.fill_between(time_s, memory_mb, 512,
                where=(memory_mb <= 512),
                color="green", alpha=0.3, interpolate=True, label="Below 512 MB")
ax.axhline(512, color="black", lw=1.5, linestyle="--", label="Baseline 512 MB")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Memory Usage (MB)")
```

What does `fill_between(time_s, memory_mb, 512, ...)` fill relative to the dashed baseline?

- [ ] **A.** The area from `memory_mb` down to y=0 (the x-axis), wherever the condition is True
- [ ] **B.** The area between `memory_mb` and the constant horizontal line y=512, wherever the condition is True — above the baseline in red, below the baseline in green
- [ ] **C.** The area between the minimum and maximum of `memory_mb` wherever the condition is True
- [ ] **D.** The area between `memory_mb` and y=1024 (2×512) wherever the condition is True

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The `y2=512` argument (not 0) sets the lower boundary of the fill at 512 MB. The fill does not extend to the x-axis (y=0); it extends to the 512 MB baseline.
- **Option B (Correct):** `fill_between(time_s, memory_mb, 512, ...)` fills between `y1=memory_mb` and `y2=512` at each time point. Where `memory_mb > 512`, the fill extends upward from the 512 MB baseline to the memory curve (red) — representing excess memory consumption. Where `memory_mb <= 512`, the fill extends downward from the 512 MB baseline to the memory curve (green) — representing memory headroom. The `axhline(512)` dashed line marks the baseline that separates these two conditional fills.
- **Option C (Incorrect):** `fill_between` fills between two explicitly specified curves/values, not between the statistical min and max of an array. Filling between min/max would require pre-computing those statistics and using them as `y1` and `y2`.
- **Option D (Incorrect):** 1024 MB (2×512) is not involved in the fill boundary at all. The boundary is explicitly the scalar value 512 passed as `y2`.

</details>

---

### 89. Question

A data scientist creates a publication-quality area plot and uses the following colour palette with a dark background style:

```python
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(0, 4*np.pi, 400)
y = np.sin(x)

ax.fill_between(x, y, 0,
                where=(y >= 0), color="#00D1FF", alpha=0.35, label="Positive")
ax.fill_between(x, y, 0,
                where=(y < 0),  color="#FF6B6B", alpha=0.35, label="Negative")
ax.plot(x, y, color="white", lw=2)
ax.set_facecolor("#1A1A2E")
fig.patch.set_facecolor("#1A1A2E")
ax.legend(facecolor="#333355", edgecolor="white", labelcolor="white")
```

What does `ax.set_facecolor("#1A1A2E")` specifically change?

- [ ] **A.** The colour of all text elements on the axes
- [ ] **B.** The background colour of the axes data area (the rectangle where data is plotted)
- [ ] **C.** The colour of the entire figure canvas including margins
- [ ] **D.** The colour of the fill_between patches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Text colour is controlled by `ax.tick_params(colors=...)`, `ax.xaxis.label.set_color(...)`, or the `labelcolor` and `color` kwargs on text-related functions. `set_facecolor` does not affect text.
- **Option B (Correct):** `ax.set_facecolor(color)` sets the background colour of the axes "face" — the rectangular data plotting area bounded by the four axis spines. In the code, this is set to a deep dark navy `#1A1A2E`. This creates the dark background behind the fills and the white sine curve, giving the chart a dark-mode aesthetic. Note that `fig.patch.set_facecolor` separately controls the outer figure background.
- **Option C (Incorrect):** The entire figure canvas background (including margins outside the axes) is controlled by `fig.patch.set_facecolor(...)`. The `ax.set_facecolor` only affects the inner data rectangle.
- **Option D (Incorrect):** `fill_between` patch colours are set by the `color=` argument in each `fill_between` call. `ax.set_facecolor` cannot retroactively change the colour of patches that have already been added with explicit `color=` specifications.

</details>

---

### 90. Question

A data scientist creates a production-ready area chart and receives a peer review comment: "The chart's shaded region is beautiful, but readers can't tell which area represents which data series because the colours are too similar." Which of the following strategies BEST resolves this issue?

- [ ] **A.** Increase the DPI of the saved figure to make the colours sharper and more distinct
- [ ] **B.** Use maximally contrasting colours (such as red vs. blue), add distinct `label=` to each fill, and ensure `plt.legend()` is called so readers can identify each region
- [ ] **C.** Switch from `fill_between` to a bar chart, which is more legible
- [ ] **D.** Add a `title=` argument directly to `fill_between` to label each region inline

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** DPI affects pixel resolution (sharpness) of the saved image but does not change colour hue or saturation. If two colours are visually similar at 100 DPI, they remain similar at 300 DPI — the distinction is a colour palette design issue, not a resolution issue.
- **Option B (Correct):** The fix addresses both the visual and informational problems: (1) Choosing contrasting colours (e.g., using different hues like red and blue rather than two shades of blue) increases pre-attentive colour discrimination. (2) Adding `label=` to each `fill_between` call and calling `plt.legend()` provides a clear mapping between colours and data series in a legend box. Together, these resolve the identification problem without switching chart type.
- **Option C (Incorrect):** Switching to a bar chart is a drastic, chart-type change that would eliminate the area-over-time visual encoding entirely. The review comment is about colour clarity, not chart type appropriateness. Bar charts and area charts serve different purposes.
- **Option D (Incorrect):** `fill_between` does not have a `title=` argument. Artist titles are set at the axes level (`ax.set_title`) or via annotation text (`ax.annotate`). A `title=` keyword would cause a `TypeError`.

</details>
