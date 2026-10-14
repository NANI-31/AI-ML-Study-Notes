# Matplotlib Scatter Plot Objective Test — Set 10

---

### 91. Question

A junior developer writes the following script intending to generate a colormapped scatter plot with a colorbar:

```python
import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5]
scores = [50, 60, 70, 80, 90]

scatter = plt.scatter(hours, scores, c="blue", cmap="viridis")
cbar = plt.colorbar(scatter)
plt.show()
```

When run, the colorbar appears completely empty or Matplotlib raises a `TypeError` / warning. What is the bug?

- [ ] **A.** The parameter name `cmap` is spelled incorrectly.
- [ ] **B.** The developer passed a literal color string (`c="blue"`) instead of a numeric data array (e.g., `c=scores`); a colormap and colorbar require a continuous sequence of numbers to establish a scalar normalization range.
- [ ] **C.** Matplotlib does not allow colorbars on scatter plots with fewer than 10 points.
- [ ] **D.** The array `scores` must be converted to float32 before calling `plt.scatter()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `cmap` is spelled correctly.
- **Option B (Correct):** When `c="blue"` is passed, Matplotlib sets a uniform color and bypasses colormapping. The resulting `PathCollection` has no scalar data array or normalization bounds (`norm.vmin` and `norm.vmax` remain `None`). Attempting to build a `plt.colorbar()` on an unmapped artist fails or renders an uninitialized colorbar because there is no numeric range to map. The developer must pass a numeric array: `c=scores`.
- **Option C (Incorrect):** There is no minimum point requirement for colorbars.
- **Option D (Incorrect):** Standard Python integer lists are fully supported.

</details>

---

### 92. Question

A machine learning engineer extracts a feature column from a Scikit-Learn pipeline as a 2D column vector with shape `(100, 1)` and pairs it with a 1D target vector of shape `(100,)`:

```python
import matplotlib.pyplot as plt
import numpy as np

X = np.random.randn(100, 1)  # 2D shape (100, 1)
y = np.random.randn(100)  # 1D shape (100,)

plt.scatter(X, y)
```

In certain strict backend operations, array shape mismatches between `(N, 1)` and `(N,)` can cause broadcast warnings or dimensional errors. What is the most Pythonic NumPy method to guarantee `X` is flattened into a 1D vector?

- [ ] **A.** `X = X.reshape(100, 100)`
- [ ] **B.** `X = list(X.items())`
- [ ] **C.** `X = X.ravel()` or `X = X.squeeze()`
- [ ] **D.** `X = np.concatenate([X, X])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `reshape(100, 100)` requires 10,000 elements, raising a `ValueError`.
- **Option B (Incorrect):** NumPy arrays do not have an `.items()` dictionary method.
- **Option C (Correct):** In NumPy, `.ravel()` (or `.flatten()`) collapses a multi-dimensional array into a contiguous 1D array of shape `(100,)`. Alternatively, `.squeeze()` removes all single-dimensional axes. Passing `X.ravel()` guarantees strict 1D compatibility across all Matplotlib functions.
- **Option D (Incorrect):** Concatenation doubles the number of elements to 200, creating an even worse shape mismatch against `y` of length 100.

</details>

---

### 93. Question

A developer creates a scatter plot where points are colored by continuous salary values (`c=salary, cmap="plasma"`). They then invoke `plt.legend()`, expecting a discrete legend showing salary ranges.

Instead, the rendered canvas displays a legend box containing only a single purple dot with the label `"None"`. Why did this happen?

- [ ] **A.** The computer's graphics card ran out of legend cache memory.
- [ ] **B.** The colormap name must be capitalized as `cmap="Plasma"`.
- [ ] **C.** `plt.legend()` only works if all salary values are negative numbers.
- [ ] **D.** `plt.legend()` creates entries for discrete artist handles, not continuous gradients; a `PathCollection` with continuous colormapping represents a single artist, so `plt.legend()` only draws a single representative proxy dot. Continuous values must be communicated via `plt.colorbar()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** GPU legend cache memory exhaustion is not a real Matplotlib failure mode.
- **Option B (Incorrect):** Colormap names are case-insensitive or standard lowercase; capitalization is not the cause.
- **Option C (Incorrect):** Data sign has no effect on whether legends can display continuous gradients.
- **Option D (Correct):** `plt.legend()` is architected for discrete categorical series (where each distinct group has its own artist and label). A continuous colormapped scatter plot is a single unified `PathCollection` artist. Attempting to call `plt.legend()` simply draws a single default sample glyph from the collection. To visually explain continuous gradients, the proper graphic element is a **colorbar** (`plt.colorbar()`). If discrete legend boxes are specifically desired, points must be categorized into discrete bins with custom proxy artist handles.

</details>

---

### 94. Question

An automated data pipeline generates 2,000 scatter plot figures in a batch loop and saves them as PNG images:

```python
for i in range(2000):
  fig, ax = plt.subplots()
  ax.scatter(data[i, :, 0], data[i, :, 1])
  fig.savefig(f"plot_{i}.png")
```

After running for 10 minutes, the server crashes with `MemoryError: Out of Memory`. What critical Matplotlib resource management call was omitted inside the loop?

- [ ] **A.** `plt.close(fig)`
- [ ] **B.** `fig.delete()`
- [ ] **C.** `gc.reclaim_canvas()`
- [ ] **D.** `ax.empty()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, every figure created via `plt.subplots()` or `plt.figure()` is registered globally in the pyplot state machine GUI manager to enable interactive display. Saving a figure (`fig.savefig()`) does **not** destroy the figure or free its memory. In long-running batch loops, figures accumulate in RAM until memory is completely exhausted. Invoking `plt.close(fig)` (or `plt.close('all')`) explicitly unregisters the figure, allowing Python's garbage collector to immediately release all associated buffers.
- **Option B (Incorrect):** Method `fig.delete()` does not exist in Matplotlib.
- **Option C (Incorrect):** `gc.reclaim_canvas()` is not a valid Python garbage collection method.
- **Option D (Incorrect):** Method `ax.empty()` does not exist in the Matplotlib API.

</details>

---

### 95. Question

Examine the following comprehensive 4D scatter plot implementation from the reference material:

```python
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    x,
    y,
    c=values,
    s=sizes,
    cmap="viridis",
    alpha=0.6,
    edgecolors="black",
    linewidths=0.5,
)

fig.colorbar(scatter, ax=ax, label="Value Metric")

m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, m * x_line + b, linestyle="--", color="red", label="Trend")

idx = np.argmax(y)
ax.scatter(
    x[idx], y[idx], s=300, marker="*", color="gold", edgecolors="black", zorder=5
)
ax.annotate(
    "Peak",
    xy=(x[idx], y[idx]),
    xytext=(10, 10),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black"),
)

ax.legend()
```

How many distinct physical and statistical dimensions of information are simultaneously encoded in this visualization?

- [ ] **A.** Exactly 2 dimensions ($x$ and $y$ positions only).
- [ ] **B.** Exactly 3 dimensions ($x, y$ coordinates and red trendline).
- [ ] **C.** At least 4 distinct data dimensions ($x$-position, $y$-position, marker area $s$ for feature 3, colormap gradient $c$ for feature 4), plus regression trend modeling and peak outlier diagnostics.
- [ ] **D.** Exactly 8 dimensions, because there are 8 function calls.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This falsely ignores the continuous color, marker size, trendline, and peak annotation dimensions.
- **Option B (Incorrect):** This fails to recognize marker area $s$ and colorbar gradient $c$ as independent visual variables.
- **Option C (Correct):** This script implements a canonical multi-dimensional visual encoding:
  1. Horizontal Cartesian coordinate ($x$ feature)
  2. Vertical Cartesian coordinate ($y$ feature)
  3. Marker size area ($s$ scaled by `sizes` array)
  4. Marker color hue/luminance ($c$ mapped through `'viridis'` with a labeled colorbar)
  Furthermore, it integrates statistical trend estimation ($\hat{y} = mx + b$), overplotting mitigation (`alpha=0.6`, `edgecolors`), and dynamic peak identification with vector annotation.
- **Option D (Incorrect):** Number of function calls does not equal data dimensionality.

</details>

---

### 96. Question

What happens if an analyst attempts to colormap a scatter plot where all observations in the color array `c` are identical (e.g., `c = [42.0, 42.0, 42.0, 42.0]`)?

- [ ] **A.** Matplotlib raises a fatal `ZeroDivisionError: float division by zero` because $\text{vmax} - \text{vmin} = 0$.
- [ ] **B.** Matplotlib gracefully handles the zero-width normalization interval by mapping all points to the exact midpoint ($0.5$) of the colormap without throwing an unhandled exception.
- [ ] **C.** All points are rendered in solid black.
- [ ] **D.** The scatter plot is aborted and an empty canvas is shown.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While naive mathematical normalization $\frac{x - \text{vmin}}{\text{vmax} - \text{vmin}}$ would divide by zero, Matplotlib's `Normalize` class specifically checks for $\text{vmin} == \text{vmax}$ and bypasses division by zero.
- **Option B (Correct):** When all values in `c` are identical, $\text{vmin} = \text{vmax}$. Matplotlib detects this condition, expands the interval slightly (or maps directly), and assigns the central midpoint color stop ($0.5$) of the chosen colormap to all markers.
- **Option C (Incorrect):** Points do not default to black; they receive the colormap midpoint color.
- **Option D (Incorrect):** The plot is rendered normally rather than aborted.

</details>

---

### 97. Question

A data analyst working with a Pandas DataFrame `df` with columns `['age', 'income', 'credit_score', 'loan_amount']` wants to generate a scatter plot using Pandas' built-in wrapper syntax. Which syntax is valid?

- [ ] **A.** `df.plot.scatter(x='age', y='income', c='credit_score', s=df['loan_amount']/100, colormap='viridis')`
- [ ] **B.** `df.scatter('age', 'income')`
- [ ] **C.** `df.to_scatter(x='age', y='income')`
- [ ] **D.** `df.plot(type='scatter_plot', columns=['age', 'income'])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Pandas provides a convenient DataFrame plotting API via `df.plot.scatter(x, y, c=..., s=..., colormap=...)`. Column names can be passed as strings for coordinates and colors, and custom array formulas can be supplied to `s`.
- **Option B (Incorrect):** `df.scatter` is not a direct method on DataFrame objects; it lives under `df.plot.scatter`.
- **Option C (Incorrect):** Method `df.to_scatter` does not exist in Pandas.
- **Option D (Incorrect):** Parameter `type='scatter_plot'` is not a valid argument in Pandas `df.plot()`.

</details>

---

### 98. Question

When building a bubble chart where marker size `s` encodes population, `plt.colorbar()` can represent color, but how can an informative **marker size legend** (showing small, medium, and large circle sizes with corresponding numbers) be added to the plot?

- [ ] **A.** By calling `plt.sizebar()`.
- [ ] **B.** By setting `plt.legend(show_sizes=True)`.
- [ ] **C.** Marker size legends are physically impossible in Matplotlib.
- [ ] **D.** By creating dummy/proxy scatter artists with empty coordinate arrays `[]` matching sample benchmark sizes (e.g., `plt.scatter([], [], s=size, c='gray', label=f'{pop}')`) and invoking `plt.legend()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Function `plt.sizebar` does not exist in Matplotlib.
- **Option B (Incorrect):** Argument `show_sizes=True` is not supported in `plt.legend()`.
- **Option C (Incorrect):** Size legends are completely feasible using standard Matplotlib artist handles.
- **Option D (Correct):** The standard Matplotlib technique for building a bubble size legend is creating "proxy artists":
  ```python
  for pop, size in [(100_000, 20), (500_000, 100), (1_000_000, 200)]:
    plt.scatter([], [], s=size, c="gray", alpha=0.6, label=f"{pop:,}")
  plt.legend(title="Population", scatterpoints=1)
```
  Because the coordinate arrays are empty (`[], []`), nothing is drawn on the canvas, but the legend registers the marker size and label perfectly.

</details>

---

### 99. Question

A scientist generates a scatter plot containing 500,000 individual data points and saves it as a vector PDF for publication (`fig.savefig("figure.pdf")`). The resulting PDF file is over 150 Megabytes in size, and opening the document crashes Adobe Acrobat and web browsers.

Which single argument passed to `plt.scatter()` rasterizes only the dense point collection into a high-DPI bitmap while keeping axes, labels, ticks, and text as lightweight, infinitely scalable vector elements?

- [ ] **A.** `bitmap=True`
- [ ] **B.** `compress="gzip"`
- [ ] **C.** `rasterized=True`
- [ ] **D.** `vector_mode=False`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Parameter `bitmap=True` is not recognized by `plt.scatter()`.
- **Option B (Incorrect):** Parameter `compress="gzip"` is not an argument for `plt.scatter()`.
- **Option C (Correct):** In Matplotlib vector backends (PDF, SVG, EPS), every single scatter marker is saved as a discrete vector geometric path object by default. For 500,000 points, this creates millions of XML/PostScript path instructions, resulting in massive file sizes (100MB+) and sluggish rendering. Adding `rasterized=True` to `plt.scatter(x, y, rasterized=True)` instructs the backend to pre-render the point cloud into a rasterized image layer, while keeping all axis lines, tick labels, titles, and text as sharp, crisp vectors. File size drops to ~1–2 MB.
- **Option D (Incorrect):** Parameter `vector_mode=False` does not exist in Matplotlib.

</details>

---

### 100. Question

A machine learning engineer evaluates a trained regression model on test data by constructing a residual scatter plot ($\text{Residuals } e_i \text{ vs. Predicted } \hat{y}_i$).

The scatter plot displays two distinct visual anomalies simultaneously:
1. The residuals bow in a pronounced parabolic arc (curving below zero at low $\hat{y}$, rising high above zero at moderate $\hat{y}$, and dropping below zero at high $\hat{y}$).
2. The vertical spread of the residuals expands from $\pm 2$ at $\hat{y} = 10$ to $\pm 45$ at $\hat{y} = 100$.

What is the comprehensive diagnostic conclusion from this scatter pattern?

- [ ] **A.** The model is an optimal, perfectly calibrated homoscedastic linear estimator.
- [ ] **B.** The model suffers from **both non-linearity** (the true underlying relationship has polynomial/non-linear curvature that the model failed to capture) **and heteroscedasticity** (error variance expands systematically with magnitude). The engineer should introduce non-linear polynomial/spline terms and apply a variance-stabilizing transformation (such as $\log(y)$).
- [ ] **C.** The dataset has zero residuals.
- [ ] **D.** The scatter plot is suffering from screen resolution aliasing.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** An optimal model produces a structureless, constant-width horizontal cloud centered at zero.
- **Option B (Correct):** This diagnostic analysis demonstrates the power of residual scatter plots:
  - The parabolic curvature directly proves that the model underfits a **non-linear** relationship (e.g., the true function has quadratic or higher-order terms).
  - The expanding vertical fan of residuals proves that the error variance is non-constant (**heteroscedasticity**).
  Consequently, both structural defects must be remedied: adding non-linear terms (polynomial features or non-linear regression) to straighten out the curvature, and applying log or Box-Cox transformations (or weighted least squares) to stabilize the variance.
- **Option C (Incorrect):** Residuals are clearly non-zero across the entire domain.
- **Option D (Incorrect):** The funnel curvature reflects true statistical model inadequacy, not display aliasing.

</details>

---
