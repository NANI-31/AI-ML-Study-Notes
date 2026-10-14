# Matplotlib Scatter Plot Objective Test — Set 05

---

### 41. Question

A developer wants to add an explanatory color gradient legend (a colorbar) to a scatter plot where points are colored by continuous humidity measurements. Which code snippet demonstrates the correct pattern?

- [ ] **A.**
  ```python
  scatter = ax.scatter(temperature, pressure, c=humidity, cmap="Blues")
  cbar = fig.colorbar(scatter, ax=ax)
  cbar.set_label("Relative Humidity (%)")
  ```
- [ ] **B.**
  ```python
  ax.scatter(temperature, pressure, c=humidity, cmap="Blues")
  plt.add_colorbar(humidity, label="Relative Humidity (%)")
  ```
- [ ] **C.**
  ```python
  scatter = ax.scatter(temperature, pressure, c=humidity, cmap="Blues")
  cbar = ax.create_colorbar(scatter)
  ```
- [ ] **D.**
  ```python
  fig.colorbar(humidity, cmap="Blues", name="Relative Humidity (%)")
  ```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `ax.scatter()` returns a `PathCollection` (which implements the `ScalarMappable` interface). Passing this artist object to `fig.colorbar(scatter, ax=ax)` binds the colorbar to the scatter artist's normalization and colormap, and `cbar.set_label(...)` annotates the color scale with its physical unit.
- **Option B (Incorrect):** `plt.add_colorbar` is not a valid Matplotlib function.
- **Option C (Incorrect):** `Axes` objects do not have a `.create_colorbar()` method; colorbars are created on the parent `Figure` or via `plt.colorbar()`.
- **Option D (Incorrect):** Passing raw data arrays like `humidity` directly to `fig.colorbar()` raises a `TypeError` because colorbars require a `ScalarMappable` object.

</details>

---

### 42. Question

What happens if a developer executes `plt.colorbar()` without saving or passing the return object of `plt.scatter()`, as shown below?

```python
import matplotlib.pyplot as plt

plt.figure()
plt.scatter([1, 2, 3], [4, 5, 6], c=[10, 20, 30], cmap="plasma")
plt.colorbar()
plt.show()
```

- [ ] **A.** Matplotlib raises a fatal `AttributeError: no artist found`.
- [ ] **B.** The program enters an infinite rendering loop.
- [ ] **C.** Matplotlib crashes with a `SegmentationFault` in the backend renderer.
- [ ] **D.** In standard single-axes scripts, Matplotlib falls back to `plt.gca().collections[0]` and successfully renders the colorbar, but in multi-axes figures it can cause severe ambiguity or raise a `RuntimeError: No mappable was found to use for colorbar`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not raise an `AttributeError`; in simple single-axes scripts, it falls back to inspecting the current active axes.
- **Option B (Incorrect):** The function executes deterministically without entering an infinite rendering loop.
- **Option C (Incorrect):** No memory corruption or segmentation fault occurs.
- **Option D (Correct):** When invoked with no arguments in simple interactive pyplot scripts, `plt.colorbar()` attempts to locate the current active `ScalarMappable` on the current axes (`plt.gca().collections[0]`). While this often succeeds in single-plot scripts, relying on implicit detection is dangerous: in multi-axes figures or when non-mappable artists are added, Matplotlib fails with `RuntimeError: No mappable was found to use for colorbar; you must first create a mappable before calling colorbar()`. Best practice is to explicitly bind: `cbar = plt.colorbar(scatter_handle)`.

</details>

---

### 43. Question

After instantiating a colorbar object `cbar = plt.colorbar(scatter)`, which method correctly sets the descriptive text and physical unit displayed alongside the colorbar?

- [ ] **A.** `cbar.title("Pressure (kPa)")`
- [ ] **B.** `cbar.set_label("Pressure (kPa)")`
- [ ] **C.** `cbar.set_text("Pressure (kPa)")`
- [ ] **D.** `cbar.annotate("Pressure (kPa)")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `cbar.title` is not the standard method for setting the axis label along the length of the colorbar (though `cbar.ax.set_title()` can place a title atop it).
- **Option B (Correct):** `cbar.set_label("Label Text")` is the official Matplotlib API method to assign a descriptive label to the colorbar axis.
- **Option C (Incorrect):** `cbar.set_text` does not exist on `Colorbar` instances.
- **Option D (Incorrect):** `cbar.annotate` is not a valid method on colorbar objects.

</details>

---

### 44. Question

A data analyst configures a scatter plot with explicit normalization parameters:

```python
plt.scatter(x, y, c=z, cmap="viridis", vmin=10, vmax=50)
```

How does Matplotlib assign colors to points where $z = 5$ and $z = 75$?

- [ ] **A.** Points with $z = 5$ are plotted in red, and points with $z = 75$ are plotted in blue.
- [ ] **B.** Points outside the $[10, 50]$ window are dropped and omitted from the canvas.
- [ ] **C.** Points with $z \le 10$ (including $z = 5$) are clamped to the minimum color stop ($0.0$) of `'viridis'`, and points with $z \ge 50$ (including $z = 75$) are clamped to the maximum color stop ($1.0$).
- [ ] **D.** Matplotlib raises a `ValueError: OutOfBoundsData`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Colors come strictly from `'viridis'` (purple to yellow); out-of-bounds points are not replaced with red or blue unless an `over`/`under` color is explicitly configured via `cmap.set_over()` or `cmap.set_under()`.
- **Option B (Incorrect):** `vmin` and `vmax` govern color normalization mapping, not coordinate geometric clipping; data points are still plotted.
- **Option C (Correct):** `vmin` and `vmax` define the lower and upper bounds of the scalar range that maps to $[0.0, 1.0]$ in the colormap. By default, values below `vmin` are clamped to $0.0$ (the lowest color in the colormap, deep purple in viridis), and values above `vmax` are clamped to $1.0$ (the highest color, bright yellow).
- **Option D (Incorrect):** Clamping is the intended default behavior; no error is raised.

</details>

---

### 45. Question

A dataset's continuous feature $z$ is plotted with `vmin = 20` and `vmax = 100` using standard linear normalization. What normalized floating-point position in the interval $[0.0, 1.0]$ on the colormap does a data value of $z = 60$ map to?

- [ ] **A.** $0.25$
- [ ] **B.** $0.40$
- [ ] **C.** $0.60$
- [ ] **D.** $0.50$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $0.25$ corresponds to $z = 20 + 0.25 \times 80 = 40$.
- **Option B (Incorrect):** $0.40$ corresponds to $z = 20 + 0.40 \times 80 = 52$.
- **Option C (Incorrect):** $0.60$ assumes a baseline of $0$ ($60 / 100 = 0.6$), failing to subtract `vmin`.
- **Option D (Correct):** Standard linear normalization (`matplotlib.colors.Normalize`) scales a value $z$ to a fraction $f \in [0, 1]$ using the formula:
  $$f = \frac{z - \text{vmin}}{\text{vmax} - \text{vmin}} = \frac{60 - 20}{100 - 20} = \frac{40}{80} = 0.50$$
  Therefore, $z = 60$ maps to exactly $0.50$, the midpoint of the colormap.

</details>

---

### 46. Question

A scatter plot uses `cmap="coolwarm"`, `vmin = -10.0`, and `vmax = 10.0`. An outlier sensor malfunction produces a reading of $z = +45.0$. What color does this point receive on the rendered canvas?

- [ ] **A.** The exact maximum color stop ($1.0$) of the colormap (deep saturated red).
- [ ] **B.** Pure black (`#000000`) indicating invalid data.
- [ ] **C.** Neutral white/gray because it exceeds the measurement limit.
- [ ] **D.** The point is masked and rendered completely transparent.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib's default colormap normalization, any data value greater than or equal to `vmax` is clamped to the maximum colormap position ($1.0$). In `'coolwarm'`, position $1.0$ is deep saturated red.
- **Option B (Incorrect):** Black is only used if explicitly designated via `cmap.set_over('black')`.
- **Option C (Incorrect):** White/gray is the neutral midpoint at $0.0$, not the extreme outlier color.
- **Option D (Incorrect):** The point is not masked or transparent; it is rendered in full opacity at the clamped extreme color.

</details>

---

### 47. Question

A data scientist creates a $1 \times 2$ subplot grid and wants a single colorbar that clearly steals width only from the right-hand subplot (`ax2`). Which code snippet accomplishes this properly?

- [ ] **A.** `plt.colorbar(ax=fig)`
- [ ] **B.** `fig.colorbar(scatter2, ax=ax2)`
- [ ] **C.** `ax2.colorbar(scatter2)`
- [ ] **D.** `fig.colorbar(scatter2, parent="right")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Passing the `Figure` to `ax` is invalid syntax.
- **Option B (Correct):** Specifying `ax=ax2` inside `fig.colorbar(scatter2, ax=ax2)` informs Matplotlib to resize and steal layout space specifically from `ax2`, placing the colorbar adjacent to that particular subplot.
- **Option C (Incorrect):** `Axes` objects do not have a `.colorbar()` method.
- **Option D (Incorrect):** `parent="right"` is not a valid parameter for `colorbar()`.

</details>

---

### 48. Question

A financial analyst is visualizing profit/loss percentages with a diverging colormap (`'coolwarm'`). The minimum value in the dataset is $-10\%$ and the maximum value is $+50\%$.

If the plot is run with default auto-normalization (`vmin = -10, vmax = 50`), what serious visual flaw occurs?

- [ ] **A.** The plot raises a `ZeroDivisionError` because the data crosses zero.
- [ ] **B.** The negative values become invisible against the canvas background.
- [ ] **C.** The neutral midpoint of the colormap (white/gray) is automatically mapped to $(50 + (-10)) / 2 = +20\%$, meaning a company making a $+20\%$ profit will falsely appear neutral, and modest profits (like $+5\%$) will misleadingly appear as negative losses (blue).
- [ ] **D.** The colorbar cannot display percentage tick labels.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Crossing zero is completely valid and does not raise errors.
- **Option B (Incorrect):** Negative values are rendered in blue, not invisible.
- **Option C (Correct):** By default, Matplotlib maps `vmin` to $0.0$ and `vmax` to $1.0$. The midpoint color ($0.5$, which is neutral white in diverging maps) is placed at $\frac{\text{vmin} + \text{vmax}}{2} = \frac{-10 + 50}{2} = +20\%$. As a result, positive profits between $0\%$ and $20\%$ are assigned blue "cold/loss" shades! To preserve visual truth, the analyst must either set symmetric limits (`vmin=-50, vmax=50`) or use `TwoSlopeNorm(vcenter=0, vmin=-10, vmax=50)`.
- **Option D (Incorrect):** Ticks and formatters can easily display percentage symbols.

</details>

---

### 49. Question

When continuous data spans several orders of magnitude (e.g., from $1$ to $1,000,000$), linear color normalization causes almost all points to share the lowest color bin. Which normalization class from `matplotlib.colors` distributes the colormap evenly across powers of 10?

- [ ] **A.** `matplotlib.colors.LinearNorm(1, 1000000)`
- [ ] **B.** `matplotlib.colors.LogNorm(vmin=1, vmax=1000000)`
- [ ] **C.** `matplotlib.colors.PowerNorm(gamma=0)`
- [ ] **D.** `matplotlib.colors.ExponentialNorm(base=10)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `LinearNorm` does not exist; standard linear normalization is `Normalize`.
- **Option B (Correct):** `matplotlib.colors.LogNorm(vmin, vmax)` applies a base-10 logarithmic transformation to the input values before mapping to $[0.0, 1.0]$. Each decade ($1\text{--}10, 10\text{--}100, 100\text{--}1000$, etc.) receives an equal fraction of the colormap spectrum.
- **Option C (Incorrect):** `PowerNorm` applies $x^\gamma$, and $\gamma=0$ is invalid.
- **Option D (Incorrect):** `ExponentialNorm` is not a standard Matplotlib normalization class.

</details>

---

### 50. Question

Which of the following statements regarding colorbars in Matplotlib are **correct**?

1. The colorbar object exposes an `.ax` attribute representing the actual `Axes` on which the colorbar gradient and ticks are drawn.
2. A single colorbar can be bound to multiple subplots by passing a list of axes: `fig.colorbar(scatter, ax=[ax1, ax2])`.
3. Colorbars cannot display horizontal orientations; they are strictly vertical.
4. Setting `shrink=0.8` reduces the length of the colorbar to 80% of the parent axis height.

- [ ] **A.** 1, 2, and 4 only
- [ ] **B.** 1 and 3 only
- [ ] **C.** 2 and 3 only
- [ ] **D.** 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Statements 1, 2, and 4 are all correct. Statement 1 is valid because `cbar.ax` exposes the underlying axis for full customization; Statement 2 is valid because passing `ax=[ax1, ax2]` creates a shared colorbar across subplots; and Statement 4 is valid because `shrink=0.8` scales colorbar dimension to 80%. Statement 3 is false because horizontal colorbars are natively supported.
- **Option B (Incorrect):** Statement 3 is false (`orientation='horizontal'` is fully supported), and this option omits valid Statements 2 and 4.
- **Option C (Incorrect):** Statement 3 is false, and this option fails to include valid Statements 1 and 4.
- **Option D (Incorrect):** Statement 3 is false, and this option fails to include valid Statements 1 and 2.

</details>

---
