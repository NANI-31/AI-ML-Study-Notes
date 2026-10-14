# Matplotlib Line Plot Objective Test — Set 07

---

### 61. Question

A web analytics manager attempts to plot daily website visits (ranging between 20,000 and 85,000) alongside daily conversion rates (ranging between 0.018 and 0.035) on the same axes using standard calls:

```python
plt.plot(days, page_views, label="Page Views")
plt.plot(days, conversion_rate, label="Conversion Rate")
plt.legend()
plt.show()
```

What visual flaw occurs in the resulting chart?

- [ ] **A.** Matplotlib raises an `IncompatibleScaleException` and refuses to render.
- [ ] **B.** Because the vertical axis autoscales to accommodate 85,000, the conversion rate values (around 0.02) are compressed into an indistinguishable flat line resting directly along the bottom axis ($y \approx 0$).
- [ ] **C.** The page views line is automatically converted into a bar chart to balance the visual weight.
- [ ] **D.** The conversion rate line is scaled up by a factor of 100,000 automatically.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not reject disparate numerical magnitudes; it computes the bounding box across all input data.
- **Option B (Correct):** The vertical axis autoscales from 0 to approximately 90,000. On a scale of 90,000, numbers between 0.018 and 0.035 represent less than 0.00004% of the vertical canvas height. Consequently, all subtle trends, peaks, and troughs in the conversion rate collapse into a completely flat line hugging the bottom axis.
- **Option C (Incorrect):** Matplotlib never silently changes chart types.
- **Option D (Incorrect):** Matplotlib never performs implicit normalization or unit re-scaling without user instruction.

</details>

---

### 62. Question

Which object-oriented Axes method creates a secondary, independent vertical ($y$) axis that shares the same horizontal ($x$) axis with the primary plot?

- [ ] **A.** `ax2 = ax1.twiny()`
- [ ] **B.** `ax2 = ax1.secondary_axis('y')`
- [ ] **C.** `ax2 = ax1.split('vertical')`
- [ ] **D.** `ax2 = ax1.twinx()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `ax1.twiny()` creates a shared vertical y-axis with a twin horizontal axis placed along the top of the figure.
- **Option B (Incorrect):** `secondary_axis()` is designed for direct functional transformations (e.g., Celsius to Fahrenheit), not independent multi-series scaling.
- **Option C (Incorrect):** `split()` is not a method on Matplotlib Axes.
- **Option D (Correct):** `ax1.twinx()` creates a new Axes instance (`ax2`) that overlays `ax1`, sharing the exact same x-axis scaling and position while exposing a separate, independent y-axis on the right side of the figure.

</details>

---

### 63. Question

When implementing twin axes (`ax1` on the left in blue, `ax2` on the right in red), what is an essential visualization best practice to ensure readers intuitively know which axis belongs to which series?

- [ ] **A.** Match the tick label colors and spine color of each y-axis to the color of its corresponding plotted line (e.g., via `ax1.tick_params(axis='y', labelcolor='blue')`).
- [ ] **B.** Always make both lines black and let readers guess from the legend.
- [ ] **C.** Use identical scale bounds on both axes so the numbers match.
- [ ] **D.** Suppress axis labels entirely to reduce cognitive clutter.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Dual-axis charts introduce cognitive ambiguity because viewers must determine which line belongs to which vertical scale. Color-coordinating the line, axis title, and tick labels (e.g., blue curve with blue tick labels on the left; red curve with red tick labels on the right) provides immediate, pre-attentive clarity.
- **Option B (Incorrect):** Making both lines monochrome maximizes ambiguity and leads to severe chart misinterpretation.
- **Option C (Incorrect):** Forcing identical bounds defeats the entire purpose of creating a secondary axis for disparate scales.
- **Option D (Incorrect):** Suppressing labels destroys readability.

</details>

---

### 64. Question

An analyst wants to draw an unobtrusive, dashed horizontal reference line across the entire width of the plot at $y = 0$ to demarcate positive vs. negative profit margins. Which call is most idiomatic?

- [ ] **A.** `plt.plot([0, 100], [0, 0], 'k--')`
- [ ] **B.** `plt.hline(0, color='gray')`
- [ ] **C.** `plt.grid(axis='y', value=0)`
- [ ] **D.** `plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.plot()` requires hardcoding arbitrary x-coordinates, which break or truncate if horizontal data limits shift during updates.
- **Option B (Incorrect):** `plt.hline()` is not a standard Pyplot function (the function is `hlines()`, which requires explicit `xmin` and `xmax` data bounds).
- **Option C (Incorrect):** `plt.grid()` does not accept single-value targets in this manner.
- **Option D (Correct):** `plt.axhline(y=0, ...)` draws a horizontal rule spanning from the extreme left boundary ($x=0$ in axes coordinates) to the extreme right boundary ($x=1$ in axes coordinates) at the specified data coordinate $y=0$. It dynamically spans the full axes width regardless of how x-limits are scaled or zoomed.

</details>

---

### 65. Question

A pharmaceutical researcher wants to display the mean drug response curve along with its $\pm 1$ standard deviation confidence interval shaded in transparent blue. Which function is used in conjunction with `plt.plot()` to shade the area between the upper and lower bounds?

- [ ] **A.** `plt.fill_between(x, y_lower, y_upper, color='blue', alpha=0.2)`
- [ ] **B.** `plt.shade_region(y_lower, y_upper)`
- [ ] **C.** `plt.area(x, y_lower, y_upper)`
- [ ] **D.** `plt.fill_range(x, min=y_lower, max=y_upper)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.fill_between(x, y1, y2, ...)` creates a `PolyCollection` that fills the polygon between two curves $y_1(x)$ and $y_2(x)$. Setting `color='blue'` and `alpha=0.2` creates a clean, semi-transparent uncertainty band surrounding the mean line.
- **Option B (Incorrect):** `shade_region` is not a Matplotlib function.
- **Option C (Incorrect):** `plt.area()` is a pandas/ggplot convenience method, not a Matplotlib Pyplot function.
- **Option D (Incorrect):** `fill_range` does not exist in Matplotlib.

</details>

---

### 66. Question

An IoT sensor temporarily drops off the network, resulting in missing data represented as `np.nan`:

```python
time_points = [0, 1, 2, 3, 4, 5]
readings    = [10.2, 14.5, np.nan, np.nan, 18.1, 20.4]

plt.plot(time_points, readings, 'bo-')
plt.show()
```

How does Matplotlib handle the missing values when rendering the line?

- [ ] **A.** It crashes with a `NaNValueError`.
- [ ] **B.** It replaces `np.nan` with $0.0$ and connects the line downward to the x-axis.
- [ ] **C.** It leaves a visible gap (discontinuity) between index 1 and index 4, drawing no connecting line segments across the `np.nan` values.
- [ ] **D.** It linearly interpolates across the missing coordinates and draws an unbroken line between index 1 and index 4.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib is designed to handle `np.nan` values in numeric series without raising exceptions.
- **Option B (Incorrect):** Silently replacing `NaN` with $0.0$ would be statistically misleading and corrupt the chart; Matplotlib does not do this.
- **Option C (Correct):** When Matplotlib encounters `np.nan` in coordinates passed to `plt.plot()`, it lifts the drawing pen. It does not draw vertices or line segments for `NaN` elements, leaving an honest visual break in the line curve that signals missing observations to the viewer.
- **Option D (Incorrect):** Matplotlib does not perform statistical interpolation; connecting across gaps requires explicit data preprocessing with pandas (e.g., `.interpolate()`).

</details>

---

### 67. Question

An engineer wants to apply a pre-packaged styling sheet that imparts a muted gray background with white gridlines (resembling the aesthetic of the R `ggplot2` library) across all line charts. Which command activates this theme globally?

- [ ] **A.** `plt.set_theme('ggplot')`
- [ ] **B.** `plt.style.use('ggplot')`
- [ ] **C.** `plt.theme('r-style')`
- [ ] **D.** `matplotlib.config.apply('ggplot')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `set_theme()` is the theme activation function in Seaborn (`sns.set_theme()`), not pure Matplotlib.
- **Option B (Correct):** In Matplotlib, `plt.style.use('ggplot')` loads the bundled `ggplot` stylesheet from Matplotlib's style library, configuring canvas colors, white gridlines, fonts, and property cyclers globally.
- **Option C (Incorrect):** `plt.theme()` is not a Matplotlib function.
- **Option D (Incorrect):** `matplotlib.config.apply()` does not exist.

</details>

---

### 68. Question

A financial analyst wants to mark the exact date of a major interest rate announcement on a multi-month trend line using a vertical dotted red line. Which function call produces this?

- [ ] **A.** `plt.vline(x=event_x, color='red')`
- [ ] **B.** `plt.draw_vertical(event_x, 'red')`
- [ ] **C.** `plt.line(x=event_x, orientation='vertical')`
- [ ] **D.** `plt.axvline(x=event_x, color='red', linestyle=':')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The function is `vlines()`, which requires explicit `ymin` and `ymax` data boundaries rather than automatically spanning the axes.
- **Option B (Incorrect):** `draw_vertical` is not a Matplotlib method.
- **Option C (Incorrect):** `plt.line()` does not exist in Pyplot.
- **Option D (Correct):** `plt.axvline(x=..., ...)` draws a vertical line across the full vertical span of the axes (from $y=0$ to $y=1$ in axes coordinates) at the specified data coordinate $x$. Setting `color='red'` and `linestyle=':'` renders the requested dotted red event marker.

</details>

---

### 69. Question

A data scientist has a pandas DataFrame `df` with columns `['Epoch', 'Train_Loss', 'Val_Loss']`. Which concise statement plots both loss series against `Epoch` using pandas' built-in Matplotlib wrapper?

- [ ] **A.** `df.plot(x='Epoch', y=['Train_Loss', 'Val_Loss'])`
- [ ] **B.** `plt.plot_df(df, x='Epoch')`
- [ ] **C.** `df.to_line(x='Epoch')`
- [ ] **D.** `plt.plot(df.columns)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Pandas provides a `.plot()` method that wraps Matplotlib. Supplying `x='Epoch'` and a list of column names `y=['Train_Loss', 'Val_Loss']` automatically plots both series on the same axes, applies distinct colors from the cycler, and attaches a legend labeled with the column names.
- **Option B (Incorrect):** `plt.plot_df()` is not a Matplotlib function.
- **Option C (Incorrect):** `to_line()` is not a pandas DataFrame method.
- **Option D (Incorrect):** Passing column names directly to `plt.plot()` attempts to plot string headers, not numeric loss trajectories.

</details>

---

### 70. Question

Consider the following statements regarding twin axes and reference annotations in Matplotlib line charts:

1. `ax.twinx()` generates an overlay Axes instance whose x-limits are automatically synchronized with the parent axes.
2. Legends generated on `ax1` and `ax2` via independent `legend()` calls will by default render in separate, potentially overlapping legend boxes unless combined manually.
3. `plt.fill_between()` can accept a boolean condition array via the `where` parameter to shade only regions where a condition is met (e.g., `where=(actual > forecast)`).
4. Calling `plt.style.use()` resets all user-defined figure objects created prior to the call.

Which statements are **correct**?

- [ ] **A.** 1 and 4 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `twinx()` binds the horizontal axis coordinates together so zooming or panning in x updates both axes in sync.
- **Statement 2 is correct:** Because `ax1` and `ax2` are distinct Axes instances, calling `.legend()` on each produces two independent legend boxes (often both targeting `'upper right'`), requiring developers to merge line handles manually if a unified legend is desired.
- **Statement 3 is correct:** `fill_between()` includes a `where` argument that accepts a boolean mask, allowing conditional shading (such as coloring positive profit green and negative profit red).
- **Statement 4 is incorrect:** `plt.style.use()` modifies `rcParams` for subsequent plotting commands; it does not retrospectively alter or reset already-drawn artists on active figures.
- **Conclusion:** Statements 1, 2, and 3 are correct, making **Option C** the correct answer.

</details>

---
