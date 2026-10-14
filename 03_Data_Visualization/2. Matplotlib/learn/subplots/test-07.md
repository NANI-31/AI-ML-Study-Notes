# Matplotlib Subplots Objective Test — Set 07

---

### Question 61

To design an asymmetric multi-panel layout where some subplots span multiple rows or columns, a developer uses Matplotlib's `GridSpec` engine:

```python
fig = plt.figure()
gs = fig.add_gridspec(3, 3)
print(type(gs))
```

What type of object is `gs`?

- [ ] A) `<class 'numpy.ndarray'>`
- [ ] B) `<class 'matplotlib.gridspec.GridSpec'>`
- [ ] C) `<class 'matplotlib.axes.Axes'>`
- [ ] D) `<class 'list'>`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `<class 'matplotlib.gridspec.GridSpec'>`

**Explanation:**

- **A is incorrect:** `GridSpec` is a geometry specification object, not a NumPy data array.
- **B is correct:** `fig.add_gridspec(nrows, ncols)` instantiates a `matplotlib.gridspec.GridSpec` object. This class acts as a flexible virtual grid manager that allows cells to be sliced and combined using Python slice syntax to create subplots of varying spans and dimensions.
- **C is incorrect:** `GridSpec` does not create axes automatically; it provides coordinates that are subsequently passed to `fig.add_subplot()`.
- **D is incorrect:** It is a dedicated Matplotlib layout class, not a standard Python list.

</details>

---

### Question 62

In a $3 \times 3$ grid governed by `gs = fig.add_gridspec(3, 3)`, which Python slice notation creates a wide banner subplot spanning across the **entire top row** (all three columns of Row 0)?

- [ ] A) `ax_banner = fig.add_subplot(gs[:, 0])`
- [ ] B) `ax_banner = fig.add_subplot(gs[0, 0:2])`
- [ ] C) `ax_banner = fig.add_subplot(gs[1, :])`
- [ ] D) `ax_banner = fig.add_subplot(gs[0, :])` (or `gs[0, 0:3]`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax_banner = fig.add_subplot(gs[0, :])` (or `gs[0, 0:3]`)

**Explanation:**

- **A is incorrect:** `gs[:, 0]` spans all rows in Column 0 (a full-height tall left column), not a horizontal top banner.
- **B is incorrect:** `gs[0, 0:2]` only spans columns 0 and 1, leaving column 2 uncovered.
- **C is incorrect:** `gs[1, :]` spans the middle row (Row 1), not the top row (Row 0).
- **D is correct:** In 2D slice notation `gs[row_slice, col_slice]`: `0` selects Row 0, and `:` selects all available columns ($0, 1, 2$). Therefore, `gs[0, :]` spans the entire top row from the leftmost edge to the rightmost edge.

</details>

---

### Question 63

A complex dashboard requires a large focal panel spanning the bottom-left $2 \times 2$ block of a $3 \times 3$ grid (Rows 1 and 2, Columns 0 and 1). Which slice syntax allocates this specific block?

- [ ] A) `ax_focal = fig.add_subplot(gs[1:, :2])`
- [ ] B) `ax_focal = fig.add_subplot(gs[:2, 1:])`
- [ ] C) `ax_focal = fig.add_subplot(gs[1:2, 0:1])`
- [ ] D) `ax_focal = fig.add_subplot(gs[2, 2])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax_focal = fig.add_subplot(gs[1:, :2])`

**Explanation:**

- **A is correct:** In Python slicing:
  - `1:` selects rows from index 1 to the end (Rows 1 and 2).
  - `:2` selects columns up to index 2 exclusive (Columns 0 and 1).
  Together, `gs[1:, :2]` covers the 4 cells in the bottom-left $2 \times 2$ quadrant: $(1, 0), (1, 1), (2, 0), (2, 1)$.
- **B is incorrect:** `gs[:2, 1:]` selects Rows 0 and 1, Columns 1 and 2 (the top-right $2 \times 2$ block).
- **C is incorrect:** `gs[1:2, 0:1]` selects only a single cell $(1, 0)$ due to exclusive end indices.
- **D is incorrect:** `gs[2, 2]` selects only the single bottom-right corner cell.

</details>

---

### Question 64

An engineer wants to display an image alongside a thin vertical colorbar panel using a 1-row, 2-column layout. The image must occupy $75\%$ of the figure width, and the colorbar must occupy $25\%$.

Which `GridSpec` initialization implements this exact relative width distribution?

- [ ] A) `fig.add_gridspec(1, 2, column_fractions=[0.75, 0.25])`
- [ ] B) `fig.add_gridspec(1, 2, width_scales=[3, 1])`
- [ ] C) `fig.add_gridspec(1, 2, width_ratios=[3, 1])`
- [ ] D) `fig.add_gridspec(2, 1, ratio=0.75)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `fig.add_gridspec(1, 2, width_ratios=[3, 1])`

**Explanation:**

- **A is incorrect:** `column_fractions` is not a valid parameter in `GridSpec` or `add_gridspec`.
- **B is incorrect:** `width_scales` is not a recognized Matplotlib keyword argument.
- **C is correct:** `GridSpec` accepts the `width_ratios` parameter as a sequence of relative weights. With `width_ratios=[3, 1]`, the total weight is $3 + 1 = 4$. Column 0 receives $\frac{3}{4} = 75\%$ of the total available width, and Column 1 receives $\frac{1}{4} = 25\%$.
- **D is incorrect:** `nrows=2, ncols=1` creates two vertically stacked rows rather than two horizontal columns, and `ratio` is not a valid parameter.

</details>

---

### Question 65

A financial time-series chart consists of two stacked panels:
- Top panel: Stock Price chart (occupies $\frac{2}{3}$ of the total height)
- Bottom panel: MACD oscillator indicator (occupies $\frac{1}{3}$ of the total height)

Which `GridSpec` configuration establishes this vertical proportion?

- [ ] A) `fig.add_gridspec(2, 1, height_ratios=[2, 1])`
- [ ] B) `fig.add_gridspec(1, 2, height_ratios=[2, 1])`
- [ ] C) `fig.add_gridspec(2, 1, vsplit=[0.66, 0.33])`
- [ ] D) `fig.add_gridspec(2, 1, y_weights=[2, 1])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig.add_gridspec(2, 1, height_ratios=[2, 1])`

**Explanation:**

- **A is correct:** In a 2-row, 1-column layout (`nrows=2, ncols=1`), the `height_ratios` parameter sets the relative vertical heights of the rows. Passing `height_ratios=[2, 1]` allocates $\frac{2}{3} \approx 66.7\%$ of the vertical space to Row 0 (Price) and $\frac{1}{3} \approx 33.3\%$ to Row 1 (MACD).
- **B is incorrect:** `nrows=1, ncols=2` creates two side-by-side columns rather than two vertically stacked rows.
- **C is incorrect:** `vsplit` is not a valid Matplotlib argument for row height partitioning.
- **D is incorrect:** `y_weights` is not a recognized parameter in `add_gridspec`.

</details>

---

### Question 66

A data scientist is creating a classic joint-marginal distribution plot:
1. Main panel: 2D Bivariate Scatter plot
2. Top panel: 1D Histogram of variable $X$ aligned directly above the scatter plot
3. Right panel: 1D Histogram of variable $Y$ aligned directly to the right of the scatter plot

Which `GridSpec` configuration provides the ideal geometry for this layout?

- [ ] A) `gs = fig.add_gridspec(1, 3)`
- [ ] B) `gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])`
- [ ] C) `gs = fig.add_gridspec(3, 3, circular=True)`
- [ ] D) `gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])`

**Explanation:**

- **A is incorrect:** A $1 \times 3$ grid places three subplots horizontally side-by-side.
- **B is correct:** A $2 \times 2$ grid with `width_ratios=[4, 1]` and `height_ratios=[1, 4]` partitions the space into:
  - Cell `[1, 0]`: The large main scatter plot (width weight 4, height weight 4).
  - Cell `[0, 0]`: The top marginal histogram (width weight 4, height weight 1), sharing the x-axis with the scatter plot.
  - Cell `[1, 1]`: The right marginal histogram (width weight 1, height weight 4), sharing the y-axis with the scatter plot.
  - Cell `[0, 1]`: Top-right corner left empty via `delaxes()` or unoccupied.
- **C is incorrect:** `circular` is not a valid argument for `GridSpec`.
- **D is incorrect:** Equal ratios $[1, 1]$ make the marginal histograms identical in size to the main scatter plot, wasting canvas space.

</details>

---

### Question 67

Instead of manually creating a `Figure` and calling `fig.add_gridspec()`, how can custom `width_ratios` and `height_ratios` be passed directly into `plt.subplots()` in a single line?

- [ ] A) `plt.subplots(2, 2, ratios={'w': [2, 1], 'h': [1, 2]})`
- [ ] B) `plt.subplots(2, 2, width_ratios=[2, 1], height_ratios=[1, 2])`
- [ ] C) `plt.subplots(2, 2, layout_ratios=[[2, 1], [1, 2]])`
- [ ] D) `plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2]})`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.subplots(2, 2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2]})`

**Explanation:**

- **A is incorrect:** `ratios` is not a top-level parameter for `plt.subplots()`.
- **B is incorrect:** `width_ratios` and `height_ratios` cannot be passed directly as top-level arguments to `plt.subplots()`; they must be nested within `gridspec_kw`.
- **C is incorrect:** `layout_ratios` is not a valid argument for `plt.subplots()`.
- **D is correct:** `plt.subplots()` exposes the `gridspec_kw` dictionary argument, which forwards configuration keywords directly to the underlying `GridSpec` constructor. Passing `gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 2]}` applies these ratio weights seamlessly in a single clean call.

</details>

---

### Question 68

What advanced class in `matplotlib.gridspec` allows a developer to nest a secondary sub-grid inside an individual cell slice of a parent `GridSpec` (e.g., nesting a $2 \times 2$ mini-grid inside cell `gs[1, 1]`)?

- [ ] A) `SubplotTree`
- [ ] B) `NestedMatrix`
- [ ] C) `GridSpecFromSubplotSpec`
- [ ] D) `RecursiveGrid`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `GridSpecFromSubplotSpec`

**Explanation:**

- **A is incorrect:** `SubplotTree` does not exist in the Matplotlib library.
- **B is incorrect:** `NestedMatrix` is a fictitious class name.
- **C is correct:** `matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[1, 1])` is the specialized class designed to subdivide a single cell of a parent `GridSpec` into an independent nested sub-grid with its own rows, columns, and spacing ratios.
- **D is incorrect:** `RecursiveGrid` is not a valid Matplotlib class.

</details>

---

### Question 69

In the Matplotlib object model, what is the exact conceptual distinction between a `GridSpec` instance and a `SubplotSpec` instance?

- [ ] A) A `GridSpec` defines the overall geometry of the entire grid (total rows, columns, and ratios), whereas a `SubplotSpec` represents a specific localized cell or sliced region (such as `gs[0, :]`) that can be converted into an `Axes`.
- [ ] B) `GridSpec` is for 3D plots, while `SubplotSpec` is for 2D plots.
- [ ] C) `GridSpec` is deprecated and replaced by `SubplotSpec`.
- [ ] D) `SubplotSpec` is a list of figures, while `GridSpec` is a list of datasets.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A `GridSpec` defines the overall geometry of the entire grid (total rows, columns, and ratios), whereas a `SubplotSpec` represents a specific localized cell or sliced region (such as `gs[0, :]`) that can be converted into an `Axes`.

**Explanation:**

- **A is correct:** `GridSpec` is the master grid template defining the grid structure ($\text{nrows} \times \text{ncols}$, spacing, ratios). When a developer slices a `GridSpec` (e.g., `gs[0, 1]` or `gs[1:, :2]`), the indexing operation returns a `SubplotSpec`. The `SubplotSpec` encapsulates the specific positional bounding coordinates of that slice and is passed to `fig.add_subplot(subplot_spec)` to instantiate the actual `AxesSubplot`.
- **B is incorrect:** `GridSpec` and `SubplotSpec` are dimensionality-agnostic grid layout managers and apply equally to both 2D and 3D projection axes.
- **C is incorrect:** Neither class is deprecated; they are complementary components of Matplotlib's core layout engine.
- **D is incorrect:** `SubplotSpec` represents spatial grid slices, not lists of figures or datasets.

</details>

---

### Question 70

Which of the following statements regarding Matplotlib `GridSpec` layouts are **correct**?

1. Slicing a `GridSpec` uses standard Python 0-based indexing and slice ranges.
2. `width_ratios` can accept arbitrary positive floating-point or integer weights (e.g., `[3.5, 1.2]`).
3. An `Axes` created via `fig.add_subplot(gs[0, :])` has its `.get_subplotspec()` method return the underlying `SubplotSpec`.
4. Subplots created via `GridSpec` cannot share axes with other subplots.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because `fig.add_subplot(gs[r, c], sharex=other_ax)` fully supports `sharex` and `sharey`. Also, Statements 2 and 3 are omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all correct. Slicing `gs[start:stop]` adheres to 0-based, stop-exclusive Python indexing; `width_ratios` accepts arbitrary positive floats and ints; and `ax.get_subplotspec()` returns the underlying `SubplotSpec` object.

</details>

---
