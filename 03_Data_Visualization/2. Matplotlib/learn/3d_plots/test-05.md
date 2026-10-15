# 3D Matplotlib — Test 05: Meshgrid Generation & Wireframe Plots

---

### Question 41

A data scientist calls:

```python
x = np.linspace(-3, 3, 7)    # 7 points
y = np.linspace(-2, 2, 5)    # 5 points
X, Y = np.meshgrid(x, y)
```

What are the shapes of `X` and `Y`?

- [ ] A) `X.shape = (7,)`, `Y.shape = (5,)` — meshgrid returns 1-D arrays
- [ ] B) `X.shape = (7, 5)`, `Y.shape = (7, 5)` — x-size first
- [ ] C) `X.shape = (5, 7)`, `Y.shape = (5, 7)` — y-size (rows) by x-size (columns)
- [ ] D) `X.shape = (35,)`, `Y.shape = (35,)` — flattened Cartesian product

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `X.shape = (5, 7)`, `Y.shape = (5, 7)` — y-size (rows) by x-size (columns)

**Explanation:**
- **A is incorrect:** `np.meshgrid` returns 2-D arrays (grids), not 1-D input arrays. The entire purpose of meshgrid is to expand 1-D coordinate arrays into 2-D grids.
- **B is incorrect:** `np.meshgrid(x, y)` by default uses indexing='xy' (Cartesian convention). In this convention, the output arrays have shape `(len(y), len(x))` — rows correspond to y-values, columns correspond to x-values. Shape is `(5, 7)`, not `(7, 5)`.
- **C is correct:** With `np.meshgrid(x, y)` in the default 'xy' indexing mode, `X` and `Y` both have shape `(len(y), len(x)) = (5, 7)`. Each row of `X` contains a repeated copy of `x` (the 7 x-values). Each column of `Y` contains a repeated copy of `y` (the 5 y-values). This arrangement ensures that `Z = f(X, Y)` can be computed element-wise with the correct correspondence.
- **D is incorrect:** Meshgrid does not flatten the output. It preserves the 2-D grid structure. Flattening would be done explicitly with `.ravel()` or `.flatten()`.

</details>

---

### Question 42

After generating a meshgrid and computing `Z = np.sin(X) * np.cos(Y)` for a 5×7 grid, a developer attempts to plot a wireframe:

```python
ax.plot_wireframe(x, y, Z)   # passes 1-D x and y instead of 2-D X and Y
```

What is the most likely outcome?

- [ ] A) The wireframe renders correctly — `plot_wireframe` internally converts 1-D arrays to meshgrid
- [ ] B) A `ValueError` or `IndexError` is raised because `Z` has shape `(5, 7)` but `x` has shape `(7,)` and `y` has shape `(5,)`, causing a shape mismatch in the internal surface construction
- [ ] C) The wireframe renders as a flat 2D line because the 1-D arrays reduce the third dimension to zero
- [ ] D) Only the diagonal of the surface is drawn because 1-D `x` and `y` are broadcast together as if paired element-by-element

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `ValueError` or `IndexError` is raised because `Z` has shape `(5, 7)` but `x` has shape `(7,)` and `y` has shape `(5,)`, causing a shape mismatch in the internal surface construction

**Explanation:**
- **A is incorrect:** `plot_wireframe` does not automatically call `np.meshgrid` internally. It expects its `X`, `Y`, and `Z` arguments to all be 2-D arrays of the same shape. Passing 1-D arrays causes shape mismatches during the wireframe polygon construction step.
- **B is correct:** `ax.plot_wireframe(X, Y, Z)` requires all three inputs to be 2-D arrays of matching shape. If `x` (shape `(7,)`) and `y` (shape `(5,)`) are passed instead of `X` and `Y` (both shape `(5, 7)`), the function attempts to construct wireframe line segments using mismatched array dimensions, resulting in a `ValueError` or array indexing error.
- **C is incorrect:** Matplotlib does not silently collapse dimensions when arrays have mismatched shapes. An error is raised rather than producing a degenerate 2D line.
- **D is incorrect:** Broadcasting in NumPy requires explicit support in the calling function. `plot_wireframe` does not broadcast 1-D coordinate arrays together — it uses indexing-based segment construction that requires matching 2-D shapes.

</details>

---

### Question 43

A researcher generates a grid and plots a wireframe:

```python
x = np.linspace(0, 4, 5)      # [0, 1, 2, 3, 4]
y = np.linspace(0, 3, 4)      # [0, 1, 2, 3]
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
ax.plot_wireframe(X, Y, Z, color="navy")
```

How many rectangular grid cells does this wireframe contain?

- [ ] A) 20 cells (5 × 4)
- [ ] B) 12 cells (4 × 3 — the number of intervals, not the number of points)
- [ ] C) 9 cells (3 × 3 — minimum dimension squared)
- [ ] D) 8 cells (perimeter cells only)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 12 cells (4 × 3 — the number of intervals, not the number of points)

**Explanation:**
- **A is incorrect:** The number of grid cells is not determined by the number of points. With N points in a direction, there are N-1 intervals (segments between adjacent points). The number of rectangular cells is (Nx-1) × (Ny-1) = (5-1) × (4-1) = 4 × 3 = 12.
- **B is correct:** A wireframe connects adjacent grid points with line segments. With 5 x-points and 4 y-points, there are (5-1)=4 x-intervals and (4-1)=3 y-intervals, forming 4×3 = **12** rectangular cells in the grid. The wireframe draws the grid lines that form these 12 cells.
- **C is incorrect:** 9 = 3×3 would result only if both dimensions had 4 points (4-1=3 intervals each). The x-dimension has 5 points, giving 4 intervals.
- **D is incorrect:** Wireframes fill the entire interior grid, not just the perimeter. A perimeter-only wire outline would be a different kind of visualization.

</details>

---

### Question 44

A data scientist uses:

```python
X, Y = np.meshgrid(np.linspace(-np.pi, np.pi, 100),
                   np.linspace(-np.pi, np.pi, 100))
Z = np.sin(X) + np.cos(Y)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color="teal")
```

With `rstride=10` and `cstride=10` on a 100×100 grid, how many wireframe lines are drawn in the row direction and in the column direction?

- [ ] A) 100 row lines and 100 column lines (stride is ignored)
- [ ] B) 10 row lines and 10 column lines
- [ ] C) 10 row lines and 10 column lines, but only interior lines (excluding the boundary)
- [ ] D) Approximately 10 row lines and 10 column lines (rows 0, 10, 20, ..., 90 — so 10 lines per direction)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Approximately 10 row lines and 10 column lines (rows 0, 10, 20, ..., 90 — so 10 lines per direction)

**Explanation:**
- **A is incorrect:** `rstride` and `cstride` are the subsampling strides. A stride of 10 means every 10th row/column is selected, dramatically reducing the number of lines drawn.
- **B is incorrect:** The exact count needs more precision. With a 100×100 grid and `rstride=10`, rows are selected at indices 0, 10, 20, ..., 90 (10 rows selected). The last row (index 99) may not be included depending on the stride implementation. So approximately 10 row lines and 10 column lines is correct, but option B does not explain the mechanism.
- **C is incorrect:** Boundary handling varies by implementation. The first row (index 0) is typically included. The claim that boundaries are excluded is incorrect.
- **D is correct:** `rstride=10` selects every 10th row: indices 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 — giving 10 row slices. Similarly, `cstride=10` selects every 10th column, giving 10 column slices. The wireframe draws horizontal polylines along each selected row and vertical polylines along each selected column, producing approximately 10+10=20 wire lines total (plus edge handling), compared to 200 lines for stride=1.

</details>

---

### Question 45

A developer wants to visualize a 2D Gaussian surface on a 50×50 grid:

```python
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2) / 2)
```

She calls `ax.plot_wireframe(X, Y, Z)` without specifying `rstride` or `cstride`. What are the default stride values and how many grid lines are drawn?

- [ ] A) Default stride = 1; all 50 row lines and all 50 column lines are drawn (100 lines total)
- [ ] B) Default stride = 10; approximately 5 row lines and 5 column lines (10 lines total)
- [ ] C) Default stride = 1; results in 50 row polylines and 50 column polylines (100 total lines, one per grid row/column)
- [ ] D) Default stride = 5; 10 row lines and 10 column lines (20 lines total)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Default stride = 1; results in 50 row polylines and 50 column polylines (100 total lines, one per grid row/column)

**Explanation:**
- **A is incorrect:** While the statement that default stride=1 is correct, describing this as "100 lines" is slightly misleading. The wireframe draws one horizontal polyline per selected row and one vertical polyline per selected column, so with stride=1 and a 50×50 grid, there are 50 row polylines + 50 column polylines = 100 polylines. The description is quantitatively correct but option C expresses the mechanism more clearly.
- **B is incorrect:** The default stride for `plot_wireframe` is 1, not 10.
- **C is correct:** By default, `plot_wireframe` uses `rstride=1` and `cstride=1`, meaning every row and every column of the grid is rendered as a wire line. For a 50×50 grid, this produces 50 row-direction polylines and 50 column-direction polylines. With a fine 50×50 grid, this can produce a very dense wireframe that may look nearly solid.
- **D is incorrect:** The default stride is 1, not 5.

</details>

---

### Question 46

A researcher compares two wireframe plots of the same surface:

```python
# Plot A
ax1.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

# Plot B  
ax2.plot_wireframe(X, Y, Z, rstride=20, cstride=20)
```

The underlying grid `X, Y` has shape `(100, 100)`. Which statement correctly describes the visual and performance difference?

- [ ] A) Plot A renders more slowly and appears denser; Plot B renders faster and appears more sparsely gridded with larger mesh cells visible
- [ ] B) Plot A and Plot B appear identical because `plot_wireframe` automatically normalizes stride values
- [ ] C) Plot B appears smoother because larger strides cause Matplotlib to interpolate more points between grid lines
- [ ] D) Plot A raises a `MemoryError` because rendering all 100×100 = 10,000 cells simultaneously exceeds Matplotlib's memory limit

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Plot A renders more slowly and appears denser; Plot B renders faster and appears more sparsely gridded with larger mesh cells visible

**Explanation:**
- **A is correct:** `rstride=1, cstride=1` selects every row and every column, creating 100+100=200 wire polylines for a 100×100 grid. Each polyline has 100 segments. This produces a visually dense grid and requires Matplotlib to render many more polygon lines, making it slower. `rstride=20, cstride=20` selects only 5 rows (0,20,40,60,80) and 5 columns, creating 5+5=10 wire polylines total. The result is a visually sparse grid with large visible gaps between wire lines — faster to render, useful for seeing the surface shape without visual clutter.
- **B is incorrect:** `plot_wireframe` does not normalize strides. The two stride values produce dramatically different outputs with no automatic equalization.
- **C is incorrect:** Larger strides reduce the number of wire lines (coarser sampling). They do not add interpolated points or make the result smoother. The opposite is true: sparse strides produce jagged, coarsely approximated surfaces, especially where the function has high curvature.
- **D is incorrect:** A 100×100 = 10,000-cell wireframe is not computationally unusual for Matplotlib. It renders (possibly slowly), but it does not raise a `MemoryError` under normal conditions.

</details>

---

### Question 47

A data analyst wants to create a meshgrid using matrix (ij) indexing instead of the default Cartesian (xy) indexing. She uses:

```python
x = np.linspace(0, 1, 4)   # shape (4,)
y = np.linspace(0, 1, 3)   # shape (3,)
Xi, Yi = np.meshgrid(x, y, indexing='ij')
Xc, Yc = np.meshgrid(x, y, indexing='xy')
```

What are the shapes of `Xi` and `Xc`?

- [ ] A) `Xi.shape = (4, 3)`, `Xc.shape = (3, 4)`
- [ ] B) `Xi.shape = (3, 4)`, `Xc.shape = (3, 4)` — both produce the same shape
- [ ] C) `Xi.shape = (4, 3)`, `Xc.shape = (4, 3)` — both produce the same shape
- [ ] D) `Xi.shape = (3, 4)`, `Xc.shape = (4, 3)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `Xi.shape = (4, 3)`, `Xc.shape = (3, 4)`

**Explanation:**
- **A is correct:** `np.meshgrid` with `indexing='ij'` (matrix indexing) makes the output shape `(len(x), len(y)) = (4, 3)`. The first argument `x` varies along axis 0 (rows) and `y` varies along axis 1 (columns). With `indexing='xy'` (the default Cartesian indexing), the output shape is `(len(y), len(x)) = (3, 4)`. The first argument `x` varies along axis 1 (columns) and `y` varies along axis 0 (rows). The two conventions produce transposed grids.
- **B is incorrect:** The two indexing conventions produce different shapes. `'ij'` produces `(4, 3)` and `'xy'` produces `(3, 4)`.
- **C is incorrect:** Both cannot be `(4, 3)`. The `'xy'` convention produces `(len(y), len(x)) = (3, 4)`, not `(4, 3)`.
- **D is incorrect:** The shapes are swapped from the correct answer. `'ij'` → `(4, 3)` and `'xy'` → `(3, 4)`, not the other way around.

</details>

---

### Question 48

A researcher creates a wireframe visualization of the Rosenbrock function:

```python
x = np.linspace(-2, 2, 50)
y = np.linspace(-1, 3, 50)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2   # Rosenbrock function

ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, color="darkgreen", alpha=0.6)
```

A colleague says: "The Rosenbrock function has its global minimum at `(1, 1)`, where `Z=0`. If the minimum is not visible in the wireframe, it's because `rstride=5` skips that point." Is this argument valid?

- [ ] A) Yes — stride subsampling can skip the exact minimum grid point if it doesn't fall on a selected index
- [ ] B) No — `plot_wireframe` always includes the local extrema regardless of stride, because it uses adaptive sampling
- [ ] C) No — the minimum at `(1, 1)` is always visible because `(1, 1)` is always at the center of the meshgrid for the given ranges
- [ ] D) No — wireframe opacity `alpha=0.6` masks the minimum but it is still calculated at that point

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Yes — stride subsampling can skip the exact minimum grid point if it doesn't fall on a selected index

**Explanation:**
- **A is correct:** `rstride=5` and `cstride=5` select wire lines at every 5th row and column index. The point `(1, 1)` maps to a specific grid index based on the `linspace` discretization. If that index is not a multiple of 5, the wire line through the minimum is not drawn, making the minimum's exact position less visually prominent. The wireframe still approximates the region around the minimum, but the exact grid point at or nearest to `(1, 1)` may be skipped.
- **B is incorrect:** `plot_wireframe` does not use adaptive sampling. It mechanically samples every `rstride`-th row and every `cstride`-th column without any special treatment for extrema. If the minimum falls at index 25 and the stride is 5, it happens to be included (25 is divisible by 5). If it falls at index 27, it is skipped.
- **C is incorrect:** The x-range is [-2, 2] (length 4, centered at 0) and the y-range is [-1, 3] (length 4, centered at 1). The minimum `(x=1, y=1)` corresponds to x being at 75% of the x-range and y being at 50% of the y-range — it is not at the center of the meshgrid.
- **D is incorrect:** Alpha (opacity) controls the transparency of drawn elements. It does not "mask" the calculation of the minimum — the function evaluates at all grid points. The minimum is either rendered (if its row/column is selected by the stride) or not drawn (if skipped by stride). Alpha does not affect which points are drawn.

</details>

---

### Question 49

A data analyst needs to visualize the distance function `Z = np.sqrt(X**2 + Y**2)` on the domain `[-5, 5] × [-5, 5]` using a wireframe. She creates the grid:

```python
x = np.linspace(-5, 5, 51)   # 51 points including 0
y = np.linspace(-5, 5, 51)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)
```

At the origin `(0, 0)`, `Z=0`. What is the minimum value of Z visible in the wireframe, and does the wireframe correctly show a sharp cone apex at the origin?

- [ ] A) Minimum Z = 0 at the origin; the wireframe shows a perfect sharp cone apex
- [ ] B) Minimum Z = the value at the nearest non-zero grid point; the wireframe misses the exact apex if the 0-grid-point is not selected by the stride
- [ ] C) Minimum Z = 0 at the origin; the wireframe always shows the apex because `Z=0` at the origin creates a discontinuity that Matplotlib highlights
- [ ] D) Minimum Z = 0; but the wireframe cannot render a cone apex and instead renders a flat circular patch at z=0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Minimum Z = 0 at the origin; the wireframe shows a perfect sharp cone apex

**Explanation:**
- **A is correct:** With `np.linspace(-5, 5, 51)`, the 51 points include the origin at index 25 (the exact midpoint of 51 points: indices 0 through 50, middle at 25). With `rstride=1, cstride=1` (default), all 51 rows and 51 columns are drawn, including the row/column through index 25 (the origin). `Z[25, 25] = sqrt(0² + 0²) = 0`. The wireframe does pass through the origin and the cone apex at z=0 is visible. The sharpness of the apex depends on the grid resolution near the origin.
- **B is incorrect:** Since `np.linspace(-5, 5, 51)` includes x=0 exactly at index 25, and default stride=1 includes all rows/columns, the apex at (0,0) is correctly included in the wireframe.
- **C is incorrect:** Matplotlib does not have special detection for discontinuities or extrema. The apex appears because the gridpoint exists at the origin, not due to any special handling.
- **D is incorrect:** Wireframes can represent cone shapes. The cone geometry is rendered as a mesh of triangular/quadrilateral facets that converge at the apex. No flat patch is added.

</details>

---

### Question 50

A researcher accidentally uses `np.meshgrid` with `indexing='ij'` and then passes the result directly to `ax.plot_wireframe`. The grid is for a 4×3 region (x has 4 points, y has 3 points). With `indexing='ij'`, `X.shape = (4, 3)` and `Y.shape = (4, 3)`. She computes `Z = X + Y` (also shape `(4, 3)`) and plots:

```python
ax.plot_wireframe(X, Y, Z)
```

Does the wireframe render correctly despite using `'ij'` indexing?

- [ ] A) No — `plot_wireframe` requires `'xy'` indexed grids and raises a `ValueError` for `'ij'` indexed grids
- [ ] B) Yes — `plot_wireframe` accepts any 2-D arrays of matching shape for `X`, `Y`, and `Z`; the indexing convention used to create the grid does not affect the wireframe rendering
- [ ] C) Partially — the wireframe renders but with x and y axes transposed
- [ ] D) No — `'ij'` indexing produces non-rectangular grids that `plot_wireframe` cannot render

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `plot_wireframe` accepts any 2-D arrays of matching shape for `X`, `Y`, and `Z`; the indexing convention used to create the grid does not affect the wireframe rendering

**Explanation:**
- **A is incorrect:** `plot_wireframe` has no awareness of how the input arrays were created. It treats `X`, `Y`, `Z` as 2-D arrays of grid point coordinates and constructs wire segments from adjacent elements, regardless of the `meshgrid` indexing convention used.
- **B is correct:** `plot_wireframe` simply iterates over adjacent grid points in the 2-D arrays, connecting them with line segments. Whether the arrays were created with `indexing='xy'` or `indexing='ij'` is irrelevant — both produce valid 2-D arrays of matching shape. The function renders whatever geometry is described by the (X, Y, Z) coordinate arrays.
- **C is incorrect:** With `'ij'` indexing, the arrays have shape `(4, 3)` with x varying along axis 0 and y along axis 1. The wireframe will render the surface correctly in 3D. There is no axis transposition visible in the 3D view — the x and y coordinate values in the arrays directly map to the 3D world x and y axes.
- **D is incorrect:** Both `'xy'` and `'ij'` indexing produce rectangular grids. The indexing convention only affects which array axis corresponds to which coordinate direction. Both produce regular rectangular grids that `plot_wireframe` handles correctly.

</details>
