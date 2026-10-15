# Matplotlib Subplots Objective Test — Set 02

---

### Question 11

A data scientist executes the following statement to generate a multi-panel layout for feature distribution analysis:

```python
fig, axes = plt.subplots(2, 3)
print(type(axes), axes.shape)
```

What is printed to the console?

- [ ] A) `<class 'list'> (6,)`
- [ ] B) `<class 'tuple'> (2, 3)`
- [ ] C) `<class 'numpy.ndarray'> (2, 3)`
- [ ] D) `<class 'matplotlib.axes.AxesGrid'> (6,)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `<class 'numpy.ndarray'> (2, 3)`

**Explanation:**

- **A is incorrect:** Matplotlib returns a NumPy array, not a standard Python list.
- **B is incorrect:** The return container is a mutable NumPy array, not an immutable tuple.
- **C is correct:** When both `nrows > 1` and `ncols > 1` (here $2$ and $3$), `plt.subplots()` organizes the resulting `AxesSubplot` instances into a two-dimensional NumPy array (`numpy.ndarray`) whose `.shape` directly matches `(nrows, ncols)`—in this case, `(2, 3)`.
- **D is incorrect:** `AxesGrid` is a specialized class from `mpl_toolkits.axes_grid1`, not the return type of standard `plt.subplots()`.

</details>

---

### Question 12

A programmer attempts to initialize a four-panel grid by unpacking the `axes` array as follows:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)
```

What exception is raised during script execution?

- [ ] A) `IndexError: tuple index out of range`
- [ ] B) `TypeError: 'AxesSubplot' object is not iterable`
- [ ] C) `KeyError: 'ax4'`
- [ ] D) `ValueError: too many values to unpack (expected 4, got 2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ValueError: too many values to unpack (expected 4, got 2)`

**Explanation:**

- **A is incorrect:** Tuple indexing is not taking place; this is sequence unpacking.
- **B is incorrect:** The object being unpacked is the outer `numpy.ndarray`, not an individual `AxesSubplot`.
- **C is incorrect:** KeyErrors occur in dictionary lookups, not variable unpacking.
- **D is correct:** `plt.subplots(2, 2)` returns a 2D NumPy array with 2 rows and 2 columns (`shape=(2, 2)`). Iterating or unpacking over a 2D array yields its top-level elements—its rows! Therefore, the outer array yields exactly 2 row vectors. Because the left-hand side attempts to unpack 4 variables from a container of length 2, Python raises `ValueError: too many values to unpack (expected 4, got 2)`.

</details>

---

### Question 13

Which of the following syntax patterns correctly unpacks a $2 \times 2$ subplot grid into four individual `Axes` variables in a single line?

- [ ] A) `fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)`
- [ ] B) `fig, [ax1, ax2, ax3, ax4] = plt.subplots(2, 2)`
- [ ] C) `fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)`
- [ ] D) `fig, ax1, ax2, ax3, ax4 = plt.subplots(2, 2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)`

**Explanation:**

- **A is correct:** Because `plt.subplots(2, 2)` yields a 2D array consisting of two rows, each containing two subplots, the unpacking structure must mirror this nested hierarchy. The nested tuple `((ax1, ax2), (ax3, ax4))` unpacks the first row into `ax1` and `ax2`, and the second row into `ax3` and `ax4`.
- **B is incorrect:** A flat list `[ax1, ax2, ax3, ax4]` expects a sequence of length 4, but the 2D array has top-level length 2 (2 rows).
- **C is incorrect:** As shown in Question 12, a flat 4-tuple triggers `ValueError: too many values to unpack (expected 4, got 2)`.
- **D is incorrect:** `plt.subplots()` returns exactly a 2-tuple `(Figure, Axes)`. Attempting to unpack 5 variables raises `ValueError: not enough values to unpack (expected 5, got 2)`.

</details>

---

### Question 14

Consider the following two calls to `plt.subplots()`:

```python
fig1, axes1 = plt.subplots(1, 4)
fig2, axes2 = plt.subplots(4, 1)
```

Assuming default arguments (`squeeze=True`), what are the shapes of `axes1` and `axes2`?

- [ ] A) `axes1.shape == (1, 4)` and `axes2.shape == (4, 1)`
- [ ] B) `axes1.shape == (4,)` and `axes2.shape == (4,)`
- [ ] C) `axes1.shape == (4,)` and `axes2.shape == (4, 1)`
- [ ] D) `axes1.shape == (1, 4)` and `axes2.shape == (4,)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axes1.shape == (4,)` and `axes2.shape == (4,)`

**Explanation:**

- **A is incorrect:** Neither array retains a 2D shape with length-1 dimensions because `squeeze=True` eliminates singular dimensions by default.
- **B is correct:** By default, `plt.subplots` sets keyword parameter `squeeze=True`. This automatically squeezes out any singular dimensions (dimensions of length 1) from the returned NumPy array. For both a single row of 4 panels `(1, 4)` and a single column of 4 panels `(4, 1)`, the single dimension of length 1 is eliminated, leaving a 1D NumPy array of shape `(4,)`.
- **C is incorrect:** Squeeze behavior is symmetric across rows and columns; `axes2` does not retain `(4, 1)` because its single column dimension is squeezed into `(4,)`.
- **D is incorrect:** Squeeze is applied to row dimensions as well; `axes1` does not retain `(1, 4)` because its single row dimension is squeezed into `(4,)`.

</details>

---

### Question 15

What is the type and shape of `axes` when `plt.subplots()` is invoked with `squeeze=False` on a single subplot:

```python
fig, axes = plt.subplots(1, 1, squeeze=False)
```

- [ ] A) It returns a single `matplotlib.axes.AxesSubplot` scalar object.
- [ ] B) It returns a 1D NumPy array with shape `(1,)`.
- [ ] C) It returns a 2D NumPy array with shape `(1, 1)`.
- [ ] D) It returns a Python dictionary with key `(0, 0)`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) It returns a 2D NumPy array with shape `(1, 1)`.

**Explanation:**

- **A is incorrect:** A bare scalar `AxesSubplot` is only returned when `squeeze=True` (the default for $1 \times 1$).
- **B is incorrect:** A 1D array of shape `(1,)` is not produced; `squeeze=False` guarantees that multi-dimensional row and column axes are preserved.
- **C is correct:** The `squeeze` keyword argument dictates whether singular dimensions are stripped from the output array. Setting `squeeze=False` forces Matplotlib to always return a 2D NumPy array of shape `(nrows, ncols)`, regardless of whether `nrows=1`, `ncols=1`, or both. Here, it returns a 2D array of shape `(1, 1)`.
- **D is incorrect:** `plt.subplots()` never returns a dictionary; dictionary outputs are produced by `plt.subplot_mosaic()`.

</details>

---

### Question 16

A developer configures a horizontal layout with three subplots side by side using `squeeze=False`:

```python
fig, axes = plt.subplots(1, 3, squeeze=False)
```

How must the developer address the middle subplot?

- [ ] A) `axes[1]`
- [ ] B) `axes[1, 0]`
- [ ] C) `axes.middle`
- [ ] D) `axes[0, 1]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `axes[0, 1]`

**Explanation:**

- **A is incorrect:** `axes[1]` would index row index 1, which does not exist because there is only 1 row (raising `IndexError: index 1 is out of bounds for axis 0 with size 1`).
- **B is incorrect:** `axes[1, 0]` attempts to access row 1, column 0, which is out of bounds.
- **C is incorrect:** NumPy arrays have no `.middle` attribute.
- **D is correct:** Because `squeeze=False` is set, `axes` is strictly a 2D array of shape `(1, 3)` (1 row, 3 columns). To access an element, 2D coordinates `[row, col]` must be supplied. The middle plot is in row 0, column 1, which is addressed as `axes[0, 1]`.

</details>

---

### Question 17

Why is setting `squeeze=False` considered an essential best practice when writing generic, reusable plotting utility functions in production data science libraries?

- [ ] A) It improves rendering performance by compiling the figure directly to GPU buffers.
- [ ] B) It guarantees that the returned `axes` object is always a 2D NumPy array, enabling consistent two-index `axes[i, j]` access regardless of whether `nrows=1`, `ncols=1`, or `nrows, ncols > 1`.
- [ ] C) It automatically disables axis spines and tick marks across all panels.
- [ ] D) It prevents memory leaks when generating more than 100 plots in a batch loop.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It guarantees that the returned `axes` object is always a 2D NumPy array, enabling consistent two-index `axes[i, j]` access regardless of whether `nrows=1`, `ncols=1`, or `nrows, ncols > 1`.

**Explanation:**

- **A is incorrect:** `squeeze` is purely an array-shape handling parameter; it does not invoke GPU acceleration.
- **B is correct:** Under default `squeeze=True`, the return type of `axes` mutates dynamically: it is a scalar for $(1, 1)$, a 1D array for $(1, N)$ or $(N, 1)$, and a 2D array for $(M, N)$. This causes generic functions with dynamic row/column inputs to fail with `TypeError` or `IndexError` unless laden with conditional type checks. Setting `squeeze=False` guarantees that `axes` is always a 2D array, so code like `axes[r, c].plot(...)` works unconditionally.
- **C is incorrect:** Spine and tick visibility are unaffected by `squeeze`.
- **D is incorrect:** Memory management in loops is controlled via `plt.close()`, not `squeeze`.

</details>

---

### Question 18

A developer runs `fig, axes = plt.subplots(1, 2)`. They then execute the code:

```python
axes[0].plot([1, 2], [3, 4])
axes[1].plot([1, 2], [10, 20])
```

Why does 1D indexing `axes[0]` and `axes[1]` succeed here without raising an error?

- [ ] A) Because with default `squeeze=True`, a $1 \times 2$ grid squeezes into a 1D NumPy array of length 2, allowing single-index access.
- [ ] B) Because Python automatically flattens all multi-dimensional arrays inside Matplotlib.
- [ ] C) Because `axes` is a standard Python dictionary with integer keys.
- [ ] D) Because `axes[0]` is an alias for `axes[0, 0]`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Because with default `squeeze=True`, a $1 \times 2$ grid squeezes into a 1D NumPy array of length 2, allowing single-index access.

**Explanation:**

- **A is correct:** `plt.subplots(1, 2)` produces a single row of two columns. With `squeeze=True` (default), the singular row dimension is dropped, resulting in a 1D array of shape `(2,)`. Therefore, `axes[0]` accesses the first subplot and `axes[1]` accesses the second subplot cleanly.
- **B is incorrect:** Python does not automatically flatten 2D arrays; grids with multiple rows and columns remain 2D.
- **C is incorrect:** `axes` is a `numpy.ndarray`, not a dictionary.
- **D is incorrect:** 1D arrays do not have two-dimensional indices.

</details>

---

### Question 19

Which of the following statements regarding the return types and shapes of `plt.subplots()` are **correct**?

1. `plt.subplots(1, 1)` returns a single `AxesSubplot` instance, not an array.
2. `plt.subplots(3, 1)` returns a 1D NumPy array of length 3 under default settings.
3. `plt.subplots(2, 2, squeeze=False)` returns an array of shape `(2, 2)`.
4. `plt.subplots(1, 4, squeeze=False)` returns a 1D array of shape `(4,)`.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 3 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because disabling squeeze (`squeeze=False`) preserves the 2D grid shape `(1, 4)` rather than returning a 1D array of shape `(4,)`. Furthermore, Statements 2 and 3 are valid and omitted here.
- **B is incorrect:** While Statements 2 and 3 are correct, Statement 1 is also correct because a $1 \times 1$ grid under default squeeze returns a scalar `AxesSubplot` rather than an array.
- **C is incorrect:** Statement 4 is false because setting `squeeze=False` prevents singular dimension removal, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all correct. A $1 \times 1$ grid returns a scalar `AxesSubplot`; a $3 \times 1$ grid drops the column dimension to return a 1D array of shape `(3,)`; and a $2 \times 2$ grid with `squeeze=False` retains its `(2, 2)` shape.

</details>

---

### Question 20

A data scientist creates a single row of three subplots to compare three clustering models:

```python
fig, (ax_kmeans, ax_dbscan, ax_hierarchical) = plt.subplots(1, 3)
```

What is the exact data type of `ax_kmeans`?

- [ ] A) `<class 'numpy.ndarray'>`
- [ ] B) `<class 'matplotlib.figure.Figure'>`
- [ ] C) `<class 'matplotlib.axes.AxesSubplot'>` (or `matplotlib.axes.Axes`)
- [ ] D) `<class 'list'>`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `<class 'matplotlib.axes.AxesSubplot'>` (or `matplotlib.axes.Axes`)

**Explanation:**

- **A is incorrect:** The 1D NumPy array is unpacked; the individual elements inside are not arrays.
- **B is incorrect:** `Figure` is the parent container assigned to `fig`, not the individual subplots.
- **C is correct:** `plt.subplots(1, 3)` returns a tuple `(fig, axes)` where `axes` is a 1D NumPy array containing 3 individual `AxesSubplot` instances. Unpacking `(ax_kmeans, ax_dbscan, ax_hierarchical)` extracts each individual `AxesSubplot` instance directly into its respective variable.
- **D is incorrect:** Subplots are individual Matplotlib artist instances, not Python lists.

</details>

---
