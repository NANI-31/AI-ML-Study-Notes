# Matplotlib Subplots Objective Test — Set 03

---

### Question 21

A data analyst creates a $3 \times 3$ grid of subplots using `fig, axes = plt.subplots(3, 3)`. Which expression accurately addresses the bottom-right panel in this grid?

- [ ] A) `axes[3, 3]`
- [ ] B) `axes[2, 2]`
- [ ] C) `axes[9]`
- [ ] D) `axes[3][3]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axes[2, 2]`

**Explanation:**

- **A is incorrect:** Python and NumPy use 0-based indexing ($0, 1, 2$). Row index 3 and column index 3 do not exist in a $3 \times 3$ array, raising an `IndexError`.
- **B is correct:** In a $3 \times 3$ NumPy array, indices along both dimensions span from 0 to 2. The bottom row is at index 2, and the rightmost column is at index 2. Therefore, `axes[2, 2]` points to the bottom-right subplot.
- **C is incorrect:** Under default `squeeze=True`, a $3 \times 3$ grid is a 2D array, which cannot be indexed with a single scalar 9; attempting to do so raises `IndexError: index 9 is out of bounds for axis 0 with size 3`.
- **D is incorrect:** Index 3 is out of bounds for axis 0 (the rows).

</details>

---

### Question 22

In multi-panel data exploration loops, developers frequently iterate using `axes.flat`:

```python
fig, axes = plt.subplots(2, 3)
for ax in axes.flat:
  ax.plot(data)
```

What is the internal mechanism and return type of `axes.flat`?

- [ ] A) It is a 1D flat iterator (`numpy.flatiter`) that traverses the multi-dimensional array sequentially in row-major (C-style) order without creating a memory copy.
- [ ] B) It is a Python method that deletes all subplot borders and tick labels.
- [ ] C) It is a Matplotlib function that forces all subplots to have equal square aspect ratios.
- [ ] D) It is a boolean flag indicating whether the subplots are laid out horizontally.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It is a 1D flat iterator (`numpy.flatiter`) that traverses the multi-dimensional array sequentially in row-major (C-style) order without creating a memory copy.

**Explanation:**

- **A is correct:** In NumPy, `.flat` is an attribute that returns a `flatiter` object. It provides an efficient 1D iterator over all elements of the multi-dimensional array in row-major order (top-left to bottom-right). This allows developers to iterate over all $M \times N$ subplots in a single flat loop without needing nested row and column indices or allocating extra memory.
- **B is incorrect:** Modifying borders or ticks is done via `ax.spines` or `ax.tick_params()`, not `.flat`.
- **C is incorrect:** Equal aspect ratios are set via `ax.set_aspect('equal')`.
- **D is incorrect:** `.flat` is not a boolean property; it is an iterator.

</details>

---

### Question 23

An engineer has a list of 6 sensor feature names: `sensors = ['temp', 'press', 'hum', 'volt', 'curr', 'vib']`. They generate a $2 \times 3$ subplot layout and loop through features using `enumerate`:

```python
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
  ax.plot(df[sensors[i]])
  ax.set_title(sensors[i])
```

In what sequence are the sensor charts rendered across the grid panels?

- [ ] A) In column-major order: `temp` at top-left, `press` at bottom-left, `hum` at top-middle, and so on.
- [ ] B) In reverse counter-clockwise order around the perimeter of the figure.
- [ ] C) In random order depending on CPU thread scheduling.
- [ ] D) In row-major order: Row 0 displays `temp`, `press`, `hum` (left to right), and Row 1 displays `volt`, `curr`, `vib` (left to right).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) In row-major order: Row 0 displays `temp`, `press`, `hum` (left to right), and Row 1 displays `volt`, `curr`, `vib` (left to right).

**Explanation:**

- **A is incorrect:** NumPy arrays and `axes.flat` iterate in C-contiguous (row-major) order by default, not Fortran (column-major) order.
- **B is incorrect:** Iteration follows standard Cartesian matrix coordinates, not perimeter paths.
- **C is incorrect:** Iteration over NumPy arrays is strictly deterministic and single-threaded.
- **D is correct:** Standard row-major iteration flattens index `[r, c]` such that the column index advances first across the entire first row before moving to the second row. Index 0 is `(0, 0)`, index 1 is `(0, 1)`, index 2 is `(0, 2)`, index 3 is `(1, 0)`, index 4 is `(1, 1)`, and index 5 is `(1, 2)`.

</details>

---

### Question 24

What is the operational difference between `axes.flat` and `axes.ravel()` when preparing a multi-panel subplot array for iteration or indexing?

- [ ] A) `axes.flat` can be indexed with arbitrary Python slices like `axes.flat[1:4]`, whereas `axes.ravel()` cannot be sliced.
- [ ] B) `axes.ravel()` only works on 3D arrays, while `axes.flat` works on 2D arrays.
- [ ] C) `axes.flat` is a 1D iterator object, whereas `axes.ravel()` returns an actual 1D NumPy array view, which supports standard array operations, boolean masking, and fancy indexing.
- [ ] D) `axes.flat` modifies the original subplots, whereas `axes.ravel()` deletes them.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `axes.flat` is a 1D iterator object, whereas `axes.ravel()` returns an actual 1D NumPy array view, which supports standard array operations, boolean masking, and fancy indexing.

**Explanation:**

- **A is incorrect:** A `flatiter` object supports basic integer and slice indexing, but it is not a full-featured ndarray.
- **B is incorrect:** Both attributes work across any multi-dimensional NumPy array.
- **C is correct:** `axes.flat` returns a `numpy.flatiter` intended specifically for `for` loops. In contrast, `axes.ravel()` returns a true 1D `numpy.ndarray` (a flattened view of the original data). Because `axes.ravel()` is a true array, it supports advanced array operations like fancy indexing (`axes.ravel()[[0, 2, 4]]`), shape querying, and passing directly into functions expecting array-likes (e.g., `fig.colorbar(..., ax=axes.ravel().tolist())`).
- **D is incorrect:** Neither method deletes or alters the underlying Matplotlib axes.

</details>

---

### Question 25

A data pipeline dynamically processes a dataset containing $N$ features, where $N$ varies between runs. The developer fixes the number of columns to $\text{ncols} = 3$. Which formula correctly calculates the required number of rows `nrows` so that all $N$ features have an allocated subplot slot?

- [ ] A) `nrows = math.ceil(N / ncols)` (or integer formula `(N + ncols - 1) // ncols`)
- [ ] B) `nrows = N // ncols`
- [ ] C) `nrows = int(math.floor(N / ncols))`
- [ ] D) `nrows = N % ncols`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `nrows = math.ceil(N / ncols)` (or integer formula `(N + ncols - 1) // ncols`)

**Explanation:**

- **A is correct:** If $N$ is not an exact multiple of $\text{ncols}$ (for instance, $N = 7$ features with $\text{ncols} = 3$), integer division $7 // 3$ yields $2$ rows, which only holds 6 plots, leaving the 7th feature without a subplot. Using ceiling division `math.ceil(7 / 3) = 3` (or the equivalent pure-integer expression `(N + ncols - 1) // ncols = (7 + 3 - 1) // 3 = 9 // 3 = 3`) guarantees sufficient slots for all features.
- **B is incorrect:** Floor division truncates fractions, dropping the final row for incomplete grids (e.g., $7 // 3 = 2$, missing 1 plot).
- **C is incorrect:** Floor division suffers from the exact same truncation shortfall as `//`.
- **D is incorrect:** Modulo arithmetic computes the remainder ($7 \% 3 = 1$), not the total row count.

</details>

---

### Question 26

A developer attempts to address an axis in a $2 \times 2$ grid using the following expression:

```python
fig, axes = plt.subplots(2, 2)
axes[1, 2].plot([1, 2], [3, 4])
```

What happens when this line is executed?

- [ ] A) It plots into the bottom-right panel at row 1, column 1.
- [ ] B) It adds a third column to the figure on the fly.
- [ ] C) It renders the line on the desktop background.
- [ ] D) Python raises `IndexError: index 2 is out of bounds for axis 1 with size 2`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Python raises `IndexError: index 2 is out of bounds for axis 1 with size 2`.

**Explanation:**

- **A is incorrect:** The bottom-right panel is at index `[1, 1]`, not `[1, 2]`.
- **B is incorrect:** NumPy arrays have fixed shape; indexing beyond boundaries does not dynamically allocate new axes.
- **C is incorrect:** Indexing errors halt execution before drawing occurs.
- **D is correct:** For a $2 \times 2$ grid, the valid row indices are $0, 1$ and valid column indices are $0, 1$. Column index `2` is out of bounds for axis 1 (size 2), triggering an `IndexError`.

</details>

---

### Question 27

A researcher wants to customize all subplots located in the second column of a $3 \times 3$ grid (index 1) to have a light yellow background. Which slicing expression correctly extracts this column of axes?

- [ ] A) `axes[1, :]`
- [ ] B) `axes[:, 1]`
- [ ] C) `axes[1:2, 1:2]`
- [ ] D) `axes.col(1)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axes[:, 1]`

**Explanation:**

- **A is incorrect:** `axes[1, :]` selects all columns in row index 1 (the entire middle row), not the middle column.
- **B is correct:** In 2D NumPy array slicing, `axes[row_slice, col_slice]` extracts subarrays. The colon `:` selects all rows, while `1` specifies column index 1. Thus, `axes[:, 1]` extracts a 1D array containing `(axes[0, 1], axes[1, 1], axes[2, 1])`, representing all subplots in the second column.
- **C is incorrect:** `axes[1:2, 1:2]` extracts a 2D subarray containing only a single element `axes[1, 1]`, not the full column.
- **D is incorrect:** NumPy arrays do not have a `.col()` method.

</details>

---

### Question 28

When creating a $3 \times 3$ hyperparameter evaluation grid where each row represents a distinct learning rate $\eta \in [10^{-2}, 10^{-3}, 10^{-4}]$ and each column represents a batch size $B \in [16, 32, 64]$, why are nested loops `for r in range(3): for c in range(3):` often preferred over a flat loop `for ax in axes.flat:`?

- [ ] A) Because `axes.flat` runs four times slower than nested indexing.
- [ ] B) Because `axes.flat` deletes title strings on interior subplots.
- [ ] C) Because nested loops preserve direct access to row and column loop indices (`r` and `c`), making it straightforward to look up corresponding hyperparameter values from separate row and column parameter lists (e.g., `lrs[r]` and `batch_sizes[c]`).
- [ ] D) Because nested loops bypass the Matplotlib artist cache.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Because nested loops preserve direct access to row and column loop indices (`r` and `c`), making it straightforward to look up corresponding hyperparameter values from separate row and column parameter lists (e.g., `lrs[r]` and `batch_sizes[c]`).

**Explanation:**

- **A is incorrect:** The computational difference between flat iteration and nested indexing in a $3 \times 3$ grid is sub-microsecond and negligible.
- **B is incorrect:** `.flat` does not modify or delete subplot titles.
- **C is correct:** In grid experiments (such as Cartesian product grid searches), the horizontal and vertical axes of the figure inherently correspond to two independent experimental factors. Having explicit `r` and `c` loop indices allows direct mapping: `learning_rate = lrs[r]` and `batch_size = batch_sizes[c]`, making code cleaner than unravelling a flat integer index $i$ into $(i // 3, i \% 3)$.
- **D is incorrect:** Artist caching mechanics are identical regardless of how axes are referenced.

</details>

---

### Question 29

Given a grid of subplots with $\text{ncols} = 4$, if an automated loop processes item at 0-based linear index $i = 6$, what are the corresponding 2D matrix coordinates `[row, col]` in the `axes` array?

- [ ] A) `row = 1, col = 2` (`axes[1, 2]`)
- [ ] B) `row = 2, col = 1` (`axes[2, 1]`)
- [ ] C) `row = 1, col = 3` (`axes[1, 3]`)
- [ ] D) `row = 0, col = 6` (`axes[0, 6]`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `row = 1, col = 2` (`axes[1, 2]`)

**Explanation:**

- **A is correct:** Mapping a 1D linear index $i$ into 2D row-major coordinates for a grid with width $\text{ncols}$ uses integer division and modulo:
  $$\text{row} = i // \text{ncols} = 6 // 4 = 1$$
  $$\text{col} = i \% \text{ncols} = 6 \% 4 = 2$$
  Therefore, linear index 6 corresponds exactly to `axes[1, 2]` (row 1, column 2).
- **B is incorrect:** Row 2, column 1 corresponds to linear index $2 \times 4 + 1 = 9$.
- **C is incorrect:** Row 1, column 3 corresponds to linear index $1 \times 4 + 3 = 7$.
- **D is incorrect:** Column index 6 does not exist in a grid with 4 columns.

</details>

---

### Question 30

Which of the following statements regarding subplot grid indexing and traversal in Matplotlib are **correct**?

1. The syntax `axes[0, 1]` and `axes[0][1]` both successfully access the same subplot in a standard 2D NumPy array.
2. `axes.ravel()` returns a flattened copy of the array that, if modified, does not affect the original axes in the figure.
3. Looping over `axes.flat` traverses elements row-by-row (left-to-right, then top-to-bottom).
4. `fig, axes = plt.subplots(2, 2); len(axes)` evaluates to 2 (the number of rows).

- [ ] A) 1 and 2 only
- [ ] B) 2 and 3 only
- [ ] C) 1, 2, and 4 only
- [ ] D) 1, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 3, and 4 only

**Explanation:**

- **A is incorrect:** Statement 2 is false because `axes.ravel()` returns a view of the array containing direct references to the live `AxesSubplot` objects; modifying an axis via `axes.ravel()` mutates the figure canvas. Statements 3 and 4 are also omitted.
- **B is incorrect:** Statement 2 is false, and Statements 1 and 4 are omitted.
- **C is incorrect:** Statement 2 is false, and Statement 3 is omitted.
- **D is correct:** Statements 1, 3, and 4 are all correct. Multi-indexing `axes[0, 1]` and chained indexing `axes[0][1]` access the identical axis; `axes.flat` traverses elements in row-major (C-style) order; and `len(axes)` returns the size of axis 0 (the 2 rows).

</details>

---
