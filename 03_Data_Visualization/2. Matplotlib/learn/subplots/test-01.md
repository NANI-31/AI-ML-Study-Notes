# Matplotlib Subplots Objective Test — Set 01

---

### Question 1

A machine learning practitioner is using the legacy stateful pyplot interface to display four training diagnostic charts in a $2 \times 2$ grid. Which command activates the bottom-left plotting panel?

- [ ] A) `plt.subplot(2, 2, 3)`
- [ ] B) `plt.subplot(2, 2, 2)`
- [ ] C) `plt.subplot(2, 2, 0)`
- [ ] D) `plt.subplot(1, 2, 1)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `plt.subplot(2, 2, 3)`

**Explanation:**

- **A is correct:** In Matplotlib's `plt.subplot(nrows, ncols, index)` interface, subplots are indexed using 1-based numbering in row-major order (reading from left to right, top to bottom). For a $2 \times 2$ grid: index 1 is top-left, index 2 is top-right, index 3 is bottom-left, and index 4 is bottom-right.
- **B is incorrect:** Index 2 addresses the top-right subplot in a $2 \times 2$ grid.
- **C is incorrect:** Matplotlib's `plt.subplot` uses 1-based indexing; index 0 is invalid and raises a `ValueError`.
- **D is incorrect:** `plt.subplot(1, 2, 1)` creates a $1 \times 2$ layout (1 row, 2 columns) and selects the left panel, not a $2 \times 2$ grid.

</details>

---

### Question 2

A software engineer writes the following code to initialize the first panel of a multi-panel visualizer:

```python
import matplotlib.pyplot as plt

plt.subplot(2, 2, 0)
plt.plot([1, 2, 3], [4, 5, 6])
```

What is the execution result of this script?

- [ ] A) Matplotlib silently wraps index 0 to the final subplot position (index 4).
- [ ] B) The plot renders successfully because Python lists and arrays use 0-based indexing.
- [ ] C) Matplotlib raises `ValueError: num must be 1 <= num <= 4, not 0`.
- [ ] D) A single canvas without axes or tick lines is created.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Matplotlib raises `ValueError: num must be 1 <= num <= 4, not 0`.

**Explanation:**

- **A is incorrect:** Matplotlib does not support negative or modular wrap-around indexing in `plt.subplot()`.
- **B is incorrect:** While standard Python data structures use 0-based indexing, Matplotlib's `plt.subplot()` inherits MATLAB-style 1-based indexing convention ($1 \le \text{index} \le \text{nrows} \times \text{ncols}$).
- **C is correct:** In `plt.subplot(nrows, ncols, index)`, the index parameter `num` is strictly 1-based. Supplying `0` violates the constraint $1 \le \text{num} \le \text{nrows} \times \text{ncols}$, causing Matplotlib to immediately raise a `ValueError`.
- **D is incorrect:** Execution halts upon calling `plt.subplot(2, 2, 0)` before any figure canvas is rendered.

</details>

---

### Question 3

In legacy scripts, developers frequently write three-digit integer shorthand arguments such as `plt.subplot(221)` instead of comma-separated integers `plt.subplot(2, 2, 1)`. Under what technical condition is this three-digit shorthand valid in Matplotlib?

- [ ] A) Only if the total number of subplots is an even power of 2.
- [ ] B) Only if `nrows`, `ncols`, and `index` are all single-digit positive integers ($< 10$).
- [ ] C) Only if `figsize` is not explicitly declared.
- [ ] D) Only when running inside an interactive Jupyter Notebook environment.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Only if `nrows`, `ncols`, and `index` are all single-digit positive integers ($< 10$).

**Explanation:**

- **A is incorrect:** The dimension numbers do not need to be powers of two; for example, `plt.subplot(335)` is valid.
- **B is correct:** Matplotlib parses a single integer argument by extracting its decimal digits as hundreds, tens, and units places (`nrows = num // 100`, `ncols = (num // 10) % 10`, `index = num % 10`). This syntax is only mathematically unambiguous when `nrows < 10`, `ncols < 10`, and `index < 10`. For grids with 10 or more rows or columns, comma-separated syntax must be used.
- **C is incorrect:** `figsize` declaration on the parent figure has no bearing on subplot shorthand parsing.
- **D is incorrect:** The integer shorthand is supported across all standard Python environments and backend renderers.

</details>

---

### Question 4

What is the fundamental architectural difference between managing subplots via the stateful `plt.subplot()` function versus the object-oriented `plt.subplots()` function?

- [ ] A) `plt.subplot()` renders vector graphics, whereas `plt.subplots()` renders rasterized pixel bitmaps.
- [ ] B) `plt.subplot()` compiles code into C extensions, whereas `plt.subplots()` executes in pure Python.
- [ ] C) `plt.subplot()` supports 3D projections, whereas `plt.subplots()` only supports 2D Cartesian plots.
- [ ] D) `plt.subplot()` relies on a global state machine tracking the "current active" axis, whereas `plt.subplots()` instantiates and returns explicit `Figure` and `Axes` object references for modular manipulation.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.subplot()` relies on a global state machine tracking the "current active" axis, whereas `plt.subplots()` instantiates and returns explicit `Figure` and `Axes` object references for modular manipulation.

**Explanation:**

- **A is incorrect:** Graphic format (vector vs. raster) is determined by the output backend and file extension (`.png`, `.svg`, `.pdf`), not the subplot creation function.
- **B is incorrect:** Both functions are implemented within the same standard Matplotlib Python codebase.
- **C is incorrect:** Both interfaces fully support 3D projections via `projection='3d'`.
- **D is correct:** `plt.subplot()` follows a stateful procedural paradigm where subsequent pyplot calls (e.g., `plt.title()`, `plt.plot()`) implicitly modify whichever axis was last activated. In contrast, `fig, ax = plt.subplots()` is object-oriented: it returns explicit object references (`Figure` and `Axes`), allowing developers to target specific panels directly (`ax.plot()`, `ax.set_title()`) without relying on global state.

</details>

---

### Question 5

A developer creates a figure canvas and wants to attach an individual subplot axis to it using the object-oriented `Figure` method. Which snippet correctly adds a top-left subplot in a $2 \times 2$ grid?

- [ ] A)
  ```python
  fig = plt.figure()
  ax = fig.add_subplot(2, 2, 1)
```
- [ ] B)
  ```python
  fig = plt.figure()
  ax = fig.create_axis(2, 2, 1)
```
- [ ] C)
  ```python
  fig = plt.figure()
  ax = fig.subplot(2, 2, 1)
```
- [ ] D)
  ```python
  fig = plt.figure()
  ax = fig.attach_panel(pos="top-left")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax = fig.add_subplot(2, 2, 1)`

**Explanation:**

- **A is correct:** In the Matplotlib object model, `Figure.add_subplot(nrows, ncols, index)` is the canonical method for instantiating and adding an `AxesSubplot` instance to an existing `Figure` object.
- **B is incorrect:** `Figure.create_axis()` is not a valid Matplotlib method.
- **C is incorrect:** `Figure.subplot()` does not exist; `subplot()` is a module-level pyplot function, while the Figure method is named `add_subplot()`.
- **D is incorrect:** `attach_panel()` is a fictitious function name not found in the Matplotlib API.

</details>

---

### Question 6

In modern Matplotlib, what occurs if a script invokes `plt.subplot(2, 2, 1)` multiple times throughout an analysis script?

```python
plt.subplot(2, 2, 1)
plt.plot([1, 2], [3, 4])

plt.subplot(2, 2, 2)
plt.plot([1, 2], [10, 20])

plt.subplot(2, 2, 1)  # Second call to position 1
plt.plot([1, 2], [5, 6], color="red")
```

- [ ] A) A duplicate second axis is created directly over the first axis, producing severe rendering artifacts and double axis lines.
- [ ] B) Matplotlib recognizes the matching geometry and returns the existing `Axes` instance at $(2, 2, 1)$, adding the red line to the existing plot.
- [ ] C) Matplotlib raises an `AxisAlreadyExistsError`.
- [ ] D) The existing contents of subplot 1 are erased and replaced with a blank canvas before plotting the red line.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Matplotlib recognizes the matching geometry and returns the existing `Axes` instance at $(2, 2, 1)$, adding the red line to the existing plot.

**Explanation:**

- **A is incorrect:** In modern Matplotlib versions, matching subplot calls do not stack duplicate overlapping axes.
- **B is correct:** When `plt.subplot(nrows, ncols, index)` is called with identical grid geometry arguments to an already existing subplot in the current figure, Matplotlib makes that existing subplot current and returns it, rather than creating a new overlapping axis. Thus, the red line is plotted on top of the original line.
- **C is incorrect:** No exception is thrown when reactivating an existing subplot position.
- **D is incorrect:** The existing contents are preserved; pyplot commands simply append new artists to the active axis.

</details>

---

### Question 7

When invoking `plt.subplots()` without passing any positional or keyword arguments:

```python
fig, ax = plt.subplots()
```

What default grid dimensions and return types are generated?

- [ ] A) A $2 \times 2$ grid returning a `Figure` and a $2 \times 2$ NumPy array of `Axes`.
- [ ] B) An empty `Figure` and `None` for `ax`.
- [ ] C) A $1 \times 1$ grid returning a `Figure` and a single `AxesSubplot` scalar object.
- [ ] D) A $1 \times 1$ grid returning a `Figure` and a 1D NumPy array containing one `AxesSubplot` object.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) A $1 \times 1$ grid returning a `Figure` and a single `AxesSubplot` scalar object.

**Explanation:**

- **A is incorrect:** `plt.subplots()` does not default to $2 \times 2$.
- **B is incorrect:** `ax` is always an initialized plotting area, never `None`.
- **C is correct:** By default, `plt.subplots(nrows=1, ncols=1)` creates a single plot. Because the default parameter is `squeeze=True`, a $1 \times 1$ grid squeezes out all singular array dimensions, returning a single scalar `matplotlib.axes.AxesSubplot` object directly.
- **D is incorrect:** A 1D array is returned only when `nrows=1, ncols > 1`, `nrows > 1, ncols=1`, or when `squeeze=False`. For $1 \times 1$ with default squeeze, a bare scalar axis is returned.

</details>

---

### Question 8

A machine learning engineer specifies `figsize=(12, 8)` when initializing subplots:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
```

What do the numerical values `12` and `8` represent?

- [ ] A) The total number of columns and rows of data points supported by the axes.
- [ ] B) The width and height of each individual subplot in screen pixels ($px$).
- [ ] C) The width and height of each individual subplot in centimeters ($cm$).
- [ ] D) The total width and height of the entire `Figure` canvas in physical inches.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The total width and height of the entire `Figure` canvas in physical inches.

**Explanation:**

- **A is incorrect:** `figsize` governs physical rendering geometry, not dataset coordinate dimensions.
- **B is incorrect:** `figsize` is specified in physical inches, not screen pixels; pixel dimensions are the product of `figsize` and `dpi` ($12 \times 100 = 1200\text{ px}$ at 100 DPI). Furthermore, it applies to the entire figure, not per subplot.
- **C is incorrect:** Matplotlib's default dimensional units for figure size are inches, not centimeters.
- **D is correct:** In Matplotlib, `figsize=(width, height)` defines the total physical dimensions of the top-level `Figure` window in inches. The individual subplots share and subdivide this total canvas area.

</details>

---

### Question 9

A developer writes an object-oriented multi-panel script but inadvertently uses `plt.title()` instead of the `Axes` method:

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([1, 2, 3], [10, 20, 30])
plt.title("Model Convergence")
```

Where does the title string `"Model Convergence"` appear on the final figure?

- [ ] A) It appears atop `ax2` (the second subplot) because `ax2` was the most recently instantiated axis in the pyplot state machine.
- [ ] B) It appears atop `ax1` because `ax1.plot()` was the last plotting method called.
- [ ] C) It is placed centrally across the top of both subplots as a figure-level title.
- [ ] D) Matplotlib raises an `AttributeError` because `plt.title` cannot be called after `plt.subplots`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It appears atop `ax2` (the second subplot) because `ax2` was the most recently instantiated axis in the pyplot state machine.

**Explanation:**

- **A is correct:** Pyplot functions like `plt.title()` operate on the "current active axis" (`plt.gca()`). When `plt.subplots(1, 2)` instantiates subplots sequentially, `ax2` is created last, leaving it as the active axis in the global state machine. Calling `ax1.plot()` (an object-oriented call) does not alter pyplot's internal active axis pointer. Consequently, `plt.title()` erroneously titles `ax2`. In object-oriented programming, one should use `ax1.set_title("Model Convergence")`.
- **B is incorrect:** Calling an OO method directly on `ax1` does not update the global pyplot active axis state.
- **C is incorrect:** A figure-wide centered title requires `fig.suptitle()`, not `plt.title()`.
- **D is incorrect:** Mixing procedural and OO calls is valid syntax in Python, even though it leads to subtle layout bugs.

</details>

---

### Question 10

Which statement accurately describes the hierarchical relationship between a Matplotlib `Figure` and its associated `Axes` objects?

- [ ] A) An `Axes` is the top-level window holding multiple independent `Figure` objects.
- [ ] B) A `Figure` represents the entire top-level canvas frame, which can contain one or more `Axes` objects representing the actual coordinate plotting regions.
- [ ] C) `Figure` and `Axes` are exact synonyms and can be used interchangeably in all Matplotlib methods.
- [ ] D) A `Figure` can contain at most two `Axes` objects: one horizontal x-axis and one vertical y-axis.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `Figure` represents the entire top-level canvas frame, which can contain one or more `Axes` objects representing the actual coordinate plotting regions.

**Explanation:**

- **A is incorrect:** The hierarchy is the reverse: `Figure` is the parent container, and `Axes` are the children.
- **B is correct:** In the Matplotlib object model, the `Figure` is the overarching window or page boundary on which everything is drawn. An `Axes` (commonly called a subplot) is a specific rectangular coordinate region within that figure where data is plotted, containing its own x-axis, y-axis, title, ticks, and graphic artists. A single `Figure` can host any number of `Axes`.
- **C is incorrect:** `Figure` and `Axes` are distinct classes with completely different responsibilities and API methods.
- **D is incorrect:** The individual coordinate lines are `Axis` objects (`ax.xaxis`, `ax.yaxis`), whereas `Axes` represents the entire 2D (or 3D) plotting area.

</details>

---
