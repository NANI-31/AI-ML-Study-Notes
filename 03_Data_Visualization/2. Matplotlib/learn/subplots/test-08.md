# Matplotlib Subplots Objective Test — Set 08

---

### Question 71

What modern Matplotlib layout feature (introduced in version 3.4) allows developers to define multi-panel figures using visual ASCII-art string diagrams or nested string lists, returning an intuitive dictionary of named `Axes`?

- [ ] A) `plt.grid_template()`
- [ ] B) `plt.ascii_subplots()`
- [ ] C) `plt.subplot_mosaic()`
- [ ] D) `plt.layout_builder()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `plt.subplot_mosaic()`

**Explanation:**

- **A is incorrect:** `grid_template()` is not a Matplotlib function.
- **B is incorrect:** `ascii_subplots()` does not exist in the Matplotlib API.
- **C is correct:** `plt.subplot_mosaic()` (and `Figure.subplot_mosaic()`) is a high-level layout utility that allows developers to declare complex, multi-cell spanning subplot layouts visually using lists of labels or multi-line string diagrams. Instead of manipulating numerical grid coordinates, developers assign semantic string identifiers to panels.
- **D is incorrect:** `layout_builder()` is not a valid Matplotlib function.

</details>

---

### Question 72

A machine learning engineer defines a layout using `plt.subplot_mosaic()`:

```python
fig, axd = plt.subplot_mosaic([["roc", "pr"], ["loss", "conf_matrix"]])
```

What is the type of `axd`, and how does the engineer access the Receiver Operating Characteristic (`"roc"`) subplot?

- [ ] A) `axd` is a 2D NumPy array; accessed via `axd[0, 0]`.
- [ ] B) `axd` is a Python dictionary; accessed via `axd['roc']`.
- [ ] C) `axd` is a Python list of tuples; accessed via `axd[0]['roc']`.
- [ ] D) `axd` is a custom class; accessed via `axd.get_axis(0)`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axd` is a Python dictionary; accessed via `axd['roc']`.

**Explanation:**

- **A is incorrect:** Unlike `plt.subplots()`, which returns an ndarray of axes, `subplot_mosaic()` returns a dictionary mapping labels to axes.
- **B is correct:** `plt.subplot_mosaic()` returns a 2-tuple `(Figure, dict)`. The dictionary keys are the unique string identifiers supplied in the mosaic specification, and the dictionary values are the corresponding `AxesSubplot` instances. To plot the ROC curve, the developer simply indexes `axd['roc'].plot(...)`.
- **C is incorrect:** The returned axes container is a standard Python `dict`, not a list of tuples.
- **D is incorrect:** The container is a dictionary rather than a custom accessor class with `.get_axis()`.

</details>

---

### Question 73

Examine the following multi-line string mosaic specification:

```python
layout = """
    AAB
    CCD
"""
fig, axd = plt.subplot_mosaic(layout)
```

Which statement accurately describes the resulting figure geometry?

- [ ] A) Panel `A` spans the first 2 columns of Row 0, with Panel `B` in the top-right corner; Panel `C` spans the first 2 columns of Row 1, with Panel `D` in the bottom-right corner.
- [ ] B) The layout has 4 equal square subplots arranged in a single row.
- [ ] C) Panel `A` occupies the entire figure, while `B`, `C`, and `D` are hidden in the background.
- [ ] D) Matplotlib raises a `SyntaxError` because strings cannot be used to specify grids.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Panel `A` spans the first 2 columns of Row 0, with Panel `B` in the top-right corner; Panel `C` spans the first 2 columns of Row 1, with Panel `D` in the bottom-right corner.

**Explanation:**

- **A is correct:** `subplot_mosaic` parses the multi-line string as a $2 \times 3$ grid:
  - Row 0 consists of `['A', 'A', 'B']`. Because `'A'` appears in columns 0 and 1, Matplotlib creates a single wide panel spanning columns 0 and 1, and places panel `'B'` in column 2.
  - Row 1 consists of `['C', 'C', 'D']`. Panel `'C'` spans columns 0 and 1, and panel `'D'` occupies column 2.
- **B is incorrect:** Panels `A` and `C` are twice as wide as panels `B` and `D`.
- **C is incorrect:** All four panels are actively rendered with appropriate spans.
- **D is incorrect:** Multi-line string syntax is fully supported and recommended in `subplot_mosaic()`.

</details>

---

### Question 74

How can an engineer designate an **empty (unpopulated) cell** in a `subplot_mosaic` string or list specification without having to manually call `delaxes()` or `axis('off')` afterward?

- [ ] A) By inserting an exclamation mark `'!'`.
- [ ] B) By using the string `'empty'`.
- [ ] C) By inserting the integer `0`.
- [ ] D) By using a period `'.'` (or `None` in list specifications).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) By using a period `'.'` (or `None` in list specifications).

**Explanation:**

- **A is incorrect:** An exclamation mark `'!'` is treated as an active string identifier creating an axis labeled `'!'`.
- **B is incorrect:** The string `'empty'` creates an `AxesSubplot` keyed to `'empty'` rather than leaving whitespace.
- **C is incorrect:** Passing integer `0` is treated as an axis identifier or causes an unpack error in string parsing.
- **D is correct:** In `subplot_mosaic()`, a period character `'.'` (in string or list specifications) or Python `None` (in list specifications) is the dedicated sentinel for an empty cell. Matplotlib leaves that cell as completely unallocated whitespace, skipping axis creation entirely.

</details>

---

### Question 75

Which mosaic specification cleanly creates a classic joint-marginal distribution layout with:
- A top marginal histogram (`'hist_x'`) above a main 2D scatter plot (`'scatter'`)
- A right marginal histogram (`'hist_y'`) to the right of the scatter plot
- An empty top-right corner?

- [ ] A) `[['hist_x', 'hist_y'], ['scatter', '.']]`
- [ ] B) `[['hist_x', '.'], ['scatter', 'hist_y']]`
- [ ] C) `[['scatter', 'hist_x'], ['hist_y', '.']]`
- [ ] D) `[['.', 'scatter'], ['hist_x', 'hist_y']]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `[['hist_x', '.'], ['scatter', 'hist_y']]`

**Explanation:**

- **A is incorrect:** This places `'scatter'` on the bottom-left and `'hist_y'` on the top-right, misaligning the y-marginal histogram.
- **B is correct:** In a standard Cartesian coordinate setup:
  - Row 0 contains `['hist_x', '.']`: the x-marginal histogram is directly over column 0, and column 1 is empty whitespace `'.'`.
  - Row 1 contains `['scatter', 'hist_y']`: the main scatter plot occupies column 0, and the y-marginal histogram occupies column 1 adjacent to it.
- **C is incorrect:** This swaps the scatter plot and marginal positions, creating an unintuitive layout.
- **D is incorrect:** Placing the blank cell on the bottom-left leaves the main data area empty.

</details>

---

### Question 76

How can custom relative dimensions (e.g., making the main plot 4 times larger than the marginal histograms) be integrated into a `plt.subplot_mosaic()` call?

- [ ] A) `plt.subplot_mosaic(..., scale_factor=4)`
- [ ] B) `plt.subplot_mosaic(..., size_matrix=[[4, 1], [1, 4]])`
- [ ] C) `plt.subplot_mosaic(..., gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})`
- [ ] D) `plt.subplot_mosaic(..., cell_weights=(4, 1))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `plt.subplot_mosaic(..., gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]})`

**Explanation:**

- **A is incorrect:** `scale_factor` is not a recognized keyword argument for `plt.subplot_mosaic()`.
- **B is incorrect:** `size_matrix` is not a valid parameter in Matplotlib.
- **C is correct:** `plt.subplot_mosaic()` internally delegates grid construction to `GridSpec`. Passing `gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4]}` allows full customization of column widths and row heights, scaling the panels seamlessly.
- **D is incorrect:** `cell_weights` is not a valid Matplotlib argument.

</details>

---

### Question 77

A dashboard layout uses a nested list specification:

```python
layout = [["left", "top_right"], ["left", "bottom_right"]]
fig, axd = plt.subplot_mosaic(layout)
```

What is the geometry of the subplot labeled `"left"`?

- [ ] A) It is a tall panel spanning both rows (Rows 0 and 1) in Column 0.
- [ ] B) It is a wide panel spanning both columns in Row 0.
- [ ] C) Two separate identical subplots are created with the same name.
- [ ] D) The panel collapses into a single pixel.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It is a tall panel spanning both rows (Rows 0 and 1) in Column 0.

**Explanation:**

- **A is correct:** Because `"left"` occupies index `[0, 0]` and index `[1, 0]`, `subplot_mosaic` merges the two cells vertically into a single continuous tall axis spanning both rows in the left column.
- **B is incorrect:** Spanning both columns would require `"left"` to occupy `[0, 0]` and `[0, 1]`.
- **C is incorrect:** Matplotlib never instantiates duplicate axes for the same label; identical adjacent labels merge into a single spanned axis.
- **D is incorrect:** The panel fills both rows completely.

</details>

---

### Question 78

A developer attempts to create an L-shaped subplot using the following mosaic string:

```python
layout = """
    A.
    AA
"""
fig, axd = plt.subplot_mosaic(layout)
```

Why does Matplotlib reject this layout and raise a `ValueError`?

- [ ] A) Because letter `'A'` is reserved for administrative logs.
- [ ] B) Because the period `'.'` cannot be placed at the end of a line.
- [ ] C) Because mosaic strings must have at least 3 rows.
- [ ] D) Because every named subplot in Matplotlib must form a contiguous **rectangular bounding box**; non-rectangular geometries (such as L-shapes, T-shapes, or disconnected blocks) are physically impossible in Cartesian grid topology.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Because every named subplot in Matplotlib must form a contiguous **rectangular bounding box**; non-rectangular geometries (such as L-shapes, T-shapes, or disconnected blocks) are physically impossible in Cartesian grid topology.

**Explanation:**

- **A is incorrect:** Identifiers in mosaic specifications can be arbitrary single or multi-character strings.
- **B is incorrect:** Period sentinel characters can appear at any row or column position.
- **C is incorrect:** Mosaic layouts can have any arbitrary number of rows $\ge 1$.
- **D is correct:** In Matplotlib's underlying `SubplotSpec` architecture, every axis is defined by a rectangular range of rows and columns `[r_start:r_stop, c_start:c_stop]`. An L-shaped region cannot be defined by a single rectangular slice. Attempting to group cells in a non-rectangular shape causes `subplot_mosaic()` to raise `ValueError: There are non-rectangular or disjoint areas in the layout: 'A'`.

</details>

---

### Question 79

Can `sharex=True` and `sharey=True` be applied directly when calling `plt.subplot_mosaic()`?

- [ ] A) No; `subplot_mosaic` does not support shared axes under any circumstances.
- [ ] B) Yes; passing `sharex=True` and/or `sharey=True` to `plt.subplot_mosaic()` synchronizes coordinate limits across all generated mosaic panels exactly as it does in `plt.subplots()`.
- [ ] C) Only if the mosaic contains fewer than 3 subplots.
- [ ] D) Only if all subplots have single-letter names.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes; passing `sharex=True` and/or `sharey=True` to `plt.subplot_mosaic()` synchronizes coordinate limits across all generated mosaic panels exactly as it does in `plt.subplots()`.

**Explanation:**

- **A is incorrect:** `subplot_mosaic()` fully supports `sharex` and `sharey` keyword arguments.
- **B is correct:** `plt.subplot_mosaic()` accepts the standard `sharex` and `sharey` keyword arguments (including `True`, `False`, `'row'`, and `'col'`). It transparently links the axes according to the specified topology, synchronizing limits and suppressing interior tick labels.
- **C is incorrect:** Shared axes functionality is independent of the number of panels in the mosaic.
- **D is incorrect:** Identifier length has no bearing on coordinate sharing capabilities.

</details>

---

### Question 80

Which of the following statements regarding `plt.subplot_mosaic()` are **correct**?

1. It returns a tuple `(fig, ax_dict)` where `ax_dict` is a dictionary of named `AxesSubplot` instances.
2. Cells marked with `'.'` are left as unallocated whitespace without creating an `Axes` object.
3. Subplot names can be descriptive multi-character strings (e.g., `'training_loss'`, `'confusion_matrix'`).
4. `subplot_mosaic` can accept nested list-of-lists structures to build multi-level layouts.

- [ ] A) 1 and 2 only
- [ ] B) 2 and 3 only
- [ ] C) 1, 2, 3, and 4
- [ ] D) 1, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 1, 2, 3, and 4

**Explanation:**

- **A is incorrect:** While Statements 1 and 2 are correct, Statements 3 and 4 are also valid features of `subplot_mosaic`.
- **B is incorrect:** Statements 2 and 3 are correct, but Statements 1 and 4 are also true and omitted here.
- **C is correct:** All four statements are correct features of `subplot_mosaic`. It returns a `(Figure, dict)` tuple; periods define unallocated empty cells; dictionary keys correspond to semantic string names; and nested lists can be used to construct hierarchical sub-layouts.
- **D is incorrect:** Statements 1, 3, and 4 are correct, but Statement 2 is also true and omitted here.

</details>

---
