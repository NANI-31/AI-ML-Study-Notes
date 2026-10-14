# Matplotlib Subplots Objective Test — Set 04

---

### Question 31

A data scientist creates a $2 \times 2$ grid of subplots to evaluate four machine learning models:

```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
```

What is the primary visual and analytical effect of specifying `sharex=True` and `sharey=True`?

- [ ] A) It duplicates the data plotted on the first subplot across all other three subplots.
- [ ] B) It forces all subplots to display identical plot titles.
- [ ] C) It merges the four subplots into a single large 3D surface plot.
- [ ] D) It couples the coordinate axis limits and scales across all subplots, ensuring that equal physical distances represent equal numerical values across all panels for honest comparison.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It couples the coordinate axis limits and scales across all subplots, ensuring that equal physical distances represent equal numerical values across all panels for honest comparison.

**Explanation:**

- **A is incorrect:** Sharing axes links coordinate scale systems; it never duplicates or copies data points between subplots.
- **B is incorrect:** Subplot titles remain independent and are set individually per axis via `ax.set_title()`.
- **C is incorrect:** 3D surface generation is controlled via `subplot_kw={'projection': '3d'}`, not `sharex`/`sharey`.
- **D is correct:** `sharex=True` and `sharey=True` synchronize the x- and y-axis limits across all panels. If data added to one panel causes its limits to expand, all linked panels immediately expand to match. This guarantees that visual slopes, heights, and spreads can be compared directly across models without misleading differences in axis ranges.

</details>

---

### Question 32

In a $2 \times 2$ subplot grid initialized with `sharex=True` and `sharey=True`, what automatic optimization does Matplotlib perform on interior tick labels?

- [ ] A) It removes tick marks from the bottom-right subplot entirely.
- [ ] B) It automatically suppresses the x-axis tick labels on the top row and y-axis tick labels on the right column to eliminate visual clutter and redundant labeling.
- [ ] C) It converts all numerical tick labels to Roman numerals.
- [ ] D) It rotates all tick labels by 90 degrees.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It automatically suppresses the x-axis tick labels on the top row and y-axis tick labels on the right column to eliminate visual clutter and redundant labeling.

**Explanation:**

- **A is incorrect:** Outer border subplots (bottom row and left column) retain their tick marks and labels.
- **B is correct:** When axes are shared across a full grid, repeating identical tick numbers on interior subplots is redundant and clutters the figure. Matplotlib automatically hides x-tick labels for all subplots not in the bottom row, and hides y-tick labels for all subplots not in the leftmost column.
- **C is incorrect:** Numerical formatting is governed by `Formatter` objects, not axis sharing.
- **D is incorrect:** Label rotation is configured via `plt.xticks(rotation=...)` or `ax.tick_params()`.

</details>

---

### Question 33

A financial dashboard displays stock prices in Column 0 and trading volume in Column 1 across three different companies (Rows 0, 1, and 2). Stock prices range around $\$150$, while volumes range in millions.

Which parameter specification links x-axes down each column while allowing price and volume to maintain their distinct scales?

- [ ] A) `sharex=True, sharey=True`
- [ ] B) `sharex='row'`
- [ ] C) `sharex='col'`
- [ ] D) `sharex='diagonal'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `sharex='col'`

**Explanation:**

- **A is incorrect:** `sharex=True` links the x-axis across all four panels globally, which would force prices and volumes onto a shared horizontal scale.
- **B is incorrect:** `sharex='row'` links horizontal scales across subplots in the same row, which is the exact opposite of what is needed (sharing down columns).
- **C is correct:** In Matplotlib, setting `sharex='col'` enforces that x-axis limits are shared only among subplots residing within the same vertical column. Subplots in Column 0 share their x-axis, and subplots in Column 1 share their x-axis, but Column 0 and Column 1 remain independent of each other.
- **D is incorrect:** `'diagonal'` is not a valid option for `sharex` or `sharey`.

</details>

---

### Question 34

A machine learning monitoring figure consists of two rows:
- Row 0: Training and Validation Loss ($y \in [0.1, 2.5]$)
- Row 1: Training and Validation Accuracy ($y \in [0.5, 1.0]$)

Which parameter ensures that both plots in Row 0 share their vertical loss scale, and both plots in Row 1 share their vertical accuracy scale, without forcing loss and accuracy onto the same axis?

- [ ] A) `sharey='row'`
- [ ] B) `sharey=True`
- [ ] C) `sharey='col'`
- [ ] D) `sharey='both'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `sharey='row'`

**Explanation:**

- **A is correct:** Setting `sharey='row'` links vertical axis limits exclusively across subplots within the same horizontal row. All panels in Row 0 (Loss) share vertical scaling, while all panels in Row 1 (Accuracy) share their separate vertical scaling ($[0, 1]$).
- **B is incorrect:** `sharey=True` forces all subplots globally to share a single vertical scale, expanding accuracy up to $2.5$ and compressing the accuracy curves into an unreadable flat line.
- **C is incorrect:** `sharey='col'` links plots down columns, coupling loss in Row 0 to accuracy in Row 1.
- **D is incorrect:** `'both'` is not a recognized string argument for `sharey`.

</details>

---

### Question 35

When displaying a multi-panel figure in an interactive GUI window (such as the default Qt, Tk, or notebook `%matplotlib widget` backend), what dynamic behavior occurs if `sharex=True` is enabled and a user zooms in on Subplot 1?

- [ ] A) Subplot 1 zooms in, but all other subplots remain static at their original view limits.
- [ ] B) All subplots linked via `sharex=True` automatically and synchronously update their horizontal view limits to match Subplot 1 in real time.
- [ ] C) Matplotlib raises an `InteractiveSyncError` and crashes the GUI event loop.
- [ ] D) The other subplots zoom out by an inverse proportional factor.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) All subplots linked via `sharex=True` automatically and synchronously update their horizontal view limits to match Subplot 1 in real time.

**Explanation:**

- **A is incorrect:** Unlinked subplots remain static, but linked subplots update together.
- **B is correct:** `sharex` binds the internal `XAxis` view limit callbacks (`callbacks.connect('xlim_changed')`). When a user uses the interactive zoom or pan tool on any linked axis, the new coordinate limits are broadcast immediately to all shared axes, panning and zooming them synchronously.
- **C is incorrect:** Interactive synchronization is a core intended feature of Matplotlib and does not crash the GUI.
- **D is incorrect:** Linked axes move identically in the same direction, not inversely.

</details>

---

### Question 36

A developer creates two separate subplots manually using `fig.add_subplot()`. How can the second subplot be linked to share the x-axis of the first subplot?

- [ ] A) `ax2 = fig.add_subplot(2, 1, 2, link='ax1')`
- [ ] B) `ax2 = fig.add_subplot(2, 1, 2, sync_with=ax1)`
- [ ] C) `ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)`
- [ ] D) `ax2 = fig.add_subplot(2, 1, 2); ax2.bind(ax1)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)`

**Explanation:**

- **A is incorrect:** `link` is not a valid keyword parameter in `fig.add_subplot()` and will raise a `TypeError`.
- **B is incorrect:** `sync_with` is not a recognized Matplotlib parameter for axis synchronization.
- **C is correct:** In `fig.add_subplot(nrows, ncols, index, sharex=other_ax)`, passing an existing `Axes` instance to the `sharex` or `sharey` keyword argument explicitly binds the new axis to the coordinate system of the referenced axis.
- **D is incorrect:** `Axes` instances do not have a `.bind()` method for coordinate linking; attempting this raises an `AttributeError`.

</details>

---

### Question 37

In a $2 \times 2$ grid with `sharex=True`, an engineer notices that the top-left subplot (`axes[0, 0]`) has its x-axis tick labels hidden by default. If the engineer explicitly wants the numbers visible on `axes[0, 0]`, which command restores them?

- [ ] A) `axes[0, 0].show_ticks()`
- [ ] B) `axes[0, 0].set_sharex(False)`
- [ ] C) `axes[0, 0].revert_ticks()`
- [ ] D) `axes[0, 0].tick_params(labelbottom=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `axes[0, 0].tick_params(labelbottom=True)`

**Explanation:**

- **A is incorrect:** `show_ticks()` is not a Matplotlib method; tick label display is configured via `tick_params` or text artist properties.
- **B is incorrect:** Matplotlib does not provide a `set_sharex(False)` method to dynamically detach an axis after creation.
- **C is incorrect:** `revert_ticks()` does not exist in the Matplotlib API and raises an `AttributeError`.
- **D is correct:** Matplotlib hides interior tick labels on shared axes by setting their visibility flag to `False`. Calling `axes[0, 0].tick_params(labelbottom=True)` explicitly forces the bottom tick labels to render on that panel while maintaining shared coordinate limits.

</details>

---

### Question 38

Matplotlib provides a specialized helper method called `ax.label_outer()` (or calling it in a loop `for ax in axes.flat: ax.label_outer()`). What is the exact function of this method?

- [ ] A) It inspects each subplot's grid position and automatically hides tick labels and axis labels for any subplots that do not reside on the outer edges (perimeter) of the figure grid.
- [ ] B) It moves all title strings outside the canvas frame into an external text file.
- [ ] C) It draws an outer border rectangle around the entire figure.
- [ ] D) It prints the names of the outermost axes to the terminal.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It inspects each subplot's grid position and automatically hides tick labels and axis labels for any subplots that do not reside on the outer edges (perimeter) of the figure grid.

**Explanation:**

- **A is correct:** `ax.label_outer()` (introduced to streamline layout cleaning) checks whether an axis is on the first column, last column, first row, or last row. For interior panels, it automatically disables inner tick labels and inner axis labels, ensuring only the outer perimeter displays coordinate markings.
- **B is incorrect:** It does not manipulate external text files.
- **C is incorrect:** Drawing bounding boxes around the figure is done via `fig.patch`, not `label_outer()`.
- **D is incorrect:** It is a graphical artist formatting method, not a console logging utility.

</details>

---

### Question 39

A junior analyst believes that setting `sharex=True` forces all subplots to plot the exact same data arrays. Why is this a fundamental misunderstanding of the Matplotlib architecture?

- [ ] A) Because `sharex=True` only affects 3D plots, not 2D plots.
- [ ] B) Because `sharex` is an alias for `sharey`.
- [ ] C) Because `sharex=True` only synchronizes the coordinate mapping transformation and view limits ($x_{\min}$ to $x_{\max}$); each subplot remains completely autonomous regarding which datasets, line colors, markers, and artists are drawn on it.
- [ ] D) Because `sharex=True` deletes all plotted data lines.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Because `sharex=True` only synchronizes the coordinate mapping transformation and view limits ($x_{\min}$ to $x_{\max}$); each subplot remains completely autonomous regarding which datasets, line colors, markers, and artists are drawn on it.

**Explanation:**

- **A is incorrect:** `sharex` works seamlessly on 2D plots.
- **B is incorrect:** `sharex` and `sharey` govern orthogonal axes.
- **C is correct:** In Matplotlib, an `Axes` separates data representation (the artists, lines, and bars added to it) from the coordinate view transform (the `Axis` limits and ticks). `sharex=True` links the horizontal view transformation between panels; it has no control over what datasets are plotted. One panel can plot a scatter plot of Model A while another plots a bar chart of Model B on the shared scale.
- **D is incorrect:** Data lines are preserved.

</details>

---

### Question 40

Which of the following statements regarding shared axes in Matplotlib are **correct**?

1. Setting `sharex=True` allows `plt.xlim(xmin, xmax)` called on one axis to automatically update all linked axes.
2. `sharex` can accept an existing `Axes` instance (e.g., `sharex=ax1`).
3. `sharey='row'` allows each row of subplots to have an independent y-axis range while keeping panels in the same row synchronized.
4. Setting `sharex=True` permanently locks the axis limits so they can never be modified.

- [ ] A) 1 and 4 only
- [ ] B) 1, 2, and 3 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because shared axes are not locked; they remain fully dynamic and automatically update whenever new data is added or limits are modified. Statements 2 and 3 are also omitted.
- **B is correct:** Statements 1, 2, and 3 are all correct. Setting `sharex=True` propagates `xlim` adjustments across all linked panels; `sharex` accepts an existing `Axes` instance in `fig.add_subplot()`; and `sharey='row'` allows each row to track an independent metric range while keeping row subplots synchronized.
- **C is incorrect:** Statement 4 is false, Statement 2 is true, and Statements 1 and 3 are omitted.
- **D is incorrect:** Statement 4 is false, and Statements 1 and 2 are omitted.

</details>

---
