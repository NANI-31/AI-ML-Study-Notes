# Matplotlib Subplots Objective Test — Set 06

---

### Question 51

A machine learning engineer has 5 feature distributions to plot. They instantiate a $2 \times 3$ grid (which creates 6 total subplot slots) and plot the 5 features into the first five panels.

If the 6th subplot is left untouched without any cleanup code, what visual defect appears on the final canvas?

- [ ] A) The 6th panel automatically collapses its width to zero, shifting the first 5 panels to fill the canvas.
- [ ] B) Matplotlib automatically creates a duplicate plot of feature 5 in the 6th slot to fill the void.
- [ ] C) An empty, orphaned rectangular box with default $[0.0, 1.0]$ tick marks and spines appears in the bottom-right corner, making the dashboard look broken and unpolished.
- [ ] D) The figure fails to render and raises an unhandled `DanglingSubplotException`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) An empty, orphaned rectangular box with default $[0.0, 1.0]$ tick marks and spines appears in the bottom-right corner, making the dashboard look broken and unpolished.

**Explanation:**

- **A is incorrect:** Standard grid subplots have fixed geometric footprints; Matplotlib does not collapse unplotted subplots or shift neighboring axes automatically.
- **B is incorrect:** Matplotlib never clones or duplicates prior axes data into empty panels.
- **C is correct:** `plt.subplots(2, 3)` instantiates all 6 `AxesSubplot` objects with default Cartesian properties (x and y limits from $0.0$ to $1.0$, black border spines, and default tick marks). Leaving the 6th panel unaddressed leaves this empty, unpopulated box visible, which detracts from the visual quality of production reports.
- **D is incorrect:** Matplotlib renders empty subplots without throwing any layout or compilation exceptions.

</details>

---

### Question 52

Which simple command turns off the spines, tick marks, and background frame of an unused subplot `ax`, rendering the area as clean, blank whitespace?

- [ ] A) `ax.axis('off')`
- [ ] B) `ax.set_empty(True)`
- [ ] C) `ax.clear_frame()`
- [ ] D) `ax.hide()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.axis('off')`

**Explanation:**

- **A is correct:** `ax.axis('off')` is the standard, high-level Matplotlib convenience method to disable the display of all coordinate axes elements—including x and y axis lines (spines), tick marks, tick labels, and background fills. The axis still technically occupies its grid space, keeping other panels aligned, but it becomes completely invisible whitespace.
- **B is incorrect:** `set_empty()` is not a valid method on an `Axes` object.
- **C is incorrect:** `clear_frame()` is not a Matplotlib method.
- **D is incorrect:** `hide()` does not exist on `Axes`; the correct artist method is `set_visible(False)`.

</details>

---

### Question 53

A developer wants to permanently and completely remove an unused subplot object from the parent figure's internal hierarchy. Which method achieves this?

- [ ] A) `ax.destroy()`
- [ ] B) `ax.delete()`
- [ ] C) `plt.remove_axis(ax)`
- [ ] D) `fig.delaxes(ax)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `fig.delaxes(ax)`

**Explanation:**

- **A is incorrect:** `Axes` instances do not provide a `.destroy()` method.
- **B is incorrect:** `ax.delete()` is not a valid Matplotlib method and will raise an `AttributeError`.
- **C is incorrect:** `plt.remove_axis` is not a function in the `pyplot` module.
- **D is correct:** `fig.delaxes(ax)` is the canonical Figure method to remove an `Axes` instance from the figure. It detaches the axis from the figure's internal artist list (`fig.axes`) and unregisters all of its callbacks, completely eliminating it from subsequent drawing cycles.

</details>

---

### Question 54

What is the fundamental architectural difference between calling `ax.axis('off')` versus `fig.delaxes(ax)` on an unused subplot?

- [ ] A) `ax.axis('off')` clears only the plotted line artists, whereas `fig.delaxes(ax)` exports the axis to an image file before removal.
- [ ] B) `ax.axis('off')` preserves the `Axes` object inside `fig.axes` and maintains its layout footprint, while `fig.delaxes(ax)` permanently purges the `Axes` from `fig.axes` and completely removes it from layout manager tracking.
- [ ] C) `ax.axis('off')` minimizes the axis bounding box to zero pixels, while `fig.delaxes(ax)` merely detaches its title text.
- [ ] D) There is no functional difference; `fig.delaxes(ax)` is simply an internal Python alias for `ax.axis('off')`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.axis('off')` preserves the `Axes` object inside `fig.axes` and maintains its layout footprint, while `fig.delaxes(ax)` permanently purges the `Axes` from `fig.axes` and completely removes it from layout manager tracking.

**Explanation:**

- **A is incorrect:** `ax.axis('off')` toggles axis element visibility rather than deleting data, and `fig.delaxes()` does not export axes to disk.
- **B is correct:** `ax.axis('off')` is purely an artist visibility toggle: the `Axes` object remains registered in `fig.axes` and continues to reserve its allocated cell in the grid geometry. In contrast, `fig.delaxes(ax)` removes the axis from the `Figure.axes` list entirely. This means automated solvers (like `tight_layout` or `constrained_layout`) ignore the deleted axis and will not allocate margins or padding for it.
- **C is incorrect:** `ax.axis('off')` preserves the original bounding box dimensions, and `delaxes` unregisters the full axis rather than just its title.
- **D is incorrect:** The two methods operate at fundamentally different architectural levels: one toggles artist visibility flags, while the other mutates figure structure.

</details>

---

### Question 55

In addition to `ax.axis('off')`, which method on an `Axes` instance hides the entire subplot along with any lines or titles attached to it?

- [ ] A) `ax.set_transparency(1.0)`
- [ ] B) `ax.mask_all()`
- [ ] C) `ax.set_visible(False)`
- [ ] D) `ax.alpha(0)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.set_visible(False)`

**Explanation:**

- **A is incorrect:** `set_transparency()` is not an `Axes` method in Matplotlib.
- **B is incorrect:** `mask_all()` is a fictitious function name.
- **C is correct:** In the Matplotlib artist hierarchy, all visual elements inherit from `matplotlib.artist.Artist`, which exposes the `.set_visible(b: bool)` method. Invoking `ax.set_visible(False)` tells the renderer to skip drawing the axis entirely during render passes, rendering it completely invisible while preserving its object state in memory.
- **D is incorrect:** `alpha` is a property set via `set_alpha()`, and setting `alpha=0` does not disable tick computation or layout bounding evaluations.

</details>

---

### Question 56

A data pipeline plots $N$ features onto a grid with $M$ total subplots ($M > N$). Which code pattern programmatically and cleanly removes all unused subplots at the end of the loop?

- [ ] A)
  ```python
  for j in range(N, len(axes.flat)):
    fig.delaxes(axes.flat[j])
```
- [ ] B)
  ```python
  axes.flat[N:].delete()
```
- [ ] C)
  ```python
  fig.clear_unused(keep=N)
```
- [ ] D)
  ```python
  for ax in axes.flat:
    if ax.is_empty():
      ax.erase()
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
for j in range(N, len(axes.flat)):
  fig.delaxes(axes.flat[j])
```

**Explanation:**

- **A is correct:** Iterating from index $N$ (the first unused slot) up to the total number of allocated subplots `len(axes.flat)` and calling `fig.delaxes(axes.flat[j])` (or `axes.flat[j].axis('off')`) dynamically cleans up all empty panels, regardless of how many excess slots were generated.
- **B is incorrect:** NumPy slices do not have a `.delete()` method.
- **C is incorrect:** `fig.clear_unused()` is not a Matplotlib method.
- **D is incorrect:** `is_empty()` and `erase()` are not standard Matplotlib methods.

</details>

---

### Question 57

A junior programmer attempts to hide an unused subplot `ax_empty` by running:

```python
ax_empty.plot([], [])
ax_empty.set_title("")
```

What is the visual outcome of this attempt?

- [ ] A) The axis area collapses horizontally, causing neighboring subplots to resize and overlap.
- [ ] B) The subplot is successfully hidden and rendered as clean whitespace.
- [ ] C) Matplotlib raises an `EmptyPlotError` during canvas draw time.
- [ ] D) The attempt fails completely: the empty axis box with its black spines and default tick lines ($0.0$ to $1.0$) remains fully visible on the canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The attempt fails completely: the empty axis box with its black spines and default tick lines ($0.0$ to $1.0$) remains fully visible on the canvas.

**Explanation:**

- **A is incorrect:** Plotting empty data has no effect on grid geometry or neighbor subplot bounds.
- **B is incorrect:** Passing empty data to `plot()` creates an empty `Line2D` object but does not hide the axis frame, spines, or ticks.
- **C is incorrect:** Matplotlib accepts empty lists without throwing any runtime or draw errors.
- **D is correct:** Plotting an empty sequence `[]` adds an empty `Line2D` artist to the axis, but does not alter the visibility of the underlying `AxesSubplot` frame, ticks, or spines. The unpopulated axis continues to render as an empty box with visible borders. To hide it, the developer must call `ax_empty.axis('off')` or `fig.delaxes(ax_empty)`.

</details>

---

### Question 58

When 5 subplots are arranged in a $2 \times 3$ grid and the 6th subplot is hidden via `axes[1, 2].axis('off')`, where is the resulting blank space situated?

- [ ] A) In the exact geometric center between Row 0 and Row 1.
- [ ] B) In the bottom-right corner (Row 1, Column 2), with the first 5 subplots remaining in their fixed grid positions.
- [ ] C) The 5 subplots automatically shift and re-center themselves across the bottom row.
- [ ] D) On the left side of Row 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) In the bottom-right corner (Row 1, Column 2), with the first 5 subplots remaining in their fixed grid positions.

**Explanation:**

- **A is incorrect:** Subplots are positioned according to rigid grid cells; removing cell (1, 2) does not shift panels to the geometric center.
- **B is correct:** Standard grids created by `plt.subplots(2, 3)` are static matrices. Subplots 0 to 4 occupy indices $(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)$. Hiding subplot $(1, 2)$ simply leaves the bottom-right cell as empty space. To create custom centered or spanning layouts, one should use `GridSpec` or `subplot_mosaic`.
- **C is incorrect:** Matplotlib's grid layout engines do not automatically re-center remaining subplots across the row when an axis is hidden.
- **D is incorrect:** Subplots 0 to 4 stay in their assigned positions, so the blank space remains at Row 1, Column 2, not on Row 0.

</details>

---

### Question 59

Which of the following statements regarding deleting or hiding subplots in Matplotlib are **correct**?

1. `fig.delaxes(ax)` reduces the length of `fig.axes` by 1.
2. An axis hidden with `ax.set_visible(False)` remains in the `fig.axes` list.
3. `ax.axis('off')` prevents any text titles added via `ax.set_title()` from being displayed.
4. Calling `plt.cla()` on an axis hides its spines and borders.

- [ ] A) 2 and 3 only
- [ ] B) 1 and 4 only
- [ ] C) 1, 2, and 4 only
- [ ] D) 1 and 2 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1 and 2 only

**Explanation:**

- **A is incorrect:** Statement 3 is false because `ax.axis('off')` hides spines and ticks but leaves any explicit title set via `ax.set_title()` rendered. Statement 1 is also omitted.
- **B is incorrect:** Statement 4 is false because `plt.cla()` clears plotted data artists (lines, bars, markers) but leaves the axis spines and ticks intact. Statement 2 is also omitted.
- **C is incorrect:** Statements 3 and 4 are both false.
- **D is correct:** Statements 1 and 2 are accurate. `fig.delaxes(ax)` removes the axis from `fig.axes`, reducing `len(fig.axes)` by 1; and `set_visible(False)` only alters render visibility while keeping the axis registered in `fig.axes`.

</details>

---

### Question 60

A developer creates a figure with 6 subplots:

```python
fig, axes = plt.subplots(2, 3)
print(len(fig.axes))
fig.delaxes(axes[1, 2])
print(len(fig.axes))
```

What two values are printed to the console?

- [ ] A) `6` and `6`
- [ ] B) `2` and `1`
- [ ] C) `6` and `5`
- [ ] D) `6` and `0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `6` and `5`

**Explanation:**

- **A is incorrect:** `fig.delaxes` mutates `fig.axes`, so the length does not stay 6.
- **B is incorrect:** `fig.axes` contains individual `Axes` instances (total 6), not row counts.
- **C is correct:** Initially, `plt.subplots(2, 3)` populates `fig.axes` with all 6 instantiated `AxesSubplot` objects (`len(fig.axes) == 6`). Calling `fig.delaxes(axes[1, 2])` removes that specific subplot from the figure, reducing the length of `fig.axes` to `5`.
- **D is incorrect:** Only the specified axis is removed, not all axes.

</details>

---
