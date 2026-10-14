# Matplotlib Subplots Objective Test — Set 05

---

### Question 41

A developer configures inter-subplot spacing using `plt.subplots_adjust()`:

```python
plt.subplots_adjust(wspace=0.4, hspace=0.6)
```

What do the numerical values `0.4` and `0.6` represent in Matplotlib?

- [ ] A) `wspace=0.4` means horizontal spacing between subplots is $40\%$ of the average subplot width; `hspace=0.6` means vertical spacing is $60\%$ of the average subplot height.
- [ ] B) They represent physical millimeter offsets between subplots.
- [ ] C) They represent screen pixel gaps scaled by screen DPI.
- [ ] D) They represent the aspect ratio scaling factors of the canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `wspace=0.4` means horizontal spacing between subplots is $40\%$ of the average subplot width; `hspace=0.6` means vertical spacing is $60\%$ of the average subplot height.

**Explanation:**

- **A is correct:** In Matplotlib's `subplots_adjust(wspace=..., hspace=...)`, `wspace` (width space) and `hspace` (height space) are normalized quantities expressed as a **fraction of the average axis width and height**, respectively. A `wspace=0.4` creates an empty horizontal gap equal to $40\%$ of the width of a single subplot.
- **B is incorrect:** Spacing parameters in `subplots_adjust` are dimensionless fractions of subplot dimensions, not fixed physical millimeters.
- **C is incorrect:** Spacing values are not raw screen pixel counts and do not scale directly with screen DPI.
- **D is incorrect:** Aspect ratios are managed by `ax.set_aspect()`, not spacing adjustments in `subplots_adjust`.

</details>

---

### Question 42

A machine learning engineer carefully calculates custom subplot margins and spacing using `plt.subplots_adjust()`, but then appends `plt.tight_layout()` right before saving the figure:

```python
plt.subplots_adjust(left=0.15, right=0.85, wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.savefig("metrics.png")
```

What happens to the custom `left`, `right`, `wspace`, and `hspace` values when `plt.tight_layout()` is called?

- [ ] A) `plt.tight_layout()` averages its automated values with the developer's custom values.
- [ ] B) `plt.tight_layout()` raises a `LayoutConflictError` and halts execution.
- [ ] C) The custom values are saved in an external JSON configuration file.
- [ ] D) `plt.tight_layout()` recalculates the entire figure layout and completely overwrites the custom `subplots_adjust()` parameters.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.tight_layout()` recalculates the entire figure layout and completely overwrites the custom `subplots_adjust()` parameters.

**Explanation:**

- **A is incorrect:** Matplotlib does not compute an average blend of manual and automated parameters.
- **B is incorrect:** Calling `tight_layout()` after `subplots_adjust()` does not throw an exception.
- **C is incorrect:** Matplotlib does not export layout states to external JSON files.
- **D is correct:** `plt.tight_layout()` is an automated layout solver that calculates margin and spacing parameters from scratch to prevent label collisions. When executed, it directly recalculates `left`, `right`, `bottom`, `top`, `wspace`, and `hspace` and applies them to the figure, completely obliterating any prior manual settings established by `plt.subplots_adjust()`. To preserve custom margins, do not call `plt.tight_layout()` afterward.

</details>

---

### Question 43

When creating a seamless image mosaic or a 2D joint density matrix (where adjacent heatmaps share borders without any whitespace between them), which spacing setting eliminates all inter-panel gaps?

- [ ] A) `plt.subplots_adjust(gap=None)`
- [ ] B) `plt.subplots_adjust(wspace=0, hspace=0)`
- [ ] C) `plt.tight_layout(padding=0)`
- [ ] D) `fig.set_seamless(True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.subplots_adjust(wspace=0, hspace=0)`

**Explanation:**

- **A is incorrect:** `gap` is not a valid parameter for `subplots_adjust()`.
- **B is correct:** Setting `wspace=0` eliminates all horizontal padding between columns, and `hspace=0` eliminates all vertical padding between rows. This causes adjacent subplot spines to touch directly, creating a continuous, seamless tile mosaic.
- **C is incorrect:** `tight_layout` still preserves boundary space for tick marks and spines; it does not guarantee flush, zero-gap axis seams.
- **D is incorrect:** `set_seamless()` is not a Matplotlib method.

</details>

---

### Question 44

In `plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.85)`, what coordinate system defines the parameters `left`, `right`, `bottom`, and `top`?

- [ ] A) Screen pixels measured from the top-left corner of the monitor.
- [ ] B) Data coordinate space of the first subplot ($x_1, y_1$).
- [ ] C) Normalized figure coordinates spanning from $0.0$ (bottom-left of figure) to $1.0$ (top-right of figure).
- [ ] D) Physical printer points where $1 = \frac{1}{72}\text{ inch}$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Normalized figure coordinates spanning from $0.0$ (bottom-left of figure) to $1.0$ (top-right of figure).

**Explanation:**

- **A is incorrect:** Screen pixels depend on monitor DPI and display scaling; Matplotlib uses resolution-independent normalized units here.
- **B is incorrect:** Margin adjustments define the outer bounding box of the grid on the canvas, independent of data values inside the axes.
- **C is correct:** In Matplotlib, the position of the subplot grid within the `Figure` canvas is specified in normalized figure coordinates: $(0, 0)$ is the extreme bottom-left corner of the canvas, and $(1, 1)$ is the extreme top-right corner. Therefore, `left=0.1` leaves a 10% margin on the left, `right=0.9` leaves a 10% margin on the right, `bottom=0.15` leaves a 15% margin on the bottom, and `top=0.85` leaves a 15% margin on the top.
- **D is incorrect:** Points are used in typography and line widths, not figure-level fraction bounds.

</details>

---

### Question 45

What common data visualization defect does `plt.tight_layout()` primarily solve when rendering a multi-panel figure?

- [ ] A) It fixes NaN values inside data arrays by linear interpolation.
- [ ] B) It converts rasterized scatter markers into vector fonts.
- [ ] C) It compresses the file size of exported PNG images by 50%.
- [ ] D) It automatically measures the bounding boxes of axis tick labels, axis labels, and subplot titles, adjusting subplot padding to prevent labels on one panel from colliding with or overlapping adjacent subplots.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It automatically measures the bounding boxes of axis tick labels, axis labels, and subplot titles, adjusting subplot padding to prevent labels on one panel from colliding with or overlapping adjacent subplots.

**Explanation:**

- **A is incorrect:** `tight_layout()` is a layout geometry manager; it never modifies underlying data arrays.
- **B is incorrect:** Vector/raster rendering is governed by artist properties, not layout solvers.
- **C is incorrect:** File compression is handled by image export encoders (e.g., Pillow/libpng).
- **D is correct:** In standard layouts, long x-axis labels, y-axis labels, or large subplot titles often protrude beyond their axes and collide with neighboring subplots or get clipped by figure edges. `plt.tight_layout()` inspects the rendered bounding boxes of all text elements and dynamically resizes and repositions the subplots to ensure all labels remain legible and collision-free.

</details>

---

### Question 46

In modern Matplotlib (v3.6+), what is the recommended, state-of-the-art layout engine that dynamically avoids label collisions while also seamlessly accommodating shared colorbars and suptitles during figure creation?

- [ ] A) `fig, axes = plt.subplots(..., layout="constrained")`
- [ ] B) `fig, axes = plt.subplots(..., auto_fix=True)`
- [ ] C) `fig, axes = plt.subplots(..., layout="pack")`
- [ ] D) `fig, axes = plt.subplots(..., align="auto")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig, axes = plt.subplots(..., layout="constrained")`

**Explanation:**

- **A is correct:** Matplotlib's modern layout manager is invoked via `layout="constrained"` (or `constrained_layout=True`). Unlike `tight_layout()` (which operates as an afterthought solver on static text), `constrained_layout` uses an active topological constraint solver. It dynamically adjusts subplots as new elements (colorbars, suptitles, legends) are added, maintaining optimal spacing and alignment without requiring post-hoc adjustment calls.
- **B is incorrect:** `auto_fix` is not a valid keyword argument for `plt.subplots()` and raises a `TypeError`.
- **C is incorrect:** `layout="pack"` is not a recognized layout engine in Matplotlib; supported values include `'constrained'`, `'tight'`, or `'compressed'`.
- **D is incorrect:** `align="auto"` is not a valid parameter for `plt.subplots()`.

</details>

---

### Question 47

A developer creates a figure with a main header using `fig.suptitle("Model Evaluation Dashboard")` and then invokes `plt.tight_layout()`. They discover that the main super-title directly collides with and overlaps the top row of subplot titles.

How can `tight_layout()` be instructed to leave room for the figure title?

- [ ] A) `plt.tight_layout(leave_title_alone=True)`
- [ ] B) `plt.tight_layout(rect=[0, 0, 1, 0.95])`
- [ ] C) `fig.suptitle("Model Evaluation Dashboard", zorder=10)`
- [ ] D) `plt.tight_layout(y_offset=10)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.tight_layout(rect=[0, 0, 1, 0.95])`

**Explanation:**

- **A is incorrect:** `leave_title_alone` is a fictitious argument not accepted by `plt.tight_layout()`.
- **B is correct:** `plt.tight_layout(rect=[left, bottom, right, top])` restricts the automated layout solver to fit all subplots entirely inside the specified rectangular bounding box in normalized figure coordinates. By specifying `top=0.95` (`rect=[0, 0, 1, 0.95]`), the top 5% of the figure canvas is reserved exclusively for `fig.suptitle()`, completely preventing overlaps.
- **C is incorrect:** Setting `zorder=10` merely dictates drawing stack order and does not prevent visual collisions between superimposed text strings.
- **D is incorrect:** `y_offset` is not a valid keyword parameter in `plt.tight_layout()`.

</details>

---

### Question 48

What happens if a developer attempts to call `plt.subplots_adjust()` on a figure initialized with `layout="constrained"`?

```python
fig, axes = plt.subplots(2, 2, layout="constrained")
plt.subplots_adjust(wspace=0.3)
```

- [ ] A) Matplotlib silently applies both settings by taking the average between the constrained positions and the manual margins.
- [ ] B) An `AttributeError` is raised immediately because `subplots_adjust` is deleted from the `pyplot` module when `constrained_layout` is enabled.
- [ ] C) Matplotlib raises a warning indicating that `subplots_adjust` is incompatible with `constrained_layout`, and the manual adjustment call disables or resets the constrained layout engine.
- [ ] D) The subplots are automatically cleared of all plotted lines and markers, resetting the axes to empty white canvases.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Matplotlib raises a warning indicating that `subplots_adjust` is incompatible with `constrained_layout`, and the manual adjustment call disables or resets the constrained layout engine.

**Explanation:**

- **A is incorrect:** Matplotlib cannot blend or average the two layout systems because they use conflicting coordinate calculations.
- **B is incorrect:** `subplots_adjust` remains an available function in the module; it is not dynamically deleted from `pyplot`.
- **C is correct:** `constrained_layout` and `subplots_adjust` represent mutually exclusive layout paradigms. `constrained_layout` actively solves geometric constraints, while `subplots_adjust` imposes static manual coordinates. Matplotlib emits a `UserWarning: This figure was using a layout engine that is incompatible with subplots_adjust and it has been reset`, disabling the automated engine.
- **D is incorrect:** Plot artists and data are not cleared when conflicting layout calls are made.

</details>

---

### Question 49

In `plt.tight_layout(pad=2.0)`, what does the argument `pad=2.0` represent?

- [ ] A) An absolute spacing of 2.0 inches between all subplots.
- [ ] B) Padding between the figure edge and the edges of subplots, expressed as a fraction of the font size.
- [ ] C) An expansion factor that doubles the resolution (DPI) of the figure.
- [ ] D) A 2-second rendering timeout for slow GUI backends.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Padding between the figure edge and the edges of subplots, expressed as a fraction of the font size.

**Explanation:**

- **A is incorrect:** `pad` is not measured in inches.
- **B is correct:** In `plt.tight_layout()`, the `pad` parameter defines the perimeter padding between the figure canvas edge and the edges of the subplots, expressed as a multiple/fraction of the font size. Other related parameters include `h_pad` (padding between rows) and `w_pad` (padding between columns).
- **C is incorrect:** Figure DPI is controlled via `dpi=...` in `plt.figure()` or `fig.savefig()`.
- **D is incorrect:** `pad` has nothing to do with time delays.

</details>

---

### Question 50

Which of the following statements regarding spacing and layout management in Matplotlib subplots are **correct**?

1. `wspace=0` and `hspace=0` remove all whitespace between subplots.
2. `plt.tight_layout()` must be called before plotting any lines on the axes, or it will throw an error.
3. In `plt.subplots_adjust()`, `left` must always be strictly less than `right`.
4. `layout="constrained"` is supported natively directly inside `plt.subplots()`.

- [ ] A) 1, 3, and 4 only
- [ ] B) 1 and 2 only
- [ ] C) 2 and 3 only
- [ ] D) 2, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 1, 3, and 4 only

**Explanation:**

- **A is correct:** Statements 1, 3, and 4 are accurate. Setting `wspace=0` and `hspace=0` removes all internal whitespace; violating `left < right` produces an inverted, invalid layout; and `layout="constrained"` is the modern standard keyword syntax.
- **B is incorrect:** Statement 2 is false because `plt.tight_layout()` does not raise an error when called before or after plotting commands; it simply adjusts boundaries based on elements present at invocation time. Also, Statements 3 and 4 are omitted.
- **C is incorrect:** Statement 2 is false, and Statements 1 and 4 are omitted.
- **D is incorrect:** Statement 2 is false, and Statement 1 is omitted.

</details>

---
