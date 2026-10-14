# Matplotlib Subplots Objective Test — Set 09

---

### Question 81

A data scientist creates a $2 \times 2$ grid of subplots and wants to add a single, centralized main title across the entire figure canvas above all subplots. Which method achieves this?

- [ ] A) `plt.title("Executive Dashboard", global_title=True)`
- [ ] B) `axes.set_master_title("Executive Dashboard")`
- [ ] C) `axes[0, 0].set_title("Executive Dashboard", span_all=True)`
- [ ] D) `fig.suptitle("Executive Dashboard", fontsize=16, y=0.98)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `fig.suptitle("Executive Dashboard", fontsize=16, y=0.98)`

**Explanation:**

- **A is incorrect:** `plt.title()` operates on the current active axis; it does not take a `global_title` parameter.
- **B is incorrect:** `axes` objects do not have a `set_master_title()` method.
- **C is incorrect:** Subplot titles cannot be extended across the entire canvas using a `span_all` parameter.
- **D is correct:** `fig.suptitle(t)` (Super Title) is the dedicated `Figure` method to render a centered headline across the entire figure canvas above all constituent subplots. The `y` argument positions it in normalized figure coordinates (e.g., `y=0.98` places it near the very top edge).

</details>

---

### Question 82

In modern Matplotlib (v3.4+), what dedicated Figure-level methods allow developers to add a single unified horizontal label and vertical label that span across all subplots, eliminating redundant per-axis labels?

- [ ] A) `fig.supxlabel("Training Epochs")` and `fig.supylabel("Loss / Accuracy Metric")`
- [ ] B) `fig.master_xlabel()` and `fig.master_ylabel()`
- [ ] C) `plt.global_xlabel()` and `plt.global_ylabel()`
- [ ] D) `axes.set_common_labels()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig.supxlabel("Training Epochs")` and `fig.supylabel("Loss / Accuracy Metric")`

**Explanation:**

- **A is correct:** Matplotlib 3.4 introduced `fig.supxlabel()` and `fig.supylabel()` (Super X/Y Labels). These methods position centered, figure-wide axis descriptions centered along the bottom and left margins of the entire subplot grid, replacing cumbersome manual `fig.text()` coordinate calculations.
- **B is incorrect:** `master_xlabel` and `master_ylabel` do not exist in Matplotlib.
- **C is incorrect:** `plt.global_xlabel` and `plt.global_ylabel` are fictitious function names.
- **D is incorrect:** `set_common_labels` is not a method on `Axes` or `Figure`.

</details>

---

### Question 83

All subplots in a $2 \times 3$ grid plot the same three models (Model A, Model B, Model C) with identical colors. Instead of repeating identical legends in every panel, the developer wants a single shared legend placed above the grid.

Which code snippet implements this pattern correctly?

- [ ] A)
  ```python
  fig.legend_all(axes)
```
- [ ] B)
  ```python
  axes.combine_legends()
```
- [ ] C)
  ```python
  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc="upper center",
      bbox_to_anchor=(0.5, 1.02),
      ncol=3,
  )
```
- [ ] D)
  ```python
  plt.legend(target=fig)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
)
```

**Explanation:**

- **A is incorrect:** `legend_all()` is not a valid method in Matplotlib.
- **B is incorrect:** `combine_legends()` does not exist in the Matplotlib API.
- **C is correct:** This is the standard, production-grade pattern:
  1. Extract the artist handles and label strings from any representative subplot: `handles, labels = ax.get_legend_handles_labels()`.
  2. Bind them to the figure level: `fig.legend(handles, labels, ...)`.
  3. Position it centrally using `loc="upper center"` and `bbox_to_anchor=(0.5, 1.02)`, with `ncol=3` to arrange the entries in a clean horizontal row.
- **D is incorrect:** Passing `target=fig` inside `ax.legend()` is not supported and raises a `TypeError`.

</details>

---

### Question 84

Why is placing a separate, identical `ax.legend()` inside every individual panel of a 6-panel grid considered poor data visualization engineering?

- [ ] A) It causes Matplotlib to throw a `LegendLimitExceeded` runtime warning.
- [ ] B) It introduces severe visual redundancy, clutters the data plotting areas, wastes valuable screen space, and increases cognitive load on the viewer.
- [ ] C) It silently alters the line styles of duplicate series to dotted lines to differentiate panels.
- [ ] D) It forces the figure rendering backend to re-rasterize all vector paths into 72 DPI bitmaps.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It introduces severe visual redundancy, clutters the data plotting areas, wastes valuable screen space, and increases cognitive load on the viewer.

**Explanation:**

- **A is incorrect:** Matplotlib permits any number of legends without raising runtime warnings.
- **B is correct:** Edward Tufte's data-to-ink ratio principle dictates eliminating redundant non-data ink. When multiple subplots share the same categories and color schemes, repeating the exact same legend box six times consumes plotting area and forces viewers to repeatedly parse identical keys. A single, clean figure-level legend (`fig.legend()`) clarifies the entire dashboard.
- **C is incorrect:** Creating multiple legends does not modify line styles or artist properties.
- **D is incorrect:** The presence of multiple legends has no effect on vector vs. raster backend rendering logic.

</details>

---

### Question 85

A data scientist generates a $2 \times 2$ grid of image heatmaps (`im1`, `im2`, `im3`, `im4`) where all images share a common normalization scale $[0.0, 1.0]$. They want a single, shared colorbar that steals space evenly from all four subplots.

Which code snippet accomplishes this?

- [ ] A) `fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.05, pad=0.04)`
- [ ] B) `axes.add_colorbar(im1)`
- [ ] C) `plt.colorbar(all_axes=True)`
- [ ] D) `fig.colorbar(im1, share_all=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.05, pad=0.04)`

**Explanation:**

- **A is correct:** In `fig.colorbar(mappable, ax=...)`, the `ax` parameter accepts an individual `Axes` or a sequence of `Axes`. Passing a flattened list of all subplots (`axes.ravel().tolist()`) informs Matplotlib to resize and steal margin space proportionally from all subplots in the grid, placing the colorbar neatly alongside the entire group.
- **B is incorrect:** `axes.add_colorbar()` is not a method on `Axes` or ndarrays.
- **C is incorrect:** `plt.colorbar()` does not accept an `all_axes` parameter.
- **D is incorrect:** `share_all` is not a valid keyword parameter in `fig.colorbar()`.

</details>

---

### Question 86

To achieve pixel-perfect placement of a global colorbar without altering or shrinking existing subplot boundaries, an engineer allocates a dedicated colorbar axis (`cax`) via `fig.add_axes([left, bottom, width, height])`:

```python
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax)
```

What do the coordinates `[0.92, 0.15, 0.02, 0.7]` define?

- [ ] A) Pixel coordinates on a 1080p monitor.
- [ ] B) Grid cell indices $(row=0.92, col=0.15)$.
- [ ] C) Data coordinate limits $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$.
- [ ] D) Normalized figure coordinates where the colorbar starts $92\%$ from the left edge, $15\%$ from the bottom edge, with a width of $2\%$ and height of $70\%$ of the figure canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Normalized figure coordinates where the colorbar starts $92\%$ from the left edge, $15\%$ from the bottom edge, with a width of $2\%$ and height of $70\%$ of the figure canvas.

**Explanation:**

- **A is incorrect:** `fig.add_axes()` coordinates are normalized fractions $[0, 1]$, not fixed hardware pixel offsets.
- **B is incorrect:** The 4-element sequence represents bounding box coordinates, not row/column grid indices.
- **C is incorrect:** Figure-level axes placement is independent of data coordinate limits ($x_{\min}, y_{\min}$, etc.).
- **D is correct:** `fig.add_axes([left, bottom, width, height])` positions an axis explicitly in normalized canvas space ($[0, 1]$):
  - `left=0.92`: 92% from the left edge (placing it in the right margin).
  - `bottom=0.15`: 15% from the bottom edge.
  - `width=0.02`: 2% of the canvas width (a slim bar).
  - `height=0.7`: 70% of the canvas height.
  Passing `cax=cax` draws the colorbar directly inside this dedicated box without resizing any existing subplots.

</details>

---

### Question 87

A UI designer wants the entire background canvas of a dashboard figure to be dark navy blue, while each individual subplot plotting area has a crisp white background. Which combination of calls achieves this?

- [ ] A) `plt.style.use('navy'); plt.style.use('white')`
- [ ] B) `fig.patch.set_facecolor('#0a192f')` and `for ax in axes.flat: ax.set_facecolor('white')`
- [ ] C) `axes.set_background('white', parent='navy')`
- [ ] D) `fig.set_colors(canvas='navy', plots='white')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.patch.set_facecolor('#0a192f')` and `for ax in axes.flat: ax.set_facecolor('white')`

**Explanation:**

- **A is incorrect:** `plt.style.use()` applies global styles across the entire application and cannot apply distinct contrasting colors to figure and axes simultaneously.
- **B is correct:** In the Matplotlib object model, the `Figure` background patch is accessed via `fig.patch.set_facecolor(...)`, which paints the entire outer window frame. The individual plotting regions are configured via `ax.set_facecolor(...)`. Combining them creates a stylish card-style dashboard aesthetic.
- **C is incorrect:** `set_background()` with a `parent` argument is not a valid Matplotlib method.
- **D is incorrect:** `set_colors()` is not a method on `Figure`.

</details>

---

### Question 88

In a multi-panel column of subplots, the y-axis labels on different rows have different character lengths (e.g., Row 0 label is `"Loss"`, Row 1 label is `"Validation Mean Squared Error"`). Because tick labels have different widths, the vertical axis labels become horizontally staggered and misaligned.

Which Matplotlib method automatically aligns all y-axis labels in a column into a straight, neat vertical line?

- [ ] A) `plt.straighten_labels()`
- [ ] B) `axes.align_text()`
- [ ] C) `fig.align_ylabels(axes[:, 0])` (or `fig.align_labels()`)
- [ ] D) `fig.justify_y()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `fig.align_ylabels(axes[:, 0])` (or `fig.align_labels()`)

**Explanation:**

- **A is incorrect:** `straighten_labels()` is not a Matplotlib method.
- **B is incorrect:** `align_text()` does not exist on `Axes` arrays.
- **C is correct:** `fig.align_ylabels(axs)` (and its companion `fig.align_xlabels()`, or `fig.align_labels()`) is a specialized alignment tool. It inspects all subplots in the provided sequence and shifts their axis labels to align cleanly along a single common line, eliminating awkward staggering caused by varying tick label widths.
- **D is incorrect:** `justify_y()` is not a recognized Matplotlib function.

</details>

---

### Question 89

Which of the following statements regarding figure-level dashboard annotations in Matplotlib are **correct**?

1. `fig.suptitle()` can accept font formatting parameters such as `fontsize`, `fontweight`, and `color`.
2. `fig.legend()` can be styled with multiple columns using the `ncol` (or `ncols`) parameter.
3. `fig.supxlabel()` was introduced in Matplotlib 3.4.
4. Calling `fig.colorbar()` without `cax` or `ax` automatically deletes all subplots.

- [ ] A) 1 and 2 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** While Statements 1 and 2 are true, Statement 3 is also true (`fig.supxlabel` and `fig.supylabel` were officially added in Matplotlib 3.4).
- **B is incorrect:** Statement 4 is false because `fig.colorbar()` never deletes subplots; if `ax` is omitted, it defaults to the current active axes or raises a `RuntimeError`. Statement 1 is omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all correct. `fig.suptitle` supports font styling properties; `fig.legend` supports multi-column layout via `ncol`; and `fig.supxlabel`/`fig.supylabel` provide native figure-level axis labeling.

</details>

---

### Question 90

When positioning a figure title via `fig.suptitle("Performance Report", y=0.98)`, why is specifying `y=0.98` particularly advantageous in dashboard engineering?

- [ ] A) It positions the text in normalized figure coordinates near the very top edge ($98\%$ from the bottom), ensuring a clean buffer of whitespace above the highest subplot titles to avoid ugly text collisions.
- [ ] B) It sets the opacity of the title to $98\%$.
- [ ] C) It sets the font size to 98 points.
- [ ] D) It rotates the title by 98 degrees.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It positions the text in normalized figure coordinates near the very top edge ($98\%$ from the bottom), ensuring a clean buffer of whitespace above the highest subplot titles to avoid ugly text collisions.

**Explanation:**

- **A is correct:** By default, `fig.suptitle()` places the title at $y=0.98$ (or $y=0.90$ depending on layout engine). Explicitly anchoring `y=0.98` or `y=1.0` in normalized canvas coordinates keeps the headline elevated, preventing it from crashing into the top row of subplot titles when `tight_layout` or custom margins are active.
- **B is incorrect:** Text opacity is governed by the `alpha` keyword argument, not `y`.
- **C is incorrect:** Font size is configured via `fontsize`, not coordinate parameter `y`.
- **D is incorrect:** Text rotation is controlled by `rotation`, not `y`.

</details>

---
