# Matplotlib Ticks & Scales Objective Test — Set 09

---

### Question 81

A deep learning engineer designs a training convergence plot. They want major gridlines to render as prominent solid lines (`linestyle='-'`, `alpha=0.7`) and minor gridlines to render as subtle dotted lines (`linestyle=':'`, `alpha=0.3`).

Which pair of commands establishes this dual-resolution grid?

- [ ] A)
  ```python
  ax.grid(True, which="major", linestyle="-", alpha=0.7)
  ax.grid(True, which="minor", linestyle=":", alpha=0.3)
```
- [ ] B)
  ```python
  ax.set_grids(primary="-", secondary=":")
```
- [ ] C)
  ```python
  ax.grid_major(style="-", opacity=0.7)
  ax.grid_minor(style=":", opacity=0.3)
```
- [ ] D)
  ```python
  ax.grid(levels=[1, 2], styles=["-", ":"])
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
ax.grid(True, which="major", linestyle="-", alpha=0.7)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
```

**Explanation:**

- **A is correct:** In Matplotlib, `ax.grid()` accepts the `which` parameter (`'major'`, `'minor'`, or `'both'`). Calling `ax.grid()` separately for major and minor grids allows developers to bind distinct line styles, widths, and opacities to each tick collection, creating a clean visual hierarchy where primary milestones stand out over subtle intermediate guide marks.
- **B is incorrect:** `ax.set_grids()` is a non-existent method.
- **C is incorrect:** `grid_major()` and `grid_minor()` do not exist in Matplotlib; all grid configurations route through `ax.grid()`.
- **D is incorrect:** `levels` is not a valid parameter in `ax.grid()`.

</details>

---

### Question 82

In a horizontal bar chart displaying feature importance rankings, horizontal gridlines create visual clutter and compete with the horizontal bars. An analyst wants gridlines to appear **only along the numerical x-axis** (vertical guide lines).

Which command configures this single-axis gridline orientation?

- [ ] A) `ax.grid(direction='vertical')`
- [ ] B) `ax.grid(orientation='vertical')`
- [ ] C) `ax.grid_columns()`
- [ ] D) `ax.grid(axis='x', linestyle='--', alpha=0.5)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.grid(axis='x', linestyle='--', alpha=0.5)`

**Explanation:**

- **A is incorrect:** `direction='vertical'` is not a valid parameter for `ax.grid()`; it raises an unexpected keyword argument error.
- **B is incorrect:** `orientation` is not recognized by `ax.grid()`.
- **C is incorrect:** `grid_columns()` does not exist in the Matplotlib API.
- **D is correct:** In `ax.grid()`, the `axis` parameter accepts `'both'` (default), `'x'`, or `'y'`. Specifying `axis='x'` directs Matplotlib to render gridlines corresponding exclusively to the horizontal x-ticks—which form vertical reference lines spanning the plot. In horizontal bar charts, this allows viewers to read numeric bar values without horizontal line interference.

</details>

---

### Question 83

A researcher notices that default grid lines are rendered on top of data points, cutting through scatter markers and obscuring data transparency.

Which command forces grid lines to be drawn **beneath** all plotted data artists (lines, markers, bars)?

- [ ] A) `ax.grid(layer='background')`
- [ ] B) `ax.set_axisbelow(True)` (or `ax.set_axisbelow('line')`)
- [ ] C) `ax.grid(zorder=-1000)`
- [ ] D) `ax.send_grid_to_back()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_axisbelow(True)` (or `ax.set_axisbelow('line')`)

**Explanation:**

- **A is incorrect:** `layer='background'` is not a recognized argument in `ax.grid()`.
- **B is correct:** By default in certain styles or older Matplotlib versions, axis grid lines can render above data patches. Calling `ax.set_axisbelow(True)` establishes that gridlines, ticks, and spine strokes are drawn with a z-order beneath all data artists (`Line2D`, `PathCollection`, `Patch`). Setting `'line'` places gridlines below data lines while keeping them above filled patches.
- **C is incorrect:** Setting `zorder` directly in `grid()` does not reliably coordinate with axis spine and tick drawing order across all backends.
- **D is incorrect:** `send_grid_to_back()` is a fictitious method name.

</details>

---

### Question 84

A complex hyperparameter search chart contains 8 model variants. Placing the legend inside the plot area covers critical data curves. The engineer wants the legend anchored **outside the plot area to the right**, with its top edge aligned with the top spine.

Which command executes this external positioning?

- [ ] A) `ax.legend(loc='outside_right')`
- [ ] B) `ax.legend(dock='right')`
- [ ] C) `ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)`
- [ ] D) `ax.legend(position='margin')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)`

**Explanation:**

- **A is incorrect:** `loc='outside_right'` is not a valid location string in standard Matplotlib legend placement.
- **B is incorrect:** `dock='right'` is a fictitious keyword argument.
- **C is correct:** In Matplotlib, `bbox_to_anchor=(x, y)` specifies the bounding box anchor point in normalized axes coordinates (where $(0, 0)$ is bottom-left and $(1, 1)$ is top-right). Setting `bbox_to_anchor=(1.02, 1.0)` places the anchor $2\%$ outside the right spine, level with the top spine. Pairing this with `loc='upper left'` attaches the top-left corner of the legend box to this external anchor point, positioning the entire legend neatly in the right margin.
- **D is incorrect:** `position='margin'` is not supported by `ax.legend()`.

</details>

---

### Question 85

When configuring an external legend via `ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))`, what coordinate system governs the values `(1.05, 1.0)` by default?

- [ ] A) Normalized axes coordinate space (where $(0, 0)$ is the bottom-left corner of the axes box and $(1, 1)$ is the top-right corner of the axes box).
- [ ] B) Raw pixel dimensions on the computer monitor.
- [ ] C) Normalized figure coordinate space relative to the full window frame.
- [ ] D) The data coordinate units of the plotted series.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Normalized axes coordinate space (where $(0, 0)$ is the bottom-left corner of the axes box and $(1, 1)$ is the top-right corner of the axes box).

**Explanation:**

- **A is correct:** In `ax.legend(bbox_to_anchor=...)`, when a 2-tuple `(x, y)` is supplied without an explicit `bbox_transform`, Matplotlib defaults to using `ax.transAxes` (normalized axes coordinate space). In this space, $x = 0$ is the left spine, $x = 1.0$ is the right spine, $y = 0$ is the bottom spine, and $y = 1.0$ is the top spine. An $x$-value of $1.05$ represents a location $5\%$ beyond the right spine.
- **B is incorrect:** Pixel coordinates would require values on the order of hundreds or thousands of display pixels, governed by display DPI.
- **C is incorrect:** Normalized figure coordinates are governed by `fig.transFigure`, which is not the default transform when calling `ax.legend()`.
- **D is incorrect:** Data coordinate space is governed by `ax.transData`, which is not used by default for legend anchors.

</details>

---

### Question 86

An analyst successfully places a legend outside the right spine using `bbox_to_anchor=(1.05, 1.0)`. However, upon saving the plot with `plt.savefig("chart.png")`, the legend is completely cut off and missing from the output image file.

What single argument passed to `plt.savefig()` resolves this truncation by dynamically recalculating the exported image canvas to encompass the external legend?

- [ ] A) `crop=False`
- [ ] B) `expand_margins=True`
- [ ] C) `keep_legend=True`
- [ ] D) `bbox_inches='tight'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `bbox_inches='tight'`

**Explanation:**

- **A is incorrect:** `crop=False` is not a valid parameter in `plt.savefig()`.
- **B is incorrect:** `expand_margins=True` does not exist in Matplotlib.
- **C is incorrect:** `keep_legend=True` is a fictitious keyword argument.
- **D is correct:** When elements reside outside the axes box (such as external legends or rotated tick labels), standard `savefig()` only exports the pre-allocated figure canvas, slicing off anything protruding into outer margin space. Passing `bbox_inches='tight'` instructs the rendering backend to evaluate the bounding boxes of all artist objects (including legends and titles) and expand the exported canvas boundary so nothing is clipped.

</details>

---

### Question 87

A data scientist designs a wide dashboard panel and wants to position the legend horizontally **beneath the x-axis tick labels** arranged in 4 columns.

Which command accomplishes this bottom-centered multi-column layout?

- [ ] A) `ax.legend(bottom=True, cols=4)`
- [ ] B) `ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)`
- [ ] C) `ax.legend(align='horizontal', columns=4)`
- [ ] D) `ax.legend(footer=True, span=4)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)`

**Explanation:**

- **A is incorrect:** `bottom=True` and `cols=4` are not valid arguments for `ax.legend()`.
- **B is correct:** In normalized axes coordinates, $x = 0.5$ represents the horizontal center of the plot, and $y = -0.15$ represents a position $15\%$ beneath the bottom spine (providing clearance below the x-axis tick labels). Pairing `loc='upper center'` anchors the top-center of the legend box to $(0.5, -0.15)$, and `ncol=4` arranges the legend entries into four side-by-side horizontal columns.
- **C is incorrect:** `align='horizontal'` and `columns=4` do not exist in `ax.legend()`.
- **D is incorrect:** `footer=True` and `span=4` are fabricated parameter names.

</details>

---

### Question 88

According to data visualization expert Edward Tufte and modern graphical design theory, which configuration of grid lines maximizes the data-to-ink ratio and prevents grid lines from distracting from data trends?

- [ ] A) Thick black solid lines with $100\%$ opacity (`color='black', linewidth=2.0, alpha=1.0`).
- [ ] B) Fluorescent yellow dashed lines (`color='yellow', linestyle='--'`).
- [ ] C) Subtle, muted light-gray lines with low opacity and thin stroke (`color='#e0e0e0', linewidth=0.6, alpha=0.5`).
- [ ] D) High-contrast alternating red and green stripes.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Subtle, muted light-gray lines with low opacity and thin stroke (`color='#e0e0e0', linewidth=0.6, alpha=0.5`).

**Explanation:**

- **A is incorrect:** Thick black solid lines create visual clutter ("chartjunk") that dominates the plot and distracts viewers from detecting data trends.
- **B is incorrect:** Fluorescent colors introduce harsh visual vibration and reduce graphical readability.
- **C is correct:** The data-to-ink ratio principle dictates that non-data ink (spines, gridlines, ticks) should be visually subordinate to data ink (markers, trendlines, bars). Using thin (`linewidth=0.6`), low-opacity (`alpha=0.5`), light-gray (`#e0e0e0`) gridlines provides essential coordinate reference without visually overpowering the data.
- **D is incorrect:** High-contrast alternating stripes severely degrade visual aesthetics and violate Edward Tufte's minimal non-data ink guidelines.

</details>

---

### Question 89

In polar coordinate projections (`plt.subplot(projection='polar')`), what do the two sets of gridlines correspond to?

- [ ] A) Angular coordinates ($\theta$ / azimuth rays radiating from the origin) and radial coordinates ($r$ / concentric circles expanding outward).
- [ ] B) Cartesian $x$ and $y$ rectilinear gridlines.
- [ ] C) Hexagonal tessellation cells.
- [ ] D) Logarithmic frequency bands.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Angular coordinates ($	heta$ / azimuth rays radiating from the origin) and radial coordinates ($r$ / concentric circles expanding outward).

**Explanation:**

- **A is correct:** In a polar projection (`projection='polar'`), coordinates are expressed in angle $	heta$ and radius $r$. Matplotlib's polar axes transform the horizontal and vertical tick locators into angular spokes (rays emerging from the origin at degree or radian intervals) and concentric circular rings (representing radial distance thresholds).
- **B is incorrect:** Cartesian $x$ and $y$ rectilinear gridlines are used in standard rectangular Cartesian plots, not polar plots.
- **C is incorrect:** Hexagonal tessellations are used in hexbin plots (`ax.hexbin`), not standard polar grids.
- **D is incorrect:** Logarithmic frequency bands describe spectrograms or Bode plots, not generic polar axes.

</details>

---

### Question 90

Which of the following statements regarding grid lines and legends in Matplotlib are **correct**?

1. `ax.set_axisbelow(True)` ensures that data series lines and markers render above gridlines.
2. `bbox_to_anchor` allows positioning a legend box anywhere in figure or axes space, including completely outside the plotting boundaries.
3. `ax.grid(axis='y')` enables gridlines along the vertical y-ticks, producing horizontal reference lines across the plot.
4. Adding an external legend automatically suppresses all minor tick marks on the axis.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Includes Statement 4, which is false, and omits Statements 2 and 3.
- **B is incorrect:** Includes Statement 4, which is false, and omits Statement 1.
- **C is incorrect:** Includes Statement 4, which is false, and omits Statement 1.
- **D is correct:** Statements 1, 2, and 3 are all correct. Statement 1 is true because `set_axisbelow(True)` forces gridlines beneath data artists. Statement 2 is true because `bbox_to_anchor` provides arbitrary 2D placement of legends inside or outside the axes. Statement 3 is true because gridlines along the y-axis form horizontal reference lines across the canvas. Statement 4 is false because adding a legend is independent of tick locators and never suppresses minor ticks.

</details>

---
