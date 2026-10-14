# Matplotlib Time Series & Temporal Visualization — Set 04

---

### Question 31

A data engineer visualizes 180 days of daily energy consumption. Because date strings like `"2026-03-15"` are wide, adjacent labels collide horizontally along the bottom axis into an illegible line.

Which built-in Figure-level method automatically rotates date tick labels, right-aligns their text anchors, and expands the figure's bottom margin to prevent label truncation?

- [ ] A) `plt.straighten_dates()`
- [ ] B) `fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')`
- [ ] C) `ax.rotate_dates(degrees=45)`
- [ ] D) `fig.align_date_ticks(mode='auto')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')`

**Explanation:**

- **A is incorrect:** `straighten_dates()` does not exist in `pyplot` or Matplotlib.
- **B is correct:** `Figure.autofmt_xdate()` is Matplotlib's dedicated convenience method for date visualization. It iterates through all subplots in the figure, rotates the x-axis tick labels (defaulting to 30 degrees), sets their horizontal alignment to `'right'`, and increases the figure canvas bottom margin (defaulting to `bottom=0.2` or 20%) so slanted date labels do not get clipped by the window boundary.
- **C is incorrect:** `rotate_dates()` is not a method on `Axes`.
- **D is incorrect:** `align_date_ticks()` is a non-existent method.

</details>

---

### Question 32

When date labels are rotated clockwise by 45 degrees, why is setting `ha='right'` (horizontal alignment `'right'`) mechanically essential?

- [ ] A) Because Matplotlib's layout engine crashes if rotated text uses center alignment.
- [ ] B) Because `ha='right'` converts the font rendering engine to SVG vector mode.
- [ ] C) Because `ha='right'` moves the labels to the right vertical spine.
- [ ] D) Because standard labels default to `ha='center'`; rotating text about its center causes the top of the text box to drift to the right, causing the label to appear visually positioned under the next tick mark to the right, whereas `ha='right'` anchors the top-right corner of the label directly beneath its true tick mark.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Because standard labels default to `ha='center'`; rotating text about its center causes the top of the text box to drift to the right, causing the label to appear visually positioned under the next tick mark to the right, whereas `ha='right'` anchors the top-right corner of the label directly beneath its true tick mark.

**Explanation:**

- **A is incorrect:** Matplotlib does not crash on centered rotated text; it simply produces misleading visual alignments.
- **B is incorrect:** Alignment does not affect vector vs. raster rendering modes.
- **C is incorrect:** Horizontal alignment anchors the text relative to individual tick mark coordinates, not the right vertical spine.
- **D is correct:** When text is rotated 45 degrees with `ha='center'`, the pivot point is at the midpoint of the text box. The top-left of the text swings downward to the left, while the top-right of the text swings upward to the right, causing the text to visually align with the adjacent tick mark to the right. Specifying `ha='right'` places the anchor point at the end of the text string, placing the top-right corner precisely at the tick mark endpoint.

</details>

---

### Question 33

What is the operational difference between executing `fig.autofmt_xdate(rotation=45)` versus calling `ax.tick_params(axis='x', rotation=45)` on an individual subplot?

- [ ] A) `fig.autofmt_xdate()` operates at the figure level, automatically setting `ha='right'`, expanding the bottom figure margin to prevent clipping, and handling shared subplots, whereas `ax.tick_params()` only rotates the label text on that single axis without adjusting margins or horizontal alignment.
- [ ] B) `ax.tick_params()` only works on numerical data, while `fig.autofmt_xdate()` works only on strings.
- [ ] C) `fig.autofmt_xdate()` deletes minor ticks from the canvas.
- [ ] D) There is no difference; `fig.autofmt_xdate()` is an alias for `ax.tick_params()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig.autofmt_xdate()` operates at the figure level, automatically setting `ha='right'`, expanding the bottom figure margin to prevent clipping, and handling shared subplots, whereas `ax.tick_params()` only rotates the label text on that single axis without adjusting margins or horizontal alignment.

**Explanation:**

- **A is correct:** `fig.autofmt_xdate()` is an automated multi-step layout manager: it iterates through all subplots, sets label rotation, enforces `ha='right'`, and adjusts the bottom bounding margin of the figure. In contrast, `ax.tick_params(axis='x', rotation=45)` merely alters the rotation angle property of text artists on that single axis; the user must manually set `ha='right'` and call `tight_layout()` to prevent clipping and misaligned anchors.
- **B is incorrect:** `tick_params` works on any axis type, including datetime axes.
- **C is incorrect:** `autofmt_xdate` does not delete minor ticks.
- **D is incorrect:** They are distinct methods with different scopes and behaviors.

</details>

---

### Question 34

A financial chart requires clean visual orientation:
- **Major grid lines** aligned with month starts (solid lines, `alpha=0.6`)
- **Minor grid lines** aligned with weekly ticks (dotted lines, `alpha=0.25`)

Which code snippet establishes this dual-resolution grid binding?

- [ ] A) `ax.set_grid_hierarchy(major="solid", minor="dotted")`
- [ ] B) `ax.grid(major=True, minor=True)`
- [ ] C)
  ```python
  ax.grid(True, which="major", linestyle="-", alpha=0.6)
  ax.grid(True, which="minor", linestyle=":", alpha=0.25)
```
- [ ] D) `ax.date_grid(levels=[1, 2], styles=["-", ":"])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
ax.grid(True, which="major", linestyle="-", alpha=0.6)
ax.grid(True, which="minor", linestyle=":", alpha=0.25)
```

**Explanation:**

- **A is incorrect:** `set_grid_hierarchy()` is a fictitious method.
- **B is incorrect:** `ax.grid()` does not accept `major=True` or `minor=True`; it uses the `which` parameter.
- **C is correct:** In Matplotlib, grid lines bind directly to tick mark locations determined by the axis locators. Calling `ax.grid()` separately for `which="major"` (using solid lines) and `which="minor"` (using dotted lines with lower opacity) creates a clean, professional dual-resolution grid where monthly milestones stand out over subtle weekly guides.
- **D is incorrect:** `date_grid()` does not exist in Matplotlib.

</details>

---

### Question 35

When creating a multi-panel time-series dashboard with vertically stacked subplots sharing the same date axis (`fig, axes = plt.subplots(3, 1, sharex=True)`), how does `fig.autofmt_xdate()` treat the interior subplots?

- [ ] A) It rotates date labels on every subplot, causing them to collide with titles of lower subplots.
- [ ] B) Because `sharex=True` suppresses tick labels on interior subplots, `autofmt_xdate()` selectively formats and rotates only the labels on the bottom-most subplot, keeping interior panels clean.
- [ ] C) It unshares the horizontal axes, giving each panel independent date limits.
- [ ] D) It crashes because it cannot process arrays of `Axes`.

<details>
<summary><b>View Answer & Explanation</b></summary>
**Correct Answer:** B) Because `sharex=True` suppresses tick labels on interior subplots, `autofmt_xdate()` selectively formats and rotates only the labels on the bottom-most subplot, keeping interior panels clean.

**Explanation:**

- **A is incorrect:** `autofmt_xdate()` is designed to check label visibility and avoid rendering redundant interior labels.
- **B is correct:** `fig.autofmt_xdate()` inspects all axes in the figure. When `sharex=True` is enabled, Matplotlib automatically hides the tick label artists on upper and interior subplots. `autofmt_xdate()` honors this configuration, rotating only the visible labels on the bottom subplot while maintaining synchronized date panning across all panels.
- **C is incorrect:** It never breaks or modifies axis sharing bindings.
- **D is incorrect:** `autofmt_xdate()` natively iterates over figure axes arrays without error.

</details>

---

### Question 36

An analyst executes `fig.autofmt_xdate(rotation=45)` and saves the figure using `plt.savefig("report.png")`. Upon inspection, the bottom half of the year strings are cut off outside the exported image boundary.

What command or parameter guarantees that the saved image canvas dynamically expands to include the full bounding box of all rotated date labels?

- [ ] A) `plt.savefig("report.png", clip=False)`
- [ ] B) `plt.savefig("report.png", expand_canvas=True)`
- [ ] C) `plt.savefig("report.png", pad_bottom=50)`
- [ ] D) `plt.savefig("report.png", bbox_inches='tight')` (or calling `plt.tight_layout()` before saving)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.savefig("report.png", bbox_inches='tight')` (or calling `plt.tight_layout()` before saving)

**Explanation:**

- **A is incorrect:** `clip=False` is not a valid parameter for `plt.savefig()`.
- **B is incorrect:** `expand_canvas` is a fictitious argument.
- **C is incorrect:** `pad_bottom` does not exist in `plt.savefig()`.
- **D is correct:** By default, `savefig()` exports the figure within its nominal canvas boundaries. If slanted labels protrude beyond the pre-allocated bottom margin, they are clipped. Supplying `bbox_inches='tight'` instructs the rendering backend to calculate the true outer bounding box of all rendered text artists (including rotated labels and titles) and expand the exported image dimensions so that zero characters are clipped.

</details>

---

### Question 37

In a compact time-series widget, slanted date labels are pressed too close to the horizontal axis line, causing letters to collide visually with the spine.

Which parameter in `ax.tick_params()` increases the distance between the tick marks and the date label text boxes?

- [ ] A) `ax.tick_params(axis='x', pad=8)`
- [ ] B) `ax.tick_params(axis='x', text_margin=8)`
- [ ] C) `ax.tick_params(axis='x', spacing=8)`
- [ ] D) `ax.tick_params(axis='x', buffer=8)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.tick_params(axis='x', pad=8)`

**Explanation:**

- **A is correct:** In `ax.tick_params()`, the `pad` parameter dictates the spacing (in points) between the end of the tick mark and the bounding box of the tick label. Increasing `pad` (e.g., `pad=8` or `pad=10`) pushes the date labels downward away from the axis line, eliminating visual collisions.
- **B is incorrect:** `text_margin` is not a recognized parameter in `tick_params()`.
- **C is incorrect:** `spacing` is not a valid argument for `ax.tick_params()`.
- **D is incorrect:** `buffer` is not supported by `tick_params()`.

</details>

---

### Question 38

In a minimalist executive summary dashboard, a designer wants to hide the physical tick notch lines along the date axis while keeping the rotated date labels clearly visible along the baseline.

Which command achieves this clean styling?

- [ ] A) `ax.tick_params(axis='x', labels_only=True)`
- [ ] B) `ax.set_ticks(show_lines=False)`
- [ ] C) `ax.tick_params(axis='x', bottom=False)` (or `ax.tick_params(axis='x', length=0)`)
- [ ] D) `ax.xaxis.remove_notches()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.tick_params(axis='x', bottom=False)` (or `ax.tick_params(axis='x', length=0)`)

**Explanation:**

- **A is incorrect:** `labels_only` is not a recognized parameter in `tick_params()`.
- **B is incorrect:** `set_ticks()` does not take `show_lines=False`.
- **C is correct:** Setting `bottom=False` toggles off the drawing of tick mark stroke lines along the bottom spine, while leaving `labelbottom=True` active so date text continues to render. Alternatively, setting `length=0` reduces the stroke length of the tick notches to zero. Both approaches produce a modern, notch-free aesthetic.
- **D is incorrect:** `remove_notches()` does not exist in the Matplotlib API.

</details>

---

### Question 39

When plotting a multi-year time series, an engineer notices that even with `rotation=45`, date labels overlap because ticks are placed every 3 days.

What is the proper architectural remedy to eliminate overlap at the source rather than attempting to shrink the font to an unreadable size?

- [ ] A) Delete 80% of the raw data rows from the Pandas DataFrame.
- [ ] B) Adjust the tick locator to a coarser time scale—such as `mdates.MonthLocator(interval=2)` or `mdates.AutoDateLocator()`—so that fewer tick marks are generated along the axis.
- [ ] C) Convert the figure to a 3D surface plot.
- [ ] D) Set `ax.set_xlim()` to span only 24 hours.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Adjust the tick locator to a coarser time scale—such as `mdates.MonthLocator(interval=2)` or `mdates.AutoDateLocator()`—so that fewer tick marks are generated along the axis.

**Explanation:**

- **A is incorrect:** Data resolution should never be destroyed simply to fix axis label density. Plotted data curves can remain high-resolution while axis ticks are spaced widely.
- **B is correct:** In Matplotlib's architecture, tick frequency is governed by the `Locator`. When labels overlap on long time horizons, the solution is to replace dense locators (e.g. daily ticks) with macro-level locators such as `MonthLocator(interval=2)` or `YearLocator()`, or let `AutoDateLocator` automatically choose an appropriate step size.
- **C is incorrect:** 3D surface projection is irrelevant to 2D time-series tick spacing.
- **D is incorrect:** Restricting view limits to 24 hours destroys the multi-year perspective.

</details>

---

### Question 40

Which of the following statements regarding time-series aesthetic formatting in Matplotlib are **correct**?

1. `fig.autofmt_xdate()` automatically sets horizontal alignment `ha='right'` for rotated date labels.
2. `ax.grid(axis='x')` creates vertical reference gridlines corresponding to the date ticks.
3. Setting `pad=10` in `ax.tick_params()` pushes labels further away from the axis line.
4. Calling `fig.autofmt_xdate()` converts all datetime objects in memory into Unix timestamps.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is completely false because `autofmt_xdate()` only modifies visual text artist properties and has zero effect on in-memory data types; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all accurate. `autofmt_xdate()` handles right-alignment; `grid(axis='x')` produces vertical time gridlines; and `pad` controls text offset distance from tick marks.

</details>

---
