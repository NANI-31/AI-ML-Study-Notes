# Matplotlib Time Series & Temporal Visualization — Set 09

---

### Question 81

An economist wants to highlight a 6-month macroeconomic recession period ($2025	ext{-}04	ext{-}01$ to $2025	ext{-}10	ext{-}01$) by shading the entire vertical column behind the data series in light gray.

Which Matplotlib method accomplishes this full-height vertical span shading?

- [ ] A) `ax.axvspan(pd.to_datetime("2025-04-01"), pd.to_datetime("2025-10-01"), color='gray', alpha=0.2, label='Recession')`
- [ ] B) `ax.shade_x("2025-04-01", "2025-10-01")`
- [ ] C) `ax.vertical_highlight(4, 10)`
- [ ] D) `ax.axhspan(pd.to_datetime("2025-04-01"), pd.to_datetime("2025-10-01"))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.axvspan(pd.to_datetime("2025-04-01"), pd.to_datetime("2025-10-01"), color='gray', alpha=0.2, label='Recession')`

**Explanation:**

- **A is correct:** `Axes.axvspan(xmin, xmax, ymin=0, ymax=1, ...)` draws a vertical span (rectangle) across the plotting area. The x-coordinates are evaluated in data space (accepting datetime objects directly on date axes), while the vertical span defaults to normalized axes coordinates ($y=0$ to $y=1$), guaranteeing that the shading spans the full height of the axes regardless of how the vertical data scale or limits change.
- **B is incorrect:** `shade_x()` is a fictitious method.
- **C is incorrect:** `vertical_highlight()` does not exist in the Matplotlib API.
- **D is incorrect:** `axhspan()` shades horizontal bands along the y-axis, not vertical spans along the x-axis.

</details>

---

### Question 82

To mark a critical server deployment event that occurred at precisely `2026-03-15 14:00:00`, an engineer wants a single, thin vertical red dashed line drawn across the figure.

Which command renders this event marker?

- [ ] A) `ax.vline(x=pd.Timestamp("2026-03-15 14:00:00"))`
- [ ] B) `ax.draw_event("2026-03-15 14:00:00")`
- [ ] C) `ax.axhline(y=pd.Timestamp("2026-03-15 14:00:00"))`
- [ ] D) `ax.axvline(x=pd.Timestamp("2026-03-15 14:00:00"), color='red', linestyle='--', linewidth=1.5, label='Deployment')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.axvline(x=pd.Timestamp("2026-03-15 14:00:00"), color='red', linestyle='--', linewidth=1.5, label='Deployment')`

**Explanation:**

- **A is incorrect:** `Axes.vline()` does not exist; Matplotlib provides `vlines()` which requires explicit `ymin` and `ymax` data coordinates rather than spanning the full axes height automatically.
- **B is incorrect:** `draw_event` is not a valid Matplotlib method.
- **C is incorrect:** `axhline()` draws horizontal lines at specified y-coordinates, not vertical time markers.
- **D is correct:** `Axes.axvline(x=coord, ...)` draws a vertical line spanning the entire vertical height of the plotting area at the specified x-axis data coordinate. Passing a timestamp object places the line precisely at the deployment moment.

</details>

---

### Question 83

A financial chart requires an explanatory callout pointing to a sudden flash crash:
- Arrow tip points to timestamp `crash_date` at price coordinate `$150.0`
- Text box is placed offset at timestamp `crash_date + pd.Timedelta(days=15)` at price coordinate `$220.0`

Which method call implements this connected callout with an arrow?

- [ ] A) `ax.text(crash_date, 150, "Flash Crash", arrow=True)`
- [ ] B) `ax.callout("Flash Crash", at=(crash_date, 150))`
- [ ] C)
  ```python
  ax.annotate(
      "Flash Crash",
      xy=(crash_date, 150),
      xytext=(crash_date + pd.Timedelta(days=15), 220),
      arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
  )
```
- [ ] D) `ax.point_to(crash_date, 150, text="Flash Crash")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
ax.annotate(
    "Flash Crash",
    xy=(crash_date, 150),
    xytext=(crash_date + pd.Timedelta(days=15), 220),
    arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
)
```

**Explanation:**

- **A is incorrect:** `ax.text()` does not accept an `arrow=True` argument for callouts.
- **B is incorrect:** `callout()` is not a method on `Axes`.
- **C is correct:** `Axes.annotate(text, xy, xytext, arrowprops=...)` is Matplotlib's dedicated annotation engine. The `xy` tuple sets the target data coordinate where the arrow points (accepting a datetime object for $x$), while `xytext` sets the coordinate location for the descriptive text box. Supplying `arrowprops` automatically renders the connecting arrow.
- **D is incorrect:** `point_to()` is a fabricated method name.

</details>

---

### Question 84

In `ax.annotate()`, what parameter controls the coordinate interpretation of the target point `xy=(timestamp, y)`?

- [ ] A) `target_system='cartesian'`
- [ ] B) `xycoords='data'` (the default, indicating that coordinates match the axis data transforms)
- [ ] C) `units='datetime'`
- [ ] D) `coord_mode='canvas'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `xycoords='data'` (the default, indicating that coordinates match the axis data transforms)

**Explanation:**

- **A is incorrect:** `target_system` is not an argument in `ax.annotate()`.
- **B is correct:** `xycoords` dictates the coordinate system for `xy`. The default is `'data'`, which uses the active axis data transform. Because the horizontal axis uses date coordinates, passing `(timestamp, y_value)` seamlessly positions the arrow tip at the precise calendar moment and vertical data value.
- **C is incorrect:** `units='datetime'` is not a recognized parameter in `annotate()`.
- **D is incorrect:** `coord_mode` is a fictitious argument.

</details>

---

### Question 85

An industrial pipeline monitors factory equipment. Maintenance occurs across multiple planned intervals:
- Period 1: Jan 10 to Jan 12
- Period 2: Feb 15 to Feb 17
- Period 3: Mar 20 to Mar 22

How can an engineer efficiently shade all three maintenance windows without creating duplicate duplicate entries in the legend?

- [ ] A) Iterate through the tuples of start and end dates, calling `ax.axvspan()` on each; assign `label='Maintenance'` only to the first call (or set `label='_nolegend_'` on subsequent calls) so the legend displays a single unified entry.
- [ ] B) Call `ax.axvspan()` once with a 3D matrix of dates.
- [ ] C) Matplotlib automatically merges identical legend labels by default without user intervention.
- [ ] D) Maintenance windows cannot be plotted inside loops.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Iterate through the tuples of start and end dates, calling `ax.axvspan()` on each; assign `label='Maintenance'` only to the first call (or set `label='_nolegend_'` on subsequent calls) so the legend displays a single unified entry.

**Explanation:**

- **A is correct:** In Matplotlib, every artist with a non-empty label that does not start with an underscore is added to the legend list. If `ax.axvspan(..., label='Maintenance')` is called three times in a loop, the legend will contain three identical entries. By attaching `label='Maintenance'` only during the first iteration (or prefixing with `_` thereafter), a single clean entry appears in the legend.
- **B is incorrect:** `axvspan` expects scalar coordinate bounds, not multidimensional arrays.
- **C is incorrect:** Matplotlib does not automatically deduplicate legend entries; duplicate labels create duplicate legend items.
- **D is incorrect:** Loops are standard and widely used for multi-span shading.

</details>

---

### Question 86

What is the fundamental difference between placing text using `ax.text(timestamp, y, "Event")` versus `ax.annotate("Event", xy=(timestamp, y))`?

- [ ] A) `ax.text()` converts timestamps to roman numerals.
- [ ] B) `ax.annotate()` cannot accept datetime coordinates.
- [ ] C) `ax.text()` only works on the terminal console.
- [ ] D) `ax.text()` places a static text box anchored directly at `(x, y)` without arrows or relative offset transforms; `ax.annotate()` is specifically architected for callouts, allowing the text box to be positioned in offset points or normalized coordinates away from the data point, connected via an arrow.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.text()` places a static text box anchored directly at `(x, y)` without arrows or relative offset transforms; `ax.annotate()` is specifically architected for callouts, allowing the text box to be positioned in offset points or normalized coordinates away from the data point, connected via an arrow.

**Explanation:**

- **A is incorrect:** `ax.text()` does not convert timestamps to roman numerals.
- **B is incorrect:** `ax.annotate()` fully supports datetime coordinates in `xy`.
- **C is incorrect:** `ax.text()` renders graphical text on the canvas, not on the terminal console.
- **D is correct:** While `ax.text(x, y, s)` is useful for simple stationary text placements, `ax.annotate()` provides sophisticated offset positioning: `xy` identifies the exact data point of interest, while `xytext` and `textcoords='offset points'` allow placing the label 20 points above and 30 points to the right, connected by customizable arrow styles (`arrowprops`).

</details>

---

### Question 87

When calling `ax.axvline(x=event_date)` on a chart where data spans 2026-01-01 to 2026-03-31, but `event_date` is 2026-06-01, what happens to the horizontal axis limits?

- [ ] A) The axis limits remain locked to Q1 because vertical lines never affect limits.
- [ ] B) The line is rejected with an out-of-bounds error.
- [ ] C) By default, adding artists that extend beyond the current data limits causes autoscaling to expand `ax.get_xlim()` to include June 2026, compressing the Q1 data to the left side of the plot unless autoscaling is locked or `set_xlim()` is reapplied.
- [ ] D) The figure canvas is rotated 90 degrees.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) By default, adding artists that extend beyond the current data limits causes autoscaling to expand `ax.get_xlim()` to include June 2026, compressing the Q1 data to the left side of the plot unless autoscaling is locked or `set_xlim()` is reapplied.

**Explanation:**

- **A is incorrect:** `axvline` participates in data limit autoscaling along the x-axis.
- **B is incorrect:** Out-of-bounds lines are not rejected; limits expand to encompass them.
- **C is correct:** In Matplotlib's default autoscale mode (`ax.autoscale(True)`), newly added artists update the bounding data limits. If an event line is drawn at a future date outside the initial series range, the horizontal view limits expand outward to encompass that date. To prevent this, developers call `ax.set_xlim(start_date, end_date)` *after* drawing event lines or set `ax.autoscale(False)`.
- **D is incorrect:** View scaling never rotates the canvas.

</details>

---

### Question 88

How does a developer include an `axvspan` shaded region in the plot legend with an accurate translucent color swatch?

- [ ] A) `axvspan` cannot have a legend swatch.
- [ ] B) Provide the `label` parameter inside `ax.axvspan(..., label='Recession')` and call `ax.legend()`.
- [ ] C) Create an artificial scatter point at the origin.
- [ ] D) Use `plt.text_legend('Recession')`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Provide the `label` parameter inside `ax.axvspan(..., label='Recession')` and call `ax.legend()`.

**Explanation:**

- **A is incorrect:** `Polygon` artists returned by `axvspan` are fully supported by legend handlers.
- **B is correct:** `ax.axvspan()` creates a `Polygon` patch artist. Assigning it a `label` (e.g. `label='Recession'`) informs Matplotlib's legend builder to extract its face color and alpha transparency, creating a matching translucent rectangular swatch in the legend box.
- **C is incorrect:** Dummy scatter points are unnecessary workarounds.
- **D is incorrect:** `text_legend` does not exist in Matplotlib.

</details>

---

### Question 89

An automated monitoring dashboard detects transient latency spikes. An engineer wants to plot the continuous latency time series as a blue line, and overlay **prominent red scatter dots** specifically at outlier timestamps where latency exceeded $500	ext{ms}$.

Which code pattern implements this outlier overlay?

- [ ] A)
  ```python
  ax.plot(df.index, df["latency"], color="blue", label="Latency")
  outliers = df[df["latency"] > 500]
  ax.scatter(
      outliers.index,
      outliers["latency"],
      color="red",
      s=40,
      zorder=3,
      label="Anomaly (>500ms)",
  )
```
- [ ] B) `ax.plot(df.index, df['latency'], filter='>500', marker='red')`
- [ ] C) `ax.highlight_points(threshold=500)`
- [ ] D) `ax.scatter_if(df['latency'] > 500)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
ax.plot(df.index, df["latency"], color="blue", label="Latency")
outliers = df[df["latency"] > 500]
ax.scatter(
    outliers.index,
    outliers["latency"],
    color="red",
    s=40,
    zorder=3,
    label="Anomaly (>500ms)",
)
```

**Explanation:**

- **A is correct:** In production time-series diagnostics, overlaying filtered subsets via `ax.scatter()` is standard practice. Filtering `outliers = df[df['latency'] > 500]` isolates anomaly rows, and passing `outliers.index` and `outliers['latency']` to `ax.scatter()` plots distinct marker dots (`s=40`, `color="red"`, `zorder=3`) directly on top of the continuous blue trajectory.
- **B is incorrect:** `ax.plot()` does not take a `filter` argument.
- **C is incorrect:** `highlight_points()` is a non-existent method in Matplotlib.
- **D is incorrect:** `scatter_if()` does not exist in the Matplotlib API.

</details>

---

### Question 90

Which of the following statements regarding temporal annotations and event highlights in Matplotlib are **correct**?

1. `ax.axvspan()` draws vertical rectangular bands spanning the entire height of the axis regardless of y-limits.
2. `ax.axvline()` draws a vertical line at a specified datetime coordinate.
3. `ax.annotate()` supports connected arrow callouts with custom styling via `arrowprops`.
4. Adding an `axvspan` deletes all grid lines across the subplot.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is completely false because `axvspan` does not delete or alter grid lines; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all accurate features of Matplotlib's event annotation toolkit.

</details>

---
