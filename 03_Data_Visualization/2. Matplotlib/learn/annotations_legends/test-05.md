# Matplotlib Annotations & Legends — Test 05: Reference Lines

---

### Question 41

A data scientist adds a random-chance baseline to an ROC curve plot:

```python
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.2, label="Chance Level")
```

What is the primary visual effect of `ax.axhline(y=0.5, ...)`?

- [ ] A) A horizontal line at y=0.5 that spans only the data range — from the leftmost to the rightmost plotted data point
- [ ] B) A horizontal line at y=0.5 that spans the full visible width of the axes, from the left edge to the right edge of the plot area
- [ ] C) A horizontal bar chart bar at height y=0.5 across all x-tick positions
- [ ] D) A reference point (marker) at the center of the axes at coordinates (x_mid, 0.5)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A horizontal line at y=0.5 that spans the full visible width of the axes, from the left edge to the right edge of the plot area

**Explanation:**

- **A is incorrect:** `ax.axhline` spans the **entire axes width** — from x=0 to x=1 in axes-fraction coordinates — regardless of the data's x-range. It is not limited to the data extent. For a line that spans only the data extent, `ax.hlines(y=0.5, xmin=x_start, xmax=x_end)` would be used.
- **B is correct:** `ax.axhline` draws a horizontal line that extends across the complete visible x-axis from edge to edge of the axes box. The line is "infinite" in the sense that it is drawn in axes-fraction x-coordinates [0, 1], so it always touches both the left and right walls of the axes no matter how the x-axis limits change. This makes it ideal for permanent reference lines like chance level baselines.
- **C is incorrect:** `ax.axhline` is not a bar chart function. It draws a continuous horizontal line, not discrete bars at tick positions.
- **D is incorrect:** `ax.axhline` draws a complete line, not a single point marker. The line crosses the entire width of the plot area.

</details>

---

### Question 42

A researcher adds both a horizontal and vertical reference line to mark the operating point on a precision-recall curve:

```python
ax.axhline(y=precision_threshold, color="blue", linestyle=":", alpha=0.7)
ax.axvline(x=recall_threshold, color="red", linestyle=":", alpha=0.7)
```

At what point do these two lines intersect?

- [ ] A) At the center of the axes box (50% width, 50% height)
- [ ] B) At the data coordinates `(recall_threshold, precision_threshold)` — the operating point
- [ ] C) At the top-right corner of the axes
- [ ] D) The lines do not intersect because horizontal and vertical lines are plotted on separate layers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) At the data coordinates `(recall_threshold, precision_threshold)` — the operating point

**Explanation:**

- **A is incorrect:** The intersection point depends on the values `precision_threshold` and `recall_threshold`, not on the axes geometry. The center of the axes would only coincide with the intersection if both thresholds happened to be at the midpoints of their respective axis ranges.
- **B is correct:** `ax.axhline(y=precision_threshold)` is a horizontal line at the y-value `precision_threshold`, spanning the full x-range. `ax.axvline(x=recall_threshold)` is a vertical line at the x-value `recall_threshold`, spanning the full y-range. These two lines intersect at precisely the data coordinates `(recall_threshold, precision_threshold)` — visually marking the operating point on the precision-recall curve. This crosshair pattern is a common technique for highlighting a specific operating threshold.
- **C is incorrect:** The intersection would be at the top-right corner only if `recall_threshold = x_max` and `precision_threshold = y_max` — a special edge case, not the general rule.
- **D is incorrect:** Both lines are rendered on the same axes layer. Horizontal and vertical lines can and do intersect when drawn on the same axes.

</details>

---

### Question 43

A machine learning engineer wants to add a horizontal reference line from epoch 20 to epoch 60 only (not spanning the full axes width). Which function is correct?

- [ ] A) `ax.axhline(y=0.3, xmin=20, xmax=60)`
- [ ] B) `ax.hlines(y=0.3, xmin=20, xmax=60, colors="gray", linestyles="--")`
- [ ] C) `ax.axhline(y=0.3, xmin=0.33, xmax=1.0)` where 0.33 and 1.0 are axes fractions
- [ ] D) `ax.plot([20, 60], [0.3, 0.3], color="gray", linestyle="--")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.hlines(y=0.3, xmin=20, xmax=60, colors="gray", linestyles="--")`

**Explanation:**

- **A is incorrect:** `ax.axhline`'s `xmin` and `xmax` parameters accept axes-fraction values in [0, 1], not data coordinates. Passing `xmin=20, xmax=60` would be interpreted as 2000% and 6000% of the axes width — way outside the visible range.
- **B is correct:** `ax.hlines(y, xmin, xmax)` is the function designed for horizontal lines with explicit data-coordinate x-extents. `xmin=20, xmax=60` are interpreted as data-space x-values, drawing a horizontal line at y=0.3 from x=20 to x=60. This is the key distinction between `axhline` (spans axes width in fraction space) and `hlines` (spans a specified data range).
- **C is incorrect:** If `xmin=0.33` corresponds to epoch 20 and `xmax=1.0` corresponds to epoch 60 in a hypothetical [0, 60] range, this calculation would be data-range-dependent and fragile. More importantly, option B is the correct idiomatic approach.
- **D is incorrect while technically valid:** `ax.plot([20, 60], [0.3, 0.3])` does draw a horizontal segment from epoch 20 to 60 at y=0.3, and this would work. However, `ax.hlines` is the purpose-built function for horizontal lines with data-coordinate extents and should be preferred for clarity and when drawing multiple horizontal lines simultaneously.

</details>

---

### Question 44

A data scientist adds a vertical threshold line and includes it in the legend:

```python
ax.plot(x, y_pred, label="Predictions", color="steelblue")
vl = ax.axvline(x=50, color="orange", linestyle="--", linewidth=2,
                label="Deployment Cutoff")
ax.legend()
```

Does `ax.axvline` support the `label` argument for legend inclusion?

- [ ] A) No — `axvline` always requires a separate `ax.plot` call to add the line to the legend
- [ ] B) Yes — `axvline` supports `label` and the resulting line artist appears in the legend alongside the plotted data series
- [ ] C) No — `axvline` returns a `Line2D` that is automatically excluded from `ax.legend()`
- [ ] D) Yes — but only when `linestyle` is set to a non-solid style like `"--"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `axvline` supports `label` and the resulting line artist appears in the legend alongside the plotted data series

**Explanation:**

- **A is incorrect:** `ax.axvline` directly supports the `label` keyword argument. No additional `ax.plot` call is needed for legend inclusion.
- **B is correct:** `ax.axvline` (like `ax.axhline`, `ax.axspan`, `ax.bar`, and most other plotting functions) accepts a `label` keyword argument. When `label="Deployment Cutoff"` is provided, the returned `Line2D` artist is registered with that label and appears in the legend when `ax.legend()` is called. The legend entry shows a short sample of the line style (dashed orange line) next to the label text.
- **C is incorrect:** `axvline` does return a `Line2D` object, but it is NOT automatically excluded from the legend. The `label` argument is what controls legend inclusion — providing a label makes it eligible; not providing one (or using a label starting with `_`) excludes it.
- **D is incorrect:** There is no linestyle restriction on legend eligibility. Any `axvline` with a non-underscore label appears in the legend regardless of whether its linestyle is solid, dashed, dotted, or any other style.

</details>

---

### Question 45

A researcher uses `xmin` and `xmax` with `ax.axvline` to limit the span:

```python
ax.axvline(x=25, ymin=0.2, ymax=0.8, color="purple", lw=1.5)
```

What does `ymin=0.2, ymax=0.8` specify?

- [ ] A) The vertical line spans from y=0.2 to y=0.8 in data units
- [ ] B) The vertical line spans from 20% to 80% of the axes height (in axes-fraction coordinates)
- [ ] C) The vertical line is visible only when the y-axis range is between 0.2 and 0.8
- [ ] D) The vertical line color opacity ranges from 0.2 (bottom) to 0.8 (top)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The vertical line spans from 20% to 80% of the axes height (in axes-fraction coordinates)

**Explanation:**

- **A is incorrect:** `ymin` and `ymax` in `ax.axvline` are axes-fraction parameters (in [0, 1]), not data-coordinate parameters. To span specific data y-values, `ax.vlines(x=25, ymin=data_ymin, ymax=data_ymax)` would be required.
- **B is correct:** In `ax.axvline`, the `ymin` and `ymax` optional parameters control how far the vertical line extends along the y-axis in **axes-fraction** coordinates. `ymin=0.2` means the line starts 20% from the bottom of the axes box, and `ymax=0.8` means it ends at 80% from the bottom — leaving 20% gaps at both the top and the bottom. This creates a partial vertical reference line that doesn't touch the top and bottom walls.
- **C is incorrect:** `ymin` and `ymax` are rendering extent parameters, not visibility conditions. The line is visible regardless of the y-axis data range.
- **D is incorrect:** There is no gradient opacity feature for `axvline`. Opacity is a uniform `alpha` value applied to the entire line, not a gradient between `ymin` and `ymax`.

</details>

---

### Question 46

A data analyst plots a training loss curve and marks the early stopping epoch using both a vertical line and an annotation:

```python
early_stop_epoch = 35
ax.axvline(x=early_stop_epoch, color="red", linestyle="--", lw=1.5)
ax.text(early_stop_epoch + 1, ax.get_ylim()[1] * 0.95,
        "Early Stop",
        color="red", va="top", ha="left", fontsize=9)
```

Why does `ax.get_ylim()[1] * 0.95` place the text near the top of the plot?

- [ ] A) `get_ylim()[1]` returns the maximum y-value of the plotted data, and multiplying by 0.95 moves the text 5% below that maximum
- [ ] B) `get_ylim()` returns a tuple `(y_min, y_max)`, so `[1]` is `y_max`; multiplying by 0.95 places the text at 95% of `y_max` in data coordinates, near the top axis wall
- [ ] C) `get_ylim()[1]` returns 1.0 (since y-limits are normalized), so the text is placed at y=0.95 in axes fraction
- [ ] D) `ax.get_ylim()[1] * 0.95` is always equivalent to `ax.transAxes` coordinate 0.95 regardless of data range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `get_ylim()` returns a tuple `(y_min, y_max)`, so `[1]` is `y_max`; multiplying by 0.95 places the text at 95% of `y_max` in data coordinates, near the top axis wall

**Explanation:**

- **A is incorrect:** `get_ylim()` returns the **axis limits**, not necessarily the data extremes. If the y-axis was manually set or autoscaled with margins, `y_max` might be slightly higher than the maximum plotted value.
- **B is correct:** `ax.get_ylim()` returns the tuple `(y_lower, y_upper)` of the current y-axis limits. Indexing with `[1]` retrieves `y_upper`. Multiplying by 0.95 gives a value 5% below `y_upper` in data space. Since the text uses default `transData` (no transform specified), this positions the text near the top of the visible y-range in data coordinates. Combined with `va="top"`, the text label appears just inside the top of the plot area.
- **C is incorrect:** `get_ylim()[1]` returns the actual y-axis upper limit in data coordinates (e.g., 2.5 if y-axis goes to 2.5), not a normalized [0, 1] value. Data-coordinate y-limits depend entirely on the data and axis settings.
- **D is incorrect:** `ax.get_ylim()[1] * 0.95` is a data-coordinate value. It is equivalent to `transAxes` y=0.95 only in the special case where `ylim=(0, 1)`. In general, they represent different physical positions on the axes.

</details>

---

### Question 47

A developer draws a baseline loss reference at y=2.3026 (natural log of 10, representing 10-class random baseline cross-entropy):

```python
ax.axhline(y=np.log(10), color="dimgray", linestyle="-.",
           linewidth=1, label=f"Random Baseline (ln 10 ≈ {np.log(10):.3f})")
```

What linestyle does `linestyle="-."` produce?

- [ ] A) A series of dashes of uniform length
- [ ] B) An alternating dash-dot pattern (one dash, one dot, one dash, one dot, ...)
- [ ] C) A dotted line with closely spaced dots
- [ ] D) A long-dash short-dash pattern (double-dash alternation)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An alternating dash-dot pattern (one dash, one dot, one dash, one dot, ...)

**Explanation:**

- **A is incorrect:** A uniform dashed line uses `linestyle="--"`. The `"-."` pattern includes both dashes and dots in alternation.
- **B is correct:** In Matplotlib, `linestyle="-."` (or `"dashdot"`) produces an alternating sequence of a dash followed by a dot: `— · — · — ·`. This is the "dash-dot" pattern, visually distinct from pure dashes (`"--"`) and pure dots (`":"`). It is commonly used for secondary reference lines to distinguish them from primary solid lines and regular dashed lines.
- **C is incorrect:** A dotted line uses `linestyle=":"`. Dots are produced by very short dashes with tight spacing, not by `"-."`.
- **D is incorrect:** A long-dash short-dash pattern is not a standard named linestyle in Matplotlib. Custom dash patterns can be created with `set_dashes([6, 2, 2, 2])` on `Line2D` objects, but `"-."` is specifically the dash-dot pattern.

</details>

---

### Question 48

A data analyst adds multiple reference lines to a validation accuracy plot and then calls `ax.legend()`. She notices that only some of the `axhline` calls appear in the legend. What is the most likely cause?

- [ ] A) `ax.legend()` only includes `axhline` lines and excludes `ax.plot` lines
- [ ] B) The `axhline` calls without a `label` argument (or with labels starting with `_`) are excluded from the legend
- [ ] C) Matplotlib limits the legend to a maximum of 3 artists
- [ ] D) `axhline` lines are excluded from legends unless `zorder` is set higher than the default

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The `axhline` calls without a `label` argument (or with labels starting with `_`) are excluded from the legend

**Explanation:**

- **A is incorrect:** `ax.legend()` collects ALL artists from the axes that have non-underscore labels — whether from `ax.plot`, `ax.axhline`, `ax.scatter`, `ax.bar`, or any other plotting function. There is no preference for or exclusion of any particular function type.
- **B is correct:** Matplotlib's auto-legend system collects artists whose `label` attribute is set to a non-empty, non-underscore-prefixed string. `axhline` calls with `label="something"` appear in the legend; calls without `label=` or with `label="_nolegend_"` (or any `_`-prefixed string) are silently excluded. This is the standard mechanism for controlling which artists appear in legends across all plot types.
- **C is incorrect:** Matplotlib has no hard limit on the number of legend entries. A legend can contain as many entries as needed.
- **D is incorrect:** `zorder` controls rendering order (which artist appears on top in overlapping regions). It has no effect on legend inclusion. Legend eligibility is determined solely by the `label` attribute.

</details>

---

### Question 49

A researcher adds a vertical line at the median of a distribution:

```python
median_val = np.median(data)
ax.axvline(x=median_val, color="darkviolet", lw=2,
           linestyle="--", label=f"Median = {median_val:.2f}")
```

A colleague suggests using `ax.vlines(median_val, ymin=0, ymax=1)` instead. What is the key difference in behavior?

- [ ] A) `ax.vlines` uses axes-fraction coordinates for `ymin` and `ymax`, so `ymin=0, ymax=1` spans the full axes height — identical to `axvline`
- [ ] B) `ax.vlines` uses data coordinates for `ymin` and `ymax`, so `ymin=0, ymax=1` draws a vertical segment only from y=0 to y=1 in data space — unlike `axvline` which spans the full axes regardless of data range
- [ ] C) `ax.vlines` is deprecated and will raise a `FutureWarning`
- [ ] D) `ax.axvline` draws a dashed line by default; `ax.vlines` always draws a solid line

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.vlines` uses data coordinates for `ymin` and `ymax`, so `ymin=0, ymax=1` draws a vertical segment only from y=0 to y=1 in data space — unlike `axvline` which spans the full axes regardless of data range

**Explanation:**

- **A is incorrect:** `ax.vlines` does NOT use axes-fraction coordinates. Its `ymin` and `ymax` parameters are data coordinates. Only `ax.axvline`'s `ymin`/`ymax` parameters use axes fractions.
- **B is correct:** This is the critical distinction. `ax.axvline(x=median_val)` always spans the full visible height of the axes (equivalent to axes fractions [0, 1]) regardless of the y-axis data range. `ax.vlines(median_val, ymin=0, ymax=1)` draws from data y=0 to data y=1. If the y-axis range is [0.5, 3.0], the `vlines` segment would be clipped to [0.5, 1.0] (only the portion visible in the range), while `axvline` would extend through the full visible height [0.5, 3.0].
- **C is incorrect:** `ax.vlines` is an active, maintained Matplotlib function, not deprecated.
- **D is incorrect:** Both `axvline` and `vlines` default to solid lines. The linestyle is controlled by the `linestyle` (or `ls`) argument passed explicitly.

</details>

---

### Question 50

A data scientist adds a zero-crossing reference line to a residual plot:

```python
ax.axhline(y=0, color="black", lw=0.8, linestyle="-")
ax.scatter(y_pred, residuals, alpha=0.5, s=20)
```

What rendering order concern exists with this code, and how can it be addressed?

- [ ] A) `axhline` always renders on top of scatter points; set `zorder=0` on `axhline` to move it behind the scatter
- [ ] B) By default, `axhline` is added before `scatter` in the code, so it has a lower `zorder` (rendered first/behind); no change is needed since scatter points should appear on top of the reference line
- [ ] C) `scatter` always renders below `axhline`; set `zorder=3` on scatter to make points appear above the line
- [ ] D) `axhline` and `scatter` always render at the same `zorder` and cannot be reordered

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) By default, `axhline` is added before `scatter` in the code, so it has a lower `zorder` (rendered first/behind); no change is needed since scatter points should appear on top of the reference line

**Explanation:**

- **A is incorrect:** `axhline` does not automatically render on top. Artists added later have higher default `zorder` (or the same `zorder` but are drawn later, appearing on top). In this code, `scatter` is added after `axhline`, so scatter points appear above the line by default.
- **B is correct:** In Matplotlib, when artists have equal `zorder` values, later additions are rendered on top of earlier ones. `axhline` is called first, then `scatter`. This means the scatter points visually appear above the zero-line, which is the correct intended behavior for a residual plot (the reference line should be visible behind the data points). No `zorder` adjustment is needed — the code order already achieves the correct layering.
- **C is incorrect:** With the given code order, scatter points are already above `axhline` by default. Explicitly setting `zorder=3` on scatter would reinforce this but is unnecessary.
- **D is incorrect:** Artists can always be reordered using the `zorder` keyword argument. The claim that they cannot be reordered is incorrect.

</details>
