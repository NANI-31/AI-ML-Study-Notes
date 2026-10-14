# Matplotlib Figure & Axes Objective Test — Set 06
# Topic: Manual Axes Placement with add_axes (Questions 51–60)

---

### Question 51

A developer creates a custom axes layout using `fig.add_axes()`:

```python
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.15, 0.12, 0.70, 0.75])
ax.plot([1, 2, 3], [4, 5, 6])
```

The four values in `[0.15, 0.12, 0.70, 0.75]` represent which geometric properties in which coordinate system?

- [ ] A) `[x_center, y_center, width, height]` in data coordinates
- [ ] B) `[left, bottom, width, height]` in normalized figure coordinates (0 to 1)
- [ ] C) `[x_pixels, y_pixels, width_pixels, height_pixels]` in pixel coordinates
- [ ] D) `[left, bottom, right, top]` in normalized figure coordinates (0 to 1)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `[left, bottom, width, height]` in normalized figure coordinates (0 to 1)

**Explanation:**

- **A is incorrect:** The coordinates are in normalized figure space (0 to 1), not data coordinates. Data coordinates depend on the plotted data range and are set by `ax.set_xlim()` and `ax.set_ylim()`.
- **B is correct:** `fig.add_axes([left, bottom, width, height])` uses **normalized figure coordinates** where (0,0) is the bottom-left corner and (1,1) is the top-right corner: `left=0.15` = the left edge of the axes starts at 15% from the figure left; `bottom=0.12` = the bottom edge starts at 12% from the figure bottom; `width=0.70` = the axes spans 70% of the figure width; `height=0.75` = the axes spans 75% of the figure height. This places the axes from (0.15, 0.12) to (0.85, 0.87) in figure coordinates.
- **C is incorrect:** The `add_axes` coordinate system is normalized (0–1), not pixel-based. Pixel coordinates are never used directly in Matplotlib's high-level API.
- **D is incorrect:** The format `[left, bottom, right, top]` would be `[0.15, 0.12, 0.85, 0.87]` for this axes. The actual format is `[left, bottom, width, height]` — the fourth parameter is the HEIGHT, not the top edge position.

</details>

---

### Question 52

A developer calculates the expected bounds of an axes created with:

```python
ax = fig.add_axes([0.2, 0.3, 0.5, 0.4])
```

What are the **right** and **top** edges of this axes in normalized figure coordinates?

- [ ] A) Right = 0.5, Top = 0.4
- [ ] B) Right = 0.7, Top = 0.7
- [ ] C) Right = 0.5, Top = 0.7
- [ ] D) Right = 0.7, Top = 0.4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Right = 0.7, Top = 0.7

**Explanation:**

- **A is incorrect:** 0.5 and 0.4 are the WIDTH and HEIGHT, not the right and top coordinates.
- **B is correct:** Given `[left=0.2, bottom=0.3, width=0.5, height=0.4]`: Right edge = left + width = 0.2 + 0.5 = **0.7**; Top edge = bottom + height = 0.3 + 0.4 = **0.7**. So the axes spans from (0.2, 0.3) in the bottom-left to (0.7, 0.7) in the top-right — in normalized figure coordinates.
- **C is incorrect:** Right = 0.5 would mean width = 0 (0.2 + 0 = 0.2 ≠ 0.5). Top = 0.7 is correct, but right = 0.5 is wrong.
- **D is incorrect:** Right = 0.7 is correct, but Top = 0.4 would mean height = 0.1 (0.3 + 0.1 = 0.4). The actual height is 0.4, giving top = 0.3 + 0.4 = 0.7.

</details>

---

### Question 53

A developer tries to create an axes that extends beyond the figure boundary:

```python
fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.1, 0.1, 1.2, 0.8])  # width = 1.2
```

What happens?

- [ ] A) A `ValueError` is raised because width > 1.0 is invalid
- [ ] B) The axes is created and placed partly outside the visible canvas — the portion beyond the figure boundary is clipped and not visible
- [ ] C) Matplotlib automatically rescales the axes to fit within [0, 1]
- [ ] D) The figure automatically resizes to accommodate the wider axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The axes is created and placed partly outside the visible canvas — the portion beyond the figure boundary is clipped and not visible

**Explanation:**

- **A is incorrect:** Matplotlib does NOT raise a `ValueError` for out-of-bounds `add_axes` coordinates. The API accepts any float values — it is the developer's responsibility to use valid normalized coordinates.
- **B is correct:** `width=1.2` means the axes would span from `left=0.1` to `left+width=1.3` in normalized figure coordinates. Since the figure canvas ends at 1.0, the rightmost 30% of the axes (from 1.0 to 1.3) extends beyond the canvas and is **clipped** — it is simply not rendered. Only the portion within [0, 1] is visible. This is a common `add_axes` bug when developers miscalculate coordinates, especially when mixing `[left, bottom, width, height]` with `[left, bottom, right, top]` conventions.
- **C is incorrect:** Matplotlib does not auto-rescale the axes. It accepts the given coordinates as-is and clips the out-of-bounds portion.
- **D is incorrect:** The figure size does NOT automatically expand to contain out-of-bounds axes. `figsize` is fixed unless explicitly changed.

</details>

---

### Question 54

A developer needs to create two axes with NO overlap and NO gap between them — the right edge of the left axes must exactly touch the left edge of the right axes:

```python
fig = plt.figure(figsize=(10, 5))
ax_left  = fig.add_axes([0.05, 0.10, _____, 0.80])
ax_right = fig.add_axes([_____, 0.10, _____, 0.80])
```

Assuming they share the full width from 0.05 to 0.95 (leaving 5% margin on each side) and should be equal width with zero gap, what are the missing values?

- [ ] A) `ax_left: width=0.45`, `ax_right: left=0.50, width=0.45`
- [ ] B) `ax_left: width=0.45`, `ax_right: left=0.55, width=0.45`
- [ ] C) `ax_left: width=0.90`, `ax_right: left=0.95, width=0.90`
- [ ] D) `ax_left: width=0.40`, `ax_right: left=0.50, width=0.50`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax_left: width=0.45`, `ax_right: left=0.50, width=0.45`

**Explanation:**

- **A is correct:** Total available width = 0.95 - 0.05 = 0.90 (leaving 5% margins on each side). With two equal axes and zero gap: each width = 0.90 / 2 = **0.45**. Left axes: `[0.05, 0.10, 0.45, 0.80]` — right edge = 0.05 + 0.45 = 0.50. Right axes: `left=0.50` (exactly touching), width=0.45 — right edge = 0.50 + 0.45 = 0.95. Perfect adjacency with no overlap and no gap.
- **B is incorrect:** `ax_right: left=0.55` would create a gap from 0.50 to 0.55 (5% of figure width) between the two axes. The right edge of `ax_left` is 0.50, but `ax_right` starts at 0.55, leaving a visible gap.
- **C is incorrect:** `width=0.90` for each axes would cause massive overlap — each axes would span almost the full figure width, overlapping significantly.
- **D is incorrect:** Widths 0.40 and 0.50 are unequal, and the geometry doesn't add up to fill the 0.90 available width seamlessly from 0.05 to 0.95.

</details>

---

### Question 55

A developer uses `fig.add_axes()` to create an inset axes at specific normalized coordinates:

```python
fig = plt.figure(figsize=(8, 6))
ax_main = fig.add_axes([0.08, 0.08, 0.85, 0.85])
ax_inset = fig.add_axes([0.65, 0.15, 0.20, 0.25])
```

In this layout, if the main axes has data x-range [0, 100] and y-range [0, 50], do the inset axes `[0.65, 0.15, 0.20, 0.25]` coordinates refer to data units of the main axes?

- [ ] A) Yes — the inset is positioned relative to the main axes data coordinates
- [ ] B) No — all `add_axes` coordinates are in normalized FIGURE coordinates (0 to 1), independent of any other axes or data range
- [ ] C) Yes — but only when the main axes is created first
- [ ] D) The inset axes inherits the coordinate system of the main axes automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — all `add_axes` coordinates are in normalized FIGURE coordinates (0 to 1), independent of any other axes or data range

**Explanation:**

- **A is incorrect:** `add_axes` coordinates are NEVER in data units. They are always normalized figure coordinates (0–1). The data range [0,100] × [0,50] of the main axes has no relationship to the `add_axes` positioning parameters.
- **B is correct:** `fig.add_axes([left, bottom, width, height])` always uses **normalized figure coordinates** where (0,0)=bottom-left of figure and (1,1)=top-right of figure. The inset at `[0.65, 0.15, 0.20, 0.25]` starts at 65% from the left, 15% from the bottom, spans 20% width and 25% height of the FIGURE — completely independent of the main axes data range or position. Both axes exist in the same figure coordinate space.
- **C is incorrect:** The creation order does not affect the coordinate system. All `add_axes` calls always use the same normalized figure coordinate system.
- **D is incorrect:** Axes created with `add_axes` have independent coordinate systems for their DATA — each axes maintains its own `xlim`/`ylim`. The FIGURE-LEVEL positioning is in normalized coordinates, not inherited data coordinates.

</details>

---

### Question 56

A developer creates a non-grid multi-panel layout using `add_axes` for a machine learning report:

```python
fig = plt.figure(figsize=(12, 8))

# Wide top panel for training curves
ax_top = fig.add_axes([0.05, 0.55, 0.90, 0.40])

# Two narrow bottom panels
ax_bot_left  = fig.add_axes([0.05, 0.08, 0.42, 0.38])
ax_bot_right = fig.add_axes([0.53, 0.08, 0.42, 0.38])
```

What is the gap between `ax_bot_left` and `ax_bot_right` in normalized figure coordinates?

- [ ] A) 0.00 (no gap — they touch)
- [ ] B) 0.11 (gap from right edge of ax_bot_left at 0.47 to left edge of ax_bot_right at 0.53 — but actually 0.53-0.47=0.06)
- [ ] C) 0.06 (right edge of ax_bot_left = 0.05+0.42=0.47; left edge of ax_bot_right = 0.53; gap = 0.53-0.47 = 0.06)
- [ ] D) 0.42 (the width of one bottom panel)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.06 (right edge of ax_bot_left = 0.05+0.42=0.47; left edge of ax_bot_right = 0.53; gap = 0.53-0.47 = 0.06)

**Explanation:**

- **A is incorrect:** Right edge of `ax_bot_left` = left + width = 0.05 + 0.42 = 0.47. Left edge of `ax_bot_right` = 0.53. They do not touch (0.47 ≠ 0.53).
- **B is incorrect:** The gap is 0.53 - 0.47 = 0.06, not 0.11. Option B's arithmetic is stated as "0.11" but then corrected inline — the correct value is 0.06.
- **C is correct:** `ax_bot_left` right edge = 0.05 + 0.42 = **0.47**. `ax_bot_right` left edge = **0.53**. Gap = 0.53 − 0.47 = **0.06** in normalized figure coordinates. At `figsize=(12, 8)`, this corresponds to 0.06 × 12 = 0.72 inches of physical space between the two panels.
- **D is incorrect:** 0.42 is the width of each bottom panel, not the gap between them.

</details>

---

### Question 57

A developer creates two overlapping axes using `add_axes` for a specialized visualization:

```python
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Same coordinates
ax1.plot([1, 2, 3], [1, 4, 9], color="blue")
ax2.plot([1, 2, 3], [9, 4, 1], color="red")
```

What is the visual result of two axes with identical `add_axes` coordinates?

- [ ] A) A `ValueError` is raised — duplicate coordinates are not allowed
- [ ] B) Both axes are rendered stacked on the same position — the second axes (`ax2`) appears on top of the first; both datasets are visible but on separate axes objects with independent tick systems
- [ ] C) Only the last axes (`ax2`) is rendered — the first is overwritten
- [ ] D) The two datasets are merged into a single axes automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Both axes are rendered stacked on the same position — the second axes (`ax2`) appears on top of the first; both datasets are visible but on separate axes objects with independent tick systems

**Explanation:**

- **A is incorrect:** Matplotlib allows multiple axes with identical or overlapping `add_axes` coordinates. No error is raised.
- **B is correct:** Matplotlib renders axes in creation order (later axes appear on top). Two axes with identical coordinates create a **layered overlay** — both are drawn to the figure, with `ax2` rendered on top of `ax1`. Both datasets are visible. Each axes maintains its own independent tick marks, labels, and coordinate system. This is intentional in some visualization patterns (e.g., dual y-axis charts where `ax2 = ax1.twinx()` creates a second axes sharing the same position). In this case, both axes have the same (default) x and y scale, so the plots overlay in the same data space.
- **C is incorrect:** `add_axes` never overwrites existing axes — it always creates new ones and adds them to the figure's axes list.
- **D is incorrect:** Matplotlib does not merge datasets from separate axes objects. Each `Axes` maintains its own independent artists, coordinate system, and tick configuration.

</details>

---

### Question 58

A developer creates a polar axes using `add_axes` with a projection:

```python
fig = plt.figure(figsize=(6, 6))
ax_polar = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
ax_polar.plot(theta, r)
```

Is `projection="polar"` a valid parameter for `fig.add_axes()`?

- [ ] A) No — polar projections require `plt.subplot(projection="polar")` syntax
- [ ] B) Yes — `fig.add_axes()` accepts a `projection` keyword for specialized coordinate systems including `"polar"`, `"3d"`, and others
- [ ] C) No — `add_axes` only creates standard Cartesian axes
- [ ] D) Yes — but only when the figure uses `constrained_layout=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `fig.add_axes()` accepts a `projection` keyword for specialized coordinate systems including `"polar"`, `"3d"`, and others

**Explanation:**

- **A is incorrect:** Polar projections are not limited to `plt.subplot()` syntax. `fig.add_axes()`, `fig.add_subplot()`, and `plt.subplots()` all accept the `projection=` keyword.
- **B is correct:** `fig.add_axes([left, bottom, width, height], projection="polar")` is valid Matplotlib syntax. The `projection` keyword selects the axes class to instantiate — `"polar"` creates a `PolarAxes`, `"3d"` creates an `Axes3D` (requires `mpl_toolkits.mplot3d`). This allows placing specialized axes (polar, geographic, 3D) at arbitrary positions using the `add_axes` coordinate system.
- **C is incorrect:** `add_axes` is not limited to Cartesian axes. The `projection` parameter provides access to all registered axes classes.
- **D is incorrect:** The `projection` parameter works regardless of layout engine. It is independent of `constrained_layout`, `tight_layout`, or any other layout setting.

</details>

---

### Question 59

A developer uses `fig.add_axes()` to create a colorbar-like side panel:

```python
fig = plt.figure(figsize=(8, 6))
ax_main  = fig.add_axes([0.05, 0.10, 0.80, 0.85])
ax_cbar  = fig.add_axes([0.88, 0.10, 0.04, 0.85])
```

In this layout, what is the physical width (in inches) of the colorbar axes `ax_cbar`?

- [ ] A) 0.04 inches
- [ ] B) 0.32 inches
- [ ] C) 4 inches
- [ ] D) 0.88 inches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.32 inches

**Explanation:**

- **A is incorrect:** 0.04 is the normalized width (4% of figure width) — not the physical width in inches.
- **B is correct:** Physical width = normalized width × figure width = 0.04 × 8 inches = **0.32 inches**. Normalized figure coordinates are fractions of the actual figure dimensions. With `figsize=(8, 6)`, each 1% of figure width equals 0.08 inches. The colorbar axes at 4% width = 0.04 × 8 = 0.32 inches wide. This is a typical narrow colorbar width.
- **C is incorrect:** 4 inches would require a normalized width of 0.5 (50% of the 8-inch figure), not 0.04.
- **D is incorrect:** 0.88 is the normalized LEFT POSITION of the colorbar axes, not its width. The width parameter is 0.04.

</details>

---

### Question 60

A developer creates a figure with `add_axes` and wants to retrieve the axes' bounds after creation:

```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.12, 0.15, 0.75, 0.70])
bounds = ax.get_position().bounds
print(bounds)
```

What does `ax.get_position().bounds` return?

- [ ] A) The pixel coordinates of the axes on screen
- [ ] B) The data limits `(xmin, ymin, xmax, ymax)` of the plotted data
- [ ] C) The normalized figure coordinates `(left, bottom, width, height)` — same as what was passed to `add_axes`
- [ ] D) The figure size in inches `(width, height)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The normalized figure coordinates `(left, bottom, width, height)` — same as what was passed to `add_axes`

**Explanation:**

- **A is incorrect:** `get_position()` returns normalized figure coordinates, not pixel screen coordinates. To get pixel coordinates, you would use `ax.get_window_extent()`.
- **B is incorrect:** Data limits (the range of plotted data) are obtained via `ax.get_xlim()` and `ax.get_ylim()`. `get_position()` is about the axes' position within the figure, not the data range.
- **C is correct:** `ax.get_position()` returns a `Bbox` object. `.bounds` on a `Bbox` returns a tuple `(x0, y0, width, height)` — i.e., `(left, bottom, width, height)` in normalized figure coordinates. For `ax = fig.add_axes([0.12, 0.15, 0.75, 0.70])`, this prints `(0.12, 0.15, 0.75, 0.70)`. This is useful for programmatically querying axes layout and for dynamically repositioning axes with `ax.set_position([new_left, new_bottom, new_width, new_height])`.
- **D is incorrect:** Figure size in inches is retrieved via `fig.get_size_inches()` which returns `(width, height)`. `ax.get_position()` is axes-specific, not figure-level.

</details>
