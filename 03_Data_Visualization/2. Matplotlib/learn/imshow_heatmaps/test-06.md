# Matplotlib Imshow & Heatmaps — Test 06
## Colorbar Engineering

---

### Question 51

A developer creates a heatmap and adds a colorbar but the colorbar appears **very wide** relative to the heatmap, taking up nearly half the subplot width. Which parameter controls the colorbar's thickness (fraction of the parent axes it occupies)?

```python
im = ax.imshow(data, cmap='plasma')
cbar = fig.colorbar(im, ax=ax)  # Colorbar too wide
```

- [ ] A) `fig.colorbar(im, ax=ax, width=0.05)` — sets absolute width in inches
- [ ] B) `fig.colorbar(im, ax=ax, shrink=0.5)` — shrinks the colorbar length to 50% of axes height
- [ ] C) `fig.colorbar(im, ax=ax, fraction=0.046)` — controls the fraction of parent axes width that the colorbar occupies
- [ ] D) `fig.colorbar(im, ax=ax, pad=0.01)` — reduces the gap between image and colorbar, making it appear thinner

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `fig.colorbar(im, ax=ax, fraction=0.046)` — controls the fraction of parent axes width that the colorbar occupies

**Explanation:**

- **A is incorrect:** `width` is not a valid `fig.colorbar()` parameter. Colorbar dimensions are controlled through `fraction`, `shrink`, `aspect`, and `pad` — none of which use absolute inch measurements.
- **B is incorrect:** `shrink` controls the **length** (height for vertical colorbars) as a fraction of the parent axes height — not the width/thickness. `shrink=0.5` would make the colorbar 50% as tall as the axes while the thickness remains unchanged.
- **C is correct:** `fraction` controls what fraction of the parent axes' width (for vertical colorbars) is stolen to create the colorbar. The default is `fraction=0.15` (15% of axes width). To make the colorbar thinner, use a smaller value: `fraction=0.046` is the commonly recommended value for a visually balanced colorbar. This is the key parameter for controlling colorbar thickness.
- **D is incorrect:** `pad` controls the **gap** (spacing) between the image and the colorbar — not the colorbar thickness. `pad=0.01` would bring the colorbar closer to the image but not change its width.

</details>

---

### Question 52

A developer wants a **horizontal colorbar** below the heatmap instead of the default vertical colorbar on the right:

```python
im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax, orientation='horizontal')
```

Which additional parameter should be set to position the colorbar **below** the image with adequate spacing?

- [ ] A) `location='bottom'` inside `fig.colorbar()`
- [ ] B) `pad=0.1` inside `fig.colorbar()` — increases the gap between the image bottom and the colorbar
- [ ] C) `ax.set_position([0, 0.2, 1, 0.8])` — manually repositions the colorbar below
- [ ] D) `orientation='bottom'` — the correct string for below-image placement

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `location='bottom'` inside `fig.colorbar()`

**Explanation:**

- **A is correct:** In Matplotlib ≥ 3.4, `fig.colorbar(im, ax=ax, location='bottom')` automatically places the colorbar below the axes with the correct orientation. The `location` parameter accepts `'left'`, `'right'`, `'top'`, `'bottom'` — and automatically sets both the orientation and the placement side. For older versions, `orientation='horizontal'` combined with manual `ax_cbar` positioning achieves the same effect.
- **B is incorrect:** `pad` controls the gap between the parent axes and the colorbar — but `orientation='horizontal'` alone doesn't guarantee the colorbar appears BELOW the image. The default position for horizontal colorbars may appear at the bottom, but `location='bottom'` is the explicit and correct parameter.
- **C is incorrect:** `ax.set_position()` repositions the parent image axes, not the colorbar. Repositioning the axes does not create or move a colorbar.
- **D is incorrect:** `'bottom'` is not a valid value for `orientation`. Valid `orientation` values are `'vertical'` and `'horizontal'`. The directional placement uses the `location` parameter.

</details>

---

### Question 53

A developer customizes colorbar tick labels for a heatmap of normalized confusion matrix values [0.0, 1.0]:

```python
im = ax.imshow(norm_cm, cmap='Blues', vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
cbar.set_label("Recall (%)", fontsize=12)
```

What does `cbar.set_label("Recall (%)", fontsize=12)` add to the colorbar?

- [ ] A) A title displayed above the colorbar
- [ ] B) A label on the colorbar's long axis (the y-axis for vertical colorbars) — positioned alongside the colorbar strip to describe the metric being encoded
- [ ] C) Labels on each individual tick mark replacing the numbers
- [ ] D) A text box inside each colorbar segment showing the percentage

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A label on the colorbar's long axis (the y-axis for vertical colorbars) — positioned alongside the colorbar strip to describe the metric being encoded

**Explanation:**

- **A is incorrect:** `cbar.set_label()` does not create a title above the colorbar. For a vertical colorbar, the label appears rotated along the colorbar's right side. Colorbar titles would require accessing `cbar.ax.set_title()`.
- **B is correct:** `cbar.set_label("Recall (%)", fontsize=12)` adds a descriptive text label along the **long axis** of the colorbar (alongside the color strip). For a vertical colorbar, it appears as rotated text on the right side; for a horizontal colorbar, it appears below or above the strip. This is the standard way to annotate what the colorbar's colors represent — in this case, "Recall (%)".
- **C is incorrect:** Individual tick mark labels are set via `cbar.set_ticklabels(['0%', '25%', ...])`— which is already called on the previous line. `set_label()` adds a single axis-level label, not per-tick labels.
- **D is incorrect:** `set_label()` does not inject text inside the colorbar segments. Segment text would require direct `cbar.ax.text()` calls or annotation objects.

</details>

---

### Question 54

A developer examines the `shrink` and `aspect` parameters of `fig.colorbar()`:

```python
cbar1 = fig.colorbar(im, ax=ax, shrink=0.8)
cbar2 = fig.colorbar(im, ax=ax, aspect=30)
```

Which statements correctly describe what each parameter controls?

- [ ] A) `shrink=0.8` makes the colorbar 80% as WIDE as the default; `aspect=30` sets the colorbar height-to-width ratio to 30:1 making it taller
- [ ] B) `shrink=0.8` makes the colorbar 80% as LONG (tall for vertical) as the parent axes height; `aspect=30` sets the colorbar's length-to-width ratio — higher values make it thinner
- [ ] C) Both `shrink` and `aspect` control the same colorbar dimension — they cannot be used together
- [ ] D) `shrink=0.8` clips the colorbar range to 80% of vmax; `aspect=30` sets the number of discrete color bands shown

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `shrink=0.8` makes the colorbar 80% as LONG (tall for vertical) as the parent axes height; `aspect=30` sets the colorbar's length-to-width ratio — higher values make it thinner

**Explanation:**

- **A is incorrect:** `shrink` controls LENGTH, not width. `aspect` is a ratio of length-to-width (not an absolute height value).
- **B is correct:** `shrink` (default 1.0): scales the colorbar **length** as a fraction of the parent axes. `shrink=0.8` makes the vertical colorbar 80% as tall as the image axes height — centering it with whitespace above and below. `aspect` (default ~20): the ratio of long-axis-length to short-axis-width. Higher `aspect` values = longer but thinner colorbar. `aspect=30` makes the colorbar slightly thinner than default (ratio 30:1 instead of 20:1). Lower `aspect` (e.g., 5) makes it stubbier/wider.
- **C is incorrect:** `shrink` and `aspect` control different dimensions (length vs. length-to-width ratio) and can be used simultaneously: `fig.colorbar(im, shrink=0.8, aspect=30)` is valid.
- **D is incorrect:** `shrink` has nothing to do with the data range or `vmax`. Clipping the color range requires modifying `vmin`/`vmax`. `aspect` controls geometric proportions, not discrete color band count (which would require `norm=BoundaryNorm(...)`).

</details>

---

### Question 55

A developer creates a 2×2 subplot grid of heatmaps with a **single shared colorbar** spanning the full right side:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ims = []
for ax, data in zip(axes.flatten(), datasets):
    im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)
    ims.append(im)

cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(),
                    shrink=0.95, aspect=40)
cbar.set_label("Value", fontsize=11)
```

Why is `ax=axes.ravel().tolist()` (a list of all axes) passed instead of `ax=axes[0, 0]`?

- [ ] A) Passing all axes allows `fig.colorbar()` to auto-compute the correct `vmin`/`vmax` from all datasets
- [ ] B) Passing a list of axes instructs Matplotlib to steal space from ALL listed axes proportionally to place the colorbar, creating a single tall bar aligned with the full grid height — rather than stealing from only one subplot and leaving it smaller
- [ ] C) Passing all axes creates one colorbar per subplot but stacked vertically
- [ ] D) The `ax=` parameter is irrelevant for multi-subplot colorbars — only `ims[0]` matters

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Passing a list of axes instructs Matplotlib to steal space from ALL listed axes proportionally to place the colorbar, creating a single tall bar aligned with the full grid height — rather than stealing from only one subplot and leaving it smaller

**Explanation:**

- **A is incorrect:** `fig.colorbar()` never auto-computes data range from multiple images. `vmin`/`vmax` are set on each individual `imshow` call. The colorbar uses the norm of `ims[0]` regardless of how many axes are listed.
- **B is correct:** When `ax=axes.ravel().tolist()` (a Python list of all 4 axes objects) is passed, Matplotlib's `make_axes()` machinery allocates colorbar space by shrinking all listed axes equally. This produces a colorbar that spans the full height of the 2×2 grid — visually aligned with all 4 subplots. If only `ax=axes[0,0]` were passed, space would be stolen from only the top-left subplot, making it narrower than the other three — creating a visually unbalanced layout.
- **C is incorrect:** Passing a list of axes always creates ONE colorbar (not one per subplot). Multiple colorbars require multiple `fig.colorbar()` calls.
- **D is incorrect:** The `ax=` parameter is critical for multi-subplot layout. It determines from which axes space is stolen for the colorbar, directly affecting the final layout balance.

</details>

---

### Question 56

A developer creates a colorbar for an imshow plot and wants to change the colorbar's tick **font size** and **rotation**. Which code correctly does this?

- [ ] A) `cbar.set_ticks(fontsize=8, rotation=45)` — passes styling directly to `set_ticks`
- [ ] B) `cbar.ax.tick_params(labelsize=8, labelrotation=45)` — accesses the underlying axes of the colorbar and applies tick parameter styling
- [ ] C) `cbar.set_fontsize(8)` — a dedicated colorbar font method
- [ ] D) `fig.colorbar(im, fontsize=8, rotation=45)` — font settings go in the `colorbar()` call

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `cbar.ax.tick_params(labelsize=8, labelrotation=45)` — accesses the underlying axes of the colorbar and applies tick parameter styling

**Explanation:**

- **A is incorrect:** `cbar.set_ticks()` accepts a list of tick positions (e.g., `[0, 0.5, 1.0]`) — not styling parameters like `fontsize` or `rotation`. Passing `fontsize=8` as an argument would raise a `TypeError`.
- **B is correct:** The `Colorbar` object has a `.ax` attribute that provides access to its underlying `Axes` object. All standard axes styling methods work on `cbar.ax`: `cbar.ax.tick_params(labelsize=8, labelrotation=45)` applies to all tick labels on the colorbar axis. This is the correct and comprehensive approach to colorbar tick styling.
- **C is incorrect:** `cbar.set_fontsize()` is not a real Colorbar method. No such method exists in the Matplotlib Colorbar API.
- **D is incorrect:** `fig.colorbar()` does not accept `fontsize` or `rotation` as keyword arguments. These styling options must be applied after colorbar creation via `cbar.ax.tick_params()` or by iterating over tick label text objects.

</details>

---

### Question 57

A developer needs to add only **5 evenly-spaced tick marks** on the colorbar for a heatmap with `vmin=0, vmax=100`:

```python
im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=100)
cbar = fig.colorbar(im, ax=ax)
# Add exactly 5 ticks at: 0, 25, 50, 75, 100
```

Which code correctly sets exactly 5 evenly-spaced ticks?

- [ ] A) `cbar.set_ticks(5)` — integer argument sets tick count
- [ ] B) `cbar.set_ticks(np.linspace(0, 100, 5))` — explicitly passes the 5 tick positions as an array
- [ ] C) `cbar.locator = plt.MaxNLocator(5)` — sets a locator for automatic placement
- [ ] D) `cbar.nticks = 5` — attribute assignment sets tick count

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `cbar.set_ticks(np.linspace(0, 100, 5))` — explicitly passes the 5 tick positions as an array

**Explanation:**

- **A is incorrect:** `cbar.set_ticks(5)` does not accept an integer count. `set_ticks()` requires an explicit **sequence of tick positions** (list or array). Passing an integer raises a `TypeError` or is misinterpreted.
- **B is correct:** `np.linspace(0, 100, 5)` generates `[0., 25., 50., 75., 100.]` — exactly 5 evenly-spaced values. `cbar.set_ticks([0, 25, 50, 75, 100])` places tick marks at these exact positions on the colorbar. This is the standard and most explicit method for controlling colorbar tick placement.
- **C is incorrect:** While `MaxNLocator` is a valid tick locator for axes, assigning it via `cbar.locator = ...` is not the standard API. The correct approach for locator-based colorbar ticks uses `cbar.locator = ticker.MaxNLocator(nbins=5)` followed by `cbar.update_ticks()` — but this is more complex and less reliable than explicit `set_ticks()`.
- **D is incorrect:** `cbar.nticks` is not a valid Colorbar attribute. No such attribute exists in the Matplotlib Colorbar API.

</details>

---

### Question 58

A developer creates a colorbar and wants to **reverse its direction** — so the maximum value is at the bottom and minimum at the top (for a vertical colorbar). Which is the correct approach?

- [ ] A) `cbar.invert_axis()` — a dedicated colorbar inversion method
- [ ] B) `cbar.ax.invert_yaxis()` — inverts the y-axis of the colorbar's underlying axes object
- [ ] C) `fig.colorbar(im, orientation='vertical_reversed')` — a built-in orientation string
- [ ] D) Reverse the colormap instead: use `cmap='viridis_r'` — `_r` suffix reverses both the colormap and the colorbar direction

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `cbar.ax.invert_yaxis()` — inverts the y-axis of the colorbar's underlying axes object

**Explanation:**

- **A is incorrect:** `cbar.invert_axis()` is not a real Matplotlib method. The Colorbar API does not have a standalone inversion method.
- **B is correct:** Since `cbar.ax` is a standard Matplotlib `Axes` object, all standard axes methods apply. `cbar.ax.invert_yaxis()` inverts the y-axis direction of the colorbar — placing the maximum value at the bottom and minimum at the top. This technique applies to vertical colorbars; for horizontal colorbars, use `cbar.ax.invert_xaxis()`.
- **C is incorrect:** `'vertical_reversed'` is not a valid `orientation` string. Valid options are only `'vertical'` and `'horizontal'`.
- **D is incorrect:** While `cmap='viridis_r'` reverses the colormap colors (dark purple ↔ yellow), it does NOT reverse the colorbar direction/axis. The colorbar tick positions and values remain unchanged — only which color corresponds to which value changes. To reverse the displayed axis direction, `cbar.ax.invert_yaxis()` is needed.

</details>

---

### Question 59

A developer creates a **symmetric diverging colorbar** for a correlation matrix and notices the tick at 0 is missing. The current code is:

```python
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
cbar = fig.colorbar(im, ax=ax)
```

How does the developer ensure tick marks appear at exactly `-1.0`, `-0.5`, `0.0`, `+0.5`, `+1.0`?

- [ ] A) `cbar.set_ticks([-1, -0.5, 0, 0.5, 1])` followed by `cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '+0.5', '+1.0'])`
- [ ] B) `cbar.autoscale()` — automatically adds the zero tick
- [ ] C) `ax.imshow(corr, ticks=[-1, -0.5, 0, 0.5, 1])` — ticks are specified in `imshow`
- [ ] D) `fig.colorbar(im, boundaries=[-1, -0.5, 0, 0.5, 1])` — `boundaries` sets the tick positions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `cbar.set_ticks([-1, -0.5, 0, 0.5, 1])` followed by `cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '+0.5', '+1.0'])`

**Explanation:**

- **A is correct:** `cbar.set_ticks(positions)` places tick marks at the specified positions. `cbar.set_ticklabels(labels)` sets the display text for each tick. Together they give full control over tick placement and labeling. The list `[-1, -0.5, 0, 0.5, 1]` places 5 evenly-spaced ticks including the crucial 0.0 center tick.
- **B is incorrect:** `cbar.autoscale()` is not a standard Colorbar method, and even if it were, it would not guarantee a tick at exactly 0. The auto-ticker may skip 0 if it doesn't fall on a "nice" auto-interval.
- **C is incorrect:** `ax.imshow()` does not accept a `ticks=` parameter. Tick control belongs to the colorbar API, not `imshow`.
- **D is incorrect:** `boundaries` in `fig.colorbar()` creates a **discrete/quantized** colorbar with boundaries between color bands — used for categorical/discrete colormaps with `BoundaryNorm`. It does not set tick positions for a continuous colorbar.

</details>

---

### Question 60

A developer has a colorbar where `pad=0.05` produces too large a gap between the heatmap and the colorbar. What does `pad` control, and what value reduces the gap?

- [ ] A) `pad` controls colorbar thickness; use `pad=0.01` to make it thinner
- [ ] B) `pad` controls the fractional gap between the parent axes and the colorbar; use `pad=0.01` to reduce spacing to 1% of the parent axes width
- [ ] C) `pad` controls the colorbar tick label indentation; smaller values move labels closer to the color strip
- [ ] D) `pad` is the number of empty pixels between the image border and colorbar; use `pad=1` for minimal separation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `pad` controls the fractional gap between the parent axes and the colorbar; use `pad=0.01` to reduce spacing to 1% of the parent axes width

**Explanation:**

- **A is incorrect:** `pad` controls the **gap** (spacing), not the colorbar thickness. Colorbar thickness is controlled by `fraction` (for absolute proportion) or `aspect` (for length-to-width ratio).
- **B is correct:** `pad` in `fig.colorbar()` sets the fractional distance between the parent axes edge and the colorbar, expressed as a fraction of the parent axes width (for vertical colorbars). Default `pad` is approximately 0.05 (5% of axes width). To reduce the gap: `fig.colorbar(im, ax=ax, pad=0.01)` creates only a 1% gap — bringing the colorbar much closer to the image. A very small `pad` (e.g., 0.01) nearly eliminates the space between image and colorbar.
- **C is incorrect:** Tick label indentation from the color strip is controlled by tick properties (accessible via `cbar.ax.tick_params(pad=...)`), not the `fig.colorbar()` `pad` parameter — a naming collision worth knowing.
- **D is incorrect:** `pad` is a **fractional** parameter (dimensionless, relative to parent axes size), not an absolute pixel count. `pad=1` would create a gap equal to 100% of the parent axes width — placing the colorbar far to the right of the figure, likely outside the visible area.

</details>
