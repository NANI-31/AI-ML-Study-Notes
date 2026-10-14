# Matplotlib Annotations & Legends — Test 06: Region & Span Highlighting

---

### Question 51

A data scientist highlights the overfitting region on a training curve with:

```python
ax.axvspan(xmin=40, xmax=100,
           facecolor="salmon",
           alpha=0.15,
           label="Overfitting Region")
```

How does `axvspan` determine the vertical extent of the shaded region?

- [ ] A) It spans only between the minimum and maximum y-values of the plotted data within x=[40, 100]
- [ ] B) It spans the full height of the axes box from the bottom wall to the top wall, regardless of the y-axis data range
- [ ] C) It spans from y=0 to y=100 because no `ymin`/`ymax` arguments are provided
- [ ] D) It spans from the y-axis minimum to `y=alpha*y_max` (scaled by the opacity)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It spans the full height of the axes box from the bottom wall to the top wall, regardless of the y-axis data range

**Explanation:**

- **A is incorrect:** `axvspan` does not examine the plotted data to determine its vertical extent. It is a geometric span that covers the full axes height independently of what data is plotted within that x range.
- **B is correct:** `ax.axvspan(xmin, xmax)` draws a vertical shaded band that spans the **full y-extent** of the axes — from `ymin=0` to `ymax=1` in axes-fraction coordinates by default. This means the band always covers the entire visible height of the plot area, making it a clean background region regardless of the y-axis scale or the plotted data values within that x range.
- **C is incorrect:** The default `ymin` and `ymax` for `axvspan` are 0 and 1 in axes-fraction coordinates, not in data coordinates. There is no relationship between the default extent and the value 100.
- **D is incorrect:** `alpha` controls the opacity (transparency) of the shaded region, not its geometric extent. The span covers the full axes height at 15% opacity — the visual intensity is reduced, not the geometric coverage.

</details>

---

### Question 52

A researcher adds confidence band shading around a calibration line:

```python
ax.axhspan(ymin=0.48, ymax=0.52,
           facecolor="lightblue",
           alpha=0.4,
           edgecolor="none")
```

What does `edgecolor="none"` do?

- [ ] A) Makes the entire shaded span invisible
- [ ] B) Removes the border outline of the shaded rectangle, producing a smooth band with no visible edges
- [ ] C) Prevents the span from being included in the legend
- [ ] D) Makes the span's fill colour transparent at the edges

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Removes the border outline of the shaded rectangle, producing a smooth band with no visible edges

**Explanation:**

- **A is incorrect:** `edgecolor="none"` removes only the border stroke, not the fill. The filled band with `facecolor="lightblue"` and `alpha=0.4` remains visible.
- **B is correct:** `ax.axhspan` creates a `Polygon` (rectangle) patch. `edgecolor` controls the colour of the patch's border strokes. Setting `edgecolor="none"` (or equivalently `edgecolor="None"` or `ec="none"`) suppresses the border outline, producing a clean shaded band with no visible border lines at the top and bottom edges. This is the typical preference for background highlighting where a visible border would appear too prominent.
- **C is incorrect:** Legend inclusion is controlled by the `label` argument. `edgecolor` has no effect on whether the span appears in the legend.
- **D is incorrect:** There is no gradient or edge-fade transparency effect. `edgecolor="none"` is a binary "no border" setting. The uniform `alpha=0.4` applies to the entire fill.

</details>

---

### Question 53

A machine learning engineer highlights two separate anomaly windows on a time-series plot:

```python
anomaly_windows = [(120, 145), (230, 260)]
for xmin, xmax in anomaly_windows:
    ax.axvspan(xmin, xmax, facecolor="red", alpha=0.1)
```

Each `axvspan` call returns an artist. Does calling `ax.legend()` after this code produce legend entries for the anomaly spans?

- [ ] A) Yes — all `axvspan` calls automatically generate legend entries labeled "axvspan"
- [ ] B) No — no `label` argument was provided, so neither span appears in the legend
- [ ] C) Yes — only the first `axvspan` call appears in the legend; duplicates are suppressed
- [ ] D) No — `axvspan` patches are always excluded from Matplotlib legends

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — no `label` argument was provided, so neither span appears in the legend

**Explanation:**

- **A is incorrect:** Matplotlib does not auto-generate labels for unnamed artists. Without a `label=` argument, the artist label defaults to `_nolegend_` (an underscore-prefixed string), which is the convention for "do not include in legend."
- **B is correct:** Legend inclusion requires a non-underscore `label` argument. Since neither `axvspan` call includes `label="Anomaly Window"` (or any label), both spans are excluded from the legend. To add one legend entry representing both spans, pass `label="Anomaly Window"` to only the first call (or create a proxy artist using `matplotlib.patches.Patch`).
- **C is incorrect:** Matplotlib does not suppress "duplicate" artists from legends — it includes all artists with valid labels. If both spans had labels, both would appear. With no labels, neither appears.
- **D is incorrect:** `axvspan` patches (Polygon artists) are fully eligible for legend inclusion when a `label` is provided. They are not categorically excluded from legends.

</details>

---

### Question 54

A researcher uses `axhspan` to highlight a "safe operating zone":

```python
ax.axhspan(ymin=0.8, ymax=0.95,
           xmin=0.1, xmax=0.9,
           facecolor="lightgreen",
           alpha=0.3)
```

What does specifying `xmin=0.1, xmax=0.9` do?

- [ ] A) Limits the horizontal span of the shaded region to 10% to 90% of the axes width in axes-fraction coordinates
- [ ] B) Limits the shaded region to x-data range from 0.1 to 0.9
- [ ] C) Shifts the shade start 10% from the left and clips it 10% before the right edge in data units
- [ ] D) `axhspan` does not accept `xmin`/`xmax` — this raises a `TypeError`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Limits the horizontal span of the shaded region to 10% to 90% of the axes width in axes-fraction coordinates

**Explanation:**

- **A is correct:** `ax.axhspan` supports optional `xmin` and `xmax` arguments that control the horizontal extent of the shaded region, expressed as axes-fraction values in [0, 1]. `xmin=0.1` starts the shading at 10% from the left edge of the axes box, and `xmax=0.9` ends it at 90% from the left. This creates a shaded band that does not extend to the full width — leaving 10% margins on each side. This is consistent with how `ax.axhline`'s `xmin`/`xmax` work (axes fractions, not data coordinates).
- **B is incorrect:** `axhspan`'s `xmin`/`xmax` are axes-fraction values, not data-coordinate x-values. To shade a specific x-data range, `ax.axvspan` (for x-range) or polygon patches would be needed.
- **C is incorrect:** The coordinates are axes fractions (relative to the plot box), not data units. The interpretation is geometric percentage of the axes width.
- **D is incorrect:** `ax.axhspan` does accept `xmin` and `xmax` as optional keyword arguments for partial-width shading. No `TypeError` is raised.

</details>

---

### Question 55

A data analyst uses both `axhspan` and `axvspan` to create a highlighted rectangular region:

```python
ax.axhspan(0.7, 0.9, alpha=0.2, facecolor="blue")
ax.axvspan(20, 40, alpha=0.2, facecolor="red")
```

What does the overlapping intersection region look like?

- [ ] A) The intersection shows only the blue color because `axhspan` is called first
- [ ] B) The intersection shows only the red color because `axvspan` is called later and draws on top
- [ ] C) The intersection appears as a blended color (a mixture of semi-transparent red and semi-transparent blue) because both spans have `alpha=0.2`
- [ ] D) No overlap occurs because `axhspan` and `axvspan` are on separate rendering layers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The intersection appears as a blended color (a mixture of semi-transparent red and semi-transparent blue) because both spans have `alpha=0.2`

**Explanation:**

- **A is incorrect:** With `alpha=0.2`, both spans are semi-transparent. The later-drawn span does not completely cover the earlier one. Both contribute to the rendered colour in the overlap region through alpha compositing.
- **B is incorrect:** While `axvspan` is rendered after `axhspan` (painter's algorithm), the red span's `alpha=0.2` means it is 80% transparent — the blue span underneath is still partially visible through the red span.
- **C is correct:** Alpha compositing blends the colours of overlapping semi-transparent layers. In the intersection region, the axes background shows through both spans. The final colour in the overlap is an additive mixture of the blue span (at 20% opacity) and the red span (at 20% opacity) composited over the background. The exact colour depends on the compositing math but visually appears as a darker blue-red blend distinct from either span alone.
- **D is incorrect:** All artists in a single axes are rendered on the same layer (unless `zorder` creates separation). `axhspan` and `axvspan` both produce `Polygon` patches that overlap in shared screen space.

</details>

---

### Question 56

A researcher highlights every other epoch in a cyclical training schedule:

```python
cycle_length = 20
n_cycles = 5
for i in range(n_cycles):
    if i % 2 == 0:
        ax.axvspan(i * cycle_length, (i + 1) * cycle_length,
                   facecolor="lavender", alpha=0.3)
```

This shades epochs [0, 20], [40, 60], and [80, 100]. What does `alpha=0.3` contribute to the visualization's effectiveness?

- [ ] A) It ensures the shaded bands have hard, clearly visible edges
- [ ] B) It makes the shaded bands visually recessive (subtle background highlights) so the plotted data lines remain the visual focus rather than the shading
- [ ] C) It reduces the rendering time by 70%
- [ ] D) It prevents the shaded bands from receiving hover tooltips in interactive backends

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It makes the shaded bands visually recessive (subtle background highlights) so the plotted data lines remain the visual focus rather than the shading

**Explanation:**

- **A is incorrect:** `alpha=0.3` (30% opacity) produces soft, translucent bands. The "hard, clearly visible edges" would require `alpha=1.0` with a border color. Low alpha specifically softens the visual impact.
- **B is correct:** Background span shading serves as a contextual reference layer — it labels regions without dominating the visualization. A low `alpha` (here 0.3 = 30% opacity) keeps the shaded bands in the visual background, allowing the training curves plotted on top to remain the primary visual focus. The shading is visible enough to communicate the cyclic pattern but not so strong that it distracts from the main data. This is the canonical use of low-alpha spans in data visualization.
- **C is incorrect:** `alpha` affects the visual rendering compositing calculation but has negligible impact on rendering speed. The geometry (number of spans, vertices) determines rendering performance far more than opacity values.
- **D is incorrect:** `alpha` is a visual property only. It has no effect on interactive event handling, hover detection, or tooltip behavior in Matplotlib's interactive backends.

</details>

---

### Question 57

A developer adds a shaded uncertainty region between lower and upper confidence bounds using `axhspan`:

```python
ax.axhspan(ymin=ci_lower, ymax=ci_upper, facecolor="steelblue", alpha=0.2)
```

A colleague suggests using `ax.fill_between(x, ci_lower_array, ci_upper_array, ...)` instead. What is the key limitation of the `axhspan` approach for this use case?

- [ ] A) `axhspan` does not support `facecolor` — only solid colors can be applied
- [ ] B) `axhspan` draws a constant-height horizontal band across the full x-range; if the confidence interval width varies with x, `axhspan` cannot represent it — only `fill_between` handles variable-width bands
- [ ] C) `axhspan` cannot be added to a legend
- [ ] D) `axhspan` with `facecolor="steelblue"` will conflict with plotted line colours of the same colour

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axhspan` draws a constant-height horizontal band across the full x-range; if the confidence interval width varies with x, `axhspan` cannot represent it — only `fill_between` handles variable-width bands

**Explanation:**

- **A is incorrect:** `axhspan` fully supports `facecolor`. Named colours like `"steelblue"` are valid.
- **B is correct:** `ax.axhspan(ymin, ymax)` draws a rectangular band at constant `ymin` and `ymax` values across the full axes width. This represents a fixed-height horizontal strip. In contrast, a confidence interval around a curve (e.g., from a bootstrapped model) varies in width at different x-positions — the upper bound is higher in some regions and lower in others. `ax.fill_between(x, lower_array, upper_array)` is designed exactly for this purpose: it fills the area between two arrays of y-values that change with x, producing a variable-width band that tracks the actual confidence bounds.
- **C is incorrect:** `axhspan` supports the `label` argument and appears in the legend like any other labeled artist.
- **D is incorrect:** Having the same `facecolor` as a plotted line does not create a conflict. The alpha transparency distinguishes the band from the solid line.

</details>

---

### Question 58

A data analyst sets `alpha=0.0` on an `axvspan`. What is the visual result?

- [ ] A) A fully opaque white band
- [ ] B) No visible shading — the span is completely invisible
- [ ] C) A span visible only as a border outline (edge) with no fill
- [ ] D) An error: `alpha` cannot be set to 0 for `axvspan`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No visible shading — the span is completely invisible

**Explanation:**

- **A is incorrect:** `alpha=0.0` means fully transparent (0% opacity), not white. White fill would require `facecolor="white", alpha=1.0`.
- **B is correct:** `alpha=0.0` sets the span's opacity to zero, making it completely transparent and thus invisible. The `Polygon` patch still exists as an object on the axes, but it renders as a fully transparent rectangle — visually equivalent to no shading. This is occasionally useful when toggling span visibility programmatically (`artist.set_alpha(0.3)` to show, `artist.set_alpha(0.0)` to hide) without removing the object from the axes.
- **C is incorrect:** A border-only span (edge visible, no fill) would require `facecolor="none"` (or `fill=False`) while keeping `edgecolor` set to a visible colour. `alpha=0.0` applies to the entire artist including any edge colour, making both fill and edge invisible.
- **D is incorrect:** `alpha=0.0` is a valid value for any Matplotlib artist. No error is raised.

</details>

---

### Question 59

A researcher shades a "safe" performance zone and a "danger" zone on the same axes:

```python
ax.axhspan(0.9, 1.0, facecolor="green", alpha=0.15, label="Safe Zone")
ax.axhspan(0.0, 0.6, facecolor="red", alpha=0.15, label="Danger Zone")
```

The y-axis range is [0, 1.0]. What is the unshaded y-range between these two spans?

- [ ] A) [0.0, 0.9] (from y=0 up to y=0.9)
- [ ] B) [0.6, 0.9] (the gap between the top of the danger zone and the bottom of the safe zone)
- [ ] C) [0.6, 1.0] (from y=0.6 to the top of the axes)
- [ ] D) There is no unshaded region because the two spans together cover y=[0, 0.6] and y=[0.9, 1.0]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) [0.6, 0.9] (the gap between the top of the danger zone and the bottom of the safe zone)

**Explanation:**

- **A is incorrect:** The unshaded region is not the full [0, 0.9] range. The danger zone covers [0, 0.6], so the range [0, 0.6] is shaded red.
- **B is correct:** The danger zone spans y=[0.0, 0.6] and the safe zone spans y=[0.9, 1.0]. The region y=[0.6, 0.9] falls between the top of the danger zone and the bottom of the safe zone — it is not covered by either `axhspan` call and remains unshaded. This represents the intermediate "caution" zone where neither green nor red highlighting applies.
- **C is incorrect:** The safe zone starts at y=0.9, so the range [0.6, 1.0] is not fully unshaded — the segment [0.9, 1.0] is shaded green.
- **D is incorrect:** The two spans do not cover the entire y-axis. The gap between the danger zone top (0.6) and the safe zone bottom (0.9) is the unshaded region [0.6, 0.9].

</details>

---

### Question 60

A machine learning engineer adds a span to highlight the validation period on a time-series and wants it to appear **behind** all plotted lines. Which approach correctly places the `axvspan` below the line plots?

- [ ] A) Call `ax.axvspan(...)` after all `ax.plot(...)` calls — later artists are rendered behind earlier ones
- [ ] B) Call `ax.axvspan(..., zorder=0)` before any `ax.plot(...)` calls — the explicit low `zorder` ensures rendering below lines
- [ ] C) Set `alpha=0.01` to make the span nearly invisible and appear to be behind the lines
- [ ] D) Use `ax.add_patch(Rectangle(...))` instead of `ax.axvspan` to control layering

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Call `ax.axvspan(..., zorder=0)` before any `ax.plot(...)` calls — the explicit low `zorder` ensures rendering below lines

**Explanation:**

- **A is incorrect:** In Matplotlib's rendering system, when artists have equal `zorder`, later additions render **on top** of earlier ones — not behind. So calling `axvspan` after `ax.plot` would place the span above (in front of) the lines, obscuring them.
- **B is correct:** `zorder` is the correct parameter for controlling rendering order. The default `zorder` for `ax.plot` lines is approximately 2. Setting `zorder=0` on `axvspan` places it below all default-zorder artists, ensuring the shading appears as a background layer behind the data lines. Calling the `axvspan` before or after `ax.plot` is irrelevant when `zorder` is explicitly set.
- **C is incorrect:** Setting `alpha=0.01` makes the span nearly invisible (1% opacity), which defeats the purpose of adding it. The span would be unreadable rather than visually behind the lines.
- **D is incorrect:** While `Rectangle` patches can be used as an alternative to `axvspan`, `ax.add_patch` also uses the `zorder` system. The key is always the `zorder` parameter, and `axvspan` itself supports `zorder`. There is no need to switch to `add_patch` just for layering control.

</details>
