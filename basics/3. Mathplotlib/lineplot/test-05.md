# Matplotlib Line Plot Objective Test — Set 05

---

### 41. Question

What is the default `zorder` assigned by Matplotlib to standard line artists (`Line2D`) created via `plt.plot()`?

- [ ] **A.** `zorder = 0`
- [ ] **B.** `zorder = 1`
- [ ] **C.** `zorder = 2`
- [ ] **D.** `zorder = 5`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `zorder = 0` is reserved for background canvas elements and backplates.
- **Option B (Incorrect):** `zorder = 1` is the default drawing layer for patches (such as bar charts, rectangles, and polygons).
- **Option C (Correct):** In Matplotlib's default layer hierarchy, standard lines (`Line2D`) have a default **`zorder = 2`**. Text elements and annotations default to `zorder = 3`. This ensures lines render on top of filled background patches but underneath axis text and titles.
- **Option D (Incorrect):** Higher values like `zorder = 5` must be set explicitly by the user to lift specific artists above standard layers.

</details>

---

### 42. Question

An engineer highlights an outage window using a semi-transparent vertical span:

```python
plt.plot(time, network_traffic, color='navy', label="Traffic")  # default zorder=2
plt.axvspan(14, 18, color='crimson', alpha=0.6, zorder=3)
plt.legend()
plt.show()
```

The engineer observes that the crimson span visually overlaps and covers the navy network traffic line, obscuring fine line variations. How can this visual layering issue be resolved so the navy line sits **on top** of the shaded span?

- [ ] **A.** Set `zorder=5` (or any value greater than 3) on the `plt.plot()` call
- [ ] **B.** Set `alpha=0.0` on the `plt.plot()` call
- [ ] **C.** Reorder the imports in the script
- [ ] **D.** Pass `above=True` to `plt.axvspan()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Artists with higher `zorder` values are rendered on top of artists with lower `zorder` values. Because `axvspan` was given `zorder=3` and the line defaults to `zorder=2`, the span draws over the line. Increasing the line's layer via `plt.plot(..., zorder=5)` (or lowering the span's layer to `zorder=1`) guarantees the line renders above the fill.
- **Option B (Incorrect):** Setting `alpha=0.0` makes the line completely invisible.
- **Option C (Incorrect):** Python import statements have no effect on canvas artist drawing order.
- **Option D (Incorrect):** `above` is not a valid parameter of `plt.axvspan()`.

</details>

---

### 43. Question

Two experimental trajectories intersect on a line plot:
- Line Alpha is drawn with `plt.plot(x, y1, color='blue', zorder=2.8)`
- Line Beta is drawn with `plt.plot(x, y2, color='orange', zorder=4.1)`

At the exact coordinate where the two lines cross, which line's pixels will be rendered on top?

- [ ] **A.** Line Alpha, because it was defined with a lower float value
- [ ] **B.** Line Beta, because its higher `zorder` (4.1 > 2.8) gives it drawing precedence
- [ ] **C.** The lines blend into a green pixel color automatically
- [ ] **D.** The line plotted last in code execution sequence always renders on top regardless of `zorder`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lower `zorder` artists are drawn earlier in the rendering pipeline, placing them underneath higher `zorder` artists.
- **Option B (Correct):** Matplotlib sorts all artist primitives across an axes by their `zorder` property before rasterization. Because Line Beta has `zorder = 4.1` while Line Alpha has `zorder = 2.8`, Line Beta is drawn later in the pipeline and its pixels paint over Line Alpha at the point of intersection.
- **Option C (Incorrect):** Matplotlib does not blend RGB colors of opaque lines; the top layer replaces underlying pixels.
- **Option D (Incorrect):** Execution order is only used as a tie-breaker when two artists share the exact same `zorder`.

</details>

---

### 44. Question

What is the special behavior of setting `loc="best"` (or `loc=0`) in `plt.legend()`?

- [ ] **A.** It places the legend at the exact physical center of the canvas.
- [ ] **B.** It centers the legend horizontally above the figure title.
- [ ] **C.** It docks the legend in a floating pop-up window outside the canvas.
- [ ] **D.** It automatically evaluates candidate positions and selects the location that minimizes visual overlap with plotted data lines.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The center position is explicitly requested via `loc="center"`.
- **Option B (Incorrect):** Placing the legend above the title requires manual coordinate specification via `bbox_to_anchor`.
- **Option C (Incorrect):** `loc="best"` operates entirely within the primary axes boundaries.
- **Option D (Correct):** In Matplotlib, `loc="best"` (code 0) activates an automatic layout algorithm. It creates a spatial bounding grid of candidate locations (e.g., upper right, upper left, lower left, etc.) and tests them against the bounding boxes of existing plotted lines and vertices, choosing the location that incurs the least overlap.

</details>

---

### 45. Question

Which integer code corresponds to the `"upper left"` position when passed to `plt.legend(loc=...)`?

- [ ] **A.** `0`
- [ ] **B.** `1`
- [ ] **C.** `2`
- [ ] **D.** `4`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `0` corresponds to `'best'`.
- **Option B (Incorrect):** `1` corresponds to `'upper right'`.
- **Option C (Correct):** Matplotlib maps numerical location codes as follows:
  - `0`: `'best'`
  - `1`: `'upper right'`
  - `2`: `'upper left'`
  - `3`: `'lower left'`
  - `4`: `'lower right'`
  Therefore, code `2` designates `'upper left'`.
- **Option D (Incorrect):** `4` corresponds to `'lower right'`.

</details>

---

### 46. Question

A data visualization engineer needs to place a legend completely **outside** the chart canvas to the right, aligned with the top of the plot area, so that no data traces are obscured:

```python
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.)
```

What do the coordinates `(1.05, 1.0)` in `bbox_to_anchor` represent?

- [ ] **A.** Physical pixel coordinates $(x=1.05\text{ px}, y=1.0\text{ px})$ from the bottom-left monitor corner
- [ ] **B.** Normalized axes coordinates where $(0, 0)$ is the bottom-left and $(1, 1)$ is the top-right of the plot axes; $1.05$ places the legend anchor 5% past the right axes spine, and $1.0$ aligns it with the top spine
- [ ] **C.** Data coordinate values corresponding to $x = 1.05$ and $y = 1.0$ on the plotted data scales
- [ ] **D.** Inches measured from the center of the figure window

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `bbox_to_anchor` uses transformed coordinate systems, not absolute monitor pixels.
- **Option B (Correct):** When passed a 2-tuple without specifying a custom transform, `bbox_to_anchor` defaults to normalized axes coordinates where $(0, 0)$ represents the bottom-left corner of the axes and $(1, 1)$ represents the top-right corner. Specifying $x = 1.05$ offsets the legend 5% outside the right edge of the plot area, and $y = 1.0$ aligns its top edge (`loc='upper left'`) with the top edge of the axes.
- **Option C (Incorrect):** It does not use data space coordinates; data space anchoring requires `transform=ax.transData`.
- **Option D (Incorrect):** It does not use physical inches.

</details>

---

### 47. Question

An analyst plots a secondary reference line that serves as an internal guide. The analyst wants this line visible on the canvas but wishes to **suppress it from appearing in the legend**. Which technique achieves this?

- [ ] **A.** Set `label='_nolegend_'` or prefix the label string with an underscore `_`
- [ ] **B.** Set `legend=False` in `plt.plot()`
- [ ] **C.** Set `alpha=0.99`
- [ ] **D.** Pass `visible=False` to `plt.legend()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, any artist whose `label` string begins with an underscore (`'_'`) is designated as private and automatically ignored by `plt.legend()`. Common conventions include `label='_nolegend_'` or omitting the `label` parameter entirely.
- **Option B (Incorrect):** `legend` is not a valid parameter of `plt.plot()`; passing it raises an unrecognized keyword argument error.
- **Option C (Incorrect):** Modifying `alpha` changes line opacity on the canvas, not its presence in the legend.
- **Option D (Incorrect):** `visible=False` on the legend hides the entire legend box, suppressing all labels rather than just the reference line.

</details>

---

### 48. Question

Which call completely removes the border box and background fill of a legend, displaying only the legend text and line markers floating cleanly?

- [ ] **A.** `plt.legend(border=0)`
- [ ] **B.** `plt.legend(transparent=True)`
- [ ] **C.** `plt.legend(outline='none')`
- [ ] **D.** `plt.legend(frameon=False)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `border` is not a valid parameter of `plt.legend()`.
- **Option B (Incorrect):** `transparent=True` is an argument for `plt.savefig()`, not `plt.legend()`.
- **Option C (Incorrect):** `outline` is not a recognized keyword argument for `plt.legend()`.
- **Option D (Correct):** Setting `frameon=False` in `plt.legend()` disables rendering of the legend's bounding box frame and background patch, leaving only the marker glyphs, line swatches, and label text.

</details>

---

### 49. Question

A developer notices that background gridlines are rendering on top of filled bar charts or thick data lines. Which Axes method guarantees that gridlines are always placed **below** all data artists?

- [ ] **A.** `ax.grid_depth('back')`
- [ ] **B.** `plt.grid(zorder=99)`
- [ ] **C.** `ax.send_to_back('grid')`
- [ ] **D.** `ax.set_axisbelow(True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** There is no `grid_depth` method in Matplotlib.
- **Option B (Incorrect):** Setting `zorder=99` on the grid would force it to the very front of the visual hierarchy, worsening the problem.
- **Option C (Incorrect):** There is no `send_to_back` method on Matplotlib Axes.
- **Option D (Correct):** Calling `ax.set_axisbelow(True)` forces tick lines and gridlines to have a `zorder` below all data artists (lines, patches, collections), ensuring they never cross in front of data representations.

</details>

---

### 50. Question

Consider the following statements regarding layering and legend behavior in Matplotlib:

1. Setting `zorder` accepts floating-point numbers as well as integers.
2. If two artists have the exact same `zorder`, the artist created later in the Python code is rendered on top.
3. The legend frame opacity can be tuned using `framealpha` (e.g., `plt.legend(framealpha=0.8)`).
4. Calling `plt.legend()` when no plotted artists have `label` parameters set will successfully generate an empty box without emitting any terminal warnings.

Which statements are **correct**?

- [ ] **A.** 1 and 4 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `zorder` can be any real number (e.g., `zorder=2.5`).
- **Statement 2 is correct:** When `zorder` values are identical, Matplotlib preserves procedural draw order: later artists paint over earlier artists.
- **Statement 3 is correct:** `framealpha` accepts a float in $[0.0, 1.0]$ controlling the transparency of the legend's rectangular background card.
- **Statement 4 is incorrect:** Calling `plt.legend()` with zero labeled artists emits a `UserWarning: No artists with labels found to put in legend. Note that artists whose label start with an underscore are ignored when legend() is called with no argument.`
- **Conclusion:** Statements 1, 2, and 3 are correct, making **Option C** the correct answer.

</details>

---
