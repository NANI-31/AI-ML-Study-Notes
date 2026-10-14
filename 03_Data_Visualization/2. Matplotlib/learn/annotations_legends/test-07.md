# Matplotlib Annotations & Legends — Test 07: Legend Construction & Internal Positioning

---

### Question 61

A data scientist plots two model performance curves but forgets to add `label=` arguments:

```python
ax.plot(epochs, train_acc, color="steelblue")
ax.plot(epochs, val_acc, color="coral")
ax.legend()
```

What warning or behavior does Matplotlib produce?

- [ ] A) The legend is displayed with automatically generated labels "Line 1" and "Line 2"
- [ ] B) Matplotlib raises a `LegendError` and the plot fails to render
- [ ] C) The legend is rendered but empty — no artists appear — and Matplotlib emits a `UserWarning: No artists with labels found to put in legend`
- [ ] D) The legend is rendered with the line colours shown as swatches but no text labels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The legend is rendered but empty — no artists appear — and Matplotlib emits a `UserWarning: No artists with labels found to put in legend`

**Explanation:**

- **A is incorrect:** Matplotlib does not auto-generate labels. Artists without explicit `label` arguments receive a default label starting with `_` (specifically `"_child0"`, `"_child1"`, etc.), which are excluded from legend collection by convention.
- **B is incorrect:** No `LegendError` exists in Matplotlib. Missing labels result in a warning, not an exception. The plot renders successfully.
- **C is correct:** When `ax.legend()` is called and no artists with valid (non-underscore) labels are found, Matplotlib creates an empty legend box and emits `UserWarning: No artists with labels found to put in legend. If you wish to add a label, pass a non-empty string to the label keyword.` This is a common debugging clue for forgetting the `label=` argument in plot calls.
- **D is incorrect:** Matplotlib does not auto-populate swatches for unlabeled artists. The legend box is empty — no colour swatches or text entries appear.

</details>

---

### Question 62

A researcher correctly labels her model curves and calls the legend:

```python
ax.plot(x, train_f1, label="Train F1", color="blue")
ax.plot(x, val_f1, label="Val F1", color="orange")
ax.legend(loc="lower right")
```

What does `loc="lower right"` specify?

- [ ] A) The legend is anchored 80% from the left and 20% from the bottom of the figure
- [ ] B) The legend box is placed in the lower-right corner of the axes area — Matplotlib positions it with some margin from the axes edges
- [ ] C) The legend font size is set to the right-aligned small size preset
- [ ] D) The legend label text is right-aligned inside the legend box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The legend box is placed in the lower-right corner of the axes area — Matplotlib positions it with some margin from the axes edges

**Explanation:**

- **A is incorrect:** `loc="lower right"` specifies a named position within the axes area, not explicit percentage coordinates. Explicit fraction coordinates would require `bbox_to_anchor`.
- **B is correct:** `loc` specifies the placement zone of the legend box within the axes. `"lower right"` places the legend in the lower-right region of the axes interior, with automatic internal margin to prevent the legend from touching the axes spines. Matplotlib uses a fixed offset from the corner and may adjust slightly to avoid overlapping plotted data when `loc="best"` is used. Named locations include `"upper right"` (1), `"upper left"` (2), `"lower left"` (3), `"lower right"` (4), `"right"` (5), `"center left"` (6), `"center right"` (7), `"lower center"` (8), `"upper center"` (9), `"center"` (10).
- **C is incorrect:** `loc` has no relationship to font size. Font size is set via `fontsize=` in `ax.legend()`.
- **D is incorrect:** Text alignment within legend entries is controlled separately (if at all). `loc` only controls the position of the entire legend box.

</details>

---

### Question 63

A developer uses numeric location codes instead of string names for the legend:

```python
ax.legend(loc=1)
ax.legend(loc=4)
```

Which positions do `loc=1` and `loc=4` correspond to?

- [ ] A) `loc=1` → lower left; `loc=4` → upper right
- [ ] B) `loc=1` → upper right; `loc=4` → lower right
- [ ] C) `loc=1` → center; `loc=4` → lower left
- [ ] D) `loc=1` → upper left; `loc=4` → center right

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `loc=1` → upper right; `loc=4` → lower right

**Explanation:**

- **A is incorrect:** `loc=1` maps to `"upper right"` and `loc=4` maps to `"lower right"`, not the positions described.
- **B is correct:** Matplotlib's numeric location codes follow this mapping: 0 = `"best"`, 1 = `"upper right"`, 2 = `"upper left"`, 3 = `"lower left"`, 4 = `"lower right"`, 5 = `"right"`, 6 = `"center left"`, 7 = `"center right"`, 8 = `"lower center"`, 9 = `"upper center"`, 10 = `"center"`. So `loc=1` places the legend in the upper-right and `loc=4` places it in the lower-right.
- **C is incorrect:** `loc=1` is upper right (not center) and `loc=4` is lower right (not lower left).
- **D is incorrect:** `loc=1` is upper right (not upper left) and `loc=4` is lower right (not center right).

</details>

---

### Question 64

A machine learning engineer uses `loc="best"` for a legend:

```python
ax.legend(loc="best")
```

What algorithm does Matplotlib use to determine the "best" position?

- [ ] A) It places the legend at the position with the most visible empty space relative to the plotted data, minimizing overlap with data artists
- [ ] B) It places the legend at a fixed default position in the upper right, ignoring data layout
- [ ] C) It performs a full visual salience analysis using image processing to find the region with lowest contrast
- [ ] D) It samples 20 random positions and selects the one with the highest distance from all data points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It places the legend at the position with the most visible empty space relative to the plotted data, minimizing overlap with data artists

**Explanation:**

- **A is correct:** `loc="best"` triggers Matplotlib's legend placement algorithm, which evaluates all 9 standard interior positions (excluding `"best"` itself) and selects the one that minimizes the overlap between the legend box and the plotted artists (lines, scatter points, bars, patches). It counts how many data points fall within each potential legend position's bounding box and chooses the position with the minimum overlap. This is a heuristic algorithm — it works well for typical plots but can fail for very cluttered or unusually distributed data.
- **B is incorrect:** `"upper right"` (`loc=1`) is the hardcoded default when no `loc` is specified. `loc="best"` explicitly activates the placement algorithm.
- **C is incorrect:** Matplotlib does not use image processing or visual salience analysis for legend placement. The algorithm is purely geometric, based on counting data point overlaps.
- **D is incorrect:** The algorithm does not use random sampling. It deterministically evaluates all standard named positions (up to 10) and picks the one with minimum data overlap.

</details>

---

### Question 65

A data analyst plots three curves and explicitly passes handles and labels to `ax.legend`:

```python
line1, = ax.plot(x, y1, color="blue")
line2, = ax.plot(x, y2, color="green")
line3, = ax.plot(x, y3, color="red")

ax.legend(handles=[line1, line3],
          labels=["Model A", "Model C"])
```

What appears in the legend?

- [ ] A) All three lines with their default labels (overriding the explicit `handles` argument)
- [ ] B) Only the two lines `line1` and `line3` with labels "Model A" and "Model C" — `line2` is excluded
- [ ] C) An error — if `handles` is provided, `labels` cannot also be specified
- [ ] D) All three lines with "Model A", automatically generated label for `line2`, and "Model C"

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Only the two lines `line1` and `line3` with labels "Model A" and "Model C" — `line2` is excluded

**Explanation:**

- **A is incorrect:** When `handles` is explicitly specified, Matplotlib uses exactly those artists for the legend. The auto-collection of all labeled artists is bypassed.
- **B is correct:** `ax.legend(handles=[list], labels=[list])` creates legend entries exactly for the artists in `handles`, paired with the strings in `labels`. Only `line1` (blue) and `line3` (red) are in the `handles` list, so the legend shows exactly two entries: "Model A" → blue swatch, "Model C" → red swatch. `line2` (green) is intentionally omitted.
- **C is incorrect:** Providing both `handles` and `labels` is the correct and complete way to use explicit legend construction. Both arguments can and should be used together.
- **D is incorrect:** The legend does not auto-generate entries for artists not in the `handles` list. The `handles` list fully defines which artists are included.

</details>

---

### Question 66

A researcher's legend is placed at `loc="upper left"` but overlaps a dense cluster of data points in the upper-left corner. Without using `bbox_to_anchor`, which `loc` option would she try next to avoid overlap in the lower half of the chart?

- [ ] A) `loc="lower center"`
- [ ] B) `loc="upper right"`
- [ ] C) `loc="best"` — lets Matplotlib select the least-overlapping position automatically
- [ ] D) Both A and C are valid approaches to avoid upper-left overlap

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Both A and C are valid approaches to avoid upper-left overlap

**Explanation:**

- **A is partially correct:** `loc="lower center"` explicitly moves the legend to the lower-center interior of the axes, which would avoid the upper-left cluster. This is a direct, predictable repositioning.
- **B is incorrect:** `loc="upper right"` moves the legend to the upper-right area. If the data cluster is only in the upper-left, the upper-right might be clear. But this is not guaranteed to avoid overlap — the question specifies "avoid overlap in the lower half," suggesting a lower position is preferred.
- **C is partially correct:** `loc="best"` asks Matplotlib's algorithm to find the minimum-overlap position among all standard locations, which could select a lower position if data is sparse there.
- **D is correct:** Both `loc="lower center"` (a specific, predictable move to the bottom) and `loc="best"` (algorithmic optimization) are valid strategies to avoid the upper-left overlap. The researcher could use either: `"lower center"` for explicit control, or `"best"` to let Matplotlib choose among all interior positions. Both approaches address the overlap concern.

</details>

---

### Question 67

A developer uses the following legend call and is surprised that the legend has an extra "None" entry at the top:

```python
ax.plot(x, y, color="steelblue", label="Accuracy")
ax.axhline(y=0.5, color="gray")
ax.legend()
```

Why does the legend show correctly for "Accuracy" but might include an unexpected entry for the `axhline`?

- [ ] A) `axhline` automatically assigns `label="Horizontal Line"` which appears in the legend
- [ ] B) Without a `label` argument, `axhline` generates a label starting with `_` which is excluded — no unexpected entry appears
- [ ] C) `axhline` assigns `label=None` which appears in the legend as the string "None"
- [ ] D) `axhline` is always added twice to the axes, producing a duplicate entry

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Without a `label` argument, `axhline` generates a label starting with `_` which is excluded — no unexpected entry appears

**Explanation:**

- **A is incorrect:** Matplotlib does not auto-generate descriptive labels like `"Horizontal Line"` for unlabeled artists. The auto-assigned default label starts with `_`, not a descriptive string.
- **B is correct:** When `axhline` is called without `label=`, Matplotlib internally assigns a default label like `"_line0"` (underscore-prefixed). By convention, artists whose labels begin with `_` are excluded from the auto-legend collection in `ax.legend()`. The legend therefore shows only the "Accuracy" entry from the `ax.plot` call and correctly omits the unlabeled `axhline`. The developer's concern about an "unexpected entry" is unfounded — the legend is clean.
- **C is incorrect:** Matplotlib assigns an underscore-prefixed string as the default label, not Python's `None` object. Even if `None` were the label, Python's `ax.legend()` filtering would skip it. The string "None" does not appear as a legend entry from unlabeled artists.
- **D is incorrect:** `axhline` is added exactly once to the axes. It does not create duplicate artists.

</details>

---

### Question 68

A data scientist has a legend that is too tall due to many entries. She wants to reduce the vertical space by arranging entries in two columns. Which call achieves this?

- [ ] A) `ax.legend(columns=2)`
- [ ] B) `ax.legend(ncol=2)`
- [ ] C) `ax.legend(layout="2col")`
- [ ] D) `ax.legend(orientation="horizontal")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.legend(ncol=2)`

**Explanation:**

- **A is incorrect:** `columns=` is not a valid keyword argument for `ax.legend`. The correct parameter is `ncol` (or `ncols` in Matplotlib ≥ 3.6).
- **B is correct:** `ncol=N` (and its alias `ncols=N` in newer Matplotlib versions) specifies the number of columns for the legend layout. `ncol=2` arranges legend entries into two columns, reducing the legend's vertical height at the cost of increased horizontal width. This is the standard approach for multi-column legend layouts.
- **C is incorrect:** `layout="2col"` is not a valid `ax.legend` parameter. Layout string arguments are used in other Matplotlib contexts (e.g., `constrained_layout`), not legend configuration.
- **D is incorrect:** `orientation="horizontal"` is not a valid `ax.legend` parameter. The `ncol` approach achieves a horizontal spread of entries. A fully horizontal legend with all entries in a single row would require `ncol=len(handles)`.

</details>

---

### Question 69

A researcher passes `frameon=False` to `ax.legend()`. What visual change occurs?

- [ ] A) The legend box border and background fill are removed — only the marker swatches and text labels float on the plot
- [ ] B) The legend text becomes transparent
- [ ] C) The legend is hidden entirely and must be recalled with `ax.get_legend().set_visible(True)`
- [ ] D) The legend frame changes from a rectangle to a rounded capsule shape

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The legend box border and background fill are removed — only the marker swatches and text labels float on the plot

**Explanation:**

- **A is correct:** `frameon=False` removes the legend frame — specifically both the rectangular border stroke and the white background fill patch. The legend entries (colour swatches and text labels) remain visible but appear to "float" directly over the plot with no surrounding box. This produces a minimal, clean legend appearance commonly used in publication figures where the legend box would clutter the visualization.
- **B is incorrect:** `frameon=False` does not affect text transparency. The label text remains fully opaque.
- **C is incorrect:** The legend is still visible when `frameon=False`. The legend box frame (border + background) is removed, but the content (text and handle markers) is still rendered. The legend artist itself is not hidden.
- **D is incorrect:** Rounded legend corners are created by `fancybox=True`, which rounds the rectangle corners. `frameon=False` removes the frame entirely rather than changing its shape.

</details>

---

### Question 70

A machine learning engineer wants to set a custom legend title and title font size. Which code correctly achieves this?

- [ ] A) `ax.legend(title="Models", title_fontsize=13)`
- [ ] B) `ax.legend(header="Models", header_fontsize=13)`
- [ ] C) `legend = ax.legend(); legend.set_title("Models", prop={"size": 13})`
- [ ] D) Both A and C are valid approaches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Both A and C are valid approaches

**Explanation:**

- **A is partially correct:** `ax.legend(title="Models", title_fontsize=13)` is a direct, one-call approach. The `title` argument adds a bold title above the legend entries, and `title_fontsize` sets the title's font size. Both are valid keyword arguments for `ax.legend()`.
- **B is incorrect:** `header=` and `header_fontsize=` are not valid `ax.legend()` parameters. These names do not exist in Matplotlib's legend API.
- **C is partially correct:** `legend.set_title("Models", prop={"size": 13})` is also a valid approach. After creating the legend, `set_title()` is called on the returned `Legend` object. The `prop` argument accepts a font properties dictionary. Both title text and font size are set correctly.
- **D is correct:** Both Option A (direct keyword arguments) and Option C (post-creation method call on the legend object) are valid and equivalent approaches to setting a legend title and title font size. The direct keyword approach (A) is more concise; the method call approach (C) is useful when modifying an existing legend.

</details>
