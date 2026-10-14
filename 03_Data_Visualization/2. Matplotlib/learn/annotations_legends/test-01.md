# Matplotlib Annotations & Legends — Test 01: Text Fundamentals

---

### Question 1

A data scientist adds a text label to a scatter plot using:

```python
ax.text(3.5, 7.2, "Peak F1 = 0.94",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="darkgreen")
```

What does `va="bottom"` control?

- [ ] A) The text is placed so that the bottom edge of its bounding box aligns with y=7.2
- [ ] B) The text is placed so that the top edge of its bounding box aligns with y=7.2
- [ ] C) The text is placed so that its vertical center aligns with y=7.2
- [ ] D) The text is rendered below the current axes bounding box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The text is placed so that the bottom edge of its bounding box aligns with y=7.2

**Explanation:**

- **A is correct:** `va="bottom"` (vertical alignment) anchors the **bottom** edge of the text's bounding box to the specified y-coordinate (7.2). The text therefore appears **above** y=7.2, growing upward from the anchor point. This is useful for placing labels just above data points.
- **B is incorrect:** `va="top"` anchors the **top** edge to the specified y-coordinate, making the text appear below the anchor point. With `va="top"`, the text would descend from y=7.2 downward.
- **C is incorrect:** `va="center"` aligns the text's vertical midpoint to the y-coordinate. This centers the text vertically around y=7.2 rather than placing it above or below.
- **D is incorrect:** `va` is a text alignment parameter, not a spatial constraint relative to the axes box. It controls how the text string is anchored to the provided coordinates, not whether it appears inside or outside the axes bounding box.

</details>

---

### Question 2

A developer writes:

```python
ax.text(0.5, 0.5, "Accuracy: 98.3%",
        ha="left",
        va="center",
        fontsize=14,
        color="navy")
```

What visual result does `ha="left"` produce?

- [ ] A) The text is drawn so that its horizontal center aligns with x=0.5
- [ ] B) The text is drawn so that its left edge aligns with x=0.5, and the text extends to the right
- [ ] C) The text is drawn so that its right edge aligns with x=0.5, and the text extends to the left
- [ ] D) The text label is left-aligned within the axes box regardless of the x coordinate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The text is drawn so that its left edge aligns with x=0.5, and the text extends to the right

**Explanation:**

- **A is incorrect:** `ha="center"` would center the text horizontally around x=0.5. With `ha="left"`, x=0.5 is the left anchor, not the center.
- **B is correct:** `ha="left"` anchors the **left edge** of the text bounding box to the specified x-coordinate. The text string grows to the right from x=0.5. This is the default for most text rendering — the anchor point is the left baseline of the first character.
- **C is incorrect:** `ha="right"` would anchor the **right edge** to x=0.5 and make the text extend leftward. With `ha="left"`, the extension is rightward.
- **D is incorrect:** `ha` (horizontal alignment) controls how the text is anchored to the specific (x, y) coordinate pair, not a global alignment of all text within the axes frame. It does not "left-align to the axes box."

</details>

---

### Question 3

A researcher adds a text bounding box:

```python
ax.text(2.0, 5.5, "Threshold",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4",
                  fc="lightyellow",
                  ec="orange",
                  lw=2,
                  alpha=0.85))
```

What does `ec="orange"` control in the `bbox` dictionary?

- [ ] A) The fill colour (face colour) of the bounding box rectangle
- [ ] B) The colour of the border (edge) outline of the bounding box
- [ ] C) The font colour of the text string inside the box
- [ ] D) The background colour of the entire axes behind the bounding box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The colour of the border (edge) outline of the bounding box

**Explanation:**

- **A is incorrect:** `fc` ("face colour") controls the fill colour of the bounding box interior. In this code, `fc="lightyellow"` sets the box background. `ec` is a different property.
- **B is correct:** `ec` stands for "edge colour" and controls the colour of the bounding box's outline (border stroke). Setting `ec="orange"` renders the rectangular border of the box in orange. Combined with `lw=2` (line width), this produces a visible 2-point orange border around the text.
- **C is incorrect:** The text colour is set separately via the `color` (or `c`) argument to `ax.text()`, not via the `bbox` dictionary. The `bbox` dict controls only the box's visual appearance, not the text inside it.
- **D is incorrect:** The `bbox` argument creates a patch specifically around the individual text artist. It does not modify the axes background, which is controlled by `ax.set_facecolor()`.

</details>

---

### Question 4

A data analyst places a model score annotation and uses `boxstyle="round,pad=0.3"`. What does the `pad=0.3` parameter control?

- [ ] A) The border radius of the rounded corners in data units
- [ ] B) The internal padding (white space) between the text characters and the bounding box edge, expressed in points
- [ ] C) The opacity of the bounding box at 30%
- [ ] D) The line width of the bounding box border in points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The internal padding (white space) between the text characters and the bounding box edge, expressed in points

**Explanation:**

- **A is incorrect:** The rounding of the corners is controlled by the style specification in the string (the `round` part of `"round,pad=0.3"`). The `pad` value does not set the corner radius.
- **B is correct:** In Matplotlib's `boxstyle` string, `pad` specifies the padding between the text content and the enclosing box boundary. A value of `0.3` adds 0.3 units (in the text's coordinate space, typically proportional to font size) of space on all sides between the text characters and the box edge. A larger `pad` value creates a more spacious box; a smaller value produces a tighter fit.
- **C is incorrect:** Opacity of the bounding box is controlled by `alpha` in the `bbox` dictionary (here it is not shown, defaulting to opaque). `pad` has nothing to do with transparency.
- **D is incorrect:** Line width is controlled by `lw` (or `linewidth`) in the `bbox` dictionary. `pad` is purely about interior spacing.

</details>

---

### Question 5

A machine learning engineer writes the following to annotate a training loss curve:

```python
fig, ax = plt.subplots()
ax.plot(epochs, train_loss, label="Train")
ax.text(epochs[-1] + 0.5, train_loss[-1],
        f"Final: {train_loss[-1]:.3f}",
        va="center",
        ha="left",
        fontsize=9)
```

Why is `epochs[-1] + 0.5` used as the x-coordinate rather than `epochs[-1]`?

- [ ] A) Adding 0.5 converts the epoch index from zero-based to one-based indexing
- [ ] B) It places the text label 0.5 data units to the right of the last data point, preventing the text from overlapping with the plotted marker
- [ ] C) It shifts the text into axes-fraction coordinates where 0.5 represents the horizontal center
- [ ] D) It accounts for the fact that `ha="left"` shifts x by half a character width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It places the text label 0.5 data units to the right of the last data point, preventing the text from overlapping with the plotted marker

**Explanation:**

- **A is incorrect:** Zero-to-one-based index conversion is irrelevant here. `epochs[-1]` is the last epoch's value. Adding 0.5 is a spatial offset for label positioning, not an index correction.
- **B is correct:** Since `ha="left"` anchors the left edge of the text at the x-coordinate, placing the text at exactly `epochs[-1]` would start the text label directly on top of the last data point. Adding 0.5 data units of horizontal offset creates a visual gap between the marker and the beginning of the text string, improving readability. This is a common axis-end label pattern in line chart visualizations.
- **C is incorrect:** `ax.text` uses data coordinates by default (no `transform` is specified, so `transform=ax.transData` is assumed). A value of `epochs[-1] + 0.5` is a data-space coordinate, not a fraction of the axes width. Axes-fraction coordinates range from 0 to 1 and require `transform=ax.transAxes`.
- **D is incorrect:** `ha="left"` anchors the text's left edge to the given x-coordinate; it does not add any implicit character-width offset to the coordinate itself. The positioning is entirely determined by the explicit x-coordinate provided.

</details>

---

### Question 6

A researcher applies two style properties to a text label and gets an unexpected result:

```python
ax.text(5, 8, "AUC = 0.87", fontweight="bold", fontstyle="italic")
```

Which statement correctly describes the rendered text?

- [ ] A) The text is rendered in bold only because `fontstyle` overrides `fontweight`
- [ ] B) The text is rendered in italic only because bold and italic cannot be combined in Matplotlib
- [ ] C) The text is rendered in bold and italic simultaneously — both properties are applied independently
- [ ] D) A `ValueError` is raised because `fontweight` and `fontstyle` cannot be passed together to `ax.text`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The text is rendered in bold and italic simultaneously — both properties are applied independently

**Explanation:**

- **A is incorrect:** `fontweight` and `fontstyle` are independent text properties in Matplotlib. Neither overrides the other. `fontstyle="italic"` affects the slant of the glyphs; `fontweight="bold"` affects the stroke weight.
- **B is incorrect:** Bold and italic are orthogonal typographic axes supported by most font families. When the current font family has a bold-italic variant, Matplotlib renders the text with both properties. Even when the exact bold-italic variant is unavailable, Matplotlib synthesizes it from the bold and italic variants separately.
- **C is correct:** Matplotlib's `Text` artist stores font properties in a `FontProperties` object that maintains `weight`, `style`, `size`, and `family` independently. Setting both `fontweight="bold"` and `fontstyle="italic"` on the same call applies both properties simultaneously, producing bold-italic text. The rendering depends on the active backend and available font variants.
- **D is incorrect:** There is no restriction against using `fontweight` and `fontstyle` together. They are well-documented, accepted keyword arguments for `ax.text()` and all other Matplotlib text functions.

</details>

---

### Question 7

A data scientist uses `alpha=0.5` on both the text colour and the bounding box:

```python
ax.text(4, 6, "Validation",
        color="blue",
        alpha=0.5,
        bbox=dict(boxstyle="square", fc="lightblue", alpha=0.3))
```

When `alpha` is specified both in the top-level `ax.text` call and inside the `bbox` dict, which elements are affected by each alpha?

- [ ] A) The top-level `alpha=0.5` controls the bounding box; the `bbox` dict `alpha=0.3` controls the text characters
- [ ] B) The top-level `alpha=0.5` controls the entire text artist (including both the text glyphs and the bounding box); the `bbox` dict `alpha=0.3` overrides the box alpha specifically
- [ ] C) The two alpha values are averaged: final alpha for everything is (0.5 + 0.3) / 2 = 0.4
- [ ] D) Only the `bbox` dict `alpha=0.3` takes effect; the top-level `alpha` argument is ignored for text artists

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The top-level `alpha=0.5` controls the entire text artist (including both the text glyphs and the bounding box); the `bbox` dict `alpha=0.3` overrides the box alpha specifically

**Explanation:**

- **A is incorrect:** The top-level `alpha` parameter applies to the text artist as a whole (both glyphs and patch), not just the box. The `bbox` dict's `alpha` is a more specific override for the box patch.
- **B is correct:** When `alpha=0.5` is passed to `ax.text()`, it sets the artist-level alpha, which applies to the text glyphs and, by default, to the background patch. However, the `bbox` dictionary creates a `FancyBboxPatch` for the background, and its `alpha=0.3` sets the patch-specific alpha, overriding the artist-level value for the box specifically. The result is: text glyphs at 50% opacity (from top-level alpha) and the bounding box patch at 30% opacity (from the bbox dict).
- **C is incorrect:** Matplotlib does not average alpha values. Each element uses its most specific alpha specification — the text glyphs use 0.5 and the box uses 0.3 independently.
- **D is incorrect:** The top-level `alpha` argument is valid and actively applied to the text glyphs. It is not ignored. The `bbox` dict's alpha supplements it for the box only.

</details>

---

### Question 8

A developer is building a publication figure and needs to place an "R² = 0.97" label at the top-right corner of the axes, ensuring it stays in the corner regardless of data range changes. Without using `transform`, she writes:

```python
ax.text(x_max * 0.95, y_max * 0.95, "R² = 0.97", ha="right", va="top")
```

A colleague says this approach is fragile. Why?

- [ ] A) `ha="right"` is not compatible with data coordinates and causes clipping
- [ ] B) If the data range changes (axis autoscaling), `x_max` and `y_max` change too, potentially moving the label away from the top-right corner
- [ ] C) `ax.text` does not accept percentage-based coordinates like `0.95 * x_max`
- [ ] D) Text placed in data coordinates is rendered below all plot lines and cannot be seen

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) If the data range changes (axis autoscaling), `x_max` and `y_max` change too, potentially moving the label away from the top-right corner

**Explanation:**

- **A is incorrect:** `ha="right"` is fully compatible with data coordinates. It simply anchors the right edge of the text to the specified x-coordinate, regardless of the coordinate system used.
- **B is correct:** When text is placed using data coordinates that are derived from the current data range (e.g., `x_max * 0.95`), any change to the plotted data or axis limits causes `x_max` to change, shifting the label position. If additional data is added later, the axis autoscales and the `0.95 × x_max` label may no longer appear in the top-right corner. The robust solution is `transform=ax.transAxes` with coordinates `(0.95, 0.95)`, which always represents 95% of the axes width and height regardless of data range.
- **C is incorrect:** `ax.text` accepts any numerical x and y coordinates, including expressions like `x_max * 0.95`. There is no restriction on computed or fractional values.
- **D is incorrect:** Text artists are rendered on top of most plot artists (lines, patches) by default. They are not rendered "below" the plot lines unless explicitly given a very low `zorder`.

</details>

---

### Question 9

A researcher places text with `ha="center"` and `va="center"` at exact data coordinates:

```python
ax.text(5.0, 3.5, "Decision Boundary",
        ha="center", va="center",
        fontsize=12, rotation=45)
```

What effect does `rotation=45` have on the text rendering?

- [ ] A) The text is rendered at a 45° angle, rotated counter-clockwise around the anchor point (5.0, 3.5)
- [ ] B) The text is rendered horizontally but shifted 45 pixels diagonally from (5.0, 3.5)
- [ ] C) The text bounding box is rotated 45° but the text characters remain horizontal inside the box
- [ ] D) A `ValueError` is raised because `rotation` is not a valid argument for `ax.text`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The text is rendered at a 45° angle, rotated counter-clockwise around the anchor point (5.0, 3.5)

**Explanation:**

- **A is correct:** `rotation` in `ax.text()` specifies the angle in degrees (counter-clockwise from horizontal) at which the text string is rendered. With `rotation=45`, the text "Decision Boundary" is tilted 45° counter-clockwise, appearing at a diagonal angle. The rotation pivot is the anchor point defined by `ha` and `va`, which is `(5.0, 3.5)` since both are set to `"center"`.
- **B is incorrect:** `rotation` is not a pixel-offset parameter. It controls the angular orientation of the rendered text, not its position. The anchor point remains at (5.0, 3.5).
- **C is incorrect:** The entire text string, including individual characters, is rotated as a single unit. There is no mechanism in Matplotlib's standard `ax.text` that rotates the bounding box independently of the characters.
- **D is incorrect:** `rotation` is a fully documented and commonly used keyword argument for `ax.text()`. It accepts numeric angles (in degrees) or the strings `"horizontal"` (=0°) and `"vertical"` (=90°).

</details>

---

### Question 10

A data analyst places five text labels in a loop and finds that two of them overlap a dense cluster of scatter points. Which `ax.text` properties should she adjust to improve legibility without repositioning the text?

- [ ] A) Increase `fontsize` and set `fontweight="bold"` to make the text overpower the overlapping markers
- [ ] B) Add `bbox=dict(fc="white", ec="none", alpha=0.7)` to each text call, creating a white "knockout" background behind each label that partially obscures the overlapping markers
- [ ] C) Set `zorder=0` on each text call to render the labels behind the scatter markers
- [ ] D) Set `clip_on=False` on each text call to allow text to extend beyond the axes bounding box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add `bbox=dict(fc="white", ec="none", alpha=0.7)` to each text call, creating a white "knockout" background behind each label that partially obscures the overlapping markers

**Explanation:**

- **A is incorrect:** Making text larger and bolder makes it more prominent but does not prevent the underlying scatter points from visually competing with the text. The text and markers still overlap; the text is just easier to read but visually busier.
- **B is correct:** A semi-transparent white bounding box (`fc="white", alpha=0.7`) acts as a "knockout" or "halo" effect. It renders a white patch behind the text that partially covers overlapping markers, creating a clean visual separation between the label and the data points beneath it. Setting `ec="none"` removes the border so the patch is invisible except as a white background. This is a standard technique in publication visualization for legible in-chart annotations.
- **C is incorrect:** Setting `zorder=0` renders the text behind the scatter markers, making the overlap worse (text would be hidden). To keep text on top, a higher `zorder` is needed, not a lower one.
- **D is incorrect:** `clip_on=False` allows text to render outside the axes bounding box (useful for labels near the boundary). It does not improve legibility when the text overlaps with data inside the axes. It does not create any background knockout effect.

</details>
