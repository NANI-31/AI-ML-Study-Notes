# Matplotlib Annotations & Legends — Test 04: Arrow Styles & Geometry

---

### Question 31

A developer annotates a convergence point on a gradient curve:

```python
ax.annotate("Converged",
            xy=(80, 0.01),
            xytext=(60, 0.08),
            arrowprops=dict(arrowstyle="->",
                            color="darkred",
                            lw=1.5))
```

What does `arrowstyle="->"` produce at the arrow tip?

- [ ] A) A filled triangle arrowhead at the tail (xytext side) pointing away from the target
- [ ] B) A simple open arrowhead (V-shape, not filled) at the tip pointing toward `xy=(80, 0.01)`
- [ ] C) A bidirectional arrowhead at both the tip and the tail
- [ ] D) A filled circle marker at the annotation target point

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A simple open arrowhead (V-shape, not filled) at the tip pointing toward `xy=(80, 0.01)`

**Explanation:**

- **A is incorrect:** `"->"` places the arrowhead at the tip end (the `xy` target), not at the tail. The arrowhead points toward `xy`, not away from it.
- **B is correct:** `arrowstyle="->"` is Matplotlib's standard forward arrow style. It produces a simple open V-shaped arrowhead at the arrow tip — the end closest to `xy=(80, 0.01)`. The arrow shaft runs from `xytext=(60, 0.08)` and the V-shaped head points at the convergence point. "Open" means the arrowhead is two line strokes forming a chevron, not a filled polygon.
- **C is incorrect:** A bidirectional arrowhead requires `arrowstyle="<->"`. The single `"->"` places a head only at the forward (xy) end.
- **D is incorrect:** A circle marker at the target is a different visualization element (added via `ax.scatter` or `ax.plot` with `marker="o"`). `arrowstyle="->"` does not produce marker glyphs.

</details>

---

### Question 32

A researcher uses a curved arrow to annotate a saddle point:

```python
ax.annotate("Saddle Point",
            xy=(0.0, 0.0),
            xytext=(-1.5, 1.0),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=0.3"))
```

What does `connectionstyle="arc3,rad=0.3"` control?

- [ ] A) The arrowhead angle, set to 0.3 radians (≈17°)
- [ ] B) The curvature of the arrow path — `rad=0.3` produces a gently curved (bowed) arc connecting `xytext` to `xy`
- [ ] C) The radius of a circular arc in data units centered at the midpoint between `xytext` and `xy`
- [ ] D) The number of segments used to approximate the curved arrow path (0.3 × 10 = 3 segments)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The curvature of the arrow path — `rad=0.3` produces a gently curved (bowed) arc connecting `xytext` to `xy`

**Explanation:**

- **A is incorrect:** `rad` in `arc3` is not an angle measurement. The arrowhead angle is a property of `arrowstyle`, not `connectionstyle`.
- **B is correct:** `connectionstyle="arc3,rad=R"` specifies a cubic Bezier arc connecting the annotation tail to the arrow tip. The `rad` parameter controls how much the arc bows sideways — `rad=0` produces a straight line, positive `rad` bows the arc to one side, and negative `rad` bows it to the opposite side. A value of 0.3 produces a noticeably curved arrow, while 0.1 would be nearly straight. This is commonly used to annotate clustered points where a straight arrow would overlap other elements.
- **C is incorrect:** `rad` is a dimensionless curvature parameter for the Bezier control points, not a data-unit radius. The arc shape is computed in display space.
- **D is incorrect:** `rad` is a continuous curvature parameter, not a segment count. The arrow path is always a smooth cubic Bezier curve regardless of `rad` magnitude.

</details>

---

### Question 33

A machine learning engineer compares four arrow styles and needs one with a filled triangular arrowhead (solid, not open). Which `arrowstyle` should she use?

- [ ] A) `arrowstyle="->"`
- [ ] B) `arrowstyle="-|>"`
- [ ] C) `arrowstyle="fancy"`
- [ ] D) `arrowstyle="simple"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `arrowstyle="-|>"`

**Explanation:**

- **A is incorrect:** `"->"` produces an open, V-shaped (unfilled) arrowhead. The two lines form a chevron without a filled interior.
- **B is correct:** `"-|>"` produces a line (`-`) ending in a filled triangular arrowhead (`|>`). The `|>` token indicates a triangle filled with the arrow colour. This gives a solid, filled arrowhead that is visually prominent and commonly used in publication-quality figures for clean annotation arrows.
- **C is incorrect:** `"fancy"` produces a fancy-shaped arrow with a curved body and wide filled head — it is more decorative and less commonly used for precise data annotation. It also requires additional `mutation_scale` tuning.
- **D is incorrect:** `"simple"` produces a filled arrowhead similar to `"fancy"` but with a straight body of adjustable width. It is thicker than a typical annotation arrow and may be too visually heavy for most data annotation use cases.

</details>

---

### Question 34

A data scientist uses `shrink=0.1` in her annotation arrow:

```python
ax.annotate("Precision Drop",
            xy=(15, 0.72),
            xytext=(25, 0.90),
            arrowprops=dict(arrowstyle="->",
                            shrink=0.1))
```

What does `shrink=0.1` do to the rendered arrow?

- [ ] A) Reduces the arrowhead size to 10% of the default
- [ ] B) Moves the arrowhead back 10% of the arrow's total length from both the tail and the tip, creating a gap between the annotation target and the arrow endpoints
- [ ] C) Reduces the arrow line width by 10%
- [ ] D) Reduces the arrow's opacity to 10%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Moves the arrowhead back 10% of the arrow's total length from both the tail and the tip, creating a gap between the annotation target and the arrow endpoints

**Explanation:**

- **A is incorrect:** Arrowhead size is controlled by `mutation_scale` (for `arrowstyle` arrows) or `headwidth`/`headlength` (for `FancyArrow`-style dicts). `shrink` does not modify arrowhead dimensions.
- **B is correct:** `shrink` is a fractional parameter that shortens both ends of the arrow by the specified fraction of the total arrow length. A value of 0.1 pulls the tail back 10% from `xytext` and pulls the tip 10% away from `xy`. This creates a small visual gap at both ends, which is useful when the annotation target is a marker or patch and you want the arrow tip to stop just short of the center rather than overlapping the marker symbol.
- **C is incorrect:** Line width is controlled by `lw` (or `linewidth`) inside the `arrowprops` dictionary. `shrink` has no effect on stroke thickness.
- **D is incorrect:** Opacity is controlled by `alpha` inside `arrowprops`. `shrink` is a geometric shortening parameter.

</details>

---

### Question 35

A researcher annotates a ROC threshold with a two-headed arrow between two operating points:

```python
ax.annotate("",
            xy=(0.2, 0.85),
            xytext=(0.6, 0.55),
            arrowprops=dict(arrowstyle="<->",
                            color="purple",
                            lw=2))
```

Why is the first argument (the annotation text) an empty string `""`?

- [ ] A) Empty string is required by `ax.annotate` when using bidirectional arrow styles
- [ ] B) The annotation is used purely to draw the arrow connector between two points without rendering any text label
- [ ] C) Empty string triggers a special "line segment" mode that bypasses the arrow renderer
- [ ] D) The text will be automatically populated from the `xy` coordinate values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The annotation is used purely to draw the arrow connector between two points without rendering any text label

**Explanation:**

- **A is incorrect:** There is no requirement to use an empty string with bidirectional arrows. The `arrowstyle` choice is independent of the annotation text content.
- **B is correct:** `ax.annotate` always requires a text string as its first argument. Passing `""` (empty string) renders no visible text, making the call function solely as an arrow or line connector between two points. This pattern is commonly used to draw distance indicators, comparison brackets, or span arrows between data features without attaching any label text.
- **C is incorrect:** Empty string does not activate any special "line segment" mode. The empty text is processed normally as a `Text` artist with zero characters; the arrow rendering is controlled by `arrowprops` regardless.
- **D is incorrect:** Matplotlib does not automatically populate annotation text from coordinate values. The text is always exactly what is passed as the first argument.

</details>

---

### Question 36

A developer uses the classical `FancyArrow` style (without `arrowstyle`) to annotate a data point:

```python
ax.annotate("Critical Point",
            xy=(5.2, 3.8),
            xytext=(7.0, 5.5),
            arrowprops=dict(
                width=2,
                headwidth=10,
                headlength=8,
                fc="steelblue",
                ec="darkblue",
                shrink=0.05
            ))
```

What does `headwidth=10` control?

- [ ] A) The total length of the arrowhead triangle (from base to tip)
- [ ] B) The width of the arrowhead perpendicular to the arrow shaft direction
- [ ] C) The number of pixels in the arrowhead's widest dimension
- [ ] D) The ratio of arrowhead width to shaft width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The width of the arrowhead perpendicular to the arrow shaft direction

**Explanation:**

- **A is incorrect:** The length of the arrowhead from its base to its tip is controlled by `headlength` (here `headlength=8`), not `headwidth`.
- **B is correct:** In the `FancyArrow`-style `arrowprops` dictionary (the older API that does not use `arrowstyle`), `headwidth` specifies the total width of the arrowhead measured perpendicular to the direction of the arrow shaft. It determines how wide the triangular head appears. A `headwidth=10` with `width=2` produces a wide arrowhead (10 units wide) relative to a narrow shaft (2 units wide), creating the classic wide-headed annotation arrow. Units are in points.
- **C is incorrect:** `headwidth` is measured in typographic points, not screen pixels. The number of pixels depends on the figure DPI and rendering backend.
- **D is incorrect:** `headwidth` is an absolute dimension, not a ratio. The shaft width is independently set by `width`.

</details>

---

### Question 37

A researcher uses `connectionstyle="angle3,angleA=90,angleB=0"` for an annotation:

```python
ax.annotate("Corner Turn",
            xy=(10, 5),
            xytext=(2, 9),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=90,angleB=0"))
```

What path does `angle3` produce between `xytext` and `xy`?

- [ ] A) A straight line at 90° from xytext and 0° from xy (impossible — they would never meet)
- [ ] B) A path that leaves `xytext` at 90° (vertical) and arrives at `xy` at 0° (horizontal), connected by a single smooth quadratic Bezier curve
- [ ] C) A right-angle path with two straight segments meeting at a corner
- [ ] D) An arc of a circle with the angle between the two tangents equal to 90°

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A path that leaves `xytext` at 90° (vertical) and arrives at `xy` at 0° (horizontal), connected by a single smooth quadratic Bezier curve

**Explanation:**

- **A is incorrect:** `angle3` does produce a valid connecting path. While two straight lines at fixed angles would only meet at a single point, `angle3` uses a quadratic Bezier curve as the connection, which smoothly transitions between the two required tangent directions.
- **B is correct:** The `angle3` connection style in Matplotlib creates a **quadratic Bezier curve** where `angleA` specifies the departure angle from the tail (`xytext`) and `angleB` specifies the arrival angle at the tip (`xy`). With `angleA=90`, the arrow leaves `xytext` going vertically upward; with `angleB=0`, it arrives at `xy` going horizontally. The quadratic Bezier smoothly interpolates between these two tangent directions, producing a gentle curved elbow.
- **C is incorrect:** A right-angle path with two straight segments would be `connectionstyle="angle,angleA=90,angleB=0"` (without the `3`). The `3` in `angle3` specifies the cubic or quadratic Bezier interpolation mode.
- **D is incorrect:** A circular arc connector uses `connectionstyle="arc"` or `connectionstyle="arc3"`, not `"angle3"`. `angle3` is specifically the Bezier-based angular connection style.

</details>

---

### Question 38

A data analyst places a double-ended comparison arrow and observes the arrowheads are very small relative to the line. She changes:

```python
# Before
arrowprops=dict(arrowstyle="<->", lw=1)

# After
arrowprops=dict(arrowstyle="<->", lw=1, mutation_scale=20)
```

What does `mutation_scale=20` change?

- [ ] A) It increases the arrow line width to 20 points
- [ ] B) It scales the arrowhead size — a larger `mutation_scale` produces larger arrowheads relative to the default
- [ ] C) It sets the arrow length to 20% of the axes width
- [ ] D) It changes the arrow color to 20% opacity

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It scales the arrowhead size — a larger `mutation_scale` produces larger arrowheads relative to the default

**Explanation:**

- **A is incorrect:** Line width is controlled by `lw` (already set to 1) in `arrowprops`. `mutation_scale` does not affect line width.
- **B is correct:** `mutation_scale` is a scaling factor applied to the arrowhead geometry for `FancyArrowPatch` objects (created when using `arrowstyle` in `arrowprops`). The default is approximately 10–15. Increasing `mutation_scale=20` enlarges the arrowhead(s) proportionally. For the `"<->"` style, both arrowheads become larger. This is useful when arrowheads appear too small relative to the figure size or line length.
- **C is incorrect:** Arrow length is determined by the positions of `xy` and `xytext`, not by `mutation_scale`. The parameter affects only arrowhead geometry.
- **D is incorrect:** Opacity is controlled by `alpha`. `mutation_scale` is a purely geometric scaling parameter with no relationship to colour or transparency.

</details>

---

### Question 39

A machine learning engineer reviews the following annotation and predicts its visual output:

```python
ax.annotate("",
            xy=(0.7, 0.4),
            xytext=(0.3, 0.7),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>",
                color="forestgreen",
                lw=2,
                mutation_scale=15
            ))
```

What is rendered?

- [ ] A) A filled green triangle arrowhead at (0.7, 0.4) connected to (0.3, 0.7) by a green line, with no text
- [ ] B) A text label at (0.3, 0.7) in axes fraction coordinates with a filled green arrow pointing to (0.7, 0.4)
- [ ] C) Nothing is rendered because the text string is empty and `arrowprops` requires non-empty text
- [ ] D) An error: `xycoords="axes fraction"` cannot be combined with `arrowstyle="-|>"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A filled green triangle arrowhead at (0.7, 0.4) connected to (0.3, 0.7) by a green line, with no text

**Explanation:**

- **A is correct:** The text is an empty string `""`, so no text label is rendered. `arrowstyle="-|>"` draws a line from `xytext=(0.3, 0.7)` ending with a filled triangular arrowhead at `xy=(0.7, 0.4)`. Both coordinates use `axes fraction`, placing the arrow within the axes box. The arrow is forestgreen with `lw=2` thickness and larger than default arrowheads due to `mutation_scale=15`. This pattern (empty text + arrowprops) is used to draw bare annotation arrows without labels.
- **B is incorrect:** The text string is `""`, so no text label appears. There is no visible text at (0.3, 0.7).
- **C is incorrect:** Empty string is a fully valid annotation text. `arrowprops` has no requirement for non-empty text. The annotation renders correctly as an arrow-only element.
- **D is incorrect:** `xycoords="axes fraction"` is compatible with all `arrowstyle` values including `"-|>"`. The coordinate system and arrow style are independent parameters.

</details>

---

### Question 40

A researcher wants an arrow that exits `xytext` going **to the left**, curves downward, and arrives at `xy` going **downward**. Which `connectionstyle` configuration achieves this?

- [ ] A) `connectionstyle="arc3,rad=-0.3"`
- [ ] B) `connectionstyle="angle3,angleA=180,angleB=-90"`
- [ ] C) `connectionstyle="arc3,rad=0.3"`
- [ ] D) `connectionstyle="angle,angleA=0,angleB=90"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `connectionstyle="angle3,angleA=180,angleB=-90"`

**Explanation:**

- **A is incorrect:** `arc3,rad=-0.3` produces a curved arc without specific angle constraints for departure and arrival. The departure and arrival angles are not explicitly controlled; only the curvature direction (bow) is set. This cannot guarantee the "exits left, arrives downward" behavior.
- **B is correct:** `angle3` with explicit `angleA` and `angleB` controls departure and arrival angles. `angleA=180` means the arrow departs `xytext` going to the left (180° from the positive x-axis). `angleB=-90` means the arrow arrives at `xy` going downward (-90° from positive x-axis, i.e., straight down). The quadratic Bezier interpolation between these two tangent directions creates the required path.
- **C is incorrect:** `arc3,rad=0.3` creates a gently bowed arc but does not specify departure or arrival tangent angles. The path direction at each endpoint is determined by the arc geometry, not by explicit angle constraints.
- **D is incorrect:** `angle` (without the `3`) produces a path with two straight line segments meeting at a corner, not a smooth Bezier curve. Additionally, `angleA=0` would exit rightward (positive x), not leftward.

</details>
