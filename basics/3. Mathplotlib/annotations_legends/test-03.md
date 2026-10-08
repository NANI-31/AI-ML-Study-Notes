# Matplotlib Annotations & Legends — Test 03: Point-and-Shoot Annotations

---

### Question 21

A data scientist uses `ax.annotate` to label a peak on a loss curve:

```python
ax.annotate("Best Epoch",
            xy=(42, 0.083),
            xytext=(55, 0.15),
            arrowprops=dict(arrowstyle="->"))
```

Which argument specifies **where the arrow points to** (the annotated data point)?

- [ ] A) `xytext=(55, 0.15)` — this is the target of the annotation arrow
- [ ] B) `xy=(42, 0.083)` — this is the data point the arrow tip touches
- [ ] C) Both `xy` and `xytext` define the start and end of the arrow with no distinguished target
- [ ] D) The target is automatically determined by the nearest local minimum in the plotted data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `xy=(42, 0.083)` — this is the data point the arrow tip touches

**Explanation:**

- **A is incorrect:** `xytext=(55, 0.15)` specifies the position of the **text label** (where "Best Epoch" is rendered). The arrow *originates* from the text and points *toward* the `xy` target.
- **B is correct:** In `ax.annotate`, `xy` is the "point being annotated" — the data feature the annotation calls attention to. The arrow tip is placed at (42, 0.083). The text "Best Epoch" and the arrow tail are placed at `xytext=(55, 0.15)`. The arrow thus runs from (55, 0.15) toward (42, 0.083).
- **C is incorrect:** The two parameters have clearly distinguished semantic roles: `xy` is always the annotation target (arrow tip); `xytext` is the text label position (arrow tail).
- **D is incorrect:** `ax.annotate` does not perform any data analysis to find local extrema. It places the annotation at exactly the coordinates provided.

</details>

---

### Question 22

A researcher annotates an overfitting point on a validation curve:

```python
ax.annotate("Overfitting starts",
            xy=(30, val_loss[30]),
            xytext=(40, val_loss[30] + 0.05),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", color="crimson"))
```

What does `xycoords="data"` specify?

- [ ] A) The annotation is drawn using the figure's coordinate system
- [ ] B) The `xy` target coordinate (30, val_loss[30]) is interpreted in data space (the same coordinate system as the plotted data)
- [ ] C) The `xytext` text position is given in data coordinates
- [ ] D) All coordinates in the annotation are locked and cannot change when the plot is resized

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The `xy` target coordinate (30, val_loss[30]) is interpreted in data space (the same coordinate system as the plotted data)

**Explanation:**

- **A is incorrect:** Figure coordinates require `xycoords="figure fraction"`. `xycoords="data"` uses the axes data coordinate system (same as the x and y axis values).
- **B is correct:** `xycoords` specifies the coordinate system for the `xy` (arrow tip) argument. `"data"` means the coordinates are in the same space as the plotted data — epoch 30 on the x-axis and the validation loss value at that epoch on the y-axis. This is the most common setting and the default when `xycoords` is not specified.
- **C is incorrect:** The coordinate system for `xytext` is controlled by the separate `textcoords` parameter, not `xycoords`. Here `textcoords="data"` handles the text position.
- **D is incorrect:** `xycoords="data"` does not lock coordinates against resizing. It specifies the coordinate system, not the rendering behavior. When the axes is panned or zoomed, a `"data"` annotation moves with the data.

</details>

---

### Question 23

A developer annotates a confidence score with the text label offset in points from the data point:

```python
ax.annotate("High Confidence",
            xy=(7.3, 0.94),
            xytext=(15, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"))
```

What does `textcoords="offset points"` mean for the `xytext=(15, 10)` value?

- [ ] A) The text is placed 15 data units to the right and 10 data units above the annotation target
- [ ] B) The text is placed 15 points to the right and 10 points above the `xy` target coordinate in screen (display) pixels
- [ ] C) The text is placed 15% of the axes width to the right and 10% of the axes height above the target
- [ ] D) The text is placed 15 points (1/72 inch) to the right and 10 points above the `xy` data coordinate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The text is placed 15 points (1/72 inch) to the right and 10 points above the `xy` data coordinate

**Explanation:**

- **A is incorrect:** `"offset points"` uses typographic points (1/72 inch), not data units. 15 data units would be `textcoords="data"` with `xytext=(7.3+15, 0.94+10)`.
- **B is incorrect:** Points and pixels are different units. A typographic point is 1/72 of an inch; a pixel depends on screen resolution (DPI). At 100 DPI, 1 point ≈ 1.39 pixels. `"offset points"` uses points, not pixels.
- **C is incorrect:** Axes-fraction offsets would require `textcoords="axes fraction"`, which maps coordinates to [0, 1] fractions of the axes dimensions, not percentage offsets from the target.
- **D is correct:** `textcoords="offset points"` interprets `xytext` as a (dx, dy) offset measured in typographic points (where 1 point = 1/72 inch) relative to the screen-projected position of the `xy` target coordinate. The text is placed 15 points to the right and 10 points above the annotation target in display space. This is scale-independent: the physical distance stays the same regardless of data range or figure size.

</details>

---

### Question 24

A machine learning engineer writes two annotation calls and is confused about the output:

```python
# Call A
ax.annotate("Recall Drop",
            xy=(0.7, 0.6),
            xytext=(0.5, 0.4))

# Call B
ax.annotate("Recall Drop",
            xy=(0.5, 0.4),
            xytext=(0.7, 0.6))
```

Both calls draw an arrow between the same two points. How do the two arrows differ?

- [ ] A) Call A and Call B produce identical arrows because arrow direction is determined by proximity, not argument order
- [ ] B) In Call A, the arrow tip points to (0.7, 0.6) and the text is at (0.5, 0.4); in Call B, the arrow tip points to (0.5, 0.4) and the text is at (0.7, 0.6)
- [ ] C) In Call A, the arrow points from (0.7, 0.6) toward (0.5, 0.4); in Call B, the arrow points from (0.5, 0.4) toward (0.7, 0.6)
- [ ] D) The two calls draw the same text at different positions but no arrow is drawn without an explicit `arrowprops` argument

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) In Call A, the arrow points from (0.7, 0.6) toward (0.5, 0.4); in Call B, the arrow points from (0.5, 0.4) toward (0.7, 0.6)

**Explanation:**

- **A is incorrect:** Arrow direction is determined by the semantics of `xy` (tip) and `xytext` (tail/text), not by proximity. Swapping the coordinates produces opposite arrow directions.
- **B is incorrect:** This describes the text positions correctly but mis-states the arrow directions. Call A's arrow tip is at `xy=(0.7, 0.6)` (not at xytext), and Call B's arrow tip is at `xy=(0.5, 0.4)`.
- **C is correct:** In `ax.annotate`, the arrow runs from `xytext` (where the text and arrow tail are) to `xy` (where the arrow tip points). In Call A: arrow runs from (0.5, 0.4) to (0.7, 0.6) — pointing toward (0.7, 0.6). In Call B: arrow runs from (0.7, 0.6) to (0.5, 0.4) — pointing toward (0.5, 0.4). Swapping `xy` and `xytext` reverses the arrow direction.
- **D is incorrect:** Both calls have no `arrowprops` argument. Without `arrowprops`, `ax.annotate` places the text at `xytext` with no arrow drawn. So the correct observation is that neither call draws an arrow — but the text positions are still different in each call.

</details>

---

### Question 25

A data scientist wants to place an annotation label in the top-right corner of the axes (using `transAxes` fraction coordinates) pointing to a specific data point. She writes:

```python
ax.annotate("Global Minimum",
            xy=(w1_opt, w2_opt),
            xytext=(0.85, 0.90),
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black"))
```

What makes this annotation particularly useful compared to using `textcoords="data"` for `xytext`?

- [ ] A) `"axes fraction"` allows the text to be placed outside the figure
- [ ] B) With `textcoords="axes fraction"`, the text label stays fixed in the top-right corner of the axes box even if the data range changes, while the arrow dynamically tracks the data point `xy=(w1_opt, w2_opt)` in data space
- [ ] C) `"axes fraction"` enables the text to use bold font automatically
- [ ] D) Using `textcoords="axes fraction"` disables the arrow and the text is placed without any connecting line

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) With `textcoords="axes fraction"`, the text label stays fixed in the top-right corner of the axes box even if the data range changes, while the arrow dynamically tracks the data point `xy=(w1_opt, w2_opt)` in data space

**Explanation:**

- **A is incorrect:** `"axes fraction"` coordinates span [0, 1] × [0, 1] of the axes box. They cannot place text outside the figure unless values exceed this range (which could clip the text). This is not its key benefit.
- **B is correct:** This is the "mixed-coordinate" annotation pattern. `xycoords="data"` places the arrow tip at the exact data coordinates of the minimum. `textcoords="axes fraction"` places the text label in a fixed corner of the axes box. When the axes is zoomed or the data range changes, the text stays in the corner (like a metric card) while the arrow automatically re-points to wherever the data minimum appears in the new view. This is especially valuable for diagnostic plots where the key data point may be near any part of the axes.
- **C is incorrect:** Font weight is controlled by the `fontweight` argument to `ax.annotate`, not by the coordinate system used for `textcoords`.
- **D is incorrect:** `textcoords="axes fraction"` does not disable the arrow. The arrow is enabled by the presence of `arrowprops`. The arrow connects the `xytext` position (in axes fraction space) to the `xy` position (in data space), with Matplotlib handling the coordinate system bridging internally.

</details>

---

### Question 26

A researcher places 10 annotations in a loop using `textcoords="offset points"` to label data points. She notices labels from dense clusters are overlapping. Which parameter combination best addresses overlap without changing annotation targets?

- [ ] A) Increase `arrowprops` `lw` to 3.0 — thicker arrows push labels further apart
- [ ] B) Increase the `xytext` offset magnitudes and add `arrowprops` with longer arrows to give more text space
- [ ] C) Set `xycoords="figure fraction"` to distribute labels evenly across the figure
- [ ] D) Set `clip_on=False` to allow labels to extend beyond the axes bounding box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Increase the `xytext` offset magnitudes and add `arrowprops` with longer arrows to give more text space

**Explanation:**

- **A is incorrect:** Arrow line width (`lw`) controls only the stroke thickness of the arrow shaft. It has no effect on the physical separation between label positions. Thicker arrows do not push labels apart.
- **B is correct:** Overlap occurs when annotations at nearby data points have small `xytext` offsets that all cluster in the same direction. Increasing the magnitude of offsets (e.g., from `(15, 10)` to `(40, 30)` points) spreads the text labels further from the data points, reducing overlap. When labels are spread further out, longer arrows are needed to connect them back to their targets. For highly overlapping scenarios, manually choosing different offset directions for each annotation (`(30, 20)` for one, `(-30, 20)` for another) provides even better separation.
- **C is incorrect:** Switching to `xycoords="figure fraction"` would convert the annotation target coordinates to figure-fraction space, which is entirely incorrect for data-specific annotations. It would completely detach labels from their intended data points.
- **D is incorrect:** `clip_on=False` allows labels to render outside the axes box but does not reduce overlap between labels that are all within the axes. Labels that both extend outside the axes with small offsets can still overlap with each other.

</details>

---

### Question 27

A data analyst annotates three model comparison points on an ROC curve:

```python
for name, (fpr, tpr) in models.items():
    ax.annotate(name,
                xy=(fpr, tpr),
                xytext=(fpr + 0.05, tpr - 0.08),
                fontsize=8,
                arrowprops=dict(arrowstyle="-"))
```

What does `arrowstyle="-"` (a bare dash) produce visually?

- [ ] A) A dashed line style (like `linestyle="--"`)
- [ ] B) No arrow head on either end — a plain straight line segment connecting `xytext` to `xy`
- [ ] C) An error: `"-"` is not a valid `arrowstyle`
- [ ] D) An arrow head pointing from `xy` back to `xytext` (reversed direction)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No arrow head on either end — a plain straight line segment connecting `xytext` to `xy`

**Explanation:**

- **A is incorrect:** `arrowstyle="-"` refers to the connector style (no arrowhead), not the line dash pattern. A dashed line would require `linestyle="--"` inside the `arrowprops` dictionary (e.g., `arrowprops=dict(arrowstyle="-", linestyle="--")`).
- **B is correct:** In Matplotlib's arrow style nomenclature, `"-"` specifies a simple line with no arrowhead at either end. It produces a plain straight line connector between the text and the annotation target — useful when a visual link is needed without the directional semantics of an arrowhead. Other styles like `"->"` add a forward arrowhead, `"<-"` adds a reverse arrowhead, and `"<->"` adds heads on both ends.
- **C is incorrect:** `"-"` is a fully valid `arrowstyle` value. Matplotlib's arrow patch supports `"-"`, `"->"`, `"<-"`, `"<->"`, `"-|>"`, `"fancy"`, `"simple"`, and others.
- **D is incorrect:** `arrowstyle="-"` produces no arrowhead at all. The direction reversal would require `"<-"`.

</details>

---

### Question 28

A developer places an annotation using `xycoords="axes fraction"` for both the target and text:

```python
ax.annotate("Converged",
            xy=(0.98, 0.12),
            xytext=(0.75, 0.30),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->"))
```

If the axes data range is `xlim=(0, 200)` and `ylim=(0, 1)`, where does the arrow tip appear?

- [ ] A) At data coordinate (0.98, 0.12) — near the origin, bottom-left of the axes
- [ ] B) At 98% of the axes width from the left and 12% of the axes height from the bottom — near the bottom-right corner of the axes box
- [ ] C) At figure coordinate (0.98, 0.12) — near the top-right of the entire figure canvas
- [ ] D) The annotation fails because `xycoords="axes fraction"` cannot be used for the arrow target

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) At 98% of the axes width from the left and 12% of the axes height from the bottom — near the bottom-right corner of the axes box

**Explanation:**

- **A is incorrect:** `xycoords="axes fraction"` does not use data coordinates. With this setting, (0.98, 0.12) is entirely unrelated to the data ranges [0, 200] × [0, 1]. The data range is irrelevant here.
- **B is correct:** `xycoords="axes fraction"` maps (0.98, 0.12) to 98% of the axes width (almost at the right edge) and 12% of the axes height from the bottom (near the bottom). The arrow tip appears in the bottom-right region of the axes box. This placement is independent of the data coordinate ranges.
- **C is incorrect:** Figure coordinates require `xycoords="figure fraction"`. `"axes fraction"` is relative to the individual axes box, not the whole figure canvas.
- **D is incorrect:** `"axes fraction"` is a fully supported `xycoords` value. Common `xycoords` options include `"data"`, `"axes fraction"`, `"axes points"`, `"figure fraction"`, and others.

</details>

---

### Question 29

A researcher annotates the final epoch on a training plot without specifying `arrowprops`:

```python
ax.annotate(f"Final Loss: {loss[-1]:.3f}",
            xy=(epochs[-1], loss[-1]),
            xytext=(epochs[-1] - 10, loss[-1] + 0.05))
```

What is the visual output of this call?

- [ ] A) The text is placed at `xytext` with an arrow pointing to `xy`
- [ ] B) Only the text is rendered at `xytext` with no arrow connecting it to `xy`
- [ ] C) The text is placed at `xy` with no offset — `xytext` is ignored without `arrowprops`
- [ ] D) A `TypeError` is raised because `arrowprops` is required when both `xy` and `xytext` are specified

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Only the text is rendered at `xytext` with no arrow connecting it to `xy`

**Explanation:**

- **A is incorrect:** Without `arrowprops`, no arrow is drawn. The `arrowprops` dictionary is what activates the arrow connector in `ax.annotate`.
- **B is correct:** When `arrowprops` is omitted (or set to `None`), `ax.annotate` behaves like `ax.text` positioned at `xytext`. The text is rendered at the `xytext` position with no visual connection to the `xy` target. The `xy` argument is still processed internally but has no visible effect in the output without an arrow or other visual connector.
- **C is incorrect:** Without `arrowprops`, `xytext` is not ignored. The text is rendered at the `xytext` position. Only the arrow (which would link `xytext` to `xy`) is absent.
- **D is incorrect:** `arrowprops` is optional in `ax.annotate`. Providing both `xy` and `xytext` without `arrowprops` is perfectly valid and produces the described no-arrow behavior. No `TypeError` is raised.

</details>

---

### Question 30

A data scientist wants to annotate a data point where both the annotation text and the arrow tip use **axes fraction** coordinates rather than data coordinates. Which code achieves this?

- [ ] A) `ax.annotate("Label", xy=(0.6, 0.8), xytext=(0.8, 0.9))` (no transform)
- [ ] B) `ax.annotate("Label", xy=(0.6, 0.8), xytext=(0.8, 0.9), xycoords="axes fraction", textcoords="axes fraction")`
- [ ] C) `ax.annotate("Label", xy=(0.6, 0.8), xytext=(0.8, 0.9), transform=ax.transAxes)`
- [ ] D) `ax.annotate("Label", xy=(0.6, 0.8), xytext=(0.8, 0.9), xycoords="data", textcoords="axes fraction")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.annotate("Label", xy=(0.6, 0.8), xytext=(0.8, 0.9), xycoords="axes fraction", textcoords="axes fraction")`

**Explanation:**

- **A is incorrect:** Without `xycoords` specified, the default is `"data"` for both the arrow tip and the text position. The coordinates (0.6, 0.8) would be interpreted as data coordinates, not axes fractions.
- **B is correct:** Setting both `xycoords="axes fraction"` and `textcoords="axes fraction"` ensures both the arrow tip (at 0.6 x-fraction, 0.8 y-fraction) and the text (at 0.8 x-fraction, 0.9 y-fraction) are positioned relative to the axes box dimensions. Both points remain fixed in the axes-relative space regardless of data range changes.
- **C is incorrect:** `ax.annotate` does not accept a `transform` keyword argument for unified coordinate transformation the way `ax.text` does. The coordinate systems for `xy` and `xytext` are specified separately via `xycoords` and `textcoords`.
- **D is incorrect:** `xycoords="data"` would interpret `xy=(0.6, 0.8)` in data coordinates, which is not the requirement. Only `textcoords` would use axes fraction in this call, meaning the arrow tip and text use different coordinate systems.

</details>
