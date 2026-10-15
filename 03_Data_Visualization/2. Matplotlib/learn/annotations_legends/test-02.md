# Matplotlib Annotations & Legends — Test 02: Coordinate Transformations

---

### Question 11

A data scientist places a model-quality stamp using:

```python
ax.text(0.97, 0.97, "R² = 0.95",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=11, fontweight="bold")
```

She then changes the x-axis limits from `[0, 10]` to `[0, 100]`. What happens to the text label?

- [ ] A) The text moves to x=97 in data space, which is now near the right edge of the new x-axis range
- [ ] B) The text remains at the top-right corner of the axes box because `transAxes` coordinates are independent of data range changes
- [ ] C) The text disappears because `transAxes` coordinates outside [0, 1] are clipped
- [ ] D) The text is recalculated to remain at 97% of the new data range (x=96.97)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The text remains at the top-right corner of the axes box because `transAxes` coordinates are independent of data range changes

**Explanation:**

- **A is incorrect:** If the text used `transform=ax.transData` (the default), then coordinates (0.97, 0.97) would be data coordinates. With the new x-axis range [0, 100], x=0.97 would place the text near the left edge, not the right. But the code uses `transAxes`, which is entirely unaffected by data range.
- **B is correct:** `transform=ax.transAxes` maps the coordinate space [0, 1] × [0, 1] to the axes bounding box. The point (0.97, 0.97) always refers to 97% of the axes width from the left and 97% of the axes height from the bottom — regardless of what data range the axes displays. Changing `xlim` from [0, 10] to [0, 100] does not change the axes bounding box geometry, so the text stays at the top-right corner.
- **C is incorrect:** The coordinates (0.97, 0.97) are well within [0, 1] × [0, 1], the valid `transAxes` range. No clipping occurs.
- **D is incorrect:** `transAxes` does not re-evaluate coordinates based on data range. The coordinate (0.97, 0.97) in axes-fraction space is a static, fixed position relative to the axes box.

</details>

---

### Question 12

A developer writes the following:

```python
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Label A: Data coordinates
ax.text(9, 9, "Data", transform=ax.transData)

# Label B: Axes fraction coordinates
ax.text(0.9, 0.9, "Axes", transform=ax.transAxes)
```

Both labels render near the top-right of the plot. If the x-axis is then zoomed to `[0, 5]`, what happens to each label?

- [ ] A) Both labels move to the center-right of the new view
- [ ] B) "Data" moves off-screen (x=9 is outside the new [0, 5] range); "Axes" stays at the top-right corner
- [ ] C) "Axes" moves off-screen; "Data" stays at the top-right corner
- [ ] D) Both labels stay at the top-right because Matplotlib always pins text to the nearest axes corner

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) "Data" moves off-screen (x=9 is outside the new [0, 5] range); "Axes" stays at the top-right corner

**Explanation:**

- **A is incorrect:** Only the `transData` label moves with data zoom — the `transAxes` label is immune to data range changes.
- **B is correct:** `transform=ax.transData` places "Data" at the world coordinate (9, 9). After zooming x to [0, 5], the coordinate x=9 is outside the visible range and the label is rendered outside the axes box (or clipped, depending on `clip_on`). In contrast, `transform=ax.transAxes` places "Axes" at the normalized position (0.9, 0.9) relative to the axes box. Zooming the x-axis changes the data-to-pixel mapping but not the axes box geometry, so the axes-fraction label stays at 90% width and 90% height of the axes box — the top-right corner.
- **C is incorrect:** The behaviors are reversed. `transAxes` labels are immune to data zoom, not `transData` labels.
- **D is incorrect:** Matplotlib does not automatically pin text to the nearest axes corner. Text position is determined solely by the coordinate and transform specified. Only `transAxes` (or `transFigure`) coordinates are geometrically fixed relative to the axes/figure box.

</details>

---

### Question 13

A researcher adds a watermark text to a multi-axes figure using `fig.transFigure`:

```python
fig.text(0.5, 0.02, "Confidential — Internal Use Only",
         ha="center", va="bottom",
         transform=fig.transFigure,
         fontsize=9, color="gray", alpha=0.5)
```

Where does this text appear relative to the figure?

- [ ] A) At the center of the first axes object at y=0.02 in data coordinates
- [ ] B) At 50% of the figure width and 2% of the figure height from the bottom — horizontally centered in the figure, just above the bottom edge
- [ ] C) At 50% of the figure width and 2% from the top of the figure
- [ ] D) This code raises an error because `fig.text` requires `transform=ax.transAxes`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) At 50% of the figure width and 2% of the figure height from the bottom — horizontally centered in the figure, just above the bottom edge

**Explanation:**

- **A is incorrect:** `fig.transFigure` maps (0.5, 0.02) to figure coordinates, not to data coordinates of any axes. The coordinate system spans the entire figure canvas, not any individual subplot's data range.
- **B is correct:** `transform=fig.transFigure` uses normalized figure coordinates where (0, 0) is the bottom-left of the figure canvas and (1, 1) is the top-right. The point (0.5, 0.02) is at 50% of the figure width (horizontally centered) and 2% of the figure height from the bottom. `va="bottom"` anchors the text bottom edge to this position, so the text appears just above the bottom edge of the entire figure — ideal for a figure-level footer watermark.
- **C is incorrect:** y=0.02 in `transFigure` coordinates is near the bottom, not the top. The top would be near y=0.98 or y=1.0.
- **D is incorrect:** `fig.text()` is a valid `Figure`-level method for placing text using figure coordinates. The `transform` argument is optional (it defaults to `fig.transFigure`) but explicitly passing it is correct and valid. The method does not require `ax.transAxes`.

</details>

---

### Question 14

A data analyst places the same coordinate `(0.5, 0.5)` in two different calls:

```python
# Call 1: Default transform
ax.text(0.5, 0.5, "Default", color="blue")

# Call 2: Explicit transAxes
ax.text(0.5, 0.5, "Axes", color="red", transform=ax.transAxes)
```

The axes has `xlim=(0, 10)` and `ylim=(0, 8)`. Where does each label appear?

- [ ] A) Both labels appear at the center of the axes box
- [ ] B) "Default" appears at data point (0.5, 0.5) — near the bottom-left corner of the plotted area; "Axes" appears at the center of the axes box
- [ ] C) "Default" appears at the center of the axes box; "Axes" appears at data point (0.5, 0.5)
- [ ] D) Both labels appear at data point (0.5, 0.5), near the bottom-left corner

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) "Default" appears at data point (0.5, 0.5) — near the bottom-left corner of the plotted area; "Axes" appears at the center of the axes box

**Explanation:**

- **A is incorrect:** The two calls use different coordinate systems, so they place text at different locations. Only the `transAxes` call places text at the visual center of the axes box.
- **B is correct:** When no `transform` argument is provided to `ax.text()`, the default is `transform=ax.transData`. With `xlim=(0, 10)` and `ylim=(0, 8)`, the data coordinate (0.5, 0.5) is very close to the bottom-left corner (5% of the x-range, 6.25% of the y-range). In contrast, `transform=ax.transAxes` maps (0.5, 0.5) to 50% of the axes width and 50% of the axes height — the visual center of the axes box.
- **C is incorrect:** The default transform is `transData`, not `transAxes`. So "Default" uses data coordinates and "Axes" uses axes-fraction coordinates, not the other way around.
- **D is incorrect:** Only Call 1 uses data coordinates (0.5, 0.5). Call 2 with `transAxes` places "Axes" at the axes center, not at data coordinate (0.5, 0.5).

</details>

---

### Question 15

A machine learning engineer adds metric cards to a grid of subplots:

```python
for ax, metric, value in zip(axes.flat, metrics, values):
    ax.text(0.95, 0.05,
            f"{metric}: {value:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(fc="white", ec="lightgray", pad=0.3))
```

A new teammate asks: "Why not use `transform=fig.transFigure` for all labels?" What is the key reason to prefer `ax.transAxes` here?

- [ ] A) `fig.transFigure` does not support the `bbox` argument
- [ ] B) `ax.transAxes` places each label at the bottom-right corner of its own subplot, regardless of the subplot's position in the figure grid; `fig.transFigure` would place all labels at the same absolute figure position
- [ ] C) `ax.transAxes` is faster to render than `fig.transFigure` for multi-subplot figures
- [ ] D) `fig.transFigure` coordinates range from -1 to 1, so (0.95, 0.05) would be off the figure canvas

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.transAxes` places each label at the bottom-right corner of its own subplot, regardless of the subplot's position in the figure grid; `fig.transFigure` would place all labels at the same absolute figure position

**Explanation:**

- **A is incorrect:** `fig.transFigure` supports the same text kwargs (including `bbox`) as `ax.transAxes`. The `bbox` argument is part of the `Text` artist properties, not the coordinate transform.
- **B is correct:** In a grid of subplots, each `ax` has its own axes box at a different absolute position on the figure canvas. `ax.transAxes` maps (0.95, 0.05) to the bottom-right corner of that specific subplot. Because each axes has its own `transAxes` transform, the same (0.95, 0.05) coordinate places the label correctly in each subplot's bottom-right corner. Using `fig.transFigure` would require computing the absolute figure coordinates for each subplot's bottom-right corner separately — much more complex and not robust to layout changes.
- **C is incorrect:** There is no documented performance difference between `transAxes` and `transFigure` for text rendering. Both are affine transformations with negligible computational cost.
- **D is incorrect:** `fig.transFigure` coordinates range from 0 to 1, not -1 to 1. (0.95, 0.05) would be valid `transFigure` coordinates near the right edge and bottom of the figure.

</details>

---

### Question 16

A researcher runs the following code and is confused about where the text appears:

```python
ax.set_xlim(100, 200)
ax.set_ylim(50, 150)
ax.text(0.1, 0.9, "Label", fontsize=12)   # no transform specified
```

Where does "Label" appear?

- [ ] A) Near the top-left corner of the axes box (10% from left, 90% up)
- [ ] B) Near the bottom-left corner of the axes (x=0.1 and y=0.9 are both close to the left-bottom edge of the data range)
- [ ] C) The text is invisible because (0.1, 0.9) is outside the data range [100, 200] × [50, 150]
- [ ] D) The text appears at the top of the figure because y=0.9 is interpreted as 90% of figure height

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The text is invisible because (0.1, 0.9) is outside the data range [100, 200] × [50, 150]

**Explanation:**

- **A is incorrect:** That position (10% from left, 90% from bottom within the axes box) would require `transform=ax.transAxes`. Without the transform argument, the default is `transData`.
- **B is incorrect:** The data ranges are x∈[100, 200] and y∈[50, 150]. The data coordinates (0.1, 0.9) are far outside both ranges — x=0.1 is 99.9 units to the left of the leftmost visible x value, and y=0.9 is 49.1 units below the minimum y. There is no "bottom-left edge at 0.1"; the bottom-left of the data range starts at (100, 50).
- **C is correct:** Because no `transform` is specified, `ax.text` uses `transform=ax.transData` by default. The data coordinates (0.1, 0.9) are entirely outside the visible data range x∈[100, 200], y∈[50, 150]. The text is rendered far off-screen (to the left and below the visible axes area) and is invisible in the output. By default `clip_on=True` for text, so it is clipped. The correct fix is `transform=ax.transAxes` if a normalized position was intended.
- **D is incorrect:** Without `transform=fig.transFigure`, the y=0.9 value is interpreted as a data coordinate of 0.9, not as 90% of figure height.

</details>

---

### Question 17

Which `transform` argument should be used to place a shared title above a grid of subplots at exactly the horizontal center of the entire figure?

- [ ] A) `transform=axes[0].transAxes` — uses the first subplot's coordinate system
- [ ] B) `transform=fig.transFigure` — uses the whole figure's normalized coordinate system
- [ ] C) `transform=axes[-1].transData` — uses the last subplot's data coordinates
- [ ] D) No `transform` is needed — `fig.suptitle()` handles this automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) No `transform` is needed — `fig.suptitle()` handles this automatically

**Explanation:**

- **A is incorrect:** `axes[0].transAxes` would center the text within the first subplot only. For a 2×3 grid, the first subplot occupies only one-third of the figure width; the "center" of its `transAxes` is not the center of the figure.
- **B is incorrect:** While `fig.transFigure` at (0.5, 0.98) would place text at the horizontal center of the figure (and is a valid approach when using `fig.text`), this is not the cleanest or most recommended approach for a figure-level title.
- **C is incorrect:** Data coordinates from a specific subplot are not appropriate for placing figure-level titles. The data range is subplot-specific and unrelated to figure geometry.
- **D is correct:** `fig.suptitle("Your Title", fontsize=14)` is the purpose-built Matplotlib API for placing a super-title (shared title) above all subplots, automatically centered on the figure. It internally uses figure coordinates and respects subplot spacing. It is the idiomatic and recommended way to add a figure-level heading without manual coordinate calculation.

</details>

---

### Question 18

A developer wants to add a diagonal watermark text across a figure that reads "DRAFT". She uses:

```python
fig.text(0.5, 0.5, "DRAFT",
         fontsize=60,
         color="lightgray",
         alpha=0.3,
         ha="center",
         va="center",
         rotation=45,
         transform=fig.transFigure)
```

Where is the center of the "DRAFT" text placed?

- [ ] A) At the center of the first (top-left) subplot
- [ ] B) At the exact center of the figure canvas (50% of figure width, 50% of figure height)
- [ ] C) At the center of the last (bottom-right) subplot
- [ ] D) At 50% of the data range of the first subplot's x-axis and y-axis

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) At the exact center of the figure canvas (50% of figure width, 50% of figure height)

**Explanation:**

- **A is incorrect:** `transform=fig.transFigure` uses the figure canvas coordinate system, not the coordinate system of any individual subplot. (0.5, 0.5) in `transFigure` refers to the center of the entire figure.
- **B is correct:** `fig.transFigure` normalizes the entire figure canvas to the [0, 1] × [0, 1] space. The coordinate (0.5, 0.5) maps to the exact center of the figure canvas — horizontally and vertically centered across all subplots, margins, and padding. Combined with `ha="center", va="center"`, the text string's center is placed at the figure's center, and `rotation=45` tilts the text diagonally — the classic watermark layout.
- **C is incorrect:** The bottom-right subplot center would be at a figure-fraction coordinate above (0.5, 0.5), depending on the grid layout. (0.5, 0.5) in figure coordinates is the geometric center of the entire canvas.
- **D is incorrect:** `transform=fig.transFigure` has no relationship to any subplot's data range. The coordinates are purely geometric fractions of the figure canvas dimensions.

</details>

---

### Question 19

A data analyst asks: "When should I use `transform=ax.transData` explicitly in `ax.text()`?" Which answer is most accurate?

- [ ] A) Always — it is a best practice to specify the transform explicitly even though it is the default
- [ ] B) When placing text that should remain fixed in the corner of the axes box regardless of data zoom
- [ ] C) When placing text that should be positioned relative to specific data values and should move when the axis is panned or zoomed
- [ ] D) When placing text on a figure object using `fig.text()` instead of `ax.text()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) When placing text that should be positioned relative to specific data values and should move when the axis is panned or zoomed

**Explanation:**

- **A is incorrect:** While being explicit is good practice for clarity, specifying `transform=ax.transData` is redundant since it is already the default for `ax.text()`. The practical answer concerns when it is semantically meaningful to use it.
- **B is incorrect:** Text that should remain fixed in the corner regardless of zoom should use `transform=ax.transAxes` (axes-fraction coordinates), not `transData`. `transData` is tied to the data range and moves with zoom/pan.
- **C is correct:** `transform=ax.transData` places text at specific data coordinates. When the user pans or zooms the interactive plot, the text moves with the data it annotates — staying at, say, "the point where x=5.7 and y=3.2". This is appropriate for annotations like "Peak at epoch 42" placed directly next to a data point, where the annotation should follow the data if the axes range changes.
- **D is incorrect:** `fig.text()` uses `fig.transFigure` by default, not `ax.transData`. Specifying `transform=ax.transData` in a `fig.text()` call is unusual and technically maps figure-space coordinates through an axes transform, producing counter-intuitive results.

</details>

---

### Question 20

A researcher discovers a coordinate-system bug in a colleague's code:

```python
ax.set_xlim(0, 50)
ax.set_ylim(0, 1)

ax.text(0.85, 0.92, "AUC = 0.91",
        fontsize=10,
        color="darkblue")   # no transform specified
```

What is the bug and what is the symptom?

- [ ] A) The text uses data coordinates (0.85, 0.92), which are inside the data range, so the label appears near the bottom-left corner of the axes
- [ ] B) The text uses data coordinates (0.85, 0.92). Since x=0.85 is outside `xlim=[0, 50]`, the text is clipped and invisible
- [ ] C) The text appears correctly in the top-right corner because Matplotlib auto-detects that values near [0, 1] should use `transAxes`
- [ ] D) The text uses data coordinates (0.85, 0.92). With `xlim=[0, 50]`, x=0.85 maps to near the left edge of the plot; with `ylim=[0, 1]`, y=0.92 maps to near the top of the plot — making the label appear top-left, not top-right

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The text uses data coordinates (0.85, 0.92). With `xlim=[0, 50]`, x=0.85 maps to near the left edge of the plot; with `ylim=[0, 1]`, y=0.92 maps to near the top of the plot — making the label appear top-left, not top-right

**Explanation:**

- **A is incorrect:** x=0.85 and y=0.92 are indeed inside the data ranges (x∈[0, 50] and y∈[0, 1]), so the text is not clipped. But the position is near the top-left, not the bottom-left — because y=0.92 is near the top of the y-range [0, 1].
- **B is incorrect:** x=0.85 is within `xlim=[0, 50]`, so no x-clipping occurs. The text is visible, just in the wrong position.
- **C is incorrect:** Matplotlib does not auto-detect that values near [0, 1] indicate axes-fraction coordinates. The transform is determined solely by the `transform` keyword argument (defaulting to `transData`). There is no heuristic that switches transforms based on value magnitude.
- **D is correct:** Without `transform=ax.transAxes`, the text uses `transData`. In data space: x=0.85 on a range [0, 50] is 1.7% from the left edge — near the left. y=0.92 on a range [0, 1] is 92% from the bottom — near the top. So the label appears near the **top-left** corner, not the intended top-right. The fix is `transform=ax.transAxes`, where (0.85, 0.92) correctly represents 85% from the left and 92% from the bottom — the top-right area.

</details>
