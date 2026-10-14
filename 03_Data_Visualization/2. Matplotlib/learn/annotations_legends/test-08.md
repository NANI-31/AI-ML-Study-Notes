# Matplotlib Annotations & Legends — Test 08: External Legend Pinning

---

### Question 71

A data scientist places a legend outside the axes using:

```python
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
```

What does `bbox_to_anchor=(1.02, 1)` specify?

- [ ] A) The center of the legend box is placed at axes fraction (1.02, 1) — just past the right edge of the axes
- [ ] B) The legend is placed 1.02 data units to the right and 1 data unit above the origin
- [ ] C) The anchor point defined by `loc` on the legend box (upper-left corner) is pinned to the axes fraction coordinate (1.02, 1)
- [ ] D) The legend is placed 102% of the figure width to the right and 100% of the figure height upward

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The anchor point defined by `loc` on the legend box (upper-left corner) is pinned to the axes fraction coordinate (1.02, 1)

**Explanation:**

- **A is incorrect:** `bbox_to_anchor` does not pin the center of the legend. The specific corner that gets pinned is determined by the `loc` argument. `loc="upper left"` means the upper-left corner of the legend box is what gets anchored to the specified coordinate.
- **B is incorrect:** `bbox_to_anchor` coordinates are in axes-fraction space by default (where [0, 1] × [0, 1] spans the axes box), not in data coordinates. (1.02, 1) means 2% beyond the right edge and at the top of the axes box.
- **C is correct:** The interaction between `loc` and `bbox_to_anchor` is the key concept: `loc` identifies which corner of the legend box to use as the anchor point; `bbox_to_anchor` specifies where that corner is pinned in axes-fraction coordinates. With `loc="upper left"`, the **upper-left corner** of the legend box is anchored at (1.02, 1) in axes fraction — which is 2% to the right of the axes right edge and at the top of the axes box. The legend therefore appears just outside the right side of the axes.
- **D is incorrect:** `bbox_to_anchor` defaults to axes-fraction coordinates, not figure-fraction coordinates. (1.02, 1) is 102% of the axes width (2% outside the right edge), not 102% of the figure width.

</details>

---

### Question 72

A researcher places a legend below the axes for a figure that will be printed on a narrow page:

```python
ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, -0.12),
          ncol=4,
          frameon=True,
          fancybox=True)
```

Where does the legend appear and how are entries arranged?

- [ ] A) At the center of the axes area with 4 columns of entries and a rounded frame
- [ ] B) Centered horizontally below the axes (at 50% of axes width, 12% below the bottom edge) with 4 entries per row and a rounded frame
- [ ] C) At the upper center of the axes with 4 columns — below the title
- [ ] D) At (0.5, -0.12) in figure-fraction coordinates — near the bottom-center of the figure canvas

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Centered horizontally below the axes (at 50% of axes width, 12% below the bottom edge) with 4 entries per row and a rounded frame

**Explanation:**

- **A is incorrect:** `bbox_to_anchor=(0.5, -0.12)` places the anchor point outside the axes (negative y means below the bottom edge). The legend is not inside the axes area.
- **B is correct:** `bbox_to_anchor=(0.5, -0.12)` in axes-fraction coordinates means: x=0.5 (horizontal center of the axes box) and y=-0.12 (12% below the bottom edge of the axes box). With `loc="upper center"`, the upper-center point of the legend box is pinned to this coordinate, so the legend body hangs below the axes. `ncol=4` arranges legend entries into 4 columns per row (forming a compact horizontal legend). `fancybox=True` applies rounded corners to the frame.
- **C is incorrect:** The y=-0.12 in axes fraction places the legend below the axes, not within the upper portion of the axes area.
- **D is incorrect:** `bbox_to_anchor` defaults to axes-fraction coordinates, not figure coordinates. (0.5, -0.12) is relative to the axes box, not the figure canvas. To use figure coordinates, `bbox_transform=fig.transFigure` would be needed.

</details>

---

### Question 73

A machine learning engineer plots 8 model ROC curves and places the legend outside the right edge:

```python
fig, ax = plt.subplots(figsize=(8, 5))
# ... 8 ax.plot() calls with labels ...
ax.legend(loc="center left",
          bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
```

After running this code, the legend is cut off on the right side of the figure. What is the correct fix?

- [ ] A) Change `loc` to `"lower right"` so the legend moves inside the axes
- [ ] B) Add `plt.tight_layout(rect=[0, 0, 0.78, 1])` or use `plt.subplots_adjust(right=0.75)` to reserve space on the right side of the figure for the legend
- [ ] C) Use `fig.set_figwidth(20)` to make the figure very wide
- [ ] D) Set `ncol=8` to arrange all legend entries in a single row

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add `plt.tight_layout(rect=[0, 0, 0.78, 1])` or use `plt.subplots_adjust(right=0.75)` to reserve space on the right side of the figure for the legend

**Explanation:**

- **A is incorrect:** Changing `loc` to `"lower right"` moves the legend inside the axes, potentially overlapping the ROC curves. The external placement pattern (`bbox_to_anchor` beyond 1.0) is the intended layout; the issue is figure clipping.
- **B is correct:** The legend is cut off because it extends beyond the figure canvas boundary. `tight_layout` with the default `rect=[0, 0, 1, 1]` tries to fit all subplots within the full figure but does not account for the external legend. Using `plt.tight_layout(rect=[0, 0, 0.78, 1])` constrains the subplot area to use only 78% of the figure width, leaving 22% on the right for the legend. Alternatively, `plt.subplots_adjust(right=0.75)` manually sets the right boundary of the subplot area. Both approaches create the required white space for the external legend.
- **C is incorrect:** While a wider figure would eventually provide enough canvas space, it is not a precise or professional solution. The legend would still be cut off unless the axes-to-figure width ratio is also adjusted. It also changes the figure proportions undesirably.
- **D is incorrect:** `ncol=8` would arrange all legend entries in a single horizontal row. This would reduce the legend's height but not its width — it would still extend beyond the figure's right edge.

</details>

---

### Question 74

A developer uses the four-tuple `bbox_to_anchor` form:

```python
ax.legend(loc="upper left",
          bbox_to_anchor=(1.05, 0.7, 0.3, 0.2))
```

What does the four-tuple `(x0, y0, width, height)` specify?

- [ ] A) The bounding box of the legend box itself: placed at (1.05, 0.7) with width 0.3 and height 0.2 in data coordinates
- [ ] B) A bounding box rectangle defined by lower-left corner (1.05, 0.7) and dimensions (0.3 × 0.2) in axes-fraction space; the legend's `loc` corner is placed relative to this reference box
- [ ] C) The four corners of the legend box in axes-fraction coordinates
- [ ] D) The padding around the legend: left=1.05, bottom=0.7, right=0.3, top=0.2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A bounding box rectangle defined by lower-left corner (1.05, 0.7) and dimensions (0.3 × 0.2) in axes-fraction space; the legend's `loc` corner is placed relative to this reference box

**Explanation:**

- **A is incorrect:** The four-tuple does not specify the legend box's own dimensions. The legend size is determined by its content (handles and labels). The four-tuple defines the **reference bounding box** to which the legend is anchored.
- **B is correct:** The four-tuple form of `bbox_to_anchor` specifies a rectangular bounding box `(x0, y0, width, height)` in axes-fraction coordinates. This is a reference box, not the legend box itself. The `loc` parameter then determines which corner of the legend box is aligned to which corner of this reference bounding box. With `loc="upper left"`, the upper-left corner of the legend is pinned to the upper-left corner of the reference box at (1.05, 0.7+0.2=0.9). This form provides fine-grained control over legend alignment relative to a specific rectangular region.
- **C is incorrect:** The four-tuple specifies a box (lower-left corner + dimensions), not four explicit corner coordinates. Four corner coordinates would require a different specification format.
- **D is incorrect:** Legend padding is set via the `borderpad` and `borderaxespad` arguments, not via `bbox_to_anchor`. The four-tuple defines a geometric reference box.

</details>

---

### Question 75

A data analyst uses `bbox_transform=fig.transFigure` in her legend call:

```python
ax.legend(loc="lower center",
          bbox_to_anchor=(0.5, 0.02),
          bbox_transform=fig.transFigure,
          ncol=3)
```

Why is `bbox_transform=fig.transFigure` useful here?

- [ ] A) It doubles the figure size to accommodate the legend
- [ ] B) It makes `bbox_to_anchor=(0.5, 0.02)` be interpreted in figure-fraction coordinates — placing the legend at 50% of figure width and 2% of figure height — useful for consistent placement across figure layouts with varying subplot sizes
- [ ] C) It ensures the legend is rendered with the highest resolution (figure-level DPI)
- [ ] D) It prevents the legend from being clipped by the axes bounding box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It makes `bbox_to_anchor=(0.5, 0.02)` be interpreted in figure-fraction coordinates — placing the legend at 50% of figure width and 2% of figure height — useful for consistent placement across figure layouts with varying subplot sizes

**Explanation:**

- **A is incorrect:** `bbox_transform` does not affect figure size. It only changes the coordinate system used to interpret `bbox_to_anchor` values.
- **B is correct:** By default, `bbox_to_anchor` coordinates are relative to the axes box (using `ax.transAxes`). Setting `bbox_transform=fig.transFigure` changes the coordinate space so that `(0.5, 0.02)` is interpreted relative to the entire figure canvas. This places the legend at the horizontal center and near the bottom of the full figure — regardless of how many subplots there are, where the axes are positioned, or how `subplots_adjust` is configured. This is essential for placing a shared legend at a consistent location on multi-subplot figures.
- **C is incorrect:** `bbox_transform` specifies a coordinate transform, not a rendering resolution. DPI is controlled by `fig.set_dpi()` or `savefig(dpi=...)`.
- **D is incorrect:** Clipping behavior is controlled by `clip_on` and the figure-level clipping settings. `bbox_transform` changes the coordinate interpretation, not the clipping rules.

</details>

---

### Question 76

A researcher plots a 4-model comparison ROC chart and creates this legend:

```python
ax.legend(loc="upper left",
          bbox_to_anchor=(1.02, 1),
          borderaxespad=0,
          framealpha=0.9,
          edgecolor="lightgray")
```

What does `borderaxespad=0` do?

- [ ] A) Sets the legend border thickness to 0 points, removing the legend frame
- [ ] B) Removes the gap (padding) between the legend box and the axes boundary, placing the legend flush against the axes edge
- [ ] C) Sets the number of border columns in the legend to 0
- [ ] D) Disables the automatic repositioning of the legend when the axes is resized

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Removes the gap (padding) between the legend box and the axes boundary, placing the legend flush against the axes edge

**Explanation:**

- **A is incorrect:** The legend frame thickness is controlled by `edgecolor` (the frame border colour) and implicit line width, not `borderaxespad`. Removing the frame requires `frameon=False`.
- **B is correct:** `borderaxespad` specifies the amount of padding (in font-size units) between the axes bounding box and the legend box. The default is a small positive value that creates a visual gap between the axes edge and the legend. Setting `borderaxespad=0` eliminates this gap entirely, placing the legend directly adjacent to the axes edge. Combined with `bbox_to_anchor=(1.02, 1)`, this ensures the legend sits as close as possible to the right edge without overlap.
- **C is incorrect:** `borderaxespad` is not a column count parameter. Column count is controlled by `ncol`.
- **D is incorrect:** `borderaxespad` is a static spacing parameter, not a layout behavior toggle. It does not control how the legend responds to axes resizing.

</details>

---

### Question 77

A data scientist creates a legend outside the axes and then saves the figure:

```python
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.savefig("roc_comparison.png", dpi=150)
```

The saved image clips the legend. What `savefig` argument preserves the external legend?

- [ ] A) `plt.savefig("roc_comparison.png", dpi=150, pad_inches=2.0)`
- [ ] B) `plt.savefig("roc_comparison.png", dpi=150, bbox_inches="tight")`
- [ ] C) `plt.savefig("roc_comparison.png", dpi=150, legend_outside=True)`
- [ ] D) `plt.savefig("roc_comparison.png", dpi=150, clip_box="expanded")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.savefig("roc_comparison.png", dpi=150, bbox_inches="tight")`

**Explanation:**

- **A is incorrect:** `pad_inches` adds uniform padding around the figure's existing bounding box. It does not expand the bounding box to include external artists like legends outside the axes.
- **B is correct:** `bbox_inches="tight"` tells Matplotlib to compute the tightest bounding box that includes all visible artists in the figure — including external legends positioned outside the axes box. Without this argument, `savefig` clips the output to the figure's default bounding box (its `figsize` dimensions), cutting off any artists that extend beyond the declared figure boundaries. `bbox_inches="tight"` is the standard solution for saving figures with external legends.
- **C is incorrect:** `legend_outside=True` is not a valid `savefig` argument. External legend placement is handled through `bbox_to_anchor` and `bbox_inches="tight"` — not a dedicated save argument.
- **D is incorrect:** `clip_box="expanded"` is not a valid `plt.savefig` argument. There is no such parameter in Matplotlib's save pipeline.

</details>

---

### Question 78

A developer creates a figure with two side-by-side subplots and wants a single shared legend outside to the right. She uses:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# ... plot on both axes ...
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(1.0, 0.5))
```

Why is `fig.legend()` used instead of `ax1.legend()` or `ax2.legend()`?

- [ ] A) `fig.legend()` automatically collects handles from all subplots, so no `handles` argument is needed
- [ ] B) `fig.legend()` places the legend relative to the entire figure canvas, making it easier to position a shared legend between or outside the subplot panel
- [ ] C) `ax.legend()` does not support `bbox_to_anchor` outside the [0, 1] range
- [ ] D) `fig.legend()` is faster to render than `ax.legend()` for multi-subplot figures

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.legend()` places the legend relative to the entire figure canvas, making it easier to position a shared legend between or outside the subplot panel

**Explanation:**

- **A is incorrect:** `fig.legend()` without arguments does collect handles and labels from all axes, but the code explicitly passes `handles` and `labels` from `ax1`. When `handles` is explicitly provided, auto-collection is bypassed.
- **B is correct:** `fig.legend()` positions the legend relative to the figure canvas (using `fig.transFigure` by default for `bbox_to_anchor`). This makes it trivial to place a shared legend at a specific figure-relative position — like centered on the right edge of the full figure — without worrying about which axes' coordinate system to use. In contrast, `ax1.legend()` with `bbox_to_anchor` uses `ax1.transAxes` by default, anchoring the legend to the first subplot's boundary, which may not align well with the second subplot.
- **C is incorrect:** `ax.legend()` fully supports `bbox_to_anchor` coordinates outside [0, 1]. Values greater than 1.0 are used specifically to place legends outside the axes.
- **D is incorrect:** There is no documented performance advantage of `fig.legend()` over `ax.legend()`. Both create `Legend` artists of comparable complexity.

</details>

---

### Question 79

A researcher uses `ncol=3` in a legend with 7 entries. How does Matplotlib lay out the 7 entries in 3 columns?

- [ ] A) 3 entries in column 1, 3 entries in column 2, 1 entry in column 3
- [ ] B) 2 entries in column 1, 2 entries in column 2, 3 entries in column 3
- [ ] C) 3 entries in row 1, 3 entries in row 2, 1 entry in row 3 (reading left to right)
- [ ] D) Matplotlib raises a `ValueError` because 7 is not divisible by 3

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 3 entries in row 1, 3 entries in row 2, 1 entry in row 3 (reading left to right)

**Explanation:**

- **A is incorrect:** Matplotlib fills legends row by row (left to right, then top to bottom), not column by column. With `ncol=3`, the fill order is: row 1 gets entries 1, 2, 3; row 2 gets entries 4, 5, 6; row 3 gets entry 7 (the remainder).
- **B is incorrect:** The distribution is not uneven in the way described. Matplotlib uses a straightforward row-first fill with the specified number of columns.
- **C is correct:** With `ncol=3` and 7 entries, Matplotlib creates `ceil(7/3) = 3` rows. Row 1: entries 1, 2, 3 (left to right). Row 2: entries 4, 5, 6. Row 3: entry 7 (left column, remaining columns empty). The total legend layout is 3 columns wide and 3 rows tall, with the last row partially filled.
- **D is incorrect:** Matplotlib handles non-divisible entry counts gracefully by filling the last row partially. No error is raised for non-divisible counts.

</details>

---

### Question 80

A data analyst has an external legend that appears correctly in an interactive Matplotlib window but is clipped when exported as a PDF. What is the most reliable fix?

- [ ] A) Change the figure backend to `"PDF"` using `matplotlib.use("PDF")` before plotting
- [ ] B) Set `plt.rcParams["savefig.bbox"] = "tight"` globally so all future `savefig` calls automatically use `bbox_inches="tight"`
- [ ] C) Manually expand the figure `figsize` by a fixed amount to accommodate the legend
- [ ] D) Set `fig.subplots_adjust(right=0.80)` after calling `ax.legend()` to pre-allocate space

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Set `plt.rcParams["savefig.bbox"] = "tight"` globally so all future `savefig` calls automatically use `bbox_inches="tight"`

**Explanation:**

- **A is incorrect:** Changing the rendering backend to `"PDF"` affects the file format but does not change the bounding box used for the saved output. The legend would still be clipped if `bbox_inches` is not set to `"tight"`.
- **B is correct:** `plt.rcParams["savefig.bbox"] = "tight"` sets the default `bbox_inches` for all subsequent `plt.savefig()` calls to `"tight"`. This means Matplotlib automatically expands the save bounding box to include all visible artists (including external legends) without needing to add `bbox_inches="tight"` to every individual save call. This is the most maintainable solution for workflows that frequently export figures with external legends.
- **C is incorrect:** Manually expanding `figsize` requires knowing the exact legend dimensions in advance, which varies with font size, number of entries, and label lengths. It is fragile and not reusable.
- **D is incorrect:** `fig.subplots_adjust(right=0.80)` shrinks the axes area to 80% of the figure width, leaving 20% space on the right. This creates space for the legend without clipping the axes contents. However, if the legend is still wider than the allocated 20%, it can still be clipped when saved with the default `bbox_inches` setting. Combining this with `bbox_inches="tight"` would be the most robust solution.

</details>
