# Matplotlib Figure & Axes Objective Test — Set 07
# Topic: Inset Plots & Diagnostic Overlays (Questions 61–70)

---

### Question 61

A developer creates an inset axes inside a main ROC curve plot:

```python
fig = plt.figure(figsize=(8, 6))
ax_main  = fig.add_axes([0.10, 0.10, 0.85, 0.85])
ax_inset = fig.add_axes([0.58, 0.18, 0.28, 0.30])

ax_main.plot(fpr, tpr, color="steelblue", label="AUC = 0.91")
ax_main.set_xlabel("FPR"); ax_main.set_ylabel("TPR")

ax_inset.plot(fpr_zoom, tpr_zoom, color="tomato")
ax_inset.set_xlim(0, 0.1)
ax_inset.set_ylim(0.7, 1.0)
ax_inset.set_title("Low-FPR Zoom", fontsize=8)
```

What is the purpose of the inset axes in this context?

- [ ] A) To display a completely separate dataset unrelated to the main ROC curve
- [ ] B) To zoom into the clinically important low false-positive-rate region (FPR 0–0.1, TPR 0.7–1.0), providing detail that is hard to read at the main axes scale
- [ ] C) To act as a color legend for the main plot
- [ ] D) To replace the main axes when the user clicks on it

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) To zoom into the clinically important low false-positive-rate region (FPR 0–0.1, TPR 0.7–1.0), providing detail that is hard to read at the main axes scale

**Explanation:**

- **A is incorrect:** The inset plots `fpr_zoom` and `tpr_zoom` with limits `[0, 0.1]` and `[0.7, 1.0]` — this is a zoomed view of the same ROC curve data, not an unrelated dataset.
- **B is correct:** This is the classic **diagnostic inset** pattern used in clinical ML papers. The full ROC curve shows the entire FPR 0–1 range, but important clinical decision thresholds often operate at very low FPR (< 10%). The inset axes magnifies this critical region at `xlim(0, 0.1)`, `ylim(0.7, 1.0)` — making fine differences between models visible that would be indistinguishable at the main plot's scale. This is a standard academic visualization technique.
- **C is incorrect:** A color legend would use `ax.legend()` or `fig.legend()`, not a full axes with its own plot. The inset has its own plot curve, not legend handles.
- **D is incorrect:** Matplotlib's static figures do not have click-triggered axis swapping. That's an interactive widget behavior, not standard `add_axes` functionality.

</details>

---

### Question 62

A developer adds an inset axes to a training loss curve to show a **magnified convergence region**:

```python
fig = plt.figure(figsize=(9, 6))
ax = fig.add_axes([0.10, 0.12, 0.85, 0.82])
ax.plot(epochs, train_loss, label="Train Loss")
ax.plot(epochs, val_loss, label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend()

# Inset for final convergence region (epochs 80-100)
ax_ins = fig.add_axes([0.55, 0.45, 0.32, 0.35])
ax_ins.plot(epochs[80:], train_loss[80:], color="tab:blue")
ax_ins.plot(epochs[80:], val_loss[80:], color="tab:orange")
ax_ins.set_xlim(80, 100)
ax_ins.tick_params(labelsize=7)
```

What is wrong with the inset axes positioning `[0.55, 0.45, 0.32, 0.35]` in relation to the main axes `[0.10, 0.12, 0.85, 0.82]`?

- [ ] A) The inset is positioned outside the main axes region — it starts at 0.55 but the main axes starts at 0.10
- [ ] B) Nothing is wrong — the inset at `[0.55, 0.45, 0.32, 0.35]` falls inside the main axes region (right edge of inset = 0.55+0.32=0.87 ≤ 0.95; top = 0.45+0.35=0.80 ≤ 0.94) — this is the intended inset overlay
- [ ] C) The inset axes will render below the main axes because its z-order is lower
- [ ] D) The inset shares the same x-axis scale as the main axes, making zoom impossible

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Nothing is wrong — the inset at `[0.55, 0.45, 0.32, 0.35]` falls inside the main axes region — this is the intended inset overlay

**Explanation:**

- **A is incorrect:** The coordinates `[0.55, 0.45, 0.32, 0.35]` are in normalized FIGURE coordinates. The main axes region spans from (0.10, 0.12) to (0.95, 0.94) in figure coordinates. The inset starts at (0.55, 0.45) — which is well within the main axes figure region. The confusion arises from conflating figure coordinates with axes-data coordinates.
- **B is correct:** Inset right edge = 0.55 + 0.32 = 0.87, which is within the main axes' right bound of 0.95. Inset top = 0.45 + 0.35 = 0.80, within the main axes' top of 0.94. The inset is correctly placed inside the main axes figure region, creating the visual overlay effect. Both axes are independent objects — the inset has its own data coordinate system (`xlim(80,100)`) independent of the main axes.
- **C is incorrect:** Z-ordering between overlapping axes is determined by creation order — the inset (created second) renders on top of the main axes by default. This is the intended behavior.
- **D is incorrect:** The inset has its own independent coordinate system. `ax_ins.set_xlim(80, 100)` sets the inset's x-range to 80–100, completely independent of the main axes' x-range. The two axes never share scales unless explicitly linked.

</details>

---

### Question 63

A developer wants to draw a **rectangle on the main axes** to show which region is magnified in the inset:

```python
from matplotlib.patches import Rectangle

ax_main.add_patch(Rectangle((80, min_val), 20, val_range,
                             fill=False, edgecolor="red",
                             linewidth=1.5, linestyle="--"))
```

What coordinate system do the `(80, min_val)` coordinates use in `ax_main.add_patch()`?

- [ ] A) Normalized figure coordinates (0 to 1)
- [ ] B) The main axes DATA coordinates — the same coordinate system as the plotted x and y values
- [ ] C) Pixel coordinates from the top-left of the figure
- [ ] D) Normalized axes coordinates (transAxes), ranging from 0 to 1 within the axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The main axes DATA coordinates — the same coordinate system as the plotted x and y values

**Explanation:**

- **A is incorrect:** Normalized figure coordinates (0 to 1 over the full figure) are used by `fig.add_axes()` positioning and `fig.text()` by default. `ax.add_patch()` uses the axes' own data coordinate system by default.
- **B is correct:** `ax.add_patch(Rectangle((x0, y0), width, height, ...))` uses the **axes data coordinate system** by default. `(80, min_val)` means the rectangle starts at x=80 on the plot's epoch axis and y=`min_val` on the loss axis — exactly aligning with the data being visualized. This places the highlight rectangle accurately over the epoch range 80–100 in the plot, visually indicating which region is shown in the inset.
- **C is incorrect:** Pixel coordinates from the figure corner are not the default for `add_patch`. To use pixel-level coordinates, you would need to set `transform=fig.transFigure` or `transform=None`.
- **D is incorrect:** `transAxes` coordinates (0–1 within the axes bounding box) can be used by setting `transform=ax.transAxes` explicitly. The default transform for patches added to an axes is `ax.transData` — the data coordinate system.

</details>

---

### Question 64

A developer creates an inset and wants to connect it to the main axes region with **indicator lines** (zoom connector lines). Which Matplotlib class is designed for this purpose?

- [ ] A) `matplotlib.patches.FancyArrowPatch`
- [ ] B) `mpl_toolkits.axes_grid1.inset_locator.mark_inset`
- [ ] C) `ax.annotate()` with `arrowprops`
- [ ] D) `matplotlib.lines.Line2D` with normalized figure coordinates

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `mpl_toolkits.axes_grid1.inset_locator.mark_inset`

**Explanation:**

- **A is incorrect:** `FancyArrowPatch` creates arrowhead connectors but is not specifically designed for inset zoom connections. It would require manual coordinate calculation in multiple coordinate systems.
- **B is correct:** `mark_inset(ax_parent, ax_inset, loc1, loc2)` from `mpl_toolkits.axes_grid1.inset_locator` is the purpose-built function for drawing zoom connector lines between a parent axes and an inset axes. It draws two lines connecting corners of the inset axes to corresponding corners of the zoom region in the parent axes. `loc1` and `loc2` specify which corners to connect (1=bottom-left, 2=bottom-right, 3=top-right, 4=top-left). This creates the visual "zoom box" effect common in scientific papers.
- **C is incorrect:** `ax.annotate()` with arrows can create annotation arrows but is designed for single-point annotations, not for drawing the four-line zoom connector pattern. Using it for zoom connections would require multiple complex calls.
- **D is incorrect:** `Line2D` in normalized figure coordinates could draw lines but would not automatically track the inset and zoom region positions as the figure is resized or the axes data range changes.

</details>

---

### Question 65

A developer creates an inset axes for a mini confusion matrix embedded in a larger evaluation figure:

```python
fig = plt.figure(figsize=(10, 7))
ax_roc   = fig.add_axes([0.08, 0.12, 0.55, 0.82])
ax_cm    = fig.add_axes([0.68, 0.55, 0.28, 0.38])  # Confusion matrix
ax_stats = fig.add_axes([0.68, 0.12, 0.28, 0.35])  # Stats table

ax_roc.plot(fpr, tpr)
im = ax_cm.imshow(cm_array, cmap="Blues")
```

The confusion matrix inset uses `ax_cm.imshow()`. What styling step is commonly added to make the confusion matrix readable without axis rulers?

- [ ] A) `ax_cm.set_visible(False)`
- [ ] B) `ax_cm.axis("off")` to hide tick marks and spines, then `ax_cm.set_xticklabels()` and `ax_cm.set_yticklabels()` to add class labels
- [ ] C) `ax_cm.remove()` to detach the axes from the figure
- [ ] D) `ax_cm.set_aspect("auto")` to stretch the matrix cells

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax_cm.axis("off")` to hide tick marks and spines, then `ax_cm.set_xticklabels()` and `ax_cm.set_yticklabels()` to add class labels

**Explanation:**

- **A is incorrect:** `ax_cm.set_visible(False)` hides the entire axes including the image — making the confusion matrix invisible entirely. This defeats the purpose.
- **B is correct:** For a clean embedded confusion matrix: first, optionally use `ax_cm.axis("off")` or selectively hide spines with `ax_cm.spines[:].set_visible(False)` to remove the automatic numeric tick marks that would clutter a small inset. Then, apply meaningful class label tick marks using `ax_cm.set_xticks([0,1,2])` and `ax_cm.set_xticklabels(["Class 0", "Class 1", "Class 2"], rotation=45, fontsize=7)` for readable class labels. This is the standard pattern for embedded confusion matrices in ML evaluation figures.
- **C is incorrect:** `ax_cm.remove()` detaches and removes the axes from the figure — making it completely invisible.
- **D is incorrect:** `ax_cm.set_aspect("auto")` would stretch the confusion matrix cells to fill the axes — potentially distorting the visual. For confusion matrices, `ax_cm.set_aspect("equal")` ensures square cells, which is the standard readable format.

</details>

---

### Question 66

A developer creates a figure and adds a **legend-only inset** to provide a centralized legend for multiple subplots:

```python
fig = plt.figure(figsize=(12, 8))
axes = [fig.add_axes([c, r, 0.30, 0.35])
        for r in [0.55, 0.08]
        for c in [0.05, 0.37, 0.69]]

# Legend panel in a dedicated inset
ax_legend = fig.add_axes([0.69, 0.08, 0.30, 0.35])
```

If `ax_legend` occupies the same position as the last axes in the list comprehension, what visual issue arises?

- [ ] A) Both axes are shown at that position — the legend axes appears on top
- [ ] B) The list comprehension creates 6 axes; `ax_legend` creates a 7th that completely overlaps the 6th — both are rendered at position `[0.69, 0.08, 0.30, 0.35]`
- [ ] C) A `ValueError` is raised because two axes cannot occupy the same position
- [ ] D) The previous axes at that position is automatically deleted

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The list comprehension creates 6 axes; `ax_legend` creates a 7th that completely overlaps the 6th — both are rendered at position `[0.69, 0.08, 0.30, 0.35]`

**Explanation:**

- **A is incorrect while partially right:** Option A says "both axes are shown" which is true, but calls it "the legend axes appears on top" without explaining the full picture. Both are rendered with `ax_legend` on top due to creation order.
- **B is correct:** The list comprehension generates axes at positions for `(r, c)` combinations: bottom-right position is `r=0.08, c=0.69` → `[0.69, 0.08, 0.30, 0.35]`. The subsequent `ax_legend = fig.add_axes([0.69, 0.08, 0.30, 0.35])` creates a SEVENTH axes at the identical position. Both exist in the figure; `ax_legend` renders on top. This creates visual issues: the bottom-right data subplot is hidden beneath the legend axes. The correct design would either remove the overlap or use one fewer axes in the list comprehension.
- **C is incorrect:** Matplotlib never raises an error for overlapping axes positions. It silently stacks them.
- **D is incorrect:** `add_axes` with identical coordinates creates an additional axes — it does not replace the previous one. The `fig.axes` list grows by one.

</details>

---

### Question 67

A developer adds a secondary axes sharing the x-axis with an existing main axes to create a dual y-axis plot:

```python
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(epochs, train_loss, color="steelblue", label="Loss")
ax1.set_ylabel("Loss", color="steelblue")

ax2 = ax1.twinx()   # Creates a second axes sharing the same x-axis
ax2.plot(epochs, accuracy, color="tomato", label="Accuracy")
ax2.set_ylabel("Accuracy (%)", color="tomato")
```

Internally, what does `ax1.twinx()` do in terms of `add_axes` geometry?

- [ ] A) It creates a new axes with independent x AND y axes at a new position
- [ ] B) It creates a second axes at the same position as `ax1` with x-axis linked to `ax1.xaxis` but with an independent y-axis (typically on the right spine)
- [ ] C) It splits the main axes vertically into two halves
- [ ] D) It creates a mirror of `ax1` at the symmetrically opposite position in the figure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It creates a second axes at the same position as `ax1` with x-axis linked to `ax1.xaxis` but with an independent y-axis (typically on the right spine)

**Explanation:**

- **A is incorrect:** `twinx()` does NOT create an independent x-axis. The x-axis is shared (linked) between `ax1` and `ax2` — zooming or panning on one x-axis automatically updates the other.
- **B is correct:** `ax1.twinx()` creates a new `Axes` object (`ax2`) at the **exact same position** as `ax1` in normalized figure coordinates. The new axes' x-axis is linked to `ax1.xaxis` (they share the same x-scale). The new axes has an **independent y-axis** positioned on the right spine, allowing a second dataset with a completely different y-scale to be overlaid on the same x-axis. This is implemented internally using `ax2 = ax1.figure.add_axes(ax1.get_position(), sharex=ax1, frameon=False)` with the left and bottom spines hidden.
- **C is incorrect:** `twinx()` creates an overlay, not a split. Both axes occupy the full extent of the original axes position.
- **D is incorrect:** There is no "mirror position" concept in `twinx()`. The twin axes is placed at exactly the same normalized figure coordinates as the original.

</details>

---

### Question 68

A developer creates an inset axes for a convergence zoom and realizes the inset's tick labels are too large. They want to set the font size of ALL tick labels in the inset to 7pt. Which call is correct?

- [ ] A) `ax_inset.set_fontsize(7)`
- [ ] B) `ax_inset.tick_params(labelsize=7)`
- [ ] C) `ax_inset.ticks.set_size(7)`
- [ ] D) `ax_inset.xticklabels(fontsize=7)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax_inset.tick_params(labelsize=7)`

**Explanation:**

- **A is incorrect:** `ax.set_fontsize()` is not a valid Matplotlib `Axes` method. Setting a global font size requires `plt.rcParams["font.size"]` or per-element font size settings.
- **B is correct:** `ax.tick_params(labelsize=7)` is the correct method to set the font size of tick labels on both x and y axes simultaneously. To target only one axis: `ax.tick_params(axis="x", labelsize=7)` or `ax.tick_params(axis="y", labelsize=7)`. This is the standard way to style tick labels in inset axes where smaller fonts are needed to avoid clutter.
- **C is incorrect:** `ax.ticks` is not a valid attribute. Tick-related objects are accessed via `ax.xaxis.get_major_ticks()` etc., but `ax.ticks` doesn't exist.
- **D is incorrect:** `ax.xticklabels(fontsize=7)` is not a valid method call. The correct methods are `ax.set_xticklabels(labels, fontsize=7)` (which requires passing labels) or `ax.tick_params(axis="x", labelsize=7)`.

</details>

---

### Question 69

A developer creates a figure with an inset and a main axes, and wonders about the rendering order. The main axes was created first, then the inset:

```python
fig = plt.figure(figsize=(8, 6))
ax_main  = fig.add_axes([0.1, 0.1, 0.8, 0.8])   # Created first
ax_inset = fig.add_axes([0.6, 0.6, 0.25, 0.25]) # Created second
```

The inset partially overlaps the main axes. Which axes appears visually on top?

- [ ] A) `ax_main` — it is the primary axes and always renders on top
- [ ] B) `ax_inset` — axes created later in the figure are rendered on top of earlier ones (higher z-order by default)
- [ ] C) Neither — they are rendered side by side without overlap
- [ ] D) The axes with the smaller area is always drawn on top

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax_inset` — axes created later in the figure are rendered on top of earlier ones (higher z-order by default)

**Explanation:**

- **A is incorrect:** Creation order, not primacy, determines default rendering order. `ax_main` was created first and thus has a lower default z-order.
- **B is correct:** Matplotlib renders `Axes` objects in the order they were added to the figure. `ax_inset` (created second) has a higher default z-order and renders **on top of** `ax_main`. This is why inset axes placed inside a main axes appear in front — they are layered above the main axes in the rendering stack. To explicitly control rendering order, `ax.set_zorder(value)` can override defaults.
- **C is incorrect:** The two axes DO overlap in the figure coordinate space. `ax_inset` at `[0.6, 0.6, 0.25, 0.25]` sits within the region covered by `ax_main` at `[0.1, 0.1, 0.8, 0.8]`.
- **D is incorrect:** Z-order is determined by creation sequence (default), not by axes area. Size has no bearing on rendering priority.

</details>

---

### Question 70

A developer places an inset axes outside the main axes region:

```python
fig = plt.figure(figsize=(10, 6))
ax_main  = fig.add_axes([0.08, 0.10, 0.60, 0.85])
ax_right = fig.add_axes([0.72, 0.10, 0.25, 0.85])  # Separate side panel
```

The right edge of `ax_main` is at 0.08+0.60=0.68 and `ax_right` starts at 0.72 — with a 0.04 normalized gap. What is the physical gap width in inches for a `figsize=(10, 6)` figure?

- [ ] A) 0.04 inches
- [ ] B) 0.4 inches
- [ ] C) 4 inches
- [ ] D) 0.72 inches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.4 inches

**Explanation:**

- **A is incorrect:** 0.04 is the normalized gap (4% of figure width) — not the physical width in inches.
- **B is correct:** Physical gap = normalized gap × figure width = 0.04 × 10 inches = **0.4 inches**. The gap between the two axes panels is 0.4 inches of white space — enough to visually separate the two panels without wasting excessive space.
- **C is incorrect:** 4 inches would require a normalized gap of 0.4 (40% of the 10-inch figure), not 0.04.
- **D is incorrect:** 0.72 is the left POSITION of `ax_right`, not the gap between panels. The gap is calculated as `ax_right_left - ax_main_right = 0.72 - 0.68 = 0.04` normalized, which equals 0.4 physical inches.

</details>
