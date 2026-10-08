# Matplotlib Annotations & Legends — Test 10: Comprehensive ML Diagnostics Visualization

---

### Question 91

A machine learning engineer builds a loss diagnostics plot with annotations and reference lines:

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_loss, label="Train Loss", color="royalblue")
ax.plot(epochs, val_loss, label="Val Loss", color="tomato")

# Early stopping marker
early_stop = 47
ax.axvline(x=early_stop, color="gray", linestyle="--", lw=1, label="Early Stop")
ax.annotate("Best Val Loss",
            xy=(early_stop, val_loss[early_stop]),
            xytext=(early_stop + 8, val_loss[early_stop] + 0.05),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9)

ax.legend(loc="upper right")
```

Which component of this code ensures that the annotation arrow tip touches the actual minimum validation loss point?

- [ ] A) `xytext=(early_stop + 8, val_loss[early_stop] + 0.05)` — the text position defines the annotation target
- [ ] B) `xy=(early_stop, val_loss[early_stop])` — this data coordinate is the arrow tip and annotation target
- [ ] C) `arrowprops=dict(arrowstyle="->")` — the arrow automatically finds the nearest data minimum
- [ ] D) `ax.axvline(x=early_stop, ...)` — the vertical line identifies the epoch and the annotation snaps to it

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `xy=(early_stop, val_loss[early_stop])` — this data coordinate is the arrow tip and annotation target

**Explanation:**

- **A is incorrect:** `xytext` specifies where the text label and arrow tail are placed, not the annotation target. The text appears at epoch 55 and loss + 0.05, not at the minimum.
- **B is correct:** In `ax.annotate`, `xy` is the target coordinate — the point of interest being annotated. `xy=(early_stop, val_loss[early_stop])` precisely specifies the (epoch, loss) data coordinate at the early stop epoch. The arrow tip is positioned at this exact data point. The code correctly uses `val_loss[early_stop]` (the actual loss value at epoch `early_stop`) to pin the arrow to the exact curve position.
- **C is incorrect:** `arrowprops` configures the visual appearance of the arrow (style, colour) but performs no data analysis. The arrow does not search for minima — it connects `xytext` to `xy` as specified.
- **D is incorrect:** `ax.axvline` draws a vertical reference line independently of the annotation. The annotation does not "snap" to the vertical line. The annotation target is fully determined by the `xy` argument.

</details>

---

### Question 92

A data analyst annotates an ROC curve with the AUC value and the random-chance baseline:

```python
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance (AUC = 0.5)")
ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue", label="AUC Region")

ax.text(0.95, 0.05,
        f"AUC = {auc:.4f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11, fontweight="bold",
        bbox=dict(fc="white", ec="steelblue", pad=0.3, alpha=0.9))
```

Why is `transform=ax.transAxes` used for the AUC text stamp rather than data coordinates?

- [ ] A) ROC curve axes always range from 0 to 1, so `transAxes` and `transData` produce the same result in this specific case
- [ ] B) `transAxes` keeps the stamp fixed in the bottom-right corner regardless of any axis limit adjustments, while data coordinates might shift the label if limits are set differently from [0, 1]
- [ ] C) `transAxes` is required whenever `bbox` is specified in `ax.text`
- [ ] D) ROC data coordinates are on a different scale, so `transData` cannot place text near (0.95, 0.05)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `transAxes` keeps the stamp fixed in the bottom-right corner regardless of any axis limit adjustments, while data coordinates might shift the label if limits are set differently from [0, 1]

**Explanation:**

- **A is incorrect:** While ROC axes are commonly set to [0, 1] × [0, 1], making `transAxes` and `transData` equivalent in those exact limits, this is coincidental — not a reliable equivalence. If axis padding is added (`ax.set_xlim(-0.05, 1.05)`), data coordinate (0.95, 0.05) would no longer be in the bottom-right corner, but axes-fraction (0.95, 0.05) always would be.
- **B is correct:** `transform=ax.transAxes` is the robust choice because the axes-fraction coordinates (0.95, 0.05) are always 95% of the axes width from the left and 5% from the bottom, regardless of axis limits. If a developer later adds `ax.set_xlim(-0.05, 1.1)` for visual margin, a data-coordinate label would shift. The `transAxes` stamp stays visually fixed in the corner — the intended behavior for a metric card.
- **C is incorrect:** `bbox` can be used with any transform including `transData`. The `bbox` argument is independent of the coordinate transform.
- **D is incorrect:** Standard ROC curves use [0, 1] × [0, 1] data ranges, so (0.95, 0.05) is perfectly within data range. `transData` would render the text in the lower-right area of the actual data range, close to but not necessarily identical to the `transAxes` position.

</details>

---

### Question 93

A researcher creates a training diagnostics dashboard with annotations for multiple key events:

```python
events = {
    "LR Decay": 25,
    "Plateau Detected": 38,
    "Early Stop": 52
}

for label, epoch in events.items():
    ax.axvline(x=epoch, color="dimgray", lw=0.8, linestyle=":")
    ax.text(epoch + 0.5, ax.get_ylim()[1] * 0.98,
            label,
            fontsize=7,
            va="top",
            ha="left",
            rotation=90,
            color="dimgray")
```

Why is `rotation=90` used for these labels?

- [ ] A) Rotating 90° converts the label from data coordinates to axes coordinates
- [ ] B) Vertical text (rotated 90°) allows narrow epoch labels to be placed directly adjacent to vertical reference lines without occupying wide horizontal space on the x-axis
- [ ] C) Matplotlib requires `rotation=90` for all text placed near `axvline` markers
- [ ] D) `rotation=90` applies only to the bounding box, keeping the text characters horizontal

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Vertical text (rotated 90°) allows narrow epoch labels to be placed directly adjacent to vertical reference lines without occupying wide horizontal space on the x-axis

**Explanation:**

- **A is incorrect:** `rotation` is a visual rendering parameter that controls the angle of text characters. It has no effect on the coordinate system used to position the text.
- **B is correct:** Vertical reference lines (`axvline`) are narrow (1–2 pixels wide) and positioned at specific x-values. Horizontal text labels next to these lines require significant horizontal space, potentially overlapping with adjacent events or data. Rotating text 90° makes it vertical — the text height now extends along the y-axis instead of the x-axis, consuming very little horizontal space. Multiple vertical labels can sit side by side near adjacent reference lines without crowding each other. This is a standard technique in time-series event annotation (training logs, heartbeat markers, etc.).
- **C is incorrect:** `rotation=90` is not a Matplotlib requirement for text near `axvline`. It is a stylistic choice made by the developer for space efficiency.
- **D is incorrect:** `rotation=90` rotates the entire text artist — including the characters themselves — 90° counter-clockwise. Both the bounding box and the characters are rotated together.

</details>

---

### Question 94

A machine learning engineer adds overfitting and underfitting zone annotations to an accuracy curve:

```python
ax.axvspan(0, 15, alpha=0.1, facecolor="orange", label="Underfitting Zone")
ax.axvspan(50, 100, alpha=0.1, facecolor="red", label="Overfitting Zone")
ax.axvspan(15, 50, alpha=0.05, facecolor="green", label="Optimal Zone")

ax.legend(loc="lower right", framealpha=0.9)
```

A colleague suggests replacing all three `axvspan` calls with a single `ax.fill_between`. Why would `fill_between` be less appropriate here?

- [ ] A) `fill_between` requires the same facecolor for all regions and cannot handle multiple colours
- [ ] B) `axvspan` spans the full axes height automatically; `fill_between` requires explicit y-value arrays for the upper and lower bounds of the shaded region
- [ ] C) `fill_between` cannot add labels for the legend
- [ ] D) `fill_between` only works with continuous data arrays, not epoch integers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `axvspan` spans the full axes height automatically; `fill_between` requires explicit y-value arrays for the upper and lower bounds of the shaded region

**Explanation:**

- **A is incorrect:** `fill_between` supports different colours for different calls and can accept `where` conditions to restrict shading. Multiple colours are achievable with multiple calls or conditional `where` logic.
- **B is correct:** `axvspan(xmin, xmax)` creates a vertical band that automatically spans the full height of the axes from the bottom wall to the top wall, regardless of the y-axis range. To achieve the same full-height shading with `fill_between`, the analyst would need to explicitly provide `y1=ax.get_ylim()[0]` and `y2=ax.get_ylim()[1]` as arrays or constants, AND ensure these limits don't change with autoscaling. `axvspan` is purpose-built for full-height vertical region shading and is more concise and robust for this use case.
- **C is incorrect:** `fill_between` fully supports the `label` argument and produces legend entries for the shaded polygon.
- **D is incorrect:** Both `fill_between` and `axvspan` accept scalar x-values. Integer epochs are valid inputs for both functions.

</details>

---

### Question 95

A researcher builds a multi-class classification report visualization with a shared legend below the subplots:

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261"]
labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

for ax, metric in zip(axes.flat, metrics):
    for i, (cls_data, color, lbl) in enumerate(zip(data, colors, labels)):
        ax.plot(epochs, cls_data[metric], color=color, label=lbl if ax == axes[0, 0] else "_")

handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls,
           loc="lower center",
           bbox_to_anchor=(0.5, -0.05),
           ncol=5,
           frameon=True,
           bbox_transform=fig.transFigure)
plt.tight_layout()
```

Why does the code use `label=lbl if ax == axes[0, 0] else "_"`?

- [ ] A) To apply a different colour to the first subplot's lines
- [ ] B) To prevent duplicate legend entries — only the first subplot's lines receive proper labels; all other subplots' lines get underscore-prefixed labels that are excluded from legend collection
- [ ] C) To make the legend only reflect the first subplot's data
- [ ] D) To ensure `fig.legend` collects handles from all subplots automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) To prevent duplicate legend entries — only the first subplot's lines receive proper labels; all other subplots' lines get underscore-prefixed labels that are excluded from legend collection

**Explanation:**

- **A is incorrect:** `label=lbl if ax == axes[0, 0] else "_"` is a conditional label assignment, not a colour modifier. The colour is set by the `color=color` argument, which is the same regardless of the condition.
- **B is correct:** Each subplot plots 5 class curves. If all 30 curve artists (5 per subplot × 6 subplots) received proper labels, calling `axes[0, 0].get_legend_handles_labels()` would collect 5 handles from the first subplot, but `fig.legend` would potentially show duplicate entries if it scanned all subplots. By assigning proper labels only on `axes[0, 0]`, the `handles, lbls = axes[0, 0].get_legend_handles_labels()` retrieves exactly 5 clean handles. The underscore-prefixed `"_"` labels on all other subplots are excluded from auto-legend collection, preventing duplicates in the shared legend.
- **C is incorrect:** The figure-level legend accurately represents all subplots because all subplots use the same colours and represent the same 5 classes. The first subplot's handles serve as representative samples for the shared legend. The label filtering is a deduplication technique, not a "first subplot only" semantic restriction.
- **D is incorrect:** `fig.legend` does auto-collect from all subplots when called without explicit handles, but this would produce 30 handles (5 per subplot × 6 subplots), creating 5 duplicate entries in the legend. The explicit `handles, lbls` extraction from `axes[0, 0]` and the underscore-label filtering work together to give exactly 5 entries.

</details>

---

### Question 96

A data scientist annotates a precision-recall curve with the F1-optimal operating point:

```python
best_idx = np.argmax(f1_scores)
best_recall = recall[best_idx]
best_precision = precision[best_idx]

ax.scatter([best_recall], [best_precision], s=120, zorder=5,
           color="red", marker="*", label=f"Best F1 (Threshold={thresholds[best_idx]:.2f})")
ax.annotate(f"F1 = {f1_scores[best_idx]:.3f}",
            xy=(best_recall, best_precision),
            xytext=(best_recall - 0.15, best_precision - 0.08),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
            fontsize=9, color="darkred")
```

Why is `zorder=5` set on the scatter point?

- [ ] A) `zorder=5` makes the marker visible only in the legend, not on the main plot
- [ ] B) The scatter marker is rendered on top of the precision-recall curve and any other lower-`zorder` artists, preventing the star from being obscured by the plotted line
- [ ] C) `zorder=5` changes the marker shape from circle to star
- [ ] D) `zorder=5` is required for `marker="*"` to render correctly

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The scatter marker is rendered on top of the precision-recall curve and any other lower-`zorder` artists, preventing the star from being obscured by the plotted line

**Explanation:**

- **A is incorrect:** `zorder` controls the rendering depth of the artist on the axes, not any special legend-only visibility mode.
- **B is correct:** Default `zorder` for `ax.plot` lines is approximately 2. A scatter marker with `zorder=5` renders above any artist with `zorder < 5`. Since the precision-recall curve is a `Line2D` at default zorder (~2), the star marker at `zorder=5` is rendered on top of the curve line, ensuring it is always visible even when placed exactly on the curve. Without this, the curve might partially obscure the marker depending on rendering order.
- **C is incorrect:** The marker shape (`"*"` for star) is set by `marker="*"`. `zorder` is a purely rendering-order parameter with no effect on visual appearance.
- **D is incorrect:** `marker="*"` is a valid Matplotlib marker specification that works at any `zorder` value, including the default. No special `zorder` setting is required for star markers.

</details>

---

### Question 97

A machine learning engineer adds two separate legends to a single axes — one for model curves and one for reference lines — and encounters the two-legend overwrite problem. He applies the fix:

```python
model_legend = ax.legend(handles=model_handles, labels=model_labels,
                         loc="upper right", title="Models")
ax.add_artist(model_legend)

ref_legend = ax.legend(handles=ref_handles, labels=ref_labels,
                       loc="lower right", title="Reference Lines",
                       framealpha=0.7)
```

After this code, both legends appear. Later he adds `ax.set_title("Training Diagnostics")` and calls `plt.tight_layout()`. Does `tight_layout()` account for both legends in its spacing calculation?

- [ ] A) Yes — `tight_layout` always accounts for all artists including both legends
- [ ] B) No — `tight_layout` typically ignores legends positioned inside the axes bounding box; it only accounts for axis labels, titles, and tick labels in its spacing calculation
- [ ] C) Yes — but only for legends added via `add_artist`, not for the standard `ax.legend()`
- [ ] D) No — `tight_layout` ignores all legend objects regardless of position

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `tight_layout` typically ignores legends positioned inside the axes bounding box; it only accounts for axis labels, titles, and tick labels in its spacing calculation

**Explanation:**

- **A is incorrect:** `tight_layout` does not account for all artists. Specifically, it ignores most legend objects positioned inside the axes (the more common case), which is why external legends placed beyond the axes boundary are frequently clipped in saved figures.
- **B is correct:** `plt.tight_layout()` adjusts subplot padding to prevent overlap of axis labels, tick labels, titles, and subplot borders. It does not compute or account for the space occupied by legend objects positioned within or adjacent to the axes. Internal legends (within the [0, 1] × [0, 1] axes area) may overlap with subplots in dense figures. External legends (beyond the axes boundary) are even more likely to be cut off by `tight_layout`'s layout constraints. This is why `plt.savefig("fig.png", bbox_inches="tight")` or `fig.subplots_adjust()` is needed for external legends.
- **C is incorrect:** `tight_layout` does not distinguish between legends added via `ax.legend()` and those added via `ax.add_artist()`. Both are treated the same way (generally ignored in layout calculations).
- **D is incorrect:** `tight_layout` is not entirely oblivious to all legends — in some Matplotlib versions and configurations it makes partial accommodations. The accurate statement is that it primarily accounts for axis/title/tick elements, and legend handling is incomplete or unreliable.

</details>

---

### Question 98

A researcher creates a comprehensive confusion matrix heatmap with annotations for a 5-class classifier. She adds the class accuracy diagonal with:

```python
for i in range(5):
    ax.text(i, i, f"{cm[i, i]}\n({cm[i, i]/cm[i, :].sum()*100:.1f}%)",
            ha="center", va="center",
            fontsize=9,
            fontweight="bold",
            color="white" if cm[i, i] < cm.max() * 0.5 else "black")
```

What does the conditional `color="white" if cm[i, i] < cm.max() * 0.5 else "black"` achieve?

- [ ] A) It makes low-value cells white and high-value cells black — creating a threshold-based contrast inversion that ensures readability against the heatmap background colour
- [ ] B) It sets all cell text to white if the diagonal is below average, or black otherwise
- [ ] C) It applies white text to the header row and black text to data rows
- [ ] D) It is a syntax error — conditional expressions cannot be used inside `ax.text` `color` arguments

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It makes low-value cells white and high-value cells black — creating a threshold-based contrast inversion that ensures readability against the heatmap background colour

**Explanation:**

- **A is correct:** This is a contrast-adaptive text colouring technique. In a heatmap, dark cells (low values on a light-to-dark colormap, or high values on a dark colormap) need light-coloured text for legibility, and light cells need dark text. The condition `cm[i, i] < cm.max() * 0.5` checks if the diagonal value is below 50% of the maximum. Below the threshold → `color="white"` (for readability on dark cells); above the threshold → `color="black"` (for readability on light cells). This ensures text is legible against any heatmap cell background.
- **B is incorrect:** The comparison is against `cm.max() * 0.5` (half the maximum value), not the mean. These are different thresholds.
- **C is incorrect:** There is no header/data row distinction in the confusion matrix cell loop. The same logic applies to every diagonal cell.
- **D is incorrect:** Python's inline conditional expression (`x if condition else y`) is a standard Python expression and is fully valid as a value for any keyword argument, including `color=` in `ax.text()`.

</details>

---

### Question 99

A data scientist builds a model comparison radar chart with a shared external legend and 6 model curves. She uses:

```python
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for model_name, scores, color in zip(model_names, all_scores, palette):
    ax.plot(angles, scores, label=model_name, color=color, lw=2)
    ax.fill(angles, scores, alpha=0.1, color=color)

ax.legend(loc="upper left",
          bbox_to_anchor=(-0.15, 1.15),
          fontsize=10,
          ncol=2,
          title="Models",
          frameon=True)
plt.savefig("radar.pdf", bbox_inches="tight")
```

Why is `bbox_to_anchor=(-0.15, 1.15)` used for a polar axes?

- [ ] A) Polar axes have no valid `loc` strings, so negative `bbox_to_anchor` values compensate for this limitation
- [ ] B) For polar axes, the axes bounding box is a square enclosing the circle; negative x and y>1 values in `bbox_to_anchor` position the legend in the upper-left corner outside the circular plot area
- [ ] C) `-0.15` in `bbox_to_anchor` disables automatic legend placement
- [ ] D) `1.15` in `bbox_to_anchor` places the legend 115% of the figure height above the axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) For polar axes, the axes bounding box is a square enclosing the circle; negative x and y>1 values in `bbox_to_anchor` position the legend in the upper-left corner outside the circular plot area

**Explanation:**

- **A is incorrect:** Polar axes support all standard `loc` string values. The positioning challenge is physical — the circular plot occupies the full bounding box, so any interior legend position overlaps the data.
- **B is correct:** Polar axes (`subplot_kw=dict(polar=True)`) are rendered within a square bounding box in which the circle is inscribed. The corners of the axes bounding box contain the "triangular" negative space outside the circle. Using `bbox_to_anchor=(-0.15, 1.15)` in axes-fraction coordinates — specifically x=-0.15 (15% to the left of the axes left edge) and y=1.15 (15% above the axes top edge) — positions the legend's anchor point in the upper-left area outside the circular plot. This prevents the legend from overlapping with the circular data visualization.
- **C is incorrect:** Negative values in `bbox_to_anchor` do not disable automatic placement. They simply position the anchor point to the left of (for negative x) or above (for y>1) the axes bounding box.
- **D is incorrect:** `bbox_to_anchor` coordinates are in axes-fraction space by default (unless `bbox_transform` is overridden). `1.15` means 115% of the **axes height** from the bottom, not of the figure height.

</details>

---

### Question 100

A machine learning engineer finalizes a publication-quality training diagnostics figure combining all annotation techniques:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss subplot
ax1.plot(epochs, train_loss, color="#2196F3", lw=2, label="Train Loss")
ax1.plot(epochs, val_loss, color="#F44336", lw=2, label="Val Loss")
ax1.axvspan(overfit_start, epochs[-1], facecolor="red", alpha=0.07,
            label="Overfitting")
ax1.axvline(x=best_epoch, color="green", lw=1.5, linestyle="--",
            label="Best Epoch")
ax1.annotate(f"Min Val: {min_val_loss:.4f}",
             xy=(best_epoch, min_val_loss),
             xytext=(best_epoch + 5, min_val_loss + 0.03),
             arrowprops=dict(arrowstyle="->", color="green"))
ax1.text(0.02, 0.98, "Loss Curve", transform=ax1.transAxes,
         va="top", fontsize=10, fontweight="bold")

# ROC subplot
ax2.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc:.3f}")
ax2.plot([0, 1], [0, 1], "k--", label="Chance")
ax2.text(0.97, 0.05, f"AUC = {auc:.4f}", transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=11, fontweight="bold",
         bbox=dict(fc="white", ec="steelblue", pad=0.3))

for ax in (ax1, ax2):
    ax.legend(loc="upper right", framealpha=0.9)

plt.suptitle("Model Training Report", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("report.pdf", bbox_inches="tight")
```

Why is `rect=[0, 0, 1, 0.95]` passed to `tight_layout()`?

- [ ] A) It clips the bottom 5% of the figure, removing axis labels
- [ ] B) It constrains `tight_layout`'s layout area to the bottom 95% of the figure height, leaving the top 5% as space for `plt.suptitle("Model Training Report", ...)` to render without overlap with the subplots
- [ ] C) It scales both subplots to 95% of their original size
- [ ] D) It sets the left and right margins each to 5% of the figure width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It constrains `tight_layout`'s layout area to the bottom 95% of the figure height, leaving the top 5% as space for `plt.suptitle("Model Training Report", ...)` to render without overlap with the subplots

**Explanation:**

- **A is incorrect:** `rect=[0, 0, 1, 0.95]` defines the layout region as `[left, bottom, right, top]` in figure fractions. The bottom boundary is 0 (no clipping from the bottom). The `0.95` is the top boundary.
- **B is correct:** `plt.tight_layout(rect=[left, bottom, right, top])` constrains the subplot packing to the specified rectangular region of the figure. The tuple `[0, 0, 1, 0.95]` means: left=0 (full left), bottom=0 (full bottom), right=1 (full right), top=0.95 (stop 5% from the top). The subplots are packed within this constrained area, leaving a 5% strip at the top of the figure for `plt.suptitle(...)` to appear without overlapping the subplot titles or content. Without this `rect`, `tight_layout` would pack subplots to the full figure height, causing `suptitle` to overlap with the subplot tops.
- **C is incorrect:** The `rect` parameter defines the region for layout, not a scaling factor. Both subplots fill the `[0, 0, 1, 0.95]` region as large as possible subject to padding constraints.
- **D is incorrect:** The `rect` format is `[left, bottom, right, top]`, not `[left_margin, right_margin, ...]`. Left=0 and right=1 mean the subplots span the full horizontal width (no horizontal margins). The 5% gap is in the vertical direction (top).

</details>
