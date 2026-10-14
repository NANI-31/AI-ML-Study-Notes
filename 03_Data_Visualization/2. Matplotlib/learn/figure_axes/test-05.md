# Matplotlib Figure & Axes Objective Test — Set 05
# Topic: Layout Optimization — tight_layout & constrained_layout (Questions 41–50)

---

### Question 41

A developer creates a 3×2 subplot grid and finds that tick labels on the left column overlap with titles of the right column. Which call correctly fixes the overlap?

```python
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
# ... plotting code ...
# Fix here:
plt.tight_layout()
```

What does `plt.tight_layout()` do internally?

- [ ] A) It increases the figure size to fit all content without any overlap
- [ ] B) It adjusts the subplot spacing (`hspace`, `wspace`, top/bottom/left/right margins) so that all axes labels, titles, and tick marks fit within the figure without clipping or overlapping
- [ ] C) It removes overlapping labels by deleting duplicate tick marks
- [ ] D) It converts all text elements to a smaller font size until everything fits

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It adjusts the subplot spacing (`hspace`, `wspace`, top/bottom/left/right margins) so that all axes labels, titles, and tick marks fit within the figure without clipping or overlapping

**Explanation:**

- **A is incorrect:** `tight_layout()` does NOT resize the figure. The `figsize` remains unchanged. It redistributes the available space within the existing figure dimensions by adjusting subplot padding parameters.
- **B is correct:** `plt.tight_layout()` uses a bounding box computation engine to determine the actual rendered extents of all text elements (tick labels, axis labels, titles) and then adjusts `subplots_adjust()` parameters (`left`, `right`, `top`, `bottom`, `hspace`, `wspace`) to prevent any two artists from overlapping. It is the most commonly used automatic layout engine in Matplotlib.
- **C is incorrect:** `tight_layout()` never deletes or truncates content. It only repositions the subplot frames, not modifies or removes text content.
- **D is incorrect:** Font sizes are never modified by `tight_layout()`. The text content and styling remain unchanged; only subplot geometry is adjusted.

</details>

---

### Question 42

A developer passes keyword arguments to `fig.tight_layout()` to control spacing:

```python
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
# ... plotting ...
fig.tight_layout(pad=2.0, w_pad=1.5, h_pad=2.5)
```

What does each parameter control?

- [ ] A) `pad` = overall border padding; `w_pad` = horizontal spacing between subplots; `h_pad` = vertical spacing between subplots (all in units of font size)
- [ ] B) `pad` = DPI adjustment; `w_pad` = figure width; `h_pad` = figure height
- [ ] C) `pad` = number of subplots per row; `w_pad` = whitespace before labels; `h_pad` = header padding
- [ ] D) All three control the same outer margin — only one is needed at a time

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `pad` = overall border padding; `w_pad` = horizontal spacing between subplots; `h_pad` = vertical spacing between subplots (all in units of font size)

**Explanation:**

- **A is correct:** In `tight_layout(pad, w_pad, h_pad)`: `pad` sets the overall padding between the figure edge and the outermost subplot elements (the outer margin, in units of font-size). `w_pad` sets the horizontal padding **between** adjacent subplots (column spacing). `h_pad` sets the vertical padding **between** adjacent subplots (row spacing). All three are measured in font-size units (not inches or pixels), which makes them scale with the text size of the figure.
- **B is incorrect:** None of these parameters control DPI or figure dimensions. They are all spacing/padding controls within the existing figure space.
- **C is incorrect:** The number of subplots per row is controlled by `plt.subplots(nrows, ncols)`, not by layout parameters. `w_pad` and `h_pad` are spacing values, not content counts.
- **D is incorrect:** The three parameters serve distinct geometric purposes: `pad` controls edge padding; `w_pad` and `h_pad` control inter-subplot spacing in horizontal and vertical directions respectively.

</details>

---

### Question 43

A developer creates a figure with `constrained_layout=True` as an alternative to `tight_layout`:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
```

What is the key advantage of `constrained_layout=True` over calling `plt.tight_layout()` at the end?

- [ ] A) `constrained_layout` increases figure resolution automatically
- [ ] B) `constrained_layout` is a continuous layout solver that updates automatically as elements are added — `tight_layout()` is a one-time call that doesn't adjust if content changes after it's called
- [ ] C) `constrained_layout` is faster for small figures; `tight_layout` is faster for large figures
- [ ] D) `constrained_layout` supports 3D plots; `tight_layout` does not

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `constrained_layout` is a continuous layout solver that updates automatically as elements are added — `tight_layout()` is a one-time call that doesn't adjust if content changes after it's called

**Explanation:**

- **A is incorrect:** Neither layout engine affects figure resolution or DPI.
- **B is correct:** `constrained_layout=True` activates a **continuous constraint-based solver** that recalculates layout every time the figure is drawn (on each render event). If you add a colorbar, a legend, or more tick labels after setting up the figure, `constrained_layout` automatically adapts. In contrast, `plt.tight_layout()` is a **one-time call** — it computes and applies spacing at the moment it is called. If you add elements after that call, the layout is not re-adjusted. `constrained_layout` is generally more robust for complex, dynamic figures.
- **C is incorrect:** Performance characteristics are not a meaningful differentiation. Both operate within normal Matplotlib rendering time — the difference is behavioral (continuous vs. one-time), not a speed tradeoff.
- **D is incorrect:** Neither layout engine has specific 3D plot support differences. Both have limitations with 3D axes (`Axes3D`), but this is not a distinguishing advantage of `constrained_layout`.

</details>

---

### Question 44

A developer creates a figure but uses `tight_layout` inside `plt.figure()` at creation:

```python
fig = plt.figure(figsize=(10, 6), tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
```

Is this valid? What does `tight_layout=True` inside `plt.figure()` do?

- [ ] A) It raises a `TypeError` — `tight_layout` is not a valid `plt.figure()` parameter
- [ ] B) It is valid — it enables the tight layout engine for the figure at creation time, equivalent to calling `fig.set_tight_layout(True)`, so layout is automatically adjusted on each draw
- [ ] C) It is valid but only applies to the first subplot added
- [ ] D) It applies tight_layout once immediately at creation, before any subplots are added — so it has no effect

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It is valid — it enables the tight layout engine for the figure at creation time, equivalent to calling `fig.set_tight_layout(True)`, so layout is automatically adjusted on each draw

**Explanation:**

- **A is incorrect:** `tight_layout=True` IS a valid `plt.figure()` parameter. It's documented in Matplotlib's API as a way to enable the tight layout engine at figure creation.
- **B is correct:** Passing `tight_layout=True` to `plt.figure()` activates the tight layout engine on the figure object — equivalent to `fig.set_tight_layout(True)`. This means layout is automatically adjusted before each render (similar to how `constrained_layout=True` works). It's a convenient alternative to calling `plt.tight_layout()` at the end of every plotting script.
- **C is incorrect:** The layout engine applies to all subplots in the figure, not just the first one. It recomputes layout for all axes present at render time.
- **D is incorrect:** The layout is applied **before each draw** (not just at creation). Since no axes exist at creation time, the meaningful application happens when the figure is rendered after subplots are added.

</details>

---

### Question 45

A developer uses `tight_layout` with a `rect` parameter to reserve space for a suptitle:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Hyperparameter Search Results", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
```

What does `rect=[0, 0, 1, 0.95]` specify?

- [ ] A) The x-position, y-position, width, and height of the first subplot in data coordinates
- [ ] B) The normalized figure rectangle `[left, bottom, right, top]` within which tight_layout constrains all subplots — leaving the top 5% for the suptitle
- [ ] C) The pixel coordinates of the figure border: left=0, bottom=0, right=1px, top=0.95px
- [ ] D) The margins in inches: 0 inches left, 0 inches bottom, 1 inch right, 0.95 inches top

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The normalized figure rectangle `[left, bottom, right, top]` within which tight_layout constrains all subplots — leaving the top 5% for the suptitle

**Explanation:**

- **A is incorrect:** `rect` is not about a single subplot's position in data coordinates. It constrains the entire subplot layout region in normalized figure coordinates.
- **B is correct:** `rect=[left, bottom, right, top]` defines a normalized figure coordinate rectangle `[0, 0, 1, 0.95]` — meaning tight_layout fits all subplots into the region: left=0 (figure left edge), bottom=0 (figure bottom), right=1 (figure right edge), top=0.95 (95% up the figure height). The top 5% (from y=0.95 to y=1.0) is left empty — providing space for `fig.suptitle()` to render without overlapping the subplots. This is the standard pattern for combining `suptitle` with `tight_layout`.
- **C is incorrect:** `rect` uses normalized (0–1) figure coordinates, not pixel coordinates. The values 0 and 1 don't represent pixels.
- **D is incorrect:** The `rect` parameter uses fractional (normalized) coordinates, not physical inch measurements.

</details>

---

### Question 46

A developer creates a figure with long y-axis labels that are getting clipped at the left edge:

```python
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.000 ms", "0.250 ms", "0.500 ms", "0.750 ms", "1.000 ms"])
ax.set_ylabel("Inference Latency (milliseconds)")
```

The y-axis labels are being cut off in the saved PNG. What is the most direct fix?

- [ ] A) Reduce the font size of the y-tick labels
- [ ] B) Add `plt.tight_layout()` before `savefig()` or use `savefig(..., bbox_inches="tight")`
- [ ] C) Manually increase `figsize` width to accommodate the labels
- [ ] D) Use shorter tick label strings

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add `plt.tight_layout()` before `savefig()` or use `savefig(..., bbox_inches="tight")`

**Explanation:**

- **A is incorrect:** Reducing font size is a cosmetic workaround that degrades readability. The proper fix adjusts layout geometry, not content size.
- **B is correct:** Long y-axis labels extend beyond the left edge of the axes bounding box. `plt.tight_layout()` automatically expands the left margin to accommodate the widest tick label, preventing clipping. Alternatively, `savefig(..., bbox_inches="tight")` computes a tight bounding box that includes all artists (including labels that extend outside the default figure boundary) and saves the full content. Both are standard production fixes for label clipping.
- **C is incorrect:** While manually increasing `figsize` can help, it's an imprecise approach — you'd have to guess how much extra width is needed. `tight_layout` or `bbox_inches="tight"` compute the required space automatically.
- **D is incorrect:** Using shorter labels to fit the figure is a content compromise, not a layout fix. The correct approach is to adjust the layout to fit the existing content.

</details>

---

### Question 47

A developer compares the behavior of `plt.tight_layout()` and `plt.subplots_adjust()`:

```python
# Approach A:
plt.tight_layout(pad=1.5)

# Approach B:
plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08,
                    hspace=0.4, wspace=0.3)
```

What is the fundamental difference between these two approaches?

- [ ] A) `tight_layout` only works with `plt.subplots()`; `subplots_adjust` works with `add_axes` too
- [ ] B) `tight_layout` automatically calculates required spacing based on actual rendered text extents; `subplots_adjust` requires manual specification of all margin values in normalized figure coordinates
- [ ] C) `subplots_adjust` is deprecated in favor of `tight_layout`
- [ ] D) They produce identical results — `tight_layout` just automates `subplots_adjust` with default values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `tight_layout` automatically calculates required spacing based on actual rendered text extents; `subplots_adjust` requires manual specification of all margin values in normalized figure coordinates

**Explanation:**

- **A is incorrect:** Both `tight_layout` and `subplots_adjust` work with standard subplot grids. `tight_layout` has limited support for axes created with `add_axes`, but this is not the core distinction.
- **B is correct:** `plt.tight_layout()` **automatically computes** the minimum required margins by measuring the actual pixel extents of all text elements (tick labels, axis labels, titles) and calculating the subplot geometry needed to avoid overlap. `plt.subplots_adjust()` is a **manual API** — the developer specifies exact normalized coordinates for each margin. This requires trial-and-error and breaks when content changes (e.g., longer labels). `tight_layout` is adaptive; `subplots_adjust` is static.
- **C is incorrect:** `subplots_adjust` is not deprecated. It is still valid and useful when you need precise, reproducible layout control (e.g., for publication templates where exact margins are specified).
- **D is incorrect:** `tight_layout` does not simply apply default values to `subplots_adjust`. It performs a bounding box computation pass and derives the required margins from the actual artist extents — the result varies by figure content.

</details>

---

### Question 48

A developer creates a figure with `constrained_layout=True` and then tries to manually adjust spacing:

```python
fig, axes = plt.subplots(2, 2, constrained_layout=True)
plt.subplots_adjust(hspace=0.5)  # Manual adjustment
```

What happens?

- [ ] A) The `hspace=0.5` override is applied successfully on top of constrained layout
- [ ] B) A `UserWarning` is raised: "tight_layout not applied because axes sizes collapsed to zero" or a similar warning that the layout engine conflicts with manual adjustment, and the result may be undefined
- [ ] C) `constrained_layout` is automatically disabled when `subplots_adjust` is called
- [ ] D) `subplots_adjust` is silently ignored when `constrained_layout=True` is active

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `constrained_layout` is automatically disabled when `subplots_adjust` is called

**Explanation:**

- **A is incorrect:** The two layout systems conflict. Applying `subplots_adjust` while `constrained_layout` is active is not supported as a cumulative operation.
- **B is incorrect:** While warnings may appear in some Matplotlib versions, the primary behavior in modern Matplotlib is that calling `subplots_adjust` disables `constrained_layout`.
- **C is correct:** In Matplotlib's documentation and implementation, calling `plt.subplots_adjust()` (or `fig.subplots_adjust()`) while `constrained_layout=True` is active will issue a `UserWarning` and **disable the constrained layout engine** for that figure. The two layout systems are incompatible because they both try to control the same subplot geometry parameters. The correct approach is to use one or the other — not both simultaneously.
- **D is incorrect:** `subplots_adjust` is NOT silently ignored. It actively conflicts with and disables `constrained_layout`.

</details>

---

### Question 49

A developer discovers that `plt.tight_layout()` does not work correctly with axes created using `fig.add_axes()`:

```python
fig = plt.figure(figsize=(8, 6))
ax_main = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax_inset = fig.add_axes([0.6, 0.6, 0.25, 0.25])
plt.tight_layout()  # Does this work?
```

Why does `tight_layout()` have limited effectiveness here?

- [ ] A) `tight_layout` only adjusts the first axes in the figure
- [ ] B) `tight_layout` is designed for `subplot`-grid axes (those created with `add_subplot` or `subplots`); axes created with `add_axes` use arbitrary normalized coordinates and are not part of the subplot grid system that `tight_layout` optimizes
- [ ] C) `tight_layout` doesn't work when `figsize` is larger than `(6, 4)`
- [ ] D) `add_axes` creates axes in pixel coordinates, which `tight_layout` cannot interpret

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `tight_layout` is designed for `subplot`-grid axes (those created with `add_subplot` or `subplots`); axes created with `add_axes` use arbitrary normalized coordinates and are not part of the subplot grid system that `tight_layout` optimizes

**Explanation:**

- **A is incorrect:** `tight_layout` attempts to adjust all subplot-grid axes simultaneously — not just the first one.
- **B is correct:** `plt.tight_layout()` operates on the **subplot grid system** — axes created via `add_subplot()` or `plt.subplots()`. These have grid positions (row/column indices) that `tight_layout` can reorganize. Axes created with `fig.add_axes([left, bottom, width, height])` are positioned using absolute normalized figure coordinates, outside the subplot grid system. `tight_layout` does not attempt to adjust these and may emit a `UserWarning` indicating it cannot handle non-subplot axes. For arbitrary layouts with `add_axes`, layout must be manually managed.
- **C is incorrect:** Figure size has no bearing on `tight_layout` functionality.
- **D is incorrect:** `add_axes` uses normalized figure coordinates (0–1 range), not pixel coordinates. Both systems use normalized coordinates internally.

</details>

---

### Question 50

A developer is reviewing code from a colleague and finds:

```python
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
for ax in axes.flatten():
    ax.plot(np.random.randn(100))
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Run Sample")
fig.savefig("grid.png", dpi=200)
```

The saved figure has clipped labels. The developer adds `plt.tight_layout()` before `savefig()`. A senior engineer says: "`bbox_inches='tight'` in `savefig` would also fix it — but they're not identical." What is the difference?

- [ ] A) `tight_layout()` changes the figure size; `bbox_inches='tight'` only affects the crop
- [ ] B) `tight_layout()` adjusts subplot geometry within the existing figure dimensions; `bbox_inches='tight'` computes a tight bounding box and may expand the saved image area beyond the declared figure size to include clipped artists
- [ ] C) They are identical — one is just the older API
- [ ] D) `bbox_inches='tight'` only applies to axes labels; `tight_layout()` handles all artists

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `tight_layout()` adjusts subplot geometry within the existing figure dimensions; `bbox_inches='tight'` computes a tight bounding box and may expand the saved image area beyond the declared figure size to include clipped artists

**Explanation:**

- **A is incorrect:** `tight_layout()` adjusts spacing (not figure size). `bbox_inches='tight'` adjusts the saved file crop (not the figure's internal geometry).
- **B is correct:** These two approaches solve the clipping problem via different mechanisms: `plt.tight_layout()` **rearranges the subplot geometry** (moves axes, adjusts margins) so all content fits inside the existing `figsize` rectangle — the figure dimensions stay the same. `bbox_inches='tight'` in `savefig` **does not change the figure** — instead it measures the actual bounding box of all rendered content at save time and saves a larger image if needed to include content outside the normal figure boundary. The resulting saved file may have slightly different dimensions than the declared `figsize` at the given `dpi`.
- **C is incorrect:** They are distinct mechanisms with different effects on figure geometry vs. saved file size.
- **D is incorrect:** `bbox_inches='tight'` applies to ALL artists — labels, tick marks, titles, legends, colorbars, suptitle — anything that extends outside the figure boundary. It is not limited to axis labels.

</details>
