# Matplotlib Figure & Axes Objective Test — Set 04
# Topic: Title Management — suptitle vs set_title (Questions 31–40)

---

### Question 31

A developer creates a 2×2 subplot figure with both a super-title and individual subplot titles:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Model Evaluation Dashboard", fontsize=16, fontweight="bold")

titles = ["Accuracy", "Precision", "Recall", "F1-Score"]
for ax, title in zip(axes.flatten(), titles):
    ax.set_title(title, fontsize=12)
    ax.plot(epochs, metrics[title])
```

A reviewer notices the super-title overlaps with the top subplot titles. What is the best fix?

- [ ] A) Remove `fig.suptitle()` and put the dashboard title in the first subplot using `axes[0,0].set_title()`
- [ ] B) Add `plt.tight_layout()` or `fig.tight_layout()` after the subplots, or set `y=1.02` in `suptitle` to push it above the subplot titles
- [ ] C) Set `fontsize=8` in `suptitle` so it doesn't visually clash
- [ ] D) Use `fig.text(0.5, 1.0, "Dashboard")` instead of `suptitle`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add `plt.tight_layout()` or `fig.tight_layout()` after the subplots, or set `y=1.02` in `suptitle` to push it above the subplot titles

**Explanation:**

- **A is incorrect:** Putting the dashboard title in a subplot would make it appear inside the plot area of just one panel — not spanning all panels as a true heading. The `fig.suptitle()` is semantically and visually correct for a dashboard-level title.
- **B is correct:** Two solutions exist: (1) `plt.tight_layout()` automatically adjusts subplot geometry to prevent overlapping, but it may not account for `suptitle` by itself. Adding `rect=[0, 0, 1, 0.95]` to `tight_layout()` (i.e., `plt.tight_layout(rect=[0, 0, 1, 0.95])`) reserves space at the top for the suptitle. (2) `y=1.02` in `fig.suptitle()` pushes the title slightly above the normalized figure top (y=1.0), visually clearing the top subplot titles. Both approaches are valid production fixes.
- **C is incorrect:** Reducing font size may reduce visual impact but doesn't solve the positional overlap. If the suptitle is at the same y-coordinate as subplot titles, they will still overlap regardless of font size.
- **D is incorrect:** `fig.text(0.5, 1.0, "Dashboard")` places text at normalized figure coordinates (0.5, 1.0) which is the very top edge — still potentially overlapping. `suptitle` is the purpose-built API for this use case and is the appropriate choice.

</details>

---

### Question 32

A developer uses `ax.set_title()` with the `loc` parameter:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].set_title("Left-Aligned", loc="left")
axes[1].set_title("Centered (default)", loc="center")
axes[2].set_title("Right-Aligned", loc="right")
```

What does the `loc` parameter control for `ax.set_title()`?

- [ ] A) The font weight (left=normal, center=bold, right=italic)
- [ ] B) The horizontal alignment of the title text within the axes width (left edge, center, or right edge)
- [ ] C) The y-position of the title (left=low, center=middle, right=top)
- [ ] D) Whether the title appears above or below the subplot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The horizontal alignment of the title text within the axes width (left edge, center, or right edge)

**Explanation:**

- **A is incorrect:** `loc` controls POSITION, not font weight. Font weight is set via `fontweight="bold"` parameter separately.
- **B is correct:** `ax.set_title(label, loc="left"/"center"/"right")` positions the title text **horizontally** within the axes span: `"left"` aligns the title to the left edge of the axes, `"center"` (default) centers it, and `"right"` aligns it to the right edge. This is useful for aligning subplot labels with panel letters (e.g., "(a)", "(b)") in academic figures.
- **C is incorrect:** The y-position of the title is controlled by the `pad` parameter (distance from the top spine) or the `y=` argument, not `loc`. `loc` is purely a horizontal alignment setting.
- **D is incorrect:** Titles always appear above the subplot by default. To place a title below, you would use `ax.set_xlabel()` (styled as a title) or manually position text with `ax.text()`. The `loc` parameter does not control above/below placement.

</details>

---

### Question 33

A developer creates a single-axes figure and uses both `fig.suptitle()` and `ax.set_title()`:

```python
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("ROC Curve Analysis", fontsize=14)
ax.set_title("XGBoost Model (AUC = 0.94)", fontsize=11)
ax.plot(fpr, tpr)
```

What is the visual result if no layout adjustments are made?

- [ ] A) Only `fig.suptitle()` is displayed — `ax.set_title()` is ignored when `suptitle` is present
- [ ] B) Both titles appear: `fig.suptitle()` at the top of the figure (above all content) and `ax.set_title()` just above the plot area — they may overlap if the figure is not adjusted
- [ ] C) Both titles appear centered at the same vertical position, creating bold duplicated text
- [ ] D) `ax.set_title()` appears inside the plot area at the top-left

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Both titles appear: `fig.suptitle()` at the top of the figure (above all content) and `ax.set_title()` just above the plot area — they may overlap if the figure is not adjusted

**Explanation:**

- **A is incorrect:** Neither `suptitle` nor `set_title` suppresses the other. Both are rendered simultaneously as independent text artists at their respective positions.
- **B is correct:** `fig.suptitle()` places a text artist in figure-level coordinates, centered at the top of the figure. `ax.set_title()` places a text artist in axes-level coordinates, positioned just above the top spine of the axes. In a figure with a single subplot (where the axes spans most of the figure), the suptitle and axes title can end up very close together or overlapping, because the axes top is near the figure top. The standard fix is to add vertical space with `fig.suptitle(..., y=0.98)` and `plt.tight_layout(rect=[0, 0, 1, 0.93])`.
- **C is incorrect:** The two titles are not merged or bolded — they are independent `Text` artists at different y-positions in the figure's coordinate system.
- **D is incorrect:** `ax.set_title()` always appears ABOVE the axes (outside the plotting area), not inside it. Text inside the plotting area requires `ax.text(x, y, "text")` with data coordinates.

</details>

---

### Question 34

A developer creates a figure where `fig.suptitle()` is positioned using the `x` and `y` parameters:

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Training Pipeline Results", x=0.3, y=1.05, fontsize=14, ha="left")
```

What coordinate system do `x=0.3` and `y=1.05` use in `fig.suptitle()`?

- [ ] A) Data coordinates of the first subplot
- [ ] B) Normalized figure coordinates where (0, 0) is the bottom-left and (1, 1) is the top-right of the figure
- [ ] C) Pixel coordinates from the top-left corner of the figure
- [ ] D) Axes coordinates of the currently active subplot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Normalized figure coordinates where (0, 0) is the bottom-left and (1, 1) is the top-right of the figure

**Explanation:**

- **A is incorrect:** `fig.suptitle()` operates at the figure level, not within any individual subplot's data coordinate system. `x=0.3` does not refer to a data value on the x-axis of any subplot.
- **B is correct:** `fig.suptitle(x=..., y=...)` positions the title in **normalized figure coordinates** — also called "figure fraction" coordinates. The origin (0, 0) is the bottom-left corner of the figure; (1, 0) is bottom-right; (0, 1) is top-left; (1, 1) is top-right. `x=0.3` positions the title at 30% from the left; `y=1.05` places it 5% above the top of the figure (outside the figure boundary — visible in the saved file if `bbox_inches='tight'` is used). `ha="left"` aligns the text's left edge to x=0.3.
- **C is incorrect:** Matplotlib uses normalized (0–1) coordinates for figure-level positioning, not pixel coordinates. Pixel-based text placement would require `transform=None` with data-unit pixel values, which is rare and unwieldy.
- **D is incorrect:** `suptitle` is a figure-level method and is not bound to any individual axes coordinate system. For axes-fraction coordinates, one would use `ax.text(x, y, "text", transform=ax.transAxes)`.

</details>

---

### Question 35

A developer accidentally uses `ax.set_title()` for what they thought was a super-title:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].set_title("Model Dashboard", fontsize=16)  # Intended as super-title
```

What is wrong with this approach?

- [ ] A) `set_title()` is deprecated — use `suptitle()` for all titles
- [ ] B) The title appears only above the top-left subplot, not spanning the entire figure — it is a subplot-level title, not a figure-level title
- [ ] C) `set_title()` cannot accept `fontsize` parameter
- [ ] D) The title will appear below the subplot instead of above it

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The title appears only above the top-left subplot, not spanning the entire figure — it is a subplot-level title, not a figure-level title

**Explanation:**

- **A is incorrect:** `ax.set_title()` is not deprecated. It is the correct method for individual subplot titles. The issue is using it for a purpose it's not designed for.
- **B is correct:** `ax.set_title()` places a title directly above the specific axes it is called on. `axes[0, 0].set_title("Model Dashboard")` puts the title above the **top-left subplot only** — horizontally aligned over just that panel. For a title that spans all four subplots as a heading for the entire dashboard, `fig.suptitle("Model Dashboard", fontsize=16)` is the correct API. `suptitle` positions itself at the figure level, centered across the full figure width.
- **C is incorrect:** `ax.set_title()` absolutely accepts `fontsize` as a valid keyword argument (e.g., `ax.set_title("Title", fontsize=16)`).
- **D is incorrect:** `ax.set_title()` always places the title ABOVE the axes (by default, just above the top spine). It does not appear below the subplot.

</details>

---

### Question 36

A developer creates a multi-panel figure and wants to add **panel labels** ("(a)", "(b)", "(c)") in the top-left corner of each subplot, as is common in academic publications:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
labels = ["(a)", "(b)", "(c)"]
for ax, label in zip(axes, labels):
    ax.set_title(label, loc="left", fontweight="bold", fontsize=13)
```

Is this a correct approach?

- [ ] A) No — panel labels must be placed with `fig.text()`, not `ax.set_title()`
- [ ] B) Yes — `ax.set_title(loc="left")` places the label at the left edge of the axes, which is the standard position for panel labels in publications
- [ ] C) No — `loc="left"` positions the title at the left edge of the figure, not the subplot
- [ ] D) Yes — but only if `fontsize` is ≥ 14 for the label to appear correctly

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `ax.set_title(loc="left")` places the label at the left edge of the axes, which is the standard position for panel labels in publications

**Explanation:**

- **A is incorrect:** While `fig.text()` is one approach for placing text at arbitrary figure positions, using `ax.set_title(loc="left")` is a clean, idiomatic, and widely used method for panel labels. It correctly positions the label at the top-left of each individual subplot — exactly where panel labels belong.
- **B is correct:** `ax.set_title("(a)", loc="left")` positions the title text with its left edge aligned to the left edge of the **axes** (not the figure). Combined with `fontweight="bold"`, this produces the standard bold "(a)", "(b)", "(c)" panel annotation style used in Nature, Science, and similar journals. Each call is per-axes, so each panel gets its own label aligned to its own left edge.
- **C is incorrect:** `loc="left"` in `ax.set_title()` aligns the title to the left edge of the **axes** (the specific subplot), not the figure. Figure-level left alignment would use `fig.suptitle(ha="left")`.
- **D is incorrect:** There is no fontsize requirement for `loc="left"` to function. The positioning works regardless of font size.

</details>

---

### Question 37

A developer needs to vertically position a `suptitle` to avoid overlap with top subplots. They use:

```python
fig.suptitle("Dashboard", y=0.98)
```

If `y=1.0` places the title at the very top of the figure, what does `y=0.98` do?

- [ ] A) Moves the title 2 pixels downward from the top
- [ ] B) Moves the title to 98% of the figure height from the bottom — slightly below the very top, reducing the gap to the top subplots
- [ ] C) Scales the font size to 98% of the specified value
- [ ] D) Sets the y-coordinate in data units of the first subplot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Moves the title to 98% of the figure height from the bottom — slightly below the very top, reducing the gap to the top subplots

**Explanation:**

- **A is incorrect:** The `y` parameter uses normalized figure coordinates (0 to 1), not pixels. Moving by "2 pixels" would require a tiny fractional change that depends on figure size and DPI — it's not 0.02 in normalized coordinates for 2 pixels.
- **B is correct:** In `fig.suptitle()`, `y` is in normalized figure coordinates: `y=0.0` is the figure bottom, `y=1.0` is the figure top. `y=0.98` places the title center at **98% of the figure height from the bottom** — slightly below the very top edge, which can actually REDUCE the space between the suptitle and the top subplot row. To add MORE space (avoid overlap), use `y=1.02` (above the figure top, only visible in tight-bbox saves) or use `plt.tight_layout(rect=[0, 0, 1, 0.95])`.
- **C is incorrect:** `y` has nothing to do with font scaling. Font size is a separate parameter.
- **D is incorrect:** `y` in `suptitle()` is in normalized **figure** coordinates (0–1), not in data units of any subplot.

</details>

---

### Question 38

A developer creates a figure with 3 subplots and wants different title styles for each:

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].set_title("Confusion Matrix", fontsize=12, color="navy", pad=10)
axes[1].set_title("ROC Curve", fontsize=12, style="italic")
axes[2].set_title("Precision-Recall", fontsize=12, fontweight="bold")
```

What does the `pad` parameter in `axes[0].set_title(..., pad=10)` control?

- [ ] A) The left padding (margin) before the title text starts
- [ ] B) The vertical distance in points between the top spine of the axes and the title text
- [ ] C) The font padding (letter spacing) within the title text
- [ ] D) The number of blank lines inserted above the title

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The vertical distance in points between the top spine of the axes and the title text

**Explanation:**

- **A is incorrect:** `pad` controls vertical spacing, not horizontal left margin. Left horizontal alignment is controlled by `loc="left"`.
- **B is correct:** `pad` in `ax.set_title()` controls the **vertical gap** (in points) between the top spine of the axes (the top border line) and the bottom of the title text. The default pad is `matplotlib.rcParams["axes.titlepad"]` (typically 6 points). Increasing `pad=10` pushes the title further up from the axes top, providing more breathing room — useful when the plot area is crowded or when tick labels on a secondary x-axis at the top need space.
- **C is incorrect:** Letter spacing (character spacing) is not controlled by `pad`. It would require using `fontproperties` or LaTeX rendering options. `pad` is purely about vertical spacing between spine and text.
- **D is incorrect:** `pad` uses point units (typographic points, 1/72 of an inch), not line counts. It is a continuous numeric value, not a discrete line counter.

</details>

---

### Question 39

A developer creates a figure and uses `fig.text()` instead of `fig.suptitle()`:

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.text(0.5, 0.98, "Model Performance Summary",
         ha="center", va="top", fontsize=14, fontweight="bold")
```

How does `fig.text()` differ from `fig.suptitle()` for this use case?

- [ ] A) `fig.text()` and `fig.suptitle()` are completely identical — one is just an alias for the other
- [ ] B) `fig.text()` places text at arbitrary normalized figure coordinates with full manual control; `fig.suptitle()` is a purpose-built wrapper that manages positioning and integrates with layout engines like `tight_layout`
- [ ] C) `fig.text()` creates a title that scales with DPI; `fig.suptitle()` does not scale
- [ ] D) `fig.suptitle()` supports markdown bold/italic; `fig.text()` does not

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.text()` places text at arbitrary normalized figure coordinates with full manual control; `fig.suptitle()` is a purpose-built wrapper that manages positioning and integrates with layout engines like `tight_layout`

**Explanation:**

- **A is incorrect:** They are different methods with different internal behaviors, especially regarding layout engine integration.
- **B is correct:** `fig.text(x, y, "text")` is a general-purpose text placement function that places a `Text` artist at any normalized figure coordinate with full control over alignment, rotation, and styling. `fig.suptitle()` is a specialized wrapper that: (1) stores the title in `fig._suptitle` for reference, (2) is recognized by layout engines like `tight_layout` and `constrained_layout` which can account for it when adjusting spacing, and (3) provides a cleaner API for the specific use case of a figure-level heading. `fig.text()` produces a visually similar result but the layout engines don't automatically account for it.
- **C is incorrect:** Both methods create `Text` artists that scale consistently with DPI. Neither has a special DPI scaling advantage over the other.
- **D is incorrect:** Neither `fig.text()` nor `fig.suptitle()` supports markdown formatting. Both support LaTeX math via `$...$` notation when a LaTeX TeX system is available, and both support basic HTML entity characters. Markdown bold/italic is not a Matplotlib text feature.

</details>

---

### Question 40

A developer creates a figure for a two-panel comparison of model metrics:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("XGBoost vs Random Forest — Cross-Validation", 
             fontsize=15, fontweight="bold", y=1.02)
ax1.set_title("XGBoost CV Scores", loc="center", fontsize=11)
ax2.set_title("Random Forest CV Scores", loc="center", fontsize=11)
plt.tight_layout()
plt.savefig("comparison.pdf", bbox_inches="tight")
```

Why is `bbox_inches="tight"` important here, given that `y=1.02` pushes the suptitle above the figure boundary?

- [ ] A) It increases the DPI for vector PDF export
- [ ] B) It computes the minimum bounding box that encloses all artists — including the suptitle at y=1.02 which extends above the normal figure boundary — and expands the saved figure to include it
- [ ] C) It removes the suptitle from the saved file to avoid clipping
- [ ] D) It only affects raster formats (PNG/JPG), not vector formats (PDF)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It computes the minimum bounding box that encloses all artists — including the suptitle at y=1.02 which extends above the normal figure boundary — and expands the saved figure to include it

**Explanation:**

- **A is incorrect:** `bbox_inches="tight"` does not affect DPI. DPI for export is controlled by the `dpi=` parameter in `savefig`. For vector formats (PDF, SVG), DPI is less relevant as they are resolution-independent.
- **B is correct:** `bbox_inches="tight"` instructs Matplotlib to calculate the **tight bounding box** — the smallest rectangle that encompasses all rendered artists, including titles, tick labels, and any text positioned outside the nominal figure boundary (like `suptitle` at `y=1.02`). The saved file is then sized to this bounding box, adding extra space above the figure if needed to include the suptitle. Without `bbox_inches="tight"`, the suptitle at `y=1.02` would be **clipped** in the saved PDF because it falls outside the declared figure rectangle.
- **C is incorrect:** `bbox_inches="tight"` does the opposite — it INCLUDES the suptitle by expanding the bounding box. It never removes content.
- **D is incorrect:** `bbox_inches="tight"` works for ALL Matplotlib output formats — PNG, PDF, SVG, EPS, TIFF, etc. It's a universal bounding box computation step that happens before the backend renders the figure to any format.

</details>
