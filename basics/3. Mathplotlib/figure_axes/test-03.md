# Matplotlib Figure & Axes Objective Test — Set 03
# Topic: Color & Styling Hierarchy (Questions 21–30)

---

### Question 21

A developer creates a figure with a dark background for a dashboard:

```python
fig, ax = plt.subplots(figsize=(8, 5))
fig.set_facecolor("#1A1A2E")
ax.plot([1, 2, 3], [4, 5, 6], color="cyan")
plt.show()
```

After running this, the developer saves the figure:

```python
plt.savefig("dashboard.png")
```

The saved PNG has a **white background** instead of the dark `#1A1A2E`. What caused this?

- [ ] A) `fig.set_facecolor()` only works for interactive display, not file export
- [ ] B) `plt.savefig()` by default uses `facecolor="white"`, overriding the figure's facecolor unless `facecolor=fig.get_facecolor()` is passed explicitly
- [ ] C) PNG format does not support dark background colors
- [ ] D) `#1A1A2E` is an invalid hex color code that Matplotlib silently ignores

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.savefig()` by default uses `facecolor="white"`, overriding the figure's facecolor unless `facecolor=fig.get_facecolor()` is passed explicitly

**Explanation:**

- **A is incorrect:** `fig.set_facecolor()` works correctly for both interactive display and file export — but only when the facecolor is explicitly passed to `savefig`. The method itself is not backend-limited.
- **B is correct:** `plt.savefig()` has a `facecolor` parameter that defaults to `"white"` (or `"auto"` in newer Matplotlib versions that reads from the figure). In many Matplotlib versions, the default overrides the figure's actual facecolor unless you explicitly pass `facecolor=fig.get_facecolor()`. The correct pattern is: `plt.savefig("dashboard.png", facecolor=fig.get_facecolor())` or `plt.savefig("dashboard.png", facecolor="#1A1A2E")`.
- **C is incorrect:** PNG fully supports dark and arbitrary background colors, including transparency. It is a lossless raster format with full RGB/RGBA support.
- **D is incorrect:** `#1A1A2E` is a valid CSS hex color (dark navy). Matplotlib accepts all standard hex color codes. Invalid codes raise a `ValueError`, they are not silently ignored.

</details>

---

### Question 22

A developer wants the **inner plot area** (inside the axes) to appear light gray (`#F0F0F0`) while keeping the outer figure background white. Which code correctly achieves this?

- [ ] A) `fig.set_facecolor("#F0F0F0")`
- [ ] B) `ax.set_facecolor("#F0F0F0")`
- [ ] C) `fig.patch.set_facecolor("#F0F0F0")`
- [ ] D) `plt.rcParams["figure.facecolor"] = "#F0F0F0"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_facecolor("#F0F0F0")`

**Explanation:**

- **A is incorrect:** `fig.set_facecolor("#F0F0F0")` changes the **outer figure background** (the area outside and around the axes) — not the inner plot region. The inner plot area would remain white (default).
- **B is correct:** `ax.set_facecolor("#F0F0F0")` changes the background color of the **axes rectangle** — the region bounded by the spines where data is drawn. This is the inner plot area. Equivalently, `ax.patch.set_facecolor("#F0F0F0")` achieves the same result since `ax.patch` is the `Rectangle` backing the axes interior.
- **C is incorrect:** `fig.patch` is the figure's background `Rectangle`, not the axes background. Styling `fig.patch` changes the outer figure background — same as `fig.set_facecolor()`.
- **D is incorrect:** `plt.rcParams["figure.facecolor"]` sets the global default for the **figure** background, not individual axes backgrounds. It would apply to all newly created figures' outer backgrounds.

</details>

---

### Question 23

A developer creates a figure and then checks the default colors:

```python
fig, ax = plt.subplots()
print(fig.get_facecolor())
print(ax.get_facecolor())
```

What are the **default facecolors** for a fresh figure and axes?

- [ ] A) Figure: `(0, 0, 0, 1)` (black); Axes: `(1, 1, 1, 1)` (white)
- [ ] B) Figure: `(1, 1, 1, 1)` (white); Axes: `(1, 1, 1, 1)` (white)
- [ ] C) Figure: `(0.94, 0.94, 0.94, 1)` (light gray `#F0F0F0`); Axes: `(1, 1, 1, 1)` (white)
- [ ] D) Both return the string `"white"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Figure: `(0.94, 0.94, 0.94, 1)` (light gray `#F0F0F0`); Axes: `(1, 1, 1, 1)` (white)

**Explanation:**

- **A is incorrect:** The default figure background is NOT black. Matplotlib's default figure background is a light gray.
- **B is incorrect:** The figure background is NOT white by default. The Matplotlib default figure facecolor is `"#F0F0F0"` (light gray), which returns as the RGBA tuple `(0.941..., 0.941..., 0.941..., 1.0)`.
- **C is correct:** By default (`rcParams["figure.facecolor"] = "#F0F0F0"`), the figure's outer background is light gray, returning the RGBA tuple approximately `(0.941, 0.941, 0.941, 1.0)`. The axes inner background defaults to white — RGBA `(1.0, 1.0, 1.0, 1.0)`. This creates the visual distinction where the plot area appears white against a light gray figure border.
- **D is incorrect:** `get_facecolor()` returns an RGBA tuple `(r, g, b, a)`, not a string. The string `"white"` is an input format — the getter always returns normalized float tuples.

</details>

---

### Question 24

A developer sets the facecolor on both the figure and axes for a dark-themed ML dashboard:

```python
fig, ax = plt.subplots(figsize=(10, 6))
fig.set_facecolor("#0D1117")   # GitHub-dark outer background
ax.set_facecolor("#161B22")    # Slightly lighter inner plot area
ax.plot(epochs, val_loss, color="#58A6FF", linewidth=2)
ax.tick_params(colors="white")
ax.spines[:].set_color("white")
ax.set_xlabel("Epoch", color="white")
ax.set_ylabel("Validation Loss", color="white")
```

What is the visual relationship between `fig.set_facecolor()` and `ax.set_facecolor()` in this layout?

- [ ] A) They control the same background — one overrides the other
- [ ] B) `fig.set_facecolor()` sets the outer figure margin background (the area outside the axes); `ax.set_facecolor()` sets the inner plot region background; they create a two-tone layered visual effect
- [ ] C) `ax.set_facecolor()` sets the axes background but `fig.set_facecolor()` has no visual effect when axes are present
- [ ] D) `fig.set_facecolor()` affects only the figure title area above the axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig.set_facecolor()` sets the outer figure margin background (the area outside the axes); `ax.set_facecolor()` sets the inner plot region background; they create a two-tone layered visual effect

**Explanation:**

- **A is incorrect:** They control different backgrounds and do NOT override each other. Both settings coexist, each affecting their own layer.
- **B is correct:** The `Figure` and `Axes` have independent background `Rectangle` patches. `fig.set_facecolor("#0D1117")` colors the **outer** figure area — the margins, padding, and any space outside the axes bounding box. `ax.set_facecolor("#161B22")` colors the **inner** axes region where data is drawn. In the dark dashboard example, this creates a two-tone effect: a very dark outer border (`#0D1117`) surrounding a slightly lighter inner plot area (`#161B22`), giving depth to the visualization.
- **C is incorrect:** `fig.set_facecolor()` has a clear and visible effect — it colors the figure's outer margin. The outer margins are always visible around the axes unless margins are zero.
- **D is incorrect:** `fig.set_facecolor()` affects the entire figure background rectangle, not just the title area. The full figure canvas (height × width) is colored by `fig.patch`.

</details>

---

### Question 25

A developer wants to make the **figure background transparent** for use in a presentation with a colored slide background. Which `savefig` call achieves this?

- [ ] A) `plt.savefig("plot.png", facecolor="none")`
- [ ] B) `plt.savefig("plot.png", transparent=True)`
- [ ] C) `plt.savefig("plot.png", alpha=0)`
- [ ] D) `plt.savefig("plot.png", background="transparent")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.savefig("plot.png", transparent=True)`

**Explanation:**

- **A is incorrect:** While `facecolor="none"` sets the figure patch to transparent, using `"none"` as a color string may not propagate correctly to all backends and does not apply to the axes background. `transparent=True` is the correct, comprehensive parameter.
- **B is correct:** `plt.savefig("plot.png", transparent=True)` is the dedicated parameter for making the figure background transparent. When `transparent=True`, both the figure background (`fig.patch`) and the axes background (`ax.patch`) are rendered with alpha=0 in the saved file, making them fully transparent. This is essential for overlaying Matplotlib plots on presentation slides, web pages, or other colored backgrounds.
- **C is incorrect:** `alpha=` is not a valid top-level parameter for `savefig()`. Setting transparency requires the `transparent=True` parameter or using `facecolor` with an RGBA tuple.
- **D is incorrect:** `background="transparent"` is not a valid `savefig()` parameter. This would raise a `TypeError`.

</details>

---

### Question 26

A developer creates a figure and sets edge colors:

```python
fig, ax = plt.subplots(figsize=(6, 4))
fig.set_edgecolor("navy")
ax.set_facecolor("lightyellow")
```

What does `fig.set_edgecolor("navy")` visually affect?

- [ ] A) The color of the data line plotted on the axes
- [ ] B) The border/frame line around the outer figure rectangle (the figure patch edge)
- [ ] C) The color of the axis spines (the four lines forming the axes box)
- [ ] D) The tick mark colors on both x and y axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The border/frame line around the outer figure rectangle (the figure patch edge)

**Explanation:**

- **A is incorrect:** Data line colors are controlled by the `color=` parameter in plotting functions like `ax.plot(color="navy")`. `fig.set_edgecolor()` has nothing to do with plotted data.
- **B is correct:** `fig.set_edgecolor("navy")` styles the **edge (border line)** of the `Figure`'s background patch rectangle — the outermost frame of the entire figure. By default this is typically invisible or very thin. Setting it to "navy" draws a navy-colored border around the entire figure canvas. This is useful for creating framed figures in reports.
- **C is incorrect:** Axis spine colors are controlled via `ax.spines["top"].set_color(...)`, `ax.spines["bottom"].set_color(...)`, etc. Spines are part of the `Axes` object, not the `Figure` patch.
- **D is incorrect:** Tick mark colors are controlled via `ax.tick_params(colors=...)` or individually on `ax.xaxis` and `ax.yaxis`. The figure edge color has no effect on tick marks.

</details>

---

### Question 27

A developer is building a seaborn-style white grid theme manually using Matplotlib primitives:

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("white")
ax.grid(True, color="#CCCCCC", linewidth=0.8, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#AAAAAA")
ax.spines["bottom"].set_color("#AAAAAA")
```

What is the effect of `ax.spines["top"].set_visible(False)`?

- [ ] A) Hides the top tick marks on the x-axis
- [ ] B) Removes the top border line of the axes box — the line that runs along the top of the plot area
- [ ] C) Hides the title text above the axes
- [ ] D) Removes the top row of data from the plot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Removes the top border line of the axes box — the line that runs along the top of the plot area

**Explanation:**

- **A is incorrect:** Tick marks (tick lines and tick labels) are part of `ax.xaxis` and `ax.yaxis`, not spines. Hiding a spine does not affect tick marks.
- **B is correct:** `ax.spines` is a dictionary of `Spine` objects: `"top"`, `"bottom"`, `"left"`, `"right"`. Each spine is the line forming one side of the axes border box. `ax.spines["top"].set_visible(False)` hides the **top border line** of the plotting area, giving the plot a more open, modern look (common in seaborn's default style). The code also hides the right spine, creating the characteristic "L-shaped" axes border.
- **C is incorrect:** The title is a `Text` artist controlled by `ax.set_title()`. Spines are border lines, not text elements.
- **D is incorrect:** Spines are purely cosmetic border lines. They have no effect on what data is plotted or visible within the axes coordinate system.

</details>

---

### Question 28

A developer uses the following code to apply a dark background theme to multiple subplots simultaneously:

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.set_facecolor("#121212")

for ax in axes.flatten():
    ax.set_facecolor("#1E1E1E")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444444")
```

What does `axes.flatten()` accomplish, and why is it used here?

- [ ] A) It converts the 2D (2×3) NumPy array of `Axes` objects into a 1D array for simple iteration
- [ ] B) It creates flat (non-3D) axes suitable for 2D plotting
- [ ] C) It removes duplicate axes from the grid
- [ ] D) It normalizes axis scales to the range [0, 1]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It converts the 2D (2×3) NumPy array of `Axes` objects into a 1D array for simple iteration

**Explanation:**

- **A is correct:** `plt.subplots(2, 3)` returns `axes` as a **2D NumPy array** of shape `(2, 3)`. To iterate over all 6 axes without nested loops (`for row in axes: for ax in row:`), `axes.flatten()` converts the (2×3) array into a flat 1D array of 6 elements — enabling the clean `for ax in axes.flatten():` single-loop pattern. This is the canonical way to apply uniform styling to all subplots in a grid.
- **B is incorrect:** "Flat" in `ndarray.flatten()` refers to the array dimensionality (1D vs 2D), not the type of plot (2D vs 3D). `Axes` objects for 2D and 3D plots are entirely different classes; `flatten()` has nothing to do with plotting dimensionality.
- **C is incorrect:** `flatten()` is a NumPy array operation that reshapes the array — it doesn't analyze or remove any axes content. All 6 axes remain present after flattening.
- **D is incorrect:** Axis scale normalization is done via `ax.set_xlim()`, `ax.set_ylim()`, or `ax.autoscale()`. NumPy's `flatten()` is an array shape operation, not an axis scale operation.

</details>

---

### Question 29

A developer creates the following plot and is surprised by the visual output:

```python
fig, ax = plt.subplots(figsize=(6, 4))
fig.set_facecolor("black")
ax.set_facecolor("black")
ax.plot([1, 2, 3], [1, 4, 9], color="black")
plt.show()
```

What does the user see?

- [ ] A) A black figure with a clearly visible parabolic curve
- [ ] B) A completely black figure — the curve is invisible because the line color matches the background
- [ ] C) An error because black axes are unsupported
- [ ] D) A figure with white spines but a black interior

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A completely black figure — the curve is invisible because the line color matches the background

**Explanation:**

- **A is incorrect:** The curve IS plotted, but it is `color="black"` on a `black` axes background — it is completely invisible. Matplotlib does not warn you when line color matches background color.
- **B is correct:** All three elements — figure background, axes background, and line color — are black. The curve `[1,4,9]` is drawn in black on a black background, making it invisible. The spines (axes border) are also rendered but may be hard to see against the black background depending on their default color. This is a common mistake when theming figures for dark backgrounds without updating line colors to high-contrast alternatives (e.g., `color="cyan"` or `color="white"`).
- **C is incorrect:** Black backgrounds are fully supported in Matplotlib. No error is raised.
- **D is incorrect:** Spine colors default to black in a standard Matplotlib configuration. On a black background, even spines would be invisible (or barely distinguishable). The spines are not automatically set to white when the background becomes black.

</details>

---

### Question 30

A developer wants to create a figure where the **axes background is semi-transparent** so that a background image shows through. Which approach correctly sets 50% transparency for the axes background?

- [ ] A) `ax.set_facecolor("white", alpha=0.5)`
- [ ] B) `ax.patch.set_alpha(0.5)`
- [ ] C) `ax.set_alpha(0.5)`
- [ ] D) `ax.background_alpha = 0.5`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.patch.set_alpha(0.5)`

**Explanation:**

- **A is incorrect:** `ax.set_facecolor()` does not accept an `alpha` keyword argument. To set both color and transparency, you must either use an RGBA tuple like `ax.set_facecolor((1, 1, 1, 0.5))` or call `ax.patch.set_alpha(0.5)` after setting the color.
- **B is correct:** `ax.patch` is the `Rectangle` artist that fills the axes background. `ax.patch.set_alpha(0.5)` sets its transparency to 50% — meaning the background is half-transparent, allowing content behind the figure (e.g., a background image set on the figure) to show through at half intensity. This is the correct approach for semi-transparent axes backgrounds.
- **C is incorrect:** `ax.set_alpha(0.5)` sets the alpha (transparency) of the **entire Axes artist**, including all its children (data lines, ticks, labels). This would make everything in the axes 50% transparent — not just the background patch.
- **D is incorrect:** `ax.background_alpha` is not a valid Matplotlib attribute. This would raise an `AttributeError` if set, or simply do nothing (Python allows setting arbitrary attributes on objects but Matplotlib doesn't read `background_alpha` for rendering).

</details>
