# Matplotlib Imshow & Heatmaps — Test 05
## Geometry & Orientations (aspect, origin)

---

### Question 41

A developer displays a feature matrix from an ML pipeline where the array shape is `(1000, 8)` — 1000 samples × 8 features:

```python
feature_matrix = np.random.rand(1000, 8)
ax.imshow(feature_matrix, cmap='viridis')
```

The resulting image appears as an extremely tall, pencil-thin vertical strip that is unreadable. What parameter fixes this?

- [ ] A) `ax.imshow(feature_matrix, cmap='viridis', aspect='equal')` — forces square pixels
- [ ] B) `ax.imshow(feature_matrix, cmap='viridis', aspect='auto')` — stretches the image to fill the subplot dimensions, regardless of pixel aspect ratio
- [ ] C) `ax.imshow(feature_matrix, cmap='viridis', figsize=(20, 4))` — the `figsize` parameter goes inside `imshow`
- [ ] D) `ax.imshow(feature_matrix.T, cmap='viridis')` — transpose the matrix to fix the aspect

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.imshow(feature_matrix, cmap='viridis', aspect='auto')` — stretches the image to fill the subplot dimensions, regardless of pixel aspect ratio

**Explanation:**

- **A is incorrect:** `aspect='equal'` is the **default** — it enforces square pixels, meaning each data cell occupies equal width and height in display space. For a (1000, 8) matrix, equal pixel aspect means 1000 rows spread vertically while only 8 columns spread horizontally — producing the pencil-thin strip. This makes the problem worse, not better.
- **B is correct:** `aspect='auto'` tells Matplotlib to **ignore pixel aspect ratio** and stretch the image to fill the available subplot space. The (1000, 8) array will expand its 8 columns horizontally and compress its 1000 rows vertically to fit the axes rectangle — producing a readable wide heatmap. This is essential for non-square feature matrices, attention maps, and any array where the row/column ratio is extreme.
- **C is incorrect:** `figsize` is a parameter of `plt.figure()` or `plt.subplots()` — not of `ax.imshow()`. Passing `figsize=` to `imshow` raises a `TypeError`.
- **D is incorrect:** Transposing gives a (8, 1000) matrix — now 8 rows × 1000 columns — which produces an extremely wide, flat strip. This makes it equally unreadable in the opposite direction.

</details>

---

### Question 42

A developer sets `aspect='equal'` explicitly on a square 10×10 matrix. What does this guarantee about the rendered image?

- [ ] A) The entire figure uses equal horizontal and vertical margins
- [ ] B) Each data cell is rendered as a perfect square in display coordinates — the width and height of each pixel/cell are equal
- [ ] C) The colormap is applied with equal weight to all cells regardless of their values
- [ ] D) The image fills the full axes rectangle without any whitespace padding

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Each data cell is rendered as a perfect square in display coordinates — the width and height of each pixel/cell are equal

**Explanation:**

- **A is incorrect:** `aspect='equal'` on an axes (from `imshow`) controls the data coordinate aspect ratio — not figure-level margins. Figure margins are controlled by `plt.tight_layout()`, `plt.subplots_adjust()`, or `fig.set_constrained_layout(True)`.
- **B is correct:** `aspect='equal'` enforces that one unit in the x-direction equals one unit in the y-direction in display space. For an image displayed with `imshow`, this means each data cell occupies equal width and height — producing perfectly **square pixels**. This is standard for computer vision images (where pixels are square by convention), spatial maps, and square matrices. For non-square arrays, `aspect='equal'` causes Matplotlib to adjust the axes dimensions (possibly adding whitespace) to maintain the pixel aspect.
- **C is incorrect:** Colormap weighting is entirely independent of the `aspect` parameter. All cells are always treated equally by the colormap normalization regardless of aspect setting.
- **D is incorrect:** With `aspect='equal'` on a non-square array, Matplotlib will NOT expand the image to fill the axes rectangle — it will maintain the data aspect ratio and add whitespace (letterboxing) as needed.

</details>

---

### Question 43

A developer creates a geographic heat map of temperature data and calls:

```python
temp_grid = np.random.rand(90, 180)  # 90 rows (latitudes), 180 columns (longitudes)
ax.imshow(temp_grid, origin='upper', cmap='hot')
```

With `origin='upper'`, where is row 0 (index 0) positioned in the rendered image?

- [ ] A) At the bottom-left of the image — row 0 represents the southernmost latitude (Cartesian convention)
- [ ] B) At the top-left of the image — row 0 is displayed in the topmost row of the rendered image (matrix/image convention)
- [ ] C) At the center of the image — `origin='upper'` centers the data array
- [ ] D) `origin='upper'` applies only to the colorbar, not the image data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) At the top-left of the image — row 0 is displayed in the topmost row of the rendered image (matrix/image convention)

**Explanation:**

- **A is incorrect:** Row 0 at the bottom-left is the `origin='lower'` convention (Cartesian coordinate system where y=0 is at the bottom). `origin='upper'` is the opposite.
- **B is correct:** `origin='upper'` (the **default** in Matplotlib) places row index 0 at the **top** of the displayed image — matching standard matrix notation and image file conventions. Row 0 is the topmost row; row N-1 is the bottommost row. The y-axis increases downward. This is the convention used by cameras, screens, and image files (JPEG, PNG) — where row 0 is the topmost scanline.
- **C is incorrect:** `origin='upper'/'lower'` has no "centering" behavior. It exclusively controls whether the first array row appears at the top or bottom of the display.
- **D is incorrect:** `origin` applies entirely to the image data rendering (row ordering) — it has no effect on the colorbar positioning or orientation.

</details>

---

### Question 44

A developer creates a **mathematical function heatmap** using `imshow` and uses `origin='lower'`. A colleague who is used to `origin='upper'` notices the image appears "flipped vertically." Which statement correctly explains the difference between the two origin settings?

- [ ] A) `origin='lower'` rotates the image 90° clockwise; `origin='upper'` is the standard view
- [ ] B) `origin='lower'` places row 0 at the **bottom** of the image (y-axis increases upward) — matching Cartesian coordinate convention; `origin='upper'` places row 0 at the **top** (y-axis increases downward) — matching image/matrix convention
- [ ] C) `origin='lower'` and `origin='upper'` are identical for symmetric arrays
- [ ] D) `origin='lower'` applies only to contour plots; `origin='upper'` applies only to `imshow`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `origin='lower'` places row 0 at the **bottom** of the image (y-axis increases upward) — matching Cartesian coordinate convention; `origin='upper'` places row 0 at the **top** (y-axis increases downward) — matching image/matrix convention

**Explanation:**

- **A is incorrect:** `origin` does not rotate the image. It flips the vertical axis direction (y-axis goes up vs. down) — a vertical mirror, not a rotation.
- **B is correct:** The two conventions serve different use cases: **`origin='upper'`** (default) — row 0 at top, y-axis increases downward. Used for: image files, confusion matrices, and any matrix where row 0 is the "first" row visually. **`origin='lower'`** — row 0 at bottom, y-axis increases upward. Used for: mathematical functions, geographic data, scientific spatial fields where y (or latitude) increases upward. For `f(x,y)` heatmaps, `origin='lower'` is correct because low y-values should appear at the bottom (Cartesian convention). Forgetting to set `origin='lower'` vertically flips the entire function landscape.
- **C is incorrect:** Even symmetric arrays look flipped between `origin='upper'` and `origin='lower'` unless the array is perfectly symmetric about the horizontal midline (which `np.random.rand` would never be, for example).
- **D is incorrect:** Both `origin='upper'` and `origin='lower'` apply to `ax.imshow()`. `origin` is also accepted by `ax.contourf()` and related functions. Neither setting is exclusive to one plot type.

</details>

---

### Question 45

A developer calls `ax.imshow(data, origin='lower')` on a 5×5 array. Where does `data[0, 0]` appear in the rendered image?

- [ ] A) Top-left corner
- [ ] B) Bottom-left corner
- [ ] C) Center of the image
- [ ] D) Top-right corner

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Bottom-left corner

**Explanation:**

- **A is incorrect:** `data[0, 0]` at the top-left is the `origin='upper'` convention. With `origin='lower'`, the first row appears at the bottom.
- **B is correct:** With `origin='lower'`, row index 0 is placed at the **bottom** of the rendered image. Column index 0 is always at the **left** (x-axis always increases rightward regardless of origin). Therefore `data[0, 0]` — row 0, column 0 — appears at the **bottom-left** corner. Conversely, `data[N-1, M-1]` (last row, last column) appears at the **top-right** corner.
- **C is incorrect:** `imshow` does not center any specific array element. Array elements are placed at grid positions determined by their row/column indices and the origin setting.
- **D is incorrect:** `data[0, 0]` is column 0 — always placed at the left side. It cannot appear at the right corner for any standard `origin` setting.

</details>

---

### Question 46

A developer is creating a **loss landscape visualization** and evaluates f(x, y) = x² + y² on a 2D grid:

```python
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

ax.imshow(Z, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
ax.set_xlabel("x"); ax.set_ylabel("y")
```

Without `origin='lower'`, what would go wrong with this visualization?

- [ ] A) The minimum of Z (at x=0, y=0) would appear at the bottom of the image instead of the center
- [ ] B) The entire Z matrix would be flipped vertically — the minimum (at center) would be correct, but high values at y=+3 would appear at the bottom and high values at y=-3 would appear at the top, inverting the y-axis meaning
- [ ] C) `extent` requires `origin='lower'` — without it, the extent coordinates are ignored
- [ ] D) `np.meshgrid` creates arrays in `origin='lower'` format — using `origin='upper'` would cause a dimension mismatch

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The entire Z matrix would be flipped vertically — the minimum (at center) would be correct, but high values at y=+3 would appear at the bottom and high values at y=-3 would appear at the top, inverting the y-axis meaning

**Explanation:**

- **A is incorrect:** The minimum of f(x,y) = x² + y² is at (0,0) — the center of the array. Flipping vertically moves the top and bottom rows but the center row stays at center. The minimum at center is not affected by vertical flip.
- **B is correct:** `np.meshgrid(x, y)` creates `Y` such that `Y[0, :]` has the smallest y-values and `Y[-1, :]` has the largest y-values. With `origin='upper'` (default): row 0 appears at the top — so small y-values appear at the top and large y-values at the bottom. But physically, y=+3 (large y) should be at the **top** of a Cartesian plot. So `origin='upper'` inverts the Cartesian y-axis: the bottom of the image corresponds to y=+3 and the top corresponds to y=-3. `origin='lower'` fixes this: row 0 (small y) at bottom, last row (large y) at top — matching standard Cartesian convention.
- **C is incorrect:** `extent` works independently of `origin`. Without `origin='lower'`, the extent labels would be applied but still with the inverted y-axis — labeling would be inconsistent with the data but no error is raised.
- **D is incorrect:** `np.meshgrid` does not create data in any specific "origin" format. It creates 2D arrays with no notion of display orientation. Dimension mismatches are caused by incorrect meshgrid usage, not by `origin` settings.

</details>

---

### Question 47

A developer notices that `ax.imshow(data)` with the default `aspect='equal'` adds unexpected whitespace above and below the image in a figure with `figsize=(6, 6)`. The array is `(30, 100)`. What causes the whitespace?

- [ ] A) The colorbar is stealing space from the axes
- [ ] B) `aspect='equal'` enforces that each cell has equal width and height in display units — for a wide (30 rows × 100 cols) array in a square figure, the image must be short (30/100 of the figure width) to maintain square pixels, causing whitespace above/below
- [ ] C) Matplotlib automatically adds 10% padding around all images for aesthetic reasons
- [ ] D) The `figsize=(6, 6)` is incompatible with non-square arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `aspect='equal'` enforces that each cell has equal width and height in display units — for a wide (30 rows × 100 cols) array in a square figure, the image must be short (30/100 of the figure width) to maintain square pixels, causing whitespace above/below

**Explanation:**

- **A is incorrect:** No colorbar is mentioned. Whitespace from a colorbar would only appear at the side (for vertical colorbars) — not above and below.
- **B is correct:** `aspect='equal'` (the default) makes each data cell render as a square. In a `(6, 6)` figure, the full width is available for the 100 columns. Each cell gets width = 6"/100 = 0.06" per column. With square pixels, each row also gets 0.06" height. Total image height = 30 × 0.06" = 1.8" — far less than the 6" figure height. The remaining 4.2" appears as whitespace above and below the image (letterboxing). The fix is `aspect='auto'` — which stretches the image to fill the axes rectangle.
- **C is incorrect:** Matplotlib adds standard axis tick/label padding but not a 10% fixed image padding. The whitespace here is caused by aspect ratio enforcement.
- **D is incorrect:** `figsize` is not incompatible with any array shape. The whitespace behavior is a predictable consequence of `aspect='equal'` on a non-square array in a square figure, not an incompatibility.

</details>

---

### Question 48

A developer uses `extent` parameter to set real-world coordinates:

```python
data = np.random.rand(50, 100)
ax.imshow(data, extent=[0, 10, 0, 5], origin='lower', aspect='auto')
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Height (km)")
```

What does `extent=[0, 10, 0, 5]` specify?

- [ ] A) `[left_col, right_col, bottom_row, top_row]` — maps array indices to real-world coordinates
- [ ] B) `[xmin, xmax, ymin, ymax]` — sets the real-world coordinate bounds for the image: x from 0 to 10 km, y from 0 to 5 km
- [ ] C) `[rows, cols, vmin, vmax]` — overrides the array dimensions and color range
- [ ] D) `[left_pad, right_pad, top_pad, bottom_pad]` — whitespace padding in inches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `[xmin, xmax, ymin, ymax]` — sets the real-world coordinate bounds for the image: x from 0 to 10 km, y from 0 to 5 km

**Explanation:**

- **A is incorrect:** `extent` does not use array index notation. It specifies real-world data coordinates in x-y space, not row-column indices.
- **B is correct:** The `extent=[xmin, xmax, ymin, ymax]` parameter in `ax.imshow()` maps the image to a real-world coordinate system: the image x-axis spans [0, 10] and y-axis spans [0, 5]. The 100-column array maps to x∈[0,10] (each column = 0.1 km); the 50-row array maps to y∈[0,5] (each row = 0.1 km). Combined with `origin='lower'`, y=0 is at the bottom. This allows `ax.set_xlabel("Distance (km)")` and `ax.set_ylabel("Height (km)")` to show meaningful real-world tick values on the axes.
- **C is incorrect:** `extent` has no relationship to `vmin`/`vmax` or color scaling. It exclusively controls the spatial coordinate mapping of the image on the axes.
- **D is incorrect:** Whitespace padding (in inches) is controlled by `plt.subplots_adjust()` or `plt.tight_layout()` — not by `extent`. The `extent` parameter has nothing to do with figure margins.

</details>

---

### Question 49

A developer creates two subplots — one with `origin='upper'` and one with `origin='lower'` — using the same data:

```python
data = np.arange(9).reshape(3, 3)
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(data, origin='upper')
ax2.imshow(data, origin='lower')
```

In `ax1` (`origin='upper'`), where is the value `6` displayed? In `ax2` (`origin='lower'`), where is `6` displayed?

- [ ] A) ax1: top-left; ax2: bottom-left
- [ ] B) ax1: bottom-left; ax2: top-left
- [ ] C) ax1: bottom-right; ax2: top-right
- [ ] D) Both axes show `6` at the same position

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) ax1: bottom-left; ax2: top-left

**Explanation:**

- **A is incorrect:** `6` is in `data[2, 0]` — row 2, column 0. With `origin='upper'`, row 2 is at the BOTTOM (not top). With `origin='lower'`, row 2 is at the TOP. So ax1 shows `6` at bottom-left and ax2 shows it at top-left.
- **B is correct:** `data[2, 0] = 6` — row index 2, column index 0. Column 0 is always on the LEFT for both origins. Row index 2 (the last row in a 3-row matrix): **`origin='upper'`** — row 0 at top, row 2 at **bottom** → `6` at **bottom-left** in ax1. **`origin='lower'`** — row 0 at bottom, row 2 at **top** → `6` at **top-left** in ax2. The origin setting vertically mirrors the row order while column order remains the same.
- **C is incorrect:** Column 0 is on the left, not the right. `6` is in column 0 — it cannot appear on the right side.
- **D is incorrect:** `origin='upper'` and `origin='lower'` produce vertically mirrored displays for the same data. The positions of `6` are different in the two subplots.

</details>

---

### Question 50

A developer uses `ax.set_aspect(2.0)` on an `imshow` plot. What does this do relative to the default `aspect='equal'`?

- [ ] A) Makes the y-axis 2× the length of the x-axis in display coordinates, stretching the image vertically by factor 2
- [ ] B) Doubles the figure size
- [ ] C) Makes each pixel 2× as wide as it is tall — stretching the image horizontally by factor 2
- [ ] D) `set_aspect(2.0)` is invalid for `imshow` axes — only strings `'equal'` and `'auto'` are accepted

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Makes the y-axis 2× the length of the x-axis in display coordinates, stretching the image vertically by factor 2

**Explanation:**

- **A is correct:** `ax.set_aspect(ratio)` sets the ratio of y-unit length to x-unit length in display space. `ax.set_aspect(2.0)` means: one unit on the y-axis occupies 2× the display space of one unit on the x-axis. For an image where each cell is 1×1 data unit: cells are rendered 1 unit wide and 2 units tall — producing a vertically stretched image where each cell is twice as tall as wide.
- **B is incorrect:** `set_aspect` modifies the axes coordinate scaling — not the figure size. Figure size is set at creation with `figsize=`.
- **C is incorrect:** `set_aspect(2.0)` makes y-units larger than x-units (ratio > 1 = taller cells). Making cells wider would require `set_aspect(0.5)` (ratio < 1 = shorter/wider cells).
- **D is incorrect:** `ax.set_aspect()` accepts numeric float values in addition to strings `'equal'` and `'auto'`. Numeric values directly set the y/x unit length ratio. This is valid for any axes type including those containing `imshow` images.

</details>
