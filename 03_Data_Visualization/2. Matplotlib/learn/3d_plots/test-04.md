# 3D Matplotlib — Test 04: 3D Bar Charts with `bar3d`

---

### Question 31

A data analyst creates a 3D bar chart with:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.bar3d(x=2, y=3, z=0,
         dx=1, dy=1, dz=5,
         color="steelblue", shade=True)
```

Where exactly is the **bottom front-left corner** of this bar, and what is its **height**?

- [ ] A) Bottom corner at (1, 2, 0), height = 5
- [ ] B) Bottom corner at (2, 3, 0), height = 5
- [ ] C) Bottom corner at (2, 3, 5), height = 0
- [ ] D) Bottom corner at (0, 0, 0), height calculated from dx*dy*dz = 5

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Bottom corner at (2, 3, 0), height = 5

**Explanation:**
- **A is incorrect:** The `x` and `y` arguments to `bar3d` directly specify the position of the bottom corner, not offsets from another origin. The bottom-left-front corner is at `(x, y, z) = (2, 3, 0)`, not at `(1, 2, 0)`.
- **B is correct:** In `ax.bar3d(x, y, z, dx, dy, dz)`, the arguments `x=2, y=3, z=0` define the world-space position of the bottom front-left corner of the bar. The argument `dz=5` defines the height of the bar along the z-axis. The bar occupies the volume from `x=2` to `x=3` (span `dx=1`), from `y=3` to `y=4` (span `dy=1`), and from `z=0` to `z=5` (span `dz=5`).
- **C is incorrect:** `(2, 3, 5)` would be the top of the bar, not the bottom corner. The bottom corner is at z=0 and the top is at z=0+dz=5.
- **D is incorrect:** `dx*dy*dz = 1*1*5 = 5` is the volume of the bar in cubic units, not the height. The height is `dz=5`. The origin of the bar is explicitly set by the `(x, y, z)` arguments, not computed from dimension products.

</details>

---

### Question 32

A researcher wants to build a 3D histogram of a 2D joint distribution. She bins data into a 4×3 grid and creates bar heights from bin counts. She sets up:

```python
x_edges = np.array([0, 1, 2, 3, 4])   # 4 bins in x: 0-1, 1-2, 2-3, 3-4
y_edges = np.array([0, 1, 2, 3])       # 3 bins in y: 0-1, 1-2, 2-3
x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1])
x_pos = x_pos.ravel()   # shape (12,)
y_pos = y_pos.ravel()   # shape (12,)
z_pos = np.zeros(12)    # all bars start at z=0
dx = dy = 0.8           # bar footprint (narrower than bin for visual separation)
dz = counts.ravel()     # bar heights from bin counts, shape (12,)

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True)
```

How many individual 3D bars does this call render?

- [ ] A) 7 (sum of x_edges and y_edges lengths)
- [ ] B) 4 bars (one per x-bin, ignoring y)
- [ ] C) 12 bars (one per 2D bin in the 4×3 grid)
- [ ] D) 1 bar (bar3d renders a single bar per call)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 12 bars (one per 2D bin in the 4×3 grid)

**Explanation:**
- **A is incorrect:** The number of bars is the number of 2D bin cells, which is 4 × 3 = 12. Adding the edge array lengths (5 + 4 = 9) or their differences (4 + 3 = 7) has no connection to the number of bars.
- **B is incorrect:** The 2D grid covers both x and y dimensions. There are 4 x-bins × 3 y-bins = 12 cells, not 4.
- **C is correct:** `np.meshgrid` on the lower bin edges creates a 3×4 grid of positions. After `ravel()`, `x_pos` and `y_pos` each have shape `(12,)`. When arrays are passed to `ax.bar3d`, Matplotlib renders one bar per element — so 12 elements → 12 bars. Each bar represents one cell of the 4×3 2D histogram.
- **D is incorrect:** `ax.bar3d` can render multiple bars in a single call when given array arguments. The function iterates over the input arrays and renders one bar per position.

</details>

---

### Question 33

Examine the following `bar3d` call:

```python
ax.bar3d(x=1.5, y=2.0, z=4.0,
         dx=0.5, dy=0.5, dz=3.0,
         color="tomato", shade=False)
```

A junior developer says: "The `z=4.0` argument sets the height of the bar." Is this correct?

- [ ] A) Yes — `z` sets the total height of the bar above the floor
- [ ] B) No — `z` sets the **base elevation** (bottom z-coordinate) of the bar; the height is `dz=3.0`, so the top of the bar is at z=7.0
- [ ] C) No — `z` sets the **center** of the bar in the z-direction; the bar extends from z-dz/2 to z+dz/2, i.e., from 2.5 to 5.5
- [ ] D) No — `z` is always automatically set to 0 by `bar3d`; the provided value is ignored

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `z` sets the **base elevation** (bottom z-coordinate) of the bar; the height is `dz=3.0`, so the top of the bar is at z=7.0

**Explanation:**
- **A is incorrect:** `z` is not the height of the bar. `dz` is the height. The `z` argument specifies the starting elevation (the z-coordinate of the bottom face of the bar).
- **B is correct:** In `ax.bar3d(x, y, z, dx, dy, dz)`: `z` is the **base (bottom) z-coordinate**. The bar rises from z up to z + dz. Here, the bar starts at z=4.0 and has height dz=3.0, so the top face is at z = 4.0 + 3.0 = **7.0**. This is the key distinction: the developer confused the base position with the height. The equivalent of "height" in `bar3d` is always `dz`.
- **C is incorrect:** `bar3d` does not use a center-based parameterization. The `(x, y, z)` triplet is the position of the bottom front-left corner, not the center. The bar does not extend symmetrically around `(x, y, z)`.
- **D is incorrect:** The `z` argument is fully respected by `bar3d` and is not overridden to 0. Floating bars with `z > 0` are a useful feature — for example, stacked bar charts in 3D are created by setting `z` to the top of the previous segment.

</details>

---

### Question 34

A data scientist builds a stacked 3D bar chart with two series per category. The first series bars run from z=0 to z=val1, and the second series bars should stack on top:

```python
# First layer
ax.bar3d(x_pos, y_pos, np.zeros_like(val1), dx, dy, val1, color="steelblue")

# Second layer — which is correct?
```

Which second-layer call correctly stacks the second bars on top of the first?

- [ ] A) `ax.bar3d(x_pos, y_pos, np.zeros_like(val2), dx, dy, val2, color="coral")`
- [ ] B) `ax.bar3d(x_pos, y_pos, val1, dx, dy, val2, color="coral")`
- [ ] C) `ax.bar3d(x_pos, y_pos, val1 + val2, dx, dy, val2, color="coral")`
- [ ] D) `ax.bar3d(x_pos, y_pos, val2, dx, dy, val1, color="coral")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.bar3d(x_pos, y_pos, val1, dx, dy, val2, color="coral")`

**Explanation:**
- **A is incorrect:** Setting `z=np.zeros_like(val2)` places the second layer's bars starting at z=0, which causes them to overlap the first layer from the bottom rather than stacking on top. This produces overlapping bars rather than stacked bars.
- **B is correct:** To stack the second layer on top of the first, the base elevation `z` of the second bar must equal the top of the first bar. The top of the first bar at each position is `0 + val1[i] = val1[i]`. Therefore, setting `z=val1` places the second bar's bottom at the exact height where the first bar ends. The second bar then rises from `val1` to `val1 + val2`, creating a visually stacked effect.
- **C is incorrect:** Setting `z=val1 + val2` places the bottom of the second bar at the total combined height, adding `dz=val2` on top of that. The second bar would extend from `val1+val2` to `val1+2*val2`, not stacking correctly.
- **D is incorrect:** Setting `z=val2` and `dz=val1` swaps the base and height values. The second bar would start at `val2` and extend by `val1`, which is neither stacked correctly nor semantically meaningful.

</details>

---

### Question 35

The `shade=True` parameter in `ax.bar3d` produces which visual effect?

- [ ] A) It applies a shadow cast by the bars onto the floor of the 3D box
- [ ] B) It shades the different faces of each rectangular bar with different luminosities to simulate directional lighting, giving the bars a 3D appearance
- [ ] C) It applies a smooth gradient along the height of each bar to encode the `dz` value as colour intensity
- [ ] D) It draws a semi-transparent background plane behind all bars to improve contrast

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It shades the different faces of each rectangular bar with different luminosities to simulate directional lighting, giving the bars a 3D appearance

**Explanation:**
- **A is incorrect:** `shade=True` does not cast floor shadows. Matplotlib 3D does not support shadow rendering. Shading refers to face luminosity modulation, not drop shadows.
- **B is correct:** `shade=True` in `bar3d` applies a Phong-like shading model that assigns different brightness levels to the top face, front face, and side faces of each rectangular bar based on their orientation relative to a light source direction. This differential luminosity makes the bars appear genuinely 3D rather than flat-coloured boxes, helping viewers distinguish depth and form.
- **C is incorrect:** The shading does not encode the `dz` value as a gradient along the bar's height. Colour can be independently specified via the `color` argument, but `shade=True` modulates luminosity on each face, not along the z-dimension of the bar.
- **D is incorrect:** No background plane is drawn by `shade=True`. The 3D bounding box and grid panes are separate Matplotlib features controlled by `ax.set_pane_color` and grid visibility settings.

</details>

---

### Question 36

A data analyst plots a sales comparison bar chart:

```python
products = ["Alpha", "Beta", "Gamma"]
channels = ["Online", "Retail"]
x_pos = np.array([0, 1, 2])
y_pos_online = np.zeros(3)      # y = 0 for Online
y_pos_retail = np.ones(3)       # y = 1 for Retail
dx, dy = 0.6, 0.6
sales_online = np.array([120, 85, 200])
sales_retail = np.array([95, 140, 110])

ax.bar3d(x_pos, y_pos_online, 0, dx, dy, sales_online, color="steelblue", label="Online")
ax.bar3d(x_pos, y_pos_retail, 0, dx, dy, sales_retail, color="coral", label="Retail")
```

After running this code, why does calling `ax.legend()` produce an empty legend with no entries?

- [ ] A) `ax.bar3d` does not support the `label` argument at all and silently ignores it
- [ ] B) `ax.legend()` must be called immediately after each `bar3d` call, not after all calls are complete
- [ ] C) The `label` argument is accepted by `bar3d` but the resulting `Poly3DCollection` artist may not be automatically included in the automatic legend handle detection in all Matplotlib versions
- [ ] D) 3D axes require `ax.legend(loc="upper left", mode="3d")` with an explicit `mode` argument to display bar labels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The `label` argument is accepted by `bar3d` but the resulting `Poly3DCollection` artist may not be automatically included in the automatic legend handle detection in all Matplotlib versions

**Explanation:**
- **A is incorrect:** `ax.bar3d` does accept the `label` keyword argument and assigns it to the returned `Poly3DCollection` object. The label is stored on the artist, but whether it appears in the auto-legend depends on how Matplotlib collects legend handles.
- **B is incorrect:** `ax.legend()` can be called at any point after artists are added. The order of `legend()` relative to `bar3d` calls does not matter as long as all artists are on the axes when `legend()` is invoked.
- **C is correct:** The `bar3d` function returns a `Poly3DCollection` object. In some Matplotlib versions, `Poly3DCollection` artists are not automatically detected by the legend handle-finding algorithm, resulting in an empty legend even when labels are set. The workaround is to create proxy artists: `from matplotlib.patches import Patch; handles = [Patch(color="steelblue", label="Online"), Patch(color="coral", label="Retail")]; ax.legend(handles=handles)`.
- **D is incorrect:** No `mode="3d"` argument exists for `ax.legend()`. The `legend()` function in Matplotlib 3D works the same as in 2D regarding its keyword arguments.

</details>

---

### Question 37

A developer calculates the total 3D volume occupied by a collection of `bar3d` bars:

```python
dx = 0.8
dy = 0.6
dz = np.array([3, 5, 2, 7, 4])   # heights of 5 bars
```

What is the total volume occupied by all 5 bars combined?

- [ ] A) 21.0 cubic units (sum of dz values only)
- [ ] B) 10.08 cubic units (dx × dy × mean(dz))
- [ ] C) 25.2 cubic units (dx × dy × dz total, summing: 0.8 × 0.6 × 21)
- [ ] D) 2.88 cubic units (dx × dy × min(dz))

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 25.2 cubic units (dx × dy × dz total, summing: 0.8 × 0.6 × 21)

**Explanation:**
- **A is incorrect:** Summing only `dz` values gives 3+5+2+7+4=21, but this ignores the footprint of each bar (dx × dy). The volume of one bar is dx × dy × dz, not just dz.
- **B is incorrect:** `dx × dy × mean(dz) = 0.8 × 0.6 × 4.2 = 2.016`. This computes the average bar volume (one bar's worth), not the total across all 5 bars.
- **C is correct:** Each bar has volume `dx × dy × dz_i = 0.8 × 0.6 × dz_i = 0.48 × dz_i`. Total volume = `0.48 × (3 + 5 + 2 + 7 + 4) = 0.48 × 21 = 10.08`... wait, let me recalculate: `0.8 × 0.6 = 0.48`. `0.48 × 21 = 10.08`. But option C says `0.8 × 0.6 × 21 = 25.2`... Actually `0.8 × 0.6 = 0.48`, not 1.2. `0.48 × 21 = 10.08`. So option C's arithmetic is wrong (it says 25.2 but should be 10.08), and option B's value of 10.08 matches the correct calculation but is described as the "mean" calculation. The correct total volume is 10.08 cubic units.

*(Correction: Option B has the right numerical value 10.08 but wrong description ("mean" instead of "total"). The question intended C to be correct. Let's reassign: correct answer is B with the right value, from the formula dx × dy × sum(dz) = 0.8 × 0.6 × 21 = 10.08.)*

**Revised Correct Answer:** B) 10.08 cubic units

- **B is correct:** Total volume = dx × dy × Σ(dz_i) = 0.8 × 0.6 × (3+5+2+7+4) = 0.48 × 21 = **10.08** cubic units.
- **C is incorrect:** The option states `0.8 × 0.6 × 21 = 25.2`, which is arithmetic error (0.8 × 0.6 = 0.48, not 1.2).
- **A is incorrect:** Sums only heights, ignoring the bar footprint area.
- **D is incorrect:** Uses only the minimum height, not the total.

</details>

---

### Question 38

A data scientist creates a 3D bar chart and sets `color` to an array:

```python
colors = ["steelblue", "coral", "gold", "mediumseagreen"]
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)
```

`x_pos`, `y_pos`, `z_pos`, `dz` each have length 4. What does passing a list of 4 colour strings do?

- [ ] A) Only the first colour is used; `bar3d` ignores all colours after the first element
- [ ] B) Each of the 4 bars is rendered in its corresponding colour from the list
- [ ] C) The colour list is applied cyclically to the 6 faces of each bar
- [ ] D) Matplotlib raises a `ValueError` because `color` must be a single string, not a list

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Each of the 4 bars is rendered in its corresponding colour from the list

**Explanation:**
- **A is incorrect:** When a list of colours with the same length as the number of bars is passed, Matplotlib applies one colour per bar, not just the first.
- **B is correct:** When `color` is a list or array of colour specifications with length equal to the number of bars (here 4), `ax.bar3d` maps one colour to each bar. `colors[0]="steelblue"` applies to the first bar, `colors[1]="coral"` to the second, and so on. This is useful for distinguishing bars by category.
- **C is incorrect:** The colour list length (4) matches the number of bars, not the number of faces per bar (6). Per-face colour assignment would require a different approach (directly modifying the `Poly3DCollection` face colours).
- **D is incorrect:** `bar3d` fully supports list-valued `color` arguments. This is a documented feature for multi-colour 3D bar charts.

</details>

---

### Question 39

A researcher creates a 3D bar visualization of a confusion matrix (4×4 for a 4-class classifier). How many individual bars should the `bar3d` call render for the complete confusion matrix?

- [ ] A) 4 bars (one per class diagonal)
- [ ] B) 8 bars (4 diagonal + 4 off-diagonal summary bars)
- [ ] C) 12 bars (4 rows × 3 off-diagonal elements per row)
- [ ] D) 16 bars (4 predicted classes × 4 true classes = all cells of the 4×4 matrix)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 16 bars (4 predicted classes × 4 true classes = all cells of the 4×4 matrix)

**Explanation:**
- **A is incorrect:** Showing only the diagonal (4 correct-classification cells) omits the 12 off-diagonal misclassification cells. A complete confusion matrix visualization requires all cells.
- **B is incorrect:** 8 bars is an arbitrary partial representation. A full confusion matrix has 4 × 4 = 16 cells, each representing the count of samples from one true class predicted as one predicted class.
- **C is incorrect:** 12 bars omits the diagonal. A complete visualization includes all 16 entries (including the 4 diagonal entries representing correct predictions).
- **D is correct:** A 4-class confusion matrix is a 4×4 grid with 4×4 = 16 cells. Each cell `(i, j)` contains the count of samples from true class i that were predicted as class j. A 3D bar chart places one bar at each of the 16 grid positions, with bar height proportional to the count. The bar chart thus has exactly 16 bars.

</details>

---

### Question 40

A developer incorrectly writes:

```python
ax.bar3d(x=0, y=0, z=0, dx=1, dy=1, dz=0)
```

What does `dz=0` produce?

- [ ] A) A bar with zero height — renders as a flat rectangle (the top face only) at z=0
- [ ] B) A `ValueError: dz must be positive`
- [ ] C) A bar that stretches from z=0 to z=-1 (negative dz flips direction)
- [ ] D) The entire figure is suppressed and nothing renders

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A bar with zero height — renders as a flat rectangle (the top face only) at z=0

**Explanation:**
- **A is correct:** Setting `dz=0` creates a bar with zero height, which collapses the 3D box into a flat surface. Matplotlib does not raise an error for `dz=0`. The bar degenerates into a flat rectangle in the z=0 plane, visible as the top/bottom face without any vertical extent. This is occasionally a useful degenerate case for floor markers, but is usually an unintentional bug when plotting histograms.
- **B is incorrect:** Matplotlib's `bar3d` does not validate that `dz` is strictly positive. Zero and even negative values are geometrically meaningful (a degenerate bar or a downward-extending bar), so no `ValueError` is raised.
- **C is incorrect:** `dz=0` means the bar has zero extent, not that it extends negatively. Negative `dz` values would flip the direction of the bar (extending downward from the base z), but `dz=0` produces a zero-height bar.
- **D is incorrect:** A zero-height bar is a valid (if degenerate) geometric object. Matplotlib renders it as a flat polygon at the base z-level without suppressing the figure.

</details>
