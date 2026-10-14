# Matplotlib Imshow & Heatmaps — Test 08
## Contour & Surface Projections

---

### Question 71

A developer creates a 2D grid for a contour plot:

```python
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)
```

What are the shapes of `X`, `Y`, and `Z`?

- [ ] A) `X.shape = (50,)`, `Y.shape = (50,)`, `Z.shape = (50, 50)`
- [ ] B) `X.shape = (50, 50)`, `Y.shape = (50, 50)`, `Z.shape = (50, 50)` — all three are 2D grids of the same shape
- [ ] C) `X.shape = (50, 1)`, `Y.shape = (1, 50)`, `Z.shape = (50, 50)` — broadcasting arrays
- [ ] D) `X.shape = (2500,)`, `Y.shape = (2500,)`, `Z.shape = (2500,)` — flattened coordinate arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `X.shape = (50, 50)`, `Y.shape = (50, 50)`, `Z.shape = (50, 50)` — all three are 2D grids of the same shape

**Explanation:**

- **A is incorrect:** `X` and `Y` are not 1D after `np.meshgrid()`. `meshgrid` explicitly creates 2D coordinate arrays.
- **B is correct:** `np.meshgrid(x, y)` with `x.shape=(50,)` and `y.shape=(50,)` creates: `X` (50×50) where each row contains the x-values `[-2, ..., 2]`; `Y` (50×50) where each column contains the y-values `[-2, ..., 2]`. `Z = np.sin(X) * np.cos(Y)` applies element-wise operations on the same-shape arrays, producing `Z.shape = (50, 50)`. This is the standard pattern for evaluating any `f(x, y)` over a 2D grid.
- **C is incorrect:** `np.meshgrid()` does NOT return broadcasting-compatible arrays of shapes `(50,1)` and `(1,50)`. That is the output of `np.ogrid[-2:2:50j, -2:2:50j]`. `np.meshgrid()` explicitly broadcasts and returns full 2D arrays.
- **D is incorrect:** `meshgrid` returns 2D arrays, not flattened 1D arrays. Flattened versions would be obtained via `X.ravel()` or `X.flatten()`.

</details>

---

### Question 72

A developer compares `plt.contour()` and `plt.contourf()`. Which statement correctly distinguishes them?

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.contour(X, Y, Z, levels=10)    # Line contours
ax2.contourf(X, Y, Z, levels=10)   # Filled contours
```

- [ ] A) `contour` fills regions with solid colors; `contourf` draws only the boundary lines
- [ ] B) `contour` draws **line contours** (isolines connecting points of equal value); `contourf` draws **filled contours** (colored regions between isoline levels)
- [ ] C) `contourf` is faster than `contour` because it has fewer elements to render
- [ ] D) `contour` requires integer `levels`; `contourf` accepts float levels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `contour` draws **line contours** (isolines connecting points of equal value); `contourf` draws **filled contours** (colored regions between isoline levels)

**Explanation:**

- **A is incorrect:** This reverses the definitions. `contourf` (with the `f` for "filled") fills regions; `contour` (without `f`) draws lines.
- **B is correct:** `plt.contour(X, Y, Z, levels=10)` draws **isolines** — thin curves connecting all points where `Z` has the same value. Each level is a line. `plt.contourf(X, Y, Z, levels=10)` draws **filled regions** — solid color bands between adjacent isoline levels, creating a choropleth-like map. `contourf` is typically combined with a colorbar; `contour` is often used to overlay line labels on top of `contourf`.
- **C is incorrect:** `contourf` is not necessarily faster — filled polygon rendering is computationally more complex than line rendering. The speed difference is typically negligible for standard grid sizes.
- **D is incorrect:** Both `contour` and `contourf` accept integer counts (number of levels) or explicit float arrays for level positions. There is no dtype restriction on the `levels` parameter for either function.

</details>

---

### Question 73

A developer adds **contour line labels** on top of a filled contour plot:

```python
X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = X**2 + Y**2

cf = ax.contourf(X, Y, Z, levels=10, cmap='viridis')
cs = ax.contour(X, Y, Z, levels=10, colors='white', linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=8)
fig.colorbar(cf, ax=ax)
```

What does `ax.clabel(cs, inline=True, fontsize=8)` do?

- [ ] A) Adds a legend entry for each contour line level
- [ ] B) Places numeric labels directly **on** the contour lines, cutting away the line locally (inline=True) so the text is readable against the filled background
- [ ] C) Adds colorbar tick labels aligned with contour levels
- [ ] D) Creates a separate text box outside the axes listing all contour level values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Places numeric labels directly **on** the contour lines, cutting away the line locally (inline=True) so the text is readable against the filled background

**Explanation:**

- **A is incorrect:** `ax.clabel()` does not create legend entries. Legend entries for contour lines require `ax.legend()` with custom handles, not `clabel`.
- **B is correct:** `ax.clabel(cs, inline=True, fontsize=8)` places **contour level labels** (e.g., "2.0", "4.0", "6.0") directly ON the contour lines at strategic positions. `inline=True` removes the contour line segment beneath the text — creating a gap in the line — so the white number is readable against the colored background without overlap. `fontsize=8` controls the label text size. This is the standard technique for labeling line contours overlaid on filled contour plots.
- **C is incorrect:** Colorbar tick labels are managed through the colorbar object (`cbar.set_ticks()`, `cbar.set_ticklabels()`). `ax.clabel()` operates exclusively on contour line sets (`ContourSet` objects).
- **D is incorrect:** `clabel` places labels ON the contour lines within the axes — not in a separate text box outside the axes boundaries.

</details>

---

### Question 74

A developer creates a **loss landscape** contour plot for a 2-parameter model and wants to mark the global minimum:

```python
theta1 = np.linspace(-3, 3, 200)
theta2 = np.linspace(-3, 3, 200)
T1, T2 = np.meshgrid(theta1, theta2)
Loss = (T1 - 0.5)**2 + 2*(T2 + 1.0)**2 + 0.3*T1*T2

cf = ax.contourf(T1, T2, Loss, levels=30, cmap='hot_r')
min_idx = np.unravel_index(np.argmin(Loss), Loss.shape)
ax.plot(T1[min_idx], T2[min_idx], 'g*', markersize=15, label='Global Min')
```

The minimum marker appears at the correct data coordinates. What does `np.unravel_index(np.argmin(Loss), Loss.shape)` compute?

- [ ] A) The row and column indices of the minimum value in the flattened array, converted to 2D indices matching `Loss.shape`
- [ ] B) The x and y data coordinates of the minimum loss value directly
- [ ] C) The gradient of the loss at the minimum point
- [ ] D) The row/column of the maximum (argmax) value — `argmin` returns max for loss functions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The row and column indices of the minimum value in the flattened array, converted to 2D indices matching `Loss.shape`

**Explanation:**

- **A is correct:** `np.argmin(Loss)` returns the **flat index** of the minimum value in the 2D array — treating the array as if it were 1D. For a (200, 200) array, this flat index is in [0, 39999]. `np.unravel_index(flat_idx, Loss.shape)` converts this flat index to a tuple of `(row_idx, col_idx)` in the 2D array. `min_idx = (row, col)`, so `T1[min_idx]` and `T2[min_idx]` retrieve the actual `theta1` and `theta2` coordinates at the minimum — which are then plotted as a green star marker.
- **B is incorrect:** `np.unravel_index` returns array INDEX positions (integers), not data coordinate values. The data coordinates are obtained by indexing into `T1` and `T2` using the returned indices: `T1[min_idx]`, `T2[min_idx]`.
- **C is incorrect:** `np.argmin` and `np.unravel_index` have no relationship to gradient computation. Gradients would require `np.gradient(Loss)` or analytical differentiation.
- **D is incorrect:** `argmin` returns the index of the **minimum** value — not the maximum. For loss functions, we seek the minimum. `argmax` would find the maximum loss.

</details>

---

### Question 75

A developer calls `plt.contourf(X, Y, Z, levels=20)`. The `levels=20` parameter produces 20 filled color bands. A colleague asks: what is the relationship between `levels` and the number of isoline boundaries?

- [ ] A) `levels=20` creates 20 isolines and 20 filled regions — equal number of both
- [ ] B) `levels=20` creates 20 filled color bands and 21 isoline boundaries (one boundary per band edge — top and bottom of each band)
- [ ] C) `levels=20` creates 20 isolines and 19 filled regions (one region between each pair of adjacent isolines)
- [ ] D) `levels=20` creates exactly 20 isolines; filled regions are unlimited and determined by the data range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `levels=20` creates 20 isolines and 19 filled regions (one region between each pair of adjacent isolines)

**Explanation:**

- **A is incorrect:** The number of filled regions is always one less than the number of isoline boundaries. 20 boundaries → 19 regions between them.
- **B is incorrect:** This gets the relationship slightly backwards. `levels=20` in `contourf` creates 20 level **boundaries** (isolines), producing 19 filled bands between them. The 21-boundary description would correspond to 20 bands.
- **C is correct:** When `levels=N` is passed as an integer, `contourf` automatically selects N "nice" level values spanning the data range. These N values are the **isoline boundaries**. Between each pair of adjacent boundaries, there is one filled region. N isolines → N-1 filled regions. For `levels=20`: 20 isoline boundaries → 19 filled color bands. (Note: `contour` with `levels=20` similarly draws 20 lines.)
- **D is incorrect:** The filled regions are not "unlimited" — they are exactly N-1 (one fewer than the number of level boundaries). The filled regions are a fundamental geometric consequence of the boundary count.

</details>

---

### Question 76

A developer creates a contour plot and uses an explicit array for `levels` instead of an integer:

```python
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
ax.contourf(X, Y, Z, levels=levels, cmap='Blues')
```

`Z` has values ranging from 0.0 to 1.0. What happens to regions of `Z` where values are below 0.1 or above 0.9?

- [ ] A) These regions are filled with the minimum and maximum colormap colors respectively
- [ ] B) These regions are **not filled** — `contourf` only fills regions between the specified level boundaries; values outside the level range are left as the axes background color
- [ ] C) An error is raised because all Z values must be within the specified level range
- [ ] D) The missing regions are automatically filled by extending the first and last levels to the data min/max

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) These regions are **not filled** — `contourf` only fills regions between the specified level boundaries; values outside the level range are left as the axes background color

**Explanation:**

- **A is incorrect:** By default, regions outside the specified level range are not filled at all. To fill them with special colors, `extend='both'` must be passed: `ax.contourf(X, Y, Z, levels=levels, extend='both')`.
- **B is correct:** `contourf` fills only the regions between consecutive pairs of the specified levels. With `levels=[0.1, 0.3, 0.5, 0.7, 0.9]`, the filled bands are: [0.1-0.3], [0.3-0.5], [0.5-0.7], [0.7-0.9]. Regions where Z < 0.1 or Z > 0.9 are left **transparent/unfilled** — showing the axes background color (typically white). To extend coverage: `extend='both'` fills Z<0.1 with the colormap minimum and Z>0.9 with the colormap maximum.
- **C is incorrect:** No error is raised when Z values fall outside the level range. The values simply don't get color-filled regions in the output.
- **D is incorrect:** Automatic extension to data min/max does NOT happen by default. This behavior requires the explicit `extend='min'`, `extend='max'`, or `extend='both'` parameter.

</details>

---

### Question 77

A developer creates a **decision boundary visualization** for a binary classifier using `contourf`:

```python
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
            cmap='RdBu', alpha=0.6)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
```

What does `ax.contour(xx, yy, Z, levels=[0.5], ...)` specifically draw?

- [ ] A) A histogram of probability values around 0.5
- [ ] B) A single **decision boundary line** — the isoline connecting all points where the predicted probability equals exactly 0.5 (the classification threshold)
- [ ] C) The 50th percentile contour — which is always at the center of the plot
- [ ] D) A horizontal line at y=0.5 in data coordinates

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A single **decision boundary line** — the isoline connecting all points where the predicted probability equals exactly 0.5 (the classification threshold)

**Explanation:**

- **A is incorrect:** `contour` draws geometric isolines on the 2D spatial grid — not histograms. Histograms require `plt.hist()`.
- **B is correct:** `levels=[0.5]` draws exactly ONE contour line: the set of all `(x, y)` grid points where `Z = model.predict_proba(...)[:, 1] = 0.5`. This is the **decision boundary** — the curve where the classifier assigns equal probability to both classes. Points where Z > 0.5 are classified as class 1 (shown in blue `RdBu` half); points where Z < 0.5 are classified as class 0 (red half). The black `linewidth=2` boundary line makes the decision surface clearly visible against the colored background.
- **C is incorrect:** `Z=0.5` is not necessarily at the geometric center of the plot. The decision boundary's shape and position depend entirely on the model's learned parameters — it could be curved, linear, non-symmetric, or even disconnected.
- **D is incorrect:** `contour` draws a curve in the `(xx, yy)` data space where Z achieves the specified level value — not a horizontal line at a fixed y-coordinate. A horizontal line at y=0.5 would only occur if the decision boundary happened to be a horizontal line, which is a special case.

</details>

---

### Question 78

A developer creates a `contourf` plot and then applies `plt.colorbar()`. They notice the colorbar has only 5 discrete color blocks instead of a smooth gradient. What causes this?

```python
cf = ax.contourf(X, Y, Z, levels=5, cmap='plasma')
fig.colorbar(cf)
```

- [ ] A) The `plasma` colormap only supports 5 discrete colors
- [ ] B) `contourf` with `levels=5` creates 4 filled color bands (between 5 level boundaries) — the colorbar reflects the actual discrete bands rather than a continuous gradient, because there are only 4 distinct regions to represent
- [ ] C) The colorbar always shows discrete blocks when `levels` is less than 10
- [ ] D) `fig.colorbar(cf)` for `contourf` output always produces discrete colorbar blocks — use `fig.colorbar(im)` from `imshow` for smooth gradients

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `contourf` with `levels=5` creates 4 filled color bands (between 5 level boundaries) — the colorbar reflects the actual discrete bands rather than a continuous gradient, because there are only 4 distinct regions to represent

**Explanation:**

- **A is incorrect:** `plasma` is a continuous colormap with 256 color steps — it fully supports smooth gradients. The discreteness comes from the `levels` parameter, not the colormap.
- **B is correct:** `contourf` renders a finite number of filled regions: `levels=5` → 5 boundaries → 4 filled bands. The colorbar for `contourf` reflects the actual data: 4 discrete bands, each with a uniform color sampled from `plasma`. Unlike `imshow` which applies a continuous colormap to a smooth data range, `contourf` is inherently discrete — the colorbar honestly shows the 4 actual colors used. To get a smoother appearance, increase `levels` (e.g., `levels=50`).
- **C is incorrect:** There is no fixed threshold of 10 for discrete vs. smooth appearance. Any `levels` value produces that exact number of discrete bands. `levels=100` produces 99 bands that LOOK continuous; `levels=5` produces 4 clearly discrete bands.
- **D is incorrect:** `fig.colorbar(cf)` correctly reflects the contourf's discrete band structure. `fig.colorbar(im)` from `imshow` shows a smooth gradient because `imshow` maps continuous data values — the difference is in the plotting function, not the colorbar call.

</details>

---

### Question 79

A developer creates a **3D surface projection** using `ax.contourf()` with the `zdir` parameter:

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax.contourf(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
```

What does `zdir='z', offset=Z.min()` accomplish for the `contourf` projection?

- [ ] A) Flips the contour plot vertically below the z-axis
- [ ] B) Projects the filled contour plot onto the **floor** of the 3D axes at the z-coordinate `Z.min()` — creating a shadow-like 2D projection beneath the 3D surface
- [ ] C) Scales the z-values of the contourf by `Z.min()` as a normalization factor
- [ ] D) Removes all Z values below the minimum — equivalent to clipping

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Projects the filled contour plot onto the **floor** of the 3D axes at the z-coordinate `Z.min()` — creating a shadow-like 2D projection beneath the 3D surface

**Explanation:**

- **A is incorrect:** `zdir='z'` and `offset=Z.min()` do not flip any axes. They specify the projection plane and its position.
- **B is correct:** In 3D Matplotlib axes, `ax.contourf()` accepts `zdir` to specify which axis is "depth" and `offset` to specify where the 2D contour is placed along that axis. `zdir='z', offset=Z.min()` places the filled contour in the **horizontal plane** (the "floor") at z=Z.min() — the lowest z-coordinate of the surface. This creates a **plan-view projection** that appears as a colored shadow beneath the 3D surface, helping viewers understand the spatial structure from both 3D and top-down perspectives simultaneously.
- **C is incorrect:** `offset` specifies the 3D coordinate position of the projection plane — not a scaling factor. It has units matching the z-axis data.
- **D is incorrect:** `offset=Z.min()` does not clip the data. All Z values are still used to compute the contour levels — the `offset` only determines WHERE in 3D space the resulting 2D contour is rendered.

</details>

---

### Question 80

A developer creates a `contourf` visualization and adds `extend='both'` to handle out-of-range values:

```python
levels = np.linspace(-2, 2, 11)  # [-2, -1.6, ..., 1.6, 2]
cf = ax.contourf(X, Y, Z, levels=levels, cmap='RdBu_r', extend='both')
```

Z has values ranging from -3.5 to +3.5. What color do regions where Z > 2 receive, and how is this controlled?

- [ ] A) White — values above the colormap maximum always appear white
- [ ] B) The colormap's **over color** — by default the maximum colormap color; customizable with `cmap.set_over('darkred')`
- [ ] C) Transparent — out-of-range regions become invisible with `extend='both'`
- [ ] D) The same color as the Z=2.0 level — the last defined color is extended

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The colormap's **over color** — by default the maximum colormap color; customizable with `cmap.set_over('darkred')`

**Explanation:**

- **A is incorrect:** Values above the maximum level do not default to white. They receive the colormap's "over" color — which defaults to the colormap's maximum color (the last color in the colormap lookup table).
- **B is correct:** With `extend='both'`, `contourf` fills regions outside the level range using the colormap's **under** color (for Z < levels.min()) and **over** color (for Z > levels.max()). By default, the over color is the last color of the colormap (for `RdBu_r`: the dark red extreme). To customize: `cmap = plt.cm.RdBu_r.copy(); cmap.set_over('darkred'); cmap.set_under('darkblue')`. The `extend` parameter on the `fig.colorbar(cf, extend='both')` also adds triangular arrows at both ends of the colorbar indicating the extended ranges.
- **C is incorrect:** `extend='both'` does NOT make out-of-range regions transparent. It fills them with the over/under colors. Transparency would require alpha channel manipulation.
- **D is incorrect:** The over color is a separate special color from the last defined level color. By default they may be similar (both at the colormap extreme), but they are technically different: the last level color comes from the colormap LUT at position 1.0, while the over color is a special attribute that can be independently set.

</details>
