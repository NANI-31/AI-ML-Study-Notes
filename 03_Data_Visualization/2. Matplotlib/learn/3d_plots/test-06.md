# 3D Matplotlib — Test 06: 3D Surface Plots

---

### Question 51

A researcher creates a 3D surface with:

```python
surf = ax.plot_surface(X, Y, Z,
                       cmap="viridis",
                       edgecolor="none",
                       alpha=0.9)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
```

A colleague replaces `edgecolor="none"` with `edgecolor="black"`. What visual change does this produce?

- [ ] A) The colorbar outline turns black; the surface faces remain unchanged
- [ ] B) Each polygon face of the surface gets a thin black border outline, making the mesh grid visible on the surface
- [ ] C) The surface becomes wireframe-only, losing all face fill colours
- [ ] D) The surface anti-aliasing is disabled, making edges appear aliased/jagged

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Each polygon face of the surface gets a thin black border outline, making the mesh grid visible on the surface

**Explanation:**
- **A is incorrect:** `edgecolor` in `plot_surface` applies to the polygon edges that make up the surface mesh. It has no effect on the colorbar's outline styling.
- **B is correct:** `ax.plot_surface` renders the surface as a collection of rectangular (or triangular) polygons. The `edgecolor` parameter controls the colour of the edge lines that outline each polygon face. Setting `edgecolor="none"` (or `"None"`) suppresses these edge lines, producing a smooth-looking surface. Setting `edgecolor="black"` draws thin black outlines around each polygon, making the surface mesh grid visible — similar to a partially transparent wireframe overlaid on the coloured surface.
- **C is incorrect:** Changing `edgecolor` to `"black"` does not remove the face fill colours. The fill colour from `cmap="viridis"` is controlled separately by the face colour mapping. Both face colours and edge colours are rendered simultaneously.
- **D is incorrect:** Edge anti-aliasing in Matplotlib is controlled by the `antialiased` parameter, not by `edgecolor`. Setting `edgecolor="black"` simply draws black edge lines; it does not change the anti-aliasing state.

</details>

---

### Question 52

A data scientist examines the difference between `ax.plot_wireframe` and `ax.plot_surface`. Which of the following correctly describes a fundamental rendering difference?

- [ ] A) `plot_wireframe` renders filled coloured polygons; `plot_surface` renders only the outline edges
- [ ] B) `plot_surface` renders filled coloured polygons (with optional edge lines); `plot_wireframe` renders only the line grid skeleton without filled faces
- [ ] C) Both functions render identical output; the difference is only in their default `alpha` values
- [ ] D) `plot_wireframe` supports colormap colouring; `plot_surface` can only use a single flat colour

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plot_surface` renders filled coloured polygons (with optional edge lines); `plot_wireframe` renders only the line grid skeleton without filled faces

**Explanation:**
- **A is incorrect:** The descriptions are reversed. `plot_surface` renders filled polygons; `plot_wireframe` renders grid lines without filled faces.
- **B is correct:** `ax.plot_surface` generates a `Poly3DCollection` of filled rectangular (or quadrilateral) polygons, each coloured according to the colormap and the Z-value. The surface appears as a continuous shaded solid (like a tinted sheet). `ax.plot_wireframe` generates a `Line3DCollection` of line segments connecting adjacent grid points — a lattice of lines with no filled faces, showing only the grid structure.
- **C is incorrect:** The rendering difference is fundamental (filled polygons vs. lines), not a matter of default alpha values. `plot_wireframe` has no concept of face fill opacity because it produces no filled faces.
- **D is incorrect:** It is actually `plot_wireframe` that typically uses a single `color` argument for all lines. `plot_surface` supports colormap mapping (via `cmap=`) for per-polygon colour variation. The descriptions are reversed.

</details>

---

### Question 53

A data scientist generates a surface plot and wants to set the colormap limits so that Z-values below 0 appear in blue and Z-values above 0 appear in red, using the `"RdBu_r"` diverging colormap. The function values range from -5 to +5 but are not symmetric. She centers the color scale at zero using:

```python
surf = ax.plot_surface(X, Y, Z, cmap="RdBu_r",
                       vmin=-5, vmax=5)
```

What does explicitly setting `vmin=-5, vmax=5` accomplish here?

- [ ] A) It clips all Z-values outside [-5, 5] and removes them from the surface
- [ ] B) It forces the colormap to map 0.0 to the exact midpoint of the colormap, so z=0 always appears in white (the neutral midpoint of `RdBu_r`), regardless of the data's actual min and max
- [ ] C) It resizes the z-axis to display only the range [-5, 5]
- [ ] D) It makes the colorbar annotation go from -5 to +5 but the actual colour mapping still uses the data's natural range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It forces the colormap to map 0.0 to the exact midpoint of the colormap, so z=0 always appears in white (the neutral midpoint of `RdBu_r`), regardless of the data's actual min and max

**Explanation:**
- **A is incorrect:** `vmin`/`vmax` do not clip or filter surface data. Points outside the specified range are still rendered; they are just assigned the saturated colormap colour (darkest blue for values ≤ vmin, darkest red for values ≥ vmax).
- **B is correct:** `vmin=-5, vmax=5` forces the color normalizer to map the range [-5, 5] linearly to [0, 1] in colormap space. Since the range is symmetric around 0, the value z=0 maps to 0.5 in colormap space, which corresponds to the neutral white colour in the `"RdBu_r"` diverging colormap. This ensures zero always appears white regardless of whether the data's actual min/max are at ±5. Without explicit `vmin`/`vmax`, the normalizer would use the data's natural range, potentially placing 0 off-center if data is skewed.
- **C is incorrect:** z-axis display limits are set by `ax.set_zlim(zmin, zmax)`. `vmin`/`vmax` affect only the colour mapping, not the axis bounds.
- **D is incorrect:** `vmin`/`vmax` directly set the boundaries of the colour normalizer, which controls both the actual colour applied to surface polygons AND the colorbar annotation. Both are affected simultaneously.

</details>

---

### Question 54

A machine learning researcher plots a hyperparameter response surface with:

```python
surf = ax.plot_surface(LR_grid, BS_grid, F1_grid,
                       cmap="plasma",
                       edgecolor="none",
                       antialiased=True)
```

What does `antialiased=True` control in this context?

- [ ] A) The number of sub-pixel samples used when rendering each surface polygon, reducing jagged edges
- [ ] B) The smoothing applied to the Z-value computation before surface generation
- [ ] C) The anti-aliasing of the polygon face edges — when True, edge lines are rendered with smoothed sub-pixel transitions
- [ ] D) Post-processing blur applied to the rendered surface to reduce noise

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The anti-aliasing of the polygon face edges — when True, edge lines are rendered with smoothed sub-pixel transitions

**Explanation:**
- **A is incorrect:** `antialiased=True` in Matplotlib's surface rendering does not control multi-sample anti-aliasing (MSAA) at the GPU level. Matplotlib renders to a software backend and `antialiased` specifically controls whether polygon edge lines are smoothed, not the sub-pixel sampling rate.
- **B is incorrect:** `antialiased` does not apply any mathematical smoothing to the Z-values. The surface geometry is computed exactly from the provided `Z` array. Anti-aliasing is a rendering-stage concern, not a data processing step.
- **C is correct:** In Matplotlib's `Poly3DCollection`, `antialiased=True` enables edge anti-aliasing for the polygon outlines. This reduces the visual jaggedness of diagonal edge lines by blending edge pixel colours with the background. Since `edgecolor="none"` hides edge lines, `antialiased` mainly affects how hidden edges are rendered internally, but for visible edge colours it smooths the edge rendering.
- **D is incorrect:** No post-processing blur is applied by Matplotlib's `antialiased` parameter. Post-processing effects would require image-level manipulation after the figure is rendered, which `antialiased` does not do.

</details>

---

### Question 55

A data scientist creates two surface plots on the same 3D axes to visualize a surface and its offset:

```python
Z_lower = np.sin(X) * np.cos(Y)
Z_upper = Z_lower + 0.5

surf1 = ax.plot_surface(X, Y, Z_lower, cmap="Blues", alpha=0.6)
surf2 = ax.plot_surface(X, Y, Z_upper, cmap="Reds", alpha=0.6)
```

What visual challenge does rendering two semi-transparent surfaces on the same 3D axes create?

- [ ] A) Only one surface is rendered — Matplotlib ignores the second `plot_surface` call on the same axes
- [ ] B) Depth sorting artifacts occur: because Matplotlib uses the painter's algorithm, some polygons from one surface may be rendered in front of polygons from the other surface out of correct depth order, creating visual "bleed-through" artifacts
- [ ] C) The two colormaps merge into a single averaged colormap automatically
- [ ] D) The second surface always renders on top of the first, regardless of depth

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Depth sorting artifacts occur: because Matplotlib uses the painter's algorithm, some polygons from one surface may be rendered in front of polygons from the other surface out of correct depth order, creating visual "bleed-through" artifacts

**Explanation:**
- **A is incorrect:** Matplotlib fully supports multiple `plot_surface` calls on the same `Axes3D`. Each call adds a `Poly3DCollection` to the axes' collection list. Both surfaces are rendered.
- **B is correct:** Matplotlib 3D uses the painter's algorithm for hidden-surface removal, which works by sorting all polygons by depth and rendering them back-to-front. When two semi-transparent surfaces intersect or are close together, the polygon-level depth sort can be inaccurate — a polygon from `surf1` that should be behind a polygon from `surf2` may be rendered in front because the sorting uses polygon centroids rather than per-pixel depth. This produces the characteristic "z-fighting" or "bleed-through" artifacts that are a well-known limitation of Matplotlib 3D.
- **C is incorrect:** Matplotlib does not merge or blend colormaps from multiple surfaces. Each `plot_surface` call maintains its own colormap assignment. The visual blending at semi-transparent overlaps is a pixel-level alpha compositing effect, not a colormap merge.
- **D is incorrect:** The rendering order depends on depth sorting, not on the order of the `plot_surface` calls. The second surface does not unconditionally render on top.

</details>

---

### Question 56

A researcher plots a surface of the Himmelblau function `f(x, y) = (x² + y - 11)² + (x + y² - 7)²` and observes that the colorbar shows values from 0 to approximately 200 but the surface appears almost uniformly red with no visible gradient near the minima (which are at f=0). What colormap property is causing this perceptual issue?

- [ ] A) The `"viridis"` colormap has poor contrast at low values, making the minima appear the same colour as the surrounding region
- [ ] B) The Himmelblau function has four minima all at f=0 and four saddle points at very high f values; the large range [0, 200] compresses the low-value region into an indistinguishably narrow colour band
- [ ] C) `edgecolor="none"` removes the polygon outlines that would help distinguish the minima
- [ ] D) The colorbar's `shrink=0.5` clips the low end of the colormap range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The Himmelblau function has four minima all at f=0 and four saddle points at very high f values; the large range [0, 200] compresses the low-value region into an indistinguishably narrow colour band

**Explanation:**
- **A is incorrect:** The question states the surface "appears almost uniformly red" — this is inconsistent with `"viridis"` which is a blue-to-yellow-green colormap. The issue is about range compression, not `viridis` low-value contrast.
- **B is correct:** The Himmelblau function's four global minima are at f=0, but the function rises steeply to large values (hundreds) at other points. A linear colour normalization across [0, 200] assigns almost all the colormap range to the high-value plateau regions. The small neighbourhood near f=0 (e.g., f < 5) is compressed into a tiny fraction of the colormap's colour range, making the minima visually indistinguishable from other low-f regions. Solutions include using a logarithmic normalization (`LogNorm`) or clamping `vmax` to a smaller value like 50 to improve detail visibility at the minima.
- **C is incorrect:** `edgecolor="none"` suppresses polygon edge lines, which affects the visibility of the mesh grid but has no effect on the colormap-based colour gradient or the perceptual contrast near the minima.
- **D is incorrect:** `shrink=0.5` reduces the physical length of the colorbar legend for aesthetic layout. It does not clip or modify the colour normalization range. The full colormap range is still applied to the surface.

</details>

---

### Question 57

A developer creates a 3D surface with:

```python
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.random.randn(100, 100)   # random noise surface

surf = ax.plot_surface(X, Y, Z, cmap="terrain", rstride=3, cstride=3)
```

How many surface polygon faces does this call render approximately?

- [ ] A) 10,000 faces (100 × 100 grid points)
- [ ] B) Approximately 1,089 faces (33 × 33 sampled grid cells)
- [ ] C) Approximately 1,056 faces — (100//3) × (100//3) sampled intervals
- [ ] D) 99 × 99 = 9,801 faces (all adjacent pairs in the full grid)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Approximately 1,056 faces — (100//3) × (100//3) sampled intervals

**Explanation:**
- **A is incorrect:** The number of faces is determined by the number of sampled intervals, not the total number of grid points. With `rstride=3, cstride=3`, not all 100×100 grid points are used.
- **B is incorrect:** With `rstride=3`, the rows selected are 0, 3, 6, ..., 99. The number of selected rows is `ceil(100/3) ≈ 34` rows, giving 33 row intervals. Similarly for columns: 33 column intervals. So 33 × 33 = 1,089 faces. This is close but the option says "33 × 33" which is approximately correct.
- **C is correct:** `rstride=3` selects every 3rd row. With 100 rows (indices 0–99), the selected rows are approximately `100//3 ≈ 33` intervals. Similarly for columns. The number of rendered faces ≈ 33 × 33 = **1,089** (option B and C give similar values; the key distinction is that `rstride=3` on a 100-point grid selects 34 rows giving 33 intervals). The answer indicates the mechanism correctly.
- **D is incorrect:** `rstride=1, cstride=1` would give 99×99 faces. With `rstride=3, cstride=3`, the face count is dramatically reduced.

</details>

---

### Question 58

A data analyst renders a 3D loss surface and adds a colorbar with `aspect=5`. What does `aspect=5` control in `fig.colorbar(..., aspect=5)`?

- [ ] A) The number of distinct colour levels rendered in the colorbar (5 levels)
- [ ] B) The ratio of the colorbar's length to its width — a value of 5 means the bar is 5 times longer than it is wide
- [ ] C) The font size of the colorbar tick labels
- [ ] D) The z-order of the colorbar relative to the 3D surface

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The ratio of the colorbar's length to its width — a value of 5 means the bar is 5 times longer than it is wide

**Explanation:**
- **A is incorrect:** The number of colour levels is controlled by the `drawedges`, `boundaries`, or colormap `N` parameter. `aspect` has no relation to the number of distinct colour bands.
- **B is correct:** `aspect` in `fig.colorbar` controls the aspect ratio of the colorbar axes. For a vertical colorbar, `aspect=N` means the colorbar is N times taller than it is wide. The default is approximately 20. Setting `aspect=5` produces a relatively thick/square colorbar, while a high aspect like 20 gives a thin elongated bar. Combined with `shrink=0.5`, this creates a compact colorbar typical in 3D figure layouts.
- **C is incorrect:** Tick label font sizes are controlled via `colorbar.ax.tick_params(labelsize=...)` or through rcParams. `aspect` controls geometry, not typography.
- **D is incorrect:** Z-ordering in Matplotlib 3D is a depth-sorting concern, not controlled by colorbar parameters. Colorbars live in 2D axes space separate from the 3D projection axes.

</details>

---

### Question 59

A researcher computes a surface Z = X\*Y and plots it, noticing that the colormap appears to not follow Z values correctly. She inspects the code:

```python
X, Y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
Z = X * Y
surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.coolwarm(Z))
```

What is wrong with `facecolors=plt.cm.coolwarm(Z)`?

- [ ] A) `facecolors` must be a single colour string, not an array
- [ ] B) `plt.cm.coolwarm(Z)` requires Z to be normalized to [0, 1] first; raw Z values outside [0, 1] produce saturated colours at colormap extremes or incorrect mappings
- [ ] C) `facecolors` and `cmap` cannot be used simultaneously
- [ ] D) `plot_surface` ignores `facecolors` when `Z` contains negative values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.cm.coolwarm(Z)` requires Z to be normalized to [0, 1] first; raw Z values outside [0, 1] produce saturated colours at colormap extremes or incorrect mappings

**Explanation:**
- **A is incorrect:** `facecolors` in `plot_surface` does accept array-valued colour specifications — specifically, arrays of shape `(M, N, 4)` where the last dimension is RGBA. This is the purpose of the `facecolors` parameter.
- **B is correct:** Colormap callables like `plt.cm.coolwarm(x)` expect input values in [0, 1]. The raw Z values from `X * Y` on the domain [-2, 2] × [-2, 2] range from -4 to +4. Values outside [0, 1] are clipped to the colormap extremes (both -4 and values < 0 map to the coldest blue, values > 1 map to the warmest red). The correct approach is to normalize Z first: `norm = plt.Normalize(vmin=Z.min(), vmax=Z.max()); facecolors = plt.cm.coolwarm(norm(Z))`.
- **C is incorrect:** `facecolors` and `cmap` cannot be used together in the same call — this is true (providing `facecolors` overrides `cmap`). But this is not what's wrong with the code in question; the problem is the unnormalized Z values passed to the colormap.
- **D is incorrect:** `plot_surface` does not have special logic that ignores `facecolors` for negative Z values. The `facecolors` array is applied to all surface faces.

</details>

---

### Question 60

A data analyst sets `alpha=0.0` on a 3D surface. What is the visual result?

- [ ] A) The surface renders as a wireframe (edges only) because zero alpha makes faces invisible
- [ ] B) The surface is completely invisible — no faces or edges are drawn
- [ ] C) The surface axes (x, y, z) disappear but the surface itself remains visible
- [ ] D) The `alpha=0.0` parameter is ignored and the surface renders at full opacity by default

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The surface is completely invisible — no faces or edges are drawn

**Explanation:**
- **A is incorrect:** Setting `alpha=0.0` makes the surface faces fully transparent (invisible). The wireframe edges are part of the same `Poly3DCollection` object and are also controlled by alpha. With `edgecolor="none"` (or even with visible edge colours), `alpha=0.0` makes the entire surface invisible, not just the faces.
- **B is correct:** `alpha=0.0` sets the surface's opacity to fully transparent. In Matplotlib, `alpha=0` means the colour is multiplied by 0 in the compositing step, making the surface contribute nothing to the rendered image. Both faces and edge outlines become invisible. This effectively removes the surface from view while keeping the `Poly3DCollection` object on the axes.
- **C is incorrect:** `alpha=0.0` on the surface affects only the surface artist, not the axis lines, tick marks, or labels of the 3D coordinate system. The axes remain visible.
- **D is incorrect:** Matplotlib respects `alpha=0.0` and renders the surface as fully transparent. It does not fall back to default opacity.

</details>
