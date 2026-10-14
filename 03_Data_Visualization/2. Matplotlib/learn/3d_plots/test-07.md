# 3D Matplotlib — Test 07: Surface Stride & Performance Optimization

---

### Question 61

A data engineer benchmarks the rendering time of a `plot_surface` call on a 200×200 grid. He tries four different stride configurations. Rank the following configurations from **fewest** to **most** rendered polygon faces:

- Configuration P: `rstride=1, cstride=1`
- Configuration Q: `rstride=4, cstride=4`
- Configuration R: `rstride=10, cstride=2`
- Configuration S: `rstride=2, cstride=10`

- [ ] A) P < Q < R = S < configuration not listed
- [ ] B) Q < R = S < P, so fewest to most: Q, R=S, P
- [ ] C) All four produce the same number of faces because total polygon count = grid_points / (rstride × cstride)
- [ ] D) Fewest to most faces: Q (≈625), R = S (≈990), P (≈39,601)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Fewest to most faces: Q (≈625), R = S (≈990), P (≈39,601)

**Explanation:**
- **A is incorrect:** Configurations R and S do not produce the fewest faces — Configuration Q with stride=4 in both directions does.
- **B is incorrect:** P produces the most faces, not the fewest. Stride 1 uses every row and column.
- **C is incorrect:** The face count formula is not grid_points / (rstride × cstride). It is `floor((rows-1)/rstride) × floor((cols-1)/cstride)`. For a 200×200 grid: P (rstride=1, cstride=1) → 199×199 ≈ 39,601; Q (rstride=4, cstride=4) → 49×49 ≈ 2,401; R (rstride=10, cstride=2) → 19×99 ≈ 1,881; S (rstride=2, cstride=10) → 99×19 ≈ 1,881. Order fewest to most: Q < R=S < P.
- **D is correct:** With a 200×200 grid (199 intervals per direction): **P** (1,1): 199×199 ≈ 39,601 faces. **Q** (4,4): ~49×49 ≈ 2,401 faces. **R** (10,2): ~19×99 ≈ 1,881 faces. **S** (2,10): ~99×19 ≈ 1,881 faces. R and S produce the same count because multiplication is commutative: 19×99 = 99×19. Fewest to most: Q, then R=S, then P. The option's approximate values indicate approximate calculation; the ordering Q < R=S < P is correct.

</details>

---

### Question 62

A data analyst has a 500×500 meshgrid surface that renders in 45 seconds. She wants to reduce render time to under 5 seconds while preserving an acceptable visual quality. Which stride configuration achieves approximately a 10× speedup?

- [ ] A) `rstride=3, cstride=3`
- [ ] B) `rstride=10, cstride=10`
- [ ] C) `rstride=50, cstride=50`
- [ ] D) `rstride=100, cstride=100`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `rstride=10, cstride=10`

**Explanation:**
- **A is incorrect:** `rstride=3, cstride=3` reduces the face count by approximately 3×3=9×, giving a ~5× speedup (since render time scales roughly with face count). This achieves ~9 seconds, not under 5 seconds.
- **B is correct:** `rstride=10, cstride=10` reduces the rendered face count by approximately 10×10=100× compared to stride 1. Render time scales roughly linearly with polygon count, so a 100× face reduction achieves approximately a 100× speedup (from 45 seconds to ~0.45 seconds). Even with overhead, a 10-stride achieves well under 5 seconds while maintaining recognizable surface shape.
- **C is incorrect:** `rstride=50, cstride=50` on a 500×500 grid selects only 10 rows × 10 columns, producing a 10×10=100-face surface that will look very coarse and jaggy. While it would render in milliseconds, the visual quality would be unacceptably low for most purposes.
- **D is incorrect:** `rstride=100, cstride=100` on a 500×500 grid selects only 5 rows and 5 columns, producing just ~4×4=16 faces. The surface would be too coarse to represent the function meaningfully.

</details>

---

### Question 63

A researcher sets `rstride=0` in a `plot_surface` call. What happens?

- [ ] A) The surface renders with stride=1 (minimum allowed stride, applied when 0 is passed)
- [ ] B) A `ValueError` is raised because stride must be a positive integer (≥1)
- [ ] C) No rows are sampled, producing a surface with only column lines (a series of parallel curves)
- [ ] D) The surface collapses to a flat plane at z=0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `ValueError` is raised because stride must be a positive integer (≥1)

**Explanation:**
- **A is incorrect:** Matplotlib does not silently clamp `rstride=0` to 1. Passing 0 as a stride causes a `ValueError` because stride=0 would result in a division by zero or infinite loop in the row-sampling logic.
- **B is correct:** `rstride` and `cstride` must be positive integers. Setting either to 0 causes Matplotlib to raise a `ValueError` during surface construction because the sampling step size cannot be zero. The error typically occurs inside the `Poly3DCollection` construction when the function attempts to iterate over rows with a step of 0.
- **C is incorrect:** A stride of 0 does not mean "select no rows" — it would mean "step by 0", which is an undefined iteration. Matplotlib raises an error rather than producing column-only rendering.
- **D is incorrect:** The stride parameter controls sampling density, not Z-values. A zero stride would not collapse the surface to z=0. The surface is defined by the Z array, which is unaffected by stride selection.

</details>

---

### Question 64

A developer renders the same surface with two configurations:

```python
# Config A
ax.plot_surface(X, Y, Z, rstride=5, cstride=1)

# Config B
ax.plot_surface(X, Y, Z, rstride=1, cstride=5)
```

The grid is 100×100. How do the two visual results differ?

- [ ] A) Config A and Config B produce identical outputs because rstride and cstride are always applied symmetrically
- [ ] B) Config A creates a surface with coarse row-direction grid lines (5 rows of horizontal wire lines) and fine column-direction grid lines; Config B creates the opposite — fine row lines and coarse column lines
- [ ] C) Config A renders faster because rstride=5 is applied first and terminates the computation earlier
- [ ] D) Config A appears smoother vertically and Config B appears smoother horizontally in the 3D view

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Config A creates a surface with coarse row-direction grid lines (5 rows of horizontal wire lines) and fine column-direction grid lines; Config B creates the opposite — fine row lines and coarse column lines

**Explanation:**
- **A is incorrect:** `rstride` and `cstride` independently control row and column sampling. Different values produce different visual results that are not symmetric.
- **B is correct:** `rstride=5, cstride=1` samples every 5th row (resulting in ~20 row-aligned polylines) but every column (100 column-aligned polylines). The surface polygon cells are tall rectangles (5 rows × 1 column). `rstride=1, cstride=5` does the opposite: every row is sampled (100 row polylines) but every 5th column (~20 column polylines). The polygon cells are wide rectangles (1 row × 5 columns). The visual texture of the mesh is different — elongated in one direction for A and the perpendicular direction for B.
- **C is incorrect:** Matplotlib processes both `rstride` and `cstride` in the same surface construction pass. Neither parameter "terminates the computation earlier." The render time difference between the two is negligible since both configurations produce 100 × 20 = 2,000 polygon faces (just oriented differently).
- **D is incorrect:** "Smoother" in this context is a perceptual judgment. The polygon cell orientation differs, but neither configuration is necessarily "smoother" in a mathematical sense. The two surfaces represent the same function sampled with different cell aspect ratios.

</details>

---

### Question 65

A researcher identifies that her loss surface plot with a 1000×1000 grid takes 3 minutes to render. She considers four optimizations. Which recommendation is MOST effective for reducing render time?

- [ ] A) Switching from `cmap="viridis"` to `cmap="gray"` — sequential colormaps render faster than perceptually uniform ones
- [ ] B) Increasing `rstride` and `cstride` from 1 to 10 — reduces the number of rendered polygon faces by approximately 100×
- [ ] C) Adding `antialiased=False` — disabling anti-aliasing eliminates per-edge smoothing passes
- [ ] D) Reducing `alpha` from 1.0 to 0.9 — semi-transparency requires fewer depth sorting steps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Increasing `rstride` and `cstride` from 1 to 10 — reduces the number of rendered polygon faces by approximately 100×

**Explanation:**
- **A is incorrect:** The choice of colormap (`viridis` vs `gray`) has negligible impact on render time. Both are implemented as lookup tables evaluated element-wise. The dominant cost in 3D surface rendering is polygon geometry processing, not colormap evaluation.
- **B is correct:** Increasing stride from 1 to 10 reduces polygon count from ~999×999≈998,001 faces to ~99×99≈9,801 faces — a reduction factor of approximately 100×. Since Matplotlib's 3D rendering time scales roughly linearly with polygon count (including sorting time which scales super-linearly), a 100× reduction in polygon count translates to a very significant speedup — from 3 minutes to seconds or less. This is by far the most impactful optimization available within Matplotlib.
- **C is incorrect:** `antialiased=False` disables per-edge anti-aliasing smoothing, which is a modest saving. The primary render bottleneck is polygon count and depth sorting, not edge smoothing. The speedup from this option is minor compared to stride adjustment.
- **D is incorrect:** Lower alpha values do not reduce depth sorting steps. Matplotlib's painter's algorithm sorts all polygons by depth regardless of alpha value. Transparency compositing is a per-pixel step and actually adds overhead for semi-transparent surfaces because they must be blended with underlying layers.

</details>

---

### Question 66

A data scientist uses `count_only` mode to estimate how many polygon faces will be rendered before committing to a full render. For a 300×200 grid with `rstride=5, cstride=8`, what is the exact face count?

- [ ] A) `floor((300-1)/5) × floor((200-1)/8) = 59 × 24 = 1,416`
- [ ] B) `(300/5) × (200/8) = 60 × 25 = 1,500`
- [ ] C) `(300-1) × (200-1) / (5 × 8) = 59,701 / 40 ≈ 1,492`
- [ ] D) `300 × 200 / (5 × 8) = 60,000 / 40 = 1,500`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `floor((300-1)/5) × floor((200-1)/8) = 59 × 24 = 1,416`

**Explanation:**
- **A is correct:** The number of polygon faces rendered is determined by the number of row intervals × column intervals after stride sampling. Row intervals: `floor((nrows - 1) / rstride) = floor(299 / 5) = floor(59.8) = 59`. Column intervals: `floor((ncols - 1) / cstride) = floor(199 / 8) = floor(24.875) = 24`. Total faces = 59 × 24 = **1,416**.
- **B is incorrect:** `(300/5) × (200/8) = 60 × 25 = 1,500` does not apply the (n-1) correction for intervals (there are always one fewer intervals than points in each dimension) and does not use floor division. The correct starting count is (300-1) rows of intervals and (200-1) column intervals.
- **C is incorrect:** Dividing total adjacent pairs `(300-1)×(200-1) = 59,701` by `5×8=40` gives 1,492.5, which is not an integer and uses the wrong formula. The correct approach uses `floor` on each dimension separately before multiplying.
- **D is incorrect:** This divides grid points (not intervals) by stride product. Grid points are 300×200=60,000 but each face requires 4 points, and strides must be applied to intervals (N-1), not points (N).

</details>

---

### Question 67

A researcher wants to use `plot_surface` with `rstride` and `cstride` but also wants to ensure the **boundary rows** (first and last row) are always included in the rendering regardless of stride. Which statement is true about default Matplotlib stride behavior?

- [ ] A) Matplotlib always includes the first and last row/column of the grid in the rendered surface, regardless of stride
- [ ] B) Stride selects rows at indices `0, rstride, 2*rstride, ...` — the last row is included only if `(nrows-1)` is divisible by `rstride`
- [ ] C) To force boundary inclusion, the analyst must pass `count_mode='vertex'` as an additional argument
- [ ] D) Boundary rows can only be ensured by setting `rstride=1`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Matplotlib always includes the first and last row/column of the grid in the rendered surface, regardless of stride

**Explanation:**
- **A is correct:** Matplotlib's `plot_surface` implementation always includes the first row (index 0) and the last row (index -1) in the rendered surface, even if the last index is not a multiple of `rstride`. This ensures the surface always has a complete boundary and does not appear cut off or floating. The same applies to columns. Internal rows are sampled at stride intervals, but the first and last are always included.
- **B is incorrect:** While it is true that stride selects rows at 0, rstride, 2*rstride, ..., Matplotlib explicitly handles the boundary case by appending the final row even when (nrows-1) is not evenly divisible by rstride. The boundary is always complete.
- **C is incorrect:** `count_mode='vertex'` is not a valid argument for `plot_surface`. Boundary inclusion is automatic and requires no special arguments.
- **D is incorrect:** Setting `rstride=1` would include every row (ensuring boundaries), but boundaries are included at any stride value by Matplotlib's implementation. `rstride=1` is not the only way to ensure boundary inclusion.

</details>

---

### Question 68

A machine learning engineer renders a hyperparameter search surface on a 50×50 grid and is happy with the visual quality at `rstride=2, cstride=2`. She then upsamples the grid to 500×500 for a publication figure but forgets to adjust strides. What visual problem occurs?

- [ ] A) The surface disappears because 500×500 grids are too large for `plot_surface`
- [ ] B) The surface renders with rstride=2 on 500 rows, producing too many polygon faces (≈ 250×250 ≈ 62,500 faces) making the render very slow and the figure visually over-dense with mesh lines
- [ ] C) The stride of 2 causes aliasing artifacts because 500 is not divisible by 2
- [ ] D) The colormap saturates at the extremes because the increased point density changes the color normalization

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The surface renders with rstride=2 on 500 rows, producing too many polygon faces (≈ 250×250 ≈ 62,500 faces) making the render very slow and the figure visually over-dense with mesh lines

**Explanation:**
- **A is incorrect:** Matplotlib has no hard upper limit on grid size. A 500×500 grid renders (slowly), it does not disappear.
- **B is correct:** `rstride=2, cstride=2` on a 50×50 grid produced ~25×25=625 faces — a comfortable number. On a 500×500 grid with the same stride, the face count becomes ~250×250=62,500 faces — 100× more. This causes dramatically longer render times and, if edge colours are visible, an extremely dense mesh pattern that may look nearly solid and obscure the surface shape. The stride should be scaled proportionally (e.g., `rstride=20, cstride=20`) to maintain similar visual density when the grid size increases.
- **C is incorrect:** 500 is divisible by 2 (500/2=250), so there is no aliasing from indivisibility. More importantly, stride-based aliasing is not a recognized artifact in Matplotlib surface rendering.
- **D is incorrect:** The color normalization is based on Z-value extremes, not on point density. Upsampling the grid adds more points but does not change the function's range or extremes (assuming the same x, y domain and function). Colormap saturation is not caused by increased grid density.

</details>

---

### Question 69

A data scientist compares the output of:

```python
# Plot A: using rcount and ccount (newer API)
ax.plot_surface(X, Y, Z, rcount=30, ccount=30)

# Plot B: using rstride and cstride (older API)
ax.plot_surface(X, Y, Z, rstride=5, cstride=5)
```

The grid `X, Y` has shape `(150, 150)`. What is the key difference between `rcount`/`ccount` and `rstride`/`cstride`?

- [ ] A) `rcount=30` always produces exactly 30 row samples regardless of grid size; `rstride=5` always skips every 5th row regardless of grid size
- [ ] B) `rcount` and `rstride` are identical; `rcount=30` is an alias for `rstride=30`
- [ ] C) `rcount` controls the number of colormap levels; `rstride` controls the mesh resolution
- [ ] D) `rcount=30` is the maximum stride allowed; values above 30 cause a `ValueError`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `rcount=30` always produces exactly 30 row samples regardless of grid size; `rstride=5` always skips every 5th row regardless of grid size

**Explanation:**
- **A is correct:** This is the fundamental distinction between the two APIs. `rcount`/`ccount` specify the **maximum number of samples** (rows or columns) to include in the surface, scaling the effective stride automatically based on grid size. `rcount=30` on a 150-row grid uses an internal stride of `floor(150/30)=5`. On a 300-row grid, it would automatically use stride 10. `rstride`/`cstride`, in contrast, specify a **fixed step size** that does not adapt to grid dimensions. `rstride=5` always selects every 5th row regardless of whether the grid is 50 or 5000 rows. The `rcount`/`ccount` API (introduced in Matplotlib 3.x) is more robust for publication where absolute sample count matters more than fixed stride.
- **B is incorrect:** They have opposite semantics — one specifies absolute sample count, the other specifies a step size. They are not aliases.
- **C is incorrect:** Neither `rcount` nor `rstride` controls colormap levels. Both control mesh sampling density only.
- **D is incorrect:** `rcount=30` is not a maximum stride — it is a maximum sample count parameter. There is no documented upper limit on `rcount` values (other than practical memory constraints).

</details>

---

### Question 70

A researcher sets `rstride=1, cstride=1000` on a 100×100 grid. What is the visual result?

- [ ] A) A standard surface with 100×100 polygon faces (cstride is capped at the grid size)
- [ ] B) A set of parallel curves running in the row direction — only 1 column line is rendered because cstride=1000 exceeds the grid width of 100 columns
- [ ] C) An error: `cstride` cannot exceed the number of columns
- [ ] D) A surface with very wide polygon cells: the 100 rows produce 99 row-intervals but the 1000 column stride means only ~0.1 column intervals → essentially no column-direction wires

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A set of parallel curves running in the row direction — only 1 column line is rendered because cstride=1000 exceeds the grid width of 100 columns

**Explanation:**
- **A is incorrect:** `cstride=1000` on a 100-column grid effectively selects only column 0 (the first boundary column), since the next stride step at column 1000 exceeds the grid. The result is not 100×100 faces.
- **B is correct:** `cstride=1000` on a grid with 100 columns means the only selected column indices are 0 (always included as the first boundary) and possibly 99 (the last boundary, also always included). With virtually no intermediate columns sampled, the "surface" degenerates into a small number of near-degenerate faces or appears as row-direction curves at the column boundaries. The visual resembles a set of parallel polylines rather than a solid surface.
- **C is incorrect:** Matplotlib does not validate that stride values are within the grid bounds. Setting `cstride` larger than the grid dimension is allowed; it simply results in very sparse column sampling.
- **D is incorrect:** The description in D is approximately correct in describing the outcome but arrives at "no column-direction wires" which is not quite right — the boundary columns are still included. The face cells are extremely wide, and only boundary-adjacent faces are rendered.

</details>
