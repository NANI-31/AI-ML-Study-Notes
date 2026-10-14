# 3D Matplotlib — Test 08: 3D Contour Projections

---

### Question 71

A researcher plots a surface and adds a contour projection to the **floor** (z-wall):

```python
Z_min = Z.min()
ax.contour(X, Y, Z, zdir='z', offset=Z_min, cmap='coolwarm', levels=10)
```

What exactly does `zdir='z', offset=Z_min` tell Matplotlib?

- [ ] A) Draw contour lines in the z-direction and scale their height to Z_min
- [ ] B) Project the 2D level curves onto the horizontal plane at `z = Z_min`, displaying them as "shadow" contours on the floor of the 3D box
- [ ] C) Rotate the contour lines so they face the z-axis viewer at elevation Z_min degrees
- [ ] D) Draw contours at Z_min above the xy-plane using the z-axis as the normal direction

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Project the 2D level curves onto the horizontal plane at `z = Z_min`, displaying them as "shadow" contours on the floor of the 3D box

**Explanation:**
- **A is incorrect:** `zdir` specifies the plane onto which contours are projected, not the "direction" of line drawing in a spatial sense. It does not scale heights.
- **B is correct:** `ax.contour(X, Y, Z, zdir='z', offset=Z_min)` computes the 2D level curves of Z in the XY plane and then places (projects) those curves at the fixed z-coordinate `z = Z_min`. Since Z_min is the lowest z-value of the surface, this effectively draws a "shadow" or "footprint" of the surface's topographic contours on the floor of the 3D bounding box. This is a standard technique for showing the overall shape of a 3D surface from above without obscuring the surface itself.
- **C is incorrect:** `offset` is a coordinate value in the world space, not an angle in degrees. There is no rotation of contour lines involved.
- **D is incorrect:** The contours are placed AT `z = Z_min`, not "Z_min above the xy-plane". If Z_min is negative, the contours appear below z=0. The `zdir='z'` specifies that the projection plane is perpendicular to the z-axis (i.e., a horizontal plane), and `offset` gives the z-coordinate of that plane.

</details>

---

### Question 72

A data scientist adds three contour projections to a loss surface:

```python
X_min, X_max = X.min(), X.max()
Y_min, Y_max = Y.min(), Y.max()
Z_min = Z.min()

ax.contour(X, Y, Z, zdir='z', offset=Z_min, cmap='hot_r', levels=8)
ax.contour(X, Y, Z, zdir='x', offset=X_min, cmap='hot_r', levels=8)
ax.contour(X, Y, Z, zdir='y', offset=Y_max, cmap='hot_r', levels=8)
```

On which three "walls" of the 3D bounding box do these projections appear?

- [ ] A) Floor (bottom), left wall (x=X_max), back wall (y=Y_min)
- [ ] B) Floor (z=Z_min), left wall (x=X_min), back wall (y=Y_max)
- [ ] C) Ceiling (top), front wall (x=X_min), right wall (y=Y_max)
- [ ] D) Floor (z=Z_min), front wall (x=X_max), right wall (y=Y_min)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Floor (z=Z_min), left wall (x=X_min), back wall (y=Y_max)

**Explanation:**
- **A is incorrect:** `zdir='x', offset=X_min` places contours on the wall at x=X_min (left wall), not x=X_max. `zdir='y', offset=Y_max` places contours at y=Y_max (back wall), not y=Y_min.
- **B is correct:** Each `contour` call with `zdir` projects the level curves onto a plane perpendicular to the specified direction at the given offset. `zdir='z', offset=Z_min` → projects onto the floor plane at z=Z_min. `zdir='x', offset=X_min` → projects onto the left wall (plane at x=X_min). `zdir='y', offset=Y_max` → projects onto the back wall (plane at y=Y_max). These three walls are standard choices for shadow-projection visualizations because they are visible from the default camera angle.
- **C is incorrect:** The ceiling (top) would require `zdir='z', offset=Z_max`, not Z_min. And the front wall would be at x=X_max, not X_min.
- **D is incorrect:** The floor is correctly at Z_min, but `zdir='x', offset=X_max` would give the right wall (not front wall at X_min) and `zdir='y', offset=Y_min` gives a front/bottom y-wall, not the back y-wall at Y_max.

</details>

---

### Question 73

A researcher forgets to include `offset` in a contour projection:

```python
ax.contour(X, Y, Z, zdir='z', cmap='plasma', levels=6)
```

What happens to the contour lines?

- [ ] A) A `TypeError` is raised because `offset` is a required argument when `zdir` is specified
- [ ] B) The contours default to `offset=0` and are drawn on the z=0 plane
- [ ] C) The contour lines float at their actual Z-values on the surface — they appear embedded in the surface face rather than projected onto a wall
- [ ] D) The contour lines are drawn at the Z-values where each contour level intersects the surface, creating a set of floating curves at varying heights in the 3D space

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The contour lines are drawn at the Z-values where each contour level intersects the surface, creating a set of floating curves at varying heights in the 3D space

**Explanation:**
- **A is incorrect:** `offset` is an optional parameter for `ax.contour`. Omitting it does not raise an error.
- **B is incorrect:** When `offset` is not provided, the contours are not automatically placed at z=0. The default behavior without `offset` draws contours at their natural z-levels rather than projecting them onto a fixed plane.
- **C is incorrect:** Without `offset`, the contours appear at their actual z-coordinate levels, but they are separate line artists drawn in 3D space — they are not "embedded in the surface face" (which would require surface-level rendering at those exact coordinates).
- **D is correct:** When `offset` is omitted, `ax.contour` with `zdir='z'` draws each contour level at its natural z-height. The contour line for level `L` appears at z=L in 3D space, creating floating contour rings at different heights. These "floating" contours can look unanchored and visually confusing, which is why the `offset` argument is typically used to project all contours onto a single reference plane (the bounding wall).

</details>

---

### Question 74

A data analyst plots a 3D surface and adds floor contour projection:

```python
X_min = X.min()   # = -3.0
ax.contour(X, Y, Z, zdir='x', offset=X_min, cmap='viridis', levels=5)
```

On which plane do these contours appear, and what do they represent geometrically?

- [ ] A) The horizontal plane at z=-3.0; they show the Z-value distribution from above
- [ ] B) The vertical plane at x=-3.0 (the left-side wall of the 3D box); they show the projection of Z-level curves as viewed from the left
- [ ] C) The plane parallel to the xz-plane passing through y=-3.0
- [ ] D) The contours are drawn on the surface itself at y=-3.0 cross-sections

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The vertical plane at x=-3.0 (the left-side wall of the 3D box); they show the projection of Z-level curves as viewed from the left

**Explanation:**
- **A is incorrect:** `zdir='x'` projects onto a plane perpendicular to the x-axis (a yz-plane), not a horizontal plane. The horizontal plane (floor) uses `zdir='z'`.
- **B is correct:** `zdir='x'` specifies that the projection plane is perpendicular to the x-axis. The `offset=X_min=-3.0` sets the x-coordinate of this plane at x=-3.0. In a standard 3D bounding box where x ranges from -3 to +3, x=-3.0 is the left boundary wall. The contours show how the z-values of the surface project onto this yz-plane — effectively a "side view" shadow of the surface's level curves when viewed from the negative x direction.
- **C is incorrect:** A plane parallel to the xz-plane would use `zdir='y'`, not `zdir='x'`. `zdir='y'` produces projections on a plane at a fixed y-value.
- **D is incorrect:** The contours are not drawn on the surface itself. They are projected onto the fixed plane at x=X_min, creating a wall-mounted shadow visualization separate from the 3D surface.

</details>

---

### Question 75

A machine learning researcher creates three shadow projections of a loss landscape and notices that the floor contour looks correct but the side-wall contours (`zdir='x'` and `zdir='y'`) appear mirrored/inverted compared to expected. What is the most likely cause?

- [ ] A) The `levels` argument produces different contour orderings for `zdir='x'` than for `zdir='z'`
- [ ] B) The meshgrid was created with `indexing='ij'` which transposes the x and y axes relative to the expected orientation, causing the x-wall and y-wall projections to appear with swapped axes
- [ ] C) The colormap `"viridis"` applies a reversed mapping for sidewall projections automatically
- [ ] D) The `offset` value for the x-wall should be `X_max` instead of `X_min` to project onto the visible wall at the current camera angle

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The meshgrid was created with `indexing='ij'` which transposes the x and y axes relative to the expected orientation, causing the x-wall and y-wall projections to appear with swapped axes

**Explanation:**
- **A is incorrect:** The `levels` argument specifies Z-value thresholds for contours and applies consistently across all `zdir` values. There is no differential contour ordering based on `zdir`.
- **B is correct:** When `np.meshgrid(x, y, indexing='ij')` is used, `X[i, j]` varies along axis 0 (i) and `Y[i, j]` varies along axis 1 (j). With the default `'xy'` indexing, `X[i, j]` varies along axis 1 (j) and `Y[i, j]` along axis 0 (i). Matplotlib's `contour` function internally assumes `'xy'` (Cartesian) indexing when computing and projecting level curves. Using `'ij'`-indexed grids can cause the projected contour axes to be transposed, making the side-wall projections appear mirrored or rotated relative to the surface.
- **C is incorrect:** Matplotlib does not apply different colormap orientations for different `zdir` values. The same colormap is applied uniformly.
- **D is incorrect:** Whether `X_min` or `X_max` is used depends on which wall the analyst wants to project onto. The "mirroring" artifact is not caused by incorrect offset values — it is caused by axis ordering issues in the input data.

</details>

---

### Question 76

A data scientist generates contour projections on all three walls and the floor. She uses:

```python
levels = np.linspace(Z.min(), Z.max(), 8)

ax.contour(X, Y, Z, zdir='z', offset=Z.min(), levels=levels, cmap='RdYlBu')
ax.contour(X, Y, Z, zdir='x', offset=X.min(), levels=levels, cmap='RdYlBu')
ax.contour(X, Y, Z, zdir='y', offset=Y.max(), levels=levels, cmap='RdYlBu')
```

Why is it important to pass the **same** `levels` array to all three contour calls?

- [ ] A) Using the same levels ensures all three projections use the same colour scale, so corresponding contour lines in all three views represent the same Z-value and can be visually traced across views
- [ ] B) Using the same levels reduces total render time by 3× because the levels are computed once and reused
- [ ] C) Matplotlib raises an error if different `levels` are used across multiple `ax.contour` calls on the same axes
- [ ] D) The same levels argument forces Matplotlib to use a single shared colorbar that covers all three projections

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Using the same levels ensures all three projections use the same colour scale, so corresponding contour lines in all three views represent the same Z-value and can be visually traced across views

**Explanation:**
- **A is correct:** When the same `levels` array is used for all three contour projections, a contour line at level L has the same colour in the floor projection, the left-wall projection, and the back-wall projection. This visual consistency allows viewers to trace the path of a single Z-value across all three "shadow" views simultaneously, making the 3D topology much easier to interpret. If different levels were used, the same colour might represent different Z-values in different projections, destroying this interpretability.
- **B is incorrect:** Matplotlib computes the level curves independently for each `ax.contour` call. There is no caching or reuse of level computations across separate calls. The levels array affects only where contour lines are drawn, not computation speed.
- **C is incorrect:** Matplotlib places no restriction on using different `levels` values across multiple contour calls on the same axes. Different calls are independent.
- **D is incorrect:** A single colorbar covers one contour collection by default. Creating a unified colorbar that covers all three `ax.contour` calls requires explicit setup (creating a ScalarMappable with the shared norm and colormap). The `levels` argument alone does not create a shared colorbar.

</details>

---

### Question 77

A data analyst adds a floor contour to a loss surface and observes that the contour lines appear cluttered and too dense near the gradient-descent minimum where Z changes rapidly. Which parameter adjustment reduces the density of contour lines?

- [ ] A) Reduce `alpha` to 0.3 — lower opacity spreads the visual density
- [ ] B) Increase `linewidths` from the default to 2.0 — thicker lines take more space, automatically reducing count
- [ ] C) Reduce the number of entries in the `levels` array or use `levels=5` (a small integer) to request fewer contour lines
- [ ] D) Set `zdir='y'` instead of `zdir='z'` — side-wall projections inherently render fewer contour lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Reduce the number of entries in the `levels` array or use `levels=5` (a small integer) to request fewer contour lines

**Explanation:**
- **A is incorrect:** Reducing `alpha` makes individual contour lines more transparent but does not reduce their count. Clutter from too many lines is not resolved by opacity — there are still the same number of overlapping lines.
- **B is incorrect:** `linewidths` controls the stroke thickness of each contour line. Increasing thickness makes each line more prominent but does not reduce the number of lines drawn. If anything, thicker lines make density worse visually.
- **C is correct:** The `levels` parameter directly controls how many and which Z-values generate contour lines. Passing an integer like `levels=5` asks Matplotlib to choose 5 evenly spaced levels, reducing line count from potentially dozens to 5. Alternatively, providing a manually constructed array like `np.linspace(Z.min(), Z.max(), 6)` gives explicit control. Fewer levels = fewer lines = less visual clutter.
- **D is incorrect:** Changing `zdir` changes which wall the contours project onto; it does not affect the number of contour levels drawn. The same density issue would appear on the side wall with the same `levels` specification.

</details>

---

### Question 78

A researcher examines why her side-wall contours look correct from one camera angle but appear "behind" the surface (invisible) from another angle. What is the underlying mechanism?

- [ ] A) Side-wall contours use a different rendering pipeline that only activates for specific camera elevations
- [ ] B) The contour artists on the side walls are `Line3DCollection` objects subject to the same painter's algorithm depth sort as the surface polygons; from some camera angles, surface polygons are sorted in front of the wall contours, hiding them
- [ ] C) `ax.contour` with `zdir='x'` only works for positive camera azimuth angles (0° to 180°)
- [ ] D) The side-wall contour visibility is controlled by `ax.set_pane_color` — if the pane is opaque, contours cannot be seen through it

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The contour artists on the side walls are `Line3DCollection` objects subject to the same painter's algorithm depth sort as the surface polygons; from some camera angles, surface polygons are sorted in front of the wall contours, hiding them

**Explanation:**
- **A is incorrect:** There is no camera-elevation-specific rendering pipeline for side-wall contours. The visibility is affected by depth sorting, not by elevation-specific rendering modes.
- **B is correct:** Wall-projected contour lines are `Line3DCollection` or `Path3DCollection` objects added to the 3D axes at fixed positions. Like all 3D artists in Matplotlib, they are sorted and rendered using the painter's algorithm based on depth. From camera angles where the surface polygons are geometrically closer to the viewer than the wall position, the surface polygons are rendered on top of the wall contours, hiding them. This is the same depth-sorting limitation that affects all overlapping 3D elements in Matplotlib.
- **C is incorrect:** There is no documented restriction of `zdir='x'` to specific azimuth angle ranges. The contours are computed and rendered the same way regardless of camera orientation; visibility is determined by depth sorting, not by azimuth filtering.
- **D is incorrect:** `ax.set_pane_color` controls the background colour of the 3D bounding box pane walls (the grey planes visible behind the plot). Even with an opaque pane colour, wall contours drawn at the pane's exact coordinate position are rendered as overlaid line artists. The pane colour affects the background behind the contours, not their visibility.

</details>

---

### Question 79

A data scientist wants to add a **filled** contour projection (filled contour regions instead of just contour lines) to the floor of a 3D surface plot. Which function achieves this?

- [ ] A) `ax.contour(X, Y, Z, zdir='z', offset=Z_min, filled=True)`
- [ ] B) `ax.contourf(X, Y, Z, zdir='z', offset=Z_min, cmap='viridis')`
- [ ] C) `ax.contour_filled(X, Y, Z, zdir='z', offset=Z_min)`
- [ ] D) `ax.plot_surface(X, Y, np.full_like(Z, Z_min), facecolors=...)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.contourf(X, Y, Z, zdir='z', offset=Z_min, cmap='viridis')`

**Explanation:**
- **A is incorrect:** `ax.contour` draws contour **lines** only. It does not have a `filled=True` parameter. Contour lines and filled contours are separate functions in Matplotlib.
- **B is correct:** `ax.contourf` is the filled contour variant of `ax.contour`. In 3D, it supports the same `zdir` and `offset` parameters as `ax.contour`, projecting filled contour regions (bands of colour between consecutive level thresholds) onto the specified wall plane. The result is a colour-gradient "shadow" on the floor showing the density of the loss landscape.
- **C is incorrect:** `ax.contour_filled` is not a valid Matplotlib method. The correct function name is `contourf` (with an 'f' suffix, not a separate word).
- **D is incorrect:** `ax.plot_surface` with a constant Z would produce a flat coloured surface at z=Z_min, which is not the same as filled contours. Creating custom filled contours this way requires complex per-polygon colour assignment that doesn't reproduce the contourf mapping.

</details>

---

### Question 80

A researcher uses `zdir='y', offset=Y_max` to project loss surface contours onto the back wall and then adds the surface with `alpha=0.4`:

```python
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4, edgecolor='none')
ax.contour(X, Y, Z, zdir='y', offset=Y.max(), cmap='viridis',
           levels=8, linewidths=1.5)
```

A colleague suggests adding `zorder=5` to the `ax.contour` call to ensure contours are always drawn in front of the surface. Will this work?

- [ ] A) Yes — `zorder=5` explicitly sets the depth ordering of the contour collection above all surface polygons
- [ ] B) No — `zorder` in Matplotlib 3D has limited effectiveness; it controls 2D drawing order within each z-depth group but does not override the 3D painter's algorithm depth sort between different artists
- [ ] C) Yes — but only when the contours and surface have non-overlapping depth extents
- [ ] D) No — `ax.contour` does not accept a `zorder` argument in 3D mode

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `zorder` in Matplotlib 3D has limited effectiveness; it controls 2D drawing order within each z-depth group but does not override the 3D painter's algorithm depth sort between different artists

**Explanation:**
- **A is incorrect:** While `zorder` in 2D Matplotlib strongly controls rendering order, in 3D Matplotlib it is substantially weaker. The painter's algorithm that sorts polygons by depth can override `zorder` values when depth-sorting places surface polygons "in front of" the contours in 3D space.
- **B is correct:** In Matplotlib 3D, the final rendering order is primarily determined by the painter's algorithm's depth sort. The `zorder` parameter can influence the order in which tied elements are rendered (within the same depth group) but does not reliably override the 3D spatial depth ordering. Adding `zorder=5` to the contour call may help in some cases but will not reliably guarantee that wall contours always appear in front of the surface from all camera angles. This is a known limitation of Matplotlib's 3D rendering system.
- **C is incorrect:** The limitation of `zorder` in 3D applies regardless of whether depth extents overlap. Even non-overlapping depth extents can have sorting artifacts due to the centroid-based depth approximation used by the painter's algorithm.
- **D is incorrect:** `ax.contour` does accept a `zorder` argument in 3D mode. The argument is not rejected or ignored — it is simply not reliably effective at overriding 3D depth sorting in the way it works in 2D.

</details>
