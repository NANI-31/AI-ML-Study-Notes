# 3D Matplotlib — Test 03: 3D Line Plots & Trajectories

---

### Question 21

A physicist generates a parametric helix using:

```python
t = np.linspace(0, 6 * np.pi, 300)
x = 2 * np.cos(t)
y = 2 * np.sin(t)
z = t / (2 * np.pi)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z, color="royalblue", linewidth=2)
```

After running this code, what are the approximate **x-range**, **y-range**, and **z-range** of the helix?

- [ ] A) x ∈ [-2, 2], y ∈ [-2, 2], z ∈ [0, 3]
- [ ] B) x ∈ [0, 2], y ∈ [0, 2], z ∈ [0, 6π]
- [ ] C) x ∈ [-1, 1], y ∈ [-1, 1], z ∈ [0, 3π]
- [ ] D) x ∈ [-2, 2], y ∈ [-2, 2], z ∈ [0, 6π]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) x ∈ [-2, 2], y ∈ [-2, 2], z ∈ [0, 3]

**Explanation:**
- **A is correct:** `x = 2*cos(t)` oscillates between -2 and +2, so x ∈ [-2, 2]. `y = 2*sin(t)` likewise oscillates in [-2, 2]. For the z-range: `t` runs from 0 to `6π`. `z = t / (2π)` divides by 2π, so at `t = 6π`: `z = 6π / (2π) = 3`. Thus z ∈ [0, 3]. The helix completes exactly 3 full turns (each turn spans 2π in t, which maps to z increment of 1).
- **B is incorrect:** `cos(t)` and `sin(t)` oscillate through negative values as well, so the x and y ranges include negative values down to -2. The z range is [0, 3], not [0, 6π] — the division by 2π compresses the z extent.
- **C is incorrect:** The radius of the helix is 2 (the coefficient), not 1. `2*cos(t)` and `2*sin(t)` give ranges [-2, 2], not [-1, 1].
- **D is incorrect:** The z-range is [0, 3], not [0, 6π]. The formula `z = t / (2π)` explicitly divides `t` by 2π, compressing the range from [0, 6π] down to [0, 3].

</details>

---

### Question 22

A machine learning engineer traces the gradient descent path on a loss landscape. She stores 50 optimization steps as:

```python
# w1, w2 are parameter arrays of shape (50,)
# loss is the corresponding loss at each step of shape (50,)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(w1, w2, loss, color="red", linewidth=2, marker="o", markersize=4)
ax.plot_surface(W1_grid, W2_grid, Loss_surface,
                cmap="viridis", alpha=0.3, edgecolor="none")
```

Why is the optimization path rendered as a 3D line rather than as a second scatter plot?

- [ ] A) `ax.plot` is the only function that supports the `marker` argument in 3D; `ax.scatter` ignores markers in 3D mode
- [ ] B) `ax.plot` connects consecutive steps with line segments, showing the directional sequence of the descent path, while `ax.scatter` would show only unconnected points with no implied ordering
- [ ] C) `ax.scatter` cannot overlay on top of `ax.plot_surface`; only `ax.plot` supports layering with surfaces
- [ ] D) `ax.plot` is faster for small datasets; `ax.scatter` is always used for datasets larger than 1000 points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.plot` connects consecutive steps with line segments, showing the directional sequence of the descent path, while `ax.scatter` would show only unconnected points with no implied ordering

**Explanation:**
- **A is incorrect:** Both `ax.plot` and `ax.scatter` support marker arguments in 3D mode. `ax.scatter` uses `marker=` and `ax.plot` uses `marker=` as well. Neither function restricts marker usage in 3D.
- **B is correct:** The fundamental difference between `ax.plot` and `ax.scatter` for trajectory visualization is that `ax.plot` draws line segments connecting consecutive data points in order, revealing the sequential path from start to finish. For gradient descent visualization, this connectivity is essential: the path shows how the optimizer moves from one parameter position to the next. `ax.scatter` renders isolated points with no connecting lines, losing the sequential flow information.
- **C is incorrect:** Both `ax.plot` and `ax.scatter` can be overlaid on top of `ax.plot_surface`. Matplotlib 3D supports multiple drawing calls on the same axes regardless of type.
- **D is incorrect:** The choice between `ax.plot` and `ax.scatter` is made based on visual semantics (connected vs. unconnected), not on dataset size thresholds. Both are efficient for 50-point datasets.

</details>

---

### Question 23

The following code generates a Lissajous curve in 3D space:

```python
t = np.linspace(0, 2 * np.pi, 500)
a, b, c_freq = 3, 2, 1
x = np.sin(a * t)
y = np.sin(b * t + np.pi / 4)
z = np.cos(c_freq * t)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(x, y, z, linewidth=1.5)
```

An analyst notices the curve appears to intersect itself in the 3D view. What is the correct interpretation?

- [ ] A) This is a rendering bug; parametric curves in 3D cannot intersect themselves
- [ ] B) The self-intersection is real in 3D space because the different frequency ratios (3:2:1) cause the curve to revisit the same (x, y, z) coordinate at multiple t values
- [ ] C) The self-intersection is a 2D projection artifact; the curve may appear to cross when viewed from one camera angle but does not actually share 3D coordinates
- [ ] D) Self-intersections only occur when `a == b == c_freq`; with different frequencies the curve is always a simple closed loop

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The self-intersection is a 2D projection artifact; the curve may appear to cross when viewed from one camera angle but does not actually share 3D coordinates

**Explanation:**
- **A is incorrect:** Parametric curves can legitimately revisit the same 3D coordinates (true self-intersection), and rendering does not introduce spurious intersections.
- **B is incorrect:** While Lissajous figures with certain frequency ratios do produce true self-intersections in 2D, a 3D Lissajous curve with three independent frequency components is much less likely to revisit exactly the same (x, y, z) triple. The analyst is seeing an apparent crossing that likely arises from projection.
- **C is correct:** When a 3D curve is projected onto the 2D screen by the camera, two distinct points in 3D space can project to the same 2D pixel position, creating an apparent crossing. Rotating the camera (changing `elev` or `azim` via `view_init`) will reveal that the crossing disappears from other viewpoints, confirming it is a projection artifact rather than a true 3D self-intersection.
- **D is incorrect:** Self-intersections in parametric curves depend on whether the frequency ratios create rational relationships that allow the curve to return to the same coordinates, not simply on equality of all frequencies. Equal frequencies would trace an ellipse, not necessarily a self-intersecting curve.

</details>

---

### Question 24

A researcher animates a particle trajectory by appending new coordinates and re-plotting:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
line, = ax.plot([], [], [])

def update(frame):
    line.set_data(x_traj[:frame], y_traj[:frame])
    line.set_3d_properties(z_traj[:frame])
    return line,
```

Why is `line.set_3d_properties(z_traj[:frame])` called separately instead of including `z` in `set_data`?

- [ ] A) `set_data` in 3D mode accepts three arguments but requires `set_3d_properties` to finalize the rendering pipeline
- [ ] B) `set_data` is a 2D method inherited from `Line2D` that handles only `(x, y)` data; the z-component of a 3D line must be updated separately via `set_3d_properties`, which is an `Axes3D`-specific extension
- [ ] C) `set_3d_properties` is required to set the line colour and width after each frame update
- [ ] D) In Matplotlib 3D, `set_data` updates the line's screen coordinates while `set_3d_properties` updates the world coordinates used for depth sorting

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `set_data` is a 2D method inherited from `Line2D` that handles only `(x, y)` data; the z-component of a 3D line must be updated separately via `set_3d_properties`, which is an `Axes3D`-specific extension

**Explanation:**
- **A is incorrect:** `set_data` in the standard 2D `Line2D` class accepts only two arguments (xdata, ydata). It does not support a third argument. The "3D" information is handled by the mplot3d extension's `set_3d_properties` method.
- **B is correct:** When `ax.plot` creates a 3D line, it returns a `Line3D` object (a subclass of `Line2D`) that inherits `set_data(x, y)` from `Line2D` for the x and y arrays, and adds the mplot3d-specific `set_3d_properties(z)` method to update the z-component separately. This design reflects the layered architecture of mplot3d on top of 2D Matplotlib. Both must be called to fully update the 3D line's data.
- **C is incorrect:** `set_3d_properties` is used only for z-coordinate data updates, not for visual styling properties like colour or line width. Those would be set via `line.set_color()` or `line.set_linewidth()`.
- **D is incorrect:** `set_data` does not operate on "screen coordinates" — all coordinate management in Matplotlib 3D happens in world space. `set_data` and `set_3d_properties` together update the 3D world coordinates; the projection to screen space is handled internally.

</details>

---

### Question 25

A data scientist visualizes multiple training runs by plotting one 3D line per run:

```python
colors = plt.cm.coolwarm(np.linspace(0, 1, n_runs))
for i, (w1_path, w2_path, loss_path) in enumerate(training_runs):
    ax.plot(w1_path, w2_path, loss_path,
            color=colors[i], linewidth=1.5, alpha=0.7)
```

`np.linspace(0, 1, n_runs)` is used as input to `plt.cm.coolwarm(...)`. What does this expression produce?

- [ ] A) An array of `n_runs` evenly spaced line widths from 0 to 1
- [ ] B) An array of `n_runs` RGBA colour tuples sampled at equal intervals along the `coolwarm` colormap
- [ ] C) An array of `n_runs` opacity values ranging from fully transparent to fully opaque
- [ ] D) An array of `n_runs` angle values used to rotate the 3D camera incrementally

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An array of `n_runs` RGBA colour tuples sampled at equal intervals along the `coolwarm` colormap

**Explanation:**
- **A is incorrect:** `plt.cm.coolwarm(...)` is a colormap callable. It takes scalar values in [0, 1] and returns RGBA colour tuples, not line widths. Line width is a float controlling stroke thickness and is set separately via `linewidth=`.
- **B is correct:** `plt.cm.coolwarm` is a `LinearSegmentedColormap` object. When called with an array of floats in [0, 1], it evaluates the colormap at each position and returns an array of shape `(n_runs, 4)` where each row is an RGBA tuple. `np.linspace(0, 1, n_runs)` creates `n_runs` evenly spaced values from 0 to 1, so the returned colours are evenly sampled from the cool-blue end to the warm-red end of the `coolwarm` colormap.
- **C is incorrect:** Opacity/alpha in Matplotlib is a scalar in [0, 1] passed via `alpha=`. The fourth component of each RGBA tuple from the colormap is the alpha channel, but the entire output of `plt.cm.coolwarm(...)` is colour data, not a list of standalone alpha values.
- **D is incorrect:** Camera angles are controlled via `ax.view_init(elev=..., azim=...)`. The colormap callable has nothing to do with camera rotation.

</details>

---

### Question 26

A researcher adds a dashed reference line from the start to the end of a gradient descent path:

```python
start = (w1_path[0], w2_path[0], loss_path[0])
end = (w1_path[-1], w2_path[-1], loss_path[-1])

ax.plot([start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        linestyle="--", color="gray", linewidth=1)
```

This code draws a straight line segment from `start` to `end`. How many line segments does this call draw?

- [ ] A) 0 — `ax.plot` requires at least 3 points to draw visible lines
- [ ] B) 1 — two (x, y, z) coordinate pairs define exactly one line segment
- [ ] C) 3 — one segment per axis (x-direction, y-direction, z-direction)
- [ ] D) 2 — `ax.plot` always splits a two-point input into two half-segments for depth-sorting purposes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1 — two (x, y, z) coordinate pairs define exactly one line segment

**Explanation:**
- **A is incorrect:** `ax.plot` in 3D requires at least 2 points to draw a segment. Two points are sufficient to define a single line segment. The minimum is 2, not 3.
- **B is correct:** The arrays passed to `ax.plot` contain exactly two elements each: `[start_coord, end_coord]` for x, y, and z. Two consecutive (x, y, z) points define exactly one line segment. The `linestyle="--"` makes this segment appear as a dashed line, and `color="gray"` colours it grey.
- **C is incorrect:** The number of segments is determined by `len(x) - 1`, where `x` is the x-coordinate array. With 2 points, there is `2 - 1 = 1` segment. Matplotlib does not separately decompose the segment into per-axis components.
- **D is incorrect:** Matplotlib 3D does perform depth-sorting of line segments for correct occlusion rendering, but this sorting does not split a single segment into two. A 2-point input produces 1 segment regardless of depth-sorting operations.

</details>

---

### Question 27

A scientist generates a 3D spiral trajectory and wants to color the line by time `t`, using a colormap. Since `ax.plot` produces a single `Line3D` object with one colour, she instead uses:

```python
from matplotlib.collections import LineCollection

# Create segments from consecutive points
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
```

What shape does `segments` have if `t` has 200 elements?

- [ ] A) `(200, 2, 3)` — 200 segments, each with 2 endpoints of 3 coordinates
- [ ] B) `(199, 2, 3)` — 199 segments, each with 2 endpoints of 3 coordinates
- [ ] C) `(200, 3, 2)` — 200 points, 3 axes, 2 endpoints per segment
- [ ] D) `(199, 3)` — 199 segments described by 3D direction vectors

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `(199, 2, 3)` — 199 segments, each with 2 endpoints of 3 coordinates

**Explanation:**
- **A is incorrect:** With N points there are N-1 segments, not N. With 200 points there are 199 consecutive segments connecting adjacent pairs.
- **B is correct:** `points` has shape `(200, 1, 3)` — 200 rows of single-point "groups" with 3 coordinates each. `points[:-1]` has shape `(199, 1, 3)` and `points[1:]` also has shape `(199, 1, 3)`. `np.concatenate([points[:-1], points[1:]], axis=1)` concatenates along axis=1, giving shape `(199, 2, 3)`. Each of the 199 segments is described by 2 endpoints (start and end), each with 3 coordinates (x, y, z).
- **C is incorrect:** The shape `(200, 3, 2)` does not match the construction. The concatenation is along axis=1 (the segment endpoint axis), not axis=2.
- **D is incorrect:** `(199, 3)` would be direction vectors (differences between consecutive points), not the segment representation. The segment representation stores both endpoint coordinates, not just the difference.

</details>

---

### Question 28

A machine learning engineer superimposes a loss trajectory over a loss surface and notices that part of the trajectory **appears below the surface** (clipped/hidden), even though the loss values at each step are on the surface. What is the likely cause?

- [ ] A) `ax.plot_surface` renders with `zorder=10` by default, which always draws surfaces on top of all other artists
- [ ] B) Matplotlib 3D uses a painter's algorithm for depth sorting, which can incorrectly order the surface and trajectory artists based on their approximate centroid depths rather than per-pixel depth testing
- [ ] C) The `alpha=0.3` transparency on the surface causes the trajectory to become invisible wherever it overlaps
- [ ] D) `ax.plot` draws lines below the z=0 plane by default; adding `zdir="z"` fixes the ordering

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Matplotlib 3D uses a painter's algorithm for depth sorting, which can incorrectly order the surface and trajectory artists based on their approximate centroid depths rather than per-pixel depth testing

**Explanation:**
- **A is incorrect:** Matplotlib 3D does not assign fixed `zorder` values of 10 to surfaces. The `zorder` system in 3D Matplotlib is weaker than in 2D because the depth sorting is done geometrically, not via a simple `zorder` override.
- **B is correct:** This is a well-known limitation of Matplotlib's 3D rendering. Matplotlib uses the painter's algorithm: it sorts all 3D artists (polygons, lines, collections) by their approximate depth (typically centroid or average z in camera space) and renders them back-to-front. For a surface composed of many small polygons, this centroid-based sorting can incorrectly place some surface polygons "in front of" lines that actually lie on or above the surface, causing the trajectory to appear partially hidden. The fix is often to slightly elevate the trajectory (`z_traj + small_offset`) or to reduce surface opacity.
- **C is incorrect:** `alpha=0.3` makes the surface transparent, which would actually make the trajectory more visible through the surface, not less.
- **D is incorrect:** `ax.plot` does not draw lines below z=0 by default. The `zdir` parameter in `ax.plot` is used for 2D data projected onto a 3D plane, not for controlling depth ordering of 3D lines.

</details>

---

### Question 29

A researcher plots the trajectory of a robot arm using:

```python
ax.plot(x, y, z, "b-o", markersize=3)
ax.plot([x[0]], [y[0]], [z[0]], "go", markersize=10, label="Start")
ax.plot([x[-1]], [y[-1]], [z[-1]], "r*", markersize=12, label="End")
```

Why are the start and end points plotted with separate `ax.plot` calls instead of being marked within the trajectory `ax.plot` call?

- [ ] A) The trajectory call uses a format string `"b-o"` which forbids the use of per-point size modifiers
- [ ] B) Different marker types (`"o"` and `"*"`) and different sizes (10 and 12) for specific points cannot be specified within a single `ax.plot` call — separate calls allow independent styling for each special point
- [ ] C) Matplotlib 3D requires start and end points to be added as separate artists to include them in the legend
- [ ] D) The `label` argument is not supported by `ax.plot` in 3D mode; it only works in separate calls using `plt.legend`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Different marker types (`"o"` and `"*"`) and different sizes (10 and 12) for specific points cannot be specified within a single `ax.plot` call — separate calls allow independent styling for each special point

**Explanation:**
- **A is incorrect:** Format strings like `"b-o"` set a default marker for all points in that call, but this does not "forbid" size modifiers. `markersize` can be specified alongside a format string. The limitation is that a single `ax.plot` call applies one uniform style to all its points.
- **B is correct:** A single `ax.plot` call applies one marker type and one marker size uniformly to every point in the series. To display the start point with a large green circle (`"go"`, `markersize=10`) and the end point with a large red star (`"r*"`, `markersize=12`) while keeping the trajectory dots small (`markersize=3`), three separate calls are necessary. Each call can independently specify its own colour, marker type, and size.
- **C is incorrect:** `ax.plot` in 3D fully supports the `label` argument regardless of whether it's a separate call or not. The legend system works the same as in 2D — each artist with a `label` appears in the legend when `ax.legend()` is called.
- **D is incorrect:** The `label` argument is fully supported by `ax.plot` in 3D. It works identically to 2D `plot` calls, and the legend displays correctly.

</details>

---

### Question 30

A data analyst wants to add vertical "drop lines" from each point of a 3D trajectory down to the z=0 plane, creating a "lollipop" visualization effect. Which code pattern achieves this?

- [ ] A) `ax.vlines(x, ymin=0, ymax=z)` — draws vertical lines from z=0 up to each point
- [ ] B) For each point `(xi, yi, zi)`: `ax.plot([xi, xi], [yi, yi], [0, zi], "k-", alpha=0.3)`
- [ ] C) `ax.bar3d(x, y, np.zeros_like(z), 0, 0, z, color="lightgray", alpha=0.3)`
- [ ] D) `ax.stem(x, y, z, linefmt="k-", markerfmt=" ")` — generates automatic stem lines to z=0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) For each point `(xi, yi, zi)`: `ax.plot([xi, xi], [yi, yi], [0, zi], "k-", alpha=0.3)`

**Explanation:**
- **A is incorrect:** `ax.vlines` is a 2D Axes method for drawing vertical lines at fixed x-positions. The `Axes3D` object does not have a `vlines` method with a `ymax` parameter that maps to the z-axis. Calling `ax.vlines` on a 3D axes would raise an `AttributeError` or produce unexpected results.
- **B is correct:** Drawing a drop line from `(xi, yi, zi)` to `(xi, yi, 0)` is achieved by passing two-element arrays to `ax.plot`: the x-array `[xi, xi]` (same x at top and bottom), y-array `[yi, yi]` (same y), and z-array `[0, zi]` (from z=0 up to the point's z-value). A loop over all trajectory points creates one vertical segment per point. This is the standard pattern for lollipop/stem visualizations in 3D Matplotlib.
- **C is incorrect:** `ax.bar3d` with `dx=0, dy=0` would create zero-volume bars (degenerate columns) which typically render as nothing or cause rendering artifacts. `bar3d` is designed for bars with nonzero width and depth, not for thin vertical lines.
- **D is incorrect:** `ax.stem` is a 2D Axes method. While newer Matplotlib versions have added some 3D support, `ax.stem` on an `Axes3D` does not work with standard `(x, y, z)` trajectory data in the same way. The standard approach remains option B.

</details>
