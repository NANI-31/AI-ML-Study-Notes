# 3D Matplotlib — Test 09: Camera Angles & Perspective

---

### Question 81

A data scientist calls `ax.view_init(elev=0, azim=0)` on a 3D loss surface. Describe the resulting view.

- [ ] A) Top-down birds-eye view looking straight down the z-axis onto the xy-plane
- [ ] B) Standard default isometric-like view at 30° elevation and 45° azimuth
- [ ] C) A side-elevation view looking along the positive x-axis into the yz-plane, with no z-axis foreshortening visible from above
- [ ] D) A front-facing view looking along the positive y-axis into the xz-plane

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) A side-elevation view looking along the positive x-axis into the yz-plane, with no z-axis foreshortening visible from above

**Explanation:**
- **A is incorrect:** `elev=90, azim=-90` produces the top-down birds-eye view. `elev=0` places the camera at the equatorial level of the z-axis, not above it.
- **B is incorrect:** The default view in Matplotlib 3D is approximately `elev=30, azim=-60`. `elev=0, azim=0` is explicitly different from the default.
- **C is correct:** `elev=0` places the camera exactly at the height of the xy-plane (0° above the equator), removing any perspective from above. `azim=0` positions the camera along the positive x-axis direction. From this viewpoint, the viewer looks along the x-axis toward the origin, seeing the yz-plane face-on. The x-axis appears as a single point (depth), and the y and z axes form the visible 2D projection. This is a "side view" equivalent.
- **D is incorrect:** Looking along the positive y-axis into the xz-plane would require `elev=0, azim=90` (or `-90` depending on convention). `azim=0` points along the x-axis, not the y-axis.

</details>

---

### Question 82

A researcher wants a top-down birds-eye view of a 3D scatter cluster. She uses:

```python
ax.view_init(elev=90, azim=-90)
```

What does the resulting 2D-like projection show, and why is `azim=-90` used?

- [ ] A) The surface is viewed from below; `azim=-90` rotates the view under the z-axis
- [ ] B) The view is directly from above (z-axis pointing toward viewer); `azim=-90` rotates the azimuth so that the positive x-axis points to the right in the projected view, matching the conventional 2D orientation
- [ ] C) `elev=90` creates a top view but `azim=-90` cancels it, returning to the default view
- [ ] D) `azim=-90` is required to prevent gimbal lock when `elev=90`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The view is directly from above (z-axis pointing toward viewer); `azim=-90` rotates the azimuth so that the positive x-axis points to the right in the projected view, matching the conventional 2D orientation

**Explanation:**
- **A is incorrect:** `elev=90` places the camera above the data, looking straight down — not below. "Below" would require `elev=-90`.
- **B is correct:** `elev=90` points the camera directly downward along the z-axis, producing a view equivalent to a 2D top-down plot. With this elevation, the rendered image looks like a standard 2D scatter or contour plot. However, the azimuth determines how the x and y axes are oriented in the projected image. `azim=-90` rotates the azimuth to align the positive x-axis with the rightward screen direction and positive y-axis with the upward screen direction — matching the conventional Cartesian xy-plane orientation that analysts expect in a 2D-like top view.
- **C is incorrect:** `azim=-90` does not cancel `elev=90`. Azimuth and elevation are independent parameters. Azimuth rotates the camera around the z-axis; elevation tilts it up or down. They do not cancel each other.
- **D is incorrect:** Gimbal lock is a mechanical/quaternion concept related to degree-of-freedom ambiguity at extreme angles. While `elev=90` is a degenerate angle where azimuth rotation around the z-axis aligns with another rotation, Matplotlib simply uses the azimuth value to orient the 2D projection axes. `azim=-90` is a practical choice for axis alignment, not a technical requirement to prevent locking.

</details>

---

### Question 83

A developer animates a 3D loss surface by rotating the camera in a loop:

```python
for azim in range(0, 360, 5):
    ax.view_init(elev=25, azim=azim)
    plt.pause(0.05)
```

What direction does the camera rotate as `azim` increases from 0 to 360?

- [ ] A) Clockwise when viewed from above (decreasing azimuth is counter-clockwise)
- [ ] B) Counter-clockwise when viewed from above — increasing azimuth rotates the camera counter-clockwise around the z-axis
- [ ] C) The camera rotates clockwise in screen space regardless of elevation
- [ ] D) The rotation direction depends on whether `elev` is positive or negative

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Counter-clockwise when viewed from above — increasing azimuth rotates the camera counter-clockwise around the z-axis

**Explanation:**
- **A is incorrect:** Matplotlib's azimuth convention follows the standard mathematical polar coordinate convention where increasing angle is counter-clockwise when viewed from above (positive z-axis looking down). Clockwise would be decreasing azimuth.
- **B is correct:** In Matplotlib 3D, `azim` is the angle in degrees measured counter-clockwise from the positive x-axis around the z-axis (right-hand rule convention). Increasing `azim` from 0° to 360° sweeps the camera counter-clockwise around the z-axis when viewed from above. This creates a counter-clockwise rotation of the 3D scene in the rendered images.
- **C is incorrect:** The rotation direction in screen space depends on the elevation angle (`elev`). At `elev=0`, increasing azimuth rotates horizontally. At positive elevation, the relationship between azimuth change and screen-space rotation is more complex due to the 3D perspective projection.
- **D is incorrect:** While elevation affects how azimuth changes appear visually (the projected rotation magnitude changes), the underlying direction of increasing azimuth (counter-clockwise around z when viewed from above) is consistent regardless of elevation sign.

</details>

---

### Question 84

A machine learning engineer wants to show a gradient descent path on a loss surface from the perspective of **looking along the y-axis** (front view, seeing the xz-plane). Which `view_init` call produces this view?

- [ ] A) `ax.view_init(elev=0, azim=90)`
- [ ] B) `ax.view_init(elev=30, azim=0)`
- [ ] C) `ax.view_init(elev=0, azim=0)`
- [ ] D) `ax.view_init(elev=90, azim=0)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.view_init(elev=0, azim=90)`

**Explanation:**
- **A is correct:** `elev=0` places the camera in the horizontal plane (no tilt above or below the equator). `azim=90` positions the camera 90° counter-clockwise from the x-axis — this places the camera along the positive y-axis direction. Looking from the y-axis toward the origin, the viewer sees the xz-plane face-on. The x-axis runs horizontally and the z-axis runs vertically in the projected view. This is the "front view" looking along the y-axis.
- **B is incorrect:** `elev=30, azim=0` is closer to a standard isometric view looking from the x-direction at 30° elevation. It shows the yz-plane at an angle, not a pure xz-plane view from the y-axis.
- **C is incorrect:** `elev=0, azim=0` places the camera along the positive x-axis, giving a view into the yz-plane — not the xz-plane from the y-axis direction.
- **D is incorrect:** `elev=90, azim=0` is the top-down view (looking straight down the z-axis). The elevation of 90° puts the camera directly above, not along the y-axis.

</details>

---

### Question 85

A researcher creates four subplots of the same loss surface, each from a different camera angle:

```python
views = [(30, -60), (0, 0), (90, -90), (20, 120)]
for ax, (elev, azim) in zip(axes, views):
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.view_init(elev=elev, azim=azim)
```

Which of the four `(elev, azim)` pairs produces a view where the surface appears as a flat 2D contour-like image with no perspective depth?

- [ ] A) `(30, -60)` — the default Matplotlib view
- [ ] B) `(0, 0)` — looking horizontally from the x-axis direction
- [ ] C) `(90, -90)` — the top-down birds-eye view
- [ ] D) `(20, 120)` — a diagonal view with perspective

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `(90, -90)` — the top-down birds-eye view

**Explanation:**
- **A is incorrect:** `(30, -60)` is the standard default 3D perspective view showing all three axes at angles. It has visible depth and perspective.
- **B is incorrect:** `(0, 0)` is a horizontal side view showing the yz-plane. It still has two visible axes (y and z) with a visible x-depth, appearing as a 2D side profile. But the surface appears as a flat projection of all x-values onto a single vertical plane — not a contour-like top view.
- **C is correct:** `(90, -90)` points the camera straight down the z-axis. With elevation=90°, the camera looks directly downward, making the z-axis disappear into a single point. The surface appears as a flat 2D image in the xy-plane — exactly equivalent to a 2D top-view contour plot or heatmap. This is the perspective that removes all depth perception and presents the surface as a flat map.
- **D is incorrect:** `(20, 120)` is a tilted diagonal view that shows the surface from a non-standard angle with some depth and perspective visible. It is not a flat 2D projection.

</details>

---

### Question 86

A data scientist wants to display a 3D scatter with the camera elevated 45° above the xy-plane and rotated 135° around the z-axis counter-clockwise from the positive x-axis. Which `view_init` call achieves this?

- [ ] A) `ax.view_init(elev=135, azim=45)`
- [ ] B) `ax.view_init(elev=45, azim=45)`
- [ ] C) `ax.view_init(elev=45, azim=135)`
- [ ] D) `ax.view_init(azim=45, elev=135)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.view_init(elev=45, azim=135)`

**Explanation:**
- **A is incorrect:** `elev=135` exceeds the practical range of elevation angles. Elevation is measured from the xy-plane (0°) to directly above (90°). 135° would go past vertical and begin looking from below, which is not "45° above the xy-plane."
- **B is incorrect:** `elev=45, azim=45` correctly sets the elevation at 45° above the xy-plane, but `azim=45` is only 45° of azimuthal rotation, not 135° as required.
- **C is correct:** `elev=45` places the camera 45° above the xy-plane (halfway between horizontal and top-down). `azim=135` rotates the camera 135° counter-clockwise from the positive x-axis around the z-axis. In Matplotlib's azimuth convention, increasing azimuth is counter-clockwise. Both values match the specification exactly.
- **D is incorrect:** `ax.view_init(azim=45, elev=135)` has the parameter names reversed in the function call — `view_init(elev, azim)` is the parameter order, so this would set `elev=45` (from `azim=45`) and `azim=135` (from `elev=135`) due to keyword argument naming. Actually, D and C are equivalent if keyword arguments are used. But the issue is that the values assigned to the keywords in D are: `azim=45` and `elev=135` — elevation 135° is out of the valid range, making D incorrect for a "45° above xy-plane, 135° azimuth" requirement.

</details>

---

### Question 87

A data analyst uses `ax.view_init` and then saves the figure:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.view_init(elev=35, azim=-45)
plt.savefig("surface.png", dpi=150, bbox_inches="tight")
plt.show()
```

A colleague says: "You don't need `view_init` before `savefig` — the view only takes effect after `show()` is called." Is this claim correct?

- [ ] A) Yes — `view_init` is a rendering hint that only materializes when the display backend executes `show()`
- [ ] B) No — `view_init` modifies the axes camera state immediately; `savefig` reads the current camera state when rendering the file, so the specified view is captured in the saved image
- [ ] C) Yes — but only in non-interactive backends like `Agg`; interactive backends apply `view_init` immediately
- [ ] D) No — `savefig` always uses the default view `(elev=30, azim=-60)` regardless of `view_init`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `view_init` modifies the axes camera state immediately; `savefig` reads the current camera state when rendering the file, so the specified view is captured in the saved image

**Explanation:**
- **A is incorrect:** `view_init` is not a lazy rendering hint. It immediately modifies the `elev` and `azim` attributes of the `Axes3D` object. These attributes are read during any rendering operation — whether triggered by `savefig`, `show`, or the figure's `draw` method.
- **B is correct:** `ax.view_init(elev=35, azim=-45)` directly sets the camera elevation and azimuth properties on the axes object. When `plt.savefig(...)` is called, Matplotlib renders the figure using the current state of all artists, including the current camera view. The saved PNG file reflects `elev=35, azim=-45`. The colleague's claim is incorrect — `show()` is not required for `view_init` to take effect.
- **C is incorrect:** `view_init` applies immediately in all backends (interactive and non-interactive). The backend type does not affect when camera state changes take effect — it only affects when and how rendering is triggered.
- **D is incorrect:** `savefig` renders whatever the current axes state is at the time of the call. If `view_init` has been called before `savefig`, the custom view is used. The default view is only used when no `view_init` call has been made.

</details>

---

### Question 88

A researcher creates an animation that changes camera angle while keeping the same surface data. She uses:

```python
from matplotlib.animation import FuncAnimation

def animate(frame):
    ax.view_init(elev=30, azim=frame * 2)
    return []

ani = FuncAnimation(fig, animate, frames=180, interval=50)
```

What happens to the surface data across frames of this animation?

- [ ] A) The surface data is re-rendered from scratch each frame, including full surface recalculation
- [ ] B) The surface polygon geometry does not change; only the camera projection matrix is updated, which is much more efficient than re-drawing the surface polygons
- [ ] C) `ax.view_init` clears the axes and redraws all artists on each call
- [ ] D) The animation does not work because `view_init` cannot be called inside an animation function

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The surface polygon geometry does not change; only the camera projection matrix is updated, which is much more efficient than re-drawing the surface polygons

**Explanation:**
- **A is incorrect:** `view_init` does not trigger recalculation of surface polygon vertices. The `Poly3DCollection` data is fixed from the initial `plot_surface` call. Only the camera transformation (projection matrix) changes.
- **B is correct:** `ax.view_init(elev, azim)` updates internal camera parameters (`ax.elev` and `ax.azim`) and marks the axes as needing a redraw. On the next rendering call, the existing polygon data is re-projected onto the 2D screen using the new camera transformation. The surface geometry itself (vertex coordinates, colours) is not recomputed — only the 2D screen positions of the pre-computed 3D vertices are recalculated. This makes camera-rotation animations much faster than animations that actually change data.
- **C is incorrect:** `view_init` does not call `ax.cla()` or any axes-clearing method. It only updates the camera state variables. All existing artists remain on the axes.
- **D is incorrect:** `view_init` is fully compatible with animation functions. The pattern shown — calling `view_init` inside `animate` to create a camera-rotation animation — is a standard and common Matplotlib 3D animation technique.

</details>

---

### Question 89

A data scientist tests how `roll` affects the 3D view:

```python
ax.view_init(elev=30, azim=45, roll=30)
```

What does the `roll` parameter control, and in which Matplotlib version was it introduced?

- [ ] A) `roll` controls the depth-to-width aspect ratio of the 3D box; introduced in Matplotlib 2.0
- [ ] B) `roll` rotates the camera around the viewing axis (the axis pointing from the camera toward the scene), tilting the "up" direction; introduced in Matplotlib 3.5
- [ ] C) `roll` is an alias for `azim` and accepts the same range of values; introduced in Matplotlib 3.0
- [ ] D) `roll` controls the field-of-view angle (zoom level) of the 3D perspective projection; introduced in Matplotlib 3.4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `roll` rotates the camera around the viewing axis (the axis pointing from the camera toward the scene), tilting the "up" direction; introduced in Matplotlib 3.5

**Explanation:**
- **A is incorrect:** The depth/width aspect ratio of the 3D box is controlled by `ax.set_box_aspect`. `roll` has nothing to do with spatial proportions.
- **B is correct:** `roll` is a third Euler angle that rotates the camera around the line of sight (the axis pointing from the camera position toward the scene center). In photography terms, it is the "bank" or "tilt" rotation that tilts the camera sideways. A non-zero roll causes the horizon line to appear tilted in the rendered image. The `roll` parameter was added to `view_init` in Matplotlib 3.5.
- **C is incorrect:** `roll` and `azim` are completely different rotations. `azim` rotates the camera around the world z-axis (yaw); `roll` rotates the camera around its own viewing axis (bank). They are not aliases.
- **D is incorrect:** Field-of-view / zoom is conceptually controlled by `ax.set_proj_type('ortho'/'persp')` or the `focal_length` parameter in perspective projection mode. `roll` is a rotation parameter, not a zoom parameter.

</details>

---

### Question 90

A machine learning engineer renders four views of the same neural network loss landscape and labels them. He claims view D is an **orthographic** (parallel projection) view. Examine the code:

```python
# View D code
ax.set_proj_type('ortho')
ax.view_init(elev=30, azim=45)
```

What is the key visual difference between `set_proj_type('ortho')` and the default `set_proj_type('persp')`?

- [ ] A) In orthographic projection, parallel lines remain parallel in the rendered image; in perspective projection, parallel lines converge toward a vanishing point
- [ ] B) Orthographic projection adds a shadow cast onto the floor; perspective projection does not
- [ ] C) Orthographic projection shows a wider field of view; perspective projection is zoomed in by a factor of 2
- [ ] D) Orthographic projection is only available for `plot_wireframe`; perspective projection works with all surface types

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) In orthographic projection, parallel lines remain parallel in the rendered image; in perspective projection, parallel lines converge toward a vanishing point

**Explanation:**
- **A is correct:** This is the fundamental mathematical distinction between the two projection types. In orthographic (parallel) projection, the projection rays are parallel — size does not diminish with distance, and lines that are parallel in 3D space remain parallel in the 2D projection. In perspective projection, projection rays converge at the camera (eye point), causing distant objects to appear smaller and parallel lines to converge toward vanishing points. Orthographic is useful for technical drawings and measurements; perspective creates more natural-looking depth.
- **B is incorrect:** Neither orthographic nor perspective projection automatically adds shadows. Shadow effects in Matplotlib must be explicitly programmed (e.g., by plotting projections onto wall planes). The `set_proj_type` function only changes the projection mathematics.
- **C is incorrect:** The apparent field of view difference between `'ortho'` and `'persp'` is about geometry, not a fixed 2× zoom factor. In practice, switching from perspective to orthographic often makes the plot appear slightly larger or smaller depending on the focal length, but this is not a fixed "2× zoom" relationship.
- **D is incorrect:** Both `'ortho'` and `'persp'` projection types work with all 3D visualization functions — `plot_surface`, `plot_wireframe`, `scatter`, `plot`, and `bar3d`. There is no function-level restriction on projection type.

</details>
