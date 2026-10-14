# 3D Matplotlib — Test 10: Machine Learning 3D Applications

---

### Question 91

A machine learning researcher plots a 3D loss landscape and identifies a **saddle point** in the surface. Which geometric description correctly characterizes a saddle point in a 3D loss surface?

- [ ] A) A point where the loss is a local minimum in all directions — the gradient is zero and the Hessian is positive definite
- [ ] B) A point where the loss is a local maximum in all directions — the gradient is zero and the Hessian is negative definite
- [ ] C) A point where the gradient is zero but the Hessian has both positive and negative eigenvalues — the loss is a local minimum in some directions and a local maximum in others
- [ ] D) A point where the loss function is discontinuous and the gradient is undefined

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) A point where the gradient is zero but the Hessian has both positive and negative eigenvalues — the loss is a local minimum in some directions and a local maximum in others

**Explanation:**
- **A is incorrect:** A point where the gradient is zero and the Hessian is positive definite (all eigenvalues positive) is a **local minimum** — the loss increases in all directions from this point. Saddle points are not minima.
- **B is incorrect:** A point where the gradient is zero and the Hessian is negative definite (all eigenvalues negative) is a **local maximum** — the loss decreases in all directions. Saddle points are not maxima.
- **C is correct:** A saddle point has zero gradient (a critical point) but an indefinite Hessian — one with both positive and negative eigenvalues. Along the eigenvectors with positive eigenvalues, the loss curves upward (local min direction). Along the eigenvectors with negative eigenvalues, the loss curves downward (local max direction). The visual shape is a hyperbolic paraboloid — like a horse saddle or Pringles chip. In high-dimensional neural network loss landscapes, saddle points are far more common than local minima and can slow gradient descent by creating nearly flat plateaux.
- **D is incorrect:** Saddle points are smooth critical points of differentiable loss functions. The gradient exists and equals zero at the saddle point. The function is continuous and differentiable at and around it.

</details>

---

### Question 92

A data scientist visualizes a convex quadratic loss surface:

```python
W1 = np.linspace(-3, 3, 60)
W2 = np.linspace(-3, 3, 60)
W1_grid, W2_grid = np.meshgrid(W1, W2)
Loss = 2 * W1_grid**2 + 0.5 * W2_grid**2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(W1_grid, W2_grid, Loss, cmap="coolwarm",
                       edgecolor="none", alpha=0.8)
```

Where is the global minimum of this loss surface, and what is the minimum loss value?

- [ ] A) Global minimum at (W1=1, W2=1), Loss = 2.5
- [ ] B) Global minimum at (W1=0, W2=0), Loss = 0
- [ ] C) Global minimum at (W1=-3, W2=-3), Loss determined by the boundary
- [ ] D) The surface has no global minimum because it is a quadratic and unbounded above

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Global minimum at (W1=0, W2=0), Loss = 0

**Explanation:**
- **A is incorrect:** Substituting W1=1, W2=1: Loss = 2(1)² + 0.5(1)² = 2 + 0.5 = 2.5. While this is a valid point, it is not the minimum. The minimum of `2*W1² + 0.5*W2²` is achieved when both terms are minimized simultaneously, which requires W1=0 and W2=0.
- **B is correct:** The loss function `Loss = 2*W1² + 0.5*W2²` is a sum of two non-negative terms. Each term is minimized at zero: `2*W1²` is minimized at W1=0, and `0.5*W2²` is minimized at W2=0. At (W1=0, W2=0): Loss = 2(0)² + 0.5(0)² = 0. Since both terms are non-negative, this is the global minimum. The surface is a convex elliptic paraboloid (bowl shape) with its vertex at the origin.
- **C is incorrect:** The global minimum is at the interior point (0, 0), not at the boundary. The boundary values W1=±3, W2=±3 give higher loss values (e.g., Loss at (-3,-3) = 2(9) + 0.5(9) = 18 + 4.5 = 22.5).
- **D is incorrect:** The loss function `2*W1² + 0.5*W2²` is bounded below by 0 and has a well-defined global minimum at (0, 0). While it is unbounded above (grows without limit as W1, W2 → ±∞), this does not prevent it from having a global minimum.

</details>

---

### Question 93

A machine learning engineer visualizes a gradient descent trajectory overlaid on a non-convex loss surface with multiple local minima. The trajectory starts at `(W1=-2, W2=2)` and converges to a local minimum at `(W1=-1.5, W2=0.5)` rather than the global minimum at `(W1=0.5, W2=-0.5)`. Which visualization technique BEST communicates that the optimizer is trapped in a local minimum?

- [ ] A) Change the loss surface colormap from `"viridis"` to `"plasma"` to better highlight the depth of the minima
- [ ] B) Add 3D contour projections on all three walls and mark both the local and global minima with distinct 3D scatter markers (e.g., red star at the local minimum, green star at the global minimum) so the spatial relationship between the trajectory endpoint and the global minimum is clear
- [ ] C) Increase the surface `rstride` and `cstride` to 20 to reveal the full depth of each minimum basin
- [ ] D) Set `elev=90, azim=-90` to show a top-down view that makes all minima equidistant from the viewer

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add 3D contour projections on all three walls and mark both the local and global minima with distinct 3D scatter markers (e.g., red star at the local minimum, green star at the global minimum) so the spatial relationship between the trajectory endpoint and the global minimum is clear

**Explanation:**
- **A is incorrect:** Changing the colormap improves the colour rendering but does not communicate the concept of "trapped in a local minimum." Both minima may appear visually similar in depth with any colormap if not explicitly annotated.
- **B is correct:** Explicitly marking both the trajectory's convergence point (local minimum) and the unreached global minimum with distinct markers (different colours, different marker styles) directly communicates the trapping. Combined with contour floor projections (which show the topology and the multiple basins from above), the viewer can clearly see that the trajectory endpoint is in a different basin from the global minimum. This is the most direct and informative visualization strategy for this diagnostic message.
- **C is incorrect:** Increasing strides reduces rendering detail, making minima basins appear more coarse and actually harder to distinguish. Finer strides (smaller stride values) would show more surface detail, but even then, visually distinguishing local from global minima requires annotation.
- **D is incorrect:** A top-down view (`elev=90`) shows the xy-plane topology well but removes the z-dimension (loss height) from the view. In a top-down view, the depth of each minimum's basin (its loss value) is invisible — all minima appear as points in the xy-plane without any indication of which is deeper (global) vs shallower (local).

</details>

---

### Question 94

A researcher uses 3D scatter to visualize a PCA-reduced dataset with 3 principal components:

```python
pca = PCA(n_components=3)
embedding = pca.fit_transform(X_scaled)   # shape (N, 3)

ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=y_labels, cmap="Set1", s=15, depthshade=True)
```

A colleague claims that the 3D PCA scatter "shows the true separability of classes in the original high-dimensional space." Is this claim fully accurate?

- [ ] A) Yes — PCA preserves all class separation information when reducing to 3 components
- [ ] B) No — PCA retains only the directions of maximum variance (not necessarily maximum class separation); the true separability may be better or worse than what is visible in the 3D scatter, and information in discarded components is lost
- [ ] C) Yes — `n_components=3` is specifically chosen to maximize class discrimination when `y_labels` is available
- [ ] D) No — 3D PCA scatter only works for datasets with exactly 3 classes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — PCA retains only the directions of maximum variance (not necessarily maximum class separation); the true separability may be better or worse than what is visible in the 3D scatter, and information in discarded components is lost

**Explanation:**
- **A is incorrect:** PCA maximizes variance reconstruction but is unsupervised — it has no knowledge of class labels (`y_labels`) during the dimensionality reduction. The first 3 principal components may not align with the directions that best separate classes. Some class-discriminative structure may be encoded in principal components 4, 5, 6, etc., which are discarded.
- **B is correct:** PCA finds directions of maximum variance in the data. Class boundaries may or may not align with these directions. A dimensionality reduction that maximizes class separation would use Linear Discriminant Analysis (LDA) or t-SNE/UMAP, not PCA. The 3D PCA scatter shows the data's top-3 variance directions, which provides a useful but potentially incomplete view of class separability. Discarded components may contain class-relevant information.
- **C is incorrect:** PCA does not use `y_labels` at all — it is an unsupervised method. The `y_labels` are only used for scatter colouring after the dimensionality reduction is complete. `n_components=3` refers to the number of variance-maximizing directions retained, not to the number of classes.
- **D is incorrect:** PCA scatter works with any number of classes. The `c=y_labels` argument maps class labels to colours using the colormap. This works with 2, 10, or 100 classes (with appropriate colormaps for large class counts).

</details>

---

### Question 95

A data scientist visualizes a hyperparameter response surface with:

```python
# Grid search over learning rate and batch size
LR = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
BS = np.array([16, 32, 64, 128, 256])
LR_grid, BS_grid = np.meshgrid(LR, BS)
F1_grid = results_matrix   # shape (5, 5), F1 scores from grid search

ax.plot_surface(LR_grid, BS_grid, F1_grid, cmap="RdYlGn", edgecolor="k")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Batch Size")
ax.set_zlabel("F1 Score")
```

The x-axis (Learning Rate) spans from 0.001 to 0.1 — a 100× range — while the y-axis (Batch Size) spans 16 to 256 — a 16× range. What visual problem does this create and how should it be fixed?

- [ ] A) The surface appears elongated in the LR direction — fix by using `ax.set_box_aspect([1, 1, 1])`
- [ ] B) The linear x-axis compresses the small LR values (0.001–0.01) visually close together while the large values (0.01–0.1) appear spread apart, obscuring the response at low learning rates — fix by using a log-scaled x-axis or log-transforming LR before plotting
- [ ] C) The surface appears wider in the BS direction because 256 > 0.1 — fix by normalizing both axes to [0, 1] before meshgrid
- [ ] D) There is no visual problem — matplotlib automatically equalizes axes ranges for 3D plots

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The linear x-scale compresses the small LR values (0.001–0.01) visually close together while the large values (0.01–0.1) appear spread apart, obscuring the response at low learning rates — fix by using a log-scaled x-axis or log-transforming LR before plotting

**Explanation:**
- **A is incorrect:** `ax.set_box_aspect([1, 1, 1])` controls the physical aspect ratio of the 3D bounding box (making it a cube) but does not address the non-linear spacing of learning rate values. The visual compression of small LR values on a linear scale is a data representation issue, not a box geometry issue.
- **B is correct:** Learning rates are conventionally explored on a logarithmic scale because the effect of changing LR from 0.001 to 0.002 is comparable in magnitude to changing from 0.05 to 0.1 (both are 2× changes). On a linear axis, the interval 0.001–0.01 appears visually smaller than 0.01–0.1, even though both represent a 10× change. The fix is to use `np.log10(LR)` as the x-axis values (or plot on a log scale), which spaces the learning rate values according to their order of magnitude, giving each decade equal visual weight.
- **C is incorrect:** Normalizing to [0, 1] would solve the range disparity between LR (max 0.1) and BS (max 256) but would destroy the interpretability of the axis labels. The fundamental issue is the non-linear spacing of LR values, not the different ranges of the two axes.
- **D is incorrect:** Matplotlib 3D does not automatically equalize axis ranges or apply log scaling. The axis scales directly reflect the data values passed to `meshgrid`.

</details>

---

### Question 96

A researcher visualizes a non-convex loss surface with two local minima:

```python
Z = (W1**2 - 1)**2 + W2**2    # Double-well potential
```

From the top-down view (`elev=90, azim=-90`), what shape do the level curves of this double-well potential look like near the two minima?

- [ ] A) Two concentric circles centered at the origin
- [ ] B) Two separate closed oval/elliptical contours centered at W1 = -1 and W1 = +1 (at W2 = 0)
- [ ] C) A single elongated ellipse centered at the origin
- [ ] D) Four symmetric circles arranged at the four corners of the domain

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Two separate closed oval/elliptical contours centered at W1 = -1 and W1 = +1 (at W2 = 0)

**Explanation:**
- **A is incorrect:** Concentric circles centered at the origin would imply a single radially symmetric minimum at (0, 0). The double-well function has a local **maximum** at (0, 0) (since `(0² - 1)² = 1`) and two minima at W1=±1.
- **B is correct:** The double-well potential `(W1² - 1)² + W2²` has two global minima where `(W1² - 1)² = 0` (i.e., W1 = ±1) AND `W2² = 0` (i.e., W2 = 0). So the minima are at (-1, 0) and (+1, 0) where Z=0. Near each minimum, the level curves form closed contours (approximately elliptical for small deviations). Because the two minima are separated by a saddle/ridge along W1=0, the level curves form two distinct closed loops at low Z values, one centered at W1=-1 and one at W1=+1.
- **C is incorrect:** A single elongated ellipse would suggest a single minimum along an elongated valley, which is not the case for the double-well. The W1=0 ridge separates the two basin regions.
- **D is incorrect:** The function's symmetry is along the W1 axis (having minima at W1=±1) and is monotonically increasing in |W2|. There are no corner minima. The four-corner pattern would suggest a different functional form.

</details>

---

### Question 97

A machine learning engineer adds gradient descent optimization path markers to a 3D loss surface. After running the code, she notices the path trajectory appears below the surface (Z-values of the path match the surface Z, but the path is rendered behind the surface faces). Which code change most reliably fixes this rendering artifact?

- [ ] A) Increase the surface `alpha` from 0.8 to 1.0 to make it fully opaque and hide the path behind it intentionally
- [ ] B) Add a small positive offset to the trajectory Z-values: `loss_path_elevated = loss_path + 0.01 * Z_range`, so the path floats slightly above the surface
- [ ] C) Set `zorder=100` on the `ax.plot` call for the trajectory to guarantee it renders on top of all surface polygons
- [ ] D) Switch the surface to `plot_wireframe` instead of `plot_surface`, which renders the surface as lines and avoids face-polygon occlusion of the trajectory

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Add a small positive offset to the trajectory Z-values: `loss_path_elevated = loss_path + 0.01 * Z_range`, so the path floats slightly above the surface

**Explanation:**
- **A is incorrect:** Increasing alpha from 0.8 to 1.0 makes the surface MORE opaque, making the trajectory LESS visible rather than more. This is the opposite of the desired fix.
- **B is correct:** Since Matplotlib uses a painter's algorithm that sorts by approximate centroid depth, true on-surface geometry (Z trajectory values matching surface Z) creates an ambiguous depth ordering. By adding a small positive Z offset (e.g., `0.01 × Z_range`), the trajectory is moved slightly "above" the surface in world space. This ensures the trajectory's depth position is consistently in front of the surface polygons from any camera angle, solving the rendering artifact. The offset is typically small enough to be visually imperceptible while guaranteeing correct depth ordering.
- **C is incorrect:** `zorder` has limited effectiveness in Matplotlib 3D (as discussed in test-08 Question 80). The painter's algorithm can override `zorder` values based on geometric depth, so setting `zorder=100` does not reliably guarantee the trajectory appears in front of the surface.
- **D is incorrect:** Switching to `plot_wireframe` removes the filled surface faces that cause the occlusion, which would make the trajectory visible. However, this changes the visualization type significantly — the analyst loses the continuous coloured surface. This is a workaround that sacrifices visual information rather than a targeted fix for the depth artifact.

</details>

---

### Question 98

A data scientist creates a 3D visualization of the neural network training dynamics for a simple two-parameter network. She plots the loss surface and marks 100 SGD training steps as a colored trajectory, using the step number as the colour variable:

```python
from matplotlib.collections import Line3DCollection

points = np.array([w1_steps, w2_steps, loss_steps]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = Line3DCollection(segments, cmap="autumn", linewidth=2)
lc.set_array(np.arange(len(segments)))
ax.add_collection3d(lc)
fig.colorbar(lc, ax=ax, label="SGD Step")
```

Why is `Line3DCollection` used instead of a single `ax.plot` call with `c=step_number`?

- [ ] A) `ax.plot` does not support 3D coordinates and requires `Line3DCollection` for all 3D line rendering
- [ ] B) `ax.plot` can only colour a line with a single uniform colour; `Line3DCollection` enables per-segment colour variation by mapping each segment to a position in the colormap
- [ ] C) `Line3DCollection` renders faster than `ax.plot` for trajectories with more than 50 points
- [ ] D) `ax.plot` requires all three coordinate arrays to have the same data type; `Line3DCollection` handles mixed integer/float coordinate arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.plot` can only colour a line with a single uniform colour; `Line3DCollection` enables per-segment colour variation by mapping each segment to a position in the colormap

**Explanation:**
- **A is incorrect:** `ax.plot(x, y, z)` is the primary and simplest way to draw 3D lines on `Axes3D`. `Line3DCollection` is a more advanced class for rendering multiple line segments with individual styling, but `ax.plot` is fully capable of 3D line rendering.
- **B is correct:** `ax.plot(x, y, z)` applies one `color` argument to the entire line. To colour-encode a continuous variable along the line (like SGD step number), each segment must be individually coloured — which is exactly what `Line3DCollection` enables. By passing a scalar array to `lc.set_array()`, each segment is mapped to a colour from the colormap based on its scalar value. This is the standard pattern for "rainbow trajectories" in Matplotlib.
- **C is incorrect:** For 100 points, the performance difference between `ax.plot` and `Line3DCollection` is negligible. The choice is driven by the need for per-segment colour variation, not performance.
- **D is incorrect:** Both `ax.plot` and `Line3DCollection` accept NumPy arrays and handle float/integer types. Mixed data types are not a distinguishing factor between the two approaches.

</details>

---

### Question 99

A researcher builds a 3D visualization comparing model performance across a hyperparameter grid. She finds that the response surface has a very flat region (plateau) and a sharp cliff where performance drops precipitously. She wants to highlight the cliff visually. Which technique is MOST effective?

- [ ] A) Use `rstride=1, cstride=1` (finest possible stride) to maximize polygon density near the cliff
- [ ] B) Use a diverging colormap centered at the cliff's F1 threshold, combined with `vmin` and `vmax` set to the range just around the cliff, to amplify the colour contrast at the decision boundary between good and poor hyperparameter regions
- [ ] C) Add `depthshade=True` to the scatter markers to fade the flat plateau region
- [ ] D) Set the colormap to `"binary"` to create a binary black/white visualization of the cliff

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Use a diverging colormap centered at the cliff's F1 threshold, combined with `vmin` and `vmax` set to the range just around the cliff, to amplify the colour contrast at the decision boundary between good and poor hyperparameter regions

**Explanation:**
- **A is incorrect:** Increasing stride density (decreasing `rstride`/`cstride`) improves the surface geometry approximation but does not change the colour encoding. A fine-stride surface still colours the plateau and the cliff with the same colormap proportionally, without emphasizing the cliff specifically.
- **B is correct:** By setting `vmin` and `vmax` to the F1 range bracketing the cliff (e.g., `vmin=0.65, vmax=0.85` if the cliff is at F1=0.75), the full colormap range is concentrated in the narrow region of interest. Using a diverging colormap (like `"RdYlGn"`) centered at the cliff threshold (e.g., using `TwoSlopeNorm` or `vmin`/`vmax` centering) makes the cliff appear as a sharp, high-contrast colour transition while the flat plateau appears uniformly coloured. This directly amplifies the visual representation of the cliff.
- **C is incorrect:** `depthshade=True` is a parameter for `ax.scatter`, not for `plot_surface`. It controls depth-based alpha modulation for scatter points, not surface polygon shading. It is unrelated to emphasizing performance cliffs.
- **D is incorrect:** A binary (`"binary"`) colormap would create a sharp visual transition at the midpoint of the full F1 range, but only if `vmin`/`vmax` are carefully set to center at the cliff value. Using `"binary"` without adjustment might not place the black/white boundary at the cliff. More informatively, a diverging colormap centered at the cliff with custom `vmin`/`vmax` provides richer information about the relative performance on both sides of the cliff.

</details>

---

### Question 100

A senior machine learning engineer presents a comprehensive 3D loss surface figure to her team. The figure includes: a colored surface, floor/side-wall contour projections, a gradient descent trajectory, and start/end markers. She receives feedback that "the interactive rotation doesn't work in the PDF export." What is the correct explanation and remedy?

- [ ] A) The figure's `dpi` setting is too high, preventing interactive rotation in PDFs; reduce `dpi` to 72 to enable interactive rotation
- [ ] B) PDF and PNG are raster/static formats that do not support interactive 3D camera rotation; interactive rotation requires either a Jupyter notebook with `%matplotlib widget` backend or an HTML export (e.g., via `mpld3` or Plotly)
- [ ] C) `ax.view_init` must be called with `interactive=True` to enable rotation in exported documents
- [ ] D) Only PNG format supports interactive 3D rotation; change `savefig` to use `.png` extension

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) PDF and PNG are raster/static formats that do not support interactive 3D camera rotation; interactive rotation requires either a Jupyter notebook with `%matplotlib widget` backend or an HTML export (e.g., via `mpld3` or Plotly)

**Explanation:**
- **A is incorrect:** DPI affects the resolution (pixel density) of the raster output. It has no relationship to interactive capabilities. Even at DPI=72, a PDF cannot support interactive 3D camera rotation. Interactivity is a format capability, not a resolution setting.
- **B is correct:** This is the correct technical explanation. PDF and PNG are static document formats. A PDF can embed vector graphics and text, but cannot execute Python code or maintain live interactive state for 3D camera rotation. Interactive 3D rotation in Matplotlib requires a live Python kernel running the figure's event loop. In Jupyter notebooks, `%matplotlib widget` (using the `ipympl` backend) enables interactive rotation within the notebook. For standalone web-ready interactive 3D plots, tools like Plotly Express, Bokeh, or Three.js-based libraries create HTML exports that support interactive rotation without requiring a Python kernel.
- **C is incorrect:** `ax.view_init` does not have an `interactive=True` argument. Setting the view interactivity is done at the backend level (e.g., choosing `%matplotlib widget` in Jupyter or `plt.ion()` for interactive mode), not through a `view_init` parameter.
- **D is incorrect:** PNG is a raster (bitmap) image format, even more static than PDF (which at least supports vector graphics and embedded JavaScript in some PDF viewers). PNG cannot support interactive 3D rotation in any form.

</details>
