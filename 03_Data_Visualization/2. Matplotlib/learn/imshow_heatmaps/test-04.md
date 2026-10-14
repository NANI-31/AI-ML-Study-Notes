# Matplotlib Imshow & Heatmaps — Test 04
## Interpolation Modes

---

### Question 31

A developer renders a 4×4 heatmap of model accuracy scores and notices the result looks blurry with smooth gradients between cells:

```python
scores = np.array([
    [0.91, 0.85, 0.78, 0.70],
    [0.88, 0.82, 0.75, 0.68],
    [0.84, 0.79, 0.72, 0.65],
    [0.80, 0.75, 0.68, 0.60],
])
ax.imshow(scores, cmap='RdYlGn', interpolation='bilinear')
```

Why is `interpolation='bilinear'` wrong for this use case, and what should be used instead?

- [ ] A) `bilinear` is wrong because it is too slow; use `nearest` for speed
- [ ] B) `bilinear` smooths between cell values, creating gradients that suggest continuous variation between discrete score cells — use `interpolation='nearest'` to render each cell as a solid uniform-color block
- [ ] C) `bilinear` is wrong because it requires float64 arrays; use `bicubic` instead
- [ ] D) `bilinear` inverts the colormap direction; use `interpolation='none'` to fix the color order

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `bilinear` smooths between cell values, creating gradients that suggest continuous variation between discrete score cells — use `interpolation='nearest'` to render each cell as a solid uniform-color block

**Explanation:**

- **A is incorrect:** Performance is not the reason to avoid `bilinear`. For small arrays like (4,4), all interpolation modes are effectively instantaneous. The issue is semantic correctness, not speed.
- **B is correct:** `interpolation='bilinear'` upsamples the image by computing weighted averages between adjacent pixel values, producing smooth color gradients between cells. For a matrix of discrete scores, this is misleading: the boundary between score 0.91 and 0.85 does not have a meaningful intermediate value — it's a discrete cell boundary. `interpolation='nearest'` renders each cell as a solid block with its exact colormap color, with sharp edges at cell boundaries. This is the correct choice for **confusion matrices, heatmaps, discrete grids, and any annotated matrix**.
- **C is incorrect:** `bilinear` works with all numeric dtypes including float32, float64, and uint8. Dtype is not a factor in choosing interpolation mode.
- **D is incorrect:** Interpolation mode has no effect on colormap direction. The colormap is determined by `cmap=` parameter and `vmin`/`vmax`. `interpolation='none'` is similar to `nearest` in intent (no interpolation) but may behave differently depending on the display backend.

</details>

---

### Question 32

A developer renders the same 8×8 matrix with different interpolation modes side-by-side. Which description correctly pairs each mode with its visual effect?

```python
modes = ['nearest', 'bilinear', 'bicubic', 'gaussian']
```

- [ ] A) `nearest`: smooth gradients; `bilinear`: solid blocks; `bicubic`: maximum blur; `gaussian`: pixel-exact
- [ ] B) `nearest`: solid uniform-color blocks (pixel-exact); `bilinear`: smooth linear gradients; `bicubic`: smoother curved gradients with possible ringing artifacts; `gaussian`: softened blur preserving overall shape
- [ ] C) `nearest` and `bilinear` are identical for integer arrays; `bicubic` and `gaussian` are identical for arrays under 16×16
- [ ] D) `nearest`: pixelated with black borders; `bilinear`: anti-aliased edges; `bicubic`: compressed image; `gaussian`: expanded image

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `nearest`: solid uniform-color blocks (pixel-exact); `bilinear`: smooth linear gradients; `bicubic`: smoother curved gradients with possible ringing artifacts; `gaussian`: softened blur preserving overall shape

**Explanation:**

- **A is incorrect:** This reverses `nearest` and `bilinear`. `nearest` produces solid blocks (no smoothing), not smooth gradients.
- **B is correct:** Each mode's visual behavior: **`nearest`** — assigns each screen pixel the color of its nearest data point; produces hard-edged solid blocks (pixelated look, but faithful to discrete data). **`bilinear`** — linear interpolation between 4 surrounding data points; produces smooth but slightly blurry appearance. **`bicubic`** — cubic spline interpolation over a 4×4 neighborhood; produces smoother curves than bilinear but can introduce overshoot/ringing at high-contrast edges. **`gaussian`** — Gaussian kernel blurring; produces the softest look while preserving the overall spatial structure.
- **C is incorrect:** `nearest` and `bilinear` produce visibly different results for any array larger than 1×1 when displayed larger than its natural pixel size. They are not identical for integer arrays.
- **D is incorrect:** None of these modes add black borders, compress/expand the image dimensions, or perform anti-aliasing in the traditional graphics sense. These descriptions are fabricated.

</details>

---

### Question 33

A developer asks: "What is the difference between `interpolation='nearest'` and `interpolation='none'`?"

- [ ] A) They are completely identical — both produce pixel-exact rendering with no smoothing
- [ ] B) `'none'` requests no interpolation but delegates the actual rendering to the display backend — for vector backends (PDF, SVG), `'none'` may still apply some backend-level anti-aliasing; `'nearest'` always forces nearest-neighbor in the raster image before export
- [ ] C) `'none'` applies linear interpolation; `'nearest'` applies no interpolation
- [ ] D) `'none'` is only valid for uint8 arrays; `'nearest'` works for all dtypes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `'none'` requests no interpolation but delegates the actual rendering to the display backend — for vector backends (PDF, SVG), `'none'` may still apply some backend-level anti-aliasing; `'nearest'` always forces nearest-neighbor in the raster image before export

**Explanation:**

- **A is incorrect:** While `nearest` and `none` are visually similar in many contexts, they are not identical in behavior. The distinction matters for publication-quality vector output.
- **B is correct:** `interpolation='none'` tells Matplotlib "perform no interpolation at the Matplotlib level" — but the actual rendering may still apply backend-level smoothing (especially for PDF/SVG/PS vector backends where images are embedded). `interpolation='nearest'` explicitly computes nearest-neighbor upsampling in the image data before sending to the backend, guaranteeing pixel-exact blocks regardless of backend. For raster backends (PNG, JPEG), they appear identical. For vector backends, `nearest` is more reliable for ensuring sharp cell boundaries.
- **C is incorrect:** This reverses the definitions. `'none'` requests no Matplotlib-level interpolation; `'nearest'` applies nearest-neighbor — they are both no-smoothing approaches, not opposites.
- **D is incorrect:** Both `'none'` and `'nearest'` work with all array dtypes. Dtype restriction is not a property of interpolation mode.

</details>

---

### Question 34

A developer displays a 512×512 satellite image using `bicubic` interpolation and notices strange bright/dark halos around high-contrast edges (e.g., coastlines). What causes these artifacts?

- [ ] A) The `bicubic` colormap distorts colors at high-contrast boundaries
- [ ] B) Bicubic interpolation uses cubic splines that can **overshoot** beyond the local data range at sharp transitions — producing ringing artifacts (Gibbs phenomenon) that appear as bright or dark halos around edges
- [ ] C) The satellite image has incorrect `vmin`/`vmax` settings causing edge clipping
- [ ] D) `bicubic` only works correctly for square arrays — 512×512 is too large

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Bicubic interpolation uses cubic splines that can **overshoot** beyond the local data range at sharp transitions — producing ringing artifacts (Gibbs phenomenon) that appear as bright or dark halos around edges

**Explanation:**

- **A is incorrect:** Colormaps and interpolation modes are independent operations. The colormap maps scalar values to colors — interpolation operates on the scalar data before colormap application. A `bicubic` interpolation artifact is a data value artifact, not a colormap artifact.
- **B is correct:** Cubic spline interpolation fits smooth curves through data points. At sharp discontinuities (e.g., ocean pixels at 0 adjacent to land pixels at 200), the cubic fit overshoots — interpolated values briefly exceed the data range (e.g., going to -10 or 210). These overshoots then map to extreme colormap colors (darker or brighter than the actual data values), appearing as halos or ringing artifacts. This is the **Gibbs phenomenon** applied to image interpolation. For images with sharp edges, `nearest` or `bilinear` are safer choices.
- **C is incorrect:** Ringing artifacts are not caused by `vmin`/`vmax` settings. They are a mathematical property of cubic spline interpolation at discontinuities.
- **D is incorrect:** `bicubic` has no array size limit. It works correctly for any size array. 512×512 is a standard image size.

</details>

---

### Question 35

A developer creates a **publication figure** of a confusion matrix for a 3-class classifier and wants pixel-perfect sharp cell boundaries visible even in the exported PDF:

```python
cm = np.array([[45, 3, 2], [4, 38, 8], [1, 5, 44]])
fig.savefig("confusion_matrix.pdf", dpi=300)
```

Which interpolation setting guarantees sharp boundaries in the PDF output?

- [ ] A) `interpolation='bilinear'` — best for PDF vector output
- [ ] B) `interpolation='none'` — delegates to PDF backend which applies optimal smoothing
- [ ] C) `interpolation='nearest'` — forces nearest-neighbor rasterization before PDF embedding, guaranteeing sharp cell boundaries
- [ ] D) No interpolation setting is needed — PDF output is always pixel-perfect regardless of `interpolation`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `interpolation='nearest'` — forces nearest-neighbor rasterization before PDF embedding, guaranteeing sharp cell boundaries

**Explanation:**

- **A is incorrect:** `bilinear` produces smooth color gradients between cells — exactly what you do NOT want for a confusion matrix where each cell must have a uniform solid color with sharp boundaries.
- **B is incorrect:** `interpolation='none'` for vector backends (PDF, SVG) may apply backend-level anti-aliasing smoothing. For critical publication figures requiring pixel-exact sharp boundaries, relying on backend behavior is not guaranteed.
- **C is correct:** `interpolation='nearest'` computes the upsampled image at the Matplotlib level using nearest-neighbor (each display pixel gets the color of its nearest data cell). The result is then embedded in the PDF as a raster image with sharp, guaranteed cell boundaries regardless of the backend. For confusion matrices, annotated heatmaps, and discrete matrices in publications, `interpolation='nearest'` is the standard and correct choice.
- **D is incorrect:** PDF rendering of embedded raster images can vary by viewer and backend. The `interpolation` parameter does matter for ensuring consistent appearance across all PDF viewers. Without explicit `nearest`, some viewers may apply smoothing when zooming.

</details>

---

### Question 36

A developer investigates what happens when they call `ax.imshow()` on a very small array displayed in a large figure:

```python
tiny = np.array([[0, 1], [2, 3]])  # 2×2 array
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(tiny, cmap='viridis', interpolation='nearest')
```

The 2×2 array is displayed in an 800×800 pixel figure (approximately). What does `nearest` interpolation produce for this extreme upscaling?

- [ ] A) Each of the 4 cells fills roughly 1/4 of the figure as a solid uniformly-colored block — each "pixel" becomes approximately 400×400 screen pixels
- [ ] B) The image is automatically resized to 800×800 by duplicating and interpolating values
- [ ] C) `nearest` cannot upscale by more than 10× — an error is raised for extreme upscaling
- [ ] D) Each cell produces a smooth gradient to the next cell because the extreme upscaling forces bilinear interpolation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Each of the 4 cells fills roughly 1/4 of the figure as a solid uniformly-colored block — each "pixel" becomes approximately 400×400 screen pixels

**Explanation:**

- **A is correct:** `interpolation='nearest'` assigns each screen pixel the color of its nearest data point. For a 2×2 array displayed at 800×800 pixels: each data cell maps to approximately 400×400 screen pixels. The result is 4 large solid-color blocks — the array's 4 values rendered as 4 giant uniformly-colored squares. This is the "blocky/pixelated" look characteristic of nearest-neighbor upscaling. This is exactly the desired behavior for showing discrete data at large scale.
- **B is incorrect:** Matplotlib does not duplicate or create new data values. The rendering upscales the display without changing the underlying 2×2 data array.
- **C is incorrect:** There is no upscaling limit for `nearest` interpolation. Any upscaling factor is valid — the array is simply displayed with larger blocks.
- **D is incorrect:** `nearest` does NOT switch to bilinear for extreme upscaling. The interpolation mode is always obeyed. Bilinear would only be used if `interpolation='bilinear'` were specified.

</details>

---

### Question 37

A developer applies `interpolation='gaussian'` to a heatmap of sensor readings. A manager asks: "How does the Gaussian kernel affect the displayed values — does it change the underlying data?" What is the correct answer?

- [ ] A) Yes — `gaussian` interpolation permanently modifies the underlying NumPy array to store smoothed values
- [ ] B) No — `gaussian` interpolation only affects the visual rendering; the underlying data array is unchanged. The smoothing is applied during the display upsampling process, not to the data itself
- [ ] C) Yes — `gaussian` interpolation replaces each value with the weighted average of its 3×3 neighborhood in the original array
- [ ] D) No — `gaussian` interpolation simply applies a gray overlay to the entire image without changing pixel values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `gaussian` interpolation only affects the visual rendering; the underlying data array is unchanged. The smoothing is applied during the display upsampling process, not to the data itself

**Explanation:**

- **A is incorrect:** `ax.imshow()` never modifies the input NumPy array. The array is read-only from Matplotlib's perspective — no in-place modifications occur.
- **B is correct:** All `interpolation` modes in `ax.imshow()` are **display-only** operations. They control how the image is upsampled from its native resolution to the display resolution — affecting the rendered pixels on screen or in the exported file. The original NumPy array `data` passed to `imshow` remains completely unchanged. If you call `data.copy()` after `ax.imshow(data, interpolation='gaussian')`, you get the original unsmoothed values. The Gaussian smoothing exists only in the rendered output.
- **C is incorrect:** The 3×3 weighted average description relates to data-level Gaussian filtering (e.g., `scipy.ndimage.gaussian_filter(data, sigma=1)`). The `interpolation='gaussian'` parameter in `imshow` applies Gaussian smoothing during the upsampling stage — not as a pre-processing step on the raw data values.
- **D is incorrect:** `gaussian` interpolation applies spatial blurring (smoothed upsampling), not a gray overlay. Color values change based on the Gaussian-weighted neighborhood — it is not a simple uniform overlay.

</details>

---

### Question 38

A developer needs to display a 1000×1000 satellite image and wants to compare two interpolation approaches for detecting fine structure:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(satellite, cmap='gray', interpolation='nearest')
ax2.imshow(satellite, cmap='gray', interpolation='bicubic')
```

For which use case should the developer prefer `bicubic` over `nearest`?

- [ ] A) When displaying discrete labeled maps where cell boundaries must be sharp
- [ ] B) When displaying natural photographic or continuous-field data (satellite imagery, medical scans) where smooth appearance aids interpretation and no discrete boundaries need preservation
- [ ] C) When the array contains integer values between 0 and 255
- [ ] D) When the figure will be saved as PNG (raster format)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) When displaying natural photographic or continuous-field data (satellite imagery, medical scans) where smooth appearance aids interpretation and no discrete boundaries need preservation

**Explanation:**

- **A is incorrect:** Discrete labeled maps (land cover, confusion matrices, categorical grids) require `nearest` — sharp boundaries are essential for reading discrete categories. `bicubic` would blur the boundaries, making it impossible to see where one class ends and another begins.
- **B is correct:** For **continuous-field data** like satellite imagery, weather maps, or medical scans, the underlying phenomenon is physically smooth — hard pixel edges (from `nearest`) would create an artificially blocky appearance that doesn't reflect the true spatial continuity. `bicubic` interpolation creates a smooth, natural-looking rendering that better represents the continuous physical data. The tradeoff (ringing artifacts at sharp edges) is usually acceptable for natural imagery that lacks extreme discontinuities.
- **C is incorrect:** Array dtype (uint8 vs. float vs. int) does not determine which interpolation mode is appropriate. The choice depends on the semantic nature of the data (discrete vs. continuous), not its storage type.
- **D is incorrect:** PNG is a raster format — the interpolation mode affects how the image looks at the display resolution, regardless of output format. Both PNG and PDF behave the same way for `nearest` vs. `bicubic`.

</details>

---

### Question 39

A developer creates a heatmap of a symmetric distance matrix (100×100) and uses `interpolation='bilinear'`. After adding cell text annotations, they notice the text values don't align with the visual color at each cell position. What is the root cause?

- [ ] A) The text annotation uses wrong indices — `ax.text(i, j)` should be `ax.text(j, i)`
- [ ] B) `bilinear` interpolation blends cell colors across boundaries — the center of each annotated cell shows a blended color influenced by neighbors, not the cell's true value. Use `interpolation='nearest'` so each cell has a uniform color matching its annotated value
- [ ] C) Cell text annotations are displayed on a separate layer above the interpolated image, causing a parallax offset
- [ ] D) The distance matrix is symmetric — `imshow` automatically transposes symmetric matrices, shifting the visual by one cell

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `bilinear` interpolation blends cell colors across boundaries — the center of each annotated cell shows a blended color influenced by neighbors, not the cell's true value. Use `interpolation='nearest'` so each cell has a uniform color matching its annotated value

**Explanation:**

- **A is incorrect:** Index transposition (`i,j` vs `j,i`) is a separate bug related to coordinate axis ordering. While important, it's not what causes text-color misalignment due to interpolation.
- **B is correct:** When `bilinear` interpolation is active, each displayed pixel's color is a weighted blend of surrounding data values. At the visual center of cell `[i, j]`, the color is dominated by `data[i,j]` but influenced by adjacent cells. For large matrices or high-contrast neighbors, this blending is visible — the cell center color won't exactly match the pure colormap color for `data[i,j]`. With `interpolation='nearest'`, each cell is a uniform solid block — the annotated text value and the cell background color perfectly correspond to the same data value `data[i,j]`.
- **C is incorrect:** There is no parallax or layer offset in Matplotlib's rendering. Text and image are rendered in the same axes coordinate system.
- **D is incorrect:** `imshow` does NOT automatically transpose symmetric matrices. Array `data[i,j]` is always displayed at row i, column j — symmetry is irrelevant to how Matplotlib renders the array.

</details>

---

### Question 40

A developer creates a downsampled thumbnail of a large heatmap (from 200×200 to 20×20) using:

```python
large = np.random.rand(200, 200)
small = large[::10, ::10]  # 20×20 thumbnail
ax.imshow(small, interpolation='nearest')
```

A colleague argues: "Using `nearest` interpolation means we lose information because we're not averaging the 10×10 blocks." Is the colleague correct about what `interpolation='nearest'` does in this context?

- [ ] A) Yes — `nearest` discards 99% of the original data
- [ ] B) No — the data loss occurred during downsampling `large[::10, ::10]`, not during `imshow`. The `interpolation` parameter only controls display upsampling — it has no role in the data reduction step
- [ ] C) No — `nearest` automatically averages 10×10 blocks internally when detecting downsampled data
- [ ] D) Yes — `nearest` uses only the top-left pixel of each block, discarding the rest

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — the data loss occurred during downsampling `large[::10, ::10]`, not during `imshow`. The `interpolation` parameter only controls display upsampling — it has no role in the data reduction step

**Explanation:**

- **A is incorrect:** The colleague is confused about WHAT performs the data reduction. `nearest` interpolation in `imshow` is not responsible for the data loss — it only controls how the 20×20 array is rendered at display resolution.
- **B is correct:** The data reduction from 200×200 to 20×20 happens in NumPy slicing: `large[::10, ::10]` selects every 10th element in both dimensions — this IS a form of strided downsampling that discards most values (the colleague is right that information is lost, but wrong about WHERE). Once `small` is created (20×20), `ax.imshow(small, interpolation='nearest')` only controls how this 20×20 array is displayed at screen resolution — rendering each cell as a solid block. The `interpolation` parameter has no access to or knowledge of the original `large` array.
- **C is incorrect:** `imshow` never performs automatic block averaging. It operates only on the array it receives (`small`). It has no mechanism to "look back" at the original data.
- **D is incorrect:** `nearest` in `imshow` assigns each screen pixel the nearest data value from the INPUT array (`small`, already 20×20). The "top-left pixel discarding" confusion refers to the strided slicing `[::10, ::10]` in NumPy — which does select every 10th element — but this is a NumPy operation, not an `imshow` operation.

</details>
