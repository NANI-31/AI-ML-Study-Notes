# 3D Matplotlib — Test 02: 3D Scatter Plots

---

### Question 11

A data scientist visualizes a PCA cluster analysis by running:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc = ax.scatter(pc1, pc2, pc3,
                c=cluster_labels,
                cmap="tab10",
                s=40,
                depthshade=True)
fig.colorbar(sc, ax=ax, label="Cluster ID")
```

`cluster_labels` is an integer array with values 0 through 9. A reviewer notes that the colorbar shows a continuous gradient from 0 to 9 rather than 10 discrete colour bands. What is the most likely cause?

- [ ] A) `depthshade=True` blends cluster colours with a depth gradient, merging distinct colours into a continuous ramp
- [ ] B) `cmap="tab10"` only has 10 discrete colours but `colorbar` renders the mapping as a continuous linear gradient unless the colormap is explicitly discretized
- [ ] C) The `s=40` argument forces Matplotlib to use a sequential colormap regardless of the `cmap` parameter
- [ ] D) `fig.colorbar` always renders continuous gradients; discrete bands require `BoundaryNorm` and explicit tick placement on the colorbar

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `fig.colorbar` always renders continuous gradients; discrete bands require `BoundaryNorm` and explicit tick placement on the colorbar

**Explanation:**
- **A is incorrect:** `depthshade=True` modulates the *alpha* (transparency) of points based on their depth in the projection to create a 3D depth illusion. It does not affect how the colormap is applied to the `c` values. Depth shading is an overlay on top of the colour determined by `c` and `cmap`.
- **B is incorrect:** `tab10` is indeed a qualitative colormap with 10 discrete colours, but when Matplotlib maps integer values 0–9 through it using the default linear `Normalize`, the colorbar still renders continuously because the normalizer maps real-valued positions along [0, 1] to colours. The scatter glyphs themselves appear in discrete colours, but the bar legend renders a smooth ramp.
- **C is incorrect:** The marker size `s` has no relationship to colormap selection or the normalization mode. It controls only the rendered point area in points².
- **D is correct:** `fig.colorbar` by default renders a smooth continuous gradient from the minimum to the maximum value. To produce N discrete colour bands, the scatter must be created with a `BoundaryNorm`: `from matplotlib.colors import BoundaryNorm; norm = BoundaryNorm(boundaries=range(11), ncolors=10)`. The colorbar then shows distinct bands with one colour per integer cluster label.

</details>

---

### Question 12

Examine the following 3D scatter code:

```python
np.random.seed(0)
x = np.random.randn(200)
y = np.random.randn(200)
z = np.random.randn(200)
magnitude = np.sqrt(x**2 + y**2 + z**2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc = ax.scatter(x, y, z, c=magnitude, cmap="plasma", s=30, depthshade=False)
fig.colorbar(sc, ax=ax, label="Euclidean Distance from Origin")
```

What does `depthshade=False` change about the visual output compared to the default `depthshade=True`?

- [ ] A) Points farther from the camera are made smaller to simulate perspective — `False` makes all points the same apparent size
- [ ] B) Points farther from the camera are rendered with lower alpha (more transparent) — `False` keeps all points at full opacity, so the colour from `c=magnitude` is fully visible without depth-based dimming
- [ ] C) `depthshade` controls whether z-axis tick lines are drawn through the scatter cloud — `False` removes those lines
- [ ] D) `depthshade=False` disables the colormap entirely and renders all points in the default blue marker colour

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Points farther from the camera are rendered with lower alpha (more transparent) — `False` keeps all points at full opacity, so the colour from `c=magnitude` is fully visible without depth-based dimming

**Explanation:**
- **A is incorrect:** `depthshade` does not affect marker size. Matplotlib 3D scatter does not implement perspective scaling of point size based on depth by default. All points use the size specified by `s` regardless of their z-position or depth shading mode.
- **B is correct:** `depthshade=True` applies a depth-based alpha modulation: points that project farther from the viewer are rendered with reduced alpha, creating a sense of depth. With `depthshade=False`, this alpha modulation is disabled and every point is rendered at full opacity, meaning the colour derived from `c=magnitude` and `cmap="plasma"` is displayed without any dimming overlay.
- **C is incorrect:** `depthshade` is exclusively a per-point alpha modulation mechanism. It has no relationship to the visibility of axis tick lines or grid lines on the 3D bounding box.
- **D is incorrect:** `depthshade` does not interact with colormap assignment. The `cmap="plasma"` and `c=magnitude` parameters are fully active regardless of `depthshade` state.

</details>

---

### Question 13

A researcher creates a 3D scatter plot where point size encodes a fourth variable:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sizes = (confidence_scores * 200) + 10   # scales 0.0–1.0 → 10–210
ax.scatter(x, y, z, s=sizes, c=cluster_id, cmap="Set2", depthshade=True)
```

A colleague argues that `s=sizes` and `c=cluster_id` together encode **five** dimensions of data. Is this claim correct?

- [ ] A) No — `s` and `c` both encode the same dimension; they must map to the same variable to be valid
- [ ] B) Yes — `x`, `y`, `z` encode three spatial dimensions, `c` encodes a fourth (categorical cluster), and `s` encodes a fifth (continuous confidence score), for a total of 5 dimensions
- [ ] C) No — `depthshade=True` consumes the "size" channel, overriding `s=sizes` with depth-computed sizes
- [ ] D) Yes, but only if `cmap="Set2"` contains exactly as many colours as there are unique `cluster_id` values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Yes — `x`, `y`, `z` encode three spatial dimensions, `c` encodes a fourth (categorical cluster), and `s` encodes a fifth (continuous confidence score), for a total of 5 dimensions

**Explanation:**
- **A is incorrect:** `s` and `c` are completely independent visual encoding channels in Matplotlib 3D scatter. `s` controls marker area (in points²) and encodes one variable; `c` controls marker colour and can encode a different, unrelated variable. They do not need to map to the same variable.
- **B is correct:** This is a standard 5D scatter plot pattern used in data visualization. Position in (x, y, z) encodes three numeric or ordinal variables. Colour (`c`) encodes a fourth variable — here a categorical cluster identity. Marker area (`s`) encodes a fifth continuous variable — here the confidence score. This technique allows analysts to explore five-dimensional relationships in a single static plot.
- **C is incorrect:** `depthshade=True` modulates point alpha (transparency) based on camera depth, not point size. The `s` parameter remains fully effective and is not overridden by depth shading.
- **D is incorrect:** The number of colours in `Set2` does not determine whether the 5D encoding claim is logically valid. The dimensionality argument holds independently of colormap capacity.

</details>

---

### Question 14

Consider this 3D scatter code intended to color-encode training loss:

```python
ax.scatter(epoch, lr, batch_size,
           c=val_loss,
           cmap="hot_r",
           s=60,
           depthshade=False)
```

A reviewer replaces `cmap="hot_r"` with `cmap="hot"`. What changes in the resulting visualization?

- [ ] A) `"hot_r"` is the reverse of `"hot"` — high `val_loss` values (worse performance) are rendered in dark/black with `"hot_r"` but in bright/white with `"hot"`
- [ ] B) `"hot"` and `"hot_r"` are identical; the `_r` suffix has no effect on colourmap rendering
- [ ] C) `"hot_r"` maps high values to cool colours; swapping to `"hot"` maps high values to warm colours, changing the semantics entirely
- [ ] D) The colormap reversal only affects the colorbar legend, not the colours applied to the scatter points themselves

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `"hot_r"` is the reverse of `"hot"` — high `val_loss` values (worse performance) are rendered in dark/black with `"hot_r"` but in bright/white with `"hot"`

**Explanation:**
- **A is correct:** In Matplotlib, appending `_r` to any colormap name produces its reversed version. The `"hot"` colormap runs from black → dark red → orange → yellow → white (low to high). Its reverse `"hot_r"` runs from white → yellow → orange → dark red → black. So with `c=val_loss` and `cmap="hot_r"`, low loss values (good) map to bright/white and high loss values (bad) map to dark/black. Switching to `"hot"` reverses this, mapping low loss to dark and high loss to white.
- **B is incorrect:** The `_r` suffix is a documented Matplotlib convention that reverses every colormap. It has a definite and significant visual effect.
- **C is incorrect:** `"hot_r"` does not map high values to cool colours — it maps high values to dark (black) colours within the same warm palette. Cool colours (blues/greens) would come from completely different colormaps like `"cool"` or `"winter"`.
- **D is incorrect:** The `_r` reversal applies uniformly to both the scatter point colours and the colorbar legend. Both reflect the reversed mapping.

</details>

---

### Question 15

A machine learning engineer plots the 3D position of word embeddings:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                c=similarity_scores,
                cmap="RdYlGn",
                s=20,
                alpha=0.6,
                depthshade=True)
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Cosine Similarity")
```

The `shrink=0.5` argument in `fig.colorbar` controls which property of the colorbar?

- [ ] A) The number of colour levels rendered in the colorbar
- [ ] B) The fraction of the axes height that the colorbar occupies (shrinks it to 50% of the full axes height)
- [ ] C) The opacity of the colorbar, making it semi-transparent at 0.5 (fully opaque at 1.0)
- [ ] D) The width of the colorbar relative to the standard default width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The fraction of the axes height that the colorbar occupies (shrinks it to 50% of the full axes height)

**Explanation:**
- **A is incorrect:** The number of colour levels in the colorbar is controlled by `ticks` or `boundaries` or the colormap's `N` parameter. `shrink` has no effect on colour resolution.
- **B is correct:** `shrink` is a fractional multiplier applied to the length of the colorbar. A value of `0.5` means the colorbar is rendered at 50% of the height (for vertical colorbars) or 50% of the width (for horizontal colorbars) that it would otherwise occupy. This is commonly used with 3D axes because the default colorbar can be disproportionately large relative to the 3D bounding box.
- **C is incorrect:** Colorbar transparency is controlled by setting the alpha property on the colorbar's axes or its patch, not by `shrink`. The `shrink` parameter has nothing to do with opacity.
- **D is incorrect:** The width of the colorbar relative to its length is controlled by `aspect` (in option D's scenario, `aspect=10` makes the colorbar 10× longer than it is wide). `shrink` scales the overall length, not the aspect ratio.

</details>

---

### Question 16

A data scientist debugs the following code:

```python
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(x_train, y_train, z_train, c="steelblue", s=50, depthshade=True)
ax.scatter(x_test, y_test, z_test, c="coral", s=50, depthshade=True)
```

She notices that `depthshade=True` sometimes makes a steelblue point look nearly the same color as a coral point when they happen to be at similar depths. What is the underlying reason?

- [ ] A) `depthshade` blends the point colour toward a uniform grey as depth increases, converging all colours toward the same grey at maximum depth
- [ ] B) `depthshade` converts the original colour to its complementary colour based on depth, causing originally-different colours to cross when at the same depth offset
- [ ] C) `depthshade=True` requires the `cmap` argument — without it, all points default to the same internal colour palette
- [ ] D) `depthshade` applies random dithering to prevent z-fighting, which can accidentally swap colours between nearby points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `depthshade` blends the point colour toward a uniform grey as depth increases, converging all colours toward the same grey at maximum depth

**Explanation:**
- **A is correct:** Matplotlib's `depthshade` implementation applies a grey-toned alpha overlay that becomes more opaque for points at greater rendering depth (farther from the camera). At maximum depth, the overlay can substantially desaturate and darken the original colour, pulling distinctly different colours (like steelblue and coral) toward a similar muted grey tone. This is why depth shading can visually merge colours that should be distinguishable.
- **B is incorrect:** `depthshade` does not compute complementary colours. It applies a linear alpha modulation using a grey colour mixed with the original, not a hue rotation or complement computation.
- **C is incorrect:** `depthshade` works independently of whether `cmap` is specified. It applies to whatever colours are assigned to the points, whether from a colormap or a literal colour string like `"steelblue"`.
- **D is incorrect:** `depthshade` is not related to z-fighting (the rendering artifact where two coplanar surfaces flicker). Z-fighting is a GPU rendering issue. `depthshade` is a visual encoding technique applied in Python before rasterization.

</details>

---

### Question 17

A researcher has 500 data points and wants each point sized proportionally to a feature `importance` (range 0–1), but also wants the minimum visible size to be 10 points² and maximum 200 points². Which sizing expression correctly maps `importance` to this range?

- [ ] A) `s = importance`
- [ ] B) `s = importance * 200`
- [ ] C) `s = importance * 190 + 10`
- [ ] D) `s = (importance - 0.5) * 200 + 100`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `s = importance * 190 + 10`

**Explanation:**
- **A is incorrect:** `s = importance` maps the range [0, 1] to point sizes [0, 1] points². Points with `importance=0` are invisible (0 points²) and all points are microscopic. The output range [0, 1] does not match the desired [10, 200].
- **B is incorrect:** `s = importance * 200` maps [0, 1] to [0, 200]. The minimum size is 0 (invisible for `importance=0`), not 10 as required.
- **C is correct:** Linear interpolation between a minimum of 10 and a maximum of 200 uses the formula `s_min + (s_max - s_min) * importance = 10 + 190 * importance`. When `importance=0`, `s = 10`; when `importance=1`, `s = 200`. This correctly maps the full [0, 1] input range to [10, 200] output range.
- **D is incorrect:** `s = (importance - 0.5) * 200 + 100` maps [0, 1] to: at `importance=0` → `(-0.5 * 200) + 100 = 0`; at `importance=1` → `(0.5 * 200) + 100 = 200`. The minimum is 0, not 10, and negative sizes for values below 0.05 would cause errors.

</details>

---

### Question 18

A data analyst plots a 3D scatter of sensor readings and adds a colorbar using:

```python
sc = ax.scatter(x, y, z, c=temperature, cmap="inferno")
cb = fig.colorbar(sc, ax=ax, pad=0.1)
```

What does the `pad=0.1` argument to `fig.colorbar` control?

- [ ] A) The number of discrete colour levels in the colorbar (10 levels)
- [ ] B) The fractional spacing between the edge of the 3D axes and the left edge of the colorbar
- [ ] C) The transparency of the colorbar at 10% opacity
- [ ] D) The tick label rotation in degrees (10° tilt)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The fractional spacing between the edge of the 3D axes and the left edge of the colorbar

**Explanation:**
- **A is incorrect:** The number of colour levels is controlled by `drawedges`, `boundaries`, or the colormap's internal resolution (`N`). `pad` has no relationship to level count.
- **B is correct:** `pad` is a fractional value (relative to the axes width) specifying the gap between the parent axes and the colorbar. The default is approximately 0.05 for vertical colorbars. Setting `pad=0.1` increases this gap to 10% of the axes width, pushing the colorbar further away from the 3D box — often necessary because 3D axes have more visual content extending toward the colorbar position than 2D axes.
- **C is incorrect:** Colorbar transparency is not controlled by `pad`. Opacity would be set via `colorbar.ax.patch.set_alpha(...)` or by making the parent scatter semi-transparent.
- **D is incorrect:** Tick label rotation is controlled via `colorbar.ax.tick_params(labelrotation=...)` or `colorbar.ax.set_yticklabels(..., rotation=...)`. The `pad` parameter is purely about positional spacing.

</details>

---

### Question 19

Which of the following `ax.scatter` calls would produce **four** visual encoding channels for a 3D dataset?

- [ ] A) `ax.scatter(x, y, z, c="blue", s=50)`
- [ ] B) `ax.scatter(x, y, z, c=feature_4, cmap="viridis", s=50, marker="^")`
- [ ] C) `ax.scatter(x, y, z, c=feature_4, cmap="viridis", s=feature_5 * 100)`
- [ ] D) `ax.scatter(x, y, z, c=feature_4, cmap="viridis", s=feature_5 * 100, alpha=feature_6)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.scatter(x, y, z, c=feature_4, cmap="viridis", s=feature_5 * 100)`

**Explanation:**
- **A is incorrect:** This encodes three channels: x, y, z. `c="blue"` is a constant colour (not a data channel) and `s=50` is a constant size. Only three data dimensions are encoded.
- **B is incorrect:** x, y, z = 3 spatial channels; `c=feature_4` = 1 colour channel; `marker="^"` is a constant visual attribute (not a data variable). Total = 4 channels, but `marker` is not a variable encoding — wait, this actually also has 4 channels with x, y, z, feature_4. Let's re-examine: x, y, z (3) + feature_4 via colour (1) = 4 channels. However, option C is more directly correct as it adds both colour AND size as independent data variables, giving x, y, z, feature_4, feature_5 = 5 channels. The question asks for **four**, making option B valid too. But C is MORE than 4. Let me re-read — "four" encoding channels. Option B has exactly 4 (x, y, z, feature_4). Option C has 5. Option D also has more.
- **C is incorrect in context:** C encodes x(1), y(2), z(3), feature_4-colour(4), feature_5-size(5) = 5 encoding channels, which is more than four. However, since the question asks which produces FOUR channels and C produces FIVE, B is the correct answer for exactly four. But reviewing B again: x, y, z, feature_4 = 4 channels, with marker=constant. B gives exactly 4.

*(Note: The intended correct answer is C for the purpose of this question as the richest encoding. Let's verify: x,y,z=3, c=feature_4=1, s=feature_5=1 → 5 channels. The question says "four". B gives x,y,z=3, c=feature_4=1 → 4. So B is correct for exactly four.)*

**Corrected Answer:** B

- **B is correct:** x, y, z encode three spatial dimensions; `c=feature_4` with `cmap="viridis"` adds a fourth continuous colour encoding dimension. `s=50` is constant, not a variable, and `marker="^"` is a constant visual attribute. Total variable data channels = 4.
- **C is incorrect (for this question):** C encodes five dimensions — x, y, z (3 spatial) + feature_4 via colour + feature_5 via size = 5 channels total.
- **D is incorrect:** D encodes six dimensions — x, y, z (3) + colour + size + alpha = 6 channels (if alpha is per-point), which exceeds four.

</details>

---

### Question 20

A data scientist creates two 3D scatter plots side-by-side and wants them to share the same colour scale for direct comparison. She uses:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
sc1 = ax1.scatter(x1, y1, z1, c=vals1, cmap="viridis", vmin=0, vmax=10)
sc2 = ax2.scatter(x2, y2, z2, c=vals2, cmap="viridis", vmin=0, vmax=10)
fig.colorbar(sc1, ax=[ax1, ax2])
```

What is the purpose of `vmin=0, vmax=10` in both scatter calls?

- [ ] A) To clip the data so that values outside [0, 10] are removed from the scatter before rendering
- [ ] B) To pin the colormap's mapping range to [0, 10] regardless of the actual min/max values in `vals1` or `vals2`, ensuring both scatters use the same colour scale
- [ ] C) To set the z-axis limits for both subplots simultaneously
- [ ] D) To control the marker size range, mapping the smallest marker to 0 and the largest to 10

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) To pin the colormap's mapping range to [0, 10] regardless of the actual min/max values in `vals1` or `vals2`, ensuring both scatters use the same colour scale

**Explanation:**
- **A is incorrect:** `vmin`/`vmax` do not filter or clip data points. Points with `vals` outside the range [0, 10] are still rendered, but their colour is saturated to the lowest or highest colormap colour.
- **B is correct:** `vmin` and `vmax` set the normalization boundaries for the colour mapping. With `vmin=0, vmax=10`, the colormap range [0, 1] is mapped to the data range [0, 10] in both scatter objects. This guarantees that the same data value produces the same colour in both subplots, enabling direct visual comparison. Without explicit `vmin`/`vmax`, each scatter normalizes independently to its own `min(vals)` and `max(vals)`, which would make colour comparisons across subplots meaningless.
- **C is incorrect:** z-axis limits are set via `ax.set_zlim(zmin, zmax)`. `vmin`/`vmax` are scatter colour-mapping parameters and have no relationship to spatial axis limits.
- **D is incorrect:** Marker size is set via the `s` parameter. `vmin`/`vmax` are part of the colour normalization (`Normalize`) machinery, not the size encoding.

</details>
