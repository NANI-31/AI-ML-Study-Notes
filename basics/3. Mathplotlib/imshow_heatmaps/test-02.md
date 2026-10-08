# Matplotlib Imshow & Heatmaps — Test 02
## Dynamic Range Scaling (vmin, vmax)

---

### Question 11

A developer creates a side-by-side comparison of two model attention maps:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

attn_a = np.random.uniform(0.1, 0.9, (8, 8))   # Attention map A
attn_b = np.random.uniform(0.3, 0.7, (8, 8))   # Attention map B — narrower range

ax1.imshow(attn_a, cmap='Reds')
ax2.imshow(attn_b, cmap='Reds')
ax1.set_title("Model A Attention")
ax2.set_title("Model B Attention")
plt.show()
```

A colleague reviews the figure and concludes "Model B's attention is just as spread across extremes as Model A." Why is this conclusion wrong?

- [ ] A) The `Reds` colormap distorts attention values above 0.5
- [ ] B) Without locking `vmin` and `vmax`, each subplot auto-scales independently — attn_b's narrower [0.3, 0.7] range is stretched to fill the same full colormap as attn_a's [0.1, 0.9] range, making them falsely appear equal in contrast
- [ ] C) The `imshow` function averages both arrays before display when used in subplots
- [ ] D) Model B's attention map has fewer pixels than Model A, causing the visual distortion

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Without locking `vmin` and `vmax`, each subplot auto-scales independently — attn_b's narrower [0.3, 0.7] range is stretched to fill the same full colormap as attn_a's [0.1, 0.9] range, making them falsely appear equal in contrast

**Explanation:**

- **A is incorrect:** The `Reds` colormap maps scalar values linearly — it does not distort values above 0.5. All values are mapped proportionally through the colormap.
- **B is correct:** Without explicit `vmin`/`vmax`, each `imshow` call independently maps its own data range to the full colormap. `attn_a` maps [0.1, 0.9] → full red spectrum; `attn_b` maps [0.3, 0.7] → the same full red spectrum. A value of 0.3 in Model B appears as dark red (low attention) — yet a value of 0.3 in Model A would appear as a light red (moderate-low attention). This makes Model B appear to have the same contrast range as Model A — a dishonest comparison. Fix: `vmin=0.0, vmax=1.0` on both.
- **C is incorrect:** `imshow` renders each subplot's data independently. There is no inter-subplot averaging.
- **D is incorrect:** Both arrays are `(8, 8)` — same number of pixels. Array size has no role in causing the visual distortion described.

</details>

---

### Question 12

A developer fixes the attention map comparison by adding `vmin=0.0, vmax=1.0` to both subplots. A new colleague asks: "Why not just use `vmin=attn_a.min(), vmax=attn_a.max()` as the shared range?" What is the problem with this approach?

- [ ] A) No problem — using Model A's range as the reference is always correct for comparison
- [ ] B) If attn_b has values outside attn_a's range, those values will be clipped — any attention values in B below attn_a.min() appear at the colormap minimum, and values above attn_a.max() appear at the maximum, losing detail at the extremes
- [ ] C) `attn_a.min()` and `attn_a.max()` are float values — they cannot be passed to `vmin`/`vmax` which only accept integers
- [ ] D) Matplotlib ignores `vmin`/`vmax` when the values are derived from a different array than the one being displayed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) If attn_b has values outside attn_a's range, those values will be clipped — any attention values in B below attn_a.min() appear at the colormap minimum, and values above attn_a.max() appear at the maximum, losing detail at the extremes

**Explanation:**

- **A is incorrect:** Using one dataset's range as the reference is not "always correct." It only works when the second dataset is guaranteed to be within the first's range. If attn_b has values outside [attn_a.min(), attn_a.max()], information is lost through clipping.
- **B is correct:** `vmin`/`vmax` act as **hard clips**. Any data value below `vmin` is displayed at the colormap minimum color; any value above `vmax` is displayed at the colormap maximum color. If attn_b has values below attn_a.min() or above attn_a.max(), those values are clipped — their precise values become indistinguishable. The correct approach is to compute the global range: `vmin=min(attn_a.min(), attn_b.min())`, `vmax=max(attn_a.max(), attn_b.max())`.
- **C is incorrect:** `vmin` and `vmax` accept any Python numeric type — floats, integers, or NumPy scalars. Float values are fully supported and most common.
- **D is incorrect:** Matplotlib does not track which array a `vmin`/`vmax` was derived from. These are simply numeric bounds applied to the normalization — their origin is irrelevant.

</details>

---

### Question 13

A developer creates a heatmap of neural network weight magnitudes:

```python
weights = np.random.randn(16, 16) * 0.01  # Small initialization values

im = ax.imshow(weights, cmap='RdBu', vmin=-1, vmax=1)
fig.colorbar(im)
```

The colormap range is `[-1, 1]` but actual data only spans `[-0.03, 0.03]`. What does the heatmap look like?

- [ ] A) A vivid red-blue pattern showing the weight differences clearly
- [ ] B) An almost entirely neutral gray/white image — all values are near zero relative to the [-1, 1] range, so all pixels map to the colormap's center color
- [ ] C) The colormap auto-adjusts to [-0.03, 0.03] despite the `vmin`/`vmax` specification
- [ ] D) A `ValueError` is raised because data range is smaller than the specified `vmin`/`vmax` range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An almost entirely neutral gray/white image — all values are near zero relative to the [-1, 1] range, so all pixels map to the colormap's center color

**Explanation:**

- **A is incorrect:** A vivid red-blue pattern would only appear if the data spanned a significant fraction of the `vmin`/`vmax` range. Values in [-0.03, 0.03] are only 3% of the total [-1, 1] range — far too small to show visible color variation.
- **B is correct:** With `vmin=-1, vmax=1`, the colormap maps -1 → pure red and +1 → pure blue, with 0 → neutral center. All weight values near 0 (within ±0.03) map to colors extremely close to the neutral center — appearing as a nearly uniform gray/white image. This is why setting `vmin`/`vmax` appropriately to the **actual data range** matters: here `vmin=-0.05, vmax=0.05` would reveal the true weight structure.
- **C is incorrect:** When `vmin`/`vmax` are explicitly specified, Matplotlib **never** overrides them with auto-scaling. The specified bounds are used exactly as provided.
- **D is incorrect:** No error is raised when data values fall within a specified `vmin`/`vmax` range that is wider than the data. This is perfectly valid behavior — data values between vmin and vmax are displayed normally; values outside would be clipped.

</details>

---

### Question 14

A developer uses `vmin` and `vmax` to create a **locked colorbar** across 4 subplots in a hyperparameter search grid:

```python
results = [mat1, mat2, mat3, mat4]  # F1 scores, each (5, 5)
global_min = min(m.min() for m in results)
global_max = max(m.max() for m in results)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, mat in zip(axes.flatten(), results):
    im = ax.imshow(mat, cmap='YlOrRd',
                   vmin=global_min, vmax=global_max)

fig.colorbar(im, ax=axes.ravel().tolist())
```

What does `fig.colorbar(im, ax=axes.ravel().tolist())` accomplish that `fig.colorbar(im, ax=axes[0,0])` would not?

- [ ] A) It creates 4 separate colorbars, one per subplot
- [ ] B) It creates a single shared colorbar that resizes itself by stealing space equally from all 4 subplots, rather than only from the last subplot or one subplot
- [ ] C) It applies the colormap to all subplots simultaneously
- [ ] D) It locks `vmin`/`vmax` to be consistent across all subplots

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It creates a single shared colorbar that resizes itself by stealing space equally from all 4 subplots, rather than only from the last subplot or one subplot

**Explanation:**

- **A is incorrect:** Passing a list of axes to `fig.colorbar()` creates a **single** colorbar, not one per subplot. To create 4 colorbars, `fig.colorbar()` would need to be called 4 times.
- **B is correct:** When `ax=axes.ravel().tolist()` (a list of all axes) is passed, Matplotlib's colorbar machinery identifies all the involved subplots and steals space proportionally from all of them to place the single colorbar. This produces a taller/bigger colorbar that spans the full height of the 2×2 grid and looks visually balanced. If only `ax=axes[0,0]` were used, space would only be stolen from the top-left subplot, making it noticeably smaller than the others.
- **C is incorrect:** The colormap is applied at `ax.imshow()` call time, not at `fig.colorbar()` time. `fig.colorbar()` only creates the colorbar legend — it doesn't re-apply or change any colormap.
- **D is incorrect:** `vmin`/`vmax` locking is achieved by the explicit `vmin=global_min, vmax=global_max` in each `ax.imshow()` call — not by `fig.colorbar()`. The colorbar function has no role in normalizing the data.

</details>

---

### Question 15

A developer computes: with `vmin=20` and `vmax=80`, what **normalized value** does a raw data value of `50` receive in the colormap (assuming linear normalization)?

- [ ] A) 0.333
- [ ] B) 0.500
- [ ] C) 0.625
- [ ] D) 0.750

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.500

**Explanation:**

- **A is incorrect:** 0.333 would correspond to `(50 - 20) / (80 - 20) = 30/60 = 0.5` — wait, that IS 0.5. Let me recalculate: `(value - vmin) / (vmax - vmin) = (50 - 20) / (80 - 20) = 30 / 60 = 0.5`.
- **B is correct:** Matplotlib's linear normalization formula is: `normalized = (value - vmin) / (vmax - vmin)`. For `value=50, vmin=20, vmax=80`: `normalized = (50 - 20) / (80 - 20) = 30 / 60 = 0.500`. This normalized value of 0.5 means the data point maps to the exact **center** of the colormap. For diverging colormaps (like `coolwarm`), the center is the neutral color (white/gray). For sequential colormaps, it is the midpoint color.
- **C is incorrect:** 0.625 would result from `(50 - 20) / (80 - 20)` only if `vmin=20` and `vmax=68`. With `vmax=80`: `(50-20)/(80-20) = 30/60 ≠ 0.625`.
- **D is incorrect:** 0.750 would result from `(50 - 20) / (80 - 60)` — but `vmax=80` not 60. With the given values: `30/60 = 0.5`.

</details>

---

### Question 16

A developer wants to highlight only the **top 10% of values** in a heatmap as the "hot" region and make all other values appear neutral:

```python
data = np.random.rand(20, 20)
p90 = np.percentile(data, 90)

im = ax.imshow(data, cmap='hot', vmin=p90, vmax=data.max())
```

What does this `vmin=p90` setting accomplish?

- [ ] A) Values below p90 are hidden (invisible/transparent)
- [ ] B) Values below p90 are all displayed at the colormap minimum color (black for `hot`) — only the top 10% of values show the full hot color gradient
- [ ] C) The colormap is clipped to show only values in [p90, max] with the remaining range being white
- [ ] D) An error is raised because `vmin` must be less than the data minimum

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Values below p90 are all displayed at the colormap minimum color (black for `hot`) — only the top 10% of values show the full hot color gradient

**Explanation:**

- **A is incorrect:** `vmin`/`vmax` do not make values invisible or transparent. Transparency requires RGBA arrays with alpha channels. Values below `vmin` are clipped to the colormap's minimum color — they remain visible.
- **B is correct:** Setting `vmin=p90` clips all values below the 90th percentile to the colormap minimum. For the `hot` colormap, the minimum color is black. The color gradient (black → red → orange → yellow → white) is then applied only to the top 10% of values (from p90 to data.max()). This effectively highlights only the highest-intensity region with colorful representation while flattening everything below to black.
- **C is incorrect:** The `hot` colormap minimum is black, not white. White is the maximum of the `hot` colormap. Values below `vmin` map to the minimum color (black), not white.
- **D is incorrect:** `vmin` does not need to be less than the data minimum. Setting `vmin` above the data minimum is a valid and intentional technique for selective highlighting. No error is raised.

</details>

---

### Question 17

A developer creates a before/after comparison of an image denoising model:

```python
noisy  = original + np.random.normal(0, 30, original.shape)
clean  = denoising_model(noisy)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(original, cmap='gray')
axes[1].imshow(noisy, cmap='gray')
axes[2].imshow(clean, cmap='gray')
```

`original` has range [0, 255], `noisy` has range [-45, 310] due to noise, `clean` has range [2, 251]. Without `vmin`/`vmax`, what visual distortion occurs in the comparison?

- [ ] A) All three images appear identical because `gray` normalizes everything uniformly
- [ ] B) The noisy image auto-scales to [-45, 310] — making pixels at 0 appear mid-gray and pixels at 255 appear as a slightly lighter gray, compressing the apparent contrast compared to original and clean
- [ ] C) The noisy image clips all negative values to black and values above 255 to white automatically
- [ ] D) No distortion occurs — auto-scaling makes all three images visually accurate for comparison

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The noisy image auto-scales to [-45, 310] — making pixels at 0 appear mid-gray and pixels at 255 appear as a slightly lighter gray, compressing the apparent contrast compared to original and clean

**Explanation:**

- **A is incorrect:** Each `imshow` call independently auto-scales to its own data range. There is no uniform normalization across subplots.
- **B is correct:** Auto-scaling for each image: `original` maps [0,255] → full gray range; `noisy` maps [-45, 310] → full gray range. In the `noisy` image, the value 0 maps to `(0-(-45))/(310-(-45)) = 45/355 ≈ 0.127` (dark gray, not black), and 255 maps to `(255+45)/355 ≈ 0.845` (light gray, not white). This compresses the perceived contrast — the noisy image appears washed out compared to the original and clean images. The fix: `vmin=0, vmax=255` on all three subplots.
- **C is incorrect:** Auto-scaling does NOT clip to [0, 255]. It maps the actual data range (including negative values and values above 255) to the full colormap. Clipping would only occur if explicit `vmin=0, vmax=255` were set.
- **D is incorrect:** Auto-scaling explicitly causes distortion in multi-panel comparisons because each panel uses a different data range as its reference.

</details>

---

### Question 18

A developer uses `Normalize` from Matplotlib to create a **two-slope normalization** for a diverging metric:

```python
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

data = np.array([[-5, -2, 0], [1, 8, 15], [0, -1, 3]])
norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=15)

im = ax.imshow(data, cmap='RdBu_r', norm=norm)
fig.colorbar(im)
```

What does `TwoSlopeNorm(vmin=-5, vcenter=0, vmax=15)` accomplish that standard `vmin=-5, vmax=15` does NOT?

- [ ] A) It applies two separate colormaps (one for positive, one for negative)
- [ ] B) It ensures the colormap center (neutral color) aligns exactly with `vcenter=0`, even though the negative range [-5, 0] and positive range [0, 15] have different widths
- [ ] C) It clips all values between -5 and 15 to exactly 0
- [ ] D) It creates a logarithmic scale between vmin and vmax

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It ensures the colormap center (neutral color) aligns exactly with `vcenter=0`, even though the negative range [-5, 0] and positive range [0, 15] have different widths

**Explanation:**

- **A is incorrect:** `TwoSlopeNorm` uses a single colormap — it does not apply separate colormaps to positive and negative ranges. The colormap (e.g., `RdBu_r`) is applied continuously but with different compression rates on each side of vcenter.
- **B is correct:** With standard linear normalization (`vmin=-5, vmax=15`): the neutral center of the colormap maps to `(-5+15)/2 = 5`, not 0. This means the value 0 would appear slightly blue (above center), misrepresenting it as mildly positive. `TwoSlopeNorm` splits the normalization at `vcenter=0`: the left half of the colormap maps [-5, 0] and the right half maps [0, 15]. Zero always maps to the exact colormap center (neutral white/gray in `RdBu_r`), regardless of the asymmetry in range widths.
- **C is incorrect:** `TwoSlopeNorm` does not clip or collapse values. It remaps the normalization function while preserving all values' distinctions within [vmin, vmax].
- **D is incorrect:** `TwoSlopeNorm` is a piecewise linear normalization, not logarithmic. For log scaling, `LogNorm` from `matplotlib.colors` would be used.

</details>

---

### Question 19

A developer displays a heatmap and observes that the **colorbar does not match the expected range**. They inspect:

```python
data = np.random.uniform(-3, 3, (10, 10))
im = ax.imshow(data, cmap='bwr', vmin=-2, vmax=2)
cbar = fig.colorbar(im)
```

What colorbar tick range will be shown, and what happens to data values of -3 and +3?

- [ ] A) Colorbar shows [-3, 3]; the data values -3 and +3 are shown at the extremes
- [ ] B) Colorbar shows [-2, 2]; data values below -2 appear at the pure blue extreme; values above +2 appear at the pure red extreme — both are clipped
- [ ] C) Colorbar shows [-2, 2]; data values outside [-2, 2] are hidden (transparent)
- [ ] D) Colorbar shows [-3, 3] because `imshow` overrides the specified `vmin`/`vmax` with actual data range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Colorbar shows [-2, 2]; data values below -2 appear at the pure blue extreme; values above +2 appear at the pure red extreme — both are clipped

**Explanation:**

- **A is incorrect:** The colorbar reflects the normalization range set by `vmin`/`vmax` — which is [-2, 2]. The actual data range [-3, 3] is wider but the colorbar does not auto-expand to match it.
- **B is correct:** `vmin=-2, vmax=2` locks the colormap range. The colorbar inherits this range and shows ticks from -2 to +2. Data values of -3 and +3 are **clipped**: -3 maps to the same pure blue as -2; +3 maps to the same pure red as +2. This clipping is visible as the colorbar edges being "saturated" — a common indicator of clipped data. To show clipped data as a special color (e.g., dark gray for out-of-range), use `cmap.set_under()` and `cmap.set_over()`.
- **C is incorrect:** Clipped values are not transparent — they are displayed at the colormap extreme colors (pure blue/red for `bwr`). Transparency requires alpha channel manipulation.
- **D is incorrect:** Matplotlib does NOT override explicitly specified `vmin`/`vmax` with auto-scaling. When `vmin` and `vmax` are provided, they are used exactly as specified, regardless of the actual data range.

</details>

---

### Question 20

A developer wants the colorbar to show out-of-range values with special indicator colors (triangular arrows at colorbar ends). Which code achieves this?

- [ ] A) `fig.colorbar(im, extend='both')` — adds triangular arrows at both ends of the colorbar indicating clipped values above vmax and below vmin
- [ ] B) `fig.colorbar(im, clip=True)` — enables clipping indicator arrows
- [ ] C) `ax.imshow(data, vmin=vmin, vmax=vmax, extend='both')` — the `extend` parameter goes in `imshow`
- [ ] D) `cbar.set_overflow_arrows(True)` — a dedicated method for overflow indicators

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `fig.colorbar(im, extend='both')` — adds triangular arrows at both ends of the colorbar indicating clipped values above vmax and below vmin

**Explanation:**

- **A is correct:** `fig.colorbar(im, extend='both')` adds triangular **extension caps** at both ends of the colorbar: a downward-pointing triangle at the minimum end (indicating values below `vmin` exist and are clipped) and an upward-pointing triangle at the maximum end (for values above `vmax`). Options include `extend='min'`, `extend='max'`, or `extend='both'`. The extension triangles are colored using `cmap.set_under(color)` and `cmap.set_over(color)` (or the colormap's default extremes).
- **B is incorrect:** `clip=True` is not a parameter of `fig.colorbar()`. There is no such argument. Clipping already happens implicitly when `vmin`/`vmax` are set — no flag is needed to enable it.
- **C is incorrect:** `extend='both'` is a parameter of `fig.colorbar()` — NOT of `ax.imshow()`. The `imshow` function does not accept an `extend` parameter.
- **D is incorrect:** `cbar.set_overflow_arrows()` is not a real Matplotlib method. The actual API uses `fig.colorbar(im, extend='both')` at creation time, or `cbar.set_extend('both')` after creation.

</details>
