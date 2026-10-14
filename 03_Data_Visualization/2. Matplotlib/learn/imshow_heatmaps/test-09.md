# Matplotlib Imshow & Heatmaps — Test 09
## Feature Correlation & Distance Matrices

---

### Question 81

A developer creates a **Pearson correlation heatmap** for 6 features using `imshow`:

```python
corr = np.corrcoef(feature_matrix.T)  # Shape: (6, 6)
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1,
               interpolation='nearest', aspect='equal')
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.set_yticklabels(feature_names)
```

`np.corrcoef(feature_matrix.T)` receives `feature_matrix.T` (transposed). If `feature_matrix.shape = (1000, 6)`, what is the shape of the transposed input and the output correlation matrix?

- [ ] A) Input: `(1000, 6)` → Output: `(1000, 1000)` — correlations between samples
- [ ] B) Input: `(6, 1000)` → Output: `(6, 6)` — correlations between the 6 features
- [ ] C) Input: `(6, 1000)` → Output: `(6, 1000)` — same shape as input
- [ ] D) Input: `(1000, 6)` → Output: `(6, 6)` — transpose happens inside `corrcoef`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Input: `(6, 1000)` → Output: `(6, 6)` — correlations between the 6 features

**Explanation:**

- **A is incorrect:** Passing `feature_matrix.T` (transposed) means `np.corrcoef()` receives a `(6, 1000)` array — rows are features, columns are samples. `corrcoef` treats each **row** as a variable and computes pairwise correlations, producing a `(6, 6)` matrix.
- **B is correct:** `feature_matrix.shape = (1000, 6)` — 1000 samples, 6 features. `feature_matrix.T.shape = (6, 1000)` — transposed so 6 rows (features), 1000 columns (samples). `np.corrcoef(X)` computes the correlation matrix where each row of X is a variable. With a `(6, 1000)` input: `corrcoef` returns a `(6, 6)` symmetric matrix where `corr[i, j]` is the Pearson correlation between feature i and feature j across the 1000 samples. This is the desired feature-feature correlation matrix.
- **C is incorrect:** `corrcoef` always returns a square matrix (N×N where N = number of variables/rows in input). The output shape `(6, 1000)` would never occur.
- **D is incorrect:** `feature_matrix` (without `.T`) has shape `(1000, 6)`. Passing it to `corrcoef` would compute correlations between 1000 "variables" (samples), producing a `(1000, 1000)` matrix — not the intended feature correlation matrix.

</details>

---

### Question 82

A developer creates a correlation heatmap and notices the **diagonal** values are all `1.0`. They consider removing the diagonal for visual clarity using:

```python
np.fill_diagonal(corr, np.nan)
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
```

What is the visual effect of setting diagonal values to `NaN` in the imshow array?

- [ ] A) Diagonal cells appear white (the default NaN color in Matplotlib)
- [ ] B) An error is raised — `imshow` cannot handle NaN values
- [ ] C) Diagonal cells appear as the colormap's "bad" color (default: transparent/white depending on the colormap's `set_bad()` setting)
- [ ] D) Diagonal cells show the midpoint color (neutral gray/white for `coolwarm`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Diagonal cells appear as the colormap's "bad" color (default: transparent/white depending on the colormap's `set_bad()` setting)

**Explanation:**

- **A is incorrect:** NaN cells do not automatically appear white. They receive the colormap's "bad" color, which varies by colormap. For most colormaps, the default "bad" color is transparent (appears as the axes background). For some, it defaults to a specific gray.
- **B is incorrect:** `imshow` handles NaN values gracefully — no error is raised. NaN values are recognized as masked/invalid and rendered using the colormap's bad color.
- **C is correct:** Matplotlib colormaps have a special "bad" color for masked/NaN values, set via `cmap.set_bad(color)`. By default, most colormaps use `'white'` or transparent for the bad color. `np.fill_diagonal(corr, np.nan)` sets diagonal cells to NaN, which `imshow` renders using this bad color — effectively masking the diagonal. For `coolwarm`, the default bad color is transparent, making diagonal cells appear as the axes background (typically white).
- **D is incorrect:** The midpoint of `coolwarm` is white/neutral — which LOOKS like the bad/NaN rendering for some colormaps. But this is coincidental: NaN values do NOT map through the colormap normalization; they bypass it and directly use the bad color. Setting `vmin=-1, vmax=1` maps 0.0 to the midpoint — but NaN is not 0.0.

</details>

---

### Question 83

A developer creates a **pairwise cosine distance matrix** (values in [0, 2]) for document embeddings and must choose the correct colormap and vmin/vmax:

```python
# cosine_dist[i,j] = 1 - cosine_similarity(doc_i, doc_j)
# Values: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite
cosine_dist = pairwise_distances(embeddings, metric='cosine')

im = ax.imshow(cosine_dist, cmap=???, vmin=0, vmax=2,
               interpolation='nearest', aspect='equal')
```

Which colormap is semantically most appropriate for this distance matrix?

- [ ] A) `coolwarm` — diverging colormap with neutral center at 1.0 (orthogonal)
- [ ] B) `YlOrRd` — sequential colormap: white/yellow (similar docs) to dark red (dissimilar docs)
- [ ] C) `tab10` — qualitative colormap for categorical distance values
- [ ] D) `bwr` — diverging colormap showing blue for low distances and red for high distances

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `YlOrRd` — sequential colormap: white/yellow (similar docs) to dark red (dissimilar docs)

**Explanation:**

- **A is incorrect:** `coolwarm` (diverging) implies a meaningful bipolar structure with a neutral midpoint. Cosine distance [0, 2] has no inherent "neutral" concept — 1.0 (orthogonal) is not a meaningful midpoint between "similar" and "dissimilar." Diverging colormaps require data with a semantically meaningful center (like correlations through zero, or residuals around zero).
- **B is correct:** Cosine distance [0, 2] is **unipolar**: 0 = most similar, 2 = most dissimilar (no sign change, no neutral center). A **sequential** colormap is correct: light color (yellow/white) for small distances (similar documents) and dark/intense color (dark red) for large distances (dissimilar documents). `YlOrRd` (yellow-orange-red) or `Blues` are appropriate. The sequential mapping correctly encodes "more distance = more intense color."
- **C is incorrect:** Distance values are continuous in [0, 2] — not categorical. A qualitative colormap would assign arbitrary distinct colors to different distance magnitudes, making magnitude comparison impossible.
- **D is incorrect:** `bwr` (blue-white-red) is diverging — blue for low, red for high. While the color direction (blue=similar, red=dissimilar) is intuitive, the diverging structure implies a neutral white midpoint at 1.0 — suggesting "orthogonal" is a neutral state. For distance matrices, the semantics favor sequential colormaps that consistently show "more = darker/more intense."

</details>

---

### Question 84

A developer adds **feature labels** to both x and y axes of a correlation heatmap. The x-tick labels appear overlapping horizontally. Which code correctly rotates them to 45° with right-alignment?

```python
feature_names = ['age', 'income', 'credit_score', 'loan_amt', 'employment_yrs', 'debt_ratio']
ax.set_xticks(range(6))
# Fix overlapping x-tick labels:
```

- [ ] A) `ax.set_xticklabels(feature_names, rotation=45)`
- [ ] B) `ax.set_xticklabels(feature_names, rotation=45, ha='right')`
- [ ] C) `ax.xaxis.set_tick_params(rotation=45)` — no need for `ha`
- [ ] D) `plt.xticks(rotation=45, ha='center')` — center alignment with 45° rotation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_xticklabels(feature_names, rotation=45, ha='right')`

**Explanation:**

- **A is incorrect:** `rotation=45` rotates the labels but without `ha='right'`, the rotation pivot is the label center. At 45° with center alignment, labels drift visually away from their tick positions — the label text doesn't align cleanly below the tick mark.
- **B is correct:** `rotation=45, ha='right'` rotates labels 45° counterclockwise AND aligns the RIGHT edge of each label with the tick position. This makes the tick pointer appear to point to the right end of the label text — producing the clean diagonal label arrangement standard in heatmaps. The combination of `rotation=45` + `ha='right'` is the most common and visually correct way to display diagonal x-axis labels in Matplotlib.
- **C is incorrect:** `ax.xaxis.set_tick_params(rotation=45)` does set rotation, but without the `ha='right'` horizontal alignment correction. The labels will be rotated but not properly aligned with their tick marks. Also, the method is `ax.tick_params(axis='x', rotation=45)`, not `set_tick_params`.
- **D is incorrect:** `ha='center'` with 45° rotation centers the label horizontally at the tick mark — producing the "drifting" misalignment problem. For 45° labels, `ha='right'` is the correct alignment choice.

</details>

---

### Question 85

A developer creates a symmetric correlation heatmap and wants to display only the **lower triangle** to avoid redundancy:

```python
corr = np.corrcoef(data.T)  # 8×8 symmetric matrix
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Upper triangle mask
corr_masked = np.where(mask, np.nan, corr)

im = ax.imshow(corr_masked, cmap='coolwarm', vmin=-1, vmax=1)
```

`np.triu(np.ones_like(corr, dtype=bool), k=1)` — what does `k=1` specify?

- [ ] A) `k=1` includes the main diagonal in the mask — masking the diagonal and everything above it
- [ ] B) `k=1` excludes the main diagonal from the mask — masking only the elements ABOVE the diagonal (upper triangle without the diagonal itself)
- [ ] C) `k=1` masks every other diagonal above the main diagonal
- [ ] D) `k=1` specifies a 1-unit offset in pixel coordinates for the mask boundary

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `k=1` excludes the main diagonal from the mask — masking only the elements ABOVE the diagonal (upper triangle without the diagonal itself)

**Explanation:**

- **A is incorrect:** `k=1` in `np.triu()` starts the upper triangle **one above** the main diagonal. The main diagonal (k=0) is NOT masked. `k=0` would include the diagonal in the upper triangle mask.
- **B is correct:** `np.triu(ones, k=0)` returns the upper triangle INCLUDING the main diagonal. `np.triu(ones, k=1)` returns the upper triangle **excluding** the main diagonal (starting one position above the diagonal). So `corr_masked` retains the lower triangle AND the main diagonal (all 1.0s), while setting the upper triangle to NaN. This produces a heatmap showing the full lower triangle with diagonal values visible — the most common half-matrix visualization for correlation matrices.
- **C is incorrect:** `k=1` does not skip alternate diagonals. It simply shifts the starting diagonal of the upper triangle by 1 position — affecting one contiguous region, not alternating diagonals.
- **D is incorrect:** `k` in NumPy tri/triu/tril functions refers to **diagonal offsets** in the array index space — not pixel coordinates. `k=0` is the main diagonal, `k=1` is one above, `k=-1` is one below.

</details>

---

### Question 86

A developer creates a **Jaccard similarity matrix** for binary features (values in [0, 1]) and annotates it:

```python
n = 8
jacc = compute_jaccard_matrix(binary_features)  # (8, 8), values in [0, 1]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(jacc, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')

for i in range(n):
    for j in range(n):
        color = 'white' if jacc[i, j] > 0.5 else 'black'
        ax.text(j, i, f"{jacc[i,j]:.2f}", ha='center', va='center',
                color=color, fontsize=8)

ax.set_title("Jaccard Similarity Matrix")
fig.colorbar(im, ax=ax, shrink=0.85)
plt.tight_layout()
```

The threshold `jacc[i,j] > 0.5` for white text is chosen because `Greens` colormap becomes **dark green** above 0.5. Is this threshold universally correct for all sequential colormaps?

- [ ] A) Yes — all sequential colormaps transition from light to dark at exactly the 0.5 normalized value
- [ ] B) No — the luminance crossover point varies by colormap. For `Blues`, it is around 0.5–0.6; for `hot`, the background remains dark from 0.0 onwards. The correct approach is to compute perceived luminance from the RGBA colormap values
- [ ] C) Yes — Matplotlib guarantees all colormaps have equal luminance at the normalized midpoint
- [ ] D) No — the threshold should always be 0.3 for all colormaps regardless of the data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — the luminance crossover point varies by colormap. For `Blues`, it is around 0.5–0.6; for `hot`, the background remains dark from 0.0 onwards. The correct approach is to compute perceived luminance from the RGBA colormap values

**Explanation:**

- **A is incorrect:** Sequential colormaps do NOT all cross from light-to-dark at exactly 0.5. The luminance progression differs significantly by colormap — some start dark (like `hot` which begins black at 0), others start very light.
- **B is correct:** The luminance transition point is colormap-specific. For `Greens` (light green → dark green), 0.5 is a reasonable threshold. For `hot` (black → red → yellow → white), the background is DARK from 0.0 to ~0.5 — requiring WHITE text for low values (opposite of Greens). The most robust approach: `rgba = cmap(norm_val)`, then compute luminance = `0.299*R + 0.587*G + 0.114*B` (relative luminance formula). Use white text when luminance < 0.5, black text when luminance ≥ 0.5.
- **C is incorrect:** Matplotlib does NOT guarantee uniform luminance at the normalized midpoint. Perceptually uniform colormaps (`viridis`, `plasma`) have monotone luminance by design, but the crossover point is not mandated at 0.5.
- **D is incorrect:** A universal threshold of 0.3 is incorrect for many colormaps. Some colormaps are already dark below 0.3, others are still light above 0.3. No single universal threshold applies across all colormaps.

</details>

---

### Question 87

A developer creates a heatmap for a **pairwise Euclidean distance matrix** where the diagonal should be 0 (self-distance). They notice the diagonal appears as a light color (near-minimum) in `YlOrRd`, while off-diagonal large values appear dark red. What does this correctly indicate about the matrix semantics?

- [ ] A) The colormap is incorrectly applied — diagonal should appear dark (maximum color)
- [ ] B) The colormap correctly encodes semantic distance: 0 (diagonal = self-distance = yellow/light = "no distance"), large values (dark red = "far apart") — the visualization is semantically correct
- [ ] C) `YlOrRd` reverses distance semantics — 0 should be dark and maximum should be light
- [ ] D) The diagonal values are being overwritten by `imshow` with NaN during rendering

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The colormap correctly encodes semantic distance: 0 (diagonal = self-distance = yellow/light = "no distance"), large values (dark red = "far apart") — the visualization is semantically correct

**Explanation:**

- **A is incorrect:** The colormap is applied correctly. Diagonal = 0 (minimum distance) should appear as the minimum color (light yellow in `YlOrRd`). This is accurate — a point is zero distance from itself.
- **B is correct:** `YlOrRd` maps: minimum value → yellow/light (no distance), maximum value → dark red (maximum distance). The diagonal of a Euclidean distance matrix is always 0 (self-distance), which correctly maps to the lightest color. Large off-diagonal values indicate points that are far apart — correctly mapped to dark red. This encoding is semantically correct: **light = similar/close, dark = dissimilar/far**. This is the standard distance matrix visualization.
- **C is incorrect:** `YlOrRd` does NOT reverse distance semantics. Light = small distance = similar = intuitively "nearby" (correct). The suggestion that 0 should appear dark contradicts distance visualization conventions.
- **D is incorrect:** `imshow` does not overwrite any array values. The underlying NumPy array is read-only within `imshow`. No values are modified during rendering.

</details>

---

### Question 88

A developer labels the axes of a correlation heatmap and discovers the **y-axis tick labels are in reverse order** — the first feature appears at the bottom instead of the top. The `origin='upper'` default should place the first row at the top. What likely caused the reversal?

- [ ] A) `ax.set_yticklabels(feature_names)` — without reversing the list, labels are applied bottom-to-top
- [ ] B) Matplotlib's `imshow` always reverses the y-axis regardless of origin setting
- [ ] C) `ax.set_yticks(range(n))` inverts the y-axis when n > 5
- [ ] D) The correlation matrix was transposed before `imshow` — `corr.T` changes row ordering

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.set_yticklabels(feature_names)` — without reversing the list, labels are applied bottom-to-top

**Explanation:**

- **A is correct:** Matplotlib applies y-tick labels in **increasing y-coordinate order** (bottom to top) by default. For `imshow` with `origin='upper'`, the data rows go top-to-bottom, but the y-axis tick positions increase downward (row 0 at y=0 at TOP, row n-1 at y=n-1 at BOTTOM). When `ax.set_yticklabels(feature_names)` is called, labels are assigned to ticks in the order they appear on the axis — which depends on the axis direction. If the y-axis is inverted (as `imshow` `origin='upper'` does), labels are assigned from top to bottom. The fix: ensure the label order matches the visual row order.
- **B is incorrect:** `imshow` does NOT always reverse the y-axis. With `origin='upper'` (default), row 0 appears at the top (y-axis appears inverted from standard Cartesian). With `origin='lower'`, row 0 is at the bottom (standard Cartesian). The y-axis reversal is intentional and controlled.
- **C is incorrect:** `ax.set_yticks(range(n))` sets tick positions — it does not invert the y-axis for any value of n. Axis inversion is set by `ax.invert_yaxis()` or the `origin` parameter of `imshow`.
- **D is incorrect:** Transposing the correlation matrix (`corr.T`) of a symmetric matrix produces the same matrix (`corr.T == corr` for symmetric matrices). Transposing a symmetric correlation matrix does not change row ordering.

</details>

---

### Question 89

A developer creates a heatmap comparing **model performance across datasets × hyperparameter settings** (a 5×8 grid):

```python
results = np.random.rand(5, 8)  # 5 datasets, 8 hyperparameter configs
datasets = ['D1', 'D2', 'D3', 'D4', 'D5']
configs  = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=1,
               interpolation='nearest', aspect='auto')
ax.set_xticks(range(8)); ax.set_xticklabels(configs, rotation=45, ha='right')
ax.set_yticks(range(5)); ax.set_yticklabels(datasets)
fig.colorbar(im, ax=ax, label='Accuracy', shrink=0.9)
```

Why is `aspect='auto'` used here instead of `aspect='equal'`?

- [ ] A) `aspect='auto'` is faster to render for large matrices
- [ ] B) `aspect='auto'` stretches the 5×8 non-square matrix to fill the `(12, 5)` figure proportions, making each cell visible and the grid readable — `aspect='equal'` would produce 5 tall rows and 8 square columns, making the cells very wide and wasting the figure height
- [ ] C) `aspect='auto'` is required when using `interpolation='nearest'`
- [ ] D) `aspect='auto'` disables the colorbar's aspect ratio constraint

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `aspect='auto'` stretches the 5×8 non-square matrix to fill the `(12, 5)` figure proportions, making each cell visible and the grid readable — `aspect='equal'` would produce 5 tall rows and 8 square columns, making the cells very wide and wasting the figure height

**Explanation:**

- **A is incorrect:** Aspect ratio computation is negligible in rendering time. Performance is not the reason to choose `aspect='auto'` over `aspect='equal'`.
- **B is correct:** The matrix is 5 rows × 8 columns — not square. With `aspect='equal'` in a wide figure `(12, 5)`: each column gets width = 12"/8 ≈ 1.5", so with square pixels, each row would be 1.5" tall → 5 rows × 1.5" = 7.5" — far taller than the 5" figure. Matplotlib would letterbox the image to maintain square pixels, leaving whitespace above/below. With `aspect='auto'`: the 5 rows fill the 5" height (1" per row) and the 8 columns fill the 12" width (1.5" per column) — non-square cells, but the figure space is fully utilized and all cells are clearly visible.
- **C is incorrect:** `interpolation='nearest'` and `aspect=` are independent parameters with no dependency relationship. `nearest` can be used with any aspect setting.
- **D is incorrect:** `aspect='auto'` affects the axes/image aspect ratio — not the colorbar. The colorbar has its own separate `aspect` parameter (in `fig.colorbar()`) that controls its length-to-width ratio independently.

</details>

---

### Question 90

A developer exports the correlation heatmap with:

```python
fig.savefig("correlation_matrix.pdf", dpi=300, bbox_inches='tight',
            facecolor=fig.get_facecolor())
```

Why is `facecolor=fig.get_facecolor()` explicitly passed to `savefig()`?

- [ ] A) It is unnecessary — `savefig` always uses the figure's facecolor automatically
- [ ] B) Without it, `savefig` may override the figure's facecolor with a default white or transparent background, especially in non-interactive backends — explicitly passing it preserves the intended background color in the exported file
- [ ] C) `fig.get_facecolor()` converts the figure to grayscale for PDF export
- [ ] D) It enables transparent background in the PDF output

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Without it, `savefig` may override the figure's facecolor with a default white or transparent background, especially in non-interactive backends — explicitly passing it preserves the intended background color in the exported file

**Explanation:**

- **A is incorrect:** `savefig()` does NOT always use the figure's facecolor. Matplotlib's `savefig` has its own default facecolor setting (configurable via `rcParams['savefig.facecolor']`) which may differ from the figure's current facecolor. In headless/Agg backend environments, this default is sometimes `'white'` regardless of the figure's actual facecolor.
- **B is correct:** `facecolor=fig.get_facecolor()` is a **defensive programming** pattern. `fig.get_facecolor()` retrieves the figure's actual current background color (e.g., custom dark gray for dark-theme figures). Passing it to `savefig()` ensures the exported PDF/PNG uses this exact color — not whatever `rcParams['savefig.facecolor']` is set to. This is especially important in production pipelines where the Matplotlib rcParams may be different from interactive defaults.
- **C is incorrect:** `facecolor` controls the background color — it has no grayscale conversion effect. Grayscale output would require converting the colormap or using a grayscale colormap.
- **D is incorrect:** Transparent backgrounds require `transparent=True` in `savefig()`. Passing `facecolor=fig.get_facecolor()` uses the figure's actual background color — which is opaque unless specifically set to transparent.

</details>
