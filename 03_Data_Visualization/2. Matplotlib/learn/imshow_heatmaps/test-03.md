# Matplotlib Imshow & Heatmaps — Test 03
## Colormap Taxonomies

---

### Question 21

A developer is visualizing a **feature correlation matrix** with values strictly in [-1.0, +1.0]. A colleague suggests using `cmap='Blues'`. Why is this a poor choice?

- [ ] A) `Blues` is too slow to render for large matrices
- [ ] B) `Blues` is a sequential colormap — it maps both negative and positive correlations to the same blue gradient, making it impossible to visually distinguish positive from negative correlations
- [ ] C) `Blues` does not support values below 0 — a `ValueError` is raised
- [ ] D) `Blues` has poor perceptual uniformity and should never be used for scientific visualization

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `Blues` is a sequential colormap — it maps both negative and positive correlations to the same blue gradient, making it impossible to visually distinguish positive from negative correlations

**Explanation:**

- **A is incorrect:** Colormap rendering speed is negligible for any colormap of reasonable size. `Blues` is not measurably slower than any other colormap.
- **B is correct:** `Blues` is a **sequential** colormap (light → dark blue), designed for data with a single direction (low → high, 0 → 1). For correlation matrices, the critical visual distinction is **sign**: negative (-1 to 0) vs. positive (0 to +1). A sequential colormap assigns lighter blue to -1 and darker blue to +1 — a value of -0.9 and +0.9 look equally saturated (both near the extremes) but on the same color gradient. The correct choice is a **diverging** colormap like `coolwarm`, `bwr`, or `seismic`, which maps negative → one hue, zero → neutral, positive → opposite hue.
- **C is incorrect:** `Blues` handles any float values including negatives. With `vmin=-1, vmax=1`, it maps -1 to the lightest blue and +1 to the darkest blue — no error is raised. The problem is conceptual (wrong choice), not technical.
- **D is incorrect:** `Blues` is a perceptually reasonable sequential colormap. The issue is not perceptual uniformity — it's the wrong **category** (sequential instead of diverging) for this data type.

</details>

---

### Question 22

A developer is visualizing **K-means cluster assignments** for 8 clusters on a 2D grid. They use `cmap='viridis'`. A data scientist says this is wrong. Which colormap category and specific colormap is more appropriate?

- [ ] A) Sequential; use `plasma` — it has more distinct color stops than `viridis`
- [ ] B) Diverging; use `seismic` — cluster indices go from 0 to 7, which is bipolar
- [ ] C) Qualitative; use `tab10` or `Set1` — cluster labels are categorical integers with no meaningful ordering or magnitude relationship
- [ ] D) Cyclic; use `hsv` — cluster indices wrap around, requiring a cyclic colormap

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Qualitative; use `tab10` or `Set1` — cluster labels are categorical integers with no meaningful ordering or magnitude relationship

**Explanation:**

- **A is incorrect:** `plasma` is also a sequential (perceptually uniform) colormap. Sequential colormaps imply ordered magnitude — cluster 7 would appear "more" than cluster 0. But cluster IDs are arbitrary labels with no meaningful ordering.
- **B is incorrect:** Diverging colormaps are for data with a meaningful midpoint (zero) with opposite-sign extremes. Cluster IDs [0, 7] are not bipolar — there is no "negative" cluster concept. `seismic` would make cluster 0 and 7 look like opposing extremes with cluster 4 as neutral — completely wrong semantics.
- **C is correct:** Cluster assignments are **categorical** — each cluster is a discrete label with no inherent ordering or magnitude. `viridis` would make cluster 7 look "more intense" than cluster 0. The correct choice is a **qualitative colormap**: `tab10` provides 10 visually distinct, perceptually separated colors (one per category) with no implied ordering. `Set1` is another option. These colormaps are specifically designed to make each category maximally distinguishable.
- **D is incorrect:** `hsv` is a cyclic colormap for data that wraps around (like angles or time of day). Cluster IDs do not wrap — cluster 7 is not "adjacent" to cluster 0. Also, `hsv` has poor perceptual uniformity.

</details>

---

### Question 23

A developer asks: "Why was `jet` (rainbow colormap) deprecated as the default and replaced by `viridis` in Matplotlib 2.0?" What is the primary technical reason?

- [ ] A) `jet` uses too many colors and is slow to compute
- [ ] B) `jet` has non-uniform perceptual luminance — the green and cyan regions appear artificially bright, creating false visual features (phantom ridges) in data that don't correspond to actual data structures
- [ ] C) `jet` is not compatible with colorbar rendering
- [ ] D) `jet` only supports values in [0, 255] and cannot handle float data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `jet` has non-uniform perceptual luminance — the green and cyan regions appear artificially bright, creating false visual features (phantom ridges) in data that don't correspond to actual data structures

**Explanation:**

- **A is incorrect:** Colormap computation complexity is negligible for all standard colormaps. Performance was never a reason to deprecate `jet`.
- **B is correct:** The core problem with `jet` (and all rainbow colormaps) is **perceptual non-uniformity**. Equal numerical steps in the data do NOT produce equal perceived color differences. The green/cyan portion of `jet` is perceptually much brighter than the red or blue portions — causing human visual perception to perceive "features" (bright bands, ridges) in data that are artifacts of the colormap, not real data patterns. `viridis` was designed using CIECAM02 color appearance modeling to be **perceptually uniform**: equal data steps produce equal perceived color steps. It is also accessible to color-blind viewers and converts well to grayscale.
- **C is incorrect:** `jet` works with colorbars just like any other colormap. The deprecation was entirely about perceptual quality, not technical compatibility.
- **D is incorrect:** `jet` handles all numeric types and ranges — it is normalized like any other colormap. The value range restriction is not a `jet`-specific issue.

</details>

---

### Question 24

A developer needs to visualize **loss landscape** values ranging from 0.001 (minimum) to 10.0 (maximum) — spanning 4 orders of magnitude. Which colormap + normalization combination is most appropriate?

- [ ] A) `cmap='viridis', vmin=0, vmax=10` — standard linear scaling
- [ ] B) `cmap='viridis', norm=LogNorm(vmin=0.001, vmax=10)` — log normalization reveals structure across multiple orders of magnitude
- [ ] C) `cmap='coolwarm', vmin=0, vmax=10` — diverging colormap needed for loss values
- [ ] D) `cmap='gray', norm=LogNorm()` — grayscale is always best for scientific data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `cmap='viridis', norm=LogNorm(vmin=0.001, vmax=10)` — log normalization reveals structure across multiple orders of magnitude

**Explanation:**

- **A is incorrect:** With linear scaling `vmin=0, vmax=10`, all values near 0.001 (which is 0.01% of the range) would be compressed into a tiny sliver of the colormap near dark blue. The vast majority of the color range would represent [5, 10], hiding all structure near the minimum. Linear scaling fails when data spans multiple orders of magnitude.
- **B is correct:** `LogNorm(vmin=0.001, vmax=10)` applies logarithmic normalization: values are mapped on a log scale before colormap application. This means [0.001, 0.01], [0.01, 0.1], [0.1, 1], and [1, 10] each receive equal visual real estate — the structure at low loss values (near the minimum) is just as visible as structure at high loss values. This is standard practice for loss landscapes and scientific data with large dynamic range.
- **C is incorrect:** `coolwarm` is a diverging colormap for bipolar data (negative to positive through a neutral zero). Loss values are strictly positive — a diverging colormap is semantically wrong and would suggest a "neutral" loss value that doesn't exist.
- **D is incorrect:** `gray` is not universally "best for scientific data" — this is a false claim. Grayscale is appropriate for some data (medical images, print-safe figures) but `viridis` with `LogNorm` is the correct technical choice for this multi-scale loss landscape.

</details>

---

### Question 25

A developer calls `plt.get_cmap('plasma')` and inspects it. Which property correctly describes `plasma`?

- [ ] A) Diverging — maps through blue → white → red with a neutral center
- [ ] B) Sequential and perceptually uniform — maps from deep blue-purple through magenta to bright yellow, designed for accessibility and grayscale compatibility
- [ ] C) Qualitative — assigns distinct colors to discrete categories
- [ ] D) Cyclic — wraps around so that the start and end colors are identical

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Sequential and perceptually uniform — maps from deep blue-purple through magenta to bright yellow, designed for accessibility and grayscale compatibility

**Explanation:**

- **A is incorrect:** `plasma` is not diverging. Diverging colormaps have a neutral midpoint and opposing hue directions. `plasma` has a single monotone progression without a neutral center.
- **B is correct:** `plasma` is one of Matplotlib's four **perceptually uniform sequential** colormaps (along with `viridis`, `inferno`, and `magma`), introduced in Matplotlib 1.5. It progresses from deep blue-purple (low values) through magenta/pink (mid values) to bright yellow (high values). It is: (1) perceptually uniform (equal data steps → equal perceived color steps), (2) accessible to colorblind viewers (uses no pure red/green confusion pairs), and (3) converts meaningfully to grayscale (monotone luminance).
- **C is incorrect:** `plasma` is not qualitative. Qualitative colormaps (`tab10`, `Set1`, `Paired`) use maximally distinct hues for discrete categories — they have no monotone progression.
- **D is incorrect:** Cyclic colormaps (`twilight`, `hsv`) wrap around — start and end are the same hue. `plasma` goes from blue-purple to yellow with no wrapping.

</details>

---

### Question 26

A developer visualizes a **sentiment analysis heatmap** where values range from -1.0 (strongly negative) to +1.0 (strongly positive), with 0.0 being neutral. They use `cmap='coolwarm'` with `vmin=-1, vmax=1`. A colleague says to also set `vcenter=0` using `TwoSlopeNorm`. Is this necessary?

- [ ] A) Yes — without `TwoSlopeNorm`, `coolwarm` places its center at 0.5 (the mathematical midpoint), not at 0.0
- [ ] B) No — `coolwarm` is a symmetric diverging colormap; with `vmin=-1, vmax=1`, the center automatically aligns with 0.0 because the range is perfectly symmetric around zero
- [ ] C) Yes — `coolwarm` is a sequential colormap and always needs `TwoSlopeNorm` to work correctly
- [ ] D) No — `vmin`/`vmax` are always ignored when using diverging colormaps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `coolwarm` is a symmetric diverging colormap; with `vmin=-1, vmax=1`, the center automatically aligns with 0.0 because the range is perfectly symmetric around zero

**Explanation:**

- **A is incorrect:** The concern about `TwoSlopeNorm` applies when the negative and positive ranges have **different magnitudes** (e.g., `vmin=-0.5, vmax=2.0`). With symmetric `vmin=-1, vmax=1`, linear normalization maps 0 to exactly `(0-(-1))/(1-(-1)) = 0.5` — the exact center of the colormap. For `coolwarm`, the center (0.5 normalized) is white/neutral. So the center is at 0.0 data value — as intended.
- **B is correct:** With `vmin=-1, vmax=1` (symmetric around 0), standard linear normalization correctly places 0 at the colormap center (normalized value 0.5). `coolwarm`'s neutral white is at normalized 0.5. No `TwoSlopeNorm` is needed for symmetric ranges. `TwoSlopeNorm` is only needed when the range is asymmetric (e.g., `vmin=-2, vmax=5`) and you still want `vcenter=0` to be at the colormap center.
- **C is incorrect:** `coolwarm` is a **diverging** colormap, not sequential. It does not require `TwoSlopeNorm` — standard linear normalization with symmetric bounds works correctly.
- **D is incorrect:** `vmin`/`vmax` are never ignored for any colormap type. They always define the normalization bounds.

</details>

---

### Question 27

A developer must choose a colormap for a **class activation map (CAM)** visualization in a CNN, where values represent feature importance (0 = unimportant, 1 = most important). The map will be overlaid on the original image. Which is the most appropriate choice?

- [ ] A) `cmap='coolwarm'` — diverging map highlights positive and negative activations
- [ ] B) `cmap='jet'` — the rainbow colormap produces maximum contrast for publication figures
- [ ] C) `cmap='hot'` or `cmap='Reds'` — sequential, perceptually intuitive heatmaps where low values are cool/light and high values are warm/intense
- [ ] D) `cmap='tab20'` — qualitative colormap provides distinct colors for each activation level

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `cmap='hot'` or `cmap='Reds'` — sequential, perceptually intuitive heatmaps where low values are cool/light and high values are warm/intense

**Explanation:**

- **A is incorrect:** `coolwarm` is diverging — designed for data with meaningful negative values. Activation maps are strictly [0, 1] (unipolar, no negative values) — a diverging colormap's blue half is wasted on the range [0, 0.5] and the red half on [0.5, 1], creating a confusing visual center that doesn't correspond to any meaningful zero.
- **B is incorrect:** While `jet` produces high contrast, it introduces perceptual artifacts (false features due to non-uniform luminance). For scientific visualization overlaid on images, `jet` can confuse viewers about activation intensity. Modern practice recommends perceptually uniform alternatives.
- **C is correct:** For activation maps (strictly non-negative, 0=none to 1=full), **sequential** colormaps are semantically correct. `hot` (black → red → orange → yellow → white) or `Reds` (white → dark red) are standard choices for CAMs. `hot` with `alpha=0.5` overlaid on the original image is the Grad-CAM visualization standard — low activations are dark/transparent, high activations are bright/intense.
- **D is incorrect:** `tab20` is a qualitative colormap for discrete categorical data. Activation values are continuous [0, 1] — applying a qualitative colormap would make smoothly varying activations appear as random color patches, destroying the spatial continuity of the activation map.

</details>

---

### Question 28

A developer examines these four colormaps and must classify each as sequential (S), diverging (D), or qualitative (Q):

`viridis`, `seismic`, `Paired`, `inferno`

- [ ] A) viridis=S, seismic=S, Paired=D, inferno=Q
- [ ] B) viridis=S, seismic=D, Paired=Q, inferno=S
- [ ] C) viridis=D, seismic=D, Paired=Q, inferno=S
- [ ] D) viridis=S, seismic=Q, Paired=S, inferno=D

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) viridis=S, seismic=D, Paired=Q, inferno=S

**Explanation:**

- **A is incorrect:** `seismic` is diverging (blue→white→red), not sequential. `Paired` is qualitative, not diverging. `inferno` is sequential, not qualitative.
- **B is correct:** Classification: `viridis` = **Sequential** (dark blue → green → yellow, perceptually uniform). `seismic` = **Diverging** (blue → white → red, symmetric diverging for earthquake/seismic data). `Paired` = **Qualitative** (12 paired distinct colors for categorical data). `inferno` = **Sequential** (black → purple → red → orange → yellow, perceptually uniform). These are the standard Matplotlib colormap categories.
- **C is incorrect:** `viridis` is sequential, not diverging. It has no neutral midpoint and flows in a single direction.
- **D is incorrect:** `seismic` is diverging (blue-white-red with neutral center), not qualitative. `Paired` is qualitative (distinct categorical colors), not sequential.

</details>

---

### Question 29

A developer prints the colormap for `viridis` at value 0.0, 0.5, and 1.0 using:

```python
cmap = plt.cm.viridis
print(cmap(0.0))   # RGBA at minimum
print(cmap(0.5))   # RGBA at midpoint
print(cmap(1.0))   # RGBA at maximum
```

Which description matches the expected output colors?

- [ ] A) (1.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 1.0), (0.0, 0.0, 1.0, 1.0) — red to gray to blue
- [ ] B) Approximately dark blue-purple at 0.0, blue-green-teal at 0.5, bright yellow at 1.0
- [ ] C) Black at 0.0, dark red at 0.5, bright white at 1.0 — this is the `hot` colormap description
- [ ] D) Light blue at 0.0, white at 0.5, dark red at 1.0 — this is `coolwarm`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Approximately dark blue-purple at 0.0, blue-green-teal at 0.5, bright yellow at 1.0

**Explanation:**

- **A is incorrect:** This describes a red→gray→blue colormap (like `bwr` reversed or similar diverging map). `viridis` does not use pure red.
- **B is correct:** `viridis` progression: 0.0 → dark blue-purple `(0.267, 0.005, 0.329)`, 0.5 → blue-teal-green `(0.128, 0.566, 0.551)`, 1.0 → bright yellow `(0.993, 0.906, 0.144)`. `cmap(scalar)` returns an RGBA tuple in [0,1]. The viridis colormap's distinctive progression from dark purple through blue-green to yellow is its recognizable signature.
- **C is incorrect:** Black → dark red → white describes the `hot` colormap. `hot` is an alternative sequential colormap often used for thermal/heat visualization.
- **D is incorrect:** Light blue → white → dark red describes `coolwarm` reversed, or `RdBu`. `coolwarm` is diverging: dark blue at 0.0, white at 0.5, dark red at 1.0 (not reversed).

</details>

---

### Question 30

A developer visualizes a **confusion matrix** using `imshow` and must select the correct colormap. The matrix has values from 0 to 500 (counts), where high values on the diagonal are desirable. Which colormap and logic applies?

- [ ] A) `coolwarm` with `vmin=-500, vmax=500` — diverging to show high diagonal vs. low off-diagonal
- [ ] B) `Blues` or `YlOrRd` (sequential) with `vmin=0, vmax=max_count` — high counts should appear darkest/most intense, low counts lightest
- [ ] C) `tab10` — each confusion matrix cell has a unique category requiring a qualitative colormap
- [ ] D) `gray_r` (reversed gray) — black cells indicate correct predictions, white cells indicate errors

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `Blues` or `YlOrRd` (sequential) with `vmin=0, vmax=max_count` — high counts should appear darkest/most intense, low counts lightest

**Explanation:**

- **A is incorrect:** Confusion matrix counts are strictly non-negative (0 to max_count). There are no negative values — a diverging colormap is semantically wrong. Setting `vmin=-500` would place 0 (a common and important value) at the colormap center (neutral), not at the minimum.
- **B is correct:** Confusion matrix values are **unipolar** (0 to N) — a sequential colormap is correct. `Blues` maps 0 (white/light) to max_count (dark blue), making high-count cells (correct classifications on the diagonal) visually prominent. `YlOrRd` (yellow → orange → red) is another standard choice. The key is that high counts should be visually intense, low counts should be pale — exactly what sequential colormaps provide.
- **C is incorrect:** Confusion matrix cells contain **numeric counts** — they are not categorical labels. A qualitative colormap like `tab10` would assign arbitrary distinct colors to different count values, making the magnitude comparison completely impossible.
- **D is incorrect:** While `gray_r` would work technically (reversed gray: 0=white, max=black), it is not the standard choice. `Blues` or `YlOrRd` are more commonly used as they provide better perceptual contrast between the diagonal (high counts) and off-diagonal (low counts) cells.

</details>
