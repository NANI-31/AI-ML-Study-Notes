# Matplotlib Scatter Plot Objective Test — Set 04

---

### 31. Question

A data analyst wants to map exam scores (ranging from 0 to 100) to a continuous color gradient across scatter markers. Which parameter call in `plt.scatter()` correctly activates continuous colormapping?

- [ ] **A.** `plt.scatter(hours, scores, color_gradient=scores)`
- [ ] **B.** `plt.scatter(hours, scores, palette=scores)`
- [ ] **C.** `plt.scatter(hours, scores, hue=scores)`
- [ ] **D.** `plt.scatter(hours, scores, c=scores, cmap="viridis")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `color_gradient` is not a valid parameter in Matplotlib.
- **Option B (Incorrect):** `palette` is used in Seaborn, not base Matplotlib.
- **Option C (Incorrect):** `hue` is a high-level grouping parameter in Seaborn, not Matplotlib's `plt.scatter()`.
- **Option D (Correct):** In Matplotlib, passing a sequence of numeric values to `c` (such as `c=scores`) instructs the artist to map those numbers through a colormap lookup table. Specifying `cmap="viridis"` designates the specific color lookup palette to use.

</details>

---

### 32. Question

Since Matplotlib version 2.0, what is the default colormap applied to scatter plots and heatmaps when a continuous numeric array is passed to `c` without specifying `cmap`?

- [ ] **A.** `'jet'`
- [ ] **B.** `'viridis'`
- [ ] **C.** `'rainbow'`
- [ ] **D.** `'coolwarm'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `'jet'` was the legacy default prior to Matplotlib 2.0, but it was deprecated as the default due to severe perceptual flaws.
- **Option B (Correct):** `'viridis'` is the default sequential colormap in modern Matplotlib. It was engineered specifically to be perceptually uniform in lightness, monotonic across color gradients, and fully interpretable by individuals with red-green color blindness.
- **Option C (Incorrect):** `'rainbow'` suffers from the same non-uniform luminance banding as `'jet'` and is not the default.
- **Option D (Incorrect):** `'coolwarm'` is a diverging colormap, not the default sequential map.

</details>

---

### 33. Question

Why does modern data visualization science strongly discourage the use of the legacy `'jet'` (rainbow) colormap in continuous scatter plots?

- [ ] **A.** The `'jet'` colormap consumes four times more GPU VRAM than `'viridis'`.
- [ ] **B.** Matplotlib raises a `DeprecationWarning` and halts execution if `'jet'` is passed in modern versions.
- [ ] **C.** Its luminance profile is non-monotonic (having artificial brightness peaks in cyan and yellow), which introduces illusory visual boundaries and masks true continuous gradients.
- [ ] **D.** The colors in `'jet'` invert automatically when exported to SVG vector files.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Colormaps are simple 256-element RGBA lookup tables and have identical, negligible memory footprints.
- **Option B (Incorrect):** `'jet'` is still included in Matplotlib for backward compatibility and does not halt execution.
- **Option C (Correct):** The `'jet'` colormap transitions through deep blue, cyan, yellow, and red. Its luminance curve rises and plummets abruptly, creating sharp perceptual bands (especially around cyan and yellow) where small numerical changes look like dramatic thresholds, while large numerical shifts in green appear flat. Furthermore, when printed in grayscale or viewed by color-blind individuals, monotonic ordering is completely destroyed.
- **Option D (Incorrect):** Vector exports preserve the exact colors assigned; no automatic color inversion occurs.

</details>

---

### 34. Question

Colormaps in Matplotlib are organized into three primary structural categories. Which row correctly matches each category to its defining characteristics and canonical example?

- [ ] **A.** Sequential: Monotonically increasing lightness for ordered numeric data (`'viridis'`); Diverging: Two contrasting hues diverging from a central neutral midpoint (`'coolwarm'`); Qualitative: Distinct unrelated hues for unordered categorical classes (`'tab10'`).
- [ ] **B.** Sequential: Random hue cycling (`'tab20'`); Diverging: Monotonic dark-to-light progression (`'Blues'`); Qualitative: Mirrored bipolar gradients (`'seismic'`).
- [ ] **C.** Sequential: Equal lightness throughout (`'Set1'`); Diverging: Continuous rainbow cycling (`'hsv'`); Qualitative: Single-hue saturation gradients (`'Greens'`).
- [ ] **D.** Sequential: Bipolar red-blue balance (`'bwr'`); Diverging: Unordered random colors (`'Dark2'`); Qualitative: Monotonically increasing luminance (`'magma'`).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):**
  - **Sequential** colormaps (e.g., `'viridis'`, `'plasma'`, `'Blues'`) vary monotonically in luminance, ideal for continuous values representing magnitude from low to high.
  - **Diverging** colormaps (e.g., `'coolwarm'`, `'seismic'`, `'bwr'`) emphasize deviation in both directions from a critical neutral baseline (such as zero, mean, or median).
  - **Qualitative** colormaps (e.g., `'tab10'`, `'Set1'`, `'Dark2'`) use distinct visual hues with similar luminance to distinguish unordered categorical classes without implying hierarchical precedence.
- **Option B (Incorrect):** This option erroneously classifies `'tab20'` (a qualitative map) as sequential, `'Blues'` (a sequential map) as diverging, and `'seismic'` (a diverging map) as qualitative.
- **Option C (Incorrect):** This option misclassifies qualitative `'Set1'` as sequential, cyclic `'hsv'` as diverging, and sequential `'Greens'` as qualitative.
- **Option D (Incorrect):** This option misclassifies diverging `'bwr'` as sequential, qualitative `'Dark2'` as diverging, and sequential `'magma'` as qualitative.

</details>

---

### 35. Question

A climatologist is analyzing regional temperature anomalies, where values range from $-6.0^\circ\text{C}$ (much colder than normal) through $0.0^\circ\text{C}$ (baseline average) to $+6.0^\circ\text{C}$ (much warmer than normal).

Which colormap is best suited for visual encoding of this continuous variable?

- [ ] **A.** A sequential colormap such as `'Greens'`
- [ ] **B.** A diverging colormap such as `'coolwarm'` (or `'bwr'`)
- [ ] **C.** A qualitative colormap such as `'tab10'`
- [ ] **D.** A cyclic colormap such as `'twilight'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A sequential colormap like `'Greens'` progresses monotonically from light to dark green, failing to distinguish between negative and positive deviations relative to the zero baseline.
- **Option B (Correct):** When data possesses a natural, meaningful central midpoint (here, an anomaly of $0.0^\circ\text{C}$), a **diverging colormap** such as `'coolwarm'` maps the neutral baseline to a desaturated light gray/white, negative anomalies to cold blue hues, and positive anomalies to warm red hues.
- **Option C (Incorrect):** Qualitative colormaps are for discrete unordered categories and do not convey continuous numerical magnitude or sign.
- **Option D (Incorrect):** Cyclic colormaps (`'twilight'`) are designed for periodic angular data (such as wind directions $0^\circ\text{--}360^\circ$ or phase angles), where the highest and lowest values wrap around and share the same color.

</details>

---

### 36. Question

A health dataset contains patient age ranging continuously from 18 to 85 years. What category of colormap should be assigned to `c=age`?

- [ ] **A.** Qualitative colormap (e.g., `'Pastel1'`)
- [ ] **B.** Diverging colormap (e.g., `'seismic'`)
- [ ] **C.** Sequential colormap (e.g., `'viridis'`, `'plasma'`, or `'Blues'`)
- [ ] **D.** Cyclic colormap (e.g., `'hsv'`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Qualitative maps assign unrelated categorical hues, destroying the continuous ordered nature of age.
- **Option B (Incorrect):** Age is strictly positive and has no natural bipolar center (e.g., there are no "negative" ages to balance positive ages around zero).
- **Option C (Correct):** Age is an ordered, strictly positive continuous measurement. A **sequential colormap** (such as `'viridis'` or `'plasma'`) maps younger ages to darker/lower-luminance shades and older ages to lighter/higher-luminance shades in a smooth, intuitive monotonic progression.
- **Option D (Incorrect):** Age is linear and does not loop back periodically.

</details>

---

### 37. Question

A botanist is plotting the Iris dataset, where each observation belongs to one of three discrete species: *Setosa*, *Versicolor*, or *Virginica*. Why is passing `c=species_encoded` directly to a continuous sequential colormap like `cmap='viridis'` poor visualization practice?

- [ ] **A.** Matplotlib raises an internal rasterization exception when categorical integers are passed to `c`.
- [ ] **B.** It forces all points to be rendered in black and white.
- [ ] **C.** It causes species with ID 2 to occupy twice the physical marker size as species with ID 1.
- [ ] **D.** A continuous sequential colormap falsely implies an ordered quantitative spectrum between categories (suggesting *Versicolor* is the mathematical average of *Setosa* and *Virginica*), rather than showing distinct, non-hierarchical nominal groups.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib can technically map integer arrays `[0, 1, 2]` to continuous colormaps without error.
- **Option B (Incorrect):** The points will receive distinct colors from the gradient, not black and white.
- **Option C (Incorrect):** Marker size is controlled by `s`, not `c`.
- **Option D (Correct):** Nominal categorical variables have no inherent mathematical order or magnitude. Using a sequential continuous colormap implies an artificial progression and metric distance between species. Discrete categories should be rendered using distinct qualitative colors (via separate `scatter` calls with a discrete legend or a qualitative map like `'tab10'`).

</details>

---

### 38. Question

A data analyst is using the `'viridis'` colormap but wants low values to appear light yellow and high values to appear deep purple (the exact inverse of standard viridis). How can this be accomplished directly in the `cmap` parameter string?

- [ ] **A.** `cmap="viridis_r"`
- [ ] **B.** `cmap="inverse(viridis)"`
- [ ] **C.** `cmap="viridis.reverse()"`
- [ ] **D.** `cmap="-viridis"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, appending the suffix `_r` to any registered colormap name (e.g., `"viridis_r"`, `"plasma_r"`, `"coolwarm_r"`, `"Blues_r"`) generates the reversed version of that colormap lookup table.
- **Option B (Incorrect):** Function-style syntax like `inverse(...)` is not valid syntax for colormap names.
- **Option C (Incorrect):** Calling a Python method `.reverse()` inside the string argument is invalid.
- **Option D (Incorrect):** Prefixing a negative sign does not invert colormaps in Matplotlib.

</details>

---

### 39. Question

Compare the following two code snippets:

```python
# Snippet 1
plt.scatter(x, y, c=["red", "blue", "green"], cmap="viridis")

# Snippet 2
plt.scatter(x, y, c=[10, 20, 30], cmap="viridis")
```

What is the difference in how Matplotlib processes `c` and `cmap` between these snippets?

- [ ] **A.** Snippet 1 raises a `TypeError`, while Snippet 2 executes successfully.
- [ ] **B.** Both snippets use `'viridis'` to interpolate between red, blue, and green.
- [ ] **C.** In Snippet 1, because `c` is a sequence of explicit color names, Matplotlib applies those colors directly and completely ignores `cmap`; in Snippet 2, because `c` is a sequence of numeric floats/ints, Matplotlib normalizes the numbers and maps them through `cmap='viridis'`.
- [ ] **D.** Snippet 2 raises a `ValueError` because continuous numbers cannot be supplied to `c`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Snippet 1 does not raise a `TypeError`; Matplotlib natively accepts sequences of color specifications.
- **Option B (Incorrect):** Matplotlib does not blend user-supplied color strings with `'viridis'`.
- **Option C (Correct):** If `c` contains valid color specifications (color names, hex codes, or RGBA tuples), Matplotlib treats them as direct literal color assignments for each marker and ignores `cmap`. When `c` is a sequence of numerical scalars, Matplotlib recognizes that continuous colormapping is requested, normalizes the scalars, and looks up the corresponding RGBA values from `cmap`.
- **Option D (Incorrect):** Numeric arrays are the standard input for colormapped scatter plots.

</details>

---

### 40. Question

Which of the following statements regarding colormap mechanics in `plt.scatter()` are **correct**?

1. Passing a single string such as `c="blue"` sets a uniform color for all points and ignores `cmap`.
2. The length of a numeric array passed to `c` must match the length of `x` and `y`.
3. Matplotlib colormaps internally discretize continuous data into exactly 256 RGBA color lookup bins.
4. Calling `plt.colorbar()` without a colormapped artist reference will automatically find and link to any line plot on the axes.

- [ ] **A.** 1 and 4 only
- [ ] **B.** 1, 2, and 3 only
- [ ] **C.** 2 and 4 only
- [ ] **D.** 2, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While Statement 1 is correct (single string overrides `cmap`), Statement 4 is false because line plots (`Line2D`) lack scalar colormapping and cannot be automatically linked to `plt.colorbar()`.
- **Option B (Correct):** Statements 1, 2, and 3 are all correct. Statement 1 is valid because a single color string sets a uniform facecolor; Statement 2 is valid because numeric colormapping requires a 1-to-1 correspondence with coordinates; and Statement 3 is valid because standard Matplotlib colormaps use 256-level RGBA lookup tables. Statement 4 is false.
- **Option C (Incorrect):** Statement 4 is false, and this option fails to include valid Statement 1.
- **Option D (Incorrect):** Statement 4 is false, and this option omits valid Statement 1.

</details>

---
