# Matplotlib Histogram Objective Test — Set 02

---

### 11. Question

A developer passes a **2D NumPy array** to `plt.hist()` by mistake:

```python
data_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
plt.hist(data_2d, bins=5)
```

What happens?

- [ ] **A.** Matplotlib raises a `ValueError` because `plt.hist()` only accepts 1D arrays
- [ ] **B.** Matplotlib treats each **column** of the 2D array as a separate dataset and draws a grouped histogram with 3 series
- [ ] **C.** Matplotlib flattens the array into 1D `[1,2,3,4,5,6,7,8,9]` automatically before plotting
- [ ] **D.** Matplotlib treats each **row** of the 2D array as a separate series and draws an overlaid histogram with 3 series

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does NOT raise a `ValueError` for 2D arrays. It has a defined behavior for multi-dimensional input — it interprets columns as separate data series.
- **Option B (Correct):** When a 2D array is passed, `plt.hist()` treats each **column** as a separate dataset and plots them as a grouped (side-by-side) histogram. For a (3×3) array, there are 3 columns, resulting in 3 grouped histogram series drawn in different colors. This is equivalent to passing a list of 1D arrays: `plt.hist([col1, col2, col3], bins=5)`.
- **Option C (Incorrect):** `plt.hist()` does NOT auto-flatten 2D arrays. If you want to flatten first, you must explicitly call `data_2d.flatten()` or `data_2d.ravel()` before passing to `plt.hist()`.
- **Option D (Incorrect):** The grouping is by **columns**, not rows. For a (3×3) array, columns are treated as separate series — so 3 series each of length 3, not 3 series each of length 3 from rows (though in a square matrix the count happens to be the same). The conceptual distinction matters for non-square arrays.

</details>

---

### 12. Question

A developer creates a histogram with `rwidth=0.8`. What visual effect does this have?

- [ ] **A.** Scales each bar's height to 80% of its actual frequency count
- [ ] **B.** Introduces a gap between adjacent bars — each bar occupies only 80% of its bin width, with 20% whitespace between bars
- [ ] **C.** Sets the histogram width (figure width) to 80% of the default
- [ ] **D.** Applies 80% transparency (alpha=0.8) to all bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `rwidth` does NOT affect bar heights. Frequencies are always computed from actual data counts. Scaling heights would require normalization parameters like `density=True`.
- **Option B (Correct):** `rwidth` (relative width) controls what fraction of the bin width each bar occupies. `rwidth=0.8` makes each bar 80% of its bin width — leaving a 10% gap on each side between adjacent bars. This visual separation helps readers distinguish individual bars, especially in dense histograms. `rwidth=1.0` (default) means bars touch with no gap; `rwidth=0.8` introduces visible whitespace.
- **Option C (Incorrect):** `rwidth` is a bar-width parameter, not a figure-dimension parameter. Figure width is controlled by `figsize=(width, height)` in `plt.figure()` or `plt.subplots()`.
- **Option D (Incorrect):** Transparency is controlled by the `alpha` parameter (e.g., `plt.hist(data, alpha=0.5)`). `rwidth` is purely a geometric width parameter.

</details>

---

### 13. Question

A developer calls:

```python
counts, bins, patches = plt.hist(data, bins=15)
print(bins[0], bins[-1])
```

What do `bins[0]` and `bins[-1]` represent?

- [ ] **A.** The height of the first bar and the height of the last bar
- [ ] **B.** The left edge of the first bin and the right edge of the last bin — `bins[0] = min(data)` and `bins[-1] = max(data)`
- [ ] **C.** The center of the first bin and the center of the last bin
- [ ] **D.** The count in the first bin and the count in the last bin

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Bar heights (frequency counts or densities) are stored in the `counts` array, not in `bins`. The `bins` array contains only edge positions.
- **Option B (Correct):** The `bins` array returned by `plt.hist()` contains the **bin edge boundaries**. `bins[0]` is the left edge of the very first bin — equal to `min(data)` for default range. `bins[-1]` is the right edge of the very last bin — equal to `max(data)`. All intermediate values `bins[1]` through `bins[-2]` are the shared edges between adjacent bins. There are always `N+1` edges for `N` bins.
- **Option C (Incorrect):** Bin centers would be `(bins[i] + bins[i+1]) / 2` for each bin. The `bins` array contains edges, not centers. To get the centers: `centers = (bins[:-1] + bins[1:]) / 2`.
- **Option D (Incorrect):** Count values per bin are in the `counts` array (first return value). The `bins` array is purely for geometric positioning of bin boundaries.

</details>

---

### 14. Question

A developer uses `orientation='horizontal'` in `plt.hist()`. How does this change the histogram?

- [ ] **A.** The bars extend left-to-right (horizontally) instead of bottom-to-top (vertically), and the data values appear on the y-axis while frequencies appear on the x-axis
- [ ] **B.** The histogram is rotated 90° — data values remain on the x-axis but bars now extend horizontally
- [ ] **C.** The bars are drawn with a horizontal gradient fill
- [ ] **D.** The histogram bins are evenly distributed horizontally across the full figure width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `orientation='horizontal'` rotates the entire histogram: the **data values** (bin positions) move to the **y-axis**, and **frequencies or densities** (bar length) appear on the **x-axis**. Bars extend horizontally from left (zero) to right (count). This orientation is useful when comparing distributions side-by-side with a shared y-axis, or when the data labels are long and need vertical space.
- **Option B (Incorrect):** With `orientation='horizontal'`, data values DO move from x-axis to y-axis — the description "data values remain on the x-axis" is incorrect. The axes are fully transposed.
- **Option C (Incorrect):** `orientation='horizontal'` is a layout parameter, not a styling parameter. Gradient fills would require custom patch styling using `matplotlib.colors` or `PatchCollection` — completely unrelated to orientation.
- **Option D (Incorrect):** The `orientation` parameter changes the axis along which bins are laid out — it does not affect how bins are spaced or distributed relative to the figure width.

</details>

---

### 15. Question

A developer creates a histogram:

```python
np.random.seed(0)
data = np.random.exponential(scale=2, size=300)
plt.hist(data, bins=25, color="coral", edgecolor="black")
plt.axvline(np.mean(data), color="blue", linestyle="--", label="Mean")
plt.axvline(np.median(data), color="green", linestyle="-", label="Median")
plt.legend()
plt.show()
```

The mean line appears **to the right** of the median line. What does this indicate about the distribution?

- [ ] **A.** Left-skewed distribution — the mean is pulled toward the left tail
- [ ] **B.** Right-skewed (positively skewed) distribution — the long right tail pulls the mean above the median
- [ ] **C.** Normal distribution — mean and median should always be equal
- [ ] **D.** Bimodal distribution — two peaks cause the mean to shift right of the median

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In a left-skewed distribution, the mean is pulled **left** (below) the median due to the long left tail. Here, mean > median, which is the opposite pattern.
- **Option B (Correct):** When `mean > median`, the distribution is **right-skewed (positively skewed)**. The exponential distribution is a classic example — it has a peak near zero and a long tail extending to the right. Extreme values in the right tail inflate the arithmetic mean above the median. The `axvline` visualization shows this clearly: the mean line (blue, dashed) sits to the right of the median line (green, solid). For exponential with scale=2: theoretical mean=2, theoretical median≈1.386.
- **Option C (Incorrect):** A normal (Gaussian) distribution has mean = median = mode — they coincide at the center of the symmetric bell curve. The fact that mean ≠ median here confirms this is NOT a normal distribution.
- **Option D (Incorrect):** A bimodal distribution has two separate peaks. While it CAN have mean ≠ median, bimodality itself is not what causes mean > median — right-skewness is. The exponential distribution (used here) is unimodal with a single peak near zero.

</details>

---

### 16. Question

A developer wants to compare two distributions using **fully overlapping** histograms but finds one distribution's bars are completely hidden behind the other:

```python
plt.hist(group_a, bins=20, color="blue")
plt.hist(group_b, bins=20, color="red")
```

What is the BEST fix to make both distributions visible?

- [ ] **A.** Use `plt.subplot(1, 2, 1)` and `plt.subplot(1, 2, 2)` to separate them into different panels
- [ ] **B.** Add `alpha=0.5` to both `plt.hist()` calls to make both distributions semi-transparent
- [ ] **C.** Increase `bins=20` to `bins=50` — more bins prevents overlap
- [ ] **D.** Use different `figsize` values for each distribution

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Using subplots separates the histograms into different axes — making direct overlap comparison harder. The strength of overlapping histograms is that they share the same axes for direct visual comparison. Separating them defeats this purpose (though it's valid for other comparison scenarios).
- **Option B (Correct):** `alpha=0.5` (or `alpha=0.6`–`0.7`) makes each histogram bar **semi-transparent** — both distributions remain visible on the same axes. Where they overlap, colors blend. This is the standard technique for overlaying multiple distributions. The code becomes: `plt.hist(group_a, bins=20, color="blue", alpha=0.5)` and `plt.hist(group_b, bins=20, color="red", alpha=0.5)`.
- **Option C (Incorrect):** More bins does not reduce overlap — in fact, it may make the overlap worse by creating more bars. Overlap occurs because both histograms cover the same x-range; the number of bins doesn't change this.
- **Option D (Incorrect):** `figsize` affects the figure dimensions (physical size in inches). It does not affect transparency or visual overlap between histogram bars plotted on the same axes.

</details>

---

### 17. Question

A developer plots:

```python
data = np.random.normal(60, 8, 1000)
plt.hist(data, bins=30, histtype='step', color='navy', linewidth=2)
```

What does `histtype='step'` produce compared to the default `histtype='bar'`?

- [ ] **A.** Identical to `histtype='bar'` but with `alpha=0.5` applied automatically
- [ ] **B.** An unfilled step-line outline of the histogram shape — only the perimeter of the bars is drawn with no fill color
- [ ] **C.** A smoothed continuous curve instead of rectangular bars
- [ ] **D.** A stacked histogram where each bin value is cumulated from the previous

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `histtype='step'` has nothing to do with transparency. No automatic alpha is applied. The key difference is fill vs. no fill — not transparency.
- **Option B (Correct):** `histtype='step'` draws only the **outline** of the histogram — a stepped line following the top and sides of each bin — without filling the interior with color. This is useful for overlaying multiple histograms because the step outlines don't obscure each other (unlike filled bars). The `color='navy'` applies to the line color. `linewidth=2` makes the outline thicker. `histtype='stepfilled'` is a variant that draws the step outline AND fills the interior.
- **Option C (Incorrect):** `histtype='step'` still produces the rectangular stepped shape — it is NOT a smooth curve. Smooth density estimation (kernel density estimation, KDE) requires separate tools like `scipy.stats.gaussian_kde` or seaborn's `kdeplot`.
- **Option D (Incorrect):** Stacking behavior is controlled by `stacked=True` or `histtype='barstacked'`, not `histtype='step'`. The `step` type changes the visual rendering style (line vs. filled bar), not the stacking algorithm.

</details>

---

### 18. Question

A developer creates a histogram for a dataset with values ranging from 0 to 1000, but there is one extreme outlier at value 50000. They use `plt.hist(data, bins=30)`. What visual problem occurs?

- [ ] **A.** Matplotlib raises an error because outliers exceed the bin range
- [ ] **B.** All 30 bins are compressed into the range [0, 1000] — the outlier creates 29 empty bins at the far right, squashing all meaningful data into a few barely-visible bins on the left
- [ ] **C.** The outlier is automatically removed before binning
- [ ] **D.** The outlier creates a separate 31st bin automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does NOT raise errors for outliers. The `bins` range automatically expands to `[min(data), max(data)]`, which includes the outlier.
- **Option B (Correct):** When `bins=30` is used with integer count, Matplotlib spans bins from `min(data)=0` to `max(data)=50000`. Each bin has width `50000/30 ≈ 1667`. Almost all meaningful data (0–1000) falls into the very first bin `[0, 1667)` — appearing as a single tall bar on the far left. The remaining 29 bins cover 1667–50000, which are mostly empty except the last bin containing the outlier. The histogram becomes **uninformative** for the main data distribution. The fix is to use `range=(0, 1100)` to clip the outlier from the histogram range.
- **Option C (Incorrect):** Matplotlib never auto-removes outliers. Unless `range=(min_val, max_val)` is explicitly set to clip data, all values including extreme outliers are included in the binning.
- **Option D (Incorrect):** Matplotlib does not create extra bins for outliers. The `bins` count is exactly as specified. Data outside a custom `range=(a,b)` is excluded, but within default auto-range, all data is accommodated within the specified bin count.

</details>

---

### 19. Question

A developer correctly uses the `range` parameter to handle the outlier problem:

```python
data = np.concatenate([np.random.normal(100, 15, 999), [50000]])
plt.hist(data, bins=30, range=(50, 200))
```

What happens to the outlier value 50000?

- [ ] **A.** It is included in the last bin `[190, 200]`
- [ ] **B.** It is **silently excluded** — data outside `range=(50, 200)` is not counted in any bin
- [ ] **C.** It causes the histogram to automatically extend the range to 50000
- [ ] **D.** Matplotlib raises a `ValueError` because 50000 exceeds the specified range

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Data values outside the specified `range` are NOT included in boundary bins. The `range` parameter clips the binning domain — anything outside [50, 200] is completely ignored in the count.
- **Option B (Correct):** The `range=(min_val, max_val)` parameter in `plt.hist()` specifies the **binning domain**. Only data values within `[50, 200]` are assigned to bins and counted. The outlier value 50000 falls outside this range and is **silently excluded** from all bins and counts. This is the correct way to focus a histogram on the meaningful data range while ignoring extreme outliers. The total count in the histogram will be 999, not 1000.
- **Option C (Incorrect):** The `range` parameter is a hard boundary — Matplotlib respects it exactly. It does NOT auto-extend to include out-of-range values. The whole purpose of the `range` parameter is to prevent this auto-extension.
- **Option D (Incorrect):** No `ValueError` is raised for out-of-range data. Exclusion is silent and expected behavior. `range` is a filter, not a validation constraint.

</details>

---

### 20. Question

A developer runs the following and examines the `patches` return value:

```python
counts, bins, patches = plt.hist([10, 20, 30, 40, 50], bins=5)
print(type(patches[0]))
```

What type does `patches[0]` return, and what can the developer do with it?

- [ ] **A.** `numpy.float64` — the numeric height of the first bar
- [ ] **B.** `matplotlib.patches.Rectangle` — a graphical artist representing the first bar, which can be styled with `set_facecolor()`, `set_edgecolor()`, etc.
- [ ] **C.** `matplotlib.lines.Line2D` — a line segment representing the bar outline
- [ ] **D.** `tuple` — containing `(left_edge, right_edge, height)` of the first bar

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The heights (frequency counts or densities) are in the `counts` array — the first return value. `patches` contains graphical artist objects, not numeric values.
- **Option B (Correct):** Each element of `patches` is a `matplotlib.patches.Rectangle` object — the actual graphical bar drawn on the axes. Since it is a full artist object, the developer can manipulate it directly: `patches[0].set_facecolor("gold")` changes the first bar's fill color; `patches[0].set_edgecolor("black")` adds a black edge. This enables powerful custom styling like coloring specific bars differently based on their position or value — e.g., highlighting the tallest bar or bars in a specific value range.
- **Option C (Incorrect):** `Line2D` objects represent lines (like those from `ax.plot()`). Histogram bars are rectangular patches, represented by `Rectangle` objects, not `Line2D`. The `histtype='step'` variant produces line-based rendering but still returns `Polygon` patches, not `Line2D`.
- **Option D (Incorrect):** `patches[0]` is a full artist object, not a simple tuple of coordinates. While the `Rectangle` object has methods to query its position (`get_x()`, `get_width()`, `get_height()`), `patches[0]` itself is a complex `Rectangle` artist, not a plain Python tuple.

</details>
