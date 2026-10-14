# Matplotlib Histogram Objective Test — Set 04

---

### 31. Question

A developer calls `plt.hist()` with `density=True` on the following dataset:

```python
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
counts, bins, patches = plt.hist(data, bins=2, density=True)
```

The two bins are `[1, 5.5)` and `[5.5, 10]`, each containing 5 data points. What is the density height of each bar?

- [ ] **A.** 5.0 for each bar (raw count)
- [ ] **B.** 0.5 for each bar (proportion of total)
- [ ] **C.** ≈0.1111 for each bar
- [ ] **D.** 1.0 for each bar (normalized to sum to 2)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 5.0 is the raw count per bin. `density=True` transforms raw counts into probability density values — not the same as count.
- **Option B (Incorrect):** 0.5 is the relative frequency (proportion) per bin: 5/10 = 0.5. But `density=True` produces **probability density**, not relative frequency. Density = count / (total × bin_width).
- **Option C (Correct):** Bin width = (10 - 1) / 2 = 4.5. Density = count / (total × width) = 5 / (10 × 4.5) = 5/45 ≈ **0.1111**. Verification: total area = 0.1111 × 4.5 + 0.1111 × 4.5 = 0.5 + 0.5 = 1.0 ✓. The density formula ensures the histogram integrates to 1 (total area = 1).
- **Option D (Incorrect):** 1.0 per bar would give total area = 1.0 × 4.5 + 1.0 × 4.5 = 9.0 ≠ 1. This is incorrect — density values must satisfy the area-equals-1 constraint.

</details>

---

### 32. Question

A developer creates a density histogram where **bar heights exceed 1.0**. A colleague insists this is a bug. Is the colleague correct?

```python
data = np.random.uniform(0, 0.5, 1000)  # Values between 0 and 0.5
counts, bins, patches = plt.hist(data, bins=10, density=True)
print(counts.max())  # Prints approximately 2.0
```

- [ ] **A.** Yes — probability density values can never exceed 1.0 because they represent probabilities
- [ ] **B.** No — density values CAN exceed 1.0 when bin widths are less than 1. The constraint is that total AREA equals 1, not that individual heights ≤ 1
- [ ] **C.** Yes — a value of 2.0 indicates an implementation error in Matplotlib
- [ ] **D.** No — density values always equal exactly 1.0 regardless of bin width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This confuses **probability** (always ≤ 1) with **probability density** (can exceed 1). Probability density is not a probability — it is a rate. The density value represents probability per unit width, not raw probability.
- **Option B (Correct):** The `density=True` constraint is: **total area = 1**, i.e., Σ(density_i × width_i) = 1. For uniform data on [0, 0.5] with bins=10: bin width = 0.05. Density = 1/0.5 = **2.0** (theoretical). Area per bin = 2.0 × 0.05 = 0.1. Total area = 2.0 × 10 × 0.05 = 1.0 ✓. Since width < 1, density must be > 1 to compensate and maintain area = 1. Individual bar heights can be ANY positive value — there is no upper bound of 1.0 on density.
- **Option C (Incorrect):** A density of 2.0 is mathematically correct for this dataset. The uniform distribution on [0, 0.5] has a PDF of f(x) = 1/(0.5-0) = 2 — so the histogram bars should be at height 2.0. This is expected behavior, not a bug.
- **Option D (Incorrect):** Density values do NOT always equal 1.0. The density value depends on the bin width and count. Only when all bins have width = 1.0 AND the data is uniform would each bar's density equal the proportion — but even then it wouldn't necessarily be 1.0.

</details>

---

### 33. Question

A developer verifies the `density=True` normalization manually:

```python
counts, bins, patches = plt.hist(data, bins=5, density=True)
bin_widths = np.diff(bins)
total_area = np.sum(counts * bin_widths)
print(total_area)
```

What should `total_area` print (within floating point precision)?

- [ ] **A.** The total number of data samples (e.g., 500)
- [ ] **B.** 1.0
- [ ] **C.** 0.5
- [ ] **D.** The number of bins (5)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The total sample count would be the sum of RAW (non-normalized) counts. With `density=True`, counts are normalized so the total area — not the total count — equals 1.
- **Option B (Correct):** `np.diff(bins)` computes bin widths (the differences between adjacent edges). `counts * bin_widths` computes the area of each bar (height × width = density × width). `np.sum(...)` sums all areas. By definition of `density=True` normalization, this total area always equals **1.0** (within floating-point precision, e.g., 0.9999999999). This is the mathematical verification of the density normalization guarantee.
- **Option C (Incorrect):** 0.5 would be the result if only half the density bars were summed, or if the normalization was to 0.5 rather than 1.0. The integral of a valid probability density function is 1.0, not 0.5.
- **Option D (Incorrect):** The total area has no relationship to the number of bins. Changing `bins=5` to `bins=50` still yields a total area of 1.0 — the normalization redistributes density values to maintain the area constraint.

</details>

---

### 34. Question

A developer calculates density values from raw counts manually:

```python
# Dataset: 50 samples, 3 bins of equal width 10
# Raw counts: [8, 30, 12]
# Total samples: 50, bin width: 10
```

What are the **density values** for the three bins?

- [ ] **A.** `[0.16, 0.60, 0.24]` (relative frequencies)
- [ ] **B.** `[0.016, 0.060, 0.024]` (density = count / (total × width))
- [ ] **C.** `[8, 30, 12]` (unchanged raw counts)
- [ ] **D.** `[0.08, 0.30, 0.12]` (density = count / total)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `[0.16, 0.60, 0.24]` are the **relative frequencies** (proportions): 8/50=0.16, 30/50=0.60, 12/50=0.24. These are NOT density values. Relative frequency would require bin width=1 to equal density.
- **Option B (Correct):** Density = count / (total × width). Bin 1: 8/(50×10) = 8/500 = **0.016**. Bin 2: 30/(50×10) = 30/500 = **0.060**. Bin 3: 12/(50×10) = 12/500 = **0.024**. Verification: total area = (0.016 + 0.060 + 0.024) × 10 = 0.1 × 10 = **1.0** ✓.
- **Option C (Incorrect):** Raw counts [8, 30, 12] are what `density=False` (default) returns. `density=True` transforms these into density values by dividing by (total × width).
- **Option D (Incorrect):** `count/total` gives relative frequency (probability per bin), not density. Density further divides by bin width to account for bin size: density = (count/total) / width = relative_frequency / width.

</details>

---

### 35. Question

A developer calls `plt.hist(data, bins=20, cumulative=True)` WITHOUT `density=True`. What does each bar height represent?

- [ ] **A.** The fraction of data points with values less than or equal to that bin's upper edge (values from 0 to 1)
- [ ] **B.** The total count of all data points with values ≤ that bin's upper edge (cumulative frequency count)
- [ ] **C.** The probability density up to that bin's upper edge
- [ ] **D.** The count in just that bin (same as `cumulative=False`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Values from 0 to 1 (fractions) would require `density=True, cumulative=True` together. Without `density=True`, the y-axis shows raw cumulative counts, not fractions.
- **Option B (Correct):** `cumulative=True` without `density=True` produces a **cumulative frequency histogram** where each bar's height = the total number of data points with values ≤ that bin's right edge. The first bar = count in bin 1. The second bar = count in bins 1+2. The last bar = total sample size (all data points). The y-axis maximum equals `len(data)`. This visualizes the empirical cumulative distribution in raw count form.
- **Option C (Incorrect):** Probability density requires `density=True`. Without it, bar heights are counts (integers), not densities (floats representing rates).
- **Option D (Incorrect):** `cumulative=True` fundamentally changes bar height semantics. Without cumulation, each bar shows count within that bin only. With cumulation, each bar shows the running total from the leftmost bin through the current bin.

</details>

---

### 36. Question

A developer uses `cumulative=True, density=True` together:

```python
plt.hist(data, bins=30, cumulative=True, density=True, histtype="step")
plt.xlabel("Value")
plt.ylabel("Cumulative Probability")
```

What does the resulting chart show, and what is special about the final (rightmost) bar?

- [ ] **A.** The survival function (1 - CDF); the final bar reaches 0.0
- [ ] **B.** The empirical cumulative distribution function (ECDF); the final bar always reaches exactly 1.0
- [ ] **C.** A normalized frequency chart; the final bar reaches the total sample count
- [ ] **D.** A reverse CDF; bars decrease from left to right

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The survival function (1-CDF) would be obtained with `cumulative=-1` — the reverse cumulative option. With `cumulative=True` (positive), the chart shows the forward CDF increasing from left to right.
- **Option B (Correct):** `cumulative=True, density=True` together produce the **empirical cumulative distribution function (ECDF)**. Each bar height represents the estimated probability P(X ≤ x) for x equal to that bin's right edge. The bars increase monotonically from 0 (leftmost, where no data has been accumulated yet) to **exactly 1.0** at the rightmost bar (all data has been accumulated). This is the defining property of a CDF: it must reach 1.0 at the maximum data value.
- **Option C (Incorrect):** Reaching the total sample count would occur with `cumulative=True` but `density=False`. Adding `density=True` normalizes to [0, 1], so the final value is 1.0, not the sample count.
- **Option D (Incorrect):** `cumulative=True` (positive value) produces an increasing function (forward CDF). A decreasing reverse CDF would require `cumulative=-1`.

</details>

---

### 37. Question

A developer uses `cumulative=-1` in `plt.hist()`. What does this produce?

- [ ] **A.** A histogram with bars sorted in descending frequency order
- [ ] **B.** The survival function / exceedance probability: each bar height = P(X > x) for that bin's upper edge — a decreasing function from left to right
- [ ] **C.** An error — `cumulative` only accepts `True` or `False`
- [ ] **D.** The same output as `cumulative=True` but mirrored around the y-axis

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Sorting bars by frequency is not related to the `cumulative` parameter. Bar order always follows the data's numeric range (left to right), not frequency magnitude.
- **Option B (Correct):** `cumulative=-1` produces the **reverse cumulative** (survival function). Each bar height shows the count (or probability, if `density=True`) of data points with values **greater than** that bin's upper edge. The function decreases monotonically from left to right: the leftmost bar has the highest value (almost all data exceeds the minimum value) and the rightmost bar approaches 0 (almost no data exceeds the maximum value). With `density=True`, the leftmost value approaches 1.0 and rightmost approaches 0.0.
- **Option C (Incorrect):** `cumulative=-1` is valid documented Matplotlib behavior. The parameter accepts `True` (forward cumulation), `False` (no cumulation), and `-1` (reverse cumulation).
- **Option D (Incorrect):** Mirroring around the y-axis would flip the x-axis direction. `cumulative=-1` keeps the same x-axis order but plots the complementary (1-CDF) function — starting high on the left and decreasing to zero on the right.

</details>

---

### 38. Question

A developer creates overlapping histograms for a model comparison and uses `histtype='step'` instead of `alpha`:

```python
plt.hist(model_a_errors, bins=25, histtype='step', color='blue', label='Model A')
plt.hist(model_b_errors, bins=25, histtype='step', color='red', label='Model B')
plt.legend()
```

What is a key **advantage** of `histtype='step'` over `alpha=0.5` for multi-distribution comparison?

- [ ] **A.** `histtype='step'` always produces a smoother visual output
- [ ] **B.** Step outlines never overlap visually — each distribution's shape is fully visible without one obscuring the other, even when distributions have the same range
- [ ] **C.** `histtype='step'` is faster to render for large datasets
- [ ] **D.** Step histograms automatically normalize to density

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `histtype='step'` produces the same rectangular stepped shape as `histtype='bar'` — it is not smoother. Smoothness requires KDE, not a different histogram type.
- **Option B (Correct):** With `histtype='step'`, each histogram is drawn as an unfilled outline (perimeter only). Since there is NO fill color — just a line — multiple step histograms can be overlaid without any distribution being hidden. Even when distribution A and distribution B fully overlap, both outlines are visible simultaneously. With `alpha=0.5` filled bars, one distribution's semi-transparent fill can still partially obscure the other, especially in dense overlap regions. Step outlines provide the clearest visual separation for distribution comparison.
- **Option C (Incorrect):** Rendering speed depends on the backend and the number of elements, not significantly on `histtype`. For large datasets, the number of bins (patches) is the same regardless of fill type.
- **Option D (Incorrect):** `histtype='step'` does NOT affect normalization. Density normalization is controlled by `density=True` — completely independent of `histtype`.

</details>

---

### 39. Question

A developer computes cumulative counts for quality control:

```python
# Defect counts per 5-unit inspection interval:
raw_counts = [4, 9, 15, 8, 6, 3]
```

What are the **cumulative counts** for all 6 intervals?

- [ ] **A.** `[4, 13, 28, 36, 42, 45]`
- [ ] **B.** `[4, 9, 15, 8, 6, 3]` (unchanged)
- [ ] **C.** `[4, 13, 22, 30, 36, 45]`
- [ ] **D.** `[45, 41, 32, 17, 9, 3]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Cumulative counts are computed as running totals: Interval 1: 4; Interval 2: 4+9=13; Interval 3: 13+15=28; Interval 4: 28+8=36; Interval 5: 36+6=42; Interval 6: 42+3=**45**. Result: `[4, 13, 28, 36, 42, 45]`. The final cumulative value (45) equals the total sample count — verifying correctness.
- **Option B (Incorrect):** The raw counts unchanged represent a non-cumulative histogram. `cumulative=True` transforms these into running totals, not keeps them static.
- **Option C (Incorrect):** `[4, 13, 22, 30, 36, 45]` has an error at interval 3: 13+15=28, not 22. The subsequent values are also wrong because of this error propagating forward.
- **Option D (Incorrect):** `[45, 41, 32, 17, 9, 3]` is the **reverse cumulative** (survival function) — obtained with `cumulative=-1`. It starts at the total (45) and subtracts each count going left to right, reaching the last bin's count (3) at the end.

</details>

---

### 40. Question

A developer wants to compare the error distributions of two models that have **very different sample sizes** (Model A: 10,000 samples; Model B: 500 samples) on the same histogram. Which approach gives a **fair visual comparison**?

- [ ] **A.** `plt.hist(a, bins=30); plt.hist(b, bins=30)` — raw counts side-by-side
- [ ] **B.** `plt.hist([a, b], bins=30, density=True)` — density normalization makes both distributions comparable regardless of sample size
- [ ] **C.** Subsample Model A down to 500 samples before plotting
- [ ] **D.** Use separate figures for each model — overlaying distributions of different sizes is always misleading

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** With raw counts, Model A bars reach heights up to ~2000+ while Model B bars reach ~50. The Model B distribution is visually dwarfed by Model A — the comparison is misleading. A small Model B bar looks flat and featureless next to tall Model A bars even if the shapes are similar.
- **Option B (Correct):** `density=True` normalizes both distributions so their total areas equal 1.0, regardless of the original sample sizes. This makes the **shape** of each distribution directly comparable — tall bars indicate high relative density, not just large sample count. The `[a, b]` list syntax draws both on the same axes with `density=True` applied to each independently. This is the correct tool for comparing distributions with vastly different N.
- **Option C (Incorrect):** Subsampling Model A to 500 discards 95% of the data and introduces additional sampling variability into what should be a well-estimated distribution. `density=True` achieves the comparison goal without discarding information.
- **Option D (Incorrect):** Overlaying distributions of different sample sizes is NOT inherently misleading when `density=True` is used. Separate figures make direct comparison harder, not easier. The key is proper normalization, not separation.

</details>
