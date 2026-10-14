# Matplotlib Histogram Objective Test — Set 06

---

### 51. Question

A developer receives the following **broken code** from a colleague and must identify the bug:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(50, 10, 500)

# Bug is somewhere below
counts, bins = plt.hist(data, bins=15, density=True)
print(len(bins))  # Developer expects 16
```

What is the bug?

- [ ] **A.** `density=True` is incompatible with the tuple unpacking `counts, bins`
- [ ] **B.** `plt.hist()` returns **three** values `(counts, bins, patches)` — unpacking into only two variables raises a `ValueError`
- [ ] **C.** `bins=15` is invalid — must use explicit bin edges
- [ ] **D.** `len(bins)` returns 15, not 16, so the developer's expectation is wrong

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `density=True` is fully compatible with tuple unpacking. It only changes the values in `counts` from raw integers to float density values — it does not affect the return structure.
- **Option B (Correct):** `plt.hist()` always returns a **3-tuple**: `(counts, bins, patches)`. Unpacking into only two variables (`counts, bins = plt.hist(...)`) raises: `ValueError: too many values to unpack (expected 2)`. The fix is `counts, bins, patches = plt.hist(...)` or to use `_` to discard patches: `counts, bins, _ = plt.hist(...)`.
- **Option C (Incorrect):** `bins=15` (scalar integer) is perfectly valid. Matplotlib interprets it as "create 15 equal-width bins spanning [min(data), max(data)]".
- **Option D (Incorrect):** `len(bins)` correctly returns 16 for `bins=15` (N+1 = 15+1 = 16). The developer's expectation is correct. The bug is the unpacking error, not the length expectation.

</details>

---

### 52. Question

A developer computes **bin centers** from the returned `bins` array to plot a line over the histogram:

```python
counts, bins, patches = plt.hist(data, bins=10, density=True, alpha=0.5)
bin_centers = (bins[:-1] + bins[1:]) / 2

from scipy.stats import norm
x = np.linspace(bins[0], bins[-1], 200)
plt.plot(x, norm.pdf(x, np.mean(data), np.std(data)), color="red", linewidth=2)
```

What does `(bins[:-1] + bins[1:]) / 2` compute?

- [ ] **A.** The raw frequency counts — same as the `counts` array
- [ ] **B.** The midpoint x-coordinate of each bin: the average of the left and right edge of every bin
- [ ] **C.** The total width of all bins combined
- [ ] **D.** The cumulative sum of bin edges

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Frequency counts are in the `counts` array (first return value). `(bins[:-1] + bins[1:]) / 2` performs arithmetic on the bin edges — it never touches `counts`.
- **Option B (Correct):** `bins[:-1]` is an array of left edges `[e0, e1, …, e_{N-1}]`. `bins[1:]` is an array of right edges `[e1, e2, …, e_N]`. Adding them element-wise: `[e0+e1, e1+e2, …]`. Dividing by 2: `[(e0+e1)/2, (e1+e2)/2, …]` — these are the **midpoints** of each bin. Bin centers are useful for plotting a theoretical PDF curve at bin center positions, or for bar-chart-style annotations at each bin's center x-coordinate.
- **Option C (Incorrect):** Total combined width = `bins[-1] - bins[0]` = the full data range. Or per-bin widths = `np.diff(bins)`. Neither is what `(bins[:-1] + bins[1:]) / 2` computes.
- **Option D (Incorrect):** Cumulative sum of bin edges would be `np.cumsum(bins)` — a monotonically increasing array. The midpoint formula produces values in the same range as the original bin edges, not a cumulative sum.

</details>

---

### 53. Question

A developer needs to plot a histogram of **integer counts** representing dice rolls (values 1–6) and wants each face to have exactly one bar. Which call is correct?

```python
rolls = np.random.randint(1, 7, size=600)  # Values: 1, 2, 3, 4, 5, 6
```

- [ ] **A.** `plt.hist(rolls, bins=6)` — Matplotlib will automatically create one bin per integer
- [ ] **B.** `plt.hist(rolls, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])` — explicit half-integer edges guarantee one bar per face
- [ ] **C.** `plt.hist(rolls, bins=7)` — 7 bins are needed for 6 faces
- [ ] **D.** `plt.bar([1,2,3,4,5,6], np.bincount(rolls)[1:])` — bar chart is required; histograms cannot handle integers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.hist(rolls, bins=6)` creates 6 bins spanning [1, 6] with bin width = (6-1)/6 ≈ 0.833. This means bins like [1.0, 1.833), [1.833, 2.667) — each bin does NOT correspond exactly to one integer. Integers at boundary values (e.g., 2) might be shared between bins, producing distorted counts.
- **Option B (Correct):** Using **half-integer edges** `[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]` guarantees: bin 1 = [0.5, 1.5) captures only value 1; bin 2 = [1.5, 2.5) captures only value 2; …; bin 6 = [5.5, 6.5] captures only value 6. Each integer falls cleanly in the middle of its bin. This is the standard technique for histogramming discrete integer data.
- **Option C (Incorrect):** `bins=7` creates 7 equal-width bins spanning [1, 6], with width = 5/7 ≈ 0.714. This produces 7 non-integer-aligned bins — worse than `bins=6` for discrete integer data.
- **Option D (Incorrect):** `plt.bar()` is a valid alternative, but it's wrong to say histograms "cannot handle integers." `plt.hist()` with appropriate bin edges works correctly for integer data. Both approaches are valid; `plt.hist()` with half-integer edges is a standard pattern.

</details>

---

### 54. Question

A developer calculates the **expected density value** for the following scenario:

- Dataset: 200 samples
- Bin: `[20, 25)` — contains exactly 30 samples
- Bin width: 5

What is the probability density height of this bar when `density=True`?

- [ ] **A.** 30 (raw count)
- [ ] **B.** 0.15 (proportion: 30/200)
- [ ] **C.** 0.03 (density: 30 / (200 × 5))
- [ ] **D.** 6.0 (density: 30 / 5)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 30 is the raw count — what `density=False` (default) returns. With `density=True`, raw counts are transformed into density values.
- **Option B (Incorrect):** 0.15 is the **relative frequency** (proportion = 30/200 = 0.15). This is NOT the density. Relative frequency does not account for bin width. Density = relative_frequency / width = 0.15 / 5 = 0.03.
- **Option C (Correct):** Probability density = count / (total_samples × bin_width) = 30 / (200 × 5) = 30 / 1000 = **0.03**. Verification: total area = Σ(density × width) for all bins = 1.0. For this bin: 0.03 × 5 = 0.15 (the probability of a value falling in [20, 25)). All bins' probability contributions must sum to 1.0.
- **Option D (Incorrect):** 30/5 = 6.0 divides count by width but ignores the total sample size. This would give density values that sum to total_count/width instead of 1.0 — not a valid probability density.

</details>

---

### 55. Question

A developer creates a **stepfilled** histogram with low alpha to show feature importance overlap:

```python
important   = np.random.normal(0.7, 0.1, 300)
unimportant = np.random.normal(0.3, 0.15, 300)

plt.hist(important,   bins=20, histtype='stepfilled', alpha=0.4,
         color="green", label="Important Features")
plt.hist(unimportant, bins=20, histtype='stepfilled', alpha=0.4,
         color="gray",  label="Unimportant Features")
plt.xlabel("Feature Importance Score")
plt.legend()
```

What visual difference exists between `histtype='stepfilled'` and `histtype='step'`?

- [ ] **A.** `stepfilled` draws bars filled with color AND an outline; `step` draws only the outline with no fill
- [ ] **B.** `stepfilled` draws only a solid outline; `step` draws a dashed outline
- [ ] **C.** `stepfilled` normalizes to density; `step` does not
- [ ] **D.** `stepfilled` supports `alpha`; `step` does not support transparency

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `histtype='step'` draws only the **outline** (perimeter line) of the histogram shape — no fill color inside the bars. `histtype='stepfilled'` draws the same stepped outline PLUS a **solid fill** inside the bars. With `alpha=0.4`, the fill becomes semi-transparent, allowing distributions to be visually overlaid. The combination `histtype='stepfilled', alpha=0.4` creates filled semi-transparent bars — similar to the default `histtype='bar', alpha=0.4` but with a slightly different edge rendering.
- **Option B (Incorrect):** Neither `step` nor `stepfilled` produces dashed outlines by default. Dashing is controlled by `linestyle='--'` on the line, not by `histtype`.
- **Option C (Incorrect):** Neither `histtype` setting affects normalization. `density=True` is always the parameter that controls density normalization — completely independent of `histtype`.
- **Option D (Incorrect):** Both `step` and `stepfilled` support `alpha` transparency. The `alpha` parameter applies to any `histtype`. `step` with `alpha=0.5` produces a semi-transparent outline line.

</details>

---

### 56. Question

A developer generates a **multi-select question** scenario. Which of the following statements about `np.random.seed()` and histograms are **TRUE**? (Select all that apply)

```python
np.random.seed(42)
data1 = np.random.normal(0, 1, 500)

np.random.seed(42)
data2 = np.random.normal(0, 1, 500)

np.random.seed(99)
data3 = np.random.normal(0, 1, 500)
```

- [ ] **A.** `data1` and `data2` will produce identical histograms because they use the same seed
- [ ] **B.** `data3` will produce a different histogram shape from `data1` (for small n, seed affects shape)
- [ ] **C.** `data1` and `data3` will produce identical histograms because both are drawn from `N(0,1)`
- [ ] **D.** The random seed has no effect on `plt.hist()` rendering — only data values matter for the histogram

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:

- **Option A (Correct — TRUE):** `np.random.seed(42)` resets the PRNG to the same starting state. Generating 500 values from `N(0,1)` after seed 42 produces the exact same sequence both times. `data1 == data2` element-wise. Their histograms are identical.
- **Option B (Correct — TRUE):** Different seeds (42 vs 99) produce different pseudo-random sequences. For n=500, the sample is affected by random variability. The histogram shapes will be similar (both approximate `N(0,1)`) but not identical — bar heights will differ due to the different specific values generated. For small n, seed changes are visibly apparent in histogram shape.
- **Option C (Incorrect — FALSE):** Same distribution parameters do NOT guarantee identical histograms when seeds differ. The PRNG sequence is different for seed 99 vs seed 42, so `data3 ≠ data1`. Their histograms will be SIMILAR in shape (both Gaussian) but NOT identical.
- **Option D (Correct — TRUE):** `plt.hist()` is a purely deterministic function — given the same data array, it always produces the same histogram regardless of any random seed. The seed affects data GENERATION (numpy calls), not histogram RENDERING (matplotlib calls). Matplotlib has no internal randomness.

</details>

---

### 57. Question

A developer creates a histogram with `bins=[0, 1, 2, 5, 10, 20]` (non-uniform widths). Without `density=True`, how does the **visual area** of each bar relate to the **count** of data in that bin?

- [ ] **A.** The visual area of each bar equals the count — bars with more data always appear taller AND wider proportionally
- [ ] **B.** Without `density=True`, bar HEIGHT = count. Since bins have different widths, bars with equal counts but different widths have different AREAS — making visual area a misleading indicator of data frequency
- [ ] **C.** Matplotlib automatically adjusts bar heights for non-uniform bins to keep areas proportional
- [ ] **D.** Non-uniform bin edges are not supported without `density=True` — an error is raised

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Without `density=True`, bar height = count. But bars have DIFFERENT widths (1, 1, 3, 5, 10 for the given edges). A wide bin `[5, 10)` (width=5) with count=50 has area = height×width = 50×5 = 250, while a narrow bin `[0, 1)` (width=1) with count=50 has area = 50×1 = 50. Equal counts produce different visual areas — the wider bin LOOKS larger even though it contains the same number of data points.
- **Option B (Correct):** Without `density=True`, `plt.hist()` sets each bar's height to the raw count. For non-uniform bins, this creates a **misleading** visualization: wider bins appear to dominate visually (large area) even if they contain the same number of points as narrower bins. This is why `density=True` is recommended for non-uniform bin edges — it divides by bin width, making AREA proportional to data count and producing a visually honest representation.
- **Option C (Incorrect):** Matplotlib does NOT automatically adjust heights for non-uniform bins unless `density=True` is specified. The raw-count mode (`density=False`) always sets height = count, regardless of bin width.
- **Option D (Incorrect):** Non-uniform bin edges are fully supported without `density=True`. No error is raised. The result is just potentially misleading visually (wider bins appear disproportionately large).

</details>

---

### 58. Question

A developer wants to display a histogram **horizontally** for a comparison alongside a box plot:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ax1.boxplot(data, vert=True)
ax2.hist(data, bins=20, orientation='horizontal', color='steelblue')
ax2.set_xlabel("Frequency")
```

With `sharey=True` and `orientation='horizontal'`, what appears on the y-axis of the histogram subplot (`ax2`)?

- [ ] **A.** Frequency counts (bar lengths)
- [ ] **B.** The data values (the continuous variable being distributed)
- [ ] **C.** Bin numbers (1 through 20)
- [ ] **D.** Probability density values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In a horizontal histogram, frequency counts (bar lengths) appear on the **x-axis** — which is why the developer labeled `ax2.set_xlabel("Frequency")`. The x-axis shows how many data points fall in each horizontal bar.
- **Option B (Correct):** With `orientation='horizontal'`, the data values (the measured variable) are plotted on the **y-axis**, and bar length (frequency) appears on the x-axis. This is the transposed version of the standard vertical histogram. With `sharey=True`, both subplots share the same y-axis scale — the boxplot's y-axis (which shows data values) and the horizontal histogram's y-axis (which also shows data values) are aligned. This creates a powerful side-by-side visualization where the histogram shape and the boxplot's quartiles can be directly compared at the same y-scale.
- **Option C (Incorrect):** Bin numbers are not displayed on any axis. Axes always show either data values or frequencies — not ordinal bin indices.
- **Option D (Incorrect):** Probability density appears only when `density=True` is specified. Without it, the default shows raw frequency counts. And with horizontal orientation, density would be on the x-axis, not y-axis.

</details>

---

### 59. Question

A developer creates a histogram for **feature engineering diagnostics** and observes that a feature has a uniform distribution. What does this indicate for tree-based ML models?

```python
feature = np.random.uniform(0, 100, 1000)
plt.hist(feature, bins=20, edgecolor="black", color="teal")
plt.title("Feature Distribution")
```

- [ ] **A.** The feature is ideal — uniform features always lead to the best tree splits
- [ ] **B.** A uniform distribution means all values are equally probable; this provides moderate information gain for tree splits but may indicate the feature lacks natural clusters or patterns that correlate with the target
- [ ] **C.** The feature should be removed — uniform distributions always indicate random noise with no predictive value
- [ ] **D.** A uniform distribution triggers a Matplotlib bug where bars have incorrect heights

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Uniform features are not universally "ideal." While they provide splits at any threshold with roughly equal information, the USEFULNESS of a feature depends on its correlation with the target variable — not its marginal distribution shape.
- **Option B (Correct):** A uniform distribution (flat histogram with approximately equal bar heights across all bins) means the feature values are spread evenly across its range with no natural clustering or concentration. For tree-based models (Decision Trees, Random Forests, XGBoost): splits can be made at any threshold, providing gradual information gain. However, if the feature is uniformly distributed AND uncorrelated with the target, it adds no predictive value. The uniform distribution alone is a diagnostic signal — it motivates checking the feature's correlation with the target (mutual information, feature importance scores) rather than blindly removing or keeping it.
- **Option C (Incorrect):** Uniform distribution does NOT automatically mean random noise. Many meaningful features have approximately uniform distributions (e.g., sensor readings across a calibrated range, randomly assigned treatment dosages in experiments). The distribution shape alone cannot determine predictive value.
- **Option D (Incorrect):** Matplotlib handles uniform distributions correctly and accurately. A uniform distribution produces approximately equal-height bars, which is the mathematically correct representation. No bugs are triggered.

</details>

---

### 60. Question

A developer runs this code and is surprised by the output:

```python
np.random.seed(1)
data = np.array([1.0] * 100)  # 100 identical values

counts, bins, patches = plt.hist(data, bins=10)
print(counts)
print(bins)
```

What will the `counts` and `bins` arrays look like?

- [ ] **A.** `counts = [10, 10, 10, ...]` (uniform distribution across 10 bins) and `bins` spans [0, 1] in 10 equal steps
- [ ] **B.** `counts = [100, 0, 0, ...]` — all 100 values fall in the first bin because all values are identical (1.0), creating zero-width bins that Matplotlib handles by defaulting to a small range
- [ ] **C.** Matplotlib raises a `ValueError` because a zero-variance dataset (all identical values) cannot be binned
- [ ] **D.** `counts = [0, 0, ..., 100]` — all values fall in the last bin because 1.0 is treated as the maximum

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** All 100 values are exactly 1.0 — a zero-variance dataset. There is no spread to distribute uniformly across 10 bins. All values are identical and will fall into a single bin.
- **Option B (Correct):** When all values are identical (e.g., all = 1.0), `min(data) = max(data) = 1.0`. Matplotlib detects this zero-range condition and automatically adds a small default range offset (approximately ±0.5) around the single value to create a displayable histogram. With this artificial range, the bin containing 1.0 gets all 100 counts, while the other 9 bins are empty: `counts = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]`. The `bins` array spans approximately [0.5, 1.5] in 10 equal steps. This is Matplotlib's graceful handling of zero-variance data.
- **Option C (Incorrect):** Matplotlib does NOT raise a `ValueError` for zero-variance data. It uses a fallback range to ensure the histogram is still displayable. This is documented behavior.
- **Option D (Incorrect):** With the artificial range added by Matplotlib (e.g., [0.5, 1.5]), the value 1.0 falls in the middle of the range — typically in the 5th or 6th bin, not the last bin. The exact bin depends on Matplotlib's range padding logic.

</details>
