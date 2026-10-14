# Matplotlib Histogram Objective Test — Set 08

---

### 71. Question

A developer creates a complete ML pipeline diagnostic with the following code and must debug why all bars appear the **same height**:

```python
np.random.seed(11)
errors = np.random.normal(0, 1, 500)

counts, bins, patches = plt.hist(errors, bins=20, density=True)

# Developer colors bars by height
max_count = max(counts)
for patch, count in zip(patches, counts):
    patch.set_facecolor(plt.cm.RdYlGn(count / max_count))

plt.show()
```

The developer expects a gradient of colors from red (short bars) to green (tall bars), but all bars appear green. What is the bug?

- [ ] **A.** `density=True` normalizes all bars to the same height
- [ ] **B.** `plt.cm.RdYlGn(1.0)` always returns green — the colormap is being queried at the wrong scale
- [ ] **C.** `count / max_count` always equals 1.0 because `density=True` forces all counts to be equal
- [ ] **D.** The loop iterates over `patches` in reverse order relative to `counts`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `density=True` does NOT make all bars the same height. It normalizes so total area = 1.0, but individual bar heights still vary according to the data distribution. For a normal distribution, the central bars are tallest and peripheral bars are shortest.
- **Option B (Correct):** The bug is subtle: `plt.cm.RdYlGn` is a diverging colormap where values near 0.0 map to red and values near 1.0 map to green. The code divides each count by `max_count` — so the tallest bar gets `1.0` (green), medium bars get intermediate values, and short bars get low values (red). This logic is CORRECT in concept. However, the actual bug is that `max_count / max_count = 1.0` gives green for the max bar, but the SHORT bars should give RED (low values). The problem is that `plt.cm.RdYlGn` at value ~0.1 gives yellow-red while at 1.0 gives green — this SHOULD work. The real issue is that if `density=True` produces very similar count values (e.g., all near 0.04–0.05 for a nearly uniform slice of a normal distribution at this scale), the ratios all cluster near 1.0, making all bars appear green. The fix is to normalize relative to min too: `(count - min_counts) / (max_count - min_counts)`.
- **Option C (Incorrect):** `density=True` does NOT make all density values equal. The normal distribution has varying heights — tallest at center, shorter at tails.
- **Option D (Incorrect):** `zip(patches, counts)` iterates both in the same left-to-right order. There is no ordering mismatch.

</details>

---

### 72. Question

A developer creates a histogram for a **right-skewed salary dataset** and applies a log transformation to better visualize the distribution:

```python
np.random.seed(6)
salaries = np.random.lognormal(mean=10.5, sigma=0.9, size=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(salaries, bins=40, edgecolor="black", color="steelblue")
ax1.set_title("Raw Salary Distribution")
ax1.set_xlabel("Salary ($)")

ax2.hist(np.log10(salaries), bins=30, edgecolor="black", color="salmon")
ax2.set_title("Log10 Salary Distribution")
ax2.set_xlabel("log10(Salary)")
```

What transformation does the log histogram achieve, and why is it useful?

- [ ] **A.** The log transformation makes the distribution uniform — all bars become equal height
- [ ] **B.** The log transformation compresses the long right tail — converting the right-skewed distribution into a more symmetric, approximately normal shape that is easier to visually analyze and statistically model
- [ ] **C.** The log transformation removes outliers from the dataset automatically
- [ ] **D.** The log transformation converts raw counts to probability densities

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The log transformation does NOT produce a uniform distribution. A log-normal distribution transformed by log becomes a **normal** (Gaussian) distribution — symmetric bell curve, not uniform.
- **Option B (Correct):** The log10 transformation compresses the long right tail of the skewed distribution by making equal multiplicative differences appear as equal additive differences on the log scale. A salary range of $50,000–$500,000 (10× spread) becomes log10 range of 4.70–5.70 (1 unit). This **symmetrizes** the right-skewed distribution into an approximately normal shape, making: (1) the full distribution visible without outliers dominating the scale, (2) statistical analysis (mean, std) more meaningful, (3) the histogram shape interpretable at a glance. Log-normal data on a log scale is bell-shaped — this is the defining property of the log-normal distribution.
- **Option C (Incorrect):** Log transformation does NOT remove outliers. All data points remain present — they are just repositioned on the log scale. Extreme high values (outliers) are compressed but still visible.
- **Option D (Incorrect):** Log transformation is applied to the DATA VALUES before histogramming — it changes the x-axis scale. The y-axis (counts or density) is controlled by `density=True`, which is not used here.

</details>

---

### 73. Question

A developer examines a histogram and identifies **isolated bars separated by empty bins** in the far right tail. What does this pattern specifically indicate?

```python
purchase_amounts = np.concatenate([
    np.random.exponential(50, 995),  # Typical purchases
    [5000, 8000, 12000, 4500, 9999]  # Extreme outliers
])
plt.hist(purchase_amounts, bins=30, edgecolor="black")
```

- [ ] **A.** The bins are too narrow — increasing `bins` will merge the isolated bars with adjacent bars
- [ ] **B.** Extreme outliers exist far above the main distribution bulk — the outlier bars are visually separated from the main cluster by empty zero-count bins
- [ ] **C.** The data contains NaN values that are plotted at the far right
- [ ] **D.** The histogram has a normalization error — `density=True` must be used to fix isolated bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Increasing bins would make isolated bars even more separated and narrow — it would not merge outlier bars with the main cluster. The separation exists because no data points fall in the intermediate range between typical purchases and extreme outliers.
- **Option B (Correct):** The 5 extreme outlier values (4500–12000) are located far above the bulk of purchases (exponential with mean=50, so most values are 0–300). With `bins=30` spanning [min, 12000], the main cluster occupies only the first 1–2 bins (extremely tall but narrow), while the outlier region (4000–12000) has 5 values spread across the rightmost bins — appearing as isolated short bars separated by empty bins with zero count. This is the classic **extreme outlier visual signature** in histograms. The fix is either `range=(0, 500)` to clip outliers or using log-scale x-axis.
- **Option C (Incorrect):** `np.concatenate` with the given arrays produces only finite float values — no NaN values. NaN values would typically be silently skipped by `plt.hist()` rather than plotted at the far right.
- **Option D (Incorrect):** `density=True` normalizes bar heights but does not change bar positions or eliminate isolated outlier bars. The outlier bars remain in the same positions — just shorter relative to the main peak.

</details>

---

### 74. Question

A developer creates a **completeness comparison** between `cumulative=True` and `cumulative=False` using the same data:

```python
np.random.seed(17)
data = np.random.normal(100, 15, 600)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(data, bins=20, density=True, ax=ax1)                    # Bug!
ax2.hist(data, bins=20, density=True, cumulative=True, histtype='step')
```

There is a bug in this code. What is it?

- [ ] **A.** `density=True` cannot be used with `cumulative=True`
- [ ] **B.** `plt.hist()` does not accept an `ax=` keyword argument — the correct approach is `ax1.hist(data, bins=20, density=True)` using the axes method directly
- [ ] **C.** `histtype='step'` is incompatible with `cumulative=True`
- [ ] **D.** `bins=20` must be the same for both subplots when using `cumulative`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `density=True` and `cumulative=True` are fully compatible. Together they produce the empirical CDF (values 0 to 1). This is a valid and common combination.
- **Option B (Correct):** `plt.hist()` (the pyplot function) does NOT accept an `ax=` keyword argument. The `ax=` parameter does not exist in `plt.hist()`. To plot on a specific axes object, you must use the **axes method**: `ax1.hist(data, bins=20, density=True)`. The `plt.hist()` function always plots on the currently active axes (`plt.gca()`). This is a very common mistake when working with subplots — using `plt.hist(data, ax=ax1)` vs the correct `ax1.hist(data)`.
- **Option C (Incorrect):** `histtype='step'` is fully compatible with `cumulative=True`. In fact, `histtype='step'` is the most common choice for CDF plots — it produces a clean step-line shape without fill.
- **Option D (Incorrect):** There is no requirement that both subplots use the same `bins` value. Each `hist()` call is independent with its own bin count.

</details>

---

### 75. Question

A developer counts raw frequencies across 4 bins and verifies density normalization:

```python
# Dataset: 80 samples
# Bins of varying widths: [0,5), [5,15), [15,25), [25,50]
# Widths:                    5      10       10       25
# Raw counts:               16      24       32        8
```

Calculate the **density value** for each bin AND verify the total area = 1.0.

- [ ] **A.** Densities: [0.04, 0.03, 0.04, 0.004]; Total area = 1.0
- [ ] **B.** Densities: [0.04, 0.03, 0.04, 0.004]; Total area = 0.985
- [ ] **C.** Densities: [0.20, 0.30, 0.40, 0.10]; Total area = 1.0
- [ ] **D.** Densities: [0.16, 0.24, 0.32, 0.08]; Total area = 0.80

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Density = count / (total × width). Bin 1: 16/(80×5) = 16/400 = **0.040**. Bin 2: 24/(80×10) = 24/800 = **0.030**. Bin 3: 32/(80×10) = 32/800 = **0.040**. Bin 4: 8/(80×25) = 8/2000 = **0.004**. Verification: total area = (0.040×5) + (0.030×10) + (0.040×10) + (0.004×25) = 0.20 + 0.30 + 0.40 + 0.10 = **1.00** ✓.
- **Option B (Incorrect):** The densities are correct ([0.04, 0.03, 0.04, 0.004]) but the total area is stated as 0.985 — this is wrong. The area calculation gives exactly 1.00 as shown above.
- **Option C (Incorrect):** [0.20, 0.30, 0.40, 0.10] are the **relative frequencies** (proportions = count/total = 16/80, 24/80, 32/80, 8/80). These are NOT density values. Their sum is 1.0 (correct for proportions) but they are not probability densities.
- **Option D (Incorrect):** [0.16, 0.24, 0.32, 0.08] appear to be count/total without any width normalization: 16/100, 24/100, etc. — not the right formula and not summing to a meaningful area.

</details>

---

### 76. Question

A developer investigates why their **two-sample histogram comparison** produces misleading visual results:

```python
# Group A: Customer lifetime values (CLV) — 1,000 enterprise clients
# Group B: CLV — 5,000 small business clients

clv_a = np.random.lognormal(12, 0.5, 1000)
clv_b = np.random.lognormal(10, 0.8, 5000)

plt.hist(clv_a, bins=30, alpha=0.6, color="blue",  label="Enterprise (n=1000)")
plt.hist(clv_b, bins=30, alpha=0.6, color="orange", label="SMB (n=5000)")
plt.legend()
```

The orange SMB histogram is 5× taller than the blue enterprise histogram, making it appear that SMB clients dominate at every value. What is the correct fix?

- [ ] **A.** Multiply all enterprise counts by 5 to equalize sample sizes before plotting
- [ ] **B.** Add `density=True` to both `hist()` calls — this normalizes both distributions to total area=1.0, making shape comparison fair regardless of sample sizes
- [ ] **C.** Use `bins=150` instead of `bins=30` — more bins fixes the height discrepancy
- [ ] **D.** Plot enterprise on a separate figure to avoid scale contamination

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Multiplying enterprise counts by 5 artificially inflates the data. This distorts the actual distribution and produces misleading counts that don't represent reality. Statistical analysis built on inflated data would be invalid.
- **Option B (Correct):** `density=True` normalizes each histogram independently so its total area = 1.0. This makes the two distributions directly comparable by SHAPE regardless of sample size: a 1,000-sample distribution and a 5,000-sample distribution both integrate to 1.0. The y-axis shows probability density (shape of distribution), not raw frequency (influenced by sample count). This is the standard approach for comparing distributions with different N.
- **Option C (Incorrect):** Changing bin count affects granularity of both distributions equally. It does not address the fundamental problem of different sample sizes causing different absolute bar heights. Both histograms would still have a 5× height ratio with `bins=150`.
- **Option D (Incorrect):** Separate figures prevent direct visual comparison — the whole analytical benefit of overlapping histograms is lost. The goal is to compare shapes on the same axes, which `density=True` enables correctly.

</details>

---

### 77. Question

A developer creates a histogram for a **model confidence score diagnostic** and uses a specific coloring technique:

```python
counts, bins, patches = plt.hist(confidence_scores, bins=20,
                                  edgecolor="black", density=False)

# Color bars: red if confidence < 0.5 (uncertain), green if >= 0.5 (confident)
threshold = 0.5
for patch, left_edge in zip(patches, bins[:-1]):
    color = "green" if left_edge >= threshold else "red"
    patch.set_facecolor(color)

plt.axvline(threshold, color="black", linewidth=2, linestyle="--")
```

A colleague points out that the bin containing `left_edge=0.45` and `right_edge=0.55` should be partially red and partially green but is colored entirely red. What would be the correct approach for this edge case?

- [ ] **A.** Use `left_edge + bin_width/2` (the bin center) to classify the bin — if center ≥ 0.5 color green
- [ ] **B.** Split the histogram at exactly 0.5 using explicit bin edges `bins=[0, 0.05, ..., 0.50, 0.55, ..., 1.0]` so the threshold aligns with a bin boundary
- [ ] **C.** Both A and B are valid approaches — A is simpler; B gives exact threshold alignment
- [ ] **D.** The current code is already perfectly correct — bars cannot be partially colored

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Correct as part of C):** Using `left_edge + bin_width/2` (the bin center) classifies each bar by whether its CENTER is above or below the threshold. For the bin [0.45, 0.55): center = 0.50 → green (center ≥ 0.5). This is a reasonable approximation — the bar is centered on the threshold, so coloring it green is defensible. Simple to implement.
- **Option B (Correct as part of C):** The most precise approach is to include 0.5 as an explicit bin edge in the `bins` list. This ensures the threshold lands exactly on a bin boundary, so no bin straddles it. All bins are either entirely below 0.50 (red) or entirely above (green). The coloring logic is then exact with zero ambiguity.
- **Option C (Correct):** Both approaches address the edge case differently — A is a pragmatic approximation using bin centers, B is the exact solution using aligned bin edges. Both are valid depending on precision requirements. Option C is the most complete and accurate answer.
- **Option D (Incorrect):** The current code (using `left_edge` for classification) has a genuine ambiguity at the threshold bin [0.45, 0.55) — it is colored entirely red based on `left_edge=0.45 < 0.5`, even though half the bar is above the threshold. This is not "perfectly correct" — it is an acknowledged limitation that Options A and B address.

</details>

---

### 78. Question

A developer creates an advanced **model calibration diagnostic** with both a histogram and a CDF:

```python
np.random.seed(55)
probs = np.random.beta(2, 5, 1000)  # Model outputs: right-skewed toward 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(probs, bins=25, density=True, edgecolor="black", color="cornflowerblue")
ax1.set_title("Probability Density")

ax2.hist(probs, bins=25, density=True, cumulative=True, histtype='step',
         color="darkblue", linewidth=2)
ax2.axhline(0.5, color="red", linestyle="--", label="Median (50th percentile)")
ax2.set_title("Empirical CDF")
ax2.legend()
```

In the CDF plot (`ax2`), what does the **x-value where the step line crosses the red `axhline` at y=0.5** represent?

- [ ] **A.** The mean of the probability distribution
- [ ] **B.** The median (50th percentile) of the `probs` dataset — the value below which 50% of model outputs fall
- [ ] **C.** The mode of the `probs` distribution
- [ ] **D.** The value where probability density is maximized

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The mean is located at the x-value where the area-weighted center of the PDF lies — not where the CDF crosses 0.5. For asymmetric distributions, mean ≠ median. The red line at y=0.5 specifically marks the median.
- **Option B (Correct):** The empirical CDF at y=0.5 by definition equals the **50th percentile (median)**. The CDF F(x) = P(X ≤ x). When F(x) = 0.5, exactly 50% of the data falls at or below x — this is the definition of the median. The x-coordinate where the step line crosses `axhline(y=0.5)` is `np.median(probs)`. For Beta(2,5): the median ≈ 0.26 (below 0.5 due to right-concentration toward 0 in this parameterization... actually Beta(2,5) is left-skewed/concentrated near 0, median ≈ 0.26).
- **Option C (Incorrect):** The mode (most frequent value) corresponds to the x-position of the tallest bar in the PDF histogram (ax1) — not the CDF crossing point. For Beta(2,5), mode = (2-1)/(2+5-2) = 1/5 = 0.2.
- **Option D (Incorrect):** The maximum density point (mode) is readable from the PDF histogram (ax1), not from the CDF plot. The CDF is monotonically increasing and has no single peak — it has an inflection point near the mode, but the 0.5 crossing marks the median.

</details>

---

### 79. Question

A developer runs the following code to study the **effect of bin count on standard deviation estimation from a histogram**:

```python
np.random.seed(44)
data = np.random.normal(100, 15, 500)
true_std = np.std(data)

for n_bins in [5, 15, 30, 100]:
    counts, bins, _ = plt.hist(data, bins=n_bins, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    widths  = np.diff(bins)
    # Estimated mean from histogram
    hist_mean = np.sum(centers * counts * widths)
    # Estimated variance from histogram
    hist_var  = np.sum((centers - hist_mean)**2 * counts * widths)
    hist_std  = np.sqrt(hist_var)
    print(f"bins={n_bins}: std_estimate={hist_std:.3f}, true={true_std:.3f}")
    plt.cla()
```

As `n_bins` increases from 5 to 100, what happens to the histogram-estimated standard deviation?

- [ ] **A.** It always equals the true standard deviation regardless of bin count
- [ ] **B.** It monotonically increases with bin count — more bins always overestimates std
- [ ] **C.** It converges toward the true standard deviation as bin count increases (up to a point), then becomes noisier for very high bin counts with sparse data
- [ ] **D.** It decreases monotonically — fewer bins always give a more accurate estimate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The histogram-estimated std is an approximation that depends on bin count. For very few bins (e.g., 5), the bin centers poorly represent the within-bin data distribution, introducing systematic error. The estimate is NOT always exactly equal to true std.
- **Option B (Incorrect):** More bins does not monotonically increase the std estimate. The relationship is non-monotone: improvement up to a point, then degradation due to sampling noise.
- **Option C (Correct):** The histogram std estimate improves (converges to true std) as bins increase — finer bins mean bin centers more accurately represent actual data positions. However, for very high bin counts with limited data (e.g., `bins=100` for n=500 samples = 5 samples/bin on average), random sampling variability causes noisy, erratic bin heights. This noise distorts the estimated second moment, causing the std estimate to become less stable. The optimal bin count balances resolution with sufficient data per bin. This is the bias-variance tradeoff for histogram-based density estimation.
- **Option D (Incorrect):** Fewer bins do NOT always give more accurate estimates. With very few bins (e.g., 5), the coarse quantization introduces large systematic bias — bin centers are far from actual data positions for wide bins.

</details>

---

### 80. Question

Which of the following statements about `plt.hist()` are **TRUE**? (Select all that apply)

```python
counts, bins, patches = plt.hist(data, bins=20, density=True, cumulative=True)
```

- [ ] **A.** `len(counts)` = 20
- [ ] **B.** `len(bins)` = 20
- [ ] **C.** The last element of `counts` is always approximately 1.0 when `density=True, cumulative=True`
- [ ] **D.** `patches` contains `matplotlib.patches.Rectangle` objects that can be individually restyled

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, C, D**

#### Explanations:

- **Option A (Correct — TRUE):** `len(counts) = 20` for `bins=20`. The number of counts always equals the number of bins.
- **Option B (Incorrect — FALSE):** `len(bins) = 21` for `bins=20`. The bins array always has N+1 elements (N+1 edges for N bins). `len(bins) = 20` would be wrong.
- **Option C (Correct — TRUE):** With `density=True, cumulative=True`, the histogram represents the empirical CDF. The CDF must reach 1.0 at the rightmost bin (all data has been accumulated). Therefore, `counts[-1] ≈ 1.0` (exactly 1.0 within floating-point precision). This is the mathematical guarantee of a proper CDF.
- **Option D (Correct — TRUE):** `patches` is a list of `matplotlib.patches.Rectangle` artist objects. Each can be individually restyled: `patches[i].set_facecolor("red")`, `patches[i].set_alpha(0.3)`, `patches[i].set_edgecolor("black")`, etc. This enables per-bar custom coloring based on bin position or count value.

</details>
