# Matplotlib Histogram Objective Test — Set 07

---

### 61. Question

A developer creates a histogram diagnostic for **feature distribution shift** between training and production data:

```python
np.random.seed(7)
train_feature = np.random.normal(50, 10, 2000)
prod_feature  = np.random.normal(65, 12, 2000)  # Shifted in production

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(train_feature, bins=30, alpha=0.55, color="steelblue", label="Training")
ax.hist(prod_feature,  bins=30, alpha=0.55, color="tomato",    label="Production")
ax.axvline(np.mean(train_feature), color="navy",   linestyle="--", label="Train Mean")
ax.axvline(np.mean(prod_feature),  color="darkred",linestyle="--", label="Prod Mean")
ax.legend()
ax.set_title("Feature Distribution: Train vs Production")
```

The production mean is 15 units higher than the training mean. What is this phenomenon called and what ML risk does it pose?

- [ ] **A.** Overfitting — the model memorized training data and cannot generalize
- [ ] **B.** Data leakage — production data is contaminating the training pipeline
- [ ] **C.** Dataset shift (covariate shift) — the input feature distribution changed between training and deployment, likely causing model performance degradation
- [ ] **D.** Class imbalance — the positive class is underrepresented in production

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Overfitting describes a model that performs well on training data but poorly on test data from the SAME distribution. Distribution shift is a different problem — the deployment distribution itself has changed.
- **Option B (Incorrect):** Data leakage means future/target information accidentally appears in training features. The histogram shows distribution shift (different means) — not leakage. Leakage would typically make the training distribution look artificially good, not shift the mean.
- **Option C (Correct):** When the feature distribution in production differs significantly from training (here: mean shifts from 50 to 65, a 1.5σ shift), this is **covariate shift** — a form of **dataset shift**. The model was trained to make predictions based on input distributions centered at 50; when production inputs are consistently higher (centered at 65), the model encounters input regions it rarely saw during training, leading to degraded prediction quality. The histogram overlay with mean reference lines is the standard visual diagnostic for detecting this drift.
- **Option D (Incorrect):** Class imbalance refers to the ratio of positive to negative samples in the target variable — not the distribution of input features. The histogram here shows a continuous INPUT feature, not class labels.

</details>

---

### 62. Question

A developer computes the following and asks what the output will be:

```python
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
counts, bins, _ = plt.hist(data, bins=5, density=False)
print(counts)
print(np.sum(counts))
```

What does `np.sum(counts)` equal?

- [ ] **A.** 1.0 (always — by normalization)
- [ ] **B.** 10 (equal to the total number of data samples)
- [ ] **C.** 5 (equal to the number of bins)
- [ ] **D.** 100 (equal to the maximum data value)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.sum(counts) = 1.0` only when `density=True`. With `density=False` (default), `counts` contains raw integer frequency counts. The sum of all bin counts equals the total number of data samples.
- **Option B (Correct):** With `density=False`, each element of `counts` is the number of data points in that bin. Summing all bins gives the total sample count. This dataset has 10 values spread across 5 bins (2 per bin for this uniform dataset): `counts = [2, 2, 2, 2, 2]`. `np.sum(counts) = 10` = total samples. This is always true for `density=False`: every data point is assigned to exactly one bin, so the counts must sum to `len(data)`.
- **Option C (Incorrect):** The sum of counts equals the total samples (10), not the number of bins (5). Confusing bin count with sample count is a common misconception.
- **Option D (Incorrect):** 100 is the maximum data value — it has no relationship to the sum of counts. The sum of counts is always `len(data)` for `density=False`.

</details>

---

### 63. Question

A developer creates a histogram for model prediction probabilities and observes the following code. Identify the **conceptual error**:

```python
probs = model.predict_proba(X_test)[:, 1]  # Positive class probabilities

# Developer's analysis:
counts, bins, _ = plt.hist(probs, bins=10, density=True)
print("Sum of all density values:", np.sum(counts))
print("This should equal 1.0 since density=True")
```

What is the conceptual error in the developer's comment?

- [ ] **A.** `density=True` should not be used for probability values between 0 and 1
- [ ] **B.** The sum of density values is NOT necessarily 1.0 — it is the sum of `density × bin_width` (the total AREA) that equals 1.0, not the raw sum of heights
- [ ] **C.** `predict_proba` returns values that are incompatible with `plt.hist()`
- [ ] **D.** `bins=10` is too few for probability data — at least 100 bins are required

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `density=True` is perfectly valid for probability values [0, 1]. There are no restrictions on the input data range for density histograms.
- **Option B (Correct):** The critical misconception: with `density=True`, **the AREA integrates to 1.0** — NOT the raw sum of bar heights. `np.sum(counts)` sums the density values, which equals `1.0 / mean_bin_width`. For bins=10 on [0, 1]: each bin has width 0.1. `np.sum(density × width) = 1.0`, but `np.sum(density) = 1.0 / 0.1 = 10.0`. The correct verification is: `np.sum(counts * np.diff(bins)) ≈ 1.0`. `np.sum(counts)` alone does NOT equal 1.0 unless bin width happens to be exactly 1.0.
- **Option C (Incorrect):** `model.predict_proba(X)[:, 1]` returns a 1D NumPy array of floats between [0, 1] — perfectly compatible with `plt.hist()`. No incompatibility exists.
- **Option D (Incorrect):** `bins=10` is a reasonable default for probability histograms. There is no minimum bin count requirement for probability data.

</details>

---

### 64. Question

A developer builds a loss curve histogram for **hyperparameter search diagnostics**:

```python
np.random.seed(20)
# 200 hyperparameter combinations — final validation losses
val_losses = np.concatenate([
    np.random.normal(0.85, 0.03, 150),  # Most configs: mediocre
    np.random.normal(0.45, 0.05, 50),   # Best configs: low loss
])

plt.hist(val_losses, bins=25, edgecolor="black", color="orchid")
plt.axvline(np.mean(val_losses), color="red",   linestyle="--", label="Mean")
plt.axvline(np.median(val_losses), color="blue", linestyle="-",  label="Median")
plt.xlabel("Validation Loss"); plt.legend()
```

What is the **distribution shape**, and is the mean or median a better summary statistic here?

- [ ] **A.** Bimodal — median is more robust; it is less influenced by the large cluster at 0.85 than the mean
- [ ] **B.** Normal — mean and median are equivalent
- [ ] **C.** Right-skewed — the mean underestimates typical performance
- [ ] **D.** Uniform — neither mean nor median is informative

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The data is a **mixture of two normal distributions** — 150 values near 0.85 and 50 values near 0.45. This creates a **bimodal distribution** with two peaks. The large cluster (150 configs at 0.85) heavily influences the mean, pulling it toward 0.85 and away from the interesting low-loss cluster (50 configs at 0.45). The **median** is more robust here — it represents the "typical" hyperparameter result without being dominated by the majority cluster. However, for bimodal data, neither mean nor median fully captures the distribution; reporting BOTH clusters separately (via the histogram itself) is most informative.
- **Option B (Incorrect):** The data is clearly bimodal (two distinct peaks), not normally distributed. For a normal distribution, mean = median. The bimodal structure means they differ and neither alone adequately summarizes the distribution.
- **Option C (Incorrect):** Right-skewed distributions have a single peak with a tail to the right. Two separate clusters at 0.45 and 0.85 with a valley between them is bimodal, not right-skewed.
- **Option D (Incorrect):** A uniform distribution has roughly equal probability density across all values. Two concentrated clusters with low density between them is the opposite of uniform.

</details>

---

### 65. Question

A developer writes a function to compute which **bin index** a value falls into, given explicit bin edges:

```python
def find_bin(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i+1]:
            return i
    # Handle last bin (fully closed)
    if value == edges[-1]:
        return len(edges) - 2
    return -1  # Out of range

edges = [0, 10, 20, 30, 40, 50]
print(find_bin(20, edges))   # Q: What does this print?
print(find_bin(50, edges))   # Q: What does this print?
print(find_bin(55, edges))   # Q: What does this print?
```

What are the three printed values?

- [ ] **A.** `1`, `4`, `-1`
- [ ] **B.** `2`, `5`, `-1`
- [ ] **C.** `2`, `4`, `-1`
- [ ] **D.** `1`, `5`, `0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `find_bin(20, edges)`: checks i=0: `0 ≤ 20 < 10`? No. i=1: `10 ≤ 20 < 20`? No (20 is NOT < 20). i=2: `20 ≤ 20 < 30`? Yes → returns **2**, not 1.
- **Option B (Incorrect):** `find_bin(50, edges)`: 50 does not satisfy `edges[i] ≤ 50 < edges[i+1]` for any i (since `50 < 50` is False). Falls through to the last-bin check: `value == edges[-1]` → `50 == 50` → True → returns `len(edges)-2 = 6-2 = 4`. Not 5.
- **Option C (Correct):** `find_bin(20, edges)`: i=2 check: `20 ≤ 20 < 30` → True → returns **2** (bin index 2 = interval [20,30)). `find_bin(50, edges)`: no half-open interval matches (50 is not < 50); last-bin special case: `50 == edges[-1]=50` → True → returns `len(edges)-2 = 4` (bin index 4 = [40,50]). `find_bin(55, edges)`: 55 > 50, no interval matches, last-bin check fails (55 ≠ 50) → returns **-1** (out of range). Result: `2, 4, -1`.
- **Option D (Incorrect):** `find_bin(55, edges)` cannot return 0 — 55 is beyond all bin edges and the function correctly returns -1 for out-of-range values.

</details>

---

### 66. Question

A developer creates a **data quality diagnostic** histogram for missing value rates across features:

```python
missing_rates = np.array([0.01, 0.02, 0.01, 0.45, 0.50, 0.48, 0.03, 0.02,
                           0.47, 0.51, 0.02, 0.49, 0.01, 0.50, 0.03])

plt.hist(missing_rates, bins=10, edgecolor="black", color="salmon")
plt.axvline(0.20, color="red", linestyle="--", linewidth=2, label="20% Threshold")
plt.xlabel("Missing Rate"); plt.ylabel("Feature Count")
plt.title("Feature Missing Rate Distribution")
plt.legend()
```

The histogram shows two distinct clusters: one near 0–5% and one near 45–51%. What does this bimodal pattern indicate about the dataset?

- [ ] **A.** The features are all highly correlated with each other
- [ ] **B.** Two groups of features exist: one group with very low missingness (likely well-collected fields) and another group with ~50% missingness (likely optional or conditionally collected fields — possibly requiring separate imputation strategies)
- [ ] **C.** The random seed should be changed to produce a more uniform distribution
- [ ] **D.** The histogram bin count (`bins=10`) is causing artificial bimodality

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Feature missingness rates describe data collection patterns, not feature-to-feature correlations. Correlation is measured by correlation matrices or mutual information — not by the distribution of missing rates.
- **Option B (Correct):** The bimodal distribution in missing rates reveals two distinct feature groups: (1) Features with near-zero missingness (0–5%) — these are mandatory, consistently collected fields (e.g., customer ID, transaction date). (2) Features with ~50% missingness — these are optional, conditional, or survey-based fields that roughly half the records don't have. This is actionable insight: the two groups likely require different imputation strategies. Low-missingness features might use simple mean/mode imputation; high-missingness features might be flagged, binned, or handled with indicator variables.
- **Option C (Incorrect):** Missing rates are computed from the actual dataset — they are deterministic counts, not random samples. Changing a random seed has no effect on missing rate calculations.
- **Option D (Incorrect):** With `bins=10` over range [0.01, 0.51] = width 0.50, each bin is 0.05 wide. The two clusters (0–5% and 45–51%) are separated by a 40% gap — clearly bimodal regardless of bin count. Bimodality is not an artifact here; it reflects genuine data structure.

</details>

---

### 67. Question

A developer creates a histogram with `bins=50` for a dataset of only **20 samples**. Most bins are empty. What is the recommended action?

- [ ] **A.** Use `density=True` to fill the empty bins with interpolated values
- [ ] **B.** Reduce bins to a value appropriate for the sample size — Sturges' rule suggests `k ≈ 1 + log₂(20) ≈ 6` bins for 20 samples
- [ ] **C.** Add more data by duplicating the 20 samples to fill all 50 bins
- [ ] **D.** Use `cumulative=True` to eliminate empty bins automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `density=True` normalizes the heights of existing non-empty bars but does NOT fill empty bins with values. Empty bins remain empty (count = 0, density = 0) regardless of normalization. `density=True` does not perform interpolation.
- **Option B (Correct):** **Sturges' rule**: optimal bin count k ≈ 1 + log₂(n) = 1 + log₂(20) ≈ 1 + 4.32 ≈ **5–6 bins** for n=20 samples. With `bins=50` and only 20 data points, each bin contains on average 0.4 samples — most are empty, and the histogram shows a noisy spike pattern with no meaningful distributional information. Reducing to 5–6 bins produces a legible histogram that reveals the distribution's gross shape.
- **Option C (Incorrect):** Duplicating samples is a form of synthetic data augmentation that distorts the true distribution. It does not solve the root problem (too many bins for too few samples) — it just inflates sample size artificially.
- **Option D (Incorrect):** `cumulative=True` changes bar heights to cumulative counts but does NOT remove empty bins. Bins with zero count still appear as flat segments in a cumulative histogram (the cumulative count doesn't increase where there are no data points).

</details>

---

### 68. Question

A developer computes the **theoretical frequency** for each bin of a normal distribution:

```python
# N(60, 10), n=500 samples, 5 equal-width bins spanning [30, 90]
# Each bin: width=12, range = [30,42), [42,54), [54,66), [66,78), [78,90]
# P(X in [54, 66)) for N(60,10) ≈ 0.383 (within ±0.6σ of mean)
expected_count = 500 * 0.383
print(expected_count)
```

What does `expected_count` approximately equal?

- [ ] **A.** 38.3
- [ ] **B.** 191.5
- [ ] **C.** 383
- [ ] **D.** 76.6

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 38.3 would be `500 × 0.0766`. The probability 0.383 is directly multiplied by the sample size 500: 500 × 0.383 = 191.5.
- **Option B (Correct):** Expected count = total samples × probability = 500 × 0.383 = **191.5**. The central bin [54, 66) spans 60±6 on a N(60,10) distribution — this is ±0.6σ from the mean. The empirical rule gives ≈38.3% probability in this range (slightly less than the ±1σ rule of 68.3%). So approximately 191–192 of the 500 samples are expected in the central bin — the tallest bar in the histogram, reflecting the Gaussian peak.
- **Option C (Incorrect):** 383 would be the expected count if probability were 0.766 (twice 0.383). This is incorrect — P ≈ 0.383 for ±0.6σ, not 0.766.
- **Option D (Incorrect):** 76.6 = 500 × 0.1532. This does not correspond to the given probability of 0.383.

</details>

---

### 69. Question

A developer creates an **age distribution** histogram for a dataset and notices the histogram shows a heavy left tail with most ages concentrated above 60:

```python
ages = survey_data["age"]
plt.hist(ages, bins=20, edgecolor="black")
plt.axvline(np.mean(ages), color="red",   linestyle="--", label=f"Mean={np.mean(ages):.1f}")
plt.axvline(np.median(ages), color="green", linestyle="-", label=f"Median={np.median(ages):.1f}")
```

The mean shows 58.3 and the median shows 63.0. What skewness type is this and what does it mean?

- [ ] **A.** Right-skewed — mean > median means long right tail of older respondents
- [ ] **B.** Left-skewed (negative skew) — mean < median; the left tail of younger respondents pulls the mean downward below the median
- [ ] **C.** No skew — the difference between 58.3 and 63.0 is negligible
- [ ] **D.** Bimodal — mean and median being different always indicates two peaks

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Right-skewed means mean > median (a long right tail pulls the mean UP). Here mean=58.3 < median=63.0 — the mean is BELOW the median, indicating left-skew, not right-skew.
- **Option B (Correct):** **Left-skewed (negatively skewed)**: mean (58.3) < median (63.0). Most ages are concentrated in the upper range (60+) — consistent with "heavy concentration above 60" as described. A tail of younger respondents extends to the LEFT (toward lower ages), pulling the arithmetic mean downward below the median. In the left-skewed pattern: mean < median < mode. The histogram peak (mode) is at the high end (~65+), the median is at 63, and the mean is pulled down to 58.3 by a minority of younger respondents.
- **Option C (Incorrect):** A 4.7-unit difference between mean and median on an age scale (58.3 vs 63.0) represents a meaningful skew. "Negligible" would be a difference of <0.5 units relative to the scale. This is not negligible skew.
- **Option D (Incorrect):** Mean ≠ median does NOT automatically indicate bimodality. Any skewed distribution (unimodal with an asymmetric tail) has mean ≠ median. Bimodality requires two visible peaks in the histogram — a separate visual observation from the mean-median relationship.

</details>

---

### 70. Question

A developer creates the following complete ML diagnostic histogram and must answer: which `axvline` call produces the line that is **always to the left** of the other in a right-skewed distribution?

```python
np.random.seed(33)
latency = np.random.lognormal(mean=2, sigma=0.7, size=1000)  # API response times (ms)

plt.hist(latency, bins=40, density=True, edgecolor="black", alpha=0.7, color="teal")
line_a = plt.axvline(np.mean(latency),   color="red",  linestyle="--", linewidth=2)
line_b = plt.axvline(np.median(latency), color="lime", linestyle="-",  linewidth=2)
line_c = plt.axvline(np.percentile(latency, 95), color="purple", linestyle=":", linewidth=2)
plt.legend(["Mean", "Median", "P95"])
```

In a log-normal (right-skewed) distribution, what is the correct ordering of these three lines from **left to right**?

- [ ] **A.** Median < Mean < P95
- [ ] **B.** Mean < Median < P95
- [ ] **C.** P95 < Median < Mean
- [ ] **D.** Mean = Median < P95

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For a **right-skewed (log-normal) distribution**: the **mode** < **median** < **mean** < **P95** (or other high percentiles). The median is always less than the mean for right-skewed distributions (the long right tail inflates the arithmetic mean above the 50th percentile). The 95th percentile (P95) is by definition at the far right of the distribution — 95% of data points fall below this value. So the left-to-right ordering is: **Median (lime) → Mean (red) → P95 (purple)**. The lime median line is leftmost, the red mean line is to its right, and the purple P95 line is far to the right in the tail.
- **Option B (Incorrect):** Mean < Median describes a **left-skewed** distribution. For right-skewed (log-normal), mean > median — the mean is pulled to the RIGHT of the median by extreme values.
- **Option C (Incorrect):** P95 < Median would mean 95% of data is below the median — which contradicts the definition of the 50th percentile (median). P95 is always greater than the median (P50).
- **Option D (Incorrect):** For a log-normal distribution, mean ≠ median. The log-normal mean = exp(μ + σ²/2) and median = exp(μ). With σ=0.7: mean = exp(2 + 0.245) ≈ exp(2.245) ≈ 9.44; median = exp(2) ≈ 7.39. Mean > Median, not equal.

</details>
