# Matplotlib Histogram Objective Test — Set 05

---

### 41. Question

A developer examines the following code and must identify which line contains a **bug**:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(50, 10, 1000)

# Line A
counts, bins, patches = plt.hist(data, bins=20, density=True)

# Line B
print(f"Counts array length: {len(counts)}")     # Prints 20

# Line C
print(f"Bins array length: {len(bins)}")          # Prints 21

# Line D
total_area = np.sum(counts) * np.mean(np.diff(bins))
print(f"Total area: {total_area:.4f}")            # Should print ~1.0
```

Which line contains a subtle **approximation error** that only works correctly for **equal-width bins**?

- [ ] **A.** Line A — `density=True` is the wrong parameter for area normalization
- [ ] **B.** Line B — `len(counts)` should equal 21, not 20
- [ ] **C.** Line D — `np.mean(np.diff(bins))` only gives the correct area for equal-width bins; for non-uniform bins, use `np.sum(counts * np.diff(bins))`
- [ ] **D.** Line C — `len(bins)` should equal 20, not 21

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `density=True` is the correct parameter for producing probability density output where total area = 1. This line has no bug.
- **Option B (Incorrect):** `len(counts) = 20` is correct — there are exactly 20 frequency/density values for 20 bins. This comment is accurate.
- **Option C (Correct):** `np.mean(np.diff(bins))` computes the **average** bin width, then multiplies by the sum of counts. This only gives the correct total area when ALL bins have equal width (so average = each individual width). For equal-width bins (as in this case with `bins=20` scalar), it works. But for non-uniform bin edges (e.g., `bins=[0, 5, 15, 30, 100]`), the widths differ — using the mean width introduces an approximation error. The **correct formula** for any bin layout is `np.sum(counts * np.diff(bins))` — element-wise multiplication of each density by its specific width, then summing. This is always exact.
- **Option D (Incorrect):** `len(bins) = 21` is correct — for 20 bins, there are always 21 edge boundaries. This comment is accurate.

</details>

---

### 42. Question

A developer uses `histtype='barstacked'` with two arrays:

```python
class_0 = np.random.normal(40, 8, 300)
class_1 = np.random.normal(60, 8, 300)

plt.hist([class_0, class_1], bins=20, histtype='barstacked',
         color=['#FF7043', '#42A5F5'],
         label=['Class 0', 'Class 1'])
plt.legend()
```

How does `histtype='barstacked'` differ from `stacked=True` with `histtype='bar'`?

- [ ] **A.** `barstacked` uses outline-only rendering; `stacked=True` uses filled bars
- [ ] **B.** They produce identical visual output — both stack bars on top of each other within each bin
- [ ] **C.** `barstacked` normalizes to density automatically; `stacked=True` does not
- [ ] **D.** `barstacked` works only with 2 distributions; `stacked=True` works with any number

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Both `histtype='barstacked'` and `stacked=True, histtype='bar'` produce filled bar charts. Neither uses outline-only rendering. Outline-only is `histtype='step'`.
- **Option B (Correct):** `histtype='barstacked'` and `stacked=True` (with default `histtype='bar'`) produce **equivalent visual output** — both stack the second dataset's bars on top of the first within each bin. In Matplotlib's implementation, `histtype='barstacked'` is simply the explicit way to specify stacking via the `histtype` string, while `stacked=True` achieves the same result while keeping `histtype='bar'` as the base rendering mode. The resulting figures are visually identical.
- **Option C (Incorrect):** Neither `barstacked` nor `stacked=True` automatically normalizes to density. Density normalization always requires the separate `density=True` parameter, regardless of the `histtype` or `stacked` setting.
- **Option D (Incorrect):** Both `histtype='barstacked'` and `stacked=True` support any number of distributions passed as a list. There is no 2-distribution limit on either approach.

</details>

---

### 43. Question

A developer creates a histogram for response time data (in seconds) and patches specific bars with custom colors based on threshold:

```python
counts, bins, patches = plt.hist(response_times, bins=15, edgecolor="black")

# Color bars: green if < 2s, yellow if 2-5s, red if > 5s
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge < 2:
        patch.set_facecolor("green")
    elif left_edge < 5:
        patch.set_facecolor("yellow")
    else:
        patch.set_facecolor("red")
```

Why is `bins[:-1]` used (instead of `bins`) in the `zip()` call?

- [ ] **A.** `bins[:-1]` removes the last bin edge to match the length of `patches` (N edges vs N patches)
- [ ] **B.** `bins[:-1]` skips the first edge (index 0) which is always 0
- [ ] **C.** `bins[:-1]` reverses the bin order for right-to-left color assignment
- [ ] **D.** `bins[:-1]` is identical to `bins` — the `[:-1]` slice has no effect

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `bins` has length `N+1` (e.g., 16 for 15 bins) while `patches` has length `N` (15). To `zip()` them together, they must have the same length. `bins[:-1]` slices off the **last** bin edge, giving length `N` (15) — matching `patches`. Each element `bins[i]` (for i = 0 to N-1) is the **left edge** of the i-th bin, which the code uses to determine the threshold-based color for that bar. This is the standard pattern for per-bar patch styling based on bin position.
- **Option B (Incorrect):** `bins[:-1]` removes the LAST element (index -1), not the first. The first edge `bins[0]` IS included in `bins[:-1]`. Slicing from the start would use `bins[1:]` to skip the first edge.
- **Option C (Incorrect):** `bins[:-1]` does NOT reverse order. It simply removes the last element. Reversal would require `bins[::-1]`.
- **Option D (Incorrect):** `bins[:-1]` is fundamentally different from `bins`. `bins` has 16 elements; `bins[:-1]` has 15. The `zip()` with `patches` (15 elements) would truncate to the shorter of the two if full `bins` were used — but using `bins[:-1]` is the explicit, clear approach.

</details>

---

### 44. Question

A developer creates a histogram of house prices (right-skewed data):

```python
np.random.seed(3)
prices = np.random.lognormal(mean=12, sigma=0.8, size=500)  # In dollars

plt.hist(prices, bins=30, edgecolor="black", color="steelblue")
plt.axvline(np.mean(prices), color="red",   linestyle="--", label="Mean")
plt.axvline(np.median(prices), color="lime", linestyle="-",  label="Median")
plt.legend()
```

In this right-skewed distribution, where does the mean appear relative to the median and the histogram peak?

- [ ] **A.** Mean = Median = peak (all three coincide at the center)
- [ ] **B.** Mean is to the LEFT of the median and peak (mean < median < mode)
- [ ] **C.** Mean is to the RIGHT of both the median and the histogram peak (mode < median < mean)
- [ ] **D.** Mean and median are equal; both appear to the right of the peak

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This describes a **symmetric (normal) distribution** where mean = median = mode. A log-normal distribution (used here) is right-skewed, so these three statistics are NOT equal.
- **Option B (Incorrect):** Mean to the LEFT of median describes **left-skewed (negative skew)** distributions. Log-normal distributions are right-skewed — the mean is pulled to the RIGHT of the median.
- **Option C (Correct):** For **right-skewed distributions** (positive skew), the ordering is: **mode < median < mean**. The histogram peak (mode) is at the lowest values (near the left side where the bulk of affordable houses cluster). The median is further right (half the houses are cheaper, half more expensive). The mean is pulled furthest right by the small number of extremely expensive houses (the long right tail) — causing mean > median. The log-normal distribution is the textbook example of this pattern (income, house prices, stock returns).
- **Option D (Incorrect):** In a right-skewed distribution, mean > median — they are NOT equal. The mean is inflated by large outliers in the right tail.

</details>

---

### 45. Question

A developer observes this code and must predict whether the result is left-skewed, right-skewed, or symmetric:

```python
np.random.seed(8)
# Simulate exam scores capped at 100 with most students scoring high
scores = 100 - np.random.exponential(scale=10, size=600)
scores = scores[scores >= 0]  # Remove any negatives
plt.hist(scores, bins=25)
```

What is the distribution shape and why?

- [ ] **A.** Right-skewed — exponential distributions always produce right-skewed histograms
- [ ] **B.** Left-skewed — most values cluster near 100 (the upper cap) with a tail extending toward lower scores
- [ ] **C.** Normal — subtracting from 100 converts exponential to Gaussian
- [ ] **D.** Uniform — the `scale=10` parameter creates uniform spacing

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The raw exponential distribution IS right-skewed. But `100 - exponential` REVERSES the skew. Subtracting from a constant flips the distribution. The long right tail of the exponential becomes a long LEFT tail after the transformation.
- **Option B (Correct):** `np.random.exponential(scale=10)` generates values concentrated near 0 with a long right tail. Subtracting from 100: `100 - exponential` shifts the bulk near 100 (since most exponential values are small, near 0) and creates a long tail toward LOWER scores (since large exponential values create low scores). The result is **left-skewed (negatively skewed)**: most students score near 100, with a long tail of lower-performing students. This models a ceiling effect common in easy exams.
- **Option C (Incorrect):** Subtracting from a constant FLIPS the distribution (mirrors it), it does NOT convert it to a Gaussian/normal distribution. The resulting distribution is still exponential-shaped, just mirrored.
- **Option D (Incorrect):** `scale=10` is the mean of the exponential distribution, not a spacing parameter. The exponential distribution remains highly non-uniform — it has an exponentially decaying density, not a flat (uniform) density.

</details>

---

### 46. Question

A developer creates a histogram for a dataset of customer purchase amounts and wants to add a **vertical line at the 90th percentile**:

```python
np.random.seed(22)
purchases = np.random.lognormal(3, 1, 800)

counts, bins, patches = plt.hist(purchases, bins=30, edgecolor="black", alpha=0.7)
p90 = np.percentile(purchases, 90)
plt.axvline(p90, color="purple", linewidth=2, linestyle=":",
            label=f"90th Percentile: ${p90:.0f}")
plt.legend()
```

What does the vertical line at the 90th percentile indicate in the context of this histogram?

- [ ] **A.** 90% of the histogram bars are to the left of this line
- [ ] **B.** 90% of all purchase values fall at or below this line — the line divides the top 10% of spenders from the rest
- [ ] **C.** The bar at this position has a height equal to 90% of the maximum bar height
- [ ] **D.** 90% of the total probability density area is to the right of this line

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The 90th percentile line is positioned based on the DATA distribution, not the number of bars. It may fall within any bar — not necessarily after exactly 90% of the bins.
- **Option B (Correct):** `np.percentile(data, 90)` returns the value below which **90% of the data points** fall. The `axvline` at this value divides the histogram into two regions: the area to the left contains bars representing 90% of all purchases (the majority of customers), while the area to the right represents the top 10% of spenders (high-value customers). This is useful for identifying VIP customer thresholds or outlier boundaries.
- **Option C (Incorrect):** The 90th percentile has no relationship to bar heights relative to the maximum bar. Bar height reflects frequency count — it is unrelated to the percentile position.
- **Option D (Incorrect):** 90% of the area to the RIGHT would describe the **10th percentile** (where only 10% is to the left). The 90th percentile has 90% of area to the LEFT, not the right.

</details>

---

### 47. Question

A developer creates a histogram of model residuals (prediction errors) for regression diagnostics:

```python
np.random.seed(5)
residuals = np.random.normal(0, 1, 800)
plt.hist(residuals, bins=30, density=True, edgecolor="black", alpha=0.8)
plt.axvline(0, color="black", linewidth=1.5, linestyle="-", label="Zero")
plt.axvline(np.mean(residuals), color="red", linestyle="--", label="Mean")
```

For a **well-calibrated regression model**, what pattern should the residuals histogram show?

- [ ] **A.** A right-skewed distribution with mean >> 0 (large positive bias)
- [ ] **B.** A bimodal distribution with peaks at -1 and +1 (symmetric errors around zero)
- [ ] **C.** A roughly symmetric, approximately normal (Gaussian) distribution centered near zero, with the red mean line close to the black zero line
- [ ] **D.** A uniform distribution between -3 and +3 (equal probability of all error magnitudes)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A right-skewed distribution with mean >> 0 indicates **systematic positive bias** — the model consistently over-predicts. This is a sign of model miscalibration, not good performance.
- **Option B (Incorrect):** Bimodal residuals at -1 and +1 would indicate the model makes two distinct types of errors — suggesting a structural problem (e.g., the model is missing a binary categorical feature). Well-calibrated models produce unimodal residuals.
- **Option C (Correct):** For a well-calibrated regression model, residuals should follow a **zero-mean normal distribution**: (1) Centered at zero — no systematic over/under-prediction. (2) Approximately normal — OLS regression theory assumes normally distributed errors. (3) Symmetric — errors in both directions are equally likely. The red mean line should nearly coincide with the black zero line. Any significant deviation indicates bias. Skewness or heavy tails indicate model misspecification or heteroscedasticity.
- **Option D (Incorrect):** Uniform residuals would suggest the model has zero predictive ability over a fixed range — a trivially bad model. Real models with any predictive power produce errors that decrease in probability as magnitude increases, following an approximately bell-shaped distribution.

</details>

---

### 48. Question

A developer uses `plt.hist()` to diagnose **bimodal data** from a customer segmentation dataset:

```python
np.random.seed(12)
segment_a = np.random.normal(30, 5, 400)
segment_b = np.random.normal(70, 5, 400)
combined  = np.concatenate([segment_a, segment_b])

plt.hist(combined, bins=30, edgecolor="black", color="mediumpurple")
```

What will the histogram look like, and what does it indicate?

- [ ] **A.** A single bell curve centered at 50 — the two segments merge into one normal distribution
- [ ] **B.** Two separate peaks: one near 30 and one near 70, with a valley between them — indicating two distinct sub-populations within the data
- [ ] **C.** A uniform distribution between 20 and 80 — concatenating two normals creates uniformity
- [ ] **D.** A right-skewed distribution because `segment_b` has larger values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Two well-separated normal distributions (means 30 and 70, each with std=5) do NOT merge into a single bell curve centered at 50. They remain distinct because the separation (40 units) is 8 standard deviations — the distributions barely overlap. The combined histogram reveals both peaks clearly.
- **Option B (Correct):** With `segment_a ~ N(30, 5)` and `segment_b ~ N(70, 5)`, the concatenated data creates a **bimodal distribution**: two peaks near 30 and 70 with a valley between approximately 50 (where very few data points exist). This pattern in a histogram immediately signals that the data contains two distinct sub-populations — in this case, two customer segments with fundamentally different behavior patterns. Bimodality is a key diagnostic signal that simple mean/variance statistics are misleading.
- **Option C (Incorrect):** Concatenating two normal distributions creates a **bimodal mixture**, not a uniform distribution. Uniform distributions require each value in the range to be equally probable — this is not the case for a mixture of two Gaussians.
- **Option D (Incorrect):** The distribution is not right-skewed. Both distributions are symmetric normals, and the combined distribution is also symmetric (both have equal sample sizes and equal-width distributions). The overall shape is bimodal and symmetric around the midpoint 50.

</details>

---

### 49. Question

A developer creates the following histogram and asks: "Why is the histogram almost invisible when plotted after a large dataset?"

```python
plt.figure(figsize=(10, 5))

# Large dataset - plotted first
counts1, bins1, _ = plt.hist(large_data, bins=40, color="blue")

# Small dataset - plotted second (disappears behind first)
counts2, bins2, _ = plt.hist(small_data, bins=40, color="red")
```

The red histogram is invisible. What is the most likely cause?

- [ ] **A.** `plt.hist()` can only hold one dataset at a time — the second call overwrites the first
- [ ] **B.** Both histograms have `alpha=1.0` (fully opaque) by default — the red bars are hidden BEHIND the blue bars where they overlap, since blue was drawn first
- [ ] **C.** The `figsize` is too small to show both histograms
- [ ] **D.** `plt.hist()` automatically removes duplicates when the same `bins` count is used twice

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.hist()` does NOT overwrite previous histograms. Multiple calls to `plt.hist()` on the same axes ADD new histogram artists — they coexist as independent overlapping patches. Both histograms exist; the issue is visibility due to opacity.
- **Option B (Correct):** Both histograms default to `alpha=1.0` (fully opaque). The blue histogram is drawn first; the red histogram is drawn on top. Where the red bars overlap blue bars, the red bars ARE visible (on top). However, if `large_data` produces much taller bars than `small_data`, the red bars may be shorter and appear to be hidden. But more commonly with fully opaque defaults, the second histogram's bars completely occlude the first where they overlap in position and height. The fix is `alpha=0.5` on both. Additionally, if `small_data`'s bars are entirely within the range and height of `large_data`'s bars, they are completely hidden.
- **Option C (Incorrect):** `figsize=(10, 5)` is a perfectly standard figure size. Visibility issues from size would only occur with extremely small figures (e.g., 1×1 inches).
- **Option D (Incorrect):** `plt.hist()` does not deduplicate or merge datasets. Each call independently bins and draws its own dataset. The bin count parameter `bins=40` being the same for both calls is irrelevant — the data values differ.

</details>

---

### 50. Question

A developer creates the following multi-distribution histogram for an ML paper:

```python
np.random.seed(99)
train_loss = np.random.normal(0.25, 0.05, 1000)
val_loss   = np.random.normal(0.30, 0.08, 1000)
test_loss  = np.random.normal(0.35, 0.06, 1000)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(train_loss, bins=25, alpha=0.6, color="#4CAF50", label="Train")
ax.hist(val_loss,   bins=25, alpha=0.6, color="#2196F3", label="Validation")
ax.hist(test_loss,  bins=25, alpha=0.6, color="#FF5722", label="Test")
ax.axvline(np.mean(train_loss), color="#1B5E20", linestyle="--", linewidth=1.5)
ax.axvline(np.mean(val_loss),   color="#0D47A1", linestyle="--", linewidth=1.5)
ax.axvline(np.mean(test_loss),  color="#BF360C", linestyle="--", linewidth=1.5)
ax.set_xlabel("Loss Value"); ax.set_ylabel("Frequency")
ax.set_title("Train / Validation / Test Loss Distribution")
ax.legend()
```

What does the **separation between the mean lines** of training and test loss indicate in ML model evaluation?

- [ ] **A.** A bug in the plotting code — all three means should be identical
- [ ] **B.** The model generalizes perfectly — separate means confirm the model learned distinct distributions
- [ ] **C.** A **generalization gap**: the training loss is lower than test loss on average, indicating the model performs better on data it was trained on than on unseen test data — a sign of potential overfitting
- [ ] **D.** Data preprocessing error — train/val/test splits should always produce identical loss distributions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Different means are expected and diagnostically meaningful — not a bug. If all means were identical, it would suggest the model performs equally on all splits, which is rare and would actually indicate potential data leakage.
- **Option B (Incorrect):** Separate means do NOT indicate perfect generalization. Lower training loss than test loss indicates overfitting tendency, not perfect generalization. Perfect generalization would show train ≈ val ≈ test means.
- **Option C (Correct):** The histogram shows `train_loss ~ N(0.25, 0.05)` and `test_loss ~ N(0.35, 0.06)` — the test mean (0.35) is 40% higher than the train mean (0.25). This **generalization gap** (test loss > training loss) is the signature of **overfitting**: the model has fit the training data well (low training loss) but does not generalize as well to unseen test data (higher test loss). The histogram visualization makes this gap immediately visible through the separation of mean reference lines and the horizontal shift of the loss distributions.
- **Option D (Incorrect):** Train/val/test splits should produce SIMILAR but not IDENTICAL distributions for a well-generalizing model. If the underlying data-generating process is the same, distributions will be approximately equal. Systematic differences (as shown here) indicate a real performance gap, not a preprocessing error.

</details>
