# Matplotlib Histogram Objective Test — Set 09

---

### 81. Question

A developer creates a comprehensive **class probability distribution diagnostic** for a multi-class classifier:

```python
np.random.seed(88)
n = 800

# Softmax probabilities for the predicted class (should be high for good model)
good_model_probs = np.random.beta(8, 2, n)   # Concentrated near 1.0
poor_model_probs = np.random.beta(1.5, 1.5, n)  # Spread near 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(good_model_probs, bins=20, edgecolor="black",
             color="mediumseagreen", density=True)
axes[0].set_title("Well-Trained Model"); axes[0].set_xlabel("Predicted Probability")

axes[1].hist(poor_model_probs, bins=20, edgecolor="black",
             color="tomato", density=True)
axes[1].set_title("Poorly Trained Model"); axes[1].set_xlabel("Predicted Probability")
```

For the **well-trained model**, where is the histogram peak located and what does it mean?

- [ ] **A.** Peak near 0.5 — the model predicts with 50% confidence for all samples
- [ ] **B.** Peak near 1.0 — the model assigns high probability to the correct class for most samples, indicating confident and accurate predictions
- [ ] **C.** Peak near 0.0 — the model predicts the wrong class with high confidence
- [ ] **D.** No clear peak — Beta(8,2) always produces a uniform distribution

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Beta(1.5, 1.5) (the poor model) produces values concentrated near 0.5 — not Beta(8, 2). A peak at 0.5 indicates maximum uncertainty (the model cannot distinguish between classes).
- **Option B (Correct):** Beta(α, β) has mode = (α-1)/(α+β-2) for α,β > 1. For Beta(8, 2): mode = (8-1)/(8+2-2) = 7/8 = **0.875** — concentrated near 1.0. This represents a well-trained model: most predictions assign high probability (>0.8) to the correct class. A left-heavy histogram (bulk at high probabilities) is the desired shape for classifier probability outputs when evaluated on correctly classified samples.
- **Option C (Incorrect):** A peak near 0.0 (using `predict_proba` for the predicted class) would indicate the model is actively predicting the wrong class with high confidence — the worst possible scenario, worse even than random chance.
- **Option D (Incorrect):** Beta(8, 2) is strongly right-concentrated (near 1.0) — the opposite of uniform. A uniform distribution would require Beta(1, 1). Beta(8, 2) has a clear, distinct mode at 0.875.

</details>

---

### 82. Question

A developer creates a histogram for **response latency monitoring** in a production ML system:

```python
np.random.seed(66)
# API latencies in milliseconds — mostly fast with occasional slow outliers
latencies = np.concatenate([
    np.random.exponential(scale=20, size=950),   # Normal: fast responses
    np.random.uniform(500, 2000, size=50)         # Anomalous: slow responses
])

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(latencies, bins=40, edgecolor="black", color="slateblue", alpha=0.8)
ax.axvline(np.percentile(latencies, 99), color="red", linewidth=2,
           linestyle="--", label="P99 Latency")
ax.set_xlabel("Latency (ms)"); ax.legend()
ax.set_title("API Response Latency Distribution")
```

Why is the **P99 line** more appropriate than the **mean line** for this latency SLA (Service Level Agreement) monitoring?

- [ ] **A.** P99 is easier to compute than mean
- [ ] **B.** P99 represents the worst 1% of latency experiences — directly relevant for SLA guarantees ("99% of requests complete within X ms") — the mean is pulled down by the fast majority and hides severe tail latencies
- [ ] **C.** The mean is undefined for exponential distributions
- [ ] **D.** P99 automatically removes the 50 outlier values before computing the threshold

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** P99 requires sorting and indexing — it is not significantly simpler than computing the mean (`np.mean()`). Computational simplicity is not the reason.
- **Option B (Correct):** The mean latency is dominated by the 950 fast exponential responses (mean ≈ 20ms) and is pulled only slightly by the 50 slow outliers. It might show ~70ms — appearing acceptable. But P99 captures the 99th percentile — the threshold below which 99% of requests complete. For SLA compliance ("99% of API calls respond in under X ms"), P99 directly maps to the contractual guarantee. The slow 500–2000ms responses are visible at P99 even when the mean appears fast. P99 (and P95, P99.9) are the industry-standard latency metrics for production systems.
- **Option C (Incorrect):** The mean IS well-defined for exponential distributions — it equals the `scale` parameter (20ms here). Exponential distributions have a finite mean.
- **Option D (Incorrect):** P99 does NOT remove outlier values. It simply marks the value at the 99th rank position. All 1000 latency values (including the 50 slow ones) are included in the computation. P99 ≈ 99th value when sorted = somewhere in the 500–2000ms range.

</details>

---

### 83. Question

A developer creates a histogram for **training data quality audit** and observes that 3 features show this pattern after `plt.hist()`:

```python
for col in ["age", "salary", "credit_score"]:
    data_col = df[col].dropna().values
    plt.figure()
    counts, bins, _ = plt.hist(data_col, bins=30)
    plt.title(col)
    plt.show()
```

The `age` histogram shows values from 0–120 (normal), but `salary` shows a sharp spike at exactly 999999 with all other bars near zero. What does the spike at 999999 indicate?

- [ ] **A.** A small number of employees earn $999,999 — this is a legitimate extreme value
- [ ] **B.** 999999 is a **sentinel/placeholder value** used to encode missing salaries (e.g., "salary unknown") — this is a data quality issue where nulls were encoded as a magic number instead of `NaN`
- [ ] **C.** The histogram has a binning error — `bins=30` is too few for salary data
- [ ] **D.** The histogram is correct — log-normal salary distributions always have a small cluster at the maximum value

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While some individuals earn very high salaries, a sharp isolated spike at EXACTLY 999999 with all other bars dwarfed is not consistent with a legitimate salary distribution. Real high-earner distributions have smooth tails, not sharp isolated spikes at a round number.
- **Option B (Correct):** `999999` (or `9999`, `-1`, `0`, `99999`) are classic **sentinel values** — special numeric codes used in legacy systems to represent "missing," "unknown," or "not applicable." When data engineers encode missing values as magic numbers instead of `NaN`, they contaminate the feature with artificial outliers. The histogram reveals this: a spike at exactly 999999, isolated from all realistic salary values, is a strong indicator of a sentinel-encoded missing value. The fix: `df["salary"].replace(999999, np.nan)` before modeling.
- **Option C (Incorrect):** `bins=30` is a reasonable choice for salary data. The spike at 999999 would be visible at any bin count — it is a genuine data artifact, not a binning artifact.
- **Option D (Incorrect):** Log-normal salary distributions produce smooth right tails — not isolated spikes at any specific value. The "maximum value spike" pattern is unique to encoded sentinel values.

</details>

---

### 84. Question

A developer creates a **model error analysis** histogram comparing absolute errors across two models:

```python
np.random.seed(13)
model_a_errors = np.abs(np.random.normal(0, 5, 400))   # MAE ≈ 4
model_b_errors = np.abs(np.random.normal(0, 3, 400))   # MAE ≈ 2.4

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(model_b_errors, bins=25, alpha=0.6, color="mediumseagreen", label="Model B")
ax.hist(model_a_errors, bins=25, alpha=0.6, color="tomato", label="Model A")
ax.axvline(np.mean(model_a_errors), color="darkred",   linestyle="--", linewidth=2)
ax.axvline(np.mean(model_b_errors), color="darkgreen", linestyle="--", linewidth=2)
ax.set_xlabel("Absolute Error"); ax.legend()
```

The absolute error `|N(0, σ)|` follows a **half-normal distribution**. What is the shape of a half-normal histogram?

- [ ] **A.** Symmetric bell curve centered at 0
- [ ] **B.** Right-skewed: peak at 0, decaying toward higher errors — most predictions are accurate (small error) with progressively fewer large errors
- [ ] **C.** Uniform: equal number of predictions at each error level
- [ ] **D.** Left-skewed: peak far from 0, with most predictions having large errors

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.abs(normal)` takes only the non-negative half of the Gaussian. The result is NOT symmetric — it is a half-normal distribution, asymmetric with a peak at 0.
- **Option B (Correct):** The **half-normal distribution** (absolute value of a zero-mean normal) is peaked at 0 and right-skewed: most absolute errors are small (near 0), with progressively fewer large errors extending into the right tail. The histogram shows a peak near zero absolute error (accurate predictions) and a long right tail of occasional large errors. Model B (σ=3) has a narrower, taller peak near 0 than Model A (σ=5), indicating better overall accuracy. The mean absolute error (MAE) is the mean of this distribution: MAE = σ×√(2/π) ≈ 0.798σ.
- **Option C (Incorrect):** A uniform distribution of absolute errors would mean the model makes equally many 1-unit errors, 10-unit errors, and 100-unit errors — this would indicate zero predictive ability and a systematically broken model.
- **Option D (Incorrect):** Left-skewed with peak far from 0 would mean most predictions are wildly inaccurate with only occasional accurate predictions — a disastrously performing model, not a typical regression model.

</details>

---

### 85. Question

A developer adds **annotation text** to the tallest bar of a histogram:

```python
counts, bins, patches = plt.hist(data, bins=15, edgecolor="black")

# Find tallest bar and annotate it
max_idx = np.argmax(counts)
max_count = counts[max_idx]
bar_center = (bins[max_idx] + bins[max_idx + 1]) / 2

plt.annotate(f"Peak: {int(max_count)}",
             xy=(bar_center, max_count),
             xytext=(bar_center + 2, max_count + 5),
             arrowprops=dict(arrowstyle="->", color="red"),
             fontsize=10, color="red")
```

What do `xy` and `xytext` represent in `plt.annotate()`?

- [ ] **A.** `xy` = figure coordinates of the arrowhead tip; `xytext` = figure coordinates of the text label
- [ ] **B.** `xy` = data coordinates of the point being annotated (arrowhead tip at the bar top); `xytext` = data coordinates of where the annotation text is placed (offset from the bar)
- [ ] **C.** `xy` = pixel coordinates; `xytext` = normalized axes coordinates
- [ ] **D.** Both `xy` and `xytext` are in normalized figure coordinates (0 to 1)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `xy` and `xytext` default to **data coordinates**, not figure coordinates. Figure coordinates would require specifying `xycoords='figure fraction'`.
- **Option B (Correct):** By default, both `xy` and `xytext` in `plt.annotate()` use the **axes data coordinate system** — the same coordinate system as the plotted data. `xy=(bar_center, max_count)` places the arrowhead tip at the top of the tallest bar (in data x and y coordinates). `xytext=(bar_center + 2, max_count + 5)` places the "Peak: N" text label 2 data units to the right and 5 data units above the bar top. The `arrowprops` dictionary draws an arrow from the text location to the `xy` point. This is standard annotation for highlighting histogram features.
- **Option C (Incorrect):** `plt.annotate()` does not mix coordinate systems by default. Both `xy` and `xytext` use the same coordinate system (default: data coordinates). Different systems can be specified via `xycoords=` and `textcoords=` keywords.
- **Option D (Incorrect):** Normalized figure coordinates (0 to 1) would require `xycoords='figure fraction'` and `textcoords='figure fraction'`. Without these keywords, both positions are in data coordinates.

</details>

---

### 86. Question

A developer writes a function to detect **distribution type** from histogram characteristics:

```python
def classify_distribution(data):
    mean_val   = np.mean(data)
    median_val = np.median(data)
    skew_ratio = (mean_val - median_val) / np.std(data)

    if abs(skew_ratio) < 0.1:
        return "Symmetric (possibly normal or uniform)"
    elif skew_ratio > 0.1:
        return "Right-skewed"
    else:
        return "Left-skewed"
```

The function uses `(mean - median) / std` as a skewness proxy. For a dataset where `mean=45`, `median=52`, `std=15`, what does the function return?

- [ ] **A.** `"Right-skewed"` — mean > median
- [ ] **B.** `"Left-skewed"` — mean < median, skew_ratio = (45-52)/15 = -0.467
- [ ] **C.** `"Symmetric"` — the difference is within ±0.1
- [ ] **D.** `"Right-skewed"` — standard deviation > 10 always indicates right skew

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Right-skew requires mean > median. Here mean=45 < median=52 — this is left-skew, not right-skew.
- **Option B (Correct):** `skew_ratio = (mean - median) / std = (45 - 52) / 15 = -7 / 15 = -0.467`. Since `-0.467 < -0.1`, the else branch executes: `return "Left-skewed"`. The function uses Pearson's second skewness coefficient (`3×(mean-median)/std` in its standard form, here simplified). Mean < median indicates the left tail pulls the mean below the median — left-skewed (negative skew). The magnitude -0.467 is well beyond the ±0.1 symmetric threshold.
- **Option C (Incorrect):** `abs(-0.467) = 0.467` which is NOT < 0.1. The symmetric threshold requires |skew_ratio| < 0.1 — clearly not satisfied here.
- **Option D (Incorrect):** Standard deviation magnitude has no relationship to skewness direction. A large std simply indicates high spread — it is orthogonal to whether the distribution is left-skewed, right-skewed, or symmetric.

</details>

---

### 87. Question

A developer creates a histogram of **bootstrap sample means** to visualize the sampling distribution:

```python
np.random.seed(25)
population = np.random.exponential(scale=30, size=10000)

bootstrap_means = []
for _ in range(1000):
    sample = np.random.choice(population, size=100, replace=True)
    bootstrap_means.append(np.mean(sample))

plt.hist(bootstrap_means, bins=30, edgecolor="black", density=True, color="goldenrod")
plt.axvline(np.mean(population), color="red", linewidth=2, linestyle="--",
            label=f"Population Mean: {np.mean(population):.1f}")
plt.xlabel("Sample Mean"); plt.title("Bootstrap Distribution of Sample Means")
plt.legend()
```

What distribution shape should `bootstrap_means` form according to the **Central Limit Theorem**?

- [ ] **A.** Right-skewed (exponential) — the sampling distribution inherits the shape of the population
- [ ] **B.** Approximately normal (Gaussian) — regardless of the population's right-skewed shape, the distribution of sample means converges to normal for large enough sample sizes
- [ ] **C.** Bimodal — two clusters form from averaging different parts of the exponential distribution
- [ ] **D.** Uniform — averaging 100 samples always produces a flat distribution

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The **Central Limit Theorem (CLT)** specifically states that the sampling distribution of the mean converges to normal REGARDLESS of the underlying population distribution — even if the population is exponential, Poisson, or otherwise non-normal.
- **Option B (Correct):** The **CLT** guarantees: for independent random samples of size n ≥ 30 (approximately), the distribution of sample means X̄ converges to a **normal distribution** with mean = μ (population mean) and standard error = σ/√n. Here: n=100, population is Exponential(30), population σ ≈ 30. SE = 30/√100 = 3.0. The bootstrap means should form a bell curve centered at the population mean ≈ 30, with std ≈ 3.0. The histogram of `bootstrap_means` beautifully demonstrates the CLT in action — the exponential population produces a normal sampling distribution.
- **Option C (Incorrect):** Bimodality in sample means would require a bimodal population AND very small n. For n=100 with a unimodal exponential population, the CLT strongly dominates — sample means are unimodal and approximately normal.
- **Option D (Incorrect):** Averaging produces a distribution concentrated around the population mean with reduced variance — NOT uniform. A uniform distribution of sample means would violate the variance-reducing property of averaging.

</details>

---

### 88. Question

A developer creates a **histogram of residuals vs. fitted values** diagnostic for heteroscedasticity:

```python
# Split residuals into 3 groups based on fitted value magnitude
low_fitted    = residuals[fitted < np.percentile(fitted, 33)]
medium_fitted = residuals[(fitted >= np.percentile(fitted, 33)) &
                           (fitted < np.percentile(fitted, 67))]
high_fitted   = residuals[fitted >= np.percentile(fitted, 67)]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, group, title in zip(axes,
    [low_fitted, medium_fitted, high_fitted],
    ["Low Fitted", "Medium Fitted", "High Fitted"]):
    ax.hist(group, bins=20, density=True, edgecolor="black")
    ax.axvline(0, color="red", linewidth=1.5)
    ax.set_title(title)
```

If the histograms show narrow (low spread) residuals for low fitted values but wide (high spread) residuals for high fitted values, what does this indicate?

- [ ] **A.** The model is well-specified — narrow residuals at low fitted values confirm accuracy
- [ ] **B.** Heteroscedasticity — residual variance increases with fitted value magnitude, violating OLS regression assumptions
- [ ] **C.** The data has bimodal distribution in the target variable
- [ ] **D.** The `sharey=True` parameter is causing the wide spread illusion

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Narrow residuals at LOW fitted values do not confirm overall model accuracy — they indicate the model predicts well for smaller-magnitude predictions but may struggle for larger-magnitude predictions.
- **Option B (Correct):** **Heteroscedasticity** (non-constant variance): when residual spread increases systematically with the magnitude of fitted values, the assumption of constant error variance (homoscedasticity) required by OLS regression is violated. Histogram comparison across fitted-value groups is a visual diagnostic: if the "Low Fitted" histogram is narrow (residuals within ±2) and "High Fitted" is wide (residuals within ±20), heteroscedasticity is confirmed. Remedies include: log-transforming the target, weighted least squares, or robust regression.
- **Option C (Incorrect):** Bimodality in the target variable would show up in the target distribution histogram — not necessarily in the residuals. Residual spread increasing with fitted value is specifically the signature of heteroscedasticity, not bimodal targets.
- **Option D (Incorrect):** `sharey=True` fixes the y-axis scale across all three subplots — this HELPS fair comparison (not distorts it). It ensures the y-axis range is consistent, so differences in histogram width (x-spread) reflect genuine residual variance differences, not scale artifacts.

</details>

---

### 89. Question

A developer uses `np.histogram()` (without Matplotlib) to compute bin counts, then plots manually:

```python
np.random.seed(77)
data = np.random.normal(70, 10, 300)

counts, bin_edges = np.histogram(data, bins=15, density=False)
bin_widths = np.diff(bin_edges)
bin_centers = bin_edges[:-1] + bin_widths / 2

# Plot using bar chart instead of hist
plt.bar(bin_centers, counts, width=bin_widths * 0.9, edgecolor="black")
plt.xlabel("Score"); plt.ylabel("Count")
```

What is the advantage of using `np.histogram()` + `plt.bar()` over directly calling `plt.hist()`?

- [ ] **A.** `np.histogram()` is always more accurate than `plt.hist()` for density calculations
- [ ] **B.** The two-step approach allows modifying `counts` before plotting — e.g., smoothing, re-normalizing, or selectively dropping bins — without affecting the underlying data
- [ ] **C.** `plt.bar()` produces vector-format output while `plt.hist()` only produces raster
- [ ] **D.** `np.histogram()` supports more bin edge options than `plt.hist()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.hist()` internally calls `np.histogram()` for its computation. They use the same algorithm and produce identical numerical results. There is no accuracy difference.
- **Option B (Correct):** Separating computation (`np.histogram()`) from visualization (`plt.bar()`) gives full control over the counts array before plotting. Use cases: (1) **Smoothing**: apply a moving average to `counts` to reduce bin-to-bin noise. (2) **Re-normalization**: custom normalization (not just density=True). (3) **Selective plotting**: zero out certain bins (e.g., hide outlier bins). (4) **Multiple representations**: use the same `counts` for different plot types (bar chart, step line, filled area). This is powerful when `plt.hist()` alone doesn't provide sufficient control over post-computation transformations.
- **Option C (Incorrect):** Both `plt.bar()` and `plt.hist()` produce the same type of Matplotlib output — raster or vector depending on the save format (PNG = raster, PDF/SVG = vector). The plotting function used does not determine the output format.
- **Option D (Incorrect):** `plt.hist()` passes its bin arguments directly to `np.histogram()`. Both support the same bin specifications: integer count, sequence of edges, or string algorithms ('auto', 'fd', 'scott', 'rice', etc.).

</details>

---

### 90. Question

A developer creates the following **A/B test diagnostic** histogram and must select the correct interpretation:

```python
np.random.seed(34)
control_revenue   = np.random.lognormal(5.5, 0.6, 500)   # Control group
treatment_revenue = np.random.lognormal(5.7, 0.65, 500)  # Treatment group

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(control_revenue,   bins=35, alpha=0.55, color="steelblue",
        density=True, label=f"Control (mean=${np.mean(control_revenue):.0f})")
ax.hist(treatment_revenue, bins=35, alpha=0.55, color="darkorange",
        density=True, label=f"Treatment (mean=${np.mean(treatment_revenue):.0f})")
ax.axvline(np.mean(control_revenue),   color="navy",   linestyle="--", linewidth=2)
ax.axvline(np.mean(treatment_revenue), color="saddlebrown", linestyle="--", linewidth=2)
ax.set_xlabel("Revenue per User ($)"); ax.legend()
ax.set_title("A/B Test: Revenue Distribution")
```

The treatment mean is higher, but the histograms **overlap substantially**. What additional analysis is needed beyond visual inspection?

- [ ] **A.** More bins — `bins=100` will separate the distributions visually
- [ ] **B.** A statistical significance test (e.g., t-test or Mann-Whitney U test) to determine whether the mean difference is statistically significant or could be explained by random sampling variability
- [ ] **C.** Using `density=False` instead — raw counts provide more conclusive evidence
- [ ] **D.** The substantial overlap confirms the treatment had no effect — no further analysis needed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** More bins increase granularity but do not change the underlying distributional overlap. The overlap exists because the two distributions genuinely have similar ranges — finer bins simply show the same overlap more granularly without adding analytical power.
- **Option B (Correct):** **Statistical hypothesis testing** is the necessary next step. Visual histogram overlap does NOT determine statistical significance. Even if two distributions overlap substantially, their means can still be statistically significantly different if the sample size is large enough and the effect size is real. For A/B revenue tests: (1) **t-test** (parametric, assumes normality — reasonable after log transform) or (2) **Mann-Whitney U test** (non-parametric, appropriate for right-skewed revenue data). The test produces a p-value indicating the probability of observing this mean difference by chance under the null hypothesis. A histogram alone cannot replace statistical inference.
- **Option C (Incorrect):** Switching to `density=False` changes bar heights from probability density to raw counts. For two groups of equal size (n=500 each), raw counts are proportional to density — the visual comparison remains the same. `density=True` is actually better for shape comparison and was correctly chosen.
- **Option D (Incorrect):** Histogram overlap does NOT imply zero effect. Two distributions can substantially overlap AND have a statistically significant mean difference. Effect size and overlap are related but not equivalent — a small effect (low Cohen's d) produces high overlap; a large effect produces low overlap. Visual overlap is a rough guide but never conclusive without formal testing.

</details>
