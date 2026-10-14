# Matplotlib Histogram Objective Test — Set 10

---

### 91. Question

A developer builds a **complete ML feature audit pipeline** and examines the following capstone code:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis

np.random.seed(42)
features = {
    "income":      np.random.lognormal(10, 1.2, 1000),
    "age":         np.random.normal(42, 12, 1000),
    "loan_amount": np.random.exponential(15000, 1000),
    "credit_util": np.random.beta(2, 5, 1000),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Distribution Audit", fontsize=15, fontweight="bold")

for ax, (name, data) in zip(axes.flatten(), features.items()):
    ax.hist(data, bins=30, density=True, edgecolor="black", alpha=0.75)
    ax.axvline(np.mean(data),   color="red",   linestyle="--", label="Mean")
    ax.axvline(np.median(data), color="lime",  linestyle="-",  label="Median")
    ax.set_title(f"{name}\nskew={skew(data):.2f}, kurt={kurtosis(data):.2f}")
    ax.legend(fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("feature_audit.png", dpi=200, bbox_inches="tight")
plt.close()
```

For the `income` feature (log-normal), which statement is TRUE about the mean and median lines?

- [ ] **A.** Mean line (red) appears to the LEFT of the median line (lime)
- [ ] **B.** Mean line (red) appears to the RIGHT of the median line (lime), and the skew value will be POSITIVE
- [ ] **C.** Mean and median coincide — log-normal distributions are symmetric
- [ ] **D.** Median line appears to the right of the mean line, and skew is negative

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Mean to the LEFT of median describes left-skewed distributions (mean < median). Log-normal distributions are right-skewed — mean > median.
- **Option B (Correct):** Log-normal distributions are **right-skewed (positively skewed)**. The mathematical relationship: log-normal mean = exp(μ + σ²/2) > exp(μ) = log-normal median. For `lognormal(10, 1.2)`: mean = exp(10 + 0.72) ≈ exp(10.72) ≈ 45,150; median = exp(10) ≈ 22,026. The mean (red dashed) appears **to the right** of the median (lime solid) in the histogram. The `skew()` function returns a **positive value** for right-skewed distributions (scipy's skewness formula gives positive values when the right tail dominates). This is also confirmed by the long right tail visible in the income histogram.
- **Option C (Incorrect):** Log-normal distributions are NOT symmetric. Symmetry would give mean = median = mode — this only holds for the normal distribution. Log-normal has mode < median < mean.
- **Option D (Incorrect):** This describes left-skewed behavior (median > mean, negative skew). Log-normal is right-skewed — the opposite applies.

</details>

---

### 92. Question

In the capstone code from Question 91, the developer uses `plt.tight_layout(rect=[0, 0, 1, 0.95])`. A colleague asks why `rect=[0, 0, 1, 0.95]` is used instead of simply `plt.tight_layout()`. What is the correct explanation?

- [ ] **A.** `rect=[0, 0, 1, 0.95]` sets the figure background color to 95% opacity
- [ ] **B.** `rect=[0, 0, 1, 0.95]` reserves the top 5% of the figure for `fig.suptitle()` — preventing the super title from overlapping the subplot titles
- [ ] **C.** `rect` controls the DPI of the saved figure — 0.95 sets 95% of maximum DPI
- [ ] **D.** `rect=[0, 0, 1, 0.95]` is identical to plain `tight_layout()` — the default rect is [0, 0, 1, 1]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `rect` in `tight_layout()` is a layout boundary specification in normalized figure coordinates — it has nothing to do with opacity or background color.
- **Option B (Correct):** `plt.tight_layout(rect=[left, bottom, right, top])` constrains all subplots to fit within the specified rectangle in normalized figure coordinates. `rect=[0, 0, 1, 0.95]` means subplots occupy from (0,0) to (1, 0.95) — leaving the top 5% (y=0.95 to y=1.0) free for the `fig.suptitle("Feature Distribution Audit")`. Without this, `tight_layout()` positions subplots to use the full height, causing the suptitle to overlap the top subplot titles. This is the standard pattern for suptitle + tight_layout coexistence.
- **Option C (Incorrect):** DPI is controlled by the `dpi=` parameter in `plt.savefig()` or `plt.figure()`. `tight_layout(rect=...)` has no effect on DPI.
- **Option D (Incorrect):** While the default `rect` in `tight_layout()` is indeed `[0, 0, 1, 1]`, explicitly passing `[0, 0, 1, 0.95]` is NOT identical — it deliberately shrinks the top boundary from 1.0 to 0.95, reserving space for the suptitle. The difference of 5% in the top boundary is exactly the point of using this parameter.

</details>

---

### 93. Question

A developer calculates the **Freedman-Diaconis rule** for optimal bin width and compares it with Sturges' rule:

```python
np.random.seed(50)
data = np.random.exponential(scale=25, size=500)

n = len(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

# Freedman-Diaconis: bin_width = 2 * IQR * n^(-1/3)
fd_width = 2 * iqr * (n ** (-1/3))
fd_bins  = int(np.ceil((data.max() - data.min()) / fd_width))

# Sturges: bins = ceil(1 + log2(n))
import math
sturges_bins = math.ceil(1 + math.log2(n))

print(f"FD bins: {fd_bins}, Sturges bins: {sturges_bins}")
```

For `n=500` samples from an exponential distribution, why does the **Freedman-Diaconis** rule typically produce MORE bins than Sturges' rule?

- [ ] **A.** Freedman-Diaconis overestimates bin count because it uses IQR instead of std
- [ ] **B.** Sturges' rule is derived for normal distributions and uses log₂(n) ≈ 9–10 bins for n=500; Freedman-Diaconis adapts to the actual data spread via IQR and scales as n^(1/3) ≈ 7.9, leading to finer bins for skewed or heavy-tailed data
- [ ] **C.** Freedman-Diaconis always produces exactly 2× more bins than Sturges
- [ ] **D.** Both rules produce identical bin counts for n=500

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Using IQR instead of std is not a flaw — it is a deliberate design choice. IQR is **robust to outliers** (which are common in exponential distributions). Std is inflated by extreme values, which would produce even wider bins. IQR gives a more accurate measure of typical data spread.
- **Option B (Correct):** **Sturges' rule** (k = 1 + log₂(n)) was designed for **normally distributed** data. For n=500: Sturges gives ≈ 10 bins. **Freedman-Diaconis** uses bin_width = 2 × IQR × n^(-1/3) and scales with n^(1/3) (cube root) rather than log₂(n). For n=500: n^(1/3) ≈ 7.94. For an exponential distribution with scale=25: IQR ≈ 17.3 (ln(2)×25 to ... range). FD produces finer bins (more bin count) to reveal the exponential tail structure that Sturges would collapse into too few bins. FD is preferred for non-normal, skewed, or heavy-tailed distributions.
- **Option C (Incorrect):** There is no fixed ratio between FD and Sturges bin counts. The ratio depends on sample size, IQR, and data range — it varies with every dataset.
- **Option D (Incorrect):** The two rules use fundamentally different formulas and scale differently with n. For n=500, Sturges gives ≈10 bins while FD typically gives 20–40+ bins for right-skewed data with n=500.

</details>

---

### 94. Question

A developer creates a **multi-select debugging challenge**. The following code has **multiple bugs**. Which lines are INCORRECT?

```python
np.random.seed(99)
data = np.random.normal(50, 10, 400)

# Line 1: Compute histogram
counts, bins = plt.hist(data, bins=20, density=True)       # Line 1

# Line 2: Verify area
total = np.sum(counts)                                       # Line 2
print(f"Total area: {total:.3f}")  # Developer expects ~1.0

# Line 3: Find bin containing value 55
target = 55
bin_idx = np.searchsorted(bins, target) - 1                # Line 3

# Line 4: Plot median line
plt.axvline(np.median, color="red", linestyle="--")         # Line 4
```

- [ ] **A.** Line 1 and Line 4
- [ ] **B.** Line 1, Line 2, and Line 4
- [ ] **C.** Line 2 and Line 3
- [ ] **D.** All four lines contain bugs

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Line 1 (BUG):** `plt.hist()` returns 3 values `(counts, bins, patches)`. Unpacking into only 2 variables raises `ValueError: too many values to unpack`. Fix: `counts, bins, patches = plt.hist(...)` or `counts, bins, _ = plt.hist(...)`.
- **Line 2 (BUG):** `np.sum(counts)` sums raw density heights — NOT the total area. With `density=True`, the area constraint is `np.sum(counts * np.diff(bins)) ≈ 1.0`, not `np.sum(counts) ≈ 1.0`. For 20 equal-width bins on [30,70] range: bin_width ≈ 2.0, so `np.sum(counts) ≈ 1.0/2.0 = 0.5`. The developer's expectation of ~1.0 from `np.sum(counts)` is incorrect. Fix: `total = np.sum(counts * np.diff(bins))`.
- **Line 3 (CORRECT):** `np.searchsorted(bins, 55)` finds the insertion index of 55 in the sorted `bins` array, then `-1` gives the bin index containing 55. This is a valid and correct approach for finding bin indices.
- **Line 4 (BUG):** `np.median` is a **function object** — it is not called here (missing parentheses and argument). `plt.axvline(np.median, ...)` passes the function object itself, not a numeric value. Fix: `plt.axvline(np.median(data), color="red", linestyle="--")`.
- **Option B is correct:** Lines 1, 2, and 4 contain bugs. Line 3 is correct.

</details>

---

### 95. Question

A developer creates a histogram for **model output calibration analysis** comparing three models:

```python
np.random.seed(14)
# True labels: 500 positives, 500 negatives
y_true = np.array([1]*500 + [0]*500)

# Model outputs: predicted probabilities for positive class
overconfident = np.where(y_true==1,
                          np.random.beta(9, 1, 1000),
                          np.random.beta(1, 9, 1000))
underconfident = np.where(y_true==1,
                           np.random.beta(2, 2, 1000),
                           np.random.beta(2, 2, 1000))
calibrated    = np.where(y_true==1,
                          np.random.beta(7, 3, 1000),
                          np.random.beta(3, 7, 1000))
```

For the **underconfident** model, what does its probability histogram likely show?

- [ ] **A.** Two sharp peaks near 0.0 and 1.0 — the model makes decisive predictions
- [ ] **B.** A broad distribution concentrated near 0.5 — the model fails to distinguish positive from negative cases, assigning similar probabilities regardless of true label
- [ ] **C.** A right-skewed distribution with most probabilities near 1.0
- [ ] **D.** A uniform distribution between 0 and 1 — all probability values equally likely

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Sharp peaks near 0.0 and 1.0 describe the **overconfident** model — it makes decisive predictions (high probability for positives, low for negatives). The underconfident model is the opposite.
- **Option B (Correct):** The underconfident model uses `Beta(2,2)` for BOTH positive and negative samples — an identical, symmetric distribution centered at 0.5 for all samples regardless of true label. This means the model outputs probabilities concentrated near 0.5 for all samples — it cannot discriminate positive from negative. A histogram of all 1000 outputs would show a single peak near 0.5. This is the signature of a **poorly discriminating, underconfident model** — it "hedges" all predictions toward 50-50 uncertainty without committing to either class.
- **Option C (Incorrect):** Right-skewed with most probabilities near 1.0 would describe a model that always predicts positive — a maximally biased (over-predicting positive) model, not an underconfident one.
- **Option D (Incorrect):** A uniform distribution between 0 and 1 would be `Beta(1,1)` — theoretically the worst possible model (random output). The underconfident model here uses `Beta(2,2)` which is concentrated near 0.5, not uniformly spread across [0,1].

</details>

---

### 96. Question

A developer writes the following **production-safe histogram batch generator** with proper memory management:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_feature_histogram(data: np.ndarray, feature_name: str,
                                 save_path: str, bins: int = 30) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, patches = ax.hist(data, bins=bins, density=True,
                                          edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(data),   color="red",  linestyle="--",
               label=f"Mean={np.mean(data):.2f}")
    ax.axvline(np.median(data), color="lime", linestyle="-",
               label=f"Median={np.median(data):.2f}")
    ax.set_xlabel(feature_name); ax.set_ylabel("Density")
    ax.set_title(f"Distribution: {feature_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "skewness": float(((np.mean(data)-np.median(data))/np.std(data))),
        "n_bins": bins,
        "max_density": float(counts.max()),
    }
```

Which aspects of this function demonstrate **production-best-practices**? (Select all that apply)

- [ ] **A.** `matplotlib.use("Agg")` — sets non-interactive backend before pyplot import for headless server execution
- [ ] **B.** `plt.close(fig)` — deregisters figure from manager after saving, preventing memory accumulation in batch loops
- [ ] **C.** `facecolor=fig.get_facecolor()` — ensures the saved file uses the figure's actual background color
- [ ] **D.** Returning a `dict` of computed statistics — enables programmatic downstream processing beyond just the visual output

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C, D**

#### Explanations:

- **Option A (Correct — TRUE):** `matplotlib.use("Agg")` before `import matplotlib.pyplot` is the correct pattern for headless servers (Docker containers, CI/CD, batch jobs without a display). Agg is a non-interactive raster backend that doesn't require a display. Setting it BEFORE pyplot import ensures it takes effect correctly.
- **Option B (Correct — TRUE):** `plt.close(fig)` after `savefig()` is the essential memory management step for batch figure generation. Without it, every function call adds a figure object to Matplotlib's figure manager, consuming RAM proportional to figure count. For batch processing 1000+ features, omitting `plt.close(fig)` would cause memory exhaustion.
- **Option C (Correct — TRUE):** `facecolor=fig.get_facecolor()` ensures the saved PNG uses the figure's actual background color. Without this, `savefig()` may default to white or "auto" depending on Matplotlib version — potentially overriding a custom figure facecolor. Explicitly passing the current facecolor is defensive programming that works across all Matplotlib versions.
- **Option D (Correct — TRUE):** Returning computed statistics as a dictionary allows the caller to: aggregate statistics across all features, sort features by skewness, identify features needing transformation, log metrics to MLflow/Weights & Biases, etc. A function that ONLY saves a plot file provides no programmatic interface — returning statistics makes the function far more useful in production ML pipelines. All four aspects are genuine production best practices.

</details>

---

### 97. Question

A developer interprets histograms for **target variable diagnostics** in a regression problem:

```python
np.random.seed(60)
house_prices = np.random.lognormal(12.5, 0.5, 800)  # Right-skewed prices

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.hist(house_prices, bins=30, edgecolor="black", color="steelblue", density=True)
ax1.set_title("Raw Price (Right-Skewed)"); ax1.set_xlabel("Price ($)")

ax2.hist(np.log(house_prices), bins=30, edgecolor="black", color="salmon", density=True)
ax2.set_title("Log Price (Approximately Normal)"); ax2.set_xlabel("log(Price)")
```

When training a **linear regression model**, why might log-transforming the target variable (right panel) be preferable?

- [ ] **A.** Log transformation reduces the number of training samples needed
- [ ] **B.** Linear regression assumes normally distributed residuals — a log-normal target transformed to log scale satisfies this assumption better, improving coefficient estimates and prediction intervals
- [ ] **C.** Log-transformed targets always improve R² scores
- [ ] **D.** The log transformation removes the need for feature scaling

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Transforming the target variable does not change the number of training samples. All 800 observations are still used in either case.
- **Option B (Correct):** OLS linear regression theory assumes **normally distributed residuals** (or equivalently, that the relationship between features and target is linear and errors are homoscedastic). A right-skewed target (raw house prices) violates these assumptions: residuals will also be right-skewed, standard errors and p-values will be incorrect, and prediction intervals will be asymmetric. Log-transforming a log-normal target produces approximately normal residuals, satisfying OLS assumptions. The model then predicts log(price), which can be exponentiated back to the original scale. The second histogram (right panel) confirming near-normality of log(price) is the visual justification for this transformation choice.
- **Option C (Incorrect):** Log transformation does not guarantee R² improvement. If the linear relationship is weak or the features are uninformative, R² remains low regardless of target transformation.
- **Option D (Incorrect):** Feature scaling (standardization/normalization of input features) is separate from target variable transformation. Log-transforming the target does not affect whether input features need scaling.

</details>

---

### 98. Question

A developer creates a histogram for **class conditional feature analysis** in a binary classification problem:

```python
np.random.seed(71)
feature_class0 = np.random.normal(40, 8, 600)   # Negative class
feature_class1 = np.random.normal(55, 10, 400)  # Positive class

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(feature_class0, bins=25, density=True, alpha=0.6,
        color="#4CAF50", label="Class 0 (n=600)")
ax.hist(feature_class1, bins=25, density=True, alpha=0.6,
        color="#F44336", label="Class 1 (n=400)")
ax.axvline(50, color="black", linewidth=2, linestyle="--", label="Decision Boundary")
ax.legend(); ax.set_xlabel("Feature Value")
ax.set_title("Class-Conditional Feature Distribution")
```

The two distributions **substantially overlap** in the range 35–65. What does this overlap imply for a classifier using only this feature?

- [ ] **A.** The classifier will achieve 100% accuracy because the means are different
- [ ] **B.** The classifier will make irreducible errors in the overlap region — samples in this range have mixed class membership, making perfect discrimination impossible with this feature alone
- [ ] **C.** The overlap is caused by using `density=True` — switching to raw counts will eliminate it
- [ ] **D.** The decision boundary at 50 is wrong — moving it to 47.5 will eliminate all classification errors

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Different means do NOT guarantee perfect accuracy. The overlap region contains samples from BOTH classes — even the optimal classifier makes errors in this zone because the same feature value can belong to either class.
- **Option B (Correct):** The **Bayes error rate** (irreducible error) is positive whenever class-conditional distributions overlap. In the overlap region [35–65], both Class 0 and Class 1 samples are present. For any decision boundary in this region, some samples will be misclassified. The optimal Bayes classifier minimizes but cannot eliminate this error — it is irreducible given only this feature. The overlap is a fundamental property of the data distributions, not a modeling artifact. Adding more features with better separability can reduce (but not necessarily eliminate) this irreducible error.
- **Option C (Incorrect):** `density=True` normalizes bar heights — it does NOT affect the distributional overlap. The raw data samples from both classes fall in the same range regardless of y-axis normalization. Switching to raw counts changes bar heights but the x-axis overlap is identical.
- **Option D (Incorrect):** No single threshold eliminates all classification errors when distributions overlap. Moving the boundary from 50 to 47.5 reduces false positives but increases false negatives — it optimizes the tradeoff (e.g., toward higher precision or recall) but cannot achieve zero total error when distributions overlap.

</details>

---

### 99. Question

A developer reviews a **colleague's histogram code** and identifies a subtle behavioral issue:

```python
np.random.seed(22)
data = np.random.normal(100, 20, 300)

# Colleague's code
fig, ax = plt.subplots()
n, bins, patches = ax.hist(data, bins=15)

# Colleague tries to add KDE overlay using histogram data
from scipy.stats import gaussian_kde
kde = gaussian_kde(bins)          # BUG: using bins instead of data
x_range = np.linspace(bins[0], bins[-1], 200)
ax.plot(x_range, kde(x_range) * len(data) * np.mean(np.diff(bins)), color="red")
```

What is the bug, and what is the correct fix?

- [ ] **A.** `gaussian_kde` should use `counts` not `bins` — `kde = gaussian_kde(counts)`
- [ ] **B.** `gaussian_kde` should use the original `data` array — `kde = gaussian_kde(data)`. Using `bins` fits KDE to 16 edge values, not to the 300 actual data samples
- [ ] **C.** The scaling factor `len(data) * np.mean(np.diff(bins))` is wrong — it should be `len(bins)`
- [ ] **D.** `gaussian_kde` cannot be used with `ax.plot()` — it requires a separate figure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Using `counts` (the frequency values) for KDE would fit a density estimator to 15 count values — also wrong. KDE should estimate the density of the ORIGINAL data distribution.
- **Option B (Correct):** `gaussian_kde(bins)` fits a kernel density estimator to the **16 bin edge values** (a length-16 array of evenly-spaced numbers from min to max). This does NOT represent the original 300 data samples — it represents the histogram coordinate structure. The result is a smooth curve over bin edges, not a smooth approximation of the data distribution. The correct code is `gaussian_kde(data)` — fitting KDE to all 300 original data points. KDE then produces a smooth density estimate of the underlying distribution that can be overlaid on the histogram for comparison.
- **Option C (Incorrect):** The scaling factor `len(data) * np.mean(np.diff(bins))` correctly converts probability density (from KDE) to count scale to match the non-normalized histogram y-axis. This is the standard conversion: density × n_samples × bin_width. The formula is correct.
- **Option D (Incorrect):** `gaussian_kde` is a scipy function that returns a callable density estimator. `kde(x_range)` evaluates density at x_range positions, returning a numpy array. `ax.plot(x_range, y_values)` works perfectly with any numpy array — there is no restriction on using KDE with `ax.plot()`.

</details>

---

### 100. Question

A developer creates a **final capstone diagnostic dashboard** for a complete ML experiment. Identify which statements about the following code are TRUE:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(100)
train  = np.random.normal(0.25, 0.08, 1000)
val    = np.random.normal(0.30, 0.10, 1000)
test   = np.random.normal(0.35, 0.09, 1000)
errors = np.abs(np.random.normal(0, 0.12, 1000))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("ML Experiment Diagnostic Dashboard", fontsize=14, fontweight="bold")

# Panel A: Loss distributions
axes[0,0].hist([train, val, test], bins=25, density=True, alpha=0.6,
               color=["green","blue","red"], label=["Train","Val","Test"])
axes[0,0].set_title("Loss Distributions"); axes[0,0].legend()

# Panel B: Cumulative error CDF
axes[0,1].hist(errors, bins=30, density=True, cumulative=True, histtype='step',
               color="purple", linewidth=2)
axes[0,1].axhline(0.95, color="red", linestyle="--", label="P95")
axes[0,1].set_title("Error CDF"); axes[0,1].legend()

# Panel C: Error density
axes[1,0].hist(errors, bins=30, density=True, edgecolor="black", color="coral")
axes[1,0].axvline(np.mean(errors), color="blue", linestyle="--", label="Mean")
axes[1,0].set_title("Absolute Error Distribution"); axes[1,0].legend()

# Panel D: Log-scale errors
axes[1,1].hist(np.log1p(errors * 100), bins=30, edgecolor="black", color="teal")
axes[1,1].set_title("Log-Scale Error Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("dashboard.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
```

Which statements are TRUE? (Select all that apply)

- [ ] **A.** Panel A uses a list of arrays `[train, val, test]` — this creates a grouped side-by-side histogram with 3 color-coded series
- [ ] **B.** Panel B's CDF step line will reach y=1.0 at the rightmost point because `density=True, cumulative=True` is used
- [ ] **C.** `np.log1p(errors * 100)` in Panel D computes `log(1 + errors×100)` — using `log1p` avoids numerical issues when `errors×100` is near 0
- [ ] **D.** `plt.close(fig)` at the end is unnecessary — figures created with `matplotlib.use("Agg")` are automatically garbage collected after `savefig()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C**

#### Explanations:

- **Option A (Correct — TRUE):** Passing `[train, val, test]` as the first argument to `ax.hist()` with `color=["green","blue","red"]` and `label=["Train","Val","Test"]` creates a **grouped histogram** — side-by-side bars within each bin for the three series. Each series gets its own color and label for the legend. This is the multi-series list-input mode of `plt.hist()`.
- **Option B (Correct — TRUE):** `density=True, cumulative=True` together produce the empirical CDF. By mathematical definition, the CDF must reach **1.0** at the maximum data value — all data has been accumulated by the rightmost bin. `counts[-1] ≈ 1.0` always (within floating-point precision). The `axhline(0.95)` red line intersects the step curve at the 95th percentile of errors.
- **Option C (Correct — TRUE):** `np.log1p(x)` computes `log(1 + x)` with **higher numerical precision** than `np.log(1 + x)` for small x. When `errors×100` values are near 0 (small absolute errors), `1 + errors×100 ≈ 1` and `log(1 + ε) ≈ ε` — `log1p` maintains precision in this regime while `np.log(1 + x)` can suffer from catastrophic cancellation in floating point.
- **Option D (Incorrect — FALSE):** `matplotlib.use("Agg")` does NOT automatically garbage-collect figures after `savefig()`. The Agg backend removes the GUI window (there is none), but the `Figure` Python object remains registered in Matplotlib's figure manager until explicitly closed with `plt.close(fig)`. Without `plt.close(fig)`, batch loops would accumulate figure objects in memory regardless of backend. `plt.close(fig)` is always necessary for proper memory management.

</details>
