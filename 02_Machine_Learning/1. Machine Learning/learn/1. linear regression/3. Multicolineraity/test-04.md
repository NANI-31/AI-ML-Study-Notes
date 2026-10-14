# Multicollinearity Objective Test — Set 04

---

### 1. Question

In statistical theory, how is the Variance Inflation Factor (VIF) for the $j$-th feature calculated?

- [ ] A) By calculating the ratio of the target variable's standard deviation to the feature's standard deviation.
- [ ] B) By regressing the $j$-th feature against all other independent features to obtain $R_j^2$, then calculating $\text{VIF}_j = \frac{1}{1 - R_j^2}$.
- [ ] C) By multiplying the eigenvalues of the covariance matrix by sample size $n$.
- [ ] D) By dividing the mean squared error of the full model by the mean squared error of a model with the $j$-th feature removed.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** VIF measures collinearity among input features, not the scaling relationship with the target variable $y$.
- **Option B (Correct):** To find the VIF of feature $x_j$, we run an "auxiliary regression" predicting $x_j$ using all other predictors $x_1, \dots, x_{j-1}, x_{j+1}, \dots, x_p$. The coefficient of determination from this auxiliary regression is $R_j^2$. The VIF is then defined as $\text{VIF}_j = \frac{1}{1 - R_j^2}$.
- **Option C (Incorrect):** While eigenvalues of $X^T X$ are used to compute condition indices, they do not directly give VIF.
- **Option D (Incorrect):** Comparing full vs. reduced model errors relates to partial F-tests, not the formula for VIF.

</details>

---

### 2. Question

What is the theoretical minimum possible value that a Variance Inflation Factor (VIF) can take?

- [ ] A) $-1.0$
- [ ] B) $0.0$
- [ ] C) $1.0$
- [ ] D) $10.0$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** VIF is based on $R^2 \ge 0$, so the denominator $1 - R^2$ is always $\le 1$ and positive. VIF can never be negative.
- **Option B (Incorrect):** For VIF to equal 0, the denominator would have to be infinite, which is mathematically impossible since $1 - R^2 \in (0, 1]$.
- **Option C (Correct):** When a feature $x_j$ is completely orthogonal (uncorrelated) to all other features, the auxiliary regression explains zero variance ($R_j^2 = 0$). In this case, $\text{VIF}_j = \frac{1}{1 - 0} = 1.0$. A VIF of $1.0$ represents absolute, pristine independence.
- **Option D (Incorrect):** $10.0$ is an informal heuristic threshold for severe collinearity, not the theoretical minimum.

</details>

---

### 3. Question

An auxiliary regression predicting feature $x_3$ from all other features yields an $R_3^2$ of $0.90$.

What is the VIF for feature $x_3$?

- [ ] A) $0.10$
- [ ] B) $1.11$
- [ ] C) $9.00$
- [ ] D) $10.00$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $0.10$ is the tolerance ($1 - R^2$), which is the reciprocal of VIF, not VIF itself.
- **Option B (Incorrect):** Computed by $1 / 0.90 \approx 1.11$, forgetting to subtract $R^2$ from 1.
- **Option C (Incorrect):** Simply multiplying $R^2$ by 10.
- **Option D (Correct):** Applying the definition:
  $$\text{Tolerance} = 1 - R_3^2 = 1 - 0.90 = 0.10$$
  $$\text{VIF}_3 = \frac{1}{1 - R_3^2} = \frac{1}{0.10} = 10.00$$
  This indicates that $90\%$ of the variance in $x_3$ is already accounted for by other features in the model, inflating its coefficient variance by a factor of 10.

</details>

---

### 4. Question

What is the relationship between Variance Inflation Factor (VIF) and "Tolerance" in regression diagnostics?

- [ ] A) $\text{Tolerance} = \text{VIF}^2$
- [ ] B) $\text{Tolerance} = \frac{1}{\text{VIF}}$
- [ ] C) $\text{Tolerance} = \ln(\text{VIF})$
- [ ] D) $\text{Tolerance} = \text{VIF} - 1$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Squaring VIF would magnify large numbers rather than invert them.
- **Option B (Correct):** Tolerance is defined as $1 - R_j^2$, representing the proportion of variance in $x_j$ that is *not* shared with the other predictors. Since $\text{VIF}_j = \frac{1}{1 - R_j^2}$, it follows directly that $\text{Tolerance}_j = \frac{1}{\text{VIF}_j}$. A tolerance below $0.10$ corresponds to a VIF greater than $10$.
- **Option C (Incorrect):** Tolerance is a linear reciprocal relationship, not logarithmic.
- **Option D (Incorrect):** Tolerance is bounded between $0$ and $1$, whereas $\text{VIF} - 1$ can be arbitrarily large.

</details>

---

### 5. Question

If a feature has a VIF of $25.0$, by what factor is the standard error of its estimated regression coefficient inflated compared to if it were completely orthogonal to the other features?

- [ ] A) $2.5$ times
- [ ] B) $5.0$ times
- [ ] C) $25.0$ times
- [ ] D) $625.0$ times

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Dividing by 10 is an arbitrary scaling error.
- **Option B (Correct):** The variance of the estimated coefficient is proportional to the VIF: $\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n-1)s_j^2} \times \text{VIF}_j$. Standard error is the square root of the variance: $\text{SE}(\hat{\beta}_j) \propto \sqrt{\text{VIF}_j}$. Since $\sqrt{25} = 5.0$, the standard error is inflated by exactly $5$ times.
- **Option C (Incorrect):** $25.0$ is the factor by which the *variance* is inflated, not the standard error.
- **Option D (Incorrect):** $625 = 25^2$, which would be squaring the variance inflation factor.

</details>

---

### 6. Question

When using `statsmodels.stats.outliers_influence.variance_inflation_factor`, what common bug occurs if the user fails to prepend a constant column (`add_constant`) to the feature matrix?

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# User passes raw feature matrix without intercept:
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

- [ ] A) The code throws an immediate `KeyError: 'const'` exception.
- [ ] B) The function computes uncentered $R^2$ values, often producing artificially huge VIF scores for features with non-zero means.
- [ ] C) The function returns negative VIF values for all features.
- [ ] D) The function automatically adds a constant behind the scenes, so results are identical.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `variance_inflation_factor` takes a NumPy array and column index; it does not check for named columns and will not raise a `KeyError`.
- **Option B (Correct):** `statsmodels` computes auxiliary regressions through the origin (without an intercept) if no constant column is provided. When features have large non-zero means, an uncentered regression produces an inflated, uncentered $R^2$, leading to massively exaggerated VIF calculations that do not reflect true collinearity.
- **Option C (Incorrect):** Uncentered $R^2$ values in statsmodels remain in $[0, 1]$, so VIFs remain positive.
- **Option D (Incorrect):** Statsmodels does *not* automatically add a constant inside `variance_inflation_factor`; the user must explicitly call `sm.add_constant(X)`.

</details>

---

### 7. Question

Suppose feature $x_1$ has a VIF of $1.02$. What does this indicate about $x_1$?

- [ ] A) $x_1$ is almost entirely redundant with the other features and should be dropped immediately.
- [ ] B) $x_1$ has virtually no linear correlation with any other predictor in the model and provides distinct, unique information.
- [ ] C) $x_1$ has zero correlation with the dependent variable $y$.
- [ ] D) The regression model will experience high coefficient instability for $x_1$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** High redundancy corresponds to large VIF values (e.g., $> 10$), not values near 1.
- **Option B (Correct):** A VIF of $1.02$ means the auxiliary $R^2$ is roughly $0.02$ ($2\%$). The other features explain only $2\%$ of the variance in $x_1$, proving that $x_1$ is virtually independent of the other predictors and carries unique signal.
- **Option C (Incorrect):** VIF measures relationships exclusively among independent variables; it has no knowledge of or relationship with the target variable $y$.
- **Option D (Incorrect):** Low VIF ($1.02$) guarantees minimal variance inflation and maximum coefficient stability.

</details>

---

### 8. Question

Can a feature's VIF ever be calculated as infinite ($\infty$)?

- [ ] A) No, because mathematical packages clamp all outputs between $1.0$ and $100.0$.
- [ ] B) Yes, when a feature can be predicted perfectly ($R_j^2 = 1.0$) from a linear combination of other features, the denominator $1 - R_j^2$ becomes zero.
- [ ] C) Yes, but only when all data values in the feature column are negative numbers.
- [ ] D) No, because floating point numbers cannot represent infinity.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Statistical packages do not arbitrarily clamp VIF; when denominator is zero, NumPy returns `np.inf` or raises a warning.
- **Option B (Correct):** In cases of perfect multicollinearity (e.g., $x_3 = 2x_1 + 5x_2$), the auxiliary regression achieves an $R^2$ of exactly $1.0$. Dividing $1$ by $(1 - 1.0) = 0$ yields a division by zero, resulting in a VIF of infinity ($\infty$).
- **Option C (Incorrect):** The signs of data values do not affect linear regression $R^2$.
- **Option D (Incorrect):** IEEE 754 floating point arithmetic explicitly defines positive and negative infinity (`inf`, `-inf`).

</details>

---

### 9. Question

An analyst is diagnosing multicollinearity across 4 features. The auxiliary $R^2$ values are:
* Feature 1: $R_1^2 = 0.50$
* Feature 2: $R_2^2 = 0.75$
* Feature 3: $R_3^2 = 0.80$
* Feature 4: $R_4^2 = 0.95$

Which feature has a VIF of $20.0$?

- [ ] A) Feature 1
- [ ] B) Feature 2
- [ ] C) Feature 3
- [ ] D) Feature 4

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $\text{VIF}_1 = 1 / (1 - 0.50) = 1 / 0.50 = 2.0$.
- **Option B (Incorrect):** $\text{VIF}_2 = 1 / (1 - 0.75) = 1 / 0.25 = 4.0$.
- **Option C (Incorrect):** $\text{VIF}_3 = 1 / (1 - 0.80) = 1 / 0.20 = 5.0$.
- **Option D (Correct):** $\text{VIF}_4 = \frac{1}{1 - 0.95} = \frac{1}{0.05} = 20.0$. Feature 4 has $95\%$ of its variance explained by the remaining three features, causing a twenty-fold inflation in the variance of its coefficient estimate.

</details>

---

### 10. Question

Why does VIF increase exponentially rather than linearly as the auxiliary $R_j^2$ approaches $1.0$?

- [ ] A) Because VIF is a hyperbolic function $\frac{1}{1 - R_j^2}$ with a vertical asymptote at $R_j^2 = 1.0$.
- [ ] B) Because Scikit-Learn applies an exponential activation function to VIF outputs.
- [ ] C) Because $R^2$ values are always exponentiated before calculating variance.
- [ ] D) Because of rounding errors in floating-point operations.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The function $f(R^2) = \frac{1}{1 - R^2}$ is a rational function with a pole at $R^2 = 1$. As $R^2$ moves from $0.5 \to 0.8 \to 0.9 \to 0.99$, VIF jumps from $2 \to 5 \to 10 \to 100$. This steep asymptotic curve means that small increases in correlation near unity produce explosive growth in estimation variance.
- **Option B (Incorrect):** Scikit-Learn does not compute VIF (it is computed in statsmodels or custom code), and no activation function is involved.
- **Option C (Incorrect):** $R^2$ is already the squared multiple correlation; no additional exponentiation is applied.
- **Option D (Incorrect):** The explosive growth is a pure mathematical property of dividing by a number approaching zero, not floating-point rounding error.

</details>
