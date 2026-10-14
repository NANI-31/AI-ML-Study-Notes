# Multicollinearity Objective Test — Set 08

---

### 1. Question

In the presence of two highly collinear features $x_1$ and $x_2$, how does **Ridge Regression (L2 regularization)** handle the distribution of their coefficients compared to OLS?

- [ ] A) Ridge drives one coefficient to zero and doubles the other.
- [ ] B) Ridge shrinks both coefficients towards each other, distributing weights approximately equally between the two collinear features.
- [ ] C) Ridge flips the sign of both coefficients to negative values.
- [ ] D) Ridge produces identical results to OLS regardless of the regularization parameter $\alpha$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Zeroing out one feature is a property of L1 regularization (Lasso), not L2 regularization.
- **Option B (Correct):** Because the L2 penalty minimizes the sum of squared coefficients ($\sum \beta_j^2$), splitting a weight across two identical features (e.g., $1^2 + 1^2 = 2$) incurs half the penalty of concentrating that weight into a single feature ($2^2 + 0^2 = 4$). Consequently, Ridge regression naturally groups collinear features and assigns them balanced, stable, equalized coefficients.
- **Option C (Incorrect):** Ridge shrinks magnitudes toward zero; it does not arbitrarily invert coefficient signs.
- **Option D (Incorrect):** When $\alpha > 0$, the Ridge estimator $(X^T X + \alpha I)^{-1} X^T y$ adds a positive constant $\alpha$ to the diagonal, which fundamentally stabilizes the ill-conditioned matrix and alters the coefficients compared to OLS.

</details>

---

### 2. Question

How does **Lasso Regression (L1 regularization)** typically behave when presented with a group of highly collinear features?

- [ ] A) It keeps all collinear features and gives them identical coefficients.
- [ ] B) It tends to arbitrarily select one feature from the correlated group, assign it a non-zero coefficient, and drive all other features in the group to exactly zero.
- [ ] C) It throws a `LinAlgError: Singular matrix` runtime exception.
- [ ] D) It calculates the average of all features and discards the target variable.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Distributing weights equally across correlated features is the behavior of Ridge (L2), not Lasso (L1).
- **Option B (Correct):** Due to the sharp corners of the L1 diamond constraint boundary, Lasso tends to select one representative feature from a cluster of collinear predictors and forces the rest to exactly $0.0$. Which specific feature is chosen can be essentially arbitrary or sensitive to minor sample fluctuations.
- **Option C (Incorrect):** Lasso optimization uses coordinate descent or subgradient methods that do not invert $X^T X$, avoiding singular matrix errors.
- **Option D (Incorrect):** Lasso is a supervised algorithm that directly optimizes squared error relative to the target variable $y$.

</details>

---

### 3. Question

Why does adding $\alpha I$ to the matrix $X^T X$ in Ridge regression mathematically eliminate the singularity caused by multicollinearity?

- [ ] A) Because adding a positive constant $\alpha > 0$ to the diagonal shifts all eigenvalues of $X^T X$ upward by $\alpha$, guaranteeing that all eigenvalues are strictly positive and the matrix is invertible.
- [ ] B) Because the identity matrix $I$ deletes all off-diagonal covariance entries.
- [ ] C) Because it converts the design matrix into a sparse format.
- [ ] D) Because it forces the determinant of $X^T X$ to equal exactly zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In multicollinear data, one or more eigenvalues of $X^T X$ are zero or near-zero, making the matrix singular or ill-conditioned. In Ridge regression, the modified matrix is $(X^T X + \alpha I)$. If the eigenvalues of $X^T X$ are $\lambda_i \ge 0$, the eigenvalues of $(X^T X + \alpha I)$ are $\lambda_i + \alpha$. For any $\alpha > 0$, every eigenvalue is strictly positive ($\lambda_i + \alpha > 0$), guaranteeing that the matrix has full rank, a non-zero determinant, and a stable, unique inverse.
- **Option B (Incorrect):** Off-diagonal entries remain unchanged; only the diagonal entries receive the $+\alpha$ addition.
- **Option C (Incorrect):** Adding $\alpha I$ does not sparsify the matrix; it densifies the diagonal.
- **Option D (Incorrect):** A determinant of zero defines a singular, non-invertible matrix; Ridge ensures the determinant is strictly positive.

</details>

---

### 4. Question

Why is **Elastic Net** often considered superior to Lasso when dealing with high-dimensional datasets containing clusters of collinear features?

- [ ] A) Because Elastic Net runs 1,000 times faster than ordinary least squares.
- [ ] B) Because Elastic Net combines L1 and L2 penalties, retaining Lasso's feature selection capability while using Ridge's grouping effect to select entire clusters of correlated features together.
- [ ] C) Because Elastic Net never shrinks coefficients.
- [ ] D) Because Elastic Net does not require hyperparameter tuning.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Elastic Net solves a more complex optimization problem and is generally slower to fit than simple OLS.
- **Option B (Correct):** Lasso suffers from a known limitation: when features are strongly correlated, it arbitrarily picks one and drops the rest. Elastic Net blends the L1 and L2 penalties:
  $$\mathcal{L}_{\text{ElasticNet}} = \text{Loss} + \alpha \rho \sum |\beta_j| + \frac{\alpha (1 - \rho)}{2} \sum \beta_j^2$$
  The strictly convex L2 term creates a "grouping effect" (correlated features enter or leave the model together), while the L1 term enforces sparsity (irrelevant groups are dropped).
- **Option C (Incorrect):** Elastic Net shrinks coefficients via both L1 and L2 mechanisms.
- **Option D (Incorrect):** Elastic Net requires tuning two hyperparameters: total penalty strength $\alpha$ and the mixing ratio $\rho$ (`l1_ratio`).

</details>

---

### 5. Question

Before applying Ridge, Lasso, or Elastic Net to a dataset with multicollinear features, why is feature standardization (e.g., `StandardScaler`) MANDATORY?

- [ ] A) Because scikit-learn throws an unhandled exception if features are unstandardized.
- [ ] B) Because regularization penalties treat all coefficients equally; features with larger unstandardized scales will have artificially smaller coefficients and thus be penalized unfairly less than features with small numerical scales.
- [ ] C) Because standardization eliminates all correlation between features.
- [ ] D) Because regularization only works on integers.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The code will execute without raising an error, but the resulting model will be statistically flawed.
- **Option B (Correct):** The penalty terms ($\sum |\beta_j|$ and $\sum \beta_j^2$) penalize the absolute size of coefficients uniformly. If feature $x_1$ is measured in dollars ($10,000$) and $x_2$ in millions ($0.01$), the model must assign a tiny coefficient to $x_1$ and a huge coefficient to $x_2$ to achieve the same effect. The penalty will severely penalize $x_2$'s coefficient while barely touching $x_1$, solely due to units of measurement. Standardization puts all features on equal footing ($z$-scores).
- **Option C (Incorrect):** Standardization preserves linear correlations between features.
- **Option D (Incorrect):** Regularization is formulated for real continuous variables.

</details>

---

### 6. Question

An ML engineer fits Ridge Regression on a dataset with high multicollinearity and tunes the hyperparameter $\alpha$ over the range $[0.001, 1000.0]$.

As $\alpha$ increases from $0.001$ to $1000.0$, what happens to the **bias** and **variance** of the coefficient estimates?

- [ ] A) Bias decreases, variance increases.
- [ ] B) Bias increases, variance decreases.
- [ ] C) Both bias and variance increase simultaneously.
- [ ] D) Both bias and variance decrease simultaneously to zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This describes reducing regularization, which moves the model toward unconstrained, high-variance OLS.
- **Option B (Correct):** This is the fundamental bias-variance trade-off of regularization. At $\alpha = 0$, OLS is unbiased ($E[\hat{\beta}] = \beta$), but collinearity makes its variance huge. As $\alpha$ increases, the penalty introduces deliberate shrinkage bias ($E[\hat{\beta}] \ne \beta$), but it dramatically compresses coefficient variance and stabilizes the estimates, often reducing overall Mean Squared Error.
- **Option C (Incorrect):** Regularization constrains parameters, which suppresses variance, not increases it.
- **Option D (Incorrect):** You cannot decrease both simultaneously; shrinkage always introduces bias away from the unconstrained OLS solution.

</details>

---

### 7. Question

A data analyst fits three models on a dataset containing four mutually collinear features:
* `x1` (sq ft)
* `x2` (sq meters)
* `x3` (lot size)
* `x4` (tax assessment value)

The target is house price. Which model will output coefficients where `x1` and `x2` have non-zero, nearly identical positive weights?

- [ ] A) Lasso with high $\alpha$
- [ ] B) Ridge regression
- [ ] C) DecisionTreeClassifier
- [ ] D) Ordinary Least Squares without regularization

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lasso will pick either `x1` or `x2` and set the other to exactly zero.
- **Option B (Correct):** Ridge regression shrinks correlated features toward each other, dividing predictive credit smoothly between `x1` and `x2` and assigning them proportional, stable positive weights.
- **Option C (Incorrect):** `DecisionTreeClassifier` is for categorical classification, not continuous house price regression.
- **Option D (Incorrect):** Because `x1` and `x2` are exact linear transformations of each other (perfect collinearity), unregularized OLS is singular and fails to produce unique coefficients.

</details>

---

### 8. Question

What parameter in `sklearn.linear_model.ElasticNet` controls the balance between the L1 (Lasso) and L2 (Ridge) penalties?

- [ ] A) `alpha`
- [ ] B) `l1_ratio`
- [ ] C) `max_iter`
- [ ] D) `tol`

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `alpha` controls the overall strength of the combined regularization penalty.
- **Option B (Correct):** `l1_ratio` specifies the mixing balance:
  * When `l1_ratio = 1.0`, ElasticNet is equivalent to pure Lasso (L1).
  * When `l1_ratio = 0.0`, ElasticNet is equivalent to pure Ridge (L2).
  * Values between $0$ and $1$ (e.g., `l1_ratio = 0.5`) provide a hybrid mixture of both penalties.
- **Option C (Incorrect):** `max_iter` defines the maximum number of optimization iterations.
- **Option D (Incorrect):** `tol` defines the convergence tolerance for the solver.

</details>

---

### 9. Question

An ML engineer fits a Lasso model on a dataset where features $x_1, x_2,$ and $x_3$ have mutual correlations above $0.95$. In cross-validation split 1, Lasso selects $x_1$ and sets $x_2, x_3$ to zero. In cross-validation split 2, Lasso selects $x_3$ and sets $x_1, x_2$ to zero.

Why does Lasso display this unstable selection behavior across folds?

- [ ] A) The scikit-learn installation is corrupted.
- [ ] B) Because the collinear features carry nearly identical predictive information, minor sample fluctuations in each fold tilt the optimization slightly in favor of a different feature.
- [ ] C) Lasso is an unsupervised clustering algorithm that re-clusters per fold.
- [ ] D) The features were standardized, which confuses the coordinate descent solver.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This is expected mathematical behavior of L1 penalties on collinear data, not a software bug.
- **Option B (Correct):** When multiple predictors share the same variance, the objective function has multiple near-optimal sparse solutions. A tiny difference in sample covariance in fold 1 makes $x_1$ marginally better; in fold 2, a different subset of rows makes $x_3$ marginally better. Lasso latches onto whichever is momentarily superior, setting the others to zero and creating feature selection instability across cross-validation folds.
- **Option C (Incorrect):** Lasso is a supervised regression algorithm.
- **Option D (Incorrect):** Standardization is required for Lasso and does not cause coordinate descent to fail.

</details>

---

### 10. Question

If the goal of an analysis is strictly to produce a **parsimonious model** (a model with the fewest possible features that still predicts well) from a wide dataset with extensive multicollinearity, which regularization method should be chosen?

- [ ] A) Ridge regression, because it retains all original features.
- [ ] B) Lasso regression, because its L1 penalty enforces sparsity by driving redundant coefficients to zero.
- [ ] C) Ordinary Least Squares without penalties.
- [ ] D) Polynomial regression with degree 5.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Ridge shrinks coefficients toward zero, but practically never sets them exactly to zero. All features remain in the model.
- **Option B (Correct):** Parsimony means achieving maximum explanatory or predictive power with the minimum number of parameters. Lasso naturally eliminates redundant, collinear features by forcing their coefficients to exactly zero, leaving a compact, sparse subset of active predictors.
- **Option C (Incorrect):** OLS retains all features and produces erratic coefficients under collinearity.
- **Option D (Incorrect):** Higher-degree polynomial regression dramatically expands the feature space and intensifies multicollinearity.

</details>
