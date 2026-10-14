# Regularization Objective Test — Set 05

---

### 41. Question

A developer writes the following code to build a regularized pipeline:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipeline.fit(X_train, y_train)

train_r2 = pipeline.score(X_train, y_train)
test_r2 = pipeline.score(X_test, y_test)
print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
```

What metric does `pipeline.score()` return for a Ridge regression pipeline?

- [ ] A) Mean Squared Error (MSE) — the average squared prediction error.
- [ ] B) Root Mean Squared Error (RMSE) — the standard deviation of residuals.
- [ ] C) R² (coefficient of determination) — the proportion of variance in y explained by the model.
- [ ] D) Mean Absolute Error (MAE) — the average absolute prediction error.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** MSE is not what scikit-learn's `score()` method returns for regression estimators. MSE can be computed separately using `sklearn.metrics.mean_squared_error`, but `score()` on regressors returns R².

- **Option B (Incorrect):** RMSE is the square root of MSE and is also not the metric returned by `score()`. Like MSE, it can be computed explicitly but is not the default `score()` output for any scikit-learn regressor.

- **Option C (Correct):** For all scikit-learn regression estimators (including `Ridge`, `Lasso`, `LinearRegression`, and pipelines ending in a regressor), `score()` computes and returns R² = 1 − SSE/SST. This is the proportion of variance in the target variable y explained by the model, a scale-independent metric.

- **Option D (Incorrect):** MAE (Mean Absolute Error) is not the metric returned by `score()`. It can be computed using `sklearn.metrics.mean_absolute_error`. The `score()` method for regression models exclusively returns R².

</details>

---

### 42. Question

An ML engineer trains Lasso models with three different alpha values and checks the number of non-zero coefficients:

```python
from sklearn.linear_model import Lasso
import numpy as np

for alpha in [0.001, 1.0, 1000.0]:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    nonzero = (model.coef_ != 0).sum()
    print(f"alpha={alpha}: non-zero coefs = {nonzero}")
```

The dataset has 15 features. Assuming all features are genuinely predictive, which output pattern is most expected?

- [ ] A) alpha=0.001: 15; alpha=1.0: 15; alpha=1000.0: 15
- [ ] B) alpha=0.001: 15; alpha=1.0: ~8; alpha=1000.0: 0
- [ ] C) alpha=0.001: 0; alpha=1.0: ~8; alpha=1000.0: 15
- [ ] D) alpha=0.001: ~8; alpha=1.0: ~8; alpha=1000.0: ~8

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** At alpha = 1000.0, the Lasso penalty is extremely strong and will drive virtually all coefficients to zero, even for genuinely predictive features. Retaining all 15 features at very high alpha is inconsistent with Lasso's behavior.

- **Option B (Correct):** At very small alpha (0.001), the L1 penalty is negligible and Lasso behaves nearly like OLS — all 15 genuinely predictive features retain non-zero coefficients. As alpha increases to 1.0, the penalty is moderate and less important features start to zero out, leaving roughly half. At alpha = 1000.0, the penalty dominates and virtually all coefficients collapse to zero, leaving the model to predict near the intercept (mean).

- **Option C (Incorrect):** This reverses the expected pattern. Small alpha means weak regularization (near-OLS behavior), so many features are retained. Large alpha means strong regularization, zeroing out features. Option C has the pattern backwards.

- **Option D (Incorrect):** Lasso is highly sensitive to alpha. Roughly half the features staying non-zero across orders-of-magnitude alpha changes (0.001 to 1000) is implausible. The regularization path changes dramatically with alpha, as shown by Lasso's regularization path plots.

</details>

---

### 43. Question

A researcher computes the Pearson correlation between every pair of features in a 10-feature dataset. She finds that 15 pairs out of 45 have absolute correlations above 0.80. She concludes: "Since only 15 pairs exceed the threshold, multicollinearity is not a serious problem."

Which critique of this reasoning is most valid?

- [ ] A) The conclusion is correct because 15 out of 45 pairs is less than 50%, indicating the majority of feature pairs are uncorrelated.
- [ ] B) The conclusion is flawed because pairwise correlations do not capture multicollinearity involving three or more features that are jointly collinear even if no single pair exceeds the threshold.
- [ ] C) The conclusion is correct because only absolute correlations above 0.95 signal multicollinearity.
- [ ] D) The conclusion is correct because the threshold of 0.80 is too strict and should be 0.70.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The proportion of pairs above the threshold is not a valid criterion for ruling out multicollinearity. Even a single highly collinear pair can cause significant coefficient instability if those features share substantial predictive information. Fraction of correlated pairs does not determine the severity of the problem.

- **Option B (Correct):** Pairwise Pearson correlations are a useful but incomplete diagnostic tool for multicollinearity. Multicollinearity can also arise from higher-order relationships where, for example, X3 ≈ 0.5×X1 + 0.5×X2, even if no pairwise correlation between X1, X2, or X3 exceeds the threshold. These higher-dimensional collinearities require tools like the Variance Inflation Factor (VIF) to detect reliably.

- **Option C (Incorrect):** The "correct" threshold for flagging multicollinearity is domain-dependent, but the common working threshold is often cited as |r| > 0.7 or |r| > 0.8. Whether 0.80 or 0.95 is used, the fundamental critique remains: pairwise correlations miss multicollinearity involving three or more features.

- **Option D (Incorrect):** This option argues about the threshold value, not the fundamental methodological limitation of relying solely on pairwise correlations. Whether the threshold is 0.70, 0.80, or 0.95 is secondary to the point that pairwise analysis cannot capture higher-order collinearities.

</details>

---

### 44. Question

A data scientist is comparing the coefficient for feature `X` across OLS, Ridge (alpha=1), and Ridge (alpha=100) models fitted on the same data, where X is highly collinear with another feature Y.

Which ordering of coefficient magnitudes for X is most expected?

- [ ] A) |β_OLS| = |β_Ridge_1| = |β_Ridge_100|
- [ ] B) |β_OLS| ≥ |β_Ridge_1| ≥ |β_Ridge_100|
- [ ] C) |β_OLS| ≤ |β_Ridge_1| ≤ |β_Ridge_100|
- [ ] D) |β_Ridge_100| > |β_Ridge_1| > |β_OLS|

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** All three coefficient magnitudes being equal would mean Ridge regularization has no shrinkage effect, which contradicts its core property. Ridge's L2 penalty progressively shrinks coefficients as alpha increases.

- **Option B (Correct):** OLS with no penalty produces the unconstrained maximum likelihood estimate for β, which (in the presence of multicollinearity) can be inflated. As alpha increases in Ridge, the L2 penalty adds progressively stronger shrinkage pressure on the coefficient magnitudes. At alpha = 1, some shrinkage occurs. At alpha = 100, much stronger shrinkage occurs. So the expected ordering is |β_OLS| ≥ |β_Ridge_1| ≥ |β_Ridge_100|.

- **Option C (Incorrect):** This suggests that increasing alpha increases coefficient magnitude, which is the opposite of Ridge behavior. Higher alpha → stronger penalty → smaller coefficient magnitudes.

- **Option D (Incorrect):** Same error as Option C — this states that the Ridge model with alpha = 100 has the largest coefficient, which contradicts the shrinkage property of Ridge regularization.

</details>

---

### 45. Question

A regression model for predicting energy consumption is fitted with `Ridge(alpha=0)`. A colleague reviews the code and says: "You're just running OLS — using Ridge with alpha=0 is pointless."

Is the colleague's statement accurate?

- [ ] A) No, Ridge with alpha=0 applies a very small L2 penalty that improves numerical stability compared to plain OLS.
- [ ] B) Yes, Ridge with alpha=0 is mathematically equivalent to plain OLS because the penalty term vanishes completely.
- [ ] C) No, Ridge with alpha=0 uses a different optimizer than OLS, producing slightly different results.
- [ ] D) Yes, but only for datasets with fewer than 100 observations.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** When alpha = 0, the Ridge penalty term is α × Σβⱼ² = 0 × Σβⱼ² = 0, regardless of the values of βⱼ. There is literally no penalty contribution, so Ridge with alpha = 0 is not applying any regularization — not even a small one.

- **Option B (Correct):** The Ridge objective is SSE + α × Σβⱼ². When alpha = 0, this becomes SSE + 0 = SSE, which is the pure OLS objective. The optimization problem is identical, and the solution (given the same training data and features) is the same as OLS. The colleague's statement is mathematically accurate.

- **Option C (Incorrect):** While Ridge and OLS can use different underlying numerical methods (Ridge via the modified normal equation (XᵀX + αI)β = Xᵀy), when alpha = 0, the systems converge to the same mathematical solution: (XᵀX)β = Xᵀy — identical to the OLS normal equations.

- **Option D (Incorrect):** The equivalence between Ridge (alpha=0) and OLS is a mathematical identity that holds regardless of the number of observations. Sample size does not change the algebra of the objective function.

</details>

---

### 46. Question

An analyst runs the following code and observes that `test_r2` is −0.15:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
print(f"Train R²: {train_r2:.3f}")  # prints 0.940
print(f"Test R²: {test_r2:.3f}")   # prints -0.150
```

Which explanation best accounts for the large gap between train and test R²?

- [ ] A) The `score()` method uses a different formula for training data than for test data, producing inconsistent results.
- [ ] B) A negative test R² is mathematically impossible; there must be a bug in the code.
- [ ] C) The large gap (0.94 training vs. −0.15 test) indicates severe overfitting: the model has memorized the training data's noise rather than generalizing the true pattern.
- [ ] D) A test R² of −0.15 indicates the model is almost perfect; negative R² values mean smaller absolute error.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** scikit-learn's `score()` method computes R² using the same formula (1 − SSE/SST) regardless of whether training or test data is provided. The formula is not modified between training and test sets.

- **Option B (Incorrect):** A negative test R² is mathematically possible when SSE > SST on the held-out test set. This happens when the model's predictions are systematically worse than predicting the test set's mean. It is not a bug — it is meaningful signal indicating poor generalization.

- **Option C (Correct):** A training R² of 0.94 paired with a test R² of −0.15 is a severe overfitting pattern. The model has essentially memorized the training data (including noise), fitting very well there. When applied to new test observations, the learned noise patterns produce predictions that are systematically worse than a naive mean prediction — hence SSE > SST and R² < 0 on the test set.

- **Option D (Incorrect):** A negative test R² does not indicate near-perfect performance. Negative R² means the model performs worse than a simple constant predictor (the mean). A smaller (more negative) test R² indicates worse performance, not better. The interpretation is the opposite of what this option claims.

</details>

---

### 47. Question

A machine learning engineer is building a real estate pricing model with 12 features. She has reason to believe the true relationship uses all 12 features (no true sparsity). The features include several correlated pairs. She wants the best generalization while maintaining interpretable non-zero weights for every feature.

Which model best fits these requirements?

- [ ] A) Lasso with moderate alpha, because it performs feature selection and simplifies the model.
- [ ] B) Ridge with an appropriately tuned alpha, because it stabilizes coefficients without zeroing out features.
- [ ] C) Plain OLS, because there is no benefit to regularization when all features are genuinely relevant.
- [ ] D) Lasso with very small alpha, because near-zero alpha Lasso behaves identically to Ridge.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lasso with moderate alpha will zero out some features, directly violating the requirement to maintain interpretable non-zero weights for every feature. When the true model is dense (all features contribute), Lasso's feature selection is counterproductive.

- **Option B (Correct):** Ridge regression is ideally suited when all features are genuinely relevant and several are correlated. Ridge stabilizes the coefficient estimates by shrinking them proportionally through the L2 penalty, improving generalization without discarding any feature. With properly tuned alpha, Ridge delivers both stability and non-zero coefficients for all 12 features.

- **Option C (Incorrect):** Plain OLS is unregularized and, in the presence of correlated feature pairs, will produce high-variance, unstable coefficients. Even when all features are genuinely relevant, OLS can overfit by amplifying noise through collinearity. Ridge outperforms OLS in this scenario.

- **Option D (Incorrect):** Lasso and Ridge use different penalty shapes (L1 vs. L2), and they produce different solutions even at very small alpha. Near-zero alpha Lasso approaches OLS, not Ridge behavior. The two methods are not equivalent at any alpha. More importantly, even with very small alpha, Lasso retains the L1 geometry that can occasionally zero coefficients, while Ridge never does.

</details>

---

### 48. Question

A data scientist observes these two outcomes on the same housing dataset:

- **Model A (OLS)**: training R² = 0.98, test R² = 0.96
- **Model B (Ridge, alpha=2)**: training R² = 0.95, test R² = 0.97

Which analysis is most accurate?

- [ ] A) Model A is strictly better because it has higher training R².
- [ ] B) Model B is strictly better because Ridge always produces higher test R² than OLS.
- [ ] C) Model B has better generalization performance (higher test R²) with a small training R² sacrifice, suggesting the L2 penalty reduced overfitting in this dataset.
- [ ] D) Both models are equally good because the difference between test R² values (0.96 vs. 0.97) is negligible and within noise.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Higher training R² is not the objective in machine learning — generalization to unseen data is. Model A's higher training R² (0.98 vs. 0.95) may reflect a slight overfitting to training noise that manifests as lower test R² (0.96 vs. 0.97). Test performance is the relevant criterion.

- **Option B (Incorrect):** Ridge does not always produce higher test R² than OLS. Ridge outperforms OLS when coefficient variance (due to multicollinearity or overfitting) is reduced by regularization. On well-conditioned datasets with no multicollinearity, OLS can match or exceed Ridge. "Always" is too absolute.

- **Option C (Correct):** Model B (Ridge) accepts a 3-percentage-point reduction in training R² (0.95 vs. 0.98) in exchange for a 1-percentage-point improvement in test R² (0.97 vs. 0.96). This is the bias-variance tradeoff in action: Ridge introduces some training bias but reduces variance, resulting in better generalization. The Ridge model is preferable if test performance is the primary goal.

- **Option D (Incorrect):** While 0.01 difference in test R² may sometimes be within noise, the question asks for the most accurate analysis based on the values provided. Dismissing the comparison as "negligible" ignores the systematic pattern (Ridge training lower, Ridge test higher) that reveals a meaningful trade-off. Option C is the more analytically rigorous answer.

</details>

---

### 49. Question

A data scientist applies `StandardScaler` only to the test set before making predictions with a Ridge model that was fitted on unscaled training data. What problem does this introduce?

- [ ] A) No problem; the model will automatically adjust for the different scaling between training and test sets.
- [ ] B) The predictions will be on a different scale than intended, but R² will still be correct.
- [ ] C) The scaler fitted on test data uses different mean and variance statistics than those of the training data, causing the test features to be transformed differently from how the training features were represented, leading to incorrect predictions.
- [ ] D) The problem only occurs if the test set has more features than the training set.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** scikit-learn models do not automatically adjust for differences in feature scaling between training and test sets. The Ridge model's coefficients were fitted assuming the training set's scale. Applying a different scale transformation to test data breaks this assumption.

- **Option B (Incorrect):** If the test features are transformed using different statistics (test mean/variance vs. training mean/variance), the transformed test values correspond to different points in feature space than the model expects. Both the predictions and R² will be incorrect.

- **Option C (Correct):** `StandardScaler` learns the mean (μ) and standard deviation (σ) of each feature from the data it is fitted on. If fitted on training data, it uses training μ and σ. If independently fitted on test data, it uses test μ and σ. Since training and test sets typically have different empirical statistics, the test features are scaled differently from how the model was trained, producing incorrect feature representations and therefore incorrect predictions. The correct approach is to fit `StandardScaler` once on training data and apply the same transformation (using training statistics) to both training and test data.

- **Option D (Incorrect):** The scaling mismatch problem occurs regardless of whether training and test sets have the same number of features. The issue is about different scaling statistics, not different feature counts.

</details>

---

### 50. Question

A team member claims: "Because Lasso and Ridge both use the same OLS loss term (SSE) and only differ in the penalty type, they will always produce similar coefficients for any dataset."

Which statement is the most accurate rebuttal?

- [ ] A) The claim is correct; the difference in L1 vs L2 penalty only affects training speed, not the final coefficients.
- [ ] B) The claim is incorrect because Lasso and Ridge produce fundamentally different solutions, particularly in their sparsity: Lasso can produce exact zeros while Ridge cannot, and their solutions can differ dramatically especially with correlated features.
- [ ] C) The claim is partially correct; they only differ when alpha > 1.
- [ ] D) The claim is correct when features are uncorrelated, in which case both methods produce identical non-zero coefficients for all features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The choice of penalty (L1 vs L2) fundamentally changes the geometry of the constrained optimization problem and therefore the solution. L1 has a diamond-shaped feasible region that touches coordinate axes (producing zeros), while L2 has a circular region that never touches axes. The final coefficients, not just training speed, differ significantly.

- **Option B (Correct):** While both methods minimize the same OLS loss plus a penalty, the geometry of the L1 (diamond) vs L2 (sphere) constraint regions leads to qualitatively different solutions. Lasso can produce exact zeros (sparse solutions), while Ridge produces shrunken but non-zero coefficients. With correlated features, Lasso arbitrarily picks one and zeros the others, while Ridge distributes weight among them. These differences can produce dramatically different coefficient vectors on the same data.

- **Option C (Incorrect):** The difference between Lasso and Ridge solutions exists for any non-zero alpha, not only when alpha > 1. Even at alpha = 0.001, the two methods produce mathematically distinct solutions with different sparsity properties.

- **Option D (Incorrect):** Even when features are uncorrelated, Lasso and Ridge produce different coefficient magnitudes due to their different penalty structures (absolute vs squared). Lasso applies a soft-thresholding operation that can zero coefficients, while Ridge applies a shrinkage factor. Their coefficients are not identical for uncorrelated features.

</details>
