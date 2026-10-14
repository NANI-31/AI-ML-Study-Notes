# Regularization Objective Test — Set 06

---

### 51. Question

An engineer wants to verify whether her Ridge model is predicting correctly. She inspects the model's attributes after fitting:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
pipe.fit(X_train, y_train)

ridge_step = pipe.named_steps['ridge']
print(ridge_step.coef_)
print(ridge_step.intercept_)
```

What do `coef_` and `intercept_` contain after fitting?

- [ ] A) `coef_` contains the validation R² for each feature; `intercept_` contains the overall model R².
- [ ] B) `coef_` contains the learned coefficient (weight) for each feature; `intercept_` contains the fitted bias term (β₀).
- [ ] C) `coef_` contains the L2 penalty applied to each feature; `intercept_` contains the regularization alpha.
- [ ] D) `coef_` contains the raw Pearson correlations between each feature and the target; `intercept_` contains the mean of y_train.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `coef_` stores the learned regression coefficients, not validation R² per feature. R² is a model-level metric computed by `score()`, not a per-feature metric stored in `coef_`. The `intercept_` is not R².

- **Option B (Correct):** After fitting any scikit-learn linear model (including Ridge), `coef_` is a numpy array containing the learned weights (one per feature), and `intercept_` is a scalar (or array in multi-output cases) containing the fitted bias term. Together they define the linear model: ŷ = coef_ · X + intercept_.

- **Option C (Incorrect):** `coef_` does not contain penalty values. The L2 penalty is specified via the `alpha` constructor parameter and is incorporated into the optimization objective, not stored per-feature post-fit. `intercept_` stores the bias term, not alpha.

- **Option D (Incorrect):** `coef_` stores regression coefficients estimated by the regularized optimization process, not raw Pearson correlations. While there is a mathematical relationship between OLS coefficients and correlations in simple cases, `coef_` is not simply the correlation vector. `intercept_` is the fitted bias, which may or may not equal the mean of y_train depending on the model and data.

</details>

---

### 52. Question

An ML engineer fits a Lasso model and a Ridge model on a dataset with 10 features, then checks how many coefficients are exactly zero:

```python
lasso = Lasso(alpha=2.0)
ridge = Ridge(alpha=2.0)

lasso.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

print("Lasso zeros:", (lasso.coef_ == 0).sum())
print("Ridge zeros:", (ridge.coef_ == 0).sum())
```

Which output is most consistent with expected behavior?

- [ ] A) Lasso zeros: 0; Ridge zeros: 10
- [ ] B) Lasso zeros: 5; Ridge zeros: 0
- [ ] C) Lasso zeros: 0; Ridge zeros: 0
- [ ] D) Lasso zeros: 5; Ridge zeros: 5

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This reverses the behavior of Lasso and Ridge. Lasso (L1) is the method that drives coefficients to exact zero, while Ridge (L2) rarely (essentially never) produces exact zeros. Having Lasso produce 0 zeros and Ridge produce 10 zeros contradicts the fundamental properties of both methods.

- **Option B (Correct):** Lasso (L1) with alpha = 2.0 and 10 features will typically zero out some coefficients, especially less predictive ones. Ridge (L2) almost never produces exact zeros — its coefficients approach zero asymptotically but remain non-zero for any finite alpha. So Lasso zeros ≈ 5 (moderate feature selection) and Ridge zeros = 0 (no exact zeros) is consistent with expected behavior.

- **Option C (Incorrect):** Ridge producing 0 zeros is correct, but Lasso producing 0 zeros with alpha = 2.0 (a non-negligible regularization strength) is unlikely for a 10-feature dataset. At alpha = 2.0, Lasso is expected to zero out at least some features.

- **Option D (Incorrect):** Ridge producing 5 zeros contradicts Ridge's behavior. Ridge's L2 penalty is applied as a smooth quadratic constraint that shrinks coefficients but never drives them to exact zero for any finite alpha and non-zero features.

</details>

---

### 53. Question

A model is trained on a dataset where the target `y` ranges from 50,000 to 500,000 (house prices in dollars). The model achieves MSE = 4,500,000,000 and R² = 0.82.

A second model predicting temperature (range: 0–50°C) achieves MSE = 8 and R² = 0.79.

Which statement correctly compares these two models?

- [ ] A) The housing model is better because its R² is higher (0.82 > 0.79).
- [ ] B) The temperature model is better because its MSE is much lower (8 vs. 4.5 billion).
- [ ] C) R² values can be compared directly across the two models (0.82 vs. 0.79), but the MSE values cannot be compared because they carry different units (dollars² vs. °C²).
- [ ] D) Both models can only be compared using RMSE because R² is always biased upward on training data.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** While R² = 0.82 > 0.79 shows the housing model explains a slightly higher proportion of its target's variance, this comparison (if done) is only valid for R². Concluding one model is "better" from R² alone ignores the context of each task. However, R² is at least a comparable metric between the two models.

- **Option B (Incorrect):** MSE = 8 (in °C²) vs. MSE = 4,500,000,000 (in dollars²) cannot be directly compared because they measure error in different units with wildly different scales. A "lower" MSE for the temperature model does not mean it is a "better" model — the scales are incomparable.

- **Option C (Correct):** R² is a dimensionless, scale-independent ratio (proportion of variance explained), so comparing R² values across different prediction tasks is legitimate. MSE, however, carries the squared units of the target variable: dollars² for housing and °C² for temperature. These units are incommensurable — you cannot meaningfully compare MSE values across models with different target scales or units.

- **Option D (Incorrect):** RMSE does share the same issue as MSE for cross-task comparison — it carries the units of the target (dollars vs. °C), making it incomparable across the two problems as well. Also, R² is not always biased upward on training data in the way described; it is a property of OLS that training R² ≥ 0, but this is not described as a bias toward 1.

</details>

---

### 54. Question

A developer writes the following faulty code to evaluate a Ridge model:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)  # scales all data at once

X_train = X_all_scaled[:800]
X_test = X_all_scaled[800:]
y_train, y_test = y[:800], y[800:]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

What is wrong with this approach?

- [ ] A) `StandardScaler` cannot be applied before splitting; it must always be used after splitting.
- [ ] B) The entire dataset (including test data) was used to compute the scaler's mean and standard deviation, leaking test set distribution information into the preprocessing step.
- [ ] C) Ridge cannot be combined with StandardScaler; only Lasso supports preprocessing pipelines.
- [ ] D) The code will raise a `ValueError` because `fit_transform` and `score` cannot be used in the same script.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The issue is not that `StandardScaler` cannot be applied before splitting — the issue is that `fit_transform` on the entire dataset (before splitting) incorporates test set statistics into the scaler. The correct approach is to fit the scaler on training data only and then transform both training and test data using training statistics.

- **Option B (Correct):** By calling `scaler.fit_transform(X)` on the full dataset before splitting, the scaler learns the global mean and standard deviation of X, including the 200 test samples. When the model is later evaluated on the test set, those test samples have been scaled using statistics they "contributed to." This is a subtle form of data leakage — the model evaluation is slightly optimistic because test data information influenced the preprocessing.

- **Option C (Incorrect):** Ridge and StandardScaler are fully compatible. Both can be combined in a `make_pipeline` or used sequentially. The restriction described does not exist in scikit-learn.

- **Option D (Incorrect):** There is no incompatibility between `fit_transform` and `score` in the same script. These are methods of different objects (`StandardScaler` and `Ridge` respectively) that can coexist without raising errors.

</details>

---

### 55. Question

A data analyst fits three models (OLS, Ridge, Lasso) on a dataset with two perfectly collinear features: `X2 = 3 × X1`. The true target is `y = 5×X1 + ε`. She inspects the coefficients.

Which statement about the OLS solution is most accurate?

- [ ] A) OLS will find a unique solution of β₁ = 5 and β₂ = 0, correctly identifying X1 as the true predictor.
- [ ] B) OLS will find β₁ = 5 and β₂ = 5/3, distributing weight perfectly based on scaling.
- [ ] C) OLS will fail to find a unique solution because the design matrix is rank-deficient when X2 = 3×X1, producing infinitely many solutions with the constraint 3β₂ + β₁ = 5.
- [ ] D) OLS will set both coefficients to 0 because perfect collinearity makes the problem unsolvable.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** With X2 = 3×X1 exactly, the design matrix is rank-deficient (the columns are linearly dependent). OLS cannot uniquely identify which feature is "the true predictor." There are infinitely many coefficient pairs (β₁, β₂) that produce identical predictions, all satisfying β₁ + 3β₂ = 5. OLS has no basis for preferring β₁ = 5, β₂ = 0 over any other valid solution.

- **Option B (Incorrect):** The condition β₁ + 3β₂ = 5 has infinitely many solutions. The pair (5, 5/3) would give 5 + 3×(5/3) = 5 + 5 = 10 ≠ 5. This is not even a valid solution satisfying the constraint. (Valid example: β₁ = 2, β₂ = 1 gives 2 + 3×1 = 5 ✓.)

- **Option C (Correct):** Perfect collinearity (X2 = 3×X1) makes the XᵀX matrix singular — it cannot be inverted. The OLS normal equations have infinitely many solutions, all lying on the hyperplane β₁ + 3β₂ = 5. In practice, software may fail with an error, produce an arbitrary solution (often not the one with β₁ = 5), or return NaN/inf for the coefficients. Ridge resolves this by adding αI to XᵀX, restoring invertibility.

- **Option D (Incorrect):** OLS does not automatically set coefficients to 0 in response to perfect collinearity. The problem is not unsolvability in the sense of producing zeros — it is over-determinacy (too many valid solutions). The solver may produce a particular solution depending on numerical implementation, but it will not necessarily produce zeros.

</details>

---

### 56. Question

A stakeholder reviews an R² of 0.91 for an observational study's regression model predicting customer lifetime value from marketing spend, demographics, and product usage. She says: "An R² of 0.91 proves that increasing marketing spend by $1,000 will increase lifetime value by the amount shown in the marketing spend coefficient."

What is the most precise critique of this claim?

- [ ] A) The claim is correct; high R² validates the causal interpretation of regression coefficients.
- [ ] B) The claim is incorrect because R² of 0.91 means the model explains only 91% of variance, not 100%, so the coefficient is inaccurate by 9%.
- [ ] C) The claim is incorrect because regression coefficients in observational studies represent estimated associations under model assumptions, not causal effects; increasing marketing spend to confirm the coefficient's causal meaning requires experimental or causal inference methods.
- [ ] D) The claim is incorrect only if the marketing spend variable was not scaled before fitting.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** R² measures predictive fit (how much variance is explained), not causality. A high R² can be achieved by models with confounders, spurious correlations, and reverse causation. Predictive accuracy does not validate causal interpretation of individual coefficients.

- **Option B (Incorrect):** R² = 0.91 does not mean the coefficient is "inaccurate by 9%." R² is a global measure of variance explained. The remaining 9% unexplained variance reflects random error and model limitations, but it does not indicate the coefficient is off by a fixed percentage. The causal concern is entirely separate from R²'s magnitude.

- **Option C (Correct):** In observational studies, regression coefficients estimate the association between a predictor and the outcome, conditional on other included variables. Due to confounding variables, selection bias, and reverse causation that are common in observational data, these associations cannot be interpreted as causal effects without additional assumptions and methods. Randomized experiments (A/B tests) or causal inference techniques (difference-in-differences, instrumental variables) are required to establish that a $1,000 increase in marketing spend causes the predicted change in lifetime value.

- **Option D (Incorrect):** The causal interpretation problem is fundamental to observational regression and exists regardless of whether features were scaled. Scaling affects coefficient magnitude and the fairness of regularization penalties, but it does not resolve or create the causation vs. correlation issue.

</details>

---

### 57. Question

A Ridge model is trained with `alpha = 0.01` and then again with `alpha = 10,000`. Which behavior is expected for the model's `intercept_` value across these two settings?

- [ ] A) The `intercept_` will be exactly 0 at both alpha values because Ridge does not fit an intercept term.
- [ ] B) The `intercept_` will dramatically shrink toward zero as alpha increases from 0.01 to 10,000.
- [ ] C) The `intercept_` remains relatively stable as alpha increases because by default, scikit-learn's Ridge does not regularize the intercept term.
- [ ] D) The `intercept_` grows proportionally with alpha to compensate for the shrunken feature coefficients.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** scikit-learn's Ridge (with default `fit_intercept=True`) does fit and return a non-zero intercept term. The intercept represents the predicted value when all features are zero (or at their mean, if features are standardized).

- **Option B (Incorrect):** By default in scikit-learn, the Ridge penalty applies to the feature coefficients (β₁, β₂, ..., βₚ) but does NOT apply to the intercept (β₀). Therefore, as alpha increases, the feature coefficients shrink toward zero, but the intercept is estimated separately and is not directly penalized.

- **Option C (Correct):** scikit-learn's `Ridge(fit_intercept=True)` (the default) fits the intercept separately from the penalized feature coefficients. The L2 penalty `α × Σβⱼ²` sums over j = 1 to p (the feature indices), not j = 0. This means the intercept is not shrunk by increasing alpha. When features are standardized (zero mean), the intercept approximates the mean of y_train and remains relatively stable as alpha changes.

- **Option D (Incorrect):** The intercept does not grow to compensate for shrunken feature coefficients. When features are centered (standardized), the intercept is approximately ȳ regardless of alpha. It does not "compensate" for coefficient shrinkage by growing.

</details>

---

### 58. Question

An analyst computes an absolute Pearson correlation matrix for a 4-feature dataset and finds:

| Pair     | |r|  |
|----------|------|
| X1, X2   | 0.91 |
| X1, X3   | 0.15 |
| X1, X4   | 0.08 |
| X2, X3   | 0.12 |
| X2, X4   | 0.09 |
| X3, X4   | 0.87 |

Based on a working threshold of |r| > 0.7, which pairs signal potential multicollinearity?

- [ ] A) Only X1 and X2.
- [ ] B) X1 and X2, and X3 and X4.
- [ ] C) All pairs because even small correlations accumulate.
- [ ] D) No pairs because none of the correlations equals exactly 1.0.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While X1–X2 (|r| = 0.91) correctly flags a potential collinearity issue, X3–X4 (|r| = 0.87) also exceeds the 0.7 threshold and should be flagged. Identifying only one problematic pair is incomplete.

- **Option B (Correct):** Applying the threshold |r| > 0.7 to each pair: X1–X2 (0.91 > 0.7 ✓) and X3–X4 (0.87 > 0.7 ✓) both exceed the threshold. The remaining pairs (X1–X3: 0.15, X1–X4: 0.08, X2–X3: 0.12, X2–X4: 0.09) are all well below 0.7. Therefore, two pairs signal potential multicollinearity.

- **Option C (Incorrect):** Small correlations do not "accumulate" in a straightforward additive way to create multicollinearity. Multicollinearity is about strong linear relationships between specific predictors. Low pairwise correlations (0.08–0.15) among most pairs do not cumulatively create a collinearity problem.

- **Option D (Incorrect):** A correlation must be exactly 1.0 (perfect linear dependence) for the design matrix to be truly singular. Working thresholds like |r| > 0.7 are heuristics for identifying problematic-but-not-perfect collinearity that can still destabilize OLS estimates. Requiring |r| = 1.0 as the criterion would miss the practically important cases.

</details>

---

### 59. Question

A machine learning engineer observes that when she increases alpha in Ridge from 1 to 100, the training R² drops from 0.89 to 0.74, but the test R² improves from 0.81 to 0.84. When she further increases alpha to 10,000, both training R² (0.42) and test R² (0.51) drop.

What is the engineer's best strategy based on these observations?

- [ ] A) Use alpha = 10,000 because it has the lowest training R², indicating the least overfitting.
- [ ] B) Use alpha = 1 because OLS (minimum regularization) is always the safest choice.
- [ ] C) Use alpha = 100 because it achieves the highest test R² (0.84) and represents the sweet spot between bias and variance for this dataset.
- [ ] D) Collect more training data before choosing any alpha, because regularization is only needed for small datasets.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Lowest training R² does not mean least overfitting — it means the model is fitting the training data poorly (underfitting). The goal is to maximize held-out test performance, not to minimize training performance. Alpha = 10,000 causes both training and test R² to drop significantly, indicating severe underfitting where the model can no longer capture true signal.

- **Option B (Incorrect):** Alpha = 1 provides the weakest regularization tested and produces the lowest test R² (0.81), suggesting that some coefficient variance is still inflating test error at this setting. OLS (alpha = 0 or very small) is not always the safest choice when multicollinearity or overfitting is present.

- **Option C (Correct):** The principle of hyperparameter tuning is to select the value that maximizes held-out (test) performance. At alpha = 100, the test R² reaches its peak (0.84) — the regularization has successfully reduced coefficient variance enough to improve generalization without overly constraining the model. This is the optimal alpha based on the observations provided.

- **Option D (Incorrect):** Regularization is beneficial for datasets of all sizes, not only small ones. While more data can help reduce overfitting, it does not eliminate the need for appropriate regularization when multicollinearity or high-variance coefficients are present. The choice of alpha should be based on model performance, not dataset size alone.

</details>

---

### 60. Question

A researcher wants to evaluate her Ridge model rigorously using k-fold cross-validation. She writes:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

pipe = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
scores = cross_val_score(pipe, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}")
print(f"Mean CV R²: {np.mean(scores):.3f}")
```

What does the `scores` array contain, and why is the pipeline used instead of scaling X separately?

- [ ] A) `scores` contains 5 MSE values; the pipeline is used to speed up computation.
- [ ] B) `scores` contains 5 R² values (one per fold); the pipeline ensures the StandardScaler is fitted only on the training fold in each iteration, preventing test fold data from leaking into the scaler.
- [ ] C) `scores` contains 5 accuracy percentages; the pipeline is used to combine the scaler and ridge into one object.
- [ ] D) `scores` contains a single R² value averaged across all folds; the pipeline prevents multicollinearity.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The `scoring='r2'` argument instructs `cross_val_score` to use R² as the metric. MSE would require `scoring='neg_mean_squared_error'`. Also, speed is not the primary motivation for using a pipeline.

- **Option B (Correct):** With `cv=5`, `cross_val_score` splits the data into 5 folds and returns 5 individual R² scores (one per fold). The pipeline is critical: in each fold, `cross_val_score` calls `pipe.fit()` on the training portion of that fold, which causes `StandardScaler` to be fitted only on training fold data. The held-out test fold is then transformed using those training-fold statistics. This prevents data leakage — if X were scaled on the full dataset before cross-validation, test fold statistics would contaminate the scaler.

- **Option C (Incorrect):** R² values are not accuracy percentages (which apply to classification). With `scoring='r2'`, the returned scores are R² values, which can range from negative to 1.0. Also, while combining scaler and ridge into one object is a benefit of `make_pipeline`, it is not the primary statistical reason for using a pipeline in cross-validation.

- **Option D (Incorrect):** `cross_val_score` returns an array of scores (one per fold), not a single pre-averaged value. `np.mean(scores)` then averages them. The pipeline does not prevent multicollinearity — it ensures proper per-fold scaling to prevent data leakage.

</details>
