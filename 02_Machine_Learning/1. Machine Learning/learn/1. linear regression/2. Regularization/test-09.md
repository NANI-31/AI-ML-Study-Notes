# Regularization Objective Test — Set 09

---

### 81. Question

A machine learning engineer trains a Lasso model with `alpha = 10,000` and observes that all coefficients in `model.coef_` are exactly 0. She then evaluates the model on the test set with `model.score(X_test, y_test)`.

What value does `model.score()` return, and why?

- [ ] A) R² = 1.0, because the model has fully converged and all regularization has been applied.
- [ ] B) R² = 0.0, because a model with all-zero coefficients predicts only the intercept (which approximates the training mean ȳ_train), giving performance equivalent to the baseline mean model.
- [ ] C) R² = −1.0, indicating the model performs exactly twice as poorly as the mean baseline.
- [ ] D) `model.score()` raises a `ValueError` because R² is undefined when all coefficients are zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** All-zero coefficients do not mean "fully converged to an optimal solution" — they mean the regularization penalty is so strong that all feature information has been suppressed. The model predicts nearly the training mean for every input, achieving the absolute minimum expressiveness.

- **Option B (Correct):** When all coefficients in `coef_` are zero, the model predicts only the intercept for every input: ŷᵢ = intercept_ (which approximates ȳ_train). This is equivalent to the baseline model that always predicts the training mean. By the definition of R², the baseline mean model achieves R² = 0 on the training set. On the test set, it achieves approximately R² ≈ 0, though the exact value depends on how closely ȳ_train matches ȳ_test. Conceptually, the answer is R² ≈ 0.

- **Option C (Incorrect):** R² = −1.0 would occur if SSE = 2 × SST on the test set. An all-zero coefficient model predicts the training mean, which is a reasonable baseline. For R² to be −1.0 specifically, the training mean would need to be systematically far from the test observations — which can happen but is not guaranteed or the expected result.

- **Option D (Incorrect):** `model.score()` does not raise an error when all coefficients are zero. It still computes R² = 1 − SSE/SST using the model's predictions (which all equal the intercept). The computation is valid and returns a well-defined number.

</details>

---

### 82. Question

A data scientist observes the following on a 6-feature dataset after fitting Ridge with multiple alpha values:

```
alpha=0:     coef_ = [12.4, -8.3, 15.1, -9.2, 11.7, -14.5]
alpha=1:     coef_ = [5.2,  -3.1, 6.4,  -4.0, 4.8,  -6.1]
alpha=100:   coef_ = [0.8,  -0.5, 0.9,  -0.6, 0.7,  -0.9]
alpha=10000: coef_ = [0.008, -0.005, 0.009, -0.006, 0.007, -0.009]
```

None of the coefficients at any alpha value are exactly zero. What property of Ridge does this demonstrate?

- [ ] A) Ridge with sufficient alpha eventually produces exact zeros, just not at these alpha levels.
- [ ] B) Ridge's L2 penalty is a smooth quadratic constraint, which creates a circular feasible region with no corner points on the coordinate axes; therefore, the OLS loss ellipses always touch this region at interior points where all coefficients are non-zero.
- [ ] C) The dataset has too many features for Ridge to perform feature selection; Lasso should be used instead.
- [ ] D) The large alpha values have caused numerical overflow, which prevents coefficients from reaching zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** For Ridge regression, coefficients approach zero asymptotically as alpha → ∞ but never reach exactly zero for any finite alpha. This is a mathematical property, not a matter of "not enough alpha." Even at alpha = 10,000,000, Ridge coefficients would be extremely small but still non-zero.

- **Option B (Correct):** The L2 constraint `Σβⱼ² ≤ t` defines a sphere (or hypersphere in high dimensions) as the feasible region. A sphere is smooth with no sharp corners — it is curved everywhere. When the OLS loss ellipses expand outward from the unconstrained OLS minimum and first touch this sphere, they make contact at a curved interior point, not at a coordinate axis corner. Interior points of the sphere have all non-zero coordinates by definition. This geometric property guarantees Ridge never produces exact zeros.

- **Option C (Incorrect):** The number of features (6) does not prevent Ridge from doing "feature selection" — Ridge is not designed to do feature selection at all. The reason Ridge doesn't produce zeros has nothing to do with the number of features; it is due to the geometry of the L2 constraint. Lasso should indeed be used for feature selection, but the explanation for Ridge's behavior is geometric, not feature-count related.

- **Option D (Incorrect):** The pattern shows coefficients approaching zero as alpha increases (12.4 → 0.008 for the first coefficient), which is correct behavior and not numerical overflow. Numerical overflow would produce NaN, inf, or extremely large values — the opposite of the shrinkage pattern shown.

</details>

---

### 83. Question

A model selection committee is reviewing three candidate models for a fraud detection system:

| Model                         | Train R² | Test R² | Non-zero features |
|-------------------------------|----------|---------|-------------------|
| OLS                           | 0.96     | 0.71    | 40                |
| Ridge (alpha=5)               | 0.91     | 0.84    | 40                |
| Lasso (alpha=0.3)             | 0.87     | 0.81    | 12                |

The fraud team requires: (1) maximum test predictive accuracy, and (2) interpretable model with minimum features.

Which model is the most appropriate under these joint requirements, and what trade-off does it involve?

- [ ] A) OLS, because it has the highest training R².
- [ ] B) Ridge (alpha=5), because it has the best test R² (0.84) but retains all 40 features.
- [ ] C) Lasso (alpha=0.3), because it balances high test R² (0.81, close to Ridge's 0.84) with dramatically fewer features (12 vs. 40), meeting the interpretability requirement.
- [ ] D) OLS, because regularization always reduces interpretability.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** OLS has the highest training R² (0.96) but the lowest test R² (0.71), indicating the highest overfitting. It also uses all 40 features. Neither criterion (test accuracy nor interpretability) favors OLS.

- **Option B (Incorrect):** Ridge has the highest test R² (0.84), satisfying criterion 1. However, it retains all 40 features, failing criterion 2 (minimum features for interpretability). The question asks for joint satisfaction of both requirements, so Ridge alone is not the answer.

- **Option C (Correct):** Lasso achieves test R² = 0.81, which is close to Ridge's 0.84 (a 3-percentage-point difference), while reducing the model to 12 features (vs. 40). This satisfies criterion 2 (interpretability with minimal features) and approximately satisfies criterion 1 (near-best test accuracy). The trade-off is a small reduction in test predictive accuracy (−0.03 vs. Ridge) in exchange for a 70% reduction in active features, which greatly improves interpretability and deployability. Given both requirements are equally important, Lasso is the most appropriate joint choice.

- **Option D (Incorrect):** Regularization (Ridge and Lasso) generally improves interpretability compared to OLS in collinear settings. Ridge produces stable, shrunken coefficients (more interpretable than the wild OLS swings), and Lasso produces sparse models (further improving interpretability by eliminating irrelevant features). OLS is not more interpretable when its coefficients are unreliable and unstable.

</details>

---

### 84. Question

An analyst uses cross-validation to select the optimal alpha for Ridge and prints the results:

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

alphas = [0.01, 0.1, 1, 10, 100, 1000]
results = {}

for alpha in alphas:
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    results[alpha] = scores.mean()
    print(f"alpha={alpha}: Mean CV R² = {scores.mean():.3f}")
```

Output:
```
alpha=0.01:  Mean CV R² = 0.763
alpha=0.1:   Mean CV R² = 0.781
alpha=1:     Mean CV R² = 0.804
alpha=10:    Mean CV R² = 0.812
alpha=100:   Mean CV R² = 0.799
alpha=1000:  Mean CV R² = 0.743
```

Based on this output, which alpha should be selected for the final model?

- [ ] A) alpha = 0.01, because lower regularization is always safer.
- [ ] B) alpha = 1000, because maximum regularization produces the most stable model.
- [ ] C) alpha = 10, because it achieves the highest mean cross-validation R² (0.812), indicating the best bias-variance balance on the training data.
- [ ] D) alpha = 0.1, because it is close to zero and therefore closest to OLS, which is the reference model.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Lower regularization is not always safer. At alpha = 0.01, mean CV R² = 0.763, which is lower than at alpha = 10 (0.812). Low regularization here corresponds to higher variance (overfitting on training folds), which reduces CV performance. The data shows that some regularization improves performance.

- **Option B (Incorrect):** At alpha = 1000, mean CV R² = 0.743, the lowest among all tested values. Maximum regularization is causing underfitting — the model has become too constrained to capture the signal. "Most stable" does not mean "best performing."

- **Option C (Correct):** Alpha = 10 achieves the highest mean CV R² (0.812) across 5 folds. The mean CV R² is the standard criterion for hyperparameter selection via cross-validation — it estimates generalization performance on the training data without using the test set. Alpha = 10 represents the optimal bias-variance balance among the tested values: enough regularization to reduce variance compared to alpha = 0.01, but not so much that it introduces excessive bias like alpha = 1000.

- **Option D (Incorrect):** Proximity to OLS is not a criterion for selecting alpha. The purpose of cross-validation is to find the empirically best-performing alpha, not to minimize distance from OLS. Alpha = 0.1 achieves only 0.781, well below alpha = 10's 0.812.

</details>

---

### 85. Question

A data scientist encounters a dataset where `feature_A` and `feature_B` have a correlation of 0.99. She fits OLS and gets coefficients: `β_A = 1000, β_B = −998`. She is alarmed by the large, opposite-sign coefficients.

Which statement best explains and addresses this situation?

- [ ] A) The large coefficients confirm that both features have enormous individual effects on the target and should be retained as-is.
- [ ] B) The near-cancellation pattern (β_A ≈ −β_B) is a classic OLS artifact of multicollinearity. OLS arbitrarily spread the combined effect (~2 net units) across the two features with enormous individual magnitudes that nearly cancel. Ridge would shrink both toward moderate, stable values.
- [ ] C) The large coefficients indicate the model is numerically unstable and the data should be discarded.
- [ ] D) Negative β_B proves that feature_B has a harmful effect on the target and should be removed from the model.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The large coefficients (1000 and −998) do not reflect large individual effects — they largely cancel each other out. The net effect of a unit change in both features simultaneously would be approximately 1000 − 998 = 2 units, which may be the true effect. OLS has artificially inflated the individual magnitudes while preserving the combined prediction. These coefficients are not interpretable as individual feature effects.

- **Option B (Correct):** With correlation 0.99, feature_A and feature_B carry nearly identical information. OLS cannot separate their individual contributions and produces coefficients that sum to approximately the correct combined effect but with individually huge, offsetting values. β_A + β_B ≈ 1000 + (−998) = 2, which may be the true joint effect. Ridge regularization would shrink both β_A and β_B to moderate, positive values close to 1 each (distributing the ~2 unit effect evenly between the two correlated features), producing interpretable, stable estimates.

- **Option C (Incorrect):** Large coefficients due to multicollinearity indicate a statistical estimation problem, not a data quality error. The data itself is not invalid — the issue is the OLS estimator's sensitivity to near-collinear features. The data should not be discarded; regularization should be applied.

- **Option D (Incorrect):** The negative coefficient for feature_B is an artifact of multicollinearity, not a true negative causal relationship. When two features are almost perfectly correlated, OLS exploits the redundancy to produce opposite-sign coefficients that internally cancel, improving training fit through a mathematical coincidence rather than causal structure. Removing feature_B based on this negative coefficient would be a mistake.

</details>

---

### 86. Question

A developer examines a Lasso model fitted with a pipeline and wants to compute predictions manually without calling `predict()`. The pipeline contains:

1. `StandardScaler` (fitted on training data)
2. `Lasso(alpha=2.0)` (fitted on scaled training data)

For a new observation `x_raw = [500, 3, 8]` (raw, unscaled), what is the correct manual prediction procedure?

- [ ] A) Use `x_raw` directly in ŷ = coef_ · x_raw + intercept_ because Lasso accounts for scaling internally.
- [ ] B) Scale `x_raw` using the training scaler (same mean and std as training), then compute ŷ = coef_ · x_scaled + intercept_.
- [ ] C) Average the raw features to get a single value, then multiply by the sum of all coefficients.
- [ ] D) Use `x_raw` in ŷ = coef_ · x_raw + intercept_, but then divide the result by the number of features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lasso does not account for scaling internally. The `coef_` and `intercept_` values are fitted in the scaled feature space (where features have zero mean and unit variance). Applying these coefficients directly to raw unscaled features produces incorrect predictions because the model expects standardized inputs.

- **Option B (Correct):** The correct manual prediction procedure mirrors what the pipeline does: (1) scale `x_raw` using the training scaler's parameters (subtract training mean, divide by training std — the same transformation applied during fitting); (2) compute ŷ = coef_ · x_scaled + intercept_. This places the new observation in the same feature space the model was trained in, producing a correct prediction.

- **Option C (Incorrect):** Averaging raw features and multiplying by the sum of coefficients is not a valid linear prediction formula. The linear model computes a weighted sum where each coefficient multiplies its corresponding feature, not an average times the coefficient sum.

- **Option D (Incorrect):** Dividing the prediction by the number of features has no basis in the linear model formula ŷ = coef_ · x + intercept_. The number of features does not appear in the prediction computation.

</details>

---

### 87. Question

A data scientist builds a pipeline with an intentional flaw:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scales all of X outside the pipeline

pipe = make_pipeline(Ridge(alpha=10.0))  # No scaler inside pipeline
scores = cross_val_score(pipe, X_scaled, cv=5, scoring='r2')
print(scores.mean())
```

Compare this to the correct implementation:

```python
pipe_correct = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
scores_correct = cross_val_score(pipe_correct, X, cv=5, scoring='r2')
print(scores_correct.mean())
```

Why is the first implementation flawed?

- [ ] A) There is no flaw; pre-scaling X before cross-validation is equivalent to scaling inside a pipeline.
- [ ] B) The first implementation scales X using the entire dataset (including all fold test data) before cross-validation begins, leaking test fold statistics into the scaler. The correct implementation scales only within each training fold during cross-validation.
- [ ] C) The flaw is that the first pipeline lacks Ridge regression; it only contains a scaler.
- [ ] D) The flaw is that `cross_val_score` cannot accept pre-scaled data; it requires raw data.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Pre-scaling X on the entire dataset before cross-validation is NOT equivalent to scaling inside a pipeline during cross-validation. This is the subtle and important difference that this question tests. When X is pre-scaled on all data, the scaler's mean and standard deviation incorporate statistics from all 5 fold test sets, creating information leakage.

- **Option B (Correct):** In the first implementation, `scaler.fit_transform(X)` uses the mean and standard deviation of the entire dataset X (all folds combined). When cross-validation then splits X_scaled into training and test folds, each test fold's features have already been scaled using statistics that include that test fold's own data — a form of data leakage. In `pipe_correct`, `cross_val_score` with a pipeline calls `StandardScaler.fit()` only on the current fold's training data in each iteration, then applies `transform()` to the validation fold using training-fold statistics. This is the leak-free approach that produces a genuinely unbiased generalization estimate.

- **Option C (Incorrect):** The first pipeline `make_pipeline(Ridge(alpha=10.0))` does contain Ridge regression (as the only step). The flaw is in how scaling is applied before the pipeline, not in the pipeline's contents.

- **Option D (Incorrect):** `cross_val_score` can accept both pre-scaled and raw data — it makes no restriction on input scaling. The issue is the statistical validity of pre-scaling, not a technical restriction of the function.

</details>

---

### 88. Question

A data scientist is debugging why her Lasso model produces very different coefficients when she changes only the order of feature columns in the design matrix (from `[X1, X2, X3]` to `[X3, X1, X2]`). She expects coefficient magnitudes to stay the same; only their positions in `coef_` should shift.

Is her expectation correct?

- [ ] A) No, changing feature order changes the Lasso optimization landscape and can produce fundamentally different coefficient magnitudes.
- [ ] B) Yes, reordering features only reorders the coefficients in `coef_`; the magnitudes themselves remain the same.
- [ ] C) Yes for Ridge, but no for Lasso, because Lasso's L1 penalty treats earlier-indexed features preferentially.
- [ ] D) No, changing feature order changes the L2 penalty applied by Lasso.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The Lasso optimization problem is symmetric with respect to feature permutation — swapping columns of X and correspondingly reordering the coefficient vector produces an equivalent optimization problem. The magnitudes of the optimal coefficients are determined by the features' predictive content, not by their column order in X.

- **Option B (Correct):** Reordering feature columns only reorders the elements of `coef_`. The optimization landscape for the overall Lasso objective is invariant to column permutations of X. If X1, X2, X3 had coefficients [2.1, 0.0, 3.4], then reordering to [X3, X1, X2] should produce coefficients [3.4, 2.1, 0.0] — the same magnitudes in a different order. This is because the L1 penalty Σ|βⱼ| is symmetric (sums absolute values in any order), and the SSE term is also invariant to column permutation when β is correspondingly reordered.

- **Option C (Incorrect):** Lasso does not treat earlier-indexed features preferentially. The L1 penalty is the sum of absolute values `|β₁| + |β₂| + ... + |βₚ|`, which is symmetric — it does not depend on the order or index of the features. Both Ridge and Lasso are invariant to feature column permutation.

- **Option D (Incorrect):** Lasso uses the L1 penalty (sum of absolute values), not L2. Changing feature order does not affect the penalty: the sum of absolute values is commutative and independent of order.

</details>

---

### 89. Question

An engineer presents a comparison of three metrics (MSE, RMSE, R²) for a salary prediction model:

| Metric | Training Set | Test Set |
|--------|-------------|---------|
| MSE    | 4,200       | 5,800   |
| RMSE   | 64.8        | 76.2    |
| R²     | 0.88        | 0.83    |

A junior analyst says: "The MSE went from 4,200 to 5,800 — that's a 38% increase, which is much worse than the 5-point R² drop. We should use MSE as the primary model evaluation metric."

What is the most accurate critique of this reasoning?

- [ ] A) The analyst is correct; MSE is always the preferred metric for regression evaluation.
- [ ] B) The percentage change in MSE (38%) cannot be meaningfully interpreted on its own without knowing the target variable's scale. R² (scale-independent) provides a more stable, comparable measure across settings. The 5-point R² gap reflects that the model generalizes well (0.83 test vs. 0.88 train).
- [ ] C) RMSE should be used instead of both MSE and R² because it is in the same units as the target.
- [ ] D) The analyst is correct; larger percentage changes in MSE always indicate more severe generalization problems than smaller changes in R².

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** MSE is a valid and commonly used regression metric, but it is scale-dependent (it carries squared units of the target). "Always preferred" is too absolute — the best metric depends on the evaluation context and whether scale-independence matters.

- **Option B (Correct):** MSE = 4,200 and MSE = 5,800 are in salary² units (e.g., dollars²). The 38% increase sounds alarming but depends entirely on the scale of salaries. If salaries range from $30,000 to $200,000, an RMSE of $64.8–76.2 is in actual salary units and may represent a 0.05% error on a $150,000 salary — negligible in context. R² = 0.83 on the test set indicates the model explains 83% of test set salary variance — a strong result. The R² drop from 0.88 to 0.83 is modest and indicates reasonable generalization. The analyst's reasoning conflates raw MSE changes with generalization severity without accounting for the target scale.

- **Option C (Incorrect):** RMSE is on the same scale as the target (salary in dollars), which is an advantage for interpretability. However, it is still scale-dependent and cannot be compared across different prediction tasks (e.g., salary model vs. temperature model). Neither RMSE nor MSE is universally superior to R² for cross-model comparison.

- **Option D (Incorrect):** Percentage changes in MSE are not more informative than R² changes without context about the target variable's scale and variance. A 38% MSE increase could represent a minor or severe generalization problem depending on the absolute scale of the errors. R² changes are more directly interpretable as changes in explained variance fraction.

</details>

---

### 90. Question

A product data scientist compares two approaches for handling multicollinearity in a regression model:

**Approach 1:** Drop one feature from each highly correlated pair before fitting OLS.
**Approach 2:** Keep all features and apply Ridge regression with an appropriately tuned alpha.

Which statement best captures the trade-offs between these approaches?

- [ ] A) Approach 1 is always superior because removing features eliminates multicollinearity entirely.
- [ ] B) Approach 2 is always superior because Ridge is mathematically guaranteed to outperform feature dropping.
- [ ] C) Approach 1 risks discarding genuinely informative features if the dropped features contain unique variance; Approach 2 retains all information by keeping features but stabilizes coefficients through shrinkage, generally providing better predictive performance when features are jointly important.
- [ ] D) Both approaches produce identical predictions because Ridge effectively zeroes out dropped-equivalent features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Dropping features does reduce collinearity, but it risks discarding features that contain unique predictive information not shared by the retained feature. The dropped feature may have subtle independent effects on the target that improve generalization when retained. Additionally, dropping requires a decision rule (which feature to drop) that is not always clear-cut.

- **Option B (Incorrect):** Ridge is not mathematically guaranteed to outperform feature dropping. In some cases (genuinely redundant features), dropping one of a correlated pair can produce a simpler, equally performant model. In others (features with shared AND unique variance), Ridge outperforms. The correct approach is empirical comparison via cross-validation.

- **Option C (Correct):** Approach 1 (dropping) is simpler but risks information loss. If the dropped feature contains any unique variance not captured by the retained feature, the model becomes less predictive. Approach 2 (Ridge) retains all features and uses the L2 penalty to stabilize coefficient estimates, preventing the OLS instability caused by high correlation. Ridge can leverage the combined information from all correlated features, often producing better predictive performance than dropping features arbitrarily. The trade-off is that Ridge requires alpha tuning, while feature dropping is a one-time decision.

- **Option D (Incorrect):** Ridge does not produce predictions identical to OLS on a reduced feature set. Ridge distributes weight across all retained features, producing different coefficient magnitudes and often different predictions than OLS on a subset of features.

</details>
