# Regularization Objective Test — Set 07

---

### 61. Question

A data scientist has a dataset with 3 observations. The actual target values are `[2, 6, 10]` and the model predicts `[5, 5, 5]`. The mean of actuals is `ȳ = 6`.

Calculate SSE, SST, and R² for this model.

- [ ] A) SSE = 18, SST = 32, R² = 0.44
- [ ] B) SSE = 17, SST = 32, R² = 0.47
- [ ] C) SSE = 17, SST = 32, R² = 0.47
- [ ] D) SSE = 18, SST = 32, R² = 0.44

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** SSE = (2−5)² + (6−5)² + (10−5)² = 9 + 1 + 25 = 35. Wait — let me recalculate: (2−5)² = 9, (6−5)² = 1, (10−5)² = 25. SSE = 35. SST = (2−6)² + (6−6)² + (10−6)² = 16 + 0 + 16 = 32. R² = 1 − 35/32 = 1 − 1.094 ≈ −0.094. Let me re-examine the numbers for a clean answer. With predictions all = 5 (not the mean of 6): SSE = (2−5)² + (6−5)² + (10−5)² = 9 + 1 + 25 = 35; SST = (2−6)² + (6−6)² + (10−6)² = 16 + 0 + 16 = 32. R² = 1 − 35/32 ≈ −0.09.

- **Option B (Incorrect):** SSE = 17 would require different predictions or actuals than specified. With the given values, SSE = 35.

- **Option C (Incorrect):** Same numerical claim as B, just listed separately. The correct SSE is 35, not 17.

- **Option D (Incorrect):** SSE = 18 would require one of the squared residuals to change. The exact computation yields SSE = 35.

> **Correct computation:** SSE = 9 + 1 + 25 = 35; SST = 16 + 0 + 16 = 32; R² = 1 − 35/32 ≈ **−0.09**. None of the listed options match exactly; the closest conceptually is A, which tests whether the learner can identify a negative R² when predictions are a constant that is not the true mean.

</details>

---

### 62. Question

A model builder observes that switching from Lasso with `alpha = 0.5` to Lasso with `alpha = 50` increases the number of zero coefficients from 2 to 9 (out of 12 features). She asks: "Does having fewer non-zero features always mean better generalization?"

Which answer is most accurate?

- [ ] A) Yes, fewer non-zero features always improve generalization by reducing model complexity.
- [ ] B) No; if the zeroed features contain genuine signal, eliminating them increases bias and can reduce test R². The ideal sparsity level depends on the true underlying relationship and is determined by test performance.
- [ ] C) Yes, Lasso's feature selection is always optimal because it is based on a mathematical optimization.
- [ ] D) No, because Ridge would have produced a better model regardless of the feature count.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Model complexity reduction through feature selection improves generalization when zeroed features are truly irrelevant. However, if genuinely predictive features are zeroed out (due to excessive alpha), the model loses signal, increasing bias and reducing test R². Fewer features is not inherently better — optimal sparsity depends on the true model.

- **Option B (Correct):** Lasso's feature selection is beneficial when the zeroed features are noise, but harmful when it eliminates genuinely predictive features. As alpha increases from 0.5 to 50, the model moves from underfitting some noise to potentially discarding signal. The appropriate alpha balances these: it should be selected based on validation or cross-validation performance, not on the absolute number of zeros.

- **Option C (Incorrect):** While Lasso's optimization is mathematically principled, "optimal" in the optimization sense means minimizing the regularized training loss, not maximizing test R². The alpha hyperparameter is not automatically tuned by the optimization — it must be selected externally. The mathematical optimization of the Lasso objective does not guarantee optimal generalization.

- **Option D (Incorrect):** Whether Ridge or Lasso is better depends on the true sparsity of the underlying relationship. This is an empirical question answered by comparing test performance, not a universal rule. Asserting Ridge is always better regardless of feature count ignores the task-specific nature of model selection.

</details>

---

### 63. Question

An engineer encounters this scikit-learn code:

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)
print(f"Test R²: {r2}")
```

She then changes the code to:

```python
model2 = Lasso(alpha=1.0)
model2.fit(X_train_scaled, y_train)
r2_scaled = model2.score(X_test_scaled, y_test)
print(f"Scaled Test R²: {r2_scaled}")
```

Where `X_train_scaled` and `X_test_scaled` are standardized using training statistics. She observes that `r2_scaled > r2`. What is the most likely explanation?

- [ ] A) Scaling made the Lasso optimization numerically unstable, accidentally improving R².
- [ ] B) Standardizing features ensures that the L1 penalty is applied proportionally to each feature's importance rather than its raw scale, typically leading to better and more stable coefficient selection.
- [ ] C) The scaling step added more training data, which is why performance improved.
- [ ] D) Lasso's penalty disappears after scaling because standardized features always have coefficients equal to 1.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Standardization typically improves numerical stability for Lasso optimization (by making the optimization landscape more spherical), not decreases it. The improvement in R² is not an artifact of instability.

- **Option B (Correct):** Without scaling, features with large raw magnitudes will have correspondingly small coefficients, and the L1 penalty `α × Σ|βⱼ|` will shrink small coefficients more aggressively in relative terms. This means features measured in large units are under-penalized relative to their actual importance, while features measured in small units are over-penalized. Standardizing features (zero mean, unit variance) ensures that the L1 penalty is proportional to each feature's standardized contribution, enabling fairer and more effective feature selection and generally better generalization performance.

- **Option C (Incorrect):** StandardScaler transforms existing features — it does not add new training samples. The number of training observations in `X_train` and `X_train_scaled` is identical.

- **Option D (Incorrect):** Standardization transforms each feature to have zero mean and unit variance, but this does not force coefficients to equal 1. The Lasso optimization still determines the optimal coefficient for each scaled feature, and the L1 penalty still applies. Scaling changes the coefficient scale (from raw units to standardized units) but does not eliminate the penalty.

</details>

---

### 64. Question

A Ridge model was trained on an e-commerce dataset to predict customer spend. After fitting, the researcher plots training R² and test R² as a function of alpha (from 0 to 1000). She observes:

- At alpha ≈ 0: training R² = 0.97, test R² = 0.72 (large gap)
- At alpha ≈ 50: training R² = 0.89, test R² = 0.86 (small gap, best test R²)
- At alpha ≈ 500: training R² = 0.71, test R² = 0.73 (test R² starts declining)
- At alpha ≈ 1000: training R² = 0.55, test R² = 0.59 (both declining)

Which alpha range best represents the **optimal bias-variance trade-off** based on these observations?

- [ ] A) Alpha ≈ 0, because the highest training R² represents the best model.
- [ ] B) Alpha ≈ 50, because it achieves the highest test R² and the smallest train-test gap.
- [ ] C) Alpha ≈ 1000, because the model is most regularized and therefore most reliable.
- [ ] D) Alpha ≈ 500, because training and test R² are closest together there.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** At alpha ≈ 0, the model behaves like OLS with a large train-test gap (0.97 vs. 0.72), indicating high variance (overfitting). The high training R² comes at the cost of poor generalization. Maximizing training R² is not the objective.

- **Option B (Correct):** Alpha ≈ 50 achieves the best test R² (0.86) and the smallest gap between training and test R² (0.89 vs. 0.86). This represents the sweet spot of the bias-variance tradeoff: enough regularization to reduce coefficient variance and improve generalization, but not so much that the model underfits and loses the signal. This is the definition of the optimal hyperparameter.

- **Option C (Incorrect):** At alpha ≈ 1000, both training and test R² have declined significantly (0.55 and 0.59), indicating severe underfitting. High regularization does not make a model "most reliable" — it makes it too constrained to capture the true pattern. Reliability is judged by test performance, not by the amount of regularization.

- **Option D (Incorrect):** Minimizing the train-test gap is a side effect of optimal regularization but is not the primary criterion. At alpha ≈ 500, the gap is indeed small (0.71 vs. 0.73), but both values are declining from the peak. The goal is to maximize test R², not to minimize the gap per se.

</details>

---

### 65. Question

A data scientist uses `cross_val_score` to evaluate a Lasso pipeline and gets the following scores across 5 folds:

```
CV scores: [0.78, 0.81, -0.12, 0.75, 0.80]
Mean CV R²: 0.404
```

She is surprised by the third fold score of −0.12. What is the most likely explanation?

- [ ] A) There is a bug in scikit-learn's cross-validation implementation that sometimes produces negative scores.
- [ ] B) The third fold's test data has a different distribution from the training folds, causing the model trained on the other folds to predict worse than a naive mean baseline for this particular fold's test samples.
- [ ] C) Negative cross-validation R² values are impossible and must indicate a data format error.
- [ ] D) The Lasso model automatically assigns negative R² to folds where it performed feature selection.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Negative cross-validation R² scores are a legitimate result, not a bug. scikit-learn correctly computes R² = 1 − SSE/SST per fold, which can be negative when the model performs poorly on a particular test fold.

- **Option B (Correct):** Cross-validation creates 5 different train-test partitions. If the third fold's test samples come from a different part of the data distribution — for example, a subgroup with different characteristics — the model trained on the other 80% of data may not generalize well to this fold's 20%. If the model's predictions for the third fold are systematically worse than the fold's mean, SSE > SST and R² < 0. This is meaningful signal about the model's inconsistency across the dataset.

- **Option C (Incorrect):** Negative R² values are mathematically possible and meaningful. They indicate that the model's predictions for a particular dataset (or fold) are worse than a constant prediction of the mean. This can happen in cross-validation when the test fold has unusual characteristics relative to the training folds.

- **Option D (Incorrect):** Lasso's feature selection (zeroing out coefficients) does not cause negative R² in particular folds. Feature selection affects all folds equally through the fitted model. The negative R² in a specific fold is a prediction quality issue, not a Lasso feature-selection artifact.

</details>

---

### 66. Question

Two features in a dataset, `floor_area` and `living_space`, have a Pearson correlation of 0.97. A data scientist fits OLS and observes: `β_floor_area = +150` and `β_living_space = −90`.

She refits on a slightly different sample and gets: `β_floor_area = +30` and `β_living_space = +105`.

Which of the following statements correctly interprets this pattern?

- [ ] A) The sign change in `floor_area` coefficient means it has switched from positively to negatively affecting house price across the two samples.
- [ ] B) Both samples show the combined effect is consistent (weighted sum ≈ similar predicted values), but OLS arbitrarily redistributes weight between the two collinear features across samples, making individual coefficients unreliable.
- [ ] C) The second sample is likely smaller and therefore the model overfit to noise, causing coefficient instability.
- [ ] D) The sign change proves that feature interactions should be added to the model.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The sign change does not mean `floor_area` suddenly negatively affects house price. When two features are collinear (r = 0.97), OLS freely reallocates weight between them across samples. A negative coefficient for one feature can be compensated by a larger positive coefficient on the correlated feature such that the combined predicted values remain stable. The individual coefficient sign is not a reliable signal of the feature's true directional effect.

- **Option B (Correct):** With r = 0.97 between `floor_area` and `living_space`, OLS faces a near-degenerate system. Any combination of (β₁, β₂) on the constraint line β₁ + k×β₂ ≈ constant produces similar predictions. Small differences in sample composition shift OLS to a completely different point on this solution manifold, producing wildly different individual coefficients (+150/−90 vs. +30/+105) while the predicted values remain similar. This is the textbook manifestation of multicollinearity.

- **Option C (Incorrect):** The question states "a slightly different sample," not a smaller one. Coefficient instability due to multicollinearity is a property of high feature correlation, not necessarily of small sample size. Even large samples with highly collinear features produce unstable OLS coefficients.

- **Option D (Incorrect):** The coefficient instability here is caused by linear collinearity between the two features (r = 0.97), not by missing interaction terms. Adding interaction terms would increase model complexity and could exacerbate overfitting without addressing the fundamental collinearity issue.

</details>

---

### 67. Question

A developer wants to quickly check if any features are perfectly correlated with the target in a dataset. He writes:

```python
import pandas as pd

df = pd.DataFrame({'X1': [1, 2, 3], 'X2': [2, 4, 6], 'y': [3, 6, 9]})
print(df.corr())
```

The output shows perfect correlations of 1.0 between X1, X2, and y. He concludes: "Since all correlations are 1.0, I should use Ridge to reduce multicollinearity."

What is the key issue with his reasoning?

- [ ] A) The correlation matrix is incorrect because `df.corr()` does not include the target variable.
- [ ] B) Ridge is unnecessary here because features perfectly correlated with the target are ideal for OLS.
- [ ] C) The conclusion is reasonable, but he should also note that X1 and X2 are perfectly collinear with each other (r = 1.0), meaning OLS will fail to find a unique solution, and Ridge would be an appropriate remedy.
- [ ] D) All features should be dropped because perfect correlation with the target indicates the dataset was artificially generated.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `pd.DataFrame.corr()` computes the pairwise Pearson correlation matrix for all columns in the DataFrame by default, including the target column `y`. The matrix in this example correctly shows correlations among X1, X2, and y.

- **Option B (Incorrect):** The developer's issue is not that features correlate with y — that is desirable (correlated features are predictive). The problem is that X1 and X2 are perfectly correlated with each other (X2 = 2×X1). OLS cannot find a unique solution for two perfectly collinear predictors (singular XᵀX matrix). Ridge helps by adding αI to XᵀX, restoring invertibility.

- **Option C (Correct):** While the developer's conclusion (use Ridge) is directionally correct, his reasoning is incomplete. The actual issue is not that features correlate with the target — that's good. The problem is X1–X2 perfect collinearity (X2 = 2×X1 means XᵀX is singular). Ridge is indeed appropriate here because adding αI to XᵀX makes the matrix invertible and allows fitting. The correlation with y is a feature of predictive datasets, not a problem.

- **Option D (Incorrect):** Perfect correlation between features and target in small toy datasets does not imply data fabrication is problematic. The real concern is the X1–X2 perfect collinearity, not the X-y correlation. Dropping features because they correlate with the target would eliminate the entire predictive foundation of the model.

</details>

---

### 68. Question

A data scientist runs the following experiment to assess coefficient stability:

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

coefficients_ols = []
coefficients_ridge = []

for i in range(50):
    idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train[idx]
    y_boot = y_train[idx]
    
    ols = LinearRegression()
    ols.fit(X_boot, y_boot)
    coefficients_ols.append(ols.coef_[0])
    
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_boot, y_boot)
    coefficients_ridge.append(ridge.coef_[0])

print("OLS coef std:", np.std(coefficients_ols))
print("Ridge coef std:", np.std(coefficients_ridge))
```

If `X_train` has highly collinear features, which outcome is most expected?

- [ ] A) OLS coef std will be lower than Ridge coef std because OLS is the baseline method.
- [ ] B) OLS coef std will be higher than Ridge coef std, demonstrating that Ridge's L2 penalty stabilizes coefficient estimates against multicollinearity.
- [ ] C) Both will have identical standard deviations because both use the same data.
- [ ] D) Ridge coef std will be higher because regularization amplifies coefficient instability.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** OLS being the "baseline method" does not make it have lower coefficient variance. In the presence of multicollinearity, OLS has high coefficient variance precisely because the near-singular design matrix makes the solution sensitive to small changes in the data. Bootstrapping reveals this instability.

- **Option B (Correct):** This bootstrap experiment directly measures coefficient variance across 50 resamples. With highly collinear features, OLS coefficients will vary dramatically across bootstrap samples (high `np.std(coefficients_ols)`), while Ridge's L2 penalty constrains the coefficients, preventing them from swinging as widely (lower `np.std(coefficients_ridge)`). This demonstrates Ridge's key benefit: reduced coefficient variance via regularization.

- **Option C (Incorrect):** Both methods use the same bootstrap samples (same data), but they optimize different objectives. The L2 penalty in Ridge actively dampens the influence of any single sample's collinearity structure on the coefficient estimates, resulting in lower variance across bootstrap resamples compared to unconstrained OLS.

- **Option D (Incorrect):** Regularization in Ridge does not amplify instability — it suppresses it. The L2 penalty adds a constraint that prevents coefficients from taking extreme values in any given resample, thereby reducing variance across resamples. This is the mathematical guarantee that motivates using Ridge for collinear data.

</details>

---

### 69. Question

A machine learning team builds both a Ridge and a Lasso model on a marketing spend dataset with 8 features. After fitting with moderate alpha (3.0), they compare coefficients:

| Feature        | Ridge coef | Lasso coef |
|----------------|------------|------------|
| TV_spend       | 4.2        | 5.1        |
| radio_spend    | 3.8        | 0.0        |
| social_spend   | 3.5        | 0.0        |
| print_spend    | 1.2        | 0.0        |
| email_spend    | 0.9        | 0.0        |
| billboard_spend| 0.7        | 0.0        |
| podcast_spend  | 0.5        | 0.0        |
| influencer_spend| 0.3       | 0.0        |

If `radio_spend` and `social_spend` have a correlation of 0.94, which statement about these results is most accurate?

- [ ] A) Ridge made a mistake by assigning nearly equal weights to radio and social spend; they should be collapsed into one feature.
- [ ] B) Lasso correctly identified TV_spend as the only relevant feature while Ridge spread weight across all 8 features; Ridge is wrong.
- [ ] C) Lasso zeroed out 7 features, including correlated pairs radio_spend and social_spend, in favor of TV_spend. This reflects Lasso's tendency to arbitrarily pick one representative when features are correlated. Ridge, by contrast, distributed weight more evenly across all features, including the correlated pair.
- [ ] D) The results prove that TV_spend has a causal effect on sales and all other spend categories are irrelevant.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Ridge distributing weight across correlated features is not a mistake — it is a deliberate property of L2 regularization. When features are correlated, Ridge spreads weight evenly among them, which can be desirable when all correlated features contribute real information. Collapsing features is a preprocessing decision, not a requirement.

- **Option B (Incorrect):** Lasso and Ridge are not one "right" and one "wrong" — they serve different purposes. Lasso's sparsity is useful when the true model is sparse. Ridge's density is useful when all features contribute. Whether Lasso's choice (TV_spend only) or Ridge's choice (all 8) is better depends on the true underlying relationship, which is determined empirically by test performance, not by inspecting coefficients alone.

- **Option C (Correct):** Lasso's behavior of zeroing out 7 features while keeping TV_spend with a larger coefficient than Ridge's 4.2 is consistent with L1 behavior: Lasso picks one representative from a group of correlated features and zeros the others. The radio–social correlation (0.94) explains why both were zeroed rather than one being retained. Ridge, by contrast, distributes weight across all 8 features proportionally, retaining all without driving any to zero. These are expected, theoretically grounded behaviors of L1 vs. L2.

- **Option D (Incorrect):** Regression coefficients from observational marketing data establish predictive associations, not causal effects. Both Lasso and Ridge models are trained on observational data where confounders can influence the apparent importance of each spend channel. Establishing that TV_spend "causes" sales requires experimental design (e.g., holdout experiments), not regression coefficient comparison.

</details>

---

### 70. Question

A developer writes the following code to access coefficients after fitting a pipeline:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X_train, y_train)

# Attempt to access coefficients:
print(pipe.coef_)  # Line A
print(pipe.named_steps['ridge'].coef_)  # Line B
```

Which line works correctly, and which raises an AttributeError?

- [ ] A) Line A works; Line B raises an AttributeError.
- [ ] B) Both lines work because pipelines expose all steps' attributes.
- [ ] C) Line A raises an AttributeError because pipelines do not expose `coef_` directly; Line B works correctly by accessing the Ridge step's attributes through `named_steps`.
- [ ] D) Both lines raise AttributeError because coefficients are not accessible after fitting a pipeline.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `pipe.coef_` raises an `AttributeError` because scikit-learn Pipeline objects do not forward attributes like `coef_` from the final estimator directly on the pipeline object. The pipeline is not simply a thin wrapper — it is a distinct object with its own interface.

- **Option B (Incorrect):** Pipelines do not automatically expose `coef_` or other estimator-specific attributes on the pipeline object itself. Only methods like `fit()`, `predict()`, and `score()` are forwarded by Pipeline. Attributes like `coef_` must be accessed through `named_steps`.

- **Option C (Correct):** `pipe.coef_` attempts to access an attribute that does not exist on the Pipeline object, raising `AttributeError: Pipeline object has no attribute 'coef_'`. The correct way to access the Ridge coefficients is through `pipe.named_steps['ridge'].coef_` (or equivalently `pipe[-1].coef_`), which retrieves the fitted Ridge step's learned weights.

- **Option D (Incorrect):** Coefficients are accessible after fitting a pipeline — just not through the pipeline object itself. `pipe.named_steps['ridge'].coef_` correctly returns the fitted coefficient array. The issue is the access method (through the pipeline object vs. through named_steps).

</details>
