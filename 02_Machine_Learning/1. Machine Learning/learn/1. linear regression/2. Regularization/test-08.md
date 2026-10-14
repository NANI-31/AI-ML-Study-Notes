# Regularization Objective Test — Set 08

---

### 71. Question

A data science team is debugging the following pipeline code that is producing unexpectedly poor test R² results:

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)  # BUG

lasso = Lasso(alpha=0.5)
lasso.fit(X_train_s, y_train)
print(lasso.score(X_test_s, y_test))
```

Identify the bug and explain its effect.

- [ ] A) The bug is that `Lasso(alpha=0.5)` should be `Lasso(alpha=5.0)`; the alpha value is too low.
- [ ] B) The bug is that `train_test_split` without `random_state` produces non-reproducible splits, causing poor performance.
- [ ] C) The bug is `scaler.fit_transform(X_test)` — the scaler is being refitted on the test set, using test mean/std to scale test features instead of training mean/std. This mismatch between training-scale and test-scale features produces incorrect model inputs during prediction.
- [ ] D) There is no bug; `fit_transform` on the test set is a standard practice to normalize each split independently.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The alpha value determines regularization strength and affects coefficient magnitude, not the correctness of the preprocessing pipeline. While alpha tuning matters, it is not the source of the described bug.

- **Option B (Incorrect):** Omitting `random_state` makes the split non-reproducible but does not cause a systematic bug in the scaling logic. A different random split each run is a reproducibility concern, not a data leakage or transformation error.

- **Option C (Correct):** The critical bug is on line `X_test_s = scaler.fit_transform(X_test)`. This refits the `StandardScaler` using the test set's own mean and standard deviation, then transforms the test data accordingly. However, the Lasso model was trained on features scaled with the training set's statistics. During prediction, the model expects test features scaled by the training mean and standard deviation. Using different scaling statistics for test data means the model receives feature values in a different space than it was trained in, producing incorrect and systematically biased predictions. The correct approach: `X_test_s = scaler.transform(X_test)` (using the already-fitted scaler).

- **Option D (Incorrect):** Refitting the scaler on the test set is not standard practice — it is a data preprocessing error. The correct practice is to fit the scaler on training data only and apply that same fitted scaler (using `transform`, not `fit_transform`) to the test set.

</details>

---

### 72. Question

An engineer examines the following experimental results comparing OLS and Ridge for a single dataset:

| Alpha | Model   | Train R² | Test R² | Coefficient variance (across 100 bootstraps) |
|-------|---------|----------|---------|----------------------------------------------|
| N/A   | OLS     | 0.98     | 0.75    | 2,800                                        |
| 0.1   | Ridge   | 0.97     | 0.79    | 1,200                                        |
| 10    | Ridge   | 0.93     | 0.83    | 340                                          |
| 100   | Ridge   | 0.81     | 0.80    | 82                                           |
| 1000  | Ridge   | 0.62     | 0.61    | 11                                           |

Which alpha best represents the **optimal** hyperparameter based on all three metrics?

- [ ] A) Alpha = 0.1, because it has the highest training R² among all Ridge models.
- [ ] B) Alpha = 10, because it achieves the best test R² (0.83) while substantially reducing coefficient variance (340 vs. OLS's 2,800).
- [ ] C) Alpha = 1000, because it has the lowest coefficient variance, indicating the most stable model.
- [ ] D) Alpha = 100, because it has the smallest difference between training and test R².

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Maximizing training R² is not the goal — generalization to new data is. Alpha = 0.1 has the highest Ridge training R² (0.97) but a lower test R² (0.79) than alpha = 10 (0.83). Prioritizing training performance leads to overfitting.

- **Option B (Correct):** Alpha = 10 achieves the highest test R² (0.83) among all configurations, demonstrating the best generalization. It also reduces coefficient variance from OLS's 2,800 to 340 — an 88% reduction — indicating much more stable and reliable coefficient estimates. This alpha represents the sweet spot of the bias-variance tradeoff: enough regularization to cut variance substantially, without so much that test R² begins to decline.

- **Option C (Incorrect):** Alpha = 1000 has the lowest coefficient variance (11), but both training and test R² have dropped severely (0.62 and 0.61). The model is underfitting — it has become too constrained to capture the true signal. Lowest variance does not mean best model; extremely biased (underfit) models can also have low variance.

- **Option D (Incorrect):** Minimizing the train-test gap is related to the bias-variance tradeoff but is not the primary optimization criterion. At alpha = 100, the gap is 0.01 (0.81 vs. 0.80), but test R² (0.80) is below the peak at alpha = 10 (0.83). Maximizing test R² is the correct objective; a minimal gap is a desirable property but not the direct target.

</details>

---

### 73. Question

A data scientist is tasked with explaining why the OLS model fails for a dataset where the number of features (p = 300) exceeds the number of observations (n = 150).

Which of the following correctly identifies the problem and the appropriate remedy?

- [ ] A) OLS fails because R² cannot be computed when p > n. Using Lasso or Ridge allows R² computation.
- [ ] B) OLS fails because the p × p design matrix (XᵀX) is singular when p > n, making its inverse undefined. Regularization (Ridge or Lasso) adds a diagonal term to XᵀX (Ridge: αI), restoring invertibility and enabling fitting.
- [ ] C) OLS fails because it requires exactly p = n; adding more observations until p = n resolves the problem without regularization.
- [ ] D) OLS fails only when p > n and all features are numeric; categorical features resolve the singularity.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** R² is computable as long as predictions can be generated. The issue with p > n for OLS is not R² computation but the non-existence of a unique solution for the OLS normal equations. R² is not the blocking point; matrix invertibility is.

- **Option B (Correct):** When p > n, the n × p design matrix X has rank at most n < p. Therefore XᵀX (a p × p matrix) is also rank-deficient (singular) and cannot be inverted. The OLS normal equations (XᵀX)β = Xᵀy have infinitely many solutions rather than a unique one. Ridge regression modifies the normal equations to (XᵀX + αI)β = Xᵀy. Adding αI (a positive definite matrix) to XᵀX guarantees the result is positive definite and invertible for any α > 0, enabling a unique solution. Lasso handles this differently (via coordinate descent), but also produces a valid solution.

- **Option C (Incorrect):** Adding more observations until n = p would help (a square XᵀX), but this is often impractical, and a full-rank XᵀX requires n > p along with linear independence among features. The practical remedy in machine learning is regularization (Ridge/Lasso), not waiting to collect enough data.

- **Option D (Incorrect):** The singular design matrix problem when p > n applies regardless of whether features are numeric or categorical. A one-hot encoded categorical feature still contributes columns to the design matrix, and if p > n still holds after encoding, the matrix is still rank-deficient.

</details>

---

### 74. Question

An analyst builds a model predicting quarterly revenue using 5 economic indicators. The model has:

- Ridge coefficients (after scaling): `[8.5, −2.3, 0.1, 0.09, −0.04]`
- Intercept: `120,000`

She notes: "The first coefficient (8.5) is much larger in magnitude, so feature 1 is definitely the most economically important driver of revenue."

Is this interpretation valid?

- [ ] A) Yes, standardized Ridge coefficients directly reflect each feature's proportional economic importance.
- [ ] B) No, because Ridge shrinks all coefficients equally toward zero, making them all equally important regardless of magnitude.
- [ ] C) Yes, but only if the features are on the same original scale before standardization.
- [ ] D) Largely yes — after standardizing features, the magnitude of Ridge coefficients reflects the relative predictive importance of each feature within this model on this dataset, though it does not establish economic causality.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** "Directly reflect proportional economic importance" is too strong — Ridge coefficients after standardization reflect predictive importance (relative contribution to the model's predictions), not necessarily causal or economic importance. Confounders and collinearity can distort even standardized coefficients.

- **Option B (Incorrect):** Ridge does not shrink all coefficients equally toward zero — it shrinks them proportionally based on the optimization (specifically, it shrinks features with larger coefficients more). Features genuinely more predictive retain larger shrunken coefficients, so coefficient magnitudes are not rendered meaningless. The statement that all features are equally important regardless of magnitude is incorrect.

- **Option C (Incorrect):** Standardizing features before fitting (which is done here — "after scaling") already accounts for the original scale differences. The intercept condition in Option C is already satisfied by the stated analysis.

- **Option D (Correct):** After standardizing features (which places all features on a comparable scale), the magnitude of Ridge coefficients does reflect the relative predictive contribution of each feature within this particular model on this particular dataset. A coefficient of 8.5 vs. 0.04 suggests feature 1 has much greater predictive weight. However, this is a model-fit interpretation under Ridge's regularization — it does not establish that feature 1 causally drives revenue by that amount, especially in observational data with potential confounders and multicollinearity.

</details>

---

### 75. Question

A senior analyst reviews the following code that a junior engineer claims "properly implements Lasso with cross-validation":

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

best_alpha = None
best_score = -np.inf

for alpha in [0.01, 0.1, 1.0, 10.0]:
    lasso = Lasso(alpha=alpha)
    scores = cross_val_score(lasso, X_test, y_test, cv=3)  # BUG
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_alpha = alpha

print(f"Best alpha: {best_alpha}")
```

Identify the bug.

- [ ] A) `cross_val_score` does not accept Lasso; only Ridge can be used with it.
- [ ] B) The cross-validation is being run on `X_test, y_test` instead of `X_train, y_train`. This means the test set is being used for hyperparameter tuning, which is data leakage and gives an optimistically biased estimate of the best alpha.
- [ ] C) The bug is that `cv=3` should be `cv=5`; fewer than 5 folds always produces incorrect results.
- [ ] D) The loop should run `model.fit()` before `cross_val_score`; otherwise Lasso has not been initialized.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `cross_val_score` accepts any scikit-learn estimator, including Lasso. There is no restriction preventing Lasso from being used with `cross_val_score`.

- **Option B (Correct):** The critical bug is `cross_val_score(lasso, X_test, y_test, cv=3)`. Hyperparameter tuning (selecting the best alpha) should be done using the training set (with internal cross-validation), not the test set. By running cross-validation on `X_test, y_test`, the engineer is using test set information to select the hyperparameter. The selected alpha will be optimistically tuned to the specific test set, and the final reported `best_score` will not represent unbiased generalization performance. The correct call should use `X_train, y_train`.

- **Option C (Incorrect):** The number of folds (3 vs. 5) affects the variance of the cross-validation estimate, but using 3 folds is not inherently incorrect. The fundamental bug is using the test set for cross-validation, not the number of folds.

- **Option D (Incorrect):** `cross_val_score` handles fitting internally — it clones the estimator, fits it on each training fold, and evaluates on each validation fold. There is no need to call `model.fit()` before `cross_val_score`. The estimator passed in (Lasso) only needs to be instantiated with the desired alpha.

</details>

---

### 76. Question

A machine learning team is evaluating whether their regularized linear model is appropriate for a policy decision. The model predicts student test scores from school funding, teacher experience, class size, and neighborhood income.

Which combination of statements correctly characterizes the model's limitations for policy use?

- [ ] A) High R² proves the model can be used to design causal policies, and multicollinearity does not matter for prediction.
- [ ] B) The model's predictions are useful for forecasting, but its coefficients should not be used for causal policy design without addressing confounders and establishing causal identification.
- [ ] C) Regularization completely removes the confounding effect of neighborhood income, making the coefficients causally valid.
- [ ] D) The model is unfit for any purpose because regularization introduces bias.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** High R² measures predictive association, not causal structure. A model trained on observational data (where school funding correlates with many unmeasured factors like parental involvement, district resources, and demographics) cannot be used for causal policy decisions without causal identification. Multicollinearity also affects coefficient reliability for interpretation.

- **Option B (Correct):** Regularized linear models can provide useful predictions when the goal is forecasting (e.g., predicting likely test score outcomes for a new cohort of students). However, their coefficients represent associations in observational data, not causal effects. Using the "school funding coefficient" to decide how much funding to allocate assumes the coefficient reflects a causal effect — which requires ruling out confounders (e.g., neighborhood income confounds the relationship between funding and outcomes). Causal methods (instrumental variables, regression discontinuity, natural experiments) are required for policy design.

- **Option C (Incorrect):** Regularization (Ridge or Lasso) adjusts coefficient magnitudes to reduce variance and improve generalization, but it does not remove the statistical confounding of neighborhood income. Confounding is a causal structure problem (neighborhood income causes both funding levels and test scores), not a coefficient magnitude problem. No regularization method eliminates confounders — only experimental or causal identification strategies do.

- **Option D (Incorrect):** Regularization introducing some bias is a deliberate tradeoff for lower variance and better generalization — it does not make the model "unfit for any purpose." The model is well-suited for prediction and forecasting. The limitation is specifically about causal inference, not prediction generally.

</details>

---

### 77. Question

A data scientist observes that for her regression dataset, increasing Ridge's alpha from 1 to 500 causes the test R² to go from 0.80 to 0.83. She concludes: "I should always use the maximum alpha I can find."

What is wrong with this reasoning?

- [ ] A) Nothing is wrong; a higher test R² always justifies a higher alpha.
- [ ] B) The maximum alpha should always be 1.0 by convention in Ridge regression.
- [ ] C) Test R² can continue to improve with higher alpha only up to a point; beyond the optimal alpha, excessive regularization causes underfitting, and test R² will decline. The maximum alpha is not necessarily optimal.
- [ ] D) She should use Lasso instead because Ridge is always suboptimal at high alpha values.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Test R² improves with alpha only in the range where reducing coefficient variance outweighs the increase in bias. Beyond the optimal alpha, the model loses too much signal (high bias) and test R² starts declining. Maximizing alpha does not guarantee maximizing test R².

- **Option B (Incorrect):** There is no conventional maximum of 1.0 for Ridge's alpha. The optimal alpha is dataset-dependent and can be any positive value — it is determined by cross-validation, not by convention.

- **Option C (Correct):** The bias-variance tradeoff describes an inverted U-shape for test R² as a function of regularization strength. As alpha increases from 0: initially, reducing variance improves test R². Beyond the optimal alpha, the additional bias from over-constraining coefficients begins to dominate, and test R² declines. The engineer observed the improvement phase (0.80 to 0.83) but has not yet tested whether further alpha increases continue to improve or start to degrade test R². Selecting the maximum possible alpha is statistically unjustified without empirical verification.

- **Option D (Incorrect):** Lasso and Ridge have different properties (sparsity vs. density), and neither is universally optimal. At high alpha values, both Lasso and Ridge can underfit. The engineer's observation about alpha performance doesn't support switching to Lasso.

</details>

---

### 78. Question

A junior data scientist is asked to explain what happens geometrically when the Lasso constraint forces a coefficient to exactly zero. Which explanation is most accurate?

- [ ] A) The Lasso constraint sphere has no corners, so solutions are always at interior points away from zero.
- [ ] B) The Lasso constraint's L1 ball (diamond shape in 2D) has sharp corners at the coordinate axes. The OLS loss ellipses expand outward from the OLS minimum; when the first ellipse touches the L1 ball, it typically makes contact at a corner where one or more axes are set to zero.
- [ ] C) Lasso forces coefficients to zero by computing the covariance between each feature and the residual and setting it to zero if below a threshold.
- [ ] D) Lasso forces coefficients to zero by iteratively removing the feature with the worst training R².

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Option A describes the L2 (Ridge) constraint, which IS a smooth sphere with no corners. The L1 (Lasso) constraint has a diamond shape with sharp corners at the coordinate axes — the opposite of what Option A describes. It is the L1 constraint's corners that enable exact zero solutions.

- **Option B (Correct):** The geometric intuition for Lasso's sparsity is: the feasible region defined by `Σ|βⱼ| ≤ t` (for regularization strength t = 1/α) forms a diamond (L1 ball) in 2D or a cross-polytope in higher dimensions. This shape has corners that lie exactly on the coordinate axes. The OLS loss function forms concentric ellipses around the unconstrained OLS solution. When these expanding ellipses first touch the L1 constraint region, they typically make contact at a corner — a point where one or more coefficients are exactly zero. This geometric argument explains why Lasso produces sparse solutions while Ridge (smooth sphere constraint) does not.

- **Option C (Incorrect):** This describes a simplified heuristic of the coordinate descent algorithm used to solve Lasso, where the soft-thresholding step zeros coefficients below a threshold. While coordinate descent with soft-thresholding is indeed how Lasso is solved computationally, the question asks for the geometric interpretation, not the algorithmic mechanism.

- **Option D (Incorrect):** Lasso does not iteratively remove features based on training R². It simultaneously optimizes all coefficients via the regularized loss function. The feature removal (zeroing) is an emergent property of the optimization geometry, not a stepwise selection procedure based on R² scores.

</details>

---

### 79. Question

A data scientist applies the following Ridge model and checks predictions:

```python
from sklearn.linear_model import Ridge
import numpy as np

# Fitted model on scaled features
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# Model attributes after fitting
print(ridge.coef_)       # [3.2, -1.5, 0.8]
print(ridge.intercept_)  # 42.0

# New scaled observation
x_new = np.array([[2.0, 1.0, -1.0]])
prediction = ridge.predict(x_new)
print(prediction)
```

What is the predicted value?

- [ ] A) ŷ = 3.2(2) + (−1.5)(1) + 0.8(−1) = 6.4 − 1.5 − 0.8 = 4.1
- [ ] B) ŷ = 3.2(2) + (−1.5)(1) + 0.8(−1) + 42 = 6.4 − 1.5 − 0.8 + 42 = 46.1
- [ ] C) ŷ = 3.2 + (−1.5) + 0.8 + 42 = 44.5 (sum of all coefficients plus intercept)
- [ ] D) ŷ = (3.2 × 2 − 1.5 × 1 + 0.8 × (−1)) × 42 = 4.1 × 42 = 172.2

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The calculation of the linear combination is correct (3.2×2 + (−1.5)×1 + 0.8×(−1) = 6.4 − 1.5 − 0.8 = 4.1), but the intercept (42.0) is omitted. The full linear prediction model is ŷ = coef_ · x + intercept_, and the intercept must always be included.

- **Option B (Correct):** ŷ = 3.2×2.0 + (−1.5)×1.0 + 0.8×(−1.0) + 42.0 = 6.4 − 1.5 − 0.8 + 42.0 = 46.1. This is the correct application of the linear model formula: the dot product of coefficients and features, plus the intercept. `ridge.predict()` computes exactly this.

- **Option C (Incorrect):** This option simply sums all coefficients and the intercept, treating each coefficient as if the corresponding feature value were 1. This is only correct when all feature values equal 1, which is not the case here (x_new = [2.0, 1.0, −1.0]).

- **Option D (Incorrect):** Multiplying the linear combination (4.1) by the intercept (42) as a scale factor has no mathematical basis in linear regression. The intercept is added to the linear combination, not multiplied with it.

</details>

---

### 80. Question

A researcher is explaining why SST on a test set can differ from SST on the training set. A student says: "SST is fixed because it's just the variance of y, and y doesn't change between training and test sets."

What is the correct rebuttal?

- [ ] A) The student is correct; SST is always the same across training and test sets because it depends only on the target variable y.
- [ ] B) The student is incorrect; SST is computed using the mean (ȳ) of the specific dataset being evaluated. The training set and test set have different samples with different means, so SST will differ between them.
- [ ] C) The student is incorrect; SST for the test set uses the model's predicted mean, not the actual test mean.
- [ ] D) The student is correct only when the data is i.i.d. (identically and independently distributed).

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** SST = Σ(yᵢ − ȳ)². The ȳ here is the mean of the specific sample (training or test). Because training and test sets contain different observations, their empirical means (ȳ_train ≠ ȳ_test in general) and their variances will differ. SST is not a fixed constant of the full dataset.

- **Option B (Correct):** When scikit-learn computes R² on the test set via `model.score(X_test, y_test)`, it uses ȳ_test (the mean of the test labels) to compute SST. SST_test = Σ(yᵢ_test − ȳ_test)². This is different from SST_train = Σ(yᵢ_train − ȳ_train)² because the test set's observations and mean are different from the training set's. The student's error was assuming ȳ is the same across sets.

- **Option C (Incorrect):** SST is computed using the mean of the actual y values in the evaluation set — not the model's predicted mean. SST represents the variance of the true targets around their actual mean, serving as the baseline (null model) against which the model's SSE is compared.

- **Option D (Incorrect):** Even for i.i.d. data, the empirical means of randomly drawn training and test sets will differ due to sampling variability. The i.i.d. assumption means they come from the same distribution, so their expected means are equal, but the realized sample means in any finite split will differ.

</details>
