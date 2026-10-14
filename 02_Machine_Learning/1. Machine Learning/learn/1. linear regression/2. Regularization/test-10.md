# Regularization Objective Test — Set 10

---

### 91. Question

A senior engineer reviews this code used to tune a Lasso model across a dataset with severe multicollinearity:

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

alpha_candidates = [0.001, 0.01, 0.1, 1, 10, 100]
cv_scores = {}

for alpha in alpha_candidates:
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha))
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    cv_scores[alpha] = scores.mean()

best_alpha = max(cv_scores, key=cv_scores.get)
final_model = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha))
final_model.fit(X_train, y_train)
print("Final Test R²:", final_model.score(X_test, y_test))
```

Which statement correctly evaluates this code?

- [ ] A) The code is flawed because `cross_val_score` should be run on `X_test, y_test`, not `X_train, y_train`.
- [ ] B) The code is correct — it uses cross-validation on the training set to select alpha, fits the final model on all training data, and evaluates once on the untouched test set. This is the standard ML pipeline for hyperparameter tuning.
- [ ] C) The code is flawed because `StandardScaler` should be applied outside the pipeline to ensure consistent scaling across all folds.
- [ ] D) The code is flawed because Lasso cannot be used with `make_pipeline`.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Running `cross_val_score` on `X_test, y_test` is the data leakage pattern that this code deliberately avoids. The test set must remain completely unseen during hyperparameter tuning. Using `X_train, y_train` for cross-validation is the correct approach.

- **Option B (Correct):** This code follows the gold standard ML pipeline: (1) split data into train and test before any modeling; (2) use 5-fold cross-validation on training data only to select the best alpha; (3) refit the final model on the full training set using the best alpha; (4) evaluate once on the held-out test set. The pipeline ensures StandardScaler is fitted only on training fold data within each CV iteration, preventing leakage. The test set is touched only at the final evaluation step.

- **Option C (Incorrect):** Scaling outside the pipeline and before cross-validation is precisely the data leakage antipattern described in earlier questions. Placing StandardScaler inside the pipeline ensures it is fitted fresh on each fold's training data during cross-validation, avoiding contamination from test fold statistics.

- **Option D (Incorrect):** Lasso is fully compatible with `make_pipeline`. scikit-learn's `make_pipeline` accepts any estimator that implements `fit()`, `predict()`, and `score()`. Lasso implements all three. There is no restriction on using Lasso in a pipeline.

</details>

---

### 92. Question

For a 5-observation dataset, an analyst computes the following during model evaluation:

- Actuals: `[20, 25, 30, 35, 40]`
- Model predictions: `[22, 22, 32, 38, 36]`
- ȳ (mean of actuals) = 30

Calculate R² step by step and identify the correct answer.

- [ ] A) R² = 0.82
- [ ] B) R² = 0.90
- [ ] C) R² = 0.75
- [ ] D) R² = 0.68

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Step-by-step:
  - SSE = (20−22)² + (25−22)² + (30−32)² + (35−38)² + (40−36)²
  - SSE = 4 + 9 + 4 + 9 + 16 = 42
  - SST = (20−30)² + (25−30)² + (30−30)² + (35−30)² + (40−30)²
  - SST = 100 + 25 + 0 + 25 + 100 = 250
  - R² = 1 − 42/250 = 1 − 0.168 = **0.832 ≈ 0.82**
  
  The model explains approximately 82% of the variance in the target.

- **Option B (Incorrect):** R² = 0.90 would require SSE = 0.10 × SST = 25. The computed SSE is 42, giving a lower R² than 0.90.

- **Option C (Incorrect):** R² = 0.75 would require SSE = 0.25 × SST = 62.5. The computed SSE (42) is less than 62.5, yielding a higher R² than 0.75.

- **Option D (Incorrect):** R² = 0.68 would require SSE = 0.32 × SST = 80. The computed SSE (42) is far less than 80, producing a higher R² than 0.68.

</details>

---

### 93. Question

A data scientist encounters the following code that claims to compare OLS and Lasso fairly:

```python
from sklearn.linear_model import LinearRegression, Lasso

ols = LinearRegression()
lasso = Lasso(alpha=1.0)

ols.fit(X_train, y_train)
lasso.fit(X_train, y_train)  # No scaling applied

print("OLS test R²:", ols.score(X_test, y_test))
print("Lasso test R²:", lasso.score(X_test, y_test))
```

The dataset has features on very different scales: `income` (range $10K–$500K), `age` (range 20–70), `num_children` (range 0–8). The conclusion is: "Lasso performs worse, so Ridge or OLS should be used."

What is the key methodological flaw in this comparison?

- [ ] A) The flaw is that OLS and Lasso cannot be fairly compared using the same test set.
- [ ] B) The comparison is fair because both models use the same training and test data.
- [ ] C) The flaw is that Lasso was not scaled before fitting. The L1 penalty applies disproportionately to features with smaller numerical scales (like `age` and `num_children`), causing over-penalization of those features and under-penalization of `income`. This biases the comparison against Lasso, not against OLS, which has no penalty to distort.
- [ ] D) The flaw is that `alpha = 1.0` should be `alpha = 0.0` for a fair comparison with OLS.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Using the same test set to evaluate both models is not a flaw — it is the correct approach for fair comparison. Separate test sets would make comparison impossible.

- **Option B (Incorrect):** The comparison is not fair despite using the same data, because the two models operate under different assumptions about feature scale. OLS has no penalty term, so scale doesn't affect which solution it finds (OLS coefficients naturally adjust to the scale of each feature). Lasso's L1 penalty `α × Σ|βⱼ|` is applied to raw coefficient magnitudes, which are scale-dependent.

- **Option C (Correct):** Without standardizing features, `income` (range in hundreds of thousands) will have a very small coefficient (e.g., 0.001), while `num_children` (range 0–8) will have a much larger coefficient (e.g., 5000) to achieve the same predictive contribution. The Lasso penalty `|β_income| + |β_age| + |β_children|` will heavily penalize the large `num_children` coefficient while leaving the tiny `income` coefficient nearly unaffected, regardless of each feature's actual predictive importance. This distorts the L1 penalty's intent and biases the model against features with small natural scales, making the comparison against OLS unfair. Standardizing before Lasso is mandatory for a meaningful comparison.

- **Option D (Incorrect):** Setting alpha = 0 in Lasso would recover OLS (approximately), making the comparison trivial rather than correcting the fairness issue. The problem is the missing standardization, not the alpha value.

</details>

---

### 94. Question

A researcher is explaining the concept of SST as a "baseline variance" to a junior colleague. She says: "SST tells us how much variance we need to explain. If our model achieves SSE = SST, that's actually a perfect model."

What is wrong with this statement?

- [ ] A) Nothing is wrong; SSE = SST means the model explains all variance (R² = 1.0).
- [ ] B) The statement is incorrect; SSE = SST means R² = 0 (no variance explained), indicating the model performs exactly as well as a constant mean prediction — not perfectly.
- [ ] C) The statement is incorrect; SSE = SST means the model has zero residuals, which is perfect.
- [ ] D) The statement is partially correct; SSE = SST means R² = 0.5, which is adequate performance.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** SSE = SST does not give R² = 1.0. R² = 1 − SSE/SST = 1 − SST/SST = 1 − 1 = 0. For R² = 1.0, we need SSE = 0 (perfect predictions with zero residuals), not SSE = SST.

- **Option B (Correct):** R² = 1 − SSE/SST. When SSE = SST: R² = 1 − 1 = 0. This means the model explains zero variance — it performs identically to a model that predicts the constant mean (ȳ) for every observation. SST is the total sum of squared deviations from the mean; a model with SSE = SST has residuals as large as the deviations from the mean, meaning it has gained nothing over the simplest possible prediction. This is the opposite of a perfect model.

- **Option C (Incorrect):** Zero residuals means SSE = 0, not SSE = SST. When residuals are all zero, every prediction equals the actual value, giving a perfect fit (R² = 1.0). SSE = SST implies large residuals equal in total to the target's total variance.

- **Option D (Incorrect):** R² = 0.5 would require SSE = 0.5 × SST, not SSE = SST. When SSE = SST, R² = 0 exactly (as shown in Option B's explanation).

</details>

---

### 95. Question

An engineer encounters this code and is asked to identify the issue:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

model = Ridge(alpha=5.0)
scaler = StandardScaler()

# Fitting
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# Prediction on a new observation
x_new = [[800, 3, 12]]
prediction = model.predict(x_new)  # predicts on unscaled data
```

What is the issue, and what is the correct fix?

- [ ] A) The issue is that `Ridge(alpha=5.0)` should be `Ridge(alpha=50.0)` for proper regularization.
- [ ] B) The issue is that `model.predict(x_new)` receives raw unscaled features `x_new`, while the model was trained on `StandardScaler`-transformed features. The correct fix is `model.predict(scaler.transform(x_new))`.
- [ ] C) There is no issue; Ridge automatically applies the inverse of StandardScaler during prediction.
- [ ] D) The issue is that `x_new` must be a numpy array, not a Python list. The fix is `model.predict(np.array(x_new))`.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The alpha value is not the source of the bug. Alpha = 5.0 is a valid hyperparameter that produces a regularized model. The bug is in how the new data is preprocessed for prediction.

- **Option B (Correct):** The model's coefficients (`coef_`) were fitted in the scaled feature space (where training features have zero mean and unit variance). When `model.predict(x_new)` receives raw features [800, 3, 12], the model computes `coef_ · x_new + intercept_` using raw values in a scaled model space — an incorrect computation. The fix is to first apply the same transformation that was applied during training: `x_new_scaled = scaler.transform([[800, 3, 12]])`, then `model.predict(x_new_scaled)`. Using a `Pipeline` avoids this error entirely by chaining scaler and model together.

- **Option C (Incorrect):** scikit-learn's Ridge does not internally apply or invert StandardScaler during prediction. Ridge is unaware that a scaler was used during training — it only knows its own fitted coefficients. The scaler must be applied explicitly (or via a pipeline) before calling `predict()`.

- **Option D (Incorrect):** scikit-learn's `predict()` accepts Python lists and converts them to numpy arrays internally. While best practice is to use numpy arrays, a Python list of lists `[[800, 3, 12]]` is a valid input that will not cause an error. The bug is the missing scaling step, not the data type.

</details>

---

### 96. Question

A model comparison study produces the following training and test R² values across increasing Lasso alpha values:

| Alpha  | Train R² | Test R² |
|--------|----------|---------|
| 0.001  | 0.975    | 0.720   |
| 0.01   | 0.970    | 0.754   |
| 0.1    | 0.955    | 0.803   |
| 1.0    | 0.921    | 0.831   |
| 10.0   | 0.872    | 0.811   |
| 100.0  | 0.701    | 0.694   |
| 1000.0 | 0.420    | 0.398   |

Which description of the underlying trends is most accurate?

- [ ] A) As alpha increases, both train and test R² monotonically increase; regularization consistently helps.
- [ ] B) Train R² monotonically decreases as alpha increases; test R² first increases (as variance is reduced), then decreases (as bias becomes too large), peaking around alpha = 1.0.
- [ ] C) Train R² monotonically increases as alpha increases; test R² remains constant throughout.
- [ ] D) Both train and test R² follow identical trajectories, differing only in absolute value.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Train R² does not monotonically increase with alpha — it monotonically decreases as the regularization penalty prevents the model from fitting the training data as freely. The claim that test R² consistently increases is also incorrect — it peaks and then declines as underfitting takes over.

- **Option B (Correct):** The table shows: (1) Train R² decreases monotonically from 0.975 to 0.420 as alpha increases — expected because stronger penalty = worse training fit. (2) Test R² increases from 0.720 to 0.831 (peaking around alpha = 1.0) as regularization reduces overfitting, then decreases from 0.831 back to 0.398 as excessive regularization causes underfitting. This inverted-U shape of test R² vs. alpha is the fundamental bias-variance tradeoff visualization for Lasso (and Ridge).

- **Option C (Incorrect):** Train R² does not increase with alpha — the opposite is true. Adding a larger penalty to the training objective prevents the model from minimizing SSE as effectively, reducing training R². Test R² does not remain constant; it follows the inverted-U pattern as shown.

- **Option D (Incorrect):** Train R² and test R² follow qualitatively different trajectories. Train R² is monotonically decreasing (always above or equal to test R² for the same model). Test R² follows an inverted U. They are not parallel — their divergence and convergence patterns carry important information about the bias-variance tradeoff.

</details>

---

### 97. Question

A policy analyst asks: "Can I use the Lasso model's feature selection output to conclude which marketing channels definitively cause sales increases?"

Which answer best addresses the limits of Lasso for causal inference?

- [ ] A) Yes, Lasso's L1 penalty selects only causally important features by design.
- [ ] B) No; Lasso selects features based on their predictive association with the outcome in the training data, not their causal status. A feature that is a proxy for an unmeasured cause can be selected by Lasso even though it is not causal.
- [ ] C) Yes, if Lasso's alpha was selected via cross-validation, the selected features are causally validated.
- [ ] D) No, Lasso features are never predictively useful and should not be used for any decision-making.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lasso's L1 penalty selects features that minimize the regularized prediction error on training data — not features that are causally related to the outcome. A confounding variable (one that correlates with both a causal feature and the outcome) can be selected by Lasso as a surrogate predictor without being causal.

- **Option B (Correct):** Lasso identifies features that are predictively associated with sales in the observed data. However, predictive association is not the same as causation. A marketing channel might be selected by Lasso because it correlates with another unobserved causal factor (e.g., a major product launch coinciding with heavy TV advertising). Claiming the selected Lasso features "definitively cause" sales increases requires causal identification — randomized experiments, instrumental variables, or other causal inference methods that go beyond observational regression.

- **Option C (Incorrect):** Cross-validation alpha selection validates that the chosen alpha produces good generalization performance — it does not validate the causal status of selected features. Cross-validation evaluates predictive accuracy, not causal structure.

- **Option D (Incorrect):** Lasso-selected features are often predictively useful and can absolutely inform business decisions, forecasting, and planning. The limitation is specifically about causal claims (implying interventions will have the modeled effects), not about predictive usefulness in general.

</details>

---

### 98. Question

An ML engineer runs the following experiment to understand the effect of alpha in Ridge on coefficient magnitudes and R²:

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

results = []
for alpha in np.logspace(-2, 5, 50):  # 50 values from 0.01 to 100000
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, y_train)
    results.append({
        'alpha': alpha,
        'max_coef': np.max(np.abs(ridge.coef_)),
        'train_r2': ridge.score(X_scaled, y_train)
    })
```

Which description of the expected pattern in `max_coef` and `train_r2` as alpha increases is correct?

- [ ] A) `max_coef` increases and `train_r2` increases as alpha grows.
- [ ] B) `max_coef` decreases monotonically toward 0 and `train_r2` decreases monotonically as alpha grows.
- [ ] C) `max_coef` oscillates between positive and negative values while `train_r2` remains constant.
- [ ] D) `max_coef` decreases to exactly 0 at some finite alpha; `train_r2` also drops to 0 at that same point.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Higher alpha imposes a stronger L2 penalty, forcing Ridge to keep coefficients smaller. `max_coef` must therefore decrease (not increase) as alpha grows. A higher penalty means smaller coefficients. Similarly, smaller coefficients prevent the model from fitting the training data as well, reducing `train_r2`.

- **Option B (Correct):** As alpha increases from 0.01 to 100,000: `max_coef` decreases monotonically from near-OLS values (large, unconstrained) toward 0 asymptotically (but never reaches 0 for finite alpha). `train_r2` also decreases monotonically from near-OLS performance toward 0 (when all coefficients are nearly zero, the model predicts near the intercept/mean, giving R² ≈ 0 on training data). Both trends are monotonically decreasing across 50 orders-of-magnitude alpha values. This is the regularization path for Ridge.

- **Option C (Incorrect):** Ridge coefficients do not oscillate between positive and negative values. They follow a smooth, monotonically decreasing path in magnitude from their OLS values toward zero. `train_r2` also does not remain constant — it decreases as alpha-induced shrinkage prevents the model from fitting the training data.

- **Option D (Incorrect):** Ridge coefficients approach zero asymptotically as alpha → ∞ but never reach exactly zero for any finite alpha. There is no finite alpha at which `max_coef` hits exactly 0. Similarly, `train_r2` approaches 0 asymptotically but does not drop to exactly 0 until alpha is mathematically infinite.

</details>

---

### 99. Question

A data science team trains three models — OLS, Ridge, and Lasso — on a dataset of 1,000 houses with 15 features (including some highly correlated pairs). They report:

| Model           | Train R² | Test R² | # Zero Coefficients |
|-----------------|----------|---------|---------------------|
| OLS             | 0.98     | 0.74    | 0                   |
| Ridge (α=5)     | 0.93     | 0.83    | 0                   |
| Lasso (α=0.5)   | 0.89     | 0.81    | 6                   |

A manager asks: "Which model should we deploy for predictions?"

Which recommendation is most analytically justified?

- [ ] A) OLS, because it achieves the highest training R², indicating the most powerful model.
- [ ] B) Ridge (alpha=5), because it achieves the highest test R² (0.83) and retains all 15 features, providing both generalization and comprehensive feature coverage.
- [ ] C) Lasso (alpha=0.5), because fewer features always makes a better production model.
- [ ] D) All three models are equivalent because they all use linear regression.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** OLS achieves the highest training R² (0.98) but the lowest test R² (0.74) — a strong sign of overfitting, likely due to high-variance coefficients from multicollinearity. Deploying a model based on training performance maximization leads to poor real-world generalization.

- **Option B (Correct):** Ridge with alpha = 5 achieves the best test R² (0.83), indicating the best generalization performance. It retains all 15 features (no zero coefficients), allowing comprehensive feature coverage for future predictions when all input features are available. The L2 penalty has successfully reduced coefficient variance caused by multicollinearity without discarding potentially useful feature information. For a prediction-focused deployment, test R² is the primary criterion, and Ridge is the winner.

- **Option C (Incorrect):** Fewer features does not automatically produce better production models. Lasso's 6 zero coefficients reduce the model to 9 features, which may simplify deployment but sacrifices some predictive accuracy (0.81 vs. 0.83 test R²). If interpretability or sparse feature inputs are required, Lasso may be preferred — but the question asks for predictive deployment, where Ridge's 0.83 test R² is superior.

- **Option D (Incorrect):** While all three are linear models, they solve different optimization problems (OLS: pure SSE; Ridge: SSE + L2 penalty; Lasso: SSE + L1 penalty) and produce different coefficient solutions with different generalization properties. They are emphatically not equivalent — as evidenced by test R² values ranging from 0.74 to 0.83.

</details>

---

### 100. Question

A researcher reflects on the complete study of regularization and poses this capstone question: "Given all that you know about L1, L2, multicollinearity, R², and feature scaling, which of the following statements about the full regularization workflow are correct?" (Select all that apply.)

- [ ] A) Feature standardization is essential before L1/L2 regularization because the penalty is applied to raw coefficients that depend on feature scale.
- [ ] B) Multicollinearity always causes poor predictive R², even on well-separated training/test splits.
- [ ] C) Alpha hyperparameter tuning should use cross-validation on training data, and final evaluation should use a completely held-out test set that was never used during tuning.
- [ ] D) Lasso is preferred when the true underlying relationship is sparse; Ridge is preferred when many correlated features contribute jointly to the outcome.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A, C, D**

#### Explanations:

- **Option A (Correct):** Before applying L1 or L2 regularization, features must be standardized (typically to zero mean and unit variance using StandardScaler). The penalty term `α × Σ|βⱼ|` (L1) or `α × Σβⱼ²` (L2) is applied to the coefficient magnitudes directly. Without standardization, features measured in large units receive small coefficients (and thus smaller penalties), while features in small units receive large coefficients (and thus larger penalties) — an unfair, scale-dependent distortion of the regularization. Standardizing ensures each feature's coefficient is penalized proportional to its predictive contribution, not its measurement scale.

- **Option B (Incorrect):** Multicollinearity primarily affects the stability and interpretability of individual coefficients — not the model's overall predictive accuracy. A model with severe multicollinearity can achieve excellent train and test R² if the correlated features together explain the target variance well. The key reference question (Q1 in test.md) specifically states this: "Multicollinearity can coexist with high held-out performance." Option B contradicts this foundational concept.

- **Option C (Correct):** The correct workflow for hyperparameter tuning is: (1) set aside a test set before any modeling; (2) use cross-validation on training data only to select alpha; (3) refit the final model on all training data with the best alpha; (4) evaluate once on the never-seen test set. Using the test set during hyperparameter selection constitutes data leakage and produces optimistically biased performance estimates.

- **Option D (Correct):** This is the fundamental use case guidance: Lasso (L1) produces sparse solutions by zeroing out irrelevant features — ideal when only a small subset of features is truly predictive (sparse true model). Ridge (L2) retains all features with shrunken coefficients — ideal when many correlated features each contribute meaningful (even if small) predictive information and discarding any would lose signal (dense true model). Neither is universally better; the choice depends on the true sparsity of the underlying relationship.

</details>
