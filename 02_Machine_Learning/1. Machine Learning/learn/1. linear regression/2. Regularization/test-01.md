# Regularization Objective Test — Set 01

---

### 1. Question

A junior data scientist is learning about multicollinearity in a sales prediction model. Two features — `ad_spend` and `brand_awareness_score` — move almost in lockstep with each other across all observations.

Which of the following statements about this situation are correct?

- [ ] A) The individual coefficients for `ad_spend` and `brand_awareness_score` will always be uniquely and reliably estimated by OLS.
- [ ] B) Removing one of the two correlated features guarantees that the model's held-out R² will drop significantly.
- [ ] C) The model's overall predictions may remain accurate even though individual coefficients are unstable.
- [ ] D) Multicollinearity primarily harms the model's training loss but leaves test-set predictions completely unaffected.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** When two features are highly correlated, OLS cannot reliably separate their individual contributions. The system of equations becomes nearly degenerate, and many different combinations of coefficients produce nearly identical predictions, making individual coefficient estimates highly unstable rather than unique.

- **Option B (Incorrect):** Removing a highly correlated feature does not guarantee a drop in held-out R². Because the two features carry largely redundant information, the remaining feature can compensate and maintain similar predictive power. The impact on R² depends on the specific dataset.

- **Option C (Correct):** Multicollinearity primarily damages the stability and interpretability of individual coefficients rather than the model's overall predictive accuracy. The collinear features together still capture variance in the target, so held-out R² can remain strong even when coefficients swing wildly across resamples.

- **Option D (Incorrect):** This reverses the actual problem. Multicollinearity affects coefficient estimates (a statistical inference concern) far more than it affects test-set predictions. In fact, training loss may look perfectly fine while coefficient instability is severe.

</details>

---

### 2. Question

A machine learning engineer is comparing two loss functions for linear regression. Model A adds `α × Σ|βⱼ|` to the OLS loss. Model B adds `α × Σβⱼ²` to the OLS loss.

Which statements are correct?

- [ ] A) Model A uses L2 regularization and is known as Ridge regression.
- [ ] B) Model B uses L1 regularization and is known as Lasso regression.
- [ ] C) Model A (L1) has the property of producing sparse solutions where some coefficients become exactly zero.
- [ ] D) Model B (L2) routinely drives coefficients to exactly zero when alpha is large enough.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Model A adds the sum of absolute weight values (`Σ|βⱼ|`), which is the L1 penalty. A model using L1 regularization is called Lasso (Least Absolute Shrinkage and Selection Operator), not Ridge. Ridge uses the squared penalty (L2).

- **Option B (Incorrect):** Model B adds the sum of squared weight values (`Σβⱼ²`), which is the L2 penalty. A model using L2 regularization is called Ridge regression, not Lasso.

- **Option C (Correct):** The L1 penalty's geometric shape (a diamond in 2D) creates corner points on the coordinate axes. When the OLS contours intersect the constraint region at these corners, one or more coefficients are driven to exactly zero, producing sparse models — a property unique to L1 regularization.

- **Option D (Incorrect):** The L2 penalty's circular/elliptical constraint region has no corners on the axes. The OLS loss function touches this region at a point that is almost never exactly on an axis, meaning Ridge regression shrinks coefficients toward zero but rarely (practically never) sets them to exactly zero.

</details>

---

### 3. Question

A data analyst computes R² using the formula `R² = 1 − SSE/SST` for a test set. She obtains SSE = 120 and SST = 100.

What is R², and what does it mean?

- [ ] A) R² = 0.2, indicating the model explains 20% of the variance.
- [ ] B) R² = 1.2, indicating the model explains more than 100% of the variance.
- [ ] C) R² = −0.2, indicating the model performs worse than a naive baseline that always predicts the target mean.
- [ ] D) R² cannot be computed because SSE > SST is a mathematical impossibility.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** R² = 1 − 120/100 = 1 − 1.2 = −0.2, not +0.2. Computing 120/100 = 1.2 and subtracting from 1 yields a negative result, which this option incorrectly states as positive.

- **Option B (Incorrect):** 1.2 is the ratio SSE/SST, not R². R² = 1 − SSE/SST = 1 − 1.2 = −0.2. An R² value above 1.0 is not achievable; this is a sign the formula was applied incorrectly.

- **Option C (Correct):** R² = 1 − 120/100 = −0.2. A negative R² means the model's total prediction error (SSE = 120) exceeds the variance the baseline mean model would produce (SST = 100). The model performs strictly worse than simply predicting the mean of the target for every observation.

- **Option D (Incorrect):** SSE can absolutely exceed SST on held-out or test data. SST is fixed as the variance around the training mean, while SSE depends on how well the model's predictions match the test targets. When predictions are systematically off, SSE > SST and R² < 0 is a mathematically valid and practically meaningful result.

</details>

---

### 4. Question

An ML team is deciding between Ridge and Lasso for a customer churn model that has 200 features, most of which are suspected to be noise. The business team explicitly requires a model that outputs a short, interpretable list of the most important features.

Which regularization method best matches this requirement, and why?

- [ ] A) Ridge, because it stabilizes coefficients among correlated features by distributing weight evenly.
- [ ] B) Ridge, because it can set the coefficients of noise features to exactly zero.
- [ ] C) Lasso, because it tends to produce sparse solutions by driving noise feature coefficients to exactly zero.
- [ ] D) Plain OLS, because no penalty is needed when the dataset has many features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Ridge's property of distributing weight evenly among correlated features means it retains all 200 features in the model with non-zero (though shrunken) coefficients. This does not satisfy the requirement for a short, interpretable feature list.

- **Option B (Incorrect):** Ridge regression (L2) cannot set coefficients to exactly zero due to the shape of its penalty contour. The L2 constraint region is a smooth sphere with no corners on the coordinate axes, so coefficients are shrunk but never eliminated.

- **Option C (Correct):** Lasso (L1 penalty) produces sparse solutions by setting irrelevant feature coefficients exactly to zero. With 200 features, most of which are noise, Lasso naturally performs automated feature selection, leaving only the truly predictive features with non-zero weights — directly satisfying the business requirement.

- **Option D (Incorrect):** With 200 features and most being noise, plain OLS would overfit severely by fitting noise rather than signal. Without any regularization, OLS has no mechanism to suppress irrelevant features, leading to poor generalization.

</details>

---

### 5. Question

A researcher explains R² to a colleague in a business meeting. Which of the following properties of R² is mathematically correct?

- [ ] A) R² always lies between 0 and 1 for any trained model evaluated on any dataset.
- [ ] B) An R² of 0 always means the model has zero predictive power and its coefficients are all zero.
- [ ] C) R² is dimensionless and therefore comparable across models predicting targets in different units.
- [ ] D) R² is equivalent to MSE scaled by the number of observations.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** R² is bounded between 0 and 1 only on the training set when OLS is used (because OLS minimizes SSE, guaranteeing SSE ≤ SST on training data). On a held-out test set, SSE can exceed SST if the model generalizes poorly, producing R² < 0. The [0, 1] bound is not universal.

- **Option B (Incorrect):** R² = 0 means the model explains no more variance than the baseline model that always predicts the mean — not necessarily that all coefficients are zero. A model could have non-zero coefficients yet still produce an R² of 0 on a particular dataset if its predictions happen to match the total-variance baseline exactly.

- **Option C (Correct):** R² = 1 − SSE/SST is a ratio of two variance terms, making it unitless and scale-independent. This means an R² of 0.85 for a salary model and an R² of 0.85 for a temperature model are directly comparable, unlike MSE or MAE, which carry the squared or absolute units of the target variable.

- **Option D (Incorrect):** MSE = SSE / n measures the average squared error per observation. R² = 1 − SSE/SST normalizes by the target variance, not by n. These are fundamentally different normalizations serving different purposes; R² is not a scaled version of MSE.

</details>

---

### 6. Question

For four observations, actual target values are `[3, 5, 7, 9]` and the model predicts `[4, 4, 8, 8]`. The mean of the actuals is `ȳ = 6`.

Calculate R² for this model.

- [ ] A) R² = 0.90
- [ ] B) R² = 0.80
- [ ] C) R² = 0.70
- [ ] D) R² = 0.60

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** SSE = (3−4)² + (5−4)² + (7−8)² + (9−8)² = 1 + 1 + 1 + 1 = 4. SST = (3−6)² + (5−6)² + (7−6)² + (9−6)² = 9 + 1 + 1 + 9 = 20. R² = 1 − 4/20 = 1 − 0.2 = 0.80. R² = 0.90 would require SSE = 2, but the computed SSE is 4.

- **Option B (Correct):** SSE = (3−4)² + (5−4)² + (7−8)² + (9−8)² = 1 + 1 + 1 + 1 = 4. SST = (3−6)² + (5−6)² + (7−6)² + (9−6)² = 9 + 1 + 1 + 9 = 20. R² = 1 − 4/20 = 1 − 0.20 = 0.80. The model explains 80% of the target variance.

- **Option C (Incorrect):** R² = 0.70 would require SSE/SST = 0.30, meaning SSE = 6. The actual SSE is 4, giving SSE/SST = 0.20 and R² = 0.80.

- **Option D (Incorrect):** R² = 0.60 would require SSE/SST = 0.40, meaning SSE = 8. The computed SSE is only 4, so this answer overstates the residual error by a factor of 2.

</details>

---

### 7. Question

A data engineer refits a linear regression model on five different bootstrap samples of the same housing dataset. He observes that the coefficient for `lot_size` varies between −230 and +180 across the five fits, while the coefficient for `garage_spaces` stays close to +45 in every fit.

What does this pattern most likely indicate?

- [ ] A) `lot_size` has no real relationship with the target price; the sign changes confirm this.
- [ ] B) `garage_spaces` is also unstable but the engineer made an error reading the results.
- [ ] C) `lot_size` is likely collinear with one or more other features, causing high-variance, unreliable coefficient estimates.
- [ ] D) The bootstrap procedure is flawed and cannot be used to assess coefficient stability.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A coefficient swinging from negative to positive across different samples does not mean the feature has no relationship with the target. It often means the opposite — the feature's contribution is tangled up with another correlated feature, making its individual contribution unidentifiable, not absent.

- **Option B (Incorrect):** The fact that `garage_spaces` remains stable while `lot_size` varies strongly suggests the instability is a property of `lot_size`'s relationships in the data, not a reading error. Differential stability across features is a diagnostic signal, not an artifact.

- **Option C (Correct):** Coefficient instability across bootstrap resamples — especially sign reversals — is a classic diagnostic symptom of multicollinearity. When `lot_size` is highly correlated with another feature (e.g., `living_area`), the model arbitrarily reallocates weight between them in each resample, producing widely varying and unreliable individual estimates.

- **Option D (Incorrect):** Bootstrap resampling is a standard, statistically sound technique for assessing the variance and stability of model estimates. It is widely used precisely because it reveals how sensitive parameters are to slight variations in the training data.

</details>

---

### 8. Question

A product analyst is explaining regularization to a stakeholder. She says: "When alpha equals zero in Ridge regression, the model is identical to plain OLS." A skeptical colleague says: "When alpha is very large in Ridge, the model will perfectly learn the true pattern with no noise."

Which statements about these two claims are correct?

- [ ] A) Both claims are correct.
- [ ] B) The first claim is correct; the second claim is incorrect.
- [ ] C) The first claim is incorrect; the second claim is correct.
- [ ] D) Both claims are incorrect.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The second claim is wrong, so both cannot be correct. Very large alpha does not help the model learn the true pattern — it destroys the model's ability to learn any pattern at all.

- **Option B (Correct):** When alpha = 0, the Ridge penalty term `α × Σβⱼ²` disappears entirely, leaving the pure OLS objective. Ridge with alpha = 0 is mathematically identical to OLS. This first claim is correct. The second claim is incorrect: a very large alpha forces all coefficients toward zero regardless of the true signal, producing a model that predicts near the intercept (mean) for all inputs — a classic underfitting scenario, not a better-learning one.

- **Option C (Incorrect):** The first claim is mathematically correct. When alpha = 0, no penalty is added to the loss, so the Ridge objective reduces to the OLS objective exactly, producing the same solution.

- **Option D (Incorrect):** The first claim is correct (alpha = 0 recovers OLS). Only the second claim is incorrect. Declaring both incorrect misidentifies the first claim as wrong.

</details>

---

### 9. Question

A model trained on a real estate dataset produces the following coefficient vector after fitting: `[0, 0, 0, 1.2, 0, 0, 3.4, 0]`. Eight features were provided during training.

Which regularization method most likely produced this output, and what property does it demonstrate?

- [ ] A) Ridge regression with a moderate alpha; it demonstrates that Ridge distributes weight evenly.
- [ ] B) Plain OLS; it demonstrates that OLS performs automatic feature selection.
- [ ] C) Lasso regression with a sufficiently large alpha; it demonstrates sparsity through exact zero coefficients.
- [ ] D) Ridge regression with alpha = 0; it demonstrates that all features always receive non-zero weights.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Ridge regression (L2) distributes weight across correlated features but does not drive coefficients to exact zeros. A Ridge model with 8 features would retain all 8 features with non-zero (though possibly small) coefficients. The presence of six exact zero coefficients is inconsistent with Ridge behavior.

- **Option B (Incorrect):** Plain OLS has no mechanism for automatic feature selection. When features are included in the design matrix, OLS assigns non-zero weights to all of them unless the matrix is rank-deficient. Plain OLS would not produce six exact zero coefficients for an unconstrained fit with 8 provided features.

- **Option C (Correct):** Lasso (L1 penalty) is specifically known for producing sparse solutions. Its geometric constraint — a diamond-shaped region in coefficient space — causes the OLS loss contours to first touch the constraint region at a corner where one or more coefficients are exactly zero. With a sufficiently large alpha, Lasso drives most coefficients to zero, retaining only the most predictive features.

- **Option D (Incorrect):** Ridge with alpha = 0 is identical to plain OLS and assigns non-zero weights to all features in the model. It would not produce the sparse solution shown.

</details>

---

### 10. Question

An engineer sees that a colleague's Ridge regression model has extremely high training R² (0.997) but moderate test R² (0.71), and the test R² drops further to 0.55 when the alpha is increased from 1.0 to 1000.0.

What is the most accurate diagnosis of what is happening as alpha increases from 1.0 to 1000.0?

- [ ] A) The model is moving from underfitting to a well-tuned state.
- [ ] B) The increase in alpha is correcting overfitting and the test R² drop confirms final convergence.
- [ ] C) The model is moving from a mildly overfit state toward increasing underfitting as the regularization penalty becomes excessively strong.
- [ ] D) The test R² drop proves that Ridge is inferior to OLS for this dataset.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The model starts at a high-alpha side of the spectrum and is actually moving further into underfitting, not toward a well-tuned state. The model was already working reasonably (test R² = 0.71) before the alpha increase, so "from underfitting to well-tuned" is backwards.

- **Option B (Incorrect):** While increasing alpha does penalize complexity, a drop from test R² 0.71 to 0.55 is not "convergence" — it is degradation in held-out performance. True correction of overfitting would be shown by an increase in test R², not a decrease. An excessive alpha eventually causes the coefficients to collapse toward zero, losing the signal.

- **Option C (Correct):** The gap between training R² (0.997) and test R² (0.71) at alpha = 1.0 suggests mild overfitting. Increasing alpha to 1000 shrinks the coefficients so aggressively that the model loses its ability to capture true signal, driving test R² down further to 0.55 — a sign of increasing underfitting. The ideal alpha lies between these extremes.

- **Option D (Incorrect):** Choosing an excessively large alpha and observing degraded test performance does not make Ridge generally inferior to OLS. It demonstrates that hyperparameter selection is critical. At an appropriate alpha, Ridge typically outperforms OLS when multicollinearity is present.

</details>
