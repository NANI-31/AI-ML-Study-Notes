# Multicollinearity Objective Test — Set 02

---

### 1. Question

A data analyst fits a linear regression model on a dataset of 1,000 observations. When adding a single new observation to the dataset, the estimated coefficient for `feature_A` swings from $+4.82$ to $-2.15$, while the model's overall $R^2$ remains virtually unchanged at $0.84$.

What does this extreme coefficient volatility indicate?

- [ ] A) The model has converged to a global minimum with high numerical precision.
- [ ] B) Severe multicollinearity is causing high variance in the parameter estimates.
- [ ] C) The learning rate was set too high during ordinary least squares estimation.
- [ ] D) The residuals are perfectly normally distributed with zero autocorrelation.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Extreme sensitivity to minor sample variations is the hallmark of instability and ill-conditioning, not high precision.
- **Option B (Correct):** When features are collinear, the loss surface contains long, flat, shallow valleys where many different coefficient combinations yield almost identical residual sum of squares. Adding just one observation tilts the surface slightly, sending the optimal point sliding drastically along the valley floor.
- **Option C (Incorrect):** Standard OLS solves closed-form normal equations or uses SVD decomposition; it does not typically rely on an iterative learning rate unless fitted via SGDRegressor.
- **Option D (Incorrect):** Residual normality does not prevent coefficient instability caused by collinear design columns.

</details>

---

### 2. Question

An economist estimates a Cobb-Douglas production function using linear regression with inputs:
* Capital ($K$)
* Labor ($L$)
* Energy consumption ($E$)

The estimated coefficient for Energy ($E$) is $-0.45$ ($p = 0.02$). Domain experts assert that adding energy input cannot reduce output. A correlation check reveals that $K$, $L$, and $E$ all have mutual correlations greater than $0.92$.

Why did Energy receive a counter-intuitive negative coefficient?

- [ ] A) The model suffers from omitted variable bias.
- [ ] B) Multicollinearity caused the model to compensate for overlapping variance by flipping the sign of one collinear predictor.
- [ ] C) The OLS estimator is mathematically biased whenever $n > 100$.
- [ ] D) Energy consumption is non-stationary and must be differenced.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Omitted variable bias occurs when an important predictor is left *out* of the model; here, all relevant inputs are present, but their heavy mutual overlap distorts their weights.
- **Option B (Correct):** When multiple predictors share common variance, OLS coefficients represent the *partial* effect of each variable after controlling for all others. The model over-credits one collinear variable (e.g., assigning a larger positive weight to Capital) and balances the equation by assigning a negative weight to Energy.
- **Option C (Incorrect):** The Gauss-Markov theorem proves that OLS is unbiased under standard assumptions, even under multicollinearity; high variance, not bias, causes individual sample estimates to land far from the true parameter value.
- **Option D (Incorrect):** While non-stationarity is relevant in time-series, sign flipping among mutually correlated inputs in cross-sectional or production functions is directly attributable to collinearity.

</details>

---

### 3. Question

What is the impact of multicollinearity on the confidence intervals of regression coefficients?

- [ ] A) Confidence intervals become excessively narrow, creating a false sense of certainty.
- [ ] B) Confidence intervals become excessively wide, increasing the probability of Type II errors (failing to reject false null hypotheses).
- [ ] C) Confidence intervals shrink to a single point because standard errors approach zero.
- [ ] D) Confidence intervals are unaffected because standard errors depend only on sample size $n$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Narrow confidence intervals occur when standard errors are small. Multicollinearity inflates standard errors, which widens confidence intervals.
- **Option B (Correct):** Because the variance of estimated coefficients is inflated by the factor VIF, the standard error $\text{SE}(\hat{\beta}_j)$ grows large. Wide confidence intervals often encompass zero, causing the t-test to fail to reject $H_0: \beta_j = 0$ even when the feature is genuinely important (Type II error).
- **Option C (Incorrect):** Standard errors approach infinity (not zero) as collinearity approaches perfection.
- **Option D (Incorrect):** Standard errors depend on residual variance $\sigma^2$, sample size $n$, feature variance, and crucially, the degree of collinearity with other features ($1 - R_j^2$).

</details>

---

### 4. Question

Under the Gauss-Markov theorem, is the OLS estimator still Best Linear Unbiased Estimator (BLUE) in the presence of severe (but imperfect) multicollinearity?

- [ ] A) No, because multicollinearity introduces severe systemic bias into the parameter estimates.
- [ ] B) No, because multicollinearity renders the estimator non-linear.
- [ ] C) Yes, OLS remains unbiased and retains minimum variance among all linear unbiased estimators, but that minimum variance may still be unacceptably large.
- [ ] D) Yes, but only if the sample size is smaller than the number of features ($n < p$).

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Imperfect multicollinearity does not violate any Gauss-Markov assumption. Expected values $E[\hat{\beta}]$ still equal the true parameter $\beta$, so OLS remains strictly unbiased.
- **Option B (Incorrect):** The OLS estimator remains a linear function of the dependent variable $y$ regardless of feature correlation.
- **Option C (Correct):** As long as $X^T X$ is invertible (no perfect collinearity), OLS is still technically BLUE. However, being "best" (having lowest variance among unbiased estimators) is cold comfort when the absolute variance of all unbiased estimators is enormous. This motivates biased estimators like Ridge regression, which trade a little bias for a massive reduction in variance.
- **Option D (Incorrect):** When $n < p$, $X^T X$ is strictly non-invertible, and standard OLS cannot even be computed without regularization or dimensionality reduction.

</details>

---

### 5. Question

A data scientist examines the summary table of an OLS regression model fitted using `statsmodels`:

```
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             14.201     12.850      1.105      0.272     -11.200      39.602
x2            -11.450     11.900     -0.962      0.338     -35.000      12.100
==============================================================================
F-statistic: 84.52 (p-val: 1.2e-15), R-squared: 0.812
```

What key pattern in this table indicates severe multicollinearity between `x1` and `x2`?

- [ ] A) The standard errors are very small relative to the coefficients.
- [ ] B) The t-statistics are large while the F-statistic is near zero.
- [ ] C) High $R^2$ and highly significant F-statistic alongside high p-values ($> 0.05$) and wide confidence intervals for both individual features.
- [ ] D) The sum of the coefficients equals approximately zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Standard errors here are quite large ($12.85$ and $11.90$) relative to the coefficient magnitudes, which deflates the t-statistic.
- **Option B (Incorrect):** The F-statistic is $84.52$ (huge), while the individual t-statistics are tiny ($1.105$ and $-0.962$).
- **Option C (Correct):** The overall model is clearly explaining variance ($R^2 = 0.812, F = 84.52, p \approx 0$). Yet neither `x1` nor `x2` achieves statistical significance ($p = 0.272$ and $0.338$), and their confidence intervals span wide ranges from negative to positive. This is the unmistakable diagnostic pattern of multicollinearity.
- **Option D (Incorrect):** The coefficients offsetting each other is a symptom of collinearity, but the diagnostic confirmation comes from the joint significance vs. individual insignificance.

</details>

---

### 6. Question

What happens to the condition number of the design matrix $X$ when strong multicollinearity is present?

- [ ] A) It approaches zero.
- [ ] B) It approaches 1.0.
- [ ] C) It becomes extremely large (typically exceeding 30 or 100).
- [ ] D) It becomes negative.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The condition number is the ratio of the largest to smallest singular value ($\kappa = \sigma_{\max} / \sigma_{\min}$). Because singular values are non-negative and $\sigma_{\max} \ge \sigma_{\min}$, the condition number is always $\ge 1.0$; it can never approach zero.
- **Option B (Incorrect):** A condition number of 1.0 represents a perfectly orthogonal, perfectly conditioned matrix where all features are completely uncorrelated.
- **Option C (Correct):** In the presence of multicollinearity, the smallest singular value $\sigma_{\min}$ approaches zero, causing the condition number $\kappa$ to blow up. Condition numbers above 30 indicate moderate to strong ill-conditioning, and values above 100 signal severe numerical instability.
- **Option D (Incorrect):** Condition numbers are defined as ratios of singular values (or norms), which are strictly non-negative.

</details>

---

### 7. Question

A machine learning engineer evaluates a linear regression model on both a training set and a holdout test set drawn from the same underlying distribution. Strong multicollinearity exists among the training features.

Assuming the joint distribution between features does NOT shift between train and test, what is the expected performance on the test set?

- [ ] A) The test $R^2$ will always be negative because collinear models cannot make predictions.
- [ ] B) The test predictions may remain reasonably accurate and achieve an $R^2$ close to the training $R^2$.
- [ ] C) The model will crash during inference with an ill-conditioned matrix error.
- [ ] D) The test residuals will be exactly zero for all test samples.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Multicollinearity does not prevent linear regression from predicting the response variable; predictions can be very accurate on data with the same correlation structure.
- **Option B (Correct):** While individual coefficients are unstable, their linear combination $\hat{y} = X\hat{\beta}$ is well-constrained along the subspace where data exists. If new test observations maintain the same collinear relationships, predictions will be stable and predictive metrics like $R^2$ or RMSE will remain solid.
- **Option C (Incorrect):** Matrix inversion occurs only during training. Inference requires only a dot product ($\hat{y} = X_{\text{test}} \hat{\beta}$), which never crashes due to matrix conditioning.
- **Option D (Incorrect):** Test residuals will almost never be zero in real-world noisy data.

</details>

---

### 8. Question

In an inference scenario, what specific condition WILL cause a linear model trained with multicollinear features to produce catastrophic prediction errors?

- [ ] A) Evaluating the model on a test set that is twice as large as the training set.
- [ ] B) When the correlation structure between the collinear features breaks down in production (distribution shift / covariance shift).
- [ ] C) Standardizing the target variable before computing predictions.
- [ ] D) Converting continuous features to 64-bit floating point numbers.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Sample size of the test set has no adverse effect on model stability.
- **Option B (Correct):** Because the model assigned large offsetting coefficients (e.g., $+100$ to Feature 1 and $-95$ to Feature 2), predictions rely entirely on Feature 1 and Feature 2 canceling each other out. If a new production sample has a high Feature 1 but a low Feature 2 (breaking the historical correlation), the large weights will not cancel, producing massive, nonsensical prediction errors.
- **Option C (Incorrect):** Standardizing the target is a standard rescaling step and does not induce prediction breakdown.
- **Option D (Incorrect):** Higher numerical precision (64-bit float) actually improves arithmetic stability during linear operations.

</details>

---

### 9. Question

Why does multicollinearity make feature importance interpretation misleading in multiple linear regression?

- [ ] A) The magnitude and sign of a feature's coefficient depend heavily on which other correlated features happen to be included in the model.
- [ ] B) Scikit-learn randomly shuffles coefficient values when features have correlation $> 0.70$.
- [ ] C) P-values under multicollinearity are mathematically guaranteed to equal 1.0 for all features.
- [ ] D) R-squared drops to 0 whenever two features have non-zero covariance.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In linear regression, each coefficient represents the isolated effect of a variable holding all other variables constant. When two variables are correlated, the model divides credit between them arbitrarily. Dropping or adding a third correlated variable can cause another variable's coefficient to double, shrink to zero, or invert its sign, making ranking features by coefficient magnitude completely unreliable.
- **Option B (Incorrect):** Scikit-learn executes deterministic mathematical algorithms; it never randomly shuffles coefficients.
- **Option C (Incorrect):** P-values are inflated, but they are computed from the t-distribution and rarely equal exactly 1.0.
- **Option D (Incorrect):** R-squared reflects total explained variance and is typically high when multiple predictive (even collinear) variables are present.

</details>

---

### 10. Question

Which of the following geometric visualizations correctly depicts the loss function surface of a linear regression model suffering from severe multicollinearity between two features?

- [ ] A) A perfectly circular bowl with a sharp, distinct minimum point at the center.
- [ ] B) An extremely elongated, flat, narrow elliptical valley where the bottom resembles a long trough.
- [ ] C) A saddle point where one direction curves upward and the other curves downward.
- [ ] D) A completely flat horizontal plane with zero slope in all directions.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A circular bowl occurs when features are perfectly orthogonal (uncorrelated) and have equal variance, making the optimum well-defined and stable.
- **Option B (Correct):** Strong correlation between features stretches the contours of the Mean Squared Error loss surface into an elongated elliptical trough. Moving along the floor of this valley changes the coefficient values dramatically while changing the overall loss by an imperceptible amount, creating high parameter uncertainty.
- **Option C (Incorrect):** A saddle point has indefinite curvature (eigenvalues of opposite signs). The OLS loss function is convex with a positive semi-definite Hessian matrix, so it never has a saddle point.
- **Option D (Incorrect):** The surface is not flat everywhere; it slopes steeply upwards when moving perpendicular to the collinearity valley.

</details>
