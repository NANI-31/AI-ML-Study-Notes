# Regularization Objective Test — Set 02

---

### 11. Question

A data scientist is building a model to predict hospital readmission rates. She has two features — `previous_admissions` and `comorbidity_index` — with a Pearson correlation of 0.93. A colleague says: "Just ignore the correlation; it won't affect your results."

Which response is most accurate?

- [ ] A) The colleague is correct because Pearson correlation only measures linear relationships, and the model captures non-linear ones too.
- [ ] B) High correlation between predictors will inflate R² on the training set, but it has no effect on coefficients.
- [ ] C) The high correlation is a warning sign of multicollinearity that can destabilize individual coefficient estimates, even if overall predictive accuracy remains acceptable.
- [ ] D) A correlation of 0.93 confirms that `comorbidity_index` causes readmissions, so it should be the sole predictor.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The statement that "correlation only measures linear relationships" is true, but it does not justify ignoring a 0.93 correlation. High linear correlation directly triggers multicollinearity, which destabilizes OLS coefficient estimates regardless of any non-linear aspects of the relationship.

- **Option B (Incorrect):** Multicollinearity does not inflate R² on training data; R² is determined by how well the features collectively predict the target. What multicollinearity does affect is the reliability of individual coefficients, making them highly sensitive to which samples happen to be in the training set.

- **Option C (Correct):** A correlation of 0.93 far exceeds common thresholds (~0.7) for concerning multicollinearity. While the model may still predict overall outcomes reasonably well, the individual coefficients for the two correlated features become unreliable — they can swing dramatically across resamples, making interpretation and policy inference dangerous.

- **Option D (Incorrect):** Correlation is not causation. A high Pearson correlation between two predictors tells us they move together, but it says nothing about whether either causes readmissions. Causal claims require experimental design or rigorous causal inference methods, not correlation coefficients.

</details>

---

### 12. Question

A researcher has three observations: actuals = `[10, 20, 30]` and predictions = `[10, 20, 30]`. The target mean is `ȳ = 20`.

What is R² for this model?

- [ ] A) R² = 0.0
- [ ] B) R² = 0.5
- [ ] C) R² = 0.8
- [ ] D) R² = 1.0

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** R² = 0 means the model predicts no better than the baseline (the mean). Here, predictions are perfect — every actual value is predicted exactly — so SSE = 0 and R² = 1.0, not 0.0.

- **Option B (Incorrect):** R² = 0.5 would indicate that SSE/SST = 0.5, meaning half the variance is unexplained. With perfect predictions (SSE = 0), this is not the case.

- **Option C (Incorrect):** R² = 0.8 would indicate SSE = 0.2 × SST. With predictions identical to actuals, SSE = (10−10)² + (20−20)² + (30−30)² = 0. Any SSE > 0 is needed to produce R² < 1.0.

- **Option D (Correct):** SSE = (10−10)² + (20−20)² + (30−30)² = 0 + 0 + 0 = 0. SST = (10−20)² + (20−20)² + (30−20)² = 100 + 0 + 100 = 200. R² = 1 − 0/200 = 1.0. Perfect predictions yield R² = 1.0 by definition.

</details>

---

### 13. Question

A product manager hears that Ridge regression can be used to handle multicollinearity. She asks: "Does Ridge actually remove the correlation between features from the data?"

Which of the following is the most accurate answer?

- [ ] A) Yes, Ridge identifies and removes correlated features from the feature matrix before fitting.
- [ ] B) Yes, Ridge transforms the feature matrix using an orthogonalization step to eliminate correlations.
- [ ] C) No, Ridge only penalizes coefficient magnitudes; the correlation structure in the data remains unchanged.
- [ ] D) No, Ridge eliminates correlations indirectly by setting one of each correlated pair to zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Ridge regression does not remove or filter features from the input matrix. It applies a penalty to the coefficient magnitudes during fitting but the feature matrix X remains identical to what is passed in.

- **Option B (Incorrect):** Ridge does not perform an orthogonalization preprocessing step. Mathematically, the Ridge solution modifies the normal equations by adding αI to XᵀX before inversion, but this is done implicitly during optimization — not as an explicit data transformation.

- **Option C (Correct):** Ridge regularization addresses the consequences of multicollinearity (unstable, inflated coefficient estimates) by shrinking coefficients toward zero. However, the underlying correlations between features in the dataset X are completely unaffected — the same features remain, with the same relationships. Ridge changes the model parameters, not the data.

- **Option D (Incorrect):** Setting one of each correlated pair to zero is the behavior of Lasso (L1), not Ridge (L2). Ridge retains all features with non-zero shrunken weights. Neither method eliminates the actual correlation between features in the data.

</details>

---

### 14. Question

A junior ML engineer writes the following code to apply Ridge regression:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```

A colleague says: "You must scale features before using Ridge, or the result is meaningless." Is the colleague correct?

- [ ] A) No, Ridge regression automatically scales features internally before applying the penalty.
- [ ] B) Yes, without scaling, features on larger scales receive proportionally smaller penalties, causing the regularization to unfairly penalize features with smaller numeric scales.
- [ ] C) No, scikit-learn's Ridge implementation adds a pre-scaling step by default.
- [ ] D) Yes, but only because `model.score()` requires scaled input to compute R² correctly.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** scikit-learn's Ridge does not internally scale features. It applies the L2 penalty directly to the raw coefficients, which are sensitive to the scale of each feature. Features with large numeric ranges dominate the fit while suffering less relative shrinkage, distorting the regularization.

- **Option B (Correct):** The Ridge penalty `α × Σβⱼ²` penalizes the squared coefficient values. A feature measured in thousands (e.g., house price in dollars) will naturally produce a smaller coefficient than the same information measured in tens, meaning it receives less relative penalty. Without standardizing features to zero mean and unit variance, the penalty is applied unequally across features, undermining the fairness and effectiveness of regularization.

- **Option C (Incorrect):** scikit-learn's Ridge class does not perform automatic feature scaling. Scaling must be done explicitly using `StandardScaler` (or another scaler) before fitting, typically via a `Pipeline`.

- **Option D (Incorrect):** `model.score()` computes R² = 1 − SSE/SST regardless of whether features are scaled. The issue with unscaled features is not that `score()` breaks — it is that the penalty term in the Ridge objective is applied inequitably, leading to suboptimal coefficient shrinkage.

</details>

---

### 15. Question

A Pearson correlation coefficient of `r = −0.88` is computed between two features in a dataset. Which statement correctly interprets this value?

- [ ] A) The two features have a strong positive linear relationship, suggesting multicollinearity.
- [ ] B) The absolute value of 0.88 indicates a strong linear relationship; the negative sign means they move in opposite directions, and this can still cause multicollinearity.
- [ ] C) A negative correlation means the two features are completely independent and multicollinearity is not a concern.
- [ ] D) An r of −0.88 is outside the valid range of Pearson correlation and indicates a calculation error.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The sign of r = −0.88 is negative, indicating the two features move in opposite directions (as one increases, the other decreases). This is a strong negative linear relationship, not a positive one.

- **Option B (Correct):** The Pearson correlation range is [−1, +1]. The absolute value |r| = 0.88 indicates strong linear dependence between the two features. The negative sign tells us the relationship is inverse (negative slope), but the strength of the association is just as high as a positive 0.88 correlation. When evaluating multicollinearity, it is the absolute value |r| that matters — a high |r| (whether positive or negative) signals potential multicollinearity.

- **Option C (Incorrect):** Independence would require r ≈ 0. A value of r = −0.88 indicates a very strong linear relationship, which is precisely the type of collinearity that can destabilize OLS coefficient estimates.

- **Option D (Incorrect):** Pearson correlation is defined for the full range [−1, +1]. A value of −0.88 is well within this valid range and indicates a strong negative linear relationship. There is no calculation error.

</details>

---

### 16. Question

A model predicts housing prices. On the training set, R² = 0.96. On the test set, R² = 0.93. A team member says: "We should still apply Lasso because R² near 1.0 guarantees our coefficients are causally correct."

Which statement about this claim is accurate?

- [ ] A) The claim is correct because high R² on both sets confirms that the model has learned true causal relationships.
- [ ] B) The claim is correct because Lasso on an already-good model cannot hurt performance.
- [ ] C) The claim is incorrect: high predictive R² is not evidence that coefficients represent causal effects; observational correlations can produce high R² without capturing true causal mechanisms.
- [ ] D) The claim is incorrect because R² = 0.96 on training data proves the model is overfitting severely.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Predictive accuracy (R²) measures how well a model fits data but cannot establish causality. A spurious variable that happens to correlate with the target can produce very high R² while having zero causal relevance. Causal inference requires controlled experiments or specialized methods (instrumental variables, difference-in-differences, etc.) beyond standard regression.

- **Option B (Incorrect):** Applying Lasso to an already well-fitting model can hurt performance if the alpha is not carefully tuned. If Lasso drives important features to zero, test R² can drop. "Cannot hurt" is factually wrong.

- **Option C (Correct):** High R² indicates strong predictive correlation between features and the target in the observed data. However, observational data contains confounders, selection biases, and spurious correlations that can inflate R² without reflecting true causal pathways. Policy decisions based on observational regression coefficients require causal validation, not just high R².

- **Option D (Incorrect):** A training R² of 0.96 with a test R² of 0.93 shows a small train-test gap (3 percentage points), which is consistent with mild generalization — not severe overfitting. Severe overfitting typically produces a large gap (e.g., training R² ≈ 1.0, test R² ≈ 0.3).

</details>

---

### 17. Question

An engineer tunes Lasso with `alpha = 0.001`. Later she increases it to `alpha = 500`. Compared to the original setting, what should she expect to observe in the fitted model?

- [ ] A) More non-zero coefficients and a lower training bias.
- [ ] B) Fewer non-zero coefficients, lower model complexity, and potentially worse test R² if key features are zeroed out.
- [ ] C) Identical coefficient values because Lasso coefficients do not depend on alpha.
- [ ] D) Higher training R² and higher test R² because stronger regularization always improves both.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Increasing alpha strengthens the L1 penalty, driving more coefficients toward (and eventually to) zero. This reduces the number of non-zero coefficients and increases training bias (the model is less free to fit the training data). More non-zero coefficients and lower bias are outcomes of decreasing alpha, not increasing it.

- **Option B (Correct):** A much larger alpha (500 vs. 0.001) amplifies the Lasso penalty aggressively, setting more and more coefficients to exactly zero. This simplifies the model dramatically. If the zeroed features contained genuine signal, the test R² will decline because the model can no longer leverage that information. This is the bias-variance tradeoff in action: high alpha → high bias, potentially lower test performance.

- **Option C (Incorrect):** Lasso coefficients are strongly dependent on alpha. Alpha controls the weight given to the penalty relative to the fit quality. Setting alpha = 500 versus alpha = 0.001 produces fundamentally different coefficient vectors.

- **Option D (Incorrect):** Stronger regularization does not guarantee higher R² on either the training or test set. In fact, very high alpha tends to produce a constant-prediction model (near the intercept/mean), which generally has lower R² on both sets than a well-tuned alpha.

</details>

---

### 18. Question

A data scientist uses the following code to compare models:

```python
from sklearn.linear_model import LinearRegression, Ridge

ols = LinearRegression()
ridge = Ridge(alpha=0.1)

ols.fit(X_train, y_train)
ridge.fit(X_train, y_train)

print("OLS train R²:", ols.score(X_train, y_train))
print("Ridge train R²:", ridge.score(X_train, y_train))
```

On the same training data, which model is expected to have a higher training R²?

- [ ] A) Ridge, because regularization always improves training performance.
- [ ] B) They will always be identical because both are linear models.
- [ ] C) OLS, because regularization introduces a penalty that prevents OLS from minimizing training SSE to its unconstrained minimum.
- [ ] D) It depends on which model has more features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Regularization trades training performance for better generalization. By adding a penalty to the loss, Ridge prevents the model from fitting the training data as closely as OLS does. Ridge training R² is generally lower than or equal to OLS training R².

- **Option B (Incorrect):** While both are linear models in their predictions, they optimize different objectives. OLS minimizes pure SSE; Ridge minimizes SSE + α × Σβⱼ². The added penalty causes Ridge to reach a different (and generally higher-SSE) solution on the training set.

- **Option C (Correct):** OLS minimizes SSE without any constraint, achieving the globally lowest possible SSE on the training data for a linear model. Ridge adds a constraint (the L2 penalty), which prevents the coefficients from reaching the unconstrained OLS minimum. Therefore, Ridge's training SSE ≥ OLS training SSE, and Ridge training R² ≤ OLS training R².

- **Option D (Incorrect):** The comparison between OLS and Ridge training R² is independent of the number of features. The conclusion (OLS has higher or equal training R²) holds regardless of the feature count because it is determined by the presence or absence of the penalty term in the objective function.

</details>

---

### 19. Question

A baseline model always predicts the mean of the training targets (`ȳ`) for every input. What R² score does this model achieve on the training set?

- [ ] A) R² = 1.0, because it is the simplest possible model.
- [ ] B) R² = 0.5, because it explains half the variance by using the mean.
- [ ] C) R² = 0.0, because its predictions produce SSE equal to SST.
- [ ] D) R² can be any value between 0 and 1 depending on how spread out the target values are.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A model that always predicts the mean is the least informative possible baseline — it uses no features at all. R² = 1.0 would mean perfect predictions with zero residuals, which is clearly not the case when every prediction is the mean.

- **Option B (Incorrect):** The mean model does not explain half the variance. It literally uses no features to deviate from the mean, so its SSE equals exactly SST. There is no partial explanation.

- **Option C (Correct):** When the model predicts ȳ for every observation, each residual is (yᵢ − ȳ), making SSE = Σ(yᵢ − ȳ)² = SST. Therefore R² = 1 − SSE/SST = 1 − SST/SST = 1 − 1 = 0. The baseline mean model defines the zero point of R².

- **Option D (Incorrect):** The mean model always yields exactly R² = 0 on the training set, regardless of how spread out the target values are. The computation SSE = SST is an algebraic identity when predictions are the training mean.

</details>

---

### 20. Question

A machine learning team is building a text sentiment model with 500 features, most of which are word frequency counts. They suspect fewer than 20 features are actually informative. Which pair of regularization choices best matches this scenario?

- [ ] A) Ridge with small alpha; it will identify and zero out the 480 noise features.
- [ ] B) OLS with feature selection post-hoc; it will be as effective as regularization.
- [ ] C) Lasso with a moderate alpha; it promotes sparsity and can automatically zero out uninformative features.
- [ ] D) Ridge with large alpha; it will zero out all 500 features and start fresh.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Ridge (L2 penalty) does not zero out features. Even with a small alpha, Ridge shrinks all coefficients proportionally but retains non-zero values for all 500 features. It cannot perform feature selection in the way described.

- **Option B (Incorrect):** Plain OLS with 500 features and mostly noise will likely overfit severely. Post-hoc feature selection from an OLS fit on noisy, high-dimensional data is unreliable compared to Lasso's built-in regularized feature selection. Regularization during fitting is statistically superior.

- **Option C (Correct):** Lasso (L1 penalty) is the appropriate choice when a sparse solution is expected — i.e., when only a small fraction of features are truly predictive. With moderate alpha, Lasso sets the coefficients of the ~480 noise features to exactly zero and retains small non-zero weights for the ~20 informative features, achieving both regularization and automatic feature selection simultaneously.

- **Option D (Incorrect):** Ridge with large alpha pushes all coefficients toward zero but never reaches exactly zero. Even at extreme alpha, all 500 features remain in the model with tiny non-zero coefficients. This is not equivalent to "starting fresh" with zero features, and it would severely underfit.

</details>
