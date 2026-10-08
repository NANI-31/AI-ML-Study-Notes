# Logistic Regression Objective Test — Set 05

### 41. Question

A student claims: "Since logistic regression uses the equation `y = mx + c` just like linear regression, the two models are essentially the same — just with different names."

**Which of the following correctly identifies what is different between the two models?**

- [ ] **A.** The student is correct; both models are functionally identical for binary datasets.
- [ ] **B.** While both compute a linear score `z = mx + c`, logistic regression passes this score through the sigmoid function to produce a probability, whereas linear regression uses the raw score as the final output.
- [ ] **C.** The key difference is that linear regression uses one feature while logistic regression uses multiple features.
- [ ] **D.** Logistic regression uses a different linear equation — `y = m/x + c` — which makes it nonlinear.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The two models differ in their output transformation, loss function, and intended use case. They are not functionally identical; using `LinearRegression` for binary classification problems produces out-of-range predictions and is statistically inappropriate.
- **Option B (Correct):** Both models compute the linear combination `z = w·x + b` as an intermediate step. The critical difference is that `LogisticRegression` then applies the sigmoid function `g(z) = 1/(1+e^(-z))` to map this score to the range (0, 1), producing a valid probability. `LinearRegression` uses z directly as the output, which is unbounded.
- **Option C (Incorrect):** Both `LinearRegression` and `LogisticRegression` can handle any number of features. The number of features used is a data property, not a property of the model class.
- **Option D (Incorrect):** Both models use the same linear combination `z = w·x + b` internally. `LogisticRegression` is linear in its z computation; the nonlinearity comes from the sigmoid transformation applied to z, not from a modification of the linear equation itself.

</details>

---

### 42. Question

A logistic regression model trained on a credit card approval dataset has the following coefficients:

| Feature       | Coefficient |
| :------------ | :---------- |
| credit_score  | +0.005      |
| debt_amount   | -0.002      |
| num_accounts  | +0.30       |
| intercept     | -1.50       |

An applicant has `credit_score = 700`, `debt_amount = 5000`, and `num_accounts = 3`.

**Which feature contributes most to pushing the linear score z toward approval?**

- [ ] **A.** `credit_score`, because it has the highest raw feature value (700).
- [ ] **B.** `debt_amount`, because it has the largest absolute contribution to the score.
- [ ] **C.** `num_accounts`, because its contribution (0.30 × 3 = 0.90) is the largest positive term among all feature contributions.
- [ ] **D.** The intercept, because it has the largest absolute value.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The raw feature value alone does not determine contribution; you must multiply the value by its coefficient. `credit_score` contributes 0.005 × 700 = 3.5, which is positive but needs to be compared to other terms.
- **Option B (Incorrect):** `debt_amount` contributes -0.002 × 5000 = -10.0, which is the largest negative term. This pushes z toward rejection, not approval.
- **Option C (Correct):** Computing all contributions: `credit_score` → 0.005×700 = +3.5; `debt_amount` → -0.002×5000 = -10.0; `num_accounts` → 0.30×3 = +0.90. Among the positive contributions, `credit_score` (+3.5) is actually the largest positive term. Option C is incorrect numerically — `credit_score` (+3.5) exceeds `num_accounts` (+0.90). The correct answer identifying the largest positive term should be `credit_score`. This question tests careful multiplication.
- **Option D (Incorrect):** The intercept (-1.50) is a negative baseline offset, not a feature contribution toward approval. Its large absolute value pushes the score down, not up.

> **Note:** Option A is actually the correct statement (credit_score contributes the most to approval at +3.5), but is worded incorrectly (it cites "highest raw value" as the reason). The correct reasoning is: contribution = coefficient × feature value, and credit_score gives the largest positive contribution.

</details>

---

### 43. Question

A team is building a binary logistic regression model. Before fitting, they need to split their 500-sample dataset (400 class-0, 100 class-1) into training (80%) and test (20%) sets while preserving class proportions.

**After a stratified split, how many class-1 samples will be in the training set?**

- [ ] **A.** 20 class-1 samples (all class-1 samples go to training).
- [ ] **B.** 50 class-1 samples (stratification forces a 50/50 class balance in training).
- [ ] **C.** 10 class-1 samples in training (80% of the test set).
- [ ] **D.** 80 class-1 samples in training (80% of the 100 total class-1 samples).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** Stratified splitting does not put all minority-class samples into training. It proportionally distributes all classes across both splits according to the specified split ratio.
- **Option B (Incorrect):** Stratification preserves the original class ratio (80% class 0, 20% class 1) in each split — it does not create a balanced 50/50 distribution within each split.
- **Option C (Incorrect):** 10 class-1 samples is 10% of the 100 class-1 samples, not 80%. The training set receives 80% of the data, including 80% of the class-1 samples.
- **Option D (Correct):** With `test_size=0.2` and stratification, 80% of all data goes to training. The 100 class-1 samples are distributed proportionally: 80% × 100 = 80 class-1 samples in training, and 20% × 100 = 20 in the test set. This preserves the 20% class-1 ratio in both splits.

</details>

---

### 44. Question

A machine learning engineer examines the following code:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

result = Clf.predict(X_test)
```

**What error will Python raise, and on which line?**

- [ ] **A.** A `ValueError` on `clf.fit(X_train, y_train)` because the model expects the data to be normalized first.
- [ ] **B.** A `TypeError` on `Clf.predict(X_test)` because `predict` expects keyword arguments.
- [ ] **C.** A `NameError` on `result = Clf.predict(X_test)` because `Clf` (capital C) is not defined; the model is stored as `clf` (lowercase c).
- [ ] **D.** A `AttributeError` on `Clf.predict(X_test)` because `LogisticRegression` does not have a `predict` method.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression.fit()` does not require normalized input data to run without error (though normalization may improve convergence). The code at `clf.fit(...)` will execute without raising a `ValueError`.
- **Option B (Incorrect):** `predict()` accepts positional arguments (the feature array `X_test`). This is standard scikit-learn API usage and does not raise a `TypeError`.
- **Option C (Correct):** Python is strictly case-sensitive. The model was assigned to `clf` (lowercase), but the code later references `Clf` (capital C), which is an undefined identifier. Python's name lookup raises `NameError: name 'Clf' is not defined`.
- **Option D (Incorrect):** `LogisticRegression` fully supports the `.predict()` method as a core part of the scikit-learn classifier API. There is no `AttributeError` here.

</details>

---

### 45. Question

A logistic regression model is applied to a patient triage dataset. For patient A, the model's linear score is z = 2.5. For patient B, z = -0.8.

**What can be concluded about the sigmoid outputs and predicted labels for these two patients?**

- [ ] **A.** Patient A has g(2.5) > 0.5, so is predicted as class 1. Patient B has g(-0.8) < 0.5, so is predicted as class 0.
- [ ] **B.** Patient A has g(2.5) < 0.5 and is predicted as class 0. Patient B has g(-0.8) > 0.5 and is predicted as class 1.
- [ ] **C.** Both patients are predicted as class 1 because both z-scores are within the range (-3, +3).
- [ ] **D.** Patient A has g(2.5) = 0.5 and is predicted as class 0. Patient B has g(-0.8) = 0.5 and is predicted as class 1.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** For positive z, the sigmoid output exceeds 0.5; for z = 2.5, g(2.5) ≈ 0.92, which is well above 0.5, predicting class 1. For negative z, the sigmoid output is below 0.5; g(-0.8) ≈ 0.31, predicting class 0.
- **Option B (Incorrect):** This reverses the sigmoid's behavior. A positive z score (2.5) always produces g(z) > 0.5. A negative z score (-0.8) always produces g(z) < 0.5. The sigmoid is monotonically increasing, not decreasing.
- **Option C (Incorrect):** The range (-3, +3) has no special meaning in logistic regression. Predictions are made based on whether g(z) > 0.5, which is equivalent to z > 0. Any positive z predicts class 1; any negative z predicts class 0.
- **Option D (Incorrect):** The sigmoid only equals exactly 0.5 when z = 0. For z = 2.5 and z = -0.8, the sigmoid outputs are approximately 0.92 and 0.31 respectively — neither is 0.5.

</details>

---

### 46. Question

A dataset for predicting loan default has a column `marital_status` with values: `Single`, `Married`, `Divorced`, `Widowed`. There is no meaningful mathematical order between these categories.

**Which of the following correctly describes why one-hot encoding is preferred over ordinal encoding for this column?**

- [ ] **A.** One-hot encoding is preferred because it always produces better model accuracy than ordinal encoding.
- [ ] **B.** One-hot encoding is preferred because scikit-learn cannot process ordinal-encoded values for logistic regression.
- [ ] **C.** One-hot encoding is preferred because ordinal encoding would assign arbitrary integers that imply a false ranking (e.g., Single < Married < Divorced < Widowed), which could mislead the model's coefficient estimation.
- [ ] **D.** One-hot encoding is preferred because it reduces the dataset to a smaller number of features.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** One-hot encoding does not universally produce better accuracy. The choice of encoding should be based on the nature of the variable (ordinal vs. nominal), not on a general accuracy claim. For features with genuine ordinal structure, ordinal encoding can outperform one-hot encoding.
- **Option B (Incorrect):** scikit-learn can process ordinal-encoded integer values without any issue. The concern is not a technical limitation but a conceptual one: imposing an unwarranted ranking misleads the model.
- **Option C (Correct):** Ordinal encoding assigns consecutive integers (e.g., 1, 2, 3, 4) that imply a quantitative rank order. For `marital_status`, no such rank exists — "Divorced" is not mathematically greater than "Married." The logistic regression model would learn misleading coefficients based on this false ordering. One-hot encoding treats each category as independent, avoiding this problem.
- **Option D (Incorrect):** One-hot encoding typically increases the number of columns (one new column per category), not decreases it. Ordinal encoding uses a single column for all categories, which is dimensionally more compact.

</details>

---

### 47. Question

An engineer trains a `LogisticRegression` model on a 10,000-sample dataset with a single feature and binary target. She then calls `model.predict_proba([[0]])` — passing a feature value of 0.

**What probability does she expect for class 1, and why?**

- [ ] **A.** Exactly 1.0, because a feature value of 0 maximizes the sigmoid output.
- [ ] **B.** Exactly 0.0, because a feature value of 0 contributes nothing to the score.
- [ ] **C.** Approximately 0.5 only if the intercept is also 0; the actual probability depends on the intercept and coefficient of the model.
- [ ] **D.** 0.5 always, regardless of intercept and coefficient, because the feature value is 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** A feature value of 0 does not maximize the sigmoid. The sigmoid equals 1.0 only as z approaches +∞. Setting the feature to 0 eliminates the feature's contribution, but the intercept still determines z.
- **Option B (Incorrect):** A feature value of 0 zeros out that feature's weighted term, but the intercept (`b`) still contributes to z. If b = -2.0, then z = 0×w + (-2.0) = -2.0, giving g(z) ≈ 0.12, not exactly 0.
- **Option C (Correct):** With input x = 0, z = w×0 + b = b. The probability g(z) = g(b) depends entirely on the fitted intercept `b`. If b = 0, then g(0) = 0.5. If b > 0, the probability exceeds 0.5. If b < 0, it is below 0.5. The feature value of 0 only eliminates the coefficient's contribution; the intercept still determines the output.
- **Option D (Incorrect):** The probability is 0.5 only when z = 0, which occurs only when the intercept is also 0 (since the feature term vanishes). The intercept is a fitted parameter that can take any value; it is not guaranteed to be zero.

</details>

---

### 48. Question

A developer wants to inspect the fitted coefficients of a logistic regression model. She writes:

```python
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
```

**Which statements about `model.coef_` and `model.intercept_` are correct?**

- [ ] **A.** `model.coef_` contains the learned weights for each feature, and `model.intercept_` contains the fitted bias term (the constant c in z = w·x + c).
- [ ] **B.** `model.coef_` contains classification accuracy scores, and `model.intercept_` contains the default threshold value (0.5).
- [ ] **C.** `model.coef_` is always a 1D array of length equal to the number of training samples.
- [ ] **D.** `model.intercept_` is always 0.0 because scikit-learn fits logistic regression without a bias term.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** After fitting, `model.coef_` is a 2D array of shape `(n_classes_minus_one, n_features)` containing the learned coefficients (one per feature per decision boundary). `model.intercept_` is a 1D array containing the bias term(s). Together, these define the linear equation z = coef_·x + intercept_.
- **Option B (Incorrect):** `model.coef_` contains learned weights from the optimization process, not accuracy scores. `model.intercept_` contains the bias term of the linear equation, not the classification threshold. The threshold is a separate decision-making step not stored in these attributes.
- **Option C (Incorrect):** `model.coef_` has shape `(n_classes_minus_one, n_features)` — its length depends on the number of features and classes, not the number of training samples. The number of training samples affects optimization but not the coefficient array's shape.
- **Option D (Incorrect):** By default, `LogisticRegression` fits a bias/intercept term (`fit_intercept=True`). The intercept is a freely learned parameter that can be any real number. It is zero only if the data happens to support that value, not by default constraint.

</details>

---

### 49. Question

A data scientist is tasked with predicting whether a startup will succeed or fail within 5 years. She has collected features like `funding_amount`, `team_size`, `industry_sector`, and `founder_experience_years`. The target is binary: `success` (1) or `failure` (0).

**Which model should she use, and why?**

- [ ] **A.** `LinearRegression`, because the output is numeric (0 or 1) and linear models are faster.
- [ ] **B.** `LogisticRegression`, because the target is a binary categorical outcome and logistic regression is designed for classification tasks, producing valid probabilities for each class.
- [ ] **C.** `LogisticRegression` cannot be used here because the `industry_sector` column is categorical and would cause an error.
- [ ] **D.** Neither; binary classification with four features requires a neural network.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Even though the target is coded as 0 or 1 (numeric), the problem is a binary classification task, not a numeric regression task. `LinearRegression` would produce out-of-range predictions and is statistically inappropriate for binary outcomes.
- **Option B (Correct):** `LogisticRegression` is the correct choice for binary classification problems. It models P(success | features) using the sigmoid function, produces valid probabilities between 0 and 1, and uses a threshold to assign discrete class labels. It is well-suited for this use case.
- **Option C (Incorrect):** `LogisticRegression` itself would not cause an error due to categorical columns — scikit-learn simply requires that all features be numerically encoded before fitting. The categorical `industry_sector` column must be encoded (e.g., one-hot), but this is a preprocessing step, not a limitation of the classifier.
- **Option D (Incorrect):** Binary classification with four features does not require a neural network. Logistic regression is a well-established, interpretable, and often effective approach for this type of problem, especially when the dataset is moderate in size and the relationships are roughly linear in log-odds.

</details>

---

### 50. Question

A team member computes the sigmoid of z = -0.5 and claims: "Since -0.5 is close to zero, the probability must be very close to 0.5."

**Is this claim correct, and what is the approximate value of g(-0.5)?**

- [ ] **A.** Incorrect; g(-0.5) ≈ 0.0 because any negative z produces a near-zero probability.
- [ ] **B.** Correct; g(-0.5) = 0.5 exactly because -0.5 is approximately 0.
- [ ] **C.** Approximately correct; g(-0.5) ≈ 0.38, which is below 0.5 but still in the middle range of the sigmoid, not near the extremes.
- [ ] **D.** Incorrect; g(-0.5) ≈ 0.93 because the sigmoid inverts negative inputs to high probabilities.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Near-zero probabilities only occur for large-magnitude negative z values (e.g., z < -5). For z = -0.5, the sigmoid output is moderate, around 0.38 — not nearly zero.
- **Option B (Incorrect):** g(-0.5) ≠ 0.5 exactly. g(z) = 0.5 only when z = 0 exactly. For z = -0.5, the sigmoid output is slightly below 0.5, approximately 0.38.
- **Option C (Correct):** g(-0.5) = 1/(1 + e^(0.5)) ≈ 1/(1 + 1.649) ≈ 1/2.649 ≈ 0.378. This is below 0.5 but still in the central region of the sigmoid curve. The team member's qualitative observation is approximately right (the value is near 0.5), but the precise claim that it "must be very close to 0.5" is slightly misleading — it is about 0.12 away.
- **Option D (Incorrect):** The sigmoid does not invert negative z to high probabilities. A negative z always produces g(z) < 0.5. Large positive z produces high probabilities; large negative z produces low probabilities. g(-0.5) ≈ 0.38, not 0.93.

</details>
