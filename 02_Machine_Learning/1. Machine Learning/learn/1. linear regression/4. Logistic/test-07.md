# Logistic Regression Objective Test — Set 07

### 61. Question

A binary logistic regression model is trained to classify whether a patient has a disease. After fitting, the engineer checks `model.classes_` and sees `array([0, 1])`. She calls `model.predict_proba(X_test)` and gets an array where the second column contains values like 0.83, 0.12, 0.67.

**What do these second-column values represent?**

- [ ] **A.** The predicted class labels (0 or 1) for each sample in the test set.
- [ ] **B.** The R² contribution of each feature to the prediction.
- [ ] **C.** The probability that each test sample belongs to class 1 (disease present), since the second column corresponds to the second entry in `model.classes_` (which is 1).
- [ ] **D.** The confidence interval width around the predicted class label.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The predicted class labels (0 or 1) are returned by `model.predict()`, not `model.predict_proba()`. `predict_proba` always returns probabilities, not discrete labels.
- **Option B (Incorrect):** R² is a model-level evaluation metric computed from `LinearRegression.score()`. It has no role in the output of `predict_proba()`.
- **Option C (Correct):** `predict_proba` returns a 2D array where each column corresponds to a class in the order stored in `model.classes_`. Since `model.classes_ = [0, 1]`, the first column is P(class 0) and the second column is P(class 1). The values 0.83, 0.12, 0.67 are the probabilities of each respective test sample having the disease (class 1).
- **Option D (Incorrect):** `predict_proba` does not return confidence intervals. Confidence intervals would require bootstrapping or Bayesian inference. Each value in `predict_proba` output is a point probability estimate, not an interval.

</details>

---

### 62. Question

A machine learning student reads: "Logistic regression outputs a probability by applying the sigmoid to the linear score." She then runs the following code:

```python
import math

z = 2.0
g = 1 / (1 + math.exp(-z))
print(round(g, 4))
```

**What does the code print?**

- [ ] **A.** 0.5000
- [ ] **B.** 0.1192
- [ ] **C.** 0.8808
- [ ] **D.** 2.0000

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** g(z) = 0.5 only when z = 0. For z = 2.0, the output is above 0.5 because the sigmoid is monotonically increasing and positive z values produce probabilities greater than 0.5.
- **Option B (Incorrect):** 0.1192 is approximately g(-2.0), not g(2.0). This would result from computing `1 / (1 + math.exp(2))` instead of `1 / (1 + math.exp(-2))`.
- **Option C (Correct):** g(2.0) = 1 / (1 + e^(-2)) = 1 / (1 + 0.1353) ≈ 1 / 1.1353 ≈ 0.8808. This is the correct sigmoid value for z = 2.0, confirming a high probability of class 1.
- **Option D (Incorrect):** The sigmoid function never returns the raw z value itself. The output of g(z) is always bounded between 0 and 1; returning 2.0 would violate this constraint and would indicate a coding error.

</details>

---

### 63. Question

A team is preparing features for a loan model. They have a column `payment_history` with values: `Never_missed`, `1-2_missed`, `3-5_missed`, `6+_missed`. These categories represent increasing levels of payment delinquency.

**Which encoding correctly handles this column?**

- [ ] **A.** One-hot encoding, because payment history is a string column and one-hot encoding is required for all string columns.
- [ ] **B.** Ordinal encoding (e.g., `Never_missed=0, 1-2_missed=1, 3-5_missed=2, 6+_missed=3`), because the categories represent a meaningful natural order of delinquency severity.
- [ ] **C.** One-hot encoding, because there are only four categories, making one-hot encoding computationally trivial.
- [ ] **D.** No encoding needed; logistic regression can rank text categories automatically.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** One-hot encoding is not required simply because a column contains strings. The choice depends on whether the column is nominal (no order) or ordinal (with order). `payment_history` has a clear order: no misses < 1-2 misses < 3-5 misses < 6+ misses.
- **Option B (Correct):** `payment_history` is an ordinal variable with a well-defined, meaningful rank order representing increasing levels of financial risk. Ordinal encoding (0, 1, 2, 3) correctly captures this progression and allows the model to learn that higher delinquency levels have proportionally stronger effects.
- **Option C (Incorrect):** The number of categories is not the deciding factor for encoding choice. Even a small number of ordinal categories should use ordinal encoding to preserve their meaningful rank structure.
- **Option D (Incorrect):** scikit-learn's `LogisticRegression` requires numerical input. It cannot process raw string values or infer any ranking from textual category names. Encoding must be performed explicitly before fitting.

</details>

---

### 64. Question

A data scientist is explaining the relationship between the linear score z and the predicted class in logistic regression.

**Which of the following statements correctly maps z-score ranges to class predictions?**

- [ ] **A.** z > 1.0 → predict class 1; z < -1.0 → predict class 0; -1 ≤ z ≤ 1 → no prediction.
- [ ] **B.** z > 0 → g(z) > 0.5 → predict class 1; z ≤ 0 → g(z) ≤ 0.5 → predict class 0 (with z = 0 mapping to label 0 under the strict threshold).
- [ ] **C.** z > 0.5 → predict class 1; z ≤ 0.5 → predict class 0.
- [ ] **D.** z = 0.5 → decision boundary; z > 0.5 → class 1; z < 0.5 → class 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** There is no "no prediction" zone in logistic regression. Every input gets a prediction. The decision boundary is at z = 0 (g(z) = 0.5), not at z = ±1. For any z, the model produces a prediction.
- **Option B (Correct):** The sigmoid function satisfies g(z) > 0.5 if and only if z > 0, and g(z) < 0.5 if and only if z < 0. At z = 0, g(z) = 0.5 exactly, which under scikit-learn's strict greater-than threshold maps to class 0. This is the correct mapping between z values and predicted class labels.
- **Option C (Incorrect):** The decision boundary in logistic regression is at z = 0, not z = 0.5. At z = 0.5, g(0.5) ≈ 0.62, which is above the 0.5 probability threshold. The threshold of 0.5 applies to the probability (the sigmoid output), not to z directly.
- **Option D (Incorrect):** The decision boundary in z-space is z = 0 (corresponding to g(z) = 0.5 in probability space). The 0.5 threshold applies to the probability output, not to the z value. Confusing these two domains is a common error.

</details>

---

### 65. Question

A logistic regression model is fitted on a training set with `LogisticRegression(max_iter=100)`. During fitting, scikit-learn prints a `ConvergenceWarning`. A colleague suggests changing the variable name to fix it.

**What is the actual cause of and fix for the `ConvergenceWarning`?**

- [ ] **A.** The warning is caused by a mismatch in variable names between the model and the data; renaming both resolves it.
- [ ] **B.** The warning means the optimization algorithm did not converge within 100 iterations; increasing `max_iter` (e.g., `max_iter=1000`) or scaling the features gives the optimizer more steps or better-conditioned data to converge.
- [ ] **C.** The warning indicates the model has overfit; reducing `max_iter` to fewer iterations resolves it.
- [ ] **D.** A `ConvergenceWarning` in scikit-learn means the dataset is too small; adding more training samples always resolves it.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Variable names have absolutely no effect on model fitting or optimization convergence. Renaming variables does not change the mathematical optimization process.
- **Option B (Correct):** A `ConvergenceWarning` from scikit-learn's logistic regression indicates the iterative solver (e.g., LBFGS, SAG) did not reach the convergence criterion within `max_iter` iterations. The two most common fixes are: (1) increasing `max_iter` to allow more optimization steps, and (2) scaling features (e.g., `StandardScaler`) so the optimization landscape is better conditioned, allowing faster convergence.
- **Option C (Incorrect):** Reducing `max_iter` would make convergence less likely, not more. The warning is about insufficient iterations, not too many. Overfitting is a different concern and would not trigger a `ConvergenceWarning`.
- **Option D (Incorrect):** While more data can sometimes help, a `ConvergenceWarning` is not primarily caused by dataset size. It is caused by the optimization algorithm running out of iterations. Small datasets can actually converge faster; large, high-dimensional datasets with unscaled features are more prone to convergence issues.

</details>

---

### 66. Question

A developer inspects the result of a `train_test_split` call and finds the following:

- Training set: 750 samples (class 0: 600, class 1: 150)
- Test set: 250 samples (class 0: 230, class 1: 20)

**The original dataset had 1000 samples with 750 class-0 and 250 class-1 samples. Was stratification used in the split, and how can you tell?**

- [ ] **A.** Yes, stratification was used; the training set has more class-0 samples than class-1 samples, which always indicates stratification.
- [ ] **B.** No, stratification was likely not used; the original 25% class-1 ratio is not preserved. With stratification, the test set should have had 62–63 class-1 samples (25% of 250), not 20.
- [ ] **C.** Yes, stratification was used because both splits contain at least some samples from both classes.
- [ ] **D.** It is impossible to determine from the counts whether stratification was used.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Having more class-0 than class-1 samples in the training set is expected regardless of stratification, since class 0 is the majority. This does not distinguish stratified from non-stratified splits.
- **Option B (Correct):** The original dataset has 250/1000 = 25% class-1 samples. With stratification, the test set (250 samples) should contain approximately 250 × 0.25 = 62.5 ≈ 62 or 63 class-1 samples. The actual test set contains only 20 class-1 samples (8%), which is a significant deviation from the original 25% ratio — indicating that stratification was not used and the random split produced an under-representative test set for class 1.
- **Option C (Incorrect):** Having some samples from both classes is a necessary but not sufficient condition for stratification. A random split will often include both classes, especially for majority/minority splits. What stratification guarantees is that the proportions match — not just that both classes are present.
- **Option D (Incorrect):** You can absolutely determine whether stratification was used by comparing the class ratios in each split to the original dataset's class ratio. If the ratios are preserved (within rounding), stratification was likely used; if they deviate significantly, it was likely not.

</details>

---

### 67. Question

A logistic regression model trained for binary classification is examined. The engineer queries `model.intercept_` and finds `[-3.5]`.

**What does this intercept value mean in the context of the model's predictions?**

- [ ] **A.** The intercept sets a negative baseline for z, meaning a sample with all features equal to zero would have z = -3.5 and a low predicted probability of class 1 (approximately 3%).
- [ ] **B.** The intercept of -3.5 means the model predicts class 0 for 3.5% of all samples.
- [ ] **C.** A negative intercept means the model was fitted incorrectly and should be retrained with a non-negative intercept constraint.
- [ ] **D.** The intercept of -3.5 is the R² score of the model, stored in `model.intercept_`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** The intercept is the value of z when all feature values are zero: z = w·0 + b = b = -3.5. g(-3.5) = 1/(1 + e^3.5) ≈ 1/(1 + 33.1) ≈ 0.029 ≈ 3%. This means a "baseline" sample (all features = 0) has only about a 3% predicted probability of being class 1.
- **Option B (Incorrect):** The intercept does not represent a percentage of samples. It is the bias term that shifts the decision boundary. It is a coefficient in the linear equation, not a prediction count or proportion.
- **Option C (Incorrect):** A negative intercept is perfectly valid and common in logistic regression. It simply means the baseline (all-zero features) predicts predominantly class 0. scikit-learn does not constrain the intercept to be non-negative.
- **Option D (Incorrect):** `model.intercept_` stores the bias term of the linear equation, not any evaluation metric. R² would be stored nowhere in the fitted model — it is computed on-demand by `model.score()` for regression models only.

</details>

---

### 68. Question

A scientist wants to use a logistic regression model to predict three outcomes: `healthy` (0), `at-risk` (1), and `diseased` (2). She imports `LogisticRegression` from `sklearn.linear_model` and fits it on this three-class dataset.

**Which statements about this setup are correct?**

- [ ] **A.** `LogisticRegression` supports multi-class classification natively; it does not need to be replaced with a different model.
- [ ] **B.** `model.predict_proba(X_test)` will return a 2D array with three columns — one probability per class per sample.
- [ ] **C.** `LogisticRegression` cannot handle three classes and will raise an error during `fit()`.
- [ ] **D.** `model.predict()` will return probabilities between 0 and 1 for each of the three classes.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** scikit-learn's `LogisticRegression` supports multi-class classification through strategies like One-vs-Rest (OvR) or multinomial (softmax). The `multi_class` parameter controls this behavior. No separate model class is required for three-class problems.
- **Option B (Correct):** For a three-class problem, `predict_proba` returns an array of shape `(n_samples, 3)`. Each row contains three probabilities (one per class) that sum to 1.0. The column order matches `model.classes_`.
- **Option C (Incorrect):** `LogisticRegression.fit()` handles multi-class targets without raising errors. scikit-learn automatically detects the number of unique classes in y and applies the appropriate multi-class strategy.
- **Option D (Incorrect):** `model.predict()` returns discrete class labels (0, 1, or 2 in this case), not probabilities. Probabilities are returned by `model.predict_proba()`. The distinction between `predict()` and `predict_proba()` applies in both binary and multi-class settings.

</details>

---

### 69. Question

A model is trained on a dataset where the feature `num_previous_claims` has a positive coefficient of +0.45. A risk analyst says: "If we could reduce a customer's number of previous claims from 5 to 2, the predicted risk would improve."

**Is this statement correct, and by how much would z change?**

- [ ] **A.** Incorrect; reducing a feature with a positive coefficient increases z and therefore increases risk.
- [ ] **B.** Correct; reducing `num_previous_claims` from 5 to 2 (a decrease of 3) would reduce z by 0.45 × 3 = 1.35, lowering the predicted risk probability.
- [ ] **C.** Incorrect; the coefficient affects the direction only, and a change of 3 units would have no measurable effect.
- [ ] **D.** Correct; z would increase by 1.35 because reducing a positive feature value increases the positive contribution.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Reducing a feature with a positive coefficient decreases z (since z = +0.45 × feature_value). A decrease in the feature value multiplied by a positive coefficient produces a negative change in z, which lowers the predicted probability.
- **Option B (Correct):** z = 0.45 × num_previous_claims + other terms. A reduction from 5 to 2 means a change of -3 in the feature value. Change in z = 0.45 × (-3) = -1.35. A decrease of 1.35 in z reduces g(z), lowering the predicted risk probability. The analyst's statement is correct.
- **Option C (Incorrect):** A change of 3 units in the feature value has a direct, measurable, and quantifiable effect on z. The coefficient (+0.45) scales the feature's contribution to z linearly. A 3-unit decrease causes a 1.35 decrease in z.
- **Option D (Incorrect):** Reducing the feature value (−3) multiplied by a positive coefficient (+0.45) produces a negative contribution to z: 0.45 × (−3) = −1.35. z decreases by 1.35; it does not increase.

</details>

---

### 70. Question

A team uses the following code to evaluate their model but receives an unexpected result:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))   # prints 0.94
print(model.score(X_train, y_train)) # prints 0.76
```

**What does this unexpected pattern (test accuracy > training accuracy) most likely indicate?**

- [ ] **A.** The model has overfit to the test data and generalized poorly to the training data.
- [ ] **B.** The code has a bug; scikit-learn guarantees that training accuracy always exceeds or equals test accuracy.
- [ ] **C.** The test set may be easier to classify than the training set, possibly because the random split produced an unrepresentative (too easy) test set. Using stratification and cross-validation would give more reliable estimates.
- [ ] **D.** The model's `max_iter` was too low, causing the optimizer to underfit the training data, explaining the lower training accuracy.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Overfitting typically causes high training accuracy and low test accuracy — the opposite of what is observed. What we see (high test, lower training) does not match the overfitting pattern.
- **Option B (Incorrect):** scikit-learn does not guarantee that training accuracy always exceeds test accuracy. It often does in practice (especially when overfitting), but any single random split can produce a test set that happens to be easier than the training set.
- **Option C (Correct):** The most likely explanation is that the random split (without stratification) produced a test set that is easier to classify — perhaps containing more clear-cut cases or better-balanced classes. This is why a single train/test split can be misleading. Adding `stratify=y` and using cross-validation would produce more reliable and representative performance estimates.
- **Option D (Incorrect):** A `max_iter` that is too low causes a `ConvergenceWarning` and may underfit the training data, but the difference in accuracy here (0.94 vs. 0.76) would need to be very dramatic to attribute it to convergence issues. Moreover, this would cause both train and test accuracy to suffer, not just training accuracy.

</details>
