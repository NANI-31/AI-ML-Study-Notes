# Logistic Regression Objective Test — Set 08

### 71. Question

A machine learning team examines their binary classifier and realizes the model is outputting the same label (class 0) for every single test sample, despite both classes being present in the test set.

**Which of the following is NOT a plausible reason for this behavior?**

- [ ] **A.** Severe class imbalance: the model learned that predicting class 0 always is optimal for accuracy.
- [ ] **B.** Features are not contributing meaningful signal, and the intercept dominates — a large negative intercept drives all predictions toward class 0.
- [ ] **C.** The `LogisticRegression` class has a built-in rule that prevents predicting class 1 unless explicitly enabled via a `allow_class1=True` parameter.
- [ ] **D.** The threshold of 0.5 may be too high relative to the probability distribution of the model's outputs for the minority class.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Plausible — not the answer):** Severe class imbalance is a well-known cause of a model always predicting the majority class. With 99% class-0 training data, the optimizer learns that class 0 minimizes log-loss even without ever predicting class 1.
- **Option B (Plausible — not the answer):** A very large negative intercept with weak feature coefficients would produce z << 0 for almost all inputs, yielding g(z) ≈ 0 and consistently predicting class 0.
- **Option C (Correct answer — this is NOT plausible):** `LogisticRegression` in scikit-learn has no such parameter as `allow_class1=True`. This restriction does not exist. The model can and does predict both classes based on learned coefficients and the data.
- **Option D (Plausible — not the answer):** If the model's predicted probabilities for class 1 are consistently below 0.5 (even for true class-1 samples), the default threshold will always map them to class 0. Lowering the threshold (e.g., to 0.3) using `predict_proba` and custom thresholding can reveal this issue.

</details>

---

### 72. Question

A classification model is trained and then evaluated with this code:

```python
preds = model.predict(X_test)
proba = model.predict_proba(X_test)

# For one sample:
print(preds[5])        # prints 1
print(proba[5])        # prints [0.48, 0.52]
```

**Are these outputs consistent with each other, and why?**

- [ ] **A.** No; the prediction should be 0 because the class-1 probability (0.52) is not high enough to trigger a positive prediction.
- [ ] **B.** Yes; the class-1 probability is 0.52, which is strictly greater than the default threshold of 0.5, so `predict()` correctly assigns label 1.
- [ ] **C.** No; `predict()` and `predict_proba()` use different thresholds by default, so their outputs may contradict each other.
- [ ] **D.** Yes; `predict()` always assigns label 1 when `predict_proba` produces any value above 0.5 in either column.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** 0.52 > 0.5 is true, so label 1 is the correct prediction. The default threshold of 0.5 requires the class-1 probability to be strictly greater than 0.5 for a positive prediction. 0.52 satisfies this condition.
- **Option B (Correct):** `predict()` uses `predict_proba()` internally and applies a 0.5 threshold to the class-1 probability column. Here P(class 1) = 0.52 > 0.5, so the label is 1. P(class 0) = 0.48 and P(class 1) = 0.52 is consistent with the reported prediction of 1.
- **Option C (Incorrect):** `predict()` and `predict_proba()` use the same default threshold (0.5). `predict()` is literally derived from `predict_proba()` by applying the threshold. They are not independent computations with different thresholds.
- **Option D (Incorrect):** The criterion for `predict()` assigning label 1 is specifically that the class-1 probability (the second column for binary classes) exceeds the threshold, not that any column exceeds it. The class-0 probability exceeding 0.5 would predict class 0.

</details>

---

### 73. Question

A feature engineering team has a column `city` with 50 unique values. A junior engineer proposes using one-hot encoding, which would add 50 new binary columns. A senior engineer suggests ordinal encoding instead to limit column expansion.

**Which statement correctly evaluates the senior engineer's suggestion?**

- [ ] **A.** The senior engineer is correct; ordinal encoding is always preferred when there are more than 10 unique categories.
- [ ] **B.** The senior engineer's concern about dimensionality is valid, but ordinal encoding is inappropriate for `city` (a nominal variable). Better alternatives include target encoding or dimensionality reduction techniques.
- [ ] **C.** The junior engineer is always correct; one-hot encoding is mandatory for all categorical features regardless of cardinality.
- [ ] **D.** Both engineers are wrong; `city` should be dropped entirely since geographic features never improve logistic regression models.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** There is no threshold of "10 unique categories" above which ordinal encoding becomes preferred. The correct encoding choice depends on whether the variable is nominal or ordinal, regardless of cardinality. Ordinal encoding is inappropriate for nominal variables at any cardinality.
- **Option B (Correct):** The senior engineer is right to be concerned about high-cardinality one-hot encoding adding 50 columns, which can increase model complexity and sparse representation. However, the solution is not ordinal encoding, which would impose a meaningless numerical rank on 50 cities. Better alternatives include target encoding (replacing city with the mean of the target per city), frequency encoding, embedding approaches, or grouping cities by meaningful attributes (region, size).
- **Option C (Incorrect):** One-hot encoding is not mandatory for all categorical features. For high-cardinality nominal features, alternatives like target encoding are often more practical and effective. The "mandatory" claim is incorrect.
- **Option D (Incorrect):** Geographic features like city can contain meaningful signal (e.g., regional economic conditions, local market characteristics) that improves model performance. Blanket recommendations to drop geographic features are not correct.

</details>

---

### 74. Question

A data scientist is explaining why logistic regression is preferred over linear regression for a binary classification task. She says:

"Even if linear regression happens to predict values between 0 and 1 for most training samples, there are still fundamental problems with using it for classification."

**Which of the following correctly identifies those fundamental problems?**

- [ ] **A.** Linear regression is not bounded to (0, 1) for all possible inputs, so it can produce invalid probability estimates for new samples far from the training data.
- [ ] **B.** Linear regression minimizes squared error, which is not the appropriate loss function for a binary classification problem where probabilistic outputs are needed.
- [ ] **C.** Linear regression always produces identical predictions to logistic regression for binary targets, making the choice irrelevant.
- [ ] **D.** Linear regression cannot be applied to datasets with more than 100 samples in binary classification settings.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** Even if linear regression produces values near 0 and 1 on the training set, its predictions are unbounded. For new test samples with feature values outside the training range, predictions can easily fall below 0 or above 1, which cannot be interpreted as valid probabilities.
- **Option B (Correct):** Squared error (OLS) is appropriate for estimating continuous values. For binary classification, the probabilistic nature of the problem requires a loss function that properly handles probability outputs (i.e., log-loss/cross-entropy, as used by logistic regression). Using squared error for binary targets produces biased coefficient estimates under a probabilistic interpretation.
- **Option C (Incorrect):** Linear regression and logistic regression produce different predictions for the same data. They use different loss functions, apply different output transformations, and optimize different objectives. Their outputs are generally not identical.
- **Option D (Incorrect):** Linear regression has no sample size restriction. It can be applied to datasets of any size, including those with more than 100 samples. This claim is factually incorrect.

</details>

---

### 75. Question

A team splits their data with:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7, stratify=y
)
```

Then later in the code:

```python
X_Train, X_Test, y_Train, y_Test = train_test_split(
    X, y, test_size=0.3, random_state=7, stratify=y
)
```

**What is true about the two splits?**

- [ ] **A.** The two splits will be different because the second call creates new random splits regardless of `random_state`.
- [ ] **B.** The two splits will be identical because both calls use the same parameters and the same `random_state`, producing the same deterministic output.
- [ ] **C.** The two splits will differ because different variable names (`X_train` vs. `X_Train`) cause Python to use different memory allocations.
- [ ] **D.** The second split will fail with a `ValueError` because the same data cannot be split twice.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `random_state` explicitly seeds the random number generator. Two calls with identical arguments and the same seed will always produce the same output, not different splits.
- **Option B (Correct):** `train_test_split` is a deterministic function given the same inputs and `random_state`. Both calls use the same `X`, `y`, `test_size=0.3`, `random_state=7`, and `stratify=y`. The resulting splits (`X_train` and `X_Train`, etc.) will contain identical data — only the variable names differ.
- **Option C (Incorrect):** Variable names in Python are just labels that reference objects in memory. Different variable names do not affect memory allocation, computational behavior, or the output of `train_test_split`. The function's output depends entirely on its input arguments.
- **Option D (Incorrect):** There is no restriction on calling `train_test_split` multiple times on the same data. It is a pure function that computes splits without modifying the input arrays.

</details>

---

### 76. Question

A logistic regression model for predicting loan approval has these coefficients after fitting:

| Feature        | Coefficient |
| :------------- | :---------- |
| credit_score   | +0.008      |
| annual_income  | +0.00002    |
| outstanding_debt| -0.003     |
| intercept      | -2.0        |

An applicant has `credit_score = 720`, `annual_income = 60000`, `outstanding_debt = 10000`.

**Which feature contributes the most to pushing this applicant's z-score toward approval?**

- [ ] **A.** `credit_score`, with a contribution of 0.008 × 720 = +5.76.
- [ ] **B.** `annual_income`, with a contribution of 0.00002 × 60000 = +1.20.
- [ ] **C.** `outstanding_debt`, because it has the largest absolute coefficient.
- [ ] **D.** The intercept, because it has the largest absolute value.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** The contribution of each feature to z is coefficient × value. For `credit_score`: 0.008 × 720 = 5.76. For `annual_income`: 0.00002 × 60000 = 1.20. For `outstanding_debt`: -0.003 × 10000 = -30.0. The largest positive contribution (toward approval) comes from `credit_score` at +5.76.
- **Option B (Incorrect):** `annual_income` contributes +1.20, which is positive but smaller than `credit_score`'s +5.76. The largest positive push toward approval comes from `credit_score`.
- **Option C (Incorrect):** `outstanding_debt` has the largest contribution in absolute terms (-30.0), but it pushes z toward rejection, not approval. The question asks specifically about pushing toward approval (positive contributions).
- **Option D (Incorrect):** The intercept (-2.0) has a negative value that pushes z away from approval. Although its absolute value (2.0) is larger than `annual_income`'s contribution (1.20), it is a negative baseline and does not push toward approval.

</details>

---

### 77. Question

A developer writes the following code snippet:

```python
from sklearn.linear_model import LogisticRegression, LinearRegression

log_reg = LogisticRegression()
lin_reg = LinearRegression()

log_reg.fit(X_train, y_train)
lin_reg.fit(X_train, y_train)

score1 = log_reg.score(X_test, y_test)
score2 = lin_reg.score(X_test, y_test)

print(score1, score2)
```

The dataset has binary labels (0 or 1). **Which of the following correctly describes what `score1` and `score2` represent?**

- [ ] **A.** `score1` is R² and `score2` is accuracy; both are between 0 and 1.
- [ ] **B.** Both `score1` and `score2` are R² because both models use a linear equation internally.
- [ ] **C.** `score1` is classification accuracy (a value between 0 and 1 representing the fraction of correct labels); `score2` is R² (which can be negative for poorly fitting models).
- [ ] **D.** Both `score1` and `score2` are classification accuracy because the target is binary (0 or 1).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** This reverses the assignments. `score1` (from `LogisticRegression`) is accuracy, not R². `score2` (from `LinearRegression`) is R², not accuracy.
- **Option B (Incorrect):** Both models sharing a linear equation internally does not mean their `score()` methods return the same metric. The metric returned by `score()` depends on the model's class (regressor or classifier), not on its internal computation.
- **Option C (Correct):** `LogisticRegression.score()` returns classification accuracy — the proportion of test samples where the predicted label matches the true label, bounded between 0 and 1. `LinearRegression.score()` returns R² — the coefficient of determination, which can technically be negative (indicating the model fits worse than a horizontal mean line).
- **Option D (Incorrect):** The `score()` metric is not determined by whether the target is binary. `LinearRegression.score()` always returns R² regardless of the target's data type. A binary 0/1 target does not convert `LinearRegression` into a classifier.

</details>

---

### 78. Question

An engineer uses `predict_proba()` and notices that for sample k, the output is `[0.0001, 0.9999]`. She asks: "Can I trust that this prediction is definitely correct?"

**Which response correctly addresses this question?**

- [ ] **A.** Yes, a probability of 0.9999 means the model is essentially certain; such predictions are always correct.
- [ ] **B.** Not necessarily; a very high probability reflects the model's confidence given its learned parameters, but the model can still be wrong. Calibration and held-out evaluation determine true reliability.
- [ ] **C.** Yes; scikit-learn validates predictions against ground truth during `predict_proba()` and only outputs high probabilities when they match the true label.
- [ ] **D.** No; a probability of 0.9999 indicates the model is overconfident and should be retrained.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** High model confidence does not guarantee correctness. A model can be systematically miscalibrated, have seen only similar (but misleading) training examples, or encounter distributional shift. High probability is an expression of the model's internal state, not a guarantee of ground truth.
- **Option B (Correct):** `predict_proba()` returns the model's estimated probability given its fitted parameters. A high value means the model is confident according to its learned decision boundary, but this confidence can be misplaced if the model is overconfident, poorly calibrated, or if the sample is genuinely unusual. Model calibration (e.g., Platt scaling) and evaluation on held-out data determine whether high probabilities are reliable.
- **Option C (Incorrect):** scikit-learn does not validate predictions against ground truth during `predict_proba()`. The method computes probabilities purely from the model's fitted weights and the input features. It has no access to the true label at prediction time.
- **Option D (Incorrect):** A high probability of 0.9999 does not inherently indicate overconfidence. For clearly-separable samples far from the decision boundary, such probabilities may be well-calibrated and reliable. Whether the model is overconfident requires calibration analysis (e.g., reliability diagrams), not just observing a high probability value.

</details>

---

### 79. Question

A model has been trained and stored as:

```python
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

A new engineer joins the team and writes:

```python
output = Classifier.predict(X_test)
```

**What error will occur, and what would fix it?**

- [ ] **A.** A `NameError` because `Classifier` (capital C) is not defined; the variable is named `classifier` (lowercase c). The fix is to use `classifier.predict(X_test)`.
- [ ] **B.** A `TypeError` because `predict()` requires two arguments (X_test and y_test). The fix is to add `y_test`.
- [ ] **C.** A `ValueError` because `X_test` must be preprocessed before `predict()`. The fix is to call `.transform(X_test)` first.
- [ ] **D.** No error; Python is case-insensitive for variable names, so `Classifier` and `classifier` both refer to the same object.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Python's variable namespace lookup is exact and case-sensitive. `classifier` and `Classifier` are two distinct identifiers. Since only `classifier` was defined, referencing `Classifier` raises `NameError: name 'Classifier' is not defined`. The fix is simply to use the correct case: `classifier.predict(X_test)`.
- **Option B (Incorrect):** `predict()` requires only the feature array `X_test`. It does not need `y_test` — the true labels are only needed for evaluation (e.g., in `score()` or `accuracy_score()`), not for generating predictions.
- **Option C (Incorrect):** `predict()` does not require a separate `.transform()` call. If preprocessing is needed (e.g., scaling), it should be applied before fitting and consistently before predicting, but there is no mandatory `.transform()` step built into `predict()`.
- **Option D (Incorrect):** Python is strictly case-sensitive. Variable name lookup performs exact string matching. `Classifier` and `classifier` never refer to the same object regardless of their types.

</details>

---

### 80. Question

A team compares the decision-making process of `LogisticRegression.predict()` with manually thresholding `predict_proba()` outputs. They write:

```python
import numpy as np
preds_from_predict = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
preds_manual = (proba > 0.5).astype(int)

print(np.array_equal(preds_from_predict, preds_manual))
```

**What does this code print, and why?**

- [ ] **A.** `False`, because `predict()` uses a different internal threshold than 0.5.
- [ ] **B.** `False`, because `predict_proba()` and `predict()` compute probabilities differently.
- [ ] **C.** `True`, because `model.predict()` internally uses `predict_proba()` and applies a threshold of `> 0.5` to the class-1 probabilities, which is identical to what the manual thresholding does.
- [ ] **D.** The code raises a `ValueError` because `np.array_equal` cannot compare integer and boolean arrays.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** scikit-learn's `LogisticRegression.predict()` does use 0.5 as its default threshold for binary classification. The result of `predict()` is equivalent to `(predict_proba()[:, 1] > 0.5).astype(int)`.
- **Option B (Incorrect):** `predict()` internally calls `predict_proba()` to compute probabilities and then applies the threshold. Both methods use the same probability computation; they are not independent.
- **Option C (Correct):** `model.predict()` is effectively defined as `(model.predict_proba(X_test)[:, 1] > 0.5).astype(int)` for binary classifiers. The manual thresholding code replicates this exact logic. Therefore, `preds_from_predict` and `preds_manual` will be element-wise identical, and `np.array_equal()` returns `True`.
- **Option D (Incorrect):** `np.array_equal()` can compare integer arrays with boolean arrays; numpy handles the type coercion automatically. Both arrays ultimately represent 0/1 values and would be equal element-wise. This line does not raise an error.

</details>
