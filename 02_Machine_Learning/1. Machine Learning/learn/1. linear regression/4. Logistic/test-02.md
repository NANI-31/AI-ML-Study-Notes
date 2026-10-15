# Logistic Regression Objective Test — Set 02

### 11. Question

A data scientist is examining the decision boundary of a binary logistic regression model. She wants to know the exact value of the linear score z at which the model switches from predicting label 0 to predicting label 1 using the default threshold.

**At which value of z does the decision boundary lie, and what is the corresponding probability?**

- [ ] **A.** z = 1.0, corresponding to g(z) = 0.73.
- [ ] **B.** z = -1.0, corresponding to g(z) = 0.27.
- [ ] **C.** z = 0.0, corresponding to g(z) = 0.5.
- [ ] **D.** z = 0.5, corresponding to g(z) = 0.62.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** g(1.0) ≈ 0.73, which exceeds the 0.5 threshold, meaning z = 1.0 would predict label 1, but it is not the boundary. The boundary is the value where probability equals exactly 0.5.
- **Option B (Incorrect):** g(-1.0) ≈ 0.27, which is below 0.5, so this predicts label 0. This is not the decision boundary either.
- **Option C (Correct):** At z = 0, g(0) = 1/(1 + e^0) = 1/2 = 0.5 exactly. The default 0.5 threshold means the decision boundary is at z = 0. For z > 0, g(z) > 0.5 (predict 1); for z < 0, g(z) < 0.5 (predict 0).
- **Option D (Incorrect):** g(0.5) ≈ 0.62, which is above 0.5, so this would predict label 1. The decision boundary is specifically where the probability equals the threshold, i.e., where z = 0.

</details>

---

### 12. Question

A junior developer copies the following code snippet from a tutorial and tries to run it:

```python
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train, y_train)

preds = LR.predict(X_test)
proba = lr.predict_proba(X_test)
```

**Which line will cause an error, and what is the exact nature of the error?**

- [ ] **A.** `LR.fit(X_train, y_train)` — `LogisticRegression` does not support a `.fit()` method; use `.train()` instead.
- [ ] **B.** `preds = LR.predict(X_test)` — `predict()` cannot be called after `predict_proba()`.
- [ ] **C.** `proba = lr.predict_proba(X_test)` — `lr` (lowercase) was never defined; the model is stored as `LR` (uppercase), so Python raises a `NameError`.
- [ ] **D.** The code runs without error because Python resolves `lr` to `LR` when the types match.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression` fully supports the `.fit()` method; this is the standard scikit-learn API for training a model. There is no `.train()` method in scikit-learn.
- **Option B (Incorrect):** `predict()` and `predict_proba()` are independent methods and can be called in any order after fitting. There is no restriction preventing `predict()` from being called before `predict_proba()`.
- **Option C (Correct):** Python is strictly case-sensitive. The model was stored as `LR` (uppercase), but the code attempts to call `.predict_proba()` on `lr` (lowercase), which is a completely different, undefined identifier. This raises `NameError: name 'lr' is not defined`.
- **Option D (Incorrect):** Python never performs name resolution based on object type. It performs exact identifier lookup in the current namespace. No amount of type similarity causes Python to resolve one variable name to another.

</details>

---

### 13. Question

A logistic regression model predicts the probability of loan approval. The model has two features: `income` (coefficient = +0.002) and `loan_amount` (coefficient = -0.001), with an intercept of 0.5.

For a customer with `income = 50000` and `loan_amount = 80000`:

**What is the linear score z, and what is the approximate classification?**

- [ ] **A.** z = 0.5, g(z) ≈ 0.62, classified as approved.
- [ ] **B.** z = 20.5, g(z) ≈ 1.0, classified as approved.
- [ ] **C.** z = -29.5, g(z) ≈ 0.0, classified as rejected.
- [ ] **D.** z = 0.0, g(z) = 0.5, classified as rejected.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This ignores the feature contributions. z = 0.002×50000 + (-0.001)×80000 + 0.5 = 100 - 80 + 0.5 = 20.5, not 0.5.
- **Option B (Correct):** Calculating: z = 0.002×50000 + (-0.001)×80000 + 0.5 = 100 - 80 + 0.5 = 20.5. A z-score of 20.5 is an extremely large positive value, so g(20.5) ≈ 1.0. Since 1.0 > 0.5, the customer is classified as approved.
- **Option C (Incorrect):** A negative z would require the loan-related penalty to dominate. Here, income contributes +100 and loan_amount contributes -80, giving a positive net contribution plus the intercept: z = 20.5, not -29.5.
- **Option D (Incorrect):** z = 0 would only occur if the sum of all weighted features plus intercept equalled zero. Here z = 20.5, which is far from 0. Additionally, z = 0 maps to probability 0.5, which under the strict greater-than rule would classify as rejected (label 0), but this is not the correct z anyway.

</details>

---

### 14. Question

A team is using `train_test_split` and sets `random_state=42`. A colleague asks why `random_state` matters.

**Which of the following correctly explains the purpose of `random_state` in `train_test_split`?**

- [ ] **A.** It seeds the random number generator so that the same split is reproduced every time the code is run, enabling reproducible experiments.
- [ ] **B.** It controls the ratio of training to test data; 42 means 42% of the data goes to training.
- [ ] **C.** It enables stratified splitting automatically; the value 42 activates class-balanced splits.
- [ ] **D.** It determines the number of shuffles applied to the dataset before splitting; higher values apply more shuffles.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `random_state` seeds the pseudo-random number generator used during shuffling. By fixing the seed to 42 (or any integer), the split is deterministic — re-running the same code always produces the same train/test partitions, which is critical for reproducible machine learning experiments.
- **Option B (Incorrect):** The training/test ratio is controlled by the `test_size` (or `train_size`) parameter, not `random_state`. A `random_state` of 42 has no connection to a 42% split ratio.
- **Option C (Incorrect):** Stratified splitting is controlled by the `stratify` parameter. Setting `random_state=42` only fixes the randomness seed; it does not activate or configure stratification in any way.
- **Option D (Incorrect):** `random_state` does not set a number of shuffles; it seeds the random state used for a single shuffle before splitting. The value is a seed integer, not a count of operations.

</details>

---

### 15. Question

A `LogisticRegression` model is fitted and its coefficients are inspected. The model has a single feature `age`, with coefficient = +0.05 and intercept = -3.0.

For a person aged 40:

**What is z, and what label does `predict()` assign?**

- [ ] **A.** z = 2.0 + (-3.0) = -1.0, probability < 0.5, label = 0.
- [ ] **B.** z = 0.05 × 40 = 2.0, probability > 0.5, label = 1.
- [ ] **C.** z = 0.05 × 40 + (-3.0) = -1.0, probability < 0.5, label = 0.
- [ ] **D.** z = 0.05 × 40 - (-3.0) = 5.0, probability > 0.5, label = 1.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The calculation is nearly correct but describes it misleadingly. z = 0.05 × 40 + (-3.0) = 2.0 - 3.0 = -1.0 is correct, but option A writes "2.0 + (-3.0)" which is mathematically equivalent; the result -1.0 and label 0 are correct, but this option doesn't properly show the coefficient application step.
- **Option B (Incorrect):** This correctly computes the feature term (0.05 × 40 = 2.0) but omits the intercept (-3.0). The full linear score must include the intercept: z = 2.0 + (-3.0) = -1.0, not 2.0.
- **Option C (Correct):** z = 0.05 × 40 + (-3.0) = 2.0 - 3.0 = -1.0. Since z is negative, g(-1.0) ≈ 0.27, which is below the 0.5 threshold. `predict()` therefore assigns label 0.
- **Option D (Incorrect):** This subtracts a negative intercept (double negation), computing z = 2.0 - (-3.0) = 5.0. This is mathematically incorrect; the intercept is added as-is (i.e., -3.0 is added, not subtracted as a negative).

</details>

---

### 16. Question

A data analyst examines a dataset for predicting student pass/fail outcomes. The dataset has two categorical columns:

- `grade_level`: Grade 6, Grade 7, Grade 8 (ordered by school progression)
- `school_type`: Public, Private, Charter (no meaningful order)

**Which combination of encoding choices is correct?**

- [ ] **A.** Both columns should use one-hot encoding, since encoding consistency is more important than mathematical validity.
- [ ] **B.** Both columns should use ordinal encoding, since integers are simpler and all categorical columns can be ordered alphabetically.
- [ ] **C.** `grade_level` should use ordinal encoding (Grade 6 < Grade 7 < Grade 8); `school_type` should use one-hot encoding since it has no natural order.
- [ ] **D.** `school_type` should use ordinal encoding, since private schools are generally considered higher quality, implying a ranking.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** While consistency may simplify pipelines, encoding consistency is not a reason to apply one-hot encoding to an ordinal variable. Applying one-hot encoding to `grade_level` discards its meaningful rank order, losing information.
- **Option B (Incorrect):** Alphabetical ordering is not a valid rationale for ordinal encoding. The question is whether there exists a meaningful real-world rank. `school_type` has no inherent numeric rank, and imposing one would mislead the model.
- **Option C (Correct):** `grade_level` has a clear, meaningful, natural ordering (Grade 6 < Grade 7 < Grade 8), making ordinal encoding appropriate. `school_type` is nominal with no inherent rank, so one-hot encoding is correct.
- **Option D (Incorrect):** Subjective quality assumptions (e.g., private schools being "higher quality") should not be used to justify ordinal encoding. The encoding choice must be based on whether the feature's values have an objective, well-defined natural rank that the model should learn from.

</details>

---

### 17. Question

A `LinearRegression` model and a `LogisticRegression` model are both trained on the same binary dataset. A junior analyst claims that both models use the same underlying optimization objective (minimizing squared error) since they are both linear models.

**Which statement correctly addresses this claim?**

- [ ] **A.** The analyst is correct; both models minimize squared error since both compute a linear combination of features.
- [ ] **B.** The analyst is partially correct; `LogisticRegression` minimizes squared error but applies the sigmoid afterward.
- [ ] **C.** The analyst is incorrect; `LinearRegression` minimizes squared error (OLS), while `LogisticRegression` minimizes log-loss (negative log-likelihood) via maximum likelihood estimation.
- [ ] **D.** The analyst is incorrect; `LogisticRegression` minimizes squared error, and `LinearRegression` minimizes log-loss.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Both models do compute linear combinations of features, but they use entirely different loss functions. Sharing a linear computation does not mean sharing an optimization objective.
- **Option B (Incorrect):** `LogisticRegression` does not minimize squared error. It minimizes the log-loss (binary cross-entropy), which is derived from the maximum likelihood framework for probabilistic classification.
- **Option C (Correct):** `LinearRegression` uses Ordinary Least Squares (OLS), which minimizes the sum of squared differences between predictions and targets. `LogisticRegression` uses maximum likelihood estimation, minimizing the negative log-likelihood (log-loss), which is appropriate for probabilistic outputs.
- **Option D (Incorrect):** This reverses the correct association. `LinearRegression` uses squared error; `LogisticRegression` uses log-loss.

</details>

---

### 18. Question

A `LogisticRegression` model is trained on a dataset where the target column `severity` has three possible values: 0, 1, and 2. The engineer calls `model.predict_proba(X_test)` and receives an array with three columns.

**What does each column represent in this multi-class `predict_proba` output?**

- [ ] **A.** The three columns represent the raw linear scores (z-values) for each of the three classes.
- [ ] **B.** The three columns represent confidence intervals around the predicted class — lower bound, point estimate, and upper bound.
- [ ] **C.** The three columns represent the probabilities of classes 0, 1, and 2 respectively, and each row sums to 1.0.
- [ ] **D.** The three columns represent the probability of class 0 versus class 1, class 1 versus class 2, and class 0 versus class 2 in pairwise comparisons.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `predict_proba` outputs probabilities (values between 0 and 1), not raw z-scores. Raw scores are available via `model.decision_function()`, not `predict_proba()`.
- **Option B (Incorrect):** `predict_proba` does not output confidence intervals. Confidence intervals would require a different statistical framework (e.g., bootstrapping or Bayesian methods). Each column is a class probability, not a bounds estimate.
- **Option C (Correct):** In multi-class logistic regression, `predict_proba` returns a probability for each class. For a 3-class problem, each row contains three values — one per class — that sum to 1.0. The columns correspond to the class labels in the order stored in `model.classes_`.
- **Option D (Incorrect):** This describes a one-vs-one decomposition approach, not the output format of `predict_proba`. `predict_proba` returns probabilities for each individual class (not pairwise comparisons) in a single array.

</details>

---

### 19. Question

An engineer produces the following two predictions for the same test set:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

lin_model = LinearRegression()
log_model = LogisticRegression()

lin_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

lin_preds = lin_model.predict(X_test)   # e.g., [-0.3, 1.4, 0.6, ...]
log_preds = log_model.predict(X_test)   # e.g., [0, 1, 1, ...]
```

**Which of the following statements correctly describes the difference in the output of these two `predict()` calls?**

- [ ] **A.** Both `lin_preds` and `log_preds` contain continuous values between 0 and 1 since the sigmoid is applied internally in both cases.
- [ ] **B.** `lin_preds` contains unbounded continuous values (which may be negative or greater than 1), while `log_preds` contains discrete class labels (0 or 1).
- [ ] **C.** `lin_preds` contains class labels (0 or 1) derived from thresholding at 0.5, while `log_preds` contains probabilities between 0 and 1.
- [ ] **D.** Both calls return identical outputs because both models use a linear combination of features as their core computation.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The sigmoid is applied only in `LogisticRegression`. `LinearRegression.predict()` returns the raw linear output `y = mx + c`, which is unbounded and can produce values like -0.3 or 1.4.
- **Option B (Correct):** `LinearRegression.predict()` outputs the continuous real-valued prediction of the fitted line, which is unbounded (can be negative, zero, or greater than 1). `LogisticRegression.predict()` applies the sigmoid and threshold internally to return discrete class labels (0 or 1).
- **Option C (Incorrect):** This reverses the descriptions. `LinearRegression` returns continuous unbounded values, not class labels. `LogisticRegression.predict()` returns class labels, not probabilities. (Probabilities are available via `predict_proba()`.)
- **Option D (Incorrect):** While both models compute a linear combination as an intermediate step, the final output of `predict()` differs fundamentally: linear regression returns the raw continuous value; logistic regression applies the sigmoid and threshold to return a binary class label.

</details>

---

### 20. Question

A fraud detection team evaluates their `LogisticRegression` model using `model.score(X_test, y_test)`. The result is 0.97. A stakeholder interprets this as "the model correctly identifies fraud 97% of the time."

**Is this interpretation correct?**

- [ ] **A.** Yes, `model.score()` for classifiers returns the fraud detection rate (sensitivity/recall), which measures how often fraud is correctly identified.
- [ ] **B.** Yes, 0.97 directly means the model has 97% precision on the fraud class.
- [ ] **C.** Not necessarily. `model.score()` returns overall accuracy — the fraction of all predictions (fraud and non-fraud combined) that are correct. With a highly imbalanced dataset, this can be misleading.
- [ ] **D.** No; `model.score()` for `LogisticRegression` returns the R² value, not a classification metric.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `model.score()` returns accuracy (overall proportion correct), not recall/sensitivity. Recall specifically measures how many actual fraud cases were detected, which requires a separate calculation using `recall_score()` from `sklearn.metrics`.
- **Option B (Incorrect):** `model.score()` returns accuracy, not precision. Precision would measure how many flagged-as-fraud cases were actually fraudulent, requiring `precision_score()`.
- **Option C (Correct):** `LogisticRegression.score()` returns classification accuracy — the fraction of all test samples (both classes) where the predicted label matches the true label. With a heavily imbalanced dataset (e.g., 97% non-fraud), a model that always predicts "not fraud" would achieve 97% accuracy while catching zero fraud cases. The 0.97 score does not confirm effective fraud detection.
- **Option D (Incorrect):** `model.score()` returns R² for regression models (`LinearRegression`), not for classifiers. For `LogisticRegression`, `score()` returns classification accuracy.

</details>
