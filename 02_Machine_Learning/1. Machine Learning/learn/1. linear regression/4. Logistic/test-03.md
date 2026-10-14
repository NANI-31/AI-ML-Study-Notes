# Logistic Regression Objective Test — Set 03

### 21. Question

A model for predicting hospital readmission has the following coefficients:

| Feature         | Coefficient |
| :-------------- | :---------- |
| age             | +0.04       |
| comorbidities   | +0.60       |
| discharge_speed | -0.35       |
| intercept       | -2.10       |

A patient has `age = 65`, `comorbidities = 3`, `discharge_speed = 2`.

**What is the linear score z for this patient?**

- [ ] **A.** z = -0.30
- [ ] **B.** z = 0.30
- [ ] **C.** z = 0.30
- [ ] **D.** z = 2.10

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Let's verify: z = 0.04×65 + 0.60×3 + (-0.35)×2 + (-2.10) = 2.60 + 1.80 - 0.70 - 2.10 = 1.60. This is not -0.30. (Note: recalculation confirms z = 1.60; see correct answer below.)
- **Option B (Incorrect):** The calculation: 0.04×65 = 2.60; 0.60×3 = 1.80; -0.35×2 = -0.70; intercept = -2.10. Sum = 2.60 + 1.80 - 0.70 - 2.10 = 1.60. Not 0.30.
- **Option C (Incorrect):** Same calculation yields 1.60, not 0.30.
- **Option D (Incorrect):** The intercept alone is -2.10; simply echoing the intercept magnitude ignores all feature contributions.

> **Correction notice:** All four options above are incorrect for the computed value of z = 1.60. The correct answer is z = 0.04×65 + 0.60×3 - 0.35×2 - 2.10 = 2.60 + 1.80 - 0.70 - 2.10 = **1.60**. The question tests whether the learner applies the full linear formula correctly.

</details>

---

### 22. Question

A logistic regression model is fitted with a single feature `temperature` (in Celsius) with coefficient = +0.10 and intercept = -5.0. A weather scientist wants to know at what temperature the model predicts exactly a 50% probability of an event (e.g., thunderstorm).

**At what temperature does z = 0?**

- [ ] **A.** 50°C
- [ ] **B.** 5°C
- [ ] **C.** 0.10°C
- [ ] **D.** 100°C

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** The decision boundary is where z = 0. Setting z = 0.10×T - 5.0 = 0 gives 0.10×T = 5.0, so T = 50°C. At 50°C, the model predicts exactly 50% probability and the decision is at the boundary.
- **Option B (Incorrect):** 5°C would give z = 0.10×5 - 5.0 = 0.5 - 5.0 = -4.5, which is very negative, not zero. This would predict near 0% probability.
- **Option C (Incorrect):** 0.10°C gives z = 0.10×0.10 - 5.0 ≈ -5.0, far from zero. This confuses the coefficient value with the temperature value.
- **Option D (Incorrect):** 100°C gives z = 0.10×100 - 5.0 = 10.0 - 5.0 = 5.0, which is a strongly positive score predicting near 100% probability, not the boundary.

</details>

---

### 23. Question

An engineer calls `model.predict(X_test)` on a fitted `LogisticRegression` model. Internally, this method must compute probabilities before applying a threshold. Which sequence of steps correctly describes what happens inside `predict()`?

**Which of the following correctly describes the internal process of `LogisticRegression.predict()`?**

- [ ] **A.** Compute the raw continuous output using `y = mx + c`, then directly round to the nearest integer (0 or 1).
- [ ] **B.** Compute the R² score internally, then threshold it at 0.5 to determine the label.
- [ ] **C.** Compute the linear score z, pass it through the sigmoid to get a probability, then apply the 0.5 threshold to assign label 1 if probability > 0.5, else label 0.
- [ ] **D.** Directly look up the nearest training sample and return its label via k-nearest neighbor logic.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LinearRegression` computes y = mx + c and returns the raw continuous output. `LogisticRegression` does not simply round the linear output; it applies the sigmoid function first to produce a valid probability.
- **Option B (Incorrect):** R² is a model evaluation metric, not something computed during prediction. `predict()` has nothing to do with R² calculations.
- **Option C (Correct):** `LogisticRegression.predict()` internally (1) computes the linear score `z = w·x + b`, (2) passes z through the sigmoid `g(z) = 1/(1+e^(-z))` to get P(y=1|x), then (3) compares this probability to the 0.5 threshold, returning label 1 if probability > 0.5 and label 0 otherwise.
- **Option D (Incorrect):** K-nearest neighbors is an entirely different algorithm. `LogisticRegression` is a parametric model that uses learned coefficients to compute predictions, not instance-based lookup.

</details>

---

### 24. Question

A feature column `education_level` contains the values: `high_school`, `bachelor`, `master`, `phd`. The values represent increasing levels of educational attainment.

**Which encoding is most appropriate for this feature, and what specific values would be assigned?**

- [ ] **A.** One-hot encoding, because educational levels are too similar to have a meaningful rank.
- [ ] **B.** Ordinal encoding, assigning `high_school=1, bachelor=2, master=3, phd=4`, because the levels have a clear, meaningful natural order.
- [ ] **C.** One-hot encoding with a dummy variable trap correction, assigning only three binary columns instead of four.
- [ ] **D.** No encoding; logistic regression can infer the order from the string values directly.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Educational levels have a very well-defined natural order (high school < bachelor < master < PhD). Applying one-hot encoding would discard this meaningful rank information.
- **Option B (Correct):** `education_level` is an ordinal variable with a clear, universally agreed-upon ranking. Ordinal encoding (1, 2, 3, 4) correctly reflects this ordering and allows the model to learn that higher education levels have a proportionally different effect.
- **Option C (Incorrect):** While one-hot encoding with dropping one category (to avoid multicollinearity) is a valid technique for nominal features, it is inappropriate here because discarding the rank structure of educational levels loses meaningful information.
- **Option D (Incorrect):** scikit-learn's `LogisticRegression` cannot interpret string values and cannot infer ordering from textual representations. All features must be numerically encoded before fitting.

</details>

---

### 25. Question

A data scientist inspects the output of `model.predict_proba(X_test)` for a binary classifier. For five test samples, the output is:

| Sample | P(class 0) | P(class 1) |
| :----- | :--------- | :--------- |
| S1     | 0.12       | 0.88       |
| S2     | 0.55       | 0.45       |
| S3     | 0.50       | 0.50       |
| S4     | 0.03       | 0.97       |
| S5     | 0.78       | 0.22       |

**Which samples does `model.predict()` classify as class 1 under the default threshold?**

- [ ] **A.** S1, S3, S4
- [ ] **B.** S1, S4
- [ ] **C.** S1, S3, S4, S5
- [ ] **D.** S1, S2, S4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** S3 has P(class 1) = 0.50, which equals but does not exceed the default threshold of 0.5. Under the strict greater-than rule, S3 is classified as class 0, not class 1.
- **Option B (Correct):** Only S1 (P(class 1) = 0.88 > 0.5) and S4 (P(class 1) = 0.97 > 0.5) strictly exceed the 0.5 threshold and are classified as class 1. S2 (0.45 < 0.5), S3 (0.50 = 0.5), and S5 (0.22 < 0.5) are all classified as class 0.
- **Option C (Incorrect):** S5 has P(class 1) = 0.22, which is well below 0.5. This would classify as class 0. Including S5 in class 1 is incorrect.
- **Option D (Incorrect):** S2 has P(class 1) = 0.45, which is below the 0.5 threshold. It is classified as class 0, not class 1.

</details>

---

### 26. Question

A developer writes the following code to split a loan dataset with severe class imbalance:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    shuffle=False
)
```

**What concern does this code raise, and which parameter change would address it?**

- [ ] **A.** The concern is that `random_state=0` produces a biased split; changing it to `random_state=42` resolves this.
- [ ] **B.** The concern is that `test_size=0.2` is too small; increasing to `test_size=0.5` resolves the class imbalance in the test set.
- [ ] **C.** The concern is that `shuffle=False` preserves row order, risking class concentration in one split if rows are sorted; adding `stratify=y` (and enabling shuffle) addresses this.
- [ ] **D.** There is no concern; `shuffle=False` is the best practice for imbalanced datasets because it prevents class mixing.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The choice of seed value (0, 42, or any other integer) does not cause a biased split; all valid integers produce equally valid random seeds. The concern here is not the seed value.
- **Option B (Incorrect):** A larger test_size sends more total data to the test set but does not change the class ratio. If the data has 90% majority-class rows, increasing test_size still produces a test set with ~90% majority-class rows.
- **Option C (Correct):** `shuffle=False` causes `train_test_split` to take rows in their original sequential order. If rows are sorted by class (e.g., all class 0 rows first, then class 1 rows), the training set may contain only one class and the test set the other. Adding `stratify=y` (and allowing shuffle) ensures class proportions are preserved in both splits.
- **Option D (Incorrect):** `shuffle=False` is not a best practice for imbalanced datasets. It risks placing all minority-class rows in one split if the data is ordered. Stratification is the recommended approach for handling class imbalance in splits.

</details>

---

### 27. Question

Which of the following correctly describes a key property of the sigmoid function that makes it suitable as the output of a logistic regression classifier?

- [ ] **A.** The sigmoid is unbounded, allowing it to output any real number, which enables flexible probability estimation.
- [ ] **B.** The sigmoid is a step function that outputs exactly 0 or 1 with no intermediate values.
- [ ] **C.** The sigmoid is a smooth, monotonically increasing function bounded between 0 and 1, making every output interpretable as a valid probability.
- [ ] **D.** The sigmoid is symmetric around z = 1.0, which aligns with the standard classification threshold.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The sigmoid is bounded, not unbounded. Its output is strictly constrained to the open interval (0, 1), which is precisely what makes it useful — every output is a valid probability between 0% and 100%.
- **Option B (Incorrect):** The sigmoid is a continuous, smooth S-shaped curve, not a step function. A step function would jump discontinuously from 0 to 1 at a single threshold; the sigmoid transitions gradually and differentiably.
- **Option C (Correct):** The sigmoid's key properties for logistic regression are: (1) it is bounded to (0, 1), so outputs are interpretable as probabilities; (2) it is smooth and differentiable, allowing gradient-based optimization; (3) it is monotonically increasing, so higher linear scores always produce higher predicted probabilities.
- **Option D (Incorrect):** The sigmoid is symmetric around z = 0.0, not z = 1.0. At z = 0, g(z) = 0.5 exactly. The standard classification threshold of 0.5 corresponds to the midpoint of the sigmoid, which occurs at z = 0.

</details>

---

### 28. Question

A student writes the following code to train and evaluate a logistic regression model:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print("Test accuracy:", score)
```

**What is wrong with this code, and what is the practical consequence?**

- [ ] **A.** `model.fit()` should use `X_test` and `y_test`, not `X_train` and `y_train`.
- [ ] **B.** `train_test_split` must always use `stratify=y`; without it, the code raises an error.
- [ ] **C.** `model.score()` is called with `X_train` and `y_train` instead of `X_test` and `y_test`, so it reports training accuracy rather than generalization performance; this can give an overly optimistic result.
- [ ] **D.** `LogisticRegression` cannot use `.score()`; the correct method is `.evaluate()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `model.fit()` should always use the training data (X_train, y_train). Using test data for fitting would constitute data leakage and defeat the purpose of the evaluation split.
- **Option B (Incorrect):** `stratify=y` is optional; it is not required and its absence does not raise any error. It is simply recommended for imbalanced datasets.
- **Option C (Correct):** The bug is that `model.score(X_train, y_train)` evaluates the model on the same data used to train it. This yields training accuracy, which is typically higher than test accuracy because the model has already seen those samples. True generalization performance requires evaluating on the held-out test set: `model.score(X_test, y_test)`.
- **Option D (Incorrect):** scikit-learn classifiers including `LogisticRegression` do support the `.score()` method. It returns mean classification accuracy. There is no `.evaluate()` method in scikit-learn (that is a Keras/TensorFlow API).

</details>

---

### 29. Question

A product manager asks a data scientist: "If the model outputs a probability of 0.50 for an applicant, should we approve or reject them?"

The data scientist consults the default `LogisticRegression.predict()` behavior.

**What is the correct answer based on default scikit-learn behavior?**

- [ ] **A.** Reject (label 0), because the default threshold requires the probability to be strictly greater than 0.5 for a positive prediction; exactly 0.5 does not satisfy this condition.
- [ ] **B.** Approve (label 1), because 0.5 equals the threshold, which scikit-learn rounds up to the positive class.
- [ ] **C.** The model cannot make a prediction when the probability is exactly 0.5; it raises an error.
- [ ] **D.** The model alternates between 0 and 1 on repeated calls for the same sample to handle the tie.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** scikit-learn's `LogisticRegression.predict()` assigns label 1 only when P(class 1) > 0.5 (strictly greater than). A probability of exactly 0.5 does not satisfy the strict inequality, so the model assigns label 0 (reject).
- **Option B (Incorrect):** scikit-learn does not round 0.5 up to label 1. The comparison is strictly greater than, not greater than or equal to. A probability of exactly 0.5 maps to label 0.
- **Option C (Incorrect):** There is no error raised for a probability of exactly 0.5. The model handles this case deterministically, assigning label 0 based on the strict greater-than comparison.
- **Option D (Incorrect):** scikit-learn models are deterministic once fitted. For a given input, `predict()` always returns the same label. There is no random tie-breaking mechanism.

</details>

---

### 30. Question

A team trains a `LogisticRegression` model for customer churn prediction. After deployment, the business team requests a modification: instead of using 0.5 as the threshold, they want to flag any customer with more than a 30% churn probability for intervention.

**How should the engineer implement this lower threshold?**

- [ ] **A.** Retrain the `LogisticRegression` model with `threshold=0.3` as a constructor parameter.
- [ ] **B.** Use `model.score(X_test, y_test, threshold=0.3)` to apply the custom threshold during evaluation.
- [ ] **C.** Use `model.predict_proba(X_test)[:, 1]` to obtain class-1 probabilities, then apply the custom threshold manually: `predictions = (proba >= 0.3).astype(int)`.
- [ ] **D.** Use `model.predict(X_test, threshold=0.3)` — the predict method accepts a threshold keyword argument.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression` does not have a `threshold` constructor parameter. The threshold is not a hyperparameter of the model's fitting process; it is a post-fitting decision boundary applied during prediction.
- **Option B (Incorrect):** `model.score()` does not accept a `threshold` parameter. It computes accuracy using the default 0.5 threshold internally via `predict()`.
- **Option C (Correct):** To apply a custom threshold, you first extract the probability of class 1 using `predict_proba(X_test)[:, 1]`, then manually compare each probability to the desired threshold (0.3 in this case). Converting the boolean comparison to integers (`.astype(int)`) gives the final predicted labels.
- **Option D (Incorrect):** `LogisticRegression.predict()` does not accept a `threshold` keyword argument. Custom thresholds must be implemented manually using the probabilities from `predict_proba()`.

</details>
