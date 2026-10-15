# Logistic Regression Objective Test — Set 06

### 51. Question

A binary logistic regression model for predicting customer subscription has a coefficient of -1.2 for the feature `cancellation_rate`. A product analyst says: "A negative coefficient means cancellation_rate is not important for the model."

**Which of the following correctly addresses this claim?**

- [ ] **A.** The analyst is correct; negative coefficients indicate features that were not selected during training.
- [ ] **B.** The analyst is incorrect; the negative sign indicates direction (higher cancellation rate → lower subscription probability), and the magnitude (1.2) indicates a strong effect — making it an important feature.
- [ ] **C.** The analyst is partially correct; the feature has some importance only if the coefficient is negative and greater than -1.
- [ ] **D.** The analyst is correct; negative coefficients make the feature value contribute positively, cancelling out any negative effect.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Non-zero coefficients (positive or negative) confirm that a feature was fitted by the model and contributes to predictions. A feature not selected would have a coefficient of 0 (e.g., via L1 regularization), not a negative value.
- **Option B (Correct):** The sign of a coefficient encodes direction: a negative coefficient means that as the feature value increases, the linear score decreases, lowering the predicted probability. The magnitude (1.2) quantifies the strength of that effect. A large negative coefficient represents a strong negative relationship, making it a significant feature.
- **Option C (Incorrect):** There is no rule that negative coefficients between -1 and 0 are "partially important." Importance is related to magnitude (and, for meaningful comparison, feature scale), not to whether the coefficient falls within any particular range around -1.
- **Option D (Incorrect):** A negative coefficient does not cancel itself out or reverse direction. Multiplying a positive feature value by a negative coefficient produces a negative contribution to z, which reduces the probability. There is no double-negative effect here.

</details>

---

### 52. Question

A machine learning engineer writes the following code to get the probability that a specific customer will subscribe to a plan:

```python
model = LogisticRegression()
model.fit(X_train, y_train)

prob = model.predict_proba(X_test[0])
print(prob)
```

**What is the likely problem with this code, and what is the correct fix?**

- [ ] **A.** `predict_proba` is not a valid method; use `predict_probability` instead.
- [ ] **B.** `X_test[0]` extracts a 1D array (a single row), but `predict_proba` expects a 2D input; the fix is to use `X_test[[0]]` or `X_test[0].reshape(1, -1)`.
- [ ] **C.** The issue is that `predict_proba` requires the model to be evaluated on all test samples at once; individual rows cannot be passed.
- [ ] **D.** `X_test[0]` produces a scalar, so `predict_proba` raises a `TypeError`; the fix is to convert it to a Python list.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `predict_proba` is a valid and standard scikit-learn method for classifiers. There is no method called `predict_probability`.
- **Option B (Correct):** When you index a 2D numpy array with a single integer (`X_test[0]`), you get a 1D array of shape `(n_features,)`. scikit-learn's `predict_proba` expects a 2D input of shape `(n_samples, n_features)`. Passing a 1D array raises a `ValueError`. The fix is `X_test[[0]]` (double bracket indexing, which preserves the 2D shape) or `X_test[0].reshape(1, -1)`.
- **Option C (Incorrect):** `predict_proba` can be called on any subset of the test data, including a single row — as long as it is correctly shaped as a 2D array. There is no requirement to pass all test samples simultaneously.
- **Option D (Incorrect):** `X_test[0]` on a 2D numpy array returns a 1D numpy array (one row with all feature values), not a scalar. The issue is dimensionality (1D vs 2D), not a scalar-vs-array mismatch.

</details>

---

### 53. Question

A logistic regression model produces the following z-scores (linear scores) for four applicants before the sigmoid is applied:

| Applicant | z     |
| :-------- | :---- |
| W         | 0.001 |
| X         | -0.001|
| Y         | 5.0   |
| Z         | -5.0  |

**Which pair of applicants has predicted probabilities that are closest to each other?**

- [ ] **A.** W and X, because their z-scores (0.001 and -0.001) are both near zero, placing their probabilities near 0.5.
- [ ] **B.** Y and Z, because both have large-magnitude z-scores, placing their probabilities at the same extreme.
- [ ] **C.** W and Z, because both are likely to be misclassified at the decision boundary.
- [ ] **D.** X and Y, because opposing signs produce cancelling effects on the sigmoid.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** g(0.001) ≈ 0.5002 and g(-0.001) ≈ 0.4998. These two values are extremely close to each other (and both very close to 0.5). W and Z have the smallest difference in predicted probabilities because their z-scores are nearly equal in magnitude and opposite in sign, keeping both outputs near 0.5.
- **Option B (Incorrect):** Y (z = 5.0) has g(z) ≈ 0.993 and Z (z = -5.0) has g(z) ≈ 0.007. These are at opposite extremes, not close to each other. Large-magnitude z-scores push the sigmoid toward its asymptotes in opposite directions.
- **Option C (Incorrect):** W (z = 0.001) is near the decision boundary, but Z (z = -5.0) is far from it, producing g(z) ≈ 0.007. Their probabilities are far apart.
- **Option D (Incorrect):** "Opposing signs produce cancelling effects" mischaracterizes the sigmoid. Opposing signs produce values on opposite sides of 0.5. X (z = -0.001, g ≈ 0.4998) and Y (z = 5.0, g ≈ 0.993) are far apart in predicted probability.

</details>

---

### 54. Question

A developer wants to build a pipeline that trains a logistic regression model and then predicts whether new loan applications will be approved or rejected. She writes:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
predictions = model.predict(X_new)
model.fit(X_train, y_train)
```

**What is wrong with this code?**

- [ ] **A.** `model.predict()` cannot be called on a `LogisticRegression` object created without arguments.
- [ ] **B.** The `fit()` method must be called before `predict()`, but since both are defined in the same script, Python automatically handles the order.
- [ ] **C.** `model.predict()` is called before `model.fit()`. The model has not been trained yet, so calling `predict()` before `fit()` raises a `NotFittedError`.
- [ ] **D.** The code is correct; scikit-learn models cache the `predict()` call and execute it after `fit()` completes.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression()` can be instantiated without arguments; all parameters have valid defaults. The instantiation itself is correct.
- **Option B (Incorrect):** Python executes statements sequentially in the order they appear. It does not automatically reorder or defer method calls. `predict()` is called before `fit()` and therefore executes on an unfitted model.
- **Option C (Correct):** scikit-learn requires that a model be fitted (via `.fit()`) before calling `.predict()`. Calling `predict()` on an unfitted model raises `sklearn.exceptions.NotFittedError`, indicating that the model is not initialized. The fix is to move `model.fit(X_train, y_train)` before `model.predict(X_new)`.
- **Option D (Incorrect):** scikit-learn has no deferred execution or caching mechanism for method calls. Each method call executes immediately when Python reaches that line. There is no automatic reordering of operations.

</details>

---

### 55. Question

An analyst wants to determine whether a `LogisticRegression` model's predictions are influenced more by `age` or `income`. She notices: coefficient of `age` = 0.05, coefficient of `income` = 0.00003.

**Without additional context, which feature appears to have more influence on the linear score, and what important caveat applies?**

- [ ] **A.** `income` has more influence because it has many more data points associated with it.
- [ ] **B.** `age` appears to have more influence based on the raw coefficient magnitude (0.05 > 0.00003), but this comparison is only valid if both features are on the same scale; if `income` is in thousands, its effective contribution per meaningful unit may be comparable or larger.
- [ ] **C.** `income` has more influence because financial variables always dominate age-based variables in logistic models.
- [ ] **D.** The magnitudes are irrelevant; only the sign of the coefficient determines the feature's influence.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The number of data points associated with a feature does not determine its coefficient magnitude. Coefficients are fitted from the data during optimization and reflect the relationship between the feature and the log-odds of the target.
- **Option B (Correct):** Comparing raw coefficient magnitudes is only valid when features are on comparable scales. `age` (in years, typically 18–80) and `income` (in dollars, potentially 20,000–200,000) are on very different scales. A unit change in `age` (1 year) is meaningfully different from a unit change in `income` (1 dollar). If `income` were in thousands, a one-unit change would correspond to $1,000, and its effective contribution per realistic change in income could exceed that of `age`.
- **Option C (Incorrect):** There is no general rule that financial variables dominate age-based variables in logistic models. The coefficient values depend on the specific data, relationships, and scaling used.
- **Option D (Incorrect):** The magnitude of a coefficient determines the strength of the feature's effect on z; the sign determines the direction. Both are necessary for understanding influence. Magnitude alone, especially when features are scaled similarly, is the primary indicator of strength.

</details>

---

### 56. Question

A junior ML engineer accidentally calls `model.predict_proba()` on a `LinearRegression` object instead of a `LogisticRegression` object.

**What happens?**

- [ ] **A.** `LinearRegression.predict_proba()` returns raw continuous predictions normalized between 0 and 1.
- [ ] **B.** Python raises an `AttributeError` because `LinearRegression` does not have a `predict_proba()` method; this method is exclusive to classifier objects.
- [ ] **C.** `predict_proba()` works identically on both `LinearRegression` and `LogisticRegression`.
- [ ] **D.** `predict_proba()` on `LinearRegression` applies the sigmoid function to its continuous outputs before returning them.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `LinearRegression` has no normalization step that would bound its outputs to [0, 1]. The raw `predict()` output is unbounded.
- **Option B (Correct):** `predict_proba()` is part of scikit-learn's `ClassifierMixin` interface. `LinearRegression` inherits from `RegressorMixin` and does not implement `predict_proba()`. Calling it raises `AttributeError: 'LinearRegression' object has no attribute 'predict_proba'`.
- **Option C (Incorrect):** `predict_proba()` is only available on classifiers. It is not part of the regressor interface, so these two models have fundamentally different available methods.
- **Option D (Incorrect):** `LinearRegression` does not apply the sigmoid function to its outputs at any point. It is a purely linear model with no sigmoid step. The sigmoid is only used by `LogisticRegression`.

</details>

---

### 57. Question

A team is evaluating a binary classifier trained on a balanced dataset (50% class 0, 50% class 1). The model achieves `model.score()` = 0.72 on the test set.

**Which of the following correctly interprets this result?**

- [ ] **A.** 0.72 is the R² of the classifier, indicating it explains 72% of the variance in the labels.
- [ ] **B.** 0.72 means the classifier correctly predicted the label for 72% of all test samples, regardless of class.
- [ ] **C.** 0.72 means the classifier correctly identifies 72% of all class-1 samples (recall for class 1).
- [ ] **D.** 0.72 means 72% of the model's class-1 predictions were actually class-1 (precision for class 1).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** R² is returned by `LinearRegression.score()`, not by classifiers. For `LogisticRegression`, `score()` returns classification accuracy, not R².
- **Option B (Correct):** For a classifier, `model.score(X_test, y_test)` returns accuracy — the proportion of all test samples (both classes) for which the predicted label matches the true label. A score of 0.72 means 72 out of every 100 test samples were correctly classified.
- **Option C (Incorrect):** This describes recall (sensitivity) for class 1, which is the fraction of actual class-1 samples that were correctly predicted as class 1. Recall is computed using `recall_score()` from `sklearn.metrics`, not from `model.score()`.
- **Option D (Incorrect):** This describes precision for class 1 — the fraction of samples predicted as class 1 that are truly class 1. Precision is computed using `precision_score()` from `sklearn.metrics`.

</details>

---

### 58. Question

A logistic regression model has a very large positive intercept (e.g., intercept = 8.0) and all feature coefficients are near zero.

**What does this imply about the model's predictions?**

- [ ] **A.** The model will always predict label 0 because a large intercept shifts the decision boundary away from the data.
- [ ] **B.** The model will almost always predict label 1, since z ≈ 8.0 for most inputs (g(8.0) ≈ 1.0), regardless of feature values.
- [ ] **C.** A large positive intercept is invalid and causes scikit-learn to raise an error during fitting.
- [ ] **D.** The model will predict label 1 only for samples with feature values that exactly cancel the intercept.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A large positive intercept pushes z toward large positive values, which maps to g(z) ≈ 1.0. This predicts class 1, not class 0. A large positive intercept would cause the opposite of what option A suggests.
- **Option B (Correct):** With near-zero feature coefficients, the feature terms contribute negligible amounts to z. The intercept (8.0) dominates, making z ≈ 8.0 for most inputs. g(8.0) ≈ 0.9997, which is well above 0.5. The model therefore predicts label 1 for nearly every input, essentially ignoring the features.
- **Option C (Incorrect):** scikit-learn imposes no restrictions on the magnitude or sign of the intercept. Any real-valued intercept is mathematically valid; large intercepts are unusual but not erroneous from the library's perspective.
- **Option D (Incorrect):** The model predicts label 1 when g(z) > 0.5, i.e., when z > 0. With z ≈ 8.0, this condition is almost always satisfied without requiring any specific feature value combination. Cancelling a large positive intercept would require features with large negative contributions, which is not possible here since coefficients are near zero.

</details>

---

### 59. Question

A data scientist splits her dataset using `train_test_split` with `test_size=0.25`. She then fits a `LogisticRegression` model on `X_train`, evaluates it with `model.score(X_test, y_test)`, and gets 0.88.

A colleague suggests rerunning with a different `random_state` to get a higher score. 

**Which of the following is the most accurate assessment of this suggestion?**

- [ ] **A.** The suggestion is correct; a different random state will always produce a higher test accuracy.
- [ ] **B.** The suggestion is irrelevant because accuracy never changes with different random states for logistic regression.
- [ ] **C.** The suggestion is partially valid but problematic: changing `random_state` changes which samples go into the test set, and a higher score may reflect a favorable split rather than a genuinely better model.
- [ ] **D.** The suggestion is correct; the `random_state` controls the model's optimization convergence, so different values produce genuinely different models.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** There is no guarantee that a different random state will produce a higher score. Different splits can produce higher, lower, or similar test scores depending on which samples end up in the test set.
- **Option B (Incorrect):** Accuracy does vary across different random states because different splits expose the model to different test samples, some of which may be easier or harder for the model to classify correctly.
- **Option C (Correct):** Changing `random_state` changes the train/test partition. If the selected split happens to put easier-to-classify samples in the test set, the score will be higher — but this reflects a favorable data split, not an improvement in the model itself. Cherry-picking the random state that maximizes test accuracy is a form of data snooping and produces overly optimistic performance estimates. Cross-validation is the proper solution.
- **Option D (Incorrect):** `random_state` in `train_test_split` controls the shuffling used for data splitting, not the logistic regression optimization. The model fitting is deterministic for a given training set. The model itself doesn't change based on the split's random state; only which data it trains on changes.

</details>

---

### 60. Question

Consider a binary logistic regression problem where the target labels are `cat` (class 0) and `dog` (class 1). A trained model is called with `model.predict_proba([[3.2, 1.5]])` and returns `[[0.25, 0.75]]`.

**Which of the following correctly interprets this output?**

- [ ] **A.** The model is 25% confident it is a dog and 75% confident it is a cat.
- [ ] **B.** The first value (0.25) represents the probability of class 1 (dog) and the second value (0.75) represents the probability of class 0 (cat).
- [ ] **C.** The model predicts a 25% probability of class 0 (cat) and a 75% probability of class 1 (dog), and would classify this sample as a dog.
- [ ] **D.** The output means the model is 75% accurate on this dataset and should be trusted for 75% of predictions.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** This swaps the class labels. The output `[0.25, 0.75]` means P(class 0) = 0.25 and P(class 1) = 0.75. Since class 0 is `cat` and class 1 is `dog`, the model is 75% confident it is a dog, not a cat.
- **Option B (Incorrect):** This reverses the column order. scikit-learn's `predict_proba` always orders columns by ascending class label. The first column corresponds to the class with the smaller label (class 0 = cat, probability 0.25) and the second to class 1 (dog, probability 0.75).
- **Option C (Correct):** `predict_proba` returns `[P(class 0), P(class 1)]`. Here P(cat) = 0.25 and P(dog) = 0.75. Since 0.75 > 0.5 (default threshold), `predict()` would classify this sample as class 1 (dog).
- **Option D (Incorrect):** `predict_proba` returns per-sample class probabilities, not overall model accuracy. The 0.75 value is this specific sample's probability of being class 1, not a measure of the model's accuracy across the entire dataset.

</details>
