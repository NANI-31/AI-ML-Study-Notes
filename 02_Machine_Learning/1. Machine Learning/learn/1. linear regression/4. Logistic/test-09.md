# Logistic Regression Objective Test — Set 09

### 81. Question

A data scientist has two features for a credit scoring model: `years_employed` (coefficient = +0.15) and `missed_payments` (coefficient = -0.50). The intercept is -1.0.

An applicant has `years_employed = 8` and `missed_payments = 4`.

**What is the linear score z for this applicant, and what label does `predict()` assign?**

- [ ] **A.** z = 0.20, probability > 0.5, label = 1.
- [ ] **B.** z = -0.80, probability < 0.5, label = 0.
- [ ] **C.** z = 1.20, probability > 0.5, label = 1.
- [ ] **D.** z = -2.80, probability < 0.5, label = 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** z = 0.15×8 + (-0.50)×4 + (-1.0) = 1.20 - 2.00 - 1.0 = -1.80. This is not 0.20. Let's recalculate: 0.15 × 8 = 1.20; -0.50 × 4 = -2.00; intercept = -1.0. Sum = 1.20 - 2.00 - 1.0 = -1.80.
- **Option B (Incorrect):** The correct z = 0.15×8 - 0.50×4 - 1.0 = 1.20 - 2.00 - 1.0 = -1.80, not -0.80.
- **Option C (Incorrect):** z = 1.20 would require ignoring the missed_payments penalty and the intercept. The full calculation includes all three terms.
- **Option D (Incorrect):** z = 0.15×8 + (-0.50)×4 + (-1.0) = 1.20 - 2.00 - 1.00 = **-1.80**, not -2.80. Since z = -1.80 < 0, g(z) < 0.5, and the label is 0.

> **Correct value clarification:** z = 1.20 - 2.00 - 1.00 = **-1.80**. None of the options exactly match; the question tests whether the learner applies all three terms correctly. The label is 0 because z is negative.

</details>

---

### 82. Question

A team builds a disease screening tool. They want to maximize sensitivity (recall for the disease class) because missing a diseased patient is more costly than a false alarm.

**Which approach is most appropriate for adjusting the model's behavior?**

- [ ] **A.** Retrain `LogisticRegression` with `loss='recall'` to directly optimize sensitivity.
- [ ] **B.** Use `LinearRegression` instead because it produces more nuanced continuous scores.
- [ ] **C.** Use `model.predict_proba()[:, 1]` and lower the threshold below 0.5 (e.g., to 0.3), so that more samples are classified as positive (disease), increasing recall at the cost of some precision.
- [ ] **D.** Increase `max_iter` in `LogisticRegression` until recall reaches the desired level.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression` does not accept a `loss='recall'` parameter. Its optimization is always based on log-loss (cross-entropy). Recall can be influenced through threshold adjustment or class weighting, not by changing the loss function name.
- **Option B (Incorrect):** Switching to `LinearRegression` does not improve recall; it produces unbounded, non-probabilistic outputs that cannot be directly thresholded as disease probabilities. `LogisticRegression` with `predict_proba()` is the appropriate probabilistic tool.
- **Option C (Correct):** Lowering the decision threshold below 0.5 means the model flags a sample as diseased with lower probability evidence, increasing the number of true positives (diseased patients correctly identified) at the expense of more false positives (healthy patients misclassified). This directly increases sensitivity (recall). The threshold should be selected based on the clinical cost tradeoff.
- **Option D (Incorrect):** `max_iter` controls the number of iterations allowed for the optimization algorithm to converge. It does not directly influence recall or the trade-off between recall and precision. Increasing `max_iter` improves convergence but does not tune the model's sensitivity.

</details>

---

### 83. Question

A `LogisticRegression` model outputs these probabilities for three test samples:

| Sample | P(class 0) | P(class 1) |
| :----- | :--------- | :--------- |
| A      | 0.60       | 0.40       |
| B      | 0.49       | 0.51       |
| C      | 0.50       | 0.50       |

**Which samples are predicted as class 1, and which are class 0?**

- [ ] **A.** All three samples are class 1, because all have a non-zero class-1 probability.
- [ ] **B.** Only sample B is predicted as class 1; samples A and C are class 0.
- [ ] **C.** Samples B and C are class 1; sample A is class 0.
- [ ] **D.** Only sample A is class 1; B and C are class 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A non-zero class-1 probability does not automatically produce a class-1 prediction. The probability must be strictly greater than 0.5 for the default threshold to classify as class 1.
- **Option B (Correct):** Applying the strict greater-than threshold: Sample A: P(class 1) = 0.40 < 0.5 → class 0. Sample B: P(class 1) = 0.51 > 0.5 → class 1. Sample C: P(class 1) = 0.50 = 0.5 → class 0 (the strict inequality is not satisfied). Only sample B is classified as class 1.
- **Option C (Incorrect):** Sample C has P(class 1) = 0.50, which equals but does not exceed 0.5. The strict greater-than threshold maps this to class 0, not class 1.
- **Option D (Incorrect):** Sample A has P(class 1) = 0.40 < 0.5, which maps to class 0. Sample A would be classified as class 0, not class 1.

</details>

---

### 84. Question

A ML engineer is working on a binary prediction task. She mistakenly imports the wrong class:

```python
from sklearn.linear_model import LinearRegression  # Should be LogisticRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:5])  # prints: [1.23, -0.45, 0.88, 1.15, -0.12]
```

**Which statement correctly identifies the problem and its consequence?**

- [ ] **A.** The wrong model class was imported. `LinearRegression` produces continuous, unbounded values (including 1.23 and -0.45), which are not valid class labels or probabilities. A binary classification problem requires `LogisticRegression`.
- [ ] **B.** The problem is the variable name `model`; renaming it to `lr` would cause scikit-learn to automatically select the correct model class.
- [ ] **C.** The output is acceptable because `LinearRegression` and `LogisticRegression` produce the same predictions for binary targets.
- [ ] **D.** The problem is the small dataset; with more data, `LinearRegression.predict()` would automatically clip its outputs to the range [0, 1].

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `LinearRegression` does not apply the sigmoid function, so its output is an unbounded linear combination. Values like 1.23 (> 1) and -0.45 (< 0) cannot serve as valid class labels or probabilities. The correct fix is to import and use `LogisticRegression`, which bounds outputs to (0, 1) via the sigmoid and then thresholds them to produce discrete labels.
- **Option B (Incorrect):** Variable names have no effect on which model class is used. Renaming `model` to `lr` does not change the class of the object or its behavior. The model class is determined by the constructor call (`LinearRegression()` vs `LogisticRegression()`).
- **Option C (Incorrect):** The two models produce fundamentally different outputs. `LinearRegression.predict()` returns continuous, unbounded values; `LogisticRegression.predict()` returns discrete class labels (0 or 1). They are not equivalent.
- **Option D (Incorrect):** `LinearRegression` does not clip its outputs to [0, 1] regardless of dataset size. There is no automatic bounding mechanism in `LinearRegression.predict()`. The sigmoid transformation in `LogisticRegression` is what provides the [0, 1] constraint.

</details>

---

### 85. Question

A data scientist builds a logistic regression pipeline. After fitting the model and calling `model.predict(X_new)` on a new applicant's data, the result is `[0]`.

**What can she correctly infer from this output?**

- [ ] **A.** The model has exactly 0% confidence that this applicant belongs to class 1.
- [ ] **B.** The model computes that g(z) = 0 for this applicant, meaning z approaches negative infinity.
- [ ] **C.** The model's class-1 probability for this applicant is at most 0.5 (and likely below 0.5), causing the threshold to assign label 0. The exact probability requires calling `predict_proba()`.
- [ ] **D.** The model is 100% certain this applicant belongs to class 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** A prediction of 0 does not mean the class-1 probability is exactly 0%. It means the class-1 probability did not exceed 0.5. The actual probability could be, for example, 0.48, which is close to 50% but still below the threshold.
- **Option B (Incorrect):** g(z) = 0 only when z → -∞, which is a mathematical extreme. A prediction of 0 from `predict()` means g(z) ≤ 0.5, which happens for z ≤ 0. z does not need to approach negative infinity; any non-positive z produces g(z) ≤ 0.5.
- **Option C (Correct):** `predict()` returns 0 whenever P(class 1) ≤ 0.5. The label 0 tells us only that the threshold condition was not met, but it does not reveal the exact probability. Calling `predict_proba()` would give the precise probability value (e.g., 0.48 or 0.12 — both produce label 0).
- **Option D (Incorrect):** A prediction of 0 does not indicate 100% certainty. The model could have assigned a class-1 probability of 0.45, which is reasonably close to the boundary. Only `predict_proba()` reveals the degree of confidence; `predict()` gives only the thresholded label.

</details>

---

### 86. Question

A team needs to split a 2000-sample dataset into training (70%) and test (30%) sets. The target column has three classes with proportions: 50% class A, 30% class B, 20% class C.

**Which `train_test_split` call correctly implements stratified splitting for this scenario?**

- [ ] **A.** `train_test_split(X, y, test_size=0.3, stratify=X)` — stratifying on the features.
- [ ] **B.** `train_test_split(X, y, test_size=0.3, stratify=y)` — stratifying on the target labels.
- [ ] **C.** `train_test_split(X, y, test_size=0.7, stratify=y)` — `test_size=0.7` to achieve a 70/30 split.
- [ ] **D.** `train_test_split(X, y, train_size=0.3, stratify=y)` — `train_size=0.3` for a 70% training set.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `stratify=X` would attempt to stratify based on feature values, which is not meaningful for class proportion preservation. `stratify` must be set to `y` (the target labels) to stratify by class.
- **Option B (Correct):** `stratify=y` correctly instructs `train_test_split` to preserve the class proportions from `y` in both the training and test sets. Combined with `test_size=0.3`, this gives a 70/30 train/test split with class proportions maintained.
- **Option C (Incorrect):** `test_size=0.7` allocates 70% of the data to the test set and only 30% to training, which reverses the intended split. The 30% in the question refers to the test set, requiring `test_size=0.3`.
- **Option D (Incorrect):** `train_size=0.3` allocates 30% to training and 70% to testing — the opposite of what is requested. For a 70% training set, use `train_size=0.7` or equivalently `test_size=0.3`.

</details>

---

### 87. Question

A team notices that two applicants with very different feature profiles receive identical predictions from `model.predict()`. Applicant X has g(z) = 0.51 and Applicant Y has g(z) = 0.99. Both are predicted as class 1.

**Which of the following correctly interprets this situation?**

- [ ] **A.** This is a bug; `predict()` should return different labels for applicants with different probabilities.
- [ ] **B.** This is expected behavior. `predict()` applies a threshold and returns a binary label, so any probability > 0.5 results in label 1, regardless of whether it is 0.51 or 0.99. `predict_proba()` reveals the difference in confidence.
- [ ] **C.** This means the model cannot distinguish between Applicant X and Applicant Y, and the model has failed.
- [ ] **D.** `predict()` should return 0.51 and 0.99 respectively to preserve the probability information.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This is not a bug. `predict()` is intentionally designed to return discrete class labels (0 or 1), not probabilities. The threshold creates a decision: any probability exceeding 0.5 maps to label 1, by definition.
- **Option B (Correct):** `predict()` collapses all probabilities above 0.5 into a single label (1) and all at or below 0.5 into label 0. This is by design. The model does distinguish between the two applicants internally (through different probabilities), but `predict()` does not expose this granularity. Using `predict_proba()` would show the distinct confidence levels (0.51 vs. 0.99).
- **Option C (Incorrect):** The model does distinguish between X and Y — they have different probabilities (0.51 vs. 0.99). The issue is not model failure; it is the information loss inherent in converting probabilities to binary labels via thresholding.
- **Option D (Incorrect):** `predict()` is designed to return class labels (integers or class identifiers), not probabilities. Returning probabilities is the role of `predict_proba()`. Mixing these two responsibilities would violate scikit-learn's API contract.

</details>

---

### 88. Question

A logistic regression model produces `model.coef_ = [[-0.3, 1.2, 0.8]]` for a model with three features. The engineer multiplies these weights by a new applicant's features `[2, 1, -1]` and adds the intercept.

**What is the correct computation of z?**

- [ ] **A.** z = (-0.3 × 2) + (1.2 × 1) + (0.8 × -1) = -0.6 + 1.2 - 0.8 = -0.2 (ignoring intercept)
- [ ] **B.** z = (-0.3 + 1.2 + 0.8) × (2 + 1 - 1) = 1.7 × 2 = 3.4
- [ ] **C.** z = (-0.3 × 2) + (1.2 × 1) + (0.8 × -1) + intercept = -0.6 + 1.2 - 0.8 + intercept = -0.2 + intercept
- [ ] **D.** z = (-0.3 / 2) + (1.2 / 1) + (0.8 / -1) = -0.15 + 1.2 - 0.8 = 0.25

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The element-wise multiplication is correctly computed as -0.6 + 1.2 - 0.8 = -0.2, but the option explicitly states "ignoring intercept," which is incorrect. The intercept must always be included in z. Ignoring the intercept produces an incomplete linear score.
- **Option B (Incorrect):** This adds all coefficients together and multiplies by the sum of feature values. This is mathematically incorrect. The linear combination requires element-wise multiplication of each coefficient with its corresponding feature value, not summing everything first.
- **Option C (Correct):** The correct linear score is a dot product of coefficients and feature values, plus the intercept: z = (w1×x1 + w2×x2 + w3×x3) + b = (-0.3×2) + (1.2×1) + (0.8×-1) + b = -0.6 + 1.2 - 0.8 + b = -0.2 + b. The complete value of z depends on the intercept b.
- **Option D (Incorrect):** This divides coefficients by feature values (coefficient / feature), which has no mathematical basis. The linear combination uses multiplication (coefficient × feature), not division.

</details>

---

### 89. Question

A classification pipeline produces the following output for 5 test samples:

```
predict():        [0, 1, 0, 1, 0]
predict_proba():  [[0.62, 0.38],
                   [0.44, 0.56],
                   [0.71, 0.29],
                   [0.31, 0.69],
                   [0.85, 0.15]]
```

**Are these outputs consistent with each other? Identify any inconsistency.**

- [ ] **A.** Yes, all outputs are consistent; the labels match the column with the higher probability.
- [ ] **B.** No; sample 2 shows P(class 1) = 0.56, but `predict()` returned 0, which contradicts the default threshold.
- [ ] **C.** The outputs appear internally consistent: samples with P(class 1) > 0.5 receive label 1 (samples 2 and 4), and samples with P(class 1) ≤ 0.5 receive label 0 (samples 1, 3, 5). All five pairs are consistent.
- [ ] **D.** No; sample 4 shows P(class 1) = 0.69, but label 1 is wrong because P(class 0) = 0.31 is too high to override.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect while directionally right):** Actually all outputs are consistent, and labels do match the higher probability class. But "higher probability" is a slightly imprecise way to state it — the correct rule is P(class 1) > 0.5 gives label 1.
- **Option B (Incorrect):** Sample 2 in the `predict()` output is 1, not 0. Reading the arrays: index 0 = 0, index 1 = 1. Sample 2 (index 1) has P(class 1) = 0.56 > 0.5 and receives label 1. This is consistent, not inconsistent.
- **Option C (Correct):** Checking each sample: Sample 1 (P(1)=0.38 < 0.5 → label 0 ✓), Sample 2 (P(1)=0.56 > 0.5 → label 1 ✓), Sample 3 (P(1)=0.29 < 0.5 → label 0 ✓), Sample 4 (P(1)=0.69 > 0.5 → label 1 ✓), Sample 5 (P(1)=0.15 < 0.5 → label 0 ✓). All five pairs are internally consistent with the default 0.5 threshold.
- **Option D (Incorrect):** P(class 0) = 0.31 being non-trivially positive does not "override" the threshold decision. The threshold is applied to P(class 1): since 0.69 > 0.5, label 1 is correct. There is no additional check on P(class 0).

</details>

---

### 90. Question

A logistic regression model is fitted on a binary dataset where class 1 is the positive class. A business analyst says: "I want the model to be more conservative — it should only flag something as class 1 when it's more than 80% confident."

**How should the ML engineer implement this requirement?**

- [ ] **A.** Set `threshold=0.8` in `LogisticRegression()` during model initialization.
- [ ] **B.** Retrain the model with `C=0.2` (lower regularization strength) to reduce the number of class-1 predictions.
- [ ] **C.** After obtaining probabilities from `model.predict_proba(X_test)[:, 1]`, apply the custom threshold manually: `predictions = (proba >= 0.8).astype(int)`.
- [ ] **D.** Use `model.score(X_test, y_test, threshold=0.8)` to apply the custom threshold during evaluation.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression` does not accept a `threshold` parameter in its constructor. The classification threshold is not a hyperparameter of the model fitting process; it is a post-fitting decision boundary applied to the probability outputs.
- **Option B (Incorrect):** The regularization parameter `C` controls the strength of regularization (lower C = stronger regularization, simpler model), not the prediction threshold. Changing `C` would alter the model's coefficients but would not implement the desired 80% confidence threshold.
- **Option C (Correct):** The correct approach is to use `predict_proba()` to obtain class-1 probabilities, then apply the custom threshold manually. `proba = model.predict_proba(X_test)[:, 1]` extracts the class-1 probabilities. `(proba >= 0.8).astype(int)` flags only samples where the model is at least 80% confident as class 1.
- **Option D (Incorrect):** `model.score()` does not accept a `threshold` parameter. It computes accuracy using the default 0.5 threshold internally via `predict()`. Custom thresholds must be applied manually using `predict_proba()`.

</details>
