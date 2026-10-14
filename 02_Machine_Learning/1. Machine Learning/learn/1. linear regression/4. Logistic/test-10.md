# Logistic Regression Objective Test — Set 10

### 91. Question

A data science team debates whether the choice between `LogisticRegression` and `LinearRegression` matters when the target values are 0 and 1. One team member claims: "Both models can output values between 0 and 1 sometimes, so the choice doesn't really matter."

**Which of the following correctly rebuts this claim?**

- [ ] **A.** The claim is correct; when training data targets are only 0 and 1, both models always produce outputs in [0, 1].
- [ ] **B.** The claim is incorrect; `LinearRegression` can produce outputs outside [0, 1] for any input, not just the training set. For test samples, it may output values like 1.4 or -0.3, which are invalid probabilities. `LogisticRegression` guarantees outputs in (0, 1) via the sigmoid.
- [ ] **C.** The claim is correct for small datasets where the linear regression line naturally stays within the target range.
- [ ] **D.** The claim is incorrect only when the dataset has more than 1000 samples; for smaller datasets, both models behave identically.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `LinearRegression` is not guaranteed to produce outputs in [0, 1] even during training, and certainly not during testing. It fits an unbounded line to the data; for training samples close to the regression line, outputs may happen to be near 0 or 1, but for extrapolation or outliers, outputs can easily fall outside this range.
- **Option B (Correct):** The sigmoid function in `LogisticRegression` mathematically constrains its output to the open interval (0, 1) for any finite input. `LinearRegression` has no such constraint. On test samples with different feature distributions than the training set, `LinearRegression` will often produce values outside [0, 1], making it inappropriate for producing valid probabilities.
- **Option C (Incorrect):** The number of samples does not determine whether `LinearRegression` stays within [0, 1]. Even on small datasets, the fitted line can produce out-of-range predictions for new samples. The range of outputs depends on the model's structure (bounded vs. unbounded), not dataset size.
- **Option D (Incorrect):** There is no sample size threshold above or below which the two models behave identically. Their fundamental structural difference (sigmoid vs. no sigmoid) applies at any dataset size.

</details>

---

### 92. Question

A logistic regression model outputs the following for a test set of 4 samples:

| Sample | z (linear score) |
| :----- | :--------------- |
| 1      | 3.0              |
| 2      | -0.1             |
| 3      | 0.0              |
| 4      | -4.5             |

**Which samples are predicted as class 1 by `model.predict()`?**

- [ ] **A.** Only Sample 1, because it is the only sample with z > 0.
- [ ] **B.** Samples 1 and 2, because both have z values greater than -1.0.
- [ ] **C.** Samples 1, 2, and 3, because z ≥ 0 predicts class 1.
- [ ] **D.** All four samples, because any positive z is sufficient and negative z values only reduce probability slightly.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** The decision boundary in z-space is z = 0. For z > 0, g(z) > 0.5 → label 1. For z ≤ 0, g(z) ≤ 0.5 → label 0. Sample 1 (z = 3.0 > 0 → class 1). Sample 2 (z = -0.1 < 0 → class 0). Sample 3 (z = 0.0 → g(z) = 0.5 → class 0, because strict greater-than is required). Sample 4 (z = -4.5 < 0 → class 0). Only Sample 1 receives label 1.
- **Option B (Incorrect):** The threshold is not at z > -1.0. The probability threshold of 0.5 corresponds to z = 0. Sample 2 has z = -0.1, which is negative, giving g(-0.1) ≈ 0.475 < 0.5 → class 0.
- **Option C (Incorrect):** z ≥ 0 is not the correct rule. The strict greater-than requirement means only z > 0 predicts class 1. At z = 0 exactly, g(z) = 0.5, which does not exceed the threshold and maps to class 0.
- **Option D (Incorrect):** Negative z values do not "slightly reduce" probability — they reduce it below 0.5, which determines the class assignment. z = -4.5 gives g(z) ≈ 0.011, which is well below 0.5 and clearly predicts class 0.

</details>

---

### 93. Question

A machine learning course instructor asks students: "What is the relationship between the sigmoid function and the classification threshold in logistic regression?"

**Which response is most technically accurate?**

- [ ] **A.** The sigmoid and the threshold are the same thing; the sigmoid outputs directly determine the class without any additional step.
- [ ] **B.** The threshold is applied before the sigmoid: the linear score is thresholded at 0 first, then the sigmoid maps the result to a probability.
- [ ] **C.** The sigmoid transforms the linear score z into a probability in (0, 1); the threshold is then applied to this probability to make the binary classification decision. These are two separate, sequential steps.
- [ ] **D.** The sigmoid determines the optimal threshold automatically; the 0.5 threshold is simply read off the sigmoid curve at x = 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The sigmoid and the threshold are distinct operations. The sigmoid is a mathematical function that maps z to a probability. The threshold is a separate decision rule applied after the sigmoid to convert the probability to a class label.
- **Option B (Incorrect):** The order is reversed here. The sigmoid is applied first (to z, producing probability), and then the threshold is applied (to the probability, producing the class label). Applying the threshold to z first and then the sigmoid would be mathematically incorrect and would change the result.
- **Option C (Correct):** The correct pipeline is: (1) compute z = w·x + b (linear score); (2) apply sigmoid g(z) = 1/(1+e^(-z)) to get P(class 1) ∈ (0, 1); (3) compare P(class 1) to the threshold (default 0.5) to assign the class label. These are two logically separate operations that occur sequentially.
- **Option D (Incorrect):** The sigmoid does not automatically determine the threshold. The value 0.5 comes from the convention that g(0) = 0.5, and 0.5 is a natural midpoint for probability-based classification. The threshold is a design choice; 0.5 is simply the default, not something "read off" the sigmoid automatically.

</details>

---

### 94. Question

A team has built a logistic regression model to classify bank transactions as fraudulent (1) or legitimate (0). The model's `predict_proba()` output for a batch of transactions is:

```
[[0.95, 0.05],
 [0.02, 0.98],
 [0.60, 0.40],
 [0.48, 0.52]]
```

**Which transactions would be flagged as fraud using a threshold of 0.6 (instead of the default 0.5)?**

- [ ] **A.** Transactions 2 and 4.
- [ ] **B.** Transaction 2 only.
- [ ] **C.** Transactions 2, 3, and 4.
- [ ] **D.** All four transactions.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Transaction 4 has P(fraud) = 0.52, which is above 0.5 but below 0.6. Under the stricter 0.6 threshold, 0.52 does not exceed the threshold, so Transaction 4 is not flagged as fraud.
- **Option B (Correct):** Applying the 0.6 threshold to the class-1 (fraud) probabilities: Transaction 1 (0.05 < 0.6 → not fraud), Transaction 2 (0.98 ≥ 0.6 → fraud), Transaction 3 (0.40 < 0.6 → not fraud), Transaction 4 (0.52 < 0.6 → not fraud). Only Transaction 2 is flagged at this stricter threshold.
- **Option C (Incorrect):** Transaction 3 has P(fraud) = 0.40, which is below both 0.5 and 0.6. It would not be flagged as fraud under either threshold.
- **Option D (Incorrect):** Transaction 1 has P(fraud) = 0.05 and Transaction 3 has P(fraud) = 0.40 — neither exceeds the 0.6 threshold. Flagging all four transactions is incorrect.

</details>

---

### 95. Question

A developer writes code to compare two classifiers on the same test set:

```python
from sklearn.linear_model import LogisticRegression, LinearRegression

lr = LogisticRegression()
lin = LinearRegression()

lr.fit(X_train, y_train)
lin.fit(X_train, y_train)

print(lr.score(X_test, y_test))
print(lin.score(X_test, y_test))
```

A colleague sees that `lin.score()` outputs 0.68 and `lr.score()` outputs 0.85. She says: "Since R² = 0.68, the linear model explains more of the variance in the data."

**What is wrong with this interpretation?**

- [ ] **A.** Nothing; R² of 0.68 and accuracy of 0.85 can be directly compared to conclude linear regression explains more variance.
- [ ] **B.** The two scores represent different metrics. `lr.score()` returns classification accuracy (0.85 = 85% correct labels); `lin.score()` returns R² (0.68 = 68% of variance explained). These are incomparable metrics measuring different things.
- [ ] **C.** R² of 0.68 means the linear model is 68% accurate, so the logistic model (85%) is clearly better.
- [ ] **D.** Both scores represent accuracy, so comparing them is valid.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** R² and accuracy are fundamentally different metrics. Accuracy measures the fraction of correct predictions; R² measures the proportion of variance in the target variable explained by the model. Comparing them as if they measure the same thing is a category error.
- **Option B (Correct):** `LogisticRegression.score()` returns classification accuracy: 0.85 means 85% of test labels were correctly predicted. `LinearRegression.score()` returns R²: 0.68 means the linear model explains 68% of the variance in the 0/1 binary target. These metrics are measured on different scales and answer different questions; they cannot be compared to conclude which model is "better."
- **Option C (Incorrect):** R² is not the same as accuracy. An R² of 0.68 does not mean "68% of predictions are correct." These metrics have different interpretations. Treating R² as accuracy is a misconception.
- **Option D (Incorrect):** The two scores do not represent the same metric. `LogisticRegression.score()` returns classification accuracy, while `LinearRegression.score()` returns R². Comparing them as if they are the same is incorrect.

</details>

---

### 96. Question

A junior engineer is confused about when to use `stratify=y` in `train_test_split`. She asks: "Is stratification needed for every dataset, or only in specific situations?"

**Which of the following best answers her question?**

- [ ] **A.** Stratification is always required; omitting it causes `train_test_split` to raise a warning.
- [ ] **B.** Stratification is never needed when `random_state` is set, because a fixed seed guarantees balanced splits.
- [ ] **C.** Stratification is particularly important when the target variable is imbalanced — when one class is much more frequent than another — to ensure both splits reflect the same class proportions. For balanced datasets, the default random split usually produces similar proportions anyway.
- [ ] **D.** Stratification is only needed when `test_size` is less than 0.1, since small test sets are the only ones at risk of class imbalance.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Stratification is optional, not required. Omitting `stratify=y` does not produce a warning. The parameter defaults to `None`, in which case a simple random split is performed.
- **Option B (Incorrect):** Setting `random_state` makes the split reproducible but does not make it stratified. A fixed seed produces the same random (non-stratified) split each time. A randomly generated split may still under-represent minority classes.
- **Option C (Correct):** Stratification is most critical when the target is imbalanced. For example, with 95% class A and 5% class B, a random split might accidentally place most or all class-B samples in training or testing, making evaluation unreliable. For balanced datasets (e.g., 50/50 classes), random splits usually produce similar proportions without stratification, though stratification remains good practice.
- **Option D (Incorrect):** There is no specific threshold for `test_size` below which stratification becomes necessary. The risk of class under-representation depends on class imbalance and the total number of minority-class samples, not on the `test_size` fraction alone.

</details>

---

### 97. Question

A model is fitted and its coefficient for the feature `discount_offered` is found to be -0.90. A business analyst interprets this as: "Offering discounts hurts customer retention — a higher discount reduces the chance of renewal."

**Is this interpretation correct, and what does the coefficient actually imply?**

- [ ] **A.** Correct; a negative coefficient always means the feature has a harmful business effect.
- [ ] **B.** Incorrect; negative coefficients mean the feature has no predictive value.
- [ ] **C.** Not necessarily correct in the business sense. The coefficient of -0.90 means that as `discount_offered` increases, the linear score z decreases, which lowers the predicted probability of the positive class (renewal). But this could reflect a correlation where larger discounts are given to at-risk customers — the coefficient reflects association, not direct causation.
- [ ] **D.** Correct; logistic regression coefficients directly measure causal effects, so a coefficient of -0.90 definitively proves discounts reduce renewal.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** A negative coefficient indicates a negative association between the feature and the log-odds of the positive class, but "harmful business effect" is a causal claim. The coefficient reflects correlation (or association) in the training data, not a causal mechanism.
- **Option B (Incorrect):** A coefficient of -0.90 (non-zero) indicates the feature does have predictive value in the model. A feature with no predictive value would have a coefficient of 0 (or near 0, especially with L1 regularization).
- **Option C (Correct):** The technical interpretation is accurate: a larger `discount_offered` value leads to a lower z and lower predicted renewal probability. However, interpreting this causally as "discounts hurt retention" is premature. A likely confound is that discounts are typically offered to at-risk or dissatisfied customers. The coefficient captures this correlation, not the causal effect of discounts on retention. Causality requires controlled experiments or causal inference methods, not logistic regression coefficients alone.
- **Option D (Incorrect):** Logistic regression coefficients measure statistical association (conditioned on other variables in the model), not causal effects. Establishing causation requires randomized experiments or rigorous causal inference frameworks, not observational regression coefficients.

</details>

---

### 98. Question

A logistic regression model is trained and then the following code is run:

```python
import numpy as np

z = np.array([0.0, 1.0, -1.0, 10.0, -10.0])
sigmoid = 1 / (1 + np.exp(-z))
labels = (sigmoid > 0.5).astype(int)
print(labels)
```

**What does the code print?**

- [ ] **A.** `[1 1 0 1 0]`
- [ ] **B.** `[0 1 0 1 0]`
- [ ] **C.** `[1 1 1 1 0]`
- [ ] **D.** `[0 0 0 1 0]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** z = 0.0 gives sigmoid = 0.5, which is NOT strictly greater than 0.5, so label = 0, not 1. The first element should be 0.
- **Option B (Correct):** Computing sigmoid for each z: z=0.0 → g=0.5 → 0.5 > 0.5 is False → label 0; z=1.0 → g≈0.73 > 0.5 → label 1; z=-1.0 → g≈0.27 > 0.5 is False → label 0; z=10.0 → g≈1.0 > 0.5 → label 1; z=-10.0 → g≈0.0 > 0.5 is False → label 0. Result: `[0 1 0 1 0]`.
- **Option C (Incorrect):** z = -1.0 gives sigmoid ≈ 0.27, which is below 0.5, so label = 0 not 1. The third element should be 0.
- **Option D (Incorrect):** z = 1.0 gives sigmoid ≈ 0.73 > 0.5 → label 1, not 0. The second element should be 1.

</details>

---

### 99. Question

A team trains a `LogisticRegression` model and examines `model.predict_proba(X_test)`. They notice that for sample k, the output is `[0.5, 0.5]`.

**Which of the following correctly explains this output and its classification?**

- [ ] **A.** The model is completely random for this sample; `[0.5, 0.5]` indicates that no learning occurred.
- [ ] **B.** The sample lies exactly on the decision boundary (z = 0), giving equal probability to both classes. Under the strict greater-than threshold, it is classified as class 0.
- [ ] **C.** The output `[0.5, 0.5]` is impossible; scikit-learn ensures probabilities never exactly equal 0.5 in practice.
- [ ] **D.** The model assigns equal probability to both classes, and `predict()` would return class 1 for this sample since the classes are equal and it defaults to the higher class.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `[0.5, 0.5]` does not indicate random behavior or failed learning. It means the model's linear score z = 0 for this specific sample, placing it exactly on the mathematical decision boundary. The model is functioning correctly; this particular sample happens to fall at the boundary.
- **Option B (Correct):** g(0) = 0.5 exactly, which occurs when z = 0 (i.e., the linear combination of features and weights equals zero). This places the sample exactly on the decision boundary where both classes have equal probability. Under scikit-learn's strict greater-than rule (P(class 1) > 0.5), a probability of exactly 0.5 does not meet the threshold and is assigned to class 0.
- **Option C (Incorrect):** While `[0.5, 0.5]` is rare in practice (it requires z = 0 exactly, which depends on the exact feature values and fitted weights), it is mathematically possible and scikit-learn does not prevent it. Floating-point rounding makes it uncommon but not impossible.
- **Option D (Incorrect):** When class-1 probability equals 0.5, scikit-learn does not default to the "higher class" or use any tie-breaking rule that favors class 1. The strict greater-than threshold assigns class 0 when P(class 1) = 0.5.

</details>

---

### 100. Question

A complete classification pipeline in Python is shown below. Identify all issues:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset: X (features), y (binary labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

probabilities = LR_model.predict_proba(x_test)
labels = lr_model.predict(X_test)
accuracy = LR_model.score(x_test, y_test)

print(labels, accuracy)
```

**Which of the following correctly identifies all the bugs in this code?**

- [ ] **A.** Only one bug: `x_test` (lowercase x) on line 9 should be `X_test`.
- [ ] **B.** Two bugs: `x_test` on line 9 and `lr_model` on line 10 are both undefined identifiers.
- [ ] **C.** Three bugs: `x_test` on line 9 (should be `X_test`), `lr_model` on line 10 (should be `LR_model`), and `x_test` on line 11 (should be `X_test`).
- [ ] **D.** No bugs; Python automatically resolves near-matches to the correct variable names.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** There are more than one bug. While `x_test` on line 9 is indeed a bug (`NameError: name 'x_test' is not defined`), `lr_model` on line 10 and `x_test` on line 11 are additional distinct errors. Identifying only one bug is incomplete.
- **Option B (Incorrect):** Two bugs is closer but still incomplete. Line 9 has `x_test`, line 10 has `lr_model`, and line 11 has `x_test` again — that is three separate occurrences of undefined variable references. Each would cause a `NameError` when Python reaches that line.
- **Option C (Correct):** Three distinct bugs: (1) Line 9: `x_test` should be `X_test` — Python's case-sensitivity means `x_test` and `X_test` are different identifiers; `x_test` was never defined. (2) Line 10: `lr_model` should be `LR_model` — the model was stored as `LR_model` (uppercase LR), not `lr_model`. (3) Line 11: `x_test` again should be `X_test`. Each of these three errors would raise a `NameError` at runtime.
- **Option D (Incorrect):** Python never automatically resolves "near-match" or case-variant variable names. Its identifier lookup is exact and case-sensitive. There are no auto-correction mechanisms for undefined variable names in standard Python execution.

</details>
