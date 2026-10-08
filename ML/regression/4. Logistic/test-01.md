# Logistic Regression Objective Test — Set 01

### 1. Question

A data scientist computes the sigmoid of five different z-scores before comparing the results to a 0.5 classification threshold.

| z-score | g(z) (approximate) |
| :------ | :------------------ |
| 6.0     | ?                   |
| -6.0    | ?                   |
| 0.5     | ?                   |
| -0.5    | ?                   |
| 0.0     | ?                   |

**Which of the following correctly pairs the z-scores with their approximate sigmoid outputs?**

- [ ] **A.** `g(6.0) ≈ 0`, `g(-6.0) ≈ 1`, `g(0.0) = 0.5`
- [ ] **B.** `g(6.0) ≈ 1`, `g(-6.0) ≈ 0`, `g(0.0) = 0.5`
- [ ] **C.** `g(6.0) ≈ 0.5`, `g(-6.0) ≈ 0.5`, `g(0.0) = 1`
- [ ] **D.** `g(6.0) ≈ 1`, `g(-6.0) ≈ 1`, `g(0.0) = 0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This reverses the behavior of the sigmoid. Large positive z drives `e^(-z)` toward 0, making `g(z) ≈ 1`. Large negative z drives `e^(-z)` toward infinity, making `g(z) ≈ 0`. Option A swaps these two outcomes.
- **Option B (Correct):** `g(6.0) ≈ 1` because `e^(-6)` is nearly 0, so `1/(1+0) ≈ 1`. `g(-6.0) ≈ 0` because `e^6` is very large. `g(0.0) = 1/(1+1) = 0.5` exactly. All three pairings are correct.
- **Option C (Incorrect):** The sigmoid does not output 0.5 for large-magnitude z-scores; 0.5 is only the exact output at z = 0. Large z-scores push the output toward the extremes (0 or 1), not toward the midpoint.
- **Option D (Incorrect):** Both `g(6.0)` and `g(-6.0)` cannot both be approximately 1. The sigmoid is asymmetric around z = 0; positive z gives output close to 1 and negative z gives output close to 0.

</details>

---

### 2. Question

A logistic regression model for email spam detection has the following linear score equation:

`z = 0.8 × (word_count) + 1.2 × (has_link) - 0.5 × (is_reply) - 1.0`

An email arrives with `word_count = 0`, `has_link = 0`, and `is_reply = 0`.

**What is the linear score z for this email, and how does it classify under a 0.5 threshold?**

- [ ] **A.** `z = 1.0`, probability > 0.5, classified as spam.
- [ ] **B.** `z = -1.0`, probability < 0.5, classified as not spam.
- [ ] **C.** `z = 0.0`, probability = 0.5, classified as not spam.
- [ ] **D.** `z = -0.5`, probability < 0.5, classified as not spam.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Substituting the values gives `z = 0.8×0 + 1.2×0 - 0.5×0 - 1.0 = -1.0`, not +1.0. The intercept is subtracted, not added.
- **Option B (Correct):** All feature values are 0, so each weighted term evaluates to 0. Only the intercept (`-1.0`) remains, giving `z = -1.0`. A negative z means `g(z) < 0.5`, so the email is classified as not spam.
- **Option C (Incorrect):** z = 0.0 would only occur if all feature contributions exactly cancelled the intercept. Here the intercept is -1.0 and all feature terms contribute 0, so z = -1.0, not 0.0.
- **Option D (Incorrect):** The intercept alone is -1.0; no feature contributes anything since all are 0. `z = -0.5` would require a different intercept value.

</details>

---

### 3. Question

A machine learning engineer fits a `LogisticRegression` model and examines the output of `model.predict_proba(X_test)`. One row returns `[0.34, 0.66]`.

**Which of the following statements correctly interpret this output?**

- [ ] **A.** The model assigns a 66% probability of class 1 and a 34% probability of class 0, and predicts label 1.
- [ ] **B.** The model assigns a 66% probability of class 1 and a 34% probability of class 0, and predicts label 0.
- [ ] **C.** The two values are independent estimates; they do not need to sum to 1.
- [ ] **D.** The first column always represents the probability of a positive outcome regardless of class labeling.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `predict_proba` returns `[P(class 0), P(class 1)]`. Here P(class 0) = 0.34 and P(class 1) = 0.66. Since 0.66 > 0.5 (the default threshold), `predict()` outputs label 1.
- **Option B (Incorrect):** Because the class-1 probability (0.66) exceeds the default threshold of 0.5, the model predicts label 1, not label 0.
- **Option C (Incorrect):** The two probabilities must sum to exactly 1.0 because they represent mutually exclusive and exhaustive outcomes for a binary classifier. They are mathematically dependent, not independent.
- **Option D (Incorrect):** The first column of `predict_proba` always corresponds to the lowest class label (class 0 in binary classification), not to a "positive" outcome in a general sense. The class ordering depends on `model.classes_`.

</details>

---

### 4. Question

Two analysts are debating the sigmoid function's output range. Analyst X claims the output can be exactly 0 or exactly 1 for extreme inputs. Analyst Y disagrees, saying the output only approaches but never actually reaches those values.

**Who is correct, and why?**

- [ ] **A.** Analyst X is correct: for z = 100, Python's floating-point arithmetic computes g(100) as exactly 1.0, making it mathematically reachable.
- [ ] **B.** Both are partially correct: the mathematical sigmoid never reaches 0 or 1, but discrete implementations always return exactly 0.5.
- [ ] **C.** Analyst Y is correct in all contexts: `g(z) = 1/(1 + e^(-z))` never equals exactly 0 or 1 for any finite z, since `e^(-z)` is always positive.
- [ ] **D.** Analyst Y is mathematically correct (the true sigmoid asymptotically approaches 0 and 1), but Analyst X is also practically correct in that floating-point representations will round extreme values to exactly 0.0 or 1.0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** This is only half-true and misses the mathematical context. Stating Analyst X is simply "correct" without acknowledging the asymptotic nature is mathematically inaccurate.
- **Option B (Incorrect):** Discrete implementations do not always return 0.5; they return floating-point approximations that can be 0.0 or 1.0 for very extreme z values due to numerical precision limits.
- **Option C (Incorrect):** While mathematically true in pure mathematics, this ignores the practical reality of floating-point arithmetic, where extreme z values (e.g., z = 800) cause `e^(-800)` to underflow to 0.0, making the computed output exactly 1.0.
- **Option D (Correct):** Mathematically, `1/(1 + e^(-z))` only asymptotically approaches 0 and 1. However, in finite-precision floating-point arithmetic, `e^(-z)` underflows to 0.0 for sufficiently large z, causing the computed output to become exactly 1.0. Both perspectives are valid in their respective contexts.

</details>

---

### 5. Question

A classifier is trained to detect fraudulent transactions. The positive class (fraud) is labeled 1, and the legitimate class is labeled 0. The model produces these probabilities for five transactions:

| Transaction | P(class 1) |
| :---------- | :--------- |
| T1          | 0.49       |
| T2          | 0.51       |
| T3          | 0.50       |
| T4          | 0.01       |
| T5          | 0.99       |

**Which transactions does `model.predict()` classify as fraud (label 1) using the default threshold?**

- [ ] **A.** T1, T2, T3, T5
- [ ] **B.** T2, T3, T5
- [ ] **C.** T2, T5
- [ ] **D.** T1, T2, T5

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** T1 (0.49) and T3 (0.50) do not exceed the default threshold of 0.5 strictly. T1 is below 0.5, and T3 equals 0.5 exactly, which maps to label 0 under the strict greater-than rule.
- **Option B (Incorrect):** T3 has a probability of exactly 0.50. Under scikit-learn's default `predict()` behavior, the threshold requires the class-1 probability to be strictly greater than 0.5; a value of exactly 0.5 yields label 0.
- **Option C (Correct):** T2 (0.51 > 0.5 → label 1) and T5 (0.99 > 0.5 → label 1) are the only two transactions strictly exceeding 0.5. T1 (0.49), T3 (0.50), and T4 (0.01) all yield label 0.
- **Option D (Incorrect):** T1 has a class-1 probability of 0.49, which is below 0.5, so it is classified as label 0 (not fraud), not label 1.

</details>

---

### 6. Question

A developer writes the following code to fit and evaluate a logistic regression model:

```python
from sklearn.linear_model import LogisticRegression

Model = LogisticRegression()
Model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(acc)
```

**What happens when this code executes?**

- [ ] **A.** The code runs correctly and prints the classification accuracy of the model.
- [ ] **B.** Python raises a `NameError` on the `model.score(...)` line because `model` is undefined; the fitted object is stored as `Model`.
- [ ] **C.** scikit-learn silently maps `model` to `Model` because both refer to a `LogisticRegression` object in the same scope.
- [ ] **D.** The code raises an error on `Model.fit(X_train, y_train)` because the class must be instantiated as `model` (lowercase) by convention.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The code cannot run to completion because `model` (lowercase) is never defined. The `score()` call raises a `NameError` before any output is produced.
- **Option B (Correct):** Python is strictly case-sensitive. `Model` (capital M) and `model` (lowercase m) are two entirely different identifiers. Since only `Model` was assigned, referencing `model` raises `NameError: name 'model' is not defined`.
- **Option C (Incorrect):** Python's namespace lookup has no concept of "similar" variable names. It performs exact string matching on identifiers. scikit-learn has no access to Python's variable namespace and cannot perform any such mapping.
- **Option D (Incorrect):** scikit-learn has no requirement that `LogisticRegression` be instantiated with a lowercase variable name. The object can be assigned to any valid Python identifier. The error occurs on the `model.score` line, not during instantiation.

</details>

---

### 7. Question

A logistic regression model is trained on a loan dataset. The coefficient for the feature `income` is +0.003 and the coefficient for `debt_ratio` is -0.75.

**Which of the following correctly describes how increasing each feature by one unit affects the linear score z?**

- [ ] **A.** Increasing `income` by 1 decreases z by 0.003; increasing `debt_ratio` by 1 decreases z by 0.75.
- [ ] **B.** Increasing `income` by 1 increases z by 0.003; increasing `debt_ratio` by 1 decreases z by 0.75.
- [ ] **C.** Increasing `income` by 1 increases z by 0.003; increasing `debt_ratio` by 1 increases z by 0.75.
- [ ] **D.** The sign of a coefficient determines the feature's importance, not its directional effect on z.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `income` has a positive coefficient (+0.003), which means an increase in `income` adds to z. Only `debt_ratio` with its negative coefficient (-0.75) reduces z.
- **Option B (Correct):** In the linear combination `z = w1×income + w2×debt_ratio + ...`, a one-unit increase in `income` adds `+0.003` to z, and a one-unit increase in `debt_ratio` adds `-0.75` to z (reduces z by 0.75).
- **Option C (Incorrect):** `debt_ratio` has a negative coefficient (-0.75). A one-unit increase in `debt_ratio` reduces z by 0.75; it does not increase z.
- **Option D (Incorrect):** A coefficient's sign directly encodes the direction of the effect on z: positive means increase, negative means decrease. The magnitude reflects the strength of the effect. Both sign and magnitude matter.

</details>

---

### 8. Question

A team builds a model to classify customer churn (will churn = 1, will not churn = 0). They find that 95% of customers did not churn, and they need to split data for training and evaluation.

**Which approach correctly addresses this class imbalance during the train-test split?**

- [ ] **A.** Use `test_size=0.95` to allocate more data to training and mitigate the imbalance.
- [ ] **B.** Use `shuffle=False` so that churned customers are kept together and not separated into different splits.
- [ ] **C.** Use `stratify=y` so that both the training and test sets maintain the same 95/5 class ratio as the original dataset.
- [ ] **D.** Class imbalance is irrelevant to `train_test_split`; the issue should only be addressed after the split using oversampling techniques.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Allocating more data to training does not change the class distribution. A 95/5 imbalance at the dataset level will remain 95/5 in any random subset, regardless of `test_size`.
- **Option B (Incorrect):** Disabling shuffle splits rows in their original order, which may worsen the imbalance if data is sorted by target label. It does not guarantee proportional class representation in each split.
- **Option C (Correct):** `stratify=y` instructs `train_test_split` to ensure that the fraction of each class in the training set and the test set mirrors the fraction in the full dataset. This prevents the minority class (churned) from being accidentally absent or severely under-represented in the test set.
- **Option D (Incorrect):** While oversampling is a valid post-split technique, class imbalance is very relevant to `train_test_split`. Without stratification, the test set may contain too few minority-class samples to meaningfully evaluate the model's performance on that class.

</details>

---

### 9. Question

An engineer imports `LinearRegression` from `sklearn.linear_model` and fits it on a binary classification dataset (target values are 0 or 1). She then calls `model.score(X_test, y_test)`.

**What does `model.score()` return in this context?**

- [ ] **A.** It returns classification accuracy (proportion of correctly predicted labels), the same metric `LogisticRegression.score()` would return.
- [ ] **B.** It returns R² (the coefficient of determination), which measures how well the line fits the continuous predictions to the 0/1 target values — not classification accuracy.
- [ ] **C.** It returns the mean squared error between predictions and true labels.
- [ ] **D.** It raises a `ValueError` because `LinearRegression` cannot be scored on binary target values.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `LinearRegression.score()` does not return classification accuracy. Classification accuracy is a metric specific to classifiers. `LinearRegression` returns R², which measures variance explained.
- **Option B (Correct):** `LinearRegression.score()` always returns R², the coefficient of determination, defined as `1 - SS_res/SS_tot`. Even when the target is binary (0 or 1), R² measures how well the continuous predictions track the binary labels — a different measure from classification accuracy.
- **Option C (Incorrect):** `score()` does not return mean squared error. To compute MSE you would use `mean_squared_error(y_test, model.predict(X_test))` from `sklearn.metrics`.
- **Option D (Incorrect):** `LinearRegression` accepts any numeric target array, including binary 0/1 arrays. There is no restriction or error related to the binary nature of the target.

</details>

---

### 10. Question

A team has a dataset with a nominal column `region` containing five unique values: `North`, `South`, `East`, `West`, and `Central`. They must encode this column before feeding it into a `LogisticRegression` model.

**Which encoding strategy is most appropriate, and why?**

- [ ] **A.** Ordinal encoding, assigning integers 1–5 alphabetically, since scikit-learn requires integer inputs.
- [ ] **B.** No encoding is needed; scikit-learn can handle string values in feature columns directly.
- [ ] **C.** Ordinal encoding, because it produces fewer columns than one-hot encoding and reduces dimensionality.
- [ ] **D.** One-hot encoding, because `region` is a nominal feature with no natural order, and assigning integers would impose a false ranking on the categories.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** While scikit-learn requires numeric input, alphabetical ordering does not represent any meaningful mathematical relationship between regions. Ordinal encoding implies `Central < East < North`, which is a meaningless and potentially harmful assumption.
- **Option B (Incorrect):** scikit-learn's `LogisticRegression` cannot directly process string-valued feature columns. Passing strings raises a `ValueError`. All features must be numeric before fitting.
- **Option C (Incorrect):** While ordinal encoding does produce fewer columns, the reduced dimensionality comes at the cost of introducing a false ordinal relationship between unordered categories, which can mislead the model's optimization.
- **Option D (Correct):** `region` is a nominal categorical variable — its categories have no inherent ranking or order. One-hot encoding creates a separate binary column for each category, allowing the model to treat each region as an independent input without implying any mathematical ordering between them.

</details>
