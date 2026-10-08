# Logistic Regression Objective Test

### 1. Question

1. A fraud-risk team computes the linear score `z = mx + c` for four loan applications before passing each score through the sigmoid function `g(z) = 1 / (1 + e^(-z))`. The default classification threshold is 0.5 on the class-1 probability.

| Applicant | z-score |
| :-------- | :------ |
| P         | 4.8     |
| Q         | -3.1    |
| R         | 0.0     |
| S         | -0.02   |

**Which of the following statements about the resulting probabilities and classifications are correct?**

- [ ] **A.** For applicant P (z = 4.8), `g(z)` is very close to 1, so the applicant is classified as approved.
- [ ] **B.** For applicant Q (z = -3.1), `g(z)` is very close to 1, so the applicant is classified as approved.
- [ ] **C.** For applicant S (z = -0.02), `g(z)` stays close to 0, since any negative z-score pushes probability toward 0.
- [ ] **D.** For applicant R (z = 0.0), `g(z)` equals exactly 0.5, and since `predict()` needs probability strictly above 0.5, the label is 0.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, D**

#### Explanations:
- **Option A (Correct):** A large positive z-score drives `e^(-z)` toward 0, so `g(z)` approaches `1 / (1 + 0) = 1`. A probability near 1 exceeds the 0.5 threshold, so the applicant is classified as approved.
- **Option B (Incorrect):** A large negative z-score drives `e^(-z)` toward infinity, so `g(z)` approaches `1 / ∞ = 0`, not 1; this applicant would be classified as rejected, not approved.
- **Option C (Incorrect):** The sigmoid's behavior near 0 is gradual, not extreme: a small z-score like -0.02 keeps `g(z)` very close to 0.5, not close to 0; only large-magnitude negative scores push `g(z)` toward 0.
- **Option D (Correct):** At `z = 0`, `g(z) = 1 / (1 + e^0) = 1/2 = 0.5` exactly. Since the decision rule requires the probability to be strictly greater than 0.5 for label 1, a probability of exactly 0.5 yields label 0.

</details>

---

### 2. Question

A machine learning engineer is plotting the sigmoid function `g(x) = 1 / (1 + e^(-x))` to sanity-check its shape before wiring it into a classifier.

**Which statement correctly describes the sigmoid function's behavior at the extremes of x?**

- [ ] **A.** As x approaches negative infinity, g(x) approaches 0; as x approaches positive infinity, g(x) approaches 1.
- [ ] **B.** As x approaches negative infinity, g(x) approaches 1; as x approaches positive infinity, g(x) approaches 0.
- [ ] **C.** As x approaches either negative or positive infinity, g(x) approaches 0.5, the midpoint of its range.
- [ ] **D.** As x approaches negative infinity, g(x) approaches negative infinity; as x approaches positive infinity, g(x) approaches positive infinity.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** As `x` becomes very negative, `e^(-x)` grows toward infinity, causing `g(x)` to approach `1 / ∞ = 0`. As `x` becomes very positive, `e^(-x)` approaches 0, so `g(x)` approaches `1 / (1 + 0) = 1`.
- **Option B (Incorrect):** This swaps the two limits; the sigmoid function is monotonically increasing and rises from 0 to 1 as `x` increases, not the reverse.
- **Option C (Incorrect):** 0.5 is the value of `g(x)` specifically at `x = 0`, not at either extreme. The extremes represent the horizontal asymptotes where the curve levels off at 0 and 1.
- **Option D (Incorrect):** The sigmoid function is strictly bounded between 0 and 1 across its entire domain; it never grows unboundedly like the linear term `mx + c`.

</details>

---

### 3. Question

A senior data analyst inspects a fitted logistic regression model and finds that the coefficient for `credit_score` is -0.12.

**Which statement correctly interprets this coefficient?**

- [ ] **A.** A negative coefficient means the `credit_score` feature was excluded from the fitted model entirely.
- [ ] **B.** As `credit_score` rises, the score and approval probability increase, since negative coefficients amplify positive features.
- [ ] **C.** As `credit_score` rises, the negative coefficient pulls the linear score down, lowering the approval probability.
- [ ] **D.** A coefficient's sign never affects predicted probability; only its magnitude matters for the outcome.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** A non-zero coefficient confirms the feature is active and contributing to the model's linear score. An excluded feature would not appear in `model.coef_` at all.
- **Option B (Incorrect):** A negative coefficient reduces the score as the feature value grows; it does not amplify or increase the probability.
- **Option C (Correct):** Because the coefficient is negative, increasing `credit_score` reduces the linear combination $z$. Because the sigmoid function is strictly monotonically increasing, a lower $z$ directly results in a lower predicted approval probability.
- **Option D (Incorrect):** The sign indicates the direction of the relationship (whether the feature increases or decreases the log-odds), whereas the magnitude determines the strength of the effect. Both are critical for determining the final outcome.

</details>

---

### 4. Question

A backend engineer writes `LR = LogisticRegression()` and `LR.fit(X_train, y_train)` early in a script. Further down, in a different code cell, the engineer writes `predictions = lr.predict(X_test)` without ever creating a variable named `lr`.

**What happens when this code runs, and why?**

- [ ] **A.** Python raises a "not defined" error for `lr`, since `lr` and `LR` are treated as two distinct case-sensitive names.
- [ ] **B.** The code runs successfully, because Python treats all variable names as case-insensitive during execution by default.
- [ ] **C.** scikit-learn automatically resolves `lr` to the most recently created model object, no matter which name was used.
- [ ] **D.** The code raises an error, because a single `LogisticRegression` object cannot be reused across multiple variable names.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Python is case-sensitive, so `lr` and `LR` are two distinct, unrelated variable names. Because `lr` was never assigned, calling `lr.predict(...)` causes Python to raise a "not defined" error (`NameError`).
- **Option B (Incorrect):** Python variable names are strictly case-sensitive, not case-insensitive; `lr` and `LR` never refer to the same object.
- **Option C (Incorrect):** scikit-learn has no mechanism to auto-resolve undefined variable names to recently created objects; variable resolution is handled entirely by Python's namespace lookup, not by the library.
- **Option D (Incorrect):** A fitted model object can be referenced by any number of variable names or reused; the failure here is a basic undefined variable error, not a scikit-learn restriction.

</details>

---

### 5. Question

Continuing with the loan-approval dataset, an engineer wants to bring the `employment_type` column (salaried, self-employed, business, student — no natural order) into the model. The same team is separately reviewing a hypothetical `service_priority` column with values low, medium, and high, which does have a natural rank.

**Which of the following statements about encoding these two columns are correct?**

- [ ] **A.** `employment_type` should use one-hot encoding, since its categories (salaried, self-employed, business, student) have no natural order.
- [ ] **B.** `service_priority` (low, medium, high) should use ordinal encoding, assigning increasing integers (1, 2, 3) that reflect its natural rank.
- [ ] **C.** `employment_type` should use ordinal encoding, since alphabetical order (business, salaried, self-employed, student) gives it a valid ranking.
- [ ] **D.** Scikit-learn models cannot accept integer-encoded categorical columns at all, so one-hot encoding is mandatory for every categorical column.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** `employment_type` has no natural order, making it a nominal variable best suited for one-hot encoding. This approach creates a distinct binary column for each category rather than imposing an artificial ranking.
- **Option B (Correct):** Ordinal encoding is designed for categories with a natural order. Because low, medium, and high form a clear ranking, assigning increasing integers reflects that progression correctly.
- **Option C (Incorrect):** Alphabetical ordering of category names does not represent a meaningful mathematical or real-world rank. Because `employment_type` categories lack an intrinsic hierarchy, one-hot encoding should be used instead of ordinal encoding.
- **Option D (Incorrect):** scikit-learn algorithms accept integer-encoded numerical inputs without issue. The reason to avoid ordinal encoding for nominal features is that it forces an unintended numeric hierarchy onto unordered categories, not because the library cannot process integers.

</details>

---

### 6. Question

A team fits an ordinary `LinearRegression` model — instead of `LogisticRegression` — on the loan-approval dataset, where the target `approved` is 0 (rejected) or 1 (approved). Calling `lr.predict(X_test)` produces predicted values that include numbers like -1.145 and other values outside the 0-1 range.

**Which of the following statements about this modeling choice and its output are correct?**

- [ ] **A.** The out-of-range predictions stem from an incorrect `train_test_split` setup, not from the choice of model class.
- [ ] **B.** A value like -1.145 falls outside 0-1 because `LinearRegression`'s line (`y = mx + c`) is unbounded and can produce any value from negative to positive infinity.
- [ ] **C.** Switching to `LogisticRegression` fixes this, since it feeds the same linear equation through the sigmoid function, which is bounded strictly between 0 and 1.
- [ ] **D.** `model.score()` on this `LinearRegression` fit returns classification accuracy, the same metric it would return for `LogisticRegression`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B, C**

#### Explanations:
- **Option A (Incorrect):** The out-of-range value (-1.145) stems from the unbounded nature of the linear equation $mx + c$, not from any split configuration; an alternate split would still generate predictions outside 0–1.
- **Option B (Correct):** The linear equation $y = mx + c$ has neither an upper nor a lower bound, allowing it to produce values below 0 or above 1 that cannot serve as valid probabilities.
- **Option C (Correct):** `LogisticRegression` routes the linear score into the sigmoid function $g(x) = \frac{1}{1 + e^{-x}}$, compressing any real-valued input into the range strictly between 0 and 1 to represent a valid probability.
- **Option D (Incorrect):** `LinearRegression.score()` outputs the $R^2$ coefficient of determination for continuous regression fit, whereas classification accuracy is returned by classifier implementations such as `LogisticRegression`.

</details>

---

### 7. Question

A machine learning engineer fits both a `LinearRegression` and a `LogisticRegression` model, both imported from `sklearn.linear_model`, on the same train/test split of a loan-approval dataset, then compares how the two model classes behave in code.

**Which of the following statements correctly describe differences between the two model classes?**

- [ ] **A.** `model.score(X_test, y_test)` returns the R² score for both `LinearRegression` and `LogisticRegression` models alike.
- [ ] **B.** `LogisticRegression` coefficients are fit by minimizing squared error, exactly like `LinearRegression` does.
- [ ] **C.** `model.score()` returns classification accuracy for `LogisticRegression`, but returns R² for `LinearRegression`.
- [ ] **D.** `model.predict()` on `LogisticRegression` compares the class-1 probability to the default 0.5 threshold to derive its label.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C, D**

#### Explanations:
- **Option A (Incorrect):** `model.score()` does not return the same metric across both classes; only `LinearRegression` returns the $R^2$ score to measure continuous fit, whereas classifier models return accuracy.
- **Option B (Incorrect):** `LinearRegression` fits parameters by minimizing squared error (ordinary least squares), while `LogisticRegression` fits its coefficients using maximum likelihood estimation (minimizing log-loss).
- **Option C (Correct):** `LogisticRegression.score()` computes mean classification accuracy (the fraction of test samples correctly classified), whereas `LinearRegression.score()` computes the coefficient of determination ($R^2$).
- **Option D (Correct):** Calling `predict()` on a `LogisticRegression` model uses `predict_proba` under the hood; if the predicted class-1 probability exceeds the default decision threshold of 0.5, it outputs label 1; otherwise, it assigns label 0.

</details>

---

### 8. Question

A fintech startup's product manager is scoping two new models: one team wants to predict whether a loan application will be approved or rejected, and another team wants to predict the exact price a house will sell for. The product manager needs to explain to stakeholders which problem type each task falls under.

**Which statement correctly classifies the two tasks?**

- [ ] **A.** Both tasks are regression problems, because both rely on numeric input features to generate predictions.
- [ ] **B.** Loan approval is classification, since its outcome is discrete (approved or rejected); house price is regression, a continuous value.
- [ ] **C.** Loan approval is regression, since approval confidence varies smoothly; house price is classification, since prices fall into set brackets.
- [ ] **D.** Both tasks are classification problems, because any task using numeric input features counts as classification.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A problem is not classified as regression simply because it uses numeric inputs. Both classification and regression models commonly use numeric input features; the distinction depends entirely on whether the target outcome is discrete or continuous.
- **Option B (Correct):** Loan approval has two discrete outcomes (approved or rejected), which defines a classification task. House price prediction estimates a quantity across a continuous numeric range, which defines a regression task.
- **Option C (Incorrect):** This reverses the fundamental definitions: loan approval results in categorical buckets (classification), while house price is an unbounded continuous value (regression).
- **Option D (Incorrect):** The nature of the input features has no bearing on the problem type; the mathematical definition depends solely on the structure of the target variable (discrete vs. continuous).

</details>

---

### 9. Question

A team is preparing the loan-approval dataset from the session (columns: `age`, `income`, `credit_score`, `employment_type`, `loan_amount`, `approved`) for `train_test_split`. In this dataset, 90% of applications are approved and only 10% are rejected, and the team is worried a random split could leave the test set with almost no rejected examples.

**Which `train_test_split` call correctly addresses this concern?**

- [ ] **A.** `train_test_split(x, y, test_size=0.2, random_state=42)` — the default settings already balance class ratios automatically.
- [ ] **B.** `train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)` — stratifying on `y` keeps the approved/rejected ratio consistent across train and test.
- [ ] **C.** `train_test_split(x, y, test_size=0.8, random_state=42)` — a larger `test_size` guarantees more rejected rows land in the test set.
- [ ] **D.** `train_test_split(x, y, random_state=42, shuffle=False)` — disabling shuffling preserves the original class ratio.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `train_test_split` performs a random split by default and does not balance class proportions automatically; an uneven 90/10 split will still risk under-representing the minority class in the test set.
- **Option B (Correct):** Setting `stratify=y` ensures that the proportion of each class label (approved vs. rejected) matches between the original dataset, the training set, and the test set, preventing minority-class starvation.
- **Option C (Incorrect):** Increasing `test_size` sends more overall data to the test split, but it does not guarantee proportional representation of classes; the minority class can still be under-represented or unevenly distributed.
- **Option D (Incorrect):** Disabling shuffling (`shuffle=False`) splits rows in their original file order, which does not preserve class proportions and risks severe bias if rows are sorted or grouped by target label.

</details>

---

### 10. Question

A backend engineer calls `model.predict_proba(X_test)` on a fitted logistic regression model and reads the row for one applicant: the second column (probability of class 1) shows 0.83.

**Which statement correctly describes this row's full output and predicted label?**

- [ ] **A.** The first column (probability of class 0) must also show 0.83, since scikit-learn always mirrors both columns.
- [ ] **B.** The first column shows an unrelated value that does not need to sum with the second column, and the predicted label is 0.
- [ ] **C.** The first column shows 0.17, and since 0.83 does not exceed the default threshold, the predicted label is 0.
- [ ] **D.** The first column shows 0.17, since the two columns always sum to 1, and since 0.83 exceeds the default 0.5 threshold, the predicted label is 1.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** The two columns are complementary probabilities representing mutually exclusive binary classes, not identical mirrored copies. Because they sum to 1, a class-1 probability of 0.83 requires the class-0 probability to be 0.17.
- **Option B (Incorrect):** The probabilities across classes for any given row must sum to 1. They are directly dependent on each other, not arbitrary or unrelated values.
- **Option C (Incorrect):** While the class-0 probability is correctly calculated as 0.17, the class-1 probability of 0.83 is strictly greater than the default 0.5 threshold, which produces a predicted label of 1, not 0.
- **Option D (Correct):** `predict_proba` columns always sum to 1, so the class-0 probability is $1 - 0.83 = 0.17$. Because the class-1 probability (0.83) is above the default decision threshold of 0.5, `predict()` assigns a label of 1.

</details>
