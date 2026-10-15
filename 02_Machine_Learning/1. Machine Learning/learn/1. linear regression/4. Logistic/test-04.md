# Logistic Regression Objective Test — Set 04

### 31. Question

A data analyst compares what the `predict()` method returns for `LinearRegression` versus `LogisticRegression` on the same test set of a binary classification problem.

**Which of the following statements are correct?**

- [ ] **A.** `LinearRegression.predict()` can return values below 0 or above 1 because its output is an unbounded linear combination.
- [ ] **B.** `LogisticRegression.predict()` returns only the values 0 or 1 — discrete class labels — not probabilities.
- [ ] **C.** Both `LinearRegression.predict()` and `LogisticRegression.predict()` return values bounded between 0 and 1.
- [ ] **D.** `LogisticRegression.predict()` returns probabilities in the range (0, 1) for each sample.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** `LinearRegression` computes `y = w·x + b` with no bounding step. The result is the raw dot product plus intercept, which can be any real number, including negative values or values greater than 1.
- **Option B (Correct):** `LogisticRegression.predict()` applies the sigmoid function internally and then applies a threshold (default 0.5) to produce a binary class label: 0 or 1. It does not return probabilities — those require `predict_proba()`.
- **Option C (Incorrect):** `LinearRegression.predict()` is not bounded between 0 and 1. Only the sigmoid function produces bounded (0, 1) output; `LinearRegression` has no sigmoid step.
- **Option D (Incorrect):** `LogisticRegression.predict()` returns class labels (0 or 1), not probabilities. Probabilities are returned by `predict_proba()`.

</details>

---

### 32. Question

A machine learning team is preparing a medical diagnosis dataset with the following columns:

- `blood_pressure`: numeric (mmHg)
- `age`: numeric (years)
- `gender`: Male, Female (no natural order)
- `risk_tier`: Low, Medium, High (natural order)
- `diagnosis`: 0 (healthy) or 1 (disease) — **target**

**Which columns require encoding before fitting a `LogisticRegression` model, and what encoding should be applied?**

- [ ] **A.** Only `gender` needs encoding using ordinal encoding; all other columns are already numeric.
- [ ] **B.** `gender` needs one-hot encoding (nominal), and `risk_tier` needs ordinal encoding (ordinal). The numeric and target columns need no encoding.
- [ ] **C.** All columns including `diagnosis` need one-hot encoding before the model can be fitted.
- [ ] **D.** `gender` needs ordinal encoding (alphabetically: Female=1, Male=2), and `risk_tier` needs one-hot encoding.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Only encoding `gender` with ordinal encoding is incorrect on both counts. `gender` is nominal and should use one-hot encoding; `risk_tier` is ordinal and also requires encoding (ordinal), not being left as raw strings.
- **Option B (Correct):** `gender` (Male, Female) is a nominal variable with no natural rank — one-hot encoding avoids imposing a false numeric hierarchy. `risk_tier` (Low, Medium, High) is ordinal and should be encoded as integers reflecting its rank (e.g., Low=1, Medium=2, High=3). `blood_pressure` and `age` are already numeric. `diagnosis` is the target column and does not need encoding — it is already in the correct numeric form.
- **Option C (Incorrect):** One-hot encoding the target column `diagnosis` would be incorrect and harmful. The target variable is passed as y (the labels array), not as a feature column. Encoding 0/1 target values with one-hot encoding makes no sense in standard supervised learning.
- **Option D (Incorrect):** Applying alphabetical ordinal encoding to `gender` imposes a false rank (Female < Male), which is meaningless and potentially discriminatory in modeling. `risk_tier` has a real ordinal rank and should use ordinal encoding, not one-hot encoding.

</details>

---

### 33. Question

A `LogisticRegression` model has been fitted on a healthcare dataset. The coefficient for the feature `bmi` is +0.08.

**Which of the following correctly describes the relationship between `bmi` and the predicted probability of disease?**

- [ ] **A.** A higher BMI directly reduces the probability of disease because the coefficient is a small positive number (less than 1).
- [ ] **B.** As BMI increases by one unit, the linear score z increases by 0.08, which increases the sigmoid output and thus raises the predicted probability of disease.
- [ ] **C.** The coefficient +0.08 means BMI has no meaningful effect on the prediction, since values less than 0.1 are treated as zero by scikit-learn.
- [ ] **D.** As BMI increases, the linear score z decreases by 0.08, reducing the probability of disease.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A positive coefficient (+0.08) means BMI increases the linear score, which increases (not decreases) the predicted probability. The magnitude being less than 1 does not make it negative in effect.
- **Option B (Correct):** Each unit increase in BMI adds +0.08 to z. Since the sigmoid function is monotonically increasing, a higher z produces a higher sigmoid output (probability). Therefore, higher BMI → higher z → higher predicted probability of disease.
- **Option C (Incorrect):** scikit-learn does not threshold coefficients to zero based on their magnitude. A coefficient of +0.08 actively contributes to the linear score; small but non-zero coefficients have real, proportional effects.
- **Option D (Incorrect):** A positive coefficient (+0.08) adds to z when the feature increases; it does not decrease z. A negative coefficient would decrease z.

</details>

---

### 34. Question

A developer runs the following code:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)
print(proba.shape)
```

The dataset has 200 test samples and the target has 2 classes.

**What does `print(proba.shape)` output?**

- [ ] **A.** `(200,)` — one probability per sample.
- [ ] **B.** `(200, 1)` — one column for the class-1 probability.
- [ ] **C.** `(200, 2)` — two columns: one for P(class 0) and one for P(class 1).
- [ ] **D.** `(2, 200)` — rows are classes, columns are samples.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `predict_proba` returns a 2D array for binary classification, not a 1D array. A 1D array is the shape of `predict()` output (class labels).
- **Option B (Incorrect):** `predict_proba` always returns a column for each class. For a binary problem (2 classes), it returns 2 columns — one for each class — not just 1 column.
- **Option C (Correct):** For a binary classification problem with 200 test samples, `predict_proba` returns a 2D numpy array of shape `(200, 2)`. The first column contains P(class 0) and the second column contains P(class 1) for each sample.
- **Option D (Incorrect):** The output format is always `(n_samples, n_classes)`, where n_samples is the number of test rows. scikit-learn does not transpose this to `(n_classes, n_samples)`.

</details>

---

### 35. Question

A loan application scoring system uses a `LogisticRegression` model. The classifier outputs a class-1 (approved) probability of 0.51 for a borderline applicant.

An analyst argues: "This result is barely above the threshold. We should use `LinearRegression` instead, since it gives a more nuanced continuous score rather than a binary decision."

**Which statements about this argument are correct?**

- [ ] **A.** The analyst is correct that `LogisticRegression.predict()` produces a binary output, but incorrect that `LinearRegression` is the right solution — `predict_proba()` can provide the nuanced probability needed.
- [ ] **B.** The analyst is correct in all respects; `LinearRegression` produces more meaningful outputs for borderline cases.
- [ ] **C.** The analyst is wrong; `LogisticRegression.predict()` already returns probabilities, not binary labels.
- [ ] **D.** Both models produce the same output for binary targets, so the choice makes no difference.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** The analyst correctly observes that `predict()` returns binary labels (0 or 1), so borderline cases look identical to confident cases. However, the solution is not to switch to `LinearRegression` (which produces unbounded, non-probabilistic values). Instead, use `predict_proba()` to access the continuous probability (0.51 in this case), which provides the nuanced score the analyst desires.
- **Option B (Incorrect):** `LinearRegression` on a binary target produces unbounded values (possibly negative or > 1), which are not valid probabilities and can be harder to interpret, not easier. Using `predict_proba()` with `LogisticRegression` is the superior approach.
- **Option C (Incorrect):** `LogisticRegression.predict()` returns discrete class labels (0 or 1), not probabilities. Probabilities are available through `predict_proba()`. The analyst's observation about the binary output of `predict()` is correct.
- **Option D (Incorrect):** The two models produce fundamentally different outputs. `LinearRegression.predict()` returns continuous unbounded values; `LogisticRegression.predict()` returns class labels. They are not equivalent.

</details>

---

### 36. Question

A team splits their dataset using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

The dataset has 1000 samples, with 800 labeled as class 0 and 200 labeled as class 1.

**How many class-0 and class-1 samples will be in the test set?**

- [ ] **A.** Exactly 100 class-0 and 100 class-1 samples (50/50 split).
- [ ] **B.** 160 class-0 and 40 class-1 samples if random state 42 happens to produce an even split.
- [ ] **C.** 160 class-0 and 40 class-1 samples, guaranteed by the stratify parameter.
- [ ] **D.** The distribution cannot be predicted; it depends on the random_state value.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Stratification preserves the original class ratio (80% class 0, 20% class 1), not a 50/50 split. A 50/50 test set would require oversampling or class balancing, not stratification.
- **Option B (Incorrect):** The stratify parameter guarantees the class ratio in the test set regardless of the random_state. The result is deterministic (given the same random_state), not a coincidence of the seed.
- **Option C (Correct):** With `test_size=0.2` and 1000 total samples, the test set has 200 samples. `stratify=y` ensures these 200 samples preserve the 80/20 class ratio: 200 × 0.80 = 160 class-0 and 200 × 0.20 = 40 class-1 samples.
- **Option D (Incorrect):** With `stratify=y`, the class ratio in the test set is not random — it is guaranteed to match the original dataset's ratio. The `random_state` only determines which specific rows are selected, not the class distribution.

</details>

---

### 37. Question

A team wants to compare the performance of a `LogisticRegression` and a `LinearRegression` model on a regression task (predicting house prices, a continuous value). The engineer correctly uses `LinearRegression` for the regression task but accidentally also fits `LogisticRegression` on the same data.

**Which of the following correctly describes the problem with fitting `LogisticRegression` on a regression task?**

- [ ] **A.** `LogisticRegression` is designed for classification and will attempt to classify house prices into discrete categories, producing meaningless binary outputs instead of price predictions.
- [ ] **B.** `LogisticRegression` will work fine since both models use linear combinations of features; only the name differs.
- [ ] **C.** `LogisticRegression` will automatically switch to regression mode when the target values are continuous floats.
- [ ] **D.** `LogisticRegression` will raise a `ValueError` if any target value is not exactly 0 or 1.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `LogisticRegression` is a classification model. When fitted on continuous house price targets, it treats each unique price as a separate class and produces class labels and probabilities. The output of `predict()` will be one of the training price values, which is meaningless for regression. The model is being applied to the wrong problem type.
- **Option B (Incorrect):** While both models compute a linear combination internally, the output step differs fundamentally: `LogisticRegression` adds a sigmoid and threshold, making it a classifier. The core computation being linear does not make them interchangeable for all tasks.
- **Option C (Incorrect):** scikit-learn models do not automatically switch between modes based on target data type. `LogisticRegression` always behaves as a classifier, regardless of whether the target contains integers, floats, or other numeric values.
- **Option D (Incorrect):** `LogisticRegression` does not require target values to be exactly 0 or 1. It can handle any integer class labels (e.g., 0/1/2 for multi-class, or even string labels). However, fitting on continuous floating-point targets is semantically inappropriate even if it runs without error.

</details>

---

### 38. Question

A data scientist is explaining the difference between classification and regression to a new team member. She presents two prediction tasks:

- **Task 1:** Predicting whether an email is spam (yes/no)
- **Task 2:** Predicting next month's electricity bill in dollars

**Which of the following correctly classifies these tasks and explains why?**

- [ ] **A.** Both tasks are classification, since both involve making a prediction from input data.
- [ ] **B.** Task 1 is regression (it predicts a confidence score), and Task 2 is classification (prices fall into high/medium/low tiers).
- [ ] **C.** Both tasks are regression, since both use numeric features to compute an output.
- [ ] **D.** Task 1 is classification (discrete outcome: spam or not spam) and Task 2 is regression (continuous outcome: a dollar amount on a numeric scale).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** Making a prediction from input data is not the definition of classification; it applies to all supervised learning. The distinction lies in whether the output is discrete (classification) or continuous (regression).
- **Option B (Incorrect):** This reverses the definitions. Spam detection has a discrete outcome (spam / not spam), which is classification. Electricity bill prediction has a continuous dollar value as output, which is regression. The labels are also misleading: "confidence score" is the model's internal mechanism, not the nature of the task.
- **Option C (Incorrect):** Using numeric features does not make a task regression. The problem type is determined by the nature of the target variable (output), not the inputs.
- **Option D (Correct):** Task 1 (spam detection) has a discrete categorical output — the email either is or is not spam — which defines classification. Task 2 (electricity bill) produces a continuous numeric value that can take any amount across a range, which defines regression.

</details>

---

### 39. Question

A `LogisticRegression` model is fitted and then inspected. The engineer accesses `model.coef_` and sees an array of shape `(1, 5)` for a binary classification problem with 5 features.

**What does the `(1, 5)` shape indicate?**

- [ ] **A.** One class was predicted and five incorrect predictions were made.
- [ ] **B.** There is one set of coefficients (one decision boundary) with five values — one per feature — for the binary classification problem.
- [ ] **C.** The model has one feature and five classes.
- [ ] **D.** The model ran for five training iterations and one coefficient was updated each time.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `model.coef_` shape has nothing to do with prediction counts or correct/incorrect predictions. It describes the coefficient matrix structure.
- **Option B (Correct):** For binary logistic regression, there is one decision boundary (one linear equation), so `coef_` has shape `(1, n_features)`. Here, `(1, 5)` means 1 decision boundary with 5 coefficients — one learned weight per feature.
- **Option C (Incorrect):** The shape `(1, 5)` means 1 decision boundary and 5 features. If the model had one feature and five classes, the shape would be `(5, 1)` in a multi-class setting using OvR (one-vs-rest).
- **Option D (Incorrect):** `model.coef_` is not related to training iterations or update steps. It contains the final learned weights after optimization is complete. The number of training iterations is controlled by `max_iter` and is unrelated to the shape of `coef_`.

</details>

---

### 40. Question

A team builds a fraud detection classifier. They observe that when they call `model.predict()`, the model only ever predicts class 0 (not fraud) — even for transactions their domain experts know are fraudulent.

**Which of the following is the most likely cause of this behavior?**

- [ ] **A.** The `LogisticRegression` class does not support imbalanced datasets and must be replaced with `LinearRegression`.
- [ ] **B.** The `predict()` method is broken when called without specifying `threshold=0.5`.
- [ ] **C.** The model has too many features and has overfit to the training data.
- [ ] **D.** The training data is severely imbalanced (e.g., 99% not-fraud, 1% fraud), causing the model to learn that always predicting class 0 minimizes overall loss, since accuracy is high even without detecting any fraud.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:
- **Option A (Incorrect):** `LogisticRegression` does support imbalanced datasets; it has a `class_weight` parameter (e.g., `class_weight='balanced'`) to help address this. Switching to `LinearRegression` would not fix the problem and is inappropriate for a classification task.
- **Option B (Incorrect):** `predict()` does not accept a `threshold` parameter. The default 0.5 threshold is always applied internally. The threshold is not "unspecified" when not passed explicitly.
- **Option C (Incorrect):** While overfitting is a concern, a model that always predicts one class is more characteristic of underfitting or class imbalance. Overfitting would typically result in the model memorizing training patterns, not consistently predicting a single class.
- **Option D (Correct):** With severe class imbalance (99% not-fraud), a model that always predicts class 0 achieves 99% overall accuracy, which minimizes standard log-loss during training. The model learns this "lazy" strategy because it is penalized very little for missing the rare fraud cases. Using `class_weight='balanced'`, stratification during splitting, or alternative evaluation metrics (e.g., F1, AUC) can address this.

</details>
