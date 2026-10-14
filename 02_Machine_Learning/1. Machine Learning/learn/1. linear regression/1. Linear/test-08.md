# Linear Regression Objective Test — Set 08

---

### 71. Question

A machine learning engineer writes the following code snippet and wants to know if it is correct:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("houses.csv")

X = df[["SquareFeet", "Bedrooms", "Bathrooms"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)  # Fits on the full dataset

score = model.score(X, y)  # Evaluates on the same full dataset
print(score)
```

**What is the primary issue with this code, and what does `model.score(X, y)` return?**

- [ ] **A.** No issue; using the full dataset for both training and scoring is the standard approach for linear regression.
- [ ] **B.** The model is fitted and evaluated on the same data without a train/test split. `model.score(X, y)` returns the training $R^2$, which will be overly optimistic and does not estimate how well the model generalizes to new data.
- [ ] **C.** `model.score(X, y)` raises an error because the full dataset cannot be used for scoring.
- [ ] **D.** The issue is that `score()` returns MSE instead of $R^2$ for scikit-learn regression models.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Fitting and evaluating on the same full dataset is not the standard approach — it is a common beginner mistake. It produces inflated performance metrics because the model has already "seen" all the data it is being evaluated on. The standard approach uses `train_test_split` to hold out a separate test set that was never used during training.
- **Option B (Correct):** Without a train/test split, the model is fitted on all rows of `X` and `y`. When `model.score(X, y)` is called with the same `X` and `y`, it evaluates the model on the same data it was trained on — this is the training $R^2$. Training $R^2$ is always at least as high as test $R^2$ (and often substantially higher) because the model has optimized its parameters to fit this exact data. The score does not represent the model's generalization ability to unseen observations.
- **Option C (Incorrect):** `model.score(X, y)` does not raise an error when called with the full training dataset. scikit-learn will compute and return the $R^2$ score on whatever data is passed to `score()`. No exception is raised; the concern is statistical validity, not a code error.
- **Option D (Incorrect):** For regression models in scikit-learn, `model.score(X, y)` returns $R^2$ (the coefficient of determination), not MSE. To compute MSE, you would use `sklearn.metrics.mean_squared_error(y, model.predict(X))`. The `.score()` method for regressors is definitively $R^2$.

</details>

---

### 72. Question

A machine learning practitioner wants to inspect the full prediction equation of a trained model with 2 features. She runs:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
```

The output is:
```
Intercept: 50.0
Coefficients: [3.5, -1.2]
```

The features are `[HoursStudied, AbsenceDays]`. **Write out the full prediction equation, and calculate the predicted score for a student with 8 hours studied and 3 absence days.**

- [ ] **A.** $\hat{y} = 3.5x_1 - 1.2x_2$; predicted score = 24.4
- [ ] **B.** $\hat{y} = 50 + 3.5x_1 - 1.2x_2$; predicted score = 74.4
- [ ] **C.** $\hat{y} = 50 + 3.5x_1 - 1.2x_2$; predicted score = 78.0
- [ ] **D.** $\hat{y} = 50 \times (3.5x_1 - 1.2x_2)$; predicted score = 1,220

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The equation omits the intercept (50.0). Without the intercept, the prediction at $x_1 = 8, x_2 = 3$ would be $3.5(8) - 1.2(3) = 28 - 3.6 = 24.4$. This is a common error — forgetting to add the intercept to the prediction.
- **Option B (Correct):** The full equation is $\hat{y} = 50 + 3.5x_1 - 1.2x_2$, combining the intercept and both coefficients. For $x_1 = 8, x_2 = 3$: $\hat{y} = 50 + 3.5(8) - 1.2(3) = 50 + 28 - 3.6 = 74.4$. The predicted exam score is 74.4.
- **Option C (Incorrect):** 78.0 would result from a different calculation — for example, not subtracting the absence penalty correctly: $50 + 28 = 78$ (omitting the $-1.2 \times 3 = -3.6$ term). The full calculation including the absence days penalty gives 74.4.
- **Option D (Incorrect):** Multiplying the intercept by the linear combination ($50 \times (3.5x_1 - 1.2x_2)$) is not the linear regression formula. The intercept is added (not multiplied), and the standard formula is $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$.

</details>

---

### 73. Question

An engineer is debugging a gradient descent loop that is not learning correctly. After reviewing, she finds this problematic line:

```python
# Intended gradient for the slope
grad_m = sum(e for e in errors) / len(errors)  # Bug!
```

The correct formula should multiply each error by its corresponding $x$ value.

**What is the bug, and what does this incorrect formula actually compute?**

- [ ] **A.** No bug; both formulas compute the same gradient because the x-values are constants.
- [ ] **B.** The bug is that `len(errors)` should be replaced with `len(x_values)`.
- [ ] **C.** The formula computes the gradient for the intercept ($\text{grad}_c = \frac{1}{n}\sum e_i$), not the slope. The correct slope gradient is $\text{grad}_m = \frac{1}{n}\sum e_i \cdot x_i$, which weights each error by its feature value.
- [ ] **D.** The bug is that `sum()` should be `mean()` — Python lists do not support `mean()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The two formulas are not equivalent. $\sum e_i / n$ (average error) gives the gradient for the intercept, while $\sum e_i x_i / n$ (average weighted error) gives the gradient for the slope. They produce the same result only in the degenerate case where all $x$-values are equal to 1.
- **Option B (Incorrect):** In a correct implementation, `len(errors)` and `len(x_values)` are typically equal (both equal $n$, the number of training samples), so this swap would not fix the real bug. The actual problem is the missing $x$-weighting in the summation, not the denominator.
- **Option C (Correct):** The formula $\frac{1}{n}\sum e_i$ computes the gradient of the loss with respect to the intercept $c$ (or $\beta_0$). This is because the intercept term contributes a constant "1" to every prediction — its derivative with respect to each prediction is 1, so the gradient is just the average of the errors. For the slope $m$, each prediction is $m \cdot x_i$, so the derivative involves $x_i$ — the correct formula is $\frac{1}{n}\sum e_i \cdot x_i$. Using the intercept gradient formula for the slope update will produce wrong slope updates.
- **Option D (Incorrect):** Python does not have a built-in `mean()` function for lists, but `sum(e for e in errors) / len(errors)` is a perfectly valid way to compute the mean of the errors. There is no `sum()` vs. `mean()` issue. The bug is the missing multiplication by $x_i$.

</details>

---

### 74. Question

An ML researcher is comparing two approaches to fitting a linear regression model:

**Approach 1:** Use scikit-learn's `LinearRegression().fit(X_train, y_train)` (uses the Normal Equation internally).
**Approach 2:** Run gradient descent for 10,000 iterations with a well-tuned learning rate.

**Which of the following statements about the two approaches are correct?** (Select all that apply.)

- [ ] **A.** Approach 1 (Normal Equation) finds the exact optimal parameters in a single step, without iterating.
- [ ] **B.** Approach 2 (gradient descent) approximates the solution by iteratively moving in the direction that reduces the loss.
- [ ] **C.** Approach 1 is always faster than Approach 2, regardless of dataset size.
- [ ] **D.** Both approaches, when fully converged, should arrive at the same (or very similar) optimal coefficient values for the given training data.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** scikit-learn's `LinearRegression` uses the Normal Equation (or a numerically stable variant via SVD decomposition) to find the exact least-squares solution in a single matrix computation: $\boldsymbol{\beta} = (X^T X)^{-1} X^T y$. There is no iteration — it computes the exact optimal parameters directly.
- **Option B (Correct):** Gradient descent is an iterative optimization algorithm that updates parameters incrementally by moving in the negative gradient direction of the loss. After enough iterations with a well-tuned learning rate, it converges close to the optimal solution but does so through many small steps rather than a single closed-form computation.
- **Option C (Incorrect):** The Normal Equation involves matrix inversion, which has computational complexity approximately $O(n^3)$ where $n$ is the number of features. For datasets with very large numbers of features, matrix inversion becomes extremely slow, and gradient descent (which scales better with large datasets) can be faster in practice. The claim that Approach 1 is "always faster regardless of dataset size" is incorrect.
- **Option D (Correct):** Both approaches minimize the same objective (mean squared error) on the same training data. The Normal Equation finds the exact global minimum analytically. Gradient descent converges to the same minimum iteratively (for a convex loss surface like MSE). At full convergence, both should produce the same (or numerically very close) coefficient values.

</details>

---

### 75. Question

A student is confused about when RMSE is smaller vs. larger than MSE. She creates a 4-point dataset with the following squared errors: $[0.01, 0.04, 0.09, 0.16]$.

**Is RMSE smaller or larger than MSE for this dataset, and why?**

- [ ] **A.** RMSE < MSE, because taking a square root always reduces a number.
- [ ] **B.** RMSE > MSE, because MSE = 0.075 (which is less than 1), and taking the square root of a number less than 1 produces a larger result: RMSE = $\sqrt{0.075} \approx 0.274$.
- [ ] **C.** RMSE = MSE, because the squared errors are all less than 1.
- [ ] **D.** RMSE cannot be computed from these squared errors; you need the original (unsquared) errors.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Taking a square root reduces a number only when that number is greater than 1. For numbers between 0 and 1, the square root is greater than the original number (e.g., $\sqrt{0.25} = 0.5 > 0.25$). "Always reduces" is a false generalization.
- **Option B (Correct):** MSE = $(0.01 + 0.04 + 0.09 + 0.16) / 4 = 0.30 / 4 = 0.075$. Since $0.075 < 1$: RMSE = $\sqrt{0.075} \approx 0.274$. Since $0.274 > 0.075$, RMSE is larger than MSE in this case. This demonstrates the important nuance: RMSE > MSE when MSE < 1, and RMSE < MSE when MSE > 1.
- **Option C (Incorrect):** RMSE = MSE only at the exact value of 1.0 (since $\sqrt{1} = 1$). The fact that all squared errors are less than 1 means their average (MSE) is also less than 1, making RMSE larger than MSE — not equal.
- **Option D (Incorrect):** RMSE can absolutely be computed from squared errors. RMSE = $\sqrt{\text{MSE}}$, and MSE can be computed directly from the given squared errors as their mean. The original (unsquared) errors are not needed to compute MSE or RMSE if the squared errors are already provided.

</details>

---

### 76. Question

A data scientist has a model with 4 input features. After fitting with scikit-learn, she wants to verify which feature has the largest influence on predictions. She accesses `model.coef_` and gets `[0.5, 3.2, -8.1, 0.02]`.

**She concludes: "Feature 3 (with coefficient −8.1) has the strongest influence on predictions." Is this conclusion valid?**

- [ ] **A.** Yes; the feature with the largest absolute coefficient always has the greatest influence on predictions.
- [ ] **B.** Partially; if all features are on the same scale (e.g., all standardized), then the largest absolute coefficient indicates the feature with the most influence per unit change. However, if features are on different scales, the raw coefficient magnitude cannot be directly compared to determine influence.
- [ ] **C.** No; only positive coefficients indicate influential features; negative coefficients mean the feature reduces the output and is therefore unimportant.
- [ ] **D.** No; `model.coef_` values are always between −1 and +1 and represent probabilities, not influence magnitudes.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The largest absolute coefficient indicates the feature that causes the largest change in prediction per unit change in that feature. However, "per unit change" is meaningful only when features are on comparable scales. If Feature 3 is measured in millimeters while others are measured in kilometers, a large coefficient for Feature 3 might actually represent a very small effect in practical terms.
- **Option B (Correct):** This is the nuanced, correct answer. If all features are standardized (zero mean, unit variance), then comparing absolute coefficient magnitudes is a valid way to rank feature influence. Feature 3 with $|\text{coef}| = 8.1$ would indeed have the most influence per standard deviation change. However, if features are on different original scales (e.g., age in years vs. income in dollars), the raw coefficients reflect both the relationship strength and the feature's unit scale. A coefficient of −8.1 for a feature measured in millimeters might represent less practical influence than a coefficient of 0.5 for a feature measured in thousands of dollars. Scale standardization is required for fair coefficient comparison.
- **Option C (Incorrect):** The sign of a coefficient indicates direction (positive = feature increases the prediction; negative = feature decreases it), not importance or influence. A large negative coefficient (−8.1) indicates a strong negative relationship — the feature has a significant influence on predictions, in the decreasing direction. Negative coefficients are not "unimportant."
- **Option D (Incorrect):** Linear regression coefficients are not bounded between −1 and +1, and they are not probabilities. They can take any real value, positive or negative, with any magnitude. Values between −1 and +1 describe probabilities (e.g., from logistic regression) or correlation coefficients, not linear regression weights.

</details>

---

### 77. Question

A software engineer tries to predict project completion time (in days) from team size and complexity score. He fits a multiple linear regression model. The team size ranges from 2 to 20 people; complexity score ranges from 1 to 100.

He checks: `model.coef_ = [-1.5, 0.3]`

**He concludes: "Team size (coefficient −1.5) matters much more than complexity score (coefficient 0.3) because its absolute coefficient is much larger."**

**Is this interpretation correct?**

- [ ] **A.** Yes; a coefficient of −1.5 always indicates 5× more influence than a coefficient of 0.3.
- [ ] **B.** Possibly incorrect; team size ranges from 2–20 (span of 18 units) while complexity ranges from 1–100 (span of 99 units). When inputs have different scales and ranges, raw coefficient magnitudes cannot be directly compared to determine relative importance without considering the feature ranges or standardizing.
- [ ] **C.** Yes; negative coefficients always indicate more important features than positive coefficients.
- [ ] **D.** Correct only if both features were standardized before fitting.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A coefficient of −1.5 vs. 0.3 does not straightforwardly imply 5× more influence in all contexts. The practical effect depends on the range of each feature. Team size changes by at most 18 units (from 2 to 20), contributing at most $18 \times 1.5 = 27$ days of effect. Complexity changes by up to 99 units (1 to 100), contributing up to $99 \times 0.3 = 29.7$ days of effect. Over their realistic ranges, complexity score actually has a comparable or slightly larger practical impact on predicted completion time.
- **Option B (Correct):** When features are measured on different scales and ranges, raw coefficient magnitudes cannot be directly compared. The coefficient tells us the change in output per unit change in the feature, but "one unit" of team size (one additional person) is very different from "one unit" of complexity score. To fairly compare feature importance, the coefficients should be multiplied by the feature's standard deviation (or range), or the features should be standardized to a common scale before fitting.
- **Option C (Incorrect):** Negative vs. positive signs indicate the direction of the relationship, not the magnitude of influence. A negative coefficient of −1.5 indicates that more team members predict shorter completion time (inverse relationship), while +0.3 for complexity predicts longer time. Both could represent substantial or negligible effects depending on the feature scales.
- **Option D (Incorrect):** The condition "only if both features were standardized before fitting" is the opposite of the actual requirement. If features WERE standardized, then comparing absolute coefficients WOULD be valid. The question states they were not standardized, making the comparison unreliable without adjusting for scale.

</details>

---

### 78. Question

A regression model achieves an $R^2$ of 0.87 on the test set. A business stakeholder asks: "Does this mean the model's predictions are always within 13% of the true value?"

**Which response corrects this misunderstanding?**

- [ ] **A.** Correct; $R^2 = 0.87$ means the average prediction error is 13% of the true value.
- [ ] **B.** Incorrect; $R^2 = 0.87$ means the model explains 87% of the variance in the target variable. It does not directly quantify per-prediction accuracy as a percentage of the true value. The 13% is not the average error percentage — it indicates that 13% of the variance is unexplained.
- [ ] **C.** Correct; $R^2 = 0.87$ is equivalent to saying predictions are 87% accurate.
- [ ] **D.** Incorrect; $R^2$ of 0.87 is always a poor result; models should achieve $R^2 \geq 0.99$ to be useful.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** $R^2$ does not directly measure per-prediction accuracy as a percentage of the true value. The metric for "how close predictions are to true values on a percentage basis" is Mean Absolute Percentage Error (MAPE), not $R^2$. The "13%" figure refers to unexplained variance, not a per-prediction percentage error.
- **Option B (Correct):** $R^2$ (coefficient of determination) measures the proportion of variance in the target variable explained by the model. $R^2 = 0.87$ means the model's predictions account for 87% of the variance in the test labels — the remaining 13% of variance is unexplained error. This does NOT mean predictions are always within 13% of the true value; an individual prediction could be off by a larger or smaller amount. $R^2$ is an aggregate variance-based measure, not a per-prediction accuracy percentage.
- **Option C (Incorrect):** Equating $R^2 = 0.87$ to "87% accuracy" confuses a regression metric with a classification metric. "Accuracy" is a classification concept (proportion of correct class predictions). In regression, there are no "correct" or "incorrect" predictions in the discrete sense — predictions are continuous values evaluated by how close they are to the true values.
- **Option D (Incorrect):** $R^2 = 0.87$ is generally a strong result in many real-world applications. The threshold for a "good" $R^2$ depends entirely on the domain and use case — in some fields (e.g., social science), 0.6 is considered good; in others (e.g., physical engineering), 0.99 might be expected. There is no universal rule that models must achieve $R^2 \geq 0.99$.

</details>

---

### 79. Question

A junior data scientist writes a training loop that uses list comprehensions but makes a subtle mistake:

```python
m = 0.5
c = 2.0
lr = 0.1
x_values = [1, 2, 3]
y_values = [3, 5, 7]

preds = [m * x + c for x in x_values]
errors = [p - y for p in preds for y in y_values]  # Bug!
print(errors)
```

**What does `errors` actually contain, and what should the correct line be?**

- [ ] **A.** `errors` contains 3 values (one per observation). No bug exists.
- [ ] **B.** `errors` contains 9 values — all combinations of each prediction with each y-value — because the nested `for` loop iterates over `preds` and `y_values` independently. The correct line should be: `errors = [p - y for p, y in zip(preds, y_values)]`
- [ ] **C.** `errors` raises a `TypeError` because list comprehensions cannot contain two `for` clauses.
- [ ] **D.** `errors` contains 6 values because the loop pairs each prediction with two y-values.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The code contains a bug. The nested `for` loop (`for p in preds for y in y_values`) is not `zip(preds, y_values)` — it produces a Cartesian product of all (p, y) combinations, resulting in $3 \times 3 = 9$ values, not 3.
- **Option B (Correct):** In Python, `[expr for x in a for y in b]` iterates over all combinations of `a` and `b` (Cartesian product). With `preds` having 3 elements and `y_values` having 3 elements, the nested comprehension generates $3 \times 3 = 9$ pairs — each prediction paired with each y-value. This is not the intended behavior (which is pairing corresponding elements). The correct code uses `zip(preds, y_values)` to pair elements positionally: `errors = [p - y for p, y in zip(preds, y_values)]`, producing 3 residuals — one per observation.
- **Option C (Incorrect):** Python list comprehensions can absolutely contain multiple `for` clauses — this is valid Python syntax. The problem is not a `TypeError` or syntax error; the code runs without error but produces incorrect results (9 values instead of 3).
- **Option D (Incorrect):** The Cartesian product of a 3-element list with a 3-element list produces $3 \times 3 = 9$ pairs, not 6. Each of the 3 predictions is paired with all 3 y-values, giving 9 total combinations.

</details>

---

### 80. Question

A product team trains a model to predict customer churn probability. Their ML engineer suggests using `LinearRegression` from scikit-learn.

A data scientist objects: "Linear regression is not appropriate for predicting a probability — it can produce predictions outside the [0, 1] range."

**Is the data scientist's concern valid?**

- [ ] **A.** No; `LinearRegression` automatically clamps its output to [0, 1] for binary targets.
- [ ] **B.** Yes; `LinearRegression` can produce predictions less than 0 or greater than 1 because it fits an unbounded linear function. Probability outputs should be bounded to [0, 1], requiring a model designed for that purpose (e.g., logistic regression).
- [ ] **C.** No; any regression model can predict probabilities if the target variable is defined as 0 or 1.
- [ ] **D.** Yes; but only when the dataset has fewer than 100 observations.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `LinearRegression` does not apply any output clamping or sigmoid transformation. Its prediction formula $\hat{y} = \beta_0 + \sum \beta_i x_i$ is unbounded and can produce any real number, positive or negative, including values well outside [0, 1].
- **Option B (Correct):** The data scientist's concern is valid. A linear regression model for a target in {0, 1} (churn/no-churn) can produce predictions like −0.3 or 1.5, which have no valid probability interpretation. Probabilities must lie in [0, 1]. Logistic regression applies a sigmoid function to the linear combination, mapping any real value to a valid probability in (0, 1). Linear regression is not designed for this use case.
- **Option C (Incorrect):** Defining the target variable as 0 or 1 does not constrain the linear regression model's output. The model learns coefficients that minimize MSE, and the resulting fitted line can cross below 0 or above 1 for certain input values. A binary-coded target makes the model's predictions approximate probabilities on average, but individual predictions can easily violate [0, 1] bounds.
- **Option D (Incorrect):** The probability-range violation problem with linear regression is not a small-dataset issue. It is a fundamental model design issue: linear regression is an unbounded predictor and cannot guarantee [0, 1] outputs regardless of dataset size. The concern is universal, not limited to small datasets.

</details>
