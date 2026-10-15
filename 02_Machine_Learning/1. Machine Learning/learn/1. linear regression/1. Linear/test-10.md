# Linear Regression Objective Test — Set 10

---

### 91. Question

A data scientist wants to evaluate a multiple linear regression model on the test set using three metrics simultaneously. She writes:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = model.score(X_test, y_test)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
```

The output is: `MAE: 3.20, MSE: 15.00, RMSE: 3.87, R²: 0.91`

**Which of the following statements about these results are correct?** (Select all that apply.)

- [ ] **A.** RMSE (3.87) < MSE (15.00), which is expected because MSE = 15.00 > 1.
- [ ] **B.** The model explains 91% of the variance in the test set target variable.
- [ ] **C.** MAE (3.20) is smaller than RMSE (3.87) because RMSE applies a squaring penalty that amplifies larger individual errors before taking the square root.
- [ ] **D.** Because MSE = 15.00, the model is performing poorly and should be discarded without evaluating the domain context.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C**

#### Explanations:
- **Option A (Correct):** RMSE = $\sqrt{\text{MSE}} = \sqrt{15.00} \approx 3.87 < 15.00 = \text{MSE}$. When MSE > 1, its square root (RMSE) is smaller than MSE numerically. This is consistent with the mathematical property of square roots for values greater than 1.
- **Option B (Correct):** $R^2 = 0.91$ means the model explains 91% of the variance in the test labels. The remaining 9% of variance is unexplained by the model's linear predictions.
- **Option C (Correct):** RMSE consistently exceeds or equals MAE (for the same error distribution). This is because RMSE squares errors before averaging, giving disproportionate weight to large errors, then takes the square root. This penalization of large errors makes RMSE larger than or equal to MAE. Here: $3.87 > 3.20$, which confirms that some errors are larger than the MAE average, and their squared values inflate RMSE above MAE.
- **Option D (Incorrect):** Whether MSE = 15.0 is "poor" depends entirely on the scale and context of the target variable. If predicting house prices in millions of dollars, MSE = 15 might be excellent. If predicting blood pressure in mmHg, it might indicate poor performance. No absolute threshold exists for "poor" MSE — domain context and baseline comparisons are required before discarding a model.

</details>

---

### 92. Question

A ML engineer is debugging why two training runs with identical data and parameters produce slightly different final coefficient values. The only difference is that Run 1 shuffles data before training and Run 2 does not.

For scikit-learn's `LinearRegression` (which uses the Normal Equation), **does the order of training data rows affect the final fitted parameters?**

- [ ] **A.** Yes; row order significantly affects the Normal Equation computation and produces different coefficients.
- [ ] **B.** No; the Normal Equation is a closed-form matrix computation that involves sums over all training observations — it is invariant to the order in which rows appear in the matrix.
- [ ] **C.** Yes; but only when the dataset has more than 100 rows.
- [ ] **D.** No; but only when `random_state` is set in `LinearRegression()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The Normal Equation computes $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T y$, which involves matrix multiplications ($X^T X$ and $X^T y$). These matrix products sum all observations simultaneously — the result is a scalar value per entry that depends on the totality of all rows, not their order. Row permutation changes the order of rows in $X$ but not the values of $X^T X$ or $X^T y$.
- **Option B (Correct):** Matrix multiplication is order-invariant with respect to the rows of the input matrix when computing $X^T X$ (a square matrix) and $X^T y$. Both quantities are summations over all training observations, and the sum $\sum_i x_i x_i^T$ gives the same result regardless of which order the terms are summed. Therefore, the Normal Equation solution is unaffected by row shuffling — the fitted coefficients will be numerically identical.
- **Option C (Incorrect):** The invariance of the Normal Equation to row order holds regardless of dataset size. It is a mathematical property of matrix operations, not a size-dependent quirk.
- **Option D (Incorrect):** `LinearRegression()` in scikit-learn does not have a `random_state` parameter because the Normal Equation is deterministic — it requires no random initialization or sampling. The result is always the same unique analytical solution for a given dataset.

</details>

---

### 93. Question

A data scientist uses the following code to fit a regression model and then wants to understand the exact meaning of `model.coef_`:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.array([[2, 3], [4, 5], [6, 7], [8, 9]])
y_train = np.array([11, 19, 27, 35])

model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
```

The data satisfies $y = 1 + 3x_1 + 1x_2$ exactly (e.g., $1 + 3(2) + 1(3) = 10 \ne 11$, hmm — actually: $1 + 3(2) + 1(3) = 1+6+3=10$). Let's use the relationship $y = x_1 + x_2 + x_1 \times 1 + ...$. Note: $11=2+9$, $19=4+15$, $27=6+21$, $35=8+27$. Actually: for row 1: $2+3=5$, not 11. Let's try: $y = 2x_1 + x_2 + 2$: $2(2)+3+2=9$, not 11. Try $y = 3x_1 + x_2 + 2$: $3(2)+3+2=11$ ✓; $3(4)+5+2=19$ ✓; $3(6)+7+2=27$ ✓; $3(8)+9+2=35$ ✓.

So the true relationship is $y = 2 + 3x_1 + x_2$.

**What will `model.coef_` and `model.intercept_` print?**

- [ ] **A.** `coef_ = [2.0, 1.0]`, `intercept_ = 3.0`
- [ ] **B.** `coef_ = [3.0, 1.0]`, `intercept_ = 2.0`
- [ ] **C.** `coef_ = [1.0, 1.0]`, `intercept_ = 0.0`
- [ ] **D.** `coef_ = [4.0, 2.0]`, `intercept_ = 1.0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Coefficients [2.0, 1.0] and intercept 3.0 would produce: $3 + 2(2) + 1(3) = 3+4+3=10 \ne 11$. This does not match the training data.
- **Option B (Correct):** The true relationship is $y = 2 + 3x_1 + x_2$. Scikit-learn's `LinearRegression` will recover the exact parameters when the data lies perfectly on the linear surface with no noise. `model.coef_` will be `[3.0, 1.0]` (coefficients for $x_1$ and $x_2$ respectively) and `model.intercept_` will be `2.0`. Verifying: $2 + 3(2) + 1(3) = 2+6+3=11$ ✓; $2 + 3(4) + 1(5) = 2+12+5=19$ ✓.
- **Option C (Incorrect):** Coefficients [1.0, 1.0] and intercept 0.0 produce: $0 + 1(2) + 1(3) = 5 \ne 11$. This does not fit the data.
- **Option D (Incorrect):** Coefficients [4.0, 2.0] and intercept 1.0 produce: $1 + 4(2) + 2(3) = 1+8+6=15 \ne 11$. This does not fit the data.

</details>

---

### 94. Question

A senior data scientist presents a regression problem where the dataset has one outlier: 99 predictions have errors of exactly 1.0, but one prediction has an error of 20.0.

**Calculate the exact MAE and MSE for this 100-observation dataset, and identify which metric is more affected by the outlier.**

- [ ] **A.** MAE = 1.19; MSE = 5.0; MAE is more affected.
- [ ] **B.** MAE = 1.19; MSE = 4.99; MSE is far more affected because $20^2 = 400$ vs. $|20| = 20$.
- [ ] **C.** MAE = 1.0; MSE = 5.0; both are equally affected.
- [ ] **D.** MAE = 1.19; MSE = 4.99; both are equally affected.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The MAE calculation is correct (1.19), but MSE ≠ 5.0. Let's compute: MAE = $(99 \times 1 + 1 \times 20) / 100 = (99 + 20)/100 = 119/100 = 1.19$. MSE = $(99 \times 1^2 + 1 \times 20^2) / 100 = (99 + 400)/100 = 499/100 = 4.99$. Option A's MSE of 5.0 is wrong; it should be 4.99. Also, MSE is far more affected than MAE by the outlier, not less.
- **Option B (Correct):** MAE = $(99 \times 1 + 20)/100 = 119/100 = 1.19$. MSE = $(99 \times 1 + 400)/100 = 499/100 = 4.99$. The outlier contributes 20 to MAE's numerator (vs. 1 for normal observations — 20× more). It contributes 400 to MSE's numerator (vs. 1 for normal observations — 400× more). Squaring the outlier error inflates its MSE contribution by a factor of $20^2 / 20 = 20$ compared to its MAE contribution. MSE is far more sensitive to the single outlier.
- **Option C (Incorrect):** MAE = 1.0 would require all 100 errors to be exactly 1.0, but one is 20.0. MAE ≠ 1.0; it must be higher due to the outlier. Also, the metrics are not equally affected.
- **Option D (Incorrect):** The numerical values (MAE = 1.19, MSE = 4.99) are correct, but "both are equally affected" is wrong. The outlier inflates MAE from 1.0 (if all errors were 1) to 1.19 — an increase of 0.19 (19%). It inflates MSE from 1.0 (if all errors were 1) to 4.99 — an increase of 3.99 (399%). MSE is massively more affected.

</details>

---

### 95. Question

An ML engineer is implementing a complete from-scratch training loop that includes both slope and intercept. She wants to write the most efficient list comprehension to compute both `grad_m` and `grad_c` in a single pass over the data. She starts with:

```python
errors = [p - y for p, y in zip(preds, y_values)]
```

**Which of the following correctly computes both gradients without repeating the iteration?**

- [ ] **A.** `grad_m = sum(errors) / n`; `grad_c = sum(e * x for e, x in zip(errors, x_values)) / n`
- [ ] **B.** `grad_m = sum(e * x for e, x in zip(errors, x_values)) / n`; `grad_c = sum(errors) / n`
- [ ] **C.** `grad_m = sum(errors) * sum(x_values) / n`; `grad_c = sum(errors) / n`
- [ ] **D.** `grad_m = sum(e + x for e, x in zip(errors, x_values)) / n`; `grad_c = sum(errors) / n`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This assigns the intercept gradient formula to `grad_m` and the slope gradient formula to `grad_c` — the formulas are swapped. `grad_m` requires multiplying each error by its $x$ value; `grad_c` uses the simple average of errors.
- **Option B (Correct):** The slope gradient $\text{grad}_m = \frac{1}{n}\sum e_i \cdot x_i$ is correctly computed by `sum(e * x for e, x in zip(errors, x_values)) / n`. The intercept gradient $\text{grad}_c = \frac{1}{n}\sum e_i$ is correctly computed by `sum(errors) / n`. Both formulas are correct, and their labels are correct.
- **Option C (Incorrect):** `sum(errors) * sum(x_values) / n` computes the product of two sums divided by $n$ — this is not the same as $\frac{1}{n}\sum e_i x_i$. The sum of products ($\sum e_i x_i$) is not equal to the product of sums ($\sum e_i \times \sum x_i$) — this is a fundamental property of algebra. The correct formula requires pairing each error with its corresponding $x$ value.
- **Option D (Incorrect):** `sum(e + x for e, x in zip(errors, x_values)) / n` computes the average of $(e_i + x_i)$ — the sum of each error and its $x$ value — which is not the gradient of the slope. The slope gradient requires the product $e_i \times x_i$, not the sum $e_i + x_i$.

</details>

---

### 96. Question

A researcher has a simple regression model $\hat{y} = mx + c$ with $m = 4$ and $c = -10$. She wants to find the input $x$ that produces a prediction of exactly $\hat{y} = 30$.

**Which of the following are correct approaches to find $x$?** (Select all that apply.)

- [ ] **A.** Set $4x - 10 = 30$, solve: $4x = 40$, therefore $x = 10$.
- [ ] **B.** Set $x = (\hat{y} - c) / m = (30 - (-10)) / 4 = 40 / 4 = 10$.
- [ ] **C.** Set $x = \hat{y} / m = 30 / 4 = 7.5$, ignoring the intercept.
- [ ] **D.** Set $x = (\hat{y} + c) / m = (30 + (-10)) / 4 = 20 / 4 = 5$, adding rather than subtracting the intercept.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B**

#### Explanations:
- **Option A (Correct):** Substituting $\hat{y} = 30$ into $4x - 10 = 30$: add 10: $4x = 40$; divide: $x = 10$. Verification: $4(10) - 10 = 40 - 10 = 30$ ✓. This is the direct algebraic approach.
- **Option B (Correct):** Rearranging $\hat{y} = mx + c$ for $x$: $x = (\hat{y} - c) / m$. Substituting: $x = (30 - (-10)) / 4 = (30 + 10) / 4 = 40 / 4 = 10$ ✓. This is the general formula for finding the input from a desired output in a linear model, and it produces the same result as Option A.
- **Option C (Incorrect):** Ignoring the intercept gives $x = 30/4 = 7.5$. Verifying: $4(7.5) - 10 = 30 - 10 = 20 \ne 30$. The intercept must be accounted for in the inverse calculation.
- **Option D (Incorrect):** Adding the intercept instead of subtracting it gives $x = (30 + (-10))/4 = 20/4 = 5$. Verifying: $4(5) - 10 = 20 - 10 = 10 \ne 30$. The correct rearrangement of $mx + c = \hat{y}$ is $x = (\hat{y} - c)/m$ — the intercept is subtracted from both sides, not added.

</details>

---

### 97. Question

A senior engineer explains to a junior colleague: "For the from-scratch gradient descent loop, the number of steps (iterations) is a hyperparameter — just like the learning rate."

**Which of the following correctly distinguishes hyperparameters from model parameters in the context of linear regression?**

- [ ] **A.** Both the learning rate and $m$ are hyperparameters; neither is learned from data.
- [ ] **B.** The slope $m$ and intercept $c$ are model parameters learned by fitting (via gradient descent or the Normal Equation). The learning rate and number of steps are hyperparameters — settings chosen before training that control the optimization process, not values learned from data.
- [ ] **C.** The learning rate is a model parameter because gradient descent uses it to update $m$ and $c$.
- [ ] **D.** All four values ($m$, $c$, learning rate, number of steps) are hyperparameters; none are learned from data.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The slope $m$ is a model parameter, not a hyperparameter. $m$ is the value that gradient descent is actively learning from data by minimizing the loss function. Hyperparameters (learning rate, number of steps) are set externally before training begins and are not adjusted by the gradient descent updates themselves.
- **Option B (Correct):** In linear regression: **Model parameters** are $m$ (slope) and $c$ (intercept) — they are computed/learned during training to minimize the loss on the training data. **Hyperparameters** are the learning rate ($\alpha$) and number of training iterations — they are chosen by the engineer before training and control how the optimization proceeds, but they are not themselves learned from the data. The distinction is: model parameters change during training as the model learns; hyperparameters are fixed external settings.
- **Option C (Incorrect):** The learning rate is not a model parameter. It is a configuration setting for the optimization algorithm (gradient descent). The fact that it appears in the update rule for $m$ and $c$ doesn't make it a model parameter — it is an external control constant, not a variable that gradient descent optimizes or adjusts.
- **Option D (Incorrect):** $m$ and $c$ are definitively model parameters — they are the values that gradient descent updates at every step of training, guided by the gradient of the loss. Calling all four quantities hyperparameters misclassifies the role of $m$ and $c$.

</details>

---

### 98. Question

A data science team is comparing two evaluation approaches for a regression model:

- **Approach 1:** Evaluate on the training set (`model.score(X_train, y_train)`) → $R^2 = 0.95$
- **Approach 2:** Evaluate on the test set (`model.score(X_test, y_test)`) → $R^2 = 0.67$

**Which of the following are valid conclusions from these results?** (Select all that apply.)

- [ ] **A.** The large gap between training $R^2$ (0.95) and test $R^2$ (0.67) suggests the model is overfitting the training data.
- [ ] **B.** The test $R^2$ of 0.67 is the more reliable estimate of how the model will generalize to new, unseen data.
- [ ] **C.** The training $R^2$ of 0.95 means the model is excellent and the test $R^2$ can be ignored.
- [ ] **D.** To improve generalization, the team could consider collecting more training data, adding regularization, or simplifying the model.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** A training $R^2$ of 0.95 paired with a test $R^2$ of 0.67 is a significant gap indicating overfitting. The model has learned to fit the training data very well (including noise and idiosyncrasies of that sample) but this performance does not transfer to unseen observations. This is the classic train-test gap that signals overfitting.
- **Option B (Correct):** The test set is data that was not used during training. Evaluating on the test set simulates how the model performs on genuinely new observations. The test $R^2$ of 0.67 is therefore the more reliable indicator of real-world generalization performance, not the training $R^2$.
- **Option C (Incorrect):** A high training $R^2$ alone does not make a model excellent. The purpose of a model is to generalize to new data, not to memorize training data. The test $R^2$ of 0.67 is the relevant measure of the model's usefulness. Ignoring the test performance while celebrating training performance is the core mistake of overfitting.
- **Option D (Correct):** Strategies to reduce overfitting and improve generalization include: (1) collecting more training data (more samples reduce the model's ability to memorize any specific sample); (2) regularization (constrains model complexity to reduce variance); (3) feature reduction or simpler model structure (reduces degrees of freedom for overfitting). All three are valid and standard approaches.

</details>

---

### 99. Question

A graduate student wants to understand what happens to the MSE gradient when the model is already at the global minimum. She asks: "What is the gradient at the optimal parameters?"

**Which statement is correct?**

- [ ] **A.** At the global minimum, the gradient is at its maximum value because the loss is at its minimum.
- [ ] **B.** At the global minimum, the gradient of the MSE loss with respect to all parameters equals zero — this is precisely the mathematical condition that defines a minimum.
- [ ] **C.** At the global minimum, the gradient equals the learning rate.
- [ ] **D.** At the global minimum, the gradient is undefined because the minimum is a singular point.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This inverts the relationship. At the minimum, the loss is at its lowest point — the surface is flat at that point, meaning the gradient (slope of the loss surface) equals zero, not its maximum. Maximums of the gradient occur on the steepest parts of the loss surface, far from the minimum.
- **Option B (Correct):** By definition, a minimum of a differentiable function occurs where the gradient equals zero ($\nabla L = 0$). At the global minimum of the MSE loss, $\partial L / \partial m = 0$ and $\partial L / \partial c = 0$. This flat, zero-gradient condition means gradient descent makes no further updates: $m_{\text{new}} = m - \alpha \times 0 = m$. The parameters do not change, which is correct — convergence has been achieved.
- **Option C (Incorrect):** The gradient is a quantity derived from the data and current parameters; it has no inherent relationship to the learning rate. At the minimum, the gradient equals zero regardless of the learning rate's value. The learning rate scales the gradient in the update formula but does not equal the gradient at any point.
- **Option D (Incorrect):** The MSE loss function is a smooth, continuously differentiable quadratic function. It has a well-defined gradient at every point, including the minimum. The gradient at the minimum is zero (not undefined) — zero is a perfectly valid, well-defined gradient value.

</details>

---

### 100. Question

A capstone review question: An ML engineer builds a complete pipeline to predict apartment monthly rent from three features: `SquareMeters`, `Balconies`, and `FloorNumber`. She uses the following code:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

X = df[["SquareMeters", "Balconies", "FloorNumber"]]
y = df["MonthlyRent"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Num coefficients:", len(model.coef_))
print("R2:", model.score(X_test, y_test))
print("MAE:", mean_absolute_error(y_test, y_pred))
```

**Which of the following statements about this pipeline are correct?** (Select all that apply.)

- [ ] **A.** `model.coef_` will contain exactly 3 values — one for each input feature.
- [ ] **B.** `model.intercept_` will contain exactly 1 scalar value — the shared baseline rent when all three features are zero.
- [ ] **C.** `model.score(X_test, y_test)` returns the test-set $R^2$ — the proportion of variance in rent explained by the model on unseen data.
- [ ] **D.** Using `test_size=0.2` with a dataset of 1000 rows, `X_train` will have 800 rows and `X_test` will have 200 rows.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, C, D**

#### Explanations:
- **Option A (Correct):** `model.coef_` in scikit-learn's `LinearRegression` is a 1D numpy array with exactly one element per input feature column. With 3 features (SquareMeters, Balconies, FloorNumber), `model.coef_` has exactly 3 values. `len(model.coef_)` prints `3`.
- **Option B (Correct):** In standard linear regression (single-output), `model.intercept_` is a scalar float — the single baseline predicted value when all input features are zero. There is always exactly one intercept regardless of the number of features. It represents $\beta_0$ in the equation $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$.
- **Option C (Correct):** For regression models in scikit-learn, `model.score(X_test, y_test)` computes and returns the $R^2$ (coefficient of determination) on the provided test set. Since `X_test` and `y_test` were held out during training (not used in `model.fit()`), this is a valid estimate of generalization performance on unseen data.
- **Option D (Correct):** `test_size=0.2` allocates 20% of the data to the test set: $1000 \times 0.2 = 200$ test rows and $1000 \times 0.8 = 800$ training rows. `X_train` has 800 rows and `X_test` has 200 rows.

</details>
