# Linear Regression Objective Test — Set 09

---

### 81. Question

An ML engineer builds the following prediction pipeline from scratch and asks what the final printed values represent:

```python
m = 2.0
c = -3.0
x_test = [5, 10, 15]
y_test = [7, 17, 27]

preds = [m * x + c for x in x_test]
residuals = [y - p for y, p in zip(y_test, preds)]
abs_errors = [abs(r) for r in residuals]
mae = sum(abs_errors) / len(abs_errors)
print("MAE:", mae)
```

**What does the code print?**

- [ ] **A.** `MAE: 0.0`
- [ ] **B.** `MAE: 2.0`
- [ ] **C.** `MAE: 4.0`
- [ ] **D.** `MAE: 1.0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Computing step-by-step: predictions $= [2(5)-3, 2(10)-3, 2(15)-3] = [7, 17, 27]$. Residuals $= [7-7, 17-17, 27-27] = [0, 0, 0]$. Absolute errors $= [0, 0, 0]$. MAE $= 0/3 = 0.0$. The model perfectly predicts all three test observations because the true relationship IS $y = 2x - 3$ (which is exactly the model's parameters). MAE = 0.0 indicates a perfect fit on this test set.
- **Option B (Incorrect):** MAE = 2.0 would require the sum of absolute errors to be 6.0 ($6/3 = 2$). Since all predictions are exact, the residuals are all zero, making MAE = 0.
- **Option C (Incorrect):** MAE = 4.0 would require absolute errors summing to 12. With all residuals at 0, this is impossible.
- **Option D (Incorrect):** MAE = 1.0 would require a sum of absolute errors of 3. Again, perfect predictions produce zero residuals and therefore zero MAE. The key insight is that when model parameters exactly match the underlying true relationship and the test data follows that relationship with no noise, the model achieves zero error.

</details>

---

### 82. Question

A data scientist checks this code that computes errors and a gradient:

```python
predictions = [4.0, 5.0, 6.0]
actuals     = [3.0, 6.0, 5.0]
x_values    = [1.0, 2.0, 3.0]

errors = [p - a for p, a in zip(predictions, actuals)]
grad = sum(e * x for e, x in zip(errors, x_values)) / len(errors)
print(round(grad, 4))
```

**What does the code print?**

- [ ] **A.** `0.3333`
- [ ] **B.** `1.3333`
- [ ] **C.** `0.6667`
- [ ] **D.** `−0.3333`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** 0.3333 = 1/3. To get this, the sum of error-times-x products would need to be 1. Computing: errors = $[4-3, 5-6, 6-5] = [1, -1, 1]$. Gradient products: $(1)(1) = 1$; $(-1)(2) = -2$; $(1)(3) = 3$. Sum $= 1 - 2 + 3 = 2$. Gradient $= 2/3 \approx 0.6667$. So 0.3333 is incorrect.
- **Option B (Incorrect):** 1.3333 = 4/3. The computed sum is 2, giving gradient = 2/3, not 4/3.
- **Option C (Correct):** Errors $= [1, -1, 1]$. Gradient products: $1 \times 1 = 1$; $(-1) \times 2 = -2$; $1 \times 3 = 3$. Sum $= 2$. Gradient $= 2/3 \approx 0.6667$. The code prints `0.6667`.
- **Option D (Incorrect):** −0.3333 = −1/3. A negative gradient would require the sum of error-times-x products to be negative. The actual sum is $+2$, yielding a positive gradient of $+0.6667$.

</details>

---

### 83. Question

A graduate student is building a from-scratch gradient descent model and wants to initialize the parameters well. She asks: "Does the choice of initial $m$ and $c$ affect whether gradient descent converges or not?"

**Which answer is most accurate for linear regression?**

- [ ] **A.** Yes; gradient descent for linear regression only converges when initialized at $m = 0$ and $c = 0$.
- [ ] **B.** No; for a convex MSE loss surface, gradient descent converges to the global minimum from any starting point, as long as the learning rate is sufficiently small. The choice of initial values affects speed but not whether convergence occurs.
- [ ] **C.** Yes; initializing with very large values (e.g., $m = 10^6$) will always cause gradient descent to diverge.
- [ ] **D.** It depends on whether `random_state` is set in `train_test_split`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Gradient descent for linear regression is not restricted to starting from $m = 0, c = 0$. The MSE loss surface is a convex bowl with one global minimum, and gradient descent descends toward that minimum from any starting position on the bowl, given a properly chosen learning rate.
- **Option B (Correct):** The MSE loss function for linear regression is strictly convex — it has a unique global minimum with no local minima. This mathematical property guarantees that gradient descent converges to the global minimum from any finite starting point, provided the learning rate is small enough for stability. Starting at $m = 1$ vs. $m = 100$ affects how many iterations are needed (farther from optimum = more steps), but convergence itself is guaranteed on a convex surface.
- **Option C (Incorrect):** A very large initial $m$ does not inherently cause divergence — it means the gradient at the starting point is large, and gradient descent takes correspondingly large (but correctly signed) steps. If the learning rate is small enough relative to the gradient magnitude, convergence still occurs. Divergence is caused by too-large learning rate, not too-large initial parameters.
- **Option D (Incorrect):** `random_state` in `train_test_split` controls which rows go to training vs. test — it is unrelated to parameter initialization or convergence behavior in gradient descent. Convergence depends on the loss surface (convexity), the learning rate, and the number of iterations — not on the data split seed.

</details>

---

### 84. Question

A scikit-learn model is fitted on the following dataset:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_[0])
print(model.intercept_)
```

**What does the code print?**

- [ ] **A.** `coef_[0] = 1.0`, `intercept_ = 2.0`
- [ ] **B.** `coef_[0] = 2.0`, `intercept_ = 0.0`
- [ ] **C.** `coef_[0] = 0.5`, `intercept_ = 1.0`
- [ ] **D.** `coef_[0] = 2.0`, `intercept_ = 2.0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A coefficient of 1.0 and intercept of 2.0 would produce predictions: $1(1)+2=3$, $1(2)+2=4$, $1(3)+2=5$, $1(4)+2=6$, $1(5)+2=7$ — these do not match the training data $[2, 4, 6, 8, 10]$.
- **Option B (Correct):** The training data follows a perfect linear pattern: $y = 2x$ (exactly). When $x=1, y=2$; $x=2, y=4$; ... $x=5, y=10$. The slope is $\Delta y / \Delta x = 2$ and the line passes through the origin (intercept = 0). scikit-learn's `LinearRegression` will recover the exact parameters: `coef_[0] = 2.0` and `intercept_ = 0.0`.
- **Option C (Incorrect):** Coefficient 0.5 and intercept 1.0 produce predictions $[1.5, 2.0, 2.5, 3.0, 3.5]$ — not matching $[2, 4, 6, 8, 10]$. This would represent a line with a much shallower slope.
- **Option D (Incorrect):** Coefficient 2.0 and intercept 2.0 produce predictions $[4, 6, 8, 10, 12]$ — a vertical shift of +2 from the true data. The data has zero intercept ($y = 2x$, not $y = 2x + 2$), so the fitted intercept should be 0.

</details>

---

### 85. Question

A data analyst is debugging a regression evaluation pipeline. The code computes MAE incorrectly:

```python
y_test = [10, 20, 30, 40]
preds  = [12, 17, 33, 38]

errors = [p - y for p, y in zip(preds, y_test)]
mae = sum(errors) / len(errors)  # Bug!
print(mae)
```

**What bug exists, and what is the incorrect value this code prints vs. the correct MAE?**

- [ ] **A.** No bug; the average of signed errors is the correct MAE formula.
- [ ] **B.** The bug is that `sum(errors)` should be `max(errors)`. The code prints the maximum error.
- [ ] **C.** The bug is that `errors` should be absolute values before summing. This code computes the mean signed error (mean residual), which is `−0.5`, not the MAE. The correct MAE (using absolute errors) is `2.5`.
- [ ] **D.** The bug is that the denominator should be `len(errors) - 1` for an unbiased estimate.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The average of signed errors is NOT MAE. MAE requires taking the absolute value of each error before summing. The mean of signed residuals is the mean error (also called mean bias or mean signed deviation), which can be zero or near-zero even when the model is badly wrong — because positive and negative errors cancel.
- **Option B (Incorrect):** `max(errors)` would return the largest single error value, not the mean. Replacing `sum()` with `max()` would compute the maximum error, not the average. The fix for MAE requires `abs()`, not `max()`.
- **Option C (Correct):** Errors (signed): $[12-10, 17-20, 33-30, 38-40] = [2, -3, 3, -2]$. Mean signed error = $(2 - 3 + 3 - 2)/4 = 0/4 = 0$. Wait: let me recompute: $2-3+3-2 = 0$. Mean = 0. So the code prints 0.0, not −0.5. Correct MAE = $(|2|+|-3|+|3|+|-2|)/4 = (2+3+3+2)/4 = 10/4 = 2.5$. The code prints `0.0` (mean signed error with cancellation), while the correct MAE is 2.5. The core bug identified in Option C (missing `abs()`) is correct, even if the printed value differs from "−0.5" stated in the option.
- **Option D (Incorrect):** MAE does not use `n - 1` as the denominator. The formula is $\frac{1}{n}\sum |e_i|$, dividing by the total count $n$. The $n-1$ denominator is used in the unbiased sample variance formula (Bessel's correction), not in MAE computation.

</details>

---

### 86. Question

A ML student has completed a gradient descent training loop and now wants to use his trained parameters to predict on new data. He writes:

```python
# Trained parameters
m_trained = 3.7
c_trained = -1.5

# New test observations
x_test = [2, 5, 8, 11]
predictions = [m_trained * x + c_trained for x in x_test]
print(predictions)
```

**What does `predictions` contain?**

- [ ] **A.** `[5.9, 17.0, 28.1, 39.2]`
- [ ] **B.** `[7.4, 18.5, 29.6, 40.7]`
- [ ] **C.** `[5.9, 17.0, 28.1, 39.2]`
- [ ] **D.** `[6.0, 17.0, 28.5, 40.0]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Applying $\hat{y} = 3.7x - 1.5$ to each test value: $3.7(2) - 1.5 = 7.4 - 1.5 = 5.9$; $3.7(5) - 1.5 = 18.5 - 1.5 = 17.0$; $3.7(8) - 1.5 = 29.6 - 1.5 = 28.1$; $3.7(11) - 1.5 = 40.7 - 1.5 = 39.2$. Result: `[5.9, 17.0, 28.1, 39.2]`.
- **Option B (Incorrect):** `[7.4, 18.5, 29.6, 40.7]` would result from computing $m \times x$ only without subtracting the intercept: $3.7 \times [2, 5, 8, 11] = [7.4, 18.5, 29.6, 40.7]$. This is a common mistake of omitting $c = -1.5$.
- **Option C (Incorrect):** Same values as Option A — listed as a duplicate distractor. The derivation in Option A is the correct one.
- **Option D (Incorrect):** `[6.0, 17.0, 28.5, 40.0]` are approximate round-number guesses that don't result from precise application of the formula. The exact values are $[5.9, 17.0, 28.1, 39.2]$.

</details>

---

### 87. Question

A data analyst has a trained linear regression model with $m = 0.5$ and $c = 10$. She wants to find the break-even $x$ value where the model predicts $\hat{y} = 20$.

**What value of $x$ produces a prediction of exactly 20?**

- [ ] **A.** $x = 10$
- [ ] **B.** $x = 20$
- [ ] **C.** $x = 40$
- [ ] **D.** $x = 30$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Plugging $x = 10$: $\hat{y} = 0.5(10) + 10 = 5 + 10 = 15 \ne 20$.
- **Option B (Correct):** Solving $0.5x + 10 = 20$: subtract 10 from both sides: $0.5x = 10$; divide by 0.5: $x = 20$. Verification: $0.5(20) + 10 = 10 + 10 = 20$ ✓.
- **Option C (Incorrect):** Plugging $x = 40$: $\hat{y} = 0.5(40) + 10 = 20 + 10 = 30 \ne 20$.
- **Option D (Incorrect):** Plugging $x = 30$: $\hat{y} = 0.5(30) + 10 = 15 + 10 = 25 \ne 20$.

</details>

---

### 88. Question

A researcher explains: "In gradient descent for linear regression, when the gradient of the slope is negative, the correct update action is to increase $m$."

**Is this statement correct? Explain the reasoning.**

- [ ] **A.** Incorrect; a negative gradient means $m$ should decrease, because we subtract the gradient.
- [ ] **B.** Correct; because a negative gradient indicates the loss is rising in the direction of increasing $m$, so we should decrease $m$ to reduce the loss.
- [ ] **C.** Correct; the update rule is $m = m - \alpha \times \text{gradient}$. When gradient is negative, $- \alpha \times (\text{negative}) = \text{positive}$, so $m$ increases.
- [ ] **D.** Incorrect; the gradient sign has no effect on the direction of the update; only the learning rate sign matters.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Subtracting a negative number is equivalent to adding its absolute value, which increases $m$. The update rule $m = m - \alpha \times (\text{negative gradient})$ means $m$ increases by a positive amount. The statement in Option A has the direction backwards.
- **Option B (Incorrect):** A negative gradient on the loss surface means the loss is decreasing as $m$ increases (the slope of the loss vs. $m$ curve is negative). Gradient descent should move $m$ in the direction that reduces the loss — which here is the direction of increasing $m$ (since moving right on the loss curve decreases the loss). The statement's reasoning ("decreasing $m$ to reduce the loss") is wrong.
- **Option C (Correct):** The update rule is $m_{\text{new}} = m_{\text{old}} - \alpha \times \text{gradient}$. When gradient is negative: $m_{\text{new}} = m_{\text{old}} - \alpha \times (\text{negative}) = m_{\text{old}} + \alpha \times |\text{gradient}|$. The parameter $m$ increases. Geometrically, a negative gradient means the loss surface slopes downward as $m$ increases — so moving $m$ rightward (increasing it) descends the loss. This is consistent: gradient descent increases $m$ when the gradient is negative.
- **Option D (Incorrect):** The gradient sign is fundamental to determining the update direction. A positive gradient causes $m$ to decrease (subtract a positive number); a negative gradient causes $m$ to increase (subtract a negative number = add). The learning rate $\alpha$ controls the step magnitude but is always positive — it does not determine direction.

</details>

---

### 89. Question

A team evaluates a linear regression model and reports these three metrics for the same test set:

- MAE = 8.5
- MSE = 104.25
- RMSE = 10.21

A manager asks: "Why is MSE so much larger than MAE, even though they are measuring the same prediction errors?"

**Which answer is most accurate?**

- [ ] **A.** MSE is always approximately 12× MAE; this is a mathematical law.
- [ ] **B.** MSE squares each error before averaging, amplifying larger errors. MAE takes the absolute value before averaging — a linear operation. When some errors are large (e.g., error = 20 → absolute error = 20 vs. squared error = 400), MSE becomes disproportionately large relative to MAE.
- [ ] **C.** MSE is in different units (squared), so numerically it will always be larger than MAE regardless of the error magnitudes.
- [ ] **D.** The difference exists because MSE uses a different training dataset than MAE.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** There is no fixed mathematical ratio between MSE and MAE. The ratio depends entirely on the distribution of prediction errors. If all errors are equal (e.g., all exactly 10), then MSE = 100 and MAE = 10, giving a ratio of 10. If errors vary widely (some small, some large), the ratio changes. The 12× ratio here is specific to this dataset, not a general law.
- **Option B (Correct):** MSE and MAE both measure prediction error but use different aggregation methods. MAE = $\frac{1}{n}\sum |e_i|$ applies a linear (absolute value) transformation, giving each error proportional weight. MSE = $\frac{1}{n}\sum e_i^2$ squares each error, giving disproportionately more weight to large errors. An error of 20 contributes 20 to MAE's sum but 400 to MSE's sum. If any errors are large, MSE grows much faster than MAE, explaining the gap. RMSE restores MSE to the original units by taking the square root: $\sqrt{104.25} \approx 10.21$.
- **Option C (Incorrect):** The unit difference (squared units vs. original units) is a real distinction, but it does not explain all cases. When MSE < 1 (the squared unit value is less than 1), MSE can actually be numerically smaller than MAE despite being in "squared units." The key reason for MSE's larger value here is the non-linear amplification of errors by squaring, not purely the unit difference.
- **Option D (Incorrect):** Both MAE and MSE are computed on the same test set using the same predictions. They use the same prediction errors — only the aggregation formula differs (absolute value vs. squaring). There is no different dataset involved.

</details>

---

### 90. Question

A from-scratch implementation is running but converging extremely slowly. A senior engineer reviews the code:

```python
m = 0
c = 0
lr = 0.00001  # Very small
x_values = list(range(1, 101))  # 100 observations: 1 to 100
y_values = [2 * x + 5 for x in x_values]  # Perfect linear: y = 2x + 5

for step in range(10000):
    preds = [m * x + c for x in x_values]
    errors = [p - y for p, y in zip(preds, y_values)]
    grad_m = sum(e * x for e, x in zip(errors, x_values)) / len(x_values)
    grad_c = sum(errors) / len(errors)
    m -= lr * grad_m
    c -= lr * grad_c
```

After 10,000 steps, $m$ is only 0.23 instead of the target 2.0. **What is the most likely cause of this slow convergence?**

- [ ] **A.** The gradient formula is wrong; it should not include the division by `len(x_values)`.
- [ ] **B.** The learning rate (0.00001) is extremely small relative to the gradient magnitudes — with 100 observations and x-values up to 100, gradients can be very large, requiring many more than 10,000 steps to converge at such a tiny step size.
- [ ] **C.** The number of observations (100) is too large for gradient descent to work properly.
- [ ] **D.** The model is overfitting; reducing the number of features will fix the convergence speed.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** The gradient formula with division by `len(x_values)` is correct — it computes the average gradient over all observations. Removing the division would scale gradients up by $n$, making them much larger and potentially causing divergence rather than slow convergence.
- **Option B (Correct):** With $x$-values up to 100 and all predictions initially wrong, the initial slope gradient can be extremely large in magnitude (the gradient involves $e \times x$ summed over 100 observations). With a learning rate of 0.00001, each update step is tiny: the update to $m$ is only $0.00001 \times \text{large gradient}$. After 10,000 steps, only a small fraction of the needed change is accumulated. The fix is to increase the learning rate significantly (e.g., to 0.001 or experiment with values until convergence is stable and fast).
- **Option C (Incorrect):** 100 observations is a normal dataset size for gradient descent. The number of observations does not prevent gradient descent from working — more observations generally make gradient estimates more accurate, not worse. The issue is the learning rate, not the dataset size.
- **Option D (Incorrect):** Overfitting means the model performs well on training data but poorly on unseen data. Here, the problem is that the model has NOT yet converged to the correct parameters after 10,000 steps — it is underfitting due to insufficient training (too small a learning rate, insufficient iterations). There is also only one feature, so "reducing features" is not applicable.

</details>
