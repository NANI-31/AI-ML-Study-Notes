### 1. Question

A DevOps engineer trains the from-scratch linear regression loop with the learning rate set far above the usual small starting value (such as 0.01) while keeping the number of steps fixed. During training, the slope m swings wildly between very different values from one step to the next instead of settling down.

**What does this behavior indicate, and how should it be fixed?**

### Options

- [ ] The number of steps is the only cause of the issue; the learning rate must remain fixed at 0.01.
- [x] **The learning rate is too large, causing overshoot and unstable, oscillating updates; reduce the learning rate.** _(Correct Answer)_
- [ ] The learning rate is too small, so convergence is simply slow; increase the number of steps to let m settle.
- [ ] The gradient formula must be wrong, because gradient descent always converges regardless of the learning rate used.

### Answer Explanation

- **A:** Wrong — it wrongly forbids adjusting the learning rate, when the learning rate is exactly the setting that needs to change here.
- **B (Correct):** Correct — an oscillating or blowing-up parameter across steps is the signature of too large a learning rate, which causes each update to overshoot the ideal value; the fix is to reduce it, since it is a tunable setting found by experimentation rather than solved for mathematically.
- **C:** Wrong — a too-small learning rate produces the opposite symptom, slow steady movement toward a value, not wild oscillation, so this diagnosis does not match the described behavior.
- **D:** Wrong — the described instability is a well-understood consequence of a poorly chosen learning rate, not evidence of a broken gradient formula; gradient descent does not always converge regardless of the learning rate.

---

### 2. Question

A data scientist runs the following in a Python shell:

```python
a = [2, 4, 6]
b = [10, 20, 30]
result = list(zip(a, b))

```

**Which statement correctly describes how `zip(a, b)` combines these two lists into `result`?**

### Options

- [ ] It reverses the order, pairing `b`'s value with `a`'s value at each matching position.
- [ ] It concatenates `a` and `b` end-to-end into a single flat list of six values, forming no pairs.
- [ ] It groups each entire list into one combined tuple, instead of pairing individual elements.
- [x] **It pairs corresponding elements from `a` and `b` positionally, producing one tuple per index.** _(Correct Answer)_

### Answer Explanation

- **A:** Wrong — this reverses the order of elements within each pair (`b`'s value first), which is not how `zip()` pairs positional elements from the arguments in the order given; the actual result is `[(2, 10), (4, 20), (6, 30)]`, not `[(10, 2), (20, 4), (30, 6)]`.
- **B:** Wrong — this simply concatenates the two lists rather than pairing elements together into tuples; `zip()` does not flatten its arguments into one combined list.
- **C:** Wrong — this groups each entire list into a single tuple instead of pairing elements positionally across the two lists; `zip()` never collapses an entire argument into one tuple.
- **D (Correct):** Correct — `zip()` pairs up corresponding elements from the two lists in order by index, producing `result = [(2, 10), (4, 20), (6, 30)]`.

---

### 3. Question

A data analyst has already loaded a housing dataset with pandas and explored it with `df.describe()` and `df.corr()`. The analyst now fits a scikit-learn `LinearRegression` model using the following setup:

```python
X = df[["SquareFeet", "Bedrooms", "AgeYears"]]
y = df["PriceLakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
model.fit(X_train, y_train)

```

**Which of the following statements about this setup are correct?**

### Options

- [ ] Since three variables are used, there will also be three separate intercept terms, one per variable.
- [ ] The coefficients define a single best-fit line through the three-dimensional input space.
- [x] **There will be exactly three coefficients (one per input variable) and exactly one intercept.** _(Correct Answer)_
- [x] **X holds the independent variables (predictors) and y holds the dependent variable being predicted.** _(Correct Answer)_

### Answer Explanation

- **A:** Wrong — it directly contradicts the rule that there is exactly one intercept no matter how many predictors are used; intercepts are not duplicated per variable.
- **B:** Wrong — with more than one input variable, the best-fit relationship is no longer a single line; it generalizes to a plane (or higher-dimensional surface) through the input space.
- **C (Correct):** Correct — with multiple input variables, each variable gets its own coefficient (three coefficients here, one for square feet, one for bedrooms, one for age), but there is always only a single standalone intercept term.
- **D (Correct):** Correct — X is assigned the independent variables (predictors) — square feet, bedrooms, and age — while y is assigned the dependent variable, the price being predicted.

---

### 4. Question

Machine learning models work by finding a pattern in existing data and using it to predict new values. Before building any model, a product manager wants to get an early sense of which of square feet, bedrooms, and age (years) is likely to matter most for predicting a house's price in a housing dataset.

**Which action should be taken first to identify likely predictors?**

### Options

- [ ] Fit a full LinearRegression model on all three inputs and inspect its MAE on the test set directly.
- [x] **Compute the correlation between each input and price to see how strongly they move together.** _(Correct Answer)_
- [ ] Run train_test_split and check how many rows land in the resulting test set overall.
- [ ] Apply one-hot encoding to every numeric column before examining the raw data values.

### Answer Explanation

- **A:** Wrong — fitting a full model and reading an error metric like MAE requires training first; it is a downstream evaluation step, not an early exploratory one.
- **B (Correct):** Correct — checking correlation between each candidate input and the target is the first, lightweight step for sensing which variables are likely useful predictors, before any model is trained.
- **C:** Wrong — splitting data into train and test sets is a preparation step for training, not a way to learn which variables relate to price.
- **D:** Wrong — one-hot encoding converts categorical columns into numeric form; square feet, bedrooms, and age are already numeric, so encoding does not apply here and reveals nothing about predictive strength.

---

### 5. Question

A senior ML engineer is building a from-scratch gradient-descent training loop for linear regression. The update rule `m = m - learning_rate * gradient` requires computing the derivative of the loss function at every single step, so the engineer needs to pick a loss function that supports this.

**Which loss function should be chosen, and why?**

### Options

- [x] **Squared loss, because it is smoothly differentiable everywhere, unlike the absolute-value loss.** _(Correct Answer)_
- [ ] Absolute loss, because it never inflates the size of large errors the way squaring does.
- [ ] Squared loss, because it always produces a smaller numeric value than absolute loss.
- [ ] Absolute loss, because its derivative is undefined only at an error of exactly zero.

### Answer Explanation

- **A (Correct):** Correct — squared loss is smoothly differentiable at every value of the error, which is exactly what a derivative-based update rule like gradient descent needs; the absolute-value function is not differentiable and is therefore ruled out for this method. Squaring also guarantees a positive value so errors never cancel, and it follows from the reasoning behind maximum likelihood estimation.
- **B:** Wrong — "never inflating large errors" is not the deciding factor here; the loss must be usable inside a derivative-based update rule, and absolute loss fails that requirement.
- **C:** Wrong — squared loss does not always produce a smaller number than absolute loss; for an error whose absolute value exceeds 1, squaring makes it larger, not smaller.
- **D:** Wrong — it understates the problem: since the entire method depends on taking a derivative of the loss at exactly that kink point during training, this non-differentiability makes absolute loss impractical for this gradient-based approach, not merely "undefined only there."

---

### 6. Question

A product manager evaluates a regression model on four test houses, comparing actual price (in lakhs) against the model's predicted price:

| House | Actual (lakhs) | Predicted (lakhs) |
| :---- | :------------- | :---------------- |
| 1     | 50             | 52                |
| 2     | 60             | 55                |
| 3     | 40             | 44                |
| 4     | 70             | 68                |

**Which of the following statements about this model's evaluation are correct?**

### Options

- [ ] RMSE will always be smaller than MSE for any dataset, since a square root reduces a value.
- [x] **The mean squared error (MSE) computed for this four-house test set is 12.25 (lakhs-squared).** _(Correct Answer)_
- [x] **The root mean squared error (RMSE) here is approximately 3.5 lakhs, in the same units as price.** _(Correct Answer)_
- [x] **The mean absolute error (MAE) computed for this four-house test set is 3.25 lakhs.** _(Correct Answer)_

### Answer Explanation

- **A:** Wrong — RMSE is only smaller than MSE when MSE is greater than 1 (as it happens to be here); when MSE is a fraction less than 1, its square root is actually larger, so "always" is a false generalization.
- **B (Correct):** Correct — the squared errors are 4, 25, 16, 4, giving MSE = (4 + 25 + 16 + 4) / 4 = 12.25.
- **C (Correct):** Correct — RMSE is the square root of MSE, $\sqrt{12.25} \approx 3.5$, which brings the error back into the original price units (lakhs) rather than squared units.
- **D (Correct):** Correct — the absolute errors are 2, 5, 4, 2, giving MAE = (2 + 5 + 4 + 2) / 4 = 3.25 lakhs. This also illustrates why squared-error metrics like MSE are more sensitive to a single unusually large deviation (such as house 2's gap of 5) than MAE is — worth remembering when outliers are present, since MSE penalizes that error more heavily by squaring it.

---

### 7. Question

A senior data analyst fits a simple linear regression model relating square feet ($x$) to price in lakhs ($y$) and obtains slope $m = 0.05$ and intercept $c = 10$.

**What price does the model predict for a house with 800 square feet?**

### Options

- [ ] 40 lakhs
- [ ] 60 lakhs
- [x] **50 lakhs** _(Correct Answer)_
- [ ] 45 lakhs

### Answer Explanation

- **A:** Wrong — this is $m \times x$ alone ($0.05 \times 800 = 40$), dropping the intercept term entirely.
- **B:** Wrong — this only results from adding the intercept twice ($40 + 10 + 10 = 60$), double-counting $c$.
- **C (Correct):** Correct — using $y = mx + c$, the prediction is $0.05 \times 800 + 10 = 40 + 10 = 50$ lakhs.
- **D:** Wrong — it does not correspond to any correct or partially-correct application of the formula given these values, and would only arise from an arithmetic slip.

---

### 8. Question

A senior data analyst runs `df.corr()` on a housing dataset and gets the following pairwise correlations against price:

| Pair                     | Correlation (r) |
| :----------------------- | :-------------- |
| SquareFeet vs PriceLakhs | 0.82            |
| Bedrooms vs PriceLakhs   | 0.35            |
| AgeYears vs PriceLakhs   | -0.61           |
| SquareFeet vs Bedrooms   | 0.20            |

**Which of the following interpretations of this correlation output are correct?**

### Options

- [x] **Square feet shows the strongest relationship with price among the three predictors, since 0.82 is the largest absolute correlation.** _(Correct Answer)_
- [ ] Because bedrooms has a lower correlation (0.35) than square feet, bedrooms should be dropped from the model entirely.
- [x] **Age (years) has a negative correlation with price, meaning that older houses tend to be associated with noticeably lower prices.** _(Correct Answer)_
- [ ] A correlation of 0.20 between square feet and bedrooms means including both together will make the coefficients undefined.

### Answer Explanation

- **A (Correct):** Correct — comparing absolute values of the three correlations against price (0.82, 0.35, 0.61), square feet's 0.82 is the largest, making it the input most strongly related to price by this measure.
- **B:** Wrong — correlation with the target is only an early, first-pass indication of usefulness, not a hard rule for exclusion; a lower correlation does not automatically mean a variable contributes nothing once combined with other predictors.
- **C (Correct):** Correct — a negative correlation between age and price means the two move in opposite directions: as age increases, price tends to decrease, i.e., older houses are associated with lower prices.
- **D:** Wrong — a modest correlation between two predictors (0.20) is normal and does not prevent a model from fitting or make its coefficients undefined; predictors are commonly somewhat correlated with each other without breaking the fitting process.

---

### 9. Question

A senior ML engineer is manually tracing through the first update step of the gradient-descent loop for simple linear regression, using the no-intercept form `y_cap = m * x`. The toy dataset and settings are:

```python
x_values = [1, 2, 3]
y_values = [4, 6, 8]
starting m = 0
learning_rate = 0.1

```

Recall the update sequence: `predictions = [m*x for x in x_values]`, `errors = [p - y for p, y in zip(predictions, y_values)]`, `gradient = sum(e*x for e, x in zip(errors, x_values)) / len(x_values)`, then `m = m - learning_rate * gradient`.

**Which of the following statements about this first update step are correct?**

### Options

- [x] **The gradient from the initial errors is negative, since every prediction starts at 0 while actual y is positive.** _(Correct Answer)_
- [x] **After the first update step, the value of m increases from 0 to approximately 1.33.** _(Correct Answer)_
- [ ] The update rule adds the gradient directly to m, so a negative gradient decreases m on the first step.
- [ ] If the learning rate were increased to 1.0, the resulting m after one step would be smaller than with 0.1.

### Answer Explanation

- **A (Correct):** Correct — with $m = 0$, every prediction is 0, so the errors are $0 - 4 = -4$, $0 - 6 = -6$, and $0 - 8 = -8$; the gradient is $((-4 \times 1) + (-6 \times 2) + (-8 \times 3)) / 3 = -40 / 3 \approx -13.33$, which is negative.
- **B (Correct):** Correct — applying the update rule, $m = 0 - 0.1 \times (-13.33) = 1.33$, so $m$ rises from 0 to approximately 1.33.
- **C:** Wrong — the update rule explicitly subtracts the gradient (`m - learning_rate * gradient`), not adds it; subtracting a negative gradient is what increases $m$ here, the opposite of what this claims.
- **D:** Wrong — a larger learning rate scales the same negative gradient by a bigger factor, producing a larger step: with `learning_rate = 1.0`, $m = 0 - 1.0 \times (-13.33) = 13.33$, far larger than 1.33, not smaller.

---

### 10. Question

A backend engineer builds a simple linear regression model to predict house price (in lakhs) from square footage, using the equation $y = mx + c$. After fitting, the engineer wants to explain to a colleague exactly what the constant $c$ represents in this equation.

**Which statement correctly describes the intercept $c$ in this model?**

### Options

- [x] **It is the value of price predicted by the model when square feet equals zero.** _(Correct Answer)_
- [ ] It is the amount price changes for each additional one-unit increase in square feet.
- [ ] It is the average vertical gap between observed price and predicted price across all points.
- [ ] It is a measure of how strongly square feet and price move together.

### Answer Explanation

- **A (Correct):** Correct — in $y = mx + c$, the intercept $c$ is defined as the value of $y$ when $x$ is 0 — here, the predicted price when square feet is 0.
- **B:** Wrong — that description belongs to the slope $m$, which captures how much $y$ changes per unit change in $x$, not the constant term.
- **C:** Wrong — the "vertical gap between observed and predicted values" describes the error (residual) at a point, a completely different quantity from the intercept.
- **D:** Wrong — it describes correlation, which measures how strongly two variables move together; correlation is checked before fitting a model and is unrelated to the intercept term produced by fitting one.
