# Linear Regression Objective Test — Set 03

---

### 21. Question

A business analyst is presented with two regression models predicting quarterly revenue:

- **Model A**: $\hat{y} = 5x + 0$ (passes through the origin, zero intercept)
- **Model B**: $\hat{y} = 5x + 50$ (same slope, positive intercept)

**Which statement correctly describes the geometric relationship between these two lines?**

- [ ] **A.** They are perpendicular because they start at different points.
- [ ] **B.** They intersect at the point $x = 10$ and never cross again.
- [ ] **C.** They are parallel lines — both have the same slope ($m = 5$), so they never intersect and are separated by a constant vertical distance of 50 units.
- [ ] **D.** They are the same line because they share the same slope.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Perpendicular lines have slopes that are negative reciprocals of each other (if one has slope $m$, the other has slope $-1/m$). Both lines here have slope $m = 5$, so they are parallel, not perpendicular.
- **Option B (Incorrect):** Two lines can only intersect at one point if they are not parallel and have different slopes. Here, both lines have slope $+5$ — parallel lines never intersect at any point, so they cannot cross at $x = 10$ or anywhere else.
- **Option C (Correct):** Two lines with the same slope are parallel. The vertical distance between them is always $|c_A - c_B| = |0 - 50| = 50$ units, regardless of the value of $x$. Model B's line is shifted 50 units higher than Model A's everywhere, and they never cross.
- **Option D (Incorrect):** Two lines are identical only if they share both the same slope AND the same intercept. Model A and Model B have the same slope ($m = 5$) but different intercepts (0 vs. 50), so they are distinct parallel lines, not the same line.

</details>

---

### 22. Question

An engineer is solving a linear regression problem in reverse. The fitted model is $\hat{y} = 3x - 6$. A client wants a prediction of exactly $\hat{y} = 15$.

**What value of $x$ produces $\hat{y} = 15$?**

- [ ] **A.** $x = 5$
- [ ] **B.** $x = 3$
- [ ] **C.** $x = 7$
- [ ] **D.** $x = 9$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Plugging $x = 5$: $\hat{y} = 3(5) - 6 = 15 - 6 = 9 \ne 15$. This is incorrect.
- **Option B (Incorrect):** Plugging $x = 3$: $\hat{y} = 3(3) - 6 = 9 - 6 = 3 \ne 15$. This is incorrect.
- **Option C (Correct):** Solving $3x - 6 = 15$: add 6 to both sides: $3x = 21$; divide by 3: $x = 7$. Verifying: $3(7) - 6 = 21 - 6 = 15$ ✓. The value $x = 7$ produces the target prediction of 15.
- **Option D (Incorrect):** Plugging $x = 9$: $\hat{y} = 3(9) - 6 = 27 - 6 = 21 \ne 15$. This is incorrect.

</details>

---

### 23. Question

A data scientist is explaining why the Mean Squared Error (MSE) is more sensitive to outliers than the Mean Absolute Error (MAE). She gives a simple example with 4 observations:

- Actuals: $[10, 20, 30, 40]$
- Predictions: $[11, 19, 29, 60]$ (the last prediction is a large outlier error)

**Which statement best describes the disproportionate effect of the final prediction error on MSE vs. MAE?**

- [ ] **A.** Both MAE and MSE are equally affected by large errors, since they both average the same residuals.
- [ ] **B.** The final error of $|60 - 40| = 20$ contributes 20 to MAE but $20^2 = 400$ to the MSE sum before averaging — squaring amplifies large errors far more than small ones.
- [ ] **C.** MAE is more sensitive to the large error because it uses absolute values, which are always larger than squared values.
- [ ] **D.** MSE and MAE produce identical results because both are averages of prediction errors.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** MAE and MSE are not equally sensitive to large errors. MAE sums absolute errors (linear scale), while MSE sums squared errors (quadratic scale). Squaring a large error gives it a disproportionately larger weight. For small errors (magnitude < 1), squaring makes them smaller; for large errors (magnitude > 1), squaring makes them larger.
- **Option B (Correct):** For the 4th observation, the residual magnitude is $|60 - 40| = 20$. In MAE, this contributes 20 to the sum. In MSE, this contributes $20^2 = 400$ to the sum. The other three errors are small (1 each), contributing 1 each to MAE and 1 each to MSE. The large 400 in MSE dwarfs the three small 1s, pushing the overall MSE much higher. This asymmetric amplification is why MSE is far more sensitive to outlier errors than MAE.
- **Option C (Incorrect):** Absolute values are not always larger than squared values. For errors with magnitude less than 1, the absolute value is larger than the square (e.g., $|0.5| = 0.5 > 0.5^2 = 0.25$). For errors with magnitude greater than 1, the square is larger (e.g., $|20| = 20 < 20^2 = 400$). The large outlier error here has magnitude 20, so squaring makes it much larger — making MSE more sensitive to this outlier, not MAE.
- **Option D (Incorrect):** MAE and MSE use the same residuals but aggregate them differently. MAE takes the average of absolute values; MSE takes the average of squared values. These produce different numerical results, especially when large errors are present. They are not identical metrics.

</details>

---

### 24. Question

A senior ML engineer is explaining the geometry of multiple linear regression to a new team member. The team is building a model with exactly 4 input features.

**What is the correct geometric description of what this model produces?**

- [ ] **A.** A line in 4-dimensional space.
- [ ] **B.** A hyperplane in a 5-dimensional space (4 feature dimensions + 1 output dimension).
- [ ] **C.** A curve in 5-dimensional space, because 4 features always produce non-linear boundaries.
- [ ] **D.** A plane in 3-dimensional space, since 4 features are simplified to 3 effective dimensions.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A line is the geometric representation of simple linear regression in 2D space (1 input + 1 output). With 4 input features, the model cannot be represented as a simple line — each additional feature adds a dimension, requiring a higher-dimensional flat surface.
- **Option B (Correct):** With 4 input features ($x_1, x_2, x_3, x_4$) and one output ($y$), the model exists in a 5-dimensional space. The relationship $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4$ defines a hyperplane — a flat surface — in this 5-dimensional space. This is the correct generalization: 1 feature → line in 2D; 2 features → plane in 3D; $k$ features → hyperplane in $(k+1)$D.
- **Option C (Incorrect):** Linear regression always produces a flat geometric surface (line, plane, or hyperplane). The word "linear" in the model name specifically guarantees no curvature. Curves, polynomials, and non-linear boundaries require different models (polynomial regression, neural networks, etc.).
- **Option D (Incorrect):** The number of dimensions in the model's geometric space is not reduced by the number of features. Each feature contributes its own axis, so 4 features occupy 4 input dimensions, plus 1 output dimension, giving 5 total dimensions. There is no compression to 3 dimensions.

</details>

---

### 25. Question

Two data scientists debate whether `random_state` in `train_test_split` matters. Scientist A says it has no effect on model performance. Scientist B says it controls reproducibility.

```python
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Which statement correctly describes the role of `random_state`?**

- [ ] **A.** It controls the model's regularization strength and affects which coefficients are learned.
- [ ] **B.** It seeds the random number generator, ensuring that the split is reproducible; the two code blocks will produce different (but both valid) train/test splits.
- [ ] **C.** Setting `random_state=0` always produces the first 80% of rows as training data; `random_state=42` produces the last 80%.
- [ ] **D.** It has no effect — `train_test_split` always produces the same split regardless of `random_state`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `random_state` is a parameter of the data splitting function, not the model. It has nothing to do with regularization strength, which is a model hyperparameter (like `alpha` in Ridge). The `random_state` controls which rows are randomly assigned to training vs. test sets.
- **Option B (Correct):** `random_state` seeds the pseudorandom number generator used for shuffling and splitting. A fixed seed value produces the same shuffle every time the code is re-run, making the split reproducible. Different seed values (0 vs. 42) produce different splits — different rows end up in training vs. test — but both splits are statistically valid random samples. Reproducibility is the primary purpose.
- **Option C (Incorrect):** `random_state` does not make splits sequential or ordered. The data is randomly shuffled before splitting, regardless of the seed. `random_state=0` does not take the first 80% of rows sequentially; it selects a pseudo-random 80% determined by the seed value.
- **Option D (Incorrect):** Without a fixed `random_state`, `train_test_split` produces a different random split every time it is run. Setting `random_state` to a fixed integer is precisely what makes splits reproducible — it is not true that the split is always the same regardless of this parameter.

</details>

---

### 26. Question

A machine learning student is confused about a training loop detail. The gradient descent update rule should be:

```python
m = m - learning_rate * gradient
```

But the student accidentally writes:

```python
m = m + learning_rate * gradient
```

**What effect does this mistake have on the training process?**

- [ ] **A.** No effect — adding or subtracting the gradient produces the same result because the gradient is always negative.
- [ ] **B.** The model converges slightly more slowly but eventually reaches the correct minimum.
- [ ] **C.** The update moves in the wrong direction, ascending the loss surface instead of descending it. When the gradient is positive (indicating a slope uphill toward minimum), this will move $m$ further away from the optimal value, potentially causing divergence.
- [ ] **D.** The update accidentally computes gradient ascent, which is correct in supervised regression.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The gradient is not always negative. It can be positive or negative depending on the current position of $m$ relative to the minimum. If the gradient is positive (meaning increasing $m$ would increase the loss), the correct action is to subtract it (move left). Adding a positive gradient would move $m$ further right — away from the minimum.
- **Option B (Incorrect):** Using addition instead of subtraction does not cause slower convergence to the correct answer; it causes divergence. The update consistently moves in the opposite direction from where the minimum is located, so it moves the parameters away from the optimum at every step.
- **Option C (Correct):** Gradient descent minimizes the loss by moving in the negative gradient direction: $m_{\text{new}} = m_{\text{old}} - \alpha \times \text{gradient}$. The gradient points in the direction of steepest ascent on the loss surface. Subtracting it moves toward the minimum (descent). Adding it would move toward the maximum — gradient ascent — which is the opposite direction and causes the loss to increase rather than decrease, leading to divergence in practice.
- **Option D (Incorrect):** Gradient ascent is a technique used in other contexts (e.g., maximizing likelihoods in certain optimization frameworks), but it is not correct for minimizing the regression loss function (MSE). Supervised regression requires minimizing the loss, which requires gradient descent (subtraction), not gradient ascent (addition).

</details>

---

### 27. Question

A data scientist examines the Pearson correlation between all pairs of features in a 3-feature dataset for predicting house prices:

| Feature Pair                      | r     |
|-----------------------------------|-------|
| SquareFeet ↔ NumRooms             | 0.93  |
| SquareFeet ↔ DistanceFromCity     | -0.12 |
| NumRooms ↔ DistanceFromCity       | -0.08 |

**Which observation about this correlation table is most important for modeling?**

- [ ] **A.** `DistanceFromCity` should be dropped because its correlations are negative.
- [ ] **B.** `SquareFeet` and `NumRooms` are highly correlated with each other ($r = 0.93$), which may cause both predictors to carry redundant information when included together in a model.
- [ ] **C.** All three features should be dropped because none has a correlation above $r = 0.95$ with any other feature.
- [ ] **D.** The negative correlations mean `DistanceFromCity` has an inverse causal relationship with price.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Negative correlations between two predictors do not indicate a problem with either feature. The sign in predictor-predictor correlations shows direction (as one increases, the other decreases), not quality or usefulness. `DistanceFromCity` correlates negatively with `SquareFeet` and `NumRooms` — this is perfectly valid information, not a reason to drop the feature.
- **Option B (Correct):** When two features correlate very highly with each other (here, $r = 0.93$ between `SquareFeet` and `NumRooms`), they carry largely overlapping information. Including both in a model can cause issues with coefficient reliability — the model may struggle to separate their individual contributions. This is an early signal of what is formally called multicollinearity, and it is worth investigating further or choosing between the two correlated predictors.
- **Option C (Incorrect):** There is no rule that features must exceed any correlation threshold to be useful — the $r = 0.93$ pair already exceeds 0.90. The threshold 0.95 is not a standard benchmark. Also, the table shows inter-feature correlations, not correlations with the target; the latter would need separate evaluation.
- **Option D (Incorrect):** Correlation between predictors does not establish causation with the price (the target). A negative correlation between `DistanceFromCity` and `SquareFeet` is a statistical observation — it might mean that properties further from the city tend to be smaller, but this is a descriptive finding, not a causal claim about what drives prices.

</details>

---

### 28. Question

A junior developer asks: "What does the `model.score(X_test, y_test)` method return for a `LinearRegression` model in scikit-learn? Is it MAE, MSE, or something else?"

**Which answer is correct?**

- [ ] **A.** It returns the Mean Absolute Error (MAE) on the test set.
- [ ] **B.** It returns the Root Mean Squared Error (RMSE) on the test set.
- [ ] **C.** It returns the $R^2$ (coefficient of determination) score, a value between $-\infty$ and $1$ indicating how well the model explains variance in the target.
- [ ] **D.** It returns the proportion of correct predictions, similar to accuracy in classification.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** `model.score()` does not compute MAE. To obtain MAE, you would use `sklearn.metrics.mean_absolute_error(y_test, model.predict(X_test))`. The `.score()` method for regression models returns $R^2$, not any absolute or squared error metric.
- **Option B (Incorrect):** `model.score()` does not return RMSE. RMSE can be computed manually as the square root of `mean_squared_error(y_test, model.predict(X_test))`. The `.score()` method for regressors is specifically $R^2$.
- **Option C (Correct):** For all scikit-learn regression models (including `LinearRegression`), the `.score(X_test, y_test)` method returns the $R^2$ score — the coefficient of determination. It measures the proportion of variance in the target variable explained by the model. An $R^2$ of 1.0 means a perfect fit; 0.0 means the model does no better than predicting the mean; negative values mean the model is worse than a constant mean predictor.
- **Option D (Incorrect):** Accuracy (proportion of correct predictions) is a classification metric, not a regression metric. In regression, predictions are continuous values, so "correct vs. incorrect" is not well-defined in the way it is for classification. `model.score()` for regression returns $R^2$, which measures explained variance.

</details>

---

### 29. Question

A machine learning student encounters this code:

```python
a = [1, 2, 3, 4]
b = [10, 20]

result = list(zip(a, b))
print(result)
```

**What does `result` contain, and why?**

- [ ] **A.** `[(1, 10), (2, 20), (3, None), (4, None)]` — zip fills missing values with `None`.
- [ ] **B.** `[(1, 10), (2, 20)]` — zip stops at the length of the shortest iterable.
- [ ] **C.** An error is raised because `zip()` requires both lists to have equal length.
- [ ] **D.** `[(1, 10), (2, 20), (3, 10), (4, 20)]` — zip cycles through the shorter list to fill remaining pairs.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Python's built-in `zip()` does not fill missing values with `None`. That behavior belongs to `itertools.zip_longest()`, a different function that explicitly allows unequal-length iterables and pads shorter ones with a fill value (default `None`).
- **Option B (Correct):** Python's built-in `zip()` stops producing pairs as soon as the shortest iterable is exhausted. Here, `b` has only 2 elements, so `zip(a, b)` produces only 2 pairs: `(1, 10)` and `(2, 20)`. The remaining elements `3` and `4` from `a` are silently dropped because there are no corresponding elements in `b`.
- **Option C (Incorrect):** `zip()` does not raise an error for lists of different lengths. It silently truncates the output to the length of the shortest input. This can be a source of bugs in from-scratch ML loops if the engineer assumes all zipped lists have equal length.
- **Option D (Incorrect):** `zip()` does not cycle or repeat the shorter list. Cycling behavior requires `itertools.cycle()` combined with `zip()`. The plain `zip()` stops at the shortest length without repeating any elements.

</details>

---

### 30. Question

A senior data analyst is explaining the difference between a zero-correlation predictor and a zero-slope coefficient. She says:

"Even if `df.corr()` shows correlation 0 between a feature and the target, the fitted model coefficient for that feature will not necessarily be zero in multiple regression."

**Is this statement accurate? Explain why or why not.**

- [ ] **A.** Incorrect; a correlation of exactly 0 with the target always guarantees the fitted coefficient will be exactly 0.
- [ ] **B.** Correct; the pairwise correlation between a feature and the target ignores the relationships between features, while the fitted coefficient in multiple regression accounts for the presence of all other features simultaneously — a feature with zero pairwise correlation could have a non-zero partial effect.
- [ ] **C.** Incorrect; scikit-learn automatically removes features with zero correlation from the model before fitting.
- [ ] **D.** Correct; but only when `random_state` in `train_test_split` is set to a non-zero value.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Pairwise (marginal) correlation and the partial regression coefficient in a multiple regression model are related but not identical. A feature may have near-zero pairwise correlation with the target yet still contribute predictive value when correlated features are controlled for. Conversely, a feature correlated with the target may receive a near-zero coefficient if another feature already captures that relationship. Zero pairwise correlation does not guarantee a zero coefficient.
- **Option B (Correct):** The Pearson correlation $r$ between a single feature and the target measures their marginal (unconditional) linear relationship, ignoring all other features. In contrast, a multiple regression coefficient estimates the partial relationship between that feature and the target after accounting for all other predictors in the model. These two quantities can differ substantially, especially when predictors are correlated with each other. A feature with $r \approx 0$ with the target can still receive a non-zero coefficient if its relationship with the target becomes apparent only after controlling for other variables (a phenomenon related to Simpson's paradox or suppressor variables).
- **Option C (Incorrect):** scikit-learn's `LinearRegression` does not automatically filter features based on their correlation with the target. The model accepts all provided feature columns and computes coefficients for each of them during fitting. Feature selection based on correlation is a manual preprocessing step, not automatic behavior.
- **Option D (Incorrect):** The `random_state` parameter of `train_test_split` determines how data is shuffled for splitting — it has nothing to do with the relationship between correlations and model coefficients. This is not a relevant condition for the accuracy of the analyst's statement.

</details>
