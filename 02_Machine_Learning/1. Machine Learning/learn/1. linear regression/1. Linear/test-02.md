# Linear Regression Objective Test — Set 02

---

### 11. Question

A product analyst is exploring a retail dataset to determine which features might help predict weekly sales. She runs the following:

```python
df[["StoreSize", "NumEmployees", "AdvertisingBudget", "WeeklySales"]].corr()
```

The result shows:
- `StoreSize` vs `WeeklySales`: $r = 0.76$
- `NumEmployees` vs `WeeklySales`: $r = 0.43$
- `AdvertisingBudget` vs `WeeklySales`: $r = 0.91$

**Which of the following conclusions is valid?**

- [ ] **A.** `AdvertisingBudget` is the best predictor of sales; all other features should be dropped before fitting any model.
- [ ] **B.** Because `NumEmployees` has the lowest correlation, it has no relationship with sales and is useless.
- [ ] **C.** `AdvertisingBudget` shows the strongest linear association with `WeeklySales` among the three, based on having the largest absolute correlation.
- [ ] **D.** A correlation of 0.91 proves that increasing the advertising budget causes sales to increase.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Correlation with the target is only an early exploratory signal — it does not provide grounds for dropping every other feature before model fitting. Features with lower individual correlations may still contribute unique predictive information when combined with others in a multiple regression model.
- **Option B (Incorrect):** A lower correlation does not mean "no relationship" or "useless." $r = 0.43$ between NumEmployees and WeeklySales indicates a moderate positive relationship. Lower correlation only means the linear association is weaker than the other two — it does not mean the feature should be discarded without further evaluation.
- **Option C (Correct):** Among the three features, `AdvertisingBudget` has the largest absolute Pearson correlation ($0.91$) with `WeeklySales`. This makes it the feature with the strongest observed linear relationship with the target in this dataset, based purely on pairwise correlation analysis.
- **Option D (Incorrect):** Correlation never proves causation. A high $r = 0.91$ shows a strong linear co-movement between the two variables but provides no evidence that one causes the other. Larger stores might both spend more on advertising and sell more for unrelated structural reasons, for example.

</details>

---

### 12. Question

A machine learning engineer writes the following code to build and inspect a linear regression model:

```python
from sklearn.linear_model import LinearRegression

X = df[["Rooms", "Bathrooms", "ParkingSpots", "FloorLevel", "DistanceToStation"]]
y = df["RentPrice"]

model = LinearRegression()
model.fit(X_train, y_train)

print(len(model.coef_))
```

**What does this code print?**

- [ ] **A.** 1
- [ ] **B.** 5
- [ ] **C.** 6
- [ ] **D.** It raises a `TypeError` because `len()` cannot be applied to `model.coef_`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A value of 1 would imply only one coefficient, which would mean simple (not multiple) regression with a single feature. There are 5 features in `X`, so there will be 5 coefficients.
- **Option B (Correct):** `model.coef_` is a 1D numpy array with exactly one element per input feature. Since `X` has 5 columns (Rooms, Bathrooms, ParkingSpots, FloorLevel, DistanceToStation), `model.coef_` has 5 elements, and `len(model.coef_)` prints `5`.
- **Option C (Incorrect):** 6 would be the count if the intercept were included in `coef_`, but scikit-learn stores the intercept separately in `model.intercept_`. The `coef_` array contains only the feature weights — one per predictor column, which is 5 here.
- **Option D (Incorrect):** `model.coef_` is a standard numpy array after fitting. `len()` is a valid Python built-in that works on numpy arrays (it returns the size of the first dimension). This call does not raise an error.

</details>

---

### 13. Question

A data scientist fits a simple linear regression model and obtains $m = -3$ and $c = 120$. The input feature $x$ represents the number of days since a product launch, and the target $y$ is daily website traffic (in thousands of visits).

**Which of the following interpretations of $m = -3$ is correct?**

- [ ] **A.** The website starts with 3,000 visitors on day zero.
- [ ] **B.** Website traffic increases by 3,000 visits per day on average.
- [ ] **C.** For each additional day since launch, predicted daily traffic decreases by 3,000 visits.
- [ ] **D.** The slope of -3 indicates no meaningful relationship between time since launch and traffic.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The starting value (traffic on day zero) is given by the intercept $c = 120$, meaning 120,000 visits on launch day. The slope $m = -3$ has nothing to do with the starting value.
- **Option B (Incorrect):** A negative slope ($m = -3$) indicates a decreasing trend, not an increasing one. An increase of 3,000 visits per day would require a positive slope of $+3$.
- **Option C (Correct):** The slope $m = -3$ means that for each one-unit increase in $x$ (one additional day since launch), the predicted output $y$ decreases by 3 (in thousands, so 3,000 visits). A negative slope represents a declining relationship between the feature and the target.
- **Option D (Incorrect):** A non-zero slope (whether positive or negative) always indicates a relationship between the input and output. Only a slope of exactly zero would mean no relationship. $m = -3$ describes a meaningful declining trend.

</details>

---

### 14. Question

A developer uses the following Python code in a from-scratch regression training loop:

```python
x_data = [1, 3, 5, 7]
y_data = [2, 6, 10, 14]

pairs = list(zip(x_data, y_data))
print(pairs)
```

**What does `pairs` contain after this code runs?**

- [ ] **A.** `[(2, 1), (6, 3), (10, 5), (14, 7)]` — each pair is (y, x) because zip reads y first.
- [ ] **B.** `[(1, 2), (3, 6), (5, 10), (7, 14)]` — each tuple pairs the x value with the corresponding y value at the same index.
- [ ] **C.** `[1, 3, 5, 7, 2, 6, 10, 14]` — zip flattens both lists into one sequence.
- [ ] **D.** `[([1, 3, 5, 7], [2, 6, 10, 14])]` — zip wraps both entire lists into one tuple.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** `zip(x_data, y_data)` pairs elements in the order the arguments are provided: x first, y second within each tuple. The result is (x, y) pairs, not (y, x). The first argument to `zip()` determines which value appears first in each tuple.
- **Option B (Correct):** `zip(x_data, y_data)` creates pairs by matching each element from `x_data` with the element at the same position in `y_data`. It produces: `(1, 2)`, `(3, 6)`, `(5, 10)`, `(7, 14)` — one tuple per index, in the order (x, y).
- **Option C (Incorrect):** `zip()` does not flatten lists into a single sequence. Concatenating lists end-to-end would require `x_data + y_data`. The `zip()` function creates tuples of corresponding elements, preserving the pairing structure.
- **Option D (Incorrect):** `zip()` never packages an entire list as one element of a tuple. It operates element-by-element, producing one short tuple per pair of corresponding elements, not one giant tuple containing both full lists.

</details>

---

### 15. Question

A data scientist uses `train_test_split` to prepare data for a linear regression model:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=99
)
```

The dataset originally has 400 rows. **How many rows are in `X_train` and how many are in `X_test`?**

- [ ] **A.** `X_train` has 300 rows; `X_test` has 100 rows.
- [ ] **B.** `X_train` has 100 rows; `X_test` has 300 rows.
- [ ] **C.** `X_train` has 200 rows; `X_test` has 200 rows.
- [ ] **D.** `X_train` has 75 rows; `X_test` has 25 rows.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** `test_size=0.25` reserves 25% of the 400 rows for the test set: $400 \times 0.25 = 100$ rows. The remaining 75% go to the training set: $400 \times 0.75 = 300$ rows. `X_train` has 300 rows and `X_test` has 100 rows.
- **Option B (Incorrect):** This reverses the split. `test_size=0.25` means 25% goes to testing (100 rows), not 75%. The majority of the data (300 rows) goes to training, not testing.
- **Option C (Incorrect):** A 50/50 split would require `test_size=0.5`. The parameter 0.25 specifies 25% for testing, which with 400 total rows gives an unequal split of 300 train and 100 test.
- **Option D (Incorrect):** 75 and 25 rows would be correct only if the total dataset had 100 rows ($100 \times 0.75 = 75$, $100 \times 0.25 = 25$). With 400 rows, the counts are 300 and 100 respectively.

</details>

---

### 16. Question

A researcher is evaluating two simple linear regression models on the same dataset. Model A has the equation $\hat{y} = 2.5x + 10$ and Model B has $\hat{y} = 2.5x - 10$.

For a new observation with $x = 8$, **how do the predictions of Model A and Model B compare?**

- [ ] **A.** Model A predicts 30; Model B predicts 10. Model A predicts 20 units higher.
- [ ] **B.** Both models predict the same value because they share the same slope.
- [ ] **C.** Model A predicts 30; Model B predicts 10. Model A predicts exactly 20 more than Model B.
- [ ] **D.** Model A predicts 40; Model B predicts 20. Model A predicts exactly 20 more than Model B.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The values 30 and 10 are correctly computed here. Model A: $2.5(8) + 10 = 20 + 10 = 30$. Model B: $2.5(8) - 10 = 20 - 10 = 10$. The difference is indeed 20. This option is numerically correct but duplicated by the more precisely worded Option C.
- **Option B (Incorrect):** Same slope does not mean same prediction. The intercept shifts the entire regression line up or down. Model A has intercept $+10$ and Model B has $-10$, making their predictions differ by exactly 20 units for any value of $x$.
- **Option C (Correct):** Model A: $2.5 \times 8 + 10 = 20 + 10 = 30$. Model B: $2.5 \times 8 - 10 = 20 - 10 = 10$. Difference: $30 - 10 = 20$. The two models always differ by exactly 20 (the difference in intercepts: $10 - (-10) = 20$) for any input $x$, because they are parallel lines separated by a constant vertical gap of 20.
- **Option D (Incorrect):** 40 and 20 would result from a slope of approximately 3.75 (since $3.75 \times 8 = 30$, then $30 + 10 = 40$), not $2.5$. With slope 2.5: $2.5 \times 8 = 20$, then $+10 = 30$ and $-10 = 10$.

</details>

---

### 17. Question

A student asks: "In linear regression, if I have $k$ input features, how many total parameters does the model learn?"

**Which answer is correct?**

- [ ] **A.** $k$ parameters total: one coefficient for each feature, and no intercept unless explicitly specified.
- [ ] **B.** $k + 1$ parameters total: one coefficient for each of the $k$ features, plus exactly one intercept.
- [ ] **C.** $2k$ parameters total: one slope and one intercept for each feature.
- [ ] **D.** $k^2$ parameters: the model learns pairwise interaction terms between all features.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** Standard linear regression (as implemented in scikit-learn with `fit_intercept=True`) always learns an intercept term by default. So the total is $k$ feature coefficients + 1 intercept = $k + 1$ parameters, not $k$.
- **Option B (Correct):** Multiple linear regression with $k$ input features learns exactly $k$ coefficients ($\beta_1, \beta_2, \ldots, \beta_k$) — one per feature — and exactly one intercept ($\beta_0$), for a total of $k + 1$ learnable parameters. This is consistent with the equation $\hat{y} = \beta_0 + \beta_1 x_1 + \ldots + \beta_k x_k$.
- **Option C (Incorrect):** There is no concept of "one slope and one intercept per feature" in standard linear regression. All features share a single common intercept, and each feature has its own coefficient (slope in its own dimension), not a separate intercept.
- **Option D (Incorrect):** Standard linear regression does not include pairwise interaction terms — those would require explicit feature engineering (creating $x_i \times x_j$ columns) or a polynomial model. The base linear model always has $k + 1$ parameters, not $k^2$.

</details>

---

### 18. Question

A regression model is trained to predict monthly electricity bills (in dollars) from average daily temperature. The trained model is $\hat{y} = -1.2x + 200$, where $x$ is the temperature in Celsius.

**What does the negative slope $-1.2$ tell us about this model?**

- [ ] **A.** The model is invalid because electricity bills cannot have a negative relationship with temperature.
- [ ] **B.** For each 1°C increase in daily temperature, predicted electricity bill decreases by $1.20.
- [ ] **C.** The model predicts that electricity bills are always negative.
- [ ] **D.** When temperature is zero, the predicted bill is negative.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A negative slope is mathematically and practically valid. In many climates, warmer temperatures reduce heating costs, which could lower electricity bills. Whether this makes domain sense depends on the specific context. A negative slope is not a sign of model invalidity.
- **Option B (Correct):** The slope $m = -1.2$ means that for each one-unit increase in temperature (1°C), the predicted electricity bill decreases by $\$1.20$. This is the rate-of-change interpretation: a negative slope indicates an inverse linear relationship between temperature and predicted bill.
- **Option C (Incorrect):** The sign of the slope does not mean all predictions are negative. For example, at $x = 10°C$: $\hat{y} = -1.2(10) + 200 = -12 + 200 = \$188$ — a positive predicted bill. Predictions are negative only when $x$ is very large (above $200 / 1.2 \approx 167°C$), which is unrealistic for daily temperatures.
- **Option D (Incorrect):** When $x = 0°C$: $\hat{y} = -1.2(0) + 200 = \$200$. The intercept is $+200$, so the prediction at zero temperature is positive, not negative.

</details>

---

### 19. Question

A senior engineer explains why residuals matter. She gives three model predictions and actual values for a 3-house dataset:

| House | Actual $y$ | Predicted $\hat{y}$ |
|-------|-----------|---------------------|
| 1     | 80        | 75                  |
| 2     | 60        | 65                  |
| 3     | 100       | 95                  |

**Using the convention $e_i = y_i - \hat{y}_i$, what are the three residuals?**

- [ ] **A.** $[+5, -5, +5]$
- [ ] **B.** $[-5, +5, -5]$
- [ ] **C.** $[+5, +5, +5]$
- [ ] **D.** $[5, 5, 5]$ (absolute values, always positive)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Applying $e_i = y_i - \hat{y}_i$: House 1: $80 - 75 = +5$. House 2: $60 - 65 = -5$. House 3: $100 - 95 = +5$. The residuals are $[+5, -5, +5]$. A positive residual means the model under-predicted; a negative residual means it over-predicted.
- **Option B (Incorrect):** These are the negative of the correct residuals, which would result from using $\hat{y}_i - y_i$ instead of $y_i - \hat{y}_i$. The question specifies the convention $e_i = y_i - \hat{y}_i$, so signs must be actual minus predicted.
- **Option C (Incorrect):** All three residuals being $+5$ would require all three predictions to be exactly 5 units below the actuals. House 2 was over-predicted ($\hat{y} = 65 > y = 60$), giving a negative residual of $-5$, not $+5$.
- **Option D (Incorrect):** Taking absolute values would give $[5, 5, 5]$ only as an intermediate step in computing MAE — it is not the definition of a residual. Residuals preserve sign, which is important for diagnosing the direction of prediction errors.

</details>

---

### 20. Question

A product data scientist is preparing features for a multiple regression model to predict customer lifetime value. She is deciding whether to include both `TotalOrders` and `AverageOrderValue` as separate features.

She notes that `df.corr()` shows `TotalOrders` correlates with `LifetimeValue` at $r = 0.72$ and `AverageOrderValue` at $r = 0.68$.

**Which statement most accurately describes how to use this information?**

- [ ] **A.** Since $r = 0.72 > r = 0.68$, only `TotalOrders` should be included and `AverageOrderValue` dropped.
- [ ] **B.** Both features can be included as candidates in the model; the correlation values are early evidence of their individual linear relationships with the target, but the model may benefit from including both.
- [ ] **C.** Since both correlations are below 0.80, neither feature is strong enough to be useful in the model.
- [ ] **D.** The correlation between `TotalOrders` and `AverageOrderValue` with the target confirms that orders cause higher lifetime value.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** A marginally higher pairwise correlation with the target is not a sufficient basis for dropping a feature. `AverageOrderValue` may capture different aspects of customer behavior than `TotalOrders`, and including both could improve prediction. Preliminary correlation analysis is exploratory, not a strict elimination rule.
- **Option B (Correct):** The correlations ($0.72$ and $0.68$) are moderately strong signals that both features are individually linearly related to the target. Both are useful candidates for inclusion in the model. Further model evaluation (comparing performance with and without each feature) would be the appropriate next step, not a unilateral exclusion based on pairwise correlation alone.
- **Option C (Incorrect):** There is no universal rule that only features with $r > 0.80$ are "strong enough." $r = 0.68$ and $r = 0.72$ represent moderate-to-strong linear associations that are valuable in many real-world prediction scenarios. Threshold selection is context-dependent and exploratory, not absolute.
- **Option D (Incorrect):** Correlation never establishes causation. High $r$ with the target shows linear co-movement, not a causal mechanism. Customers with high lifetime value may order often for reasons that reflect other underlying factors (loyalty, pricing, promotions) that drive both outcomes simultaneously.

</details>
