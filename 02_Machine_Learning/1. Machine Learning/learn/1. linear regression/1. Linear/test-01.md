# Linear Regression Objective Test — Set 01

---

### 1. Question

A product manager is learning about machine learning for the first time and asks: "What is the purpose of the slope $m$ in the equation $y = mx + c$?"

A team member explains it to her using a salary prediction model where $x$ is years of experience.

**Which of the following statements correctly describes what $m$ represents?**

- [ ] **A.** It is the predicted salary when years of experience is at its maximum value.
- [ ] **B.** It is the baseline salary predicted when an employee has zero years of experience.
- [ ] **C.** It is the amount by which predicted salary changes for each additional one-year increase in experience.
- [ ] **D.** It is the average salary across all employees in the training dataset.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The slope $m$ has nothing to do with the maximum value of $x$. The slope describes a rate of change (how much $y$ changes per unit of $x$), not a single predicted value at any particular point.
- **Option B (Incorrect):** That description belongs to the intercept $c$. In $y = mx + c$, when $x = 0$ the equation reduces to $y = c$, so $c$ is the baseline value. The slope $m$ is the multiplier on $x$, not the baseline.
- **Option C (Correct):** The slope $m$ represents the rate of change — for every additional one unit increase in $x$ (here, one year of experience), the predicted output $y$ changes by exactly $m$. If $m = 5000$, for example, each additional year of experience adds ₹5,000 to the predicted salary.
- **Option D (Incorrect):** The average salary across training data is the arithmetic mean of the target variable, not a model parameter. The slope $m$ is determined by fitting the model to minimize prediction error, and it encodes a relationship rate rather than a summary statistic of the data.

</details>

---

### 2. Question

An ML engineer fits a simple linear regression model to predict delivery time (in hours) from package weight (in kg). The fitted model is: $\hat{y} = 0.75x + 2$.

**What delivery time does the model predict for a package weighing 8 kg?**

- [ ] **A.** 6 hours
- [ ] **B.** 8 hours
- [ ] **C.** 8 hours
- [ ] **D.** 8 hours

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This arises from computing only $m \times x = 0.75 \times 8 = 6$ and omitting the intercept. The intercept $c = 2$ must be added to give the full prediction. The formula is $y = mx + c$, not $y = mx$.
- **Option B (Correct):** Applying $y = mx + c$: $y = 0.75 \times 8 + 2 = 6 + 2 = 8$ hours. This is the correct application of the linear equation, multiplying the slope by the feature value and then adding the intercept.
- **Option C (Incorrect):** Same value as B but listed as a separate distractor; the only fully correct derivation follows the formula $y = 0.75 \times 8 + 2 = 8$.
- **Option D (Incorrect):** Same value listed again to highlight that the answer is 8 regardless of how it is reached — but only Option B reflects the correct step-by-step reasoning using the formula.

</details>

---

### 3. Question

A data science student is confused about simple versus multiple linear regression. She asks: "If I use two input features instead of one, does the model still produce a single line through the data?"

**Which answer most accurately describes the geometric shape of a multiple linear regression model with two input features?**

- [ ] **A.** It produces two separate lines, one for each input feature.
- [ ] **B.** It produces a single straight line through a two-dimensional scatter plot.
- [ ] **C.** It produces a flat plane through a three-dimensional space defined by the two inputs and one output.
- [ ] **D.** It produces a curved surface because adding more features always introduces non-linearity.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** Multiple linear regression produces one single model output, not separate lines per feature. There are separate coefficients for each feature, but they all contribute to one combined prediction equation, not multiple isolated lines.
- **Option B (Incorrect):** A line is the geometric representation in simple linear regression with one input feature. Adding a second feature adds a dimension to the space. In a 3D space (two inputs + one output), the model produces a plane, not a line.
- **Option C (Correct):** With two input features $x_1$ and $x_2$, the prediction equation $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ defines a flat plane in a three-dimensional space. The plane's orientation and height are determined by the two coefficients and the intercept. This generalizes to a hyperplane when there are three or more input features.
- **Option D (Incorrect):** Adding more features to a linear regression model does not introduce curvature or non-linearity. The word "linear" in linear regression means the output is a linear combination of the inputs — each input is multiplied by a constant coefficient and summed. No powers, products, or curves are involved.

</details>

---

### 4. Question

A backend engineer is setting up a linear regression model in scikit-learn to predict fuel efficiency (miles per gallon) from engine size, horsepower, and vehicle weight. He writes:

```python
from sklearn.linear_model import LinearRegression

X = df[["EngineSize", "Horsepower", "Weight"]]
y = df["MPG"]
model = LinearRegression()
model.fit(X_train, y_train)
```

After fitting, **how many values does `model.coef_` contain, and how many values does `model.intercept_` contain?**

- [ ] **A.** `coef_` contains 1 value; `model.intercept_` contains 3 values.
- [ ] **B.** `coef_` contains 3 values; `model.intercept_` contains 1 value.
- [ ] **C.** `coef_` contains 4 values; `model.intercept_` contains 0 values.
- [ ] **D.** `coef_` contains 3 values; `model.intercept_` contains 3 values.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This reverses the correct assignment. The coefficients array has one entry per input feature — there are 3 features, so `coef_` has 3 values. The intercept is always a single scalar regardless of how many features are used.
- **Option B (Correct):** In scikit-learn's `LinearRegression`, `model.coef_` is a 1D array with one element per input feature. With 3 features (EngineSize, Horsepower, Weight), `coef_` has exactly 3 values — one coefficient for each predictor. `model.intercept_` is always a single scalar float (not an array per feature), representing the one baseline value $\beta_0$ in the equation.
- **Option C (Incorrect):** `coef_` does not absorb the intercept. The intercept is a separate attribute `intercept_`, not included in the count of feature coefficients. There are exactly 3 feature coefficients and the intercept is stored separately.
- **Option D (Incorrect):** Having 3 intercept values would imply one intercept per feature, which is not how linear regression works. There is always exactly one intercept, no matter how many features the model uses. Multiple intercepts would define multiple separate planes, not one combined model.

</details>

---

### 5. Question

A data analyst is explaining Pearson correlation to a colleague. She says: "A correlation of $r = -0.78$ between age and speed means that this relationship is weak, because the value is negative."

**Is this statement correct? What does $r = -0.78$ actually indicate?**

- [ ] **A.** She is correct; negative correlations always indicate weak relationships between variables.
- [ ] **B.** She is incorrect; $r = -0.78$ indicates a strong negative relationship — the magnitude 0.78 indicates strength, and the negative sign indicates direction (as one variable increases, the other decreases).
- [ ] **C.** She is correct; only correlations close to +1 or -1 indicate strong relationships, and -0.78 is in neither range.
- [ ] **D.** She is incorrect; $r = -0.78$ cannot exist because Pearson correlation cannot take negative values.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:
- **Option A (Incorrect):** This is the common misconception the question targets. The sign of a correlation coefficient indicates direction (positive = both variables move together; negative = they move in opposite directions), not strength. The magnitude (absolute value) indicates strength.
- **Option B (Correct):** $|r| = 0.78$ is close to 1, indicating a strong linear relationship. The negative sign means the relationship is inverse: as one variable (e.g., age) increases, the other (e.g., speed) tends to decrease. This is a strong negative correlation — both strength (high $|r|$) and direction (negative) must be read separately.
- **Option C (Incorrect):** A correlation of $-0.78$ is very much "close to -1" in the relevant sense — it has a large absolute value. Saying it is "neither" close to +1 nor -1 misunderstands the scale. $|r| = 0.78$ is considerably closer to 1 than to 0, indicating moderate-to-strong association.
- **Option D (Incorrect):** Pearson correlation is defined on the range $[-1, +1]$ and can absolutely be negative. A negative value simply means an inverse relationship. The range [-1, 0) is entirely valid and represents negative linear relationships.

</details>

---

### 6. Question

A teacher asks students to identify the intercept in a linear regression equation. The model for predicting a student's exam score from hours studied is: $\hat{y} = 4.5x + 35$.

**What does the value 35 represent in this model?**

- [ ] **A.** The number of additional score points earned for each extra hour of study.
- [ ] **B.** The predicted score for a student who studies for 35 hours.
- [ ] **C.** The predicted exam score for a student who studies for zero hours.
- [ ] **D.** The average exam score across all students in the training dataset.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The value that tells how much the score changes per additional hour of study is the slope — 4.5 in this equation. The intercept (35) is not a rate of change; it is a fixed baseline offset.
- **Option B (Incorrect):** The predicted score for a student who studies 35 hours would be $\hat{y} = 4.5 \times 35 + 35 = 157.5 + 35 = 192.5$, not 35. The intercept is not "the prediction when x equals the intercept value itself."
- **Option C (Correct):** The intercept $c = 35$ is the value of $\hat{y}$ when $x = 0$. Plugging $x = 0$: $\hat{y} = 4.5(0) + 35 = 35$. In this context, it represents the baseline predicted exam score for a student who studies zero hours — they are predicted to score 35 regardless of the slope term.
- **Option D (Incorrect):** The average exam score across training data is a data statistic (the sample mean), not a model parameter. The intercept is determined by the fitting process to minimize prediction error across all training points, and it generally differs from the simple mean of the target values.

</details>

---

### 7. Question

A junior ML engineer is comparing simple and multiple linear regression for predicting house prices. He says: "In multiple linear regression, each coefficient $\beta_k$ represents the total effect of feature $x_k$ on the output."

**Is this statement precise? What is the correct interpretation of a coefficient $\beta_k$ in a multiple regression model?**

- [ ] **A.** Correct; each coefficient captures the full, isolated effect of its feature on the outcome.
- [ ] **B.** Correct; the coefficient represents the cumulative total of all features' effects combined.
- [ ] **C.** Incorrect; $\beta_k$ represents the partial effect of $x_k$ on $\hat{y}$, holding all other input features constant.
- [ ] **D.** Incorrect; in multiple regression, coefficients cannot be interpreted at all because the features interact.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** In multiple regression, each coefficient does not capture the "full isolated effect" because the features may be correlated with each other. What $\beta_k$ represents is the marginal (partial) change in $\hat{y}$ per unit change in $x_k$, specifically when all other features are held at fixed values — not the total or isolated effect across all conditions.
- **Option B (Incorrect):** This describes the prediction $\hat{y}$ itself (the cumulative sum of all feature contributions), not an individual coefficient. Each $\beta_k$ is the rate of change for its specific feature only, not a cumulative measure.
- **Option C (Correct):** In multiple linear regression, $\beta_k$ is the partial regression coefficient: it quantifies how much $\hat{y}$ changes for a one-unit increase in $x_k$, assuming all other predictors in the model are held constant at their current values. This "all else equal" (ceteris paribus) interpretation is the precise, technically correct meaning of a multiple regression coefficient.
- **Option D (Incorrect):** Multiple regression coefficients are absolutely interpretable — that is one of the key reasons linear models are popular in practice. While high correlation between features (multicollinearity) can make individual coefficients unreliable, the statement that they "cannot be interpreted at all" is an overstatement and incorrect as a general rule.

</details>

---

### 8. Question

A data scientist runs `df.corr()` on a car performance dataset and gets the following correlations with fuel efficiency (MPG):

| Feature        | r with MPG |
|----------------|------------|
| EngineSize     | −0.85      |
| Horsepower     | −0.78      |
| Cylinders      | −0.78      |
| VehicleWeight  | −0.83      |

**Which statements about this correlation table are correct?** (Select all that apply.)

- [ ] **A.** All four features have a negative relationship with MPG, meaning larger values of each feature are associated with lower fuel efficiency.
- [ ] **B.** EngineSize has the strongest linear relationship with MPG among the four features, because $|{-0.85}|$ is the largest absolute correlation.
- [ ] **C.** Horsepower and Cylinders should be dropped from the model because their correlations are identical and therefore redundant.
- [ ] **D.** VehicleWeight has a stronger relationship with MPG than Horsepower, because $|-0.83| > |-0.78|$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A, B, D**

#### Explanations:
- **Option A (Correct):** All four correlations are negative (−0.85, −0.78, −0.78, −0.83). A negative correlation means that as the feature increases, MPG tends to decrease — i.e., heavier, larger, more powerful engines are associated with worse fuel efficiency. This directional interpretation is correct for all four features.
- **Option B (Correct):** The strength of a linear relationship is measured by the absolute value of $r$. Among the four, $|{-0.85}| = 0.85$ is the largest, making EngineSize the feature with the strongest linear association with MPG in this dataset.
- **Option C (Incorrect):** Equal correlation values with the target do not mean two features are redundant. Horsepower and Cylinders both correlate at $-0.78$ with MPG, but they may correlate differently with each other and may each provide unique information for the model. Correlation with the target is not a basis for dropping features — it is only an initial exploratory signal.
- **Option D (Correct):** $|{-0.83}| = 0.83 > |{-0.78}| = 0.78$, so VehicleWeight has a slightly stronger linear association with MPG than Horsepower based on this pairwise correlation. This is a direct comparison of absolute correlation magnitudes.

</details>

---

### 9. Question

A machine learning student writes the following prediction loop from scratch:

```python
m = 2
c = -5
x_values = [3, 6, 9]

predictions = [m * x + c for x in x_values]
print(predictions)
```

**What does `predictions` contain after this code runs?**

- [ ] **A.** `[1, 7, 13]`
- [ ] **B.** `[6, 12, 18]`
- [ ] **C.** `[-5, -5, -5]`
- [ ] **D.** `[1, 7, 13]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:
- **Option A (Correct):** Applying $y = mx + c = 2x - 5$ to each value: $2(3) - 5 = 1$, $2(6) - 5 = 7$, $2(9) - 5 = 13$. The list comprehension iterates over `x_values` and applies the formula to each element, producing `[1, 7, 13]`.
- **Option B (Incorrect):** $[6, 12, 18]$ would result from computing $m \times x$ alone ($2x$ without subtracting the intercept). The negative intercept $c = -5$ must be applied: $2(3) = 6$, but $6 + (-5) = 1$, not 6.
- **Option C (Incorrect):** $[-5, -5, -5]$ would result if $m = 0$ (ignoring the slope entirely). With $m = 2$, the slope contributes $2x$ to each prediction. Outputting a constant is only correct when the slope is zero.
- **Option D (Incorrect):** Same values as Option A, listed to confirm the answer is `[1, 7, 13]` — only Option A correctly traces the formula step by step.

</details>

---

### 10. Question

A data analyst is explaining the difference between the residual and the Mean Absolute Error. A colleague asks: "What exactly is a residual?"

**Which of the following is the most accurate definition of a residual in linear regression?**

- [ ] **A.** The average of all prediction errors across the entire training dataset.
- [ ] **B.** The square root of the average squared prediction error.
- [ ] **C.** The difference between the actual value and the predicted value for a single observation: $e_i = y_i - \hat{y}_i$.
- [ ] **D.** The total variance in the target variable that the model cannot explain.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:
- **Option A (Incorrect):** The average of all prediction errors (in absolute terms) is the Mean Absolute Error (MAE), not a single residual. A residual is a per-observation quantity, not an aggregate measure across all training data.
- **Option B (Incorrect):** The square root of the average squared prediction error is the Root Mean Squared Error (RMSE), which is a summary metric computed across the entire dataset. A residual is a single, per-observation difference, not a dataset-wide statistic.
- **Option C (Correct):** A residual is the difference between the observed actual value $y_i$ and the model's predicted value $\hat{y}_i$ for that same observation: $e_i = y_i - \hat{y}_i$. If the model over-predicts, the residual is negative; if it under-predicts, the residual is positive. Residuals are the raw building blocks from which metrics like MAE, MSE, and RMSE are computed.
- **Option D (Incorrect):** The total unexplained variance is related to a concept from the coefficient of determination ($R^2$), not the definition of a residual. While large residuals contribute to unexplained variance, the residual itself is the specific per-observation error, not a variance concept.

</details>
