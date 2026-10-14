# Multicollinearity Objective Test — Set 01

---

### 1. Question

A data science intern is working with a tabular dataset of residential properties and notices that `square_feet`, `square_meters`, and `floor_area_sq_yards` are all present as independent variables.

Which of the following describes the type of multicollinearity present in this dataset?

- [ ] A) Stochastic multicollinearity, because the features are only loosely related through random noise.
- [ ] B) Perfect multicollinearity, because the features are exact deterministic linear transformations of each other.
- [ ] C) Structural multicollinearity, because the features are polynomial powers of a single independent variable.
- [ ] D) Orthogonal collinearity, because all three features have zero covariance with the target variable.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Stochastic (imperfect) multicollinearity occurs when variables are strongly correlated but not mathematically identical (e.g., house size and number of rooms). Here, square meters and square feet have an exact mathematical conversion factor, making the relationship deterministic, not stochastic.
- **Option B (Correct):** Perfect multicollinearity arises when one feature can be expressed as an exact linear function of one or more other features (e.g., $1 \text{ sq meter} \approx 10.7639 \text{ sq feet}$). In matrix terms, this causes exact linear dependency, making the design matrix singular and non-invertible.
- **Option C (Incorrect):** Structural multicollinearity refers to collinearity created by mathematical feature transformations such as polynomial terms ($x, x^2, x^3$) or interaction terms ($x_1 \cdot x_2$), not identical unit conversions.
- **Option D (Incorrect):** Orthogonality means variables are completely uncorrelated (inner product is zero). Here, the features have a correlation of 1.0, which is the exact opposite of orthogonal.

</details>

---

### 2. Question

Consider two regression models predicting employee salary:
* **Model 1:** Predictors are `years_experience` and `job_level`.
* **Model 2:** Predictors are `years_experience`, `job_level`, and `months_experience`.

What will happen to the ordinary least squares (OLS) normal equations when attempting to fit Model 2?

- [ ] A) The normal equations will yield infinitely many parameter combinations that minimize squared error.
- [ ] B) The training mean squared error will dramatically increase compared to Model 1.
- [ ] C) The model will automatically set the coefficient of `months_experience` to zero.
- [ ] D) Scikit-Learn will throw an unhandled syntax error before gradient descent can begin.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Because `months_experience` is an exact multiple ($12 \times$) of `years_experience`, the feature columns are linearly dependent. The matrix $X^T X$ becomes singular (determinant zero), meaning no unique inverse exists. Infinitely many combinations of coefficients produce the exact same prediction and identical minimum SSE.
- **Option B (Incorrect):** Adding redundant features never increases training error in OLS. Training error either stays identical or decreases; in this case of perfect collinearity, the minimum achievable training SSE remains unchanged.
- **Option C (Incorrect):** Standard OLS does not perform automatic feature selection or zeroing out; that is a specific property of L1-penalized regression (Lasso).
- **Option D (Incorrect):** Scikit-Learn's `LinearRegression` uses LAPACK SVD (`scipy.linalg.lstsq`), which produces a minimum-norm least-squares solution rather than crashing with a syntax error.

</details>

---

### 3. Question

An analyst fits a linear regression model to predict retail store sales using `advertising_tv_spend`, `advertising_radio_spend`, and `total_advertising_spend`, where `total_advertising_spend = advertising_tv_spend + advertising_radio_spend`.

What statistical condition describes this matrix configuration?

- [ ] A) Full column rank.
- [ ] B) Rank deficiency.
- [ ] C) Full row rank.
- [ ] D) Overdetermined non-collinear system.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Full column rank requires all feature columns to be linearly independent. Because the third column is the exact sum of the first two, the columns are linearly dependent, so the matrix cannot have full column rank.
- **Option B (Correct):** When at least one column is a linear combination of others, the rank of the $n \times p$ design matrix is strictly less than $p$ (the number of columns). This state is termed rank deficiency.
- **Option C (Incorrect):** Row rank concerns sample independence across rows ($n$). While the row rank also equals the column rank, the core problem for parameter estimation in regression ($n > p$) is column rank deficiency.
- **Option D (Incorrect):** An overdetermined system has more observations than features ($n > p$). While this dataset is likely overdetermined, it is strictly collinear and rank-deficient, not non-collinear.

</details>

---

### 4. Question

Which of the following real-world datasets is LEAST likely to suffer from severe multicollinearity?

- [ ] A) Predicting vehicle fuel efficiency using `engine_displacement`, `cylinder_count`, and `horsepower`.
- [ ] B) Predicting human blood pressure using `body_weight_kg`, `body_weight_lbs`, and `waist_circumference`.
- [ ] C) Predicting apartment rent using `latitude`, `longitude`, and `distance_to_downtown`.
- [ ] D) Predicting server CPU temperature using `ambient_room_temperature`, `network_packet_count`, and `hard_drive_total_capacity`.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Vehicle engine displacement, cylinder count, and horsepower are tightly coupled mechanical characteristics with very high mutual correlation.
- **Option B (Incorrect):** Including both kilograms and pounds creates an exact deterministic collinearity ($r = 1.0$), plus waist circumference is highly correlated with weight.
- **Option C (Incorrect):** `distance_to_downtown` is geometrically derived directly from spatial coordinates (`latitude` and `longitude`), creating strong geometric multicollinearity.
- **Option D (Correct):** Ambient room temperature, network traffic rate, and total disk storage capacity measure fundamentally different physical and computing domains with very low mutual collinearity.

</details>

---

### 5. Question

Two ML engineers are arguing about whether multicollinearity affects non-linear models. 

Which statement accurately contrasts how multicollinearity affects Linear Regression versus Random Forests?

- [ ] A) Linear regression produces unstable coefficients, whereas Random Forests can split on correlated features with minimal degradation to predictive accuracy.
- [ ] B) Random Forests fail to train because tree building requires computing the matrix inverse $(X^T X)^{-1}$.
- [ ] C) Linear regression handles multicollinearity smoothly, while Random Forests crash when two features have correlation exceeding 0.90.
- [ ] D) Both models experience severe coefficient sign flipping when multicollinearity is present.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Tree-based models like Random Forests evaluate split criteria one feature and one threshold at a time; they do not invert matrices or solve simultaneous linear equations. Thus, multicollinearity does not compromise their predictive performance, though it does dilute tree-based feature importance metrics.
- **Option B (Incorrect):** Decision trees and ensemble tree methods do not use linear algebra or matrix inversion during training; they use greedy splitting rules (e.g., Gini impurity or MSE reduction).
- **Option C (Incorrect):** This is backwards. Linear models suffer numerical instability under collinearity, whereas tree models handle correlated features without algorithmic breakdown.
- **Option D (Incorrect):** Decision trees do not have coefficients or signs; they partition the input space using orthogonal step boundaries.

</details>

---

### 6. Question

A data science team trains a multiple linear regression model. One engineer observes:
> *"The overall F-statistic of the regression has a p-value of 0.0001 (highly significant), but every single individual feature's t-test has a p-value greater than 0.15 (statistically insignificant)."*

What is the most likely diagnosis for this phenomenon?

- [ ] A) Severe underfitting caused by excessive L1 regularization.
- [ ] B) Extreme multicollinearity among the predictor variables.
- [ ] C) The target variable is normally distributed with zero variance.
- [ ] D) The dataset violates the homoscedasticity assumption but has zero feature correlation.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** L1 regularization would set redundant coefficients to zero, which sharpens the remaining features rather than causing all OLS t-tests to fail simultaneously.
- **Option B (Correct):** This is the classic diagnostic fingerprint of multicollinearity. The joint model explains significant variance (high $R^2$ and significant $F$-test), but because the predictors share redundant variance, the model cannot assign credit to any single predictor, inflating standard errors and driving individual $t$-statistics down.
- **Option C (Incorrect):** If the target variable had zero variance, $R^2$ would be undefined or 0, and the overall F-test would not be statistically significant.
- **Option D (Incorrect):** Heteroscedasticity affects the validity of standard errors, but a stark discrepancy between a significant overall F-test and universally non-significant individual t-tests is the hallmark signature of collinearity.

</details>

---

### 7. Question

What is the difference between collinearity and multicollinearity?

- [ ] A) Collinearity refers strictly to categorical variables, while multicollinearity refers strictly to continuous numerical variables.
- [ ] B) Collinearity involves a linear relationship between exactly two features; multicollinearity involves a linear relationship among three or more features.
- [ ] C) Collinearity affects only the intercept term, whereas multicollinearity affects only the slope coefficients.
- [ ] D) Collinearity is detected using VIF, while multicollinearity is detected only using pairwise correlation.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Both concepts apply to continuous and encoded numerical variables; the distinction is the number of variables involved, not the data type.
- **Option B (Correct):** Collinearity strictly denotes a high linear correlation between a pair of features ($x_1$ and $x_2$). Multicollinearity generalizes this to situations where one feature can be expressed as a linear combination of two or more other features ($x_1 \approx c_2 x_2 + c_3 x_3 + \dots$).
- **Option C (Incorrect):** Both collinearity and multicollinearity distort the estimation of feature slopes and their standard errors; they do not partition effects between intercept and slope in this manner.
- **Option D (Incorrect):** Pairwise correlation can only detect bivariate collinearity; VIF was specifically designed to detect multicollinearity across multiple variables.

</details>

---

### 8. Question

A data analyst is evaluating a dataset with 5 input features. The analyst finds that no pair of features has a Pearson correlation coefficient greater than $0.45$.

Can the analyst conclude with 100% certainty that the dataset is free from multicollinearity?

- [ ] A) Yes, because a correlation coefficient below 0.50 mathematically guarantees zero variance inflation.
- [ ] B) Yes, because pairwise correlation is both a necessary and sufficient condition for multicollinearity.
- [ ] C) No, because one feature could still be highly predictable from a linear combination of the other four features.
- [ ] D) No, because Pearson correlation only measures non-linear relationships.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Low pairwise correlations do not guarantee low VIF. A variable can have modest correlations with individual variables while being almost completely explained by their linear combination.
- **Option B (Incorrect):** Pairwise correlation is neither necessary nor sufficient for ruling out multicollinearity involving three or more variables.
- **Option C (Correct):** This is the classic blind spot of bivariate correlation analysis. For example, if $x_5 = x_1 + x_2 + x_3 + x_4$, the pairwise correlation between $x_5$ and any individual $x_i$ might only be $\approx 0.50$, yet perfect multicollinearity exists.
- **Option D (Incorrect):** Pearson correlation measures linear relationships, not non-linear relationships.

</details>

---

### 9. Question

Why does standardizing (z-score normalization) continuous features NOT eliminate multicollinearity between independent variables?

- [ ] A) Standardization shifts and rescales features linearly, which preserves the exact Pearson correlation between features.
- [ ] B) Standardization changes the sign of negative correlations into positive correlations.
- [ ] C) Standardization only affects the dependent variable $y$, leaving the feature matrix $X$ untouched.
- [ ] D) Standardization turns all continuous variables into categorical variables.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Standardizing subtracts the mean and divides by the standard deviation ($z = \frac{x - \mu}{\sigma}$). Since this is a strictly linear transformation, the correlation coefficient between any pair of standardized features is mathematically identical to that between the raw features ($r_{z_1, z_2} = r_{x_1, x_2}$).
- **Option B (Incorrect):** Standardization does not invert the direction of relationships; positive correlations remain positive and negative correlations remain negative.
- **Option C (Incorrect):** Feature standardization is applied directly to the columns of the design matrix $X$.
- **Option D (Incorrect):** Standardization preserves continuous numerical scales; it does not discretize or categorize data.

</details>

---

### 10. Question

In a housing dataset, an analyst creates two new features:
1. `rooms_per_household = total_rooms / households`
2. `bedrooms_per_household = total_bedrooms / households`

If `total_rooms` and `total_bedrooms` already had high collinearity, what is the most likely consequence for the newly engineered features?

- [ ] A) The engineered features will automatically have zero correlation.
- [ ] B) The engineered features will likely inherit strong multicollinearity because both were divided by the same denominator.
- [ ] C) The regression model will experience an immediate division-by-zero runtime error on all rows.
- [ ] D) The VIF of both features will drop to exactly 1.0.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Dividing two collinear variables by a common positive scaling factor does not eliminate their underlying collinearity; in many cases, it reinforces or preserves the correlation.
- **Option B (Correct):** Because `total_rooms` and `total_bedrooms` are already strongly correlated, scaling them both by the same feature (`households`) means houses with high room ratios will also have high bedroom ratios. The newly engineered features will continue to exhibit high collinearity.
- **Option C (Incorrect):** As long as `households` is strictly positive (non-zero), no division-by-zero error occurs.
- **Option D (Incorrect):** A VIF of 1.0 represents complete independence, which will not occur given the shared source data and common denominator.

</details>
