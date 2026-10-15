# Multicollinearity Objective Test — Set 05

---

### 1. Question

A data analyst computes VIF scores for a feature set and observes the following output:

| Feature | VIF |
| :--- | :---: |
| `bedrooms` | 3.2 |
| `bathrooms` | 4.1 |
| `garage_spaces` | 2.5 |
| `house_sqft` | 14.8 |
| `lot_size` | 1.8 |

Based on standard statistical conventions, which feature definitively requires remediation due to severe multicollinearity?

- [ ] A) `lot_size`
- [ ] B) `garage_spaces`
- [ ] C) `bedrooms`
- [ ] D) `house_sqft`

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** A VIF of $1.8$ is well below $5$, indicating low collinearity and excellent independence.
- **Option B (Incorrect):** $2.5$ is safely in the acceptable range.
- **Option C (Incorrect):** $3.2$ is below the conservative threshold of $5.0$.
- **Option D (Correct):** In applied statistics, a VIF threshold of $5$ (conservative) or $10$ (standard) is used to identify problematic features. `house_sqft` with a VIF of $14.8$ exceeds both thresholds, indicating that over $93\%$ of its variance is redundant with the other property characteristics.

</details>

---

### 2. Question

What is the "Dummy Variable Trap" in multiple linear regression?

- [ ] A) Encoding categorical variables as floating point numbers instead of integers.
- [ ] B) Including one dummy column for every single level of a categorical variable while also retaining the model's intercept term, causing perfect multicollinearity.
- [ ] C) Forgetting to apply one-hot encoding to binary (0/1) features.
- [ ] D) Using target encoding when cross-validation is not configured properly.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Data type representation (float vs. int) does not induce the dummy variable trap.
- **Option B (Correct):** If a categorical variable has $k$ levels and all $k$ binary columns are added to a model that includes an intercept, the sum of all $k$ columns across every observation equals exactly $1.0$ (matching the constant intercept column $\mathbf{1}$). This creates perfect linear dependency ($\sum_{j=1}^k D_j = \text{const}$), leading to a singular matrix that cannot be inverted. The standard fix is to drop one category (using $k-1$ dummies, e.g., `pd.get_dummies(..., drop_first=True)`).
- **Option C (Incorrect):** Binary features do not require multi-column one-hot encoding.
- **Option D (Incorrect):** Target leakage is a risk of target encoding, but it is distinct from the dummy variable trap.

</details>

---

### 3. Question

A data scientist creates dummy variables for a 3-category column `region` (`North`, `South`, `East`) using `pd.get_dummies(df['region'])` without setting `drop_first=True`. She then fits `statsmodels.OLS` with `sm.add_constant`.

What will be the Variance Inflation Factor for the dummy columns?

- [ ] A) Exactly $1.0$ for all three columns.
- [ ] B) Exactly $0.0$.
- [ ] C) Infinite ($\infty$) or an extremely large floating-point value reflecting near-infinite condition numbers.
- [ ] D) Exactly $3.0$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A VIF of 1 indicates total independence; here, the sum of the columns is perfectly predictable.
- **Option B (Incorrect):** VIF has a theoretical minimum of 1.0; it can never be 0.
- **Option C (Correct):** Because `North + South + East = 1` for every row, and the constant column also equals `1`, the auxiliary regression predicting any of these columns from the remaining two plus the constant yields an $R^2$ of exactly $1.0$. Hence, $\text{VIF} = 1 / (1 - 1) = 1 / 0 = \infty$. In numerical libraries, this results in `inf` or astronomical values ($10^{15}$).
- **Option D (Incorrect):** VIF is not the count of categories.

</details>

---

### 4. Question

An engineer fits a polynomial regression model:
$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$$
where $x$ takes positive values ranging between $10$ and $50$. 

The VIF scores for $x, x^2,$ and $x^3$ all exceed $150$. What type of multicollinearity is this, and what is the standard method to mitigate it?

- [ ] A) Stochastic multicollinearity; mitigate by removing the target variable $y$.
- [ ] B) Structural multicollinearity; mitigate by mean-centering the predictor ($x - \bar{x}$) before computing powers.
- [ ] C) Perfect multicollinearity; mitigate by setting all coefficients to zero.
- [ ] D) Outlier-induced collinearity; mitigate by clipping negative values.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Collinearity caused by mathematical transformations of the same base variable is structural, not stochastic.
- **Option B (Correct):** Structural multicollinearity is a byproduct of mathematical transformations like powers ($x^2, x^3$) or interaction terms ($x_1 \cdot x_2$). Because $x$ is strictly positive, higher powers are strongly correlated with lower powers. Subtracting the sample mean ($\tilde{x} = x - \bar{x}$) makes the linear and quadratic components approximately orthogonal, slashing VIF scores dramatically.
- **Option C (Incorrect):** Polynomial terms are non-linear transformations and are not linearly identical; collinearity is imperfect (though severe).
- **Option D (Incorrect):** The problem is not caused by outliers, but by the monotonic positive relationship between powers of positive numbers.

</details>

---

### 5. Question

A machine learning practitioner is reviewing two common threshold guidelines for VIF:
* Guideline 1: VIF $> 5$
* Guideline 2: VIF $> 10$

Why do some statisticians prefer the stricter threshold of $5$ over $10$?

- [ ] A) Because a VIF of $5$ corresponds to an auxiliary $R^2$ of $0.80$, meaning that $80\%$ of that feature's variance is shared with other features.
- [ ] B) Because a VIF of $5$ means the sample size is divided by $5$.
- [ ] C) Because modern computers cannot calculate matrix inverses when VIF exceeds $5$.
- [ ] D) Because VIF $> 5$ violates the assumption of normally distributed errors.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When $\text{VIF} = 5$, $\text{Tolerance} = 1 / 5 = 0.20$, which implies $R^2 = 0.80$. Many researchers consider a situation where four-fifths ($80\%$) of a feature's variance is shared with other predictors to already represent excessive redundancy, warranting remediation before it reaches the looser $10$ threshold ($R^2 = 0.90$).
- **Option B (Incorrect):** VIF does not alter the number of rows in the dataset.
- **Option C (Incorrect):** Numerical libraries routinely invert matrices with condition numbers corresponding to VIFs in the hundreds or thousands.
- **Option D (Incorrect):** Multicollinearity has no mathematical bearing on the normality of regression errors.

</details>

---

### 6. Question

When should a data scientist NOT automatically drop a feature despite a VIF exceeding $10$?

- [ ] A) When the feature is a control variable that domain theory dictates must remain in the model, and prediction/causal inference on that specific coefficient is not the primary objective.
- [ ] B) When the feature has an auxiliary $R^2$ of exactly $1.0$.
- [ ] C) When the target variable is binary.
- [ ] D) When the sample size is smaller than 20 rows.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In econometrics, epidemiology, and social sciences, certain control variables (e.g., age, income, education) must be included to prevent omitted variable bias on the primary treatment variable, even if the controls are mutually collinear. If researchers only need to interpret the treatment coefficient, high VIF among control variables can be safely tolerated.
- **Option B (Incorrect):** An $R^2$ of $1.0$ represents perfect collinearity, which makes the normal equations unsolvable without dropping a variable or regularizing.
- **Option C (Incorrect):** Multicollinearity distorts logistic regression as severely as linear regression.
- **Option D (Incorrect):** Small sample sizes exacerbate the instability caused by high VIF, making it even more hazardous.

</details>

---

### 7. Question

An analyst runs an automated pruning script that iteratively calculates VIF for all features and drops the feature with the highest VIF until all VIF scores are below $5.0$.

Why must VIF scores be recalculated AFTER EVERY SINGLE FEATURE IS DROPPED, rather than dropping all features with initial VIF $> 5$ at once?

- [ ] A) Because statsmodels crashes if more than one feature is removed at a time.
- [ ] B) Because dropping one collinear feature often causes the VIF of several remaining collinear features to drop immediately below the threshold.
- [ ] C) Because VIF values are permanently saved to disk and cannot be overwritten.
- [ ] D) Because dropping features changes the sample size $n$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Statsmodels or pandas can easily drop multiple columns simultaneously.
- **Option B (Correct):** Collinearity is a shared property between variables. If features $A, B,$ and $C$ are all mutually collinear, all three may initially show VIFs $> 20$. Dropping feature $A$ eliminates the source of redundancy, often causing the VIFs of both $B$ and $C$ to plummet from $20$ down to $1.8$. Dropping all three simultaneously would needlessly discard valuable, distinct information.
- **Option C (Incorrect):** DataFrames are in-memory objects and recalculate dynamically.
- **Option D (Incorrect):** Dropping columns (features) changes dimension $p$, not sample count $n$.

</details>

---

### 8. Question

What is the relationship between sample size $n$ and the severity of multicollinearity's impact on parameter estimates?

- [ ] A) Larger sample sizes reduce VIF scores directly.
- [ ] B) While larger sample sizes do not reduce VIF itself (which depends only on feature correlation), large $n$ decreases baseline standard error, helping to offset variance inflation.
- [ ] C) Increasing sample size guarantees that perfect multicollinearity disappears.
- [ ] D) Sample size has zero mathematical interaction with the standard errors of regression coefficients.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** VIF is calculated from $R_j^2$, which reflects correlation structure, not row count. As $n \to \infty$, the sample correlation approaches the population correlation, so VIF does not shrink to 1.
- **Option B (Correct):** The variance of $\hat{\beta}_j$ is given by $\frac{\sigma^2}{(n-1)s_j^2} \times \text{VIF}_j$. Notice that $(n-1)$ is in the denominator. Even if VIF is high (e.g., $10$), having a massive sample size ($n = 1,000,000$) can make the base term so minuscule that the resulting standard error remains tiny and estimates remain relatively stable.
- **Option C (Incorrect):** Perfect multicollinearity is an algebraic rank deficiency that persists regardless of whether $n$ is $10$ or $10,000,000$.
- **Option D (Incorrect):** Standard error is inversely proportional to $\sqrt{n}$.

</details>

---

### 9. Question

A data analyst encounters high VIF values after adding an interaction term between two continuous variables:
`interaction = age * income`

Both `age`, `income`, and `interaction` show VIF values exceeding $35$.

What is the most statistically sound first step to reduce this structural multicollinearity without removing the interaction effect?

- [ ] A) Center both `age` and `income` by subtracting their respective means before calculating the product term: `interaction = (age - mean_age) * (income - mean_income)`.
- [ ] B) Discard the target variable and train an unsupervised model.
- [ ] C) Log-transform the target variable while leaving the features uncentered.
- [ ] D) Multiply `age` by a factor of 100.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When two variables with positive means are multiplied, the product term is inherently correlated with both individual components. Mean-centering transforms the features so that their means are zero, drastically reducing the linear correlation between the main effects and the interaction term while preserving the exact interpretability of the interaction hypothesis.
- **Option B (Incorrect):** Discarding the target defeats the entire purpose of a supervised regression task.
- **Option C (Incorrect):** Transforming $y$ alters the scale of residuals, but does nothing to change the correlation among the feature columns in $X$.
- **Option D (Incorrect):** Multiplying by a constant is a linear rescaling that does not change correlations or VIF scores.

</details>

---

### 10. Question

Which of the following scenarios represents a "false alarm" where a high VIF does NOT compromise the validity of the statistical analysis?

- [ ] A) Predicting loan default where credit score and debt-to-income ratio have high VIF, and the bank needs to justify loan rejections based on individual coefficients.
- [ ] B) In a medical study where age, weight, and blood pressure are collinear, but the goal is solely to predict 5-year patient survival risk accurately on new patient data from the same population.
- [ ] C) An economist who needs to estimate the exact price elasticity of demand for milk to recommend a tax policy.
- [ ] D) A legal audit examining whether gender has a statistically significant independent coefficient in salary discrimination litigation.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Adverse action notices require legally defensible explanations of individual feature contributions, which high VIF corrupts.
- **Option B (Correct):** In pure risk prediction without causal or structural coefficient claims, multicollinearity does not bias predictions on patients drawn from the same clinical distribution. The model predicts overall survival accurately even if credit is shared arbitrarily among physiological markers.
- **Option C (Incorrect):** Price elasticity is the direct coefficient of price; collinearity with other market variables would make elasticity estimates erratic and untrustworthy.
- **Option D (Incorrect):** Legal discrimination claims hinge on the isolated coefficient and p-value for gender, which are distorted by collinearity with tenure or role.

</details>
