# Multicollinearity Objective Test — Set 03

---

### 1. Question

A data analyst generates the following correlation matrix using `df.corr()` for five features in a marketing dataset:

| Feature | clicks | impressions | spend | ctr | conversions |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **clicks** | 1.00 | 0.94 | 0.91 | 0.12 | 0.88 |
| **impressions** | 0.94 | 1.00 | 0.96 | -0.05 | 0.82 |
| **spend** | 0.91 | 0.96 | 1.00 | -0.02 | 0.85 |
| **ctr** | 0.12 | -0.05 | -0.02 | 1.00 | 0.15 |
| **conversions** | 0.88 | 0.82 | 0.85 | 0.15 | 1.00 |

If `conversions` is the target variable to predict, which pair of predictor features exhibits the strongest mutual collinearity?

- [ ] A) `clicks` and `ctr` ($r = 0.12$)
- [ ] B) `impressions` and `spend` ($r = 0.96$)
- [ ] C) `conversions` and `clicks` ($r = 0.88$)
- [ ] D) `impressions` and `ctr` ($r = -0.05$)

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** $r = 0.12$ indicates a very weak linear relationship, nowhere near collinearity.
- **Option B (Correct):** Collinearity is evaluated between *independent input features*. Among all pairs of inputs, `impressions` and `spend` share the highest mutual Pearson correlation ($r = 0.96$), indicating they convey almost identical scale information about campaign size.
- **Option C (Incorrect):** While $r = 0.88$ is strong, `conversions` is the dependent target variable, not a predictor. High correlation between a predictor and the target is desirable; collinearity specifically refers to correlation *among predictors*.
- **Option D (Incorrect):** $r = -0.05$ indicates virtually zero linear correlation.

</details>

---

### 2. Question

What is the primary conceptual limitation of relying solely on a bivariate correlation matrix to diagnose multicollinearity?

- [ ] A) Pearson correlation can only be computed for datasets with more than 10,000 rows.
- [ ] B) A correlation matrix only captures relationships between pairs of variables and cannot detect collinearity among combinations of three or more variables.
- [ ] C) Correlation values are sensitive to unit scaling (e.g., measuring in meters vs. centimeters).
- [ ] D) The diagonal of a correlation matrix is always zero, masking single-variable variance.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Pearson correlation can be computed on any sample size $n \ge 2$.
- **Option B (Correct):** Bivariate correlation tests only pairwise associations ($x_i$ with $x_j$). However, multicollinearity frequently occurs when a feature is linearly predictable from a combination of multiple other features (e.g., $x_3 \approx 0.5 x_1 + 0.5 x_2$) even if its pairwise correlation with each individual feature is modest (e.g., $r \approx 0.5$).
- **Option C (Incorrect):** Pearson correlation is scale-invariant; multiplying a feature by a constant or converting units does not change its correlation value.
- **Option D (Incorrect):** The diagonal of a correlation matrix is always 1.0 (a variable's correlation with itself), not zero.

</details>

---

### 3. Question

An analyst is inspecting a Seaborn heatmap created using `sns.heatmap(df.corr(), annot=True)`. 

Which visual characteristic indicates potential pairwise collinearity requiring intervention?

- [ ] A) Cells along the diagonal showing a value of 1.0.
- [ ] B) Off-diagonal cells displaying colors corresponding to correlation values near $+0.85$ or $-0.85$.
- [ ] C) Off-diagonal cells displaying values between $-0.10$ and $+0.10$.
- [ ] D) The color bar ranging from $-1.0$ to $+1.0$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The diagonal always represents each variable correlated with itself ($r = 1.0$), which is standard and expected for any correlation matrix.
- **Option B (Correct):** Strong correlation between distinct independent features (typically $|r| \ge 0.80$ to $0.90$) signals severe pairwise redundancy, warning the practitioner that one feature may need to be pruned or regularized.
- **Option C (Incorrect):** Values near zero indicate near-orthogonal, independent features, which is the ideal condition for ordinary least squares estimation.
- **Option D (Incorrect):** A color bar spanning from $-1.0$ to $+1.0$ is standard visualization configuration for correlation heatmaps.

</details>

---

### 4. Question

A junior developer runs `df.corr()` on a Pandas DataFrame containing both numeric and string columns and notices that several categorical columns are missing from the output.

Why are the categorical columns excluded?

- [ ] A) Categorical columns must be converted to datetime objects before correlation can be calculated.
- [ ] B) Pearson correlation requires numerical inputs to calculate covariances and standard deviations; Pandas skips non-numeric columns by default unless numeric encoding is applied.
- [ ] C) Pandas automatically deletes columns with string datatypes from memory.
- [ ] D) Categorical variables mathematically cannot possess collinearity with other features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Datetime objects are temporal data and do not directly enable Pearson correlation without extraction of numeric intervals.
- **Option B (Correct):** Pearson correlation requires continuous or numeric values to calculate means, deviations, and products. Pandas `df.corr(numeric_only=True)` automatically ignores string or object columns. Categorical variables must be one-hot encoded or ordinal encoded before their correlations can be assessed.
- **Option C (Incorrect):** Pandas does not delete columns from the DataFrame; it simply omits them from the returned correlation DataFrame.
- **Option D (Incorrect):** Categorical variables can easily create multicollinearity, especially when one-hot encoded (the classic "dummy variable trap").

</details>

---

### 5. Question

Which rule of thumb is most commonly recommended in statistical practice as an initial warning threshold for high pairwise correlation between two predictor variables?

- [ ] A) Absolute Pearson correlation $|r| > 0.10$
- [ ] B) Absolute Pearson correlation $|r| > 0.40$
- [ ] C) Absolute Pearson correlation $|r| > 0.80$ to $0.85$
- [ ] D) Absolute Pearson correlation $|r| = 0.00$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** $|r| = 0.10$ is negligible correlation and poses no collinearity threat whatsoever.
- **Option B (Incorrect):** $|r| = 0.40$ represents moderate correlation, which standard regression models handle comfortably without severe coefficient degradation.
- **Option C (Correct):** In applied statistics and machine learning, an absolute pairwise correlation exceeding $0.80$ (or $0.85$) is widely recognized as the red-flag zone where standard errors begin to inflate substantially and parameter estimates become erratic.
- **Option D (Incorrect):** A correlation of zero indicates complete orthogonality, which is the ideal state for regression features.

</details>

---

### 6. Question

An analyst calculates the correlation matrix for three features: $x_1$, $x_2$, and $x_3$. The matrix shows:
* $\text{corr}(x_1, x_2) = 0.15$
* $\text{corr}(x_1, x_3) = 0.20$
* $\text{corr}(x_2, x_3) = 0.18$

However, when running regression, the variance inflation factor (VIF) for $x_1$ is 42.0!

How is this mathematical discrepancy possible?

- [ ] A) The analyst made a coding mistake because VIF is mathematically bound to equal $1 / (1 - r_{12}^2)$.
- [ ] B) $x_1$ is strongly correlated with a linear combination of $x_2$ and $x_3$ (e.g., $x_1 \approx 2x_2 - 3x_3$), even though its individual pairwise correlations are small.
- [ ] C) VIF only measures negative correlation, which pairwise Pearson correlation ignores.
- [ ] D) Scikit-Learn scales VIF by the sample size $n$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The formula $\text{VIF} = 1 / (1 - r_{12}^2)$ applies *only* when there are exactly two features in the entire model. With three or more features, the formula uses $R_j^2$ from the multiple regression of $x_j$ on all other predictors.
- **Option B (Correct):** This is the definitive demonstration of why pairwise correlations are insufficient. A target feature $x_1$ can be almost perfectly reconstructed by a linear combination of two or more other features ($R_1^2 \to 1.0$), resulting in an astronomical VIF, even while its individual pairwise projections onto $x_2$ and $x_3$ separately appear modest.
- **Option C (Incorrect):** Both Pearson correlation and VIF account for negative relationships; VIF squares the multiple correlation ($R_j^2$), capturing all variance regardless of direction.
- **Option D (Incorrect):** VIF is calculated from $R_j^2$ and is scale-invariant with respect to sample size $n$.

</details>

---

### 7. Question

Suppose two features $x_1$ and $x_2$ have a Pearson correlation coefficient of $r = -0.95$.

Does this strong negative correlation cause multicollinearity, or does multicollinearity only apply to positive correlations?

- [ ] A) Multicollinearity applies equally to strong negative correlations because the linear dependency is just as strong ($x_2 \approx -k \cdot x_1$).
- [ ] B) Strong negative correlation prevents multicollinearity because the features cancel each other out.
- [ ] C) Negative correlation only causes multicollinearity if the intercept term is forced to zero.
- [ ] D) Negative correlation indicates orthogonality.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Linear dependency depends on the *magnitude* (absolute value) of the linear relationship, not its direction. A negative correlation of $-0.95$ means one variable can be predicted from the other with $90.25\%$ accuracy ($R^2 = (-0.95)^2 = 0.9025$), creating the exact same matrix ill-conditioning and coefficient variance inflation as a positive correlation of $+0.95$.
- **Option B (Incorrect):** Negative correlation does not "cancel out" collinearity; it creates the same rank deficiency in the feature space.
- **Option C (Incorrect):** The intercept term does not change the linear dependency between feature columns.
- **Option D (Incorrect):** Orthogonality means zero correlation ($r = 0.0$), not negative correlation.

</details>

---

### 8. Question

What is the relationship between the Pearson correlation coefficient between two features ($r_{12}$) and their pairwise Variance Inflation Factor?

- [ ] A) $\text{VIF} = 1 - r_{12}$
- [ ] B) $\text{VIF} = \frac{1}{1 - r_{12}^2}$
- [ ] C) $\text{VIF} = \sqrt{r_{12}}$
- [ ] D) $\text{VIF} = \frac{r_{12}}{1 + r_{12}}$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This is a linear subtraction, which does not model variance explosion as correlation approaches 1.
- **Option B (Correct):** In a bivariate model with only two features $x_1$ and $x_2$, the $R^2$ obtained from regressing $x_1$ on $x_2$ is simply $r_{12}^2$. Therefore, the VIF simplifies directly to $\text{VIF} = \frac{1}{1 - R^2} = \frac{1}{1 - r_{12}^2}$. As $r_{12} \to 1$ or $-1$, the denominator approaches 0, and VIF approaches infinity.
- **Option C (Incorrect):** Square root would reduce values rather than accelerating to infinity near unity.
- **Option D (Incorrect):** This does not correspond to any known collinearity diagnostic.

</details>

---

### 9. Question

If two features have a pairwise Pearson correlation of $r = 0.90$, what is their exact bivariate VIF?

- [ ] A) $1.90$
- [ ] B) $5.26$
- [ ] C) $9.00$
- [ ] D) $10.00$

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Calculated by simple addition ($1 + 0.90$), which is mathematically incorrect.
- **Option B (Correct):** Using the bivariate VIF formula:
  $$r^2 = (0.90)^2 = 0.81$$
  $$\text{VIF} = \frac{1}{1 - r^2} = \frac{1}{1 - 0.81} = \frac{1}{0.19} \approx 5.26$$
  A VIF of $5.26$ exceeds the common threshold of $5.0$, confirming moderate-to-high multicollinearity.
- **Option C (Incorrect):** Mistaking $r \times 10 = 9.0$ for VIF.
- **Option D (Incorrect):** Approximating without computing the exact reciprocal of $0.19$.

</details>

---

### 10. Question

A data scientist creates a pairplot using `sns.pairplot(df)` on four continuous features. 

What visual pattern between two features on the scatter plot indicates that pairwise collinearity is absent?

- [ ] A) Data points tightly clustered along a narrow diagonal line sloping upward from left to right.
- [ ] B) Data points tightly clustered along a narrow diagonal line sloping downward from left to right.
- [ ] C) A diffuse, circular, shotgun-pattern cloud of points with no discernable linear trend.
- [ ] D) A parabolic U-shaped curve.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A narrow upward diagonal indicates a strong positive linear correlation ($r \approx 1.0$), signaling high collinearity.
- **Option B (Incorrect):** A narrow downward diagonal indicates a strong negative linear correlation ($r \approx -1.0$), also signaling high collinearity.
- **Option C (Correct):** A spherical or circular scatter cloud shows that knowing the value of feature A provides zero predictive information about feature B ($r \approx 0.0$). This represents independence and absence of collinearity.
- **Option D (Incorrect):** A U-shaped curve indicates a non-linear relationship. While Pearson correlation may be low, the variables are still functionally dependent.

</details>
