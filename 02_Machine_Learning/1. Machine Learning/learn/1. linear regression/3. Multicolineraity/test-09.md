# Multicollinearity Objective Test — Set 09

---

### 1. Question

How does Principal Component Analysis (PCA) mathematically resolve multicollinearity among input features?

- [ ] A) By deleting all features that have negative correlations with the target.
- [ ] B) By performing an orthogonal linear transformation that projects the original correlated features into a new set of linearly uncorrelated variables (principal components).
- [ ] C) By multiplying the feature matrix by the inverse of the target vector.
- [ ] D) By converting all continuous features into categorical bins.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** PCA is an unsupervised technique that does not examine or use the target variable $y$.
- **Option B (Correct):** PCA computes the eigenvectors of the feature covariance (or correlation) matrix. Because the covariance matrix is symmetric, its eigenvectors are mutually orthogonal. Projecting the original data onto these eigenvectors yields principal components whose mutual covariance and correlation are mathematically guaranteed to be exactly zero, completely eliminating multicollinearity.
- **Option C (Incorrect):** PCA does not involve target vector inversion.
- **Option D (Incorrect):** PCA preserves continuous representations; it does not discretize features into bins.

</details>

---

### 2. Question

What is the correlation between any two distinct Principal Components ($PC_i$ and $PC_j$, where $i \ne j$) generated from a standard PCA transformation?

- [ ] A) Exactly $+1.0$
- [ ] B) Exactly $-1.0$
- [ ] C) Exactly $0.0$
- [ ] D) It equals the correlation of the original two most collinear features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A correlation of $+1.0$ would mean perfect collinearity, the exact opposite of what PCA achieves.
- **Option B (Incorrect):** A correlation of $-1.0$ also indicates perfect collinearity.
- **Option C (Correct):** By definition and construction, principal component axes are chosen to be mutually orthogonal (perpendicular in space). Consequently, the empirical covariance and Pearson correlation between any two distinct principal components is identically zero ($\text{corr}(PC_i, PC_j) = 0$ for all $i \ne j$).
- **Option D (Incorrect):** The transformation decouples the original correlations completely.

</details>

---

### 3. Question

An ML engineer fits a multiple linear regression model using Principal Components as predictors:
$$\hat{y} = \beta_0 + \beta_1 PC_1 + \beta_2 PC_2 + \beta_3 PC_3$$

What will be the Variance Inflation Factor (VIF) for each of these three principal components?

- [ ] A) Exactly $1.0$ for each component.
- [ ] B) Infinite ($\infty$) for all components.
- [ ] C) Approximately equal to the condition number of the original matrix.
- [ ] D) $0.0$ for each component.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Because all principal components have zero mutual correlation ($\text{Cov}(PC_i, PC_j) = 0$), regressing any principal component $PC_j$ on the remaining components yields an auxiliary $R_j^2$ of exactly $0.0$. Thus:
  $$\text{VIF}_j = \frac{1}{1 - R_j^2} = \frac{1}{1 - 0} = 1.0$$
  Every principal component has a VIF of exactly $1.0$, representing pristine, zero-collinearity conditioning.
- **Option B (Incorrect):** Infinite VIF occurs under perfect collinearity ($R^2 = 1.0$).
- **Option C (Incorrect):** PCA rotates the coordinate system into the eigenspace, neutralizing the original matrix's ill-conditioning.
- **Option D (Incorrect):** The theoretical minimum value of VIF is $1.0$, not $0.0$.

</details>

---

### 4. Question

What is the primary operational trade-off when using Principal Component Regression (PCR) to solve multicollinearity?

- [ ] A) The model becomes faster to train, but requires 100 times more memory.
- [ ] B) Multicollinearity is completely eliminated, but the direct interpretability of individual original features is lost because each principal component is an abstract linear combination of all original inputs.
- [ ] C) The training $R^2$ is mathematically guaranteed to reach $1.0$.
- [ ] D) PCR can only be evaluated on datasets with an odd number of features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** PCR reduces dimensionality and uses less memory, not more.
- **Option B (Correct):** While PCR completely cures multicollinearity and prevents matrix singularity, each principal component is a weighted mixture of all original variables (e.g., $PC_1 = 0.52 \cdot \text{sqft} + 0.49 \cdot \text{rooms} - 0.31 \cdot \text{age}$). A stakeholder cannot ask: *"What is the dollar value of an extra bedroom?"*, because bedrooms no longer exist as an isolated coefficient.
- **Option C (Incorrect):** Reducing the number of components discards some feature variance, so training $R^2$ is typically slightly lower than the full OLS model.
- **Option D (Incorrect):** PCA works on any number of features $\ge 1$.

</details>

---

### 5. Question

A data scientist uses `sklearn.decomposition.PCA` on three collinear features: `sqft`, `total_rooms`, and `bedrooms`. The explained variance ratio of the components is:
* Component 1 ($PC_1$): $89.5\%$
* Component 2 ($PC_2$): $8.2\%$
* Component 3 ($PC_3$): $2.3\%$

What does this distribution of explained variance indicate?

- [ ] A) The three features are largely orthogonal and should all be retained independently.
- [ ] B) The vast majority of the collective information ($89.5\%$) across all three features lies along a single underlying dimension representing "overall property size."
- [ ] C) The PCA transformation failed because Component 1 does not equal 100%.
- [ ] D) The target variable is uncorrelated with house size.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** If features were orthogonal, variance would be distributed roughly equally ($33\%$ each).
- **Option B (Correct):** In severe multicollinearity, the data cloud stretches primarily along one dominant axis. Here, $PC_1$ captures nearly $90\%$ of all variance across the three features, confirming that they are all tracking a single latent construct ("size"). Retaining only $PC_1$ preserves nearly all physical variation while compressing three collinear columns into one.
- **Option C (Incorrect):** In real data with noise, a single component rarely explains $100\%$ of total variance.
- **Option D (Incorrect):** PCA is unsupervised and does not evaluate correlation with the target.

</details>

---

### 6. Question

Why is feature scaling (e.g., `StandardScaler`) essential BEFORE performing PCA to address multicollinearity?

- [ ] A) Because PCA cannot run on negative numbers.
- [ ] B) Because PCA maximizes variance; an unscaled feature with a large numerical range (e.g., income in dollars) will arbitrarily dominate the first principal component regardless of its actual importance or correlation structure.
- [ ] C) Because scaling ensures the target variable is normally distributed.
- [ ] D) Because scikit-learn will throw a `ValueError` if the data variance is not equal to 1.0.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** PCA works on real numbers, both positive and negative.
- **Option B (Correct):** PCA finds directions of maximum variance. If feature $x_1$ has a scale in the thousands (e.g., square feet: variance $\approx 100,000$) and $x_2$ has a scale in single digits (e.g., bedrooms: variance $\approx 2$), the first principal component will align almost $100\%$ with $x_1$ simply due to unit choice. Standardization equalizes column variances to $1.0$, ensuring PCA reflects true correlation structure rather than measurement units.
- **Option C (Incorrect):** PCA is applied to input features $X$, not target $y$.
- **Option D (Incorrect):** Scikit-Learn will run PCA on unscaled data without crashing, but the resulting components will be statistically distorted.

</details>

---

### 7. Question

What is a fundamental difference between Principal Component Regression (PCR) and Partial Least Squares (PLS) regression when handling multicollinear features?

- [ ] A) PCR is unsupervised (it constructs components based solely on feature variance without looking at $y$), whereas PLS is supervised (it constructs components that maximize covariance between the features and the target $y$).
- [ ] B) PCR uses neural networks, while PLS uses decision trees.
- [ ] C) PLS only works when features are perfectly orthogonal.
- [ ] D) PCR can only be used for binary classification.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** PCR first applies standard unsupervised PCA to $X$ and then regresses $y$ on the chosen components. A potential flaw is that the direction of greatest variance in $X$ might not be the direction most predictive of $y$. In contrast, PLS finds latent components that simultaneously explain variance in $X$ and maximize correlation with $y$, making it a supervised dimensionality reduction technique tailored for multicollinear regression.
- **Option B (Incorrect):** Both PCR and PLS are linear algebraic techniques, not deep neural networks or decision trees.
- **Option C (Incorrect):** PLS was explicitly invented for high-dimensional, highly collinear data (popular in chemometrics and spectroscopy).
- **Option D (Incorrect):** PCR is a general regression technique for continuous targets.

</details>

---

### 8. Question

An analyst plots a "Scree Plot" after running PCA on 10 collinear features. 

What is the purpose of the scree plot in this context?

- [ ] A) To display the p-values of individual t-tests.
- [ ] B) To visualize the eigenvalues (or percentage of explained variance) for each successive principal component to determine the optimal number of components to retain (the "elbow").
- [ ] C) To check whether the regression residuals are homoscedastic.
- [ ] D) To identify outlier rows in the training set.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Scree plots display unsupervised PCA eigenvalues, not regression t-test p-values.
- **Option B (Correct):** A scree plot graphs each principal component on the x-axis against its eigenvalue or percentage of explained variance on the y-axis. By locating the "elbow" (the point where the downward curve flattens out), the analyst can choose how many components to keep to capture the bulk of information while discarding noise and eliminating collinearity.
- **Option C (Incorrect):** Residual homoscedasticity is evaluated using residual-versus-fitted plots, not scree plots.
- **Option D (Incorrect):** Outlier detection uses leverage, Cook's distance, or boxplots.

</details>

---

### 9. Question

A data scientist fits PCA on a training set of collinear features, selects the top 2 components, and trains a linear regression model. 

When new production data arrives for inference, how must the new samples be processed before feeding them into the trained model?

- [ ] A) Train a new PCA model on the incoming production batch.
- [ ] B) Transform the new raw features using the exact `StandardScaler` and `PCA` transformers fitted on the training data (`pca.transform(scaler.transform(X_new))`).
- [ ] C) Pass the raw unscaled features directly to `model.predict()`.
- [ ] D) Discard all features except the target variable.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Fitting a new PCA model on inference batches causes data leakage, inconsistent coordinate axes, and catastrophic prediction failure.
- **Option B (Correct):** The model weights were learned on the principal component coordinate system established by the training set. New inference data must first be normalized using the training means and standard deviations, then projected onto the exact same training eigenvector basis using `pca.transform()`. This is best organized in a Scikit-Learn `Pipeline`.
- **Option C (Incorrect):** Passing raw features into a model expecting 2 principal components will raise a feature shape mismatch error.
- **Option D (Incorrect):** The target variable is what you are trying to predict; it is not available at inference time.

</details>

---

### 10. Question

If an analyst retains ALL $p$ principal components from a dataset with $p$ collinear features and runs linear regression on all $p$ components, what will be the model's predictions compared to the original unregularized OLS model?

- [ ] A) The predictions will be mathematically identical to the original OLS model predictions.
- [ ] B) The predictions will all equal zero.
- [ ] C) The predictions will have significantly higher Mean Squared Error.
- [ ] D) The model will crash due to infinite multicollinearity.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** PCA is an invertible orthogonal change of basis. If all $p$ principal components are retained, the column space spanned by the components is identical to the column space spanned by the original features ($\text{span}(PC) = \text{span}(X)$). Therefore, the fitted projection vector $\hat{y}$ and all predictions are mathematically identical to unconstrained OLS. PCR only modifies predictions when one or more low-variance components are discarded.
- **Option B (Incorrect):** The full rank transformation preserves the full linear subspace.
- **Option C (Incorrect):** Without dropping components, the loss minimization problem finds the exact same global optimum.
- **Option D (Incorrect):** The principal components are mutually orthogonal, so their covariance matrix is a diagonal matrix, avoiding collinearity crashes.

</details>
