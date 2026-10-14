# Matplotlib Scatter Plot Objective Test — Set 09

---

### 81. Question

A data scientist needs to visualize a classification dataset with three distinct classes (`"Class A"`, `"Class B"`, `"Class C"`). Which code pattern correctly generates distinct color-coded points and an automatic, labeled discrete legend?

- [ ] **A.**
  ```python
  plt.scatter(x, y, classes=["Class A", "Class B", "Class C"])
  plt.show()
  ```
- [ ] **B.**
  ```python
  plt.scatter(x, y, multi_class=True)
  plt.legend()
  ```
- [ ] **C.**
  ```python
  plt.scatter(x, y, label="All Classes")
  plt.legend()
  ```
- [ ] **D.**
  ```python
  for cls in np.unique(classes):
    mask = classes == cls
    plt.scatter(x[mask], y[mask], label=cls, alpha=0.7)
  plt.legend()
  plt.show()
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `classes` is not a valid keyword argument in `plt.scatter()`; calling it raises a `TypeError`.
- **Option B (Incorrect):** `multi_class` is not supported in the Matplotlib API.
- **Option C (Incorrect):** Passing a single static label string lumps all observations into one generic legend item without differentiating between the classes.
- **Option D (Correct):** In Matplotlib, the standard idiomatic way to plot categorical multi-class data is to iterate through the unique class labels. In each iteration, a boolean mask filters the subset of $(x, y)$ coordinates belonging to that class, and a separate `plt.scatter()` call is made with `label=cls`. Matplotlib automatically cycles through distinct default colors and registers separate proxy artist handles, allowing `plt.legend()` to construct a complete, labeled discrete legend.

</details>

---

### 82. Question

A scatter plot of income ($x$) versus annual discretionary entertainment spending ($y$) displays the following geometric pattern:
- For low income ($x < \$30,000$), points are tightly clustered near $y \approx \$1,000$.
- As income increases toward $\$200,000$, the vertical spread of points widens dramatically into an expansive cone/funnel shape, with spending ranging anywhere from $\$2,000$ to $\$40,000$.

What formal statistical violation in linear regression does this cone-shaped dispersion pattern indicate?

- [ ] **A.** Heteroscedasticity (non-constant error variance)
- [ ] **B.** Multicollinearity
- [ ] **C.** Autoregressive lag correlation
- [ ] **D.** Perfect separation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** **Heteroscedasticity** refers to the condition where the variance of the residuals (or dependent variable $y$) is unequal across the range of values of an independent predictor $x$. Visually on a scatter plot, it appears as a classic cone, fan, or trumpet shape widening (or narrowing) along the horizontal axis, violating the Gauss-Markov homoscedasticity assumption of ordinary least squares (OLS) regression.
- **Option B (Incorrect):** Multicollinearity refers to high linear correlation between two or more independent predictor features ($X_1$ and $X_2$).
- **Option C (Incorrect):** Autoregressive lag correlation describes time-series dependence between consecutive time periods ($y_t$ vs $y_{t-1}$).
- **Option D (Incorrect):** Perfect separation occurs when a hyperplane completely divides two discrete classification groups.

</details>

---

### 83. Question

In contrast to heteroscedasticity, what visual pattern on a scatter plot indicates that the linear regression assumption of **homoscedasticity** is satisfied?

- [ ] **A.** Points form an exponential parabolic curve.
- [ ] **B.** Points are partitioned into two completely detached circular clusters.
- [ ] **C.** Points are dispersed in a uniform horizontal band of constant vertical width/thickness around the regression line throughout the entire domain of $x$.
- [ ] **D.** All points lie in the first quadrant.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A parabolic curve indicates non-linearity, not homoscedasticity.
- **Option B (Incorrect):** Detached clusters suggest distinct multi-modal subpopulations.
- **Option C (Correct):** **Homoscedasticity** (constant variance) means that the conditional variance $\text{Var}(Y \mid X)$ remains unchanged across all levels of $X$. On a scatter plot with a regression line, the points scatter in a parallel band of uniform thickness from the left boundary to the right boundary, with no widening or contracting funnel shapes.
- **Option D (Incorrect):** Coordinate quadrant placement has no bearing on variance uniformity.

</details>

---

### 84. Question

A biomedical researcher plots drug dosage ($x$) against therapeutic symptom relief ($y$). The scatter plot shows that relief rises steeply as dosage increases from 0 to 10 mg, but beyond 20 mg, the curve flattens out into an asymptotic horizontal plateau (law of diminishing marginal returns).

Which mathematical model or feature transformation should replace simple linear regression $\hat{y} = m x + b$ to properly capture this curvature?

- [ ] **A.** High-order polynomial of degree 10: $\hat{y} = \sum_{i=0}^{10} c_i x^i$
- [ ] **B.** Logarithmic model $\hat{y} = a + b \ln(x)$ or saturating Hill/Michaelis-Menten kinetic model
- [ ] **C.** Inverted quadratic model with positive leading coefficient: $\hat{y} = +a x^2$
- [ ] **D.** Linear model with interaction term: $\hat{y} = m x \cdot y + b$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A 10th-degree polynomial will overfit the noise and produce severe Runge oscillations rather than monotonic plateauing.
- **Option B (Correct):** Natural processes displaying diminishing returns and plateau saturation are canonically modeled using a logarithmic function ($\hat{y} = a + b \ln x$) or non-linear saturating equations (such as Michaelis-Menten or logistic growth models). These functions rise rapidly at low levels and gradually flatten out asymptotically.
- **Option C (Incorrect):** An upward quadratic model ($\hat{y} = ax^2$ with $a > 0$) curves upward faster and faster, which is the exact opposite of a flattening plateau.
- **Option D (Incorrect):** Placing the target $y$ on the right-hand side of the regression equation is circular and invalid.

</details>

---

### 85. Question

A machine learning engineer plots two features ($X_1$ on the horizontal axis, $X_2$ on the vertical axis) for two classification categories: Class 0 (blue dots) and Class 1 (orange triangles).

The scatter plot shows that a single straight ruler can be drawn across the canvas such that all blue dots lie on one side of the line and all orange triangles lie on the other side.

What fundamental machine learning property does this visual scatter pattern demonstrate?

- [ ] **A.** The two classes are **linearly separable** in the 2D feature space, meaning a linear classifier (such as Logistic Regression or a linear Support Vector Machine) can achieve 100% training accuracy.
- [ ] **B.** The dataset has severe multicollinearity and must be discarded.
- [ ] **C.** The features $X_1$ and $X_2$ have zero mutual information.
- [ ] **D.** The classification problem requires a deep neural network with at least 5 hidden layers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When two categories in a 2D scatter plot can be completely divided by a single straight line (a 1D hyperplane in $\mathbb{R}^2$), the classes are formally **linearly separable**. A basic linear decision boundary (e.g., $w_1 X_1 + w_2 X_2 + b = 0$) perfectly discriminates between the classes without requiring kernel tricks or non-linear models.
- **Option B (Incorrect):** Linearly separable classes do not imply invalid multicollinearity.
- **Option C (Incorrect):** The features clearly provide strong discriminatory information about class membership.
- **Option D (Incorrect):** Deep multi-layer networks are unnecessary when simple linear hyperplanes suffice.

</details>

---

### 86. Question

A researcher wants each categorical class in a scatter plot to have both a distinct color and a distinct marker shape to ensure accessibility for color-blind viewers and grayscale printing.

Which code structure achieves this cleanly?

- [ ] **A.**
  ```python
  plt.scatter(x, y, color_blind_mode=True)
  ```
- [ ] **B.**
  ```python
  plt.scatter(x, y, shape_and_color=classes)
  ```
- [ ] **C.**
  ```python
  plt.scatter(x, y, markers=["o", "s", "^"], colors=["blue", "orange", "green"])
  ```
- [ ] **D.**
  ```python
  styles = {
      "Setosa": ("#1f77b4", "o"),
      "Versicolor": ("#ff7f0e", "s"),
      "Virginica": ("#2ca02c", "^"),
  }

  for species, (color, marker) in styles.items():
    mask = df["species"] == species
    ax.scatter(
        df.loc[mask, "petal_length"],
        df.loc[mask, "petal_width"],
        c=color,
        marker=marker,
        label=species,
        s=60,
    )
  ax.legend()
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Parameter `color_blind_mode` does not exist in `plt.scatter()`.
- **Option B (Incorrect):** Parameter `shape_and_color` is not a recognized Matplotlib keyword argument.
- **Option C (Incorrect):** `plt.scatter()` cannot accept a list or sequence of marker symbols in a single call; passing a list to `marker` raises an error.
- **Option D (Correct):** Defining a mapping dictionary that couples each categorical label to an explicit color and geometric marker symbol (`styles = {label: (color, marker)}`) and looping through groups is the robust, production-grade pattern for multi-attribute discrete scatter plotting. It guarantees accessible, dual-encoded visual differentiation and builds a clean legend.

</details>

---

### 87. Question

A data scientist is inspecting a scatter plot matrix (pairplot) of housing features. The scatter plot between `square_footage` ($X_1$) and `number_of_rooms` ($X_2$) displays an extremely tight diagonal line with Pearson correlation $r = 0.96$.

When fitting a multiple linear regression model predicting `house_price` using both features, what statistical modeling problem does this tight scatter alignment signal?

- [ ] **A.** Underfitting due to high bias
- [ ] **B.** Severe multicollinearity, which inflates the variance of regression coefficient estimates and makes individual feature weights unstable and hard to interpret
- [ ] **C.** Zero variance in the target variable
- [ ] **D.** Non-invertibility of the target loss function

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Underfitting describes poor overall predictive capacity; multicollinearity is a relationship between predictors.
- **Option B (Correct):** When two predictor features form an almost deterministic linear line on a scatter plot ($r \approx 1.0$), they carry redundant information. In multiple linear regression, this **multicollinearity** inflates standard errors of the regression coefficients (high Variance Inflation Factor, VIF), causing estimates to fluctuate wildly and rendering individual p-values unreliable.
- **Option C (Incorrect):** High correlation between predictors does not mean target variance is zero.
- **Option D (Incorrect):** Multicollinearity affects parameter variance and conditioning, not loss function invertibility.

</details>

---

### 88. Question

After running a K-Means clustering algorithm ($K=3$) on customer purchase data, a data scientist wants to visualize the resulting clusters and overlay the 3 learned cluster centroids as large black 'X' marks.

Which snippet completes this visualization correctly?

- [ ] **A.**
  ```python
  plt.scatter(
      x, y, c=cluster_labels, cmap="viridis", alpha=0.6
  )  # Points colored by cluster
  plt.scatter(
      centroids[:, 0],
      centroids[:, 1],
      c="black",
      s=250,
      marker="X",
      linewidths=2,
      label="Centroids",
  )
  plt.legend()
  plt.show()
```
- [ ] **B.**
  ```python
  plt.scatter(x, y, kmeans=centroids)
  ```
- [ ] **C.**
  ```python
  plt.plot_clusters(x, y, cluster_labels, centroids)
  ```
- [ ] **D.**
  ```python
  plt.scatter(centroids, type="center")
  ```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** This is the canonical workflow for cluster visualization:
  1. Plot the customer observations colored by their discrete cluster assignments using `c=cluster_labels` and a distinct colormap.
  2. Overlay the centroid coordinates (`centroids[:, 0]`, `centroids[:, 1]`) in a secondary `plt.scatter()` call with high visual contrast (large size `s=250`, distinct color `c='black'`, unique marker `marker='X'`).
- **Option B (Incorrect):** `kmeans` is not a valid parameter in `plt.scatter()`.
- **Option C (Incorrect):** `plt.plot_clusters` is not a Matplotlib plotting method.
- **Option D (Incorrect):** Keyword argument `type='center'` does not exist in `plt.scatter()`.

</details>

---

### 89. Question

Which of the following statements regarding scatter plot diagnostics in exploratory data analysis (EDA) are **correct**?

1. A random spherical cloud of scatter points centered at $(0, 0)$ indicates that linear correlation $r \approx 0$, but does not prove that the variables are strictly statistically independent (they could have non-linear dependence).
2. Heteroscedasticity can often be stabilized by applying variance-stabilizing transformations (such as $\log(y)$ or Box-Cox) to the dependent variable.
3. Anscombe's Quartet demonstrates that four distinct datasets with identical linear regression lines and Pearson correlation $r$ can look radically different on scatter plots.
4. Overlapping cluster points can be separated visually by applying t-SNE or UMAP dimensionality reduction before scatter plotting.

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** While Statements 1 and 3 are true, this option omits valid Statements 2 (variance-stabilizing transforms) and 4 (manifold learning separation).
- **Option B (Incorrect):** While Statements 2 and 4 are true, this option omits valid Statements 1 and 3.
- **Option C (Incorrect):** While Statements 1, 2, and 3 are true, this option fails to include valid Statement 4.
- **Option D (Correct):** Statements 1, 2, 3, and 4 are all correct foundational principles: Statement 1 correctly notes that linear $r \approx 0$ does not rule out non-linear dependence; Statement 2 is valid because log/Box-Cox transforms stabilize variance; Statement 3 is valid because Anscombe's quartet proves the necessity of scatter plotting; and Statement 4 is valid because t-SNE/UMAP project high-dimensional clusters into distinct 2D groupings.

</details>

---

### 90. Question

In linear regression diagnostics, what is a **residual scatter plot** ($\text{Residuals } e_i \text{ vs. Fitted Values } \hat{y}_i$), and what does an ideal homoscedastic linear model look like on this plot?

- [ ] **A.** It plots residuals $e_i = y_i - \hat{y}_i$ on the vertical axis against predicted $\hat{y}_i$ on the horizontal axis; an ideal model displays a random, structureless horizontal scatter cloud centered around the horizontal zero line ($e = 0$) with constant vertical spread.
- [ ] **B.** It plots $y$ versus $x$; an ideal model forms a 45-degree line passing through $(0, 0)$.
- [ ] **C.** It plots slope $m$ versus intercept $b$; an ideal model forms a circle.
- [ ] **D.** It plots the training loss against epoch iterations.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** A **residual plot** plots the regression errors ($e_i = y_i - \hat{y}_i$) on the y-axis against the fitted/predicted values $\hat{y}_i$ (or predictor $x_i$) on the x-axis. In a well-specified, homoscedastic linear model, the residuals should be completely random noise: evenly dispersed around the reference line $e = 0$ with no curvature (which would indicate non-linearity) and no funnel shapes (which would indicate heteroscedasticity).
- **Option B (Incorrect):** Plotting $y$ vs $x$ is the raw data scatter plot, not a residual plot.
- **Option C (Incorrect):** Plotting parameter estimates is parameter space, not residual analysis.
- **Option D (Incorrect):** Plotting loss against epochs is a training loss curve.

</details>

---
