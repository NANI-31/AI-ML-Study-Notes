### 1. Question

A product analyst is preparing a review of a linear-regression housing model. Which statements are correct?

### Options

- [x] **A)** Multicollinearity can coexist with high held-out performance. _(Correct)_
- [ ] **B)** MAE and MSE are always easier to interpret than `R²`. _(Incorrect)_
- [x] **C)** Correlated features can get arbitrary-looking coefficients. _(Correct)_
- [ ] **D)** High `R²` guarantees true causal coefficient effects. _(Incorrect)_

### Explanation

- **A is correct:** Multicollinearity affects parameter stability and individual feature interpretation, not necessarily the overall predictive accuracy on a held-out test set. A model can achieve a strong $R^2$ while having collinear features.
- **B is incorrect:** Raw MAE and MSE are scale-dependent and depend on the target variable's units, whereas $R^2$ provides a standardized, scale-independent measure of variance explained.
- **C is correct:** When features are highly correlated, linear regression can arbitrarily split weight among them (e.g., placing large weight on one and near-zero or opposite weight on another), making the coefficients unstable and arbitrary.
- **D is incorrect:** High correlation or predictive accuracy ($R^2$) does not establish causality; observational regression coefficients cannot be interpreted as true causal effects without rigorous causal inference conditions.

---

### 2. Question

A senior data analyst is debugging why one penalty can remove a feature while the other usually retains correlated features.

Which observations are correct?

### Options

- [ ] **A)** L1 adds the sum of squared weight values to the loss. _(Incorrect)_
- [x] **B)** L1 can drive some coefficients exactly to zero. _(Correct)_
- [ ] **C)** L2 usually drives all coefficients exactly to zero. _(Incorrect)_
- [x] **D)** L2 shrinks coefficient magnitudes toward zero. _(Correct)_

### Explanation

- **A is incorrect:** L1 regularization (Lasso) adds the sum of absolute weights ($\sum |\beta_i|$) to the loss function, not squared weights (which is used by L2).
- **B is correct:** Lasso's L1 penalty can shrink coefficients to exactly zero when the penalty parameter $\alpha$ is sufficiently large, effectively performing automated feature selection.
- **C is incorrect:** Ridge regression (L2 penalty) pushes coefficients asymptotically toward zero to prevent overfitting, but it rarely drives them exactly to zero.
- **D is correct:** Ridge's L2 penalty shrinks the magnitude of all coefficients toward zero while retaining correlated features.

---

### 3. Question

An ML lead is building a regularization experiment for a housing model. The dataset has several highly correlated features, and both predictive performance and coefficient stability matter.

Which choices are justified?

### Options

- [x] **A)** Compute a numeric-only correlation matrix. _(Correct)_
- [ ] **B)** Conclude negative `R²` is mathematically impossible. _(Incorrect)_
- [x] **C)** Compare held-out `R²` across alpha candidates. _(Correct)_
- [x] **D)** Refit on different samples to inspect coefficient swings. _(Correct)_
- [ ] **E)** Choose Lasso solely because it always beats Ridge. _(Incorrect)_

### Explanation

- **A is correct:** Correlation analysis helps identify pairs of variables that move together, highlighting potential multicollinearity among continuous numeric features.
- **B is incorrect:** $R^2$ can be negative on held-out or test data when the residual sum of squares exceeds the total sum of squares ($\text{SSE} / \text{SST} > 1$), indicating performance worse than predicting the baseline mean.
- **C is correct:** Hyperparameter tuning involves comparing validation or held-out $R^2$ across candidate values of $\alpha$ to select the best trade-off between bias and variance.
- **D is correct:** Refitting on different subsamples directly reveals coefficient variance and instability caused by multicollinear predictors.
- **E is incorrect:** Neither model is universally superior; Ridge often performs better when there are many correlated features with small effects, while Lasso excels when only a sparse subset of features is truly predictive.

---

### 4. Question

A data scientist is reviewing a housing dataset where `square_feet` and `num_rooms` rise together almost perfectly. A teammate says the individual coefficients can still be trusted because the model’s predictions are accurate.

What is the stronger concern?

### Options

- [ ] **A)** Overfitting is definitely present and severe throughout the data. _(Incorrect)_
- [x] **B)** Multicollinearity can make coefficient explanations unstable. _(Correct)_
- [ ] **C)** The target variable must be removed before fitting. _(Incorrect)_
- [ ] **D)** Predictive accuracy proves all coefficients are reliable. _(Incorrect)_

### Explanation

- **A is incorrect:** High correlation between predictors indicates collinearity, but correlation alone does not prove that the model is severely overfitting the dataset.
- **B is correct:** When predictor variables rise together almost perfectly, the model cannot reliably separate their individual contributions, resulting in high variance and unstable coefficient estimates.
- **C is incorrect:** The target variable is the outcome being modeled and is necessary for fitting the regression model.
- **D is incorrect:** High predictive accuracy on the overall outcome does not guarantee that individual feature coefficients are stable, reliable, or causally meaningful.

---

### 5. Question

For three observations, the actual values are 1, 2, and 3, while predictions are 2, 3, and 4. The target mean is 2.

Using the error form of $R^2$, what is the score?

### Options

- [ ] **A)** `1.0` _(Incorrect)_
- [ ] **B)** `0.5` _(Incorrect)_
- [x] **C)** `-0.5` _(Correct)_
- [ ] **D)** `-2.5` _(Incorrect)_

### Explanation

- **A is incorrect:** A score of 1.0 represents a perfect fit ($SSE = 0$), but each prediction is off by 1.
- **B is incorrect:** It has the right magnitude but the wrong sign. The prediction errors are larger than the total variance around the mean, driving $R^2$ into negative values.
- **C is correct:**
  - $\text{SSE} = (1 - 2)^2 + (2 - 3)^2 + (3 - 4)^2 = 1 + 1 + 1 = 3$
  - $\text{SST} = (1 - 2)^2 + (2 - 2)^2 + (3 - 2)^2 = 1 + 0 + 1 = 2$
  - $R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{3}{2} = -0.5$
- **D is incorrect:** It overstates the residual shortfall compared to the actual computed sums.

---

### 6. Question

A DevOps scientist is preparing recommendations after an experiment where plain linear regression had training $R^2$ near 0.999 and test $R^2$ near 0.997, but coefficients changed materially across repeated samples.

Which recommendations are correct?

### Options

- [ ] **A)** The close train-test scores alone prove that coefficients are stable. _(Incorrect)_
- [x] **B)** Regularization can trade a small increase in training error for better generalization. _(Correct)_
- [x] **C)** Ridge is a reasonable option when the goal is stability without discarding features. _(Correct)_
- [x] **D)** A reliable `alpha` comparison can be strengthened by using multiple validation splits. _(Correct)_
- [ ] **E)** The changing coefficients necessarily prove that a feature has no relationship with the target. _(Incorrect)_

### Explanation

- **A is incorrect:** High and consistent train-test predictive scores do not guarantee coefficient stability; multicollinearity can cause coefficients to fluctuate wildly across different samples even with excellent predictive accuracy.
- **B is correct:** Regularization adds a penalty on coefficient size to the loss function, accepting a slight increase in training bias/error in exchange for reduced variance and more stable generalization.
- **C is correct:** Ridge regression (L2 regularization) shrinks coefficient values toward zero to improve stability and curb variance, but unlike Lasso, it retains all features without driving weights to absolute zero.
- **D is correct:** Assessing regularization strength ($\alpha$) across multiple validation splits or through $k$-fold cross-validation provides a significantly more reliable estimate of generalization than a single split.
- **E is incorrect:** Fluctuating or unstable coefficients often stem from correlation between predictors (collinearity) rather than a lack of true relationship with the outcome variable.

---

### 7. Question

A junior analyst uses a working threshold of about 0.7 absolute correlation and finds that two housing features have an absolute correlation around 0.96.

Under the stated working threshold, what should this pattern signal?

### Options

- [ ] **A)** The two feature columns are entirely unrelated. _(Incorrect)_
- [ ] **B)** The signed correlation between them is negative. _(Incorrect)_
- [ ] **C)** The highly correlated feature pair should become the target. _(Incorrect)_
- [x] **D)** The strongly related pair deserves more attention. _(Correct)_

### Explanation

- **A is incorrect:** An absolute correlation value of 0.96 indicates an extremely strong linear relationship, not an absence of correlation.
- **B is incorrect:** Absolute correlation ($|r|$) measures magnitude only; it does not indicate whether the underlying signed correlation is positive or negative.
- **C is incorrect:** High correlation between predictor variables indicates collinearity among inputs; it does not turn those features into the target variable.
- **D is correct:** When the absolute correlation exceeds the established ~0.7 threshold, it signals strong collinearity between features that warrants closer inspection, potential feature selection, or regularization.

---

### 8. Question

A business analyst sees a housing model’s MSE of 1,099 but does not know the target’s scale.

Why is $R^2$ often easier to use for comparing these housing models?

### Options

- [x] **A)** It is a scale-independent variance proportion. _(Correct)_
- [ ] **B)** It converts each error into money. _(Incorrect)_
- [ ] **C)** It removes held-out evaluation needs. _(Incorrect)_
- [ ] **D)** It always lies between `0` and `1`. _(Incorrect)_

### Explanation

- **A is correct:** $R^2$ measures the proportion of target variance explained by the model. Because it is normalized and unitless (scale-independent), it allows direct comparison without needing to know the target variable's scale or units.
- **B is incorrect:** $R^2$ is a unitless ratio, not a conversion of error into monetary values.
- **C is incorrect:** Evaluating models on held-out or validation data remains essential to assess generalization and check for overfitting.
- **D is incorrect:** $R^2$ is not bounded strictly between 0 and 1; on held-out or test data, it can be negative if the model performs worse than predicting the simple mean.

---

### 9. Question

An ML engineer must stabilize coefficients in a housing model with severe multicollinearity, while keeping every feature available for interpretation rather than removing features outright.

Which estimator and behavior best match the goal?

### Options

- [ ] **A)** Lasso, because it adds an L1 penalty and selects features. _(Incorrect)_
- [x] **B)** Ridge, because it applies an L2 penalty and usually keeps features. _(Correct)_
- [ ] **C)** Plain linear regression, because it has no extra penalty term. _(Incorrect)_
- [ ] **D)** A much larger `alpha`, because it guarantees exact zero coefficients. _(Incorrect)_

### Explanation

- **A is incorrect:** While Lasso ($L_1$ penalty) is effective for feature selection, it drives some coefficients to exactly zero, which directly conflicts with the goal of keeping every feature available for interpretation.
- **B is correct:** Ridge regression applies an $L_2$ penalty that shrinks coefficient magnitudes toward zero to curb instability caused by multicollinearity while typically retaining all features in the model.
- **C is incorrect:** Plain linear regression (OLS) contains no regularization penalty and remains the unstable baseline susceptible to high variance when predictors are collinear.
- **D is incorrect:** Stronger shrinkage (a larger $\alpha$) in Ridge regression asymptotically pushes coefficients closer to zero, but it does not drive them to exact zero.

---

### 10. Question

A backend engineer tunes Lasso and observes that `alpha` is extremely large, all coefficients have become tiny, and held-out $R^2$ has fallen.

What is the most likely outcome?

### Options

- [ ] **A)** The model has learned a better feature ranking. _(Incorrect)_
- [ ] **B)** The model has eliminated all multicollinearity. _(Incorrect)_
- [x] **C)** The model is underfitting because shrinkage is too strong. _(Correct)_
- [ ] **D)** The model is overfitting despite very small coefficients. _(Incorrect)_

### Explanation

- **A is incorrect:** Over-shrinkage does not imply that the model has learned a better feature ranking.
- **B is incorrect:** Regularization penalizes coefficient sizes, but it does not alter or eliminate the underlying correlation between features in the dataset.
- **C is correct:** A very large $\alpha$ imposes excessive penalty on the weights, shrinking them so close to zero that the model loses the capacity to capture true patterns, resulting in severe underfitting and a drop in $R^2$.
- **D is incorrect:** Overfitting typically involves overly complex models with inflated variance; shrinking coefficients heavily reduces model complexity and leads to underfitting (high bias), not overfitting.
