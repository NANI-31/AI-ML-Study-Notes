# Multicollinearity Objective Test — Set 06

---

### 1. Question

A data scientist implements the following recursive feature pruning algorithm in Python:

```python
def prune_vif(X, thresh=5.0):
    cols = list(X.columns)
    while True:
        vifs = [variance_inflation_factor(X[cols].values, i) for i in range(len(cols))]
        max_vif = max(vifs)
        if max_vif > thresh:
            idx = vifs.index(max_vif)
            cols.pop(idx)
        else:
            break
    return X[cols]
```

Assuming `X` includes a constant column, what is the primary operational logic of this function?

- [ ] A) It drops all features with VIF $> 5.0$ simultaneously in a single pass.
- [ ] B) It iteratively drops the single feature with the highest VIF and recalculates all remaining VIFs until none exceed the threshold.
- [ ] C) It trains a Lasso regression model and extracts non-zero coefficients.
- [ ] D) It transforms the feature space into orthogonal principal components.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The `while True` loop removes one column at a time (`cols.pop(idx)`) per iteration and recomputes the VIF list in the subsequent loop pass.
- **Option B (Correct):** Iterative (greedy) backward elimination based on VIF is the standard heuristic for feature pruning. By dropping only the single worst offending feature per iteration, the remaining features often have their VIF scores dramatically reduced, preventing unnecessary loss of predictive columns.
- **Option C (Incorrect):** No Lasso model or regularization penalty is instantiated in this snippet.
- **Option D (Incorrect):** It prunes columns from the original DataFrame; it does not perform PCA or coordinate rotation.

</details>

---

### 2. Question

When deciding which of two highly collinear features to eliminate from a regression model, which criterion is generally the MOST sensible to use?

- [ ] A) Keep the feature with the alphabetically earlier column name.
- [ ] B) Keep the feature that has higher business interpretability, better measurement reliability, or a stronger correlation with the target variable.
- [ ] C) Always keep the feature with the higher numerical variance, regardless of what it measures.
- [ ] D) Keep whichever feature was collected most recently.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Alphabetical ordering is arbitrary and unrelated to data quality or predictive power.
- **Option B (Correct):** Practical feature selection requires domain-informed trade-offs:
  1. Measurement reliability (e.g., automated sensor data vs. noisy manual surveys).
  2. Bivariate correlation with the target $y$ (preserving more predictive signal).
  3. Actionability and business meaning (e.g., marketing can control `ad_spend`, but cannot directly control `organic_brand_score`).
- **Option C (Incorrect):** High variance can be an artifact of arbitrary scaling (e.g., measuring in millimeters vs. meters) and does not indicate superior information.
- **Option D (Incorrect):** Collection recency alone does not ensure higher data fidelity or stronger predictive relevance.

</details>

---

### 3. Question

An analyst must choose between keeping `house_sqft` (continuous, measured in square feet) or `room_count` (discrete integer count) in a model predicting house price. The two features have a correlation of $0.92$.

Why is `house_sqft` usually preferred over `room_count` in this modeling context?

- [ ] A) Because continuous measurements provide more granular, higher-resolution information about living space than discrete integer counts.
- [ ] B) Because discrete integer features always cause OLS models to produce biased intercept estimates.
- [ ] C) Because `house_sqft` is guaranteed to have zero correlation with any neighborhood characteristics.
- [ ] D) Because Scikit-Learn cannot fit linear regressions on integer features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** A 4-room house could be an 800 sq ft cramped apartment or a 3,000 sq ft luxury loft. Continuous square footage captures fine-grained physical variation that integer room counts collapse into coarse bins, retaining more accurate signal while resolving the collinearity between the two.
- **Option B (Incorrect):** Discrete count features do not violate Gauss-Markov assumptions or introduce intercept bias.
- **Option C (Incorrect):** Square footage often correlates with neighborhood wealth or lot size.
- **Option D (Incorrect):** Scikit-Learn handles integer inputs without difficulty by converting them to floating point arrays.

</details>

---

### 4. Question

What is a potential risk of using automated Stepwise Regression (Forward/Backward Selection) based solely on p-values to solve multicollinearity?

- [ ] A) It can flip arbitrary decisions based on slight random variations in the training sample, yielding unstable feature sets.
- [ ] B) It always selects all features regardless of p-value.
- [ ] C) It guarantees an $R^2$ of 1.0 on out-of-sample test data.
- [ ] D) It transforms all linear relationships into logarithmic curves.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Stepwise selection based on p-values is notoriously unstable in the presence of multicollinearity. When two features share variance, tiny random fluctuations in the sample can cause feature A to have $p = 0.049$ and feature B to have $p = 0.051$. The algorithm retains A and discards B, giving the illusion that A is vital and B is worthless, when in reality their roles could easily reverse on a different resample.
- **Option B (Incorrect):** Stepwise algorithms discard features whose p-values exceed the stopping criterion.
- **Option C (Incorrect):** Automated stepwise selection often causes severe overfitting, leading to worse out-of-sample performance.
- **Option D (Incorrect):** Stepwise regression selects among existing linear columns; it does not apply non-linear mathematical transformations.

</details>

---

### 5. Question

A machine learning engineer uses Recursive Feature Elimination (RFE) with an OLS estimator to reduce a feature set from 50 features to 15.

How does RFE decide which feature to eliminate at each step?

- [ ] A) It computes the correlation of each feature with the target and drops the lowest one.
- [ ] B) It fits the model on all current features, evaluates feature weights or importances, and removes the feature with the smallest absolute coefficient (or lowest ranking criterion).
- [ ] C) It generates a PCA plot and drops features in the lower quadrant.
- [ ] D) It calculates k-means clusters and removes outlier points.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** RFE is a wrapper method that fits the full predictive model, whereas univariate correlation with the target is a filter method.
- **Option B (Correct):** RFE trains the underlying estimator on the active feature set, ranks features according to their coefficients (`coef_`) or feature importances, removes the weakest feature(s), and repeats the process recursively until the target number of features is reached.
- **Option C (Incorrect):** RFE does not involve PCA projections.
- **Option D (Incorrect):** RFE selects feature columns; it does not cluster or remove observation rows.

</details>

---

### 6. Question

An analyst is trying to decide whether to drop `car_curb_weight` or `engine_displacement` from an automotive fuel efficiency regression model.

Both features have VIF $> 12$. The analyst runs two simple regressions:
* `MPG ~ car_curb_weight`: $R^2 = 0.72$
* `MPG ~ engine_displacement`: $R^2 = 0.61$

If only one feature can be retained to avoid collinearity, which one should the analyst retain, and why?

- [ ] A) `engine_displacement`, because having a lower $R^2$ reduces the danger of overfitting.
- [ ] B) `car_curb_weight`, because it individually explains a higher proportion of variance in the target ($72\%$ vs. $61\%$).
- [ ] C) Both should be dropped, leaving no predictors in the model.
- [ ] D) Whichever feature has the smaller mean value across all observations.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Intentionally choosing a substantially weaker predictor throws away useful signal.
- **Option B (Correct):** When two features are largely redundant with each other, selecting the one that displays a stronger standalone relationship with the target variable preserves the greatest predictive power for the final model while eliminating collinearity.
- **Option C (Incorrect):** Dropping both predictors discards all information regarding vehicle size and weight.
- **Option D (Incorrect):** The mean value of a feature is an arbitrary scale property and does not reflect predictive utility.

</details>

---

### 7. Question

What is a significant drawback of resolving multicollinearity purely by dropping features?

- [ ] A) The model's coefficients become complex imaginary numbers.
- [ ] B) You permanently discard any unique, non-redundant variance that the eliminated feature might have contributed to predicting the target.
- [ ] C) It causes Scikit-Learn models to run 100 times slower.
- [ ] D) It violates the assumption that residuals have zero mean.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Coefficients in linear regression remain real numbers.
- **Option B (Correct):** Even if two features have $r = 0.85$, that still leaves $(1 - 0.85^2) \approx 28\%$ of their variance unshared. Dropping one variable completely discards that unique $28\%$ of variance, which might have captured a critical nuance of the target phenomenon. This trade-off motivates feature combination or regularization instead of outright deletion.
- **Option C (Incorrect):** Dropping features reduces dimensionality and makes models train faster, not slower.
- **Option D (Incorrect):** OLS with an intercept guarantees that residuals sum to zero regardless of which subset of features is used.

</details>

---

### 8. Question

In a marketing attribution model, `google_search_spend` and `bing_search_spend` have a correlation of $0.94$. 

A stakeholder insists: *"We cannot drop Bing spend because our executive team needs to know if Bing campaigns are profitable."*

Why does keeping both features in an unregularized OLS model fail to satisfy the stakeholder's need?

- [ ] A) Because OLS will fail to run due to an operating system error.
- [ ] B) Because the severe multicollinearity will cause the model to estimate unstable, untrustworthy coefficients for both Google and Bing, rendering any profitability conclusions statistically invalid.
- [ ] C) Because Bing spend is always zero in marketing datasets.
- [ ] D) Because Google search spend will automatically be assigned an $R^2$ of zero.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** OLS will execute and return numbers, but the numbers will be unreliable.
- **Option B (Correct):** The stakeholder wants accurate attribution, but unregularized OLS cannot provide it when two channels move together. The model might estimate a large positive ROI for Google and a negative ROI for Bing simply because they fluctuated together. Presenting these unstable coefficients to executives will lead to erroneous business decisions.
- **Option C (Incorrect):** Bing represents a real marketing channel with positive spend.
- **Option D (Incorrect):** $R^2$ is an overall model or regression property, not a property of a single spend column.

</details>

---

### 9. Question

Which scikit-learn class is specifically designed to perform univariate feature selection by retaining the top $k$ features based on statistical tests with the target?

- [ ] A) `sklearn.decomposition.PCA`
- [ ] B) `sklearn.feature_selection.SelectKBest`
- [ ] C) `sklearn.preprocessing.StandardScaler`
- [ ] D) `sklearn.cluster.KMeans`

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** PCA is a dimensionality reduction technique that creates composite orthogonal features.
- **Option B (Correct):** `SelectKBest` scores features individually against the target variable using a statistical test (such as `f_regression` or `mutual_info_regression`) and retains the $k$ highest-scoring features.
- **Option C (Incorrect):** `StandardScaler` standardizes feature scales ($z$-scores) but does not select or drop features.
- **Option D (Incorrect):** `KMeans` is an unsupervised clustering algorithm.

</details>

---

### 10. Question

A data science team trains a regression model on a dataset with 8 features. Two features ($x_1$ and $x_2$) are collinear. 

When $x_2$ is removed from the model:
* Model $R^2$ drops from $0.851$ to $0.849$.
* The VIF of $x_1$ drops from $18.4$ to $1.2$.
* The p-value of $x_1$ drops from $0.42$ to $0.0002$.

What does this result demonstrate?

- [ ] A) Removing $x_2$ caused catastrophic loss of predictive power.
- [ ] B) Removing $x_2$ successfully stabilized $x_1$'s coefficient and restored its statistical significance with virtually zero loss in overall predictive accuracy.
- [ ] C) $x_1$ is an irrelevant feature that should also be removed.
- [ ] D) The model is now severely underfitting.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** An $R^2$ drop of $0.002$ ($0.851 \to 0.849$) is negligible.
- **Option B (Correct):** This is the ideal outcome of feature pruning. Almost all of $x_2$'s variance was redundant with $x_1$. Dropping $x_2$ eliminated variance inflation (VIF $18.4 \to 1.2$), contracted $x_1$'s standard error, allowed $x_1$ to emerge as a highly statistically significant driver ($p = 0.0002$), and left overall model explanatory power essentially intact.
- **Option C (Incorrect):** $x_1$ now has $p = 0.0002$, indicating it is a very strong and relevant predictor.
- **Option D (Incorrect):** An $R^2$ of $0.849$ is strong performance, not severe underfitting.

</details>
