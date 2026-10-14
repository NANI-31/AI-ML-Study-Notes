# Multicollinearity Objective Test — Set 07

---

### 1. Question

A data analyst is modeling hospital patient outcomes using two highly collinear features:
* `systolic_blood_pressure` (correlation with diastolic $= 0.88$)
* `diastolic_blood_pressure` (correlation with systolic $= 0.88$)

Instead of dropping one of these critical clinical measures, what domain-informed feature engineering technique can the analyst use to combine them into a single, clinically standard, non-collinear feature?

- [ ] A) Multiply them together: `systolic * diastolic`.
- [ ] B) Compute Mean Arterial Pressure (MAP) or Pulse Pressure: $\text{Pulse Pressure} = \text{systolic} - \text{diastolic}$.
- [ ] C) One-hot encode both continuous pressure readings into 200 dummy columns.
- [ ] D) Invert both features by taking $\frac{1}{\text{systolic}}$ and $\frac{1}{\text{diastolic}}$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Multiplying them creates an uncentered interaction term that remains strongly correlated with both individual measurements.
- **Option B (Correct):** In medicine, Mean Arterial Pressure ($\text{MAP} \approx \text{diastolic} + \frac{1}{3}(\text{systolic} - \text{diastolic})$) or Pulse Pressure ($\text{systolic} - \text{diastolic}$) are standard, well-validated clinical indices. Combining them into an established domain metric preserves the clinical information from both readings while collapsing two collinear columns into one stable, interpretable predictor.
- **Option C (Incorrect):** One-hot encoding continuous variables causes severe dimensionality explosion and does not solve collinearity.
- **Option D (Incorrect):** Taking reciprocals does not eliminate the strong monotonic correlation between the two readings.

</details>

---

### 2. Question

In an e-commerce sales model, an engineer finds that `total_items_purchased` and `total_order_amount_dollars` have a Pearson correlation of $0.93$.

Which engineered feature effectively breaks this collinearity while retaining valuable customer behavioral insights?

- [ ] A) `average_item_price = total_order_amount_dollars / total_items_purchased`
- [ ] B) `order_sum = total_order_amount_dollars + total_items_purchased`
- [ ] C) `item_squared = total_items_purchased ** 2`
- [ ] D) `item_log = np.log(total_items_purchased)`

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `total_items_purchased` measures cart volume, while `total_order_amount_dollars` conflates cart volume with item expensiveness. Dividing the two yields `average_item_price`, which measures a distinct behavioral dimension: is the buyer purchasing luxury goods or budget commodities? Using `total_items_purchased` alongside `average_item_price` captures both quantity and quality with minimal collinearity.
- **Option B (Incorrect):** Adding them adds apples to oranges (items + dollars) and produces a feature that remains $90\%+$ correlated with both original components.
- **Option C (Incorrect):** Squaring `total_items_purchased` creates a polynomial term that is strongly correlated with the original count.
- **Option D (Incorrect):** Log-transforming one variable preserves rank order and does not break collinearity with total spend.

</details>

---

### 3. Question

A real estate dataset contains:
* `total_rooms`
* `number_of_bedrooms`

Both have VIF scores exceeding $20.0$. An analyst creates:
`non_bedroom_rooms = total_rooms - number_of_bedrooms`

What does this feature combination accomplish?

- [ ] A) It creates negative infinity values across all rows.
- [ ] B) It isolates common living areas (e.g., kitchens, living rooms, dining rooms) from sleeping areas, reducing mutual redundancy and lowering VIF scores.
- [ ] C) It completely deletes `total_rooms` from computer memory.
- [ ] D) It transforms the target variable into a classification label.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Since `total_rooms` is always $\ge \text{number_of_bedrooms}$, the difference is always non-negative ($\ge 0$).
- **Option B (Correct):** By subtracting bedrooms from total rooms, the analyst partitions house space into two distinct, interpretable functional zones: sleeping space (`number_of_bedrooms`) and shared living space (`non_bedroom_rooms`). This separation drastically reduces the collinearity between predictors while retaining the full structural information of the house.
- **Option C (Incorrect):** The original columns remain in the DataFrame unless explicitly dropped.
- **Option D (Incorrect):** Transforming input features does not modify the target variable $y$.

</details>

---

### 4. Question

An educational researcher collects student test scores across three exams: `exam_1`, `exam_2`, and `exam_3`. All three scores have mutual correlations $> 0.85$.

If the researcher wants to measure overall academic capability without collinearity destabilizing the regression model, which approach is most appropriate?

- [ ] A) Keep all three exam scores and ignore the condition number warning.
- [ ] B) Create an unweighted or weighted average: `composite_academic_score = (exam_1 + exam_2 + exam_3) / 3`.
- [ ] C) Drop `exam_2` and `exam_3` and claim that only `exam_1` measures academic achievement.
- [ ] D) Multiply all three scores together.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Ignoring severe collinearity leaves the model vulnerable to coefficient instability and misleading hypothesis tests.
- **Option B (Correct):** When multiple features are repeated measurements of the same underlying latent construct (e.g., student ability), averaging them into a composite index reduces measurement noise and eliminates multicollinearity, producing a single, stable, highly reliable predictor.
- **Option C (Incorrect):** Arbitrarily choosing one exam discards performance variance across different exam formats or testing dates.
- **Option D (Incorrect):** Multiplying percentages or scores produces an exponential metric with extreme skewness.

</details>

---

### 5. Question

When combining two collinear features via a ratio ($z = x_1 / x_2$), what critical data quality check MUST be performed first?

- [ ] A) Confirm that all values in $x_1$ are prime numbers.
- [ ] B) Verify that the denominator $x_2$ does not contain zeros or near-zero values that would cause division-by-zero errors or explosive outliers.
- [ ] C) Verify that the correlation between $x_1$ and $y$ is negative.
- [ ] D) Ensure that $x_1$ and $x_2$ have identical standard deviations.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Prime numbers are irrelevant in continuous feature engineering.
- **Option B (Correct):** Any ratio feature $x_1 / x_2$ risks evaluating to `inf` or `NaN` whenever $x_2 = 0$. Even values close to zero can create extreme, heavy-tailed outliers that distort regression slopes. Practitioners must either filter out zeros, add a small constant (epsilon), or verify that $x_2$ is strictly positive.
- **Option C (Incorrect):** The sign of the target correlation does not prevent division.
- **Option D (Incorrect):** Different standard deviations do not affect the mathematical validity of a ratio.

</details>

---

### 6. Question

An HR analytics team has `base_salary`, `bonus`, and `equity_grant_value` predicting employee tenure. All three compensation metrics are mutually collinear ($r > 0.80$).

Which combination strategy preserves domain meaning while eliminating collinearity?

- [ ] A) Replace all three with `total_compensation = base_salary + bonus + equity_grant_value`.
- [ ] B) Replace tenure with a random number generator.
- [ ] C) Drop `base_salary` because base pay is never relevant to employee retention.
- [ ] D) Multiply `bonus` by $-1$.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In total rewards analysis, employees evaluate their overall earnings package. Aggregating the three compensation streams into `total_compensation` reflects the true economic construct of interest, eliminates collinearity among the three components, and yields a clean, interpretable coefficient.
- **Option B (Incorrect):** Replacing the target with random numbers destroys all predictive utility.
- **Option C (Incorrect):** Base salary is often the single largest and most stable component of compensation; dropping it would be ungrounded.
- **Option D (Incorrect):** Multiplying by $-1$ flips the sign but keeps the exact same magnitude of correlation.

</details>

---

### 7. Question

Why does computing an index or composite score from multiple collinear survey questions (e.g., Likert scale questions measuring "customer satisfaction") often yield a BETTER predictor than selecting any single question?

- [ ] A) Because composite indices reduce random measurement error (noise) through averaging, increasing overall reliability (Cronbach's alpha).
- [ ] B) Because composite scores automatically guarantee that the regression $R^2$ will equal 1.0.
- [ ] C) Because individual survey questions are not compatible with pandas.
- [ ] D) Because averaging questions converts categorical responses into complex numbers.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Classical test theory establishes that any single survey item contains substantial random noise. When multiple items probe the same latent trait (satisfaction), their common variance reflects true satisfaction, while their idiosyncrasies cancel out upon averaging. This creates a composite metric with higher reliability, stronger correlation with the target, and zero multicollinearity issues.
- **Option B (Incorrect):** Composite indices improve reliability, but real-world $R^2$ values remain bounded below 1.0.
- **Option C (Incorrect):** Pandas easily handles individual survey items as integers or floats.
- **Option D (Incorrect):** Averaging produces real numbers, not complex numbers.

</details>

---

### 8. Question

A data analyst is working with warehouse logistics data containing:
* `box_length`
* `box_width`
* `box_height`

All three dimensions are mutually correlated ($r \approx 0.82$) because larger boxes tend to expand proportionally in all dimensions.

What is the most natural geometric combination feature to replace these three variables in a shipping cost regression model?

- [ ] A) `box_diagonal = box_length - box_width`
- [ ] B) `box_volume = box_length * box_width * box_height`
- [ ] C) `box_ratio = box_length / box_length`
- [ ] D) `box_count = box_length ** 0`

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Subtracting width from length does not capture box capacity or shipping space.
- **Option B (Correct):** Shipping costs and courier pricing are directly determined by package volume (dimensional weight). Multiplying length $\times$ width $\times$ height directly models the physical space occupied by the parcel, condensing three collinear linear dimensions into a single, highly relevant, non-collinear physical metric.
- **Option C (Incorrect):** Any non-zero variable divided by itself equals a constant $1.0$, which provides zero variance.
- **Option D (Incorrect):** Any number raised to power $0$ equals $1.0$, producing an uninformative constant column.

</details>

---

### 9. Question

Suppose an analyst replaces two collinear features $x_1$ and $x_2$ with their difference: $d = x_1 - x_2$. 

Under what condition will the new feature $d$ have virtually zero correlation with $x_1$?

- [ ] A) When $x_1$ and $x_2$ have approximately equal variance and are positively correlated with $r \approx \frac{\text{Var}(x_1)}{\text{Var}(x_1) + \text{Var}(x_2)}$.
- [ ] B) Only when both $x_1$ and $x_2$ are negative.
- [ ] C) When $x_1$ is completely uncorrelated with $x_2$.
- [ ] D) It is mathematically impossible for a difference to have lower correlation than the original features.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** From the covariance formula, $\text{Cov}(x_1, x_1 - x_2) = \text{Var}(x_1) - \text{Cov}(x_1, x_2)$. For this covariance to equal zero, we need $\text{Cov}(x_1, x_2) \approx \text{Var}(x_1)$. When $x_1$ and $x_2$ have equal variances and high correlation, the difference $d$ captures only the orthogonal discrepancy between them, decoupling the difference from the level of $x_1$.
- **Option B (Incorrect):** Signs of the values do not determine the covariance of differences.
- **Option C (Incorrect):** If $x_1$ and $x_2$ are uncorrelated, $\text{Cov}(x_1, x_1 - x_2) = \text{Var}(x_1) > 0$.
- **Option D (Incorrect):** Differences between positively correlated variables routinely decouple from the base levels; this is the core principle behind contrast coding.

</details>

---

### 10. Question

What is the primary advantage of domain-driven feature combination over automated blind techniques like PCA?

- [ ] A) Domain-driven features preserve clear, human-understandable business interpretations, whereas PCA creates abstract mathematical linear combinations that are difficult to explain to non-technical stakeholders.
- [ ] B) Domain-driven features guarantee an $R^2$ of exactly 1.0.
- [ ] C) Domain-driven features are calculated without writing any Python code.
- [ ] D) PCA can only be applied to datasets with fewer than 5 rows.

<details>
<summary>View Answer & Explanation</summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** An engineered feature like `debt_to_income_ratio` or `price_per_square_foot` carries direct, actionable business intuition. When a model's coefficient changes on `price_per_square_foot`, business leaders immediately understand the impact. In contrast, explaining that a house price dropped due to a shift in `Principal_Component_4` is opaque and impractical in regulated or executive settings.
- **Option B (Incorrect):** No feature engineering technique guarantees perfect $R^2$.
- **Option C (Incorrect):** Feature combination requires writing transformation code in pandas or scikit-learn.
- **Option D (Incorrect):** PCA is routinely used on massive datasets with millions of rows.

</details>
