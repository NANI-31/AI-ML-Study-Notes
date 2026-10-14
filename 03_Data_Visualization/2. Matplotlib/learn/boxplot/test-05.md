# Matplotlib Box Plot Objective Test — Set 05

---

### 41. Question

A data scientist creates a comparative box plot for cross-validation scores of three classifiers over 10 folds:

```python
import matplotlib.pyplot as plt
import numpy as np

lr_cv   = [0.81, 0.83, 0.82, 0.84, 0.80, 0.83, 0.82, 0.83, 0.81, 0.82]
svm_cv  = [0.78, 0.92, 0.84, 0.71, 0.95, 0.76, 0.88, 0.90, 0.73, 0.86]
xgb_cv  = [0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.90, 0.91, 0.89, 0.90]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([lr_cv, svm_cv, xgb_cv],
                tick_labels=["Logistic Reg", "SVM", "XGBoost"],
                patch_artist=True,
                notch=False)
ax.set_ylabel("CV Accuracy")
ax.set_title("10-Fold Cross-Validation Accuracy")
```

Based on the data alone, which classifier shows the **highest variance** across CV folds?

- [ ] **A.** Logistic Regression — its scores range from 0.80 to 0.84
- [ ] **B.** SVM — its scores range from 0.71 to 0.95, with a spread of 0.24
- [ ] **C.** XGBoost — its scores are consistently around 0.90–0.92
- [ ] **D.** All three have equal variance because they all have 10 folds

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Logistic Regression has a very narrow range (0.80–0.84), indicating low variance. Its IQR is approximately 0.82–0.83 = 0.01, producing a very short box — a stable but moderately performing model.
- **Option B (Correct):** SVM has the highest variance: scores fluctuate from 0.71 to 0.95 — a range of 0.24. This wide range produces the tallest box among the three, with potentially long whiskers. High cross-validation variance suggests the SVM is sensitive to which data points are in the training vs. validation folds — a sign of instability.
- **Option C (Incorrect):** XGBoost scores are tightly clustered (0.89–0.92), making its box very narrow. This indicates HIGH stability (low variance) combined with HIGH performance — the ideal combination for a production model.
- **Option D (Incorrect):** Having the same number of folds (10) does not make variance equal. Variance is a property of the SPREAD of values, not the count. The three classifiers show dramatically different spreads.

</details>

---

### 42. Question

A developer creates a box plot of employee salaries and finds the result looks unexpected:

```python
salaries = [[45000, 52000, 48000, 51000, 55000],
            [62000, 68000, 71000, 65000, 74000]]

fig, ax = plt.subplots()
ax.boxplot(salaries, tick_labels=["Junior", "Senior"])
```

A colleague suggests the code has an issue. What is it?

- [ ] **A.** Salaries are too large to be passed as raw values — they should be divided by 1000
- [ ] **B.** `tick_labels` must have the same number of elements as there are data points (5), not groups (2)
- [ ] **C.** No issue — this correctly creates two side-by-side box plots for Junior and Senior salaries
- [ ] **D.** The code fails because `salaries` must be a NumPy array, not a list of lists

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib handles large numbers without issues. Values like 45000–74000 render correctly on the y-axis. You may optionally divide by 1000 for cleaner axis labels, but it's not required and the code doesn't fail.
- **Option B (Incorrect):** `tick_labels` must match the number of **groups** (2), not data points per group (5). Here, `tick_labels=["Junior", "Senior"]` correctly has 2 elements matching 2 groups. If it had 5 elements, THAT would be an error.
- **Option C (Correct):** The code is syntactically and semantically correct. `ax.boxplot([[group1], [group2]])` accepts a list of lists and creates one box per inner list. `tick_labels=["Junior", "Senior"]` correctly labels the two groups. The result is two side-by-side vertical box plots comparing Junior and Senior salary distributions.
- **Option D (Incorrect):** `ax.boxplot()` accepts Python lists, lists of lists, NumPy arrays, and Pandas Series — not just NumPy arrays. Lists of lists are explicitly supported and commonly used for multi-group box plots.

</details>

---

### 43. Question

A machine learning practitioner evaluates residuals from three regression models:

```
Model 1 (Linear): Compact box centered near 0, short equal whiskers
Model 2 (Polynomial): Box centered near 0, very long upper whisker, several high outliers
Model 3 (Neural Net): Box centered near −5, symmetric whiskers, no outliers
```

From the box plot descriptions, which model has the **best residual distribution** for a well-fitted regression model?

- [ ] **A.** Model 2 — the long upper whisker shows it can capture extreme patterns
- [ ] **B.** Model 3 — no outliers indicates perfect fit
- [ ] **C.** Model 1 — compact box centered near 0 with short symmetric whiskers indicates consistently small, unbiased errors
- [ ] **D.** All three are equally good because they all have the same median position (near 0)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Long upper whiskers and high outliers indicate the model makes occasional **large positive errors** — it significantly overestimates certain data points. For regression, we want small, symmetric errors. Large outlier residuals suggest poor generalization.
- **Option B (Incorrect):** Model 3 has NO outliers, but its box is centered near **−5** — a systematic bias. A consistent negative residual means the model systematically **underestimates** the target values. Even without outliers, a biased model is problematic. A well-fitted model should have residuals centered at or very near 0.
- **Option C (Correct):** The ideal residual distribution for a well-fitted regression model is: (1) **centered near 0** (unbiased — no systematic over- or underestimation), (2) **compact box** (low IQR — errors are consistently small), (3) **short symmetric whiskers** (no extreme tails — errors are bounded and symmetric). Model 1 satisfies all three criteria.
- **Option D (Incorrect):** Model 3 is centered near −5, NOT near 0. This eliminates the claim that all three have the same median position. Model 3's systematic bias makes it inferior to Model 1.

</details>

---

### 44. Question

A developer wants to colour each box in a multi-group box plot with different colours using `patch_artist=True`. Which of the following code segments correctly achieves this?

```python
bp = ax.boxplot([group1, group2, group3, group4],
                patch_artist=True)
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
```

- [ ] **A.** `bp["boxes"].set_facecolor(colors)` — pass the list directly
- [ ] **B.** `for patch, color in zip(bp["boxes"], colors): patch.set_facecolor(color)` — iterate and set individually
- [ ] **C.** `ax.boxplot(..., facecolors=colors)` — pass colours at creation time
- [ ] **D.** `bp["boxes"].facecolor = colors` — assign directly to the attribute

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `bp["boxes"]` is a Python list, not a NumPy array or single artist. You cannot call `.set_facecolor()` on a list. Each element (`PathPatch`) must be styled individually.
- **Option B (Correct):** `zip(bp["boxes"], colors)` pairs each `PathPatch` box with its corresponding colour. `patch.set_facecolor(color)` applies the fill colour to each individual box. This is the standard and idiomatic Matplotlib pattern for per-group colouring of box plots. It works because `patch_artist=True` converts boxes to `PathPatch` objects that support `set_facecolor()`.
- **Option C (Incorrect):** `ax.boxplot()` does not have a `facecolors=` parameter. You cannot pass a colour list at creation time — colouring must be done after creation by accessing `bp["boxes"]`.
- **Option D (Incorrect):** Matplotlib artist attributes like `facecolor` are not directly assignable as Python object attributes using `=`. You must call the setter method `set_facecolor()`. Direct attribute assignment (`artist.facecolor = value`) does not propagate to the rendering pipeline.

</details>

---

### 45. Question

A developer creates a box plot where the dataset has a **very large IQR relative to the median**:

```
Q₁ = 10, Median = 35, Q₃ = 90
IQR = 80
Lower fence = 10 − 1.5×80 = 10 − 120 = −110
Upper fence = 90 + 1.5×80 = 90 + 120 = 210
```

A data point of value **−50** exists in the dataset. Is it an outlier?

- [ ] **A.** Yes — −50 is below Q₁ so it must be an outlier
- [ ] **B.** Yes — −50 is below the median so it is an outlier
- [ ] **C.** No — −50 is above the lower fence of −110, so it falls within the whisker range and is NOT an outlier
- [ ] **D.** Yes — negative values are always plotted as outliers in box plots

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Being below Q₁ does not make a value an outlier. Values between the lower fence (Q₁−1.5×IQR) and Q₁ are part of the lower whisker range — they are included within the normal spread of data.
- **Option B (Incorrect):** Being below the median is completely normal — by definition, 50% of data points are below the median. Median position has nothing to do with outlier classification.
- **Option C (Correct):** The outlier rule is strictly: a point is an outlier if it falls **below Q₁−1.5×IQR** or **above Q₃+1.5×IQR**. Lower fence = −110. Since −50 > −110, the value −50 is **within the fence** and is NOT an outlier. It is part of the lower whisker range. The lower whisker would extend to the actual minimum within the fence, which could include −50. This example highlights that a large IQR produces wide fences that accept many seemingly extreme values as non-outliers.
- **Option D (Incorrect):** The sign (positive or negative) of a value is irrelevant to outlier classification. What matters is whether the value falls outside the 1.5×IQR fence boundaries, which are computed relative to Q₁ and Q₃.

</details>

---

### 46. Question

A developer wants to plot box plots for **6 features** of a dataset to inspect their distributions side-by-side. They use a 2D NumPy array where each column is a feature:

```python
import numpy as np

data_matrix = np.random.randn(200, 6)  # 200 samples, 6 features

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot(data_matrix,
           tick_labels=[f"Feature {i+1}" for i in range(6)])
```

How does Matplotlib interpret the 2D array `data_matrix` when passed to `boxplot()`?

- [ ] **A.** It creates a single box plot using all 1200 values (200×6) as one distribution
- [ ] **B.** It creates 6 box plots, one per column — each column's 200 values form one box
- [ ] **C.** It creates 200 box plots, one per row — each row's 6 values form one box
- [ ] **D.** An error is raised because `boxplot` only accepts 1D arrays

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does NOT flatten 2D arrays into a single distribution. Each column is treated as a separate dataset for a separate box.
- **Option B (Correct):** When a 2D NumPy array is passed to `ax.boxplot()`, Matplotlib interprets each **column** as a separate dataset. A (200×6) array creates **6 box plots**, each computed from the 200 values in that column. This makes 2D array input a convenient way to visualise multiple feature distributions simultaneously without manually extracting columns.
- **Option C (Incorrect):** Box plots are computed per column, not per row. Creating 200 box plots from rows would generate a very different result (each box with only 6 values) and would require transposing the array: `ax.boxplot(data_matrix.T)`.
- **Option D (Incorrect):** `ax.boxplot()` explicitly supports 2D arrays (column-wise), lists of lists, and lists of 1D arrays. It is not limited to 1D arrays.

</details>

---

### 47. Question

A developer creates a box plot comparison of **house price residuals** from two models:

```python
lasso_resid = [-12000, -8000, -5000, -2000, 1000, 3000, 5000, 8000, 11000, 50000]
ridge_resid = [-9000,  -6000, -3000, -1000, 500,  2000, 4000, 7000, 10000, 12000]
```

For Lasso: Q₁≈−5500, Q₃≈7000, IQR≈12500, Upper fence = 7000+1.5×12500 = 25750. Is 50000 an outlier for Lasso?

- [ ] **A.** No — 50000 is the maximum observed value and maximum is always a whisker endpoint
- [ ] **B.** Yes — 50000 > 25750 (upper fence), so it is plotted as an individual flier marker above the upper whisker
- [ ] **C.** No — outliers are only identified when the value exceeds 3×IQR above Q₃
- [ ] **D.** It depends on the `whis` parameter — the default is `whis=1.5` but it can be changed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The maximum value is NOT always a whisker endpoint. If the maximum exceeds the upper fence (Q₃+1.5×IQR), it becomes a **flier marker** and the whisker stops at the last non-outlier value. Only when the maximum is within the fence does it serve as the whisker endpoint.
- **Option B (Correct):** Upper fence for Lasso = 25750. Since 50000 > 25750, the value 50000 is **beyond the fence** and is classified as an outlier (flier). Matplotlib plots it as a separate marker above the upper whisker cap. The upper whisker extends to the largest non-outlier value (11000, since 11000 ≤ 25750).
- **Option C (Incorrect):** The default outlier rule uses 1.5×IQR fences (not 3×IQR). The 3×IQR threshold is used for the more extreme "far outlier" classification. By default (`whis=1.5`), the standard fence at 1.5×IQR is applied.
- **Option D (Partially correct but misleading):** While D correctly identifies that `whis` is configurable, Option B is still the most direct and correct answer to the specific question asked. The question specifies the default `whis=1.5` context implicitly. The key point is that with the given Q₁, Q₃, and IQR values, 50000 IS an outlier under default settings.

</details>

---

### 48. Question

A developer uses `whis=2.0` instead of the default `whis=1.5` in a box plot:

```python
ax.boxplot(data, whis=2.0)
```

What effect does changing `whis=2.0` have?

- [ ] **A.** The box becomes 2× wider
- [ ] **B.** The whisker fences extend to Q₁−2.0×IQR and Q₃+2.0×IQR — fewer data points are classified as outliers because the fences are wider
- [ ] **C.** The whiskers extend to exactly 2 units above and below the box
- [ ] **D.** The number of boxes doubles to show 2 distributions per group

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `whis` controls the whisker extension multiplier, not box width. Box width is controlled by `widths` parameter.
- **Option B (Correct):** `whis` is the multiplier for IQR in the whisker calculation. With `whis=2.0`: Lower fence = Q₁ − 2.0×IQR; Upper fence = Q₃ + 2.0×IQR. Compared to the default `whis=1.5`, the fences are extended further from the quartiles. This means **more data points fall within the fences** (fewer are classified as outliers). The whiskers are longer, and fewer flier markers appear. `whis=2.0` is less sensitive to outliers than `whis=1.5`.
- **Option C (Incorrect):** `whis=2.0` means 2.0×IQR, not 2 absolute data units. The fence distance depends on the IQR of the specific dataset, not a fixed unit value.
- **Option D (Incorrect):** `whis` has no effect on the number of boxes. It only modifies the fence calculation for whisker/outlier determination.

</details>

---

### 49. Question

A developer creates a box plot of **test scores** and a colleague asks: "Why is the mean not visible?" The developer checks the code:

```python
scores = [55, 62, 68, 71, 75, 78, 82, 86, 89, 93]
ax.boxplot(scores)
```

What is the explanation for the mean not appearing?

- [ ] **A.** The mean equals the median in this dataset, so they overlap
- [ ] **B.** By default, `showmeans=False` — the mean is not displayed in a standard box plot unless explicitly requested
- [ ] **C.** The mean cannot be displayed in a box plot — it requires a bar chart
- [ ] **D.** `patch_artist=True` must be enabled before the mean can be shown

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Whether the mean equals the median depends on the data distribution. For this dataset: mean = (55+62+68+71+75+78+82+86+89+93)/10 = 759/10 = 75.9; median = (75+78)/2 = 76.5. They are not equal. Even if they were, they would overlap visually — not cause the mean to be absent.
- **Option B (Correct):** By default, `ax.boxplot()` uses `showmeans=False`. The mean is intentionally NOT shown by default because box plots are designed to display **median-based** (robust) statistics. The mean is sensitive to outliers and is considered a secondary statistic in box plot context. To show it, pass `showmeans=True` (for a point marker) or `showmeans=True, meanline=True` (for a line).
- **Option C (Incorrect):** Box plots CAN display the mean — using `showmeans=True`. The mean is simply hidden by default. It can be shown as a triangle marker or a dashed line.
- **Option D (Incorrect):** `patch_artist=True` is unrelated to mean display. `patch_artist` converts box outlines to filled patches. `showmeans=True` is the only parameter needed to display the mean, regardless of `patch_artist` state.

</details>

---

### 50. Question

A developer wants to use a box plot to **compare the distribution of a feature across three target classes** in a classification problem. The feature is "Age" and the classes are "Class 0", "Class 1", "Class 2". The developer writes:

```python
ages_class0 = df[df["target"] == 0]["age"].values
ages_class1 = df[df["target"] == 1]["age"].values
ages_class2 = df[df["target"] == 2]["age"].values

ax.boxplot([ages_class0, ages_class1, ages_class2],
           tick_labels=["Class 0", "Class 1", "Class 2"])
```

What insight does this box plot provide for **feature selection** in ML?

- [ ] **A.** The number of samples in each class
- [ ] **B.** Whether the "Age" feature separates the classes — non-overlapping boxes suggest age is a useful discriminating feature; heavily overlapping boxes suggest age alone provides little class separation
- [ ] **C.** The correlation coefficient between age and target
- [ ] **D.** The accuracy improvement when age is added to the model

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The number of samples per class is not directly readable from a box plot. While the whisker length and outlier count give hints about sample distribution, exact sample sizes require additional labels or a count bar chart.
- **Option B (Correct):** A box plot of a feature stratified by class is a key **exploratory data analysis (EDA)** technique for feature selection. If the three boxes are at distinctly different heights with minimal overlap (e.g., Class 0 ages=20–35, Class 1 ages=40–55, Class 2 ages=60–75), "Age" is a strong discriminating feature. If all three boxes heavily overlap (all spanning 25–65), "Age" alone poorly separates the classes and may have low predictive value. This visual inspection quickly guides feature selection before model training.
- **Option C (Incorrect):** Correlation coefficients are computed numerically (e.g., `np.corrcoef()`) and cannot be read from a box plot. Box plots show distribution shapes, centers, and spreads — not correlation measures.
- **Option D (Incorrect):** Model accuracy improvement from adding a feature requires actually training models with and without the feature. A box plot is a pre-modelling EDA tool that hints at separability but cannot quantify accuracy impact.

</details>
