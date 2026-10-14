### 91. Question

A machine learning engineer evaluates a binary classifier in scikit-learn but accidentally swaps the argument order, running:
```python
cm = confusion_matrix(y_pred, y_test)
```
instead of the standard signature `confusion_matrix(y_true, y_pred)`.

**What is the exact structural consequence of this argument swap on the returned confusion matrix?**

- [ ] A) The matrix is transposed: False Positives ($FP$) and False Negatives ($FN$) are swapped, while True Positives ($TP$) and True Negatives ($TN$) remain on the diagonal
- [ ] B) Scikit-learn raises a `ValueError` immediately because `y_pred` cannot be passed first
- [ ] C) The entire matrix is negated (all positive values become negative)
- [ ] D) All four quadrants are rotated 180 degrees

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The matrix is transposed: False Positives ($FP$) and False Negatives ($FN$) are swapped, while True Positives ($TP$) and True Negatives ($TN$) remain on the diagonal

**Explanation:**
- **A is correct:** Scikit-learn's `confusion_matrix` defines rows as the first argument ($y_1$) and columns as the second argument ($y_2$).
  - Standard call `(y_true, y_pred)`: Row = True, Col = Predicted $\implies \begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$.
  - Swapped call `(y_pred, y_true)`: Row = Predicted, Col = True $\implies \begin{bmatrix} TN & FN \\ FP & TP \end{bmatrix}$.
  Notice that the diagonal entries ($TN$ and $TP$) remain in place, while the off-diagonal entries ($FP$ and $FN$) are reflected across the main diagonal (i.e., matrix transposition $CM^T$). This will cause an engineer to misinterpret False Positives as False Negatives and vice versa.
- **B is incorrect:** Both `y_pred` and `y_test` are 1D arrays of identical shape and type; Python executes the call without raising any error.
- **C is incorrect:** Count data cannot become negative.
- **D is incorrect:** The matrix is transposed, not rotated 180 degrees.

</details>

---

### 92. Question

A data science intern evaluates a model's Area Under the ROC Curve using scikit-learn:
```python
from sklearn.metrics import roc_auc_score

# Line A
score = roc_auc_score(y_test, model.predict(X_test))
```
The senior ML lead points out that Line A contains a critical methodological error.

**What is the error, and how should Line A be corrected?**

- [ ] A) `roc_auc_score` requires passing `X_test` directly before `y_test`
- [ ] B) `roc_auc_score` only accepts categorical strings, not numerical arrays
- [ ] C) `roc_auc_score` must be passed the confusion matrix rather than predictions
- [ ] D) `predict()` outputs discrete binary labels ($0$ or $1$), creating a crude two-segment step function; `roc_auc_score` requires continuous probability scores from `predict_proba(X_test)[:, 1]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `predict()` outputs discrete binary labels ($0$ or $1$), creating a crude two-segment step function; `roc_auc_score` requires continuous probability scores from `predict_proba(X_test)[:, 1]`

**Explanation:**
- **A is incorrect:** The true labels `y_test` must always be passed first.
- **B is incorrect:** Scikit-learn requires numerical or boolean arrays.
- **C is incorrect:** `roc_auc_score` requires instance-level probability scores across all thresholds, not a summary matrix.
- **D is correct:** Passing discrete binary predictions from `.predict()` into `roc_auc_score` forces the function to evaluate only a single threshold point ($\tau = 0.50$). This collapses the smooth ROC curve into a crude triangle connecting $(0, 0) \to (FPR, TPR) \to (1, 1)$, underestimating the true AUC. To compute the true Area Under the Curve across all thresholds, one must supply **continuous probability scores** for the positive class:
  ```python
  score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
  ```

</details>

---

### 93. Question

The **Matthews Correlation Coefficient (MCC)** is defined as:
$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

**What are the range and interpretation of the Matthews Correlation Coefficient?**

- [ ] A) Range $[0, 1]$, where 0 indicates random prediction and 1 indicates perfect prediction
- [ ] B) Range $[-1, +1]$, where $+1$ represents perfect prediction, $0$ indicates random guessing, and $-1$ represents total disagreement (inverse prediction)
- [ ] C) Range $[-\infty, +\infty]$, with 0 indicating zero loss
- [ ] D) Range $[0, 100]$, representing a percentage score

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Range $[-1, +1]$, where $+1$ represents perfect prediction, $0$ indicates random guessing, and $-1$ represents total disagreement (inverse prediction)

**Explanation:**
- **A is incorrect:** MCC can be negative; it is not bounded below by 0.
- **B is correct:** The Matthews Correlation Coefficient is essentially the Pearson correlation coefficient between true and predicted binary indicator vectors:
  - **$\text{MCC} = +1.0$:** Perfect prediction ($FP = FN = 0$).
  - **$\text{MCC} = 0.0$:** No correlation; equivalent to random coin tossing.
  - **$\text{MCC} = -1.0$:** Perfect inverse prediction (every prediction is wrong).
  Crucially, MCC incorporates **all four quadrants of the confusion matrix** ($TP, TN, FP, FN$), making it one of the most robust metrics for highly imbalanced binary datasets.
- **C is incorrect:** MCC is mathematically bounded within $[-1, 1]$.
- **D is incorrect:** It is a correlation coefficient, not a percentage index.

</details>

---

### 94. Question

**Log Loss (Binary Cross-Entropy)** evaluates the quality of predicted probabilities rather than discrete binary decisions:
$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \Big]$$

Suppose a fraud detection model outputs a predicted fraud probability of **$p = 0.99$** for a transaction that is actually legitimate ($y = 0$).

**What is the loss penalty contributed by this single misclassified transaction?**

- [ ] A) $-\ln(0.99) \approx 0.010$
- [ ] B) $0.000$
- [ ] C) $-\ln(1 - 0.99) = -\ln(0.01) \approx 4.605$
- [ ] D) $(0.99 - 0)^2 = 0.9801$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) $-\ln(1 - 0.99) = -\ln(0.01) \approx 4.605$

**Explanation:**
- **A is incorrect:** This would be the loss if the true label were $1$ ($-\ln(0.99) \approx 0.010$, a very small penalty for being right).
- **B is incorrect:** An error cannot carry zero loss.
- **C is correct:** For a sample where $y = 0$, the first term $y \ln(p)$ vanishes, leaving:
  $$\text{Loss}_i = -\ln(1 - p_i) = -\ln(1 - 0.99) = -\ln(0.01) = \ln(100) \approx 4.605$$
  Log Loss severely and non-linearly penalizes **confident wrong predictions**. Being 99% confident on a wrong label incurs a massive penalty ($4.605$) compared to being uncertain ($p = 0.51 \implies -\ln(0.49) \approx 0.713$).
- **D is incorrect:** This is the squared error penalty (Brier score), not Log Loss.

</details>

---

### 95. Question

A machine learning practitioner splits an imbalanced dataset containing 9,500 genuine transactions and 500 fraudulent transactions using scikit-learn:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

**What essential guarantee does the parameter `stratify=y` provide for the test set?**

- [ ] A) It ensures that the test set preserves the exact original 95:5 ratio of genuine to fraudulent transactions
- [ ] B) It forces the training set to be undersampled to an exact 50:50 class balance
- [ ] C) It automatically scales all features to mean 0 and variance 1 before splitting
- [ ] D) It removes duplicate transactions from the test split

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It ensures that the test set preserves the exact original 95:5 ratio of genuine to fraudulent transactions

**Explanation:**
- **A is correct:** In imbalanced datasets, standard random splitting can suffer from sampling variance, potentially placing disproportionately few (or even zero) minority class examples into the test set. Specifying `stratify=y` performs **stratified sampling**, enforcing that the training set and testing set both retain the exact target label distribution ($95\%$ class 0, $5\%$ class 1) of the overall dataset.
- **B is incorrect:** `stratify` preserves proportions; it does not resample or rebalance data to 50:50.
- **C is incorrect:** Feature scaling must be performed via `StandardScaler`, not `train_test_split`.
- **D is incorrect:** Stratified splitting does not perform deduplication.

</details>

---

### 96. Question

When executing `print(classification_report(y_true, y_pred))` in scikit-learn, which columns of metrics are displayed for each class in the resulting text table?

- [ ] A) Accuracy, Specificity, AUC-ROC, and Log Loss
- [ ] B) Mean Squared Error, R-Squared, MAE, and RMSE
- [ ] C) False Positive Rate, False Negative Rate, Brier Score, and Power
- [ ] D) Precision, Recall, F1-Score, and Support

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Precision, Recall, F1-Score, and Support

**Explanation:**
- **A is incorrect:** Classification report does not display Specificity or AUC-ROC.
- **B is incorrect:** These are regression metrics.
- **C is incorrect:** These are statistical hypothesis testing terms not formatted in the standard report.
- **D is correct:** Scikit-learn's `classification_report` formats a standardized table containing:
  - **Precision:** Positive predictive value per class
  - **Recall:** Sensitivity per class
  - **F1-Score:** Harmonic mean of precision and recall per class
  - **Support:** The number of actual occurrences of each class in `y_true`
  At the bottom, it outputs overall Accuracy, Macro Average, and Weighted Average.

</details>

---

### 97. Question

When computing precision or F1-score on a sparse or conservative model, scikit-learn often triggers an `UndefinedMetricWarning` because the denominator ($TP + FP$) is zero.

**Which argument should be passed to functions like `precision_score`, `f1_score`, or `classification_report` to cleanly suppress this warning and assign a default value?**

- [ ] A) `handle_unknown='ignore'`
- [ ] B) `zero_division=0`
- [ ] C) `errors='coerce'`
- [ ] D) `missing_values=np.nan`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `zero_division=0`

**Explanation:**
- **A is incorrect:** `handle_unknown='ignore'` is a parameter of `OneHotEncoder`, not metric functions.
- **B is correct:** When no positive predictions are made, $TP = 0$ and $FP = 0$, leading to $\frac{0}{0}$. Passing `zero_division=0` (or `zero_division=1` or `zero_division=np.nan`) explicitly instructs scikit-learn to substitute that value when a zero-division occurs, cleanly preventing terminal warnings and script interruptions.
- **C is incorrect:** `errors='coerce'` is used in Pandas `to_numeric()`.
- **D is incorrect:** `missing_values` is used in `SimpleImputer`.

</details>

---

### 98. Question

In clinical diagnostics and inter-rater reliability studies, **Cohen's Kappa ($\kappa$)** is defined as:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
where $p_o$ is the observed accuracy and $p_e$ is the expected hypothetical accuracy under chance agreement.

**What is the core conceptual purpose of Cohen's Kappa?**

- [ ] A) To optimize the learning rate of stochastic gradient descent
- [ ] B) To measure the linear correlation between continuous feature variables
- [ ] C) To evaluate classification agreement while explicitly controlling for and removing the agreement that would occur by pure chance
- [ ] D) To compute the area under the Precision-Recall curve

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) To evaluate classification agreement while explicitly controlling for and removing the agreement that would occur by pure chance

**Explanation:**
- **A is incorrect:** Kappa is an evaluation metric, not an optimization algorithm.
- **B is incorrect:** This defines Pearson correlation.
- **C is correct:** In imbalanced datasets, two raters (or a classifier and ground truth) will achieve high agreement simply by guessing the majority class. Cohen's Kappa standardizes observed accuracy ($p_o$) against the agreement expected purely by random chance ($p_e$):
  - $\kappa = 1.0$: Complete agreement beyond chance.
  - $\kappa = 0.0$: Agreement is no better than random chance.
  - $\kappa < 0.0$: Agreement is worse than chance.
- **D is incorrect:** Kappa is not an area-under-the-curve metric.

</details>

---

### 99. Question

A hospital risk-stratification system needs to prioritize intensive care resources. The clinical leadership mandates: *"Rank all incoming patients by heart failure risk and immediately alert cardiologists for the top 5% highest-risk patients, regardless of whether their absolute probability exceeds 0.50."*

**Which scikit-learn method must the deployment engineer invoke on the trained classifier?**

- [ ] A) `model.predict_proba(X)[:, 1]` to retrieve continuous posterior probabilities for ranking and percentile slicing
- [ ] B) `model.predict(X)` to get binary discrete admission decisions
- [ ] C) `model.score(X, y)` to get the mean classification accuracy
- [ ] D) `model.get_params()` to inspect the hyperparameters

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `model.predict_proba(X)[:, 1]` to retrieve continuous posterior probabilities for ranking and percentile slicing

**Explanation:**
- **A is correct:** To select the top $5\%$ highest-risk patients, the engineer cannot rely on discrete $0/1$ outputs from `predict()`. Calling `model.predict_proba(X)[:, 1]` returns continuous probability estimates $P(y=1|x) \in [0, 1]$. The team can then sort the cohort descending by probability and flag the top $5\text{th}$ percentile threshold using `np.percentile(probabilities, 95)`.
- **B is incorrect:** `predict()` applies a rigid 0.5 threshold, which might flag 0.1% or 40% of patients rather than the required top 5%.
- **C is incorrect:** `score()` returns overall accuracy, not instance-level patient scores.
- **D is incorrect:** `get_params()` returns model configurations.

</details>

---

### 100. Question

In modern scikit-learn (version 1.0+), which dedicated display utility class enables developers to compute and plot an ROC curve directly from an estimator and test data in a single unified method call?

- [ ] A) `sklearn.metrics.plot_confusion_matrix`
- [ ] B) `sklearn.metrics.CurvePlotter`
- [ ] C) `matplotlib.pyplot.roc_curve`
- [ ] D) `sklearn.metrics.RocCurveDisplay.from_estimator`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `sklearn.metrics.RocCurveDisplay.from_estimator`

**Explanation:**
- **A is incorrect:** `plot_confusion_matrix` was deprecated and removed in scikit-learn 1.2 in favor of `ConfusionMatrixDisplay`.
- **B is incorrect:** `CurvePlotter` is not a scikit-learn class.
- **C is incorrect:** Matplotlib has no built-in `roc_curve` module; `roc_curve` is part of scikit-learn.
- **D is correct:** Modern scikit-learn provides `RocCurveDisplay`:
  ```python
  from sklearn.metrics import RocCurveDisplay

  RocCurveDisplay.from_estimator(model, X_test, y_test)
  plt.show()
  ```
  This method handles prediction probability extraction, threshold sweeping, TPR/FPR computation, AUC calculation, and Matplotlib plotting in a single, robust, production-grade call. Similarly, `PrecisionRecallDisplay.from_estimator` exists for PR curves.

</details>
