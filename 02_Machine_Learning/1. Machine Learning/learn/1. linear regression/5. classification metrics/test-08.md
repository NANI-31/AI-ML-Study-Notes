### 71. Question

When constructing a **Precision-Recall (PR) Curve**, which metrics are assigned to the vertical ($y$) and horizontal ($x$) axes, respectively?

- [ ] A) Vertical: False Positive Rate; Horizontal: True Positive Rate
- [ ] B) Vertical: Specificity; Horizontal: Sensitivity
- [ ] C) Vertical: Recall; Horizontal: Precision
- [ ] D) Vertical: Precision; Horizontal: Recall

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Vertical: Precision; Horizontal: Recall

**Explanation:**
- **A is incorrect:** This is the inverted ROC curve axis configuration.
- **B is incorrect:** Specificity vs. Sensitivity is an alternate medical formulation of ROC analysis.
- **C is incorrect:** In standard machine learning literature and scikit-learn, Recall is plotted on the horizontal axis and Precision on the vertical axis.
- **D is correct:** By universal convention:
  - **Horizontal axis ($x$):** Recall ($\frac{TP}{TP + FN}$, Sensitivity) ranging from $0.0$ to $1.0$.
  - **Vertical axis ($y$):** Precision ($\frac{TP}{TP + FP}$, Positive Predictive Value) ranging from $0.0$ to $1.0$.
  The curve tracks how precision evolves as the decision cutoff is progressively relaxed to capture higher recall.

</details>

---

### 72. Question

On a standard ROC curve, the baseline performance of an uninformative random classifier is a fixed diagonal line from $(0, 0)$ to $(1, 1)$ with an AUC of $0.50$.

**What constitutes the baseline performance of a random classifier on a Precision-Recall (PR) Curve?**

- [ ] A) A horizontal line at $y = \frac{P}{P + N}$, corresponding to the prevalence (fraction) of positive instances in the dataset
- [ ] B) A fixed diagonal line extending from $(0, 0)$ to $(1, 1)$ with an area of 0.50
- [ ] C) A vertical line at $x = 0.50$
- [ ] D) A curve that hugs the bottom-left origin with an area of 0.00

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A horizontal line at $y = \frac{P}{P + N}$, corresponding to the prevalence (fraction) of positive instances in the dataset

**Explanation:**
- **A is correct:** A random classifier that randomly predicts a fraction of instances as positive will, on average, capture positives and negatives in exact proportion to their underlying population frequencies. Therefore, its expected precision is simply the positive prevalence:
  $$\text{Baseline Precision} = \frac{\text{Actual Positives}}{\text{Total Samples}} = \frac{P}{P + N}$$
  This appears as a **flat horizontal line** across all recall values. If a dataset has only $2\%$ fraud ($P / (P+N) = 0.02$), the baseline PR curve has an area of only **$0.02$**, not $0.50$.
- **B is incorrect:** The diagonal line is unique to the ROC curve; the PR curve baseline depends directly on class balance.
- **C is incorrect:** A vertical line has zero area and does not represent random classification.
- **D is incorrect:** The baseline is strictly non-zero and equals the class balance fraction.

</details>

---

### 73. Question

A machine learning engineer evaluates an anti-fraud model on a transaction log where fraudulent transactions make up exactly **$1.0\%$** of all records ($P / (P+N) = 0.01$). The model achieves an Area Under the Precision-Recall Curve (AUC-PR / Average Precision) of **$0.65$**.

**How should this performance be interpreted in relation to random chance?**

- [ ] A) The model is only 15% better than random guessing (since $0.65 - 0.50 = 0.15$)
- [ ] B) The model failed because a valid classifier must achieve an AUC-PR of at least 0.90
- [ ] C) The model represents a massive 65-fold improvement over random guessing, whose baseline AUC-PR is only 0.01
- [ ] D) The model's accuracy on the fraud class is exactly 65%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The model represents a massive 65-fold improvement over random guessing, whose baseline AUC-PR is only 0.01

**Explanation:**
- **A is incorrect:** This mistake treats the random baseline as 0.50 (confusing PR curves with ROC curves). The baseline here is $0.01$.
- **B is incorrect:** An AUC-PR of 0.65 on a 1% rare-event dataset is typically exceptional in production systems.
- **C is correct:** On a dataset with 1% positive prevalence, the random guessing baseline on a PR plot is a flat line at $y = 0.01$, enclosing an area of $\text{AUC-PR}_{\text{baseline}} = 0.01$. Achieving an AUC-PR of $0.65$ represents a **65-fold enhancement ($65 \times$)** in precision across all recall levels relative to chance.
- **D is incorrect:** AUC-PR integrates precision-recall trade-offs across all thresholds; it is not a point accuracy estimate.

</details>

---

### 74. Question

Why is the **Precision-Recall (PR) Curve** universally recommended over the **ROC Curve** when assessing classification models on highly skewed, severely imbalanced datasets (e.g., fraud, rare diseases, cyber-intrusions)?

- [ ] A) PR curves require significantly less memory to compute in scikit-learn
- [ ] B) PR curves focus exclusively on the positive minority class and do not incorporate True Negatives ($TN$), preventing a vast negative majority from masking false alarms
- [ ] C) PR curves are always completely linear, making them easier to explain to executive stakeholders
- [ ] D) The ROC curve cannot be plotted if the positive class has fewer than 100 samples

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) PR curves focus exclusively on the positive minority class and do not incorporate True Negatives ($TN$), preventing a vast negative majority from masking false alarms

**Explanation:**
- **A is incorrect:** Computational complexity is identical between both curves ($O(N \log N)$ for sorting).
- **B is correct:** The ROC curve's x-axis is $FPR = \frac{FP}{TN + FP}$. When $TN$ is massive (e.g., 1,000,000 legitimate users), generating 10,000 false alarms yields a tiny $FPR = 0.01$ (looking great on an ROC curve). However, on the PR curve, the y-axis is $\text{Precision} = \frac{TP}{TP + FP}$. If there are only 100 true frauds, catching 80 yields $\text{Precision} = \frac{80}{80 + 10,000} \approx 0.008$ ($0.8\%$). The PR curve **exposes the unacceptable false alarm rate immediately**, making it vastly superior for imbalanced diagnostics.
- **C is incorrect:** PR curves are non-linear and often exhibit irregular step-like trajectories.
- **D is incorrect:** ROC curves can be computed on any sample size.

</details>

---

### 75. Question

In scikit-learn, **Average Precision (AP)** is computed to summarize the Precision-Recall curve into a single scalar metric. Which mathematical formula represents how `average_precision_score` calculates this area?

- [ ] A) $\text{AP} = \frac{1}{N} \sum_{i=1}^N (P_i + R_i)$
- [ ] B) $\text{AP} = \max_{\tau} (\text{Precision}(\tau) \times \text{Recall}(\tau))$
- [ ] C) $\text{AP} = \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- [ ] D) $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$

**Explanation:**
- **A is incorrect:** This is an unweighted average of precision and recall values.
- **B is incorrect:** This is the peak product of precision and recall.
- **C is incorrect:** This is the harmonic mean formula without the factor of 2.
- **D is correct:** Scikit-learn implements Average Precision using the formula:
  $$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$
  where $P_n$ and $R_n$ are the precision and recall at the $n$-th operating threshold. This calculates a **weighted average of precision values**, where the weight applied to each precision is the step-increase in Recall achieved by lowering the threshold to that point.

</details>

---

### 76. Question

As a classification model's decision threshold is lowered from $1.0$ down to $0.0$, allowing Recall to increase continuously from $0.0$ up to $1.0$, **what general trajectory does the Precision-Recall curve exhibit?**

- [ ] A) An overall downward or declining trajectory, where higher recall generally requires accepting lower precision
- [ ] B) A strictly increasing upward trajectory that starts at 0 and ends at 1
- [ ] C) A horizontal line that stays constant at 0.50
- [ ] D) A parabolic curve that peaks at Recall = 0.50 and drops to zero at both ends

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) An overall downward or declining trajectory, where higher recall generally requires accepting lower precision

**Explanation:**
- **A is correct:** To increase Recall, the model must lower its decision threshold to capture more positive cases. In doing so, it inevitably begins classifying ambiguous and borderline instances as positive, admitting False Positives into the predicted pool. As a result, **Precision generally drops as Recall rises**, creating a downward sloping curve from the top-left toward the bottom-right.
- **B is incorrect:** Precision cannot systematically increase as a model becomes more permissive and admits more false alarms.
- **C is incorrect:** Only an uninformative random classifier produces a flat horizontal line.
- **D is incorrect:** The curve does not form an inverted parabola.

</details>

---

### 77. Question

On a Precision-Recall plot where Recall is on the x-axis and Precision is on the y-axis, **which coordinate represents a theoretically "perfect" binary classifier?**

- [ ] A) The origin $(0.0, 0.0)$
- [ ] B) The bottom-right corner $(1.0, 0.0)$
- [ ] C) The top-right corner $(1.0, 1.0)$
- [ ] D) The top-left corner $(0.0, 1.0)$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The top-right corner $(1.0, 1.0)$

**Explanation:**
- **A is incorrect:** The origin represents zero recall and zero precision.
- **B is incorrect:** $(1.0, 0.0)$ represents capturing all positives ($R=1$) but with an infinite number of false alarms ($P=0$).
- **C is correct:** A perfect classifier captures **$100\%$ of all actual positive instances** ($\text{Recall} = 1.0$) while committing **zero false positives** ($\text{Precision} = 1.0$). This corresponds to the **top-right coordinate $(1.0, 1.0)$**, enclosing an Area Under the PR Curve of **$1.0$**. (Contrast with the ROC curve, where the perfect classifier coordinate is the top-left $(0.0, 1.0)$).
- **D is incorrect:** $(0.0, 1.0)$ represents perfect precision but zero recall (finding zero positive cases).

</details>

---

### 78. Question

A clinical data science team compares two diagnostic models for a disease screening clinic that mandates a **minimum Recall of 80%** ($\text{Recall} \ge 0.80$):
- **Model 1:** Achieves $98\%$ precision at $30\%$ recall, but its precision drops precipitously to $20\%$ at $80\%$ recall.
- **Model 2:** Achieves $75\%$ precision at $30\%$ recall, and maintains $70\%$ precision at $80\%$ recall.

**Which model should the clinic deploy to satisfy its operational constraint?**

- [ ] A) Model 1, because its peak precision ($98\%$) is significantly higher than Model 2's peak ($75\%$)
- [ ] B) Model 2, because it delivers far higher precision ($70\%$ vs. $20\%$) at the required $80\%$ recall operating threshold
- [ ] C) Neither model can be used because precision must exceed $99\%$ at all operating points
- [ ] D) Model 1, because models with high initial precision always have higher total AUC

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Model 2, because it delivers far higher precision ($70\%$ vs. $20\%$) at the required $80\%$ recall operating threshold

**Explanation:**
- **A is incorrect:** Model 1's 98% precision is only achievable at 30% recall, leaving 70% of sick patients undetected, which violates the clinic's mandate.
- **B is correct:** The clinic's operational requirement is fixed at $\text{Recall} \ge 0.80$. At this required operating point on the PR curve, Model 2 maintains a high precision of **$70\%$** (only 3 false alarms for every 7 true cases), whereas Model 1 collapses to **$20\%$** (4 false alarms for every 1 true case). Therefore, Model 2 is vastly superior for the clinic's operating constraints.
- **C is incorrect:** 70% precision at 80% recall is often viable and acceptable in medical screening.
- **D is incorrect:** A sharp drop in precision significantly reduces total area.

</details>

---

### 79. Question

Unlike ROC curves, which are strictly monotonic step functions, empirical Precision-Recall curves plotted from real-world test sets frequently exhibit an irregular, **jagged "zig-zag" shape**.

**What mathematical phenomenon causes this zig-zag behavior in empirical PR curves?**

- [ ] A) Rounding errors inside Python's floating-point unit
- [ ] B) Overfitting in the tree depth of gradient boosting models
- [ ] C) Random shuffling of training folds during cross-validation
- [ ] D) As the threshold is lowered, encountering a True Positive increases both Precision and Recall, but encountering a False Positive increases the denominator of Precision without changing Recall, causing Precision to drop

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) As the threshold is lowered, encountering a True Positive increases both Precision and Recall, but encountering a False Positive increases the denominator of Precision without changing Recall, causing Precision to drop

**Explanation:**
- **A is incorrect:** Floating point precision does not create macroscopic jagged drops.
- **B is incorrect:** Zig-zagging occurs on any empirical evaluation regardless of model family.
- **C is incorrect:** The curve is plotted on a fixed evaluation set, not during cross-validation shuffling.
- **D is correct:** When scanning threshold $\tau$ downward through sorted predictions:
  - If the next sample is an actual positive ($TP$): $TP$ increments by 1. This increases Recall ($\frac{TP}{P}$ moves right) and increases Precision ($\frac{TP}{TP+FP}$ jumps up).
  - If the next sample is an actual negative ($FP$): $FP$ increments by 1. Recall does not change ($\Delta R = 0$), but the denominator of Precision grows ($\frac{TP}{TP+FP}$ drops straight down).
  This alternating sequence of vertical drops and diagonal upward-right movements generates the characteristic jagged zig-zag pattern of empirical PR curves.

</details>

---

### 80. Question

In scikit-learn, which function should be used to compute the area under the Precision-Recall curve using the standard step-wise weighted average without introducing linear interpolation bias?

- [ ] A) `sklearn.metrics.average_precision_score`
- [ ] B) `sklearn.metrics.auc(recall, precision)`
- [ ] C) `sklearn.metrics.roc_auc_score`
- [ ] D) `sklearn.metrics.f1_score`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `sklearn.metrics.average_precision_score`

**Explanation:**
- **A is correct:** In scikit-learn, `average_precision_score(y_true, y_score)` calculates Average Precision (AP) as the weighted sum of precisions at each threshold. As noted in Davis & Goadrich (2006) and the scikit-learn documentation, using linear trapezoidal interpolation on raw PR points via `auc(recall, precision)` creates an overly optimistic interpolation bias. `average_precision_score` avoids this bias by computing the exact step-function area.
- **B is incorrect:** Applying trapezoidal `auc()` to PR points overestimates the area due to non-linear curvature between points.
- **C is incorrect:** This computes the Area Under the ROC Curve, not the PR Curve.
- **D is incorrect:** This computes the F1 score at a single discrete threshold.

</details>
