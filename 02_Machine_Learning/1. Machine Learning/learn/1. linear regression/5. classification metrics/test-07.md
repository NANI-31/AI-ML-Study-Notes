### 61. Question

When plotting a standard **Receiver Operating Characteristic (ROC) Curve**, which metrics are assigned to the vertical ($y$) and horizontal ($x$) axes, respectively?

- [ ] A) Vertical: Precision; Horizontal: Recall
- [ ] B) Vertical: True Positive Rate (Sensitivity); Horizontal: False Positive Rate ($1 - \text{Specificity}$)
- [ ] C) Vertical: True Positive Rate; Horizontal: False Negative Rate
- [ ] D) Vertical: Accuracy; Horizontal: Decision Threshold

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Vertical: True Positive Rate (Sensitivity); Horizontal: False Positive Rate ($1 - \text{Specificity}$)

**Explanation:**
- **A is incorrect:** Precision vs. Recall defines the Precision-Recall (PR) Curve, not the ROC Curve.
- **B is correct:** By definition, the ROC Curve plots:
  - **Y-axis:** True Positive Rate ($TPR = \text{Recall} = \text{Sensitivity} = \frac{TP}{TP + FN}$)
  - **X-axis:** False Positive Rate ($FPR = 1 - \text{Specificity} = \frac{FP}{TN + FP}$)
  as the classification decision cutoff varies continuously from $\infty$ down to $-\infty$.
- **C is incorrect:** The x-axis is False Positive Rate ($FPR$), not False Negative Rate ($FNR$).
- **D is incorrect:** Accuracy and Thresholds are not the Cartesian coordinate axes of the plot.

</details>

---

### 62. Question

On an ROC plot with $FPR$ on the horizontal axis and $TPR$ on the vertical axis, **which Cartesian coordinate represents a theoretically "perfect" binary classifier?**

- [ ] A) Coordinate $(1.0, 1.0)$
- [ ] B) Coordinate $(0.0, 0.0)$
- [ ] C) Coordinate $(0.0, 1.0)$
- [ ] D) Coordinate $(0.5, 0.5)$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Coordinate $(0.0, 1.0)$

**Explanation:**
- **A is incorrect:** $(1.0, 1.0)$ is the point where the model predicts positive for every sample ($\tau = 0$), yielding $100\%$ Recall but also $100\%$ False Positive Rate.
- **B is incorrect:** $(0.0, 0.0)$ is the point where the model predicts negative for every sample ($\tau = 1$), yielding $0$ False Positives but also $0$ True Positives.
- **C is correct:** A perfect classifier catches $100\%$ of actual positives ($TPR = 1.0$) while committing zero false alarms on negatives ($FPR = 0.0$). This corresponds uniquely to the **top-left corner $(0.0, 1.0)$** of the unit square.
- **D is incorrect:** $(0.5, 0.5)$ lies on the diagonal line representing random coin-flip performance.

</details>

---

### 63. Question

What is the fundamental theoretical meaning of the diagonal dashed line extending from $(0, 0)$ to $(1, 1)$ on an ROC plot?

- [ ] A) It represents the expected performance of an uninformative classifier making random guesses, with an Area Under the Curve (AUC) of 0.50
- [ ] B) It represents the boundary where Precision equals Recall
- [ ] C) It represents an optimal classifier trained with balanced class weights
- [ ] D) It represents the point where True Negatives equal True Positives

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It represents the expected performance of an uninformative classifier making random guesses, with an Area Under the Curve (AUC) of 0.50

**Explanation:**
- **A is correct:** The diagonal line $y = x$ represents a classifier that cannot distinguish between classes at all. For any threshold, it flags positive and negative samples at the exact same rate ($TPR = FPR$). The area beneath this diagonal line across the unit square $[0, 1] \times [0, 1]$ is exactly:
  $$\text{Area} = \frac{1}{2} \times 1.0 \times 1.0 = 0.50$$
  Any valid machine learning model must achieve an ROC curve that arches significantly above this diagonal line ($\text{AUC} > 0.50$).
- **B is incorrect:** Precision is not reflected on the ROC diagonal.
- **C is incorrect:** An optimal classifier hugs the top-left corner, far above the diagonal.
- **D is incorrect:** The diagonal line equates $TPR$ and $FPR$, which depend on their respective class totals, not raw equality between $TP$ and $TN$.

</details>

---

### 64. Question

According to statistical probability theory and the Wilcoxon-Mann-Whitney U statistic, what is the precise **probabilistic interpretation** of the Area Under the ROC Curve (AUC-ROC)?

- [ ] A) The probability that the model's accuracy on an unseen test set will exceed 50%
- [ ] B) The percentage of total variance explained by the model's first principal component
- [ ] C) The probability that a sample predicted as positive is truly an actual positive
- [ ] D) The probability that a randomly chosen positive instance will be assigned a higher predicted score than a randomly chosen negative instance

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The probability that a randomly chosen positive instance will be assigned a higher predicted score than a randomly chosen negative instance

**Explanation:**
- **A is incorrect:** AUC does not measure probability of achieving a target accuracy.
- **B is incorrect:** This describes explained variance ratio in PCA, completely unrelated to ROC analysis.
- **C is incorrect:** This is the definition of Precision (Positive Predictive Value).
- **D is correct:** Mathematically, the AUC of a classifier $f(x)$ equals:
  $$\text{AUC} = P\big(f(x^+) > f(x^-)\big)$$
  where $x^+$ is a randomly drawn instance from the positive class and $x^-$ is a randomly drawn instance from the negative class (with ties broken by $+0.5$). If $\text{AUC} = 0.88$, there is an $88\%$ probability that the model scores an arbitrary positive sample higher than an arbitrary negative sample.

</details>

---

### 65. Question

A machine learning intern fits a binary classification model and evaluates it on a test set, resulting in an AUC-ROC of $0.15$.

**What does an AUC of 0.15 indicate about the model, and how can it be salvaged?**

- [ ] A) The model learned pure noise and is completely uninformative; it must be discarded
- [ ] B) The model learned strong discriminatory signal but has inverted labels; inverting all predicted probabilities ($1 - p$) yields an excellent AUC of 0.85
- [ ] C) The dataset suffered from severe multicollinearity; removing half the features will restore AUC to 0.50
- [ ] D) The model has an accuracy of exactly 15% across all thresholds

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The model learned strong discriminatory signal but has inverted labels; inverting all predicted probabilities ($1 - p$) yields an excellent AUC of 0.85

**Explanation:**
- **A is incorrect:** An uninformative random model yields an AUC of $\approx 0.50$, not $0.15$. An AUC far below $0.50$ indicates strong systematic information that is simply upside-down.
- **B is correct:** An AUC of $0.15$ means that negative samples are systematically receiving higher scores than positive samples $85\%$ of the time ($1.0 - 0.15 = 0.85$). This typically happens when the target labels $0$ and $1$ were accidentally flipped during data loading or model mapping. Reversing the predictions ($\hat{y}_{\text{new}} = 1 - \hat{y}$) inverts the ROC curve across the diagonal, yielding:
  $$\text{AUC}_{\text{corrected}} = 1.0 - 0.15 = 0.85$$
- **C is incorrect:** Multicollinearity does not systematically flip AUC to $0.15$.
- **D is incorrect:** AUC measures ranking discrimination, not raw accuracy.

</details>

---

### 66. Question

Suppose every predicted probability score $p_i$ produced by a model on a test set undergoes a strictly monotonic positive transformation: $p_i^{\text{new}} = \log\left(\frac{p_i}{1 - p_i}\right)$ (mapping probabilities to log-odds).

**What happens to the model's AUC-ROC after this transformation?**

- [ ] A) The AUC increases significantly because log-odds have an unbounded dynamic range
- [ ] B) The AUC drops to zero because probabilities must strictly reside between 0 and 1
- [ ] C) The AUC remains strictly unchanged
- [ ] D) The AUC is squared

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The AUC remains strictly unchanged

**Explanation:**
- **A is incorrect:** Expanding scale range does not alter relative sample rankings.
- **B is incorrect:** Scikit-learn's `roc_auc_score` accepts any continuous numerical decision score ($(-\infty, +\infty)$); it does not require bounded probabilities.
- **C is correct:** AUC-ROC is a **rank-based metric**. It depends solely on whether positive instances are ranked higher than negative instances, not on their absolute numerical values. Any strictly monotonic transformation $g(p)$ preserves the inequality:
  $$p_A > p_B \iff g(p_A) > g(p_B)$$
  Because the pairwise rank ordering of all instances is preserved perfectly, the ROC curve and its AUC remain **strictly identical**.
- **D is incorrect:** Transformations that preserve order have zero effect on AUC.

</details>

---

### 67. Question

Why is AUC-ROC widely considered an essential standard benchmark metric in academic machine learning publications for binary classification?

- [ ] A) It evaluates the model's global ranking capability across all possible decision thresholds, making it threshold-independent
- [ ] B) It automatically guarantees that the model will achieve at least 90% accuracy in production
- [ ] C) It is the only metric that can be calculated without knowing ground-truth labels
- [ ] D) It calculates the economic profit of a model directly in monetary currency

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It evaluates the model's global ranking capability across all possible decision thresholds, making it threshold-independent

**Explanation:**
- **A is correct:** Metrics like Precision, Recall, and Accuracy depend on choosing a specific decision cutoff $\tau$ (such as $0.5$). However, the optimal threshold depends on operational business costs that vary across organizations. AUC-ROC integrates performance across **all operating points** simultaneously, isolating the intrinsic discriminatory power of the representation from the choice of threshold.
- **B is incorrect:** A model can have high AUC (e.g., 0.95) but poor accuracy if evaluated at an inappropriate threshold under severe class imbalance.
- **C is incorrect:** AUC strictly requires ground truth labels ($y_{\text{true}}$).
- **D is incorrect:** AUC is a dimensionless probability rank metric, not an economic utility metric.

</details>

---

### 68. Question

A credit card fraud detection system is evaluated on 1,000,000 transactions where 100 are fraud (positive) and 999,900 are legitimate (negative). The model achieves an apparently stellar AUC-ROC of **$0.96$**. However, when deployed, fraud analysts complain that the system generates over 10,000 false alarms every day, overwhelming the team.

**What fundamental limitation of the ROC curve explains this discrepancy?**

- [ ] A) The ROC curve can only be computed when classes are exactly 50/50 balanced
- [ ] B) Scikit-learn computes ROC curves using approximation algorithms that overestimate AUC by 50%
- [ ] C) The model was evaluated without setting `random_state`
- [ ] D) The massive count of True Negatives keeps the False Positive Rate ($FPR = \frac{FP}{TN + FP}$) tiny even when thousands of False Positives are generated

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The massive count of True Negatives keeps the False Positive Rate ($FPR = \frac{FP}{TN + FP}$) tiny even when thousands of False Positives are generated

**Explanation:**
- **A is incorrect:** ROC curves can be computed on any dataset regardless of class balance.
- **B is incorrect:** ROC curves are calculated via exact sorting and trapezoidal integration.
- **C is incorrect:** `random_state` controls pseudo-randomness, not the mathematical definition of FPR.
- **D is correct:** In the ROC curve, the horizontal axis is $FPR = \frac{FP}{TN + FP}$. With $999,900$ true negatives, if the model generates $10,000$ false positives:
  $$FPR = \frac{10,000}{999,900 + 10,000} \approx \frac{10,000}{1,009,900} \approx 0.0099 \approx 1\%$$
  An $FPR$ of only $1\%$ looks near-perfect on an ROC plot! However, since there are only 100 actual fraud cases, generating 10,000 false positives means that for every 1 true fraud caught, there are 100 false alarms ($\text{Precision} < 1\%$). Under severe class imbalance, **ROC curves present an overly optimistic assessment**, and teams must inspect the **Precision-Recall curve** instead.

</details>

---

### 69. Question

A researcher plots the ROC curves of two classifiers, Model Alpha and Model Beta. The curve for Model Alpha lies strictly above the curve for Model Beta across the entire range of $FPR \in (0, 1)$.

**What conclusion can be drawn with mathematical certainty?**

- [ ] A) Model Beta achieves higher precision than Model Alpha at every operating threshold
- [ ] B) Model Alpha achieves higher Sensitivity than Model Beta at every single False Positive Rate
- [ ] C) Model Alpha has lower AUC-ROC than Model Beta
- [ ] D) Both models have the exact same F1 score at every threshold

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Model Alpha achieves higher Sensitivity than Model Beta at every single False Positive Rate

**Explanation:**
- **A is incorrect:** Higher ROC dominance does not strictly guarantee higher precision across all operating points due to differing threshold distributions.
- **B is correct:** The vertical axis of the ROC plot represents Sensitivity ($TPR$). If Curve Alpha lies strictly above Curve Beta for every value of $FPR$ on the horizontal axis, then:
  $$TPR_{\text{Alpha}}(FPR) > TPR_{\text{Beta}}(FPR) \quad \forall FPR \in (0, 1)$$
  This means that no matter what false alarm rate ($FPR$) an organization is willing to tolerate, Model Alpha will always detect a higher percentage of actual positive cases than Model Beta. Consequently, Model Alpha also possesses a strictly higher AUC.
- **C is incorrect:** The higher curve encloses a strictly greater area beneath it.
- **D is incorrect:** F1 scores will differ significantly between the two models.

</details>

---

### 70. Question

A simple classifier evaluated across various thresholds produces the following four discrete $(FPR, TPR)$ coordinates on its ROC curve:
- Point 1: $(0.0, 0.0)$
- Point 2: $(0.2, 0.6)$
- Point 3: $(0.5, 0.8)$
- Point 4: $(1.0, 1.0)$

Using standard trapezoidal numerical integration (as performed by `sklearn.metrics.auc`), **what is the Area Under the Curve (AUC)?**

- [ ] A) 0.60
- [ ] B) 0.65
- [ ] C) 0.72
- [ ] D) 0.80

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.72

**Explanation:**
- **A is incorrect:** Underestimates the area under the segments.
- **B is incorrect:** $0.65$ misses the height contribution of the middle trapezoid.
- **C is correct:** The total area under the curve is computed by summing the areas of the three trapezoids formed between the consecutive points:
  1. **Segment 1** from $(0.0, 0.0)$ to $(0.2, 0.6)$:
     $$\text{Area}_1 = (0.2 - 0.0) \times \frac{0.0 + 0.6}{2} = 0.2 \times 0.3 = 0.06$$
  2. **Segment 2** from $(0.2, 0.6)$ to $(0.5, 0.8)$:
     $$\text{Area}_2 = (0.5 - 0.2) \times \frac{0.6 + 0.8}{2} = 0.3 \times 0.7 = 0.21$$
  3. **Segment 3** from $(0.5, 0.8)$ to $(1.0, 1.0)$:
     $$\text{Area}_3 = (1.0 - 0.5) \times \frac{0.8 + 1.0}{2} = 0.5 \times 0.9 = 0.45$$
  Summing the three trapezoidal areas:
  $$\text{Total AUC} = 0.06 + 0.21 + 0.45 = 0.72$$
- **D is incorrect:** $0.80$ overestimates the integrated area.

</details>
