### 51. Question

In scikit-learn, calling `model.predict(X)` on a trained binary classifier (such as `LogisticRegression`) maps predicted probabilities $P(y=1|x)$ to discrete class labels using which default decision threshold ($\tau$)?

- [ ] A) $\tau = 0.50$
- [ ] B) $\tau = 0.00$
- [ ] C) $\tau = \text{Class 1 Prevalence in the training set}$
- [ ] D) $\tau = 0.75$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) $\tau = 0.50$

**Explanation:**
- **A is correct:** By universal convention, scikit-learn's `predict()` method evaluates the binary decision rule:
  $$\hat{y} = \begin{cases} 1 & \text{if } P(y=1|x) \ge 0.50 \\ 0 & \text{if } P(y=1|x) < 0.50 \end{cases}$$
  This threshold corresponds to picking whichever class has the highest estimated posterior probability ($P(y=1|x) \ge P(y=0|x)$).
- **B is incorrect:** $\tau = 0.0$ would classify every single observation as positive ($1$).
- **C is incorrect:** Although thresholding at the training prevalence is often optimal for imbalanced data, scikit-learn does not adjust this automatically.
- **D is incorrect:** $\tau = 0.75$ would be a conservative custom threshold, not the built-in default.

</details>

---

### 52. Question

A machine learning engineer raises the decision cutoff threshold of a customer churn model from $\tau = 0.50$ to $\tau = 0.85$.

**What are the expected directional shifts in True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), and True Negatives ($TN$)?**

- [ ] A) $TP$ increases, $FP$ increases, $FN$ decreases, $TN$ decreases
- [ ] B) $TP$ and $FP$ both increase, while $FN$ and $TN$ remain unchanged
- [ ] C) $TP$ remains unchanged, while $FP$ increases and $TN$ decreases
- [ ] D) $TP$ decreases (or stays same), $FP$ decreases (or stays same), $FN$ increases (or stays same), $TN$ increases (or stays same)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) $TP$ decreases (or stays same), $FP$ decreases (or stays same), $FN$ increases (or stays same), $TN$ increases (or stays same)

**Explanation:**
- **A is incorrect:** This describes the effect of lowering the threshold to make the classifier more permissive.
- **B is incorrect:** Raising the threshold makes positive predictions harder to obtain, reducing the total count of positive predictions ($TP + FP$).
- **C is incorrect:** $FP$ cannot increase when the model becomes more conservative.
- **D is correct:** When the threshold is raised from $0.50$ to $0.85$, only samples with extreme confidence ($P \ge 0.85$) are classified as positive. Consequently:
  - Fewer samples are predicted positive $\implies TP$ and $FP$ both decrease (or stay constant).
  - More samples are predicted negative $\implies TN$ and $FN$ both increase (or stay constant).
  - Typically, this raises **Precision** while lowering **Recall**.

</details>

---

### 53. Question

A medical risk classifier outputs continuous probabilities for 6 patients. The actual status (1 = high risk, 0 = low risk) and predicted probabilities are:

| Patient | Actual Status | Predicted Probability $P(y=1|x)$ |
| :------ | :-----------: | :------------------------------: |
| P1      |       0       |               0.25               |
| P2      |       1       |               0.45               |
| P3      |       0       |               0.55               |
| P4      |       1       |               0.65               |
| P5      |       0       |               0.75               |
| P6      |       1       |               0.85               |

**What are the confusion matrix counts ($TP, FP, TN, FN$) when evaluating at threshold $\tau = 0.50$?**

- [ ] A) $TP = 3, FP = 0, TN = 3, FN = 0$
- [ ] B) $TP = 2, FP = 2, TN = 1, FN = 1$
- [ ] C) $TP = 1, FP = 1, TN = 2, FN = 2$
- [ ] D) $TP = 2, FP = 1, TN = 2, FN = 1$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) $TP = 2, FP = 2, TN = 1, FN = 1$

**Explanation:**
- **A is incorrect:** This would assume a perfect classifier.
- **B is correct:** At threshold $\tau = 0.50$, predictions are positive ($1$) if $P \ge 0.50$ and negative ($0$) if $P < 0.50$:
  - **P1:** Actual 0, $P = 0.25 < 0.50 \implies$ Predicted 0 $\implies$ **True Negative (TN)**
  - **P2:** Actual 1, $P = 0.45 < 0.50 \implies$ Predicted 0 $\implies$ **False Negative (FN)**
  - **P3:** Actual 0, $P = 0.55 \ge 0.50 \implies$ Predicted 1 $\implies$ **False Positive (FP)**
  - **P4:** Actual 1, $P = 0.65 \ge 0.50 \implies$ Predicted 1 $\implies$ **True Positive (TP)**
  - **P5:** Actual 0, $P = 0.75 \ge 0.50 \implies$ Predicted 1 $\implies$ **False Positive (FP)**
  - **P6:** Actual 1, $P = 0.85 \ge 0.50 \implies$ Predicted 1 $\implies$ **True Positive (TP)**
  Tallying the results:
  $$TP = 2 \text{ (P4, P6)}, \quad FP = 2 \text{ (P3, P5)}, \quad TN = 1 \text{ (P1)}, \quad FN = 1 \text{ (P2)}$$
- **C is incorrect:** Miscounts the true positives.
- **D is incorrect:** Miscounts the false positive entries.

</details>

---

### 54. Question

A common intuition in machine learning is that *"lowering the decision cutoff threshold always decreases precision."*

**Is this statement strictly mathematically true under all circumstances?**

- [ ] A) Yes, it is a proven mathematical law across all probability distributions
- [ ] B) Yes, because increasing the volume of positive predictions automatically dilutes the positive pool
- [ ] C) No, because if all newly captured samples between the old and new threshold happen to be True Positives, Precision will actually increase
- [ ] D) No, because precision is threshold-independent

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) No, because if all newly captured samples between the old and new threshold happen to be True Positives, Precision will actually increase

**Explanation:**
- **A is incorrect:** It is an empirical rule of thumb, not a strict mathematical theorem.
- **B is incorrect:** Dilution only occurs if the newly admitted samples contain false positives.
- **C is correct:** Precision is $\frac{TP}{TP + FP}$. Consider a model evaluated at threshold $\tau = 0.60$ with $TP = 2$ and $FP = 2 \implies \text{Precision} = \frac{2}{4} = 0.50$. When lowering the threshold to $\tau = 0.40$, two additional samples are classified as positive. If both samples are actual positives (ground truth = 1), then $TP$ increases to $4$ while $FP$ remains $2$. The new precision becomes:
  $$\text{Precision} = \frac{4}{4 + 2} = \frac{4}{6} \approx 0.667$$
  Here, lowering the threshold **increased** precision from $0.50$ to $0.667$. Precision is not guaranteed to be strictly monotonic with respect to threshold changes.
- **D is incorrect:** Precision changes directly as the threshold shifts.

</details>

---

### 55. Question

In clinical epidemiology and ROC analysis, **Youden's J statistic** (also called Youden's Index) is frequently used to select an optimal classification cutoff:
$$J = \text{Sensitivity} + \text{Specificity} - 1$$

**What does the threshold that maximizes Youden's J statistic represent geometrically on the ROC curve?**

- [ ] A) The point on the ROC curve with the maximum vertical distance above the diagonal chance line ($TPR - FPR$)
- [ ] B) The point where the ROC curve intersects the line $y = 0.5$
- [ ] C) The point with the lowest overall False Negative Rate regardless of False Positives
- [ ] D) The coordinate where Precision equals Recall

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The point on the ROC curve with the maximum vertical distance above the diagonal chance line ($TPR - FPR$)

**Explanation:**
- **A is correct:** Since $\text{Specificity} = 1 - FPR$, substituting into Youden's formula yields:
  $$J = \text{Sensitivity} + (1 - FPR) - 1 = TPR - FPR$$
  On the ROC plot (where the y-axis is $TPR$ and the x-axis is $FPR$), the diagonal random baseline is $y = x$ ($TPR = FPR$). The value $TPR - FPR$ represents the vertical distance between the ROC curve and the diagonal line. Maximizing $J$ finds the operating point that maximizes the gap between true positives and false positives.
- **B is incorrect:** The point of intersection with $y=0.5$ has no special optimality property.
- **C is incorrect:** Minimizing $FNR$ alone corresponds to $TPR = 1.0$ at the top right, where $FPR = 1.0$.
- **D is incorrect:** Precision is not plotted on the ROC curve axes.

</details>

---

### 56. Question

A credit card fraud team conducts a cost-benefit analysis. A False Negative (missing a fraudulent transaction) costs the bank \$200 in chargebacks. A False Positive (declining a legitimate cardholder) costs the bank \$10 in customer service inquiries.

According to statistical decision theory, **what is the theoretically optimal probability cutoff threshold ($\tau^*$) that minimizes expected financial loss?**

- [ ] A) $\tau^* = 0.500$
- [ ] B) $\tau^* = 0.200$
- [ ] C) $\tau^* = 0.100$
- [ ] D) $\tau^* \approx 0.048$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) $\tau^* \approx 0.048$

**Explanation:**
- **A is incorrect:** $\tau = 0.50$ assumes symmetric costs where false alarms and misses cost the exact same amount.
- **B is incorrect:** $\frac{10}{50} = 0.20$, which does not use the total cost denominator.
- **C is incorrect:** Rough estimate that ignores the exact decision formula.
- **D is correct:** In binary cost-sensitive decision theory, the optimal threshold $\tau^*$ that minimizes expected financial risk under well-calibrated probabilities is:
  $$\tau^* = \frac{\text{Cost}(FP)}{\text{Cost}(FP) + \text{Cost}(FN)} = \frac{10}{10 + 200} = \frac{10}{210} \approx 0.0476 \approx 0.048$$
  Because a miss costs 20 times more than a false alarm, the bank should flag any transaction with an estimated fraud probability of **$4.8\%$ or higher**.

</details>

---

### 57. Question

A data scientist needs to generate predictions from a trained scikit-learn model using a custom decision threshold of $\tau = 0.35$ rather than the default $0.50$.

**Which Python code snippet correctly implements this custom threshold?**

- [ ] A) `y_pred = model.predict(X_test, threshold=0.35)`
- [ ] B) `y_pred = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)`
- [ ] C) `y_pred = model.decision_function(X_test) > 0.35`
- [ ] D) `y_pred = model.predict_proba(X_test) >= 0.35`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `y_pred = (model.predict_proba(X_test)[:, 1] >= 0.35).astype(int)`

**Explanation:**
- **A is incorrect:** Scikit-learn's `predict()` method does not accept a `threshold` argument.
- **B is correct:** `model.predict_proba(X_test)[:, 1]` extracts the predicted probabilities for the positive class (column index 1). Comparing this 1D array against `>= 0.35` produces a boolean array, and `.astype(int)` converts `True` to $1$ and `False` to $0$, implementing the exact custom decision rule.
- **C is incorrect:** `decision_function()` returns signed log-odds / geometric margins ($(-\infty, +\infty)$), not probabilities in $[0, 1]$.
- **D is incorrect:** Applying `>= 0.35` to the entire $N \times 2$ matrix produces a $N \times 2$ boolean matrix rather than a $1\text{D}$ vector of predicted binary labels.

</details>

---

### 58. Question

To evaluate how well a model's predicted probabilities match empirical reality across various thresholds, researchers compute the **Brier Score**:
$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
where $p_i \in [0, 1]$ is the predicted probability and $y_i \in \{0, 1\}$ is the actual binary outcome.

**Which of the following statements regarding the Brier Score is correct?**

- [ ] A) Higher Brier scores indicate superior probability calibration
- [ ] B) A model that predicts $p_i = 0.5$ for every patient on a 50/50 balanced dataset receives a Brier score of 0.50
- [ ] C) The Brier score is strictly a strictly proper scoring rule where 0.0 represents perfect prediction
- [ ] D) Brier score operates exclusively on discrete thresholded labels ($0$ or $1$)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The Brier score is strictly a strictly proper scoring rule where 0.0 represents perfect prediction

**Explanation:**
- **A is incorrect:** The Brier Score is a loss / error metric; lower values denote better calibration and accuracy.
- **B is incorrect:** For $y_i \in \{0, 1\}$ and $p_i = 0.5$, $(0.5 - 1)^2 = 0.25$ and $(0.5 - 0)^2 = 0.25$. The mean is $0.25$, not $0.50$.
- **C is correct:** The Brier Score is essentially the Mean Squared Error applied to probabilities. If every positive sample receives $p_i = 1.0$ and every negative sample receives $p_i = 0.0$, the Brier Score is $(1 - 1)^2 = (0 - 0)^2 = 0.0$ (perfect calibration and discrimination). It is a mathematically "proper scoring rule" because it is uniquely minimized when the predicted probabilities equal the true underlying conditional probabilities.
- **D is incorrect:** Brier score requires continuous probability estimates $p_i \in [0, 1]$.

</details>

---

### 59. Question

A junior developer runs experiments adjusting the classification threshold from $\tau = 0.20$ to $\tau = 0.80$ and asks how this adjustment alters the **ROC curve** of the model.

**What is the correct technical answer?**

- [ ] A) Shifting the decision threshold does NOT alter the ROC curve itself; it merely moves the operating point to a different position along the existing curve
- [ ] B) Raising the threshold shifts the entire ROC curve upward, increasing the Area Under the Curve (AUC)
- [ ] C) Lowering the threshold rotates the ROC curve clockwise around the origin
- [ ] D) The ROC curve disappears and is replaced by a single static coordinate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Shifting the decision threshold does NOT alter the ROC curve itself; it merely moves the operating point to a different position along the existing curve

**Explanation:**
- **A is correct:** By definition, the ROC curve is constructed by computing the pair $(FPR(\tau), TPR(\tau))$ across **all possible continuous thresholds $\tau \in [-\infty, +\infty]$**. Therefore, changing the operating threshold used in production simply selects a specific $(FPR, TPR)$ coordinate on the pre-existing curve. The curve itself, as well as its AUC, is completely invariant to choice of operational threshold.
- **B is incorrect:** The threshold does not move the curve or change AUC.
- **C is incorrect:** Thresholds do not rotate the curve.
- **D is incorrect:** The curve represents the full spectrum of all thresholds.

</details>

---

### 60. Question

Why do classifiers like Support Vector Machines (SVM with Platt scaling) or Naive Bayes often require **probability calibration** (e.g., using scikit-learn's `CalibratedClassifierCV`) before applying custom probability cutoffs?

- [ ] A) Uncalibrated models cannot compute predictions on testing sets
- [ ] B) Probability calibration converts non-linear classifiers into linear models
- [ ] C) Scikit-learn will throw a `ValueError` if a custom threshold is used on an uncalibrated classifier
- [ ] D) Their raw outputs preserve ranking order but distort absolute probabilities, making cutoff thresholds like 0.35 unreflective of true event probabilities

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Their raw outputs preserve ranking order but distort absolute probabilities, making cutoff thresholds like 0.35 unreflective of true event probabilities

**Explanation:**
- **A is incorrect:** Uncalibrated models produce predictions without difficulty.
- **B is incorrect:** Calibration maps output values to $[0, 1]$; it does not alter the underlying decision boundary geometry.
- **C is incorrect:** Python will happily execute array thresholding on any numerical vector without raising errors.
- **D is correct:** Models like Naive Bayes (which assumes complete feature independence, pushing probabilities toward 0 or 1) and SVMs (which optimize geometric margins rather than log-loss) often rank samples accurately but produce severely distorted probability values. If a model outputs $0.99$ for samples with an actual empirical risk of only $60\%$, setting a decision threshold at $\tau = 0.50$ or $\tau = 0.35$ will not produce the expected business outcomes. Calibration aligns predicted scores with true observed empirical frequencies.

</details>
