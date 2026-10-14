### 31. Question

The **F1 Score** balances Precision and Recall into a single evaluation metric. Which of the following expressions represents the F1 score formulated directly from the elements of the confusion matrix?

- [ ] A) $\frac{TP + TN}{TP + TN + FP + FN}$
- [ ] B) $\frac{TP}{TP + FP + FN}$
- [ ] C) $\frac{2(TP + TN)}{2TP + FP + FN}$
- [ ] D) $\frac{2TP}{2TP + FP + FN}$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) $\frac{2TP}{2TP + FP + FN}$

**Explanation:**
- **A is incorrect:** This is the formula for overall Accuracy.
- **B is incorrect:** This is the Jaccard Index (Threat Score / Critical Success Index), which lacks the factor of 2 in the numerator.
- **C is incorrect:** True Negatives ($TN$) are completely excluded from the F1 score.
- **D is correct:** Substituting Precision ($\frac{TP}{TP + FP}$) and Recall ($\frac{TP}{TP + FN}$) into the harmonic mean formula yields:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}{\frac{TP}{TP + FP} + \frac{TP}{TP + FN}} = \frac{2TP}{2TP + FP + FN}$$

</details>

---

### 32. Question

A churn prediction model demonstrates a Precision of $0.90$ and a Recall of $0.60$ on an unseen validation dataset.

**What is the F1 Score of this model, and how does it compare to the simple arithmetic mean of Precision and Recall?**

- [ ] A) F1 score is 0.720, which is strictly lower than the arithmetic mean (0.750)
- [ ] B) F1 score is 0.750, exactly equal to the arithmetic mean
- [ ] C) F1 score is 0.780, which is higher than the arithmetic mean
- [ ] D) F1 score is 0.540, which equals the geometric product of precision and recall

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) F1 score is 0.720, which is strictly lower than the arithmetic mean (0.750)

**Explanation:**
- **A is correct:** The harmonic mean is calculated as:
  $$F_1 = 2 \times \frac{0.90 \times 0.60}{0.90 + 0.60} = 2 \times \frac{0.54}{1.50} = \frac{1.08}{1.50} = 0.720$$
  The arithmetic mean is:
  $$\text{Arithmetic Mean} = \frac{0.90 + 0.60}{2} = 0.750$$
  Because the harmonic mean disproportionately penalizes the lower component (Recall = 0.60), $F_1 = 0.720 < 0.750$.
- **B is incorrect:** The harmonic mean only equals the arithmetic mean when Precision and Recall are identical.
- **C is incorrect:** The harmonic mean can never exceed the arithmetic mean.
- **D is incorrect:** 0.540 is simply $P \times R$, omitting the remaining terms of the harmonic mean.

</details>

---

### 33. Question

Why does machine learning universally adopt the **harmonic mean** for the F1 score rather than the standard **arithmetic mean**?

- [ ] A) The harmonic mean is computationally faster to evaluate on massive GPU clusters
- [ ] B) The arithmetic mean cannot handle decimal values between 0 and 1
- [ ] C) The harmonic mean heavily penalizes extreme imbalances, preventing a model with near-zero recall from appearing acceptable
- [ ] D) The harmonic mean automatically incorporates True Negatives into the calculation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The harmonic mean heavily penalizes extreme imbalances, preventing a model with near-zero recall from appearing acceptable

**Explanation:**
- **A is incorrect:** The arithmetic mean is computationally simpler than the harmonic mean.
- **B is incorrect:** The arithmetic mean is fully capable of averaging continuous values between 0 and 1.
- **C is correct:** Consider a pathological model that predicts positive for only one single obvious case: it achieves $\text{Precision} = 1.0$ and $\text{Recall} = 0.01$. The arithmetic mean would be $\frac{1.0 + 0.01}{2} = 0.505$ (a falsely respectable score). The harmonic mean, however, yields:
  $$F_1 = 2 \times \frac{1.0 \times 0.01}{1.0 + 0.01} \approx 0.0198$$
  The harmonic mean penalizes models that sacrifice one metric to artificially inflate the other.
- **D is incorrect:** The harmonic mean does not include True Negatives.

</details>

---

### 34. Question

A natural language processing classifier achieves a Precision of exactly $0.80$ and a Recall of exactly $0.80$ on a document classification benchmark.

**What is the F1 Score of this classifier?**

- [ ] A) 0.64
- [ ] B) 0.80
- [ ] C) 0.85
- [ ] D) 0.72

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.80

**Explanation:**
- **A is incorrect:** 0.64 is $0.80 \times 0.80$.
- **B is correct:** When Precision and Recall are identical ($P = R = x$), the F1 score reduces to:
  $$F_1 = 2 \times \frac{x \times x}{x + x} = \frac{2x^2}{2x} = x$$
  For $P = R = 0.80$:
  $$F_1 = 2 \times \frac{0.80 \times 0.80}{0.80 + 0.80} = \frac{1.28}{1.60} = 0.80$$
  When $P = R$, the harmonic mean, geometric mean, and arithmetic mean all converge to the exact same value.
- **C is incorrect:** The F1 score cannot exceed the maximum of precision and recall.
- **D is incorrect:** 0.72 is the F1 score for $P=0.9, R=0.6$, not $P=R=0.8$.

</details>

---

### 35. Question

A medical research team tests a screening tool on 10,000 patient records where:
- $TP = 100$
- $FP = 25$
- $FN = 75$
- $TN = 9,800$

The team then collects data from an additional 90,000 healthy individuals, all of whom are correctly classified as negative ($TN$ increases by 90,000 to 99,800), while $TP$, $FP$, and $FN$ remain unchanged.

**What is the effect of this 90,000 increase in True Negatives on the F1 Score?**

- [ ] A) The F1 score increases significantly toward 1.0
- [ ] B) The F1 score drops toward 0.0 because the positive class became rarer
- [ ] C) The F1 score is divided by 10
- [ ] D) The F1 score remains completely unchanged

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The F1 score remains completely unchanged

**Explanation:**
- **A is incorrect:** Accuracy would increase toward 1.0, but F1 does not use $TN$.
- **B is incorrect:** F1 is computed strictly from the counts $TP, FP, FN$, not from the total background size.
- **C is incorrect:** F1 is completely insulated from changes in the count of true negatives.
- **D is correct:** The formula for F1 is:
  $$F_1 = \frac{2TP}{2TP + FP + FN}$$
  Notice that $TN$ does not appear anywhere in the formula. Because $TP = 100$, $FP = 25$, and $FN = 75$ did not change:
  $$F_1 = \frac{2(100)}{2(100) + 25 + 75} = \frac{200}{300} \approx 0.667$$
  The F1 score remains exactly $0.667$ regardless of whether $TN$ is 9,800 or 99,800.

</details>

---

### 36. Question

An evaluation batch of financial transactions yields:
- $TP = 40$
- $FP = 10$
- $FN = 50$
- $TN = 900$

**What is the F1 Score of this model?**

- [ ] A) 0.571
- [ ] B) 0.800
- [ ] C) 0.444
- [ ] D) 0.622

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.571

**Explanation:**
- **A is correct:**
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{40}{40 + 10} = \frac{40}{50} = 0.800$$
  $$\text{Recall} = \frac{TP}{TP + FN} = \frac{40}{40 + 50} = \frac{40}{90} \approx 0.4444$$
  $$F_1 = 2 \times \frac{0.800 \times 0.4444}{0.800 + 0.4444} = \frac{0.7111}{1.2444} \approx 0.5714 \approx 0.571$$
  Alternatively, using the direct formula:
  $$F_1 = \frac{2TP}{2TP + FP + FN} = \frac{2(40)}{2(40) + 10 + 50} = \frac{80}{80 + 60} = \frac{80}{140} = \frac{4}{7} \approx 0.5714$$
- **B is incorrect:** $0.800$ is Precision alone.
- **C is incorrect:** $0.444$ is Recall alone.
- **D is incorrect:** $0.622$ is the simple arithmetic mean ($\frac{0.800 + 0.444}{2} = 0.622$), which fails to apply the harmonic penalty.

</details>

---

### 37. Question

Two binary classifiers are evaluated on the same test dataset:
- **Classifier 1:** Precision = $0.50$, Recall = $0.50$
- **Classifier 2:** Precision = $0.90$, Recall = $0.10$

Notice that both classifiers have the exact same simple arithmetic mean: $\frac{0.50 + 0.50}{2} = 0.50$ and $\frac{0.90 + 0.10}{2} = 0.50$.

**What are the F1 scores of Classifier 1 and Classifier 2, respectively?**

- [ ] A) Classifier 1: 0.50; Classifier 2: 0.50
- [ ] B) Classifier 1: 0.25; Classifier 2: 0.09
- [ ] C) Classifier 1: 0.50; Classifier 2: 0.18
- [ ] D) Classifier 1: 0.40; Classifier 2: 0.20

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Classifier 1: 0.50; Classifier 2: 0.18

**Explanation:**
- **A is incorrect:** This would assume the arithmetic mean.
- **B is incorrect:** These values are the products $P \times R$, not the harmonic means.
- **C is correct:**
  For Classifier 1:
  $$F_{1, \text{Class 1}} = 2 \times \frac{0.50 \times 0.50}{0.50 + 0.50} = \frac{0.50}{1.00} = 0.50$$
  For Classifier 2:
  $$F_{1, \text{Class 2}} = 2 \times \frac{0.90 \times 0.10}{0.90 + 0.10} = \frac{0.18}{1.00} = 0.18$$
  Despite identical arithmetic means, Classifier 1's F1 score ($0.50$) is nearly three times that of Classifier 2 ($0.18$). This illustrates how severely the F1 score punishes poor recall.
- **D is incorrect:** Arithmetic computation errors.

</details>

---

### 38. Question

Which of the following mathematical inequalities holds true for any non-trivial binary classification model with non-negative Precision ($P$) and Recall ($R$)?

- [ ] A) $F_1 \ge \frac{P + R}{2}$ in all circumstances
- [ ] B) $F_1 \le \frac{P + R}{2}$, with strict equality holding if and only if $P = R$
- [ ] C) $F_1 = \sqrt{P \times R}$
- [ ] D) $F_1 > \max(P, R)$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) $F_1 \le \frac{P + R}{2}$, with strict equality holding if and only if $P = R$

**Explanation:**
- **A is incorrect:** The harmonic mean is strictly bounded above by the arithmetic mean.
- **B is correct:** By the fundamental **AM-GM-HM inequality** (Arithmetic Mean $\ge$ Geometric Mean $\ge$ Harmonic Mean):
  $$\frac{P + R}{2} \ge \sqrt{P \times R} \ge \frac{2PR}{P + R} = F_1$$
  Equality holds across all three means if and only if $P = R$. Whenever $P \ne R$, the F1 score is strictly strictly less than the arithmetic mean.
- **C is incorrect:** $\sqrt{P \times R}$ is the geometric mean (also known as the G-mean / Fowlkes-Mallows index), not the F1 score.
- **D is incorrect:** The harmonic mean is always less than or equal to the maximum of the two numbers.

</details>

---

### 39. Question

In an anomaly-detection pipeline where anomalies constitute only $0.05\%$ of all network traffic, why do engineering teams track the **F1 Score** rather than **Accuracy** as their primary key performance indicator?

- [ ] A) F1 score is unaffected by model calibration errors, whereas accuracy requires perfectly calibrated probabilities
- [ ] B) Accuracy cannot be computed when the dataset has more than 10,000 samples
- [ ] C) F1 score can be optimized using standard convex gradient descent directly in scikit-learn
- [ ] D) F1 focuses exclusively on the positive minority class and cannot be gamed by predicting the dominant negative class

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) F1 focuses exclusively on the positive minority class and cannot be gamed by predicting the dominant negative class

**Explanation:**
- **A is incorrect:** Neither metric requires calibrated probabilities; both operate on discrete thresholded labels.
- **B is incorrect:** Accuracy can be computed on any sample size.
- **C is incorrect:** The F1 score is a non-differentiable step function of classification thresholds and cannot be optimized directly with gradient descent.
- **D is correct:** When the negative class is overwhelming ($99.95\%$), a dummy model predicting all negative achieves $99.95\%$ accuracy but has $TP = 0 \implies F_1 = 0.0$. Because the F1 score uses only $TP, FP, FN$ and completely ignores $TN$, it provides an honest measure of minority anomaly capture that cannot be inflated by background noise.

</details>

---

### 40. Question

A conservative model outputs predictions on a test set. It predicts negative ($0$) for every single observation. Consequently:
- $TP = 0$
- $FP = 0$
- $FN = 50$
- $TN = 950$

In accordance with standard statistical machine learning conventions (such as `zero_division=0` in scikit-learn), **what is the value assigned to the F1 Score?**

- [ ] A) 0.0
- [ ] B) 1.0
- [ ] C) Undefined / NaN
- [ ] D) 0.5

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.0

**Explanation:**
- **A is correct:** When a model makes zero positive predictions, $TP = 0$ and $FP = 0$, leading to a division by zero in Precision ($\frac{0}{0}$). By standard convention in machine learning and scikit-learn (`zero_division=0`), Precision is defined as $0.0$, Recall is $\frac{0}{50} = 0.0$, and the resulting F1 Score is mathematically set to **0.0**.
- **B is incorrect:** A model that fails to catch any positive cases cannot receive a perfect score of 1.0.
- **C is incorrect:** While raw arithmetic results in $0/0$, production libraries and standard evaluation frameworks resolve the boundary condition to 0.0.
- **D is incorrect:** 0.5 would falsely indicate moderate utility.

</details>
