### 1. Question

A machine learning engineer evaluates a customer churn prediction model on a test set of 1,000 subscription accounts. The confusion matrix shows:
- True Positives ($TP$) = 150
- False Positives ($FP$) = 50
- True Negatives ($TN$) = 720
- False Negatives ($FN$) = 80

**What is the overall classification accuracy of the churn model on this evaluation batch?**

- [ ] A) 0.72
- [ ] B) 0.80
- [ ] C) 0.87
- [ ] D) 0.92

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.87

**Explanation:**
- **A is incorrect:** 0.72 represents only the True Negative Rate relative to the whole dataset ($\frac{720}{1000}$), ignoring all correct positive churn predictions.
- **B is incorrect:** 0.80 represents either $\frac{TN + FN}{1000} = \frac{800}{1000}$ or misses 70 correct predictions from arithmetic mistakes.
- **C is correct:** Classification accuracy is the proportion of total correct predictions (both positive and negative) over the total sample size:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{150 + 720}{150 + 50 + 720 + 80} = \frac{870}{1000} = 0.87$$
- **D is incorrect:** 0.92 is obtained by mistakenly calculating $\frac{TN + TP + FP}{1000} = \frac{920}{1000}$, improperly treating false alarms as successes.

</details>

---

### 2. Question

A cybersecurity team deploys an intrusion detection system (IDS) where the positive class ($1$) indicates a malicious network intrusion and the negative class ($0$) indicates benign traffic. A packet stream that was completely benign is mistakenly flagged by the classifier as an active cyberattack, triggering an automated server quarantine.

**Which quadrant of the confusion matrix and type of statistical error does this event represent?**

- [ ] A) False Positive (Type I Error)
- [ ] B) False Negative (Type II Error)
- [ ] C) True Positive (Power)
- [ ] D) True Negative (Specificity)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) False Positive (Type I Error)

**Explanation:**
- **A is correct:** The ground truth was negative (benign traffic), but the model predicted positive (malicious intrusion). An erroneous positive prediction on an actual negative is a False Positive ($FP$), classically defined in statistical hypothesis testing as a Type I Error.
- **B is incorrect:** A False Negative (Type II error) occurs when an actual positive is erroneously classified as negative (e.g., an actual cyberattack slipping past undetected as benign traffic).
- **C is incorrect:** A True Positive occurs when an actual positive is correctly identified as positive (e.g., an actual cyberattack correctly detected).
- **D is incorrect:** A True Negative occurs when an actual negative is correctly recognized as negative (e.g., normal benign traffic allowed through without alert).

</details>

---

### 3. Question

An automated quality-inspection model inspects microchips on an assembly line. A microchip with a microscopic hairline fracture (a defective unit, actual class = 1) is classified by the computer vision model as non-defective (predicted class = 0) and shipped to a customer.

**How is this prediction classified in the confusion matrix?**

- [ ] A) False Positive ($FP$)
- [ ] B) False Negative ($FN$)
- [ ] C) True Negative ($TN$)
- [ ] D) True Positive ($TP$)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) False Negative ($FN$)

**Explanation:**
- **A is incorrect:** A False Positive would mean a flawless chip was incorrectly flagged as defective.
- **B is correct:** The item actually belongs to the positive class (defective), but the model produced a negative prediction (non-defective). This failure to detect a true condition constitutes a False Negative ($FN$), also termed a Type II error or "miss."
- **C is incorrect:** A True Negative occurs when a sound chip is correctly predicted as non-defective.
- **D is incorrect:** A True Positive occurs when a defective chip is correctly caught and flagged as defective.

</details>

---

### 4. Question

A fraud detection dataset contains 10,000 credit card transactions. Due to the extreme rarity of financial fraud, exactly 9,900 transactions are legitimate (negative class) and 100 transactions are fraudulent (positive class). A naive baseline classifier predicts that every single transaction is legitimate without inspecting any features.

**What is the accuracy of this naive model, and what fundamental evaluation issue does this illustrate?**

- [ ] A) Accuracy is 0.01; illustrates the class imbalance penalty
- [ ] B) Accuracy is 0.50; illustrates random chance performance
- [ ] C) Accuracy is 0.90; illustrates precision-recall divergence
- [ ] D) Accuracy is 0.99; illustrates the Accuracy Paradox

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Accuracy is 0.99; illustrates the Accuracy Paradox

**Explanation:**
- **A is incorrect:** 0.01 is the proportion of fraud cases in the dataset (prevalence rate), not the model's accuracy.
- **B is incorrect:** 0.50 would be the expected balanced accuracy or random guess rate on a balanced binary dataset, not the accuracy of an all-negative classifier on this skewed dataset.
- **C is incorrect:** 0.90 is mathematically arbitrary and does not match the actual count of $\frac{9900}{10000}$.
- **D is correct:** The model correctly identifies all 9,900 legitimate transactions as legitimate ($TN = 9,900$) but misses all 100 fraudulent transactions ($FN = 100, TP = 0, FP = 0$). Its accuracy is:
  $$\text{Accuracy} = \frac{0 + 9900}{10000} = 0.99 \text{ (or } 99\%)$$
  Despite achieving 99% accuracy, the classifier is completely useless because its recall for fraud is $0.0$. This failure mode is universally known as the **Accuracy Paradox**.

</details>

---

### 5. Question

In a binary classification setting with classes labeled $0$ and $1$, a team sets up their confusion matrix using scikit-learn's standard `confusion_matrix(y_true, y_pred)` convention.

**Which of the following correctly describes the layout of the returned $2 \times 2$ NumPy array?**

- [ ] A) Row 0: $[TP, FP]$; Row 1: $[FN, TN]$
- [ ] B) Row 0: $[TP, FN]$; Row 1: $[FP, TN]$
- [ ] C) Row 0: $[TN, FP]$; Row 1: $[FN, TP]$
- [ ] D) Row 0: $[FP, TN]$; Row 1: $[TP, FN]$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Row 0: $[TN, FP]$; Row 1: $[FN, TP]$

**Explanation:**
- **A is incorrect:** This places positive actuals in Row 0 and swaps the order of True Positives and Negatives.
- **B is incorrect:** This format places predictions on rows and actuals on columns, reversing scikit-learn's standard indexing.
- **C is correct:** In `sklearn.metrics.confusion_matrix`, rows represent **actual (ground truth)** classes ($0$ then $1$) and columns represent **predicted** classes ($0$ then $1$):
  $$\begin{bmatrix} C_{00} & C_{01} \\ C_{10} & C_{11} \end{bmatrix} = \begin{bmatrix} \text{Actual 0, Pred 0 } (TN) & \text{Actual 0, Pred 1 } (FP) \\ \text{Actual 1, Pred 0 } (FN) & \text{Actual 1, Pred 1 } (TP) \end{bmatrix}$$
- **D is incorrect:** This layout reverses the column order and scrambles the diagonal.

</details>

---

### 6. Question

A diagnostic lab validates an antigen test on 500 clinical blood samples. The matrix of results is:
- True Positives = 180
- False Positives = 20
- True Negatives = 270
- False Negatives = 30

**What is the Miss Rate (False Negative Rate) of this antigen test?**

- [ ] A) 0.143
- [ ] B) 0.069
- [ ] C) 0.100
- [ ] D) 0.060

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.143

**Explanation:**
- **A is correct:** The Miss Rate (False Negative Rate, $FNR$) is the proportion of actual positive cases that were mistakenly missed by the test:
  $$FNR = \frac{FN}{TP + FN} = \frac{30}{180 + 30} = \frac{30}{210} \approx 0.142857 \approx 0.143$$
- **B is incorrect:** $0.069$ is the False Positive Rate ($FPR = \frac{FP}{TN + FP} = \frac{20}{270 + 20} = \frac{20}{290} \approx 0.069$).
- **C is incorrect:** $0.100$ is obtained by dividing $FN$ by the total predictions made as positive ($\frac{30}{180 + 120}$ or similar misaligned denominators).
- **D is incorrect:** $0.060$ is the proportion of false negatives relative to the entire sample size ($\frac{30}{500} = 0.060$).

</details>

---

### 7. Question

A loan default classifier is tested on 400 commercial loan applicants. The model achieves an accuracy of 85%. There were 40 actual defaults ($TP + FN = 40$) and 360 non-defaults ($TN + FP = 360$). If the model successfully flagged 30 of the defaults as defaults ($TP = 30$), how many non-defaults were incorrectly flagged as defaults ($FP$)?

- [ ] A) 10
- [ ] B) 50
- [ ] C) 30
- [ ] D) 20

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 50

**Explanation:**
- **A is incorrect:** 10 is the count of False Negatives ($FN = \text{Actual Defaults} - TP = 40 - 30 = 10$).
- **B is correct:** We can solve for the unknowns step-by-step:
  1. Total correct predictions: $\text{Accuracy} \times \text{Total} = 0.85 \times 400 = 340$.
  2. Correct predictions comprise $TP + TN = 340$.
  3. Since $TP = 30$, we have $TN = 340 - 30 = 310$.
  4. Total actual non-defaults = $TN + FP = 360$.
  5. Therefore, $FP = 360 - TN = 360 - 310 = 50$.
- **C is incorrect:** 30 matches $TP$, not $FP$.
- **D is incorrect:** 20 is derived from subtracting default count from another arbitrary sub-total.

</details>

---

### 8. Question

Suppose two binary classification models, Model Alpha and Model Beta, are evaluated on the exact same dataset of 1,000 cases with 100 positive and 900 negative instances. Both models achieve an identical overall accuracy of 91.0%.
- Model Alpha has $TP = 20, FP = 10$.
- Model Beta has $TP = 80, FP = 70$.

**Which statement accurately compares the diagnostic behavior of these two models?**

- [ ] A) Both models deliver identical utility because their accuracy is identical (91%)
- [ ] B) Model Alpha captured 80% of all positive cases, whereas Model Beta captured only 20%
- [ ] C) Model Alpha has higher sensitivity, whereas Model Beta has higher specificity
- [ ] D) Model Beta identified four times as many true positive cases as Model Alpha, despite having the same accuracy

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Model Beta identified four times as many true positive cases as Model Alpha, despite having the same accuracy

**Explanation:**
- **A is incorrect:** Identical overall accuracy masks radically different operational trade-offs; in high-stakes positive detection (like fraud or disease), Model Beta is vastly more active and catches 80 cases compared to Alpha's 20.
- **B is incorrect:** Model Beta captured $\frac{80}{100} = 80\%$ of positives, while Model Alpha captured only $\frac{20}{100} = 20\%$, reversing the statement.
- **C is incorrect:** Model Beta has higher sensitivity ($\frac{80}{100} = 0.80$ vs. Alpha's $\frac{20}{100} = 0.20$), while Model Alpha has higher specificity ($\frac{890}{900} \approx 0.989$ vs. Beta's $\frac{830}{900} \approx 0.922$).
- **D is correct:** Model Beta achieves $TP = 80$, which is exactly four times the $TP = 20$ achieved by Model Alpha ($80 / 20 = 4$). This highlights why accuracy alone cannot differentiate between a conservative model that rarely predicts positive and an aggressive model that flags many positives.

</details>

---

### 9. Question

A senior auditor analyzes a classification report. The report specifies the following marginal sums:
- Total samples: $N = 500$
- Number of samples predicted positive: $120$
- Number of samples predicted negative: $380$
- Number of actual positive samples: $100$
- Number of actual negative samples: $400$
- True Positives ($TP$) = $70$

**What is the value of True Negatives ($TN$)?**

- [ ] A) 310
- [ ] B) 330
- [ ] C) 350
- [ ] D) 380

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 350

**Explanation:**
- **A is incorrect:** 310 results from subtracting both $TP$ and $FP$ from total actual negatives.
- **B is incorrect:** 330 results from subtracting $TP$ ($70$) from total predicted negatives ($380$ - arithmetic confusion).
- **C is correct:**
  1. We know actual negatives = $TN + FP = 400$.
  2. We know predicted positives = $TP + FP = 120$.
  3. Since $TP = 70$, $FP = 120 - 70 = 50$.
  4. Therefore, $TN = \text{Actual Negatives} - FP = 400 - 50 = 350$.
  Alternatively, predicted negatives = $TN + FN = 380$. Since actual positives = $TP + FN = 100 \implies FN = 30$. Thus $TN = 380 - 30 = 350$.
- **D is incorrect:** 380 is the total count of predicted negatives ($TN + FN$), not $TN$ alone.

</details>

---

### 10. Question

In statistical classification, the total number of errors made by a binary classifier on any evaluation dataset is always strictly equal to which of the following expressions?

- [ ] A) $TP + TN$
- [ ] B) $FP + FN$
- [ ] C) $(TP + FP) - (TN + FN)$
- [ ] D) $\text{Total Samples} - (FP + FN)$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) $FP + FN$

**Explanation:**
- **A is incorrect:** $TP + TN$ is the total count of correct predictions.
- **B is correct:** Every prediction made by a classifier is either correct ($TP$ or $TN$) or incorrect ($FP$ or $FN$). Therefore, the total count of misclassifications (errors) is strictly the sum of False Positives (Type I errors) and False Negatives (Type II errors):
  $$\text{Total Errors} = FP + FN = \text{Total Samples} - (TP + TN)$$
- **C is incorrect:** This calculates the difference between total predicted positives and total predicted negatives, which can be zero, positive, or negative.
- **D is incorrect:** $\text{Total Samples} - (FP + FN)$ equals $TP + TN$ (the total correct predictions), not the total errors.

</details>
