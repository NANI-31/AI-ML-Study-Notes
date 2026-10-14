### 81. Question

In a multiclass classification problem with $K = 4$ classes, the confusion matrix is represented as a $4 \times 4$ contingency grid where rows correspond to true labels and columns correspond to predicted labels.

**What do the diagonal elements ($C_{00}, C_{11}, C_{22}, C_{33}$) of this matrix represent?**

- [ ] A) The False Positive counts for each respective class
- [ ] B) The False Negative counts for each respective class
- [ ] C) The True Positive counts (correctly classified samples) for each respective class
- [ ] D) The total support (sample size) of each respective class

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The True Positive counts (correctly classified samples) for each respective class

**Explanation:**
- **A is incorrect:** False Positives for class $k$ are located in the off-diagonal entries of column $k$ (sum of column $k$ excluding $C_{kk}$).
- **B is incorrect:** False Negatives for class $k$ are located in the off-diagonal entries of row $k$ (sum of row $k$ excluding $C_{kk}$).
- **C is correct:** By definition, an entry $C_{ij}$ represents samples where actual class is $i$ and predicted class is $j$. The diagonal entries where $i = j$ ($C_{00}, C_{11}, C_{22}, C_{33}$) represent instances where the model's prediction matched the true class perfectly—the **True Positives** for each individual class. The trace of the matrix ($\sum C_{ii}$) equals the total number of correct predictions.
- **D is incorrect:** The support of class $i$ is the row sum $\sum_j C_{ij}$, not the diagonal entry alone.

</details>

---

### 82. Question

In scikit-learn's `f1_score(y_true, y_pred, average='macro')`, **how is the Macro-average F1 score calculated?**

- [ ] A) By aggregating total TP, total FP, and total FN globally across all classes before calculating F1
- [ ] B) By computing the F1 score independently for each class and calculating their unweighted arithmetic mean
- [ ] C) By computing the F1 score for each class and weighting each score by the class's support
- [ ] D) By selecting the maximum F1 score achieved among the classes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) By computing the F1 score independently for each class and calculating their unweighted arithmetic mean

**Explanation:**
- **A is incorrect:** This defines **Micro-averaging** (`average='micro'`).
- **B is correct:** **Macro-averaging** computes the evaluation metric independently for every class $k \in \{1, \dots, K\}$ and then computes their unweighted arithmetic average:
  $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F_1^{(k)}$$
  Crucially, Macro-averaging treats **every class equally**, making it highly sensitive to poor performance on small minority classes.
- **C is incorrect:** This defines **Weighted-averaging** (`average='weighted'`).
- **D is incorrect:** Macro-averaging computes a mean, not a maximum.

</details>

---

### 83. Question

In scikit-learn's `f1_score(y_true, y_pred, average='weighted')`, **how is the Weighted-average F1 score calculated?**

- [ ] A) By weighting each class's F1 score inversely to its class frequency
- [ ] B) By calculating the geometric mean of per-class F1 scores
- [ ] C) By assigning a manual weight multiplier determined by cross-validation
- [ ] D) By computing the F1 score independently for each class and weighting each by its support (true sample count)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) By computing the F1 score independently for each class and weighting each by its support (true sample count)

**Explanation:**
- **A is incorrect:** Weighted averaging weights proportionally to class frequency, not inversely.
- **B is incorrect:** It computes an arithmetic weighted mean, not a geometric mean.
- **C is incorrect:** Weights are determined strictly by empirical class sample counts in `y_true`, not cross-validation hyperparameters.
- **D is correct:** **Weighted-averaging** weights each class's metric by its representation (support $N_k$) in the true dataset:
  $$\text{Weighted-F1} = \sum_{k=1}^K \left( \frac{N_k}{N_{\text{total}}} \right) F_1^{(k)}$$
  This accounts for class imbalance, ensuring that larger classes contribute proportionally more to the overall scalar metric.

</details>

---

### 84. Question

In standard single-label multiclass classification, which remarkable mathematical identity always holds for **Micro-averaging** (`average='micro'`)?

- [ ] A) $\text{Micro-Precision} = \text{Micro-Recall} = \text{Micro-F1} = \text{Overall Accuracy}$
- [ ] B) $\text{Micro-F1} = \text{Macro-F1}$
- [ ] C) $\text{Micro-Precision} + \text{Micro-Recall} = 1.0$
- [ ] D) $\text{Micro-F1} \le \text{Macro-F1}$ always

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) $\text{Micro-Precision} = \text{Micro-Recall} = \text{Micro-F1} = \text{Overall Accuracy}$

**Explanation:**
- **A is correct:** In single-label multiclass classification, every single misclassification by the model is simultaneously a False Positive for the predicted class and a False Negative for the true class. Therefore, when summing across all classes:
  $$\sum_k FP_k = \sum_k FN_k$$
  Because total false positives equal total false negatives, Micro-Precision ($\frac{\sum TP}{\sum TP + \sum FP}$) and Micro-Recall ($\frac{\sum TP}{\sum TP + \sum FN}$) are strictly equal. Their harmonic mean (Micro-F1) is also identical, and each equals $\frac{\sum TP}{N_{\text{total}}}$, which is the **Overall Accuracy**.
- **B is incorrect:** Micro-F1 and Macro-F1 diverge whenever classes are imbalanced and perform differently.
- **C is incorrect:** Precision and Recall are not complements; they are identical under micro-averaging.
- **D is incorrect:** Micro-F1 can be greater than, equal to, or less than Macro-F1.

</details>

---

### 85. Question

A 3-class customer intent model achieves the following per-class F1 scores and sample counts on a test set:
- **Class A (General Inquiry):** $F_1 = 0.90$, Support = $800$
- **Class B (Technical Issue):** $F_1 = 0.80$, Support = $150$
- **Class C (Billing Dispute):** $F_1 = 0.10$, Support = $50$

**What is the Macro-average F1 score of this model?**

- [ ] A) 0.845
- [ ] B) 0.750
- [ ] C) 0.600
- [ ] D) 0.900

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.600

**Explanation:**
- **A is incorrect:** $0.845$ is the Weighted-average F1 score.
- **B is incorrect:** $0.750$ ignores the third class and averages only Classes A and B ($\frac{0.90 + 0.80}{2} = 0.85$).
- **C is correct:** The Macro-average F1 score is the simple unweighted arithmetic mean of the per-class F1 scores across all $K = 3$ classes:
  $$\text{Macro-F1} = \frac{F_{1, A} + F_{1, B} + F_{1, C}}{3} = \frac{0.90 + 0.80 + 0.10}{3} = \frac{1.80}{3} = 0.600$$
  Notice how the disastrous performance on minority Class C ($0.10$) pulls the overall Macro-F1 down to $0.600$, properly alerting engineers to the defect.
- **D is incorrect:** 0.900 is the score of Class A alone.

</details>

---

### 86. Question

For the exact same 3-class model from Question 85:
- **Class A:** $F_1 = 0.90$, Support = $800$
- **Class B:** $F_1 = 0.80$, Support = $150$
- **Class C:** $F_1 = 0.10$, Support = $50$
- **Total Support:** $800 + 150 + 50 = 1,000$

**What is the Weighted-average F1 score of this model?**

- [ ] A) 0.600
- [ ] B) 0.845
- [ ] C) 0.800
- [ ] D) 0.900

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.845

**Explanation:**
- **A is incorrect:** $0.600$ is the Macro-average F1 score.
- **B is correct:** The Weighted-average F1 weights each class's F1 score by its proportion of the total test set:
  $$\text{Weighted-F1} = \left(\frac{800}{1000} \times 0.90\right) + \left(\frac{150}{1000} \times 0.80\right) + \left(\frac{50}{1000} \times 0.10\right)$$
  $$\text{Weighted-F1} = (0.80 \times 0.90) + (0.15 \times 0.80) + (0.05 \times 0.10) = 0.720 + 0.120 + 0.005 = 0.845$$
  Because majority Class A dominates $80\%$ of the data, the weighted score ($0.845$) masks the near-total failure of the model on minority Class C ($F_1 = 0.10$).
- **C is incorrect:** $0.800$ is the score of Class B.
- **D is incorrect:** $0.900$ is the score of Class A.

</details>

---

### 87. Question

An oncology research institute is classifying biopsy tissue into three categories: Normal ($N = 10,000$), Benign Tumor ($N = 2,000$), and Rare Aggressive Carcinoma ($N = 50$).

**Which averaging metric must the clinical validation committee inspect to ensure the model has not completely failed on the rare aggressive carcinoma?**

- [ ] A) Weighted-average F1, because it reflects population prevalence
- [ ] B) Micro-average F1, because it is identical to overall accuracy
- [ ] C) Overall Accuracy, because high accuracy guarantees high reliability
- [ ] D) Macro-average F1, because it weights all three categories equally regardless of sample size

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Macro-average F1, because it weights all three categories equally regardless of sample size

**Explanation:**
- **A is incorrect:** Weighted-average F1 will give only $\frac{50}{12050} \approx 0.4\%$ weight to the carcinoma class, masking a complete failure.
- **B is incorrect:** Micro-average F1 equals overall accuracy, which can be $99.6\%$ even if the model predicts zero carcinomas.
- **C is incorrect:** Accuracy suffers from the Accuracy Paradox in extreme imbalance.
- **D is correct:** In **Macro-averaging**, the Rare Carcinoma class accounts for exactly $\frac{1}{3}$ ($33.3\%$) of the reported score. If the model achieves zero recall on carcinoma ($F_1 = 0.0$), the Macro-F1 cannot exceed $\frac{1.0 + 1.0 + 0.0}{3} \approx 0.67$, immediately exposing the clinical deficiency to auditors.

</details>

---

### 88. Question

A 3-class confusion matrix is evaluated for a vehicle classification system (Class 0 = Sedan, Class 1 = SUV, Class 2 = Truck). Rows indicate actual classes and columns indicate predicted classes:

$$\begin{bmatrix}
50 & 5 & 5 \\
10 & 40 & 10 \\
5 & 15 & 60
\end{bmatrix}$$

**What is the Precision for Class 0 (Sedan)?**

- [ ] A) 0.769
- [ ] B) 0.833
- [ ] C) 0.500
- [ ] D) 0.714

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.769

**Explanation:**
- **A is correct:** For Class 0:
  - True Positives ($TP_0$) = Entry $C_{00} = 50$.
  - False Positives ($FP_0$) = Sum of column 0 excluding $C_{00}$ (samples predicted as Class 0 that were actually other classes) = $C_{10} + C_{20} = 10 + 5 = 15$.
  $$\text{Precision}_0 = \frac{TP_0}{TP_0 + FP_0} = \frac{50}{50 + 15} = \frac{50}{65} \approx 0.7692 \approx 0.769$$
- **B is incorrect:** $0.833$ is the Recall for Class 0 ($\frac{50}{50 + 5 + 5} = \frac{50}{60} \approx 0.833$).
- **C is incorrect:** $0.500$ is raw $TP_0 / 100$.
- **D is incorrect:** Misaggregates the column elements.

</details>

---

### 89. Question

Using the exact same confusion matrix from Question 88:

$$\begin{bmatrix}
50 & 5 & 5 \\
10 & 40 & 10 \\
5 & 15 & 60
\end{bmatrix}$$

**What is the Recall (Sensitivity) for Class 0 (Sedan)?**

- [ ] A) 0.769
- [ ] B) 0.600
- [ ] C) 0.833
- [ ] D) 0.714

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.833

**Explanation:**
- **A is incorrect:** $0.769$ is the Precision for Class 0 ($\frac{50}{65}$).
- **B is incorrect:** $0.600$ is the total count of actual Class 0 samples ($50 + 5 + 5 = 60$).
- **C is correct:** For Class 0:
  - True Positives ($TP_0$) = Entry $C_{00} = 50$.
  - False Negatives ($FN_0$) = Sum of row 0 excluding $C_{00}$ (actual Class 0 samples incorrectly predicted as other classes) = $C_{01} + C_{02} = 5 + 5 = 10$.
  - Total actual Class 0 samples = $TP_0 + FN_0 = 50 + 10 = 60$.
  $$\text{Recall}_0 = \frac{TP_0}{TP_0 + FN_0} = \frac{50}{60} \approx 0.8333 \approx 0.833$$
- **D is incorrect:** Incorrect denominator aggregation.

</details>

---

### 90. Question

Why does computing `precision_score(y_true, y_pred, average='micro')` produce an identical numerical result to `accuracy_score(y_true, y_pred)` in scikit-learn?

- [ ] A) Because scikit-learn uses accuracy as a fast computational fallback when micro-average is called
- [ ] B) Because in multiclass classification, every prediction error is simultaneously counted as a False Positive for the predicted class and a False Negative for the true class, making total FP equal to total FN
- [ ] C) Because Micro-averaging discards all False Positives from the denominator
- [ ] D) Because micro-averaging only evaluates binary datasets

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Because in multiclass classification, every prediction error is simultaneously counted as a False Positive for the predicted class and a False Negative for the true class, making total FP equal to total FN

**Explanation:**
- **A is incorrect:** It is not a fallback; it is a direct mathematical consequence of the metric definition.
- **B is correct:** In any multiclass prediction where each sample receives exactly one predicted class label:
  1. If a sample of true class $A$ is incorrectly predicted as class $B$, class $B$ gets one $FP$ and class $A$ gets one $FN$.
  2. Therefore, summing across all classes, the total number of False Positives globally equals the total number of False Negatives globally: $\sum FP = \sum FN = \text{Total Errors}$.
  3. The micro-precision formula is:
     $$\text{Micro-Precision} = \frac{\sum TP}{\sum TP + \sum FP} = \frac{\text{Total Correct}}{\text{Total Correct} + \text{Total Errors}} = \frac{\text{Total Correct}}{\text{Total Samples}} = \text{Accuracy}$$
- **C is incorrect:** False Positives are fully included in the micro-precision formula.
- **D is incorrect:** Micro-averaging is primarily used for multiclass and multilabel problems.

</details>
