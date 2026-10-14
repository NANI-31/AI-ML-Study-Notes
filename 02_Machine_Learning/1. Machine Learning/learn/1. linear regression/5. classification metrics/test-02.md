### 11. Question

A production email spam filter processes a test set of 1,000 incoming messages. The classifier yields the following confusion matrix results:
- True Positives ($TP$) = 80
- False Positives ($FP$) = 20
- True Negatives ($TN$) = 850
- False Negatives ($FN$) = 50

**What is the Precision of this spam classifier?**

- [ ] A) 0.80
- [ ] B) 0.615
- [ ] C) 0.977
- [ ] D) 0.850

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.80

**Explanation:**
- **A is correct:** Precision (also called Positive Predictive Value) is defined as the fraction of positive predictions that are truly positive:
  $$\text{Precision} = \frac{TP}{TP + FP} = \frac{80}{80 + 20} = \frac{80}{100} = 0.80$$
  This means that when the filter flags an email as spam, it is correct 80% of the time.
- **B is incorrect:** $0.615$ is the Recall of the model ($\frac{TP}{TP + FN} = \frac{80}{80 + 50} = \frac{80}{130} \approx 0.615$).
- **C is incorrect:** $0.977$ is the Specificity of the model ($\frac{TN}{TN + FP} = \frac{850}{850 + 20} = \frac{850}{870} \approx 0.977$).
- **D is incorrect:** $0.850$ represents the ratio of True Negatives to total emails ($\frac{850}{1000}$).

</details>

---

### 12. Question

In clinical trial epidemiology and binary diagnostic testing, the metric **Specificity** evaluates which of the following performance characteristics?

- [ ] A) The proportion of individuals who tested positive and actually carry the target condition
- [ ] B) The proportion of all subjects in the entire cohort whose disease status was accurately diagnosed
- [ ] C) The probability that an infected patient will be detected by a positive test result
- [ ] D) The proportion of healthy (disease-free) individuals who are correctly identified as negative

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The proportion of healthy (disease-free) individuals who are correctly identified as negative

**Explanation:**
- **A is incorrect:** This defines Precision or Positive Predictive Value ($PPV = \frac{TP}{TP + FP}$).
- **B is incorrect:** This defines overall Accuracy ($\frac{TP + TN}{\text{Total}}$).
- **C is incorrect:** This defines Sensitivity or Recall ($TPR = \frac{TP}{TP + FN}$).
- **D is correct:** Specificity, also referred to as the True Negative Rate ($TNR$), measures the ability of a test to correctly identify non-diseased individuals:
  $$\text{Specificity} = \frac{TN}{TN + FP}$$
  A high-specificity test rarely gives a false positive result on a healthy patient.

</details>

---

### 13. Question

A machine learning team is debating whether their new production model should be optimized to maximize Precision rather than Recall.

**In which of the following operational scenarios is maximizing Precision the most critical objective?**

- [ ] A) An airport baggage scanner detecting concealed weapons and explosives
- [ ] B) An automated fraud system that instantly freezes user bank accounts with zero human review
- [ ] C) An automated early-screening model for a fatal but treatable infectious disease
- [ ] D) A factory sensor predicting catastrophic turbine rotor failure during flight

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An automated fraud system that instantly freezes user bank accounts with zero human review

**Explanation:**
- **A is incorrect:** Missing a concealed weapon (a False Negative) is catastrophic for public safety; thus, Recall must be prioritized.
- **B is correct:** When an automated system freezes a bank account without human intervention, a False Positive means an innocent customer is locked out of their funds, leading to extreme customer churn, legal liability, and brand damage. Therefore, the system must have very high Precision to ensure that when it freezes an account, it is almost certainly fraudulent.
- **C is incorrect:** Missing a patient with a fatal disease (a False Negative) could lead to preventable death; Recall is the primary metric in medical screening.
- **D is incorrect:** A turbine failure in flight represents catastrophic loss of life; missing a defective rotor (False Negative) is unacceptable, necessitating high Recall.

</details>

---

### 14. Question

An anti-malware system scans 2,000 clean software executables. It achieves a Specificity (True Negative Rate) of 96.0%.

**How many clean executable files were incorrectly flagged as malware (False Positives)?**

- [ ] A) 40
- [ ] B) 160
- [ ] C) 80
- [ ] D) 20

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 80

**Explanation:**
- **A is incorrect:** 40 is $2,000 \times 0.02$, which would assume a 98% specificity.
- **B is incorrect:** 160 is $2,000 \times 0.08$, corresponding to 92% specificity.
- **C is correct:**
  1. The total count of clean (actual negative) files is $N = TN + FP = 2,000$.
  2. $\text{Specificity} = \frac{TN}{TN + FP} = 0.96 \implies TN = 0.96 \times 2,000 = 1,920$.
  3. The number of False Positives is:
     $$FP = \text{Total Negatives} - TN = 2,000 - 1,920 = 80$$
  Equivalently, the False Positive Rate is $1 - 0.96 = 0.04$, and $2,000 \times 0.04 = 80$.
- **D is incorrect:** 20 is the error count if the total sample was 500, not 2,000.

</details>

---

### 15. Question

In statistical auditing and genomics, the **False Discovery Rate (FDR)** is the complementary counterpart to Precision. A machine learning model achieves a Precision of $0.82$.

**What is the False Discovery Rate of this model?**

- [ ] A) 0.18
- [ ] B) 0.82
- [ ] C) 0.09
- [ ] D) 0.22

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 0.18

**Explanation:**
- **A is correct:** By mathematical definition, the False Discovery Rate ($FDR$) is the proportion of positive predictions that are false positives:
  $$FDR = \frac{FP}{TP + FP} = 1 - \frac{TP}{TP + FP} = 1 - \text{Precision}$$
  $$FDR = 1 - 0.82 = 0.18$$
- **B is incorrect:** 0.82 is the Precision itself.
- **C is incorrect:** 0.09 is half of the actual complement.
- **D is incorrect:** 0.22 is a basic arithmetic subtraction error ($1 - 0.82 \ne 0.22$).

</details>

---

### 16. Question

A credit risk model achieves a Specificity of $0.935$ on an external validation set of legitimate credit applications.

**What is the False Positive Rate ($FPR$) of the model on this validation set?**

- [ ] A) 0.935
- [ ] B) 0.135
- [ ] C) 0.070
- [ ] D) 0.065

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 0.065

**Explanation:**
- **A is incorrect:** 0.935 is the Specificity (True Negative Rate).
- **B is incorrect:** 0.135 is an arithmetic error ($1.0 - 0.935 \ne 0.135$).
- **C is incorrect:** 0.070 is an imprecise rounding that ignores the 0.005 component.
- **D is correct:** The False Positive Rate (FPR, or fall-out) represents the proportion of actual negative instances that are incorrectly categorized as positive:
  $$FPR = \frac{FP}{TN + FP} = 1 - \frac{TN}{TN + FP} = 1 - \text{Specificity}$$
  $$FPR = 1 - 0.935 = 0.065$$

</details>

---

### 17. Question

Two candidate content-moderation models are evaluated on 10,000 social media posts:
- **Model X:** Flags 500 posts as toxic; 450 of them are actually toxic ($TP = 450, FP = 50$).
- **Model Y:** Flags 1,200 posts as toxic; 600 of them are actually toxic ($TP = 600, FP = 600$).

**Which statement accurately compares the Precision of these two models?**

- [ ] A) Model Y has higher precision because it detected 600 toxic posts compared to only 450 by Model X
- [ ] B) Model X has substantially higher precision (0.90) than Model Y (0.50), generating far fewer false alarms per alert
- [ ] C) Both models have equal precision because both captured more than 400 true positive instances
- [ ] D) Model Y has precision of 0.80, while Model X has precision of 0.70

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Model X has substantially higher precision (0.90) than Model Y (0.50), generating far fewer false alarms per alert

**Explanation:**
- **A is incorrect:** Finding more total true positives reflects higher Recall or count, not higher Precision.
- **B is correct:** Precision evaluates the purity of the positive alerts:
  $$\text{Precision}_X = \frac{450}{450 + 50} = \frac{450}{500} = 0.90$$
  $$\text{Precision}_Y = \frac{600}{600 + 600} = \frac{600}{1200} = 0.50$$
  Model X achieves 90% precision versus Model Y's 50%. Model X creates far fewer false accusations.
- **C is incorrect:** Precision is a ratio of positive predictive success, not an absolute raw count threshold.
- **D is incorrect:** The calculated values are 0.90 and 0.50, not 0.70 and 0.80.

</details>

---

### 18. Question

A validation trial for an algorithmic medical triage tool tests 500 patient records and produces:
- $TP = 90$
- $FP = 40$
- $TN = 360$
- $FN = 10$

**What is the Specificity of the triage tool?**

- [ ] A) 0.692
- [ ] B) 0.900
- [ ] C) 0.900
- [ ] D) 0.720

*(Note: Ensure option text precision)*

Wait, let's format distinct option values:
- [ ] A) 0.692
- [ ] B) 0.820
- [ ] C) 0.900
- [ ] D) 0.973

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.900

**Explanation:**
- **A is incorrect:** $0.692$ is the Precision ($\frac{90}{90 + 40} = \frac{90}{130} \approx 0.692$).
- **B is incorrect:** $0.820$ represents an improper combination of categories.
- **C is correct:** Specificity is the ratio of true negatives to actual negatives:
  $$\text{Specificity} = \frac{TN}{TN + FP} = \frac{360}{360 + 40} = \frac{360}{400} = 0.900 \text{ (or } 90.0\%)$$
- **D is incorrect:** $0.973$ is obtained by mistakenly calculating $\frac{TN}{TN + FN} = \frac{360}{370} \approx 0.973$ (the Negative Predictive Value).

</details>

---

### 19. Question

A data scientist adds 500,000 easily classified negative examples (benign non-spam emails from verified domains) to a spam evaluation test set. The model correctly classifies all 500,000 as negative ($TN$ increases by 500,000), while the counts of $TP$, $FP$, and $FN$ remain completely unchanged.

**How does this massive addition of true negatives affect the model's Precision and Accuracy?**

- [ ] A) Precision remains completely unchanged, while Accuracy increases significantly toward 1.0
- [ ] B) Precision increases dramatically toward 1.0, while Accuracy remains completely unchanged
- [ ] C) Both Precision and Accuracy increase dramatically toward 1.0
- [ ] D) Both Precision and Accuracy remain completely unchanged

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Precision remains completely unchanged, while Accuracy increases significantly toward 1.0

**Explanation:**
- **A is correct:**
  1. Precision is calculated as $\frac{TP}{TP + FP}$. Since neither $TP$ nor $FP$ changed, Precision is mathematically invariant to the number of True Negatives.
  2. Accuracy is calculated as $\frac{TP + TN}{\text{Total}}$. Since $TN$ and $\text{Total}$ both increased by 500,000, the ratio $\frac{TP + TN + 500000}{\text{Total} + 500000}$ approaches $1.0$.
- **B is incorrect:** Precision formula contains no $TN$ term in either numerator or denominator.
- **C is incorrect:** Precision cannot increase because true negative additions do not enter the precision calculation.
- **D is incorrect:** Accuracy changes whenever correct predictions are added to the evaluation set.

</details>

---

### 20. Question

English jurist William Blackstone formulated the famous legal principle: *"Better that ten guilty persons escape than that one innocent suffer."*

**From a machine learning metric perspective, designing a judicial sentencing support algorithm to strictly uphold Blackstone's formulation requires prioritizing which metric?**

- [ ] A) Accuracy
- [ ] B) Recall
- [ ] C) Log Loss
- [ ] D) Precision

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Precision

**Explanation:**
- **A is incorrect:** High accuracy can still convict an unacceptable number of innocent people if the overwhelming majority of suspects are guilty.
- **B is incorrect:** Maximizing Recall prioritizes capturing every single guilty person ($TP$), which inevitably increases False Positives (wrongful convictions of innocent people)—the exact opposite of Blackstone's principle.
- **C is incorrect:** Log Loss is a probability calibration loss function, not an alignment metric for asymmetric decision error costs.
- **D is correct:** Blackstone's principle states that the cost of a False Positive (convicting an innocent person) is vastly higher than the cost of a False Negative (letting a guilty person escape). To ensure that virtually every person declared guilty is truly guilty, the model must maximize **Precision** ($\frac{TP}{TP + FP}$), keeping False Positives as close to zero as possible.

</details>
