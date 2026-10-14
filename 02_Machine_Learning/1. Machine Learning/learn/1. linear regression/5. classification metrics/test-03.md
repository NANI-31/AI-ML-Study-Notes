### 21. Question

In clinical diagnostics and statistical machine learning, **Sensitivity** (also referred to as **Recall** or **True Positive Rate**) measures which of the following properties?

- [ ] A) The proportion of positive predictions made by the model that turned out to be correct
- [ ] B) The proportion of actual positive cases in the dataset that were correctly identified by the model
- [ ] C) The ratio of true positives to false positives generated across all thresholds
- [ ] D) The total percentage of samples in the cohort that received an accurate classification

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The proportion of actual positive cases in the dataset that were correctly identified by the model

**Explanation:**
- **A is incorrect:** This defines Precision ($\frac{TP}{TP + FP}$).
- **B is correct:** Recall (Sensitivity or True Positive Rate, $TPR$) measures the model's coverage of ground-truth positive cases:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  It directly answers: "Out of all actual positive cases in existence, what fraction did the model successfully catch?"
- **C is incorrect:** This is the ratio $\frac{TP}{FP}$, which is an odds ratio rather than a standardized rate.
- **D is incorrect:** This defines overall Accuracy ($\frac{TP + TN}{\text{Total}}$).

</details>

---

### 22. Question

A hospital oncology department tests an AI model designed to detect malignant lung nodules from CT scans. Evaluation on 1,000 scans yields:
- $TP = 120$
- $FP = 10$
- $TN = 840$
- $FN = 30$

**What is the Recall (Sensitivity) of this lung nodule detection model?**

- [ ] A) 0.923
- [ ] B) 0.880
- [ ] C) 0.800
- [ ] D) 0.857

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.800

**Explanation:**
- **A is incorrect:** $0.923$ is the Precision ($\frac{120}{120 + 10} = \frac{120}{130} \approx 0.923$).
- **B is incorrect:** $0.880$ is obtained by an incorrect combination of terms.
- **C is correct:** Recall is calculated as:
  $$\text{Recall} = \frac{TP}{TP + FN} = \frac{120}{120 + 30} = \frac{120}{150} = 0.80 \text{ (or } 80.0\%)$$
- **D is incorrect:** $0.857$ is the F1 score ($2 \times \frac{0.923 \times 0.800}{0.923 + 0.800} \approx 0.857$).

</details>

---

### 23. Question

A data science team is deploying models across multiple domain areas.

**In which of the following applications is maximizing Recall strictly paramount, even if it leads to a noticeable decline in Precision?**

- [ ] A) An earthquake early-warning system detecting imminent seismic tremors to halt high-speed trains
- [ ] B) A music streaming recommendation system suggesting recommended songs in a daily mix
- [ ] C) An automated photo tagging feature identifying friends in holiday pictures
- [ ] D) A targeted advertising model selecting consumers to receive a discount coupon

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) An earthquake early-warning system detecting imminent seismic tremors to halt high-speed trains

**Explanation:**
- **A is correct:** Failing to detect an actual earthquake (a False Negative) leads to train derailments, catastrophic infrastructure destruction, and loss of life. False alarms (False Positives) cause temporary travel delays. Because the cost of a False Negative is catastrophic, Recall must be prioritized near 100%.
- **B is incorrect:** In entertainment recommendations, a bad song recommendation (False Positive) is mildly annoying, but missing a song (False Negative) has zero life-or-death consequence. Precision is prioritized to keep the user engaged.
- **C is incorrect:** In photo tagging, misidentifying a person (False Positive) is visually noticeable and awkward; high precision is preferred.
- **D is incorrect:** Advertising budgets are limited; marketing teams prefer high precision to avoid wasting ad spend on non-buyers.

</details>

---

### 24. Question

An epidemiologist reports that a newly developed rapid antigen test exhibits a Sensitivity of $0.940$ ($94.0\%$).

**What is the Miss Rate (False Negative Rate, $FNR$) of this diagnostic test?**

- [ ] A) 0.940
- [ ] B) 0.120
- [ ] C) 0.080
- [ ] D) 0.060

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 0.060

**Explanation:**
- **A is incorrect:** 0.940 is the Sensitivity itself.
- **B is incorrect:** 0.120 is double the true complement.
- **C is incorrect:** 0.080 is an arithmetic subtraction error.
- **D is correct:** Sensitivity ($TPR$) and Miss Rate ($FNR$) are exact mathematical complements because they divide the actual positive class into identified positives and missed positives:
  $$FNR = \frac{FN}{TP + FN} = 1 - \frac{TP}{TP + FN} = 1 - \text{Sensitivity}$$
  $$FNR = 1.0 - 0.940 = 0.060 \text{ (or } 6.0\%)$$

</details>

---

### 25. Question

A data analyst changes the classification threshold of a model, which introduces 100 additional False Positives ($FP$). However, the counts of True Positives ($TP$) and False Negatives ($FN$) remain completely unchanged.

**How does this change affect the model's Recall?**

- [ ] A) Recall decreases substantially because more errors were introduced
- [ ] B) Recall remains completely unchanged
- [ ] C) Recall increases because the total positive predictions increased
- [ ] D) Recall drops to zero

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Recall remains completely unchanged

**Explanation:**
- **A is incorrect:** While total error count increased, Recall specifically ignores False Positives.
- **B is correct:** Recall is strictly defined as $\frac{TP}{TP + FN}$. The formula depends exclusively on the actual positive class instances. Because neither $TP$ nor $FN$ changed, Recall is mathematically invariant to the count of False Positives ($FP$).
- **C is incorrect:** Total predicted positives ($TP + FP$) increased, which lowers Precision, but Recall only evaluates coverage of actual positives.
- **D is incorrect:** Recall cannot drop to zero unless $TP$ drops to zero.

</details>

---

### 26. Question

An industrial computer vision model inspects 50 manufactured engine components that have verified structural micro-cracks ($TP + FN = 50$). The model correctly flags 45 of these defective components, while 5 slip through unnoticed as normal components.

**What is the model's defect-detection Recall?**

- [ ] A) 0.450
- [ ] B) 0.500
- [ ] C) 0.900
- [ ] D) 0.950

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 0.900

**Explanation:**
- **A is incorrect:** 0.450 is $\frac{45}{100}$, assuming an arbitrary denominator of 100 instead of the actual positive count of 50.
- **B is incorrect:** 0.500 reflects the total positive count, not the detection rate.
- **C is correct:** Defect Recall is the fraction of actual defective units identified:
  $$\text{Recall} = \frac{TP}{TP + FN} = \frac{45}{45 + 5} = \frac{45}{50} = 0.900 \text{ (or } 90.0\%)$$
- **D is incorrect:** 0.950 would require catching 47.5 components out of 50.

</details>

---

### 27. Question

A national healthcare system designs a two-stage medical screening pipeline for a serious disease:
- **Stage 1:** A fast, low-cost, non-invasive digital screening test applied to the entire population.
- **Stage 2:** An expensive, invasive confirmatory biopsy given only to individuals who test positive in Stage 1.

**What primary metric should Stage 1 be optimized for to ensure the pipeline is effective?**

- [ ] A) High Recall, to ensure virtually zero diseased individuals are missed in the initial screening
- [ ] B) High Precision, to guarantee that no healthy patient ever receives a follow-up test
- [ ] C) High Specificity, to minimize the total number of people screened
- [ ] D) Low Sensitivity, to reduce the overall operational workload of Stage 2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) High Recall, to ensure virtually zero diseased individuals are missed in the initial screening

**Explanation:**
- **A is correct:** In a multi-stage screening architecture, Stage 1 serves as a wide funnel. Any diseased individual missed at Stage 1 (a False Negative) is lost from the system and will not receive treatment. Therefore, Stage 1 must maximize **Recall (Sensitivity)**. The subsequent confirmatory test (Stage 2) provides the necessary Precision by filtering out any false positives generated by Stage 1.
- **B is incorrect:** Demanding high precision in Stage 1 would require a conservative threshold that inevitably produces False Negatives (missed sick patients).
- **C is incorrect:** Maximizing specificity alone would prioritize ruling out healthy people at the expense of missing active cases.
- **D is incorrect:** Low sensitivity means high false negatives, defeating the purpose of population screening.

</details>

---

### 28. Question

A credit card fraud model outputs continuous risk probabilities $P(y=1|x) \in [0, 1]$. An engineer lowers the decision cutoff from $\tau = 0.50$ to $\tau = 0.15$.

**What is the guaranteed mathematical effect of this threshold reduction on the model's Recall?**

- [ ] A) Recall must decrease because the model becomes less selective
- [ ] B) Recall fluctuates randomly with no predictable direction
- [ ] C) Recall remains strictly identical regardless of threshold shifts
- [ ] D) Recall will either increase or remain unchanged; it can never decrease

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Recall will either increase or remain unchanged; it can never decrease

**Explanation:**
- **A is incorrect:** Lowering the threshold classifies more instances as positive, which can only capture more (or the same number of) true positives.
- **B is incorrect:** The relationship between decision threshold and recall is monotonically non-increasing; it is not random.
- **C is incorrect:** Recall will change if there are any actual positive samples whose predicted probabilities lie between the old threshold ($0.50$) and the new threshold ($0.15$).
- **D is correct:** When the cutoff $\tau$ is lowered, the decision rule $\hat{y} = \mathbb{I}(P \ge \tau)$ becomes more permissive. Any sample that was previously predicted positive remains positive, and additional samples whose probabilities fall in $[0.15, 0.50)$ are now predicted positive. Because $TP$ can only increase or stay constant, and actual positives ($TP + FN$) is fixed, Recall $\frac{TP}{\text{Actual Positives}}$ is a **monotonically non-increasing function of the threshold**. Therefore, lowering $\tau$ can never decrease Recall.

</details>

---

### 29. Question

An audit of a credit bureau's bankruptcy prediction model reveals that on a validation batch containing 250 actual corporate bankruptcies ($TP + FN = 250$), the model achieves a Recall of $0.880$ ($88.0\%$).

**How many actual bankruptcies did the model fail to detect (False Negatives)?**

- [ ] A) 12
- [ ] B) 30
- [ ] C) 220
- [ ] D) 28

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 30

**Explanation:**
- **A is incorrect:** 12 would correspond to a Miss Rate of $0.048$.
- **B is correct:**
  1. Total actual bankruptcies = $TP + FN = 250$.
  2. True Positives captured: $TP = 250 \times 0.880 = 220$.
  3. False Negatives missed:
     $$FN = \text{Actual Positives} - TP = 250 - 220 = 30$$
  Alternatively, the Miss Rate is $1 - 0.880 = 0.120$, and $250 \times 0.120 = 30$.
- **C is incorrect:** 220 is the count of True Positives detected, not the missed cases.
- **D is incorrect:** 28 is an arithmetic error.

</details>

---

### 30. Question

A cybersecurity analyst compares two malware detection engines on a known test suite of 200 zero-day malware threats:
- **Engine Alpha:** Flags 140 of the threats correctly and misses 60 ($TP = 140, FN = 60$).
- **Engine Beta:** Flags 190 of the threats correctly and misses 10 ($TP = 190, FN = 10$).

**Which statement accurately compares their Recall performance?**

- [ ] A) Engine Alpha has a Recall of 0.85, whereas Engine Beta has a Recall of 0.90
- [ ] B) Both engines have equivalent sensitivity because both caught over 100 threats
- [ ] C) Engine Beta achieves 95% Recall, leaving only a 5% Miss Rate compared to Engine Alpha's 30% Miss Rate
- [ ] D) Engine Alpha is superior because catching fewer threats reduces system log overhead

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Engine Beta achieves 95% Recall, leaving only a 5% Miss Rate compared to Engine Alpha's 30% Miss Rate

**Explanation:**
- **A is incorrect:** Engine Alpha's recall is $\frac{140}{200} = 0.70$ (not $0.85$), and Engine Beta's recall is $\frac{190}{200} = 0.95$ (not $0.90$).
- **B is incorrect:** An absolute count of 100 is not a benchmark for equivalence; Beta caught 50 more threats than Alpha.
- **C is correct:**
  $$\text{Recall}_{\text{Beta}} = \frac{190}{190 + 10} = \frac{190}{200} = 0.95 \text{ (Miss Rate } = 5\%)$$
  $$\text{Recall}_{\text{Alpha}} = \frac{140}{140 + 60} = \frac{140}{200} = 0.70 \text{ (Miss Rate } = 30\%)$$
  Engine Beta leaves only a 5% window of vulnerability, whereas Engine Alpha lets nearly a third of all threats bypass detection.
- **D is incorrect:** In security, letting threats through to save log overhead is a disastrous operational failure.

</details>
