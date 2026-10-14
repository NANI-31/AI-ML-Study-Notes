### 41. Question

The generalized **$F_\beta$ Score** allows machine learning engineers to adjust the relative balance between Precision and Recall:
$$F_\beta = (1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

**Which value of $\beta$ configures the metric to weight Recall as exactly twice as important as Precision?**

- [ ] A) $\beta = 0.5$
- [ ] B) $\beta = 1.0$
- [ ] C) $\beta = 2.0$
- [ ] D) $\beta = 4.0$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) $\beta = 2.0$

**Explanation:**
- **A is incorrect:** $\beta = 0.5$ weights Precision twice as heavily as Recall (used when false alarms are twice as costly as misses).
- **B is incorrect:** $\beta = 1.0$ weights Precision and Recall equally, yielding the standard $F_1$ score.
- **C is correct:** By definition, the parameter $\beta$ indicates how many times more important Recall is compared to Precision. Setting $\beta = 2.0$ yields the **$F_2$ score**, which weights Recall twice as heavily as Precision:
  $$F_2 = (1 + 2^2) \frac{P \times R}{2^2 P + R} = 5 \times \frac{P \times R}{4P + R}$$
- **D is incorrect:** $\beta = 4.0$ would weight Recall 4 times as heavily as Precision ($\beta^2 = 16$).

</details>

---

### 42. Question

A machine learning model evaluated on an industrial equipment failure dataset achieves a Precision of $0.80$ and a Recall of $0.50$.

**What is the $F_2$ Score of this model?**

- [ ] A) 0.615
- [ ] B) 0.541
- [ ] C) 0.714
- [ ] D) 0.400

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 0.541

**Explanation:**
- **A is incorrect:** $0.615$ is the standard $F_1$ score ($2 \times \frac{0.80 \times 0.50}{0.80 + 0.50} = \frac{0.80}{1.30} \approx 0.6154$).
- **B is correct:** The $F_2$ score is computed with $\beta = 2$ and $\beta^2 = 4$:
  $$F_2 = (1 + 2^2) \times \frac{P \times R}{4P + R} = 5 \times \frac{0.80 \times 0.50}{4(0.80) + 0.50} = 5 \times \frac{0.40}{3.20 + 0.50} = \frac{2.00}{3.70} \approx 0.5405 \approx 0.541$$
  Notice that because $F_2$ weights Recall twice as much as Precision, the score pulls strongly down toward the lower Recall value ($0.50$), landing at $0.541$ (below the $F_1$ score of $0.615$).
- **C is incorrect:** $0.714$ is the $F_{0.5}$ score, which weights Precision higher.
- **D is incorrect:** $0.400$ is simply the product $P \times R$.

</details>

---

### 43. Question

On the same model with Precision = $0.80$ and Recall = $0.50$, the team now calculates the **$F_{0.5}$ Score** to evaluate its suitability for a spam classification system.

**What is the $F_{0.5}$ Score of the model?**

- [ ] A) 0.541
- [ ] B) 0.615
- [ ] C) 0.650
- [ ] D) 0.714

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 0.714

**Explanation:**
- **A is incorrect:** $0.541$ is the $F_2$ score.
- **B is incorrect:** $0.615$ is the $F_1$ score.
- **C is incorrect:** $0.650$ is the simple arithmetic mean of Precision and Recall.
- **D is correct:** For $F_{0.5}$, we set $\beta = 0.5$, so $\beta^2 = 0.25$:
  $$F_{0.5} = (1 + 0.25) \times \frac{P \times R}{0.25P + R} = 1.25 \times \frac{0.80 \times 0.50}{0.25(0.80) + 0.50} = 1.25 \times \frac{0.40}{0.20 + 0.50} = \frac{0.50}{0.70} \approx 0.7143 \approx 0.714$$
  Because $F_{0.5}$ weights Precision twice as heavily as Recall, the metric climbs toward Precision ($0.80$), resulting in $0.714$.

</details>

---

### 44. Question

What is the mathematical limit of the $F_\beta$ score as the parameter $\beta$ approaches infinity ($\beta \to \infty$)?

- [ ] A) $\lim_{\beta \to \infty} F_\beta = \text{Recall}$
- [ ] B) $\lim_{\beta \to \infty} F_\beta = \text{Precision}$
- [ ] C) $\lim_{\beta \to \infty} F_\beta = 1.0$
- [ ] D) $\lim_{\beta \to \infty} F_\beta = 0.0$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) $\lim_{\beta \to \infty} F_\beta = \text{Recall}$

**Explanation:**
- **A is correct:** Dividing the numerator and denominator of the $F_\beta$ formula by $\beta^2$:
  $$F_\beta = \frac{(1 + \beta^2) P \cdot R}{\beta^2 P + R} = \frac{\left(\frac{1}{\beta^2} + 1\right) P \cdot R}{P + \frac{R}{\beta^2}}$$
  Taking the limit as $\beta \to \infty$, the terms $\frac{1}{\beta^2} \to 0$ and $\frac{R}{\beta^2} \to 0$:
  $$\lim_{\beta \to \infty} F_\beta = \frac{(0 + 1) P \cdot R}{P + 0} = \frac{P \cdot R}{P} = \text{Recall}$$
  Thus, setting $\beta \to \infty$ means caring exclusively about Recall.
- **B is incorrect:** Precision is obtained when $\beta \to 0$.
- **C is incorrect:** The limit depends on the actual Recall of the model, not a constant 1.0.
- **D is incorrect:** The limit does not vanish to 0 unless Recall itself is 0.

</details>

---

### 45. Question

Conversely, what is the mathematical limit of the $F_\beta$ score as the parameter $\beta$ approaches zero ($\beta \to 0$)?

- [ ] A) $\lim_{\beta \to 0} F_\beta = \text{Recall}$
- [ ] B) $\lim_{\beta \to 0} F_\beta = 0.5$
- [ ] C) $\lim_{\beta \to 0} F_\beta = \text{Precision}$
- [ ] D) $\lim_{\beta \to 0} F_\beta = \text{Accuracy}$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) $\lim_{\beta \to 0} F_\beta = \text{Precision}$

**Explanation:**
- **A is incorrect:** Recall corresponds to $\beta \to \infty$.
- **B is incorrect:** The limit depends on the model's actual Precision.
- **C is correct:** In the $F_\beta$ formulation:
  $$F_\beta = \frac{(1 + \beta^2) P \cdot R}{\beta^2 P + R}$$
  Setting $\beta = 0$:
  $$\lim_{\beta \to 0} F_\beta = \frac{(1 + 0) P \cdot R}{0 \cdot P + R} = \frac{P \cdot R}{R} = \text{Precision}$$
  Thus, setting $\beta \to 0$ corresponds to evaluating performance purely by Precision.
- **D is incorrect:** Accuracy involves True Negatives, which are not present in the $F_\beta$ family.

</details>

---

### 46. Question

A software team is deploying an AI assistant that automatically categorizes incoming customer support emails and generates pre-written responses. If the model miscategorizes an email (False Positive for that topic), an irrelevant automated reply is sent, annoying the customer and degrading trust. If the model is uncertain (False Negative), the email simply falls back to a human agent.

**Which metric should the development team select as the primary optimization loss/target?**

- [ ] A) $F_3$ score
- [ ] B) $F_{0.5}$ score
- [ ] C) $F_2$ score
- [ ] D) Unweighted Recall

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) $F_{0.5}$ score

**Explanation:**
- **A is incorrect:** $F_3$ heavily prioritizes recall over precision, which would send out many incorrect automated replies.
- **B is correct:** In this architecture, False Positives (sending a wrong automated response) carry a high operational and reputational penalty. False Negatives carry a minimal penalty because the email is safely handled by a human agent. Therefore, the team needs a metric that weights **Precision higher than Recall**. The **$F_{0.5}$ score** weights precision twice as much as recall, making it the ideal objective function.
- **C is incorrect:** $F_2$ prioritizes Recall over Precision.
- **D is incorrect:** Maximizing Recall would force the system to automate every email even at very low confidence, maximizing bad replies.

</details>

---

### 47. Question

An emergency healthcare system deploys a sepsis early-warning monitor in an intensive care unit (ICU). Early antibiotic intervention saves lives, whereas a missed sepsis infection (False Negative) frequently results in septic shock and mortality within hours. A false alarm (False Positive) merely causes a nurse to draw a routine blood culture.

**Which of the following metrics should the ICU clinical committee select to evaluate this monitor?**

- [ ] A) Precision
- [ ] B) Specificity
- [ ] C) $F_{0.5}$ score
- [ ] D) $F_2$ score

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) $F_2$ score

**Explanation:**
- **A is incorrect:** Maximizing Precision would make the alarm conservative, missing early-stage septic patients.
- **B is incorrect:** Specificity minimizes false alarms on healthy patients, but does not measure how many septic patients are caught.
- **C is incorrect:** $F_{0.5}$ emphasizes Precision over Recall, the opposite of what is needed in life-critical medical screening.
- **D is correct:** Sepsis detection requires an asymmetric objective where **catching true cases (Recall) is vastly more critical than avoiding false alarms (Precision)**. The **$F_2$ score** weights Recall twice as heavily as Precision, providing an optimal mathematical alignment with the clinical cost matrix.

</details>

---

### 48. Question

Which of the following formulations expresses the generalized $F_\beta$ score directly in terms of the confusion matrix elements ($TP, FP, FN$)?

- [ ] A) $F_\beta = \frac{(1 + \beta^2) TP}{(1 + \beta^2) TP + FP + \beta^2 FN}$
- [ ] B) $F_\beta = \frac{(1 + \beta^2) TP}{(1 + \beta^2) TP + \beta^2 FP + FN}$
- [ ] C) $F_\beta = \frac{2TP}{2TP + \beta^2 FP + FN}$
- [ ] D) $F_\beta = \frac{(1 + \beta^2)(TP + TN)}{\text{Total}}$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) $F_\beta = \frac{(1 + \beta^2) TP}{(1 + \beta^2) TP + FP + \beta^2 FN}$

**Explanation:**
- **A is correct:** Substituting Precision ($\frac{TP}{TP + FP}$) and Recall ($\frac{TP}{TP + FN}$) into $F_\beta = (1 + \beta^2) \frac{P \cdot R}{\beta^2 P + R}$:
  $$F_\beta = (1 + \beta^2) \frac{\frac{TP^2}{(TP+FP)(TP+FN)}}{\beta^2 \frac{TP}{TP+FP} + \frac{TP}{TP+FN}} = \frac{(1 + \beta^2) TP}{\beta^2 (TP + FN) + (TP + FP)} = \frac{(1 + \beta^2) TP}{(1 + \beta^2) TP + FP + \beta^2 FN}$$
  Notice that $\beta^2$ scales the False Negative penalty $FN$. When $\beta = 2$, misses ($FN$) are penalized 4 times as heavily as false alarms ($FP$).
- **B is incorrect:** This incorrectly attaches $\beta^2$ to $FP$ instead of $FN$.
- **C is incorrect:** The factor in the numerator must be $(1 + \beta^2)$, not a static 2.
- **D is incorrect:** True Negatives are not part of the $F_\beta$ metric family.

</details>

---

### 49. Question

A financial institution evaluates two credit card fraud detection algorithms using the $F_2$ score ($\beta = 2$):
- **Model Alpha:** Precision = $0.90$, Recall = $0.40$
- **Model Beta:** Precision = $0.50$, Recall = $0.80$

**What are the respective $F_2$ scores of Model Alpha and Model Beta?**

- [ ] A) Model Alpha: 0.720; Model Beta: 0.720
- [ ] B) Model Alpha: 0.650; Model Beta: 0.650
- [ ] C) Model Alpha: 0.450; Model Beta: 0.714
- [ ] D) Model Alpha: 0.541; Model Beta: 0.615

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Model Alpha: 0.450; Model Beta: 0.714

**Explanation:**
- **A is incorrect:** These assume an equal-weight metric or arithmetic combination.
- **B is incorrect:** $0.650$ is the simple arithmetic mean of precision and recall for both models.
- **C is correct:** Using the $F_2$ formula ($5 \times \frac{P \cdot R}{4P + R}$):
  For Model Alpha:
  $$F_{2, \text{Alpha}} = 5 \times \frac{0.90 \times 0.40}{4(0.90) + 0.40} = 5 \times \frac{0.36}{3.60 + 0.40} = \frac{1.80}{4.00} = 0.450$$
  For Model Beta:
  $$F_{2, \text{Beta}} = 5 \times \frac{0.50 \times 0.80}{4(0.50) + 0.80} = 5 \times \frac{0.40}{2.00 + 0.80} = \frac{2.00}{2.80} \approx 0.7143 \approx 0.714$$
  Under $F_2$, Model Beta ($0.714$) is vastly superior to Model Alpha ($0.450$) because Model Beta delivers 80% recall.
- **D is incorrect:** Calculated using wrong parameter values.

</details>

---

### 50. Question

In the context of the $F_\beta$ metric, what does the parameter $\beta$ represent conceptually in business decision-making?

- [ ] A) The ratio of True Negatives to False Positives
- [ ] B) How many times more important Recall is considered relative to Precision
- [ ] C) The probability threshold above which a prediction is classified positive
- [ ] D) The percentage of training data reserved for cross-validation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) How many times more important Recall is considered relative to Precision

**Explanation:**
- **A is incorrect:** This relates to Specificity.
- **B is correct:** The parameter $\beta$ explicitly measures the relative weight assigned to Recall versus Precision. Specifically, setting $\beta = k$ states that Recall is deemed **$k$ times as important as Precision**. If false negatives are 3 times more costly to the enterprise than false positives, the appropriate metric is $F_3$.
- **C is incorrect:** The decision threshold is usually denoted by $\tau$ or $t$, ranging between 0 and 1.
- **D is incorrect:** This defines the test split ratio (e.g., `test_size=0.2`).

</details>
