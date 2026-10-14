### 1. Question

A product manager reviews a churn model's confusion matrix on a batch of 100 customers: true positives = 40, false positives = 10, true negatives = 30, false negatives = 20.

**What is the model's accuracy on this batch?**

- [ ] A) 0.40
- [ ] B) 0.60
- [x] C) 0.70 _(Correct)_
- [ ] D) 0.80

### Explanations

- **A) 0.40 (Incorrect):** Only reflects true positives divided by the total, ignoring true negatives which also count as correct predictions.
- **B) 0.60 (Incorrect):** Undercounts the correct predictions; it does not match $(40 + 30) / 100$.
- **C) 0.70 (Correct):** Accuracy is calculated as:
  $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total}} = \frac{40 + 30}{100} = 0.70$$
- **D) 0.80 (Incorrect):** Overstates the correct predictions; adding false positives or false negatives into the numerator causes this error.

---

### 2. Question

A senior data analyst lowers the classification cutoff for a disease-prediction model from 0.5 to 0.1.

**What is the most likely effect on the model's predictions and metrics?**

- [ ] A) The confusion matrix remains unchanged, since a cutoff never affects predicted labels
- [x] B) More cases get predicted positive, raising recall while lowering precision _(Correct)_
- [ ] C) More cases get predicted negative, which increases precision immediately
- [ ] D) Accuracy always increases automatically whenever the cutoff is lowered

### Explanations

- **A) The confusion matrix remains unchanged, since a cutoff never affects predicted labels (Incorrect):** The cutoff decides whether a predicted probability maps to a positive or negative label, so changing the threshold directly alters the counts in the confusion matrix.
- **B) More cases get predicted positive, raising recall while lowering precision (Correct):** A lower cutoff makes it easier for samples to be classified as positive. This identifies more true positives (raising recall) while simultaneously admitting more false positives (lowering precision).
- **C) More cases get predicted negative, which increases precision immediately (Incorrect):** Lowering the threshold requires a lower probability to qualify as a positive, resulting in fewer negative predictions, not more.
- **D) Accuracy always increases automatically whenever the cutoff is lowered (Incorrect):** Accuracy depends on the overall balance of true positives and true negatives. Lowering the threshold introduces additional false positives, which can easily decrease overall accuracy.

---

### 3. Question

A data scientist fits a logistic regression model and calls `model.predict_proba(X_test)` instead of `model.predict(X_test)`.

**What does the output of `predict_proba` represent?**

- [ ] A) Binary class labels, either 0 or 1, for each patient
- [ ] B) The confusion matrix computed from predictions and actual labels
- [ ] C) The F1 score summarizing precision and recall together
- [x] D) Probability estimates between 0 and 1 for each class _(Correct)_

### Explanations

- **A) Binary class labels, either 0 or 1, for each patient (Incorrect):** Binary 0/1 discrete labels come from `predict()`, which applies a decision threshold (defaulting to 0.5) to the underlying probabilities.
- **B) The confusion matrix computed from predictions and actual labels (Incorrect):** The confusion matrix is an evaluation summary created by cross-tabulating predicted labels with true labels, not a direct output of `predict_proba`.
- **C) The F1 score summarizing precision and recall together (Incorrect):** The F1 score is a performance metric computed from precision and recall after predictions have been evaluated against ground truth labels.
- **D) Probability estimates between 0 and 1 for each class (Correct):** Logistic regression's sigmoid function calculates continuous probability values between 0 and 1, and `predict_proba` returns these raw predicted probabilities for each class.

---

### 4. Question

A senior data analyst built a spam classifier and wants to know: out of every email the model labeled as spam, how many were actually spam? The analyst needs the one metric that answers exactly this question, based only on the model's positive predictions.

**Which metric should the analyst check?**

- [x] A) Precision, the positive-prediction accuracy metric _(Correct)_
- [ ] B) Recall, the metric for actual-positive coverage rate
- [ ] C) Accuracy, the metric for overall prediction correctness
- [ ] D) F1 score, the metric balancing precision and recall

### Explanations

- **A) Precision, the positive-prediction accuracy metric (Correct):** Precision is defined as $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$, which directly answers "out of all instances predicted as positive (spam), what fraction was actually positive."
- **B) Recall, the metric for actual-positive coverage rate (Incorrect):** Recall evaluates coverage among ground-truth positive cases ($\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$), measuring how many actual spams were successfully flagged rather than the purity of positive predictions.
- **C) Accuracy, the metric for overall prediction correctness (Incorrect):** Accuracy evaluates performance across both positive and negative predictions combined ($\frac{\text{TP} + \text{TN}}{\text{Total}}$), rather than isolating the reliability of the positive prediction slice.
- **D) F1 score, the metric balancing precision and recall (Incorrect):** The F1 score is the harmonic mean of precision and recall; while it incorporates precision, it balances it with recall rather than measuring positive prediction accuracy alone.

---

### 5. Question

A senior data analyst evaluates a fraud model on a batch of 200 transactions and gets the following confusion matrix:

|                    | Predicted Fraud | Predicted Genuine |
| :----------------- | :-------------: | :---------------: |
| **Actual Fraud**   |     TP = 48     |      FN = 50      |
| **Actual Genuine** |     FP = 2      |     TN = 100      |

**Which statements about this confusion matrix are correct?**

- [ ] A) F1 score (about 0.65) is closer to recall alone (about 0.49) than to the simple average of precision and recall (about 0.72)
- [x] B) Precision is 0.96, computed as 48 true positives divided by 48 true positives plus 2 false positives _(Correct)_
- [x] C) Recall is approximately 0.49, computed as 48 true positives divided by 48 true positives plus 50 false negatives _(Correct)_
- [x] D) F1 score is about 0.65, lower than the simple average because the harmonic mean penalizes the gap _(Correct)_

### Explanations

- **A) F1 score (about 0.65) is closer to recall alone (about 0.49) than to the simple average of precision and recall (about 0.72) (Incorrect):** Although the harmonic mean leans toward the smaller value, F1 ($\approx 0.65$) is only $0.07$ away from the simple average ($0.72$), but $0.16$ away from recall ($0.49$). Therefore, it is actually closer to the simple average.
- **B) Precision is 0.96, computed as 48 true positives divided by 48 true positives plus 2 false positives (Correct):** Precision measures the proportion of positive identifications that were actually correct:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{48}{48 + 2} = \frac{48}{50} = 0.96$$
- **C) Recall is approximately 0.49, computed as 48 true positives divided by 48 true positives plus 50 false negatives (Correct):** Recall measures the proportion of actual positives that were identified correctly:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{48}{48 + 50} = \frac{48}{98} \approx 0.49$$
- **D) F1 score is about 0.65, lower than the simple average because the harmonic mean penalizes the gap (Correct):**
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.96 \times 0.49}{0.96 + 0.49} \approx 0.65$$
  Because the harmonic mean severely penalizes large discrepancies between precision and recall, the resulting F1 score is noticeably lower than the arithmetic mean ($0.72$).

---

### 6. Question

A machine learning engineer splits a small fraud dataset containing 27 genuine and 8 fraud transactions using `train_test_split`. One split happens to place zero fraud transactions in the test set, making the resulting confusion matrix impossible to interpret meaningfully.

**Which change should the engineer make to fix this reliably?**

- [x] A) Set `stratify=y` in `train_test_split` so the split preserves the class proportions _(Correct)_
- [ ] B) Increase `test_size` to 0.5 so the test set contains more rows overall
- [ ] C) Set `random_state` to a larger fixed integer value for the split
- [ ] D) Remove the fraud examples from the dataset entirely before splitting

### Explanations

- **A) Set `stratify=y` in `train_test_split` so the split preserves the class proportions (Correct):** Passing the target labels to `stratify` ensures that both the training and testing partitions preserve the original ratio of fraud to genuine transactions, reliably preventing zero-instance splits in minority classes.
- **B) Increase `test_size` to 0.5 so the test set contains more rows overall (Incorrect):** Merely increasing the size of the test split does not ensure proportional representation; random variation can still leave the minority class underrepresented or entirely missing.
- **C) Set `random_state` to a larger fixed integer value for the split (Incorrect):** Modifying the random seed only alters the pseudo-random shuffling sequence; it does not enforce structural balance across splits.
- **D) Remove the fraud examples from the dataset entirely before splitting (Incorrect):** Dropping fraud samples discards the essential target class the model is intended to learn and detect, invalidating the fraud detection task completely.

---

### 7. Question

A cross-functional team is deciding, for several classifiers, whether to prioritize recall (catching every actual positive) over precision (only flagging cases the model is confident about).

**In which of these situations should the team prioritize recall over precision?**

- [x] A) A hospital screening test for a serious disease, where missing an actual case delays treatment _(Correct)_
- [x] B) A cancer risk model, where failing to flag an at-risk patient means they get no further testing _(Correct)_
- [ ] C) An email spam filter, where marking a legitimate business email as spam causes it to be missed by the recipient
- [ ] D) A promotional newsletter classifier, where an occasional legitimate email lands in spam with no real consequence

### Explanations

- **A) A hospital screening test for a serious disease, where missing an actual case delays treatment (Correct):** A missed disease case represents a false negative with potentially life-threatening delays in treatment. Because the cost of a false negative is extremely high, recall must be prioritized to catch as many true cases as possible.
- **B) A cancer risk model, where failing to flag an at-risk patient means they get no further testing (Correct):** Failing to identify an at-risk patient (a false negative) deprives them of essential diagnostic follow-up. In medical screening pipelines, maximizing recall takes precedence over precision.
- **C) An email spam filter, where marking a legitimate business email as spam causes it to be missed by the recipient (Incorrect):** In this scenario, the primary risk is a false positive (a critical legitimate email being incorrectly filtered into spam). Therefore, precision must be prioritized over recall to ensure high confidence before filtering.
- **D) A promotional newsletter classifier, where an occasional legitimate email lands in spam with no real consequence (Incorrect):** In low-stakes tasks with minimal consequence for either error type, there is no strong imperative to heavily skew optimization toward recall over precision.

---

### 8. Question

A hospital risk model outputs predicted probabilities for 8 patients along with their actual outcomes (1 = disease present, 0 = disease absent):

| Patient | Predicted Probability | Actual Outcome |
| :------ | :-------------------: | :------------: |
| P1      |         0.15          |       0        |
| P2      |         0.35          |       1        |
| P3      |         0.55          |       0        |
| P4      |         0.65          |       1        |
| P5      |         0.25          |       1        |
| P6      |         0.75          |       0        |
| P7      |         0.45          |       1        |
| P8      |         0.85          |       1        |

**Which statements about these two cutoffs are correct?**

- [ ] A) Lowering the cutoff from 0.5 to 0.3 always decreases precision, since more patients are predicted positive
- [x] B) At cutoff 0.5, the confusion matrix has TP=2, FP=2, TN=1, FN=3, which gives a recall of exactly 0.40 _(Correct)_
- [x] C) At cutoff 0.3, recall improves to 0.80 because two more actual-positive patients (P2 and P7) are now captured _(Correct)_
- [x] D) At cutoff 0.3, precision rises to about 0.67, higher than at cutoff 0.5, since the new positive predictions were correct _(Correct)_

### Explanations

- **A) Lowering the cutoff from 0.5 to 0.3 always decreases precision, since more patients are predicted positive (Incorrect):** While lowering a threshold often increases false positives and lowers precision, it is not an absolute rule. In this specific dataset, precision actually increases from $0.50$ to $\approx 0.67$ when moving from cutoff $0.5$ to $0.3$ because both newly captured cases (P2 and P7) are true positives.
- **B) At cutoff 0.5, the confusion matrix has TP=2, FP=2, TN=1, FN=3, which gives a recall of exactly 0.40 (Correct):** At threshold $0.5$, patients predicted positive ($\ge 0.5$) are P3, P4, P6, and P8:
  - **TP = 2** (P4, P8)
  - **FP = 2** (P3, P6)
  - **TN = 1** (P1)
  - **FN = 3** (P2, P5, P7)
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{2}{2 + 3} = 0.40$$
- **C) At cutoff 0.3, recall improves to 0.80 because two more actual-positive patients (P2 and P7) are now captured (Correct):** At threshold $0.3$, patients predicted positive ($\ge 0.3$) are P2, P3, P4, P6, P7, and P8. TP increases to 4 (P2, P4, P7, P8) and FN drops to 1 (P5):
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{4}{4 + 1} = 0.80$$
- **D) At cutoff 0.3, precision rises to about 0.67, higher than at cutoff 0.5, since the new positive predictions were correct (Correct):** At cutoff $0.3$, FP remains at 2 (P3, P6) while TP increases to 4:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{4}{4 + 2} = \frac{4}{6} \approx 0.67$$
  This is higher than the precision at cutoff $0.5$ ($\frac{2}{2 + 2} = 0.50$).

---

### 9. Question

A data science team debates how the F1 score behaves compared to precision and recall.

**Which statements about the F1 score are correct?**

- [ ] A) F1 score always equals the simple arithmetic average of precision and recall
- [ ] B) True negatives are used directly inside the F1 score formula itself
- [x] C) F1 score is defined as the harmonic mean of precision and recall _(Correct)_
- [x] D) F1 score penalizes a large precision-recall gap, pulling toward the lower value _(Correct)_

### Explanations

- **A) F1 score always equals the simple arithmetic average of precision and recall (Incorrect):** F1 only equals the arithmetic average when precision and recall are identical. Whenever they differ, the harmonic mean pulls the F1 score below the simple average.
- **B) True negatives are used directly inside the F1 score formula itself (Incorrect):** F1 is calculated exclusively from precision (TP, FP) and recall (TP, FN); true negatives are completely excluded from the formula.
- **C) F1 score is defined as the harmonic mean of precision and recall (Correct):** F1 is formally defined as the harmonic mean:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **D) F1 score penalizes a large precision-recall gap, pulling toward the lower value (Correct):** Because it is a harmonic mean, the metric is dragged toward the smaller number, heavily penalizing large imbalances between precision and recall.

---

### 10. Question

A backend engineer builds a confusion matrix for a fraud-detection model, with predictions in columns (Positive = fraud, Negative = genuine) and actuals in rows (fraud or genuine). One transaction was actually genuine, but the model predicted it as fraud.

**Which quadrant of the confusion matrix does this transaction belong to?**

- [ ] A) True Positive — the model correctly predicted fraud when the transaction was actually fraud
- [x] B) False Positive — the model predicted fraud when the transaction was actually genuine _(Correct)_
- [ ] C) True Negative — the model correctly predicted genuine when the transaction was actually genuine
- [ ] D) False Negative — the model predicted genuine when the transaction was actually fraud

### Explanations

- **A) True Positive — the model correctly predicted fraud when the transaction was actually fraud (Incorrect):** A true positive requires both the prediction and actual outcome to be positive (fraud). In this scenario, the actual transaction was genuine.
- **B) False Positive — the model predicted fraud when the transaction was actually genuine (Correct):** The prediction was positive (fraud) while the true ground label was negative (genuine), which defines a false positive (Type I error).
- **C) True Negative — the model correctly predicted genuine when the transaction was actually genuine (Incorrect):** A true negative requires the prediction to be negative (genuine); here, the model incorrectly flagged the transaction as positive (fraud).
- **D) False Negative — the model predicted genuine when the transaction was actually fraud (Incorrect):** A false negative occurs when an actual positive (fraud) is erroneously predicted as negative (genuine); here, the reverse occurred.

---
