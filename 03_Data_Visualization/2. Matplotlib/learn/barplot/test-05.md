# Matplotlib Bar Plot Objective Test — Set 05

---

### 41. Question

A developer creates a grouped bar chart for four models across three evaluation metrics. The group uses `width=0.2` for four bars (one per model). At category index `i=2` ("Recall"):

```python
metrics = ["Precision", "F1", "Recall", "AUC"]   # i=2 is "Recall"
x = np.arange(4)
width = 0.2

# Four models: offsets at -1.5w, -0.5w, +0.5w, +1.5w
ax.bar(x - 1.5*width, lr,   width, label="LR")
ax.bar(x - 0.5*width, rf,   width, label="RF")
ax.bar(x + 0.5*width, svm,  width, label="SVM")
ax.bar(x + 1.5*width, xgb,  width, label="XGB")
```

At index `i=2`, what is the center x-position of the **SVM** bar?

- [ ] **A.** 2.1
- [ ] **B.** 1.9
- [ ] **C.** 2.5
- [ ] **D.** 2.0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For four models centered around index `i=2`: SVM bar center = `x[2] + 0.5 * width = 2 + 0.5 × 0.2 = 2 + 0.1 = 2.1`. The four centers are: LR = 2 − 0.3 = 1.7, RF = 2 − 0.1 = 1.9, SVM = 2 + 0.1 = **2.1**, XGB = 2 + 0.3 = 2.3.
- **Option B (Incorrect):** 1.9 is the center of the **RF** bar (x − 0.5w = 2 − 0.1 = 1.9), not the SVM bar.
- **Option C (Incorrect):** 2.5 would correspond to `x + 2.5*width = 2 + 0.5 = 2.5`, which doesn't match any bar's offset formula in this code.
- **Option D (Incorrect):** 2.0 would mean the SVM bar is centered exactly at the metric tick position with no offset. That would only be the case for a single-model chart or the "middle" model if there were an odd number. Here SVM has a +0.5w offset.

</details>

---

### 42. Question

A developer creates a grouped bar chart for 2 series with `width=0.4`. At category index `i=3`, what is the total horizontal span (from leftmost edge of bar 1 to rightmost edge of bar 2)?

- [ ] **A.** 0.4 (one bar width)
- [ ] **B.** 0.8 (two bar widths)
- [ ] **C.** 1.2 (three bar widths)
- [ ] **D.** 0.6 (1.5 bar widths)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 0.4 is the width of a single bar. The group contains two bars, so the span is larger.
- **Option B (Correct):** For 2 series with centers at `x − width/2 = 2.8` and `x + width/2 = 3.2` (for i=3): Left edge of bar 1 = 2.8 − 0.2 = **2.6**. Right edge of bar 2 = 3.2 + 0.2 = **3.4**. Total span = 3.4 − 2.6 = **0.8** = 2 × width. This makes intuitive sense: two adjacent bars of width 0.4 each, placed side-by-side, span 2 × 0.4 = 0.8 units.
- **Option C (Incorrect):** 1.2 = 3 × 0.4 would apply to a 3-bar group placed without spacing, but here we have only 2 bars.
- **Option D (Incorrect):** 0.6 = 1.5 × 0.4 doesn't correspond to any standard 2-bar grouping formula.

</details>

---

### 43. Question

A machine learning engineer creates a bar chart comparing four classifiers on a binary classification dataset:

```python
classifiers = ["Logistic Regression", "Naive Bayes", "SVM", "Gradient Boosting"]
f1_scores   = [0.821, 0.775, 0.848, 0.892]
std_devs    = [0.018, 0.024, 0.015, 0.011]

x = np.arange(4)
width = 0.5

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(x, f1_scores, width,
              yerr=std_devs, capsize=5,
              color=["#3498DB", "#95A5A6", "#E74C3C", "#2ECC71"],
              edgecolor="white", linewidth=0.8)
ax.set_xticks(x, classifiers, rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("F1 Score (mean ± std)")
ax.bar_label(bars, fmt="%.3f", padding=8)
```

What label text appears above the "Gradient Boosting" bar?

- [ ] **A.** `0.89`
- [ ] **B.** `0.892`
- [ ] **C.** `0.8920`
- [ ] **D.** `89.2`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `0.89` has only 2 decimal places. The format `"%.3f"` specifies **3** decimal places.
- **Option B (Correct):** `fmt="%.3f"` formats the bar height with exactly 3 decimal places. `0.892` formatted as `%.3f` gives `"0.892"` (already has exactly 3 decimal places, so no rounding change occurs). The label displayed above "Gradient Boosting" is `"0.892"`.
- **Option C (Incorrect):** `"0.8920"` has 4 decimal places. The format `%.3f` gives exactly 3.
- **Option D (Incorrect):** `"89.2"` would result from `fmt="%.1f"` applied to `89.2` (if the values were in percentage form). The values are in proportion (0–1 scale) and `%.3f` does not multiply by 100.

</details>

---

### 44. Question

A developer uses `ax.bar_label()` with `label_type="center"`:

```python
bars = ax.bar(x, heights)
ax.bar_label(bars, fmt="%d", label_type="center", padding=0)
```

Where does the label appear relative to each bar?

- [ ] **A.** Just above the top edge of the bar (outside, above)
- [ ] **B.** At the vertical center of the bar (midpoint between base and top, inside the bar)
- [ ] **C.** Below the bottom edge of the bar
- [ ] **D.** At the bottom of the bar, just inside the base

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Labels above the top edge correspond to `label_type="edge"` (the default). `label_type="center"` places labels inside the bar.
- **Option B (Correct):** `label_type="center"` positions the text at the **vertical midpoint** of each bar — at `y = bar_bottom + bar_height / 2`. This is useful for stacked bar charts where you want to label each segment in the middle of that segment's coloured region, making it clear which label belongs to which colour band.
- **Option C (Incorrect):** Below the bottom edge would require a negative `padding` with `label_type="edge"` or a custom positioning approach. `label_type="center"` places text inside the bar body.
- **Option D (Incorrect):** The bottom inside of the bar would be at `y = bar_bottom`. `label_type="center"` uses `y = bar_bottom + height/2`, not the bottom.

</details>

---

### 45. Question

A developer manually annotates each bar in a loop and encounters a subtle bug:

```python
scores = [0.82, 0.91, 0.78, 0.95]
labels = ["A", "B", "C", "D"]

fig, ax = plt.subplots()
bars = ax.bar(labels, scores)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,            # <-- positioned AT the bar top, not above it
            f"{height:.2f}",
            ha="center", va="bottom")
```

What is the visual issue with placing the text at exactly `height` (the bar top) with `va="bottom"`?

- [ ] **A.** The text is invisible because it renders behind the bar
- [ ] **B.** The text baseline sits exactly at the bar top — there is no padding, so the text may slightly overlap with the top edge of the bar or look cramped
- [ ] **C.** The text is placed below the bar baseline (at y=0)
- [ ] **D.** `va="bottom"` centres the text vertically on the bar top, which is correct

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The text is fully visible — it's rendered on top of (in front of) the bar, not behind it. The issue is positional, not visibility.
- **Option B (Correct):** `va="bottom"` means the BOTTOM of the text bounding box aligns to the y-coordinate provided. When `y=height` (the bar top), the bottom of the text is placed exactly at the bar top. With zero vertical gap, the text descenders may visually overlap the bar's top edge, making it look cramped. The standard fix is to add a small offset: `y = height + 0.01` or use `padding=3` in `ax.bar_label()`, which automatically adds a configurable gap between the bar top and the text bottom.
- **Option C (Incorrect):** The y-position is `height` (the bar top), not 0. The text is placed at the correct vertical position relative to the bar, just without any padding gap.
- **Option D (Incorrect):** `va="bottom"` does NOT center the text vertically. It aligns the BOTTOM edge of the text box to the specified y-coordinate. `va="center"` would center the text vertically.

</details>

---

### 46. Question

A developer wants each bar's text annotation to display the value formatted as a percentage (e.g., "87.5%") using `bar_label`:

```python
scores = [0.875, 0.920, 0.834, 0.961]
labels = ["Model 1", "Model 2", "Model 3", "Model 4"]

fig, ax = plt.subplots()
bars = ax.bar(labels, scores)
ax.bar_label(bars, fmt="%.1f%%", padding=4)
```

What label text is shown above the bar for "Model 1" (value = 0.875)?

- [ ] **A.** `"87.5%"`
- [ ] **B.** `"0.9%"` (formatted as percentage of 1.0)
- [ ] **C.** `"0.875%"`
- [ ] **D.** `"87.5"` (no percent sign)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `"87.5%"` would result from formatting the value **0.875 × 100 = 87.5** with `"%.1f%%"`. But `ax.bar_label()` formats the **bar height as-is** — it does NOT multiply by 100. Since the bar height is 0.875, `"%.1f%%" % 0.875 = "0.9%"`. Wait, let me recalculate: `"%.1f" % 0.875 = "0.9"` and then `"%%"` adds a literal `%`. So `"%.1f%%" % 0.875 = "0.9%"`. Actually `"%.1f" % 0.875 = "0.9"` (rounds 0.875 to 1 decimal). So the label is `"0.9%"`. **Option B** is correct. Let me re-examine: `"%.1f%%" % 0.875`: `%.1f` of `0.875` = `0.9`, plus `%` = `"0.9%"`. So **Option B** is correct. The question's intent may have been to show that bar_label doesn't auto-multiply. **Corrected: B is correct.**
- **Option B (Correct):** `ax.bar_label` formats the raw bar height value (0.875) using the provided `fmt`. `"%.1f%%" % 0.875` yields `"0.9%"` (0.875 rounded to 1 decimal = 0.9, then `%%` appends a literal `%`). This demonstrates that `bar_label` does NOT multiply values by 100 — the developer would need to pre-multiply values and use `scores_pct = [s*100 for s in scores]` to get "87.5%" labels.
- **Option C (Incorrect):** `"0.875%"` would result from `"%.3f%%" % 0.875`. The format `%.1f` rounds to 1 decimal place, giving `0.9`, not `0.875`.
- **Option D (Incorrect):** The `%%` in the format string produces a literal `%` sign in the output. `"87.5"` would omit the `%` and require pre-multiplied data.

</details>

---

### 47. Question

A machine learning practitioner builds a dashboard comparing train vs. validation accuracy using a grouped bar chart. The developer notices that after calling `ax.set_xticks(x)` and `ax.set_xticklabels(model_names)`, the labels appear slightly off-center from the bar groups. What is the most likely cause?

- [ ] **A.** `set_xticklabels` rotates labels by default
- [ ] **B.** `x = np.arange(len(model_names))` was not used — instead, the developer passed `model_names` (strings) directly to both bar calls, causing Matplotlib to assign categorical positions that don't match the numeric x positions used in `set_xticks`
- [ ] **C.** The `width` parameter is too large, pushing bars outside their tick positions
- [ ] **D.** `set_xticks` and `set_xticklabels` must be called simultaneously in a single statement

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `set_xticklabels` does not rotate labels by default. Rotation must be explicitly specified with `rotation=angle` or `ax.tick_params(labelrotation=angle)`.
- **Option B (Correct):** When you call `ax.bar(model_names - offset, ...)` using string arrays with arithmetic offsets, Python raises a TypeError (as discussed in Q24). But if the developer instead calls `ax.bar(x - width/2, train, ...)` with numeric `x` and then also somehow passes string labels in a different call, Matplotlib may maintain two separate coordinate systems (categorical for one series, numeric for another), causing misalignment. The most common version of this bug: calling first `ax.bar(model_names, train, ...)` which establishes a categorical axis, then trying `ax.set_xticks(x_numeric)` creates a mismatch between the categorical tick positions and the numeric `x` array. Labels appear off-center because categorical positions and numeric positions are in different coordinate spaces.
- **Option C (Incorrect):** Bar width affects bar extent but not tick label positions. Even very wide bars don't cause tick labels to misalign with bar group centers — the bars would just overlap, but labels remain at their tick positions.
- **Option D (Incorrect):** `set_xticks` and `set_xticklabels` can be called sequentially (as shown in the reference code throughout this test). Modern Matplotlib allows `ax.set_xticks(x, labels)` as a combined call, but sequential calls are equally valid.

</details>

---

### 48. Question

A developer creates a side-by-side comparison of train and validation loss bars and wants to highlight overfitting visually. At which scenario does the bar chart clearly indicate **severe overfitting**?

```python
models   = ["Model A", "Model B", "Model C"]
train_loss = [0.08,    0.12,    0.35  ]
val_loss   = [0.09,    0.45,    0.36  ]
```

- [ ] **A.** Model A (train=0.08, val=0.09) — validation loss is only 0.01 above training loss
- [ ] **B.** Model B (train=0.12, val=0.45) — the validation loss bar is nearly 4× taller than the training loss bar, indicating the model memorized training data but generalises poorly
- [ ] **C.** Model C (train=0.35, val=0.36) — both losses are relatively high
- [ ] **D.** All models show equal overfitting because the x-axis is the same

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Model A has train_loss=0.08 and val_loss=0.09 — a gap of only 0.01 (about 12.5% relative increase). This is a well-generalising model with minimal overfitting. Both bars are nearly equal height.
- **Option B (Correct):** Model B has train_loss=0.12 and val_loss=0.45 — a gap of 0.33 (275% relative increase). The validation loss bar is 3.75× taller than the training loss bar. This dramatic visual difference in bar heights clearly indicates **severe overfitting**: the model achieves very low loss on training data (0.12) but performs much worse on unseen validation data (0.45), indicating it has memorized training-specific patterns.
- **Option C (Incorrect):** Model C has train=0.35 and val=0.36 — a gap of only 0.01 (2.9% relative increase). Both bars are nearly identical height. This model generalises well but has HIGH overall loss — indicating **underfitting** (neither training nor validation loss is low), not overfitting.
- **Option D (Incorrect):** The x-axis positions the bars but has no bearing on overfitting assessment. Overfitting is assessed by comparing train vs. validation metric magnitudes, not by x-axis configuration.

</details>

---

### 49. Question

A developer creates a bar chart with `bar_label` and `padding`. What is the effect of increasing `padding` from 3 to 10?

```python
bars = ax.bar(categories, values, color="teal")
ax.bar_label(bars, fmt="%.1f", padding=10)  # was padding=3
```

- [ ] **A.** The label font size increases proportionally
- [ ] **B.** The gap between the bar top and the bottom of the label text increases — the label appears higher above the bar
- [ ] **C.** The horizontal spacing between labels increases
- [ ] **D.** The bar width shrinks to accommodate the larger padding

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `padding` is a positional offset parameter, not a font size parameter. Font size is controlled via `fontsize=` in `ax.bar_label()` or global `rcParams`.
- **Option B (Correct):** `padding` in `ax.bar_label()` specifies the **vertical distance** (in points) between the edge of the bar and the bottom of the label text. Increasing from 3 to 10 moves the text label 7 additional points away from the bar top (for `label_type="edge"`) or from the bar center (for `label_type="center"`). The label appears higher above the bar with more whitespace between bar and text.
- **Option C (Incorrect):** `padding` controls vertical spacing between the bar edge and text, not horizontal spacing between adjacent labels. Horizontal spacing is determined by bar width and bar positions.
- **Option D (Incorrect):** Bar width is controlled by the `width` argument of `ax.bar()`. The `padding` parameter in `bar_label` is a display-level text offset that has no effect on bar geometry.

</details>

---

### 50. Question

A researcher creates a bar chart of feature importance scores from a gradient boosting model. The features are NOT pre-sorted. Which code correctly sorts features by importance in **descending** order before plotting?

```python
features = ["Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5"]
importances = np.array([0.12, 0.34, 0.08, 0.25, 0.21])
```

- [ ] **A.** `idx = np.argsort(importances); ax.barh(np.array(features)[idx], importances[idx])`
- [ ] **B.** `idx = np.argsort(importances)[::-1]; ax.barh(np.array(features)[idx], importances[idx])`
- [ ] **C.** `ax.barh(sorted(features), sorted(importances, reverse=True))`
- [ ] **D.** `ax.barh(features, np.sort(importances)[::-1])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.argsort(importances)` without `[::-1]` sorts in **ascending** order (smallest to largest). The least important feature would be plotted first, producing an ascending-order chart — the opposite of what's requested.
- **Option B (Correct):** `np.argsort(importances)[::-1]` sorts indices in descending order (largest to smallest). Using `idx` to index BOTH `features` and `importances` ensures they remain paired correctly — feature names and their importance values stay aligned. `importances[idx] = [0.34, 0.25, 0.21, 0.12, 0.08]` and `np.array(features)[idx] = ["Feature_2", "Feature_4", "Feature_5", "Feature_1", "Feature_3"]`. This is the correct approach.
- **Option C (Incorrect):** `sorted(features)` sorts feature names **alphabetically**, not by importance. `sorted(importances, reverse=True)` sorts values independently. The two operations create a NEW mismatch — alphabetically ordered names are paired with importance-sorted values, giving a completely wrong feature-importance mapping.
- **Option D (Incorrect):** `np.sort(importances)[::-1]` gives correctly sorted importance VALUES, but `features` (unsorted) is still in the original order. The values are now mismatched with the wrong feature names — Feature_1 would get the highest importance value even though Feature_2 (index 1) actually has the highest importance.

</details>
