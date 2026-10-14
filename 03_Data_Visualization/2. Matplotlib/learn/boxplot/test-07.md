# Matplotlib Box Plot Objective Test — Set 07

---

### 61. Question

A machine learning engineer builds a comprehensive model evaluation plot. Examine the code and determine what `bp["boxes"][2]` refers to:

```python
models = ["Ridge", "Lasso", "ElasticNet", "SVR"]
residuals = [ridge_res, lasso_res, enet_res, svr_res]

bp = ax.boxplot(residuals,
                tick_labels=models,
                patch_artist=True)

colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
```

What does `bp["boxes"][2]` refer to?

- [ ] **A.** The Ridge model's box
- [ ] **B.** The Lasso model's box
- [ ] **C.** The ElasticNet model's box (index 2 = 3rd element = 0-indexed)
- [ ] **D.** The SVR model's box

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `bp["boxes"][0]` (index 0) refers to Ridge — the first model in the `residuals` list.
- **Option B (Incorrect):** `bp["boxes"][1]` (index 1) refers to Lasso — the second model.
- **Option C (Correct):** Python uses 0-based indexing. `bp["boxes"][2]` is the **3rd element** (index 2) of the `boxes` list, corresponding to the 3rd data group in `residuals`: **ElasticNet**. `colors[2] = "#2ECC71"` (green) is applied to ElasticNet's box with 80% opacity.
- **Option D (Incorrect):** `bp["boxes"][3]` (index 3) refers to SVR — the fourth and final model.

</details>

---

### 62. Question

A data scientist creates a box plot for two competing recommendation algorithms and enables notched boxes. The plot shows:

- **Algorithm A**: notch spans from **4.1 to 4.5** (NDCG@10 scores)
- **Algorithm B**: notch spans from **4.3 to 4.7** (NDCG@10 scores)

The notches **overlap** in the range 4.3–4.5. What conclusion is correct?

- [ ] **A.** Algorithm B is definitively better — its notch is higher
- [ ] **B.** Algorithm A is definitively better — it has a lower IQR
- [ ] **C.** The median NDCG@10 scores are NOT statistically significantly different at ~95% confidence, because the notches overlap
- [ ] **D.** The medians are exactly equal at the overlap point (4.4)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Algorithm B has a higher notch position, meaning its median NDCG is higher. However, statistical significance is the question — and with overlapping notches, the difference is NOT statistically significant. "Higher median" ≠ "definitively better" without significance.
- **Option B (Incorrect):** IQR (box width) indicates variability, not which algorithm is better. Algorithm A's IQR and notch width are not determinable from the notch bounds alone. And regardless of IQR, the question is about median comparison significance.
- **Option C (Correct):** Overlapping notches mean: the 95% confidence intervals for the two medians **overlap**, so we **cannot reject** the null hypothesis that the medians are equal. The difference between Algorithm A's median and Algorithm B's median is not statistically significant at approximately 95% confidence. More data or a formal statistical test (e.g., Mann-Whitney U) would be needed to confirm significance.
- **Option D (Incorrect):** Overlapping notches do NOT mean the medians are equal at the overlap point. The medians are distinct values (likely around 4.3 for A and 4.5 for B, based on notch centers). The overlap indicates only that the difference is statistically uncertain — not that the medians are the same value.

</details>

---

### 63. Question

A developer creates a box plot for the sorted dataset:

```
[5, 8, 10, 12, 14, 15, 17, 19, 21, 25, 30]
```

For n=11: Median (Q₂) = 6th value = **15**. Lower half = [5,8,10,12,14] → Q₁ = 10. Upper half = [17,19,21,25,30] → Q₃ = 21. IQR = 11.

Lower fence = 10 − 1.5×11 = **−6.5**. Upper fence = 21 + 1.5×11 = **37.5**.

Now a value of **32** is added to the dataset. What happens to the box plot?

- [ ] **A.** 32 is an outlier (32 < 37.5 is false — actually 32 < 37.5 so it's NOT an outlier; whisker extends to 32)
- [ ] **B.** 32 becomes a flier because it is above Q₃
- [ ] **C.** 32 is within the upper fence (32 < 37.5), so the upper whisker now extends to 32 — no new flier appears
- [ ] **D.** 32 changes Q₃ to a higher value, which changes IQR and creates new outliers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Correct framing):** The question states that 32 < 37.5, meaning 32 IS within the upper fence. So the whisker extends to 32. Option C states this correctly.
- **Option B (Incorrect):** Being above Q₃ is not sufficient to be a flier. The outlier rule requires the value to be beyond the 1.5×IQR fence, not just beyond Q₃. Values between Q₃ and Q₃+1.5×IQR fall in the upper whisker range — they are non-outlier extreme values.
- **Option C (Correct):** 32 < upper fence of 37.5 → 32 is within the whisker range. Before adding 32, the upper whisker extended to 30 (the old maximum). After adding 32, the upper whisker extends to **32** (the new maximum within the fence). No flier appears because 32 does not exceed the fence. Note: adding 32 also changes Q₁, Q₂, Q₃, and IQR slightly for the now 12-element dataset, but since the question uses the pre-computed fences from the 11-element dataset, option C correctly addresses whether 32 is within those fences.
- **Option D (Incorrect):** While adding a data point does technically change quartile values (especially with the dataset growing from n=11 to n=12), the change is typically minor and the new fences remain similar. The premise of option D (that adding 32 creates new outliers) is not supported — a moderate value like 32 near the previous maximum is unlikely to dramatically shift the quartiles enough to reclassify existing values as outliers.

</details>

---

### 64. Question

A developer creates a box plot and passes a **Pandas DataFrame column** as input:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "temperature": [22.1, 23.5, 21.8, 24.2, 25.1, 26.3, 23.9, 22.7, 80.0],
    "humidity":    [45, 52, 48, 55, 60, 58, 47, 51, 53]
})

fig, ax = plt.subplots()
ax.boxplot(df["temperature"])
```

What visual element would make the value **80.0** stand out in this box plot?

- [ ] **A.** It would appear as an extra-tall bar
- [ ] **B.** It would be plotted as an individual flier marker (outlier point) above the upper whisker cap
- [ ] **C.** The box would extend up to 80.0 since it's the maximum value
- [ ] **D.** It would not be visible — box plots hide maximum values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Box plots don't use bars. They use a box (Q₁–Q₃), a median line, whiskers, caps, and flier markers. There are no "tall bars" in a box plot.
- **Option B (Correct):** For the temperature data: most values cluster around 21–26°C. Q₁≈22.3, Q₃≈25.1, IQR≈2.8. Upper fence = 25.1 + 1.5×2.8 = 25.1 + 4.2 = 29.3. Since 80.0 >> 29.3 (far beyond the fence), it is classified as an **extreme outlier** and plotted as an individual flier marker point well above the whisker cap. This makes the data entry error (presumably 80.0 should be ~23.0 or similar) immediately obvious in the visualization.
- **Option C (Incorrect):** The box extends only from Q₁ to Q₃. The maximum value determines where the whisker ends (if within the fence) or becomes a flier marker (if beyond the fence). With 80.0 far beyond the fence, the box does not extend to it — the flier marker appears above the whisker cap.
- **Option D (Incorrect):** Box plots specifically show extreme values through flier markers. 80.0 would be the most visually prominent element in the plot — exactly the opposite of "not visible."

</details>

---

### 65. Question

A developer wants to **suppress the cap lines** at the ends of whiskers to create a cleaner visual style (no horizontal tick marks at whisker ends). Which approach achieves this?

- [ ] **A.** `ax.boxplot(data, showcaps=False)`
- [ ] **B.** `ax.boxplot(data, caps=False)`
- [ ] **C.** After plotting: `for cap in bp["caps"]: cap.set_visible(False)`
- [ ] **D.** Both A and C are valid approaches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Valid):** `showcaps=False` is a valid `ax.boxplot()` parameter that prevents cap lines from being drawn at whisker ends. This is the clean, one-parameter approach to hide caps at creation time.
- **Option B (Invalid):** `caps=False` is not a recognized parameter. This would raise a `TypeError`. The correct parameter is `showcaps=False`.
- **Option C (Valid):** After plotting, iterating over `bp["caps"]` and calling `cap.set_visible(False)` on each cap artist hides the cap lines. This post-hoc approach is equivalent to `showcaps=False` but requires more code. It's useful when you want to hide caps after conditional logic or when modifying an existing plot.
- **Option D (Correct):** Both A (`showcaps=False` at creation) and C (post-hoc `.set_visible(False)`) are valid approaches to suppress cap display. D correctly identifies both as working solutions.

</details>

---

### 66. Question

A researcher creates a box plot to compare **response time variability** across four server regions. After examining the plot, they notice that Region C has a very wide box (large IQR) and Region D has a very narrow box (small IQR) but with several high outlier points above the upper whisker. What is the correct interpretation?

- [ ] **A.** Region D is worse because it has outliers
- [ ] **B.** Region C has higher variability in the middle 50% of response times; Region D is more consistent for most requests but has occasional extreme latency spikes
- [ ] **C.** Region C is worse because it has a larger box
- [ ] **D.** The outliers in Region D are errors in the data and should be ignored

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The presence of outliers alone doesn't make a region "worse." Outliers in latency data represent rare but severe latency events. Whether this is acceptable depends on the application's tolerance for occasional spikes vs. consistent moderate variability.
- **Option B (Correct):** Region C's wide box means its IQR is large — the middle 50% of response times are highly variable. Region D's narrow box means the middle 50% is consistent, but the outliers above the whisker represent occasional extreme latency events (perhaps due to cold cache misses, network congestion, or garbage collection pauses). For user experience, both patterns are concerning but in different ways: Region C has consistently inconsistent service; Region D provides mostly good service with rare but severe spikes.
- **Option C (Incorrect):** "Worse" requires defining what matters. A larger IQR means more variability in typical performance, which is indeed undesirable. But Region D's outlier spikes could also be undesirable (e.g., for real-time applications). "C is worse" is too simplistic.
- **Option D (Incorrect):** Outliers in box plots are not automatically data errors. In infrastructure monitoring, outlier latency values often represent real events (service degradation, network issues). Assuming they're errors without investigation would ignore genuine performance problems.

</details>

---

### 67. Question

A developer creates a box plot with custom `boxprops` styling but without `patch_artist=True`:

```python
bp = ax.boxplot(data,
                boxprops=dict(color="navy", linewidth=2))
```

What does `boxprops=dict(color="navy", linewidth=2)` style in this context?

- [ ] **A.** The fill color inside the box
- [ ] **B.** The color and width of the BOX OUTLINE (the Line2D rectangle edges) — since `patch_artist=False` by default, `color` affects the line color, not the fill
- [ ] **C.** Both the outline and fill of the box
- [ ] **D.** Nothing — `boxprops` requires `patch_artist=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Without `patch_artist=True`, the box has NO fill. Boxes are drawn as Line2D outlines. `color` in `boxprops` affects the LINE COLOR of the rectangular outline, not a fill color (there is no fill to color).
- **Option B (Correct):** Without `patch_artist=True`, the box is rendered as a set of `Line2D` segments forming a rectangle. `boxprops=dict(color="navy", linewidth=2)` sets the **outline color** to navy and the **outline thickness** to 2 points. The box appears as a navy-colored rectangle outline with no fill. This is distinct from the fill behavior you get with `patch_artist=True` and `set_facecolor()`.
- **Option C (Incorrect):** Without `patch_artist=True`, there is no fill. `boxprops` in this context only affects the Line2D outline properties.
- **Option D (Incorrect):** `boxprops` does NOT require `patch_artist=True`. It always applies Line2D properties to the box outline. The difference is: with `patch_artist=False`, only stroke properties (`color`, `linewidth`, `linestyle`) are meaningful in `boxprops`; with `patch_artist=True`, the box becomes a `PathPatch` and you can additionally use `facecolor`, `alpha`, etc.

</details>

---

### 68. Question

A developer creates a multi-panel box plot figure for a machine learning pipeline report and adds a unified super-title:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metric_data = [train_acc, val_acc, test_f1, test_auc]
titles = ["Train Accuracy", "Val Accuracy", "Test F1", "Test AUC"]

for ax, data, title in zip(axes.flatten(), metric_data, titles):
    bp = ax.boxplot(data, patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    ax.set_title(title, fontsize=11)
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.7)

plt.suptitle("Model Performance Distribution (5-Fold CV)", fontsize=14, y=1.02)
plt.tight_layout()
```

Why is `y=1.02` specified in `plt.suptitle()`?

- [ ] **A.** To increase the font size by 2%
- [ ] **B.** To position the super-title at 102% of the figure height — slightly above the topmost subplot titles to prevent overlap
- [ ] **C.** To center the super-title horizontally at position 1.02
- [ ] **D.** `y=1.02` triggers the super-title to appear on a separate page

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `y` is a positional parameter (y-coordinate in figure coordinates), not a font size multiplier. Font size is controlled by `fontsize=`.
- **Option B (Correct):** `plt.suptitle(y=1.02)` places the super-title at y=1.02 in **figure coordinates** (where 0=bottom, 1=top of figure). Setting y=1.02 positions the title slightly ABOVE the figure's top boundary (102% of the figure height), preventing it from overlapping with the individual subplot titles (`ax.set_title()`) in the top row. Without this offset, `tight_layout()` often causes the super-title to crowd the subplot titles.
- **Option C (Incorrect):** Horizontal centering uses the `x` parameter (default x=0.5). The `y` parameter controls vertical placement.
- **Option D (Incorrect):** `y=1.02` is purely a layout coordinate. It has no effect on pagination or page breaks.

</details>

---

### 69. Question

A developer wants to create a robust summary statistic comparison between the mean and median across multiple groups. They create box plots and add mean markers:

```python
bp = ax.boxplot([group_A, group_B, group_C],
                showmeans=True,
                meanprops=dict(marker="^", markerfacecolor="red", markersize=8))
```

For a **right-skewed** group (like Group A with a long upper tail), how would the mean marker typically appear relative to the median line?

- [ ] **A.** The mean marker is below the median line (at a lower y-value)
- [ ] **B.** The mean marker is above the median line (at a higher y-value) — the high outliers and upper tail pull the mean upward
- [ ] **C.** The mean marker is exactly on the median line (they always coincide)
- [ ] **D.** The mean marker appears outside the box above the upper whisker cap

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In a right-skewed distribution, the UPPER tail is longer and contains higher values. These extreme high values pull the arithmetic mean UPWARD, above the median. If the mean were below the median, this would indicate LEFT skew.
- **Option B (Correct):** For **right-skewed (positively skewed)** distributions, the mean is always greater than the median because high outlier values and the long upper tail disproportionately inflate the arithmetic mean. In the box plot, the red triangle marker (mean) would appear ABOVE the horizontal median line inside the box. The separation between the mean marker and median line is a visual measure of the distribution's asymmetry.
- **Option C (Incorrect):** The mean and median coincide only in perfectly symmetric, unimodal distributions. For any skewed distribution, they diverge. The direction of skew determines which is larger.
- **Option D (Incorrect):** The mean value is calculated from the full dataset including outliers. For a moderately skewed distribution, the mean is typically pulled toward (but not necessarily beyond) the long tail. It appears within the box or between the box and the whisker cap, not necessarily above the upper whisker. Only in extreme cases would the mean fall above the upper whisker.

</details>

---

### 70. Question

A developer saves a box plot figure for a publication:

```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot([sample_A, sample_B, sample_C],
           tick_labels=["Treatment A", "Treatment B", "Control"],
           patch_artist=True,
           notch=True)
ax.set_ylabel("Biomarker Level (ng/mL)")
ax.set_title("Biomarker Distribution by Treatment Group")

plt.tight_layout()
plt.savefig("biomarker_boxplot.pdf", dpi=300, bbox_inches="tight")
plt.show()
```

Why must `plt.savefig()` be called **before** `plt.show()`?

- [ ] **A.** `savefig` must always precede `show` as a Python syntax rule
- [ ] **B.** In non-interactive backends (e.g., Agg), `plt.show()` clears the current figure after displaying it — calling `savefig` afterward saves a blank figure
- [ ] **C.** `bbox_inches="tight"` only functions correctly before `show()`
- [ ] **D.** `dpi=300` is incompatible with interactive display

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** There is no general Python syntax rule requiring this order. It is a Matplotlib-specific operational behavior related to figure state management.
- **Option B (Correct):** In Matplotlib's non-interactive backends (commonly used in scripts and Jupyter notebooks), `plt.show()` renders the current figure to screen and then **clears the figure** (effectively resetting it). Any subsequent `plt.savefig()` call would save an **empty/blank figure** because the figure was cleared by `show()`. The correct workflow is always: create → customize → `savefig()` → `show()`. This ensures the figure content is saved before the display/clear cycle.
- **Option C (Incorrect):** `bbox_inches="tight"` computes bounding box from the current figure artists regardless of display state. It works correctly both before and after `show()` (though after `show()` the figure is cleared). The parameter itself has no timing dependency.
- **Option D (Incorrect):** `dpi=300` is a file-output resolution parameter and is completely independent of interactive display. `plt.show()` renders at screen resolution; `dpi=300` affects only the saved file quality.

</details>
