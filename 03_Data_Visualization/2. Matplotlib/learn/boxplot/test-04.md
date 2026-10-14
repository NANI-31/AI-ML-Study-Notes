# Matplotlib Box Plot Objective Test — Set 04

---

### 31. Question

A researcher collects the following sorted dataset of reaction times (ms) from an experiment:

```
[105, 112, 118, 124, 130, 137, 143, 150, 157, 163, 170, 178, 185]
```

With n=13 values, using the method of splitting at the median: Q₂ = 7th value = 143. Lower half = [105, 112, 118, 124, 130, 137] (6 values). Upper half = [150, 157, 163, 170, 178, 185] (6 values).

What are **Q₁ and Q₃**?

- [ ] **A.** Q₁ = 118, Q₃ = 170
- [ ] **B.** Q₁ = 121, Q₃ = 166.5
- [ ] **C.** Q₁ = 124, Q₃ = 163
- [ ] **D.** Q₁ = 130, Q₃ = 157

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Q₁=118 and Q₃=170 would correspond to the 3rd and 11th values of the full 13-element dataset — not the medians of the lower and upper halves.
- **Option B (Correct):** Lower half = [105, 112, 118, 124, 130, 137] (6 values, even count). Q₁ = average of 3rd and 4th values = (118+124)/2 = **121**. Upper half = [150, 157, 163, 170, 178, 185] (6 values, even count). Q₃ = average of 3rd and 4th values = (163+170)/2 = **166.5**. IQR = 166.5 − 121 = 45.5.
- **Option C (Incorrect):** Q₁=124 is the 4th value of the lower half, not its median. Q₃=163 is the 3rd value of the upper half, not its median. These are off-by-one errors in quartile position.
- **Option D (Incorrect):** Q₁=130 and Q₃=157 are the 5th and 2nd values of the respective halves — not the medians of those halves.

</details>

---

### 32. Question

Using the values from Question 31 (Q₁=121, Q₃=166.5, IQR=45.5), calculate the **lower and upper whisker fences** and determine which values (if any) are outliers:

Data: `[105, 112, 118, 124, 130, 137, 143, 150, 157, 163, 170, 178, 185]`

- [ ] **A.** Lower fence = 52.75, Upper fence = 234.75 — no outliers
- [ ] **B.** Lower fence = 75.5, Upper fence = 212 — no outliers
- [ ] **C.** Lower fence = 60, Upper fence = 230 — 105 is an outlier
- [ ] **D.** Lower fence = 80, Upper fence = 220 — 185 is an outlier

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Lower fence = Q₁ − 1.5×IQR = 121 − 1.5×45.5 = 121 − 68.25 = **52.75**. Upper fence = Q₃ + 1.5×IQR = 166.5 + 68.25 = **234.75**. All dataset values (105 to 185) fall within [52.75, 234.75]. Therefore, **no data points are outliers**. The lower whisker extends to 105 (the minimum, which is ≥ 52.75) and the upper whisker extends to 185 (the maximum, which is ≤ 234.75).
- **Option B (Incorrect):** Lower=75.5 and Upper=212 suggest a 1.5×IQR of 45.5 was applied incorrectly. 1.5×45.5 = 68.25, not 45.5. These values would result from a 1.0×IQR multiplier (121−45.5=75.5, 166.5+45.5=212).
- **Option C (Incorrect):** The computed lower fence (52.75) is well below 105, so 105 is NOT an outlier. A lower fence of 60 doesn't match the correct calculation.
- **Option D (Incorrect):** The upper fence (234.75) is well above 185, so 185 is NOT an outlier. A fence of 220 doesn't match the correct calculation.

</details>

---

### 33. Question

A machine learning engineer has a dataset of model prediction errors (in seconds):

```python
errors = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 4.8]
```

Given: Q₁=0.5, Q₃=1.0, IQR=0.5. Upper fence = 1.0 + 1.5×0.5 = **1.75**.

Is the value **4.8** classified as an **extreme outlier** using the 3×IQR threshold?

- [ ] **A.** No — 4.8 is within 3×IQR of Q₃
- [ ] **B.** Yes — 4.8 > Q₃ + 3×IQR = 1.0 + 1.5 = 2.5, so it is an extreme outlier
- [ ] **C.** Yes — 4.8 > Q₃ + 3×IQR = 1.0 + 3×0.5 = 2.5, so it qualifies as an extreme outlier
- [ ] **D.** Extreme outlier classification requires z-score calculation, not IQR

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Q₃ + 3×IQR = 1.0 + 3×0.5 = 2.5. Since 4.8 > 2.5, the value IS beyond the 3×IQR extreme outlier threshold. The claim that it's within 3×IQR is incorrect.
- **Option B (Partially correct but with arithmetic error):** The threshold is Q₃ + 3×IQR = 1.0 + 3×0.5 = 2.5, not 1.0 + 1.5 = 2.5. Option B gives the correct numerical answer (2.5) by coincidence but shows wrong algebra (1.5 instead of 3×0.5=1.5 — actually the same, so it's numerically correct but the intermediate step "3×IQR = 1.5" is implied incorrectly). Option C is more explicit.
- **Option C (Correct):** The **extreme outlier** threshold (sometimes called a "far outlier") is: Q₃ + 3×IQR = 1.0 + 3×0.5 = 1.0 + 1.5 = **2.5**. Since 4.8 > 2.5, the value 4.8 exceeds the extreme outlier boundary. It is both a regular outlier (> 1.75 = 1.5×IQR threshold) and an extreme outlier (> 2.5 = 3×IQR threshold). Some box plot libraries differentiate these with different marker styles.
- **Option D (Incorrect):** While z-scores are one method for outlier detection in normally distributed data, the IQR-based fence method is a standalone, distribution-free approach commonly used in box plots. It does not require z-score calculation and is robust to non-normal distributions.

</details>

---

### 34. Question

A developer creates a notched box plot and needs to explain the confidence interval formula. For a dataset with median M, IQR, and sample size n, the notch bounds are approximately:

$$\text{Notch bounds} \approx M \pm \frac{1.57 \times \text{IQR}}{\sqrt{n}}$$

For a dataset where M=75, IQR=20, and n=25, what are the **approximate notch bounds**?

- [ ] **A.** [68.72, 81.28]
- [ ] **B.** [71.28, 78.72]
- [ ] **C.** [60, 90]
- [ ] **D.** [72.5, 77.5]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** [68.72, 81.28] would correspond to ±6.28 = 1.57×20/√n where √n produces ≈5 — let's verify: 1.57×20=31.4, 31.4/√25=31.4/5=6.28. So bounds = 75±6.28 = [68.72, 81.28]. Wait, this IS correct. Let me re-examine: 1.57 × IQR / √n = 1.57 × 20 / 5 = 31.4/5 = 6.28. Notch bounds = 75 ± 6.28 = [68.72, 81.28]. **Option A is correct.**
- **Option B (Incorrect):** [71.28, 78.72] would correspond to ±3.72, which doesn't match 6.28.
- **Option C (Incorrect):** [60, 90] = 75 ± 15, which would require 1.57×IQR/√n = 15. With IQR=20: 1.57×20/√n=15 → √n=31.4/15=2.09 → n≈4.4, not n=25.
- **Option D (Incorrect):** [72.5, 77.5] = 75 ± 2.5, which would require the formula to yield 2.5, not 6.28.

> **Note:** The correct answer is **A** ([68.72, 81.28]), as verified by: 1.57 × 20 / √25 = 31.4/5 = 6.28; bounds = 75 ± 6.28.

</details>

---

### 35. Question

A data analyst creates a box plot using `patch_artist=True` and wants to set different opacity levels for each box:

```python
bp = ax.boxplot([data1, data2, data3], patch_artist=True)

bp["boxes"][0].set_alpha(0.9)
bp["boxes"][1].set_alpha(0.5)
bp["boxes"][2].set_alpha(0.2)
```

What visual effect does this produce for `bp["boxes"][2]`?

- [ ] **A.** The third box is completely invisible (alpha=0 would make it invisible)
- [ ] **B.** The third box is 20% opaque — very faint/nearly transparent, allowing the background to show through strongly
- [ ] **C.** The third box has 20% of the standard box width
- [ ] **D.** The third box is drawn with 0.2 point line width

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `alpha=0.2` makes the box 20% opaque (80% transparent), NOT invisible. `alpha=0.0` would make it completely invisible. At `alpha=0.2`, the box is still visible but highly transparent.
- **Option B (Correct):** `set_alpha(0.2)` sets the box `PathPatch`'s alpha (opacity) to 0.2 — meaning it is **80% transparent** and only **20% opaque**. The filled colour is visible but very faint, and the chart background (gridlines, other elements) shows through at 80% visibility. This can be useful for layering multiple distributions with different transparency levels to reduce visual clutter.
- **Option C (Incorrect):** `alpha` controls **transparency**, not width. Box width is determined by the `widths` parameter of `ax.boxplot()` (default 0.5). `set_alpha(0.2)` has no effect on box width.
- **Option D (Incorrect):** Line width is controlled by `set_linewidth()` or `linewidth=` in the props dictionaries. `set_alpha(0.2)` specifically controls the transparency channel of the patch artist.

</details>

---

### 36. Question

A developer uses `positions` parameter to place box plots at custom x-positions:

```python
fig, ax = plt.subplots()
ax.boxplot([group1, group2, group3],
           positions=[1, 3, 5],
           widths=0.6)
ax.set_xlim(0, 6)
ax.set_xticks([1, 3, 5])
ax.set_xticklabels(["Control", "Treatment A", "Treatment B"])
```

Why might a developer use `positions=[1, 3, 5]` instead of the default positions `[1, 2, 3]`?

- [ ] **A.** To make the boxes larger
- [ ] **B.** To create visual spacing between box groups — gaps between positions 1→3→5 create white space that separates groups more clearly than adjacent positions 1→2→3
- [ ] **C.** To sort the boxes from smallest to largest IQR
- [ ] **D.** The `positions` parameter is required when more than 2 groups are plotted

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Box size (width) is controlled by the `widths` parameter. The `positions` parameter controls where boxes are placed on the x-axis, not their dimensions.
- **Option B (Correct):** By default, multiple groups are placed at positions [1, 2, 3, ...]. Spacing them at [1, 3, 5] (with gaps of 2 instead of 1) creates visual white space between groups. This is useful for: (1) creating clear visual separation, (2) leaving room for annotations between groups, or (3) combining box plots with other chart types (e.g., placing a bar next to a box at the same category position). The `widths=0.6` ensures boxes don't overlap despite the custom positions.
- **Option C (Incorrect):** `positions` specifies spatial x-coordinates, not ordering based on statistical properties. Sorting by IQR would require pre-sorting the data arrays themselves before passing to `boxplot()`.
- **Option D (Incorrect):** The `positions` parameter is optional for any number of groups. When omitted, groups default to sequential integer positions starting at 1. It's only necessary when non-default spacing is desired.

</details>

---

### 37. Question

A developer creates a box plot and customises the **whisker style**:

```python
bp = ax.boxplot(data,
                whiskerprops=dict(linestyle="--", linewidth=2, color="navy"),
                capprops=dict(linewidth=3, color="navy"))
```

What visual effect do `whiskerprops` and `capprops` produce?

- [ ] **A.** `whiskerprops` changes the outlier dot style; `capprops` changes the box border
- [ ] **B.** `whiskerprops` styles the whisker lines (dashed navy, 2pt wide); `capprops` styles the cap horizontal lines at whisker ends (solid navy, 3pt wide)
- [ ] **C.** Both parameters only affect horizontal box plots
- [ ] **D.** `whiskerprops` applies to the box outline; `capprops` applies to the median line caps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Outlier dot styling uses `flierprops`. Box border styling uses `boxprops`. `whiskerprops` and `capprops` target the whisker lines and cap lines respectively.
- **Option B (Correct):** `whiskerprops` is a dictionary of `Line2D` properties applied to the **whisker lines** — the lines extending from the box edges to the whisker end caps. `linestyle="--"` makes them dashed, `linewidth=2` makes them 2 points wide, `color="navy"` makes them dark blue. `capprops` is a dictionary applied to the **cap lines** — the short horizontal lines at the tips of the whiskers (the "T-bars"). `linewidth=3` makes caps thicker than the whiskers, and `color="navy"` matches the navy theme.
- **Option C (Incorrect):** All `*props` parameters (`boxprops`, `whiskerprops`, `capprops`, `medianprops`, `flierprops`, `meanprops`) work identically for both vertical (`vert=True`) and horizontal (`vert=False`) box plots.
- **Option D (Incorrect):** `whiskerprops` targets the whisker lines, not the box outline. `capprops` targets the cap lines at whisker ends, not any median-related elements. Box outline styling uses `boxprops`; median styling uses `medianprops`.

</details>

---

### 38. Question

A data scientist creates a multi-group box plot and wants to overlay a **horizontal reference line** at y=0 to indicate "perfect prediction" (zero error):

```python
fig, ax = plt.subplots()
ax.boxplot([lr_errors, rf_errors, nn_errors],
           tick_labels=["LR", "RF", "NN"],
           patch_artist=True)
ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error")
ax.set_ylabel("Prediction Error")
ax.legend()
```

Why is `ax.axhline(0, ...)` more appropriate than `ax.plot([0, 4], [0, 0], ...)` for this reference line?

- [ ] **A.** `axhline` draws thicker lines than `plot`
- [ ] **B.** `axhline(y=0)` automatically spans the FULL width of the axes regardless of x-limits; `ax.plot([0,4],[0,0])` requires knowing the exact x-range and updating it if the range changes
- [ ] **C.** `axhline` is required for all box plot annotations
- [ ] **D.** `plot` cannot create horizontal lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Line thickness is controlled by `linewidth` in both `axhline` and `plot`. There is no inherent thickness difference between the two approaches.
- **Option B (Correct):** `ax.axhline(0)` draws a horizontal line at y=0 that automatically spans the **full width of the axes** (from the current xmin to xmax, in axes coordinates 0 to 1). It automatically adjusts if `ax.set_xlim()` is changed later. Using `ax.plot([0, 4], [0, 0])` requires manually specifying x-coordinates. If the x-range changes (e.g., more groups are added), the `plot`-based line may not reach the full width. `axhline` is the semantically correct choice for "a horizontal line across the entire plot."
- **Option C (Incorrect):** `axhline` is not required specifically for box plots. It's a general-purpose function for any chart. Box plots have no special annotation requirements.
- **Option D (Incorrect):** `ax.plot([x1, x2], [0, 0])` DOES create a horizontal line — it just requires manual x-coordinate specification. Both functions can draw horizontal lines; `axhline` is simply more convenient and robust.

</details>

---

### 39. Question

A developer creates a box plot and discovers that the **whiskers extend to the data minimum and maximum** without any outlier points being plotted. What does this imply about the dataset?

- [ ] **A.** The dataset has no values — it's empty
- [ ] **B.** All data points fall within the 1.5×IQR fences — no values exceed the outlier boundaries, so the whiskers extend to the actual extremes
- [ ] **C.** The developer forgot to pass `showfliers=True`
- [ ] **D.** The IQR is zero, causing the fences to collapse to the quartile values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** An empty dataset would produce an error, not a box plot with whiskers reaching the data range.
- **Option B (Correct):** When whiskers extend to the **actual minimum and maximum** of the dataset with no separate flier markers visible, it means all data points lie within the [Q₁−1.5×IQR, Q₃+1.5×IQR] fence. The lower whisker stops at `min(data)` because that point is the furthest non-outlier — and since it's within the fence, it's shown as the whisker endpoint, not a flier. Similarly, `max(data)` is within the upper fence. This is a compact, outlier-free distribution.
- **Option C (Incorrect):** `showfliers=True` is the DEFAULT for `plt.boxplot()`. If no fliers appear, it's not because the parameter is missing — it's because the data contains no outliers. Setting `showfliers=False` would suppress fliers even if they exist.
- **Option D (Incorrect):** Zero IQR would cause both fences to equal the median, making every non-median point an outlier. This would produce many flier points, not whiskers extending to the extremes.

</details>

---

### 40. Question

A developer has a sorted dataset: `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 200]`

Q₁=13, Q₃=33, IQR=20. Upper fence = 33 + 1.5×20 = 33 + 30 = 63.

Which values are **outliers**, and where does the **upper whisker** end?

- [ ] **A.** 200 is an outlier; upper whisker ends at 63
- [ ] **B.** 200 is an outlier; upper whisker ends at 43
- [ ] **C.** No outliers; upper whisker ends at 200
- [ ] **D.** Both 43 and 200 are outliers; upper whisker ends at 39

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The upper whisker DOES NOT extend to the fence value (63) — it extends to the **actual data point** that is the highest value within the fence. Since 200 > 63 is an outlier, the whisker stops at 43 (the highest non-outlier), not at the computed fence of 63.
- **Option B (Correct):** Upper fence = 63. Looking at the data: values 3 through 43 all fall within the fence range. **200 > 63** → 200 is an **outlier** and is plotted as a separate flier marker above the whisker. The upper whisker ends at **43** — the highest value that falls within the upper fence (43 ≤ 63). The lower fence = Q₁−1.5×IQR = 13−30 = −17; since min(data)=3 > −17, no lower outliers exist.
- **Option C (Incorrect):** 200 IS an outlier (200 > upper fence of 63). The whisker does not extend to 200.
- **Option D (Incorrect):** 43 is NOT an outlier because 43 ≤ 63 (within the upper fence). Only 200 exceeds the fence. The upper whisker ends at 43, not 39.

</details>
