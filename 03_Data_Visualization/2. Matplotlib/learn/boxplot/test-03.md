# Matplotlib Box Plot Objective Test — Set 03

---

### 21. Question

A data scientist creates a multi-group box plot to compare delivery times for three shipping carriers:

```python
carrier_A = [2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3]
carrier_B = [1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 4.5]
carrier_C = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

fig, ax = plt.subplots()
bp = ax.boxplot(
    [carrier_A, carrier_B, carrier_C],
    tick_labels=["Carrier A", "Carrier B", "Carrier C"]
)
```

How many **`Line2D` objects** does `bp["whiskers"]` contain for this 3-group plot?

- [ ] **A.** 3 (one per group)
- [ ] **B.** 6 (two per group: one lower whisker, one upper whisker)
- [ ] **C.** 9 (three per group)
- [ ] **D.** 4 (two whiskers × two ends)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Each box plot group has TWO whiskers (one extending below Q₁, one extending above Q₃). With 3 groups, there are 3×2 = 6 whisker Line2D objects, not 3.
- **Option B (Correct):** Each box in a Matplotlib box plot generates **two whisker Line2D objects**: a lower whisker (from Q₁ to the lower fence data point) and an upper whisker (from Q₃ to the upper fence data point). With 3 groups, `bp["whiskers"]` contains **6** Line2D objects: indices [0,1] for Carrier A, [2,3] for Carrier B, [4,5] for Carrier C. Similarly, `bp["caps"]` contains 6 Line2D objects (two caps per group), and `bp["boxes"]` contains 3 objects (one per group).
- **Option C (Incorrect):** Three per group would imply each group has three whisker elements, which does not correspond to any standard box plot structure.
- **Option D (Incorrect):** 4 whiskers would only apply to a 2-group plot (2 groups × 2 whiskers each = 4). With 3 groups, there are 6.

</details>

---

### 22. Question

A developer inspects the returned `bp` dictionary from a multi-group box plot and wants to change the **median line colour** of only the second group (Carrier B) to red:

```python
bp["medians"][1].set_color("red")
```

Is this the correct approach, and will it work?

- [ ] **A.** No — median colours must be set globally via `medianprops=dict(color="red")`
- [ ] **B.** Yes — `bp["medians"]` is a list of Line2D objects, one per group; `bp["medians"][1]` accesses Carrier B's median and `set_color("red")` changes only that line
- [ ] **C.** No — `bp["medians"]` contains Path objects, not Line2D objects; `set_color` is not a valid method
- [ ] **D.** Yes — but only if `patch_artist=True` was used in the original `boxplot()` call

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `medianprops=dict(color="red")` in the `boxplot()` call applies the same colour to ALL medians in the plot. For per-group customisation, you must access individual elements from the `bp` dictionary after plotting.
- **Option B (Correct):** `bp["medians"]` is a Python list of `Line2D` objects, one per group in order. For a 3-group plot, `bp["medians"][0]` = Carrier A, `bp["medians"][1]` = Carrier B, `bp["medians"][2]` = Carrier C. `Line2D.set_color("red")` is a valid method that changes the line colour. This targeted approach allows per-group median styling without affecting others.
- **Option C (Incorrect):** Median lines are always `Line2D` objects regardless of whether `patch_artist=True` is used. Only the boxes change from `Line2D` to `PathPatch` when `patch_artist=True` is set — the medians, whiskers, caps, and fliers remain `Line2D` objects.
- **Option D (Incorrect):** `patch_artist=True` affects only the box type (Line2D → PathPatch). Median lines remain `Line2D` objects whether or not `patch_artist=True` is used, and their colour can always be changed via `bp["medians"][i].set_color(...)`.

</details>

---

### 23. Question

A developer creates a box plot with the `sym` shortcut to style outliers:

```python
ax.boxplot(data, sym="r+")
```

What does `sym="r+"` specify?

- [ ] **A.** Red triangle-up markers for outliers
- [ ] **B.** Red plus-sign (`+`) markers for outliers
- [ ] **C.** Removes all outlier symbols (suppresses fliers)
- [ ] **D.** Uses the default symbol but makes all elements red

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Red triangle-up markers would require `sym="r^"`. The `^` character in Matplotlib's short marker specification denotes a triangle-up. The `+` character specifically denotes a plus sign.
- **Option B (Correct):** `sym="r+"` uses the compact Matplotlib colour+marker string format: `"r"` = red colour, `"+"` = plus-sign marker shape. Outlier data points beyond the whisker fences are displayed as red `+` marks instead of the default `o` (circle) markers. This is equivalent to `flierprops=dict(marker="+", color="red")`.
- **Option C (Incorrect):** To suppress (hide) all flier markers, you would use `showfliers=False` in the `boxplot()` call. `sym=""` (empty string) can also suppress outliers in some contexts. `sym="r+"` does not suppress outliers — it styles them.
- **Option D (Incorrect):** `sym` is specifically a shorthand for **flier (outlier) symbol** styling only. It does not affect the box, median line, whiskers, or caps. Colouring all elements red would require individual `props` dictionaries for each component.

</details>

---

### 24. Question

A developer creates a box plot and wants to **hide the outlier markers** entirely to show a cleaner view:

```python
ax.boxplot(data, showfliers=False)
```

What is the visual effect of `showfliers=False`?

- [ ] **A.** Outlier values are removed from the dataset before computing quartiles
- [ ] **B.** The whiskers now extend to the absolute minimum and maximum of the full dataset
- [ ] **C.** Outlier markers are not drawn, but the whiskers still extend only to the last non-outlier data point; the outliers are hidden but not eliminated from statistics
- [ ] **D.** Both the whiskers and outlier markers are hidden, showing only the box and median

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `showfliers=False` is purely a **display** setting. It does not modify the dataset or recompute the quartiles without the outliers. Q₁, Q₂, Q₃, and IQR are computed from the full dataset including all extreme values. Only the rendering of outlier markers is suppressed.
- **Option B (Incorrect):** Whisker extension rules remain unchanged with `showfliers=False`. The whiskers still extend to the last data point within the 1.5×IQR fence, not to the absolute data extremes. If a value is beyond the fence, it remains invisible but the whisker still stops at the last non-outlier value.
- **Option C (Correct):** `showfliers=False` only suppresses the **visual rendering** of outlier markers. The box (Q₁–Q₃), median line, whiskers (to the last non-outlier point), and caps are all drawn normally. The outlier values remain in the dataset for quartile computation but are not shown as individual plotted points. This gives a cleaner visual while still faithfully representing the box and whisker positions.
- **Option D (Incorrect):** Whiskers are not hidden by `showfliers=False`. Only the individual outlier marker points are hidden. The whiskers and caps remain visible.

</details>

---

### 25. Question

A developer's box plot code has a subtle bug. Identify the issue:

```python
groups = [[10, 15, 20, 25, 30],
          [12, 18, 22, 28, 35],
          [8,  16, 21, 26, 33]]

fig, ax = plt.subplots()
ax.boxplot(groups, tick_labels=["Morning", "Afternoon", "Evening", "Night"])
plt.show()
```

What is wrong with this code?

- [ ] **A.** `groups` should be a NumPy array, not a list of lists
- [ ] **B.** The `tick_labels` list has 4 elements but there are only 3 groups — this will cause an error or misaligned labels
- [ ] **C.** `plt.boxplot()` cannot accept a list of lists — only a flat 1D array
- [ ] **D.** `patch_artist=True` is required when passing multiple groups

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.boxplot()` accepts Python lists of lists without issue. NumPy arrays are not required — lists work identically. Matplotlib internally converts list-of-lists to the appropriate structure.
- **Option B (Correct):** The `groups` list has **3** sub-lists (Morning, Afternoon, Evening), but `tick_labels` has **4** elements ("Morning", "Afternoon", "Evening", "Night"). Matplotlib expects `len(tick_labels) == len(data groups)`. This mismatch raises a `ValueError` in modern Matplotlib versions. The fix is to remove "Night" from `tick_labels` (making it 3 elements) or add a fourth data group.
- **Option C (Incorrect):** `plt.boxplot()` is specifically designed to accept **lists of arrays** (or lists of lists) to create multi-group box plots. Passing `[group1, group2, group3]` is the standard way to create side-by-side box plots. Flat 1D arrays create single box plots.
- **Option D (Incorrect):** `patch_artist=True` is optional and unrelated to whether multiple groups are passed. Multi-group box plots work without `patch_artist=True`; the boxes are just drawn as line outlines rather than filled patches.

</details>

---

### 26. Question

A researcher analyzes the distribution of model residuals. Looking at a box plot, the **median line is very close to Q₁** (near the bottom of the box) and the **upper whisker is much longer** than the lower whisker. What does this indicate about the residual distribution?

- [ ] **A.** The distribution is approximately symmetric (normal)
- [ ] **B.** The distribution is **right-skewed** (positive skew) — the tail extends further toward higher positive residual values
- [ ] **C.** The distribution is **left-skewed** (negative skew) — the tail extends further toward lower negative residual values
- [ ] **D.** The data has no outliers and the model performs well

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A symmetric distribution would show the median line centered in the box (equidistant from Q₁ and Q₃) and whiskers of approximately equal length. The described pattern (median near Q₁, long upper whisker) is a clear indicator of asymmetry.
- **Option B (Correct):** When the median line sits **close to Q₁** and the **upper whisker (or upper half of the box) is longer**, the data has a **positive (right) skew**. This means more data is clustered at lower values with a long tail of higher values. For residuals, this indicates the model frequently makes small errors but occasionally makes large positive errors — the distribution is not symmetric around zero.
- **Option C (Incorrect):** Left-skew (negative skew) would show the **opposite** pattern: the median line would be close to Q₃ (near the top of the box), and the **lower whisker** would be longer than the upper. The tail would extend toward more negative values.
- **Option D (Incorrect):** Skewness and outliers are separate attributes. The described pattern indicates distributional asymmetry (skew), not the absence of outliers. A well-performing model's residual box plot should be centered near 0 and approximately symmetric — neither of which is described here.

</details>

---

### 27. Question

A developer compares median line position within the box across two datasets:

- **Dataset X**: Median very close to Q₃ (near top of box), long lower whisker
- **Dataset Y**: Median approximately in the center of the box, whiskers of similar length

Which dataset is **left-skewed (negatively skewed)**?

- [ ] **A.** Dataset Y — center median indicates left skew
- [ ] **B.** Dataset X — median near Q₃ with a longer lower whisker indicates the data trails toward lower values (left skew)
- [ ] **C.** Both datasets — any visible whiskers indicate skew
- [ ] **D.** Neither — box plots cannot detect skewness direction

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Dataset Y with a centered median and equal-length whiskers indicates a **symmetric** (approximately normal) distribution, not left skew.
- **Option B (Correct):** **Left-skewed (negatively skewed)** distributions have: (1) a longer tail toward lower (negative) values, (2) more data concentrated at higher values, causing the median to sit **closer to Q₃** (the upper quartile). The **lower whisker** is longer because the sparse tail pulls it farther below Q₁. Dataset X perfectly matches this pattern: median near Q₃ + long lower whisker = left/negative skew.
- **Option C (Incorrect):** Whisker length variation indicates distributional asymmetry, but equal-length whiskers (Dataset Y) indicate symmetry, not skew. The presence of whiskers alone does not indicate skew — all box plots have whiskers.
- **Option D (Incorrect):** Box plots are specifically used to assess skewness direction. The position of the median line within the box and the relative lengths of the whiskers are the primary visual indicators of skewness in box plots.

</details>

---

### 28. Question

A developer creates a box plot and uses `tick_labels` with a custom label list:

```python
data_groups = [exam_physics, exam_chemistry, exam_biology]

fig, ax = plt.subplots()
ax.boxplot(data_groups,
           tick_labels=["Physics", "Chemistry", "Biology"])
ax.set_ylabel("Score")
```

What is the difference between `tick_labels` (in `boxplot()`) and calling `ax.set_xticklabels(["Physics", "Chemistry", "Biology"])` afterward?

- [ ] **A.** There is no difference — they produce identical results
- [ ] **B.** `tick_labels` in `boxplot()` sets labels directly on the correct positions; `set_xticklabels` works but requires `ax.set_xticks()` to be called first with explicit positions
- [ ] **C.** `tick_labels` uses a different font than `set_xticklabels`
- [ ] **D.** `tick_labels` is deprecated — always use `set_xticklabels`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For the simple use case of labeling groups at positions 1, 2, 3 (Matplotlib's default box positions), both approaches produce the same result. `tick_labels=["Physics", "Chemistry", "Biology"]` in the `boxplot()` call internally maps labels to the group positions. `ax.set_xticklabels(["Physics", "Chemistry", "Biology"])` after the call does the same. The practical difference is timing: `tick_labels` is set at plot creation time; `set_xticklabels` modifies it after. For standard use, both are equivalent.
- **Option B (Partially correct):** While it's true that `set_xticklabels` can require `set_xticks` for non-standard positions, for the default box plot positions (1, 2, 3, ...), Matplotlib automatically places ticks at integer positions and `set_xticklabels` maps labels to them correctly without explicit `set_xticks`. This distinction matters only for non-default `positions=` arguments.
- **Option C (Incorrect):** Both methods use the same default tick label font. There is no font difference between the two approaches.
- **Option D (Incorrect):** `tick_labels` is NOT deprecated. It is the recommended way to set labels during `boxplot()` creation. Older code may use `labels=` (which was the previous parameter name), but `tick_labels` is current.

</details>

---

### 29. Question

A developer creates a box plot with `notch=True`:

```python
data = [15, 18, 20, 22, 25, 27, 30, 32, 35, 38, 40]
fig, ax = plt.subplots()
ax.boxplot(data, notch=True)
plt.show()
```

What does the notch (V-shape pinch) in the box visually represent?

- [ ] **A.** The exact location of the mean
- [ ] **B.** The 95% confidence interval around the median — the V-shaped indentation marks the upper and lower bounds of the median estimate's confidence interval
- [ ] **C.** The region where the data is most dense (modal area)
- [ ] **D.** The boundary between the first and second standard deviations

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The notch does not indicate the mean. The center line of the notch is the median (same as in a non-notched box plot). The notch itself represents the confidence interval around the MEDIAN, not the mean's location.
- **Option B (Correct):** `notch=True` creates a V-shaped or hourglass-shaped indentation in the box around the median line. The notch represents the **approximate 95% confidence interval** (CI) for the median, computed as: Notch bounds ≈ Median ± 1.57 × IQR / √n. The notch is a graphical significance test: if the notches of two box plots **do not overlap**, the medians are statistically significantly different at approximately 95% confidence.
- **Option C (Incorrect):** The modal area (highest density region) is not directly shown by box plots. Violin plots (using kernel density estimation) better represent the modal structure of a distribution. Box plots show quartile-based summaries, not density.
- **Option D (Incorrect):** Standard deviations are not part of the box plot framework. Box plots use quartile-based statistics (Q₁, Q₂, Q₃, IQR). The notch is a confidence interval for the median, not a standard deviation boundary.

</details>

---

### 30. Question

A researcher creates two notched box plots to compare student performance across two teaching methods:

```python
method_A = [65, 70, 72, 75, 78, 80, 82, 85, 88, 90]
method_B = [55, 60, 63, 66, 68, 70, 73, 76, 80, 85]

fig, ax = plt.subplots()
ax.boxplot([method_A, method_B],
           notch=True,
           tick_labels=["Method A", "Method B"])
```

The resulting plot shows that the notches of the two boxes **overlap significantly**. What can the researcher conclude?

- [ ] **A.** The medians are proven to be statistically identical
- [ ] **B.** There is no statistically significant difference in medians at the ~95% confidence level — the data is consistent with the medians being equal
- [ ] **C.** The medians are significantly different because the boxes themselves are at different heights
- [ ] **D.** Overlapping notches mean one method is superior to the other

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Overlapping notches do NOT prove that the medians are identical — they only indicate **insufficient evidence** to reject the null hypothesis that medians are equal (at ~95% confidence). It is possible that the medians ARE different but the sample sizes are too small or the variance too large to detect the difference with this test.
- **Option B (Correct):** When notches overlap, the conclusion is: "We **cannot reject** the null hypothesis that the two medians are equal at approximately 95% confidence." This is a **non-significant** result — it does not prove equality, only that the data doesn't provide strong evidence of a difference. The notch overlap test is a graphical approximation of a significance test for medians.
- **Option C (Incorrect):** The positions (heights) of the boxes reflect the absolute median values of each dataset, not the statistical significance of the difference. Box height difference is expected since the datasets have different medians (Method A has higher scores). Statistical significance is assessed by NOTCH overlap, not box position.
- **Option D (Incorrect):** Overlapping notches suggest there is NO statistically significant evidence that one method is superior. If anything, overlapping notches suggest the researcher cannot conclude one method is better based on median comparison alone.

</details>
