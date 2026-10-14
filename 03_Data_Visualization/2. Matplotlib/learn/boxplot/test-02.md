# Matplotlib Box Plot Objective Test — Set 02

---

### 11. Question

A researcher collects the following sorted dataset of protein concentrations (μg/mL):

```
[12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
```

Using the standard method (median of lower/upper halves), what are **Q₁** and **Q₃**?

- [ ] **A.** Q₁ = 18, Q₃ = 36
- [ ] **B.** Q₁ = 19.5, Q₃ = 37.5
- [ ] **C.** Q₁ = 21, Q₃ = 36
- [ ] **D.** Q₁ = 15, Q₃ = 39

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For n=11, the median (Q₂) is the 6th value = 27. The lower half (excluding the median) = [12, 15, 18, 21, 24] (5 values). Q₁ = median of lower half = 3rd value = **18**. The upper half (excluding median) = [30, 33, 36, 39, 42] (5 values). Q₃ = median of upper half = 3rd value = **36**. So Q₁=18, Q₃=36, IQR=18.
- **Option B (Incorrect):** 19.5 = (18+21)/2 and 37.5 = (36+39)/2 would apply if the quartiles were computed by averaging adjacent values — this occurs with some interpolation methods or even-count halves. With 5-element halves (odd count), the median is the single middle value (18 for lower half, 36 for upper half).
- **Option C (Incorrect):** Q₁=21 would be the 4th value of the lower half, not the median (3rd). Q₃=36 is correct, but Q₁ is wrong.
- **Option D (Incorrect):** Q₁=15 and Q₃=39 correspond to the 2nd and 10th values of the full dataset — not the quartile positions. These values would be reached if quartile positions were calculated differently (e.g., using percentile positions 2 and 10 out of 11).

</details>

---

### 12. Question

Using the dataset from Question 11 (Q₁=18, Q₃=36, IQR=18), calculate the **lower and upper whisker fences**:

- [ ] **A.** Lower fence = −9, Upper fence = 63
- [ ] **B.** Lower fence = 0, Upper fence = 54
- [ ] **C.** Lower fence = −9, Upper fence = 60
- [ ] **D.** Lower fence = 12, Upper fence = 42

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The 1.5×IQR fences are: Lower fence = Q₁ − 1.5×IQR = 18 − 1.5×18 = 18 − 27 = **−9**. Upper fence = Q₃ + 1.5×IQR = 36 + 1.5×18 = 36 + 27 = **63**. These fences define the outer boundaries. Data points beyond these fences would be classified as outliers (fliers). The whiskers themselves extend to the most extreme data points WITHIN the fences.
- **Option B (Incorrect):** Lower=0 and Upper=54 would correspond to 1.5×IQR offsets of 18 (not 27). This error arises from using IQR=12 instead of IQR=18, or applying a 1.0×IQR multiplier instead of 1.5×.
- **Option C (Incorrect):** Lower fence = −9 is correct, but Upper fence = 60 is wrong. 36 + 1.5×18 = 36 + 27 = 63, not 60.
- **Option D (Incorrect):** Lower=12 and Upper=42 are simply the minimum and maximum of the dataset. The whiskers extend to these values because no outliers exist (all data falls within [−9, 63]), but the FENCES themselves are −9 and 63 — not the actual data extremes.

</details>

---

### 13. Question

For the dataset from Question 11, the lower fence is −9 and the upper fence is 63. The whisker ends are drawn at the actual data values. Where do the **whiskers** extend to?

- [ ] **A.** Lower whisker extends to Q₁−1.5×IQR = −9; Upper whisker extends to Q₃+1.5×IQR = 63
- [ ] **B.** Lower whisker extends to the minimum data value within the fence = 12; Upper whisker extends to the maximum data value within the fence = 42
- [ ] **C.** Lower whisker extends to 0 (the absolute minimum); Upper whisker extends to 100 (the axis maximum)
- [ ] **D.** Lower whisker extends to Q₁ = 18; Upper whisker extends to Q₃ = 36 (whiskers stay at box edges)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This is the most common misconception. Whiskers do NOT extend to the computed fence values (Q₁−1.5×IQR and Q₃+1.5×IQR) unless a data point happens to land exactly at the fence. Whiskers extend to the **actual data value** that is furthest from the median but still WITHIN the fence. If no data falls at −9, the whisker stops at the nearest real data point (here: 12).
- **Option B (Correct):** The dataset's minimum is 12 and maximum is 42. Both 12 and 42 fall within the fences [−9, 63] — so neither is an outlier. The lower whisker extends to the furthest data point ≥ lower fence = **12**. The upper whisker extends to the furthest data point ≤ upper fence = **42**. No data points are beyond the fences, so no flier markers appear.
- **Option C (Incorrect):** Whiskers have nothing to do with absolute values of 0 or axis scale limits. They are determined entirely by the data and the 1.5×IQR rule relative to Q₁ and Q₃.
- **Option D (Incorrect):** If whiskers stayed at the box edges (Q₁ and Q₃), there would be no whiskers at all — only a box. The purpose of whiskers is to extend beyond the box to show the data range excluding outliers.

</details>

---

### 14. Question

A developer adds a new data point of **95** to the dataset from Question 11. Given the original fences (lower = −9, upper = 63), is 95 an **outlier**?

- [ ] **A.** No — 95 is within 2 standard deviations of the mean
- [ ] **B.** Yes — 95 > upper fence of 63, so it is plotted as an individual flier marker
- [ ] **C.** No — 95 is the new maximum and becomes the new upper whisker end
- [ ] **D.** Yes — 95 is more than Q₃ × 2 from the median

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Box plots use the IQR-based fence rule (1.5×IQR), not standard deviations. Whether 95 is within 2 standard deviations is irrelevant to box plot outlier classification. Box plots are specifically designed as a ROBUST alternative that does not require normal distribution assumptions.
- **Option B (Correct):** The upper fence is 63. Since 95 > 63, the value 95 lies beyond the upper fence and is classified as an **outlier (flier)**. Matplotlib plots it as an individual marker point (by default a circle `o`) above the upper whisker cap. The upper whisker itself would extend to the highest non-outlier value (42, since 95 is the only value beyond the fence).
- **Option C (Incorrect):** A value beyond the upper fence CANNOT become the upper whisker end — it becomes a flier marker. The upper whisker stops at the highest data value within the fence (42 in the original dataset), and 95 is separately plotted above the cap.
- **Option D (Incorrect):** "Q₃ × 2" is not a box plot outlier criterion. The standard rule is: point is an outlier if it falls below Q₁−1.5×IQR or above Q₃+1.5×IQR. The criterion uses the fence formula, not multiples of Q₃.

</details>

---

### 15. Question

A developer creates a box plot and wants to **customise the outlier markers** to appear as red diamond shapes:

```python
fig, ax = plt.subplots()
ax.boxplot(data, flierprops=dict(marker="D", color="red", markersize=8))
```

What does the `flierprops` parameter control?

- [ ] **A.** The properties of the median line (colour, thickness)
- [ ] **B.** The visual appearance of the outlier (flier) data points — their marker shape, colour, and size
- [ ] **C.** The fill colour and transparency of the box rectangle
- [ ] **D.** The properties of the whisker lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The median line's visual properties are controlled by `medianprops=dict(color=..., linewidth=...)`. `flierprops` specifically targets outlier point markers.
- **Option B (Correct):** `flierprops` is a dictionary that controls the appearance of **outlier (flier) markers** — the individual data points plotted beyond the whisker ends. Accepted keys include `marker` (shape: `"o"`, `"D"`, `"+"`, `"x"`, `"^"`, etc.), `color` or `markeredgecolor` / `markerfacecolor` (colouring), and `markersize` (point size). Setting `marker="D"` gives diamond shapes, `color="red"` makes them red, and `markersize=8` makes them 8 points large.
- **Option C (Incorrect):** Box fill colour and transparency require `patch_artist=True` in the `boxplot()` call and then accessing `bp["boxes"][i].set_facecolor(...)`. `flierprops` has nothing to do with box fill.
- **Option D (Incorrect):** Whisker line properties are controlled by `whiskerprops=dict(color=..., linestyle=..., linewidth=...)`. `flierprops` only affects the outlier markers.

</details>

---

### 16. Question

A developer tries the following code to colour the box fill but it does NOT work as expected:

```python
fig, ax = plt.subplots()
bp = ax.boxplot([10, 15, 20, 25, 30, 35, 40],
                boxprops=dict(facecolor="lightblue"))
plt.show()
```

Why does the box NOT appear light blue?

- [ ] **A.** `boxprops` is not a valid parameter — use `box_color="lightblue"` instead
- [ ] **B.** `facecolor` in `boxprops` requires a hex colour code, not a named colour
- [ ] **C.** By default, boxes are drawn as Line2D outlines without fill — `patch_artist=True` must be set to enable the box as a fillable patch that accepts `facecolor`
- [ ] **D.** The `facecolor` key only works in `flierprops`, not in `boxprops`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `boxprops` IS a valid parameter of `ax.boxplot()`. It accepts a dictionary of Line2D properties for the box outline. The issue is not the parameter name.
- **Option B (Incorrect):** Matplotlib accepts both named colours (like `"lightblue"`, `"red"`) and hex codes (`"#ADD8E6"`) everywhere colour is accepted. Named colours work perfectly in `facecolor`.
- **Option C (Correct):** By default, Matplotlib draws the box as a **Line2D** object (just an outline, like drawing a rectangle with lines). Line2D objects only have stroke colour (`color`) — they have no fill. To make the box a **fillable patch** (`PathPatch`), you must pass `patch_artist=True`. With `patch_artist=True`, `bp["boxes"][i]` becomes a `PathPatch` that accepts `set_facecolor()` and `set_alpha()`. Without `patch_artist=True`, `facecolor` in `boxprops` is silently ignored or has no visible effect.
- **Option D (Incorrect):** `facecolor` in `flierprops` controls the fill of the outlier marker shape (e.g., the interior of a diamond marker). It's unrelated to `boxprops`. The issue is specifically about `patch_artist=True` being required.

</details>

---

### 17. Question

A developer correctly enables `patch_artist=True` and colours boxes:

```python
fig, ax = plt.subplots()
bp = ax.boxplot([data_A, data_B, data_C],
                tick_labels=["A", "B", "C"],
                patch_artist=True)

colours = ["#FF9999", "#99CCFF", "#99FF99"]
for patch, colour in zip(bp["boxes"], colours):
    patch.set_facecolor(colour)
```

What Python type are the elements of `bp["boxes"]` when `patch_artist=True`?

- [ ] **A.** `Line2D` objects
- [ ] **B.** `PathPatch` objects (fillable patch artists)
- [ ] **C.** `Rectangle` objects (the same type as bar chart bars)
- [ ] **D.** `Polygon` objects

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Without `patch_artist=True`, `bp["boxes"]` contains `Line2D` objects (line-drawn rectangles with no fill capability). With `patch_artist=True`, they are converted to `PathPatch` objects.
- **Option B (Correct):** When `patch_artist=True`, Matplotlib creates the box as a **`PathPatch`** (from `matplotlib.patches`). `PathPatch` is a general path-based patch that supports `set_facecolor()`, `set_edgecolor()`, `set_alpha()`, `set_hatch()`, and other patch properties. This is why `patch_artist=True` is a prerequisite for box fill colouring.
- **Option C (Incorrect):** `Rectangle` (from `matplotlib.patches`) is used for bar chart rectangles and annotation boxes. Box plot boxes with `patch_artist=True` are `PathPatch` objects, not `Rectangle` objects — though both inherit from the `Patch` base class and share many style methods.
- **Option D (Incorrect):** `Polygon` is another patch type for arbitrary polygon shapes. Box plot boxes are `PathPatch` objects, not `Polygon` objects.

</details>

---

### 18. Question

A data analyst creates a box plot of temperature readings (°C) across four seasons. The data for Summer is:

```
Summer: [28, 29, 30, 31, 31, 32, 33, 34, 35, 36, 52]
```

Assuming Q₁=30, Q₃=35, IQR=5 for Summer. Which data point is a **flier (outlier)**?

- [ ] **A.** 28 (below Q₁)
- [ ] **B.** 36 (above Q₃)
- [ ] **C.** 52 (above Q₃ + 1.5×IQR = 35 + 7.5 = 42.5)
- [ ] **D.** 33 (inside the box but not at the median)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 28 is below Q₁=30, but it is NOT an outlier. The lower fence = Q₁ − 1.5×IQR = 30 − 7.5 = 22.5. Since 28 > 22.5, the value 28 falls within the lower fence and is within the whisker range. It appears as a regular data point that the lower whisker extends to.
- **Option B (Incorrect):** 36 is above Q₃=35, but the upper fence = Q₃ + 1.5×IQR = 35 + 7.5 = 42.5. Since 36 < 42.5, the value 36 is within the upper fence and is NOT an outlier. The upper whisker extends to 36.
- **Option C (Correct):** 52 > upper fence of 42.5. Therefore, 52 is beyond the whisker boundary and qualifies as an **outlier (flier)**. It is plotted as a separate marker point above the upper whisker cap. The upper whisker itself stops at 36 (the highest non-outlier value).
- **Option D (Incorrect):** Values inside the box (between Q₁ and Q₃) are never outliers. 33 is between Q₁=30 and Q₃=35, well within the box. It does not appear as a separate marker — it's simply part of the data that determines the quartile positions.

</details>

---

### 19. Question

A developer wants to show the **mean** as a dashed line inside each box alongside the median. Which combination of parameters achieves this?

- [ ] **A.** `showmeans=True` alone
- [ ] **B.** `showmeans=True, meanline=True`
- [ ] **C.** `meanline=True` alone
- [ ] **D.** `displaymean=True, meantype="line"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `showmeans=True` alone displays the mean as a **point marker** (triangle `^` by default) inside the box, not as a line. The mean is visible but as a dot/triangle, not a horizontal line crossing the box width.
- **Option B (Correct):** `showmeans=True, meanline=True` together display the mean as a **horizontal dashed line** spanning the full width of the box — similar in appearance to the median line but typically in a different colour (dashed green by default). `showmeans=True` enables mean display; `meanline=True` specifies that it should be rendered as a line (rather than a point marker). Both parameters must be provided together for line representation.
- **Option C (Incorrect):** `meanline=True` alone has no effect — it requires `showmeans=True` to be activated. Without `showmeans=True`, `meanline=True` is silently ignored.
- **Option D (Incorrect):** `displaymean=True` and `meantype="line"` are not valid Matplotlib parameters. These would cause a `TypeError` for unexpected keyword arguments.

</details>

---

### 20. Question

A developer creates a box plot for a dataset where **all 8 values are identical** (all equal to 50):

```python
identical_data = [50, 50, 50, 50, 50, 50, 50, 50]
plt.boxplot(identical_data)
plt.show()
```

How does the box plot render?

- [ ] **A.** A box spanning from 0 to 100 with the median at 50
- [ ] **B.** A thin flat horizontal line at y=50 — the box has zero height (IQR=0), no whiskers extend, and no fliers appear
- [ ] **C.** An error is raised because IQR=0 causes division by zero
- [ ] **D.** A single point at y=50 with no box or lines

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The box boundaries are determined by Q₁ and Q₃ of the DATA, not by absolute value ranges. With all values = 50: Q₁=50, Q₂=50, Q₃=50. The box spans from 50 to 50 — zero height, not 0 to 100.
- **Option B (Correct):** When all values are identical: Q₁=Q₂=Q₃=50, IQR=0. The box collapses to a zero-height horizontal line at y=50. Whisker calculations: lower fence = 50−1.5×0=50, upper fence = 50+1.5×0=50. No data falls outside the fence (since all points ARE the fence), so no fliers appear. The whiskers also have zero length (they extend from Q₁=50 to the minimum=50, and from Q₃=50 to the maximum=50). The visual result is a single flat line at y=50.
- **Option C (Incorrect):** IQR=0 does NOT cause a division by zero in `plt.boxplot()`. Matplotlib handles zero-IQR datasets gracefully, rendering the degenerate flat-line box without errors.
- **Option D (Incorrect):** Even with all identical values, Matplotlib still draws the box (as a degenerate flat line), the median line (at the same position), whiskers (of zero length), and caps. The rendering is a visible horizontal line, not an invisible single point.

</details>
