# Matplotlib Bar Plot Objective Test — Set 04

---

### 31. Question

A researcher creates a stacked bar chart for a survey with three response categories:

```python
categories = ["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"]
positive = [40, 55, 30, 20, 15]
neutral  = [25, 20, 35, 30, 25]
negative = [35, 25, 35, 50, 60]

fig, ax = plt.subplots()
ax.bar(categories, positive, label="Positive")
ax.bar(categories, neutral,  label="Neutral")
ax.bar(categories, negative, label="Negative")
ax.legend()
```

What visual problem does this code produce?

- [ ] **A.** The bars will overlap each other from the bottom — all three series start at y=0, so the "Negative" bars paint over the "Neutral" bars, which paint over the "Positive" bars
- [ ] **B.** A `ValueError` is raised because stacked bars require numeric x-positions, not strings
- [ ] **C.** The legend displays entries in reverse order
- [ ] **D.** Only the first series (positive) is visible because subsequent calls clear the previous bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The critical error is the missing `bottom=` argument. When `ax.bar()` is called three times without `bottom=`, each series draws its bars starting from y=0. The "Neutral" bars completely cover the "Positive" bars (same x-positions, starting at y=0), and the "Negative" bars then cover both. The visual result is that only the "Negative" series (last drawn) is visible for each category. To create a stacked bar chart, `neutral` must use `bottom=positive`, and `negative` must use `bottom=np.array(positive) + np.array(neutral)`.
- **Option B (Incorrect):** `plt.bar()` fully supports categorical string x-positions. No `ValueError` is raised due to string categories — that error would occur only if you tried to perform arithmetic on the string array (as in grouped bars).
- **Option C (Incorrect):** The legend order follows the order of `ax.bar()` calls (Positive → Neutral → Negative). This is typically the correct stacking order. The legend itself is not reversed, though the visual bars are incorrect due to missing `bottom=`.
- **Option D (Incorrect):** Subsequent `ax.bar()` calls do NOT clear previous bars. Each call ADDS new `Rectangle` artists to the axes. The problem is that the new bars are drawn starting from y=0 and painted over the previous ones (with the last call on top), not that they clear them.

</details>

---

### 32. Question

A developer correctly implements a 3-layer stacked bar chart:

```python
classes   = ["Cat", "Dog", "Bird", "Fish"]
group1    = [120, 95, 60, 80]
group2    = [80, 70, 40, 50]
group3    = [50, 45, 30, 35]

bottom2 = np.array(group1)
bottom3 = np.array(group1) + np.array(group2)

ax.bar(classes, group1, label="Group 1")
ax.bar(classes, group2, bottom=bottom2, label="Group 2")
ax.bar(classes, group3, bottom=bottom3, label="Group 3")
```

For the "Dog" category, what is the total height of the stacked bar?

- [ ] **A.** 45 (only Group 3)
- [ ] **B.** 165 (Group 1 + Group 2)
- [ ] **C.** 210 (Group 1 + Group 2 + Group 3)
- [ ] **D.** 95 (only Group 1)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 45 is the height contribution of Group 3 alone for "Dog". The total bar height is the sum of all three segments.
- **Option B (Incorrect):** 165 = Group 1 + Group 2 = 95 + 70. This is `bottom3` for "Dog" — the starting baseline of Group 3, not the final total height.
- **Option C (Correct):** Total stacked bar height = Group 1 + Group 2 + Group 3 = 95 + 70 + 45 = **210**. Group 1 forms the base (0 to 95), Group 2 stacks on top (95 to 165), and Group 3 sits at the top (165 to 210). The complete bar for "Dog" reaches y=210.
- **Option D (Incorrect):** 95 is only Group 1's value — the height of the bottom segment for "Dog". The full stacked bar includes all three segments.

</details>

---

### 33. Question

A developer creates a stacked bar chart and needs to compute the `bottom` array for the third segment. The first and second segments have these values:

```python
seg1 = np.array([30, 50, 40, 20])
seg2 = np.array([20, 30, 25, 35])
seg3 = np.array([10, 20, 15, 25])
```

Which code correctly computes `bottom_for_seg3`?

- [ ] **A.** `bottom_for_seg3 = seg1 + seg2`
- [ ] **B.** `bottom_for_seg3 = seg3`
- [ ] **C.** `bottom_for_seg3 = np.cumsum([seg1, seg2, seg3])`
- [ ] **D.** `bottom_for_seg3 = seg1 * seg2`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The `bottom` argument for segment 3 must be the cumulative sum of all segments BELOW it — segments 1 and 2. `bottom_for_seg3 = seg1 + seg2` computes element-wise addition: `[30+20, 50+30, 40+25, 20+35] = [50, 80, 65, 55]`. These are the y-positions where segment 3 begins for each category. NumPy arrays support element-wise addition with `+` directly.
- **Option B (Incorrect):** `bottom_for_seg3 = seg3` would use segment 3's OWN values as the starting baseline, which is meaningless. The baseline must reflect where the PREVIOUS segments end.
- **Option C (Incorrect):** `np.cumsum([seg1, seg2, seg3])` computes a cumulative sum across the 2D array of stacked segments, producing a 2D result or flattened array — not a 1D array suitable as the `bottom` for seg3. This is not the correct approach.
- **Option D (Incorrect):** `seg1 * seg2` computes element-wise multiplication: `[600, 1500, 1000, 700]`. These values have no relationship to the required baseline heights and would be enormously wrong as `bottom` values.

</details>

---

### 34. Question

A data analyst creates a 100% normalized stacked bar chart for market share across three regions:

```python
regions = ["North", "South", "East", "West"]
brand_A = np.array([300, 150, 220, 180])
brand_B = np.array([200, 250, 130, 260])
brand_C = np.array([100, 100, 150, 60])

total = brand_A + brand_B + brand_C

pct_A = (brand_A / total) * 100
pct_B = (brand_B / total) * 100
pct_C = (brand_C / total) * 100
```

For the "North" region, what are the percentage shares of Brand A, B, and C? (Hint: North has brand_A=300, brand_B=200, brand_C=100)

- [ ] **A.** A=50%, B=33.3%, C=16.7%
- [ ] **B.** A=60%, B=40%, C=20%
- [ ] **C.** A=33.3%, B=33.3%, C=33.3%
- [ ] **D.** A=45%, B=30%, C=25%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** North total = 300 + 200 + 100 = **600**. Brand A: (300/600)×100 = **50%**. Brand B: (200/600)×100 = **33.3%** (repeating). Brand C: (100/600)×100 = **16.7%** (repeating). These sum to 100% exactly (50 + 33.3 + 16.7 = 100). This is the correct percentage distribution for the North region.
- **Option B (Incorrect):** 60%, 40%, 20% would sum to 120% — mathematically impossible for shares of a whole. These numbers assume a total of 500 (300/500=60%, 200/500=40%, 100/500=20%), which is incorrect. The actual total is 600.
- **Option C (Incorrect):** Equal thirds (33.3% each) would only occur if brand_A = brand_B = brand_C for North. Since brand_A=300 > brand_B=200 > brand_C=100, they cannot share equally.
- **Option D (Incorrect):** 45% + 30% + 25% = 100%, but the values are wrong. Brand A percentage = 50%, not 45%. Brand B = 33.3%, not 30%. Brand C = 16.7%, not 25%.

</details>

---

### 35. Question

A developer creates a stacked horizontal bar chart for time allocation per project phase. Which parameter in `ax.barh()` is equivalent to `bottom` in `ax.bar()`?

```python
phases = ["Design", "Development", "Testing", "Deployment"]
week1 = [2, 5, 1, 1]
week2 = [1, 4, 2, 1]
week3 = [1, 3, 3, 2]

ax.barh(phases, week1, label="Week 1")
ax.barh(phases, week2, ___=week1, label="Week 2")   # What goes here?
```

- [ ] **A.** `bottom=week1`
- [ ] **B.** `left=week1`
- [ ] **C.** `start=week1`
- [ ] **D.** `offset=week1`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `bottom=` is the parameter for stacking in **vertical** bar charts (`ax.bar()`). In `ax.barh()`, using `bottom=` would be applying a vertical offset — but horizontal bars stack horizontally, not vertically. The correct horizontal-stacking parameter is different.
- **Option B (Correct):** In `ax.barh()`, the `left=` parameter specifies where the bar begins on the **x-axis** (the horizontal baseline). Just as `bottom=` in `ax.bar()` shifts the vertical starting position upward for stacking, `left=` in `ax.barh()` shifts the horizontal starting position rightward. `ax.barh(phases, week2, left=week1)` draws Week 2 bars starting where Week 1 bars end — creating horizontal stacking.
- **Option C (Incorrect):** `start=` is not a valid parameter for `ax.barh()`. This would raise a `TypeError` for an unexpected keyword argument.
- **Option D (Incorrect):** `offset=` is not a valid parameter for `ax.barh()` either. There is no such argument in the Matplotlib bar/barh API.

</details>

---

### 36. Question

A developer adds error bars to a grouped bar chart for model accuracy with standard deviations:

```python
models = ["Model A", "Model B", "Model C"]
mean_acc = [0.884, 0.921, 0.897]
std_acc  = [0.012, 0.008, 0.015]

x = np.arange(3)
fig, ax = plt.subplots()
bars = ax.bar(x, mean_acc, yerr=std_acc, capsize=6,
              error_kw={"ecolor": "black", "elinewidth": 1.5})
ax.set_xticks(x, models)
```

For "Model C" with `mean_acc=0.897` and `std_acc=0.015`, what are the **upper and lower error bar tips** (endpoints)?

- [ ] **A.** Upper = 0.912, Lower = 0.882
- [ ] **B.** Upper = 0.015, Lower = −0.015
- [ ] **C.** Upper = 0.897, Lower = 0.882
- [ ] **D.** Upper = 0.906, Lower = 0.888

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Error bars with `yerr=std_acc` are drawn symmetrically around the bar top. The bar height for Model C is `mean_acc = 0.897`. The error bar extends `±std_acc = ±0.015` from the bar top. Upper tip = `0.897 + 0.015 = 0.912`. Lower tip = `0.897 − 0.015 = 0.882`. The `capsize=6` adds small horizontal caps at these endpoints (6 points wide each side).
- **Option B (Incorrect):** Upper=0.015 and Lower=−0.015 are the std_acc values themselves, not the coordinates of the error bar tips. The tips are positioned relative to the bar height (mean_acc), not relative to zero.
- **Option C (Incorrect):** Upper=0.897 would place the upper cap at the bar top itself (zero extension), and Lower=0.882 (only going down) would represent asymmetric error bars. Symmetric `yerr` extends both above and below the bar top.
- **Option D (Incorrect):** Upper=0.906 would correspond to std_acc=0.009, and Lower=0.888 would be mean − 0.009. These don't match std_acc=0.015.

</details>

---

### 37. Question

A developer wants to add **asymmetric error bars** to a bar chart where upper and lower errors are different:

```python
heights = [75, 88, 62, 91]
labels  = ["Q1", "Q2", "Q3", "Q4"]

lower_err = [5, 8, 4, 7]
upper_err = [10, 6, 9, 5]

yerr = [lower_err, upper_err]   # 2×N array for asymmetric errors

ax.bar(labels, heights, yerr=yerr, capsize=5)
```

For "Q1" with `height=75`, `lower_err=5`, `upper_err=10`, what are the lower and upper error bar endpoints?

- [ ] **A.** Lower = 65, Upper = 85
- [ ] **B.** Lower = 70, Upper = 85
- [ ] **C.** Lower = 70, Upper = 80
- [ ] **D.** Lower = 80, Upper = 85

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Lower=65 would require lower_err=10 and Upper=85 would require upper_err=10. But lower_err=5 and upper_err=10 for Q1. Lower endpoint = 75−5=70, upper endpoint = 75+10=85.
- **Option B (Correct):** When `yerr` is a 2×N array `[lower_err, upper_err]`, Matplotlib interprets: `lower_err[i]` = amount to subtract from bar height (downward error), `upper_err[i]` = amount to add to bar height (upward error). For Q1: Lower endpoint = `75 − 5 = 70`. Upper endpoint = `75 + 10 = 85`. The lower cap is at y=70 and the upper cap is at y=85.
- **Option C (Incorrect):** Lower=70 is correct, but Upper=80 would correspond to upper_err=5. For Q1, upper_err=10, so the upper endpoint is 75+10=85.
- **Option D (Incorrect):** Lower=80 would correspond to the height being higher (e.g., height=85 with lower_err=5), or lower_err=−5 which is not the case. For Q1 with height=75 and lower_err=5, the lower endpoint is 70.

</details>

---

### 38. Question

A developer uses `capsize=8` in a bar chart with error bars. What does `capsize` control?

- [ ] **A.** The height of the error bar line (how far it extends from the bar top)
- [ ] **B.** The width (in points) of the horizontal cap lines at the tips of the error bars
- [ ] **C.** The number of standard deviations represented by the error bar
- [ ] **D.** The font size of the error value labels displayed next to the caps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The extent (length) of the error bar line is controlled by `yerr` (for vertical bars) or `xerr` (for horizontal bars) — the actual numerical error values determine how far the bar extends. `capsize` controls the appearance of the caps, not their position.
- **Option B (Correct):** `capsize` specifies the **half-width** of the horizontal cap lines in **points** (unit of typography). The cap is a short horizontal line drawn at the tip of each error bar. `capsize=8` draws caps that are 8 points wide on each side. This "T-bar" shape helps viewers identify the exact endpoint of the error interval. Without `capsize` (or `capsize=0`), only a vertical line is drawn with no cap.
- **Option C (Incorrect):** The number of standard deviations is encoded in the `yerr` value itself. If you want ±2σ, you pass `yerr = 2 * std`. The `capsize` parameter is purely a display aesthetic affecting cap width, not statistical interpretation.
- **Option D (Incorrect):** Error bar values are not automatically labeled by Matplotlib's error bar rendering. `capsize` does not produce or size text labels. Labeling error values would require explicit `ax.text()` calls.

</details>

---

### 39. Question

A developer uses `ecolor` in the `error_kw` dictionary:

```python
ax.bar(x, heights, yerr=errors,
       error_kw={"ecolor": "crimson", "elinewidth": 2, "capsize": 7})
```

What does `ecolor="crimson"` specifically colour?

- [ ] **A.** The fill colour of the bars
- [ ] **B.** The error bar lines and cap lines
- [ ] **C.** The x-axis tick labels
- [ ] **D.** The text annotations above each bar

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The fill colour of the bars is controlled by the `color` parameter of `ax.bar()`, not `ecolor`. `ecolor` relates specifically to error bar styling.
- **Option B (Correct):** `ecolor` stands for "error colour." It controls the colour of: (1) the vertical error bar lines extending above and below the bar tops, and (2) the horizontal cap lines at the tips of the error bars. Setting `ecolor="crimson"` makes both the vertical lines and horizontal caps crimson red, while the bars themselves retain their own `color` setting. This allows the error bars to stand out visually from the bars.
- **Option C (Incorrect):** Tick label colours are controlled by `ax.tick_params(labelcolor=...)`. `ecolor` in `error_kw` applies only to error bar artists, not axis labels.
- **Option D (Incorrect):** Text annotations are created and styled separately (via `ax.text()` or `ax.bar_label()`). `ecolor` only affects the error bar line artists created by the `yerr`/`xerr` rendering.

</details>

---

### 40. Question

A data engineer creates a 100% stacked bar chart but realises the percentages for the "West" region do not sum to 100%. The raw counts are:

```
West: Brand A = 180, Brand B = 260, Brand C = 60
```

A developer computes:

```python
total_west = 180 + 260 + 60   # = 500
pct_A = (180 / 500) * 100     # = 36%
pct_B = (260 / 500) * 100     # = 52%
pct_C = (60  / 500) * 100     # = 12%
```

The stacked chart uses `bottom` values. What should `bottom_C` (the starting height for Brand C's bar in the West column) be set to?

- [ ] **A.** 36 (just Brand A's percentage)
- [ ] **B.** 52 (just Brand B's percentage)
- [ ] **C.** 88 (Brand A% + Brand B% = 36 + 52)
- [ ] **D.** 100 (the total, placing Brand C at the top)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Using only Brand A's percentage (36%) as the bottom for Brand C would mean Brand C starts just above Brand A, leaving Brand B without any stacking offset. Brand B would still render from 0, creating overlap.
- **Option B (Incorrect):** Using only Brand B's percentage (52%) would ignore Brand A's contribution to the stack. Brand C must sit above BOTH Brand A and Brand B.
- **Option C (Correct):** The `bottom` for Brand C's bar must account for ALL segments below it: Brand A (36%) + Brand B (52%) = **88%**. Brand C's bar starts at y=88 and extends to y=88+12=100. This correctly places Brand C as the topmost segment with the bar reaching exactly 100%.
- **Option D (Incorrect):** Using 100 as the bottom would place Brand C starting at the top of the full 100% bar with no room to extend upward — its segment would extend beyond 100%, making the chart incorrect.

</details>
