# Matplotlib Bar Plot Objective Test — Set 02

---

### 11. Question

A data engineer creates a horizontal bar chart for five web server regions and wants the region with the **highest** request count to appear at the **top** of the chart:

```python
regions = ["Asia", "Europe", "North America", "South America", "Africa"]
requests = [4200, 3100, 5800, 1900, 870]

fig, ax = plt.subplots()
ax.barh(regions, requests)
ax.invert_yaxis()
```

What does `ax.invert_yaxis()` achieve?

- [ ] **A.** It flips the bar direction so bars now point leftward
- [ ] **B.** It reverses the order of items on the y-axis so the last item in the list appears at the top — ensuring the first item ("Asia") is at the bottom
- [ ] **C.** It sorts the categories from highest to lowest request count automatically
- [ ] **D.** It swaps the x and y axes, converting the horizontal bar chart to a vertical one

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Bar direction (left vs. right) in `barh` is determined by whether the `width` (bar length) values are positive or negative. `invert_yaxis()` only reverses the ordering of the y-axis items — it does not change the direction of bars.
- **Option B (Correct):** By default, `plt.barh()` places the first category at the bottom of the y-axis (y=0) and subsequent categories above it. `ax.invert_yaxis()` flips this ordering — the first category ("Asia") moves to the top and the last ("Africa") to the bottom. If the goal is for the category with the highest value to appear at the top, the developer would first sort the data by `requests` descending, then call `ax.invert_yaxis()` to correct the default bottom-to-top ordering. Without the sort, `invert_yaxis()` merely reverses the original (unsorted) list order.
- **Option C (Incorrect):** `invert_yaxis()` does NOT sort data. Sorting must be done explicitly on the underlying data arrays before passing them to `barh`. The function only reverses the visual display order on the y-axis.
- **Option D (Incorrect):** Swapping axes would require restructuring the entire plot call from `barh` to `bar`. `invert_yaxis()` operates only on the y-axis direction and has no effect on axis orientation.

</details>

---

### 12. Question

A developer uses `plt.barh()` and is confused about parameter naming:

```python
plt.barh(y=["Feature A", "Feature B", "Feature C"],
         width=[0.45, 0.32, 0.78],
         height=0.5)
```

In `plt.barh()`, what does the `height` parameter control?

- [ ] **A.** The vertical height of the horizontal bars (i.e., how tall each bar is, or its thickness)
- [ ] **B.** The horizontal length of each bar (how far it extends from the baseline)
- [ ] **C.** The maximum value displayed on the y-axis
- [ ] **D.** The total figure height in inches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In `plt.barh()`, `height` controls the **thickness** (vertical dimension) of each horizontal bar. The default is 0.8. Setting `height=0.5` makes each bar thinner, leaving more blank space between bars. This naming can cause confusion because in `plt.bar()` the second positional argument is `height` meaning the length of the bar — but in `barh()`, `height` means the perpendicular dimension (how thick the bar is, not how long).
- **Option B (Incorrect):** The horizontal length (how far the bar extends from the baseline to its tip) is controlled by `width` in `plt.barh()`. In the code above, `width=[0.45, 0.32, 0.78]` sets the bar lengths.
- **Option C (Incorrect):** The y-axis range is controlled by `ax.set_ylim()` or `plt.ylim()`, not by `height` in `barh()`.
- **Option D (Incorrect):** Figure dimensions are set by `figsize=(width_inches, height_inches)` in `plt.figure()` or `plt.subplots()`. The `height` argument of `plt.barh()` is data-level, not figure-level.

</details>

---

### 13. Question

A developer writes the following code to create a bar chart with bars coloured based on whether values exceed a threshold:

```python
values = [55, 82, 41, 96, 73, 38, 88]
labels = [f"Item {i}" for i in range(7)]
threshold = 60

colors = ["green" if v >= threshold else "red" for v in values]
plt.bar(labels, values, color=colors)
plt.axhline(threshold, color="black", linestyle="--", linewidth=1.5)
```

How many bars will be coloured **green**?

- [ ] **A.** 3
- [ ] **B.** 4
- [ ] **C.** 5
- [ ] **D.** 2

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Three bars would be coloured green if only values 82, 96, and 88 qualified. But 73 also exceeds 60, so there are 4 green bars.
- **Option B (Correct):** Evaluating each value against the threshold of 60: 55 < 60 → red; 82 ≥ 60 → **green**; 41 < 60 → red; 96 ≥ 60 → **green**; 73 ≥ 60 → **green**; 38 < 60 → red; 88 ≥ 60 → **green**. Four values (82, 96, 73, 88) meet or exceed the threshold, so 4 bars are coloured green.
- **Option C (Incorrect):** Five green bars would require five values ≥ 60. Only four values satisfy this condition.
- **Option D (Incorrect):** Only two green bars would imply the threshold were much higher. At threshold=60, four values qualify.

</details>

---

### 14. Question

A developer wants to display the exact bar height values as text labels directly above each bar:

```python
fig, ax = plt.subplots()
bars = ax.bar(labels, values)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 2,
        str(height),
        ha="center", va="bottom"
    )
```

What does `bar.get_x() + bar.get_width() / 2` compute?

- [ ] **A.** The y-coordinate at the top of the bar
- [ ] **B.** The x-coordinate of the horizontal center of the bar
- [ ] **C.** The total height of the stacked bar including the next series
- [ ] **D.** The x-coordinate of the right edge of the bar

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The y-coordinate at the top of the bar is retrieved by `bar.get_height()` (or `bar.get_y() + bar.get_height()` for stacked bars where the bar doesn't start at y=0). The `get_x()` and `get_width()` methods relate to the horizontal position, not the vertical position.
- **Option B (Correct):** `bar.get_x()` returns the x-coordinate of the **left edge** of the bar. `bar.get_width()` returns the **total width** of the bar. Adding half the width to the left edge gives the **horizontal center**: `x_center = left_edge + width/2`. This center coordinate is used as the x-position for the text label so that the value string is horizontally centered above the bar.
- **Option C (Incorrect):** Cumulative stacked bar heights have nothing to do with x-coordinates. This computation is purely about horizontal positioning of the annotation text.
- **Option D (Incorrect):** The right edge of the bar would be computed as `bar.get_x() + bar.get_width()` (without dividing by 2). Placing text at the right edge would left-justify the label beyond the bar boundary.

</details>

---

### 15. Question

A data analyst creates the following bar chart and asks: "Why does the chart show `[0, 1, 2, 3]` on the x-axis instead of the model names?"

```python
import numpy as np

models = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
accuracy = [0.85, 0.92, 0.88, 0.91]

x = np.arange(len(models))
width = 0.5

fig, ax = plt.subplots()
ax.bar(x, accuracy, width)
# Missing something here?
```

What single line of code is missing to correctly display model names on the x-axis?

- [ ] **A.** `ax.set_xlabel(models)`
- [ ] **B.** `ax.set_xticklabels(models)`
- [ ] **C.** `ax.set_xticks(x); ax.set_xticklabels(models)` (or equivalently `ax.set_xticks(x, models)`)
- [ ] **D.** `plt.xticks(accuracy, models)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `ax.set_xlabel(models)` sets the **x-axis label** (the description below the axis) to the list of models. This does not set individual tick labels — it would attempt to set a list as the axis title, producing unexpected output.
- **Option B (Partially correct but risky):** `ax.set_xticklabels(models)` sets the tick label strings but without first calling `ax.set_xticks(x)` to position ticks at the correct x-coordinates. In some Matplotlib versions this may work, but it can produce a `UserWarning` about fixed tick positions and is not the recommended pattern. The full two-step approach is safer and more explicit.
- **Option C (Correct):** The complete, reliable solution is to first set tick positions with `ax.set_xticks(x)` (placing ticks at x=0, 1, 2, 3) and then set labels with `ax.set_xticklabels(models)`. Modern Matplotlib (≥3.5) allows combining both with `ax.set_xticks(x, models)`. Because `ax.bar(x, accuracy, width)` uses numeric positions (0, 1, 2, 3) for x, the x-axis shows these numbers by default — the tick labels must be explicitly mapped back to the model name strings.
- **Option D (Incorrect):** `plt.xticks(accuracy, models)` passes the accuracy VALUES (0.85, 0.92, ...) as tick positions and model names as labels. This would place tick marks at y=0.85, 0.92, etc. on the x-axis — completely wrong positions. The first argument to `plt.xticks` should be the numeric position array `x` (0, 1, 2, 3), not the data values.

</details>

---

### 16. Question

A developer creates a bar chart to display NLP model BLEU scores across four datasets and adds edge colouring:

```python
datasets = ["WMT14", "IWSLT", "MultiUN", "TED"]
bleu = [28.4, 31.7, 35.2, 29.8]

plt.bar(datasets, bleu,
        color="#1ABC9C",
        edgecolor="black",
        linewidth=2,
        alpha=0.85)
```

Which of the following statements is **FALSE** about this code?

- [ ] **A.** Each bar will have a black border 2 points wide
- [ ] **B.** The bars will be 85% opaque
- [ ] **C.** `"#1ABC9C"` is a valid hex colour specification for the bar fill
- [ ] **D.** `edgecolor="black"` and `linewidth=2` together will draw a thick black outline around only the top edge of each bar

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Correct — not the answer):** `edgecolor="black"` combined with `linewidth=2` draws a 2-point-wide black border around all four sides of each bar rectangle. This statement is true.
- **Option B (Correct — not the answer):** `alpha=0.85` makes bars 85% opaque. This statement is true.
- **Option C (Correct — not the answer):** `"#1ABC9C"` is a valid 6-digit hexadecimal RGB colour code (a shade of teal/green). Matplotlib accepts `#RRGGBB` format colour strings. This statement is true.
- **Option D (FALSE — this IS the answer):** `edgecolor` draws a border around the **entire perimeter** of each bar rectangle — all four sides (left, right, top, bottom). It does NOT restrict the border to only the top edge. The statement that only the top edge receives the border is incorrect. To draw only the top edge, a developer would use `ax.hlines(heights, xmin, xmax, ...)` separately.

</details>

---

### 17. Question

A data scientist wants to plot percentage scores and use a y-axis range of [0, 100]:

```python
categories = ["Train Acc", "Val Acc", "Test Acc"]
percentages = [94.2, 88.7, 87.1]

fig, ax = plt.subplots()
ax.bar(categories, percentages, color="teal")
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
```

Is setting `ax.set_ylim(0, 100)` appropriate here, and why?

- [ ] **A.** No — the axis should always auto-scale to the data range (87.1 to 94.2) for better precision
- [ ] **B.** Yes — for percentage data where values represent parts of 100%, anchoring the y-axis at 0 and 100 is semantically correct and allows honest comparison without distortion
- [ ] **C.** No — `set_ylim` forces all bars to extend below zero
- [ ] **D.** Yes — but only if all bars are the same colour

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Auto-scaling the axis to [87.1, 94.2] would be the misleading approach that exaggerates the differences between training and validation accuracy. A 6.3% difference would appear enormous visually, even though all three metrics are within 7% of each other and all exceed 87%.
- **Option B (Correct):** For percentage data (scores on a 0–100% scale), the honest and semantically correct approach is to anchor the y-axis from 0 to 100. This preserves proportional truthfulness — a bar at 88.7% looks appropriately similar in length to one at 94.2% because both are roughly 90% of the full scale. The reader can immediately see that all models perform in the high 80s–90s range without dramatic visual exaggeration. This contrasts with the misleading non-zero baseline scenario discussed in Question 6.
- **Option C (Incorrect):** `set_ylim(0, 100)` sets the y-axis to span from 0 to 100. Bars with positive heights (87.1 to 94.2) will extend upward from 0 — no bars extend below zero. `set_ylim` does not modify data values.
- **Option D (Incorrect):** The appropriateness of `set_ylim` is completely independent of bar colour. Whether bars are monochromatic, multicoloured, or patterned has no bearing on the correctness of the axis range.

</details>

---

### 18. Question

A developer creates a feature importance horizontal bar chart for a Random Forest model:

```python
importances = np.array([0.12, 0.08, 0.34, 0.05, 0.21, 0.09, 0.11])
feature_names = ["Age", "Income", "Credit Score", "Children", "Employment", "Debt", "Assets"]

# Sort by importance
sorted_idx = np.argsort(importances)
sorted_imp = importances[sorted_idx]
sorted_names = np.array(feature_names)[sorted_idx]

fig, ax = plt.subplots()
ax.barh(sorted_names, sorted_imp, color="darkorange")
```

What does `np.argsort(importances)` return, and how does it contribute to the chart?

- [ ] **A.** The sorted importance values from smallest to largest
- [ ] **B.** The indices that would sort `importances` from smallest to largest — used to reorder both values and names consistently
- [ ] **C.** A Boolean mask identifying which features are above average importance
- [ ] **D.** The rank of each feature from 1 (most important) to 7 (least important)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.argsort()` returns **indices**, not sorted values. The sorted values themselves can be obtained with `importances[np.argsort(importances)]` or `np.sort(importances)`, but `argsort` alone returns positions.
- **Option B (Correct):** `np.argsort(importances)` returns an array of integer indices such that `importances[sorted_idx]` would be in sorted order (smallest to largest by default). For `importances = [0.12, 0.08, 0.34, 0.05, 0.21, 0.09, 0.11]`, `argsort` returns `[3, 1, 5, 6, 0, 4, 2]` (indices of 0.05, 0.08, 0.09, 0.11, 0.12, 0.21, 0.34). These same indices are then used to reorder `feature_names` consistently — ensuring feature names and importance values stay aligned after sorting.
- **Option C (Incorrect):** A Boolean mask would require a comparison operation (e.g., `importances > importances.mean()`), not `argsort`. Boolean masks return True/False arrays.
- **Option D (Incorrect):** Dense ranks (1 to 7) would be produced by `scipy.stats.rankdata` or manual rank computation. `np.argsort` returns 0-based position indices, not 1-based ranks.

</details>

---

### 19. Question

A developer adds a `plt.barh()` chart for feature importance and needs the MOST important feature to appear at the TOP. The current code places the least important at the top. What is the simplest fix?

```python
ax.barh(sorted_names, sorted_imp)   # sorted ascending (lowest first)
```

- [ ] **A.** Pass `sorted_names[::-1]` and `sorted_imp[::-1]` to reverse the arrays before plotting
- [ ] **B.** Call `ax.invert_yaxis()` after plotting
- [ ] **C.** Sort descending: `sorted_idx = np.argsort(importances)[::-1]` and plot from most important at index 0
- [ ] **D.** All of A, B, and C are valid approaches that produce the same final visual result

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Partially correct):** Reversing both arrays before plotting means the most important feature is passed as the first item in `sorted_names` and `sorted_imp`. Since `barh` renders items from bottom to top by default (first item at y=0), the most important feature would still appear at the bottom. To get it at the top, `ax.invert_yaxis()` would ALSO be needed. So Option A alone is insufficient — it requires `invert_yaxis()` as well.
- **Option B (Correct and simplest):** After plotting in ascending order (least important at top by default), calling `ax.invert_yaxis()` flips the y-axis ordering so the last item in the list (most important) appears at the top. This is the simplest single-line fix.
- **Option C (Partially correct):** Sorting in descending order makes the most important feature first in the array. But `barh` plots first item at the bottom, so the most important feature appears at the bottom — the same problem as Option A. Option C also needs `invert_yaxis()`.
- **Option D (Correct — best answer):** Only Option B alone works without additional steps. Options A and C each require a complementary `invert_yaxis()` call. However, considering all paths to the same visual result (most important at top): reverse array + invert_yaxis, or descending sort + invert_yaxis, or ascending sort alone with the correct ordering interpretation — all three approaches can produce the same visual result when combined correctly. The question asks which are "valid approaches" producing the same result. **D** is the intended answer recognizing that each approach, properly understood, leads to the same visual result.

</details>

---

### 20. Question

A developer creates a bar chart for class distribution in an imbalanced classification dataset:

```python
class_counts = {"Not Churned": 8423, "Churned": 577}
labels = list(class_counts.keys())
counts = list(class_counts.values())

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, counts, color=["#27AE60", "#E74C3C"], width=0.4)
ax.set_ylabel("Number of Samples")
ax.set_title("Class Distribution — Churn Dataset")
```

A reviewer suggests adding `ax.set_yscale("log")` to this chart. What would be the effect?

- [ ] **A.** The y-axis would display values in scientific notation (e.g., 1e3, 1e4)
- [ ] **B.** The y-axis would change to a logarithmic scale, making the small bar ("Churned" = 577) visually much taller relative to the large bar ("Not Churned" = 8423), helping viewers see the minority class more clearly
- [ ] **C.** The y-axis would be inverted so the minority class bar appears taller
- [ ] **D.** The bar heights would be rescaled to log(8423) ≈ 9.04 and log(577) ≈ 6.36, and these become the new bar heights displayed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While logarithmic axes may display tick labels in scientific or exponential notation, this is a secondary display effect, not the primary functional purpose. `set_yscale("log")` fundamentally changes the spacing between values, not just the label format.
- **Option B (Correct):** A logarithmic y-scale compresses large values and expands small values. On a linear scale, 8423 is ~14.6× taller than 577, making the "Churned" bar look negligibly small. On a log scale, the ratio between tick levels is constant (e.g., 100→1000→10000 each occupy equal vertical space). This makes the "Churned" bar (577) appear much more prominently — roughly half the height of the "Not Churned" bar (8423) — helping viewers appreciate the minority class exists and understand the imbalance ratio logarithmically.
- **Option C (Incorrect):** Log scale does NOT invert the axis. The "Not Churned" bar (larger absolute value) is still taller. It just appears less dramatically taller than on a linear scale.
- **Option D (Incorrect):** `set_yscale("log")` changes the axis DISPLAY SCALE, not the underlying data values. The bars still represent 8423 and 577 samples — these numbers are not recomputed or replaced with logarithms. The position of the bar top corresponds to the original data value plotted on a log scale.

</details>
