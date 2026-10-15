# Matplotlib Bar Plot Objective Test — Set 01

---

### 1. Question

A data analyst writes the following code to plot monthly sales:

```python
import matplotlib.pyplot as plt
import numpy as np

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales  = [320, 480, 410, 570, 390]

plt.bar(months, sales)
plt.title("Monthly Sales")
plt.ylabel("Units Sold")
plt.show()
```

What is the **default bar width** used by `plt.bar()` when the `width` parameter is not specified?

- [ ] **A.** 0.5
- [ ] **B.** 1.0
- [ ] **C.** 0.8
- [ ] **D.** 0.35

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A default width of 0.5 is not the Matplotlib default. The bars would appear noticeably narrow with visible gaps between them, which is not the typical `plt.bar()` appearance.
- **Option B (Incorrect):** Width = 1.0 would make adjacent bars touch edge-to-edge with zero visible gap between them (since each bar is centered on its x-position and spans the full unit interval). This is not the default.
- **Option C (Correct):** The `width` parameter of `plt.bar()` defaults to **0.8**. This means each bar occupies 80% of the available unit interval and leaves a 20% gap (10% on each side) between adjacent bars, producing the familiar spaced bar chart appearance. The Matplotlib documentation explicitly states `width=0.8` as the default value.
- **Option D (Incorrect):** 0.35 is a common manually chosen width for **grouped bar charts** with two series (so that two 0.35-wide bars fit within a 0.8-unit space), but it is not the default width for `plt.bar()`.

</details>

---

### 2. Question

A researcher wants to create a horizontal bar chart to compare the word count of five textbook chapters. Which function call is correct?

- [ ] **A.** `plt.bar(chapters, word_counts, orientation="horizontal")`
- [ ] **B.** `plt.barh(chapters, word_counts)`
- [ ] **C.** `plt.hbar(chapters, word_counts)`
- [ ] **D.** `plt.bar(word_counts, chapters)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.bar()` does not have an `orientation` parameter. There is no way to make `plt.bar()` draw horizontal bars by passing a keyword argument — a completely separate function `plt.barh()` exists for this purpose.
- **Option B (Correct):** `plt.barh(y, width)` is the correct function for horizontal bar charts. The first argument `y` specifies the vertical positions (category labels or numeric positions), and `width` specifies how far each bar extends horizontally. `plt.barh(chapters, word_counts)` correctly maps chapter names to the y-axis and word counts to the x-axis (bar length).
- **Option C (Incorrect):** `plt.hbar()` does not exist in Matplotlib's API. This will raise an `AttributeError`. The correct function name is `plt.barh()` (bar-h, not h-bar).
- **Option D (Incorrect):** `plt.bar(word_counts, chapters)` passes word counts as x-positions and chapters as heights — both arguments are semantically reversed. This would try to use the chapter name strings as bar heights, which would cause an error.

</details>

---

### 3. Question

A developer adds a horizontal gridline to a bar chart for readability:

```python
fig, ax = plt.subplots()
ax.bar(categories, values, color="steelblue")
ax.grid(axis="y", linestyle="--", alpha=0.7)
```

Which of the following statements about `axis="y"` in `ax.grid()` is correct?

- [ ] **A.** It displays gridlines at every y-tick position, running **horizontally** across the chart
- [ ] **B.** It displays gridlines at every x-tick position, running **vertically** through each bar
- [ ] **C.** It draws a single horizontal line at y=0 (the baseline)
- [ ] **D.** It enables both x and y gridlines simultaneously

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `ax.grid(axis="y")` draws gridlines at every **y-axis tick position**, and those lines run **horizontally** across the entire plot. This is the standard technique for bar charts because the horizontal reference lines help viewers read bar heights accurately without the clutter of vertical lines (which would cross through the bars themselves).
- **Option B (Incorrect):** Gridlines at x-tick positions running vertically would be produced by `axis="x"`. For bar charts this creates vertical lines that bisect each bar, which is visually distracting and generally not recommended.
- **Option C (Incorrect):** A single baseline at y=0 is rendered by `ax.axhline(0, ...)`. The `grid(axis="y")` call draws lines at ALL y-tick positions, not just at zero.
- **Option D (Incorrect):** Drawing both axis gridlines simultaneously is produced by `ax.grid(True)` or `ax.grid(axis="both")`. The `axis="y"` argument specifically restricts gridlines to the y-axis only.

</details>

---

### 4. Question

A data analyst colors each bar individually using a list:

```python
colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
categories = ["Q1", "Q2", "Q3", "Q4"]
revenue = [120, 185, 150, 210]

plt.bar(categories, revenue, color=colors)
```

If the `colors` list contained only **3** elements instead of 4, what would happen?

- [ ] **A.** The fourth bar would have no colour (invisible/transparent)
- [ ] **B.** The fourth bar would default to the standard Matplotlib blue (`"C0"`)
- [ ] **C.** A `ValueError` would be raised because the number of colours must match the number of bars
- [ ] **D.** The colour list would cycle back to the first colour, making the fourth bar `"#E74C3C"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not silently produce invisible bars when the colour list is too short. A length mismatch is flagged as an error.
- **Option B (Incorrect):** There is no fallback to the default colour cycle when a mismatched colour list is provided. Matplotlib enforces that the list length equals the number of data points.
- **Option C (Correct):** When `color=` receives a list that is shorter than the number of bars, Matplotlib raises a `ValueError` (specifically something like "could not broadcast input array from shape (3,) into shape (4,)"). Every bar must have exactly one colour entry when a list is passed to `color=`.
- **Option D (Incorrect):** Matplotlib does not cycle through a short colour list to fill remaining bars. If cycling behaviour is desired, the developer would need to explicitly construct a longer list (e.g., using `(colors * n)[:num_bars]`). Auto-cycling from a list does not happen automatically.

</details>

---

### 5. Question

A developer writes:

```python
plt.bar(x, heights, edgecolor="black", linewidth=1.5)
```

What visual effect does `edgecolor="black"` add to the bars?

- [ ] **A.** It colours the interior fill of all bars black
- [ ] **B.** It draws a black border outline around the perimeter of each bar rectangle
- [ ] **C.** It changes the axis spine colour to black
- [ ] **D.** It adds a black horizontal line at the top edge of each bar only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `edgecolor` affects only the border stroke of the bar rectangles — not the interior fill. The fill colour is controlled by the `color` (or `facecolor`) parameter. Setting `edgecolor="black"` leaves the interior fill unchanged.
- **Option B (Correct):** `edgecolor="black"` draws a black border around the complete perimeter of each bar rectangle (all four sides: left, right, top, and bottom). Combined with `linewidth=1.5`, the border is 1.5 points wide. This technique is commonly used in publications to add visual crispness and separate adjacent bars.
- **Option C (Incorrect):** Axis spine colours are controlled via `ax.spines["bottom"].set_color(...)` or `plt.rcParams["axes.edgecolor"]`. The `edgecolor` argument of `plt.bar()` applies only to the individual bar patch artists, not the surrounding frame.
- **Option D (Incorrect):** `edgecolor` outlines the entire rectangle — not just the top edge. If only the top edge were desired, a developer would use `ax.hlines(heights, xmin, xmax, ...)` or similar, not `edgecolor`.

</details>

---

### 6. Question

A machine learning practitioner plots the validation accuracy of four models:

```python
models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
val_acc = [0.874, 0.812, 0.923, 0.891]

plt.bar(models, val_acc)
plt.ylim(0.80, 0.95)
plt.ylabel("Validation Accuracy")
plt.title("Model Comparison")
```

A reviewer flags this chart as **misleading**. What is the most likely reason?

- [ ] **A.** The bar chart should use `barh` for model comparison
- [ ] **B.** Setting `ylim(0.80, 0.95)` truncates the y-axis — the bars no longer start at 0, so the visual bar lengths exaggerate small differences in accuracy, making a 0.011 gap look enormous
- [ ] **C.** The `val_acc` values should be multiplied by 100 to display as percentages
- [ ] **D.** Model names on the x-axis will overlap; `barh` is required instead

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While `barh` can be used for model comparisons (especially with long names), using vertical `bar` is not inherently wrong. The reviewer's flag is about data ethics, not chart orientation.
- **Option B (Correct):** Setting `ylim(0.80, 0.95)` cuts off the bottom 80% of the y-axis. This means the visual bar for Logistic Regression (0.874) appears only marginally shorter than Random Forest (0.923), but the bar for Decision Tree (0.812) appears roughly half as tall as Random Forest — when in reality the difference is only 0.111 (about 13%). Non-zero y-axis baselines in bar charts dramatically distort proportional comparisons because readers instinctively judge bar length as representing absolute value from zero.
- **Option C (Incorrect):** Displaying as percentages (e.g., 87.4%) vs. proportions (0.874) is a presentational choice, not an ethical violation. The truncated axis is the primary concern regardless of units.
- **Option D (Incorrect):** While short model names like "Logistic Regression" might cause x-axis crowding, the four names given are distinct and Matplotlib would rotate or wrap them. The main issue is the non-zero axis baseline distortion.

</details>

---

### 7. Question

A developer sets `alpha=0.6` on a bar chart:

```python
plt.bar(categories, values, color="royalblue", alpha=0.6)
```

What does `alpha=0.6` control?

- [ ] **A.** The bar width as a fraction of available space (60% of each unit interval)
- [ ] **B.** The transparency of the bars — 60% opaque, 40% transparent
- [ ] **C.** The scaling of the bar heights to 60% of their actual values
- [ ] **D.** The minimum visible bar height threshold (bars below 60 units become invisible)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Bar width is controlled by the `width` parameter (default 0.8). The `alpha` parameter has no relationship to bar width.
- **Option B (Correct):** `alpha` controls **opacity/transparency** of the patch artist. `alpha=0.6` makes the bars 60% opaque (40% transparent), allowing whatever is behind the bars (gridlines, background) to show through with 40% visibility. `alpha=1.0` is fully opaque (no transparency) and `alpha=0.0` is fully transparent (invisible).
- **Option C (Incorrect):** The bar heights are determined entirely by the `height` (second positional) argument of `plt.bar()`. `alpha` does not scale or modify the data values in any way.
- **Option D (Incorrect):** There is no "minimum height threshold" behaviour associated with `alpha`. Even a bar with height=0 would render at zero height regardless of `alpha`.

</details>

---

### 8. Question

A developer writes the following code:

```python
x = np.arange(5)
heights = [42, 67, 53, 78, 61]

plt.bar(x, heights, width=0.6, align="edge")
plt.show()
```

With `align="edge"`, where is the **left edge** of the first bar positioned relative to `x[0] = 0`?

- [ ] **A.** At x = −0.3 (centered with the left edge at −0.3)
- [ ] **B.** At x = 0 (the left edge aligns to the x-coordinate exactly)
- [ ] **C.** At x = 0.3 (shifted right by half the width)
- [ ] **D.** At x = −0.6 (the bar extends fully to the left of the x-coordinate)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This describes `align="center"` behaviour, where the bar is centered on the x-coordinate (so left edge = x − width/2 = 0 − 0.3 = −0.3, right edge = x + width/2 = 0 + 0.3 = 0.3).
- **Option B (Correct):** With `align="edge"`, the **left edge** of the bar aligns exactly to the x-coordinate. So with x[0]=0, the bar's left edge starts at exactly x=0 and extends to x=0+0.6=0.6. The bar is NOT centered on x=0 but instead begins at x=0 and spans rightward.
- **Option C (Incorrect):** x=0.3 would be the center of the bar in `align="center"` mode, or the right edge of a hypothetical left-aligned half-bar. It is not the left-edge start position in `align="edge"` mode.
- **Option D (Incorrect):** x=−0.6 would place the left edge a full width to the left of the coordinate, which doesn't correspond to either alignment mode.

</details>

---

### 9. Question

A data scientist creates a bar chart with the following code and notices that the bars are completely filled black with no visible colour variation:

```python
scores = [0.88, 0.76, 0.91, 0.83]
labels = ["Model A", "Model B", "Model C", "Model D"]

plt.bar(labels, scores, color="black")
```

Which change would give each bar a **distinct colour** from Matplotlib's default colour cycle?

- [ ] **A.** Replace `color="black"` with `color="cycle"`
- [ ] **B.** Remove the `color="black"` argument entirely — Matplotlib will automatically apply the default colour cycle to each bar
- [ ] **C.** Set `color=None` to trigger automatic colouring
- [ ] **D.** Both B and C would assign the default colour cycle to individual bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `color="cycle"` is not a valid Matplotlib colour name. This would raise a `ValueError` because "cycle" is not a recognised colour specification.
- **Option B (Correct):** When no `color=` argument is provided to `plt.bar()`, Matplotlib assigns the **next colour from the current axes colour cycle** to the entire bar collection. However, note that all bars in a single `plt.bar()` call share ONE colour from the cycle — not one colour per bar. To get individual colours, you would pass a list. But removing the `color="black"` argument gives the collection a single cycle colour rather than black, which is what the question is asking (distinct from black).
- **Option C (Incorrect):** Setting `color=None` does not reliably trigger automatic per-bar colouring. It may raise a warning or produce unexpected results depending on the Matplotlib version.
- **Option D (Incorrect):** Both B and C do not both work reliably. Only removing the argument (Option B) is the well-defined, documented approach. `color=None` behaviour is not the same as omitting the argument.

</details>

---

### 10. Question

A researcher creates a publication-quality bar chart and adds axis labels and title:

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(categories, values, color="#2980B9", edgecolor="white", linewidth=0.8)
ax.set_xlabel("Product Category", fontsize=12)
ax.set_ylabel("Revenue (USD Thousands)", fontsize=12)
ax.set_title("Q3 Revenue by Category", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

What does `ax.spines["top"].set_visible(False)` accomplish?

- [ ] **A.** It hides the top tick marks on the y-axis
- [ ] **B.** It removes the top horizontal border line of the axes frame (the top spine), giving the chart a cleaner, more modern look
- [ ] **C.** It prevents any data from being plotted above the top axis limit
- [ ] **D.** It makes the topmost bar transparent

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Tick marks on the y-axis are controlled by `ax.tick_params(top=False)` or `ax.yaxis.set_tick_params(...)`. Spines and tick marks are separate components. The spine is the line forming the axes border; tick marks are small perpendicular lines along that border.
- **Option B (Correct):** Matplotlib axes have four **spines** — the four border lines forming the rectangular axes frame: "top", "bottom", "left", and "right". `ax.spines["top"].set_visible(False)` hides the top horizontal border line. Combined with `ax.spines["right"].set_visible(False)`, this creates the popular "open" axes style (only left and bottom borders visible) used in many publications and modern data visualisations to reduce visual clutter.
- **Option C (Incorrect):** Y-axis limits (the maximum displayable value) are controlled by `ax.set_ylim(...)`. Spine visibility has no effect on data clipping or axis ranges.
- **Option D (Incorrect):** Spines are axis frame elements, completely separate from the bar patches (data artists). Modifying spine visibility has no effect on any individual bars' appearance.

</details>
