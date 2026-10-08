# Matplotlib Line Plot Objective Test — Set 09

---

### 81. Question

In the semiotics and visual grammar of data visualization, what fundamental meaning does a line connecting two plotted points $(x_1, y_1)$ and $(x_2, y_2)$ inherently communicate to the human brain?

- [ ] **A.** That the two points belong to identical categories and have identical values
- [ ] **B.** That the data was collected using an automated robotic sensor rather than manual entry
- [ ] **C.** That there exists a continuous domain, a meaningful sequence, a physical rate of change (slope $\frac{\Delta y}{\Delta x}$), and valid intermediate values between the two observations
- [ ] **D.** That the vertical axis must be scaled logarithmically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A line connects points with potentially different values; it does not indicate identical values.
- **Option B (Incorrect):** Visual grammar is independent of data collection methodology.
- **Option C (Correct):** In human visual perception, a continuous stroke connecting two points implies continuity. It conveys that the horizontal axis represents an ordered, continuous domain (such as time, distance, temperature, or a mathematical parameter) where intermediate states exist and where the slope of the connecting segment represents a meaningful rate of change.
- **Option D (Incorrect):** Line connectivity does not dictate logarithmic scaling.

</details>

---

### 82. Question

A junior analyst creates the following plot to display annual department expenditures:

```python
departments = ["Human Resources", "Marketing", "Engineering", "Finance", "Legal"]
spending    = [1.2, 3.8, 8.5, 2.1, 1.7]

plt.plot(departments, spending, marker='o', linestyle='-')
plt.ylabel("Expenditure ($M)")
plt.show()
```

Why is using a line chart for this dataset fundamentally considered bad data visualization practice?

- [ ] **A.** Because Matplotlib will crash if strings are passed to the horizontal axis.
- [ ] **B.** Because departments are discrete, unordered nominal categories; drawing connecting lines falsely implies a continuous transition, an intermediate state between "Marketing" and "Engineering", and a physical rate of change across arbitrary category boundaries.
- [ ] **C.** Because department names should always be sorted by vowel count before plotting.
- [ ] **D.** Because line plots cannot render numbers formatted in millions.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib supports categorical x-axes with string values without crashing.
- **Option B (Correct):** Nominal categories (departments, product types, geographical regions) have no intrinsic mathematical continuum or order. Connecting "Marketing" to "Engineering" with a line segment visually implies that one can traverse halfway between Marketing and Engineering and experience an expenditure of \$6.15M. Furthermore, the slope between them is an arbitrary artifact of whatever order the list was written in. A **bar chart** (`plt.bar()`) is the ethically and technically correct choice.
- **Option C (Incorrect):** Categorical sorting should be based on value magnitude or logical hierarchy, not vowel count.
- **Option D (Incorrect):** Line plots handle values in any numerical range.

</details>

---

### 83. Question

What is the **"Permutation Test"** commonly used by visualization specialists to determine whether a line chart is ethically and mathematically valid for a given dataset?

- [ ] **A.** If randomly shuffling the order of items along the horizontal axis fundamentally changes the shape of the curve, its peaks, its troughs, and its slopes, the data is discrete and a line chart **must not** be used.
- [ ] **B.** If calculating the permutations of the dataset takes more than 1 second, a line chart is required.
- [ ] **C.** If permuting the colors of the lines alters the legend, a pie chart must be substituted.
- [ ] **D.** If the number of points is prime, a line chart is forbidden.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For true continuous or ordered series (e.g., chronological time or distance along a transect), the sequence along the x-axis is physically fixed; you cannot put Tuesday before Monday. However, for discrete nominal categories (e.g., regions: North, South, East, West), any permutation is equally valid. If reordering the categories creates an entirely new set of "trends," "peaks," and "drops," then those visual trends are illusory artifacts of list ordering. Therefore, failing the permutation test proves that a line chart is inappropriate and a bar chart must be used.
- **Option B (Incorrect):** The test is a conceptual test of data semantics, not an algorithm performance benchmark.
- **Option C (Incorrect):** Legend color permutations are unrelated to the validity of line charts.
- **Option D (Incorrect):** Mathematical primality of sample size has no bearing on visualization chart selection.

</details>

---

### 84. Question

How does the rule regarding **zero-baselines** on the vertical axis differ between **bar charts** and **line plots**?

- [ ] **A.** Both bar charts and line charts must strictly start at zero under all circumstances without exception.
- [ ] **B.** Line charts must always start at zero, but bar charts can start at any arbitrary number.
- [ ] **C.** Neither chart type should ever include zero on the vertical axis.
- [ ] **D.** Bar charts encode numerical values through the physical height and area of the bar, making a zero baseline mandatory to avoid distorting proportions; line charts encode values via position and slopes, allowing non-zero baselines to focus on subtle variations (e.g., human body temperature), provided the truncated axis is clearly labeled.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Forcing a line chart of human body temperature (spanning $97.5^\circ\text{F}$ to $103.0^\circ\text{F}$) to start at $0^\circ\text{F}$ flattens the curve into an unreadable line, obscuring critical clinical fevers.
- **Option B (Incorrect):** This completely reverses the fundamental rules of graphical integrity.
- **Option C (Incorrect):** Zero baselines are foundational in many visualization paradigms.
- **Option D (Correct):** In a bar chart, the reader's eye compares the relative area/length of bars. Truncating the baseline from 0 to 90 makes a value of 100 appear ten times larger than a value of 91, severely deceiving the viewer. In contrast, line charts communicate position and slope ($\Delta y / \Delta x$). While intentional axis manipulation to exaggerate trivial noise is unethical, zooming into a non-zero range on a line chart is acceptable when analyzing small variations around a high baseline (e.g., stock market indices or vital signs), as long as the axis is explicitly labeled.

</details>

---

### 85. Question

A data analyst plots quarterly customer satisfaction scores using string labels for dates with irregular time gaps:

```python
dates = ["2023-01-01", "2023-01-08", "2023-01-15", "2023-11-30"]
scores = [82, 84, 85, 89]

plt.plot(dates, scores, 'o-')
plt.show()
```

What visual distortion does this code produce?

- [ ] **A.** The 10-month gap between January 15 and November 30 is rendered with the exact same visual spacing as the 7-day gap between January 1 and January 8, visually falsifying the rate of change.
- [ ] **B.** Matplotlib automatically computes the date deltas and inserts appropriate spacing for strings.
- [ ] **C.** The line turns red automatically to indicate missing months.
- [ ] **D.** The chart fails to render because dates must always be integers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When strings are passed to `plt.plot()`, Matplotlib treats them as categorical units spaced at uniform unit distances ($x = 0, 1, 2, 3$). Consequently, a 319-day gap between Jan 15 and Nov 30 appears identical in horizontal width to a 7-day gap between Jan 1 and Jan 8. This creates an optical illusion that score growth accelerated dramatically over a uniform timeline, when in reality it occurred over nearly an entire year. Converting strings to true `datetime` objects before plotting ensures accurate, proportional time spacing.
- **Option B (Incorrect):** Matplotlib does not infer temporal spacing from raw string objects; it treats strings as plain discrete categories.
- **Option C (Incorrect):** Matplotlib does not alter line colors based on date gaps.
- **Option D (Incorrect):** Categorical string plotting is supported, but it distorts irregular time intervals.

</details>

---

### 86. Question

A data analyst creates a performance report displaying the hourly power generation of 35 distinct wind turbines on a single axes. The resulting visualization consists of 35 tangled, overlapping colored lines that resemble a ball of yarn (commonly referred to as a **"spaghetti chart"**).

Which visualization design strategy effectively resolves this visual clutter?

- [ ] **A.** Increase the line thickness of all 35 lines to `linewidth=5`
- [ ] **B.** Remove the legend and change the background to black
- [ ] **C.** Use a small-multiples grid layout (`plt.subplots`) or highlight one focal turbine in bold color while rendering the other 34 in muted, semi-transparent gray
- [ ] **D.** Replace all lines with 35 overlapping pie charts

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Increasing stroke width causes thick overlapping swatches of color, making individual traces even harder to discern.
- **Option B (Incorrect):** Removing the legend leaves the viewer with zero ability to identify any turbine.
- **Option C (Correct):** The "spaghetti chart" suffers from cognitive overload. Two proven design patterns solve this:
  1. **Small Multiples:** Creating a grid of miniature line plots (e.g., via `plt.subplots(5, 7)`), giving each turbine its own clean space.
  2. **Highlight & Context:** Plotting 34 reference lines in semi-transparent light gray (`color='lightgray'`, `alpha=0.4`) and rendering the single turbine of interest in a prominent bold color (e.g., `color='navy'`, `linewidth=2.5`).
- **Option D (Incorrect):** Pie charts cannot communicate hourly temporal trends.

</details>

---

### 87. Question

What is the principle of **"Banking to 45 Degrees"** proposed by statistician William S. Cleveland regarding line chart aspect ratios?

- [ ] **A.** Line charts must always be tilted at a 45-degree angle inside the presentation slide.
- [ ] **B.** Axis tick labels must always be rotated to $45^\circ$ using `plt.xticks(rotation=45)`.
- [ ] **C.** The aspect ratio (ratio of canvas width to height) should be chosen such that the average absolute slope of critical line segments hovers around $45^\circ$, maximizing human perceptual accuracy in detecting relative changes in slope.
- [ ] **D.** The chart must contain exactly 45 data points.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The chart canvas itself remains horizontal; banking refers to adjusting the aspect ratio so internal slopes average $45^\circ$.
- **Option B (Incorrect):** Rotating tick labels is a method to prevent label overlap, not Cleveland's banking principle.
- **Option C (Correct):** Human visual perception is most sensitive to differences in orientation when slopes are near $\pm 45^\circ$. If a line chart is extremely wide and short, all slopes flatten near $0^\circ$, obscuring trends; if it is extremely tall and narrow, all slopes cluster near $90^\circ$, exaggerating noise. Adjusting the aspect ratio via `figsize=(width, height)` so that significant slopes average approximately $45^\circ$ optimizes visual readability.
- **Option D (Incorrect):** Banking applies to arbitrary dataset sizes.

</details>

---

### 88. Question

A marketing agency presents a chart showing "Cumulative New App Users Over Time":

```python
cumulative_users = np.cumsum(daily_new_users)
plt.plot(days, cumulative_users, 'g-')
```

Why is presenting **only** a cumulative line chart often considered a deceptive or misleading visualization tactic when a company's acquisition is collapsing?

- [ ] **A.** Because cumulative lines can only grow or plateau ($\frac{dy}{dt} \ge 0$), giving a superficial visual impression of continuous growth even when daily new user acquisition has plummeted to near zero.
- [ ] **B.** Because cumulative sums cause floating-point arithmetic overflow in Python.
- [ ] **C.** Because cumulative lines cannot be plotted with green strokes.
- [ ] **D.** Because Matplotlib will refuse to render `np.cumsum()` arrays without an enterprise license.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Because cumulative values represent the running integral of daily metrics, they are monotonically non-decreasing. Even if daily signups fall from 10,000/day to 2/day, the cumulative curve still rises to the right (merely slightly flattening). Viewers glancing at the upward trajectory mistakenly conclude the product is thriving, whereas plotting actual daily signups clearly reveals the business collapse.
- **Option B (Incorrect):** NumPy easily handles cumulative integer sums without overflow for standard metrics.
- **Option C (Incorrect):** Format and color styling apply equally to cumulative series.
- **Option D (Incorrect):** Matplotlib is open-source and free.

</details>

---

### 89. Question

A field biologist samples river water temperature only twice a year: once on January 15 ($4^\circ\text{C}$) and once on July 15 ($24^\circ\text{C}$). They draw a single straight line connecting the two data points.

Why is this line plot scientifically and ethically problematic?

- [ ] **A.** Because lines must contain at least 100 points to be legally published.
- [ ] **B.** Because Celsius cannot be represented on a linear axis.
- [ ] **C.** Because water freezes at $4^\circ\text{C}$.
- [ ] **D.** Because linear interpolation across a 6-month seasonal gap assumes a steady, monotonic warming rate, completely masking complex spring weather shifts, cold snaps, and daily temperature volatility.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** There is no legal minimum point threshold, but ethical visualization mandates scientific validity.
- **Option B (Incorrect):** Temperature scales in Celsius are standard linear scales.
- **Option C (Incorrect):** Water freezes at $0^\circ\text{C}$, not $4^\circ\text{C}$.
- **Option D (Correct):** Drawing an unbroken line segment between sparse, distant temporal observations asserts linear interpolation across unobserved time. It falsely implies that temperature rose steadily and uniformly each day between January and July, hiding true seasonal variance, spring chills, and atmospheric volatility. For such sparse observations, showing points without connecting lines (or clearly stating that the line is an unverified interpolation) is proper practice.

</details>

---

### 90. Question

Review the following statements regarding the ethics and appropriateness of line charts:

1. A line chart is ethically valid whenever the horizontal axis represents an ordered, continuous continuum where intermediate values and slopes are meaningful.
2. If changing the order of categories on the x-axis completely changes the visual shape of the chart, a bar chart should be used instead of a line chart.
3. Converting date strings to `pd.to_datetime()` before plotting ensures that physical spacing along the x-axis reflects actual elapsed time rather than uniform category indices.
4. Truncating the vertical axis on a line chart is acceptable when analyzing subtle variations around a large non-zero baseline, provided the axis is explicitly labeled and the truncation is not used to exaggerate statistically insignificant noise.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** Continuous domains (time, distance, temperature) provide the physical justification for connecting points with a line.
- **Statement 2 is correct:** Failing the permutation test confirms that categories are nominal, making connecting lines deceptive.
- **Statement 3 is correct:** Datetime objects map to linear numerical scales (e.g., timestamps or Matplotlib date floats), preserving proportional chronological gaps.
- **Statement 4 is correct:** Unlike bar charts (which require a zero baseline to preserve area proportions), line charts encode position and change; zooming in on a non-zero range is legitimate if transparently labeled and contextually justified.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
