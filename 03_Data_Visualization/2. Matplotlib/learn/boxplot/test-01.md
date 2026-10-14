# Matplotlib Box Plot Objective Test — Set 01

---

### 1. Question

A data analyst creates a basic box plot to visualise the distribution of daily website response times (in milliseconds):

```python
import matplotlib.pyplot as plt

response_times = [120, 135, 142, 148, 155, 160, 162, 165, 170, 185, 340]

fig, ax = plt.subplots()
ax.boxplot(response_times)
ax.set_ylabel("Response Time (ms)")
ax.set_title("Server Response Time Distribution")
plt.show()
```

What does the **horizontal line inside the box** of the resulting plot represent?

- [ ] **A.** The arithmetic mean of `response_times`
- [ ] **B.** The median (Q₂ / 50th percentile) of `response_times`
- [ ] **C.** The midpoint between the minimum and maximum values
- [ ] **D.** The value at which 75% of the data falls below

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** By default, `plt.boxplot()` does NOT display the mean as a line. The center line represents the **median**. To overlay the mean, you must explicitly pass `showmeans=True` (which adds a point marker) or `showmeans=True, meanline=True` (which adds a dashed line). This is a common misconception — many learners confuse the median line with the mean.
- **Option B (Correct):** The horizontal line drawn through the interior of the box represents the **median** — the value that divides the dataset into the lower 50% and upper 50%. For the given dataset of 11 values sorted in ascending order, the median is the 6th value = 160 ms. The median is a robust statistic, unaffected by the single outlier (340 ms).
- **Option C (Incorrect):** The midpoint between minimum and maximum is the **midrange** = (120 + 340) / 2 = 230 ms. This measure is extremely sensitive to outliers and is NOT what the box plot's center line shows.
- **Option D (Incorrect):** The value below which 75% of data falls is **Q₃** (the third quartile / 75th percentile), which is represented by the **top edge** of the box — not the center line.

</details>

---

### 2. Question

A developer generates the following box plot and a colleague asks: "How many components does a standard box plot have?"

A complete, standard Matplotlib box plot for a single group consists of which elements?

- [ ] **A.** Box, median line, mean line, two whiskers, two caps — 6 components
- [ ] **B.** Box, median line, two whiskers, two caps, and individual flier (outlier) markers
- [ ] **C.** Box, median line, mean line, standard deviation lines, two caps — 5 components
- [ ] **D.** Box and whiskers only — 2 components

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A standard box plot does NOT include a mean line by default. The mean line only appears when `showmeans=True` and `meanline=True` are explicitly specified. The default has no mean representation.
- **Option B (Correct):** A standard Matplotlib box plot consists of: (1) the **box** spanning Q₁ to Q₃, (2) the **median line** inside the box at Q₂, (3) two **whiskers** extending from the box edges toward the data extremes, (4) two **caps** (short horizontal lines) at the whisker ends, and (5) **flier markers** (individual dots or crosses) for any data points outside the whisker boundaries. These are all accessible through the dictionary returned by `ax.boxplot()` via keys `'boxes'`, `'medians'`, `'whiskers'`, `'caps'`, and `'fliers'`.
- **Option C (Incorrect):** Standard deviation lines are not part of a box plot. Box plots use quartile-based statistics (median, Q₁, Q₃, IQR), not mean and standard deviation. Adding standard deviation would create a different chart type.
- **Option D (Incorrect):** "Box and whiskers only" is an incomplete description. Box plots always include the median line, caps at whisker ends, and potentially flier markers for outliers.

</details>

---

### 3. Question

A researcher studies the annual rainfall (mm) in five cities:

```
Sorted data: [280, 320, 380, 420, 460, 510, 560, 620, 680]
```

Using the standard inclusive quartile method, what is **Q₂ (the median)** of this 9-element dataset?

- [ ] **A.** 430 (average of 4th and 5th values)
- [ ] **B.** 510 (5th value)
- [ ] **C.** 460 (5th value in 0-indexed view)
- [ ] **D.** 485 (average of 460 and 510)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Averaging the 4th and 5th values (420 and 460) gives 440 — the approach used when the dataset has an EVEN number of elements. For an odd-count dataset (n=9), the median is the single middle value.
- **Option B (Correct):** With 9 values (odd count), the median is at position ⌈n/2⌉ = ⌈9/2⌉ = 5th value. Counting: 280(1), 320(2), 380(3), 420(4), **460(5)**, 510(6), 560(7), 620(8), 680(9). Wait — the 5th value is 460, not 510. The sorted list places values at positions: pos 1=280, 2=320, 3=380, 4=420, 5=460, 6=510, 7=560, 8=620, 9=680. The 5th value is **460**. However, the options list 510 as Option B. Let me recount: dataset = [280, 320, 380, 420, 460, 510, 560, 620, 680], positions 1–9, middle = position 5 = **460**. **Corrected answer: C (460).**
- **Option C (Correct — revised):** The 5th element of the 9-element sorted dataset is **460**. For any dataset with n elements, the median is at position (n+1)/2 = (9+1)/2 = 5. The 5th element is 460. **C is the correct answer.**
- **Option D (Incorrect):** Averaging 460 and 510 (positions 5 and 6) applies to even-count datasets. With n=9 (odd), a single median value exists at position 5.

</details>

---

### 4. Question

A data scientist explains the **Five-Number Summary** to an intern. Match each component to its correct description:

Which of the following correctly describes the **Q₁ (First Quartile)** in a box plot?

- [ ] **A.** The value below which 50% of the data falls — represented by the center line of the box
- [ ] **B.** The value below which 25% of the data falls — represented by the **bottom edge** of the box
- [ ] **C.** The value below which 75% of the data falls — represented by the **top edge** of the box
- [ ] **D.** The average of the minimum and median values

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The value below which 50% of the data falls is **Q₂ (the median)**, represented by the center horizontal line inside the box — not Q₁.
- **Option B (Correct):** **Q₁** is the first quartile — the value at the **25th percentile**. Exactly 25% of the data lies at or below Q₁. In a vertical box plot, Q₁ is represented by the **lower (bottom) edge** of the rectangular box. The box spans from Q₁ to Q₃, with the median line dividing it internally.
- **Option C (Incorrect):** The value below which 75% of the data falls is **Q₃ (the third quartile)**, represented by the **upper (top) edge** of the box.
- **Option D (Incorrect):** Q₁ is a percentile-based statistic, not a midpoint calculation. The average of minimum and median is not a standard statistical measure and is not used in box plot construction.

</details>

---

### 5. Question

A developer creates a box plot of exam scores and adds a title and axis labels:

```python
scores = [45, 52, 58, 63, 67, 72, 75, 78, 80, 85, 91, 95]

fig, ax = plt.subplots(figsize=(5, 7))
ax.boxplot(scores)
ax.set_title("Exam Score Distribution", fontsize=14)
ax.set_ylabel("Score (out of 100)")
ax.set_xlabel("Midterm Exam")
plt.show()
```

By default, in which orientation does `ax.boxplot()` render the box?

- [ ] **A.** Horizontal — the box extends left and right along the x-axis
- [ ] **B.** Diagonal — tilted at 45° for better readability
- [ ] **C.** Vertical — the box extends upward and downward along the y-axis
- [ ] **D.** Circular — the box is rendered as a violin shape by default

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Horizontal orientation is the NON-default. To create a horizontal box plot, you must explicitly pass `vert=False` (or in newer Matplotlib versions, `orientation="horizontal"`). Without this argument, the default is vertical.
- **Option B (Incorrect):** Box plots are never rendered diagonally. They are axis-aligned rectangles. A 45° orientation is not a supported feature of Matplotlib's `boxplot()` function.
- **Option C (Correct):** By default, `ax.boxplot()` creates a **vertical** box plot where: the y-axis represents the data value scale, the box spans Q₁ (bottom) to Q₃ (top) vertically, the median line is a horizontal line inside the box, and whiskers extend upward and downward. The x-axis only serves as a categorical position axis for the group label(s).
- **Option D (Incorrect):** A violin plot is a completely different chart type — `ax.violinplot()` — that shows the probability density of the data as a smoothed shape. `ax.boxplot()` always creates a rectangular box shape, never a violin shape.

</details>

---

### 6. Question

A data engineer wants to create a **horizontal** box plot of processing latencies to align with a horizontal dashboard layout. Which of the following code snippets correctly produces a horizontal box plot?

- [ ] **A.** `ax.boxplot(latencies, horizontal=True)`
- [ ] **B.** `ax.boxplot(latencies, vert=False)`
- [ ] **C.** `ax.hboxplot(latencies)`
- [ ] **D.** `ax.boxplot(latencies, orientation=90)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `horizontal=True` is not a valid parameter for `ax.boxplot()`. Using this would cause a `TypeError` for an unexpected keyword argument.
- **Option B (Correct):** `vert=False` is the correct parameter to produce a **horizontal** box plot. When `vert=False`, the data values are plotted along the **x-axis** and the group position is on the y-axis — boxes extend left and right instead of up and down. Note: In Matplotlib ≥3.10, `orientation="horizontal"` is also valid, but `vert=False` is the most widely supported approach across all modern versions.
- **Option C (Incorrect):** `ax.hboxplot()` does not exist in the Matplotlib API. This would raise an `AttributeError`.
- **Option D (Incorrect):** `orientation=90` is not a valid argument. Orientation can be specified as `vert=False` (or `orientation="horizontal"` in ≥3.10). A numeric angle value is not supported.

</details>

---

### 7. Question

A developer creates a basic box plot and is asked what the **IQR** represents visually:

```python
data = [10, 14, 17, 19, 22, 25, 28, 31, 35, 40, 42]
bp = plt.boxplot(data)
```

What does the **Interquartile Range (IQR)** correspond to in the box plot geometry?

- [ ] **A.** The length of both whiskers combined
- [ ] **B.** The height (vertical extent) of the rectangular box — from Q₁ to Q₃
- [ ] **C.** The distance from the minimum to the maximum value (total range)
- [ ] **D.** The distance from the median to Q₃ (upper half of the box)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The whisker lengths depend on the data distribution and the 1.5×IQR rule. The combined whisker length varies by dataset and does not directly equal the IQR.
- **Option B (Correct):** IQR = Q₃ − Q₁ represents the middle 50% of the data. In a box plot, the **rectangular box** spans from Q₁ (bottom edge) to Q₃ (top edge). The **height of the box** is therefore visually equal to the IQR. A narrow box indicates low spread in the middle 50% of data; a wide box indicates high spread.
- **Option C (Incorrect):** The distance from minimum to maximum is the **total range** (max − min). The total range is highly sensitive to outliers. The IQR focuses only on the middle 50%, making it a more robust measure of spread.
- **Option D (Incorrect):** The distance from the median to Q₃ is the **upper half** of the IQR — equal to (Q₃ − Q₂). This is NOT the full IQR. The full IQR spans from Q₁ to Q₃.

</details>

---

### 8. Question

A machine learning practitioner compares two models' prediction error distributions:

```python
model_a_errors = [0.1, 0.3, 0.2, 0.5, 0.4, 0.3, 0.6, 0.2, 0.3, 0.4]
model_b_errors = [0.1, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.1]

plt.boxplot([model_a_errors, model_b_errors],
            tick_labels=["Model A", "Model B"])
plt.ylabel("Absolute Error")
plt.show()
```

A reviewer looks at the box plot and notes that **Model B has a narrower box** than Model A. What does this indicate?

- [ ] **A.** Model B makes more predictions than Model A
- [ ] **B.** Model B has a smaller IQR — the middle 50% of its errors are more tightly concentrated, indicating lower error variance
- [ ] **C.** Model B's errors are always smaller than all of Model A's errors
- [ ] **D.** Model B's median error is lower than Model A's median error

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The box plot box width reflects the IQR (spread of the middle 50% of values), not the count of predictions. Both models have 10 errors in this example, but box width is determined by the data spread, not data count.
- **Option B (Correct):** A narrower box directly corresponds to a smaller IQR (Q₃ − Q₁). For Model B, the errors are tightly clustered (mostly 0.1–0.3), so Q₁ and Q₃ are close together. For Model A, errors range more widely (0.1–0.6), giving a larger IQR. A smaller IQR means the middle 50% of predictions have lower error variability — indicating more consistent and stable model performance.
- **Option C (Incorrect):** A narrower box means the middle 50% is more concentrated, but it doesn't guarantee that ALL of Model B's errors are smaller than ALL of Model A's. The models' value ranges can overlap significantly.
- **Option D (Incorrect):** Box width relates to spread (IQR), not median position. The median line position (low or high within the box) determines median value comparisons. A narrower box tells us nothing directly about the relative median heights of the two models.

</details>

---

### 9. Question

A developer passes the following data to `plt.boxplot()` and checks the returned structure:

```python
import numpy as np

data = np.random.normal(50, 10, 100)
result = plt.boxplot(data)
print(type(result))
print(result.keys())
```

What type does `plt.boxplot()` return, and what are the keys in its return value?

- [ ] **A.** Returns a `list` of `Line2D` objects for each visual element
- [ ] **B.** Returns a `dict` with keys `'boxes'`, `'medians'`, `'whiskers'`, `'caps'`, `'fliers'`
- [ ] **C.** Returns a `Figure` object containing the drawn box plot
- [ ] **D.** Returns `None` — `plt.boxplot()` is a side-effect-only function

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The return value is not a flat list. It is a structured dictionary containing separate lists for each component type, allowing targeted customisation of individual elements (e.g., changing only the median line colour).
- **Option B (Correct):** `plt.boxplot()` returns a **Python dictionary** with string keys identifying each box plot component: `'boxes'` (the rectangular box Line2D artists), `'medians'` (the median lines), `'whiskers'` (the whisker line pairs), `'caps'` (the cap lines at whisker ends), `'fliers'` (the outlier point markers), and optionally `'means'` if `showmeans=True`. These values are lists of `Line2D` or `PathPatch` objects that can be accessed for post-hoc styling.
- **Option C (Incorrect):** The `Figure` object is created by `plt.figure()` or `plt.subplots()`. `plt.boxplot()` returns the artist dictionary, not the figure.
- **Option D (Incorrect):** `plt.boxplot()` has a return value. While it also has side effects (drawing on the current axes), it explicitly returns the artist dictionary for programmatic access and customisation.

</details>

---

### 10. Question

A developer is writing documentation and needs to explain when to prefer a **horizontal box plot** over a vertical one. Which scenario BEST justifies using `vert=False`?

- [ ] **A.** When the dataset has fewer than 10 data points
- [ ] **B.** When comparing multiple groups with long category names that would overlap if placed on the x-axis, or when the metric axis is naturally horizontal (e.g., response time on a timeline)
- [ ] **C.** When the data contains outliers that must be displayed below the x-axis
- [ ] **D.** When `patch_artist=True` is used to colour the boxes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The number of data points has no bearing on the appropriate orientation. Both vertical and horizontal box plots work with any data size. Orientation is a layout and readability choice.
- **Option B (Correct):** Horizontal box plots (`vert=False`) are preferred when: (1) category/group names are long strings that would require awkward rotation on the x-axis — placing them on the y-axis allows easy horizontal reading; (2) the metric value axis is conceptually horizontal (e.g., a time axis, response latency bar, or percentage scale aligned with other horizontal elements in a dashboard). Feature importance comparisons and model name comparisons commonly use horizontal box plots for this reason.
- **Option C (Incorrect):** Outliers appear outside the whiskers regardless of orientation. Vertical box plots show lower outliers below the lower whisker cap, and horizontal box plots show left-side outliers to the left of the left cap. The presence of outliers alone does not justify orientation choice.
- **Option D (Incorrect):** `patch_artist=True` (enabling filled coloured boxes) works identically for both vertical and horizontal orientations. This styling choice is completely independent of the `vert=` parameter.

</details>
