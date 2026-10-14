# Matplotlib Pie Chart Objective Test — Set 01

---

### 1. Question

A data analyst writes the following code to visualize class distribution in a machine learning dataset:

```python
import matplotlib.pyplot as plt

counts = [30, 45, 25]
plt.pie(counts)
plt.show()
```

What total percentage and total angular sweep do the values in `counts` represent on the resulting chart?

- [ ] **A.** Exactly 100% of the data and a complete $360^\circ$ circle, because Matplotlib automatically normalizes input values when their sum is greater than 1.0.
- [ ] **B.** Only $27.8\%$ ($100^\circ$) of a circle, because the raw numbers do not contain a percentage sign.
- [ ] **C.** The function raises a `ValueError` because the numbers were not explicitly converted to fractions between $0.0$ and $1.0$ beforehand.
- [ ] **D.** Exactly $180^\circ$ (a semicircle) because 3 categories default to half-plane projection.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, if the sum of the input array $\sum x > 1.0$, `plt.pie()` automatically normalizes each value by the sum of all elements ($x_i / \sum x$). Here, $30 + 45 + 25 = 100$, so the values represent $30\%$, $45\%$, and $25\%$ respectively, spanning the entire $360^\circ$ circle.
- **Option B (Incorrect):** Matplotlib does not require percentage signs or raw percentages; it performs relative fractional division across the total sum.
- **Option C (Incorrect):** `plt.pie()` accepts arbitrary positive numerical magnitudes (integers, floats) and normalizes them automatically.
- **Option D (Incorrect):** The number of categories does not constrain the circular sweep; three normalized slices span the full $360^\circ$ circumference.

</details>

---

### 2. Question

A data scientist needs to calculate the exact angular sweep $\theta$ (in degrees) occupied by a category with 45 observations in a survey dataset totaling 180 responses. What is the wedge angle rendered by `plt.pie()`?

- [ ] **A.** $45^\circ$
- [ ] **B.** $120^\circ$
- [ ] **C.** $90^\circ$
- [ ] **D.** $75^\circ$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** $45^\circ$ confuses the raw count of 45 observations with the angular degrees of a $360^\circ$ circle.
- **Option B (Incorrect):** $120^\circ$ corresponds to one-third of a circle ($\frac{1}{3} \times 360^\circ$), which would require 60 observations.
- **Option C (Correct):** The angular sweep of any wedge in a full pie chart is given by:
  $$\theta = \left(\frac{v_i}{\sum v}\right) \times 360^\circ = \left(\frac{45}{180}\right) \times 360^\circ = 0.25 \times 360^\circ = 90^\circ$$
  The category represents exactly one-quarter of the pie ($25\%$), which equals a $90^\circ$ right angle.
- **Option D (Incorrect):** $75^\circ$ corresponds to a fraction of $75/360 \approx 20.83\%$, not $25\%$.

</details>

---

### 3. Question

When calling `patches, texts, autotexts = ax.pie(sizes, labels=categories, autopct='%1.1f%%')`, what Python objects are returned inside the `patches` list?

- [ ] **A.** A list of `matplotlib.lines.Line2D` objects representing the slice perimeter borders
- [ ] **B.** A list of `matplotlib.patches.Wedge` artist objects representing each circular sector
- [ ] **C.** A 2D NumPy array of RGB pixel values corresponding to slice fills
- [ ] **D.** A list of `matplotlib.text.Annotation` arrow pointers

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Perimeter outlines belong to the wedge patches themselves; they are not separate `Line2D` artists.
- **Option B (Correct):** `ax.pie()` constructs each pie slice as an instance of `matplotlib.patches.Wedge`. The returned `patches` variable contains a list of these `Wedge` objects, which can be styled individually (e.g., setting face colors, edge colors, or hatch patterns).
- **Option C (Incorrect):** Matplotlib operates using high-level vector artist objects, not rasterized pixel matrices.
- **Option D (Incorrect):** Annotations are separate text-and-arrow artists; standard pie slices do not generate `Annotation` objects unless explicitly coded.

</details>

---

### 4. Question

An analyst configures the percentage formatting of a customer churn chart using `autopct='%1.1f%%'`. What is the purpose of the double percent sign `%%` at the end of the format string?

- [ ] **A.** It instructs Matplotlib to run a modulo division operation on the angle.
- [ ] **B.** It squares the percentage value for exponential scaling.
- [ ] **C.** It acts as an escape sequence in Python string formatting to print a literal `%` symbol following the numerical value.
- [ ] **D.** It specifies that two decimal digits must be appended after the integer.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** In Python string formatting, `%` introduces a format specifier; it does not perform arithmetic modulo on the plotted artist.
- **Option B (Incorrect):** Formatting strings format textual representation; they never perform mathematical exponentiation.
- **Option C (Correct):** In classic Python string interpolation (C-style formatting), a single `%` introduces a formatting conversion code (e.g., `%f` for float). To output a literal percent character (`%`) in the rendered text, it must be escaped by typing `%%`. Thus, `'%1.1f%%'` formats a float with one decimal place followed by a literal `%` (e.g., `"24.5%"`).
- **Option D (Incorrect):** The number of decimal places is controlled by the `.1` in `%1.1f`, not by duplicating the percent symbol.

</details>

---

### 5. Question

A software engineer creates a pie chart on a wide rectangular canvas:

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.pie([40, 35, 25], labels=["A", "B", "C"])
plt.show()
```

Upon rendering, the pie chart appears squished horizontally into an elongated, oval ellipse rather than a circle. Which command ensures the pie chart renders as a true circle?

- [ ] **A.** `ax.set_shape('circle')`
- [ ] **B.** `ax.axis('equal')`
- [ ] **C.** `plt.pie_round(True)`
- [ ] **D.** `fig.set_aspect_ratio(1.0)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `set_shape()` is not an Axes method in Matplotlib.
- **Option B (Correct):** When the figure canvas is rectangular (e.g., 10 inches wide by 4 inches tall), the axes coordinate system stretches disproportionately along the wider dimension. Calling `ax.axis('equal')` (or `plt.axis('equal')`) forces an aspect ratio of 1:1, ensuring that one unit in $x$ equals one unit in $y$ so that the pie renders as a perfect circle.
- **Option C (Incorrect):** `plt.pie_round()` does not exist.
- **Option D (Incorrect):** Figures do not have a `set_aspect_ratio()` method; aspect ratios are enforced at the Axes level via `ax.set_aspect('equal')` or `ax.axis('equal')`.

</details>

---

### 6. Question

A manufacturing quality report plots defect counts across three assembly lines: Line 1 has 60 defects, Line 2 has 90 defects, and Line 3 has 150 defects. What is the sweep angle of Line 1's wedge?

- [ ] **A.** $60^\circ$
- [ ] **B.** $90^\circ$
- [ ] **C.** $108^\circ$
- [ ] **D.** $72^\circ$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $60^\circ$ confuses the raw count of 60 with the angle in degrees.
- **Option B (Incorrect):** $90^\circ$ corresponds to $25\%$ ($75$ defects out of $300$).
- **Option C (Incorrect):** $108^\circ$ corresponds to $30\%$ ($90$ defects out of $300$), which is the angle for Line 2.
- **Option D (Correct):** First, calculate total defects: $60 + 90 + 150 = 300$.
  Then, compute Line 1's proportion: $\frac{60}{300} = 0.20$ ($20\%$).
  Finally, compute the angular sweep: $0.20 \times 360^\circ = \mathbf{72^\circ}$.

</details>

---

### 7. Question

An analyst attempts to plot quarterly net profit including a loss quarter:

```python
profits = [120, -35, 80, 45]
plt.pie(profits)
plt.show()
```

What is the result when Matplotlib executes this statement?

- [ ] **A.** It plots the negative value in red pointing toward the center.
- [ ] **B.** It raises a `ValueError: Wedge sizes must be non-negative`.
- [ ] **C.** It inverts the negative number using its absolute value ($|-35| = 35$) without warning.
- [ ] **D.** It subtracts 35 from the total sum and draws a partial semicircle.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib wedges cannot have negative angular sweeps or point inward.
- **Option B (Correct):** Pie charts represent non-negative parts of a whole (area and angular quantities cannot be negative). Passing negative numbers into `plt.pie()` causes Matplotlib to raise `ValueError: Wedge sizes must be non-negative`. Datasets with negative values must be plotted using bar charts or waterfall charts.
- **Option C (Incorrect):** Matplotlib never silently takes absolute values of user data without explicit instruction.
- **Option D (Incorrect):** Negative values are not subtracted from other wedges; execution halts with a `ValueError`.

</details>

---

### 8. Question

Which parameter in `plt.pie()` controls the radial distance of the category labels from the center of the pie chart (where $1.0$ corresponds to the outer circle perimeter)?

- [ ] **A.** `labeldistance`
- [ ] **B.** `pctdistance`
- [ ] **C.** `radius_offset`
- [ ] **D.** `text_padding`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `labeldistance` specifies the radial distance at which category labels are drawn, measured as a fraction of the circle radius. The default value is `1.1`, which places labels slightly outside the perimeter of the pie. Setting `labeldistance=0.8` draws labels inside the wedges.
- **Option B (Incorrect):** `pctdistance` controls the radial distance of the numerical percentage strings (`autopct`), defaulting to `0.6` (inside the wedge).
- **Option C (Incorrect):** `radius_offset` is not a valid parameter of `plt.pie()`.
- **Option D (Incorrect):** `text_padding` is not a recognized keyword argument.

</details>

---

### 9. Question

Given a category that accounts for exactly $\frac{1}{3}$ ($33.3333\%$) of total observations, what exact string will be rendered inside the wedge if `autopct='%1.0f%%'` is specified?

- [ ] **A.** `'33.3%'`
- [ ] **B.** `'33%'`
- [ ] **C.** `'0.33'`
- [ ] **D.** `'33.33%'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** One decimal place would require `%1.1f%%`.
- **Option B (Correct):** In `%1.0f%%`, `.0f` specifies zero decimal places, rounding the floating-point percentage $33.3333\%$ to the nearest integer, which is `33`. The trailing `%%` adds a literal percent symbol, producing `'33%'`.
- **Option C (Incorrect):** Autopct operates on percentages ($0$ to $100$), not raw fractional decimals ($0.0$ to $1.0$).
- **Option D (Incorrect):** Two decimal places would require `%1.2f%%`.

</details>

---

### 10. Question

Review the following statements regarding the fundamentals of Matplotlib pie charts:

1. If `autopct` is provided, `ax.pie()` returns a 3-tuple `(patches, texts, autotexts)`.
2. Slices in `plt.pie()` are drawn in the exact sequential order of the input array.
3. If `autopct` is omitted, `ax.pie()` returns only a 2-tuple `(patches, texts)`.
4. Setting `plt.axis('equal')` ensures that circular wedges do not distort into ellipses on non-square figure windows.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** When `autopct` is supplied, Matplotlib generates percentage labels and returns them in the third element (`autotexts`).
- **Statement 2 is correct:** Wedges are plotted sequentially starting from the base angle, preserving the order of elements in the input list.
- **Statement 3 is correct:** Without `autopct`, no percentage text artists are created, so `ax.pie()` returns only `(patches, texts)`.
- **Statement 4 is correct:** `plt.axis('equal')` forces a 1:1 aspect ratio, preventing geometric distortion.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
