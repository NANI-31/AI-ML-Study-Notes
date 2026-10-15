# Matplotlib Pie Chart Objective Test — Set 06

---

### 51. Question

A data analyst passes three fractional numbers summing to less than 1.0 into `plt.pie()`:

```python
plt.pie([0.2, 0.3, 0.1])
plt.show()
```

What visual phenomenon occurs on the rendered canvas?

- [ ] **A.** Matplotlib raises an `IncompleteDataError` and halts execution.
- [ ] **B.** Matplotlib draws an **incomplete partial pie** that covers only $60\%$ of the circle ($216^\circ$), leaving a $40\%$ ($144^\circ$) empty wedge gap.
- [ ] **C.** Matplotlib automatically normalizes the three numbers by dividing each by $0.6$, drawing a full $360^\circ$ circle.
- [ ] **D.** The missing $0.4$ fraction is rendered in black with the label `"Unknown"`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib fully supports fractional arrays whose sum is less than or equal to 1.0.
- **Option B (Correct):** When the sum of the input array $\sum x \le 1.0$, Matplotlib treats the values as explicit pre-calculated proportions of a unit circle rather than raw counts needing normalization. The values $0.2 + 0.3 + 0.1 = 0.6$ occupy:
  $$\text{Sweep} = 0.6 \times 360^\circ = 216^\circ$$
  The remaining $1.0 - 0.6 = 0.4$ ($144^\circ$) is left as an empty, unplotted wedge opening, forming a partial "Pac-Man" shaped chart.
- **Option C (Incorrect):** Automatic normalization only activates when $\sum x > 1.0$ (or when `normalize=True` is explicitly passed).
- **Option D (Incorrect):** Matplotlib does not invent placeholder categories or black wedges.

</details>

---

### 52. Question

A progress dashboard visualizes task completion where $25\%$ of tasks are in progress and $50\%$ are completed, with raw input `[0.25, 0.50]`:

```python
plt.pie([0.25, 0.50])
```

What is the exact angular size (in degrees) of the **unplotted empty gap** remaining in the circle?

- [ ] **A.** $90^\circ$
- [ ] **B.** $180^\circ$
- [ ] **C.** $45^\circ$
- [ ] **D.** $0^\circ$ (full circle)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The sum of the inputs is $0.25 + 0.50 = 0.75$.
  Because $\sum x < 1.0$, Matplotlib does not normalize.
  The unplotted fractional void is $1.0 - 0.75 = 0.25$.
  The angular gap is:
  $$\theta_{\text{gap}} = 0.25 \times 360^\circ = \mathbf{90^\circ}$$
- **Option B (Incorrect):** $180^\circ$ corresponds to a remaining fraction of $0.50$.
- **Option C (Incorrect):** $45^\circ$ corresponds to a remaining fraction of $0.125$.
- **Option D (Incorrect):** The circle is not complete because the sum is less than 1.0.

</details>

---

### 53. Question

An engineer has a dataset of decimal proportions $[0.2, 0.3]$ and wants Matplotlib to expand them to fill a **full $360^\circ$ circle** (so they represent $40\%$ and $60\%$) rather than rendering a partial pie. Which parameter achieves this?

- [ ] **A.** `full_circle=True`
- [ ] **B.** `expand=True`
- [ ] **C.** `autoscale=True`
- [ ] **D.** `normalize=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `full_circle` is not a valid parameter.
- **Option B (Incorrect):** `expand` is not a recognized Matplotlib keyword argument.
- **Option C (Incorrect):** `autoscale` does not govern pie wedge normalization.
- **Option D (Correct):** By default, `normalize=None`, which preserves partial pies when $\sum x \le 1.0$. Explicitly passing `normalize=True` forces Matplotlib to normalize the array regardless of its sum:
  $$\frac{0.2}{0.5} = 40\%, \quad \frac{0.3}{0.5} = 60\%$$
  This scales the two slices to span the full $360^\circ$ circumference.

</details>

---

### 54. Question

What is the visual output produced by calling `plt.pie([100])`?

- [ ] **A.** A single horizontal line across the canvas
- [ ] **B.** An empty canvas with a `SingleValueWarning`
- [ ] **C.** A single complete $360^\circ$ circular wedge filling the entire pie
- [ ] **D.** A square with 100 pixels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Pie charts render wedges, not horizontal lines.
- **Option B (Incorrect):** Matplotlib handles single-element arrays without error.
- **Option C (Correct):** Passing a single positive value creates a single wedge whose proportion of the total is $100 / 100 = 1.0$ ($100\%$). It sweeps a full $360^\circ$ circle, rendering a solid, unbroken circular disc.
- **Option D (Incorrect):** `plt.pie()` always generates polar wedges, never Cartesian squares.

</details>

---

### 55. Question

A data visualization designer wants to enlarge the physical size of a pie chart relative to its axes box. Which parameter in `plt.pie()` scales the outer radius (where $1.0$ is the standard default)?

- [ ] **A.** `scale=1.3`
- [ ] **B.** `radius=1.3`
- [ ] **C.** `size=1.3`
- [ ] **D.** `diameter=2.6`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `scale` is not an argument of `plt.pie()`.
- **Option B (Correct):** The `radius` keyword argument accepts a float specifying the radius of the pie in axes data units (default is `1.0`). Setting `radius=1.3` scales the pie up by 30%, making it physically larger within the plot area.
- **Option C (Incorrect):** `size` is not a valid parameter.
- **Option D (Incorrect):** Matplotlib parameterizes circles by radius, not diameter.

</details>

---

### 56. Question

A scientific journal requires black-and-white print figures where categories are distinguished by pattern textures rather than color. How can hatch patterns (such as stripes and dots) be applied to pie wedges?

- [ ] **A.** Pass `texture=['stripes', 'dots']` to `plt.pie()`
- [ ] **B.** Pass `pattern=True` to `plt.pie()`
- [ ] **C.** Hatching cannot be applied to circular wedges in Matplotlib
- [ ] **D.** Iterate through the returned `patches` list and call `patch.set_hatch('//')` on individual wedges, or pass `hatch` in `wedgeprops`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `texture` is not a recognized Matplotlib keyword argument.
- **Option B (Incorrect):** `pattern` is not a valid parameter.
- **Option C (Incorrect):** `Wedge` patches are standard Matplotlib `Patch` objects that fully support hatching.
- **Option D (Correct):** Because `patches` is a list of `matplotlib.patches.Wedge` objects, standard patch styling methods apply. Slices can be given distinctive accessibility patterns by setting `patch.set_hatch('//')`, `patch.set_hatch('\\\\')`, `patch.set_hatch('..')`, or `patch.set_hatch('xx')`, or by supplying a hatch pattern through `wedgeprops`.

</details>

---

### 57. Question

According to data visualization best practices, in what order should categories be sorted around a pie chart to optimize viewer comprehension?

- [ ] **A.** Slices should be sorted from largest to smallest, starting at 12 o'clock and progressing clockwise, with any small "Other" category placed last.
- [ ] **B.** Slices must always be sorted alphabetically regardless of size.
- [ ] **C.** Slices should alternate between the largest and smallest categories.
- [ ] **D.** Slices should be randomized on every refresh to prevent position bias.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Eye-tracking and cognitive perception research show that human viewers naturally begin reading circular dials at 12 o'clock and scan clockwise. Arranging slices in descending order of magnitude establishes an intuitive rank hierarchy, allowing viewers to quickly perceive relative market share. A consolidated "Other" category should always be placed at the end of the sequence regardless of its size.
- **Option B (Incorrect):** Alphabetical sorting scrambles slice sizes unpredictably, making relative comparison difficult.
- **Option C (Incorrect):** Alternating sizes creates visual clutter and frustrates size comparisons.
- **Option D (Incorrect):** Randomization introduces instability and destroys readability.

</details>

---

### 58. Question

What exception is raised if a developer explicitly passes `normalize=False` with values that sum to greater than 1.0:

```python
plt.pie([20, 50, 60], normalize=False)
```

- [ ] **A.** `OverflowError: angle exceeded 360`
- [ ] **B.** `ValueError: When normalize is False, the sum of x must be <= 1`
- [ ] **C.** `AssertionError: pie slices overlap`
- [ ] **D.** Slices are drawn on top of each other in multiple overlapping revolutions without error

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Python does not throw `OverflowError` here.
- **Option B (Correct):** In Matplotlib, setting `normalize=False` asserts that the input values are literal fractional proportions of a single unit circle ($x_i \in [0, 1]$ and $\sum x \le 1.0$). If the sum exceeds $1.0$ ($20 + 50 + 60 = 130 > 1.0$), Matplotlib raises `ValueError: When normalize is False, the sum of x must be <= 1`.
- **Option C (Incorrect):** Matplotlib uses standard `ValueError` validation rather than an `AssertionError`.
- **Option D (Incorrect):** Overlapping spiral revolutions are not permitted; execution halts with a `ValueError`.

</details>

---

### 59. Question

A financial dataset contains extreme skewness: Category A accounts for $99.9\%$ of market volume, while Category B accounts for $0.1\%$.

What visual flaw occurs when plotting this distribution with `plt.pie([99.9, 0.1], labels=['A', 'B'], autopct='%1.1f%%')`?

- [ ] **A.** Category B is rendered as a hairline slit of $0.36^\circ$, causing its label and percentage text to collide and overlap directly over Category A's boundary.
- [ ] **B.** Category B is completely deleted from the plot.
- [ ] **C.** The chart converts into a square scatter plot.
- [ ] **D.** Matplotlib crashes with a `ZeroDivisionError`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Category B's angular sweep is $\frac{0.1}{100} \times 360^\circ = 0.36^\circ$ (barely a fraction of a single degree). On standard screens, this forms an imperceptible hairline wedge. Matplotlib still attempts to render the text strings `"B"` and `"0.1%"` at that coordinate, resulting in text overlapping Category A. Highly skewed distributions are better communicated via KPI text callouts or horizontal bar charts.
- **Option B (Incorrect):** Matplotlib draws all non-zero elements; it does not delete Category B.
- **Option C (Incorrect):** Chart types are never silently swapped.
- **Option D (Incorrect):** Normalization divides by $100.0$, which is non-zero.

</details>

---

### 60. Question

Consider the following statements regarding normalization, partial pies, and wedge scaling in Matplotlib:

1. When $\sum x < 1.0$ and `normalize=None`, Matplotlib renders a partial pie with an open angular gap.
2. The `radius` argument can be set to any positive float (e.g., `radius=0.8` or `radius=1.5`).
3. Explicitly setting `normalize=True` guarantees that input values are normalized to occupy a full $360^\circ$ circle.
4. Setting `normalize=False` requires that the sum of the input array is $\le 1.0$.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** $\sum x < 1.0$ produces a partial pie with an unplotted sector equal to $1.0 - \sum x$.
- **Statement 2 is correct:** `radius` scales the circle geometry within the plot axes.
- **Statement 3 is correct:** `normalize=True` forces normalization by $\sum x$, spanning $360^\circ$.
- **Statement 4 is correct:** `normalize=False` validates that $\sum x \le 1.0$, throwing a `ValueError` if the sum exceeds 1.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
