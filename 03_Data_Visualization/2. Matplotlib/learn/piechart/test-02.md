# Matplotlib Pie Chart Objective Test — Set 02

---

### 11. Question

In Matplotlib's default coordinate system, what are the **default starting position** and **angular direction** used by `plt.pie()` to render the first slice?

- [ ] **A.** Starts at 12 o'clock (positive y-axis) and proceeds clockwise
- [ ] **B.** Starts at 6 o'clock (negative y-axis) and proceeds counterclockwise
- [ ] **C.** Starts at 3 o'clock ($0^\circ$ along the positive x-axis) and proceeds counterclockwise
- [ ] **D.** Starts at 9 o'clock (negative x-axis) and proceeds clockwise

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 12 o'clock clockwise is the traditional business charting convention, but Matplotlib follows standard mathematical polar coordinates by default (`startangle=0`, `counterclock=True`).
- **Option B (Incorrect):** Slices never start at 6 o'clock by default.
- **Option C (Correct):** By default, `plt.pie()` aligns with standard trigonometric polar coordinates: $0^\circ$ corresponds to the positive x-axis (the 3 o'clock position on an analog clock face), and angles increase in the **counterclockwise** direction (`counterclock=True`).
- **Option D (Incorrect):** 9 o'clock corresponds to $180^\circ$, which is not the default starting angle.

</details>

---

### 12. Question

A financial analyst wants the largest segment of a budget pie chart to begin at the very top of the circle (12 o'clock position) and sweep forward. Which parameter configuration achieves this?

- [ ] **A.** `origin='top'`
- [ ] **B.** `rotation='vertical'`
- [ ] **C.** `align='12'`
- [ ] **D.** `startangle=90`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `origin` is not a valid parameter of `plt.pie()`.
- **Option B (Incorrect):** `rotation` is not a recognized pie chart argument.
- **Option C (Incorrect):** Clock face strings like `'12'` are not accepted by Matplotlib.
- **Option D (Correct):** The `startangle` argument rotates the starting position of the first slice by a specified number of degrees counterclockwise from the positive x-axis ($0^\circ$). Rotating by $90^\circ$ (`startangle=90`) shifts the starting boundary to the positive y-axis, which corresponds exactly to the 12 o'clock position.

</details>

---

### 13. Question

An analyst writes the following configuration:

```python
plt.pie([25, 75], labels=["Quarter 1", "Remaining"], startangle=90, counterclock=False)
```

Where does the "Quarter 1" wedge start, and what clock position does its trailing edge reach?

- [ ] **A.** Starts at 12 o'clock and sweeps clockwise to end at 3 o'clock
- [ ] **B.** Starts at 12 o'clock and sweeps counterclockwise to end at 9 o'clock
- [ ] **C.** Starts at 3 o'clock and sweeps clockwise to end at 6 o'clock
- [ ] **D.** Starts at 9 o'clock and sweeps counterclockwise to end at 6 o'clock

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `startangle=90` anchors the leading edge of the first slice at 12 o'clock. `counterclock=False` specifies that wedges sweep in a clockwise direction. Because Quarter 1 accounts for $25\%$ of the total ($25 / 100$), its angular sweep is $0.25 \times 360^\circ = 90^\circ$. Sweeping $90^\circ$ clockwise from 12 o'clock lands precisely at the 3 o'clock position.
- **Option B (Incorrect):** Sweeping counterclockwise from 12 o'clock would land at 9 o'clock, but `counterclock=False` was specified.
- **Option C (Incorrect):** The start angle is 12 o'clock, not 3 o'clock.
- **Option D (Incorrect):** The start angle is 12 o'clock, not 9 o'clock.

</details>

---

### 14. Question

An environmental scientist plots land use where Forest occupies $50\%$ ($180^\circ$) of the total area. The call specifies:

```python
plt.pie([50, 30, 20], labels=["Forest", "Agriculture", "Urban"], startangle=180)
```

Which half of the circular canvas is entirely occupied by the "Forest" wedge?

- [ ] **A.** The top half (from 9 o'clock through 12 o'clock to 3 o'clock)
- [ ] **B.** The right half (from 12 o'clock through 3 o'clock to 6 o'clock)
- [ ] **C.** The left half (from 6 o'clock through 9 o'clock to 12 o'clock)
- [ ] **D.** The bottom half (from 9 o'clock through 6 o'clock to 3 o'clock)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The top half would be covered if the sweep started at 9 o'clock and moved clockwise, or started at 3 o'clock and moved counterclockwise.
- **Option B (Incorrect):** The right half spans from $-90^\circ$ to $+90^\circ$.
- **Option C (Incorrect):** The left half spans from $90^\circ$ to $270^\circ$.
- **Option D (Correct):** `startangle=180` positions the start of the first wedge at $180^\circ$ (the 9 o'clock position on the negative x-axis). Since `counterclock=True` by default, the wedge sweeps counterclockwise: from $180^\circ$ through $270^\circ$ (6 o'clock) to $360^\circ / 0^\circ$ (3 o'clock). This covers the entire **bottom half** of the circle.

</details>

---

### 15. Question

A graphic designer wants to render clean white dividing borders of width 2 points between all pie slices. Which dictionary passed to `wedgeprops` achieves this?

- [ ] **A.** `wedgeprops={'border': 'white', 'width': 2}`
- [ ] **B.** `wedgeprops={'edgecolor': 'white', 'linewidth': 2}`
- [ ] **C.** `wedgeprops={'separator_color': 'white', 'stroke': 2}`
- [ ] **D.** `wedgeprops={'lines': 'white', 'size': 2}`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In Matplotlib's `Wedge` patch, outline color is `edgecolor`, while `width` defines the radial thickness of a donut ring.
- **Option B (Correct):** `wedgeprops` accepts keyword arguments passed directly to the underlying `matplotlib.patches.Wedge` constructors. Setting `edgecolor='white'` specifies a white boundary stroke, and `linewidth=2` establishes a crisp 2-point border width between adjacent wedges.
- **Option C (Incorrect):** `separator_color` and `stroke` are not recognized Matplotlib properties.
- **Option D (Incorrect):** `lines` and `size` are invalid wedge parameters.

</details>

---

### 16. Question

An engineer observes that the percentage text labels inside pie wedges are positioned too close to the outer edge of the chart and wants to pull them inward toward the center. Which parameter should be adjusted?

- [ ] **A.** Decrease `labeldistance`
- [ ] **B.** Set `inner_text=True`
- [ ] **C.** Decrease `pctdistance` (e.g., from default `0.6` to `0.45`)
- [ ] **D.** Increase `radius`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `labeldistance` governs category name labels, not the numerical percentage strings generated by `autopct`.
- **Option B (Incorrect):** `inner_text` is not a valid parameter.
- **Option C (Correct):** `pctdistance` represents the ratio between the center of the pie and the numerical percentage text, measured as a fraction of the circle radius. The default is `0.6`. Decreasing `pctdistance` (e.g., to `0.45`) shifts the percentage labels radially inward toward the center of the chart.
- **Option D (Incorrect):** Increasing `radius` scales the entire circle without changing the relative fractional position of the text.

</details>

---

### 17. Question

A dataset contains a category with an observation count of zero:

```python
plt.pie([50, 50, 0], labels=["Class A", "Class B", "Class C"], autopct='%1.0f%%')
plt.show()
```

How does Matplotlib handle "Class C" during rendering?

- [ ] **A.** It draws a wedge with a $0^\circ$ sweep angle (zero visual area), but may still render the text label at that boundary coordinate.
- [ ] **B.** It raises a `ZeroDivisionError` when computing normalization.
- [ ] **C.** It removes "Class C" from the legend and raises a `UserWarning`.
- [ ] **D.** It automatically distributes $1\%$ from Class A and Class B to make Class C visible.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Matplotlib handles zero values gracefully without crashing: $\frac{0}{100} \times 360^\circ = 0^\circ$. The wedge has zero angular width and zero area, so no slice is visible on the canvas. However, the category label `"Class C"` and its percentage `"0%"` may still be generated at the zero-degree vertex, potentially colliding with adjacent labels.
- **Option B (Incorrect):** Normalization divides by the total sum ($100$), which is non-zero; no division by zero occurs.
- **Option C (Incorrect):** Matplotlib does not drop elements from the label list.
- **Option D (Incorrect):** Matplotlib never manipulates user data to artificially inflate zero-sized slices.

</details>

---

### 18. Question

A web dashboard developer wants to enforce an accessible, brand-compliant color scheme across four pie slices:

```python
palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
plt.pie([20, 30, 25, 25], colors=palette)
```

How does `plt.pie()` map the items in `palette` to the four slices?

- [ ] **A.** It sorts the colors alphabetically by hex code before mapping.
- [ ] **B.** It assigns the brightest color to the largest slice and the darkest to the smallest.
- [ ] **C.** It maps colors randomly on every execution.
- [ ] **D.** It assigns colors sequentially: slice 1 receives the 1st color, slice 2 receives the 2nd, and so on.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Hex strings are not sorted alphabetically.
- **Option B (Incorrect):** Slices are not sorted by luminance or magnitude automatically.
- **Option C (Incorrect):** Color assignment is strictly deterministic.
- **Option D (Correct):** The `colors` argument takes a sequence of color codes and assigns them in 1-to-1 sequential order to the wedges as they appear in the data array:
  - 1st slice ($20$) $\rightarrow$ `'#1b9e77'`
  - 2nd slice ($30$) $\rightarrow$ `'#d95f02'`
  - 3rd slice ($25$) $\rightarrow$ `'#7570b3'`
  - 4th slice ($25$) $\rightarrow$ `'#e7298a'`

</details>

---

### 19. Question

A data analyst plots a pie chart with dark wedge colors and finds that the default black percentage text is difficult to read. The analyst captures the return objects:

```python
wedges, texts, autotexts = ax.pie(data, autopct='%1.1f%%')
```

Which loop correctly updates all percentage labels to **bold white text**?

- [ ] **A.**
```python
for t in texts:
    t.set_color('white')
    t.set_weight('bold')
```
- [ ] **B.**
```python
for at in autotexts:
    at.set_color('white')
    at.set_weight('bold')
```
- [ ] **C.**
```python
for w in wedges:
    w.text_color = 'white'
```
- [ ] **D.**
```python
ax.set_text_properties(autotexts, color='white', bold=True)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `texts` contains the category labels (outside the slices), not the percentage numbers.
- **Option B (Correct):** `autotexts` contains the `Text` artist instances for the numerical percentages inside each wedge. Iterating over `autotexts` and calling standard `Text` methods (`.set_color('white')` and `.set_weight('bold')`) directly styles the percentage labels.
- **Option C (Incorrect):** `Wedge` patches store geometric coordinates; they do not have a `text_color` attribute.
- **Option D (Incorrect):** `set_text_properties()` is not a standard Axes method.

</details>

---

### 20. Question

Consider the following statements regarding angles, rotation, and wedge customization in Matplotlib:

1. Setting `counterclock=False` changes the wedge drawing progression from counterclockwise to clockwise.
2. Setting `startangle=270` places the first slice at 6 o'clock (negative y-axis).
3. Slices with zero values (`0`) do not raise errors, but generate zero-width wedges.
4. Setting `pctdistance=1.0` positions the percentage labels directly along the outer circumference of the circle.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, 3, and 4
- [ ] **D.** 1, 2, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `counterclock=False` enforces clockwise angular progression.
- **Statement 2 is correct:** $270^\circ$ corresponds to the negative y-axis (6 o'clock position).
- **Statement 3 is correct:** Zero values are evaluated as $0^\circ$ sweeps and rendered without error.
- **Statement 4 is correct:** `pctdistance=1.0` places text at exactly $100\%$ of the circle radius, aligning it on the perimeter.
- **Conclusion:** All four statements are correct, making **Option C** the correct answer.

</details>

---
