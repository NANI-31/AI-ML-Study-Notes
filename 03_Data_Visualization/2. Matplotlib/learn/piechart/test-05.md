# Matplotlib Pie Chart Objective Test — Set 05

---

### 41. Question

A data visualization specialist creates a modern **donut chart** using the following clean syntax:

```python
plt.pie(data, labels=labels, wedgeprops=dict(width=0.4))
```

Assuming the default outer radius of `1.0`, what is the exact radius of the inner empty central hole created by this command?

- [ ] **A.** `0.6`
- [ ] **B.** `0.4`
- [ ] **C.** `0.0` (solid pie)
- [ ] **D.** `0.8`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib's `Wedge` patch, the `width` parameter defines the radial thickness of the outer ring. Given the default outer radius $r_{\text{outer}} = 1.0$, specifying `width = 0.4` means the solid wedge spans from radius $0.6$ to $1.0$. The inner radius (the central hole) is calculated as:
  $$r_{\text{inner}} = r_{\text{outer}} - \text{width} = 1.0 - 0.4 = \mathbf{0.6}$$
- **Option B (Incorrect):** `0.4` is the thickness of the ring itself, not the radius of the inner hole.
- **Option C (Incorrect):** `width=1.0` would create a solid pie with zero hole.
- **Option D (Incorrect):** A hole radius of `0.8` would require a thin ring width of $1.0 - 0.8 = 0.2$.

</details>

---

### 42. Question

An analyst is designing a donut chart with `radius=1.2` and `wedgeprops={'width': 0.45}`. What is the inner hole radius?

- [ ] **A.** `0.45`
- [ ] **B.** `0.55`
- [ ] **C.** `0.75`
- [ ] **D.** `1.00`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `0.45` is the ring width.
- **Option B (Incorrect):** `0.55` would result from subtracting $0.45$ from $1.0$, but the outer radius was explicitly set to $1.2$.
- **Option C (Correct):** The inner hole radius is determined by subtracting the ring width from the total outer radius:
  $$r_{\text{inner}} = r_{\text{outer}} - \text{width} = 1.2 - 0.45 = \mathbf{0.75}$$
- **Option D (Incorrect):** Subtracting $0.45$ from $1.2$ yields $0.75$, not $1.00$.

</details>

---

### 43. Question

What is the visual result if a developer specifies `wedgeprops=dict(width=1.0)` on a pie chart with default `radius=1.0`?

- [ ] **A.** An error is raised because donut charts cannot have integer widths.
- [ ] **B.** A tiny microscopic dot in the center.
- [ ] **C.** The chart becomes completely transparent.
- [ ] **D.** It renders as a standard, fully solid pie chart with zero central hole.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Integer float values like `1.0` are valid arguments.
- **Option B (Incorrect):** The ring spans from $0.0$ to $1.0$; it is not a dot.
- **Option C (Incorrect):** The fill color is preserved.
- **Option D (Correct):** Because the ring width ($1.0$) equals the entire radius ($1.0$), the inner boundary of the wedge reaches the origin: $r_{\text{inner}} = 1.0 - 1.0 = 0.0$. Thus, each wedge extends from the center $(0, 0)$ all the way to the outer perimeter, producing a conventional solid pie chart without any central cutout.

</details>

---

### 44. Question

What exception is raised if a developer specifies a ring width that exceeds the outer radius, such as `plt.pie(data, radius=1.0, wedgeprops=dict(width=1.3))`?

- [ ] **A.** `OverflowError: circle radius exceeded`
- [ ] **B.** `ValueError: Radial width must be less than or equal to radius`
- [ ] **C.** `GeometryWarning` (renders inverted wedges)
- [ ] **D.** `TypeError: width cannot exceed 100%`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Python does not raise an `OverflowError` for geometric parameter boundaries.
- **Option B (Correct):** In Matplotlib's `Wedge` patch implementation, the inner radius cannot be negative ($r_{\text{inner}} = r - \text{width} \ge 0$). Specifying $\text{width} > r$ would result in a mathematically invalid negative inner radius, prompting Matplotlib to raise `ValueError: Radial width must be less than or equal to radius`.
- **Option C (Incorrect):** It does not emit a warning; it halts execution with a `ValueError`.
- **Option D (Incorrect):** The parameter takes floating-point coordinates, not percent strings.

</details>

---

### 45. Question

An engineer uses the older **Center Circle Patch** method (Method B) to create a donut chart:

```python
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)
plt.savefig("donut.png", transparent=True)
```

When the exported `donut.png` is placed on a dark blue slide presentation, what undesirable artifact appears in the center of the donut?

- [ ] **A.** A solid white circle covers the dark blue slide in the center because the patch is an opaque white disc rather than a true transparent cutout.
- [ ] **B.** The entire image turns black.
- [ ] **C.** The chart collapses to zero size.
- [ ] **D.** The text labels turn invisible.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The `plt.Circle((0,0), 0.70, fc='white')` overlay technique does not create a true geometric hole; it merely paints an opaque white disc over the center of solid pie wedges. When exported with `transparent=True` and overlaid on a non-white presentation slide or webpage, the white disc stands out starkly against the background. In contrast, the modern `wedgeprops=dict(width=0.3)` approach constructs true annular ring wedges with a genuine empty void that preserves background transparency.
- **Option B (Incorrect):** Transparency exports do not turn images black.
- **Option C (Incorrect):** Image dimensions remain unchanged.
- **Option D (Incorrect):** Text labels on wedges are not altered by the central circle patch.

</details>

---

### 46. Question

A business intelligence dashboard needs to display a central summary metric (e.g., `"Total Users\n14,250"`) positioned precisely inside the empty hole of a donut chart. Which command correctly centers this text at the origin?

- [ ] **A.** `ax.title("Total Users\n14,250", loc='center')`
- [ ] **B.** `ax.annotate_center("Total Users\n14,250")`
- [ ] **C.** `ax.text(0, 0, "Total Users\n14,250", ha='center', va='center', fontsize=14, fontweight='bold')`
- [ ] **D.** `plt.donut_label("Total Users\n14,250")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `ax.title()` (or `ax.set_title()`) places text above the top spine of the plot area, not at the origin $(0, 0)$.
- **Option B (Incorrect):** `annotate_center` is not a Matplotlib method.
- **Option C (Correct):** Because a pie or donut chart is centered at Cartesian coordinates $(x=0, y=0)$, calling `ax.text(0, 0, ...)` with horizontal alignment `ha='center'` and vertical alignment `va='center'` places the text block symmetrically in the exact center of the donut hole.
- **Option D (Incorrect):** `donut_label` does not exist in Matplotlib.

</details>

---

### 47. Question

An analyst wants to render a stylish segmented donut ring where each slice has a thickness of $0.35$ and is separated from adjacent slices by a prominent white border of width 3 points. Which configuration achieves this?

- [ ] **A.** `wedgeprops={'width': 0.35, 'border': 3, 'color': 'white'}`
- [ ] **B.** `wedgeprops={'thickness': 0.35, 'edge': 'white', 'lw': 3}`
- [ ] **C.** `wedgeprops={'ring': 0.35, 'gap': 3}`
- [ ] **D.** `wedgeprops={'width': 0.35, 'edgecolor': 'white', 'linewidth': 3}`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `border` is not a valid `Wedge` patch property; the property is `linewidth`.
- **Option B (Incorrect):** `thickness` is not valid; the parameter is `width`.
- **Option C (Incorrect):** `ring` and `gap` are not recognized Matplotlib properties.
- **Option D (Correct):** In Matplotlib's `Wedge` artist, `width` specifies the radial width of the annular sector ($0.35$), `edgecolor='white'` sets the boundary stroke color to white, and `linewidth=3` establishes the 3-point separation stroke between adjacent slices.

</details>

---

### 48. Question

A developer attempts to create a donut chart using the central circle patch method:

```python
centre_circle = plt.Circle((0, 0), 0.70, fc='white', zorder=0)
fig.gca().add_artist(centre_circle)
ax.pie(sizes)
plt.show()
```

The resulting chart displays a completely solid pie chart with **no hole**. Why did the central hole fail to appear?

- [ ] **A.** The radius $0.70$ was too small to be visible.
- [ ] **B.** Because `zorder=0` was specified, the circle was drawn on the bottom layer, and the pie wedges (drawn later with default `zorder=1`) completely covered the circle.
- [ ] **C.** `plt.Circle` cannot be used inside `fig.gca()`.
- [ ] **D.** The color `'white'` is automatically converted to transparent by Matplotlib.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A radius of $0.70$ spans 70% of the entire circle and would be very large and obvious.
- **Option B (Correct):** Layer stacking is governed by `zorder` and draw sequence. Because the circle patch was given `zorder=0` (and added before `ax.pie()`), Matplotlib drew the circle first on the canvas. The subsequent `ax.pie()` call drew opaque wedges over the top of the circle, completely concealing it. To fix this, the circle must be given a higher `zorder` (e.g., `zorder=2`) or added after plotting the pie.
- **Option C (Incorrect):** Adding artists via `add_artist()` is valid Matplotlib syntax.
- **Option D (Incorrect):** Color strings like `'white'` render as fully opaque RGB $(1, 1, 1)$ by default.

</details>

---

### 49. Question

A donut chart is configured with outer radius `1.0` and `wedgeprops=dict(width=0.4)`. The solid annular ring spans from $r = 0.6$ to $r = 1.0$.

Where should `pctdistance` be positioned so that the numerical percentage text sits **directly in the radial center** of the solid donut ring?

- [ ] **A.** `pctdistance = 0.8`
- [ ] **B.** `pctdistance = 0.4`
- [ ] **C.** `pctdistance = 0.6`
- [ ] **D.** `pctdistance = 0.5`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The inner edge of the ring sits at $r_{\text{inner}} = 1.0 - 0.4 = 0.6$, and the outer edge sits at $r_{\text{outer}} = 1.0$. The radial midpoint of the solid ring is the arithmetic mean:
  $$r_{\text{mid}} = \frac{r_{\text{inner}} + r_{\text{outer}}}{2} = \frac{0.6 + 1.0}{2} = \frac{1.6}{2} = \mathbf{0.8}$$
  Setting `pctdistance=0.8` centers the text labels perfectly between the inner hole and outer perimeter.
- **Option B (Incorrect):** `0.4` places the text inside the empty central hole ($r < 0.6$).
- **Option C (Incorrect):** `0.6` aligns the text right on the inner rim of the hole.
- **Option D (Incorrect):** `0.5` sits floating inside the inner hole.

</details>

---

### 50. Question

Consider the following statements regarding donut chart architecture in Matplotlib:

1. `wedgeprops=dict(width=w)` creates true annular wedges with an empty transparent center void.
2. The radius of the central hole created by `wedgeprops` is given by $r_{\text{inner}} = r_{\text{outer}} - \text{width}$.
3. Central KPI annotations can be added using `ax.text(0, 0, ..., ha='center', va='center')`.
4. The `pctdistance` parameter should be tuned based on the ring width to keep percentage labels centered inside the solid band.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** Annular wedges preserve true background transparency.
- **Statement 2 is correct:** Radial width subtracts directly from outer radius.
- **Statement 3 is correct:** Cartesian origin $(0, 0)$ corresponds to the exact center of the hole.
- **Statement 4 is correct:** Centering labels within the ring band requires setting `pctdistance` to the radial midpoint $\frac{r_{\text{inner}} + r_{\text{outer}}}{2}$.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
