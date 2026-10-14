# Matplotlib Line Plot Objective Test — Set 02

---

### 11. Question

A data analyst is designing a dashboard and wants data points along a KPI line chart to be rendered as **diamond markers** instead of standard circles. Which marker code character should be used?

- [ ] **A.** `'^'`
- [ ] **B.** `'s'`
- [ ] **C.** `'p'`
- [ ] **D.** `'D'` or `'d'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `'^'` specifies an upward-pointing triangle marker.
- **Option B (Incorrect):** `'s'` specifies a square marker.
- **Option C (Incorrect):** `'p'` specifies a five-sided pentagon marker.
- **Option D (Correct):** In Matplotlib, `'D'` specifies a standard diamond marker, and `'d'` specifies a thin diamond marker. Both render diamond geometries at each vertex coordinate.

</details>

---

### 12. Question

A UX designer requests a "hollow dot" appearance for a high-priority metrics line: the line itself should be solid blue, and the circular markers should have a **white interior fill** with a **solid blue border** of width 2 points. Which line plot call produces this exact visual styling?

- [ ] **A.** `plt.plot(x, y, 'bo-', fill='none', border=2)`
- [ ] **B.** `plt.plot(x, y, 'wo-', linecolor='blue', edgecolor='blue')`
- [ ] **C.** `plt.plot(x, y, color='blue', marker='o', alpha=0.5, hollow=True)`
- [ ] **D.** `plt.plot(x, y, color='blue', marker='o', markerfacecolor='white', markeredgecolor='blue', markeredgewidth=2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `fill='none'` and `border=2` are not valid Matplotlib line artist parameters and will raise a `TypeError: unexpected keyword argument`.
- **Option B (Incorrect):** `linecolor` and `edgecolor` are not supported keyword parameters for `plt.plot()`; the parameters are `color` and `markeredgecolor`.
- **Option C (Incorrect):** Matplotlib does not provide a `hollow=True` keyword argument.
- **Option D (Correct):** Matplotlib provides fine-grained marker styling keywords: `markerfacecolor` (or `mfc`) sets the interior color, `markeredgecolor` (or `mec`) sets the perimeter outline color, and `markeredgewidth` (or `mew`) sets the thickness of the outline in points. Setting `mfc='white'`, `mec='blue'`, and `mew=2` with `color='blue'` creates crisp hollow markers with blue borders.

</details>

---

### 13. Question

An analyst writes the following code to customize the background grid of an economic trends line chart:

```python
plt.plot(years, gdp_growth)
plt.grid(True, linestyle=':', linewidth=1.2, color='gray', alpha=0.5)
```

What visual effect does this configuration produce on the gridlines?

- [ ] **A.** It draws solid black gridlines across all minor and major intervals with 100% opacity.
- [ ] **B.** It draws semi-transparent, gray dotted gridlines that subtly aid value estimation without overpowering the data curve.
- [ ] **C.** It raises a `ValueError` because `plt.grid()` only accepts a boolean `True` or `False`.
- [ ] **D.** It suppresses the grid completely because `alpha=0.5` is below the activation threshold.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The gridlines are not solid black; `linestyle=':'` sets dotted lines, `color='gray'` sets gray color, and `alpha=0.5` creates 50% transparency.
- **Option B (Correct):** `plt.grid(True, ...)` accepts standard line property arguments. Here, `linestyle=':'` renders dotted lines, `color='gray'` provides a muted tone, and `alpha=0.5` ensures the grid stays visually unobtrusive in the background.
- **Option C (Incorrect):** `plt.grid()` accepts rich styling keyword arguments including `linestyle` (or `ls`), `linewidth` (or `lw`), `color` (or `c`), and `alpha`.
- **Option D (Incorrect):** Any positive `alpha` value above 0 renders visible strokes; `alpha=0.5` represents 50% opacity, which is clearly visible.

</details>

---

### 14. Question

A clinical trial monitor plots patient blood glucose levels over 24 hours:

```python
hours = list(range(24))
glucose = [95, 102, 110, 140, 165, 180, 150, 130, 115, 98, ..., 92]

plt.plot(hours, glucose, 'r-')
plt.xlim(6, 18)
plt.ylim(80, 200)
plt.show()
```

What happens to data points recorded at hours 2, 4, 20, and 22?

- [ ] **A.** They are permanently deleted from the underlying `hours` and `glucose` Python lists.
- [ ] **B.** They are clipped from the visible axes window, but the internal line coordinates remain intact and will reappear if limits are expanded.
- [ ] **C.** An `IndexOutOfBoundsError` is thrown during rasterization because coordinates exceed the viewing box.
- [ ] **D.** Their coordinate values are compressed and plotted along the edge borders of the chart.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Calling `plt.xlim()` or `plt.ylim()` modifies the viewport of the axes; it never alters or deletes the underlying data structures.
- **Option B (Correct):** `plt.xlim(6, 18)` sets the horizontal viewport boundaries. Data points outside this range are clipped by the axes bounding box during rendering, leaving the original data arrays unaltered.
- **Option C (Incorrect):** Matplotlib handles out-of-bounds coordinates gracefully using clipping algorithms; no exception is thrown.
- **Option D (Incorrect):** Coordinates outside the viewing limits are not clamped or squashed to the edges; they are simply excluded from the viewport.

</details>

---

### 15. Question

A marine oceanographer is plotting ocean water temperature against depth below sea level. Depth ranges from 0 meters (surface) to 1,000 meters. Scientifically, deeper water should appear lower on the vertical axis, with the ocean surface (0 m) at the very top. Which command achieves this vertical axis orientation?

- [ ] **A.** `plt.ylim(0, 1000, reverse=True)`
- [ ] **B.** `plt.yaxis_reverse()`
- [ ] **C.** `plt.ylim(1000, 0)`
- [ ] **D.** `plt.plot(temperature, -depth)` followed by manual string relabeling

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plt.ylim()` does not accept a `reverse` keyword parameter.
- **Option B (Incorrect):** There is no function named `plt.yaxis_reverse()` in Matplotlib.
- **Option C (Correct):** Passing boundaries in reverse order to `plt.ylim(bottom, top)`—specifically `plt.ylim(1000, 0)`—or calling `plt.gca().invert_yaxis()` inverts the vertical direction so that 0 is placed at the top and 1000 at the bottom.
- **Option D (Incorrect):** Negating the depth data requires tedious manual compensation of ticks and labels; inverting the axis limits directly via `plt.ylim(1000, 0)` is the idiomatic solution.

</details>

---

### 16. Question

When comparing `plt.plot(x, y, 'o')` with `plt.scatter(x, y)` for plotting a discrete set of 10,000 points without connecting lines, which statement accurately captures their structural difference?

- [ ] **A.** `plt.plot(x, y, 'o')` creates a single `Line2D` artist whose markers are drawn at vertices, making it significantly faster than `plt.scatter()`, which creates a `PathCollection` where each point can have individual sizes and colors.
- [ ] **B.** `plt.scatter()` is strictly a wrapper around `plt.plot()` that offers no additional functionality.
- [ ] **C.** `plt.plot(x, y, 'o')` requires 10 times more memory because it renders invisible connecting lines between all points.
- [ ] **D.** `plt.scatter()` cannot plot circular points, whereas `plt.plot()` can.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.plot(x, y, 'o')` instantiates a single `Line2D` object with uniform marker styling across all points. In contrast, `plt.scatter()` creates a `PathCollection` designed to map per-point sizes and colormaps. Because of this added per-point flexibility, `scatter()` incurs substantial overhead compared to `plot()` when rendering large datasets.
- **Option B (Incorrect):** `plt.scatter()` is a distinct artist constructor (`PathCollection`) supporting arrays of individual colors and sizes, not a mere alias of `plot()`.
- **Option C (Incorrect):** Setting `linestyle='none'` (or omitting line tokens in `fmt`) tells the backend not to construct line segments, making `plot()` highly memory-efficient.
- **Option D (Incorrect):** Both functions support circular points and virtually identical marker shapes.

</details>

---

### 17. Question

A software engineer executes the following line plot command without supplying an explicit $x$ sequence:

```python
plt.plot([12, 24, 36, 48, 60])
```

What are the exact $(x, y)$ coordinates of the **fourth** plotted data point?

- [ ] **A.** $(4, 48)$
- [ ] **B.** $(3, 48)$
- [ ] **C.** $(4, 60)$
- [ ] **D.** $(3, 36)$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The index 4 corresponds to the fifth element ($60$), not the fourth element.
- **Option B (Correct):** In zero-indexed ordering, the elements correspond to:
  - 1st point: index $0 \rightarrow (0, 12)$
  - 2nd point: index $1 \rightarrow (1, 24)$
  - 3rd point: index $2 \rightarrow (2, 36)$
  - 4th point: index $3 \rightarrow (3, 48)$
  Therefore, the fourth point sits at $(3, 48)$.
- **Option C (Incorrect):** $(4, 60)$ is the fifth (final) point in the sequence.
- **Option D (Incorrect):** $(3, 36)$ incorrectly maps index 3 to the third element instead of the fourth.

</details>

---

### 18. Question

Which parameter controls the diameter/size of markers placed along a line trace in `plt.plot()`?

- [ ] **A.** `markerwidth`
- [ ] **B.** `markersize` (or `ms`)
- [ ] **C.** `point_radius`
- [ ] **D.** `scale`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `markerwidth` is not a valid parameter; outline thickness is `markeredgewidth`.
- **Option B (Correct):** In Matplotlib, marker diameter is configured using `markersize` or its shorthand abbreviation `ms`, measured in typographic points (e.g., `markersize=8`).
- **Option C (Incorrect):** `point_radius` is not a Matplotlib parameter.
- **Option D (Incorrect):** `scale` is not an argument of `plt.plot()`.

</details>

---

### 19. Question

A data analyst styles a line plot using the following call:

```python
plt.plot(x, y, marker='s', markersize=10, markeredgewidth=3, markeredgecolor='black', markerfacecolor='yellow')
```

Which statement accurately describes the visual appearance of each data vertex?

- [ ] **A.** A black circle with a 10-point yellow star inside
- [ ] **B.** A hollow triangular marker with yellow fill and no outline
- [ ] **C.** A 3-pixel diamond marker with transparent edges
- [ ] **D.** A yellow square with a thick, 3-point black border

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The marker shape is `'s'` (square), not a circle or star.
- **Option B (Incorrect):** The marker is square, not triangular, and has a defined outline.
- **Option C (Incorrect):** The marker is neither diamond-shaped nor transparent.
- **Option D (Correct):** Marker `'s'` defines a square shape. `markersize=10` establishes a prominent 10-point marker size. `markerfacecolor='yellow'` fills the interior of each square with yellow, and `markeredgecolor='black'` with `markeredgewidth=3` renders a bold 3-point black outline around each square.

</details>

---

### 20. Question

Consider the following statements regarding axes customization and marker properties in Matplotlib line plots:

1. `plt.xlabel()` and `plt.ylabel()` accept font formatting arguments such as `fontsize`, `fontweight`, and `color`.
2. `plt.xlim()` can accept tuples, lists, or two positional floats to establish horizontal limits.
3. Setting `markeredgewidth=0` completely removes the marker perimeter outline.
4. Setting `plt.grid(False)` will raise an exception if gridlines were not previously activated.

Which statements are **correct**?

- [ ] **A.** 1 and 4 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `plt.xlabel("Time (s)", fontsize=12, fontweight='bold', color='darkblue')` is completely valid and applies typography customization.
- **Statement 2 is correct:** `plt.xlim(0, 10)`, `plt.xlim((0, 10))`, and `plt.xlim([0, 10])` are all valid syntaxes accepted by Matplotlib.
- **Statement 3 is correct:** Setting `markeredgewidth=0` eliminates the border stroke around markers, rendering only the face fill.
- **Statement 4 is incorrect:** `plt.grid(False)` is an idempotent state toggle; if the grid is already off, it cleanly remains off without raising any error.
- **Conclusion:** Statements 1, 2, and 3 are correct, making **Option C** the correct answer.

</details>

---
