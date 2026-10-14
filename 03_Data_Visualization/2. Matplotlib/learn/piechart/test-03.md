# Matplotlib Pie Chart Objective Test — Set 03

---

### 21. Question

A data analyst wants to draw visual attention to the "Security Breaches" category (the first of four categories) by pulling its wedge slightly outward from the center of the pie. Which parameter passed to `plt.pie()` achieves this?

- [ ] **A.** `offset=[10, 0, 0, 0]`
- [ ] **B.** `explode=(0.1, 0, 0, 0)`
- [ ] **C.** `detach={'Security Breaches': 0.1}`
- [ ] **D.** `pull=(0.25, 0, 0, 0)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `offset` is not a valid parameter of `plt.pie()`, and absolute pixels like 10 are not used.
- **Option B (Correct):** In Matplotlib, slice separation is configured via the `explode` parameter. It accepts a sequence (tuple or list) matching the length of the data array, where each element specifies the radial offset as a fraction of the circle radius. Passing `explode=(0.1, 0, 0, 0)` offsets the first slice outward by 10% of the radius while leaving the other three slices joined at the center.
- **Option C (Incorrect):** `detach` is not a recognized Matplotlib keyword argument.
- **Option D (Incorrect):** `pull` is not a valid parameter.

</details>

---

### 22. Question

A machine learning engineer writes the following code to emphasize two classes:

```python
classes = [70, 15, 10, 5]
explode = (0.1, 0.05)
plt.pie(classes, explode=explode)
plt.show()
```

What occurs when this script is executed?

- [ ] **A.** Matplotlib raises a `ValueError: len(explode) == len(x)`.
- [ ] **B.** Matplotlib assumes the remaining two slices have an explode offset of `0.0`.
- [ ] **C.** The chart renders with only the first two slices.
- [ ] **D.** The second slice is exploded by the sum of the two numbers ($0.15$).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Matplotlib strictly requires that if `explode` is provided, its length must exactly equal the number of wedges being plotted ($N = \text{len}(x)$). Because `classes` has 4 elements and `explode` has only 2 elements, Matplotlib raises a `ValueError` indicating that the lengths do not match.
- **Option B (Incorrect):** Matplotlib does not automatically pad missing explode elements with zeros.
- **Option C (Incorrect):** Data slices are not omitted based on explode length.
- **Option D (Incorrect):** Offsets are mapped index-by-index, not accumulated.

</details>

---

### 23. Question

Along which mathematical trajectory does an exploded pie slice move outward from the chart origin $(0, 0)$?

- [ ] **A.** Strictly along the horizontal x-axis
- [ ] **B.** Strictly along the vertical y-axis
- [ ] **C.** Along the radial bisector line passing through the midpoint angle $\frac{\theta_{\text{start}} + \theta_{\text{end}}}{2}$ of that wedge
- [ ] **D.** Vertically upward toward the 12 o'clock position

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Slices only move horizontally if their bisector angle happens to be $0^\circ$ or $180^\circ$.
- **Option B (Incorrect):** Slices only move vertically if their bisector angle happens to be $90^\circ$ or $270^\circ$.
- **Option C (Correct):** In polar coordinates, an exploded wedge is translated radially outward along its angle of symmetry (the angular bisector halfway between the wedge's starting angle $\theta_{\text{start}}$ and ending angle $\theta_{\text{end}}$). This ensures the slice separates symmetrically from both adjacent wedges.
- **Option D (Incorrect):** Slices do not all move toward 12 o'clock; each moves outward along its own individual radial trajectory.

</details>

---

### 24. Question

What is a major visual layout defect that often occurs when specifying an excessively large explode value, such as `explode=(0.5, 0, 0, 0)`?

- [ ] **A.** The slice turns completely transparent.
- [ ] **B.** Matplotlib inverts the slice back into the center hole.
- [ ] **C.** The remaining three slices are automatically deleted.
- [ ] **D.** The exploded slice can be pushed outside the axes boundaries or collide with the chart title, legend, or canvas edges.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Explode displacement alters spatial coordinates; it has no effect on alpha opacity.
- **Option B (Incorrect):** Positive explode values always move outward away from the origin.
- **Option C (Incorrect):** Other slices remain rendered at their respective positions.
- **Option D (Correct):** `explode=0.5` shifts the wedge outward by 50% of the entire radius. Because standard axes limits are set tightly around the circle ($[-1.1, 1.1]$), a large displacement often pushes the wedge past the figure boundary, truncating it or causing collisions with the title and legend. Modest offsets ($0.05$ to $0.15$) are standard practice.

</details>

---

### 25. Question

A developer adds `shadow=True` to `plt.pie()`. Why do modern data visualization guidelines and information design principles strongly discourage the use of drop shadows on pie charts?

- [ ] **A.** Shadows add non-data ink, create optical illusions of 3D depth, and blur the precise geometric boundaries of the wedges, impairing angular judgment.
- [ ] **B.** Enabling shadows increases image file size by more than 50 megabytes.
- [ ] **C.** Shadows prevent Matplotlib from calculating `autopct` percentages.
- [ ] **D.** The `shadow` parameter has been deprecated and removed in modern Python 3.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** As articulated by Edward Tufte and modern visualization research, drop shadows represent unnecessary "chartjunk." Shadows obscure clean wedge boundaries, introduce false depth cues, and distort the perceived surface area of wedges without communicating any additional information. Clean, flat 2D graphics maximize data-ink ratio and visual precision.
- **Option B (Incorrect):** Raster drop shadows add negligible file size overhead (typically a few kilobytes).
- **Option C (Incorrect):** `autopct` calculations are purely mathematical and execute independently of visual shadows.
- **Option D (Incorrect):** `shadow=True` remains supported in Matplotlib, though discouraged in publication guidelines.

</details>

---

### 26. Question

When a pie chart has 6 or 7 categories, direct radial labels along the circumference often collide with each other. Which approach cleanly decouples the labels from the circle perimeter using a docked legend?

- [ ] **A.**
```python
wedges, texts, autotexts = ax.pie(sizes, autopct='%1.0f%%')
ax.legend(wedges, categories, loc='center left', bbox_to_anchor=(1, 0, 0.5))
```
- [ ] **B.** `ax.pie(sizes, labels=categories, legend='external')`
- [ ] **C.** `ax.dock_legend(categories, side='right')`
- [ ] **D.** `plt.pie(sizes, show_labels=False, legend_popup=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Omitting the `labels` argument from `ax.pie()` suppresses the cluttered perimeter text labels. Capturing the returned `wedges` list and passing `ax.legend(wedges, categories, loc='center left', bbox_to_anchor=(1, 0, 0.5))` creates a clean, vertical legend docked neatly outside the right border of the circular chart.
- **Option B (Incorrect):** `legend='external'` is not a valid parameter of `ax.pie()`.
- **Option C (Incorrect):** `dock_legend` is not a Matplotlib method.
- **Option D (Incorrect):** `show_labels` and `legend_popup` do not exist in Matplotlib.

</details>

---

### 27. Question

What is the visual effect of setting `rotatelabels=True` in `plt.pie()`?

- [ ] **A.** It animates the pie chart so it spins continuously on screen.
- [ ] **B.** It rotates each category text label so that it aligns radially with the orientation of its respective wedge.
- [ ] **C.** It rotates the legend box upside down.
- [ ] **D.** It randomizes text rotation angles between $-90^\circ$ and $+90^\circ$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `rotatelabels` affects static text orientation; it does not produce animation.
- **Option B (Correct):** When `rotatelabels=True`, Matplotlib computes the angle of each wedge's radial centerline and rotates the corresponding label text to match that angle. This orients the labels radiating outward like spokes of a wheel, which can help fit labels around compact circular charts.
- **Option C (Incorrect):** It only affects the text labels generated for wedges, not the legend box.
- **Option D (Incorrect):** Text rotation is mathematically aligned with the wedge bisector, never random.

</details>

---

### 28. Question

A data analyst notices that the title of a pie chart is partially overlapping the top wedge (which starts at 12 o'clock). How can the title be moved upward to provide adequate clearance?

- [ ] **A.** Call `plt.title("Title", pad=25)` to increase vertical padding between the axes and the title text
- [ ] **B.** Call `plt.pie(..., title_offset=10)`
- [ ] **C.** Call `plt.title("Title", vshift=50)`
- [ ] **D.** Decrease the font size of the title to 2 points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.title()` accepts the `pad` argument (in typographic points). Setting `pad=25` (or higher) moves the title baseline higher above the top spine of the axes, preventing visual collisions with top-positioned wedges or upward-exploded slices.
- **Option B (Incorrect):** `plt.pie()` does not handle title positioning.
- **Option C (Incorrect):** `vshift` is not a parameter of `plt.title()`.
- **Option D (Incorrect):** Shrinking the font size to 2 points makes the title unreadable and does not solve the spatial padding issue.

</details>

---

### 29. Question

Why do standard pie charts created via `plt.pie()` not display rectangular bounding box spines, x-axis tick marks, or y-axis coordinate numbers by default?

- [ ] **A.** Because Matplotlib crashes if tick marks are enabled for circular data.
- [ ] **B.** Because the operating system window manager disables coordinate rendering.
- [ ] **C.** Because `plt.pie()` internally disables axes ticks and hides the rectangular spines (`ax.set_frame_on(False)`) to display a clean circular canvas.
- [ ] **D.** Because pie charts are rendered on a 3D surface rather than a 2D Cartesian plane.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Axes ticks can be manually re-enabled without crashing, though they are visually inappropriate.
- **Option B (Incorrect):** Window managers do not dictate Matplotlib artist visibility.
- **Option C (Correct):** In Cartesian space, axes spines and numerical ticks belong to linear rectangular grids. Because pie charts represent polar proportions where Cartesian coordinates $(x, y)$ have no meaningful user interpretation, `plt.pie()` automatically executes `ax.set_frame_on(False)` and turns off tick marks to ensure an uncluttered presentation.
- **Option D (Incorrect):** Standard Matplotlib pie charts are purely 2D artists rendered on standard axes.

</details>

---

### 30. Question

Consider the following statements regarding slice explosion and layout management in Matplotlib pie charts:

1. The `explode` tuple values are specified as fractions of the circle's radius.
2. Setting `explode=(0, 0, 0.1, 0)` offsets only the third slice.
3. If `explode` is passed as a list instead of a tuple, Matplotlib raises a `TypeError`.
4. `rotatelabels=True` aligns text radially with each slice's center angle.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** An explode value of `0.1` represents a physical displacement equal to 10% of the radius.
- **Statement 2 is correct:** Offsets correspond 1-to-1 with slice indices; `0.1` at index 2 displaces only the 3rd wedge.
- **Statement 3 is incorrect:** `explode` accepts any sequence type, including Python lists (`[0, 0, 0.1, 0]`) and NumPy arrays; it does not require a tuple strictly.
- **Statement 4 is correct:** `rotatelabels=True` aligns label text orientation with each wedge's radial vector.
- **Conclusion:** Statements 1, 2, and 4 are correct, making **Option C** the correct answer.

</details>

---
