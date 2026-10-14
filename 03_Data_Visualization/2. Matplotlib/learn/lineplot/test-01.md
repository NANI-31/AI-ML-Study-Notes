# Matplotlib Line Plot Objective Test — Set 01

---

### 1. Question

A machine learning practitioner is monitoring training loss across iterations and executes the following statement:

```python
import matplotlib.pyplot as plt

loss_values = [0.85, 0.62, 0.44, 0.31, 0.22]
plt.plot(loss_values)
plt.show()
```

Because only a single sequence was provided to `plt.plot()`, what values does Matplotlib automatically generate and assign to the horizontal ($x$) axis?

- [ ] **A.** Floating-point values normalized between $0.0$ and $1.0$ ($[0.0, 0.25, 0.5, 0.75, 1.0]$)
- [ ] **B.** One-indexed integers ($[1, 2, 3, 4, 5]$)
- [ ] **C.** Zero-indexed integers matching the array indices ($[0, 1, 2, 3, 4]$)
- [ ] **D.** An empty x-axis with tick marks suppressed until explicitly defined via `plt.xticks()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not normalize single-sequence inputs to the continuous interval $[0.0, 1.0]$. Normalization requires explicit mathematical scaling by the user.
- **Option B (Incorrect):** While humans often count starting from 1 (e.g., Epoch 1 to 5), Python and Matplotlib follow standard zero-based indexing rules; the generated range starts at 0, not 1.
- **Option C (Correct):** When `plt.plot()` receives only a single positional argument `y`, Matplotlib interprets those numbers as y-coordinates and automatically creates matching x-coordinates using `range(len(y))`, which evaluates to `[0, 1, 2, 3, 4]`.
- **Option D (Incorrect):** Matplotlib never leaves the x-axis empty or disables tick marks when plotting a single array; it automatically builds a linear numeric scale based on the array's zero-based element indices.

</details>

---

### 2. Question

An analytics engineer needs to trace weekly sensor readings using a compact format string shorthand (`fmt`) that renders a **red line**, with **circle markers** at each data vertex, and a **solid line** connecting them. Which `fmt` string achieves this?

- [ ] **A.** `'r.-'`
- [ ] **B.** `'red-circle'`
- [ ] **C.** `'rc--'`
- [ ] **D.** `'ro-'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The `'.'` character denotes a small "point" marker rather than a standard "circle" marker (`'o'`).
- **Option B (Incorrect):** Matplotlib format strings do not accept natural-language words like `'red-circle'`; passing an unrecognized multi-letter string raises a `ValueError`.
- **Option C (Incorrect):** In format shorthand, `'c'` stands for cyan (not circle), and `'--'` specifies a dashed line (not solid).
- **Option D (Correct):** In Matplotlib's `fmt = '[color][marker][line]'` shorthand syntax, `'r'` represents red, `'o'` represents a circle marker, and `'-'` represents a solid line. Combining them as `'ro-'` produces red circle markers joined by a solid line.

</details>

---

### 3. Question

A junior developer wants to emphasize a primary benchmark trajectory by increasing its stroke width. They write:

```python
plt.plot(x, benchmark_y, linewidth=4.0)
```

Which of the following statements correctly describes the behavior of the `linewidth` parameter in Matplotlib?

- [ ] **A.** Setting `linewidth=4.0` specifies the line thickness in printer points (pt), where larger numbers produce heavier, thicker strokes.
- [ ] **B.** Values greater than 1.0 invert stroke visibility, converting the line into an ultra-thin hairline.
- [ ] **C.** The `linewidth` parameter only accepts integer values between 1 and 3; passing `4.0` triggers an `InvalidLineWidthError`.
- [ ] **D.** Line width is measured as a percentage of total canvas width; `linewidth=4.0` renders a stroke occupying 40% of the figure window.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, `linewidth` (or its shorthand alias `lw`) accepts a float value measured in typographic points (points = 1/72 inch). Increasing the number increases the visual thickness of the rendered line trace.
- **Option B (Incorrect):** This reflects a common misconception; higher values make lines thicker, whereas values approaching 0 make them hairline-thin and eventually invisible.
- **Option C (Incorrect):** `linewidth` accepts arbitrary positive floating-point values (e.g., `0.5`, `1.75`, `4.0`, `12.0`).
- **Option D (Incorrect):** `linewidth` is defined in absolute points, not as a percentage of axes or figure dimensions.

</details>

---

### 4. Question

Which of the following Matplotlib format strings correctly renders a **black dashed line** with **no markers**?

- [ ] **A.** `'b--'`
- [ ] **B.** `'k:'`
- [ ] **C.** `'k--'`
- [ ] **D.** `'w--'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** In Matplotlib single-letter color codes, `'b'` represents **blue**, not black.
- **Option B (Incorrect):** While `'k'` correctly designates black (from the "Key" in CMYK printing), the `':'` token produces a dotted line rather than a dashed line.
- **Option C (Correct):** `'k'` is the color code for black and `'--'` is the symbol for a dashed line. Omitting any marker symbol renders the dashed line without vertex points.
- **Option D (Incorrect):** `'w'` represents white, which would render an invisible white line against a standard white canvas.

</details>

---

### 5. Question

An engineer inspects the return value of a line plot call in an interactive Python shell:

```python
result = plt.plot([1, 2, 3], [10, 20, 30])
```

What Python type is stored in `result`?

- [ ] **A.** A single `matplotlib.figure.Figure` instance
- [ ] **B.** A Python `list` containing one `matplotlib.lines.Line2D` object
- [ ] **C.** A 2D NumPy array containing the interpolated pixel coordinates of the line
- [ ] **D.** `None`, because `plt.plot()` executes purely via side effects on the current axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Figures are created via `plt.figure()` or `plt.subplots()`; `plt.plot()` operates on the current axes (`Axes`) and returns line artists.
- **Option B (Correct):** `plt.plot()` always returns a `list` of `matplotlib.lines.Line2D` artist objects representing the line(s) added to the axes. Because one pair of sequences was plotted, the list contains exactly one `Line2D` object (e.g., `[<matplotlib.lines.Line2D object at 0x...>]`).
- **Option C (Incorrect):** Matplotlib does not return pixel coordinate matrices from `plot()`; it instantiates artist objects that the rendering backend draws later.
- **Option D (Incorrect):** `plt.plot()` does have side effects (modifying the current active `Axes`), but it explicitly returns the list of created line artists so users can customize them programmatically.

</details>

---

### 6. Question

A data analyst needs to plot a signal trajectory such that only the discrete observation points are shown as **green upward-pointing triangles**, with **no connecting line** between them. Which format string or keyword configuration achieves this?

- [ ] **A.** `plt.plot(x, y, 'g^', linestyle='none')`
- [ ] **B.** `plt.plot(x, y, 'g-^')`
- [ ] **C.** `plt.plot(x, y, 'v--', color='green')`
- [ ] **D.** `plt.plot(x, y, 'gt')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `'g^'` specifies green color (`'g'`) and upward-pointing triangle markers (`'^'`). Because no line token is included (or explicitly suppressed with `linestyle='none'`), Matplotlib renders only the triangle markers at each data coordinate without drawing connecting segments.
- **Option B (Incorrect):** The `'-'` token explicitly specifies a solid line, so `'g-^'` draws both the triangles and the connecting solid line.
- **Option C (Incorrect):** `'v'` denotes downward-pointing triangles, and `'--'` explicitly draws a dashed connecting line.
- **Option D (Incorrect):** `'t'` is not a valid marker symbol in Matplotlib; upward triangle is `'^'`.

</details>

---

### 7. Question

Consider the following three format strings passed to `plt.plot()`:

```python
plt.plot(x, y, 'ro--')  # Call 1
plt.plot(x, y, 'r--o')  # Call 2
plt.plot(x, y, '--or')  # Call 3
```

Which statement accurately describes how Matplotlib parses these three strings?

- [ ] **A.** Only Call 1 is valid; Calls 2 and 3 raise a `ValueError` because format tokens must strictly follow the `[color][marker][line]` sequence.
- [ ] **B.** Call 1 and Call 2 succeed, but Call 3 fails because line style tokens cannot appear at the start of a format string.
- [ ] **C.** All three calls are completely valid and render visually identical plots because Matplotlib parses color, marker, and linestyle components independently regardless of order.
- [ ] **D.** Call 1 creates a red dashed line with circles, Call 2 creates an orange line with red circles, and Call 3 creates a double-dashed black line.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib's format string parser is order-independent; the color, marker, and line style characters can appear in any permutation.
- **Option B (Incorrect):** Line style tokens such as `'--'`, `'-.'`, or `':'` can appear at the beginning, middle, or end of the format string without syntax errors.
- **Option C (Correct):** Matplotlib's internal `_process_plot_format` helper uses regex scanning to identify the color (`'r'`), marker (`'o'`), and linestyle (`'--'`) tokens independently. Consequently, `'ro--'`, `'r--o'`, and `'--or'` parse to the exact same visual styling: red dashed line with circular markers.
- **Option D (Incorrect):** The letters do not represent different secondary colors based on positioning; `'r'` always means red, `'o'` always means circle, and `'--'` always means dashed.

</details>

---

### 8. Question

A software engineer writes the following code containing a conflicting color specification:

```python
plt.plot(x, y, 'ro-', color='deepskyblue')
```

What visual color will the rendered line and markers have?

- [ ] **A.** Red, because the compact format string takes precedence over keyword arguments
- [ ] **B.** Purple, because Matplotlib blends red and blue into a hybrid composite color
- [ ] **C.** An exception is raised: `MatplotlibDeprecationError: Conflicting color specifications`
- [ ] **D.** Deep sky blue, because explicit keyword arguments override format string parameters

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Shorthand format strings do not take precedence over explicit keyword arguments.
- **Option B (Incorrect):** Matplotlib does not blend conflicting color strings; it resolves precedence deterministically.
- **Option C (Incorrect):** Supplying both a `fmt` color and a `color` keyword argument does not raise an exception; Matplotlib simply overrides the shorthand color with the explicit keyword.
- **Option D (Correct):** In Matplotlib, explicit keyword arguments (e.g., `color='deepskyblue'`, `linewidth=2`) take precedence over values specified in the `fmt` shorthand string. The line and its markers will be rendered in deep sky blue.

</details>

---

### 9. Question

A scientist wants to plot 50 overlapping Monte Carlo simulation paths on a single axes. If each line is rendered at default full opacity, the overlapping strokes create a solid block of color that obscures path density. Which parameter should the scientist adjust to reveal where trajectories cluster?

- [ ] **A.** `zorder=0.1`
- [ ] **B.** `alpha=0.15`
- [ ] **C.** `linewidth=-2`
- [ ] **D.** `density=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `zorder` controls the vertical drawing order (layering hierarchy) of artists, not their transparency.
- **Option B (Correct):** The `alpha` parameter controls transparency on a scale from `0.0` (completely transparent/invisible) to `1.0` (completely opaque). Setting `alpha=0.15` makes each path semi-transparent, allowing overlapping lines to accumulate optical density and visually highlight high-probability trajectories.
- **Option C (Incorrect):** `linewidth` must be non-negative; negative line widths raise a `ValueError`.
- **Option D (Incorrect):** `density` is a parameter of `plt.hist()`, not `plt.plot()`. Passing it to `plot()` raises an `AttributeError` or `TypeError`.

</details>

---

### 10. Question

Review the following four statements regarding line plot styles and format strings in Matplotlib. Which statements are **correct**?

1. The line style code `'-.'` produces a dash-dot line pattern.
2. The color code `'k'` represents black, whereas `'b'` represents blue.
3. The format string `'m*:'` produces a magenta line with star markers and a dotted stroke.
4. Setting `linewidth=0` deletes the artist object from the axes canvas completely.

- [ ] **A.** 1 and 4 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Statement 4 is false.
- **Option B (Incorrect):** Statement 1 is also correct.
- **Option C (Correct):**
  - **Statement 1 is correct:** `'-.'` is the standard Matplotlib symbol for a dash-dot line style.
  - **Statement 2 is correct:** In Matplotlib's single-letter palette, `'k'` stands for black (from printing terminology) and `'b'` stands for blue.
  - **Statement 3 is correct:** In `'m*:'`, `'m'` is magenta, `'*'` is the star marker, and `':'` specifies a dotted line.
  - **Statement 4 is incorrect:** Setting `linewidth=0` renders the line stroke with zero thickness (making it visually invisible), but the `Line2D` artist object still exists in the axes artist list (`ax.lines`) and its vertices are still processed during layout and autoscaling.
- **Option D (Incorrect):** Statement 4 is false.

</details>

---
