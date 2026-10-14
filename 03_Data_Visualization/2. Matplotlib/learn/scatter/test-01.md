# Matplotlib Scatter Plot Objective Test — Set 01

---

### 1. Question

A machine learning engineer needs to visualize the relationship between two continuous variables: `engine_displacement` (in liters) and `highway_mpg` (miles per gallon). Which Matplotlib function call creates a standard two-dimensional scatter plot where each observation is represented as an independent point in Cartesian space?

- [ ] **A.** `plt.scatter(engine_displacement, highway_mpg)`
- [ ] **B.** `plt.plot_scatter(engine_displacement, highway_mpg)`
- [ ] **C.** `plt.bivariate(engine_displacement, highway_mpg)`
- [ ] **D.** `plt.points(engine_displacement, highway_mpg)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, `plt.scatter(x, y)` is the primary function used to generate a 2D Cartesian scatter plot where individual observation pairs $(x_i, y_i)$ are plotted as discrete glyph markers.
- **Option B (Incorrect):** `plt.plot_scatter` does not exist in the Matplotlib API; calling it raises an `AttributeError`.
- **Option C (Incorrect):** `plt.bivariate` is not a valid Matplotlib function. While scatter plots display bivariate data, the API method is named `scatter`.
- **Option D (Incorrect):** `plt.points` is not a Matplotlib plotting function; points are drawn using `scatter()` or `plot(..., 'o')`.

</details>

---

### 2. Question

Consider the following two implementations for rendering 10,000 data points without varying colors or sizes:

```python
# Approach 1
plt.plot(x, y, "o")

# Approach 2
plt.scatter(x, y)
```

Which statement accurately contrasts the internal mechanics and performance of these two approaches?

- [ ] **A.** `plt.scatter()` is substantially faster because it compiles points into a GPU shader buffer, whereas `plt.plot()` iterates sequentially in Python bytecode.
- [ ] **B.** `plt.plot(x, y, 'o')` returns a `PathCollection`, whereas `plt.scatter(x, y)` returns a `Line2D` object.
- [ ] **C.** `plt.plot(x, y, 'o')` creates a single `Line2D` artist assuming uniform marker attributes, making it significantly faster for large static datasets than `plt.scatter()`, which builds a flexible `PathCollection` supporting individual marker sizing and coloring.
- [ ] **D.** Both approaches have identical runtime performance and memory overhead because `plt.scatter()` is merely an alias for `plt.plot(..., 'o')`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Standard Matplotlib does not execute GPU shaders for `plt.scatter()`. In fact, `plt.scatter()` has greater overhead than `plt.plot()` because it creates individual properties for vertex collections.
- **Option B (Incorrect):** The return types are reversed: `plt.plot()` returns a list of `Line2D` artists, while `plt.scatter()` returns a `PathCollection`.
- **Option C (Correct):** `plt.plot(x, y, 'o')` treats the markers as vertices of a single `Line2D` object where all points share identical size, color, and style, bypassing per-marker property buffers. Conversely, `plt.scatter()` creates a `PathCollection` designed to handle individual size, color, and alpha mappings per vertex, incurring higher computational and memory overhead.
- **Option D (Incorrect):** `plt.scatter()` is not an alias for `plt.plot()`; they instantiate completely different artist classes (`PathCollection` vs `Line2D`) with different capabilities.

</details>

---

### 3. Question

A data analyst executes the following code snippet and inspects the return type of `scatter_plot`:

```python
import matplotlib.pyplot as plt

hours = [1, 2, 3, 4, 5]
scores = [50, 55, 65, 70, 80]

scatter_plot = plt.scatter(hours, scores)
print(type(scatter_plot))
```

What Python class type is printed to the console?

- [ ] **A.** `<class 'matplotlib.lines.Line2D'>`
- [ ] **B.** `<class 'matplotlib.collections.PathCollection'>`
- [ ] **C.** `<class 'matplotlib.patches.Circle'>`
- [ ] **D.** `<class 'matplotlib.artist.ScatterGlyph'>`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `matplotlib.lines.Line2D` is the artist returned by line plot calls like `plt.plot()`, not `plt.scatter()`.
- **Option B (Correct):** `plt.scatter()` returns a `matplotlib.collections.PathCollection` instance. This collection artist groups the 2D geometric paths representing the markers and provides methods for colormapping, alpha blending, and colorbar linking.
- **Option C (Incorrect):** Although markers may appear as circles, Matplotlib does not create individual `Circle` patch objects for each point; it groups them in a vectorized `PathCollection`.
- **Option D (Incorrect):** There is no class named `ScatterGlyph` in the Matplotlib artist hierarchy.

</details>

---

### 4. Question

What happens when a developer executes the following script with mismatched coordinate array lengths?

```python
import matplotlib.pyplot as plt

x_coords = [10, 20, 30, 40, 50]
y_coords = [100, 150, 200]

plt.scatter(x_coords, y_coords)
plt.show()
```

- [ ] **A.** Matplotlib cycles `y_coords` cyclically (`[100, 150, 200, 100, 150]`) to match the length of `x_coords`.
- [ ] **B.** Matplotlib pads `y_coords` with `NaN` values and plots only the first 3 points without raising an exception.
- [ ] **C.** Matplotlib truncates `x_coords` to length 3 and plots the first 3 pairs.
- [ ] **D.** Matplotlib raises a `ValueError: x and y must be the same size`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not recycle or repeat values across dimensional coordinate vectors $x$ and $y$.
- **Option B (Incorrect):** Matplotlib does not auto-pad unequal coordinate arrays with `NaN` or `None`.
- **Option C (Incorrect):** Matplotlib does not silently truncate arrays of differing lengths; strict 1-to-1 coordinate correspondence is enforced.
- **Option D (Correct):** In Matplotlib, coordinate arrays passed to `plt.scatter(x, y)` must have identical shape and length. If `len(x) != len(y)`, NumPy vector validation inside Matplotlib triggers `ValueError: x and y must be the same size`.

</details>

---

### 5. Question

A researcher wants to plot experimental measurements where points are shaped as five-pointed stars instead of standard circular dots. Which parameter specification in `plt.scatter()` achieves this?

- [ ] **A.** `plt.scatter(x, y, marker="*")`
- [ ] **B.** `plt.scatter(x, y, shape="star")`
- [ ] **C.** `plt.scatter(x, y, glyph="pentagram")`
- [ ] **D.** `plt.scatter(x, y, style="star5")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The `marker` parameter accepts standard Matplotlib marker shorthand strings, where `'*'` denotes a five-pointed star glyph. Other common markers include `'o'` (circle), `'s'` (square), `'^'` (triangle up), and `'x'` (cross).
- **Option B (Incorrect):** There is no parameter named `shape` in `plt.scatter()`. Supplying `shape="star"` causes a `TypeError: scatter() got an unexpected keyword argument 'shape'`.
- **Option C (Incorrect):** `glyph` is not a valid parameter name in Matplotlib.
- **Option D (Incorrect):** `style` is not a parameter for controlling marker geometry in `plt.scatter()`.

</details>

---

### 6. Question

Examine the following code:

```python
import matplotlib.pyplot as plt

age = [20, 25, 30, 35, 40]
salary = [25000, 35000, 45000, 60000, 75000]

plt.scatter(age, salary, color="r", marker="s")
plt.show()
```

What visual rendering is produced on the plot?

- [ ] **A.** Blue circular dots connected by a solid red line.
- [ ] **B.** Red square markers plotted at the specified $(age, salary)$ Cartesian coordinates without connecting lines.
- [ ] **C.** Red circular dots connected by dashed line segments.
- [ ] **D.** Square markers with red edges and transparent fills connected by step functions.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.scatter()` never draws connecting lines between points, and `marker="s"` specifies squares, not circular dots.
- **Option B (Correct):** `color="r"` sets the uniform marker color to red, `marker="s"` renders each point as a filled square glyph, and `plt.scatter()` places discrete points at $(x_i, y_i)$ without any connecting line segments.
- **Option C (Incorrect):** `marker="s"` generates squares, not circles, and no dashed lines are rendered by `plt.scatter()`.
- **Option D (Incorrect):** Standard `color="r"` sets both face and edge colors to solid red by default; it does not leave the fill transparent.

</details>

---

### 7. Question

In Matplotlib's `plt.scatter(x, y, s=100)`, what does the numerical value `100` passed to `s` represent?

- [ ] **A.** The diameter of each circular marker in screen pixels ($px$).
- [ ] **B.** The radius of each circular marker in points ($pt$).
- [ ] **C.** The area of each marker in points squared ($pt^2$).
- [ ] **D.** The circumference of each circular marker in millimeters ($mm$).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `s` does not represent diameter in pixels. In typography and Matplotlib graphics, $s$ is specified in typographic points squared ($\text{pt}^2$).
- **Option B (Incorrect):** `s` does not scale linearly with marker radius. If `s` represented radius, doubling `s` would quadruple the visual area, which causes severe perceptual distortion in bubble charts.
- **Option C (Correct):** In Matplotlib, the marker size parameter `s` defines the marker area in typographic points squared ($\text{pt}^2$, where $1\text{ pt} = \frac{1}{72}\text{ inch}$). This proportional area scaling ensures that visual weight accurately reflects numeric values.
- **Option D (Incorrect):** Marker size in Matplotlib is never parameterized by circumference in millimeters.

</details>

---

### 8. Question

A data analyst wants to construct a fully labeled scatter plot with axis descriptions and a main header. Which combination of Matplotlib commands correctly labels the plot?

```python
import matplotlib.pyplot as plt

plt.scatter([1, 2, 3], [10, 20, 30])
# [Labeling lines]
plt.show()
```

- [ ] **A.** `plt.xtext("Study Hours"); plt.ytext("Exam Score"); plt.header("Score Analysis")`
- [ ] **B.** `plt.set_axis_name('x', "Study Hours"); plt.set_axis_name('y', "Exam Score")`
- [ ] **C.** `plt.axis_labels("Study Hours", "Exam Score"); plt.caption("Score Analysis")`
- [ ] **D.** `plt.xlabel("Study Hours"); plt.ylabel("Exam Score"); plt.title("Score Analysis")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.xtext`, `plt.ytext`, and `plt.header` do not exist in Matplotlib.
- **Option B (Incorrect):** `plt.set_axis_name` is not a standard Matplotlib pyplot method.
- **Option C (Incorrect):** `plt.axis_labels` and `plt.caption` are not valid Matplotlib functions.
- **Option D (Correct):** The standard pyplot interface provides `plt.xlabel()` to set the horizontal axis label, `plt.ylabel()` to set the vertical axis label, and `plt.title()` to specify the figure title.

</details>

---

### 9. Question

A dataset consists of the following 5 sample pairs:

$$\{(1, 50), (2, 55), (3, 65), (4, 70), (5, 80)\}$$

When plotted via `plt.scatter(x, y)`, what underlying bivariate statistical pattern is visually evident from bottom-left to top-right?

- [ ] **A.** A strong positive monotonic / linear correlation where $y$ increases systematically as $x$ increases.
- [ ] **B.** A strong negative linear correlation where $y$ decreases systematically as $x$ increases.
- [ ] **C.** Perfect heteroscedasticity with zero global directional trend.
- [ ] **D.** A sinusoidal periodic oscillation with alternating crests and troughs.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** As $x$ increases from $1$ to $5$, the $y$ values rise strictly from $50$ to $80$ ($50 < 55 < 65 < 70 < 80$). In a 2D Cartesian scatter plot, points that progress upward from bottom-left to top-right demonstrate a positive linear/monotonic correlation.
- **Option B (Incorrect):** A negative linear correlation requires points to trend downward from top-left to bottom-right ($y$ decreases as $x$ increases).
- **Option C (Incorrect):** The points follow a clear upward diagonal trajectory rather than a zero-trend fan of expanding variance.
- **Option D (Incorrect):** The coordinate sequence is strictly increasing without periodic reversals or crests.

</details>

---

### 10. Question

A data scientist executes the following code but notices that no legend box appears on the rendered canvas:

```python
import matplotlib.pyplot as plt

x = [20, 25, 30, 35, 40]
y = [25000, 35000, 45000, 60000, 75000]

plt.scatter(x, y, s=100, marker="*", color="r", label="Engineers")
plt.show()
```

What is the root cause of the missing legend?

- [ ] **A.** `label="Engineers"` cannot be applied to scatter plots; legends only support line plots created with `plt.plot()`.
- [ ] **B.** The script omitted `plt.legend()`, which is required to instruct Matplotlib to construct and render the legend artist using registered labels.
- [ ] **C.** The marker `'*'` is incompatible with legend glyph compilation and silently suppresses the legend box.
- [ ] **D.** Scatter plots require the label parameter to be named `legend_label` rather than `label`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `PathCollection` objects returned by `plt.scatter()` fully support legend rendering via proxy artist handles.
- **Option B (Correct):** Supplying `label="Engineers"` registers the text string with the artist inside the current axes, but the legend box itself is not built until `plt.legend()` (or `ax.legend()`) is explicitly invoked.
- **Option C (Incorrect):** All standard marker glyphs, including `'*'`, `'o'`, and `'s'`, are supported in legend proxy representations.
- **Option D (Incorrect):** The keyword argument is `label`, not `legend_label`.

</details>

---
