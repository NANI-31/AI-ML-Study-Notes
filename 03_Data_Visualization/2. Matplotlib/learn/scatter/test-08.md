# Matplotlib Scatter Plot Objective Test — Set 08

---

### 71. Question

A data analyst identifies a severe outlier in an experimental dataset: observation $(6.0, 50.0)$, while all other observations have $y \le 15.0$.

Which technique is the standard Matplotlib approach to visually emphasize this specific point with a large golden star glyph?

- [ ] **A.** Modifying the global configuration `rcParams['scatter.outlier'] = (6.0, 50.0)`.
- [ ] **B.** Calling `plt.filter_point(6.0, 50.0, highlight=True)`.
- [ ] **C.** Overlaying a secondary `plt.scatter(6.0, 50.0, color='gold', s=250, marker='*', edgecolors='black', zorder=5)` call on top of the primary scatter plot.
- [ ] **D.** Invoking `plt.highlight(6.0, 50.0, shape='star')`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** There is no `rcParams` setting for designating outliers.
- **Option B (Incorrect):** `plt.filter_point` is not a valid Matplotlib function.
- **Option C (Correct):** In Matplotlib, visual emphasis or annotation of specific points is accomplished by layer composition: plotting the full dataset via the primary `plt.scatter()` call, and then issuing a secondary `plt.scatter()` targeting the specific coordinate $(6.0, 50.0)$ with distinctive attributes (e.g., larger size `s=250`, unique marker `marker='*'`, contrasting color `'gold'`, and a high `zorder`).
- **Option D (Incorrect):** `plt.highlight` does not exist in the Matplotlib API.

</details>

---

### 72. Question

Examine the syntax of Matplotlib's annotation method:

```python
plt.annotate(
    "Critical Threshold",
    xy=(target_x, target_y),
    xytext=(text_x, text_y),
    arrowprops=dict(arrowstyle="->"),
)
```

What is the distinct role of `xy` versus `xytext`?

- [ ] **A.** `xy` defines the text position, and `xytext` defines the origin of the coordinate axes.
- [ ] **B.** `xy` specifies the point of interest being annotated (where the arrowhead points), while `xytext` specifies the coordinate where the text label string is placed (the tail of the arrow).
- [ ] **C.** `xy` is the start of the arrow, and `xytext` is the font size.
- [ ] **D.** `xy` and `xytext` are completely interchangeable synonyms.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `xy` is the target point being pointed to, not the text position.
- **Option B (Correct):** In `plt.annotate(text, xy=..., xytext=...)`:
  - `xy`: The Cartesian coordinate $(x, y)$ of the target data feature being highlighted (the tip of the arrow points exactly here).
  - `xytext`: The Cartesian coordinate where the text string will be drawn. If an arrow is configured via `arrowprops`, the arrow begins at `xytext` and terminates at `xy`.
- **Option C (Incorrect):** `xytext` accepts coordinate tuples, not font sizes.
- **Option D (Incorrect):** Swapping `xy` and `xytext` reverses the arrow direction.

</details>

---

### 73. Question

A developer executes the following annotation command on a scatter plot:

```python
plt.annotate(
    "Possible Outlier",
    xy=(6.0, 50.0),
    xytext=(4.5, 55.0),
    arrowprops=dict(arrowstyle="->", color="crimson"),
)
```

Where does the tip of the crimson arrowhead point, and where is the text "Possible Outlier" rendered?

- [ ] **A.** The arrowhead tip points directly at $(6.0, 50.0)$, and the text label begins at $(4.5, 55.0)$.
- [ ] **B.** The arrowhead tip points at $(4.5, 55.0)$, and the text is rendered at $(6.0, 50.0)$.
- [ ] **C.** Both the text and the arrowhead are drawn at $(0, 0)$.
- [ ] **D.** The arrow points from $(0, 0)$ to $(6.0, 50.0)$, ignoring $(4.5, 55.0)$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Parameter `xy=(6.0, 50.0)` sets the target coordinate for the arrowhead tip. Parameter `xytext=(4.5, 55.0)` sets the placement anchor for the string label "Possible Outlier". The arrow originates from the text anchor and points toward the outlier at $(6.0, 50.0)$.
- **Option B (Incorrect):** This describes what would occur if `xy` and `xytext` were reversed.
- **Option C (Incorrect):** Coordinates do not default to $(0, 0)$ when explicit values are provided.
- **Option D (Incorrect):** The arrow does not ignore `xytext` or originate from $(0, 0)$.

</details>

---

### 74. Question

What visual mistake occurs if a programmer inadvertently swaps `xy` and `xytext`:

```python
# Intended: Annotate the outlier at (8, 95) with text positioned at (6, 110)
plt.annotate(
    "Peak", xy=(6, 110), xytext=(8, 95), arrowprops=dict(arrowstyle="->")
)
```

- [ ] **A.** Matplotlib raises a `CoordinateMismatchError`.
- [ ] **B.** The text "Peak" is drawn upside down.
- [ ] **C.** The arrow disappears completely from the figure.
- [ ] **D.** The text "Peak" is rendered directly on top of the data point at $(8, 95)$, while the arrow points backward away from the data point into empty space at $(6, 110)$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not raise an exception; coordinate swapping is syntactically valid but visually flawed.
- **Option B (Incorrect):** Swapping coordinates does not invert text rendering.
- **Option C (Incorrect):** The arrow remains visible; its endpoints are merely misplaced.
- **Option D (Correct):** Because `xy` defines the arrow tip destination and `xytext` defines the text label position, setting `xy=(6, 110)` causes the arrow to point into empty whitespace at $(6, 110)$, while `xytext=(8, 95)` stamps the text label right over the data point at $(8, 95)$, occluding the marker and confusing the viewer.

</details>

---

### 75. Question

Which dictionary passed to `arrowprops` creates a clean, dark, thin arrow with a standard arrowhead?

- [ ] **A.** `arrowprops=dict(type='standard_arrow', fill=True)`
- [ ] **B.** `arrowprops=dict(arrowstyle='->', color='black', lw=1.0)`
- [ ] **C.** `arrowprops=dict(shape='vector', head_size=5)`
- [ ] **D.** `arrowprops=dict(render='pointer', stroke='dark')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Key `type` is not a valid parameter for `arrowprops`.
- **Option B (Correct):** In Matplotlib, `arrowprops` accepts a dictionary configuring an `AnnotationArrow` or `FancyArrowPatch`. Setting `arrowstyle='->'` selects a simple line arrow with an open head, `color='black'` establishes the line and head color, and `lw=1.0` sets the stroke width. Other valid arrowstyles include `'-|>'`, `'fancy'`, and `'simple'`.
- **Option C (Incorrect):** Keys `shape` and `head_size` are not supported in `arrowprops`.
- **Option D (Incorrect):** Keys `render` and `stroke` are not valid arguments in Matplotlib.

</details>

---

### 76. Question

When annotating points in a scatter plot, why is specifying `textcoords="offset points"` combined with `xytext=(10, 10)` widely considered superior to specifying absolute data coordinates?

- [ ] **A.** It forces the text label to use 10-point font size.
- [ ] **B.** It automatically translates the label text into multiple languages.
- [ ] **C.** It positions the text label at a constant physical offset (10 points right, 10 points up) relative to the target data point, ensuring the label never shifts or collides with the marker if axis limits or data scales are subsequently zoomed or resized.
- [ ] **D.** It converts 2D Cartesian coordinates into polar angles.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Font size is configured via `fontsize=10`, not `xytext`.
- **Option B (Incorrect):** Matplotlib does not perform language translation.
- **Option C (Correct):** By default, `xytext` is interpreted in raw data coordinates. If axis scales change, the label might move drastically. By setting `textcoords="offset points"`, `xytext=(10, 10)` is interpreted as a physical displacement of 10 typographic points ($10/72$ inch) relative to `xy`. Regardless of how the axes are panned or zoomed, the label maintains a constant, neat buffer distance from the marker.
- **Option D (Incorrect):** Polar transformation is handled by projection systems, not `textcoords`.

</details>

---

### 77. Question

A data analyst has 5 cities with coordinates `x`, `y`, and names `cities = ["Tokyo", "Delhi", "Shanghai", "Sao Paulo", "Mexico City"]`.

Which loop correctly annotates every city next to its respective scatter point?

- [ ] **A.**
  ```python
  plt.scatter(x, y)
  for xi, yi, name in zip(x, y, cities):
    plt.annotate(
        name, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9
    )
```
- [ ] **B.**
  ```python
  plt.scatter(x, y)
  plt.annotate_all(cities, x, y)
  ```
- [ ] **C.**
  ```python
  plt.scatter(x, y, labels=cities)
  ```
- [ ] **D.**
  ```python
  for i in range(len(cities)):
    plt.text(cities[i])
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Iterating through the vectors using `zip(x, y, cities)` extracts the coordinate pair $(x_i, y_i)$ and string label `name` for each observation. Calling `plt.annotate(name, (xi, yi), xytext=(5, 5), textcoords="offset points")` places each label consistently 5 points above and to the right of its marker.
- **Option B (Incorrect):** `plt.annotate_all` is not a valid Matplotlib function.
- **Option C (Incorrect):** `plt.scatter()` does not take a `labels` parameter for automated point annotation; `label` is singular and assigns a name to the entire collection for the legend.
- **Option D (Incorrect):** `plt.text()` requires $(x, y)$ coordinate arguments: `plt.text(x, y, text)`.

</details>

---

### 78. Question

A machine learning engineer wants to programmatically identify and highlight the observation with the highest $y$-value in a dataset, without hardcoding numerical coordinates:

```python
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([10, 15, 12, 20, 18, 30])
```

Which code snippet dynamically locates the peak and marks it with an enlarged star?

- [ ] **A.**
  ```python
  plt.scatter(x.max(), y.max(), s=300, marker="*")
  ```
- [ ] **B.**
  ```python
  plt.highlight_top(y, s=300)
  ```
- [ ] **C.**
  ```python
  plt.scatter(x[-1], y[-1], s=300, marker="*")
  ```
- [ ] **D.**
  ```python
  idx = np.argmax(y)
  plt.scatter(x[idx], y[idx], s=300, marker="*", color="red", zorder=5)
  ```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Plotting `(x.max(), y.max())` pairs the maximum of $x$ with the maximum of $y$. If the maximum $y$ occurs at a different $x$ coordinate (e.g., in non-monotonic data), this plots an artificial "phantom" point where no actual observation exists.
- **Option B (Incorrect):** `plt.highlight_top` is not a Matplotlib method.
- **Option C (Incorrect):** `[-1]` selects the last element in the array, which is only the maximum if the array is already sorted by $y$.
- **Option D (Correct):** `idx = np.argmax(y)` returns the exact array index where the maximum $y$ value resides. Indexing both arrays via `x[idx]` and `y[idx]` guarantees that the marker is placed at the true coordinate pair corresponding to the peak observation.

</details>

---

### 79. Question

A bivariate dataset exhibits a strong positive linear relationship ($y \approx 2x$). Observation $P$ has coordinates $(x = 50, y = 10)$.

In univariate inspection:
- $x = 50$ is near the sample median of $X$.
- $y = 10$ is near the sample median of $Y$.

Why is observation $P$ nonetheless classified as a severe bivariate outlier on a scatter plot?

- [ ] **A.** Because Matplotlib flags any point whose coordinates are integers ending in zero.
- [ ] **B.** While both $x = 50$ and $y = 10$ are completely normal and typical within their respective 1D marginal distributions, their joint combination severely violates the strong positive correlation structure (at $x = 50$, the expected value is $\hat{y} \approx 100$, so $y = 10$ has a massive residual).
- [ ] **C.** Because the point lies in the third quadrant.
- [ ] **D.** Because its Euclidean distance from the origin is less than 100.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Coordinate values ending in zero have no special statistical meaning.
- **Option B (Correct):** This is the classic definition of a **bivariate outlier**. Marginal univariate inspection (e.g., standard boxplots or histograms of $X$ and $Y$ independently) will completely fail to detect point $P$ because both values lie near their respective medians. However, in 2D Cartesian space, where $Y$ is tightly coupled to $X$ ($y \approx 2x$), the point $(50, 10)$ lies far away from the regression line, exhibiting a massive residual $e = 10 - 100 = -90$.
- **Option C (Incorrect):** Being located in the first quadrant (both coordinates positive) has no bearing on outlier classification.
- **Option D (Incorrect):** Euclidean distance from the origin is not the metric used for bivariate regression outlier detection.

</details>

---

### 80. Question

Which of the following statements regarding point annotations in Matplotlib are **correct**?

1. If `arrowprops=None` (the default), `plt.annotate()` renders the text label at `xy` without drawing an arrow.
2. `plt.text(x, y, "Label")` can also place text at Cartesian coordinates, but lacks built-in arrow vector support.
3. The `bbox` parameter in `plt.annotate()` can draw a styled bounding box (e.g., rounded rectangle with background fill) behind the text label.
4. Setting `arrowstyle='fancy'` produces an arrow with a curved head and tapering tail.

- [ ] **A.** 1 and 2 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, 3, and 4
- [ ] **D.** 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** While Statements 1 and 2 are true, this option omits valid Statements 3 (bbox support) and 4 (fancy arrowstyle).
- **Option B (Incorrect):** While Statements 2 and 4 are true, this option omits valid Statements 1 and 3.
- **Option C (Correct):** Statements 1, 2, 3, and 4 are all correct: `arrowprops=None` omits the arrow; `plt.text()` is the coordinate text primitive lacking arrows; `bbox` styles background frames behind labels; and `'fancy'` is a recognized curved arrowhead style.
- **Option D (Incorrect):** While Statements 3 and 4 are true, this option fails to include valid Statements 1 and 2.

</details>

---
