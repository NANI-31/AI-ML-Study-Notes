# Matplotlib Scatter Plot Objective Test — Set 02

---

### 11. Question

A data analyst notices that circular scatter markers blend together and lose their individual boundaries in moderately dense clusters. Which parameter combination adds a crisp dark perimeter around each marker?

- [ ] **A.** `plt.scatter(x, y, border_color="black", border_width=1.0)`
- [ ] **B.** `plt.scatter(x, y, outline="black", stroke=1.0)`
- [ ] **C.** `plt.scatter(x, y, edgecolors="black", linewidths=0.8)`
- [ ] **D.** `plt.scatter(x, y, perimeter="black", thickness=0.8)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `border_color` and `border_width` are not valid parameters in Matplotlib.
- **Option B (Incorrect):** `outline` and `stroke` are terms from CSS/SVG, not Matplotlib's `PathCollection` API.
- **Option C (Correct):** In Matplotlib, `edgecolors` controls the color of the marker boundary (e.g., `"black"` or `"gray"`), and `linewidths` defines the border line thickness in points. Applying `edgecolors="black"` and `linewidths=0.8` visually separates adjacent overlapping points.
- **Option D (Incorrect):** `perimeter` and `thickness` are not recognized keyword arguments in `plt.scatter()`.

</details>

---

### 12. Question

A financial analyst plots stock volatility against quarterly returns and wants subtle dotted reference gridlines behind the scatter points. Which command correctly configures this grid?

- [ ] **A.** `plt.background_grid(type="dots", opacity=0.5)`
- [ ] **B.** `plt.mesh(style="dotted", alpha=0.5)`
- [ ] **C.** `plt.add_gridlines(dash=":", transparency=0.5)`
- [ ] **D.** `plt.grid(True, linestyle=":", alpha=0.6)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.background_grid` is not a Matplotlib function.
- **Option B (Incorrect):** `plt.mesh` is not a valid grid-enabling command.
- **Option C (Incorrect):** `plt.add_gridlines` is not part of the pyplot module.
- **Option D (Correct):** `plt.grid(True)` activates gridlines on major tick marks, `linestyle=":"` sets the line pattern to dotted, and `alpha=0.6` softens their visual prominence so data markers remain the primary visual focus.

</details>

---

### 13. Question

By default, Matplotlib calculates coordinate boundaries automatically with a small margin around data extremes. What function call explicitly overrides this behavior to clamp the horizontal axis between 0 and 100?

- [ ] **A.** `plt.xlim(0, 100)`
- [ ] **B.** `plt.set_xrange(0, 100)`
- [ ] **C.** `plt.xbound(min=0, max=100)`
- [ ] **D.** `plt.axis_horizontal(0, 100)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.xlim(xmin, xmax)` (or `ax.set_xlim(left, right)`) sets the lower and upper data limits of the current x-axis, overriding the default auto-scaling margin.
- **Option B (Incorrect):** `plt.set_xrange` is not a valid Matplotlib method.
- **Option C (Incorrect):** While `ax.set_xbound()` exists on `Axes` objects, `plt.xbound` is not the standard pyplot convention and does not take keyword arguments `min`/`max`.
- **Option D (Incorrect):** `plt.axis_horizontal` does not exist in Matplotlib.

</details>

---

### 14. Question

A study records vehicle curb weight (in pounds) and fuel economy (miles per gallon):

$$\{(2000, 38), (2500, 32), (3200, 26), (4000, 19), (4800, 14)\}$$

When plotted via `plt.scatter(weight, mpg)`, which statistical relationship is displayed?

- [ ] **A.** Strong positive linear correlation with an upward trajectory from $(0, 0)$.
- [ ] **B.** Strong negative linear correlation where fuel economy decreases monotonically as vehicle weight increases.
- [ ] **C.** Quadratic U-shaped relationship with a distinct minimum at 3200 lbs.
- [ ] **D.** Homoscedastic random noise with Pearson correlation coefficient $r = 0.0$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Fuel economy decreases as weight increases; it does not rise from $(0, 0)$.
- **Option B (Correct):** As vehicle weight increases ($2000 \to 2500 \to 3200 \to 4000 \to 4800$), fuel efficiency drops systematically ($38 \to 32 \to 26 \to 19 \to 14$). On a Cartesian coordinate grid, points sloping downward from the top-left to bottom-right indicate a strong negative correlation ($r \approx -1.0$).
- **Option C (Incorrect):** The curve is strictly downward-sloping without any upward turnaround or U-shaped inflection.
- **Option D (Incorrect):** The data exhibits an almost perfectly deterministic linear inverse relationship, not random noise.

</details>

---

### 15. Question

A data engineer generates two independent standard normal vectors using NumPy and plots them:

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.randn(1000)
y = np.random.randn(1000)

plt.scatter(x, y)
plt.show()
```

What visual distribution and correlation structure does the resulting scatter plot exhibit?

- [ ] **A.** A sharp diagonal line with slope $m = 1.0$ and Pearson $r = 1.0$.
- [ ] **B.** An elliptical cluster oriented along the 45-degree line indicating strong covariance.
- [ ] **C.** A circular, spherically symmetric cloud centered at $(0, 0)$ indicating zero linear correlation ($r \approx 0$).
- [ ] **D.** A funnel-shaped dispersion widening exponentially from left to right.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Independent random vectors do not collapse onto a deterministic diagonal line.
- **Option B (Incorrect):** An elliptical orientation requires non-zero covariance; independent standard normal variables have covariance $\text{Cov}(X, Y) = 0$.
- **Option C (Correct):** Because $X$ and $Y$ are independently drawn from $\mathcal{N}(0, 1)$, their joint probability distribution is bivariate isotropic Gaussian centered at $(0, 0)$. The scatter plot appears as a uniform circular cloud densest at the origin, with Pearson correlation $r \approx 0$.
- **Option D (Incorrect):** Funnel-shaped dispersion indicates heteroscedasticity, whereas independent identically distributed normal variables have constant conditional variance.

</details>

---

### 16. Question

Which code snippet demonstrates the modern, object-oriented Matplotlib convention for creating a scatter plot on an explicit `Axes` instance?

- [ ] **A.** `fig = plt.new_plot(); fig.scatter(x, y)`
- [ ] **B.** `ax = plt.create_canvas(); ax.draw_scatter(x, y)`
- [ ] **C.** `scatter = matplotlib.Scatter(x, y); plt.render(scatter)`
- [ ] **D.** `fig, ax = plt.subplots(); ax.scatter(x, y)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.new_plot` does not exist, and `Figure` objects do not have a `.scatter()` plotting method.
- **Option B (Incorrect):** `plt.create_canvas` and `ax.draw_scatter` are not valid Matplotlib methods.
- **Option C (Incorrect):** Matplotlib does not instantiate standalone `matplotlib.Scatter` objects that are rendered via a top-level `plt.render()` function.
- **Option D (Correct):** The canonical object-oriented approach calls `fig, ax = plt.subplots()` to create a `Figure` and an associated `AxesSubplot`, followed by invoking the `ax.scatter(x, y)` method on the specific axis object.

</details>

---

### 17. Question

A researcher needs to plot "hollow" circular markers (transparent interior with a solid blue border) so that densely overlapping points remain individually identifiable. Which configuration achieves this?

- [ ] **A.** `plt.scatter(x, y, hollow=True, color="blue")`
- [ ] **B.** `plt.scatter(x, y, facecolors="none", edgecolors="blue", linewidths=1.2)`
- [ ] **C.** `plt.scatter(x, y, fill=False, border="blue")`
- [ ] **D.** `plt.scatter(x, y, transparent=True, stroke="blue")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `hollow=True` is not a supported keyword argument in Matplotlib.
- **Option B (Correct):** In `plt.scatter()`, setting `facecolors="none"` disables internal glyph filling (making the interior 100% transparent), while `edgecolors="blue"` and `linewidths=1.2` render a clean, hollow perimeter.
- **Option C (Incorrect):** `fill=False` is used in patch artists (like `Polygon` or `Rectangle`), but `plt.scatter()` controls glyph fills via `facecolors="none"`. `border` is also invalid.
- **Option D (Incorrect):** `transparent=True` and `stroke="blue"` are invalid parameters for `plt.scatter()`.

</details>

---

### 18. Question

When visualizing 2D spatial coordinates (e.g., GPS coordinates or physical particle trajectories in meters), why is invoking `ax.set_aspect('equal')` critical?

- [ ] **A.** It forces 1 unit along the horizontal axis to have the exact same physical length in pixels/inches as 1 unit along the vertical axis, preventing geometric distortion of shapes and Euclidean distances.
- [ ] **B.** It ensures that the number of data points on the left half of the plot equals the number on the right half.
- [ ] **C.** It rescales all $(x, y)$ values to the normalized range $[0.0, 1.0]$.
- [ ] **D.** It forces the legend box to match the exact width and height of the main axes frame.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** By default, Matplotlib stretches axes independently to fill the rectangular figure frame. For physical coordinates, maps, or spatial trajectories, this distorts angles and relative distances. `ax.set_aspect('equal')` enforces a 1:1 aspect ratio so that isotropic geometric shapes (such as circles) remain circular rather than stretched into ellipses.
- **Option B (Incorrect):** Aspect ratio governs coordinate metric geometry, not data sample counts across quadrants.
- **Option C (Incorrect):** `set_aspect('equal')` does not mutate or normalize raw data values.
- **Option D (Incorrect):** Aspect ratio applies to axis scaling geometry, not legend bounding boxes.

</details>

---

### 19. Question

Which of the following statements regarding Matplotlib scatter markers and color styling are **correct**?

1. `plt.scatter()` can accept an RGB or RGBA tuple (e.g., `(0.2, 0.4, 0.8, 0.5)`) for uniform color specification.
2. A single call to `plt.scatter(x, y, marker=['o', 's', '^'])` can automatically assign different marker geometries to individual points in the array.
3. Setting `c='none'` and `edgecolors='red'` creates hollow red markers.
4. If $x$ and $y$ contain `np.nan` values at the same index, Matplotlib skips plotting that specific point without throwing an error.

- [ ] **A.** 1 and 2 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 2 and 4 only
- [ ] **D.** 1, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** While Statement 1 is correct (Matplotlib accepts 3- or 4-element RGB/RGBA float tuples for uniform coloring), Statement 2 is false because `plt.scatter()` cannot accept an array or list of multiple marker shapes in a single call.
- **Option B (Incorrect):** Statement 2 is false because passing a list to `marker` raises an error, and this combination omits valid Statements 1 and 4.
- **Option C (Incorrect):** Statement 2 is false, and this option fails to include Statements 1 and 3.
- **Option D (Correct):** Statements 1, 3, and 4 are all correct. Statement 1 is valid because RGB/RGBA tuples are standard color specifications; Statement 3 is valid because `c='none'` (or `facecolors='none'`) disables marker fill while `edgecolors='red'` renders the hollow perimeter; and Statement 4 is valid because Matplotlib masks out `np.nan` coordinate pairs without raising exceptions. Statement 2 is false because varying marker styles across points requires separate `plt.scatter()` calls.

</details>

---

### 20. Question

A biotechnology researcher is plotting bacterial colony counts spanning from 10 to $100,000,000$ across time. In a linear scatter plot, 90% of the points are compressed into a tiny strip near the bottom axis. Which command transforms the vertical axis to an exponential scale where each major division represents a power of 10?

- [ ] **A.** `plt.ytransform("exponential")`
- [ ] **B.** `plt.set_ypower(10)`
- [ ] **C.** `plt.yscale("log")`
- [ ] **D.** `plt.axis_scale(y="orders_of_magnitude")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plt.ytransform` does not exist in Matplotlib.
- **Option B (Incorrect):** `plt.set_ypower` is not a valid method.
- **Option C (Correct):** `plt.yscale('log')` (or `ax.set_yscale('log')`) switches the vertical axis scaling from linear to base-10 logarithmic. This spreads observations across several orders of magnitude evenly, making exponential growth curves appear linear.
- **Option D (Incorrect):** `plt.axis_scale` is not a valid Matplotlib function.

</details>

---
