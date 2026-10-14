# Matplotlib Scatter Plot Objective Test — Set 07

---

### 61. Question

A data scientist executes the following NumPy code to overlay an ordinary least-squares linear trendline on a scatter plot:

```python
m, b = np.polyfit(x, y, 1)
```

In standard algebraic notation $\hat{y} = m x + b$, what do the returned variables `m` and `b` represent?

- [ ] **A.** `m` is the Pearson correlation coefficient, and `b` is the standard deviation of residuals.
- [ ] **B.** `m` is the slope of the best-fit line, and `b` is the vertical ($y$-axis) intercept.
- [ ] **C.** `m` is the degrees of freedom, and `b` is the p-value of the regression slope.
- [ ] **D.** `m` is the $x$-intercept, and `b` is the coefficient of determination ($R^2$).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.polyfit` computes polynomial coefficients; it does not return correlation coefficients or residual standard deviations.
- **Option B (Correct):** `np.polyfit(x, y, deg)` fits a polynomial of degree `deg` via least-squares minimization. For `deg=1`, it returns a tuple/array `[m, b]` corresponding to the 1st-degree polynomial $P(x) = p_1 x^1 + p_0$. Thus, `m` is the slope ($\frac{\Delta y}{\Delta x}$) and `b` is the $y$-intercept ($y$ value when $x = 0$).
- **Option C (Incorrect):** `polyfit` does not perform statistical hypothesis testing or return p-values.
- **Option D (Incorrect):** `m` is the slope, not the $x$-intercept, and `b` is not $R^2$.

</details>

---

### 62. Question

A machine learning study fits a linear regression model between weekly training hours ($x$) and sprint speed index ($y$). The resulting fit yields slope $m = 3.5$ and intercept $b = 12.0$.

What is the predicted speed index $\hat{y}$ on the trendline for an athlete who trains $x = 8.0$ hours per week?

- [ ] **A.** $28.0$
- [ ] **B.** $32.5$
- [ ] **C.** $44.0$
- [ ] **D.** $40.0$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $28.0$ accounts only for $m \times x = 3.5 \times 8 = 28.0$, forgetting to add the intercept $b$.
- **Option B (Incorrect):** $32.5$ is an arithmetic error ($3.5 \times 8 + 4.5$).
- **Option C (Incorrect):** $44.0$ corresponds to $4.0 \times 8 + 12.0$.
- **Option D (Correct):** Evaluating the linear regression trendline equation $\hat{y} = m x + b$:
  $$\hat{y} = (3.5 \times 8.0) + 12.0 = 28.0 + 12.0 = 40.0$$
  The predicted trendline coordinate is exactly $(8.0, 40.0)$.

</details>

---

### 63. Question

A developer attempts to draw a trendline across a scatter plot by calling:

```python
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color="red")
```

If the raw array `x` contains unsorted, randomly ordered values (e.g., `x = [5, 1, 4, 2, 3]`), what visual bug appears on the screen?

- [ ] **A.** The line connects points in the order they appear in the array, producing an ugly, jagged zigzag scribble traversing back and forth across the canvas instead of a smooth straight line.
- [ ] **B.** Matplotlib raises an `UnorderedArrayError` and refuses to plot.
- [ ] **C.** The line is rendered vertically along the y-axis.
- [ ] **D.** The scatter points are automatically sorted, mutating the original dataset.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.plot()` connects consecutive elements in array order. If `x` is unsorted, `plt.plot()` draws a line from $(5, \hat{y}_5)$ down to $(1, \hat{y}_1)$, back up to $(4, \hat{y}_4)$, and so forth. This produces a chaotic crisscrossing scribble. To fix this, one should generate an independent, monotonically sorted evaluation domain: `x_line = np.linspace(x.min(), x.max(), 100)`.
- **Option B (Incorrect):** Matplotlib does not check if coordinates are sorted; it blindly draws line segments between array indices.
- **Option C (Incorrect):** The line will not collapse to the vertical axis.
- **Option D (Incorrect):** Plotting functions never mutate the input arrays.

</details>

---

### 64. Question

What is the canonical best-practice code pattern for generating and overlaying a clean, continuous trendline across a scatter plot?

- [ ] **A.**
  ```python
  plt.scatter(x, y)
  plt.trendline(x, y, style="dashed")
  ```
- [ ] **B.**
  ```python
  plt.scatter(x, y)
  plt.draw_line(np.mean(x), np.mean(y))
  ```
- [ ] **C.**
  ```python
  plt.scatter(x, y, alpha=0.7)
  m, b = np.polyfit(x, y, 1)
  x_line = np.linspace(x.min(), x.max(), 100)
  plt.plot(x_line, m * x_line + b, color="red", linestyle="--", label="Trend")
  plt.legend()
  ```
- [ ] **D.**
  ```python
  plt.scatter(x, y)
  plt.regression(order=1)
  ```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plt.trendline` does not exist in standard Matplotlib.
- **Option B (Incorrect):** `plt.draw_line` is not a Matplotlib function.
- **Option C (Correct):** This pattern is the canonical workflow used in the reference material:
  1. Plot the raw scatter data: `plt.scatter(x, y)`.
  2. Compute slope and intercept: `m, b = np.polyfit(x, y, 1)`.
  3. Create a smooth, ordered 1D grid spanning the data domain: `x_line = np.linspace(x.min(), x.max(), 100)`.
  4. Evaluate and overlay the linear function: `plt.plot(x_line, m * x_line + b, ...)`.
- **Option D (Incorrect):** `plt.regression` is not a Matplotlib method (though Seaborn has `regplot`).

</details>

---

### 65. Question

A scientist observes that the relationship between enzyme temperature and reaction rate is quadratic (parabolic with an inverted U-shape). Which code snippet fits a 2nd-degree polynomial curve and plots it?

- [ ] **A.**
  ```python
  coeffs = np.polyfit(temp, rate, 2)
  x_curve = np.linspace(temp.min(), temp.max(), 200)
  y_curve = np.polyval(coeffs, x_curve)
  plt.plot(x_curve, y_curve, color="crimson", label="Quadratic Fit")
  ```
- [ ] **B.**
  ```python
  coeffs = np.polyfit(temp, rate, order="parabola")
  plt.curve(coeffs)
  ```
- [ ] **C.**
  ```python
  plt.fit_polynomial(temp, rate, power=2)
  ```
- [ ] **D.**
  ```python
  m, b = np.polyfit(temp**2, rate, 1)
  plt.plot(temp, m * temp + b)
  ```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `np.polyfit(temp, rate, 2)` fits a 2nd-degree polynomial $\hat{y} = a x^2 + b x + c$, returning coefficients $[a, b, c]$. `np.polyval(coeffs, x_curve)` cleanly evaluates the polynomial at each point along the dense grid `x_curve`, and `plt.plot()` renders the smooth parabolic curve.
- **Option B (Incorrect):** `order="parabola"` is invalid syntax; `deg` must be an integer (e.g., `2`), and `plt.curve` does not exist.
- **Option C (Incorrect):** `plt.fit_polynomial` is not a valid Matplotlib function.
- **Option D (Incorrect):** Fitting $temp^2$ alone omits the linear term ($b \cdot temp$), failing to fit general non-centered parabolas.

</details>

---

### 66. Question

When a linear regression model ($\text{degree}=1$) is fitted to a strictly parabolic (quadratic) dataset, how does the mismatch manifest visually on a scatter plot with regression overlay?

- [ ] **A.** The slope $m$ automatically evaluates to $\infty$.
- [ ] **B.** The straight linear trendline cuts through the curve, with data points systematically bowing below the line at the extremes and bulging above the line in the center (or vice versa), revealing systematic non-random residual bias.
- [ ] **C.** The points on the scatter plot collapse onto the horizontal axis.
- [ ] **D.** The colorbar disappears from the canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Slope calculation is numerically stable and will simply be the average secant slope, not infinity.
- **Option B (Correct):** Underfitting a non-linear relationship with a 1st-degree linear model produces structured, correlated residuals. Visually, the straight trendline cannot bend to follow the data curvature; points at the edges will systematically lie on one side of the line, while points in the center lie on the opposite side.
- **Option C (Incorrect):** Coordinate points remain at their true $(x, y)$ positions.
- **Option D (Incorrect):** Colorbar rendering is completely independent of regression line geometry.

</details>

---

### 67. Question

A linear trendline on a scatter plot passes exactly through the coordinate points $(2.0, 15.0)$ and $(6.0, 35.0)$. What is the numerical slope $m$ and $y$-intercept $b$ of this line?

- [ ] **A.** $m = 20.0, b = 15.0$
- [ ] **B.** $m = 4.0, b = 7.0$
- [ ] **C.** $m = 2.5, b = 10.0$
- [ ] **D.** $m = 5.0, b = 5.0$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $20.0$ is $\Delta y = 35 - 15$, failing to divide by $\Delta x$.
- **Option B (Incorrect):** $4.0$ is $\Delta x = 6 - 2$, not the slope.
- **Option C (Incorrect):** Arithmetic calculation error.
- **Option D (Correct):**
  1. Calculate slope $m$:
     $$m = \frac{y_2 - y_1}{x_2 - x_1} = \frac{35.0 - 15.0}{6.0 - 2.0} = \frac{20.0}{4.0} = 5.0$$
  2. Calculate $y$-intercept $b$ using $(2.0, 15.0)$:
     $$b = y_1 - m x_1 = 15.0 - (5.0 \times 2.0) = 15.0 - 10.0 = 5.0$$
  Thus, $m = 5.0$ and $b = 5.0$ (Equation: $\hat{y} = 5.0x + 5.0$). Checking point 2: $5.0(6.0) + 5.0 = 35.0$.

</details>

---

### 68. Question

A statistician wants to visually highlight the regression residuals (the vertical errors $e_i = y_i - \hat{y}_i$) by drawing vertical line segments between each scatter point $(x_i, y_i)$ and its corresponding prediction $(x_i, \hat{y}_i)$ on the trendline.

Which Matplotlib function is designed specifically to draw these vertical segments in a single vectorized call?

- [ ] **A.** `plt.axvline()`
- [ ] **B.** `plt.errorbar_full()`
- [ ] **C.** `plt.vlines(x, ymin=np.minimum(y, y_pred), ymax=np.maximum(y, y_pred), color='gray', linestyle=':')`
- [ ] **D.** `plt.drop_lines(x, y, y_pred)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plt.axvline()` draws an infinite vertical line spanning the entire axes height at a single $x$ coordinate.
- **Option B (Incorrect):** `errorbar_full` does not exist in Matplotlib.
- **Option C (Correct):** `plt.vlines(x, ymin, ymax)` draws vertical line segments from `ymin` to `ymax` at each coordinate in vector `x`. This provides an elegant, vectorized method to render residual error segments connecting data points to the fitted regression curve.
- **Option D (Incorrect):** `plt.drop_lines` is not a valid Matplotlib function.

</details>

---

### 69. Question

A data analyst overlays a thick trendline on a scatter plot, but notices that the line completely obscures several key data points that lie directly beneath it. How can the artist layering order be controlled so that the scatter markers are drawn **on top of** the trendline?

- [ ] **A.** Set `zorder` such that the scatter plot has a higher value than the line plot (e.g., `ax.scatter(..., zorder=3)` and `ax.plot(..., zorder=2)`).
- [ ] **B.** Call `plt.bring_to_front('scatter')`.
- [ ] **C.** Set `depth='front'` inside `plt.scatter()`.
- [ ] **D.** The order cannot be changed; lines are always rendered on top of points in Matplotlib.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, the rendering hierarchy is governed by the `zorder` property. Artists with higher `zorder` values are painted on top of artists with lower `zorder` values. Setting `scatter` to `zorder=3` and the regression line to `zorder=2` ensures that scatter markers are drawn crisp and unobstructed over the trendline.
- **Option B (Incorrect):** `plt.bring_to_front` does not exist in Matplotlib.
- **Option C (Incorrect):** `depth` is not a recognized Matplotlib keyword argument.
- **Option D (Incorrect):** Rendering order is fully customizable using `zorder`.

</details>

---

### 70. Question

Which of the following statements regarding regression overlays on scatter plots are **correct**?

1. Ordinary least-squares regression via `np.polyfit(x, y, 1)` minimizes the sum of squared vertical residuals $\sum (y_i - \hat{y}_i)^2$.
2. A high coefficient of determination ($R^2 \approx 1.0$) between $x$ and $y$ conclusively proves that changes in $x$ causally produce changes in $y$.
3. A single severe bivariate outlier can exert immense leverage and substantially distort the slope of a linear trendline.
4. Polynomial models of excessive degree (e.g., `np.polyfit(x, y, 15)`) on small datasets cause extreme overfitting with Runge's phenomenon (wild oscillations near boundaries).

- [ ] **A.** 1 and 2 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** While Statement 1 is correct (least-squares minimizes squared vertical errors), Statement 2 is false because correlation never proves causality; lurking confounders or spurious co-occurrences can produce high $R^2$ without causal links.
- **Option B (Incorrect):** Statement 2 is false (correlation does not imply causation), and this option omits valid Statements 1 and 4.
- **Option C (Incorrect):** Statement 2 is false, and this option fails to include valid Statement 3.
- **Option D (Correct):** Statements 1, 3, and 4 are all correct. Statement 1 is valid because OLS regression minimizes $\sum e_i^2$; Statement 3 is valid because high-leverage outliers strongly distort linear slopes; and Statement 4 is valid because excessive polynomial degrees cause severe boundary oscillations (Runge's phenomenon). Statement 2 is false.

</details>

---
