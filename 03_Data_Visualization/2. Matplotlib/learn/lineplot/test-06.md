# Matplotlib Line Plot Objective Test — Set 06

---

### 51. Question

A student generates sample coordinates to plot a continuous mathematical curve using NumPy:

```python
x = np.linspace(0, 10, 5)
```

What exact array of coordinates is produced by this function call?

- [ ] **A.** `array([0., 2., 4., 6., 8.])`
- [ ] **B.** `array([2., 4., 6., 8., 10.])`
- [ ] **C.** `array([0., 2.5, 5., 7.5, 10.])`
- [ ] **D.** `array([0., 1., 2., 3., 4.])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This would occur if the interval was $[0, 8]$ or if `endpoint=False` was used over $[0, 10)$ with step 2.
- **Option B (Incorrect):** The array starts at 0, not 2.
- **Option C (Correct):** `np.linspace(start, stop, num)` generates `num` evenly spaced samples over the closed interval $[\text{start}, \text{stop}]$ (with `endpoint=True` by default). The step size between points is calculated as:
  $$\text{step} = \frac{\text{stop} - \text{start}}{\text{num} - 1} = \frac{10 - 0}{5 - 1} = \frac{10}{4} = 2.5$$
  The resulting points are $[0.0, 2.5, 5.0, 7.5, 10.0]$.
- **Option D (Incorrect):** This is `np.arange(5)`, not a linspace spanning to 10.

</details>

---

### 52. Question

A physics student plots a full cycle of a sine wave using the following code:

```python
x = np.linspace(0, 2 * np.pi, 9)
```

What is the exact mathematical step size (in radians) between adjacent sampled points in `x`?

- [ ] **A.** $\frac{\pi}{2}$ radians
- [ ] **B.** $\frac{2\pi}{9}$ radians
- [ ] **C.** $\frac{\pi}{8}$ radians
- [ ] **D.** $\frac{\pi}{4}$ radians

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $\frac{\pi}{2}$ would result if only 5 sample points were specified: $\frac{2\pi - 0}{5 - 1} = \frac{\pi}{2}$.
- **Option B (Incorrect):** Dividing by 9 occurs only when `endpoint=False`.
- **Option C (Incorrect):** $\frac{\pi}{8}$ corresponds to 17 sample points.
- **Option D (Correct):** With `endpoint=True`, $N = 9$ points partition the interval $[0, 2\pi]$ into $N - 1 = 8$ equal sub-intervals. The step size is:
  $$\Delta x = \frac{2\pi - 0}{9 - 1} = \frac{2\pi}{8} = \frac{\pi}{4}\text{ radians}$$

</details>

---

### 53. Question

A data analyst writes the following script intending to render a smooth cosine wave:

```python
x = np.linspace(0, 2 * np.pi, 5)
y = np.cos(x)
plt.plot(x, y, 'b-')
plt.show()
```

Instead of a smooth curve, the resulting chart displays an angular, diamond-like polygon made of four straight line segments. What causes this visual artifact?

- [ ] **A.** Matplotlib draws straight lines between adjacent discrete coordinate vertices; 5 sample points across a full wave period are insufficient to resolve the curvature (undersampling).
- [ ] **B.** Matplotlib's `plot()` function is incapable of rendering smooth curves and requires a 3D surface plugin.
- [ ] **C.** The `'b-'` format string enforces a rigid polygonal constraint on trigonometric functions.
- [ ] **D.** Floating-point truncation in `np.cos()` converts transcendental curves into piece-wise linear segments.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** A line plot in computer graphics is inherently a piecewise linear interpolation between discrete vertices. With only 5 points across $[0, 2\pi]$ ($x = 0, \frac{\pi}{2}, \pi, \frac{3\pi}{2}, 2\pi$), the values are $y = [1, 0, -1, 0, 1]$. Connecting these five coordinates with straight segments forms a V-shaped or diamond polygon. Increasing the sampling resolution (e.g., `num=200`) provides sufficient vertex density to appear seamlessly smooth to the human eye.
- **Option B (Incorrect):** Matplotlib renders smooth curves effortlessly when provided with a sufficient density of sample points.
- **Option C (Incorrect):** `'b-'` simply denotes blue solid stroke; it does not alter geometry.
- **Option D (Incorrect):** `np.cos()` calculates standard 64-bit IEEE floating-point values with high numerical precision.

</details>

---

### 54. Question

When sampling a continuous function to display on a standard desktop monitor (e.g., $1920 \times 1080$ resolution), which sampling density in `np.linspace(start, stop, num)` represents the best engineering balance between visual smoothness and rendering performance?

- [ ] **A.** `num = 4` points
- [ ] **B.** `num = 50,000,000` points
- [ ] **C.** `num = 200` to `500` points
- [ ] **D.** `num = 1` point

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 4 points produces severe polygonal stepping and aliasing artifacts for any non-linear curve.
- **Option B (Incorrect):** 50 million points wastes hundreds of megabytes of RAM and causes noticeable rendering lag or canvas freezes without offering any perceptible visual improvement over 500 points on a 1080p display.
- **Option C (Correct):** Across a typical chart spanning 800–1200 horizontal pixels on screen, 200 to 500 points ensures that each line segment spans only a couple of pixels, making vertices imperceptible and rendering mathematically smooth curves while executing virtually instantaneously.
- **Option D (Incorrect):** A single point cannot form a line plot (requires at least 2 points to draw a segment).

</details>

---

### 55. Question

Why is `np.linspace(start, stop, num)` generally preferred over `np.arange(start, stop, step)` when generating coordinates for continuous function plotting in Matplotlib?

- [ ] **A.** `np.arange()` cannot generate floating-point numbers.
- [ ] **B.** `np.linspace()` runs 1,000 times faster because it executes on the GPU.
- [ ] **C.** Matplotlib rejects arrays created by `np.arange()`.
- [ ] **D.** Due to IEEE 754 floating-point rounding errors, `np.arange()` with non-integer step sizes can produce an unpredictable number of elements and may unpredictably include or exclude the upper bound.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `np.arange()` does accept float steps (e.g., `step=0.1`).
- **Option B (Incorrect):** `np.linspace()` runs on the CPU via standard NumPy vectorization.
- **Option C (Incorrect):** Matplotlib accepts any NumPy array or sequence.
- **Option D (Correct):** Because floating-point numbers cannot always be represented exactly in binary, accumulated additions in `np.arange(0.0, 1.0, 0.1)` can result in slight overshoots or undershoots (e.g., final value reaching $0.9999999999999999$ vs $1.0000000000000002$), causing the array size to fluctuate between 10 and 11 elements. In contrast, `np.linspace()` guarantees an exact point count (`num`) and guarantees that the endpoints match `start` and `stop` precisely.

</details>

---

### 56. Question

An engineer plots the polynomial function $f(x) = x^3 - 4x$ over the interval $[-3, 3]$:

```python
x = np.linspace(-3, 3, 100)
y = x**3 - 4*x
plt.plot(x, y, 'purple')
plt.grid(True)
plt.show()
```

At what coordinate does the curve intersect the $y$-axis, and what are its roots (where $y = 0$)?

- [ ] **A.** $y$-intercept at $(0, 0)$; roots at $x = -2, 0, 2$
- [ ] **B.** $y$-intercept at $(0, 4)$; roots at $x = -4, 4$
- [ ] **C.** $y$-intercept at $(0, -4)$; roots at $x = -1, 1$
- [ ] **D.** The curve has no real roots and does not intersect the $y$-axis

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Evaluating at $x = 0$ yields $f(0) = 0^3 - 4(0) = 0$, giving the $y$-intercept at $(0, 0)$. Setting $f(x) = 0$ gives $x(x^2 - 4) = 0$, whose real roots are $x = 0$ and $x = \pm\sqrt{4} = \pm 2$. Thus, the roots are $x = -2, 0, 2$.
- **Option B (Incorrect):** $f(0)$ is 0, not 4.
- **Option C (Incorrect):** $f(1) = 1 - 4 = -3 \neq 0$.
- **Option D (Incorrect):** The function is a cubic polynomial with three distinct real roots.

</details>

---

### 57. Question

An ML practitioner wants to plot the logistic sigmoid activation function $\sigma(x) = \frac{1}{1 + e^{-x}}$ over the domain $[-6, 6]$:

```python
x = np.linspace(-6, 6, 200)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
```

What are the horizontal asymptotes of this curve as $x \to -\infty$ and $x \to +\infty$, and what is the value at $x = 0$?

- [ ] **A.** Asymptotes at $y = -1$ and $y = +1$; value at $x = 0$ is $0.0$
- [ ] **B.** Asymptotes at $y = 0$ and $y = 1$; value at $x = 0$ is $0.5$
- [ ] **C.** Asymptotes at $y = -\infty$ and $y = +\infty$; value at $x = 0$ is $1.0$
- [ ] **D.** Asymptotes at $y = 0.5$ and $y = 1.0$; value at $x = 0$ is $0.25$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The hyperbolic tangent function ($\tanh(x)$) ranges from $-1$ to $+1$, not the standard sigmoid.
- **Option B (Correct):** For the logistic sigmoid:
  - As $x \to -\infty$, $e^{-x} \to +\infty$, so $y \to \frac{1}{1 + \infty} = 0$.
  - At $x = 0$, $e^{0} = 1$, so $y = \frac{1}{1 + 1} = 0.5$.
  - As $x \to +\infty$, $e^{-x} \to 0$, so $y \to \frac{1}{1 + 0} = 1$.
  Therefore, the horizontal asymptotes are $y = 0$ and $y = 1$.
- **Option C (Incorrect):** Sigmoid is bounded within the open interval $(0, 1)$.
- **Option D (Incorrect):** $y = 0.5$ is the midpoint value at $x = 0$, not an asymptote.

</details>

---

### 58. Question

A student attempts to plot the reciprocal function $f(x) = \frac{1}{x}$ across a domain containing zero:

```python
x = np.linspace(-2, 2, 200)
y = 1 / x
plt.plot(x, y)
plt.ylim(-10, 10)
plt.show()
```

The rendered chart displays an unwanted, near-vertical line connecting the deep negative curve on the left of $x=0$ to the steep positive curve on the right. Why does this artifact appear, and how is it cleanly prevented?

- [ ] **A.** The artifact is caused by a bug in `plt.ylim()`; removing `plt.ylim()` eliminates the vertical segment.
- [ ] **B.** The artifact appears because `plt.plot()` connects all consecutive $(x, y)$ coordinate pairs with line segments; it is prevented by splitting the domain into two separate arrays or inserting `np.nan` at the discontinuity.
- [ ] **C.** The artifact occurs because 200 points is too high a resolution; reducing to 10 points fixes it.
- [ ] **D.** It is prevented by passing `discontinuous=True` to `plt.plot()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.ylim()` merely clips the viewable range; without it, the vertical line simply extends to floating-point infinity.
- **Option B (Correct):** Matplotlib has no intrinsic knowledge of mathematical calculus or infinite asymptotes; it simply draws a line segment between point $i$ and point $i+1$. Because the sample points cross $x = 0$, the line connects the large negative value immediately before zero to the large positive value immediately after zero. Inserting `np.nan` at the asymptote tells Matplotlib to break the line path, or plotting the negative and positive intervals in two separate `plt.plot()` calls eliminates the connecting artifact.
- **Option C (Incorrect):** Reducing resolution still leaves the connecting segment.
- **Option D (Incorrect):** There is no `discontinuous` parameter in `plt.plot()`.

</details>

---

### 59. Question

What is the result of calling `np.linspace(0, 1, 5, endpoint=False)`?

- [ ] **A.** `array([0. , 0.2, 0.4, 0.6, 0.8])`
- [ ] **B.** `array([0.  , 0.25, 0.5 , 0.75, 1.  ])`
- [ ] **C.** `array([0.2, 0.4, 0.6, 0.8, 1. ])`
- [ ] **D.** An empty array

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When `endpoint=False`, the stop value ($1.0$) is excluded from the generated samples, and the step size is computed as $\Delta x = \frac{\text{stop} - \text{start}}{\text{num}} = \frac{1 - 0}{5} = 0.2$. The resulting 5 points are $[0.0, 0.2, 0.4, 0.6, 0.8]$.
- **Option B (Incorrect):** This is the output when `endpoint=True` (the default).
- **Option C (Incorrect):** The array starts at 0.0, not 0.2.
- **Option D (Incorrect):** A valid non-empty array of length 5 is returned.

</details>

---

### 60. Question

Consider the following statements regarding continuous curve visualization using NumPy and Matplotlib:

1. NumPy vectorized operations (e.g., `y = np.sin(x)`) compute output arrays significantly faster than iterating with Python `for` loops.
2. If `y` contains `np.nan` values, Matplotlib skips those points and leaves a visual gap in the line rather than drawing across the missing values.
3. Too few sample points in `np.linspace` causes visible polygonal discretization errors.
4. Setting `endpoint=True` in `np.linspace(start, stop, num)` guarantees that `array[-1] == stop`.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** NumPy universal functions (ufuncs) are implemented in compiled C code and execute vectorized batch loops orders of magnitude faster than interpreted Python `for` loops.
- **Statement 2 is correct:** Matplotlib treats `np.nan` values in line data as breaks or pen-up events, resulting in clean visual gaps.
- **Statement 3 is correct:** Connecting too few points creates visible straight-edge chords rather than smooth contours.
- **Statement 4 is correct:** `endpoint=True` explicitly ensures that the final value in the generated sequence matches the specified `stop` parameter.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
