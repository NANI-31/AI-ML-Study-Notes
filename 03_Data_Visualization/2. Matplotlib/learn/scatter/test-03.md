# Matplotlib Scatter Plot Objective Test — Set 03

---

### 21. Question

A data analyst is designing a bubble chart where marker size represents factory output. Factory Alpha has an output of 100 units and is plotted with marker size `s = 50`. Factory Beta has an output of 400 units and is plotted with `s = 200`.

What is the ratio of the visual diameter (or width) of Factory Beta's marker to Factory Alpha's marker?

- [ ] **A.** $4.0$
- [ ] **B.** $2.0$
- [ ] **C.** $16.0$
- [ ] **D.** $1.414$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** $4.0$ is the ratio of the marker areas ($s_2 / s_1 = 200 / 50 = 4$), not their diameters.
- **Option B (Correct):** In Matplotlib, the parameter `s` scales the **area** of the glyph ($A \propto s$). The area of a circle is related to diameter $d$ by $A = \frac{\pi}{4} d^2$, which implies $d \propto \sqrt{s}$. Therefore, the ratio of visual diameters is:
  $$\frac{d_{\text{Beta}}}{d_{\text{Alpha}}} = \sqrt{\frac{s_{\text{Beta}}}{s_{\text{Alpha}}}} = \sqrt{\frac{200}{50}} = \sqrt{4} = 2.0$$
  Factory Beta's bubble is twice as wide as Factory Alpha's bubble, while occupying 4 times the visual area.
- **Option C (Incorrect):** $16.0$ would result from squaring the area ratio ($4^2$), which is mathematically inverted.
- **Option D (Incorrect):** $1.414$ is $\sqrt{2}$, which would apply if the area ratio were $2.0$.

</details>

---

### 22. Question

In an urban analytics dashboard, City X has a population of 250,000 and is rendered with marker size `s = 40`. City Y has a population of 1,000,000. Assuming marker area must be strictly directly proportional to population, what numerical value must be supplied to `s` for City Y?

- [ ] **A.** `160`
- [ ] **B.** `640`
- [ ] **C.** `80`
- [ ] **D.** `1600`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Because Matplotlib's `s` parameter directly controls marker area in points squared, proportional scaling follows the linear relationship:
  $$s_Y = s_X \times \left(\frac{\text{Pop}_Y}{\text{Pop}_X}\right) = 40 \times \left(\frac{1,000,000}{250,000}\right) = 40 \times 4 = 160$$
- **Option B (Incorrect):** $640$ occurs if a programmer mistakenly squares the population ratio ($40 \times 4^2$), falsely assuming `s` controls radius.
- **Option C (Incorrect):** $80$ is only double the size ($40 \times 2$), underrepresenting the fourfold population difference.
- **Option D (Incorrect):** $1600$ is an arithmetic order-of-magnitude calculation error.

</details>

---

### 23. Question

A junior developer mistakenly sets `s = radius` where `radius = [10, 20, 40]`, intending to show quantities that double at each step. What cognitive visual distortion occurs when human observers examine this plot?

- [ ] **A.** Observers perceive all three bubbles as identical because human eyes cannot distinguish circular area differences.
- [ ] **B.** The perceived visual weight scales linearly with the underlying numbers because screen pixels naturally counteract 2D area expansion.
- [ ] **C.** The largest bubble appears only slightly larger than the smallest bubble due to optical compression.
- [ ] **D.** The visual area of the markers scales quadratically ($A \propto \text{radius}^2$), meaning a value that doubled from 10 to 20 occupies 4 times more visual area, severely exaggerating differences in the underlying data.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The human visual cortex is highly sensitive to relative 2D area; observers immediately see massive size differences.
- **Option B (Incorrect):** Screen pixels do not counteract geometric area scaling.
- **Option C (Incorrect):** The area increases by a factor of 16 ($40^2 / 10^2 = 1600 / 100 = 16$), causing extreme exaggeration rather than compression.
- **Option D (Correct):** The human eye perceives the visual weight of 2D shapes by their surface **area**, not their linear radius or diameter. Because $A = \pi r^2$, if `s` is mapped directly to a linear metric instead of area (or if a linear radius is passed directly without taking the square root), the visual area expands quadratically. A 2x increase in the data produces a 4x visual footprint, violating Edward Tufte's "Lie Factor" principle.

</details>

---

### 24. Question

What happens when a developer executes the following script where the sizing array `s` contains 4 elements but coordinate arrays contain 5 elements?

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]
s = [20, 50, 80, 120]

plt.scatter(x, y, s=s)
plt.show()
```

- [ ] **A.** Matplotlib uses the scalar mean of `s` (67.5) to size all 5 points.
- [ ] **B.** The fifth point is automatically rendered with size `s = 0` (invisible).
- [ ] **C.** Matplotlib raises a `ValueError: s must be a scalar, or the same size as x and y`.
- [ ] **D.** Matplotlib recycles the first element (`s[0] = 20`) for the fifth point.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not compute the average of a vector to fall back to a scalar.
- **Option B (Incorrect):** Matplotlib does not silently assign size 0 to missing values.
- **Option C (Correct):** In `plt.scatter(x, y, s=...)`, `s` must either be a single scalar float/int (applied uniformly to all points) or an array-like collection whose length exactly matches the length of $x$ and $y$. If lengths mismatch (`len(s) != len(x)`), Matplotlib raises `ValueError: s must be a scalar, or the same size as x and y`.
- **Option D (Incorrect):** Array broadcasting in Matplotlib does not cycle elements of unequal non-scalar sequences.

</details>

---

### 25. Question

In the reference notebook, continuous feature scaling for marker sizes frequently uses an affine transformation formula:

```python
sizes = (values / values.max()) * 200 + 20
```

What is the primary technical and visualization purpose of including the constant additive offset `+ 20`?

- [ ] **A.** It establishes a guaranteed minimum baseline marker size so that observations with values near zero remain visible and clickable rather than shrinking to zero pixels.
- [ ] **B.** It converts the measurement units from points squared to screen pixels.
- [ ] **C.** It prevents Matplotlib from throwing a zero-division exception during rendering.
- [ ] **D.** It automatically adds a 20% alpha transparency buffer to overlapping markers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** If sizes are scaled purely proportionally as `(values / max) * 200`, a data point with value $0.0$ (or near zero) evaluates to `s = 0.0` points$^2$. A marker with zero area is completely invisible to the user. Adding a positive baseline offset (e.g., `+ 20`) guarantees that even the minimum data point has a perceptible glyph footprint.
- **Option B (Incorrect):** Units in Matplotlib remain typographic points squared ($\text{pt}^2$); additive constants do not alter dimensional units.
- **Option C (Incorrect):** Zero marker size (`s = 0`) does not raise a zero-division error in Matplotlib; it simply draws nothing.
- **Option D (Incorrect):** Marker size parameters do not control alpha transparency.

</details>

---

### 26. Question

What is the result of passing negative values inside the marker size array, as shown below?

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [10, 20, 30]
s = [50, -25, 100]

plt.scatter(x, y, s=s)
plt.show()
```

- [ ] **A.** The negative point is rendered with inverted RGB colors.
- [ ] **B.** Matplotlib takes the absolute value (`abs(-25) = 25`) and renders the point normally.
- [ ] **C.** The negative point is drawn as an inverted upside-down triangle (`marker='v'`).
- [ ] **D.** Matplotlib raises a `ValueError` indicating that marker sizes cannot be negative.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Negative marker sizes do not invert facecolors.
- **Option B (Incorrect):** Matplotlib does not silently coerce negative sizes with `abs()`.
- **Option C (Incorrect):** Marker shape is governed by the `marker` argument, never by the sign of `s`.
- **Option D (Correct):** Geometric area is strictly non-negative. In Matplotlib, if any element in the array passed to `s` is negative ($s_i < 0$), validation logic in `PathCollection` raises `ValueError: s must be a positive number or array of positive numbers` (or `ValueError: collection sizes must be non-negative`).

</details>

---

### 27. Question

Which of the following statements regarding bubble chart sizing in Matplotlib are **correct**?

1. The default marker size in Matplotlib scatter plots when `s` is omitted is governed by `rcParams['lines.markersize']**2` (typically $6^2 = 36\text{ pt}^2$).
2. Multiplying marker diameter by 3 requires multiplying parameter `s` by 9.
3. Setting `s=0` raises a `ValueError: ZeroSizeNotAllowed`.
4. Marker size `s` is independent of figure DPI scaling when saving to vector formats (PDF/SVG).

- [ ] **A.** 1 and 3 only
- [ ] **B.** 1, 2, and 4 only
- [ ] **C.** 2 and 3 only
- [ ] **D.** 2, 3, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While Statement 1 is correct (default marker size derives from `lines.markersize**2`), Statement 3 is false because `s=0` is mathematically valid and does not raise an error; it merely produces an invisible point.
- **Option B (Correct):** Statements 1, 2, and 4 are all correct. Statement 1 is valid because omitted `s` defaults to $6^2 = 36\text{ pt}^2$; Statement 2 is valid because area scales with the square of diameter ($3^2 = 9$); and Statement 4 is valid because points are absolute typographic units ($1/72$ inch) preserved in vector exports. Statement 3 is false.
- **Option C (Incorrect):** Statement 3 is false (`s=0` raises no error), and this option omits valid Statements 1 and 4.
- **Option D (Incorrect):** Statement 3 is false, and this option fails to include valid Statement 1.

</details>

---

### 28. Question

A data scientist needs to visualize wealth distribution across 500 individuals where net worth spans from $\$10,000$ to $\$10,000,000,000$ (a 6-order-of-magnitude power-law distribution). If linear size scaling (`s = net_worth * k`) is applied, the billionaire's bubble is 1,000,000 times larger than the average citizen's, engulfing the entire screen.

What non-linear mathematical transformation on `net_worth` is best practice for scaling `s` to preserve relative ordering while keeping all bubbles legible?

- [ ] **A.** Inverted reciprocal scaling: `s = 1.0 / net_worth`
- [ ] **B.** Factorial transformation: `s = math.factorial(net_worth)`
- [ ] **C.** Logarithmic or square-root power transformation: `s = np.log10(net_worth) * scale_factor` or `s = np.sqrt(net_worth) * scale_factor`
- [ ] **D.** Trigonometric sine transformation: `s = np.sin(net_worth)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Reciprocal scaling inverts the relationship, making the poorest individuals have the largest bubbles and billionaires have tiny bubbles.
- **Option B (Incorrect):** Factorials grow hyper-exponentially and overflow floating-point limits.
- **Option C (Correct):** For heavily skewed power-law or Pareto-distributed variables spanning multiple orders of magnitude, applying a monotonic compressing function like $\log_{10}(x)$ or power-law root $x^{0.5}$ (or $x^{0.3}$) compresses the extreme dynamic range into a compact interval, preventing outlier bubbles from dominating the entire canvas.
- **Option D (Incorrect):** The sine function oscillates between $-1$ and $+1$, destroying monotonic ordering.

</details>

---

### 29. Question

In typography and Matplotlib, $1\text{ point} = \frac{1}{72}\text{ inch}$. If a square marker (`marker='s'`) is plotted with size `s = 144`, what is the physical edge length of the square on the rendered page?

- [ ] **A.** $\frac{1}{6}\text{ inch}$ ($0.167\text{ inches}$, or $12\text{ points}$)
- [ ] **B.** $2\text{ inches}$
- [ ] **C.** $\frac{1}{144}\text{ inch}$
- [ ] **D.** $14.4\text{ millimeters}$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For a square marker, the parameter `s` represents the total area in points squared ($A = 144\text{ pt}^2$). The side length of the square is:
  $$\text{side} = \sqrt{s} = \sqrt{144} = 12\text{ points}$$
  Converting points to physical inches:
  $$\text{side in inches} = \frac{12\text{ pt}}{72\text{ pt/inch}} = \frac{1}{6}\text{ inch} \approx 0.1667\text{ inches}$$
- **Option B (Incorrect):** $2\text{ inches}$ equals $144\text{ points}$, which would only be the edge length if `s` represented side length squared in inches ($144^2\text{ pt}^2$).
- **Option C (Incorrect):** $\frac{1}{144}\text{ inch}$ is an arbitrary inverted fraction.
- **Option D (Incorrect):** $12\text{ pt} = 12 \times \frac{25.4}{72}\text{ mm} = 4.23\text{ mm}$, not $14.4\text{ mm}$.

</details>

---

### 30. Question

A scatter plot displays three distinct bubbles with areas `s = 25`, `s = 100`, and `s = 400`. What is the exact ratio of their visual diameters ($d_1 : d_2 : d_3$)?

- [ ] **A.** $25 : 100 : 400$ ($1 : 4 : 16$)
- [ ] **B.** $1 : 3 : 9$
- [ ] **C.** $1 : 1.414 : 2$
- [ ] **D.** $5 : 10 : 20$ ($1 : 2 : 4$)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** $1 : 4 : 16$ is the ratio of their areas ($s$), not their linear diameters.
- **Option B (Incorrect):** $1 : 3 : 9$ corresponds to areas scaling as $1 : 9 : 81$.
- **Option C (Incorrect):** $1 : 1.414 : 2$ corresponds to areas scaling as $1 : 2 : 4$.
- **Option D (Correct):** Because diameter is proportional to the square root of area ($d \propto \sqrt{s}$):
  $$d_1 \propto \sqrt{25} = 5$$
  $$d_2 \propto \sqrt{100} = 10$$
  $$d_3 \propto \sqrt{400} = 20$$
  The ratio $5 : 10 : 20$ simplifies directly to $1 : 2 : 4$. Each bubble doubles in diameter as area quadruples.

</details>

---
