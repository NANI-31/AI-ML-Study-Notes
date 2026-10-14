# Matplotlib Ticks & Scales Objective Test — Set 04

---

### Question 31

A data scientist models the growth of parameters in Large Language Models (ranging from $10^6$ in early transformers to $10^{12}$ in modern clusters). 

Which command transforms the vertical axis so that each tenfold increase (order of magnitude) occupies an equal visual distance on the canvas?

- [ ] A) `ax.set_yscale('exponential')`
- [ ] B) `ax.set_yscale('log')` (or `plt.yscale('log')`)
- [ ] C) `ax.set_yscale('power')`
- [ ] D) `ax.yaxis.set_geometric(True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_yscale('log')` (or `plt.yscale('log')`)

**Explanation:**

- **A is incorrect:** `'exponential'` is not a recognized scale name in Matplotlib.
- **B is correct:** Setting `ax.set_yscale('log')` applies a logarithmic coordinate transformation (base 10 by default) to the vertical axis. Under this transformation, equal coordinate distances correspond to equal multiplicative ratios: the distance from $10^6$ to $10^7$ is identical to the distance from $10^{11}$ to $10^{12}$, enabling clear inspection of data spanning multiple orders of magnitude.
- **C is incorrect:** `'power'` is not a valid scale string in `set_yscale()`.
- **D is incorrect:** `set_geometric` does not exist in the Matplotlib API.

</details>

---

### Question 32

An algorithm engineer analyzes execution times of binary search trees across input sizes that double at each step ($N = 2, 4, 8, 16, 32, \dots, 1024$).

Which command establishes a horizontal logarithmic scale using base $2$ rather than the default base $10$?

- [ ] A) `ax.set_xscale('log2')`
- [ ] B) `ax.set_xscale('binary')`
- [ ] C) `ax.set_xscale('log', factor=2)`
- [ ] D) `ax.set_xscale('log', base=2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.set_xscale('log', base=2)`

**Explanation:**

- **A is incorrect:** `'log2'` is not a registered scale name string in Matplotlib.
- **B is incorrect:** `'binary'` is not a valid scale parameter in `set_xscale()`.
- **C is incorrect:** The parameter is `base`, not `factor`.
- **D is correct:** In Matplotlib, the logarithmic scale accepts the keyword argument `base`. Setting `ax.set_xscale('log', base=2)` configures the underlying log transformation to use base 2, placing major ticks and grid lines at powers of two ($2^0, 2^1, 2^2, 2^3, \dots$).

</details>

---

### Question 33

An engineer attempts to plot gradient descent loss values that decrease toward zero:

```python
losses = [1.5, 0.4, 0.08, 0.01, 0.0, -0.05]
ax.plot(losses)
ax.set_yscale('log')
```

What issue arises because the dataset contains zero and negative numbers?

- [ ] A) Because $\log(x)$ is mathematically undefined for $x \le 0$, Matplotlib emits a warning and masks or clips the non-positive data points, rendering them invisible or truncating the curve.
- [ ] B) Matplotlib automatically takes the absolute value of all numbers and plots them without warning.
- [ ] C) The entire Python kernel crashes with an unhandled `FloatingPointDivisionError`.
- [ ] D) Zero is converted to $1.0$ and negative numbers are plotted on an inverted mirror axis.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Because $\log(x)$ is mathematically undefined for $x \le 0$, Matplotlib emits a warning and masks or clips the non-positive data points, rendering them invisible or truncating the curve.

**Explanation:**

- **A is correct:** The real-valued logarithm $\log(x)$ is only defined for strictly positive real numbers ($x > 0$). When a dataset contains $0.0$ or negative values and standard `'log'` scaling is applied, Matplotlib cannot compute coordinate transforms for those points. By default, it emits a `UserWarning` and masks or clips the invalid points, preventing them from being displayed. For datasets crossing zero into negative territory, a symmetrical log scale (`'symlog'`) must be used instead.
- **B is incorrect:** Matplotlib never silently applies `abs()` to user data.
- **C is incorrect:** Matplotlib gracefully handles invalid domains using masked arrays or NaN clipping rather than crashing the Python process.
- **D is incorrect:** Standard log scaling does not create mirror axes.

</details>

---

### Question 34

On a standard base-10 logarithmic axis (spanning $10^0$ to $10^3$), what do the minor tick marks represent by default within each decade?

- [ ] A) Fractional percentages ($10.25, 10.5, 10.75$).
- [ ] B) Powers of $e$ ($\approx 2.718, 7.389$).
- [ ] C) Integer multiples of the decade baseline ($2, 3, 4, 5, 6, 7, 8, 9 \times 10^k$), which appear with progressively narrowing visual spacing toward the next decade.
- [ ] D) Halfway points spaced at uniform visual intervals.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Integer multiples of the decade baseline ($2, 3, 4, 5, 6, 7, 8, 9 \times 10^k$), which appear with progressively narrowing visual spacing toward the next decade.

**Explanation:**

- **A is incorrect:** Minor ticks do not represent decimal fractional percentages.
- **B is incorrect:** Base-10 log minor ticks are not powers of $e$.
- **C is correct:** In logarithmic scale formatting, minor ticks demarcate intermediate integer multiples within each decade: $20, 30, 40, 50, 60, 70, 80, 90$ between $10$ and $100$. Because $\log(2) \approx 0.301$, $\log(3) \approx 0.477$, etc., the distance between successive integers compresses geometrically, producing the classic logarithmic tick cluster that bunches up toward the next major decade tick.
- **D is incorrect:** Minor ticks on log scales are not placed at visually uniform midpoint intervals.

</details>

---

### Question 35

A researcher wants minor ticks on a base-10 log axis to appear **only** at multiples of $2$ and $5$ within each decade (e.g., $20$ and $50$ between $10$ and $100$), suppressing the other cluttered intermediate ticks.

Which parameter in `ax.set_yscale('log', ...)` controls these intermediate subdivision locations?

- [ ] A) `minor_divisions=[2, 5]`
- [ ] B) `subs=[2, 5]` (or `subs=(2, 5)`)
- [ ] C) `intermediate_ticks=[2, 5]`
- [ ] D) `sub_ticks=[2, 5]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `subs=[2, 5]` (or `subs=(2, 5)`)

**Explanation:**

- **A is incorrect:** `minor_divisions` is not a recognized parameter in `set_yscale()`.
- **B is correct:** In Matplotlib's `LogScale`, the `subs` parameter specifies the sequence of integer multiples of the base where minor ticks are placed within each decade. Supplying `subs=[2, 5]` (or passing it to `matplotlib.ticker.LogLocator(subs=[2, 5])`) restricts minor ticks exclusively to $2 \times 10^k$ and $5 \times 10^k$, creating a clean, uncluttered minor grid.
- **C is incorrect:** `intermediate_ticks` is not a valid parameter.
- **D is incorrect:** `sub_ticks` is a fictitious argument name.

</details>

---

### Question 36

When plotting learning rate range tests (the LR Finder technique proposed by Leslie Smith), learning rates are swept across five orders of magnitude from $\eta = 10^{-6}$ to $\eta = 10^{-1}$.

Why is using a logarithmic horizontal axis (`ax.set_xscale('log')`) mathematically imperative for this diagnostic visualization?

- [ ] A) Because neural network loss functions are non-convex.
- [ ] B) Because linear scales increase GPU memory consumption during inference.
- [ ] C) Because the learning rate finder algorithm only outputs strings.
- [ ] D) Because on a linear scale, the lower 90% of the tested learning rate range ($10^{-6}$ through $10^{-2}$) is crammed into a tiny, unreadable 1% sliver near zero, whereas a log scale distributes each decade of learning rate evenly across the plot.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Because on a linear scale, the lower 90% of the tested learning rate range ($10^{-6}$ through $10^{-2}$) is crammed into a tiny, unreadable 1% sliver near zero, whereas a log scale distributes each decade of learning rate evenly across the plot.

**Explanation:**

- **A is incorrect:** Loss function convexity has no influence on canvas coordinate scale selection.
- **B is incorrect:** Scale choice affects 2D canvas transformation, not GPU memory allocation.
- **C is incorrect:** Learning rate finders output floating-point numbers, not strings.
- **D is correct:** On a linear scale spanning $0.0$ to $0.1$, the value $10^{-2} = 0.01$ sits at just $10\%$ of the axis length, and $10^{-3} = 0.001$ sits at $1\%$. All crucial initial learning rate dynamics occurring between $10^{-6}$ and $10^{-3}$ would be crushed into an illegible clump against the left spine. A logarithmic scale allocates equal visual distance to $[10^{-6}, 10^{-5}]$, $[10^{-5}, 10^{-4}]$, etc., providing uniform visibility across every decade.

</details>

---

### Question 37

In complex systems and network theory, a data scientist investigates a power-law relationship $y = C \cdot x^{-\alpha}$. 

When both axes are transformed to logarithmic scales via `ax.loglog()` (or setting both `set_xscale('log')` and `set_yscale('log')`), what visual shape does this power-law distribution assume?

- [ ] A) A perfectly straight line whose downward slope is equal to $-\alpha$.
- [ ] B) A bell-shaped Gaussian curve.
- [ ] C) A horizontal sigmoid curve with upper and lower asymptotes.
- [ ] D) A parabolic arc opening upward.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A perfectly straight line whose downward slope is equal to $-\alpha$.

**Explanation:**

- **A is correct:** Taking the logarithm of both sides of $y = C x^{-\alpha}$ yields $\log(y) = \log(C) - \alpha \log(x)$. On a log-log plot, the horizontal and vertical coordinate axes represent $\log(x)$ and $\log(y)$. Therefore, the equation represents a linear equation $Y = \beta_0 + \beta_1 X$, where the y-intercept is $\log(C)$ and the constant linear slope is $-\alpha$. Observing a straight line on a log-log plot is the foundational diagnostic test for power-law phenomena.
- **B is incorrect:** Gaussian bell curves do not form straight lines on log-log axes.
- **C is incorrect:** Sigmoidal curves have varying slopes and horizontal asymptotes.
- **D is incorrect:** Parabolic arcs correspond to quadratic polynomials, not power laws.

</details>

---

### Question 38

In modern Matplotlib, how can a developer configure the logarithmic scale to mask non-positive values (treating them as `NaN` so they are skipped by line plotters) instead of clipping them to a small positive epsilon?

- [ ] A) `ax.set_yscale('log', on_error='ignore')`
- [ ] B) `ax.set_yscale('log', drop_zeros=True)`
- [ ] C) `ax.set_yscale('log', nonpositive='mask')`
- [ ] D) `ax.set_yscale('log', filter='positive_only')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.set_yscale('log', nonpositive='mask')`

**Explanation:**

- **A is incorrect:** `on_error` is not a valid argument for `set_yscale()`.
- **B is incorrect:** `drop_zeros` does not exist in the Matplotlib API.
- **C is correct:** In Matplotlib's `LogScale`, the `nonpositive` keyword parameter determines how values $\le 0$ are handled. Setting `nonpositive='mask'` dynamically masks non-positive numbers as invalid/NaN so that continuous line plots break cleanly across those points without drawing distorted artifacts. The alternative, `nonpositive='clip'`, clamps non-positive numbers to an extremely small positive number.
- **D is incorrect:** `filter` is not a parameter accepted by `set_yscale()`.

</details>

---

### Question 39

An epidemiology model simulates infectious disease spread where infected case counts grow exponentially according to $I(t) = I_0 e^{r t}$.

When plotted with time $t$ on a linear horizontal axis and case count $I(t)$ on a logarithmic vertical axis (a semi-log plot), what visual characteristic identifies constant exponential growth?

- [ ] A) An oscillating sinusoidal wave.
- [ ] B) A straight upward-sloping line whose constant gradient is proportional to the growth rate $r$.
- [ ] C) A step function with discontinuous vertical jumps.
- [ ] D) An inverted U-shaped parabola.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A straight upward-sloping line whose constant gradient is proportional to the growth rate $r$.

**Explanation:**

- **A is incorrect:** Sinusoidal oscillations indicate periodic cycles, not monotonic exponential growth.
- **B is correct:** Taking the natural log of $I(t) = I_0 e^{r t}$ yields $\ln I(t) = \ln I_0 + r t$. When the y-axis is logarithmic and the x-axis is linear, this maps directly to a straight line with slope $r$. If the slope begins to bend downward or flatten, it provides immediate visual proof that the transmission rate has slowed and the epidemic is sub-exponential.
- **C is incorrect:** Step functions denote discrete transitions, not continuous exponential processes.
- **D is incorrect:** Parabolas represent quadratic functions, not exponential curves.

</details>

---

### Question 40

Which of the following statements regarding logarithmic scales in Matplotlib are **correct**?

1. Setting `ax.set_yscale('log')` renders equal physical distances on the canvas for each order of magnitude (e.g., $10^1$ to $10^2$ occupies the same height as $10^2$ to $10^3$).
2. Matplotlib supports custom logarithmic bases such as `base=2` or `base=math.e`.
3. Standard logarithmic scales cannot plot negative numbers or zero without masking or specialized thresholding.
4. On a base-10 log scale, the visual distance from $1$ to $10$ is twice the visual distance from $10$ to $100$.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because each decade occupies identical physical length; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all correct. In logarithmic projections, each multiplicative decade occupies an identical physical length; custom bases like `base=2` are fully supported; and non-positive numbers are mathematically invalid and must be masked or transformed using `symlog`.

</details>

---
