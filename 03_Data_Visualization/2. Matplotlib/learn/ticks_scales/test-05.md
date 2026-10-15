# Matplotlib Ticks & Scales Objective Test — Set 05

---

### Question 41

A machine learning diagnostic script plots regression residuals ($y - \hat{y}$) across a test set. The residuals span from large negative errors ($-10^4$) to large positive errors ($+10^4$), with thousands of near-perfect predictions clustering closely around zero ($[-0.01, +0.01]$).

Why does applying a standard logarithmic scale (`ax.set_yscale('log')`) fail catastrophically for this dataset, and what specialized scale solves this problem?

- [ ] A) Standard log scales invert the colors of the plot; the solution is to use `ax.set_yscale('linear_log')`.
- [ ] B) Standard log scales only work for integers; the solution is to use `ax.set_yscale('float_log')`.
- [ ] C) Standard log scales cannot plot numbers greater than $1000$; the solution is to use `ax.set_yscale('big_log')`.
- [ ] D) Standard log scales cannot evaluate negative numbers or zero ($\log(x)$ is undefined for $x \le 0$); the solution is Symmetrical Log (`'symlog'`), which maps a linear region around zero and logarithmic regions for positive and negative tails.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Standard log scales cannot evaluate negative numbers or zero ($\log(x)$ is undefined for $x \le 0$); the solution is Symmetrical Log (`'symlog'`), which maps a linear region around zero and logarithmic regions for positive and negative tails.

**Explanation:**

- **A is incorrect:** `'linear_log'` is not a recognized scale name in Matplotlib.
- **B is incorrect:** `'float_log'` does not exist in the Matplotlib API.
- **C is incorrect:** `'big_log'` is a fictitious scale parameter.
- **D is correct:** When data crosses zero into negative territory, a pure logarithmic transformation is mathematically impossible because $\log(x)$ approaches $-\infty$ as $x \to 0^+$ and is undefined for $x \le 0$. Matplotlib's Symmetrical Log (`'symlog'`) scale solves this by partitioning the axis into three distinct regions: a linear scale around zero ($[-\text{linthresh}, +\text{linthresh}]$) where values pass smoothly through zero, flanked by logarithmic scales on both the positive ($x > \text{linthresh}$) and negative ($x < -\text{linthresh}$) sides.

</details>

---

### Question 42

An engineer configures a symmetrical log scale for financial profit-and-loss data:

```python
ax.set_yscale('symlog', linthresh=0.1)
```

What does the `linthresh` parameter explicitly define in this coordinate system?

- [ ] A) The maximum pixel width of the vertical axis spine.
- [ ] B) The line thickness of the plotted curves.
- [ ] C) The range $[-\text{linthresh}, +\text{linthresh}]$ (here $[-0.1, +0.1]$) within which the axis operates as a standard linear scale, avoiding the logarithmic singularity at zero.
- [ ] D) The threshold of points below which data is discarded as noise.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) The range $[-	ext{linthresh}, +	ext{linthresh}]$ (here $[-0.1, +0.1]$) within which the axis operates as a standard linear scale, avoiding the logarithmic singularity at zero.

**Explanation:**

- **A is incorrect:** `linthresh` is not a pixel dimension or spine stroke width.
- **B is incorrect:** `linthresh` does not control the line thickness of plotted data curves.
- **C is correct:** `linthresh` (linear threshold) is the foundational parameter of the `symlog` scale. It establishes the half-width of the linear zero-crossing corridor: for all values $|y| \le \text{linthresh}$, coordinates are transformed linearly. Once $|y| > \text{linthresh}$, the transformation transitions smoothly into logarithmic scaling. Choosing an appropriate `linthresh` guarantees that values near zero remain fully visible without collapsing into asymptotic distortion.
- **D is incorrect:** `symlog` does not filter out or discard data points.

</details>

---

### Question 43

In conjunction with `linthresh`, the `symlog` scale accepts a parameter named `linscale`:

```python
ax.set_yscale('symlog', linthresh=1.0, linscale=2.0)
```

What visual aspect of the plot does `linscale` control?

- [ ] A) The font scale factor applied to negative tick numbers.
- [ ] B) The relative physical size allocated to the linear region on the canvas, measured in units of logarithmic decades (e.g., `linscale=2.0` allocates twice the height of a single log decade to the linear range).
- [ ] C) The number of CPU threads used to render linear interpolation.
- [ ] D) The line style of the zero baseline.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The relative physical size allocated to the linear region on the canvas, measured in units of logarithmic decades (e.g., `linscale=2.0` allocates twice the height of a single log decade to the linear range).

**Explanation:**

- **A is incorrect:** `linscale` does not scale tick label font sizes.
- **B is correct:** By default (`linscale=1.0`), the entire linear corridor $[-\text{linthresh}, +\text{linthresh}]$ is allocated a physical display height equivalent to one decade (a factor of 10) of the adjoining logarithmic scale. Increasing `linscale` (e.g., `linscale=2.0`) stretches the linear portion of the axis so that it occupies more vertical space relative to the logarithmic decades, preventing high-density data near zero from appearing overly compressed.
- **C is incorrect:** `linscale` does not affect multi-threading or CPU rendering.
- **D is incorrect:** Zero baseline line style is governed by `axhline()`, not `linscale`.

</details>

---

### Question 44

A deep learning practitioner plots the weight gradients of a deep neural network across layers. The gradients vary from $-10^2$ to $+10^2$, but $95\%$ of all weights have tiny gradients between $-10^{-3}$ and $+10^{-3}$.

If the practitioner plots this on a standard linear scale, what visual failure occurs?

- [ ] A) The large gradients at $\pm 10^2$ force the linear axis limits to expand outward, squashing all the subtle gradient differences around $\pm 10^{-3}$ into an indistinguishable flat horizontal smear along the zero baseline.
- [ ] B) The gradients will automatically wrap around the edges of the canvas like a torus.
- [ ] C) The plot fails to render and displays a blank screen.
- [ ] D) Negative gradients are plotted as positive gradients.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The large gradients at $\pm 10^2$ force the linear axis limits to expand outward, squashing all the subtle gradient differences around $\pm 10^{-3}$ into an indistinguishable flat horizontal smear along the zero baseline.

**Explanation:**

- **A is correct:** On a linear scale, physical distance is strictly proportional to absolute numerical difference. Spanning $[-100, +100]$ requires $200$ units of dynamic range. A critical variation between $0.0001$ and $0.001$ occupies a mere $0.0009 / 200 \approx 0.00045\%$ of the axis height—rendering vanishing gradients completely invisible. Setting `symlog` with `linthresh=1e-4` allows both the macroscopic outliers at $\pm 10^2$ and the microscopic variations at $10^{-4}$ to be inspected simultaneously.
- **B is incorrect:** Linear coordinates do not wrap around canvas edges.
- **C is incorrect:** Linear plots render normally without blanking out.
- **D is incorrect:** Linear scaling never mutates signs or takes absolute values.

</details>

---

### Question 45

In binary classification evaluation and rare-event modeling (such as fraud detection or ad click-through rate prediction), a data scientist wants to visualize probabilities $p$ that concentrate intensely near $0.0$ ($p \approx 10^{-5}$) and near $1.0$ ($p \approx 0.9999$).

Which specialized non-linear scale maps probabilities using the transformation $f(p) = \log\left(\frac{p}{1 - p}\right)$, stretching both extremes ($p \to 0$ and $p \to 1$) while compressing the middle near $p = 0.5$?

- [ ] A) `ax.set_yscale('sigmoid')`
- [ ] B) `ax.set_yscale('softmax')`
- [ ] C) `ax.set_yscale('probit_linear')`
- [ ] D) `ax.set_yscale('logit')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.set_yscale('logit')`

**Explanation:**

- **A is incorrect:** `'sigmoid'` is not a registered scale string in Matplotlib.
- **B is incorrect:** `'softmax'` does not exist as a Matplotlib scale.
- **C is incorrect:** `'probit_linear'` is not a built-in Matplotlib scale.
- **D is correct:** Matplotlib includes a built-in `'logit'` scale (`matplotlib.scale.LogitScale`). The logit transform maps the probability interval $(0, 1)$ onto $(-\infty, +\infty)$ via the log-odds formula $\operatorname{logit}(p) = \ln(p / (1 - p))$. This transformation expands both tails symmetrically: probabilities close to zero ($10^{-4}, 10^{-3}$) and probabilities close to one ($0.999, 0.9999$) are visually magnified, providing unparalleled diagnostic clarity for highly imbalanced classification probabilities.

</details>

---

### Question 46

What is the strict mathematical domain restriction enforced by Matplotlib when using the `logit` scale (`ax.set_yscale('logit')`)?

- [ ] A) Data must consist entirely of negative real numbers.
- [ ] B) Data can be any real number from $-\infty$ to $+\infty$.
- [ ] C) Data values must reside strictly within the open probability interval $(0, 1)$; values $\le 0.0$ or $\ge 1.0$ are invalid and cause masking, clipping, or errors.
- [ ] D) Data values must be prime integers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Data values must reside strictly within the open probability interval $(0, 1)$; values $\le 0.0$ or $\ge 1.0$ are invalid and cause masking, clipping, or errors.

**Explanation:**

- **A is incorrect:** The logit scale operates on positive probability inputs, not negative-only data.
- **B is incorrect:** The domain of logit is bounded in $(0, 1)$, not unbounded across $(-\infty, +\infty)$.
- **C is correct:** The logit function $\ln(p / (1-p))$ has vertical asymptotes at both $p = 0$ (where it approaches $-\infty$) and $p = 1$ (where it approaches $+\infty$). Consequently, the domain of the logit transformation is strictly the open unit interval $(0, 1)$. Attempting to plot points where $p \le 0$ or $p \ge 1$ triggers domain clipping or masking in Matplotlib's `LogitTransform`.
- **D is incorrect:** Logit scale applies to continuous floating-point probabilities, not prime integers.

</details>

---

### Question 47

A calibrated classifier produces predicted probabilities on a test set. When evaluating the calibration curve with `ax.set_xscale('logit')` and `ax.set_yscale('logit')`, what does the line $y = x$ represent?

- [ ] A) A completely random, uninformative classifier with $AUC = 0.5$.
- [ ] B) Perfect empirical calibration, where a predicted probability of $p$ matches an observed empirical frequency of exactly $p$ across all orders of probability magnitude.
- [ ] C) An overfitted decision tree that has memorized the training labels.
- [ ] D) A model that predicts the negative class with $100\%$ confidence.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Perfect empirical calibration, where a predicted probability of $p$ matches an observed empirical frequency of exactly $p$ across all orders of probability magnitude.

**Explanation:**

- **A is incorrect:** On calibration diagrams, $y = x$ represents perfect calibration; random classification is represented by $y = x$ on ROC curves.
- **B is correct:** In probability calibration curves (reliability diagrams), the horizontal axis represents predicted probability $\hat{p}$ and the vertical axis represents the true empirical positive fraction $y$. The identity line $y = x$ represents perfect calibration: whenever the model predicts a probability of $0.001$, the event occurs $0.1\%$ of the time; whenever it predicts $0.999$, it occurs $99.9\%$ of the time. Plotting on a logit-logit grid allows calibration to be audited with equal visual precision in both ultra-rare and ultra-common probability regimes.
- **C is incorrect:** Overfitted memorization produces irregular step-functions, not the identity line.
- **D is incorrect:** Constant negative predictions collapse to the horizontal baseline $y = 0$.

</details>

---

### Question 48

If a research team invents a proprietary non-linear coordinate transformation (for example, a specialized atmospheric pressure scale $f(z) = e^{-z / H}$), how can they register and use this custom transformation natively with `ax.set_yscale("atmosphere")`?

- [ ] A) By creating a custom subclass of `matplotlib.scale.ScaleBase` implementing the forward and inverse `Transform` classes, and registering it globally via `matplotlib.scale.register_scale(MyAtmosphereScale)`.
- [ ] B) By editing the Python C-extension source code of Matplotlib.
- [ ] C) By passing a lambda function directly to `ax.set_yscale(lambda z: np.exp(-z / H))`.
- [ ] D) Matplotlib scales are hardcoded and cannot be extended under any circumstances.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) By creating a custom subclass of `matplotlib.scale.ScaleBase` implementing the forward and inverse `Transform` classes, and registering it globally via `matplotlib.scale.register_scale(MyAtmosphereScale)`.

**Explanation:**

- **A is correct:** Matplotlib's scale architecture is completely modular. Developers can subclass `matplotlib.scale.ScaleBase`, define custom forward and inverse coordinate projection transforms (subclassing `matplotlib.transforms.Transform`), implement default tick locators/formatters, and register the new scale string via `matplotlib.scale.register_scale(CustomScaleClass)`. Once registered, any axis can activate it via `ax.set_yscale("atmosphere")`.
- **B is incorrect:** Modifying C-extensions is unnecessary; the scale architecture is fully exposed in Python.
- **C is incorrect:** `set_yscale()` accepts string names of registered scales, not bare lambda functions (which raise a `ValueError`).
- **D is incorrect:** Matplotlib's scale registry is explicitly designed for user extensibility.

</details>

---

### Question 49

In a `symlog` plot spanning from $-10^3$ to $+10^3$ with `linthresh=1`, what major tick sequence is generated by default across the zero transition?

- [ ] A) Only negative ticks are shown: `[-1000, -100, -10]`.
- [ ] B) Every integer between $-1000$ and $+1000$.
- [ ] C) Only the single tick `[0]`.
- [ ] D) A symmetric sequence containing negative decades, zero, and positive decades, such as `[-1000, -100, -10, -1, 0, 1, 10, 100, 1000]`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) A symmetric sequence containing negative decades, zero, and positive decades, such as `[-1000, -100, -10, -1, 0, 1, 10, 100, 1000]`.

**Explanation:**

- **A is incorrect:** `symlog` spans both positive and negative poles, not just negative values.
- **B is incorrect:** Symmetrical log scales do not enumerate all 2,000 integers linearly.
- **C is incorrect:** Zero is only one of many landmark ticks generated across the full range.
- **D is correct:** Matplotlib's `SymmetricalLogLocator` generates ticks that step symmetrically through powers of the base in the negative domain ($-10^3, -10^2, -10^1, -10^0$), passes through the linear zero anchor ($0$), and continues through powers of the base in the positive domain ($10^0, 10^1, 10^2, 10^3$).

</details>

---

### Question 50

Which of the following statements regarding specialized Matplotlib scale transformations are **correct**?

1. The `symlog` scale uses a linear transformation inside $|x| \le \text{linthresh}$ to prevent the logarithmic singularity at zero.
2. In a `symlog` axis, `linscale` controls the visual proportion allocated to the linear region relative to one logarithmic decade.
3. The `logit` scale maps the probability value $p = 0.5$ to the origin point $0.0$ on the transformed coordinate axis.
4. The `logit` scale is fully capable of plotting negative numbers such as $-0.5$ without errors.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 1, 2, and 3 only
- [ ] D) 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because logit inputs are bounded in $(0, 1)$; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is correct:** Statements 1, 2, and 3 are all correct. `symlog` enforces linear behavior within $|x| \le \text{linthresh}$; `linscale` defines the display length of the linear segment relative to a log decade; and $\operatorname{logit}(0.5) = \ln(1) = 0.0$, placing $50\%$ probability at the center of the axis.
- **D is incorrect:** Statement 4 is false because negative values cannot be passed to the logit transform.

</details>

---
