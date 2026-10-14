# Matplotlib Line Plot Objective Test — Set 08

---

### 71. Question

An epidemiologist is tracking an outbreak exhibiting exponential transmission ($y = a \cdot e^{kt}$). When plotted on a standard linear vertical scale, the curve curves sharply upward, making it difficult to distinguish whether the transmission rate is constant, accelerating, or decelerating.

Which command converts the vertical axis to a logarithmic scale, transforming exponential growth into a straight line?

- [ ] **A.** `plt.yscale('exponential')`
- [ ] **B.** `plt.transform_axis('y', 'linear_to_log')`
- [ ] **C.** `plt.yscale('log')`
- [ ] **D.** `plt.y_linear(False)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `'exponential'` is not a recognized scale name in Matplotlib.
- **Option B (Incorrect):** `transform_axis` is not a Matplotlib Pyplot method.
- **Option C (Correct):** Calling `plt.yscale('log')` transforms the vertical axis to a base-10 logarithmic scale. Taking the logarithm of exponential growth $y = a \cdot e^{kt}$ yields $\ln(y) = \ln(a) + kt$, which is a linear equation with slope $k$. Thus, any exponential trajectory renders as a straight line on a semi-logarithmic chart, where the slope directly reflects the growth rate.
- **Option D (Incorrect):** `plt.y_linear()` does not exist.

</details>

---

### 72. Question

A financial analyst applies a logarithmic vertical scale to a portfolio balance dataset that contains several zero and negative values representing net debt:

```python
balances = [5000, 1200, 450, 0, -300, 800]
plt.plot(balances)
plt.yscale('log')
plt.show()
```

What happens when Matplotlib attempts to render this plot?

- [ ] **A.** It calculates complex imaginary numbers and plots them on a secondary plane.
- [ ] **B.** Non-positive values ($\le 0$) cannot be log-transformed ($\log(x)$ is mathematically undefined for $x \le 0$), so Matplotlib either clips/masks them with a `UserWarning` or drops them from the visible curve.
- [ ] **C.** Matplotlib automatically adds a constant offset of $+10,000$ to all data points.
- [ ] **D.** The chart inverts the negative numbers to positive values and displays them in purple.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib 2D line rendering operates strictly in the real number domain; it does not map imaginary planes.
- **Option B (Correct):** Because the logarithm is only defined for strictly positive real numbers ($x > 0$), zero and negative values cannot be placed on a standard logarithmic scale. Matplotlib masks non-positive coordinates, emitting a warning such as `UserWarning: Data has no positive values, and therefore cannot be log-scaled` or dropping invalid points, resulting in broken line segments.
- **Option C (Incorrect):** Matplotlib never silently shifts or adds offsets to user data.
- **Option D (Incorrect):** Matplotlib never negates data points without user instruction.

</details>

---

### 73. Question

An engineer needs to display a signal on a logarithmic scale to reveal variations across several orders of magnitude, but the dataset crosses through zero into negative values (spanning from $-1000$ to $+1000$). Which specialized scale accommodates positive, zero, and negative values?

- [ ] **A.** `plt.yscale('symlog')` (Symmetrical Log scale)
- [ ] **B.** `plt.yscale('bilog')`
- [ ] **C.** `plt.yscale('signed_log')`
- [ ] **D.** `plt.yscale('inverse_log')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Matplotlib provides the **`symlog`** (symmetrical log) scale, which applies a logarithmic scale to both positive and negative values while defining a small linear threshold (`linthresh`) region around zero. This avoids the singularity at zero while allowing orders-of-magnitude scaling in both positive and negative domains.
- **Option B (Incorrect):** `'bilog'` is not a valid Matplotlib scale name.
- **Option C (Incorrect):** `'signed_log'` is not recognized; the proper term is `'symlog'`.
- **Option D (Incorrect):** `'inverse_log'` is not a valid scale.

</details>

---

### 74. Question

Which Pyplot convenience function plots data directly with a **linear horizontal axis** and a **logarithmic vertical axis** in a single call?

- [ ] **A.** `plt.semilogx(x, y)`
- [ ] **B.** `plt.loglog(x, y)`
- [ ] **C.** `plt.linear_log(x, y)`
- [ ] **D.** `plt.semilogy(x, y)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.semilogx()` applies a log scale to the horizontal ($x$) axis and keeps the vertical axis linear.
- **Option B (Incorrect):** `plt.loglog()` sets **both** the horizontal and vertical axes to logarithmic scales.
- **Option C (Incorrect):** `plt.linear_log()` does not exist in Matplotlib.
- **Option D (Correct):** `plt.semilogy(x, y)` is the dedicated Pyplot shortcut for creating a semi-log plot where the $y$-axis is logarithmic and the $x$-axis remains linear.

</details>

---

### 75. Question

A management consultant presents two charts to an executive team: Chart 1 has a **linear y-axis**, and Chart 2 has a **logarithmic y-axis**. Both charts display a straight line with a constant positive upward slope.

How does the real-world trajectory of the underlying metric differ between Chart 1 and Chart 2?

- [ ] **A.** In Chart 1, the metric grows by a constant absolute amount each period ($\Delta y = c$); in Chart 2, the metric grows by a constant percentage/rate each period ($\% \Delta y = r$, compound exponential growth).
- [ ] **B.** In Chart 1, growth is exponential; in Chart 2, growth is linear.
- [ ] **C.** Both charts represent identical flat metrics with zero growth.
- [ ] **D.** Chart 2 represents decreasing values over time.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** On a linear scale, a constant slope represents constant additive increments ($\frac{\Delta y}{\Delta x} = \text{constant}$, arithmetic progression). On a logarithmic scale, equal vertical distances represent equal multiplicative ratios (e.g., $10 \to 100 \to 1000$), so a straight line indicates constant percentage compound growth ($y = a \cdot e^{kx}$).
- **Option B (Incorrect):** This completely inverts the mathematical interpretation of linear vs logarithmic scales.
- **Option C (Incorrect):** Upward sloping lines represent positive growth in both charts.
- **Option D (Incorrect):** A positive slope on a log scale represents multiplying growth, not a decline.

</details>

---

### 76. Question

An investor wants to compare the historical growth trajectories of two tech stocks: Stock Alpha trades around \$3,200 per share, while Stock Beta trades around \$25 per share.

Plotting raw dollar share prices on a linear chart completely flattens Stock Beta. What mathematical normalization technique should the investor apply before calling `plt.plot()` to compare their true performance?

- [ ] **A.** Subtract the mean from each stock: $y - \mu$
- [ ] **B.** Replace all values in Stock Beta with random numbers between 3,000 and 3,200
- [ ] **C.** Divide each series by its initial value and multiply by 100: $y_{\text{norm}} = \left(\frac{y}{y[0]}\right) \times 100$ (Rebasing to a common index of 100)
- [ ] **D.** Square the values of Stock Beta

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Subtracting the mean preserves the absolute dollar spread (e.g., \$50 swings vs \$1 swings), failing to equalize percentage growth.
- **Option B (Incorrect):** Fabricating fake data violates scientific and visualization ethics.
- **Option C (Correct):** Rebasing both series to a base 100 index ($y_{\text{indexed}} = \frac{y_t}{y_0} \times 100$) anchors both stocks at 100 at time $t=0$. Subsequent points cleanly reflect cumulative percentage return (e.g., an index value of 150 indicates a +50% gain for both stocks), allowing direct visual comparison on the same axes.
- **Option D (Incorrect):** Squaring alters the growth rate non-linearly and distorts financial returns.

</details>

---

### 77. Question

A warehouse logistics team is tracking the number of pallets stored in a holding bay over time. Pallets arrive and depart instantly in discrete batches (e.g., $+20$ pallets at 10:00 AM, $-15$ pallets at 1:30 PM). Between these events, inventory remains strictly constant.

Which plotting function or style correctly communicates this piecewise constant behavior without falsely implying gradual, continuous pallet movement?

- [ ] **A.** Standard linear `plt.plot(time, pallets)`
- [ ] **B.** Spline interpolation with `scipy.interpolate.make_interp_spline`
- [ ] **C.** `plt.scatter(time, pallets)` connected with bezier curves
- [ ] **D.** `plt.step(time, pallets, where='post')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Standard `plt.plot()` draws diagonal lines connecting event timestamps, falsely implying that pallets were steadily arriving or departing at a continuous rate throughout the intermediate hours.
- **Option B (Incorrect):** Spline interpolation introduces artificial wavy curves and non-existent intermediate values.
- **Option C (Incorrect):** Smooth curves falsely imply gradual changes.
- **Option D (Correct):** `plt.step()` (or `plt.plot(..., drawstyle='steps-post')`) renders horizontal lines for periods of stability followed by vertical jumps at transaction timestamps. This faithfully mirrors piecewise constant physical reality, such as inventory counts, bank balances, or server thread allocations.

</details>

---

### 78. Question

On a standard base-10 logarithmic scale spanning from $10^1 = 10$ to $10^3 = 1000$, what is the **geometric midpoint** that sits at the exact physical center of the axis between 10 and 1000?

- [ ] **A.** $100$
- [ ] **B.** $505$
- [ ] **C.** $250$
- [ ] **D.** $500$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** On a logarithmic scale, physical distance is proportional to the difference in exponents: $\log_{10}(10) = 1$ and $\log_{10}(1000) = 3$. The midpoint in log space is $\frac{1 + 3}{2} = 2$, which corresponds to $10^2 = \mathbf{100}$. Therefore, the value 100 sits exactly halfway between 10 and 1000 on a log axis.
- **Option B (Incorrect):** $505$ is the **arithmetic mean** ($\frac{10 + 1000}{2} = 505$), which sits at the center of a *linear* scale.
- **Option C (Incorrect):** 250 is neither the arithmetic nor geometric midpoint.
- **Option D (Incorrect):** 500 is close to the linear average, not the logarithmic midpoint.

</details>

---

### 79. Question

Why do many data visualization experts caution against using dual-axis charts to compare two independent time series, recommending normalized index line charts instead?

- [ ] **A.** Dual-axis charts cannot be exported to PNG format.
- [ ] **B.** Computational overhead of a second Axes slows rendering by 500%.
- [ ] **C.** Dual-axis charts cause Matplotlib to throw deprecation warnings.
- [ ] **D.** By manipulating the independent minimum and maximum limits of the two vertical axes, a chart designer can arbitrarily stretch, compress, and force visual intersection points between two completely unrelated series, creating an optical illusion of strong correlation or causation.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib exports dual-axis figures to all standard image formats seamlessly.
- **Option B (Incorrect):** Computational overhead of a second Axes is negligible.
- **Option C (Incorrect):** `twinx()` is a fully supported, non-deprecated core feature.
- **Option D (Correct):** Dual-axis charts have two independent vertical degrees of freedom. By artificially tuning `ax1.set_ylim()` and `ax2.set_ylim()`, a designer can make unrelated series appear to track each other perfectly or cross at arbitrary dates. Indexing both series to a percentage baseline (e.g., base 100) eliminates arbitrary scaling manipulation by putting both series on an honest, unified scale.

</details>

---

### 80. Question

Consider the following statements regarding logarithmic scaling and stepped line plots:

1. `plt.yscale('log')` uses base 10 by default, but the base can be modified (e.g., `base=2`).
2. Stepped plots created with `plt.step()` support `where='pre'`, `where='post'`, and `where='mid'` to determine whether the vertical step occurs before, after, or halfway between points.
3. Setting `plt.yscale('log')` automatically sets `plt.xscale('log')`.
4. Rebasing time series to an index of 100 allows direct percentage comparison of trends that differ by orders of magnitude in raw units.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `plt.yscale('log', base=2)` is supported and configures a binary logarithmic scale.
- **Statement 2 is correct:** The `where` argument controls step transition placement (`'pre'`, `'post'`, or `'mid'`).
- **Statement 3 is incorrect:** Setting `plt.yscale('log')` modifies only the vertical axis; horizontal axis scaling remains unaffected.
- **Statement 4 is correct:** Indexing to 100 normalizes disparate units into a common relative percentage scale.
- **Conclusion:** Statements 1, 2, and 4 are correct, making **Option C** the correct answer.

</details>

---
