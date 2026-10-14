# Matplotlib Area Plot Objective Test — Set 03

---

### 21. Question

A data analyst fills between two curves representing server response time bounds:

```python
x = np.array([1, 2, 3, 4, 5])
fast = np.array([10, 12, 11, 13, 10])
slow = np.array([30, 35, 28, 40, 32])

plt.fill_between(x, fast, slow, color="gold", alpha=0.5)
plt.plot(x, fast, "g-", label="Best Case")
plt.plot(x, slow, "r-", label="Worst Case")
plt.legend()
plt.ylabel("Response Time (ms)")
```

At x=4, the shaded gold region spans from y=13 to y=40. What does this shaded region visually represent?

- [ ] **A.** The average response time at each request count
- [ ] **B.** The plausible range of server response times between the best-case and worst-case boundaries
- [ ] **C.** The cumulative total response time from x=1 to x=4
- [ ] **D.** The standard deviation of response times at x=4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The average response time would be a single curve at each x-value, not a shaded band. Visualising the average requires computing `(fast + slow) / 2` and plotting it as a line.
- **Option B (Correct):** `fill_between(x, fast, slow)` shades the entire vertical span between the `fast` (best-case/lower bound) and `slow` (worst-case/upper bound) response time curves. At x=4, this band spans 13 ms to 40 ms, representing the 27 ms range within which actual server response times could fall. The shaded area gives a reader an immediate visual sense of how variable or reliable the server performance is.
- **Option C (Incorrect):** Cumulative totals would require computing `np.cumsum` and plotting a running sum. `fill_between` does not accumulate values across x-positions.
- **Option D (Incorrect):** Standard deviation requires statistical computation across multiple samples. The shaded band here is simply the geometric difference between two explicit boundary arrays, not a statistical spread measure.

</details>

---

### 22. Question

A researcher draws the area under a probability density function (PDF) curve:

```python
from scipy.stats import norm

x = np.linspace(-4, 4, 500)
pdf = norm.pdf(x, loc=0, scale=1)   # Standard normal PDF

fig, ax = plt.subplots()
ax.plot(x, pdf, "k-", lw=2)
ax.fill_between(x, pdf, alpha=0.3, color="purple")
ax.set_xlabel("z-score")
ax.set_ylabel("Density")
```

What does the total shaded purple area under the standard normal PDF equal?

- [ ] **A.** 0.5 (half the distribution is positive)
- [ ] **B.** 0 (the mean of the distribution)
- [ ] **C.** 1.0 (the total probability integrates to 1)
- [ ] **D.** The value depends on the `alpha=0.3` setting

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 0.5 is the probability of the standard normal distribution being positive (i.e., the area from z=0 to z=+∞). The code fills the area under the entire curve from −4 to +4, which spans nearly the full distribution.
- **Option B (Incorrect):** 0 is the mean (location parameter) of the standard normal, not the integral of its PDF. The area under a PDF is always 1 for any valid probability distribution.
- **Option C (Correct):** By definition, the integral of any valid probability density function over its entire support equals 1.0 — representing 100% total probability. The standard normal PDF from −∞ to +∞ integrates to exactly 1. Over the range −4 to +4 used here, the area is approximately 0.99994, practically equal to 1.0.
- **Option D (Incorrect):** `alpha` is a display parameter that controls the visual opacity of the shaded region in the rendered image. It has no effect on the mathematical area under the curve. The actual geometric area remains ≈1.0 regardless of `alpha`.

</details>

---

### 23. Question

A data scientist wants to shade the region between the 10th percentile and 90th percentile of a bootstrapped prediction:

```python
x = np.arange(0, 50)
median = 100 + 2 * x
p10 = median - (1.5 + 0.05 * x)     # 10th percentile
p90 = median + (1.5 + 0.05 * x)     # 90th percentile

plt.fill_between(x, p10, p90, color="royalblue", alpha=0.25, label="P10–P90 Band")
plt.plot(x, median, "b-", lw=2, label="Median Forecast")
```

At `x=20`, what is the total width of the P10–P90 shaded band?

- [ ] **A.** 1.5 units
- [ ] **B.** 3.0 units
- [ ] **C.** 5.0 units
- [ ] **D.** 10.0 units

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 1.5 units is half the initial band width at x=0 (where the half-width = 1.5 + 0.05×0 = 1.5, full width = 3.0). At x=20, the width is larger.
- **Option B (Incorrect):** 3.0 units is the full width at x=0 (half-width=1.5, full=3.0). At x=20, `half-width = 1.5 + 0.05×20 = 1.5 + 1.0 = 2.5`, so full width = 5.0.
- **Option C (Correct):** At x=20, the half-width (from median to each boundary) = `1.5 + 0.05×20 = 1.5 + 1.0 = 2.5`. The full width of the band = `p90 - p10 = 2 × 2.5 = 5.0` units. This represents the spread of the 80% prediction interval at x=20.
- **Option D (Incorrect):** 10.0 units would correspond to x=85 where half-width = `1.5 + 0.05×85 = 5.75` and full width = 11.5. No clean x-value gives exactly 10.0.

</details>

---

### 24. Question

A data analyst writes:

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.fill_between(x, y1, y2, color="green", alpha=0.4)
plt.plot(x, y1, label="sin(x)")
plt.plot(x, y2, label="cos(x)")
```

At certain x-values, `sin(x) > cos(x)`, and at others, `cos(x) > sin(x)`. How does `fill_between` handle these inversions?

- [ ] **A.** It only fills where `y1 > y2` and ignores sections where `y2 > y1`
- [ ] **B.** It fills the region between the two curves at every x-value, regardless of which curve is on top — the result is a continuous green band that alternates between being above sin(x) and above cos(x)
- [ ] **C.** It raises an error when the two curves cross each other
- [ ] **D.** It plots only the envelope (the outer boundary) and fills to the x-axis

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Without the `where` parameter, `fill_between` does not filter based on which curve is higher. It fills between `y1` and `y2` everywhere across the x-range, regardless of the relative ordering of `y1` and `y2`.
- **Option B (Correct):** `fill_between(x, y1, y2)` always fills the region between the two curves at every x. Where `sin(x) > cos(x)` the green band appears above `cos(x)` up to `sin(x)`; where `cos(x) > sin(x)` the band appears above `sin(x)` up to `cos(x)`. The band continuously covers the gap between the two curves throughout, creating a visual representation of the difference between them.
- **Option C (Incorrect):** `fill_between` handles curve crossings gracefully. The crossing points are just points where y1=y2 (the band has zero height). No error is raised.
- **Option D (Incorrect):** `fill_between(x, y1, y2)` with both `y1` and `y2` specified does not involve the x-axis at all. It fills between the two curves. Filling to the x-axis would require one of the arguments to be `0`.

</details>

---

### 25. Question

A researcher visualises two overlapping uncertainty bands from two different models:

```python
plt.fill_between(x, lower_A, upper_A, color="blue", alpha=0.25, label="Model A")
plt.fill_between(x, lower_B, upper_B, color="red", alpha=0.25, label="Model B")
```

In regions where the two bands overlap, what colour does the overlapping area appear?

- [ ] **A.** White (because the two translucent fills cancel each other)
- [ ] **B.** A purple-ish blend resulting from the additive mixing of blue (alpha=0.25) and red (alpha=0.25) over the white background
- [ ] **C.** The second colour (red) completely overrides the first (blue) in the overlap region
- [ ] **D.** Black (maximum opacity when two fills overlap)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Translucent fills do not cancel each other. Alpha compositing blends the colours additively over the background, not subtractively.
- **Option B (Correct):** With `alpha=0.25`, each fill is 75% transparent. In the overlap region, the background (white) is seen through both fills. The compositing result mixes blue and red over white, producing a purple/violet hue. The exact shade depends on the alpha compositing formula. This is a useful visual indicator that both models agree on uncertainty in the overlap zone.
- **Option C (Incorrect):** With `alpha=0.25`, the second (red) fill is not opaque — you can still see the blue fill underneath. `alpha=1.0` would be needed for the second fill to completely hide the first.
- **Option D (Incorrect):** Two fills with alpha=0.25 each do not combine to produce black. The maximum possible combined opacity is bounded by the compositing formula and with blue and red inputs would produce a dark purple at most, not black.

</details>

---

### 26. Question

A developer uses `fill_between` with the keyword argument `step="post"`:

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 2, 7, 4])

plt.fill_between(x, y, step="post", color="teal", alpha=0.5)
```

What does `step="post"` change about the filled region?

- [ ] **A.** The fill uses smooth cubic spline interpolation between data points
- [ ] **B.** The fill boundary is drawn as a step function where the value changes at the END (right side) of each interval — the value holds flat until the next data point, then steps
- [ ] **C.** `step="post"` is not a valid argument for `fill_between`
- [ ] **D.** The fill is shifted one x-step to the right relative to the data points

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Smooth cubic spline interpolation is not what `step` does. The `step` parameter creates discrete step-function fills, not smooth curves.
- **Option B (Correct):** The `step` parameter in `fill_between` mirrors the `drawstyle` option in `plt.step`. With `step="post"`, the value from each data point extends horizontally to the RIGHT until the next x-value, where it then steps up or down. In other words, the y-value at position `i` holds constant over the interval `[x[i], x[i+1])`. This is appropriate for visualising data that remains constant until updated — like server metrics sampled at discrete intervals or stock price fills.
- **Option C (Incorrect):** `step` is a valid documented parameter of `fill_between`. It accepts `None` (default, linear interpolation), `"pre"`, `"post"`, or `"mid"`.
- **Option D (Incorrect):** The fill is not spatially shifted. `step="post"` controls the shape of the boundary (step vs. linear) but keeps the fill aligned with the actual x-positions of the data points.

</details>

---

### 27. Question

A developer places two `fill_between` artists on the same axes. The first fill uses `alpha=0.6` and the second uses `alpha=0.6` but is placed on top of the first (higher zorder). Which statement about the effective visual opacity in the overlap zone is correct?

- [ ] **A.** The overlap appears with `alpha = 0.6 + 0.6 = 1.2`, which is clamped to 1.0 (fully opaque)
- [ ] **B.** The overlap appears darker than either fill alone because alpha compositing stacks the opacities: effective opacity ≈ `1 − (1−0.6)² = 1 − 0.16 = 0.84`
- [ ] **C.** The overlap appears with `alpha = 0.6 × 0.6 = 0.36` (multiplicative blending)
- [ ] **D.** The overlap appears identical to a single fill with `alpha=0.6` because the two layers merge automatically

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Alpha values do not add linearly. The compositing formula is not a simple arithmetic sum. While the result is clamped to [0,1], the formula is not addition.
- **Option B (Correct):** Standard "over" alpha compositing stacks transparency multiplicatively. The effective transparency of two layers each with `alpha=0.6` is `(1-0.6) × (1-0.6) = 0.4 × 0.4 = 0.16`, meaning 16% of the background shows through. Effective opacity = `1 − 0.16 = 0.84`. The overlap area therefore appears significantly darker and more opaque than either individual fill (which passes 40% of background through). This stacking effect is why overlapping translucent fills can quickly produce unexpectedly dark patches.
- **Option C (Incorrect):** Multiplicative blending of `0.6 × 0.6 = 0.36` would be the combined opacity if the two layers were multiplied — this is not how standard "over" compositing works. Option B's formula is the correct one.
- **Option D (Incorrect):** Matplotlib does not merge overlapping artists. Each `PolyCollection` is rendered independently, and the overlap receives cumulative opacity from both layers.

</details>

---

### 28. Question

A developer writes the following code and saves the figure:

```python
fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 10, 300)
y = np.sin(x)

ax.fill_between(x, y, alpha=0.35, color="mediumvioletred")
ax.plot(x, y, color="mediumvioletred", lw=2)
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_title("Sine Wave Area")

fig.savefig("sine_area.png", dpi=150, bbox_inches="tight")
```

What does `bbox_inches="tight"` do when saving this figure?

- [ ] **A.** It crops the white border/padding around the figure to the tightest bounding box that contains all visible elements
- [ ] **B.** It increases the DPI to 300 for a tighter, sharper image
- [ ] **C.** It forces the fill area to extend to the figure boundary
- [ ] **D.** It locks the axis limits so they cannot be changed by the saved output

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `bbox_inches="tight"` instructs Matplotlib to compute the tightest bounding box that contains all visible artists in the figure (including axis labels, titles, tick labels, and legends that may extend beyond the default figure boundaries). Any excess white space outside the data region is trimmed. This is especially useful when saving figures for publication or embedding in documents, as it removes unwanted margins.
- **Option B (Incorrect):** DPI (dots per inch) is controlled separately by the `dpi=150` argument already present in the code. `bbox_inches="tight"` has no effect on DPI or image sharpness.
- **Option C (Incorrect):** `bbox_inches` controls the save bounding box, not the rendering extent of any plotted artists. The fill area stays bounded by the data and axis limits regardless of save settings.
- **Option D (Incorrect):** `bbox_inches` has no effect on axis limits. Axis limits are set by `ax.set_xlim` and `ax.set_ylim` and are not modified by save parameters.

</details>

---

### 29. Question

A data engineer creates an area chart with a logarithmic y-axis:

```python
x = np.arange(1, 11)
y = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683])  # 3^x

fig, ax = plt.subplots()
ax.fill_between(x, y, alpha=0.4, color="crimson")
ax.set_yscale("log")
ax.set_ylabel("Value (log scale)")
```

On a logarithmic y-scale, how does the filled region appear compared to a linear y-scale?

- [ ] **A.** Identical — fill_between coordinates are absolute, so log-scaling has no effect on the filled region
- [ ] **B.** The fill appears compressed near the x-axis and expanded at the top because log scaling compresses large values and expands small values, transforming the shape of the filled region
- [ ] **C.** The fill disappears because log scale cannot represent filled regions
- [ ] **D.** The fill appears as a straight diagonal line because logarithms of powers of 3 are linear

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Log scaling fundamentally changes how data coordinates map to display pixels. Although the data values are the same, their visual positions change dramatically. The fill region is transformed along with the axis scale.
- **Option B (Correct):** On a log y-scale, equal ratios map to equal vertical distances (e.g., going from 1→3→9→27 always spans the same display height because log(3)=constant). The fill region that appears as an exponentially growing area on a linear scale becomes a linearly growing region on a log scale — the boundary curve (3^x) becomes a straight line on a log plot, and the fill below it becomes a triangular rather than exponential region.
- **Option C (Incorrect):** Logarithmic scale axes work perfectly well with `fill_between`. Values must be positive (as they are here), and the fill is rendered correctly in log space.
- **Option D (Incorrect):** While log(3^x) = x·log(3) is indeed linear in x (making the boundary a straight line on a log-y plot), the filled region itself is not "a straight diagonal line" — it is the entire area from the x-axis (log scale baseline) up to that straight line, which is a triangular/trapezoidal filled area.

</details>

---

### 30. Question

A developer wants to add a second y-axis and fill between two curves on the secondary axis:

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

x = np.arange(0, 10)
temp = np.array([22, 25, 28, 30, 27, 24, 23, 26, 29, 31])
humidity = np.array([60, 55, 50, 45, 52, 58, 62, 54, 48, 42])

ax1.plot(x, temp, "r-", label="Temperature")
ax2.fill_between(x, humidity, alpha=0.3, color="blue")
ax2.plot(x, humidity, "b-", label="Humidity")
```

The `fill_between` is called on `ax2` (the twin axis). What baseline does it use?

- [ ] **A.** y=0 on the ax1 (primary) y-axis scale
- [ ] **B.** y=0 on the ax2 (secondary) y-axis scale — the fill extends from the humidity curve down to y=0 in ax2's coordinate system
- [ ] **C.** The minimum value of the temperature array (22)
- [ ] **D.** The minimum value of the humidity array (42)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Each axes object has its own independent coordinate system. `ax2.fill_between` operates entirely within `ax2`'s data coordinate space. The `ax1` y-axis scale is irrelevant to fills on `ax2`.
- **Option B (Correct):** `ax2.fill_between(x, humidity)` fills from the `humidity` curve down to `y2=0` — but in ax2's y-coordinate system. Since `ax2` is set up to display humidity values (40–65 range), the baseline y=0 in ax2's data space would be at the very bottom of ax2's range (or even below the visible area if the y-axis doesn't start at 0). The fill is drawn in ax2's own coordinate system exclusively.
- **Option C (Incorrect):** 22 is the minimum temperature value — from `ax1`'s data. `ax2.fill_between` does not cross-reference `ax1`'s data or axis limits.
- **Option D (Incorrect):** While 42 is the minimum of the `humidity` array, `fill_between` without explicit `y2` uses `0` as the baseline, not the array minimum. If `y=0` is below the visible range of `ax2`, the fill would extend beyond the visible plot area.

</details>
