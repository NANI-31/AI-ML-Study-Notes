# Matplotlib Area Plot Objective Test — Set 06

---

### 51. Question

A data engineer creates a stacked area chart of product revenue:

```python
months = np.arange(1, 13)
revenue_A = np.array([30, 35, 28, 42, 38, 50, 45, 55, 48, 60, 58, 70])
revenue_B = np.array([20, 22, 18, 25, 24, 30, 28, 35, 30, 40, 38, 45])
revenue_C = np.array([15, 18, 12, 20, 22, 25, 20, 28, 25, 32, 30, 38])

plt.stackplot(months, revenue_A, revenue_B, revenue_C,
              labels=["Product A", "Product B", "Product C"])
plt.legend(loc="upper left")
```

At month 6, what is the total stacked height of the chart (top of the Product C layer)?

- [ ] **A.** 25 (only Product C's value)
- [ ] **B.** 80 (Product A + Product B at month 6)
- [ ] **C.** 105 (Product A + Product B + Product C)
- [ ] **D.** 50 (Product A alone at month 6)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 25 is Product C's individual contribution at month 6. In a stacked area chart, Product C sits on top of Products A and B, so the top of its layer is not at y=25 but at the sum of all three.
- **Option B (Incorrect):** 80 = 50 + 30 = Product A + Product B at month 6. This is the baseline from which Product C starts (y=80), not the top of Product C's layer.
- **Option C (Correct):** In `plt.stackplot`, each series stacks on top of the previous one. At month 6: Product A = 50 (bottom layer, from y=0 to y=50). Product B = 30 (second layer, from y=50 to y=80). Product C = 25 (top layer, from y=80 to y=105). Total stacked height = 50+30+25 = 105.
- **Option D (Incorrect):** 50 is only Product A's value — the top of the first (bottom) layer. Products B and C add 30 and 25 more units above it.

</details>

---

### 52. Question

A developer creates a three-series stacked area chart. At time t=5:

- Layer 1 (bottom): value = 40
- Layer 2 (middle): value = 25
- Layer 3 (top): value = 15

What are the y-coordinates of the **bottom baseline** and **top boundary** of the middle layer (Layer 2) at t=5?

- [ ] **A.** Bottom = 0, Top = 25
- [ ] **B.** Bottom = 40, Top = 65
- [ ] **C.** Bottom = 25, Top = 40
- [ ] **D.** Bottom = 40, Top = 80

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Bottom=0 would be correct for Layer 1 (the bottom-most layer), not Layer 2. Layer 2 sits on top of Layer 1.
- **Option B (Correct):** In a stacked area chart, each layer's bottom boundary is the cumulative sum of all layers below it. Layer 2's bottom = Layer 1's value = 40. Layer 2's top = 40 + 25 = 65. The middle layer's visible height is 25 units (its own value), spanning from y=40 to y=65.
- **Option C (Incorrect):** Bottom=25, Top=40 reverses the correct values. 25 is Layer 2's own value (its height), and 40 is Layer 1's value (Layer 2's baseline). They are not the bottom and top coordinates.
- **Option D (Incorrect):** Top=80 would require Layer 2's value to be 80−40=40, but Layer 2's value is 25. The correct top is 40+25=65.

</details>

---

### 53. Question

A data scientist uses `plt.stackplot` with the `labels` keyword:

```python
x = np.arange(0, 10)
s1 = np.random.randint(10, 20, 10)
s2 = np.random.randint(5, 15, 10)
s3 = np.random.randint(3, 10, 10)

fig, ax = plt.subplots()
ax.stackplot(x, s1, s2, s3,
             labels=["Service A", "Service B", "Service C"],
             colors=["#FF6B6B", "#4ECDC4", "#45B7D1"],
             alpha=0.8)
ax.legend(loc="upper right")
```

What does calling `ax.legend()` after `stackplot` add to the chart?

- [ ] **A.** Nothing — `stackplot` does not support the `labels` argument and ignores it
- [ ] **B.** A legend box with colour-coded entries for each stacked layer: "Service A" (red), "Service B" (teal), "Service C" (blue)
- [ ] **C.** A legend showing only the total stacked height, not individual layers
- [ ] **D.** An error — `stackplot` legends must be created manually using `matplotlib.patches.Patch`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `stackplot` does support the `labels` argument. The labels are assigned to the `PolyCollection` artists returned by the function and are picked up by `ax.legend()`.
- **Option B (Correct):** `plt.stackplot` returns a list of `PolyCollection` objects — one per data series. When `labels` are provided, each `PolyCollection` is tagged with its corresponding label. `ax.legend()` collects these labeled artists and creates a legend box with one entry per layer: "Service A" with its red colour swatch, "Service B" with teal, "Service C" with blue. This is the standard, built-in way to create a legend for `stackplot`.
- **Option C (Incorrect):** The legend shows individual entries per layer, not the total. There is no automatic "total" summary entry in a standard `stackplot` legend.
- **Option D (Incorrect):** While creating proxy patches manually (`matplotlib.patches.Patch`) is an alternative approach, it is not required. `stackplot` with `labels=` directly supports `ax.legend()` without any manual workarounds.

</details>

---

### 54. Question

A developer uses `baseline="sym"` in `plt.stackplot`:

```python
plt.stackplot(x, series1, series2, series3,
              baseline="sym", alpha=0.7)
```

What does `baseline="sym"` change about the stacked area chart?

- [ ] **A.** It makes all series symmetric around their mean values
- [ ] **B.** It centres the stacked layers around y=0 (a symmetric/streamgraph layout), with layers spread above and below the zero axis rather than stacking from the bottom
- [ ] **C.** It sorts the layers so that the largest values are always at the bottom
- [ ] **D.** It makes the chart vertically symmetric by mirroring the entire plot about the x-axis

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `baseline="sym"` does not centre individual series around their own means. It repositions the stacking baseline to balance the stack symmetrically about y=0.
- **Option B (Correct):** `baseline="sym"` implements the "streamgraph" or "ThemeRiver" layout. Instead of stacking all layers from y=0 upward (the default `baseline="zero"`), the symmetric baseline shifts all layers so that the total stack is centered around y=0. Layers spread both above and below the horizontal axis, creating a visually balanced, organic-looking stream shape. This is often used in artistic data visualisations of evolving proportions over time.
- **Option C (Incorrect):** Sorting layers by size is not what `baseline="sym"` does. Layer ordering follows the input argument order regardless of `baseline`.
- **Option D (Incorrect):** `baseline="sym"` does not mirror or duplicate the chart. It repositions the single stack symmetrically about zero.

</details>

---

### 55. Question

A developer wants to change the colour order of layers in a `stackplot`. Currently:

```python
plt.stackplot(t, cpu, memory, disk)
```

The default colours are the first three colours of Matplotlib's default colour cycle. How can the developer explicitly assign specific colours?

- [ ] **A.** `plt.stackplot(t, cpu, memory, disk, color=["crimson", "gold", "steelblue"])`
- [ ] **B.** `plt.stackplot(t, cpu, memory, disk, colors=["crimson", "gold", "steelblue"])`
- [ ] **C.** `plt.stackplot(t, cpu, memory, disk, facecolor=["crimson", "gold", "steelblue"])`
- [ ] **D.** `for layer, c in zip(plt.stackplot(...), ["crimson","gold","steelblue"]): layer.set_color(c)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The singular `color=` parameter is not the correct keyword for `stackplot`. The function expects the plural `colors=` to pass a list of colours for multiple series.
- **Option B (Correct):** `plt.stackplot` accepts a `colors=` keyword argument (plural) that takes an iterable of colour specifications — one per series. `colors=["crimson", "gold", "steelblue"]` assigns crimson to `cpu`, gold to `memory`, and steelblue to `disk` layers respectively.
- **Option C (Incorrect):** `facecolor` is a valid patch parameter but is not the top-level keyword argument for `stackplot`'s colour list. The correct argument is `colors=`.
- **Option D (Incorrect):** While post-hoc colour assignment is technically possible using the returned `PolyCollection` list and `set_facecolor`, this is unnecessarily complicated when `colors=` provides a clean, direct way to specify colours at creation time.

</details>

---

### 56. Question

A data analyst creates a stacked area chart showing three browser market share series over 12 months:

```python
months = np.arange(1, 13)
chrome  = np.array([55, 56, 57, 58, 57, 59, 60, 62, 61, 63, 64, 65])
firefox = np.array([25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 17, 16])
others  = np.array([20, 20, 20, 20, 21, 20, 20, 19, 20, 19, 19, 19])

plt.stackplot(months, chrome, firefox, others,
              labels=["Chrome", "Firefox", "Others"],
              colors=["#4285F4", "#FF6600", "#AAA"])
```

A colleague claims that "Firefox's market share layer appears to be shrinking visually in the chart." Is this claim correct, and why?

- [ ] **A.** Yes — Firefox's layer appears to shrink because its values decrease from 25 to 16 over the year
- [ ] **B.** No — Firefox's layer appears to be shifting upward (its baseline moves with Chrome's growth), making its shrinkage harder to perceive visually
- [ ] **C.** Yes — Firefox's layer shrinks both in absolute height and its baseline remains fixed at Chrome's final value
- [ ] **D.** No — the layer's height cannot change because all shares sum to 100% in every month

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Firefox's actual values decrease from 25 to 16 — a real reduction of 9 percentage points. In a stacked area chart, Firefox sits directly on top of Chrome's layer. Its visible height (the thickness of the orange band) equals Firefox's value at each month. Since those values genuinely decrease, the Firefox layer does appear visually thinner as the months progress. Unlike the "fluctuating baseline" optical illusion (which applies to layers above a rising lower layer), here Firefox sits on top of Chrome, which is also growing. This means Firefox's absolute position (top edge) might stay roughly constant or change less dramatically, but its THICKNESS (layer height) clearly decreases.
- **Option B (Incorrect):** While it's true that Firefox's baseline (bottom edge) shifts upward as Chrome's values grow, the HEIGHT of Firefox's layer equals its own values (25→16), which does decrease. The shrinkage in thickness is real and visible.
- **Option C (Incorrect):** Firefox's baseline does not remain fixed — it follows Chrome's values. The baseline rises as Chrome grows, partially offsetting the downward shift of Firefox's top edge.
- **Option D (Incorrect):** The values do not sum to exactly 100% in every month (e.g., month 1: 55+25+20=100, month 7: 60+20+20=100, month 12: 65+16+19=100 — they do sum to 100). But this doesn't prevent individual layer heights from changing. The question's premise about height is correct.

</details>

---

### 57. Question

A data scientist creates a five-layer stacked area chart for server infrastructure metrics:

```python
t = np.arange(0, 24)   # hours
cpu_idle   = 40 + np.zeros(24)    # always 40%
cpu_user   = 20 + np.random.uniform(-5, 5, 24)
cpu_sys    = 10 + np.random.uniform(-2, 2, 24)
io_wait    = 8  + np.random.uniform(-3, 3, 24)
irq        = 2  + np.zeros(24)    # always 2%

plt.stackplot(t, cpu_idle, cpu_user, cpu_sys, io_wait, irq,
              labels=["Idle", "User", "Sys", "I/O Wait", "IRQ"])
```

Layer `io_wait` has random values around 8% per hour. In the stacked chart, a viewer watches the `io_wait` layer's band and notices it appears wavy even though they might expect it to look "level." Why?

- [ ] **A.** The `io_wait` values are random, so the layer inherits random variation in both its top and bottom edges
- [ ] **B.** The `io_wait` layer's bottom edge follows the cumulative shape of `cpu_idle + cpu_user + cpu_sys` (which fluctuates due to random components in `cpu_user` and `cpu_sys`), causing both the top and bottom of `io_wait` to appear wavy even if `io_wait`'s own values were constant
- [ ] **C.** The `plt.stackplot` function adds artificial noise for visual aesthetics
- [ ] **D.** The random seed was not fixed, causing different values to be generated each time

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Partially correct but incomplete):** Yes, `io_wait` has random values, which causes its layer HEIGHT to fluctuate. But the question specifically asks why it APPEARS wavy — and even if `io_wait` were constant, the layer would still appear wavy because the baseline (bottom edge) fluctuates.
- **Option B (Correct):** This is the key "fluctuating baseline" problem with stacked area charts. The `io_wait` layer sits on top of `cpu_idle + cpu_user + cpu_sys`. The `cpu_user` and `cpu_sys` layers both have random fluctuations. Even if `io_wait` were a constant 8% per hour, its bottom edge would move up and down following the cumulative sum of the three layers below it. Combined with `io_wait`'s own random variation, both the bottom and top edges of the band appear wavy. This makes it very difficult for viewers to assess the true trend of any non-bottom layer in a stacked area chart.
- **Option C (Incorrect):** `plt.stackplot` does not add artificial noise. The waviness comes from the actual data and the stacking mechanics.
- **Option D (Incorrect):** While fixing the random seed ensures reproducibility, the waviness would still appear regardless of which random values are generated. The structural cause (fluctuating baseline) is the real issue, not randomness itself.

</details>

---

### 58. Question

A developer uses `plt.stackplot` to show parts-of-a-whole data, but one of the series has a **negative value** at a specific time step:

```python
t = np.array([1, 2, 3, 4, 5])
A = np.array([30, 35, 40, 38, 42])
B = np.array([20, 22, -5, 18, 25])   # negative at t=3!
C = np.array([15, 18, 14, 20, 16])

plt.stackplot(t, A, B, C)
```

What visual anomaly occurs at t=3 where `B[2] = -5`?

- [ ] **A.** The chart raises a `ValueError` because `stackplot` requires all non-negative inputs
- [ ] **B.** At t=3, Layer B extends downward (below its baseline from Layer A), creating a visual dip or gap that makes the stacked chart misrepresent the parts-of-a-whole relationship
- [ ] **C.** Layer B's negative value at t=3 is automatically clamped to 0
- [ ] **D.** Layer B disappears at t=3 and reappears at t=4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.stackplot` does not validate that all inputs are non-negative. Negative values are accepted and rendered without raising errors. The visual result is what becomes problematic, not the function call.
- **Option B (Correct):** At t=3, Layer B has value −5. In the stacking, Layer B's top boundary = Layer A's top + B's value = 40 + (−5) = 35. This means Layer B's top is actually BELOW Layer A's top — creating a visual dip where the B layer appears to go backwards (downward) from the stacking baseline. Layer C then stacks above y=35. The result is a confusing visual anomaly where the "parts-of-a-whole" narrative breaks down, as the layers no longer form a clean monotonically increasing stack at t=3.
- **Option C (Incorrect):** Matplotlib does not automatically clamp values in `stackplot`. The actual input value (−5) is used directly in the stacking calculation.
- **Option D (Incorrect):** Layers don't disappear. The negative value is included in the polygon but causes the top edge of Layer B to dip below Layer A's top edge, creating a geometric inconsistency in the stack.

</details>

---

### 59. Question

A data analyst uses `plt.stackplot` with `baseline="wiggle"`. What does the wiggle baseline achieve?

- [ ] **A.** It adds a sinusoidal oscillation to the baseline for visual interest
- [ ] **B.** It positions the baseline to minimise the total angular "wiggle" or slope of the sum line, producing a visually streamlined flow shape sometimes used for aesthetically pleasing streamgraphs
- [ ] **C.** It centres each individual series around its own mean value
- [ ] **D.** It sorts layers by their variability (most wiggly series on the outside)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The wiggle baseline does not add a sinusoidal pattern. It is a data-driven baseline optimisation, not a decorative oscillation.
- **Option B (Correct):** The "wiggle" baseline (also known as the "Streamgraph" layout, from the Havre et al. 2002 paper) computes a baseline offset at each time step that minimises the weighted average slope of the stacked layers' boundaries. The goal is to make the stream visually appear to "flow" with minimal aggregate angular change — producing a smooth, organic-looking chart often used in music listening history, brand evolution, or narrative flow visualisations. It is more of a visual design optimisation than a statistical baseline.
- **Option C (Incorrect):** Centering around individual series means is not what `baseline="wiggle"` does. That would be more like `baseline="sym"` applied per-series.
- **Option D (Incorrect):** `baseline="wiggle"` does not reorder series. Layer ordering follows the input argument order regardless of variability.

</details>

---

### 60. Question

A developer creates a stacked area chart for a 7-day market share dataset and then plots both the stacked chart and a legend. The legend entries appear in **reverse order** compared to the stacked layers (top layer listed first in the legend, bottom layer last). Why does this happen?

- [ ] **A.** `plt.stackplot` always reverses the legend order as a bug
- [ ] **B.** `plt.legend()` by default collects artists in the order they were added to the axes, which matches the layer order from bottom to top; the apparent reverse order depends on `loc` parameter only
- [ ] **C.** By default, `plt.legend()` collects artists in the order they were added (bottom to top matching the stackplot argument order), but the visual chart shows the bottom layer at the physical bottom and top layer at the top — if the viewer reads the legend top-to-bottom while comparing to the chart bottom-to-top, the correspondence feels reversed
- [ ] **D.** `stackplot` reverses the `labels` list internally before assigning them to PolyCollections

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** There is no bug — the legend order is consistent with the order artists were added to the axes. The "reversed" appearance is a perceptual mismatch, not a technical error.
- **Option B (Incorrect):** While legend order does follow the order artists were added, and this does match the stackplot bottom-to-top order, this explanation does not capture WHY users perceive it as reversed.
- **Option C (Correct):** `plt.stackplot(x, s1, s2, s3)` adds three `PolyCollection` objects to the axes: s1 first (bottom layer), s2 second (middle), s3 third (top). `plt.legend()` lists them in the same order: s1 first (top of legend box), s2 second, s3 last. In the chart, s1 is physically at the bottom and s3 is at the top. When a viewer reads the legend downward (s1, s2, s3) and then looks at the chart from bottom to top (s1, s2, s3), the order actually matches — it only feels reversed if one compares legend top entry to chart visual top layer. The standard fix is to use `ax.legend(handles[::-1], labels[::-1])` to reverse both lists.
- **Option D (Incorrect):** `stackplot` does not reverse the labels list. The labels are assigned to `PolyCollections` in the input argument order.

</details>
