# Matplotlib Area Plot Objective Test — Set 01

---

### 1. Question

A data analyst writes the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)

plt.fill_between(x, y)
plt.show()
```

What is the default lower boundary used by `fill_between` when only `y1` is provided and `y2` is omitted?

- [ ] **A.** The minimum value of `y`
- [ ] **B.** The value 0 (the horizontal x-axis)
- [ ] **C.** The mean value of `y`
- [ ] **D.** The previous y-value in the series

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `fill_between` does not automatically compute the minimum of the input array and use that as the baseline. The minimum of `np.sin(x)` is −1, not 0. The shaded region would appear very different if the minimum were used.
- **Option B (Correct):** When `y2` is omitted, Matplotlib sets it to `0` by default. The documentation signature is `fill_between(x, y1, y2=0, ...)`. The shaded polygon is therefore drawn between the `y=sin(x)` curve and the horizontal line `y=0`. Positive portions of the sine curve produce shading above the x-axis; negative portions produce shading below it.
- **Option C (Incorrect):** The mean of `y` is not used as a default baseline. Although computing and plotting the mean as a reference line is a common analytics pattern, it is not the automatic default.
- **Option D (Incorrect):** There is no "previous y-value" baseline in `fill_between`. The function is not a step-fill or cumulative operation — it fills between two complete arrays (or between one array and a scalar baseline).

</details>

---

### 2. Question

A developer writes:

```python
x = np.array([0, 1, 2, 3, 4])
y = np.array([5, 8, 3, 7, 4])

plt.fill_between(x, y, alpha=0.4, color="steelblue")
plt.plot(x, y, color="steelblue", linewidth=2)
plt.show()
```

Why is `plt.plot(x, y, ...)` called separately in addition to `plt.fill_between`?

- [ ] **A.** `fill_between` also draws the line automatically, so the second call creates a duplicate
- [ ] **B.** `fill_between` only draws the filled polygon region; it does not draw a visible boundary line — `plt.plot` is needed to render the crisp upper-boundary line on top
- [ ] **C.** Without `plt.plot`, the `alpha` parameter is ignored
- [ ] **D.** `fill_between` requires a preceding `plt.plot` call to initialize the x-axis scale

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `fill_between` does NOT automatically draw boundary lines. It returns a `PolyCollection` — a filled polygon — with no visible stroke on its edges by default. Calling `plt.plot` is not redundant; it adds the only visible crisp boundary line.
- **Option B (Correct):** The `fill_between` function creates a filled shaded area but does not draw the curve outline. For publication-quality figures it is standard practice to add an explicit `plt.plot` (or `ax.plot`) call with matching colour and a thicker `linewidth` so the boundary curve is sharp and clearly defined, contrasting against the translucent fill.
- **Option C (Incorrect):** `alpha` is a transparency parameter applied directly to the `PolyCollection` patch created by `fill_between`. It is entirely independent of whether `plt.plot` is also called. The `alpha` takes effect regardless.
- **Option D (Incorrect):** Matplotlib initialises axis scales automatically based on all data passed to any plotting function. `fill_between` alone is sufficient to establish x-axis and y-axis limits. No "initialisation" call to `plt.plot` is required.

</details>

---

### 3. Question

A researcher fills the area between two curves:

```python
x = np.linspace(0, 5, 100)
y1 = x ** 2
y2 = x ** 2 + 4

plt.fill_between(x, y1, y2, color="orange", alpha=0.5)
plt.show()
```

Which region is shaded orange?

- [ ] **A.** The area between `y=x²` and the x-axis
- [ ] **B.** The area between `y=x²+4` and the x-axis
- [ ] **C.** The vertical band between the curves `y=x²` and `y=x²+4` — a uniform strip of height 4 at every x
- [ ] **D.** The area between `y=x²+4` and `y=4`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Shading between `y=x²` and the x-axis would require `plt.fill_between(x, y1, 0)` or simply `plt.fill_between(x, y1)`. Here `y2=y1+4` is provided, not `y2=0`.
- **Option B (Incorrect):** Shading from `y=x²+4` down to the x-axis (`y=0`) would require `plt.fill_between(x, y2, 0)`. Passing `y2` as the second argument and the baseline `0` as the third.
- **Option C (Correct):** `fill_between(x, y1, y2)` fills the region between the two curves at every x-value. Since `y2 = y1 + 4`, the vertical gap between the curves is always exactly 4 units, regardless of x. The shaded band is a uniform-height strip running between the two parabolas across the full x-range.
- **Option D (Incorrect):** `y=4` is a constant horizontal line at y=4. The fill between `y=x²+4` and `y=4` would start at x=0 (where both equal 4) and grow upward as x increases. This is not what the code produces.

</details>

---

### 4. Question

What Python object type does `ax.fill_between(x, y1, y2)` return?

- [ ] **A.** `Line2D`
- [ ] **B.** `Polygon`
- [ ] **C.** `PolyCollection`
- [ ] **D.** `PathPatch`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `Line2D` is the object returned by `ax.plot(...)`. It represents a single line or curve, not a filled region. `fill_between` does not return a `Line2D`.
- **Option B (Incorrect):** A `Polygon` is a single closed patch with a fixed number of vertices. While the fill region is conceptually polygonal, Matplotlib represents it as a `PolyCollection`, which is a collection of potentially many polygonal patches (useful when the filled area has discontinuities from `where` conditions).
- **Option C (Correct):** `ax.fill_between` returns a `PolyCollection` — a collection of one or more filled polygon patches. The `PolyCollection` class allows the fill to handle discontinuities (e.g., from `where` conditions that split the shaded area into separate islands), which a single `Polygon` could not represent.
- **Option D (Incorrect):** `PathPatch` is a Matplotlib artist for arbitrary vector paths. It is not the return type of `fill_between`. `PathPatch` is typically used for custom vector graphics.

</details>

---

### 5. Question

A data engineer writes:

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 15, 25, 18])

plt.fill_between(x, y, color="teal", alpha=0.3, label="Range")
plt.legend()
plt.title("Monthly Output")
plt.show()
```

The `label="Range"` argument is set. How does the legend entry for the shaded area appear?

- [ ] **A.** A small circle (scatter marker) coloured teal with label "Range"
- [ ] **B.** A short line segment coloured teal with label "Range"
- [ ] **C.** A small solid-coloured rectangle (colour swatch) matching the fill colour and opacity with label "Range"
- [ ] **D.** No legend entry appears because `fill_between` does not support the `label` argument

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Circle scatter markers are the legend representation for `plt.scatter`. The `fill_between` function creates a patch, not scatter points.
- **Option B (Incorrect):** Line segments in the legend represent `Line2D` artists from `plt.plot`. The `fill_between` returns a `PolyCollection`, which has a square patch icon in the legend, not a line.
- **Option C (Correct):** When `label=` is provided to `fill_between`, Matplotlib creates a legend entry represented by a small square coloured patch that matches the fill colour (teal) and opacity (`alpha=0.3`). This accurately conveys the visual appearance of the shaded region in the legend.
- **Option D (Incorrect):** `fill_between` fully supports the `label` argument. The label is stored on the returned `PolyCollection` artist and is picked up automatically by `plt.legend()`. This is the standard way to include filled areas in plot legends.

</details>

---

### 6. Question

A developer wants to shade the area under a revenue curve from the x-axis up to the curve:

```python
months = np.arange(1, 13)
revenue = np.array([40, 55, 48, 70, 65, 90, 80, 95, 75, 110, 105, 130])

plt.fill_between(months, revenue, 0, color="seagreen", alpha=0.4)
plt.plot(months, revenue, color="seagreen", linewidth=2)
plt.show()
```

Is explicitly passing `0` as `y2` equivalent to omitting `y2`?

- [ ] **A.** No — omitting `y2` uses the minimum of `revenue` as the baseline
- [ ] **B.** Yes — the default value of `y2` in `fill_between` is `0`, so explicitly passing `0` produces identical output
- [ ] **C.** No — omitting `y2` creates an error because `y2` is a required argument
- [ ] **D.** No — omitting `y2` uses the median of `revenue` as the baseline

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The default baseline is 0, not the minimum of the `y1` array. Using the minimum would require manually computing `revenue.min()` and passing it as `y2`.
- **Option B (Correct):** The full Matplotlib signature for `fill_between` is `fill_between(x, y1, y2=0, ...)`. The default value of `y2` is the scalar `0`. Explicitly passing `0` as the third argument and omitting `y2` both produce identical shaded regions — the area between the revenue curve and the horizontal line at y=0.
- **Option C (Incorrect):** `y2` is an optional argument with a default of `0`. Omitting it does not cause an error. Only `x` and `y1` are required.
- **Option D (Incorrect):** The default is 0, not the median. Using the median would produce a different baseline that cuts through the data rather than anchoring at the zero line.

</details>

---

### 7. Question

A research engineer sets `alpha=0.0` on a `fill_between` call:

```python
plt.fill_between(x, y1, y2, color="purple", alpha=0.0)
```

What is the visual result?

- [ ] **A.** A solid purple filled region (alpha=0.0 means fully opaque in Matplotlib)
- [ ] **B.** A completely invisible fill — the shaded area is transparent and not visible
- [ ] **C.** A purple outline (border) with no fill interior
- [ ] **D.** An error is raised because alpha cannot be set to 0 for patch artists

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In Matplotlib (and standard graphics conventions), `alpha=0.0` means fully transparent (0% opacity), not fully opaque. Fully opaque is `alpha=1.0`.
- **Option B (Correct):** `alpha=0.0` sets the fill's opacity to zero, making the `PolyCollection` completely invisible. The shaded region exists as an artist object in memory but is not rendered in the output image. This is sometimes used programmatically to "hide" a fill without removing the artist object, enabling it to be toggled visible later.
- **Option C (Incorrect):** Setting `alpha=0.0` makes the entire patch (including any edges) invisible. If a visible border outline were desired, it would require `facecolor="none"` with a specified `edgecolor`.
- **Option D (Incorrect):** `alpha=0.0` is a perfectly valid value for any Matplotlib artist. Values from 0.0 to 1.0 are all accepted. No error is raised.

</details>

---

### 8. Question

A data scientist fills between two simple curves:

```python
x = np.array([0, 1, 2, 3])
y1 = np.array([1, 3, 2, 4])
y2 = np.array([2, 5, 4, 6])

plt.fill_between(x, y1, y2, color="coral", alpha=0.6)
```

At x=2, what is the vertical height of the shaded region?

- [ ] **A.** 2 units (from y=2 to y=4)
- [ ] **B.** 6 units (from y=0 to y=6)
- [ ] **C.** 4 units (the value of y2 at x=2)
- [ ] **D.** 1 unit (from y=3 to y=4)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** At x=2, `y1=2` and `y2=4`. The `fill_between` function shades the vertical strip between these two values. The height of that strip is `y2 - y1 = 4 - 2 = 2` units. The fill spans from y=2 up to y=4 at this x-coordinate.
- **Option B (Incorrect):** 6 units would be the height from y=0 (the x-axis) up to y=6 (the value of y2 at x=3). This confuses the maximum value of y2 with the height of the shaded region at x=2.
- **Option C (Incorrect):** 4 is the absolute value of y2 at x=2, not the height of the shaded region. The fill starts at y1=2, not at y=0. The height is y2−y1=4−2=2, not y2=4.
- **Option D (Incorrect):** 1 unit would be the height from y=3 to y=4. But at x=2, y1=2 (not 3). y1=3 occurs at x=1, not x=2.

</details>

---

### 9. Question

A developer wants to add a title and axis labels to an area fill chart. Which of the following code structures is correct?

```python
x = np.linspace(0, 10, 100)
y = np.exp(-x / 3)

fig, ax = plt.subplots()
ax.fill_between(x, y, alpha=0.4, color="goldenrod")
ax.plot(x, y, color="goldenrod", lw=2)
```

- [ ] **A.** `plt.title("Decay Curve"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude")`
- [ ] **B.** `ax.set_title("Decay Curve"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")`
- [ ] **C.** Both A and B are valid approaches — `plt.title` and `ax.set_title` refer to the same underlying axes when using `fig, ax = plt.subplots()`
- [ ] **D.** Neither approach works — title and labels must be set before `ax.fill_between` is called

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Partially correct):** `plt.title`, `plt.xlabel`, and `plt.ylabel` target the currently active axes. After `fig, ax = plt.subplots()`, the single axes `ax` is the current axes. These `plt.*` calls are therefore equivalent to calling the methods directly on `ax`.
- **Option B (Partially correct):** `ax.set_title`, `ax.set_xlabel`, and `ax.set_ylabel` explicitly target `ax`. This is the object-oriented (OO) style and is unambiguous — especially important for multi-subplot figures.
- **Option C (Correct):** Both approaches are valid for a single-axes figure. In a single `plt.subplots()` plot, `plt.title()` internally calls `plt.gca().set_title()`, which is the same `ax`. The `plt.*` state-machine style and the OO style are equivalent here. In multi-subplot figures, the explicit OO style (Option B) is safer.
- **Option D (Incorrect):** Matplotlib artists (titles, labels, fills, plots) are independent of each other's order. You can set the title before or after `fill_between`. All artists are composed and rendered together at `plt.show()`.

</details>

---

### 10. Question

A developer mistakenly writes:

```python
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 60)   # different length!

plt.fill_between(x, y)
plt.show()
```

What happens?

- [ ] **A.** Matplotlib interpolates both arrays to a common length and plots successfully
- [ ] **B.** A `ValueError` is raised because `x` and `y1` must have the same length
- [ ] **C.** The fill is plotted using only the first 50 elements of `y`
- [ ] **D.** The fill is plotted using only the last 60 elements of `x`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not silently interpolate or resize mismatched arrays. Array shapes must match exactly. Automatic interpolation would hide bugs and produce incorrect visualisations without any warning.
- **Option B (Correct):** `fill_between(x, y1)` requires that `x` and `y1` have the same length because they define corresponding (x, y) coordinate pairs. When `x` has 50 elements and `y` has 60 elements, NumPy cannot broadcast them into aligned pairs, and Matplotlib raises a `ValueError` (typically something like "x and y1 must have equal lengths").
- **Option C (Incorrect):** Matplotlib does not silently truncate arrays. The length mismatch is detected and an error is raised immediately.
- **Option D (Incorrect):** Matplotlib does not silently truncate `x` either. The error is raised before any plotting occurs.

</details>
