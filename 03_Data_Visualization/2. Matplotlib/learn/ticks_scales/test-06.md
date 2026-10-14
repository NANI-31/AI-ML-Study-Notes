# Matplotlib Ticks & Scales Objective Test — Set 06

---

### Question 51

In the architecture of the `matplotlib.ticker` module, what primary responsibility belongs to a **`Locator`** subclass?

- [ ] A) It parses RGB hex color strings into CMYK vectors.
- [ ] B) It mathematically determines the coordinate positions where tick marks should be placed along an axis based on data limits and scaling.
- [ ] C) It renders fonts using FreeType typography glyphs.
- [ ] D) It calculates the bounding box of legend containers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) It mathematically determines the coordinate positions where tick marks should be placed along an axis based on data limits and scaling.

**Explanation:**

- **A is incorrect:** Color parsing, hex decoding, and color space transformations are handled by `matplotlib.colors`, not `ticker.Locator`.
- **B is correct:** In Matplotlib's ticker architecture, tick management is divided into two distinct abstractions: `Locator` and `Formatter`. A `Locator` (such as `MultipleLocator`, `MaxNLocator`, or `LogLocator`) evaluates the current coordinate limits and axis scale, computing an array of numerical coordinates indicating where tick marks (and accompanying gridlines) must physically reside along the axis.
- **C is incorrect:** Font rasterization and glyph rendering are managed by the `matplotlib.font_manager` and FreeType rendering backends.
- **D is incorrect:** Calculating legend bounding boxes and layout positioning is managed by `matplotlib.legend.Legend` and `BboxBase` layout transforms.

</details>

---

### Question 52

An engineer visualizes time-series sensor data over 60 seconds. They want major tick marks to appear at exact fixed intervals of $10$ seconds ($0, 10, 20, 30, 40, 50, 60$).

Which ticker class and configuration achieves this arithmetic step spacing?

- [ ] A)
  ```python
  from matplotlib.ticker import MultipleLocator

  ax.xaxis.set_major_locator(MultipleLocator(10))
```
- [ ] B)
  ```python
  from matplotlib.ticker import FixedStep

  ax.xaxis.set_step(10)
```
- [ ] C)
  ```python
  ax.xaxis.set_major_locator(IntervalLocator(10))
```
- [ ] D)
  ```python
  ax.set_ticks(frequency=10)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
from matplotlib.ticker import MultipleLocator

ax.xaxis.set_major_locator(MultipleLocator(10))
```

**Explanation:**

- **A is correct:** `matplotlib.ticker.MultipleLocator(base)` places ticks at exact integer multiples of `base` ($0, 10, 20, 30, \dots$). Attaching it via `ax.xaxis.set_major_locator(MultipleLocator(10))` ensures that regardless of dynamic panning, zooming, or window resizing, major ticks always snap to clean multiples of 10.
- **B is incorrect:** `FixedStep` and `ax.xaxis.set_step()` do not exist in Matplotlib; attempting to call them raises an `AttributeError`.
- **C is incorrect:** `IntervalLocator` is a fabricated class name that does not exist in `matplotlib.ticker`.
- **D is incorrect:** `ax.set_ticks(frequency=10)` is invalid; `set_ticks` expects an explicit list of coordinate locations and does not accept a `frequency` keyword argument.

</details>

---

### Question 53

When creating dynamic visualization dashboards where data ranges vary unpredictably, setting a rigid `MultipleLocator` can cause either overlapping tick clutter (if the range expands) or an empty axis (if the range contracts).

Which flexible locator automatically selects an aesthetically pleasing, human-friendly interval (snapping to round numbers like $1, 2, 2.5, 5, 10$) while strictly guaranteeing that the total number of major intervals does not exceed $N$?

- [ ] A) `RandomLocator(N)`
- [ ] B) `UniformLocator(N)`
- [ ] C) `CeilLocator(N)`
- [ ] D) `matplotlib.ticker.MaxNLocator(nbins=N)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `matplotlib.ticker.MaxNLocator(nbins=N)`

**Explanation:**

- **A is incorrect:** `RandomLocator` does not exist in Matplotlib.
- **B is incorrect:** `UniformLocator` is not a valid Matplotlib class.
- **C is incorrect:** `CeilLocator` is a fictitious locator name.
- **D is correct:** `matplotlib.ticker.MaxNLocator(nbins=N)` is Matplotlib's default smart locator for linear axes. It dynamically inspects the current view limits and chooses an interval step that produces at most $N$ intervals, while ensuring that tick locations snap to clean, readable "round" decimal intervals (such as multiples of 1, 2, 2.5, 5, or 10). It prevents visual clutter when data ranges expand dynamically.

</details>

---

### Question 54

A plot has major ticks positioned at $0, 10, 20, 30$. An engineer wants to add minor tick subdivisions so that there is a minor tick halfway between each major tick (at $5, 15, 25$).

Which locator accomplishes this subdivision?

- [ ] A) `ax.xaxis.set_minor_locator(HalfLocator())`
- [ ] B) `ax.xaxis.set_minor_locator(AutoMinorLocator(5))`
- [ ] C) `ax.xaxis.set_minor_locator(AutoMinorLocator(2))`
- [ ] D) `ax.xaxis.set_minor_locator(MinorSplitter(2))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.xaxis.set_minor_locator(AutoMinorLocator(2))`

**Explanation:**

- **A is incorrect:** `HalfLocator` does not exist in the Matplotlib ticker module.
- **B is incorrect:** `AutoMinorLocator(5)` divides each major interval into 5 equal segments, which for a 10-unit major interval places minor ticks at steps of 2 units ($2, 4, 6, 8$), rather than at the halfway midpoint.
- **C is correct:** `matplotlib.ticker.AutoMinorLocator(n)` subdivides each major tick interval into $n$ equal sub-intervals. Passing $n = 2$ splits each $10$-unit interval into two equal parts of width $5$, placing minor ticks precisely at the midpoints ($5, 15, 25, \dots$).
- **D is incorrect:** `MinorSplitter` is a non-existent class in Matplotlib.

</details>

---

### Question 55

A specialized computer vision heatmap requires the numerical y-axis ticks and labels to be completely absent, while maintaining the black border spine and allowing user interactivity.

Which locator subclass cleanly removes all ticks from an axis?

- [ ] A) `ax.yaxis.set_major_locator(ZeroLocator())`
- [ ] B) `ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())`
- [ ] C) `ax.yaxis.set_major_locator(EmptyLocator())`
- [ ] D) `ax.yaxis.set_major_locator(NoneLocator())`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())`

**Explanation:**

- **A is incorrect:** `ZeroLocator` is a fictitious class name that does not exist in Matplotlib.
- **B is correct:** `matplotlib.ticker.NullLocator()` is a dedicated null-object locator that always returns an empty coordinate array. Attaching it via `ax.yaxis.set_major_locator(NullLocator())` cleanly removes all major ticks and labels along that axis without destroying spines, titles, or canvas interactivity.
- **C is incorrect:** `EmptyLocator` does not exist in the `matplotlib.ticker` module.
- **D is incorrect:** `NoneLocator` is not a recognized Matplotlib class.

</details>

---

### Question 56

An analyst requires tick marks placed at an irregular, pre-calculated sequence of critical statistical thresholds: `thresholds = [0.05, 0.23, 0.67, 0.99]`.

Which locator class takes this explicit list of coordinates and locks them as the active tick locations?

- [ ] A) `matplotlib.ticker.FixedLocator(thresholds)`
- [ ] B) `matplotlib.ticker.CustomLocator(thresholds)`
- [ ] C) `matplotlib.ticker.StaticLocator(thresholds)`
- [ ] D) `matplotlib.ticker.ExplicitLocator(thresholds)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `matplotlib.ticker.FixedLocator(thresholds)`

**Explanation:**

- **A is correct:** `matplotlib.ticker.FixedLocator(locs)` takes an explicit sequence of numerical coordinates and locks them as the unchanging tick positions along that axis. In fact, calling `ax.set_xticks(locs)` internally constructs and binds a `FixedLocator` with those coordinate points.
- **B is incorrect:** `CustomLocator` does not exist in `matplotlib.ticker`.
- **C is incorrect:** `StaticLocator` is a fictitious class name.
- **D is incorrect:** `ExplicitLocator` is not a valid Matplotlib locator class.

</details>

---

### Question 57

When working with logarithmic axes (`set_yscale('log')`), which dedicated locator class generates ticks that step geometrically according to powers of the logarithm base?

- [ ] A) `PowerLocator`
- [ ] B) `GeometricLocator`
- [ ] C) `ExpLocator`
- [ ] D) `matplotlib.ticker.LogLocator(base=10)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `matplotlib.ticker.LogLocator(base=10)`

**Explanation:**

- **A is incorrect:** `PowerLocator` does not exist in Matplotlib.
- **B is incorrect:** `GeometricLocator` is not a recognized ticker class.
- **C is incorrect:** `ExpLocator` is a fabricated locator name.
- **D is correct:** `matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,), numdecs=4)` is specifically designed for logarithmic coordinate transformations. It computes tick coordinates at integer powers of the logarithm base ($base^k$, e.g., $10^0, 10^1, 10^2, \dots$), and can also accept `subs` to position minor ticks at intermediate multiples within each decade.

</details>

---

### Question 58

In an index-based time-series chart where data points represent trading days $0, 1, 2, \dots, N$, an engineer wants ticks to occur every 5 days, but starting at day 2 (i.e., days $2, 7, 12, 17, 22, \dots$).

Which locator class supports this `(base, offset)` indexing configuration?

- [ ] A) `StepLocator(base=5, start=2)`
- [ ] B) `ShiftedLocator(5, offset=2)`
- [ ] C) `matplotlib.ticker.IndexLocator(base=5, offset=2)`
- [ ] D) `PeriodicLocator(period=5, phase=2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `matplotlib.ticker.IndexLocator(base=5, offset=2)`

**Explanation:**

- **A is incorrect:** `StepLocator` does not exist in Matplotlib.
- **B is incorrect:** `ShiftedLocator` is a fabricated class name.
- **C is correct:** `matplotlib.ticker.IndexLocator(base, offset)` is tailored for indexed data sequences (such as trading days or discrete batch steps). It places ticks at coordinates $x = 	ext{offset} + k \cdot 	ext{base}$ for integers $k \ge 0$. With `base=5` and `offset=2`, ticks appear at indices $2, 7, 12, 17, 22, \dots$.
- **D is incorrect:** `PeriodicLocator` is not a valid class in `matplotlib.ticker`.

</details>

---

### Question 59

A developer writes the code:

```python
fig, ax = plt.subplots()
ax.plot(data)
ax.grid(True, which='minor', linestyle=':', color='gray')
```

When the plot is displayed, **no minor gridlines appear at all**. What is the technical reason for this absence, and how is it resolved?

- [ ] A) The color `'gray'` is not supported by minor gridlines; it must be set to `'black'`.
- [ ] B) By default, Matplotlib axes initialize with a `NullLocator` for minor ticks; because no minor ticks exist, no minor gridlines can be drawn until a minor locator (e.g., `AutoMinorLocator()`) is explicitly attached to the axis.
- [ ] C) `ax.grid()` only works for major grids; minor grids require calling `ax.minor_grid(True)`.
- [ ] D) The linestyle `':'` is deprecated and renders as completely transparent whitespace.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) By default, Matplotlib axes initialize with a `NullLocator` for minor ticks; because no minor ticks exist, no minor gridlines can be drawn until a minor locator (e.g., `AutoMinorLocator()`) is explicitly attached to the axis.

**Explanation:**

- **A is incorrect:** `'gray'` is a standard, fully supported color name.
- **B is correct:** Gridlines in Matplotlib do not calculate their own independent coordinates—they strictly bind to the tick coordinates provided by the axis locators. By default, an `Axes` initializes its minor ticks with a `NullLocator()`, meaning zero minor ticks are generated. Calling `ax.grid(which='minor')` requests gridlines at all minor tick coordinates, but because the coordinate list is empty, nothing renders. Calling `ax.xaxis.set_minor_locator(AutoMinorLocator())` instantiates minor ticks, allowing minor gridlines to render immediately.
- **C is incorrect:** `ax.grid()` natively supports `which='minor'`; there is no `minor_grid()` method.
- **D is incorrect:** The dotted linestyle `':'` is fully supported.

</details>

---

### Question 60

Which of the following statements regarding Matplotlib Locators are **correct**?

1. `MultipleLocator` places ticks at uniform arithmetic steps (e.g., every 5 units).
2. `MaxNLocator` dynamically chooses round, human-readable numbers to avoid awkward fractional tick marks.
3. Locators are assigned to an axis via methods like `ax.xaxis.set_major_locator()`.
4. Assigning a new major locator to an axis automatically erases all data lines and patches previously plotted on that axis.

- [ ] A) 1, 2, and 3 only
- [ ] B) 1 and 4 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 1, 2, and 3 only

**Explanation:**

- **A is correct:** Statements 1, 2, and 3 are all accurate. Statement 1 is correct because `MultipleLocator(v)` steps strictly by multiples of $v$. Statement 2 is correct because `MaxNLocator` dynamically bounds tick counts while snapping to round intervals. Statement 3 is correct because `set_major_locator()` is the standard setter for axis locators.
- **B is incorrect:** Statement 4 is false; assigning a new locator only affects axis tick coordinates and never touches or erases plotted data artists (`Line2D`, `PathCollection`, etc.).
- **C is incorrect:** Misses Statement 1 (which is completely true) and includes false Statement 4.
- **D is incorrect:** Omits Statements 1 and 2 (both true) and erroneously includes false Statement 4.

</details>

---
