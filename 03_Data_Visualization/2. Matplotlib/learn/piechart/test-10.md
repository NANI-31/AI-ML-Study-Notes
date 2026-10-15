# Matplotlib Pie Chart Objective Test — Set 10

---

### 91. Question

A data analyst runs the following script to plot quarterly business unit results:

```python
import matplotlib.pyplot as plt

units = ["Retail", "Cloud", "Logistics", "Hardware"]
margins = [450, 820, -120, 310]

plt.pie(margins, labels=units)
plt.show()
```

What exception is raised by Matplotlib when executing this code?

- [ ] **A.** `TypeError: sequence must contain positive integers only`
- [ ] **B.** `ValueError: Wedge sizes must be non-negative`
- [ ] **C.** `NegativeSliceError: angle cannot sweep backwards`
- [ ] **D.** `ArithmeticError: sum of margins is less than zero`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.pie()` accepts both floats and integers; the error is not a `TypeError`.
- **Option B (Correct):** In Matplotlib, wedge geometry requires that every input value represents a non-negative magnitude ($x_i \ge 0$). Because `margins` contains `-120`, Matplotlib's input validation triggers an immediate exception: `ValueError: Wedge sizes must be non-negative`. Net negative financial losses cannot be rendered as circular pie wedges.
- **Option C (Incorrect):** `NegativeSliceError` is not a valid Python or Matplotlib exception class.
- **Option D (Incorrect):** The total sum ($1460$) is positive, so the failure is specifically caused by the presence of a negative element.

</details>

---

### 92. Question

A machine learning researcher attempts to highlight the third class in a multi-class dataset:

```python
labels = ["Normal", "Warning", "Critical", "Fatal"]
samples = [1200, 350, 85, 20]
explode = (0, 0, 0.15)

plt.pie(samples, labels=labels, explode=explode)
plt.show()
```

What specific runtime error does Matplotlib raise?

- [ ] **A.** `IndexError: tuple index out of range`
- [ ] **B.** `KeyError: 'Fatal'`
- [ ] **C.** `TypeError: explode must be a list`
- [ ] **D.** `ValueError: len(explode) == len(x)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib validates sequence length before indexing individual elements.
- **Option B (Incorrect):** `KeyError` occurs with dictionary mappings, not sequence inputs.
- **Option C (Incorrect):** `explode` accepts tuples, lists, or arrays; the issue is length mismatch.
- **Option D (Correct):** Matplotlib enforces strict 1-to-1 cardinality between the data array `x` and the `explode` sequence. Because `samples` contains 4 elements while `explode` contains only 3 elements, Matplotlib raises `ValueError: len(explode) == len(x)`. To explode only the third class, the tuple must have 4 elements: `explode=(0, 0, 0.15, 0)`.

</details>

---

### 93. Question

An engineer creates a pie chart inside a wide rectangular figure window for an executive dashboard:

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.pie([55, 30, 15], labels=["Prod A", "Prod B", "Prod C"])
plt.show()
```

The resulting chart displays an elongated oval ellipse rather than a circle. What single line of code added before `plt.show()` cleanly resolves this distortion?

- [ ] **A.** `ax.force_circular()`
- [ ] **B.** `ax.axis('equal')` (or `ax.set_aspect('equal')`)
- [ ] **C.** `plt.pie(circular=True)`
- [ ] **D.** `fig.set_dpi(100)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `force_circular` is not an Axes method.
- **Option B (Correct):** In Matplotlib, non-square figure dimensions stretch the Cartesian axes box unless an aspect ratio constraint is explicitly imposed. Calling `ax.axis('equal')` or `ax.set_aspect('equal')` locks the horizontal and vertical scaling to equal data units, guaranteeing that circular wedges render as a true circle.
- **Option C (Incorrect):** `circular` is not a parameter of `plt.pie()`.
- **Option D (Incorrect):** Modifying DPI scales pixel density without altering the rectangular coordinate ratio.

</details>

---

### 94. Question

A junior analyst attempts to set percentage precision on a pie chart:

```python
plt.pie([40, 60], autopct=0.1)
```

When executed, what exception is raised by the Python interpreter?

- [ ] **A.** `ValueError: Invalid float format`
- [ ] **B.** `KeyError: 0.1`
- [ ] **C.** `TypeError: 'float' object is not callable`
- [ ] **D.** `AttributeError: float has no attribute 'format'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Python evaluates callable/string checks before format string parsing.
- **Option B (Incorrect):** Dictionaries are not involved.
- **Option C (Correct):** In `plt.pie()`, the `autopct` parameter accepts either a string format template (such as `'%1.1f%%'`) or a callable function/lambda. If an object is passed that is neither a string nor `None`, Matplotlib attempts to invoke it as a function: `autopct(pct)`. Because `0.1` is a floating-point number and cannot be invoked as a function, Python raises `TypeError: 'float' object is not callable`. The correct string syntax is `autopct='%1.1f%%'`.
- **Option D (Incorrect):** Python checks callable invocation syntax first.

</details>

---

### 95. Question

A developer attempts to build a donut chart using the central patch overlay method:

```python
fig, ax = plt.subplots()
centre_circle = plt.Circle((0, 0), 0.70, fc='white', zorder=1)
ax.add_artist(centre_circle)
ax.pie([50, 30, 20])
plt.show()
```

The resulting figure displays a standard solid pie chart with **no central hole**. What change fixes the script so the white circle appears on top of the wedges?

- [ ] **A.** Set the circle's radius to `1.5`
- [ ] **B.** Set `zorder=2` (or any value greater than the default pie wedge layer) on `centre_circle`, or add the circle artist **after** calling `ax.pie()`
- [ ] **C.** Change the circle fill color to `'black'`
- [ ] **D.** Replace `plt.Circle` with `plt.Wedge`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Radius `1.5` would cover the entire pie completely.
- **Option B (Correct):** Matplotlib renders artist primitives in order of `zorder` and sequential code execution. Because `ax.pie()` was called after `add_artist()`, the opaque wedges painted over the top of the circle. Increasing the circle's layer to `zorder=2` (or moving `ax.add_artist(centre_circle)` to after `ax.pie()`) ensures the white central disc renders on top of the wedges, creating the donut hole illusion.
- **Option C (Incorrect):** Changing color does not fix the layer stacking problem.
- **Option D (Incorrect):** `plt.Circle` is completely valid when properly layered.

</details>

---

### 96. Question

A data visualization specialist wants to customize all percentage labels generated by `ax.pie()` simultaneously (setting font size to 11, font weight to bold, and color to white). Which Matplotlib utility function applies these properties across the `autotexts` list in a single clean call?

- [ ] **A.** `ax.format_texts(autotexts, size=11, weight="bold", color="white")`
- [ ] **B.** `autotexts.set_style(size=11, bold=True)`
- [ ] **C.** `plt.setp(autotexts, size=11, weight="bold", color="white")`
- [ ] **D.** `plt.text_all(autotexts, fontsize=11)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `format_texts` is not an Axes method.
- **Option B (Incorrect):** Python lists do not have a `set_style` method.
- **Option C (Correct):** `plt.setp()` (**set p**roperties) is Matplotlib's standard vectorized property setter. It accepts a list of artist objects (such as `autotexts`) and applies keyword properties (`size=11`, `weight="bold"`, `color="white"`) to every element in the collection in a single line of code without requiring a manual `for` loop.
- **Option D (Incorrect):** `text_all` does not exist in Pyplot.

</details>

---

### 97. Question

A developer generates a pie chart without perimeter slice labels and attaches a dedicated legend:

```python
wedges, texts = ax.pie(sizes)
ax.legend(wedges, categories, loc="center left", bbox_to_anchor=(1, 0, 0.5))
```

Why is passing `wedges` as the first argument into `ax.legend()` necessary?

- [ ] **A.** Because Python requires all lists to be alphabetized before creating a legend.
- [ ] **B.** Because `ax.pie()` deletes the default handles array.
- [ ] **C.** Because `wedges` sets the font size of the legend.
- [ ] **D.** Because `wedges` contains the list of `matplotlib.patches.Wedge` artists that provide the color swatches and graphical handles needed to render the legend keys.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Alphabetization is not required.
- **Option B (Incorrect):** `ax.pie()` does not delete internal structures.
- **Option C (Incorrect):** Font size is governed by `fontsize`, not handles.
- **Option D (Correct):** In Matplotlib, `ax.legend(handles, labels)` expects artist handles whose face colors represent the plotted data. When `labels` is omitted from `ax.pie()`, the wedge artists lack automatic label metadata. Passing the returned `wedges` list explicitly provides the graphical patch handles so Matplotlib can render matching colored swatches next to each entry in `categories`.

</details>

---

### 98. Question

A backend batch reporting service generates 500 pie charts sequentially in a loop and exports each to disk using `plt.savefig()`:

```python
for report_id, data in client_datasets.items():
    fig, ax = plt.subplots()
    ax.pie(data['shares'], labels=data['categories'])
    plt.savefig(f"reports/{report_id}.png")
```

After processing approximately 30 reports, the system slows down and triggers:

`RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface are retained until explicitly closed and may consume too much memory.`

What single line of code placed at the end of each loop iteration resolves this memory leak?

- [ ] **A.** `plt.clf(clear_all=True)`
- [ ] **B.** `plt.close(fig)` (or `plt.close('all')`)
- [ ] **C.** `del ax`
- [ ] **D.** `ax.clear_memory()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.clf()` clears canvas elements, but the Figure instance itself remains registered in pyplot's global manager (`Gcf`), retaining memory.
- **Option B (Correct):** In Matplotlib's `pyplot` interface, every instantiated Figure is stored in an internal global registry until explicitly destroyed. In iterative loops, failing to call `plt.close(fig)` leaks figure canvases and artist trees in RAM, eventually exhausting system memory. Calling `plt.close(fig)` unregisters the figure and frees memory immediately.
- **Option C (Incorrect):** `del ax` deletes a local variable reference; it does not destroy the global pyplot figure registry.
- **Option D (Incorrect):** `clear_memory` is not an Axes method.

</details>

---

### 99. Question

A data analyst uses pandas to generate a quick pie chart directly from a categorical column:

```python
df['Device_Type'].value_counts().plot.pie(autopct='%1.1f%%', ylabel='')
```

Why is `ylabel=''` explicitly specified in this pandas plotting statement?

- [ ] **A.** Because pandas will raise an error if `ylabel` is not a string.
- [ ] **B.** Because setting `ylabel=''` turns the chart into a donut chart.
- [ ] **C.** Because pandas automatically sets the series name as a vertical y-axis label (rendering `"Device_Type"` along the left margin), which is visually redundant on a circular pie chart and should be suppressed.
- [ ] **D.** Because `ylabel` sets the color palette in pandas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plot.pie()` executes without error if `ylabel` is omitted.
- **Option B (Incorrect):** Donut charts require `wedgeprops={'width': ...}`; `ylabel` does not alter geometry.
- **Option C (Correct):** By default, pandas' `.plot.pie()` method treats the Series name as an axis label and calls `ax.set_ylabel(self.name)`. On a circular pie chart, this places a vertical text string (e.g., `"Device_Type"`) rotated $90^\circ$ on the left side of the circle, which looks awkward and unnecessary. Passing `ylabel=''` cleanly clears this label.
- **Option D (Incorrect):** Color palettes are configured via `colormap` or `colors`.

</details>

---

### 100. Question

Review the following publication-grade donut chart implementation:

```python
fig, ax = plt.subplots(figsize=(6, 6))

wedges, texts, autotexts = ax.pie(
    [45, 30, 15, 10],
    labels=["Search", "Direct", "Social", "Referral"],
    autopct="%1.1f%%",
    pctdistance=0.8,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    colors=["#2b5c8f", "#4682b4", "#6baed6", "#bdd7e7"]
)

ax.text(0, 0, "Traffic\nSources", ha="center", va="center", fontsize=13, fontweight="bold")
ax.axis("equal")
plt.show()
```

Which statement accurately describes the final rendered visual structure of this chart?

- [ ] **A.** A solid pie chart with black borders starting at 6 o'clock and moving counterclockwise
- [ ] **B.** A rectangular bar chart with four blue columns
- [ ] **C.** An oval donut with text placed at the top edge of the figure
- [ ] **D.** A circular donut chart with an inner hole radius of $0.65$, segmented by 2-point white borders, sweeping clockwise from 12 o'clock, with percentage text centered inside the $0.35$-wide ring at $r=0.8$, and bold text `"Traffic\nSources"` centered in the hole.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The chart is an annular donut (not solid), has white borders (not black), starts at 12 o'clock (not 6 o'clock), and sweeps clockwise.
- **Option B (Incorrect):** `ax.pie()` produces a polar circular chart, not a rectangular bar chart.
- **Option C (Incorrect):** `ax.axis('equal')` ensures a perfect circle (not oval), and `ax.text(0, 0)` places text in the exact center hole.
- **Option D (Correct):** Every component aligns with publication-grade standards:
  - Outer radius is $1.0$; with `width=0.35`, the inner hole has radius $1.0 - 0.35 = 0.65$.
  - Slices have crisp 2-point white dividing borders (`edgecolor='white', linewidth=2`).
  - `startangle=90, counterclock=False` begins at 12 o'clock and proceeds clockwise.
  - Ring band spans $r \in [0.65, 1.0]$ with midpoint $r = \frac{0.65 + 1.0}{2} = 0.825$, so `pctdistance=0.8` centers percentage text inside the band.
  - `ax.text(0, 0, ...)` anchors the central KPI in the middle of the void.
  - `ax.axis('equal')` ensures true circular geometry.

</details>

---
