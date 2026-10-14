# Matplotlib Line Plot Objective Test — Set 04

---

### 31. Question

In modern versions of Matplotlib, what is the default color palette used by the property cycler (`axes.prop_cycler`) to assign line colors when no explicit color is specified?

- [ ] **A.** `classic` (`['b', 'g', 'r', 'c', 'm', 'y', 'k']`)
- [ ] **B.** `viridis` (perceptually uniform sequential colormap)
- [ ] **C.** `grayscale` (monochrome shades from black to light gray)
- [ ] **D.** `tab10` (Tableau 10 categorical palette: `C0` through `C9`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** The `classic` palette (pure blue, green, red, cyan, magenta, yellow, black) was used prior to Matplotlib 2.0 and is now deprecated as the default.
- **Option B (Incorrect):** `viridis` is the default continuous colormap used for image heatmaps (`imshow`) and contour plots, not the discrete categorical cycle for `plt.plot()`.
- **Option C (Incorrect):** `grayscale` is a specialized accessibility style, not the default configuration.
- **Option D (Correct):** Since Matplotlib 2.0, the default categorical color cycle is based on the **`tab10`** palette (Tableau 10). It cycles through ten carefully curated, perceptually distinct colors denoted symbolically as `'C0'`, `'C1'`, `'C2'`, ..., `'C9'`. The first default color (`'C0'`) is a muted steel blue (`#1f77b4`), followed by orange (`'C1'`), green (`'C2'`), and red (`'C3'`).

</details>

---

### 32. Question

A data analyst writes a script comparing two retail divisions:

```python
# Version 1
plt.plot(in_store_sales, label="In-Store")
plt.plot(online_sales, label="Online")
plt.legend()
plt.show()
```

Later, another engineer modifies the script by reordering the plotting calls:

```python
# Version 2
plt.plot(online_sales, label="Online")
plt.plot(in_store_sales, label="In-Store")
plt.legend()
plt.show()
```

What visual change occurs between Version 1 and Version 2?

- [ ] **A.** No visual change occurs because Matplotlib matches colors to the alphabetical string value of `label`.
- [ ] **B.** The line colors swap: "In-Store" was blue (`C0`) in Version 1 but becomes orange (`C1`) in Version 2, while "Online" changes from orange (`C1`) to blue (`C0`).
- [ ] **C.** The legend text updates, but both lines remain rendered in identical shades of green.
- [ ] **D.** Matplotlib raises an `AmbiguousColorMappingError` due to inconsistent script ordering.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Matplotlib never parses label text or uses strings to assign colors; color cycle advancement is strictly driven by the procedural order of plotting calls.
- **Option B (Correct):** Whichever series is passed to the first `plt.plot()` call consumes `'C0'` (default steel blue), and the second call consumes `'C1'` (default orange). Reversing the execution order swaps which series is drawn first, causing "Online" to be assigned `'C0'` (blue) and "In-Store" to be assigned `'C1'` (orange).
- **Option C (Incorrect):** The lines are not green; they receive the first two colors of the cycle (`C0` blue and `C1` orange).
- **Option D (Incorrect):** Reordering calls is completely valid standard Python code; no error is raised.

</details>

---

### 33. Question

An ML engineer plots three training curves using default color cycling:

```python
plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Validation")
plt.plot(test_loss, label="Test")
plt.legend()
```

In Matplotlib's default `tab10` color cycle, which colors are assigned to `Train`, `Validation`, and `Test` respectively?

- [ ] **A.** Train: Blue (`C0`), Validation: Orange (`C1`), Test: Green (`C2`)
- [ ] **B.** Train: Red (`C0`), Validation: Green (`C1`), Test: Blue (`C2`)
- [ ] **C.** Train: Black (`C0`), Validation: White (`C1`), Test: Gray (`C2`)
- [ ] **D.** Train: Yellow (`C0`), Validation: Cyan (`C1`), Test: Magenta (`C2`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The first three entries of the default `tab10` cycle are:
  1. `C0`: Blue (`#1f77b4`) $\rightarrow$ assigned to `Train`
  2. `C1`: Orange (`#ff7f0e`) $\rightarrow$ assigned to `Validation`
  3. `C2`: Green (`#2ca02c`) $\rightarrow$ assigned to `Test`
- **Option B (Incorrect):** This mimics the classic primary color scheme, not the modern `tab10` cycle.
- **Option C (Incorrect):** The default palette is multi-chromatic, not grayscale.
- **Option D (Incorrect):** Yellow, cyan, and magenta are later or deprecated palette tokens, not the first three `tab10` colors.

</details>

---

### 34. Question

A scientist wants to plot data across two separate subplots but wants a specific metric ("Pressure") to have the exact same blue color in both subplots, regardless of what other series are plotted before it. How can the scientist explicitly specify the first color of the active cycle?

- [ ] **A.** `color='default_1'`
- [ ] **B.** `color='cycle[0]'`
- [ ] **C.** `color='C0'`
- [ ] **D.** `color='auto'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `'default_1'` is not a valid Matplotlib color string.
- **Option B (Incorrect):** `'cycle[0]'` is not a recognized syntax; the standard shorthand is `'C0'`.
- **Option C (Correct):** Matplotlib supports special color notation strings of the form `'CN'`, where `N` is an integer index from `0` to `9` (e.g., `'C0'`, `'C1'`, `'C2'`). Specifying `color='C0'` explicitly binds the artist to the first color in the current active property cycler, ensuring absolute color consistency across disparate subplots or figures.
- **Option D (Incorrect):** Passing `'auto'` raises a `ValueError: 'auto' is not a valid color value`.

</details>

---

### 35. Question

An analytics team is tracking 12 different regional branches on a single line chart without specifying manual colors:

```python
for branch_name, sales_series in branches_12.items():
    plt.plot(sales_series, label=branch_name)
```

What color will Matplotlib assign to the **11th** regional branch plotted?

- [ ] **A.** It raises a `CycleExhaustedError` because `tab10` only contains 10 colors.
- [ ] **B.** It defaults to black (`#000000`) for all series beyond the 10th.
- [ ] **C.** It cycles back to the beginning of the cycler and assigns `'C0'` (steel blue).
- [ ] **D.** It generates a random RGB color using `numpy.random`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The property cycler is an infinite cyclic iterator (`itertools.cycle`); it never exhausts or raises an error.
- **Option B (Incorrect):** Matplotlib does not default to black after 10 entries; it loops back to the start.
- **Option C (Correct):** The cycler operates modulo the length of the palette ($N \pmod{10}$). The 11th series corresponds to index 10, which evaluates to $10 \pmod{10} = 0$, re-assigning `'C0'` (the same steel blue as the 1st series). The 12th series receives `'C1'` (orange).
- **Option D (Incorrect):** Color assignment is strictly deterministic based on the active palette cycler, never random.

</details>

---

### 36. Question

A developer executes the following single-line plot command:

```python
lines = plt.plot(x, y1, 'r-', x, y2, 'b--', x, y3, 'g:')
```

How many `Line2D` objects are created and stored inside the `lines` list?

- [ ] **A.** 1
- [ ] **B.** 6
- [ ] **C.** 3
- [ ] **D.** 0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Each `(x, y, fmt)` group produces an independent line artist; they are not merged into a single multi-segment object.
- **Option B (Incorrect):** The $x$ and $y$ arrays are coordinate pairs defining lines, not individual artists; 6 arrays define 3 lines.
- **Option C (Correct):** `plt.plot()` supports arbitrary variable-length positional triplets: `plt.plot(x1, y1, fmt1, x2, y2, fmt2, ...)`. In this call, three distinct series are plotted: a red solid line, a blue dashed line, and a green dotted line. Consequently, `lines` is a list containing exactly **3** distinct `Line2D` artist objects.
- **Option D (Incorrect):** The call successfully instantiates 3 line artists; the return list is not empty.

</details>

---

### 37. Question

An organization wants all data visualizations across the company to adopt a branded corporate palette (`#1A365D`, `#E53E3E`, `#319795`) instead of default `tab10`. How can this be configured globally at the start of a script?

- [ ] **A.** `matplotlib.set_palette(['#1A365D', '#E53E3E', '#319795'])`
- [ ] **B.** `plt.color_cycle(['#1A365D', '#E53E3E', '#319795'])`
- [ ] **C.** Global colors cannot be changed; colors must be passed manually to every `plt.plot()` call.
- [ ] **D.** `from cycler import cycler; plt.rcParams['axes.prop_cycler'] = cycler(color=['#1A365D', '#E53E3E', '#319795'])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `matplotlib.set_palette()` is not a standard Matplotlib function (it is a Seaborn convenience method).
- **Option B (Incorrect):** `plt.color_cycle()` was a legacy Matplotlib 1.x function that has long been deprecated and removed.
- **Option C (Incorrect):** Matplotlib offers extensive global configuration via `rcParams`.
- **Option D (Correct):** Matplotlib uses the `cycler` library to manage property cycles. Overwriting `plt.rcParams['axes.prop_cycler']` with a `cycler(color=[...])` object globally resets the default series colors for all subsequent line plots in the runtime session.

</details>

---

### 38. Question

A financial analyst builds a two-panel dashboard comparing revenue between 2023 and 2024:
- Panel 1 plots 2023 revenue first, then 2024 revenue.
- Panel 2 plots 2024 revenue first, then 2023 revenue.

Both panels rely on default automatic color cycling. Why does this design represent poor visualization practice?

- [ ] **A.** Because `tab10` colors cannot be rendered on side-by-side subplots.
- [ ] **B.** Because Matplotlib will throw a warning when identical variable names are plotted in inverted order.
- [ ] **C.** Because Panel 2 will automatically inherit the axis scaling of Panel 1.
- [ ] **D.** Because the semantic meaning of the colors is inverted between panels: 2023 is blue in Panel 1 but orange in Panel 2, forcing the viewer to constantly cross-reference legends and causing cognitive dissonance.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Multi-panel subplots handle color cycles cleanly and without rendering errors.
- **Option B (Incorrect):** Matplotlib does not track variable names or issue warnings regarding call order.
- **Option C (Incorrect):** Subplots maintain independent axis limits unless explicitly linked with `sharex` or `sharey`.
- **Option D (Correct):** Color should maintain a consistent, stable semantic mapping across related visual panels. When execution order dictates color assignment, inverted plotting calls cause "2023" to appear blue in one chart and orange in the adjacent chart. This breaks visual consistency and forces cognitive fatigue. To prevent this, developers should explicitly pass fixed colors (e.g., `color='#2B6CB0'` for 2023 and `color='#ED8936'` for 2024).

</details>

---

### 39. Question

A developer wants to plot a reference baseline in black and then two experimental models using automatic cycle colors:

```python
plt.plot(x, baseline, color='black', label="Baseline")
plt.plot(x, model_1, label="Model 1")
plt.plot(x, model_2, label="Model 2")
plt.legend()
```

Does the first call `plt.plot(..., color='black')` advance the underlying property cycler?

- [ ] **A.** Yes; calling `plt.plot()` advances the property cycler regardless of whether an explicit color was supplied, so `Model 1` receives `C1` (orange) and `Model 2` receives `C2` (green).
- [ ] **B.** No; passing an explicit keyword color bypasses and does not advance the property cycler, so `Model 1` receives `C0` (blue) and `Model 2` receives `C1` (orange).
- [ ] **C.** The cycler is permanently paused whenever a string color is detected.
- [ ] **D.** All subsequent lines are forced to render in black.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, each invocation of `Axes.plot()` that constructs a `Line2D` artist requests the next entry from the axis property cycler. Even if an explicit keyword argument like `color='black'` overrides the line's color, the cycler was already incremented during call setup. Therefore, the next line (`Model 1`) receives `'C1'` (orange) rather than `'C0'`.
- **Option B (Incorrect):** This is a common intuition, but Matplotlib's internal line construction consumes a cycler element on every `plot()` call before applying keyword overrides.
- **Option C (Incorrect):** The cycler is an iterator that continues advancing normally.
- **Option D (Incorrect):** Explicit color applies strictly to the artist instantiated in that specific call.

</details>

---

### 40. Question

Review the following statements regarding multi-series line plots and color cycling in Matplotlib:

1. Passing multiple `(x, y)` pairs inside a single `plt.plot()` call assigns distinct colors to each series from the active color cycler.
2. Label text passed to `label="Some Name"` has zero influence on the color assigned to a line.
3. Specifying `color='C3'` explicitly selects the 4th color of the active cycle (zero-indexed).
4. `plt.legend()` automatically collects labels from all line artists in the current axes that have non-empty label strings.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** `plt.plot(x, y1, x, y2)` advances the cycler for each pair, giving `y1` and `y2` different colors automatically.
- **Statement 2 is correct:** `label` is strictly metadata for the legend and has no impact on palette indexing.
- **Statement 3 is correct:** `'CN'` syntax uses zero-indexing: `C0` is the 1st color, `C1` the 2nd, `C2` the 3rd, and `C3` the 4th.
- **Statement 4 is correct:** `plt.legend()` queries the axes for all artists with a valid label string (labels not starting with an underscore) and builds the legend entries accordingly.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
