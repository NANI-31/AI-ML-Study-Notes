# Matplotlib Ticks & Scales Objective Test — Set 01

---

### Question 1

In modern Matplotlib (v3.3+), executing `ax.set_xticklabels(["Alpha", "Beta", "Gamma"])` on a newly created `AxesSubplot` without previously calling `ax.set_xticks()` raises a prominent `UserWarning`. 

What is the technical cause of this warning, and how is it resolved?

- [ ] A) Setting tick labels directly attaches a `FixedFormatter` without first establishing a `FixedLocator`, meaning Matplotlib's default dynamic locator may change tick count during resize/zoom and cause label misalignment; calling `ax.set_xticks(ticks)` first establishes fixed tick coordinates.
- [ ] B) `set_xticklabels()` requires Greek lettering to be encoded via LaTeX math strings `$\alpha$`, so passing standard string literals triggers a font warning.
- [ ] C) Matplotlib requires `ax.set_xticklabels()` to be passed as a dictionary mapping float coordinates to strings rather than a Python list.
- [ ] D) The warning indicates that `set_xticklabels()` has been removed and replaced by `ax.set_labels()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Setting tick labels directly attaches a `FixedFormatter` without first establishing a `FixedLocator`, meaning Matplotlib's default dynamic locator may change tick count during resize/zoom and cause label misalignment; calling `ax.set_xticks(ticks)` first establishes fixed tick coordinates.

**Explanation:**

- **A is correct:** In Matplotlib, an axis has both a `Locator` (determining coordinate positions) and a `Formatter` (determining display text). Calling `set_xticklabels(labels)` automatically applies a `FixedFormatter`. If the underlying locator is still a dynamic locator (such as `AutoDateLocator` or `MaxNLocator`), zooming, panning, or resizing the canvas dynamically alters the number of tick positions, causing the fixed list of labels to map onto the wrong coordinates. Matplotlib emits `UserWarning: FixedFormatter should only be used together with FixedLocator`. Calling `ax.set_xticks(ticks)` explicitly installs a `FixedLocator` at those exact coordinates, ensuring a 1-to-1 persistent match.
- **B is incorrect:** String encoding has nothing to do with `FixedFormatter`; arbitrary strings are supported without LaTeX syntax.
- **C is incorrect:** `set_xticklabels()` accepts sequences (lists or tuples) of strings, not dictionaries.
- **D is incorrect:** `set_xticklabels()` is not removed; it remains part of the API when paired with `set_xticks()`.

</details>

---

### Question 2

A machine learning engineer evaluates 4 neural network architectures (`['ResNet', 'VGG', 'DenseNet', 'EfficientNet']`) with classification accuracies `[0.88, 0.82, 0.91, 0.94]`. 

Which code snippet correctly constructs a bar plot with discrete integer positions and clean model names on the horizontal axis?

- [ ] A) `plt.bar(['ResNet', 'VGG', 'DenseNet', 'EfficientNet'], [0.88, 0.82, 0.91, 0.94]); plt.set_positions([1, 2, 3, 4])`
- [ ] B)
  ```python
  x_pos = [0, 1, 2, 3]
  models = ["ResNet", "VGG", "DenseNet", "EfficientNet"]
  acc = [0.88, 0.82, 0.91, 0.94]
  plt.bar(x_pos, acc)
  plt.xticks(x_pos, models)
```
- [ ] C)
  ```python
  plt.bar(acc, x_pos)
  plt.xticks(models)
```
- [ ] D)
  ```python
  plt.plot(models, acc)
  plt.xticks(tick_type="categorical")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B)
```python
x_pos = [0, 1, 2, 3]
models = ["ResNet", "VGG", "DenseNet", "EfficientNet"]
acc = [0.88, 0.82, 0.91, 0.94]
plt.bar(x_pos, acc)
plt.xticks(x_pos, models)
```

**Explanation:**

- **A is incorrect:** `plt.set_positions()` does not exist in the Matplotlib API.
- **B is correct:** In classic Matplotlib bar chart workflows, numeric coordinates (such as integer indices `[0, 1, 2, 3]`) are passed to `plt.bar()`. Calling `plt.xticks(ticks, labels)` overrides the numeric coordinates on the x-axis, mapping each integer index directly to the corresponding semantic model name in `models`.
- **C is incorrect:** Passing `acc` as the x-coordinate swaps axes, and `plt.xticks(models)` without explicit coordinate positions will not align correctly with integer bar centers.
- **D is incorrect:** `tick_type` is not a valid parameter for `plt.xticks()`.

</details>

---

### Question 3

A data analyst executes the following code on an existing subplot `ax`:

```python
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xticklabels(["Q1", "Q2", "Q3"])
```

What behavior occurs when the canvas is drawn?

- [ ] A) Matplotlib throws an uncatchable `RuntimeError` and terminates the Python process.
- [ ] B) Matplotlib automatically creates two additional labels `"Q4"` and `"Q5"` to match the 5 tick locations.
- [ ] C) Matplotlib applies `"Q1"`, `"Q2"`, and `"Q3"` to coordinates $0, 10,$ and $20$, while ticks at $30$ and $40$ are left with blank labels (or emits a warning regarding mismatched lengths).
- [ ] D) The ticks at $0, 10,$ and $20$ are removed from the canvas, leaving only ticks $30$ and $40$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Matplotlib applies `"Q1"`, `"Q2"`, and `"Q3"` to coordinates $0, 10,$ and $20$, while ticks at $30$ and $40$ are left with blank labels (or emits a warning regarding mismatched lengths).

**Explanation:**

- **A is incorrect:** Matplotlib does not terminate the Python process on label count mismatches.
- **B is incorrect:** Matplotlib never invents or extrapolates synthetic label strings like `"Q4"` or `"Q5"`.
- **C is correct:** When the list of labels is shorter than the list of specified tick locations, Matplotlib zips the available labels onto the tick coordinates sequentially. The remaining unassigned tick marks (here, at coordinates $30$ and $40$) receive empty strings, rendering tick marks without text. In recent versions, Matplotlib also flags this length mismatch with a warning or visual inconsistency.
- **D is incorrect:** The tick marks themselves remain at all 5 coordinate locations defined by `set_xticks()`.

</details>

---

### Question 4

A visualization specialist wants to create a clean sparkline or mini-plot where data is plotted, but all horizontal tick marks, tick labels, and numbers are completely suppressed along the x-axis.

Which command achieves this cleanly while preserving the axis spine and plotted data lines?

- [ ] A) `ax.set_xlim(None, None)`
- [ ] B) `ax.xaxis.delete()`
- [ ] C) `ax.spines['bottom'].set_visible(False)`
- [ ] D) `ax.set_xticks([])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.set_xticks([])`

**Explanation:**

- **A is incorrect:** `set_xlim(None, None)` triggers automatic data limit autoscaling rather than removing ticks.
- **B is incorrect:** `XAxis` objects do not have a `.delete()` method.
- **C is incorrect:** Hiding the bottom spine removes the bounding baseline stroke, but leaves the tick lines and numeric text labels floating in whitespace.
- **D is correct:** Passing an empty sequence `[]` to `ax.set_xticks([])` (or `plt.xticks([])`) establishes a `FixedLocator` with zero tick marks. This cleanly eliminates all major tick lines and labels along that axis while keeping the bottom spine, grid configuration, and plotted data completely intact.

</details>

---

### Question 5

A financial ML pipeline generates revenue predictions in thousands of dollars: `revenues = [150, 320, 480, 610, 790]`. An engineer wants the y-axis to display ticks at these exact numbers, with each label formatted as `"$150k"`, `"$320k"`, etc.

Which code snippet achieves this using the Object-Oriented interface?

- [ ] A)
  ```python
  ax.set_yticks(revenues)
  ax.set_yticklabels([f"${val}k" for val in revenues])
```
- [ ] B)
  ```python
  ax.set_yticks([f"${val}k" for val in revenues])
```
- [ ] C)
  ```python
  ax.set_yticklabels(revenues, prefix="$", suffix="k")
```
- [ ] D)
  ```python
  ax.yaxis.format_currency(revenues, unit="k")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
ax.set_yticks(revenues)
ax.set_yticklabels([f"${val}k" for val in revenues])
```

**Explanation:**

- **A is correct:** In Matplotlib's OO design, `ax.set_yticks()` sets the numeric coordinates where tick marks will appear, and `ax.set_yticklabels()` supplies the corresponding formatted display strings. Using list comprehension `[f"${val}k" for val in revenues]` maps each numeric coordinate into its financial string representation.
- **B is incorrect:** `set_yticks()` requires numeric coordinates (floats or ints); passing formatted string literals directly into `set_yticks()` causes a `TypeError` or `ValueError`.
- **C is incorrect:** `set_yticklabels()` does not take `prefix` or `suffix` keyword arguments.
- **D is incorrect:** `format_currency` is not a method on `YAxis`.

</details>

---

### Question 6

An engineer plots a scatter plot with points at $x = [10, 20, 30]$. They subsequently run:

```python
ax.set_xticklabels(["Low", "Medium", "High"])
```

Does running this command alter the underlying numerical x-coordinates of the plotted data points?

- [ ] A) Yes; the data points are automatically transformed to coordinates $0.0, 1.0,$ and $2.0$.
- [ ] B) No; modifying tick labels alters only the human-readable text rendered on the canvas; the underlying data points remain at their original numerical coordinates.
- [ ] C) Yes; Matplotlib re-indexes the Pandas dataframe associated with the plot.
- [ ] D) No, but it converts the x-axis projection into a 3D perspective.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No; modifying tick labels alters only the human-readable text rendered on the canvas; the underlying data points remain at their original numerical coordinates.

**Explanation:**

- **A is incorrect:** Data coordinate geometry is completely decouple from tick label rendering.
- **B is correct:** In Matplotlib's graphics pipeline, plotted artists (`Line2D`, `PathCollection`, etc.) store their data values in data coordinates. Tick labels are text display artists attached to the `Axis` helper. Changing tick labels merely alters the text strings rendered at specific tick coordinates; it never shifts, transforms, or mutates the underlying data points.
- **C is incorrect:** Matplotlib does not mutate the caller's input DataFrames.
- **D is incorrect:** Projections are managed during axis creation (`projection='3d'`), not by label changes.

</details>

---

### Question 7

Which pair of methods allows a developer to programmatically inspect the current numerical tick coordinates and the text strings currently assigned to the x-axis of an `Axes` object?

- [ ] A) `ax.read_ticks()` and `ax.read_labels()`
- [ ] B) `ax.xaxis.locations` and `ax.xaxis.texts`
- [ ] C) `ax.get_xticks()` and `[label.get_text() for label in ax.get_xticklabels()]`
- [ ] D) `ax.inspect_x(values=True)` and `ax.inspect_x(strings=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.get_xticks()` and `[label.get_text() for label in ax.get_xticklabels()]`

**Explanation:**

- **A is incorrect:** `read_ticks()` and `read_labels()` do not exist in the Matplotlib API.
- **B is incorrect:** `xaxis.locations` and `xaxis.texts` are not valid attributes on `XAxis`.
- **C is correct:** `ax.get_xticks()` returns a 1D NumPy array of the numeric coordinates where major ticks are located. `ax.get_xticklabels()` returns a list of Matplotlib `Text` artist instances; to retrieve their rendered string values, one calls `.get_text()` on each text artist.
- **D is incorrect:** `inspect_x` is not a valid method on `Axes`.

</details>

---

### Question 8

A weather monitoring dashboard tracks temperature readings across five sensor locations. An engineer wants the x-axis to display ticks at integer positions $1, 2, 3, 4, 5$ with degree Celsius labels: `"15°C"`, `"20°C"`, `"25°C"`, `"30°C"`, `"35°C"`.

Using the stateful `pyplot` interface, which command correctly configures both the positions and the labels simultaneously?

- [ ] A) `plt.ticks(x=[1, 2, 3, 4, 5], labels=["15°C", "20°C", "25°C", "30°C", "35°C"])`
- [ ] B) `plt.xlabel(ticks=[1, 2, 3, 4, 5], text=["15°C", "20°C", "25°C", "30°C", "35°C"])`
- [ ] C) `plt.set(xticks=[1, 2, 3, 4, 5], format="°C")`
- [ ] D) `plt.xticks([1, 2, 3, 4, 5], ["15°C", "20°C", "25°C", "30°C", "35°C"])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.xticks([1, 2, 3, 4, 5], ["15°C", "20°C", "25°C", "30°C", "35°C"])`

**Explanation:**

- **A is incorrect:** The function is `plt.xticks()`, not `plt.ticks()`.
- **B is incorrect:** `plt.xlabel()` sets the axis title text (e.g., `"Temperature"`), not tick mark positions.
- **C is incorrect:** `plt.set()` does not exist with this signature.
- **D is correct:** In the `pyplot` stateful interface, `plt.xticks(ticks, labels)` accepts two positional arguments: the first is an array-like sequence of numeric tick positions, and the second is the sequence of display label strings.

</details>

---

### Question 9

A multi-metric dashboard uses a twin axis to display training loss on the primary y-axis and learning rate on the secondary y-axis:

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_yticks([0.0, 0.5, 1.0])
```

Why does setting `ax1.set_yticks([0.0, 0.5, 1.0])` leave `ax2`'s tick positions and labels completely untouched?

- [ ] A) `ax1.twinx()` creates an independent `Axes` instance that shares the horizontal x-axis but possesses its own distinct `YAxis` object, complete with independent limits, locators, and formatters.
- [ ] B) Because `ax2` is rendered in background memory and only synchronizes when `plt.show()` is invoked.
- [ ] C) Because `twinx` disables all ticks on `ax2` by default until `ax2.enable_ticks()` is called.
- [ ] D) Because `ax1.set_yticks()` only applies to integers, whereas `ax2` defaults to floating-point numbers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax1.twinx()` creates an independent `Axes` instance that shares the horizontal x-axis but possesses its own distinct `YAxis` object, complete with independent limits, locators, and formatters.

**Explanation:**

- **A is correct:** `twinx()` instantiates a new `AxesSubplot` overlaying the original canvas. The two axes share the exact same `XAxis` transform and coordinate system, but each has its own separate `YAxis` object positioned on opposite sides (left vs. right). Consequently, configuring ticks on `ax1` has zero effect on the ticks, scale, or limits of `ax2`.
- **B is incorrect:** Matplotlib does not defer twin axis synchronization to `plt.show()`.
- **C is incorrect:** `ax2` has active ticks placed on the right spine by default; there is no `enable_ticks()` method.
- **D is incorrect:** `set_yticks()` accepts arbitrary floating-point numbers.

</details>

---

### Question 10

In Matplotlib's internal object model, what core classes represent the horizontal and vertical coordinate axes, managing the associated locators, formatters, tick marks, and tick labels?

- [ ] A) `matplotlib.canvas.FigureCanvas` and `matplotlib.backend.Renderer`
- [ ] B) `matplotlib.axis.XAxis` and `matplotlib.axis.YAxis` (accessed via `ax.xaxis` and `ax.yaxis`)
- [ ] C) `matplotlib.scale.ScaleManager` and `matplotlib.spines.SpineManager`
- [ ] D) `matplotlib.ticker.TickContainer` and `matplotlib.lines.Line2D`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `matplotlib.axis.XAxis` and `matplotlib.axis.YAxis` (accessed via `ax.xaxis` and `ax.yaxis`)

**Explanation:**

- **A is incorrect:** `FigureCanvas` and `Renderer` handle low-level GUI backend drawing and rasterization, not axis tick mechanics.
- **B is correct:** In Matplotlib's artist hierarchy, an `Axes` instance owns two `Axis` instances: `ax.xaxis` (`matplotlib.axis.XAxis`) and `ax.yaxis` (`matplotlib.axis.YAxis`). These classes manage tick generation, tick placement via `Locator` subclasses, label text formatting via `Formatter` subclasses, and major/minor tick artist collections.
- **C is incorrect:** `ScaleManager` and `SpineManager` are not the primary axis controllers in the Matplotlib artist model.
- **D is incorrect:** `TickContainer` is not a standard Matplotlib class, and `Line2D` represents individual plotted lines.

</details>

---
