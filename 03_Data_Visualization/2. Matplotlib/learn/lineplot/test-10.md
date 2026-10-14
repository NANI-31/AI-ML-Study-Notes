# Matplotlib Line Plot Objective Test — Set 10

---

### 91. Question

A data science intern runs the following script to compare experiment timestamps against sensor measurements:

```python
import matplotlib.pyplot as plt

timestamps = [10.1, 10.2, 10.3, 10.4, 10.5]
readings   = [52.1, 54.3, 53.8, 55.0]

plt.plot(timestamps, readings)
plt.show()
```

What exception is raised by Matplotlib when executing this code?

- [ ] **A.** `IndexError: list index out of range`
- [ ] **B.** `DimensionMismatchWarning` (warning only; Matplotlib truncates the longer array automatically)
- [ ] **C.** `TypeError: incompatible coordinate buffers`
- [ ] **D.** `ValueError: x and y must have same first dimension, but have shapes (5,) and (4,)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib performs dimension validation prior to iterating through indices, raising a `ValueError`.
- **Option B (Incorrect):** Matplotlib never silently truncates data; it enforces strict shape parity to prevent silent misalignment of timestamps with measurements.
- **Option C (Incorrect):** Standard Python lists of floats have compatible data types; the failure is strictly a dimension length mismatch.
- **Option D (Correct):** In Matplotlib, `plt.plot(x, y)` requires that the first dimension (length) of the horizontal sequence $x$ exactly matches the first dimension of the vertical sequence $y$. Because `timestamps` has 5 elements and `readings` has 4 elements, Matplotlib raises `ValueError: x and y must have same first dimension, but have shapes (5,) and (4,)`.

</details>

---

### 92. Question

A developer executes the following plotting commands:

```python
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.legend()
plt.show()
```

When the plot displays, the legend appears as a tiny, empty rectangular outline, and the Python console prints:

`UserWarning: No artists with labels found to put in legend. Note that artists whose label start with an underscore are ignored when legend() is called with no argument.`

What single change resolves this problem?

- [ ] **A.** Pass `labels=["Train", "Val"]` inside `plt.show()`
- [ ] **B.** Supply the `label` parameter inside each `plt.plot()` call: `plt.plot(..., label="Train")` and `plt.plot(..., label="Val")`
- [ ] **C.** Call `plt.legend(force=True)`
- [ ] **D.** Rename the variables `train_acc` and `val_acc` to start with capital letters

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.show()` does not accept labels or handle legend configuration.
- **Option B (Correct):** In Matplotlib, calling `plt.legend()` without arguments queries the current axes for all artist objects that have a user-defined label. If no `label` parameter was passed to `plt.plot()`, the lines retain their default internal name (which begins with an underscore, e.g., `'_line0'`), causing `legend()` to find zero plottable handles and emit a `UserWarning`. Adding `label="Train"` and `label="Val"` to the respective `plot()` calls populates the legend correctly.
- **Option C (Incorrect):** `plt.legend()` has no `force` parameter.
- **Option D (Incorrect):** Variable identifiers in Python have no bearing on artist metadata.

</details>

---

### 93. Question

An engineer plots a high-frequency sensor signal containing 3,000,000 data points. The resulting interactive window takes over 15 seconds to render and lags severely when zoomed or panned.

Which technique provides the most effective optimization for rendering massive continuous line datasets without freezing the GUI?

- [ ] **A.** Switch line color to `'k'` (black) because black pixels require fewer CPU clock cycles to rasterize
- [ ] **B.** Set `alpha=0.01` so Matplotlib skips drawing opaque vectors
- [ ] **C.** Downsample the series using systematic slicing (e.g., `plt.plot(x[::50], y[::50])`) or algorithms like LTTB (Largest Triangle Three Buckets), and enable path chunking via `plt.rcParams['agg.path.chunksize'] = 10000`
- [ ] **D.** Pass `gpu=True` to `plt.plot()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** The rendering bottleneck in 2D vector graphics is vertex transformation and line rasterization geometry, which is completely independent of RGB pixel color values.
- **Option B (Incorrect):** Adding transparency (`alpha < 1`) actually **increases** rendering overhead because it triggers alpha-blending calculations on every overlapping pixel.
- **Option C (Correct):** Human monitors have finite physical pixels (typically 1920 horizontally); attempting to render 3 million vector segments forces the renderer to draw thousands of lines per screen pixel, causing severe GPU/CPU pipeline congestion. Downsampling using algorithms like LTTB (which preserves visual peaks and troughs) or uniform slicing reduces the payload to a few thousand points without perceptible loss of waveform fidelity. Additionally, configuring `agg.path.chunksize` allows the rasterizer to break massive continuous paths into manageable sub-paths, preventing crashes.
- **Option D (Incorrect):** Standard Matplotlib does not provide a native `gpu=True` argument.

</details>

---

### 94. Question

A developer attempts to switch from Pyplot state-machine syntax to Matplotlib's recommended Object-Oriented (OO) API:

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y)
ax.title("Performance Trajectory")   # Line 3
ax.xlabel("Timestamp (ms)")          # Line 4
plt.show()
```

When run, the script crashes at Line 3 with `TypeError: 'str' object is not callable` or `AttributeError: 'Axes' object has no attribute 'title'`.

How should Line 3 and Line 4 be rewritten in the Object-Oriented interface?

- [ ] **A.** `ax.set_title("Performance Trajectory")` and `ax.set_xlabel("Timestamp (ms)")`
- [ ] **B.** `ax.put_title("Performance Trajectory")` and `ax.put_xlabel("Timestamp (ms)")`
- [ ] **C.** `ax.title = "Performance Trajectory"` and `ax.xlabel = "Timestamp (ms)"`
- [ ] **D.** `ax.draw_text("Performance Trajectory", position='top')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib's Object-Oriented API, methods on the `Axes` object typically use the `set_` prefix for properties that are called as bare functions in the procedural `pyplot` interface:
  - `plt.title()` $\rightarrow$ `ax.set_title()`
  - `plt.xlabel()` $\rightarrow$ `ax.set_xlabel()`
  - `plt.ylabel()` $\rightarrow$ `ax.set_ylabel()`
  - `plt.xlim()` $\rightarrow$ `ax.set_xlim()`
  - `plt.ylim()` $\rightarrow$ `ax.set_ylim()`
  On an `Axes` object, `ax.title` is an internal `Text` artist instance, not a callable setter method, so attempting `ax.title(...)` fails.
- **Option B (Incorrect):** `put_title` is not a Matplotlib method.
- **Option C (Incorrect):** `ax.title` cannot be assigned a raw string directly; it requires calling `ax.set_title()`.
- **Option D (Incorrect):** `draw_text` is not a standard Axes method.

</details>

---

### 95. Question

An analyst writes a line plot call with conflicting line style specifications:

```python
plt.plot(x, y, 'r-', linestyle=':')
```

What visual line style is rendered on the canvas?

- [ ] **A.** A solid red line (`'-'`), because shorthand format strings take precedence over keyword arguments
- [ ] **B.** An alternating solid-and-dotted hybrid stroke
- [ ] **C.** A Python syntax error is raised before execution begins
- [ ] **D.** A dotted red line (`':'`), because explicit keyword arguments override format string parameters

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Format string tokens are lower-precedence default fallbacks.
- **Option B (Incorrect):** Matplotlib does not merge contradictory line styles into a hybrid stroke.
- **Option C (Incorrect):** Matplotlib cleanly resolves keyword precedence without raising exceptions.
- **Option D (Correct):** In Matplotlib's argument resolution rules, explicit keyword arguments (`linestyle=':'`) strictly override any conflicting property tokens embedded inside the format string (`'r-'`). Therefore, the color `'r'` (red) from the format string is retained, but the solid line style `'-'` is overridden by the dotted style `':'`, rendering a **dotted red line**.

</details>

---

### 96. Question

A data scientist creates an annotated chart highlighting a critical inflection point on an ML convergence curve:

```python
plt.plot(epochs, loss, 'b-', label="Loss")
plt.annotate(
    "Overfitting Threshold",
    xy=(42, 0.18),
    xytext=(55, 0.45),
    arrowprops=dict(facecolor='crimson', shrink=0.05)
)
```

In the `plt.annotate()` call, what do the `xy` and `xytext` parameters designate?

- [ ] **A.** `xy` defines the coordinates of the text label; `xytext` defines the tip of the arrow.
- [ ] **B.** Both parameters specify bounding box diagonal corners for the text box.
- [ ] **C.** `xy` defines the point being pointed to (the target data coordinate at $(42, 0.18)$); `xytext` defines the coordinate location where the text label itself is placed ($(55, 0.45)$).
- [ ] **D.** `xy` is the figure dimension in inches; `xytext` is the text font size.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This inverts the roles of the two coordinate pairs.
- **Option B (Incorrect):** Bounding boxes for text are configured via the `bbox` dictionary, not `xy`/`xytext`.
- **Option C (Correct):** In `plt.annotate(text, xy, xytext, ...)`:
  - `xy`: Represents the point of interest (the arrow tip pointing at the specific data point $(x=42, y=0.18)$).
  - `xytext`: Represents the location where the annotation text label will be rendered ($(x=55, y=0.45)$).
  The connecting arrow points from `xytext` toward `xy`.
- **Option D (Incorrect):** Neither parameter controls physical canvas inches or font size.

</details>

---

### 97. Question

A data analyst processes daily sales trends across 200 distinct store branches using a script that generates and saves a separate plot for each branch:

```python
for store_id, sales_data in store_records.items():
    fig = plt.figure(figsize=(10, 4))
    plt.plot(sales_data['day'], sales_data['revenue'])
    plt.title(f"Store {store_id}")
    plt.savefig(f"reports/store_{store_id}.png")
```

After processing approximately 50 stores, the machine slows to a crawl and crashes with an `OutOfMemoryError`. Why did this happen?

- [ ] **A.** Matplotlib stores all past figures in an internal global manager (`Gcf`) indefinitely until explicitly closed; because `plt.close(fig)` was omitted inside the loop, 200 high-resolution figures remained in active RAM.
- [ ] **B.** The `.png` format consumes exponential memory with each write operation.
- [ ] **C.** Python's `for` loop cannot execute more than 20 iterations when dealing with Matplotlib figures.
- [ ] **D.** `plt.savefig()` requires internet access, causing buffer accumulation.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** When figures are created via `plt.figure()`, the `pyplot` module maintains a persistent global dictionary of all instantiated figures to allow state-based access. In batch loops, creating new figures without calling `plt.close(fig)` or `plt.close('all')` prevents garbage collection, accumulating canvas arrays and artist hierarchies in memory until system RAM is exhausted.
- **Option B (Incorrect):** PNG file exports write to disk and release local buffers; the leak stems from pyplot retaining the Figure objects in RAM.
- **Option C (Incorrect):** Python loops have no arbitrary 20-iteration limit.
- **Option D (Incorrect):** `plt.savefig()` operates strictly on the local filesystem.

</details>

---

### 98. Question

A scientist has an array of experimental measurements with known error bounds:
- Mean measurement: `mean_signal`
- Standard error: `sem`

Which combination of calls produces a clean, professional visualization showing the mean trend line flanked by an error envelope?

- [ ] **A.**
```python
plt.plot(time, mean_signal, color='#1f77b4', linewidth=2, label="Mean Signal")
plt.fill_between(time, mean_signal - sem, mean_signal + sem, color='#1f77b4', alpha=0.25, label="±1 SEM")
plt.legend()
```
- [ ] **B.**
```python
plt.plot(time, mean_signal)
plt.plot(time, sem, error=True)
```
- [ ] **C.**
```python
plt.error_line(time, mean_signal, sem)
```
- [ ] **D.**
```python
plt.plot(time, mean_signal, fill_range=sem)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Combining `plt.plot()` for the solid central mean trajectory with `plt.fill_between(x, y - error, y + error, alpha=0.25)` matching the line's color is the standard, publication-grade pattern for showing continuous uncertainty ribbons.
- **Option B (Incorrect):** `error=True` is not a valid parameter of `plt.plot()`.
- **Option C (Incorrect):** `plt.error_line()` does not exist in Matplotlib (the discrete error bar function is `plt.errorbar()`).
- **Option D (Incorrect):** `fill_range` is not a recognized keyword argument for `plt.plot()`.

</details>

---

### 99. Question

Consider the following script building a multi-layer diagnostic plot:

```python
plt.figure(figsize=(8, 4))
plt.axhspan(90, 100, color='lightgreen', alpha=0.5, zorder=1, label="Safe Zone")
plt.plot(time, baseline, 'k--', linewidth=1.5, zorder=2, label="Baseline")
plt.plot(time, reading, color='navy', marker='o', markersize=5, zorder=4, label="Reading")
plt.axhline(y=95, color='darkgreen', linestyle=':', zorder=3, label="Target")
plt.legend(loc="lower right")
```

When all elements are rendered, what is the exact stacking order from **bottom layer (drawn first)** to **top layer (drawn last)**?

- [ ] **A.** Reading $\rightarrow$ Target $\rightarrow$ Baseline $\rightarrow$ Safe Zone
- [ ] **B.** Baseline $\rightarrow$ Reading $\rightarrow$ Target $\rightarrow$ Safe Zone
- [ ] **C.** Target $\rightarrow$ Safe Zone $\rightarrow$ Reading $\rightarrow$ Baseline
- [ ] **D.** Safe Zone ($z=1$) $\rightarrow$ Baseline ($z=2$) $\rightarrow$ Target ($z=3$) $\rightarrow$ Reading ($z=4$)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** This reverses the z-order hierarchy.
- **Option B (Incorrect):** Misorders the layer hierarchy.
- **Option C (Incorrect):** Misorders the layer hierarchy.
- **Option D (Correct):** Artists are rendered in ascending order of their `zorder` attribute:
  1. Lowest ($z=1$): `axhspan` (Safe Zone background fill)
  2. Next ($z=2$): `baseline` (Black dashed line)
  3. Next ($z=3$): `axhline` (Dotted dark green target line)
  4. Highest ($z=4$): `reading` (Navy line with circular markers, rendering over all lines and fills)

</details>

---

### 100. Question

Review the following four statements regarding comprehensive Matplotlib line plot mastery:

1. In `plt.plot(x, y, fmt)`, the format string shorthand `'g^:'` produces a green dotted line with upward-pointing triangle markers.
2. The property cycler determines line colors based strictly on the procedural order in which `plot()` calls execute, not by the alphabetical value or content of the `label` string.
3. Connecting unordered nominal categories with a line plot violates visualization ethics by falsely implying continuous transitions and non-existent intermediate gradients.
4. Calling `plt.figure(figsize=(w, h))` must precede `plt.plot()` calls to ensure the plotted data is drawn onto the sized canvas.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, and 4 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Statement 1 is correct:** In `'g^:'`, `'g'` is green, `'^'` is upward triangle, and `':'` is dotted line style.
- **Statement 2 is correct:** The property cycler advances sequentially with each artist construction; `label` metadata has zero influence on palette cycling.
- **Statement 3 is correct:** Connecting nominal categories with line strokes falsely implies a continuum and meaningful slopes where none exist; discrete categories require bar charts.
- **Statement 4 is correct:** `plt.figure()` instantiates the canvas; calling it after `plt.plot()` creates a second, empty figure while leaving the plot on the initial default canvas.
- **Conclusion:** All four statements are correct, making **Option D** the correct choice.

</details>

---
