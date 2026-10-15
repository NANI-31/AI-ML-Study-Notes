# Matplotlib Line Plot Objective Test — Set 03

---

### 21. Question

A data engineer executes the following block of code to generate an executive report:

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [10, 25, 18, 30])
plt.figure(figsize=(12, 5))
plt.title("Quarterly Server Latency")
plt.show()
```

What visual problem occurs in the rendered output?

- [ ] **A.** The line plot is rendered inside the 12x5 inch window, but its stroke color defaults to yellow.
- [ ] **B.** A `CanvasOrderingError` is raised because `plt.figure()` cannot be called once data has been ingested.
- [ ] **C.** The plot is automatically scaled down to a 1x1 inch thumbnail because of the conflicting execution sequence.
- [ ] **D.** Two separate figures are produced: Figure 1 contains the plotted line with default sizing, and Figure 2 is an empty 12x5 inch canvas with the title.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Calling `plt.figure()` does not retrospectively resize the previously instantiated active figure or alter stroke colors.
- **Option B (Incorrect):** Matplotlib does not throw an error; it treats the second call as an explicit request for a new canvas.
- **Option C (Incorrect):** Matplotlib does not downscale the figure into a thumbnail; it creates two full-sized figures.
- **Option D (Correct):** In Matplotlib's state-machine interface, calling `plt.plot()` before any `plt.figure()` automatically instantiates a default figure (Figure 1) and draws the line onto it. The subsequent call to `plt.figure(figsize=(12, 5))` opens a **new, separate** figure (Figure 2) that receives the subsequent `plt.title()` call. As a result, two windows are generated: one containing the untitled line plot and one containing an empty titled figure.

</details>

---

### 22. Question

A machine learning researcher prepares a high-resolution figure for publication with the following specifications:

```python
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(epochs, validation_accuracy)
plt.savefig("val_acc.png")
```

What are the exact pixel dimensions (width $\times$ height) of the resulting exported image file `val_acc.png` (assuming default bounding box settings)?

- [ ] **A.** $800 \times 500$ pixels
- [ ] **B.** $1500 \times 800$ pixels
- [ ] **C.** $1200 \times 750$ pixels
- [ ] **D.** $600 \times 400$ pixels

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** $800 \times 500$ pixels would result from a DPI of 100 ($8 \times 100 = 800$, $5 \times 100 = 500$).
- **Option B (Incorrect):** This miscalculates the product of width and height with DPI.
- **Option C (Correct):** The pixel dimensions of an exported figure are calculated as:
  $$\text{Width in pixels} = \text{figsize width (inches)} \times \text{DPI} = 8 \times 150 = 1200\text{ px}$$
  $$\text{Height in pixels} = \text{figsize height (inches)} \times \text{DPI} = 5 \times 150 = 750\text{ px}$$
  Therefore, the rendered raster output is exactly $1200 \times 750$ pixels.
- **Option D (Incorrect):** $600 \times 400$ would correspond to an undersized DPI of 75.

</details>

---

### 23. Question

A junior developer wants to adjust the dimensions of their line plot right at display time. They write:

```python
plt.plot(timestamps, cpu_usage)
plt.show(figsize=(10, 4))
```

What is the result of running this code?

- [ ] **A.** Matplotlib resizes the existing window to 10 inches by 4 inches immediately before popping up the interactive viewer.
- [ ] **B.** The figure renders in full-screen presentation mode with a 10:4 aspect ratio.
- [ ] **C.** The figure is saved automatically to disk as `figsize_10x4.png`.
- [ ] **D.** `plt.show()` does not accept a `figsize` argument; it either ignores unrecognized keyword arguments or raises a `TypeError`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.show()` is responsible only for starting the GUI main loop and displaying rendered figures; it does not handle canvas dimension configuration.
- **Option B (Incorrect):** Full-screen presentation is configured via backend window managers or `plt.get_current_fig_manager().full_screen_toggle()`, not by passing `figsize` to `show()`.
- **Option C (Incorrect):** `plt.show()` never saves files to disk; disk writes require `plt.savefig()`.
- **Option D (Correct):** Canvas geometry must be established at figure instantiation time via `plt.figure(figsize=(width, height))`. Passing `figsize` to `plt.show()` is a common mistake; `plt.show()` accepts arguments related to backend event loops (like `block=True`), not layout dimensions.

</details>

---

### 24. Question

What is the primary difference between `plt.clf()`, `plt.cla()`, and `plt.close()` in Matplotlib workflow management?

- [ ] **A.** `plt.cla()` clears the current axes; `plt.clf()` clears the current figure (removing all axes); `plt.close()` terminates and frees the figure window from memory.
- [ ] **B.** `plt.clf()` clears the axes, `plt.cla()` closes the GUI window, and `plt.close()` clears the title text only.
- [ ] **C.** All three are identical aliases that perform the exact same cleanup routine.
- [ ] **D.** `plt.close()` clears internal cache without affecting visible windows, while `plt.clf()` closes Python entirely.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Matplotlib establishes a clear hierarchy of cleanup operations:
  - `plt.cla()` (**C**lear **A**xes): Clears all plotted data and artists from the currently active axes, leaving the figure intact.
  - `plt.clf()` (**C**lear **F**igure): Clears the entire figure, removing all subplots and axes from the canvas.
  - `plt.close()`: Closes the figure window entirely and releases its associated memory and GUI backend resources.
- **Option B (Incorrect):** This reverses `cla` and `clf` and mischaracterizes `close`.
- **Option C (Incorrect):** They operate on different levels of the Matplotlib artist hierarchy (Axes, Figure, and Window Manager).
- **Option D (Incorrect):** `plt.clf()` does not exit the Python interpreter, and `plt.close()` specifically closes active figures.

</details>

---

### 25. Question

A backend batch job generates 5,000 diagnostic line plots in a sequential loop and exports each one using `plt.savefig()`. After processing approximately 20 charts, the process emits:

`RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface are retained until explicitly closed and may consume too much memory.`

What single line of code should be placed at the end of each loop iteration to resolve this memory leak?

- [ ] **A.** `plt.clf(all=True)`
- [ ] **B.** `plt.grid(False)`
- [ ] **C.** `plt.close('all')` or `plt.close(fig)`
- [ ] **D.** `del plt`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `plt.clf()` clears canvas elements, but the Figure instance itself remains registered in pyplot's global figure manager (`_pylab_helpers.Gcf`), retaining memory.
- **Option B (Incorrect):** Turning off gridlines has no impact on figure memory allocation.
- **Option C (Correct):** Pyplot maintains global references to all instantiated figures until they are explicitly destroyed. In batch processing loops, calling `plt.close(fig)` (or `plt.close('all')`) unregisters the figure from the pyplot manager and allows Python's garbage collector to reclaim allocated memory.
- **Option D (Incorrect):** Deleting the module reference `plt` causes subsequent loop iterations to fail with a `NameError`.

</details>

---

### 26. Question

What are the default dimensions (in inches) of a Matplotlib figure when created without specifying `figsize` under standard default `rcParams`?

- [ ] **A.** `[6.4, 4.8]` inches
- [ ] **B.** `[10.0, 8.0]` inches
- [ ] **C.** `[5.0, 5.0]` inches
- [ ] **D.** `[19.2, 10.8]` inches

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In modern Matplotlib (v2.0+), the default figure size configured in `mpl.rcParams['figure.figsize']` is **`[6.4, 4.8]` inches** with a 4:3 aspect ratio. At default 100 DPI, this yields an initial canvas of $640 \times 480$ pixels.
- **Option B (Incorrect):** $10.0 \times 8.0$ inches is an older default from early scientific plotting libraries.
- **Option C (Incorrect):** Square $5.0 \times 5.0$ is not the standard Matplotlib default canvas.
- **Option D (Incorrect):** $19.2 \times 10.8$ corresponds to a 1080p 16:9 widescreen canvas, which must be set explicitly.

</details>

---

### 27. Question

Consider the following script executed in a standard non-interactive Python script environment:

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
plt.savefig("trend.png")
```

When the user inspects `trend.png` after the window is closed, they discover that the saved image is completely blank or corrupt. Why did this occur?

- [ ] **A.** `plt.savefig()` only works if invoked on a multi-subplot layout.
- [ ] **B.** Matplotlib requires a paid license to execute both `show()` and `savefig()` in the same runtime.
- [ ] **C.** `trend.png` failed because `.png` files do not support continuous line plots.
- [ ] **D.** In GUI backends, closing the window in `plt.show()` destroys or clears the current figure state; calling `plt.savefig()` afterwards operates on an empty canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `plt.savefig()` functions equally well on single-axes and multi-subplot figures.
- **Option B (Incorrect):** Matplotlib is open-source software distributed under a BSD-compatible license; there are no licensing restrictions.
- **Option C (Incorrect):** PNG is a standard raster format universally used for Matplotlib line plots.
- **Option D (Correct):** In traditional GUI backends, closing the figure window at the conclusion of `plt.show()` triggers canvas deallocation and resets the current active figure (`plt.gcf()`). Consequently, any subsequent call to `plt.savefig()` captures a newly reset, blank canvas. Best practice dictates calling `plt.savefig()` **before** `plt.show()`.

</details>

---

### 28. Question

An analyst wants to create two separate line plot figures in the same script, alternating between them to add curves. Which approach correctly switches between figures using their integer identifiers?

- [ ] **A.** `plt.select_canvas(1)` and `plt.select_canvas(2)`
- [ ] **B.** `plt.figure(1)` to activate or create Figure 1, and `plt.figure(2)` to activate or create Figure 2
- [ ] **C.** `plt.switch_to("Figure 1")`
- [ ] **D.** Figures cannot be modified once another figure has been opened

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** There is no `select_canvas()` function in the `pyplot` interface.
- **Option B (Correct):** `plt.figure(num)` accepts an integer identifier. If a figure with that `num` already exists, Matplotlib sets it as the current active figure; if it does not exist, a new figure with that identifier is created. This allows developers to seamlessly direct subsequent plotting commands to specific figures.
- **Option C (Incorrect):** `switch_to()` is not a valid Matplotlib function.
- **Option D (Incorrect):** Matplotlib allows full multi-figure state switching and concurrent figure manipulation.

</details>

---

### 29. Question

A data visualization specialist wants to export a line chart without excess white margins around the axes. Which keyword argument passed to `plt.savefig()` automatically trims surrounding whitespace?

- [ ] **A.** `tight_margins=True`
- [ ] **B.** `crop_white=True`
- [ ] **C.** `bbox_inches='tight'`
- [ ] **D.** `pad=0`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** `tight_margins` is not a valid parameter of `plt.savefig()`.
- **Option B (Incorrect):** `crop_white` is not a recognized Matplotlib argument.
- **Option C (Correct):** Passing `bbox_inches='tight'` to `plt.savefig()` instructs Matplotlib to compute the tight bounding box around all figure elements (axes, labels, titles, legends) and crop excess canvas whitespace, producing clean, publication-ready images.
- **Option D (Incorrect):** `pad=0` alone does not activate tight bounding box calculation.

</details>

---

### 30. Question

Review the following statements regarding Matplotlib figure lifecycle and configuration:

1. `plt.figure(figsize=(10, 6))` measures width and height in inches, not pixels or centimeters.
2. Increasing the `dpi` (dots per inch) increases the pixel resolution of an exported image without altering the relative typographic proportion of titles and labels.
3. Multiple calls to `plt.plot()` without calling `plt.figure()` in between will automatically draw multiple lines on the same axes.
4. Calling `plt.close()` without arguments closes all active figures simultaneously.

Which statements are **correct**?

- [ ] **A.** 1 and 4 only
- [ ] **B.** 2 and 3 only
- [ ] **C.** 1, 2, and 3 only
- [ ] **D.** 1, 2, 3, and 4

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** `figsize=(width, height)` specifies physical canvas dimensions in inches.
- **Statement 2 is correct:** `dpi` scales the pixel density. Because font sizes are specified in points (1/72 inch), higher DPI values cleanly increase pixel fidelity while maintaining consistent visual proportions.
- **Statement 3 is correct:** Matplotlib's default state retains the active axes; successive `plt.plot()` calls overlay new line traces on top of previously plotted data.
- **Statement 4 is incorrect:** Calling `plt.close()` without arguments closes only the **current active figure** (`plt.gcf()`). To close all open figures simultaneously, one must pass the string `'all'` (`plt.close('all')`).
- **Conclusion:** Statements 1, 2, and 3 are correct, making **Option C** the correct choice.

</details>

---
