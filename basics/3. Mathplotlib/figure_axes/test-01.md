# Matplotlib Figure & Axes Objective Test — Set 01
# Topic: Object Hierarchy & Pyplot vs. OO API (Questions 1–10)

---

### Question 1

A developer inspects the Matplotlib object hierarchy for the first time and draws the following diagram:

```
Figure
  └── Canvas
  └── Axes
        └── Axis (x)
        └── Axis (y)
        └── Lines, Patches, Artists...
```

Which statement about this hierarchy is **correct**?

- [ ] A) `Canvas` is a subclass of `Axes` and renders only the plot region
- [ ] B) `Axis` (singular) and `Axes` (plural) refer to the same object — both represent the coordinate plotting area
- [ ] C) `Figure` is the top-level container; `Canvas` is the backend rendering surface attached to it; `Axes` is the coordinate system where data is drawn; `Axis` is the x or y ruler containing ticks and labels
- [ ] D) `Figure` contains one and only one `Axes` object by design

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `Figure` is the top-level container; `Canvas` is the backend rendering surface attached to it; `Axes` is the coordinate system where data is drawn; `Axis` is the x or y ruler containing ticks and labels

**Explanation:**

- **A is incorrect:** `Canvas` is NOT a subclass of `Axes`. The `Canvas` (`FigureCanvas`) is the backend-specific rendering surface attached to the `Figure` — it translates the Matplotlib artist tree into pixels (for raster backends like Agg) or vector paths (for PDF/SVG backends). It operates at the figure level, not the axes level.
- **B is incorrect:** `Axis` (singular) and `Axes` (plural) are two completely different classes — one of the most common misconceptions in Matplotlib. `Axes` (with an 'e') is the plotting coordinate system (the "box" where your data lives). `Axis` (without 'e') is the individual x-axis or y-axis ruler that contains ticks, tick labels, and the axis line (spine). Each `Axes` object has two `Axis` objects: `ax.xaxis` and `ax.yaxis`.
- **C is correct:** This is the complete, accurate Matplotlib hierarchy. `Figure` → `Canvas` (backend renderer), `Figure` → one or more `Axes` (coordinate spaces) → each `Axes` → two `Axis` objects (x and y rulers) plus collections of `Line2D`, `Patch`, `Text`, and other `Artist` objects.
- **D is incorrect:** A `Figure` can contain multiple `Axes` objects — this is the entire point of `plt.subplots(2, 3)`, `fig.add_axes(...)`, and `fig.add_subplot(...)`. A single-axes figure is the simplest case, not a design constraint.

</details>

---

### Question 2

A developer writes the following code using the **Pyplot stateful API**:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Loss Curve")
plt.xlabel("Epoch")
ax = plt.gca()
print(type(ax))
```

What does `plt.gca()` return?

- [ ] A) The current `Figure` object
- [ ] B) The current `Axes` object (the active coordinate system where the last plot was drawn)
- [ ] C) The current `Canvas` (backend renderer)
- [ ] D) A new blank `Axes` object appended to the figure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The current `Axes` object (the active coordinate system where the last plot was drawn)

**Explanation:**

- **A is incorrect:** `plt.gca()` stands for "get current axes" — NOT "get current figure." To retrieve the current figure object, use `plt.gcf()` ("get current figure").
- **B is correct:** `plt.gca()` returns the **current active `Axes`** object from Matplotlib's implicit state machine. In this code, `plt.figure()` creates a figure, `plt.plot()` implicitly creates an `Axes` and makes it the current one, and `plt.gca()` retrieves that `Axes`. The print would show `<class 'matplotlib.axes._subplots.AxesSubplot'>` or similar. This is part of the Pyplot stateful API where Matplotlib tracks the "current" figure and axes globally.
- **C is incorrect:** The `Canvas` is not directly accessible via `plt.gca()`. To access the canvas, use `fig.canvas` after obtaining the figure via `plt.gcf()`.
- **D is incorrect:** `plt.gca()` retrieves the **existing** current axes — it does NOT create a new one (unless no axes exist yet, in which case it creates one). It does not append additional axes.

</details>

---

### Question 3

A machine learning engineer switches from the Pyplot API to the Object-Oriented API for a production dashboard. Which of the following code blocks demonstrates the **Object-Oriented (OO) paradigm** correctly?

- [ ] A)
```python
plt.figure()
plt.plot(epochs, loss)
plt.title("Training Loss")
```

- [ ] B)
```python
fig = plt.figure()
plt.plot(epochs, loss)
ax = plt.gca()
ax.set_title("Training Loss")
```

- [ ] C)
```python
fig, ax = plt.subplots()
ax.plot(epochs, loss)
ax.set_title("Training Loss")
```

- [ ] D)
```python
plt.subplots()
plt.plot(epochs, loss)
plt.title("Training Loss")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)

**Explanation:**

- **A is incorrect:** This is pure Pyplot stateful API — `plt.figure()`, `plt.plot()`, `plt.title()` all use the global implicit state machine. There is no explicit reference to the `Figure` or `Axes` objects.
- **B is incorrect:** While `fig = plt.figure()` gets an explicit Figure reference, `plt.plot()` and `plt.gca()` still rely on the global state machine for drawing. Mixing OO and stateful approaches is not considered clean OO design, especially in multi-axes scenarios.
- **C is correct:** `fig, ax = plt.subplots()` is the canonical OO pattern — it creates and returns both the `Figure` and `Axes` objects as explicit Python variables. All subsequent operations (`ax.plot()`, `ax.set_title()`) are called directly on the `ax` object, with no reliance on global Pyplot state. This is required for libraries, multithreaded apps, and complex dashboards.
- **D is incorrect:** `plt.subplots()` without capturing the return values `fig, ax = ...` discards the objects. The subsequent `plt.plot()` and `plt.title()` again use the stateful API — this is still Pyplot style, not OO style.

</details>

---

### Question 4

A developer asks: "When is it **strictly necessary** to use the Object-Oriented API instead of the Pyplot stateful API?" Which answer is most accurate?

- [ ] A) Always — the Pyplot API is deprecated and should never be used
- [ ] B) When creating any figure with more than one axis (subplot)
- [ ] C) When building reusable library functions, working in multithreaded environments, or managing multiple figures simultaneously where relying on global state would produce ambiguous or incorrect results
- [ ] D) Only when exporting figures with `dpi > 100`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) When building reusable library functions, working in multithreaded environments, or managing multiple figures simultaneously where relying on global state would produce ambiguous or incorrect results

**Explanation:**

- **A is incorrect:** The Pyplot API is NOT deprecated. It remains the recommended approach for interactive work, exploratory analysis, quick prototyping, and Jupyter notebooks where global state is acceptable and convenient. `plt.plot()`, `plt.show()`, etc., are still first-class Matplotlib features.
- **B is incorrect:** The Pyplot API handles subplots perfectly well — `plt.subplot(2, 1, 1)`, `plt.subplot(2, 1, 2)` switch between panels using the global state machine. Multi-subplot figures don't by themselves require the OO API (though it becomes more readable).
- **C is correct:** The OO API is strictly necessary when: (1) writing library/module code where the "current figure" might be someone else's figure in the calling code; (2) running in multithreaded environments where the global Pyplot state is not thread-safe; (3) working programmatically with many figures simultaneously and needing explicit control over which figure and axes receive operations.
- **D is incorrect:** DPI level has no relationship to which API should be used. Both APIs can set DPI via `figsize`/`dpi` parameters or `fig.set_dpi()`.

</details>

---

### Question 5

A data scientist examines the following code and identifies the type of each object:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
print(type(fig))
print(type(axes))
print(type(axes[0, 0]))
```

What does each `print` statement output?

- [ ] A) `Figure`, `ndarray`, `AxesSubplot`
- [ ] B) `Figure`, `list`, `AxesSubplot`
- [ ] C) `Canvas`, `ndarray`, `Axes`
- [ ] D) `Figure`, `ndarray`, `Axis`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `Figure`, `ndarray`, `AxesSubplot`

**Explanation:**

- **A is correct:** `fig` is a `matplotlib.figure.Figure` object. `axes` (the second return value of `plt.subplots(2, 2)`) is a **2D NumPy `ndarray`** of shape `(2, 2)` containing `AxesSubplot` objects. `axes[0, 0]` uses NumPy 2D indexing to retrieve the `AxesSubplot` in the top-left position. The precise class name is `matplotlib.axes._subplots.AxesSubplot` (or `Axes` in Matplotlib ≥ 3.6 which unified the naming).
- **B is incorrect:** `axes` is a NumPy `ndarray`, not a Python `list`. `plt.subplots(nrows, ncols)` returns a NumPy array when either nrows > 1 or ncols > 1, enabling 2D indexing like `axes[0, 0]`. A Python list would not support this syntax.
- **C is incorrect:** `fig` is a `Figure`, not a `Canvas`. The `Canvas` is accessed via `fig.canvas`. The `Canvas` is not what `plt.subplots()` returns.
- **D is incorrect:** `axes[0, 0]` returns an `AxesSubplot` (the coordinate plotting area), NOT an `Axis`. `Axis` (singular) is the x/y ruler — accessed as `axes[0, 0].xaxis` or `axes[0, 0].yaxis`.

</details>

---

### Question 6

A developer examines a figure's `Artist` tree and finds the following relationship:

```
Figure
  ├── FigureCanvasAgg   ← backend renderer
  ├── Axes [0]
  │     ├── XAxis
  │     ├── YAxis
  │     ├── Line2D
  │     └── Text (title)
  └── Text (suptitle)
```

Where does `fig.suptitle("My Title")` place the resulting `Text` artist?

- [ ] A) Inside `Axes[0]` as a child of the first subplot
- [ ] B) Directly on the `Figure` object — as a child of the `Figure`, not any individual `Axes`
- [ ] C) On the `Canvas` backend renderer layer
- [ ] D) On `Axes[0].XAxis` as a label artist

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Directly on the `Figure` object — as a child of the `Figure`, not any individual `Axes`

**Explanation:**

- **A is incorrect:** `fig.suptitle()` is a **Figure-level** operation. The resulting `Text` object is added to the `Figure`'s artist list, not to any `Axes`. This is precisely why it appears as a "super title" that spans the entire figure above all subplots.
- **B is correct:** `fig.suptitle()` adds a `Text` artist directly to the `Figure` container. In the artist tree shown, the `Text (suptitle)` node is a direct child of `Figure` — at the same level as the `Axes` objects. This means it is positioned in normalized figure coordinates (0 to 1), spans the full figure width, and appears above all subplots. Contrast this with `ax.set_title()` which creates a `Text` child of that specific `Axes`.
- **C is incorrect:** The `Canvas` is the rendering surface — it does not contain artist children. The `Figure` contains artists; the `Canvas` renders them. You cannot add a `Text` artist "to the canvas" directly.
- **D is incorrect:** `Axes[0].XAxis` contains tick labels and the x-axis label. The `suptitle` is a separate, figure-level text element positioned above the subplot area — not associated with any `Axis` object.

</details>

---

### Question 7

A developer uses `plt.figure()` and then `fig.add_subplot()` in a loop. Each call to `fig.add_subplot(1, 1, 1)` with the same arguments does NOT add a new axes — instead it replaces or returns the existing one. Why?

- [ ] A) `add_subplot` always creates a new axes regardless of arguments
- [ ] B) `add_subplot` checks the subplot specification (nrows, ncols, index) and returns an existing axes if one with the same spec already exists in the figure, avoiding duplication
- [ ] C) `add_subplot` fails silently and returns `None` on the second call
- [ ] D) The global state machine prevents multiple `add_subplot` calls to the same figure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `add_subplot` checks the subplot specification (nrows, ncols, index) and returns an existing axes if one with the same spec already exists in the figure, avoiding duplication

**Explanation:**

- **A is incorrect:** `fig.add_subplot(1, 1, 1)` called twice with identical arguments does NOT create two separate axes. Matplotlib is designed to detect when the same subplot grid position is requested and return the existing axes rather than creating a duplicate.
- **B is correct:** `Figure.add_subplot()` internally compares the subplot specification tuple `(nrows, ncols, index)` against existing subplots. If a match is found, the existing `AxesSubplot` is returned (and set as the current axes). This prevents the common mistake of accidentally creating layered invisible axes during interactive development when a cell is re-run. To force a new axes at the same position (replacing the old one), you must call `fig.clf()` first or use `ax = fig.add_subplot(1, 1, 1, label="new_unique_label")` to distinguish it.
- **C is incorrect:** `add_subplot` never returns `None` on a valid call — it always returns an `AxesSubplot` object.
- **D is incorrect:** The global state machine tracks the "current" axes but does not prevent multiple `add_subplot` calls. The deduplication is a property of `Figure.add_subplot()` itself, independent of global state.

</details>

---

### Question 8

A developer calls `plt.close(fig)` after generating a figure in a loop. A colleague suggests using `fig.clf()` instead. What is the key difference between `plt.close(fig)` and `fig.clf()`?

- [ ] A) `plt.close(fig)` clears all axes but keeps the figure in memory; `fig.clf()` deletes the figure from memory
- [ ] B) `plt.close(fig)` removes the figure from Matplotlib's figure manager and frees its memory; `fig.clf()` clears all artists from the figure but keeps the figure object alive and registered
- [ ] C) Both are identical — they both destroy the figure and free memory
- [ ] D) `plt.close(fig)` only works in interactive backends; `fig.clf()` works in all backends

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.close(fig)` removes the figure from Matplotlib's figure manager and frees its memory; `fig.clf()` clears all artists from the figure but keeps the figure object alive and registered

**Explanation:**

- **A is incorrect:** This reverses the actual behavior. It is `fig.clf()` (clear figure) that keeps the figure object alive while removing its content, and `plt.close(fig)` that deregisters and frees it.
- **B is correct:** `plt.close(fig)` calls `Figure.close()` internally, which removes the figure from Matplotlib's internal `_pylab_helpers.Gcf` figure manager, disconnects it from any GUI window, and allows Python's garbage collector to free the memory (once no other references hold the figure). `fig.clf()` ("clear figure") removes all axes, text, patches, and other artists from the figure object, but the figure itself remains registered in the manager and in memory — it is blank but still alive. Use `fig.clf()` when you want to reuse the same figure object; use `plt.close(fig)` when you are done with the figure entirely.
- **C is incorrect:** They are NOT identical. `fig.clf()` produces a blank-but-alive figure; `plt.close(fig)` destroys and deregisters it.
- **D is incorrect:** `plt.close()` works across all backends — interactive (Qt5Agg, TkAgg) and non-interactive (Agg). The distinction is about memory management, not backend compatibility.

</details>

---

### Question 9

A developer writes the following function for a model evaluation library:

```python
def plot_roc(fpr, tpr, auc_score):
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
```

A senior engineer says: "This function is not safe for use in a library." Why?

- [ ] A) `plt.show()` cannot be called inside a function
- [ ] B) The function relies on Matplotlib's global state machine — it implicitly modifies `plt.gcf()`, making it unsafe when called from code that may have an existing active figure, potentially drawing on the wrong figure
- [ ] C) `figsize=(6, 5)` is an invalid tuple format for library code
- [ ] D) Library functions must always use `dpi=300` to be production-safe

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The function relies on Matplotlib's global state machine — it implicitly modifies `plt.gcf()`, making it unsafe when called from code that may have an existing active figure, potentially drawing on the wrong figure

**Explanation:**

- **A is incorrect:** `plt.show()` CAN be called inside a function. Whether it should be is a design question, but it's not a syntax or runtime error.
- **B is correct:** The function uses `plt.figure()`, `plt.plot()`, `plt.xlabel()`, etc. — all of which operate on the **global current figure**. If the calling code has already created a figure (e.g., the user created `fig, ax = plt.subplots()` before calling this function), the `plt.figure()` inside the function creates a NEW figure and sets it as current — but if the library were written differently and used `plt.plot()` without first creating a figure, it might draw on the caller's figure. More importantly, the correct pattern for library code is the OO API: `fig, ax = plt.subplots(); ax.plot(...); return fig, ax` — letting the caller control display, saving, and figure lifecycle.
- **C is incorrect:** `figsize=(6, 5)` is perfectly valid Python and Matplotlib syntax for both library and script code.
- **D is incorrect:** There is no rule requiring `dpi=300` in library functions. DPI is a configuration choice, not a library safety requirement.

</details>

---

### Question 10

A developer asks: "What is the `Figure.patch` attribute?" Which answer is correct?

- [ ] A) The network of spine lines forming the axes border
- [ ] B) A `Rectangle` patch artist that fills the entire figure background — styling it changes the figure's background color
- [ ] C) The collection of all `Line2D` objects drawn on the figure
- [ ] D) The list of `Axes` objects contained in the figure

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `Rectangle` patch artist that fills the entire figure background — styling it changes the figure's background color

**Explanation:**

- **A is incorrect:** Spine lines are `Spine` objects accessible via `ax.spines["top"]`, `ax.spines["bottom"]`, etc. They form the border of the plot area within an `Axes` — not the figure background.
- **B is correct:** `fig.patch` is a `matplotlib.patches.Rectangle` that fills the **entire figure canvas** from corner to corner. Calling `fig.patch.set_facecolor("lightgray")` is equivalent to `fig.set_facecolor("lightgray")` — both modify the figure's outer background color. Similarly, each `Axes` also has an `ax.patch` (`Rectangle`) that fills the inner plotting area, which is styled separately via `ax.set_facecolor()` or `ax.patch.set_facecolor()`.
- **C is incorrect:** `Line2D` objects are stored in `ax.lines` (a list on the `Axes` object), not on the `Figure` directly. `Figure.patch` is a single `Rectangle`, not a collection of lines.
- **D is incorrect:** The list of `Axes` objects is accessible via `fig.axes` (a list). `fig.patch` is a single background rectangle, not an axes collection.

</details>
