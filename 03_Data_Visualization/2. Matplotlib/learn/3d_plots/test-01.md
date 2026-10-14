# 3D Matplotlib — Test 01: 3D Setup & Projection Architecture

---

### Question 1

A machine learning engineer writes the following setup code to prepare a 3D figure:

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.set_zlabel("Loss")
```

When this code is executed, what happens?

- [ ] A) The subplot is created with a 3D projection and the z-axis label is set to "Loss" without errors
- [ ] B) A `NotImplementedError` is raised because `add_subplot(111)` is deprecated in favour of `add_subplot(1, 1, 1)`
- [ ] C) An `AttributeError` is raised because the default 2D `Axes` object has no `set_zlabel` method
- [ ] D) The plot renders correctly but `set_zlabel` silently draws the label on the y-axis instead

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) An `AttributeError` is raised because the default 2D `Axes` object has no `set_zlabel` method

**Explanation:**
- **A is incorrect:** Without `projection="3d"` in `add_subplot`, Matplotlib creates a standard 2D `Axes` object (`matplotlib.axes.Axes`), not an `Axes3D` instance. The 2D Axes class does not have `set_zlabel`, `set_zlim`, or any other z-axis method.
- **B is incorrect:** `add_subplot(111)` and `add_subplot(1, 1, 1)` are fully equivalent shorthands — neither is deprecated. The number 111 is decoded as rows=1, cols=1, index=1.
- **C is correct:** The 2D `Axes` object returned by `fig.add_subplot(111)` (with no `projection` keyword) does not expose a `set_zlabel` attribute. Calling it raises `AttributeError: 'Axes' object has no attribute 'set_zlabel'`. The fix is to pass `projection="3d"`: `ax = fig.add_subplot(111, projection="3d")`.
- **D is incorrect:** Matplotlib does not silently redirect z-axis calls to the y-axis. The call raises an error immediately rather than producing any visual output.

</details>

---

### Question 2

Which of the following code snippets correctly creates a single 3D axis using the `plt.subplots` factory function?

- [ ] A) `fig, ax = plt.subplots(1, 1, projection="3d")`
- [ ] B) `fig, ax = plt.subplots(subplot_kw={"projection": "3d"})`
- [ ] C) `fig, ax = plt.subplots(kind="3d")`
- [ ] D) `fig, ax = plt.subplots(1, 1, axes_class=Axes3D)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig, ax = plt.subplots(subplot_kw={"projection": "3d"})`

**Explanation:**
- **A is incorrect:** `plt.subplots` does not accept a `projection` keyword at the top level. Positional arguments to `plt.subplots` are `nrows` and `ncols`. Passing `projection="3d"` at the top level raises a `TypeError`.
- **B is correct:** `plt.subplots` accepts a `subplot_kw` dictionary that is forwarded as keyword arguments to every subplot it creates. `subplot_kw={"projection": "3d"}` is the idiomatic way to create a 3D-enabled figure-axes pair with the `plt.subplots` factory. Internally it passes `projection="3d"` to `add_subplot`.
- **C is incorrect:** `kind="3d"` is not a parameter of `plt.subplots`. This pattern is used by pandas `.plot(kind=...)`, not Matplotlib's figure factory.
- **D is incorrect:** While it is technically possible to instantiate `Axes3D` directly using the `axes_class` keyword in some Matplotlib versions, this is a low-level approach that is not the documented modern API. The canonical method is the `subplot_kw` dictionary.

</details>

---

### Question 3

After creating a 3D axis with `ax = fig.add_subplot(111, projection="3d")`, a researcher sets the axis labels with:

```python
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
```

A colleague proposes replacing `ax.set_zlabel("PC3")` with `ax.zaxis.set_label_text("PC3")`. Which statement about this replacement is accurate?

- [ ] A) The replacement is invalid because `ax.zaxis` is a private attribute and should not be accessed directly
- [ ] B) Both statements produce the same visual result; `set_zlabel` is a convenience wrapper around the lower-level `zaxis.set_label_text` call
- [ ] C) The replacement targets the y-axis label by mistake because `zaxis` is an alias for `yaxis` in `Axes3D`
- [ ] D) `ax.zaxis.set_label_text` requires a font size argument and raises `TypeError` when called with only a string

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Both statements produce the same visual result; `set_zlabel` is a convenience wrapper around the lower-level `zaxis.set_label_text` call

**Explanation:**
- **A is incorrect:** `ax.zaxis` is a public `ZAxis` object (a subclass of `XAxis`/`YAxis`) that Matplotlib exposes on every `Axes3D` instance. It is part of the documented API and is not a private attribute.
- **B is correct:** `Axes3D.set_zlabel(text)` is a thin convenience wrapper. Internally it delegates to `self.zaxis.set_label_text(text)`. Both paths write to the same `ZAxis` label, producing identical visual output. Many z-axis formatting operations can be performed on either `ax` or `ax.zaxis` directly.
- **C is incorrect:** `ax.zaxis` refers exclusively to the z-axis of the 3D projection. It is a distinct object from `ax.yaxis` and `ax.xaxis`.
- **D is incorrect:** `set_label_text` accepts a single positional string argument and optional keyword arguments (such as `fontsize`). The `fontsize` argument is not required; calling `ax.zaxis.set_label_text("PC3")` without it is perfectly valid.

</details>

---

### Question 4

A data scientist runs the following code and is surprised by the result:

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # explicit import

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
print(type(ax))
```

What does the `print` statement output?

- [ ] A) `<class 'matplotlib.axes._axes.Axes'>`
- [ ] B) `<class 'mpl_toolkits.mplot3d.axes3d.Axes3D'>`
- [ ] C) `<class 'matplotlib.figure.Figure'>`
- [ ] D) `<class 'mpl_toolkits.mplot3d.axes3d.Axes3DSubplot'>`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `<class 'mpl_toolkits.mplot3d.axes3d.Axes3DSubplot'>`

**Explanation:**
- **A is incorrect:** `matplotlib.axes._axes.Axes` is the standard 2D axes class. The `projection="3d"` keyword causes Matplotlib's projection registry to return an `Axes3D`-derived object instead.
- **B is incorrect:** While `Axes3D` is the conceptual base class, Matplotlib dynamically generates a `Axes3DSubplot` subclass (inheriting from both `Axes3D` and `SubplotBase`) when the axis is created inside a figure grid. The actual runtime type reported by `type()` is `Axes3DSubplot`.
- **C is incorrect:** `fig` is the `Figure` instance; `ax` is the axes object returned as the second element of the `subplots` tuple.
- **D is correct:** When created via `add_subplot` or `subplots`, Matplotlib creates a dynamically generated `Axes3DSubplot` class that inherits the full `Axes3D` interface but also includes subplot-positioning methods from `SubplotBase`. This is why `type(ax)` prints `Axes3DSubplot`.

</details>

---

### Question 5

A researcher wants to restrict the visible z-axis range of a 3D scatter plot to show only loss values between 0.0 and 2.5. Which call achieves this?

- [ ] A) `ax.zlim(0.0, 2.5)`
- [ ] B) `ax.set_zlim([0.0, 2.5])`
- [ ] C) `ax.set_zlim(0.0, 2.5)`
- [ ] D) `ax.zaxis.set_lim(0.0, 2.5)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.set_zlim(0.0, 2.5)`

**Explanation:**
- **A is incorrect:** `ax.zlim()` does not exist as a method. The naming convention in Matplotlib uses `set_` prefixes for setter methods: `set_xlim`, `set_ylim`, `set_zlim`.
- **B is incorrect:** `ax.set_zlim` accepts the two bounds as separate positional arguments `(zmin, zmax)`, not as a single list. Passing a list `[0.0, 2.5]` may work in some Matplotlib versions (because the list is unpacked internally) but is not the documented signature. The reliable form passes both bounds as separate arguments.
- **C is correct:** `ax.set_zlim(zmin, zmax)` is the documented `Axes3D` method for setting the z-axis display range. `ax.set_zlim(0.0, 2.5)` restricts the visible z range to [0.0, 2.5], trimming or padding the plot accordingly.
- **D is incorrect:** `ax.zaxis` has no `set_lim` method. Axis limits are controlled through the parent `Axes3D` object methods, not via the individual axis objects.

</details>

---

### Question 6

A team is embedding three 3D subplots in a single figure arranged in a 1×3 grid. They write:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={"projection": "3d"})
```

How many `Axes3D` objects does `axes` contain, and what is its shape?

- [ ] A) 1 `Axes3D` object; `axes` is a scalar, not an array
- [ ] B) 3 `Axes3D` objects; `axes` is a 1-D array of shape `(3,)`
- [ ] C) 3 `Axes3D` objects; `axes` is a 2-D array of shape `(1, 3)`
- [ ] D) 6 `Axes3D` objects; `plt.subplots` doubles the count when `projection="3d"` is used

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 3 `Axes3D` objects; `axes` is a 1-D array of shape `(3,)`

**Explanation:**
- **A is incorrect:** `axes` is scalar only when both `nrows=1` AND `ncols=1` AND `squeeze=True` (the default). With `ncols=3`, `plt.subplots` returns an array.
- **B is correct:** `plt.subplots(1, 3, ...)` creates a 1-row, 3-column grid of 3 subplots. By default `squeeze=True`, so a grid that is 1 row × 3 columns is squeezed to a 1-D NumPy array of shape `(3,)`. Each element is an `Axes3DSubplot` object (the 3D projection). Accessing the three axes uses `axes[0]`, `axes[1]`, `axes[2]`.
- **C is incorrect:** The 2-D shape `(1, 3)` is returned only when `squeeze=False` is set explicitly. With the default `squeeze=True`, a 1-row grid is reduced to a 1-D array.
- **D is incorrect:** `plt.subplots` does not double the number of axes based on projection type. It creates exactly `nrows × ncols` axes objects, which here is 1 × 3 = 3.

</details>

---

### Question 7

Which of the following imports is sufficient on its own to make `projection="3d"` work in Matplotlib ≥ 3.4?

- [ ] A) `from mpl_toolkits.mplot3d import Axes3D`
- [ ] B) `import matplotlib.pyplot as plt` (no additional import needed)
- [ ] C) `import mpl_toolkits` (package-level import registers the projection)
- [ ] D) `from mpl_toolkits import mplot3d`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `import matplotlib.pyplot as plt` (no additional import needed)

**Explanation:**
- **A is incorrect:** In older Matplotlib versions (< 3.4), explicitly importing `from mpl_toolkits.mplot3d import Axes3D` was required to register the `"3d"` projection with Matplotlib's projection registry. Since Matplotlib 3.4, the `mpl_toolkits` package is auto-registered as part of the standard import, making this explicit import unnecessary (though harmless).
- **B is correct:** From Matplotlib 3.4 onward, importing `matplotlib.pyplot` automatically triggers the registration of the `mpl_toolkits.mplot3d` projection backend. Using `projection="3d"` in `add_subplot` or `subplots` works without any explicit `mpl_toolkits` import.
- **C is incorrect:** `import mpl_toolkits` alone is a package-level import that does not necessarily import its `mplot3d` sub-package or register the 3D projection. The subpackage must be imported (explicitly or via the auto-registration path in Matplotlib ≥ 3.4).
- **D is incorrect:** `from mpl_toolkits import mplot3d` does import the subpackage and does register the projection in older Matplotlib versions, but is not the minimal requirement in Matplotlib ≥ 3.4, where `import matplotlib.pyplot` is sufficient.

</details>

---

### Question 8

A developer sets axis limits on a freshly created 3D axes and then plots data:

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 10)
ax.scatter([6, -6], [0, 0], [5, 5])  # x-values outside xlim
```

What is the visual effect of plotting points at `x = 6` and `x = -6` when `xlim` is set to `(-5, 5)`?

- [ ] A) The scatter call raises a `ValueError` because the data is outside the specified limits
- [ ] B) The limits are automatically expanded to fit the out-of-range points, overriding the `set_xlim` call
- [ ] C) The points are clipped to the x boundary at ±5 and rendered at the edges of the visible volume
- [ ] D) The points are plotted but fall outside the visible axis box, so they appear cut off or invisible depending on the renderer

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) The points are plotted but fall outside the visible axis box, so they appear cut off or invisible depending on the renderer

**Explanation:**
- **A is incorrect:** Matplotlib does not raise errors for out-of-range data. The data is always plotted; axis limits control what portion of the coordinate space is visible.
- **B is incorrect:** Calling `set_xlim` after plotting data in 2D Matplotlib does sometimes get overridden by autoscaling, but when `set_xlim` is called before scatter and autoscaling is not explicitly re-triggered, the explicitly set limits persist. In 3D, explicitly set limits generally hold.
- **C is incorrect:** Matplotlib does not clip 3D points to axis boundaries in the way that 2D clipping works. Points outside the limits are not clipped to the edge — they are simply outside the rendered volume.
- **D is correct:** When `ax.set_xlim(-5, 5)` is set before scatter, the axis view window is [-5, 5]. Points at x=6 and x=-6 lie outside this window. In Matplotlib 3D, these points are rendered geometrically but fall outside the visible bounding box, causing them to appear cut off or entirely invisible depending on the 3D backend renderer and the camera angle.

</details>

---

### Question 9

A data scientist wants to create a 2-row, 2-column grid where the top-left cell is 3D and the remaining three cells are standard 2D axes. Which approach correctly achieves this mixed 2D/3D layout?

- [ ] A) Pass `projection="3d"` to `plt.subplots(2, 2)` — Matplotlib applies 3D only to the first subplot automatically
- [ ] B) Create the figure with `plt.figure()` and then add each subplot individually using `fig.add_subplot(2, 2, 1, projection="3d")` for the 3D cell and `fig.add_subplot(2, 2, n)` for the 2D cells
- [ ] C) Use `plt.subplots(2, 2, subplot_kw={"projection": "3d"})` and convert three of the four axes back to 2D with `ax.remove()`
- [ ] D) Create the figure with `plt.figure()` and use `fig.add_axes([left, bottom, width, height], projection="3d")` for all four subplots

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Create the figure with `plt.figure()` and then add each subplot individually using `fig.add_subplot(2, 2, 1, projection="3d")` for the 3D cell and `fig.add_subplot(2, 2, n)` for the 2D cells

**Explanation:**
- **A is incorrect:** `subplot_kw` in `plt.subplots` applies the same keyword arguments to every subplot in the grid. There is no mechanism to apply `projection="3d"` selectively to only the first cell through the factory method.
- **B is correct:** When individual control over each subplot's projection is needed, the `fig.add_subplot(row, col, index, **kwargs)` API must be used for each cell separately. The 3D cell gets `projection="3d"` and the 2D cells are created without any projection keyword, defaulting to 2D `Axes`.
- **C is incorrect:** `ax.remove()` removes the axes object from the figure entirely; it does not convert it to a different projection type. Once created as 3D, an axes cannot be "downgraded" to 2D.
- **D is incorrect:** `fig.add_axes` creates axes with absolute figure-relative coordinates, not in a grid layout. Additionally, it does not guarantee the structured 2×2 grid positioning, and the `projection="3d"` argument would apply it to all four axes anyway.

</details>

---

### Question 10

A researcher prints the axis tick labels on the z-axis of a 3D plot and observes that the numerical labels are overlapping. Which approach correctly reduces z-axis tick density?

- [ ] A) `ax.set_zticks([0, 1, 2, 3])`  — sets explicit tick positions on the z-axis
- [ ] B) `ax.z_major_locator(MaxNLocator(4))` — directly sets the locator on the z-axis
- [ ] C) `ax.zaxis.set_tick_params(labelsize=6)` — reduces the font size so labels fit without changing tick count
- [ ] D) `ax.set_zlim(auto=True)` — enables automatic tick placement that prevents overlap

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.set_zticks([0, 1, 2, 3])`  — sets explicit tick positions on the z-axis

**Explanation:**
- **A is correct:** `ax.set_zticks(ticks)` is the `Axes3D` method that explicitly specifies which z-values receive tick marks. By passing a short list like `[0, 1, 2, 3]`, the analyst controls exactly how many ticks appear, preventing overlap. This is the direct analogue of `ax.set_xticks` and `ax.set_yticks`.
- **B is incorrect:** `ax.z_major_locator(...)` is not a valid `Axes3D` method. Setting a locator on the z-axis is done via `ax.zaxis.set_major_locator(MaxNLocator(4))`. The method name in option B does not exist.
- **C is incorrect:** Reducing `labelsize` makes individual tick labels smaller but does not reduce the number of ticks. If many ticks are close together, smaller font will help readability marginally but will not resolve severe overlap from excessive tick density.
- **D is incorrect:** `ax.set_zlim(auto=True)` is not valid syntax for `set_zlim`. Even if autoscaling is enabled, it controls the visible range, not the tick spacing. Auto-range selection does not specifically address label overlap.

</details>
