# Matplotlib Figure & Axes Objective Test — Set 10
# Topic: Comprehensive Production Scenarios & Debugging (Questions 91–100)

---

### Question 91

A developer creates the following production ML reporting pipeline and encounters a bug:

```python
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(14, 10), constrained_layout=True)
fig.suptitle("Model Evaluation Report", fontsize=16, fontweight="bold")

# Top row: two wide panels
ax_loss = fig.add_axes([0.05, 0.55, 0.42, 0.38])
ax_roc  = fig.add_axes([0.53, 0.55, 0.42, 0.38])

# Bottom row: three narrow panels
ax_cm    = fig.add_axes([0.05, 0.08, 0.27, 0.38])
ax_pr    = fig.add_axes([0.37, 0.08, 0.27, 0.38])
ax_feat  = fig.add_axes([0.69, 0.08, 0.27, 0.38])
```

The developer notices `constrained_layout=True` has no effect on the manual `add_axes` layout. What explains this and what is the correct fix?

- [ ] A) `constrained_layout=True` conflicts with `suptitle` — remove `suptitle` to activate it
- [ ] B) `constrained_layout` only manages axes created via `add_subplot`/`subplots` — axes created with `add_axes` are not part of the constraint system; the fix is to use `GridSpec` or `subplot_mosaic` for complex grid layouts, or manage spacing manually
- [ ] C) Set `constrained_layout=False` and add `plt.tight_layout()` at the end
- [ ] D) Replace `fig.add_axes()` with `fig.add_subplot()` using fractional positions

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `constrained_layout` only manages axes created via `add_subplot`/`subplots` — `add_axes` axes are not part of the constraint system

**Explanation:**

- **A is incorrect:** `constrained_layout` and `suptitle` are compatible — in fact, `constrained_layout` is designed to account for `suptitle` in its geometry calculations when the axes are part of the grid system.
- **B is correct:** `constrained_layout` operates on Matplotlib's **subplot grid system** — axes with assigned `SubplotSpec` positions (created via `add_subplot`, `plt.subplots`, `GridSpec`, or `subplot_mosaic`). Axes created with `fig.add_axes([l, b, w, h])` have manually specified normalized positions and are NOT part of any grid system. The `constrained_layout` engine ignores them and cannot reposition them. For complex non-grid layouts, the alternatives are: (1) `matplotlib.gridspec.GridSpec` for programmatic grid layouts, (2) `fig.subplot_mosaic()` for named panel layouts, or (3) fully manual `add_axes` spacing without any layout engine.
- **C is incorrect:** `tight_layout()` also has limited effectiveness with `add_axes`-created axes — the same limitation applies. Switching from `constrained_layout` to `tight_layout` does not solve the core incompatibility.
- **D is incorrect:** `add_subplot()` does not accept fractional position arguments. It uses grid indices (row, column, index) for positioning within a subplot grid.

</details>

---

### Question 92

A developer creates a complete 4-panel publication figure using `GridSpec`:

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 9))
fig.suptitle("Neural Network Training Analysis", fontsize=14, y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       left=0.08, right=0.95,
                       top=0.92, bottom=0.08,
                       hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])   # Top-left
ax2 = fig.add_subplot(gs[0, 1])   # Top-right
ax3 = fig.add_subplot(gs[1, 0])   # Bottom-left
ax4 = fig.add_subplot(gs[1, 1])   # Bottom-right
```

What does `gs[0, 0]` return in this context?

- [ ] A) The `Axes` object at position (0,0)
- [ ] B) A `SubplotSpec` object defining the grid cell at row 0, column 0 — which is then passed to `fig.add_subplot()` to create the `Axes` at that position
- [ ] C) The normalized figure coordinates `[left, bottom, width, height]` of that cell
- [ ] D) An integer index (0) identifying the first subplot

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) A `SubplotSpec` object defining the grid cell at row 0, column 0

**Explanation:**

- **A is incorrect:** `gs[0, 0]` does NOT return an `Axes` object. It returns a `SubplotSpec` — a descriptor of a grid position. The actual `Axes` is created by `fig.add_subplot(gs[0, 0])`.
- **B is correct:** `gridspec.GridSpec` indexing with `gs[row, col]` returns a **`SubplotSpec`** object — a lightweight descriptor that defines which cell(s) of the grid this subplot occupies. Passing this `SubplotSpec` to `fig.add_subplot()` creates an `AxesSubplot` positioned at that grid location. `GridSpec` supports slicing for spanning multiple cells: `gs[0, :]` creates an axes spanning the full top row; `gs[:, 1]` spans the full right column.
- **C is incorrect:** `gs[0, 0]` does not return normalized coordinates. The `SubplotSpec` contains grid position information but not direct `[l, b, w, h]` coordinates (though these can be derived from it via `gs[0,0].get_position(fig)`).
- **D is incorrect:** `gs[0, 0]` is not a simple integer index. It is a `SubplotSpec` object with row/column information from the 2D grid.

</details>

---

### Question 93

A developer creates the following figure for a model monitoring dashboard and must debug a rendering issue:

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

fig.set_facecolor("#F8F9FA")
for ax in axes.flatten():
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(pad=1.2)
plt.savefig("dashboard.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
```

Why is `facecolor=fig.get_facecolor()` passed to `savefig()`?

- [ ] A) To apply the facecolor to all axes in the saved file
- [ ] B) To prevent `savefig()` from defaulting to white (or "auto") and overriding the custom figure background `#F8F9FA` set by `fig.set_facecolor()`
- [ ] C) Because `bbox_inches="tight"` resets the facecolor to transparent
- [ ] D) It is redundant — `savefig()` always uses the figure's current facecolor

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) To prevent `savefig()` from defaulting to white and overriding the custom `#F8F9FA` background

**Explanation:**

- **A is incorrect:** `facecolor` in `savefig()` controls only the **figure** background patch — not individual axes backgrounds. Axes backgrounds (`ax.set_facecolor("#FFFFFF")`) are part of the figure content and are always rendered regardless of the `facecolor` parameter.
- **B is correct:** In some Matplotlib versions, `savefig()` has a `facecolor` parameter that defaults to `"white"` (or in newer versions, `"auto"` which reads from the figure). To be safe and explicit across Matplotlib versions, passing `facecolor=fig.get_facecolor()` ensures the saved file uses exactly the same background color as the figure display — `#F8F9FA`. Without this, the custom background might be overridden by the default white in the saved file.
- **C is incorrect:** `bbox_inches="tight"` does not affect the facecolor. It only changes the crop boundary of the saved image.
- **D is incorrect:** While newer Matplotlib versions (3.3+) use `facecolor="auto"` which reads from the figure, older versions default to white. Using `facecolor=fig.get_facecolor()` is defensive programming that works correctly across all Matplotlib versions.

</details>

---

### Question 94

A developer creates a dual-panel figure with shared x-axis for a training timeline:

```python
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                      sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
ax_top.plot(epochs, loss, label="Loss")
ax_bot.plot(epochs, lr, label="LR", color="orange")
ax_top.legend(); ax_bot.legend()
plt.tight_layout()
```

What does `height_ratios=[3, 1]` in `gridspec_kw` accomplish?

- [ ] A) Sets the top axes 3× the DPI of the bottom axes
- [ ] B) Makes the top axes 3 times taller than the bottom axes within the shared figure height
- [ ] C) Limits the top axes to 3 data units in height
- [ ] D) Creates 3 rows in the top panel and 1 row in the bottom panel

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Makes the top axes 3 times taller than the bottom axes within the shared figure height

**Explanation:**

- **A is incorrect:** `height_ratios` controls subplot height proportions, not DPI or resolution settings.
- **B is correct:** `height_ratios=[3, 1]` passes to the underlying `GridSpec` and divides the available figure height (after margins) between the two subplot rows in a 3:1 ratio. The top axes (`ax_top`) receives 3/(3+1) = 75% of the available height; the bottom axes (`ax_bot`) receives 1/(3+1) = 25%. This is perfect for the "main plot + indicator" pattern common in financial charts and ML training dashboards — a large main panel for the loss curve and a slim bottom panel for the learning rate schedule.
- **C is incorrect:** `height_ratios` controls the pixel/inch height of the axes in the figure, not the data coordinate range. Data limits are controlled by `ax.set_ylim()`.
- **D is incorrect:** `height_ratios=[3, 1]` defines the proportional HEIGHT of each ROW — not the number of rows within each panel. The figure still has exactly 2 rows (2 subplot panels).

</details>

---

### Question 95

A developer writes the following code to create a complete ML evaluation figure and encounters a `UserWarning` about tight_layout:

```python
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax in axes.flatten():
    ax.plot(np.random.randn(50))
plt.tight_layout(pad=0.1)  # UserWarning: tight_layout : falling back...
```

The warning says "tight_layout : falling back to Agg renderer." What causes this warning?

- [ ] A) The Agg backend is not installed correctly
- [ ] B) `tight_layout` requires a renderer to measure text extents; when running in a non-interactive environment without a display (e.g., headless server), it falls back to the Agg renderer to obtain these measurements — this is informational and usually harmless
- [ ] C) `pad=0.1` is too small and causes a renderer crash
- [ ] D) `tight_layout` cannot work with 2×2 subplot grids

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `tight_layout` requires a renderer to measure text extents; when running in a non-interactive environment, it falls back to Agg

**Explanation:**

- **A is incorrect:** The Agg backend is Matplotlib's built-in, always-available raster renderer. The warning is about USING Agg as a fallback, not about it being missing.
- **B is correct:** `tight_layout` needs to measure the actual rendered widths and heights of text elements (tick labels, axis labels, titles) to compute required margins. This measurement requires a **renderer**. In interactive environments (with a display), the active GUI renderer is used. In headless environments (servers, CI pipelines, scripts without display), no GUI renderer is active — so `tight_layout` falls back to the **Agg renderer** to perform these measurements. The `UserWarning` is informational — `tight_layout` still works correctly using Agg measurements. The warning can be suppressed by explicitly using the Agg backend from the start (`matplotlib.use("Agg")`).
- **C is incorrect:** `pad=0.1` is valid (though small). Very small pad values might cause content overlap but don't cause renderer warnings or crashes.
- **D is incorrect:** `tight_layout` fully supports 2×2 (and arbitrary) subplot grids. Grid size has no bearing on this warning.

</details>

---

### Question 96

A developer creates a figure programmatically and needs to retrieve a specific axes by its position after the fact:

```python
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
# ... add 50 lines of plotting code ...

# Later: need to add a title to the middle-right subplot
target_ax = fig.axes[3]   # Is this correct?
target_ax.set_title("Middle Right")
```

Is `fig.axes[3]` the middle-right subplot for a 3×2 grid?

- [ ] A) Yes — `fig.axes` is indexed in row-major order: index 3 = row 1 (middle), column 1 (right)
- [ ] B) No — `fig.axes[3]` returns the 4th axes in creation order, which for `plt.subplots(3, 2)` is index 3 = row 1 (0-indexed middle row), column 1 (right column) in row-major order — so it IS the middle-right
- [ ] C) No — `fig.axes` is indexed in column-major order: index 3 = column 1, row 0 = top-right
- [ ] D) Yes, but `fig.axes` indexing is unreliable — use `axes[1, 1]` instead

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Yes, but `fig.axes` indexing is unreliable — use `axes[1, 1]` instead

**Explanation:**

- **A is partially correct but misleading:** While index 3 does correspond to the middle-right in this specific case, relying on `fig.axes[n]` is fragile.
- **B is partially correct** mathematically for this specific case: `plt.subplots(3, 2)` creates axes in row-major order. Index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1). So `fig.axes[3]` IS the middle-right (row=1, col=1). BUT this is fragile.
- **C is incorrect:** `plt.subplots` fills in **row-major** (C) order, not column-major (Fortran) order.
- **D is correct:** While `fig.axes[3]` happens to work here, `fig.axes` ordering is fragile — adding colorbars, insets, or twin axes inserts additional axes into `fig.axes` at unexpected indices, breaking the correspondence. The correct, robust approach is to use the **`axes` array** returned by `plt.subplots()`: `axes[1, 1]` explicitly retrieves row=1, col=1 (middle-right) regardless of how many other axes exist in the figure. The `axes` return value is the safe, explicit reference.

</details>

---

### Question 97

A developer creates a comprehensive figure export pipeline and must verify the output is correct:

```python
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, linewidth=2, color="#2196F3")
ax.fill_between(x, y - std, y + std, alpha=0.25, color="#2196F3")
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Signal Amplitude", fontsize=12)
ax.set_title("Mean ± 1σ Signal", fontsize=13)
ax.legend(["Mean", "±1σ Band"], loc="upper right")

fig.savefig("signal.svg", bbox_inches="tight")
fig.savefig("signal.png", dpi=300, bbox_inches="tight")
plt.close(fig)
```

After `plt.close(fig)`, the developer tries to add more data to the axes:

```python
ax.plot(x2, y2, color="red")
```

What happens?

- [ ] A) The new data is added to the closed figure and appears in the next `savefig()` call
- [ ] B) `ax.plot()` executes without error but the data is NOT rendered — the axes object still exists in Python memory but is detached from the figure manager after `plt.close(fig)`, and any subsequent saves would require `fig.savefig()` again (the `ax` Python object is still accessible)
- [ ] C) A `ValueError` is raised: "Cannot plot on a closed figure"
- [ ] D) The axes automatically reopens its figure window to display the new data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.plot()` executes without error but the figure is detached from the manager — the `ax` Python object remains accessible

**Explanation:**

- **A is incorrect:** The figure is closed (deregistered from the manager). While `ax.plot()` can technically modify the axes Python object, the figure is no longer connected to any active renderer or display. There is no "next savefig() call" that would render this new data without explicitly calling `fig.savefig()` again.
- **B is correct:** `plt.close(fig)` deregisters the figure from Matplotlib's figure manager but does NOT destroy the Python objects. The `fig` and `ax` variables still reference valid Python objects in memory (assuming no other code cleared them). `ax.plot(x2, y2)` can modify the `ax` object's internal state (adding a new `Line2D` to `ax.lines`). If `fig.savefig()` is called again, the new data WOULD appear. However, in interactive mode, the closed figure's window is gone — the update won't be displayed interactively. This is a subtle but important distinction: `plt.close()` closes the WINDOW/MANAGER registration, not the Python objects.
- **C is incorrect:** Matplotlib does NOT raise a `ValueError` when plotting on a "closed" axes. The Python objects remain valid; only the manager registration is removed.
- **D is incorrect:** `plt.close(fig)` closes the figure window. Plotting on the axes does not automatically reopen windows or re-register with the figure manager.

</details>

---

### Question 98

A developer creates a production-grade figure factory function for an ML experiment tracker:

```python
def make_experiment_figure(run_data: dict, output_path: str,
                            dark_mode: bool = False) -> None:
    bg_color  = "#1A1A2E" if dark_mode else "#FFFFFF"
    ax_color  = "#16213E" if dark_mode else "#F8F9FA"
    text_color = "white"  if dark_mode else "black"
    line_color = "#00D2FF" if dark_mode else "#1565C0"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor(bg_color)
    ax.set_facecolor(ax_color)

    ax.plot(run_data["epochs"], run_data["loss"],
            color=line_color, linewidth=2)
    ax.set_xlabel("Epoch", color=text_color, fontsize=12)
    ax.set_ylabel("Loss",  color=text_color, fontsize=12)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    fig.suptitle(run_data["name"], color=text_color, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
```

Which element of this function is responsible for ensuring the suptitle does not overlap the plot area in the saved file?

- [ ] A) `bbox_inches="tight"` in `savefig()`
- [ ] B) `plt.tight_layout(rect=[0, 0, 1, 0.94])` — reserving the top 6% of the figure for the suptitle
- [ ] C) `fig.suptitle(..., color=text_color)` — the color prevents overlap
- [ ] D) `facecolor=fig.get_facecolor()` in `savefig()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `plt.tight_layout(rect=[0, 0, 1, 0.94])` — reserving the top 6% for the suptitle

**Explanation:**

- **A is incorrect:** `bbox_inches="tight"` ensures the saved file includes all artists (preventing clipping), but it does NOT prevent the suptitle from overlapping the plot area during layout — it only ensures the suptitle is included in the saved boundary.
- **B is correct:** `plt.tight_layout(rect=[0, 0, 1, 0.94])` constrains all subplots to the rectangle from (0,0) to (1,0.94) in normalized figure coordinates. This reserves the top 6% (y from 0.94 to 1.0) of the figure height exclusively for the `suptitle`. The tight layout engine then fits all axes content within the 0–0.94 vertical range, preventing the plot area from encroaching on the suptitle space. This is the standard pattern for preventing suptitle overlap.
- **C is incorrect:** Text color (`color=text_color`) controls visual appearance but has absolutely no effect on positioning or overlap prevention.
- **D is incorrect:** `facecolor=fig.get_facecolor()` ensures background color consistency in the saved file — it has no role in layout geometry or overlap prevention.

</details>

---

### Question 99

A developer creates a multi-figure ML report pipeline and must ensure figures are correctly enumerated. They use:

```python
fig1 = plt.figure(1, figsize=(8, 5))
fig2 = plt.figure(2, figsize=(8, 5))
fig3 = plt.figure(1)  # What does this do?
```

What does `plt.figure(1)` do when a figure with number 1 already exists?

- [ ] A) Creates a new figure with a new auto-assigned number (e.g., 4)
- [ ] B) Returns the existing figure with number 1 and sets it as the current figure — no new figure is created
- [ ] C) Raises a `ValueError` because figure number 1 is already in use
- [ ] D) Replaces figure 1 with a new blank figure, deleting all previous content

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Returns the existing figure with number 1 and sets it as the current figure — no new figure is created

**Explanation:**

- **A is incorrect:** Passing an explicit integer to `plt.figure()` requests that specific figure number. Matplotlib does not create a new auto-numbered figure when an explicit number is given.
- **B is correct:** `plt.figure(num)` with an integer argument: if a figure with that number already exists in the figure manager, Matplotlib **returns the existing figure** and makes it the current active figure. No new figure is created. `fig3` would be the same Python object as `fig1` (`fig3 is fig1` evaluates to `True`). This is by design — it allows reactivating a specific figure window by its number in an interactive session. If the number does NOT exist, a new figure is created with that number.
- **C is incorrect:** No error is raised for reusing an existing figure number. This is explicitly supported behavior.
- **D is incorrect:** Returning an existing figure does NOT clear or reset its content. All previously plotted artists remain intact. To clear a figure, explicit `fig.clf()` must be called.

</details>

---

### Question 100

A developer reviews the following complete production figure generation script and must identify ALL correct statements about the code:

```python
import matplotlib
matplotlib.use("Agg")          # Line A
import matplotlib.pyplot as plt
import numpy as np

def generate_training_report(history: dict, save_path: str) -> None:
    fig = plt.figure(figsize=(12, 8), dpi=150,    # Line B
                     constrained_layout=True)      # Line C
    fig.suptitle("Training Report", fontsize=15,   # Line D
                 fontweight="bold")

    gs = fig.add_gridspec(2, 2)                    # Line E
    ax1 = fig.add_subplot(gs[0, :])               # Line F — top full row
    ax2 = fig.add_subplot(gs[1, 0])               # Line G
    ax3 = fig.add_subplot(gs[1, 1])               # Line H

    ax1.plot(history["loss"],     label="Train")
    ax1.plot(history["val_loss"], label="Val", linestyle="--")
    ax1.legend(); ax1.set_title("Loss")

    ax2.bar(range(len(history["metrics"])), history["metrics"])
    ax2.set_title("Epoch Metrics")

    ax3.imshow(history["cm"], cmap="Blues")
    ax3.set_title("Confusion Matrix")

    fig.savefig(save_path, bbox_inches="tight",    # Line I
                facecolor=fig.get_facecolor())
    plt.close(fig)                                 # Line J

generate_training_report(history, "report.png")
```

Which statements are TRUE?

- [ ] A) Line A sets a non-interactive backend before pyplot is imported — correct for server/headless execution
- [ ] B) Line F (`gs[0, :]`) creates an axes spanning BOTH columns of row 0 — a full-width top panel
- [ ] C) Line C (`constrained_layout=True`) is compatible with `add_gridspec` axes — unlike `add_axes`, GridSpec axes participate in the constraint solver
- [ ] D) Line J is unnecessary — figures created with `constrained_layout=True` are automatically closed after `savefig()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A, B, and C are TRUE. D is FALSE.

**Explanation:**

- **A is TRUE:** `matplotlib.use("Agg")` MUST be called before `import matplotlib.pyplot as plt`. This sets the non-interactive Agg backend which does not require a display — essential for headless servers, Docker containers, CI/CD pipelines, and batch scripts. Calling `matplotlib.use()` after pyplot is imported raises a warning and may have no effect.
- **B is TRUE:** `gs[0, :]` uses Python slice notation on the `GridSpec` — `:` means "all columns." For a 2×2 grid, `gs[0, :]` selects row 0, all columns (0 and 1), creating a `SubplotSpec` that spans the full width of the figure. `fig.add_subplot(gs[0, :])` creates a single wide axes occupying the entire top row — perfect for a full-width loss curve.
- **C is TRUE:** `constrained_layout=True` fully integrates with `GridSpec`. Axes created via `fig.add_gridspec()` + `fig.add_subplot(gs[...])` are part of the subplot grid system and ARE managed by the constrained layout engine. This is unlike `fig.add_axes([l,b,w,h])` which bypasses the layout engine entirely.
- **D is FALSE:** `constrained_layout=True` has NO automatic figure-closing behavior. It is a layout engine setting that adjusts spacing — it has no lifecycle management. `plt.close(fig)` (Line J) is absolutely necessary to deregister the figure from the Matplotlib manager and prevent memory accumulation. Without Line J, every call to `generate_training_report()` would leave a figure object in memory.

</details>
