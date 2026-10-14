# Matplotlib Time Series & Temporal Visualization — Set 06

---

### Question 51

A financial quant creates a dual-axis chart comparing daily closing stock prices with trading volumes:

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
```

What core architectural relationship is established between `ax1` and `ax2` by calling `ax1.twinx()`?

- [ ] A) `ax1` and `ax2` share the exact same y-axis transform but possess independent horizontal x-axes.
- [ ] B) `ax2` is rendered as an overlay image mask that cannot accept new data plots.
- [ ] C) `ax1` and `ax2` share the identical horizontal x-axis coordinate system and date scale, while maintaining completely independent vertical y-axes on opposite sides of the canvas (left and right).
- [ ] D) `ax2` is a duplicate figure placed on a secondary computer monitor.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax1` and `ax2` share the identical horizontal x-axis coordinate system and date scale, while maintaining completely independent vertical y-axes on opposite sides of the canvas (left and right).

**Explanation:**

- **A is incorrect:** `twinx()` shares the x-axis, not the y-axis; sharing the y-axis is accomplished via `twiny()`.
- **B is incorrect:** `ax2` is a fully functional `Axes` object capable of rendering lines, bars, and collections.
- **C is correct:** `ax1.twinx()` instantiates a new `AxesSubplot` that physically overlays `ax1`. It binds its x-axis transform to `ax1.xaxis`, guaranteeing that both axes share the same horizontal timeline ruler and zoom limits. Concurrently, it creates an independent `YAxis` on the right side of the canvas (`ax2.yaxis`), allowing two series with completely different units (e.g. stock price in dollars vs volume in shares) to be plotted across time without scale distortion.
- **D is incorrect:** `twinx()` creates a subplot within the same figure canvas, not a secondary display window.

</details>

---

### Question 52

In a dual-axis stock analysis figure (`ax1` for Price in USD, `ax2` for Volume in shares), how should the visual design be configured so trading volume bars do not obscure the primary price line chart?

- [ ] A) Plot volume on `ax2` using semi-transparent bar plots (`alpha=0.3`), and set `ax2.set_ylim(0, max_vol * 4)` so the volume bars occupy only the bottom 25% of the chart, leaving the price curve uncluttered above.
- [ ] B) Volume must always be plotted with solid black fill at 100% opacity.
- [ ] C) Set `ax1.set_visible(False)` so only volume is visible during market hours.
- [ ] D) Plot volume on the horizontal axis and dates on the vertical axis.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Plot volume on `ax2` using semi-transparent bar plots (`alpha=0.3`), and set `ax2.set_ylim(0, max_vol * 4)` so the volume bars occupy only the bottom 25% of the chart, leaving the price curve uncluttered above.

**Explanation:**

- **A is correct:** In professional financial plotting (e.g., Bloomberg or TradingView), volume is visually subordinated so it does not collide with price trajectory. Setting `alpha=0.3` reduces visual weight, and setting `ax2.set_ylim(0, max_vol * 4)` artificially expands the volume axis ceiling so volume bars peak at only 25% of total plot height, keeping the price curve clearly separated in the upper 75% of the canvas.
- **B is incorrect:** Solid 100% opaque bars completely occlude price data lines drawn underneath.
- **C is incorrect:** Hiding `ax1` removes the price curve entirely.
- **D is incorrect:** Financial convention strictly places time on the horizontal axis.

</details>

---

### Question 53

When creating a dual-axis plot with `ax.twinx()`, an engineer calls `ax.legend()` and `ax2.legend()`. When the chart renders, two separate legend boxes overlap awkwardly in the top-right corner.

Which modern Matplotlib pattern consolidates handles and labels from both twin axes into a **single, unified legend box**?

- [ ] A) `plt.legend(ax, ax2)`
- [ ] B) `ax.merge_legends(ax2)`
- [ ] C) `fig.unify_legends([ax, ax2])`
- [ ] D)
  ```python
  lines1, labels1 = ax.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D)
```python
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
```

**Explanation:**

- **A is incorrect:** `plt.legend()` does not take two `Axes` objects as positional arguments.
- **B is incorrect:** `merge_legends()` does not exist in the Matplotlib API.
- **C is incorrect:** `unify_legends()` is a fictitious Figure method.
- **D is correct:** In Matplotlib, each `Axes` independently maintains its own collection of labeled artists. Calling `.legend()` on both creates two competing legend boxes. The established canonical solution is to call `get_legend_handles_labels()` on both `ax` and `ax2`, concatenate the resulting lists (`lines1 + lines2`, `labels1 + labels2`), and pass the merged lists to a single `ax.legend()` call.

</details>

---

### Question 54

In a dual-axis plot, enabling grid lines on both `ax` and `ax2` via `ax.grid(True)` and `ax2.grid(True)` produces an unreadable visual lattice where horizontal grid lines from the left and right axes clash at unaligned heights.

What is the recommended design practice to eliminate this grid collision?

- [ ] A) Enable 3D perspective to separate the grid lines in depth.
- [ ] B) Turn off grid lines on the secondary axis (`ax2.grid(False)`) so that only the primary axis governs the horizontal reference grid lines.
- [ ] C) Set both grid lines to fluorescent green so they cancel each other out.
- [ ] D) Delete the secondary y-axis entirely and plot volume on a pie chart.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Turn off grid lines on the secondary axis (`ax2.grid(False)`) so that only the primary axis governs the horizontal reference grid lines.

**Explanation:**

- **A is incorrect:** 3D perspective is inappropriate for dual-axis 2D time series.
- **B is correct:** Because `ax` and `ax2` have different numerical ranges and separate tick locators, their major tick heights rarely match. Drawing horizontal gridlines for both creates visual chaos ("chartjunk"). Best practice is to set `ax2.grid(False)` so horizontal gridlines bind strictly to the primary metric, while vertical date gridlines remain shared and clean.
- **C is incorrect:** Bright colors exacerbate visual interference.
- **D is incorrect:** Deleting the axis eliminates the secondary time-series metric.

</details>

---

### Question 55

If a developer executes `ax.set_xlim(pd.to_datetime("2026-01-01"), pd.to_datetime("2026-03-31"))` on the primary axis `ax`, what happens to the view limits of the twin axis `ax2 = ax.twinx()`?

- [ ] A) `ax2` retains its original limits and must be updated manually.
- [ ] B) `ax2` resets its view limits to year 1970.
- [ ] C) `ax2` automatically synchronizes to the identical January 1 to March 31 date window because its horizontal x-axis transform is shared with `ax`.
- [ ] D) Matplotlib crashes due to an unhandled synchronization deadlock.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax2` automatically synchronizes to the identical January 1 to March 31 date window because its horizontal x-axis transform is shared with `ax`.

**Explanation:**

- **A is incorrect:** Manual updating of `ax2` is not necessary because the horizontal transform is shared.
- **B is incorrect:** It does not reset to 1970.
- **C is correct:** `twinx()` links the two `Axes` through a shared `XAxis` transform (`ax2._shared_axes['x']`). Modifying view limits on either axis via `set_xlim()`, panning, or zooming automatically propagates to the other axis, keeping both series synchronized along the timeline.
- **D is incorrect:** Shared axes are a core, fully tested feature of Matplotlib that do not deadlock.

</details>

---

### Question 56

A data visualization designer wants the left y-axis tick numbers and spine to be rendered in blue (matching the primary price line) and the right y-axis tick numbers and spine in crimson (matching the secondary inflation line).

Which pair of commands configures this color-coded axis clarity?

- [ ] A)
  ```python
  ax.tick_params(axis="y", labelcolor="blue", colors="blue")
  ax2.tick_params(axis="y", labelcolor="crimson", colors="crimson")
```
- [ ] B)
  ```python
  ax.set_color("blue")
  ax2.set_color("crimson")
```
- [ ] C)
  ```python
  ax.yaxis.tint("blue")
  ax2.yaxis.tint("crimson")
```
- [ ] D)
  ```python
  ax.set_palette(["blue", "crimson"])
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
ax.tick_params(axis="y", labelcolor="blue", colors="blue")
ax2.tick_params(axis="y", labelcolor="crimson", colors="crimson")
```

**Explanation:**

- **A is correct:** In `ax.tick_params()`, setting `axis="y"` and `colors="blue"` changes both the tick marks and labels of the primary vertical axis to blue. Setting `colors="crimson"` on `ax2` styles the secondary vertical axis. This color coordination immediately informs viewers which data curve corresponds to which vertical ruler, eliminating reading ambiguity.
- **B is incorrect:** `ax.set_color()` does not exist on `Axes`.
- **C is incorrect:** `yaxis.tint()` is a fabricated method name.
- **D is incorrect:** `set_palette()` sets default line cycle colors, not axis spine/tick colors.

</details>

---

### Question 57

When plotting net profit margin (which ranges from $-20\%$ to $+30\%$) on `ax` and revenue (which ranges from $\$0$ to $\$50	ext{M}$) on `ax2`, the zero baseline on `ax` does not align with the bottom of `ax2`, creating visual dissonance.

What technique mathematically aligns the zero baseline of `ax` with a desired coordinate on `ax2`?

- [ ] A) Call `ax.align_zeros(ax2)`.
- [ ] B) Enable `plt.zero_lock(True)`.
- [ ] C) Set `ax.set_yscale('zero')`.
- [ ] D) Compute a shared proportional scale factor or manually adjust `set_ylim()` on both axes so that the zero coordinate corresponds to the identical relative vertical fraction (e.g. $y=0$ at 40% height on both axes).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Compute a shared proportional scale factor or manually adjust `set_ylim()` on both axes so that the zero coordinate corresponds to the identical relative vertical fraction (e.g. $y=0$ at 40% height on both axes).

**Explanation:**

- **A is incorrect:** While Seaborn or third-party libraries sometimes offer alignment helpers, core Matplotlib has no built-in `ax.align_zeros()` method on basic `Axes`.
- **B is incorrect:** `zero_lock` does not exist in `pyplot`.
- **C is incorrect:** `'zero'` is not a recognized Matplotlib scale.
- **D is correct:** Because each axis calculates limits independently, a zero value on `ax` can sit at mid-height while zero on `ax2` sits at the very bottom. To align them, developers compute the ratio of positive-to-negative spans and adjust `ylim` bounds symmetrically so that zero maps to the exact same normalized vertical position on both axes.

</details>

---

### Question 58

An analyst wants to label the horizontal date axis on a twin-axis plot:

```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
```

Where should the horizontal axis label (`"Date"`) be applied?

- [ ] A) `ax2.set_xlabel("Date")`
- [ ] B) `ax1.set_xlabel("Date")`
- [ ] C) Both `ax1.set_xlabel("Date")` and `ax2.set_xlabel("Date")` must be called simultaneously.
- [ ] D) `fig.set_date_label("Date")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax1.set_xlabel("Date")`

**Explanation:**

- **A is incorrect:** In `twinx()`, the bottom spine, tick labels, and axis title remain bound to the primary `ax1` instance; setting it on `ax2` can cause text clipping or fail to render depending on backend.
- **B is correct:** `ax1` owns the primary horizontal axis structure (including the bottom spine and label). Calling `ax1.set_xlabel("Date")` places the label beneath the bottom tick marks cleanly.
- **C is incorrect:** Calling it on both axes can render duplicate overlapping text strings.
- **D is incorrect:** `set_date_label` is not a valid method on `Figure`.

</details>

---

### Question 59

A batch processing job generates 500 dual-axis charts inside a loop using `fig, ax = plt.subplots()` and `ax.twinx()`. After 150 iterations, the worker runs out of RAM and crashes.

Why did garbage collection fail to reclaim memory, and what single command resolves this?

- [ ] A) `twinx()` holds a GPU lock; calling `torch.cuda.empty_cache()` is required.
- [ ] B) The operating system page file was full; increasing virtual memory is required.
- [ ] C) Matplotlib retains global references to all instantiated figures in `matplotlib._pylab_helpers.Gcf.figs`; calling `plt.close(fig)` at the end of each iteration explicitly destroys the figure canvas and frees memory.
- [ ] D) Dual axes cannot be created inside loops in Python 3.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Matplotlib retains global references to all instantiated figures in `matplotlib._pylab_helpers.Gcf.figs`; calling `plt.close(fig)` at the end of each iteration explicitly destroys the figure canvas and frees memory.

**Explanation:**

- **A is incorrect:** Matplotlib does not hold PyTorch GPU locks.
- **B is incorrect:** The issue is an application-level memory leak, not OS page file sizing.
- **C is correct:** In Matplotlib's `pyplot` state machine, every figure created via `plt.subplots()` is cached in a global manager dictionary (`Gcf`). Because `twinx()` creates multiple axes artists per figure, memory accumulates rapidly. Simply letting `fig` go out of scope does not trigger garbage collection because the global reference persists. Calling `plt.close(fig)` explicitly deregisters the figure and reclaims memory.
- **D is incorrect:** Dual axes are completely valid inside loops when closed properly.

</details>

---

### Question 60

Which of the following statements regarding `ax.twinx()` in time-series visualization are **correct**?

1. `ax.twinx()` creates an overlay axis sharing the horizontal date coordinate system.
2. The primary and secondary axes have completely independent vertical y-axes and view limits.
3. Consolidating legend handles from both axes requires passing combined lists to `ax.legend()`.
4. Enabling `ax.twinx()` automatically doubles the frame rate of video animations.

- [ ] A) 1, 2, and 3 only
- [ ] B) 1 and 4 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 1, 2, and 3 only

**Explanation:**

- **A is correct:** Statements 1, 2, and 3 are all accurate features of `twinx()`. It shares the horizontal timeline, maintains independent y-limits, and requires manual handle merging for a unified legend box.
- **B is incorrect:** Includes Statement 4, which is completely false (twin axes have no relationship to video frame rate).
- **C is incorrect:** Includes false Statement 4 and omits Statement 1.
- **D is incorrect:** Includes false Statement 4 and omits Statements 1 and 2.

</details>

---
