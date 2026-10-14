# Matplotlib Ticks & Scales Objective Test — Set 10

---

### Question 91

An ML engineer plots a Cosine Annealing learning rate schedule with linear warmup across $50,000$ training iterations. The learning rate ranges from $10^{-7}$ during initial warmup to a peak of $10^{-3}$, decaying back to $10^{-6}$.

To inspect the schedule clearly with concise exponential labels (`"1e-7"`, `"1e-5"`, `"1e-3"`), which configuration combines a logarithmic scale with clean scientific formatting?

- [ ] A) `ax.set_yscale('linear'); ax.set_format('scientific')`
- [ ] B)
  ```python
  from matplotlib.ticker import FuncFormatter

  ax.set_yscale("log")
  ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0e}"))
```
- [ ] C) `ax.set_yscale('log'); ax.yaxis.format_exponents()`
- [ ] D) `ax.yaxis.scale_scientific(powers=[-7, -3])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B)
```python
from matplotlib.ticker import FuncFormatter

ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0e}"))
```

**Explanation:**

- **A is incorrect:** `ax.set_format('scientific')` is a fictitious method that raises an `AttributeError`.
- **B is correct:** `ax.set_yscale("log")` establishes a base-10 logarithmic projection. Pairing this with `FuncFormatter(lambda y, pos: f"{y:.0e}")` formats each numeric tick coordinate $y$ as a concise scientific exponent string (e.g., `"1e-07"`, `"1e-05"`, `"1e-03"`), preventing cumbersome multi-decimal floats (`0.0000001`) from cluttering the vertical margin.
- **C is incorrect:** `ax.yaxis.format_exponents()` does not exist in Matplotlib.
- **D is incorrect:** `ax.yaxis.scale_scientific()` is not a valid Matplotlib method.

</details>

---

### Question 92

During neural network training, an engineer wants to pinpoint the exact epoch (between epochs $0$ and $100$) where validation loss diverges. They set major ticks every 10 epochs and attach minor ticks every 2 epochs, synchronized with a subtle minor grid.

Which code block achieves this diagnostic precision?

- [ ] A) `ax.set_ticks(every=10, sub=2); ax.grid(True)`
- [ ] B) `ax.xaxis.set_major_locator(MultipleLocator(2)); ax.xaxis.set_minor_locator(MultipleLocator(10))`
- [ ] C)
  ```python
  from matplotlib.ticker import AutoMinorLocator, MultipleLocator

  ax.xaxis.set_major_locator(MultipleLocator(10))
  ax.xaxis.set_minor_locator(AutoMinorLocator(5))
  ax.grid(True, which="major", linestyle="-", alpha=0.6)
  ax.grid(True, which="minor", linestyle=":", alpha=0.3)
```
- [ ] D) `ax.epoch_grid(step=10, substep=2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(True, which="major", linestyle="-", alpha=0.6)
ax.grid(True, which="minor", linestyle=":", alpha=0.3)
```

**Explanation:**

- **A is incorrect:** `ax.set_ticks(every=10, sub=2)` uses fictitious arguments that raise a `TypeError`.
- **B is incorrect:** Reverses the roles of major and minor locators; setting major locator to 2 and minor locator to 10 creates major ticks every 2 epochs and completely renders the minor locator redundant.
- **C is correct:** `MultipleLocator(10)` sets major tick intervals at $0, 10, 20, \dots, 100$. `AutoMinorLocator(5)` divides each 10-epoch interval into 5 sub-bins, creating minor tick marks at exact 2-epoch increments ($2, 4, 6, 8, \dots$). Enabling `which="minor"` grid lines with a faint dotted stroke allows the engineer to trace vertical alignments directly to the exact divergence epoch.
- **D is incorrect:** `ax.epoch_grid()` does not exist in the Matplotlib library.

</details>

---

### Question 93

When evaluating an image classifier on the 10 classes of CIFAR-10, an engineer plots a $10 \times 10$ confusion matrix. By default, axes show floating-point ticks ($0.0, 2.5, 5.0, 7.5$).

Which sequence of calls replaces these continuous floats with discrete integer ticks mapped to categorical class names, preventing diagonal text overlap?

- [ ] A) `ax.set_categories(classes, rotate=True)`
- [ ] B) `ax.set_xticklabels(classes); ax.set_yticklabels(classes)`
- [ ] C) `ax.xaxis.set_classes(classes)`
- [ ] D)
  ```python
  ticks = range(len(classes))
  ax.set_xticks(ticks)
  ax.set_xticklabels(classes, rotation=45, ha="right")
  ax.set_yticks(ticks)
  ax.set_yticklabels(classes)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D)
```python
ticks = range(len(classes))
ax.set_xticks(ticks)
ax.set_xticklabels(classes, rotation=45, ha="right")
ax.set_yticks(ticks)
ax.set_yticklabels(classes)
```

**Explanation:**

- **A is incorrect:** `ax.set_categories()` is a non-existent method in Matplotlib.
- **B is incorrect:** Invoking `set_xticklabels()` without first establishing discrete tick coordinates with `set_xticks()` raises a `UserWarning: FixedFormatter should only be used together with FixedLocator` and maps labels to arbitrary floating-point tick positions.
- **C is incorrect:** `ax.xaxis.set_classes()` is not a valid method.
- **D is correct:** Explicitly setting `ax.set_xticks(ticks)` establishes fixed integer coordinates for each class index ($0$ through $9$). Supplying `ax.set_xticklabels(classes, rotation=45, ha="right")` cleanly maps each class name with 45-degree right-aligned rotation, preventing text collision, while `set_yticks()` and `set_yticklabels()` label the vertical actual class axis.

</details>

---

### Question 94

A practitioner plots a Precision-Recall threshold sweep where classification threshold $T \in [0.0, 1.0]$ is on the x-axis, Precision is plotted against the primary y-axis, and Recall is plotted against a secondary twin y-axis (`ax2 = ax.twinx()`).

Which configuration sets both vertical axes to show clean percentages ($0\%$ to $100\%$) bounded strictly within $[0.0, 1.05]$?

- [ ] A)
  ```python
  from matplotlib.ticker import PercentFormatter

  for a in [ax, ax2]:
    a.set_ylim(0.0, 1.05)
    a.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
```
- [ ] B) `ax.set_twin_format(PercentFormatter(100))`
- [ ] C) `plt.percent_scale([ax, ax2])`
- [ ] D) `ax.yaxis.share_with(ax2.yaxis, format='%')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
from matplotlib.ticker import PercentFormatter

for a in [ax, ax2]:
  a.set_ylim(0.0, 1.05)
  a.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
```

**Explanation:**

- **A is correct:** Because `twinx()` creates independent `YAxis` instances, limits and formatters must be configured on both `ax` and `ax2`. Iterating over `[ax, ax2]` and applying `set_ylim(0.0, 1.05)` unifies their coordinate ranges, and attaching `PercentFormatter(xmax=1.0)` converts decimal probabilities ($0.0$ to $1.0$) into clean percentage strings ($0\%$ to $100\%$) across both the left and right borders.
- **B is incorrect:** `set_twin_format()` does not exist on `Axes` objects.
- **C is incorrect:** `plt.percent_scale()` is a fictitious function.
- **D is incorrect:** `ax.yaxis.share_with()` is not a recognized Matplotlib method.

</details>

---

### Question 95

During reinforcement learning policy optimization, gradient norms experience extreme variance: most updates have small norms around $0.05$, but policy collapses trigger sudden explosion spikes up to $50,000$.

Which scale configuration handles this dynamic range effectively without masking zero or small gradient updates?

- [ ] A) `ax.set_yscale('linear')`
- [ ] B) `ax.set_yscale('symlog', linthresh=0.1, linscale=1.0)`
- [ ] C) `ax.set_yscale('log')`
- [ ] D) `ax.set_yscale('binary')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_yscale('symlog', linthresh=0.1, linscale=1.0)`

**Explanation:**

- **A is incorrect:** A linear scale will flatten all normal gradient updates ($0.05$) into an unreadable line at zero due to the $50,000$ outlier spike.
- **B is correct:** `symlog` (symmetrical log) with `linthresh=0.1` treats gradient norms below $0.1$ with linear resolution (allowing zero-gradient checks and small step inspections), while smoothly transitioning to logarithmic decades ($10^0, 10^1, 10^2, 10^3, 10^4$) for larger gradient magnitudes, making both normal training and exploding updates visible in a single unified view.
- **C is incorrect:** A standard log scale cannot plot zero or near-zero values without clipping artifacts.
- **D is incorrect:** `'binary'` is not a valid Matplotlib scale.

</details>

---

### Question 96

In a gradient-boosted decision tree model (e.g., XGBoost), a feature importance horizontal bar chart displays 15 features sorted in descending order of importance.

Which pair of commands correctly configures the categorical y-axis so that the **most important feature appears at the top**?

- [ ] A) `ax.set_yticks(range(15)); ax.set_yticklabels(features)`
- [ ] B) `ax.set_yticks(features); ax.order('descending')`
- [ ] C)
  ```python
  ax.set_yticks(range(len(features)))
  ax.set_yticklabels(features)
  ax.invert_yaxis()
```
- [ ] D) `ax.sort_y(reverse=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.invert_yaxis()
```

**Explanation:**

- **A is incorrect:** Standard Cartesian axes place index $0$ at the bottom, so Feature 0 (the top feature) would appear at the bottom of the chart rather than at the top.
- **B is incorrect:** `ax.order('descending')` is not a valid Matplotlib method.
- **C is correct:** By default, vertical axes increment from bottom to top. Calling `ax.invert_yaxis()` flips the direction so that index $0$ sits at the top of the canvas, adhering to standard executive reporting conventions where the highest-ranking item heads the list.
- **D is incorrect:** `ax.sort_y()` does not exist in the Matplotlib API.

</details>

---

### Question 97

An architect creates a $1 \times 2$ subplot figure comparing cross-entropy loss (Panel 1) and validation accuracy (Panel 2):

```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # Potential bug!
```

Why is specifying `sharey=True` in this multi-metric dashboard considered an egregious architectural defect?

- [ ] A) Because `sharey=True` crashes when used with two subplots.
- [ ] B) Because Matplotlib converts all plots to binary black-and-white when y-axes are shared.
- [ ] C) Because `sharey=True` forces both panels to be rendered on the GPU.
- [ ] D) Because cross-entropy loss is an unbounded metric in $[0, \infty)$ (often starting $> 2.0$), whereas accuracy is a bounded percentage in $[0.0, 1.0]$; sharing the y-axis forces accuracy to be evaluated on the loss scale, severely compressing the accuracy curve.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Because cross-entropy loss is an unbounded metric in $[0, \infty)$ (often starting $> 2.0$), whereas accuracy is a bounded percentage in $[0.0, 1.0]$; sharing the y-axis forces accuracy to be evaluated on the loss scale, severely compressing the accuracy curve.

**Explanation:**

- **A is incorrect:** `sharey=True` is fully valid syntax and does not crash when used with two subplots.
- **B is incorrect:** Sharing axes has no effect on color rendering or colormaps.
- **C is incorrect:** Axis sharing does not alter the rendering engine or invoke GPU computation.
- **D is correct:** `sharey=True` binds the vertical coordinate transforms of both panels to an identical coordinate range. If initial training loss is $4.5$, both subplots will adopt a y-range of $[0, 4.5]$. The accuracy curve—which naturally lives in $[0.0, 1.0]$—will be squashed into the bottom 20% of Panel 2, obscuring subtle accuracy variations. Different metrics with differing dimensional units and ranges must never share vertical axes.

</details>

---

### Question 98

A data scientist plots a probability calibration curve (reliability diagram) for a medical risk predictor. 

Which configuration establishes the canonical reliability diagram setup: both axes spanning strictly $[0.0, 1.0]$, formatted as percentages, with a diagonal dashed black identity line representing perfect calibration?

- [ ] A)
  ```python
  from matplotlib.ticker import PercentFormatter

  ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
  ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
```
- [ ] B)
  ```python
  ax.plot_calibration(diagonal=True, units='%')
```
- [ ] C)
  ```python
  ax.identity_line()
  ax.set_limits(0, 100, format='pct')
```
- [ ] D)
  ```python
  ax.set_scale('calibration')
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
from matplotlib.ticker import PercentFormatter

ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
```

**Explanation:**

- **A is correct:** In probability calibration curves, both predicted probabilities (x-axis) and observed positive frequencies (y-axis) are probabilities in $[0, 1]$. Drawing `ax.plot([0, 1], [0, 1], "k--")` renders the 45-degree diagonal reference line. Setting limits to $[0, 1]$ and attaching `PercentFormatter(xmax=1.0)` converts decimal coordinates into percentages on both axes.
- **B is incorrect:** `ax.plot_calibration()` does not exist in Matplotlib; calibration curves are constructed using scikit-learn or raw Matplotlib primitives.
- **C is incorrect:** `identity_line()` and `set_limits()` are fictitious methods.
- **D is incorrect:** `'calibration'` is not a recognized scale name in Matplotlib.

</details>

---

### Question 99

In a hyperparameter grid search heatmap (learning rate vs. batch size), an engineer wants the learning rate labels to appear along the **top horizontal edge** of the heatmap (like column headers in a matrix table), rather than along the bottom edge.

Which `tick_params` call moves the tick marks and labels from the bottom to the top?

- [ ] A) `ax.tick_params(move_to='top')`
- [ ] B) `ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)`
- [ ] C) `ax.xaxis.set_position('header')`
- [ ] D) `ax.invert_xaxis()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)`

**Explanation:**

- **A is incorrect:** `move_to='top'` is not a valid parameter in `tick_params()`; calling it raises a `ValueError`.
- **B is correct:** In Matplotlib, horizontal ticks can be independently enabled or disabled on the top and bottom spines. Setting `top=True, labeltop=True` activates tick marks and text labels along the upper border, while `bottom=False, labelbottom=False` removes them from the lower border. This is the standard pattern for rendering table-like matrix heatmaps.
- **C is incorrect:** `ax.xaxis.set_position('header')` is invalid; `set_position()` expects coordinate offsets or spine locators, not string keywords like `'header'`.
- **D is incorrect:** `invert_xaxis()` reverses the horizontal direction (left-to-right), but does not move labels from the bottom spine to the top spine.

</details>

---

### Question 100

A senior ML systems engineer reviews an automated pipeline that generates 1,000 model evaluation figures inside an overnight batch workflow:

```python
for model_name, data in models.items():
  fig, ax = plt.subplots()
  ax.plot(data["epochs"], data["val_loss"])
  ax.set_title(model_name)
  ax.xaxis.set_major_locator(MultipleLocator(5))
  fig.savefig(f"{model_name}.png", bbox_inches="tight")
  # End of iteration!
```

After generating 200 figures, the server crashes with `MemoryError: Out of memory`. What essential resource-cleanup call was omitted from the loop?

- [ ] A) `ax.delete_ticks()`
- [ ] B) `fig.clear_cache()`
- [ ] C) `plt.close(fig)` (or `plt.close('all')`)
- [ ] D) `gc.collect(force=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `plt.close(fig)` (or `plt.close('all')`)

**Explanation:**

- **A is incorrect:** `ax.delete_ticks()` does not exist in Matplotlib and would not free figure memory allocations.
- **B is incorrect:** `fig.clear_cache()` is a fictitious method.
- **C is correct:** In Matplotlib's `pyplot` state machine, every call to `plt.subplots()` registers the created `Figure` inside a global module-level dictionary (`matplotlib._pylab_helpers.Gcf`) so that active windows can be tracked. Even if the local variables `fig` and `ax` fall out of scope at the end of the loop, the global dictionary retains strong references to them, preventing Python's garbage collector from freeing the figure canvas, raster buffers, and tick artists. Calling `plt.close(fig)` explicitly deregisters the figure and frees its memory buffers.
- **D is incorrect:** Calling `gc.collect()` cannot reclaim figures because strong references remain in `Gcf.figs` until explicitly closed.

</details>

---
