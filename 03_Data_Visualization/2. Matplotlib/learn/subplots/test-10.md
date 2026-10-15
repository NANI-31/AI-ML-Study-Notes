# Matplotlib Subplots Objective Test — Set 10

---

### Question 91

A deep learning engineer builds a live training dashboard to monitor a neural network across 100 epochs. The dashboard requires a $2 \times 2$ grid:
- Top-left: Training Loss
- Top-right: Validation Loss
- Bottom-left: Training Accuracy
- Bottom-right: Validation Accuracy

Which initialization pattern provides the most readable, Pythonic tuple unpacking with shared horizontal epoch axes?

- [ ] A) `fig, axes = plt.subplots(4)`
- [ ] B) `fig, ((ax_tr_loss, ax_val_loss), (ax_tr_acc, ax_val_acc)) = plt.subplots(2, 2, figsize=(12, 8), sharex=True)`
- [ ] C) `fig, [ax1, ax2, ax3, ax4] = plt.subplots(2, 2)`
- [ ] D) `fig, ax_dict = plt.subplots(2, 2, keys=['loss', 'acc'])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `fig, ((ax_tr_loss, ax_val_loss), (ax_tr_acc, ax_val_acc)) = plt.subplots(2, 2, figsize=(12, 8), sharex=True)`

**Explanation:**

- **A is incorrect:** `plt.subplots(4)` creates a 1D column/row of 4 subplots, not a $2 \times 2$ grid.
- **B is correct:** Unpacking the $2 \times 2$ array into nested 2-tuples directly assigns semantic, self-documenting variable names to each panel (`ax_tr_loss`, `ax_val_loss`, `ax_tr_acc`, `ax_val_acc`). Furthermore, setting `sharex=True` synchronizes the epoch axis across all panels, automatically hiding redundant interior tick labels on the top row.
- **C is incorrect:** Attempting to unpack a 2D array of shape `(2, 2)` into a flat 4-element list raises `ValueError: too many values to unpack (expected 4, got 2)`.
- **D is incorrect:** `keys` is not a valid argument for `plt.subplots()`; named dictionaries are returned by `subplot_mosaic()`.

</details>

---

### Question 92

In a regression model diagnostic suite, Panel 2 displays the **Residuals vs. Fitted Values** scatter plot ($e_i = y_i - \hat{y}_i$ vs. $\hat{y}_i$).

Which plotting command adds the critical horizontal dashed zero-reference line that allows practitioners to detect residual bias and heteroscedasticity?

- [ ] A) `ax2.grid(direction='horizontal')`
- [ ] B) `ax2.plot([0, 0], [1, 1])`
- [ ] C) `ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)`
- [ ] D) `ax2.set_zero_line()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax2.axhline(0, color='red', linestyle='--', linewidth=1.2)`

**Explanation:**

- **A is incorrect:** `grid()` draws gridlines across all major ticks, not a specific highlighted zero baseline.
- **B is incorrect:** `[0, 0], [1, 1]` draws a short diagonal segment from $(0, 1)$ to $(0, 1)$.
- **C is correct:** In Matplotlib, `ax.axhline(y=0)` draws an infinite horizontal reference line spanning the entire width of the subplot regardless of x-axis scaling. Adding a prominent dashed red line (`color='red', linestyle='--'`) at $y = 0$ provides the visual anchor necessary to verify if residuals are symmetrically dispersed around zero with homoscedastic variance.
- **D is incorrect:** `set_zero_line()` does not exist in Matplotlib.

</details>

---

### Question 93

A classification model evaluation dashboard combines four distinct visual diagnostics:
1. ROC Curve with the diagonal random-guess reference line
2. Precision-Recall Curve
3. Confusion Matrix heatmap
4. Feature Importance horizontal bar chart

Which command correctly overlays the standard diagonal random-chance baseline on the ROC Curve subplot `ax_roc` ($x \in [0, 1], y \in [0, 1]$)?

- [ ] A) `ax_roc.diagonal()`
- [ ] B) `ax_roc.axline(slope=0, intercept=0)`
- [ ] C) `ax_roc.set_chance_line()`
- [ ] D) `ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Chance (AUC = 0.50)')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Chance (AUC = 0.50)')`

**Explanation:**

- **A is incorrect:** `diagonal()` is not a method on `Axes`.
- **B is incorrect:** An `axline` with `slope=0` and `intercept=0` draws a horizontal line along the x-axis ($y=0$), not a diagonal reference line.
- **C is incorrect:** `set_chance_line()` does not exist in the Matplotlib API.
- **D is correct:** In ROC curve analysis, a non-discriminating random classifier produces a straight 45-degree diagonal line connecting $(0, 0)$ to $(1, 1)$. Drawing `ax.plot([0, 1], [0, 1], color='navy', linestyle='--')` renders this standard visual benchmark, against which the trained model's true positive rate curve is compared.

</details>

---

### Question 94

A data engineer is building an exploratory data analysis (EDA) dashboard across 6 variables. Features 0 to 3 are continuous numerical metrics (requiring histograms `ax.hist()`), while Features 4 and 5 are categorical attributes (requiring bar charts `ax.bar()`).

Which code structure cleanly accomplishes this within a single dynamic loop?

- [ ] A)
  ```python
  fig, axes = plt.subplots(2, 3, figsize=(15, 8))
  for i, ax in enumerate(axes.flat):
    col = df.columns[i]
    if pd.api.types.is_numeric_dtype(df[col]):
      ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="black")
    else:
      counts = df[col].value_counts()
      ax.bar(counts.index, counts.values, color="coral")
    ax.set_title(col)
```
- [ ] B)
  ```python
  fig, axes = plt.subplots(2, 3, type_switch=True)
```
- [ ] C)
  ```python
  axes.auto_plot(df)
```
- [ ] D)
  ```python
  fig.plot_mixed(df.columns)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
  col = df.columns[i]
  if pd.api.types.is_numeric_dtype(df[col]):
    ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="black")
  else:
    counts = df[col].value_counts()
    ax.bar(counts.index, counts.values, color="coral")
  ax.set_title(col)
```

**Explanation:**

- **A is correct:** This reflects idiomatic data science workflow: instantiating a grid via `plt.subplots(2, 3)`, iterating across subplots via `axes.flat`, and applying conditional plotting logic (`hist()` for continuous data, `bar()` for discrete value counts) per panel while dynamically setting subplot titles.
- **B is incorrect:** `type_switch` is not a valid parameter of `plt.subplots()`.
- **C is incorrect:** NumPy arrays of axes do not possess an `auto_plot` method.
- **D is incorrect:** `fig.plot_mixed()` is a fictitious function name not present in the Matplotlib API.

</details>

---

### Question 95

During model training, a developer writes an epoch-by-epoch live visualizer:

```python
for epoch in range(1000):
  train_one_epoch()
  fig, axes = plt.subplots(1, 2)  # Inside training loop!
  axes[0].plot(train_losses)
  axes[1].plot(val_losses)
  plt.pause(0.01)
```

Within a few minutes, training slows to a crawl and the machine runs out of RAM. What is the root cause of this catastrophic failure, and how must the architecture be refactored?

- [ ] A) Matplotlib limits GUI events to 100 draw calls per minute, causing the operating system to throttle CPU execution.
- [ ] B) `plt.pause()` creates an unclosed thread on every iteration that blocks Python's global interpreter lock (GIL).
- [ ] C) Calling `plt.subplots()` inside the loop creates 1,000 separate `Figure` and `Axes` objects in memory without closing them. The figure and axes must be initialized **once outside the loop**, and either updated dynamically (`line.set_data()`) or cleared per step (`ax.cla()`).
- [ ] D) `plt.pause(0.01)` causes `FigureCanvas` buffers to overwrite disk swap files until physical storage is exhausted.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Calling `plt.subplots()` inside the loop creates 1,000 separate `Figure` and `Axes` objects in memory without closing them. The figure and axes must be initialized **once outside the loop**, and either updated dynamically (`line.set_data()`) or cleared per step (`ax.cla()`).

**Explanation:**

- **A is incorrect:** Matplotlib does not impose a 100 draw calls per minute quota.
- **B is incorrect:** `plt.pause()` handles GUI event loop polling; it does not spawn runaway unclosed OS threads.
- **C is correct:** Every invocation of `plt.subplots()` allocates a new top-level GUI canvas and multiple artist buffers. Calling it 1,000 times inside an iterative training loop floods the pyplot GUI manager with thousands of orphaned figures, triggering massive memory leaks and severe garbage collection slowdowns. The correct design instantiates `fig, axes = plt.subplots()` a single time prior to the loop, and inside the loop simply clears the axes (`ax.cla()`) or updates line data objects (`line.set_ydata(...)`).
- **D is incorrect:** `plt.pause()` does not write to disk swap or exhaust physical storage.

</details>

---

### Question 96

A researcher tests three neural network architectures across learning rates and batch sizes, producing three validation score matrices. They arrange three heatmaps in a $1 \times 3$ grid using `ax.imshow()`.

To ensure that color comparisons across all three architectures are mathematically valid and visual scales are unified, what two configurations must be enforced?

- [ ] A) The colormap must be inverted on the middle panel, and labels must be set to bold.
- [ ] B) All three `imshow()` calls must share identical `vmin` and `vmax` bounds, and a single shared colorbar should represent all three panels.
- [ ] C) Each panel must use a completely different colormap (e.g., `'Reds'`, `'Blues'`, `'Greens'`).
- [ ] D) The figure aspect ratio must be set to `0.33`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) All three `imshow()` calls must share identical `vmin` and `vmax` bounds, and a single shared colorbar should represent all three panels.

**Explanation:**

- **A is incorrect:** Inverting colormaps on arbitrary panels destroys visual comparability.
- **B is correct:** By default, `imshow()` scales colors locally to the min and max of each individual array. If Model A scores range $[0.5, 0.7]$ and Model B scores range $[0.8, 0.95]$, identical colors would represent completely different performance numbers. Enforcing global `vmin=global_min` and `vmax=global_max` across all calls anchors the colormap uniformly, allowing direct visual comparison across all panels alongside a single shared colorbar.
- **C is incorrect:** Using different colormaps prevents direct visual comparison.
- **D is incorrect:** Figure aspect ratio has no bearing on color normalization truth.

</details>

---

### Question 97

In a classification diagnostic dashboard, the Confusion Matrix subplot `ax_cm` displays 10 text class names along the horizontal axis. Due to tight column boundaries, the class names overlap and collide horizontally.

Which command rotates the x-axis tick labels by 45 degrees with right-alignment to ensure perfect legibility?

- [ ] A) `ax_cm.tick_params(axis='x', rotation=45)` (or `plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')`)
- [ ] B) `ax_cm.tilt_labels(45)`
- [ ] C) `ax_cm.rotate_text('x', angle=45)`
- [ ] D) `ax_cm.set_angle(45)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax_cm.tick_params(axis='x', rotation=45)` (or `plt.setp(ax_cm.get_xticklabels(), rotation=45, ha='right')`)

**Explanation:**

- **A is correct:** In Matplotlib, `ax.tick_params(axis='x', rotation=45)` or using `plt.setp(ax.get_xticklabels(), rotation=45, ha='right')` cleanly tilts the labels at a 45-degree angle. Setting the horizontal alignment anchor to `'right'` aligns the end of the text label directly beneath its corresponding tick mark.
- **B is incorrect:** `tilt_labels` is not a valid method on `Axes`.
- **C is incorrect:** `rotate_text` does not exist in the Matplotlib API.
- **D is incorrect:** `set_angle` is not an `Axes` method.

</details>

---

### Question 98

A data scientist evaluates 5-fold cross-validation results by plotting validation curves for each fold. They stack the 5 folds vertically in a $5 \times 1$ grid:

```python
fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True, sharey=True)
```

Why is setting both `sharex=True` and `sharey=True` critical for cross-validation fold comparison?

- [ ] A) It ensures that the folds run in parallel on 5 CPU cores.
- [ ] B) It forces each fold to have identical training data.
- [ ] C) It reduces figure file size by $80\%$.
- [ ] D) It locks all 5 panels to the exact same vertical metric range and horizontal epoch range, allowing variance and instability across folds to be immediately visible without optical distortion.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It locks all 5 panels to the exact same vertical metric range and horizontal epoch range, allowing variance and instability across folds to be immediately visible without optical distortion.

**Explanation:**

- **A is incorrect:** Subplot parameterization does not control Python multiprocessing or CPU thread execution.
- **B is incorrect:** Cross-validation data partitioning is handled upstream in scikit-learn/PyTorch.
- **C is incorrect:** Sharing axes does not reduce file size by 80%.
- **D is correct:** In cross-validation evaluation, practitioners need to visually detect whether certain folds suffer from severe instability or divergence. If each subplot auto-scales independently, a fold with massive swings might visually appear identical to a fold with tiny fluctuations. Shared axes enforce identical spatial scales, making variance differences instantly noticeable.

</details>

---

### Question 99

When preparing an end-to-end 6-panel machine learning dashboard for submission to an academic journal, which `savefig()` configuration produces a publication-quality vector graphic with cropped whitespace and no label truncation?

- [ ] A) `fig.savefig('dashboard.jpg', quality=50)`
- [ ] B) `fig.savefig('dashboard.png', dpi=72)`
- [ ] C) `fig.savefig('dashboard.pdf', dpi=300, bbox_inches='tight')`
- [ ] D) `fig.export_journal('dashboard', format='eps')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `fig.savefig('dashboard.pdf', dpi=300, bbox_inches='tight')`

**Explanation:**

- **A is incorrect:** JPEG is a lossy raster format that introduces compression artifacts around text, lines, and sharp axis spines.
- **B is incorrect:** PNG at 72 DPI is low resolution and will appear pixelated and blurry when scaled or printed in academic papers.
- **C is correct:** PDF is an infinitely scalable vector format preferred by scientific publishers (LaTeX, IEEE, ACM). Adding `dpi=300` ensures embedded raster elements render in high resolution, and `bbox_inches='tight'` computes the tightest bounding box encompassing all titles, legends, and outer axis labels, preventing any peripheral text from being clipped.
- **D is incorrect:** `export_journal()` is a fictitious function name not present in the Matplotlib library.

</details>

---

### Question 100

A senior ML architect reviews a full diagnostic dashboard script:

```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Training & Val Loss
ax1.plot(epochs, train_loss, label="Train")
ax1.plot(epochs, val_loss, label="Val")
ax1.set_title("Loss Curves")
ax1.legend()

# Panel 2: Generalization Gap
ax2.plot(epochs, np.array(val_loss) - np.array(train_loss), color="red")
ax2.set_title("Generalization Gap (Val - Train)")
ax2.axhline(0, linestyle="--", color="black")

# Panel 3: ROC Curve
ax3.plot(fpr, tpr, color="darkorange", label=f"AUC = {roc_auc:.2f}")
ax3.plot([0, 1], [0, 1], "navy", linestyle="--")
ax3.set_title("ROC Curve")
ax3.legend()

# Panel 4: Precision-Recall
ax4.plot(recall, precision, color="green")
ax4.set_title("Precision-Recall Curve")

fig.suptitle("ResNet-50 Diagnostic Suite", fontsize=16, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

If the model begins to **overfit severely** after Epoch 20, how will this pathology manifest visually in Panels 1 and 2?

- [ ] A) Both Train and Val loss will drop to 0 simultaneously.
- [ ] B) In Panel 1, `train_loss` continues downward while `val_loss` reverses and climbs upward; in Panel 2, the generalization gap diverges sharply above the zero line into positive territory.
- [ ] C) The ROC curve in Panel 3 will collapse into a vertical line.
- [ ] D) The figure title will disappear from the canvas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) In Panel 1, `train_loss` continues downward while `val_loss` reverses and climbs upward; in Panel 2, the generalization gap diverges sharply above the zero line into positive territory.

**Explanation:**

- **A is incorrect:** Overfitting causes validation loss to diverge and increase, not drop to zero alongside training loss.
- **B is correct:** The classic hallmark of overfitting is diverging training and validation performance: the network continues memorizing training samples (driving training loss toward zero), but loses generalization capability on unseen validation data (causing validation loss to plateau and rise). In Panel 2, the generalization gap metric $(\text{Val} - \text{Train})$ sharply diverges upward away from the baseline $0$, providing immediate diagnostic confirmation of overfitting.
- **C is incorrect:** Overfitting degrades validation ROC-AUC but does not collapse the ROC curve into a vertical line.
- **D is incorrect:** Figure titles and axis text annotations remain completely static regardless of plotted loss values.

</details>

---
