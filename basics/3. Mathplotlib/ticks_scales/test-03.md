# Matplotlib Ticks & Scales Objective Test — Set 03

---

### Question 21

A data scientist models an anomaly detection threshold where scores range from $-5.0$ to $+50.0$. For presentation, they want the viewing window strictly bounded between $0.0$ and $25.0$, clipping outlier spikes outside this view.

Which command achieves this in the Object-Oriented interface?

- [ ] A) `ax.set_xlim(0.0, 25.0)` (or `ax.set_xlim(left=0.0, right=25.0)`)
- [ ] B) `ax.clip_bounds(0.0, 25.0)`
- [ ] C) `ax.xaxis.set_range(0.0, 25.0)`
- [ ] D) `ax.restrict_x([0.0, 25.0])`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.set_xlim(0.0, 25.0)` (or `ax.set_xlim(left=0.0, right=25.0)`)

**Explanation:**

- **A is correct:** In Matplotlib's OO interface, `ax.set_xlim(left, right)` explicitly defines the visible coordinate span of the horizontal axis. Any data points with x-values less than $0.0$ or greater than $25.0$ remain in the dataset but fall outside the active viewport clipping boundary.
- **B is incorrect:** `clip_bounds` is not a valid method on an `Axes` object.
- **C is incorrect:** `xaxis.set_range` does not exist in the Matplotlib API.
- **D is incorrect:** `restrict_x` is not a recognized Matplotlib function.

</details>

---

### Question 22

A competitive leaderboard dashboard plots model benchmark rankings (Rank 1 to Rank 20). In standard Cartesian plots, smaller values appear at the bottom, which would place Rank 1 at the bottom and Rank 20 at the top.

Which command cleanly inverts the vertical axis so that Rank 1 appears proudly at the very top of the plot?

- [ ] A) `ax.reverse_y()`
- [ ] B) `ax.flip(axis='vertical')`
- [ ] C) `ax.invert_yaxis()` (or `ax.set_ylim(20, 1)`)
- [ ] D) `ax.yaxis.sort(ascending=False)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.invert_yaxis()` (or `ax.set_ylim(20, 1)`)

**Explanation:**

- **A is incorrect:** `reverse_y` is not an `Axes` method in Matplotlib.
- **B is incorrect:** `flip` is not a recognized method on `Axes`.
- **C is correct:** In Matplotlib, calling `ax.invert_yaxis()` flips the direction of the vertical axis, causing coordinate values to decrease as you move upward. Equivalently, passing reversed boundary values to `ax.set_ylim(bottom=20, top=1)` directly establishes Rank 1 at the top margin and Rank 20 at the bottom baseline.
- **D is incorrect:** `yaxis.sort` does not exist on `YAxis`.

</details>

---

### Question 23

An engineer wants to guarantee that a training loss plot always starts at a lower baseline of $y = 0.0$, but wants Matplotlib to automatically calculate the upper bound based on the maximum loss observed in the data.

How can this partial (one-sided) boundary constraint be applied?

- [ ] A) `ax.set_ylim(0.0, 'auto_scale')`
- [ ] B) `ax.set_ylim(bottom=0.0)` (or `plt.ylim(ymin=0.0)`)
- [ ] C) `ax.set_ylim(0.0, float('inf'))`
- [ ] D) `ax.lock_baseline(0.0)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.set_ylim(bottom=0.0)` (or `plt.ylim(ymin=0.0)`)

**Explanation:**

- **A is incorrect:** Passing a string `'auto_scale'` raises a `ValueError` because limits must be numeric.
- **B is correct:** In `ax.set_ylim(bottom=None, top=None)`, leaving one boundary as `None` (or omitting it while supplying the other as a keyword argument like `bottom=0.0`) tells Matplotlib to pin that specific boundary to the specified value while allowing the opposite boundary to continue autoscaling dynamically to fit the data.
- **C is incorrect:** Passing `float('inf')` will produce transform errors or invalid rendering bounding boxes.
- **D is incorrect:** `lock_baseline` is not a Matplotlib method.

</details>

---

### Question 24

By default, Matplotlib adds a subtle padding margin (approximately $5\%$) beyond the minimum and maximum data points so markers don't touch the outer spines. 

Which method call completely removes this padding on the horizontal axis, forcing the view limits to fit **strictly flush** against the data extrema?

- [ ] A) `ax.flush_x(True)`
- [ ] B) `ax.set_padding(0)`
- [ ] C) `ax.remove_margins('x')`
- [ ] D) `ax.autoscale(enable=True, axis='x', tight=True)` (or `ax.margins(x=0)`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.autoscale(enable=True, axis='x', tight=True)` (or `ax.margins(x=0)`)

**Explanation:**

- **A is incorrect:** `flush_x` is not a valid method in Matplotlib.
- **B is incorrect:** `set_padding` is not a recognized method on `Axes`.
- **C is incorrect:** `remove_margins` does not exist in the Matplotlib API.
- **D is correct:** Setting `tight=True` in `ax.autoscale(enable=True, axis='x', tight=True)` instructs the autoscaling engine to eliminate the default data margin, pinning the viewport bounds exactly to $\min(x)$ and $\max(x)$. Equivalently, `ax.margins(x=0)` sets the proportional margin padding on the x-axis to zero.

</details>

---

### Question 25

What is the effect of invoking `ax.margins(x=0.10, y=0.20)` on a scatter plot?

- [ ] A) It adds an empty margin buffer equal to $10\%$ of the data range along the left and right edges, and $20\%$ of the data range along the bottom and top edges, prior to rendering.
- [ ] B) It shrinks the entire figure window by $10\%$ horizontally and $20\%$ vertically.
- [ ] C) It sets the opacity of the outer plot region to $10\%$ and $20\%$.
- [ ] D) It adds 10 pixels of padding on the x-axis and 20 pixels on the y-axis.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) It adds an empty margin buffer equal to $10\%$ of the data range along the left and right edges, and $20\%$ of the data range along the bottom and top edges, prior to rendering.

**Explanation:**

- **A is correct:** `ax.margins(x, y)` configures the fractional padding added beyond the extrema of the data during autoscaling. Supplying `x=0.10` and `y=0.20` adds $10\%$ extra space on both sides of the horizontal data span and $20\%$ extra space above and below the vertical data span, ensuring points near the edges have ample breathing room.
- **B is incorrect:** `ax.margins()` adjusts data coordinate limits, not the outer GUI window dimensions.
- **C is incorrect:** Margins do not alter artist opacity.
- **D is incorrect:** Margins are fractional ratios of the data range, not fixed pixel offsets.

</details>

---

### Question 26

A script dynamically inspects the current visible limits of an axis after plotting complex geometries. Which methods return the active limits as 2-tuples of floats?

- [ ] A) `ax.read_x_bounds()` and `ax.read_y_bounds()`
- [ ] B) `ax.xaxis.limits` and `ax.yaxis.limits`
- [ ] C) `ax.get_xlim()` and `ax.get_ylim()`
- [ ] D) `ax.bounds(axis='x')` and `ax.bounds(axis='y')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.get_xlim()` and `ax.get_ylim()`

**Explanation:**

- **A is incorrect:** `read_x_bounds` and `read_y_bounds` are fictitious function names.
- **B is incorrect:** `xaxis.limits` and `yaxis.limits` are not attributes in Matplotlib.
- **C is correct:** `ax.get_xlim()` returns a 2-tuple `(left, right)` representing the current minimum and maximum values along the x-axis in data coordinates. Similarly, `ax.get_ylim()` returns `(bottom, top)` for the y-axis.
- **D is incorrect:** `ax.bounds` does not exist as an `Axes` method.

</details>

---

### Question 27

A marketing team presents a bar chart comparing customer retention rates across two quarters: $97.5\%$ in Q1 and $98.5\%$ in Q2. The chart's y-axis limits are set using `plt.ylim(97.0, 99.0)`.

Why is this configuration widely flagged as a violation of ethical data visualization standards?

- [ ] A) Because Matplotlib renders truncated floats with visual anti-aliasing artifacts.
- [ ] B) Because truncating the baseline exaggerates a tiny $1\%$ absolute increase into a dramatic visual tenfold height difference, deceptively misleading viewers who perceive bar height proportionally from zero.
- [ ] C) Because percentages can only be plotted on logarithmic scales.
- [ ] D) Because `plt.ylim()` automatically rounds decimal percentages down to integers.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Because truncating the baseline exaggerates a tiny $1\%$ absolute increase into a dramatic visual tenfold height difference, deceptively misleading viewers who perceive bar height proportionally from zero.

**Explanation:**

- **A is incorrect:** Anti-aliasing quality is unrelated to axis limits.
- **B is correct:** In bar charts, viewers intuitively encode quantity via the proportional length/height of the bar from the baseline. Truncating the y-axis (e.g., starting at $97.0\%$ instead of $0\%$) distorts the visual ratio: a bar of height $1.5$ (representing $98.5\%$) appears three times taller than a bar of height $0.5$ (representing $97.5\%$), creating the false impression of massive growth when the true difference is negligible. Responsible visualization ethics dictate starting bar charts at zero (or using a dot/line plot if focusing on variance).
- **C is incorrect:** Percentages are typically plotted on linear scales; log scales cannot represent zero or negative values.
- **D is incorrect:** `plt.ylim()` preserves exact floating-point values without rounding.

</details>

---

### Question 28

When visualizing a $100 \times 100$ confusion matrix or image tensor using `ax.imshow(matrix)`, why is row index $0$ rendered at the top of the vertical axis by default rather than at the bottom?

- [ ] A) It is a legacy software defect that cannot be fixed without breaking backwards compatibility.
- [ ] B) Because computer monitors scan pixels from bottom to top.
- [ ] C) Because Matplotlib rotates the matrix by $180^\circ$ to conserve memory.
- [ ] D) Because `imshow()` follows standard matrix and digital image convention (where row 0 begins at the top and index increases downward), which Matplotlib enforces by defaulting to an inverted y-axis (`origin='upper'`).

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Because `imshow()` follows standard matrix and digital image convention (where row 0 begins at the top and index increases downward), which Matplotlib enforces by defaulting to an inverted y-axis (`origin='upper'`).

**Explanation:**

- **A is incorrect:** Image origin convention is an intentional design choice matching 2D matrix storage, not a software bug.
- **B is incorrect:** Hardware monitor scanlines do not dictate Matplotlib's coordinate transformations.
- **C is incorrect:** Matplotlib does not rotate the input data array in memory.
- **D is correct:** In linear algebra and computer vision, 2D array coordinates index elements as `matrix[row, col]`, where Row 0 is the top row and row index increments downward. To match this natural indexing, `ax.imshow()` defaults to `origin='upper'`, which inverts the y-axis so $y = 0$ is at the top. Standard Cartesian plots (where $y = 0$ is at the bottom) can be restored if desired by setting `origin='lower'`.

</details>

---

### Question 29

In an astronomical spectroscopy plot, stellar brightness is measured in magnitudes, where smaller or more negative numbers denote brighter stars. A researcher wants the x-axis to run backwards from $+15$ on the left to $-2$ on the right.

Which command accomplishes this reversed horizontal span?

- [ ] A) `ax.set_xlim(15, -2)` (or `ax.invert_xaxis()`)
- [ ] B) `ax.reverse(axis='x')`
- [ ] C) `ax.xaxis.flip_left_right()`
- [ ] D) `ax.set_xlim(-2, 15, direction='backward')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.set_xlim(15, -2)` (or `ax.invert_xaxis()`)

**Explanation:**

- **A is correct:** In Matplotlib, the first argument to `set_xlim(left, right)` sets the coordinate value at the left spine, and the second argument sets the coordinate at the right spine. Passing `left=15` and `right=-2` creates an inverted horizontal axis where values decrease from left to right. `ax.invert_xaxis()` achieves the same inversion toggle on existing limits.
- **B is incorrect:** `reverse` is not an `Axes` method in Matplotlib.
- **C is incorrect:** `flip_left_right` does not exist on `XAxis`.
- **D is incorrect:** `direction='backward'` is not a valid parameter for `set_xlim()`.

</details>

---

### Question 30

A machine learning engineer plots a Precision-Recall Curve. To ensure the curve is evaluated against the full theoretical probability envelope, they set:

```python
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
```

Why is setting `top=1.05` instead of `top=1.0` a standard visualization best practice in model evaluation curves?

- [ ] A) Because neural network precision can legally reach $105\%$.
- [ ] B) Because Matplotlib requires all axes to be divisible by 5.
- [ ] C) It provides a small vertical buffer above $1.0$, preventing curves that achieve perfect precision ($1.0$) from rendering directly on top of the upper spine, thereby ensuring the line remains clearly visible.
- [ ] D) It forces Matplotlib to compute confidence intervals automatically.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) It provides a small vertical buffer above $1.0$, preventing curves that achieve perfect precision ($1.0$) from rendering directly on top of the upper spine, thereby ensuring the line remains clearly visible.

**Explanation:**

- **A is incorrect:** Precision is mathematically bounded in $[0.0, 1.0]$; it cannot exceed $100\%$.
- **B is incorrect:** Matplotlib does not impose divisibility constraints on limits.
- **C is correct:** In many classification tasks, models achieve perfect precision ($1.0$) at low recall thresholds. If the upper y-limit is capped strictly at $1.0$, the plotted line trajectory overlaps the top black bounding spine of the subplot, obscuring line thickness, color, and marker symbols. Adding a modest buffer (e.g., $1.05$) keeps the line within the visible canvas while preserving the standard $0$ to $1$ interpretability.
- **D is incorrect:** Axis limits do not trigger statistical confidence interval calculations.

</details>

---
