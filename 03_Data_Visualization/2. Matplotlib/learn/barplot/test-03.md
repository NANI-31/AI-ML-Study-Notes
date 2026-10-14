# Matplotlib Bar Plot Objective Test — Set 03

---

### 21. Question

A data analyst creates a bar chart comparing quarterly expenses across departments. The categories are long strings. Which chart type is recommended and why?

```python
departments = [
    "Human Resources", "Research & Development",
    "Marketing & Sales", "Information Technology",
    "Operations & Logistics"
]
expenses = [245, 820, 380, 510, 430]
```

- [ ] **A.** Vertical bar chart — long names can be rotated 45° with `plt.xticks(rotation=45, ha="right")`
- [ ] **B.** Horizontal bar chart (`plt.barh`) — long category names fit naturally on the y-axis without rotation or overlap
- [ ] **C.** Both are equally appropriate; the choice is purely aesthetic
- [ ] **D.** Only pie charts can handle long category labels effectively

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Partially viable but not preferred):** Rotating x-axis labels 45° can work, but with very long strings like "Research & Development" and "Operations & Logistics," even 45° rotation may cause overlapping labels or require excessive figure width. This is a workaround for a limitation of vertical bar charts, not the best solution.
- **Option B (Correct):** Horizontal bar charts (`plt.barh`) place category labels on the y-axis, which runs vertically. Long strings can be displayed horizontally without any truncation, rotation, or overlap because there is ample vertical space between rows. This is the established best practice in data visualisation for categories with long names — feature importance charts, survey results, and ranked lists all commonly use `barh` for this reason.
- **Option C (Incorrect):** The choice is not purely aesthetic — there is a clear functional reason to prefer `barh`. With 5 long category names, vertical bars require label manipulation that `barh` avoids entirely. Best practice guidelines consistently recommend horizontal bars for long labels.
- **Option D (Incorrect):** Pie charts are generally not recommended for 5+ categories because comparing arc sizes is cognitively harder than comparing lengths. Pie charts do not solve the label-length problem and introduce other comparison difficulties.

</details>

---

### 22. Question

A developer writes the following code to create a multi-colour bar chart for an NLP benchmark:

```python
tasks  = ["Sentiment", "NER", "QA", "Summarization"]
f1_scores = [0.882, 0.791, 0.843, 0.756]

palette = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
plt.bar(tasks, f1_scores, color=palette, width=0.55,
        edgecolor="white", linewidth=1.0, alpha=0.9)
plt.ylim(0, 1)
plt.ylabel("F1 Score")
```

What is the total number of rendering layers (artists) that each bar rectangle consists of in this configuration?

- [ ] **A.** 1 layer — just the coloured fill
- [ ] **B.** 2 layers — a coloured fill with a white border
- [ ] **C.** 3 layers — fill, outer border, and inner glow
- [ ] **D.** 4 layers — one layer per task

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A bar without `edgecolor` or with `edgecolor="none"` would consist of a single fill layer. Here, `edgecolor="white"` is explicitly set, adding a white border.
- **Option B (Correct):** Each `Rectangle` patch artist in Matplotlib consists of a **face (fill)** and an **edge (border)**. With `color=palette` (sets face colour) and `edgecolor="white"` (sets edge colour), each bar has a coloured interior fill and a white border outline drawn on top. These are two rendering components of the same patch object — not separate artist objects, but two composited visual layers rendered as one `Rectangle`.
- **Option C (Incorrect):** Matplotlib's standard `bar` rectangles do not have an "inner glow" layer. Glow effects require custom rendering with `matplotlib.patheffects` or layered patches with varying alpha, which is not present in this code.
- **Option D (Incorrect):** The four bars are four separate `Rectangle` artist objects — but each individual bar still has the same internal structure (face + edge). The question asks about layers within a single bar, not total artists in the chart.

</details>

---

### 23. Question

A researcher compares training and test accuracy for three models and produces the following chart:

```python
models = ["Model X", "Model Y", "Model Z"]
train_acc = [0.95, 0.98, 0.87]
test_acc  = [0.92, 0.84, 0.86]

x = np.arange(3)
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, train_acc, width, label="Train", color="#2980B9")
ax.bar(x + width/2, test_acc,  width, label="Test",  color="#E74C3C")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
```

For **Model Y** (index `i = 1`), what are the exact x-positions of the **center** of the Train bar and the **center** of the Test bar?

- [ ] **A.** Train center = 0.825, Test center = 1.175
- [ ] **B.** Train center = 0.65, Test center = 1.35
- [ ] **C.** Train center = 1.175, Test center = 0.825
- [ ] **D.** Train center = 1.0, Test center = 1.35

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For index `i = 1` (Model Y): `x[1] = 1`. Train bar center = `x[1] - width/2 = 1 - 0.35/2 = 1 - 0.175 = 0.825`. Test bar center = `x[1] + width/2 = 1 + 0.35/2 = 1 + 0.175 = 1.175`. The two bars are symmetrically placed around the tick position x=1, each offset by 0.175 units.
- **Option B (Incorrect):** Train=0.65 and Test=1.35 would correspond to `width = 0.70` (since 1 − 0.70/2 = 0.65 and 1 + 0.70/2 = 1.35). With `width=0.35`, the offset is 0.175, not 0.35.
- **Option C (Incorrect):** This reverses the train and test positions. The Train bar uses `x - width/2 = 0.825` (to the left) and Test uses `x + width/2 = 1.175` (to the right), not the other way around.
- **Option D (Incorrect):** Train center = 1.0 would mean no offset (bar centered exactly at x=1), but the code subtracts `width/2`. Test center = 1.35 doesn't match the formula with width=0.35.

</details>

---

### 24. Question

A developer attempts to create a grouped bar chart but uses category strings directly for arithmetic:

```python
models = ["LR", "RF", "SVM"]
acc = [0.85, 0.92, 0.88]
prec = [0.83, 0.90, 0.86]

width = 0.35
fig, ax = plt.subplots()
ax.bar(models - width/2, acc, width, label="Accuracy")   # Bug here?
ax.bar(models + width/2, prec, width, label="Precision")
```

What error does this code produce, and what is the correct fix?

- [ ] **A.** `TypeError` — you cannot apply arithmetic to a list of strings; fix: replace `models` with `x = np.arange(len(models))` and use `x - width/2`
- [ ] **B.** `ValueError` — `width/2` is too small for string offsets; fix: increase `width` to 1.0
- [ ] **C.** No error — Matplotlib automatically converts string positions to integers for arithmetic
- [ ] **D.** `IndexError` — the models list has 3 elements but offset requires 4; fix: add a dummy model

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `models` is a Python list of strings `["LR", "RF", "SVM"]`. You cannot perform arithmetic (`-` or `+`) on a list of strings. `models - width/2` raises a `TypeError: unsupported operand type(s) for -: 'list' and 'float'`. The correct approach is to create a numeric position array: `x = np.arange(len(models))` which gives `array([0, 1, 2])`. Then `x - width/2` (array subtraction) works correctly. Finally, `ax.set_xticks(x); ax.set_xticklabels(models)` maps the numeric positions back to string labels.
- **Option B (Incorrect):** The error is not a `ValueError` related to width size. The fundamental issue is the type mismatch — strings cannot be subtracted. Increasing width would not resolve a `TypeError`.
- **Option C (Incorrect):** Matplotlib does NOT automatically convert string positions to integers for arithmetic. When strings are passed to `plt.bar()`, Matplotlib handles them internally as categorical positions, but this conversion happens inside `bar()` — not when you try to do arithmetic with the string list BEFORE passing it to `bar()`.
- **Option D (Incorrect):** The error is not an `IndexError` — there is no index out of bounds. The issue is that you're attempting arithmetic on non-numeric types.

</details>

---

### 25. Question

A developer creates a properly working grouped bar chart for three models and four metrics:

```python
metrics = ["Accuracy", "Precision", "Recall", "F1"]
lr_scores  = [0.85, 0.83, 0.81, 0.82]
rf_scores  = [0.92, 0.90, 0.89, 0.895]
svm_scores = [0.88, 0.86, 0.85, 0.855]

x = np.arange(len(metrics))   # [0, 1, 2, 3]
width = 0.25

fig, ax = plt.subplots()
ax.bar(x - width,      lr_scores,  width, label="LR")
ax.bar(x,              rf_scores,  width, label="RF")
ax.bar(x + width,      svm_scores, width, label="SVM")

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
```

For the "Recall" metric (index `i = 2`), what are the **left edges** of the LR, RF, and SVM bars respectively?

- [ ] **A.** LR: 1.625, RF: 1.875, SVM: 2.125
- [ ] **B.** LR: 1.75, RF: 2.0, SVM: 2.25
- [ ] **C.** LR: 1.875, RF: 2.125, SVM: 2.375
- [ ] **D.** LR: 1.5, RF: 1.75, SVM: 2.0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Left edges of 1.625, 1.875, 2.125 would correspond to bar centers at 1.75, 2.0, 2.25 with `width=0.25` (left edge = center − width/2 = 1.75 − 0.125 = 1.625). These are the left edges if centers are at those positions — but let's verify the centers first.
- **Option B (Correct):** For index `i = 2` (Recall), `x[2] = 2`. The bar centers are: LR = `x - width = 2 - 0.25 = 1.75`, RF = `x = 2.0`, SVM = `x + width = 2 + 0.25 = 2.25`. The bar center IS where `ax.bar()` places the bar (it's centered on that coordinate by default, with `align="center"`). The LEFT EDGE of each bar = center − width/2: LR = `1.75 − 0.125 = 1.625`. Wait — that gives left edges at Option A values. Let me reconsider: with `align="center"`, the bar is centered ON the x-coordinate passed. So LR center = 1.75, RF center = 2.0, SVM center = 2.25. Left edges: LR = 1.75 − 0.125 = 1.625, RF = 1.875, SVM = 2.125. This matches Option A. The answer depends on the question — if asking for centers, Option B is correct. The question asks for left edges so Option A is correct. **Revised: Option A is correct for left edges.**
- **Option C (Incorrect):** Not matching the formula.
- **Option D (Incorrect):** Not matching the formula.

> **Note:** The correct answer for left edges is **A** (1.625, 1.875, 2.125). The bar centers are at 1.75, 2.0, 2.25 (Option B values). **Corrected answer: A**.

</details>

---

### 26. Question

A developer creates a 3-series grouped bar chart with `width=0.25` for four categories. At category index `i=0` ("Q1"), what is the gap (empty space) between the rightmost bar of one category group and the leftmost bar of the next category group?

Given: centers at `i-width`, `i`, `i+width` = −0.25, 0.0, +0.25 for group 0.
Next group (i=1): centers at 0.75, 1.0, 1.25.

- [ ] **A.** 0.00 (bars are touching)
- [ ] **B.** 0.25 (one bar-width gap)
- [ ] **C.** 0.375 (one-and-a-half bar-widths)
- [ ] **D.** 0.50 (two bar-widths gap)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** If bars were touching, the right edge of the last bar in group 0 would exactly equal the left edge of the first bar in group 1. That's not the case here.
- **Option B (Incorrect):** A 0.25-unit gap would be one bar-width. The actual gap is larger.
- **Option C (Incorrect):** 0.375 units = 1.5 × 0.25. Let's calculate and verify.
- **Option D (Correct):** Right edge of the SVM bar in group 0: center = 0.25, right edge = 0.25 + 0.125 = **0.375**. Left edge of the LR bar in group 1: center = 0.75, left edge = 0.75 − 0.125 = **0.625**. Gap = 0.625 − 0.375 = **0.25**. Hmm, that gives 0.25. Let me recheck: center of rightmost bar of group 0 = `0 + width = 0 + 0.25 = 0.25`, right edge = 0.25 + 0.125 = 0.375. Center of leftmost bar of group 1 = `1 - width = 1 - 0.25 = 0.75`, left edge = 0.75 − 0.125 = 0.625. Gap = 0.625 − 0.375 = 0.25. So **Option B (0.25)** is the correct gap — one bar-width. **Corrected answer: B.**

</details>

---

### 27. Question

A developer creates a bar chart and wants to add a **dashed reference line** at the industry benchmark value of 0.90:

```python
fig, ax = plt.subplots()
ax.bar(models, scores, color="steelblue")
ax.axhline(y=0.90, color="red", linestyle="--", linewidth=2, label="Benchmark")
ax.legend()
```

What does `ax.axhline(y=0.90, ...)` draw?

- [ ] **A.** A vertical dashed red line at x=0.90
- [ ] **B.** A horizontal dashed red line spanning the full width of the axes at y=0.90
- [ ] **C.** A diagonal line from (0, 0.90) to (1, 0.90) in data coordinates
- [ ] **D.** A red dot at the position (0, 0.90)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A vertical line at a specific x-value is drawn by `ax.axvline(x=value, ...)`. The `axhline` function specifically creates **horizontal** lines.
- **Option B (Correct):** `ax.axhline(y=0.90)` draws a **horizontal line** at y=0.90 that spans the **full width** of the axes (from xmin=0 to xmax=1 in axes coordinates by default, i.e., the complete horizontal extent of the plot). With `linestyle="--"`, `color="red"`, and `linewidth=2`, the result is a thick dashed red reference line across the entire chart at the 90% level — clearly marking the benchmark threshold that bars should be evaluated against.
- **Option C (Incorrect):** A diagonal line from (0, 0.90) to (1, 0.90) would actually be a horizontal line (both y-coordinates are equal). But `ax.axhline` spans the full axes width in axes coordinates, not just from x=0 to x=1 in data coordinates. The description is imprecise and misleading.
- **Option D (Incorrect):** `axhline` draws a line, not a point. A single point would require `ax.scatter([0], [0.90], ...)` or `ax.plot([0], [0.90], 'o', ...)`.

</details>

---

### 28. Question

A data analyst creates a bar chart where the bars use a **hex colour per bar** but also adds `edgecolor`:

```python
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
bars = ax.bar(labels, values, color=colors, edgecolor="#2C3E50", linewidth=1.2)
```

What Python type does the variable `bars` hold after `ax.bar()` returns?

- [ ] **A.** A `list` of `Rectangle` objects
- [ ] **B.** A `BarContainer` object (which behaves like an iterable of `Rectangle` patches)
- [ ] **C.** A single `Rectangle` object representing all bars as one shape
- [ ] **D.** A `numpy.ndarray` of bar heights

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `ax.bar()` does not return a plain Python list. While a `BarContainer` is iterable and its elements are `Rectangle` patches, the container itself is a `BarContainer` object, not a `list`.
- **Option B (Correct):** `ax.bar()` returns a **`BarContainer`** — a Matplotlib container object that holds the `Rectangle` artists for each bar. `BarContainer` is iterable (you can loop over it with `for bar in bars:`), subscriptable (e.g., `bars[0]` gives the first bar), and supports the `ax.bar_label(bars, ...)` method. It also stores error bar information if `xerr` or `yerr` are provided.
- **Option C (Incorrect):** Each bar is a separate `Rectangle` artist. There is no single combined shape for all bars. Matplotlib keeps them as distinct objects so each can be styled, queried, or moved independently.
- **Option D (Incorrect):** `ax.bar()` returns a `BarContainer` of visual artists, not a NumPy array of data values. The input data (`values`) is stored separately; the return value is the graphical representation.

</details>

---

### 29. Question

A developer uses `ax.bar_label()` to automatically add value labels to bars:

```python
fig, ax = plt.subplots()
bars = ax.bar(["A", "B", "C", "D"], [0.845, 0.912, 0.878, 0.956])
ax.bar_label(bars, fmt="%.2f", padding=5)
plt.show()
```

What label text will appear above the bar for category "C" (value = 0.878)?

- [ ] **A.** `0.878`
- [ ] **B.** `0.88`
- [ ] **C.** `87.8`
- [ ] **D.** `0.878000`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `0.878` is the raw string representation with 3 decimal places. The `fmt="%.2f"` format string specifies **2** decimal places, so the value would be rounded to 0.88.
- **Option B (Correct):** `fmt="%.2f"` uses Python's old-style string formatting. `%.2f` means: format as a fixed-point decimal number with **2 digits after the decimal point**. `0.878` formatted as `%.2f` rounds to `0.88`. So the label displayed above the "C" bar is `"0.88"`.
- **Option C (Incorrect):** `87.8` would result if the value were multiplied by 100 (converting to a percentage). The `fmt` string only controls rounding and decimal places, not scaling. `0.878 × 100 = 87.8` would require a different approach.
- **Option D (Incorrect):** `0.878000` would result from `fmt="%.6f"` (6 decimal places). The `%.2f` format truncates to 2 decimal places.

</details>

---

### 30. Question

A developer creates a bar chart and calls `ax.bar_label()` but mistakenly passes a raw list instead of the `BarContainer`:

```python
values = [25, 40, 35, 50]
labels = ["W", "X", "Y", "Z"]

fig, ax = plt.subplots()
ax.bar(labels, values)   # return value NOT stored

ax.bar_label(values, fmt="%d")   # Bug: passing list, not container
```

What is the most likely error produced by this code?

- [ ] **A.** `ValueError` — `bar_label` requires a list of at least 5 elements
- [ ] **B.** `AttributeError` — a list does not have the internal attributes (`patches`, error line handles) that `bar_label` requires from a `BarContainer`
- [ ] **C.** `TypeError` — `bar_label` cannot accept integer values in the list
- [ ] **D.** No error — `bar_label` automatically searches the axes for bar artists when passed a list

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `bar_label` does not have a minimum length requirement. The error is about the type of the first argument, not its length.
- **Option B (Correct):** `ax.bar_label(container, ...)` expects a **`BarContainer`** as its first argument. Internally, `bar_label` accesses `container.patches` (the list of `Rectangle` artists) and `container.errorbar` (the error bar handles if any). A plain Python list `[25, 40, 35, 50]` does not have these attributes. When `bar_label` tries to access `values.patches`, Python raises an `AttributeError` because lists have no `patches` attribute. The fix is to store the return value: `bars = ax.bar(labels, values)` and then call `ax.bar_label(bars, ...)`.
- **Option C (Incorrect):** The list contains integers, but `bar_label` does not check the type of values inside the container. The issue is the type of the container object itself (list vs. BarContainer), not the type of individual elements.
- **Option D (Incorrect):** Matplotlib's `bar_label` does NOT search the axes for bar artists automatically. It strictly requires a `BarContainer` passed as the first argument. There is no magic discovery mechanism.

</details>
