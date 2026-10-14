# Matplotlib Imshow & Heatmaps — Test 07
## Annotated Heatmaps & Cell Text Formatting

---

### Question 61

A developer writes cell text annotations for a 3×4 confusion matrix and produces a bug:

```python
cm = np.array([[50, 3, 1, 0],
               [2, 45, 4, 1],
               [0, 2, 48, 3]])

ax.imshow(cm, cmap='Blues', interpolation='nearest')

# Developer's loop — BUG HERE
for i in range(3):       # rows
    for j in range(4):   # cols
        ax.text(i, j, cm[i, j], ha='center', va='center')
```

What bug exists in the `ax.text()` call, and what is the correct fix?

- [ ] A) `cm[i, j]` should be `cm[j, i]` — the array is accessed in wrong order
- [ ] B) `ax.text(i, j, ...)` passes `(row, col)` but `ax.text` expects `(x, y)` = `(col, row)` — the text labels are transposed across the diagonal. Fix: `ax.text(j, i, cm[i, j], ...)`
- [ ] C) `ha='center'` is invalid for numeric text — use `ha='right'`
- [ ] D) `ax.text` requires float coordinates — wrap with `ax.text(float(i), float(j), ...)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.text(i, j, ...)` passes `(row, col)` but `ax.text` expects `(x, y)` = `(col, row)` — the text labels are transposed across the diagonal. Fix: `ax.text(j, i, cm[i, j], ...)`

**Explanation:**

- **A is incorrect:** `cm[i, j]` accesses the correct matrix value — row i, column j. The array indexing is correct. The bug is in the coordinate arguments to `ax.text()`, not in the array access.
- **B is correct:** `ax.text(x, y, text)` uses **display coordinates** where `x` is the horizontal axis (columns) and `y` is the vertical axis (rows). So the correct call is `ax.text(j, i, cm[i, j], ...)` — x=j (column), y=i (row). The bug `ax.text(i, j, ...)` passes i (row) as x and j (column) as y — transposing every label. For a non-square matrix or any matrix where i≠j for most cells, this causes all text to appear in the wrong cell, shifted across the diagonal.
- **C is incorrect:** `ha='center'` (horizontal alignment) is valid for any text content including numeric values. It centers the text horizontally at the specified x coordinate.
- **D is incorrect:** `ax.text()` accepts integer, float, or numpy scalar coordinates. No explicit float conversion is needed.

</details>

---

### Question 62

A developer creates an annotated heatmap of a 5×5 weight matrix and wants **dynamic text color** — white text on dark background cells, black text on light background cells. What is the standard threshold approach?

```python
weights = np.random.randn(5, 5)
im = ax.imshow(weights, cmap='RdBu', vmin=-2, vmax=2)

for i in range(5):
    for j in range(5):
        # Dynamic text contrast
        val = weights[i, j]
        # What threshold logic goes here?
        text_color = ???
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                color=text_color, fontsize=9)
```

Which threshold logic correctly determines text color based on the background?

- [ ] A) `text_color = "white" if val > 0 else "black"` — white for positive values, black for negative
- [ ] B) `norm_val = (val - (-2)) / (2 - (-2)); text_color = "white" if norm_val < 0.5 else "black"` — normalize the value and threshold at the colormap midpoint
- [ ] C) `text_color = "white" if abs(val) > 1.0 else "black"` — white for large-magnitude values
- [ ] D) `text_color = "black"` — always use black for maximum contrast on all colormaps

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `norm_val = (val - (-2)) / (2 - (-2)); text_color = "white" if norm_val < 0.5 else "black"` — normalize the value and threshold at the colormap midpoint

**Explanation:**

- **A is incorrect:** Thresholding on sign (`val > 0`) only works for symmetric diverging colormaps where 0 is the midpoint. It fails for sequential colormaps or asymmetric ranges. Also, for `RdBu`, dark blue (negative extreme) requires white text, but dark red (positive extreme) also requires white text — sign-based logic gives black text for dark red, which is unreadable.
- **B is correct:** The correct approach is to normalize the value to [0, 1] using the colormap's range: `norm_val = (val - vmin) / (vmax - vmin)`. Then threshold at 0.5 (the colormap midpoint): `norm_val < 0.5` → the background is in the first half of the colormap (dark blue for `RdBu`) → use **white** text; `norm_val >= 0.5` → background is in the second half → use **black** text. This is colormap-aware contrast control that generalizes to any sequential or diverging colormap.
- **C is incorrect:** `abs(val) > 1.0` catches both large positive and large negative values (dark red AND dark blue), which is better than A but misses the asymmetry for `RdBu` where mid-range colors have different luminance.
- **D is incorrect:** Always using black text fails on dark background cells (dark blue for `val = -2`). Black text on a very dark blue background is nearly invisible.

</details>

---

### Question 63

A developer annotates a 4×4 heatmap with formatted text. What does the format string `f"{val:.1%}"` produce for a value of `0.8745`?

- [ ] A) `"0.8745"`
- [ ] B) `"87.5%"`
- [ ] C) `"87.4%"` — rounded down to 1 decimal place as a percentage
- [ ] D) `"0.87%"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `"87.5%"`

**Explanation:**

- **A is incorrect:** `:.1%` applies percentage formatting — it multiplies by 100 and adds a `%` sign. It does not produce the raw decimal `"0.8745"`.
- **B is correct:** The `%` format specifier in Python f-strings multiplies the value by 100 and appends a `%` character. `:.1%` additionally rounds to 1 decimal place. For `0.8745`: `0.8745 × 100 = 87.45` → rounded to 1 decimal = `87.5` → formatted as `"87.5%"`. This is the standard format for percentage recall values in confusion matrix annotations.
- **C is incorrect:** `87.4%` would be the result if rounding down (floor) were used. Python's default rounding is **round-half-to-even** (banker's rounding), and `87.45` rounds to `87.5` (since 4 rounds toward even — 5 in the tenths place from 87.45 → 87.5 is correct standard rounding).
- **D is incorrect:** `"0.87%"` would result from `:.2%` formatting without the ×100 multiplication having occurred — but `%` always multiplies by 100. The correct output for `:.1%` with value 0.8745 is `"87.5%"`.

</details>

---

### Question 64

A developer creates an annotated heatmap for a 10×10 similarity matrix and notices all text labels are outside the cells — each label appears diagonally offset from its cell center. What is the most likely cause?

```python
for i in range(10):
    for j in range(10):
        ax.text(j + 0.5, i + 0.5, f"{sim[i,j]:.2f}",
                ha='center', va='center')
```

- [ ] A) The `+ 0.5` offset is wrong — `ax.imshow` centers cells at integer coordinates (0, 1, 2...), so `ax.text(j, i, ...)` without any offset places text at cell centers
- [ ] B) `ha='center'` and `va='center'` are conflicting alignment options — use only one
- [ ] C) The format string `:.2f` cannot be used inside `ax.text()` for image annotations
- [ ] D) `sim[i, j]` should be `sim[j, i]` — the offset causes the diagonal shift

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The `+ 0.5` offset is wrong — `ax.imshow` centers cells at integer coordinates (0, 1, 2...), so `ax.text(j, i, ...)` without any offset places text at cell centers

**Explanation:**

- **A is correct:** `ax.imshow()` centers each cell at its integer coordinate. Cell `[i, j]` is centered at coordinate `(x=j, y=i)`. Adding `+0.5` to both coordinates shifts the text to the cell's **right and lower edge** (or upper, depending on origin) — not its center. The correct call is simply `ax.text(j, i, ...)` with no offset. The `+ 0.5` offset is a common mistake made by developers who assume `imshow` uses the top-left corner of cells as the origin rather than the center.
- **B is incorrect:** `ha='center'` (horizontal alignment) and `va='center'` (vertical alignment) are complementary, not conflicting. Together they center the text both horizontally and vertically at the specified coordinate point — exactly what is needed.
- **C is incorrect:** F-string format specifiers work perfectly inside `ax.text()`. `f"{sim[i,j]:.2f}"` produces a formatted string like `"0.85"` — a standard Python string. `ax.text()` accepts any string argument.
- **D is incorrect:** Reversing `i,j` to `j,i` in `sim[j,i]` would transpose the displayed values, but this is the data indexing — not the cause of the spatial offset. The offset is caused by the `+0.5` coordinate error.

</details>

---

### Question 65

A developer creates an annotated heatmap and uses `fontweight='bold'` for certain cells to highlight them:

```python
for i in range(rows):
    for j in range(cols):
        weight = 'bold' if data[i,j] == data.max() else 'normal'
        ax.text(j, i, f"{data[i,j]:.1f}",
                ha='center', va='center',
                fontsize=8, fontweight=weight)
```

The `data.max()` call inside the loop is repeatedly evaluated. What is the correct optimization?

- [ ] A) Replace `data.max()` with `max(data)` — the built-in is faster for 2D arrays
- [ ] B) Compute `max_val = data.max()` before the loop and compare `data[i,j] == max_val` inside — avoiding repeated `data.max()` calls which recompute the maximum every iteration
- [ ] C) Use `np.argmax(data)` instead of `data.max()` for efficiency
- [ ] D) The repeated `data.max()` calls are negligible — no optimization is needed

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Compute `max_val = data.max()` before the loop and compare `data[i,j] == max_val` inside — avoiding repeated `data.max()` calls which recompute the maximum every iteration

**Explanation:**

- **A is incorrect:** `max(data)` for a 2D NumPy array iterates over rows and returns the row with the maximum value — not a scalar maximum. This is incorrect behavior. `max(data)` and `data.max()` behave differently for 2D arrays.
- **B is correct:** `data.max()` scans the entire array every call — O(n²) work per loop iteration. With `rows × cols` iterations (e.g., 10×10 = 100 iterations), this wastes 100× the computation needed. The optimization is straightforward: `max_val = data.max()` computed once before the loop (O(n²) total), then `data[i,j] == max_val` is an O(1) comparison inside the loop. For large matrices this is a meaningful efficiency gain.
- **C is incorrect:** `np.argmax(data)` returns the **flat index** of the maximum value, not the maximum value itself. Using it for comparison would require further manipulation (`np.unravel_index`) and doesn't directly help with the repeated-call inefficiency.
- **D is incorrect:** For large matrices (e.g., 100×100 = 10,000 iterations, each calling `data.max()` on a 100×100 array = 10,000 comparisons per call), the total extra work is 10,000 × 10,000 = 100,000,000 comparisons — non-negligible. The optimization is meaningful for production code.

</details>

---

### Question 66

A developer annotates a heatmap and wants to display values **larger than 1000** in scientific notation (e.g., `1.2e+03`) and values below 1000 in standard decimal format:

```python
for i in range(rows):
    for j in range(cols):
        val = data[i, j]
        # Choose format
        label = ???
        ax.text(j, i, label, ha='center', va='center', fontsize=7)
```

Which formatting logic is correct?

- [ ] A) `label = f"{val:.2e}" if val > 1000 else f"{val:.0f}"`
- [ ] B) `label = f"{val:.2g}"` — `g` format automatically switches between fixed and scientific
- [ ] C) `label = str(round(val, 2))`
- [ ] D) Both A and B are correct approaches, each with different tradeoffs

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Both A and B are correct approaches, each with different tradeoffs

**Explanation:**

- **A is correct (as part of D):** `f"{val:.2e}"` explicitly formats as scientific notation (`1.23e+03`), and `f"{val:.0f}"` formats as integer-rounded decimal. The conditional `if val > 1000` gives explicit control over the threshold.
- **B is correct (as part of D):** The `:.2g` format specifier automatically chooses between fixed-point (`f`) and scientific (`e`) notation based on the magnitude — using `e` notation when the exponent is < -4 or ≥ the precision. For `val=1200, :.2g` → `1.2e+03`; for `val=85, :.2g` → `85`. The `.2g` approach is more compact but the threshold is automatic (not user-controlled).
- **C is incorrect:** `str(round(val, 2))` converts to string but may produce unwanted trailing zeros or very long decimal representations for large numbers (e.g., `1234.56789` → `"1234.57"`). For values > 1000, this produces standard decimal — not the desired scientific notation.
- **D is correct:** Both A and B solve the problem — A gives explicit threshold control, B uses Python's built-in smart format switching. The choice depends on whether the developer wants fine-grained threshold control (A) or automatic behavior (B).

</details>

---

### Question 67

A developer creates a custom heatmap annotation function that adds cell borders to highlight specific cells:

```python
import matplotlib.patches as patches

for i in range(rows):
    for j in range(cols):
        if highlight_mask[i, j]:
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5),   # lower-left corner
                1.0, 1.0,              # width, height
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
```

What does `(j - 0.5, i - 0.5)` represent for the `Rectangle` lower-left corner?

- [ ] A) The center of cell [i, j] offset by -0.5 in both directions — correct for aligning the 1×1 rectangle exactly around the cell
- [ ] B) The top-left corner of the cell — wrong; should be `(j, i - 1)`
- [ ] C) The lower-left corner in pixel coordinates — needs conversion to data coordinates
- [ ] D) Only correct for `origin='lower'` — for `origin='upper'`, use `(j - 0.5, i + 0.5)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The center of cell [i, j] offset by -0.5 in both directions — correct for aligning the 1×1 rectangle exactly around the cell

**Explanation:**

- **A is correct:** `ax.imshow()` centers cell `[i, j]` at data coordinate `(x=j, y=i)`. A 1×1 cell extends from `(j-0.5, i-0.5)` to `(j+0.5, i+0.5)`. `patches.Rectangle((j-0.5, i-0.5), 1.0, 1.0, ...)` creates a rectangle with: lower-left corner at `(j-0.5, i-0.5)`, width=1.0, height=1.0. This places the rectangle exactly around the cell with a red border outline. The `facecolor='none'` makes it transparent so the heatmap color shows through.
- **B is incorrect:** `(j, i - 1)` would not correctly align the rectangle. The correct lower-left is `(j-0.5, i-0.5)` — half a cell width/height offset from the cell center.
- **C is incorrect:** `ax.add_patch()` uses **data coordinates** by default (`transform=ax.transData`). The coordinates `(j-0.5, i-0.5)` are already in data space — no pixel conversion is needed.
- **D is incorrect:** The rectangle coordinates use the same data space as `imshow` regardless of `origin`. The `origin` parameter affects how array row indices map to y-coordinates in the axes — but once mapped, the coordinate system is the same. `(j-0.5, i-0.5)` correctly places the rectangle around cell `[i,j]` for both `origin='upper'` and `origin='lower'`.

</details>

---

### Question 68

A developer annotates a heatmap with percentage values and uses `va='baseline'` instead of `va='center'`. What visual effect does this produce?

- [ ] A) No visible difference — `'baseline'` and `'center'` are equivalent for numeric text
- [ ] B) The text baseline (bottom of most characters) aligns with the specified y-coordinate — text appears shifted upward relative to cell center because descenders are not accounted for
- [ ] C) `va='baseline'` raises `ValueError` — only `'top'`, `'center'`, and `'bottom'` are valid
- [ ] D) The text aligns with the top of the cell rather than the center

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) The text baseline (bottom of most characters) aligns with the specified y-coordinate — text appears shifted upward relative to cell center because descenders are not accounted for

**Explanation:**

- **A is incorrect:** `'baseline'` and `'center'` produce visually different results. `'center'` vertically centers the text bounding box at the specified y-coordinate. `'baseline'` places the text baseline (the bottom of characters like A, B, C — not descenders like g, p, y) at the y-coordinate.
- **B is correct:** With `va='baseline'`, the text is positioned so its **typographic baseline** aligns with the specified y-coordinate. Since the cell center is the specified y, and the baseline is the bottom of the character body (above descenders but below ascenders), the text visually appears shifted upward from center. The visible text body sits above the y-coordinate. For heatmap annotations where cells must look centered, `va='center'` is always preferred.
- **C is incorrect:** `va='baseline'` is a valid `verticalalignment` option in Matplotlib alongside `'top'`, `'center'`, `'bottom'`, `'center_baseline'`. No `ValueError` is raised.
- **D is incorrect:** `va='top'` would place the top of the text bounding box at the y-coordinate. `va='baseline'` places the typographic baseline at y — the text body sits above, not at the top.

</details>

---

### Question 69

A developer creates a production-grade annotated heatmap function and wants to automatically skip annotation for cells where `data[i, j] == 0` (to reduce clutter in sparse matrices):

```python
for i in range(rows):
    for j in range(cols):
        if data[i, j] != 0:
            ax.text(j, i, f"{data[i,j]}", ha='center', va='center', fontsize=8)
```

A colleague suggests using `if data[i, j]:` instead of `if data[i, j] != 0:`. Are they equivalent for all cases?

- [ ] A) Yes — `if data[i, j]:` is equivalent to `if data[i, j] != 0:` for all numeric types
- [ ] B) No — `if data[i, j]:` is falsy for 0, 0.0, NaN is truthy (non-zero but not equal to any value). `!= 0` is more explicit and handles NaN differently (NaN != 0 is True, so NaN cells would still be annotated)
- [ ] C) No — `if data[i, j]:` raises `TypeError` for NumPy scalar values
- [ ] D) Yes for integers; no for floats — float `0.0` evaluates as `True` in Python

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) No — `if data[i, j]:` is falsy for 0, 0.0, NaN is truthy (non-zero but not equal to any value). `!= 0` is more explicit and handles NaN differently (NaN != 0 is True, so NaN cells would still be annotated)

**Explanation:**

- **A is incorrect:** The behavior diverges for NaN (Not a Number) values. `if np.nan:` is truthy (NaN is treated as a non-zero float), so NaN cells WOULD be annotated. `np.nan != 0` is also True, so both would annotate NaN cells — but their behavior for other edge cases (None, complex zeros) differs.
- **B is correct:** For standard numeric data: `if val:` and `if val != 0:` are equivalent for integers and floats (0 → falsy, non-zero → truthy). The critical divergence is **NaN**: `float('nan') != 0` evaluates to `True` (NaN is not equal to 0), so `!= 0` annotates NaN cells. `if float('nan'):` is also truthy (NaN is non-zero). In practice they behave the same for NaN — but the explicit `!= 0` is clearer in intent and avoids Pythonic truthiness ambiguities. The colleague's suggestion is functionally equivalent for most practical data but less explicit.
- **C is incorrect:** `if numpy_scalar:` is valid Python — NumPy scalars implement `__bool__` and evaluate to True/False based on their numeric value. No `TypeError` is raised.
- **D is incorrect:** `float(0.0)` in Python evaluates to `False` (falsy). `if 0.0:` is `False`. Both floats and integers follow the same truthiness rule: zero is falsy, non-zero is truthy.

</details>

---

### Question 70

A developer adds text annotations to every cell of a 20×20 heatmap with `fontsize=9`. The text is overlapping and unreadable. Without changing the data, which approach best resolves this?

- [ ] A) Increase `figsize` so each cell is physically larger, and reduce `fontsize` to fit within cells
- [ ] B) Use `fontsize=6` and `fontweight='bold'` — smaller bold text fits in smaller cells
- [ ] C) Annotate only cells where `abs(data[i,j]) > threshold` — selectively annotate only high-magnitude cells to reduce clutter
- [ ] D) All of A, B, and C are valid approaches depending on the use case

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) All of A, B, and C are valid approaches depending on the use case

**Explanation:**

- **A is correct (as part of D):** Increasing `figsize` gives each cell more physical display space, allowing larger text. Reducing `fontsize` simultaneously ensures text fits within the larger cells. This is appropriate when the visualization must show all values.
- **B is correct (as part of D):** `fontsize=6` with `fontweight='bold'` makes text smaller but more legible at reduced sizes. Appropriate for dense matrices where all values must be shown.
- **C is correct (as part of D):** Selective annotation (only cells above a threshold) reduces clutter while preserving the most important information. This is the standard approach for confusion matrices — annotating all cells including near-zero off-diagonal values adds noise without insight.
- **D is correct:** All three strategies are legitimate and commonly used in production heatmaps. The choice depends on: (A) when all values are equally important and display space is available; (B) when the figure size is fixed; (C) when sparse annotation of key values is the goal (e.g., highlighting high-confusion cell pairs in a multi-class confusion matrix).

</details>
