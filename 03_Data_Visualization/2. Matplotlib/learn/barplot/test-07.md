# Matplotlib Bar Plot Objective Test — Set 07

---

### 61. Question

A data scientist creates a bar chart for model comparison and sets `width=0.9`:

```python
models = ["Baseline", "Improved", "Ensemble"]
f1 = [0.72, 0.85, 0.91]

plt.bar(models, f1, width=0.9, color="steelblue")
```

With `width=0.9`, what visual effect occurs compared to the default `width=0.8`?

- [ ] **A.** Bars appear slightly wider with a smaller visible gap between them
- [ ] **B.** Bars are 12.5% taller
- [ ] **C.** The chart overflows the figure boundary
- [ ] **D.** The bars merge into a single rectangle because 0.9 > 0.8

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Increasing `width` from 0.8 to 0.9 makes each bar slightly wider, reducing the visible gap between adjacent bars from 0.2 units to 0.1 units (since `gap = 1 − width` for unit-spaced bars). The bars appear slightly fatter with a narrower whitespace between them. At `width=1.0`, bars would be touching edge-to-edge with zero gap.
- **Option B (Incorrect):** `width` controls the horizontal dimension of bars, not their height. Bar height is determined by the second argument (the `f1` values). Changing `width` has no effect on bar height.
- **Option C (Incorrect):** `width=0.9` does not cause overflow. Each bar is still within its 1-unit interval on the x-axis. Width values up to 1.0 (and even slightly beyond) don't cause figure overflow — they just reduce or eliminate inter-bar spacing.
- **Option D (Incorrect):** At `width=0.9`, bars still have distinct positions and do not merge. They would only merge into a continuous rectangle if: (1) they shared exactly the same x-position, or (2) the width were large enough for adjacent bars to overlap (which requires width >1.0 for unit-spaced bars). At width=0.9, a 0.1-unit gap remains.

</details>

---

### 62. Question

A developer creates a grouped bar chart for 3 models × 4 metrics and wants the bar groups visually separated with some space between groups. Which approach achieves this while maintaining proper tick positioning?

- [ ] **A.** Increase the `width` argument so bars spread further apart
- [ ] **B.** Use `x = np.arange(len(metrics)) * 1.5` to space tick positions 1.5 units apart instead of 1.0 unit apart, then scale bar widths accordingly
- [ ] **C.** Add blank string categories between each metric group
- [ ] **D.** Use `plt.subplots` to create separate axes per metric

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Increasing `width` makes individual bars wider but reduces inter-bar gaps within a group. It compresses groups together rather than separating them. With very large `width`, bars within a group could overlap.
- **Option B (Correct):** Multiplying the `x` position array by a scale factor (e.g., 1.5) spreads category tick positions 1.5 units apart instead of the default 1.0 unit. With wider inter-group spacing, there's natural visual separation between metric groups. The `width` of individual bars remains small (e.g., 0.25) to keep them within their group, while the space between groups (0.5 units) creates clear visual separation. `plt.xticks(x, metrics)` (using the scaled `x`) correctly places labels at the wider positions.
- **Option C (Incorrect):** Inserting blank categories as spacers creates phantom tick positions on the x-axis. It is a hack that produces extra empty spaces but also leaves empty tick marks and potentially confusing gaps in the axis labels.
- **Option D (Incorrect):** Separate axes per metric would produce 4 separate subplots — not a grouped bar chart. This is a different chart structure altogether, useful for comparing individual metrics but not for grouped visual comparison.

</details>

---

### 63. Question

A developer creates a model comparison dashboard with the following grouped bar layout:

```python
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
lr   = [0.852, 0.841, 0.823, 0.832]
rf   = [0.921, 0.918, 0.907, 0.912]
xgb  = [0.934, 0.929, 0.921, 0.925]

x = np.arange(4)
width = 0.28

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width,     lr,  width, label="LR",  color="#3498DB")
b2 = ax.bar(x,             rf,  width, label="RF",  color="#E74C3C")
b3 = ax.bar(x + width,     xgb, width, label="XGB", color="#2ECC71")
ax.set_xticks(x, metrics)
ax.legend()
ax.set_ylim(0.7, 1.0)
```

A reviewer objects: "The y-axis starts at 0.7, which is misleading." Is the reviewer's concern valid for this type of chart?

- [ ] **A.** No — truncating the y-axis is always acceptable for model comparison charts
- [ ] **B.** Yes — the reviewer is correct; y-axis should start at 0 for all bar charts without exception
- [ ] **C.** Yes and No — starting at 0 is the ethical standard for bar charts; the truncated baseline makes the differences between LR (0.85) and XGB (0.93) appear far larger than the actual ~9% gap; however, the reviewer should suggest ylim(0,1) and note that the visual differences should match perceptual proportionality
- [ ] **D.** No — since all values are between 0.7 and 1.0, the truncation is irrelevant

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** Truncating the y-axis is NOT always acceptable. It specifically distorts bar charts because bar length is the primary visual encoding and viewers judge bars as proportional to their full height from zero.
- **Option B (Incorrect):** The "always" qualifier makes this too absolutist. For line charts, truncating axes is often acceptable (and sometimes preferred). For bar charts specifically, starting at 0 is the strong recommendation — but "without exception" is overly rigid.
- **Option C (Correct):** The concern is valid because in a bar chart, the visual length of each bar from the x-axis is the primary perceptual cue. With `ylim(0.7, 1.0)`, the LR bars (≈0.85) appear only about half as tall as the XGB bars (≈0.93), making the gap appear enormous (visually ≈50% different) when the actual percentage point difference is only ≈9%. Setting `ylim(0, 1)` provides honest proportional representation — XGB (0.93) would appear only slightly taller than LR (0.85), accurately reflecting that both are high-performing models within 10% of each other.
- **Option D (Incorrect):** The truncation IS relevant precisely BECAUSE all values are clustered between 0.7 and 1.0. The compressed y-axis window exaggerates small differences between models that are actually closely comparable.

</details>

---

### 64. Question

A developer creates a bar chart with `figsize=(12, 6)` and `tight_layout()`:

```python
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(long_category_names, values, color="coral")
ax.set_xticks(range(len(long_category_names)))
ax.set_xticklabels(long_category_names, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chart.png", dpi=150)
```

Why is `ha="right"` combined with `rotation=45`?

- [ ] **A.** `ha="right"` right-aligns the label text within the tick label bounding box, unrelated to rotation
- [ ] **B.** `ha="right"` sets the horizontal anchor point to the right end of the label text, so that after rotating 45°, the tick mark aligns with the END of the label text rather than its centre — preventing labels from appearing shifted to the right of the tick mark
- [ ] **C.** `ha="right"` rotates labels in the opposite direction from `rotation=45` for a combined 90° turn
- [ ] **D.** Without `ha="right"`, labels would overlap even with rotation

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** While technically `ha` does control horizontal alignment, the "unrelated to rotation" claim is wrong. The purpose of combining `ha="right"` with `rotation=45` is specifically to control the visual alignment of rotated labels relative to their tick marks.
- **Option B (Correct):** When labels are rotated 45°, the text pivots around its anchor point. With `ha="center"` (default), the CENTER of the label text is anchored to the tick mark position. After rotating, the text drifts to the right. With `ha="right"`, the RIGHT END of the label text is anchored to the tick mark. This means after 45° clockwise rotation, the visible end of the text (the rightmost character) points toward the tick mark, making it clear which tick each label belongs to. The combination `rotation=45, ha="right"` is a standard pattern for long x-axis labels that overlapped horizontally.
- **Option C (Incorrect):** `ha` does not add or subtract from the `rotation` angle. They are independent parameters. `ha="right"` moves the text anchor, not the rotation angle.
- **Option D (Incorrect):** Labels would overlap WITHOUT rotation — `rotation=45` reduces overlap. `ha="right"` refines the VISUAL ALIGNMENT of the rotated labels, not their overlap prevention. Labels with `rotation=45, ha="center"` would still avoid overlap (since they're rotated) but would appear misaligned relative to tick marks.

</details>

---

### 65. Question

A developer creates a bar chart and then adds a second bar chart to the SAME axes for comparison:

```python
fig, ax = plt.subplots()
bars1 = ax.bar(x - 0.2, vals_A, 0.4, label="System A", color="blue")
bars2 = ax.bar(x + 0.2, vals_B, 0.4, label="System B", color="orange")

ax.set_xticks(x)
ax.set_xticklabels(category_labels)
ax.legend()
ax.set_ylabel("Response Time (ms)")
```

The developer wants to add value labels only to `bars2` (System B) using `bar_label`. Which call is correct?

- [ ] **A.** `ax.bar_label(bars1, bars2, fmt="%.1f")`
- [ ] **B.** `ax.bar_label(bars2, fmt="%.1f", padding=4)`
- [ ] **C.** `ax.bar_label(ax.containers[1], fmt="%.1f")`
- [ ] **D.** Both B and C are valid, correct ways to label only System B bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** `ax.bar_label()` takes a single `BarContainer` as its first argument. Passing two containers `(bars1, bars2)` as two positional arguments would pass `bars2` as `fmt` — which is wrong. The function signature is `bar_label(container, labels=None, fmt=..., ...)`.
- **Option B (Correct):** `bars2` holds the `BarContainer` returned by the second `ax.bar()` call (System B). `ax.bar_label(bars2, fmt="%.1f", padding=4)` correctly targets only the System B bars, formatting each bar's height with 1 decimal place and adding 4 points of padding above each bar.
- **Option C (Correct):** `ax.containers` is a list of all container artists on the axes. `ax.containers[0]` = `bars1` (first added), `ax.containers[1]` = `bars2` (second added). Passing `ax.containers[1]` is equivalent to passing `bars2` directly — both refer to the same `BarContainer` object.
- **Option D (Correct):** Both B and C reference the same `BarContainer` object (bars2 / ax.containers[1]) and produce identical labeling. Both are valid.

</details>

---

### 66. Question

A developer creates a model evaluation bar chart with 5 models and notices that the x-axis tick labels ("Random Forest" etc.) appear cut off at the right edge. After trying `rotation=0` (horizontal labels), the text overflows. What is the BEST comprehensive fix?

- [ ] **A.** Increase the figure width: `fig, ax = plt.subplots(figsize=(14, 5))`
- [ ] **B.** Use `plt.barh()` instead to place long model names on the y-axis
- [ ] **C.** Rotate labels 45° with `ax.set_xticklabels(labels, rotation=45, ha="right")` and call `plt.tight_layout()` to adjust margins
- [ ] **D.** All of A, B, and C are valid solutions depending on context

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Valid):** Increasing figure width gives more horizontal space for labels, allowing full text with `rotation=0`. Effective for moderate label lengths with enough horizontal space.
- **Option B (Valid):** `barh` moves labels to the y-axis where they are displayed horizontally without any rotation or width constraints. Ideal for long labels and is the recommendation from data visualisation guidelines. The downside is a layout change from vertical to horizontal bars.
- **Option C (Valid):** `rotation=45` reduces horizontal overlap, and `ha="right"` aligns the rotated text correctly to its tick mark. `tight_layout()` adjusts figure margins to prevent the angled labels from being cut off by the figure boundary. This is the minimum-change fix that preserves vertical bar orientation.
- **Option D (Correct):** All three approaches solve the overflow/cutoff problem through different means. The best choice depends on context: Option A for simple fixes with wide figures, Option B for 5+ long labels (data viz best practice), Option C as a balance between space and orientation. Since all three are valid and the question says "depending on context," D is the comprehensive correct answer.

</details>

---

### 67. Question

A data engineer creates the following stacked bar chart and observes that the total bar height is incorrect for "March":

```python
months = ["January", "February", "March"]
revenue = [50000, 65000, 72000]
costs   = [30000, 40000, 45000]
profit  = [20000, 25000, 27000]

fig, ax = plt.subplots()
ax.bar(months, revenue, label="Revenue")
ax.bar(months, costs,   bottom=revenue, label="Costs")
ax.bar(months, profit,  bottom=np.array(revenue)+np.array(costs), label="Profit")
```

For "March", what is the total bar height displayed?

- [ ] **A.** 72000 (Revenue only)
- [ ] **B.** 117000 (Revenue + Costs = 72000 + 45000)
- [ ] **C.** 144000 (Revenue + Costs + Profit = 72000 + 45000 + 27000)
- [ ] **D.** 102000 (Costs + Profit = 45000 + 27000)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 72000 is the height of the Revenue segment alone. The stacked bar includes all three segments.
- **Option B (Incorrect):** 117000 = Revenue + Costs is `bottom_profit` (where the Profit layer STARTS), not the final total height.
- **Option C (Correct):** The total stacked bar height = Revenue + Costs + Profit = 72000 + 45000 + 27000 = **144000**. Revenue forms the base (0 to 72000), Costs stacks on top (72000 to 117000), and Profit adds the final segment (117000 to 144000). The bar tip reaches y=144000. Note: In a business context, a stacked Revenue + Costs + Profit chart doesn't represent meaningful accounting, but the question is purely about stacking mechanics.
- **Option D (Incorrect):** 102000 = Costs + Profit = 45000 + 27000, ignoring Revenue. This would be the bar height if Revenue were the base (not stacked) and only Costs+Profit were stacked — which is not how the code works.

</details>

---

### 68. Question

A developer wants to add gradient-like visual distinction to the bars in a chart by using different **alpha** values per bar:

```python
values    = [55, 78, 42, 91, 68]
alphas    = [0.4, 0.6, 0.5, 0.9, 0.7]
labels    = ["A", "B", "C", "D", "E"]

fig, ax = plt.subplots()
for i, (val, alpha) in enumerate(zip(values, alphas)):
    ax.bar(labels[i], val, color="royalblue", alpha=alpha)
```

What is the key **disadvantage** of this loop-based approach compared to a single `ax.bar(labels, values, ...)` call?

- [ ] **A.** Individual `ax.bar()` calls cannot be used with string labels
- [ ] **B.** Each loop iteration creates a separate `BarContainer`, producing multiple legend entries if `label=` is used — and `ax.bar_label()` cannot operate across multiple containers simultaneously
- [ ] **C.** The loop approach is syntactically invalid in Python 3
- [ ] **D.** Per-bar alpha cannot be achieved — alpha applies to the entire axes

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Individual `ax.bar()` calls with string x-positions work correctly. Matplotlib handles categorical string axes per call. This is not a limitation.
- **Option B (Correct):** When `ax.bar()` is called N times in a loop (once per bar), each call returns a separate `BarContainer` with one bar. If `label="My Series"` were used in each call, all N calls would add N legend entries for the same series — creating a cluttered legend with 5 "royalblue" entries. Additionally, `ax.bar_label()` operates on a single `BarContainer` — to label all 5 bars, you'd need to call `bar_label` 5 times (once per container) or collect all containers. A single `ax.bar(labels, values, alpha=alphas)` call with a list of alphas would avoid these issues — though note that Matplotlib does NOT directly support a list of alphas in a single `ax.bar()` call; per-bar alpha requires either the loop approach or post-hoc patch modification.
- **Option C (Incorrect):** The loop syntax is valid Python 3 — `zip(values, alphas)` and enumerate are standard Python builtins. The code runs without syntax errors.
- **Option D (Incorrect):** Per-bar alpha IS achievable through the loop approach. The limitation is on the legend and `bar_label` integration, not on the alpha application itself.

</details>

---

### 69. Question

A developer creates a bar chart comparing cross-validation F1 scores across models. The `cv_scores` are lists of 5-fold results:

```python
models = ["LR", "RF", "XGB", "NN"]
cv_scores = {
    "LR":  [0.81, 0.83, 0.82, 0.80, 0.84],
    "RF":  [0.90, 0.92, 0.91, 0.89, 0.93],
    "XGB": [0.93, 0.94, 0.92, 0.94, 0.95],
    "NN":  [0.88, 0.91, 0.89, 0.87, 0.90],
}

means = [np.mean(cv_scores[m]) for m in models]
stds  = [np.std(cv_scores[m])  for m in models]

fig, ax = plt.subplots()
bars = ax.bar(models, means, yerr=stds, capsize=6,
              color=["#3498DB","#2ECC71","#E74C3C","#9B59B6"],
              error_kw={"elinewidth": 2})
ax.set_ylim(0, 1.1)
ax.bar_label(bars, fmt="%.2f", padding=8)
```

For the "RF" model, `means[1] = np.mean([0.90, 0.92, 0.91, 0.89, 0.93])`. What is the displayed label?

- [ ] **A.** `"0.91"`
- [ ] **B.** `"0.90"`
- [ ] **C.** `"0.92"`
- [ ] **D.** `"0.910"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Mean of RF = (0.90 + 0.92 + 0.91 + 0.89 + 0.93) / 5 = 4.55 / 5 = **0.91**. Formatted with `"%.2f"`: `"0.91"`. The label displayed above the RF bar is `"0.91"`.
- **Option B (Incorrect):** `"0.90"` would require mean = 0.90 exactly. The actual mean is 0.91.
- **Option C (Incorrect):** `"0.92"` would require mean = 0.92. The actual mean of the five RF scores is 0.91.
- **Option D (Incorrect):** `"0.910"` has 3 decimal places. The format `"%.2f"` specifies 2 decimal places, giving `"0.91"` (no trailing zero beyond the 2nd decimal).

</details>

---

### 70. Question

A machine learning engineer creates a dashboard showing class-wise precision for a 5-class model:

```python
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer"]
precision   = [0.94, 0.97, 0.88, 0.82, 0.91]

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(class_names, precision,
              color=["#1ABC9C" if p >= 0.90 else "#E74C3C" for p in precision],
              width=0.6, edgecolor="#2C3E50", linewidth=0.5)
ax.axhline(0.90, color="black", linestyle="--", alpha=0.7, label="90% Target")
ax.set_ylim(0.75, 1.00)
ax.set_ylabel("Precision")
ax.legend()
ax.bar_label(bars, fmt="%.2f", padding=3)
```

Which bars appear in **red** (below 90% precision)?

- [ ] **A.** "Bird" and "Cat"
- [ ] **B.** Only "Cat"
- [ ] **C.** "Cat" and "Deer"
- [ ] **D.** "Bird", "Cat", and "Deer"

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The colour condition is `"#1ABC9C" if p >= 0.90 else "#E74C3C"`. Evaluating each class: Airplane (0.94 ≥ 0.90 → green), Automobile (0.97 ≥ 0.90 → green), Bird (0.88 < 0.90 → **red**), Cat (0.82 < 0.90 → **red**), Deer (0.91 ≥ 0.90 → green). Two bars are red: "Bird" (0.88) and "Cat" (0.82).
- **Option B (Incorrect):** Only "Cat" would be red if the threshold were somewhere between 0.82 and 0.88 (e.g., threshold=0.85). At threshold=0.90, "Bird" (0.88) also fails.
- **Option C (Incorrect):** "Deer" has precision=0.91 ≥ 0.90, so it meets the threshold and appears green. Only "Bird" and "Cat" are red.
- **Option D (Incorrect):** "Deer" (0.91) and "Bird" (0.88) — "Deer" passes the 0.90 threshold. Only "Bird" and "Cat" fail.

</details>
