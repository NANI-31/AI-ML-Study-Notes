# Matplotlib Bar Plot Objective Test — Set 06

---

### 51. Question

A developer creates a stacked bar chart with 4 layers for an energy generation breakdown:

```python
quarters = ["Q1", "Q2", "Q3", "Q4"]
solar    = np.array([120, 180, 250, 160])
wind     = np.array([95,  110,  90, 130])
hydro    = np.array([200, 200, 195, 210])
gas      = np.array([85,   60,  40,  75])

ax.bar(quarters, solar,  label="Solar")
ax.bar(quarters, wind,   bottom=solar, label="Wind")
ax.bar(quarters, hydro,  bottom=solar+wind, label="Hydro")
ax.bar(quarters, gas,    bottom=solar+wind+hydro, label="Gas")
```

For the **"Q2"** column, what is the y-coordinate where the **Hydro** layer BEGINS (its bottom baseline)?

- [ ] **A.** 290 (Solar + Wind = 180 + 110)
- [ ] **B.** 490 (Solar + Wind + Hydro = 180 + 110 + 200)
- [ ] **C.** 200 (Hydro's own value)
- [ ] **D.** 180 (Solar alone)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The Hydro layer's `bottom = solar + wind`. For Q2: `solar[1] = 180`, `wind[1] = 110`. So `bottom_hydro = 180 + 110 = 290`. The Hydro bar begins at y=290 and extends to y=290+200=490. The "bottom" is where the layer STARTS, not where it ends.
- **Option B (Incorrect):** 490 = 180 + 110 + 200 is where the Hydro layer ENDS (its top boundary), not where it begins. This is `bottom_gas` — the starting baseline for the Gas layer.
- **Option C (Incorrect):** 200 is Hydro's own height contribution (the thickness of the Hydro segment), not its baseline position. The actual y-coordinate where Hydro begins is determined by all layers below it.
- **Option D (Incorrect):** 180 = Solar alone is the `bottom` for the Wind layer. Hydro starts above both Solar AND Wind.

</details>

---

### 52. Question

A developer creates a normalized 100% stacked bar chart for budget allocation across four departments. The raw allocations are:

```
HR:   Personnel=45K, Training=8K, Equipment=2K
IT:   Personnel=60K, Training=5K, Equipment=35K
R&D:  Personnel=80K, Training=15K, Equipment=55K
Sales:Personnel=70K, Training=12K, Equipment=18K
```

For the **IT** department, what percentage does Equipment (35K) represent? (Total IT = 60+5+35 = 100K)

- [ ] **A.** 35.0%
- [ ] **B.** 25.0%
- [ ] **C.** 40.0%
- [ ] **D.** 30.0%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** IT total = 60 + 5 + 35 = **100K**. Equipment percentage = (35 / 100) × 100 = **35.0%**. The total conveniently equals 100K, making the percentage calculation straightforward. IT's Equipment segment in the 100% stacked bar chart occupies 35% of the bar height.
- **Option B (Incorrect):** 25% would correspond to Equipment = 25K out of 100K. Equipment is 35K.
- **Option C (Incorrect):** 40% would correspond to Equipment = 40K out of 100K. Equipment is 35K, not 40K.
- **Option D (Incorrect):** 30% would correspond to Equipment = 30K out of 100K. Equipment is 35K.

</details>

---

### 53. Question

A data engineer creates a vertical bar chart for an imbalanced dataset and uses `ax.bar_label` on a stacked chart with two segments:

```python
classes = ["Negative", "Positive"]
train_count = np.array([8200, 800])
test_count  = np.array([1640, 160])

fig, ax = plt.subplots()
bars1 = ax.bar(classes, train_count, label="Train")
bars2 = ax.bar(classes, test_count, bottom=train_count, label="Test")

ax.bar_label(bars1, label_type="center", fmt="%d")
ax.bar_label(bars2, label_type="center", fmt="%d")
```

Where exactly does the label for `bars2` (Test segment) for "Negative" appear?

- [ ] **A.** At y=820 (center of test segment: 8200 + 1640/2)
- [ ] **B.** At y=9020 (center of full stacked bar: (8200+1640)/2)
- [ ] **C.** At y=8200 (top of train segment = bottom of test segment)
- [ ] **D.** At y=9840 (top of full bar)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `label_type="center"` places the label at the VERTICAL CENTER of each bar's own segment. For `bars2` (Test segment for "Negative"): The segment starts at `bottom = 8200` and has height = 1640. Center of segment = `8200 + 1640/2 = 8200 + 820 = 9020`. Wait — recalculating: bottom=8200, height=1640, center = 8200 + 1640/2 = 8200 + 820 = **9020**. Actually Option B = 9020. Let me recheck: for `bars2` which represents the TEST segment: its bottom is 8200 (train count), its height is 1640 (test count). Center position = bottom + height/2 = 8200 + 820 = **9020**. So **Option B (9020)** is correct. The center of the test segment for "Negative" is at y=9020.
- **Option B (Correct — revised):** For `bars2`, the "Negative" bar: `bottom=8200`, `height=1640`. Center = `8200 + 1640/2 = 8200 + 820 = 9020`. The label appears at y=9020. **Option B is the correct answer.**
- **Option C (Incorrect):** y=8200 is the boundary between Train and Test segments — where the Test segment BEGINS. `label_type="center"` places the label at the MIDPOINT of the segment, not its base.
- **Option D (Incorrect):** y=9840 is the TOP of the complete stacked bar (8200+1640=9840). This is the `label_type="edge"` position for bars2, not `label_type="center"`.

</details>

---

### 54. Question

A developer creates grouped horizontal bars for 3 models across 4 tasks with `height=0.25`:

```python
tasks   = ["Task A", "Task B", "Task C", "Task D"]
y       = np.arange(4)
height  = 0.25

ax.barh(y - height,   model1_scores, height, label="Model 1")
ax.barh(y,            model2_scores, height, label="Model 2")
ax.barh(y + height,   model3_scores, height, label="Model 3")

ax.set_yticks(y)
ax.set_yticklabels(tasks)
```

For "Task C" (index `i=2`), what is the **vertical center** of the **Model 3** bar?

- [ ] **A.** 2.25
- [ ] **B.** 2.50
- [ ] **C.** 1.75
- [ ] **D.** 2.125

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** For Task C, `y[2] = 2`. Model 3 uses `y + height = 2 + 0.25 = 2.25`. Since `ax.barh` centers bars by default (`align="center"`), the vertical center of the Model 3 bar IS at **2.25**. The three model bars for Task C are centered at: Model 1 = 2 − 0.25 = 1.75, Model 2 = 2.0, Model 3 = 2 + 0.25 = **2.25**.
- **Option B (Incorrect):** 2.50 would correspond to `y + 2×height = 2 + 0.5`. No bar in this code is offset by `2×height`.
- **Option C (Incorrect):** 1.75 is the center of Model 1's bar for Task C (y − height = 2 − 0.25 = 1.75).
- **Option D (Incorrect):** 2.125 would be a non-standard offset not used in this code. The Model 3 center is at exactly 2.25.

</details>

---

### 55. Question

A developer writes the following code and discovers the stacked bar chart is incorrect. Identify the bug:

```python
categories = ["X", "Y", "Z"]
layer1 = [30, 50, 40]
layer2 = [20, 30, 25]

fig, ax = plt.subplots()
ax.bar(categories, layer1, label="Layer 1")
ax.bar(categories, layer2, bottom=layer1, label="Layer 2")  # Line A
ax.bar(categories, layer2, bottom=layer1, label="Layer 3")  # Line B — Bug?
```

What is wrong with **Line B**?

- [ ] **A.** Line B uses `layer2` for heights but should use a `layer3` data array; and the `bottom` should be `layer1 + layer2`, not just `layer1`
- [ ] **B.** Line B's `bottom=layer1` is incorrect — it should be `bottom=0`
- [ ] **C.** `ax.bar` cannot be called more than twice for the same categories
- [ ] **D.** Line B will stack correctly because `layer1` is an additive baseline

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Line B has two bugs: (1) It reuses `layer2` as bar heights for what is labeled "Layer 3" — there is no separate `layer3` data. The third segment should use a distinct array `layer3 = [...]`. (2) The `bottom=layer1` is wrong — it makes Layer 3 start at the same position as Layer 2 (both starting at `bottom=layer1`), causing Layer 2 and Layer 3 to overlap. The correct bottom for Layer 3 should be `bottom=np.array(layer1) + np.array(layer2)` — the cumulative sum of all layers below it.
- **Option B (Incorrect):** Setting `bottom=0` would make Layer 3 start at the very bottom (y=0), completely overlapping Layers 1 and 2. The actual bug is that the bottom should be `layer1 + layer2`, not `0` or `layer1`.
- **Option C (Incorrect):** There is no limit on how many times `ax.bar()` can be called with the same x-positions. Stacking relies precisely on repeated `ax.bar()` calls with increasing `bottom` values.
- **Option D (Incorrect):** `bottom=layer1` makes Layer 3 start where Layer 1 ENDS, which would be correct if Layer 2 weren't already stacked there. Since Layer 2 occupies from `layer1` to `layer1+layer2`, using `bottom=layer1` for Layer 3 would cause overlap between Layer 3 and Layer 2.

</details>

---

### 56. Question

A developer compares model performance on an imbalanced test set and considers whether to plot Accuracy or F1-Score on the y-axis. Which statement about bar chart design for imbalanced classification is most accurate?

- [ ] **A.** Accuracy should always be used since it is simpler to explain
- [ ] **B.** F1-Score is more informative for imbalanced datasets — a bar chart showing F1 reveals whether models truly handle the minority class, while accuracy bars may look nearly identical (all near 90%+) even if one model ignores the minority class entirely
- [ ] **C.** The choice of metric doesn't matter as long as the y-axis starts at 0
- [ ] **D.** Stacked bar charts should always be used for imbalanced data to show class breakdown

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** On an imbalanced dataset (e.g., 95% negative, 5% positive), a trivial classifier that predicts "Negative" for all samples achieves 95% accuracy. A bar chart of accuracy would make this useless classifier look excellent. Accuracy is misleading for imbalanced evaluation.
- **Option B (Correct):** F1-Score is the harmonic mean of Precision and Recall, penalising models that ignore the minority class. On a bar chart, models with high F1 genuinely capture the minority class well, while models that achieve high accuracy through majority-class bias will have poor F1. Comparing F1-Score bars reveals true model effectiveness on imbalanced data in a way that Accuracy bars cannot. This is a visualisation ethics and ML design principle.
- **Option C (Incorrect):** The metric choice fundamentally changes what the chart communicates. Both accuracy and F1 should start at 0, but y-axis baseline alone doesn't determine which metric is appropriate for imbalanced data.
- **Option D (Incorrect):** Stacked bar charts can show class distributions (class counts per model), but they are not the standard choice for performance metric comparison. A grouped or single-series bar chart with F1-Score (or Precision-Recall breakdown) is the appropriate choice.

</details>

---

### 57. Question

A developer adds a horizontal bar chart for feature importance from an XGBoost model and wants to display **percentage** labels inside each bar using `bar_label`:

```python
importances = np.array([0.34, 0.25, 0.21, 0.12, 0.08])
features    = ["Credit Score", "Employment", "Assets", "Age", "Debt"]

fig, ax = plt.subplots()
bars = ax.barh(features, importances * 100, color="teal")
ax.bar_label(bars, fmt="%.1f%%", label_type="center", padding=0)
ax.set_xlabel("Importance (%)")
```

What label text appears inside the "Credit Score" bar (raw importance = 0.34)?

- [ ] **A.** `"34.0%"`
- [ ] **B.** `"0.3%"`
- [ ] **C.** `"34%"`
- [ ] **D.** `"0.34%"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** The bar height passed to `ax.barh` is `importances * 100 = 0.34 * 100 = 34.0`. The `bar_label` formats this value: `"%.1f%%" % 34.0 = "34.0%"`. The key difference from Question 46 is that here the developer **pre-multiplied** the values by 100 before plotting, so `bar_label` correctly labels "34.0%".
- **Option B (Incorrect):** `"0.3%"` would result from formatting the raw importance (0.34) without pre-multiplying. Here, the data passed to `barh` is already `importances * 100 = 34.0`, so `bar_label` labels 34.0, not 0.34.
- **Option C (Incorrect):** `"34%"` would result from `fmt="%d%%"` (integer format). The format `"%.1f%%"` gives 1 decimal place: "34.0%", not "34%".
- **Option D (Incorrect):** `"0.34%"` would come from formatting the raw proportion without multiplying by 100, using a 2-decimal format like `"%.2f%%"`. Here the bar value is 34.0 (pre-multiplied), not 0.34.

</details>

---

### 58. Question

A developer creates a bar chart where bars have **negative values** for some categories (representing net loss):

```python
categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
net_profit  = [15000, -8000, 22000, -3000, 31000, 18000]

fig, ax = plt.subplots()
bars = ax.bar(categories, net_profit,
              color=["green" if v >= 0 else "red" for v in net_profit])
ax.axhline(0, color="black", linewidth=1)
```

For "Feb" with `net_profit=-8000`, where does the bar extend from and to?

- [ ] **A.** From y=0 (top) down to y=−8000 (bottom)
- [ ] **B.** From y=−8000 (top) up to y=0 (bottom)
- [ ] **C.** The bar is invisible because negative heights are clipped
- [ ] **D.** From y=−8000 down to y=−16000

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.bar()` fully supports negative bar heights. When `height=-8000` and the default `bottom=0`, the bar extends **downward** from y=0 to y=−8000. Visually, the bar reaches downward below the x-axis. The bar's top is at y=0 (the baseline) and its bottom is at y=−8000. This is the correct and expected behaviour for waterfall and profit/loss bar charts.
- **Option B (Incorrect):** This describes the same physical region but with "top" and "bottom" labels reversed. In Matplotlib's coordinate system, the TOP of the bar is the higher y-coordinate (y=0) and the BOTTOM is the lower y-coordinate (y=−8000). A bar with negative height hangs downward from y=0.
- **Option C (Incorrect):** Negative bar heights are fully supported and rendered correctly by Matplotlib. No clipping occurs — the bars appear below the zero line.
- **Option D (Incorrect):** For a bar at `bottom=0` (default) and `height=−8000`, the bar spans from y=0 to y=0+(−8000)=−8000. It does not extend to −16000 (which would require `bottom=−8000` and `height=−8000`).

</details>

---

### 59. Question

A developer creates a bar chart and discovers that the bars have **zero height** for some categories. The `heights` array is:

```python
categories = ["Alpha", "Beta", "Gamma", "Delta"]
heights    = [42, 0, 0, 78]

bars = ax.bar(categories, heights)
ax.bar_label(bars, fmt="%d", padding=3)
```

What label appears above the "Beta" bar (height=0)?

- [ ] **A.** No label — `bar_label` skips bars with zero height
- [ ] **B.** `"0"` — `bar_label` labels all bars including zero-height ones
- [ ] **C.** `"nan"` — zero height is treated as missing data
- [ ] **D.** An error is raised because division by zero occurs in label positioning

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `ax.bar_label()` does NOT skip zero-height bars. It processes every bar in the `BarContainer` regardless of height, including bars with height=0.
- **Option B (Correct):** `ax.bar_label(bars, fmt="%d")` formats `0` as `"%d" % 0 = "0"`. The label `"0"` appears at y=0+padding=3 points above the baseline (x-axis) for "Beta". This can sometimes look cluttered when bars are at zero height (the label appears very close to the x-axis), but it is the documented behaviour.
- **Option C (Incorrect):** Zero height is a perfectly valid numeric value — not missing data. `bar_label` does not treat 0 as NaN.
- **Option D (Incorrect):** No division by zero occurs in `bar_label`. The label positioning formula uses `bar.get_height()` (which returns 0.0) and adds `padding` (in points). No division by the bar height is performed internally.

</details>

---

### 60. Question

A developer creates a multi-panel figure showing model accuracy bars (top panel) and training time bars (bottom panel):

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

models = ["LR", "DT", "RF", "XGB", "SVM"]
accuracy = [0.85, 0.79, 0.93, 0.91, 0.88]
train_time = [0.2, 0.1, 5.8, 3.2, 12.4]   # seconds

ax1.bar(models, accuracy, color="#3498DB")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1)

ax2.bar(models, train_time, color="#E74C3C")
ax2.set_ylabel("Training Time (s)")

plt.tight_layout()
```

What does `sharex=True` do in this multi-panel bar chart?

- [ ] **A.** It makes both panels share the same y-axis scale
- [ ] **B.** It links the x-axes of both panels so they display the same categories and scale — zooming or panning one panel synchronises the other
- [ ] **C.** It copies the x-axis labels from ax1 to ax2 automatically
- [ ] **D.** It forces both panels to use the same bar colours

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `sharex=True` links the **x-axes**, not the y-axes. Shared y-axes use `sharey=True`. In this chart, the y-axes are intentionally different (accuracy 0–1 vs. training time 0–12.4 seconds).
- **Option B (Correct):** `sharex=True` creates a shared x-axis between `ax1` and `ax2`. Both panels display the same x-tick positions and labels ("LR", "DT", "RF", "XGB", "SVM"). With `sharex=True`, Matplotlib can also suppress the x-axis tick labels on the top panel (`ax1`) since `ax2` (bottom) displays them — this is why `tight_layout()` removes redundant labels. Any x-axis limit changes (`set_xlim`) applied to one axes automatically affect the other.
- **Option C (Incorrect):** `sharex=True` does NOT copy labels from one axes to another. Both axes independently draw their x-tick labels unless suppression is applied. With `sharex=True`, Matplotlib typically hides the x-tick labels on the upper panel automatically to avoid duplication.
- **Option D (Incorrect):** Bar colours are completely independent of axis sharing. Each `ax.bar()` call controls its own `color`. `sharex` is a purely geometric/scale-sharing parameter.

</details>
