# Matplotlib Histogram Objective Test — Set 03

---

### 21. Question

A developer has a dataset with `min=0` and `max=50`. They call `plt.hist(data, bins=10)`. A data point has value exactly `25`. Into which bin does it fall, and why?

- [ ] **A.** Bin 5: `[20, 25)` — 25 is the right boundary of bin 5, so it is excluded
- [ ] **B.** Bin 6: `[25, 30)` — 25 is the left boundary of bin 6, so it is included
- [ ] **C.** Bin 5: `[20, 25]` — 25 is included in bin 5 because the last element is always included
- [ ] **D.** Bin 5 or 6 — behavior is undefined when a value falls exactly on a bin edge

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** In the left-closed convention `[a, b)`, the right boundary `b` is **excluded**. So `[20, 25)` does NOT include 25. The value 25 cannot belong to bin 5 under this convention.
- **Option B (Correct):** Bin width = (50-0)/10 = 5. Edges: 0, 5, 10, 15, 20, **25**, 30, 35, 40, 45, 50. Bin 6 is `[25, 30)`. Since bins are **left-closed** (include the left boundary), the value 25 is included in bin 6 — it is the left edge of bin 6. This is consistent with Matplotlib's `[a, b)` convention for all bins except the last.
- **Option C (Incorrect):** The "last element included" special case only applies to the **very last bin** of the histogram (the rightmost bin). Bin 5 `[20, 25)` is not the last bin — the last bin is `[45, 50]`. So the fully-closed exception does not apply here.
- **Option D (Incorrect):** Matplotlib's behavior at exact bin boundaries is well-defined and deterministic, not undefined. The left-closed convention `[a, b)` unambiguously assigns boundary values to the bin that starts at that value.

</details>

---

### 22. Question

A developer studies the under-smoothing problem in histograms:

```python
np.random.seed(15)
data = np.random.normal(50, 10, 500)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(data, bins=2)
plt.title("bins=2")

plt.subplot(1, 3, 2)
plt.hist(data, bins=20)
plt.title("bins=20")

plt.subplot(1, 3, 3)
plt.hist(data, bins=200)
plt.title("bins=200")
```

Which description best matches the `bins=2` histogram?

- [ ] **A.** A smooth bell curve showing the normal distribution shape clearly
- [ ] **B.** Two very wide bars that collapse the entire data range into 2 groups — skewness and the bell shape are completely hidden
- [ ] **C.** 2 bars with very noisy, jagged heights due to sample variance
- [ ] **D.** The histogram fails silently and produces no output

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Two bins cannot possibly show a bell curve — a normal distribution requires enough bins to reveal the characteristic symmetric peak and tails. With only 2 bins, all distributional shape information is destroyed.
- **Option B (Correct):** With `bins=2`, the entire data range is divided into just 2 intervals. Each bar represents roughly half the samples (≈250). The result is two wide, roughly equal-height bars that reveal nothing about the distribution shape. This is severe **under-smoothing** — too few bins collapse the data into undifferentiated chunks, hiding skewness, peaks, and modality. The normal distribution's bell shape is completely invisible.
- **Option C (Incorrect):** Noisy, jagged heights (like random spikes of varying heights) are characteristic of **over-smoothing** (`bins=200` for 500 data points) — where bins are so narrow that random sampling noise dominates. With only 2 bins and ≈250 points per bin, random variance in count is minimal relative to the bar height.
- **Option D (Incorrect):** `bins=2` is a perfectly valid argument. Matplotlib will draw the 2-bin histogram without error. It simply produces a visually uninformative result.

</details>

---

### 23. Question

What visual problem does the `bins=200` histogram produce for a dataset of only 500 samples?

- [ ] **A.** The bars become invisible because 200 bins require at least 200,000 samples
- [ ] **B.** Many bins contain 0 or 1 data points — producing an irregular spiked pattern where sample noise dominates over the true distribution shape (over-smoothing)
- [ ] **C.** Matplotlib automatically reduces bins to 50 when bins > sample size / 4
- [ ] **D.** The histogram becomes a smooth KDE curve when bins exceed 100

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** There is no minimum sample size requirement per bin count. Even with 200 bins and 500 samples, Matplotlib draws the histogram — it just produces very sparse, irregular bars.
- **Option B (Correct):** With 200 bins for 500 samples, each bin on average contains only 2.5 data points. Due to random sampling variability, many bins will contain 0, 1, 2, 3, or 4 points — producing a highly irregular "spike" pattern that reflects **random sample noise** rather than the true population density. This is **over-smoothing** (too many narrow bins). The ideal bin count balances smoothness (enough data per bin) against resolution (enough bins to show shape). Common heuristics include Sturges' rule: `k ≈ 1 + log₂(n)` ≈ 10 for n=500, or Scott's rule.
- **Option C (Incorrect):** Matplotlib does NOT automatically reduce bin counts. It always uses exactly the specified `bins` value, regardless of sample size. The responsibility for choosing an appropriate bin count is entirely the developer's.
- **Option D (Incorrect):** `bins=200` does not convert the histogram to a KDE curve. KDE requires separate computation (e.g., `scipy.stats.gaussian_kde`). No matter how many bins are specified, `plt.hist()` always produces a rectangular bar histogram.

</details>

---

### 24. Question

A developer passes an explicit non-uniform bin edge list:

```python
data = [3, 7, 12, 18, 22, 28, 33, 37, 45, 49]
counts, bins, patches = plt.hist(data, bins=[0, 10, 20, 40, 50])
print(counts)
```

How many data points fall in each bin `[0,10)`, `[10,20)`, `[20,40)`, `[40,50]`?

- [ ] **A.** `[2, 2, 4, 2]`
- [ ] **B.** `[2, 3, 3, 2]`
- [ ] **C.** `[2, 2, 3, 3]`
- [ ] **D.** `[3, 2, 3, 2]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** Assigning each value: `[0,10)`: 3, 7 → count=**2**. `[10,20)`: 12, 18 → count=**2**. `[20,40)`: 22, 28, 33, 37 → count=**4**. `[40,50]`: 45, 49 → count=**2**. Result: `[2, 2, 4, 2]`. Note the bin `[20,40)` is twice as wide as the others — it naturally captures more data points when the data is roughly uniform in that range.
- **Option B (Incorrect):** `[2, 3, 3, 2]` sums to 10 correctly, but the assignment is wrong. Bin `[10,20)` has values 12 and 18 (count=2, not 3), and bin `[20,40)` has 22, 28, 33, 37 (count=4, not 3).
- **Option C (Incorrect):** `[2, 2, 3, 3]` assigns only 3 values to `[20,40)` and 3 to `[40,50]`. But [40,50] only contains 45 and 49 (count=2), not 3.
- **Option D (Incorrect):** `[3, 2, 3, 2]` would mean 3 values in `[0,10)`. But only 3 and 7 fall in [0,10) — that's only 2 values.

</details>

---

### 25. Question

A developer examines the following code and must predict what `len(counts)`, `len(bins)`, and `len(patches)` will be:

```python
edges = [0, 5, 10, 20, 35, 50, 80, 100]
counts, bins, patches = plt.hist(data, bins=edges)
```

- [ ] **A.** `len(counts)=8`, `len(bins)=8`, `len(patches)=8`
- [ ] **B.** `len(counts)=7`, `len(bins)=8`, `len(patches)=7`
- [ ] **C.** `len(counts)=8`, `len(bins)=7`, `len(patches)=8`
- [ ] **D.** `len(counts)=7`, `len(bins)=7`, `len(patches)=7`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** When using an explicit edge list of length 8, there are 7 bins (intervals between adjacent edges), not 8. `len(counts)` equals the number of bins = 7, not 8.
- **Option B (Correct):** The `edges` list has 8 values: `[0, 5, 10, 20, 35, 50, 80, 100]`. This defines 7 bins (7 intervals between 8 edges). Therefore: `len(counts) = 7` (one count per bin), `len(bins) = 8` (the original edge list), `len(patches) = 7` (one Rectangle per bin). The relationship `len(bins) = len(counts) + 1 = len(patches) + 1` always holds.
- **Option C (Incorrect):** This reverses the counts/bins lengths. `len(bins)` is always one MORE than `len(counts)`, not one less.
- **Option D (Incorrect):** `len(bins)` cannot equal `len(counts)`. The bins array stores edges (N+1 values) while counts stores per-bin frequencies (N values). They can never be equal.

</details>

---

### 26. Question

A developer creates two histograms on the same axes using a **list of arrays**:

```python
group_a = np.random.normal(40, 5, 300)
group_b = np.random.normal(60, 5, 300)

plt.hist([group_a, group_b], bins=20,
         label=["Group A", "Group B"],
         color=["steelblue", "tomato"])
plt.legend()
```

What does passing `[group_a, group_b]` as the first argument produce?

- [ ] **A.** An error — `plt.hist()` only accepts a single 1D array
- [ ] **B.** Two separate overlapping histograms drawn in different colors, with bars for each group side-by-side within each bin (grouped bar mode)
- [ ] **C.** A single merged histogram combining all 600 values into one distribution
- [ ] **D.** A stacked histogram where group_b bars are stacked on top of group_a bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `plt.hist()` explicitly supports a list of arrays as input. This is documented behavior for creating multi-series histograms in a single function call.
- **Option B (Correct):** When a list of arrays is passed, `plt.hist()` draws **grouped (side-by-side) bars** within each bin — multiple bars appear per bin position, one for each input series. With `color=["steelblue", "tomato"]` and `label=["Group A", "Group B"]`, each group is drawn in its own color. Within each bin, Group A and Group B bars appear side by side. This is the grouped histogram mode (analogous to grouped bar charts).
- **Option C (Incorrect):** The arrays are NOT merged. Each is treated as a separate series with its own counts, colors, and labels. To merge, you would pass `np.concatenate([group_a, group_b])` as a single 1D array.
- **Option D (Incorrect):** Stacking is NOT the default behavior. To stack, add `stacked=True`. Without it, the default mode is side-by-side grouped bars.

</details>

---

### 27. Question

A developer creates a stacked histogram:

```python
data_a = np.random.normal(50, 8, 400)
data_b = np.random.normal(55, 8, 400)
plt.hist([data_a, data_b], bins=15, stacked=True,
         color=["#4CAF50", "#2196F3"], alpha=0.8,
         label=["Class 0", "Class 1"])
```

In a stacked histogram, what does the **total height of each stacked bar** represent?

- [ ] **A.** The average count of the two classes in that bin
- [ ] **B.** The sum of counts from both classes in that bin — the combined frequency of all data in that bin range
- [ ] **C.** The maximum count of the two classes in that bin
- [ ] **D.** Always 100% — stacked histograms are normalized to 1.0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Stacked bar height = sum, not average. Averaging would produce misleading results when the two classes have different sample sizes.
- **Option B (Correct):** In a stacked histogram (`stacked=True`), each bin contains two stacked bar segments — the bottom segment for Class 0 (height = count of Class 0 in that bin) and the top segment for Class 1 (height = count of Class 1 in that bin). The **total bar height** equals the **sum** of both class counts in that bin, representing the combined frequency. The visual distinction between the two colored segments shows the relative contribution of each class.
- **Option C (Incorrect):** Using the maximum of the two classes would lose information about the smaller class and wouldn't produce additive stacking behavior.
- **Option D (Incorrect):** `stacked=True` alone does NOT normalize to 100%. The bars show raw counts (or densities if `density=True`), not proportions. To create a 100%-stacked (proportional) histogram, additional normalization code would be required beyond just `stacked=True`.

</details>

---

### 28. Question

A developer adds mean and standard deviation annotations to a histogram:

```python
np.random.seed(10)
data = np.random.normal(75, 12, 1000)
mu, sigma = np.mean(data), np.std(data)

plt.hist(data, bins=30, edgecolor="black", alpha=0.7)
plt.axvline(mu, color="red", linewidth=2, label=f"Mean={mu:.1f}")
plt.axvline(mu - sigma, color="orange", linestyle="--", label="-1σ")
plt.axvline(mu + sigma, color="orange", linestyle="--", label="+1σ")
plt.legend()
```

What percentage of data points theoretically fall between the two orange dashed lines (μ−σ to μ+σ) for a normal distribution?

- [ ] **A.** 50%
- [ ] **B.** 68.3%
- [ ] **C.** 95.4%
- [ ] **D.** 99.7%

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 50% would correspond to the interquartile range (Q1 to Q3, approximately μ−0.674σ to μ+0.674σ for a normal distribution) — a narrower interval than ±1σ.
- **Option B (Correct):** The empirical rule (68-95-99.7 rule) for a normal distribution states that approximately **68.3%** of data falls within ±1 standard deviation of the mean (between μ−σ and μ+σ). The two orange `axvline` markers at μ−σ and μ+σ enclose this central 68.3% region. This is why the histogram bars between the orange lines should visually contain about 68% of the total bar area.
- **Option C (Incorrect):** 95.4% corresponds to the ±2σ range (μ−2σ to μ+2σ) — a wider interval than the ±1σ shown by the orange lines.
- **Option D (Incorrect):** 99.7% corresponds to the ±3σ range (μ−3σ to μ+3σ) — an even wider interval encompassing almost all of the normal distribution's mass.

</details>

---

### 29. Question

A developer creates a histogram with `edgecolor="black"`. What specific visual change does this make?

- [ ] **A.** Changes the color of the x and y axis spine lines to black
- [ ] **B.** Adds a black border/outline around each individual histogram bar, visually separating adjacent bars
- [ ] **C.** Fills the histogram bars with a black gradient from bottom to top
- [ ] **D.** Sets the histogram background to black

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Axis spine colors are controlled by `ax.spines["bottom"].set_color("black")` etc. The `edgecolor` parameter in `plt.hist()` applies to the bar patches, not the axes frame.
- **Option B (Correct):** `edgecolor="black"` adds a black outline/border around each rectangular bar patch. By default (`edgecolor=None`), adjacent bars share no visual separator — they blend together when using the same fill color. Adding `edgecolor="black"` creates clear black borders around every bar, making individual bins visually distinct. This is especially important for fine-grained histograms where bars are narrow.
- **Option C (Incorrect):** Gradient fills within bars are not supported by the `edgecolor` parameter. Gradient requires custom `PatchCollection` styling or `matplotlib.colors.LinearSegmentedColormap` applied per-patch.
- **Option D (Incorrect):** Plot background color is controlled by `ax.set_facecolor("black")` or `fig.set_facecolor("black")`. The `edgecolor` parameter only affects bar outlines.

</details>

---

### 30. Question

A developer inspects a histogram of model prediction confidence scores and observes this pattern: a large peak near 0.0–0.1, a gradual decline toward 0.5, and almost no predictions between 0.5–1.0. What distribution shape is this, and what ML issue might it indicate?

- [ ] **A.** Normal distribution — the model is well-calibrated
- [ ] **B.** Right-skewed distribution — the model assigns high confidence to almost all predictions (most predictions near 1.0)
- [ ] **C.** Left-skewed distribution — the model is mostly assigning low confidence scores (most predictions near 0.0), possibly indicating a poorly trained or threshold-miscalibrated model
- [ ] **D.** Bimodal distribution — two distinct groups of predictions at 0.0 and 1.0

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A normal distribution would show a symmetric bell curve centered around 0.5 for confidence scores. The described pattern (peak at 0.0–0.1, declining toward 0.5, empty above 0.5) is NOT symmetric.
- **Option B (Incorrect):** Right-skewed (high confidence near 1.0) would show a peak at the HIGH end of the scale (0.9–1.0) with a long left tail toward 0.0 — the exact opposite of the described pattern.
- **Option C (Correct):** The described pattern — large peak at LOW values (0.0–0.1), declining toward 0.5, and no values above 0.5 — represents a **left-skewed distribution** in confidence space. Most predictions are concentrated at very low confidence scores. This pattern could indicate: (1) the model is classifying everything as the negative class, (2) the model is poorly trained and outputs near-zero probabilities, (3) a calibration issue where the sigmoid/softmax outputs are compressed near 0, or (4) a class imbalance problem where the model defaults to low-confidence predictions for a rare positive class.
- **Option D (Incorrect):** A bimodal distribution would show two distinct peaks — one near 0.0 and one near 1.0 — with a valley in between. The described pattern has a single large peak at 0.0 with a monotone decline — this is unimodal and left-skewed, not bimodal.

</details>
