# Matplotlib Histogram Objective Test — Set 01

---

### 1. Question

A developer runs the following code:

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.array([12, 18, 24, 31, 35, 42, 48, 55, 61, 67])
counts, bins, patches = plt.hist(data, bins=5)
```

What is the **bin width** and what are the **6 bin edge values**?

- [ ] **A.** Bin width = 10; edges = [12, 22, 32, 42, 52, 62]
- [ ] **B.** Bin width = 11; edges = [12, 23, 34, 45, 56, 67]
- [ ] **C.** Bin width = 11.0; edges = [12.0, 23.0, 34.0, 45.0, 56.0, 67.0]
- [ ] **D.** Bin width = 55/5 = 11; edges = [0, 11, 22, 33, 44, 55]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** A width of 10 would result from `(62 - 12) / 5 = 10`, which is wrong. The max value is 67, not 62. The formula uses the actual data min and max.
- **Option B (Incorrect):** The formula is correct: `(67 - 12) / 5 = 55 / 5 = 11`. However, presenting the bin width as the integer 11 without showing that Matplotlib works in floats is imprecise. More critically, the answer omits the decimal representation that Matplotlib returns — but the core values are the same as C. The key distinction is that **C** correctly represents Matplotlib's float output.
- **Option C (Correct):** Bin width = `(max - min) / bins = (67 - 12) / 5 = 55 / 5 = 11.0`. The 6 bin edges are: `[12.0, 23.0, 34.0, 45.0, 56.0, 67.0]`. Matplotlib calculates edges as `min + i × width` for i = 0, 1, …, N. The `bins` array always has exactly `N + 1 = 6` elements for `bins=5`.
- **Option D (Incorrect):** Starting the bin edges at 0 is wrong. `plt.hist()` uses the **data range** `[min(data), max(data)]` as the default span — it does NOT start from 0 unless 0 happens to be the minimum value. The edges start at `min(data) = 12`.

</details>

---

### 2. Question

A developer calls `plt.hist(data, bins=8)` on a dataset and then inspects the returned arrays:

```python
counts, bins, patches = plt.hist(data, bins=8)
print(len(counts), len(bins), len(patches))
```

Which output is **always correct** regardless of the data?

- [ ] **A.** `8 8 8`
- [ ] **B.** `8 9 8`
- [ ] **C.** `9 8 9`
- [ ] **D.** `8 9 9`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `len(bins)` is never equal to `len(counts)`. The `bins` array stores bin **edges** — for N bins there are always N+1 edges (think of N fence posts requiring N+1 poles). So `len(bins) = 9` for `bins=8`, not 8.
- **Option B (Correct):** `counts` has length 8 (one frequency value per bin). `bins` has length **9** (8 bins → 9 edges: `[e0, e1, e2, …, e8]`). `patches` has length 8 (one `Rectangle` artist per bar). This is the fundamental relationship: `len(counts) == bins == len(patches)` and `len(bins) == num_bins + 1`.
- **Option C (Incorrect):** `len(counts)` equals the number of bins (8), not 9. Swapping counts and bins lengths is the classic misconception this question targets.
- **Option D (Incorrect):** `len(patches) = 9` is wrong — there are exactly 8 rectangular bar patches for 8 bins. `patches` and `counts` always have the same length.

</details>

---

### 3. Question

What is Matplotlib's **default number of bins** when `plt.hist(data)` is called without a `bins` argument?

- [ ] **A.** 20
- [ ] **B.** 10
- [ ] **C.** 5
- [ ] **D.** 50

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 20 bins is not the default. A developer might assume this because 20 is a common choice in practice, but the factory default is 10.
- **Option B (Correct):** Matplotlib's default `bins` parameter is **10**. When `plt.hist(data)` is called without specifying `bins`, it divides the data range `[min, max]` into exactly 10 equal-width bins. This default is controlled by `matplotlib.rcParams["hist.bins"]`.
- **Option C (Incorrect):** 5 bins would produce a very coarse histogram. This is not Matplotlib's default — it would significantly under-smooth most distributions.
- **Option D (Incorrect):** 50 bins would produce a fine-grained histogram. This is not the default — it would over-smooth many small datasets by creating many empty or near-empty bins.

</details>

---

### 4. Question

A developer passes an **explicit sequence** of bin edges to `plt.hist()`:

```python
data = np.array([2, 8, 14, 22, 28, 35, 41, 48])
counts, bins, patches = plt.hist(data, bins=[0, 10, 30, 50])
```

How many bars are drawn, and what are the **widths** of each bar?

- [ ] **A.** 4 bars with widths [10, 20, 20]
- [ ] **B.** 3 bars with widths [10, 20, 20]
- [ ] **C.** 3 bars with widths [10, 10, 10]
- [ ] **D.** 4 bars with widths [0, 10, 30, 50]

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 4 edges define 3 bins (not 4 bars). The number of bars equals the number of bins, which is `len(edges) - 1 = 4 - 1 = 3`.
- **Option B (Correct):** With `bins=[0, 10, 30, 50]`, there are 3 bins: `[0,10)`, `[10,30)`, `[30,50]`. Their widths are: bin 1: `10 - 0 = 10`; bin 2: `30 - 10 = 20`; bin 3: `50 - 30 = 20`. Three bars are drawn with widths 10, 20, and 20. This is the key advantage of explicit bin edges — you can create **unequal-width bins** tailored to the data's natural groupings.
- **Option C (Incorrect):** Widths of [10, 10, 10] would require evenly spaced edges like [0, 10, 20, 30]. The given edges [0, 10, 30, 50] produce unequal widths because the intervals are not uniform.
- **Option D (Incorrect):** The values [0, 10, 30, 50] are the bin **edges**, not the bar widths. There are 4 edges but only 3 bars (one per bin interval between adjacent edges).

</details>

---

### 5. Question

A dataset has `min = 30` and `max = 180`. A developer calls `plt.hist(data, bins=10)`. A new data point has value `105`. Into which bin does it fall?

- [ ] **A.** Bin 5: `[90, 105]`
- [ ] **B.** Bin 6: `[105, 120)`
- [ ] **C.** Bin 6: `[90, 105)`
- [ ] **D.** Bin 6: `[105, 120]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This option has the wrong bin boundaries and an incorrect inclusive-right interval. Standard bins use left-closed, right-open intervals `[a, b)` except for the last bin.
- **Option B (Incorrect):** `[105, 120)` would be bin 6 starting AT 105 — but 105 is a LEFT edge, not a right edge. The bin containing 105 has 105 as its RIGHT boundary.
- **Option C (Correct):** Bin width = `(180 - 30) / 10 = 150 / 10 = 15`. Bin edges: 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180. Bin 6 spans `[90, 105)` (left edge = 90, right edge = 105). The value 105 is exactly on the right boundary — by Matplotlib's convention, 105 falls in the NEXT bin `[105, 120)` which is Bin 7... Actually wait — let me reconsider. The bins are: Bin1=[30,45), Bin2=[45,60), Bin3=[60,75), Bin4=[75,90), Bin5=[90,105), Bin6=[105,120), Bin7=[120,135), Bin8=[135,150), Bin9=[150,165), Bin10=[165,180]. Value 105 falls in **Bin 6** = `[105, 120)`. Option C says Bin 6 = `[90, 105)` which is actually Bin 5. The correct bin for value 105 is `[105, 120)` = Bin 6.

**Revised Correct Answer: B** — `[105, 120)` is Bin 6. The value 105 sits at the left boundary of the `[105, 120)` interval — since bins are left-closed, 105 is included in `[105, 120)`.

</details>

---

### 6. Question

A developer asks: "What is the key difference between a **histogram** and a **bar chart** in Matplotlib?" Which answer is most accurate?

- [ ] **A.** Histograms use `ax.bar()` while bar charts use `ax.hist()` — they are just aliases
- [ ] **B.** Histograms visualize the distribution of a **continuous quantitative** variable by grouping values into bins; bar charts display values for **discrete categorical** variables where each bar represents a separate category
- [ ] **C.** Histograms and bar charts are identical — the only difference is that histograms add `edgecolor="black"` by default
- [ ] **D.** Bar charts automatically sort the x-axis; histograms do not sort data

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This is completely backwards. `ax.hist()` creates histograms; `ax.bar()` creates bar charts. They are different functions with different inputs, algorithms, and visual outputs — not aliases for each other.
- **Option B (Correct):** This is the fundamental conceptual distinction. A **histogram** takes a 1D array of continuous numeric values, automatically bins them into intervals, counts how many values fall in each interval, and draws bars whose heights represent frequencies. The x-axis is a continuous numeric range. A **bar chart** takes pre-computed values for discrete named categories (e.g., "Monday", "Tuesday") and draws one bar per category. The x-axis is categorical. Confusing these two is one of the most common visualization mistakes.
- **Option C (Incorrect):** Histograms and bar charts are fundamentally different in their data model and algorithm. Edge color is a cosmetic parameter, not a defining characteristic. `plt.hist()` does not add black edges by default — the default `edgecolor` is `None`.
- **Option D (Incorrect):** Neither function sorts by default in the way described. `ax.hist()` orders bins by their numeric left edge (naturally sorted). `ax.bar()` displays categories in the order they are provided — no automatic alphabetical sorting occurs.

</details>

---

### 7. Question

A developer sets `np.random.seed(42)` before generating data for a histogram. A colleague says "changing the seed to 99 will change the shape of the histogram." Is this correct?

- [ ] **A.** Yes — different seeds always produce completely different distribution shapes
- [ ] **B.** No — the seed only affects the starting point of the pseudo-random sequence, so for large samples the histogram shape converges to the theoretical distribution regardless of seed
- [ ] **C.** Yes — the seed directly controls the bin widths of the histogram
- [ ] **D.** No — `np.random.seed()` has no effect on NumPy arrays; it only affects Python's built-in `random` module

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** Different seeds produce different specific sample values, but for large samples (e.g., n=1000 or n=10000), the histogram shape approximates the **theoretical population distribution** closely regardless of seed. The seed changes WHICH specific random values are generated, but not the overall distributional shape for large n.
- **Option B (Correct):** `np.random.seed(n)` sets the starting state of the pseudo-random number generator. It determines the exact sequence of random values generated — ensuring **reproducibility** (same seed = same values every run). For small samples, different seeds can produce noticeably different histogram shapes. For large samples, the law of large numbers ensures all seeds produce similar bell-curve (or whatever the theoretical shape is) histograms. The seed affects data generation, NOT the plotting algorithm.
- **Option C (Incorrect):** `np.random.seed()` has zero effect on bin widths. Bin widths are determined by the `bins` parameter and the data range `[min, max]`. The random seed only controls the generated data values.
- **Option D (Incorrect):** `np.random.seed()` absolutely affects NumPy's random number generation. It IS the function used to seed NumPy's Mersenne Twister PRNG. Python's built-in `random` module has its own separate seed function `random.seed()`.

</details>

---

### 8. Question

A developer creates a histogram with `bins=4` for the dataset `[5, 10, 15, 20, 25, 30, 35, 40]`. What are the **4 bin intervals**?

- [ ] **A.** `[5,10)`, `[10,15)`, `[15,20)`, `[20,40]`
- [ ] **B.** `[5,13.75)`, `[13.75,22.5)`, `[22.5,31.25)`, `[31.25,40]`
- [ ] **C.** `[0,10)`, `[10,20)`, `[20,30)`, `[30,40]`
- [ ] **D.** `[5,15)`, `[15,25)`, `[25,35)`, `[35,45]`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** These are unequal-width bins — three bins of width 5 and one of width 20. Matplotlib creates equal-width bins when given an integer `bins` count. The first three bins having width 5 and the last having width 20 is not consistent with the formula.
- **Option B (Correct):** Bin width = `(max - min) / bins = (40 - 5) / 4 = 35 / 4 = 8.75`. Edges: 5.0, 13.75, 22.5, 31.25, 40.0. Four bins: `[5, 13.75)`, `[13.75, 22.5)`, `[22.5, 31.25)`, `[31.25, 40]`. Note that bin widths are not always round numbers — they depend on the data range divided by the bin count.
- **Option C (Incorrect):** Starting at 0 is wrong — the bin range starts at `min(data) = 5`. Bins from 0 would be correct only if the minimum data value were 0.
- **Option D (Incorrect):** Edges [5, 15, 25, 35, 45] imply bin width = 10. But (40-5)/4 = 8.75 ≠ 10. Also, the last edge extends to 45 beyond max(data)=40, which is not how Matplotlib computes default bins.

</details>

---

### 9. Question

A developer runs the following code:

```python
np.random.seed(7)
data = np.random.normal(loc=100, scale=15, size=500)
plt.hist(data, bins=20, color="steelblue", edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")
plt.grid(axis="y", alpha=0.5)
plt.show()
```

What does `plt.grid(axis="y", alpha=0.5)` add to the histogram?

- [ ] **A.** Horizontal gridlines running across the plot at each y-tick position, rendered at 50% transparency
- [ ] **B.** Vertical gridlines at each bin edge position with 50% transparency
- [ ] **C.** A semi-transparent gray background covering 50% of the plot area
- [ ] **D.** Gridlines on both x and y axes at 50% transparency

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** `plt.grid(axis="y", alpha=0.5)` adds **horizontal** gridlines (running left-to-right) at each y-axis tick mark position. The `axis="y"` parameter restricts gridlines to the y-direction only (horizontal lines). `alpha=0.5` sets 50% transparency — making them visible but not overwhelming. This is a common styling choice for histograms to help readers estimate bar heights without cluttering the plot with vertical lines too.
- **Option B (Incorrect):** `axis="y"` does NOT add vertical gridlines at bin edges. Vertical gridlines (at x-axis positions) would require `axis="x"` or `axis="both"`. Gridlines appear at tick mark positions, not at bin edges.
- **Option C (Incorrect):** `plt.grid()` does not add a background fill or shading. It draws thin lines at tick positions. Background color is controlled by `ax.set_facecolor()`.
- **Option D (Incorrect):** `axis="y"` explicitly restricts gridlines to the y-axis only. Both-axis gridlines would require `axis="both"` (or just `plt.grid(alpha=0.5)` since "both" is the default when `axis` is not specified).

</details>

---

### 10. Question

A developer has the dataset `[100, 200, 300, 400, 500]` and calls `plt.hist(data, bins=4)`. Which statement about the last bin is TRUE?

- [ ] **A.** The last bin uses the same half-open interval `[400, 500)` as all other bins — value 500 falls outside all bins
- [ ] **B.** The last bin uses a **fully closed** interval `[400, 500]` — value 500 IS included in the last bin
- [ ] **C.** The last bin extends beyond 500 to ensure all values fit
- [ ] **D.** Matplotlib raises an error when the maximum value exactly equals the last bin edge

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This describes the interval convention for all bins EXCEPT the last one. If the last bin were `[400, 500)`, the maximum value 500 would fall outside all bins — which would silently drop it from the histogram count. Matplotlib avoids this by making the last bin special.
- **Option B (Correct):** Matplotlib's bin interval convention is: all bins except the last use **left-closed, right-open** intervals `[a, b)`. The **last bin** uses a **fully closed** interval `[a, b]` — ensuring the maximum data value is always captured. For `bins=4` on `[100, 500]`: edges = [100, 200, 300, 400, 500]. Bins: `[100,200)`, `[200,300)`, `[300,400)`, `[400,500]`. The last bin includes both endpoints.
- **Option C (Incorrect):** Matplotlib does NOT extend bins beyond the data range. The last bin edge is exactly `max(data) = 500`. No extension occurs.
- **Option D (Incorrect):** No error is raised. Matplotlib gracefully handles values at exact bin boundaries using the closed-last-bin convention. This is deliberate design to ensure 100% of data values are counted.

</details>
