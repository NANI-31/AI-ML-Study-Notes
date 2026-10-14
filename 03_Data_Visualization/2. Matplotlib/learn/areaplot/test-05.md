# Matplotlib Area Plot Objective Test — Set 05

---

### 41. Question

A geophysicist uses `fill_betweenx` to shade a horizontal pressure range on a vertical depth profile:

```python
depth = np.array([0, 10, 20, 30, 40, 50])       # y-axis (vertical)
pressure = np.array([1.0, 1.5, 2.2, 3.0, 3.8, 4.5])

fig, ax = plt.subplots()
ax.plot(pressure, depth, "b-", lw=2)
ax.fill_betweenx(depth, pressure, 0,
                 color="skyblue", alpha=0.4)
ax.invert_yaxis()
ax.set_xlabel("Pressure (atm)")
ax.set_ylabel("Depth (m)")
```

What is the primary conceptual difference between `fill_between` and `fill_betweenx`?

- [ ] **A.** `fill_between` fills along the y-axis; `fill_betweenx` fills along the x-axis — they are exact opposites
- [ ] **B.** `fill_between` shades horizontally between two y-values at each x-position; `fill_betweenx` shades vertically between two x-values at each y-position
- [ ] **C.** `fill_betweenx` is deprecated; use `fill_between` with transposed arguments instead
- [ ] **D.** `fill_betweenx` creates horizontal bars, while `fill_between` creates vertical bars

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The axes labels are transposed but not exactly "opposites." The key distinction is the orientation of the fill — vertical (fill_between) vs. horizontal (fill_betweenx) — not which axis they "fill along."
- **Option B (Correct):** `fill_between(x, y1, y2)` shades the vertical band between two y-value curves at each x-position — the sweep direction is horizontal (you move left to right across x-values). `fill_betweenx(y, x1, x2)` shades the horizontal band between two x-value curves at each y-position — the sweep direction is vertical. In the depth profile above, the y-axis represents depth (vertical) and the x-axis represents pressure (horizontal). `fill_betweenx` is the natural choice for filling across the horizontal (pressure) direction while sweeping through the vertical (depth) positions.
- **Option C (Incorrect):** `fill_betweenx` is a fully supported, actively maintained Matplotlib function. It is not deprecated. Transposing arguments to `fill_between` would also not produce the same orientation.
- **Option D (Incorrect):** Neither function creates "bars." Both create smoothly filled polygon regions. The shape depends on the input curve data, not discrete bar-like structures.

</details>

---

### 42. Question

A developer uses `fill_betweenx` with the following parameters:

```python
y = np.array([0, 1, 2, 3, 4])     # vertical axis
x1 = np.array([2, 3, 4, 3, 2])    # left boundary
x2 = np.array([5, 6, 7, 6, 5])    # right boundary

plt.fill_betweenx(y, x1, x2, color="salmon", alpha=0.5)
```

At `y=2`, what is the horizontal width of the shaded region?

- [ ] **A.** 3 units (from x=4 to x=7)
- [ ] **B.** 2 units (from x=2 to x=4)
- [ ] **C.** 7 units (from x=0 to x=7)
- [ ] **D.** 4 units (from x=3 to x=7)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** At `y=2` (index 2), `x1[2]=4` and `x2[2]=7`. `fill_betweenx(y, x1, x2)` shades the horizontal band between `x1` and `x2` at each y-position. At y=2, the fill spans from x=4 to x=7, a horizontal width of `7 - 4 = 3` units.
- **Option B (Incorrect):** x1=2 and x2=4 with width=2 corresponds to y=0 (index 0), not y=2.
- **Option C (Incorrect):** Width of 7 from x=0 to x=7 would imply x1=0 as the left boundary, which is not the case. The left boundary is x1[2]=4.
- **Option D (Incorrect):** Width of 4 from x=3 to x=7 would correspond to x1=3 (which occurs at y=1 and y=3, not y=2).

</details>

---

### 43. Question

A data analyst creates a horizontal confidence band using `fill_betweenx`:

```python
response_rate = np.linspace(0, 1, 100)   # y-axis (0 to 100%)
baseline_dose = 50 + 10 * np.log1p(response_rate * 10)
lower_dose = baseline_dose - 5
upper_dose = baseline_dose + 5

fig, ax = plt.subplots()
ax.fill_betweenx(response_rate, lower_dose, upper_dose,
                 color="green", alpha=0.3, label="Dose ±5 mg")
ax.plot(baseline_dose, response_rate, "g-", lw=2, label="Median Dose")
```

In this dose-response plot, what does the shaded green region represent?

- [ ] **A.** The range of response rates achievable at a fixed dose of 50 mg
- [ ] **B.** The range of doses (within ±5 mg of the median) that corresponds to each response rate — a horizontal uncertainty band in the dose axis
- [ ] **C.** The cumulative dose delivered over all response levels
- [ ] **D.** The region where the dose exceeds 50 mg

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** The shaded region is a horizontal band in the dose (x) direction, not a vertical range of response rates. It does not represent the variability in response at a single fixed dose.
- **Option B (Correct):** `fill_betweenx(y, x1, x2)` shades horizontally between `lower_dose` and `upper_dose` at each response rate level. At every response rate value on the y-axis, the shaded width shows the ±5 mg uncertainty range around the median dose required to achieve that response rate. This is the standard way to display dose uncertainty in pharmacological dose-response curves — the y-axis is the response and the x-axis is the dose.
- **Option C (Incorrect):** Cumulative dose would require `np.cumsum` applied to the dose array and would produce an area below the cumulative curve, not a horizontal band.
- **Option D (Incorrect):** The condition "dose exceeds 50 mg" would require a `where` parameter. The fill here has no such condition and covers all response rates across the ±5 mg band.

</details>

---

### 44. Question

A meteorologist uses `fill_betweenx` to show wind speed range at different altitudes:

```python
altitude = np.array([0, 500, 1000, 2000, 3000, 5000])   # meters
min_wind = np.array([2, 5, 8, 15, 20, 35])
max_wind = np.array([8, 12, 18, 28, 38, 55])

plt.fill_betweenx(altitude, min_wind, max_wind, color="lightsteelblue", alpha=0.6)
plt.plot(min_wind, altitude, "b--", label="Min Wind")
plt.plot(max_wind, altitude, "b-", label="Max Wind")
plt.xlabel("Wind Speed (km/h)")
plt.ylabel("Altitude (m)")
```

At altitude 2000m, what is the horizontal span of the shaded wind speed range?

- [ ] **A.** 15 km/h (from 0 to 15)
- [ ] **B.** 43 km/h (from 15 to 28, spanning across both boundaries)
- [ ] **C.** 13 km/h (from 15 to 28)
- [ ] **D.** 28 km/h (the max wind speed at 2000m)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** 15 km/h is the value of `min_wind` at 2000m (the minimum boundary), not the width of the shaded band. The width is the difference between max and min wind, not the min value itself.
- **Option B (Incorrect):** 43 km/h would be the width at 3000m: `max_wind - min_wind = 38 - 20 = 18` — actually not 43. 43 = 28 + 15 which is their sum, not difference. This is mathematically incorrect.
- **Option C (Correct):** At altitude=2000m, `min_wind=15` and `max_wind=28`. The horizontal width of the `fill_betweenx` shading is `max_wind - min_wind = 28 - 15 = 13` km/h. The shaded region spans from x=15 to x=28 at this altitude level.
- **Option D (Incorrect):** 28 km/h is the absolute value of `max_wind` at 2000m — the upper boundary of the band — not its width. Width requires subtracting the lower boundary.

</details>

---

### 45. Question

A developer uses `fill_betweenx` but mistakenly swaps arguments compared to `fill_between`:

```python
# Intended: horizontal fill for a depth profile
depth = np.linspace(0, 100, 50)
concentration = np.sin(depth / 10) * 5 + 10

# Developer's code (possibly wrong):
plt.fill_between(depth, concentration)   # Option A usage
plt.fill_betweenx(depth, concentration)  # Option B usage
```

For a vertical depth profile where depth is the y-axis and concentration is the x-axis, which call is semantically correct?

- [ ] **A.** `fill_between(depth, concentration)` — fills the area under the concentration curve with depth on the x-axis
- [ ] **B.** `fill_betweenx(depth, concentration)` — fills the area to the left of the concentration curve with depth on the y-axis, appropriate for vertical profiles
- [ ] **C.** Both calls produce identical plots for this data
- [ ] **D.** Neither call is correct — `ax.barh` should be used for horizontal fills

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `fill_between(depth, concentration)` treats `depth` as the x-axis values and `concentration` as the y-axis values, filling below the curve. This would plot depth horizontally and concentration vertically — wrong for a depth profile (depth should be vertical).
- **Option B (Correct):** `fill_betweenx(depth, concentration)` treats `depth` as the y-axis values and `concentration` as the x-axis values, filling the region between the concentration curve and `x=0`. This correctly places depth on the vertical axis and concentration on the horizontal axis — the standard orientation for depth profiles in oceanography, atmospheric science, and geophysics.
- **Option C (Incorrect):** The two calls produce completely different plot orientations. `fill_between` creates a vertically-oriented fill (curve sweeps across x), while `fill_betweenx` creates a horizontally-oriented fill (curve sweeps across y). They are not equivalent.
- **Option D (Incorrect):** `ax.barh` is for categorical or discrete bar charts, not for continuous filled area curves. `fill_betweenx` is the correct function for continuous horizontal fill along a vertical axis.

</details>

---

### 46. Question

A developer computes daily accumulated rainfall using `np.cumsum` and plots it as a cumulative area chart:

```python
daily_rain = np.array([3, 7, 0, 12, 5, 0, 0, 8, 15, 4])  # mm per day
days = np.arange(1, 11)
cumulative = np.cumsum(daily_rain)

plt.fill_between(days, cumulative, alpha=0.4, color="steelblue")
plt.plot(days, cumulative, "b-", lw=2)
plt.ylabel("Cumulative Rainfall (mm)")
```

What are the cumulative rainfall values plotted at days 4, 5, and 6?

- [ ] **A.** day 4=12, day 5=5, day 6=0
- [ ] **B.** day 4=22, day 5=27, day 6=27
- [ ] **C.** day 4=22, day 5=5, day 6=0
- [ ] **D.** day 4=12, day 5=17, day 6=17

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** These are the daily (non-cumulative) values on those days, not the cumulative totals. `np.cumsum` computes running totals, not individual daily values.
- **Option B (Correct):** `np.cumsum([3,7,0,12,5,0,0,8,15,4])` computes: day1=3, day2=10, day3=10, day4=22, day5=27, day6=27, day7=27, day8=35, day9=50, day10=54. At day 4: 3+7+0+12=22. At day 5: 22+5=27. At day 6: 27+0=27 (flat — no rain on day 6). The cumulative curve never decreases (since all daily values are ≥0) and stays flat on days with zero rainfall.
- **Option C (Incorrect):** day4=22 is correct, but days 5 and 6 show daily values (5 and 0) rather than cumulative totals.
- **Option D (Incorrect):** day4=12 is the daily value at day 4, not the cumulative. The cumulative at day 4 should be 3+7+0+12=22. Day5=17 is also incorrect.

</details>

---

### 47. Question

A data analyst visualises weekly cumulative sales for two products:

```python
weeks = np.arange(1, 9)
product_A_weekly = np.array([20, 35, 28, 42, 18, 55, 30, 48])
product_B_weekly = np.array([15, 22, 40, 18, 35, 28, 45, 32])

cum_A = np.cumsum(product_A_weekly)
cum_B = np.cumsum(product_B_weekly)

plt.fill_between(weeks, cum_A, alpha=0.4, color="coral", label="Product A")
plt.fill_between(weeks, cum_B, alpha=0.4, color="teal", label="Product B")
```

At week 4, what are the cumulative totals for Product A and Product B?

- [ ] **A.** A=42, B=18
- [ ] **B.** A=125, B=95
- [ ] **C.** A=83, B=40 (wrong)
- [ ] **D.** A=42, B=95

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** These are the individual weekly values at week 4 (day 4 in the array), not cumulative totals. `np.cumsum` sums all preceding values.
- **Option B (Correct):** `cum_A` at week 4 = 20+35+28+42 = 125. `cum_B` at week 4 = 15+22+40+18 = 95. By week 4, Product A has cumulatively sold 125 units and Product B has sold 95 units. Note that the fills overlap significantly in this region since both baselines start at y=0.
- **Option C (Incorrect):** cum_A=83 = 20+35+28 (only 3 weeks). cum_B=40 = 15+22+40-37 — this is not a valid cumulative computation. Neither value is correct for week 4.
- **Option D (Incorrect):** cum_A=42 is the week-4 individual value, not the cumulative. cum_B=95 is the correct cumulative for B but paired with the wrong (non-cumulative) value for A.

</details>

---

### 48. Question

A developer plots a cumulative area chart of server requests handled per hour:

```python
hours = np.arange(0, 24)
requests = np.array([120, 95, 80, 75, 90, 140, 280, 420, 510,
                     490, 475, 430, 380, 410, 395, 460, 500,
                     520, 480, 430, 360, 300, 240, 180])
cumulative = np.cumsum(requests)

plt.fill_between(hours, cumulative, color="indigo", alpha=0.35)
plt.plot(hours, cumulative, "purple", lw=2)
plt.xlabel("Hour of Day")
plt.ylabel("Total Requests Handled")
```

What is the visual shape of the cumulative fill curve expected to be?

- [ ] **A.** A symmetric bell curve peaking at noon
- [ ] **B.** A monotonically increasing curve that becomes steeper during peak hours and flatter during off-peak hours
- [ ] **C.** A curve that rises and falls with the hourly request count
- [ ] **D.** A flat horizontal line at the mean request rate

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** A cumulative sum is always non-decreasing (since all request counts are positive). It cannot form a symmetric bell — it rises continuously and never falls.
- **Option B (Correct):** `np.cumsum` of a non-negative array always produces a monotonically non-decreasing sequence. The slope of the cumulative curve at any hour equals the hourly request count. During peak hours (when hundreds of requests per hour are added), the cumulative curve rises steeply. During off-peak hours (80–120 requests), the curve rises slowly. The result is an s-shaped or hockey-stick-shaped upward curve that accumulates to the total daily request count at hour 23.
- **Option C (Incorrect):** The raw `requests` array rises and falls with hourly traffic. But `np.cumsum(requests)` is always non-decreasing — the curve only grows, it never drops, even during slow hours. The "rising and falling" characteristic applies to the derivative (hourly rate) not the cumulative.
- **Option D (Incorrect):** A flat line would require constant hourly request counts. The filled area would only be flat if the cumulative curve were flat — but cumsum of any positive array is strictly increasing.

</details>

---

### 49. Question

A developer uses `fill_between` to visualise the area between `np.cumsum` of two data series:

```python
t = np.arange(0, 8)
series_A = np.array([10, 15, 8, 20, 12, 18, 25, 10])
series_B = np.array([8,  12, 10, 15, 16, 14, 20, 12])

cum_A = np.cumsum(series_A)   # [10, 25, 33, 53, 65, 83, 108, 118]
cum_B = np.cumsum(series_B)   # [8,  20, 30, 45, 61, 75, 95,  107]

plt.fill_between(t, cum_A, cum_B, color="gold", alpha=0.4,
                 label="Cumulative Gap A-B")
```

At t=6 (index 6), what is the height of the shaded gap between the two cumulative curves?

- [ ] **A.** 5 units (25 - 20)
- [ ] **B.** 13 units (108 - 95)
- [ ] **C.** 8 units (108 - 95 — recalculate: 108-95=13)
- [ ] **D.** 3 units (108 - 95 — not correct)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** 5 units = 25−20 corresponds to t=1 (index 1), not t=6. cum_A[1]=25, cum_B[1]=20.
- **Option B (Correct):** At t=6 (index 6): cum_A[6]=10+15+8+20+12+18+25=108. cum_B[6]=8+12+10+15+16+14+20=95. The height of the fill at t=6 is `cum_A[6] - cum_B[6] = 108 - 95 = 13` units.
- **Option C (Incorrect):** Option C states "8 units" with the same calculation showing 108−95=13, which is internally contradictory. The correct value is 13.
- **Option D (Incorrect):** 3 units is also stated with the same calculation, which is also incorrect. The gap is 13 units.

</details>

---

### 50. Question

A data scientist creates a cumulative area chart for a model's training error convergence:

```python
epochs = np.arange(1, 21)
epoch_loss = 2.5 * np.exp(-0.2 * epochs) + 0.1

# Incorrect cumulative approach the developer considers:
cumulative_loss = np.cumsum(epoch_loss)

plt.fill_between(epochs, cumulative_loss, color="tomato", alpha=0.4)
```

Why is plotting `np.cumsum(epoch_loss)` for a training loss curve likely **not meaningful** or informative?

- [ ] **A.** `np.cumsum` cannot be applied to floating-point arrays
- [ ] **B.** Training loss values are meant to be compared epoch-by-epoch at their actual scale; cumulating them produces an ever-increasing total loss sum that grows even as the model converges — this number is not a useful training diagnostic
- [ ] **C.** The exponential array causes `np.cumsum` to overflow
- [ ] **D.** `fill_between` cannot accept `np.cumsum` output as an argument

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `np.cumsum` works perfectly with floating-point arrays. There is no restriction to integer types.
- **Option B (Correct):** Training loss is a per-epoch metric representing how well the model fits the training data at that point. It starts high and should decrease as training progresses. Plotting the cumulative sum of training loss produces an always-increasing curve (since all loss values are positive), which grows even as the model converges. This cumulative metric has no standard interpretation in machine learning and provides no useful insight into convergence or overfitting. Cumulative area charts are appropriate for metrics like cumulative revenue or total requests — quantities where the running total is meaningful. For loss curves, plotting the actual epoch-by-epoch values (potentially with a confidence band from multiple runs) is the standard approach.
- **Option C (Incorrect):** `np.exp(-0.2 * epochs)` produces values between 0 and 1. After adding 0.1, all values are small floats. Cumulating 20 values in the range [0.1, 2.6] gives a total of at most ~25 — no risk of floating-point overflow.
- **Option D (Incorrect):** `fill_between` accepts any valid NumPy array as `y1`. The output of `np.cumsum` is a valid array and is fully compatible with `fill_between`.

</details>
