# Matplotlib Time Series & Temporal Visualization — Set 07

---

### Question 61

An engineer visualizes noisy daily sensor data over 365 days. To highlight the underlying seasonal trend without discarding high-frequency volatility, they want to overlay a 7-day rolling average.

Which Pandas and Matplotlib code snippet computes and plots this rolling curve?

- [ ] A) `ax.plot(df.index, df['sensor'].mean(7))`
- [ ] B)
  ```python
  df["rolling_7d"] = df["sensor"].rolling(window=7).mean()
  ax.plot(df.index, df["sensor"], alpha=0.3, color="gray", label="Raw")
  ax.plot(df.index, df["rolling_7d"], color="blue", lw=2, label="7-Day MA")
```
- [ ] C) `ax.plot(df.index, df['sensor'].smooth(7))`
- [ ] D) `ax.rolling_plot(df['sensor'], window=7)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B)
```python
df["rolling_7d"] = df["sensor"].rolling(window=7).mean()
ax.plot(df.index, df["sensor"], alpha=0.3, color="gray", label="Raw")
ax.plot(df.index, df["rolling_7d"], color="blue", lw=2, label="7-Day MA")
```

**Explanation:**

- **A is incorrect:** `Series.mean(7)` is invalid syntax; `mean()` takes an axis parameter, not a window size.
- **B is correct:** `Series.rolling(window=7).mean()` calculates the 7-period trailing moving average. Plotting the raw measurements with low opacity (`alpha=0.3`, `color="gray"`) and the smoothed line with bold styling (`color="blue"`, `lw=2`) establishes an intuitive visual hierarchy where overall trajectory stands out clearly over daily noise.
- **C is incorrect:** `Series.smooth()` is not a valid Pandas Series method.
- **D is incorrect:** `Axes.rolling_plot()` does not exist in Matplotlib.

</details>

---

### Question 62

What is the fundamental difference between a **trailing** rolling window (`df['val'].rolling(7).mean()`) and a **centered** rolling window (`df['val'].rolling(7, center=True).mean()`) when visualizing retrospective historical data?

- [ ] A) A trailing window averages the current day and the previous 6 days, introducing a visual lag (phase shift) where trend peaks appear delayed in time; a centered window averages 3 days before, the current day, and 3 days after, aligning peaks with their true historical calendar dates.
- [ ] B) A centered window only works on weekend dates.
- [ ] C) A trailing window deletes negative values from the array.
- [ ] D) A centered window converts the plot into a Polar projection.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) A trailing window averages the current day and the previous 6 days, introducing a visual lag (phase shift) where trend peaks appear delayed in time; a centered window averages 3 days before, the current day, and 3 days after, aligning peaks with their true historical calendar dates.

**Explanation:**

- **A is correct:** In time-series analysis, trailing windows calculate $rac{1}{W} \sum_{i=0}^{W-1} x_{t-i}$, which inherently lags behind rapid changes by approximately $W/2$ periods. In retrospective data analysis (where future data within the historical record is already known), setting `center=True` shifts the window center to $t$, aligning moving average inflection points directly with the actual underlying peaks and troughs of the original curve.
- **B is incorrect:** Window centering applies to any frequency and is not restricted to weekends.
- **C is incorrect:** Neither window type filters or alters negative values.
- **D is incorrect:** Projections are completely unrelated to moving average window placement.

</details>

---

### Question 63

For real-time streaming dashboards where recent observations should carry more weight than distant historical points, which smoothing technique applies an exponentially decreasing weighting factor?

- [ ] A) `df['val'].cumulative_mean()`
- [ ] B) `df['val'].boxcar(10)`
- [ ] C) `df['val'].shift(10).mean()`
- [ ] D) `df['val'].ewm(span=30).mean()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `df['val'].ewm(span=30).mean()`

**Explanation:**

- **A is incorrect:** Cumulative mean weights all past observations equally.
- **B is incorrect:** `boxcar` is not a standard Pandas Series method for exponential smoothing.
- **C is incorrect:** Shifting simply delays the series in time without exponential weighting.
- **D is correct:** Pandas `ewm(span=30).mean()` computes the Exponentially Weighted Moving Average (EWMA). By assigning geometrically decreasing weights to older observations ($w_k = (1 - lpha)^k$), EWMA reacts more quickly to recent market regime shifts or sensor anomalies than a rigid rectangular rolling window.

</details>

---

### Question 64

In data visualization theory (e.g., Edward Tufte's graphical integrity guidelines), what color and opacity pairing best communicates the relationship between raw volatility and smoothed trend?

- [ ] A) Bright red solid line for raw data, bright neon green solid line for smoothed data.
- [ ] B) Both lines plotted in solid black with identical line width.
- [ ] C) Muted, low-opacity gray or light-colored thin stroke (`alpha=0.25`, `lw=1`) for the raw noisy series, beneath a vibrant, high-contrast, thicker stroke (`alpha=1.0`, `lw=2.5`) for the smoothed curve.
- [ ] D) Flashing dashed lines with checkerboard patterns.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Muted, low-opacity gray or light-colored thin stroke (`alpha=0.25`, `lw=1`) for the raw noisy series, beneath a vibrant, high-contrast, thicker stroke (`alpha=1.0`, `lw=2.5`) for the smoothed curve.

**Explanation:**

- **A is incorrect:** Competing saturated complementary colors cause visual vibration and eye fatigue.
- **B is incorrect:** Identical solid black lines make it impossible to distinguish raw noise from smoothed trend.
- **C is correct:** Subordinating the raw observations using faint alpha (`alpha=0.25`) preserves the context of variance and extreme spikes without distracting from the macro trend represented by the bold, solid smoothed line.
- **D is incorrect:** Checkerboard patterns introduce chartjunk and degrade readability.

</details>

---

### Question 65

A financial risk officer visualizes a Bollinger Band volatility envelope across 100 trading days:
- Moving average: $\mu_t = 	ext{rolling}(20).	ext{mean}()$
- Standard deviation: $\sigma_t = 	ext{rolling}(20).	ext{std}()$
- Bands: $\mu_t \pm 2\sigma_t$

Which code snippet computes and shades this volatility corridor around the moving average?

- [ ] A) `ax.envelope(mu, 2 * sigma)`
- [ ] B)
  ```python
  mu = df["Close"].rolling(20).mean()
  sigma = df["Close"].rolling(20).std()
  ax.plot(df.index, mu, color="blue", label="20-Day SMA")
  ax.fill_between(
      df.index,
      mu - 2 * sigma,
      mu + 2 * sigma,
      color="blue",
      alpha=0.15,
      label="Volatility Band (2σ)",
  )
```
- [ ] C) `ax.band(mu, sigma, width=2)`
- [ ] D) `ax.plot_bollinger(df['Close'], window=20)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B)
```python
mu = df["Close"].rolling(20).mean()
sigma = df["Close"].rolling(20).std()
ax.plot(df.index, mu, color="blue", label="20-Day SMA")
ax.fill_between(
    df.index,
    mu - 2 * sigma,
    mu + 2 * sigma,
    color="blue",
    alpha=0.15,
    label="Volatility Band (2σ)",
)
```

**Explanation:**

- **A is incorrect:** `ax.envelope()` is not a valid Matplotlib method.
- **B is correct:** `ax.fill_between(x, y1, y2)` shades the 2D polygon bounded vertically between $y_1 = \mu - 2\sigma$ and $y_2 = \mu + 2\sigma$ across timestamps `df.index`. Setting `alpha=0.15` renders a subtle translucent confidence corridor that highlights volatility expansions and contractions.
- **C is incorrect:** `ax.band()` is a non-existent method in the Matplotlib API.
- **D is incorrect:** `plot_bollinger()` is not a built-in Matplotlib method.

</details>

---

### Question 66

When computing a 30-day moving average on a daily series, the first 29 entries in the resulting Pandas Series are `NaN` (null) because fewer than 30 observations precede them.

How does Matplotlib's `ax.plot()` handle these initial `NaN` values?

- [ ] A) Matplotlib automatically ignores `NaN` values, omitting them from the rendering pipeline and smoothly beginning the plotted line at day 30 where valid numerical floats appear.
- [ ] B) Matplotlib crashes immediately with a `NullValueException`.
- [ ] C) Matplotlib replaces all `NaN` values with zero, causing the line to spike from 0.
- [ ] D) Matplotlib converts the entire series to an empty plot.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Matplotlib automatically ignores `NaN` values, omitting them from the rendering pipeline and smoothly beginning the plotted line at day 30 where valid numerical floats appear.

**Explanation:**

- **A is correct:** In Matplotlib, floating-point `NaN` values in data arrays represent discontinuous breaks. When `NaN` entries appear at the beginning (or middle) of a series, Matplotlib's path generator simply lifts the pen, resuming rendering at the first valid non-NaN coordinate. Consequently, the moving average line starts naturally at day 30 without throwing errors or requiring manual slicing.
- **B is incorrect:** Matplotlib does not raise exceptions on `NaN` floats.
- **C is incorrect:** It never synthesizes or coerces `NaN` values to zero.
- **D is incorrect:** Valid coordinates after the `NaN` segment are plotted normally.

</details>

---

### Question 67

What is the fundamental architectural difference between **resampling** a time series (`df.resample('ME').mean()`) versus applying a **rolling window** (`df.rolling(30).mean()`)?

- [ ] A) Resampling only works on integers, whereas rolling works on floats.
- [ ] B) Rolling changes the calendar dates of the index, whereas resampling preserves the original dates.
- [ ] C) Resampling is for Fourier transforms, while rolling is for wavelet transforms.
- [ ] D) Resampling aggregates data into non-overlapping chronological bins, downsampling the frequency (reducing the number of rows to one per month), whereas rolling applies a moving window that retains the original daily index and row count.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Resampling aggregates data into non-overlapping chronological bins, downsampling the frequency (reducing the number of rows to one per month), whereas rolling applies a moving window that retains the original daily index and row count.

**Explanation:**

- **A is incorrect:** Data types are not restricted; both operations work on floating-point numeric data.
- **B is incorrect:** Rolling retains the original index timestamps, whereas resampling defines new bin timestamps.
- **C is incorrect:** Resampling and rolling are temporal grouping and windowing techniques, not Fourier or wavelet transforms.
- **D is correct:** `resample()` acts as a temporal `groupby()`, collapsing multiple raw observations into discrete, non-overlapping buckets (e.g. converting 365 daily points into 12 monthly points). In contrast, `rolling()` slides a window row-by-row across the data, generating a smoothed value for every single timestamp while strictly preserving the continuous daily resolution and row count.

</details>

---

### Question 68

An analyst plots cumulative forecast error over time to assess whether a machine learning model exhibits systematic directional bias (under-predicting or over-predicting).

Which transformation computes this metric across time for plotting?

- [ ] A) `(actual - forecast).rolling(10).mean()`
- [ ] B) `(actual - forecast).pct_change()`
- [ ] C) `(actual - forecast).cumsum()`
- [ ] D) `(actual - forecast).diff()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `(actual - forecast).cumsum()`

**Explanation:**

- **A is incorrect:** Rolling mean measures local average error, not cumulative drift.
- **B is incorrect:** Percentage change computes relative period-over-period growth.
- **C is correct:** `Series.cumsum()` computes the cumulative sum of errors $\sum_{t=1}^T (y_t - \hat{y}_t)$. Plotting this series over time reveals drift: an unbiased model oscillates around zero, whereas an upward or downward slope reveals persistent under-estimation or over-estimation.
- **D is incorrect:** `diff()` computes discrete first differences.

</details>

---

### Question 69

To isolate seasonal intraday patterns in server load, an engineer subtracts the 24-hour centered rolling average from the raw hourly CPU utilization:

```python
detrended = df['cpu'] - df['cpu'].rolling(24, center=True).mean()
ax.plot(df.index, detrended)
```

What does this detrended visualization highlight?

- [ ] A) Long-term multi-year structural changes.
- [ ] B) Short-term cyclical intra-day oscillations and unexpected spike anomalies, with the macro baseline drift removed.
- [ ] C) The total cumulative energy consumed by the server.
- [ ] D) The geographical IP location of server requests.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Short-term cyclical intra-day oscillations and unexpected spike anomalies, with the macro baseline drift removed.

**Explanation:**

- **A is incorrect:** Subtracting the rolling mean removes long-term trends rather than highlighting them.
- **B is correct:** Detrending by subtracting the local moving average eliminates baseline drift (such as gradual multi-week traffic growth), isolating high-frequency cyclic variations (e.g., morning vs night traffic cycles) and transient surge spikes along a stationary zero baseline.
- **C is incorrect:** Energy integration requires mathematical summation over time, not baseline subtraction.
- **D is incorrect:** IP geography cannot be extracted from single-series load subtraction.

</details>

---

### Question 70

Which of the following statements regarding time-series smoothing and rolling statistics are **correct**?

1. `df.rolling(window=7).mean()` calculates a 7-period moving average.
2. `center=True` shifts the moving average window to align with the midpoint of the interval.
3. `ax.fill_between()` can be used to shade volatility bands around a rolling mean curve.
4. `df.rolling()` permanently alters the underlying sampling frequency of the index.

- [ ] A) 1, 2, and 3 only
- [ ] B) 1 and 4 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 1, 2, and 3 only

**Explanation:**

- **A is correct:** Statements 1, 2, and 3 are all correct. Statement 1 describes rolling mean; Statement 2 describes centering; Statement 3 describes volatility shading via `fill_between()`.
- **B is incorrect:** Includes false Statement 4 and omits Statements 2 and 3.
- **C is incorrect:** Includes false Statement 4 and omits Statement 1.
- **D is incorrect:** Includes false Statement 4 and omits Statements 1 and 2.

</details>

---
