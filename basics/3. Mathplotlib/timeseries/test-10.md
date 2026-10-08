# Matplotlib Time Series & Temporal Visualization — Set 10

---

### Question 91

When validating a machine learning forecasting model (e.g. XGBoost or Temporal Fusion Transformer), an engineer splits a 3-year dataset into contiguous chronological partitions:
- Training: $2023	ext{-}01	ext{-}01$ to $2024	ext{-}12	ext{-}31$
- Validation: $2025	ext{-}01	ext{-}01$ to $2025	ext{-}06	ext{-}30$
- Testing: $2025	ext{-}07	ext{-}01$ to $2025	ext{-}12	ext{-}31$

Which visualization pattern most effectively communicates this non-shuffled temporal partitioning?

- [ ] A) A random scatter plot with points colored arbitrarily.
- [ ] B) A pie chart showing the percentage breakdown of dataset rows.
- [ ] C) A continuous timeline plot of the target metric where the background is partitioned into three color-coded contiguous spans using `ax.axvspan()` (e.g., green for Train, amber for Validation, blue for Test), with vertical dashed lines at split boundaries.
- [ ] D) Invert the horizontal axis so test data is plotted before training data.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) A continuous timeline plot of the target metric where the background is partitioned into three color-coded contiguous spans using `ax.axvspan()` (e.g., green for Train, amber for Validation, blue for Test), with vertical dashed lines at split boundaries.

**Explanation:**

- **A is incorrect:** Random scatter plots destroy chronological continuity.
- **B is incorrect:** Pie charts communicate proportions but completely discard the time sequence essential for verifying that no future data leaked into training folds.
- **C is correct:** In ML time series papers, standard practice is plotting the full metric trajectory while using translucent `axvspan` shading across the three chronological regimes (Train, Validation, Test) alongside vertical boundary lines. This immediately demonstrates that temporal causality was preserved and no lookahead leakage occurred.
- **D is incorrect:** Inverting the axis violates time progression conventions.

</details>

---

### Question 92

A quantitative developer plots equity candlestick or daily closing prices spanning six months. When plotted against a standard continuous Matplotlib calendar datetime axis, why do annoying **flat horizontal gaps** appear every 5 trading days?

- [ ] A) The Matplotlib rendering cache runs out of memory on Fridays.
- [ ] B) Financial markets are closed on Saturdays, Sundays, and exchange holidays; because a calendar datetime axis is continuous and linear in time, the missing weekend dates occupy physical horizontal space on the axis, creating flat gaps where zero trading occurred.
- [ ] C) Matplotlib automatically censors market data on weekends due to SEC regulations.
- [ ] D) Financial data requires logarithmic time axes.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Financial markets are closed on Saturdays, Sundays, and exchange holidays; because a calendar datetime axis is continuous and linear in time, the missing weekend dates occupy physical horizontal space on the axis, creating flat gaps where zero trading occurred.

**Explanation:**

- **A is incorrect:** It is a geometric calendar spacing phenomenon, not memory caching.
- **B is correct:** Standard calendar axes assign fixed horizontal space to every day. Because equity markets close on weekends and public holidays, there are no data rows for those days. When lines connect Friday close to Monday open, the line bridges across the empty Saturday/Sunday coordinate span, producing flat gaps and distorted candle widths.
- **C is incorrect:** Regulatory censorship does not cause Matplotlib axis gaps.
- **D is incorrect:** Logarithmic time axes are irrelevant to weekend market closure spacing.

</details>

---

### Question 93

To eliminate artificial weekend gaps in financial trading charts, which technique plots continuous bars while maintaining human-readable date labels?

- [ ] A) Plot the data against uniform integer indices ($0, 1, 2, \dots, N-1$), and use `ax.set_xticks()` and `ax.set_xticklabels()` (or an `IndexFormatter`) to map sparse integer positions back to actual trading date strings.
- [ ] B) Delete all Mondays from the dataset.
- [ ] C) Set `ax.xaxis.set_scale('business_days')`.
- [ ] D) Enable `plt.fill_weekends(mode='interpolate')`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Plot the data against uniform integer indices ($0, 1, 2, \dots, N-1$), and use `ax.set_xticks()` and `ax.set_xticklabels()` (or an `IndexFormatter`) to map sparse integer positions back to actual trading date strings.

**Explanation:**

- **A is correct:** In professional financial plotting libraries (such as `mplfinance`), market data is plotted against ordinal integer indices $0, 1, \dots, N-1$. Because trading days are indexed contiguously, Friday (index $k$) is immediately followed by Monday (index $k+1$) with zero empty gap in between. The x-axis ticks are then placed at regular integer intervals and labeled with formatted date strings corresponding to those specific trading sessions.
- **B is incorrect:** Deleting real trading days destroys data integrity.
- **C is incorrect:** `business_days` scale is not a built-in Matplotlib scale.
- **D is incorrect:** `fill_weekends()` is a fictitious function.

</details>

---

### Question 94

After training a time-series model, a data scientist generates a residual diagnostic plot ($e_t = y_t - \hat{y}_t$ over time). 

Which visual elements are critical on a professional residual time-series diagnostic chart?

- [ ] A) A 3D wireframe mesh.
- [ ] B) Only the positive residuals plotted on a logarithmic scale.
- [ ] C) A pie chart showing the percentage of positive vs negative errors.
- [ ] D) Residual scatter points plotted over time, a horizontal reference baseline at zero (`ax.axhline(0, color='black', linestyle='--')`), and $2\sigma$ control limits (`ax.axhline(2*std, color='red', linestyle=':')`) to detect heteroscedasticity or structural breaks.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) Residual scatter points plotted over time, a horizontal reference baseline at zero (`ax.axhline(0, color='black', linestyle='--')`), and $2\sigma$ control limits (`ax.axhline(2*std, color='red', linestyle=':')`) to detect heteroscedasticity or structural breaks.

**Explanation:**

- **A is incorrect:** 3D meshes add visual distortion without diagnostic value for 2D residuals.
- **B is incorrect:** Omission of negative residuals prevents diagnosing under-predictions.
- **C is incorrect:** Pie charts completely discard the temporal ordering of errors.
- **D is correct:** Evaluating time-series residuals requires checking for stationarity and constant variance (homoscedasticity). Plotting $e_t$ over time with a reference zero line (`axhline(0)`) and $\pm 2\sigma$ threshold lines allows researchers to quickly diagnose variance explosions, autocorrelation clusters, or regime shifts where errors fail to behave as white noise.

</details>

---

### Question 95

In walk-forward time-series cross-validation (rolling or expanding window splits), an engineer wants to visualize the 5 cross-validation folds:

```python
# 5 folds, each with a training segment and a 1-step-ahead validation segment
```

What visual layout is canonical for displaying cross-validation split progressions?

- [ ] A) A single pie chart with 5 slices.
- [ ] B) A radar chart with 5 spokes.
- [ ] C) A horizontal stacked bar or horizontal span chart where the vertical axis represents Fold Number ($1$ through $5$) and the horizontal axis represents the timeline, showing expanding training blocks followed by distinct validation blocks.
- [ ] D) A scatter plot comparing training loss against learning rate.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) A horizontal stacked bar or horizontal span chart where the vertical axis represents Fold Number ($1$ through $5$) and the horizontal axis represents the timeline, showing expanding training blocks followed by distinct validation blocks.

**Explanation:**

- **A is incorrect:** Pie charts cannot represent timeline progressions or chronological split ordering.
- **B is incorrect:** Radar charts are designed for multi-variable profiling, not temporal cross-validation.
- **C is correct:** In Scikit-learn and time-series cross-validation literature (e.g., `TimeSeriesSplit`), the standard diagram places Fold Index on the y-axis ($1$ to $K$) and the timeline on the x-axis. Each fold is represented by horizontal bar spans: a blue bar for the training interval followed immediately by an orange bar for the validation test horizon, visually demonstrating how the training window expands or rolls forward over time without lookahead leakage.
- **D is incorrect:** Scatter plots of loss vs learning rate do not show CV split geometry.

</details>

---

### Question 96

When evaluating autocorrelation structure in time-series residuals, which classical visualization displays the correlation between the series and lagged versions of itself ($r_k = 	ext{Corr}(y_t, y_{t-k})$) across lags $k = 1, 2, \dots, 30$?

- [ ] A) A confusion matrix heatmap.
- [ ] B) An Autocorrelation Function (ACF) plot (or correlogram), rendered as a stem plot (`ax.stem()`) with horizontal dashed lines indicating the $95\%$ Bartlett confidence bounds ($[-1.96/\sqrt{N}, +1.96/\sqrt{N}]$).
- [ ] C) A ROC curve comparing true positive rate to false positive rate.
- [ ] D) A Box-Whisker plot of annual totals.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) An Autocorrelation Function (ACF) plot (or correlogram), rendered as a stem plot (`ax.stem()`) with horizontal dashed lines indicating the $95\%$ Bartlett confidence bounds ($[-1.96/\sqrt{N}, +1.96/\sqrt{N}]$).

**Explanation:**

- **A is incorrect:** Confusion matrices evaluate discrete classification outcomes, not temporal autocorrelation.
- **B is correct:** The ACF plot (correlogram) displays autocorrelation coefficients across successive lag intervals. In Matplotlib / Statsmodels, it is plotted using vertical stems (`ax.stem(lags, acf_vals)`) with a shaded horizontal confidence interval band at $\pm 1.96/\sqrt{N}$. Any stem piercing outside the band indicates statistically significant autocorrelation requiring autoregressive modeling.
- **C is incorrect:** ROC curves evaluate binary classification discrimination, not time series lags.
- **D is incorrect:** Box-Whisker plots show distribution spread, not serial correlation.

</details>

---

### Question 97

An automated ML pipeline generates 1,000 time-series diagnostic plots inside a scheduled night job:

```python
for model_name, ts_data in results.items():
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(ts_data["date"], ts_data["actual"], label="Actual")
  ax.plot(ts_data["date"], ts_data["forecast"], label="Forecast")
  fig.autofmt_xdate()
  fig.savefig(f"{model_name}.png", bbox_inches="tight")
  # End of loop
```

After 300 models, the Python process crashes with `MemoryError`. What crucial cleanup command was omitted from the loop?

- [ ] A) `plt.close(fig)` (or `plt.close('all')`) to explicitly free the figure canvas and deregister it from Matplotlib's global figure manager dictionary.
- [ ] B) `fig.autofmt_xdate(clear=True)`
- [ ] C) `ax.delete_dates()`
- [ ] D) `del model_name`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `plt.close(fig)` (or `plt.close('all')`) to explicitly free the figure canvas and deregister it from Matplotlib's global figure manager dictionary.

**Explanation:**

- **A is correct:** In Matplotlib's `pyplot` interface, every `plt.subplots()` invocation registers a strong reference to the newly created `Figure` in the global module cache (`matplotlib._pylab_helpers.Gcf`). As a result, Python's garbage collector cannot reclaim the memory even after the loop variable `fig` goes out of scope. In a 1,000-iteration loop, this accumulates gigabytes of raster buffers until an Out-Of-Memory crash occurs. Calling `plt.close(fig)` explicitly destroys the figure and releases memory.
- **B is incorrect:** `autofmt_xdate()` does not have a `clear` parameter and does not free figure memory.
- **C is incorrect:** `delete_dates()` is a non-existent method.
- **D is incorrect:** Deleting `model_name` frees trivial string bytes while leaving massive figure buffers in memory.

</details>

---

### Question 98

To visualize day-of-week seasonality (evaluating whether energy consumption surges on Mondays vs. Sundays), which exploratory visualization layout is most effective?

- [ ] A) A single 1D histogram of all dates.
- [ ] B) A 3D scatter plot of latitude and longitude.
- [ ] C) A line plot connecting all timestamps in random order.
- [ ] D) A categorical box plot or violin plot with day names (`["Monday", "Tuesday", ..., "Sunday"]`) on the horizontal axis and energy usage on the vertical axis, created using `df.index.day_name()`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) A categorical box plot or violin plot with day names (`["Monday", "Tuesday", ..., "Sunday"]`) on the horizontal axis and energy usage on the vertical axis, created using `df.index.day_name()`.

**Explanation:**

- **A is incorrect:** 1D date histograms do not group data by cyclical day-of-week categories.
- **B is incorrect:** 3D geographic plots do not visualize temporal weekly rhythms.
- **C is incorrect:** Connecting timestamps in random order destroys temporal meaning.
- **D is correct:** Grouping time series by calendar properties (such as `df.index.day_name()` or `df.index.month`) and rendering categorical boxplots displays median levels, interquartile ranges, and outlier spikes for each weekday side-by-side, immediately exposing weekly operational rhythms.

</details>

---

### Question 99

In web server monitoring, an operations team visualizes user traffic patterns across an entire year as a 2D calendar matrix where columns represent weeks of the year ($1	ext{--}52$) and rows represent days of the week (Monday through Sunday).

Which Matplotlib tool renders this calendar heatmap?

- [ ] A) `ax.plot_calendar()`
- [ ] B) `ax.imshow()` (or `pcolormesh()`) applied to a $7 	imes 52$ pivoted matrix of daily traffic values, accompanied by a colorbar (`fig.colorbar()`).
- [ ] C) `ax.pie(df['traffic'])`
- [ ] D) `ax.stem(weeks, days)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.imshow()` (or `pcolormesh()`) applied to a $7 	imes 52$ pivoted matrix of daily traffic values, accompanied by a colorbar (`fig.colorbar()`).

**Explanation:**

- **A is incorrect:** `plot_calendar()` is not a built-in Matplotlib method.
- **B is correct:** The classic GitHub-style calendar contribution heatmap is constructed by pivoting daily metrics into a $7 	imes 52$ 2D matrix (days of week $	imes$ weeks of year) and rendering the grid using `ax.imshow(matrix, cmap="Greens", aspect="auto")` or `pcolormesh()`. A colorbar maps color intensity to traffic volume.
- **C is incorrect:** Pie charts cannot visualize 2D calendar matrices.
- **D is incorrect:** Stem plots are 1D discrete line markers, not 2D color-mapped heatmaps.

</details>

---

### Question 100

Which of the following statements synthesize industry best practices for time-series visualization in machine learning and data engineering?

1. Dates should always be parsed into true `datetime` or `Timestamp` objects before plotting to enable temporal coordinate spacing and dynamic locators.
2. `fig.autofmt_xdate()` prevents overlapping date labels by applying label rotation, setting `ha='right'`, and adjusting canvas margins.
3. In batch generation pipelines, `plt.close(fig)` is mandatory to prevent unbounded memory growth and process termination.
4. Shaded uncertainty envelopes (`fill_between`) should be assigned lower `zorder` than forecast lines to maintain visual clarity.

- [ ] A) 1 and 3 only
- [ ] B) 2 and 4 only
- [ ] C) 1, 2, 3, and 4
- [ ] D) 1, 2, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 1, 2, 3, and 4

**Explanation:**

- **A is incorrect:** Omit valid best practice statements 2 and 4.
- **B is incorrect:** Omit valid best practice statements 1 and 3.
- **C is correct:** All four statements are foundational principles of enterprise time-series visualization: parsing true datetimes unlocks locators; `autofmt_xdate()` prevents collisions; `plt.close(fig)` guarantees memory stability; and `zorder` layering prevents shaded fills from drowning out forecast lines.
- **D is incorrect:** Omit valid best practice statement 3.

</details>

---
