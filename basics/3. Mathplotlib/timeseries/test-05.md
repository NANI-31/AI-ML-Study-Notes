# Matplotlib Time Series & Temporal Visualization — Set 05

---

### Question 41

An analyst wants to zoom in on Q1 of 2026 by explicitly locking the horizontal view limits from January 1, 2026 to March 31, 2026.

Which command sets these temporal coordinate limits cleanly using Pandas or Python datetime objects?

- [ ] A) `ax.set_xlim("Q1-2026")`
- [ ] B) `ax.set_dates("2026-01-01", "2026-03-31")`
- [ ] C) `ax.crop_time(start=1, end=90)`
- [ ] D) `ax.set_xlim(pd.to_datetime("2026-01-01"), pd.to_datetime("2026-03-31"))` (or `datetime.date(2026, 1, 1), datetime.date(2026, 3, 31)`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.set_xlim(pd.to_datetime("2026-01-01"), pd.to_datetime("2026-03-31"))` (or `datetime.date(2026, 1, 1), datetime.date(2026, 3, 31)`)

**Explanation:**

- **A is incorrect:** `set_xlim("Q1-2026")` is not a valid coordinate specification in Matplotlib.
- **B is incorrect:** `set_dates()` is a non-existent method on `Axes`.
- **C is incorrect:** `crop_time()` is not a valid Matplotlib method.
- **D is correct:** In Matplotlib, `ax.set_xlim(left, right)` accepts `pd.Timestamp`, `datetime.date`, or `datetime.datetime` objects when the axis is configured with temporal unit converters. Matplotlib automatically translates the datetime inputs into its internal float-day coordinates and updates the viewport boundaries accordingly.

</details>

---

### Question 42

What happens if an engineer passes raw string literals `ax.set_xlim("2026-01-01", "2026-03-31")` to an axis where datetime data was previously plotted?

- [ ] A) Matplotlib crashes immediately with a `FatalCanvasError`.
- [ ] B) In modern Matplotlib, if date unit converters are registered on the axis, Matplotlib attempts to parse standard ISO date strings automatically; however, if conversion fails or strings are ambiguous, it raises a `ConversionError` or misinterprets them as categorical coordinates, making explicit `pd.to_datetime()` best practice.
- [ ] C) The plot is converted into a pie chart.
- [ ] D) The view limits are set to zero.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) In modern Matplotlib, if date unit converters are registered on the axis, Matplotlib attempts to parse standard ISO date strings automatically; however, if conversion fails or strings are ambiguous, it raises a `ConversionError` or misinterprets them as categorical coordinates, making explicit `pd.to_datetime()` best practice.

**Explanation:**

- **A is incorrect:** Matplotlib does not throw `FatalCanvasError` when parsing strings.
- **B is correct:** While recent versions of Matplotlib include automatic string-to-date conversion helpers when date units are active on an axis, passing raw strings can introduce subtle bugs if formats are non-standard, locale-dependent, or if converters have not yet initialized. Supplying explicit `pd.to_datetime()` or `datetime.date()` objects guarantees unambiguous coordinate translation.
- **C is incorrect:** String limits do not transform a plot into a pie chart.
- **D is incorrect:** Limits are not set to zero on string inputs.

</details>

---

### Question 43

An automated daily monitoring script visualizes all data starting from the launch date (`2026-01-01`) through the latest available data, without hardcoding the end date.

How can the developer lock the left boundary at January 1, 2026 while allowing the right boundary to autoscale dynamically?

- [ ] A) `ax.set_xlim(start="2026-01-01", end="auto")`
- [ ] B) `ax.set_xlim(pd.to_datetime("2026-01-01"), float('inf'))`
- [ ] C) `ax.set_xlim(left=pd.to_datetime("2026-01-01"), right=None)`
- [ ] D) `ax.set_xlim(pd.to_datetime("2026-01-01"), "max")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.set_xlim(left=pd.to_datetime("2026-01-01"), right=None)`

**Explanation:**

- **A is incorrect:** `end="auto"` is not a supported keyword argument in `set_xlim()`.
- **B is incorrect:** Passing `float('inf')` creates invalid transform matrices and breaks coordinate rendering.
- **C is correct:** In `ax.set_xlim(left, right)`, passing `None` for either parameter instructs Matplotlib to preserve the existing limit or autoscale that specific bound to match the data extent. Setting `left=pd.to_datetime("2026-01-01")` and `right=None` locks the start at the launch date while letting the right boundary expand automatically as new days are appended.
- **D is incorrect:** `"max"` is not a recognized parameter in `set_xlim()`.

</details>

---

### Question 44

An intraday energy trading chart displays 1-minute power grid prices for March 15, 2026. The engineer wants the view restricted strictly to market trading hours: 09:30 AM to 04:00 PM.

Which command sets these intraday boundaries?

- [ ] A) `ax.set_xlim(pd.Timestamp("2026-03-15 09:30:00"), pd.Timestamp("2026-03-15 16:00:00"))`
- [ ] B) `ax.set_xlim(time(9, 30), time(16, 0))` without a date component
- [ ] C) `ax.set_hours(9.5, 16.0)`
- [ ] D) `ax.set_xlim(0930, 1600)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.set_xlim(pd.Timestamp("2026-03-15 09:30:00"), pd.Timestamp("2026-03-15 16:00:00"))`

**Explanation:**

- **A is correct:** Because Matplotlib's date coordinate axis maps continuous calendar days, intraday points require complete timestamp objects containing both the calendar date and the time of day. Passing `pd.Timestamp("2026-03-15 09:30:00")` and `pd.Timestamp("2026-03-15 16:00:00")` converts them into fractional float days, correctly framing the market session.
- **B is incorrect:** `datetime.time` objects lack date information and cannot be converted into absolute calendar coordinates.
- **C is incorrect:** `set_hours` is not a valid method on `Axes`.
- **D is incorrect:** Military time integer `0930` is interpreted as year 930 AD, not a time of day.

</details>

---

### Question 45

By default, Matplotlib adds a 5% margin of empty whitespace on both ends of an axis. An analyst wants the time-series line to touch the left and right vertical spines exactly at the first and last timestamps in the dataset.

Which command eliminates this horizontal whitespace padding?

- [ ] A) `ax.set_padding(0)`
- [ ] B) `ax.remove_margins(horizontal=True)`
- [ ] C) `ax.tight_dates()`
- [ ] D) `ax.autoscale(enable=True, axis='x', tight=True)` (or `ax.margins(x=0)`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.autoscale(enable=True, axis='x', tight=True)` (or `ax.margins(x=0)`)

**Explanation:**

- **A is incorrect:** `set_padding()` is not a method on `Axes`.
- **B is incorrect:** `remove_margins()` is a non-existent method in Matplotlib.
- **C is incorrect:** `tight_dates()` does not exist in the Matplotlib API.
- **D is correct:** Setting `ax.margins(x=0)` or calling `ax.autoscale(enable=True, axis='x', tight=True)` sets the horizontal data padding factor to zero. This ensures the view limits snap tightly to `min(dates)` and `max(dates)`, causing the data series to extend from the left spine to the right spine without empty gap margins.

</details>

---

### Question 46

An engineer wants to add a subtle 2% buffer of whitespace before the start date and after the end date to ensure endpoint marker bubbles are not clipped by the spines.

Which method call applies this specific proportional buffer?

- [ ] A) `ax.set_xlim(pad=0.02)`
- [ ] B) `ax.margins(x=0.02)`
- [ ] C) `ax.xaxis.buffer(0.02)`
- [ ] D) `ax.expand(percent=2)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.margins(x=0.02)`

**Explanation:**

- **A is incorrect:** `pad` is not a keyword argument for `set_xlim()`.
- **B is correct:** `Axes.margins(x=margin)` sets the padding fraction applied to data limits along the horizontal axis before autoscaling. Specifying `x=0.02` expands the left and right view limits by $2\%$ of the total time span, providing clean breathing room for endpoint markers.
- **C is incorrect:** `xaxis.buffer()` is not a valid method.
- **D is incorrect:** `expand(percent=2)` does not exist on `Axes`.

</details>

---

### Question 47

What is the fundamental architectural difference between filtering a Pandas DataFrame before plotting (`ax.plot(df.loc['2026-01':'2026-03'])`) versus setting view limits on the full dataset (`ax.set_xlim(...)`)?

- [ ] A) Filtering the DataFrame deletes rows from the original CSV file on the hard drive.
- [ ] B) Setting view limits permanently deletes out-of-bounds data from computer memory.
- [ ] C) Filtering the DataFrame restricts plotted data points so the y-axis autoscales strictly to the min/max values of Q1, whereas plotting the entire dataset and calling `ax.set_xlim()` retains all data in memory, allowing interactive panning across the full history (though the y-axis may reflect global data extent unless autoscaled).
- [ ] D) There is no difference in memory, rendering speed, or y-axis autoscaling.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Filtering the DataFrame restricts plotted data points so the y-axis autoscales strictly to the min/max values of Q1, whereas plotting the entire dataset and calling `ax.set_xlim()` retains all data in memory, allowing interactive panning across the full history (though the y-axis may reflect global data extent unless autoscaled).

**Explanation:**

- **A is incorrect:** Filtering a DataFrame in memory never alters or deletes disk CSV files.
- **B is incorrect:** Setting view limits never deletes data points from memory.
- **C is correct:** When a sliced DataFrame is plotted, only Q1 data points are ingested into Matplotlib; y-axis autoscaling evaluates only Q1 values. When the full dataset is plotted and `set_xlim()` is applied, all data points remain stored in the `Line2D` artist, enabling smooth interactive panning/zooming, but the y-axis may initially reflect the global vertical scale unless `ax.autoscale(axis='y')` is refreshed.
- **D is incorrect:** Memory footprint, rendering workload, and autoscale scopes differ substantially.

</details>

---

### Question 48

A data scientist creates a plot with UTC-localized timestamps (`pd.date_range("2026-01-01", periods=10, tz="UTC")`). If they attempt to set view limits using a timezone-naive timestamp:

```python
ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"))  # Naive!
```

What issue can arise?

- [ ] A) Depending on the Pandas and Matplotlib versions, comparing or setting limits between timezone-naive and timezone-aware timestamps can raise a `TypeError: Cannot compare tz-naive and tz-aware timestamps` or offset the view by the local machine's UTC offset.
- [ ] B) The chart will automatically invert the y-axis.
- [ ] C) Matplotlib converts the time to binary seconds.
- [ ] D) The figure canvas is cleared.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Depending on the Pandas and Matplotlib versions, comparing or setting limits between timezone-naive and timezone-aware timestamps can raise a `TypeError: Cannot compare tz-naive and tz-aware timestamps` or offset the view by the local machine's UTC offset.

**Explanation:**

- **A is correct:** In modern Python and Pandas, comparing or computing deltas between timezone-aware and timezone-naive timestamps is strictly disallowed and raises a `TypeError`. When an axis has been populated with UTC data, any timestamps passed to `set_xlim()` should match the timezone (e.g., `pd.Timestamp("2026-01-01", tz="UTC")`) to prevent conversion ambiguity or crash exceptions.
- **B is incorrect:** Timezone discrepancies do not cause vertical axis inversion.
- **C is incorrect:** Coordinates are not converted to binary strings.
- **D is incorrect:** Timezone differences do not clear the figure canvas.

</details>

---

### Question 49

How does Matplotlib's internal float-day calendar coordinate system handle leap days (e.g., February 29, 2024 or February 29, 2028)?

- [ ] A) It skips February 29 entirely and maps it to March 1.
- [ ] B) It halts execution and prompts the user for calendar confirmation.
- [ ] C) It assigns February 29 a negative coordinate value.
- [ ] D) It accurately models the proleptic Gregorian calendar, assigning February 29 a full, distinct integer day coordinate in leap years.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It accurately models the proleptic Gregorian calendar, assigning February 29 a full, distinct integer day coordinate in leap years.

**Explanation:**

- **A is incorrect:** Matplotlib never skips February 29 in leap years.
- **B is incorrect:** Matplotlib calculates coordinates mathematically without prompting users.
- **C is incorrect:** Coordinates are positive proleptic Gregorian day numbers.
- **D is correct:** Matplotlib's `matplotlib.dates` engine is built on the proleptic Gregorian calendar rules matching Python's `datetime` module. Leap years (years divisible by 4, except century years not divisible by 400) receive 366 distinct day coordinates, ensuring accurate mathematical interval spacing across February 29.

</details>

---

### Question 50

Which of the following statements regarding temporal view limits in Matplotlib are **correct**?

1. `ax.set_xlim()` accepts `pd.Timestamp` and `datetime.date` objects directly on date axes.
2. Setting `left=None` retains the current lower coordinate bound while adjusting only the right bound.
3. `ax.margins(x=0)` eliminates horizontal whitespace padding between data endpoints and spines.
4. Setting temporal view limits permanently deletes out-of-view data points from the underlying Pandas DataFrame.

- [ ] A) 1 and 4 only
- [ ] B) 1, 2, and 3 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is completely false because setting view limits never deletes rows from caller DataFrames; Statements 2 and 3 are also omitted.
- **B is correct:** Statements 1, 2, and 3 are all correct. `set_xlim` takes datetime objects; `None` leaves bounds unchanged; and `margins(x=0)` removes padding.
- **C is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **D is incorrect:** Statement 4 is false, and Statements 1 and 2 are omitted.

</details>

---
