# Matplotlib Time Series & Temporal Visualization — Set 01

---

### Question 1

A machine learning engineer evaluates daily inference requests by plotting a time series from a Pandas DataFrame:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
    "requests": [1200, 1450, 980, 1600],
})

fig, ax = plt.subplots()
ax.plot(df["date"], df["requests"])
```

Because the `date` column contains raw Python strings rather than datetime objects, what structural flaw occurs on the horizontal axis?

- [ ] A) Matplotlib treats the dates as discrete categorical strings, placing every single entry at integer indices $0, 1, 2, \dots$ without a continuous temporal timeline, which breaks chronological date math, calendar interval spacing, and date locators.
- [ ] B) Matplotlib automatically crashes with a `TypeError: Cannot render string as temporal coordinate`.
- [ ] C) Matplotlib automatically parses the ISO strings into UTC timestamps and applies `AutoDateLocator` by default.
- [ ] D) Matplotlib renders the strings backwards in alphabetical order from right to left.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Matplotlib treats the dates as discrete categorical strings, placing every single entry at integer indices $0, 1, 2, \dots$ without a continuous temporal timeline, which breaks chronological date math, calendar interval spacing, and date locators.

**Explanation:**

- **A is correct:** When raw strings like `"2026-01-01"` are passed to `ax.plot()`, Matplotlib routes them through its categorical unit converter. Each unique string is assigned an arbitrary discrete integer index ($0, 1, 2, \dots$). Consequently, Matplotlib has no awareness that these strings represent calendar dates: missing days are not spaced out, temporal locators (`MonthLocator`, `DayLocator`) fail completely, and panning or zooming cannot calculate time intervals. The column must first be converted via `pd.to_datetime(df["date"])`.
- **B is incorrect:** Matplotlib does not crash; it silently treats strings as categorical variables and plots them.
- **C is incorrect:** Matplotlib does not automatically parse arbitrary string columns into datetime objects; explicit conversion is required.
- **D is incorrect:** Categorical strings are assigned coordinates in order of appearance in the data array, not reversed alphabetically.

</details>

---

### Question 2

How does Matplotlib internally represent calendar dates and times along a numerical coordinate axis?

- [ ] A) As integer seconds elapsed since the Unix epoch (`1970-01-01 00:00:00 UTC`), identical to standard POSIX timestamps.
- [ ] B) As floating-point numbers representing the number of days (and fractional parts of a day) elapsed since `0001-01-01 00:00:00 UTC` plus 1, based on the proleptic Gregorian calendar.
- [ ] C) As 64-bit nanosecond integers matching NumPy's `datetime64[ns]` internal structure.
- [ ] D) As Julian Day Numbers offset by 2,400,000.5 days.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) As floating-point numbers representing the number of days (and fractional parts of a day) elapsed since `0001-01-01 00:00:00 UTC` plus 1, based on the proleptic Gregorian calendar.

**Explanation:**

- **A is incorrect:** Unix timestamps measure elapsed seconds since 1970, whereas Matplotlib's date coordinate system is day-based with an epoch starting at year 1.
- **B is correct:** In Matplotlib's temporal architecture (governed by `matplotlib.dates`), all dates are converted into standard floating-point numbers where $1.0$ corresponds to `0001-01-01 00:00:00 UTC`. For example, noon on a given day adds $0.5$ to the float. This unified float representation allows standard 2D Cartesian transformation pipelines, locators, and view limits to operate on time exactly like continuous numerical coordinates.
- **C is incorrect:** While Pandas and NumPy use nanoseconds since 1970 (`datetime64[ns]`), Matplotlib converts them to its day-based floating-point system before rendering.
- **D is incorrect:** Matplotlib does not use astronomical Julian Day Numbers.

</details>

---

### Question 3

Which pair of functions in `matplotlib.dates` handles two-way conversion between Python `datetime` / Pandas `Timestamp` objects and Matplotlib's internal float-day representation?

- [ ] A) `mdates.to_matplotlib()` and `mdates.from_matplotlib()`
- [ ] B) `mdates.dt_to_coord()` and `mdates.coord_to_dt()`
- [ ] C) `mdates.date2num()` and `mdates.num2date()`
- [ ] D) `mdates.timestamp_to_float()` and `mdates.float_to_timestamp()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `mdates.date2num()` and `mdates.num2date()`

**Explanation:**

- **A is incorrect:** `to_matplotlib()` and `from_matplotlib()` do not exist in `matplotlib.dates`.
- **B is incorrect:** `dt_to_coord()` and `coord_to_dt()` are non-existent functions in the Matplotlib API.
- **C is correct:** `matplotlib.dates.date2num(d)` takes standard Python `datetime.datetime` objects, `datetime.date` objects, or sequences thereof and converts them into Matplotlib floating-point date numbers. Conversely, `matplotlib.dates.num2date(x)` takes a Matplotlib floating-point coordinate and converts it back into a timezone-aware Python `datetime.datetime` object.
- **D is incorrect:** `timestamp_to_float()` and `float_to_timestamp()` are fictitious function names.

</details>

---

### Question 4

A data scientist creates a sequence of timestamps using NumPy:

```python
import numpy as np
import matplotlib.pyplot as plt

dates = np.arange("2026-01-01", "2026-04-01", dtype="datetime64[D]")
values = np.random.randn(len(dates))

fig, ax = plt.subplots()
ax.plot(dates, values)
```

How does modern Matplotlib (v3.0+) handle NumPy `datetime64` arrays passed to `ax.plot()`?

- [ ] A) It raises a `ValueError` because NumPy arrays must always be cast to Python lists of strings.
- [ ] B) It automatically casts the dates to integer timestamps in seconds, losing calendar month boundaries.
- [ ] C) It treats `datetime64` as raw categorical data unless `ax.set_dates(True)` is explicitly invoked.
- [ ] D) It seamlessly recognizes `datetime64` through Matplotlib's unit conversion registry, automatically converting them into Matplotlib date coordinates and attaching default temporal formatters.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It seamlessly recognizes `datetime64` through Matplotlib's unit conversion registry, automatically converting them into Matplotlib date coordinates and attaching default temporal formatters.

**Explanation:**

- **A is incorrect:** Matplotlib natively supports NumPy `datetime64` arrays without requiring conversion to string lists.
- **B is incorrect:** It does not convert them to raw second integers; it converts them to Matplotlib's float day format.
- **C is incorrect:** Matplotlib's units registry intercepts `np.datetime64` automatically without needing `ax.set_dates()`.
- **D is correct:** Matplotlib includes a built-in unit conversion framework (`matplotlib.units.registry`). When a NumPy array with dtype `datetime64` is passed, the registered converter automatically transforms the data into Matplotlib float dates, sets the axis unit to dates, and applies appropriate temporal locators and formatters.

</details>

---

### Question 5

A financial analyst plots 250 trading days of stock closing prices directly from a CSV file where the dates were read as string literals:

```python
# df['Date'] contains 250 strings: ['2025-01-02', '2025-01-03', ...]
ax.plot(df['Date'], df['Close'])
```

Upon rendering, the horizontal axis appears completely pitch-black and unreadable with thousands of overlapping labels. What is the fundamental cause of this bug?

- [ ] A) Because the dates are strings, Matplotlib treats every single row as an independent categorical label and renders 250 individual text labels side-by-side along the bottom spine, causing total visual overlap.
- [ ] B) Matplotlib defaults to 250 DPI rendering when plotting currency values.
- [ ] C) The font cache has corrupted the text renderer, printing solid black bounding boxes.
- [ ] D) The graphics card GPU ran out of texture memory while trying to calculate date deltas.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Because the dates are strings, Matplotlib treats every single row as an independent categorical label and renders 250 individual text labels side-by-side along the bottom spine, causing total visual overlap.

**Explanation:**

- **A is correct:** Categorical axes display a tick mark and text label for *every unique category* in the dataset. Because 250 distinct date strings were passed, Matplotlib dutifully renders 250 text labels crowded into a few inches of horizontal canvas space, producing a solid black band of overlapping glyphs. Converting the column to true datetimes via `pd.to_datetime(df['Date'])` allows temporal locators (`MonthLocator`) to place only 6 to 10 clean milestone ticks.
- **B is incorrect:** DPI settings govern canvas resolution, not categorical label spacing.
- **C is incorrect:** The black band is caused by colliding text characters, not font cache corruption.
- **D is incorrect:** Matplotlib uses CPU rasterization/vector backends by default, not GPU texture memory.

</details>

---

### Question 6

When working with timezone-aware datetime series (e.g., UTC vs. US/Eastern) in Pandas and Matplotlib, which statement accurately describes Matplotlib's timezone handling?

- [ ] A) Matplotlib strictly strips all timezone information and assumes all timestamps are local computer system time.
- [ ] B) Matplotlib converts timezone-aware timestamps to UTC internally; date locators and formatters can accept a `tz` parameter (e.g., `pytz.timezone` or `zoneinfo.ZoneInfo`) to display labels in a target local timezone.
- [ ] C) Matplotlib raises a `TimeZoneConflictError` if timestamps have an explicit timezone offset.
- [ ] D) Matplotlib automatically adjusts the physical width of the figure canvas based on the GMT offset.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Matplotlib converts timezone-aware timestamps to UTC internally; date locators and formatters can accept a `tz` parameter (e.g., `pytz.timezone` or `zoneinfo.ZoneInfo`) to display labels in a target local timezone.

**Explanation:**

- **A is incorrect:** Matplotlib does not strip timezones arbitrarily; it respects timezone information during numeric conversion.
- **B is correct:** When timezone-aware `datetime` objects or Pandas `DatetimeIndex` series are passed to Matplotlib, coordinates are normalized to UTC days. To format and place ticks according to a local geographic timezone (such as Wall Street market hours), classes like `mdates.DateFormatter` and `mdates.AutoDateLocator` accept a `tz` argument (e.g., `tz=ZoneInfo("America/New_York")`), ensuring tick boundaries snap to midnight in the local time zone.
- **C is incorrect:** Timezone-aware datetimes are supported natively without raising errors.
- **D is incorrect:** Timezone offsets never alter figure dimensions.

</details>

---

### Question 7

High-frequency algorithmic trading systems log order book events at microsecond resolution. An engineer plots timestamps spanning just 50 milliseconds:

```python
import pandas as pd
timestamps = pd.date_range("2026-03-15 09:30:00", periods=50000, freq="us")
```

What technical challenge arises when visualizing sub-millisecond timestamps using Matplotlib's default float-day date system?

- [ ] A) Matplotlib automatically rounds all timestamps to the nearest calendar day, rendering a single vertical point.
- [ ] B) The proleptic Gregorian calendar does not mathematically support divisions of time smaller than one second.
- [ ] C) Because Matplotlib represents dates as float days since year 1, by year 2026 the integer day number is $pprox 739,600$; adding microsecond increments ($1.157 	imes 10^{-11}$ days) approaches the 53-bit limit of standard IEEE 754 double-precision floats, leading to floating-point precision loss and tick jitter.
- [ ] D) Matplotlib converts microsecond datetimes into complex numbers with imaginary components.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Because Matplotlib represents dates as float days since year 1, by year 2026 the integer day number is $pprox 739,600$; adding microsecond increments ($1.157 	imes 10^{-11}$ days) approaches the 53-bit limit of standard IEEE 754 double-precision floats, leading to floating-point precision loss and tick jitter.

**Explanation:**

- **A is incorrect:** Matplotlib does not round dates to whole days; floating-point fractions are preserved.
- **B is incorrect:** Fractional days can mathematically represent any time increment.
- **C is correct:** In IEEE 754 double-precision floating point, numbers have 53 bits of significand precision ($pprox 15	ext{--}17$ decimal digits). A date in 2026 is $pprox 7.396 	imes 10^5$ days. One microsecond is $1 / 86,400,000,000 pprox 1.157 	imes 10^{-11}$ days. Adding $10^{-11}$ to $10^5$ requires 16 digits of precision, hitting the absolute limits of 64-bit floats. For sub-millisecond visualization, best practice is to plot elapsed milliseconds or microseconds relative to a reference start time ($t - t_0$).
- **D is incorrect:** Floating point precision degradation causes rounding errors, not complex numbers.

</details>

---

### Question 8

An analyst wishes to create a synthetic 90-day time series using Pandas with business-day frequency (excluding weekends) starting on January 1, 2026:

```python
import pandas as pd
dates = pd.date_range("2026-01-01", periods=90, freq=___)
```

Which frequency code string generates business-day timestamps?

- [ ] A) `freq="W"`
- [ ] B) `freq="D"`
- [ ] C) `freq="M"`
- [ ] D) `freq="B"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `freq="B"`

**Explanation:**

- **A is incorrect:** `"W"` generates weekly intervals (e.g., every Sunday).
- **B is incorrect:** `"D"` generates calendar-day intervals, including Saturdays and Sundays.
- **C is incorrect:** `"M"` generates month-end frequency intervals.
- **D is correct:** In Pandas frequency aliases, `"B"` denotes Business day frequency (Monday through Friday), skipping weekend calendar days. When plotted on a continuous date axis in Matplotlib, note that these missing weekend days will naturally appear as small horizontal gaps unless index-based plotting is used.

</details>

---

### Question 9

A developer plots a time series using `ax.plot(dates, values)` where `dates` is a `pd.date_range`. If they immediately execute `xmin, xmax = ax.get_xlim()`, what data types and values are returned?

- [ ] A) Two Python `float` values representing the start and end coordinates in Matplotlib's internal days-since-year-1 format (e.g., numbers around $739600.0$).
- [ ] B) Two `pd.Timestamp` objects representing the exact calendar dates.
- [ ] C) Two ISO 8601 string representations (e.g., `'2026-01-01'`).
- [ ] D) Two integer UNIX timestamps counting seconds since 1970.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Two Python `float` values representing the start and end coordinates in Matplotlib's internal days-since-year-1 format (e.g., numbers around $739600.0$).

**Explanation:**

- **A is correct:** Regardless of whether inputs were Python `datetime`, NumPy `datetime64`, or Pandas `Timestamp`, Matplotlib converts them into internal float coordinates during data ingestion. Calling `ax.get_xlim()` retrieves the raw data limits of the axis transform, which are floating-point numbers in days since year 1 (e.g., $pprox 739600.0$ for early 2026). To convert these back into human-readable datetime objects, one passes them to `matplotlib.dates.num2date()`.
- **B is incorrect:** `get_xlim()` returns raw axis floats, not Pandas `Timestamp` objects.
- **C is incorrect:** Coordinate transforms do not store string representations.
- **D is incorrect:** Matplotlib's epoch is year 1, not the 1970 Unix epoch.

</details>

---

### Question 10

Which of the following statements regarding temporal data types in Matplotlib are **correct**?

1. Passing a Pandas `DatetimeIndex` directly to `ax.plot()` automatically registers temporal unit converters.
2. Passing raw string dates causes Matplotlib to treat the axis as categorical, rendering an individual tick for every row.
3. `matplotlib.dates.date2num()` converts calendar datetime objects into Matplotlib's internal float-day representation.
4. Matplotlib requires all datetime datasets to be stored in an SQL database before plotting.

- [ ] A) 1 and 4 only
- [ ] B) 1, 2, and 3 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is completely false; Matplotlib works with in-memory Python, NumPy, and Pandas structures.
- **B is correct:** Statements 1, 2, and 3 are all correct. Statement 1 is true because Pandas datetime objects trigger the unit conversion machinery. Statement 2 is true because strings become categorical coordinates with individual ticks. Statement 3 is true because `date2num` is the canonical converter to float days.
- **C is incorrect:** Excludes Statement 1 and Statement 3, while including false Statement 4.
- **D is incorrect:** Excludes Statement 1 and Statement 2, while including false Statement 4.

</details>

---
