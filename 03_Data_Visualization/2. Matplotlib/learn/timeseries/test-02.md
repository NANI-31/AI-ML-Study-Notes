# Matplotlib Time Series & Temporal Visualization — Set 02

---

### Question 11

A quantitative analyst visualizes two years of monthly sales data. They want major tick marks to appear strictly on the first day of every month ($2025	ext{-}01	ext{-}01$, $2025	ext{-}02	ext{-}01$, etc.).

Which locator from `matplotlib.dates` accomplishes this?

- [ ] A) `ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))`
- [ ] B) `ax.xaxis.set_major_locator(mdates.CalendarMonth(start=1))`
- [ ] C) `ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))`
- [ ] D) `ax.xaxis.set_major_locator(mdates.FixedMonthLocator(12))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))`

**Explanation:**

- **A is incorrect:** `DayLocator(interval=30)` places ticks every 30 days. Because calendar months vary between 28, 30, and 31 days, ticks will rapidly drift away from the 1st of each month.
- **B is incorrect:** `CalendarMonth` does not exist in `matplotlib.dates`.
- **C is correct:** `matplotlib.dates.MonthLocator(bymonth=None, bymonthday=1, interval=1)` is specifically designed to snap ticks to calendar month boundaries. Setting `interval=1` ensures a major tick mark is positioned on the 1st day of every month regardless of varying month lengths or leap years.
- **D is incorrect:** `FixedMonthLocator` is a fabricated class name not present in Matplotlib.

</details>

---

### Question 12

An economic research bureau plots GDP data spanning four decades ($1985	ext{--}2025$). They want major ticks placed exclusively at the start of each decade (every 10 years: 1990, 2000, 2010, 2020).

Which locator configuration configures this decadal step?

- [ ] A) `ax.xaxis.set_major_locator(mdates.DecadeLocator(step=10))`
- [ ] B) `ax.xaxis.set_major_locator(mdates.EpochLocator(10))`
- [ ] C) `ax.xaxis.set_major_locator(mdates.YearLocator(interval=1))`
- [ ] D) `ax.xaxis.set_major_locator(mdates.YearLocator(base=10))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.xaxis.set_major_locator(mdates.YearLocator(base=10))`

**Explanation:**

- **A is incorrect:** `DecadeLocator` is not a valid class in `matplotlib.dates`.
- **B is incorrect:** `EpochLocator` does not exist in Matplotlib.
- **C is incorrect:** `YearLocator(interval=1)` places ticks every single year, creating 40 dense tick marks rather than decadal milestones.
- **D is correct:** `matplotlib.dates.YearLocator(base=10, month=1, day=1)` places ticks at multiples of `base` years. Setting `base=10` positions major ticks on January 1 of years that are clean multiples of 10 (e.g., 1990, 2000, 2010, 2020), providing an uncluttered macro view across multi-decade horizons.

</details>

---

### Question 13

A clinical research study monitors weekly patient metrics over 6 months. An analyst wants major ticks to snap specifically to **every Monday**.

Which locator class and argument achieves this alignment?

- [ ] A) `ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))`
- [ ] B) `ax.xaxis.set_major_locator(mdates.DayLocator(day_of_week="Monday"))`
- [ ] C) `ax.xaxis.set_major_locator(mdates.MondayLocator())`
- [ ] D) `ax.xaxis.set_major_locator(mdates.WeeklyStep(start="MO"))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))`

**Explanation:**

- **A is correct:** `matplotlib.dates.WeekdayLocator(byweekday=mdates.MO, interval=1)` places ticks on specific calendar days of the week. Matplotlib provides weekday constants (`mdates.MO`, `mdates.TU`, `mdates.WE`, `mdates.TH`, `mdates.FR`, `mdates.SA`, `mdates.SU`, corresponding to integers 0 through 6). Setting `byweekday=mdates.MO` locks every tick to calendar Mondays.
- **B is incorrect:** `DayLocator` accepts numeric day-of-the-month intervals, not day-of-week string names.
- **C is incorrect:** `MondayLocator` does not exist in Matplotlib.
- **D is incorrect:** `WeeklyStep` is a fictitious class name.

</details>

---

### Question 14

What is the crucial operational difference between `mdates.DayLocator(interval=7)` and `mdates.WeekdayLocator(byweekday=mdates.MO)`?

- [ ] A) There is no operational difference; they are direct bytecode aliases.
- [ ] B) `DayLocator(interval=7)` steps strictly by 7 calendar days starting from the 1st of each month (e.g., days 1, 8, 15, 22), meaning ticks reset at month boundaries and do not fall on consistent weekdays; `WeekdayLocator(byweekday=mdates.MO)` snaps strictly to calendar Mondays across month boundaries.
- [ ] C) `WeekdayLocator` only works for business days and deletes Saturday/Sunday data points from the canvas.
- [ ] D) `DayLocator` creates minor ticks, whereas `WeekdayLocator` can only create major ticks.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `DayLocator(interval=7)` steps strictly by 7 calendar days starting from the 1st of each month (e.g., days 1, 8, 15, 22), meaning ticks reset at month boundaries and do not fall on consistent weekdays; `WeekdayLocator(byweekday=mdates.MO)` snaps strictly to calendar Mondays across month boundaries.

**Explanation:**

- **A is incorrect:** They produce completely different tick coordinate arrays.
- **B is correct:** `DayLocator(interval=7)` steps through days of the month ($1, 8, 15, 22, \dots$). When a new month begins, the sequence resets to day 1 (so after day 29, the next tick is day 1 of the next month, an interval of only 2 or 3 days!). Furthermore, day 1 can be any weekday. In contrast, `WeekdayLocator` follows continuous 7-day cyclical calendar weeks, guaranteeing that every tick falls precisely on a Monday without resetting at month ends.
- **C is incorrect:** `WeekdayLocator` only places tick marks; it never deletes or modifies plotted data points.
- **D is incorrect:** Both locators can be attached as either major or minor locators.

</details>

---

### Question 15

An operations engineer monitors server CPU utilization across a 48-hour incident. They want major tick marks to appear every 6 hours ($00:00, 06:00, 12:00, 18:00$).

Which locator configuration accomplishes this?

- [ ] A) `ax.xaxis.set_major_locator(mdates.TimeLocator(hours=6))`
- [ ] B) `ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=360))`
- [ ] C) `ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))`
- [ ] D) `ax.xaxis.set_major_locator(mdates.SixHourLocator())`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))`

**Explanation:**

- **A is incorrect:** `TimeLocator` does not exist in `matplotlib.dates`.
- **B is incorrect:** While 360 minutes equals 6 hours, `MinuteLocator` operates within hour boundaries and cannot step across multi-hour blocks cleanly.
- **C is correct:** `matplotlib.dates.HourLocator(byhour=None, interval=6)` places ticks at regular hour intervals. By setting `interval=6`, ticks are positioned at 00:00, 06:00, 12:00, and 18:00 every day, providing clean four-per-day checkpoints.
- **D is incorrect:** `SixHourLocator` is a fictitious class name.

</details>

---

### Question 16

In an interactive data dashboard where users pan and zoom across time spans ranging from several years down to individual minutes, hardcoding a static locator like `MonthLocator()` causes ticks to either clump into solid ink or vanish completely.

Which dynamic locator automatically evaluates the visible axis span and selects the optimal date tick resolution on the fly?

- [ ] A) `mdates.AdaptiveDateLocator()`
- [ ] B) `mdates.SmartDateLocator()`
- [ ] C) `mdates.FlexibleDateLocator()`
- [ ] D) `mdates.AutoDateLocator()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `mdates.AutoDateLocator()`

**Explanation:**

- **A is incorrect:** `AdaptiveDateLocator` is a non-existent class in `matplotlib.dates`.
- **B is incorrect:** `SmartDateLocator` does not exist in Matplotlib.
- **C is incorrect:** `FlexibleDateLocator` is a fabricated name.
- **D is correct:** `matplotlib.dates.AutoDateLocator()` is Matplotlib's default dynamic locator for temporal axes. It automatically inspects the current view limits and switches its underlying generator between years, months, days, hours, minutes, and seconds to ensure an aesthetically pleasing, legible number of tick intervals (typically between 5 and 10 ticks) regardless of zoom level.

</details>

---

### Question 17

An engineer designs a detailed quarterly financial dashboard. They want:
- **Major ticks** placed at the start of each calendar month
- **Minor ticks** placed at weekly intervals (every Monday)

Which sequence of calls establishes this dual-resolution hierarchy?

- [ ] A)
  ```python
  ax.xaxis.set_major_locator(mdates.MonthLocator())
  ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
```
- [ ] B)
  ```python
  ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
  ax.xaxis.set_minor_locator(mdates.MonthLocator())
```
- [ ] C)
  ```python
  ax.set_date_levels(major="month", minor="week")
```
- [ ] D)
  ```python
  ax.xaxis.set_locators(major=30, minor=7)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A)
```python
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
```

**Explanation:**

- **A is correct:** Major ticks are governed by `set_major_locator()` and minor ticks by `set_minor_locator()`. Attaching `mdates.MonthLocator()` to the major locator and `mdates.WeekdayLocator(byweekday=mdates.MO)` to the minor locator cleanly establishes the requested visual hierarchy.
- **B is incorrect:** Reverses the hierarchy, placing major ticks on Mondays and minor ticks on months.
- **C is incorrect:** `set_date_levels` is a non-existent method on `Axes`.
- **D is incorrect:** `set_locators` does not exist on `XAxis`.

</details>

---

### Question 18

To configure quarterly ticks (January, April, July, October) using `matplotlib.dates.MonthLocator`, which argument specification correctly restricts tick marks to those specific months?

- [ ] A) `mdates.MonthLocator(quarterly=True)`
- [ ] B) `mdates.MonthLocator(bymonth=[1, 4, 7, 10])`
- [ ] C) `mdates.MonthLocator(step='quarter')`
- [ ] D) `mdates.MonthLocator(frequency=4)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `mdates.MonthLocator(bymonth=[1, 4, 7, 10])`

**Explanation:**

- **A is incorrect:** `quarterly` is an unrecognized keyword argument for `MonthLocator`.
- **B is correct:** The `bymonth` parameter of `mdates.MonthLocator` accepts an integer or sequence of integers representing the target calendar months (1 for January through 12 for December). Passing `bymonth=[1, 4, 7, 10]` restricts tick mark placement strictly to Q1, Q2, Q3, and Q4 starts.
- **C is incorrect:** `step='quarter'` is not valid syntax in Matplotlib.
- **D is incorrect:** `frequency=4` is not an argument accepted by `MonthLocator`.

</details>

---

### Question 19

A developer writes the following snippet on an axis containing three months of daily datetime data:

```python
from matplotlib.ticker import MultipleLocator
ax.xaxis.set_major_locator(MultipleLocator(30))
```

What unexpected visual behavior occurs?

- [ ] A) Matplotlib throws a fatal `DataCastException` and closes the figure.
- [ ] B) Ticks always snap cleanly to the first day of every month.
- [ ] C) Ticks appear at rigid 30.0-day intervals, causing tick dates to slowly drift out of phase with calendar month boundaries (e.g., landing on Jan 1, Jan 31, Mar 2, Apr 1).
- [ ] D) The dates are converted to binary format.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Ticks appear at rigid 30.0-day intervals, causing tick dates to slowly drift out of phase with calendar month boundaries (e.g., landing on Jan 1, Jan 31, Mar 2, Apr 1).

**Explanation:**

- **A is incorrect:** Matplotlib does not crash; `MultipleLocator` operates on raw coordinate floats.
- **B is incorrect:** `MultipleLocator(30)` has no knowledge of Gregorian calendar variable month lengths.
- **C is correct:** `MultipleLocator(30)` simply places ticks every 30.0 units. Because Matplotlib's date coordinates are float days, ticks will appear at fixed 30-day steps ($0, 30, 60, 90$). Since January has 31 days and February has 28 or 29 days, fixed 30-day steps quickly decouple from the 1st of each month, producing awkward, non-standard dates. Calendar-aware locators like `MonthLocator()` must be used instead.
- **D is incorrect:** No binary conversion takes place.

</details>

---

### Question 20

Which of the following statements regarding `matplotlib.dates` locators are **correct**?

1. `YearLocator(base=5)` places major ticks every 5 years.
2. `MonthLocator(bymonthday=15)` positions ticks on the 15th of each month.
3. `WeekdayLocator(byweekday=mdates.FR)` positions ticks strictly on Fridays.
4. `AutoDateLocator` permanently locks the axis to microsecond resolution regardless of the data span.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 1, 2, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is completely false because `AutoDateLocator` adapts dynamically from years down to seconds rather than locking to microseconds; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 3 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all accurate features of `matplotlib.dates`. `YearLocator(base=5)` handles 5-year intervals, `MonthLocator(bymonthday=15)` targets mid-month points, and `WeekdayLocator` targets specific days of the week.

</details>

---
