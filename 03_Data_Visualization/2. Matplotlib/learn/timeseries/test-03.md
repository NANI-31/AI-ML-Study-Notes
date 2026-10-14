# Matplotlib Time Series & Temporal Visualization — Set 03

---

### Question 21

In `matplotlib.dates.DateFormatter`, which format directive represents the **abbreviated English month name** (e.g., `"Jan"`, `"Feb"`, `"Mar"`)?

- [ ] A) `%b`
- [ ] B) `%B`
- [ ] C) `%m`
- [ ] D) `%M`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `%b`

**Explanation:**

- **A is correct:** In standard Python `strftime` syntax used by `matplotlib.dates.DateFormatter`, `%b` generates the abbreviated month name (e.g., `"Jan"`, `"Feb"`, `"Mar"`).
- **B is incorrect:** `%B` generates the full month name (e.g., `"January"`, `"February"`).
- **C is incorrect:** `%m` produces the zero-padded 2-digit month number (e.g., `"01"`, `"02"`, `"12"`).
- **D is incorrect:** `%M` produces zero-padded minutes (e.g., `"00"`, `"59"`), a common point of confusion for developers.

</details>

---

### Question 22

A data journalist wants x-axis tick labels to display the three-letter month abbreviation on the top line and the four-digit year directly beneath it on a second line (e.g., "Jan\n2026"):

```text
 Jan     Feb     Mar
2026    2026    2026
```

Which `DateFormatter` format string implements this multi-line layout?

- [ ] A) `mdates.DateFormatter("%b--%Y")`
- [ ] B) `mdates.DateFormatter("%b	%Y")`
- [ ] C) `mdates.DateFormatter("%b\n%Y")`
- [ ] D) `mdates.DateFormatter("%b/split/%Y")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `mdates.DateFormatter("%b\n%Y")`

**Explanation:**

- **A is incorrect:** `"%b--%Y"` prints a horizontal hyphenated string on a single line (e.g., `"Jan--2026"`).
- **B is incorrect:** `	` inserts a horizontal tab space, keeping the text on one line.
- **C is correct:** In Matplotlib's text rendering engine, newline characters `\n` are fully evaluated inside `DateFormatter` strings. Placing `\n` between `"%b"` and `"%Y"` creates a two-line label where the month sits on top and the year sits underneath, significantly reducing horizontal text width and preventing tick label collisions.
- **D is incorrect:** `"/split/"` is literal text and does not create a line break.

</details>

---

### Question 23

An IoT monitoring dashboard tracks sensor telemetry sampled every second. An engineer wants the x-axis to render 24-hour clock timestamps with hours, minutes, and seconds (e.g., `"14:35:08"`).

Which format string produces this output?

- [ ] A) `mdates.DateFormatter("%I:%M:%S %p")`
- [ ] B) `mdates.DateFormatter("%H:%M:%S")`
- [ ] C) `mdates.DateFormatter("%h:%m:%s")`
- [ ] D) `mdates.DateFormatter("%T")` without standard directives

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `mdates.DateFormatter("%H:%M:%S")`

**Explanation:**

- **A is incorrect:** `%I` is for 12-hour clock time (01 to 12) paired with `%p` (AM/PM), which would produce `"02:35:08 PM"`.
- **B is correct:** In `strftime` specifications, `%H` represents the 24-hour hour (00 to 23), `%M` represents minutes (00 to 59), and `%S` represents seconds (00 to 59). Combining them as `"%H:%M:%S"` yields clean, standard 24-hour timestamps like `"14:35:08"`.
- **C is incorrect:** `%h` is an alias for `%b` (abbreviated month), `%m` is month number, and lowercase `%s` is not universally supported standard strftime syntax for seconds in all Python platforms.
- **D is incorrect:** `%T` is not portable across all C library backends in standard Python strftime parsing.

</details>

---

### Question 24

Traditional `DateFormatter` strings can produce redundant, cluttered labels during deep zoom levels (e.g., repeating the year and month `"2026-03-15 10:00"`, `"2026-03-15 10:15"`, `"2026-03-15 10:30"` on every tick).

Which modern Matplotlib class (introduced in v3.1) dynamically adapts date formatting across zoom levels and automatically places shared context (such as the year and month) in the axis corner offset, leaving only the granular changes on the tick labels?

- [ ] A) `matplotlib.dates.SmartDateFormatter`
- [ ] B) `matplotlib.dates.MinimalDateFormatter`
- [ ] C) `matplotlib.dates.AdaptiveDateFormatter`
- [ ] D) `matplotlib.dates.ConciseDateFormatter`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `matplotlib.dates.ConciseDateFormatter`

**Explanation:**

- **A is incorrect:** `SmartDateFormatter` does not exist in `matplotlib.dates`.
- **B is incorrect:** `MinimalDateFormatter` is a fictitious class name.
- **C is incorrect:** `AdaptiveDateFormatter` is not a valid Matplotlib class.
- **D is correct:** `matplotlib.dates.ConciseDateFormatter(locator)` is specifically designed to eliminate visual clutter on datetime axes. Instead of repeating identical year and month strings on every tick mark, it shows only the relevant changing components (e.g., just the hours and minutes), while shifting the common macro-context (e.g., `"2026 Mar"`) to the corner of the axis as offset text, dramatically improving chart readability.

</details>

---

### Question 25

A developer writes the following code expecting x-axis tick labels to show `"2026-03-15"`:

```python
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%M-%d"))
```

When the plot is displayed, the labels show `"2026-00-15"` on every single day of March. What is the root cause of this bug?

- [ ] A) The developer mistakenly used `%M` (which represents minutes, `00` to `59`) instead of `%m` (which represents the calendar month, `01` to `12`).
- [ ] B) Matplotlib date coordinates zero out all months by default unless March is specified in uppercase.
- [ ] C) The Pandas `DatetimeIndex` was corrupted by the graphics renderer.
- [ ] D) `%d` overrides the month index if placed after `%M`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) The developer mistakenly used `%M` (which represents minutes, `00` to `59`) instead of `%m` (which represents the calendar month, `01` to `12`).

**Explanation:**

- **A is correct:** In `strftime` tokens, case sensitivity is critical: uppercase `%M` formats the minute of the hour, whereas lowercase `%m` formats the month of the year. Because midnight timestamps have minute equal to 0, `%M` formats as `"00"` for every day, producing `"2026-00-15"`. Correcting the string to `"%Y-%m-%d"` resolves the issue immediately.
- **B is incorrect:** Month numbers are never zeroed out by Matplotlib.
- **C is incorrect:** The data is intact; only the string formatting token was incorrect.
- **D is incorrect:** The order of directives does not cause token substitution side-effects.

</details>

---

### Question 26

An analyst requires custom single-letter day abbreviations (`"M"`, `"T"`, `"W"`, `"T"`, `"F"`) along the horizontal axis of a weekly productivity tracker.

Which approach correctly formats the axis with these custom single-letter labels?

- [ ] A) `ax.xaxis.set_major_formatter(mdates.DateFormatter("%single_day"))`
- [ ] B) `ax.set_xticklabels(["M", "T", "W", "T", "F"], mode="cycle")`
- [ ] C)
  ```python
  from matplotlib.ticker import FuncFormatter

  ax.xaxis.set_major_formatter(
      FuncFormatter(lambda x, pos: mdates.num2date(x).strftime("%a")[0])
  )
```
- [ ] D) `ax.xaxis.date_format(letters=1)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C)
```python
from matplotlib.ticker import FuncFormatter

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: mdates.num2date(x).strftime("%a")[0])
)
```

**Explanation:**

- **A is incorrect:** `%single_day` is not a valid strftime directive in Python or Matplotlib.
- **B is incorrect:** `mode="cycle"` is not a valid argument for `set_xticklabels()`.
- **C is correct:** `FuncFormatter` accepts a custom callback `func(x, pos)`. Here, `x` is the float date coordinate. Calling `mdates.num2date(x)` converts the float into a Python `datetime`, `.strftime("%a")` extracts the 3-letter weekday name (e.g., `"Mon"`, `"Tue"`), and `[0]` takes the first character, producing clean single-letter labels (`"M"`, `"T"`).
- **D is incorrect:** `date_format(letters=1)` is a fictitious method that raises an `AttributeError`.

</details>

---

### Question 27

What is the difference between the `%Y` and `%y` format directives when used in `mdates.DateFormatter`?

- [ ] A) `%Y` formats the day of the year (1 to 366), while `%y` formats the calendar year.
- [ ] B) `%Y` formats the four-digit year with century (e.g., `"2026"`), while `%y` formats the two-digit year without century (e.g., `"26"`).
- [ ] C) `%Y` is for leap years, while `%y` is for non-leap years.
- [ ] D) `%Y` formats UTC years, while `%y` formats local years.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `%Y` formats the four-digit year with century (e.g., `"2026"`), while `%y` formats the two-digit year without century (e.g., `"26"`).

**Explanation:**

- **A is incorrect:** Day of the year is formatted using `%j`.
- **B is correct:** In `strftime` standards, uppercase `%Y` outputs the complete four-digit calendar year (including century, e.g., `"2026"`). Lowercase `%y` outputs only the last two decimal digits without century (e.g., `"26"`).
- **C is incorrect:** Leap year status has no bearing on `%Y` vs `%y`.
- **D is incorrect:** Timezone handling is independent of century token formatting.

</details>

---

### Question 28

When configuring `matplotlib.dates.ConciseDateFormatter`, how must it be initialized and attached to an axis?

- [ ] A) `ax.xaxis.set_concise(True)`
- [ ] B) `ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter())` without any arguments
- [ ] C) `ax.xaxis.set_major_formatter("concise")`
- [ ] D)
  ```python
  locator = mdates.AutoDateLocator()
  formatter = mdates.ConciseDateFormatter(locator)
  ax.xaxis.set_major_locator(locator)
  ax.xaxis.set_major_formatter(formatter)
```

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D)
```python
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
```

**Explanation:**

- **A is incorrect:** `set_concise(True)` does not exist on `XAxis`.
- **B is incorrect:** `ConciseDateFormatter` requires the active locator as a positional argument; calling it with no arguments raises a `TypeError`.
- **C is incorrect:** Passing string literal `"concise"` is not supported by `set_major_formatter()`.
- **D is correct:** `matplotlib.dates.ConciseDateFormatter` requires the active `DateLocator` (typically `AutoDateLocator`) passed as its first argument upon instantiation. This allows the formatter to inspect what level of date ticks the locator has chosen (years, months, days, hours) and dynamically adjust string formatting and offset text accordingly.

</details>

---

### Question 29

An audio processing engineer logs sound pressure levels at millisecond intervals across 2 seconds. They want tick labels to display seconds and fractional milliseconds (e.g., `"12.450s"`).

Which format string directive in `DateFormatter` represents fractional seconds / microseconds?

- [ ] A) `%f`
- [ ] B) `%ms`
- [ ] C) `%u`
- [ ] D) `%F`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `%f`

**Explanation:**

- **A is correct:** In Python `strftime` syntax, `%f` formats the microsecond component as a 6-digit decimal string (`000000` to `999999`). Combining it like `"%S.%f"` produces seconds with sub-second decimal precision.
- **B is incorrect:** `%ms` is not a valid strftime directive in standard Python.
- **C is incorrect:** `%u` formats the weekday number (1 for Monday through 7 for Sunday), not microseconds.
- **D is incorrect:** `%F` is a shorthand for `%Y-%m-%d` in certain C libraries, not fractional seconds.

</details>

---

### Question 30

Which of the following statements regarding `matplotlib.dates` formatters are **correct**?

1. `%B` outputs the full month name (e.g., `"October"`), while `%b` outputs the abbreviated name (e.g., `"Oct"`).
2. `%m` produces zero-padded month numbers ($01	ext{--}12$), whereas `%d` produces zero-padded day numbers ($01	ext{--}31$).
3. `DateFormatter` can only format years; days and hours require `TimeFormatter`.
4. `ConciseDateFormatter` reduces string clutter by pushing macro time components to the corner offset.

- [ ] A) 1 and 3 only
- [ ] B) 2 and 3 only
- [ ] C) 1, 2, and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) 1, 2, and 4 only

**Explanation:**

- **A is incorrect:** Includes false Statement 3 and omits Statement 2.
- **B is incorrect:** Includes false Statement 3 and omits Statement 1.
- **C is correct:** Statements 1, 2, and 4 are all true. `%B` vs `%b` controls full vs abbreviated month names; `%m` and `%d` format month and day numbers; and `ConciseDateFormatter` offloads common macro components to the corner offset.
- **D is incorrect:** Statement 3 is completely false; `DateFormatter` handles all time units from centuries down to microseconds via standard `strftime` directives; there is no separate `TimeFormatter` in `matplotlib.dates`.

</details>

---
