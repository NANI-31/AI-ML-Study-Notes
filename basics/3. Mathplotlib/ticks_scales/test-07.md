# Matplotlib Ticks & Scales Objective Test — Set 07

---

### Question 61

In the `matplotlib.ticker` architecture, what is the precise technical role of a **`Formatter`** subclass?

- [ ] A) It computes the physical coordinates where spines intersect on the canvas.
- [ ] B) It automatically clips data arrays that exceed the current view limits.
- [ ] C) It accepts a numerical coordinate value $x$ (and optional position index $pos$) and returns the human-readable string rendered as the tick label.
- [ ] D) It calculates bounding boxes for multi-column legends.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) It accepts a numerical coordinate value $x$ (and optional position index $pos$) and returns the human-readable string rendered as the tick label.

**Explanation:**

- **A is incorrect:** Computing spine intersections and axis frame boundaries is handled by the `Spines` and `Bbox` classes.
- **B is incorrect:** Data clipping at view limits is managed by renderer clipping paths and artist clipping properties (`set_clip_on(True)`).
- **C is correct:** While a `Locator` determines *where* tick marks reside along an axis, a `Formatter` (such as `FuncFormatter`, `PercentFormatter`, or `StrMethodFormatter`) determines *what text* is displayed at those coordinates. Its `__call__(x, pos=None)` method receives coordinate $x$ and returns the formatted string rendered on the canvas.
- **D is incorrect:** Legend bounding boxes are computed by `matplotlib.legend.Legend`.

</details>

---

### Question 62

When constructing a custom dynamic formatter using `matplotlib.ticker.FuncFormatter`, what parameter signature must the custom Python callback function accept?

- [ ] A) Exactly two arguments: `(x, pos)` where `x` is the tick coordinate value (float) and `pos` is the tick position index (int or None).
- [ ] B) Exactly one argument: `(x)` representing the data coordinate.
- [ ] C) Three arguments: `(x, y, ax)` providing coordinate pairs and the parent `Axes` instance.
- [ ] D) A single dictionary argument `{'val': x, 'axis': axis}`.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) Exactly two arguments: `(x, pos)` where `x` is the tick coordinate value (float) and `pos` is the tick position index (int or None).

**Explanation:**

- **A is correct:** Matplotlib's `FuncFormatter(func)` requires `func` to adhere to the signature `func(x, pos)`. Here, `x` is the numeric coordinate value of the tick, and `pos` represents the ordinal index/counter of the tick in the currently evaluated tick sequence. Even if the callback logic ignores `pos` (e.g., `lambda x, pos: f"{x:.1f}s"`), accepting `pos` is required to prevent a `TypeError: <lambda>() takes 1 positional argument but 2 were given`.
- **B is incorrect:** Providing a function with signature `func(x)` raises a `TypeError` when called by Matplotlib's renderer.
- **C is incorrect:** `FuncFormatter` does not pass `(x, y, ax)` coordinate tuples to the callback function.
- **D is incorrect:** `FuncFormatter` does not pass dictionary payloads such as `{'val': x, 'axis': axis}`.

</details>

---

### Question 63

A binary classification model's True Positive Rate (Recall) values are continuous decimal fractions in the range $[0.0, 1.0]$. A data scientist wants the y-axis to display clean percentage strings (`"0%"`, `"20%"`, `"40%"`, $\dots$, `"100%"`).

Which ticker class and configuration achieves this without manual string formatting loops?

- [ ] A) `ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100.0))`
- [ ] B) `ax.yaxis.set_major_formatter(matplotlib.ticker.DecimalFormatter('%'))`
- [ ] C) `ax.yaxis.as_percentage()`
- [ ] D) `ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))`

**Explanation:**

- **A is incorrect:** `PercentFormatter(xmax=100.0)` assumes data values are already on a 0 to 100 scale; passing it decimal probabilities like $0.8$ would format them as `"0.8%"`, which is off by a factor of 100!
- **B is incorrect:** `DecimalFormatter` does not exist in `matplotlib.ticker`.
- **C is incorrect:** `as_percentage()` is a non-existent convenience method on `YAxis` objects.
- **D is correct:** `matplotlib.ticker.PercentFormatter(xmax=1.0)` converts numeric values by computing $(x / xmax) \times 100\%$. By specifying `xmax=1.0`, a decimal value of $0.80$ is correctly multiplied and formatted as `"80%"`, perfectly matching probability ranges in $[0.0, 1.0]$.

</details>

---

### Question 64

An engineer wants to format a financial volume axis so that numbers appear with thousands-separating commas (e.g., $1500000$ renders as `"1,500,000"`), without writing a custom Python function.

Which modern string-template formatter accomplishes this using Python 3 format specifications?

- [ ] A) `ax.yaxis.set_major_formatter(TemplateFormatter('{value:comma}'))`
- [ ] B) `ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))`
- [ ] C) `ax.yaxis.set_major_formatter(FormatString('%d,000'))`
- [ ] D) `ax.yaxis.use_commas(True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))`

**Explanation:**

- **A is incorrect:** `TemplateFormatter` does not exist in `matplotlib.ticker`.
- **B is correct:** `matplotlib.ticker.StrMethodFormatter(fmt)` utilizes Python's modern `str.format()` syntax. The variable `x` represents the tick coordinate value, and `pos` represents the tick position index. The format specifier `'{x:,.0f}'` formats `x` as a floating-point number with 0 decimal places and standard comma thousands separators (e.g., $1,500,000$).
- **C is incorrect:** `FormatString` is not a valid Matplotlib formatter; Matplotlib uses `FormatStrFormatter` which uses old C-style `printf` syntax (`%d`), but `%d,000` is literal text and does not provide dynamic thousands comma grouping.
- **D is incorrect:** `ax.yaxis.use_commas()` is a fictitious method.

</details>

---

### Question 65

A high-frequency algorithmic trading dashboard displays revenues in millions of dollars (ranging from $\$10,000,000$ to $\$80,000,000$). 

Which `FuncFormatter` lambda formats coordinates like $25000000$ as `"$25.0M"`?

- [ ] A) `FuncFormatter(lambda x: f"${x}M")`
- [ ] B) `FuncFormatter(lambda x, pos: f"${x / 1000}M")`
- [ ] C) `FuncFormatter(lambda x, pos: f"${x * 1e-6:.1f}M")`
- [ ] D) `FuncFormatter(lambda x, pos: f"${x * 1000000:.1f}M")`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `FuncFormatter(lambda x, pos: f"${x * 1e-6:.1f}M")`

**Explanation:**

- **A is incorrect:** Omits the required `pos` argument, causing a `TypeError`.
- **B is incorrect:** Dividing by 1,000 yields thousands ($K$), not millions ($M$).
- **C is correct:** Multiplying by $10^{-6}$ (or dividing by $1,000,000$) converts raw dollars into millions. The lambda signature `(x, pos)` correctly accepts both parameters required by `FuncFormatter`, producing clean strings like `"$25.0M"`.
- **D is incorrect:** Multiplying by $1,000,000$ scales the values in the wrong direction by six orders of magnitude.

</details>

---

### Question 66

In a multi-panel subplot layout with shared horizontal axes (`sharex=True`), interior subplots suppress their tick labels while keeping tick marks intact. 

Which formatter class renders tick marks on the axis line while guaranteeing that **zero text characters** are displayed?

- [ ] A) `matplotlib.ticker.NullFormatter()`
- [ ] B) `matplotlib.ticker.BlankFormatter()`
- [ ] C) `matplotlib.ticker.EmptyTextFormatter()`
- [ ] D) `matplotlib.ticker.SilentFormatter()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `matplotlib.ticker.NullFormatter()`

**Explanation:**

- **A is correct:** `matplotlib.ticker.NullFormatter()` is a specialized formatter that always returns an empty string `""` for every tick coordinate. When assigned via `ax.xaxis.set_major_formatter(NullFormatter())`, tick marks and gridlines remain intact and functional, but no text characters are rendered.
- **B is incorrect:** `BlankFormatter` is not a recognized class in Matplotlib.
- **C is incorrect:** `EmptyTextFormatter` does not exist in `matplotlib.ticker`.
- **D is incorrect:** `SilentFormatter` is a fabricated class name.

</details>

---

### Question 67

When plotting large numbers (e.g., $1000000$ to $1000005$), Matplotlib's default `ScalarFormatter` often places an offset annotation like `+1e6` in the top-left margin of the axis, and labels the ticks as `0, 1, 2, 3, 4, 5`.

Which convenience method disables this scientific offset notation and forces ticks to display full plain numbers?

- [ ] A) `ax.set_plain_numbers(True)`
- [ ] B) `ax.xaxis.disable_scientific()`
- [ ] C) `ax.yaxis.no_offset()`
- [ ] D) `ax.ticklabel_format(useOffset=False, style='plain')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.ticklabel_format(useOffset=False, style='plain')`

**Explanation:**

- **A is incorrect:** `ax.set_plain_numbers()` is a fictitious method.
- **B is incorrect:** `ax.xaxis.disable_scientific()` does not exist in the Matplotlib API.
- **C is incorrect:** `ax.yaxis.no_offset()` is not a valid method.
- **D is correct:** `ax.ticklabel_format(useOffset=False, style='plain')` configures the default `ScalarFormatter`. Setting `useOffset=False` disables the additive offset (e.g., `+1e6`), and setting `style='plain'` suppresses scientific exponential notation (e.g., `1e5`), forcing Matplotlib to render full, plain numbers on the axis.

</details>

---

### Question 68

A data scientist analyzes daily active user (DAU) trends across a year. They want the horizontal axis to display three-letter month abbreviations paired with four-digit years (e.g., `"Jan 2026"`, `"Feb 2026"`).

Which formatter from `matplotlib.dates` accomplishes this?

- [ ] A) `matplotlib.dates.MonthYearFormatter('MMM YYYY')`
- [ ] B) `matplotlib.dates.DateFormatter('%b %Y')`
- [ ] C) `matplotlib.dates.CalendarFormatter('month-year')`
- [ ] D) `matplotlib.dates.TimeFormat('%B-%y')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `matplotlib.dates.DateFormatter('%b %Y')`

**Explanation:**

- **A is incorrect:** `MonthYearFormatter` does not exist in `matplotlib.dates`.
- **B is correct:** `matplotlib.dates.DateFormatter(fmt)` formats datetime coordinates using standard Python `strftime` format codes. Here, `'%b'` formats the abbreviated month name (`"Jan"`, `"Feb"`), and `'%Y'` outputs the 4-digit year (`"2026"`), generating clean strings like `"Jan 2026"`.
- **C is incorrect:** `CalendarFormatter` is not a valid class in `matplotlib.dates`.
- **D is incorrect:** `TimeFormat` is a fabricated class name.

</details>

---

### Question 69

On a logarithmic scale axis, an engineer wants tick labels rendered as elegant mathematical powers ($10^1, 10^2, 10^3$) using LaTeX math font rendering rather than plain decimal numbers ($10, 100, 1000$).

Which built-in logarithmic formatter produces this mathematical typesetting?

- [ ] A) `LaTeXFormatter()`
- [ ] B) `PowerTextFormatter()`
- [ ] C) `matplotlib.ticker.LogFormatterMathtext()`
- [ ] D) `matplotlib.ticker.ScientificExponentFormatter()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `matplotlib.ticker.LogFormatterMathtext()`

**Explanation:**

- **A is incorrect:** `LaTeXFormatter` is not a valid class in `matplotlib.ticker`.
- **B is incorrect:** `PowerTextFormatter` does not exist in Matplotlib.
- **C is correct:** `matplotlib.ticker.LogFormatterMathtext()` is specifically tailored for logarithmic axes. Instead of outputting raw numbers ($1000$) or scientific notation ($1e3$), it formats decade coordinates into Matplotlib mathtext strings like `$10^{1}, $10^{2}, $10^{3}$`, rendering publication-grade mathematical exponents on the canvas.
- **D is incorrect:** `ScientificExponentFormatter` is a fictitious class name.

</details>

---

### Question 70

Which of the following statements regarding Matplotlib Formatters are **correct**?

1. A `FuncFormatter` callback must accept two parameters: coordinate value `x` and tick sequence position `pos`.
2. `PercentFormatter(xmax=1.0)` converts the float value $0.75$ into the string `"75%"`.
3. `NullFormatter()` returns an empty string for all ticks, effectively hiding tick text while keeping tick marks.
4. Setting a new major formatter shifts the numerical coordinate positions of the ticks along the axis.

- [ ] A) 1, 2, and 3 only
- [ ] B) 1 and 4 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) 1, 2, and 3 only

**Explanation:**

- **A is correct:** Statements 1, 2, and 3 are all correct. Statement 1 is true because `FuncFormatter` enforces the signature `func(x, pos)`. Statement 2 is true because `PercentFormatter(xmax=1.0)` converts $0.75$ to `"75%"`. Statement 3 is true because `NullFormatter` returns `""` for every tick label.
- **B is incorrect:** Statement 4 is false; formatters only govern text label rendering and never alter the numerical coordinate locations of ticks.
- **C is incorrect:** Excludes Statement 1 (which is true) and includes false Statement 4.
- **D is incorrect:** Excludes Statements 1 and 2 (both true) and includes false Statement 4.

</details>

---
