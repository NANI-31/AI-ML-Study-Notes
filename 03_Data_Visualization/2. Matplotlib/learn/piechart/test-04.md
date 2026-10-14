# Matplotlib Pie Chart Objective Test — Set 04

---

### 31. Question

When a custom Python callable function is passed to the `autopct` argument of `plt.pie()`, what single numerical argument does Matplotlib pass into that function for each slice?

- [ ] **A.** The raw integer count of observations in that category
- [ ] **B.** The wedge's angular sweep in radians
- [ ] **C.** The index position of the slice ($0, 1, 2, ...$)
- [ ] **D.** The calculated percentage value as a floating-point number on a $0$ to $100$ scale (e.g., $25.0$ for a quarter slice)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** Matplotlib does not pass the raw data counts; it only passes the calculated percentage. To display raw counts, developers must capture the raw data array via an enclosing function closure.
- **Option B (Incorrect):** Angular sweep values ($0$ to $2\pi$ or $0^\circ$ to $360^\circ$) are not passed to `autopct`.
- **Option C (Incorrect):** Slice indices are not passed to `autopct`.
- **Option D (Correct):** When `autopct` is a callable (such as a lambda or function definition), Matplotlib evaluates each wedge's relative fraction, scales it to a percentage in the range $[0.0, 100.0]$, and passes that float as the sole argument into the function.

</details>

---

### 32. Question

An analytics engineer implements a custom formatter closure to display both percentages and raw counts:

```python
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({val:d})"
    return my_autopct

data = [150, 250, 200]
plt.pie(data, autopct=make_autopct(data))
```

For the first slice ($150$ out of $600$ total), what exact two-line string is rendered inside the wedge?

- [ ] **A.** `"150%\n(25.0)"`
- [ ] **B.** `"25.0%\n(150)"`
- [ ] **C.** `"0.25%\n(150)"`
- [ ] **D.** `"25%\n(600)"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** This inverts the percentage and raw count roles.
- **Option B (Correct):** First, compute total observations: $150 + 250 + 200 = 600$.
  The percentage for the first slice is $\frac{150}{600} \times 100 = 25.0\%$.
  Inside the closure, `val` is computed as $\text{round}(25.0 \times 600 / 100.0) = 150$.
  The f-string `f"{pct:.1f}%\n({val:d})"` formats `25.0` with one decimal place, followed by a newline and `(150)`, rendering `"25.0%\n(150)"`.
- **Option C (Incorrect):** The percentage float passed to the function is $25.0$, not $0.25$.
- **Option D (Incorrect):** The count displayed is the slice count ($150$), not the total dataset sum ($600$).

</details>

---

### 33. Question

In a customer segmentation chart containing several minor segments, small slices have their percentage text overlapping illegibly. Which lambda function passed to `autopct` displays percentages only for slices strictly greater than 5%, hiding the text on smaller slices?

- [ ] **A.** `autopct=lambda pct: f'{pct:.1f}%' if pct < 5 else ''`
- [ ] **B.** `autopct=lambda pct: pct if pct > 5 else None`
- [ ] **C.** `autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else ''`
- [ ] **D.** `autopct=lambda pct: f'{pct:.1f}%' if val > 5 else False`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** This condition displays text on slices *under* 5% and hides larger slices.
- **Option B (Incorrect):** Returning `None` does not suppress the string cleanly; it renders the text `"None"` on the plot.
- **Option C (Correct):** Returning an empty string `''` instructs Matplotlib's text renderer to draw nothing. If `pct > 5`, the formatted percentage string is returned; otherwise, `''` is returned, keeping cramped minor slices clean and legible.
- **Option D (Incorrect):** `val` is not defined in the lambda parameter list, causing a `NameError`.

</details>

---

### 34. Question

A junior developer writes a custom `autopct` function to display raw transaction counts:

```python
def bad_formatter(pct):
    total = 500
    count = int(pct * total)   # Bug is on this line
    return f"{count}"
```

For a slice representing $20\%$ of the data, what number will this function erroneously display instead of the correct count of 100?

- [ ] **A.** `10000`
- [ ] **B.** `20`
- [ ] **C.** `0`
- [ ] **D.** `100`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In Matplotlib, the value passed to `pct` is already scaled as a percentage from $0$ to $100$ ($pct = 20.0$, not $0.20$). Multiplying $20 \times 500$ yields **$10,000$**. To convert a percentage back to an absolute count, one must divide by 100: $\text{count} = \text{int}(pct \times \text{total} / 100.0)$, which would yield the correct value of $100$.
- **Option B (Incorrect):** 20 is the percentage value itself, not the product of $20 \times 500$.
- **Option C (Incorrect):** The product does not evaluate to zero.
- **Option D (Incorrect):** 100 is the correct true count, but the missing division by 100 produces 10000.

</details>

---

### 35. Question

A clinical trial pie chart has a total population of $N = 1,600$ patients across four treatment arms. One of the treatment arms displays `22.5%` inside its wedge. How many patients were in that specific treatment arm?

- [ ] **A.** $225$
- [ ] **B.** $320$
- [ ] **C.** $360$
- [ ] **D.** $450$

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Option A (Incorrect):** $225$ confuses the percentage string $22.5\%$ with patient count.
- **Option B (Incorrect):** $320$ corresponds to $20.0\%$ ($0.20 \times 1600$).
- **Option C (Correct):** The count is calculated by multiplying the percentage fraction by the total cohort:
  $$\text{Count} = \left(\frac{22.5}{100}\right) \times 1600 = 0.225 \times 1600 = \mathbf{360}\text{ patients}$$
- **Option D (Incorrect):** $450$ corresponds to $28.125\%$.

</details>

---

### 36. Question

An IT operations dashboard visualizes monthly AWS cloud expenditures. Which custom `autopct` function formats each slice with its percentage and formatted dollar cost with thousands commas (e.g., `"28.4%\n($14,200)"`)?

- [ ] **A.** `autopct=lambda pct: f"${pct * 500:,.0f}"`
- [ ] **B.** `autopct=lambda pct: f"{pct:.1f}%\n({pct * 500:d})"`
- [ ] **C.** `autopct=lambda pct: f"{pct}%"`
- [ ] **D.** `autopct=lambda pct: f"{pct:.1f}%\n(${int(round(pct * 50000 / 100)):,d})"`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** This omits the percentage string and newline entirely.
- **Option B (Incorrect):** This omits the dollar sign `$` and fails to format with commas.
- **Option C (Incorrect):** This formats only the percentage, omitting dollar costs.
- **Option D (Correct):** In this lambda, `f"{pct:.1f}%"` prints the percentage to one decimal place, followed by a newline `\n`. The absolute dollar calculation $\text{int}(\text{round}(pct \times \text{total} / 100))$ computes the integer currency amount, and the `:,d` format specifier adds commas for thousands grouping preceded by a `$` sign.

</details>

---

### 37. Question

What is the difference in rendered text between `autopct='%1.2f%%'` and `autopct='%d%%'` for a category representing an exact fraction of $\frac{1}{8}$ ($12.5\%$)?

- [ ] **A.** `'%1.2f%%'` renders `"12.5%"`, while `'%d%%'` raises an error
- [ ] **B.** `'%1.2f%%'` renders `"12.50%"`, while `'%d%%'` renders `"12%" (or rounded to "13%")`
- [ ] **C.** Both format strings produce identical output
- [ ] **D.** `'%d%%'` renders the date instead of a number

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **B**

#### Explanations:

- **Option A (Incorrect):** `%1.2f%%` forces two decimal places, producing `"12.50%"`, not one decimal.
- **Option B (Correct):** The conversion code `%1.2f%%` specifies a floating-point representation with exactly two digits after the decimal point, formatting $12.5$ as `"12.50%"`. In contrast, `%d%%` specifies decimal integer formatting, truncating or rounding the float to an integer, rendering `"12%"` (or `"13%"` depending on implementation rounding).
- **Option C (Incorrect):** Floating-point precision and integer formatting produce distinct strings.
- **Option D (Incorrect):** In C-style printf format strings, `%d` denotes a decimal integer, not a calendar date.

</details>

---

### 38. Question

Which parameter in `plt.pie()` allows passing a unified dictionary of typography properties (such as font size, family, and color) applied to all generated text objects?

- [ ] **A.** `textprops={'fontsize': 12, 'color': 'navy', 'weight': 'bold'}`
- [ ] **B.** `fontdict={'size': 12}`
- [ ] **C.** `font={'style': 'bold'}`
- [ ] **D.** `text_settings={'color': 'navy'}`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **A**

#### Explanations:

- **Option A (Correct):** In `plt.pie()`, the `textprops` keyword parameter accepts a dictionary of valid `matplotlib.text.Text` properties. Passing `textprops={'fontsize': 12, 'color': 'navy', 'weight': 'bold'}` applies these typographic styling rules across all generated label and percentage text artists simultaneously.
- **Option B (Incorrect):** `fontdict` is a parameter for `plt.title()` and `plt.xlabel()`, not `plt.pie()`.
- **Option C (Incorrect):** `font` is not a valid parameter of `plt.pie()`.
- **Option D (Incorrect):** `text_settings` is not a recognized Matplotlib keyword argument.

</details>

---

### 39. Question

A data analyst configures a minimalist pie chart with category labels outside, setting `autopct=None`:

```python
result = ax.pie([40, 35, 25], labels=["A", "B", "C"], autopct=None)
```

How many elements are contained in the returned tuple `result`?

- [ ] **A.** 3 elements `(patches, texts, autotexts)` where `autotexts` is an empty list
- [ ] **B.** 1 element `(patches)`
- [ ] **C.** 0 elements (returns `None`)
- [ ] **D.** 2 elements `(patches, texts)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **D**

#### Explanations:

- **Option A (Incorrect):** When `autopct` is `None` (the default), no percentage text artists are created, and `autotexts` is not included in the return tuple at all.
- **Option B (Incorrect):** `texts` is still returned because category `labels` were provided.
- **Option C (Incorrect):** `ax.pie()` always returns the created artists.
- **Option D (Correct):** In Matplotlib, the structure of the return tuple depends dynamically on whether `autopct` is provided:
  - If `autopct is None`: Returns a 2-tuple `(patches, texts)`
  - If `autopct is not None`: Returns a 3-tuple `(patches, texts, autotexts)`
  Because `autopct=None`, `result` contains exactly **2 elements**.

</details>

---

### 40. Question

Consider the following statements regarding custom `autopct` functions and text formatting in Matplotlib:

1. Passing a callable function to `autopct` allows dynamic multiline text output using newline `\n` characters.
2. The argument passed to a custom `autopct` function is the calculated percentage float, not the raw count.
3. Returning an empty string `''` from an `autopct` function cleanly suppresses percentage display for that wedge.
4. Setting `autopct='%1.1f%%'` rounds numbers using standard floating-point representation.

Which statements are **correct**?

- [ ] **A.** 1 and 3 only
- [ ] **B.** 2 and 4 only
- [ ] **C.** 1, 2, 3, and 4
- [ ] **D.** 1, 2, and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** **C**

#### Explanations:

- **Statement 1 is correct:** Multiline strings (e.g., percentage on line 1, absolute count on line 2) are fully supported via `\n`.
- **Statement 2 is correct:** Matplotlib evaluates the wedge proportion and supplies the percentage value to the callable.
- **Statement 3 is correct:** An empty string generates an empty text artist, effectively hiding the label for that slice.
- **Statement 4 is correct:** Format specifiers like `%1.1f` round the floating-point value to the specified number of decimals.
- **Conclusion:** All four statements are correct, making **Option C** the correct choice.

</details>

---
