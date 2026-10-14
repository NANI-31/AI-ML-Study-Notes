# Matplotlib Ticks & Scales Objective Test — Set 08

---

### Question 71

In many physical science and astronomy journals, figures are formatted with tick marks pointing inward toward the data rather than outward into the margin whitespace.

Which parameter and value in `ax.tick_params()` or `plt.tick_params()` achieves this inward orientation?

- [ ] A) `orientation='interior'`
- [ ] B) `tick_type='reverse'`
- [ ] C) `side='inside'`
- [ ] D) `direction='in'`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `direction='in'`

**Explanation:**

- **A is incorrect:** `orientation='interior'` is not a valid parameter in `tick_params()`; calling it raises a `ValueError`.
- **B is incorrect:** `tick_type='reverse'` is a fictitious parameter not recognized by Matplotlib.
- **C is incorrect:** `side='inside'` is not supported; Matplotlib uses `direction` to control tick stroke orientation.
- **D is correct:** In `ax.tick_params()`, the `direction` argument accepts `'out'` (default, pointing away from the plotting region), `'in'` (pointing inward into the axes plotting area), or `'inout'` (straddling the spine equally). Setting `direction='in'` is widely adopted in scientific journals to keep outer margin whitespace clean.

</details>

---

### Question 72

A data designer wants to create a clear visual hierarchy between major and minor ticks on a dense loss chart:
- Major ticks: length $7$ points, width $1.5$ points
- Minor ticks: length $3.5$ points, width $0.8$ points

Which sequence of calls establishes these precise dimensions?

- [ ] A) `ax.set_ticks(major=(7, 1.5), minor=(3.5, 0.8))`
- [ ] B)
  ```python
  ax.tick_params(which="major", length=7, width=1.5)
  ax.tick_params(which="minor", length=3.5, width=0.8)
```
- [ ] C) `ax.xaxis.scale_ticks(7, 3.5)`
- [ ] D) `ax.tick_sizes(major_len=7, minor_len=3.5)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B)
```python
ax.tick_params(which="major", length=7, width=1.5)
ax.tick_params(which="minor", length=3.5, width=0.8)
```

**Explanation:**

- **A is incorrect:** `ax.set_ticks()` does not accept `major` or `minor` size tuples; it only accepts numerical coordinates and text labels.
- **B is correct:** In `ax.tick_params()`, the `which` parameter dictates whether styling applies to `'major'`, `'minor'`, or `'both'` tick collections. The `length` and `width` parameters set the physical stroke length and line thickness in points. Configuring major and minor ticks separately creates a clear visual hierarchy on technical plots.
- **C is incorrect:** `ax.xaxis.scale_ticks()` is a non-existent method.
- **D is incorrect:** `ax.tick_sizes()` does not exist in the Matplotlib API.

</details>

---

### Question 73

Scientific journal publications often require tick marks to appear on **all four bounding spines** (bottom, top, left, and right) while restricting text labels strictly to the bottom and left edges.

Which configuration achieves this mirrored framing?

- [ ] A) `ax.tick_params(top=True, right=True, direction='in')`
- [ ] B) `ax.mirror_axes(all_spines=True)`
- [ ] C) `ax.spines.copy_ticks()`
- [ ] D) `ax.tick_params(duplicate=True)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.tick_params(top=True, right=True, direction='in')`

**Explanation:**

- **A is correct:** By default, Matplotlib only renders tick marks on the bottom and left spines (`bottom=True`, `left=True`, `top=False`, `right=False`). Setting `top=True` and `right=True` draws tick lines across the opposing top and right spines as well. Because `labeltop` and `labelright` default to `False`, text numbers remain cleanly restricted to the bottom and left borders.
- **B is incorrect:** `mirror_axes()` is not a valid Matplotlib method.
- **C is incorrect:** `ax.spines.copy_ticks()` does not exist.
- **D is incorrect:** `duplicate=True` is not a valid keyword argument for `tick_params()`.

</details>

---

### Question 74

An engineer visualizes time-series data where the bottom x-axis displays elapsed seconds ($0, 10, 20, \dots$) and the top x-axis displays corresponding real-world timestamps (`"12:00"`, `"12:10"`, $\dots$).

To enable visible tick labels along the upper horizontal spine, which parameter must be toggled in `ax.tick_params()`?

- [ ] A) `top_text=True`
- [ ] B) `upper_labels=True`
- [ ] C) `top=True, labeltop=True`
- [ ] D) `show_top_numbers=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `top=True, labeltop=True`

**Explanation:**

- **A is incorrect:** `top_text` is not a recognized keyword argument in `tick_params()`.
- **B is incorrect:** `upper_labels` is a fictitious argument.
- **C is correct:** In Matplotlib, tick mark lines and tick text labels have separate visibility controls for each spine: `bottom`/`labelbottom`, `top`/`labeltop`, `left`/`labelleft`, `right`/`labelright`. To display ticks and labels along the top edge, both `top=True` (enabling tick notch lines) and `labeltop=True` (enabling text labels) must be set.
- **D is incorrect:** `show_top_numbers` does not exist in the Matplotlib API.

</details>

---

### Question 75

In modern, minimalist UI design (e.g., Tufte-style clean charts or web dashboards), designers frequently hide the physical tick mark lines (notches) while keeping the numeric text labels clearly visible along the baseline.

Which command removes the tick mark notches without hiding the numbers?

- [ ] A) `ax.tick_params(axis='x', labels_only=True)`
- [ ] B) `ax.tick_params(axis='x', remove_notches=True)`
- [ ] C) `ax.set_ticks(show_lines=False)`
- [ ] D) `ax.tick_params(axis='x', bottom=False)` (or `ax.tick_params(axis='x', length=0)`)

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.tick_params(axis='x', bottom=False)` (or `ax.tick_params(axis='x', length=0)`)

**Explanation:**

- **A is incorrect:** `labels_only=True` is not a recognized parameter in `tick_params()`.
- **B is incorrect:** `remove_notches` is a fictitious argument.
- **C is incorrect:** `ax.set_ticks()` does not accept `show_lines=False`.
- **D is correct:** Setting `bottom=False` disables drawing the tick mark lines along the bottom spine, while leaving `labelbottom` (which defaults to `True`) active. Alternatively, setting `length=0` sets the physical stroke length of the tick lines to zero. Both approaches produce a clean, modern aesthetic where numeric labels sit neatly beneath the axis without protruding tick marks.

</details>

---

### Question 76

What happens when an engineer executes `ax.tick_params(colors='crimson')` versus `ax.tick_params(labelcolor='crimson')`?

- [ ] A) `colors='crimson'` only affects the background fill, whereas `labelcolor` changes the gridlines.
- [ ] B) `colors='crimson'` changes both the tick mark stroke lines AND the tick label text to crimson, whereas `labelcolor='crimson'` alters ONLY the text labels, leaving the tick lines in their original color.
- [ ] C) `colors='crimson'` is deprecated and raises a syntax error.
- [ ] D) There is no difference; they are exact aliases in Python bytecode.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `colors='crimson'` changes both the tick mark stroke lines AND the tick label text to crimson, whereas `labelcolor='crimson'` alters ONLY the text labels, leaving the tick lines in their original color.

**Explanation:**

- **A is incorrect:** `colors` does not affect axis background fill; background fill is governed by `ax.set_facecolor()`.
- **B is correct:** In `ax.tick_params()`, `colors` acts as a master color setter: it assigns the specified color to both the tick mark lines and the tick text labels. In contrast, `labelcolor` specifically overrides the color of the text labels alone, allowing developers to style tick marks and text independently (e.g., subtle gray tick marks with vibrant colored labels).
- **C is incorrect:** `colors` is fully supported and is not deprecated.
- **D is incorrect:** `colors` and `labelcolor` are distinct parameters with different behaviors.

</details>

---

### Question 77

An engineer wishes to configure minor ticks across both axes so they render with width $0.5$ and length $2$, without inadvertently modifying any major tick properties.

Which argument in `ax.tick_params()` restricts the changes strictly to minor ticks?

- [ ] A) `which='minor'`
- [ ] B) `target='minor_only'`
- [ ] C) `level=2`
- [ ] D) `subticks=True`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `which='minor'`

**Explanation:**

- **A is correct:** In `ax.tick_params()`, the `which` parameter dictates the scope of the styling command. Specifying `which='minor'` ensures that parameters like `length`, `width`, `color`, and `pad` are applied exclusively to minor ticks, leaving the primary major ticks untouched.
- **B is incorrect:** `target='minor_only'` is not a valid argument in `tick_params()`.
- **C is incorrect:** `level=2` is not recognized by Matplotlib.
- **D is incorrect:** `subticks=True` is a fictitious parameter name.

</details>

---

### Question 78

What does passing `reset=True` to `ax.tick_params()` do before applying any newly specified keyword arguments?

- [ ] A) It resets the figure canvas to a white background.
- [ ] B) It deletes all plotted data series on the axis.
- [ ] C) It resets all tick styling properties (lengths, widths, colors, directions, label sizes) back to their default `rcParams` values before applying the new arguments.
- [ ] D) It resets the coordinate view limits back to $[0.0, 1.0]$.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) It resets all tick styling properties (lengths, widths, colors, directions, label sizes) back to their default `rcParams` values before applying the new arguments.

**Explanation:**

- **A is incorrect:** Canvas background color is managed by `fig.patch.set_facecolor()`, not `tick_params(reset=True)`.
- **B is incorrect:** Plotted data series are artists stored in `ax.lines`, `ax.collections`, etc.; `tick_params()` never modifies or deletes data series.
- **C is correct:** When multiple cumulative styling calls have been made to `tick_params()`, passing `reset=True` instructs Matplotlib to wipe clean any prior custom tick parameters, resetting all tick properties (color, length, width, direction, pad, label visibility) back to their baseline Matplotlib defaults before applying the newly provided arguments.
- **D is incorrect:** Coordinate view limits are managed by `set_xlim()` and `set_ylim()`, not `tick_params()`.

</details>

---

### Question 79

In modern Matplotlib, what is the most direct, Pythonic way to rotate x-axis tick labels by 45 degrees without manually retrieving the label list via `ax.get_xticklabels()`?

- [ ] A) `ax.rotate_x(45)`
- [ ] B) `ax.xaxis.set_angle(45)`
- [ ] C) `ax.set_xticklabels(rotation=45)` without supplying tick labels
- [ ] D) `ax.tick_params(axis='x', rotation=45)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `ax.tick_params(axis='x', rotation=45)`

**Explanation:**

- **A is incorrect:** `rotate_x()` does not exist on `Axes` objects.
- **B is incorrect:** `ax.xaxis.set_angle()` is not a valid Matplotlib method.
- **C is incorrect:** Calling `set_xticklabels()` without passing tick label strings raises a `TypeError: missing 1 required positional argument: 'labels'`.
- **D is correct:** `ax.tick_params(axis='x', rotation=45)` directly sets the rotation angle for the x-axis tick labels. It avoids the tedious boilerplate of fetching label artists and does not trigger `FixedFormatter` warnings because it operates directly on existing tick parameters without replacing the underlying formatter.

</details>

---

### Question 80

Which of the following statements regarding `ax.tick_params()` in Matplotlib are **correct**?

1. The `axis` parameter accepts `'x'`, `'y'`, or `'both'`.
2. The `direction` parameter accepts `'in'`, `'out'`, or `'inout'`.
3. Setting `bottom=False` removes the tick mark notches on the bottom spine while keeping the text labels intact.
4. `ax.tick_params()` must always be executed before any plotting calls (`plot`, `scatter`) or it raises an error.

- [ ] A) 1 and 4 only
- [ ] B) 1, 2, and 3 only
- [ ] C) 2 and 4 only
- [ ] D) 3 and 4 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Erroneously includes Statement 4, which is completely false.
- **B is correct:** Statements 1, 2, and 3 are all correct. Statement 1 is true because `axis` accepts `'x'`, `'y'`, or `'both'`. Statement 2 is true because `direction` accepts `'in'`, `'out'`, or `'inout'`. Statement 3 is true because `bottom=False` removes bottom tick notch lines while leaving labels visible. Statement 4 is false because `tick_params()` can be called at any time before or after plotting calls.
- **C is incorrect:** Omits Statement 1 (which is true) and includes false Statement 4.
- **D is incorrect:** Omits Statements 1 and 2 (both true) and includes false Statement 4.

</details>

---
