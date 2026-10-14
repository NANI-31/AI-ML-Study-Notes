# Matplotlib Ticks & Scales Objective Test — Set 02

---

### Question 11

When rotating x-axis tick labels by 45 degrees (`rotation=45`) to prevent horizontal collisions among long categorical strings, why is setting `ha='right'` (horizontal alignment `'right'`) strongly recommended?

- [ ] A) Because Matplotlib crashes with a `GeometryError` if diagonal text is centered.
- [ ] B) Because `ha='right'` converts the font rendering engine to vector curves.
- [ ] C) Because the default alignment is `ha='center'`; rotating text about its center causes the top of the text box to drift to the right, creating visual ambiguity about which tick mark the label belongs to, whereas `ha='right'` anchors the top-right corner of the label directly beneath the tick mark.
- [ ] D) Because `ha='right'` forces the labels to align with the right vertical spine rather than the tick marks.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) Because the default alignment is `ha='center'`; rotating text about its center causes the top of the text box to drift to the right, creating visual ambiguity about which tick mark the label belongs to, whereas `ha='right'` anchors the top-right corner of the label directly beneath the tick mark.

**Explanation:**

- **A is incorrect:** Matplotlib does not raise a `GeometryError` for centered rotated text; it renders it, but with poor visual alignment.
- **B is incorrect:** Alignment does not affect vector vs. raster rendering modes.
- **C is correct:** By default, tick label text artists have `horizontalalignment='center'`. When text is rotated 45 degrees clockwise, the rotation pivot is at the center of the text string. Consequently, the start of the label swings upward and to the right, causing the label to appear shifted underneath the next tick mark to the right. Specifying `ha='right'` (or `horizontalalignment='right'`) shifts the anchor point to the end of the text string, positioning the top-right corner of the label precisely beneath its corresponding tick mark.
- **D is incorrect:** The alignment is relative to the individual tick coordinates, not the outer right vertical spine.

</details>

---

### Question 12

A data engineer wants to rotate all x-axis tick labels of an existing subplot `ax` by 45 degrees with right-alignment using Matplotlib's property setter function `plt.setp()`.

Which command executes this correctly?

- [ ] A) `plt.setp(ax.xaxis, tilt=45, align='right')`
- [ ] B) `plt.setp(ax.get_xticks(), rotation=45)`
- [ ] C) `plt.setp(ax, xrotation=45, xalign='right')`
- [ ] D) `plt.setp(ax.get_xticklabels(), rotation=45, ha='right')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) `plt.setp(ax.get_xticklabels(), rotation=45, ha='right')`

**Explanation:**

- **A is incorrect:** `tilt` and `align` are not valid Matplotlib text properties, and `ax.xaxis` is an `Axis` object, not an iterable of text artists.
- **B is incorrect:** `ax.get_xticks()` returns a NumPy array of numeric coordinate floats, not the text label artists. Text properties like `rotation` cannot be applied to floats.
- **C is incorrect:** `xrotation` and `xalign` are not recognized properties of `Axes`.
- **D is correct:** `ax.get_xticklabels()` retrieves the list of `matplotlib.text.Text` artist objects representing the x-axis tick labels. Passing this list to `plt.setp(artists, rotation=45, ha='right')` iterates over every text artist and sets both the rotation angle and horizontal alignment cleanly.

</details>

---

### Question 13

In a production dashboard, tick labels are positioned too close to the axis line, causing text descenders (like letters "g", "y", "p") to visually touch the tick marks.

Which parameter in `ax.tick_params()` increases the whitespace buffer between the tick marks and the label text?

- [ ] A) `ax.tick_params(axis='x', pad=8)`
- [ ] B) `ax.tick_params(axis='x', buffer_dist=8)`
- [ ] C) `ax.tick_params(axis='x', text_margin=8)`
- [ ] D) `ax.tick_params(axis='x', spacing=8)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `ax.tick_params(axis='x', pad=8)`

**Explanation:**

- **A is correct:** In `ax.tick_params()`, the `pad` parameter specifies the distance in points between the tick mark endpoint and the edge of the tick label text box. Increasing `pad` (e.g., `pad=8` or `pad=10`) pushes the labels further away from the axis line, eliminating visual collisions with spines or tick marks.
- **B is incorrect:** `buffer_dist` is not a recognized parameter in `tick_params()`.
- **C is incorrect:** `text_margin` is not a valid argument for `ax.tick_params()`.
- **D is incorrect:** `spacing` is not a parameter accepted by `tick_params()`.

</details>

---

### Question 14

An engineer wants to enlarge the tick label font size across both horizontal and vertical axes simultaneously to ensure legibility on large conference projection screens.

Which call to `ax.tick_params()` accomplishes this across both axes in a single line?

- [ ] A) `ax.tick_params(font='large', scale=2)`
- [ ] B) `ax.tick_params(axis='both', which='major', labelsize=14)`
- [ ] C) `ax.tick_params(all_axes=True, size=14)`
- [ ] D) `ax.tick_params(target='text', height=14)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) `ax.tick_params(axis='both', which='major', labelsize=14)`

**Explanation:**

- **A is incorrect:** `font` and `scale` are not valid arguments for `tick_params()`.
- **B is correct:** `ax.tick_params()` controls appearance settings across tick marks and labels. Setting `axis='both'` applies the configuration to both `XAxis` and `YAxis`; `which='major'` targets major ticks; and `labelsize=14` (in points or size strings like `'large'`) adjusts the font size of the rendered label text.
- **C is incorrect:** `all_axes` is not an argument for `tick_params()`; the parameter is `axis='both'`.
- **D is incorrect:** `target` and `height` do not exist in `tick_params()`.

</details>

---

### Question 15

A financial time-series chart plots daily stock closing prices across six months. Because ISO date strings (`"2026-03-01"`, `"2026-03-15"`, etc.) are wide, they overlap heavily.

Which built-in Figure-level utility method automatically rotates x-axis date labels, right-aligns them, and adjusts the bottom figure margin to prevent label truncation?

- [ ] A) `plt.straighten_dates()`
- [ ] B) `ax.auto_align_x()`
- [ ] C) `fig.autofmt_xdate(rotation=30, ha='right')`
- [ ] D) `fig.format_calendar()`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `fig.autofmt_xdate(rotation=30, ha='right')`

**Explanation:**

- **A is incorrect:** `straighten_dates()` does not exist in the Matplotlib library.
- **B is incorrect:** `auto_align_x()` is not a method on `Axes`.
- **C is correct:** `Figure.autofmt_xdate(bottom=0.2, rotation=30, ha='right')` is a specialized convenience method designed specifically for date and time axes. It iterates over all subplots in the figure, rotates the x-axis tick labels (defaulting to 30 degrees), sets their horizontal alignment to `'right'`, and increases the figure's bottom margin to prevent labels from being clipped by the canvas edge.
- **D is incorrect:** `format_calendar()` is a fictitious function name.

</details>

---

### Question 16

In an A/B test comparison chart, a data scientist uses multi-line string labels to include both variant names and sample sizes:

```python
labels = ["Control\n(n = 12,450)", "Treatment A\n(n = 12,510)", "Treatment B\n(n = 12,480)"]
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, va='top')
```

Why is specifying `va='top'` (vertical alignment `'top'`) important when using multi-line horizontal tick labels?

- [ ] A) It ensures that the text strings are printed in reverse order from bottom to top.
- [ ] B) It centers the text vertically inside the data plotting area.
- [ ] C) It forces the top line to render in bold font.
- [ ] D) It anchors the top edge of the multi-line text box just below the tick mark, ensuring that the first line of every label aligns at a uniform distance beneath the x-axis line regardless of line count.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) It anchors the top edge of the multi-line text box just below the tick mark, ensuring that the first line of every label aligns at a uniform distance beneath the x-axis line regardless of line count.

**Explanation:**

- **A is incorrect:** `va='top'` does not reverse line ordering.
- **B is incorrect:** Tick labels reside outside the plotting area, not inside it.
- **C is incorrect:** Vertical alignment does not modify font weight or styling.
- **D is correct:** When tick labels contain newline characters `\n`, different labels may have differing numbers of lines. If vertical alignment were `'center'` or `'bottom'`, labels with two lines would shift upward closer to the axis spine compared to labels with three lines. Setting `va='top'` establishes the top boundary of the text block as the anchor point, ensuring the first line of text across all labels aligns along a single straight horizontal plane.

</details>

---

### Question 17

An exploratory data analysis script visualizes mutual information scores for 30 genomics features. Because horizontal space is limited, the engineer wants the x-axis feature names oriented completely vertically ($90^\circ$).

Which syntax correctly sets this orientation in the `pyplot` interface?

- [ ] A) `plt.xticks(rotation=90)` (or `plt.xticks(rotation='vertical')`)
- [ ] B) `plt.xticks(orientation='upright')`
- [ ] C) `plt.axis(rotate_labels=90)`
- [ ] D) `plt.xlabel(degrees=90)`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** A) `plt.xticks(rotation=90)` (or `plt.xticks(rotation='vertical')`)

**Explanation:**

- **A is correct:** In `plt.xticks()`, passing `rotation=90` (or the string alias `rotation='vertical'`) rotates the tick labels perpendicular to the axis line, allowing densely packed categorical labels to fit alongside one another without horizontal overlap.
- **B is incorrect:** `orientation` is not a valid keyword argument for `plt.xticks()`.
- **C is incorrect:** `plt.axis()` controls axis limits and aspect ratio flags (e.g., `'equal'`, `'off'`), not label rotation.
- **D is incorrect:** `plt.xlabel()` sets the axis title text, not the rotation of individual tick labels.

</details>

---

### Question 18

After rotating x-tick labels by $90^\circ$, an engineer exports the figure via `plt.savefig("feature_scores.png")` and notices that the bottom halves of all feature names are completely clipped off outside the image borders.

What is the root cause of this clipping, and how is it most robustly resolved?

- [ ] A) Rotating labels causes Matplotlib to delete pixels that exceed 100 DPI; the solution is to export at 50 DPI.
- [ ] B) Rotated labels extend beyond the default bottom figure margin allocated during canvas initialization; resolving it requires calling `plt.tight_layout()` or passing `bbox_inches='tight'` to `plt.savefig()`.
- [ ] C) The operating system graphic window buffer is corrupted; the script must be run inside a Docker container.
- [ ] D) Rotating text by $90^\circ$ is only supported in SVG format; PNG format cannot render vertical fonts.

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** B) Rotated labels extend beyond the default bottom figure margin allocated during canvas initialization; resolving it requires calling `plt.tight_layout()` or passing `bbox_inches='tight'` to `plt.savefig()`.

**Explanation:**

- **A is incorrect:** DPI defines pixel density, not canvas boundary clipping logic.
- **B is correct:** By default, Matplotlib figures have static outer margins (typically bottom margin $\approx 0.10$ or $10\%$). Long labels oriented vertically or diagonally often extend past this margin, protruding outside the figure canvas. Calling `plt.tight_layout()` or passing `bbox_inches='tight'` to `savefig()` calculates the true bounding box encompassing all outer text artists and expands the exported canvas boundaries accordingly.
- **C is incorrect:** This is standard layout geometry, not OS buffer corruption.
- **D is incorrect:** PNG fully supports vertical raster text rendering.

</details>

---

### Question 19

A designer creating a sleek dark-mode dashboard wants the tick marks (lines) to remain subtle gray (`'#555555'`), while the tick labels (text) stand out in bright cyan (`'#00ffff'`).

Which `ax.tick_params()` configuration accomplishes this cleanly?

- [ ] A) `ax.tick_params(axis='both', color='#00ffff', text_color='#555555')`
- [ ] B) `ax.tick_params(axis='both', line='#555555', font='#00ffff')`
- [ ] C) `ax.tick_params(axis='both', color='#555555', labelcolor='#00ffff')`
- [ ] D) `ax.tick_params(axis='both', stroke='#555555', fill='#00ffff')`

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** C) `ax.tick_params(axis='both', color='#555555', labelcolor='#00ffff')`

**Explanation:**

- **A is incorrect:** `color` sets the tick mark line color, and `text_color` is not a valid parameter.
- **B is incorrect:** `line` and `font` are not recognized parameters in `tick_params()`.
- **C is correct:** In `ax.tick_params()`, the `color` keyword argument controls the stroke color of the tick lines themselves, whereas the `labelcolor` parameter controls the text color of the tick labels. Setting `color='#555555'` and `labelcolor='#00ffff'` decouples the tick line styling from the label text styling.
- **D is incorrect:** `stroke` and `fill` are SVG terminology, not Matplotlib `tick_params` arguments.

</details>

---

### Question 20

Which of the following statements regarding tick label rotation and alignment in Matplotlib are **correct**?

1. The default horizontal alignment for standard horizontal x-axis tick labels is `'center'`.
2. Rotating x-tick labels by $45^\circ$ without updating horizontal alignment often makes each label visually drift toward the subsequent tick mark to the right.
3. Setting `rotation=45` alters only the visual presentation of the text and does not change the numeric coordinate values returned by `ax.get_xticks()`.
4. Rotating tick labels automatically disables grid lines across the subplot.

- [ ] A) 1 and 4 only
- [ ] B) 2 and 4 only
- [ ] C) 2, 3, and 4 only
- [ ] D) 1, 2, and 3 only

<details>
<summary><b>View Answer & Explanation</b></summary>

**Correct Answer:** D) 1, 2, and 3 only

**Explanation:**

- **A is incorrect:** Statement 4 is false because rotating tick labels does not disable grid lines; Statements 2 and 3 are also omitted.
- **B is incorrect:** Statement 4 is false, and Statements 1 and 3 are omitted.
- **C is incorrect:** Statement 4 is false, and Statement 1 is omitted.
- **D is correct:** Statements 1, 2, and 3 are all correct. Default horizontal alignment is `ha='center'`; rotating labels by 45 degrees without right-alignment causes visual drift toward adjacent ticks; and rotation modifies the text visual presentation without altering underlying numeric coordinates.

</details>

---
