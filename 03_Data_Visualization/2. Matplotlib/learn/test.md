### 1 Question

A data analyst is reviewing whether a monthly sales chart might mislead viewers. Which of the following statements reflect the data visualization ethics concerns discussed?

- [ ] Truncating the y-axis range changes the underlying data values, not just how the chart looks
- [x] **Starting the y-axis at a value other than zero can visually exaggerate small differences between data points** _(Correct)_
- [ ] Manipulating statistical significance thresholds is an unrelated concern that has nothing to do with misleading visualization
- [x] **Choosing an axis range that starts at zero (or is otherwise clearly labeled) supports honest visualization** _(Correct)_

### Answer Explanation

- **B (Correct):** Narrowing or truncating the y-axis window (e.g., starting around 90–120 instead of 0) can make small, near-flat differences look large and dramatic even though the numbers haven't changed.
- **D (Correct):** Choosing an honest axis range — typically starting at zero, or otherwise clearly labeled — was described as part of responsible visualization technique.
- **A (Incorrect):** Truncating the axis is purely a display choice; it never alters the underlying numbers, which is exactly why the technique is considered misleading rather than a data error.
- **C (Incorrect):** Manipulating statistical significance thresholds was explicitly named as a related concern deserving conscious attention alongside misleading axis scaling, not an unrelated one.

---

### 2 Question

A product manager is verifying the bar-centering math for a grouped bar chart. For a category at integer x-position 1, with bar width = 0.5, the left bar is centered at i - width/2 and the right bar at i + width/2 . What are the two resulting bar centers?

- [ ] 0.5 and 1.5
- [x] **0.75 and 1.25** _(Correct)_
- [ ] 0.25 and 1.75
- [ ] 1.0 and 1.5

### Answer Explanation

- **B (Correct):** Substituting i = 1 and width = 0.5 gives 1 - 0.5/2 = 0.75 for the left bar and 1 + 0.5/2 = 1.25 for the right bar, so the two bars sit a quarter-width to either side of the category's true position.
- **A (Incorrect):** 0.5 and 1.5 would result from offsetting by the full width instead of half the width.
- **C (Incorrect):** 0.25 and 1.75 would result from an incorrect three-quarters-width offset, overshooting the correct quarter-width offset.
- **D (Incorrect):** 1.0 and 1.5 incorrectly leaves the left bar exactly at the category position and only offsets the right bar, which does not match the symmetric offset pattern described.

---

### 3 Question

A backend engineer has built numeric x-positions `x = range(len(region_totals.index))` for a grouped bar chart, but the chart currently shows plain integers (0, 1) on the x-axis instead of the region names ("north", "south"). Which call maps the numeric positions back to the readable category labels?

- [x] **`plt.xticks(x, region_totals.index)`** _(Correct)_
- [ ] `plt.yticks(x, region_totals.index)`
- [ ] `plt.legend(region_totals.index)`
- [ ] `plt.xlabel(region_totals.index)`

### Answer Explanation

- **A (Correct):** `plt.xticks(x, region_totals.index)` overrides the automatic tick selection on the x-axis, mapping each numeric position in `x` to the corresponding label in `region_totals.index`.
- **B (Incorrect):** `plt.yticks(x, region_totals.index)` applies the same idea to the wrong axis—`yticks` controls vertical tick marks, not the horizontal category labels needed here.
- **C (Incorrect):** `plt.legend()` displays series labels set via the `label` argument in `plot()` / `bar()` calls; it does not accept tick positions and will not relabel the x-axis.
- **D (Incorrect):** `plt.xlabel()` sets a single axis title string (like "Region"), not a list of per-tick category labels.

---

### 4 Question

A senior data analyst builds a report figure with `fig, axes = plt.subplots(2, 3, figsize=(12, 6))`, then separately builds a scatter plot of study hours vs. exam score by looping over `df.groupby("subject")` and calling `plt.scatter()` once per subject group. Which of the following statements are correct?

- [x] **`axes` is an array of individual axis objects, one per subplot position, so the first subplot is addressed as `axes[0]`** _(Correct)_
- [ ] `plt.savefig()` must always be called before `plt.show()`, otherwise the resulting saved image file will come out blank
- [x] **Looping over `df.groupby("subject")` and calling `plt.scatter()` per group gives each subject its own color and legend label automatically** _(Correct)_
- [ ] When the x-axis represents discrete categories like region names, a line chart is generally clearer to read than a bar chart

### Answer Explanation

- **A (Correct):** `plt.subplots(nrows, ncols, ...)` returns `fig` and `axes`, where `axes` is an array of individual axis objects — one per grid position — and each is accessed by indexing, e.g., `axes[0]` for the first subplot.
- **C (Correct):** Calling `plt.scatter()` once per `(subject, group)` pair from the groupby loop gives each subject group its own default color and its own legend entry via the label argument, without extra manual work.
- **B (Incorrect):** No ordering dependency between `savefig()` and `show()` was described; saving captures the current figure's state and does not require `show()` to have run first.
- **D (Incorrect):** The guidance given was the opposite — when the x-axis represents discrete categories rather than a continuous variable, a bar chart is generally a more natural and less confusing choice than a line chart.

---

### 5 Question

A junior ML engineer runs `np.random.seed(42)` followed by `order_sizes = np.random.lognormal(mean=3.5, sigma=0.8, size=500)`, then plots `plt.hist(order_sizes, bins=10)`. Which of the following statements are correct?

- [ ] Increasing the histogram's bin count from 10 to 30 changes the underlying `order_sizes` values themselves, not just the chart's appearance
- [x] **Calling `np.random.seed(42)` before generating `order_sizes` ensures the same sequence of "random" values is produced on every run** _(Correct)_
- [x] **A log-normal distribution generated this way is expected to be right-skewed, with most values clustered toward the lower end and a longer tail extending toward higher values** _(Correct)_
- [x] **Plotting `order_sizes` with only 10 bins is likely to obscure the distribution's shape, collapsing much of the data into one or two bars** _(Correct)_

### Answer Explanation

- **B (Correct):** Setting a seed value before generating random numbers makes the sequence reproducible — the same seed produces the same sequence on every run, which is why it's useful for consistent teaching examples and debugging.
- **C (Correct):** A log-normal distribution of this kind produces a right-skewed shape, with most values clustered toward the lower end and a longer tail toward higher values, unlike a symmetric normal distribution or a left-skewed one.
- **D (Correct):** With very few bins (around 10), most of the data collapses into one or two bars, obscuring the distribution's shape — increasing the bin count (e.g., to 30) reveals finer detail.
- **A (Incorrect):** `bins` is purely a plotting/display parameter that changes how the histogram buckets and renders the existing values — it has no effect on the underlying `order_sizes` array itself.

---

### 6. Question

A DevOps engineer plots the "store" series first with `label="store"`, then the "online" series second with `label="online"`, sets `plt.title("Sales by Channel — H1", fontsize=14)`, and calls `plt.legend(loc="upper left")`. They then reorder the code so "online" is plotted first and "store" second, keeping the title, fontsize, and legend location calls unchanged. What changes in the resulting chart?

- [x] **The default colors assigned to each series swap, since plot order (not label text) determines which color each series receives** _(Correct)_
- [ ] The title font size resets to the default of about 8, since reordering the plot calls also resets title formatting
- [ ] The legend jumps from the upper left corner to a different corner of the chart automatically after reordering
- [ ] Nothing changes visually, since matplotlib assigns colors based on the label text rather than plot order

### Answer Explanation

- **A (Correct):** Default line colors are assigned by plotting order, not by label text, so whichever series is plotted first receives the first default color and whichever is plotted second receives the next one — reordering the calls swaps which series gets which color.
- **B (Incorrect):** The `fontsize=14` argument passed to `plt.title()` is independent of plot order; it stays 14 regardless of which series was plotted first.
- **C (Incorrect):** `plt.legend(loc="upper left")` explicitly fixes the legend's position; reordering the plot calls does not move it.
- **D (Incorrect):** Default coloring in this workflow is driven by plotting order, not by the text passed to `label`.

---

### 7 Question

A backend engineer wants to draw a line chart of monthly sales sized 9 units wide by 4 units tall, with the size applied before anything is drawn. Which sequence of calls correctly produces this?

- [ ] `plt.plot(x, y); plt.figure(figsize=(9, 4)); plt.show()`
- [x] **`plt.figure(figsize=(9, 4)); plt.plot(x, y); plt.show()`** _(Correct)_
- [ ] `plt.figure(figsize=(4, 9)); plt.plot(x, y); plt.show()`
- [ ] `plt.show(figsize=(9, 4)); plt.plot(x, y)`

### Answer Explanation

- **B (Correct):** `plt.figure(figsize=(width, height))` must be called before the plotting calls so it defines the figure the line gets drawn into, and `figsize=(9, 4)` means 9 units wide and 4 units tall.
- **A (Incorrect):** Calling `plt.figure()` after `plt.plot()` starts a new, separate (empty) figure sized 9x4 rather than sizing the figure the line was drawn on.
- **C (Incorrect):** This swaps the width and height, producing a figure 4 units wide and 9 units tall instead of the requested 9x4.
- **D (Incorrect):** `figsize` is a parameter of `plt.figure()`, not `plt.show()` — `plt.show()` does not accept a `figsize` argument.

---

### 8 Question

A senior data analyst already has `region_totals = sales_df.groupby("region").agg({"online": "sum", "store": "sum"})`, a DataFrame indexed by region with online and store columns, and wants the least amount of code to draw a grouped bar chart comparing both columns per region, without manually computing bar offsets. Which call best achieves this?

- [ ] Call `plt.bar(region_totals, region_totals.index)` directly on the full DataFrame
- [ ] Manually compute `x`, a bar `width`, and offset two `plt.bar()` calls by `width/2`
- [ ] Call `plt.hist(region_totals, bins=30)` to draw a histogram of the aggregated totals
- [x] **Call `region_totals.plot(kind="bar")` directly on the aggregated DataFrame** _(Correct)_

### Answer Explanation

- **D (Correct):** Calling `.plot(kind="bar")` directly on the aggregated DataFrame draws all of its columns (`online`, `store`) as grouped bars per index value automatically, without the analyst needing to compute any offsets.
- **A (Incorrect):** `plt.bar()` expects x-positions and a single sequence of heights, not a two-column DataFrame passed as the first argument, and it would not automatically group two series side by side.
- **B (Incorrect):** This does produce a correct grouped bar chart (it's the manual approach), but it requires explicitly computing `x`, choosing a `width`, and offsetting each series — more code than the pandas shortcut, so it does not satisfy "least amount of code."
- **C (Incorrect):** `plt.hist()` draws a histogram of a single numeric distribution, not a grouped comparison of two labeled series per category.

---

### 9 Question

A product manager wants a chart where viewers can hover over data points to see exact values, zoom into a busy region, and toggle individual series (e.g., show only "north") on and off — capabilities the static charts built so far don't provide. Which library and import pattern matches the interactive tool introduced as the natural next step?

- [ ] Matplotlib, enabled by calling `plt.interactive(True)` right before `plt.show()`
- [ ] Plotly, imported using the mismatched combination `import matplotlib.express as px`
- [x] **Plotly, installed via `pip install plotly`, commonly imported as `import plotly.express as px`** _(Correct)_
- [ ] Pandas' built-in `.plot()` method, which renders output identical to Plotly's charts

### Answer Explanation

- **C (Correct):** Plotly is the interactive charting library introduced as a preview, requiring installation via `pip install plotly`, with `plotly.express` conventionally imported as `px` — and it is specifically this library that supports hovering, zooming/panning, and toggling series.
- **A (Incorrect):** No such `plt.interactive(True)` feature was described; Matplotlib's output in the material covered so far is static.
- **B (Incorrect):** This mixes up the module names — `plotly.express`, not `matplotlib.express`, is Plotly's interface.
- **D (Incorrect):** Pandas' `.plot()` method draws through Matplotlib under the hood and produces the same static output discussed earlier, not Plotly-style interactivity.

---

### 10 Question

A product manager is reviewing a chart-styling checklist before a report ships. Which of the following statements about Matplotlib customization are accurate?

- [x] **`plt.grid(alpha=0)` produces invisible gridlines, while values closer to 1 make gridlines darker and more solid** _(Correct)_
- [ ] Setting `linewidth` close to 10 makes the line almost invisible, while values close to 0 make it very thick
- [x] **Passing `color="#265..."` as a string directly to `plt.plot()` sets the line to that exact hex color** _(Correct)_
- [x] **`marker="*"` renders each data point on the chart using a star-shaped marker instead of a plain default dot** _(Correct)_
- [ ] `plt.tight_layout()` guarantees that commenting it out will always cause visibly overlapping titles, labels, and legends

### Answer Explanation

- **A (Correct):** The `alpha` argument to `plt.grid()` controls gridline opacity, with `alpha=0` invisible and values approaching 1 darker and more solid.
- **C (Correct):** `color` accepts a hex code string passed directly to `plt.plot()`.
- **D (Correct):** `marker="*"` draws each point as a star, one of the handful of standard marker shapes commonly used for readability.
- **B (Incorrect):** This reverses the actual relationship — reducing `linewidth` toward 0 produces an almost invisible line, while increasing it toward 10 makes the line very thick, like a marker-pen stroke.
- **E (Incorrect):** Commenting out `plt.tight_layout()` was shown to make no visible difference in the specific working example discussed — it remains good general practice, but the claim that overlap "always" occurs without it overstates what was demonstrated.

---
