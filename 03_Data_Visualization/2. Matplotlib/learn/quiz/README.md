# Comprehensive 100-Question Matplotlib Mastery Quiz

Welcome to the **Comprehensive 100-Question Matplotlib Mastery Quiz**. This self-assessment suite spans the entire spectrum of Matplotlib—from foundational plotting primitives and procedural `pyplot` workflows to intermediate subplots, twin axes, styling, advanced transformations, and low-level backend architectures.

---

## Quiz Instructions

1. **Attempt Each Question First:** Read each question and the provided code snippet carefully before revealing the answer.
2. **Question Types:**
   - **Single Answer:** Select exactly one best answer.
   - **Multi Answer (Select all that apply):** Select all valid options (contains at least two correct options).
3. **Check Answers & Explanations:** Click on the **`Show Answer & Explanation`** dropdown under each question to inspect the detailed technical breakdown for both correct and incorrect options.
4. **Target Difficulty Breakdown:**
   - **Questions 1–25:** Beginner (Foundations, `pyplot`, basic charts, colors, markers, limits)
   - **Questions 26–50:** Intermediate (Object-Oriented API, subplots, twin axes, dates, images, colormaps, layout)
   - **Questions 51–75:** Advanced (GridSpec, custom transforms, polar/3D, spines, patches, custom colormaps)
   - **Questions 76–100:** Expert / Advanced (Backend architecture, Artist hierarchy, blitting, event loops, memory management, custom projections)

---

## Summary Statistics

- **Total Questions:** 100
- **Single-Answer Questions:** 60
- **Multi-Answer Questions:** 40
- **Verified Matplotlib Version:** 3.11.x (compatible with modern 3.x standards)

---

## Question 1 — Beginner

**Topic:** Matplotlib Fundamentals and Architecture  
**Type:** Single Answer

Which top-level Matplotlib object acts as the overall canvas and container for all plot elements, including one or more plotting areas (Axes)?

- A. Figure
- B. Axes
- C. Axis
- D. Artist

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Figure

### Why this is correct

A `Figure` is the top-level container in Matplotlib that holds all plot elements, including one or more `Axes` objects, the figure canvas, titles, legends, and colorbars.

### Why the other options are incorrect

- **B. Axes:** `Axes` represents an individual plotting area (or subplot) where data is plotted, contained inside a `Figure`.
- **C. Axis:** `Axis` represents a single coordinate axis (such as the X-axis or Y-axis) containing ticks, tick labels, and axis limits within an `Axes`.
- **D. Artist:** `Artist` is the abstract base class for virtually all visible objects in Matplotlib (including `Figure`, `Axes`, `Line2D`, and `Text`), rather than the top-level window/container itself.

</details>

---
## Question 2 — Beginner

**Topic:** pyplot  
**Type:** Multi Answer — Select all that apply

Which of the following statements regarding the `matplotlib.pyplot` module are correct?

- A. It provides a state-machine interface that mirrors the plotting workflow of MATLAB.
- B. Functions like `plt.plot()` automatically create and track the current Figure and current Axes if none exist.
- C. It is completely separate from Matplotlib and cannot be used alongside object-oriented Axes methods.
- D. It provides convenient top-level functions such as `plt.show()`, `plt.savefig()`, and `plt.clf()`.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, D

### Why these answers are correct

- **A:** `pyplot` provides a state-based interface designed to closely emulate the procedural plotting experience of MATLAB.
- **B:** In `pyplot`'s state machine, calling functions like `plt.plot()` automatically initializes `plt.gcf()` (current figure) and `plt.gca()` (current axes) if they do not yet exist.
- **D:** `pyplot` includes global figure management utilities such as displaying (`plt.show()`), saving (`plt.savefig()`), and clearing (`plt.clf()`).

### Why the other option is incorrect

- **C:** `pyplot` is an integral module of Matplotlib and can seamlessly coexist with the object-oriented API (e.g., using `plt.subplots()` to obtain `fig` and `ax`).

</details>

---
## Question 3 — Beginner

**Topic:** Figures and Axes  
**Type:** Single Answer

Consider the following code snippet:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
```

What are the types of the returned objects `fig` and `ax`?

- A. `fig` is a `matplotlib.figure.Figure` and `ax` is a `matplotlib.axes.Axes`
- B. `fig` is a `matplotlib.axes.Axes` and `ax` is a `matplotlib.figure.Figure`
- C. `fig` is a `matplotlib.figure.Figure` and `ax` is a `numpy.ndarray` of subplots
- D. `fig` is a string identifier and `ax` is a canvas pointer

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `fig` is a `matplotlib.figure.Figure` and `ax` is a `matplotlib.axes.Axes`

### Why this is correct

When called without row or column arguments (defaulting to 1x1), `plt.subplots()` creates and returns a single `Figure` instance and a single `Axes` instance.

### Why the other options are incorrect

- **B. `fig` is a `matplotlib.axes.Axes` and `ax` is a `matplotlib.figure.Figure`:** The return order is always `(Figure, Axes)`, not `(Axes, Figure)`.
- **C. `fig` is a `matplotlib.figure.Figure` and `ax` is a `numpy.ndarray` of subplots:** `ax` is returned as a `numpy.ndarray` only when multiple subplots are requested (e.g., `nrows > 1` or `ncols > 1`), not for the default single subplot.
- **D. `fig` is a string identifier and `ax` is a canvas pointer:** Matplotlib returns rich Python object instances (`Figure` and `Axes`), not primitive string identifiers or pointers.

</details>

---
## Question 4 — Beginner

**Topic:** Creating and displaying plots  
**Type:** Single Answer

What is the primary function of `plt.show()` in a standard interactive Python script?

- A. It converts the active figure into a static PNG file on disk.
- B. It opens the GUI window to display all open figures and starts the GUI backend event loop.
- C. It resets all global `rcParams` styling parameters to default values.
- D. It prints the numerical coordinates of all plot artists to stdout.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** B. It opens the GUI window to display all open figures and starts the GUI backend event loop.

### Why this is correct

`plt.show()` displays all currently active figure windows and blocks execution (unless interactive mode is enabled) by running the backend GUI toolkit's event loop until windows are closed.

### Why the other options are incorrect

- **A. It converts the active figure into a static PNG file on disk.:** Exporting a figure to a static file on disk is handled by `plt.savefig()`, not `plt.show()`.
- **C. It resets all global `rcParams` styling parameters to default values.:** Resetting `rcParams` is achieved using `matplotlib.rcdefaults()`, not `plt.show()`.
- **D. It prints the numerical coordinates of all plot artists to stdout.:** `plt.show()` renders visual graphics in a backend canvas/window, rather than printing coordinate arrays to stdout.

</details>

---
## Question 5 — Beginner

**Topic:** Line plots  
**Type:** Single Answer

What will be plotted along the X-axis if you pass only a single 1D list to `plt.plot()` as shown below?

```python
import matplotlib.pyplot as plt

plt.plot([10, 20, 15, 30])
```

- A. Default values `[1, 2, 3, 4]`
- B. Default index values `[0, 1, 2, 3]`
- C. The sorted values `[10, 15, 20, 30]`
- D. It raises a `TypeError` because both X and Y arrays are strictly required

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** B. Default index values `[0, 1, 2, 3]`

### Why this is correct

If only a single 1D sequence is passed to `plt.plot()`, Matplotlib treats it as the Y-values and automatically generates zero-indexed sequential integers `[0, 1, 2, ..., N-1]` for the X-axis.

### Why the other options are incorrect

- **A. Default values `[1, 2, 3, 4]`:** Matplotlib uses standard 0-based indexing (`[0, 1, 2, 3]`), not 1-based indexing (`[1, 2, 3, 4]`).
- **C. The sorted values `[10, 15, 20, 30]`:** Matplotlib preserves the exact sequential order given and generates integer index coordinates; it does not sort the data.
- **D. It raises a `TypeError` because both X and Y arrays are strictly required:** `plt.plot()` accepts a single positional argument for Y-values without raising an error.

</details>

---
## Question 6 — Beginner

**Topic:** Colors, markers, and line styles  
**Type:** Multi Answer — Select all that apply

Which of the following format strings (fmt) are valid in `plt.plot(x, y, fmt)` to specify line style, color, and marker simultaneously?

- A. 'ro--' (red color, circle marker, dashed line)
- B. 'b^:' (blue color, triangle_up marker, dotted line)
- C. 'g-' (green color, solid line, no marker)
- D. 'line=dashed,color=red,marker=square'

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** 'ro--' is a valid shorthand format string combining 'r' (red), 'o' (circle marker), and '--' (dashed line).
- **B:** 'b^:' is a valid shorthand format string combining 'b' (blue), '^' (triangle_up marker), and ':' (dotted line).
- **C:** 'g-' is a valid shorthand format string specifying 'g' (green) and '-' (solid line) with no marker.

### Why the other option is incorrect

- **D:** Format strings use compact token codes (e.g. `'rs--'`); key-value syntax like `'line=dashed...'` is invalid in the positional `fmt` argument (they must be passed as keyword arguments `linestyle='--', color='r', marker='s'`).

</details>

---
## Question 7 — Beginner

**Topic:** Scatter plots  
**Type:** Single Answer

Which Matplotlib function or method is specifically designed to create a scatter plot where individual points can have variable sizes and variable colors mapped to a colormap?

- A. `plt.scatter()` / `ax.scatter()`
- B. `plt.point()` / `ax.point()`
- C. `plt.stem()` / `ax.stem()`
- D. `plt.step()` / `ax.step()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `plt.scatter()` / `ax.scatter()`

### Why this is correct

`scatter()` creates a `PathCollection` where each point can take independent sizes via the `s` parameter and colors mapped via the `c` parameter with a colormap (`cmap`).

### Why the other options are incorrect

- **B. `plt.point()` / `ax.point()`:** There is no `plt.point()` function in Matplotlib.
- **C. `plt.stem()` / `ax.stem()`:** `plt.stem()` creates a stem-and-leaf / impulse plot with vertical stems connecting data points to a baseline.
- **D. `plt.step()` / `ax.step()`:** `plt.step()` creates a step plot (piecewise-constant line plot).

</details>

---
## Question 8 — Beginner

**Topic:** Scatter plots  
**Type:** Multi Answer — Select all that apply

Which parameters of `ax.scatter(x, y, ...)` control the transparency and marker area size of data points?

- A. `alpha` controls marker transparency (from 0.0 to 1.0)
- B. `s` controls marker size (specified in points squared)
- C. `transparency` controls marker opacity (from 0 to 100)
- D. `size` controls marker radius in centimeters

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B

### Why these answers are correct

- **A:** `alpha` is a float between 0.0 (completely transparent) and 1.0 (completely opaque) that sets artist opacity.
- **B:** `s` specifies marker size as the area of the marker in points squared (pt^2).

### Why the other options are incorrect

- **C:** Matplotlib uses `alpha`, not a parameter named `transparency`.
- **D:** The marker size parameter is `s` (in points squared), not `size` in centimeters.

</details>

---
## Question 9 — Beginner

**Topic:** Bar charts  
**Type:** Single Answer

Which method is used to create a horizontal bar chart in Matplotlib?

- A. `ax.barh()`
- B. `ax.hbar()`
- C. `ax.bar(horizontal=True)`
- D. `ax.bar_horizontal()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.barh()`

### Why this is correct

`ax.barh()` (or `plt.barh()`) is the dedicated Matplotlib function for drawing horizontal bar charts, accepting parameters for Y-coordinates (`y`) and horizontal lengths (`width`).

### Why the other options are incorrect

- **B. `ax.hbar()`:** The method is named `barh()`, not `hbar()`.
- **C. `ax.bar(horizontal=True)`:** `ax.bar()` does not take a `horizontal` keyword argument; `barh()` is used instead.
- **D. `ax.bar_horizontal()`:** `bar_horizontal()` does not exist in the Matplotlib API.

</details>

---
## Question 10 — Beginner

**Topic:** Bar charts  
**Type:** Single Answer

Which parameter in `ax.bar(x, height, ...)` is used to adjust the thickness/width of the vertical bars along the X-axis?

- A. `width`
- B. `thickness`
- C. `bar_size`
- D. `span`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `width`

### Why this is correct

The `width` parameter in `ax.bar()` controls the scalar or array-like width of the vertical bars along the X-axis (defaulting to 0.8).

### Why the other options are incorrect

- **B. `thickness`:** `thickness` is not a valid parameter of `ax.bar()`.
- **C. `bar_size`:** `bar_size` does not exist in the Matplotlib API.
- **D. `span`:** `span` is not a parameter for bar width in `ax.bar()`.

</details>

---
## Question 11 — Beginner

**Topic:** Histograms  
**Type:** Single Answer

In `ax.hist(x, bins=...)`, what does passing an integer `bins=20` do?

- A. Divides the range of data into 20 equal-width bins between the minimum and maximum values
- B. Selects the first 20 data points from `x` to construct the histogram
- C. Plots only bars that have a frequency count greater than or equal to 20
- D. Sets the maximum height of the Y-axis to 20

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Divides the range of data into 20 equal-width bins between the minimum and maximum values

### Why this is correct

When `bins` is an integer N, Matplotlib automatically calculates N equal-width bin intervals spanning from `min(x)` to `max(x)`.

### Why the other options are incorrect

- **B. Selects the first 20 data points from `x` to construct the histogram:** All valid data points in `x` are included; `bins` defines the number of intervals, not the sample size.
- **C. Plots only bars that have a frequency count greater than or equal to 20:** Bin count does not filter data by frequency threshold.
- **D. Sets the maximum height of the Y-axis to 20:** Y-axis limits are computed automatically based on the resulting bin counts, not set directly by `bins`.

</details>

---
## Question 12 — Beginner

**Topic:** Histograms  
**Type:** Multi Answer — Select all that apply

Which of the following statements about `ax.hist()` are true?

- A. Setting `density=True` normalizes the bin counts so that the total integral/area under the histogram equals 1.
- B. Setting `cumulative=True` computes a cumulative histogram.
- C. Passing a sequence of numbers (e.g., `bins=[0, 5, 10, 20]`) defines custom, non-uniform bin edges.
- D. `ax.hist()` only accepts 2D image matrices and cannot process 1D arrays.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** With `density=True`, the histogram values are normalized so that the sum of each bin's count multiplied by its bin width equals 1 (probability density).
- **B:** `cumulative=True` calculates the cumulative distribution where each bin gives the count of data points in that bin plus all previous bins.
- **C:** Providing a sequence for `bins` specifies explicit boundary edges for each bin.

### Why the other option is incorrect

- **D:** `ax.hist()` is specifically designed for 1D arrays or sequences of 1D arrays.

</details>

---
## Question 13 — Beginner

**Topic:** Pie charts  
**Type:** Single Answer

In `ax.pie(values, autopct=...)`, what is the purpose of the `autopct` parameter?

- A. To format and display the percentage value label inside each slice
- B. To automatically adjust the opacity of slices based on their area
- C. To automatically compute and apply a continuous color spectrum
- D. To automatically sort slices in descending order

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. To format and display the percentage value label inside each slice

### Why this is correct

`autopct` accepts a format string (like `'%1.1f%%'`) or a callable function to calculate, format, and display the numerical percentage value directly on each pie slice.

### Why the other options are incorrect

- **B. To automatically adjust the opacity of slices based on their area:** Slice opacity is controlled via `wedgeprops` or alpha, not `autopct`.
- **C. To automatically compute and apply a continuous color spectrum:** Colors are assigned via the `colors` parameter, not `autopct`.
- **D. To automatically sort slices in descending order:** Matplotlib preserves the input order of slices starting from `startangle`; it does not auto-sort unless done manually before plotting.

</details>

---
## Question 14 — Beginner

**Topic:** Pie charts  
**Type:** Multi Answer — Select all that apply

Which of the following parameters are valid in `ax.pie()` to customize slice positioning and orientation?

- A. `explode` to offset/separate specified slices from the center
- B. `startangle` to rotate the starting slice angle counter-clockwise from the X-axis
- C. `shadow=True` to draw a drop shadow beneath the pie chart
- D. `pivot` to convert the pie chart into a 3D hemisphere

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `explode` takes a sequence of fractional radial offsets (e.g. `[0, 0.1, 0]`) to pull selected slices outward.
- **B:** `startangle` rotates the starting position of the first wedge counter-clockwise by a specified number of degrees from the positive X-axis.
- **C:** `shadow=True` renders a subtle shadow effect under the pie wedges.

### Why the other option is incorrect

- **D:** `pivot` is not a parameter of `ax.pie()`; 2D pie charts cannot be converted into 3D hemispheres via a `pivot` argument.

</details>

---
## Question 15 — Beginner

**Topic:** Box plots  
**Type:** Single Answer

In a standard Matplotlib box plot created with `ax.boxplot(data)`, what does the central line inside the box represent by default?

- A. The median (50th percentile)
- B. The arithmetic mean
- C. The standard deviation
- D. The mode

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. The median (50th percentile)

### Why this is correct

By default, the solid line dividing the interquartile range (IQR) box represents the sample median ($Q_2$).

### Why the other options are incorrect

- **B. The arithmetic mean:** The mean is not shown by default; it can be displayed as a distinct marker/line by passing `showmeans=True`.
- **C. The standard deviation:** Box plots display quartile-based summary statistics ($Q_1, Q_2, Q_3$, and IQR whiskers), not standard deviations.
- **D. The mode:** The mode is not depicted in standard Tukey box plots.

</details>

---
## Question 16 — Beginner

**Topic:** Box plots  
**Type:** Multi Answer — Select all that apply

Which of the following elements can be customized or toggled in `ax.boxplot()`?

- A. `showmeans=True` to render the arithmetic mean
- B. `notch=True` to render a notched box representing the median confidence interval
- C. `vert=False` to produce horizontal box plots
- D. `kernel='gaussian'` to overlay a kernel density estimate curve automatically

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `showmeans=True` plots the mean value using a marker or dotted line.
- **B:** `notch=True` produces a notched shape around the median representing the 95% confidence interval of the median.
- **C:** `vert=False` orientations the box plots horizontally instead of vertically.

### Why the other option is incorrect

- **D:** `ax.boxplot()` does not perform kernel density estimation; violin plots (`ax.violinplot()`) or manual KDE computation are used for density curves.

</details>

---
## Question 17 — Beginner

**Topic:** Area plots  
**Type:** Single Answer

Which Matplotlib method is commonly used to shade the area between two curves $y_1$ and $y_2$ or between a curve and a baseline?

- A. `ax.fill_between()`
- B. `ax.shade()`
- C. `ax.paint_area()`
- D. `ax.polygon_fill()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.fill_between()`

### Why this is correct

`ax.fill_between(x, y1, y2=0, ...)` fills the polygonal region between two horizontal curves $(x, y_1)$ and $(x, y_2)$ or between $(x, y_1)$ and a constant baseline.

### Why the other options are incorrect

- **B. `ax.shade()`:** `ax.shade()` is not a standard Matplotlib Axes plotting method.
- **C. `ax.paint_area()`:** `ax.paint_area()` does not exist in the Matplotlib API.
- **D. `ax.polygon_fill()`:** `ax.polygon_fill()` does not exist; generic polygons are created with `ax.fill()` or `matplotlib.patches.Polygon`.

</details>

---
## Question 18 — Beginner

**Topic:** Subplots  
**Type:** Single Answer

What subplot configuration is created by the call `plt.subplot(2, 3, 5)`?

- A. A grid of 2 rows and 3 columns, activating the 5th subplot (row 2, column 2)
- B. A grid of 2 columns and 3 rows, activating the 5th subplot
- C. A grid of 5 rows and 3 columns with 2 active axes
- D. A single plot with 2 X-axes and 3 Y-axes

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. A grid of 2 rows and 3 columns, activating the 5th subplot (row 2, column 2)

### Why this is correct

`plt.subplot(nrows, ncols, index)` creates a grid with `nrows=2` and `ncols=3` (total 6 positions, numbered 1 to 6 in row-major order). Index 5 corresponds to row 2, column 2.

### Why the other options are incorrect

- **B. A grid of 2 columns and 3 rows, activating the 5th subplot:** The first parameter is `nrows` (2 rows) and the second is `ncols` (3 columns), not the reverse.
- **C. A grid of 5 rows and 3 columns with 2 active axes:** The first parameter represents rows (2), not 5.
- **D. A single plot with 2 X-axes and 3 Y-axes:** `plt.subplot()` partitions the Figure into a grid of subplots; it does not add multiple coordinate axes to a single plot.

</details>

---
## Question 19 — Beginner

**Topic:** Subplots  
**Type:** Multi Answer — Select all that apply

Which of the following calls create a figure containing exactly 4 subplots?

- A. `fig, axes = plt.subplots(2, 2)`
- B. `fig, axes = plt.subplots(1, 4)`
- C. `fig, axes = plt.subplots(4, 1)`
- D. `fig, axes = plt.subplots(4, 4)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** 2 x 2 = 4 subplots in a 2D square grid.
- **B:** 1 x 4 = 4 subplots arranged in 1 row of 4 columns.
- **C:** 4 x 1 = 4 subplots arranged in 4 rows of 1 column.

### Why the other option is incorrect

- **D:** 4 x 4 = 16 subplots, not 4.

</details>

---
## Question 20 — Beginner

**Topic:** Titles, labels, legends, and annotations  
**Type:** Single Answer

In the object-oriented API, which method is used to set the main title of an individual `Axes` object `ax`?

- A. `ax.set_title('My Title')`
- B. `ax.title('My Title')`
- C. `ax.set_plot_title('My Title')`
- D. `ax.add_title('My Title')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.set_title('My Title')`

### Why this is correct

In the object-oriented interface, setting attributes follows the `set_<attribute>` naming convention, so `ax.set_title()` is used. `plt.title()` is the procedural `pyplot` equivalent.

### Why the other options are incorrect

- **B. `ax.title('My Title')`:** `ax.title` is an internal attribute / not a setter method; calling `ax.title(...)` raises a `TypeError`.
- **C. `ax.set_plot_title('My Title')`:** `ax.set_plot_title()` is not a valid method in Matplotlib.
- **D. `ax.add_title('My Title')`:** `ax.add_title()` is not a standard Matplotlib Axes method.

</details>

---
## Question 21 — Beginner

**Topic:** Titles, labels, legends, and annotations  
**Type:** Multi Answer — Select all that apply

Which of the following are valid ways to specify the location of a legend using `ax.legend(loc=...)`?

- A. `loc='upper right'`
- B. `loc='lower left'`
- C. `loc='best'`
- D. `loc='middle'`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** 'upper right' is a standard location string (corresponds to code 1).
- **B:** 'lower left' is a standard location string (corresponds to code 3).
- **C:** 'best' (code 0) instructs Matplotlib to automatically find a location with minimal overlap on data artists.

### Why the other option is incorrect

- **D:** 'middle' is not a valid legend string; valid center strings are `'center'`, `'center left'`, `'center right'`, `'lower center'`, and `'upper center'`.

</details>

---
## Question 22 — Beginner

**Topic:** Axis limits and ticks  
**Type:** Single Answer

Which method explicitly sets the visible data range of the X-axis on an `Axes` object `ax`?

- A. `ax.set_xlim(left, right)`
- B. `ax.set_xrange(left, right)`
- C. `ax.xlimits(left, right)`
- D. `ax.set_xbound_range(left, right)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.set_xlim(left, right)`

### Why this is correct

`ax.set_xlim(left, right)` sets the minimum (left) and maximum (right) data view limits for the X-axis.

### Why the other options are incorrect

- **B. `ax.set_xrange(left, right)`:** `ax.set_xrange()` does not exist in Matplotlib.
- **C. `ax.xlimits(left, right)`:** `ax.xlimits()` is not a valid Matplotlib method.
- **D. `ax.set_xbound_range(left, right)`:** `ax.set_xbound_range()` is not a valid Matplotlib method (the bounding method is `ax.set_xbound()`).

</details>

---
## Question 23 — Beginner

**Topic:** Grid configuration  
**Type:** Multi Answer — Select all that apply

Which of the following are valid arguments for configuring grid lines using `ax.grid()`?

- A. `visible=True` (or passing `True` positionally) to toggle the grid on
- B. `axis='x'` or `axis='y'` to draw grid lines for only one coordinate axis
- C. `linestyle='--'` and `alpha=0.5` to customize the line style and transparency
- D. `projection='grid3d'` to project 2D grid lines into 3D isometric curves

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `ax.grid(True)` or `ax.grid(visible=True)` enables grid lines.
- **B:** The `axis` argument accepts `'both'`, `'x'`, or `'y'` to target specific grid orientations.
- **C:** `linestyle` (or `ls`), `color`, `linewidth`, and `alpha` allow full customization of grid styling.

### Why the other option is incorrect

- **D:** `projection` is not a valid argument for `ax.grid()`; projections are defined when creating an `Axes` (e.g. `plt.subplots(subplot_kw={'projection': ...})`).

</details>

---
## Question 24 — Beginner

**Topic:** Styles and themes  
**Type:** Single Answer

How can you view the list of all pre-installed styles available in Matplotlib?

- A. `plt.style.available`
- B. `plt.get_styles()`
- C. `plt.theme_list()`
- D. `matplotlib.styles.list_all()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `plt.style.available`

### Why this is correct

`plt.style.available` is a list of strings containing all style sheet names packaged with Matplotlib (e.g. `'ggplot'`, `'seaborn-v0_8'`, `'dark_background'`).

### Why the other options are incorrect

- **B. `plt.get_styles()`:** `plt.get_styles()` does not exist in the Matplotlib API.
- **C. `plt.theme_list()`:** `plt.theme_list()` does not exist in Matplotlib.
- **D. `matplotlib.styles.list_all()`:** `matplotlib.styles.list_all()` does not exist; the module is `matplotlib.style` and the attribute is `.available`.

</details>

---
## Question 25 — Beginner

**Topic:** Saving figures  
**Type:** Multi Answer — Select all that apply

Which of the following statements about `plt.savefig()` are correct?

- A. It automatically infers the image file format from the filename extension (e.g., `'plot.png'`, `'plot.pdf'`, `'plot.svg'`).
- B. Calling `plt.savefig('chart.png')` after `plt.show()` in some environments may save a blank image if the figure was already cleared upon closing.
- C. The `dpi` keyword argument controls the resolution (dots per inch) for raster image outputs.
- D. `plt.savefig()` only supports uncompressed BMP files.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Matplotlib uses the file extension in the output path to automatically select the appropriate file format canvas backend (e.g. PNG, PDF, SVG, EPS).
- **B:** In interactive GUI environments, closing the GUI window after `plt.show()` clears/closes the figure, so calling `savefig()` afterwards may result in saving an empty canvas. Saving should be done prior to `plt.show()`.
- **C:** The `dpi` parameter specifies the resolution in dots per inch for raster formats like PNG and JPEG.

### Why the other option is incorrect

- **D:** `plt.savefig()` supports numerous raster and vector formats including PNG, PDF, SVG, PS, EPS, and JPEG.

</details>

---
## Question 26 — Intermediate

**Topic:** Object-oriented Matplotlib  
**Type:** Single Answer

When working with a 2x2 grid of subplots created via `fig, axes = plt.subplots(2, 2)`, what is the most idiomatic way to iterate over all individual `Axes` objects in a single flat loop?

- A. `for ax in axes.flat:` (or `axes.ravel()`)
- B. `for ax in axes.list():`
- C. `for ax in axes.all():`
- D. `for ax in axes.subplots():`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `for ax in axes.flat:` (or `axes.ravel()`)

### Why this is correct

`axes` is returned as a 2D NumPy array. Using the NumPy iterator `.flat` (or flattening the array with `.ravel()` or `.flatten()`) allows direct iteration over all 4 `Axes` objects in a single `for` loop.

### Why the other options are incorrect

- **B. `for ax in axes.list():`:** NumPy ndarrays do not have a `.list()` method.
- **C. `for ax in axes.all():`:** The `.all()` method on NumPy arrays performs a boolean evaluation, not an iteration over elements.
- **D. `for ax in axes.subplots():`:** The array of Axes does not possess a `.subplots()` method.

</details>

---
## Question 27 — Intermediate

**Topic:** Figure and Axes management  
**Type:** Multi Answer — Select all that apply

Which of the following statements about unpacking the output of `plt.subplots()` are correct?

- A. `fig, (ax1, ax2) = plt.subplots(1, 2)` successfully unpacks a 1D array of 2 horizontal subplots.
- B. `fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)` successfully unpacks a 2x2 grid into four separate variables.
- C. Setting `squeeze=False` guarantees that the returned `axes` object is always a NumPy 2D array, regardless of row/col dimensions.
- D. `plt.subplots(3, 3)` returns a Python dictionary where keys are coordinate tuples `(i, j)`.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** When `nrows=1` and `ncols=2`, `axes` is a 1D array of length 2, matching the 2-tuple `(ax1, ax2)`.
- **B:** For a 2x2 grid, `axes` is a 2D array of shape `(2, 2)`, matching the nested tuple pattern `((ax1, ax2), (ax3, ax4))`.
- **C:** By default, extra dimensions are squeezed out; `squeeze=False` forces `axes` to always be a 2D ndarray even for `1x1`, `1xN`, or `Nx1` subplots.

### Why the other option is incorrect

- **D:** `plt.subplots()` returns a NumPy `ndarray`, not a Python dictionary (subplot dictionary returns are provided by `plt.subplot_mosaic()`).

</details>

---
## Question 28 — Intermediate

**Topic:** Figure sizing and DPI  
**Type:** Single Answer

In `plt.figure(figsize=(8, 6))`, what unit of measurement is used for the dimensions `(8, 6)`?

- A. Inches
- B. Centimeters
- C. Pixels
- D. Millimeters

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Inches

### Why this is correct

In Matplotlib, `figsize` dimensions are strictly defined in **inches** `(width, height)`. At a standard DPI of 100, `(8, 6)` corresponds to $800 \times 600$ pixels.

### Why the other options are incorrect

- **B. Centimeters:** Matplotlib does not use centimeters natively for `figsize` (conversion requires dividing cm by 2.54).
- **C. Pixels:** Pixel dimensions depend on the DPI (`pixels = inches * dpi`), whereas `figsize` sets physical inches.
- **D. Millimeters:** Millimeters are not the native unit for `figsize`.

</details>

---
## Question 29 — Intermediate

**Topic:** Axis limits and ticks  
**Type:** Single Answer

What is the recommended object-oriented method combination to place major tick marks at specific numeric locations and assign custom string labels to them?

- A. `ax.set_xticks([0, 1, 2])` followed by `ax.set_xticklabels(['A', 'B', 'C'])` (or `ax.set_xticks(..., labels=...)`)
- B. `ax.ticks([0, 1, 2], labels=['A', 'B', 'C'])`
- C. `ax.set_axis_labels([0, 1, 2], ['A', 'B', 'C'])`
- D. `ax.x_markers([0, 1, 2], text=['A', 'B', 'C'])`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.set_xticks([0, 1, 2])` followed by `ax.set_xticklabels(['A', 'B', 'C'])` (or `ax.set_xticks(..., labels=...)`)

### Why this is correct

`ax.set_xticks(ticks, labels=...)` (or calling `set_xticks()` followed by `set_xticklabels()`) explicitly specifies the tick coordinate positions and their corresponding display labels.

### Why the other options are incorrect

- **B. `ax.ticks([0, 1, 2], labels=['A', 'B', 'C'])`:** `ax.ticks()` does not exist in the Matplotlib API.
- **C. `ax.set_axis_labels([0, 1, 2], ['A', 'B', 'C'])`:** `ax.set_axis_labels()` does not exist; axis labels are set using `ax.set_xlabel()` and `ax.set_ylabel()`.
- **D. `ax.x_markers([0, 1, 2], text=['A', 'B', 'C'])`:** `ax.x_markers()` is not a valid method.

</details>

---
## Question 30 — Intermediate

**Topic:** Tick formatting  
**Type:** Multi Answer — Select all that apply

Which classes from the `matplotlib.ticker` module are used for controlling tick positioning (locators) and tick text formatting (formatters)?

- A. `MultipleLocator` to place ticks at regular arithmetic multiples
- B. `FuncFormatter` to format tick labels using a custom Python function
- C. `PercentFormatter` to display tick values as percentages automatically
- D. `AutoFontLocator` to dynamically alter the typeface of axis ticks

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `MultipleLocator(base)` places ticks at multiples of `base` (e.g. every 5 or 10 units).
- **B:** `FuncFormatter(func)` wraps a user-defined function `func(x, pos)` to construct custom string labels.
- **C:** `PercentFormatter(xmax=...)` automatically converts numeric values into percentage strings.

### Why the other option is incorrect

- **D:** `AutoFontLocator` does not exist in Matplotlib; font properties are configured via text properties or `rcParams`.

</details>

---
## Question 31 — Intermediate

**Topic:** Logarithmic and other axis scales  
**Type:** Single Answer

How do you set the Y-axis to a base-10 logarithmic scale on an `Axes` object `ax`?

- A. `ax.set_yscale('log')`
- B. `ax.set_log_y(10)`
- C. `ax.scale_y('log10')`
- D. `ax.y_logarithm(True)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.set_yscale('log')`

### Why this is correct

`ax.set_yscale('log')` (or `plt.yscale('log')`) sets the scale of the Y-axis to logarithmic (base 10 by default, customizable via `base=...`).

### Why the other options are incorrect

- **B. `ax.set_log_y(10)`:** `ax.set_log_y()` is not a Matplotlib method.
- **C. `ax.scale_y('log10')`:** `ax.scale_y()` is not a valid method.
- **D. `ax.y_logarithm(True)`:** `ax.y_logarithm()` does not exist in Matplotlib.

</details>

---
## Question 32 — Intermediate

**Topic:** Twin axes  
**Type:** Single Answer

Which method is used to create a second Y-axis that shares the exact same X-axis as an existing `Axes` object `ax`?

- A. `ax2 = ax.twinx()`
- B. `ax2 = ax.twiny()`
- C. `ax2 = ax.duplicate(axis='y')`
- D. `ax2 = ax.clone_x()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax2 = ax.twinx()`

### Why this is correct

`ax.twinx()` creates and returns a new `Axes` sharing the X-axis of `ax`, while providing an independent Y-axis on the opposite (right) side.

### Why the other options are incorrect

- **B. `ax2 = ax.twiny()`:** `ax.twiny()` shares the Y-axis and creates a secondary X-axis (on the top), which is the opposite.
- **C. `ax2 = ax.duplicate(axis='y')`:** `ax.duplicate()` does not exist in Matplotlib.
- **D. `ax2 = ax.clone_x()`:** `ax.clone_x()` is not a valid method in Matplotlib.

</details>

---
## Question 33 — Intermediate

**Topic:** Twin axes  
**Type:** Multi Answer — Select all that apply

When using `ax2 = ax1.twinx()`, which of the following statements are true?

- A. Both axes overlay the exact same physical bounding box inside the figure.
- B. `ax1` and `ax2` have independent Y-axis limits (`ylim`) and tick formatters.
- C. Changes to the X-axis limits on `ax1` automatically update on `ax2` because they share the X-axis.
- D. Calling `ax1.plot()` and `ax2.plot()` automatically consolidates both lines into a single combined legend when calling `ax1.legend()`.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `twinx()` places `ax2` directly over `ax1` with identical position geometry.
- **B:** Each twin axes instance maintains its own separate `YAxis` object with distinct limits, scales, and ticks.
- **C:** The shared X-axis ensures that pan/zoom or limit modifications to X on either axes synchronize immediately.

### Why the other option is incorrect

- **D:** Because `ax1` and `ax2` are distinct `Axes` objects, `ax1.legend()` only inspects artists attached to `ax1`. Combining legends requires collecting handles and labels from both axes manually: `lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(lines1 + lines2, labels1 + labels2)`.

</details>

---
## Question 34 — Intermediate

**Topic:** Shared axes  
**Type:** Single Answer

What is the primary effect of passing `sharex=True` to `plt.subplots(2, 1, sharex=True)`?

- A. Both subplots share their X-axis view limits and panning/zooming, and tick labels on the upper subplot are hidden by default
- B. Both subplots are merged into a single 3D plot
- C. Data arrays passed to subplot 1 are automatically copied to subplot 2
- D. The X-axis ticks are drawn only on the rightmost margin

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Both subplots share their X-axis view limits and panning/zooming, and tick labels on the upper subplot are hidden by default

### Why this is correct

`sharex=True` links the X-axis coordinate transformations across all subplots, synchronizing limits/zooming and cleanly hiding redundant inner X-tick labels.

### Why the other options are incorrect

- **B. Both subplots are merged into a single 3D plot:** Shared axes link 2D coordinate bounds; they do not convert subplots into 3D plots.
- **C. Data arrays passed to subplot 1 are automatically copied to subplot 2:** Data plotted on each subplot remains completely independent.
- **D. The X-axis ticks are drawn only on the rightmost margin:** X-axis ticks remain along the bottom edge of the bottom subplot, not the right margin.

</details>

---
## Question 35 — Intermediate

**Topic:** Shared axes  
**Type:** Multi Answer — Select all that apply

Which of the following are valid ways to share axes between subplots in Matplotlib?

- A. Passing `sharex=True` or `sharey=True` in `plt.subplots()`
- B. Passing an existing `Axes` instance to the `sharex` or `sharey` argument of `plt.subplot()` or `fig.add_subplot()`
- C. Calling `ax2.sharex(ax1)` explicitly after creating both axes
- D. Setting `ax1.merge_axes(ax2)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `plt.subplots(nrows, ncols, sharex=True, sharey=True)` is the high-level standard for sharing across grids.
- **B:** `fig.add_subplot(2, 1, 2, sharex=ax1)` directly links `ax2`'s X-axis to `ax1` during construction.
- **C:** `ax2.sharex(ax1)` links axes programmatically after creation.

### Why the other option is incorrect

- **D:** `merge_axes()` is not a Matplotlib method.

</details>

---
## Question 36 — Intermediate

**Topic:** Date/time plotting  
**Type:** Single Answer

Which module in Matplotlib provides specialized locators and formatters for plotting Python `datetime` and NumPy `datetime64` objects?

- A. `matplotlib.dates`
- B. `matplotlib.time`
- C. `matplotlib.chronos`
- D. `matplotlib.calendar`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.dates`

### Why this is correct

`matplotlib.dates` contains date ticker locators (such as `DayLocator`, `MonthLocator`, `YearLocator`) and formatters (like `DateFormatter`, `ConciseDateFormatter`).

### Why the other options are incorrect

- **B. `matplotlib.time`:** Matplotlib does not have a `matplotlib.time` module.
- **C. `matplotlib.chronos`:** `matplotlib.chronos` does not exist.
- **D. `matplotlib.calendar`:** `matplotlib.calendar` does not exist.

</details>

---
## Question 37 — Intermediate

**Topic:** Date/time plotting  
**Type:** Multi Answer — Select all that apply

When plotting time series with crowded date tick labels on the X-axis, which techniques help prevent overlapping text?

- A. Calling `fig.autofmt_xdate()` to automatically rotate tick labels and adjust horizontal alignment
- B. Using `matplotlib.dates.ConciseDateFormatter` for succinct, non-redundant date formatting
- C. Using `ax.tick_params(axis='x', rotation=45)` to rotate tick labels manually
- D. Calling `ax.set_dates_invisible(True)` to purge all time series data from memory

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `fig.autofmt_xdate()` tilts X-tick labels (by default 30 degrees) and aligns them neatly to avoid clipping.
- **B:** `ConciseDateFormatter` displays dynamic date strings (e.g. showing years only when year changes), drastically reducing label length.
- **C:** `ax.tick_params(axis='x', rotation=45)` provides direct control over tick text rotation angle.

### Why the other option is incorrect

- **D:** `ax.set_dates_invisible()` does not exist and deleting data would prevent visualization.

</details>

---
## Question 38 — Intermediate

**Topic:** Text and annotations  
**Type:** Single Answer

What is the role of the `transform` parameter when calling `ax.text(x, y, 'Label', transform=ax.transAxes)`?

- A. It specifies that coordinates `(x, y)` are relative to the Axes box (0,0 = bottom-left, 1,1 = top-right) rather than data coordinates
- B. It applies a 3D isometric rotation to the text string
- C. It translates the text from English to another language
- D. It forces the font to transform to LaTeX mode

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. It specifies that coordinates `(x, y)` are relative to the Axes box (0,0 = bottom-left, 1,1 = top-right) rather than data coordinates

### Why this is correct

`transform=ax.transAxes` changes the coordinate system so that `(0, 0)` is the bottom-left corner of the `Axes` and `(1, 1)` is the top-right corner, regardless of the data limits.

### Why the other options are incorrect

- **B. It applies a 3D isometric rotation to the text string:** Text rotation is controlled via `rotation=...`, not coordinate transforms.
- **C. It translates the text from English to another language:** Transforms in Matplotlib handle mathematical coordinate mappings, not linguistic translation.
- **D. It forces the font to transform to LaTeX mode:** LaTeX text rendering is toggled via `usetex=True` or `matplotlib.rcParams['text.usetex']`.

</details>

---
## Question 39 — Intermediate

**Topic:** Titles, labels, legends, and annotations  
**Type:** Multi Answer — Select all that apply

In `ax.annotate(text, xy=..., xytext=..., arrowprops=...)`, what do `xy` and `xytext` represent?

- A. `xy` specifies the point being annotated (the tip of the arrow)
- B. `xytext` specifies the position where the text label is placed
- C. `arrowprops` is a dictionary configuring properties (such as color, arrowstyle, width) of the connecting arrow
- D. `xy` and `xytext` must always be identical or an error is raised

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `xy` is the coordinate tuple of the target point being highlighted (where the arrow points).
- **B:** `xytext` is the coordinate tuple where the text string is positioned.
- **C:** `arrowprops` configures arrow styling such as `arrowstyle='->'`, `color='black'`, and `lw=1.5`.

### Why the other option is incorrect

- **D:** `xy` and `xytext` are typically different so that the text is offset from the target point and connected via the arrow; if they were always identical, an arrow could not be drawn.

</details>

---
## Question 40 — Intermediate

**Topic:** Custom legends  
**Type:** Single Answer

How can you place a legend completely outside the plotting area (e.g. to the upper right outside the axes)?

- A. `ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`
- B. `ax.legend(outside=True)`
- C. `ax.legend(position='external')`
- D. `ax.legend(margin='right')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`

### Why this is correct

`bbox_to_anchor=(1.05, 1)` paired with `loc='upper left'` anchors the top-left corner of the legend box just outside the right edge (x = 1.05) of the axes bounding box.

### Why the other options are incorrect

- **B. `ax.legend(outside=True)`:** `outside=True` is not a standard Matplotlib keyword argument for `ax.legend()`.
- **C. `ax.legend(position='external')`:** `position='external'` is not a valid parameter.
- **D. `ax.legend(margin='right')`:** `margin='right'` is not a valid legend positioning parameter.

</details>

---
## Question 41 — Intermediate

**Topic:** Colormaps  
**Type:** Multi Answer — Select all that apply

Which of the following colormaps in Matplotlib are classified as perceptually uniform sequential colormaps (designed to be colorblind-friendly and monotonically increase in luminance)?

- A. `viridis`
- B. `plasma`
- C. `inferno`
- D. `jet`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `viridis` is the default perceptually uniform sequential colormap introduced in Matplotlib 2.0.
- **B:** `plasma` is a perceptually uniform sequential colormap transitioning from purple to yellow.
- **C:** `inferno` is a perceptually uniform sequential colormap transitioning from black/purple through orange to yellow.

### Why the other option is incorrect

- **D:** `jet` is an older rainbow colormap with non-monotonic luminance gradients and perceptual artifacts; it is not perceptually uniform.

</details>

---
## Question 42 — Intermediate

**Topic:** imshow()  
**Type:** Single Answer

What kind of data does `ax.imshow(X)` display when `X` is a 2D NumPy array of shape `(M, N)`?

- A. A pseudocolor raster image / heatmap where array values are mapped to colors via a colormap
- B. A 3D mesh surface plot
- C. A scatter plot of M points each with N dimensions
- D. A bar chart with M rows and N columns

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. A pseudocolor raster image / heatmap where array values are mapped to colors via a colormap

### Why this is correct

For a 2D array of scalar values, `ax.imshow()` renders a raster image / heatmap by normalizing array values and mapping them through the active colormap (`cmap`).

### Why the other options are incorrect

- **B. A 3D mesh surface plot:** 3D mesh surfaces are generated with `Axes3D.plot_surface()`, not `imshow()`.
- **C. A scatter plot of M points each with N dimensions:** Scatter plots are created with `ax.scatter()`.
- **D. A bar chart with M rows and N columns:** Bar charts are created with `ax.bar()`.

</details>

---
## Question 43 — Intermediate

**Topic:** Images and heatmaps  
**Type:** Multi Answer — Select all that apply

Which parameters in `ax.imshow()` control pixel aspect ratio and vertical orientation?

- A. `origin='lower'` to place the `[0, 0]` index at the bottom-left corner (standard Cartesian orientation)
- B. `origin='upper'` to place the `[0, 0]` index at the top-left corner (standard matrix/image orientation)
- C. `aspect='auto'` to allow pixels to stretch non-square to fill the axes bounding box
- D. `aspect='isometric3d'` to project image pixels into perspective voxels

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `origin='lower'` aligns array row index 0 with the bottom of the plot, commonly used for Cartesian coordinate data.
- **B:** `origin='upper'` is the default for `imshow()`, placing row 0 at the top (matching matrix index and computer vision conventions).
- **C:** `aspect='auto'` unconstrains pixel aspect ratio so the image stretches to fill the available axes dimensions.

### Why the other option is incorrect

- **D:** `aspect` accepts `'equal'`, `'auto'`, or a float ratio; `'isometric3d'` is invalid.

</details>

---
## Question 44 — Intermediate

**Topic:** Colorbars  
**Type:** Single Answer

Consider the following code snippet:

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
data = np.random.rand(10, 10)
im = ax.imshow(data, cmap='viridis')
fig.colorbar(im, ax=ax)
```

What is the first argument passed to `fig.colorbar()`?

- A. A `ScalarMappable` artist (such as the `AxesImage` returned by `imshow` or a `PathCollection` from `scatter`)
- B. The raw 2D NumPy array `data`
- C. The string name of the colormap `'viridis'`
- D. The `Axes` object `ax`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. A `ScalarMappable` artist (such as the `AxesImage` returned by `imshow` or a `PathCollection` from `scatter`)

### Why this is correct

`fig.colorbar(mappable, ax=...)` requires a `ScalarMappable` instance (such as the object returned by `imshow()`, `contourf()`, or `scatter()`) to extract the colormap and normalization limits.

### Why the other options are incorrect

- **B. The raw 2D NumPy array `data`:** Passing raw data arrays raises a `TypeError`; the colorbar requires the mappable artist that binds normalization and colormap.
- **C. The string name of the colormap `'viridis'`:** Colormap names cannot be passed alone without a ScalarMappable.
- **D. The `Axes` object `ax`:** `ax` is supplied to the `ax` keyword argument to indicate which subplot should be shrunk to accommodate the colorbar, not as the first positional argument.

</details>

---
## Question 45 — Intermediate

**Topic:** Contour plots  
**Type:** Multi Answer — Select all that apply

What is the primary difference between `ax.contour(X, Y, Z)` and `ax.contourf(X, Y, Z)`?

- A. `ax.contour()` draws isolines (contour lines) at specified level boundaries.
- B. `ax.contourf()` draws filled polygonal regions between contour levels.
- C. Both functions require 2D coordinate grids for `X`, `Y`, and `Z` (such as generated by `np.meshgrid`).
- D. `ax.contourf()` is strictly for 3D surfaces while `ax.contour()` is strictly 2D.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `contour()` produces unfilled contour line artists (`QuadContourSet`).
- **B:** The 'f' in `contourf()` stands for 'filled', creating solid colored regions between level thresholds.
- **C:** Both accept 2D matrices representing the spatial coordinate grid and field elevations.

### Why the other option is incorrect

- **D:** Both `contour()` and `contourf()` are 2D visualization functions (though 3D projections also support `ax.contour()` at specific planes).

</details>

---
## Question 46 — Intermediate

**Topic:** tight_layout()  
**Type:** Single Answer

What problem does calling `plt.tight_layout()` or `fig.tight_layout()` primarily solve?

- A. It automatically adjusts subplot parameters and margins so that titles, axis labels, and tick labels do not overlap adjacent subplots
- B. It compresses the output file size by reducing color bit-depth
- C. It snaps all data markers to the nearest integer grid coordinate
- D. It forces all subplots to have identical aspect ratios and data limits

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. It automatically adjusts subplot parameters and margins so that titles, axis labels, and tick labels do not overlap adjacent subplots

### Why this is correct

`tight_layout()` dynamically inspects the bounding boxes of text artists (tick labels, axis labels, titles) and recalculates subplot spacings to eliminate overlaps.

### Why the other options are incorrect

- **B. It compresses the output file size by reducing color bit-depth:** Image compression is managed by file export libraries during `savefig()`, not `tight_layout()`.
- **C. It snaps all data markers to the nearest integer grid coordinate:** Data coordinates are unaffected by layout management.
- **D. It forces all subplots to have identical aspect ratios and data limits:** Aspect ratios and limits are determined by data and axis settings, not `tight_layout()`.

</details>

---
## Question 47 — Intermediate

**Topic:** constrained_layout  
**Type:** Multi Answer — Select all that apply

Which statements regarding `layout='constrained'` (or `constrained_layout=True`) in modern Matplotlib are correct?

- A. It can be activated during figure initialization via `plt.subplots(layout='constrained')` or `plt.figure(layout='constrained')`.
- B. Unlike `tight_layout()`, it dynamically re-evaluates and maintains clean spacing even when figure dimensions change before rendering.
- C. It seamlessly handles complex layouts containing colorbars and multi-row/col legends attached to subplots.
- D. It is completely incompatible with `GridSpec` and raises an `ImportError` whenever subplots are present.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Modern Matplotlib supports `layout='constrained'` (and `layout='compressed'`) directly in figure constructors.
- **B:** Constrained layout utilizes a solver that optimizes geometry continuously before each draw rather than computing static margins once.
- **C:** It automatically reserves margin space for associated colorbars and legends without distorting adjacent subplot alignments.

### Why the other option is incorrect

- **D:** Constrained layout works seamlessly with `GridSpec` and `subplot_mosaic()`.

</details>

---
## Question 48 — Intermediate

**Topic:** Figure sizing and DPI  
**Type:** Single Answer

If a figure is created with `plt.figure(figsize=(10, 5), dpi=200)`, what will be the pixel dimensions of the resulting rendered raster image?

- A. 2000 pixels wide by 1000 pixels high
- B. 1000 pixels wide by 500 pixels high
- C. 500 pixels wide by 250 pixels high
- D. 200 pixels wide by 200 pixels high

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. 2000 pixels wide by 1000 pixels high

### Why this is correct

Pixel dimension equals $\text{inches} \times \text{DPI}$. Thus, width $= 10 \times 200 = 2000$ pixels, and height $= 5 \times 200 = 1000$ pixels.

### Why the other options are incorrect

- **B. 1000 pixels wide by 500 pixels high:** This would correspond to a DPI of 100 ($10 \times 100 = 1000$).
- **C. 500 pixels wide by 250 pixels high:** This would correspond to a DPI of 50.
- **D. 200 pixels wide by 200 pixels high:** The calculation does not multiply the dimensions by the specified DPI.

</details>

---
## Question 49 — Intermediate

**Topic:** Saving figures  
**Type:** Single Answer

Which argument in `plt.savefig()` automatically computes a snug bounding box around all plotted artists and eliminates excessive surrounding whitespace from the saved file?

- A. `bbox_inches='tight'`
- B. `autocrop=True`
- C. `fit='tight'`
- D. `trim_whitespace=True`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `bbox_inches='tight'`

### Why this is correct

Passing `bbox_inches='tight'` instructs Matplotlib to calculate the bounding box containing all visible artists (including legends, titles, and labels that might otherwise be clipped) and crop the canvas accordingly.

### Why the other options are incorrect

- **B. `autocrop=True`:** `autocrop` is not a parameter of `plt.savefig()`.
- **C. `fit='tight'`:** `fit='tight'` does not exist in the Matplotlib API.
- **D. `trim_whitespace=True`:** `trim_whitespace` is not a valid parameter.

</details>

---
## Question 50 — Intermediate

**Topic:** File formats  
**Type:** Single Answer

Which of the following saved file formats preserves infinite scalability without pixelation or resolution loss because it stores geometric drawing commands rather than a grid of pixels?

- A. SVG (Scalable Vector Graphics) and PDF (Portable Document Format)
- B. PNG (Portable Network Graphics)
- C. JPEG (Joint Photographic Experts Group)
- D. GIF (Graphics Interchange Format)

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. SVG (Scalable Vector Graphics) and PDF (Portable Document Format)

### Why this is correct

SVG, PDF, and EPS are vector formats that store shapes, paths, lines, and text as mathematical primitives, allowing infinite scaling without pixelation.

### Why the other options are incorrect

- **B. PNG (Portable Network Graphics):** PNG is a raster bitmap format consisting of a fixed pixel grid; scaling causes pixelation.
- **C. JPEG (Joint Photographic Experts Group):** JPEG is a lossy raster bitmap format with fixed pixel resolution.
- **D. GIF (Graphics Interchange Format):** GIF is an 8-bit raster bitmap format.

</details>

---
## Question 51 — Advanced

**Topic:** Layout management  
**Type:** Single Answer

Which Matplotlib class enables the creation of multi-column and multi-row spanning subplots using NumPy-like slicing on a figure grid?

- A. `matplotlib.gridspec.GridSpec`
- B. `matplotlib.layout.FlexGrid`
- C. `matplotlib.figure.SubplotArray`
- D. `matplotlib.axes.GridSpan`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.gridspec.GridSpec`

### Why this is correct

`GridSpec` defines a flexible grid structure across a `Figure`. Subplots can span multiple rows and columns using slice syntax (e.g. `ax = fig.add_subplot(gs[0, :])` or `ax = fig.add_subplot(gs[1:, 0])`).

### Why the other options are incorrect

- **B. `matplotlib.layout.FlexGrid`:** `FlexGrid` does not exist in Matplotlib.
- **C. `matplotlib.figure.SubplotArray`:** `SubplotArray` does not exist in Matplotlib.
- **D. `matplotlib.axes.GridSpan`:** `GridSpan` does not exist in Matplotlib.

</details>

---
## Question 52 — Advanced

**Topic:** Subplots  
**Type:** Multi Answer — Select all that apply

Consider the modern `subplot_mosaic()` feature:

```python
fig, axd = plt.subplot_mosaic([['left', 'upper_right'],
                               ['left', 'lower_right']])
```

Which of the following statements about this code are correct?

- A. `axd` is returned as a Python dictionary where keys `'left'`, `'upper_right'`, and `'lower_right'` map to their respective `Axes` objects.
- B. The `'left'` subplot spans across both rows in the first column.
- C. Using `'.'` in place of a string label in the layout matrix creates an empty whitespace cell where no Axes is added.
- D. `subplot_mosaic()` converts all axes to 3D automatically.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `subplot_mosaic` returns an `axd` dictionary with key names matching the ASCII/nested list structure.
- **B:** Repeating `'left'` in rows 0 and 1 creates a single contiguous Axes spanning both rows.
- **C:** The dot `'.'` character acts as an intentional empty placeholder within mosaic specifications.

### Why the other option is incorrect

- **D:** Axes created by `subplot_mosaic()` are standard 2D axes unless a custom projection is specified in `per_subplot_kw`.

</details>

---
## Question 53 — Advanced

**Topic:** Multiple axes  
**Type:** Single Answer

What is the recommended modern method to create an embedded inset zoom plot inside an existing `Axes` object `ax`?

- A. `axin = ax.inset_axes([x0, y0, width, height])`
- B. `axin = ax.create_zoom_box([x0, y0, width, height])`
- C. `axin = ax.add_subaxes([x0, y0, width, height])`
- D. `axin = ax.embed_axes([x0, y0, width, height])`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `axin = ax.inset_axes([x0, y0, width, height])`

### Why this is correct

`ax.inset_axes([x0, y0, width, height])` creates a child `Axes` positioned in normalized axes coordinates $(0 \text{ to } 1)$ relative to the parent `ax`, and pairs with `ax.indicate_inset_zoom(axin)` to draw connection lines.

### Why the other options are incorrect

- **B. `axin = ax.create_zoom_box([x0, y0, width, height])`:** `create_zoom_box()` is not a Matplotlib method.
- **C. `axin = ax.add_subaxes([x0, y0, width, height])`:** `add_subaxes()` does not exist in Matplotlib.
- **D. `axin = ax.embed_axes([x0, y0, width, height])`:** `embed_axes()` does not exist in Matplotlib.

</details>

---
## Question 54 — Advanced

**Topic:** Figure and Axes management  
**Type:** Multi Answer — Select all that apply

Which of the following code snippets correctly manipulate the perimeter border lines (spines) of an `Axes` object `ax`?

- A. `ax.spines['top'].set_visible(False)` and `ax.spines['right'].set_visible(False)` to remove top and right borders
- B. `ax.spines['left'].set_position(('data', 0))` to center the Y-axis spine at $x=0$
- C. `ax.spines['bottom'].set_position(('outward', 10))` to offset the X-axis spine 10 points outward
- D. `ax.spines.delete_all()` to purge all artists from memory

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Setting `set_visible(False)` on individual spines is the standard approach to creating minimalist / Tufte-style plots.
- **B:** Spine positions can be bound to data coordinates using `('data', value)`.
- **C:** Spines can be displaced outward by a point offset using `('outward', points)`.

### Why the other option is incorrect

- **D:** `ax.spines` is a container dictionary mapping spine names (`'left'`, `'right'`, `'bottom'`, `'top'`) to `Spine` objects; it does not have a `delete_all()` method.

</details>

---
## Question 55 — Advanced

**Topic:** Tick formatting  
**Type:** Single Answer

When defining a custom tick formatting function for `matplotlib.ticker.FuncFormatter`, what are the expected parameters that the user function must accept?

- A. `def custom_formatter(x, pos):` where `x` is the tick numeric value and `pos` is the tick index position
- B. `def custom_formatter(text, font):` where `text` is the raw string and `font` is the font family
- C. `def custom_formatter(ax, fig):` where `ax` is the parent axes and `fig` is the parent figure
- D. `def custom_formatter(val):` requiring strictly a single scalar input

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `def custom_formatter(x, pos):` where `x` is the tick numeric value and `pos` is the tick index position

### Why this is correct

A `FuncFormatter` callback must accept two arguments: the numeric tick value `x` (float) and the integer position index `pos` (which may be `None` in some render passes), and return a formatted string.

### Why the other options are incorrect

- **B. `def custom_formatter(text, font):` where `text` is the raw string and `font` is the font family:** Formatters receive numeric values and tick index positions, not string/font objects.
- **C. `def custom_formatter(ax, fig):` where `ax` is the parent axes and `fig` is the parent figure:** Axes and Figure references are not passed to tick formatting callbacks.
- **D. `def custom_formatter(val):` requiring strictly a single scalar input:** Passing a 1-parameter function without a default argument for `pos` will raise a `TypeError: custom_formatter() takes 1 positional argument but 2 were given` when rendered by the ticker engine.

</details>

---
## Question 56 — Advanced

**Topic:** Twin axes  
**Type:** Multi Answer — Select all that apply

When creating three or more Y-axes on a single plot by chaining `ax.twinx()`, which steps are necessary to ensure the additional Y-axis ticks and labels do not overlap the first twin axis?

- A. Use `ax3.spines['right'].set_position(('outward', 60))` to displace the spine outward.
- B. Ensure `ax3.spines['right'].set_visible(True)` is enabled if spines were altered.
- C. Set `ax3.patch.set_visible(False)` (or ensure the underlying axes patch is transparent) so `ax3` does not block `ax1` and `ax2`.
- D. Call `ax3.set_multiaxis(True)` to enable 3D depth-sorting.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Shifting the spine outward via `('outward', offset)` prevents the third axis from colliding directly with the secondary right spine.
- **B:** The right spine on `ax3` must be visible to display the offset tick line.
- **C:** Twin axes have transparent background patches by default; keeping them transparent ensures underlying plots remain visible.

### Why the other option is incorrect

- **D:** `set_multiaxis()` is not a Matplotlib method.

</details>

---
## Question 57 — Advanced

**Topic:** Polar plots  
**Type:** Single Answer

How do you create an `Axes` configured for polar coordinates $(r, \theta)$ in Matplotlib?

- A. `fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})` (or `plt.subplot(111, polar=True)`)
- B. `fig, ax = plt.subplots(projection='radial')`
- C. `fig, ax = plt.subplots(mode='circular')`
- D. `fig, ax = plt.subplots(coordinate_system='spherical')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})` (or `plt.subplot(111, polar=True)`)

### Why this is correct

Specifying `projection='polar'` (or `polar=True`) creates a `PolarAxes` where angles $\theta$ are plotted azimuthally and radii $r$ are plotted radially.

### Why the other options are incorrect

- **B. `fig, ax = plt.subplots(projection='radial')`:** The registered projection identifier is `'polar'`, not `'radial'`.
- **C. `fig, ax = plt.subplots(mode='circular')`:** `mode` is not a valid parameter for `plt.subplots()`.
- **D. `fig, ax = plt.subplots(coordinate_system='spherical')`:** `coordinate_system` is not a valid subplot parameter.

</details>

---
## Question 58 — Advanced

**Topic:** Polar plots  
**Type:** Multi Answer — Select all that apply

Which of the following statements about polar plotting with `ax.plot(theta, r)` on a `PolarAxes` are correct?

- A. The angular coordinate `theta` must be provided in radians by default.
- B. `ax.set_theta_zero_location('N')` sets the 0-degree reference angle to point North (top).
- C. `ax.set_theta_direction(-1)` configures angles to increase clockwise instead of the default counter-clockwise.
- D. Polar axes cannot display grid lines or legends.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Matplotlib's `PolarAxes` trigonometric computations assume angle coordinates in radians.
- **B:** `set_theta_zero_location()` accepts cardinal directions (`'N'`, `'E'`, `'S'`, `'W'`, etc.) to position $0^\circ$.
- **C:** `set_theta_direction(1)` is counter-clockwise and `set_theta_direction(-1)` is clockwise.

### Why the other option is incorrect

- **D:** Polar axes fully support radial/azimuthal grid lines, tick formatters, and legends.

</details>

---
## Question 59 — Advanced

**Topic:** 3D plotting  
**Type:** Single Answer

Which method is the standard object-oriented approach to initialize a 3D subplot in modern Matplotlib?

- A. `fig, ax = plt.subplots(subplot_kw={'projection': '3d'})`
- B. `fig, ax = plt.subplots(3d=True)`
- C. `fig, ax = plt.subplots(dimension=3)`
- D. `fig, ax = plt.subplots(engine='opengl3d')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `fig, ax = plt.subplots(subplot_kw={'projection': '3d'})`

### Why this is correct

Passing `subplot_kw={'projection': '3d'}` initializes an `Axes3D` instance from the `mpl_toolkits.mplot3d` toolkit inside standard `plt.subplots()`.

### Why the other options are incorrect

- **B. `fig, ax = plt.subplots(3d=True)`:** `3d=True` is not valid Python syntax (identifiers cannot begin with a number) and is not a parameter.
- **C. `fig, ax = plt.subplots(dimension=3)`:** `dimension` is not a valid argument for `plt.subplots()`.
- **D. `fig, ax = plt.subplots(engine='opengl3d')`:** `engine` is not a parameter of `plt.subplots()`.

</details>

---
## Question 60 — Advanced

**Topic:** mplot3d  
**Type:** Single Answer

Which method on an `Axes3D` object `ax` is specifically used to render a continuous, shaded 3D polygonal surface mesh with colormapping and surface lighting?

- A. `ax.plot_surface(X, Y, Z)`
- B. `ax.plot_wireframe(X, Y, Z)`
- C. `ax.draw_mesh3d(X, Y, Z)`
- D. `ax.render_surface(X, Y, Z)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.plot_surface(X, Y, Z)`

### Why this is correct

`ax.plot_surface(X, Y, Z, cmap=..., rstride=..., cstride=...)` generates a fully shaded 3D polygonal surface mesh from coordinate matrices `X`, `Y`, and `Z`.

### Why the other options are incorrect

- **B. `ax.plot_wireframe(X, Y, Z)`:** `ax.plot_wireframe()` draws only unshaded outline grid lines connecting data points without filling surface polygons.
- **C. `ax.draw_mesh3d(X, Y, Z)`:** `draw_mesh3d()` is not a method in Matplotlib.
- **D. `ax.render_surface(X, Y, Z)`:** `render_surface()` does not exist in the Matplotlib API.

</details>

---
## Question 61 — Advanced

**Topic:** 3D plotting  
**Type:** Single Answer

In an `Axes3D` subplot, which method adjusts the 3D camera viewing angle by specifying elevation and azimuth?

- A. `ax.view_init(elev=30, azim=45)`
- B. `ax.set_camera_angle(elev=30, azim=45)`
- C. `ax.rotate_3d(pitch=30, yaw=45)`
- D. `ax.set_viewpoint(altitude=30, longitude=45)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.view_init(elev=30, azim=45)`

### Why this is correct

`ax.view_init(elev=..., azim=..., roll=..., vertical_axis=...)` configures the 3D projection viewing angles in degrees.

### Why the other options are incorrect

- **B. `ax.set_camera_angle(elev=30, azim=45)`:** `set_camera_angle()` does not exist in Matplotlib.
- **C. `ax.rotate_3d(pitch=30, yaw=45)`:** `rotate_3d()` does not exist in Matplotlib.
- **D. `ax.set_viewpoint(altitude=30, longitude=45)`:** `set_viewpoint()` is not a valid method.

</details>

---
## Question 62 — Advanced

**Topic:** Colormaps  
**Type:** Multi Answer — Select all that apply

Which normalization classes from `matplotlib.colors` allow mapping data values to colors across non-linear or centered scales?

- A. `Normalize(vmin, vmax)` for standard linear mapping
- B. `LogNorm(vmin, vmax)` for logarithmic mapping across positive dynamic ranges
- C. `TwoSlopeNorm(vcenter=0, vmin=-10, vmax=50)` for diverging data where the center point is asymmetric
- D. `QuantumNorm(wave_packet)` to automatically compute quantum probability density amplitudes

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `Normalize` provides standard linear interpolation between `vmin` and `vmax`.
- **B:** `LogNorm` maps logarithmic orders of magnitude linearly into the 0-1 color range.
- **C:** `TwoSlopeNorm` (formerly `DivergingNorm`) allows the center transition value (`vcenter`) to be positioned asymmetrically between `vmin` and `vmax`.

### Why the other option is incorrect

- **D:** `QuantumNorm` does not exist in `matplotlib.colors`; custom discrete/binning normalizations use `BoundaryNorm` or `SymLogNorm`.

</details>

---
## Question 63 — Advanced

**Topic:** Colormaps  
**Type:** Single Answer

Which class in `matplotlib.colors` is used to create a custom colormap from a simple discrete list of color hex codes or names?

- A. `matplotlib.colors.ListedColormap(['#ff0000', '#00ff00', '#0000ff'])`
- B. `matplotlib.colors.DiscreteSpectrum(['#ff0000', '#00ff00', '#0000ff'])`
- C. `matplotlib.colors.ColorListMap(['#ff0000', '#00ff00', '#0000ff'])`
- D. `matplotlib.colors.PaletteArray(['#ff0000', '#00ff00', '#0000ff'])`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.colors.ListedColormap(['#ff0000', '#00ff00', '#0000ff'])`

### Why this is correct

`ListedColormap` generates a colormap lookup table directly from a sequence of discrete color specifications.

### Why the other options are incorrect

- **B. `matplotlib.colors.DiscreteSpectrum(['#ff0000', '#00ff00', '#0000ff'])`:** `DiscreteSpectrum` does not exist in Matplotlib.
- **C. `matplotlib.colors.ColorListMap(['#ff0000', '#00ff00', '#0000ff'])`:** `ColorListMap` does not exist in Matplotlib.
- **D. `matplotlib.colors.PaletteArray(['#ff0000', '#00ff00', '#0000ff'])`:** `PaletteArray` is not a Matplotlib class.

</details>

---
## Question 64 — Advanced

**Topic:** Colorbars  
**Type:** Multi Answer — Select all that apply

What is the difference between passing `ax=ax` versus `cax=cax` to `fig.colorbar(mappable, ...)`?

- A. `cax` specifies an exact pre-existing `Axes` into which the colorbar will be drawn directly.
- B. `ax` specifies the parent `Axes` from which space will be automatically stolen to allocate a new colorbar axes.
- C. Passing `ax=[ax1, ax2]` steals space proportionally across multiple subplots to place a shared colorbar.
- D. `cax` requires that `ax` is also simultaneously provided or an error occurs.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** When `cax` is provided, Matplotlib renders the colorbar precisely within that designated `Axes` without resizing any other axes.
- **B:** When `ax` is passed, Matplotlib resizes the specified `Axes` to create an adjacent colorbar axes.
- **C:** Passing a sequence of axes to `ax` creates a unified colorbar spanning across that group of subplots.

### Why the other option is incorrect

- **D:** `cax` and `ax` are mutually exclusive in positioning logic; passing `cax` alone is completely valid and does not require `ax`.

</details>

---
## Question 65 — Advanced

**Topic:** Line plots  
**Type:** Single Answer

Which Matplotlib function is used to plot 2D vector field streamlines (showing flow direction and velocity trajectories smoothly)?

- A. `ax.streamplot(X, Y, U, V)`
- B. `ax.vector_flow(X, Y, U, V)`
- C. `ax.flow_field(X, Y, U, V)`
- D. `ax.trajectory(X, Y, U, V)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `ax.streamplot(X, Y, U, V)`

### Why this is correct

`ax.streamplot(X, Y, U, V)` plots streamlines of a 2D vector field defined on a regular grid by vector components $(U, V)$.

### Why the other options are incorrect

- **B. `ax.vector_flow(X, Y, U, V)`:** `vector_flow()` does not exist in Matplotlib.
- **C. `ax.flow_field(X, Y, U, V)`:** `flow_field()` does not exist in Matplotlib.
- **D. `ax.trajectory(X, Y, U, V)`:** `trajectory()` does not exist in Matplotlib.

</details>

---
## Question 66 — Advanced

**Topic:** Coordinate systems and transforms  
**Type:** Single Answer

In Matplotlib's transformation framework, what coordinate system does `ax.transData` represent?

- A. The coordinate system of the user's data (defined by `xlim` and `ylim`), mapping data points to display pixels
- B. The bounding box of the entire figure window, where $(0, 0)$ is bottom-left of the window and $(1, 1)$ is top-right
- C. Raw physical hardware pixels on the display monitor
- D. Inches from the bottom-left corner of the printed page

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. The coordinate system of the user's data (defined by `xlim` and `ylim`), mapping data points to display pixels

### Why this is correct

`ax.transData` transforms coordinates from data units $(x, y)$ into display/pixel space on the screen, updating dynamically as axis limits change.

### Why the other options are incorrect

- **B. The bounding box of the entire figure window, where $(0, 0)$ is bottom-left of the window and $(1, 1)$ is top-right:** The normalized coordinate system of the figure window is `fig.transFigure`.
- **C. Raw physical hardware pixels on the display monitor:** Raw display pixel coordinates correspond to the identity transform (`None` or `matplotlib.transforms.IdentityTransform()`).
- **D. Inches from the bottom-left corner of the printed page:** Physical inches are handled by figure DPI and physical transformations (`fig.dpi_scale_trans`).

</details>

---
## Question 67 — Advanced

**Topic:** Coordinate systems and transforms  
**Type:** Multi Answer — Select all that apply

Which of the following transform objects are standard built-in coordinate transformations in Matplotlib?

- A. `ax.transAxes` (normalized coordinate system of the Axes: (0,0) bottom-left to (1,1) top-right)
- B. `fig.transFigure` (normalized coordinate system of the Figure: (0,0) bottom-left to (1,1) top-right)
- C. `ax.transData` (data coordinate system)
- D. `ax.transGlobalGPS` (automatic conversion of latitude/longitude to satellite pixels without GIS packages)

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `ax.transAxes` is the 0-to-1 coordinate space of the individual Axes.
- **B:** `fig.transFigure` is the 0-to-1 coordinate space of the entire Figure canvas.
- **C:** `ax.transData` converts data coordinate values to pixel coordinates.

### Why the other option is incorrect

- **D:** `transGlobalGPS` does not exist in standard Matplotlib; geographic mappings require specialized GIS toolkits like Cartopy or GeoPandas.

</details>

---
## Question 68 — Advanced

**Topic:** Colors, markers, and line styles  
**Type:** Single Answer

How can you construct an entirely custom marker shape using vector vertices and drawing codes in Matplotlib?

- A. Pass a `matplotlib.path.Path` object to the `marker` parameter of `ax.plot()` or `ax.scatter()`
- B. Upload an `.obj` 3D mesh to `ax.set_marker_obj()`
- C. Pass a raw string of hexadecimal machine code to `marker`
- D. Custom marker shapes are not supported in Matplotlib

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Pass a `matplotlib.path.Path` object to the `marker` parameter of `ax.plot()` or `ax.scatter()`

### Why this is correct

A custom `Path` (defined with coordinate vertices and `Path.MOVETO`, `LINETO`, `CLOSEPOLY` codes) can be passed directly to the `marker` argument to render bespoke 2D marker glyphs.

### Why the other options are incorrect

- **B. Upload an `.obj` 3D mesh to `ax.set_marker_obj()`:** `set_marker_obj()` does not exist in Matplotlib.
- **C. Pass a raw string of hexadecimal machine code to `marker`:** Markers are defined via symbols, paths, or vertex sequences, not raw machine code.
- **D. Custom marker shapes are not supported in Matplotlib:** Custom markers are fully supported via `matplotlib.path.Path` and tuples of vertices.

</details>

---
## Question 69 — Advanced

**Topic:** Figure, Axes, Axis, and related objects  
**Type:** Single Answer

Which class in `matplotlib.patches` is used to create a 2D rectangular box defined by a lower-left anchor `(x, y)`, `width`, and `height` to add to an `Axes` via `ax.add_patch()`?

- A. `matplotlib.patches.Rectangle((x, y), width, height)`
- B. `matplotlib.patches.Box((x, y), width, height)`
- C. `matplotlib.patches.Square((x, y), width, height)`
- D. `matplotlib.patches.RectArea((x, y), width, height)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.patches.Rectangle((x, y), width, height)`

### Why this is correct

`matplotlib.patches.Rectangle(xy, width, height, angle=0.0, ...)` defines a 2D axis-aligned or rotated rectangle with lower-left anchor `xy`.

### Why the other options are incorrect

- **B. `matplotlib.patches.Box((x, y), width, height)`:** `Box` is not a standard patch class in `matplotlib.patches`.
- **C. `matplotlib.patches.Square((x, y), width, height)`:** `Square` does not exist in `matplotlib.patches` (squares are created using `Rectangle` with equal width and height).
- **D. `matplotlib.patches.RectArea((x, y), width, height)`:** `RectArea` does not exist in Matplotlib.

</details>

---
## Question 70 — Advanced

**Topic:** Styles and themes  
**Type:** Single Answer

How can you temporarily apply a Matplotlib style sheet to a single isolated block of code without modifying the global `rcParams` permanently?

- A. Using the context manager `with plt.style.context('dark_background'):`
- B. Calling `plt.style.isolate('dark_background')`
- C. Calling `plt.push_style('dark_background')` followed by `plt.pop_style()`
- D. Wrapping the code in `try: plt.style.use() finally: plt.style.undo()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Using the context manager `with plt.style.context('dark_background'):`

### Why this is correct

`plt.style.context(...)` is a Python context manager that temporarily applies the specified style dictionary/sheet only within the `with` block and automatically restores previous `rcParams` upon exit.

### Why the other options are incorrect

- **B. Calling `plt.style.isolate('dark_background')`:** `plt.style.isolate()` does not exist.
- **C. Calling `plt.push_style('dark_background')` followed by `plt.pop_style()`:** `plt.push_style()` does not exist in Matplotlib.
- **D. Wrapping the code in `try: plt.style.use() finally: plt.style.undo()`:** `plt.style.undo()` is not a valid method.

</details>

---
## Question 71 — Advanced

**Topic:** Styles and themes  
**Type:** Multi Answer — Select all that apply

Which of the following statements about `matplotlib.rcParams` are correct?

- A. `rcParams` is a dictionary-like container that stores global default configuration values for fonts, colors, lines, and backends.
- B. Modifications like `plt.rcParams['font.size'] = 14` affect all subsequent plots created in that session.
- C. `matplotlib.rcdefaults()` restores all `rcParams` back to Matplotlib's internal default values.
- D. `rcParams` can only be set once during Python installation and cannot be modified at runtime.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `rcParams` (runtime parameters) holds defaults for all visual and backend settings.
- **B:** Mutating `rcParams` modifies global plotting state for all newly instantiated figures/axes.
- **C:** `matplotlib.rcdefaults()` resets all settings to factory defaults.

### Why the other option is incorrect

- **D:** `rcParams` can be modified dynamically at any time in Python code or loaded from files via `plt.rc_file()`.

</details>

---
## Question 72 — Advanced

**Topic:** Date/time plotting  
**Type:** Single Answer

Which locator from `matplotlib.dates` places ticks strictly on the first day of every month?

- A. `matplotlib.dates.MonthLocator()` (or `MonthLocator(bymonthday=1)`)
- B. `matplotlib.dates.DayLocator(interval=30)`
- C. `matplotlib.dates.FirstDayTicker()`
- D. `matplotlib.dates.CalendarStepLocator('monthly')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.dates.MonthLocator()` (or `MonthLocator(bymonthday=1)`)

### Why this is correct

`MonthLocator()` locates calendar month boundaries (by default the first day of each month, accounting for varying month lengths).

### Why the other options are incorrect

- **B. `matplotlib.dates.DayLocator(interval=30)`:** `DayLocator(interval=30)` steps rigidly every 30 days, which drifts out of sync with actual calendar month boundaries.
- **C. `matplotlib.dates.FirstDayTicker()`:** `FirstDayTicker` does not exist in Matplotlib.
- **D. `matplotlib.dates.CalendarStepLocator('monthly')`:** `CalendarStepLocator` does not exist in Matplotlib.

</details>

---
## Question 73 — Advanced

**Topic:** Logarithmic and other axis scales  
**Type:** Multi Answer — Select all that apply

Which of the following axis scale types are natively supported in Matplotlib via `ax.set_yscale(scale_name)`?

- A. `'linear'`
- B. `'log'`
- C. `'symlog'` (symmetrical log, supporting positive and negative values including zero)
- D. `'parabolic'` (to square root all coordinates automatically)

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `'linear'` is the standard linear Cartesian scale.
- **B:** `'log'` is the standard logarithmic scale for positive values.
- **C:** `'symlog'` uses a linear scale near zero (within `linthresh`) and a log scale elsewhere, handling negative values smoothly.

### Why the other option is incorrect

- **D:** `'parabolic'` is not a registered scale in Matplotlib (other standard scales include `'logit'`, `'asinh'`, and `'function'`).

</details>

---
## Question 74 — Advanced

**Topic:** Multiple axes  
**Type:** Single Answer

How is a 'broken axis' (a plot with a broken/interrupted range on the Y-axis to display widely separated data points) typically created in Matplotlib?

- A. By creating two subplots stacked vertically with `sharex=True`, setting different `ylim` on each, hiding adjacent spines, and drawing diagonal break slash markers
- B. By calling `ax.set_ylim_break(start=100, end=500)`
- C. By setting `ax.broken_axis = True`
- D. By setting `ax.set_yscale('interrupted')`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. By creating two subplots stacked vertically with `sharex=True`, setting different `ylim` on each, hiding adjacent spines, and drawing diagonal break slash markers

### Why this is correct

Matplotlib creates broken axes by placing two adjacent subplots (e.g. `ax_top` and `ax_bottom`), adjusting their respective `ylim`, hiding top/bottom border spines, and drawing small diagonal tick marks (slashes) across the spine gap.

### Why the other options are incorrect

- **B. By calling `ax.set_ylim_break(start=100, end=500)`:** `set_ylim_break()` is not a Matplotlib method.
- **C. By setting `ax.broken_axis = True`:** `broken_axis` is not a property of `Axes`.
- **D. By setting `ax.set_yscale('interrupted')`:** `'interrupted'` is not a registered scale in Matplotlib.

</details>

---
## Question 75 — Advanced

**Topic:** Transparency and rendering  
**Type:** Multi Answer — Select all that apply

Which of the following statements about `zorder` and visual layering in Matplotlib are correct?

- A. Artists with a higher `zorder` value are drawn on top of artists with a lower `zorder` value.
- B. By default, grid lines typically have `zorder=0.5` or `zorder=1`, meaning lines and bars (`zorder=2` or `3`) are drawn over the grid.
- C. Setting `ax.set_axisbelow(True)` sets the grid lines and tick marks to be rendered beneath data artists.
- D. `zorder` can only be integers from 1 to 5 and cannot take custom values like 10 or 100.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `zorder` determines the painting order (back-to-front painter's algorithm) in the canvas pipeline.
- **B:** Default standard primitives have predefined z-orders (e.g. Spines/Ticks ~ 1.5-2, Lines ~ 2, Bars ~ 3, Text/Annotations ~ 3-5).
- **C:** `ax.set_axisbelow(True)` forces axis ticks and grid lines behind all data patches and lines.

### Why the other option is incorrect

- **D:** `zorder` accepts any real numeric value (including floats, negative numbers, and high numbers like 100).

</details>

---
## Question 76 — Expert / Advanced

**Topic:** Matplotlib fundamentals and architecture  
**Type:** Single Answer

Matplotlib's internal software architecture is fundamentally structured into three conceptual layers. What are these three layers from lowest to highest level of abstraction?

- A. Backend Layer -> Artist Layer -> Scripting Layer (`pyplot`)
- B. Scripting Layer -> Rendering Engine -> Kernel Layer
- C. OpenGL Shader Layer -> Data Matrix Layer -> Canvas Layer
- D. Vector Layer -> Raster Layer -> GUI Layer

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. Backend Layer -> Artist Layer -> Scripting Layer (`pyplot`)

### Why this is correct

Matplotlib's classic three-layer architecture comprises: (1) **Backend layer** (draws primitives to output devices/canvases), (2) **Artist layer** (represents all visual graph objects and layout hierarchy), and (3) **Scripting layer** (`matplotlib.pyplot`, providing the procedural user interface).

### Why the other options are incorrect

- **B. Scripting Layer -> Rendering Engine -> Kernel Layer:** Matplotlib does not define an internal 'Kernel Layer'.
- **C. OpenGL Shader Layer -> Data Matrix Layer -> Canvas Layer:** Matplotlib's core architecture does not use an OpenGL Shader Layer.
- **D. Vector Layer -> Raster Layer -> GUI Layer:** Vector and Raster are file format categories handled within backends, not the structural architectural layers.

</details>

---
## Question 77 — Expert / Advanced

**Topic:** Figure, Axes, Axis, and related objects  
**Type:** Multi Answer — Select all that apply

In Matplotlib's Artist hierarchy, Artists are categorized into *Primitives* (geometric/text elements) and *Containers* (objects that hold other artists). Which of the following classifications are correct?

- A. `Figure`, `Axes`, and `Axis` are Container Artists.
- B. `Line2D`, `Rectangle`, `Polygon`, and `Text` are Primitive Artists.
- C. `Tick` is a Container Artist that contains tick lines and tick label `Text` instances.
- D. `Figure` is a Primitive Artist that cannot contain other artists.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `Figure` (holds Axes, texts), `Axes` (holds lines, patches, texts, axes), and `Axis` (holds ticks, labels, spines) are Container Artists.
- **B:** `Line2D`, `Rectangle`, `Polygon`, and `Text` represent the actual graphical drawing primitives.
- **C:** `Tick` is a container holding `Line2D` tick marks and `Text` tick label objects.

### Why the other option is incorrect

- **D:** `Figure` is the top-level Container Artist, not a primitive.

</details>

---
## Question 78 — Expert / Advanced

**Topic:** Backends  
**Type:** Single Answer

Which of the following backend types is a non-interactive (headless) raster graphics backend that renders images entirely in memory using the Anti-Grain Geometry C++ library without requiring an active X-server or GUI display?

- A. `Agg`
- B. `TkAgg`
- C. `QtAgg`
- D. `macosx`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `Agg`

### Why this is correct

`Agg` uses the Anti-Grain Geometry engine to render high-quality raster pixels headlessly in memory (used by default on web servers and automated batch pipelines).

### Why the other options are incorrect

- **B. `TkAgg`:** `TkAgg` is an interactive GUI backend embedding the Agg rasterizer inside a Tkinter window.
- **C. `QtAgg`:** `QtAgg` is an interactive GUI backend embedding the Agg rasterizer inside a PyQt/PySide window.
- **D. `macosx`:** `macosx` is an interactive GUI backend specific to macOS Cocoa windows.

</details>

---
## Question 79 — Expert / Advanced

**Topic:** Interactive plotting  
**Type:** Multi Answer — Select all that apply

Which of the following functions manage Matplotlib's interactive mode and GUI event loop processing?

- A. `plt.ion()` enables interactive mode so plots update and draw immediately following plotting commands without blocking.
- B. `plt.ioff()` disables interactive mode, requiring explicit calls to `plt.show()` or `fig.canvas.draw()`.
- C. `plt.pause(interval)` runs the GUI event loop for `interval` seconds, updating figures and processing UI events smoothly.
- D. `time.sleep()` is completely identical to `plt.pause()` and handles GUI window event polling automatically.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `plt.ion()` turns interactive mode ON, triggering auto-redraws after each modification.
- **B:** `plt.ioff()` turns interactive mode OFF for batch script execution.
- **C:** `plt.pause()` flushes drawing queues and processes GUI window callbacks while pausing execution.

### Why the other option is incorrect

- **D:** `time.sleep()` completely freezes the Python thread and does NOT process GUI events, causing windows to become unresponsive / white-screened; `plt.pause()` must be used instead.

</details>

---
## Question 80 — Expert / Advanced

**Topic:** Event handling  
**Type:** Single Answer

How do you connect a custom event handler callback function `on_click` to mouse click events on a figure canvas?

- A. `cid = fig.canvas.mpl_connect('button_press_event', on_click)`
- B. `fig.bind_event('click', on_click)`
- C. `fig.canvas.add_listener('mouse_down', on_click)`
- D. `ax.register_callback('on_click', on_click)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `cid = fig.canvas.mpl_connect('button_press_event', on_click)`

### Why this is correct

`fig.canvas.mpl_connect(event_name, callback)` registers the callback with Matplotlib's event handling system and returns a connection ID (`cid`) for subsequent disconnection via `fig.canvas.mpl_disconnect(cid)`.

### Why the other options are incorrect

- **B. `fig.bind_event('click', on_click)`:** `bind_event()` is not a Matplotlib method.
- **C. `fig.canvas.add_listener('mouse_down', on_click)`:** `add_listener()` does not exist in the Matplotlib FigureCanvas API.
- **D. `ax.register_callback('on_click', on_click)`:** `register_callback()` on Axes is not the event listener registration method.

</details>

---
## Question 81 — Expert / Advanced

**Topic:** Event handling  
**Type:** Multi Answer — Select all that apply

When a callback function is triggered by `mpl_connect('button_press_event', callback)`, which attributes are available on the received `MouseEvent` object `event`?

- A. `event.xdata` and `event.ydata` representing the coordinate position in data space (or `None` if clicked outside an Axes)
- B. `event.x` and `event.y` representing the coordinate position in display pixel space
- C. `event.inaxes` pointing to the specific `Axes` object containing the mouse click (or `None` if outside)
- D. `event.button` indicating which mouse button was clicked (1=left, 2=middle, 3=right)
- E. `event.gpu_temperature` reporting real-time hardware thermals

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C, D

### Why these answers are correct

- **A:** `xdata` and `ydata` map pixel coordinates back into the data coordinate system of the active axes.
- **B:** `x` and `y` give the raw display pixel coordinates relative to the bottom-left of the canvas.
- **C:** `inaxes` provides a direct reference to the clicked `Axes` instance.
- **D:** `button` identifies the mouse button (1=Left, 2=Middle, 3=Right).

### Why the other option is incorrect

- **E:** Hardware telemetry attributes like `gpu_temperature` do not exist on `MouseEvent`.

</details>

---
## Question 82 — Expert / Advanced

**Topic:** Interactive plotting  
**Type:** Single Answer

What is the primary difference between `matplotlib.animation.FuncAnimation` and `matplotlib.animation.ArtistAnimation`?

- A. `FuncAnimation` generates frames dynamically by calling a user-supplied update function on each tick, whereas `ArtistAnimation` replays a pre-computed list of existing artist collections
- B. `FuncAnimation` renders 3D animations only, whereas `ArtistAnimation` renders 2D plots only
- C. `ArtistAnimation` runs on the GPU, whereas `FuncAnimation` runs on the CPU
- D. `FuncAnimation` can only export to MP4, whereas `ArtistAnimation` can only export to HTML

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `FuncAnimation` generates frames dynamically by calling a user-supplied update function on each tick, whereas `ArtistAnimation` replays a pre-computed list of existing artist collections

### Why this is correct

`FuncAnimation` executes a frame generator function iteratively (ideal for dynamic simulations and unbounded streams), while `ArtistAnimation` cycles through a pre-generated sequence of artist frames.

### Why the other options are incorrect

- **B. `FuncAnimation` renders 3D animations only, whereas `ArtistAnimation` renders 2D plots only:** Both classes support both 2D and 3D plotting.
- **C. `ArtistAnimation` runs on the GPU, whereas `FuncAnimation` runs on the CPU:** Both classes execute on the CPU within the Python/Matplotlib rendering pipeline.
- **D. `FuncAnimation` can only export to MP4, whereas `ArtistAnimation` can only export to HTML:** Both animation classes support all configured animation writers (FFMpegWriter, PillowWriter, HTML5/JSAnimation).

</details>

---
## Question 83 — Expert / Advanced

**Topic:** Performance optimization  
**Type:** Single Answer

What is the primary operational mechanism of 'blitting' (`blit=True`) in Matplotlib animations?

- A. It caches the static background of the figure (axes, spines, tick labels) as a pixel bitmap and on each frame only redraws the specific animated artists that changed, copying them directly onto the cached background
- B. It offloads rasterization from the CPU to WebGL inside a browser canvas
- C. It renders only every second frame to cut memory consumption in half
- D. It permanently removes axis labels and spines to reduce draw call complexity

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. It caches the static background of the figure (axes, spines, tick labels) as a pixel bitmap and on each frame only redraws the specific animated artists that changed, copying them directly onto the cached background

### Why this is correct

Blitting captures the static background canvas buffer via `canvas.copy_from_bbox()` and then on each frame restores this background before drawing and blitting only the updated dynamic artists via `canvas.blit()`, avoiding full canvas redraw overhead.

### Why the other options are incorrect

- **B. It offloads rasterization from the CPU to WebGL inside a browser canvas:** Blitting is a 2D memory pixel copy mechanism inside Matplotlib's backend; it does not involve WebGL offloading.
- **C. It renders only every second frame to cut memory consumption in half:** Blitting renders all requested animation frames without dropping frames.
- **D. It permanently removes axis labels and spines to reduce draw call complexity:** Blitting preserves all background decorations without removing them.

</details>

---
## Question 84 — Expert / Advanced

**Topic:** Performance optimization  
**Type:** Single Answer

When visualizing tens of thousands of independent line segments (e.g. in network graphs, financial candlesticks, or particle trails), what is the most performant Matplotlib artist to use instead of calling `ax.plot()` in a loop?

- A. `matplotlib.collections.LineCollection`
- B. Calling `ax.plot()` 50,000 times inside a `for` loop
- C. Repeatedly calling `plt.draw()` after each line point
- D. `matplotlib.patches.IndividualLineLoop`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.collections.LineCollection`

### Why this is correct

`LineCollection` groups thousands of line segments into a single C-level artist/batch draw call, avoiding the massive memory overhead and traversal cost of creating individual `Line2D` artist objects.

### Why the other options are incorrect

- **B. Calling `ax.plot()` 50,000 times inside a `for` loop:** Creating 50,000 individual `Line2D` objects creates severe Python overhead and slows canvas redraws to a crawl.
- **C. Repeatedly calling `plt.draw()` after each line point:** Triggering redraws after each point is extremely inefficient.
- **D. `matplotlib.patches.IndividualLineLoop`:** `IndividualLineLoop` does not exist in Matplotlib.

</details>

---
## Question 85 — Expert / Advanced

**Topic:** Performance optimization  
**Type:** Multi Answer — Select all that apply

Which of the following techniques optimize Matplotlib rendering performance for very large datasets?

- A. Updating existing line data via `line.set_data(x, y)` and `line.set_3d_properties()` instead of re-instantiating `ax.plot()`
- B. Setting `rasterized=True` on complex collections (like scatter plots with 100,000 points) when exporting to vector formats (PDF/SVG)
- C. Enabling path simplification using `matplotlib.rcParams['path.simplify'] = True`
- D. Disabling figure rendering completely with `fig.set_dpi(-100)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Modifying existing `Line2D` vertex buffers via `set_data()` avoids object instantiation and garbage collection overhead.
- **B:** `rasterized=True` rasterizes heavy collections into a bitmap embedded inside vector PDFs/SVGs, preventing bloated multi-gigabyte vector files.
- **C:** `path.simplify` applies the Ramer-Douglas-Peucker algorithm to remove redundant sub-pixel vertices prior to rendering.

### Why the other option is incorrect

- **D:** Negative DPI values are invalid and raise an error.

</details>

---
## Question 86 — Expert / Advanced

**Topic:** Custom projections  
**Type:** Single Answer

To implement a custom projection in Matplotlib (such as an Aitoff map projection or custom hyperbolic geometry), which core base class must be subclassed?

- A. `matplotlib.axes.Axes`
- B. `matplotlib.transforms.BboxBase`
- C. `matplotlib.figure.FigureCanvasBase`
- D. `matplotlib.lines.Line2D`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.axes.Axes`

### Why this is correct

Custom projections are implemented by subclassing `matplotlib.axes.Axes` (or `matplotlib.projections.geo.GeoAxes`), overriding its transformation pipeline (`_init_axis()`, `_set_lim_and_transforms()`), and defining the projection `name` attribute.

### Why the other options are incorrect

- **B. `matplotlib.transforms.BboxBase`:** `BboxBase` represents bounding boxes, not coordinate projection systems.
- **C. `matplotlib.figure.FigureCanvasBase`:** `FigureCanvasBase` manages canvas device rendering rather than coordinate projections.
- **D. `matplotlib.lines.Line2D`:** `Line2D` is a 1D line primitive artist.

</details>

---
## Question 87 — Expert / Advanced

**Topic:** Custom projections  
**Type:** Multi Answer — Select all that apply

Which steps are involved in creating and registering a custom projection class `MyCustomAxes` in Matplotlib?

- A. Define a unique `name` class attribute (e.g. `name = 'my_custom_proj'`) inside the subclass.
- B. Register the subclass using `matplotlib.projections.register_projection(MyCustomAxes)`.
- C. Instantiate subplots using `plt.subplot(projection='my_custom_proj')` or `plt.subplots(subplot_kw={'projection': 'my_custom_proj'})`.
- D. Recompile the Python C-extension binaries from source.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** The `name` attribute provides the string lookup key for the projection registry.
- **B:** `register_projection()` registers the class in Matplotlib's global projection dictionary.
- **C:** Once registered, the custom projection string can be passed directly to `projection=...` in figure/subplot factory functions.

### Why the other option is incorrect

- **D:** Custom projections are purely pure-Python/NumPy transformations and do not require recompiling C-extensions.

</details>

---
## Question 88 — Expert / Advanced

**Topic:** Coordinate systems and transforms  
**Type:** Single Answer

How do you convert display pixel coordinates `(px, py)` back into data coordinates `(x, y)` on an `Axes` object `ax`?

- A. `data_x, data_y = ax.transData.inverted().transform((px, py))`
- B. `data_x, data_y = ax.transData.reverse((px, py))`
- C. `data_x, data_y = ax.transData.unproject((px, py))`
- D. `data_x, data_y = ax.pixel_to_data((px, py))`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `data_x, data_y = ax.transData.inverted().transform((px, py))`

### Why this is correct

Transform objects in Matplotlib implement `.inverted()`, which returns the inverse transform node. Calling `.transform()` on `ax.transData.inverted()` maps display pixels back to data units.

### Why the other options are incorrect

- **B. `data_x, data_y = ax.transData.reverse((px, py))`:** `reverse()` is not a method on Transform objects.
- **C. `data_x, data_y = ax.transData.unproject((px, py))`:** `unproject()` is not a standard Matplotlib method.
- **D. `data_x, data_y = ax.pixel_to_data((px, py))`:** `pixel_to_data()` is not a method on `Axes`.

</details>

---
## Question 89 — Expert / Advanced

**Topic:** Coordinate systems and transforms  
**Type:** Multi Answer — Select all that apply

What is `matplotlib.transforms.blended_transform_factory(x_transform, y_transform)` used for?

- A. Creating a composite 2D transform where the X-coordinate uses one transformation and the Y-coordinate uses a different transformation.
- B. Enabling plots where the X-axis is in data units (`ax.transData`) but the Y-axis spans normalized axes units (`ax.transAxes`) from 0 to 1 (such as in `ax.axvline()`).
- C. Blending transparency gradients between two overlapping images.
- D. Constructing specialized span and shading transformations.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, D

### Why these answers are correct

- **A:** `blended_transform_factory` combines separate 1D transforms along the X and Y axes into a unified 2D `Transform`.
- **B:** This allows spanning lines like `axvline()` and `axvspan()` to lock X to data values while Y spans the full axes height $(0 \text{ to } 1)$.
- **D:** It forms the foundation of all hybrid span, bar, and annotation transformations in Matplotlib.

### Why the other option is incorrect

- **C:** Blended transforms operate purely on geometric coordinates, not image pixel color/alpha blending.

</details>

---
## Question 90 — Expert / Advanced

**Topic:** Figure and Axes management  
**Type:** Single Answer

In a long-running batch script or web server (e.g. FastAPI/Flask) that generates hundreds of figures in a loop, why does memory consumption continually increase if figures are not explicitly closed?

- A. `pyplot` maintains internal global references to every created `Figure` in the `_pylab_helpers.Gcf` manager, preventing Python's garbage collector from reclaiming memory until `plt.close(fig)` is called.
- B. NumPy arrays retain GPU memory locks permanently.
- C. The operating system refuses to release file handles after `plt.savefig()`.
- D. Matplotlib saves backup copies of all figures to a hidden disk partition.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `pyplot` maintains internal global references to every created `Figure` in the `_pylab_helpers.Gcf` manager, preventing Python's garbage collector from reclaiming memory until `plt.close(fig)` is called.

### Why this is correct

`pyplot` acts as a state manager holding strong references to all open figures inside `Gcf.figs`. Without calling `plt.close(fig)` or `plt.close('all')`, figures remain referenced in memory, causing memory leaks.

### Why the other options are incorrect

- **B. NumPy arrays retain GPU memory locks permanently.:** Standard Matplotlib rendering is CPU-bound and does not lock GPU memory.
- **C. The operating system refuses to release file handles after `plt.savefig()`.:** File handles are closed automatically after `savefig()`; the memory leak occurs in RAM inside the Python process.
- **D. Matplotlib saves backup copies of all figures to a hidden disk partition.:** Matplotlib does not write hidden copies to disk.

</details>

---
## Question 91 — Expert / Advanced

**Topic:** Figure and Axes management  
**Type:** Multi Answer — Select all that apply

What are the distinctions between `plt.close()`, `fig.clf()`, and `ax.cla()`?

- A. `ax.cla()` (or `ax.clear()`) clears all artists, lines, and text from a specific `Axes`, preserving the `Axes` object itself.
- B. `fig.clf()` (or `fig.clear()`) clears all subplots and artists from the `Figure`, keeping the empty `Figure` window intact.
- C. `plt.close(fig)` destroys the figure window, removes it from the `pyplot` manager, and releases its GUI/canvas resources.
- D. `ax.cla()` deletes the entire Python virtual environment.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `cla()` stands for 'Clear Axes' and resets the contents of a single subplot.
- **B:** `clf()` stands for 'Clear Figure' and strips all axes and decorations from the figure.
- **C:** `plt.close()` completely closes and dereferences the figure.

### Why the other option is incorrect

- **D:** `ax.cla()` is a plot-clearing method and has no effect on Python environments.

</details>

---
## Question 92 — Expert / Advanced

**Topic:** Backends  
**Type:** Single Answer

What is the key distinction between `fig.canvas.draw()` and `fig.canvas.draw_idle()`?

- A. `draw()` forces an immediate, synchronous redraw of the entire canvas, whereas `draw_idle()` requests a redraw as soon as the GUI control returns to the main event loop (coalescing multiple requests into a single paint event)
- B. `draw_idle()` puts the CPU into low-power sleep mode
- C. `draw()` renders in 2D while `draw_idle()` renders in 3D
- D. `draw_idle()` renders only when the user is not touching the keyboard or mouse for 10 seconds

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `draw()` forces an immediate, synchronous redraw of the entire canvas, whereas `draw_idle()` requests a redraw as soon as the GUI control returns to the main event loop (coalescing multiple requests into a single paint event)

### Why this is correct

`draw_idle()` batches and debounces paint requests, preventing redundant rendering when multiple artists or limits are updated in rapid succession before the GUI redraws.

### Why the other options are incorrect

- **B. `draw_idle()` puts the CPU into low-power sleep mode:** `draw_idle()` does not manage hardware power states.
- **C. `draw()` renders in 2D while `draw_idle()` renders in 3D:** Both methods support identical rendering capabilities.
- **D. `draw_idle()` renders only when the user is not touching the keyboard or mouse for 10 seconds:** `draw_idle()` fires at the next iteration of the event loop, not after user inactivity.

</details>

---
## Question 93 — Expert / Advanced

**Topic:** Backends  
**Type:** Multi Answer — Select all that apply

When using Matplotlib in a multi-threaded web server (such as Django, Flask, or FastAPI workers), which best practices should be followed to prevent race conditions and memory leaks?

- A. Set the non-interactive backend `matplotlib.use('Agg')` before importing `pyplot`.
- B. Avoid relying on global `pyplot` state functions (e.g. `plt.plot()`, `plt.gca()`) and instead instantiate figures explicitly via `Figure()` and `FigureCanvasAgg` or ensure thread isolation.
- C. Always explicitly call `plt.close(fig)` or destroy figure instances inside a `finally:` block.
- D. Share a single global `Figure` object across all concurrent web requests.

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** Configuring the headless `'Agg'` backend prevents attempts to initialize GUI toolkits in non-GUI worker threads.
- **B:** Global `pyplot` state is not thread-safe; concurrent threads mutating `plt.gca()` can cross-contaminate plots.
- **C:** Ensuring `close()` inside `finally:` guarantees memory reclamation even if exceptions occur during request handling.

### Why the other option is incorrect

- **D:** Sharing a single Figure across concurrent requests will cause catastrophic race conditions and corrupted plot outputs.

</details>

---
## Question 94 — Expert / Advanced

**Topic:** Coordinate systems and transforms  
**Type:** Single Answer

Which helper in `matplotlib.transforms` creates a transformation that adds a fixed physical offset in display points (e.g. 5 points to the right and 5 points up) to an existing coordinate transformation?

- A. `matplotlib.transforms.offset_copy(trans, fig=fig, x=5, y=5, units='points')`
- B. `matplotlib.transforms.add_points(trans, 5, 5)`
- C. `matplotlib.transforms.shift_pixels(trans, 5, 5)`
- D. `matplotlib.transforms.displace_vector(trans, 5, 5)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.transforms.offset_copy(trans, fig=fig, x=5, y=5, units='points')`

### Why this is correct

`offset_copy()` produces a new transformation that takes coordinates, maps them through the base transform, and applies a physical offset specified in `'points'`, `'inches'`, or `'dots'`.

### Why the other options are incorrect

- **B. `matplotlib.transforms.add_points(trans, 5, 5)`:** `add_points()` is not a Matplotlib method.
- **C. `matplotlib.transforms.shift_pixels(trans, 5, 5)`:** `shift_pixels()` does not exist in Matplotlib.
- **D. `matplotlib.transforms.displace_vector(trans, 5, 5)`:** `displace_vector()` does not exist in Matplotlib.

</details>

---
## Question 95 — Expert / Advanced

**Topic:** Tick formatting  
**Type:** Multi Answer — Select all that apply

Which of the following built-in ticker classes in `matplotlib.ticker` handle specialized scientific and engineering notation?

- A. `ScalarFormatter(useMathText=True)` to render scientific exponent multipliers (e.g. $\times 10^5$) in mathematical LaTeX-like formatting
- B. `EngFormatter(unit='V')` to format values with engineering SI prefixes (e.g., $1.5\text{ kV}$, $200\text{ mV}$)
- C. `LogFormatterMathtext` to format logarithmic tick labels cleanly as powers of the base (e.g. $10^{-3}, 10^0, 10^3$)
- D. `BinaryQuantumFormatter` to format numbers into quantum superposition states

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `ScalarFormatter(useMathText=True)` generates clean mathematical typography for scale offsets.
- **B:** `EngFormatter` automatically scales values to powers of 1000 and appends SI prefixes ($k, M, G, m, \mu, n$, etc.).
- **C:** `LogFormatterMathtext` displays clean base-power superscripts on log-scale axes.

### Why the other option is incorrect

- **D:** `BinaryQuantumFormatter` does not exist in Matplotlib.

</details>

---
## Question 96 — Expert / Advanced

**Topic:** Advanced customization and best practices  
**Type:** Single Answer

Which Matplotlib module is used to add artistic drop shadows, glow effects, or contrasting stroke outlines around text and path artists?

- A. `matplotlib.patheffects` (using classes like `withStroke` and `SimplePatchShadow`)
- B. `matplotlib.photoshop`
- C. `matplotlib.filters.glow`
- D. `matplotlib.fx.shadows`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `matplotlib.patheffects` (using classes like `withStroke` and `SimplePatchShadow`)

### Why this is correct

`matplotlib.patheffects` contains path effect classes (such as `withStroke`, `Stroke`, `Normal`, and `SimplePatchShadow`) that apply stylized outline and shadow drawing passes to artists via `artist.set_path_effects([...])`.

### Why the other options are incorrect

- **B. `matplotlib.photoshop`:** `matplotlib.photoshop` does not exist.
- **C. `matplotlib.filters.glow`:** `matplotlib.filters` does not exist.
- **D. `matplotlib.fx.shadows`:** `matplotlib.fx` does not exist.

</details>

---
## Question 97 — Expert / Advanced

**Topic:** Transparency and rendering  
**Type:** Single Answer

Which method is used to clip an artist so that it is rendered strictly inside the arbitrary 2D geometric boundary of another `Patch` (such as clipping an image inside a circle or polygon)?

- A. `artist.set_clip_path(patch)`
- B. `artist.set_stencil_mask(patch)`
- C. `artist.mask_to_geometry(patch)`
- D. `artist.set_clip_polygon_only(patch)`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `artist.set_clip_path(patch)`

### Why this is correct

`artist.set_clip_path(patch)` (or `artist.set_clip_path(path, transform)`) constrains rendering of the artist to the interior region of the specified geometric patch/path.

### Why the other options are incorrect

- **B. `artist.set_stencil_mask(patch)`:** `set_stencil_mask()` does not exist in Matplotlib.
- **C. `artist.mask_to_geometry(patch)`:** `mask_to_geometry()` is not a valid method.
- **D. `artist.set_clip_polygon_only(patch)`:** `set_clip_polygon_only()` does not exist in Matplotlib.

</details>

---
## Question 98 — Expert / Advanced

**Topic:** Layout management  
**Type:** Single Answer

To programmatically calculate the exact display bounding box (in pixel coordinates) of a rendered `Text` or `Axes` artist during a script, which method is called?

- A. `artist.get_window_extent(renderer=fig.canvas.get_renderer())`
- B. `artist.calculate_pixel_bounds()`
- C. `artist.get_screen_dimensions()`
- D. `artist.to_display_rect()`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. `artist.get_window_extent(renderer=fig.canvas.get_renderer())`

### Why this is correct

`artist.get_window_extent(renderer)` evaluates the exact `Bbox` of the artist in display pixel units using the canvas renderer.

### Why the other options are incorrect

- **B. `artist.calculate_pixel_bounds()`:** `calculate_pixel_bounds()` does not exist in Matplotlib.
- **C. `artist.get_screen_dimensions()`:** `get_screen_dimensions()` is not a valid method.
- **D. `artist.to_display_rect()`:** `to_display_rect()` does not exist in Matplotlib.

</details>

---
## Question 99 — Expert / Advanced

**Topic:** Styles and themes  
**Type:** Multi Answer — Select all that apply

When Matplotlib starts up, which of the following file locations and environment variables are checked in the search hierarchy to populate `rcParams`?

- A. Environment variable `MATPLOTLIBRC` (if pointing to a valid file path)
- B. `matplotlibrc` located in the current working directory
- C. `$MATPLOTLIBDATA/matplotlibrc` (the system/install-level default configuration file)
- D. System registry keys stored under `HKEY_LOCAL_MACHINE\SOFTWARE\Matplotlib`

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answers:** A, B, C

### Why these answers are correct

- **A:** `MATPLOTLIBRC` has the highest file override precedence if set to a valid file path.
- **B:** A local `matplotlibrc` in the current working directory overrides user and system defaults.
- **C:** The installed system library `matplotlibrc` serves as the baseline fallback.

### Why the other option is incorrect

- **D:** Matplotlib does not read configuration from the Windows OS registry; configuration is handled strictly via text `.matplotlibrc` files, `MATPLOTLIBRC`, or user config directories (`~/.config/matplotlib/matplotlibrc`).

</details>

---
## Question 100 — Expert / Advanced

**Topic:** Advanced customization and best practices  
**Type:** Single Answer

Why is the explicit Object-Oriented (OO) interface (`fig, ax = plt.subplots()`; `ax.plot()`) strongly favored over the procedural state-machine interface (`plt.plot()`) in production software and large data pipelines?

- A. The OO interface provides explicit, thread-safe references to distinct `Figure` and `Axes` instances, avoids hidden global state bugs, enables fine-grained control over subplots, and prevents unintended side-effects across functions
- B. The OO interface compiles Python code into native GPU assembly instructions
- C. The procedural `pyplot` interface was deprecated and removed in Python 3
- D. The OO interface only works when an internet connection is active

<details>
<summary>Show Answer & Explanation</summary>

**Correct Answer:** A. The OO interface provides explicit, thread-safe references to distinct `Figure` and `Axes` instances, avoids hidden global state bugs, enables fine-grained control over subplots, and prevents unintended side-effects across functions

### Why this is correct

The Object-Oriented API explicitly encapsulates state within object instances (`Figure`, `Axes`), making functions modular, testable, reusable, and free from hidden global state side-effects.

### Why the other options are incorrect

- **B. The OO interface compiles Python code into native GPU assembly instructions:** Matplotlib's OO interface executes within standard Python and does not compile into GPU assembly.
- **C. The procedural `pyplot` interface was deprecated and removed in Python 3:** `pyplot` is not deprecated; it remains active for interactive convenience and quick prototyping.
- **D. The OO interface only works when an internet connection is active:** Matplotlib executes entirely locally and does not require internet connectivity.

</details>

---
## Quiz Completion & Next Milestones

Congratulations on completing the **100-Question Matplotlib Mastery Quiz**!

### Scoring Guide
- **90–100%:** **Master / Architect Level** — Exceptional mastery of Matplotlib's object-oriented architecture, custom projection pipeline, and performance optimization.
- **75–89%:** **Senior Visualizer** — Strong understanding of complex multi-axes layouts, custom transformations, and color normalization.
- **60–74%:** **Competent Practitioner** — Solid grasp of everyday data visualization workflows, subplots, and styling.
- **Below 60%:** **Foundational Review Needed** — Review the official Matplotlib documentation, specifically the *Artist Tutorial*, *Transformations Tutorial*, and *Lifecycle of a Plot*.

---
