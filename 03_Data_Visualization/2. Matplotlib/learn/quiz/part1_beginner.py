# Questions 1-25: Beginner Level

questions_part1 = [
    {
        "num": 1,
        "difficulty": "Beginner",
        "topic": "Matplotlib Fundamentals and Architecture",
        "type": "single",
        "question": "Which top-level Matplotlib object acts as the overall canvas and container for all plot elements, including one or more plotting areas (Axes)?",
        "options": {
            "A": "Figure",
            "B": "Axes",
            "C": "Axis",
            "D": "Artist"
        },
        "correct": "A",
        "correct_explanation": "A `Figure` is the top-level container in Matplotlib that holds all plot elements, including one or more `Axes` objects, the figure canvas, titles, legends, and colorbars.",
        "incorrect_explanations": {
            "B": "`Axes` represents an individual plotting area (or subplot) where data is plotted, contained inside a `Figure`.",
            "C": "`Axis` represents a single coordinate axis (such as the X-axis or Y-axis) containing ticks, tick labels, and axis limits within an `Axes`.",
            "D": "`Artist` is the abstract base class for virtually all visible objects in Matplotlib (including `Figure`, `Axes`, `Line2D`, and `Text`), rather than the top-level window/container itself."
        }
    },
    {
        "num": 2,
        "difficulty": "Beginner",
        "topic": "pyplot",
        "type": "multi",
        "question": "Which of the following statements regarding the `matplotlib.pyplot` module are correct?",
        "options": {
            "A": "It provides a state-machine interface that mirrors the plotting workflow of MATLAB.",
            "B": "Functions like `plt.plot()` automatically create and track the current Figure and current Axes if none exist.",
            "C": "It is completely separate from Matplotlib and cannot be used alongside object-oriented Axes methods.",
            "D": "It provides convenient top-level functions such as `plt.show()`, `plt.savefig()`, and `plt.clf()`."
        },
        "correct": ["A", "B", "D"],
        "correct_explanations": {
            "A": "`pyplot` provides a state-based interface designed to closely emulate the procedural plotting experience of MATLAB.",
            "B": "In `pyplot`'s state machine, calling functions like `plt.plot()` automatically initializes `plt.gcf()` (current figure) and `plt.gca()` (current axes) if they do not yet exist.",
            "D": "`pyplot` includes global figure management utilities such as displaying (`plt.show()`), saving (`plt.savefig()`), and clearing (`plt.clf()`)."
        },
        "incorrect_explanations": {
            "C": "`pyplot` is an integral module of Matplotlib and can seamlessly coexist with the object-oriented API (e.g., using `plt.subplots()` to obtain `fig` and `ax`)."
        }
    },
    {
        "num": 3,
        "difficulty": "Beginner",
        "topic": "Figures and Axes",
        "type": "single",
        "question": "Consider the following code snippet:\n\n```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\n```\n\nWhat are the types of the returned objects `fig` and `ax`?",
        "options": {
            "A": "`fig` is a `matplotlib.figure.Figure` and `ax` is a `matplotlib.axes.Axes`",
            "B": "`fig` is a `matplotlib.axes.Axes` and `ax` is a `matplotlib.figure.Figure`",
            "C": "`fig` is a `matplotlib.figure.Figure` and `ax` is a `numpy.ndarray` of subplots",
            "D": "`fig` is a string identifier and `ax` is a canvas pointer"
        },
        "correct": "A",
        "correct_explanation": "When called without row or column arguments (defaulting to 1x1), `plt.subplots()` creates and returns a single `Figure` instance and a single `Axes` instance.",
        "incorrect_explanations": {
            "B": "The return order is always `(Figure, Axes)`, not `(Axes, Figure)`.",
            "C": "`ax` is returned as a `numpy.ndarray` only when multiple subplots are requested (e.g., `nrows > 1` or `ncols > 1`), not for the default single subplot.",
            "D": "Matplotlib returns rich Python object instances (`Figure` and `Axes`), not primitive string identifiers or pointers."
        }
    },
    {
        "num": 4,
        "difficulty": "Beginner",
        "topic": "Creating and displaying plots",
        "type": "single",
        "question": "What is the primary function of `plt.show()` in a standard interactive Python script?",
        "options": {
            "A": "It converts the active figure into a static PNG file on disk.",
            "B": "It opens the GUI window to display all open figures and starts the GUI backend event loop.",
            "C": "It resets all global `rcParams` styling parameters to default values.",
            "D": "It prints the numerical coordinates of all plot artists to stdout."
        },
        "correct": "B",
        "correct_explanation": "`plt.show()` displays all currently active figure windows and blocks execution (unless interactive mode is enabled) by running the backend GUI toolkit's event loop until windows are closed.",
        "incorrect_explanations": {
            "A": "Exporting a figure to a static file on disk is handled by `plt.savefig()`, not `plt.show()`.",
            "C": "Resetting `rcParams` is achieved using `matplotlib.rcdefaults()`, not `plt.show()`.",
            "D": "`plt.show()` renders visual graphics in a backend canvas/window, rather than printing coordinate arrays to stdout."
        }
    },
    {
        "num": 5,
        "difficulty": "Beginner",
        "topic": "Line plots",
        "type": "single",
        "question": "What will be plotted along the X-axis if you pass only a single 1D list to `plt.plot()` as shown below?\n\n```python\nimport matplotlib.pyplot as plt\n\nplt.plot([10, 20, 15, 30])\n```",
        "options": {
            "A": "Default values `[1, 2, 3, 4]`",
            "B": "Default index values `[0, 1, 2, 3]`",
            "C": "The sorted values `[10, 15, 20, 30]`",
            "D": "It raises a `TypeError` because both X and Y arrays are strictly required"
        },
        "correct": "B",
        "correct_explanation": "If only a single 1D sequence is passed to `plt.plot()`, Matplotlib treats it as the Y-values and automatically generates zero-indexed sequential integers `[0, 1, 2, ..., N-1]` for the X-axis.",
        "incorrect_explanations": {
            "A": "Matplotlib uses standard 0-based indexing (`[0, 1, 2, 3]`), not 1-based indexing (`[1, 2, 3, 4]`).",
            "C": "Matplotlib preserves the exact sequential order given and generates integer index coordinates; it does not sort the data.",
            "D": "`plt.plot()` accepts a single positional argument for Y-values without raising an error."
        }
    },
    {
        "num": 6,
        "difficulty": "Beginner",
        "topic": "Colors, markers, and line styles",
        "type": "multi",
        "question": "Which of the following format strings (fmt) are valid in `plt.plot(x, y, fmt)` to specify line style, color, and marker simultaneously?",
        "options": {
            "A": "'ro--' (red color, circle marker, dashed line)",
            "B": "'b^:' (blue color, triangle_up marker, dotted line)",
            "C": "'g-' (green color, solid line, no marker)",
            "D": "'line=dashed,color=red,marker=square'"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "'ro--' is a valid shorthand format string combining 'r' (red), 'o' (circle marker), and '--' (dashed line).",
            "B": "'b^:' is a valid shorthand format string combining 'b' (blue), '^' (triangle_up marker), and ':' (dotted line).",
            "C": "'g-' is a valid shorthand format string specifying 'g' (green) and '-' (solid line) with no marker."
        },
        "incorrect_explanations": {
            "D": "Format strings use compact token codes (e.g. `'rs--'`); key-value syntax like `'line=dashed...'` is invalid in the positional `fmt` argument (they must be passed as keyword arguments `linestyle='--', color='r', marker='s'`)."
        }
    },
    {
        "num": 7,
        "difficulty": "Beginner",
        "topic": "Scatter plots",
        "type": "single",
        "question": "Which Matplotlib function or method is specifically designed to create a scatter plot where individual points can have variable sizes and variable colors mapped to a colormap?",
        "options": {
            "A": "`plt.scatter()` / `ax.scatter()`",
            "B": "`plt.point()` / `ax.point()`",
            "C": "`plt.stem()` / `ax.stem()`",
            "D": "`plt.step()` / `ax.step()`"
        },
        "correct": "A",
        "correct_explanation": "`scatter()` creates a `PathCollection` where each point can take independent sizes via the `s` parameter and colors mapped via the `c` parameter with a colormap (`cmap`).",
        "incorrect_explanations": {
            "B": "There is no `plt.point()` function in Matplotlib.",
            "C": "`plt.stem()` creates a stem-and-leaf / impulse plot with vertical stems connecting data points to a baseline.",
            "D": "`plt.step()` creates a step plot (piecewise-constant line plot)."
        }
    },
    {
        "num": 8,
        "difficulty": "Beginner",
        "topic": "Scatter plots",
        "type": "multi",
        "question": "Which parameters of `ax.scatter(x, y, ...)` control the transparency and marker area size of data points?",
        "options": {
            "A": "`alpha` controls marker transparency (from 0.0 to 1.0)",
            "B": "`s` controls marker size (specified in points squared)",
            "C": "`transparency` controls marker opacity (from 0 to 100)",
            "D": "`size` controls marker radius in centimeters"
        },
        "correct": ["A", "B"],
        "correct_explanations": {
            "A": "`alpha` is a float between 0.0 (completely transparent) and 1.0 (completely opaque) that sets artist opacity.",
            "B": "`s` specifies marker size as the area of the marker in points squared (pt^2)."
        },
        "incorrect_explanations": {
            "C": "Matplotlib uses `alpha`, not a parameter named `transparency`.",
            "D": "The marker size parameter is `s` (in points squared), not `size` in centimeters."
        }
    },
    {
        "num": 9,
        "difficulty": "Beginner",
        "topic": "Bar charts",
        "type": "single",
        "question": "Which method is used to create a horizontal bar chart in Matplotlib?",
        "options": {
            "A": "`ax.barh()`",
            "B": "`ax.hbar()`",
            "C": "`ax.bar(horizontal=True)`",
            "D": "`ax.bar_horizontal()`"
        },
        "correct": "A",
        "correct_explanation": "`ax.barh()` (or `plt.barh()`) is the dedicated Matplotlib function for drawing horizontal bar charts, accepting parameters for Y-coordinates (`y`) and horizontal lengths (`width`).",
        "incorrect_explanations": {
            "B": "The method is named `barh()`, not `hbar()`.",
            "C": "`ax.bar()` does not take a `horizontal` keyword argument; `barh()` is used instead.",
            "D": "`bar_horizontal()` does not exist in the Matplotlib API."
        }
    },
    {
        "num": 10,
        "difficulty": "Beginner",
        "topic": "Bar charts",
        "type": "single",
        "question": "Which parameter in `ax.bar(x, height, ...)` is used to adjust the thickness/width of the vertical bars along the X-axis?",
        "options": {
            "A": "`width`",
            "B": "`thickness`",
            "C": "`bar_size`",
            "D": "`span`"
        },
        "correct": "A",
        "correct_explanation": "The `width` parameter in `ax.bar()` controls the scalar or array-like width of the vertical bars along the X-axis (defaulting to 0.8).",
        "incorrect_explanations": {
            "B": "`thickness` is not a valid parameter of `ax.bar()`.",
            "C": "`bar_size` does not exist in the Matplotlib API.",
            "D": "`span` is not a parameter for bar width in `ax.bar()`."
        }
    },
    {
        "num": 11,
        "difficulty": "Beginner",
        "topic": "Histograms",
        "type": "single",
        "question": "In `ax.hist(x, bins=...)`, what does passing an integer `bins=20` do?",
        "options": {
            "A": "Divides the range of data into 20 equal-width bins between the minimum and maximum values",
            "B": "Selects the first 20 data points from `x` to construct the histogram",
            "C": "Plots only bars that have a frequency count greater than or equal to 20",
            "D": "Sets the maximum height of the Y-axis to 20"
        },
        "correct": "A",
        "correct_explanation": "When `bins` is an integer N, Matplotlib automatically calculates N equal-width bin intervals spanning from `min(x)` to `max(x)`.",
        "incorrect_explanations": {
            "B": "All valid data points in `x` are included; `bins` defines the number of intervals, not the sample size.",
            "C": "Bin count does not filter data by frequency threshold.",
            "D": "Y-axis limits are computed automatically based on the resulting bin counts, not set directly by `bins`."
        }
    },
    {
        "num": 12,
        "difficulty": "Beginner",
        "topic": "Histograms",
        "type": "multi",
        "question": "Which of the following statements about `ax.hist()` are true?",
        "options": {
            "A": "Setting `density=True` normalizes the bin counts so that the total integral/area under the histogram equals 1.",
            "B": "Setting `cumulative=True` computes a cumulative histogram.",
            "C": "Passing a sequence of numbers (e.g., `bins=[0, 5, 10, 20]`) defines custom, non-uniform bin edges.",
            "D": "`ax.hist()` only accepts 2D image matrices and cannot process 1D arrays."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "With `density=True`, the histogram values are normalized so that the sum of each bin's count multiplied by its bin width equals 1 (probability density).",
            "B": "`cumulative=True` calculates the cumulative distribution where each bin gives the count of data points in that bin plus all previous bins.",
            "C": "Providing a sequence for `bins` specifies explicit boundary edges for each bin."
        },
        "incorrect_explanations": {
            "D": "`ax.hist()` is specifically designed for 1D arrays or sequences of 1D arrays."
        }
    },
    {
        "num": 13,
        "difficulty": "Beginner",
        "topic": "Pie charts",
        "type": "single",
        "question": "In `ax.pie(values, autopct=...)`, what is the purpose of the `autopct` parameter?",
        "options": {
            "A": "To format and display the percentage value label inside each slice",
            "B": "To automatically adjust the opacity of slices based on their area",
            "C": "To automatically compute and apply a continuous color spectrum",
            "D": "To automatically sort slices in descending order"
        },
        "correct": "A",
        "correct_explanation": "`autopct` accepts a format string (like `'%1.1f%%'`) or a callable function to calculate, format, and display the numerical percentage value directly on each pie slice.",
        "incorrect_explanations": {
            "B": "Slice opacity is controlled via `wedgeprops` or alpha, not `autopct`.",
            "C": "Colors are assigned via the `colors` parameter, not `autopct`.",
            "D": "Matplotlib preserves the input order of slices starting from `startangle`; it does not auto-sort unless done manually before plotting."
        }
    },
    {
        "num": 14,
        "difficulty": "Beginner",
        "topic": "Pie charts",
        "type": "multi",
        "question": "Which of the following parameters are valid in `ax.pie()` to customize slice positioning and orientation?",
        "options": {
            "A": "`explode` to offset/separate specified slices from the center",
            "B": "`startangle` to rotate the starting slice angle counter-clockwise from the X-axis",
            "C": "`shadow=True` to draw a drop shadow beneath the pie chart",
            "D": "`pivot` to convert the pie chart into a 3D hemisphere"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`explode` takes a sequence of fractional radial offsets (e.g. `[0, 0.1, 0]`) to pull selected slices outward.",
            "B": "`startangle` rotates the starting position of the first wedge counter-clockwise by a specified number of degrees from the positive X-axis.",
            "C": "`shadow=True` renders a subtle shadow effect under the pie wedges."
        },
        "incorrect_explanations": {
            "D": "`pivot` is not a parameter of `ax.pie()`; 2D pie charts cannot be converted into 3D hemispheres via a `pivot` argument."
        }
    },
    {
        "num": 15,
        "difficulty": "Beginner",
        "topic": "Box plots",
        "type": "single",
        "question": "In a standard Matplotlib box plot created with `ax.boxplot(data)`, what does the central line inside the box represent by default?",
        "options": {
            "A": "The median (50th percentile)",
            "B": "The arithmetic mean",
            "C": "The standard deviation",
            "D": "The mode"
        },
        "correct": "A",
        "correct_explanation": "By default, the solid line dividing the interquartile range (IQR) box represents the sample median ($Q_2$).",
        "incorrect_explanations": {
            "B": "The mean is not shown by default; it can be displayed as a distinct marker/line by passing `showmeans=True`.",
            "C": "Box plots display quartile-based summary statistics ($Q_1, Q_2, Q_3$, and IQR whiskers), not standard deviations.",
            "D": "The mode is not depicted in standard Tukey box plots."
        }
    },
    {
        "num": 16,
        "difficulty": "Beginner",
        "topic": "Box plots",
        "type": "multi",
        "question": "Which of the following elements can be customized or toggled in `ax.boxplot()`?",
        "options": {
            "A": "`showmeans=True` to render the arithmetic mean",
            "B": "`notch=True` to render a notched box representing the median confidence interval",
            "C": "`vert=False` to produce horizontal box plots",
            "D": "`kernel='gaussian'` to overlay a kernel density estimate curve automatically"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`showmeans=True` plots the mean value using a marker or dotted line.",
            "B": "`notch=True` produces a notched shape around the median representing the 95% confidence interval of the median.",
            "C": "`vert=False` orientations the box plots horizontally instead of vertically."
        },
        "incorrect_explanations": {
            "D": "`ax.boxplot()` does not perform kernel density estimation; violin plots (`ax.violinplot()`) or manual KDE computation are used for density curves."
        }
    },
    {
        "num": 17,
        "difficulty": "Beginner",
        "topic": "Area plots",
        "type": "single",
        "question": "Which Matplotlib method is commonly used to shade the area between two curves $y_1$ and $y_2$ or between a curve and a baseline?",
        "options": {
            "A": "`ax.fill_between()`",
            "B": "`ax.shade()`",
            "C": "`ax.paint_area()`",
            "D": "`ax.polygon_fill()`"
        },
        "correct": "A",
        "correct_explanation": "`ax.fill_between(x, y1, y2=0, ...)` fills the polygonal region between two horizontal curves $(x, y_1)$ and $(x, y_2)$ or between $(x, y_1)$ and a constant baseline.",
        "incorrect_explanations": {
            "B": "`ax.shade()` is not a standard Matplotlib Axes plotting method.",
            "C": "`ax.paint_area()` does not exist in the Matplotlib API.",
            "D": "`ax.polygon_fill()` does not exist; generic polygons are created with `ax.fill()` or `matplotlib.patches.Polygon`."
        }
    },
    {
        "num": 18,
        "difficulty": "Beginner",
        "topic": "Subplots",
        "type": "single",
        "question": "What subplot configuration is created by the call `plt.subplot(2, 3, 5)`?",
        "options": {
            "A": "A grid of 2 rows and 3 columns, activating the 5th subplot (row 2, column 2)",
            "B": "A grid of 2 columns and 3 rows, activating the 5th subplot",
            "C": "A grid of 5 rows and 3 columns with 2 active axes",
            "D": "A single plot with 2 X-axes and 3 Y-axes"
        },
        "correct": "A",
        "correct_explanation": "`plt.subplot(nrows, ncols, index)` creates a grid with `nrows=2` and `ncols=3` (total 6 positions, numbered 1 to 6 in row-major order). Index 5 corresponds to row 2, column 2.",
        "incorrect_explanations": {
            "B": "The first parameter is `nrows` (2 rows) and the second is `ncols` (3 columns), not the reverse.",
            "C": "The first parameter represents rows (2), not 5.",
            "D": "`plt.subplot()` partitions the Figure into a grid of subplots; it does not add multiple coordinate axes to a single plot."
        }
    },
    {
        "num": 19,
        "difficulty": "Beginner",
        "topic": "Subplots",
        "type": "multi",
        "question": "Which of the following calls create a figure containing exactly 4 subplots?",
        "options": {
            "A": "`fig, axes = plt.subplots(2, 2)`",
            "B": "`fig, axes = plt.subplots(1, 4)`",
            "C": "`fig, axes = plt.subplots(4, 1)`",
            "D": "`fig, axes = plt.subplots(4, 4)`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "2 x 2 = 4 subplots in a 2D square grid.",
            "B": "1 x 4 = 4 subplots arranged in 1 row of 4 columns.",
            "C": "4 x 1 = 4 subplots arranged in 4 rows of 1 column."
        },
        "incorrect_explanations": {
            "D": "4 x 4 = 16 subplots, not 4."
        }
    },
    {
        "num": 20,
        "difficulty": "Beginner",
        "topic": "Titles, labels, legends, and annotations",
        "type": "single",
        "question": "In the object-oriented API, which method is used to set the main title of an individual `Axes` object `ax`?",
        "options": {
            "A": "`ax.set_title('My Title')`",
            "B": "`ax.title('My Title')`",
            "C": "`ax.set_plot_title('My Title')`",
            "D": "`ax.add_title('My Title')`"
        },
        "correct": "A",
        "correct_explanation": "In the object-oriented interface, setting attributes follows the `set_<attribute>` naming convention, so `ax.set_title()` is used. `plt.title()` is the procedural `pyplot` equivalent.",
        "incorrect_explanations": {
            "B": "`ax.title` is an internal attribute / not a setter method; calling `ax.title(...)` raises a `TypeError`.",
            "C": "`ax.set_plot_title()` is not a valid method in Matplotlib.",
            "D": "`ax.add_title()` is not a standard Matplotlib Axes method."
        }
    },
    {
        "num": 21,
        "difficulty": "Beginner",
        "topic": "Titles, labels, legends, and annotations",
        "type": "multi",
        "question": "Which of the following are valid ways to specify the location of a legend using `ax.legend(loc=...)`?",
        "options": {
            "A": "`loc='upper right'`",
            "B": "`loc='lower left'`",
            "C": "`loc='best'`",
            "D": "`loc='middle'`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "'upper right' is a standard location string (corresponds to code 1).",
            "B": "'lower left' is a standard location string (corresponds to code 3).",
            "C": "'best' (code 0) instructs Matplotlib to automatically find a location with minimal overlap on data artists."
        },
        "incorrect_explanations": {
            "D": "'middle' is not a valid legend string; valid center strings are `'center'`, `'center left'`, `'center right'`, `'lower center'`, and `'upper center'`."
        }
    },
    {
        "num": 22,
        "difficulty": "Beginner",
        "topic": "Axis limits and ticks",
        "type": "single",
        "question": "Which method explicitly sets the visible data range of the X-axis on an `Axes` object `ax`?",
        "options": {
            "A": "`ax.set_xlim(left, right)`",
            "B": "`ax.set_xrange(left, right)`",
            "C": "`ax.xlimits(left, right)`",
            "D": "`ax.set_xbound_range(left, right)`"
        },
        "correct": "A",
        "correct_explanation": "`ax.set_xlim(left, right)` sets the minimum (left) and maximum (right) data view limits for the X-axis.",
        "incorrect_explanations": {
            "B": "`ax.set_xrange()` does not exist in Matplotlib.",
            "C": "`ax.xlimits()` is not a valid Matplotlib method.",
            "D": "`ax.set_xbound_range()` is not a valid Matplotlib method (the bounding method is `ax.set_xbound()`)."
        }
    },
    {
        "num": 23,
        "difficulty": "Beginner",
        "topic": "Grid configuration",
        "type": "multi",
        "question": "Which of the following are valid arguments for configuring grid lines using `ax.grid()`?",
        "options": {
            "A": "`visible=True` (or passing `True` positionally) to toggle the grid on",
            "B": "`axis='x'` or `axis='y'` to draw grid lines for only one coordinate axis",
            "C": "`linestyle='--'` and `alpha=0.5` to customize the line style and transparency",
            "D": "`projection='grid3d'` to project 2D grid lines into 3D isometric curves"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`ax.grid(True)` or `ax.grid(visible=True)` enables grid lines.",
            "B": "The `axis` argument accepts `'both'`, `'x'`, or `'y'` to target specific grid orientations.",
            "C": "`linestyle` (or `ls`), `color`, `linewidth`, and `alpha` allow full customization of grid styling."
        },
        "incorrect_explanations": {
            "D": "`projection` is not a valid argument for `ax.grid()`; projections are defined when creating an `Axes` (e.g. `plt.subplots(subplot_kw={'projection': ...})`)."
        }
    },
    {
        "num": 24,
        "difficulty": "Beginner",
        "topic": "Styles and themes",
        "type": "single",
        "question": "How can you view the list of all pre-installed styles available in Matplotlib?",
        "options": {
            "A": "`plt.style.available`",
            "B": "`plt.get_styles()`",
            "C": "`plt.theme_list()`",
            "D": "`matplotlib.styles.list_all()`"
        },
        "correct": "A",
        "correct_explanation": "`plt.style.available` is a list of strings containing all style sheet names packaged with Matplotlib (e.g. `'ggplot'`, `'seaborn-v0_8'`, `'dark_background'`).",
        "incorrect_explanations": {
            "B": "`plt.get_styles()` does not exist in the Matplotlib API.",
            "C": "`plt.theme_list()` does not exist in Matplotlib.",
            "D": "`matplotlib.styles.list_all()` does not exist; the module is `matplotlib.style` and the attribute is `.available`."
        }
    },
    {
        "num": 25,
        "difficulty": "Beginner",
        "topic": "Saving figures",
        "type": "multi",
        "question": "Which of the following statements about `plt.savefig()` are correct?",
        "options": {
            "A": "It automatically infers the image file format from the filename extension (e.g., `'plot.png'`, `'plot.pdf'`, `'plot.svg'`).",
            "B": "Calling `plt.savefig('chart.png')` after `plt.show()` in some environments may save a blank image if the figure was already cleared upon closing.",
            "C": "The `dpi` keyword argument controls the resolution (dots per inch) for raster image outputs.",
            "D": "`plt.savefig()` only supports uncompressed BMP files."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Matplotlib uses the file extension in the output path to automatically select the appropriate file format canvas backend (e.g. PNG, PDF, SVG, EPS).",
            "B": "In interactive GUI environments, closing the GUI window after `plt.show()` clears/closes the figure, so calling `savefig()` afterwards may result in saving an empty canvas. Saving should be done prior to `plt.show()`.",
            "C": "The `dpi` parameter specifies the resolution in dots per inch for raster formats like PNG and JPEG."
        },
        "incorrect_explanations": {
            "D": "`plt.savefig()` supports numerous raster and vector formats including PNG, PDF, SVG, PS, EPS, and JPEG."
        }
    }
]
