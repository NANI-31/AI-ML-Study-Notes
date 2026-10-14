# Questions 26-50: Intermediate Level

questions_part2 = [
    {
        "num": 26,
        "difficulty": "Intermediate",
        "topic": "Object-oriented Matplotlib",
        "type": "single",
        "question": "When working with a 2x2 grid of subplots created via `fig, axes = plt.subplots(2, 2)`, what is the most idiomatic way to iterate over all individual `Axes` objects in a single flat loop?",
        "options": {
            "A": "`for ax in axes.flat:` (or `axes.ravel()`)",
            "B": "`for ax in axes.list():`",
            "C": "`for ax in axes.all():`",
            "D": "`for ax in axes.subplots():`"
        },
        "correct": "A",
        "correct_explanation": "`axes` is returned as a 2D NumPy array. Using the NumPy iterator `.flat` (or flattening the array with `.ravel()` or `.flatten()`) allows direct iteration over all 4 `Axes` objects in a single `for` loop.",
        "incorrect_explanations": {
            "B": "NumPy ndarrays do not have a `.list()` method.",
            "C": "The `.all()` method on NumPy arrays performs a boolean evaluation, not an iteration over elements.",
            "D": "The array of Axes does not possess a `.subplots()` method."
        }
    },
    {
        "num": 27,
        "difficulty": "Intermediate",
        "topic": "Figure and Axes management",
        "type": "multi",
        "question": "Which of the following statements about unpacking the output of `plt.subplots()` are correct?",
        "options": {
            "A": "`fig, (ax1, ax2) = plt.subplots(1, 2)` successfully unpacks a 1D array of 2 horizontal subplots.",
            "B": "`fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)` successfully unpacks a 2x2 grid into four separate variables.",
            "C": "Setting `squeeze=False` guarantees that the returned `axes` object is always a NumPy 2D array, regardless of row/col dimensions.",
            "D": "`plt.subplots(3, 3)` returns a Python dictionary where keys are coordinate tuples `(i, j)`."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "When `nrows=1` and `ncols=2`, `axes` is a 1D array of length 2, matching the 2-tuple `(ax1, ax2)`.",
            "B": "For a 2x2 grid, `axes` is a 2D array of shape `(2, 2)`, matching the nested tuple pattern `((ax1, ax2), (ax3, ax4))`.",
            "C": "By default, extra dimensions are squeezed out; `squeeze=False` forces `axes` to always be a 2D ndarray even for `1x1`, `1xN`, or `Nx1` subplots."
        },
        "incorrect_explanations": {
            "D": "`plt.subplots()` returns a NumPy `ndarray`, not a Python dictionary (subplot dictionary returns are provided by `plt.subplot_mosaic()`)."
        }
    },
    {
        "num": 28,
        "difficulty": "Intermediate",
        "topic": "Figure sizing and DPI",
        "type": "single",
        "question": "In `plt.figure(figsize=(8, 6))`, what unit of measurement is used for the dimensions `(8, 6)`?",
        "options": {
            "A": "Inches",
            "B": "Centimeters",
            "C": "Pixels",
            "D": "Millimeters"
        },
        "correct": "A",
        "correct_explanation": "In Matplotlib, `figsize` dimensions are strictly defined in **inches** `(width, height)`. At a standard DPI of 100, `(8, 6)` corresponds to $800 \\times 600$ pixels.",
        "incorrect_explanations": {
            "B": "Matplotlib does not use centimeters natively for `figsize` (conversion requires dividing cm by 2.54).",
            "C": "Pixel dimensions depend on the DPI (`pixels = inches * dpi`), whereas `figsize` sets physical inches.",
            "D": "Millimeters are not the native unit for `figsize`."
        }
    },
    {
        "num": 29,
        "difficulty": "Intermediate",
        "topic": "Axis limits and ticks",
        "type": "single",
        "question": "What is the recommended object-oriented method combination to place major tick marks at specific numeric locations and assign custom string labels to them?",
        "options": {
            "A": "`ax.set_xticks([0, 1, 2])` followed by `ax.set_xticklabels(['A', 'B', 'C'])` (or `ax.set_xticks(..., labels=...)`)",
            "B": "`ax.ticks([0, 1, 2], labels=['A', 'B', 'C'])`",
            "C": "`ax.set_axis_labels([0, 1, 2], ['A', 'B', 'C'])`",
            "D": "`ax.x_markers([0, 1, 2], text=['A', 'B', 'C'])`"
        },
        "correct": "A",
        "correct_explanation": "`ax.set_xticks(ticks, labels=...)` (or calling `set_xticks()` followed by `set_xticklabels()`) explicitly specifies the tick coordinate positions and their corresponding display labels.",
        "incorrect_explanations": {
            "B": "`ax.ticks()` does not exist in the Matplotlib API.",
            "C": "`ax.set_axis_labels()` does not exist; axis labels are set using `ax.set_xlabel()` and `ax.set_ylabel()`.",
            "D": "`ax.x_markers()` is not a valid method."
        }
    },
    {
        "num": 30,
        "difficulty": "Intermediate",
        "topic": "Tick formatting",
        "type": "multi",
        "question": "Which classes from the `matplotlib.ticker` module are used for controlling tick positioning (locators) and tick text formatting (formatters)?",
        "options": {
            "A": "`MultipleLocator` to place ticks at regular arithmetic multiples",
            "B": "`FuncFormatter` to format tick labels using a custom Python function",
            "C": "`PercentFormatter` to display tick values as percentages automatically",
            "D": "`AutoFontLocator` to dynamically alter the typeface of axis ticks"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`MultipleLocator(base)` places ticks at multiples of `base` (e.g. every 5 or 10 units).",
            "B": "`FuncFormatter(func)` wraps a user-defined function `func(x, pos)` to construct custom string labels.",
            "C": "`PercentFormatter(xmax=...)` automatically converts numeric values into percentage strings."
        },
        "incorrect_explanations": {
            "D": "`AutoFontLocator` does not exist in Matplotlib; font properties are configured via text properties or `rcParams`."
        }
    },
    {
        "num": 31,
        "difficulty": "Intermediate",
        "topic": "Logarithmic and other axis scales",
        "type": "single",
        "question": "How do you set the Y-axis to a base-10 logarithmic scale on an `Axes` object `ax`?",
        "options": {
            "A": "`ax.set_yscale('log')`",
            "B": "`ax.set_log_y(10)`",
            "C": "`ax.scale_y('log10')`",
            "D": "`ax.y_logarithm(True)`"
        },
        "correct": "A",
        "correct_explanation": "`ax.set_yscale('log')` (or `plt.yscale('log')`) sets the scale of the Y-axis to logarithmic (base 10 by default, customizable via `base=...`).",
        "incorrect_explanations": {
            "B": "`ax.set_log_y()` is not a Matplotlib method.",
            "C": "`ax.scale_y()` is not a valid method.",
            "D": "`ax.y_logarithm()` does not exist in Matplotlib."
        }
    },
    {
        "num": 32,
        "difficulty": "Intermediate",
        "topic": "Twin axes",
        "type": "single",
        "question": "Which method is used to create a second Y-axis that shares the exact same X-axis as an existing `Axes` object `ax`?",
        "options": {
            "A": "`ax2 = ax.twinx()`",
            "B": "`ax2 = ax.twiny()`",
            "C": "`ax2 = ax.duplicate(axis='y')`",
            "D": "`ax2 = ax.clone_x()`"
        },
        "correct": "A",
        "correct_explanation": "`ax.twinx()` creates and returns a new `Axes` sharing the X-axis of `ax`, while providing an independent Y-axis on the opposite (right) side.",
        "incorrect_explanations": {
            "B": "`ax.twiny()` shares the Y-axis and creates a secondary X-axis (on the top), which is the opposite.",
            "C": "`ax.duplicate()` does not exist in Matplotlib.",
            "D": "`ax.clone_x()` is not a valid method in Matplotlib."
        }
    },
    {
        "num": 33,
        "difficulty": "Intermediate",
        "topic": "Twin axes",
        "type": "multi",
        "question": "When using `ax2 = ax1.twinx()`, which of the following statements are true?",
        "options": {
            "A": "Both axes overlay the exact same physical bounding box inside the figure.",
            "B": "`ax1` and `ax2` have independent Y-axis limits (`ylim`) and tick formatters.",
            "C": "Changes to the X-axis limits on `ax1` automatically update on `ax2` because they share the X-axis.",
            "D": "Calling `ax1.plot()` and `ax2.plot()` automatically consolidates both lines into a single combined legend when calling `ax1.legend()`."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`twinx()` places `ax2` directly over `ax1` with identical position geometry.",
            "B": "Each twin axes instance maintains its own separate `YAxis` object with distinct limits, scales, and ticks.",
            "C": "The shared X-axis ensures that pan/zoom or limit modifications to X on either axes synchronize immediately."
        },
        "incorrect_explanations": {
            "D": "Because `ax1` and `ax2` are distinct `Axes` objects, `ax1.legend()` only inspects artists attached to `ax1`. Combining legends requires collecting handles and labels from both axes manually: `lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(lines1 + lines2, labels1 + labels2)`."
        }
    },
    {
        "num": 34,
        "difficulty": "Intermediate",
        "topic": "Shared axes",
        "type": "single",
        "question": "What is the primary effect of passing `sharex=True` to `plt.subplots(2, 1, sharex=True)`?",
        "options": {
            "A": "Both subplots share their X-axis view limits and panning/zooming, and tick labels on the upper subplot are hidden by default",
            "B": "Both subplots are merged into a single 3D plot",
            "C": "Data arrays passed to subplot 1 are automatically copied to subplot 2",
            "D": "The X-axis ticks are drawn only on the rightmost margin"
        },
        "correct": "A",
        "correct_explanation": "`sharex=True` links the X-axis coordinate transformations across all subplots, synchronizing limits/zooming and cleanly hiding redundant inner X-tick labels.",
        "incorrect_explanations": {
            "B": "Shared axes link 2D coordinate bounds; they do not convert subplots into 3D plots.",
            "C": "Data plotted on each subplot remains completely independent.",
            "D": "X-axis ticks remain along the bottom edge of the bottom subplot, not the right margin."
        }
    },
    {
        "num": 35,
        "difficulty": "Intermediate",
        "topic": "Shared axes",
        "type": "multi",
        "question": "Which of the following are valid ways to share axes between subplots in Matplotlib?",
        "options": {
            "A": "Passing `sharex=True` or `sharey=True` in `plt.subplots()`",
            "B": "Passing an existing `Axes` instance to the `sharex` or `sharey` argument of `plt.subplot()` or `fig.add_subplot()`",
            "C": "Calling `ax2.sharex(ax1)` explicitly after creating both axes",
            "D": "Setting `ax1.merge_axes(ax2)`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`plt.subplots(nrows, ncols, sharex=True, sharey=True)` is the high-level standard for sharing across grids.",
            "B": "`fig.add_subplot(2, 1, 2, sharex=ax1)` directly links `ax2`'s X-axis to `ax1` during construction.",
            "C": "`ax2.sharex(ax1)` links axes programmatically after creation."
        },
        "incorrect_explanations": {
            "D": "`merge_axes()` is not a Matplotlib method."
        }
    },
    {
        "num": 36,
        "difficulty": "Intermediate",
        "topic": "Date/time plotting",
        "type": "single",
        "question": "Which module in Matplotlib provides specialized locators and formatters for plotting Python `datetime` and NumPy `datetime64` objects?",
        "options": {
            "A": "`matplotlib.dates`",
            "B": "`matplotlib.time`",
            "C": "`matplotlib.chronos`",
            "D": "`matplotlib.calendar`"
        },
        "correct": "A",
        "correct_explanation": "`matplotlib.dates` contains date ticker locators (such as `DayLocator`, `MonthLocator`, `YearLocator`) and formatters (like `DateFormatter`, `ConciseDateFormatter`).",
        "incorrect_explanations": {
            "B": "Matplotlib does not have a `matplotlib.time` module.",
            "C": "`matplotlib.chronos` does not exist.",
            "D": "`matplotlib.calendar` does not exist."
        }
    },
    {
        "num": 37,
        "difficulty": "Intermediate",
        "topic": "Date/time plotting",
        "type": "multi",
        "question": "When plotting time series with crowded date tick labels on the X-axis, which techniques help prevent overlapping text?",
        "options": {
            "A": "Calling `fig.autofmt_xdate()` to automatically rotate tick labels and adjust horizontal alignment",
            "B": "Using `matplotlib.dates.ConciseDateFormatter` for succinct, non-redundant date formatting",
            "C": "Using `ax.tick_params(axis='x', rotation=45)` to rotate tick labels manually",
            "D": "Calling `ax.set_dates_invisible(True)` to purge all time series data from memory"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`fig.autofmt_xdate()` tilts X-tick labels (by default 30 degrees) and aligns them neatly to avoid clipping.",
            "B": "`ConciseDateFormatter` displays dynamic date strings (e.g. showing years only when year changes), drastically reducing label length.",
            "C": "`ax.tick_params(axis='x', rotation=45)` provides direct control over tick text rotation angle."
        },
        "incorrect_explanations": {
            "D": "`ax.set_dates_invisible()` does not exist and deleting data would prevent visualization."
        }
    },
    {
        "num": 38,
        "difficulty": "Intermediate",
        "topic": "Text and annotations",
        "type": "single",
        "question": "What is the role of the `transform` parameter when calling `ax.text(x, y, 'Label', transform=ax.transAxes)`?",
        "options": {
            "A": "It specifies that coordinates `(x, y)` are relative to the Axes box (0,0 = bottom-left, 1,1 = top-right) rather than data coordinates",
            "B": "It applies a 3D isometric rotation to the text string",
            "C": "It translates the text from English to another language",
            "D": "It forces the font to transform to LaTeX mode"
        },
        "correct": "A",
        "correct_explanation": "`transform=ax.transAxes` changes the coordinate system so that `(0, 0)` is the bottom-left corner of the `Axes` and `(1, 1)` is the top-right corner, regardless of the data limits.",
        "incorrect_explanations": {
            "B": "Text rotation is controlled via `rotation=...`, not coordinate transforms.",
            "C": "Transforms in Matplotlib handle mathematical coordinate mappings, not linguistic translation.",
            "D": "LaTeX text rendering is toggled via `usetex=True` or `matplotlib.rcParams['text.usetex']`."
        }
    },
    {
        "num": 39,
        "difficulty": "Intermediate",
        "topic": "Titles, labels, legends, and annotations",
        "type": "multi",
        "question": "In `ax.annotate(text, xy=..., xytext=..., arrowprops=...)`, what do `xy` and `xytext` represent?",
        "options": {
            "A": "`xy` specifies the point being annotated (the tip of the arrow)",
            "B": "`xytext` specifies the position where the text label is placed",
            "C": "`arrowprops` is a dictionary configuring properties (such as color, arrowstyle, width) of the connecting arrow",
            "D": "`xy` and `xytext` must always be identical or an error is raised"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`xy` is the coordinate tuple of the target point being highlighted (where the arrow points).",
            "B": "`xytext` is the coordinate tuple where the text string is positioned.",
            "C": "`arrowprops` configures arrow styling such as `arrowstyle='->'`, `color='black'`, and `lw=1.5`."
        },
        "incorrect_explanations": {
            "D": "`xy` and `xytext` are typically different so that the text is offset from the target point and connected via the arrow; if they were always identical, an arrow could not be drawn."
        }
    },
    {
        "num": 40,
        "difficulty": "Intermediate",
        "topic": "Custom legends",
        "type": "single",
        "question": "How can you place a legend completely outside the plotting area (e.g. to the upper right outside the axes)?",
        "options": {
            "A": "`ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')`",
            "B": "`ax.legend(outside=True)`",
            "C": "`ax.legend(position='external')`",
            "D": "`ax.legend(margin='right')`"
        },
        "correct": "A",
        "correct_explanation": "`bbox_to_anchor=(1.05, 1)` paired with `loc='upper left'` anchors the top-left corner of the legend box just outside the right edge (x = 1.05) of the axes bounding box.",
        "incorrect_explanations": {
            "B": "`outside=True` is not a standard Matplotlib keyword argument for `ax.legend()`.",
            "C": "`position='external'` is not a valid parameter.",
            "D": "`margin='right'` is not a valid legend positioning parameter."
        }
    },
    {
        "num": 41,
        "difficulty": "Intermediate",
        "topic": "Colormaps",
        "type": "multi",
        "question": "Which of the following colormaps in Matplotlib are classified as perceptually uniform sequential colormaps (designed to be colorblind-friendly and monotonically increase in luminance)?",
        "options": {
            "A": "`viridis`",
            "B": "`plasma`",
            "C": "`inferno`",
            "D": "`jet`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`viridis` is the default perceptually uniform sequential colormap introduced in Matplotlib 2.0.",
            "B": "`plasma` is a perceptually uniform sequential colormap transitioning from purple to yellow.",
            "C": "`inferno` is a perceptually uniform sequential colormap transitioning from black/purple through orange to yellow."
        },
        "incorrect_explanations": {
            "D": "`jet` is an older rainbow colormap with non-monotonic luminance gradients and perceptual artifacts; it is not perceptually uniform."
        }
    },
    {
        "num": 42,
        "difficulty": "Intermediate",
        "topic": "imshow()",
        "type": "single",
        "question": "What kind of data does `ax.imshow(X)` display when `X` is a 2D NumPy array of shape `(M, N)`?",
        "options": {
            "A": "A pseudocolor raster image / heatmap where array values are mapped to colors via a colormap",
            "B": "A 3D mesh surface plot",
            "C": "A scatter plot of M points each with N dimensions",
            "D": "A bar chart with M rows and N columns"
        },
        "correct": "A",
        "correct_explanation": "For a 2D array of scalar values, `ax.imshow()` renders a raster image / heatmap by normalizing array values and mapping them through the active colormap (`cmap`).",
        "incorrect_explanations": {
            "B": "3D mesh surfaces are generated with `Axes3D.plot_surface()`, not `imshow()`.",
            "C": "Scatter plots are created with `ax.scatter()`.",
            "D": "Bar charts are created with `ax.bar()`."
        }
    },
    {
        "num": 43,
        "difficulty": "Intermediate",
        "topic": "Images and heatmaps",
        "type": "multi",
        "question": "Which parameters in `ax.imshow()` control pixel aspect ratio and vertical orientation?",
        "options": {
            "A": "`origin='lower'` to place the `[0, 0]` index at the bottom-left corner (standard Cartesian orientation)",
            "B": "`origin='upper'` to place the `[0, 0]` index at the top-left corner (standard matrix/image orientation)",
            "C": "`aspect='auto'` to allow pixels to stretch non-square to fill the axes bounding box",
            "D": "`aspect='isometric3d'` to project image pixels into perspective voxels"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`origin='lower'` aligns array row index 0 with the bottom of the plot, commonly used for Cartesian coordinate data.",
            "B": "`origin='upper'` is the default for `imshow()`, placing row 0 at the top (matching matrix index and computer vision conventions).",
            "C": "`aspect='auto'` unconstrains pixel aspect ratio so the image stretches to fill the available axes dimensions."
        },
        "incorrect_explanations": {
            "D": "`aspect` accepts `'equal'`, `'auto'`, or a float ratio; `'isometric3d'` is invalid."
        }
    },
    {
        "num": 44,
        "difficulty": "Intermediate",
        "topic": "Colorbars",
        "type": "single",
        "question": "Consider the following code snippet:\n\n```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfig, ax = plt.subplots()\ndata = np.random.rand(10, 10)\nim = ax.imshow(data, cmap='viridis')\nfig.colorbar(im, ax=ax)\n```\n\nWhat is the first argument passed to `fig.colorbar()`?",
        "options": {
            "A": "A `ScalarMappable` artist (such as the `AxesImage` returned by `imshow` or a `PathCollection` from `scatter`)",
            "B": "The raw 2D NumPy array `data`",
            "C": "The string name of the colormap `'viridis'`",
            "D": "The `Axes` object `ax`"
        },
        "correct": "A",
        "correct_explanation": "`fig.colorbar(mappable, ax=...)` requires a `ScalarMappable` instance (such as the object returned by `imshow()`, `contourf()`, or `scatter()`) to extract the colormap and normalization limits.",
        "incorrect_explanations": {
            "B": "Passing raw data arrays raises a `TypeError`; the colorbar requires the mappable artist that binds normalization and colormap.",
            "C": "Colormap names cannot be passed alone without a ScalarMappable.",
            "D": "`ax` is supplied to the `ax` keyword argument to indicate which subplot should be shrunk to accommodate the colorbar, not as the first positional argument."
        }
    },
    {
        "num": 45,
        "difficulty": "Intermediate",
        "topic": "Contour plots",
        "type": "multi",
        "question": "What is the primary difference between `ax.contour(X, Y, Z)` and `ax.contourf(X, Y, Z)`?",
        "options": {
            "A": "`ax.contour()` draws isolines (contour lines) at specified level boundaries.",
            "B": "`ax.contourf()` draws filled polygonal regions between contour levels.",
            "C": "Both functions require 2D coordinate grids for `X`, `Y`, and `Z` (such as generated by `np.meshgrid`).",
            "D": "`ax.contourf()` is strictly for 3D surfaces while `ax.contour()` is strictly 2D."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`contour()` produces unfilled contour line artists (`QuadContourSet`).",
            "B": "The 'f' in `contourf()` stands for 'filled', creating solid colored regions between level thresholds.",
            "C": "Both accept 2D matrices representing the spatial coordinate grid and field elevations."
        },
        "incorrect_explanations": {
            "D": "Both `contour()` and `contourf()` are 2D visualization functions (though 3D projections also support `ax.contour()` at specific planes)."
        }
    },
    {
        "num": 46,
        "difficulty": "Intermediate",
        "topic": "tight_layout()",
        "type": "single",
        "question": "What problem does calling `plt.tight_layout()` or `fig.tight_layout()` primarily solve?",
        "options": {
            "A": "It automatically adjusts subplot parameters and margins so that titles, axis labels, and tick labels do not overlap adjacent subplots",
            "B": "It compresses the output file size by reducing color bit-depth",
            "C": "It snaps all data markers to the nearest integer grid coordinate",
            "D": "It forces all subplots to have identical aspect ratios and data limits"
        },
        "correct": "A",
        "correct_explanation": "`tight_layout()` dynamically inspects the bounding boxes of text artists (tick labels, axis labels, titles) and recalculates subplot spacings to eliminate overlaps.",
        "incorrect_explanations": {
            "B": "Image compression is managed by file export libraries during `savefig()`, not `tight_layout()`.",
            "C": "Data coordinates are unaffected by layout management.",
            "D": "Aspect ratios and limits are determined by data and axis settings, not `tight_layout()`."
        }
    },
    {
        "num": 47,
        "difficulty": "Intermediate",
        "topic": "constrained_layout",
        "type": "multi",
        "question": "Which statements regarding `layout='constrained'` (or `constrained_layout=True`) in modern Matplotlib are correct?",
        "options": {
            "A": "It can be activated during figure initialization via `plt.subplots(layout='constrained')` or `plt.figure(layout='constrained')`.",
            "B": "Unlike `tight_layout()`, it dynamically re-evaluates and maintains clean spacing even when figure dimensions change before rendering.",
            "C": "It seamlessly handles complex layouts containing colorbars and multi-row/col legends attached to subplots.",
            "D": "It is completely incompatible with `GridSpec` and raises an `ImportError` whenever subplots are present."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Modern Matplotlib supports `layout='constrained'` (and `layout='compressed'`) directly in figure constructors.",
            "B": "Constrained layout utilizes a solver that optimizes geometry continuously before each draw rather than computing static margins once.",
            "C": "It automatically reserves margin space for associated colorbars and legends without distorting adjacent subplot alignments."
        },
        "incorrect_explanations": {
            "D": "Constrained layout works seamlessly with `GridSpec` and `subplot_mosaic()`."
        }
    },
    {
        "num": 48,
        "difficulty": "Intermediate",
        "topic": "Figure sizing and DPI",
        "type": "single",
        "question": "If a figure is created with `plt.figure(figsize=(10, 5), dpi=200)`, what will be the pixel dimensions of the resulting rendered raster image?",
        "options": {
            "A": "2000 pixels wide by 1000 pixels high",
            "B": "1000 pixels wide by 500 pixels high",
            "C": "500 pixels wide by 250 pixels high",
            "D": "200 pixels wide by 200 pixels high"
        },
        "correct": "A",
        "correct_explanation": "Pixel dimension equals $\\text{inches} \\times \\text{DPI}$. Thus, width $= 10 \\times 200 = 2000$ pixels, and height $= 5 \\times 200 = 1000$ pixels.",
        "incorrect_explanations": {
            "B": "This would correspond to a DPI of 100 ($10 \\times 100 = 1000$).",
            "C": "This would correspond to a DPI of 50.",
            "D": "The calculation does not multiply the dimensions by the specified DPI."
        }
    },
    {
        "num": 49,
        "difficulty": "Intermediate",
        "topic": "Saving figures",
        "type": "single",
        "question": "Which argument in `plt.savefig()` automatically computes a snug bounding box around all plotted artists and eliminates excessive surrounding whitespace from the saved file?",
        "options": {
            "A": "`bbox_inches='tight'`",
            "B": "`autocrop=True`",
            "C": "`fit='tight'`",
            "D": "`trim_whitespace=True`"
        },
        "correct": "A",
        "correct_explanation": "Passing `bbox_inches='tight'` instructs Matplotlib to calculate the bounding box containing all visible artists (including legends, titles, and labels that might otherwise be clipped) and crop the canvas accordingly.",
        "incorrect_explanations": {
            "B": "`autocrop` is not a parameter of `plt.savefig()`.",
            "C": "`fit='tight'` does not exist in the Matplotlib API.",
            "D": "`trim_whitespace` is not a valid parameter."
        }
    },
    {
        "num": 50,
        "difficulty": "Intermediate",
        "topic": "File formats",
        "type": "single",
        "question": "Which of the following saved file formats preserves infinite scalability without pixelation or resolution loss because it stores geometric drawing commands rather than a grid of pixels?",
        "options": {
            "A": "SVG (Scalable Vector Graphics) and PDF (Portable Document Format)",
            "B": "PNG (Portable Network Graphics)",
            "C": "JPEG (Joint Photographic Experts Group)",
            "D": "GIF (Graphics Interchange Format)"
        },
        "correct": "A",
        "correct_explanation": "SVG, PDF, and EPS are vector formats that store shapes, paths, lines, and text as mathematical primitives, allowing infinite scaling without pixelation.",
        "incorrect_explanations": {
            "B": "PNG is a raster bitmap format consisting of a fixed pixel grid; scaling causes pixelation.",
            "C": "JPEG is a lossy raster bitmap format with fixed pixel resolution.",
            "D": "GIF is an 8-bit raster bitmap format."
        }
    }
]
