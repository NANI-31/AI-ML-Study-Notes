# Questions 51-75: Advanced Level

questions_part3 = [
    {
        "num": 51,
        "difficulty": "Advanced",
        "topic": "Layout management",
        "type": "single",
        "question": "Which Matplotlib class enables the creation of multi-column and multi-row spanning subplots using NumPy-like slicing on a figure grid?",
        "options": {
            "A": "`matplotlib.gridspec.GridSpec`",
            "B": "`matplotlib.layout.FlexGrid`",
            "C": "`matplotlib.figure.SubplotArray`",
            "D": "`matplotlib.axes.GridSpan`"
        },
        "correct": "A",
        "correct_explanation": "`GridSpec` defines a flexible grid structure across a `Figure`. Subplots can span multiple rows and columns using slice syntax (e.g. `ax = fig.add_subplot(gs[0, :])` or `ax = fig.add_subplot(gs[1:, 0])`).",
        "incorrect_explanations": {
            "B": "`FlexGrid` does not exist in Matplotlib.",
            "C": "`SubplotArray` does not exist in Matplotlib.",
            "D": "`GridSpan` does not exist in Matplotlib."
        }
    },
    {
        "num": 52,
        "difficulty": "Advanced",
        "topic": "Subplots",
        "type": "multi",
        "question": "Consider the modern `subplot_mosaic()` feature:\n\n```python\nfig, axd = plt.subplot_mosaic([['left', 'upper_right'],\n                               ['left', 'lower_right']])\n```\n\nWhich of the following statements about this code are correct?",
        "options": {
            "A": "`axd` is returned as a Python dictionary where keys `'left'`, `'upper_right'`, and `'lower_right'` map to their respective `Axes` objects.",
            "B": "The `'left'` subplot spans across both rows in the first column.",
            "C": "Using `'.'` in place of a string label in the layout matrix creates an empty whitespace cell where no Axes is added.",
            "D": "`subplot_mosaic()` converts all axes to 3D automatically."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`subplot_mosaic` returns an `axd` dictionary with key names matching the ASCII/nested list structure.",
            "B": "Repeating `'left'` in rows 0 and 1 creates a single contiguous Axes spanning both rows.",
            "C": "The dot `'.'` character acts as an intentional empty placeholder within mosaic specifications."
        },
        "incorrect_explanations": {
            "D": "Axes created by `subplot_mosaic()` are standard 2D axes unless a custom projection is specified in `per_subplot_kw`."
        }
    },
    {
        "num": 53,
        "difficulty": "Advanced",
        "topic": "Multiple axes",
        "type": "single",
        "question": "What is the recommended modern method to create an embedded inset zoom plot inside an existing `Axes` object `ax`?",
        "options": {
            "A": "`axin = ax.inset_axes([x0, y0, width, height])`",
            "B": "`axin = ax.create_zoom_box([x0, y0, width, height])`",
            "C": "`axin = ax.add_subaxes([x0, y0, width, height])`",
            "D": "`axin = ax.embed_axes([x0, y0, width, height])`"
        },
        "correct": "A",
        "correct_explanation": "`ax.inset_axes([x0, y0, width, height])` creates a child `Axes` positioned in normalized axes coordinates $(0 \\text{ to } 1)$ relative to the parent `ax`, and pairs with `ax.indicate_inset_zoom(axin)` to draw connection lines.",
        "incorrect_explanations": {
            "B": "`create_zoom_box()` is not a Matplotlib method.",
            "C": "`add_subaxes()` does not exist in Matplotlib.",
            "D": "`embed_axes()` does not exist in Matplotlib."
        }
    },
    {
        "num": 54,
        "difficulty": "Advanced",
        "topic": "Figure and Axes management",
        "type": "multi",
        "question": "Which of the following code snippets correctly manipulate the perimeter border lines (spines) of an `Axes` object `ax`?",
        "options": {
            "A": "`ax.spines['top'].set_visible(False)` and `ax.spines['right'].set_visible(False)` to remove top and right borders",
            "B": "`ax.spines['left'].set_position(('data', 0))` to center the Y-axis spine at $x=0$",
            "C": "`ax.spines['bottom'].set_position(('outward', 10))` to offset the X-axis spine 10 points outward",
            "D": "`ax.spines.delete_all()` to purge all artists from memory"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Setting `set_visible(False)` on individual spines is the standard approach to creating minimalist / Tufte-style plots.",
            "B": "Spine positions can be bound to data coordinates using `('data', value)`.",
            "C": "Spines can be displaced outward by a point offset using `('outward', points)`."
        },
        "incorrect_explanations": {
            "D": "`ax.spines` is a container dictionary mapping spine names (`'left'`, `'right'`, `'bottom'`, `'top'`) to `Spine` objects; it does not have a `delete_all()` method."
        }
    },
    {
        "num": 55,
        "difficulty": "Advanced",
        "topic": "Tick formatting",
        "type": "single",
        "question": "When defining a custom tick formatting function for `matplotlib.ticker.FuncFormatter`, what are the expected parameters that the user function must accept?",
        "options": {
            "A": "`def custom_formatter(x, pos):` where `x` is the tick numeric value and `pos` is the tick index position",
            "B": "`def custom_formatter(text, font):` where `text` is the raw string and `font` is the font family",
            "C": "`def custom_formatter(ax, fig):` where `ax` is the parent axes and `fig` is the parent figure",
            "D": "`def custom_formatter(val):` requiring strictly a single scalar input"
        },
        "correct": "A",
        "correct_explanation": "A `FuncFormatter` callback must accept two arguments: the numeric tick value `x` (float) and the integer position index `pos` (which may be `None` in some render passes), and return a formatted string.",
        "incorrect_explanations": {
            "B": "Formatters receive numeric values and tick index positions, not string/font objects.",
            "C": "Axes and Figure references are not passed to tick formatting callbacks.",
            "D": "Passing a 1-parameter function without a default argument for `pos` will raise a `TypeError: custom_formatter() takes 1 positional argument but 2 were given` when rendered by the ticker engine."
        }
    },
    {
        "num": 56,
        "difficulty": "Advanced",
        "topic": "Twin axes",
        "type": "multi",
        "question": "When creating three or more Y-axes on a single plot by chaining `ax.twinx()`, which steps are necessary to ensure the additional Y-axis ticks and labels do not overlap the first twin axis?",
        "options": {
            "A": "Use `ax3.spines['right'].set_position(('outward', 60))` to displace the spine outward.",
            "B": "Ensure `ax3.spines['right'].set_visible(True)` is enabled if spines were altered.",
            "C": "Set `ax3.patch.set_visible(False)` (or ensure the underlying axes patch is transparent) so `ax3` does not block `ax1` and `ax2`.",
            "D": "Call `ax3.set_multiaxis(True)` to enable 3D depth-sorting."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Shifting the spine outward via `('outward', offset)` prevents the third axis from colliding directly with the secondary right spine.",
            "B": "The right spine on `ax3` must be visible to display the offset tick line.",
            "C": "Twin axes have transparent background patches by default; keeping them transparent ensures underlying plots remain visible."
        },
        "incorrect_explanations": {
            "D": "`set_multiaxis()` is not a Matplotlib method."
        }
    },
    {
        "num": 57,
        "difficulty": "Advanced",
        "topic": "Polar plots",
        "type": "single",
        "question": "How do you create an `Axes` configured for polar coordinates $(r, \\theta)$ in Matplotlib?",
        "options": {
            "A": "`fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})` (or `plt.subplot(111, polar=True)`)",
            "B": "`fig, ax = plt.subplots(projection='radial')`",
            "C": "`fig, ax = plt.subplots(mode='circular')`",
            "D": "`fig, ax = plt.subplots(coordinate_system='spherical')`"
        },
        "correct": "A",
        "correct_explanation": "Specifying `projection='polar'` (or `polar=True`) creates a `PolarAxes` where angles $\\theta$ are plotted azimuthally and radii $r$ are plotted radially.",
        "incorrect_explanations": {
            "B": "The registered projection identifier is `'polar'`, not `'radial'`.",
            "C": "`mode` is not a valid parameter for `plt.subplots()`.",
            "D": "`coordinate_system` is not a valid subplot parameter."
        }
    },
    {
        "num": 58,
        "difficulty": "Advanced",
        "topic": "Polar plots",
        "type": "multi",
        "question": "Which of the following statements about polar plotting with `ax.plot(theta, r)` on a `PolarAxes` are correct?",
        "options": {
            "A": "The angular coordinate `theta` must be provided in radians by default.",
            "B": "`ax.set_theta_zero_location('N')` sets the 0-degree reference angle to point North (top).",
            "C": "`ax.set_theta_direction(-1)` configures angles to increase clockwise instead of the default counter-clockwise.",
            "D": "Polar axes cannot display grid lines or legends."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Matplotlib's `PolarAxes` trigonometric computations assume angle coordinates in radians.",
            "B": "`set_theta_zero_location()` accepts cardinal directions (`'N'`, `'E'`, `'S'`, `'W'`, etc.) to position $0^\\circ$.",
            "C": "`set_theta_direction(1)` is counter-clockwise and `set_theta_direction(-1)` is clockwise."
        },
        "incorrect_explanations": {
            "D": "Polar axes fully support radial/azimuthal grid lines, tick formatters, and legends."
        }
    },
    {
        "num": 59,
        "difficulty": "Advanced",
        "topic": "3D plotting",
        "type": "single",
        "question": "Which method is the standard object-oriented approach to initialize a 3D subplot in modern Matplotlib?",
        "options": {
            "A": "`fig, ax = plt.subplots(subplot_kw={'projection': '3d'})`",
            "B": "`fig, ax = plt.subplots(3d=True)`",
            "C": "`fig, ax = plt.subplots(dimension=3)`",
            "D": "`fig, ax = plt.subplots(engine='opengl3d')`"
        },
        "correct": "A",
        "correct_explanation": "Passing `subplot_kw={'projection': '3d'}` initializes an `Axes3D` instance from the `mpl_toolkits.mplot3d` toolkit inside standard `plt.subplots()`.",
        "incorrect_explanations": {
            "B": "`3d=True` is not valid Python syntax (identifiers cannot begin with a number) and is not a parameter.",
            "C": "`dimension` is not a valid argument for `plt.subplots()`.",
            "D": "`engine` is not a parameter of `plt.subplots()`."
        }
    },
    {
        "num": 60,
        "difficulty": "Advanced",
        "topic": "mplot3d",
        "type": "single",
        "question": "Which method on an `Axes3D` object `ax` is specifically used to render a continuous, shaded 3D polygonal surface mesh with colormapping and surface lighting?",
        "options": {
            "A": "`ax.plot_surface(X, Y, Z)`",
            "B": "`ax.plot_wireframe(X, Y, Z)`",
            "C": "`ax.draw_mesh3d(X, Y, Z)`",
            "D": "`ax.render_surface(X, Y, Z)`"
        },
        "correct": "A",
        "correct_explanation": "`ax.plot_surface(X, Y, Z, cmap=..., rstride=..., cstride=...)` generates a fully shaded 3D polygonal surface mesh from coordinate matrices `X`, `Y`, and `Z`.",
        "incorrect_explanations": {
            "B": "`ax.plot_wireframe()` draws only unshaded outline grid lines connecting data points without filling surface polygons.",
            "C": "`draw_mesh3d()` is not a method in Matplotlib.",
            "D": "`render_surface()` does not exist in the Matplotlib API."
        }
    },
    {
        "num": 61,
        "difficulty": "Advanced",
        "topic": "3D plotting",
        "type": "single",
        "question": "In an `Axes3D` subplot, which method adjusts the 3D camera viewing angle by specifying elevation and azimuth?",
        "options": {
            "A": "`ax.view_init(elev=30, azim=45)`",
            "B": "`ax.set_camera_angle(elev=30, azim=45)`",
            "C": "`ax.rotate_3d(pitch=30, yaw=45)`",
            "D": "`ax.set_viewpoint(altitude=30, longitude=45)`"
        },
        "correct": "A",
        "correct_explanation": "`ax.view_init(elev=..., azim=..., roll=..., vertical_axis=...)` configures the 3D projection viewing angles in degrees.",
        "incorrect_explanations": {
            "B": "`set_camera_angle()` does not exist in Matplotlib.",
            "C": "`rotate_3d()` does not exist in Matplotlib.",
            "D": "`set_viewpoint()` is not a valid method."
        }
    },
    {
        "num": 62,
        "difficulty": "Advanced",
        "topic": "Colormaps",
        "type": "multi",
        "question": "Which normalization classes from `matplotlib.colors` allow mapping data values to colors across non-linear or centered scales?",
        "options": {
            "A": "`Normalize(vmin, vmax)` for standard linear mapping",
            "B": "`LogNorm(vmin, vmax)` for logarithmic mapping across positive dynamic ranges",
            "C": "`TwoSlopeNorm(vcenter=0, vmin=-10, vmax=50)` for diverging data where the center point is asymmetric",
            "D": "`QuantumNorm(wave_packet)` to automatically compute quantum probability density amplitudes"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`Normalize` provides standard linear interpolation between `vmin` and `vmax`.",
            "B": "`LogNorm` maps logarithmic orders of magnitude linearly into the 0-1 color range.",
            "C": "`TwoSlopeNorm` (formerly `DivergingNorm`) allows the center transition value (`vcenter`) to be positioned asymmetrically between `vmin` and `vmax`."
        },
        "incorrect_explanations": {
            "D": "`QuantumNorm` does not exist in `matplotlib.colors`; custom discrete/binning normalizations use `BoundaryNorm` or `SymLogNorm`."
        }
    },
    {
        "num": 63,
        "difficulty": "Advanced",
        "topic": "Colormaps",
        "type": "single",
        "question": "Which class in `matplotlib.colors` is used to create a custom colormap from a simple discrete list of color hex codes or names?",
        "options": {
            "A": "`matplotlib.colors.ListedColormap(['#ff0000', '#00ff00', '#0000ff'])`",
            "B": "`matplotlib.colors.DiscreteSpectrum(['#ff0000', '#00ff00', '#0000ff'])`",
            "C": "`matplotlib.colors.ColorListMap(['#ff0000', '#00ff00', '#0000ff'])`",
            "D": "`matplotlib.colors.PaletteArray(['#ff0000', '#00ff00', '#0000ff'])`"
        },
        "correct": "A",
        "correct_explanation": "`ListedColormap` generates a colormap lookup table directly from a sequence of discrete color specifications.",
        "incorrect_explanations": {
            "B": "`DiscreteSpectrum` does not exist in Matplotlib.",
            "C": "`ColorListMap` does not exist in Matplotlib.",
            "D": "`PaletteArray` is not a Matplotlib class."
        }
    },
    {
        "num": 64,
        "difficulty": "Advanced",
        "topic": "Colorbars",
        "type": "multi",
        "question": "What is the difference between passing `ax=ax` versus `cax=cax` to `fig.colorbar(mappable, ...)`?",
        "options": {
            "A": "`cax` specifies an exact pre-existing `Axes` into which the colorbar will be drawn directly.",
            "B": "`ax` specifies the parent `Axes` from which space will be automatically stolen to allocate a new colorbar axes.",
            "C": "Passing `ax=[ax1, ax2]` steals space proportionally across multiple subplots to place a shared colorbar.",
            "D": "`cax` requires that `ax` is also simultaneously provided or an error occurs."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "When `cax` is provided, Matplotlib renders the colorbar precisely within that designated `Axes` without resizing any other axes.",
            "B": "When `ax` is passed, Matplotlib resizes the specified `Axes` to create an adjacent colorbar axes.",
            "C": "Passing a sequence of axes to `ax` creates a unified colorbar spanning across that group of subplots."
        },
        "incorrect_explanations": {
            "D": "`cax` and `ax` are mutually exclusive in positioning logic; passing `cax` alone is completely valid and does not require `ax`."
        }
    },
    {
        "num": 65,
        "difficulty": "Advanced",
        "topic": "Line plots",
        "type": "single",
        "question": "Which Matplotlib function is used to plot 2D vector field streamlines (showing flow direction and velocity trajectories smoothly)?",
        "options": {
            "A": "`ax.streamplot(X, Y, U, V)`",
            "B": "`ax.vector_flow(X, Y, U, V)`",
            "C": "`ax.flow_field(X, Y, U, V)`",
            "D": "`ax.trajectory(X, Y, U, V)`"
        },
        "correct": "A",
        "correct_explanation": "`ax.streamplot(X, Y, U, V)` plots streamlines of a 2D vector field defined on a regular grid by vector components $(U, V)$.",
        "incorrect_explanations": {
            "B": "`vector_flow()` does not exist in Matplotlib.",
            "C": "`flow_field()` does not exist in Matplotlib.",
            "D": "`trajectory()` does not exist in Matplotlib."
        }
    },
    {
        "num": 66,
        "difficulty": "Advanced",
        "topic": "Coordinate systems and transforms",
        "type": "single",
        "question": "In Matplotlib's transformation framework, what coordinate system does `ax.transData` represent?",
        "options": {
            "A": "The coordinate system of the user's data (defined by `xlim` and `ylim`), mapping data points to display pixels",
            "B": "The bounding box of the entire figure window, where $(0, 0)$ is bottom-left of the window and $(1, 1)$ is top-right",
            "C": "Raw physical hardware pixels on the display monitor",
            "D": "Inches from the bottom-left corner of the printed page"
        },
        "correct": "A",
        "correct_explanation": "`ax.transData` transforms coordinates from data units $(x, y)$ into display/pixel space on the screen, updating dynamically as axis limits change.",
        "incorrect_explanations": {
            "B": "The normalized coordinate system of the figure window is `fig.transFigure`.",
            "C": "Raw display pixel coordinates correspond to the identity transform (`None` or `matplotlib.transforms.IdentityTransform()`).",
            "D": "Physical inches are handled by figure DPI and physical transformations (`fig.dpi_scale_trans`)."
        }
    },
    {
        "num": 67,
        "difficulty": "Advanced",
        "topic": "Coordinate systems and transforms",
        "type": "multi",
        "question": "Which of the following transform objects are standard built-in coordinate transformations in Matplotlib?",
        "options": {
            "A": "`ax.transAxes` (normalized coordinate system of the Axes: (0,0) bottom-left to (1,1) top-right)",
            "B": "`fig.transFigure` (normalized coordinate system of the Figure: (0,0) bottom-left to (1,1) top-right)",
            "C": "`ax.transData` (data coordinate system)",
            "D": "`ax.transGlobalGPS` (automatic conversion of latitude/longitude to satellite pixels without GIS packages)"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`ax.transAxes` is the 0-to-1 coordinate space of the individual Axes.",
            "B": "`fig.transFigure` is the 0-to-1 coordinate space of the entire Figure canvas.",
            "C": "`ax.transData` converts data coordinate values to pixel coordinates."
        },
        "incorrect_explanations": {
            "D": "`transGlobalGPS` does not exist in standard Matplotlib; geographic mappings require specialized GIS toolkits like Cartopy or GeoPandas."
        }
    },
    {
        "num": 68,
        "difficulty": "Advanced",
        "topic": "Colors, markers, and line styles",
        "type": "single",
        "question": "How can you construct an entirely custom marker shape using vector vertices and drawing codes in Matplotlib?",
        "options": {
            "A": "Pass a `matplotlib.path.Path` object to the `marker` parameter of `ax.plot()` or `ax.scatter()`",
            "B": "Upload an `.obj` 3D mesh to `ax.set_marker_obj()`",
            "C": "Pass a raw string of hexadecimal machine code to `marker`",
            "D": "Custom marker shapes are not supported in Matplotlib"
        },
        "correct": "A",
        "correct_explanation": "A custom `Path` (defined with coordinate vertices and `Path.MOVETO`, `LINETO`, `CLOSEPOLY` codes) can be passed directly to the `marker` argument to render bespoke 2D marker glyphs.",
        "incorrect_explanations": {
            "B": "`set_marker_obj()` does not exist in Matplotlib.",
            "C": "Markers are defined via symbols, paths, or vertex sequences, not raw machine code.",
            "D": "Custom markers are fully supported via `matplotlib.path.Path` and tuples of vertices."
        }
    },
    {
        "num": 69,
        "difficulty": "Advanced",
        "topic": "Figure, Axes, Axis, and related objects",
        "type": "single",
        "question": "Which class in `matplotlib.patches` is used to create a 2D rectangular box defined by a lower-left anchor `(x, y)`, `width`, and `height` to add to an `Axes` via `ax.add_patch()`?",
        "options": {
            "A": "`matplotlib.patches.Rectangle((x, y), width, height)`",
            "B": "`matplotlib.patches.Box((x, y), width, height)`",
            "C": "`matplotlib.patches.Square((x, y), width, height)`",
            "D": "`matplotlib.patches.RectArea((x, y), width, height)`"
        },
        "correct": "A",
        "correct_explanation": "`matplotlib.patches.Rectangle(xy, width, height, angle=0.0, ...)` defines a 2D axis-aligned or rotated rectangle with lower-left anchor `xy`.",
        "incorrect_explanations": {
            "B": "`Box` is not a standard patch class in `matplotlib.patches`.",
            "C": "`Square` does not exist in `matplotlib.patches` (squares are created using `Rectangle` with equal width and height).",
            "D": "`RectArea` does not exist in Matplotlib."
        }
    },
    {
        "num": 70,
        "difficulty": "Advanced",
        "topic": "Styles and themes",
        "type": "single",
        "question": "How can you temporarily apply a Matplotlib style sheet to a single isolated block of code without modifying the global `rcParams` permanently?",
        "options": {
            "A": "Using the context manager `with plt.style.context('dark_background'):`",
            "B": "Calling `plt.style.isolate('dark_background')`",
            "C": "Calling `plt.push_style('dark_background')` followed by `plt.pop_style()`",
            "D": "Wrapping the code in `try: plt.style.use() finally: plt.style.undo()`"
        },
        "correct": "A",
        "correct_explanation": "`plt.style.context(...)` is a Python context manager that temporarily applies the specified style dictionary/sheet only within the `with` block and automatically restores previous `rcParams` upon exit.",
        "incorrect_explanations": {
            "B": "`plt.style.isolate()` does not exist.",
            "C": "`plt.push_style()` does not exist in Matplotlib.",
            "D": "`plt.style.undo()` is not a valid method."
        }
    },
    {
        "num": 71,
        "difficulty": "Advanced",
        "topic": "Styles and themes",
        "type": "multi",
        "question": "Which of the following statements about `matplotlib.rcParams` are correct?",
        "options": {
            "A": "`rcParams` is a dictionary-like container that stores global default configuration values for fonts, colors, lines, and backends.",
            "B": "Modifications like `plt.rcParams['font.size'] = 14` affect all subsequent plots created in that session.",
            "C": "`matplotlib.rcdefaults()` restores all `rcParams` back to Matplotlib's internal default values.",
            "D": "`rcParams` can only be set once during Python installation and cannot be modified at runtime."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`rcParams` (runtime parameters) holds defaults for all visual and backend settings.",
            "B": "Mutating `rcParams` modifies global plotting state for all newly instantiated figures/axes.",
            "C": "`matplotlib.rcdefaults()` resets all settings to factory defaults."
        },
        "incorrect_explanations": {
            "D": "`rcParams` can be modified dynamically at any time in Python code or loaded from files via `plt.rc_file()`."
        }
    },
    {
        "num": 72,
        "difficulty": "Advanced",
        "topic": "Date/time plotting",
        "type": "single",
        "question": "Which locator from `matplotlib.dates` places ticks strictly on the first day of every month?",
        "options": {
            "A": "`matplotlib.dates.MonthLocator()` (or `MonthLocator(bymonthday=1)`)",
            "B": "`matplotlib.dates.DayLocator(interval=30)`",
            "C": "`matplotlib.dates.FirstDayTicker()`",
            "D": "`matplotlib.dates.CalendarStepLocator('monthly')`"
        },
        "correct": "A",
        "correct_explanation": "`MonthLocator()` locates calendar month boundaries (by default the first day of each month, accounting for varying month lengths).",
        "incorrect_explanations": {
            "B": "`DayLocator(interval=30)` steps rigidly every 30 days, which drifts out of sync with actual calendar month boundaries.",
            "C": "`FirstDayTicker` does not exist in Matplotlib.",
            "D": "`CalendarStepLocator` does not exist in Matplotlib."
        }
    },
    {
        "num": 73,
        "difficulty": "Advanced",
        "topic": "Logarithmic and other axis scales",
        "type": "multi",
        "question": "Which of the following axis scale types are natively supported in Matplotlib via `ax.set_yscale(scale_name)`?",
        "options": {
            "A": "`'linear'`",
            "B": "`'log'`",
            "C": "`'symlog'` (symmetrical log, supporting positive and negative values including zero)",
            "D": "`'parabolic'` (to square root all coordinates automatically)"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`'linear'` is the standard linear Cartesian scale.",
            "B": "`'log'` is the standard logarithmic scale for positive values.",
            "C": "`'symlog'` uses a linear scale near zero (within `linthresh`) and a log scale elsewhere, handling negative values smoothly."
        },
        "incorrect_explanations": {
            "D": "`'parabolic'` is not a registered scale in Matplotlib (other standard scales include `'logit'`, `'asinh'`, and `'function'`)."
        }
    },
    {
        "num": 74,
        "difficulty": "Advanced",
        "topic": "Multiple axes",
        "type": "single",
        "question": "How is a 'broken axis' (a plot with a broken/interrupted range on the Y-axis to display widely separated data points) typically created in Matplotlib?",
        "options": {
            "A": "By creating two subplots stacked vertically with `sharex=True`, setting different `ylim` on each, hiding adjacent spines, and drawing diagonal break slash markers",
            "B": "By calling `ax.set_ylim_break(start=100, end=500)`",
            "C": "By setting `ax.broken_axis = True`",
            "D": "By setting `ax.set_yscale('interrupted')`"
        },
        "correct": "A",
        "correct_explanation": "Matplotlib creates broken axes by placing two adjacent subplots (e.g. `ax_top` and `ax_bottom`), adjusting their respective `ylim`, hiding top/bottom border spines, and drawing small diagonal tick marks (slashes) across the spine gap.",
        "incorrect_explanations": {
            "B": "`set_ylim_break()` is not a Matplotlib method.",
            "C": "`broken_axis` is not a property of `Axes`.",
            "D": "`'interrupted'` is not a registered scale in Matplotlib."
        }
    },
    {
        "num": 75,
        "difficulty": "Advanced",
        "topic": "Transparency and rendering",
        "type": "multi",
        "question": "Which of the following statements about `zorder` and visual layering in Matplotlib are correct?",
        "options": {
            "A": "Artists with a higher `zorder` value are drawn on top of artists with a lower `zorder` value.",
            "B": "By default, grid lines typically have `zorder=0.5` or `zorder=1`, meaning lines and bars (`zorder=2` or `3`) are drawn over the grid.",
            "C": "Setting `ax.set_axisbelow(True)` sets the grid lines and tick marks to be rendered beneath data artists.",
            "D": "`zorder` can only be integers from 1 to 5 and cannot take custom values like 10 or 100."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`zorder` determines the painting order (back-to-front painter's algorithm) in the canvas pipeline.",
            "B": "Default standard primitives have predefined z-orders (e.g. Spines/Ticks ~ 1.5-2, Lines ~ 2, Bars ~ 3, Text/Annotations ~ 3-5).",
            "C": "`ax.set_axisbelow(True)` forces axis ticks and grid lines behind all data patches and lines."
        },
        "incorrect_explanations": {
            "D": "`zorder` accepts any real numeric value (including floats, negative numbers, and high numbers like 100)."
        }
    }
]
