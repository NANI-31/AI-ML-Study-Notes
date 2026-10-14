# Questions 76-100: Expert / Advanced Level

questions_part4 = [
    {
        "num": 76,
        "difficulty": "Expert / Advanced",
        "topic": "Matplotlib fundamentals and architecture",
        "type": "single",
        "question": "Matplotlib's internal software architecture is fundamentally structured into three conceptual layers. What are these three layers from lowest to highest level of abstraction?",
        "options": {
            "A": "Backend Layer -> Artist Layer -> Scripting Layer (`pyplot`)",
            "B": "Scripting Layer -> Rendering Engine -> Kernel Layer",
            "C": "OpenGL Shader Layer -> Data Matrix Layer -> Canvas Layer",
            "D": "Vector Layer -> Raster Layer -> GUI Layer"
        },
        "correct": "A",
        "correct_explanation": "Matplotlib's classic three-layer architecture comprises: (1) **Backend layer** (draws primitives to output devices/canvases), (2) **Artist layer** (represents all visual graph objects and layout hierarchy), and (3) **Scripting layer** (`matplotlib.pyplot`, providing the procedural user interface).",
        "incorrect_explanations": {
            "B": "Matplotlib does not define an internal 'Kernel Layer'.",
            "C": "Matplotlib's core architecture does not use an OpenGL Shader Layer.",
            "D": "Vector and Raster are file format categories handled within backends, not the structural architectural layers."
        }
    },
    {
        "num": 77,
        "difficulty": "Expert / Advanced",
        "topic": "Figure, Axes, Axis, and related objects",
        "type": "multi",
        "question": "In Matplotlib's Artist hierarchy, Artists are categorized into *Primitives* (geometric/text elements) and *Containers* (objects that hold other artists). Which of the following classifications are correct?",
        "options": {
            "A": "`Figure`, `Axes`, and `Axis` are Container Artists.",
            "B": "`Line2D`, `Rectangle`, `Polygon`, and `Text` are Primitive Artists.",
            "C": "`Tick` is a Container Artist that contains tick lines and tick label `Text` instances.",
            "D": "`Figure` is a Primitive Artist that cannot contain other artists."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`Figure` (holds Axes, texts), `Axes` (holds lines, patches, texts, axes), and `Axis` (holds ticks, labels, spines) are Container Artists.",
            "B": "`Line2D`, `Rectangle`, `Polygon`, and `Text` represent the actual graphical drawing primitives.",
            "C": "`Tick` is a container holding `Line2D` tick marks and `Text` tick label objects."
        },
        "incorrect_explanations": {
            "D": "`Figure` is the top-level Container Artist, not a primitive."
        }
    },
    {
        "num": 78,
        "difficulty": "Expert / Advanced",
        "topic": "Backends",
        "type": "single",
        "question": "Which of the following backend types is a non-interactive (headless) raster graphics backend that renders images entirely in memory using the Anti-Grain Geometry C++ library without requiring an active X-server or GUI display?",
        "options": {
            "A": "`Agg`",
            "B": "`TkAgg`",
            "C": "`QtAgg`",
            "D": "`macosx`"
        },
        "correct": "A",
        "correct_explanation": "`Agg` uses the Anti-Grain Geometry engine to render high-quality raster pixels headlessly in memory (used by default on web servers and automated batch pipelines).",
        "incorrect_explanations": {
            "B": "`TkAgg` is an interactive GUI backend embedding the Agg rasterizer inside a Tkinter window.",
            "C": "`QtAgg` is an interactive GUI backend embedding the Agg rasterizer inside a PyQt/PySide window.",
            "D": "`macosx` is an interactive GUI backend specific to macOS Cocoa windows."
        }
    },
    {
        "num": 79,
        "difficulty": "Expert / Advanced",
        "topic": "Interactive plotting",
        "type": "multi",
        "question": "Which of the following functions manage Matplotlib's interactive mode and GUI event loop processing?",
        "options": {
            "A": "`plt.ion()` enables interactive mode so plots update and draw immediately following plotting commands without blocking.",
            "B": "`plt.ioff()` disables interactive mode, requiring explicit calls to `plt.show()` or `fig.canvas.draw()`.",
            "C": "`plt.pause(interval)` runs the GUI event loop for `interval` seconds, updating figures and processing UI events smoothly.",
            "D": "`time.sleep()` is completely identical to `plt.pause()` and handles GUI window event polling automatically."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`plt.ion()` turns interactive mode ON, triggering auto-redraws after each modification.",
            "B": "`plt.ioff()` turns interactive mode OFF for batch script execution.",
            "C": "`plt.pause()` flushes drawing queues and processes GUI window callbacks while pausing execution."
        },
        "incorrect_explanations": {
            "D": "`time.sleep()` completely freezes the Python thread and does NOT process GUI events, causing windows to become unresponsive / white-screened; `plt.pause()` must be used instead."
        }
    },
    {
        "num": 80,
        "difficulty": "Expert / Advanced",
        "topic": "Event handling",
        "type": "single",
        "question": "How do you connect a custom event handler callback function `on_click` to mouse click events on a figure canvas?",
        "options": {
            "A": "`cid = fig.canvas.mpl_connect('button_press_event', on_click)`",
            "B": "`fig.bind_event('click', on_click)`",
            "C": "`fig.canvas.add_listener('mouse_down', on_click)`",
            "D": "`ax.register_callback('on_click', on_click)`"
        },
        "correct": "A",
        "correct_explanation": "`fig.canvas.mpl_connect(event_name, callback)` registers the callback with Matplotlib's event handling system and returns a connection ID (`cid`) for subsequent disconnection via `fig.canvas.mpl_disconnect(cid)`.",
        "incorrect_explanations": {
            "B": "`bind_event()` is not a Matplotlib method.",
            "C": "`add_listener()` does not exist in the Matplotlib FigureCanvas API.",
            "D": "`register_callback()` on Axes is not the event listener registration method."
        }
    },
    {
        "num": 81,
        "difficulty": "Expert / Advanced",
        "topic": "Event handling",
        "type": "multi",
        "question": "When a callback function is triggered by `mpl_connect('button_press_event', callback)`, which attributes are available on the received `MouseEvent` object `event`?",
        "options": {
            "A": "`event.xdata` and `event.ydata` representing the coordinate position in data space (or `None` if clicked outside an Axes)",
            "B": "`event.x` and `event.y` representing the coordinate position in display pixel space",
            "C": "`event.inaxes` pointing to the specific `Axes` object containing the mouse click (or `None` if outside)",
            "D": "`event.button` indicating which mouse button was clicked (1=left, 2=middle, 3=right)",
            "E": "`event.gpu_temperature` reporting real-time hardware thermals"
        },
        "correct": ["A", "B", "C", "D"],
        "correct_explanations": {
            "A": "`xdata` and `ydata` map pixel coordinates back into the data coordinate system of the active axes.",
            "B": "`x` and `y` give the raw display pixel coordinates relative to the bottom-left of the canvas.",
            "C": "`inaxes` provides a direct reference to the clicked `Axes` instance.",
            "D": "`button` identifies the mouse button (1=Left, 2=Middle, 3=Right)."
        },
        "incorrect_explanations": {
            "E": "Hardware telemetry attributes like `gpu_temperature` do not exist on `MouseEvent`."
        }
    },
    {
        "num": 82,
        "difficulty": "Expert / Advanced",
        "topic": "Interactive plotting",
        "type": "single",
        "question": "What is the primary difference between `matplotlib.animation.FuncAnimation` and `matplotlib.animation.ArtistAnimation`?",
        "options": {
            "A": "`FuncAnimation` generates frames dynamically by calling a user-supplied update function on each tick, whereas `ArtistAnimation` replays a pre-computed list of existing artist collections",
            "B": "`FuncAnimation` renders 3D animations only, whereas `ArtistAnimation` renders 2D plots only",
            "C": "`ArtistAnimation` runs on the GPU, whereas `FuncAnimation` runs on the CPU",
            "D": "`FuncAnimation` can only export to MP4, whereas `ArtistAnimation` can only export to HTML"
        },
        "correct": "A",
        "correct_explanation": "`FuncAnimation` executes a frame generator function iteratively (ideal for dynamic simulations and unbounded streams), while `ArtistAnimation` cycles through a pre-generated sequence of artist frames.",
        "incorrect_explanations": {
            "B": "Both classes support both 2D and 3D plotting.",
            "C": "Both classes execute on the CPU within the Python/Matplotlib rendering pipeline.",
            "D": "Both animation classes support all configured animation writers (FFMpegWriter, PillowWriter, HTML5/JSAnimation)."
        }
    },
    {
        "num": 83,
        "difficulty": "Expert / Advanced",
        "topic": "Performance optimization",
        "type": "single",
        "question": "What is the primary operational mechanism of 'blitting' (`blit=True`) in Matplotlib animations?",
        "options": {
            "A": "It caches the static background of the figure (axes, spines, tick labels) as a pixel bitmap and on each frame only redraws the specific animated artists that changed, copying them directly onto the cached background",
            "B": "It offloads rasterization from the CPU to WebGL inside a browser canvas",
            "C": "It renders only every second frame to cut memory consumption in half",
            "D": "It permanently removes axis labels and spines to reduce draw call complexity"
        },
        "correct": "A",
        "correct_explanation": "Blitting captures the static background canvas buffer via `canvas.copy_from_bbox()` and then on each frame restores this background before drawing and blitting only the updated dynamic artists via `canvas.blit()`, avoiding full canvas redraw overhead.",
        "incorrect_explanations": {
            "B": "Blitting is a 2D memory pixel copy mechanism inside Matplotlib's backend; it does not involve WebGL offloading.",
            "C": "Blitting renders all requested animation frames without dropping frames.",
            "D": "Blitting preserves all background decorations without removing them."
        }
    },
    {
        "num": 84,
        "difficulty": "Expert / Advanced",
        "topic": "Performance optimization",
        "type": "single",
        "question": "When visualizing tens of thousands of independent line segments (e.g. in network graphs, financial candlesticks, or particle trails), what is the most performant Matplotlib artist to use instead of calling `ax.plot()` in a loop?",
        "options": {
            "A": "`matplotlib.collections.LineCollection`",
            "B": "Calling `ax.plot()` 50,000 times inside a `for` loop",
            "C": "Repeatedly calling `plt.draw()` after each line point",
            "D": "`matplotlib.patches.IndividualLineLoop`"
        },
        "correct": "A",
        "correct_explanation": "`LineCollection` groups thousands of line segments into a single C-level artist/batch draw call, avoiding the massive memory overhead and traversal cost of creating individual `Line2D` artist objects.",
        "incorrect_explanations": {
            "B": "Creating 50,000 individual `Line2D` objects creates severe Python overhead and slows canvas redraws to a crawl.",
            "C": "Triggering redraws after each point is extremely inefficient.",
            "D": "`IndividualLineLoop` does not exist in Matplotlib."
        }
    },
    {
        "num": 85,
        "difficulty": "Expert / Advanced",
        "topic": "Performance optimization",
        "type": "multi",
        "question": "Which of the following techniques optimize Matplotlib rendering performance for very large datasets?",
        "options": {
            "A": "Updating existing line data via `line.set_data(x, y)` and `line.set_3d_properties()` instead of re-instantiating `ax.plot()`",
            "B": "Setting `rasterized=True` on complex collections (like scatter plots with 100,000 points) when exporting to vector formats (PDF/SVG)",
            "C": "Enabling path simplification using `matplotlib.rcParams['path.simplify'] = True`",
            "D": "Disabling figure rendering completely with `fig.set_dpi(-100)`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Modifying existing `Line2D` vertex buffers via `set_data()` avoids object instantiation and garbage collection overhead.",
            "B": "`rasterized=True` rasterizes heavy collections into a bitmap embedded inside vector PDFs/SVGs, preventing bloated multi-gigabyte vector files.",
            "C": "`path.simplify` applies the Ramer-Douglas-Peucker algorithm to remove redundant sub-pixel vertices prior to rendering."
        },
        "incorrect_explanations": {
            "D": "Negative DPI values are invalid and raise an error."
        }
    },
    {
        "num": 86,
        "difficulty": "Expert / Advanced",
        "topic": "Custom projections",
        "type": "single",
        "question": "To implement a custom projection in Matplotlib (such as an Aitoff map projection or custom hyperbolic geometry), which core base class must be subclassed?",
        "options": {
            "A": "`matplotlib.axes.Axes`",
            "B": "`matplotlib.transforms.BboxBase`",
            "C": "`matplotlib.figure.FigureCanvasBase`",
            "D": "`matplotlib.lines.Line2D`"
        },
        "correct": "A",
        "correct_explanation": "Custom projections are implemented by subclassing `matplotlib.axes.Axes` (or `matplotlib.projections.geo.GeoAxes`), overriding its transformation pipeline (`_init_axis()`, `_set_lim_and_transforms()`), and defining the projection `name` attribute.",
        "incorrect_explanations": {
            "B": "`BboxBase` represents bounding boxes, not coordinate projection systems.",
            "C": "`FigureCanvasBase` manages canvas device rendering rather than coordinate projections.",
            "D": "`Line2D` is a 1D line primitive artist."
        }
    },
    {
        "num": 87,
        "difficulty": "Expert / Advanced",
        "topic": "Custom projections",
        "type": "multi",
        "question": "Which steps are involved in creating and registering a custom projection class `MyCustomAxes` in Matplotlib?",
        "options": {
            "A": "Define a unique `name` class attribute (e.g. `name = 'my_custom_proj'`) inside the subclass.",
            "B": "Register the subclass using `matplotlib.projections.register_projection(MyCustomAxes)`.",
            "C": "Instantiate subplots using `plt.subplot(projection='my_custom_proj')` or `plt.subplots(subplot_kw={'projection': 'my_custom_proj'})`.",
            "D": "Recompile the Python C-extension binaries from source."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "The `name` attribute provides the string lookup key for the projection registry.",
            "B": "`register_projection()` registers the class in Matplotlib's global projection dictionary.",
            "C": "Once registered, the custom projection string can be passed directly to `projection=...` in figure/subplot factory functions."
        },
        "incorrect_explanations": {
            "D": "Custom projections are purely pure-Python/NumPy transformations and do not require recompiling C-extensions."
        }
    },
    {
        "num": 88,
        "difficulty": "Expert / Advanced",
        "topic": "Coordinate systems and transforms",
        "type": "single",
        "question": "How do you convert display pixel coordinates `(px, py)` back into data coordinates `(x, y)` on an `Axes` object `ax`?",
        "options": {
            "A": "`data_x, data_y = ax.transData.inverted().transform((px, py))`",
            "B": "`data_x, data_y = ax.transData.reverse((px, py))`",
            "C": "`data_x, data_y = ax.transData.unproject((px, py))`",
            "D": "`data_x, data_y = ax.pixel_to_data((px, py))`"
        },
        "correct": "A",
        "correct_explanation": "Transform objects in Matplotlib implement `.inverted()`, which returns the inverse transform node. Calling `.transform()` on `ax.transData.inverted()` maps display pixels back to data units.",
        "incorrect_explanations": {
            "B": "`reverse()` is not a method on Transform objects.",
            "C": "`unproject()` is not a standard Matplotlib method.",
            "D": "`pixel_to_data()` is not a method on `Axes`."
        }
    },
    {
        "num": 89,
        "difficulty": "Expert / Advanced",
        "topic": "Coordinate systems and transforms",
        "type": "multi",
        "question": "What is `matplotlib.transforms.blended_transform_factory(x_transform, y_transform)` used for?",
        "options": {
            "A": "Creating a composite 2D transform where the X-coordinate uses one transformation and the Y-coordinate uses a different transformation.",
            "B": "Enabling plots where the X-axis is in data units (`ax.transData`) but the Y-axis spans normalized axes units (`ax.transAxes`) from 0 to 1 (such as in `ax.axvline()`).",
            "C": "Blending transparency gradients between two overlapping images.",
            "D": "Constructing specialized span and shading transformations."
        },
        "correct": ["A", "B", "D"],
        "correct_explanations": {
            "A": "`blended_transform_factory` combines separate 1D transforms along the X and Y axes into a unified 2D `Transform`.",
            "B": "This allows spanning lines like `axvline()` and `axvspan()` to lock X to data values while Y spans the full axes height $(0 \\text{ to } 1)$.",
            "D": "It forms the foundation of all hybrid span, bar, and annotation transformations in Matplotlib."
        },
        "incorrect_explanations": {
            "C": "Blended transforms operate purely on geometric coordinates, not image pixel color/alpha blending."
        }
    },
    {
        "num": 90,
        "difficulty": "Expert / Advanced",
        "topic": "Figure and Axes management",
        "type": "single",
        "question": "In a long-running batch script or web server (e.g. FastAPI/Flask) that generates hundreds of figures in a loop, why does memory consumption continually increase if figures are not explicitly closed?",
        "options": {
            "A": "`pyplot` maintains internal global references to every created `Figure` in the `_pylab_helpers.Gcf` manager, preventing Python's garbage collector from reclaiming memory until `plt.close(fig)` is called.",
            "B": "NumPy arrays retain GPU memory locks permanently.",
            "C": "The operating system refuses to release file handles after `plt.savefig()`.",
            "D": "Matplotlib saves backup copies of all figures to a hidden disk partition."
        },
        "correct": "A",
        "correct_explanation": "`pyplot` acts as a state manager holding strong references to all open figures inside `Gcf.figs`. Without calling `plt.close(fig)` or `plt.close('all')`, figures remain referenced in memory, causing memory leaks.",
        "incorrect_explanations": {
            "B": "Standard Matplotlib rendering is CPU-bound and does not lock GPU memory.",
            "C": "File handles are closed automatically after `savefig()`; the memory leak occurs in RAM inside the Python process.",
            "D": "Matplotlib does not write hidden copies to disk."
        }
    },
    {
        "num": 91,
        "difficulty": "Expert / Advanced",
        "topic": "Figure and Axes management",
        "type": "multi",
        "question": "What are the distinctions between `plt.close()`, `fig.clf()`, and `ax.cla()`?",
        "options": {
            "A": "`ax.cla()` (or `ax.clear()`) clears all artists, lines, and text from a specific `Axes`, preserving the `Axes` object itself.",
            "B": "`fig.clf()` (or `fig.clear()`) clears all subplots and artists from the `Figure`, keeping the empty `Figure` window intact.",
            "C": "`plt.close(fig)` destroys the figure window, removes it from the `pyplot` manager, and releases its GUI/canvas resources.",
            "D": "`ax.cla()` deletes the entire Python virtual environment."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`cla()` stands for 'Clear Axes' and resets the contents of a single subplot.",
            "B": "`clf()` stands for 'Clear Figure' and strips all axes and decorations from the figure.",
            "C": "`plt.close()` completely closes and dereferences the figure."
        },
        "incorrect_explanations": {
            "D": "`ax.cla()` is a plot-clearing method and has no effect on Python environments."
        }
    },
    {
        "num": 92,
        "difficulty": "Expert / Advanced",
        "topic": "Backends",
        "type": "single",
        "question": "What is the key distinction between `fig.canvas.draw()` and `fig.canvas.draw_idle()`?",
        "options": {
            "A": "`draw()` forces an immediate, synchronous redraw of the entire canvas, whereas `draw_idle()` requests a redraw as soon as the GUI control returns to the main event loop (coalescing multiple requests into a single paint event)",
            "B": "`draw_idle()` puts the CPU into low-power sleep mode",
            "C": "`draw()` renders in 2D while `draw_idle()` renders in 3D",
            "D": "`draw_idle()` renders only when the user is not touching the keyboard or mouse for 10 seconds"
        },
        "correct": "A",
        "correct_explanation": "`draw_idle()` batches and debounces paint requests, preventing redundant rendering when multiple artists or limits are updated in rapid succession before the GUI redraws.",
        "incorrect_explanations": {
            "B": "`draw_idle()` does not manage hardware power states.",
            "C": "Both methods support identical rendering capabilities.",
            "D": "`draw_idle()` fires at the next iteration of the event loop, not after user inactivity."
        }
    },
    {
        "num": 93,
        "difficulty": "Expert / Advanced",
        "topic": "Backends",
        "type": "multi",
        "question": "When using Matplotlib in a multi-threaded web server (such as Django, Flask, or FastAPI workers), which best practices should be followed to prevent race conditions and memory leaks?",
        "options": {
            "A": "Set the non-interactive backend `matplotlib.use('Agg')` before importing `pyplot`.",
            "B": "Avoid relying on global `pyplot` state functions (e.g. `plt.plot()`, `plt.gca()`) and instead instantiate figures explicitly via `Figure()` and `FigureCanvasAgg` or ensure thread isolation.",
            "C": "Always explicitly call `plt.close(fig)` or destroy figure instances inside a `finally:` block.",
            "D": "Share a single global `Figure` object across all concurrent web requests."
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "Configuring the headless `'Agg'` backend prevents attempts to initialize GUI toolkits in non-GUI worker threads.",
            "B": "Global `pyplot` state is not thread-safe; concurrent threads mutating `plt.gca()` can cross-contaminate plots.",
            "C": "Ensuring `close()` inside `finally:` guarantees memory reclamation even if exceptions occur during request handling."
        },
        "incorrect_explanations": {
            "D": "Sharing a single Figure across concurrent requests will cause catastrophic race conditions and corrupted plot outputs."
        }
    },
    {
        "num": 94,
        "difficulty": "Expert / Advanced",
        "topic": "Coordinate systems and transforms",
        "type": "single",
        "question": "Which helper in `matplotlib.transforms` creates a transformation that adds a fixed physical offset in display points (e.g. 5 points to the right and 5 points up) to an existing coordinate transformation?",
        "options": {
            "A": "`matplotlib.transforms.offset_copy(trans, fig=fig, x=5, y=5, units='points')`",
            "B": "`matplotlib.transforms.add_points(trans, 5, 5)`",
            "C": "`matplotlib.transforms.shift_pixels(trans, 5, 5)`",
            "D": "`matplotlib.transforms.displace_vector(trans, 5, 5)`"
        },
        "correct": "A",
        "correct_explanation": "`offset_copy()` produces a new transformation that takes coordinates, maps them through the base transform, and applies a physical offset specified in `'points'`, `'inches'`, or `'dots'`.",
        "incorrect_explanations": {
            "B": "`add_points()` is not a Matplotlib method.",
            "C": "`shift_pixels()` does not exist in Matplotlib.",
            "D": "`displace_vector()` does not exist in Matplotlib."
        }
    },
    {
        "num": 95,
        "difficulty": "Expert / Advanced",
        "topic": "Tick formatting",
        "type": "multi",
        "question": "Which of the following built-in ticker classes in `matplotlib.ticker` handle specialized scientific and engineering notation?",
        "options": {
            "A": "`ScalarFormatter(useMathText=True)` to render scientific exponent multipliers (e.g. $\\times 10^5$) in mathematical LaTeX-like formatting",
            "B": "`EngFormatter(unit='V')` to format values with engineering SI prefixes (e.g., $1.5\\text{ kV}$, $200\\text{ mV}$)",
            "C": "`LogFormatterMathtext` to format logarithmic tick labels cleanly as powers of the base (e.g. $10^{-3}, 10^0, 10^3$)",
            "D": "`BinaryQuantumFormatter` to format numbers into quantum superposition states"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`ScalarFormatter(useMathText=True)` generates clean mathematical typography for scale offsets.",
            "B": "`EngFormatter` automatically scales values to powers of 1000 and appends SI prefixes ($k, M, G, m, \\mu, n$, etc.).",
            "C": "`LogFormatterMathtext` displays clean base-power superscripts on log-scale axes."
        },
        "incorrect_explanations": {
            "D": "`BinaryQuantumFormatter` does not exist in Matplotlib."
        }
    },
    {
        "num": 96,
        "difficulty": "Expert / Advanced",
        "topic": "Advanced customization and best practices",
        "type": "single",
        "question": "Which Matplotlib module is used to add artistic drop shadows, glow effects, or contrasting stroke outlines around text and path artists?",
        "options": {
            "A": "`matplotlib.patheffects` (using classes like `withStroke` and `SimplePatchShadow`)",
            "B": "`matplotlib.photoshop`",
            "C": "`matplotlib.filters.glow`",
            "D": "`matplotlib.fx.shadows`"
        },
        "correct": "A",
        "correct_explanation": "`matplotlib.patheffects` contains path effect classes (such as `withStroke`, `Stroke`, `Normal`, and `SimplePatchShadow`) that apply stylized outline and shadow drawing passes to artists via `artist.set_path_effects([...])`.",
        "incorrect_explanations": {
            "B": "`matplotlib.photoshop` does not exist.",
            "C": "`matplotlib.filters` does not exist.",
            "D": "`matplotlib.fx` does not exist."
        }
    },
    {
        "num": 97,
        "difficulty": "Expert / Advanced",
        "topic": "Transparency and rendering",
        "type": "single",
        "question": "Which method is used to clip an artist so that it is rendered strictly inside the arbitrary 2D geometric boundary of another `Patch` (such as clipping an image inside a circle or polygon)?",
        "options": {
            "A": "`artist.set_clip_path(patch)`",
            "B": "`artist.set_stencil_mask(patch)`",
            "C": "`artist.mask_to_geometry(patch)`",
            "D": "`artist.set_clip_polygon_only(patch)`"
        },
        "correct": "A",
        "correct_explanation": "`artist.set_clip_path(patch)` (or `artist.set_clip_path(path, transform)`) constrains rendering of the artist to the interior region of the specified geometric patch/path.",
        "incorrect_explanations": {
            "B": "`set_stencil_mask()` does not exist in Matplotlib.",
            "C": "`mask_to_geometry()` is not a valid method.",
            "D": "`set_clip_polygon_only()` does not exist in Matplotlib."
        }
    },
    {
        "num": 98,
        "difficulty": "Expert / Advanced",
        "topic": "Layout management",
        "type": "single",
        "question": "To programmatically calculate the exact display bounding box (in pixel coordinates) of a rendered `Text` or `Axes` artist during a script, which method is called?",
        "options": {
            "A": "`artist.get_window_extent(renderer=fig.canvas.get_renderer())`",
            "B": "`artist.calculate_pixel_bounds()`",
            "C": "`artist.get_screen_dimensions()`",
            "D": "`artist.to_display_rect()`"
        },
        "correct": "A",
        "correct_explanation": "`artist.get_window_extent(renderer)` evaluates the exact `Bbox` of the artist in display pixel units using the canvas renderer.",
        "incorrect_explanations": {
            "B": "`calculate_pixel_bounds()` does not exist in Matplotlib.",
            "C": "`get_screen_dimensions()` is not a valid method.",
            "D": "`to_display_rect()` does not exist in Matplotlib."
        }
    },
    {
        "num": 99,
        "difficulty": "Expert / Advanced",
        "topic": "Styles and themes",
        "type": "multi",
        "question": "When Matplotlib starts up, which of the following file locations and environment variables are checked in the search hierarchy to populate `rcParams`?",
        "options": {
            "A": "Environment variable `MATPLOTLIBRC` (if pointing to a valid file path)",
            "B": "`matplotlibrc` located in the current working directory",
            "C": "`$MATPLOTLIBDATA/matplotlibrc` (the system/install-level default configuration file)",
            "D": "System registry keys stored under `HKEY_LOCAL_MACHINE\\SOFTWARE\\Matplotlib`"
        },
        "correct": ["A", "B", "C"],
        "correct_explanations": {
            "A": "`MATPLOTLIBRC` has the highest file override precedence if set to a valid file path.",
            "B": "A local `matplotlibrc` in the current working directory overrides user and system defaults.",
            "C": "The installed system library `matplotlibrc` serves as the baseline fallback."
        },
        "incorrect_explanations": {
            "D": "Matplotlib does not read configuration from the Windows OS registry; configuration is handled strictly via text `.matplotlibrc` files, `MATPLOTLIBRC`, or user config directories (`~/.config/matplotlib/matplotlibrc`)."
        }
    },
    {
        "num": 100,
        "difficulty": "Expert / Advanced",
        "topic": "Advanced customization and best practices",
        "type": "single",
        "question": "Why is the explicit Object-Oriented (OO) interface (`fig, ax = plt.subplots()`; `ax.plot()`) strongly favored over the procedural state-machine interface (`plt.plot()`) in production software and large data pipelines?",
        "options": {
            "A": "The OO interface provides explicit, thread-safe references to distinct `Figure` and `Axes` instances, avoids hidden global state bugs, enables fine-grained control over subplots, and prevents unintended side-effects across functions",
            "B": "The OO interface compiles Python code into native GPU assembly instructions",
            "C": "The procedural `pyplot` interface was deprecated and removed in Python 3",
            "D": "The OO interface only works when an internet connection is active"
        },
        "correct": "A",
        "correct_explanation": "The Object-Oriented API explicitly encapsulates state within object instances (`Figure`, `Axes`), making functions modular, testable, reusable, and free from hidden global state side-effects.",
        "incorrect_explanations": {
            "B": "Matplotlib's OO interface executes within standard Python and does not compile into GPU assembly.",
            "C": "`pyplot` is not deprecated; it remains active for interactive convenience and quick prototyping.",
            "D": "Matplotlib executes entirely locally and does not require internet connectivity."
        }
    }
]
