# Matplotlib Roadmap / Syllabus

---

## 1. Introduction to Matplotlib

### 1.1 What is Matplotlib?
- Definition of Matplotlib
- Why Matplotlib is important
- Matplotlib vs other visualization libraries (Seaborn, Plotly, Bokeh)
- Advantages of Matplotlib
  - Full control over every plot element
  - Highly customizable
  - Works with NumPy and Pandas natively
  - Publication-quality figures
  - Supports interactive and static plots
- Real-world applications
  - Data Science
  - Machine Learning result visualization
  - Scientific computing
  - Business reporting
  - Academic publishing

### 1.2 Installing Matplotlib
- Installing with pip
- Installing with conda
- Verifying installation
- Importing Matplotlib
  - `import matplotlib.pyplot as plt`
  - `import matplotlib as mpl`

### 1.3 Matplotlib Architecture
- Figure object
- Axes object
- Artist hierarchy
- Backend system
  - Interactive backends
  - Non-interactive backends
- pyplot vs object-oriented API

---

## 2. Matplotlib Basics

### 2.1 Creating Your First Plot
- `plt.plot()` — basic line plot
- `plt.show()` — displaying the plot
- `plt.savefig()` — saving to file
- Inline display in Jupyter — `%matplotlib inline`
- Interactive display — `%matplotlib widget`

### 2.2 Figure and Axes
- `plt.figure()` — creating a figure
- `fig.add_axes()` — manually adding axes
- `plt.subplots()` — figure with axes grid
- `fig, ax = plt.subplots()` — unpacking
- Figure size — `figsize=(width, height)`
- Figure DPI — `dpi=100`

### 2.3 pyplot vs Object-Oriented API
- `plt.plot()` — pyplot interface
- `ax.plot()` — OO interface
- When to use each
- Recommended approach for complex figures

---

## 3. Basic Plot Types

### 3.1 Line Plot
- `plt.plot(x, y)` — basic line
- Multiple lines on one plot
- Line styles — `linestyle` or `ls`
  - `'-'` solid
  - `'--'` dashed
  - `':'` dotted
  - `'-.'` dash-dot
- Line width — `linewidth` or `lw`
- Line color — `color` or `c`
- Markers — `marker`
  - `'o'` circle
  - `'s'` square
  - `'^'` triangle
  - `'*'` star
  - `'+'` plus
- Marker size — `markersize` or `ms`

### 3.2 Scatter Plot
- `plt.scatter(x, y)` — basic scatter
- Color by value — `c=array`
- Size by value — `s=array`
- Colormap — `cmap`
- Alpha transparency — `alpha`
- Edge color — `edgecolors`

### 3.3 Bar Chart
- `plt.bar(x, height)` — vertical bar chart
- `plt.barh(y, width)` — horizontal bar chart
- Bar color — `color`
- Bar width — `width`
- Bar alignment — `align`
- Error bars — `yerr`
- Grouped bar charts
- Stacked bar charts

### 3.4 Histogram
- `plt.hist(data)` — basic histogram
- Number of bins — `bins`
- Density normalization — `density=True`
- Histogram type — `histtype`
  - `'bar'`
  - `'step'`
  - `'stepfilled'`
- Cumulative histogram — `cumulative=True`
- Overlapping histograms with alpha

### 3.5 Pie Chart
- `plt.pie(sizes)` — basic pie chart
- Labels — `labels`
- Explode slices — `explode`
- Autopct — `autopct='%1.1f%%'`
- Start angle — `startangle`
- Shadow — `shadow=True`
- Custom colors

### 3.6 Area Plot
- `plt.fill_between(x, y1, y2)` — area between curves
- `plt.fill_between(x, y)` — area under curve
- Alpha for transparency
- Step area plot

---

## 4. Plot Customization

### 4.1 Titles and Labels
- `plt.title()` — plot title
- `plt.xlabel()` — x-axis label
- `plt.ylabel()` — y-axis label
- Font size — `fontsize`
- Font weight — `fontweight`
- Title padding — `pad`
- Label padding — `labelpad`

### 4.2 Legends
- `plt.legend()` — automatic legend
- `label` parameter in plot functions
- Legend location — `loc`
  - `'upper right'`
  - `'lower left'`
  - `'best'`
  - `'outside'` with `bbox_to_anchor`
- Legend frame — `frameon`
- Legend columns — `ncol`
- Legend font size — `fontsize`
- Custom legend handles

### 4.3 Axis Limits and Ticks
- `plt.xlim()` — x-axis limits
- `plt.ylim()` — y-axis limits
- `ax.set_xlim()` and `ax.set_ylim()`
- `plt.xticks()` — custom x-axis tick values
- `plt.yticks()` — custom y-axis tick values
- Tick labels
- Tick rotation — `rotation`
- Major and minor ticks
- `ax.tick_params()` — tick customization
- `ax.xaxis.set_major_locator()`
- `ax.xaxis.set_major_formatter()`

### 4.4 Grid Lines
- `plt.grid(True)` — enabling grid
- Grid axis — `axis='x'`, `'y'`, `'both'`
- Grid line style — `linestyle`
- Grid line color — `color`
- Grid line width — `linewidth`
- Grid alpha — `alpha`
- Major and minor grid lines

### 4.5 Colors and Colormaps
- Named colors — `'red'`, `'blue'`, `'green'`
- Hex colors — `'#FF5733'`
- RGB tuples — `(0.2, 0.4, 0.8)`
- RGBA tuples — `(0.2, 0.4, 0.8, 0.5)`
- Colormaps — `cmap`
  - Sequential: `'viridis'`, `'plasma'`, `'Blues'`
  - Diverging: `'RdBu'`, `'coolwarm'`
  - Qualitative: `'tab10'`, `'Set1'`
- `plt.colorbar()` — adding colorbar

### 4.6 Text and Annotations
- `plt.text(x, y, 'text')` — adding text
- `plt.annotate()` — annotating data points
  - `xy` — point to annotate
  - `xytext` — text position
  - `arrowprops` — arrow style
- `ax.set_title()` — axes title
- Math text with LaTeX — `r'$\alpha$'`
- Font properties — `fontdict`

---

## 5. Subplots

### 5.1 Basic Subplots
- `plt.subplot(nrows, ncols, index)` — single subplot
- `plt.subplots(nrows, ncols)` — grid of subplots
- Iterating over axes array
- `plt.tight_layout()` — fixing overlap

### 5.2 Subplot Customization
- `figsize` — overall figure size
- `sharex` — shared x-axis
- `sharey` — shared y-axis
- Titles per subplot
- Labels per subplot

### 5.3 GridSpec
- `matplotlib.gridspec.GridSpec` — custom grid
- Spanning multiple rows and columns
- Irregular subplot layouts
- Nested GridSpec

### 5.4 Inset Axes
- `ax.inset_axes()` — plot within a plot
- Zoomed inset
- Adding indicator lines between inset and main plot

---

## 6. Styles and Themes

### 6.1 Built-in Styles
- `plt.style.use('style_name')`
- Available styles
  - `'ggplot'`
  - `'seaborn'`
  - `'bmh'`
  - `'fivethirtyeight'`
  - `'dark_background'`
  - `'classic'`
  - `'Solarize_Light2'`
- `plt.style.available` — list all styles
- Context manager — `with plt.style.context('style')`

### 6.2 rcParams
- `plt.rcParams` — global configuration dictionary
- Setting figure size globally
- Setting font size globally
- Setting line width globally
- `mpl.rc()` — shorthand
- `mpl.rcdefaults()` — reset to defaults
- `plt.rcdefaults()` — reset

### 6.3 Custom Styles
- Creating a `.mplstyle` file
- Defining custom rcParams
- Loading custom style

---

## 7. Advanced Plot Types

### 7.1 Box Plot
- `plt.boxplot(data)` — single box plot
- Multiple box plots
- Notched box plot — `notch=True`
- Horizontal box plot — `vert=False`
- Whisker range — `whis`
- Outlier marker style
- Custom patch artist

### 7.2 Violin Plot
- `plt.violinplot(data)` — basic violin
- Showing median — `showmedians=True`
- Showing means — `showmeans=True`
- Horizontal violin — `vert=False`
- Custom positions

### 7.3 Heatmap (imshow)
- `plt.imshow(matrix)` — displaying 2D arrays
- Colormap — `cmap`
- Interpolation — `interpolation`
- Aspect ratio — `aspect`
- `plt.colorbar()` — color scale
- Adding text annotations to cells

### 7.4 Contour Plot
- `plt.contour(X, Y, Z)` — contour lines
- `plt.contourf(X, Y, Z)` — filled contours
- Number of levels — `levels`
- Colormap
- Adding contour labels — `plt.clabel()`

### 7.5 3D Plots
- `from mpl_toolkits.mplot3d import Axes3D`
- `fig.add_subplot(projection='3d')`
- `ax.plot3D()` — 3D line
- `ax.scatter3D()` — 3D scatter
- `ax.plot_surface()` — surface plot
- `ax.plot_wireframe()` — wireframe
- Viewing angle — `ax.view_init(elev, azim)`

### 7.6 Stem Plot
- `plt.stem(x, y)` — stem plot
- Useful for discrete signals

### 7.7 Step Plot
- `plt.step(x, y)` — step function plot
- `where` parameter — `'pre'`, `'post'`, `'mid'`

### 7.8 Broken Axis Plot
- Simulating axis breaks
- Using multiple axes with shared limits

---

## 8. Working with Images

### 8.1 Displaying Images
- `plt.imshow(image)` — displaying an image array
- `plt.imread('file.png')` — reading image
- Grayscale images — `cmap='gray'`
- Color images (RGB)
- Removing axis ticks — `plt.axis('off')`

### 8.2 Image Manipulation Visualization
- Displaying multiple images in a grid
- Showing original vs processed
- Displaying pixel histograms

---

## 9. Saving Figures

### 9.1 Saving Options
- `plt.savefig('filename.png')` — saving as PNG
- `plt.savefig('filename.pdf')` — saving as PDF (vector)
- `plt.savefig('filename.svg')` — saving as SVG (vector)
- `plt.savefig('filename.jpg')` — saving as JPEG
- DPI — `dpi=300` for high resolution
- Bounding box — `bbox_inches='tight'`
- Background color — `facecolor`
- Transparency — `transparent=True`

### 9.2 Saving to Memory Buffer
- `io.BytesIO()` — saving to buffer
- Useful for web applications
- Base64 encoding for HTML embedding

---

## 10. Interactive Plots

### 10.1 Interactive Features
- `%matplotlib widget` — interactive in Jupyter
- Pan and zoom
- Mouse event handling
- `plt.ginput()` — capturing mouse clicks

### 10.2 Event Handling
- `fig.canvas.mpl_connect()` — connecting events
- `button_press_event`
- `key_press_event`
- `motion_notify_event`

### 10.3 Animations
- `matplotlib.animation.FuncAnimation`
- `matplotlib.animation.ArtistAnimation`
- Saving animations as GIF or MP4
- Animation interval and repeat

---

## 11. Matplotlib with NumPy

### 11.1 Plotting NumPy Arrays
- Plotting 1D arrays as line plots
- Plotting 2D arrays as heatmaps
- `np.linspace()` for x-axis generation
- `np.meshgrid()` for contour plots

### 11.2 Mathematical Functions
- Plotting `sin`, `cos`, `tan`
- Plotting exponential and logarithmic functions
- Plotting polynomial curves

---

## 12. Matplotlib with Pandas

### 12.1 DataFrame Plotting
- `df.plot()` — default line plot
- `df.plot.bar()` — bar chart
- `df.plot.hist()` — histogram
- `df.plot.scatter()` — scatter plot
- `df.plot.box()` — box plot
- `df.plot.pie()` — pie chart
- `df.plot.area()` — area chart

### 12.2 Series Plotting
- `series.plot()` — line plot
- `series.plot.bar()` — bar chart
- `series.value_counts().plot.bar()` — frequency plot

---

## 13. Layouts and Spacing

### 13.1 Figure Layout
- `plt.tight_layout()` — auto-adjust spacing
- `plt.subplots_adjust()` — manual spacing
  - `left`, `right`, `top`, `bottom`
  - `wspace`, `hspace`
- `fig.set_constrained_layout(True)` — constrained layout

### 13.2 Aspect Ratio
- `ax.set_aspect('equal')` — equal aspect ratio
- `ax.set_aspect(float)` — custom ratio

---

## 14. Matplotlib Best Practices

### 14.1 Efficient Plotting Code
- Use object-oriented API for complex figures
- Avoid modifying `plt.rcParams` inside loops
- Reuse figure objects when possible
- Close figures after saving — `plt.close()`

### 14.2 Readability
- Always label axes and add titles
- Use consistent colors for categories
- Use appropriate font sizes for publication
- Avoid cluttered legends

### 14.3 Common Mistakes
- Forgetting `plt.show()` in scripts
- Overwriting plots without clearing — `plt.clf()`
- Using wrong axes object in subplots
- Incorrect aspect ratio for image display
- Low DPI for saved figures

---

## 15. Matplotlib Interview Topics

### 15.1 Frequently Asked Questions
- Difference between `plt.plot()` and `ax.plot()`
- How to create subplots?
- How to add a legend?
- Difference between `plt.show()` and `plt.savefig()`
- How to change figure size?
- How to display inline in Jupyter?

### 15.2 Coding Problems
- Plot a sine and cosine wave on the same axes
- Create a 2x2 subplot grid
- Plot a bar chart with custom colors
- Display a confusion matrix as a heatmap
- Annotate the maximum point on a line plot

---

## 16. Real-World Matplotlib Projects

### 16.1 Beginner Projects
- Monthly sales bar chart
- Temperature line plot with annotations
- Student grade distribution histogram

### 16.2 Intermediate Projects
- Multi-subplot dashboard
- Stock price trend visualization
- Correlation heatmap for a dataset

### 16.3 Advanced Projects
- Animated data visualization
- Interactive data exploration figure
- Publication-ready multi-panel scientific figure

---

## 17. Matplotlib Ecosystem

### 17.1 Related Libraries
- Seaborn — statistical visualization built on Matplotlib
- Plotly — interactive web-based charts
- Bokeh — web-browser interactive plots
- Altair — declarative visualization
- Pandas plotting — built on Matplotlib

### 17.2 Matplotlib Extensions
- `mpl_toolkits` — 3D plots, axes grid
- `matplotlib.patches` — shapes and patches
- `matplotlib.colors` — custom color mapping
- `matplotlib.ticker` — tick formatting
- `matplotlib.dates` — date axis formatting

---

## 18. Testing and Debugging Matplotlib Code

### 18.1 Debugging Plots
- Checking figure and axes objects
- Verifying data before plotting
- Using `print(ax.get_lines())` to inspect lines
- Using `print(fig.get_axes())` to inspect axes

### 18.2 Testing
- `matplotlib.testing` — image comparison
- `plt.savefig()` + pixel comparison
- Non-interactive backend for testing — `matplotlib.use('Agg')`
