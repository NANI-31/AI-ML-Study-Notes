# Plotly Roadmap / Syllabus

---

## 1. Introduction to Plotly

### 1.1 What is Plotly?
- Definition of Plotly
- Plotly as an interactive visualization library
- Open-source vs Plotly Enterprise
- Plotly for Python, R, and JavaScript
- Advantages of Plotly
  - Interactive by default — zoom, pan, hover, click
  - Publication-quality charts
  - Web-embeddable HTML output
  - Wide chart type support (2D, 3D, maps, financial)
  - Integration with Dash for web apps
- Plotly vs Matplotlib — static vs interactive
- Plotly vs Seaborn — statistical vs interactive
- Plotly vs Bokeh — comparison

### 1.2 Plotly Sub-libraries
- `plotly.express` — high-level, concise API (`px`)
- `plotly.graph_objects` — low-level, full control (`go`)
- `plotly.subplots` — multi-panel figures
- `plotly.io` — input/output operations
- `plotly.colors` — colorscale utilities
- `dash` — web application framework built on Plotly

### 1.3 Installing Plotly
- Installing with pip
- Installing with conda
- Verifying installation
- Importing
  - `import plotly.express as px`
  - `import plotly.graph_objects as go`
  - `from plotly.subplots import make_subplots`

### 1.4 Rendering Modes
- Jupyter Notebook rendering — `fig.show()`
- HTML file rendering — `fig.write_html('file.html')`
- Static image — `fig.write_image('file.png')`
- Browser rendering — `fig.show(renderer='browser')`
- Google Colab — `fig.show()` auto-detects
- `pio.renderers.default` — setting default renderer

---

## 2. Plotly Express (px) — High-Level API

### 2.1 What is Plotly Express?
- High-level wrapper around `graph_objects`
- One-line chart creation
- Native Pandas DataFrame support
- Automatic legend, titles, and hover text
- `data_frame`, `x`, `y`, `color`, `size`, `symbol`, `facet_col`, `facet_row`

### 2.2 Scatter and Line Plots
- `px.scatter(df, x='col1', y='col2')` — basic scatter
- `color` — color by column
- `size` — size by column
- `symbol` — marker shape by column
- `hover_name` — label on hover
- `hover_data` — extra hover columns
- `text` — text labels on points
- `opacity` — transparency
- `trendline` — `'ols'`, `'lowess'`
- `px.line(df, x='x', y='y')` — line plot
- `px.line(df, x='x', y='y', color='group')` — multiple lines
- `line_dash` — dash style by group
- `markers` — show markers on line

### 2.3 Bar Charts
- `px.bar(df, x='cat', y='num')` — basic bar
- `color` — color by column
- `barmode` — `'group'`, `'stack'`, `'overlay'`, `'relative'`
- `orientation` — `'v'` or `'h'`
- `text_auto` — show value labels
- `error_y` — error bars

### 2.4 Histograms
- `px.histogram(df, x='col')` — basic histogram
- `nbins` — number of bins
- `color` — overlaid histograms by category
- `barmode` — `'overlay'`, `'stack'`
- `histnorm` — `'probability'`, `'density'`, `'percent'`
- `cumulative` — cumulative histogram
- `marginal` — `'rug'`, `'box'`, `'violin'`, `'histogram'`

### 2.5 Box and Violin Plots
- `px.box(df, x='cat', y='num')` — box plot
- `color` — grouped boxes
- `notched` — notched box
- `points` — show all points — `'all'`, `'outliers'`, `'suspectedoutliers'`, `False`
- `px.violin(df, x='cat', y='num')` — violin
- `box` — show box inside violin
- `points` — show data points
- `violinmode` — `'group'` or `'overlay'`

### 2.6 Pie and Donut Charts
- `px.pie(df, values='col', names='labels')` — pie chart
- `hole` — `0.4` for donut chart
- `color_discrete_sequence` — custom colors
- `hover_data` — extra hover info
- `pull` — explode slices

### 2.7 Heatmap
- `px.imshow(matrix)` — heatmap from 2D array
- `color_continuous_scale` — colorscale
- `aspect` — `'auto'`, `'equal'`
- `text_auto` — show values
- `zmin` and `zmax` — color scale limits
- `px.imshow(df)` — from DataFrame

### 2.8 Scatter Matrix (Pair Plot)
- `px.scatter_matrix(df, dimensions=['col1','col2','col3'])` — scatter matrix
- `color` — color by class
- `symbol` — marker by class
- Diagonal plots

### 2.9 Density Plots
- `px.density_contour(df, x='col1', y='col2')` — 2D density contour
- `px.density_heatmap(df, x='col1', y='col2')` — 2D density heatmap
- `marginal_x` and `marginal_y` — marginal distributions

### 2.10 Area Chart
- `px.area(df, x='x', y='y')` — area chart
- `color` — stacked areas by group
- `line_group` — grouping variable

### 2.11 Bubble Chart
- `px.scatter(df, x='x', y='y', size='col', color='group')` — bubble chart
- `size_max` — maximum bubble size

---

## 3. Graph Objects (go) — Low-Level API

### 3.1 What is graph_objects?
- Full control over every chart property
- Building complex, custom charts
- Composing traces and layouts manually
- `go.Figure()` — creating an empty figure
- `fig.add_trace()` — adding a trace
- `fig.update_layout()` — updating layout
- `fig.update_traces()` — updating traces

### 3.2 Core Traces
- `go.Scatter()` — scatter and line traces
  - `mode` — `'markers'`, `'lines'`, `'lines+markers'`, `'text'`
  - `x`, `y` — data
  - `name` — legend label
  - `marker` — marker properties
  - `line` — line properties
  - `fill` — `'tozeroy'`, `'tonexty'`
- `go.Bar()` — bar chart trace
  - `x`, `y` — data
  - `name`, `marker_color`, `orientation`
- `go.Histogram()` — histogram trace
  - `x`, `nbinsx`, `histnorm`, `cumulative_enabled`
- `go.Box()` — box plot trace
  - `y`, `name`, `boxpoints`, `notched`
- `go.Violin()` — violin trace
  - `y`, `name`, `box_visible`, `points`
- `go.Heatmap()` — heatmap trace
  - `z`, `x`, `y`, `colorscale`, `zmin`, `zmax`
- `go.Pie()` — pie chart trace
  - `values`, `labels`, `hole`, `pull`

### 3.3 Figure Layout
- `fig.update_layout()` — full layout control
  - `title` — figure title
  - `xaxis_title`, `yaxis_title` — axis labels
  - `width`, `height` — figure size
  - `showlegend` — show or hide legend
  - `legend` — legend properties
  - `font` — global font settings
  - `paper_bgcolor` — background color
  - `plot_bgcolor` — plot area color
  - `margin` — margins — `l`, `r`, `t`, `b`
  - `template` — pre-built theme

### 3.4 Axes Customization
- `xaxis` and `yaxis` in `update_layout()`
  - `title` — axis title
  - `type` — `'linear'`, `'log'`, `'date'`, `'category'`
  - `range` — axis range
  - `showgrid` — show grid
  - `gridcolor` — grid color
  - `zeroline` — show zero line
  - `tickvals` — custom tick values
  - `ticktext` — custom tick labels
  - `tickangle` — tick label rotation
  - `showticklabels` — show or hide labels

### 3.5 Marker and Line Customization
- `marker_color`, `marker_size`, `marker_symbol`
- `marker_opacity`
- `marker_line_color`, `marker_line_width`
- `line_color`, `line_width`, `line_dash`
  - `'solid'`, `'dot'`, `'dash'`, `'longdash'`, `'dashdot'`

---

## 4. Subplots

### 4.1 make_subplots
- `from plotly.subplots import make_subplots`
- `make_subplots(rows=2, cols=2)` — 2x2 grid
- `row` and `col` in `add_trace()`
- `subplot_titles` — title per panel
- `shared_xaxes` — shared x-axis
- `shared_yaxes` — shared y-axis
- `horizontal_spacing`, `vertical_spacing`

### 4.2 Mixed Chart Types in Subplots
- Scatter in row 1, bar in row 2
- Heatmap and line plot in same figure

### 4.3 Secondary Y-Axis
- `make_subplots(specs=[[{'secondary_y': True}]])`
- `fig.add_trace(trace, secondary_y=True)`
- Dual y-axis for different scales

### 4.4 Insets
- `fig.add_trace()` with `xref`, `yref` in `ax`
- Domain-based positioning

---

## 5. Plotly Templates and Themes

### 5.1 Built-in Templates
- `template='plotly'` — default
- `template='plotly_white'` — white background
- `template='plotly_dark'` — dark mode
- `template='ggplot2'` — ggplot style
- `template='seaborn'` — seaborn style
- `template='simple_white'` — minimal
- `template='none'` — no template
- `pio.templates.default` — setting global default

### 5.2 Custom Templates
- Creating a custom template object
- Setting custom colors, fonts, background
- Registering a custom template
- Applying to figures

---

## 6. Colors and Colorscales

### 6.1 Discrete Colors
- `color_discrete_sequence` — list of colors
- Built-in color sequences
  - `px.colors.qualitative.Plotly`
  - `px.colors.qualitative.Set1`
  - `px.colors.qualitative.Pastel`
- `color_discrete_map` — mapping categories to colors

### 6.2 Continuous Colorscales
- `color_continuous_scale` — colorscale name
- Built-in colorscales
  - `'Viridis'`, `'Plasma'`, `'Inferno'`, `'Magma'`
  - `'Blues'`, `'Reds'`, `'Greens'`
  - `'RdBu'`, `'Picnic'`, `'Portland'`
- `range_color` — min and max color limits
- `color_continuous_midpoint` — diverging midpoint

### 6.3 Custom Colorscales
- List of `[threshold, color]` pairs
- Linear interpolation between colors

---

## 7. Interactivity Features

### 7.1 Hover
- Default hover info
- `hovertemplate` — custom hover text
- `hoverinfo` — controlling what appears
- `hoverlabel` — hover box styling
- `customdata` — adding extra data to hover

### 7.2 Click and Selection Events
- `clickmode` — `'event'`, `'select'`, `'event+select'`
- Box select and lasso select
- Using selected data in Dash callbacks

### 7.3 Zoom and Pan
- Scroll to zoom
- Box zoom
- Drag to pan
- `dragmode` — `'zoom'`, `'pan'`, `'select'`, `'lasso'`
- `fixedrange` — disabling zoom on an axis

### 7.4 Buttons and Sliders
- `updatemenus` — dropdown menus and buttons
- `sliders` — animation and range sliders
- Play/pause button for animations

### 7.5 Animations
- `px.scatter(df, animation_frame='year')` — animated scatter
- `animation_group` — consistent object across frames
- Frame duration — `frame.duration`
- Transition duration

### 7.6 Range Slider and Selector
- Date range slider — `xaxis_rangeslider_visible=True`
- Range selector buttons — `xaxis_rangeselector`
  - `1m`, `6m`, `YTD`, `1y`, `All`

---

## 8. Advanced Plot Types

### 8.1 3D Plots
- `px.scatter_3d(df, x='x', y='y', z='z', color='cat')` — 3D scatter
- `px.line_3d(df, x='x', y='y', z='z')` — 3D line
- `go.Surface(z=Z_matrix)` — 3D surface
- `go.Mesh3d()` — 3D mesh
- Camera angle — `scene_camera`

### 8.2 Contour Plots
- `go.Contour(x=x, y=y, z=z)` — contour
- `colorscale`, `contours_coloring` — `'heatmap'`, `'lines'`
- `ncontours` — number of contour levels
- `showscale` — colorbar

### 8.3 Treemap and Sunburst
- `px.treemap(df, path=['parent', 'child'], values='col')` — treemap
- `px.sunburst(df, path=['level1', 'level2'], values='col')` — sunburst
- `color` — color by value or category
- Hierarchical data visualization

### 8.4 Funnel Chart
- `px.funnel(df, x='values', y='stages')` — funnel
- Conversion rate visualization

### 8.5 Waterfall Chart
- `go.Waterfall()` — waterfall chart
- `measure` — `'relative'`, `'total'`, `'absolute'`
- Financial data representation

### 8.6 Gantt Chart
- `px.timeline(df, x_start='start', x_end='end', y='task')` — Gantt
- `color` — color by category

### 8.7 Candlestick Chart
- `go.Candlestick(x=dates, open=o, high=h, low=l, close=c)` — candlestick
- Financial OHLC data

### 8.8 Indicator Chart
- `go.Indicator()` — KPI display
- `mode` — `'number'`, `'delta'`, `'gauge'`
- `delta_reference` — comparison value
- Gauge style — `gauge_axis_range`, `gauge_bar_color`

---

## 9. Geographic and Map Plots

### 9.1 Choropleth Map
- `px.choropleth(df, locations='country_code', color='value')` — world choropleth
- `locationmode` — `'ISO-3'`, `'USA-states'`, `'country names'`
- `color_continuous_scale` — colorscale

### 9.2 Scatter Geo
- `px.scatter_geo(df, lat='lat', lon='lon', color='cat')` — point on map
- `projection` — map projection type

### 9.3 Mapbox Scatter
- `px.scatter_mapbox(df, lat='lat', lon='lon', color='cat')` — Mapbox scatter
- `mapbox_style` — `'open-street-map'`, `'carto-positron'`
- Zoom and center settings

### 9.4 Choropleth Mapbox
- `px.choropleth_mapbox(df, geojson=gj, locations='id', color='value')` — custom regions

---

## 10. Figure Output and Export

### 10.1 Displaying Figures
- `fig.show()` — show in notebook or browser
- `renderer` parameter — `'browser'`, `'notebook'`, `'svg'`

### 10.2 Saving Figures
- `fig.write_html('output.html')` — interactive HTML
- `fig.write_image('output.png')` — PNG image
- `fig.write_image('output.pdf')` — PDF
- `fig.write_image('output.svg')` — SVG
- `fig.write_image('output.jpeg')` — JPEG
- DPI control — `scale=2` for 2x resolution
- Requires `kaleido` package for image export

### 10.3 Figure as JSON
- `fig.to_json()` — figure as JSON string
- `fig.to_dict()` — figure as Python dict
- Reconstructing from JSON — `go.Figure(json_string)`

---

## 11. Plotly with Pandas

### 11.1 DataFrames as Input
- Passing `data_frame=df` in px functions
- Column name as string — `x='column_name'`
- Automatic handling of categorical columns
- Long vs wide format data

### 11.2 Pandas Integration
- `px.data` — built-in example datasets
- Wide-form DataFrame with `px.line(df_wide)`
- Long-form DataFrame with `color='group'`

---

## 12. Plotly with NumPy

### 12.1 NumPy Arrays as Input
- `go.Scatter(x=np.array, y=np.array)` — direct NumPy input
- `go.Surface(z=np.ndarray)` — 2D array for surface
- `np.meshgrid()` for 3D surface data

---

## 13. Introduction to Dash

### 13.1 What is Dash?
- Dash as a web application framework
- Built on Plotly, Flask, and React
- Reactive callback architecture
- Deploying data apps without JavaScript

### 13.2 Basic Dash App Structure
- `app = Dash(__name__)`
- `app.layout` — defining the layout
- `@app.callback()` — reactive updates
- `app.run_server(debug=True)` — running the app

### 13.3 Dash Core Components
- `dcc.Graph()` — Plotly figure component
- `dcc.Dropdown()` — dropdown selector
- `dcc.Slider()` — slider input
- `dcc.Input()` — text input
- `dcc.Checklist()` — checkbox
- `dcc.RadioItems()` — radio buttons
- `dcc.DatePickerRange()` — date range picker

---

## 14. Plotly Best Practices

### 14.1 Effective Interactive Visualization
- Use `hovertemplate` for clean hover text
- Use `template='plotly_white'` for clean backgrounds
- Use `fig.update_layout()` for consistent styling
- Prefer `px` for standard charts and `go` for custom charts

### 14.2 Performance
- `webgl=True` for large datasets — `go.Scattergl()`
- Limit data points for smooth interaction
- Use `update_traces()` instead of recreating the figure

### 14.3 Common Mistakes
- Forgetting `fig.show()` in scripts
- Mixing `px` and `go` incorrectly
- Not installing `kaleido` for static image export
- Overcomplicating `hovertemplate` syntax
- Not setting `barmode` for grouped bars

---

## 15. Plotly Interview Topics

### 15.1 Frequently Asked Questions
- Difference between `plotly.express` and `plotly.graph_objects`
- How to save a Plotly figure as HTML?
- How to add a secondary y-axis?
- How to create animated charts?
- How to customize hover text?

### 15.2 Coding Problems
- Create an animated scatter plot by year
- Build a dual-axis line chart
- Plot a 3D surface from a NumPy array
- Create a choropleth world map
- Build a subplot grid with mixed chart types

---

## 16. Real-World Plotly Projects

### 16.1 Beginner Projects
- Interactive sales bar chart
- Animated population growth scatter
- World map of country statistics

### 16.2 Intermediate Projects
- Interactive stock price dashboard
- Multi-panel EDA dashboard with subplots
- Animated ML training progress

### 16.3 Advanced Projects
- Full Dash web app with live data updates
- Interactive financial analytics dashboard
- Multi-tab ML model evaluation app

---

## 17. Plotly Ecosystem

### 17.1 Related Libraries
- Dash — web application framework
- Kaleido — static image export
- Plotly Express — high-level API
- Pandas — data input
- NumPy — array data

### 17.2 Plotly Extensions
- `dash-bootstrap-components` — Bootstrap styling for Dash
- `dash-ag-grid` — interactive data tables
- `dash-mantine-components` — modern UI components
- `plotly-resampler` — efficient large time series
