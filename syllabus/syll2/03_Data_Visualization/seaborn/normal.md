# Seaborn Roadmap / Syllabus

---

## 1. Introduction to Seaborn

### 1.1 What is Seaborn?
- Definition of Seaborn
- Seaborn as a high-level statistical visualization library
- Built on top of Matplotlib
- Tight integration with Pandas DataFrames
- Advantages of Seaborn
  - Beautiful default aesthetics
  - Built-in statistical estimation
  - Concise API for complex plots
  - Native Pandas DataFrame support
  - Automatic handling of categorical variables
- Seaborn vs Matplotlib — when to use each
- Seaborn vs Plotly — static vs interactive

### 1.2 Installing Seaborn
- Installing with pip
- Installing with conda
- Verifying installation
- Importing Seaborn
  - `import seaborn as sns`
  - `import matplotlib.pyplot as plt`

### 1.3 Seaborn Basics
- Figure-level vs Axes-level functions
- Figure-level functions — `relplot`, `displot`, `catplot`, `lmplot`
- Axes-level functions — `scatterplot`, `histplot`, `boxplot`, `barplot`
- `FacetGrid` object
- Relationship between Seaborn and Matplotlib
- Showing plots — `plt.show()`
- Saving plots — `plt.savefig()`

### 1.4 Built-in Datasets
- `sns.load_dataset('iris')`
- `sns.load_dataset('titanic')`
- `sns.load_dataset('tips')`
- `sns.load_dataset('flights')`
- `sns.load_dataset('penguins')`
- `sns.get_dataset_names()` — list all datasets

---

## 2. Seaborn Themes and Aesthetics

### 2.1 Themes
- `sns.set_theme()` — apply default theme
- Available themes
  - `'darkgrid'` — dark background with grid
  - `'whitegrid'` — white background with grid
  - `'dark'` — dark background no grid
  - `'white'` — white background no grid
  - `'ticks'` — ticks on axes
- `sns.set_style('style_name')` — set style
- `sns.axes_style()` — get current style
- Context manager — `with sns.axes_style('style')`

### 2.2 Context (Scale)
- `sns.set_context('context_name')` — set scale
- Available contexts
  - `'paper'` — small fonts
  - `'notebook'` — default
  - `'talk'` — larger for presentations
  - `'poster'` — largest for posters
- `rc` parameter for custom scaling

### 2.3 Color Palettes
- `sns.color_palette()` — view current palette
- `sns.set_palette('palette_name')` — set palette
- Built-in palettes
  - `'deep'`
  - `'muted'`
  - `'pastel'`
  - `'bright'`
  - `'dark'`
  - `'colorblind'`
- Sequential palettes — `'Blues'`, `'Greens'`, `'viridis'`, `'plasma'`
- Diverging palettes — `'coolwarm'`, `'RdBu'`, `'BrBG'`
- `sns.palplot()` — displaying a palette
- Custom palettes using hex codes
- `sns.husl_palette()` — HUSL color space

### 2.4 Figure Size and DPI
- `plt.figure(figsize=(width, height))`
- `height` parameter in figure-level functions
- `aspect` parameter in figure-level functions
- DPI — `plt.savefig(..., dpi=150)`

---

## 3. Relational Plots

### 3.1 Scatter Plot
- `sns.scatterplot(data=df, x='col1', y='col2')` — basic scatter
- `hue` — color by a categorical variable
- `size` — size by a variable
- `style` — marker style by a variable
- `alpha` — transparency
- `palette` — color palette
- `legend` — show or hide legend
- Custom marker styles

### 3.2 Line Plot
- `sns.lineplot(data=df, x='col1', y='col2')` — basic line
- `hue` — multiple lines by category
- `style` — line style by category
- `ci` — confidence interval (deprecated → `errorbar`)
- `errorbar` — error representation
  - `'ci'` — confidence interval
  - `'sd'` — standard deviation
  - `'se'` — standard error
  - `None` — no error band
- `estimator` — aggregation function (default: mean)
- `dashes` — custom dash patterns

### 3.3 relplot (Figure-level)
- `sns.relplot(data=df, x='x', y='y', kind='scatter')` — scatter
- `sns.relplot(data=df, x='x', y='y', kind='line')` — line
- `col` — facet by column variable
- `row` — facet by row variable
- `col_wrap` — wrap columns
- `hue`, `size`, `style` — same as axes-level

---

## 4. Distribution Plots

### 4.1 Histogram
- `sns.histplot(data=df, x='col')` — basic histogram
- `bins` — number of bins
- `binwidth` — bin width
- `kde` — overlay KDE curve
- `stat` — normalization
  - `'count'` — raw count
  - `'frequency'` — count / bin width
  - `'density'` — normalized density
  - `'probability'` — fraction of total
- `hue` — overlay by category
- `multiple` — handling multiple distributions
  - `'layer'`, `'dodge'`, `'stack'`, `'fill'`
- `cumulative` — cumulative histogram
- `element` — `'bars'`, `'step'`, `'poly'`

### 4.2 KDE Plot
- `sns.kdeplot(data=df, x='col')` — 1D KDE
- `sns.kdeplot(data=df, x='col1', y='col2')` — 2D KDE
- `hue` — multiple KDE curves
- `bw_adjust` — bandwidth adjustment
- `fill` — fill under KDE curve
- `cut` — extent beyond data range
- `cumulative` — cumulative KDE
- `levels` — 2D contour levels
- `thresh` — 2D threshold

### 4.3 ECDF Plot
- `sns.ecdfplot(data=df, x='col')` — empirical CDF
- `hue` — multiple ECDFs
- `stat` — `'proportion'` or `'count'`
- `complementary` — survival function (1 - CDF)

### 4.4 Rug Plot
- `sns.rugplot(data=df, x='col')` — tick marks for data
- `height` — rug height
- `hue` — color by category
- Used as overlay on KDE or histogram

### 4.5 displot (Figure-level)
- `sns.displot(data=df, x='col', kind='hist')` — histogram
- `sns.displot(data=df, x='col', kind='kde')` — KDE
- `sns.displot(data=df, x='col', kind='ecdf')` — ECDF
- `col` — facet by column
- `row` — facet by row
- `rug` — add rug

---

## 5. Categorical Plots

### 5.1 Strip Plot
- `sns.stripplot(data=df, x='cat', y='num')` — jittered points
- `jitter` — amount of jitter
- `dodge` — separate by hue
- `order` — custom category order

### 5.2 Swarm Plot
- `sns.swarmplot(data=df, x='cat', y='num')` — non-overlapping points
- `size` — point size
- `hue` — color by category
- `dodge` — separate by hue

### 5.3 Box Plot
- `sns.boxplot(data=df, x='cat', y='num')` — box and whisker
- `hue` — grouped boxes
- `dodge` — separate hue groups
- `width` — box width
- `fliersize` — outlier marker size
- `notch` — notched box
- `whis` — whisker range
- `order` — custom category order

### 5.4 Violin Plot
- `sns.violinplot(data=df, x='cat', y='num')` — violin shape
- `hue` — grouped violins
- `split` — half violins for two hue levels
- `inner` — inner marks — `'box'`, `'quartile'`, `'point'`, `'stick'`, `None`
- `bw_adjust` — bandwidth
- `scale` — `'area'`, `'count'`, `'width'`
- `cut` — extent beyond range

### 5.5 Boxen Plot (Letter-Value Plot)
- `sns.boxenplot(data=df, x='cat', y='num')` — enhanced box plot
- Better for large datasets
- `k_depth` — depth of quantiles
- `hue` — grouped boxen

### 5.6 Bar Plot
- `sns.barplot(data=df, x='cat', y='num')` — mean bar with CI
- `estimator` — aggregation function
- `errorbar` — error representation
- `hue` — grouped bars
- `order` — category order
- `palette` — color palette

### 5.7 Count Plot
- `sns.countplot(data=df, x='cat')` — frequency count
- `hue` — stacked or grouped counts
- `order` — category order
- `stat` — `'count'` or `'proportion'`

### 5.8 Point Plot
- `sns.pointplot(data=df, x='cat', y='num')` — mean points connected
- `hue` — grouped point plots
- `dodge` — separate hue groups
- `markers` — custom markers
- `linestyles` — custom line styles
- `errorbar` — error bars

### 5.9 catplot (Figure-level)
- `sns.catplot(data=df, x='cat', y='num', kind='box')`
- `kind` options — `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, `'count'`
- `col` — facet by column
- `row` — facet by row
- `col_wrap` — wrap columns

---

## 6. Regression Plots

### 6.1 Linear Regression Plot
- `sns.regplot(data=df, x='col1', y='col2')` — scatter + regression line
- `ci` — confidence interval shade
- `order` — polynomial order
- `logistic` — logistic regression fit
- `robust` — robust regression
- `lowess` — locally weighted regression
- `x_jitter` — jitter for discrete x
- `scatter_kws` — scatter point customization
- `line_kws` — line customization

### 6.2 lmplot (Figure-level)
- `sns.lmplot(data=df, x='col1', y='col2')` — regplot with faceting
- `hue` — separate regression lines by category
- `col` — facet by column
- `row` — facet by row
- `col_wrap` — column wrapping

### 6.3 Residual Plot
- `sns.residplot(data=df, x='col1', y='col2')` — residuals vs fitted
- `order` — polynomial order
- `lowess` — LOWESS smoothing

---

## 7. Matrix Plots

### 7.1 Heatmap
- `sns.heatmap(data)` — basic heatmap
- `annot` — show values in cells
- `fmt` — format of annotations (`'.2f'`, `'d'`)
- `cmap` — colormap
- `vmin` and `vmax` — color scale limits
- `center` — center of diverging colormap
- `linewidths` — cell border width
- `linecolor` — cell border color
- `square` — equal cell dimensions
- `cbar` — show or hide colorbar
- `cbar_kws` — colorbar customization
- `mask` — masking cells (e.g., upper triangle)
- `xticklabels` and `yticklabels` — axis labels

### 7.2 Cluster Map
- `sns.clustermap(data)` — heatmap with hierarchical clustering
- `method` — linkage method (`'ward'`, `'complete'`, `'average'`)
- `metric` — distance metric
- `z_score` — normalize by z-score
- `standard_scale` — normalize by min-max
- `row_cluster` — cluster rows
- `col_cluster` — cluster columns
- `row_colors` — color coding for rows
- `col_colors` — color coding for columns
- `figsize` — figure size

---

## 8. Multi-Plot Grids

### 8.1 FacetGrid
- `g = sns.FacetGrid(data=df, col='col1', row='col2')`
- `g.map()` — applying a plotting function to each facet
- `g.map_dataframe()` — applying with DataFrame
- `col_wrap` — wrapping columns
- `hue` — color by variable
- `sharey` and `sharex` — shared axes
- `height` and `aspect` — size control
- `g.add_legend()` — adding a legend
- `g.set_axis_labels()` — axis labels
- `g.set_titles()` — facet titles

### 8.2 Pair Plot
- `sns.pairplot(data=df)` — scatter matrix
- `hue` — color by categorical variable
- `vars` — select specific variables
- `x_vars` and `y_vars` — different x and y variables
- `diag_kind` — diagonal plot type — `'hist'`, `'kde'`, `None`
- `kind` — off-diagonal plot type — `'scatter'`, `'kde'`, `'hist'`, `'reg'`
- `plot_kws` — scatter customization
- `diag_kws` — diagonal plot customization
- `corner` — lower triangle only

### 8.3 Joint Plot
- `sns.jointplot(data=df, x='col1', y='col2')` — joint + marginal
- `kind` — `'scatter'`, `'kde'`, `'hist'`, `'hex'`, `'reg'`, `'resid'`
- `hue` — color by category
- `marginal_ticks` — show ticks on marginals
- `ratio` — ratio of joint to marginal size
- `space` — space between joint and marginal
- `xlim` and `ylim` — axis limits

---

## 9. Customizing Seaborn Plots

### 9.1 Titles and Labels
- `ax.set_title('title')` — plot title
- `ax.set_xlabel('label')` — x-axis label
- `ax.set_ylabel('label')` — y-axis label
- Figure-level titles — `g.figure.suptitle()`
- Font size customization — `fontsize`

### 9.2 Legends
- `ax.legend()` — Matplotlib legend on Seaborn plot
- `legend` parameter in plot functions — `'auto'`, `'brief'`, `'full'`, `False`
- Repositioning legend — `bbox_to_anchor`
- Custom legend labels

### 9.3 Tick Customization
- `ax.set_xticks()` — custom tick positions
- `ax.set_xticklabels()` — custom tick labels
- `plt.xticks(rotation=45)` — rotated labels
- `ax.tick_params()` — tick appearance

### 9.4 Combining with Matplotlib
- Accessing underlying Matplotlib `fig` and `ax`
- `ax = sns.scatterplot(...)` — returns Axes object
- Adding Matplotlib annotations on Seaborn plots
- `plt.axhline()` and `plt.axvline()` — reference lines
- `plt.text()` — adding text to Seaborn plots

---

## 10. Seaborn with Pandas

### 10.1 Using DataFrames Directly
- `data=df` parameter in all Seaborn functions
- Column names as `x`, `y`, `hue`, `size`, `style`
- Automatic axis labels from column names
- Handling categorical dtypes automatically

### 10.2 Long vs Wide Format Data
- Long format — preferred by Seaborn
- `pd.melt()` — converting wide to long for Seaborn
- Wide format with `hue` workaround

---

## 11. Seaborn Best Practices

### 11.1 Effective Visualization
- Use `hue` to add a categorical dimension without subplots
- Use `col`/`row` in figure-level functions for faceting
- Choose appropriate plot type for data type
  - Continuous — histogram, KDE, scatter
  - Categorical — bar, box, violin, count
  - Two continuous — scatter, reg, hex
- Always include axis labels and titles

### 11.2 Readability
- Use `colorblind` palette for accessibility
- Limit hue levels to ≤ 6 for readability
- Use `plt.tight_layout()` after Seaborn plots
- Use `context('talk')` for presentations

### 11.3 Common Mistakes
- Mixing figure-level and axes-level in same figure incorrectly
- Forgetting to call `plt.show()` in scripts
- Not resetting styles between plots
- Using wide-format data without melting
- Ignoring deprecation warnings (`ci` → `errorbar`)

---

## 12. Seaborn Interview Topics

### 12.1 Frequently Asked Questions
- Difference between figure-level and axes-level functions
- When to use `boxplot` vs `violinplot`
- How to create a heatmap from a correlation matrix
- How to facet plots using `catplot` or `FacetGrid`
- Difference between `pairplot` and `jointplot`

### 12.2 Coding Problems
- Create a correlation heatmap with annotations
- Plot distribution of a feature by class using violin plot
- Create a pair plot colored by species
- Build a FacetGrid scatter plot by category
- Plot bar chart showing mean value with confidence interval

---

## 13. Real-World Seaborn Projects

### 13.1 Beginner Projects
- Iris dataset EDA with pairplot and heatmap
- Titanic survival analysis with count and bar plots
- Tips dataset distribution and regression plots

### 13.2 Intermediate Projects
- Customer segmentation visualization dashboard
- Sales trend analysis with catplot and FacetGrid
- Feature correlation and distribution report

### 13.3 Advanced Projects
- Complete EDA dashboard for a classification dataset
- Multi-dataset comparison with FacetGrid
- Publication-ready statistical figure

---

## 14. Seaborn Ecosystem

### 14.1 Related Libraries
- Matplotlib — underlying engine
- Pandas — data input format
- SciPy — statistical computations used internally
- Statsmodels — regression computations

### 14.2 Seaborn Extensions
- Seaborn `objects` interface (v0.12+)
  - Declarative plotting API
  - `so.Plot()` — new composable interface
  - Scales, transforms, and marks
- Integration with Plotly for interactivity

---

## 15. Seaborn Objects Interface (v0.12+)

### 15.1 Plot Object Basics
- `so.Plot(data=df, x='col1', y='col2')` — creating a plot
- `.add(so.Dot())` — adding a mark
- `.add(so.Line())` — adding a line
- `.add(so.Bar())` — adding bars
- `.add(so.Area())` — adding area

### 15.2 Marks
- `so.Dot()` — scatter points
- `so.Line()` — line
- `so.Bar()` — bars
- `so.Area()` — area fill
- `so.Dash()` — horizontal markers
- `so.Path()` — connected path

### 15.3 Scales and Transforms
- `so.Continuous()` — continuous scale
- `so.Nominal()` — categorical scale
- `so.Temporal()` — datetime scale
- `.scale()` — applying scales

### 15.4 Faceting in Objects Interface
- `.facet(col='col_name')` — column facets
- `.facet(row='col_name')` — row facets
- `.layout(size=(width, height))` — size control
