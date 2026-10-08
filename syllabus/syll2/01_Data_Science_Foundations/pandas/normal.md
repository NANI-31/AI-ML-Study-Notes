# Pandas Roadmap / Syllabus

---

## 1. Introduction to Pandas

### 1.1 What is Pandas?
- Definition of Pandas
- Why Pandas is important
- Pandas vs Python lists and dictionaries
- Advantages of Pandas
  - Tabular data handling
  - Built-in data alignment
  - Handling missing data
  - Powerful grouping and aggregation
  - Integration with NumPy, Matplotlib, and Scikit-learn
- Real-world applications
  - Data Science
  - Machine Learning preprocessing
  - Financial analysis
  - Business intelligence
  - Scientific research

### 1.2 Installing Pandas
- Installing with pip
- Installing with conda
- Verifying installation
- Importing Pandas
  - `import pandas as pd`

### 1.3 Pandas Basics
- Series object
- DataFrame object
- Index object
- Axes
- Data alignment basics

---

## 2. Pandas Data Structures

### 2.1 Series
- Creating a Series from a Python list
- Creating a Series from a dictionary
- Creating a Series from a scalar value
- Series attributes
  - `values`
  - `index`
  - `dtype`
  - `shape`
  - `size`
  - `name`
- Accessing Series elements
- Series with custom index

### 2.2 DataFrame
- Creating a DataFrame from a dictionary
- Creating a DataFrame from a list of lists
- Creating a DataFrame from a NumPy array
- Creating a DataFrame from a CSV file
- DataFrame attributes
  - `columns`
  - `index`
  - `shape`
  - `dtypes`
  - `values`
  - `ndim`
  - `size`
- Accessing columns
- Accessing rows

### 2.3 Index Object
- Default RangeIndex
- Custom Index
- String Index
- DatetimeIndex basics
- Setting and resetting index
- `set_index()`
- `reset_index()`

---

## 3. Reading and Writing Data

### 3.1 Reading CSV Files
- `pd.read_csv()`
- Specifying delimiter
- Header options
- Specifying columns to read (`usecols`)
- Skipping rows (`skiprows`)
- Handling missing values on read
- Encoding options

### 3.2 Reading Excel Files
- `pd.read_excel()`
- Specifying sheet name
- Reading multiple sheets
- `pd.ExcelFile()`

### 3.3 Reading JSON Files
- `pd.read_json()`
- JSON orientation types

### 3.4 Reading Other Formats
- `pd.read_sql()`
- `pd.read_html()`
- `pd.read_parquet()`
- `pd.read_clipboard()`

### 3.5 Writing Data
- `df.to_csv()`
- `df.to_excel()`
- `df.to_json()`
- `df.to_parquet()`
- `df.to_sql()`
- Writing without index (`index=False`)

---

## 4. Exploring DataFrames

### 4.1 Basic Exploration
- `head()`
- `tail()`
- `shape`
- `info()`
- `describe()`
- `dtypes`
- `columns`
- `index`

### 4.2 Value Inspection
- `value_counts()`
- `unique()`
- `nunique()`
- `isnull()`
- `notnull()`
- `any()`
- `all()`

### 4.3 Statistical Summary
- `mean()`
- `median()`
- `std()`
- `var()`
- `min()`
- `max()`
- `sum()`
- `count()`
- `quantile()`
- `cumsum()`
- `cumprod()`

---

## 5. Indexing and Selecting Data

### 5.1 Column Selection
- Single column selection (`df['col']`)
- Multiple column selection (`df[['col1', 'col2']]`)
- Attribute-style access (`df.col`)

### 5.2 Row Selection with `.loc[]`
- Selecting by label
- Selecting a range of rows by label
- Selecting rows and columns simultaneously
- Boolean indexing with `.loc[]`

### 5.3 Row Selection with `.iloc[]`
- Selecting by integer position
- Selecting a range of rows by position
- Selecting specific rows and columns by position
- Negative indexing

### 5.4 Conditional Selection
- Single condition filtering
- Multiple conditions with `&` and `|`
- `~` (NOT) operator
- `.query()` method
- `.isin()` method
- `.between()` method

### 5.5 Advanced Indexing
- MultiIndex (Hierarchical Index)
- Creating a MultiIndex
- Accessing MultiIndex levels
- Slicing MultiIndex with `.loc[]`
- `xs()` method for cross-section

---

## 6. Modifying DataFrames

### 6.1 Adding Columns
- Adding a new constant column
- Adding a computed column
- `df.assign()`
- `df.insert()`

### 6.2 Modifying Columns
- Renaming columns with `rename()`
- Renaming all columns
- Changing dtype of a column (`astype()`)
- Applying a function to a column (`apply()`, `map()`)

### 6.3 Dropping Data
- `drop()` — dropping rows
- `drop()` — dropping columns
- `dropna()` — dropping missing rows
- `drop_duplicates()` — dropping duplicate rows

### 6.4 Sorting Data
- `sort_values()` — sorting by column values
- `sort_values()` — ascending and descending
- `sort_values()` — sorting by multiple columns
- `sort_index()` — sorting by index
- `nlargest()` and `nsmallest()`

---

## 7. Handling Missing Data

### 7.1 Detecting Missing Values
- `isnull()`
- `notnull()`
- `isnull().sum()`
- `isnull().any()`

### 7.2 Dropping Missing Values
- `dropna()` — drop rows with any NaN
- `dropna(how='all')` — drop rows where all values are NaN
- `dropna(subset=[...])` — drop based on specific columns
- `dropna(axis=1)` — drop columns with NaN

### 7.3 Filling Missing Values
- `fillna()` — fill with a constant
- `fillna(method='ffill')` — forward fill
- `fillna(method='bfill')` — backward fill
- `fillna(df.mean())` — fill with column mean
- `interpolate()` — interpolation methods

### 7.4 Replacing Values
- `replace()` — replace specific values
- `replace()` — replace using dictionary
- `replace()` — replace with regex

---

## 8. Data Transformation

### 8.1 Applying Functions
- `apply()` — apply function along rows or columns
- `applymap()` / `map()` — element-wise function
- `map()` — mapping values using a dictionary or Series
- Lambda functions with `apply()`

### 8.2 String Operations
- `str.lower()` and `str.upper()`
- `str.strip()`
- `str.split()`
- `str.replace()`
- `str.contains()`
- `str.startswith()` and `str.endswith()`
- `str.len()`
- `str.extract()` — regex extraction
- `str.get_dummies()` — string dummies

### 8.3 Data Type Conversion
- `astype()` — convert dtype
- `pd.to_numeric()`
- `pd.to_datetime()`
- `pd.to_timedelta()`
- `pd.Categorical()`

### 8.4 Binning and Discretization
- `pd.cut()` — bin values into intervals
- `pd.qcut()` — quantile-based binning
- Labeling bins
- Custom bin edges

---

## 9. Grouping and Aggregation

### 9.1 GroupBy Basics
- `groupby()` — grouping by a single column
- `groupby()` — grouping by multiple columns
- GroupBy object
- Iterating over groups

### 9.2 Aggregation Functions
- `agg()` — multiple aggregation functions
- `sum()`, `mean()`, `count()`, `min()`, `max()`
- Custom aggregation functions
- `agg({'col': func})` — per-column aggregation

### 9.3 Transformation and Filtering
- `transform()` — group-wise transformation
- `filter()` — filtering groups by condition
- `apply()` — custom group operations

### 9.4 Pivot Tables
- `pivot_table()` — creating pivot tables
- `values`, `index`, `columns`, `aggfunc` parameters
- Margins and totals
- `pd.pivot()` — simple pivot without aggregation

### 9.5 Cross-tabulation
- `pd.crosstab()` — frequency table
- Normalize option
- Margins option

---

## 10. Merging and Joining DataFrames

### 10.1 Concatenation
- `pd.concat()` — stacking vertically
- `pd.concat()` — stacking horizontally
- Axis parameter
- Ignoring index
- Handling different columns

### 10.2 Merging DataFrames
- `pd.merge()` — inner join
- `pd.merge()` — left join
- `pd.merge()` — right join
- `pd.merge()` — outer join
- Merge on a single key
- Merge on multiple keys
- Merge on index
- `suffixes` parameter for overlapping columns

### 10.3 Joining DataFrames
- `df.join()` — joining on index
- `df.join()` — joining with `how` parameter
- Difference between merge and join

### 10.4 Combining DataFrames
- `df.combine_first()` — updating missing values
- `df.update()` — in-place update

---

## 11. Reshaping DataFrames

### 11.1 Stacking and Unstacking
- `stack()` — pivoting innermost column level to row
- `unstack()` — pivoting innermost row level to column

### 11.2 Melting DataFrames
- `pd.melt()` — wide to long format
- `id_vars` and `value_vars` parameters
- `var_name` and `value_name`

### 11.3 Pivoting DataFrames
- `pivot()` — reshaping with index and columns
- Handling duplicates in pivot

### 11.4 Wide and Long Format
- Wide format structure
- Long format structure
- Converting between wide and long

---

## 12. DateTime Handling

### 12.1 Parsing Dates
- `pd.to_datetime()` — parsing strings to datetime
- `format` parameter
- `errors` parameter
- Parsing from multiple columns

### 12.2 DatetimeIndex
- Creating DatetimeIndex
- `pd.date_range()` — creating date ranges
- Frequency aliases (`D`, `M`, `H`, etc.)

### 12.3 DateTime Properties
- `.dt.year`
- `.dt.month`
- `.dt.day`
- `.dt.hour`
- `.dt.minute`
- `.dt.second`
- `.dt.dayofweek`
- `.dt.day_name()`
- `.dt.month_name()`
- `.dt.quarter`

### 12.4 DateTime Operations
- Arithmetic with datetime
- Timedelta operations
- `pd.Timedelta()`
- Shifting dates
- Resampling time series

---

## 13. Categorical Data

### 13.1 Creating Categorical Data
- `pd.Categorical()`
- `astype('category')`
- Categories and ordered categories

### 13.2 Working with Categories
- `.cat.categories`
- `.cat.codes`
- `.cat.ordered`
- `rename_categories()`
- `add_categories()`
- `remove_categories()`
- `reorder_categories()`

### 13.3 Benefits of Categorical
- Memory efficiency
- Ordered comparisons
- Grouping performance

---

## 14. Performance and Optimization

### 14.1 Efficient Data Operations
- Vectorized operations over loops
- `eval()` for fast expression evaluation
- `query()` for fast filtering
- Using appropriate dtypes

### 14.2 Memory Optimization
- Downcasting numeric types
- Using `category` dtype for string columns
- `pd.read_csv()` with `dtype` parameter
- `usecols` to read only needed columns

### 14.3 Chunked Reading
- `chunksize` parameter in `read_csv()`
- Iterating over chunks
- Processing large files in chunks

---

## 15. Pandas with Other Libraries

### 15.1 Pandas + NumPy
- Converting DataFrame to NumPy array (`df.values`, `df.to_numpy()`)
- Using NumPy functions on Pandas columns
- NumPy arrays as DataFrame input

### 15.2 Pandas + Matplotlib
- `df.plot()` — plotting directly from DataFrame
- Line plot, bar plot, histogram, scatter plot, box plot, pie chart
- Subplots from DataFrame
- Customizing Pandas plots

### 15.3 Pandas + Seaborn
- Passing DataFrames to Seaborn functions
- `hue`, `x`, `y` parameters

### 15.4 Pandas + Scikit-learn
- Preparing DataFrames for ML models
- Feature matrix and target vector
- Encoding categorical features
- Splitting data

---

## 16. Data Cleaning

### 16.1 Identifying Dirty Data
- Detecting duplicates
- Detecting inconsistent data types
- Detecting wrong value ranges
- Detecting misspellings in strings

### 16.2 Cleaning Techniques
- Removing duplicates — `drop_duplicates()`
- Stripping whitespace — `str.strip()`
- Fixing data types — `astype()`
- Renaming columns
- Standardizing categorical values

### 16.3 Outlier Detection
- Z-score method
- IQR method
- Removing outliers using boolean masking

---

## 17. Advanced Pandas Concepts

### 17.1 Window Functions
- `rolling()` — rolling window calculations
- `expanding()` — expanding window calculations
- `ewm()` — exponentially weighted functions
- Rolling mean, sum, std, min, max

### 17.2 Pipe and Method Chaining
- `pipe()` — chaining custom functions
- Method chaining pattern
- Readable pipeline construction

### 17.3 Styling DataFrames
- `Styler` object
- `style.highlight_max()`
- `style.highlight_min()`
- `style.background_gradient()`
- `style.format()`
- Exporting styled DataFrames to Excel

### 17.4 Options and Settings
- `pd.set_option()`
- `pd.get_option()`
- Displaying more rows and columns
- Float formatting options

---

## 18. Pandas Best Practices

### 18.1 Writing Efficient Code
- Prefer vectorized operations
- Avoid iterating with `iterrows()` when possible
- Use `apply()` wisely
- Use appropriate dtypes from the start

### 18.2 Code Readability
- Use meaningful column names
- Chain operations clearly
- Comment GroupBy and merge logic

### 18.3 Common Mistakes
- Modifying a copy vs original (chained assignment)
- Ignoring `SettingWithCopyWarning`
- Incorrect join types
- Not resetting index after filtering
- Ignoring dtype mismatches

---

## 19. Pandas Interview Topics

### 19.1 Frequently Asked Questions
- Difference between Series and DataFrame
- `.loc[]` vs `.iloc[]`
- `merge()` vs `join()` vs `concat()`
- How to handle missing data
- What is chained indexing?
- How does GroupBy work internally?

### 19.2 Coding Problems
- Remove duplicates and keep the last occurrence
- Find top N values per group
- Reshape a wide DataFrame to long format
- Merge two DataFrames and fill missing values
- Calculate rolling average on a time series

---

## 20. Real-World Pandas Projects

### 20.1 Beginner Projects
- Student grades analysis
- Sales data summary report
- Weather data exploration

### 20.2 Intermediate Projects
- E-commerce data pipeline
- Customer segmentation using GroupBy
- Financial data cleaning and analysis

### 20.3 Advanced Projects
- End-to-end data preprocessing pipeline for ML
- Time series analysis and forecasting prep
- Multi-source data merging and reporting dashboard

---

## 21. Pandas Ecosystem

### 21.1 Related Libraries
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Plotly

### 21.2 Pandas Extensions
- GeoPandas — geospatial data
- Modin — parallel Pandas
- Dask — large-scale DataFrames
- Vaex — out-of-core DataFrames

---

## 22. Testing and Debugging Pandas Code

### 22.1 Debugging DataFrames
- Checking shapes and dtypes
- Printing intermediate results
- Using `assert` for data validation

### 22.2 Testing
- `pd.testing.assert_frame_equal()`
- `pd.testing.assert_series_equal()`
- Comparing DataFrame outputs in unit tests
