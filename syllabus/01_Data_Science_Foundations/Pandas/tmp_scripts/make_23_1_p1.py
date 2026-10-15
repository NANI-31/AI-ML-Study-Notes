import sys

p1_content = """# 23. Pandas Advanced Concepts

## 23.1 MultiIndex

### 1. Section Overview

In enterprise data engineering, tabular data rarely conforms to flat, one-dimensional coordinate systems. Real-world business phenomena—financial market order books, multi-regional cloud telemetry, genomics variant pipelines, and transactional e-commerce ledgers—are inherently multidimensional. The **Pandas MultiIndex** (hierarchical index) is the foundational data structure that provides higher-dimensional indexing capabilities to 1D `Series` and 2D `DataFrame` objects without requiring the memory overhead or complex mechanics of N-dimensional tensors or sparse coordinate matrices.

A standard Index assigns a single scalar label (such as an integer, string, or timestamp) to each row or column. In contrast, a **MultiIndex** assigns an ordered tuple of labels across two or more discrete **levels**. This enables engineers to represent arbitrary $N$-dimensional data cubes directly within standard tabular structures, execute fast sub-dimensional aggregations, perform surgical cross-sectional lookups, and align heterogeneous datasets along complex composite keys.

Mastery of `MultiIndex` is a decisive separator in technical interviews for Data Analyst, Data Scientist, and Data Engineer roles. Interviewers routinely evaluate a candidate's grasp of:
1. **The Internal Dual-Array Architecture (`levels` and `codes`)**: Understanding how Pandas decouples unique label arrays from integer position mappings to achieve massive memory compression and $O(1)$ coordinate hashing.
2. **Deterministic Selection Protocols**: Knowing when and how to navigate multi-level coordinates using scalar indexing, tuple access, cross-section slicing via `.xs()`, and arbitrary slicing via `pd.IndexSlice`.
3. **Lexicographical Sort Invariants**: Understanding why unsorted MultiIndexes trigger catastrophic $O(N)$ full table scans or raise `UnsortedIndexError`, and how sorting restores $O(\log N)$ binary search performance.
4. **Reshaping and Index Normalization**: Seamlessly transitioning between hierarchical dimensional indices and flat, denormalized relational structures via `set_index()`, `reset_index()`, and `droplevel()`.

#### Master MultiIndex Architectural Comparison

| MultiIndex Feature / Component | Core Mechanism | Low-Level Representation | Algorithmic Complexity | Primary Production Antipattern |
| :--- | :--- | :--- | :--- | :--- |
| **`levels`** | Unique labels per hierarchical dimension | FrozenList of 1D NumPy/Index arrays (strings, ints, datetimes) | $O(1)$ memory per unique label | High-cardinality unique IDs stored as outer levels |
| **`codes`** | Positional integer offsets pointing into `levels` | 1D NumPy signed integer arrays (`int8`, `int16`, `int32`, `int64`) | Compact contiguous memory | Manual array manipulation bypassing the MultiIndex API |
| **Tuple Selection (`.loc[(k1, k2)]`)** | Exact coordinate addressing | Direct hash lookup on the composite tuple key | $O(1)$ amortized dictionary lookup | Passing lists `['A', 'B']` instead of tuples `('A', 'B')` |
| **Cross-Section (`.xs(k, level=L)`)** | Level-specific extraction across arbitrary depth | Boolean mask evaluation / integer code index slicing | $O(N)$ unindexed or $O(K)$ sorted | Attempting in-place write assignment on `.xs()` view |
| **IndexSlice (`pd.IndexSlice[:, :]`)** | Range-based multidimensional slicing | Multi-slice tuple construction passed to `.loc` | $O(\log N + K)$ when lexsorted | Slicing unlexsorted indices (causes `UnsortedIndexError`) |
| **Lexicographical Sort (`sort_index`)** | Ordering composite keys sequentially by level | C/Cython quicksort/mergesort over integer codes | $O(N \log N)$ run-time | Performing multiple sequential slices without pre-sorting |
| **Index Resetting (`reset_index`)** | Moving index levels into standard data columns | Allocates new DataFrame columns and RangeIndex | $O(N \times D)$ copy operation | Unhandled column name collisions on non-named levels |

---

### 2. Topic-Wise Interview Concepts

#### 2.1 Understanding MultiIndex
- **Concept**: A `MultiIndex` is an immutable, ordered collection of tuple keys representing multiple hierarchical levels of indexing on a Pandas `Series` or `DataFrame`.
- **Why it matters**: It provides the expressiveness of multidimensional data cubes (like OLAP hypercubes) while maintaining tabular DataFrame semantics. Interviewers test this to see if you understand advanced data modeling beyond flat SQL tables.
- **Technical / Algorithmic Intuition**: Rather than storing $N$ full tuples (which would cause massive Python heap allocation and pointer chasing), Pandas uses a **compressed 2-part architecture**:
  1. `levels`: A collection of discrete 1D Index arrays, one for each level, storing only the **unique** label values.
  2. `codes`: An array of integer arrays where each integer is an index into the corresponding level's unique array. For example, if level 0 has labels `['East', 'West']`, the codes array stores `0` and `1` (often as compact `int8`), drastically minimizing RAM usage.
- **Syntax / Pandas API**:
  - `mi.levels`: Returns a `FrozenList` of unique Index objects per level.
  - `mi.codes`: Returns a `FrozenList` of integer arrays encoding the position of each label.
  - `mi.names`: Returns the names assigned to each level.
  - `mi.nlevels`: Returns the integer count of hierarchical levels.
- **Minimal Working Example**:
  ```python
  import pandas as pd

  arrays = [["US", "US", "EU", "EU"], ["NY", "CA", "LON", "BER"]]
  mi = pd.MultiIndex.from_arrays(arrays, names=["Region", "City"])
  s = pd.Series([100, 200, 300, 400], index=mi)
  print(s.index.levels)
  print(s.index.codes)
  ```
- **Expected Output / Interpretation**:
  ```text
  FrozenList([['EU', 'US'], ['BER', 'CA', 'LON', 'NY']])
  FrozenList([[1, 1, 0, 0], [3, 1, 2, 0]])
  ```
  Notice that `levels` are sorted internally upon construction, and `codes` store indices referencing those sorted unique levels.
- **Important Parameters & Options**:
  - `levels` (sequence of arrays): Unique labels for each level.
  - `codes` (sequence of arrays): Integer position markers.
  - `names` (sequence of str, default `None`): Explicit names identifying each hierarchical level.
- **Common Interview Confusion**: Candidates often confuse `mi.levels` with `mi.get_level_values()`. `mi.levels[0]` returns only the **unique** labels in level 0, whereas `mi.get_level_values(0)` returns the full, repeated 1D array of labels corresponding to every single row in the DataFrame.

#### 2.2 Creating MultiIndex
- **Concept**: Constructing hierarchical indexes from diverse data sources: tuples, discrete arrays, Cartesian products, or tabular DataFrames.
- **Why it matters**: In production, MultiIndexes originate from raw logs, grouped outputs, or matrix generation. Candidates must know the idiomatic constructor for each scenario.
- **Technical / Algorithmic Intuition**:
  - `from_tuples()`: Scans a collection of Python tuples and parses them into unique levels and integer codes. Computational complexity $O(N)$ with memory allocation for unique sets.
  - `from_arrays()`: Takes parallel arrays representing columns. Direct vector intake into internal structures.
  - `from_product()`: Computes the Cartesian product of multiple iterables ($\prod |L_i|$). Highly efficient because unique levels are known beforehand, generating codes via simple arithmetic modular strides without hashing.
  - `from_frame()`: Extracts columns from an existing DataFrame directly into a MultiIndex.
  - `df.set_index(['col1', 'col2'])`: Converts existing DataFrame columns into an index, removing them from the data block and transferring memory to the index manager.
- **Syntax / Pandas API**:
  ```python
  pd.MultiIndex.from_tuples(tuples, names=None)
  pd.MultiIndex.from_arrays(arrays, names=None)
  pd.MultiIndex.from_product(iterables, names=None)
  pd.MultiIndex.from_frame(df, names=None)
  df.set_index(keys, drop=True, append=False, inplace=False)
  ```
- **Minimal Working Example**:
  ```python
  import pandas as pd

  # 1. Cartesian Product
  departments = ["HR", "Engineering"]
  quarters = ["Q1", "Q2"]
  mi_prod = pd.MultiIndex.from_product([departments, quarters], names=["Dept", "Quarter"])

  # 2. From DataFrame
  raw_df = pd.DataFrame({"Dept": ["HR", "HR", "Eng", "Eng"], "Emp": [1, 2, 1, 2], "Salary": [50, 60, 100, 110]})
  df_hier = raw_df.set_index(["Dept", "Emp"])
  ```
- **Expected Output / Interpretation**:
  ```text
  MultiIndex([('HR', 'Q1'), ('HR', 'Q2'), ('Engineering', 'Q1'), ('Engineering', 'Q2')], names=['Dept', 'Quarter'])
  ```
- **Important Parameters & Options**:
  - `drop=True` (in `set_index`): Deletes the columns from the data block once transferred to the index. If `drop=False`, columns remain in both the index and the data block.
  - `append=False` (in `set_index`): If `True`, preserves existing index levels and appends the new columns as inner levels.
- **Common Interview Confusion**: Beginners often forget that `set_index()` returns a new DataFrame by default. If `inplace=False` (default), the original DataFrame is unmodified.

#### 2.3 Selecting MultiIndex Data
- **Concept**: Querying rows and columns indexed by a MultiIndex using basic bracket notation `[]` and hierarchical keys.
- **Why it matters**: Demonstrates ability to retrieve data slices cleanly without resorting to slow boolean masks or resetting the index.
- **Technical / Algorithmic Intuition**:
  - On a `Series` with a MultiIndex, `s[key]` treats the outer level as primary: `s['OuterKey']` slices the outer level and drops it, returning a sub-Series indexed by the remaining inner levels.
  - On a `DataFrame`, `df[col_key]` selects columns, NOT rows! If columns have a MultiIndex, `df[('OuterCol', 'InnerCol')]` selects a single column. If rows have a MultiIndex, `df['Key']` looks for a column named `'Key'` and raises a `KeyError` if it only exists in the row index.
- **Syntax / Pandas API**:
  - `series[outer_key]`: Slices outer level.
  - `series[(outer_key, inner_key)]`: Point lookup for an exact coordinate.
  - `df[outer_col_key]`: Slices outer column level.
- **Minimal Working Example**:
  ```python
  import pandas as pd

  mi = pd.MultiIndex.from_tuples([("Tech", "AAPL"), ("Tech", "MSFT"), ("Energy", "XOM")])
  s = pd.Series([180, 420, 110], index=mi)

  # Select entire outer level
  tech_stocks = s["Tech"]
  # Point lookup
  aapl_price = s[("Tech", "AAPL")]
  print(tech_stocks)
  print("AAPL:", aapl_price)
  ```
- **Expected Output / Interpretation**:
  ```text
  AAPL    180
  MSFT    420
  dtype: int64
  AAPL: 180
  ```
- **Important Parameters & Options**:
  - Exact tuple matching requires passing tuples `('Tech', 'AAPL')`. Passing a list `['Tech', 'AAPL']` is treated as fancy indexing (selecting two separate outer keys), which fails if `'AAPL'` is an inner key.
- **Common Interview Confusion**: Passing `s['Tech', 'AAPL']` in Python is automatically parsed as a tuple `s[('Tech', 'AAPL')]`. However, writing `df['Tech', 'AAPL']` attempts to look up a column with that tuple, surprising engineers expecting row selection.

#### 2.4 `.loc[]` with MultiIndex
- **Concept**: Label-based coordinate selection on rows and columns using `.loc` with multi-level keys, tuples, ranges, and `pd.IndexSlice`.
- **Why it matters**: `.loc` is the canonical, industrial way to slice multi-level tabular data. Interviewers consider mastery of `pd.IndexSlice` a hallmark of an advanced Pandas developer.
- **Technical / Algorithmic Intuition**:
  - `.loc` accepts a 2-tuple: `df.loc[row_selector, col_selector]`.
  - When rows have a MultiIndex, `row_selector` can be a scalar (outer key), a tuple (exact hierarchical coordinate), a list of tuples, or an `IndexSlice`.
  - When slicing ranges across levels (e.g. `slice('A', 'B')`), Pandas requires the MultiIndex to be **lexicographically sorted** (`df.sort_index()`). Without sorting, index bounds cannot be determined via binary search, triggering `UnsortedIndexError`.
- **Syntax / Pandas API**:
  ```python
  idx = pd.IndexSlice
  df.loc[idx[row_level0_slice, row_level1_slice], idx[col_level0_slice, col_level1_slice]]
  ```
- **Minimal Working Example**:
  ```python
  import pandas as pd

  mi = pd.MultiIndex.from_product([["A", "B"], [2022, 2023]], names=["Entity", "Year"])
  df = pd.DataFrame({"Revenue": [100, 120, 200, 240], "Profit": [10, 15, 30, 40]}, index=mi)

  idx = pd.IndexSlice
  # Select all entities for Year 2023, column Revenue
  res = df.loc[idx[:, 2023], "Revenue"]
  print(res)
  ```
- **Expected Output / Interpretation**:
  ```text
  Entity  Year
  A       2023    120
  B       2023    240
  Name: Revenue, dtype: int64
  ```
- **Important Parameters & Options**:
  - `:` (colon inside `IndexSlice`): Represents "select all elements along this level".
  - Multi-level tuple: `('A', 2022)` specifies exact coordinates.
- **Common Interview Confusion**: Writing `df.loc['A':'B', 2022]` is ambiguous to Pandas. The comma separates the row axis from the column axis! To specify multi-level row slicing, you must wrap the levels inside a tuple or use `pd.IndexSlice`: `df.loc[idx['A':'B', 2022], :]`.

#### 2.5 `.xs()` (Cross-Section Selection)
- **Concept**: Selecting a subset of data at a specific hierarchical level across the index without needing to slice or specify all preceding outer levels.
- **Why it matters**: In deep hierarchical structures (e.g., 4 or 5 levels), extracting data for a specific inner level (e.g., `City == 'London'` regardless of `Continent`, `Country`, or `State`) is tedious with `.loc`. `.xs()` provides clean cross-sectional access.
- **Technical / Algorithmic Intuition**:
  - `.xs` inspects the specified `level` (by name or integer index) and matches the target `key`.
  - By default, it drops the selected level from the resulting index (`drop_level=True`), returning a simpler index.
  - `.xs()` returns a **shallow copy/view** and is strictly read-only. You cannot assign to `.xs()` (e.g., `df.xs(...) = 5` raises `SyntaxError` or `SettingWithCopyError`).
- **Syntax / Pandas API**:
  ```python
  df.xs(key, axis=0, level=None, drop_level=True)
  ```
- **Minimal Working Example**:
  ```python
  import pandas as pd

  mi = pd.MultiIndex.from_tuples([
      ("US", "East", "NY"), ("US", "West", "SF"),
      ("EU", "West", "LON"), ("EU", "East", "BER")
  ], names=["Region", "Zone", "City"])
  df = pd.DataFrame({"Temp": [22, 18, 15, 12]}, index=mi)

  # Select all rows where Zone == 'East' across all Regions
  east_data = df.xs("East", level="Zone")
  print(east_data)
  ```
- **Expected Output / Interpretation**:
  ```text
               Temp
  Region City      
  US     NY      22
  EU     BER     12
  ```
  Notice that the `Zone` level was automatically dropped from the resulting index.
- **Important Parameters & Options**:
  - `key`: The scalar label or tuple of labels to match.
  - `axis`: `0` for row index (default), `1` for column index.
  - `level`: Integer or string name of the target level.
  - `drop_level`: Boolean (default `True`). If `False`, retains the filtered level in the resulting index.
- **Common Interview Confusion**: Candidates often believe `.xs()` can perform range queries (e.g. `df.xs(slice(2020, 2022))`). It cannot; `.xs()` only accepts exact discrete keys. For range queries across levels, `pd.IndexSlice` with `.loc` is mandatory.

#### 2.6 Resetting and Dropping MultiIndex
- **Concept**: Flattening a MultiIndex back into standard data columns (`reset_index`) or permanently stripping unneeded hierarchical levels (`droplevel`).
- **Why it matters**: Most machine learning libraries (Scikit-Learn, XGBoost, PyTorch) and SQL database drivers require flat 2D tabular formats with single-level column headers. Data engineers must know how to cleanly normalize hierarchical indices.
- **Technical / Algorithmic Intuition**:
  - `reset_index()`: Allocates new columns in the DataFrame block for each index level specified, copies index level values into those columns, and assigns a default `RangeIndex(0, 1, 2, ..., N)`.
  - `droplevel()`: Removes one or more index levels entirely without retaining their values as data columns. It discards the corresponding level and codes arrays, returning an index with $L - 1$ levels.
- **Syntax / Pandas API**:
  ```python
  df.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
  df.droplevel(level, axis=0)
  ```
- **Minimal Working Example**:
  ```python
  import pandas as pd

  mi = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)], names=["Letter", "Number"])
  df = pd.DataFrame({"Value": [10, 20, 30]}, index=mi)

  # 1. Reset specific level into column
  df_reset = df.reset_index(level="Number")
  # 2. Completely drop a level without converting to column
  df_dropped = df.droplevel("Letter")

  print("Reset Number:\\n", df_reset)
  print("\\nDropped Letter:\\n", df_dropped)
  ```
- **Expected Output / Interpretation**:
  ```text
  Reset Number:
          Number  Value
  Letter               
  A            1     10
  A            2     20
  B            1     30

  Dropped Letter:
          Value
  Number       
  1          10
  2          20
  1          30
  ```
- **Important Parameters & Options**:
  - `level`: Specific level name or index to reset/drop. If `None`, resets all levels.
  - `drop=True` (in `reset_index`): Discards the index levels entirely instead of converting them into DataFrame columns.
- **Common Interview Confusion**: When a DataFrame has multi-level **columns** (e.g. resulting from `.groupby().agg()`), calling `df.reset_index()` only resets the **row** index! To flatten multi-level column headers, engineers must either use list comprehensions (`df.columns = ['_'.join(c) for c in df.columns]`) or call `df.droplevel(axis=1)`.

---

### 3. Fresher Interview Questions

#### A. Basic Questions (Exactly 20 Questions)

##### Q1: What is a MultiIndex in Pandas and how does it fundamentally differ from a standard Index?
- **Direct Answer & Definition**: A MultiIndex (hierarchical index) is an advanced indexing object in Pandas that allows a Series or DataFrame to possess multiple hierarchical levels of row or column coordinates represented by ordered tuples.
- **Technical & Computational Reasoning**: While a standard `Index` maps a 1D sequence of scalar labels to integer row offsets, a `MultiIndex` maintains an $N$-level coordinate grid using a compressed architecture composed of shared `levels` and positional integer `codes`.
- **Practical Application / Syntax**: It is instantiated via factory methods like `pd.MultiIndex.from_tuples()` or generated automatically during multi-column operations like `df.set_index(['Region', 'City'])`.
- **Nuances, Defaults & Pitfalls**: Slicing an unlexsorted MultiIndex raises an `UnsortedIndexError`, whereas standard indices allow arbitrary slices even if unsorted.

##### Q2: How does Pandas store MultiIndex internally under the hood to optimize memory?
- **Direct Answer & Definition**: Pandas stores a MultiIndex using two primary internal data structures: a list of unique label arrays called `levels`, and a list of integer offset arrays called `codes`.
- **Technical & Computational Reasoning**: Storing repetitive string labels across millions of rows causes extreme memory bloat and pointer overhead; by storing each unique string once in a `level` Index and mapping millions of rows using compact integer `codes` (e.g. `int8`), memory consumption is reduced by up to 90%.
- **Practical Application / Syntax**: You can inspect this internal representation directly using the public attributes `df.index.levels` and `df.index.codes`.
- **Nuances, Defaults & Pitfalls**: Mutating the `levels` directly is prohibited; `levels` is an immutable `FrozenList` and modifications require reconstructing the index or using dedicated API methods.

##### Q3: What is the difference between `MultiIndex.levels` and `MultiIndex.get_level_values()`?
- **Direct Answer & Definition**: `MultiIndex.levels` returns the unique label arrays defining the hierarchy, whereas `get_level_values()` returns the full sequence of labels across every row for a given level.
- **Technical & Computational Reasoning**: `levels[i]` has length equal to the number of distinct unique categories in level $i$, whereas `get_level_values(i)` has length equal to the total number of rows in the DataFrame ($N$).
- **Practical Application / Syntax**: Use `df.index.levels[0]` to inspect unique categories, and `df.index.get_level_values('City')` when performing row-level masking or export.
- **Nuances, Defaults & Pitfalls**: Filtering a DataFrame does not automatically prune unused categories from `levels`; unused levels persist unless explicitly cleaned with `mi.remove_unused_levels()`.

##### Q4: Why does a MultiIndex save memory compared to storing duplicate string labels per row?
- **Direct Answer & Definition**: A MultiIndex acts as an internal categorical compression mechanism by storing repetitive labels once in contiguous arrays and referencing them via integer codes.
- **Technical & Computational Reasoning**: In CPython, each string object carries a 49–80 byte memory header plus character storage; storing 1,000,000 identical strings allocates 1,000,000 distinct heap pointers, whereas a MultiIndex stores one string and 1,000,000 1-byte (`int8`) or 2-byte (`int16`) integers.
- **Practical Application / Syntax**: Setting a MultiIndex on hierarchical categorical fields (`df.set_index(['State', 'Store_ID'])`) significantly reduces DataFrame memory footprint.
- **Nuances, Defaults & Pitfalls**: If all combinations of keys are globally unique with cardinality approaching $N$, the memory advantage disappears and indexing overhead increases.

##### Q5: What is `pd.MultiIndex.from_tuples()` and what data structure does it expect?
- **Direct Answer & Definition**: `pd.MultiIndex.from_tuples()` is a constructor method that converts a list or iterable of uniform-length Python tuples into a hierarchical MultiIndex.
- **Technical & Computational Reasoning**: It iterates through the collection of tuples, computes the unique values for each coordinate position to create `levels`, and assigns integer indices to populate `codes`.
- **Practical Application / Syntax**:
  ```python
  tuples = [('US', 'East'), ('US', 'West'), ('EU', 'Central')]
  mi = pd.MultiIndex.from_tuples(tuples, names=['Region', 'Zone'])
  ```
- **Nuances, Defaults & Pitfalls**: All tuples must have the exact same length; passing tuples of varying lengths raises a `ValueError`.

##### Q6: How does `pd.MultiIndex.from_arrays()` work and how does it differ from `from_tuples()`?
- **Direct Answer & Definition**: `pd.MultiIndex.from_arrays()` constructs a MultiIndex from a list of parallel 1D arrays or lists representing column vectors.
- **Technical & Computational Reasoning**: While `from_tuples()` expects row-oriented data (`[(r1_l1, r1_l2), (r2_l1, r2_l2)]`), `from_arrays()` expects column-oriented data (`[[r1_l1, r2_l1], [r1_l2, r2_l2]]`).
- **Practical Application / Syntax**:
  ```python
  arrays = [['US', 'US', 'EU'], ['East', 'West', 'Central']]
  mi = pd.MultiIndex.from_arrays(arrays, names=['Region', 'Zone'])
  ```
- **Nuances, Defaults & Pitfalls**: All input arrays must have the exact same length ($N$); otherwise, Pandas raises a `ValueError: all arrays must be same length`.

##### Q7: What does `pd.MultiIndex.from_product()` do and when is it used in production?
- **Direct Answer & Definition**: `pd.MultiIndex.from_product()` creates a MultiIndex from the Cartesian product of two or more discrete iterables.
- **Technical & Computational Reasoning**: It generates every possible combination of elements across the provided sets, calculating total rows as $N = |A| \times |B| \times |C|$ and laying out codes via cyclic integer strides.
- **Practical Application / Syntax**: Commonly used in financial modeling and grid searches to generate complete parameter matrices or asset-date grids:
  ```python
  dates = pd.date_range('2023-01-01', periods=2)
  tickers = ['AAPL', 'GOOG']
  mi = pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker'])
  ```
- **Nuances, Defaults & Pitfalls**: Cartesian products can experience combinatorial explosion; passing three lists of length 1,000 creates 1 billion index rows, immediately exhausting RAM.

##### Q8: How does `pd.MultiIndex.from_frame()` construct an index from an existing DataFrame?
- **Direct Answer & Definition**: `pd.MultiIndex.from_frame()` takes a 2D DataFrame and converts each column into a corresponding level of a new MultiIndex.
- **Technical & Computational Reasoning**: It extracts each column's underlying array as an array input, preserving column names as level names and retaining native dtypes.
- **Practical Application / Syntax**:
  ```python
  df_keys = pd.DataFrame({'Dept': ['IT', 'HR'], 'Role': ['Dev', 'Recruiter']})
  mi = pd.MultiIndex.from_frame(df_keys)
  ```
- **Nuances, Defaults & Pitfalls**: By default, it preserves the existing DataFrame column names as index level names unless overridden via the `names` parameter.

##### Q9: What happens when you call `df.set_index(['col1', 'col2'])`?
- **Direct Answer & Definition**: `df.set_index()` transfers the specified columns out of the DataFrame's data block and constructs a MultiIndex on the row axis.
- **Technical & Computational Reasoning**: The columns are removed from the internal 2D `BlockManager`, and a new `MultiIndex` object is built and assigned to the DataFrame's `index` attribute.
- **Practical Application / Syntax**: `df = df.set_index(['Department', 'Employee_ID'])`.
- **Nuances, Defaults & Pitfalls**: By default, `drop=True`, which removes the source columns from the DataFrame body. Set `drop=False` if you require those columns to remain accessible in data aggregations.

##### Q10: What does the `append=True` parameter do in `df.set_index()`?
- **Direct Answer & Definition**: The `append=True` parameter appends the specified new columns to the existing index rather than overwriting it, creating additional inner hierarchical levels.
- **Technical & Computational Reasoning**: Pandas takes the existing `Index` or `MultiIndex`, combines its levels with the new column arrays, and constructs a deeper MultiIndex containing $L_{old} + L_{new}$ levels.
- **Practical Application / Syntax**: `df.set_index('Transaction_ID', append=True)` retains the existing `Date` index and adds `Transaction_ID` as level 1.
- **Nuances, Defaults & Pitfalls**: If `append=False` (default), the entire existing index is completely overwritten and permanently discarded unless saved beforehand.

##### Q11: How do you access a specific row using a tuple key in a MultiIndex Series?
- **Direct Answer & Definition**: You access a row by passing a complete tuple of coordinates matching the hierarchy of levels to square bracket indexing `s[(key1, key2)]` or `.loc[(key1, key2)]`.
- **Technical & Computational Reasoning**: Pandas hashes the composite tuple, locates the matching offset using internal integer codes, and extracts the scalar value.
- **Practical Application / Syntax**:
  ```python
  val = s[('USA', 'California')]
  val_loc = s.loc[('USA', 'California')]
  ```
- **Nuances, Defaults & Pitfalls**: In Python, `s['USA', 'California']` is syntactic sugar for `s[('USA', 'California')]`. However, passing a list `s[['USA', 'California']]` triggers fancy indexing over level 0, looking for two separate outer keys.

##### Q12: How do you select an entire outer-level category in a MultiIndex DataFrame using `.loc`?
- **Direct Answer & Definition**: You pass the scalar label of the outer level directly to `.loc[outer_key]`.
- **Technical & Computational Reasoning**: Pandas performs partial indexing on the primary level (level 0), slicing the contiguous block corresponding to that key and returning a DataFrame with the outer level dropped.
- **Practical Application / Syntax**:
  ```python
  us_data = df.loc['USA']
  ```
- **Nuances, Defaults & Pitfalls**: This operation only works cleanly for the **outermost** level. Selecting an inner-level key directly with `.loc['California']` raises a `KeyError`.

##### Q13: Why does `df.loc[('A', 'B')]` behave differently from `df.loc[['A', 'B']]`?
- **Direct Answer & Definition**: `df.loc[('A', 'B')]` selects a single row defined by the hierarchical coordinate tuple `(level0='A', level1='B')`, whereas `df.loc[['A', 'B']]` selects all rows where level 0 is either `'A'` OR `'B'`.
- **Technical & Computational Reasoning**: In Python, a tuple `(...)` denotes a single composite dimensional coordinate, whereas a list `[...]` represents a collection of independent outer keys to retrieve.
- **Practical Application / Syntax**: Always use parentheses for coordinate tuples and square brackets for multi-key list selection: `df.loc[[('A', 1), ('B', 2)]]` retrieves two specific composite rows.
- **Nuances, Defaults & Pitfalls**: Passing a list when intending a tuple key results in unexpected shapes or `KeyError` if the second element does not exist in level 0.

##### Q14: What is `pd.IndexSlice` and why is it preferred over native Python `slice()` objects?
- **Direct Answer & Definition**: `pd.IndexSlice` is a helper object that enables natural, idiomatic Python slice syntax (`:`) across multiple dimensions and levels inside `.loc`.
- **Technical & Computational Reasoning**: Standard Python syntax does not permit writing colons `:` inside arbitrary function arguments or nested tuples without custom slice builders. `pd.IndexSlice` overrides `__getitem__` to convert slice notation into native Python `slice` objects.
- **Practical Application / Syntax**:
  ```python
  idx = pd.IndexSlice
  df.loc[idx[:, '2023'], idx['Revenue', :]]
  ```
- **Nuances, Defaults & Pitfalls**: Using `pd.IndexSlice` requires the MultiIndex to be sorted; slicing an unsorted index triggers an `UnsortedIndexError`.

##### Q15: What is an `UnsortedIndexError` and why does slicing a MultiIndex require sorting?
- **Direct Answer & Definition**: An `UnsortedIndexError` is an exception raised when attempting to slice a MultiIndex that is not monotonically sorted in lexicographical order.
- **Technical & Computational Reasoning**: Slicing an index range ($K_1$ to $K_2$) requires $O(\log N)$ binary search to identify memory start and end offsets. If the index is unsorted, binary search fails, and scanning would require an expensive $O(N)$ full table scan, which Pandas actively forbids during range slicing.
- **Practical Application / Syntax**: Fix the error by sorting the index beforehand:
  ```python
  df = df.sort_index()
  df.loc[idx['A':'C', :], :]
  ```
- **Nuances, Defaults & Pitfalls**: Point lookups (`df.loc[('A', 1)]`) can succeed on unsorted indices via hash tables, masking the fact that range slices on the same index will fail.

##### Q16: What is the purpose of the `.xs()` method in Pandas?
- **Direct Answer & Definition**: The `.xs()` (cross-section) method extracts a slice of data from a Series or DataFrame at a specific hierarchical level, regardless of its position in the hierarchy.
- **Technical & Computational Reasoning**: It eliminates the need to specify placeholders or slices for outer levels when targeting an inner level, filtering directly on the target level code array.
- **Practical Application / Syntax**:
  ```python
  # Select all rows where level 'Year' == 2023, across all countries and cities:
  df_2023 = df.xs(2023, level='Year')
  ```
- **Nuances, Defaults & Pitfalls**: `.xs()` is strictly a **read-only** extraction tool; attempting to assign values to an `.xs()` call (`df.xs(...) = 10`) fails because `.xs()` returns a copy or view that cannot propagate in-place writes.

##### Q17: What does the `level` parameter in `.xs()` control?
- **Direct Answer & Definition**: The `level` parameter specifies the target hierarchical level to filter on, accepted either as an integer level index (0-based) or as a string level name.
- **Technical & Computational Reasoning**: Pandas resolves the level identifier against `index.names` or numeric level bounds, extracting the corresponding 1D code array for matching.
- **Practical Application / Syntax**: `df.xs('Retail', level='Sector')` or `df.xs('Retail', level=1)`.
- **Nuances, Defaults & Pitfalls**: Using explicit level names (strings) is strongly recommended in production code over integer indices to prevent silent pipeline breakage if upstream schemas reorder levels.

##### Q18: What does the `drop_level=False` parameter in `.xs()` do?
- **Direct Answer & Definition**: `drop_level=False` retains the filtered level in the resulting DataFrame's MultiIndex rather than dropping it.
- **Technical & Computational Reasoning**: By default (`drop_level=True`), Pandas removes the queried level from the output index hierarchy. Setting `drop_level=False` preserves the full dimensional depth ($L$ levels), keeping the matched key as a single-value level.
- **Practical Application / Syntax**:
  ```python
  df_retained = df.xs(2023, level='Year', drop_level=False)
  ```
- **Nuances, Defaults & Pitfalls**: Dropping levels can cause unexpected duplicate index entries if the remaining levels do not form a unique composite primary key.

##### Q19: How does `df.reset_index()` handle a MultiIndex, and what happens to the level names?
- **Direct Answer & Definition**: `df.reset_index()` converts the specified MultiIndex levels into standard DataFrame columns and replaces the row index with an integer `RangeIndex`.
- **Technical & Computational Reasoning**: It materializes the full 1D array for each level (`get_level_values`), places them into the DataFrame's data block, and resets the index manager.
- **Practical Application / Syntax**: `df_flat = df.reset_index()` (resets all levels) or `df_flat = df.reset_index(level='Date')` (resets only 'Date').
- **Nuances, Defaults & Pitfalls**: If an index level has no name (`None`), `reset_index()` generates default column names like `'level_0'`, `'level_1'`, which can pollute production data schemas.

##### Q20: What is the difference between `df.reset_index(drop=True)` and `df.droplevel()`?
- **Direct Answer & Definition**: `df.reset_index(drop=True)` completely discards the index levels and replaces the entire index with a `RangeIndex`, whereas `df.droplevel()` removes one specific level while retaining the remaining hierarchical levels.
- **Technical & Computational Reasoning**: `reset_index(drop=True)` resets dimensionality back to a standard 1D RangeIndex; `droplevel()` reduces hierarchy depth from $L$ to $L - 1$ levels.
- **Practical Application / Syntax**:
  ```python
  # Keeps other levels intact:
  df_one_less_level = df.droplevel('Year')
  # Discards all levels:
  df_flat = df.reset_index(drop=True)
  ```
- **Nuances, Defaults & Pitfalls**: `droplevel()` can be executed on either rows (`axis=0`) or columns (`axis=1`), making it the premier method for flattening multi-level column headers.

---

#### B. Intermediate Questions (Exactly 20 Questions)

##### Q1: How does lexicographical sorting (`df.sort_index()`) impact MultiIndex binary search lookup performance?
- **Expected Answer**: Lexicographical sorting organizes keys sequentially level-by-level, enabling Pandas to locate slice boundaries in $O(\log N)$ time using binary search instead of $O(N)$ linear scans.
- **Explanation & Deep-Dive**: Internally, Pandas maintains a flag called `is_monotonic_increasing`. When an index is verified as sorted, range operations (`.loc['A':'C']`) execute two binary searches to find the start and stop row offsets. If unsorted, determining which rows fall within the range would require evaluating every row in the dataset ($O(N)$), which degrades real-time lookup latency.

##### Q2: What is `lexsort_depth` on a MultiIndex and how does Pandas determine if slicing is valid?
- **Expected Answer**: `lexsort_depth` is an integer property indicating the number of contiguous levels from the outermost level inward that are strictly sorted in lexicographical order.
- **Explanation & Deep-Dive**: If an index has 3 levels and `lexsort_depth == 2`, only levels 0 and 1 are sorted; level 2 contains order violations. If a query slices level 0 or level 1, Pandas allows it. If a query attempts to slice across level 2, Pandas detects that the slice depth exceeds `lexsort_depth` and immediately raises an `UnsortedIndexError`.

##### Q3: Can a MultiIndex have duplicate index labels across levels? How does lookup complexity change?
- **Expected Answer**: Yes, MultiIndexes permit duplicate composite keys unless uniqueness is explicitly enforced. Duplicates degrade point lookup performance from $O(1)$ to $O(K)$, where $K$ is the duplicate count.
- **Explanation & Deep-Dive**: When keys are unique, Pandas uses internal hash tables to jump directly to the row offset. When duplicates exist, the lookup must return a slice or boolean mask matching all instances, returning a DataFrame or Series instead of a scalar, which can cause downstream runtime type errors in production pipelines.

##### Q4: How does partial indexing work on a 3-level MultiIndex (`Continent`, `Country`, `City`)?
- **Expected Answer**: Partial indexing allows querying outer levels without specifying inner levels, automatically returning all matching child records.
- **Explanation & Deep-Dive**: Slicing `df.loc['Europe']` extracts all countries and cities in Europe. Slicing `df.loc[('Europe', 'France')]` extracts all cities in France. Partial indexing proceeds strictly from the outermost level inward. Slicing an inner level alone (`df.loc['Paris']`) without specifying outer levels is illegal and raises `KeyError`.

##### Q5: How do you slice inner levels while keeping all outer levels using `.loc` and `pd.IndexSlice`?
- **Expected Answer**: Use `pd.IndexSlice` and pass a colon `:` to represent all outer keys: `df.loc[pd.IndexSlice[:, 'TargetInnerKey'], :]`.
- **Explanation & Deep-Dive**: The colon `:` acts as a wildcard slice (`slice(None)`). This signals to the MultiIndex locator that all values in level 0 are accepted, but only rows matching `'TargetInnerKey'` in level 1 should be selected. The index must be sorted for this operation to succeed.

##### Q6: Why can you NOT use `.xs()` for range slicing (e.g., `df.xs(slice('2020', '2022'))`)?
- **Expected Answer**: `.xs()` is architected strictly for discrete point key matching, not interval or range boundary evaluation.
- **Explanation & Deep-Dive**: Under the hood, `.xs()` uses exact equality matching against the integer codes of the specified level. Passing a `slice` object causes an internal equality check failure or raises `TypeError: unhashable type: 'slice'`. Range intervals must be executed via `.loc` using `pd.IndexSlice`.

##### Q7: Can `.xs()` be used to assign values in-place (e.g., `df.xs('A') = 10`)? Why does it raise an error?
- **Expected Answer**: No, `.xs()` cannot be used for assignment; it raises a Python `SyntaxError: cannot assign to function call`.
- **Explanation & Deep-Dive**: In Python, function/method calls cannot appear on the left-hand side of an assignment operator. Even if called via chaining (`df.xs('A')['Col'] = 10`), it mutates a temporary copy or triggers Pandas' `SettingWithCopyError`. All in-place mutations on MultiIndexes must be performed using `.loc[...] = value`.

##### Q8: What is the performance difference between `.loc[pd.IndexSlice[:, 'B'], :]` and `.xs('B', level=1)`?
- **Expected Answer**: Both locate matching inner keys, but `.loc` with `IndexSlice` is optimized for in-place operations and preserves the index structure, while `.xs()` drops the queried level by default and incurs copy overhead.
- **Explanation & Deep-Dive**: If the index is sorted, `.loc` constructs a view or direct slice over the block. `.xs()` processes level dropping, rebuilding a new index object with $L - 1$ levels, which introduces minor object instantiation latency on large datasets.

##### Q9: How does `df.droplevel()` differ when applied to rows (`axis=0`) versus columns (`axis=1`)?
- **Expected Answer**: Applied to `axis=0`, it simplifies the row hierarchy; applied to `axis=1`, it flattens hierarchical multi-level column names into single-level column headers.
- **Explanation & Deep-Dive**: In ETL pipelines, `.groupby().agg({'price': ['mean', 'std']})` generates a MultiIndex on columns (`('price', 'mean')`, `('price', 'std')`). Calling `df.droplevel(0, axis=1)` strips the outer `'price'` level, leaving clean column names `['mean', 'std']`.

##### Q10: What happens if level names are `None` when calling `df.reset_index()`?
- **Expected Answer**: Pandas automatically assigns generated names formatted as `'level_0'`, `'level_1'`, etc., to the newly created DataFrame columns.
- **Explanation & Deep-Dive**: If a single-level index has no name, it is named `'index'` upon resetting. For a MultiIndex, Pandas avoids name collisions by using the prefix `'level_'` suffixed with the integer depth of that level. In production, this can break schema validation pipelines expecting business-meaningful column names.

##### Q11: How do you select multiple specific outer-level keys along with specific inner-level keys using `.loc`?
- **Expected Answer**: Pass lists of desired keys within the `IndexSlice` tuple: `df.loc[pd.IndexSlice[['US', 'EU'], ['Active', 'Pending']], :]`.
- **Explanation & Deep-Dive**: Pandas evaluates this as a Cartesian subset matching $(Region \in \{US, EU\}) \land (Status \in \{Active, Pending\})$. It uses the internal codes to gather matching row offsets efficiently without evaluating boolean masks across millions of rows.

##### Q12: What is the internal memory representation of `codes` in a MultiIndex (int8 vs int16 vs int32 vs int64)?
- **Expected Answer**: Pandas dynamically downcasts the integer dtype of each level's `codes` array based on the cardinality of unique labels in that level.
- **Explanation & Deep-Dive**: If a level has $< 128$ unique categories, Pandas allocates `int8` (1 byte per row). If cardinality is between 128 and 32,767, it uses `int16` (2 bytes per row). This dtype optimization ensures that hierarchical indices over high-volume data consume minimum possible RAM.

##### Q13: How does `MultiIndex.get_level_values()` work and does it return a copy or a view?
- **Expected Answer**: It returns an `Index` object containing the labels for that level for every row in the DataFrame, reconstructed from `levels` and `codes`.
- **Explanation & Deep-Dive**: In modern Pandas (especially with Copy-on-Write enabled), `get_level_values()` avoids copying the underlying string buffers where possible by referencing the category pool, but it materializes a new 1D `Index` object of length $N$.

##### Q14: How does `.reindex()` behave on a MultiIndex when new level combinations are introduced?
- **Expected Answer**: `.reindex()` conforms the DataFrame to the new MultiIndex, inserting `NaN` (or a specified `fill_value`) for missing coordinate combinations.
- **Explanation & Deep-Dive**: Pandas aligns existing keys with the target MultiIndex. Any coordinate tuple present in the new index but absent in the original DataFrame results in an empty row populated with missing value indicators across all data columns.

##### Q15: How can you swap levels in a MultiIndex using `.swaplevel()`?
- **Expected Answer**: `.swaplevel(i, j, axis=0)` interchanges the positions of level $i$ and level $j$ without altering the underlying data.
- **Explanation & Deep-Dive**: Under the hood, `.swaplevel()` simply swaps the pointers in the `levels` and `codes` lists for the specified levels. However, swapping levels typically renders the MultiIndex unsorted, necessitating an immediate call to `.sort_index()` if subsequent range slicing is required.

##### Q16: What is `.reorder_levels()` and how does it differ from `.swaplevel()`?
- **Expected Answer**: `.swaplevel()` exchanges exactly two levels, whereas `.reorder_levels(order)` permits arbitrary reordering of any number of levels simultaneously.
- **Explanation & Deep-Dive**: `df.reorder_levels(['City', 'Region', 'Country'])` accepts a full permutation of level names or integers, restructuring the hierarchy in a single pass rather than requiring multiple pairwise swaps.

##### Q17: How does `.isin()` operate on a MultiIndex?
- **Expected Answer**: `df.index.isin(values, level=None)` checks for membership of index keys within a given collection of tuples or level-specific scalars.
- **Explanation & Deep-Dive**: If `level=None`, `values` must be an iterable of coordinate tuples matching the full index depth. If `level='LevelName'`, `values` can be a simple list of scalar labels matching that single level, returning a boolean 1D array suitable for row filtering.

##### Q18: Why does passing a single string key to `.loc` on a MultiIndex return a DataFrame, but passing a full tuple returns a Series (or scalar)?
- **Expected Answer**: A single string matches only the outer level, returning the entire subset of inner rows as a DataFrame. A full tuple specifies a complete coordinate, narrowing the selection to a single row (returned as a Series) or a single cell (scalar).
- **Explanation & Deep-Dive**: Pandas indexing dimensionality rules state that reducing all index dimensions returns a scalar; reducing row coordinates to a single row returns a 1D Series; partially specifying coordinates retains remaining dimensional rows, returning a 2D DataFrame.

##### Q19: How do you preserve the MultiIndex structure when performing mathematical operations across DataFrames?
- **Expected Answer**: Pandas automatically aligns DataFrames based on the intersection of their MultiIndex coordinate tuples before executing operations.
- **Explanation & Deep-Dive**: In `df1 + df2`, Pandas does not sum elements by integer position. It aligns rows where `(level0, level1)` match exactly. Any coordinate present in `df1` but absent in `df2` evaluates to `NaN` in the result unless filled using `df1.add(df2, fill_value=0)`.

##### Q20: How does modern Copy-on-Write (CoW) interact with MultiIndex slicing and views?
- **Expected Answer**: Slicing a MultiIndex DataFrame under CoW returns a view referencing the parent memory buffers; the index is shared without defensive copying until either DataFrame is mutated.
- **Explanation & Deep-Dive**: In Pandas 2.0+ with CoW, slicing `sub_df = df.loc['Tech']` does not copy data arrays. If the user mutates `sub_df`, Pandas triggers a deferred copy of only the affected columns, eliminating historical `SettingWithCopyWarning` ambiguities.

---

#### C. Code-Based Questions (Exactly 10 Questions)

##### Q1: MultiIndex Construction from Cartesian Product
```python
import pandas as pd

regions = ['North', 'South']
quarters = ['Q1', 'Q2']
idx = pd.MultiIndex.from_product([regions, quarters], names=['Region', 'Quarter'])
df = pd.DataFrame({'Sales': [100, 150, 200, 250]}, index=idx)
print(df.loc['North', 'Sales'])
```
- **Expected Output**:
  ```text
  Quarter
  Q1    100
  Q2    150
  Name: Sales, dtype: int64
  ```
- **Line-by-Line Explanation**:
  1. `from_product` computes all combinations: `('North', 'Q1')`, `('North', 'Q2')`, `('South', 'Q1')`, `('South', 'Q2')`.
  2. The DataFrame is created with 4 rows indexed by this MultiIndex.
  3. `df.loc['North', 'Sales']` partially indexes the outer level `'North'` and selects the `'Sales'` column.
  4. The outer level `'North'` is dropped, returning a Series indexed by the remaining `'Quarter'` level.

##### Q2: Partial Row Selection and Shape Preservation
```python
import pandas as pd

arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
mi = pd.MultiIndex.from_arrays(arrays, names=['Letter', 'Number'])
df = pd.DataFrame({'Val': [10, 20, 30, 40]}, index=mi)
subset = df.loc[['A']]
print(subset.shape)
print(subset.index.names)
```
- **Expected Output**:
  ```text
  (2, 1)
  ['Letter', 'Number']
  ```
- **Line-by-Line Explanation**:
  1. Passing a list `['A']` instead of a scalar `'A'` preserves the outer index level.
  2. The output retains 2 rows and 1 column (`shape == (2, 1)`).
  3. The index remains a 2-level MultiIndex with names `['Letter', 'Number']`.

##### Q3: Cross-Sectional Selection Across Inner Level
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([
    ('US', 'East'), ('US', 'West'), ('UK', 'South'), ('UK', 'West')
], names=['Country', 'Region'])
df = pd.DataFrame({'Headcount': [500, 300, 200, 150]}, index=mi)
west_data = df.xs('West', level='Region')
print(west_data)
```
- **Expected Output**:
  ```text
           Headcount
  Country           
  US             300
  UK             150
  ```
- **Line-by-Line Explanation**:
  1. `df.xs` targets `level='Region'` and extracts rows where `Region == 'West'`.
  2. Because `drop_level=True` by default, `'Region'` is eliminated from the result.
  3. The output is a DataFrame indexed solely by the remaining level `'Country'`.

##### Q4: Slicing with `pd.IndexSlice`
```python
import pandas as pd

mi = pd.MultiIndex.from_product([['2022', '2023'], ['Tech', 'Health', 'Energy']])
df = pd.DataFrame({'Rev': [10, 20, 30, 40, 50, 60]}, index=mi).sort_index()

idx = pd.IndexSlice
result = df.loc[idx[:, ['Tech', 'Energy']], :]
print(len(result))
```
- **Expected Output**:
  ```text
  4
  ```
- **Line-by-Line Explanation**:
  1. `df.sort_index()` ensures the MultiIndex is lexicographically sorted.
  2. `idx[:, ['Tech', 'Energy']]` specifies all years (`:`) and filters for `'Tech'` and `'Energy'` in level 1.
  3. Out of 6 total rows, 2 categories across 2 years match, yielding exactly 4 rows.

##### Q5: Flattening Multi-Level Column Index
```python
import pandas as pd

columns = pd.MultiIndex.from_tuples([('Metric', 'Mean'), ('Metric', 'Std')])
df = pd.DataFrame([[10.5, 2.1], [20.0, 3.5]], columns=columns)
df.columns = ['_'.join(c) for c in df.columns]
print(df.columns.tolist())
```
- **Expected Output**:
  ```text
  ['Metric_Mean', 'Metric_Std']
  ```
- **Line-by-Line Explanation**:
  1. A DataFrame with a 2-level MultiIndex on columns is instantiated.
  2. A list comprehension iterates over `df.columns`, where each column identifier `c` is a tuple `('Metric', 'Mean')`.
  3. `'_'.join(c)` joins the tuple elements into a flat string, producing clean 1D column names.

##### Q6: Resetting a Specific Level
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('Alpha', 101), ('Alpha', 102), ('Beta', 201)], names=['Group', 'ID'])
df = pd.DataFrame({'Score': [88, 92, 79]}, index=mi)
df_reset = df.reset_index(level='ID')
print(df_reset.columns.tolist())
print(df_reset.index.name)
```
- **Expected Output**:
  ```text
  ['ID', 'Score']
  Group
  ```
- **Line-by-Line Explanation**:
  1. `df.reset_index(level='ID')` targets only the inner level `'ID'`.
  2. `'ID'` is converted into a regular data column alongside `'Score'`.
  3. The remaining level `'Group'` stays as the row index, with its index name preserved.

##### Q7: Dropping Levels Without Resetting
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('A', 'X', 1), ('B', 'Y', 2)], names=['L0', 'L1', 'L2'])
df = pd.DataFrame({'Data': [100, 200]}, index=mi)
df_dropped = df.droplevel(['L0', 'L2'])
print(df_dropped.index.names)
```
- **Expected Output**:
  ```text
  ['L1']
  ```
- **Line-by-Line Explanation**:
  1. `df.droplevel` accepts a list of level names to remove.
  2. Levels `'L0'` and `'L2'` are permanently discarded from the index structure.
  3. The resulting DataFrame has a single-level index named `'L1'`.

##### Q8: The `append=True` Parameter in `set_index`
```python
import pandas as pd

df = pd.DataFrame({'Cat': ['X', 'Y'], 'Subcat': ['a', 'b'], 'Val': [1, 2]})
df = df.set_index('Cat')
df = df.set_index('Subcat', append=True)
print(df.index.nlevels)
print(df.index.names)
```
- **Expected Output**:
  ```text
  2
  ['Cat', 'Subcat']
  ```
- **Line-by-Line Explanation**:
  1. `df.set_index('Cat')` creates a single-level index named `'Cat'`.
  2. `df.set_index('Subcat', append=True)` appends `'Subcat'` to the existing index.
  3. The number of levels (`nlevels`) becomes 2, with names `['Cat', 'Subcat']`.

##### Q9: Unused Levels After Filtering
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['Letter', 'Number'])
df = pd.DataFrame({'Val': [10, 20, 30]}, index=mi)
df_sub = df.loc[['A']]
print(df_sub.index.levels[0].tolist())
df_clean = df_sub.copy()
df_clean.index = df_clean.index.remove_unused_levels()
print(df_clean.index.levels[0].tolist())
```
- **Expected Output**:
  ```text
  ['A', 'B']
  ['A']
  ```
- **Line-by-Line Explanation**:
  1. Slicing rows where `Letter == 'A'` removes `'B'` from the data block.
  2. However, `df_sub.index.levels[0]` still retains `'B'` because `levels` are not recomputed for performance.
  3. Calling `.remove_unused_levels()` recalculates the unique levels, purging `'B'`.

##### Q10: Index Swapping with `.swaplevel()`
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([(1, 'A'), (2, 'B')], names=['Num', 'Alpha'])
df = pd.DataFrame({'Val': [10, 20]}, index=mi)
df_swapped = df.swaplevel('Num', 'Alpha')
print(df_swapped.index.names)
print(df_swapped.index.tolist())
```
- **Expected Output**:
  ```text
  ['Alpha', 'Num']
  [('A', 1), ('B', 2)]
  ```
- **Line-by-Line Explanation**:
  1. `swaplevel` exchanges the position of the `'Num'` and `'Alpha'` levels.
  2. The level names order becomes `['Alpha', 'Num']`.
  3. Every coordinate tuple is inverted from `(num, alpha)` to `(alpha, num)`.

---

#### D. Output-Prediction Questions (6 Questions)

##### Q1: Slicing with Missing Intermediate Coordinate
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['L1', 'L2'])
df = pd.DataFrame({'Val': [10, 20, 30]}, index=mi)
try:
    print(df.loc[('A', 3)])
except Exception as e:
    print(type(e).__name__)
```
- **Candidate Prompt**: What is printed to the console?
- **Exact Output**:
  ```text
  KeyError
  ```
- **Step-by-Step Trace**:
  1. Pandas hashes the tuple coordinate `('A', 3)`.
  2. While `'A'` exists in level 0, the combination `('A', 3)` does not exist in the index.
  3. Because this is an exact coordinate point lookup rather than a range slice, Pandas raises a `KeyError`.

##### Q2: Scalar vs List Selection Dimensionality
```python
import pandas as pd

mi = pd.MultiIndex.from_product([['X'], [1, 2]])
df = pd.DataFrame({'Val': [100, 200]}, index=mi)
res1 = df.loc['X']
res2 = df.loc[['X']]
print(type(res1).__name__, res1.shape)
print(type(res2).__name__, res2.shape)
```
- **Candidate Prompt**: What are the returned types and shapes of `res1` and `res2`?
- **Exact Output**:
  ```text
  DataFrame (2, 1)
  DataFrame (2, 1)
  ```
- **Step-by-Step Trace**:
  1. Passing scalar `'X'` selects all matching rows in level 0 and drops level 0, returning a DataFrame with shape `(2, 1)` indexed by `[1, 2]`.
  2. Passing list `['X']` selects all matching rows but preserves level 0, returning a DataFrame with shape `(2, 1)` indexed by a 2-level MultiIndex `[('X', 1), ('X', 2)]`. Both are DataFrames, but their indices have different dimensionalities.

##### Q3: Unsorted MultiIndex Slicing
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('B', 2), ('A', 1), ('B', 1)], names=['L0', 'L1'])
df = pd.DataFrame({'Val': [1, 2, 3]}, index=mi)
try:
    print(df.loc['A':'B'])
except Exception as e:
    print(type(e).__name__)
```
- **Candidate Prompt**: Does this slice succeed or fail? What exception is printed?
- **Exact Output**:
  ```text
  UnsortedIndexError
  ```
- **Step-by-Step Trace**:
  1. The MultiIndex starts with `'B'`, drops to `'A'`, then returns to `'B'`. It is not monotonic.
  2. `df.loc['A':'B']` attempts a range slice.
  3. Pandas checks `is_monotonic_increasing`, discovers the order violation, and raises `UnsortedIndexError`.

##### Q4: Preserving Levels with `drop_level=False` in `.xs()`
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('East', 2022), ('East', 2023)], names=['Region', 'Year'])
df = pd.DataFrame({'Rev': [100, 150]}, index=mi)
out = df.xs(2022, level='Year', drop_level=False)
print(out.index.nlevels)
print(out.index.names)
```
- **Candidate Prompt**: What is the number of levels and level names in `out`?
- **Exact Output**:
  ```text
  2
  ['Region', 'Year']
  ```
- **Step-by-Step Trace**:
  1. `df.xs` filters for `Year == 2022`.
  2. Because `drop_level=False`, the `'Year'` level is retained as a single-element category.
  3. The resulting index maintains `nlevels == 2` with names `['Region', 'Year']`.

##### Q5: Selecting Non-Existent Column in MultiIndex DataFrame
```python
import pandas as pd

mi = pd.MultiIndex.from_tuples([('US', 'CA'), ('US', 'NY')])
df = pd.DataFrame({'Sales': [10, 20]}, index=mi)
try:
    print(df['US'])
except Exception as e:
    print(type(e).__name__)
```
- **Candidate Prompt**: Does `df['US']` select the US rows?
- **Exact Output**:
  ```text
  KeyError
  ```
- **Step-by-Step Trace**:
  1. In a DataFrame, single bracket notation `df[key]` selects **columns**, not rows.
  2. The DataFrame columns are `['Sales']`.
  3. Looking up `'US'` searches for a column named `'US'`, fails, and raises `KeyError: 'US'`. To select rows, one must use `df.loc['US']`.

##### Q6: MultiIndex Alignment During Arithmetic Addition
```python
import pandas as pd

s1 = pd.Series([10, 20], index=pd.MultiIndex.from_tuples([('A', 1), ('A', 2)]))
s2 = pd.Series([100, 200], index=pd.MultiIndex.from_tuples([('A', 2), ('A', 3)]))
s3 = s1 + s2
print(s3.isna().sum())
print(s3.loc[('A', 2)])
```
- **Candidate Prompt**: How many NaN values are produced, and what is the value at `('A', 2)`?
- **Exact Output**:
  ```text
  2
  120.0
  ```
- **Step-by-Step Trace**:
  1. Pandas aligns `s1` and `s2` along the union of their MultiIndex coordinates: `[('A', 1), ('A', 2), ('A', 3)]`.
  2. Coordinate `('A', 1)` is missing in `s2` $\rightarrow 10 + \text{NaN} = \text{NaN}$.
  3. Coordinate `('A', 3)` is missing in `s1` $\rightarrow \text{NaN} + 200 = \text{NaN}$.
  4. Total NaNs $= 2$. At coordinate `('A', 2)`, both series have values: $20 + 100 = 120.0$.

---

#### E. Debugging Questions (Exactly 10 Questions)

##### Q1: The Ambiguous Row Slicing Bug
- **Problem Statement**: A junior developer tries to slice rows between years 2021 and 2023 for region `'US'` and receives an `UnsortedIndexError`.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_tuples([('US', 2023), ('US', 2021), ('EU', 2022)], names=['Region', 'Year'])
  df = pd.DataFrame({'Sales': [30, 10, 20]}, index=mi)
  sub = df.loc[('US', 2021):('US', 2023)]
  ```
- **Expected Correction**:
  ```python
  df_sorted = df.sort_index()
  sub = df_sorted.loc[('US', 2021):('US', 2023)]
  ```
- **Root-Cause Explanation**: The MultiIndex was constructed in non-sorted order (`2023` before `2021`). Range slicing requires lexicographical order to execute binary search boundaries. Sorting the index via `sort_index()` resolves the issue.

##### Q2: The List vs. Tuple Bracket Bug
- **Problem Statement**: The developer attempts to query a specific point coordinate `('East', 101)` but encounters a `KeyError: '101'`.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_tuples([('East', 101), ('East', 102)], names=['Zone', 'Store'])
  s = pd.Series([500, 600], index=mi)
  val = s[['East', 101]]
  ```
- **Expected Correction**:
  ```python
  val = s[('East', 101)]
  ```
- **Root-Cause Explanation**: Double brackets `s[['East', 101]]` pass a Python list, which Pandas interprets as fancy indexing requesting two separate outer keys: `'East'` and `101`. Because `101` does not exist in level 0, Pandas throws a `KeyError`. Coordinates must be passed as tuples `('East', 101)`.

##### Q3: The In-Place Mutation on `.xs()` Failure
- **Problem Statement**: An engineer attempts to update sales figures for the `'West'` region using `.xs()` and receives a `SyntaxError` or notices that data in `df` was not modified.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_tuples([('US', 'West'), ('US', 'East')])
  df = pd.DataFrame({'Sales': [100, 200]}, index=mi)
  df.xs('West', level=1)['Sales'] = 999
  ```
- **Expected Correction**:
  ```python
  idx = pd.IndexSlice
  df.loc[idx[:, 'West'], 'Sales'] = 999
  ```
- **Root-Cause Explanation**: `.xs()` returns a newly materialized DataFrame view/copy with the queried level dropped. Chained assignment on this object modifies only the temporary object and triggers a `SettingWithCopyWarning`, leaving the parent `df` unchanged. `.loc` with `pd.IndexSlice` must be used for in-place updates.

##### Q4: Column Name Collision on `reset_index()`
- **Problem Statement**: Calling `reset_index()` raises `ValueError: cannot insert Date, already exists`.
- **Buggy Code**:
  ```python
  import pandas as pd
  df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02'], 'Revenue': [100, 200]})
  df.index = pd.MultiIndex.from_tuples([('2023-01-01', 'A'), ('2023-01-02', 'B')], names=['Date', 'Cat'])
  df_reset = df.reset_index()
  ```
- **Expected Correction**:
  ```python
  # Rename or drop the conflicting column prior to reset:
  df = df.drop(columns=['Date'])
  df_reset = df.reset_index()
  ```
- **Root-Cause Explanation**: The index level is named `'Date'`, but the DataFrame data block already contains a column named `'Date'`. When `reset_index()` attempts to insert the index level into the columns, a name collision occurs.

##### Q5: Ambiguous Slicing without `pd.IndexSlice`
- **Problem Statement**: An engineer writes `df.loc[:, 'B']` expecting to select all rows in level 0 and filter level 1 for `'B'`, but receives `KeyError: 'B'`.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_product([['X', 'Y'], ['A', 'B']])
  df = pd.DataFrame({'Val': [1, 2, 3, 4]}, index=mi)
  res = df.loc[:, 'B']
  ```
- **Expected Correction**:
  ```python
  idx = pd.IndexSlice
  res = df.loc[idx[:, 'B'], :]
  ```
- **Root-Cause Explanation**: In `df.loc[row_sel, col_sel]`, the comma separates the row axis from the column axis. Writing `df.loc[:, 'B']` tells Pandas to select all rows (`:`) and select a column named `'B'`. Because `'B'` is a row index level, not a column, a `KeyError` is raised.

##### Q6: Multi-Level Column Slicing Failure
- **Problem Statement**: After grouping, a developer tries to access the `'mean'` column of `'Price'` and gets a `KeyError: 'mean'`.
- **Buggy Code**:
  ```python
  import pandas as pd
  df = pd.DataFrame({'Category': ['A', 'A', 'B'], 'Price': [10, 20, 30]})
  agg_df = df.groupby('Category').agg({'Price': ['mean', 'max']})
  mean_val = agg_df['mean']
  ```
- **Expected Correction**:
  ```python
  mean_val = agg_df[('Price', 'mean')]
  # Or flatten columns:
  agg_df.columns = agg_df.columns.droplevel(0)
  mean_val = agg_df['mean']
  ```
- **Root-Cause Explanation**: Aggregations produce a 2-level MultiIndex on columns. Column `'mean'` is nested under `'Price'`. Accessing it requires specifying the full tuple `('Price', 'mean')` or dropping the outer level.

##### Q7: Accidental Cartesian Product Duplication
- **Problem Statement**: A developer creates a MultiIndex from two lists expecting paired combinations `('A', 1)` and `('B', 2)`, but gets 4 rows instead of 2.
- **Buggy Code**:
  ```python
  import pandas as pd
  letters = ['A', 'B']
  numbers = [1, 2]
  mi = pd.MultiIndex.from_product([letters, numbers])
  ```
- **Expected Correction**:
  ```python
  mi = pd.MultiIndex.from_arrays([letters, numbers])
  # Or:
  mi = pd.MultiIndex.from_tuples(zip(letters, numbers))
  ```
- **Root-Cause Explanation**: `from_product` calculates the Cartesian product ($2 \times 2 = 4$ pairs). To create element-wise paired coordinates, `from_arrays` or `from_tuples(zip(...))` must be utilized.

##### Q8: The Silent Index Misalignment NaN Trap
- **Problem Statement**: Merging or adding two MultiIndex Series produces a Series full of `NaN` values.
- **Buggy Code**:
  ```python
  import pandas as pd
  s1 = pd.Series([1, 2], index=pd.MultiIndex.from_tuples([('US', 2021), ('US', 2022)]))
  s2 = pd.Series([10, 20], index=pd.MultiIndex.from_tuples([(2021, 'US'), (2022, 'US')]))
  s3 = s1 + s2
  ```
- **Expected Correction**:
  ```python
  s2_aligned = s2.swaplevel(0, 1)
  s3 = s1 + s2_aligned
  ```
- **Root-Cause Explanation**: In `s1`, level 0 is string country and level 1 is integer year. In `s2`, the level order is reversed. Coordinate `('US', 2021)` never matches `(2021, 'US')`, causing index alignment to fail completely and produce all `NaN` values.

##### Q9: Unused Levels Memory and Plotting Bug
- **Problem Statement**: After filtering a MultiIndex DataFrame to a single category, a categorical plot or `levels` inspection still displays all discarded categories.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_product([['Active', 'Closed'], [1, 2, 3]], names=['Status', 'ID'])
  df = pd.DataFrame({'Val': range(6)}, index=mi)
  active_df = df.loc['Active']
  # active_df.index still references 'Closed' in internal levels
  ```
- **Expected Correction**:
  ```python
  active_df = df.loc[['Active']].copy()
  active_df.index = active_df.index.remove_unused_levels()
  ```
- **Root-Cause Explanation**: For performance reasons, subsetting an index does not recreate or filter the underlying unique `levels` array. Unused categories remain stored until explicitly purged with `remove_unused_levels()`.

##### Q10: Type Mismatch in MultiIndex Tuple Lookup
- **Problem Statement**: A point lookup raises `KeyError: ('A', '1')` even though the data appears to have `'A'` and `1`.
- **Buggy Code**:
  ```python
  import pandas as pd
  mi = pd.MultiIndex.from_tuples([('A', 1), ('B', 2)], names=['Char', 'Num'])
  df = pd.DataFrame({'Score': [100, 200]}, index=mi)
  val = df.loc[('A', '1')]
  ```
- **Expected Correction**:
  ```python
  val = df.loc[('A', 1)]  # Pass integer 1, not string '1'
  ```
- **Root-Cause Explanation**: Pandas MultiIndex coordinates enforce strict data type equality. Level `'Num'` has dtype `int64`. Looking up string `'1'` causes a hash mismatch against integer `1`, triggering a `KeyError`.

---

#### F. Scenario-Based Questions (4 Questions)

##### Scenario 1: Financial Multi-Asset Risk Desk
- **Business Problem**: A quantitative risk desk ingests real-time ticks for 500 equities across 4 trading desks. Data is stored in a DataFrame with a MultiIndex: `['Desk', 'Ticker', 'Timestamp']`. The desk must extract all ticks for `'Desk_Alpha'` between 09:30:00 and 10:00:00 for tickers `['AAPL', 'MSFT']` with sub-millisecond latency.
- **Solution Code**:
  ```python
  import pandas as pd

  # Ensure pre-sorted index for O(log N) binary search
  df_sorted = df.sort_index()

  idx = pd.IndexSlice
  start_time = pd.Timestamp('2023-10-01 09:30:00')
  end_time = pd.Timestamp('2023-10-01 10:00:00')

  # Efficient multidimensional slice
  ticks = df_sorted.loc[
      idx['Desk_Alpha', ['AAPL', 'MSFT'], start_time:end_time],
      ['Price', 'Volume']
  ]
  ```
- **Engineering Rationale**: Pre-sorting via `sort_index()` allows `pd.IndexSlice` to utilize binary search across the outer desk and timestamp intervals simultaneously. Passing a list `['AAPL', 'MSFT']` inside level 1 selectively retrieves only target assets without materializing slow boolean masks over millions of rows.

##### Scenario 2: E-Commerce Regional Inventory Rebalancing
- **Business Problem**: An e-commerce warehouse platform stores inventory across a 3-level MultiIndex: `['Country', 'State', 'Warehouse_ID']`. During supply chain disruptions, the logistics team needs to calculate the total available stock for all warehouses across every state within `'USA'` and `'Canada'` without flattening the DataFrame.
- **Solution Code**:
  ```python
  import pandas as pd

  # Calculate total stock per Country and State across all Warehouses
  regional_stock = inventory_df.groupby(level=['Country', 'State'])['Stock'].sum()

  # Filter for target countries using partial indexing
  north_america_stock = regional_stock.loc[['USA', 'Canada']]
  ```
- **Engineering Rationale**: Instead of calling `reset_index()` (which incurs costly column allocations and string copying), passing `level=['Country', 'State']` directly to `.groupby()` performs aggregation natively across index levels in Cython. Slicing with `.loc[['USA', 'Canada']]` retains hierarchical structure while extracting the exact target sub-tree.

##### Scenario 3: IoT Sensor Telemetry Pipeline
- **Business Problem**: An industrial IoT pipeline monitors 10,000 vibration sensors organized hierarchically by `['Plant_ID', 'Line_ID', 'Machine_ID', 'Sensor_ID']`. An anomaly detection model requires extracting only `'Sensor_ID_99'` readings across all plants and machines, and converting the result into a flat 2D matrix for Scikit-Learn inference.
- **Solution Code**:
  ```python
  import pandas as pd

  # 1. Fast cross-sectional slice across innermost level
  sensor_99_df = telemetry_df.xs('Sensor_ID_99', level='Sensor_ID')

  # 2. Reset index to normalize coordinates into standard ML feature matrix
  ml_features = sensor_99_df.reset_index()
  X = ml_features[['Vibration_X', 'Vibration_Y', 'Temperature']].to_numpy()
  ```
- **Engineering Rationale**: `.xs()` cleanly bypasses the 3 outer levels (`Plant_ID`, `Line_ID`, `Machine_ID`) to isolate `'Sensor_ID_99'` without tedious wildcard slice syntax. Calling `reset_index()` denormalizes remaining hierarchy metadata into standard columns before `.to_numpy()` extracts contiguous C-arrays for Scikit-Learn.

##### Scenario 4: Clinical Trial Longitudinal Cohort Analysis
- **Business Problem**: A pharmaceutical research team tracks biomarker responses across patient cohorts with a MultiIndex `['Trial_Phase', 'Cohort', 'Patient_UUID', 'Visit_Num']`. A regulatory audit requires dropping the `'Trial_Phase'` level (which is constant for Phase III), resetting `'Patient_UUID'` to standard column format, while keeping `'Cohort'` and `'Visit_Num'` as row coordinates.
- **Solution Code**:
  ```python
  import pandas as pd

  # Step 1: Permanently drop unneeded constant level
  df_step1 = trial_df.droplevel('Trial_Phase', axis=0)

  # Step 2: Move Patient_UUID out of index into DataFrame body
  df_final = df_step1.reset_index(level='Patient_UUID')
  ```
- **Engineering Rationale**: Using `droplevel('Trial_Phase')` immediately purges the invariant level from memory without creating dummy columns. Following with `reset_index(level='Patient_UUID')` surgically moves the patient identifier into the feature block while preserving `['Cohort', 'Visit_Num']` as the composite index for time-series alignment.
"""

with open(r"d:\Data\GitHub\AI-ML-Study-Notes\syllabus\01_Data_Science_Foundations\Pandas\23\23.1.md", "w", encoding="utf-8") as f:
    f.write(p1_content.strip() + "\n\n")

print("Phase 1 written successfully.")
