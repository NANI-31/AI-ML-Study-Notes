# Pandas for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Pandas for Machine Learning

### 1.1 What is Pandas?
- Introduction to tabular data handling
- Why Pandas is important in Machine Learning
- Pandas vs NumPy — when to use which
- Role of Pandas in the ML ecosystem
  - Data ingestion
  - Data cleaning
  - Feature engineering
  - Exploratory Data Analysis (EDA)
  - Data preprocessing before model training

### 1.2 Why ML Engineers Use Pandas
- Structured data manipulation
- Handling heterogeneous data types
- Seamless integration with Scikit-learn
- Efficient missing data handling
- Fast aggregation and grouping
- Built-in visualization support

### 1.3 Installing and Setup
- Installing Pandas
- Jupyter Notebook / Google Colab setup
- Import conventions
  - `import pandas as pd`
  - `import numpy as np`

---

## 2. Pandas Fundamentals for ML

### 2.1 Core Data Structures
- Series — 1D labeled array
- DataFrame — 2D tabular structure
- Index — labels for rows and columns
- Importance of indexed data in ML pipelines

### 2.2 Loading ML Datasets
- `pd.read_csv()` — loading structured datasets
- `pd.read_excel()`
- `pd.read_json()`
- `pd.read_parquet()`
- Loading from URLs
- Reading datasets from Scikit-learn and converting to DataFrame

### 2.3 Data Types in ML Context
- Integer (`int64`) — count features
- Float (`float64`) — continuous features
- Boolean — binary features
- Object (`str`) — categorical features
- Category — memory-efficient categorical
- Datetime — time-based features
- `astype()` — type conversion for ML readiness

---

## 3. Exploratory Data Analysis (EDA) for ML

### 3.1 Initial Data Inspection
- `head()` and `tail()`
- `shape` — number of samples and features
- `info()` — column types and non-null counts
- `describe()` — statistical summary
- `dtypes` — column types overview
- `columns` — feature names

### 3.2 Understanding Feature Distributions
- `value_counts()` — class distribution for classification
- `unique()` — unique values per feature
- `nunique()` — cardinality of features
- Histogram of features
- `df.plot.hist()`
- `describe()` for numerical features

### 3.3 Target Variable Analysis
- Class balance check — `value_counts(normalize=True)`
- Target distribution visualization
- Imbalanced dataset detection

### 3.4 Correlation Analysis
- `df.corr()` — Pearson correlation matrix
- `df.corr(method='spearman')` — Spearman correlation
- Identifying highly correlated features
- Correlation with target variable
- Heatmap visualization with Seaborn

### 3.5 Outlier Detection
- Z-score method using `mean()` and `std()`
- IQR method using `quantile()`
- Detecting outliers per feature
- Visualizing outliers with box plots

---

## 4. Data Cleaning for ML

### 4.1 Handling Missing Values
- `isnull().sum()` — count missing per feature
- Missing percentage calculation
- `dropna()` — dropping missing rows
- `fillna()` — filling with mean, median, mode
- `fillna(method='ffill')` — forward fill for sequential data
- `fillna(method='bfill')` — backward fill
- `interpolate()` — linear interpolation
- Strategies: drop, impute, predict

### 4.2 Removing Duplicates
- `duplicated()` — detecting duplicate rows
- `drop_duplicates()` — removing duplicates
- Keeping first or last occurrence
- Duplicate detection on subset of columns

### 4.3 Fixing Data Types
- Identifying wrong dtypes
- Converting string numbers to numeric — `pd.to_numeric()`
- Fixing boolean columns
- Parsing dates — `pd.to_datetime()`
- Handling mixed-type columns

### 4.4 Fixing Categorical Inconsistencies
- Lowercase standardization — `str.lower()`
- Strip whitespace — `str.strip()`
- Replacing misspelled categories — `replace()`
- Merging similar categories

### 4.5 Handling Outliers
- Removing outliers using boolean masks
- Capping outliers — Winsorization
- Log transformation to reduce outlier effect

---

## 5. Feature Engineering with Pandas

### 5.1 Creating New Features
- Arithmetic operations between columns
- Ratio features
- Polynomial features manually
- Interaction features
- Log transformation — `np.log()`
- Square root transformation

### 5.2 Binning and Discretization
- `pd.cut()` — equal-width binning
- `pd.qcut()` — quantile-based binning
- Creating ordinal features from continuous values
- Custom bin labels

### 5.3 Encoding Categorical Features
- Label encoding using `map()` or `replace()`
- One-hot encoding — `pd.get_dummies()`
- `drop_first=True` — avoiding dummy variable trap
- Ordinal encoding using `pd.Categorical`
- Frequency encoding — encoding by category count

### 5.4 Datetime Feature Engineering
- Extracting year, month, day — `.dt.year`, `.dt.month`, `.dt.day`
- Extracting hour, minute — `.dt.hour`, `.dt.minute`
- Day of week — `.dt.dayofweek`
- Quarter — `.dt.quarter`
- Is weekend feature — custom lambda
- Days since reference date feature
- Time difference features

### 5.5 Aggregation Features
- GroupBy aggregation per entity
- `groupby().transform()` — group statistics as row-level features
- Rolling window features — `rolling().mean()`
- Cumulative features — `cumsum()`, `cummax()`

### 5.6 Text Feature Basics
- String length feature — `str.len()`
- Word count feature
- Presence of keyword — `str.contains()`
- Character count features

---

## 6. Feature Scaling and Normalization with Pandas

### 6.1 Min-Max Normalization
- Formula: `(x - min) / (max - min)`
- Manual implementation using Pandas
- Scaling to [0, 1] range

### 6.2 Standardization (Z-score)
- Formula: `(x - mean) / std`
- Manual implementation using `mean()` and `std()`

### 6.3 Robust Scaling
- Using median and IQR
- Formula: `(x - median) / IQR`
- Useful with outliers

### 6.4 Log Transformation
- `np.log1p()` for skewed distributions
- When to apply log transformation
- Effect on ML model performance

---

## 7. Handling Class Imbalance with Pandas

### 7.1 Detecting Class Imbalance
- `value_counts()` — class distribution
- Class ratio calculation
- Visualization of class distribution

### 7.2 Oversampling
- Duplicating minority class rows manually
- Random oversampling using `sample()`

### 7.3 Undersampling
- `sample()` — random undersampling of majority class
- Stratified sampling

### 7.4 Combining Oversampling and Undersampling
- Manual strategy using Pandas boolean indexing
- Combining subsets using `pd.concat()`

---

## 8. Grouping and Aggregation for ML

### 8.1 GroupBy for Feature Engineering
- `groupby()` — group by categorical feature
- `agg()` — compute multiple statistics per group
- `transform()` — bring group statistics back to row level
- Per-group mean, std, min, max, count

### 8.2 Pivot Tables for Analysis
- `pivot_table()` — cross-analysis of features
- Identifying feature interactions
- Aggregating target by feature groups

### 8.3 Cross-tabulation
- `pd.crosstab()` — feature vs target frequency
- Normalized cross-tabulation
- Detecting feature relevance for classification

---

## 9. Data Splitting with Pandas

### 9.1 Train-Test Split Manually
- `sample(frac=0.8)` — random 80% training split
- `drop()` — creating test set from remaining rows
- Resetting index after split

### 9.2 Stratified Splitting
- `groupby().apply(sample)` — stratified sampling by class
- Maintaining class balance in train and test sets

### 9.3 Cross-Validation Data Preparation
- Creating folds using `np.array_split()`
- Indexing folds with Pandas

### 9.4 Preparing Feature Matrix and Target Vector
- `df.drop(columns=['target'])` — feature matrix X
- `df['target']` — target vector y
- `df.to_numpy()` — converting to NumPy for Scikit-learn

---

## 10. Merging and Combining ML Datasets

### 10.1 Combining Datasets
- `pd.concat()` — combining train and test for preprocessing
- Resetting index after concat
- Splitting back after shared preprocessing

### 10.2 Merging Feature Tables
- `pd.merge()` — joining datasets on a key
- Left join for adding features from another table
- Many-to-one merges for feature enrichment

### 10.3 Handling Dataset Joins
- Inner join — keeping matched records
- Left join — keeping all base records
- Outer join — union of datasets

---

## 11. Time Series Data for ML

### 11.1 Handling Time Series
- Converting to DatetimeIndex — `pd.to_datetime()`
- `set_index()` with datetime column
- Sorting by date — `sort_index()`

### 11.2 Feature Extraction from Time Series
- Lag features — `shift()`
- Rolling mean and std — `rolling().mean()`
- Expanding features — `expanding().mean()`
- Exponentially weighted features — `ewm().mean()`
- Day/week/month/quarter extraction

### 11.3 Resampling
- `resample()` — aggregating by time period
- Upsampling and downsampling
- Frequency aliases (`D`, `W`, `M`, `Q`, `A`)
- Filling resampled NaN values

---

## 12. Working with Large Datasets

### 12.1 Chunked Reading
- `pd.read_csv(chunksize=N)` — reading in chunks
- Processing each chunk iteratively
- Aggregating results across chunks

### 12.2 Memory Optimization
- Downcasting numeric columns
- Using `category` dtype for low-cardinality strings
- `pd.read_csv(dtype=...)` — specifying dtypes upfront
- `usecols` — reading only required columns
- `pd.read_parquet()` — columnar, memory-efficient format

### 12.3 Efficient Operations
- `eval()` — fast arithmetic on large DataFrames
- `query()` — fast filtering
- Vectorized string operations over `apply()`

---

## 13. Pandas with Scikit-learn

### 13.1 Preparing Data for Scikit-learn
- Converting DataFrame to NumPy arrays
- Separating features and target
- Ensuring no NaN values before model fitting
- Correct dtypes for all features

### 13.2 Pipeline Integration
- Using Pandas in Scikit-learn pipelines
- `ColumnTransformer` with Pandas column names
- `FunctionTransformer` for Pandas operations

### 13.3 Result Analysis
- Storing predictions back into DataFrame
- `pd.DataFrame(predictions)` — creating result DataFrames
- Confusion matrix, classification report formatting

---

## 14. Pandas with Other ML Libraries

### 14.1 Pandas + NumPy
- `df.to_numpy()` — converting to NumPy for computation
- NumPy operations on Pandas columns
- Creating features using NumPy functions

### 14.2 Pandas + Matplotlib / Seaborn
- Feature distribution plots
- Correlation heatmaps
- Target vs feature plots
- Training curve visualization

### 14.3 Pandas + XGBoost / LightGBM
- Passing DataFrames directly to gradient boosting models
- Feature names preservation
- Feature importance extraction back to DataFrame

### 14.4 Pandas + TensorFlow / PyTorch
- Converting Pandas DataFrames to TensorFlow Datasets
- Converting to PyTorch tensors via NumPy
- Batching using Pandas and TensorFlow `from_tensor_slices()`

---

## 15. Pandas for ML Algorithm Foundations

### 15.1 Linear Regression Data Preparation
- Selecting numerical features
- Handling multicollinearity using correlation matrix
- Creating polynomial features manually

### 15.2 Classification Data Preparation
- Label encoding target variable
- One-hot encoding for multi-class
- Class weight calculation for imbalanced data

### 15.3 Clustering Data Preparation
- Feature selection for clustering
- Scaling features before K-Means
- Analyzing cluster assignments in DataFrame

### 15.4 Recommendation Systems Data Preparation
- User-item matrix construction with `pivot_table()`
- Sparse matrix preparation
- Handling missing ratings

---

## 16. Advanced Pandas for ML

### 16.1 Multi-Index for ML Data
- Hierarchical indexing for time series ML data
- `xs()` for cross-section retrieval
- Multi-level GroupBy

### 16.2 Window Functions for Sequence Features
- `rolling()` — fixed-size window
- `expanding()` — variable-size window
- `shift()` — lag features for temporal data

### 16.3 Custom Function Application
- `apply()` with complex row-wise logic
- `pipe()` — building reusable preprocessing pipelines
- `transform()` — group-aware feature engineering

---

## 17. Pandas Best Practices for ML

### 17.1 Efficient ML Code
- Use vectorized operations always
- Avoid `iterrows()` for large datasets
- Use `apply()` only when no vectorized alternative exists
- Batch operations instead of row-by-row

### 17.2 Reproducibility
- Fix random seeds for `sample()`
- Version datasets with timestamps
- Save preprocessed DataFrames to Parquet

### 17.3 Common ML Mistakes
- Data leakage — using test data in preprocessing
- Incorrect merge type causing data loss or duplication
- Encoding categorical columns on full dataset before split
- Not resetting index after operations

---

## 18. Pandas Interview Topics for ML

### 18.1 Frequently Asked Questions
- How to handle missing data in ML datasets?
- What is the difference between `map()`, `apply()`, and `applymap()`?
- How to do one-hot encoding in Pandas?
- What is the difference between `groupby().transform()` and `groupby().agg()`?
- How to detect and remove outliers in a DataFrame?
- What is `SettingWithCopyWarning` and how to avoid it?

### 18.2 ML-Oriented Coding Problems
- Build a preprocessing pipeline using Pandas
- Encode all categorical columns in a DataFrame
- Compute feature-target correlation and select top features
- Create lag and rolling features for a time series dataset
- Handle class imbalance using Pandas oversampling

---

## 19. Real-World ML Projects Using Pandas

### 19.1 Beginner Projects
- Titanic survival dataset — data cleaning and EDA
- Iris dataset — feature exploration and encoding
- House price data — feature engineering for regression

### 19.2 Intermediate Projects
- End-to-end preprocessing pipeline for a classification task
- Customer churn analysis and feature engineering
- Time series sales forecasting data preparation

### 19.3 Advanced Projects
- Multi-table feature engineering for e-commerce recommendation
- Automated EDA and feature selection report
- ML-ready dataset builder with full preprocessing pipeline

---

## 20. Pandas Roadmap for ML Engineers

### 20.1 Beginner Level
- Loading and exploring datasets
- Handling missing values
- Basic filtering and selection
- Simple feature encoding

### 20.2 Intermediate Level
- Feature engineering with GroupBy and window functions
- Merging multiple datasets
- One-hot encoding and binning
- Time series feature extraction

### 20.3 Advanced Level
- Building reusable preprocessing pipelines
- Working with large datasets using chunking
- Integration with Scikit-learn pipelines
- Custom feature transformers using `apply()` and `pipe()`
