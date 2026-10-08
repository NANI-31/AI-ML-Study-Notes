# Pandas for Deep Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Pandas for Deep Learning

### 1.1 What is Pandas?
- Introduction to structured data handling
- Why Pandas is important in Deep Learning workflows
- Pandas as the data management layer before neural network training
- Pandas vs NumPy vs TensorFlow Datasets — when to use which
- Role of Pandas in the DL pipeline
  - Data ingestion and loading
  - Data cleaning and validation
  - Feature engineering
  - Preprocessing before feeding into neural networks
  - Post-training result analysis

### 1.2 Why Deep Learning Engineers Use Pandas
- Handling raw structured datasets before tensor conversion
- Efficient data exploration and validation
- Cleaning dirty real-world datasets
- Building reproducible preprocessing pipelines
- Integrating with TensorFlow Data API and PyTorch DataLoader

### 1.3 Installing and Setup
- Installing Pandas
- Jupyter Notebook / Google Colab setup
- Import conventions
  - `import pandas as pd`
  - `import numpy as np`
  - `import tensorflow as tf`
  - `import torch`

---

## 2. Pandas Fundamentals for Deep Learning

### 2.1 Core Data Structures
- Series — 1D labeled array
- DataFrame — 2D tabular structure
- Index object — row and column labels
- Why DataFrames are the starting point before tensor conversion

### 2.2 Loading DL Datasets
- `pd.read_csv()` — structured training data
- `pd.read_excel()`
- `pd.read_json()`
- `pd.read_parquet()` — efficient columnar format for large DL datasets
- Reading from cloud storage (GCS, S3) via URL
- Loading annotation files (COCO, ImageNet label CSV)
- Reading metadata CSV files for image/audio datasets

### 2.3 Data Types for Deep Learning
- `float32` — primary dtype for neural networks
- `int64` — class labels
- `bool` — mask arrays
- `object` — file paths, captions, labels
- `category` — memory-efficient class labels
- Converting to `float32` — `astype('float32')`
- Why `float64` wastes GPU memory

---

## 3. Exploratory Data Analysis (EDA) for Deep Learning

### 3.1 Initial Data Inspection
- `head()` and `tail()` — inspecting sample rows
- `shape` — total samples and features
- `info()` — column types and null counts
- `describe()` — statistical distribution of numerical features
- `dtypes` — verifying types for DL compatibility

### 3.2 Class Distribution Analysis
- `value_counts()` — class frequency count
- `value_counts(normalize=True)` — class proportion
- Detecting imbalanced classes
- Visualizing class distribution
- Multi-label class analysis

### 3.3 Feature Correlation Analysis
- `df.corr()` — correlation matrix
- Identifying redundant features
- Correlation with target label
- Removing highly correlated features

### 3.4 Missing Data Analysis
- `isnull().sum()` — missing count per column
- Missing percentage per feature
- Patterns in missing data — MCAR, MAR, MNAR
- Impact of missing data on DL training

### 3.5 Outlier Analysis
- Z-score detection
- IQR detection
- Impact of outliers on neural network convergence
- Visualizing outliers with box plots

---

## 4. Data Cleaning for Deep Learning

### 4.1 Handling Missing Values
- `dropna()` — dropping rows with missing labels
- `fillna()` — imputing with mean, median, or zero
- `fillna(method='ffill')` — forward fill for sequential data
- `interpolate()` — for time-series based inputs
- Strategies for DL: drop incomplete samples, impute, flag with indicator columns

### 4.2 Removing Duplicates
- `duplicated()` — detecting duplicate samples
- `drop_duplicates()` — ensuring training set purity
- Duplicate detection on file path columns for image datasets

### 4.3 Fixing Data Types for DL
- `pd.to_numeric()` — converting string numbers
- `astype('float32')` — DL-ready numeric type
- `pd.to_datetime()` — datetime parsing for sequential models
- Fixing incorrectly typed label columns

### 4.4 Cleaning Label Columns
- Stripping whitespace — `str.strip()`
- Lowercasing labels — `str.lower()`
- Standardizing inconsistent class names — `replace()`
- Removing invalid or ambiguous labels

### 4.5 Handling Corrupted Data
- Flagging corrupted file paths
- Removing rows with invalid image/audio paths
- Handling zero-length or empty records

---

## 5. Feature Engineering for Deep Learning

### 5.1 Numerical Feature Engineering
- Log transformation for skewed features — `np.log1p()`
- Square root transformation
- Polynomial features
- Ratio features
- Interaction features between numerical columns

### 5.2 Categorical Encoding for DL
- Label encoding — `map()` or `pd.Categorical().codes`
- One-hot encoding — `pd.get_dummies()`
- Embedding-ready integer encoding for Embedding layers
- `pd.factorize()` — fast integer encoding for categories
- Frequency encoding — category frequency as feature

### 5.3 Datetime Feature Engineering for Sequence Models
- Year, month, day extraction — `.dt.year`, `.dt.month`, `.dt.day`
- Hour, minute extraction — `.dt.hour`, `.dt.minute`
- Day of week — `.dt.dayofweek`
- Cyclical encoding of time features
  - `sin` and `cos` encoding for hours, months
- Days since reference date
- Elapsed time features

### 5.4 Text Data Metadata Features
- String length — `str.len()`
- Word count — `str.split().apply(len)`
- Character count
- Presence of specific tokens — `str.contains()`
- Sentence count

### 5.5 Image Metadata Features
- File path storage in DataFrame
- Image label encoding
- Bounding box coordinate storage
- Augmentation strategy flags
- Train/val/test split flags

### 5.6 Window Features for Sequential Models
- Lag features — `shift(n)`
- Rolling mean — `rolling(n).mean()`
- Rolling std — `rolling(n).std()`
- Expanding features — `expanding().mean()`
- Exponentially weighted features — `ewm(span=n).mean()`
- Differencing — `df.diff()`

---

## 6. Feature Scaling for Neural Networks

### 6.1 Min-Max Normalization
- Formula: `(x - min) / (max - min)`
- Scaling to [0, 1] range
- Manual implementation using Pandas
- Required for sigmoid output layers

### 6.2 Standardization (Z-score)
- Formula: `(x - mean) / std`
- Scaling to zero mean, unit variance
- Implementation using `mean()` and `std()`
- Common for fully connected layers

### 6.3 Robust Scaling
- Median and IQR-based scaling
- Useful when outliers are present
- Less sensitive to extreme values

### 6.4 Log Transformation
- `np.log1p()` for right-skewed features
- When to use for DL inputs
- Effect on gradient stability

### 6.5 Cyclical Encoding
- `sin`/`cos` transformation for periodic features
- Encoding hour-of-day, month, day-of-week
- Preserving circular relationships for RNNs and Transformers

---

## 7. Handling Class Imbalance for Deep Learning

### 7.1 Detecting Class Imbalance
- `value_counts()` — class distribution
- Class ratio calculation
- Impact on cross-entropy loss training

### 7.2 Oversampling with Pandas
- `sample(replace=True)` — random oversampling of minority class
- Duplicate minority class rows
- `pd.concat()` — combining oversampled classes

### 7.3 Undersampling with Pandas
- `sample(n=N)` — random undersampling of majority class
- Stratified sampling

### 7.4 Class Weights Computation
- Computing class weights from `value_counts()`
- Passing class weights to TensorFlow / PyTorch loss functions
- Inverse frequency weighting

### 7.5 Focal Loss Preparation
- Computing sample weights
- Storing sample weights in DataFrame
- Passing to TensorFlow `sample_weight` parameter

---

## 8. Data Splitting for Deep Learning

### 8.1 Train-Validation-Test Split
- `sample(frac=0.7)` — training set
- `sample(frac=0.15)` — validation set
- `sample(frac=0.15)` — test set
- `reset_index(drop=True)` — after splitting

### 8.2 Stratified Split
- Maintaining class balance across splits
- `groupby().apply(sample)` — per-class stratified sampling
- Ensuring consistent distribution in DL training

### 8.3 Time-Based Split for Sequential Models
- Sorting by datetime — `sort_values()`
- Forward-looking train/val split
- Avoiding data leakage in time series

### 8.4 K-Fold Data Preparation
- Creating fold assignments using `np.array_split()`
- Fold column in DataFrame
- Iterating folds for cross-validation

### 8.5 DataFrame to Tensor Conversion
- `df.to_numpy()` — NumPy as intermediate step
- `tf.constant(df.values)` — TensorFlow tensor
- `torch.tensor(df.values)` — PyTorch tensor
- Ensuring correct dtype before conversion

---

## 9. Building Dataset Pipelines with Pandas

### 9.1 Image Dataset Pipeline
- Storing image file paths in DataFrame
- Label column preparation
- Train/val/test flag column
- Shuffling — `df.sample(frac=1)`
- Converting to TensorFlow Dataset from file paths

### 9.2 Text Dataset Pipeline
- Storing text samples in DataFrame
- Label encoding
- Train/val/test split
- Tokenization preparation
- Converting to TensorFlow Dataset with `tf.data.Dataset.from_tensor_slices()`

### 9.3 Tabular Dataset Pipeline
- Full preprocessing in Pandas
- `df.to_numpy()` conversion
- Creating TensorFlow Dataset from NumPy
- Creating PyTorch Dataset from NumPy

### 9.4 Time Series Dataset Pipeline
- Loading time series data
- Creating lag and window features
- Sliding window dataset construction
- Converting sequences to NumPy arrays
- Feeding into LSTM / GRU models

---

## 10. Pandas Integration with TensorFlow

### 10.1 DataFrame to TensorFlow Dataset
- `tf.data.Dataset.from_tensor_slices()` — from NumPy arrays
- `tf.data.Dataset.from_generator()` — from Pandas iterators
- `df.to_numpy()` — intermediate conversion

### 10.2 TensorFlow Feature Columns (Legacy)
- `tf.feature_column` with Pandas feature names
- Numeric columns
- Categorical columns

### 10.3 Keras Input from DataFrames
- Preparing feature matrix from DataFrame
- Passing feature matrix to `model.fit()`
- Using Pandas for train/val split before Keras training

### 10.4 Storing Training Results
- Storing epoch-level metrics in DataFrame
- `pd.DataFrame(history.history)` — Keras training history
- Analyzing training curves with Pandas

---

## 11. Pandas Integration with PyTorch

### 11.1 DataFrame to PyTorch Dataset
- Creating custom `Dataset` class from Pandas DataFrame
- `__len__` and `__getitem__` using DataFrame indexing
- `torch.tensor(df.values)` — conversion

### 11.2 DataLoader Preparation
- Preparing NumPy arrays from Pandas
- Creating `TensorDataset` from Pandas
- Batch size and shuffle in DataLoader

### 11.3 Storing and Analyzing Results
- Storing PyTorch predictions back to DataFrame
- `pd.DataFrame(predictions.numpy())` — prediction analysis
- Comparing ground truth and predictions in DataFrame

---

## 12. Pandas for Sequence and Time Series Deep Learning

### 12.1 Time Series Data Preparation
- Loading time series with DatetimeIndex
- Sorting by time — `sort_index()`
- Handling irregular time series
- Resampling — `resample('H').mean()`

### 12.2 Creating Sequence Windows
- Sliding window using `shift()`
- Creating input sequences as NumPy arrays
- Creating target sequences
- Handling multi-step forecasting

### 12.3 Feature Extraction for LSTMs
- Lag features — `shift(n)`
- Rolling statistics — `rolling(n).mean()`, `.std()`
- Differencing — `df.diff()`
- Decomposed trend and seasonality features

### 12.4 Multi-variate Time Series
- Multiple feature columns
- Stacking features for LSTM input
- Normalizing each feature independently

---

## 13. Pandas for NLP Preprocessing

### 13.1 Text Data Storage
- Storing text in DataFrame column
- Storing tokenized texts
- Storing token IDs in DataFrame

### 13.2 Text Metadata Features
- Text length — `str.len()`
- Word count — `str.split().apply(len)`
- Unique word count
- Presence of special characters

### 13.3 Label Management for NLP
- Multi-label encoding using `pd.get_dummies()`
- Storing token-level labels for NER
- Storing sequence-level labels for classification

### 13.4 Dataset Splitting for NLP
- Stratified splitting on label column
- Train/val/test split
- Converting to HuggingFace Dataset format from Pandas

---

## 14. Pandas for Computer Vision Preprocessing

### 14.1 Image Metadata Management
- Storing image file paths
- Storing image labels
- Storing bounding boxes as columns
- Storing segmentation mask paths

### 14.2 Dataset Organization
- Train/val/test split flags
- Class-wise counts validation
- Augmentation flags

### 14.3 COCO and Annotation File Handling
- Reading COCO JSON into DataFrame
- Reading YOLO label files into DataFrame
- Image-annotation mapping using merge

---

## 15. Regularization and Dropout Data Preparation

### 15.1 Storing Regularization Parameters
- L1/L2 penalty tracking in experiment DataFrame
- Dropout rate logging
- Hyperparameter experiment tracking using Pandas

### 15.2 Experiment Result Analysis
- Storing validation loss per epoch
- Comparing experiments across hyperparameters
- `groupby()` for experiment aggregation

---

## 16. Performance Optimization for DL Data Pipelines

### 16.1 Memory Optimization
- `astype('float32')` — reducing memory from float64
- `astype('category')` — for low-cardinality label columns
- `usecols` — loading only required columns
- `pd.read_parquet()` — efficient binary format

### 16.2 Chunked Data Processing
- `pd.read_csv(chunksize=N)` — streaming large datasets
- Processing and storing chunks as Parquet files
- Chunked normalization

### 16.3 Efficient Feature Engineering
- Vectorized string operations
- `eval()` for fast computation
- `query()` for fast filtering
- Avoid `apply()` on large DataFrames when vectorized alternative exists

---

## 17. Numerical Stability in Data Preprocessing

### 17.1 Avoiding Overflow and Underflow
- Clipping extreme values — `df.clip(lower, upper)`
- Log-transforming large values
- Standardizing before feeding to neural networks

### 17.2 Handling NaN and Inf in Features
- `df.replace([np.inf, -np.inf], np.nan)` — replacing infinities
- `df.isnull().sum()` — after transformation validation
- `np.isfinite()` checks before conversion

---

## 18. Pandas for DL Experiment Tracking

### 18.1 Storing Experiment Results
- DataFrame as experiment log
- Columns: model name, hyperparameters, train loss, val loss, accuracy
- `df.to_csv()` — saving experiment logs

### 18.2 Comparing Models
- `sort_values('val_accuracy', ascending=False)` — ranking experiments
- `groupby('model').mean()` — average performance per model
- Plotting training curves from DataFrame

### 18.3 Hyperparameter Analysis
- Correlation between hyperparameters and performance
- Pivot table for hyperparameter grid analysis
- Best configuration selection

---

## 19. Debugging Deep Learning Pipelines with Pandas

### 19.1 Data Validation Before Training
- `assert df.isnull().sum().sum() == 0` — no NaNs
- Dtype verification — all float features as float32
- Shape verification — expected number of samples
- Label distribution check — class balance

### 19.2 Prediction Analysis
- Storing predictions in DataFrame
- Comparing predictions vs ground truth
- Confusion matrix from DataFrame
- Error analysis — examining wrong predictions

### 19.3 Gradient Debugging Support
- Detecting NaN in target labels
- Detecting Inf in input features
- Logging problematic samples

---

## 20. Pandas Best Practices for Deep Learning

### 20.1 Efficient DL Preprocessing Code
- Vectorize all preprocessing
- Save preprocessed data as Parquet for reuse
- Separate preprocessing logic from training code

### 20.2 Reproducibility
- Fix random seeds in `sample()`
- Version preprocessed datasets
- Log preprocessing steps in experiment DataFrame

### 20.3 Common Mistakes in DL Pipelines
- Data leakage — normalizing on full dataset before split
- Encoding on full dataset before train/test split
- Not converting to `float32` before tensor creation
- Forgetting to reset index after filtering
- Incorrect merge causing data duplication or loss

---

## 21. Pandas Interview Topics for Deep Learning Engineers

### 21.1 Frequently Asked Questions
- How is Pandas used in a DL pipeline?
- How to handle missing data for neural network inputs?
- How to convert a Pandas DataFrame to a TensorFlow Dataset?
- How to prevent data leakage in DL preprocessing?
- How to create lag features for LSTM input?
- How to encode categorical labels for multi-class neural networks?

### 21.2 DL-Oriented Coding Problems
- Build a preprocessing pipeline for image metadata CSV
- Create sequence windows for LSTM from time series DataFrame
- Encode multi-class labels and convert to TensorFlow Dataset
- Build a stratified train/val/test split for imbalanced classification
- Store and analyze Keras training history using Pandas

---

## 22. Real-World DL Projects Using Pandas

### 22.1 Beginner Projects
- MNIST metadata analysis and preprocessing
- Titanic survival neural network data preparation
- Image classification label CSV processing

### 22.2 Intermediate Projects
- NLP sentiment analysis data pipeline
- Time series forecasting data preparation for LSTM
- Tabular deep learning preprocessing pipeline

### 22.3 Advanced Projects
- Multi-modal dataset pipeline (image paths + metadata)
- End-to-end preprocessing and experiment tracking pipeline
- Automated feature engineering pipeline for tabular DL

---

## 23. Pandas Roadmap for Deep Learning Engineers

### 23.1 Beginner Level
- Loading and exploring datasets
- Handling missing values
- Label encoding and one-hot encoding
- Basic type conversion to float32

### 23.2 Intermediate Level
- Feature engineering for DL inputs
- Train/val/test split strategies
- Pandas to NumPy to Tensor conversion
- Window features for sequential models

### 23.3 Advanced Level
- Building reusable preprocessing pipelines
- Large dataset chunking and Parquet-based workflows
- Integration with TensorFlow Data API and PyTorch DataLoader
- Experiment tracking and hyperparameter analysis with Pandas

---

## 24. Transition from Pandas to Deep Learning Frameworks

### 24.1 Why Transition Beyond Pandas for DL
- Pandas is CPU-bound — not GPU accelerated
- No automatic differentiation
- Not designed for batched tensor operations
- Memory limitations with very large datasets

### 24.2 Transition Path
- Pandas → NumPy → TensorFlow Dataset
- Pandas → NumPy → PyTorch DataLoader
- Pandas → HuggingFace Datasets (for NLP)

### 24.3 DL Framework Readiness from Pandas
- Understanding tensor shapes from DataFrame shapes
- Understanding batch dimensions
- Knowing when to leave Pandas and enter the DL training loop
