#!/usr/bin/env python3
"""
Build prompt_builder.html from the askAI .md prompt files.
Run: python build_prompt_builder.py  (from inside askAI/)
"""

import os, re, json

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Topic metadata ──────────────────────────────────────────────
TOPICS = [
    # folder, id, label, icon-emoji, md-filename, placeholder-tag
    ("01_Data_Science_Foundations", "numpy",            "NumPy",            "🔢", "numpy.md",            "[NUMPY SECTION]"),
    ("01_Data_Science_Foundations", "pandas",           "Pandas",           "🐼", "pandas.md",           "[PANDAS SECTION]"),
    ("01_Data_Science_Foundations", "eda",              "EDA",              "🔍", "eda.md",               "[EDA SECTION]"),
    ("01_Data_Science_Foundations", "sqlite",           "SQLite",           "🗄️", "sqlite.md",           "[SQLITE SECTION]"),
    ("02_Machine_Learning",         "scikit-learn",     "Scikit-learn",     "⚙️", "scikit-learn.md",     "[SKLEARN SECTION]"),
    ("02_Machine_Learning",         "pytorch",          "PyTorch",          "🔥", "pytorch.md",          "[PYTORCH SECTION]"),
    ("02_Machine_Learning",         "deep_learning",    "Deep Learning",    "🧬", "deep_learning.md",    "[DL SECTION]"),
    ("03_Data_Visualization",       "matplotlib",       "Matplotlib",       "📊", "matplotlib.md",       "[MATPLOTLIB SECTION]"),
    ("03_Data_Visualization",       "seaborn",          "Seaborn",          "🎨", "seaborn.md",          "[SEABORN SECTION]"),
    ("03_Data_Visualization",       "plotly",           "Plotly",           "📈", "plotly.md",           "[PLOTLY SECTION]"),
    ("03_Data_Visualization",       "data_visualization","Data Visualization","🖼️","data_visualization.md","[DATA VIZ SECTION]"),
    ("04_Generative_AI_and_LLMs",   "llm",              "LLMs",             "🤖", "llm.md",              "[LLM SECTION]"),
    ("04_Generative_AI_and_LLMs",   "transformers",     "Transformers (HF)","🤗", "transformers.md",     "[TRANSFORMERS SECTION]"),
    ("04_Generative_AI_and_LLMs",   "embeddings",       "Embeddings",       "🔗", "embeddings.md",       "[EMBEDDING SECTION]"),
    ("04_Generative_AI_and_LLMs",   "vector_database",  "Vector Database",  "🗃️", "vector_database.md", "[VECTOR DB SECTION]"),
    ("04_Generative_AI_and_LLMs",   "chromadb",         "ChromaDB",         "🎯", "chromadb.md",         "[CHROMADB SECTION]"),
    ("04_Generative_AI_and_LLMs",   "rag",              "RAG",              "📚", "rag.md",              "[RAG SECTION]"),
    ("04_Generative_AI_and_LLMs",   "prompt_engineering","Prompt Engineering","✍️","prompt_engineering.md","[PROMPT ENG SECTION]"),
    ("04_Generative_AI_and_LLMs",   "langchain",        "LangChain",        "⛓️", "langchain.md",       "[LANGCHAIN SECTION]"),
    ("04_Generative_AI_and_LLMs",   "langgraph",        "LangGraph",        "🕸️", "langgraph.md",       "[LANGGRAPH SECTION]"),
    ("04_Generative_AI_and_LLMs",   "langsmith",        "LangSmith",        "🔭", "langsmith.md",        "[LANGSMITH SECTION]"),
    ("04_Generative_AI_and_LLMs",   "pydantic",         "Pydantic",         "✅", "pydantic.md",         "[PYDANTIC SECTION]"),
    ("04_Generative_AI_and_LLMs",   "openai_api",       "OpenAI API",       "🌐", "openai_api.md",       "[OPENAI SECTION]"),
    ("04_Generative_AI_and_LLMs",   "claude_api",       "Claude API",       "🧠", "claude_api.md",       "[CLAUDE SECTION]"),
    ("05_Deployment",               "fastapi",          "FastAPI",          "⚡", "fastapi.md",          "[FASTAPI SECTION]"),
    ("05_Deployment",               "docker",           "Docker",           "🐳", "docker.md",           "[DOCKER SECTION]"),
]

FOLDER_LABELS = {
    "01_Data_Science_Foundations": "01 · Data Science Foundations",
    "02_Machine_Learning":         "02 · Machine Learning",
    "03_Data_Visualization":       "03 · Data Visualization",
    "04_Generative_AI_and_LLMs":   "04 · Generative AI & LLMs",
    "05_Deployment":               "05 · Deployment",
}

# ─── Hardcoded roadmaps (one per topic id) ───────────────────────
ROADMAPS = {
"numpy": """1.  Introduction to NumPy         (1.1 What is NumPy, 1.2 Installing, 1.3 Basics)
2.  Creating Arrays               (2.1 Array Creation, 2.2 Special Arrays, 2.3 Range-based, 2.4 Random)
3.  Array Attributes & Properties (3.1 Properties, 3.2 Changing Properties)
4.  Array Indexing & Slicing      (4.1 Indexing, 4.2 Slicing, 4.3 Boolean, 4.4 Fancy)
5.  Array Operations              (5.1 Arithmetic, 5.2 Vectorized, 5.3 Comparison)
6.  Broadcasting                  (6.1 Basics, 6.2 Examples, 6.3 Use Cases)
7.  Mathematical Functions        (7.1 ufuncs, 7.2 Statistical, 7.3 Trig, 7.4 Log/Exp)
8.  Shape Manipulation            (8.1 Reshape, 8.2 Transpose, 8.3 Joining, 8.4 Splitting)
9.  Iterating Through Arrays      (9.1 Basics, 9.2 Efficient nditer)
10. Copy vs View                  (10.1 Copy, 10.2 View, 10.3 Differences)
11. Sorting, Searching, Filtering (11.1 Sorting, 11.2 Searching, 11.3 Filtering)
12. NumPy Data Types              (12.1 dtypes, 12.2 Structured Arrays)
13. Matrix Operations             (13.1 Basics, 13.2 Arithmetic, 13.3 Linear Algebra, 13.4 Equations)
14. Random Module                 (14.1 Random Numbers, 14.2 Distributions, 14.3 Sampling)
15. File Handling                 (15.1 Saving, 15.2 Loading, 15.3 CSV)
16. Advanced Concepts             (16.1 Broadcasting Advanced, 16.2 Strides, 16.3 Vectorization, 16.4 Memory)
17. NumPy and Performance         (17.1 Why Fast, 17.2 Benchmarking, 17.3 Optimization)
18. NumPy with Pandas/Matplotlib  (18.1 NumPy+Pandas, 18.2 NumPy+Matplotlib)
19. NumPy for Data Science        (19.1 Cleaning, 19.2 Numerical Analysis, 19.3 Feature Engineering)
20. NumPy Best Practices          (20.1 Efficient Code, 20.2 Readability, 20.3 Common Mistakes)
21. NumPy Interview Topics        (21.1 FAQs, 21.2 Coding Problems)
22. NumPy Real-World Projects     (22.1 Beginner, 22.2 Intermediate, 22.3 Advanced)
23. NumPy Ecosystem               (23.1 Related Libraries, 23.2 NumPy in ML)
24. Testing and Debugging         (24.1 Debugging, 24.2 Testing)
25. Advanced Scientific Computing (25.1 Numerical Methods, 25.2 Signal Processing, 25.3 Simulation)
26. Miscellaneous Concepts        (26.1 Masked Arrays, 26.2 Memory Mapping, 26.3 API Internals)""",

"pandas": """1.  Introduction to Pandas               (1.1 What is Pandas, 1.2 Installing, 1.3 Basics)
2.  Creating Data Structures             (2.1 Series, 2.2 DataFrames, 2.3 Structure)
3.  Series & DataFrame Properties        (3.1 Series Props, 3.2 DF Props, 3.3 Info)
4.  Data Selection & Indexing            (4.1 Columns, 4.2 Rows, 4.3 Techniques)
5.  DataFrame Operations                 (5.1 Adding, 5.2 Updating, 5.3 Deleting, 5.4 Renaming)
6.  Data Filtering & Conditional Select  (6.1 Basic, 6.2 Multiple Conditions, 6.3 Advanced)
7.  Data Cleaning                        (7.1 Missing Values, 7.2 Handling Missing, 7.3 Invalid Data)
8.  Data Types & Type Conversion         (8.1 Types, 8.2 Conversion, 8.3 Errors)
9.  String Operations                    (9.1 Basic, 9.2 Searching, 9.3 Modification, 9.4 Real-world)
10. Sorting & Ranking                    (10.1 Sorting, 10.2 Ranking, 10.3 Top/Bottom)
11. GroupBy & Aggregation                (11.1 Basics, 11.2 Functions, 11.3 Advanced, 11.4 Real-world)
12. Combining & Merging DataFrames       (12.1 Concat, 12.2 Merge, 12.3 Join, 12.4 Combining)
13. Reshaping Data                       (13.1 Pivot, 13.2 Melt, 13.3 Techniques)
14. Working with Dates & Time            (14.1 Basics, 14.2 Components, 14.3 Ops, 14.4 Time Series)
15. Reading & Writing Data               (15.1 CSV, 15.2 Excel, 15.3 Other Formats)
16. Advanced Data Cleaning               (16.1 Duplicates, 16.2 Outliers, 16.3 Standardization)
17. Apply, Map, Lambda, Functions        (17.1 apply, 17.2 map, 17.3 applymap, 17.4 Lambda)
18. Statistical Analysis                 (18.1 Descriptive, 18.2 Relationships, 18.3 Quantiles)
19. Pandas for Data Science              (19.1 EDA, 19.2 Exploration, 19.3 Feature Engineering)
20. Pandas with NumPy                    (20.1 Integration, 20.2 Numerical Ops, 20.3 Workflow)
21. Pandas with Matplotlib & Seaborn     (21.1 Plotting, 21.2 Customizing, 21.3 Seaborn)
22. Pandas Performance & Optimization    (22.1 Why Fast, 22.2 Optimization, 22.3 Large Datasets)
23. Pandas Advanced Concepts             (23.1 MultiIndex, 23.2 Hierarchical, 23.3 Window Functions)
24. Best Practices, Testing & Debugging  (24.1 Best Practices, 24.2 Mistakes, 24.3 Debugging, 24.4 Testing)
25. Projects, Interview & Applications   (25.1-25.6: Projects, Interview Topics, Coding Problems, Ecosystem)""",

"eda": """1.  Introduction to EDA              (1.1 What is EDA, 1.2 Types, 1.3 Workflow)
2.  Understanding the Dataset        (2.1 Loading, 2.2 Inspection, 2.3 Variables)
3.  Data Quality Analysis            (3.1 Missing Values, 3.2 Handling Missing, 3.3 Duplicates, 3.4 Consistency)
4.  Data Types & Transformation      (4.1 Types, 4.2 Conversion, 4.3 Transformation)
5.  Univariate Analysis              (5.1 Numerical, 5.2 Distribution, 5.3 Visualizations, 5.4 Categorical)
6.  Bivariate Analysis               (6.1 Num vs Num, 6.2 Num vs Cat, 6.3 Cat vs Cat)
7.  Multivariate Analysis            (7.1 Multiple Numerical, 7.2 Multiple Categorical, 7.3 Num+Cat+Target)
8.  Visualization with Matplotlib    (8.1 Basic Charts, 8.2 Advanced)
9.  Statistical Visualization Seaborn (9.1 Distribution, 9.2 Relationship, 9.3 Categorical)
10. Interactive EDA with Plotly      (10.1 Interactive Charts, 10.2 Advanced Plotly)
11. Outlier Analysis                 (11.1 Understanding, 11.2 Detecting, 11.3 Handling)
12. Correlation Analysis             (12.1 Basics, 12.2 Methods, 12.3 Visualization)
13. Covariance Analysis              (13.1 Basics, 13.2 Analysis)
14. Distribution & Normality         (14.1 Distribution, 14.2 Normality Testing, 14.3 Skewness, 14.4 Kurtosis)
15. Statistical Testing              (15.1 Hypothesis, 15.2 Tests, 15.3 Interpretation)
16. Group-Based EDA                  (16.1 GroupBy, 16.2 Aggregation, 16.3 Business Segmentation)
17. Pivot Tables & Crosstabs         (17.1 Pivot Tables, 17.2 Crosstab)
18. Time-Series EDA                  (18.1 DateTime, 18.2 Analysis, 18.3 Visualization)
19. Feature Analysis                 (19.1 Distributions, 19.2 Relationships, 19.3 Feature Selection)
20. Target Variable Analysis         (20.1 Classification, 20.2 Regression, 20.3 Target vs Features)
21. Class Imbalance Analysis         (21.1 Understanding, 21.2 Detecting, 21.3 Handling)
22. Missing Data Visualization       (22.1 Patterns, 22.2 Visualization)
23. Advanced Multivariate Analysis   (23.1 Pairwise, 23.2 Dimensionality Reduction, 23.3 Clustering)
24. EDA with Scikit-learn            (24.1 Dataset Prep, 24.2 Preprocessing, 24.3 Feature Analysis)
25. Leakage Detection                (25.1 What is Leakage, 25.2 Detecting)
26. Sampling Analysis                (26.1 Techniques, 26.2 Bias, 26.3 Train-Test Distribution)
27. Data Drift & Distribution Shift  (27.1 Understanding, 27.2 Detecting, 27.3 Visualization)
28. Advanced Visualization           (28.1 Comparative, 28.2 Multivariate, 28.3 Geographic)
29. Automated EDA                    (29.1 Automation, 29.2 Libraries, 29.3 Reports)
30. EDA Dashboards                   (30.1 Design, 30.2 Tools, 30.3 Projects)
31. Domain-Specific EDA              (31.1 Sales, 31.2 E-Commerce, 31.3 Finance, 31.4 Healthcare)
32. EDA for Machine Learning         (32.1 Pre-Modeling, 32.2 Feature-Target, 32.3 Model-Oriented)""",

"sqlite": """1.  Introduction to SQLite          (1.1 What is SQLite, 1.2 Installing, 1.3 Basics)
2.  Database & Tables               (2.1 Creating DBs, 2.2 Creating Tables, 2.3 Viewing Structure)
3.  SQLite Data Types               (3.1 Storage Classes, 3.2 Type Affinity, 3.3 Handling)
4.  Inserting Data                  (4.1 INSERT Basics, 4.2 Data Types, 4.3 Bulk Insertion)
5.  Reading Data with SELECT        (5.1 Basics, 5.2 Filtering, 5.3 Logical Operators)
6.  Updating & Deleting Data        (6.1 UPDATE, 6.2 DELETE, 6.3 Safe Modification)
7.  SQLite Operators                (7.1 Comparison, 7.2 Logical, 7.3 Special, 7.4 Pattern Matching)
8.  Sorting & Limiting Data         (8.1 ORDER BY, 8.2 LIMIT/OFFSET, 8.3 Random)
9.  Aggregate Functions             (9.1 Basic, 9.2 GROUP BY, 9.3 HAVING)
10. String Functions                (10.1 Manipulation, 10.2 Processing)
11. Mathematical & Numeric Functions (11.1 Numeric, 11.2 Operations)
12. Date & Time                     (12.1 Functions, 12.2 Working with Dates, 12.3 Analysis)
13. NULL Handling                   (13.1 Understanding NULL, 13.2 Handling)
14. Constraints                     (14.1 Primary Key, 14.2 Other Constraints, 14.3 Behavior)
15. Relationships & Foreign Keys    (15.1 Relationships, 15.2 Foreign Keys, 15.3 Cascading)
16. JOIN Operations                 (16.1 INNER JOIN, 16.2 OUTER JOIN, 16.3 Other Techniques)
17. Subqueries                      (17.1 Basic, 17.2 Operators, 17.3 Applications)
18. Common Table Expressions CTEs   (18.1 Basics, 18.2 Recursive CTEs)
19. Set Operations                  (19.1 UNION/INTERSECT/EXCEPT, 19.2 Rules)
20. Views                           (20.1 Creating, 20.2 Applications)
21. Indexes                         (21.1 Basics, 21.2 Types, 21.3 Performance)
22. Query Optimization              (22.1 EXPLAIN, 22.2 Techniques, 22.3 Analysis)
23. Transactions                    (23.1 Basics, 23.2 ACID, 23.3 Use Cases)
24. Concurrency & Locking           (24.1 Basics, 24.2 Journal Modes, 24.3 Busy Handling)
25. PRAGMA Statements               (25.1 Basics, 25.2 Important PRAGMAs)
26. Database Maintenance            (26.1 Inspection, 26.2 Commands, 26.3 Health)
27. Import & Export                 (27.1 Importing, 27.2 Exporting, 27.3 CLI Commands)
28. Backup & Restore                (28.1 Backup, 28.2 Restore)
29. SQLite with Python              (29.1 sqlite3 Module, 29.2 Creating Tables, 29.3 Inserting, 29.4 Reading)
30. CRUD with Python                (30.1 Create, 30.2 Read, 30.3 Update, 30.4 Delete, 30.5 Full App)
31. Parameterized Queries Injection  (31.1 Parameterized, 31.2 SQL Injection)
32. Row Factories                   (32.1 sqlite3.Row, 32.2 Custom)
33. SQLite with Pandas              (33.1 Reading, 33.2 Writing, 33.3 Analysis)
34. SQLite with Web Apps            (34.1 Flask, 34.2 Django, 34.3 Best Practices)
35. Advanced SQL Queries            (35.1 Filtering, 35.2 CASE Statements, 35.3 Aggregation)
36. Window Functions                (36.1 Basics, 36.2 Ranking, 36.3 Analytical, 36.4 Frames)
37. JSON with SQLite                (37.1 Basics, 37.2 Functions, 37.3 Queries)
38. Full-Text Search FTS            (38.1 Basics, 38.2 Features, 38.3 Applications)""",

"scikit-learn": """1.  Introduction to Scikit-learn        (1.1 What is Sklearn, 1.2 Installing, 1.3 Basics)
2.  Understanding ML with Sklearn       (2.1 ML Basics, 2.2 Workflow, 2.3 Dataset Structure)
3.  Scikit-learn Datasets               (3.1 Built-in Datasets, 3.2 Loading, 3.3 Synthetic Data)
4.  Data Preprocessing                  (4.1 Scaling, 4.2 Encoding, 4.3 Imputation, 4.4 Pipelines)
5.  Train-Test Split                    (5.1 Basics, 5.2 Stratified, 5.3 Cross-Validation)
6.  Linear Models                       (6.1 Linear Regression, 6.2 Ridge/Lasso, 6.3 Logistic Regression)
7.  Decision Trees                      (7.1 Basics, 7.2 Hyperparameters, 7.3 Visualization)
8.  Ensemble Methods                    (8.1 Random Forest, 8.2 Gradient Boosting, 8.3 AdaBoost, 8.4 Stacking)
9.  Support Vector Machines             (9.1 SVM Basics, 9.2 Kernels, 9.3 SVC/SVR)
10. K-Nearest Neighbors                 (10.1 KNN Basics, 10.2 Distance Metrics, 10.3 Hyperparameters)
11. Naive Bayes                         (11.1 Types, 11.2 Text Classification)
12. Clustering                          (12.1 K-Means, 12.2 DBSCAN, 12.3 Hierarchical, 12.4 Evaluation)
13. Dimensionality Reduction            (13.1 PCA, 13.2 LDA, 13.3 t-SNE, 13.4 UMAP)
14. Feature Selection                   (14.1 Filter, 14.2 Wrapper, 14.3 Embedded Methods)
15. Feature Engineering                 (15.1 Polynomial, 15.2 Interaction, 15.3 Binning)
16. Model Evaluation - Classification   (16.1 Accuracy, 16.2 Precision/Recall/F1, 16.3 ROC-AUC)
17. Model Evaluation - Regression       (17.1 MSE/MAE/RMSE, 17.2 R-squared, 17.3 Residuals)
18. Hyperparameter Tuning               (18.1 Grid Search, 18.2 Random Search, 18.3 Bayesian)
19. Pipelines                           (19.1 Creating, 19.2 With Preprocessing, 19.3 Grid Search)
20. Imbalanced Data                     (20.1 Oversampling, 20.2 Undersampling, 20.3 Class Weights)
21. Model Persistence                   (21.1 joblib, 21.2 pickle, 21.3 ONNX)
22. Custom Estimators                   (22.1 BaseEstimator, 22.2 TransformerMixin, 22.3 Custom)
23. Sklearn with Pandas & NumPy         (23.1 Integration, 23.2 DataFrames in Pipelines)
24. Real-World Projects & Applications  (24.1 Classification, 24.2 Regression, 24.3 Clustering)
25. Interview Topics & Best Practices   (25.1 FAQs, 25.2 Common Mistakes, 25.3 Coding Problems)""",

"pytorch": """1.  Introduction to PyTorch             (1.1 What is PyTorch, 1.2 Installing+CUDA, 1.3 Basics)
2.  PyTorch Tensors                     (2.1 Creating, 2.2 Special, 2.3 Range-Based, 2.4 Random)
3.  Tensor Attributes & Properties      (3.1 Properties, 3.2 Datatypes, 3.3 Device/CUDA)
4.  Tensor Operations                   (4.1 Arithmetic, 4.2 Math Functions, 4.3 Reduction, 4.4 Comparison)
5.  Tensor Indexing & Slicing           (5.1 Indexing, 5.2 Slicing, 5.3 Boolean, 5.4 Advanced)
6.  Tensor Shape Operations             (6.1 Reshape/View, 6.2 Transpose, 6.3 Concatenate/Stack, 6.4 Squeeze)
7.  Autograd & Automatic Differentiation (7.1 requires_grad, 7.2 backward, 7.3 Gradient Flow, 7.4 no_grad)
8.  Neural Network Basics nn.Module     (8.1 nn.Module, 8.2 Linear Layers, 8.3 Activation Functions)
9.  Loss Functions                      (9.1 MSELoss, 9.2 CrossEntropyLoss, 9.3 BCELoss, 9.4 Custom)
10. Optimizers                          (10.1 SGD, 10.2 Adam/AdamW, 10.3 LR Schedulers)
11. Training Loop                       (11.1 Forward Pass, 11.2 Backward Pass, 11.3 Full Loop, 11.4 Validation)
12. Datasets & DataLoaders              (12.1 Dataset Class, 12.2 DataLoader, 12.3 Custom Dataset)
13. Convolutional Neural Networks CNN   (13.1 Conv2d, 13.2 Pooling, 13.3 BatchNorm, 13.4 Architecture)
14. Recurrent Neural Networks RNN       (14.1 RNN, 14.2 LSTM, 14.3 GRU, 14.4 Sequence Modeling)
15. Transfer Learning                   (15.1 Pretrained Models, 15.2 Feature Extraction, 15.3 Fine-tuning)
16. GPU Acceleration                    (16.1 CUDA, 16.2 Moving to Device, 16.3 Mixed Precision)
17. Model Saving & Loading              (17.1 state_dict, 17.2 Saving Full Model, 17.3 Checkpointing)
18. PyTorch for NLP                     (18.1 Text Preprocessing, 18.2 Embeddings, 18.3 Transformers)
19. PyTorch for Computer Vision         (19.1 torchvision, 19.2 Image Augmentation, 19.3 Object Detection)
20. Advanced Training Techniques        (20.1 Early Stopping, 20.2 Gradient Clipping, 20.3 Mixed Precision)
21. Debugging & Profiling               (21.1 Debugging, 21.2 torch.profiler, 21.3 NaN Detection)
22. TorchScript & ONNX Export           (22.1 TorchScript, 22.2 ONNX, 22.3 Deployment)
23. PyTorch Lightning                   (23.1 LightningModule, 23.2 Trainer, 23.3 Callbacks)
24. Real-World Projects                 (24.1 Image Classifier, 24.2 Text Classifier, 24.3 RAG)
25. Interview Topics & Best Practices   (25.1 FAQs, 25.2 Common Mistakes, 25.3 Optimization)""",

"deep_learning": """1.  Introduction to Deep Learning       (1.1 What is DL, 1.2 Applications, 1.3 Environment)
2.  Mathematics for Deep Learning       (2.1 Linear Algebra, 2.2 Calculus, 2.3 Probability)
3.  Neural Network Fundamentals         (3.1 Introduction, 3.2 Perceptron, 3.3 Multi-layer Network)
4.  Activation Functions                (4.1 Sigmoid/Tanh, 4.2 ReLU Variants, 4.3 Softmax, 4.4 Choosing)
5.  Loss Functions                      (5.1 Regression Losses, 5.2 Classification Losses, 5.3 Custom)
6.  Backpropagation                     (6.1 Chain Rule, 6.2 Gradient Flow, 6.3 Vanishing/Exploding)
7.  Optimization                        (7.1 SGD, 7.2 Momentum, 7.3 Adam/AdamW, 7.4 LR Schedulers)
8.  Training Techniques                 (8.1 Mini-batches, 8.2 Epochs, 8.3 Validation Split, 8.4 Early Stop)
9.  Regularization                      (9.1 Dropout, 9.2 L1/L2, 9.3 BatchNorm, 9.4 Data Augmentation)
10. Convolutional Neural Networks CNN   (10.1 Convolution, 10.2 Pooling, 10.3 Architectures, 10.4 Apps)
11. Recurrent Neural Networks RNN       (11.1 RNN, 11.2 LSTM, 11.3 GRU, 11.4 Bidirectional)
12. Attention Mechanism                 (12.1 What is Attention, 12.2 Self-Attention, 12.3 Multi-Head)
13. Transformer Architecture            (13.1 Encoder, 13.2 Decoder, 13.3 BERT, 13.4 GPT)
14. Transfer Learning                   (14.1 Pretrained Models, 14.2 Fine-tuning, 14.3 Domain Adaptation)
15. Generative Models                   (15.1 VAE, 15.2 GAN, 15.3 Diffusion Models, 15.4 Applications)
16. Computer Vision                     (16.1 Image Classification, 16.2 Object Detection, 16.3 Segmentation)
17. Natural Language Processing         (17.1 Text Classification, 17.2 NER, 17.3 Machine Translation)
18. Multi-Task & Meta Learning          (18.1 Multi-Task, 18.2 Few-Shot, 18.3 Zero-Shot)
19. Model Evaluation                    (19.1 Metrics, 19.2 Confusion Matrix, 19.3 Bias-Variance)
20. Hyperparameter Tuning               (20.1 Grid/Random Search, 20.2 Bayesian, 20.3 NAS)
21. Deep Learning Frameworks            (21.1 PyTorch, 21.2 TensorFlow/Keras, 21.3 JAX)
22. Model Deployment                    (22.1 ONNX, 22.2 TorchScript, 22.3 REST API, 22.4 Edge)
23. Real-World Projects                 (23.1 CV Projects, 23.2 NLP Projects, 23.3 GenAI Projects)
24. Interview Topics                    (24.1 Conceptual FAQs, 24.2 Math Questions, 24.3 Coding)
25. Best Practices & Optimization       (25.1 Training Tips, 25.2 Mixed Precision, 25.3 Distributed)""",

"matplotlib": """1.  Introduction to Matplotlib      (1.1 What is Matplotlib, 1.2 Installing, 1.3 Architecture)
2.  Matplotlib Basics               (2.1 First Plot, 2.2 Components, 2.3 Line Plots)
3.  Figure and Axes                 (3.1 Figure Object, 3.2 Axes Object, 3.3 Figure vs Axes)
4.  Line Plots                      (4.1 Basic, 4.2 Styling, 4.3 Multiple Lines)
5.  Bar Charts                      (5.1 Vertical, 5.2 Horizontal, 5.3 Grouped, 5.4 Stacked)
6.  Histograms                      (6.1 Basic, 6.2 Bins, 6.3 KDE Overlay)
7.  Scatter Plots                   (7.1 Basic, 7.2 Colors/Size, 7.3 Bubble Charts)
8.  Pie Charts                      (8.1 Basic, 8.2 Customizing, 8.3 Donut Charts)
9.  Box Plots & Violin Plots        (9.1 Box Plot, 9.2 Violin Plot, 9.3 Comparison)
10. Subplots                        (10.1 plt.subplots, 10.2 GridSpec, 10.3 Twin Axes)
11. Chart Customization             (11.1 Titles/Labels, 11.2 Colors/Styles, 11.3 Legends, 11.4 Annotations)
12. Axes Customization              (12.1 Limits, 12.2 Ticks, 12.3 Grid, 12.4 Log Scale)
13. Color Management                (13.1 Named Colors, 13.2 Colormaps, 13.3 Custom Palettes)
14. Text and Annotations            (14.1 text, 14.2 annotate, 14.3 Arrow Annotations)
15. Statistical Plots               (15.1 Error Bars, 15.2 Confidence Intervals, 15.3 Fill Between)
16. 3D Plotting                     (16.1 3D Axes, 16.2 Surface Plots, 16.3 3D Scatter)
17. Image Display                   (17.1 imshow, 17.2 Colorbars, 17.3 Image Processing)
18. Matplotlib Styles               (18.1 Built-in Styles, 18.2 ggplot/seaborn Style, 18.3 Custom)
19. Saving Figures                  (19.1 savefig, 19.2 DPI, 19.3 Formats: PNG/PDF/SVG)
20. Animation                       (20.1 FuncAnimation, 20.2 ArtistAnimation)
21. Matplotlib with Pandas          (21.1 DataFrame.plot, 21.2 Integration Patterns)
22. Matplotlib with Seaborn         (22.1 Combining, 22.2 Style Integration)
23. Interactive Matplotlib          (23.1 Widgets, 23.2 Callbacks)
24. Real-World Projects             (24.1 Dashboard, 24.2 Scientific Plots, 24.3 ML Visualization)
25. Best Practices & Interview      (25.1 OO vs pyplot, 25.2 Common Mistakes, 25.3 FAQs)""",

"seaborn": """1.  Introduction to Seaborn         (1.1 What is Seaborn, 1.2 Installing, 1.3 Basics)
2.  Seaborn Datasets                (2.1 Built-in Datasets, 2.2 Exploring, 2.3 Preparing Data)
3.  Figure & Style Management       (3.1 Themes, 3.2 Appearance, 3.3 Context, 3.4 Style Settings)
4.  Distribution Plots              (4.1 histplot, 4.2 kdeplot, 4.3 displot, 4.4 ecdfplot, 4.5 rugplot)
5.  Categorical Plots               (5.1 barplot, 5.2 countplot, 5.3 boxplot, 5.4 violinplot, 5.5 stripplot)
6.  Relationship Plots              (6.1 scatterplot, 6.2 lineplot, 6.3 relplot, 6.4 pairplot)
7.  Regression Plots                (7.1 regplot, 7.2 lmplot, 7.3 Residual Plots, 7.4 Confidence Bands)
8.  Matrix Plots                    (8.1 heatmap, 8.2 clustermap, 8.3 Correlation Matrix)
9.  Figure-Level vs Axes-Level      (9.1 Figure-Level, 9.2 Axes-Level, 9.3 When to Use Each)
10. FacetGrid & Grid Plots          (10.1 FacetGrid, 10.2 PairGrid, 10.3 Custom Grids)
11. Color Palettes                  (11.1 Built-in Palettes, 11.2 Custom Palettes, 11.3 color_palette)
12. Customizing Seaborn Plots       (12.1 Titles/Labels, 12.2 Axes Limits, 12.3 Legends, 12.4 Annotations)
13. Seaborn with Pandas             (13.1 DataFrame Integration, 13.2 Column Mapping, 13.3 GroupBy Viz)
14. Seaborn with Matplotlib         (14.1 Shared Figures, 14.2 Adding Matplotlib Elements)
15. Statistical Visualization       (15.1 Statistical Functions, 15.2 Confidence Intervals, 15.3 Error Bars)
16. Multi-Variable Analysis         (16.1 Hue/Size/Style, 16.2 Faceting, 16.3 Bubble Charts)
17. Time-Series Visualization       (17.1 Line Plots, 17.2 Rolling Averages, 17.3 Seasonal Plots)
18. Saving and Exporting            (18.1 savefig, 18.2 DPI/Format, 18.3 Publication Quality)
19. Real-World EDA with Seaborn     (19.1 Sales EDA, 19.2 Customer Analysis, 19.3 ML Visualization)
20. Best Practices & Interview      (20.1 Common Mistakes, 20.2 FAQs, 20.3 Seaborn vs Matplotlib)""",

"plotly": """1.  Introduction to Plotly           (1.1 What is Plotly, 1.2 Installing, 1.3 Architecture: px vs go)
2.  Plotly Basics                    (2.1 First Chart, 2.2 Figure Structure, 2.3 Plotly Express Basics)
3.  Plotly Express                   (3.1 Introduction, 3.2 Working with Data, 3.3 Common px Charts)
4.  Scatter Plots px                 (4.1 Basic Scatter, 4.2 Color/Size/Symbol, 4.3 Hover, 4.4 Bubble)
5.  Line Charts px                   (5.1 Basic Lines, 5.2 Multiple Lines, 5.3 Time-Series)
6.  Bar Charts px                    (6.1 Vertical/Horizontal, 6.2 Grouped/Stacked, 6.3 Animated Bars)
7.  Histograms & Box Plots px        (7.1 Histograms, 7.2 Box Plots, 7.3 Violin Plots)
8.  Pie & Sunburst Charts px         (8.1 Pie, 8.2 Donut, 8.3 Sunburst, 8.4 Treemap)
9.  Graph Objects go                 (9.1 Introduction, 9.2 Trace Types, 9.3 Layout, 9.4 go vs px)
10. Figure Layout & Customization    (10.1 Titles/Labels, 10.2 Colors, 10.3 Fonts, 10.4 Themes)
11. Subplots                         (11.1 make_subplots, 11.2 Shared Axes, 11.3 Custom Grid)
12. Interactive Controls             (12.1 Hover, 12.2 Zoom/Pan, 12.3 Dropdown, 12.4 Sliders)
13. Animation                        (13.1 Frame-Based, 13.2 play/pause, 13.3 Time-Series Animation)
14. 3D Visualization                 (14.1 3D Scatter, 14.2 Surface Plots, 14.3 3D Line)
15. Geographic/Map Visualization     (15.1 Scatter Geo, 15.2 Choropleth, 15.3 Mapbox)
16. Statistical Charts               (16.1 Error Bars, 16.2 Confidence Intervals, 16.3 ECDF)
17. Heatmaps & Correlation           (17.1 imshow, 17.2 Correlation Heatmap, 17.3 Annotated)
18. Facet Plots                      (18.1 facet_row/col, 18.2 Custom Facets, 18.3 Pair Plots)
19. Plotly with Pandas & NumPy       (19.1 DataFrames, 19.2 Long vs Wide Data, 19.3 Aggregations)
20. Saving & Exporting               (20.1 HTML, 20.2 Static Images, 20.3 Dash Integration)
21. Plotly Dash Basics               (21.1 What is Dash, 21.2 App Layout, 21.3 Callbacks)
22. Real-World Projects              (22.1 Sales Dashboard, 22.2 Financial Analysis, 22.3 ML Viz)
23. Best Practices & Interview       (23.1 px vs go, 23.2 Performance, 23.3 FAQs)""",

"data_visualization": """1.  Introduction to Data Visualization (1.1 What is DataViz, 1.2 Why Visualize, 1.3 Data Types)
2.  Visualization Environment          (2.1 Installing, 2.2 Importing, 2.3 Environment)
3.  Matplotlib Fundamentals            (3.1 Introduction, 3.2 First Plot, 3.3 Figure/Axes, 3.4 Plot Types)
4.  Seaborn Fundamentals               (4.1 Introduction, 4.2 Themes, 4.3 Distribution Plots, 4.4 Categorical)
5.  Plotly Fundamentals                (5.1 Introduction, 5.2 Express vs Objects, 5.3 Interactive Charts)
6.  Chart Selection Framework          (6.1 Comparison, 6.2 Distribution, 6.3 Relationship, 6.4 Composition)
7.  Design Principles                  (7.1 Visual Hierarchy, 7.2 Color Theory, 7.3 Chart Junk, 7.4 Data-Ink)
8.  Color Theory                       (8.1 Color Spaces, 8.2 Palettes, 8.3 Accessibility, 8.4 Sequential)
9.  Typography & Layout                (9.1 Fonts, 9.2 Title/Labels, 9.3 White Space, 9.4 Grid Layouts)
10. Univariate Visualization           (10.1 Histogram, 10.2 KDE, 10.3 Box Plot, 10.4 Bar Chart)
11. Bivariate Visualization            (11.1 Scatter Plot, 11.2 Line Chart, 11.3 Heatmap, 11.4 Bubble)
12. Multivariate Visualization         (12.1 Pair Plots, 12.2 Parallel Coordinates, 12.3 3D Charts)
13. Categorical Data Visualization     (13.1 Bar Charts, 13.2 Grouped/Stacked, 13.3 Pie/Donut)
14. Time-Series Visualization          (14.1 Line Charts, 14.2 Trend Lines, 14.3 Seasonal Decomposition)
15. Geographic Visualization           (15.1 Scatter Maps, 15.2 Choropleth, 15.3 Mapbox)
16. Statistical Visualization          (16.1 Error Bars, 16.2 Confidence Intervals, 16.3 Regression Lines)
17. Interactive Visualization          (17.1 Plotly Controls, 17.2 Animations, 17.3 Dash Dashboards)
18. Dashboard Design                   (18.1 Layout, 18.2 KPIs, 18.3 Filters, 18.4 Storytelling)
19. Visualization for ML               (19.1 Confusion Matrix, 19.2 ROC Curves, 19.3 Feature Importance)
20. Visualization for EDA              (20.1 Missing Value Viz, 20.2 Outlier Viz, 20.3 Correlation Matrix)
21. Advanced Visualization Techniques  (21.1 Animations, 21.2 Sankey, 21.3 Radar, 21.4 Treemap)
22. Best Practices                     (22.1 Choosing Charts, 22.2 Avoiding Deception, 22.3 Accessibility)
23. Real-World Projects                (23.1 Sales Dashboard, 23.2 ML Results, 23.3 Scientific Publication)
24. Interview Topics                   (24.1 Chart Selection FAQs, 24.2 Design Questions, 24.3 Coding)""",

"llm": """1.  Introduction to LLMs               (1.1 What are LLMs, 1.2 Why Important, 1.3 Architecture Overview)
2.  NLP Foundations for LLMs           (2.1 Text Processing, 2.2 Representations, 2.3 NLP Tasks)
3.  Tokenization                        (3.1 Basics, 3.2 Subword BPE/WordPiece, 3.3 Special Tokens)
4.  Token Embeddings                    (4.1 Word Vectors, 4.2 Contextual Embeddings, 4.3 Positional Encoding)
5.  Transformer Architecture            (5.1 Self-Attention, 5.2 Multi-Head Attention, 5.3 FFN, 5.4 Layer Norm)
6.  Pre-training                        (6.1 Objectives: CLM/MLM, 6.2 Data, 6.3 Scale, 6.4 Foundation Models)
7.  Fine-tuning & Alignment             (7.1 Supervised Fine-tuning, 7.2 RLHF, 7.3 PPO/DPO, 7.4 Constitutional AI)
8.  PEFT & Efficient Fine-tuning        (8.1 LoRA, 8.2 QLoRA, 8.3 Prefix Tuning, 8.4 Adapters)
9.  Inference & Generation              (9.1 Greedy, 9.2 Beam Search, 9.3 Sampling, 9.4 Temperature/Top-p/Top-k)
10. Context Window & Memory             (10.1 Context Length, 10.2 Sliding Window, 10.3 Long-Context Models)
11. LLM Evaluation                      (11.1 Perplexity, 11.2 BLEU/ROUGE, 11.3 LLM-as-Judge, 11.4 Benchmarks)
12. Prompt Engineering                  (12.1 Zero/Few-Shot, 12.2 CoT, 12.3 ReAct, 12.4 Structured Output)
13. LLM APIs & SDKs                     (13.1 OpenAI API, 13.2 Anthropic API, 13.3 Hugging Face, 13.4 Ollama)
14. RAG Retrieval-Augmented Generation  (14.1 Architecture, 14.2 Chunking, 14.3 Retrieval, 14.4 Reranking)
15. LLM Agents                          (15.1 ReAct Agents, 15.2 Tool Use, 15.3 Planning, 15.4 Multi-Agent)
16. LLM Safety & Alignment              (16.1 Hallucination, 16.2 Bias, 16.3 Guardrails, 16.4 Red Teaming)
17. Quantization & Compression          (17.1 INT8/INT4, 17.2 GPTQ, 17.3 AWQ, 17.4 Distillation)
18. LLM Deployment                      (18.1 vLLM, 18.2 TGI, 18.3 Batching, 18.4 Latency vs Throughput)
19. Open-Source LLMs                    (19.1 Llama, 19.2 Mistral, 19.3 Gemma, 19.4 Local Deployment)
20. Multimodal LLMs                     (20.1 Vision-Language, 20.2 GPT-4V, 20.3 LLaVA, 20.4 Applications)
21. LLM Cost Optimization               (21.1 Token Counting, 21.2 Caching, 21.3 Batching, 21.4 Model Selection)
22. LLM Observability                   (22.1 Tracing, 22.2 LangSmith, 22.3 Logging, 22.4 Monitoring)
23. Real-World LLM Applications         (23.1 Chatbots, 23.2 Document QA, 23.3 Code Assistants, 23.4 Agents)
24. Interview Topics                    (24.1 Architecture FAQs, 24.2 Training Questions, 24.3 Inference)
25. LLM Engineering Best Practices      (25.1 System Design, 25.2 Evaluation, 25.3 Production Patterns)""",

"transformers": """1.  Introduction to Transformers        (1.1 What are Transformers, 1.2 Applications, 1.3 Ecosystem)
2.  Prerequisites                       (2.1 Python, 2.2 NumPy, 2.3 ML Fundamentals, 2.4 DL Fundamentals)
3.  Neural Networks for Transformers    (3.1 Basics, 3.2 Activation Functions, 3.3 Attention)
4.  Self-Attention Mechanism            (4.1 What is Attention, 4.2 Q/K/V Matrices, 4.3 Scaled Dot-Product)
5.  Multi-Head Attention                (5.1 Multiple Heads, 5.2 Concatenation, 5.3 Applications)
6.  Transformer Architecture            (6.1 Encoder, 6.2 Decoder, 6.3 Encoder-Decoder, 6.4 Variants)
7.  Tokenization HuggingFace            (7.1 AutoTokenizer, 7.2 Encoding, 7.3 Special Tokens, 7.4 Padding)
8.  HuggingFace Hub                     (8.1 Model Hub, 8.2 Datasets Hub, 8.3 Spaces, 8.4 Model Cards)
9.  AutoClasses                         (9.1 AutoModel, 9.2 AutoTokenizer, 9.3 AutoConfig, 9.4 Pipeline)
10. Text Classification                 (10.1 Sentiment Analysis, 10.2 Text Classification, 10.3 Zero-Shot)
11. Text Generation                     (11.1 GPT-style, 11.2 generate, 11.3 Sampling Strategies)
12. Sequence-to-Sequence Models         (12.1 Translation, 12.2 Summarization, 12.3 T5/BART)
13. Question Answering                  (13.1 Extractive QA, 13.2 Abstractive QA, 13.3 RAG Integration)
14. Token Classification                (14.1 NER, 14.2 POS Tagging, 14.3 Chunking)
15. Feature Extraction & Embeddings     (15.1 CLS Token, 15.2 Mean Pooling, 15.3 Sentence Transformers)
16. Fine-tuning with Trainer API        (16.1 TrainingArguments, 16.2 Trainer, 16.3 Custom Dataset)
17. PEFT & LoRA Fine-tuning             (17.1 PEFT Library, 17.2 LoRA, 17.3 QLoRA, 17.4 Prefix Tuning)
18. Datasets Library                    (18.1 Loading Datasets, 18.2 Preprocessing, 18.3 Custom Datasets)
19. Accelerate & Multi-GPU              (19.1 Accelerate Basics, 19.2 Mixed Precision, 19.3 Distributed)
20. Model Quantization                  (20.1 BitsAndBytes, 20.2 4-bit/8-bit, 20.3 GPTQ/AWQ)
21. Vision Transformers ViT             (21.1 ViT Architecture, 21.2 Image Classification, 21.3 CLIP)
22. Multimodal Models                   (22.1 LLaVA, 22.2 BLIP, 22.3 Whisper, 22.4 Applications)
23. Model Deployment                    (23.1 Inference Endpoints, 23.2 ONNX, 23.3 TorchScript)
24. Real-World Projects                 (24.1 Text Classifier, 24.2 RAG with HF, 24.3 Fine-tuned Chatbot)
25. Best Practices & Interview          (25.1 Model Selection, 25.2 Common Mistakes, 25.3 FAQs)""",

"embeddings": """1.  Introduction to Embeddings          (1.1 What are Embeddings, 1.2 Why Important, 1.3 Applications)
2.  Mathematical Foundations            (2.1 Vectors, 2.2 Vector Operations, 2.3 Similarity Measures)
3.  Text Embeddings Basics              (3.1 Text to Embedding, 3.2 Understanding Vectors, 3.3 Dimensions)
4.  Traditional Text Representations   (4.1 BoW, 4.2 TF-IDF, 4.3 One-Hot, 4.4 N-grams)
5.  Word2Vec                            (5.1 CBOW, 5.2 Skip-gram, 5.3 Negative Sampling, 5.4 Word Analogies)
6.  GloVe & FastText                    (6.1 GloVe, 6.2 FastText, 6.3 Subword Embeddings, 6.4 OOV Handling)
7.  Contextual Embeddings               (7.1 BERT CLS, 7.2 Mean Pooling, 7.3 ELMo, 7.4 Dynamic Embeddings)
8.  Sentence Transformers               (8.1 What is SBERT, 8.2 Siamese Networks, 8.3 all-MiniLM, 8.4 Usage)
9.  OpenAI Embeddings                   (9.1 text-embedding-3-small/large, 9.2 API Usage, 9.3 RAG)
10. HuggingFace Embedding Models        (10.1 Feature Extraction, 10.2 Sentence Transformers Library)
11. Cosine Similarity                   (11.1 Formula, 11.2 NumPy Implementation, 11.3 Applications)
12. Semantic Search                     (12.1 Query Embedding, 12.2 Document Embedding, 12.3 Top-K)
13. Embedding Storage                   (13.1 NumPy/FAISS, 13.2 ChromaDB, 13.3 Pinecone, 13.4 Weaviate)
14. FAISS                               (14.1 What is FAISS, 14.2 Flat Index, 14.3 IVF, 14.4 HNSW)
15. Embedding Evaluation                (15.1 MTEB, 15.2 Cosine vs Euclidean, 15.3 Task-Specific Eval)
16. Multilingual Embeddings             (16.1 mBERT, 16.2 LaBSE, 16.3 Cross-Lingual Search)
17. Multimodal Embeddings               (17.1 CLIP, 17.2 Image+Text, 17.3 Audio Embeddings)
18. Embeddings for RAG                  (18.1 Chunking Strategy, 18.2 Embedding Pipeline, 18.3 Retrieval)
19. Embedding Fine-tuning               (19.1 Contrastive Learning, 19.2 SimCSE, 19.3 Custom Fine-tuning)
20. Embedding Optimization              (20.1 Dimensionality Reduction, 20.2 Quantization, 20.3 Caching)
21. Embedding Visualization             (21.1 t-SNE, 21.2 UMAP, 21.3 Cluster Visualization)
22. Production Embedding Systems        (22.1 Batching, 22.2 Caching, 22.3 Cost, 22.4 Latency)
23. Real-World Projects                 (23.1 Semantic Search, 23.2 Document Retrieval, 23.3 RAG System)
24. Interview Topics                    (24.1 Similarity FAQs, 24.2 Model Comparison, 24.3 Coding)
25. Best Practices                      (25.1 Model Selection, 25.2 Chunking, 25.3 Common Mistakes)""",

"vector_database": """1.  Introduction to Vector Databases   (1.1 What is a VDB, 1.2 Why Important, 1.3 Architecture)
2.  Understanding Vectors & Embeddings (2.1 Vector Basics, 2.2 Embeddings, 2.3 Embedding Models)
3.  Similarity Search                  (3.1 What is It, 3.2 Distance Metrics, 3.3 ANN Search)
4.  Approximate Nearest Neighbor ANN   (4.1 Exact vs ANN, 4.2 HNSW, 4.3 IVF, 4.4 PQ)
5.  Vector Indexing                    (5.1 Flat Index, 5.2 IVF_FLAT, 5.3 HNSW, 5.4 Quantization)
6.  Metadata Filtering                 (6.1 What is Metadata, 6.2 Pre-filtering, 6.3 Post-filtering)
7.  ChromaDB                           (7.1 Installation, 7.2 Collections, 7.3 CRUD, 7.4 Query)
8.  FAISS                              (8.1 Installation, 8.2 Index Types, 8.3 Search, 8.4 GPU)
9.  Pinecone                           (9.1 Cloud VDB, 9.2 Index Creation, 9.3 Upsert/Query, 9.4 Namespaces)
10. Weaviate                           (10.1 Introduction, 10.2 Schema, 10.3 GraphQL, 10.4 Hybrid Search)
11. Qdrant                             (11.1 Introduction, 11.2 Collections, 11.3 Payload Filtering)
12. pgvector PostgreSQL                (12.1 Extension, 12.2 Vector Column, 12.3 ivfflat/hnsw Index)
13. Hybrid Search                      (13.1 Dense + Sparse, 13.2 BM25, 13.3 RRF, 13.4 Re-ranking)
14. RAG Integration                    (14.1 Ingestion Pipeline, 14.2 Retrieval, 14.3 Context Injection)
15. Chunking Strategies                (15.1 Fixed-Size, 15.2 Sentence, 15.3 Semantic, 15.4 Overlap)
16. Embedding Pipeline                 (16.1 Document Chunk Embed Store, 16.2 Batch, 16.3 Updates)
17. Query Optimization                 (17.1 Query Embedding, 17.2 Top-K, 17.3 Re-ranking, 17.4 MMR)
18. Vector DB Performance              (18.1 Latency, 18.2 Throughput, 18.3 Scaling, 18.4 Memory)
19. Multi-Modal Vector Search          (19.1 CLIP, 19.2 Image+Text, 19.3 Audio Embeddings)
20. Production Considerations          (20.1 Persistence, 20.2 Backups, 20.3 Auth, 20.4 Monitoring)
21. VDB Comparison                     (21.1 Chroma vs Pinecone, 21.2 FAISS vs Weaviate, 21.3 Choosing)
22. Real-World Projects                (22.1 Document Search, 22.2 FAQ Bot, 22.3 Multimodal Search)
23. Interview Topics                   (23.1 HNSW FAQs, 23.2 Indexing Questions, 23.3 Coding)
24. Best Practices                     (24.1 Chunking, 24.2 Embedding Choice, 24.3 Filtering, 24.4 Scaling)""",

"chromadb": """1.  Introduction to ChromaDB           (1.1 What is ChromaDB, 1.2 Installing, 1.3 Architecture)
2.  ChromaDB Basics                    (2.1 Client, 2.2 Collections, 2.3 Operations)
3.  Documents & Data Management        (3.1 Adding Docs, 3.2 Structure, 3.3 Updating, 3.4 Deleting)
4.  Querying                           (4.1 query, 4.2 n_results, 4.3 include parameters, 4.4 Distances)
5.  Metadata Filtering                 (5.1 where, 5.2 Comparison Operators, 5.3 Logical, 5.4 Advanced)
6.  Embedding Functions                (6.1 Default Function, 6.2 OpenAI, 6.3 Sentence Transformers, 6.4 Custom)
7.  Persistent Storage                 (7.1 PersistentClient, 7.2 Directory, 7.3 Data Persistence, 7.4 Backup)
8.  ChromaDB Server Mode               (8.1 HttpClient, 8.2 Docker, 8.3 Client-Server, 8.4 Auth)
9.  Collections Management             (9.1 Creating, 9.2 Getting, 9.3 Listing, 9.4 Deleting, 9.5 Config)
10. Batch Operations                   (10.1 Batch Add, 10.2 Batch Query, 10.3 Performance)
11. ChromaDB with LangChain            (11.1 Chroma Vectorstore, 11.2 as_retriever, 11.3 RAG Chain)
12. RAG with ChromaDB                  (12.1 Ingestion, 12.2 Retrieval, 12.3 Context Injection, 12.4 Evaluation)
13. HNSW Configuration                 (13.1 hnsw:space, 13.2 M/ef_construction, 13.3 Performance Tuning)
14. ChromaDB Performance               (14.1 Index Size, 14.2 Query Speed, 14.3 Memory, 14.4 Optimization)
15. Testing & Debugging                (15.1 EphemeralClient, 15.2 Peek, 15.3 Count, 15.4 Common Errors)
16. Real-World Projects                (16.1 Document QA, 16.2 FAQ Bot, 16.3 Semantic Search, 16.4 Agent Memory)
17. ChromaDB vs Other VDBs             (17.1 vs Pinecone, 17.2 vs FAISS, 17.3 vs Weaviate, 17.4 Choosing)
18. Best Practices                     (18.1 Collection Design, 18.2 Metadata Schema, 18.3 Embedding Choice)""",

"rag": """1.  Introduction to RAG                 (1.1 What is RAG, 1.2 Architecture, 1.3 Applications)
2.  RAG Fundamentals                    (2.1 LLM Limitations, 2.2 Workflow, 2.3 Components)
3.  Setting Up RAG Environment          (3.1 Installation, 3.2 API Config, 3.3 Project Structure)
4.  Documents & Data Sources            (4.1 Doc Types, 4.2 Loaders, 4.3 Web Scraping, 4.4 APIs)
5.  Document Processing                 (5.1 Text Extraction, 5.2 Preprocessing, 5.3 Cleaning)
6.  Text Chunking                       (6.1 Fixed-Size, 6.2 Sentence, 6.3 Recursive, 6.4 Semantic, 6.5 Overlap)
7.  Embedding Generation                (7.1 OpenAI Embeddings, 7.2 Sentence Transformers, 7.3 Local Models)
8.  Vector Storage                      (8.1 ChromaDB, 8.2 FAISS, 8.3 Pinecone, 8.4 Persistent Storage)
9.  Retrieval                           (9.1 Similarity Search, 9.2 Top-K, 9.3 MMR, 9.4 Hybrid)
10. Query Processing                    (10.1 Query Expansion, 10.2 HyDE, 10.3 Multi-Query, 10.4 Decomposition)
11. Re-ranking                          (11.1 What is Re-ranking, 11.2 Cross-Encoders, 11.3 Cohere, 11.4 LLM Re-rank)
12. Context Injection                   (12.1 Prompt Template, 12.2 Context Window, 12.3 Citation)
13. RAG Pipeline with LangChain         (13.1 Retriever, 13.2 Chain, 13.3 LCEL RAG, 13.4 ConversationalRAG)
14. Conversational RAG                  (14.1 Chat History, 14.2 Follow-up Questions, 14.3 Memory)
15. RAG Evaluation                      (15.1 Context Relevance, 15.2 Faithfulness, 15.3 Answer Quality, 15.4 RAGAS)
16. Hallucination Reduction             (16.1 Grounding, 16.2 Faithfulness Check, 16.3 Self-Consistency)
17. Advanced RAG Patterns               (17.1 Parent-Child, 17.2 Summary Index, 17.3 Graph RAG, 17.4 Self-RAG)
18. Agentic RAG                         (18.1 RAG Agent, 18.2 Tool-Augmented, 18.3 Multi-Step, 18.4 ReAct RAG)
19. RAG for Specific Domains            (19.1 Legal, 19.2 Medical, 19.3 Code, 19.4 Financial)
20. Production RAG                      (20.1 Scalability, 20.2 Caching, 20.3 Monitoring, 20.4 Latency)
21. RAG with Multimodal Data            (21.1 Image+Text, 21.2 Table RAG, 21.3 Audio RAG)
22. RAG Security                        (22.1 Prompt Injection, 22.2 Data Privacy, 22.3 Access Control)
23. Real-World Projects                 (23.1 PDF QA, 23.2 Enterprise Search, 23.3 Customer Support)
24. Interview Topics                    (24.1 Architecture FAQs, 24.2 Chunking Questions, 24.3 Coding)
25. Best Practices                      (25.1 Chunking Strategy, 25.2 Embedding Choice, 25.3 Eval Loop)""",

"prompt_engineering": """1.  Introduction to Prompt Engineering  (1.1 What is PE, 1.2 How LLMs Understand Prompts, 1.3 Apps)
2.  Prompt Engineering Basics           (2.1 Anatomy, 2.2 Basic Techniques, 2.3 Writing Effective Prompts)
3.  Zero-Shot Prompting                 (3.1 Basics, 3.2 Applications, 3.3 Improving)
4.  Few-Shot Prompting                  (4.1 Basics, 4.2 Example Selection, 4.3 Format, 4.4 Dynamic Examples)
5.  Chain-of-Thought CoT               (5.1 What is CoT, 5.2 Zero-Shot CoT, 5.3 Few-Shot CoT, 5.4 Apps)
6.  System Prompts                      (6.1 Role, 6.2 Behavior, 6.3 Persona, 6.4 Constraints)
7.  Structured Output Prompting         (7.1 JSON Output, 7.2 Schema Definition, 7.3 Pydantic Integration)
8.  ReAct Prompting                     (8.1 Reasoning+Acting, 8.2 Thought/Action/Observation, 8.3 Agents)
9.  Prompt Chaining                     (9.1 Sequential Prompts, 9.2 Conditional, 9.3 Parallel, 9.4 Pipelines)
10. Role & Persona Prompting            (10.1 Expert Personas, 10.2 Contextual Roles, 10.3 Consistency)
11. Instruction Formatting              (11.1 Clarity, 11.2 Specificity, 11.3 XML/Markdown, 11.4 Delimiters)
12. Context Management                  (12.1 Context Window, 12.2 Summarization, 12.3 Compression)
13. Self-Consistency                    (13.1 Multiple Completions, 13.2 Voting, 13.3 Applications)
14. Tree-of-Thought ToT                 (14.1 What is ToT, 14.2 Search Strategies, 14.3 vs CoT)
15. Prompt Templates                    (15.1 PromptTemplate, 15.2 ChatPromptTemplate, 15.3 Variables)
16. Meta-Prompting                      (16.1 What is Meta-Prompting, 16.2 Prompt Generators, 16.3 Optimization)
17. Prompt Optimization                 (17.1 Manual, 17.2 A/B Testing, 17.3 DSPy, 17.4 LLM-as-Optimizer)
18. Prompt Security                     (18.1 Injection Attacks, 18.2 Jailbreaking, 18.3 Defense Strategies)
19. Prompt Evaluation                   (19.1 LLM-as-Judge, 19.2 RAGAS, 19.3 Human Evaluation)
20. Prompt Engineering for Agents       (20.1 Tool Descriptions, 20.2 Memory, 20.3 Planning, 20.4 Multi-Agent)
21. Model-Specific Techniques           (21.1 OpenAI Best Practices, 21.2 Claude XML, 21.3 Gemini)
22. Real-World Projects                 (22.1 Extraction Pipeline, 22.2 Classification, 22.3 RAG Prompts)
23. Interview Topics                    (23.1 Technique FAQs, 23.2 Evaluation, 23.3 Coding Problems)
24. Best Practices                      (24.1 Iteration, 24.2 Documentation, 24.3 Version Control)""",

"langchain": """1.  Introduction to LangChain           (1.1 What is LangChain, 1.2 Architecture, 1.3 Installing)
2.  LangChain Fundamentals              (2.1 Core Concepts, 2.2 Message Types, 2.3 Basic Model Interaction)
3.  Working with LLMs & Chat Models     (3.1 Model Providers, 3.2 Configuration, 3.3 Parameters)
4.  Messages & Conversation             (4.1 HumanMessage, 4.2 AIMessage, 4.3 SystemMessage, 4.4 History)
5.  Prompt Templates                    (5.1 PromptTemplate, 5.2 ChatPromptTemplate, 5.3 Variables, 5.4 Dynamic)
6.  Output Parsers                      (6.1 StrOutputParser, 6.2 JsonOutputParser, 6.3 Pydantic, 6.4 Custom)
7.  LCEL LangChain Expression Language  (7.1 Pipe Operator, 7.2 Runnable, 7.3 RunnableParallel, 7.4 Branching)
8.  Chains                              (8.1 Sequential, 8.2 Parallel, 8.3 Branching, 8.4 Custom Chains)
9.  Structured Output                   (9.1 with_structured_output, 9.2 Pydantic, 9.3 Function Calling)
10. Tools & Tool Calling                (10.1 @tool, 10.2 StructuredTool, 10.3 Tool Schema, 10.4 Binding)
11. Agents                              (11.1 ReAct Agent, 11.2 create_react_agent, 11.3 AgentExecutor)
12. Memory & State                      (12.1 ConversationHistory, 12.2 MessagePlaceholder, 12.3 External Memory)
13. Retrievers                          (13.1 VectorStoreRetriever, 13.2 Ensemble, 13.3 MultiQuery, 13.4 ParentDoc)
14. Document Loaders                    (14.1 TextLoader, 14.2 PDFLoader, 14.3 WebLoader, 14.4 Custom)
15. Text Splitters                      (15.1 CharacterSplitter, 15.2 RecursiveSplitter, 15.3 Semantic, 15.4 Token)
16. Vector Stores in LangChain          (16.1 Chroma, 16.2 FAISS, 16.3 Pinecone, 16.4 as_retriever)
17. RAG Chain with LCEL                 (17.1 Full Pipeline, 17.2 Conversational, 17.3 With History, 17.4 Streaming)
18. Streaming                           (18.1 stream, 18.2 astream, 18.3 astream_events)
19. Callbacks & Tracing                 (19.1 Callbacks, 19.2 LangSmith, 19.3 Custom Callbacks)
20. LangSmith Integration               (20.1 Tracing, 20.2 Evaluation, 20.3 Dataset Management)
21. Multi-Modal                         (21.1 Image Input, 21.2 Audio, 21.3 Multi-Modal Chains)
22. LangChain with FastAPI              (22.1 Serving Chains, 22.2 Streaming Endpoints, 22.3 Authentication)
23. Real-World Projects                 (23.1 RAG App, 23.2 Multi-Tool Agent, 23.3 Customer Support Bot)
24. LangChain Best Practices            (24.1 LCEL Patterns, 24.2 Error Handling, 24.3 Testing)
25. Interview Topics                    (25.1 LCEL FAQs, 25.2 Agent Questions, 25.3 Architecture)""",

"langgraph": """1.  Introduction to LangGraph           (1.1 What is LangGraph, 1.2 Why Important, 1.3 Ecosystem)
2.  Installation & Environment          (2.1 Installing, 2.2 Config, 2.3 Verifying)
3.  LangGraph Core Concepts             (3.1 Graph Architecture, 3.2 Building Blocks, 3.3 StateGraph)
4.  State & Annotations                 (4.1 TypedDict State, 4.2 Annotated, 4.3 add_messages, 4.4 Custom)
5.  Nodes                               (5.1 What is a Node, 5.2 Python Functions, 5.3 LLM Nodes, 5.4 Tool Nodes)
6.  Edges                               (6.1 Normal Edges, 6.2 Conditional Edges, 6.3 START/END, 6.4 Dynamic)
7.  Graph Compilation & Invocation      (7.1 compile, 7.2 invoke, 7.3 stream, 7.4 Input/Output)
8.  Tool Calling in LangGraph           (8.1 Tool Binding, 8.2 ToolNode, 8.3 Tool Results, 8.4 Error Handling)
9.  Memory & Persistence                (9.1 MemorySaver, 9.2 Checkpoints, 9.3 thread_id, 9.4 Resuming)
10. Human-in-the-Loop                   (10.1 interrupt_before, 10.2 interrupt_after, 10.3 Update State)
11. Streaming                           (11.1 stream, 11.2 astream, 11.3 astream_events, 11.4 Values/Updates)
12. Subgraphs                           (12.1 What is Subgraph, 12.2 Nested Graphs, 12.3 Communication)
13. Multi-Agent Systems                 (13.1 Supervisor Pattern, 13.2 Agent Network, 13.3 Handoffs)
14. Command & Send                      (14.1 Command, 14.2 Send, 14.3 Dynamic Routing, 14.4 Map-Reduce)
15. Long-Running Agents                 (15.1 Durable Execution, 15.2 Timeout, 15.3 Retry, 15.4 Fault Tolerance)
16. ReAct Agent Pattern                 (16.1 Thought/Action/Obs, 16.2 create_react_agent, 16.3 Custom)
17. Graph State Management              (17.1 Reading State, 17.2 Updating State, 17.3 State Transitions)
18. Conditional Routing                 (18.1 Condition Functions, 18.2 Multi-Way, 18.3 Dynamic Dispatch)
19. Error Handling & Fallbacks          (19.1 try/except in Nodes, 19.2 Fallback Nodes, 19.3 Retry Logic)
20. LangGraph with LangSmith            (20.1 Tracing Graphs, 20.2 Debugging, 20.3 Visualization)
21. LangGraph Studio                    (21.1 Local Studio, 21.2 Graph Visualization, 21.3 Debugging)
22. Production Deployment               (22.1 LangGraph Platform, 22.2 API, 22.3 Scaling)
23. Real-World Projects                 (23.1 RAG Agent, 23.2 Research Agent, 23.3 Customer Support)
24. Interview Topics                    (24.1 Architecture FAQs, 24.2 State Questions, 24.3 vs LangChain)
25. Best Practices                      (25.1 State Design, 25.2 Node Modularity, 25.3 Testing, 25.4 Observability)""",

"langsmith": """1.  Introduction to LangSmith          (1.1 What is LangSmith, 1.2 Why Use It, 1.3 Architecture)
2.  Installation & Setup               (2.1 Installation, 2.2 Account/Project, 2.3 Env Variables, 2.4 First App)
3.  LangSmith Core Concepts            (3.1 Runs, 3.2 Traces, 3.3 Projects, 3.4 Parent-Child Runs)
4.  Tracing Basics                     (4.1 Automatic Tracing, 4.2 Manual Tracing, 4.3 @traceable, 4.4 Context)
5.  Trace Inspection                   (5.1 Viewing Traces, 5.2 Run Details, 5.3 Inputs/Outputs, 5.4 Latency)
6.  LangSmith UI                       (6.1 Projects Dashboard, 6.2 Trace Explorer, 6.3 Run Details, 6.4 Filters)
7.  Datasets                           (7.1 Creating, 7.2 Adding Examples, 7.3 Types, 7.4 Management)
8.  Evaluation Basics                  (8.1 What is Evaluation, 8.2 Evaluators, 8.3 Running, 8.4 Results)
9.  LLM-as-a-Judge                     (9.1 What is LLM Judge, 9.2 Criteria, 9.3 Custom Evaluator, 9.4 Scoring)
10. Prompt Management                  (10.1 Prompt Hub, 10.2 Versioning, 10.3 Pulling, 10.4 Pushing)
11. Experiments                        (11.1 Running Experiments, 11.2 Comparing, 11.3 A/B Testing)
12. Monitoring & Alerts                (12.1 Production Monitoring, 12.2 Feedback, 12.3 Alerts, 12.4 Dashboards)
13. Human Feedback                     (13.1 Adding Feedback, 13.2 Feedback Types, 13.3 UI Feedback)
14. Tracing with LangGraph             (14.1 Graph Tracing, 14.2 Node-Level, 14.3 Visualization)
15. Tracing Custom Code                (15.1 @traceable, 15.2 RunTree, 15.3 Custom Metadata, 15.4 Tags)
16. SDK & API                          (16.1 Python SDK, 16.2 REST API, 16.3 Listing Runs, 16.4 Updating Runs)
17. Annotation Queues                  (17.1 What is Annotation, 17.2 Queuing, 17.3 Reviewing, 17.4 Labeling)
18. Testing with LangSmith             (18.1 Unit Tests, 18.2 Regression Testing, 18.3 CI Integration)
19. Cost & Token Monitoring            (19.1 Token Counts, 19.2 Cost Tracking, 19.3 Budget Alerts)
20. Real-World Applications            (20.1 RAG Evaluation, 20.2 Agent Debugging, 20.3 Prompt Optimization)
21. Best Practices                     (21.1 Project Organization, 21.2 Evaluation Design, 21.3 Monitoring)
22. Interview Topics                   (22.1 Observability FAQs, 22.2 Evaluation Design, 22.3 Alternatives)""",

"pydantic": """1.  Introduction to Pydantic           (1.1 What is Pydantic, 1.2 Installing, 1.3 Basics: BaseModel)
2.  Pydantic Models                    (2.1 Creating Models, 2.2 Data Types, 2.3 Examples)
3.  Type Validation                    (3.1 Basic, 3.2 Automatic Conversion, 3.3 Strict Mode)
4.  Field Constraints                  (4.1 Field, 4.2 String Constraints, 4.3 Numeric, 4.4 Collections)
5.  Optional & Default Values          (5.1 Optional, 5.2 Defaults, 5.3 Default Factory, 5.4 Required vs Optional)
6.  Nested Models                      (6.1 Model Inside Model, 6.2 Lists of Models, 6.3 Recursive Models)
7.  Validators                         (7.1 @field_validator, 7.2 @model_validator, 7.3 mode before/after)
8.  Custom Types                       (8.1 Annotated, 8.2 Custom Validators, 8.3 Arbitrary Types)
9.  Model Configuration                (9.1 model_config, 9.2 ConfigDict, 9.3 JSON Mode, 9.4 Extra Fields)
10. Serialization                      (10.1 model_dump, 10.2 model_dump_json, 10.3 exclude/include)
11. Deserialization                    (11.1 model_validate, 11.2 From Dict, 11.3 From JSON, 11.4 From ORM)
12. JSON Schema                        (12.1 model_json_schema, 12.2 Schema Generation, 12.3 OpenAPI)
13. Pydantic with FastAPI              (13.1 Request Body, 13.2 Response Models, 13.3 Path/Query Params)
14. Pydantic for LLM Structured Output (14.1 OpenAI .parse, 14.2 with_structured_output, 14.3 Instructor)
15. Pydantic for Data Validation       (15.1 API Validation, 15.2 Config Files, 15.3 Data Pipelines)
16. Environment Settings               (16.1 BaseSettings, 16.2 .env Files, 16.3 Secrets, 16.4 pydantic-settings)
17. Pydantic v1 vs v2                  (17.1 Migration, 17.2 Breaking Changes, 17.3 New Features)
18. Performance                        (18.1 Validation Speed, 18.2 Lazy Validation, 18.3 Profiling)
19. Real-World Projects                (19.1 LLM Output Schema, 19.2 RAG Config, 19.3 FastAPI App)
20. Interview Topics                   (20.1 Validation FAQs, 20.2 Field Constraints, 20.3 Coding Problems)
21. Best Practices                     (21.1 Model Design, 21.2 Error Messages, 21.3 Reusability)""",

"openai_api": """1.  Introduction to OpenAI              (1.1 What is OpenAI, 1.2 Platform, 1.3 Installing SDK)
2.  OpenAI Models                       (2.1 Understanding Models, 2.2 Selection, 2.3 Configuration)
3.  OpenAI API Fundamentals             (3.1 API Basics, 3.2 SDK, 3.3 Operations)
4.  Responses API                       (4.1 Introduction, 4.2 Basic Text, 4.3 Parameters, 4.4 Response Object)
5.  Chat Completions API                (5.1 Messages, 5.2 Roles, 5.3 System Prompt, 5.4 Multi-turn)
6.  Streaming                           (6.1 stream=True, 6.2 Parsing Chunks, 6.3 Async Streaming)
7.  Structured Output                   (7.1 JSON Mode, 7.2 response_format, 7.3 Pydantic .parse)
8.  Tool Calling Function Calling       (8.1 Defining Tools, 8.2 Tool Call Round-trip, 8.3 Parallel, 8.4 Errors)
9.  Embeddings API                      (9.1 text-embedding-3-small/large, 9.2 Usage, 9.3 Similarity, 9.4 RAG)
10. Moderation API                      (10.1 omni-moderation, 10.2 Categories, 10.3 Integration)
11. Images API DALL-E                   (11.1 generate, 11.2 edit, 11.3 variation, 11.4 Parameters)
12. Speech & Audio                      (12.1 TTS, 12.2 Whisper STT, 12.3 Voices, 12.4 Streaming Audio)
13. Files API                           (13.1 Uploading, 13.2 Listing, 13.3 Deleting, 13.4 Purpose)
14. Batch API                           (14.1 What is Batch, 14.2 Creating, 14.3 Polling, 14.4 50% Cost Savings)
15. Fine-tuning API                     (15.1 Preparing Data, 15.2 Creating Job, 15.3 Monitoring, 15.4 Using)
16. Assistants API                      (16.1 Assistants, 16.2 Threads, 16.3 Messages, 16.4 Runs)
17. Reasoning Models o1/o3              (17.1 Reasoning Tokens, 17.2 effort Parameter, 17.3 When to Use)
18. Async with AsyncOpenAI              (18.1 AsyncOpenAI, 18.2 await, 18.3 FastAPI Integration)
19. Token Counting tiktoken             (19.1 tiktoken, 19.2 Counting Before Sending, 19.3 Cost Estimation)
20. Error Handling                      (20.1 APIError, 20.2 RateLimitError, 20.3 Retries, 20.4 Backoff)
21. Rate Limits & Quotas               (21.1 RPM/TPM, 21.2 Tiers, 21.3 Handling, 21.4 Exponential Backoff)
22. Cost Management                    (22.1 Model Pricing, 22.2 Token Budget, 22.3 Caching, 22.4 Optimization)
23. OpenAI for RAG                     (23.1 Embedding Pipeline, 23.2 Retrieval Prompt, 23.3 Citation)
24. OpenAI Agents                      (24.1 Tool-Augmented, 24.2 ReAct, 24.3 Multi-Step, 24.4 LangGraph)
25. Production Patterns                (25.1 System Design, 25.2 Caching, 25.3 Monitoring, 25.4 Testing)
26. Real-World Projects                (26.1 Chatbot, 26.2 Document QA, 26.3 Code Assistant, 26.4 Pipeline)
27. Interview Topics                   (27.1 API FAQs, 27.2 Cost/Token, 27.3 Architecture, 27.4 Coding)
28. Best Practices                     (28.1 Prompt Design, 28.2 Error Handling, 28.3 Security, 28.4 Testing)""",

"claude_api": """1.  Introduction to Anthropic AI        (1.1 What is Anthropic, 1.2 Claude API, 1.3 SDK, 1.4 First Request)
2.  Claude Models                       (2.1 Model Families, 2.2 Selection, 2.3 Configuration)
3.  First Claude Application            (3.1 Basic Text Gen, 3.2 Message Structure, 3.3 Conversation)
4.  Prompt Engineering with Claude      (4.1 Fundamentals, 4.2 Techniques, 4.3 Advanced, 4.4 Claude Optimization)
5.  System Prompts                      (5.1 System Instructions, 5.2 Design, 5.3 Examples)
6.  Multi-turn Conversations            (6.1 History, 6.2 Context, 6.3 Chatbot, 6.4 State)
7.  Streaming                           (7.1 stream, 7.2 Parsing Events, 7.3 Async Streaming)
8.  Tool Use Function Calling           (8.1 Defining Tools, 8.2 tool_use Content Block, 8.3 tool_result, 8.4 Round-trip)
9.  Extended Thinking                   (9.1 What is Extended Thinking, 9.2 thinking_budget, 9.3 Content Block, 9.4 vs o1)
10. Structured Output                   (10.1 XML Prompting, 10.2 JSON via Prompts, 10.3 Prefill, 10.4 Pydantic)
11. Prompt Caching                      (11.1 What is Caching, 11.2 cache_control, 11.3 Read/Write Tokens, 11.4 Cost)
12. Vision Image Input                  (12.1 Image Content Block, 12.2 Base64, 12.3 URL, 12.4 Multiple Images)
13. Document Processing                 (13.1 PDF Processing, 13.2 Long Documents, 13.3 Summarization)
14. Batch API                           (14.1 MessageBatch, 14.2 Creating, 14.3 Polling, 14.4 50% Discount)
15. Error Handling                      (15.1 APIStatusError, 15.2 RateLimitError, 15.3 Retries, 15.4 Overload)
16. Async with AsyncAnthropic           (16.1 AsyncAnthropic, 16.2 await, 16.3 FastAPI Integration)
17. Token Counting                      (17.1 count_tokens, 17.2 Before Sending, 17.3 Cost Estimation)
18. Claude for RAG                      (18.1 Embedding+Retrieval, 18.2 Context Injection, 18.3 Citation)
19. Claude Agents                       (19.1 Tool-Augmented, 19.2 ReAct, 19.3 Computer Use, 19.4 Multi-Agent)
20. Computer Use                        (20.1 What is Computer Use, 20.2 bash/str_replace/computer Tools)
21. Cost Management                    (21.1 Model Pricing, 21.2 Caching ROI, 21.3 Batch Discount, 21.4 Budget)
22. OpenAI vs Claude Migration         (22.1 Key Differences, 22.2 Migration Patterns, 22.3 Gotchas)
23. Production Patterns                (23.1 System Design, 23.2 Caching, 23.3 Monitoring, 23.4 Testing)
24. Real-World Projects                (24.1 Document Analysis, 24.2 Coding Assistant, 24.3 RAG App)
25. Interview Topics                   (25.1 Architecture FAQs, 25.2 Caching/Cost, 25.3 vs OpenAI)
26. Best Practices                     (26.1 XML Prompts, 26.2 max_tokens, 26.3 Tool Design, 26.4 Caching)""",

"fastapi": """1.  Introduction to FastAPI             (1.1 What is FastAPI, 1.2 Installing, 1.3 Basics: Routes/Swagger)
2.  HTTP & API Fundamentals             (2.1 HTTP Basics, 2.2 Methods, 2.3 Status Codes, 2.4 Headers)
3.  Path Operations & Routing           (3.1 GET/POST/PUT/DELETE, 3.2 Path Params, 3.3 Query Params, 3.4 Optional)
4.  Request Body & Pydantic             (4.1 Request Body, 4.2 Pydantic Models, 4.3 Nested, 4.4 Validation)
5.  Response Models                     (5.1 response_model, 5.2 Status Codes, 5.3 Custom Responses, 5.4 JSONResponse)
6.  Dependency Injection                (6.1 Depends, 6.2 Common Dependencies, 6.3 Database Deps, 6.4 Auth)
7.  Async & Concurrent Requests         (7.1 async def, 7.2 await, 7.3 Concurrent Requests, 7.4 Background Tasks)
8.  Background Tasks                    (8.1 BackgroundTasks, 8.2 Use Cases, 8.3 Celery Integration)
9.  Authentication & Security           (9.1 OAuth2, 9.2 JWT, 9.3 API Keys, 9.4 HTTPBearer)
10. Database Integration                (10.1 SQLAlchemy, 10.2 SQLite, 10.3 PostgreSQL, 10.4 Session)
11. File Upload & Download              (11.1 UploadFile, 11.2 File, 11.3 Serving Files, 11.4 Static)
12. Middleware                          (12.1 What is Middleware, 12.2 CORS, 12.3 Custom Middleware, 12.4 Logging)
13. Error Handling                      (13.1 HTTPException, 13.2 Custom Exceptions, 13.3 Exception Handlers)
14. Testing                             (14.1 TestClient, 14.2 pytest, 14.3 Fixtures, 14.4 Mocking)
15. Routers & Large Applications        (15.1 APIRouter, 15.2 Including Routers, 15.3 Prefix/Tags)
16. WebSockets                          (16.1 WebSocket Endpoint, 16.2 Connections, 16.3 Broadcasting)
17. Streaming Responses                 (17.1 StreamingResponse, 17.2 LLM Streaming, 17.3 Server-Sent Events)
18. FastAPI with LLMs                   (18.1 Streaming Chat, 18.2 Async OpenAI, 18.3 Claude API, 18.4 RAG)
19. FastAPI with LangChain              (19.1 Serving LCEL Chains, 19.2 Streaming, 19.3 LangGraph API)
20. Deployment                          (20.1 Uvicorn, 20.2 Gunicorn, 20.3 Docker, 20.4 Cloud)
21. Performance & Optimization          (21.1 Async DB, 21.2 Caching, 21.3 Connection Pooling)
22. Real-World Projects                 (22.1 LLM API, 22.2 RAG API, 22.3 Auth Service, 22.4 Microservice)
23. Interview Topics                    (23.1 ASGI FAQs, 23.2 async vs sync, 23.3 FastAPI vs Flask)
24. Best Practices                      (24.1 Project Structure, 24.2 Dependency Injection, 24.3 Testing)""",

"docker": """1.  Introduction to Docker              (1.1 What is Docker, 1.2 Architecture, 1.3 Installing)
2.  Docker Basics                       (2.1 Commands, 2.2 Running First Container, 2.3 Lifecycle)
3.  Docker Images                       (3.1 What is Image, 3.2 Pulling, 3.3 Listing, 3.4 Layers)
4.  Dockerfile                          (4.1 What is Dockerfile, 4.2 Instructions, 4.3 Building, 4.4 Optimizing)
5.  Dockerfile Advanced                 (5.1 Multi-Stage Builds, 5.2 .dockerignore, 5.3 Layer Caching)
6.  Container Management                (6.1 run/start/stop, 6.2 exec, 6.3 logs, 6.4 inspect)
7.  Docker Networking                   (7.1 Bridge, 7.2 Host, 7.3 Overlay, 7.4 Custom Networks)
8.  Docker Volumes                      (8.1 Named Volumes, 8.2 Bind Mounts, 8.3 tmpfs, 8.4 Persistence)
9.  Docker Registry                     (9.1 Docker Hub, 9.2 push/pull, 9.3 Private Registry, 9.4 GitHub CR)
10. Docker Compose                      (10.1 What is Compose, 10.2 docker-compose.yml, 10.3 Services)
11. Docker Compose Advanced             (11.1 Depends_on, 11.2 Health Checks, 11.3 Env Vars, 11.4 Profiles)
12. Docker for Python Apps              (12.1 FastAPI Dockerfile, 12.2 Requirements, 12.3 .env, 12.4 Running)
13. Docker for ML/AI                    (13.1 PyTorch/TF Image, 13.2 GPU Support, 13.3 Model Serving)
14. Docker for LLM Applications         (14.1 Multi-service Stack, 14.2 ChromaDB, 14.3 Redis, 14.4 API Gateway)
15. GPU & CUDA in Docker               (15.1 nvidia-docker, 15.2 --gpus, 15.3 CUDA Images, 15.4 Verification)
16. Environment Variables & Secrets    (16.1 ENV, 16.2 ARG, 16.3 .env Files, 16.4 Docker Secrets)
17. Docker Security                    (17.1 Non-root User, 17.2 Read-only, 17.3 Vulnerability Scanning)
18. Image Optimization                 (18.1 Alpine Base, 18.2 Layer Caching, 18.3 Multi-Stage, 18.4 Size)
19. Docker in CI/CD                    (19.1 GitHub Actions, 19.2 Building in CI, 19.3 Push/Tag, 19.4 Deploy)
20. Container Orchestration Intro      (20.1 What is Orchestration, 20.2 Kubernetes Overview, 20.3 Swarm)
21. Docker for Production              (21.1 Restart Policies, 21.2 Resource Limits, 21.3 Logging, 21.4 Monitoring)
22. Real-World Projects                (22.1 FastAPI + ChromaDB, 22.2 ML Pipeline, 22.3 Multi-Service AI App)
23. Interview Topics                   (23.1 Image vs Container, 23.2 Layer Caching, 23.3 Compose FAQs)
24. Best Practices                     (24.1 Dockerfile Hygiene, 24.2 Security, 24.3 Compose Patterns, 24.4 CI/CD)""",
}


# ─── Read each MD for the prompt text only ───────────────────────
def read_md(folder, filename):
    path = os.path.join(BASE, folder, filename)
    if not os.path.exists(path):
        print(f"  WARN: {path} not found")
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    # Strip the trailing "How to use it:" footer line
    footer_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("**How to use it:**"):
            footer_idx = i
            break
    if footer_idx is not None:
        lines = lines[:footer_idx]
    return "\n".join(lines).strip()

# ─── Build JS data array ─────────────────────────────────────────

js_topics = []
for (folder, tid, label, icon, fname, placeholder) in TOPICS:
    prompt = read_md(folder, fname)
    roadmap = ROADMAPS.get(tid, "")
    folder_key = folder[:2]  # "01", "02" etc.
    js_topics.append({
        "id": tid,
        "folder": folder_key,
        "folderFull": folder,
        "icon": icon,
        "label": label,
        "placeholder": placeholder,
        "roadmap": roadmap,
        "prompt": prompt,
    })

topics_json = json.dumps(js_topics, ensure_ascii=False, indent=2)

# ─── HTML template ───────────────────────────────────────────────
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI/ML Study Prompt Builder</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          }
        }
      }
    }
  </script>
  <style>
    .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scroll::-webkit-scrollbar-track { background: transparent; }
    .custom-scroll::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    .custom-scroll::-webkit-scrollbar-thumb:hover { background: #475569; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans selection:bg-indigo-500 selection:text-white antialiased">
<header class="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border-b border-slate-800/80 px-6 py-4 flex items-center gap-3.5 shadow-md">
  <div class="text-3xl filter drop-shadow">🧠</div>
  <h1 class="text-lg font-bold tracking-tight text-white">AI/ML Study <span class="text-indigo-400 font-extrabold">Prompt Builder</span></h1>
  <div class="ml-auto bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-[11px] font-bold px-3 py-1 rounded-full tracking-wider uppercase">26 TOPICS · AUTO-COPY</div>
</header>
<div class="grid grid-cols-[280px_1fr] flex-1 overflow-hidden h-[calc(100vh-65px)]">
  <aside id="sidebar" class="bg-slate-900/90 border-r border-slate-800 flex flex-col overflow-hidden">
    <div class="p-3 border-b border-slate-800">
      <input type="text" id="search-input" placeholder="🔍 Search 26 topics..." oninput="renderSidebarList()"
             class="w-full bg-slate-950/80 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 rounded-lg px-3 py-2 outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
    </div>
    <div id="sidebar-list" class="p-2 space-y-1 overflow-y-auto flex-1 custom-scroll"></div>
  </aside>
  <main class="panel flex flex-col p-6 md:p-8 gap-5 overflow-y-auto bg-slate-950/40 custom-scroll" id="panel">
    <div class="text-center py-24 px-4 text-slate-400">
      <div class="text-6xl mb-4 animate-bounce">👈</div>
      <h3 class="text-lg font-bold text-slate-200 mb-2">Select a topic to get started</h3>
      <p class="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
        Pick any of the 26 AI/ML topics from the sidebar.<br>
        Paste your roadmap section — the full prompt is<br>
        <strong class="text-emerald-400 font-semibold">automatically copied to clipboard</strong> the moment you paste.
      </p>
    </div>
  </main>
</div>
<div id="toast" class="fixed bottom-6 right-6 bg-gradient-to-r from-emerald-600 to-teal-600 text-white px-5 py-3.5 rounded-xl font-medium text-sm flex items-center gap-2.5 shadow-2xl shadow-emerald-500/30 transform translate-y-20 opacity-0 transition-all duration-300 z-50 pointer-events-none">
  ✅ Prompt copied to clipboard!
</div>
<script>
// ─── DATA (injected by build script) ─────────────────────────────
const TOPICS = TOPICS_JSON_PLACEHOLDER;

const FOLDER_LABELS = {
  "01":"01 · Data Science Foundations",
  "02":"02 · Machine Learning",
  "03":"03 · Data Visualization",
  "04":"04 · Generative AI & LLMs",
  "05":"05 · Deployment",
};

// ─── STATE ───────────────────────────────────────────────────────
let activeTopic = null;
let roadmapOpen = true;

// ─── SIDEBAR ─────────────────────────────────────────────────────
function renderSidebarList() {
  const container = document.getElementById("sidebar-list");
  if (!container) return;
  const query = (document.getElementById("search-input")?.value || "").toLowerCase().trim();
  const groups = {};
  TOPICS.forEach(t => {
    if (!query || t.label.toLowerCase().includes(query) || t.id.toLowerCase().includes(query)) {
      if (!groups[t.folder]) groups[t.folder] = [];
      groups[t.folder].push(t);
    }
  });

  let html = "";
  let count = 0;
  Object.entries(groups).forEach(([f, topics]) => {
    if (topics.length === 0) return;
    count += topics.length;
    html += `<div class="px-3 pt-3 pb-1 text-[10px] font-bold text-slate-400 tracking-wider uppercase">${FOLDER_LABELS[f]}</div>`;
    topics.forEach(t => {
      const isActive = activeTopic?.id === t.id;
      html += `<div class="flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer text-xs transition-all border-l-2 ${
        isActive 
          ? "bg-indigo-600/20 text-white font-medium border-indigo-500 shadow-sm" 
          : "text-slate-300 hover:bg-indigo-600/10 hover:text-white border-transparent"
      }" onclick="selectTopic('${t.id}')">
        <span class="text-sm">${t.icon}</span><span>${t.label}</span>
      </div>`;
    });
  });

  if (count === 0) {
    html = `<div class="px-4 py-8 text-center text-xs text-slate-500">No topics match "${esc(query)}"</div>`;
  }
  container.innerHTML = html;
}

// ─── SELECT ───────────────────────────────────────────────────────
function selectTopic(id) {
  activeTopic = TOPICS.find(t => t.id === id);
  renderSidebarList();
  renderPanel();
}

// ─── PANEL ───────────────────────────────────────────────────────
function renderPanel() {
  if (!activeTopic) return;
  const t = activeTopic;
  document.getElementById("panel").innerHTML = `
    <div class="flex items-start gap-4 p-5 bg-slate-900/90 border border-slate-800 rounded-2xl shadow-sm backdrop-blur">
      <div class="text-4xl p-2.5 bg-slate-800/60 border border-slate-700/50 rounded-xl">${t.icon}</div>
      <div class="flex-1">
        <h2 class="text-xl font-bold text-white tracking-tight">${t.label}</h2>
        <p class="text-xs text-slate-400 mt-1 leading-relaxed">Paste your section below — the full teaching prompt is built and <strong class="text-slate-200 font-semibold">auto-copied to clipboard</strong> instantly.</p>
      </div>
      <div class="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold px-3 py-1 rounded-full whitespace-nowrap">📋 Auto-Copy on Paste</div>
    </div>

    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-sm hover:border-slate-700/80 transition-colors">
      <div class="flex items-center justify-between px-5 py-3.5 border-b border-slate-800/80 bg-slate-900/60 cursor-pointer select-none" onclick="toggleRoadmap()">
        <span class="text-xs font-semibold text-indigo-300 flex items-center gap-2">📚 Roadmap — reference section numbers here</span>
        <span class="text-xs text-slate-400 hover:text-slate-200 transition" id="rm-toggle">${roadmapOpen ? "▼ collapse" : "▶ expand"}</span>
      </div>
      <div class="p-4 font-mono text-xs text-slate-300 leading-relaxed whitespace-pre overflow-x-auto max-h-60 overflow-y-auto bg-slate-950/60 custom-scroll ${roadmapOpen ? '' : 'hidden'}" id="roadmap-body"></div>
    </div>

    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-sm hover:border-slate-700/80 transition-colors">
      <div class="flex items-center justify-between px-5 py-3.5 border-b border-slate-800/80 bg-slate-900/60">
        <span class="text-xs font-semibold text-slate-200 flex items-center gap-2">📝 Paste Your Section Here</span>
        <small class="text-xs text-slate-400 font-mono" id="char-count">— characters in output</small>
      </div>
      <textarea id="section-input"
        placeholder="Paste or type the section number/text here, e.g.: &#10;7. Autograd &amp; Automatic Differentiation&#10;&#10;The full prompt is built + copied automatically on paste."
        oninput="onInput()"
        onpaste="onPaste(event)"
        class="w-full min-h-[140px] bg-slate-950/80 border-0 outline-none text-slate-100 font-mono text-xs p-4 resize-y focus:bg-slate-950 focus:ring-1 focus:ring-indigo-500/50 transition placeholder:text-slate-600"
      ></textarea>
    </div>

    <div class="flex gap-3 items-center">
      <button class="flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-semibold text-sm text-white bg-gradient-to-r from-indigo-600 via-indigo-500 to-purple-600 hover:from-indigo-500 hover:to-purple-500 shadow-lg shadow-indigo-500/20 active:scale-[0.99] transition-all cursor-pointer" id="copy-btn" onclick="copyPrompt()">
        📋 Build &amp; Copy Full Prompt
      </button>
      <button class="flex items-center justify-center gap-2 py-3 px-5 rounded-xl font-semibold text-xs text-slate-300 bg-slate-900 hover:bg-slate-800 border border-slate-700/80 active:scale-[0.99] transition-all cursor-pointer" onclick="clearInput()">
        🗑️ Clear
      </button>
    </div>

    <div class="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden shadow-sm">
      <div class="flex items-center justify-between px-5 py-3.5 border-b border-slate-800/80 bg-slate-900/60 cursor-default">
        <span class="text-xs font-semibold text-indigo-300 flex items-center gap-2">👁️ Preview (first 15 lines of output)</span>
        <span class="text-xs text-slate-500 font-mono" id="char-badge">—</span>
      </div>
      <div class="p-4 font-mono text-xs text-slate-400 leading-relaxed max-h-52 overflow-y-auto whitespace-pre-wrap break-words bg-slate-950/60 custom-scroll" id="preview-body">
        <span class="text-slate-600">Start typing or paste a section to preview the output…</span>
      </div>
    </div>
  `;
  // Set roadmap content via textContent to avoid backtick-in-template-literal issues
  const rmEl = document.getElementById('roadmap-body');
  if (rmEl) rmEl.textContent = t.roadmap || '(no roadmap found)';
}

// ─── ROADMAP TOGGLE ───────────────────────────────────────────────
function toggleRoadmap() {
  const body = document.getElementById("roadmap-body");
  const tog  = document.getElementById("rm-toggle");
  if (!body) return;
  roadmapOpen = !roadmapOpen;
  body.classList.toggle("hidden", !roadmapOpen);
  tog.textContent = roadmapOpen ? "▼ collapse" : "▶ expand";
}

// ─── BUILD PROMPT ─────────────────────────────────────────────────
function buildPrompt(sectionText) {
  if (!activeTopic || !sectionText.trim()) return "";
  // Replace every occurrence of the placeholder (appears twice in most files)
  return activeTopic.prompt.split(activeTopic.placeholder).join(sectionText.trim());
}

// ─── INPUT ────────────────────────────────────────────────────────
function onInput() {
  const val = document.getElementById("section-input")?.value || "";
  updatePreview(val);
}

function onPaste(event) {
  setTimeout(() => {
    const val = document.getElementById("section-input")?.value || "";
    updatePreview(val);
    if (val.trim()) doCopy(val);
  }, 60);
}

function updatePreview(val) {
  const preview = document.getElementById("preview-body");
  const count   = document.getElementById("char-count");
  const badge   = document.getElementById("char-badge");
  if (!preview) return;
  if (!val.trim()) {
    preview.innerHTML = '<span class="text-slate-600">Start typing or paste a section to preview the output…</span>';
    if (count) count.textContent = "— characters in output";
    if (badge) badge.textContent = "—";
    return;
  }
  const full  = buildPrompt(val);
  const chars = full.length.toLocaleString();
  if (count) count.textContent = `${chars} characters in output`;
  if (badge) badge.textContent = `${chars} chars`;
  const lines   = full.split("\n");
  const snippet = lines.slice(0, 15).join("\n");
  preview.innerHTML = esc(snippet) +
    (lines.length > 15 ? `\n<span class="text-indigo-400 italic">…[${lines.length - 15} more lines]</span>` : "");
}

function esc(s) {
  return (s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ─── COPY ─────────────────────────────────────────────────────────
function copyPrompt() {
  const val = document.getElementById("section-input")?.value || "";
  if (!val.trim()) { document.getElementById("section-input")?.focus(); return; }
  doCopy(val);
}

function doCopy(val) {
  const full = buildPrompt(val);
  navigator.clipboard.writeText(full).then(showSuccess, () => {
    const ta = document.createElement("textarea");
    ta.value = full; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); document.body.removeChild(ta);
    showSuccess();
  });
}

let toastT;
function showSuccess() {
  const toast = document.getElementById("toast");
  const btn   = document.getElementById("copy-btn");
  toast.classList.remove("translate-y-20", "opacity-0");
  toast.classList.add("translate-y-0", "opacity-100");
  clearTimeout(toastT);
  toastT = setTimeout(() => {
    toast.classList.remove("translate-y-0", "opacity-100");
    toast.classList.add("translate-y-20", "opacity-0");
  }, 2800);
  if (btn) {
    btn.classList.add("!from-emerald-600", "!to-teal-600", "!shadow-emerald-500/25");
    btn.innerHTML = "✅ Copied!";
    setTimeout(() => { 
      btn.classList.remove("!from-emerald-600", "!to-teal-600", "!shadow-emerald-500/25"); 
      btn.innerHTML = "📋 Build &amp; Copy Full Prompt"; 
    }, 2500);
  }
}

function clearInput() {
  const inp = document.getElementById("section-input");
  if (inp) { inp.value = ""; inp.focus(); }
  updatePreview("");
}

// ─── INIT ─────────────────────────────────────────────────────────
renderSidebarList();
</script>
</body>
</html>
"""

# Inject the JSON data
html = html.replace("TOPICS_JSON_PLACEHOLDER", topics_json)

out_path = os.path.join(BASE, "prompt_builder.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print("Generated: " + out_path)
print("   Size: %.1f KB" % size_kb)
print("   Topics loaded: %d" % len(js_topics))
missing = [t["label"] for t in js_topics if not t["prompt"].strip()]
if missing:
    print("   WARN Missing prompts: " + str(missing))
else:
    print("   All %d prompts embedded OK" % len(js_topics))
