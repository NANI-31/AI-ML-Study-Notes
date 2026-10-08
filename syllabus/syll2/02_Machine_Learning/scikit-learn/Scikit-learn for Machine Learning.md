# Scikit-learn for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Scikit-learn for Machine Learning

### 1.1 Why Scikit-learn is the Standard ML Library
- Unified API across all ML algorithms
- Fastest path from raw data to trained model
- Best cross-validation and model selection tools in Python
- Seamless Pandas and NumPy integration
- Industry and research standard

### 1.2 Role of Scikit-learn in the ML Workflow
- Data preprocessing and feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Pipeline construction for reproducibility
- Model inspection and interpretation
- Deployment preparation

### 1.3 Installing and Setup
- Installing Scikit-learn
- Import conventions
  - `from sklearn.model_selection import train_test_split, cross_val_score`
  - `from sklearn.preprocessing import StandardScaler`
  - `from sklearn.pipeline import Pipeline`
  - `from sklearn.metrics import classification_report`

---

## 2. End-to-end ML Workflow with Scikit-learn

### 2.1 Standard ML Workflow Steps
- Load data — Pandas DataFrame or sklearn datasets
- Explore data — EDA
- Preprocess — scaling, encoding, imputation
- Split — train/validation/test
- Train — `model.fit(X_train, y_train)`
- Evaluate — metrics, cross-validation
- Tune — GridSearchCV or RandomizedSearchCV
- Finalize — refit on full training data
- Predict — `model.predict(X_test)`

### 2.2 Data Representation Convention
- Feature matrix `X` — shape `(n_samples, n_features)`
- Target vector `y` — shape `(n_samples,)` or `(n_samples, n_outputs)`
- All inputs must be numeric — no NaN, no strings
- `dtype=float32` or `float64`

---

## 3. Data Preprocessing for ML

### 3.1 Handling Missing Values
- `SimpleImputer(strategy='mean')` — numerical imputation
- `SimpleImputer(strategy='median')` — robust to outliers
- `SimpleImputer(strategy='most_frequent')` — categorical imputation
- `SimpleImputer(strategy='constant', fill_value=0)`
- `KNNImputer(n_neighbors=5)` — KNN-based imputation
- `IterativeImputer()` — multivariate imputation
- `add_indicator=True` — missingness flag column

### 3.2 Feature Scaling
- `StandardScaler` — zero mean, unit variance
  - Use with: Logistic Regression, SVM, KNN, MLP, PCA
- `MinMaxScaler` — scale to [0, 1]
  - Use with: neural networks, image data
- `RobustScaler` — median and IQR
  - Use with: datasets with outliers
- `MaxAbsScaler` — scale to [-1, 1]
  - Use with: sparse data
- Fit only on training data — no data leakage

### 3.3 Encoding Categorical Features
- `OrdinalEncoder` — ordered categories
- `OneHotEncoder(sparse_output=False, drop='first')` — nominal categories
  - `handle_unknown='ignore'` — for unseen categories
- `LabelEncoder` — target variable encoding only
- `TargetEncoder` — supervised encoding (sklearn 1.3+)
- `pd.get_dummies()` + `df.to_numpy()` — Pandas alternative

### 3.4 Feature Transformation
- `PowerTransformer(method='yeo-johnson')` — normalize skewed features
- `QuantileTransformer(output_distribution='normal')` — map to normal
- `Binarizer(threshold=0.5)` — threshold-based binary
- `PolynomialFeatures(degree=2, interaction_only=False)` — polynomial features
- `SplineTransformer(n_knots=5, degree=3)` — spline features

### 3.5 ColumnTransformer for Mixed Data
- Separate pipelines for numeric and categorical
- `ColumnTransformer(transformers=[('num', scaler, num_cols), ('cat', encoder, cat_cols)])`
- `remainder='passthrough'` — pass other columns through
- `make_column_selector(dtype_include=np.number)` — automatic type-based selection

---

## 4. Feature Engineering for ML

### 4.1 Polynomial Features
- `PolynomialFeatures(degree=2)` — creates x², x₁x₂ terms
- `include_bias=False` — exclude bias term
- `interaction_only=True` — only interaction terms

### 4.2 Custom Feature Transformers
- `FunctionTransformer(np.log1p)` — applying custom function
- Wrapping Pandas operations — `FunctionTransformer(lambda df: df.assign(...))`

### 4.3 Feature Selection
- `VarianceThreshold(threshold=0.01)` — remove near-constant features
- `SelectKBest(f_classif, k=10)` — univariate selection
- `SelectKBest(mutual_info_classif, k=10)` — mutual information
- `RFE(estimator=LogisticRegression(), n_features_to_select=10)` — recursive elimination
- `RFECV(estimator, cv=5, scoring='accuracy')` — CV-based RFE
- `SelectFromModel(RandomForestClassifier())` — model-based selection

---

## 5. Train-Test Splitting Strategies

### 5.1 Basic Split
- `train_test_split(X, y, test_size=0.2, random_state=42)` — random split
- `stratify=y` — maintain class proportion in split
- Holdout set — final evaluation only

### 5.2 Three-way Split
- Train → model fitting
- Validation → hyperparameter tuning
- Test → final evaluation only
- `train_test_split` twice — first for test, then for val

### 5.3 Time Series Split
- `TimeSeriesSplit(n_splits=5)` — forward-chaining splits
- No data leakage from future into past
- Expanding or sliding window

---

## 6. Classification Algorithms

### 6.1 Logistic Regression
- `LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)`
- Binary and multi-class (`multi_class='multinomial'`)
- `predict_proba()` — class probabilities
- `coef_` — feature weights
- When to use — linearly separable, interpretability needed

### 6.2 Support Vector Machine
- `SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)`
- `LinearSVC` — faster for linear kernel
- `kernel='poly'` — polynomial boundary
- `C` — margin trade-off
- When to use — small/medium data, non-linear boundaries

### 6.3 Decision Tree
- `DecisionTreeClassifier(max_depth=5, criterion='gini')`
- Interpretable — `export_graphviz()`, `plot_tree()`
- `feature_importances_`
- Prone to overfitting — control `max_depth`, `min_samples_leaf`

### 6.4 Random Forest
- `RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)`
- Robust to overfitting
- `oob_score=True` — out-of-bag evaluation
- `feature_importances_` — mean decrease in impurity

### 6.5 Gradient Boosting
- `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`
- `HistGradientBoostingClassifier` — histogram-based, faster, handles NaN
- Sequential tree building

### 6.6 K-Nearest Neighbors
- `KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')`
- Scale features before KNN
- Lazy learner — no training phase
- High memory cost for large datasets

### 6.7 Naive Bayes
- `GaussianNB()` — continuous features
- `MultinomialNB(alpha=1.0)` — text count features
- `BernoulliNB()` — binary features
- Fast, good baseline for text classification

### 6.8 MLP Classifier
- `MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)`
- Scale inputs — StandardScaler required
- `early_stopping=True` — prevent overfitting

---

## 7. Regression Algorithms

### 7.1 Linear Regression
- `LinearRegression()` — OLS
- `Ridge(alpha=1.0)` — L2 penalty, handles multicollinearity
- `Lasso(alpha=0.1)` — L1 penalty, sparse solution
- `ElasticNet(alpha=0.1, l1_ratio=0.5)` — L1 + L2
- `coef_` — feature coefficients

### 7.2 Support Vector Regression
- `SVR(C=1.0, epsilon=0.1, kernel='rbf')`
- `epsilon` — tube width (insensitive zone)

### 7.3 Tree-based Regression
- `DecisionTreeRegressor(max_depth=5)`
- `RandomForestRegressor(n_estimators=100)`
- `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)`
- `HistGradientBoostingRegressor` — fast, handles NaN

### 7.4 Polynomial Regression
- `PolynomialFeatures(degree=2)` + `LinearRegression()`
- Build inside a Pipeline

---

## 8. Model Evaluation for ML

### 8.1 Classification Metrics
- `accuracy_score(y_true, y_pred)` — overall accuracy
- `precision_score(y_true, y_pred, average='weighted')` — precision
- `recall_score(y_true, y_pred, average='weighted')` — recall
- `f1_score(y_true, y_pred, average='weighted')` — F1
- `classification_report(y_true, y_pred)` — full breakdown per class
- `confusion_matrix(y_true, y_pred)` — confusion matrix
- `ConfusionMatrixDisplay(cm).plot()` — visual
- `roc_auc_score(y_true, y_score)` — AUC-ROC
- `average_precision_score(y_true, y_score)` — AUC-PR

### 8.2 Regression Metrics
- `mean_squared_error(y_true, y_pred)` — MSE
- `mean_squared_error(y_true, y_pred, squared=False)` — RMSE
- `mean_absolute_error(y_true, y_pred)` — MAE
- `r2_score(y_true, y_pred)` — R²
- `mean_absolute_percentage_error(y_true, y_pred)` — MAPE

### 8.3 ROC and PR Curves
- `roc_curve(y_true, y_score)` — fpr, tpr, thresholds
- `RocCurveDisplay.from_estimator(model, X_test, y_test)` — plot
- `precision_recall_curve(y_true, y_score)` — precision, recall
- `PrecisionRecallDisplay.from_estimator(model, X_test, y_test)` — plot

---

## 9. Cross-Validation for ML

### 9.1 Cross-Validation Basics
- `cross_val_score(model, X, y, cv=5, scoring='f1_weighted')` — scores per fold
- `cross_validate(model, X, y, cv=5, return_train_score=True)` — detailed results
- `cross_val_predict(model, X, y, cv=5)` — OOF predictions

### 9.2 Stratified Cross-Validation
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Maintains class proportion per fold
- Critical for imbalanced classification

### 9.3 Repeated Cross-Validation
- `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`
- More stable estimates
- Mean and std across repeats

### 9.4 Cross-Validation for Time Series
- `TimeSeriesSplit(n_splits=5)`
- No data leakage from future
- Gap parameter for walk-forward validation

---

## 10. Hyperparameter Tuning

### 10.1 GridSearchCV
- `GridSearchCV(estimator, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)`
- `param_grid = {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}`
- `best_params_`, `best_score_`, `best_estimator_`
- `refit=True` — refit on full training set with best params
- `cv_results_` — detailed results DataFrame

### 10.2 RandomizedSearchCV
- `RandomizedSearchCV(estimator, param_distributions, n_iter=50, cv=5)`
- `param_distributions` — `scipy.stats` distributions or lists
  - `{'C': scipy.stats.loguniform(0.01, 10)}`
  - `{'n_estimators': randint(50, 500)}`
- Faster than grid search for large spaces

### 10.3 Pipeline + GridSearchCV
- `pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])`
- `param_grid = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['rbf', 'linear']}`
- `GridSearchCV(pipe, param_grid, cv=5)`

### 10.4 Successive Halving
- `HalvingGridSearchCV` — eliminates poor configs quickly
- `HalvingRandomSearchCV` — random + halving
- `factor` — proportion of candidates retained per round

---

## 11. Pipelines for ML

### 11.1 Basic Pipeline
- `Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler()), ('clf', LogisticRegression())])`
- `make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())`
- Prevents data leakage during cross-validation
- `pipe.fit(X_train, y_train)` → `pipe.predict(X_test)`

### 11.2 Full Preprocessing Pipeline
```
Pipeline:
  ColumnTransformer:
    - numeric: SimpleImputer → StandardScaler
    - categorical: SimpleImputer → OneHotEncoder
  Classifier
```
- `make_column_selector(dtype_include=np.number)` — numeric columns
- `make_column_selector(dtype_include=object)` — categorical columns

### 11.3 Pipeline Inspection
- `pipe.named_steps['clf']` — accessing classifier
- `pipe['clf']` — shorthand
- `pipe.set_params(clf__C=10)` — changing parameter
- `pipe[-1]` — last step

---

## 12. Handling Class Imbalance

### 12.1 Class Weight Parameter
- `LogisticRegression(class_weight='balanced')`
- `SVC(class_weight='balanced')`
- `RandomForestClassifier(class_weight='balanced')`
- `class_weight='balanced'` — inversely proportional to class frequency

### 12.2 Threshold Adjustment
- `model.predict_proba(X)[:, 1] > threshold` — custom threshold
- `precision_recall_curve()` — finding optimal threshold

### 12.3 Resampling (with imbalanced-learn)
- `SMOTE()` — synthetic minority oversampling
- `RandomOverSampler()` — random oversampling
- `RandomUnderSampler()` — random undersampling
- `imbalanced-learn` Pipeline — `imblearn.pipeline.Pipeline`

### 12.4 Evaluation with Imbalance
- Avoid `accuracy_score` — use `f1_score`, `roc_auc_score`
- `average='macro'` — treats all classes equally
- `confusion_matrix` — inspect per-class counts

---

## 13. Ensemble Methods for ML

### 13.1 Random Forest Tuning
- `n_estimators` — more trees reduce variance (diminishing returns >200)
- `max_features` — `'sqrt'` for classification, `'log2'`
- `max_depth` — prevents overfitting
- `min_samples_leaf` — smoothing
- `oob_score=True` — free validation estimate

### 13.2 Gradient Boosting Tuning
- `n_estimators` — number of trees
- `learning_rate` — shrinkage (0.01–0.3)
- `max_depth` — tree depth (3–6 typically)
- `subsample` — stochastic gradient boosting (0.5–0.8)
- `min_samples_leaf`, `min_child_weight`

### 13.3 Stacking
- `StackingClassifier(estimators=[('rf', RandomForest()), ('svm', SVC())], final_estimator=LogisticRegression())`
- Meta-features generated via cross-validation
- `passthrough=True` — also passes raw features

### 13.4 Voting Ensemble
- `VotingClassifier(estimators=[...], voting='soft')` — probability averaging
- `voting='hard'` — majority vote
- `weights` — custom model weights

---

## 14. Feature Importance and Model Inspection

### 14.1 Tree-based Importance
- `rf.feature_importances_` — mean decrease in impurity
- `pd.Series(importances, index=feature_names).sort_values().plot.barh()`
- Biased toward high-cardinality features

### 14.2 Permutation Importance
- `permutation_importance(model, X_test, y_test, n_repeats=10)`
- `result.importances_mean` — mean importance
- `result.importances_std` — variability
- Model-agnostic, uses validation set

### 14.3 Partial Dependence Plots
- `PartialDependenceDisplay.from_estimator(model, X, features=[0, 1, (0,1)])`
- `kind='average'` — PDP
- `kind='individual'` — ICE plots
- `kind='both'` — PDP + ICE overlay

### 14.4 Linear Model Coefficients
- `lr.coef_` — feature weights
- Standardize before interpreting coefficients
- `pd.Series(lr.coef_[0], index=feature_names)`

---

## 15. Dimensionality Reduction for ML

### 15.1 PCA for ML
- `PCA(n_components=0.95)` — retain 95% variance
- `PCA(n_components=50)` — fixed components
- `pca.explained_variance_ratio_` — variance per component
- Using as preprocessing step in Pipeline
- `PCA + LogisticRegression` — pipeline

### 15.2 t-SNE for Visualization
- `TSNE(n_components=2, perplexity=30, random_state=42)`
- For visualization only — not for preprocessing
- Colored by class label for cluster inspection

### 15.3 Feature Selection as Dimensionality Reduction
- `SelectFromModel(lasso)` — L1-based selection
- `RFE` — recursive feature removal
- Reduces noise and training time

---

## 16. Clustering for ML

### 16.1 K-Means
- `KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)`
- Elbow method — `inertia_` vs k
- Silhouette score — `silhouette_score(X, labels)`
- Cluster labels as features for downstream tasks

### 16.2 DBSCAN
- `DBSCAN(eps=0.5, min_samples=5)`
- Discovers arbitrary shapes
- `labels_` — `-1` for noise points
- Choosing `eps` with k-distance graph

### 16.3 Gaussian Mixture Models
- `GaussianMixture(n_components=3, covariance_type='full')`
- Soft cluster assignments — `predict_proba()`
- BIC for model selection — `gm.bic(X)`

---

## 17. Model Persistence

### 17.1 Saving and Loading Models
- `import joblib`
- `joblib.dump(model, 'model.pkl')` — saving
- `model = joblib.load('model.pkl')` — loading
- Saving entire Pipeline — `joblib.dump(pipe, 'pipeline.pkl')`
- `import pickle` — alternative

### 17.2 Model Versioning
- Include sklearn version in saved model metadata
- `sklearn.__version__` — check compatibility
- Save training date, parameters, and metrics alongside model

---

## 18. Scikit-learn Best Practices for ML

### 18.1 Preventing Data Leakage
- Always use `Pipeline` — fit only on training data
- `train_test_split` before any preprocessing
- `cross_validate` inside Pipeline — never manual preprocessing
- Stratify splits for classification

### 18.2 Reproducibility
- Set `random_state` in all stochastic components
- `np.random.seed(42)` — global seed
- Log all hyperparameters alongside results

### 18.3 Evaluation Rigor
- Use cross-validation — not single train/val split
- Report mean ± std across folds
- Separate validation from test — tune on val, report on test
- Use multiple metrics — accuracy + F1 + AUC

### 18.4 Common Mistakes
- Fitting scaler on full dataset — data leakage
- Tuning hyperparameters on test set — overfitting to test
- Ignoring class imbalance — misleading accuracy
- Not stratifying for imbalanced classification split
- Using `accuracy` as sole metric for imbalanced data

---

## 19. Scikit-learn Interview Topics for ML

### 19.1 Frequently Asked Questions
- What is the difference between `Pipeline` and `ColumnTransformer`?
- How to prevent data leakage in scikit-learn?
- When to use `GridSearchCV` vs `RandomizedSearchCV`?
- How to handle imbalanced classification?
- What is the difference between `feature_importances_` and `permutation_importance`?
- How does `StratifiedKFold` differ from `KFold`?

### 19.2 ML-Oriented Coding Problems
- Build a complete preprocessing + classification pipeline
- Tune hyperparameters with GridSearchCV inside a Pipeline
- Implement stratified K-fold for imbalanced dataset
- Compare Random Forest, SVM, and Logistic Regression with cross-validation
- Build a stacking ensemble and compare to base models

---

## 20. Real-World ML Projects with Scikit-learn

### 20.1 Beginner Projects
- Titanic survival classification
- Boston / California housing regression
- Iris multi-class classification

### 20.2 Intermediate Projects
- End-to-end pipeline with mixed data types (numeric + categorical)
- Customer churn prediction with hyperparameter tuning
- Credit card fraud detection with class imbalance handling

### 20.3 Advanced Projects
- Automated model comparison and selection pipeline
- Stacking ensemble for structured data competition
- Full ML experiment with preprocessing, tuning, evaluation, and persistence

---

## 21. Scikit-learn Roadmap for ML Engineers

### 21.1 Beginner Level
- `train_test_split`, `fit()`, `predict()`, `score()`
- `StandardScaler`, `OneHotEncoder`, `SimpleImputer`
- `LogisticRegression`, `RandomForestClassifier`, `LinearRegression`
- Basic metrics — `accuracy_score`, `classification_report`

### 21.2 Intermediate Level
- `Pipeline` and `ColumnTransformer`
- `cross_val_score` and `StratifiedKFold`
- `GridSearchCV` with pipeline
- `feature_importances_`, `permutation_importance`
- `RandomForest`, `GradientBoosting`, `SVM`

### 21.3 Advanced Level
- `StackingClassifier`, `VotingClassifier`
- `HalvingRandomSearchCV`
- `PartialDependenceDisplay`, calibration
- Model persistence with joblib
- Custom transformers and estimators
