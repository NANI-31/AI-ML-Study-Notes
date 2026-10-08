# Scikit-learn Roadmap / Syllabus

---

## 1. Introduction to Scikit-learn

### 1.1 What is Scikit-learn?
- Definition of Scikit-learn
- Open-source machine learning library for Python
- Built on NumPy, SciPy, and Matplotlib
- Consistent API across all algorithms
- Advantages of Scikit-learn
  - Unified `fit()`, `transform()`, `predict()` API
  - Extensive algorithm coverage
  - Built-in cross-validation and model selection
  - Pipeline support
  - Excellent documentation and community
- Real-world applications
  - Classification
  - Regression
  - Clustering
  - Dimensionality reduction
  - Model selection and evaluation
  - Preprocessing and feature engineering

### 1.2 Installing Scikit-learn
- Installing with pip
- Installing with conda
- Verifying installation
- Importing conventions
  - `import sklearn`
  - `from sklearn.linear_model import LinearRegression`
  - `from sklearn.model_selection import train_test_split`
  - `from sklearn.preprocessing import StandardScaler`

### 1.3 Scikit-learn Design Principles
- Estimator API — `fit()` method
- Predictor API — `predict()` and `predict_proba()`
- Transformer API — `transform()` and `fit_transform()`
- Pipeline API — chaining steps
- Consistent parameter naming conventions
- `get_params()` and `set_params()`
- `clone()` — deep copy of estimator

---

## 2. Data Loading and Datasets

### 2.1 Built-in Datasets
- `load_iris()` — multi-class classification
- `load_digits()` — image classification
- `load_boston()` / `load_diabetes()` — regression
- `load_breast_cancer()` — binary classification
- `load_wine()` — multi-class classification
- `load_linnerud()` — multi-output regression
- Dataset structure — `data`, `target`, `feature_names`, `target_names`, `DESCR`

### 2.2 Generated Datasets
- `make_classification()` — synthetic classification
- `make_regression()` — synthetic regression
- `make_blobs()` — clustering data
- `make_circles()` — non-linear classification
- `make_moons()` — non-linear classification
- `make_gaussian_quantiles()` — Gaussian quantile classification
- `n_samples`, `n_features`, `n_classes`, `noise`, `random_state`

### 2.3 Real-world Dataset Fetching
- `fetch_california_housing()` — regression
- `fetch_openml()` — OpenML repository datasets
- `fetch_20newsgroups()` — text classification
- `fetch_rcv1()` — Reuters corpus

### 2.4 Loading External Data
- Using Pandas `read_csv()` and converting to NumPy
- Feature matrix `X` and target vector `y` convention
- `df.values` and `df.to_numpy()` for conversion
- Ensuring correct dtypes

---

## 3. Data Preprocessing

### 3.1 Feature Scaling
- `StandardScaler` — zero mean, unit variance
  - `fit()`, `transform()`, `fit_transform()`
  - `inverse_transform()`
- `MinMaxScaler` — scale to [0, 1]
  - `feature_range` parameter
- `MaxAbsScaler` — scale to [-1, 1]
- `RobustScaler` — median and IQR based
  - Robust to outliers
- `Normalizer` — row-wise normalization
  - `norm` — `'l1'`, `'l2'`, `'max'`

### 3.2 Encoding Categorical Features
- `LabelEncoder` — encoding target labels
  - `fit()`, `transform()`, `inverse_transform()`
- `OrdinalEncoder` — encoding ordinal features
- `OneHotEncoder` — one-hot encoding
  - `sparse_output` — sparse matrix output
  - `drop` — `'first'` to avoid dummy trap
  - `handle_unknown='ignore'`
- `LabelBinarizer` — binarizing multi-class labels

### 3.3 Handling Missing Values
- `SimpleImputer` — basic imputation
  - `strategy` — `'mean'`, `'median'`, `'most_frequent'`, `'constant'`
  - `fill_value` — for constant strategy
  - `add_indicator` — add missing indicator column
- `KNNImputer` — KNN-based imputation
  - `n_neighbors`
- `IterativeImputer` — multivariate imputation

### 3.4 Polynomial and Interaction Features
- `PolynomialFeatures` — polynomial feature generation
  - `degree` — polynomial degree
  - `interaction_only` — interaction terms only
  - `include_bias` — include bias column

### 3.5 Discretization and Binning
- `KBinsDiscretizer` — bin continuous features
  - `n_bins` — number of bins
  - `encode` — `'onehot'`, `'ordinal'`
  - `strategy` — `'uniform'`, `'quantile'`, `'kmeans'`
- `Binarizer` — threshold-based binarization

### 3.6 Feature Selection Preprocessing
- `VarianceThreshold` — remove low-variance features
  - `threshold` parameter
- Removing features with all same values

### 3.7 Text Preprocessing
- `CountVectorizer` — bag-of-words
  - `max_features`, `stop_words`, `ngram_range`
- `TfidfVectorizer` — TF-IDF features
  - `max_df`, `min_df`, `max_features`
- `HashingVectorizer` — memory-efficient vectorizer

---

## 4. Feature Selection

### 4.1 Filter Methods
- `SelectKBest` — select top K features
  - `score_func` — `chi2`, `f_classif`, `f_regression`, `mutual_info_classif`
  - `k` — number of features
- `SelectPercentile` — select top percentile
- `SelectFpr`, `SelectFdr`, `SelectFwe` — statistical threshold selection

### 4.2 Wrapper Methods
- `RFE` — Recursive Feature Elimination
  - `estimator`, `n_features_to_select`, `step`
  - `ranking_` — feature ranking
  - `support_` — feature mask
- `RFECV` — RFE with cross-validation
  - `cv`, `scoring`
  - Optimal number of features

### 4.3 Embedded Methods
- `SelectFromModel` — model-based selection
  - `estimator` — any model with `feature_importances_` or `coef_`
  - `threshold` — importance threshold
  - `max_features`

### 4.4 Feature Importance
- `feature_importances_` — tree-based models
- `coef_` — linear models
- Permutation importance — `permutation_importance()`

---

## 5. Model Selection Utilities

### 5.1 Train-Test Split
- `train_test_split(X, y)`
  - `test_size` — fraction or absolute count
  - `train_size`
  - `random_state`
  - `stratify` — stratified split for classification
- Splitting multiple arrays simultaneously

### 5.2 Cross-Validation
- `cross_val_score(estimator, X, y, cv=5, scoring='accuracy')` — basic CV
- `cross_validate(estimator, X, y, cv=5, return_train_score=True)` — detailed results
- `cross_val_predict(estimator, X, y, cv=5)` — OOF predictions

### 5.3 Cross-Validation Strategies
- `KFold(n_splits=5, shuffle=True)` — K-fold
- `StratifiedKFold` — stratified for classification
- `RepeatedKFold` — repeated K-fold
- `RepeatedStratifiedKFold`
- `LeaveOneOut` — LOO cross-validation
- `LeavePOut` — leave-P-out
- `ShuffleSplit` — random permutation CV
- `StratifiedShuffleSplit`
- `TimeSeriesSplit` — time series validation
- `GroupKFold`, `GroupShuffleSplit` — group-aware splits

### 5.4 Hyperparameter Tuning
- `GridSearchCV` — exhaustive search
  - `param_grid` — parameter grid dictionary
  - `cv`, `scoring`, `n_jobs`, `refit`
  - `best_params_`, `best_score_`, `best_estimator_`
  - `cv_results_`
- `RandomizedSearchCV` — random search
  - `param_distributions` — distributions for random sampling
  - `n_iter` — number of parameter settings
- `HalvingGridSearchCV` — successive halving
- `HalvingRandomSearchCV` — random successive halving

### 5.5 Scoring Metrics
- `scoring` parameter options
  - `'accuracy'`, `'f1'`, `'f1_macro'`, `'f1_weighted'`
  - `'precision'`, `'recall'`
  - `'roc_auc'`, `'average_precision'`
  - `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'`
- `make_scorer()` — custom scoring function

---

## 6. Supervised Learning — Classification

### 6.1 Linear Models
- `LogisticRegression`
  - `C` — regularization strength (inverse)
  - `penalty` — `'l1'`, `'l2'`, `'elasticnet'`, `'none'`
  - `solver` — `'lbfgs'`, `'saga'`, `'liblinear'`
  - `multi_class` — `'ovr'`, `'multinomial'`
  - `max_iter`
- `SGDClassifier` — stochastic gradient descent
  - `loss` — `'hinge'`, `'log_loss'`, `'modified_huber'`
  - `alpha`, `l1_ratio`
  - `learning_rate`
- `RidgeClassifier` — Ridge for classification
- `Perceptron` — simple linear classifier
- `PassiveAggressiveClassifier`

### 6.2 Support Vector Machines
- `SVC` — Support Vector Classifier
  - `C` — regularization
  - `kernel` — `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`
  - `gamma` — `'scale'`, `'auto'`, float
  - `degree` — for polynomial kernel
  - `probability` — enable `predict_proba()`
  - `class_weight` — imbalanced classes
- `LinearSVC` — linear SVM (faster)
- `NuSVC` — Nu-SVM variant

### 6.3 Tree-based Models
- `DecisionTreeClassifier`
  - `criterion` — `'gini'`, `'entropy'`, `'log_loss'`
  - `max_depth`, `min_samples_split`, `min_samples_leaf`
  - `max_features`
  - `feature_importances_`
- `RandomForestClassifier`
  - `n_estimators` — number of trees
  - `max_features`, `max_depth`, `min_samples_split`
  - `bootstrap`, `oob_score`
  - `feature_importances_`
- `ExtraTreesClassifier` — extremely randomized trees
- `GradientBoostingClassifier`
  - `n_estimators`, `learning_rate`, `max_depth`, `subsample`
  - `loss` — `'log_loss'`, `'exponential'`
- `HistGradientBoostingClassifier` — histogram-based (faster)

### 6.4 Nearest Neighbors
- `KNeighborsClassifier`
  - `n_neighbors`
  - `weights` — `'uniform'`, `'distance'`
  - `metric` — `'euclidean'`, `'manhattan'`, `'minkowski'`
  - `algorithm` — `'ball_tree'`, `'kd_tree'`, `'brute'`, `'auto'`
- `RadiusNeighborsClassifier`

### 6.5 Naive Bayes
- `GaussianNB` — continuous features
  - `var_smoothing`
- `MultinomialNB` — count features
  - `alpha` — Laplace smoothing
- `BernoulliNB` — binary features
- `ComplementNB` — for imbalanced text

### 6.6 Neural Networks
- `MLPClassifier` — multi-layer perceptron
  - `hidden_layer_sizes` — tuple of layer sizes
  - `activation` — `'relu'`, `'tanh'`, `'logistic'`, `'identity'`
  - `solver` — `'adam'`, `'sgd'`, `'lbfgs'`
  - `alpha` — L2 regularization
  - `learning_rate_init`, `max_iter`, `early_stopping`

### 6.7 Ensemble Methods
- `BaggingClassifier` — bagging
  - `base_estimator`, `n_estimators`, `max_samples`, `max_features`
- `AdaBoostClassifier` — boosting
  - `base_estimator`, `n_estimators`, `learning_rate`
  - `algorithm` — `'SAMME'`, `'SAMME.R'`
- `VotingClassifier` — voting ensemble
  - `estimators`, `voting` — `'hard'`, `'soft'`
  - `weights`
- `StackingClassifier` — stacking ensemble
  - `estimators`, `final_estimator`, `cv`

---

## 7. Supervised Learning — Regression

### 7.1 Linear Models
- `LinearRegression`
  - `fit_intercept`, `normalize` (deprecated)
  - `coef_`, `intercept_`
- `Ridge` — L2 regularization
  - `alpha` — regularization strength
  - `solver`
- `Lasso` — L1 regularization
  - `alpha`, `max_iter`
  - Sparse coefficients
- `ElasticNet` — L1 + L2
  - `alpha`, `l1_ratio`
- `BayesianRidge` — Bayesian linear regression
- `SGDRegressor` — stochastic gradient descent
- `HuberRegressor` — robust to outliers

### 7.2 Support Vector Regression
- `SVR`
  - `C`, `epsilon`, `kernel`, `gamma`
- `LinearSVR` — linear SVR (faster)
- `NuSVR`

### 7.3 Tree-based Regression
- `DecisionTreeRegressor`
  - `criterion` — `'squared_error'`, `'friedman_mse'`, `'absolute_error'`, `'poisson'`
  - `max_depth`, `min_samples_split`
- `RandomForestRegressor`
  - `n_estimators`, `max_features`, `oob_score`
- `GradientBoostingRegressor`
  - `loss` — `'squared_error'`, `'absolute_error'`, `'huber'`, `'quantile'`
- `HistGradientBoostingRegressor` — histogram-based

### 7.4 Nearest Neighbors Regression
- `KNeighborsRegressor`
  - `n_neighbors`, `weights`, `metric`

### 7.5 Neural Network Regression
- `MLPRegressor`
  - `hidden_layer_sizes`, `activation`, `solver`, `alpha`

### 7.6 Other Regression Models
- `GaussianProcessRegressor` — Bayesian regression with uncertainty
- `IsotonicRegression` — monotone regression
- `RANSACRegressor` — robust regression
- `TheilSenRegressor` — robust to outliers
- `QuantileRegressor` — quantile regression
- `TweedieRegressor` — GLM regression

---

## 8. Unsupervised Learning — Clustering

### 8.1 K-Means Clustering
- `KMeans`
  - `n_clusters`, `init` — `'k-means++'`, `'random'`
  - `n_init`, `max_iter`, `tol`
  - `labels_`, `cluster_centers_`, `inertia_`
- `MiniBatchKMeans` — faster for large data
- Elbow method — inertia vs k

### 8.2 Hierarchical Clustering
- `AgglomerativeClustering`
  - `n_clusters`, `linkage` — `'ward'`, `'complete'`, `'average'`, `'single'`
  - `metric` — distance metric
  - `compute_full_tree`
- Dendrogram — `scipy.cluster.hierarchy.dendrogram`

### 8.3 Density-Based Clustering
- `DBSCAN`
  - `eps` — neighborhood radius
  - `min_samples` — minimum points for core
  - `metric`
  - `labels_` — `-1` for noise
- `HDBSCAN` — hierarchical DBSCAN (sklearn 1.3+)
- `OPTICS` — ordering-based

### 8.4 Other Clustering
- `SpectralClustering`
  - `n_clusters`, `affinity`, `assign_labels`
- `MeanShift`
  - `bandwidth`, `bin_seeding`
  - `cluster_centers_`
- `GaussianMixture` — probabilistic clustering
  - `n_components`, `covariance_type`
  - `predict()`, `predict_proba()`, `score()`
- `BirchClustering`

### 8.5 Clustering Evaluation
- `silhouette_score()` — between-cluster vs within-cluster
- `davies_bouldin_score()` — lower is better
- `calinski_harabasz_score()` — higher is better
- `adjusted_rand_score()` — for labeled ground truth
- `normalized_mutual_info_score()`
- `homogeneity_score()`, `completeness_score()`, `v_measure_score()`

---

## 9. Unsupervised Learning — Dimensionality Reduction

### 9.1 PCA
- `PCA`
  - `n_components` — number of components or variance threshold
  - `svd_solver` — `'full'`, `'arpack'`, `'randomized'`, `'auto'`
  - `whiten`
  - `explained_variance_ratio_`
  - `components_`
  - `transform()`, `inverse_transform()`
- `IncrementalPCA` — for large datasets
- `KernelPCA` — non-linear PCA
  - `kernel` — `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'cosine'`

### 9.2 Linear Discriminant Analysis
- `LinearDiscriminantAnalysis`
  - `n_components`
  - `solver` — `'svd'`, `'lsqr'`, `'eigen'`
  - Supervised dimensionality reduction

### 9.3 Manifold Learning
- `TSNE`
  - `n_components` (2 or 3)
  - `perplexity`, `learning_rate`, `n_iter`
  - `metric`
- `MDS` — Multi-Dimensional Scaling
  - `n_components`, `metric`
- `Isomap`
  - `n_neighbors`, `n_components`
- `LocallyLinearEmbedding`
  - `n_neighbors`, `n_components`
- `SpectralEmbedding`
- `UMAP` — via `umap-learn` package (not in sklearn)

### 9.4 Matrix Factorization
- `TruncatedSVD` — LSA for sparse data
  - `n_components`
- `NMF` — Non-negative Matrix Factorization
  - `n_components`, `init`, `solver`
- `LatentDirichletAllocation` — topic modeling
  - `n_components`, `max_iter`

---

## 10. Model Evaluation Metrics

### 10.1 Classification Metrics
- `accuracy_score(y_true, y_pred)`
- `precision_score(y_true, y_pred, average='binary')`
  - `average` — `'micro'`, `'macro'`, `'weighted'`, `'binary'`
- `recall_score(y_true, y_pred, average='binary')`
- `f1_score(y_true, y_pred, average='binary')`
- `classification_report(y_true, y_pred)` — full report
- `confusion_matrix(y_true, y_pred)`
- `ConfusionMatrixDisplay` — visual confusion matrix
- `roc_auc_score(y_true, y_score)` — AUC-ROC
- `average_precision_score(y_true, y_score)` — AUC-PR
- `log_loss(y_true, y_pred_proba)` — cross-entropy loss
- `matthews_corrcoef()` — MCC
- `cohen_kappa_score()` — Cohen's Kappa
- `hamming_loss()` — for multi-label
- `jaccard_score()` — Jaccard index

### 10.2 ROC and PR Curves
- `roc_curve(y_true, y_score)` — returns `fpr`, `tpr`, `thresholds`
- `RocCurveDisplay.from_estimator()` — direct plot
- `precision_recall_curve(y_true, y_score)` — returns `precision`, `recall`, `thresholds`
- `PrecisionRecallDisplay.from_estimator()` — direct plot

### 10.3 Regression Metrics
- `mean_squared_error(y_true, y_pred)`
  - `squared=False` for RMSE
- `mean_absolute_error(y_true, y_pred)`
- `r2_score(y_true, y_pred)`
- `mean_absolute_percentage_error(y_true, y_pred)`
- `median_absolute_error(y_true, y_pred)`
- `explained_variance_score(y_true, y_pred)`
- `max_error(y_true, y_pred)`
- `mean_tweedie_deviance()`
- `mean_poisson_deviance()`

### 10.4 Calibration
- `calibration_curve(y_true, y_prob)` — calibration data
- `CalibrationDisplay` — calibration plot
- `brier_score_loss()` — calibration loss

---

## 11. Pipelines

### 11.1 Pipeline Basics
- `Pipeline(steps=[('name', estimator)])` — sequential pipeline
- `make_pipeline(*estimators)` — shorthand
- `pipe.fit(X_train, y_train)` — fits all steps
- `pipe.predict(X_test)` — transforms and predicts
- `pipe.score(X_test, y_test)`
- `pipe.named_steps['name']` — accessing steps
- `pipe.set_params(step__param=value)` — setting params

### 11.2 ColumnTransformer
- `ColumnTransformer(transformers=[('name', transformer, columns)])` — per-column transforms
- `make_column_transformer()` — shorthand
- `remainder` — `'passthrough'`, `'drop'`, or transformer
- `make_column_selector(dtype_include=np.number)` — automatic column selection
- Numeric and categorical separate preprocessing

### 11.3 Pipeline with GridSearchCV
- Nested parameter naming — `step__param`
- `param_grid = {'step__param': [values]}`
- Cross-validated pipeline tuning

### 11.4 FunctionTransformer
- `FunctionTransformer(func)` — wrapping custom functions
- `validate` — input validation
- `inverse_func` — for inverse transform

### 11.5 TransformedTargetRegressor
- `TransformedTargetRegressor(regressor, transformer)` — target transformation
- Log-transform target for regression
- `func` and `inverse_func` parameters

---

## 12. Ensemble Methods

### 12.1 Bagging
- `BaggingClassifier` and `BaggingRegressor`
  - `base_estimator`, `n_estimators`
  - `max_samples`, `max_features`
  - `bootstrap`, `bootstrap_features`
  - `oob_score`

### 12.2 Random Forests
- `RandomForestClassifier` and `RandomForestRegressor`
  - `n_estimators`, `max_features`, `max_depth`
  - `min_samples_split`, `min_samples_leaf`
  - `oob_score`, `warm_start`
  - `feature_importances_`

### 12.3 Boosting
- `AdaBoostClassifier` and `AdaBoostRegressor`
- `GradientBoostingClassifier` and `GradientBoostingRegressor`
- `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor`

### 12.4 Voting
- `VotingClassifier` and `VotingRegressor`
  - `estimators` — list of `(name, estimator)` tuples
  - `voting` — `'hard'`, `'soft'`
  - `weights`

### 12.5 Stacking
- `StackingClassifier` and `StackingRegressor`
  - `estimators` — base learners
  - `final_estimator` — meta-learner
  - `cv` — cross-validation for generating meta-features
  - `passthrough` — pass original features to meta-learner

---

## 13. Multiclass and Multi-output

### 13.1 Multiclass Strategies
- `OneVsRestClassifier` — one-vs-rest
  - `estimator`
- `OneVsOneClassifier` — one-vs-one
- `OutputCodeClassifier` — error-correcting codes

### 13.2 Multi-label Classification
- `MultiOutputClassifier` — wrapping binary classifiers
- `ClassifierChain` — label dependencies

### 13.3 Multi-output Regression
- `MultiOutputRegressor`
- `RegressorChain`

---

## 14. Calibration

### 14.1 Probability Calibration
- `CalibratedClassifierCV`
  - `method` — `'sigmoid'`, `'isotonic'`
  - `cv` — cross-validation or `'prefit'`
- When to calibrate — SVM, tree-based models
- `predict_proba()` after calibration

---

## 15. Neural Networks (sklearn)

### 15.1 MLPClassifier
- `hidden_layer_sizes` — e.g., `(100,)`, `(100, 50)`
- `activation` — `'relu'`, `'tanh'`, `'logistic'`
- `solver` — `'adam'`, `'sgd'`, `'lbfgs'`
- `alpha` — L2 penalty
- `batch_size`, `learning_rate`, `learning_rate_init`
- `max_iter`, `tol`, `early_stopping`, `validation_fraction`
- `loss_curve_`, `n_iter_`

### 15.2 MLPRegressor
- Same parameters as MLPClassifier
- Output — single continuous value

---

## 16. Semi-supervised Learning

### 16.1 Label Propagation
- `LabelPropagation`
  - `kernel` — `'rbf'`, `'knn'`
  - `gamma`, `n_neighbors`
- `LabelSpreading`

### 16.2 Self-training
- `SelfTrainingClassifier`
  - `base_estimator` — any classifier with `predict_proba()`
  - `threshold` — confidence threshold for pseudo-labels
  - `max_iter`

---

## 17. Anomaly Detection

### 17.1 Outlier Detection
- `IsolationForest`
  - `n_estimators`, `contamination`, `max_features`
  - `predict()` — `1` for inliers, `-1` for outliers
- `LocalOutlierFactor`
  - `n_neighbors`, `contamination`
  - `novelty` — for new data scoring
- `OneClassSVM`
  - `kernel`, `nu`, `gamma`
- `EllipticEnvelope`
  - `contamination`, `support_fraction`

---

## 18. Model Inspection and Interpretation

### 18.1 Feature Importances
- `feature_importances_` — tree-based models
- `coef_` — linear models
- `permutation_importance(estimator, X, y)` — model-agnostic
  - `n_repeats`, `random_state`

### 18.2 Partial Dependence
- `PartialDependenceDisplay.from_estimator()` — PDP
  - `features` — list of feature indices
  - `kind` — `'average'`, `'individual'`, `'both'`
- Individual Conditional Expectation (ICE) plots

### 18.3 Inspection Utilities
- `inspection.permutation_importance()`
- `inspection.partial_dependence()`

---

## 19. Preprocessing Utilities

### 19.1 Target Encoding
- `TargetEncoder` (sklearn 1.3+)
  - `target_type`, `smooth`

### 19.2 Power Transforms
- `PowerTransformer`
  - `method` — `'yeo-johnson'`, `'box-cox'`
  - `standardize`
- `QuantileTransformer`
  - `n_quantiles`, `output_distribution` — `'normal'`, `'uniform'`

### 19.3 Spline Transformers
- `SplineTransformer`
  - `n_knots`, `degree`, `extrapolation`

---

## 20. Scikit-learn Best Practices

### 20.1 Writing Efficient ML Code
- Always use Pipeline to prevent data leakage
- Fit scaler only on training data
- Use `cross_validate` instead of manual loops
- Use `set_random_state` for reproducibility
- Use `n_jobs=-1` for parallel computation

### 20.2 Common Mistakes
- Fitting preprocessors on the full dataset before split — data leakage
- Not stratifying for imbalanced classification
- Using `accuracy` for imbalanced datasets — use `f1` or `roc_auc`
- Tuning hyperparameters on the test set
- Not using `Pipeline` — manual preprocessing errors

### 20.3 Memory and Performance
- `n_jobs=-1` for parallelization
- `warm_start=True` for incremental training
- `IncrementalPCA` and `MiniBatchKMeans` for large datasets
- `HistGradientBoosting` instead of `GradientBoosting` for speed

---

## 21. Scikit-learn Interview Topics

### 21.1 Frequently Asked Questions
- What is the difference between `fit()`, `transform()`, and `fit_transform()`?
- What is a Pipeline and why use it?
- How does cross-validation prevent data leakage?
- What is the difference between GridSearchCV and RandomizedSearchCV?
- How do you handle class imbalance in sklearn?
- What is `feature_importances_` vs `permutation_importance`?

### 21.2 Coding Problems
- Build a full preprocessing + classification pipeline
- Tune hyperparameters with GridSearchCV inside a Pipeline
- Compare multiple classifiers using cross-validation
- Implement stratified K-fold for imbalanced dataset
- Build a stacking ensemble with sklearn

---

## 22. Real-World Scikit-learn Projects

### 22.1 Beginner Projects
- Iris flower classification
- House price regression
- Customer churn binary classification

### 22.2 Intermediate Projects
- End-to-end ML pipeline with feature engineering and tuning
- Multi-class text classification with TF-IDF + Logistic Regression
- Anomaly detection in sensor data

### 22.3 Advanced Projects
- Stacking ensemble for Kaggle competition
- Automated feature selection and model comparison pipeline
- Customer segmentation with clustering + dimensionality reduction

---

## 23. Scikit-learn Ecosystem

### 23.1 Related Libraries
- NumPy — array computation
- Pandas — data manipulation
- Matplotlib / Seaborn — visualization
- SciPy — statistical functions
- Joblib — parallelization backend

### 23.2 Scikit-learn Extensions
- `imbalanced-learn` — SMOTE, oversampling, undersampling
- `scikit-optimize` — Bayesian hyperparameter optimization
- `yellowbrick` — ML visualizer for sklearn
- `eli5` — model explanation
- `shap` — SHAP values for model interpretation
- `mlxtend` — extended ML tools
