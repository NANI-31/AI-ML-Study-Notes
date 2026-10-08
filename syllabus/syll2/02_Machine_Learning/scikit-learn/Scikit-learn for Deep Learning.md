# Scikit-learn for Deep Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to Scikit-learn in Deep Learning Workflows

### 1.1 Where Scikit-learn Fits in a DL Pipeline
- Scikit-learn handles what deep learning frameworks do not
  - Structured data preprocessing before feeding to neural networks
  - Classical baseline models for DL comparison
  - Feature engineering before embedding layers
  - Model evaluation utilities — confusion matrix, ROC, PR curves
  - Hyperparameter search for DL wrapper models
  - Dimensionality reduction before visualization or DL input
- Scikit-learn + TensorFlow / PyTorch — complementary stack

### 1.2 Role of Scikit-learn in DL Pipelines
- Data preprocessing — scaling, encoding, imputation
- Feature selection before DL input
- Classical baselines — benchmarking DL gains
- Train/test/val splitting and cross-validation setup
- Model evaluation and comparison
- Dimensionality reduction for DL embedding analysis

### 1.3 Installing and Setup
- Installing Scikit-learn
- Import conventions
  - `from sklearn.preprocessing import StandardScaler, OneHotEncoder`
  - `from sklearn.model_selection import train_test_split, StratifiedKFold`
  - `from sklearn.pipeline import Pipeline`
  - `from sklearn.metrics import classification_report, roc_auc_score`
  - `import tensorflow as tf`
  - `import torch`

---

## 2. Data Preprocessing for Deep Learning Inputs

### 2.1 Feature Scaling for Neural Networks
- `StandardScaler` — zero mean, unit variance
  - Required for MLP, RNN, Transformer inputs
  - Accelerates gradient descent convergence
- `MinMaxScaler(feature_range=(0, 1))` — scale to [0, 1]
  - Preferred for sigmoid output layers and image pixel normalization
- `RobustScaler` — median and IQR scaling
  - Use when outliers are present in tabular DL datasets
- Fit only on training data — `scaler.fit(X_train)`, `scaler.transform(X_test)`
- Converting scaled output — `astype('float32')` for TensorFlow/PyTorch

### 2.2 Encoding for Embedding Layers
- `OrdinalEncoder` — integer codes for embedding layer input
  - `categories='auto'` — automatic category detection
  - Output: integer array used as embedding indices
- `LabelEncoder` — encoding class labels for `y`
- `OneHotEncoder(sparse_output=False)` — for softmax output targets
- `pd.factorize()` alternative for fast integer encoding

### 2.3 Handling Missing Values for DL Datasets
- `SimpleImputer(strategy='mean')` — numerical NaN filling
- `SimpleImputer(strategy='most_frequent')` — categorical NaN filling
- `KNNImputer(n_neighbors=5)` — structured tabular data
- `add_indicator=True` — create missingness flags as additional features
- Ensuring zero NaN before tensor conversion

### 2.4 ColumnTransformer for Tabular DL Preprocessing
- Numeric pipeline — `SimpleImputer → StandardScaler`
- Categorical pipeline — `SimpleImputer → OrdinalEncoder` (for embeddings)
- Categorical pipeline — `SimpleImputer → OneHotEncoder` (for dense input)
- `remainder='passthrough'` — pass binary or precomputed features

### 2.5 Converting to DL-ready Format
- `X_train_processed.astype('float32')` — float32 for GPU efficiency
- `tf.constant(X_train_processed)` — TensorFlow tensor
- `torch.tensor(X_train_processed, dtype=torch.float32)` — PyTorch tensor
- `tf.data.Dataset.from_tensor_slices((X, y))` — TF Dataset

---

## 3. Classical Baselines for Deep Learning Comparison

### 3.1 Why Baselines Matter in DL
- Quantifying DL uplift over classical methods
- Identifying if problem requires DL at all
- Sanity-checking DL performance
- Establishing benchmark metrics for tabular data

### 3.2 Baseline Classifiers
- `LogisticRegression()` — linear baseline
- `RandomForestClassifier(n_estimators=100)` — tree baseline
- `GradientBoostingClassifier()` or `HistGradientBoostingClassifier()` — strong baseline
- `GaussianNB()` — probabilistic baseline
- Comparing F1, AUC-ROC, accuracy with DL model

### 3.3 Baseline Regressors
- `LinearRegression()` — linear baseline
- `Ridge(alpha=1.0)` — regularized linear baseline
- `RandomForestRegressor(n_estimators=100)` — tree baseline
- Comparing RMSE, MAE, R² with DL model

### 3.4 Baseline Clustering
- `KMeans(n_clusters=k)` — cluster quality benchmark
- Comparing DL embedding clusters vs classical clusters
- Using silhouette score for both

---

## 4. Feature Selection for Deep Learning

### 4.1 Removing Irrelevant Features
- `VarianceThreshold(threshold=0.01)` — near-constant features
- `SelectKBest(mutual_info_classif, k=50)` — mutual information top-k
- `SelectFromModel(RandomForestClassifier())` — tree-based importance selection
- Reduces input dimensionality → smaller DL models → faster training

### 4.2 Correlation-based Reduction
- Computing correlation matrix — `df.corr()`
- Removing features with `|corr| > 0.95`
- Reduces redundancy in DL input

### 4.3 RFE for DL Feature Selection
- `RFE(estimator=LogisticRegression(), n_features_to_select=20)`
- `RFECV(estimator=RandomForestClassifier(), cv=5, scoring='roc_auc')`
- Select optimal feature count before DL training

### 4.4 Feature Importance from Sklearn for DL Input Design
- `RandomForestClassifier().feature_importances_` — rank features
- Use top-N features as DL input
- `permutation_importance()` — validation-set based

---

## 5. Train-Validation-Test Splitting for DL

### 5.1 Standard Three-way Split
- `train_test_split(X, y, test_size=0.2, stratify=y)` — first split for test
- `train_test_split(X_train, y_train, test_size=0.125)` — second split for val (val = 10% total)
- Test set — final evaluation only, never used during development

### 5.2 Stratified Split for DL Classification
- `StratifiedShuffleSplit(n_splits=1, test_size=0.2)` — stratified split object
- Maintaining class distribution — critical for imbalanced DL tasks
- Image, text, tabular classification

### 5.3 Time Series Split for Sequential DL
- `TimeSeriesSplit(n_splits=5)` — forward-chaining for LSTM/Transformer
- No data leakage from future timesteps
- `gap` parameter — gap between train and val splits

### 5.4 Cross-Validation Setup for DL
- `StratifiedKFold(n_splits=5, shuffle=True)` — fold definition
- Using sklearn folds to define DL training and validation sets
- `fold.split(X, y)` — train_idx, val_idx per fold

---

## 6. Sklearn Wrappers for Keras and PyTorch

### 6.1 SciKeras — Keras sklearn API
- `from scikeras.wrappers import KerasClassifier, KerasRegressor`
- `KerasClassifier(model=build_fn, epochs=50, batch_size=32)`
- `model.fit(X_train, y_train)` — sklearn-style
- `model.predict(X_test)` — predictions
- `model.score(X_test, y_test)` — accuracy

### 6.2 GridSearchCV with Keras via SciKeras
- `param_grid = {'epochs': [50, 100], 'batch_size': [32, 64], 'model__dropout': [0.2, 0.5]}`
- `GridSearchCV(KerasClassifier(model=build_fn), param_grid, cv=3)`
- Hyperparameter tuning neural networks with sklearn

### 6.3 Skorch — PyTorch sklearn API
- `from skorch import NeuralNetClassifier, NeuralNetRegressor`
- `NeuralNetClassifier(module=MyPyTorchModel, max_epochs=50, lr=0.01)`
- `model.fit(X_train, y_train)` — sklearn-style
- `GridSearchCV(NeuralNetClassifier(...), param_grid, cv=5)`

---

## 7. Model Evaluation for Deep Learning

### 7.1 Classification Evaluation
- `accuracy_score(y_true, y_pred)` — overall accuracy
- `f1_score(y_true, y_pred, average='weighted')` — F1
- `classification_report(y_true, y_pred, target_names=class_names)` — full report
- `confusion_matrix(y_true, y_pred)` — confusion matrix
- `ConfusionMatrixDisplay(cm, display_labels=class_names).plot()` — visual
- `roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')` — AUC

### 7.2 ROC and PR Curves for DL Models
- `roc_curve(y_true, y_score)` — fpr, tpr, thresholds
- `RocCurveDisplay.from_predictions(y_true, y_score)` — direct plot
- `precision_recall_curve(y_true, y_score)` — precision, recall
- `PrecisionRecallDisplay.from_predictions(y_true, y_score)` — direct plot
- Multi-class OvR curves — loop over classes

### 7.3 Regression Evaluation
- `mean_squared_error(y_true, y_pred, squared=False)` — RMSE
- `mean_absolute_error(y_true, y_pred)` — MAE
- `r2_score(y_true, y_pred)` — R²

### 7.4 Calibration for DL Classifiers
- `calibration_curve(y_true, y_pred_proba, n_bins=10)` — reliability diagram
- `CalibrationDisplay.from_predictions(y_true, y_prob).plot()`
- `brier_score_loss(y_true, y_pred_proba)` — calibration loss
- Comparing DL model calibration vs Platt scaling

---

## 8. Dimensionality Reduction for DL Analysis

### 8.1 PCA for DL Feature Analysis
- `PCA(n_components=2)` — 2D projection of DL learned features
- Plotting last hidden layer activations in 2D
- `pca.explained_variance_ratio_` — variance captured
- Comparing raw features vs learned DL representations

### 8.2 t-SNE for DL Embedding Visualization
- `TSNE(n_components=2, perplexity=30, random_state=42)`
- Applying to:
  - Pre-trained CNN feature vectors
  - Transformer [CLS] token embeddings
  - Word2Vec / GloVe / BERT embeddings
  - Autoencoder bottleneck representations
- Color by class label — checking cluster quality

### 8.3 UMAP for DL Embeddings
- `umap.UMAP(n_components=2, n_neighbors=15)` — via `umap-learn`
- Faster than t-SNE for large embedding sets
- Better global structure preservation

### 8.4 Scree Plot with sklearn PCA
- `pca.explained_variance_ratio_` — per component
- Cumulative variance plot
- Choosing optimal DL bottleneck dimension

---

## 9. Clustering for DL

### 9.1 Clustering DL Representations
- Extract DL embeddings (CNN features, BERT embeddings, autoencoder bottleneck)
- `KMeans(n_clusters=k).fit(embeddings)` — cluster embeddings
- `labels_` — cluster assignment per sample
- `silhouette_score(embeddings, labels)` — cluster quality

### 9.2 Evaluating DL Representation Quality
- `adjusted_rand_score(true_labels, cluster_labels)` — cluster vs ground truth
- `normalized_mutual_info_score(true_labels, cluster_labels)`
- Comparing raw feature clusters vs DL embedding clusters

### 9.3 Hierarchical Clustering of Embeddings
- `AgglomerativeClustering(n_clusters=k, linkage='ward').fit(embeddings)`
- Dendrogram for embedding similarity structure

---

## 10. Data Augmentation Preprocessing with sklearn

### 10.1 Tabular Data Augmentation
- `SMOTE()` (imbalanced-learn) — synthetic minority samples
- `PolynomialFeatures` — creating interaction features
- `QuantileTransformer` — normalizing extreme distributions

### 10.2 Feature Engineering for DL Tabular Models
- `SplineTransformer` — non-linear feature encoding
- `KBinsDiscretizer` — binning continuous features for embedding layers
- `TargetEncoder` — supervised categorical encoding

---

## 11. Anomaly Detection Baselines for DL

### 11.1 Classical Anomaly Detectors
- `IsolationForest(contamination=0.05)` — tree-based anomaly
- `LocalOutlierFactor(n_neighbors=20)` — density-based anomaly
- `OneClassSVM(nu=0.05, kernel='rbf')` — one-class classification
- `EllipticEnvelope(contamination=0.05)` — Gaussian-based

### 11.2 Comparing with DL Autoencoders
- Classical `IsolationForest` baseline AUC-ROC
- DL Autoencoder reconstruction error AUC-ROC
- Quantifying DL anomaly detection gain

---

## 12. Hyperparameter Search for DL Models

### 12.1 GridSearchCV with SciKeras
- `param_grid = {'epochs': [50, 100], 'batch_size': [32, 64], 'model__units': [64, 128]}`
- `GridSearchCV(KerasClassifier(model=build_fn), param_grid, cv=3, scoring='f1_weighted')`
- `best_params_` — optimal DL hyperparameters

### 12.2 RandomizedSearchCV for DL
- `param_distributions = {'lr': loguniform(1e-4, 1e-2), 'dropout': uniform(0.1, 0.5)}`
- `RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3)`
- Efficient for large DL hyperparameter spaces

### 12.3 Manual Cross-Validation for DL
- `skf = StratifiedKFold(n_splits=5, shuffle=True)`
- `for train_idx, val_idx in skf.split(X, y):`
  - `X_fold_train, X_fold_val = X[train_idx], X[val_idx]`
  - Build and train DL model on fold
  - Record validation metrics per fold
- `cross_val_scores` — mean and std

---

## 13. Pipeline for DL Preprocessing

### 13.1 Preprocessing Pipeline
- `preprocessing_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])`
- `X_train_proc = preprocessing_pipe.fit_transform(X_train)`
- `X_test_proc = preprocessing_pipe.transform(X_test)`
- Preventing leakage before DL model training

### 13.2 ColumnTransformer for Tabular DL
- Numeric → `StandardScaler` → DL dense layer
- Categorical → `OrdinalEncoder` → DL embedding layer
- `ct = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OrdinalEncoder(), cat_cols)])`

### 13.3 sklearn Pipeline + SciKeras End-to-End
- `pipe = Pipeline([('preprocess', ColumnTransformer(...)), ('model', KerasClassifier(...))])`
- Full DL pipeline — preprocessing + model
- `pipe.fit(X_raw, y)` — handles everything

---

## 14. Transfer Learning Support with sklearn

### 14.1 Using Sklearn for Feature Extraction
- Extract DL features (e.g., ResNet last layer) → NumPy array
- `LogisticRegression().fit(features_train, y_train)` — fast classifier on top
- `cross_val_score(LogisticRegression(), features, y, cv=5)` — CV evaluation

### 14.2 Fine-tuning Strategy Support
- Baseline — sklearn classifier on frozen DL features
- Fine-tuned — DL model trained end-to-end
- Comparing both with sklearn metrics

---

## 15. Model Interpretation for DL with sklearn

### 15.1 Permutation Importance for DL Models
- `permutation_importance(keras_sklearn_model, X_val, y_val, n_repeats=10)`
- Requires sklearn-compatible wrapper (SciKeras)
- Identify which input features DL model relies on most

### 15.2 SHAP Integration
- `shap.DeepExplainer(keras_model, X_background)` — DL SHAP
- `shap.KernelExplainer(model.predict_proba, X_background)` — model-agnostic SHAP
- Using sklearn-compatible wrapper for SHAP

### 15.3 Partial Dependence for DL
- `PartialDependenceDisplay.from_estimator(sklearn_wrapped_dl_model, X, features=[0, 1])`
- Requires SciKeras or Skorch wrapper

---

## 16. Evaluation Metrics Deep Dive for DL

### 16.1 Multi-class Classification Metrics
- `f1_score(y_true, y_pred, average='macro')` — class-balanced F1
- `f1_score(y_true, y_pred, average='weighted')` — frequency-weighted F1
- `top_k_accuracy_score(y_true, y_score, k=5)` — top-5 accuracy for ImageNet-style
- `roc_auc_score(y_true, y_score, multi_class='ovr')` — OvR AUC

### 16.2 Segmentation Metrics (via custom sklearn scoring)
- `make_scorer(jaccard_score, average='macro')` — IoU metric
- Pixel-level accuracy and IoU via custom scorers

### 16.3 NLP Metrics
- `f1_score(y_true, y_pred, average='weighted')` — token classification F1
- `accuracy_score` — sequence accuracy
- Sequence-level classification report

### 16.4 Regression Metrics for DL
- `mean_squared_error(y_true, y_pred, squared=False)` — RMSE for forecasting
- `mean_absolute_percentage_error` — relative error for time series

---

## 17. Semi-supervised and Self-supervised Support

### 17.1 Label Propagation for DL Pretraining
- `LabelPropagation()` — propagating labels from labeled to unlabeled
- Using propagated labels to pre-train DL model
- `LabelSpreading()` — softer variant

### 17.2 Self-training for DL
- `SelfTrainingClassifier(base_estimator=GradientBoostingClassifier(), threshold=0.9)`
- Classical self-training as baseline for DL pseudo-labeling

---

## 18. Model Persistence for DL Pipelines

### 18.1 Saving sklearn Components
- `joblib.dump(preprocessing_pipe, 'preprocessor.pkl')` — saving preprocessor
- `preprocessor = joblib.load('preprocessor.pkl')` — loading
- Save sklearn preprocessor separately from DL model weights

### 18.2 Inference Pipeline
- Load sklearn preprocessor
- Load DL model weights
- `X_proc = preprocessor.transform(X_raw)`
- `y_pred = dl_model.predict(X_proc)`

---

## 19. Scikit-learn Best Practices in DL Pipelines

### 19.1 Preventing Data Leakage
- Always fit sklearn preprocessors only on training data
- Use `Pipeline` to automate leakage prevention
- Never apply `fit_transform` on full dataset before splitting

### 19.2 Float32 Discipline
- Convert all sklearn outputs to `float32` before DL
- `X_processed.astype('float32')` — always
- `StandardScaler` output is `float64` by default — explicit cast needed

### 19.3 Reproducibility
- Set `random_state` in all sklearn components
- `np.random.seed(42)` and `tf.random.set_seed(42)` together
- Version-lock sklearn and TensorFlow/PyTorch

### 19.4 Common Mistakes in DL + sklearn Pipelines
- Fitting scaler on full data before split — data leakage
- Forgetting `astype('float32')` — GPU inefficiency
- Not stratifying for imbalanced DL classification split
- Using `OrdinalEncoder` output without converting to integer for embedding lookup

---

## 20. Scikit-learn Interview Topics for Deep Learning Engineers

### 20.1 Frequently Asked Questions
- How does sklearn integrate with TensorFlow or PyTorch?
- What sklearn tools are used even when building DL models?
- How to apply sklearn cross-validation to a Keras model?
- How to use sklearn metrics to evaluate a DL model?
- What is the difference between sklearn preprocessing and DL preprocessing layers?

### 20.2 DL-Oriented Coding Problems
- Build a preprocessing pipeline for tabular DL input (mixed numeric + categorical)
- Evaluate a Keras model using sklearn's `classification_report` and `roc_auc_score`
- Perform t-SNE visualization on BERT embeddings using sklearn
- Use `StratifiedKFold` to implement cross-validation for a neural network
- Tune a Keras MLP with `RandomizedSearchCV` using SciKeras

---

## 21. Real-World DL Projects Using Scikit-learn

### 21.1 Beginner Projects
- Preprocessing pipeline for tabular deep learning (Titanic with MLP)
- Baseline comparison — LogisticRegression vs MLP
- t-SNE visualization of CNN features on CIFAR-10

### 21.2 Intermediate Projects
- Full preprocessing + SciKeras pipeline with GridSearchCV
- Comparing sklearn GBM baseline vs Tabular DL model on structured data
- DL embedding cluster analysis with sklearn KMeans + silhouette score

### 21.3 Advanced Projects
- Semi-supervised DL with sklearn LabelPropagation + DL pseudo-labeling
- Full sklearn + DL hybrid inference pipeline with joblib persistence
- Anomaly detection comparison — IsolationForest vs Autoencoder

---

## 22. Scikit-learn Roadmap for Deep Learning Engineers

### 22.1 Beginner Level
- `train_test_split`, `StratifiedKFold`
- `StandardScaler`, `OneHotEncoder`, `SimpleImputer`
- `confusion_matrix`, `classification_report`, `roc_auc_score`
- t-SNE for DL embedding visualization

### 22.2 Intermediate Level
- `ColumnTransformer` + `Pipeline` for tabular DL preprocessing
- SciKeras / Skorch wrappers for sklearn API with DL
- `GridSearchCV` with DL models
- `permutation_importance` for DL feature analysis

### 22.3 Advanced Level
- Full sklearn + DL hybrid pipelines
- SHAP + sklearn integration for DL interpretation
- `joblib` model persistence for production inference
- Semi-supervised support for DL pretraining
