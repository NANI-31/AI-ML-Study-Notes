# XGBoost, LightGBM & CatBoost — Complete Roadmap / Syllabus

---

## 1. Introduction to Gradient Boosting Libraries

### 1.1 What is Gradient Boosting?
- Ensemble method — sequentially builds trees that correct previous errors
- Each tree fits the residual errors of the ensemble so far
- Gradient descent in function space — minimizing a loss function
- Most powerful classical ML technique for tabular data
- Why gradient boosting dominates Kaggle and industry tabular ML

### 1.2 XGBoost vs LightGBM vs CatBoost — Key Differences

| Feature | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Speed | Fast | Fastest | Fast |
| Memory | Moderate | Low | Moderate |
| Categorical handling | Manual encoding | Native (limited) | Native (best) |
| Leaf growth | Level-wise | Leaf-wise | Symmetric |
| GPU support | Yes | Yes | Yes |
| Best for | General | Large datasets | Categorical-heavy data |

### 1.3 Installing
- `pip install xgboost`
- `pip install lightgbm`
- `pip install catboost`
- Verifying installation
  - `import xgboost as xgb; xgb.__version__`
  - `import lightgbm as lgb; lgb.__version__`
  - `import catboost; catboost.__version__`

### 1.4 Import Conventions
- `import xgboost as xgb`
- `from xgboost import XGBClassifier, XGBRegressor`
- `import lightgbm as lgb`
- `from lightgbm import LGBMClassifier, LGBMRegressor`
- `from catboost import CatBoostClassifier, CatBoostRegressor, Pool`

---

## 2. XGBoost — Extreme Gradient Boosting

### 2.1 What is XGBoost?
- Developed by Tianqi Chen — 2016
- Regularized gradient boosting — L1 and L2 penalties
- Level-wise (depth-wise) tree growth
- Won 17 out of 29 Kaggle competition solutions in its debut year
- Industry-standard for structured/tabular data

### 2.2 XGBoost sklearn API — Classifier
- `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6)`
- `model.fit(X_train, y_train)` — training
- `model.predict(X_test)` — hard predictions
- `model.predict_proba(X_test)` — probability predictions
- `model.score(X_test, y_test)` — accuracy
- `eval_metric` parameter — `'logloss'`, `'auc'`, `'error'`

### 2.3 XGBoost sklearn API — Regressor
- `XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)`
- `eval_metric` — `'rmse'`, `'mae'`, `'rmsle'`

### 2.4 XGBoost Native API
- `dtrain = xgb.DMatrix(X_train, label=y_train)` — DMatrix (faster)
- `dval = xgb.DMatrix(X_val, label=y_val)`
- `params = {'objective': 'binary:logistic', 'max_depth': 6, 'eta': 0.1}`
- `evals = [(dtrain, 'train'), (dval, 'val')]`
- `model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, early_stopping_rounds=20)`
- `model.predict(dval)` — predictions from native API

### 2.5 Early Stopping
- `XGBClassifier(early_stopping_rounds=20)`
- `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)`
- `model.best_iteration` — best number of rounds
- `model.best_score` — best validation score
- Prevents overfitting without fixing `n_estimators`

### 2.6 Key Hyperparameters

#### Tree Parameters
- `n_estimators` — number of boosting rounds (trees)
- `max_depth` — maximum tree depth (default: 6)
  - Lower → simpler trees, less overfitting
  - Higher → more complex, risk of overfitting
- `min_child_weight` — minimum sum of instance weight in a leaf
  - Higher → more conservative, less overfitting
- `max_leaves` — maximum leaf nodes (for `tree_method='hist'`)
- `gamma` — minimum loss reduction to make a split (default: 0)

#### Learning Parameters
- `learning_rate` (alias: `eta`) — shrinkage per step (default: 0.3)
  - Lower → slower but often better generalization
  - Typical range: 0.01–0.3
- `n_estimators` — inversely related to `learning_rate`

#### Sampling Parameters
- `subsample` — fraction of training rows per tree (default: 1)
  - 0.6–0.9 — stochastic gradient boosting
- `colsample_bytree` — fraction of features per tree (default: 1)
- `colsample_bylevel` — fraction of features per level
- `colsample_bynode` — fraction of features per node

#### Regularization Parameters
- `reg_alpha` (L1) — encourages sparsity in weights
- `reg_lambda` (L2) — default: 1, shrinks weights
- Higher regularization → simpler model

### 2.7 Objective Functions
- `'binary:logistic'` — binary classification (outputs probabilities)
- `'binary:hinge'` — binary classification (outputs 0 or 1)
- `'multi:softmax'` — multi-class (outputs class)
- `'multi:softprob'` — multi-class (outputs probabilities)
- `'reg:squarederror'` — regression (MSE)
- `'reg:absoluteerror'` — regression (MAE)
- `'reg:squaredlogerror'` — regression (RMSLE)
- `'reg:pseudohubererror'` — regression (Huber)
- `'rank:pairwise'` — ranking
- `'survival:cox'` — survival analysis

### 2.8 Evaluation Metrics
- Classification — `'logloss'`, `'auc'`, `'aucpr'`, `'error'`, `'merror'`, `'mlogloss'`
- Regression — `'rmse'`, `'rmsle'`, `'mae'`, `'mape'`
- `eval_metric` parameter — list for multiple metrics

### 2.9 Feature Importance
- `model.feature_importances_` — sklearn API (weight-based)
- `model.get_booster().get_score(importance_type='weight')` — number of times used
- `importance_type='gain'` — average gain per split
- `importance_type='cover'` — average coverage per split
- `xgb.plot_importance(model)` — bar chart
- `xgb.plot_importance(model, importance_type='gain')` — gain importance

### 2.10 Saving and Loading
- `model.save_model('model.json')` — saving (JSON format)
- `model.load_model('model.json')` — loading
- `model.save_model('model.ubj')` — binary format (faster)
- sklearn-style — `joblib.dump(model, 'model.pkl')`

### 2.11 GPU Training with XGBoost
- `tree_method='hist'` — histogram-based (fast on CPU)
- `device='cuda'` — GPU training (XGBoost 2.0+)
- Older syntax — `tree_method='gpu_hist'`

---

## 3. LightGBM — Light Gradient Boosting Machine

### 3.1 What is LightGBM?
- Developed by Microsoft — 2017
- Leaf-wise tree growth — grows the leaf with max delta loss
- Histogram-based algorithm — bins continuous features
- GOSS — Gradient-based One-Side Sampling (samples hard instances)
- EFB — Exclusive Feature Bundling (reduces feature count)
- Fastest among the three for large datasets
- Memory efficient — handles millions of rows

### 3.2 LightGBM sklearn API — Classifier
- `LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31)`
- `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20)])`
- `model.predict(X_test)` — class predictions
- `model.predict_proba(X_test)` — probability predictions
- `model.score(X_test, y_test)` — accuracy

### 3.3 LightGBM sklearn API — Regressor
- `LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31)`
- `metric` parameter — `'rmse'`, `'mae'`, `'mse'`

### 3.4 LightGBM Native API
- `train_data = lgb.Dataset(X_train, label=y_train)` — Dataset
- `val_data = lgb.Dataset(X_val, label=y_val)`
- `params = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31, 'learning_rate': 0.05}`
- `callbacks = [lgb.early_stopping(20), lgb.log_evaluation(50)]`
- `model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[val_data], callbacks=callbacks)`
- `model.predict(X_test)` — predictions

### 3.5 Early Stopping
- `lgb.early_stopping(stopping_rounds=20)` — callback
- `lgb.log_evaluation(period=50)` — logging callback
- `model.best_iteration` — best round

### 3.6 Key Hyperparameters

#### Tree Structure
- `num_leaves` — max leaves per tree (default: 31)
  - Key parameter — more leaves → more complex
  - Typical range: 20–300
- `max_depth` — maximum tree depth (default: -1, unlimited)
  - Set to limit `num_leaves` effect
- `min_child_samples` — minimum data in a leaf (default: 20)
  - Higher → less overfitting

#### Learning Parameters
- `n_estimators` / `num_boost_round` — number of trees
- `learning_rate` — step size shrinkage (default: 0.1)

#### Sampling Parameters
- `subsample` / `bagging_fraction` — row sampling per tree
- `subsample_freq` / `bagging_freq` — frequency of bagging (set > 0 to enable)
- `colsample_bytree` / `feature_fraction` — feature fraction per tree

#### Regularization
- `reg_alpha` / `lambda_l1` — L1 regularization
- `reg_lambda` / `lambda_l2` — L2 regularization
- `min_gain_to_split` — minimum gain to split (default: 0)
- `path_smooth` — smoothing for leaf value

### 3.7 Native Categorical Feature Support
- Pass `categorical_feature=['col1', 'col2']` in `fit()` or `lgb.Dataset`
- LightGBM handles categorical natively — no need for encoding
- Internally uses optimal split for categories
- `cat_smooth` — categorical smoothing parameter

### 3.8 Objective Functions
- `'binary'` — binary classification
- `'multiclass'` — multi-class classification
  - `num_class` parameter required
- `'regression'` — regression (L2 loss)
- `'regression_l1'` — regression (L1 loss)
- `'huber'` — Huber regression
- `'mape'` — mean absolute percentage error
- `'poisson'` — Poisson regression
- `'quantile'` — quantile regression
- `'tweedie'` — Tweedie regression
- `'rank_xendcg'` — learning-to-rank

### 3.9 Feature Importance
- `model.feature_importance(importance_type='split')` — number of splits
- `model.feature_importance(importance_type='gain')` — total gain
- `lgb.plot_importance(model, importance_type='gain')` — bar chart
- `model.feature_name()` — feature names

### 3.10 Saving and Loading
- `model.save_model('model.txt')` — native text format
- `model = lgb.Booster(model_file='model.txt')` — loading native
- sklearn API — `joblib.dump(model, 'model.pkl')`

### 3.11 GPU Training
- `device='gpu'` or `device_type='gpu'` — GPU training
- `gpu_platform_id`, `gpu_device_id` — multi-GPU selection

---

## 4. CatBoost — Categorical Boosting

### 4.1 What is CatBoost?
- Developed by Yandex — 2017
- Symmetric (oblivious) tree growth — same split at each level
- Best native categorical feature handling — no manual encoding needed
- Ordered boosting — prevents target leakage
- Strong performance out-of-the-box with minimal tuning

### 4.2 CatBoost sklearn API — Classifier
- `CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)`
- `model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20)`
- `model.predict(X_test)` — class predictions
- `model.predict_proba(X_test)` — probability predictions

### 4.3 CatBoost sklearn API — Regressor
- `CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, loss_function='RMSE', verbose=0)`

### 4.4 Pool — CatBoost Data Container
- `train_pool = Pool(X_train, label=y_train, cat_features=cat_cols)` — with categoricals
- `val_pool = Pool(X_val, label=y_val, cat_features=cat_cols)`
- `model.fit(train_pool, eval_set=val_pool)`
- Efficiently passes categorical column info

### 4.5 Native Categorical Feature Handling
- `cat_features` — list of column indices or names
- No encoding needed — pass raw string/integer categories
- CatBoost uses target statistics internally
- `cat_features` in `Pool` or directly in `fit()`
- Text features — `text_features` parameter
- Embedding features — `embedding_features` parameter

### 4.6 Key Hyperparameters
- `iterations` — number of trees
- `learning_rate` — step size (default: auto)
- `depth` — tree depth (default: 6, max: 16)
- `l2_leaf_reg` — L2 regularization (default: 3)
- `border_count` — quantization bins for numerical features (default: 254)
- `bagging_temperature` — Bayesian bootstrap (default: 1)
- `random_strength` — randomness for split scoring
- `subsample` — row sampling (only for `Bernoulli` bootstrap)
- `colsample_bylevel` — feature fraction per level
- `grow_policy` — `'SymmetricTree'` (default), `'Depthwise'`, `'Lossguide'`
- `min_data_in_leaf` — minimum samples per leaf
- `one_hot_max_size` — categories threshold for one-hot encoding

### 4.7 Loss Functions / Objectives
- `'Logloss'` — binary classification
- `'CrossEntropy'` — binary with probability targets
- `'MultiClass'` — multi-class
- `'MultiClassOneVsAll'` — OvR multi-class
- `'RMSE'` — regression
- `'MAE'` — regression
- `'Huber'` — regression
- `'MAPE'` — regression
- `'Poisson'` — Poisson regression
- `'Tweedie'` — Tweedie regression
- `'PairLogit'` — ranking

### 4.8 Feature Importance
- `model.get_feature_importance()` — default importance
- `model.get_feature_importance(type='PredictionValuesChange')` — prediction change
- `model.get_feature_importance(type='LossFunctionChange')` — loss change
- `model.get_feature_importance(type='ShapValues', data=pool)` — SHAP values (built-in)
- `model.get_feature_importance(type='Interaction')` — feature interaction

### 4.9 Saving and Loading
- `model.save_model('model.cbm')` — CatBoost binary model
- `model = CatBoostClassifier(); model.load_model('model.cbm')` — loading
- `model.save_model('model.json', format='json')` — JSON format
- `model.save_model('model.onnx', format='onnx')` — ONNX export

### 4.10 GPU Training
- `task_type='GPU'` — GPU training
- `devices='0'` or `devices='0:1'` — GPU selection
- Automatically handles GPU memory

---

## 5. Early Stopping — All Three Libraries

### 5.1 Why Early Stopping
- Avoids fixing `n_estimators` / `iterations` manually
- Stops training when validation metric stops improving
- Prevents overfitting
- More efficient training

### 5.2 Early Stopping in Each Library
- XGBoost — `early_stopping_rounds=20` in `XGBClassifier()` + `eval_set=[(X_val, y_val)]`
- LightGBM — `lgb.early_stopping(stopping_rounds=20)` callback
- CatBoost — `early_stopping_rounds=20` in `fit()`

### 5.3 Best Iteration
- XGBoost — `model.best_iteration`
- LightGBM — `model.best_iteration_`
- CatBoost — `model.get_best_iteration()`

---

## 6. Hyperparameter Tuning

### 6.1 Optuna Integration
- `import optuna`
- Define trial for XGBoost / LightGBM / CatBoost
- `trial.suggest_float('learning_rate', 1e-4, 0.3, log=True)`
- `trial.suggest_int('max_depth', 3, 10)`
- `trial.suggest_int('n_estimators', 100, 1000)`
- `study.optimize(objective, n_trials=50)`

### 6.2 Optuna with LightGBM
- `optuna.integration.lightgbm.LightGBMTunerCV` — built-in tuner
- Automatically tunes key LightGBM parameters

### 6.3 GridSearchCV / RandomizedSearchCV
- sklearn API compatibility
- `GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='roc_auc')`
- `RandomizedSearchCV(LGBMClassifier(), param_dist, n_iter=50, cv=5)`

---

## 7. Feature Importance and Interpretation

### 7.1 Built-in Feature Importance
- XGBoost — `model.feature_importances_` (sklearn) or `get_score()`
- LightGBM — `model.feature_importance(importance_type='gain')`
- CatBoost — `model.get_feature_importance()`

### 7.2 SHAP Values
- `import shap`
- XGBoost — `shap.TreeExplainer(xgb_model)`
- LightGBM — `shap.TreeExplainer(lgbm_model)`
- CatBoost — built-in SHAP via `get_feature_importance(type='ShapValues')`
- `shap_values = explainer.shap_values(X_test)` — per-sample SHAP
- `shap.summary_plot(shap_values, X_test)` — beeswarm plot
- `shap.waterfall_plot(explanation[0])` — single prediction explanation
- `shap.dependence_plot('feature', shap_values, X)` — feature effect

### 7.3 Permutation Importance
- `from sklearn.inspection import permutation_importance`
- Works with sklearn API of all three libraries

---

## 8. Handling Class Imbalance

### 8.1 XGBoost
- `scale_pos_weight = neg_count / pos_count` — for binary
- `max_delta_step=1` — helps with imbalanced data
- `eval_metric='aucpr'` — use AUC-PR for imbalanced evaluation

### 8.2 LightGBM
- `is_unbalance=True` — auto-balance (sets pos_weight)
- `class_weight='balanced'` — sklearn API
- `scale_pos_weight` — manual weight for positive class

### 8.3 CatBoost
- `class_weights=[1, 10]` — per-class weight
- `auto_class_weights='Balanced'` — automatic balancing
- `auto_class_weights='SqrtBalanced'` — square root balancing

---

## 9. Cross-Validation

### 9.1 sklearn cross_val_score
- All three libraries have sklearn-compatible APIs
- `cross_val_score(XGBClassifier(), X, y, cv=5, scoring='roc_auc')`
- `cross_val_score(LGBMClassifier(), X, y, cv=5, scoring='f1_weighted')`
- `cross_val_score(CatBoostClassifier(verbose=0), X, y, cv=5)`

### 9.2 Native Cross-Validation
- XGBoost — `xgb.cv(params, dtrain, num_boost_round=500, nfold=5, early_stopping_rounds=20)`
- LightGBM — `lgb.cv(params, train_data, num_boost_round=500, nfold=5, callbacks=[lgb.early_stopping(20)])`
- Returns per-round mean and std across folds

---

## 10. Integration with sklearn Pipelines

### 10.1 Pipeline Compatibility
- All three support sklearn `Pipeline` via sklearn API
- `Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier())])`
- `Pipeline([('model', LGBMClassifier())])` — no scaling needed for tree models
- `Pipeline([('model', CatBoostClassifier(verbose=0))])`

### 10.2 ColumnTransformer + Boosting
- Preprocessing with `ColumnTransformer` before boosting
- Note: tree models don't require scaling — only embedding/one-hot matters

### 10.3 GridSearchCV in Pipeline
- `param_grid = {'model__n_estimators': [100, 300], 'model__max_depth': [4, 6]}`
- `GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')`

---

## 11. Comparison and When to Use Each

### 11.1 Use XGBoost When
- Need widest compatibility and community support
- Deploying to environments where LightGBM/CatBoost are unavailable
- Need level-wise tree control
- Working with sparse data

### 11.2 Use LightGBM When
- Dataset is large (>100k rows)
- Training speed is critical
- Memory is limited
- Have many features

### 11.3 Use CatBoost When
- Dataset has many categorical features (>5 high-cardinality columns)
- Want best out-of-the-box performance with minimal tuning
- Want built-in SHAP values
- Target encoding is desired

---

## 12. Model Persistence

### 12.1 Saving Models
- XGBoost — `model.save_model('xgb.json')` or `joblib.dump(model, 'xgb.pkl')`
- LightGBM — `model.save_model('lgbm.txt')` or `joblib.dump(model, 'lgbm.pkl')`
- CatBoost — `model.save_model('catboost.cbm')`

### 12.2 ONNX Export
- XGBoost — `model.save_model('model.onnx')` (with onnxmltools)
- CatBoost — `model.save_model('model.onnx', format='onnx')`

---

## 13. Common Patterns and Best Practices

### 13.1 General Rules
- Start with LightGBM defaults — usually fast and strong
- Use early stopping always — never fix `n_estimators` manually
- Lower `learning_rate` → need more `n_estimators` → use early stopping
- Use `colsample_bytree` and `subsample` for regularization
- Use `num_leaves` (LightGBM) as the primary complexity control

### 13.2 Common Mistakes
- Not using early stopping — wastes compute and overfits
- Setting `max_depth` too high in LightGBM — use `num_leaves` instead
- Forgetting `cat_features` in CatBoost — string columns will error
- Using accuracy as metric for imbalanced data
- Encoding categoricals before CatBoost — unnecessary and harmful

### 13.3 Reproducibility
- `random_state=42` or `seed=42` — XGBoost/LightGBM
- `random_seed=42` — CatBoost
- `nthread=1` or `n_jobs=1` — for fully deterministic results

---

## 14. Interview Topics

### 14.1 Frequently Asked Questions
- What is the difference between XGBoost, LightGBM, and CatBoost?
- What is leaf-wise vs level-wise tree growth?
- How does CatBoost handle categorical features?
- What is `scale_pos_weight` and when to use it?
- What is early stopping in gradient boosting?
- What is the difference between `gain` and `split` importance?

### 14.2 Coding Problems
- Train an XGBoost classifier with early stopping on a binary classification problem
- Tune LightGBM hyperparameters with Optuna
- Pass categorical features natively to CatBoost
- Compute SHAP values for an XGBoost model
- Compare all three on the same dataset with cross-validation

---

## 15. Real-World Projects

### 15.1 Beginner Projects
- Titanic survival prediction with XGBoost
- House price regression with LightGBM
- Customer churn prediction with CatBoost

### 15.2 Intermediate Projects
- Kaggle tabular competition with hyperparameter tuning
- Credit card fraud detection with imbalanced handling
- Multi-class classification with SHAP explanation

### 15.3 Advanced Projects
- Ensemble of XGBoost + LightGBM + CatBoost with stacking
- Full MLOps pipeline — train, tune, evaluate, export to ONNX
- Real-time inference with CatBoost ONNX model

---

## 16. Ecosystem

### 16.1 SHAP Integration
- `shap.TreeExplainer` — optimized for tree models
- Works with all three libraries
- Force plots, summary plots, waterfall plots

### 16.2 Optuna Integration
- `optuna.integration.lightgbm.LightGBMTuner` — LightGBM-specific
- Manual Optuna for XGBoost and CatBoost

### 16.3 MLflow Integration
- `mlflow.xgboost.autolog()` — automatic logging
- `mlflow.lightgbm.autolog()` — automatic logging
- `mlflow.catboost.log_model(model, 'model')` — manual logging
