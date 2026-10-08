# XGBoost, LightGBM & CatBoost for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction — Why Gradient Boosting Dominates ML

### 1.1 The Role of Gradient Boosting in Real-World ML
- Top performer on structured/tabular data — consistently wins Kaggle competitions
- Faster to train and tune than deep learning on tabular data
- Interpretable with SHAP values
- Works well without extensive feature engineering
- Production-proven — used in fraud detection, credit scoring, recommendation, search ranking

### 1.2 When to Choose Gradient Boosting Over Other Models
- Tabular data with mixed types (numeric + categorical)
- Features have non-linear relationships and interactions
- Baseline performance required quickly
- Interpretability is needed (SHAP)
- When neural networks underperform on small/medium tabular data

### 1.3 Installing and Setup
- `pip install xgboost lightgbm catboost optuna shap`
- Import conventions
  - `from xgboost import XGBClassifier, XGBRegressor`
  - `from lightgbm import LGBMClassifier, LGBMRegressor`
  - `from catboost import CatBoostClassifier, CatBoostRegressor, Pool`
  - `import optuna`
  - `import shap`

---

## 2. Data Preparation for Gradient Boosting

### 2.1 What Gradient Boosting Can Handle Natively
- Missing values — XGBoost and LightGBM handle `NaN` natively
- No feature scaling required — tree-based, scale-invariant
- High cardinality features — handle well with categorical encoding or natively
- Mixed types — numeric and categorical together

### 2.2 Feature Scaling
- **Not required** for gradient boosting — tree splits are threshold-based
- Never apply `StandardScaler` before XGBoost/LightGBM/CatBoost
- Exception — only if combining with linear models in stacking

### 2.3 Missing Values
- XGBoost — learns optimal direction for missing values during training
  - `NaN` values are routed to the best child automatically
- LightGBM — similar handling, `NaN` always goes to the zero-gain side
- CatBoost — uses default direction for NaN
- No imputation needed before training

### 2.4 Categorical Encoding Approach
- XGBoost — requires manual encoding: `OrdinalEncoder` or `OneHotEncoder`
- LightGBM — native categorical support with `categorical_feature`
- CatBoost — best native support, pass `cat_features` directly
- For XGBoost: use `OrdinalEncoder` (not OneHot — high dimension hurts trees)

### 2.5 Train/Validation/Test Split
- `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` — stratified
- Keep a held-out test set for final evaluation
- Validation set for early stopping
- `StratifiedKFold` for cross-validation

---

## 3. ML Classification Workflow

### 3.1 Binary Classification Pipeline
```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_prob))
```

### 3.2 Multi-class Classification
- XGBoost — `objective='multi:softmax'`, `num_class=n`
- LightGBM — `objective='multiclass'`, `num_class=n`
- CatBoost — `loss_function='MultiClass'`
- `predict_proba()` — shape `(n_samples, n_classes)`
- Metrics — `f1_score(average='macro')`, `roc_auc_score(multi_class='ovr')`

### 3.3 Regression Pipeline
```python
from lightgbm import LGBMRegressor
import lightgbm as lgb

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
```

---

## 4. Evaluation Metrics for ML

### 4.1 Classification Metrics
- `accuracy_score(y_test, y_pred)` — overall accuracy
- `f1_score(y_test, y_pred, average='weighted')` — F1
- `roc_auc_score(y_test, y_prob)` — AUC-ROC (binary)
- `average_precision_score(y_test, y_prob)` — AUC-PR (imbalanced)
- `classification_report(y_test, y_pred)` — full per-class report
- `confusion_matrix(y_test, y_pred)` — confusion matrix

### 4.2 Regression Metrics
- `mean_squared_error(y_test, y_pred, squared=False)` — RMSE
- `mean_absolute_error(y_test, y_pred)` — MAE
- `r2_score(y_test, y_pred)` — R²
- `mean_absolute_percentage_error(y_test, y_pred)` — MAPE

### 4.3 Choosing the Right Metric
- Imbalanced classification — use `roc_auc_score` or `average_precision_score`
- Regression with outliers — use MAE or Huber loss
- Log-scale regression — use RMSLE (`reg:squaredlogerror`)
- Multi-class — use macro F1 for balanced evaluation

---

## 5. Hyperparameter Tuning for ML

### 5.1 Key Parameters to Tune (Priority Order)

#### XGBoost Priority
1. `n_estimators` + `early_stopping_rounds` — always use early stopping
2. `learning_rate` — lower is better (0.01–0.1)
3. `max_depth` — (3–8)
4. `subsample` — (0.6–0.9)
5. `colsample_bytree` — (0.6–0.9)
6. `min_child_weight` — (1–10)
7. `gamma` — (0–5)
8. `reg_alpha`, `reg_lambda` — L1/L2 regularization

#### LightGBM Priority
1. `n_estimators` + `early_stopping_rounds`
2. `learning_rate`
3. `num_leaves` — most important (20–300)
4. `min_child_samples` — (10–100)
5. `subsample` + `subsample_freq`
6. `colsample_bytree`
7. `reg_alpha`, `reg_lambda`

#### CatBoost Priority
1. `iterations` + `early_stopping_rounds`
2. `learning_rate`
3. `depth` — (4–10)
4. `l2_leaf_reg` — (1–10)
5. `bagging_temperature` — (0–1)
6. `random_strength` — (0–10)

### 5.2 Optuna Tuning — XGBoost
```python
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'early_stopping_rounds': 50,
        'eval_metric': 'auc',
        'random_state': 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)
```

### 5.3 Optuna Tuning — LightGBM
```python
def objective(trial):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'random_state': 42
    }
    model = LGBMClassifier(**params)
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

### 5.4 Optuna Tuning — CatBoost
```python
def objective(trial):
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'early_stopping_rounds': 50,
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
```

### 5.5 LightGBM Built-in Tuner
- `optuna.integration.lightgbm.LightGBMTunerCV` — automatic LightGBM tuning
- Tunes `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`

---

## 6. Cross-Validation Best Practices

### 6.1 Sklearn CV
- `cross_val_score(XGBClassifier(**params), X, y, cv=StratifiedKFold(5), scoring='roc_auc')`
- `cross_validate(LGBMClassifier(**params), X, y, cv=5, return_train_score=True)`

### 6.2 Native CV with Early Stopping
- XGBoost native CV — `xgb.cv(params, dtrain, nfold=5, early_stopping_rounds=50)`
- LightGBM native CV — `lgb.cv(params, train_data, nfold=5, callbacks=[lgb.early_stopping(50)])`
- Returns mean + std per round across folds
- Best for finding optimal `n_estimators` before final training

### 6.3 Out-of-Fold (OOF) Predictions
- Training models on each fold, predicting on the held-out fold
- OOF predictions used as features for meta-learner in stacking
- `oof_preds = np.zeros(len(X))`
- `for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):`

---

## 7. Handling Class Imbalance

### 7.1 XGBoost for Imbalanced Data
- `scale_pos_weight = (y == 0).sum() / (y == 1).sum()` — ratio of negatives to positives
- `eval_metric='aucpr'` — use AUC-PR instead of AUC-ROC
- `max_delta_step=1` — helps convergence for imbalanced binary
- Threshold tuning — `precision_recall_curve()` to find optimal threshold

### 7.2 LightGBM for Imbalanced Data
- `is_unbalance=True` — auto-weight classes
- `scale_pos_weight` — manual weight for positive class
- `class_weight='balanced'` — sklearn API
- `metric='average_precision'` — AUC-PR metric

### 7.3 CatBoost for Imbalanced Data
- `auto_class_weights='Balanced'` — auto weight from class counts
- `class_weights=[1, w]` — manual weights
- `eval_metric='AUC'` or `'F1'`

### 7.4 Threshold Tuning
- `precision_recall_curve(y_val, y_prob)` — find best threshold
- `optimal_threshold = thresholds[np.argmax(f1_scores)]`
- Apply custom threshold — `(y_prob >= optimal_threshold).astype(int)`

---

## 8. Feature Engineering for Gradient Boosting

### 8.1 What to Engineer
- Interaction features — ratios, products of important pairs
- Aggregation features — group mean/std/min/max per category
- Lag features — for time-series ML
- Count encoding — frequency of each category value
- Target encoding — mean target per category (with cross-validation)

### 8.2 What Not to Engineer
- Polynomial features — trees find these automatically
- Feature scaling — not needed
- PCA — trees do not benefit from linear combinations

### 8.3 CatBoost Automatic Feature Generation
- Learns combinations of categorical features automatically
- `ctr_target_border_count` — target encoding precision
- No manual target encoding needed for CatBoost

---

## 9. Feature Importance and SHAP

### 9.1 Built-in Importance
- XGBoost gain importance — `model.get_booster().get_score(importance_type='gain')`
- LightGBM gain importance — `model.feature_importance(importance_type='gain')`
- CatBoost prediction change — `model.get_feature_importance()`

### 9.2 SHAP Values — Global Importance
- `explainer = shap.TreeExplainer(model)` — all three supported
- `shap_values = explainer.shap_values(X_val)` — shape `(n_samples, n_features)`
- `shap.summary_plot(shap_values, X_val)` — beeswarm/dot plot
- `shap.bar_plot(shap_values)` — mean absolute SHAP per feature

### 9.3 SHAP Values — Local Explanation
- `shap.waterfall_plot(explainer(X_val)[0])` — single prediction breakdown
- `shap.force_plot(explainer.expected_value, shap_values[0], X_val.iloc[0])` — force plot

### 9.4 SHAP Dependence Plot
- `shap.dependence_plot('feature_name', shap_values, X_val)` — feature effect
- Interaction effects — `interaction_index='auto'`

---

## 10. Model Comparison Workflow

### 10.1 Comparing All Three on Same Dataset
```python
models = {
    'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, early_stopping_rounds=50, eval_metric='auc', random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    results[name] = auc
```

### 10.2 Cross-Validation Comparison
- `cross_val_score` for each model with `StratifiedKFold(5)`
- Report mean ± std per model
- Statistical significance test — `scipy.stats.wilcoxon` or `ttest_rel`

---

## 11. Stacking Ensemble for ML

### 11.1 Building a Stacking Ensemble
- Level-0 models — XGBoost, LightGBM, CatBoost (train on folds)
- Level-1 meta-learner — LogisticRegression on OOF predictions
- `sklearn.ensemble.StackingClassifier` with all three as estimators
- Manual OOF stacking for more control

### 11.2 Weighted Average Ensemble
- `final_pred = 0.4 * xgb_prob + 0.4 * lgbm_prob + 0.2 * cat_prob`
- Tune weights by optimizing validation AUC

### 11.3 Rank Averaging
- `final_pred = 0.33 * rankdata(xgb_prob) + 0.33 * rankdata(lgbm_prob) + 0.33 * rankdata(cat_prob)`
- Rank averaging is more robust than probability averaging

---

## 12. Pipelines for Production ML

### 12.1 sklearn Pipeline with XGBoost
```python
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('model', XGBClassifier(n_estimators=300, learning_rate=0.1))
])
pipe.fit(X_train, y_train)
```

### 12.2 CatBoost with Pool in Pipeline
- CatBoost Pool must be created separately from sklearn Pipeline
- Use `FunctionTransformer` to pass `cat_features` metadata

### 12.3 Model Persistence
- `model.save_model('model.json')` — XGBoost JSON
- `model.save_model('model.txt')` — LightGBM text
- `model.save_model('model.cbm')` — CatBoost binary
- `joblib.dump(pipeline, 'pipeline.pkl')` — full sklearn pipeline

---

## 13. Kaggle and Competition ML Workflow

### 13.1 Standard Competition Approach
1. Baseline — LightGBM with default params + early stopping
2. Feature engineering — add domain knowledge features
3. Hyperparameter tuning — Optuna (50–100 trials)
4. Cross-validation — 5-fold stratified, report OOF score
5. Ensemble — XGBoost + LightGBM + CatBoost blend
6. Final submission — retrain on full training data with best `n_estimators`

### 13.2 Leaderboard Probing Awareness
- OOF score is more reliable than public LB score
- Avoid overfitting to public LB
- Use `n_estimators = best_iteration * 1.1` when retraining on full data

---

## 14. Best Practices for ML with Gradient Boosting

### 14.1 Always Do
- Use early stopping — always
- Use stratified CV — always for classification
- Report mean ± std across folds — not single split
- Use SHAP for feature importance — not split-based
- Set `random_state` / `random_seed` — for reproducibility

### 14.2 Never Do
- Scale features before gradient boosting — unnecessary
- One-hot encode high-cardinality categories before XGBoost — use ordinal
- Fix `n_estimators` without early stopping
- Evaluate only on training data
- Trust accuracy for imbalanced problems

---

## 15. Interview Topics for ML Engineers

### 15.1 Frequently Asked Questions
- How does gradient boosting differ from Random Forest?
- What is the difference between level-wise and leaf-wise growth?
- How does XGBoost handle missing values?
- When would you use CatBoost over LightGBM?
- What is `scale_pos_weight` and why is it used?
- What is the difference between gain and split importance?
- How do SHAP values explain a model?

### 15.2 ML-Oriented Coding Problems
- Build a binary classifier with LightGBM + early stopping + evaluation report
- Tune XGBoost with Optuna for a regression task
- Handle class imbalance in CatBoost for credit risk
- Compare all three models on a Kaggle dataset with cross-validation
- Build a stacking ensemble of XGBoost + LightGBM + CatBoost

---

## 16. Real-World ML Projects

### 16.1 Beginner Projects
- Credit card approval prediction (binary, class imbalance)
- House price prediction (regression)
- Customer churn classification

### 16.2 Intermediate Projects
- Kaggle tabular playground competition — full pipeline with tuning
- Loan default prediction — imbalanced, threshold tuning, SHAP
- Multi-class product category classification

### 16.3 Advanced Projects
- Stacking ensemble with 3 boosting libraries as base models
- Feature importance-driven feature selection pipeline
- Full MLOps pipeline — train, tune, log with MLflow, deploy

---

## 17. Roadmap

### 17.1 Beginner Level
- XGBClassifier / LGBMClassifier / CatBoostClassifier with defaults
- Early stopping — eval_set + early_stopping_rounds
- sklearn metrics — accuracy, F1, AUC
- Feature importance — built-in and SHAP summary plot

### 17.2 Intermediate Level
- Optuna hyperparameter tuning
- Native CV for optimal `n_estimators`
- Class imbalance handling per library
- CatBoost native categorical features
- OOF predictions for stacking

### 17.3 Advanced Level
- Full stacking ensemble pipeline
- SHAP local + global explanations
- MLflow experiment tracking
- Production deployment with ONNX or joblib pipeline
