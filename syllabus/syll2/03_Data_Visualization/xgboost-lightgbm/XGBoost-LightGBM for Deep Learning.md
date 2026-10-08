# XGBoost, LightGBM & CatBoost — Supporting Deep Learning Workflows

---

## 1. Introduction — Gradient Boosting in the Context of Deep Learning

### 1.1 Why Gradient Boosting Still Matters in DL Projects
- DL models underperform gradient boosting on small/medium tabular datasets
- Gradient boosting as a strong baseline before committing to DL
- Feature selection for DL — SHAP identifies what to feed the neural network
- Hybrid architectures — gradient boosting on DL-extracted features
- Gradient boosting as a meta-learner on DL model outputs (stacking)
- Faster training, no GPU required for tabular data

### 1.2 When to Use Gradient Boosting Instead of DL
- Tabular data with < 100k rows
- Dataset has many categorical features
- Training time budget is limited
- Interpretability is required
- No structured/unstructured mixed input

### 1.3 When to Use Both Together
- DL features (embeddings, activations) + gradient boosting classifier
- DL model probability output → gradient boosting meta-learner
- Gradient boosting for preprocessing / feature selection → DL for final prediction

---

## 2. Gradient Boosting as a Baseline for DL Comparison

### 2.1 Establishing a Strong Baseline
- Always train LightGBM/XGBoost before any DL model on tabular data
- Baseline AUC, F1, RMSE — target for DL to beat
- If DL can't beat gradient boosting baseline → reconsider DL
- Gradient boosting often wins on structured tabular data

### 2.2 Baseline Training
```python
from lightgbm import LGBMClassifier
import lightgbm as lgb

baseline = LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(0)]
baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)

baseline_auc = roc_auc_score(y_val, baseline.predict_proba(X_val)[:, 1])
print(f'LightGBM Baseline AUC: {baseline_auc:.4f}')
# → Now compare DL model AUC to this number
```

### 2.3 Reporting DL Gain Over Baseline
- `dl_gain = dl_auc - baseline_auc`
- If `dl_gain < 0.005` — gradient boosting is sufficient
- Gradient boosting baseline protects against overengineering DL

---

## 3. Feature Selection for Deep Learning Using Gradient Boosting

### 3.1 Why Feature Selection Matters for DL
- DL models on tabular data are sensitive to noisy features
- High-dimensional input → slow training, overfitting
- SHAP-based selection identifies truly predictive features
- Reduces input layer size → smaller, faster DL model

### 3.2 SHAP-based Feature Selection
```python
import shap

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_val)

# Mean absolute SHAP per feature
shap_importance = pd.DataFrame({
    'feature': X_val.columns,
    'shap_mean': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_mean', ascending=False)

# Select top-N features for DL input
top_features = shap_importance['feature'].head(30).tolist()
X_dl = X[top_features]
```

### 3.3 Permutation Importance for DL Input Design
- `permutation_importance(lgbm_model, X_val, y_val, n_repeats=10)`
- Remove features with near-zero permutation importance
- Results in smaller, cleaner input for DL

### 3.4 Gradient Boosting as Feature Ranker
- Train LightGBM → rank features by gain importance
- Use top-K features as input to DL
- Iteratively drop low-importance features → monitor DL validation metric

---

## 4. Gradient Boosting Features as DL Input

### 4.1 Leaf Index Features (Tree Embeddings)
- XGBoost leaf encoding — `model.apply(X)` — returns leaf index per tree
- `apply()` → shape `(n_samples, n_estimators)` — categorical features
- Embed leaf indices with `nn.Embedding` in DL model
- Facebook used this in 2014 — GBDT + logistic regression / DL

```python
# XGBoost leaf features
xgb_leaves = xgb_model.apply(X_train)  # (n_samples, n_trees)
# One-hot encode leaf indices and stack with original features
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
xgb_leaf_features = enc.fit_transform(xgb_leaves)

# Concatenate with original features for DL input
import scipy.sparse as sp
X_dl_input = sp.hstack([X_scaled_sparse, xgb_leaf_features])
```

### 4.2 Predicted Probability as DL Feature
- `lgbm_prob = lgbm_model.predict_proba(X)[:, 1]` — class probability
- Append to feature matrix — `X_with_prob = np.column_stack([X, lgbm_prob])`
- Useful for DL that needs calibrated priors

### 4.3 SHAP Values as DL Features
- `shap_values = explainer.shap_values(X)` — shape `(n_samples, n_features)`
- SHAP values as an alternative feature representation
- Rich signal — encodes interaction effects

---

## 5. Hybrid Architectures — Gradient Boosting + DL

### 5.1 Two-Stage Pipeline
- Stage 1 — LightGBM trains on raw tabular features → generates predictions / leaf embeddings
- Stage 2 — DL model trains on LightGBM outputs + original features
- LightGBM captures non-linear interactions → DL refines on top

### 5.2 Gradient Boosting as Meta-Learner (Stacking)
- DL model outputs class probabilities on held-out val set
- LightGBM / XGBoost trains on DL probabilities + other model outputs
- Meta-learner combines DL + classical model predictions
- Common in competitive ML (Kaggle)

```python
# DL model OOF predictions
dl_oof = dl_model.predict(X_val)  # (n_val_samples, n_classes)

# Stack OOF predictions
meta_X = np.column_stack([dl_oof, xgb_oof, lgbm_oof])

# Meta-learner
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_train)
```

### 5.3 Residual Modeling
- Train DL model → predict → compute residuals
- Train LightGBM on residuals → correct systematic errors
- `final_pred = dl_pred + lgbm_residual_pred`

---

## 6. Feature Engineering for DL Inputs Using Gradient Boosting Insights

### 6.1 Interaction Features Discovered by Trees
- `shap.dependence_plot('f1', shap_values, X)` — identify interaction between f1 and another feature
- Create explicit interaction feature — `X['f1_x_f2'] = X['f1'] * X['f2']`
- Feed enriched feature set to DL

### 6.2 Target Encoding via LightGBM
- CatBoost's internal target encoding can inspire DL feature engineering
- Apply target encoding from CatBoost to raw categoricals → use as DL input
- `Pool(X, label=y, cat_features=cat_cols)` → fit → extract encoded representation

### 6.3 Aggregation Features from Gradient Boosting Analysis
- SHAP highlights which groups matter
- Group-level statistics — mean, std, count of target per group
- Feed aggregation features to DL input layer

---

## 7. Training Dynamics — Gradient Boosting vs DL

### 7.1 Overfitting Comparison
- Gradient boosting — early stopping, num_leaves, regularization
- DL — dropout, weight decay, early stopping, more data
- Gradient boosting overfit profile — typically smoother
- DL overfit profile — harder to detect without validation curve

### 7.2 Sample Efficiency
- Gradient boosting outperforms DL on < 10k rows
- DL starts winning above 50k–100k rows (varies by architecture)
- Gradient boosting on subsets as proxy for DL training size effect

### 7.3 Training Cost Comparison
- LightGBM — seconds to minutes on CPU
- XGBoost/CatBoost — seconds to minutes on CPU
- DL — minutes to hours on GPU
- For iteration speed — always prototype with gradient boosting first

---

## 8. Gradient Boosting for Structured Data in DL Pipelines

### 8.1 Tabular DL Architectures That Compete with Gradient Boosting
- TabNet — attention-based tabular DL
- NODE — neural oblivious decision ensembles
- FT-Transformer — feature tokenizer transformer
- SAINT — self-attention and intersample attention
- Still — LightGBM/XGBoost often outperform on small datasets

### 8.2 When Gradient Boosting Wins Over Tabular DL
- Fewer than 50k training samples
- Dataset has many missing values
- Limited compute budget
- Categorical features dominate
- No benefit from complex feature interactions beyond tree depth

### 8.3 Combining Tabular DL + Gradient Boosting
- Tabular DL extracts higher-order interactions
- Gradient boosting meta-learner combines outputs
- Blending: `0.5 * tabnet_prob + 0.5 * lgbm_prob`

---

## 9. Gradient Boosting for NLP and Vision Preprocessing

### 9.1 Gradient Boosting on NLP Metadata Features
- Text length, word count, sentence count, punctuation count
- Sentiment score (from lexicon)
- Named entity count
- These metadata features — feed to LightGBM alongside DL text embeddings

### 9.2 Gradient Boosting on BERT Embeddings
- Extract [CLS] token embeddings from BERT — shape `(n_samples, 768)`
- `PCA(n_components=50).fit_transform(embeddings)` — reduce dimension
- `lgbm_model.fit(pca_embeddings, y_train)` — LightGBM on BERT features
- Fast classification on top of frozen DL embeddings

### 9.3 Gradient Boosting on CNN Features
- Extract final-layer features from pre-trained ResNet — shape `(n_samples, 2048)`
- Apply PCA → LightGBM classifier on top
- Faster than fine-tuning entire CNN for tabular-like image tasks

---

## 10. Model Interpretation in DL Context

### 10.1 Comparing SHAP Explanations — Gradient Boosting vs DL
- SHAP for gradient boosting — fast, exact via TreeExplainer
- SHAP for DL — slow, approximate via DeepExplainer or KernelExplainer
- Gradient boosting SHAP as proxy for DL feature importance

### 10.2 Cross-model Feature Consistency
- Features important in LightGBM SHAP — check if DL also relies on them
- Inconsistency → investigate data quality or DL architecture
- `shap.summary_plot` side-by-side for gradient boosting vs DL

---

## 11. Production Deployment Comparison

### 11.1 Gradient Boosting Deployment (Simpler)
- `model.save_model('model.cbm')` — CatBoost
- `model.save_model('model.json')` — XGBoost
- No GPU required for inference
- Sub-millisecond latency for tabular prediction
- ONNX export for cross-platform inference

### 11.2 DL Deployment (More Complex)
- TorchScript / ONNX / TensorRT for inference
- GPU may be required for low-latency DL serving
- Heavier infrastructure vs gradient boosting
- TorchServe / TFServing for production scale

### 11.3 When to Deploy Gradient Boosting vs DL
- Low-latency requirements → gradient boosting (no GPU)
- High throughput batch → either works
- Unstructured data (text/image/audio) → DL required
- Tabular with <50k rows → gradient boosting preferred

---

## 12. Experiment Tracking and MLOps

### 12.1 MLflow with Gradient Boosting
- `mlflow.xgboost.autolog()` — auto-logs params, metrics, model
- `mlflow.lightgbm.autolog()` — auto-logs
- `mlflow.catboost.log_model(model, 'catboost-model')` — manual
- Comparing gradient boosting runs vs DL runs in same MLflow project

### 12.2 Weights & Biases
- `wandb.init(project='tabular-ml')`
- `wandb.log({'val_auc': auc, 'best_iteration': model.best_iteration_})`
- Visualizing gradient boosting vs DL experiment runs side by side

### 12.3 Reproducibility
- `random_state=42` — XGBoost/LightGBM
- `random_seed=42` — CatBoost
- Log all hyperparameters alongside results

---

## 13. Real-World DL + Gradient Boosting Projects

### 13.1 Beginner Projects
- BERT embeddings → LightGBM classifier (sentiment analysis)
- ResNet features → XGBoost image classifier (no fine-tuning)
- Tabular DL baseline vs LightGBM comparison on structured dataset

### 13.2 Intermediate Projects
- Two-stage pipeline — LightGBM leaf features → DL model
- Stacking — BERT + TabNet + LightGBM meta-ensemble
- SHAP-guided feature engineering for tabular DL model

### 13.3 Advanced Projects
- Full hybrid tabular pipeline — gradient boosting + FT-Transformer + meta-learner
- DL + LightGBM residual modeling
- Multi-modal: CNN image features + tabular gradient boosting combined via stacking

---

## 14. Gradient Boosting Roadmap in the Context of DL

### 14.1 Beginner Level
- Train gradient boosting baseline before any DL model
- Compare baseline AUC with DL model AUC
- Use SHAP for feature selection before DL input design

### 14.2 Intermediate Level
- Extract DL features (BERT/CNN) → gradient boosting classifier
- Two-stage hybrid pipeline — gradient boosting → DL
- Stacking — gradient boosting as meta-learner over DL + classical models

### 14.3 Advanced Level
- Gradient boosting residual correction on top of DL predictions
- SHAP consistency analysis across gradient boosting and DL models
- Production hybrid deployment — gradient boosting for tabular, DL for unstructured, meta-learner combining both
