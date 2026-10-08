# PyTorch for Machine Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to PyTorch for Machine Learning

### 1.1 Why Use PyTorch for Machine Learning
- Dynamic computation graphs enable flexible model architectures
- Debugging is Pythonic — standard `print()` and `pdb` work
- Strong GPU acceleration for large tabular and structured data
- Unified framework for both classical ML and deep learning
- Gradient-based optimization available for any differentiable model
- Rich ecosystem for structured data (tabular, time series)

### 1.2 Role of PyTorch in the ML Workflow
- Feature preprocessing using tensor operations
- Implementing custom ML models as `nn.Module`
- Gradient-based training with `autograd`
- GPU-accelerated training on large datasets
- Model evaluation with `torchmetrics`
- Custom loss functions for non-standard ML objectives

### 1.3 Installing and Setup
- Installing PyTorch (CPU and CUDA versions)
- Import conventions
  - `import torch`
  - `import torch.nn as nn`
  - `import torch.optim as optim`
  - `from torch.utils.data import Dataset, DataLoader, TensorDataset`
  - `import torchmetrics`
  - `import pandas as pd`
  - `import numpy as np`

---

## 2. Data Preparation for Machine Learning

### 2.1 Loading Tabular Data
- Reading CSV with Pandas — `pd.read_csv()`
- Extracting feature matrix `X` and target `y`
- Converting to PyTorch tensors
  - `X_tensor = torch.tensor(X.values, dtype=torch.float32)`
  - `y_tensor = torch.tensor(y.values, dtype=torch.long)` — classification
  - `y_tensor = torch.tensor(y.values, dtype=torch.float32)` — regression

### 2.2 Preprocessing with NumPy + sklearn before PyTorch
- StandardScaler — `scaler.fit_transform(X_train)`
- OneHotEncoder — encoding categorical features
- `X_scaled.astype('float32')` — ensuring float32 for GPU efficiency

### 2.3 TensorDataset
- `TensorDataset(X_tensor, y_tensor)` — wrapping tensors as dataset
- `len(dataset)` — dataset size
- `dataset[0]` — first sample

### 2.4 Custom Dataset for Tabular ML
```python
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### 2.5 DataLoader for ML
- `DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)`
- `pin_memory=True` — faster CPU to GPU transfer
- `drop_last=True` — consistent batch sizes for BatchNorm

### 2.6 Train/Validation/Test Split
- Using sklearn `train_test_split` before tensor conversion
- `stratify=y` — stratified for classification
- Creating three separate `TensorDataset` and `DataLoader` objects

---

## 3. Feature Engineering with PyTorch

### 3.1 Tensor-based Feature Operations
- Polynomial features — `torch.cat([X, X**2, X**3], dim=1)`
- Log transform — `torch.log1p(X)`
- Interaction features — `X[:, 0:1] * X[:, 1:2]`
- Binning — `torch.bucketize(X, boundaries)`

### 3.2 Normalization Inside the Model
- Embedding normalization with `nn.LayerNorm`
- `nn.BatchNorm1d` as a preprocessing step inside the model
- Learnable scaling — `nn.Parameter(torch.ones(n_features))`

### 3.3 Embedding Layers for Categorical Features
- `nn.Embedding(num_categories, embedding_dim)` — categorical embedding
- Entity embedding for high-cardinality categories
- Concatenating embeddings with numerical features

---

## 4. Building ML Models in PyTorch

### 4.1 Linear Regression in PyTorch
- `nn.Linear(n_features, 1)` — single output
- `nn.MSELoss()` — loss function
- `optim.SGD(model.parameters(), lr=0.01)`
- Training loop with `loss.backward()` and `optimizer.step()`

### 4.2 Logistic Regression in PyTorch
- `nn.Linear(n_features, n_classes)` — output logits
- `nn.CrossEntropyLoss()` — includes softmax
- `nn.BCEWithLogitsLoss()` — for binary classification
- `torch.argmax(outputs, dim=1)` — predicted class

### 4.3 Multi-Layer Perceptron for Tabular ML
- Stacking `nn.Linear` + `nn.BatchNorm1d` + `nn.ReLU` + `nn.Dropout`
- Defining via `nn.Sequential` or custom `nn.Module`
- Architecture patterns
  - Wide first layer → narrowing layers → output
  - Residual connections for deep MLPs

### 4.4 Custom MLP with nn.Module
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
```

### 4.5 Tabular Model with Embeddings
- Separate embedding for each categorical column
- Concatenate embeddings + numerical features
- Pass concatenated input through MLP
- `TabNet`, `NODE`, `SAINT` — advanced tabular DL architectures

---

## 5. Loss Functions for ML Tasks

### 5.1 Classification Losses
- `nn.CrossEntropyLoss()` — multi-class classification (logits input)
- `nn.BCEWithLogitsLoss()` — binary classification (logits input)
- `nn.NLLLoss()` — negative log likelihood
- Weighted loss for class imbalance
  - `nn.CrossEntropyLoss(weight=class_weights_tensor)`
  - `class_weights = 1.0 / class_counts`

### 5.2 Regression Losses
- `nn.MSELoss()` — mean squared error
- `nn.L1Loss()` — mean absolute error
- `nn.HuberLoss(delta=1.0)` — robust to outliers
- `nn.SmoothL1Loss()` — Huber loss (legacy name)

### 5.3 Custom Loss Functions
- Subclassing `nn.Module` for custom loss
- `torch.mean((y_pred - y_true)**2 * weights)` — weighted MSE
- Focal loss for extreme class imbalance
- Ranking loss — `nn.MarginRankingLoss()`

---

## 6. Optimizers for ML

### 6.1 Optimizer Selection
- `optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)` — classic
- `optim.Adam(params, lr=1e-3, weight_decay=0)` — adaptive
- `optim.AdamW(params, lr=1e-3, weight_decay=1e-2)` — decoupled weight decay
- `optim.RMSprop(params, lr=1e-3)` — for noisy gradients

### 6.2 Learning Rate Scheduling for ML
- `lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)` — step decay
- `lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)` — adaptive
- `lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)` — cosine decay
- `lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=len(loader)*epochs)`

### 6.3 Differential Learning Rates
- `optimizer = optim.Adam([{'params': embed_params, 'lr': 1e-4}, {'params': fc_params, 'lr': 1e-3}])`
- Lower LR for pretrained embedding layers
- Higher LR for newly added layers

---

## 7. Training Loop for ML

### 7.1 Standard Training Loop
```python
model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

### 7.2 Validation Loop
```python
model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        val_loss = criterion(outputs, y_batch)
```

### 7.3 Early Stopping
- Monitoring validation loss per epoch
- Patience counter — increment if no improvement
- Saving best model state — `torch.save(model.state_dict(), 'best.pth')`
- Restoring best model after patience exceeded

### 7.4 Gradient Accumulation
- For larger effective batch size
- `if (batch_idx + 1) % accumulation_steps == 0:` — step every N batches
- `optimizer.zero_grad()` only after stepping

---

## 8. Model Evaluation for ML

### 8.1 Classification Evaluation
- `torch.argmax(outputs, dim=1)` — predicted class from logits
- `torch.sigmoid(outputs) > 0.5` — binary classification from logits
- `(preds == y).float().mean()` — accuracy
- Using `torchmetrics`
  - `torchmetrics.Accuracy(task='multiclass', num_classes=10)`
  - `torchmetrics.F1Score(task='multiclass', average='weighted')`
  - `torchmetrics.AUROC(task='binary')`
  - `torchmetrics.ConfusionMatrix(task='multiclass', num_classes=10)`
- Using sklearn — `classification_report(y_true.numpy(), y_pred.numpy())`

### 8.2 Regression Evaluation
- `torchmetrics.MeanSquaredError(squared=False)` — RMSE
- `torchmetrics.MeanAbsoluteError()` — MAE
- `torchmetrics.R2Score()` — R²
- Manual — `torch.sqrt(nn.MSELoss()(y_pred, y_true))`

### 8.3 Confusion Matrix and ROC
- `torchmetrics.ConfusionMatrix(task='multiclass', num_classes=n)` — CM
- `torchmetrics.ROC(task='binary')` — ROC curve data
- `torchmetrics.PrecisionRecallCurve(task='binary')` — PR curve data

### 8.4 Out-of-Fold Predictions
- `cross_val_predict` equivalent in PyTorch
- Manual fold splitting with sklearn `StratifiedKFold`
- Collect `model.eval()` predictions for each fold

---

## 9. Regularization for ML

### 9.1 Weight Decay (L2 Regularization)
- `optim.AdamW(model.parameters(), weight_decay=1e-2)` — decoupled
- `optim.SGD(model.parameters(), weight_decay=1e-4)` — L2

### 9.2 Dropout
- `nn.Dropout(p=0.3)` in MLP — randomly zero activations
- Active during `model.train()` — disabled during `model.eval()`

### 9.3 Batch Normalization
- `nn.BatchNorm1d(hidden_dim)` — normalizes per feature per batch
- Reduces internal covariate shift
- Behaves differently in train vs eval — running mean/std in eval

### 9.4 Label Smoothing
- `nn.CrossEntropyLoss(label_smoothing=0.1)` — prevents overconfidence
- Useful for noisy labels in tabular classification

### 9.5 Gradient Clipping
- `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — after backward

---

## 10. Handling Class Imbalance in PyTorch

### 10.1 Weighted Loss
- Compute class weights — `1.0 / class_counts`
- `class_weights = torch.tensor(weights, dtype=torch.float32).to(device)`
- `nn.CrossEntropyLoss(weight=class_weights)` — weighted CE
- `nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)` — binary weighted

### 10.2 WeightedRandomSampler
- `weights = 1.0 / class_counts[y_train]` — per-sample weights
- `sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)`
- `DataLoader(dataset, sampler=sampler, batch_size=64)` — balanced batches

### 10.3 Focal Loss
- Down-weights easy examples, up-weights hard examples
- Custom implementation via `nn.Module`
- `FL = -alpha * (1 - p_t)**gamma * log(p_t)`

---

## 11. Hyperparameter Tuning for PyTorch ML Models

### 11.1 Manual Grid Search
- Nested loops over learning rates, hidden sizes, dropout rates
- Tracking best val loss and best config

### 11.2 Optuna Integration
- `import optuna`
- `def objective(trial):`
  - `lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)`
  - `hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])`
  - `dropout = trial.suggest_float('dropout', 0.1, 0.5)`
- `study.optimize(objective, n_trials=50)`
- `study.best_params` — optimal hyperparameters

### 11.3 Ray Tune Integration
- `from ray import tune`
- Parallel hyperparameter search on multiple GPUs/CPUs
- Distributed ML model tuning

### 11.4 sklearn API via SciKeras / Skorch
- `NeuralNetClassifier(module=MyMLP, max_epochs=100, lr=0.01)`
- `GridSearchCV(NeuralNetClassifier(...), param_grid, cv=5)` — full sklearn CV

---

## 12. Cross-Validation in PyTorch

### 12.1 Manual K-Fold
- `from sklearn.model_selection import StratifiedKFold`
- `for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):`
  - Create DataLoader per fold
  - Train model per fold
  - Record val metrics per fold
- `np.mean(fold_scores)` ± `np.std(fold_scores)` — mean ± std

### 12.2 Skorch for sklearn-style CV
- `NeuralNetClassifier` with `cross_val_score()` from sklearn
- Automated fold training

---

## 13. Feature Importance for PyTorch ML Models

### 13.1 Permutation Importance
- Manual implementation — shuffle one feature, measure accuracy drop
- Using sklearn `permutation_importance` with Skorch wrapper

### 13.2 Integrated Gradients (with Captum)
- `from captum.attr import IntegratedGradients`
- `ig = IntegratedGradients(model)`
- `attributions = ig.attribute(X_test, target=1)` — feature attributions per input
- Importance = mean absolute attribution per feature

### 13.3 SHAP for PyTorch
- `shap.DeepExplainer(model, X_background)` — DeepExplainer
- `shap.GradientExplainer(model, X_background)` — GradientExplainer
- `shap_values = explainer.shap_values(X_test)` — SHAP values
- Feature importance bar chart from SHAP values

### 13.4 Gradient-based Saliency
- `X.requires_grad_(True)`
- `loss = criterion(model(X), y)`
- `loss.backward()`
- `saliency = X.grad.abs()` — gradient magnitude as importance

---

## 14. Ensembling with PyTorch ML Models

### 14.1 Averaging Ensemble
- Train N models with different seeds
- `avg_pred = sum(model_i(X) for model_i in models) / N`

### 14.2 Snapshot Ensemble
- Save model checkpoints at each cosine annealing restart
- Ensemble predictions from all saved checkpoints

### 14.3 Stacking
- First-level PyTorch MLP → OOF predictions
- Second-level sklearn model on OOF predictions + original features

---

## 15. Mixed Data (Tabular) Architectures

### 15.1 Wide and Deep Network
- Wide component — linear model over raw features
- Deep component — MLP over feature embeddings
- Combining logits — `wide_out + deep_out`

### 15.2 Entity Embedding Network
- Separate `nn.Embedding` per categorical column
- Concatenate all embeddings + numerical features
- Pass through MLP
- Used in Rossman Store Sales (Kaggle) — FastAI entity embeddings

### 15.3 Feature Tokenizer Transformer (FT-Transformer)
- Tokenize each feature as an embedding
- Apply Transformer encoder
- Classification head on [CLS] token
- State-of-the-art tabular DL architecture

---

## 16. Time Series ML with PyTorch

### 16.1 Creating Sequence Datasets
- Sliding window — `X[i:i+seq_len]` → `y[i+seq_len]`
- `torch.utils.data.Dataset` for sequence samples
- `DataLoader` with shuffling disabled for ordered sequences

### 16.2 MLP for Tabular Time Series Features
- Feature engineering — lags, rolling stats, time features
- MLP on feature vector per timestep
- Useful for stationary, feature-rich time series

### 16.3 Linear Regression on Sequences
- `nn.Linear(seq_len * n_features, 1)` — flatten sequence
- Simple but effective for short sequences

---

## 17. Model Saving and Deployment for ML

### 17.1 Saving Best Model
- `torch.save(model.state_dict(), f'best_model_fold{fold}.pth')`
- Saving config alongside — `json.dump(config, open('config.json', 'w'))`

### 17.2 Loading for Inference
- `model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))`
- `model.eval()` — set to eval mode
- `with torch.no_grad():` — disable gradients for inference

### 17.3 ONNX Export for Production
- `torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=13)`
- `onnxruntime.InferenceSession('model.onnx')` — fast CPU inference

### 17.4 TorchScript for Production
- `scripted = torch.jit.script(model)` — no Python dependency
- `scripted.save('model_scripted.pt')`

---

## 18. PyTorch Best Practices for ML

### 18.1 Training Best Practices
- Always set `random_state` — `torch.manual_seed(42)`, `np.random.seed(42)`
- Always call `model.train()` before training loop
- Always call `model.eval()` before validation/inference
- Use `torch.no_grad()` for inference — faster and less memory

### 18.2 Common Mistakes
- Forgetting `optimizer.zero_grad()` — gradient accumulation unintentionally
- Using `nn.Softmax` before `nn.CrossEntropyLoss` — double softmax
- Not moving both model and data to the same device
- Calling `.numpy()` on CUDA tensor without `.cpu().detach()` first
- Using `float64` tensors — much slower on GPU

### 18.3 Performance Tips
- `float32` everywhere — `dtype=torch.float32`
- `pin_memory=True` + `num_workers=4` in DataLoader
- `torch.compile(model)` — PyTorch 2.0 speedup
- Use `torch.cuda.amp.autocast()` even for tabular models on GPU

---

## 19. PyTorch ML Interview Topics

### 19.1 Frequently Asked Questions
- How do you implement logistic regression in PyTorch?
- What is the difference between `nn.BCELoss` and `nn.BCEWithLogitsLoss`?
- How do you handle class imbalance in PyTorch?
- How do you implement cross-validation with PyTorch?
- What is `WeightedRandomSampler` and when to use it?
- How do you compute feature importance for a PyTorch model?

### 19.2 ML-Oriented Coding Problems
- Implement a tabular classification MLP with entity embeddings
- Build a training loop with early stopping and model checkpointing
- Implement weighted cross-entropy loss for imbalanced data
- Integrate Optuna for PyTorch hyperparameter tuning
- Implement permutation feature importance for a PyTorch model

---

## 20. Real-World ML Projects with PyTorch

### 20.1 Beginner Projects
- Tabular binary classification (titanic, churn)
- Tabular multi-class classification (iris, wine)
- Simple regression (housing prices)

### 20.2 Intermediate Projects
- Entity embedding model for e-commerce tabular data
- Weighted loss + WeightedRandomSampler for fraud detection
- Optuna-tuned MLP for structured data Kaggle competition

### 20.3 Advanced Projects
- FT-Transformer for tabular data
- Stacking ensemble — PyTorch MLP + sklearn models
- Production deployment via ONNX or TorchScript

---

## 21. PyTorch Roadmap for ML Engineers

### 21.1 Beginner Level
- Tensor operations and data types
- `TensorDataset` and `DataLoader`
- `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`
- Basic training loop — forward, backward, optimizer step
- sklearn preprocessing + PyTorch training

### 21.2 Intermediate Level
- Custom `nn.Module` MLP for tabular data
- `WeightedRandomSampler` for class imbalance
- Early stopping and model checkpointing
- `torchmetrics` for evaluation
- Optuna integration for hyperparameter tuning

### 21.3 Advanced Level
- Entity embedding networks
- FT-Transformer for tabular data
- SHAP / Captum for feature importance
- ONNX export for production ML inference
- `torch.compile()` for GPU-accelerated ML
