# PyTorch for Deep Learning — Complete Syllabus / Roadmap

---

## 1. Introduction to PyTorch for Deep Learning

### 1.1 Why PyTorch Dominates Deep Learning Research and Production
- Dynamic computation graphs — easy debugging and experimentation
- Pythonic API — minimal boilerplate
- Most popular framework in NLP, CV, and scientific ML research
- Native GPU acceleration via CUDA
- Massive ecosystem — HuggingFace, PyTorch Lightning, timm, Detectron2
- PyTorch 2.0 — `torch.compile()` for production-grade performance
- Strong community — arXiv papers ship PyTorch code

### 1.2 Role of PyTorch in the DL Workflow
- Building custom neural network architectures
- Implementing training loops with full control
- Multi-GPU and distributed training
- Mixed precision training
- Transfer learning and fine-tuning pre-trained models
- Model export and production deployment

### 1.3 Installing and Setup
- CUDA-compatible PyTorch installation
- Verifying GPU — `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)`
- Import conventions
  - `import torch`
  - `import torch.nn as nn`
  - `import torch.nn.functional as F`
  - `import torch.optim as optim`
  - `from torch.utils.data import Dataset, DataLoader`
  - `import torchvision`
  - `from torchvision import transforms, models`

---

## 2. Data Pipeline for Deep Learning

### 2.1 Custom Dataset Class
- Subclassing `torch.utils.data.Dataset`
- `__init__` — store paths or data
- `__len__` — return number of samples
- `__getitem__` — load and return one sample
- Image dataset — read with PIL, apply transforms, return tensor
- Text dataset — tokenize, pad, return tensor
- Tabular dataset — return row as float32 tensor

### 2.2 Image DataLoader Pipeline
- `torchvision.datasets.ImageFolder(root, transform=...)` — folder-based dataset
- `torchvision.datasets.CIFAR10(root, train=True, download=True, transform=...)` — built-in
- `DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)`

### 2.3 Image Augmentation
- `transforms.Compose([...])` — transform pipeline
- `transforms.RandomResizedCrop(224)` — random crop + resize
- `transforms.RandomHorizontalFlip(p=0.5)` — horizontal flip
- `transforms.RandomVerticalFlip(p=0.5)` — vertical flip
- `transforms.ColorJitter(brightness=0.2, contrast=0.2)` — color augmentation
- `transforms.RandomRotation(15)` — random rotation
- `transforms.GaussianBlur(kernel_size=3)` — blur
- `transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))` — affine
- `transforms.RandomErasing(p=0.5)` — CutOut
- `transforms.AutoAugment()` — policy-based augmentation
- `transforms.v2` — new API with better composition

### 2.4 Text Tokenization and Padding
- HuggingFace tokenizer — `tokenizer(text, padding='max_length', truncation=True)`
- `torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)` — variable-length
- `DataLoader(dataset, collate_fn=my_collate_fn)` — custom batching

### 2.5 Prefetching and Performance
- `num_workers=4` — parallel data loading
- `pin_memory=True` — pinned CPU memory for faster GPU transfer
- `persistent_workers=True` — keep workers alive
- `prefetch_factor=2` — pre-load batches ahead
- `torch.utils.data.random_split(dataset, [n_train, n_val])` — random split

---

## 3. Building Deep Learning Architectures

### 3.1 Fully Connected Networks (MLP)
- Stacking `nn.Linear + nn.BatchNorm1d + nn.ReLU + nn.Dropout`
- Input layer → hidden layers → output layer
- `nn.Sequential` for simple stacks
- Custom `nn.Module` for complex architectures

### 3.2 Convolutional Neural Networks (CNN)
- Conv block — `nn.Conv2d + nn.BatchNorm2d + nn.ReLU`
- Pooling — `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`
- Fully connected head after feature extraction
- VGG-style — repeated conv blocks
- ResNet-style — skip connections
- Depthwise separable convolutions — `groups=in_channels` + pointwise

### 3.3 Residual Networks (ResNet)
- Skip (residual) connections — `out = F(x) + x`
- Identity shortcut — same dimensions
- Projection shortcut — `nn.Conv2d(in_ch, out_ch, 1)` for dimension change
- Bottleneck block — 1×1 → 3×3 → 1×1 conv
- ResNet-18, 34, 50, 101, 152 variants

### 3.4 Recurrent Networks (LSTM, GRU)
- `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)`
- `output, (h_n, c_n) = lstm(x, (h_0, c_0))`
- `output[:, -1, :]` — last timestep hidden state for classification
- Bidirectional — `bidirectional=True`; hidden_size × 2
- Stacked LSTM — `num_layers > 1`

### 3.5 Transformer from Scratch
- Positional encoding — sinusoidal or learnable
- Multi-head attention — `nn.MultiheadAttention(d_model, n_heads)`
- Feed-forward block — `nn.Linear + nn.ReLU + nn.Linear`
- Add and Norm — residual + `nn.LayerNorm`
- Encoder block — MHA + FFN with residuals
- Decoder block — masked MHA + cross-attention MHA + FFN
- `nn.TransformerEncoderLayer`, `nn.TransformerEncoder` — built-in
- Causal mask — `torch.triu(torch.ones(seq, seq), diagonal=1).bool()`

### 3.6 Attention Mechanisms
- Scaled dot-product attention — `softmax(QKᵀ / √d_k) V`
- `nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)`
- `key_padding_mask` — masking padding tokens
- `attn_mask` — causal masking
- Self-attention vs cross-attention

### 3.7 U-Net Architecture
- Encoder path — downsampling with conv blocks
- Bottleneck — deepest feature representation
- Decoder path — upsampling with `nn.ConvTranspose2d`
- Skip connections from encoder to decoder
- Used for image segmentation

---

## 4. Training Deep Neural Networks

### 4.1 Full Training Loop
```python
model = MyModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    model.train()
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(X)
            loss = criterion(outputs, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
```

### 4.2 Validation Loop
```python
model.eval()
val_loss = 0
correct = 0
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(X)
            loss = criterion(outputs, y)
        val_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
```

### 4.3 Mixed Precision Training
- `torch.cuda.amp.autocast()` — auto cast to float16
- `torch.cuda.amp.GradScaler()` — prevent underflow
- `scaler.scale(loss).backward()` — scaled gradient
- `scaler.unscale_(optimizer)` — unscale before clipping
- `scaler.step(optimizer)` — step if no inf/nan
- `scaler.update()` — update scale factor
- Speed: 1.5–3x faster on Tensor Core GPUs

### 4.4 Gradient Clipping
- `nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — L2 norm clipping
- Placed after `backward()` and before `optimizer.step()`
- Critical for RNN, Transformer training stability

### 4.5 Gradient Accumulation
- `accumulation_steps = 4` — simulate larger batch
- `loss = loss / accumulation_steps`
- `optimizer.zero_grad()` only every N batches
- `optimizer.step()` every N batches

---

## 5. Transfer Learning and Fine-Tuning

### 5.1 Loading Pre-trained Models
- `model = torchvision.models.resnet50(weights='IMAGENET1K_V2')`
- `model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')`
- `model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')`
- `model = timm.create_model('convnext_base', pretrained=True)`

### 5.2 Feature Extraction (Frozen Backbone)
- Freezing all parameters — `for p in model.parameters(): p.requires_grad_(False)`
- Replacing classifier head — `model.fc = nn.Linear(2048, n_classes)`
- Only new head parameters are trained
- Fast and memory-efficient

### 5.3 Fine-Tuning Strategies
- Unfreezing all layers — `model.requires_grad_(True)`
- Differential LR — lower LR for backbone, higher for head
  ```python
  optimizer = optim.AdamW([
      {'params': model.backbone.parameters(), 'lr': 1e-5},
      {'params': model.head.parameters(), 'lr': 1e-3}
  ])
  ```
- Gradual unfreezing — unfreeze one layer group per epoch
- Progressive resizing — train on smaller resolution first

### 5.4 HuggingFace Pre-trained Models
- `from transformers import AutoModel, AutoTokenizer`
- `model = AutoModel.from_pretrained('bert-base-uncased')`
- Configuring for classification — `AutoModelForSequenceClassification`
- `model.config` — model configuration
- `model.save_pretrained('path')` / `model.from_pretrained('path')` — save/load

---

## 6. Computer Vision with PyTorch

### 6.1 Image Classification
- ResNet, EfficientNet, ViT, ConvNeXt
- Custom CNN from scratch
- `AdaptiveAvgPool2d(1)` → flatten → `nn.Linear` — classification head
- `torchvision.models` for standard architectures

### 6.2 Object Detection
- Faster RCNN — `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`
- FCOS — `torchvision.models.detection.fcos_resnet50_fpn(pretrained=True)`
- SSD — `torchvision.models.detection.ssd300_vgg16(pretrained=True)`
- `model(images, targets)` — returns losses in training, predictions in eval
- YOLO — via Ultralytics `ultralytics` package

### 6.3 Image Segmentation
- FCN — `torchvision.models.segmentation.fcn_resnet50(pretrained=True)`
- DeepLab V3 — `torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)`
- U-Net — custom or `segmentation_models_pytorch`
- Loss — `nn.CrossEntropyLoss` for semantic, `nn.BCEWithLogitsLoss` for binary

### 6.4 Data Loading for Vision
- `torchvision.datasets.ImageFolder` — folder-based
- `torchvision.datasets.CocoDetection` — COCO format
- Efficient augmentation — `torchvision.transforms.v2`
- `torchvision.utils.draw_bounding_boxes()` — visualization

### 6.5 Evaluation for Vision
- Classification — top-1, top-5 accuracy
- Detection — mAP, IoU — `torchmetrics.detection.MeanAveragePrecision()`
- Segmentation — mIoU, pixel accuracy — `torchmetrics.JaccardIndex()`

---

## 7. Natural Language Processing with PyTorch

### 7.1 Text Tokenization
- HuggingFace tokenizer — `tokenizer(texts, padding=True, truncation=True, return_tensors='pt')`
- `input_ids`, `attention_mask` — tokenizer output tensors
- Vocabulary building with `torchtext`

### 7.2 Text Classification
- BERT-based — `AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n)`
- LSTM-based — embedding + LSTM + linear head
- Training with `nn.CrossEntropyLoss` and `AdamW`

### 7.3 Language Modeling
- Causal LM — `AutoModelForCausalLM`
- Masked LM — `AutoModelForMaskedLM`
- `nn.CrossEntropyLoss` on shifted token sequences

### 7.4 Sequence Labeling
- Token classification — `AutoModelForTokenClassification`
- CRF layer on top — `pytorch-crf`
- Named entity recognition (NER)

### 7.5 Attention Visualization
- `output = model(**inputs, output_attentions=True)`
- `attention_weights = output.attentions` — tuple of per-layer attention
- `attention_weights[layer][batch, head, seq, seq]` — shape

---

## 8. Generative Models

### 8.1 Variational Autoencoder (VAE)
- Encoder — maps input to `mu` and `log_var`
- Reparameterization — `z = mu + std * torch.randn_like(std)`
- Decoder — maps `z` to reconstructed input
- ELBO loss — reconstruction loss + KL divergence
  - `recon_loss = nn.MSELoss()(recon_x, x)`
  - `kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())`

### 8.2 Generative Adversarial Networks (GAN)
- Generator — maps noise `z` to synthetic data
- Discriminator — classifies real vs fake
- GAN training loop
  - Discriminator step — real loss + fake loss
  - Generator step — fool discriminator
- Loss — `nn.BCEWithLogitsLoss()` for vanilla GAN
- DCGAN — convolutional GAN for images

### 8.3 Diffusion Models
- Forward diffusion — adding Gaussian noise over T steps
- Reverse diffusion — learned denoising
- U-Net denoiser backbone with time step embedding
- `torch.randn_like(x)` — sample noise
- Noise schedule — `beta` values per timestep
- DDPM, DDIM sampling

---

## 9. Sequence and Time Series DL

### 9.1 LSTM for Sequence Modeling
- Sliding window dataset
- `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`
- Multi-step output — `nn.Linear(hidden_size, forecast_horizon)`
- Teacher forcing for seq2seq training

### 9.2 Temporal Convolutional Network (TCN)
- Causal convolutions — `padding = (kernel_size - 1) * dilation`
- Dilated causal convolutions — exponentially increasing receptive field
- Residual blocks

### 9.3 Transformer for Time Series
- Positional encoding for time steps
- `nn.TransformerEncoder` for sequence encoding
- Patch embedding for long sequences
- Informer, Autoformer, PatchTST architectures

### 9.4 Evaluation for Time Series
- RMSE — `torch.sqrt(nn.MSELoss()(y_pred, y_true))`
- MAE — `nn.L1Loss()(y_pred, y_true)`
- MAPE — custom implementation

---

## 10. Advanced Training Techniques

### 10.1 Mixed Precision Training
- `torch.cuda.amp.autocast()` + `GradScaler()`
- 1.5–3x speedup on modern GPUs
- Automatic dtype selection — `float16` for most ops, `float32` for stability

### 10.2 Gradient Checkpointing
- `torch.utils.checkpoint.checkpoint(function, *inputs)` — trade compute for memory
- Recomputes activations during backward instead of storing
- Enables training larger models on same GPU

### 10.3 Model Parallelism
- `nn.DataParallel(model)` — simple multi-GPU
- `torch.nn.parallel.DistributedDataParallel` — recommended for multi-GPU
- Pipeline parallelism — split model across GPUs

### 10.4 Knowledge Distillation
- Teacher model — large pre-trained model
- Student model — smaller model
- Distillation loss — `KLDivLoss(F.log_softmax(student_logits / T), F.softmax(teacher_logits / T))`
- Temperature `T` — soften probability distributions

### 10.5 Label Smoothing
- `nn.CrossEntropyLoss(label_smoothing=0.1)` — prevent overconfident predictions
- Soft targets instead of hard one-hot

### 10.6 Stochastic Depth / Drop Path
- Randomly skip residual blocks during training
- `drop_path_rate` — probability of dropping each block
- Regularization for deep networks

### 10.7 Exponential Moving Average (EMA)
- Maintain EMA of model weights
- `ema_model.param = alpha * ema_model.param + (1-alpha) * model.param`
- Use EMA weights for evaluation — smoother performance
- `timm.utils.ModelEmaV2(model, decay=0.9998)`

---

## 11. Regularization for Deep Learning

### 11.1 Dropout
- `nn.Dropout(p=0.5)` — 1D
- `nn.Dropout2d(p=0.3)` — spatial dropout for CNN
- Active only in `model.train()` mode

### 11.2 Batch Normalization
- `nn.BatchNorm1d`, `nn.BatchNorm2d`
- `running_mean`, `running_var` — used in `model.eval()`
- `momentum`, `eps`, `affine` parameters

### 11.3 Layer Normalization
- `nn.LayerNorm(normalized_shape)` — normalizes along feature dim
- Preferred for Transformers — not dependent on batch size

### 11.4 Weight Decay
- `optim.AdamW(params, weight_decay=0.01)` — decoupled WD
- `optim.SGD(params, weight_decay=1e-4)` — L2 penalty

### 11.5 Data Augmentation as Regularization
- Cutout — random erase patches — `transforms.RandomErasing()`
- MixUp — convex combination of two samples and labels
- CutMix — replace patch from one image with another
- Label smoothing with MixUp/CutMix

### 11.6 Early Stopping
- Monitor val loss per epoch
- Patience counter
- `torch.save(model.state_dict(), 'best.pth')` — save on improvement
- Restore best weights when patience exceeded

---

## 12. Hyperparameter Tuning for DL

### 12.1 Optuna for PyTorch
- `import optuna`
- `def objective(trial):`
  - `lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)`
  - `dropout = trial.suggest_float('dropout', 0.1, 0.5)`
  - `n_layers = trial.suggest_int('n_layers', 2, 6)`
  - `hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])`
- `study.optimize(objective, n_trials=100)`
- `optuna.visualization.plot_optimization_history(study)`

### 12.2 Learning Rate Finder
- Train for one epoch with LR exponentially increasing
- `torch-lr-finder` library
- Plot loss vs LR — pick LR before steepest descent

### 12.3 1-Cycle Policy
- `lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs)`
- Warm-up to max LR → annealing down
- Often reaches better optima faster

### 12.4 Weights & Biases (W&B) Integration
- `import wandb; wandb.init(project='my-dl-project', config=config)`
- `wandb.log({'train_loss': loss, 'val_acc': acc})`
- `wandb.watch(model)` — log gradients and weights
- Hyperparameter sweep — `wandb.sweep(sweep_config, function=train_fn)`

---

## 13. Multi-GPU Training

### 13.1 DataParallel (Simple Multi-GPU)
- `model = nn.DataParallel(model)` — wraps model
- Automatically splits batch across GPUs
- Gathers outputs on GPU 0
- Simple but suboptimal — GIL and GPU memory imbalance

### 13.2 DistributedDataParallel (Recommended)
- One process per GPU
- `torch.distributed.init_process_group('nccl')`
- `model = DDP(model, device_ids=[local_rank])`
- `DistributedSampler(dataset)` — ensures no data overlap
- Launch with `torchrun --nproc_per_node=4 train.py`

### 13.3 PyTorch Lightning for Multi-GPU
- `Trainer(accelerator='gpu', devices=4, strategy='ddp')`
- Abstracts DDP boilerplate
- Automatic mixed precision — `precision=16`

---

## 14. Model Evaluation for DL

### 14.1 Classification Evaluation
- `torchmetrics.Accuracy(task='multiclass', num_classes=n)` — accuracy
- `torchmetrics.F1Score(task='multiclass', average='macro')` — macro F1
- `torchmetrics.AUROC(task='binary')` — AUC-ROC
- `torchmetrics.ConfusionMatrix(task='multiclass', num_classes=n)` — CM
- Top-5 accuracy — `torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5)`

### 14.2 Detection and Segmentation Evaluation
- `torchmetrics.detection.MeanAveragePrecision()` — mAP
- `torchmetrics.JaccardIndex(task='multiclass', num_classes=n)` — mIoU
- `torchmetrics.Dice()` — Dice coefficient

### 14.3 Generation Evaluation
- `torchmetrics.image.FrechetInceptionDistance()` — FID for GANs
- `torchmetrics.text.BLEUScore()` — BLEU for NLP
- Perplexity — `torch.exp(cross_entropy_loss)`

---

## 15. Model Compression and Efficiency

### 15.1 Quantization
- `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` — dynamic quantization
- Post-training static quantization
- Quantization-aware training (QAT)
- 4x smaller model, 2-4x faster CPU inference

### 15.2 Pruning
- `torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=0.5)` — unstructured
- `torch.nn.utils.prune.random_structured(module, name='weight', amount=0.5, dim=0)` — structured
- `prune.remove(module, 'weight')` — make pruning permanent

### 15.3 Knowledge Distillation
- Soft targets from teacher model
- Distillation loss weight vs task loss weight
- Response-based, feature-based, relation-based distillation

### 15.4 ONNX Export
- `torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=17)`
- `onnxruntime.InferenceSession('model.onnx')` — fast inference
- ONNX quantization for further compression

---

## 16. Model Saving, Loading, and Deployment

### 16.1 Saving Training Checkpoints
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'val_loss': val_loss,
    'config': config,
}, f'checkpoint_epoch{epoch}.pth')
```

### 16.2 Loading Best Model
- `checkpoint = torch.load('best.pth', map_location=device)`
- `model.load_state_dict(checkpoint['model_state_dict'])`
- `model.eval()` — always after loading

### 16.3 TorchScript for Deployment
- `scripted = torch.jit.script(model)` — scripted model
- `traced = torch.jit.trace(model, example_input)` — traced model
- `scripted.save('model_scripted.pt')` — no Python dependency
- `torch.jit.load('model_scripted.pt')` — load anywhere

### 16.4 ONNX Export
- `torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=17, input_names=['input'], output_names=['output'])`
- Dynamic axes for variable batch size / sequence length
- Test with `onnxruntime`

### 16.5 HuggingFace Model Saving
- `model.save_pretrained('output_dir')` — saves config + weights
- `tokenizer.save_pretrained('output_dir')` — saves tokenizer
- `AutoModel.from_pretrained('output_dir')` — reload

---

## 17. Advanced PyTorch Features for DL

### 17.1 torch.compile (PyTorch 2.0+)
- `model = torch.compile(model)` — compile for speed
- `mode='reduce-overhead'` — smaller overhead
- `mode='max-autotune'` — most aggressive tuning
- Up to 2x speedup on A100

### 17.2 FlashAttention
- Faster and memory-efficient attention
- `nn.MultiheadAttention` uses FlashAttention automatically in PyTorch 2.0+
- Manual — via `xformers.ops.memory_efficient_attention()`

### 17.3 Custom CUDA Kernels
- `torch.autograd.Function` — custom backward
- C++ extension — `torch.utils.cpp_extension`
- CUDA extension — `.cu` files

### 17.4 Hooks
- `module.register_forward_hook(fn)` — runs after forward
- `module.register_backward_hook(fn)` — runs after backward
- Feature extraction — capturing intermediate activations
- Gradient monitoring via hooks

---

## 18. Debugging and Profiling

### 18.1 Debugging NaN and Inf
- `torch.autograd.set_detect_anomaly(True)` — detect NaN in backward
- `torch.isnan(tensor).any()` — check NaN
- `torch.isinf(tensor).any()` — check Inf
- Gradient clipping to prevent explosion
- `print(tensor.min(), tensor.max())` — range check

### 18.2 PyTorch Profiler
- `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])`
- `prof.key_averages().table(sort_by='cuda_time_total')` — profile table
- TensorBoard integration — `prof.export_chrome_trace('trace.json')`

### 18.3 Memory Profiling
- `torch.cuda.memory_summary()` — memory breakdown
- `torch.cuda.max_memory_allocated()` — peak usage
- `del tensor; torch.cuda.empty_cache()` — free memory

---

## 19. PyTorch Lightning for DL

### 19.1 LightningModule
- `class MyModel(pl.LightningModule):` — subclassing
- `training_step(batch, batch_idx)` — training logic
- `validation_step(batch, batch_idx)` — validation logic
- `configure_optimizers()` — return optimizer + scheduler
- `self.log('train_loss', loss)` — automatic logging

### 19.2 Trainer
- `trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=2, precision=16)`
- `trainer.fit(model, train_loader, val_loader)` — start training
- `trainer.test(model, test_loader)` — test evaluation
- `trainer.predict(model, loader)` — inference

### 19.3 Callbacks
- `ModelCheckpoint(monitor='val_loss', save_best_only=True)`
- `EarlyStopping(monitor='val_loss', patience=5)`
- `LearningRateMonitor(logging_interval='step')`
- Custom callbacks — `on_epoch_end()`, `on_batch_end()`

---

## 20. DL-Specific Interview Topics

### 20.1 Frequently Asked Questions
- What is the difference between `model.train()` and `model.eval()`?
- How does `torch.no_grad()` differ from `torch.inference_mode()`?
- What is gradient clipping and why is it used?
- How does mixed precision training work in PyTorch?
- What is the difference between `DataParallel` and `DistributedDataParallel`?
- How does `torch.compile()` improve training speed?
- What is the reparameterization trick in VAE?

### 20.2 DL-Oriented Coding Problems
- Implement a ResNet block from scratch
- Write a full training loop with AMP, gradient clipping, and early stopping
- Implement multi-head self-attention from scratch
- Implement a custom loss function (focal loss, dice loss)
- Build a GAN training loop
- Export a PyTorch model to ONNX

---

## 21. Real-World DL Projects with PyTorch

### 21.1 Beginner Projects
- MNIST CNN from scratch
- CIFAR-10 ResNet classification
- IMDB sentiment classification with LSTM

### 21.2 Intermediate Projects
- Transfer learning with ResNet/EfficientNet on custom dataset
- BERT fine-tuning for text classification
- U-Net semantic segmentation
- VAE for image generation

### 21.3 Advanced Projects
- DCGAN / StyleGAN for image synthesis
- Transformer from scratch for sequence tasks
- DETR for object detection
- Diffusion model for image generation
- LLM fine-tuning with LoRA

---

## 22. PyTorch DL Ecosystem Tools

### 22.1 Training Frameworks
- PyTorch Lightning — `lightning.ai`
- fast.ai — high-level API on PyTorch
- HuggingFace Accelerate — multi-GPU/TPU training

### 22.2 Model Hubs
- `timm` — PyTorch image models (600+ architectures)
- HuggingFace Hub — 500k+ models
- `torchvision.models` — standard vision models

### 22.3 Experiment Tracking
- Weights & Biases (W&B) — `wandb`
- MLflow — `mlflow.pytorch`
- TensorBoard — `torch.utils.tensorboard.SummaryWriter`
- CometML

### 22.4 Interpretability
- Captum — `IntegratedGradients`, `GradCAM`, `SHAP`
- GradCAM — `pytorch-grad-cam`

---

## 23. PyTorch DL Roadmap

### 23.1 Beginner Level
- Tensors, autograd, `nn.Module`
- Custom `Dataset` and `DataLoader`
- Training loop — forward, backward, optimizer step
- CNN for image classification
- LSTM for sequence classification

### 23.2 Intermediate Level
- Transfer learning with pre-trained models
- Mixed precision training with AMP
- ResNet skip connections
- Transformer and attention mechanisms
- HuggingFace integration
- Model checkpointing and early stopping

### 23.3 Advanced Level
- DistributedDataParallel multi-GPU training
- GAN and VAE implementation
- Diffusion models
- Custom CUDA ops and torch.compile
- Model compression — quantization, pruning, ONNX
- PyTorch Lightning for production training
