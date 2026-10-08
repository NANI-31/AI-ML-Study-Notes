# PyTorch Roadmap / Syllabus

---

## 1. Introduction to PyTorch

### 1.1 What is PyTorch?
- Definition of PyTorch
- Open-source deep learning framework by Meta AI
- Python-first, dynamic computation graph (eager execution)
- Advantages of PyTorch
  - Intuitive Pythonic API
  - Dynamic computation graphs — easy debugging
  - Strong research community adoption
  - Native GPU support via CUDA
  - Seamless NumPy integration
  - Rich ecosystem — torchvision, torchaudio, torchtext, HuggingFace
- PyTorch vs TensorFlow — key differences
- PyTorch for research vs production

### 1.2 Installing PyTorch
- Installing CPU version
- Installing CUDA GPU version — specifying CUDA version
- Installing with pip
- Installing with conda
- Verifying installation
  - `import torch`
  - `torch.__version__`
  - `torch.cuda.is_available()`
  - `torch.cuda.get_device_name(0)`

### 1.3 PyTorch Core Components
- `torch` — main package
- `torch.Tensor` — multi-dimensional array
- `torch.nn` — neural network modules
- `torch.optim` — optimization algorithms
- `torch.utils.data` — dataset and dataloader utilities
- `torch.autograd` — automatic differentiation
- `torchvision` — vision models and datasets
- `torchaudio` — audio models and datasets
- `torchtext` — NLP utilities

---

## 2. Tensors

### 2.1 Creating Tensors
- `torch.tensor([1, 2, 3])` — from Python list
- `torch.tensor([[1, 2], [3, 4]])` — 2D tensor
- `torch.zeros(3, 4)` — zeros tensor
- `torch.ones(3, 4)` — ones tensor
- `torch.empty(3, 4)` — uninitialized tensor
- `torch.full((3, 4), fill_value=5)` — filled tensor
- `torch.eye(4)` — identity matrix
- `torch.arange(0, 10, 2)` — range tensor
- `torch.linspace(0, 1, 10)` — linear spacing
- `torch.rand(3, 4)` — uniform random [0, 1)
- `torch.randn(3, 4)` — standard normal
- `torch.randint(0, 10, (3, 4))` — random integers

### 2.2 Tensor from NumPy and Back
- `torch.from_numpy(np_array)` — NumPy to tensor (shared memory)
- `tensor.numpy()` — tensor to NumPy (CPU only)
- `tensor.detach().numpy()` — detach from graph, then to NumPy
- `tensor.cpu().detach().numpy()` — from GPU to NumPy

### 2.3 Tensor Attributes
- `tensor.shape` — shape as `torch.Size`
- `tensor.dtype` — data type
- `tensor.device` — CPU or CUDA device
- `tensor.ndim` — number of dimensions
- `tensor.numel()` — total number of elements
- `tensor.is_cuda` — whether on GPU
- `tensor.requires_grad` — gradient tracking flag

### 2.4 Tensor Data Types
- `torch.float32` (default) — 32-bit float
- `torch.float64` — 64-bit float (double)
- `torch.float16` — half precision
- `torch.bfloat16` — brain float
- `torch.int32`, `torch.int64` — integers
- `torch.bool` — boolean
- `torch.complex64`, `torch.complex128`
- Casting — `tensor.float()`, `tensor.int()`, `tensor.to(torch.float32)`

### 2.5 Moving Tensors to Device
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `tensor.to(device)` — move to device
- `tensor.cuda()` — move to GPU
- `tensor.cpu()` — move to CPU
- `torch.tensor(data, device='cuda')` — create directly on GPU

---

## 3. Tensor Operations

### 3.1 Arithmetic Operations
- Element-wise addition — `a + b`, `torch.add(a, b)`
- Element-wise subtraction — `a - b`, `torch.sub(a, b)`
- Element-wise multiplication — `a * b`, `torch.mul(a, b)`
- Element-wise division — `a / b`, `torch.div(a, b)`
- Power — `a ** 2`, `torch.pow(a, 2)`
- Square root — `torch.sqrt(a)`
- Absolute value — `torch.abs(a)`
- In-place operations — `a.add_(b)`, `a.mul_(b)` (trailing underscore)

### 3.2 Matrix Operations
- Matrix multiplication — `torch.mm(A, B)` (2D)
- Batch matrix multiplication — `torch.bmm(A, B)` (3D)
- General matrix multiplication — `torch.matmul(A, B)` (any dim)
- `@` operator — `A @ B` — shorthand for matmul
- Dot product — `torch.dot(a, b)` (1D)
- Outer product — `torch.outer(a, b)`
- Matrix transpose — `tensor.T`, `tensor.transpose(0, 1)`

### 3.3 Reduction Operations
- `tensor.sum()`, `tensor.sum(dim=0)` — sum
- `tensor.mean()`, `tensor.mean(dim=1)` — mean
- `tensor.std()`, `tensor.var()` — std and variance
- `tensor.min()`, `tensor.max()` — min and max
- `tensor.argmin()`, `tensor.argmax()` — index of min/max
- `tensor.prod()` — product of all elements
- `torch.cumsum(tensor, dim=0)` — cumulative sum

### 3.4 Comparison Operations
- `torch.eq(a, b)` — element-wise equal
- `torch.ne(a, b)` — not equal
- `torch.lt(a, b)`, `torch.gt(a, b)` — less/greater than
- `torch.le(a, b)`, `torch.ge(a, b)` — less/greater or equal
- `a == b`, `a > b` — Python comparison operators

### 3.5 Broadcasting
- Same rules as NumPy broadcasting
- Shape compatibility check — trailing dimensions must match or be 1
- Scalar broadcasting
- Adding bias to batch — `(N, C, H, W) + (C, 1, 1)`

---

## 4. Tensor Indexing and Slicing

### 4.1 Basic Indexing
- `tensor[0]` — first element or row
- `tensor[0, 1]` — element at row 0, col 1
- `tensor[-1]` — last element
- Negative indexing

### 4.2 Slicing
- `tensor[0:3]` — rows 0 to 2
- `tensor[:, 1]` — column 1
- `tensor[1:3, 0:2]` — submatrix
- Step slicing — `tensor[::2]`

### 4.3 Boolean Indexing
- `tensor[tensor > 0]` — filter positive values
- `torch.where(condition, x, y)` — conditional selection
- `tensor.masked_fill(mask, value)` — in-place masking

### 4.4 Fancy Indexing
- `tensor[[0, 2, 4]]` — select specific rows
- `torch.index_select(tensor, dim, index_tensor)`
- `torch.gather(tensor, dim, index)` — gather along dim

---

## 5. Tensor Shape Manipulation

### 5.1 Reshaping
- `tensor.reshape(new_shape)` — reshape (may copy)
- `tensor.view(new_shape)` — reshape (shared memory, contiguous required)
- `tensor.flatten()` — flatten to 1D
- `torch.flatten(tensor, start_dim=1)` — flatten from dim 1

### 5.2 Adding and Removing Dimensions
- `tensor.unsqueeze(dim)` — add dimension
- `tensor.squeeze()` — remove all size-1 dimensions
- `tensor.squeeze(dim)` — remove specific size-1 dimension

### 5.3 Permuting and Transposing
- `tensor.permute(2, 0, 1)` — permute dimensions
- `tensor.transpose(0, 1)` — swap two dimensions
- `tensor.T` — transpose (2D only)
- `tensor.contiguous()` — make contiguous in memory

### 5.4 Concatenation and Stacking
- `torch.cat([t1, t2], dim=0)` — concatenate along dim
- `torch.stack([t1, t2], dim=0)` — stack (new dimension)
- `torch.vstack([t1, t2])` — vertical stack
- `torch.hstack([t1, t2])` — horizontal stack

### 5.5 Splitting
- `torch.split(tensor, split_size, dim=0)` — split into chunks
- `torch.chunk(tensor, chunks, dim=0)` — split into N chunks
- `torch.unbind(tensor, dim=0)` — unbind along dim

---

## 6. Autograd — Automatic Differentiation

### 6.1 Gradient Basics
- `requires_grad=True` — enable gradient tracking
- `tensor.grad` — accumulated gradient
- `tensor.backward()` — compute gradients
- `tensor.grad_fn` — gradient function (computation graph node)
- `tensor.is_leaf` — whether leaf node

### 6.2 Computing Gradients
- Scalar loss — `loss.backward()`
- Non-scalar — `loss.backward(gradient=torch.ones_like(loss))`
- Accessing gradients — `tensor.grad`

### 6.3 Gradient Control
- `torch.no_grad()` — disable gradient computation (inference)
  - `with torch.no_grad():`
- `tensor.detach()` — detach from computation graph
- `torch.inference_mode()` — more efficient than `no_grad()`
- `optimizer.zero_grad()` — clearing accumulated gradients

### 6.4 Gradient Accumulation
- Accumulating gradients over multiple batches
- `loss.backward()` multiple times before `optimizer.step()`
- Useful for large batches that don't fit in memory

### 6.5 Custom Autograd Functions
- `class MyFunction(torch.autograd.Function):`
  - `@staticmethod def forward(ctx, input):`
  - `@staticmethod def backward(ctx, grad_output):`
- `ctx.save_for_backward(tensors)` — saving tensors for backward

---

## 7. Neural Networks — torch.nn

### 7.1 nn.Module
- `class MyModel(nn.Module):` — subclassing
- `def __init__(self):` — defining layers
  - `super().__init__()`
  - `self.linear = nn.Linear(in, out)`
- `def forward(self, x):` — defining forward pass
- `model.parameters()` — all learnable parameters
- `model.named_parameters()` — named parameters
- `model.modules()` — all modules
- `model.children()` — direct children

### 7.2 Linear Layers
- `nn.Linear(in_features, out_features, bias=True)` — fully connected
- `nn.Bilinear(in1_features, in2_features, out_features)` — bilinear layer
- `nn.LazyLinear(out_features)` — infer in_features automatically

### 7.3 Activation Functions
- `nn.ReLU()` — rectified linear unit
- `nn.LeakyReLU(negative_slope=0.01)`
- `nn.PReLU()` — learnable negative slope
- `nn.ELU(alpha=1.0)` — exponential linear unit
- `nn.SELU()` — scaled ELU
- `nn.Sigmoid()` — sigmoid
- `nn.Tanh()` — hyperbolic tangent
- `nn.Softmax(dim=1)` — softmax
- `nn.LogSoftmax(dim=1)` — log softmax
- `nn.GELU()` — Gaussian error linear unit
- `nn.Mish()` — Mish activation
- `nn.SiLU()` — Swish
- `F.relu(x)` — functional API alternatives

### 7.4 Loss Functions
- `nn.MSELoss()` — mean squared error
- `nn.L1Loss()` — mean absolute error
- `nn.BCELoss()` — binary cross-entropy (requires sigmoid)
- `nn.BCEWithLogitsLoss()` — BCE with built-in sigmoid (numerically stable)
- `nn.CrossEntropyLoss()` — multi-class cross-entropy (includes softmax)
- `nn.NLLLoss()` — negative log likelihood
- `nn.HuberLoss()` — smooth L1 / Huber loss
- `nn.KLDivLoss()` — KL divergence
- `nn.CosineEmbeddingLoss()`
- `nn.CTCLoss()` — connectionist temporal classification

### 7.5 Normalization Layers
- `nn.BatchNorm1d(num_features)` — batch norm for 1D
- `nn.BatchNorm2d(num_features)` — batch norm for 2D (CNN)
- `nn.LayerNorm(normalized_shape)` — layer norm (Transformers)
- `nn.GroupNorm(num_groups, num_channels)` — group norm
- `nn.InstanceNorm2d(num_features)` — instance norm
- `nn.RMSNorm(normalized_shape)` — RMS norm (LLaMA)

### 7.6 Dropout Layers
- `nn.Dropout(p=0.5)` — 1D dropout
- `nn.Dropout2d(p=0.5)` — 2D spatial dropout
- `nn.AlphaDropout(p=0.5)` — self-normalizing dropout

### 7.7 Sequential and Container Modules
- `nn.Sequential(*layers)` — sequential stack
- `nn.ModuleList([layers])` — list of modules
- `nn.ModuleDict({'name': layer})` — dict of modules
- `nn.ParameterList` — list of parameters
- `nn.ParameterDict` — dict of parameters

---

## 8. Convolutional Layers

### 8.1 Conv2D
- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`
- `padding='same'` — same spatial size output
- `dilation` — dilated convolution
- `groups` — grouped convolution (depthwise when groups=in_channels)
- `bias=True` — learnable bias

### 8.2 Other Convolution Types
- `nn.Conv1d` — 1D convolution (sequences, audio)
- `nn.Conv3d` — 3D convolution (video, volumetric)
- `nn.ConvTranspose2d` — transposed convolution (upsampling)
- `nn.DepthwiseSeparableConv` — via `groups=in_channels` + pointwise

### 8.3 Pooling Layers
- `nn.MaxPool2d(kernel_size, stride)` — max pooling
- `nn.AvgPool2d(kernel_size, stride)` — average pooling
- `nn.AdaptiveMaxPool2d(output_size)` — adaptive max pooling
- `nn.AdaptiveAvgPool2d(output_size)` — adaptive average pooling
- `nn.GlobalAveragePooling` — via `AdaptiveAvgPool2d(1)`

### 8.4 Padding Layers
- `nn.ZeroPad2d(padding)` — zero padding
- `nn.ReflectionPad2d(padding)` — reflection padding
- `nn.ReplicationPad2d(padding)` — replication padding

---

## 9. Recurrent Layers

### 9.1 RNN
- `nn.RNN(input_size, hidden_size, num_layers, batch_first=True)`
- `output, h_n = rnn(input, h_0)` — forward pass
- `nonlinearity` — `'tanh'` or `'relu'`
- `bidirectional` — bidirectional RNN

### 9.2 LSTM
- `nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)`
- `output, (h_n, c_n) = lstm(input, (h_0, c_0))` — forward pass
- `dropout` — dropout between layers
- `bidirectional` — bidirectional LSTM

### 9.3 GRU
- `nn.GRU(input_size, hidden_size, num_layers, batch_first=True)`
- `output, h_n = gru(input, h_0)` — forward pass
- `bidirectional` — bidirectional GRU

### 9.4 Packed Sequences
- `nn.utils.rnn.pack_padded_sequence()` — packing variable-length sequences
- `nn.utils.rnn.pad_packed_sequence()` — unpacking
- `enforce_sorted=False` — handle unsorted sequences

---

## 10. Transformer Layers

### 10.1 Self-Attention
- `nn.MultiheadAttention(embed_dim, num_heads)`
- `attn_output, attn_weights = mha(query, key, value)`
- `key_padding_mask` — masking padded positions
- `attn_mask` — causal mask for autoregressive models

### 10.2 Transformer Modules
- `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)`
- `nn.TransformerEncoder(encoder_layer, num_layers)`
- `nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)`
- `nn.TransformerDecoder(decoder_layer, num_layers)`
- `nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)`

### 10.3 Embedding Layers
- `nn.Embedding(num_embeddings, embedding_dim)` — lookup table
- `padding_idx` — index ignored in gradient update
- `nn.EmbeddingBag` — mean/sum/max of embeddings
- Positional encoding — custom `nn.Module`

---

## 11. Optimizers

### 11.1 Core Optimizers
- `optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)`
- `optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`
- `optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)` — decoupled weight decay
- `optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)`
- `optim.Adagrad(model.parameters(), lr=1e-2)`
- `optim.Adadelta(model.parameters())`

### 11.2 Optimizer API
- `optimizer.zero_grad()` — clear gradients
- `optimizer.step()` — update parameters
- `optimizer.state_dict()` — saving optimizer state
- `optimizer.load_state_dict(state)` — loading optimizer state
- `optimizer.param_groups` — access/modify parameter groups
- Differential learning rates — multiple param groups

### 11.3 Learning Rate Schedulers
- `lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)` — step decay
- `lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)`
- `lr_scheduler.ExponentialLR(optimizer, gamma=0.95)` — exponential decay
- `lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)` — cosine annealing
- `lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)` — adaptive
- `lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch, epochs)` — 1-cycle
- `lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)` — warm restarts
- `scheduler.step()` — step the scheduler

---

## 12. Dataset and DataLoader

### 12.1 Dataset Class
- `torch.utils.data.Dataset` — base class
- `__len__(self)` — return dataset size
- `__getitem__(self, idx)` — return sample at index
- Custom dataset implementation
  - Image dataset
  - Text dataset
  - Tabular dataset

### 12.2 Built-in Datasets (torchvision)
- `torchvision.datasets.MNIST`
- `torchvision.datasets.CIFAR10`, `CIFAR100`
- `torchvision.datasets.ImageNet`
- `torchvision.datasets.ImageFolder` — folder structure dataset
- `torchvision.datasets.FashionMNIST`
- `root`, `train`, `download`, `transform` parameters

### 12.3 DataLoader
- `DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)`
- `drop_last=True` — drop last incomplete batch
- `pin_memory=True` — faster CPU to GPU transfer
- `collate_fn` — custom batch assembly
- `sampler` — custom sampling strategy
- `persistent_workers=True` — keep workers alive between epochs

### 12.4 Transforms (torchvision.transforms)
- `transforms.Compose([transforms_list])` — pipeline
- `transforms.ToTensor()` — PIL/NumPy to tensor, scale to [0,1]
- `transforms.Normalize(mean, std)` — normalize channels
- `transforms.Resize((H, W))` — resize image
- `transforms.CenterCrop(size)` — center crop
- `transforms.RandomCrop(size)` — random crop
- `transforms.RandomHorizontalFlip(p=0.5)` — random flip
- `transforms.RandomRotation(degrees)` — random rotation
- `transforms.ColorJitter(brightness, contrast, saturation, hue)` — color augmentation
- `transforms.RandomGrayscale(p=0.1)` — random grayscale
- `transforms.GaussianBlur(kernel_size)` — Gaussian blur
- `transforms.v2` — new transforms API (torchvision v2)

### 12.5 Samplers
- `SequentialSampler` — ordered sampling
- `RandomSampler` — random sampling
- `WeightedRandomSampler` — weighted sampling for class imbalance
- `SubsetRandomSampler` — sample from subset of indices
- `BatchSampler` — batch-level sampling

---

## 13. Training Loop

### 13.1 Standard Training Loop
- Setting model to train mode — `model.train()`
- Iterating over DataLoader — `for batch in dataloader:`
- Moving data to device — `X, y = X.to(device), y.to(device)`
- Forward pass — `outputs = model(X)`
- Compute loss — `loss = criterion(outputs, y)`
- Zero gradients — `optimizer.zero_grad()`
- Backward pass — `loss.backward()`
- Gradient clipping — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Update weights — `optimizer.step()`
- Step scheduler — `scheduler.step()`
- Logging loss

### 13.2 Validation Loop
- Setting model to eval mode — `model.eval()`
- `with torch.no_grad():` — disable gradients
- Forward pass only — `outputs = model(X_val)`
- Compute validation loss and metrics
- Comparing train vs val loss

### 13.3 Epoch Loop
- Outer loop over epochs
- Train loop per epoch
- Validation loop per epoch
- Logging per epoch
- Early stopping logic
- Checkpoint saving logic

### 13.4 Gradient Clipping
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — norm clipping
- `torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)` — value clipping

---

## 14. Model Saving and Loading

### 14.1 Saving and Loading State Dict (Recommended)
- `torch.save(model.state_dict(), 'model.pth')` — saving weights
- `model.load_state_dict(torch.load('model.pth'))` — loading weights
- `model.eval()` — set to eval mode after loading

### 14.2 Saving Full Model
- `torch.save(model, 'model_full.pth')` — full model
- `model = torch.load('model_full.pth')` — loading full model
- Less portable — tied to exact class definition

### 14.3 Checkpoint (Training State)
- Saving full checkpoint
  ```python
  torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
  }, 'checkpoint.pth')
  ```
- Loading checkpoint
  ```python
  checkpoint = torch.load('checkpoint.pth')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  ```

### 14.4 map_location
- `torch.load('model.pth', map_location='cpu')` — load GPU model on CPU
- `torch.load('model.pth', map_location=torch.device('cuda:0'))`

---

## 15. GPU Acceleration

### 15.1 CUDA Basics
- `torch.cuda.is_available()` — check GPU availability
- `torch.cuda.device_count()` — number of GPUs
- `torch.cuda.get_device_name(0)` — GPU name
- `torch.cuda.current_device()` — current device index

### 15.2 Moving to GPU
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `tensor.to(device)` — tensor to device
- `model.to(device)` — model to device
- `tensor.cuda()` / `tensor.cpu()` — explicit

### 15.3 Memory Management
- `torch.cuda.memory_allocated()` — current allocated memory
- `torch.cuda.max_memory_allocated()` — peak allocated
- `torch.cuda.empty_cache()` — release cached memory
- `del tensor; torch.cuda.empty_cache()` — explicit deletion

### 15.4 Mixed Precision Training
- `torch.cuda.amp.autocast()` — automatic mixed precision
- `torch.cuda.amp.GradScaler()` — gradient scaling for float16
- `with autocast():` — wrap forward pass
- `scaler.scale(loss).backward()` — scaled backward
- `scaler.step(optimizer)` — unscale and step
- `scaler.update()` — update scale factor

---

## 16. Model Evaluation and Metrics

### 16.1 Classification Metrics
- `torch.argmax(outputs, dim=1)` — predicted class
- `(preds == labels).float().mean()` — accuracy
- Using `torchmetrics` library — `torchmetrics.Accuracy()`
- `torchmetrics.F1Score()`
- `torchmetrics.AUROC()`
- `torchmetrics.ConfusionMatrix()`

### 16.2 Regression Metrics
- `nn.MSELoss()` — MSE
- `nn.L1Loss()` — MAE
- `torch.sqrt(mse_loss)` — RMSE

### 16.3 Running Metrics
- Accumulating batch metrics
- `metric.update(preds, targets)` — batch update
- `metric.compute()` — final value
- `metric.reset()` — reset state

---

## 17. Regularization Techniques

### 17.1 Dropout
- `nn.Dropout(p=0.5)` — in model definition
- Active during `model.train()` — disabled during `model.eval()`

### 17.2 Weight Decay
- `optim.Adam(model.parameters(), weight_decay=1e-4)` — L2 penalty
- `optim.AdamW` — decoupled weight decay (preferred)

### 17.3 Batch Normalization
- `nn.BatchNorm1d`, `nn.BatchNorm2d` — in model definition
- Behavior changes between `model.train()` and `model.eval()`
- `momentum`, `eps`, `affine` parameters

### 17.4 Early Stopping
- Manual implementation — monitor val loss
- Patience counter
- Restore best weights

### 17.5 Gradient Clipping
- `nn.utils.clip_grad_norm_(parameters, max_norm)` — L2 norm clipping
- `nn.utils.clip_grad_value_(parameters, clip_value)` — value clipping

---

## 18. Advanced PyTorch Features

### 18.1 torch.nn.functional
- `F.relu(x)` — functional activations
- `F.softmax(x, dim=1)` — functional softmax
- `F.cross_entropy(logits, targets)` — functional loss
- `F.conv2d(input, weight, bias)` — functional convolution
- `F.dropout(x, p=0.5, training=self.training)` — functional dropout

### 18.2 Custom Layers
- Subclassing `nn.Module` — custom forward pass
- `nn.Parameter(tensor)` — registering learnable parameters
- `self.register_buffer('name', tensor)` — non-learnable persistent state

### 18.3 Model Parallelism
- `nn.DataParallel(model)` — multi-GPU data parallel
- `torch.nn.parallel.DistributedDataParallel` — distributed training
- Splitting model across GPUs

### 18.4 TorchScript
- `torch.jit.script(model)` — scripting for deployment
- `torch.jit.trace(model, example_input)` — tracing
- Saving scripted model — `scripted_model.save('model.pt')`

### 18.5 torch.compile
- `model = torch.compile(model)` — PyTorch 2.0+ compilation
- Speed improvements via kernel fusion and optimization
- `mode` parameter — `'default'`, `'reduce-overhead'`, `'max-autotune'`

---

## 19. PyTorch Ecosystem

### 19.1 torchvision
- Pre-trained models — `torchvision.models`
  - `resnet50`, `vgg16`, `efficientnet_b0`, `vit_b_16`
- `model = torchvision.models.resnet50(weights='IMAGENET1K_V1')`
- Datasets — `torchvision.datasets`
- Transforms — `torchvision.transforms`
- `torchvision.ops` — detection utilities (NMS, box encoding)
- `torchvision.utils.make_grid()` — image grid for visualization

### 19.2 torchaudio
- Audio datasets — `torchaudio.datasets`
- Audio transforms — `torchaudio.transforms`
- Pre-trained audio models — `torchaudio.models`
- `torchaudio.load()` — loading audio files

### 19.3 torchtext
- Vocabulary building
- Text tokenization
- Pre-trained word vectors

### 19.4 HuggingFace Transformers
- PyTorch backend for BERT, GPT, T5, LLaMA
- `from transformers import AutoModel, AutoTokenizer`
- PyTorch tensors as input

---

## 20. PyTorch Best Practices

### 20.1 Efficient Training
- Always call `optimizer.zero_grad()` before backward
- Use `model.train()` and `model.eval()` appropriately
- Use `torch.no_grad()` for inference
- Use `pin_memory=True` and `num_workers=4` in DataLoader
- Use mixed precision with `autocast()`

### 20.2 Debugging
- `tensor.shape` — always verify shapes
- `print(tensor.dtype, tensor.device)` — verify type and location
- `torch.isnan(tensor).any()` — check for NaN
- `torch.isinf(tensor).any()` — check for Inf
- `anomaly_detection` — `torch.autograd.set_detect_anomaly(True)`

### 20.3 Common Mistakes
- Forgetting `optimizer.zero_grad()` — gradient accumulation bug
- Not calling `model.eval()` during validation — BN and Dropout active
- Not moving data to same device as model
- Calling `.numpy()` on GPU tensor — must call `.cpu()` first
- Using `nn.Softmax` before `nn.CrossEntropyLoss` — double softmax

---

## 21. PyTorch Interview Topics

### 21.1 Frequently Asked Questions
- What is the difference between `torch.Tensor` and `torch.tensor()`?
- What is `autograd` and how does it work?
- What is the difference between `nn.Module` and a plain function?
- Why call `model.train()` and `model.eval()`?
- What is the difference between `view()` and `reshape()`?
- What does `detach()` do?

### 21.2 Coding Problems
- Implement a custom `nn.Module`
- Implement a full training loop with validation
- Implement a custom `Dataset` class
- Build a CNN from scratch
- Implement gradient clipping in training loop

---

## 22. Real-World PyTorch Projects

### 22.1 Beginner Projects
- MNIST digit classification with MLP
- CIFAR-10 classification with CNN
- Linear regression from scratch

### 22.2 Intermediate Projects
- Transfer learning with ResNet50
- LSTM text classification
- Image segmentation with UNet basics

### 22.3 Advanced Projects
- Transformer from scratch
- GAN training loop
- Custom training framework with checkpointing and logging

---

## 23. PyTorch Ecosystem and Extensions

### 23.1 Libraries Built on PyTorch
- HuggingFace Transformers — NLP models
- PyTorch Lightning — training framework
- fast.ai — high-level training API
- Detectron2 — object detection
- MMDetection — detection framework
- MONAI — medical imaging
- PyG (PyTorch Geometric) — graph neural networks
- torchmetrics — evaluation metrics

### 23.2 Deployment Tools
- TorchScript — model serialization
- ONNX export — `torch.onnx.export()`
- TorchServe — model serving
- `torch.compile()` — PyTorch 2.0 compilation
