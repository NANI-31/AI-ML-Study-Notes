Yes. If you're going to give the AI **only one PyTorch section at a time**, use this as your reusable **master PyTorch prompt**. Each time, just replace `[PYTORCH SECTION]`.

I am learning PyTorch as part of my AI/ML and Generative AI learning journey.

I will give you **one PyTorch roadmap section at a time**. Your job is to teach me that section thoroughly, from fundamentals to advanced understanding.

Assume that I know basic Python, NumPy, and have some exposure to Scikit-Learn and ML concepts, but I want to develop a **strong practical and conceptual understanding** of PyTorch — how tensors work, how neural networks are built and trained, how autograd works, and how PyTorch is used to build and fine-tune deep learning and LLM models.

## PyTorch Section

[PYTORCH SECTION]

---

# How I want you to teach me

For every topic and subtopic in the section, follow this learning structure:

### 1. What is it?

- Give a clear definition.
- Explain the problem it solves.
- Explain why PyTorch provides this feature and how it fits in the deep learning stack.
- Place it in the PyTorch workflow: Data → DataLoader → Model → Forward Pass → Loss → Backward Pass → Optimizer Step → Evaluation.

### 2. Intuition

- Explain the concept in very simple language.
- Use a real-world analogy when useful.
- Explain the "why" before the "how".
- Then gradually move toward the technical explanation.

### 3. Prerequisites

- Mention any Python/NumPy/math/ML concepts I need to understand first.
- If necessary, briefly teach the prerequisite before continuing.

Key prerequisite areas:

- NumPy array operations (PyTorch tensors are NumPy arrays + GPU + autograd)
- Linear algebra (matrix multiplication, dot products)
- Calculus (derivatives, chain rule — for backpropagation)
- Basic ML concepts (loss function, gradient descent, overfitting)

### 4. Syntax and Core Concepts

- Show the important PyTorch API.
- Explain important classes, methods, and parameters.
- Explain what each part of the code does.
- Always show tensor shapes explicitly: `# shape: (batch_size, seq_len, hidden_dim)`.

```python
import torch
import torch.nn as nn

# Create a tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)   # torch.Size([2, 2])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
```

### 5. Examples

Give examples progressively:

**Beginner → Intermediate → Advanced**

For every important example:

- Show the code.
- Explain it line by line.
- Show the expected output including tensor shapes.
- Explain why the output is produced.

Use small, traceable examples first (e.g., a 2-neuron network on 4 data points) before scaling to realistic deep learning scenarios.

### 6. PyTorch Internals

When relevant, explain what PyTorch is doing internally.

For example:

**Tensors:**

- How tensors are stored in memory (contiguous vs. non-contiguous)
- Strides and how `view()` vs. `reshape()` differ
- How `.detach()` removes a tensor from the computation graph
- How `.numpy()` shares memory with the tensor

**Autograd:**

- How PyTorch builds a dynamic computation graph (DAG) on the forward pass
- How `.backward()` traverses the DAG in reverse to compute gradients
- What `requires_grad=True` means and which tensors track gradients
- How `torch.no_grad()` disables gradient tracking for inference
- What `.grad` accumulation means and why `optimizer.zero_grad()` is critical

**nn.Module:**

- How `nn.Module` registers parameters and submodules
- How `model.parameters()` collects all leaf tensors with `requires_grad=True`
- How `model.train()` vs. `model.eval()` changes BatchNorm and Dropout behavior
- How `state_dict()` and `load_state_dict()` work for saving/loading models

### 7. Math Behind It

Explain the mathematical foundations clearly when relevant:

- **Forward pass**: how input flows through linear layers, activation functions
- **Loss function**: cross-entropy, MSE — what they compute and why
- **Gradient**: what `∂Loss/∂w` means and how it tells us which direction to update weights
- **Chain rule**: how gradients flow backward through composite functions
- **Gradient descent update**: `w = w - lr * ∂L/∂w`
- **Batch normalization**: mean/variance normalization per batch, learnable scale/shift
- **Attention (when relevant)**: `softmax(QK^T / sqrt(d_k)) V`

Write key equations in plain text or markdown math notation. Explain every term.

### 8. Training Loop — Full Pattern

For any section involving model training, always show and explain the complete PyTorch training loop:

```python
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 1. Move to device
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # 2. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 3. Backward pass
        optimizer.zero_grad()   # clear previous gradients
        loss.backward()         # compute gradients
        optimizer.step()        # update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Explain every line. Explain what happens if any step is done out of order or skipped.

### 9. Device Management (CPU / GPU / MPS)

Explain GPU usage clearly:

- How to detect and set the device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- How to move tensors and models to device: `.to(device)`
- Why tensors on different devices cause errors
- Mixed precision training: `torch.autocast` and `GradScaler`
- `device_map="auto"` for HuggingFace models vs. manual `.to(device)` for custom models
- MPS (Apple Silicon) device support

### 10. AI/ML and GenAI Connection

For every concept, explain where it fits in real AI/ML or Generative AI applications:

- **Tensors**: the universal data format for model inputs (images, tokens, embeddings)
- **nn.Linear**: the foundation of attention layers, feed-forward blocks
- **nn.Embedding**: how token IDs become embedding vectors in LLMs
- **Autograd**: how LLMs are fine-tuned — backpropagating through billions of parameters
- **DataLoader**: how training batches are fed to the model
- **Custom Dataset**: how to wrap any dataset for LLM fine-tuning
- **`torch.no_grad()`**: inference mode — used in all production LLM serving

### 11. Performance and Memory

Explain performance and memory considerations:

- **Batch size**: larger batches = more memory, better GPU utilization
- **Mixed precision (fp16/bf16)**: `torch.autocast` — halves memory, faster on modern GPUs
- **Gradient checkpointing**: trading compute for memory in large model training
- **`torch.compile()`**: PyTorch 2.0 model compilation for faster inference
- **Memory fragmentation**: why `torch.cuda.empty_cache()` helps but isn't a full fix
- **DataLoader workers**: `num_workers > 0` for parallel data loading
- **`pin_memory=True`**: faster CPU → GPU data transfer

### 12. Common Mistakes

Show common mistakes beginners make.

For each mistake:

**Wrong approach → Why it is wrong → Correct approach**

Also mention important pitfalls such as:

- Forgetting `optimizer.zero_grad()` — gradients accumulate across batches
- Calling `loss.backward()` twice — computation graph is freed after first call
- Not calling `model.eval()` during inference — Dropout and BatchNorm behave differently
- Mixing up `.view()` and `.reshape()` on non-contiguous tensors
- Comparing tensors with `==` instead of `torch.equal()`
- Not moving tensors to the same device before operations
- Using `model.parameters()` in the optimizer after freezing layers — includes frozen params

### 13. Comparison

Compare PyTorch approaches or concepts in a table where multiple options exist.

For example:

| Approach             | Use case             | Advantages        | Disadvantages        |
| -------------------- | -------------------- | ----------------- | -------------------- |
| `nn.Sequential`      | Simple feedforward   | Compact, readable | No branching         |
| `nn.Module` subclass | Any architecture     | Full control      | More boilerplate     |
| `torch.compile()`    | Production inference | Faster            | Compilation overhead |

Also compare:

- `view()` vs. `reshape()` vs. `contiguous().view()`
- `Adam` vs. `SGD` vs. `AdamW`
- `nn.CrossEntropyLoss` vs. `nn.NLLLoss` + `nn.LogSoftmax`
- `model.eval()` + `torch.no_grad()` vs. `torch.inference_mode()`

### 14. Best Practices

Give practical rules I should follow when writing PyTorch code for real AI/ML projects.

Include:

- Always set `torch.manual_seed()` for reproducibility
- Always use `model.eval()` + `torch.no_grad()` for inference
- Always use `AdamW` instead of `Adam` for transformer fine-tuning
- Always use `device_map` or explicit `.to(device)` — never hardcode `"cuda"`
- Save checkpoints with `model.state_dict()` + `optimizer.state_dict()`
- Use `torch.autocast` for mixed precision training
- Any section-specific best practices

### 15. Interview Perspective

Give me important interview questions related to the section.

Include:

- Conceptual questions (e.g., "What is a computation graph and why does PyTorch build it dynamically?")
- Math questions (e.g., "What does `.backward()` compute and how does the chain rule apply?")
- Debugging questions (e.g., "Your model's loss is not decreasing — what could be wrong?")
- Engineering questions (e.g., "How would you train a model on a GPU with only 8GB VRAM?")
- GenAI questions (e.g., "How does gradient checkpointing reduce memory during LLM fine-tuning?")

Provide the answers and explanations after the questions.

### 16. Practice

Give me exercises at three levels:

**Beginner**

- 2–3 exercises (tensor operations, simple linear model)

**Intermediate**

- 2–3 exercises (custom Dataset, training loop, MLP for classification)

**Advanced**

- 2–3 exercises (custom nn.Module with attention, mixed precision training, model checkpointing)

Do NOT immediately provide solutions.

Let me solve them first.

When I submit my answers:

- Check them.
- Identify mistakes.
- Explain why they are wrong.
- Show the correct solution.
- Suggest a better approach if one exists.

### 17. Mini Project

Give me a small PyTorch mini-project using the concepts in this section.

Give me:

- Problem statement
- Dataset to use (toy dataset, or `torchvision.datasets`, or a simple synthetic dataset)
- Architecture requirements
- Training requirements
- Evaluation requirements
- Constraints (e.g., must run on CPU, must achieve >90% validation accuracy)
- Hints

Do not provide the complete solution until I attempt it.

---

# Important Teaching Rules

1. Do not just give definitions. Teach the reasoning and math behind every PyTorch concept.

2. **Always show tensor shapes** — after every operation, comment the resulting shape.

3. Start simple (scalar → vector → matrix → batched tensor) and gradually increase complexity.

4. Always show the full training loop when training is involved — never a partial snippet.

5. When autograd is involved, explicitly trace the computation graph with a small example.

6. When GPU/device concepts are involved, always show both CPU and GPU-compatible code.

7. When performance is relevant, give concrete memory estimates (e.g., "a 7B model in bf16 needs ~14GB VRAM").

8. Do not introduce unrelated PyTorch topics.

9. If an advanced concept depends on another concept, explain the dependency first.

10. Do not assume that I understand backpropagation, the computation graph, or CUDA deeply.

11. Use tables, computation graph diagrams, and shape-annotated code when they make concepts clearer.

12. Distinguish clearly between:

- **Tensor operation**: mathematical computation on data
- **Autograd**: automatic differentiation machinery
- **nn.Module**: model building block
- **Training loop**: the iterative optimization procedure
- **Inference**: using a trained model without gradient computation

13. Focus on understanding rather than memorization.

14. Relate every concept to real deep learning and GenAI use cases wherever meaningful.

15. Do not teach the entire PyTorch roadmap at once.

16. Teach ONLY the section I provide.

17. At the end, give me a concise **Revision Sheet** containing:

- Key concepts
- Key classes and functions
- Important tensor operations and their shape behavior
- Training loop template
- Common mistakes
- Best practices checklist
- Device/memory quick reference

18. After teaching the section, STOP and wait for me. Do not automatically move to another PyTorch section.

---

# Teaching Goal

My goal is not simply to learn PyTorch's API.

I want to understand:

**What → Why → Math → Tensor Internals → Autograd → Model Architecture → Training Loop → GPU → Fine-Tuning → Production**

so that I can confidently build, train, fine-tune, and deploy deep learning and LLM models using PyTorch.

Now teach me the following PyTorch section:

[PYTORCH SECTION]

**How to use it:** paste the prompt once, then replace `[PYTORCH SECTION]` with something like `4. Building Neural Networks with nn.Module` or a smaller subsection such as `4.1 Linear Layers, Activations, and Sequential`. This keeps each lesson focused and prevents the AI from overwhelming you.
