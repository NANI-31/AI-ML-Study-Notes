import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="5.3 Hugging Face Embedding Models - Embeddings Curriculum",
    page_icon="🤗",
    layout="wide"
)

# Title & Overview
st.title("5.3 Hugging Face Embedding Models: AutoModel Pipeline, MTEB Selection, & Hardware Inference")
st.caption("Chapter 5: Embedding Models | Domain: 3. Vector Embeddings")
st.markdown("""
The Hugging Face Hub hosts over 100,000 open-weights embedding models. While high-level libraries like `sentence-transformers` simplify usage, production machine learning systems frequently deploy raw **Hugging Face `transformers`** primitives (`AutoTokenizer` and `AutoModel`) with PyTorch, ONNX Runtime, TensorRT, or vLLM to maximize throughput, fine-tune custom pooling layers, or execute quantized inference.

This lesson explores:
* **The Raw `transformers` Manual Pipeline**: Tokenization, PyTorch attention masks, manual mean pooling, and L2 normalization.
* **MTEB Benchmark Navigation**: Evaluating models across Retrieval, STS, Classification, and Clustering.
* **Hardware Inference Engineering**: CPU (OpenVINO / ONNX) vs. GPU (FP16 / FlashAttention) vs. Edge Quantization.
* **Interactive Simulator**: Model hardware estimator, MTEB comparison matrix, and raw tensor pooling visualizer.
""")

st.divider()

# Section 1: The Raw Transformers Embedding Pipeline
st.header("1. The Raw Transformers Manual Forward Pipeline", divider="rainbow")

col_pipe1, col_pipe2 = st.columns([1, 1])

with col_pipe1:
    st.markdown("""
    ### Why Build Embeddings with Raw `transformers`?
    While `SentenceTransformer("model")` is convenient, production engineering teams often use raw `AutoTokenizer` and `AutoModel` because:
    1. **Full Control over Tensor Operations**: Custom pooling algorithms (e.g., weighted decay pooling, late chunking, token-level multi-vector ColBERT).
    2. **Zero Overhead for Custom Runtimes**: Exporting directly to **ONNX Runtime**, **TensorRT-LLM**, or **C++ LibTorch** requires raw PyTorch graph definitions.
    3. **Fused Attention & Memory Optimization**: Direct integration with **FlashAttention-2** or PyTorch's `scaled_dot_product_attention (SDPA)`.

    ### The 4-Step Raw Pipeline:
    * **Step 1 (Tokenization)**: Convert text into `input_ids` and binary `attention_mask` tensors with padding and truncation.
    * **Step 2 (Forward Pass)**: Pass through `model(**inputs)`. Extract `last_hidden_state` of shape `[batch_size, seq_len, hidden_dim]`.
    * **Step 3 (Masked Mean Pooling)**: Expand `attention_mask` to `[batch, seq_len, hidden_dim]`. Compute:
      `pooled = sum(last_hidden_state * expanded_mask) / clamp(sum(expanded_mask), min=1e-9)`
    * **Step 4 (L2 Normalization)**: Project vectors onto the unit hypersphere:
      `normalized_emb = F.normalize(pooled, p=2, dim=1)`
    """)

with col_pipe2:
    st.markdown("### Tensor Dimensions Across the Forward Graph")
    tensor_dot = """
    digraph TensorFlow {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        InputStrings [label="Batch of Texts (B=2)\\n['Hello world', 'Transformers']", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        TokenizerNode [label="AutoTokenizer\\ninput_ids: [2, 6]\\nattention_mask: [2, 6]", fillcolor="#FEF3C7"];
        
        EncoderNode [label="AutoModel Forward Pass\\nlast_hidden_state: [2, 6, 768]", fillcolor="#D1FAE5"];
        
        PoolNode [label="Masked Mean Pooling\\nAggregates tokens across seq_len\\npool_out: [2, 768]", fillcolor="#EDE9FE", color="#8B5CF6", style="filled,bold"];
        
        NormNode [label="F.normalize(p=2, dim=1)\\nUnit Hypersphere Projection\\nfinal_embeddings: [2, 768]", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        InputStrings -> TokenizerNode -> EncoderNode -> PoolNode -> NormNode;
    }
    """
    st.graphviz_chart(tensor_dot, use_container_width=False)

st.divider()

# Section 2: Navigating the MTEB Benchmark Leaderboard
st.header("2. Navigating the MTEB (Massive Text Embedding Benchmark)", divider="rainbow")

st.markdown("""
The **MTEB Benchmark** (hosted by Hugging Face) evaluates text embedding models across 8 distinct downstream NLP tasks spanning over 50 datasets:
""")

col_mteb1, col_mteb2 = st.columns([1, 1])

with col_mteb1:
    st.markdown("""
    ### Key MTEB Task Categories:
    * **1. Retrieval (MS MARCO, HotpotQA, NQ)**:
      * Evaluates finding relevant documents given a query (measured via **NDCG@10**).
      * *Crucial for RAG systems and search engines!*
    * **2. Semantic Textual Similarity (STS12 - STS22)**:
      * Evaluates how well cosine similarity correlates with human similarity ratings (Spearman correlation).
    * **3. Reranking (AskUbuntu, SciDocs)**:
      * Ranks candidate documents already retrieved by an initial search.
    * **4. Clustering & Classification (Banking77, AmazonReviews)**:
      * Measures separation of distinct intent and topic clusters (v-measure / accuracy).
    """)

with col_mteb2:
    st.markdown("""
    ### How to Select a Production Model from MTEB:
    * **Do not look only at the overall 'Average' score**:
      * A model might score 70.0 overall because of high classification scores, but perform mediocrely on RAG Retrieval.
      * *For RAG systems, filter specifically by the **Retrieval (NDCG@10)** column!*
    * **Check Context Window vs. Inference Speed**:
      * Models based on modern architectures like **ModernBERT** or **Nomic-Embed** support 8,192 tokens natively.
      * Classical MiniLM or MPNet models cap out at 256 or 512 tokens.
    * **Check Asymmetric Instruction Requirements**:
      * BAAI BGE and Microsoft E5 require instruction prefixes (`query: ` vs `passage: `). Failing to supply them severely degrades retrieval.
    """)

st.markdown("### Top Hugging Face Open-Source Embedding Models")
st.markdown("""
| Model Family | Hub Identifier | Native Dim | Context Window | Size on Disk | MTEB Retrieval (NDCG@10) | Recommended Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BGE-Small** | `BAAI/bge-small-en-v1.5` | 384 | 512 | 134 MB | 51.68 | Ultra-fast CPU / Edge mobile |
| **BGE-Large** | `BAAI/bge-large-en-v1.5` | 1024 | 512 | 1.34 GB | 54.29 | GPU / Production RAG |
| **Nomic-Embed** | `nomic-ai/nomic-embed-text-v1.5` | 768 | **8,192** | 547 MB | 53.01 | Long-document RAG, MRL support |
| **GTE-Qwen2** | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1536 | **32,768** | 3.1 GB | **57.42** | Frontier multi-hop code & legal RAG |
| **ModernBERT** | `answerdotai/ModernBERT-base` | 768 | **8,192** | 598 MB | 53.80 | FlashAttention-native high throughput |
""")

st.divider()

# Section 3: Production Code: Pure PyTorch Hugging Face Pipeline
st.header("3. Production Implementation: Pure PyTorch Hugging Face Embedder", divider="rainbow")

st.markdown("""
The production class below executes the complete end-to-end Hugging Face embedding pipeline using raw PyTorch and `transformers` primitives with attention-masked mean pooling and unit L2 normalization.
""")

st.code(r'''
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

# In production:
# from transformers import AutoTokenizer, AutoModel

class RawHuggingFaceEmbedder:
    """
    Production-grade raw Hugging Face Transformer Embedder.
    Executes tokenization, forward pass, masked mean pooling, and L2 normalization.
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        # In production:
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(device)
        # self.model.eval()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies attention-masked mean pooling across the sequence dimension.
        """
        # Expand attention mask: [batch_size, seq_len] -> [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum token embeddings where mask == 1
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        
        # Sum mask values (number of real tokens per sequence)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """
        Encodes a list of texts into L2-normalized float32 numpy arrays.
        """
        # In production:
        # encoded_input = self.tokenizer(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     max_length=max_length,
        #     return_tensors="pt"
        # ).to(self.device)
        #
        # with torch.no_grad():
        #     model_output = self.model(**encoded_input)
        #     sentence_embeddings = self._mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
        #     normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # return normalized_embeddings.cpu().numpy()

        # Deterministic simulation for illustration:
        rng = np.random.default_rng(42)
        dim = 1024 if "large" in self.model_name else 384
        vecs = rng.standard_normal((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-12)
''', language="python")

# Actual Python implementation for the interactive workbench
class HardwareInferenceSimulator:
    def __init__(self):
        self.models = {
            "BAAI/bge-small-en-v1.5": {"params_m": 33, "dim": 384, "max_ctx": 512, "disk_mb": 134, "ndcg": 51.68},
            "BAAI/bge-large-en-v1.5": {"params_m": 335, "dim": 1024, "max_ctx": 512, "disk_mb": 1340, "ndcg": 54.29},
            "nomic-ai/nomic-embed-text-v1.5": {"params_m": 137, "dim": 768, "max_ctx": 8192, "disk_mb": 547, "ndcg": 53.01},
            "Alibaba-NLP/gte-Qwen2-1.5B": {"params_m": 1500, "dim": 1536, "max_ctx": 32768, "disk_mb": 3100, "ndcg": 57.42}
        }

    def estimate_performance(self, model_key: str, hardware: str, precision: str, batch_size: int):
        m = self.models[model_key]
        params = m["params_m"]
        
        # Base latency factor
        precision_mult = 1.0 if precision == "FP32" else (0.55 if precision == "FP16" else 0.35)
        
        if "A100" in hardware:
            base_tps = 15_000 * (100 / max(params, 20))
            vram_mb = (params * (4 if precision == "FP32" else 2)) + (batch_size * m["dim"] * 4 / 1000)
        elif "T4" in hardware:
            base_tps = 3_500 * (100 / max(params, 20))
            vram_mb = (params * (4 if precision == "FP32" else 2)) + (batch_size * m["dim"] * 4 / 1000)
        elif "CPU" in hardware:
            base_tps = 450 * (100 / max(params, 20))
            vram_mb = 0 # In RAM
        else: # ONNX Quantized CPU
            base_tps = 1_100 * (100 / max(params, 20))
            vram_mb = 0
            
        throughput = base_tps * (1.0 / precision_mult) * (batch_size / 32) ** 0.5
        single_doc_latency_ms = (1000.0 / max(throughput, 1.0)) * batch_size

        return {
            "throughput_doc_sec": int(throughput),
            "latency_ms": single_doc_latency_ms,
            "vram_mb": vram_mb,
            "ndcg": m["ndcg"]
        }

st.divider()

# Section 4: Interactive Workbench: Hardware Inference & MTEB Tradeoff Analyzer
st.header("4. Interactive Workbench: Hardware Inference & MTEB Tradeoff Analyzer", divider="rainbow")

st.markdown("""
Evaluate the engineering tradeoffs between model size, retrieval quality, hardware choices, and precision formats.
Configure an inference pipeline to calculate throughput, memory consumption, and estimated latency.
""")

sim = HardwareInferenceSimulator()

col_sim_l, col_sim_r = st.columns([1, 1])

with col_sim_l:
    st.subheader("1. Deployment Settings")
    selected_hf_model = st.selectbox(
        "Select Hugging Face Model:",
        list(sim.models.keys())
    )

    selected_hw = st.selectbox(
        "Target Hardware Environment:",
        [
            "NVIDIA A100 80GB (Cloud GPU)",
            "NVIDIA T4 16GB (Cost-Effective GPU)",
            "Intel Xeon 8-Core (Standard Cloud CPU)",
            "CPU with ONNX Runtime INT8 (Optimized Edge CPU)"
        ]
    )

    selected_precision = st.radio(
        "Numeric Precision Format:",
        ["FP32 (Standard Float)", "FP16 (Half Precision)", "INT8 (Quantized)"],
        horizontal=True
    )

    batch_sz = st.slider("Inference Batch Size:", min_value=1, max_value=256, value=32, step=1)

with col_sim_r:
    st.subheader("2. Performance & Retrieval Estimation")
    est = sim.estimate_performance(selected_hf_model, selected_hw, selected_precision, batch_sz)

    m1, m2 = st.columns(2)
    m1.metric("Throughput (Docs/sec)", f"{est['throughput_doc_sec']:,}")
    m2.metric("Batch Latency", f"{est['latency_ms']:.1f} ms")

    m3, m4 = st.columns(2)
    m3.metric("MTEB Retrieval NDCG@10", f"{est['ndcg']:.2f}")
    if est["vram_mb"] > 0:
        m4.metric("GPU VRAM Footprint", f"{est['vram_mb']:.1f} MB")
    else:
        m4.metric("GPU VRAM Footprint", "0 MB (CPU Host RAM)")

    st.markdown("#### Hardware Sizing Summary")
    df_hw = pd.DataFrame({
        "Specification": [
            "Model Hub Name",
            "Hardware Environment",
            "Precision Mode",
            "Batch Size",
            "Retrieval Benchmark Quality",
            "Estimated Daily Ingestion Capacity"
        ],
        "Engineering Value": [
            selected_hf_model,
            selected_hw,
            selected_precision,
            f"{batch_sz} documents/batch",
            f"{est['ndcg']} NDCG@10",
            f"{est['throughput_doc_sec'] * 86_400:,} documents / 24 hours"
        ]
    })
    st.dataframe(df_hw, use_container_width=True)

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Hugging Face Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How do you implement Masked Mean Pooling in PyTorch from scratch without using sentence-transformers?**
    * **Answer**:
      * Given `last_hidden_state` of shape `[batch_size, seq_len, hidden_dim]` and `attention_mask` of shape `[batch_size, seq_len]`:
      * 1. Expand the attention mask by unsqueezing the last dimension to match the 3D tensor:
        `mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()`
      * 2. Perform element-wise multiplication to zero out padding token vectors:
        `sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)`
      * 3. Compute the sequence token count by summing the mask, clamping by 1e-9 to avoid zero-division:
        `sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)`
      * 4. Divide: `mean_pooled = sum_embeddings / sum_mask`.
      * 5. Apply L2 normalization: `F.normalize(mean_pooled, p=2, dim=1)`.

    **Q2: When should an enterprise deploy local Hugging Face models over hosted APIs like OpenAI?**
    * **Answer**:
      * 1. **Data Privacy & Compliance**: Highly sensitive healthcare (HIPAA) or defense data that cannot leave on-prem VPCs.
      * 2. **Ultra-Low Latency**: When embedding must happen in sub-10ms inside the local process memory without HTTP WAN network overhead.
      * 3. **High Volume Cost Economics**: At scales of hundreds of millions of tokens per day, paying $0.02-$0.13 per 1M tokens becomes significantly more expensive than running a dedicated GPU instance ($1-$2/hour).
      * 4. **Fine-Tuning Capability**: Open-source models can be contrastively fine-tuned on company-specific proprietary terminology.

    **Q3: What is the MTEB benchmark, and why is the overall leaderboard score potentially misleading for RAG?**
    * **Answer**:
      * MTEB evaluates models across 8 diverse task categories: Retrieval, STS, Classification, Clustering, PairClassification, Reranking, Summarization, and BitextMining.
      * The overall leaderboard score is a simple unweighted average across all 8 tasks.
      * A model can score extremely well on Classification and STS while having mediocre Information Retrieval performance.
      * For RAG systems, engineers must specifically filter for the **Retrieval (NDCG@10)** column on datasets like MS MARCO, HotpotQA, and NQ.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Padding Token Bleed in Unmasked Pooling**:
      * *Issue*: Taking `torch.mean(last_hidden_state, dim=1)` without multiplying by the attention mask.
      * *Consequence*: The zeros or non-zero pad token representations dilute the real sentence vector, severely corrupting short sentences in padded batches!
      * *Fix*: Always mask before summing and divide strictly by the number of unpadded tokens.

    * **Trap 2: Forgetting model.eval() and torch.no_grad()**:
      * *Issue*: Running embedding generation in default training mode without disabling autograd.
      * *Consequence*: PyTorch builds an autograd computation graph in memory for every token, causing immediate CUDA Out-Of-Memory (OOM) errors and 2x slower execution.
      * *Fix*: Wrap inference inside `model.eval()` and `with torch.no_grad():`.

    * **Trap 3: Mismatching Tokenizer max_length with Hardware Limits**:
      * *Issue*: Setting `max_length=8192` on models that only support 512 positional embeddings.
      * *Consequence*: PyTorch throws `IndexError: index out of range in self` when token IDs exceed the positional embedding matrix size.
      * *Fix*: Query `model.config.max_position_embeddings` to establish hard clipping boundaries.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: MTEB Benchmark**
      * *Definition*: Massive Text Embedding Benchmark; the industry-standard benchmark evaluating text embedding models across 8 downstream NLP tasks.
    * **Flashcard 2: Masked Mean Pooling**
      * *Definition*: Aggregating token hidden states into a sequence embedding by multiplying by the attention mask and dividing by the unpadded token count.
    * **Flashcard 3: AutoModel & AutoTokenizer**
      * *Definition*: Hugging Face classes that dynamically instantiate the correct architecture and vocabulary from a model checkpoint.
    * **Flashcard 4: ONNX Runtime Quantization**
      * *Definition*: Converting PyTorch FP32 models to INT8 representations to achieve 2x-4x faster inference speed and lower memory usage on CPUs.
    """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of Hugging Face Embedding Models:

- [ ] I can implement manual attention-masked mean pooling and unit L2 normalization in PyTorch.
- [ ] I understand how to navigate the MTEB leaderboard and why Retrieval NDCG@10 is the key metric for RAG.
- [ ] I know how to evaluate trade-offs between local open-weights models and hosted APIs.
- [ ] I understand the memory and speed benefits of FP16 mixed precision and INT8 ONNX quantization.
- [ ] I know why `model.eval()` and `torch.no_grad()` are essential for production inference pipelines.
""")
