import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# WORKBENCH MODULE-LEVEL ENGINE
# ==============================================================================

class RerankingModelComparator:
    """
    Simulates and benchmarks the four major enterprise reranker model paradigms:
    1. Hosted Cloud API (Cohere Rerank v3)
    2. Open-Source High-Capacity (BAAI BGE-Reranker-Large / BGE-m3)
    3. Lightweight Local CPU (Cross-Encoder ms-marco-MiniLM-L-6-v2)
    4. Generative LLM-as-a-Reranker (RankGPT / Pointwise LLM Scoring)
    """
    MODELS = {
        "Cohere Rerank v3 (Cloud API)": {
            "type": "Managed Cloud API",
            "context_window": "4,096 tokens",
            "latency_p95": "45 ms (network dependent)",
            "deployment": "Zero infra (REST API key)",
            "languages": "100+ Multilingual",
            "relative_ndcg": 0.94
        },
        "BAAI BGE-Reranker-v2-m3 (Local GPU)": {
            "type": "Local Open-Source (XLM-RoBERTa)",
            "context_window": "8,192 tokens",
            "latency_p95": "25 ms (NVIDIA A10G GPU)",
            "deployment": "Self-hosted ONNX / TensorRT",
            "languages": "100+ Multilingual",
            "relative_ndcg": 0.93
        },
        "ms-marco-MiniLM-L-6-v2 (Local CPU)": {
            "type": "Distilled Small Model (6 Layers)",
            "context_window": "512 tokens",
            "latency_p95": "12 ms (CPU Threadpool)",
            "deployment": "Lightweight container / Edge",
            "languages": "English Dominant",
            "relative_ndcg": 0.86
        },
        "RankGPT / LLM-Based Reranker": {
            "type": "Prompted Generative (GPT-4o-mini / Gemini Flash)",
            "context_window": "128,000 tokens",
            "latency_p95": "280 ms (Token generation)",
            "deployment": "API / Enterprise Gateway",
            "languages": "All LLM supported",
            "relative_ndcg": 0.91
        }
    }

    CANDIDATE_CORPUS = [
        {
            "id": "doc_01",
            "text": "PostgreSQL PgBouncer transaction pooling mode releases backend connections immediately when a transaction completes, preventing pool exhaustion under 10k clients.",
            "scores": {
                "Cohere Rerank v3 (Cloud API)": 0.96,
                "BAAI BGE-Reranker-v2-m3 (Local GPU)": 0.95,
                "ms-marco-MiniLM-L-6-v2 (Local CPU)": 0.91,
                "RankGPT / LLM-Based Reranker": 0.94
            }
        },
        {
            "id": "doc_02",
            "text": "Session pooling in PgBouncer allocates a server connection for the complete lifetime of a client session, causing severe queue contention during client spikes.",
            "scores": {
                "Cohere Rerank v3 (Cloud API)": 0.72,
                "BAAI BGE-Reranker-v2-m3 (Local GPU)": 0.74,
                "ms-marco-MiniLM-L-6-v2 (Local CPU)": 0.78,
                "RankGPT / LLM-Based Reranker": 0.70
            }
        },
        {
            "id": "doc_03",
            "text": "General relational database query optimization requires indexing foreign keys, analyzing EXPLAIN ANALYZE output, and vacuuming bloated dead tuples.",
            "scores": {
                "Cohere Rerank v3 (Cloud API)": 0.12,
                "BAAI BGE-Reranker-v2-m3 (Local GPU)": 0.15,
                "ms-marco-MiniLM-L-6-v2 (Local CPU)": 0.22,
                "RankGPT / LLM-Based Reranker": 0.10
            }
        }
    ]

    @classmethod
    def compare_models(cls, query: str) -> pd.DataFrame:
        rows = []
        for name, meta in cls.MODELS.items():
            rows.append({
                "Model Architecture": name,
                "Model Type": meta["type"],
                "Context Window": meta["context_window"],
                "Inference Latency": meta["latency_p95"],
                "Infrastructure Requirement": meta["deployment"],
                "Language Support": meta["languages"],
                "Benchmark NDCG@10": meta["relative_ndcg"]
            })
        return pd.DataFrame(rows)

    @classmethod
    def evaluate_candidates(cls, selected_model: str) -> List[Dict[str, Any]]:
        results = []
        for doc in cls.CANDIDATE_CORPUS:
            score = doc["scores"][selected_model]
            results.append({
                "id": doc["id"],
                "relevance_score": score,
                "text": doc["text"]
            })
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results

# ==============================================================================
# STREAMLIT PAGE CONFIG & HEADER
# ==============================================================================

st.set_page_config(
    page_title="13.2 Reranking Models - RAG Master Curriculum",
    page_icon="🤖",
    layout="wide"
)

st.title("13.2 Reranking Models: Cross-Encoders, Cohere, BGE, LLM-as-a-Reranker, & Local Deployment")
st.caption("Chapter 13: Reranking | Domain: 6. Retrieval-Augmented Generation")

st.markdown("""
Not all rerankers are architected equally. Depending on data privacy boundaries, latency constraints, 
cloud network permissions, and language diversity, enterprise engineers choose between four primary model paradigms:
* **Hosted Cloud APIs (Cohere Rerank v3)**: State-of-the-art multilingual accuracy, 4k context window, zero infrastructure overhead.
* **Open-Source Deep Models (BAAI BGE-Reranker-v2-m3)**: High-capacity 8k context window, self-hosted on private GPU clusters for compliance.
* **Lightweight Distilled Models (ms-marco-MiniLM-L-6-v2)**: Ultra-fast sub-15ms CPU inference, ideal for edge devices and low-cost microservices.
* **Generative LLM Rerankers (RankGPT / Pointwise)**: Zero fine-tuning required, leverages general reasoning capabilities of 100k+ token LLMs at higher latency.

This lesson analyzes the architectural trade-offs, inference engines, and operational profiles of each reranking model family.
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================

st.header("1. Architectural Blueprint: Neural Reranking Model Architectures", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    models_bp_dot = """
    digraph RerankingModelsBP {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.45, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=9, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_input {
            label = "1. First-Stage Candidates Pool";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            q_and_candidates [label="Incoming Query + 25 Retrieved Candidate Chunks", fillcolor="#0F172A", color="#38BDF8"];
            router [label="Model Selection & Routing Policy\\nEvaluates Privacy, SLA Latency, & Token Length", fillcolor="#334155", color="#10B981"];

            q_and_candidates -> router;
        }

        subgraph cluster_models {
            label = "2. Specialized Neural Reranker Model Paradigms";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            cohere_node [label="Cohere Rerank v3 (Cloud API)\\n4k Window | 100+ Langs | Latency ~45ms", fillcolor="#0F172A", color="#F59E0B"];
            bge_node [label="BGE-Reranker-v2-m3 (Local GPU)\\n8k Window | Self-Hosted TensorRT | Latency ~25ms", fillcolor="#0F172A", color="#EC4899"];
            minilm_node [label="ms-marco-MiniLM-L-6 (Local CPU)\\n512 Window | Distilled 6-Layer | Latency ~12ms", fillcolor="#0F172A", color="#06B6D4"];
            llm_node [label="RankGPT / LLM-as-Reranker\\nPrompt-Based Listwise Sorting | Latency ~280ms", fillcolor="#0F172A", color="#8B5CF6"];

            router -> cohere_node [label="Cloud Allowed"];
            router -> bge_node [label="Private GPU VPC"];
            router -> minilm_node [label="Ultra-Low Latency CPU"];
            router -> llm_node [label="Complex Reasoning"];
        }

        subgraph cluster_output {
            label = "3. Normalized Relevance Score Hydration Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#334155";
            fontcolor = "#38BDF8";
            fontname = "Helvetica-Bold";
            fontsize = 11;

            calibrator [label="Probability Calibrator\\nSigmoid / Min-Max Normalization", fillcolor="#334155", color="#10B981"];
            top_ranked [label="Sorted High-Precision Context Finalists", fillcolor="#0F172A", color="#10B981"];

            cohere_node -> calibrator;
            bge_node -> calibrator;
            minilm_node -> calibrator;
            llm_node -> calibrator;

            calibrator -> top_ranked;
        }
    }
    """
    st.graphviz_chart(models_bp_dot, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production enterprise architectures, teams adopt **Cohere Rerank v3** when data is permitted in commercial cloud endpoints 
due to its unmatched multilingual accuracy and 4,096-token window. 
In highly regulated on-premise environments (finance, defense, healthcare), teams deploy **BGE-Reranker-v2-m3** 
using **Triton Inference Server** with **TensorRT-LLM** or **vLLM**, achieving 20–30ms batch inference across 50 candidate pairs on a single GPU.
""")

st.divider()

# ==============================================================================
# SECTION 2: CROSS-ENCODER RERANKERS
# ==============================================================================

st.header("2. Cross-Encoder Rerankers: Joint Self-Attention Mechanics and Logit Heads", divider="rainbow")

st.markdown("""
How does the cross-encoder compute relevance, and why is it mathematically superior to bi-encoder dot products?
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("The Cross-Attention Mechanism")
    st.markdown("""
    In a Cross-Encoder:
    * The input is formatted as a single combined token sequence:
      `Input = [CLS] query_tokens [SEP] doc_tokens [SEP]`
    * Inside every Transformer layer, every query token directly computes self-attention weights 
      against every document token (`Attention = Softmax(Q * K^T / sqrt(d)) * V`).
    * The `[CLS]` token at layer 0 attends to all tokens, and by layer 12/24, its hidden state encapsulates 
      the complete joint semantic relationship between query and passage.
    * A linear classification head projects the final `[CLS]` hidden vector into a single scalar logit:
      `Relevance_Logit = W * h_[CLS] + b`
    """)

with col2:
    st.subheader("Cross-Encoder Architectural Diagram")
    ce_dot = """
    digraph CrossEncoderArch {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.35, ranksep=0.45];
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin="0.15,0.08"];
        edge [color="#38BDF8", fontname="Helvetica", fontsize=8, fontcolor="#94A3B8"];

        seq_in [label="Concatenated Input: [CLS] Query [SEP] Document [SEP]", fillcolor="#0F172A", color="#38BDF8"];
        trans_stack [label="Full Transformer Encoder Stack\\nAll-to-All Cross Attention across all layers", fillcolor="#334155", color="#EC4899"];
        cls_pool [label="Final [CLS] Token Hidden Vector\\nh_cls (Float32[768/1024])", fillcolor="#0F172A", color="#10B981"];
        linear_head [label="Binary Classification Head\\nW * h_cls + b", fillcolor="#334155", color="#F59E0B"];
        scalar_out [label="Scalar Relevance Logit / Sigmoid Probability", fillcolor="#0F172A", color="#10B981"];

        seq_in -> trans_stack -> cls_pool -> linear_head -> scalar_out;
    }
    """
    st.graphviz_chart(ce_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 3: COHERE RERANKING
# ==============================================================================

st.header("3. Cohere Reranking: Managed Cloud Infrastructure, Multilingual Support, & Token Windows", divider="rainbow")

st.markdown("""
**Cohere Rerank (v3)** is widely regarded as the commercial benchmark for managed neural reranking.
""")

col_coh1, col_coh2 = st.columns([1, 1])

with col_coh1:
    st.subheader("Operational Advantages of Cohere Rerank")
    st.markdown("""
    * **4,096-Token Context Window**:
      * Standard open-source cross-encoders (like BERT-based models) truncate text at 512 tokens. 
      * Cohere Rerank v3 supports up to 4,096 tokens per document candidate, enabling reranking of long legal clauses, 
        medical study sections, and multi-page technical documentation without aggressive chunking.
    * **Native Multilingual Alignment**:
      * Trained natively across 100+ languages. A query in Spanish can accurately rerank documents written in English or German.
    * **Zero Infrastructure Overhead**:
      * Dispatched via a single SDK call (`co.rerank(query=..., documents=..., top_n=3)`).
      * No GPU provisioning, CUDA driver management, or auto-scaling cluster maintenance.
    """)

with col_coh2:
    st.subheader("Python Integration Pattern")
    st.code('''# Cohere Rerank v3 Production Integration
import cohere

co = cohere.Client(api_key="COHERE_API_KEY")

response = co.rerank(
    model="rerank-v3.5",
    query="How does transaction isolation prevent dirty reads?",
    documents=[
        "PostgreSQL MVCC snapshot isolation guide...",
        "Relational database foreign key design..."
    ],
    top_n=3,
    return_documents=True
)

for result in response.results:
    print(f"Index: {result.index} | Score: {result.relevance_score:.4f}")
''', language="python")

st.divider()

# ==============================================================================
# SECTION 4: BGE RERANKERS
# ==============================================================================

st.header("4. BGE Rerankers: Open-Source State-of-the-Art (BGE-Reranker-Large & BGE-m3)", divider="rainbow")

st.markdown("""
Developed by the Beijing Academy of Artificial Intelligence (BAAI), the **BGE-Reranker** family represents 
the leading open-source model suite for self-hosted reranking.
""")

col_bge1, col_bge2 = st.columns([1, 1])

with col_bge1:
    st.subheader("BGE Model Family Specifications")
    st.markdown("""
    * **`bge-reranker-base` / `large`**:
      * Based on RoBERTa architecture. 512-token context window.
      * Ideal for rapid sentence-level and small chunk reranking on standard enterprise GPUs.
    * **`bge-reranker-v2-m3`**:
      * Built on XLM-RoBERTa backbone.
      * **8,192-Token Context Window**: Supports extremely long document passages.
      * Multilingual support spanning 100+ languages.
      * High ranking accuracy competitive with proprietary commercial APIs.
    """)

with col_bge2:
    st.subheader("Sentence-Transformers Local Execution Pattern")
    st.code('''# Local BGE Reranker Execution
from sentence_transformers import CrossEncoder

# Load model locally on CUDA
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=1024,
    device="cuda"
)

# Prepare candidate pairs
query = "KafkaConsumer poll thread safety"
docs = ["Kafka consumer thread model...", "Kafka producer buffer..."]
pairs = [[query, doc] for doc in docs]

# Compute scores in a single batch
scores = reranker.predict(pairs, batch_size=16)
''', language="python")

st.divider()

# ==============================================================================
# SECTION 5: LLM-BASED RERANKING
# ==============================================================================

st.header("5. LLM-Based Reranking: Pointwise, Pairwise, and Listwise Prompting (RankGPT)", divider="rainbow")

st.markdown("""
Instead of training a dedicated cross-encoder classification head, **LLM-Based Reranking** prompts an autoregressive 
foundation model to evaluate and sort documents.
""")

col_llm1, col_llm2 = st.columns([1, 1])

with col_llm1:
    st.subheader("The Three LLM Reranking Paradigms")
    st.markdown("""
    1. **Pointwise Reranking**:
       * The LLM is prompted to output a relevance score (e.g., 0 to 5) for one document at a time.
       * *Pros*: Simple prompt structure.
       * *Cons*: High latency (`O(K)` serial LLM inferences); lacks cross-document calibration.
    2. **Pairwise Reranking**:
       * The LLM compares two documents simultaneously (*"Is Doc A or Doc B more relevant?"*).
       * *Pros*: Higher accuracy than pointwise.
       * *Cons*: Quadratic complexity (`O(K^2)` comparisons).
    3. **Listwise Reranking (RankGPT)**:
       * The LLM receives the query and all 10 candidate documents in a single prompt and outputs the sorted list of document IDs:
         `"Rank: [Doc 3, Doc 1, Doc 2]"`
       * *Pros*: Full cross-document context; competitive with dedicated cross-encoders.
       * *Cons*: Slower than specialized cross-encoders (250–500ms).
    """)

with col_llm2:
    st.subheader("RankGPT Listwise Prompt Pattern")
    st.code('''# RankGPT Listwise Prompt Template
RANKGPT_PROMPT = """I will provide you with {num_docs} passages. 
Rank the passages based on their relevance to the search query.
Sort passages from most relevant to least relevant.
Output only the ordered identifiers (e.g. [1] > [3] > [2]).

Query: {query}

[1] {doc_1}
[2] {doc_2}
[3] {doc_3}

Search Ranking:"""
''', language="python")

st.divider()

# ==============================================================================
# SECTION 6: LOCAL RERANKING MODELS
# ==============================================================================

st.header("6. Local Reranking Models: Distillation, ONNX Runtime, and CPU Optimization", divider="rainbow")

st.markdown("""
When cloud APIs are prohibited and GPUs are unavailable, deploying distilled local cross-encoders on CPU 
provides sub-15ms reranking with minimal infrastructure cost.
""")

col_loc1, col_loc2 = st.columns([1, 1])

with col_loc1:
    st.subheader("Optimizing Cross-Encoders for CPU Deployment")
    st.markdown("""
    * **Distilled MiniLM Architecture**:
      * Models like `cross-encoder/ms-marco-MiniLM-L-6-v2` use only 6 Transformer layers (22M parameters). 
      * Evaluates 25 candidate pairs on a standard 4-core CPU in under 15 milliseconds!
    * **ONNX Runtime Acceleration**:
      * Exporting the PyTorch cross-encoder to ONNX graph format enables:
        * Kernel fusion (combining LayerNorm + Attention).
        * AVX-512 and ARM NEON hardware SIMD vectorization.
        * Up to 3.5x throughput acceleration on CPU.
    * **8-bit Integer Quantization (INT8)**:
      * Dynamic quantization compresses model weights from 90 MB to 23 MB with < 1% NDCG degradation.
    """)

with col_loc2:
    st.subheader("Local Inference Engine Comparison")
    local_df = pd.DataFrame([
        {"Inference Engine", "Batch Latency (25 pairs)", "RAM Footprint", "Hardware Target"},
        {"PyTorch Float32 (CPU)", "42 ms", "380 MB", "Generic Cloud VM"},
        {"ONNX Runtime (CPU)", "14 ms", "120 MB", "High-throughput CPU Pod"},
        {"ONNX INT8 Quantized (CPU)", "9 ms", "45 MB", "Edge Devices & MicroVMs"},
        {"TensorRT (NVIDIA GPU)", "3.5 ms", "1.2 GB", "High-Scale Production Cluster"}
    ])
    st.dataframe(local_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================

st.header("7. Production Implementation: Enterprise Multi-Model Reranker Adapter", divider="rainbow")

st.markdown("""
Below is an enterprise-grade Python adapter class that provides a unified interface across 
Cohere cloud reranking, local BGE models, and lightweight CPU MiniLM models.
""")

st.code(r'''
from typing import List, Dict, Any, Optional

class EnterpriseMultiModelRerankerAdapter:
    """
    Unified adapter supporting multiple reranker backends:
    - 'cohere': Cloud API via Cohere Rerank v3
    - 'local_gpu': BAAI/bge-reranker-v2-m3
    - 'local_cpu': cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    def __init__(self, backend: str = "local_cpu", model_name: Optional[str] = None):
        self.backend = backend
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = None

    def initialize(self) -> None:
        if self.backend == "local_cpu" or self.backend == "local_gpu":
            from sentence_transformers import CrossEncoder
            device = "cuda" if self.backend == "local_gpu" else "cpu"
            self._model = CrossEncoder(self.model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        if self.backend == "cohere":
            # Cohere API execution pattern
            import cohere
            co = cohere.Client()
            texts = [d["text"] for d in documents]
            response = co.rerank(query=query, documents=texts, top_n=top_k, model="rerank-v3.5")
            fused = []
            for hit in response.results:
                item = dict(documents[hit.index])
                item["relevance_score"] = float(round(hit.relevance_score, 4))
                fused.append(item)
            return fused

        else:
            # Local CrossEncoder execution
            if self._model is None:
                self.initialize()
            pairs = [[query, d["text"]] for d in documents]
            scores = self._model.predict(pairs, batch_size=32)

            fused = []
            for doc, score in zip(documents, scores):
                item = dict(doc)
                item["relevance_score"] = float(round(score, 4))
                fused.append(item)
            fused.sort(key=lambda x: x["relevance_score"], reverse=True)
            return fused[:top_k]
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================

st.header("8. Interactive Workbench: Cross-Model Reranking Comparison Lab", divider="rainbow")

st.markdown("""
Compare the four major neural reranker architectures side-by-side. 
Select a model below to evaluate candidate scoring behavior and inspect benchmark trade-offs.
""")

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    selected_model_name = st.selectbox(
        "Select Active Reranking Model:",
        list(RerankingModelComparator.MODELS.keys())
    )
    model_spec = RerankingModelComparator.MODELS[selected_model_name]

    st.markdown("#### ⚙️ Selected Model Profile")
    st.write(f"**Architecture**: `{model_spec['type']}`")
    st.write(f"**Context Window**: `{model_spec['context_window']}`")
    st.write(f"**Inference Latency**: `{model_spec['latency_p95']}`")
    st.write(f"**Deployment Requirement**: `{model_spec['deployment']}`")
    st.write(f"**Language Coverage**: `{model_spec['languages']}`")

with wb_col2:
    st.markdown("### 🏆 Live Candidate Scoring Output")
    scored_candidates = RerankingModelComparator.evaluate_candidates(selected_model_name)
    eval_df = pd.DataFrame(scored_candidates)
    st.dataframe(eval_df[["id", "relevance_score", "text"]], use_container_width=True)

st.markdown("### 📊 Comprehensive Multi-Model Benchmark Comparison")
bench_df = RerankingModelComparator.compare_models("test")
st.dataframe(bench_df, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================

st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: What are the primary trade-offs between managed API rerankers (like Cohere) and self-hosted models (like BGE)?"):
    st.markdown("""
    **Answer**:
    * **Managed API (Cohere)**:
      * *Pros*: Zero infrastructure management; 4,096-token context window; state-of-the-art multilingual accuracy across 100+ languages.
      * *Cons*: Requires sending customer data over external network boundaries (violates strict data sovereignty laws in banking/healthcare); network transit adds 30–50ms variable latency.
    * **Self-Hosted (BGE-Reranker-v2-m3)**:
      * *Pros*: 100% data privacy within your private VPC; deterministic sub-25ms latency on local GPUs; 8,192-token context window; zero third-party per-query API bills.
      * *Cons*: Requires provisioning, monitoring, and autoscaling expensive GPU instances (e.g., A10G/T4).
    """)

with st.expander("Q2: Why is ms-marco-MiniLM-L-6-v2 popular for CPU-based enterprise deployments?"):
    st.markdown("""
    **Answer**: `ms-marco-MiniLM-L-6-v2` is distilled down to only 6 Transformer layers and 22 million parameters. 
    Because of its tiny footprint, it evaluates 25 candidate pairs on a standard commodity CPU in under 15 milliseconds. 
    It enables enterprise teams to deploy two-stage retrieve-and-rerank pipelines in lightweight Docker containers on existing 
    Kubernetes CPU clusters without needing expensive dedicated GPU nodes.
    """)

with st.expander("Q3: When should teams choose Listwise LLM Reranking (RankGPT) over dedicated Cross-Encoders?"):
    st.markdown("""
    **Answer**: Listwise LLM Reranking is advantageous when:
    1. Documents require **deep multi-criteria reasoning** (e.g., evaluating candidates against complex compliance rules where traditional cross-encoders struggle).
    2. Zero infrastructure is available for deploying specialized cross-encoder models, but an enterprise LLM API (GPT-4o or Gemini Flash) is already integrated.
    3. The application requires dynamic, user-specified ranking criteria (e.g., *"Rank by recentness and technical depth"*).
    """)

with st.expander("Q4: What causes the 'Tokenizer Mismatch' problem when migrating between different reranker models?"):
    st.markdown("""
    **Answer**: Different reranker models utilize different subword tokenizers (e.g., BERT WordPiece vs. RoBERTa BPE vs. Cohere custom tokenizer). 
    A text passage that fits within 500 tokens in WordPiece may expand to 620 tokens in BPE due to different subword splitting rules. 
    If a system assumes a fixed character count or token length across models, switching from BERT to RoBERTa can trigger silent token truncation 
    at the end of documents without throwing an exception.
    """)

with st.expander("Q5: How does ONNX Runtime quantization reduce cross-encoder inference costs?"):
    st.markdown("""
    **Answer**: Cross-encoders are standard PyTorch FP32 models. Converting to ONNX format fuses attention matrix operations into single C++ kernels. 
    Applying 8-bit dynamic quantization (INT8) converts matrix multiplications from 32-bit floating-point to 8-bit integer operations. 
    This reduces RAM usage by 75% and leverages CPU AVX-512 VNNI instructions, accelerating throughput by 2.5x to 3.5x with less than 1% degradation in NDCG ranking precision.
    """)

st.subheader("Production Traps: Engineering Gotchas & Edge Cases")

col_trap1, col_trap2 = st.columns(2)

with col_trap1:
    st.error("""
    **Trap 1: Sending Unsanitized PII to Commercial Cloud Reranker APIs**
    * **Symptom**: Compliance audit failure under GDPR, HIPAA, or SOC2.
    * **Root Cause**: Streaming raw candidate documents containing employee medical notes or credit card data to external cloud endpoints.
    * **Remedy**: Deploy self-hosted open-source rerankers (such as BGE-m3) inside your private VPC, or run a local PII anonymization filter before API dispatch.
    """)

    st.error("""
    **Trap 2: Silently Truncating Candidates at 512 Tokens on BERT Cross-Encoders**
    * **Symptom**: Reranker fails to recognize relevance when the key answer is located in the second half of a 400-word paragraph.
    * **Root Cause**: BERT-based cross-encoders have a hard 512-token sequence limit (`query + document`). Texts exceeding 512 are silently truncated.
    * **Remedy**: Use high-context rerankers like `bge-reranker-v2-m3` (8k window) or Cohere Rerank v3 (4k window) for long passages.
    """)

with col_trap2:
    st.warning("""
    **Trap 3: Using Pointwise LLM Scoring Instead of Listwise Sorting**
    * **Symptom**: LLM reranker adds 1.8 seconds of latency and generates huge API bills.
    * **Root Cause**: Prompting the LLM 25 times sequentially to output a score from 1 to 5.
    * **Remedy**: Use Listwise RankGPT patterns where all 10–15 candidate snippets are passed into a single prompt for one-shot ordering.
    """)

    st.warning("""
    **Trap 4: Neglecting Batch Size Tuning on GPU Cross-Encoders**
    * **Symptom**: GPU utilization sits at 8% while search requests queue up.
    * **Root Cause**: Scoring candidate pairs one at a time (`batch_size=1`) rather than passing all 25–50 pairs in a single batched tensor.
    * **Remedy**: Configure `batch_size = 32` or `batch_size = 64` to fully saturate GPU CUDA streaming multiprocessors.
    """)

st.subheader("Core Architectural Flashcards")

flash_col1, flash_col2 = st.columns(2)

with flash_col1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary operational advantage of Cohere Rerank v3?")
        st.markdown("**Back**: A 4,096-token context window and state-of-the-art multilingual performance across 100+ languages with zero local GPU infrastructure maintenance.")

    with st.container(border=True):
        st.markdown("**Front**: Why is BGE-Reranker-v2-m3 the leading open-source reranker?")
        st.markdown("**Back**: An 8,192-token context window built on XLM-RoBERTa, enabling self-hosted private GPU deployments that compete with commercial APIs.")

with flash_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the sweet spot for ms-marco-MiniLM-L-6-v2?")
        st.markdown("**Back**: Sub-15ms inference on standard commodity CPUs, enabling cost-effective, GPU-free reranking in lightweight containers.")

    with flash_col2:
        with st.container(border=True):
            st.markdown("**Front**: What is Listwise LLM Reranking (RankGPT)?")
            st.markdown("**Back**: A prompting technique where an LLM inspects a query and all candidate passages in a single prompt, outputting the complete ordered list of document IDs.")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================

st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("""
Verify your mastery of Neural Reranking Models, Cohere, BGE, LLM Rerankers, and Local Optimization:
""")

chk1 = st.checkbox("I understand the architectural differences between Cloud APIs, Local GPU, and Local CPU rerankers.", value=True)
chk2 = st.checkbox("I can articulate the context window advantages of Cohere v3 (4k) and BGE-m3 (8k) over standard BERT (512).", value=True)
chk3 = st.checkbox("I know how to deploy distilled cross-encoders on CPU using ONNX Runtime for sub-15ms latency.", value=True)
chk4 = st.checkbox("I understand the trade-offs of Listwise LLM reranking (RankGPT) compared to dedicated Cross-Encoders.", value=True)
chk5 = st.checkbox("I know how to batch candidate pairs into a single tensor to maximize GPU saturation.", value=True)
chk6 = st.checkbox("I understand data sovereignty constraints that require self-hosted BGE models over commercial cloud APIs.", value=True)

if all([chk1, chk2, chk3, chk4, chk5, chk6]):
    st.success("✅ Phenomenal! You have mastered Neural Reranking Models and Enterprise Deployment paradigms.")
