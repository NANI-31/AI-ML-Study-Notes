import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

st.set_page_config(
    page_title="4.2 Embedding Models - ChromaDB Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("4.2 Embedding Models: OpenAI, Sentence Transformers, Hugging Face & Custom Functions")
st.caption("Chapter 4: Embeddings and Vector Representations | Domain: 5. ChromaDB")
st.markdown("""
The quality of any vector search or RAG pipeline is bounded by the quality of its **Embedding Model**. 
If the embedding model fails to capture technical nuances, no amount of HNSW graph tuning will yield relevant search hits.

Modern AI architects choose between two major embedding paradigms:
1. **Cloud-Hosted Commercial APIs (OpenAI, Cohere, Google PaLM)**:
   * Zero local GPU hardware requirements.
   * State-of-the-art multi-lingual semantic comprehension.
   * Drawbacks: Cloud API latency (15–50ms), recurring token costs, and data privacy / regulatory exposure.
2. **Local Open-Source Models (Sentence Transformers, BGE, E5 via HuggingFace / ONNX)**:
   * 100% offline, air-gapped, and zero data leakage.
   * Predictable sub-millisecond local inference on CPU or GPU.
   * Zero per-query API bills.
   * Drawbacks: Requires server RAM and local compute resources.

ChromaDB provides pluggable abstractions for all major embedding providers via its `EmbeddingFunction` interface.

In this lesson, we dissect embedding provider architectures, write production integration code for each provider, 
build a custom embedding function from scratch, and test an interactive Model Benchmark & Cost-Latency Sizing Arena.
""")

st.divider()

# Section 1: The Embedding Model Landscape
st.header("1. The Embedding Model Landscape", divider="rainbow")

col_land1, col_land2 = col_land1, col_land2 = st.columns([1.1, 0.9])

with col_land1:
    st.markdown("""
    ### Provider Topologies Compared
    
    * **1. OpenAI (`text-embedding-3-small` / `large`)**:
      * *Dimensions*: 1536 (small) or 3072 (large).
      * *Features*: Supports Matryoshka dimension truncation (e.g. truncate 1536d to 512d).
      * *Best For*: Commercial enterprise apps with variable query volumes.
    
    * **2. Local Sentence-Transformers (`all-MiniLM-L6-v2`, `bge-base-en`)**:
      * *Dimensions*: 384d (MiniLM) or 768d (BGE).
      * *Features*: Runs locally on CPU/GPU via PyTorch or ONNX Runtime.
      * *Best For*: High-throughput search, on-premises privacy, edge devices.
    
    * **3. Custom Self-Hosted Microservices (Triton / Ollama / vLLM)**:
      * *Dimensions*: Arbitrary.
      * *Features*: Wrap company-internal proprietary embedding models behind a clean gRPC/REST interface.
    """)

with col_land2:
    st.markdown("### Model Selection Decision Tree")
    tree_dot = """
    digraph ModelDecision {
        rankdir=TB;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        Start [label="Select Embedding Provider", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        QPrivacy [label="Can data leave company VPC\\n(Cloud API Allowed)?", shape=diamond, fillcolor="#FEF3C7"];

        OpenAI [label="OpenAI text-embedding-3-small\\n(High Quality / Zero Dev Ops)", fillcolor="#D1FAE5", color="#059669"];

        QCompute [label="Do you have local\\nGPU accelerators?", shape=diamond, fillcolor="#EDE9FE"];

        LocalMiniLM [label="Sentence-Transformers MiniLM (CPU)\\n(Fast / 384d / Low RAM)", fillcolor="#FEF3C7", color="#D97706"];

        BGE_GPU [label="BAAI BGE-Large (GPU)\\n(SOTA Open Source / 1024d)", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];

        Start -> QPrivacy;
        QPrivacy -> OpenAI [label="Yes (Cloud)"];
        QPrivacy -> QCompute [label="No (Air-Gapped)"];
        QCompute -> LocalMiniLM [label="CPU Only"];
        QCompute -> BGE_GPU [label="GPU Available"];
    }
    """
    st.graphviz_chart(tree_dot, use_container_width=False)

st.divider()

# Section 2: Implementation Patterns for All Major Providers
st.header("2. Implementation Patterns for All Major Providers", divider="rainbow")

col_code_a, col_code_b = col_code_a, col_code_b = st.columns([1, 1])

with col_code_a:
    st.markdown("### 1. OpenAI & Sentence-Transformers")
    st.code("""import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")

# --- Provider 1: OpenAI Embedding Function ---
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-your-openai-key",
    model_name="text-embedding-3-small"
)

collection_openai = client.get_or_create_collection(
    name="openai_kb",
    embedding_function=openai_ef
)

# --- Provider 2: Local Sentence-Transformers ---
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5",
    device="cpu"  # or "cuda"
)

collection_st = client.get_or_create_collection(
    name="local_bge_kb",
    embedding_function=st_ef
)
""", language="python")

with col_code_b:
    st.markdown("### 2. Building a Custom Embedding Function")
    st.code("""from chromadb.api.types import Documents, Embeddings
from chromadb import EmbeddingFunction
import requests
from typing import List

class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    \"\"\"Custom Embedding Function wrapping a local Ollama server.\"\"\"
    
    def __init__(self, model_name: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.url = f"{url}/api/embeddings"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "prompt": text}
            )
            data = response.json()
            embeddings.append(data["embedding"])
        return embeddings

# Use custom function in collection
ollama_ef = OllamaEmbeddingFunction()
ollama_collection = client.get_or_create_collection(
    name="ollama_private_kb",
    embedding_function=ollama_ef
)
""", language="python")

st.divider()

# Section 3: The Model Selection Decision Framework
st.header("3. The Model Selection Decision Framework", divider="rainbow")

st.markdown("""
Review the key trade-offs across latency, cost, privacy, and accuracy when choosing 
an embedding foundation for ChromaDB:
""")

PROVIDER_AUDIT = [
    {
        "Provider / Model": "OpenAI text-embedding-3-small",
        "Hosted Type": "Commercial Cloud API",
        "Cost per 1M Tokens": "$0.02 USD",
        "Inference Latency": "20–45ms (Network Bound)",
        "Air-Gapped Privacy": "❌ Transmits data to OpenAI servers",
        "MTEB Benchmark Score": "62.3 (High Precision)"
    },
    {
        "Provider / Model": "all-MiniLM-L6-v2 (Default Chroma)",
        "Hosted Type": "In-Process CPU (ONNX)",
        "Cost per 1M Tokens": "$0.00 (Free)",
        "Inference Latency": "3–5ms (CPU SIMD)",
        "Air-Gapped Privacy": "✅ 100% Offline Local RAM",
        "MTEB Benchmark Score": "56.2 (Lightweight)"
    },
    {
        "Provider / Model": "BAAI/bge-base-en-v1.5",
        "Hosted Type": "Local GPU / Server",
        "Cost per 1M Tokens": "$0.00 (Self-Hosted)",
        "Inference Latency": "2ms on GPU / 12ms on CPU",
        "Air-Gapped Privacy": "✅ 100% Offline Local RAM",
        "MTEB Benchmark Score": "63.5 (State-of-the-art)"
    },
    {
        "Provider / Model": "Cohere embed-english-v3.0",
        "Hosted Type": "Commercial Cloud API",
        "Cost per 1M Tokens": "$0.10 USD",
        "Inference Latency": "30–60ms (Network Bound)",
        "Air-Gapped Privacy": "❌ Transmits data to Cohere",
        "MTEB Benchmark Score": "64.5 (Industry Leader)"
    }
]

st.dataframe(pd.DataFrame(PROVIDER_AUDIT), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Interactive Workbench: Model Benchmark & Sizing Arena
st.header("4. Interactive Workbench: Model Benchmark & Sizing Arena", divider="rainbow")

st.markdown("""
Size your enterprise knowledge base. Adjust total document volume and query throughput 
to calculate estimated monthly API costs, RAM requirements, and retrieval latency profiles.
""")

col_bench_in, col_bench_cfg = col_bench_in, col_bench_cfg = st.columns(2)

with col_bench_in:
    st.subheader("1. Ingestion & Workload Demographics")
    num_documents = st.slider("Total Indexed Documents (Chunks):", min_value=1000, max_value=500000, value=50000, step=5000)
    avg_tokens_per_doc = st.slider("Average Tokens per Document:", min_value=50, max_value=1000, value=250, step=25)
    monthly_queries = st.slider("Monthly Inbound Search Queries:", min_value=5000, max_value=1000000, value=100000, step=10000)

with col_bench_cfg:
    st.subheader("2. Model Selection")
    eval_model = st.selectbox(
        "Choose Model to Benchmark:",
        [
            "OpenAI text-embedding-3-small (1536d / Cloud)",
            "all-MiniLM-L6-v2 (384d / Local ONNX)",
            "BAAI/bge-base-en-v1.5 (768d / Local)",
            "OpenAI text-embedding-3-large (3072d / Cloud)"
        ]
    )

# Compute estimates
total_tokens_ingest = num_documents * avg_tokens_per_doc
query_tokens_month = monthly_queries * 25  # ~25 tokens per search query
total_tokens_month = total_tokens_ingest + query_tokens_month

if "3-small" in eval_model:
    cost_per_m = 0.02
    dims = 1536
    latency = "28ms (HTTP API)"
    ram_gb = (num_documents * dims * 4) / (1024**3) * 1.5
    privacy = "☁️ Cloud API (Data leaves VPC)"
elif "MiniLM" in eval_model:
    cost_per_m = 0.0
    dims = 384
    latency = "4ms (Local CPU)"
    ram_gb = (num_documents * dims * 4) / (1024**3) * 1.5
    privacy = "🔒 100% Offline Air-Gapped"
elif "bge-base" in eval_model:
    cost_per_m = 0.0
    dims = 768
    latency = "8ms (Local CPU) / 2ms (GPU)"
    ram_gb = (num_documents * dims * 4) / (1024**3) * 1.5
    privacy = "🔒 100% Offline Air-Gapped"
else:  # 3-large
    cost_per_m = 0.13
    dims = 3072
    latency = "45ms (HTTP API)"
    ram_gb = (num_documents * dims * 4) / (1024**3) * 1.5
    privacy = "☁️ Cloud API (Data leaves VPC)"

monthly_api_cost = (total_tokens_month / 1000000.0) * cost_per_m

st.divider()

col_out1, col_out2, col_out3 = st.columns(3)
col_out1.metric(label="Estimated Monthly API Cost", value=f"${monthly_api_cost:.2f} USD")
col_out2.metric(label="Estimated Vector RAM Footprint", value=f"{ram_gb:.2f} GB RAM")
col_out3.metric(label="Average Retrieval Latency", value=latency)

st.info(f"**Privacy & Security Compliance**: {privacy}")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

st.markdown("### High-Frequency Architectural & Design Questions")

with st.expander("Q1: What is the `EmbeddingFunction` interface in ChromaDB, and how do you implement it?"):
    st.markdown("""
    **Answer**:
    * In ChromaDB, `EmbeddingFunction` is a generic callable protocol that accepts a list of document strings and returns a list of floating-point vectors:
      `def __call__(self, input: Documents) -> Embeddings`.
    * To create a custom function:
      1. Inherit from `chromadb.EmbeddingFunction`.
      2. Implement the `__call__` method to batch text strings and call your model (e.g., local Ollama, vLLM, or AWS SageMaker endpoint).
      3. Return a list of lists of floats (`List[List[float]]`).
    """)

with st.expander("Q2: Why is running an embedding model locally via ONNX Runtime faster than Python PyTorch?"):
    st.markdown("""
    **Answer**:
    * **ONNX Runtime (Open Neural Network Exchange)**:
      * Compiles the model graph, fusing operations (e.g. LayerNorm + Gelu fusion) to eliminate intermediate memory allocations.
      * Leverages low-level CPU vectorization instructions (AVX-512 on Intel/AMD, Neon on ARM) without the overhead of Python's GIL.
      * Eliminates the heavy 2GB+ PyTorch binary dependency, running efficiently inside lightweight Docker containers with minimal RAM.
    """)

with st.expander("Q3: How do you handle API rate limits when using `OpenAIEmbeddingFunction` on a large dataset?"):
    st.markdown("""
    **Answer**:
    1. **Client-Side Exponential Backoff**: Wrap API calls in `tenacity` retry loops that back off exponentially upon encountering HTTP 429 Too Many Requests.
    2. **Batching**: Group text chunks into batches of 500–1,000 documents per single API call rather than calling the API on single chunks.
    3. **Token Bucket Rate Limiting**: Throttle requests client-side to stay below your tier's Tokens Per Minute (TPM) quota.
    """)

with st.expander("Q4: What is the MTEB (Massive Text Embedding Benchmark)?"):
    st.markdown("""
    **Answer**:
    * MTEB is the universal standard benchmark for evaluating embedding models across 8 diverse NLP tasks:
      1. Retrieval (RAG / Search)
      2. Semantic Textual Similarity (STS)
      3. Classification
      4. Clustering
      5. Pair Classification
      6. Reranking
      7. Summarization
      8. Bitext Mining
    * When selecting a model for ChromaDB RAG, examine the model's **Retrieval sub-score** on the MTEB leaderboard.
    """)

with st.expander("Q5: What happens if an external embedding API experiences an outage during ingestion?"):
    st.markdown("""
    **Answer**:
    * If `OpenAIEmbeddingFunction` encounters an HTTP 500 error or network timeout, the entire `collection.add()` call fails and throws an exception.
    * In production, **pre-compute embeddings in a dedicated ETL worker pipeline** (e.g. Celery or Temporal) and pass the pre-computed float vectors directly to ChromaDB using `embeddings=vectors`.
    * This decouples ChromaDB ingestion from external third-party API availability!
    """)

st.markdown("### Production Traps & Antipatterns")

col_trap1, col_trap2 = col_trap1, col_trap2 = st.columns([1, 1])

with col_trap1:
    st.error("""
    **Trap 1: Hardcoding OpenAI API Keys in Collection Definitions**  
    Embedding API keys directly into Python scripts exposes enterprise credentials in Git. Inject keys via environment variables (`OPENAI_API_KEY`).
    """)
    st.warning("""
    **Trap 2: Changing Embedding Models on an Existing Collection**  
    Switching from MiniLM to OpenAI on an existing collection causes fatal dimensionality mismatch errors. Create a new collection for new models.
    """)

with col_trap2:
    st.error("""
    **Trap 3: Unhandled Rate Limits During Massive Batch Ingestion**  
    Ingesting 100,000 documents using OpenAI without retry backoff triggers HTTP 429 rate limit exceptions, failing the ingestion job halfway through.
    """)
    st.warning("""
    **Trap 4: Missing GPU Drivers for Heavy Local Models**  
    Attempting to run 1,024-dim models (e.g. BGE-Large) on single-core CPU instances causes 500ms+ inference latency per chunk, stalling search.
    """)

st.markdown("### Interactive Flashcards")

col_fc1, col_fc2 = col_fc1, col_fc2 = st.columns([1, 1])

with col_fc1:
    with st.expander("🗂️ Flashcard: What is an EmbeddingFunction?"):
        st.markdown("""
        **Callable Protocol**:  
        Accepts `Documents` and returns `Embeddings` (`List[List[float]]`) for ChromaDB collections.
        """)
    with st.expander("🗂️ Flashcard: Why use ONNX for local models?"):
        st.markdown("""
        **Fused CPU Execution**:  
        Optimizes graph execution and uses AVX SIMD instructions for 3ms CPU inference.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: When to choose local vs cloud?"):
        st.markdown("""
        **Privacy & Latency vs DevOps**:  
        Local for privacy and zero marginal cost; Cloud for zero GPU maintenance and multi-lingual quality.
        """)
    with st.expander("🗂️ Flashcard: What is the MTEB benchmark?"):
        st.markdown("""
        **Universal Standard**:  
        Massive Text Embedding Benchmark measuring retrieval, clustering, and semantic similarity.
        """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the architectural differences between cloud-hosted APIs and local embedding models.", value=True)
chk2 = st.checkbox("I know how to configure `OpenAIEmbeddingFunction` and `SentenceTransformerEmbeddingFunction` in ChromaDB.", value=True)
chk3 = st.checkbox("I can implement a custom `EmbeddingFunction` subclass wrapping local Ollama or private APIs.", value=True)
chk4 = st.checkbox("I understand how to handle API rate limits and why pre-computing embeddings decouples pipelines.", value=True)
chk5 = st.checkbox("I know how to evaluate embedding models using the MTEB leaderboard retrieval benchmarks.", value=True)
chk6 = st.checkbox("I am ready to advance to Chapter 4.3: Embedding Functions in ChromaDB!", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Chapter 4.2: Embedding Models and are ready for Chapter 4.3: Embedding Functions in ChromaDB!")
