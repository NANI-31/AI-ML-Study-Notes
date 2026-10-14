import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="5.1 OpenAI Embedding Models - Embeddings Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("5.1 OpenAI Embedding Models: text-embedding-3 Architecture, Matryoshka Truncation, & Production API Engineering")
st.caption("Chapter 5: Embedding Models | Domain: 3. Vector Embeddings")
st.markdown("""
OpenAI provides industry-standard hosted embedding APIs used across thousands of enterprise RAG and semantic search systems.
From the legacy `text-embedding-ada-002` to the third-generation **`text-embedding-3-small`** and **`text-embedding-3-large`**, these models offer high semantic fidelity, large context windows (8,191 tokens), and built-in **Matryoshka Representation Learning (MRL)** that allows flexible vector dimension truncation without re-training!

This lesson explores:
* **The OpenAI Embeddings Hierarchy**: Ada-002 vs. 3-Small vs. 3-Large.
* **Matryoshka Truncation & L2 Renormalization**: Shrinking 3,072 dimensions to 1,024 or 256 for 75% vector storage savings while retaining 98%+ retrieval quality.
* **API Payload Architecture & Batch Serialization**: Handling token ceilings, rate limits (RPM / TPM), exponential backoff, and cost economics.
* **Interactive Simulator**: Benchmark embedding cost, storage footprints, and Matryoshka truncation dynamics in real time.
""")

st.divider()

# Section 1: The Model Lineup & Architectural Evolution
st.header("1. OpenAI Embedding Model Lineup & Architectural Evolution", divider="rainbow")

col_lineup1, col_lineup2 = st.columns([1, 1])

with col_lineup1:
    st.markdown("""
    ### Evolution of OpenAI Embedding Models:
    * **1. Legacy Models (text-similarity-babbage-001 / davinci-001)**:
      * Era: 2020-2022. High latency, expensive, separate query and document models. Deprecated.
    * **2. text-embedding-ada-002 (Dec 2022)**:
      * Unified query and document retrieval into a single 1,536-dimensional model.
      * Replaced the complex multi-engine system with a single universal encoder.
      * Token price: $0.10 per 1M tokens.
    * **3. text-embedding-3-small (Jan 2024)**:
      * Native dimensions: 1,536 (supports Matryoshka down to 512).
      * Higher MTEB retrieval performance than ada-002 with a **5x price reduction** ($0.02 per 1M tokens).
      * The go-to default for low-latency, cost-effective vector search.
    * **4. text-embedding-3-large (Jan 2024)**:
      * Native dimensions: 3,072 (supports Matryoshka down to 1024, 512, 256).
      * Frontier accuracy on code search, multilingual benchmarks, and fine-grained legal/scientific search.
      * Token price: $0.13 per 1M tokens.
    """)

with col_lineup2:
    st.markdown("### OpenAI Embeddings API Request-Response Cycle")
    api_dot = """
    digraph OpenAIAPI {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        Client [label="Client Application\\nPayload: ['chunk 1', 'chunk 2']\\nmodel: 'text-embedding-3-small'\\ndimensions: 512", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        
        Gateway [label="OpenAI API Gateway\\nAuthentication & Rate Limiter (TPM/RPM)", fillcolor="#F1F5F9"];
        Tokenizer [label="cl100k_base / o200k Tokenizer\\nChunk Length Verification (<= 8,191 tokens)", fillcolor="#FEF3C7"];
        Encoder [label="Transformer Encoder\\nFull Forward Pass (1,536 Native Dimensions)", fillcolor="#D1FAE5"];
        Matryoshka [label="MRL Truncation & L2 Renormalization\\nSlice first 512 values -> norm to 1.0", fillcolor="#EDE9FE", color="#8B5CF6"];
        Response [label="JSON Response (200 OK)\\nList of float vectors + usage tokens", fillcolor="#DCFCE7", color="#16A34A", style="filled,bold"];

        Client -> Gateway -> Tokenizer -> Encoder -> Matryoshka -> Response -> Client;
    }
    """
    st.graphviz_chart(api_dot, use_container_width=False)

st.markdown("### Model Benchmark Comparison Table")
st.markdown("""
| Model Name | Native Dims | Matryoshka Supported? | Context Window | Price / 1M Tokens | MTEB Retrieval Avg | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **text-embedding-3-small** | 1,536 | Yes (e.g. 512) | 8,191 tokens | **$0.02** | 62.3 | Default for general RAG, high-volume search, low latency |
| **text-embedding-3-large** | 3,072 | Yes (e.g. 1024, 256) | 8,191 tokens | **$0.13** | 64.6 | Frontier legal, medical, code search, complex reasoning |
| **text-embedding-ada-002** | 1,536 | No (Fixed 1,536) | 8,191 tokens | **$0.10** | 61.0 | Legacy deployments; migration to 3-small recommended |
""")

st.divider()

# Section 2: Matryoshka Representation Learning (MRL) Deep Dive
st.header("2. Matryoshka Representation Learning (MRL) & Vector Truncation", divider="rainbow")

col_mrl1, col_mrl2 = st.columns([1, 1])

with col_mrl1:
    st.markdown("""
    ### Why Truncating Arbitrary Vectors Fails:
    In standard neural network embeddings (like Ada-002 or BERT), semantic information is distributed unpredictably across all dimensions.
    * If you chop off dimensions 513 through 1536 from an Ada-002 vector, retrieval performance plummets drastically (losing 40%+ accuracy).

    ### How Matryoshka Embeddings Work:
    Named after Russian nesting dolls (*Matryoshka*), MRL trains the model using a multi-scale loss function:
    * The first 256 dimensions are trained to be a complete, highly predictive embedding.
    * The first 512 dimensions contain the first 256 plus higher-order nuances.
    * The first 1,024 dimensions contain even more fine-grained distinctions.
    * The full 3,072 dimensions provide maximum resolution.

    ### The Golden Rule of MRL Truncation:
    When truncating via OpenAI's `dimensions` API parameter or locally:
    1. Slice the prefix: `v_truncated = v_full[:k]`
    2. **Mandatory L2 Renormalization**: You MUST divide `v_truncated` by its Euclidean norm (`norm = sqrt(sum(v_i ^ 2))`)!
       Without renormalization, cosine similarity computations will be distorted.
    """)

with col_mrl2:
    st.markdown("### The Matryoshka Nesting Architecture")
    mrl_dot = """
    digraph Matryoshka {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        subgraph cluster_full {
            label="Full Latent Vector (3,072 Dimensions)"; style="dashed"; color="#475569";
            
            D256 [label="Core Semantic Prefix\\n[0 : 256]\\n~95% Retrieval Accuracy", fillcolor="#FEF08A", color="#CA8A04", style="filled,bold"];
            D512 [label="Refined Nuance\\n[256 : 512]\\n~97% Accuracy", fillcolor="#FED7AA", color="#EA580C"];
            D1024 [label="High-Fidelity Context\\n[512 : 1024]\\n~98.5% Accuracy", fillcolor="#DDD6FE", color="#7C3AED"];
            D3072 [label="Frontier Micro-Distinctions\\n[1024 : 3072]\\n100% Benchmark Accuracy", fillcolor="#BFDBFE", color="#2563EB"];

            D256 -> D512 -> D1024 -> D3072;
        }
    }
    """
    st.graphviz_chart(mrl_dot, use_container_width=False)
    st.info("**Key Benefit**: Truncating from 3,072 to 1,024 dimensions cuts vector database RAM and indexing costs by **66.7%** while sacrificing less than 1.5% in retrieval accuracy!")

st.divider()

# Section 3: Production Code: Robust OpenAI Client Implementation
st.header("3. Production Implementation: Resilient OpenAI Batch Client", divider="rainbow")

st.markdown("""
A production embedding client must handle batch packing (up to 2,048 inputs per request), token budget calculations using `tiktoken`, automatic exponential backoff for HTTP 429 (Rate Limit) errors, and optional client-side Matryoshka truncation.
""")

st.code(r'''
import time
import math
import numpy as np
from typing import List, Dict, Any, Optional

class OpenAIEmbeddingClient:
    """
    Production-grade wrapper for OpenAI Embedding API.
    Features batch token chunking, exponential backoff, and Matryoshka L2 renormalization.
    """
    def __init__(self, api_key: str = "mock-key", model: str = "text-embedding-3-small", target_dim: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.target_dim = target_dim  # e.g., 512 or 1024
        self.max_batch_size = 512      # OpenAI allows up to 2048 texts per call

    def _l2_renormalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalizes a truncated vector to unit length (L2 norm = 1.0)."""
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def create_embeddings(self, texts: List[str], max_retries: int = 3) -> List[np.ndarray]:
        """
        Embeds a list of texts in batches with simulated retry backoff logic.
        """
        embeddings_result = []

        # Batch loop
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            
            # Simulated API call with exponential backoff retry loop
            for attempt in range(max_retries):
                try:
                    # In production:
                    # response = client.embeddings.create(
                    #     input=batch,
                    #     model=self.model,
                    #     dimensions=self.target_dim  # Matryoshka dimension parameter
                    # )
                    # batch_vectors = [item.embedding for item in response.data]
                    
                    # Simulated synthetic embeddings for illustration:
                    native_dim = 3072 if "large" in self.model else 1536
                    dim_to_use = self.target_dim if self.target_dim else native_dim
                    
                    batch_vectors = []
                    for text in batch:
                        # Deterministic pseudo-vector seeded by text hash
                        rng = np.random.default_rng(abs(hash(text)) % (2**31))
                        raw = rng.standard_normal(dim_to_use)
                        batch_vectors.append(self._l2_renormalize(raw))
                    
                    embeddings_result.extend(batch_vectors)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"OpenAI API failed after {max_retries} attempts: {e}")
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)

        return embeddings_result
''', language="python")

# Actual Python implementation for the interactive workbench
class MockOpenAIEmbeddingService:
    def __init__(self):
        self.pricing = {
            "text-embedding-3-small": {"per_1m": 0.02, "native_dim": 1536, "mrl": True},
            "text-embedding-3-large": {"per_1m": 0.13, "native_dim": 3072, "mrl": True},
            "text-embedding-ada-002": {"per_1m": 0.10, "native_dim": 1536, "mrl": False}
        }

    def calculate_cost_and_storage(self, model_name: str, num_chunks: int, avg_tokens: int, custom_dim: int):
        cfg = self.pricing[model_name]
        total_tokens = num_chunks * avg_tokens
        cost_usd = (total_tokens / 1_000_000) * cfg["per_1m"]
        
        # Storage: 4 bytes per float32 coordinate
        bytes_per_vec = custom_dim * 4
        total_mb = (num_chunks * bytes_per_vec) / (1024 * 1024)
        
        # MRL retention heuristic
        native_dim = cfg["native_dim"]
        if not cfg["mrl"]:
            accuracy_pct = 100.0 if custom_dim == native_dim else 60.0
        else:
            ratio = custom_dim / native_dim
            # Heuristic curve: log decay
            accuracy_pct = min(100.0, 93.0 + 7.0 * (custom_dim / native_dim) ** 0.5)

        return {
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "total_mb": total_mb,
            "accuracy_pct": accuracy_pct
        }

st.divider()

# Section 4: Interactive Workbench: Cost, Storage, & MRL Simulator
st.header("4. Interactive Workbench: OpenAI Cost, Storage, & Matryoshka Calculator", divider="rainbow")

st.markdown("""
Experiment with OpenAI embedding configurations. Toggle between models, adjust dataset volume, and test Matryoshka dimension truncation to quantify token costs, vector memory footprints, and expected retrieval accuracy retention.
""")

service = MockOpenAIEmbeddingService()

col_sim_in, col_sim_out = st.columns([1, 1])

with col_sim_in:
    st.subheader("1. Ingestion Configuration")
    model_sel = st.selectbox(
        "Select OpenAI Model:",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    )
    
    num_docs = st.number_input("Number of Corpus Documents / Chunks:", min_value=100, max_value=10_000_000, value=250_000, step=50_000)
    avg_tokens_per_doc = st.slider("Average Tokens per Chunk:", min_value=32, max_value=8192, value=400, step=32)
    
    native_d = service.pricing[model_sel]["native_dim"]
    supports_mrl = service.pricing[model_sel]["mrl"]
    
    if supports_mrl:
        dim_options = [256, 512, 1024, native_d] if native_d == 3072 else [256, 512, native_d]
        dim_choice = st.select_slider(
            "Matryoshka Target Dimension:",
            options=dim_options,
            value=native_d
        )
    else:
        st.warning("⚠️ text-embedding-ada-002 does NOT support Matryoshka truncation. Fixed at 1,536 dimensions.")
        dim_choice = 1536

with col_sim_out:
    st.subheader("2. Financial & Infrastructure Impact")
    calc = service.calculate_cost_and_storage(model_sel, num_docs, avg_tokens_per_doc, dim_choice)
    
    m1, m2 = st.columns(2)
    m1.metric("Total Tokens Processed", f"{calc['total_tokens']:,}")
    m2.metric("Total Embedding Cost (USD)", f"${calc['cost_usd']:.2f}")

    m3, m4 = st.columns(2)
    m3.metric("Raw Vector RAM (Float32)", f"{calc['total_mb']:.1f} MB")
    m4.metric("Estimated Accuracy Retention", f"{calc['accuracy_pct']:.1f}%")

    st.markdown("#### Cost & Infrastructure Breakdown")
    savings_vs_large = ((calc['total_mb'] / ((num_docs * 3072 * 4) / (1024 * 1024))) * 100)
    
    df_metrics = pd.DataFrame({
        "Metric": [
            "Model Name",
            "Target Dimensions",
            "Storage per Vector",
            "Total Ingestion Cost",
            "Vector Storage Footprint",
            "Relative RAM Footprint"
        ],
        "Value": [
            model_sel,
            f"{dim_choice} dimensions",
            f"{dim_choice * 4} bytes",
            f"${calc['cost_usd']:.4f}",
            f"{calc['total_mb']:.2f} MB ({calc['total_mb']/1024:.2f} GB)",
            f"{savings_vs_large:.1f}% of full 3-large uncompressed"
        ]
    })
    st.dataframe(df_metrics, use_container_width=True)

    if dim_choice < native_d:
        st.success(f"💡 By truncating from {native_d} to {dim_choice} dimensions, you reduced vector storage and index size by **{(1.0 - dim_choice/native_d)*100:.1f}%** while retaining ~**{calc['accuracy_pct']:.1f}%** semantic fidelity!")

st.divider()

# Section 5: Technical Interview Preparation
st.header("5. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "OpenAI Embeddings Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: How does Matryoshka Representation Learning (MRL) work in OpenAI text-embedding-3 models?**
    * **Answer**:
      * In traditional embeddings, information is scattered arbitrarily across coordinates. Truncating coordinates destroys vector semantics.
      * MRL trains the model using a nested loss formulation: during pretraining, early coordinate slices (e.g. [:256], [:512], [:1024]) are evaluated alongside the full vector.
      * The loss forces the network to pack the most important, coarse semantic concepts into the earliest coordinates, while later dimensions capture fine-grained subtleties.
      * Developers can safely slice the vector down to smaller dimensions (e.g. 512) and re-normalize, achieving 3x-6x savings in vector DB storage and search latency with less than 2% drop in MTEB benchmark score.

    **Q2: What is the mandatory post-processing step if you truncate an embedding client-side instead of using the API dimensions parameter?**
    * **Answer**:
      * **Unit L2 Renormalization**: If you slice a vector client-side (`v_trunc = v[:512]`), the sum of squares of the remaining elements no longer equals 1.0!
      * If you feed this unnormalized truncated vector into a vector index using dot product search, the lengths will be skewed and similarity rankings will be corrupted.
      * You MUST divide the slice by its Euclidean norm: `v_norm = v_trunc / norm(v_trunc)`.

    **Q3: How should you handle rate limits (TPM / RPM) when embedding millions of tokens with OpenAI?**
    * **Answer**:
      * 1. **Batch Packing**: Send up to 2,048 strings per HTTP POST request (staying below the 8M token per call ceiling) to minimize HTTP handshake overhead.
      * 2. **Token Bucket / Semaphore**: Track client-side tokens using `tiktoken` to throttle before hitting the API tier's TPM ceiling.
      * 3. **Exponential Backoff with Jitter**: Catch HTTP 429 errors and retry after `(2 ** attempt) + random_jitter` seconds.
      * 4. **Batch API**: Use OpenAI's asynchronous Batch API for non-realtime indexing to receive a **50% cost discount** and higher rate limits!
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Forgetting to Re-Normalize after Client-Side Slicing**:
      * *Issue*: Slicing `vec[:512]` and directly computing dot products.
      * *Consequence*: Cosine similarity is invalid because the sliced vector length is less than 1.0; retrieval ranks become completely erratic.
      * *Fix*: Always call `vec / np.linalg.norm(vec)` or let the API handle truncation via the `dimensions` parameter.

    * **Trap 2: Mixing Dimensions or Models in the Same Vector Index**:
      * *Issue*: Embedding documents with `text-embedding-ada-002` (1536-dim) and querying with `text-embedding-3-small` (1536-dim) or a 512-dim slice.
      * *Consequence*: Mathematical incompatibility; vector coordinates across different models or dimensions do NOT map to the same manifold.
      * *Fix*: Re-index the entire corpus whenever switching models or target dimensions.

    * **Trap 3: Exceeding the 8,191 Token Limit per Chunk**:
      * *Issue*: Passing a massive document without text splitting to `embeddings.create`.
      * *Consequence*: OpenAI API immediately returns HTTP 400 `invalid_request_error: This model's maximum context length is 8192 tokens`.
      * *Fix*: Enforce token-aware recursive chunking with `tiktoken` (e.g. 500-1000 tokens per chunk) before calling the API.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: text-embedding-3-small**
      * *Definition*: OpenAI's default cost-effective embedding model (1,536 native dimensions, $0.02/1M tokens, 8,191 token window).
    * **Flashcard 2: text-embedding-3-large**
      * *Definition*: OpenAI's frontier embedding model (3,072 native dimensions, $0.13/1M tokens) with state-of-the-art multilingual and code retrieval.
    * **Flashcard 3: Matryoshka Representation Learning (MRL)**
      * *Definition*: A training technique that orders latent coordinates so that early slices can be extracted as smaller, fully functional unit-normalized embeddings.
    * **Flashcard 4: L2 Renormalization**
      * *Definition*: Dividing a truncated vector by its Euclidean length so its norm equals 1.0, enabling fast dot-product cosine similarity.
    """)

st.divider()

# Section 6: Self-Assessment Checklist
st.header("6. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of OpenAI Embedding Models:

- [ ] I can describe the differences between `text-embedding-ada-002`, `text-embedding-3-small`, and `text-embedding-3-large`.
- [ ] I understand how Matryoshka Representation Learning enables dimension truncation from 3,072 to 1,024 or 512.
- [ ] I know why L2 renormalization is mandatory after slicing embedding coordinates.
- [ ] I know how to calculate token costs and storage footprints for an enterprise corpus.
- [ ] I understand production rate limiting strategies including batch packing, exponential backoff, and OpenAI Batch API.
""")
