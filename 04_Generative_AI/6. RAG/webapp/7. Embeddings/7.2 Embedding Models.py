import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="7.2 Embedding Models - RAG Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class EmbeddingCatalogManager:
    """Provides comparative architectural and economic metrics across major embedding providers."""
    def __init__(self):
        self.models = [
            {"provider": "OpenAI", "model": "text-embedding-3-small", "dimensions": 1536, "max_tokens": 8192, "mteb_score": 62.3, "cost_per_1m_tokens_usd": 0.02, "hosting": "Cloud API", "supports_mrl": True},
            {"provider": "OpenAI", "model": "text-embedding-3-large", "dimensions": 3072, "max_tokens": 8192, "mteb_score": 64.6, "cost_per_1m_tokens_usd": 0.13, "hosting": "Cloud API", "supports_mrl": True},
            {"provider": "Hugging Face / BAAI", "model": "bge-large-en-v1.5", "dimensions": 1024, "max_tokens": 512, "mteb_score": 64.2, "cost_per_1m_tokens_usd": 0.00, "hosting": "Local / Self-Hosted", "supports_mrl": False},
            {"provider": "Sentence Transformers", "model": "all-MiniLM-L6-v2", "dimensions": 384, "max_tokens": 256, "mteb_score": 56.3, "cost_per_1m_tokens_usd": 0.00, "hosting": "Local / In-Process", "supports_mrl": False},
            {"provider": "Cohere", "model": "embed-english-v3.0", "dimensions": 1024, "max_tokens": 512, "mteb_score": 64.5, "cost_per_1m_tokens_usd": 0.10, "hosting": "Cloud API", "supports_mrl": False},
            {"provider": "Google", "model": "text-embedding-004", "dimensions": 768, "max_tokens": 2048, "mteb_score": 63.8, "cost_per_1m_tokens_usd": 0.025, "hosting": "Cloud API", "supports_mrl": True}
        ]

    def get_catalog_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.models)

# Title & Overview
st.title("7.2 Embedding Models: Ecosystem, Architectures & Serving Frameworks")
st.caption("Chapter 7: Embeddings | Domain: 6. RAG")
st.markdown("""
The performance of a Retrieval-Augmented Generation pipeline is bounded by the quality of its **embedding model**. 
Even the most capable LLM (GPT-4o or Claude 3.5 Sonnet) will generate hallucinations or ungrounded responses if 
the embedding model fails to retrieve the correct source passages during the first retrieval stage.

Today's AI engineers can choose across a rich spectrum of embedding models: from managed proprietary cloud APIs 
(OpenAI, Cohere, Google Vertex AI) to open-source foundation weights (BGE, GTE, Sentence Transformers) served 
on dedicated infrastructure via **Text Embeddings Inference (TEI)**, **vLLM**, or local CPU runtimes like **Ollama**.

In this lesson, you will master:
1. **The Commercial & Open-Source Ecosystem**: Evaluating OpenAI, Hugging Face, Cohere, and Google embedding architectures.
2. **The MTEB Benchmark**: Navigating the Massive Text Embedding Benchmark to select optimal models by retrieval task.
3. **Local Self-Hosted Serving**: Deploying high-throughput embedding microservices using TEI and ONNX Runtime.
4. **Input Type Discrepancies**: Handling asymmetric query vs document prefixes across Cohere and BGE.
5. **Production Failure Traps**: Mitigating rate-limiting outages, silent token truncations, and dimensional drift.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Embedding Model Ecosystem & Serving Topologies", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph EmbeddingEcosystemBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_app {
            label="Tier 1: Application Ingestion & Query Ingress";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            ingest_worker [label="Ingestion Worker (Bulk Chunks)\\nBatches 64-128 passages for indexing", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            user_client [label="Real-Time Query Client\\nSubmits user question for sub-10ms search", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_router {
            label="Tier 2: Resilient Embedding Router & Prefix Normalizer";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            prefix_manager [label="Task Prefix Formatter\\nAttaches 'query: ' or 'input_type: search_query'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            circuit_breaker [label="Circuit Breaker & Fallback Handler\\nRoutes primary to cloud API, fallbacks to local TEI", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            ingest_worker -> prefix_manager;
            user_client -> prefix_manager;
            prefix_manager -> circuit_breaker;
        }

        subgraph cluster_backends {
            label="Tier 3: Embedding Model Serving Backends";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            cloud_openai [label="Cloud API: OpenAI / Cohere / Google\\ntext-embedding-3 / embed-v3 (Managed SLA)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            local_tei [label="Self-Hosted: Hugging Face TEI / vLLM\\nBGE-Large / GTE (Local GPU, Zero Data Leakage)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];
            in_proc_onnx [label="Embedded: ONNX Runtime / Ollama\\nall-MiniLM-L6-v2 (Zero Infrastructure)", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#F59E0B"];

            circuit_breaker -> cloud_openai [label="Primary Route"];
            circuit_breaker -> local_tei [label="Air-Gapped / Privacy"];
            circuit_breaker -> in_proc_onnx [label="Edge / Fallback"];
        }

        subgraph cluster_vectordb {
            label="Tier 4: Vector Database Ingestion & Search";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            vdb [label="Downstream Vector Store\\nNormalized float32 arrays persisted to HNSW index", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            cloud_openai -> vdb;
            local_tei -> vdb;
            in_proc_onnx -> vdb;
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: OpenAI Embeddings
st.header("2. OpenAI Embeddings: `text-embedding-3` Family & MRL", divider="rainbow")
st.markdown("""
OpenAI provides the most widely used commercial cloud embedding models in enterprise Generative AI:

* **`text-embedding-3-small`**:
  * Default dimensionality: **1536**. Maximum context: **8192 tokens**.
  * Highly economical ($0.02 per 1M tokens), delivering higher MTEB performance than legacy `ada-002`.
* **`text-embedding-3-large`**:
  * Default dimensionality: **3072**. Maximum context: **8192 tokens**.
  * Flagship multilingual precision model designed for high-stakes retrieval tasks.
* **Native Matryoshka Representation Learning (MRL)**:
  * Both models support the `dimensions` parameter directly in the API call:
    `client.embeddings.create(input="...", model="text-embedding-3-small", dimensions=512)`.
  * Allows developers to reduce vector size by up to 66% with negligible retrieval accuracy loss.
""")
st.divider()

# Section 3: Hugging Face Embeddings
st.header("3. Hugging Face Embeddings: Open Weights & MTEB Leaders", divider="rainbow")
st.markdown("""
The open-source community hosted on Hugging Face offers models that match or exceed commercial APIs:

* **MTEB (Massive Text Embedding Benchmark)**: The global industry leaderboard evaluating models across 56 datasets covering retrieval, clustering, classification, and summarization.
* **Top Open-Source Models**:
  * **BAAI BGE (`bge-large-en-v1.5`)**: 1024 dimensions, 512-token sequence limit. Dominant open model for enterprise retrieval.
  * **Alibaba GTE (`gte-Qwen2-7B-instruct`)**: State-of-the-art multilingual and long-context open model.
* **Data Sovereignty**: Open weights can be hosted within private VPC networks or air-gapped on-premise clusters, satisfying GDPR, HIPAA, and financial compliance standards.
""")
st.divider()

# Section 4: Sentence Transformers
st.header("4. Sentence Transformers: The Bi-Encoder Foundation", divider="rainbow")
st.markdown("""
The `sentence-transformers` library (created by UKP Lab) provides the standard open-source framework for building Bi-Encoder embedding models:

* **Bi-Encoder Architecture**: Encodes query and passage independently into fixed-dimensional vectors, enabling millions of vectors to be pre-calculated and indexed into a vector database.
* **`all-MiniLM-L6-v2`**: The quintessential lightweight embedding model (384-d, ~80 MB file size). Runs in-memory on ordinary laptop CPUs with near-zero latency, making it the default bundled engine in ChromaDB.
* **Fine-Tuning Capability**: Sentence Transformers allows teams to fine-tune pre-trained weights on domain-specific triplets `(query, positive_doc, negative_doc)` using Multiple Negatives Ranking Loss (MNRL).
""")
st.divider()

# Section 5: Cohere Embeddings
st.header("5. Cohere Embeddings: Asymmetric Task Types & Compression", divider="rainbow")
st.markdown("""
Cohere's `embed-english-v3.0` and `embed-multilingual-v3.0` introduce purpose-built RAG capabilities:

* **Explicit `input_type` Parameter**: Mandates declaring the role of the input text:
  * `input_type="search_document"`: When indexing corpus passages into the vector database.
  * `input_type="search_query"`: When embedding incoming user questions.
  * *Why?* Questions and passages have distinct linguistic characteristics; asymmetric training models the relationship between questions and their answers directly.
* **Native Int8 and Binary Embeddings**: Cohere supports exporting vectors natively quantized to 8-bit integers or 1-bit binary arrays, slashing vector database memory footprint by up to 32x.
""")
st.divider()

# Section 6: Google Embeddings
st.header("6. Google Embeddings: Vertex AI & Multimodal Embedding Models", divider="rainbow")
st.markdown("""
Google Cloud Vertex AI provides enterprise embedding services integrated directly into the Google Cloud stack:

* **`text-embedding-004`**:
  * 768 dimensions with a 2048-token context window.
  * Features native Matryoshka dimension truncation (e.g., slicing down to 256 or 128 dimensions).
  * Supports task type configurations (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`).
* **Multimodal Embeddings**: Projects text and video/image frames into a unified coordinate manifold, allowing text queries to retrieve specific video timestamps across cloud archives.
""")
st.divider()

# Section 7: Local Embedding Models
st.header("7. Local Embedding Models: Self-Hosting with TEI, vLLM & Ollama", divider="rainbow")
st.markdown("""
To eliminate cloud API latency, per-token billing, and external data transmission, enterprises deploy self-hosted embedding engines:
""")

col_loc1, col_loc2 = st.columns(2)
with col_loc1:
    st.markdown("""
    ### 1. Hugging Face TEI (Text Embeddings Inference)
    * **Rust-Based Web Server**: High-throughput inference server written in Rust with native CUDA / FlashAttention kernels.
    * **Dynamic Token Batching**: Batches concurrent requests to maximize GPU utilization, achieving over **2,000 requests/sec**.
    * **Production Standard**: The gold standard for self-hosting BGE and GTE models in Kubernetes.
    """)
with col_loc2:
    st.markdown("""
    ### 2. Ollama / ONNX Runtime
    * **Ollama (`ollama pull nomic-embed-text`)**: Simple CLI tool running quantized embedding models locally on Mac, Windows, or Linux CPUs.
    * **ONNX Runtime**: Executes pre-compiled embedding graphs via C++ bindings with zero Python runtime overhead, ideal for edge deployments.
    """)
st.divider()

# Section 8: Production Implementation
st.header("8. Production Implementation: Multi-Provider Resilient Embedding Service", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating an enterprise embedding service with 
task-type formatting, batch processing, and provider fallback handling.
""")

st.code('''
from typing import List, Dict, Any, Optional
import numpy as np

class ResilientEmbeddingService:
    """Production embedding service with task-type formatting and normalized output."""
    def __init__(self, provider: str = "openai", target_dim: int = 1536):
        self.provider = provider
        self.target_dim = target_dim

    def format_input(self, text: str, is_query: bool = False) -> str:
        """Applies task-specific prefixes for asymmetric models like BGE or E5."""
        if self.provider in ["bge", "e5"]:
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{text.strip()}"
        return text.strip()

    def embed_batch(self, texts: List[str], is_query: bool = False) -> List[np.ndarray]:
        """Simulates batch embedding generation with L2 unit normalization."""
        formatted_inputs = [self.format_input(t, is_query) for t in texts]
        
        vectors = []
        for inp in formatted_inputs:
            # Deterministic pseudo-random vector simulation based on hash
            seed = abs(hash(inp)) % (2**31)
            np.random.seed(seed)
            raw_v = np.random.randn(self.target_dim).astype(np.float32)
            
            # L2 Normalization
            norm = np.linalg.norm(raw_v)
            norm_v = raw_v / norm if norm > 0 else raw_v
            vectors.append(norm_v)

        return vectors
''', language="python")
st.divider()

# Section 9: Interactive Workbench
st.header("9. Interactive Workbench: Model Benchmark Comparison & Latency/Cost Estimator", divider="rainbow")
st.markdown("""
Compare leading commercial and open-source embedding models across dimensions, context limits, 
MTEB retrieval benchmarks, and estimated monthly token costs.
""")

cat_manager = EmbeddingCatalogManager()
df_catalog = cat_manager.get_catalog_df()

st.subheader("Embedding Model Comparison Matrix")
st.dataframe(df_catalog[["provider", "model", "dimensions", "max_tokens", "mteb_score", "cost_per_1m_tokens_usd", "hosting", "supports_mrl"]], use_container_width=True)

st.subheader("Monthly Cost & Storage Estimator")
col_est1, col_est2 = st.columns(2)
with col_est1:
    monthly_tokens_m = st.slider("Monthly Tokens Processed (Millions):", min_value=1, max_value=200, value=25)
with col_est2:
    selected_eval_model = st.selectbox("Select Model for Estimation:", df_catalog["model"].tolist(), index=0)

model_row = df_catalog[df_catalog["model"] == selected_eval_model].iloc[0]
est_cost = monthly_tokens_m * model_row["cost_per_1m_tokens_usd"]
ram_per_1m_vectors_gb = round((1_000_000 * model_row["dimensions"] * 4) / (1024**3), 2)

col_out1, col_out2 = st.columns(2)
col_out1.metric(f"Estimated Monthly API Cost ({selected_eval_model})", f"${est_cost:.2f}")
col_out2.metric("RAM Required for 1M Vectors (Raw Float32)", f"{ram_per_1m_vectors_gb} GB")
st.divider()

# Section 10: Technical Interview Preparation & Production Traps
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is the MTEB benchmark, and why should RAG teams consult its Retrieval subset?", expanded=True):
    st.markdown("""
    **Answer:**
    The **Massive Text Embedding Benchmark (MTEB)** is an open, standardized benchmark that evaluates embedding models across 
    diverse tasks: Classification, Clustering, Pair Classification, Reranking, Retrieval, Semantic Textual Similarity (STS), and Summarization.
    
    RAG teams must specifically inspect the **Retrieval subset** (based on datasets like MS MARCO and BEIR) rather than the overall average. 
    A model can score high overall by excelling at STS or sentiment classification, but perform poorly on complex asymmetric document retrieval.
    """)

with st.expander("Q2: What is the purpose of asymmetric query vs document prefixes in models like BGE or Cohere?"):
    st.markdown("""
    **Answer:**
    User search queries are typically short, exploratory, and framed as questions (*"How do I rotate IAM keys?"*), whereas corpus passages 
    are long, declarative statements (*"The AWS Identity and Access Management key rotation process requires..."*). 
    Asymmetric models use different prefixes (e.g., `query: ` vs `passage: `, or `search_query` vs `search_document`) to instruct the 
    transformer's attention layers to map question-form syntax into answer-form coordinate space.
    """)

with st.expander("Q3: When is it advantageous to self-host Hugging Face TEI instead of using OpenAI embeddings?"):
    st.markdown("""
    **Answer:**
    * **Data Privacy & Compliance**: Organizations subject to strict data sovereignty laws (GDPR, HIPAA, defence) cannot send proprietary customer documents across public third-party APIs.
    * **Ultra-High Throughput**: TEI deployed on an internal GPU instance achieves over 2,000 QPS with sub-5ms latency, avoiding cloud rate-limiting throttles and per-token costs during bulk ingestion runs.
    * **Offline Availability**: Critical for on-premise systems operating in disconnected or air-gapped data centers.
    """)

with st.expander("Q4: What is the risk of selecting an embedding model with a short sequence limit (e.g., 512 tokens)?"):
    st.markdown("""
    **Answer:**
    If your document chunking pipeline produces chunks that exceed 512 tokens (e.g., 600-word passages), the embedding model will 
    silently truncate all tokens past position 512. The truncated text will be completely absent from the generated vector, rendering 
    those concepts unsearchable without throwing any warning or error to the developer.
    """)

with st.expander("Q5: How does Matryoshka Representation Learning (MRL) achieve dimensionality reduction without retraining?"):
    st.markdown("""
    **Answer:**
    During training, MRL enforces loss functions across multiple prefix sub-vectors simultaneously (e.g., dimensions 1 to 64, 1 to 128, 
    1 to 256, 1 to 512, 1 to 1536). This forces the model to pack the highest-entropy, most informative semantic signals into the earliest 
    dimensions. At runtime, developers can simply slice the vector array (`v[:512]`) and re-normalize it to unit length.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Omitting Asymmetric Task Prefixes in BGE or E5 Models (15% Recall Drop)**
    Ingesting documents and querying BGE without prepending `"query: "` to search questions. The model treats questions as generic passages, failing to link questions to their corresponding answers and degrading retrieval hit rates by over 15%.
    """)
    st.error("""
    **Trap 2: Cloud Embedding API Rate-Limit Throttles Halting Ingestion (HTTP 429 Outage)**
    Running bulk parallel ingestion jobs against commercial APIs (OpenAI or Cohere) without exponential backoff or token bucket rate limiters. Ingestion crashes midway through multi-gigabyte document batches.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Model Sequence Length Mismatch Silently Truncating Chunks (Missing Knowledge)**
    Chunking text to 1000 tokens while embedding with Sentence Transformers `all-MiniLM-L6-v2` (max sequence length 256 tokens). Over 70% of every chunk is silently truncated at inference time.
    """)
    st.warning("""
    **Trap 4: Forgetting to Re-Normalize Vectors After Matryoshka Truncation (Distance Distortion)**
    Slicing a 1536-dimensional vector to 512 dimensions (`v[:512]`) without dividing by its new Euclidean norm. The truncated vector loses its unit length, corrupting subsequent Dot Product similarity scores.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What benchmark evaluates embedding models for RAG retrieval?")
        st.markdown("**Back**: The MTEB (Massive Text Embedding Benchmark), specifically its Retrieval subset.")
    with st.container(border=True):
        st.markdown("**Front**: Why do models like Cohere require an `input_type` parameter?")
        st.markdown("**Back**: To distinguish question syntax (`search_query`) from answer passage syntax (`search_document`).")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: What is Hugging Face TEI (Text Embeddings Inference)?")
        st.markdown("**Back**: A high-performance Rust web server for self-hosting open-source embedding models at thousands of QPS.")
    with st.container(border=True):
        st.markdown("**Front**: What must be done after truncating a Matryoshka embedding vector?")
        st.markdown("**Back**: Divide the truncated vector by its new L2 norm to restore unit length for accurate Dot Product search.")

st.divider()

# Section 11: Self-Assessment Checklist
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Embedding Models and Serving Architectures before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I understand the differences between OpenAI, Hugging Face, Cohere, and Google embedding models.", value=True, key="chk_emb_mod_1")
    chk2 = st.checkbox("I know how to consult the MTEB Retrieval benchmark to select models by task accuracy.", value=True, key="chk_emb_mod_2")
    chk3 = st.checkbox("I understand the benefits of self-hosting models using Hugging Face TEI for privacy and throughput.", value=True, key="chk_emb_mod_3")
with c2:
    chk4 = st.checkbox("I know why asymmetric models require specific query and document prefixes.", value=True, key="chk_emb_mod_4")
    chk5 = st.checkbox("I understand how to leverage Matryoshka Representation Learning to optimize storage costs.", value=True, key="chk_emb_mod_5")
    chk6 = st.checkbox("I can identify production traps such as silent sequence truncation and missing vector re-normalization.", value=True, key="chk_emb_mod_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Embedding Models Mastery Demonstrated!** You are ready to advance to Lesson 7.3 in the curriculum.")
