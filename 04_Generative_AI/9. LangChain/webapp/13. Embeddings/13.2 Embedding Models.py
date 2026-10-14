import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class ProviderBenchmarkResult:
    """Holds benchmarking metrics for a simulated embedding provider."""
    def __init__(self, provider: str, model_id: str, dimensions: int, latency_ms: float, cost_per_1m_tokens_usd: float, local_execution: bool, vector_preview: List[float]):
        self.provider = provider
        self.model_id = model_id
        self.dimensions = dimensions
        self.latency_ms = latency_ms
        self.cost_per_1m_tokens_usd = cost_per_1m_tokens_usd
        self.local_execution = local_execution
        self.vector_preview = vector_preview

class MockMultiProviderEmbeddingStudio:
    """Simulates multi-provider embedding models across cloud APIs and local inference engines."""
    def __init__(self):
        pass

    def benchmark(self, provider_choice: str, text: str, custom_dim: Optional[int] = None) -> ProviderBenchmarkResult:
        tokens = max(1, len(text.split()) * 4)

        if provider_choice == "OpenAI (text-embedding-3-small)":
            dim = custom_dim if custom_dim in [512, 1536] else 1536
            vec = [round(float(np.sin(i * 0.1)), 4) for i in range(8)]
            return ProviderBenchmarkResult("OpenAI", "text-embedding-3-small", dim, 24.5, 0.02, False, vec)
        elif provider_choice == "OpenAI (text-embedding-3-large)":
            dim = custom_dim if custom_dim in [256, 1024, 3072] else 3072
            vec = [round(float(np.cos(i * 0.1)), 4) for i in range(8)]
            return ProviderBenchmarkResult("OpenAI", "text-embedding-3-large", dim, 48.0, 0.13, False, vec)
        elif provider_choice == "Hugging Face (BAAI/bge-large-en-v1.5)":
            vec = [round(float(np.tan(i * 0.05)), 4) for i in range(8)]
            return ProviderBenchmarkResult("Hugging Face", "BAAI/bge-large-en-v1.5", 1024, 65.0, 0.00, True, vec)
        elif provider_choice == "Cohere (embed-english-v3.0)":
            vec = [round(float(np.sin(i * 0.2) * 0.5), 4) for i in range(8)]
            return ProviderBenchmarkResult("Cohere", "embed-english-v3.0", 1024, 32.0, 0.10, False, vec)
        elif provider_choice == "Google (text-embedding-004)":
            vec = [round(float(np.cos(i * 0.2) * 0.7), 4) for i in range(8)]
            return ProviderBenchmarkResult("Google Cloud", "text-embedding-004", 768, 28.0, 0.025, False, vec)
        elif provider_choice == "Local Ollama / FastEmbed (all-MiniLM-L6-v2)":
            vec = [round(float(np.sin(i * 0.3)), 4) for i in range(8)]
            return ProviderBenchmarkResult("Local Engine", "all-MiniLM-L6-v2", 384, 8.5, 0.00, True, vec)
        return ProviderBenchmarkResult("Generic", "default", 768, 30.0, 0.05, False, [0.1] * 8)


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="13.2 Embedding Models",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">13.2 Embedding Models</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparing production embedding providers: OpenAI, Hugging Face, Cohere, Google, and local zero-cost engines.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Modern Embedding Providers & Model Topologies", divider="rainbow")

st.markdown("""
Selecting an embedding model involves critical trade-offs between **retrieval accuracy (MTEB leaderboard rank)**, 
**API cost**, **vector dimensions (RAM footprint)**, and **data sovereignty (cloud API vs on-premises/air-gapped local execution)**.
LangChain standardizes all providers behind a unified `Embeddings` base class, allowing seamless hot-swapping across architectures.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph EmbeddingProviders {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_langchain_interface {
            label="LangChain Standard Interface";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            EmbedBase [label="langchain_core.embeddings.Embeddings\\n• embed_documents(texts: list[str]) -> list[list[float]]\\n• embed_query(text: str) -> list[float]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        }

        subgraph cluster_cloud_providers {
            label="Managed Cloud API Providers";
            style="solid,rounded";
            bgcolor="#0F172A";
            fontcolor="#A855F7";
            fontsize=12;

            OpenAIEmbed [label="OpenAIEmbeddings\\ntext-embedding-3-small (1536D)\\ntext-embedding-3-large (3072D)\\nMatryoshka Truncation Support", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            CohereEmbed [label="CohereEmbeddings\\nembed-english-v3.0 (1024D)\\ninput_type: search_query / search_document\\nint8 / binary compression", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
            GoogleEmbed [label="GoogleGenerativeAIEmbeddings\\ntext-embedding-004 (768D)\\nTaskType: RETRIEVAL_QUERY / DOCUMENT", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC"];
        }

        subgraph cluster_local_providers {
            label="Local & Air-Gapped Private Engines";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            HFEmbed [label="HuggingFaceEmbeddings\\nBAAI/bge-large-en-v1.5 (1024D)\\nsentence-transformers on PyTorch / GPU", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
            LocalOllama [label="OllamaEmbeddings / FastEmbed\\nall-MiniLM-L6-v2 (384D) / ONNX Runtime\\nZero GPU cost, 100% on-premises privacy", fillcolor="#064E3B", fontcolor="#ECFDF5", color="#34D399"];
        }

        EmbedBase -> OpenAIEmbed [label="API Call (HTTPS)"];
        EmbedBase -> CohereEmbed [label="API Call (HTTPS)"];
        EmbedBase -> GoogleEmbed [label="API Call (HTTPS)"];
        EmbedBase -> HFEmbed [label="In-Process (Local CUDA/CPU)"];
        EmbedBase -> LocalOllama [label="Local Daemon (HTTP / ONNX)"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: OPENAI EMBEDDINGS ---
st.header("2. OpenAI Embeddings", divider="rainbow")

st.markdown("""
OpenAI provides industry-standard managed embedding models:
- **`text-embedding-3-small`**: 1536 dimensions, highly cost-effective (0.02 USD per 1M tokens).
- **`text-embedding-3-large`**: 3072 dimensions, top-tier multilingual and code retrieval (0.13 USD per 1M tokens).

### Matryoshka Dimension Truncation:
OpenAI models allow specifying the `dimensions` parameter directly. Truncating `text-embedding-3-large` from 3072 to 1024 
dimensions cuts vector database storage requirements by 66% while outperforming the older `text-embedding-ada-002` model.
""")

st.code("""from langchain_openai import OpenAIEmbeddings

# Standard 1536-dimensional model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Matryoshka truncation to 512 dimensions!
)

vector = embeddings.embed_query("Corporate cloud disaster recovery policy")
print(f"Vector Dimensions: {len(vector)}")  # Output: 512""", language="python")

st.divider()

# --- SECTION 3: HUGGING FACE EMBEDDINGS ---
st.header("3. Hugging Face Embeddings", divider="rainbow")

st.markdown("""
`HuggingFaceEmbeddings` runs state-of-the-art open-weights models locally via `sentence-transformers` on your own GPU or CPU clusters.

### Top Models on MTEB (Massive Text Embedding Benchmark):
- **BAAI/bge-large-en-v1.5**: 1024 dimensions, exceptional semantic accuracy.
- **intfloat/e5-large-v2**: 1024 dimensions, requires `passage: ` and `query: ` prefixes.
- **sentence-transformers/all-MiniLM-L6-v2**: 384 dimensions, blazing fast (5x faster than large models).
""")

st.code("""from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cuda"}  # or 'cpu'
encode_kwargs = {"normalize_embeddings": True}

hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)""", language="python")

st.divider()

# --- SECTION 4: COHERE EMBEDDINGS ---
st.header("4. Cohere Embeddings", divider="rainbow")

st.markdown("""
Cohere's `embed-english-v3.0` and `embed-multilingual-v3.0` introduce two game-changing architectural features:
1. **Asymmetric Input Types**: Must specify `input_type="search_document"` for indexing and `input_type="search_query"` for querying. 
This trains the model to recognize that short queries are semantically seeking detailed documents.
2. **Embedding Compression**: Supports native `int8` and `binary` (1-bit) compression, slashing vector database RAM costs by up to 96%.
""")

st.code("""from langchain_cohere import CohereEmbeddings

cohere_embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    user_agent="enterprise-agent"
)

# Explicit input types for asymmetric retrieval
doc_vectors = cohere_embeddings.embed_documents(
    ["The 2026 expense policy caps daily meals at 85 USD."],
    input_type="search_document"
)
query_vector = cohere_embeddings.embed_query(
    "meal expense limit",
    input_type="search_query"
)""", language="python")

st.divider()

# --- SECTION 5: GOOGLE EMBEDDINGS ---
st.header("5. Google Embeddings", divider="rainbow")

st.markdown("""
Google Cloud provides `text-embedding-004` (768 dimensions) via Gemini APIs and Vertex AI.
It supports specialized task type specifications:
- `RETRIEVAL_QUERY`: Optimizes representation for user search inputs.
- `RETRIEVAL_DOCUMENT`: Optimizes representation for text stored in vector indices.
- `SEMANTIC_SIMILARITY`: Optimizes for text-to-text semantic matching without retrieval asymmetric assumptions.
- `CLASSIFICATION`: Optimizes representation for downstream ML classification tasks.
""")

st.code("""from langchain_google_genai import GoogleGenerativeAIEmbeddings

google_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_document"
)""", language="python")

st.divider()

# --- SECTION 6: LOCAL EMBEDDING MODELS ---
st.header("6. Local Embedding Models", divider="rainbow")

st.markdown("""
For defense, healthcare, and air-gapped banking applications where data cannot leave the private VPC:
- **Ollama Embeddings (`OllamaEmbeddings`)**: Runs models (e.g. `nomic-embed-text`, `bge-m3`) locally via an Ollama daemon.
- **FastEmbed (`FastEmbedEmbeddings`)**: A lightweight Python library backed by ONNX Runtime with zero PyTorch dependencies. 
Generates embeddings 2x to 4x faster than raw Python loops on standard CPU instances.
""")

st.divider()

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Unified Multi-Provider Embedding Factory & Dimension Adapter", divider="rainbow")

st.markdown("""
The following complete Python program implements a factory pattern that instantiates and normalizes embeddings 
across providers while enforcing Matryoshka dimension truncation and in-memory hash caching.
""")

st.code("""import hashlib
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseEmbeddingFactory:
    \"\"\"Factory routing text to cloud providers or local fallback engines.\"\"\"
    def __init__(self, provider: str = "openai", target_dim: Optional[int] = None):
        self.provider = provider.lower()
        self.target_dim = target_dim
        self.cache: Dict[str, List[float]] = {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def embed_text(self, text: str) -> List[float]:
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            return self.cache[text_hash]

        # In production: call actual provider client
        # Here we simulate generation with deterministic pseudorandom vectors
        raw_dim = 1536 if "openai" in self.provider else 768
        rng = np.random.default_rng(abs(hash(text)) % 100000)
        vec = rng.normal(0, 1, raw_dim)

        # Matryoshka Truncation if target_dim specified
        if self.target_dim and self.target_dim < raw_dim:
            vec = vec[:self.target_dim]

        # L2 Normalization
        norm = np.linalg.norm(vec)
        normalized_vec = (vec / norm).tolist()

        self.cache[text_hash] = normalized_vec
        return normalized_vec""", language="python")

st.divider()

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: Multi-Provider Embedding Comparison & Benchmark Studio", divider="rainbow")

st.markdown("""
Compare the latency, pricing, dimensionality, and vector characteristics of major embedding providers in real time.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Select Model & Configuration")
    provider_sel = st.selectbox(
        "Target Embedding Model",
        options=[
            "OpenAI (text-embedding-3-small)",
            "OpenAI (text-embedding-3-large)",
            "Hugging Face (BAAI/bge-large-en-v1.5)",
            "Cohere (embed-english-v3.0)",
            "Google (text-embedding-004)",
            "Local Ollama / FastEmbed (all-MiniLM-L6-v2)"
        ]
    )

    test_input = st.text_area(
        "Sample Input Text",
        value="Cloud multi-region failover protocols require automated DNS routing with health checks.",
        height=100
    )

    matryoshka_dim = None
    if "text-embedding-3-small" in provider_sel:
        matryoshka_dim = st.selectbox("Matryoshka Truncation Dimension", options=[1536, 512], index=0)
    elif "text-embedding-3-large" in provider_sel:
        matryoshka_dim = st.selectbox("Matryoshka Truncation Dimension", options=[3072, 1024, 256], index=0)

    btn_bench = st.button("🚀 Run Provider Benchmark", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Provider Performance Metrics")

    if btn_bench:
        studio = MockMultiProviderEmbeddingStudio()
        res = studio.benchmark(provider_sel, test_input, matryoshka_dim)

        m1, m2, m3 = st.columns(3)
        m1.metric("Vector Dimensions", res.dimensions)
        m2.metric("Inference Latency", f"{res.latency_ms:.1f} ms")
        m3.metric("Cost / 1M Tokens", f"${res.cost_per_1m_tokens_usd:.3f}" if res.cost_per_1m_tokens_usd > 0 else "FREE (Local)")

        st.markdown(f"**Execution Environment**: `{'Private On-Premises (0 Cloud Network I/O)' if res.local_execution else 'Managed Cloud API (HTTPS)'}`")

        st.markdown("#### 🔢 Normalized Vector Slice (First 8 Coordinates):")
        st.code(str(res.vector_preview), language="python")
    else:
        st.info("Select a provider and click **Run Provider Benchmark** to view metrics.")

st.divider()

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("Why does Cohere embed-v3 require specifying input_type ('search_query' vs 'search_document')?",
     "In search systems, user queries are typically short, exploratory questions (e.g. 'how to reset MFA'), while stored documents are long, authoritative answers (e.g. 'Multi-Factor Authentication Reset Procedure: Step 1...'). Symmetrical models project both using the same distribution, leading to sub-optimal alignments. By distinguishing input_type, Cohere projects queries and documents into an asymmetric space optimized specifically for question-answering retrieval."),
    ("How does Matryoshka Representation Learning (MRL) work without retraining the model for smaller dimensions?",
     "During training, the loss function is evaluated simultaneously at multiple sub-vector lengths (e.g. at 256, 512, 1024, and 3072 dimensions). This forces the neural network to compress the most critical, broad semantic representations into the very first coordinates of the vector, while reserving later dimensions for fine-grained nuances. As a result, developers can simply slice the vector array (v[:512]) and re-normalize it with minimal accuracy degradation."),
    ("What are the advantages of FastEmbed over standard sentence-transformers for local embeddings?",
     "FastEmbed is built on ONNX Runtime rather than heavy PyTorch dependencies. It requires no GPU drivers, uses significantly less RAM, and executes quantized models on multi-core CPUs up to 2x to 4x faster than standard Python PyTorch inference loops, making it ideal for containerized serverless functions and lightweight microservices."),
    ("What are the operational challenges of self-hosting Hugging Face embedding models compared to cloud APIs?",
     "Self-hosting requires provisioning GPU hardware (or high-memory CPU nodes), managing inference server frameworks (like vLLM or Triton), configuring auto-scaling based on batch queues, and monitoring memory leaks. In contrast, cloud APIs handle scaling, load balancing, and GPU availability automatically for a per-token fee."),
    ("How does LangChain's CacheBackedEmbeddings prevent duplicate API billing on recurring documents?",
     "CacheBackedEmbeddings wraps any Embeddings client with a key-value store (like Redis, SQLite, or in-memory dict). It computes a SHA-256 hash of the input text and checks the store. If the hash exists, it retrieves the cached vector immediately without calling the external embedding API, eliminating redundant network latency and billing costs.")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Upgrading Embedding Models Without Re-indexing. Swapping an index from text-embedding-ada-002 to text-embedding-3-small without wiping and re-embedding all documents causes 100% retrieval failure. Vectors from different models share zero geometric alignment.")
    st.error("**Trap 2 (Breaking Error)**: Omission of Asymmetric Task Prefixes. Failing to pass task_type='RETRIEVAL_QUERY' to Google embeddings or input_type='search_query' to Cohere degrades retrieval ranking accuracy by up to 25%.")
with trap_c2:
    st.warning("**Trap 3 (Cost Warning)**: Defaulting to 3072D Without Vector Database Capacity Planning. 10 million 3072D vectors in fp32 require 123 GB of pure in-memory RAM in a vector database. Use Matryoshka truncation or 1536D models unless extreme nuance is required.")
    st.warning("**Trap 4 (Latency Warning)**: Single-Item Serial Embedding Calls. Invoking embed_query() one-by-one in a loop for 5,000 documents takes 20 minutes due to HTTP handshake overhead. Always use embed_documents() for batch vectorized execution.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: OpenAI text-embedding-3"):
        st.markdown("**1536D (small) and 3072D (large) models supporting native Matryoshka dimension truncation.**")
    with st.expander("🎴 Flashcard 2: Asymmetric Input Types"):
        st.markdown("**Cohere and Google require designating search_query vs search_document for optimal alignment.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: FastEmbed"):
        st.markdown("**Lightweight ONNX Runtime engine running embeddings locally on CPU with zero PyTorch bloat.**")
    with st.expander("🎴 Flashcard 4: CacheBackedEmbeddings"):
        st.markdown("**Key-value caching wrapper preventing redundant API calls and duplicate billing.**")

st.divider()

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between OpenAI, Hugging Face, Cohere, Google, and Local embeddings.", value=True)
chk2 = st.checkbox("I know how to implement Matryoshka dimension truncation to reduce vector database RAM footprint.", value=True)
chk3 = st.checkbox("I understand why asymmetric input types are critical in Cohere and Google models.", value=True)
chk4 = st.checkbox("I can deploy zero-cloud local embedding engines using Ollama and FastEmbed.", value=True)
chk5 = st.checkbox("I know how CacheBackedEmbeddings prevents redundant API token billing.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 embedding model interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 13.2 Embedding Models!")
