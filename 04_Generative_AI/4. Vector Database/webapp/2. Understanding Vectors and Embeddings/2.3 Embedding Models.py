import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="2.3 Embedding Models - Vector DB Curriculum",
    page_icon="🤖",
    layout="wide"
)

# Title & Overview
st.title("2.3 Embedding Models: Cloud APIs, Local Open-Weights, Multilingual & Domain Encoders")
st.caption("Chapter 2: Understanding Vectors and Embeddings | Domain: 4. Vector Database")
st.markdown("""
The performance and retrieval accuracy of any vector database is fundamentally bounded by the **quality of the embedding model**.
If the embedding model generates noisy, poorly separated representations, no vector database indexing algorithm—no matter 
how sophisticated—can recover the true semantic nearest neighbors.

In production architecture, selecting an embedding model involves critical trade-offs between **retrieval recall (MTEB score)**, 
**inference latency**, **operational cost**, **data privacy (HIPAA/GDPR)**, and **hardware footprint**. 
This lesson covers the major commercial cloud APIs (OpenAI, Cohere), open-weights models (Sentence-Transformers, BGE, GTE, E5), 
local deployment runtimes (ONNX, FastEmbed, Ollama), multilingual architectures, and domain-specialized encoders.
""")

st.divider()

# Section 1: The Modern Embedding Model Landscape
st.header("1. The Modern Embedding Model Landscape & MTEB Benchmark", divider="rainbow")

st.markdown("""
The AI community evaluates embedding model quality using the **Massive Text Embedding Benchmark (MTEB)**, 
which measures retrieval, classification, clustering, pair classification, reranking, and summarization across 56+ datasets.
""")

col_mteb1, col_mteb2 = st.columns([1, 1])

with col_mteb1:
    st.markdown("""
    ### 1. Commercial Cloud API Models
    * **OpenAI `text-embedding-3-small`**:
      * *Dimensions*: 1536 (supports MRL truncation down to 512).
      * *Context Window*: 8,191 tokens.
      * *Cost*: $0.020 per 1M tokens.
      * *Verdict*: Default workhorse for general-purpose applications; low operational complexity.
    * **OpenAI `text-embedding-3-large`**:
      * *Dimensions*: 3072 (MRL down to 1024 or 256).
      * *Cost*: $0.130 per 1M tokens.
      * *Verdict*: Top-tier multilingual recall; higher vector DB storage footprint.
    * **Cohere `embed-v3` (multilingual / english)**:
      * *Dimensions*: 1024.
      * *Special Feature*: Built-in compression to `int8` and `ubinary` embeddings with native SDK support.
    """)

with col_mteb2:
    st.markdown("""
    ### 2. Leading Open-Weights Models
    * **BAAI `BGE-M3` (FlagEmbedding)**:
      * *Dimensions*: 1024.
      * *Tri-Mode Retrieval*: Outputs Dense, Sparse (lexical weights), and Multi-vector (ColBERT late interaction) simultaneously!
      * *Multilingual*: 100+ languages, 8,192 token context.
    * **`BAAI/bge-large-en-v1.5`**:
      * *Dimensions*: 1024. Long-standing king of English retrieval benchmarks.
    * **`Alibaba/gte-Qwen2-7B-instruct`**:
      * *Dimensions*: 3584. SOTA MTEB scores, handles 32k context, requires GPU.
    * **`nomic-ai/nomic-embed-text-v1.5`**:
      * *Dimensions*: 768. 8k context, fully open weights and open data, MRL capable.
    * **Sentence Transformers (`all-MiniLM-L6-v2`)**:
      * *Dimensions*: 384. Ultra-fast CPU inference (14,000 sentences/sec on CPU).
    """)

st.markdown("### Comparative Architecture & Benchmark Matrix")
model_benchmark_df = pd.DataFrame({
    "Model Name": [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "bge-base-en-v1.5",
        "bge-m3",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "gte-Qwen2-7B-instruct"
    ],
    "Provider / Architecture": [
        "Sentence-Transformers",
        "Sentence-Transformers",
        "BAAI (Open Weights)",
        "BAAI (Tri-Modal)",
        "OpenAI Cloud API",
        "OpenAI Cloud API",
        "Alibaba (LLM-based)"
    ],
    "Dimensions": [384, 768, 768, 1024, 1536, 3072, 3584],
    "Context (Tokens)": [256, 384, 512, 8192, 8191, 8191, 32768],
    "MTEB Retrieval Score": ["56.3", "60.4", "63.5", "64.8", "62.3", "64.6", "70.2"],
    "Inference Speed / Cost": [
        "Ultra-Fast (CPU)",
        "Fast (CPU/GPU)",
        "Fast (CPU/GPU)",
        "Moderate (GPU)",
        "$0.02 / 1M tokens",
        "$0.13 / 1M tokens",
        "Heavy GPU (7B params)"
    ],
    "Matryoshka (MRL)": ["No", "No", "No", "Yes", "Yes", "Yes", "Yes"]
})
st.dataframe(model_benchmark_df, use_container_width=True, hide_index=True)

st.divider()

# Section 2: Cloud API vs. Self-Hosted Local Deployment
st.header("2. Cloud APIs vs. Self-Hosted Local Inference", divider="rainbow")

st.markdown("""
Deciding whether to call a managed cloud API or self-host open-weights embedding models is a primary 
architectural crossroads in enterprise system design.
""")

col_dep1, col_dep2 = col_dep1, col_dep2 = st.columns([1, 1])

with col_dep1:
    st.markdown("""
    ### Trade-Off Breakdown
    * **Cloud APIs (OpenAI, Cohere, Voyage)**:
      * **Pros**: Zero infrastructure management, automatic scaling, immediate setup, state-of-the-art accuracy.
      * **Cons**: Data egress to third parties, recurring per-token billing, rate limit throttling during bulk data ingestion, network latency (50-200ms per batch).
    * **Self-Hosted Local Encoders (FastEmbed, ONNX, vLLM, Ollama)**:
      * **Pros**: 100% data privacy (zero egress for HIPAA/SOC2/GDPR), zero per-token marginal cost, deterministic sub-10ms latency over local sockets, no external rate limits.
      * **Cons**: Requires GPU/CPU hardware management, memory allocation, container orchestration, and model versioning.
    """)

with col_dep2:
    st.markdown("### Deployment Architecture Comparison")
    dep_dot = """
    digraph DeploymentArchitecture {
        rankdir=LR;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        AppServer [label="Enterprise App Server\\nIngestion / Search Service", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        subgraph cluster_cloud {
            label="Cloud API Paradigm"; style="dashed"; color="#EF4444";
            CloudGateway [label="Internet HTTPS Gateway\\nTLS Latency (60-150ms)", fillcolor="#FEE2E2"];
            OpenAIAPI [label="OpenAI / Cohere API\\nRate Limits: 10k TPM\\nCost: Per 1M Tokens", fillcolor="#FECACA", color="#DC2626", style="filled,bold"];
        }

        subgraph cluster_local {
            label="Self-Hosted Local Paradigm"; style="dashed"; color="#10B981";
            LocalRuntime [label="Local ONNX / FastEmbed\\nSIMD AVX-512 / GPU CUDA", fillcolor="#D1FAE5"];
            SharedMem [label="Zero-Copy IPC / Unix Socket\\nSub-5ms Latency\\n100% On-Prem Privacy", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        VectorDB [label="Vector Database\\n(Chroma / Qdrant / Milvus)", fillcolor="#EDE9FE", color="#7C3AED", style="filled,bold"];

        AppServer -> CloudGateway -> OpenAIAPI -> VectorDB;
        AppServer -> LocalRuntime -> SharedMem -> VectorDB;
    }
    """
    st.graphviz_chart(dep_dot, use_container_width=False)

st.divider()

# Section 3: Multilingual & Domain-Specific Embeddings
st.header("3. Multilingual Alignment & Domain-Specific Encoders", divider="rainbow")

st.markdown("""
Generic general-domain English models degrade rapidly when faced with specialized vocabularies or foreign languages:
""")

col_spec1, col_spec2 = col_spec1, col_spec2 = st.columns([1, 1])

with col_spec1:
    st.markdown("""
    ### 1. Multilingual Joint Encoders
    * **The Cross-Lingual Alignment Mechanism**:
      * Trained on parallel corpora across 100+ languages using translation contrastive loss.
      * The vector for `"Machine Learning"` (English) is geometrically aligned with `"Aprendizaje automático"` (Spanish), `"Maschinelles Lernen"` (German), and `"मशीन लर्निंग"` (Hindi).
    * **Enterprise Use Cases**:
      * Global knowledge bases where documentation is authored in English, but queries arrive in Japanese, French, or Arabic.
      * Cross-border customer support ticket deduplication.
    * **Leading Models**: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `Cohere embed-multilingual-v3.0`.
    """)

with col_spec2:
    st.markdown("""
    ### 2. Domain-Specific Specialized Encoders
    * **Why Generic Models Struggle**:
      * In general English, `"acute"` means `"sharp or severe"`. In clinical medicine, `"acute myocardial infarction"` is an emergency cardiac diagnosis.
      * In finance, `"equity"` means `"stocks and ownership"`, not fairness.
      * In programming, syntax symbols (`->`, `::`, `+=`) carry critical structural logic that standard tokenizers discard.
    * **Domain Families**:
      * **Biomedical & Clinical**: `BioBERT`, `Med-Embed`, `Clinical-BERT`.
      * **Finance & Sec Filings**: `FinBERT`, `Fin-Embed-Large`.
      * **Legal & Contracts**: `LegalBERT`, `Law2Vec`.
      * **Code & Software**: `jina-embeddings-v2-base-code`, `StarCoder-Embed`.
    """)

st.divider()

# Section 4: Interactive Model Selector & Cost / Throughput Calculator
st.header("4. Interactive Model Selector & TCO Cost Calculator", divider="rainbow")

st.markdown("""
Configure your application's data volume, query velocity, and privacy requirements to calculate 
monthly cloud API costs versus self-hosted compute costs, and receive an architectural recommendation:
""")

col_calc_cfg, col_calc_res = st.columns([1, 1])

with col_calc_cfg:
    st.subheader("Workload & Sizing Parameters")
    monthly_docs = st.select_slider(
        "Monthly Ingestion Document Volume",
        options=[1000, 10000, 50000, 200000, 1000000, 5000000],
        value=50000,
        format_func=lambda x: f"{x:,} docs"
    )
    tokens_per_doc = st.slider("Average Tokens per Document", 100, 2000, 500, 50)
    monthly_queries = st.select_slider(
        "Monthly Search Queries",
        options=[5000, 50000, 250000, 1000000, 5000000, 20000000],
        value=250000,
        format_func=lambda x: f"{x:,} queries"
    )
    tokens_per_query = st.slider("Average Tokens per Query", 5, 100, 20, 5)
    
    compliance_need = st.checkbox("Strict Privacy Compliance Required (Zero Data Egress / HIPAA / On-Prem)", value=False)
    multilingual_need = st.checkbox("Requires Multilingual Cross-Lingual Search (Non-English)", value=False)

with col_calc_res:
    st.subheader("Financial & Infrastructure Analysis")
    
    total_ingest_tokens = monthly_docs * tokens_per_doc
    total_query_tokens = monthly_queries * tokens_per_query
    total_monthly_tokens = total_ingest_tokens + total_query_tokens
    
    # Costs
    cost_openai_small = (total_monthly_tokens / 1_000_000) * 0.02
    cost_openai_large = (total_monthly_tokens / 1_000_000) * 0.13
    
    # Self-hosted compute estimate: 1x T4 GPU VM ($180/mo) or CPU VM ($40/mo)
    if total_monthly_tokens > 500_000_000:
        cost_self_hosted = 360.0 # 2x GPU instances
        infra_spec = "2x AWS g4dn.xlarge (NVIDIA T4 GPUs)"
    elif total_monthly_tokens > 50_000_000:
        cost_self_hosted = 180.0 # 1x GPU instance
        infra_spec = "1x AWS g4dn.xlarge (NVIDIA T4 GPU)"
    else:
        cost_self_hosted = 45.0 # CPU instance with AVX-512 FastEmbed
        infra_spec = "1x AWS c6i.xlarge (Intel Xeon AVX-512 CPU)"
        
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Total Monthly Tokens", f"{total_monthly_tokens:,.0f}", f"Ingest: {total_ingest_tokens/1e6:.1f}M | Q: {total_query_tokens/1e6:.1f}M")
        st.metric("OpenAI Small Cost", f"${cost_openai_small:.2f} / mo")
    with m2:
        st.metric("OpenAI Large Cost", f"${cost_openai_large:.2f} / mo")
        st.metric("Self-Hosted Cloud Compute", f"${cost_self_hosted:.2f} / mo", infra_spec)
        
    st.markdown("#### Architectural Recommendation")
    if compliance_need:
        rec_model = "**BGE-M3** or **bge-large-en-v1.5** via **FastEmbed / ONNX Runtime**"
        rec_rationale = "Zero data egress required. Self-hosted local inference guarantees 100% on-prem compliance."
    elif multilingual_need:
        if cost_openai_large < cost_self_hosted:
            rec_model = "**OpenAI text-embedding-3-large** or **Cohere embed-multilingual-v3.0**"
            rec_rationale = "Low token volume makes Cloud APIs significantly cheaper than dedicated cloud GPU instances."
        else:
            rec_model = "**BAAI BGE-M3** self-hosted on GPU"
            rec_rationale = "High token volume makes dedicated GPU self-hosting far more economical than per-token cloud API billing."
    else:
        if cost_openai_small < 50.0:
            rec_model = "**OpenAI text-embedding-3-small** (1536 dims, MRL truncated to 512)"
            rec_rationale = "Negligible cloud cost (< $50/mo) with zero server maintenance overhead."
        else:
            rec_model = "**bge-base-en-v1.5** via **FastEmbed CPU (ONNX)**"
            rec_rationale = "High throughput makes CPU-optimized local inference vastly cheaper than API calls."
            
    st.success(f"**Recommended Model:** {rec_model}\n\n*Rationale:* {rec_rationale}")

st.divider()

# Section 5: Executable Python Implementation
st.header("5. Executable Code: Unified Embedding Provider Abstraction", divider="rainbow")

st.markdown("""
In enterprise applications, your codebase should never be hardcoded to a single embedding vendor. 
The following modular architecture defines an abstract `BaseEmbeddingProvider` supporting pluggable 
Cloud API and Local Open-Weights providers with automated batching and vector normalization:
""")

code_embed_provider = '''import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbeddingProvider(ABC):
    """Abstract interface for all embedding providers."""
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        pass

class MockOpenAIProvider(BaseEmbeddingProvider):
    """Simulates OpenAI text-embedding-3-small with MRL truncation support."""
    def __init__(self, target_dim: int = 1536):
        self._dimension = target_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_single(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % (2**32))
        full_vec = np.random.randn(1536)
        # Apply MRL truncation if target_dim < 1536
        truncated = full_vec[:self._dimension]
        norm = np.linalg.norm(truncated)
        return truncated / (norm if norm > 0 else 1.0)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        return [self._encode_single(t) for t in texts]

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode_single(query)

class MockLocalBGEProvider(BaseEmbeddingProvider):
    """Simulates local BAAI BGE-M3 model with asymmetric prefix handling."""
    def __init__(self):
        self._dimension = 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_single(self, text: str, prefix: str = "") -> np.ndarray:
        full_input = f"{prefix}{text}"
        np.random.seed(abs(hash(full_input)) % (2**32))
        raw_vec = np.random.randn(1024)
        norm = np.linalg.norm(raw_vec)
        return raw_vec / (norm if norm > 0 else 1.0)

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        # Symmetric passage encoding
        return [self._encode_single(t, prefix="passage: ") for t in texts]

    def embed_query(self, query: str) -> np.ndarray:
        # Asymmetric query encoding
        return self._encode_single(query, prefix="query: ")

# Demonstration
openai_provider = MockOpenAIProvider(target_dim=512) # Truncated to 512 via MRL
bge_provider = MockLocalBGEProvider()

sample_doc = "Vector databases deliver sub-10ms nearest neighbor search."
q = "How fast are vector databases?"

v_openai = openai_provider.embed_query(q)
v_bge = bge_provider.embed_query(q)

print(f"OpenAI Provider Vector Shape: {v_openai.shape} | L2 Norm: {np.linalg.norm(v_openai):.4f}")
print(f"BGE Local Provider Vector Shape: {v_bge.shape} | L2 Norm: {np.linalg.norm(v_bge):.4f}")
'''

st.code(code_embed_provider, language="python")

if st.button("🚀 Execute Embedding Provider Abstraction", type="primary"):
    exec_scope = {}
    exec(code_embed_provider, globals(), exec_scope)
    
    op_prov = exec_scope["MockOpenAIProvider"](target_dim=512)
    bge_prov = exec_scope["MockLocalBGEProvider"]()
    
    v1 = op_prov.embed_query("Sample search query")
    v2 = bge_prov.embed_query("Sample search query")
    
    st.success("Embedding Providers initialized and executed successfully!")
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.markdown(f"**OpenAI Provider (MRL Truncated)**:")
        st.markdown(f"* Dimensions: `{op_prov.dimension}`")
        st.markdown(f"* Output Vector First 4 Values: `{v1[:4].round(4)}`")
        st.markdown(f"* L2 Norm: `{np.linalg.norm(v1):.4f}`")
    with col_out2:
        st.markdown(f"**BGE-M3 Local Provider**:")
        st.markdown(f"* Dimensions: `{bge_prov.dimension}`")
        st.markdown(f"* Output Vector First 4 Values: `{v2[:4].round(4)}`")
        st.markdown(f"* L2 Norm: `{np.linalg.norm(v2):.4f}`")

st.divider()

# Section 6: Technical Interview Preparation & Production Traps
st.header("6. Technical Interview Preparation", divider="rainbow")

st.subheader("High-Frequency Architecture Questions & Senior-Level Answers")

q1 = """1. What is the Massive Text Embedding Benchmark (MTEB), and what are its limitations when selecting a production model?"""
a1 = """* **What MTEB Measures**: Evaluates embedding models across 8 diverse task types: Retrieval, Reranking, Classification, Clustering, Pair Classification, STS, and Summarization across dozens of open datasets.
* **Production Limitations**:
  1. **Synthetic Data Contamination**: Many newer leaderboard models are fine-tuned directly on synthetic data generated from MTEB test sets, causing leaderboard score inflation that fails to generalize to real enterprise corpora.
  2. **Latency & Compute Neglect**: MTEB only ranks accuracy, ignoring model parameter size (e.g. comparing a 7B parameter LLM encoder to a 22M parameter MiniLM).
  3. **Domain Gap**: MTEB primarily tests general web/Wikipedia text, which does not reflect specialized enterprise vocabularies (clinical trials, proprietary codebases, or complex ERP schemas)."""

q2 = """2. When should an enterprise deploy local open-weights embeddings instead of calling OpenAI or Cohere APIs?"""
a2 = """* **1. Regulatory Compliance (Zero Data Egress)**: In healthcare (HIPAA), defense, or finance (SOC2/GDPR), transmitting customer data or employee communications to third-party cloud APIs may be strictly prohibited.
* **2. High Ingestion Volumes (Cost Crossover)**: If ingesting hundreds of millions of tokens monthly, per-token cloud API fees escalate rapidly. A single self-hosted GPU instance ($180/mo) can embed billions of tokens for a fixed cost.
* **3. Ultra-Low Latency SLAs**: Cloud API calls introduce 50-200ms of internet round-trip latency. Local in-process runtimes (e.g., FastEmbed or ONNX Runtime over Unix domain sockets) achieve sub-5ms embedding latency.
* **4. Offline / Air-Gapped Environments**: Edge devices, submarines, or isolated government data centers with zero internet access."""

q3 = """3. How does BGE-M3's tri-mode retrieval architecture work, and what are its advantages?"""
a3 = """* BGE-M3 is a 1024-dimensional multilingual model that produces three distinct representations from a single forward pass:
  1. **Dense Vector**: 1024-d unit-normalized embedding for semantic concept retrieval.
  2. **Sparse Lexical Vector**: Vocabulary-sized weights (like SPLADE) capturing exact keywords and domain terms.
  3. **Multi-Vector (ColBERT style)**: Per-token contextual vectors for fine-grained MaxSim late-interaction ranking.
* **Advantages**:
  * Eliminates the need to maintain separate models for BM25 and dense search.
  * Allows vector databases (like Qdrant or Milvus) to compute hybrid scores from a single unified ingestion pipeline."""

q4 = """4. Why does an embedding model trained on general text fail on biomedical or financial datasets, and how do you adapt it?"""
a4 = """* **Why It Fails**:
  * Specialized technical terminology is split into meaningless subwords by general tokenizers (e.g., `'pembrolizumab'` split into `['pem', 'bro', 'liz', 'umab']`).
  * Words have domain-shifted semantics (e.g., `'bear'` in finance, `'culture'` in microbiology).
* **Adaptation Strategies**:
  1. **Domain Encoders**: Use pre-trained domain models (`BioBERT`, `FinBERT`, `LegalBERT`).
  2. **Domain-Adaptive Pre-training (DAPT)**: Continue masked language modeling on raw in-domain unlabelled text.
  3. **Contrastive Fine-Tuning**: Fine-tune the encoder using domain (query, positive_passage, hard_negative) triplets using MultipleNegativesRankingLoss."""

q5 = """5. How does FastEmbed / ONNX Runtime accelerate CPU embedding inference without GPUs?"""
a5 = """* **FastEmbed / ONNX Runtime Optimization**:
  * Converts PyTorch model computation graphs into optimized ONNX graph representations.
  * Fuses consecutive transformer layers (e.g., LayerNorm + MatMul + GeLU fused into a single kernel call).
  * Leverages modern CPU vector instructions (AVX-512, Intel AMX, ARM NEON) for 8-bit quantized integer arithmetic (INT8).
* **Performance**:
  * Enables small models (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`) to process 2,000 to 10,000 sentences per second on a standard 8-core CPU without requiring an expensive NVIDIA GPU."""

for q, a in [(q1, a1), (q2, a2), (q3, a3), (q4, a4), (q5, a5)]:
    with st.expander(q, expanded=False):
        st.markdown(a)

st.divider()

# Production Traps & War Stories
st.subheader("Embedding Model Production Traps")

st.markdown("""
> [!WARNING]
> **Trap 1: The Bulk Ingestion Rate Limit Storm**  
> Teams migrating 5 million documents often write a simple loop calling `openai.embeddings.create()`. 
> Within 60 seconds, the job crashes with HTTP 429 Too Many Requests (exceeding Tier 2/3 TPM quotas). 
> Always implement exponential backoff jitter, rate limit queueing, and evaluate local open-weights batch inference 
> for large initial backfills.

> [!WARNING]
> **Trap 2: Context Window Truncation Silence**  
> A document chunk of 1,200 tokens sent to a model with a 512-token context window (e.g. `bge-base-en-v1.5`) 
> will silently truncate the last 688 tokens without raising an error! The resulting vector only represents the top 40% 
> of the chunk. Always validate token length against model context limits before embedding.

> [!WARNING]
> **Trap 3: Model Version Drift Across Ingestion Batches**  
> If half of your database collection was embedded using `text-embedding-ada-002` and the other half was embedded 
> using `text-embedding-3-small`, searches will return nonsensical garbage. Their coordinate spaces are entirely incompatible. 
> When upgrading embedding models, you must perform a complete re-indexing into a fresh collection.
""")

st.divider()

st.subheader("Rapid-Review Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.expander("🗂️ Flashcard: What is MTEB?"):
        st.markdown("""
        **Massive Text Embedding Benchmark (MTEB)**:  
        The industry standard benchmark evaluating embedding models across 8 tasks (retrieval, clustering, classification) 
        and dozens of languages.
        """)
    with st.expander("🗂️ Flashcard: What is BGE-M3?"):
        st.markdown("""
        **BGE-M3**:  
        A state-of-the-art multilingual embedding model from BAAI supporting 100+ languages, 8k context, and simultaneous 
        Dense, Sparse, and Multi-vector output modes.
        """)

with col_fc2:
    with st.expander("🗂️ Flashcard: Why Use FastEmbed / ONNX on CPU?"):
        st.markdown("""
        **FastEmbed / ONNX Runtime**:  
        Accelerates open-weights models on commodity CPUs via operator fusion and AVX-512 INT8 quantization, achieving 
        thousands of embeddings/sec without GPUs.
        """)
    with st.expander("🗂️ Flashcard: What is Cross-Lingual Alignment?"):
        st.markdown("""
        **Cross-Lingual Alignment**:  
        Training an encoder on multilingual parallel text such that concepts in different languages occupy identical 
        coordinates in the vector space, enabling cross-language search.
        """)

st.divider()

# Section 7: Self-Assessment Checklist
st.header("7. Self-Assessment Checklist", divider="rainbow")

chk1 = st.checkbox("I understand the trade-offs between OpenAI cloud APIs and local open-weights models.", value=True)
chk2 = st.checkbox("I know how to navigate the MTEB benchmark and recognize its production evaluation limits.")
chk3 = st.checkbox("I can explain how BGE-M3 generates Dense, Sparse, and Multi-vector representations simultaneously.")
chk4 = st.checkbox("I understand why domain-specific terminology necessitates specialized encoders (e.g. BioBERT, FinBERT).")
chk5 = st.checkbox("I know how to calculate total cost of ownership (TCO) between cloud API billing and dedicated GPU VMs.")
chk6 = st.checkbox("I understand how to design an abstract embedding provider interface in Python.")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.balloons()
    st.success("🎉 Outstanding! You have mastered Embedding Models and are ready for Chapter 2.4: Embedding Properties!")
