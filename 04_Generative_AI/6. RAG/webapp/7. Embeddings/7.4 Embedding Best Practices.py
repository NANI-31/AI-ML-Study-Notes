import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# Page Configuration
st.set_page_config(
    page_title="7.4 Embedding Best Practices - RAG Curriculum",
    page_icon="🧠",
    layout="wide"
)

# Module-Scope Classes for Interactive Workbench & Simulations
class MatryoshkaTruncationSimulator:
    """Simulates Matryoshka Representation Learning (MRL) dimension truncation and cost savings."""
    def __init__(self, full_dim: int = 1536):
        self.full_dim = full_dim

    def truncate_and_normalize(self, original_dim: int, target_dim: int) -> Dict[str, Any]:
        # Memory calculation: 4 bytes per float32
        full_ram_mb = (100_000 * original_dim * 4) / (1024**2)
        trunc_ram_mb = (100_000 * target_dim * 4) / (1024**2)
        savings_pct = round((1.0 - (target_dim / original_dim)) * 100, 1)

        # Empirical MTEB retention curves
        if target_dim >= 1024:
            accuracy_retention = 99.4
        elif target_dim >= 512:
            accuracy_retention = 98.2
        elif target_dim >= 256:
            accuracy_retention = 95.8
        else:
            accuracy_retention = 91.2

        return {
            "original_dim": original_dim,
            "target_dim": target_dim,
            "ram_for_100k_vectors_full_mb": round(full_ram_mb, 1),
            "ram_for_100k_vectors_trunc_mb": round(trunc_ram_mb, 1),
            "ram_reduction_pct": f"{savings_pct}%",
            "accuracy_retention_pct": f"{accuracy_retention}%"
        }

# Title & Overview
st.title("7.4 Embedding Best Practices: Governance, MRL & Cost Optimization")
st.caption("Chapter 7: Embeddings | Domain: 6. RAG")
st.markdown("""
Deploying embedding models into enterprise production requires disciplined engineering beyond simply 
calling an API endpoint. As a knowledge base scales from thousands of documents to tens of millions of passages, 
unoptimized embedding architectures incur massive cloud billing charges, exhaust vector database RAM, and degrade query throughput.

**Embedding Best Practices** govern model selection, dimensionality management via **Matryoshka Representation Learning (MRL)**, 
domain-specific fine-tuning, cross-lingual representations, and high-throughput semantic caching.

In this lesson, you will master:
1. **Model Selection Criteria**: Aligning latency SLAs, MTEB benchmarks, and privacy compliance requirements.
2. **Matryoshka Representation Learning (MRL)**: Truncating vectors by up to 75% while preserving 98%+ retrieval accuracy.
3. **Domain Fine-Tuning**: Adapting pre-trained representations to specialized legal, healthcare, and engineering vocabularies.
4. **Multilingual Alignment**: Ensuring seamless cross-lingual retrieval across global enterprise repositories.
5. **Production Failure Traps**: Avoiding unnormalized truncated vectors, cold embedding caches, and re-indexing downtime.
""")
st.divider()

# Section 1: Architectural Blueprint
st.header("1. Complete Architectural Blueprint: Enterprise Embedding Governance & Lifecycle Management", divider="rainbow")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    dot_blueprint = """
    digraph EmbeddingGovernanceBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#38BDF8", fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_governance {
            label="Tier 1: Ingestion Batching & Semantic Deduplication Cache";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#38BDF8";

            text_ingress [label="Document Passage Ingress Stream\\nRaw extracted text chunks", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            redis_cache [label="Semantic Hash & Exact Embedding Cache (Redis)\\nBypasses API invocation if text hash was already embedded", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            text_ingress -> redis_cache;
        }

        subgraph cluster_model_tier {
            label="Tier 2: Model Inference & Matryoshka Truncation";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#10B981";

            embed_worker [label="Batched Embedding Inference Engine\\nMicro-batches of 128 chunks sent to Model API / TEI", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            mrl_truncator [label="Matryoshka Truncation Layer\\nSlices 1536-d to 512-d: v = v[:512]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];
            re_normalizer [label="L2 Re-Normalization Engine\\nScales truncated slice back to ||v|| = 1.0", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            redis_cache -> embed_worker [label="Cache Miss"];
            embed_worker -> mrl_truncator -> re_normalizer;
        }

        subgraph cluster_storage_tier {
            label="Tier 3: Optimized Vector Database Storage";
            style=filled;
            color="#334155";
            fillcolor="#1E293B";
            fontname="Helvetica"; fontsize=11; fontcolor="#F59E0B";

            compressed_index [label="Quantized Vector Index (SQ8 / HNSW)\\n66% RAM savings from 512-d truncation + scalar compression", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#10B981"];

            re_normalizer -> compressed_index;
            redis_cache -> compressed_index [label="Cache Hit Reuse", style=dashed];
        }
    }
    """
    st.graphviz_chart(dot_blueprint, use_container_width=True)
st.divider()

# Section 2: Choosing Embedding Models
st.header("2. Choosing Embedding Models: The Strategic Decision Matrix", divider="rainbow")
st.markdown("""
Model selection should follow a structured engineering evaluation:

1. **Context Window Requirements**: If chunks average 800 tokens, avoid models with 256 or 512-token limits (such as `MiniLM` or standard BERT) to prevent silent truncation.
2. **Data Privacy & Air-Gapped Constraints**: If data cannot leave your corporate perimeter, select open weights (BGE, GTE) served via self-hosted **Text Embeddings Inference (TEI)**.
3. **MTEB Retrieval Benchmark**: Focus strictly on the *Retrieval* subset score rather than general model averages.
4. **Economic Scale**: Evaluate the total cost of ownership across API token fees versus GPU cloud hosting costs.
""")
st.divider()

# Section 3: Embedding Dimensions
st.header("3. Embedding Dimensions & Matryoshka Representation Learning (MRL)", divider="rainbow")
st.markdown("""
Higher dimensionality does not always translate to superior retrieval performance in production:

* **The Dimensionality Tax**: Every dimension added to a vector increases memory footprint by 4 bytes per vector in float32. At 10 million vectors:
  * 384 dimensions = 15 GB RAM.
  * 1536 dimensions = 60 GB RAM.
  * 3072 dimensions = 120 GB RAM.
* **Matryoshka Truncation in Production**:
  * Using models trained with MRL (OpenAI `text-embedding-3`, Google `text-embedding-004`, or `nomic-embed-text`), vectors can be safely truncated.
  * Slicing from 1536 down to 512 dimensions preserves **98.2% of retrieval accuracy** while cutting vector database RAM requirements by **66%**.
""")
st.divider()

# Section 4: Domain-Specific Embeddings
st.header("4. Domain-Specific Embeddings: Fine-Tuning on Proprietary Corpora", divider="rainbow")
st.markdown("""
General-purpose embedding models are pre-trained on public web crawls (Wikipedia, Common Crawl, Reddit). 
They frequently stumble on domain-specific enterprise vocabularies:

* **The Jargon Gap**: In biomedical research (*"EGFR L858R kinase mutations"*), aviation (*"ETOPS dispatch diversion rules"*), or proprietary internal microservice architectures (*"Hydra auth-proxy Envoy sidecar"*), public models have weak representation manifolds.
* **Fine-Tuning with Contrastive Loss**:
  * Generate 1,000–5,000 domain triplets: `(Anchor Question, Positive Passage, Negative Passage)`.
  * Train using **Multiple Negatives Ranking Loss (MNRL)** via Sentence Transformers.
  * Fine-tuning typically yields a **15–25% boost in Retrieval Hit Rate@5** on internal enterprise repositories.
""")
st.divider()

# Section 5: Multilingual Embeddings
st.header("5. Multilingual Embeddings: Cross-Lingual Alignment & Global Search", divider="rainbow")
st.markdown("""
Global enterprises maintain documentation in multiple languages (English, Japanese, German, Spanish):

* **Cross-Lingual Coordinate Manifolds**: Multilingual models (such as Cohere Multilingual v3, Google Vertex Multilingual, or BGE-M3) map equivalent concepts in 100+ languages into the identical geometric neighborhood.
* **Query in Language A, Retrieve in Language B**: A user in Tokyo searching in Japanese (*"有給休暇の申請方法"*) directly retrieves the English corporate HR policy (*"Paid Time Off Request Procedures"*) without requiring a translation API.
* **Language Agnostic Retrieval**: Eliminates the overhead of maintaining separate vector database collections for each spoken language.
""")
st.divider()

# Section 6: Embedding Normalization
st.header("6. Embedding Normalization: Enforcing Unit Magnitude", divider="rainbow")
st.markdown("""
Always enforce **L2 unit normalization** (`||v|| = 1.0`) across all embedding pipelines:

* **Why?**:
  1. Guarantees that Dot Product equals Cosine Similarity, enabling high-speed SIMD hardware instructions.
  2. Prevents longer text passages from dominating retrieval scores solely due to larger vector magnitudes.
  3. Prepares vectors for downstream Scalar Quantization (SQ8) and Product Quantization (PQ).
* **The Normalization Formula**:
  `v_normalized = v / np.linalg.norm(v)`.
""")
st.divider()

# Section 7: Embedding Cost Optimization
st.header("7. Embedding Cost Optimization: Caching, Batching & Truncation", divider="rainbow")
st.markdown("""
Implement four architectural strategies to minimize embedding expenses at enterprise scale:
""")

col_opt1, col_opt2 = col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    st.markdown("""
    ### 1. Semantic Hash Caching (Redis)
    * Compute a SHA-256 hash of the sanitized text string prior to calling embedding APIs.
    * In dynamic document updates where only 10% of a manual changes, cache hits bypass 90% of embedding API calls!
    
    ### 2. High-Throughput Request Batching
    * Never call embedding APIs for one chunk at a time.
    * Batch chunks into optimal payloads (64 to 128 passages per HTTP request) to saturate GPU throughput.
    """)
with col_opt2:
    st.markdown("""
    ### 3. Matryoshka Dimension Truncation
    * Truncate 1536-d vectors to 512 dimensions for secondary or non-critical collections, slashing storage costs.
    
    ### 4. Quantized Storage in Vector Database
    * Deploy 8-bit Scalar Quantization (SQ8) in the vector database to compress vectors in RAM by an additional 75%.
    """)

### Production Modern Contrast
st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production enterprise RAG architectures combine **Redis Document Hashing** with **Tiered Storage**: 
frequently queried chunks reside in high-speed RAM-cached HNSW indexes, while millions of cold archival chunks 
are stored in memory-mapped SSD storage via disk-backed engines like DiskANN or Qdrant mmap.
""")
st.divider()

# Section 8: Production Implementation
st.header("8. Production Implementation: Cached & Truncated Enterprise Embedding Service", divider="rainbow")
st.markdown("""
Below is a runnable Python implementation demonstrating an enterprise embedding service featuring 
hash-based caching, Matryoshka dimension truncation, and automatic L2 re-normalization.
""")

st.code('''
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional

class EnterpriseEmbeddingPipeline:
    """Production embedding pipeline featuring in-memory hash caching and Matryoshka truncation."""
    def __init__(self, target_dimensions: int = 512):
        self.target_dimensions = target_dimensions
        self.cache: Dict[str, np.ndarray] = {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> np.ndarray:
        text_hash = self._hash_text(text)
        
        # Check cache hit
        if text_hash in self.cache:
            return self.cache[text_hash]

        # Simulate full-dimension embedding generation (1536-d)
        seed = abs(hash(text_hash)) % (2**31)
        np.random.seed(seed)
        full_vector = np.random.randn(1536).astype(np.float32)

        # 1. Apply Matryoshka Truncation (slice first N dimensions)
        truncated_slice = full_vector[:self.target_dimensions]

        # 2. Re-normalize truncated slice to unit magnitude
        norm = np.linalg.norm(truncated_slice)
        final_vector = truncated_slice / norm if norm > 0 else truncated_slice

        # Store in cache
        self.cache[text_hash] = final_vector
        return final_vector

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]
''', language="python")
st.divider()

# Section 9: Interactive Workbench
st.header("9. Interactive Workbench: Matryoshka Dimension Truncation & Cost-Savings Laboratory", divider="rainbow")
st.markdown("""
Evaluate the empirical trade-offs of Matryoshka Representation Learning. Slicing vector dimensions 
reduces vector database RAM and storage overhead while retaining high retrieval accuracy.
""")

trunc_sim = MatryoshkaTruncationSimulator(full_dim=1536)

selected_target_dim = st.select_slider(
    "Select Target Truncated Dimensions (MRL):", 
    options=[128, 256, 512, 768, 1024, 1536], 
    value=512
)

results = trunc_sim.truncate_and_normalize(original_dim=1536, target_dim=selected_target_dim)

st.subheader("System Architecture & Economic Impact")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)

col_b1.metric("Selected Dimension", f"{results['target_dim']}")
col_b2.metric("RAM for 100k Vectors", f"{results['ram_for_100k_vectors_trunc_mb']} MB", delta=f"-{results['ram_reduction_pct']}")
col_b3.metric("Storage Reduction", results['ram_reduction_pct'])
col_b4.metric("Estimated MTEB Retention", results['accuracy_retention_pct'])

if selected_target_dim == 512:
    st.success("🎯 **The Industry Sweet Spot**: 66% RAM reduction with 98.2% accuracy retention!")
elif selected_target_dim <= 256:
    st.info("⚡ **Extreme Compression**: Best for mobile, edge devices, or multi-million vector scale.")
else:
    st.info("🛡️ **Maximum Precision**: Best for legal, medical, and mission-critical reasoning.")
st.divider()

# Section 10: Technical Interview Preparation & Production Traps
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("High-Frequency Interview Questions")
with st.expander("Q1: What is Matryoshka Representation Learning (MRL), and how does it reduce RAG vector database costs?", expanded=True):
    st.markdown("""
    **Answer:**
    Matryoshka Representation Learning (MRL) is a training methodology where loss functions are enforced across multiple prefix sub-vectors 
    simultaneously during model pre-training. This structures the embedding space so that the most essential, high-entropy semantic 
    information is concentrated in the earliest dimensions.
    
    In RAG systems, developers can truncate a 1536-dimensional vector down to 512 or 256 dimensions simply by slicing the array (`v[:512]`) 
    and re-normalizing. This reduces vector database RAM consumption and query latency by up to 66% while preserving over 98% of the model's 
    original retrieval accuracy.
    """)

with st.expander("Q2: Why must you re-normalize a vector after slicing its dimensions in MRL?"):
    st.markdown("""
    **Answer:**
    The original full-length vector was L2-normalized (`||v|| = 1.0`). When you slice off dimensions (e.g., retaining only the first 512 of 1536 values), 
    the Euclidean length of the remaining slice becomes strictly less than 1.0. If you perform Dot Product search using unnormalized truncated vectors, 
    shorter vectors will be penalized and similarity scores will become distorted. Dividing by the new slice's Euclidean norm restores unit length.
    """)

with st.expander("Q3: When is fine-tuning an embedding model justified over using an off-the-shelf foundation model?"):
    st.markdown("""
    **Answer:**
    Fine-tuning is justified when:
    1. The knowledge corpus contains heavy proprietary jargon, acronyms, or codes (e.g., medical clinical notes, legal statutes, internal microservice schemas) that public web crawls do not represent accurately.
    2. Empirical retrieval benchmarks show off-the-shelf models achieve Hit Rate@5 below production SLAs (<80%).
    3. The team has gathered or synthesized at least 1,000–5,000 high-quality training triplets `(query, positive_passage, hard_negative_passage)`.
    """)

with st.expander("Q4: How does semantic hash caching reduce operational costs in RAG ingestion pipelines?"):
    st.markdown("""
    **Answer:**
    In dynamic knowledge repositories, documents undergo frequent incremental edits (e.g., fixing typos or updating a single policy paragraph). 
    A semantic hash cache computes a deterministic hash (SHA-256) of each normalized chunk. Before calling the embedding API, the pipeline 
    checks the cache: unchanged chunks reuse existing vector embeddings, bypassing API billing and GPU compute for up to 90% of re-ingestion workloads.
    """)

with st.expander("Q5: What are the risks of using multilingual embedding models versus monolingual models?"):
    st.markdown("""
    **Answer:**
    * **Advantage**: Allows cross-lingual search (query in Japanese, retrieve in English) and unified multi-language indexing in a single collection.
    * **Trade-off (The Curse of Multilinguality)**: A multilingual model allocates its finite capacity across 100+ languages, which can result in slightly lower monolingual retrieval precision in English compared to a dedicated English-only model of the same parameter scale.
    """)

st.subheader("Production Failure Traps")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("""
    **Trap 1: Omitting Vector Re-Normalization After Matryoshka Slicing (Corrupted Search Rankings)**
    Truncating vectors to 512 dimensions (`v[:512]`) and indexing them directly into a Dot Product vector database without dividing by `np.linalg.norm(v[:512])`. The vectors lose unit magnitude, distorting similarity scores and ranking short passages ahead of relevant documents.
    """)
    st.error("""
    **Trap 2: Zero-Downtime Migration Failure During Embedding Model Upgrades (Complete Service Outage)**
    Switching the production embedding model from `text-embedding-ada-002` to `text-embedding-3-small` in code while keeping the existing vector database index active. Incoming query vectors inhabit a completely different geometric manifold than stored vectors, reducing retrieval accuracy to near zero. Always deploy a blue-green re-indexing pipeline.
    """)
with trap_c2:
    st.warning("""
    **Trap 3: Uncached Ingestion Re-Embedding Unchanged Documents (Inflated Cloud Billing)**
    Re-ingesting an entire 500,000-page document library on every minor documentation update without SHA-256 content hashing. Thousands of dollars are wasted re-embedding identical text passages repeatedly.
    """)
    st.warning("""
    **Trap 4: Missing Asymmetric Query Prefixes on Local Fine-Tuned Models (Hit Rate Drop)**
    Training a Sentence Transformers bi-encoder with query and passage prefixes, but deploying the query service without the prefix formatting layer. The model fails to map query intent into answer manifolds.
    """)

st.subheader("Interactive Knowledge Flashcards")
card_c1, card_c2 = st.columns(2)
with card_c1:
    with st.container(border=True):
        st.markdown("**Front**: What is the primary benefit of Matryoshka Representation Learning (MRL)?")
        st.markdown("**Back**: It allows vectors to be sliced to smaller dimensions, reducing database RAM costs by up to 66% with negligible accuracy loss.")
    with st.container(border=True):
        st.markdown("**Front**: Why is L2 normalization mandatory after MRL truncation?")
        st.markdown("**Back**: Slicing reduces vector magnitude below 1.0; re-normalization restores unit length for accurate Dot Product search.")
with card_c2:
    with st.container(border=True):
        st.markdown("**Front**: How does SHA-256 caching optimize embedding costs?")
        st.markdown("**Back**: It hashes chunk text before API calls, preventing costly re-embedding of unchanged passages during re-ingestion.")
    with st.container(border=True):
        st.markdown("**Front**: How do you execute a zero-downtime embedding model upgrade?")
        st.markdown("**Back**: Ingest the corpus into a new parallel vector collection using the new model, verify benchmarks, and switch query routing atomically.")

st.divider()

# Section 11: Self-Assessment Checklist
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your understanding of Embedding Best Practices and Governance before proceeding:")

c1, c2 = st.columns(2)
with c1:
    chk1 = st.checkbox("I know how to select embedding models based on context windows, MTEB benchmarks, and privacy constraints.", value=True, key="chk_emb_best_1")
    chk2 = st.checkbox("I understand how Matryoshka Representation Learning (MRL) enables vector dimension truncation.", value=True, key="chk_emb_best_2")
    chk3 = st.checkbox("I know why truncated MRL vectors must be re-normalized to unit magnitude before indexing.", value=True, key="chk_emb_best_3")
with c2:
    chk4 = st.checkbox("I understand when to fine-tune embedding models on domain-specific enterprise vocabularies.", value=True, key="chk_emb_best_4")
    chk5 = st.checkbox("I know how to implement semantic hash caching to minimize redundant API embedding costs.", value=True, key="chk_emb_best_5")
    chk6 = st.checkbox("I can identify production traps such as unnormalized truncated vectors and zero-downtime migration failures.", value=True, key="chk_emb_best_6")

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("✅ **Embedding Best Practices Mastery Demonstrated!** You have completed Chapter 7 of the RAG curriculum.")
