import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Optional

st.set_page_config(
    page_title="16.1 Embedding Interfaces - Embeddings Curriculum",
    page_icon="🦜",
    layout="wide"
)

# Title & Overview
st.title("16.1 LangChain Embedding Interfaces: Standard Contracts, Providers, & Local Models")
st.caption("Chapter 16: Embeddings with LangChain | Domain: 3. Vector Embeddings")
st.markdown("""
LangChain is the most ubiquitous orchestration framework for building LLM and RAG applications.
A key architectural reason for its dominance is **standardized abstraction**: decoupling your application logic from specific model vendors.

Instead of writing bespoke code for OpenAI, Hugging Face, or local Ollama servers, LangChain defines a single unified base interface: **`Embeddings`**.
* Any model implementing this interface can be hot-swapped into vector stores, document splitters, and retrieval chains with zero refactoring.
* It enforces a critical architectural distinction between **document embedding** (`embed_documents`) and **query embedding** (`embed_query`).

This lesson explores LangChain's embedding interface standard, supported cloud and open-source providers, and local deployment strategies.
""")

st.divider()

# Section 1: LangChain Embeddings Architecture & Base Contract
st.header("1. The LangChain `Embeddings` Base Interface Contract", divider="rainbow")

col_int1, col_int2 = st.columns([1, 1])

with col_int1:
    st.markdown("""
    ### The Two Abstract Methods
    Every LangChain embedding class inherits from `langchain_core.embeddings.Embeddings` and must implement two fundamental methods:

    * **`embed_documents(texts: List[str]) -> List[List[float]]`**:
      * Accepts a batch of document chunks.
      * Optimized for batch throughput across GPU tensors or bulk REST payloads.
      * Does **not** prepend query instructions.
    
    * **`embed_query(text: str) -> List[float]`**:
      * Accepts a single search query string.
      * Returns a single 1D vector.
      * Automatically prepends asymmetric query instruction prefixes if required by the underlying model (e.g., `"Represent this sentence for searching relevant passages: "` in BGE/E5).
    """)

with col_int2:
    st.markdown("### LangChain Embeddings Class Hierarchy")
    lc_dot = """
    digraph LangChainEmbeddings {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9, margin=0.1];
        edge [fontname="Helvetica", fontsize=8];

        BaseEmbed [label="langchain_core.embeddings.Embeddings\\n+ embed_documents(texts: List[str])\\n+ embed_query(text: str)", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];

        OpenAI [label="OpenAIEmbeddings\\n(text-embedding-3-small/large)\\nCloud REST API", fillcolor="#FEF3C7", color="#D97706"];
        HF [label="HuggingFaceEmbeddings\\n(sentence-transformers, BGE)\\nLocal PyTorch / Transformers", fillcolor="#ECFDF5", color="#10B981"];
        Ollama [label="OllamaEmbeddings\\n(nomic-embed, llama3)\\nLocal Self-Hosted Server", fillcolor="#EDE9FE", color="#8B5CF6"];
        FastEmb [label="FastEmbedEmbeddings\\n(ONNX CPU optimized)\\nLightweight Zero-PyTorch", fillcolor="#FEE2E2", color="#EF4444"];

        BaseEmbed -> OpenAI [label="Cloud API"];
        BaseEmbed -> HF [label="PyTorch Local"];
        BaseEmbed -> Ollama [label="Local Docker/Daemon"];
        BaseEmbed -> FastEmb [label="ONNX Local"];
    }
    """
    st.graphviz_chart(lc_dot, use_container_width=False)

st.markdown("### Major LangChain Embedding Providers Compared")
st.markdown("""
| Provider Class | Package | Compute Location | Default Dimensions | Primary Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **`OpenAIEmbeddings`** | `langchain-openai` | Cloud API (OpenAI) | 1536 / 3072 | Zero infrastructure management, MRL dimension truncation. |
| **`HuggingFaceEmbeddings`** | `langchain-huggingface` | Local GPU / CPU | 384 / 768 / 1024 | Free, private, state-of-the-art MTEB benchmarks (BGE/E5). |
| **`OllamaEmbeddings`** | `langchain-community` | Local Server (Ollama) | 768 (nomic) | Easy drop-in container deployment for offline air-gapped systems. |
| **`FastEmbedEmbeddings`** | `langchain-community` | Local CPU (ONNX) | 384 / 768 | Ultra-fast CPU inference, zero CUDA/PyTorch dependencies. |
""")

st.divider()

# Section 2: Production Code: Instantiating & Invoking Providers
st.header("2. Production Code: Standard Usage Across Providers", divider="rainbow")

st.markdown("""
The production snippet below demonstrates how LangChain enables identical call semantics across OpenAI, Hugging Face, and a custom local embedding subclass.
""")

st.code(r'''
from typing import List
import numpy as np

# In production with installed packages:
# from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.embeddings import Embeddings

class CustomLocalEmbeddings:
    """
    Illustrates implementing LangChain's Embeddings interface from scratch.
    """
    def __init__(self, model_name: str = "custom-minilm", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch document embedding implementation."""
        results = []
        for text in texts:
            np.random.seed(abs(hash(text)) % (2**31 - 1))
            v = np.random.randn(self.dim).astype(np.float32)
            v = v / np.maximum(np.linalg.norm(v), 1e-12)
            results.append(v.tolist())
        return results

    def embed_query(self, text: str) -> List[float]:
        """Query embedding implementation with optional instruction prefix."""
        # For asymmetric models, prepend query instruction prefix:
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        np.random.seed(abs(hash(prefixed)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        v = v / np.maximum(np.linalg.norm(v), 1e-12)
        return v.tolist()

# Demonstration of interchangeable usage:
def run_embedding_pipeline():
    # Swap this instance with OpenAIEmbeddings() or HuggingFaceEmbeddings() with 0 code changes:
    embeddings_model = CustomLocalEmbeddings(dim=384)

    # Ingest document corpus
    corpus = [
        "LangChain standardizes interactions with foundational language models.",
        "Vector stores index high-dimensional embeddings for nearest neighbor search."
    ]
    doc_vectors = embeddings_model.embed_documents(corpus)
    print(f"Generated {len(doc_vectors)} document vectors of dimension {len(doc_vectors[0])}.")

    # Embed user search query
    query = "What is the purpose of LangChain?"
    query_vector = embeddings_model.embed_query(query)
    print(f"Generated 1 query vector of dimension {len(query_vector)}.")

if __name__ == "__main__":
    run_embedding_pipeline()
''', language="python")

st.divider()

# Section 3: Interactive Multi-Provider Simulator Workbench
st.header("3. Interactive Workbench: Multi-Provider Interface Sandbox", divider="rainbow")

st.markdown("""
Test how LangChain handles document vs query embeddings across different simulated providers. Observe dimensionality differences, pricing models, and cosine alignment.
""")

class ProviderSimulator:
    def __init__(self, name: str, dim: int, cost_per_1m: float, is_local: bool):
        self.name = name
        self.dim = dim
        self.cost_per_1m = cost_per_1m
        self.is_local = is_local

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            np.random.seed(abs(hash(f"{self.name}_{t}")) % (2**31 - 1))
            v = np.random.randn(self.dim).astype(np.float32)
            vecs.append(v / np.maximum(np.linalg.norm(v), 1e-12))
        return np.array(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        prefixed = f"{self.name}_query_{text}"
        np.random.seed(abs(hash(prefixed)) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

providers = {
    "OpenAI text-embedding-3-small": ProviderSimulator("openai_small", 1536, 0.02, False),
    "OpenAI text-embedding-3-large": ProviderSimulator("openai_large", 3072, 0.13, False),
    "HuggingFace BGE-base-en-v1.5": ProviderSimulator("hf_bge_base", 768, 0.0, True),
    "Local Ollama (nomic-embed-text)": ProviderSimulator("ollama_nomic", 768, 0.0, True),
    "FastEmbed (bge-small-en ONNX)": ProviderSimulator("fastembed_small", 384, 0.0, True)
}

col_sim1, col_sim2 = st.columns([1, 1])

with col_sim1:
    st.subheader("1. Configure Provider")
    selected_p_name = st.selectbox("Select LangChain Embedding Model:", list(providers.keys()))
    provider = providers[selected_p_name]

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Dimensions", provider.dim)
    p_col2.metric("Cost / 1M Tokens", f"${provider.cost_per_1m:.2f}" if not provider.is_local else "$0.00 (Local)")
    p_col3.metric("Type", "Local (Zero API)" if provider.is_local else "Cloud API")

    st.subheader("2. Input Corpus & Query")
    doc_inputs = [
        "LangChain separates document chunk embedding from query embedding.",
        "Local models like Ollama ensure data privacy in healthcare and finance.",
        "Cloud embedding APIs provide effortless scaling without GPU infrastructure."
    ]
    user_query = st.text_input("User Search Query (`embed_query`):", value="How to maintain data privacy with embeddings?")

with col_sim2:
    st.subheader("3. Execution Output & Alignment")
    
    doc_vectors = provider.embed_documents(doc_inputs)
    query_vector = provider.embed_query(user_query)

    st.markdown(f"**`embed_documents` Output:** {len(doc_vectors)} vectors (Shape: `{doc_vectors.shape}`)")
    st.markdown(f"**`embed_query` Output:** 1 vector (Length: `{len(query_vector)}`)")

    # Calculate similarity scores
    sims = np.dot(doc_vectors, query_vector)
    
    res_table = []
    for i, (text, sim) in enumerate(zip(doc_inputs, sims)):
        res_table.append({
            "Doc Index": f"Doc {i+1}",
            "Text Passage": text[:55] + "...",
            "Cosine Similarity": round(float(sim), 4),
            "Alignment": f"{max(0.0, float(sim))*100:.1f}%"
        })
    
    st.dataframe(pd.DataFrame(res_table), use_container_width=True)

    st.markdown("#### First 6 Dimensions of Generated Vectors")
    preview_df = pd.DataFrame({
        "Query Vector": query_vector[:6],
        "Doc 1 Vector": doc_vectors[0][:6],
        "Doc 2 Vector": doc_vectors[1][:6],
        "Doc 3 Vector": doc_vectors[2][:6]
    })
    st.dataframe(preview_df.style.format("{:.4f}"), use_container_width=True)

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "LangChain Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: Why does LangChain's `Embeddings` base class strictly separate `embed_documents` from `embed_query`?**
    * **Answer**:
      * 1. **Batching vs. Single-Item Latency**: `embed_documents` accepts a `List[str]` designed for batched GPU tensor computation or bulk API calls (e.g. 2048 chunks per HTTP request). `embed_query` accepts a single `str` optimized for real-time interactive user queries.
      * 2. **Asymmetric Model Prefixing**: Asymmetric embedding architectures (like BGE, INSTRUCTOR, or E5) require distinct prompts for queries versus passages. `embed_query` automatically prepends task instructions (e.g. `"Represent this question for searching documents:"`) while `embed_documents` encodes raw passage text without instructions.

    **Q2: What is `CacheBackedEmbeddings` in LangChain and why is it critical for cost reduction?**
    * **Answer**: `CacheBackedEmbeddings` wraps any `Embeddings` provider with a key-value store (like Redis, LocalFileStore, or SQLite). When embedding documents, it computes a hash of the text:
      * If the hash is found in cache, it retrieves the pre-computed vector instantly (0ms latency, $0 cost).
      * If not found, it calls the underlying model and caches the result.
      * This prevents catastrophic API billing surges when re-running ingestion pipelines or restarting indexing workers.

    **Q3: When should you prefer `FastEmbedEmbeddings` over `HuggingFaceEmbeddings`?**
    * **Answer**: `HuggingFaceEmbeddings` requires heavy PyTorch and Transformers libraries (often a 2GB+ Docker image). `FastEmbedEmbeddings` uses ONNX Runtime, eliminating PyTorch entirely. It runs 50%+ faster on standard CPU machines, produces tiny Docker container images (< 200MB), and requires zero CUDA configuration.
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Re-embedding the Entire Corpus When Adding One New Document**:
      * *Issue*: Calling `OpenAIEmbeddings.embed_documents()` on all 50,000 documents whenever a new document is uploaded.
      * *Consequence*: Thousands of dollars wasted in duplicate API fees.
      * *Fix*: Always wrap embedding models with `CacheBackedEmbeddings.from_bytes_store()`.

    * **Trap 2: Mixing Different Embedding Model Dimensions in a Single VectorStore**:
      * *Issue*: Embedding some documents with `text-embedding-3-small` (1536 dim) and querying with `bge-base-en` (768 dim).
      * *Consequence*: Vector stores throw dimension mismatch errors or calculate nonsensical similarity metrics.
      * *Fix*: Tie the vector store instance to a single persistent embedding model instance.

    * **Trap 3: Exceeding Token Limits in `embed_documents` Without Chunking**:
      * *Issue*: Passing a 10-page text directly into `embed_documents()`.
      * *Consequence*: Model silently truncates the text after 512 or 8192 tokens, discarding the second half of the document.
      * *Fix*: Always pass documents through a `TextSplitter` before invoking `embed_documents()`.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: `Embeddings` Interface**
      * *Definition*: The foundational LangChain abstraction defining `embed_documents` and `embed_query` to enable interchangeable model providers.
    * **Flashcard 2: `embed_query()`**
      * *Definition*: The method used specifically for single search query strings, often injecting query-specific instruction prefixes.
    * **Flashcard 3: `CacheBackedEmbeddings`**
      * *Definition*: A caching wrapper that avoids redundant embedding computations and API costs by hashing text to stored vectors.
    * **Flashcard 4: `FastEmbedEmbeddings`**
      * *Definition*: A lightweight CPU-optimized ONNX runtime embedding library requiring zero PyTorch dependencies.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of LangChain Embedding Interfaces:

- [ ] I can articulate the structural difference between `embed_documents` and `embed_query`.
- [ ] I understand why asymmetric models require different handling for queries vs documents.
- [ ] I know how to wrap embedding providers with `CacheBackedEmbeddings` to reduce API costs.
- [ ] I understand the tradeoffs between cloud APIs (OpenAI) and local execution (Hugging Face, Ollama, FastEmbed).
- [ ] I can implement a custom `Embeddings` class adhering to LangChain's interface contract.
""")
