import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple

st.set_page_config(
    page_title="16.3 Retrieval with LangChain - Embeddings Curriculum",
    page_icon="🔎",
    layout="wide"
)

# Title & Overview
st.title("16.3 LangChain Retrieval: Retrievers, Similarity Scoring, & MMR Diversification")
st.caption("Chapter 16: Embeddings with LangChain | Domain: 3. Vector Embeddings")
st.markdown("""
In LangChain, a **VectorStore** is a storage engine, while a **Retriever** is a higher-level runnable interface designed for composability inside LangChain Expression Language (LCEL) chains.

Calling `.as_retriever()` transforms a static vector store into an intelligent retrieval engine supporting:
* **Standard Similarity Search**: Finding the top $K$ nearest chunks.
* **Similarity Search with Score**: Inspecting numerical distance/similarity metrics to evaluate confidence.
* **Maximal Marginal Relevance (MMR)**: Balancing relevance with **diversity** to avoid returning multiple redundant copies of the same sentence.
* **Score Thresholding**: Dynamically rejecting low-confidence distractors.
* **Retriever Configuration (`search_kwargs`)**: Bundling metadata filters, candidate pool sizes, and ranking parameters.

This lesson explores how LangChain structures retrieval mechanisms for production RAG pipelines.
""")

st.divider()

# Section 1: LangChain Retriever Architecture & Search Types
st.header("1. LangChain Retriever Architecture & Search Types", divider="rainbow")

col_ret1, col_ret2 = st.columns([1, 1])

with col_ret1:
    st.markdown("""
    ### Key Retrieval Paradigms
    * **1. The `BaseRetriever` Interface**:
      * Accepts a query string and returns a `List[Document]`.
      * Exposes `.invoke(query)` (synchronous) and `.ainvoke(query)` (asynchronous).
    * **2. Standard Similarity (`search_type="similarity"`)**:
      * Queries the vector index for the top $K$ nearest embeddings.
    * **3. Similarity with Score (`similarity_search_with_score`)**:
      * Returns tuples of `(Document, score)`, enabling custom application thresholding.
    * **4. Maximal Marginal Relevance (`search_type="mmr"`)**:
      * Fetches a large pool of $M$ candidates (`fetch_k`, e.g. 20).
      * Greedily selects $K$ final documents (e.g. 4) by optimizing:
        `MMR Score = lambda_mult * Sim(doc, query) - (1 - lambda_mult) * MaxSim(doc, already_selected_docs)`
      * Eliminates repetitive duplicate chunks!
    """)

with col_ret2:
    st.markdown("### Standard Similarity vs. MMR Diversification")
    ret_dot = """
    digraph LangChainRetrievers {
        rankdir=TD;
        size="5,3";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=8, margin=0.08];
        edge [fontname="Helvetica", fontsize=7];

        UserQ [label="Query: 'How to deploy LLMs on GPU?'", fillcolor="#EFF6FF", color="#3B82F6", style="filled,bold"];
        VStore [label="LangChain VectorStore\\n(Chroma / FAISS / Qdrant)", fillcolor="#DBEAFE"];

        subgraph cluster_standard {
            label="Standard Similarity Search"; style="dashed"; color="#EF4444";
            TopKStd [label="Top 3 Identical Chunks\\n(All describe vLLM memory limits,\\nredundant context)", fillcolor="#FEE2E2"];
        }

        subgraph cluster_mmr {
            label="MMR Retrieval (fetch_k=10, k=3)"; style="dashed"; color="#10B981";
            MMRStep1 [label="Fetch Candidate Pool\\n(Top 10 vectors)", fillcolor="#D1FAE5"];
            MMRStep2 [label="Diverse Top 3 Chunks\\n1. vLLM VRAM\\n2. TensorRT-LLM\\n3. Quantization bits", fillcolor="#A7F3D0", color="#059669", style="filled,bold"];
        }

        UserQ -> VStore;
        VStore -> TopKStd [label="Pure nearest neighbors"];
        VStore -> MMRStep1 [label="Fetch large candidate pool"];
        MMRStep1 -> MMRStep2 [label="MMR greedy selection\\npenalizes redundancy"];
    }
    """
    st.graphviz_chart(ret_dot, use_container_width=False)

st.markdown("### Retriever Search Modes in LangChain")
st.markdown("""
| Search Type | Parameters (`search_kwargs`) | Key Function | Ideal Production Scenario |
| :--- | :--- | :--- | :--- |
| `"similarity"` | `k: int` | Retrieves top $K$ nearest vectors. | Low-latency baseline when documents are distinct. |
| `"similarity_score_threshold"` | `k: int`, `score_threshold: float` | Discards chunks with similarity $< \\tau$. | Guardrail against hallucinations on irrelevant questions. |
| `"mmr"` | `k: int`, `fetch_k: int`, `lambda_mult: float` | Balances query similarity with chunk diversity. | Complex topics where documents share repetitive phrases. |
""")

st.divider()

# Section 2: Production Code: Configuring LangChain Retrievers
st.header("2. Production Code: Initializing & Invoking Retrievers", divider="rainbow")

st.markdown("""
The production snippet below demonstrates how to configure retrievers using `.as_retriever()`, tuning MMR parameters, and invoking retrieval in modern LCEL syntax.
""")

st.code(r'''
# In production environment:
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings

def configure_retrievers_demo(vectorstore):
    # 1. Standard Top-K Retriever
    standard_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 2. Similarity Score Threshold Retriever
    threshold_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.75, "k": 5}
    )

    # 3. Maximal Marginal Relevance (MMR) Retriever
    # lambda_mult: 1.0 = pure similarity, 0.0 = pure diversity. Standard default is 0.5.
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,              # Final documents returned to LLM
            "fetch_k": 10,        # Initial candidate pool retrieved from index
            "lambda_mult": 0.65,  # 65% similarity emphasis, 35% diversity emphasis
            "filter": {"department": "engineering"} # Combined metadata filtering
        }
    )

    # Invoking in modern LangChain:
    query = "How to optimize GPU memory during inference?"
    results = mmr_retriever.invoke(query)
    for doc in results:
        print(f"[{doc.metadata.get('source')}]: {doc.page_content[:80]}...")

    return mmr_retriever
''', language="python")

st.divider()

# Section 3: Interactive Retrieval & MMR Diversification Studio
st.header("3. Interactive Workbench: MMR vs. Standard Similarity Studio", divider="rainbow")

st.markdown("""
Observe how **Maximal Marginal Relevance (MMR)** solves the **redundancy trap**.
In the corpus below, several documents contain nearly identical statements about GPU VRAM, while others discuss quantization and model parallelism.
""")

class DocumentItem:
    def __init__(self, doc_id: str, text: str, category: str):
        self.doc_id = doc_id
        self.text = text
        self.category = category

class InteractiveRetrieverEngine:
    def __init__(self, dim: int = 48):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text.lower().strip())) % (2**31 - 1))
        v = np.random.randn(self.dim).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v), 1e-12)

    def calculate_mmr(
        self,
        query: str,
        corpus: List[DocumentItem],
        k: int,
        fetch_k: int,
        lambda_mult: float
    ) -> List[Tuple[DocumentItem, float]]:
        q_vec = self.embed(query)
        
        # 1. Fetch initial candidate pool
        candidates = []
        for doc in corpus:
            d_vec = self.embed(doc.text)
            sim = float(np.dot(d_vec, q_vec))
            candidates.append((doc, d_vec, sim))
        
        # Sort by similarity to query
        candidates.sort(key=lambda x: x[2], reverse=True)
        pool = candidates[:fetch_k]

        if not pool:
            return []

        # 2. Greedy selection using MMR
        selected_docs: List[Tuple[DocumentItem, float]] = []
        selected_vecs: List[np.ndarray] = []
        remaining = pool.copy()

        # Select first item (highest query similarity)
        first_item = remaining.pop(0)
        selected_docs.append((first_item[0], first_item[2]))
        selected_vecs.append(first_item[1])

        while len(selected_docs) < k and remaining:
            best_score = -float("inf")
            best_idx = -1

            for idx, (doc, d_vec, q_sim) in enumerate(remaining):
                # Calculate maximum similarity to any already selected document
                max_sim_to_selected = max([float(np.dot(d_vec, s_vec)) for s_vec in selected_vecs])
                
                # MMR formula
                mmr_val = lambda_mult * q_sim - (1.0 - lambda_mult) * max_sim_to_selected
                
                if mmr_val > best_score:
                    best_score = mmr_val
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            selected_docs.append((chosen[0], chosen[2]))
            selected_vecs.append(chosen[1])

        return selected_docs

ret_engine = InteractiveRetrieverEngine()

sample_corpus = [
    DocumentItem("doc_vram_01", "vLLM uses PagedAttention to eliminate memory fragmentation in GPU VRAM during LLM inference.", "memory"),
    DocumentItem("doc_vram_02", "PagedAttention partitions KV cache memory dynamically, significantly reducing VRAM waste.", "memory"),
    DocumentItem("doc_vram_03", "Managing KV cache VRAM allocation with page tables boosts inference throughput.", "memory"),
    DocumentItem("doc_quant_04", "AWQ and GPTQ 4-bit weight quantization reduce GPU memory footprint by 70% with minimal perplexity loss.", "quantization"),
    DocumentItem("doc_tensor_05", "Tensor parallelism splits transformer weight matrices across multiple GPUs using Megatron-LM.", "parallelism"),
    DocumentItem("doc_spec_06", "Speculative decoding uses a smaller draft model to generate candidate tokens verified in parallel.", "decoding"),
    DocumentItem("doc_batch_07", "Continuous batching schedules incoming requests iteratively at the token level rather than sequence level.", "scheduling")
]

col_cfg, col_out = st.columns([1, 1])

with col_cfg:
    st.subheader("1. Query & Corpus Overview")
    user_q = st.text_input("Search Query:", value="How to optimize GPU memory in LLM inference?")
    
    st.write(f"**Knowledge Corpus ({len(sample_corpus)} documents):**")
    st.caption("Notice docs 1, 2, and 3 are semantically repetitive variations of PagedAttention VRAM.")
    
    st.subheader("2. Search Strategy Mode")
    search_mode = st.radio(
        "Select Retriever Search Mode:",
        ["Standard Similarity Search", "Maximal Marginal Relevance (MMR)"],
        index=1
    )

    k_items = st.slider("Documents to Return ($K$):", min_value=2, max_value=5, value=3)

    if search_mode == "Maximal Marginal Relevance (MMR)":
        fetch_k_val = st.slider("Candidate Pool (`fetch_k`):", min_value=k_items, max_value=len(sample_corpus), value=6)
        lambda_val = st.slider(
            "Lambda Multiplier (`lambda_mult`):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="1.0 = Pure similarity (identical to standard search). 0.0 = Pure diversity."
        )

with col_out:
    st.subheader("3. Retrieved Document Chunks")

    if search_mode == "Standard Similarity Search":
        # Pure similarity
        q_vec = ret_engine.embed(user_q)
        scores = []
        for d in sample_corpus:
            s = float(np.dot(ret_engine.embed(d.text), q_vec))
            scores.append((d, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = scores[:k_items]

        st.warning("⚠️ **Standard Search Result**: Redundancy detected! Multiple near-identical chunks occupy the context.")
        for rank, (doc, sim) in enumerate(results, 1):
            with st.expander(f"Rank {rank} | {doc.doc_id} | Cosine: {sim:.4f}", expanded=True):
                st.markdown(f"> {doc.text}")
                st.caption(f"Category: `{doc.category}`")

    else:
        # MMR retrieval
        results = ret_engine.calculate_mmr(
            query=user_q,
            corpus=sample_corpus,
            k=k_items,
            fetch_k=fetch_k_val,
            lambda_mult=lambda_val
        )

        st.success("✅ **MMR Search Result**: Repetitive duplicates penalized! Context window contains diverse insights.")
        for rank, (doc, sim) in enumerate(results, 1):
            with st.expander(f"Rank {rank} | {doc.doc_id} | Similarity: {sim:.4f}", expanded=True):
                st.markdown(f"> {doc.text}")
                st.caption(f"Category: `{doc.category}`")

st.divider()

# Section 4: Technical Interview Preparation
st.header("4. Technical Interview Preparation", divider="rainbow")

qa_tab, traps_tab, flashcards_tab = st.tabs(["High-Frequency Questions", "Retriever Traps", "Flashcards"])

with qa_tab:
    st.markdown("""
    **Q1: What is the exact mathematical intuition behind Maximal Marginal Relevance (MMR)?**
    * **Answer**: MMR balances two opposing objectives:
      1. Maximizing relevance of the document to the user query: `Sim(doc, query)`.
      2. Minimizing redundancy with documents already chosen: `MaxSim(doc, already_selected)`.
      The parameter `lambda_mult` (between 0.0 and 1.0) controls this tradeoff:
      * When `lambda_mult = 1.0`, MMR degenerates to pure standard similarity search.
      * When `lambda_mult = 0.5`, it equally weights relevance against novelty, ensuring each retrieved chunk adds fresh factual context to the prompt.

    **Q2: What is the architectural difference between a LangChain `VectorStore` and a `Retriever`?**
    * **Answer**:
      * A **VectorStore** is a storage engine responsible for adding, deleting, updating, and storing embeddings and payloads in a specific database index (Chroma, Pinecone, FAISS).
      * A **Retriever** is a higher-level runnable abstraction (`Runnable[str, List[Document]]`) that implements `.invoke(query)`. A retriever does not have to be backed by a vector store (e.g. it can be a Web search retriever, BM25 retriever, or multi-query ensemble).

    **Q3: Why should `fetch_k` always be significantly larger than `k` in MMR?**
    * **Answer**: MMR needs a sufficiently large pool of candidate documents to choose diverse items from. If `k=4` and `fetch_k=4`, MMR has no choices to diversify—it can only reorder the same 4 candidates. Typically, `fetch_k` should be set to 3x to 5x the value of `k` (e.g. `k=4`, `fetch_k=20`).
    """)

with traps_tab:
    st.markdown("""
    * **Trap 1: Setting `fetch_k` Equal to `k` in MMR**:
      * *Issue*: Configuring `.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 4})`.
      * *Consequence*: MMR has zero candidate diversity to select from, behaving identically to standard similarity search while incurring MMR overhead.
      * *Fix*: Always set `fetch_k >= 3 * k`.

    * **Trap 2: Setting `lambda_mult` Too Low (< 0.2)**:
      * *Issue*: Setting `lambda_mult = 0.1` to force maximum diversity.
      * *Consequence*: The retriever prioritizes novelty so aggressively that it retrieves completely irrelevant off-topic documents just because they are different from the first document.
      * *Fix*: Maintain `lambda_mult` between 0.5 and 0.7 for optimal balance.

    * **Trap 3: Using `similarity_score_threshold` with Incompatible Metric Spaces**:
      * *Issue*: Setting `score_threshold=0.8` on an L2 Euclidean vector store where distances range from 0.0 to 100.0.
      * *Consequence*: All documents are discarded or all documents pass.
      * *Fix*: Verify whether your vector store returns normalized cosine similarities before applying score thresholds.
    """)

with flashcards_tab:
    st.markdown("""
    * **Flashcard 1: Maximal Marginal Relevance (MMR)**
      * *Definition*: A retrieval algorithm that iteratively selects candidates that maximize relevance to the query while minimizing similarity to already selected documents.
    * **Flashcard 2: `vectorstore.as_retriever()`**
      * *Definition*: The method that wraps a static vector store into a LangChain `Runnable` supporting LCEL pipelines and search algorithms.
    * **Flashcard 3: `lambda_mult`**
      * *Definition*: The tuning factor in MMR (0 to 1) balancing relevance to the query against diversity across retrieved documents.
    * **Flashcard 4: `fetch_k`**
      * *Definition*: The initial candidate pool size retrieved from the vector index before applying MMR selection.
    """)

st.divider()

# Section 5: Self-Assessment Checklist
st.header("5. Self-Assessment Checklist", divider="rainbow")
st.markdown("""
Verify your mastery of LangChain Retrieval:

- [ ] I can articulate the difference between a `VectorStore` and a `Retriever`.
- [ ] I understand how Maximal Marginal Relevance (MMR) solves context redundancy.
- [ ] I know how `lambda_mult` and `fetch_k` govern MMR diversification.
- [ ] I can configure score thresholding to prevent prompt hallucinations.
- [ ] I understand how to invoke retrievers in modern LangChain LCEL chains.
""")
