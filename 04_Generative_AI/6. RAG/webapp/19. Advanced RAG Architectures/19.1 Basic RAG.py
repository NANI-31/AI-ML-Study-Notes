import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="19.1 Basic RAG Architecture",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class BasicRAGEngine:
    """Canonical Linear Naive RAG Pipeline: Query -> Retrieve -> Augment -> Generate."""

    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None):
        if corpus is None:
            self.corpus = [
                {
                    "id": "doc_1",
                    "title": "Quantum Error Correction (Surface Codes)",
                    "content": "Surface codes require a physical error rate below the fault-tolerant threshold of 1.0% to achieve logical qubit suppression. Syndrome measurement is performed using stabilizer operators."
                },
                {
                    "id": "doc_2",
                    "title": "Cryogenic Control Electronics",
                    "content": "Superconducting qubits operate at dilution refrigerator temperatures of 15 millikelvin. Control pulses are delivered via coaxial lines with cryogenic attenuators to minimize thermal noise."
                },
                {
                    "id": "doc_3",
                    "title": "Quantum Memory Coherence",
                    "content": "Transmon qubit relaxation times (T1) and dephasing times (T2) are limited by dielectric loss in capacitive shunts and two-level system defects at metal-substrate interfaces."
                },
                {
                    "id": "doc_4",
                    "title": "Semiconductor Spin Qubits",
                    "content": "Silicon spin qubits demonstrate gate fidelities above 99.5% at elevated temperatures of 1 Kelvin, simplifying dilution cooling interconnect architectures."
                },
                {
                    "id": "doc_5",
                    "title": "Cloud Quantum Infrastructure",
                    "content": "Commercial quantum access is scheduled through quantum circuit compilers that decompose unitary gates into native hardware basis gates such as CZ and sqrt(X)."
                }
            ]
        else:
            self.corpus = corpus

    def _compute_mock_embedding(self, text: str) -> np.ndarray:
        """Deterministic pseudo-dense embedding projection based on character trigram hashing."""
        vec = np.zeros(64, dtype=float)
        words = text.lower().split()
        for i, word in enumerate(words):
            val = sum(ord(c) for c in word)
            idx = val % 64
            vec[idx] += 1.0 / (i + 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Naive top-k cosine similarity retrieval over static corpus embeddings."""
        query_vec = self._compute_mock_embedding(query)
        scored_candidates = []

        for doc in self.corpus:
            doc_vec = self._compute_mock_embedding(doc["content"] + " " + doc["title"])
            sim = float(np.dot(query_vec, doc_vec))
            scored_candidates.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(sim, 4)
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def augment_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """Unfiltered string concatenation of retrieved chunks into the prompt context."""
        if system_prompt is None:
            system_prompt = "You are a helpful technical assistant. Answer the user prompt strictly using the provided context."

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_blocks.append(f"[Document {i} - {doc['title']}]:\n{doc['content']}")

        joined_context = "\n\n".join(context_blocks)
        
        prompt = (
            f"=== SYSTEM INSTRUCTION ===\n{system_prompt}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{joined_context}\n\n"
            f"=== USER QUERY ===\n{query}\n\n"
            f"=== RESPONSE ==="
        )
        return prompt

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate generation from retrieved context and audit linear failure modes."""
        if not retrieved_docs or retrieved_docs[0]["score"] < 0.20:
            generation = "I am unable to answer this question based on the provided documents as no relevant context was found."
            diagnostics = ["Zero-hit or low relevance retrieval", "Parametric knowledge fallback risk"]
        else:
            top_doc = retrieved_docs[0]
            generation = (
                f"Based on {top_doc['title']}, {top_doc['content'][:140]}... "
                f"This directly addresses your query regarding '{query}'."
            )
            diagnostics = [
                "Unfiltered context stuffed directly into context window",
                "Absence of cross-encoder reranking or relevance verification",
                "Vulnerable to lost-in-the-middle attention decay if top_k > 3"
            ]

        token_estimate = len(query.split()) + sum(len(d["content"].split()) for d in retrieved_docs) + 45
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generation,
            "estimated_prompt_tokens": token_estimate,
            "diagnostics": diagnostics
        }


# --- Streamlit Layout ---

st.title("19.1 Basic RAG Architecture")
st.caption("Canonical Linear Pipeline: Query Formulation, Vector Retrieval, Context Augmentation & Generative Synthesis")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Canonical Linear RAG Architecture (Single-Path DAG)", divider="rainbow")

st.markdown(
    """
    **Basic RAG** (also referred to as *Naive RAG* or the *Standard Linear Pipeline*) is the foundational paradigm of retrieval-augmented generation.
    It operates as a strictly linear, feed-forward Directed Acyclic Graph (DAG) consisting of three sequential phases: **Query Embedding**, **Vector Retrieval**, and **Context-Augmented Generation**.
    
    While conceptually simple and fast to prototype, the basic pipeline lacks dynamic feedback loops, query refinement, reranking, and self-reflection mechanisms.
    The diagram below models the complete end-to-end execution path of the canonical linear RAG architecture.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Basic_RAG_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_indexing {
            label = "OFFLINE INDEXING PIPELINE (ONE-TIME / BATCH)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            raw_docs [label="Raw Knowledge Documents\\n(PDFs, Markdown, HTML)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            chunking [label="Static Chunking\\n(e.g., Fixed 512 Tokens, 50 Overlap)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            embed_model [label="Embedding Model\\n(Dense Vector Encoding)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            vector_db [label="Vector Database\\n(Flat / HNSW Index)", shape=cylinder, fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            raw_docs -> chunking -> embed_model -> vector_db;
        }

        subgraph cluster_online_runtime {
            label = "ONLINE RUNTIME PIPELINE (SYNCHRONOUS LINEAR DAG)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="1. User Input Query\\n'What is the error threshold?'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_embed [label="2. Query Vectorizer\\n(Identical Embedding Model)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            ann_search [label="3. Approximate Nearest Neighbor Search\\n(Cosine Similarity / Dot Product)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            top_k_chunks [label="4. Top-K Chunks Selection\\n(Static Slice: k=3 or k=5)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            prompt_stuffing [label="5. Prompt Augmentation\\n(Stuffed System + Context + Query)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            llm_gen [label="6. Frontier LLM Synthesis\\n(Autoregressive Generation)", fillcolor="#0F172A", fontcolor="#10B981"];
            client_response [label="7. Final User Response\\n(Unfiltered Completion)", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            user_query -> query_embed -> ann_search;
            ann_search -> top_k_chunks;
            top_k_chunks -> prompt_stuffing;
            user_query -> prompt_stuffing [label="Original Query Injected"];
            prompt_stuffing -> llm_gen -> client_response;
        }

        vector_db -> ann_search [label="Index Probe", style=dashed, color="#38BDF8"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: QUERY -> RETRIEVE -> GENERATE ---
st.header("2. Query, Retrieve, Generate (Linear DAG Execution, Naive Vector Search, and Unfiltered Prompt Augmentation)", divider="rainbow")

st.markdown(
    """
    The canonical linear RAG architecture executes strictly in three sequential steps.
    Each step makes major engineering assumptions that represent both its operational simplicity and its fundamental vulnerabilities.
    
    ### Step-by-Step Mechanics of the Canonical Flow
    1. **Query (Single-Vector Representation)**:
       - The user prompt is taken verbatim without transformation, spell-checking, or query expansion.
       - A dense embedding model maps the query string `q` into a fixed `d`-dimensional vector `v_q in R^d`.
       - *Assumption*: The user query uses vocabulary semantically identical to the document chunks and contains complete contextual intent.
    2. **Retrieve (Unfiltered Cosine Similarity)**:
       - The vector database computes the inner product or cosine distance between `v_q` and all stored chunk vectors `v_i`.
       - The top `K` candidates are returned based strictly on geometric proximity.
       - *Assumption*: High geometric vector proximity directly equals factual relevance and informational utility.
    3. **Generate (Direct Context Stuffing)**:
       - The raw text of all `K` retrieved chunks is concatenated sequentially into the LLM system/user prompt template.
       - The LLM performs autoregressive decoding to generate an answer.
       - *Assumption*: The LLM pays uniform attention to all retrieved chunks and will not hallucinate when context is partially incomplete.
    """
)

st.subheader("Why Basic RAG Fails in Production: The 5 Canonical Failure Modes")

col_fm1, col_fm2 = st.columns(2)
with col_fm1:
    with st.container(border=True):
        st.markdown("#### 1. Out-of-Domain Retrieval Collapse (Zero Hits)")
        st.markdown(
            """
            When a user submits queries with domain-specific acronyms, typos, or conceptual abstractions not present in the vector training set,
            vector distances collapse into an uninformative cluster. Naive RAG still passes the highest-scoring chunks, feeding **pure noise** to the LLM.
            """
        )
    with st.container(border=True):
        st.markdown("#### 2. Context Stuffing & Token Bloat")
        st.markdown(
            """
            Statically injecting `K=5` or `K=10` large chunks fills 4,000+ prompt tokens per query.
            90% of the injected tokens consist of boilerplate headers, legal disclaimers, or tangential prose, driving up API costs and TTFT latency.
            """
        )
with col_fm2:
    with st.container(border=True):
        st.markdown("#### 3. Lost-in-the-Middle Phenomenon")
        st.markdown(
            """
            Transformer self-attention exhibits significant positional bias: models attend heavily to tokens at the very beginning and very end of the prompt.
            Crucial factual evidence positioned in middle chunks (e.g., Chunk 3 of 5) is frequently ignored by the LLM.
            """
        )
    with st.container(border=True):
        st.markdown("#### 4. Semantic Disconnect & Chunk Fragmentation")
        st.markdown(
            """
            Fixed-length chunking (e.g., 500 tokens) slices tables, code snippets, and sentences mid-clause.
            The retriever fetches a fragment lacking its core subject noun, forcing the LLM to hallucinate the missing antecedent.
            """
        )

# --- SECTION 3: PRODUCTION IMPLEMENTATION ---
st.header("3. Production Implementation: Canonical Naive RAG Engine (`BasicRAGEngine`)", divider="rainbow")

st.markdown(
    """
    Below is the complete, self-contained reference implementation of the canonical `BasicRAGEngine`.
    It implements vector projection, top-K cosine similarity search, prompt template formatting, and linear generative synthesis.
    """
)

st.code(
    '''
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class BasicRAGEngine:
    """Canonical Linear Naive RAG Pipeline: Query -> Retrieve -> Augment -> Generate."""

    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None):
        if corpus is None:
            self.corpus = [
                {
                    "id": "doc_1",
                    "title": "Quantum Error Correction (Surface Codes)",
                    "content": "Surface codes require a physical error rate below the fault-tolerant threshold of 1.0% to achieve logical qubit suppression. Syndrome measurement is performed using stabilizer operators."
                },
                {
                    "id": "doc_2",
                    "title": "Cryogenic Control Electronics",
                    "content": "Superconducting qubits operate at dilution refrigerator temperatures of 15 millikelvin. Control pulses are delivered via coaxial lines with cryogenic attenuators to minimize thermal noise."
                },
                {
                    "id": "doc_3",
                    "title": "Quantum Memory Coherence",
                    "content": "Transmon qubit relaxation times (T1) and dephasing times (T2) are limited by dielectric loss in capacitive shunts and two-level system defects at metal-substrate interfaces."
                },
                {
                    "id": "doc_4",
                    "title": "Semiconductor Spin Qubits",
                    "content": "Silicon spin qubits demonstrate gate fidelities above 99.5% at elevated temperatures of 1 Kelvin, simplifying dilution cooling interconnect architectures."
                },
                {
                    "id": "doc_5",
                    "title": "Cloud Quantum Infrastructure",
                    "content": "Commercial quantum access is scheduled through quantum circuit compilers that decompose unitary gates into native hardware basis gates such as CZ and sqrt(X)."
                }
            ]
        else:
            self.corpus = corpus

    def _compute_mock_embedding(self, text: str) -> np.ndarray:
        """Deterministic pseudo-dense embedding projection based on character trigram hashing."""
        vec = np.zeros(64, dtype=float)
        words = text.lower().split()
        for i, word in enumerate(words):
            val = sum(ord(c) for c in word)
            idx = val % 64
            vec[idx] += 1.0 / (i + 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Naive top-k cosine similarity retrieval over static corpus embeddings."""
        query_vec = self._compute_mock_embedding(query)
        scored_candidates = []

        for doc in self.corpus:
            doc_vec = self._compute_mock_embedding(doc["content"] + " " + doc["title"])
            sim = float(np.dot(query_vec, doc_vec))
            scored_candidates.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "score": round(sim, 4)
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def augment_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> str:
        """Unfiltered string concatenation of retrieved chunks into the prompt context."""
        if system_prompt is None:
            system_prompt = "You are a helpful technical assistant. Answer the user prompt strictly using the provided context."

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_blocks.append(f"[Document {i} - {doc['title']}]:\\n{doc['content']}")

        joined_context = "\\n\\n".join(context_blocks)
        
        prompt = (
            f"=== SYSTEM INSTRUCTION ===\\n{system_prompt}\\n\\n"
            f"=== RETRIEVED CONTEXT ===\\n{joined_context}\\n\\n"
            f"=== USER QUERY ===\\n{query}\\n\\n"
            f"=== RESPONSE ==="
        )
        return prompt

    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate generation from retrieved context and audit linear failure modes."""
        if not retrieved_docs or retrieved_docs[0]["score"] < 0.20:
            generation = "I am unable to answer this question based on the provided documents as no relevant context was found."
            diagnostics = ["Zero-hit or low relevance retrieval", "Parametric knowledge fallback risk"]
        else:
            top_doc = retrieved_docs[0]
            generation = (
                f"Based on {top_doc['title']}, {top_doc['content'][:140]}... "
                f"This directly addresses your query regarding '{query}'."
            )
            diagnostics = [
                "Unfiltered context stuffed directly into context window",
                "Absence of cross-encoder reranking or relevance verification",
                "Vulnerable to lost-in-the-middle attention decay if top_k > 3"
            ]

        token_estimate = len(query.split()) + sum(len(d["content"].split()) for d in retrieved_docs) + 45
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generation,
            "estimated_prompt_tokens": token_estimate,
            "diagnostics": diagnostics
        }
''',
    language="python"
)

# --- SECTION 4: INTERACTIVE WORKBENCH ---
st.header("4. Interactive Workbench: Basic RAG Flow Simulator & Failure Mode Demonstrator", divider="rainbow")

st.markdown(
    """
    Test the canonical linear RAG flow with custom queries and parameter settings.
    Observe how naive top-K retrieval directly influences the augmented prompt and inspect failure mode diagnostics.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("⚙️ Query & Pipeline Controls")
    
    preset_query = st.selectbox(
        "Select Query Preset:",
        [
            "Preset 1: What is the fault-tolerant threshold for surface codes?",
            "Preset 2: At what temperature do superconducting qubits operate?",
            "Preset 3: What are the gate fidelities of silicon spin qubits?",
            "Preset 4 (Adversarial / OOD): How do photosynthetic bacteria fix carbon?"
        ],
        index=0
    )
    
    query_text = st.text_input("Active Query:", preset_query)
    top_k_val = st.slider("Retrieval Top-K Slice:", min_value=1, max_value=5, value=3, step=1)
    
    custom_system_prompt = st.text_area(
        "System Prompt Template:",
        "You are an enterprise AI assistant. Answer the user question based strictly on the retrieved documents. Do not speculate.",
        height=75
    )

with wb_col2:
    st.subheader("🔍 Execution Telemetry & Output")
    
    engine = BasicRAGEngine()
    retrieved_docs = engine.retrieve(query_text, top_k=top_k_val)
    augmented_prompt = engine.augment_prompt(query_text, retrieved_docs, custom_system_prompt)
    result = engine.generate(query_text, retrieved_docs)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Retrieved Chunks", len(retrieved_docs))
    m1_score = retrieved_docs[0]["score"] if retrieved_docs else 0.0
    m2.metric("Top-1 Cosine Score", f"{m1_score:.4f}")
    m3.metric("Est. Prompt Tokens", result["estimated_prompt_tokens"])
    
    st.markdown("#### Generated Answer")
    st.info(result["generated_answer"])
    
    st.markdown("#### Vulnerability Diagnostics")
    for diag in result["diagnostics"]:
        st.caption(f"⚠️ {diag}")

st.markdown("### Retrieved Candidate Audit Table")
retrieval_df = pd.DataFrame([
    {
        "Rank": i + 1,
        "Document ID": doc["id"],
        "Title": doc["title"],
        "Cosine Score": doc["score"],
        "Content Snippet": doc["content"][:100] + "..."
    }
    for i, doc in enumerate(retrieved_docs)
])
st.dataframe(retrieval_df, use_container_width=True)

with st.expander("📄 View Fully Formatted Augmented Prompt Stuffed to LLM", expanded=False):
    st.text(augmented_prompt)

# --- SECTION 5: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("5. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: What are the core architectural limitations of Basic (Naive) RAG?**"):
    st.markdown(
        """
        The core limitations of Basic RAG stem from its rigid, feed-forward, single-path execution:
        1. **Zero Query Transformation**: Inability to handle ambiguous, multi-part, or vocabulary-mismatched user questions.
        2. **Uncalibrated Vector Search**: Cosine similarity measures topical overlap, not factual answering capability.
        3. **Missing Reranker**: Documents are ranked by bi-encoder representations without cross-attention verification.
        4. **Uncompressed Context**: Injects irrelevant tokens, increasing latency and cost while causing lost-in-the-middle degradation.
        5. **No Verification / Self-Reflection**: Incapable of verifying whether the generation is grounded or halluncinated.
        """
    )

with st.expander("**Q2: What is the 'Lost in the Middle' phenomenon and how does it manifest in Basic RAG?**"):
    st.markdown(
        """
        Research (Liu et al., 2023) demonstrated that transformer language models possess an attention U-curve:
        retrieval performance is highest when relevant information is at the very beginning (primacy effect) or very end (recency effect) of the context window.
        
        In Basic RAG, retrieved chunks are concatenated linearly in rank order. If 5 to 10 chunks are passed, documents ranked 3 through 7 are positioned in the middle.
        The LLM's attention heads frequently fail to route information from these middle tokens into the generation, causing silent retrieval failure even when the correct document was retrieved.
        """
    )

with st.expander("**Q3: Why is cosine similarity in a bi-encoder insufficient for determining final document relevance?**"):
    st.markdown(
        """
        Bi-encoders embed the query and candidate documents independently into static vectors:
        `q = f(Q)` and `d = f(D)`. The similarity is computed as `q . d`.
        
        Because the model cannot compute joint token-to-token attention between `Q` and `D`, it misses:
        - Negation (e.g., *"not approved by FDA"* shares high vector similarity with *"approved by FDA"*).
        - Specific numeric or date constraints.
        - Subtle entity relationships.
        
        A cross-encoder reranker is required to compute all-to-all cross-attention across the combined prompt.
        """
    )

with st.expander("**Q4: How does fixed-size chunking degrade downstream generation quality in Naive RAG?**"):
    st.markdown(
        """
        Fixed-size chunking (e.g., 512 characters with 50 overlap) ignores semantic boundaries:
        - Paragraphs and thoughts are bisected mid-argument.
        - Tables and markdown lists have rows separated from column headers.
        - Pronouns (e.g., *"He was born in 1945"*) are detached from their named entities (*"Albert Einstein"*).
        
        When the retriever fetches an orphaned chunk, the LLM must either hallucinate the missing subject or reject the context.
        """
    )

with st.expander("**Q5: What is the computational complexity of the naive linear RAG DAG?**"):
    st.markdown(
        """
        - **Embedding Generation**: `O(L_q)` where `L_q` is query token length via transformer feed-forward passes.
        - **Vector Retrieval**: `O(M * log N)` using HNSW approximate nearest neighbors, or `O(N * d)` brute-force flat search over `N` vectors.
        - **Prompt Preparation**: `O(K * L_c)` string concatenation where `L_c` is average chunk length.
        - **Generation (LLM)**: `O(P^2 + G * P)` where `P` is total prompt length and `G` is generated token length due to transformer attention.
        The generation stage dominates total pipeline latency and financial cost.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Deploying Without a Similarity Score Safety Floor**  
        **Antipattern**: Statically accepting the Top-K chunks regardless of how low their cosine similarity score is.  
        **Consequence**: Out-of-domain queries retrieve irrelevant documents with scores like `0.35`, forcing the LLM to invent hallucinations from garbage context.  
        **Production Fix**: Enforce a strict minimum similarity threshold (e.g., `score >= 0.65`); if not met, gracefully abstain.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Context Stuffing to Maximize Recall**  
        **Antipattern**: Increasing Top-K from 3 to 15 chunks under the naive assumption that more context always equals better answers.  
        **Consequence**: Quadruples prompt token consumption, inflates TTFT latency to several seconds, and triggers catastrophic 'lost-in-the-middle' degradation.  
        **Production Fix**: Retrieve 20 candidates, but use a cross-encoder reranker to prune down to the top 3-4 highly relevant chunks.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Arbitrary Character-Based Chunking**  
        **Antipattern**: Slicing documents into fixed 500-character blocks using string index slicing.  
        **Consequence**: Cuts words, numbers, and tabular data in half, corrupting semantic embeddings and destroying factual precision.  
        **Production Fix**: Employ recursive character text splitters or AST-based semantic chunkers that respect sentence and markdown boundaries.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Omitting Document Metadata from Context Prompts**  
        **Antipattern**: Injecting raw chunk text into the prompt without document titles, timestamps, or section headers.  
        **Consequence**: The LLM cannot cite sources accurately and cannot disambiguate conflicting statements across different document versions.  
        **Production Fix**: Prepend clear structured headers to every injected chunk: `[Document ID | Title | Date]`.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What defines Basic (Naive) RAG?")
        st.markdown("**Back**: A strictly linear, single-path DAG pipeline consisting of Query Embedding, Top-K Vector Retrieval, Context Augmentation, and LLM Generation with zero feedback loops.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What causes Lost-in-the-Middle?")
        st.markdown("**Back**: Positional attention bias in transformer architectures, where models heavily weight information at the beginning and end of long contexts while neglecting the middle.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: Why is Bi-Encoder search insufficient?")
        st.markdown("**Back**: Bi-encoders embed query and document separately, preventing token-to-token cross-attention and missing subtle negation, numerical constraints, and complex logic.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Context Stuffing?")
        st.markdown("**Back**: The antipattern of injecting numerous uncompressed chunks into the prompt, resulting in high latency, increased token expense, and diluted attention.")

# --- SECTION 6: SELF-ASSESSMENT CHECKLIST ---
st.header("6. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of the canonical Basic RAG architecture and its operational failure modes:")

chk1 = st.checkbox("I understand the end-to-end execution flow of the linear Query -> Retrieve -> Generate DAG.", value=True)
chk2 = st.checkbox("I can identify the five canonical failure modes of Basic RAG (OOD collapse, token bloat, lost-in-the-middle, chunk fragmentation, hallucinations).", value=True)
chk3 = st.checkbox("I know why bi-encoder cosine similarity alone cannot guarantee document relevance without cross-encoder reranking.", value=True)
chk4 = st.checkbox("I understand how the lost-in-the-middle positional attention curve impacts large retrieved context windows.", value=True)
chk5 = st.checkbox("I can implement a reference BasicRAGEngine in Python with mock embeddings and prompt formatting.", value=True)
chk6 = st.checkbox("I know how to set similarity safety floors and configure structured document attribution headers to prevent basic RAG traps.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Congratulations! You have mastered the fundamentals and engineering limitations of Basic RAG.")
