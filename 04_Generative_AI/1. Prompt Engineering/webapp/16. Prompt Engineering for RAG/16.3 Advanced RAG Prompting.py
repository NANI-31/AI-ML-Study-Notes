import streamlit as st
import pandas as pd

# ==============================================================================
# Page Configuration & Master Styling
# ==============================================================================
st.set_page_config(
    page_title="16.3 Advanced RAG Prompting | GenAI Roadmap",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        will-change: transform;
        contain: layout style;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .adv-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.title("⚡ 16.3 Advanced RAG Prompting — Query Rewriting, HyDE & Reranking")
st.markdown(
    """
Welcome to **Lesson 16.3: Advanced RAG Prompting**.  
Naive RAG (simply embedding a short user query and pulling top-k vectors) frequently fails in production. Real-world user questions are often vague, ambiguous, or use different vocabulary than internal technical manuals.

Advanced RAG transforms the retrieval pipeline into an intelligent workflow: leveraging **Query Rewriting**, **Hypothetical Document Embeddings (HyDE)**, **Multi-Query Expansion**, **Hybrid Dense-Sparse Search**, and **Cross-Encoder Reranking** to ensure only the highest-fidelity evidence enters the context window.
"""
)

st.divider()

# ==============================================================================
# Section 1: The 7 Pillars of Advanced RAG Orchestration
# ==============================================================================
st.header("1. The 7 Pillars of Advanced RAG Orchestration", divider="rainbow")
st.markdown("Techniques designed to eliminate retrieval failures and vocabulary mismatches:")

adv_rag_df = pd.DataFrame({
    "Advanced Technique": [
        "1. Query Rewriting",
        "2. Query Expansion",
        "3. Multi-Query Retrieval",
        "4. HyDE (Hypothetical Doc Embeddings)",
        "5. Hybrid Search (Dense + BM25)",
        "6. Cross-Encoder Reranking",
        "7. Contextual Compression"
    ],
    "Architectural Mechanism": [
        "Resolves coreferences and chat history ambiguity into a standalone self-contained question",
        "Generates related synonyms, technical jargon, and domain keywords to broaden search breadth",
        "Generates 3-5 distinct query variations to search the vector database from multiple perspectives",
        "Prompts the LLM to hallucinate a synthetic hypothetical answer, then embeds that answer for retrieval",
        "Combines semantic vector embeddings with exact keyword BM25 scoring via Reciprocal Rank Fusion (RRF)",
        "Scores query-document pairs jointly with a heavy cross-encoder (e.g. Cohere Rerank, BGE-Reranker)",
        "Extracts only the 2-3 most relevant sentences from retrieved chunks, dropping irrelevant fluff"
    ],
    "Solves What Failure Mode?": [
        "Vague follow-ups like 'What did you mean by that?' that fail vector similarity",
        "Vocabulary mismatch between layperson users and dense technical documentation",
        "Single-angle queries that miss documents using alternative terminology",
        "The semantic distance between a short query and an 800-word explanatory paragraph",
        "Vector search failing on exact part numbers, acronyms, or specific error codes",
        "Bi-encoder vector search returning topically related but non-answering distractors",
        "Context window saturation and 'Lost in the Middle' attention dispersion"
    ]
})
st.dataframe(adv_rag_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# Section 2: Deep Dive: HyDE & Query Transformation
# ==============================================================================
st.header("2. Deep Dive: Hypothetical Document Embeddings (HyDE)", divider="rainbow")

col_hd1, col_hd2 = st.columns(2)

with col_hd1:
    with st.container(border=True):
        st.subheader("💡 Why HyDE Outperforms Naive Query Embedding")
        st.markdown(
            """
- **The Problem:** A user query is short (e.g. 6 words: *"How do I fix Postgres error 57P01?"*). A technical documentation chunk is long (400 words of architectural prose).
- In vector space, a short interrogative question often sits far away from an authoritative declarative explanation.
- **The HyDE Solution:**
  1. Prompt an LLM zero-shot: *"Write a hypothetical technical paragraph explaining how to fix Postgres error 57P01."*
  2. The generated hypothetical passage contains the relevant domain vocabulary (e.g. `pg_terminate_backend`, `FATAL: terminating connection`, `admin intervention`).
  3. Embed the **hypothetical passage** instead of the query!
  4. Search the vector DB: the hypothetical document matches actual documentation chunks with substantially higher cosine similarity.
"""
        )

with col_hd2:
    with st.container(border=True):
        st.subheader("🔀 Multi-Query Parallel Retrieval")
        st.markdown(
            """
- Real users formulate queries unpredictably.
- **Multi-Query Prompt:** Ask the LLM to generate 3 diverse perspectives:
  - *Query 1 (Conceptual):* "PostgreSQL unexpected connection termination reasons"
  - *Query 2 (Operational):* "Diagnosing error code 57P01 in AWS RDS Aurora"
  - *Query 3 (Remediation):* "Preventing idle client timeout terminations in PostgreSQL"
- Execute all 3 queries concurrently against the vector DB and take the union of unique chunks.
"""
        )

st.divider()

# ==============================================================================
# Section 3: Architecture Workflow Diagram
# ==============================================================================
st.header("3. Architecture: The Advanced RAG Retrieval Pipeline", divider="rainbow")
st.markdown("How enterprise RAG combines multi-query generation, hybrid retrieval, and cross-encoder reranking:")

st.graphviz_chart(
    """
    digraph G {
        graph [rankdir=LR, bgcolor="transparent"]
        node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.15,0.1"]
        edge [fontname="Helvetica", fontsize=9]

        raw_q [label="Ambiguous Query\n& Chat History", fillcolor="#1E293B", fontcolor="#F8FAFC"]
        rewriter [label="Query Rewriter\n& HyDE Generator", fillcolor="#0F766E", fontcolor="#FFFFFF"]
        hybrid [label="Hybrid Search Engine\n(BM25 + Dense Vectors)", fillcolor="#6366F1", fontcolor="#FFFFFF"]
        rrf [label="Reciprocal Rank Fusion\n(Top-25 Candidates)", fillcolor="#EAB308", fontcolor="#000000"]
        reranker [label="Cross-Encoder Reranker\n(Top-3 Golden Chunks)", fillcolor="#3B82F6", fontcolor="#FFFFFF"]
        gen [label="Final Grounded LLM\nGeneration with Citations", fillcolor="#10B981", fontcolor="#FFFFFF"]

        raw_q -> rewriter -> hybrid -> rrf -> reranker -> gen
    }
    """,
    use_container_width=False,
)

st.divider()

# ==============================================================================
# Section 4: Interactive Advanced RAG Pipeline Simulator
# ==============================================================================
st.header("4. Interactive Advanced RAG Pipeline Simulator", divider="rainbow")
st.markdown("Trace how an ambiguous engineering question is transformed across advanced RAG stages:")

test_query = st.text_input("Enter Ambiguous Developer Question", value="Why did our redis cluster drop keys?")

col_ps1, col_ps2 = st.columns([1, 1])

with col_ps1:
    st.subheader("⚙️ Stage 1 & 2: Rewriting & HyDE Generation")
    
    st.markdown("**1. Standalone Query Rewriting:**")
    rewritten_q = "Redis cluster unexpected key eviction root causes and memory policy"
    st.code(rewritten_q, language="text")

    st.markdown("**2. Generated HyDE Hypothetical Passage:**")
    hyde_passage = """Redis clusters evict keys unexpectedly when the maxmemory limit is reached. 
Depending on the configured maxmemory-policy (such as allkeys-lru, volatile-lru, or noeviction), 
Redis will begin evicting keys to reclaim memory. Keys with short TTLs may also be automatically 
purged by the active and passive expiration algorithms."""
    st.code(hyde_passage, language="text")

with col_ps2:
    st.subheader("🎯 Stage 3 & 4: Hybrid Search & Reranking")
    
    st.markdown("**Candidate Chunks Ranked by Cross-Encoder:**")
    rerank_results = pd.DataFrame({
        "Rank": [1, 2, 3],
        "Source Document": ["redis_memory_architecture.md", "cluster_eviction_policies.md", "redis_tls_configuration.md"],
        "Cross-Encoder Score": [0.94, 0.88, 0.12],
        "Action Taken": ["KEPT (Injected into Context)", "KEPT (Injected into Context)", "PRUNED (Distractor Removed)"]
    })
    st.dataframe(rerank_results, use_container_width=True, hide_index=True)
    st.success("✅ The distractor document (TLS configuration) was pruned despite sharing keywords, saving 450 tokens!")

st.divider()

# ==============================================================================
# Section 5: Production Python Code Patterns
# ==============================================================================
st.header("5. Production Implementation Patterns: HyDE Prompt & Pipeline", divider="rainbow")
st.markdown("Python script demonstrating programmatic HyDE generation and multi-query synthesis:")

st.code(
    '''from typing import List

HYDE_SYSTEM_PROMPT = """You are a technical knowledge synthesizer.
Given a user query, write a hypothetical authoritative technical paragraph that directly answers it.
Do not include preambles like "Here is an answer". Output ONLY the hypothetical passage.
"""

MULTI_QUERY_PROMPT = """You are an AI search assistant.
Generate exactly 3 diverse rephrasings of the following user question to retrieve relevant documentation.
Output one query per line without numbering.
Question: {question}
"""

def generate_hyde_prompt(query: str) -> str:
    return f"""<system>
{HYDE_SYSTEM_PROMPT}
</system>
<user_query>
{query}
</user_query>"""

# Example execution
sample_q = "How do we rotate AWS IAM access keys with zero downtime?"
print("HyDE Prompt:")
print(generate_hyde_prompt(sample_q))
''',
    language="python",
)

st.divider()

# ==============================================================================
# Section 6: Technical Interview Preparation
# ==============================================================================
st.header("6. Technical Interview Preparation", divider="rainbow")

tab_q, tab_t, tab_f = st.tabs(["💡 High-Frequency Q&As", "⚠️ Candidate Traps", "📇 Flashcards"])

with tab_q:
    with st.expander("Q1: What is the architectural difference between a Bi-Encoder and a Cross-Encoder in RAG?", expanded=True):
        st.markdown(
            """
- **Bi-Encoder (Vector Embeddings)**: Encodes the query and document separately into static dense vectors. Similarity is calculated using dot product or cosine similarity. It is **extremely fast** ($O(1)$ lookup via ANN indexes across millions of documents), but cannot model complex token interactions between query and text.
- **Cross-Encoder (Reranker)**: Feeds the query and document together into the transformer as a single concatenated input (`[CLS] query [SEP] document [SEP]`), computing full self-attention across all tokens. It is **highly accurate** at capturing subtle semantic relevance, but computationally heavy.
- **Production Paradigm**: Retrieve top-50 candidates cheaply with a Bi-Encoder, then rerank them down to top-3 with a Cross-Encoder.
"""
        )
    with st.expander("Q2: What is Reciprocal Rank Fusion (RRF) and why is it used in Hybrid Search?"):
        st.markdown(
            """
In Hybrid Search, BM25 returns keyword scores (e.g. 14.5) while vector search returns cosine similarity scores (e.g. 0.82). Because these score distributions cannot be normalized directly, **Reciprocal Rank Fusion (RRF)** combines them using **ordinal ranks** rather than raw scores:
`RRF_Score(d) = sum(1 / (k + rank(d)))`
This robustly merges keyword matches with semantic vectors without requiring fine-tuned weight multipliers.
"""
        )

with tab_t:
    with st.expander("Trap 1: 'Can HyDE cause hallucinations if the hypothetical passage is factually wrong?'", expanded=True):
        st.markdown(
            """
- **Naive Response**: *"Yes, if the hypothetical document has errors, the final answer will be corrupted."*
- **Winning Response**: *"No. The hypothetical document generated in HyDE is **never shown to the user** and is **never passed to the final generation prompt**. It is used purely as an intermediate search vector to query the vector database. The final generation prompt is grounded strictly on the actual retrieved enterprise documents, not the hypothetical text. If the hypothetical text contains small inaccuracies, it still contains the correct domain vocabulary, which succeeds at pulling the true documentation."*
"""
        )

with tab_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**HyDE (Hypothetical Doc Embeddings)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** Generating a synthetic answer to a query and embedding that answer to retrieve documents with higher semantic similarity.")
    with c2:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Cross-Encoder Reranker**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** A model that processes query and document simultaneously to calculate deep relevance, reranking initial vector retrieval candidates.")
    with c3:
        with st.container(border=True):
            st.markdown("#### 📇 Front")
            st.markdown("**Reciprocal Rank Fusion (RRF)**")
            with st.popover("Reveal Back"):
                st.markdown("**Back:** An algorithm combining ranked results from disparate retrieval systems (like BM25 and vector search) without normalizing raw scores.")

st.divider()

# ==============================================================================
# Section 7: Summary & Self-Assessment Checklist
# ==============================================================================
st.header("7. Lesson 16.3 Summary & Self-Assessment", divider="rainbow")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown(
        """
### Key Takeaways:
1. **Beyond Naive RAG**: Use query rewriting and multi-query expansion to eliminate vocabulary mismatch.
2. **HyDE Power**: Bridge the question-to-document length gap by embedding hypothetical answers.
3. **Hybrid Search**: Combine BM25 exact keyword matching with dense vector semantics via RRF.
4. **Two-Stage Retrieval**: Use bi-encoders for high-speed candidate retrieval and cross-encoders for high-precision reranking.
"""
    )

with col_s2:
    st.markdown("### Self-Assessment Checklist:")
    st.checkbox("I understand the mechanics and safety profile of HyDE retrieval.", value=True, key="chk_16_3_1")
    st.checkbox("I know how Bi-Encoders differ from Cross-Encoders in computational trade-offs.", value=True, key="chk_16_3_2")
    st.checkbox("I can implement Reciprocal Rank Fusion (RRF) for hybrid dense-sparse search.", value=True, key="chk_16_3_3")
    st.checkbox("I know how to design Multi-Query expansion prompts in Python.", value=True, key="chk_16_3_4")
