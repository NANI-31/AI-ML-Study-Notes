import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.2 Advanced Retrieval - LangChain Curriculum",
    page_icon="🔍",
    layout="wide"
)

# Custom Theme Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .modern-callout {
        background-color: #0B192C;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
    }
    .retrieval-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-stage {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AdvancedRetrievalEngine:
    """
    Simulates advanced retrieval transformations: Query Rewriting, HyDE,
    Multi-Query decomposition, Cross-Encoder Reranking, and Parent-Child retrieval.
    """
    def __init__(self):
        self.sample_queries = {
            "Coreference Ambiguity": {
                "raw": "How do I configure its checkpointer?",
                "history": "User previously asked about LangGraph deployment.",
                "rewritten": "How do I configure the PostgresSaver checkpointer in LangGraph deployment?"
            },
            "Comparative Decomposition": {
                "raw": "Compare LangGraph MemorySaver and PostgresSaver latency and durability",
                "sub_queries": [
                    "What is the latency and durability profile of LangGraph MemorySaver?",
                    "What is the latency and durability profile of LangGraph PostgresSaver?"
                ]
            },
            "HyDE Hypothetical Expansion": {
                "raw": "What is Reciprocal Rank Fusion?",
                "hypothetical_doc": "Reciprocal Rank Fusion (RRF) is an information retrieval algorithm that combines multiple ranked lists from disparate retrieval systems (such as BM25 lexical and dense vector search) using the formula sum(1 / (k + rank)). It evaluates ordinal position without requiring score calibration."
            }
        }

    def simulate_parent_child(self, parent_text: str, chunk_size: int = 100) -> List[Dict[str, Any]]:
        words = parent_text.split()
        children = []
        for i in range(0, len(words), chunk_size):
            snippet = " ".join(words[i:i + chunk_size])
            children.append({
                "child_id": f"child_{i // chunk_size + 1}",
                "snippet": snippet[:60] + "...",
                "word_count": len(snippet.split()),
                "parent_ref": "parent_doc_main"
            })
        return children

# Instantiate engine
retrieval_engine = AdvancedRetrievalEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced Multi-Stage Retrieval and Context Optimization Pipeline", divider="rainbow")

st.markdown("""
Enterprise retrieval systems cannot rely on naive single-shot vector lookups. 
Real-world production retrieval employs a **multi-stage optimization pipeline** that systematically addresses 
conversational ambiguity, semantic vocabulary mismatch, document noise, and context loss.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Advanced_Retrieval_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_pre_retrieval {
            label = "1. Pre-Retrieval Transformation Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            raw_query [label="Incoming Ambiguous Query\\n('How does it scale in prod?')", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            query_rewriter [label="Query Rewriter\\n(Resolves conversational context & pronouns)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            hyde_generator [label="HyDE / Multi-Query Expander\\n(Generates hypothetical doc & 3 sub-queries)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_retrieval {
            label = "2. Multi-Candidate Hybrid Retrieval";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            sparse_retriever [label="BM25 Lexical Inverted Index\\n(Exact matches & technical tokens)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            dense_retriever [label="Dense Vector Store (HNSW)\\n(Conceptual semantic similarity)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            candidate_pool [label="Unified Candidate Pool (top-30)\\n(Merged via Reciprocal Rank Fusion)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
        }

        subgraph cluster_post_retrieval {
            label = "3. Post-Retrieval Re-ranking & Contextual Compression";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            cross_encoder [label="Cross-Encoder Reranker\\n(Joint attention over Query + Text pairs: top-5)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            parent_child_resolver [label="Parent-Child Document Resolver\\n(Maps small child matches to full parent sections)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            context_compressor [label="Contextual Compressor\\n(Drops irrelevant filler sentences; isolates facts)", fillcolor="#0F172A", fontcolor="#34D399", color="#10B981"];
            llm_generator [label="Frontier LLM Generation\\n(Generates answer with exact citations)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        raw_query -> query_rewriter;
        query_rewriter -> hyde_generator;
        hyde_generator -> sparse_retriever [label="parallel dispatch"];
        hyde_generator -> dense_retriever [label="parallel dispatch"];

        sparse_retriever -> candidate_pool;
        dense_retriever -> candidate_pool;

        candidate_pool -> cross_encoder;
        cross_encoder -> parent_child_resolver [label="top-5 candidates"];
        parent_child_resolver -> context_compressor;
        context_compressor -> llm_generator [label="high-density context"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Key Principle of Advanced Retrieval:</strong> Decouple <em>the text used for indexing and vector matching</em> 
    from <em>the text passed to the LLM for synthesis</em>. Small chunks enable high-precision vector search, 
    while hierarchical parent documents preserve the narrative context necessary for accurate synthesis.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: QUERY REWRITING
# ==============================================================================
st.header("2. Query Rewriting: Resolving Conversational Coreference and Ambiguity with LLMs", divider="rainbow")

st.markdown("""
In multi-turn chat applications, user questions are frequently elliptical or pronoun-heavy 
("What about its pricing?", "Can it do that asynchronously?"). 
If fed directly into a vector database, cosine similarity fails because the pronoun possesses zero semantic overlap 
with the target subject.
""")

st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Coreference Resolution Prompt
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, rephrase the question to be a standalone, fully-qualified search query. Do NOT answer the question."),
    ("human", "Chat History:\\n{chat_history}\\n\\nFollow Up Input: {question}")
])

# rephrase_chain = rephrase_prompt | chat_model | StrOutputParser()
''', language="python")

st.markdown("""
### Core Benefits
- **Coreference Resolution**: Converts "its memory" to "LangGraph checkpointer memory".
- **Acronym Expansion**: Resolves company-internal acronyms into searchable canonical terminology.
- **De-contextualization**: Produces self-contained search queries suitable for caching.
""")

# ==============================================================================
# SECTION 3: QUERY EXPANSION
# ==============================================================================
st.header("3. Query Expansion: Hypothetical Document Embeddings (HyDE) and Step-Back Prompting", divider="rainbow")

st.markdown("""
There is an inherent **asymmetry** between user questions and answer passages:
- **Questions**: Short, inquisitive, lacking detailed technical keywords ("Why is my pod crashing with OOM?").
- **Target Passages**: Long, affirmative, technical explanations ("Linux cgroup memory limits kill container processes when memory usage exceeds limits...").
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### HyDE (Hypothetical Document Embeddings)
    1. Ask an LLM to generate a hypothetical answer to the question (even if factual accuracy is unverified).
    2. Embed the *hypothetical document* rather than the short question.
    3. The vector embedding of an affirmative passage aligns much closer in semantic space to the actual document corpus!
    """)

with col2:
    st.markdown("""
    ### Step-Back Prompting
    1. Generate a higher-level, more abstract conceptual query ("What are the principles of container memory management?").
    2. Retrieve both specific facts and high-level background concepts.
    3. Ground the final response in both foundational principles and specific details.
    """)

# ==============================================================================
# SECTION 4: MULTI-QUERY RETRIEVAL
# ==============================================================================
st.header("4. Multi-Query Retrieval: Sub-Query Decomposition and Parallel Semantic Retrieval", divider="rainbow")

st.markdown("""
A user often asks complex multi-faceted questions that span multiple distinct documents:
> *"Compare the pricing, vector indexing algorithms, and latency of Pinecone vs Qdrant."*

A single vector query will retrieve documents that mention both keywords, but will miss deep documentation on either individual engine.
""")

st.markdown("""
### Multi-Query Pipeline Workflow
1. **Decomposition**: Prompt an LLM to decompose the compound query into 3-4 orthogonal sub-queries:
   - Sub-query 1: *"Pinecone pricing models and HNSW indexing latency"*
   - Sub-query 2: *"Qdrant vector indexing algorithms and performance benchmarks"*
2. **Parallel Retrieval**: Execute concurrent vector searches across all sub-queries.
3. **Deduplication & Union**: Gather and deduplicate candidate documents, preserving unique context across all facets.
""")

# ==============================================================================
# SECTION 5: HYBRID SEARCH
# ==============================================================================
st.header("5. Hybrid Search: Dense Vector K-NN Meets Sparse Lexical Inverted Indexes", divider="rainbow")

st.markdown("""
**Hybrid Search** pairs the semantic understanding of dense neural representations with the lexical precision of sparse inverted indexes:
""")

hybrid_df = pd.DataFrame([
    {"Search Dimension": "Sparse Search (BM25)", "Mechanism": "Token frequency & Inverse Document Frequency", "Strengths": "Acronyms, code identifiers, exact numbers, rare jargon", "Weaknesses": "Zero understanding of synonyms or conceptual paraphrasing"},
    {"Search Dimension": "Dense Search (HNSW)", "Mechanism": "High-dimensional neural vector dot products", "Strengths": "Captures synonyms, intent, and semantic meaning", "Weaknesses": "Keyword blind; easily misses exact model names or error codes"},
    {"Search Dimension": "Reciprocal Rank Fusion", "Mechanism": "score = sum(1 / (k + rank)) across both rank lists", "Strengths": "Combines strengths of both without fragile score normalization", "Weaknesses": "Requires running two separate index traversals"}
])
st.dataframe(hybrid_df, use_container_width=True)

# ==============================================================================
# SECTION 6: RERANKING
# ==============================================================================
st.header("6. Reranking: Cross-Encoder Scoring, Maximal Marginal Relevance, and Cohere Rerank", divider="rainbow")

st.markdown("""
Vector databases prioritize **throughput over precision**: their approximate nearest neighbor algorithms return 
candidates in milliseconds, but cosine similarity frequently ranks superficial keyword overlaps above truly informative answers. 
**Reranking** provides high-precision reordering:
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("""
    ### Cross-Encoder Scoring
    - Takes `(Query, Document)` pairs and computes bidirectional attention across all tokens simultaneously.
    - Captures deep semantic interactions, producing a calibrated relevance score between 0.0 and 1.0.
    - Used to filter the top-20 retrieved candidates down to the top-3 to top-5 most relevant chunks.
    """)

with col_r2:
    st.markdown("""
    ### Maximal Marginal Relevance (MMR)
    - Prevents retrieval redundancy by penalizing candidates that are overly similar to documents already selected.
    - Balances **relevance** to the user query with **diversity** across the selected document pool:
    - Optimization objective: `lambda * Sim(Query, Doc) - (1 - lambda) * MaxSim(Doc, SelectedDocs)`
    """)

# ==============================================================================
# SECTION 7: CONTEXTUAL COMPRESSION
# ==============================================================================
st.header("7. Contextual Compression: Extractive Summarization, Document Filtering, and Token Pruning", divider="rainbow")

st.markdown("""
Standard chunking forces developers into an unpleasant trade-off:
- **Small Chunks (100 tokens)**: High vector search accuracy, but missing surrounding context.
- **Large Chunks (1000 tokens)**: Rich context, but introduces prompt dilution and high token costs.

**Contextual Compression** solves this: documents are retrieved in full chunks, but before passing them to the generator LLM, 
a compression model extracts only the specific sentences that directly answer the query.
""")

st.code('''from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# In LangChain:
# compressor = LLMChainExtractor.from_llm(chat_model)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=vector_retriever
# )
''', language="python")

# ==============================================================================
# SECTION 8: PARENT-CHILD RETRIEVAL
# ==============================================================================
st.header("8. Parent-Child Retrieval: Small-to-Big Retrieval, Sentence Windowing, and Hierarchical Indexing", divider="rainbow")

st.markdown("""
The **Parent-Child (Small-to-Big)** retrieval pattern is one of the most effective production techniques in modern RAG:
""")

st.markdown("""
1. **Hierarchical Indexing**:
   - Documents are divided into large **Parent Chunks** (e.g., 1,500 tokens or full sections).
   - Each parent chunk is subdivided into multiple small **Child Chunks** (e.g., 100-200 tokens or individual sentences).
2. **Search Against Children**: Dense vector embeddings are created *only for the small child chunks*. Small chunks align tightly with specific query vectors.
3. **Return the Parent Context**: When a child chunk is matched, the retriever does *not* send the small chunk to the LLM. Instead, it looks up the `parent_id` metadata tag and retrieves the full parent section!
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Advanced Retrieval Engine with HyDE, Multi-Query, and Cross-Encoder Reranking", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**Advanced Retrieval pipeline featuring query rewriting, simulated HyDE expansion, and cross-encoder reranking**.
""")

st.code('''from typing import List, Dict, Any

class AdvancedRetrievalPipeline:
    def __init__(self, doc_store: List[Dict[str, str]]):
        self.doc_store = doc_store

    def rewrite_query(self, query: str, history: str) -> str:
        """Simulates conversational coreference resolution."""
        if "its" in query.lower() or "it" in query.lower():
            return f"PostgresSaver checkpointer in LangGraph production deployment"
        return query

    def generate_hyde_doc(self, query: str) -> str:
        """Simulates hypothetical document generation."""
        return f"Hypothetical documentation explaining {query} in detail with configuration syntax."

    def retrieve_candidates(self, search_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simulates candidate retrieval with text match scoring."""
        scored = []
        tokens = set(search_text.lower().split())
        for doc in self.doc_store:
            doc_tokens = set(doc["content"].lower().split())
            score = len(tokens.intersection(doc_tokens)) / max(len(tokens), 1)
            scored.append({"id": doc["id"], "content": doc["content"], "match_score": round(score, 3)})
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        return scored[:top_k]

    def cross_encoder_rerank(self, original_query: str, candidates: List[Dict[str, Any]], top_n: int = 2) -> List[Dict[str, Any]]:
        """Simulates cross-encoder precision re-scoring."""
        for c in candidates:
            # High-precision cross-attention simulation
            precision_score = c["match_score"] * 1.5
            if "postgres" in original_query.lower() and "postgres" in c["content"].lower():
                precision_score += 0.4
            c["reranked_score"] = round(precision_score, 3)
        candidates.sort(key=lambda x: x["reranked_score"], reverse=True)
        return candidates[:top_n]

# Execution Demonstration
documents = [
    {"id": "doc_1", "content": "LangGraph MemorySaver persists thread state only in volatile process RAM."},
    {"id": "doc_2", "content": "LangGraph PostgresSaver provides durable, production-ready thread checkpointing across server restarts."},
    {"id": "doc_3", "content": "FastAPI ASGI middleware handling streaming SSE endpoints."}
]

pipeline = AdvancedRetrievalPipeline(documents)

# Turn 1: User sends ambiguous question
raw_user_query = "How do I configure its checkpointer?"
resolved_query = pipeline.rewrite_query(raw_user_query, history="Discussing LangGraph deployment")
print("Resolved Query:", resolved_query)

# Candidate generation via HyDE
hyde_text = pipeline.generate_hyde_doc(resolved_query)
candidates = pipeline.retrieve_candidates(hyde_text, top_k=3)

# Cross-Encoder Reranking
top_results = pipeline.cross_encoder_rerank(resolved_query, candidates, top_n=1)
print("Top Reranked Context:", top_results)
''', language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Query Transformation Playground & Multi-Stage Retrieval Simulator", divider="rainbow")

st.markdown("""
Test query rewriting, HyDE generation, and Parent-Child chunk mapping 
interactively.
""")

wb_view = st.selectbox(
    "Select Transformation Tool:",
    ["1. Conversational Query Rewriting Inspector", "2. HyDE Asymmetry Resolver", "3. Parent-Child Chunking Simulator"]
)

if wb_view == "1. Conversational Query Rewriting Inspector":
    st.subheader("Coreference Resolution Playground")
    sample_type = st.selectbox("Select Sample Scenario:", list(retrieval_engine.sample_queries.keys()))
    sample_data = retrieval_engine.sample_queries[sample_type]

    st.markdown(f"**Raw User Query:** `{sample_data['raw']}`")
    if "history" in sample_data:
        st.markdown(f"**Conversation Context:** `{sample_data['history']}`")
        st.success(f"**Rewritten Standalone Query:** `{sample_data['rewritten']}`")
    elif "sub_queries" in sample_data:
        st.markdown("**Decomposed Multi-Queries:**")
        for sq in sample_data["sub_queries"]:
            st.info(f"Sub-query: `{sq}`")
    else:
        st.markdown(f"**Hypothetical Passage Generated:** `{sample_data['hypothetical_doc']}`")

elif wb_view == "2. HyDE Asymmetry Resolver":
    st.subheader("Hypothetical Document Embeddings (HyDE) Simulator")
    user_q = st.text_input("Enter Question:", "How does Cross-Encoder Reranking improve RAG?")
    
    if st.button("Generate Simulated HyDE Document"):
        sim_doc = (
            f"Cross-Encoder reranking enhances retrieval-augmented generation by performing bidirectional "
            f"cross-attention between the query '{user_q}' and candidate document passages. Unlike bi-encoders "
            f"that score cosine similarity between isolated vectors, cross-encoders capture deep token-level "
            f"interactions, dramatically reducing false positives and improving precision."
        )
        st.markdown("#### Generated Hypothetical Passage:")
        st.code(sim_doc, language="markdown")
        st.success("Embedding this affirmative passage achieves significantly higher vector proximity to target technical docs!")

else:
    st.subheader("Parent-Child Chunk Mapping Inspector")
    parent_passage = st.text_area(
        "Parent Document Context (1,000 words):",
        "LangGraph v0.3 introduces complete architectural support for durable thread checkpointing. "
        "When running production agents in Kubernetes clusters, nodes can be terminated or restarted unexpectedly. "
        "By binding PostgresSaver to the workflow checkpointer, all node execution states are saved as atomic row "
        "commits in PostgreSQL. If an interrupt occurs, execution halts gracefully and can resume upon human authorization."
    )
    chunk_sz = st.slider("Child Chunk Word Window Size:", 10, 30, 15)

    child_splits = retrieval_engine.simulate_parent_child(parent_passage, chunk_size=chunk_sz)
    st.markdown(f"**Generated {len(child_splits)} Child Chunks (Indexed in Vector Store):**")
    st.dataframe(pd.DataFrame(child_splits), use_container_width=True)
    st.caption("Child vectors are searched in vector DB. When matched, the full parent passage is passed to the generation LLM.")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why does HyDE (Hypothetical Document Embeddings) improve retrieval performance?"):
    st.markdown("""
    **Answer:**  
    Vector embedding spaces reflect document genre and tone. Short questions ("What is RRF?") reside in a different vector subspace 
    than explanatory paragraphs. HyDE uses an LLM to generate a hypothetical answer passage. Even if factually imperfect, 
    the hypothetical passage matches the syntax, vocabulary, and semantic structure of actual documentation, resulting in 
    significantly higher cosine similarity to the ground-truth corpus.
    """)

with st.expander("Q2: How does Parent-Child (Small-to-Big) retrieval resolve the chunk size dilemma?"):
    st.markdown("""
    **Answer:**  
    Small chunks (100 tokens) produce tight, focused vector representations that match specific search queries with high accuracy, 
    but lack broader context. Large chunks (1,000 tokens) provide rich context for the LLM to synthesize answers, but average out 
    vector details. Parent-Child retrieval decouples indexing from synthesis: it indexes small child chunks for search accuracy, 
    and returns large parent chunks to the LLM for comprehensive synthesis.
    """)

with st.expander("Q3: When is Multi-Query decomposition strictly required in an enterprise RAG pipeline?"):
    st.markdown("""
    **Answer:**  
    Multi-query decomposition is required whenever user queries contain compound conjunctions ("Compare X and Y"), 
    temporal comparisons ("How did sales in 2023 differ from 2024?"), or multi-entity dependencies. A single vector lookup 
    cannot span distinct document topics simultaneously. Decomposing the request into orthogonal sub-queries ensures complete recall.
    """)

with st.expander("Q4: What is the primary operational trade-off of deploying Cross-Encoder rerankers?"):
    st.markdown("""
    **Answer:**  
    The primary trade-off is **inference latency**. While Bi-Encoder vector search executes in sub-10ms via HNSW indexing, 
    scoring 50 candidates through a Cross-Encoder requires 50 forward passes through a transformer, adding 100ms to 300ms of latency. 
    In production, systems mitigate this by retrieving only 20 candidates from vector search and using optimized reranker models 
    like `bge-reranker-large` or Cohere Rerank.
    """)

with st.expander("Q5: How does Maximal Marginal Relevance (MMR) prevent context window redundancy?"):
    st.markdown("""
    **Answer:**  
    Standard top-K retrieval frequently returns 4 near-duplicate chunks from the same document section. MMR uses a diversity parameter 
    (lambda) to penalize candidates that have high cosine similarity to chunks already selected in the result set, ensuring that 
    the final context includes diverse perspectives and information sources.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Executing HyDE Generation on Simple Entity or Lookup Queries**  
Using HyDE for factual identifier lookups ("What is customer 409's billing address?") wastes 1,000ms of latency and may cause 
the LLM to hallucinate realistic but fake addresses. Use HyDE only for conceptual or explanatory queries.
""")

st.error("""
**Trap 2: Feeding All Top-50 Vector Results Directly to a Cross-Encoder**  
Reranking 50-100 candidates on CPU infrastructure introduces severe latency bottlenecks (exceeding 1.5 seconds). 
Cap candidate pools to 15-20 documents before reranking.
""")

st.warning("""
**Trap 3: Using Query Rewriting Without History Truncation**  
Passing 50 turns of conversation history to the query rewriting prompt causes prompt latency inflation. 
Only pass the last 3-4 dialogue turns to the rewriter.
""")

st.warning("""
**Trap 4: Discarding Child Chunk Metadata in Parent-Child Retrieval**  
If the parent document is massive (e.g., 5,000 words), returning the entire parent document overflows the context window. 
Store mid-sized parents (500-1,000 words) or highlight child offsets within the parent.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>HyDE</h4>
        <p>Hypothetical Document Embeddings: Prompting an LLM to generate a hypothetical answer passage before embedding it to bridge question-document asymmetry.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Parent-Child Retrieval</h4>
        <p>An indexing strategy that indexes small child chunks for vector search precision and retrieves large parent chunks for generation context.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Maximal Marginal Relevance (MMR)</h4>
        <p>A reranking algorithm balancing query relevance with document diversity to prevent redundant chunks in the prompt.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Query Rewriting</h4>
        <p>Using an LLM to rewrite ambiguous, pronoun-laden user inputs into fully-qualified standalone search queries.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand why conversational coreference resolution is required before vector retrieval.", value=True)
chk_2 = st.checkbox("I know how HyDE bridges the semantic asymmetry between questions and answer passages.", value=True)
chk_3 = st.checkbox("I can implement Multi-Query decomposition for complex multi-part comparative prompts.", value=True)
chk_4 = st.checkbox("I understand the performance and precision trade-offs of Cross-Encoder reranking.", value=True)
chk_5 = st.checkbox("I know how Contextual Compression prunes prompt tokens and reduces generation costs.", value=True)
chk_6 = st.checkbox("I understand the architecture and indexing mechanics of Parent-Child (Small-to-Big) retrieval.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Advanced Retrieval competencies verified! Ready for Agentic RAG.")
