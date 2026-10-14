import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="21.3 Advanced RAG - LangGraph Curriculum",
    page_icon="🚀",
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
    .badge-adv {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class AdvancedRagSimulator:
    """
    Simulates advanced RAG strategies:
    - CRAG (Corrective RAG): Document grading + web search fallback
    - Self-RAG: Reflection tokens [Retrieve], [IsSupported], [IsUseful]
    - Adaptive RAG: Routing by query difficulty
    - Hybrid Search: Dense + BM25 Reciprocal Rank Fusion
    """
    def __init__(self):
        pass

    def run_strategy_simulation(self, strategy: str, query: str) -> Dict[str, Any]:
        trace = []
        if strategy == "Corrective RAG (CRAG)":
            trace.append({"step": 1, "component": "Vector Retrieval", "status": "Low Confidence (Score 0.52 < Threshold 0.70)"})
            trace.append({"step": 2, "component": "CRAG Evaluator", "status": "Triggered Web Search Fallback"})
            trace.append({"step": 3, "component": "Tavily Web Search", "status": "Retrieved 3 real-time web sources"})
            trace.append({"step": 4, "component": "Knowledge Strip", "status": "Synthesized combined grounded context"})
            outcome = "Successfully recovered from empty internal index via web fallback."

        elif strategy == "Self-RAG":
            trace.append({"step": 1, "component": "LLM Initial Draft", "status": "Generated draft with [Retrieve] reflection tag"})
            trace.append({"step": 2, "component": "On-Demand Retriever", "status": "Fetched 2 focused citations"})
            trace.append({"step": 3, "component": "Critic Evaluator", "status": "Reflection [IsSupported]=YES | [IsUseful]=5/5"})
            outcome = "Grounded draft verified via reflection tags."

        elif strategy == "Adaptive RAG":
            trace.append({"step": 1, "component": "Classifier Router", "status": "Classified query as 'Multi-Hop Complex'"})
            trace.append({"step": 2, "component": "Strategy Selector", "status": "Dispatched to Agentic Multi-Hop Subgraph"})
            trace.append({"step": 3, "component": "Execution", "status": "Iterated 2 retrieval hops before synthesis"})
            outcome = "Routed query dynamically to optimal retrieval depth."

        else: # Hybrid Search with RRF
            trace.append({"step": 1, "component": "Dense Vector Search", "status": "Top 5 chunks via cosine embedding"})
            trace.append({"step": 2, "component": "Sparse BM25 Search", "status": "Top 5 chunks via keyword frequency"})
            trace.append({"step": 3, "component": "Reciprocal Rank Fusion", "status": "Calculated RRF score: 1/(60 + rank)"})
            trace.append({"step": 4, "component": "Cross-Encoder Reranker", "status": "Reranked top 3 passages for final prompt"})
            outcome = "Achieved 98% recall by fusing keyword and semantic indexes."

        return {
            "strategy": strategy,
            "query": query,
            "trace": trace,
            "outcome": outcome
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Advanced and Agentic RAG Paradigms in LangGraph", divider="rainbow")

st.markdown("""
Standard naive RAG assumes that vector search always returns high-quality, relevant documents on the first try.
In complex enterprise domains, this assumption fails frequently.

**Advanced & Agentic RAG** introduces dynamic loops, self-correction, and multi-tier retrieval strategies:
- **Corrective RAG (CRAG)**: Evaluates document quality and automatically engages web search fallbacks if internal retrieval is poor.
- **Adaptive RAG**: Dynamically routes queries between direct LLM answers, simple vector lookup, or multi-step agentic graphs based on query difficulty.
- **Self-RAG**: The model generates reflection tokens to assess its own groundedness and utility on-the-fly.
- **Hybrid Search & Reranking**: Combines dense vector similarity with sparse BM25 keyword matching, refined by Cross-Encoder rerankers.
- **Agentic RAG**: Equips autonomous agents with iterative retrieval tools, multi-hop reasoning, and stopping criteria.

The blueprint below illustrates the interconnected decision topology of Advanced and Agentic RAG.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph AdvancedRagTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_ingress_adaptive {
            label="1. Ingress & Adaptive Complexity Routing";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_query [label="User Query", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            adaptive_router [label="Adaptive RAG Router\\n(Simple FAQ vs Complex Multi-Hop)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_hybrid_retrieval {
            label="2. Hybrid Search & Cross-Encoder Reranking Tier";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            dense_vector [label="Dense Vector Search\\n(Chroma / Pinecone)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            sparse_bm25 [label="Sparse Keyword Search\\n(BM25 Okapi)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            rrf_fusion [label="Reciprocal Rank Fusion (RRF)\\nMerge & Deduplicate", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
            cross_reranker [label="Cross-Encoder Reranker\\n(Cohere / BGE-Reranker-v2)", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_crag_eval {
            label="3. Corrective RAG (CRAG) & Reflection Gate";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            doc_evaluator [label="Document Relevance Evaluator\\nCorrect / Ambiguous / Incorrect", fillcolor="#701A75", fontcolor="#F5D0FE", color="#E879F9"];
            web_search_fallback [label="Web Search Fallback\\n(Tavily / Google Search)", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
        }

        subgraph cluster_self_rag {
            label="4. Self-RAG Generation & Factuality Verification";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            llm_generator [label="LLM Synthesis Node\\nProduces Grounded Answer", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            hallucination_check [label="Self-RAG Grounding Critic\\nIs Output Supported by Context?", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            final_delivery [label="Verified Answer Delivery", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
        }

        user_query -> adaptive_router;
        adaptive_router -> dense_vector [label="Standard Route"];
        adaptive_router -> sparse_bm25 [label="Keyword Match"];
        
        dense_vector -> rrf_fusion;
        sparse_bm25 -> rrf_fusion;
        rrf_fusion -> cross_reranker;
        cross_reranker -> doc_evaluator;

        doc_evaluator -> llm_generator [label="Correct / High Quality", color="#34D399"];
        doc_evaluator -> web_search_fallback [label="Incorrect / Poor Retrieval", color="#EF4444", style=dashed];
        web_search_fallback -> llm_generator [label="Web Grounded Context"];

        llm_generator -> hallucination_check;
        hallucination_check -> final_delivery [label="Passed Grounding", color="#34D399"];
        hallucination_check -> llm_generator [label="Hallucinated (Regenerate)", color="#EF4444", style=dashed];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: CORRECTIVE RAG
# ==============================================================================
st.header("2. Corrective RAG (CRAG): Web Search Fallbacks and Document Grading", divider="rainbow")

st.markdown("""
**Corrective RAG (CRAG)** addresses the reality that internal vector databases often do not contain the answer.
Instead of forcing the LLM to generate an answer from low-relevance documents, CRAG uses an **Evaluator Node**:
- **Confidence High (`Correct`)**: Strip irrelevant sentences and proceed to synthesis.
- **Confidence Low (`Incorrect`)**: Discard internal documents and trigger external web search (e.g. Tavily API).
- **Ambiguous**: Combine internal documents with web search results for a hybrid context.
""")

# ==============================================================================
# SECTION 3: ADAPTIVE RAG
# ==============================================================================
st.header("3. Adaptive RAG: Dynamic Strategy Routing Based on Query Complexity", divider="rainbow")

st.markdown("""
Not all questions require a multi-stage retrieval pipeline.
**Adaptive RAG** analyzes query intent and routes to different graph topologies:
- **Direct Path**: Factual general knowledge ("What is the speed of light?") routes directly to the LLM with zero retrieval cost.
- **Single-Shot RAG**: Focused domain questions ("What is our Q3 return policy?") route to standard vector search.
- **Multi-Hop Agentic RAG**: Complex cross-document comparisons ("Compare company A and company B revenue growth over 3 years") route to an iterative reasoning subgraph.
""")

# ==============================================================================
# SECTION 4: SELF-RAG
# ==============================================================================
st.header("4. Self-RAG: Self-Reflection, Retrieval On-Demand, and Factual Critique Loops", divider="rainbow")

st.markdown("""
**Self-RAG** trains or prompts models to output special reflection tokens during generation:
1. `[Retrieve]`: The model decides *dynamically* whether external knowledge is needed for the current sentence.
2. `[IsSupported]`: Checks whether the generated sentence is mathematically or factually supported by retrieved documents.
3. `[IsUseful]`: Evaluates whether the generated response directly answers the user's intent.

If `[IsSupported]` fails, LangGraph's conditional edge catches the critique and triggers an immediate re-write.
""")

# ==============================================================================
# SECTION 5: MULTI-QUERY RETRIEVAL
# ==============================================================================
st.header("5. Multi-Query Retrieval: Query Expansion, Parallel Fan-Out, and De-Duplication", divider="rainbow")

st.markdown("""
Users often phrase questions with specific vocabulary that misses documents written with alternative synonyms.
**Multi-Query Retrieval** prompts an LLM to generate 3-5 perspectives of the same query:
- Query 1: *"LangGraph checkpoint database setup"*
- Query 2: *"Configuring PostgresSaver in LangGraph"*
- Query 3: *"Persistent state storage LangGraph PostgreSQL"*

The node executes all queries in parallel, takes the union of retrieved documents, and eliminates duplicate chunks based on document ID.
""")

# ==============================================================================
# SECTION 6: HYBRID SEARCH
# ==============================================================================
st.header("6. Hybrid Search: Fusing Dense Semantic Embeddings and Sparse BM25 via Reciprocal Rank Fusion", divider="rainbow")

st.markdown("""
Dense vector search excels at conceptual matches but struggles with exact strings (e.g., product SKUs, error codes like `HTTP_504`).
Sparse search (BM25) excels at exact keywords but lacks semantic understanding.
**Hybrid Search** queries both indexes and merges ranks using **Reciprocal Rank Fusion (RRF)**:
`RRF_Score(d) = sum( 1.0 / (60 + rank_i(d)) )`
""")

# ==============================================================================
# SECTION 7: RE-RANKING
# ==============================================================================
st.header("7. Re-Ranking: Cross-Encoder Scoring and Context Compression", divider="rainbow")

st.markdown("""
Bi-encoder embedding models compute query and document representations independently.
A **Cross-Encoder Reranker** passes the query and document together through full cross-attention layers, scoring deep semantic alignment with significantly higher precision.
In production: retrieve 25 candidates via fast vector search, pass to a Cross-Encoder (e.g. Cohere Rerank), and feed only the top 3 highest-scoring chunks to the LLM.
""")

# ==============================================================================
# SECTION 8: RETRIEVAL EVALUATION
# ==============================================================================
st.header("8. Retrieval Evaluation: Measuring Context Precision, Context Recall, and Faithfulness (RAGAS)", divider="rainbow")

st.markdown("""
To continuously improve advanced RAG, architectures measure four standardized **RAG Triad** metrics:
- **Context Precision**: What percentage of retrieved documents are truly relevant?
- **Context Recall**: Did the retrieval capture all information needed to answer the question?
- **Faithfulness**: Is the answer derived strictly from context with zero ungrounded claims?
- **Answer Relevance**: Does the final response directly address the user's question?
""")

# ==============================================================================
# SECTION 9: AGENTIC RAG
# ==============================================================================
st.header("9. Agentic RAG: Dynamic Tool Calling, Multi-Hop Reasoning, and Autonomous Stopping Criteria", divider="rainbow")

st.markdown("""
In **Agentic RAG**, retrieval is not a fixed hardcoded node in a pipeline.
Retrieval is provided as a **Tool** to an autonomous agent loop!
The agent can:
1. Search once, read findings, realize information is missing.
2. Formulate a second targeted query and search again (Multi-Hop).
3. Evaluate its own state and decide autonomously when it has enough evidence to stop and respond.
""")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Corrective & Self-Reflective Agentic RAG Graph", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a Corrective RAG (CRAG) graph with document grading and automated fallback routing.
""")

st.code('''from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class CragState(TypedDict):
    question: str
    documents: List[Document]
    web_fallback: bool
    answer: str

def retrieve_node(state: CragState) -> dict:
    # Simulated internal vector search
    mock_docs = [
        Document(page_content="LangGraph checkpointers support PostgreSQL and SQLite.", metadata={"grade": 0.88})
    ]
    return {"documents": mock_docs}

def grade_documents_node(state: CragState) -> dict:
    docs = state["documents"]
    # If top document score is below threshold, trigger web search
    avg_grade = sum(d.metadata.get("grade", 0) for d in docs) / max(len(docs), 1)
    needs_fallback = avg_grade < 0.70
    return {"web_fallback": needs_fallback}

def web_search_fallback_node(state: CragState) -> dict:
    web_doc = Document(page_content="Real-time web search results from external API.", metadata={"grade": 0.95})
    return {"documents": state["documents"] + [web_doc]}

def generate_node(state: CragState) -> dict:
    doc_summary = " | ".join([d.page_content for d in state["documents"]])
    return {"answer": f"Verified answer based on grounded sources: {doc_summary}"}

def route_crag(state: CragState) -> str:
    if state["web_fallback"]:
        return "web_search"
    return "generate"

builder = StateGraph(CragState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_docs", grade_documents_node)
builder.add_node("web_search", web_search_fallback_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_conditional_edges("grade_docs", route_crag, {"web_search": "web_search", "generate": "generate"})
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

app = builder.compile(checkpointer=MemorySaver())
print("Corrective RAG (CRAG) Decision Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Advanced RAG Strategy Benchmark & Reflection Simulator", divider="rainbow")

st.markdown("""
Test how CRAG, Self-RAG, Adaptive RAG, and Hybrid Search process complex domain queries.
""")

sim = AdvancedRagSimulator()

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.subheader("⚙️ Strategy Configuration")
    strategy_select = st.selectbox(
        "Choose Advanced RAG Strategy",
        ["Corrective RAG (CRAG)", "Self-RAG", "Adaptive RAG", "Hybrid Search (Dense + BM25)"]
    )
    test_query_input = st.text_input("User Prompt", value="Compare LangGraph memory with MemGPT")
    run_btn = st.button("🚀 Run Strategy Simulation")

with col_sim2:
    st.subheader(f"📋 Strategy Execution Audit")
    if run_btn:
        res = sim.run_strategy_simulation(strategy_select, test_query_input)
        st.success(f"Strategy: **{res['strategy']}**")
        st.info(f"Outcome: {res['outcome']}")
        st.markdown("#### Execution Step Trace:")
        st.dataframe(pd.DataFrame(res["trace"]), use_container_width=True)

st.subheader("📊 Advanced RAG Strategy Comparison Matrix")
df_adv = pd.DataFrame({
    "Strategy": ["Corrective RAG (CRAG)", "Self-RAG", "Adaptive RAG", "Hybrid Search", "Agentic RAG"],
    "Core Innovation": ["Document grading + web fallback", "Reflection tokens [Retrieve] / [IsSupported]", "Complexity classifier query routing", "Dense + Sparse BM25 + Rerank", "Autonomous retrieval tool calling loops"],
    "Latency Impact": ["Moderate (only falls back when needed)", "Moderate (single model reflections)", "Lowest for simple, high for complex", "Low (+10ms for RRF & BM25)", "Highest (multi-turn reasoning)"],
    "Primary Problem Solved": ["Out-of-domain knowledge gaps", "Hallucinated / unsupported claims", "One-size-fits-all over-engineering", "Missing exact keyword matches", "Multi-hop cross-document reasoning"]
})
st.dataframe(df_adv, hide_index=True, use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What is Corrective RAG (CRAG) and how does it prevent hallucinations?", expanded=False):
    st.markdown("""
    **Answer**:
    CRAG introduces a document grading evaluator between retrieval and generation.
    If retrieved documents fail relevance thresholds, CRAG automatically discards them and engages external web search, preventing the model from hallucinating over poor internal context.
    """)

with st.expander("Q2: How does Reciprocal Rank Fusion (RRF) combine dense and sparse search results?", expanded=False):
    st.markdown("""
    **Answer**:
    RRF assigns each document a score based on its reciprocal rank in each search list: `sum(1 / (k + rank))`, where $k$ is a constant (typically 60).
    It merges ranks without requiring score normalization between wildly different cosine similarity and BM25 score distributions.
    """)

with st.expander("Q3: What are the three core reflection tokens in Self-RAG?", expanded=False):
    st.markdown("""
    **Answer**:
    1. `[Retrieve]`: Decides whether retrieval is needed for the current statement.
    2. `[IsSupported]`: Assesses whether the statement is factually grounded in the retrieved context.
    3. `[IsUseful]`: Evaluates whether the generated text effectively answers the user's prompt.
    """)

with st.expander("Q4: Why is a Cross-Encoder Reranker placed after vector retrieval?", expanded=False):
    st.markdown("""
    **Answer**:
    Bi-encoder vector search is fast ($O(1)$) but computes representations independently, missing fine-grained token interactions.
    Cross-encoders process query and candidate passages through full joint attention layers, yielding vastly superior relevance ranking for the top 20 candidates.
    """)

with st.expander("Q5: What distinguishes Agentic RAG from traditional pipeline RAG?", expanded=False):
    st.markdown("""
    **Answer**:
    In pipeline RAG, retrieval happens once in a hardcoded sequence.
    In Agentic RAG, retrieval is a tool given to an autonomous agent loop that can issue multiple targeted searches, evaluate results, reformulate queries, and decide when it has collected sufficient evidence.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Normalizing Raw BM25 Scores with Cosine Scores")
    st.markdown("""
    *Mechanism*: Attempting to average raw BM25 scores (unbounded 0 to 50+) with cosine similarities (-1 to 1).  
    *Consequence*: BM25 completely overwhelms dense scores, rendering vector search useless.  
    *Fix*: Always use rank-based fusion (Reciprocal Rank Fusion) rather than raw score addition.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Unbounded Multi-Hop Agentic Search Loops")
    st.markdown("""
    *Mechanism*: An Agentic RAG loop that searches continuously whenever it perceives missing details.  
    *Consequence*: Hits 15+ search hops, running up bills and timing out users.  
    *Fix*: Enforce a hard cap of `max_hops <= 3` in graph state.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Running Reranking on Massive Candidate Sets")
    st.markdown("""
    *Mechanism*: Sending 500 retrieved chunks into a Cross-Encoder model.  
    *Consequence*: GPU inference latency spikes by 3+ seconds per query.  
    *Fix*: Limit reranker candidate pools to top 20-30 chunks.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Cost Warning): Unconditional Web Search Fallbacks on Internal Data")
    st.markdown("""
    *Mechanism*: Triggering public web search when a query asks about proprietary internal HR policies.  
    *Consequence*: Web search will never find private internal data, leaking queries and wasting latency.  
    *Fix*: Restrict web search fallbacks to public knowledge domains.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Corrective RAG**\\nEvaluates internal document relevance and triggers web search fallback on poor retrieval.")
with col_fc2:
    st.info("**Reciprocal Rank Fusion**\\nRank-based algorithm merging dense vector search and sparse BM25 scores.")
with col_fc3:
    st.info("**Self-RAG**\\nDynamic self-reflection using tokens like [Retrieve] and [IsSupported].")
with col_fc4:
    st.info("**Cross-Encoder Reranker**\\nJoint-attention model scoring deep semantic relevance for top retrieved candidates.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the mechanics of Corrective RAG (CRAG) and its fallback triggers.", value=True)
c2 = st.checkbox("I know how Adaptive RAG dynamically routes queries based on difficulty.", value=True)
c3 = st.checkbox("I understand the reflection token architecture of Self-RAG.", value=True)
c4 = st.checkbox("I can implement Hybrid Search fusing dense vectors and BM25 using RRF.", value=True)
c5 = st.checkbox("I understand the architectural role of Cross-Encoder rerankers in minimizing prompt noise.", value=True)
c6 = st.checkbox("I know how Agentic RAG empowers autonomous multi-hop information gathering.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 21.3 'Advanced RAG' with full architectural and practical comprehension!")
