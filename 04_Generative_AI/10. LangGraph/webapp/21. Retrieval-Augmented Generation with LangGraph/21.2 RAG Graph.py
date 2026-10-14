import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="21.2 RAG Graph - LangGraph Curriculum",
    page_icon="🕸️",
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
    .badge-rag {
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
class RagGraphSimulator:
    """
    Simulates multi-stage RAG graph execution:
    - Query rewriting (de-referencing pronouns)
    - Retrieval simulation
    - Document relevance filtering based on similarity threshold
    - Context construction & generation
    """
    def __init__(self):
        self.mock_docs = [
            {"id": "doc_1", "text": "LangGraph Pregel engine processes nodes in atomic supersteps with barrier synchronization.", "score": 0.94},
            {"id": "doc_2", "text": "Checkpointers write state snapshots to PostgreSQL using composite keys.", "score": 0.88},
            {"id": "doc_3", "text": "Unrelated paragraph about weather forecasting and ocean currents in the Atlantic.", "score": 0.22},
            {"id": "doc_4", "text": "Subgraphs can be compiled with private state schemas or shared state schemas.", "score": 0.82}
        ]

    def execute_rag_pipeline(
        self,
        raw_query: str,
        similarity_threshold: float,
        apply_rewriter: bool
    ) -> Dict[str, Any]:
        # 1. Query Rewriting
        if apply_rewriter:
            rewritten_query = f"Detailed technical documentation on: {raw_query}"
        else:
            rewritten_query = raw_query

        # 2. Retrieval & Filtering
        filtered_docs = [d for d in self.mock_docs if d["score"] >= similarity_threshold]
        
        # 3. Context Construction
        context_str = "\n".join([f"[{d['id']}]: {d['text']}" for d in filtered_docs])
        
        # 4. Generation
        if filtered_docs:
            answer = f"Based on {len(filtered_docs)} verified sources: {filtered_docs[0]['text'][:80]}..."
        else:
            answer = "Insufficient relevant context found in database to answer confidently."

        return {
            "raw_query": raw_query,
            "rewritten_query": rewritten_query,
            "retrieved_count": len(self.mock_docs),
            "filtered_docs": filtered_docs,
            "context_str": context_str,
            "answer": answer
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Stateful RAG Graph Pipeline in LangGraph", divider="rainbow")

st.markdown("""
Naive RAG applications use simple linear scripts: `query -> search -> prompt -> answer`.
In real-world enterprise production, simple scripts fail when:
- The user's query contains ambiguous pronouns ("How do I configure it?").
- Retrieved documents are irrelevant or noisy, polluting the prompt.
- The retrieval returns zero matches, and the model attempts to guess rather than gracefully declining.

LangGraph transforms RAG from a brittle linear chain into a **Stateful, Controlled Decision Graph**:
1. **Query Analysis & Rewriting Node**: Expands, clarifies, and de-references conversational pronouns.
2. **Retrieval Node**: Fetches raw candidate documents from one or more knowledge indexes.
3. **Document Filtering (Relevance) Node**: Audits retrieved chunks and filters out noise.
4. **Context Construction Node**: Assembles token-budgeted prompt payloads with source citations.
5. **Generation Node**: Synthesizes the factual response.
6. **Self-Correction & Fallback Edges**: Conditional edges verify answer grounding and trigger retries or web search fallback if needed.

The blueprint below illustrates this multi-stage stateful RAG graph.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RagGraphTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_query_stage {
            label="1. Query Optimization & Rewriting";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            user_input [label="User Conversational Query\\n'How do I configure it?'", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            query_rewriter [label="Query Rewriter Node\\nDe-references 'it' -> 'PostgresSaver'", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        }

        subgraph cluster_retrieval_filter {
            label="2. Retrieval & Document Relevance Filtering";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            retriever_node [label="Retrieval Node\\nVector Index Top 5 Candidates", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
            doc_filter [label="Document Filter Node\\nGrades Chunks: Relevance >= 0.75", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
        }

        subgraph cluster_decision_gate {
            label="3. Relevance Gate & Context Synthesis";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            context_node [label="Context Construction Node\\nToken Budgeting & Citation Tagging", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];
            generator_node [label="LLM Generation Node\\nGenerates Grounded Response", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            fallback_node [label="Fallback / Clarify Node\\nRequests Clarification / External Search", fillcolor="#450A0A", fontcolor="#FECACA", color="#EF4444"];
        }

        user_input -> query_rewriter;
        query_rewriter -> retriever_node [label="Rewritten Query"];
        retriever_node -> doc_filter [label="Raw Passages"];
        
        doc_filter -> context_node [label="High Relevance Docs Found", color="#34D399"];
        doc_filter -> fallback_node [label="Zero Relevant Docs (Re-route)", color="#EF4444", style=dashed];
        
        context_node -> generator_node;
        generator_node -> user_input [label="Verified Answer Delivered", color="#38BDF8"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: QUERY ANALYSIS
# ==============================================================================
st.header("2. Query Analysis: Intent Classification, Entity Extraction, and Search Strategy Determination", divider="rainbow")

st.markdown("""
Before touching a database, the **Query Analysis Node** inspects the incoming user prompt to determine:
- **Query Type**: Is this a factual question, a broad thematic overview, or general conversational chit-chat?
- **Domain Entities**: Extracts targeted keywords, model names, dates, or product codes.
- **Routing Strategy**: Directs factual questions to vector retrieval, while routing conversational chit-chat directly to the LLM without performing expensive database queries!
""")

# ==============================================================================
# SECTION 3: QUERY REWRITING
# ==============================================================================
st.header("3. Query Rewriting: De-Contextualization, Expansion, and Hypothetical Document Embeddings (HyDE)", divider="rainbow")

st.markdown("""
In conversational chat, human users frequently use pronouns and shorthand:
- Turn 1: *"Tell me about LangGraph checkpointers."*
- Turn 2: *"How do I connect it to Postgres?"*

If the Turn 2 query `"How do I connect it to Postgres?"` is directly embedded into a vector store, retrieval fails because the word `"LangGraph"` or `"checkpointer"` is absent!
**Query Rewriting** takes the conversation history and de-contextualizes the query into a standalone search string:
`"How do I configure LangGraph checkpointers to use PostgreSQL?"`
""")

st.code('''# Query Rewriting Node using LLM
def query_rewriter_node(state: RagState) -> dict:
    history = state["messages"][:-1]
    latest = state["messages"][-1].content
    
    prompt = f"Given the conversation history, rewrite the latest question into a standalone search query: {latest}"
    rewritten = rewriter_llm.invoke(prompt).content
    return {"standalone_query": rewritten}''', language="python")

# ==============================================================================
# SECTION 4: RETRIEVAL
# ==============================================================================
st.header("4. Retrieval: Vector Store Querying, Multi-Query Parallel Execution, and Raw Chunk Fetching", divider="rainbow")

st.markdown("""
The **Retrieval Node** executes the search against the vector database using the optimized standalone query.
Advanced variations include:
- **Multi-Query Retrieval**: An LLM generates 3 alternative phrasings of the query; the node executes all 3 in parallel and merges candidate documents.
- **Hybrid Retrieval**: Combines semantic vector similarity search with BM25 keyword matching to catch exact keywords (e.g. error codes like `ERR_404_TIMEOUT`).
""")

# ==============================================================================
# SECTION 5: DOCUMENT FILTERING
# ==============================================================================
st.header("5. Document Filtering: Semantic Relevance Scoring, Thresholding, and Duplicate Elimination", divider="rainbow")

st.markdown("""
Vector retrieval always returns the top $k$ nearest chunks, even if none of them are actually relevant!
The **Document Filtering Node** grades each retrieved chunk:
- Chunks scoring below a similarity threshold (e.g. `score < 0.75`) are discarded.
- In high-assurance systems, an LLM evaluator grades binary relevance (`relevant` vs `irrelevant`).
- If all documents are rejected, conditional edges divert execution to a web search or human escalation node instead of allowing the model to hallucinate.
""")

# ==============================================================================
# SECTION 6: CONTEXT CONSTRUCTION
# ==============================================================================
st.header("6. Context Construction: Token Budgeting, Citation Formatting, and System Prompt Synthesis", divider="rainbow")

st.markdown("""
Once relevant documents are filtered, they must be formatted into the LLM prompt.
Effective context construction enforces:
1. **Explicit Source Identification**: Numbering documents `[Doc 1]`, `[Doc 2]` with metadata links.
2. **Token Budgeting**: Ensuring combined context does not exceed the model's optimal reasoning window.
3. **Negative Constraint Instructions**: `"If the provided documents do not contain the answer, reply that you do not know."`
""")

# ==============================================================================
# SECTION 7: LLM GENERATION
# ==============================================================================
st.header("7. LLM Generation: Grounded Output Generation, Citation Footnoting, and Confidence Calibration", divider="rainbow")

st.markdown("""
The **Generation Node** produces the final synthesized response.
It uses strict grounding prompts to incorporate bracketed citations:
`"LangGraph uses superstep barriers to coordinate parallel nodes [Doc 1], ensuring atomic state updates [Doc 3]."`
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Stage Stateful RAG Graph Pipeline", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern implementing a multi-stage RAG graph in LangGraph with query rewriting, document filtering, and conditional routing.
""")

st.code('''from typing import TypedDict, Annotated, List, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class StatefulRagState(TypedDict):
    question: str
    standalone_query: str
    documents: List[Document]
    filtered_documents: List[Document]
    answer: str
    fallback_needed: bool

# 1. Query Rewriter Node
def rewrite_query_node(state: StatefulRagState) -> dict:
    q = state["question"]
    # Rewrites pronouns and expands query
    standalone = f"LangGraph framework documentation: {q}"
    return {"standalone_query": standalone}

# 2. Retrieval Node
def retrieve_docs_node(state: StatefulRagState) -> dict:
    # Simulated vector store lookup
    mock_results = [
        Document(page_content="LangGraph supports state checkpointers backed by PostgreSQL.", metadata={"score": 0.92, "source": "docs/persistence"}),
        Document(page_content="Unrelated passage on CSS styling for web applications.", metadata={"score": 0.35, "source": "docs/css"})
    ]
    return {"documents": mock_results}

# 3. Document Filter Node
def filter_docs_node(state: StatefulRagState) -> dict:
    passed = [d for d in state["documents"] if d.metadata.get("score", 0) >= 0.75]
    return {
        "filtered_documents": passed,
        "fallback_needed": len(passed) == 0
    }

# 4. Generation Node
def generate_answer_node(state: StatefulRagState) -> dict:
    docs = state["filtered_documents"]
    context_text = " | ".join([d.page_content for d in docs])
    return {
        "answer": f"Grounded Answer based on {len(docs)} documents: {context_text}"
    }

# 5. Fallback Node
def fallback_node(state: StatefulRagState) -> dict:
    return {
        "answer": "I apologize, but no relevant documentation was found to answer your question accurately."
    }

def route_after_filter(state: StatefulRagState) -> str:
    if state["fallback_needed"]:
        return "fallback"
    return "generate"

# Build Graph
builder = StateGraph(StatefulRagState)
builder.add_node("rewrite", rewrite_query_node)
builder.add_node("retrieve", retrieve_docs_node)
builder.add_node("filter", filter_docs_node)
builder.add_node("generate", generate_answer_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "filter")
builder.add_conditional_edges("filter", route_after_filter, {"generate": "generate", "fallback": "fallback"})
builder.add_edge("generate", END)
builder.add_edge("fallback", END)

app = builder.compile(checkpointer=MemorySaver())
print("Stateful RAG Decision Graph Compiled Successfully.")''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Query Rewriting & Document Filtering Simulator", divider="rainbow")

st.markdown("""
Test how query rewriting expands ambiguous inputs and observe how relevance filtering drops noise before generation.
""")

sim = RagGraphSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("⚙️ RAG Graph Controls")
    user_query_input = st.text_input("User Prompt", value="How does it handle atomic barriers?")
    enable_rewriter = st.toggle("Enable Query Rewriting Node", value=True)
    min_score_threshold = st.slider("Document Relevance Cutoff Threshold", min_value=0.1, max_value=0.99, value=0.75, step=0.05)

    if st.button("🚀 Run RAG Graph Workflow"):
        res = sim.execute_rag_pipeline(
            raw_query=user_query_input,
            similarity_threshold=min_score_threshold,
            apply_rewriter=enable_rewriter
        )
        st.session_state["rag_graph_res"] = res

with col_sim2:
    st.subheader("📋 Pipeline State Audit")
    if "rag_graph_res" in st.session_state:
        r = st.session_state["rag_graph_res"]
        
        st.markdown(f"**Original Query**: `{r['raw_query']}`")
        st.markdown(f"**Rewritten Query**: `{r['rewritten_query']}`")
        st.markdown(f"**Retrieved Chunks**: {r['retrieved_count']} | **Filtered Chunks Retained**: {len(r['filtered_docs'])}")
        
        if r["filtered_docs"]:
            st.success(" Factual Grounding Context:")
            st.code(r["context_str"])
            st.markdown(f"**Final Response**: `{r['answer']}`")
        else:
            st.error("❌ Fallback Triggered: All chunks failed similarity threshold!")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: Why should an enterprise RAG system include a Query Rewriting node?", expanded=False):
    st.markdown("""
    **Answer**:
    In conversational chat, user prompts frequently contain unresolved pronouns, ellipses, and shorthand.
    Query rewriting de-contextualizes the query by combining it with conversation history to produce a complete, standalone search query that vector search can accurately index.
    """)

with st.expander("Q2: What is the purpose of the Document Filtering node in a RAG graph?", expanded=False):
    st.markdown("""
    **Answer**:
    Standard vector databases return top-k nearest neighbors unconditionally, even if all candidates are completely irrelevant.
    Document filtering grades retrieved chunks and discards low-relevance documents, preventing context poisoning and hallucinations.
    """)

with st.expander("Q3: How does LangGraph handle queries that return zero relevant documents?", expanded=False):
    st.markdown("""
    **Answer**:
    Using conditional edges originating from the document filtering node.
    If zero documents pass the relevance threshold, execution routes to a fallback node (requesting clarification, initiating external web search, or returning a graceful disclaimer).
    """)

with st.expander("Q4: What is HyDE (Hypothetical Document Embeddings)?", expanded=False):
    st.markdown("""
    **Answer**:
    HyDE is an advanced query rewriting technique where an LLM first generates a hypothetical answer to the user's question.
    That hypothetical passage is embedded and used to search the vector database, bridging the semantic gap between questions and answers.
    """)

with st.expander("Q5: Why is token budgeting critical during Context Construction?", expanded=False):
    st.markdown("""
    **Answer**:
    Overfilling the context window causes high latency, increased costs, and the 'lost-in-the-middle' phenomenon where the LLM overlooks vital facts placed in the center of a massive prompt.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Passing Raw Conversational Shorthand Directly to Vector Search")
    st.markdown("""
    *Mechanism*: Embedding `"What about it?"` without query rewriting.  
    *Consequence*: Retrieval returns completely unrelated chunks; the LLM hallucinates answers based on noise.  
    *Fix*: Always pass queries through a rewriting node in conversational threads.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): No Grounding Guardrails in the Generation Prompt")
    st.markdown("""
    *Mechanism*: Asking the LLM to answer without instructing it to rely strictly on provided documents.  
    *Consequence*: If documents are partially incomplete, the model fills gaps using its own pre-trained assumptions.  
    *Fix*: Enforce: *"Answer ONLY using the provided documents. If missing, say 'I do not know'."*
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Filtering All Documents with an Overly High Threshold")
    st.markdown("""
    *Mechanism*: Setting a similarity cutoff at 0.95 where valid documents naturally score 0.82.  
    *Consequence*: False negative rejections trigger fallbacks on valid user queries.  
    *Fix*: Calibrate relevance thresholds empirically across domain benchmark datasets.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Re-embedding Unmodified System Prompts Every Step")
    st.markdown("""
    *Mechanism*: Re-computing static prompt vector embeddings in real-time loops.  
    *Consequence*: Wastes API credits and adds 200ms per turn.  
    *Fix*: Pre-compute and cache static vector representations.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Query Rewriter**\\nA node de-referencing pronouns to create standalone search queries.")
with col_fc2:
    st.info("**Document Filter**\\nA quality gate rejecting low-relevance vector chunks before generation.")
with col_fc3:
    st.info("**Fallback Route**\\nA conditional branch executed when retrieval finds zero relevant evidence.")
with col_fc4:
    st.info("**Context Construction**\\nFormatting retrieved chunks into token-budgeted prompt payloads with citations.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know why stateful RAG graphs are superior to simple linear chains.", value=True)
c2 = st.checkbox("I understand the role and mechanics of Query Rewriting in conversational threads.", value=True)
c3 = st.checkbox("I know how Document Filtering eliminates context pollution and hallucinations.", value=True)
c4 = st.checkbox("I can implement conditional fallback routing for zero-match retrieval states.", value=True)
c5 = st.checkbox("I understand token budgeting and citation footnoting during Context Construction.", value=True)
c6 = st.checkbox("I know how to calibrate similarity score thresholds to balance precision and recall.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 21.2 'RAG Graph' with full architectural and practical comprehension!")
