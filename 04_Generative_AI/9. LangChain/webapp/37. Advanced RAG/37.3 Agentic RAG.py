import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="37.3 Agentic RAG - LangChain Curriculum",
    page_icon="🤖",
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
    .agent-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-grade {
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
class AgenticRAGEngine:
    """
    Simulates autonomous Agentic RAG decision-making, including:
    - Deciding whether to retrieve
    - Selecting target retriever (SQL, VectorDB, Web Search)
    - Document relevance grading
    - Self-correction / query reformulation loops
    """
    def __init__(self):
        self.retrievers = {
            "Internal Vector DB": "Dense vector database containing proprietary HR, security, and architectural documentation.",
            "Production SQL Store": "Transactional relational database holding customer accounts, inventory balances, and order statuses.",
            "Live Web Search": "External search engine providing real-time financial news, release dates, and general facts."
        }

    def evaluate_retrieval_need(self, query: str) -> Dict[str, Any]:
        """Determines if query requires retrieval or can be answered parametrically."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["hello", "hi", "how are you", "write a poem", "python function"]):
            return {
                "needs_retrieval": False,
                "reason": "General conversational or coding task satisfiable by parametric LLM knowledge.",
                "route": "DIRECT_SYNTHESIS"
            }
        elif any(w in q_lower for w in ["order", "balance", "customer", "invoice", "price"]):
            return {
                "needs_retrieval": True,
                "reason": "Transactional structured data request.",
                "route": "Production SQL Store"
            }
        elif any(w in q_lower for w in ["today", "yesterday", "stock price", "latest news", "2026"]):
            return {
                "needs_retrieval": True,
                "reason": "Real-time or temporal external facts.",
                "route": "Live Web Search"
            }
        else:
            return {
                "needs_retrieval": True,
                "reason": "Proprietary organizational knowledge or technical documentation.",
                "route": "Internal Vector DB"
            }

    def simulate_document_grade(self, query: str, doc_text: str) -> Dict[str, Any]:
        """Simulates LLM-as-a-judge document relevance grading."""
        q_tokens = set(query.lower().split())
        doc_tokens = set(doc_text.lower().split())
        overlap = len(q_tokens.intersection(doc_tokens))
        
        if overlap >= 2:
            return {"grade": "RELEVANT", "score": 0.92, "action": "PROCEED_TO_GENERATION"}
        else:
            return {"grade": "IRRELEVANT", "score": 0.28, "action": "TRIGGER_REWRITE_AND_RETRY"}

# Instantiate engine
agentic_engine = AgenticRAGEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Agentic RAG Cyclic State Machine, Adaptive Routing, and Self-Reflection", divider="rainbow")

st.markdown("""
Unlike traditional RAG pipelines that execute a fixed sequence of retrieve-then-generate steps, 
**Agentic RAG** is governed by an **autonomous cyclic state machine** (typically constructed in LangGraph). 
The agent dynamically evaluates whether retrieval is warranted, selects the optimal specialized data store, 
grades the retrieved evidence, and loops back to self-correct if initial results are deemed insufficient.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Agentic_RAG_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_intent {
            label = "1. Intent & Routing Evaluation Node";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            user_query [label="Incoming User Query\\n('What is client 502's current subscription?')", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            router_node [label="Query Router / Planner Node\\n(Evaluates parametric knowledge vs retrieval need)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        subgraph cluster_retrievers {
            label = "2. Adaptive Dynamic Retrievers";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            vector_store [label="Vector Store (Unstructured Docs)\\n(Policy manuals, guides, RFCs)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            sql_store [label="SQL / Tabular Database\\n(Structured customer tables & orders)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            web_search [label="Live Web Search Tool\\n(Tavily / Serper for external news)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_reflection {
            label = "3. Reflection & Quality Grading Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            grader_node [label="Document Relevance Grader\\n(LLM evaluates if chunks answer query)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
            rewrite_node [label="Query Reformulation Node\\n(Expands keywords & retries alternate store)", fillcolor="#0F172A", fontcolor="#F43F5E", color="#F43F5E"];
            synthesizer [label="Grounded Synthesis Node\\n(Generates citation-backed answer)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            hallucination_check [label="Hallucination / Faithfulness Check\\n(Verifies answer matches source chunks)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
        }

        user_query -> router_node;
        router_node -> synthesizer [label="no retrieval needed"];
        router_node -> vector_store [label="route = docs"];
        router_node -> sql_store [label="route = sql"];
        router_node -> web_search [label="route = web"];

        vector_store -> grader_node;
        sql_store -> grader_node;
        web_search -> grader_node;

        grader_node -> synthesizer [label="grade = RELEVANT"];
        grader_node -> rewrite_node [label="grade = IRRELEVANT (retry loop)"];
        rewrite_node -> vector_store [label="re-dispatched search"];

        synthesizer -> hallucination_check;
        hallucination_check -> rewrite_node [label="hallucinated -> retry"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Core Paradigm Shift:</strong> In Agentic RAG, retrieval is treated as a <strong>tool invocation</strong> 
    inside an autonomous decision loop rather than an unbendable prerequisite step. If the retrieved documents do not 
    contain the requisite facts, the agent does not hallucinate an answer; it rejects the documents, reformulates the search, 
    and iterates.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: AGENT DECIDES WHEN TO RETRIEVE
# ==============================================================================
st.header("2. Agent Decides When to Retrieve: Parametric Knowledge Evaluation and Confidence Thresholding", divider="rainbow")

st.markdown("""
A major cost and latency inefficiency in naive RAG is **unnecessary retrieval**. 
When a user asks:
- *"Write a Python function to reverse a linked list."*
- *"Good morning! What tools do you have access to?"*

Traditional RAG runs a vector search anyway, injecting irrelevant programming manuals into the prompt. 
In Agentic RAG, an **Intent Gatekeeper** decides whether external retrieval is justified:
""")

st.code('''from typing import Literal
from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    action: Literal["direct_response", "retrieve_vector", "query_sql", "search_web"]
    reasoning: str = Field(description="Step-by-step rationale for the decision")

# The router evaluates parametric capability vs grounded data requirement
router_prompt = "Analyze the query and decide whether retrieval is required or if it can be answered directly."
''', language="python")

# ==============================================================================
# SECTION 3: AGENT CHOOSES RETRIEVER
# ==============================================================================
st.header("3. Agent Chooses Retriever: Dynamic Routing Across Specialized Domain Stores", divider="rainbow")

st.markdown("""
Enterprise knowledge is heterogeneous:
1. **Unstructured Documents**: PDF policies, markdown guides, technical runbooks -> Vector Database.
2. **Structured Records**: Transaction histories, real-time inventory balances, employee rosters -> Relational SQL Database.
3. **Temporal Real-Time Info**: Live exchange rates, current weather, vendor press releases -> Search API (Tavily/Google).

The agent maintains **tool definitions** for each specialized retriever and dynamically selects the correct endpoint:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ### Unstructured Vector Routing
    - Triggered by conceptual queries: *"What is our parental leave policy?"*
    - Executes dense semantic search or hybrid BM25 + dense fusion over document chunks.
    """)

with col2:
    st.markdown("""
    ### Structured SQL Routing
    - Triggered by relational queries: *"How many orders did user 9901 place in September?"*
    - Translates user intent into validated SQL queries executed directly against PostgreSQL.
    """)

# ==============================================================================
# SECTION 4: ITERATIVE RETRIEVAL
# ==============================================================================
st.header("4. Iterative Retrieval: Multi-Hop Reasoning Loops and Information Gathering Trajectories", divider="rainbow")

st.markdown("""
Complex real-world inquiries cannot be satisfied in a single retrieval step. 
Consider the multi-hop question:
> *"Did the company that developed React acquire any open-source database startups in 2024?"*

To answer this:
1. **Hop 1**: Identify the company that developed React (`Meta / Facebook`).
2. **Hop 2**: Retrieve acquisitions made by Meta in 2024 focusing on databases.
3. **Hop 3**: Synthesize the consolidated answer with cross-verified citations.

In **Iterative RAG**, the agent maintains an evolving memory state, executing sequential searches where the results of 
step N formulate the specific query for step N+1.
""")

# ==============================================================================
# SECTION 5: SEARCH -> ANALYZE -> SEARCH
# ==============================================================================
st.header("5. Search -> Analyze -> Search: Interleaved Reasoning, Gap Detection, and Follow-Up Query Generation", divider="rainbow")

st.markdown("""
The **Search -> Analyze -> Search** cycle is the foundational cognitive loop of Agentic RAG:
""")

st.markdown("""
1. **Search**: The agent queries the selected information source.
2. **Analyze**: The agent evaluates the retrieved context against the original goal:
   - *What questions have been answered?*
   - *What information gaps remain unresolved?*
3. **Follow-Up Search**: If crucial evidence is missing, the agent constructs a targeted follow-up query to fill the specific gap.
4. **Termination Condition**: Once all necessary facts are assembled (or the maximum loop limit is reached), the agent transitions to final synthesis.
""")

# ==============================================================================
# SECTION 6: RETRIEVAL VALIDATION
# ==============================================================================
st.header("6. Retrieval Validation: Document Grading, Hallucination Checks, and Answer Groundedness", divider="rainbow")

st.markdown("""
A primary cause of LLM hallucination is **context poisoning**: providing the model with retrieved passages that are tangentially 
related but do not actually contain the answer. 

**Retrieval Validation** introduces two automated grading checkpoints:
""")

val_df = pd.DataFrame([
    {"Validation Checkpoint": "Document Grading (Pre-Synthesis)", "Evaluator": "Lightweight LLM Grader (SLM)", "Evaluation Question": "Do the retrieved documents contain facts relevant to answering the prompt?", "Action on Failure": "Discard chunks, reformulate query, and retry search"},
    {"Validation Checkpoint": "Answer Groundedness (Post-Synthesis)", "Evaluator": "Cross-Encoder / LLM Judge", "Evaluation Question": "Is every claim in the generated answer supported by facts in the retrieved context?", "Action on Failure": "Trigger regeneration with stricter groundedness constraints"},
    {"Validation Checkpoint": "Answer Relevance (Post-Synthesis)", "Evaluator": "Embedding Similarity / Judge", "Evaluation Question": "Does the response directly address the user's initial question?", "Action on Failure": "Route back to planner to address missing requirements"}
])
st.dataframe(val_df, use_container_width=True)

# ==============================================================================
# SECTION 7: SELF-CORRECTING RAG
# ==============================================================================
st.header("7. Self-Correcting RAG: Fallback Search, Query Reformulation, and Automated Recovery", divider="rainbow")

st.markdown("""
**Corrective RAG (CRAG)** and **Self-RAG** formalize self-correction through explicit graph states:
- **Correct State**: If the retrieved documents receive a high confidence grade, the agent proceeds immediately to synthesis.
- **Ambiguous State**: If the internal vector store yields marginal relevance, the agent triggers a **fallback search** (e.g., executing an external Web Search to corroborate).
- **Incorrect State**: If the internal documents are completely irrelevant, the agent strips the documents entirely, reformulates the query with extracted keywords, and queries an alternate retriever.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Self-RAG / Corrective-RAG (CRAG) Cyclic State Machine in LangGraph", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script implementing a complete 
**Self-Correcting RAG (CRAG) state machine with dynamic routing, document grading, and query rewriting**.
""")

st.code('''from typing import TypedDict, List, Dict, Any, Literal

# 1. State Definition
class AgenticRAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retry_count: int
    route: str
    doc_grade: str

# 2. Production CRAG Controller Simulator
class ProductionCRAGService:
    def __init__(self):
        self.mock_vector_db = {
            "langgraph checkpointer": "PostgresSaver provides durable thread state checkpointing for production LangGraph apps.",
            "kubernetes": "Kubernetes pods run containerized microservices managed by deployment controllers."
        }

    def route_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Decides retrieval necessity and target store."""
        q = state["question"].lower()
        if "hello" in q or "hi" in q:
            state["route"] = "direct"
        else:
            state["route"] = "vector_db"
        return state

    def retrieve(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Retrieves candidates from store."""
        q = state["question"].lower()
        matches = []
        for k, v in self.mock_vector_db.items():
            if any(term in q for term in k.split()):
                matches.append(v)
        state["documents"] = matches if matches else ["No relevant vector documents located."]
        return state

    def grade_documents(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Grades retrieved passages for relevance."""
        docs = " ".join(state["documents"]).lower()
        q = state["question"].lower()
        
        # Check keyword presence
        if any(term in docs for term in ["postgressaver", "checkpointing", "kubernetes"]):
            state["doc_grade"] = "RELEVANT"
        else:
            state["doc_grade"] = "IRRELEVANT"
        return state

    def rewrite_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Reformulates query for second retrieval attempt."""
        state["retry_count"] += 1
        # Reformulate query to target specific terminology
        state["question"] = f"{state['question']} checkpointer persistence"
        return state

    def generate(self, state: AgenticRAGState) -> AgenticRAGState:
        """Node: Synthesizes final response."""
        context = " ".join(state["documents"])
        state["generation"] = f"Answer derived from context: '{context}'"
        return state

# Execution Demonstration
crag = ProductionCRAGService()
initial_state: AgenticRAGState = {
    "question": "How does LangGraph save state?",
    "documents": [],
    "generation": "",
    "retry_count": 0,
    "route": "",
    "doc_grade": ""
}

# Run Step 1: Routing
s1 = crag.route_query(initial_state)
# Run Step 2: Retrieval
s2 = crag.retrieve(s1)
# Run Step 3: Grading
s3 = crag.grade_documents(s2)

if s3["doc_grade"] == "RELEVANT":
    final_state = crag.generate(s3)
    print("Generation Success:", final_state["generation"])
else:
    s_retry = crag.rewrite_query(s3)
    s_retry_ret = crag.retrieve(s_retry)
    final_state = crag.generate(s_retry_ret)
    print("Generation After Retry:", final_state["generation"])
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Adaptive Retrieval Router & Self-Correction Evaluation Simulator", divider="rainbow")

st.markdown("""
Test dynamic query routing, evaluate whether questions require retrieval, 
and observe document grading with self-correction in real time.
""")

wb_view = st.selectbox(
    "Select Agentic Workbench Mode:",
    ["1. Adaptive Intent Gatekeeper & Route Classifier", "2. Document Relevance Grader Simulator", "3. Self-RAG Recovery Loop Simulator"]
)

if wb_view == "1. Adaptive Intent Gatekeeper & Route Classifier":
    st.subheader("Intent Gatekeeper & Route Selector")
    test_query = st.text_input(
        "Enter User Prompt to Route:",
        "What is the current stock price of Nvidia today?"
    )

    if st.button("Evaluate Routing Intent"):
        decision = agentic_engine.evaluate_retrieval_need(test_query)
        st.markdown(f"**Needs External Retrieval:** `{decision['needs_retrieval']}`")
        st.markdown(f"**Target Destination / Route:** `{decision['route']}`")
        st.info(f"**Planner Rationale:** {decision['reason']}")

elif wb_view == "2. Document Relevance Grader Simulator":
    st.subheader("Document Quality Grader (Pre-Synthesis)")
    sim_q = st.text_input("User Question:", "How do I configure PostgresSaver in LangGraph?")
    sim_doc = st.text_area(
        "Retrieved Candidate Chunk:",
        "LangGraph PostgresSaver provides durable checkpointer persistence across server reboots."
    )

    if st.button("Grade Candidate Document"):
        grade_info = agentic_engine.simulate_document_grade(sim_q, sim_doc)
        st.markdown(f"**Grade:** `{grade_info['grade']}` | **Confidence Score:** `{grade_info['score']}`")
        if grade_info["grade"] == "RELEVANT":
            st.success(f"Action: `{grade_info['action']}` - Proceed directly to LLM generation.")
        else:
            st.error(f"Action: `{grade_info['action']}` - Chunks rejected. Triggering query rewrite.")

else:
    st.subheader("Self-RAG Corrective Loop Execution Trace")
    st.markdown("Trace of an autonomous recovery cycle when initial vector search yields poor relevance:")
    st.info("""
    - **Step 1 (Plan)**: Query `How do I enable persistence?` routed to Internal Vector DB.
    - **Step 2 (Retrieve)**: Vector search retrieved chunk about in-memory arrays.
    - **Step 3 (Grade)**: Evaluator scored document relevance at `0.18 (IRRELEVANT)`. Action: `REJECT`.
    - **Step 4 (Rewrite)**: Query reformulation node expanded query to `LangGraph checkpointer persistence PostgresSaver`.
    - **Step 5 (Re-retrieve)**: Secondary retrieval found exact PostgresSaver technical documentation.
    - **Step 6 (Re-grade)**: Evaluator scored relevance at `0.94 (RELEVANT)`. Action: `PROCEED`.
    - **Step 7 (Synthesize)**: LLM generated factual answer with zero hallucination.
    """)
    st.success("Self-correcting loops eliminate hallucination caused by irrelevant context passages.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: What are the foundational architectural differences between Traditional RAG and Agentic RAG?"):
    st.markdown("""
    **Answer:**  
    Traditional RAG is a rigid, feed-forward Directed Acyclic Graph (DAG) that unconditionally executes retrieve-then-generate. 
    It cannot evaluate whether retrieval was necessary, whether retrieved passages are relevant, or whether the final answer 
    was hallucinated. Agentic RAG implements an autonomous cyclic state machine: it routes conditionally across multiple tools, 
    grades document relevance before generation, and executes iterative self-correction loops when information is missing.
    """)

with st.expander("Q2: How does Corrective RAG (CRAG) handle ambiguous or irrelevant retrieval results?"):
    st.markdown("""
    **Answer:**  
    CRAG utilizes a lightweight retrieval evaluator to score retrieved document confidence. 
    - If confidence is high, it prunes the document and generates the answer.
    - If confidence is low (irrelevant), it discards the retrieved chunks and triggers an alternate search (such as public Web Search).
    - If confidence is ambiguous, it combines both internal vector chunks and external web search snippets to ensure comprehensive coverage.
    """)

with st.expander("Q3: Why is document grading performed prior to final answer generation?"):
    st.markdown("""
    **Answer:**  
    Document grading serves as a quality firewall. If irrelevant chunks are passed directly into the generation LLM's prompt, 
    the model will either produce an irrelevant answer or hallucinate plausible-sounding facts to fill the void. 
    Grading before generation catches retrieval failures early, saving expensive frontier LLM generation tokens and allowing 
    the system to rewrite the query.
    """)

with st.expander("Q4: How does multi-hop iterative retrieval avoid infinite loops?"):
    st.markdown("""
    **Answer:**  
    Iterative multi-hop systems enforce strict termination guards:
    1. A hard `recursion_limit` or max step counter (e.g., maximum 3 search hops).
    2. A convergence detector that checks whether new retrieval passes yield information deltas or repeat previous queries.
    3. State checkpointers that persist intermediate search discoveries, allowing the agent to fail gracefully if facts cannot be found.
    """)

with st.expander("Q5: What is the role of Hallucination Grading (Faithfulness Scoring) in production agent systems?"):
    st.markdown("""
    **Answer:**  
    Faithfulness evaluation compares the claims in the generated response strictly against the text in the retrieved chunks. 
    If the response introduces ungrounded factual claims not present in the reference documents, the faithfulness check flags 
    the run as hallucinated and triggers regeneration with a lower model temperature and stricter system prompt.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Building Agentic RAG Without Loop Termination Guards**  
Allowing an agent to loop until an answer is found will cause infinite loops if the user asks for proprietary secrets not in the corpus. 
Always enforce a strict `max_retries = 2` guardrail.
""")

st.error("""
**Trap 2: Using Frontier Models (GPT-4o) for Simple Document Relevance Grading**  
Running full 128k-context frontier models just to output a binary 'RELEVANT/IRRELEVANT' grade creates massive latency (1,500ms) 
and high costs. Use small, specialized models (e.g., GPT-4o-mini, Claude 3 Haiku, or fine-tuned DeBERTa) for grading.
""")

st.warning("""
**Trap 3: Unconditional Web Search Fallbacks for Proprietary Queries**  
Routing internal enterprise queries to public web search when vector search fails risks leaking company-internal queries to third-party search APIs. 
Enforce privacy-aware fallback policies.
""")

st.warning("""
**Trap 4: Discarding Intermediate Hop State in Multi-Hop Retrieval**  
Failing to maintain a structured list of gathered facts across search hops leads the agent to repeat the same search query multiple times. 
Store verified facts in an explicit state accumulator channel.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Self-RAG</h4>
        <p>A retrieval framework where an agent generates reflection tokens to evaluate when to retrieve, critique passages, and verify answer faithfulness.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Corrective RAG (CRAG)</h4>
        <p>A self-correcting RAG pipeline that evaluates retrieval confidence and dynamically triggers query rewrites or web search fallbacks.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Intent Gatekeeper</h4>
        <p>A routing node evaluating whether user requests require external retrieval or can be answered parametrically, preventing unnecessary vector search overhead.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Faithfulness Evaluator</h4>
        <p>An automated judge checking whether every factual claim in the generated answer is strictly grounded in the retrieved context.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architectural differences between feed-forward RAG and Agentic RAG cyclic state machines.", value=True)
chk_2 = st.checkbox("I know how an Intent Gatekeeper prevents unnecessary retrieval for parametric questions.", value=True)
chk_3 = st.checkbox("I can implement dynamic routing across specialized stores (VectorDB, SQL, Web Search).", value=True)
chk_4 = st.checkbox("I understand the iterative Search -> Analyze -> Search multi-hop reasoning cycle.", value=True)
chk_5 = st.checkbox("I know how Document Relevance Graders protect the generation prompt from context poisoning.", value=True)
chk_6 = st.checkbox("I understand how Corrective RAG (CRAG) executes fallback searches and automated query reformulations.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Agentic RAG competencies verified! Chapter 37 Complete.")
