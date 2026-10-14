import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="19.4 Agentic RAG Architecture",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Module-Level Engine Classes ---

class AgenticRAGEngine:
    """Cyclic Agentic RAG Engine featuring ReAct loops, tool calling, Self-RAG reflection, and Corrective (CRAG) loops."""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        
        # Mock Domain Repositories & Tools
        self.vector_store = {
            "apple_revenue_2025": "Apple Inc. reported total FY2025 revenue of $391.0 billion, up 2% year-over-year, driven by iPhone sales of $201.2 billion and Services revenue of $96.2 billion.",
            "apple_rnd_2025": "Apple Inc. invested $31.4 billion in Research and Development (R&D) in FY2025, primarily targeting Apple Intelligence, custom Apple Silicon, and spatial computing technologies.",
            "microsoft_revenue_2025": "Microsoft Corp reported FY2025 revenue of $245.1 billion, with Microsoft Cloud revenue contributing $137.0 billion (up 23% YoY).",
            "microsoft_rnd_2025": "Microsoft Corp spent $29.5 billion on Research and Development in FY2025, heavily focused on cloud AI infrastructure and Copilot integration."
        }
        
    def tool_vector_search(self, query: str) -> str:
        """Tool 1: Semantic Vector Search over corporate 10-K filings."""
        q_lower = query.lower()
        results = []
        for key, text in self.vector_store.items():
            if any(term in key for term in ["apple", "revenue", "rnd", "microsoft"] if term in q_lower):
                results.append(f"[{key.upper()}]: {text}")
        return "\n".join(results) if results else "No direct matches found in vector index."

    def tool_sql_query(self, query: str) -> str:
        """Tool 2: Structured SQL Analytics Engine."""
        if "ratio" in query.lower() or "percentage" in query.lower() or "compare" in query.lower():
            return "SQL Result: Apple R&D / Revenue = 8.03% ($31.4B / $391.0B) | Microsoft R&D / Revenue = 12.04% ($29.5B / $245.1B)."
        return "SQL Result: Query executed. Metric records aggregated successfully."

    def tool_web_search(self, query: str) -> str:
        """Tool 3: Fallback Live Web Search Connector."""
        return f"Web Search Result: Consensus analyst estimates confirm Apple FY2025 revenue reached $391.0B and Microsoft cloud ARR grew 23%."

    def self_reflect_groundedness(self, answer: str, context: str) -> Dict[str, Any]:
        """Self-RAG Reflection: Audits whether the generated claims are supported by context."""
        if not context or "No direct matches" in context:
            return {"groundedness": 0.20, "is_supported": False, "verdict": "UNSUPPORTED"}
            
        c_words = set(context.lower().split())
        a_words = [w.strip(".,;:?!()[]\"'") for w in answer.lower().split() if len(w) > 4]
        if not a_words:
            return {"groundedness": 1.0, "is_supported": True, "verdict": "SUPPORTED"}
            
        matches = [w for w in a_words if w in c_words]
        score = float(len(matches) / len(a_words))
        is_supported = score >= 0.50
        
        return {
            "groundedness": round(score, 3),
            "is_supported": is_supported,
            "verdict": "SUPPORTED" if is_supported else "HALLUCINATION_DETECTED"
        }

    def run_agentic_loop(self, user_query: str, enable_crag_fallback: bool = True) -> Dict[str, Any]:
        """Execute cyclic multi-turn ReAct reasoning and corrective RAG flow."""
        trace = []
        accumulated_context = ""
        current_query = user_query
        iteration = 0
        final_answer = ""
        resolved = False

        while iteration < self.max_iterations and not resolved:
            iteration += 1
            step_record = {"iteration": iteration, "query_used": current_query}
            
            # 1. Agent Reasoning & Tool Selection
            if iteration == 1:
                thought = "Initial query requires enterprise filing retrieval. Selecting Tool: vector_search."
                tool_selected = "vector_search"
                obs = self.tool_vector_search(current_query)
            elif iteration == 2 and "ratio" in user_query.lower():
                thought = "Retrieved financial fundamentals. Need precise financial calculation. Selecting Tool: sql_query."
                tool_selected = "sql_query"
                obs = self.tool_sql_query(current_query)
            else:
                if enable_crag_fallback:
                    thought = "Prior context deemed insufficient or ungrounded. Corrective action: Selecting Tool: web_search."
                    tool_selected = "web_search"
                    obs = self.tool_web_search(current_query)
                else:
                    thought = "Refining search terms within internal vector store."
                    tool_selected = "vector_search"
                    obs = self.tool_vector_search(current_query)

            step_record["thought"] = thought
            step_record["tool"] = tool_selected
            step_record["observation"] = obs
            
            accumulated_context += f"\n{obs}"
            
            # 2. Draft Synthesis
            draft_answer = (
                f"Based on aggregated evidence: {obs[:160]}... "
                f"Answering '{user_query}'."
            )
            step_record["draft_answer"] = draft_answer
            
            # 3. Self-Reflection
            reflection = self.self_reflect_groundedness(draft_answer, accumulated_context)
            step_record["reflection"] = reflection
            
            trace.append(step_record)
            
            if reflection["is_supported"]:
                final_answer = (
                    f"Verified Synthesis (Completed in {iteration} hops):\n"
                    f"{accumulated_context.strip()}\n\n"
                    f"Direct Answer: In FY2025, Apple generated $391.0B revenue with $31.4B in R&D (8.03%), "
                    f"while Microsoft reported $245.1B revenue with $29.5B in R&D (12.04%)."
                )
                resolved = True
            else:
                # 4. Self-Correction (CRAG loop)
                current_query = f"{user_query} verified corporate filings"

        if not resolved:
            final_answer = f"Agent terminated after {self.max_iterations} hops. Best available synthesis: {draft_answer}"

        return {
            "final_answer": final_answer,
            "total_iterations": iteration,
            "resolved": resolved,
            "execution_trace": trace
        }


# --- Streamlit Layout ---

st.title("19.4 Agentic RAG Architecture")
st.caption("Cyclic State Machines: Autonomous ReAct Loops, Structured Tool Calling, Dynamic Search, Self-Reflection & CRAG")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Cyclic Agentic RAG & Multi-Turn State Machine Architecture", divider="rainbow")

st.markdown(
    """
    While Advanced and Modular RAG pipelines improve precision, they remain fundamentally constrained by linear DAG execution:
    they make a single pass through retrieval and generation. If the retrieved context is missing a crucial fact, the pipeline has no mechanism
    to realize its mistake, formulate a follow-up query, or call a different data tool.
    
    **Agentic RAG** introduces **autonomous cyclic decision-making**. Powered by state machine loops (such as LangGraph or ReAct agents),
    an agent actively evaluates its intermediate reasoning, invokes specialized tools (Vector DBs, SQL engines, Web APIs),
    inspects candidate evidence, scores its own groundedness (**Self-RAG**), and automatically executes corrective retrieval (**CRAG**) when evidence is lacking.
    """
)

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Agentic_RAG_Architecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.7, fontname="Inter"];
        node [shape=box, style="filled,rounded", fontname="Inter", fontsize=11, color="#38BDF8", penwidth=1.5];
        edge [color="#64748B", fontname="Inter", fontsize=10, fontcolor="#94A3B8", penwidth=1.2];

        subgraph cluster_agent_core {
            label = "1. AUTONOMOUS REASONING & ORCHESTRATION CORE";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            user_query [label="User Complex Multi-Hop Query\\n'Compare Apple & MSFT R&D Intensity'", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            react_agent [label="ReAct Supervisor Agent\\n(StateGraph: Thought -> Action -> Observation)", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            tool_dispatcher [label="Tool Calling Dispatcher\\n(Structured JSON Function Calling)", fillcolor="#312E81", fontcolor="#A5B4FC"];
            
            user_query -> react_agent -> tool_dispatcher;
        }

        subgraph cluster_tool_arsenal {
            label = "2. TOOL ARSENAL (HETEROGENEOUS EXECUTION)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            tool_vector [label="Tool 1: Vector Search\\n(Unstructured 10-K Filings)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            tool_sql [label="Tool 2: Text-to-SQL DB\\n(Structured Financial Tables)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            tool_web [label="Tool 3: Live Web Search\\n(Real-Time External Corpus)", fillcolor="#0F172A", fontcolor="#38BDF8"];
            
            tool_dispatcher -> tool_vector [label="Call Vector Tool"];
            tool_dispatcher -> tool_sql [label="Call SQL Tool"];
            tool_dispatcher -> tool_web [label="Call Web Tool"];
        }

        subgraph cluster_reflection_loop {
            label = "3. CRITICAL EVALUATION & SELF-REFLECTION (SELF-RAG)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            context_agg [label="Context Accumulator & Working Memory", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            draft_synthesis [label="Candidate Response Drafter", fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            self_reflection [label="Self-Reflection Evaluator\\n- Is Context Relevant?\\n- Is Answer Grounded?\\n- Are Citations Authentic?", fillcolor="#4C1D95", fontcolor="#DDD6FE"];
            decision_gate [label="Reflection Evaluation Gate\\n(Supported & Complete?)", shape=diamond, fillcolor="#1E3A8A", fontcolor="#60A5FA"];
            
            tool_vector -> context_agg [label="Observation 1"];
            tool_sql -> context_agg [label="Observation 2"];
            tool_web -> context_agg [label="Observation 3"];
            context_agg -> draft_synthesis -> self_reflection -> decision_gate;
        }

        subgraph cluster_corrective_action {
            label = "4. SELF-CORRECTION & TERMINATION (CRAG)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            fontcolor = "#F8FAFC";
            
            corrective_rewriter [label="Corrective Query Reformulator\\n(CRAG: Expand Search Scope / Fallback)", fillcolor="#7F1D1D", fontcolor="#FCA5A5"];
            final_grounded_output [label="Final Grounded Response\\nMulti-Hop Verified Synthesis", fillcolor="#064E3B", fontcolor="#6EE7B7"];
            
            decision_gate -> final_grounded_output [label="YES: Terminate State Machine"];
            decision_gate -> corrective_rewriter [label="NO: Incomplete / Hallucination"];
            corrective_rewriter -> react_agent [label="Cyclic Feedback (Next Hop)", color="#F43F5E", penwidth=2.0];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: AGENT-BASED RETRIEVAL ---
st.header("2. Agent-Based Retrieval (Autonomous Decision-Making, ReAct Loops & Dynamic Termination)", divider="rainbow")

st.markdown(
    """
    In **Agent-Based Retrieval**, the retrieval step is not an automatic pipeline stage; it is an **autonomous action** initiated by an agent.
    The agent observes the user query, reasons about what information is required, decides whether retrieval is even necessary,
    and determines when enough evidence has been collected to stop searching.
    
    ### The ReAct (Reason + Act) Execution Pattern:
    1. **Thought**: The LLM generates an explicit internal monologue reasoning about what information is missing.
    2. **Action**: The agent selects a tool and provides formatted JSON arguments (e.g., `{"tool": "vector_search", "query": "Apple FY2025 R&D expenditure"}`).
    3. **Observation**: The system executes the tool and appends the result to the agent's scratchpad working memory.
    4. **Loop / Terminate**: The agent evaluates if the observation satisfies the goal. If yes, it emits the final answer; if no, it initiates another thought-action cycle.
    """
)

# --- SECTION 3: TOOL-BASED RETRIEVAL ---
st.header("3. Tool-Based Retrieval (Structured Function Calling, SQL Generation, Web Search & Calculation Tools)", divider="rainbow")

st.markdown(
    """
    Real-world enterprise knowledge does not reside solely in unstructured PDF text.
    It is distributed across relational databases, analytical data warehouses, calculation engines, and third-party APIs.
    
    ### Heterogeneous Tool Arsenal:
    """
)

col_tb1, col_tb2, col_tb3 = st.columns(3)
with col_tb1:
    with st.container(border=True):
        st.markdown("#### 1. Text-to-SQL Tool")
        st.markdown(
            """
            - **Target**: PostgreSQL, Snowflake, BigQuery.
            - **Capability**: Queries exact structured metrics, aggregations, sums, and historical revenue tables without fuzzy vector hallucination.
            """
        )
with col_tb2:
    with st.container(border=True):
        st.markdown("#### 2. Vector Search Tool")
        st.markdown(
            """
            - **Target**: Milvus, Qdrant, Pinecone.
            - **Capability**: Queries semantic narratives, executive commentary, policy manuals, and technical specifications.
            """
        )
with col_tb3:
    with st.container(border=True):
        st.markdown("#### 3. Real-Time Web Tool")
        st.markdown(
            """
            - **Target**: Tavily Search, Google Search API.
            - **Capability**: Retrieves live breaking news, stock prices, and information not present in the internal knowledge base.
            """
        )

# --- SECTION 4: DYNAMIC SEARCH ---
st.header("4. Dynamic Search (Adaptive Query Reformulation, Adaptive Top-K & Context-Aware Search Depth)", divider="rainbow")

st.markdown(
    """
    Static RAG uses fixed retrieval parameters (e.g., always `Top-K = 4`).
    In contrast, **Dynamic Search** enables the agent to modulate its retrieval behavior based on search confidence:
    
    - **Adaptive Top-K**: For simple factoid queries, retrieve `K = 1`. For broad thematic overviews, expand dynamically to `K = 12`.
    - **Adaptive Search Depth**: If initial chunk scores indicate high ambiguity, dynamically trigger dense index traversal with expanded HNSW `ef_search` parameters.
    - **Adaptive Query Specialization**: If the first query returns generic results, reformulate the query with exact terms discovered in the first observation.
    """
)

# --- SECTION 5: ITERATIVE RETRIEVAL ---
st.header("5. Iterative Retrieval (Multi-Hop Graph Traversal, Context Accumulation & Hypothesis Refinement)", divider="rainbow")

st.markdown(
    """
    Many complex enterprise questions cannot be answered with a single document retrieval.
    Consider the query: *"Which subsidiary acquired the startup founded by the creator of PyTorch, and what was the valuation?"*
    
    This requires **Multi-Hop Iterative Retrieval**:
    - **Hop 1**: Search for the creator of PyTorch (*Soumith Chintala* / *Adam Paszke*).
    - **Hop 2**: Search for startups founded by that individual.
    - **Hop 3**: Search for acquisitions of that startup.
    - **Hop 4**: Search for the financial valuation of that transaction.
    
    The agent maintains an in-memory **Working State Context**, accumulating intermediate findings across each hop until the full dependency graph is resolved.
    """
)

# --- SECTION 6: SELF-REFLECTION ---
st.header("6. Self-Reflection (Groundedness Self-Scoring, Answer Relevance Verification & Critique Prompts)", divider="rainbow")

st.markdown(
    """
    **Self-RAG** (Asai et al., 2023) introduced the paradigm of reflective tokens and self-critique.
    Instead of assuming that generated text is valid, the agent prompts itself to evaluate its own output before emitting it to the user.
    
    ### The Three Core Reflection Checks:
    1. **Is-Retrieved-Context-Useful?** (Did the tool return signal or irrelevant noise?)
    2. **Is-Answer-Supported? (Faithfulness)** (Can every single clause in the draft answer be traced to explicit evidence in working memory?)
    3. **Is-Answer-Complete?** (Did the answer satisfy all constraints of the user's initial prompt, or are there dangling questions?)
    """
)

# --- SECTION 7: SELF-CORRECTION ---
st.header("7. Self-Correction (Corrective Query Rewriting, Fallback Retrieval & Hallucination Repair Loops)", divider="rainbow")

st.markdown(
    """
    When the Self-Reflection check evaluates to **UNSUPPORTED** or **INCOMPLETE**, the agent does not fail.
    It initiates a **Corrective RAG (CRAG)** loop:
    
    - **Query Reformulation**: The agent analyzes why the previous search failed and generates a targeted alternative query.
    - **Fallback to Web / Alternative Repositories**: If internal vector indices have zero relevant knowledge, CRAG automatically pivots to an external search tool or secondary corpus.
    - **Hallucination Repair**: Sentences in the draft answer that failed groundedness audits are stripped out or explicitly replaced with an honest disclaimer.
    """
)

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Cyclic Agentic RAG State Machine (`AgenticRAGEngine`)", divider="rainbow")

st.markdown(
    """
    Below is the complete reference implementation of `AgenticRAGEngine`.
    It implements multi-turn ReAct reasoning, tool dispatching across vector, SQL, and web connectors,
    Self-RAG groundedness reflection, and automated Corrective RAG (CRAG) loops.
    """
)

st.code(
    '''
from typing import List, Dict, Any, Tuple, Optional

class AgenticRAGEngine:
    """Cyclic Agentic RAG Engine featuring ReAct loops, tool calling, Self-RAG reflection, and Corrective (CRAG) loops."""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.vector_store = {
            "apple_revenue_2025": "Apple Inc. reported total FY2025 revenue of $391.0 billion, up 2% year-over-year, driven by iPhone sales of $201.2 billion and Services revenue of $96.2 billion.",
            "apple_rnd_2025": "Apple Inc. invested $31.4 billion in Research and Development (R&D) in FY2025, primarily targeting Apple Intelligence, custom Apple Silicon, and spatial computing technologies.",
            "microsoft_revenue_2025": "Microsoft Corp reported FY2025 revenue of $245.1 billion, with Microsoft Cloud revenue contributing $137.0 billion (up 23% YoY).",
            "microsoft_rnd_2025": "Microsoft Corp spent $29.5 billion on Research and Development in FY2025, heavily focused on cloud AI infrastructure and Copilot integration."
        }

    def tool_vector_search(self, query: str) -> str:
        """Tool 1: Semantic Vector Search over corporate 10-K filings."""
        q_lower = query.lower()
        results = []
        for key, text in self.vector_store.items():
            if any(term in key for term in ["apple", "revenue", "rnd", "microsoft"] if term in q_lower):
                results.append(f"[{key.upper()}]: {text}")
        return "\\n".join(results) if results else "No direct matches found in vector index."

    def tool_sql_query(self, query: str) -> str:
        """Tool 2: Structured SQL Analytics Engine."""
        if "ratio" in query.lower() or "percentage" in query.lower() or "compare" in query.lower():
            return "SQL Result: Apple R&D / Revenue = 8.03% ($31.4B / $391.0B) | Microsoft R&D / Revenue = 12.04% ($29.5B / $245.1B)."
        return "SQL Result: Query executed. Metric records aggregated successfully."

    def tool_web_search(self, query: str) -> str:
        """Tool 3: Fallback Live Web Search Connector."""
        return f"Web Search Result: Consensus analyst estimates confirm Apple FY2025 revenue reached $391.0B and Microsoft cloud ARR grew 23%."

    def self_reflect_groundedness(self, answer: str, context: str) -> Dict[str, Any]:
        """Self-RAG Reflection: Audits whether the generated claims are supported by context."""
        if not context or "No direct matches" in context:
            return {"groundedness": 0.20, "is_supported": False, "verdict": "UNSUPPORTED"}
            
        c_words = set(context.lower().split())
        a_words = [w.strip(".,;:?!()[]\\"'") for w in answer.lower().split() if len(w) > 4]
        if not a_words:
            return {"groundedness": 1.0, "is_supported": True, "verdict": "SUPPORTED"}
            
        matches = [w for w in a_words if w in c_words]
        score = float(len(matches) / len(a_words))
        is_supported = score >= 0.50
        
        return {
            "groundedness": round(score, 3),
            "is_supported": is_supported,
            "verdict": "SUPPORTED" if is_supported else "HALLUCINATION_DETECTED"
        }

    def run_agentic_loop(self, user_query: str, enable_crag_fallback: bool = True) -> Dict[str, Any]:
        """Execute cyclic multi-turn ReAct reasoning and corrective RAG flow."""
        trace = []
        accumulated_context = ""
        current_query = user_query
        iteration = 0
        final_answer = ""
        resolved = False

        while iteration < self.max_iterations and not resolved:
            iteration += 1
            step_record = {"iteration": iteration, "query_used": current_query}
            
            if iteration == 1:
                thought = "Initial query requires enterprise filing retrieval. Selecting Tool: vector_search."
                tool_selected = "vector_search"
                obs = self.tool_vector_search(current_query)
            elif iteration == 2 and "ratio" in user_query.lower():
                thought = "Retrieved financial fundamentals. Need precise financial calculation. Selecting Tool: sql_query."
                tool_selected = "sql_query"
                obs = self.tool_sql_query(current_query)
            else:
                if enable_crag_fallback:
                    thought = "Prior context deemed insufficient or ungrounded. Corrective action: Selecting Tool: web_search."
                    tool_selected = "web_search"
                    obs = self.tool_web_search(current_query)
                else:
                    thought = "Refining search terms within internal vector store."
                    tool_selected = "vector_search"
                    obs = self.tool_vector_search(current_query)

            step_record["thought"] = thought
            step_record["tool"] = tool_selected
            step_record["observation"] = obs
            
            accumulated_context += f"\\n{obs}"
            
            draft_answer = (
                f"Based on aggregated evidence: {obs[:160]}... "
                f"Answering '{user_query}'."
            )
            step_record["draft_answer"] = draft_answer
            
            reflection = self.self_reflect_groundedness(draft_answer, accumulated_context)
            step_record["reflection"] = reflection
            trace.append(step_record)
            
            if reflection["is_supported"]:
                final_answer = (
                    f"Verified Synthesis (Completed in {iteration} hops):\\n"
                    f"{accumulated_context.strip()}\\n\\n"
                    f"Direct Answer: In FY2025, Apple generated $391.0B revenue with $31.4B in R&D (8.03%), "
                    f"while Microsoft reported $245.1B revenue with $29.5B in R&D (12.04%)."
                )
                resolved = True
            else:
                current_query = f"{user_query} verified corporate filings"

        if not resolved:
            final_answer = f"Agent terminated after {self.max_iterations} hops. Best available synthesis: {draft_answer}"

        return {
            "final_answer": final_answer,
            "total_iterations": iteration,
            "resolved": resolved,
            "execution_trace": trace
        }
''',
    language="python"
)

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Self-RAG & Corrective RAG (CRAG) Multi-Turn Loop Simulator", divider="rainbow")

st.markdown(
    """
    Simulate a live multi-hop agentic RAG loop.
    Observe how the agent formulates reasoning thoughts, invokes heterogeneous tools (Vector Search, SQL Analytics, Web Search),
    performs Self-RAG groundedness reflection, and initiates CRAG corrective recovery.
    """
)

wb_col1, wb_col2 = st.columns([1, 1])

with wb_col1:
    st.subheader("🤖 Agent Configuration & Goals")
    
    preset_task = st.selectbox(
        "Select Agent Task Preset:",
        [
            "Task 1: Compare Apple and Microsoft FY2025 revenue and calculate R&D intensity ratio",
            "Task 2: Retrieve Apple FY2025 Services revenue and R&D expenditure",
            "Task 3: Unanswerable query testing CRAG web search fallback"
        ],
        index=0
    )
    
    agent_query = st.text_input("Active Objective:", preset_task.split(": ", 1)[1])
    max_hops = st.slider("Max Permitted ReAct Hops (Recursion Limit):", min_value=1, max_value=5, value=3, step=1)
    enable_crag = st.toggle("Enable Corrective RAG (CRAG) Web Fallback", value=True)

with wb_col2:
    st.subheader("⚡ Execution Trace & Final Output")
    
    engine = AgenticRAGEngine(max_iterations=max_hops)
    result = engine.run_agentic_loop(agent_query, enable_crag_fallback=enable_crag)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total ReAct Hops", result["total_iterations"])
    m2.metric("Goal Resolved", "YES ✅" if result["resolved"] else "NO ❌")
    last_reflection = result["execution_trace"][-1]["reflection"] if result["execution_trace"] else {"groundedness": 0.0}
    m3.metric("Final Groundedness", f"{last_reflection['groundedness'] * 100:.1f}%")
    
    st.markdown("#### Final Agent Synthesized Response")
    if result["resolved"]:
        st.success(result["final_answer"])
    else:
        st.warning(result["final_answer"])

st.markdown("### Step-by-Step ReAct Monologue & Tool Execution Waterfall")

for step in result["execution_trace"]:
    with st.expander(f"🔄 Hop {step['iteration']}: Tool Invocation [{step['tool']}] - Verdict: {step['reflection']['verdict']}", expanded=True):
        st.markdown(f"**💭 Thought**: `{step['thought']}`")
        st.markdown(f"**🛠️ Tool Called**: `{step['tool']}`")
        st.markdown(f"**🔎 Observation**: {step['observation']}")
        st.markdown(f"**📝 Draft Answer**: {step['draft_answer']}")
        st.markdown(
            f"**🪞 Self-Reflection**: Groundedness Score = `{step['reflection']['groundedness']}` | "
            f"Supported = `{step['reflection']['is_supported']}`"
        )

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### 5 High-Impact Technical Interview Questions")

with st.expander("**Q1: How does Agentic RAG differ fundamentally from Advanced and Modular RAG?**"):
    st.markdown(
        """
        - **Advanced & Modular RAG** are **Directed Acyclic Graphs (DAGs)**. Regardless of complexity or routing, execution flows in a single forward pass: Input -> Retrieve -> Generate.
        - **Agentic RAG** is a **cyclic state machine**. The model can dynamically decide to loop backwards: it can retrieve, inspect the retrieved content, decide that the information is insufficient or contradictory, formulate a new query, call a different tool (such as SQL or web search), and iterate until its internal reflection criteria are satisfied.
        """
    )

with st.expander("**Q2: What is the Self-RAG framework and what are its special reflection tokens?**"):
    st.markdown(
        """
        **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) trains language models to emit specialized reflection tokens during generation:
        1. `[Retrieve]`: Predicts whether external retrieval is needed for the current token sequence.
        2. `[IsREL]` (Is Relevant): Evaluates whether retrieved passage `d` contains relevant information.
        3. `[IsSUP]` (Is Supported): Assesses whether the model's generated claim is completely supported by `d` (anti-hallucination token).
        4. `[IsUSE]` (Is Useful): Evaluates whether the generated response is useful and addresses the user's intent.
        
        These tokens allow the system to prune hallucinated candidate beams during beam search decoding.
        """
    )

with st.expander("**Q3: What is Corrective RAG (CRAG) and how does it handle ambiguous retrieval results?**"):
    st.markdown(
        """
        **Corrective RAG (CRAG)** implements an autonomous retrieval evaluator that scores retrieved documents into three confidence tiers:
        1. **Correct (Confidence > Upper Threshold)**: Retain documents and apply knowledge refinement (extractive stripping).
        2. **Incorrect (Confidence < Lower Threshold)**: Discard all retrieved documents and immediately execute a fallback web search query to prevent parametric hallucination.
        3. **Ambiguous (In-Between)**: Combine internal retrieved documents with web search results, using hybrid fusion before passing to generation.
        """
    )

with st.expander("**Q4: How do you prevent infinite execution loops in cyclic agentic systems?**"):
    st.markdown(
        """
        Unbounded ReAct agents can loop indefinitely if tools return unhelpful data. Production systems enforce strict guardrails:
        1. **Hard Recursion Limits**: Enforcing `max_iterations = 3` or `max_hops = 5` in LangGraph state machines.
        2. **Global Timeout Budgets**: Aborting the agent execution if total wall-clock time exceeds an SLA ceiling (e.g., 5.0 seconds).
        3. **State Cycle Detection**: Hashing tool inputs and outputs; if the agent attempts to call the exact same tool with the exact same query twice, force state transition to fallback generation.
        """
    )

with st.expander("**Q5: Why is tool schema definition (JSON Schema / Pydantic) critical for agentic retrieval?**"):
    st.markdown(
        """
        LLMs do not call Python functions directly; they generate JSON strings representing function calls.
        Clear parameter descriptions, type constraints, and docstrings serve as the prompt context for the model.
        If a schema lacks explicit field descriptions (e.g., whether a date format is `YYYY-MM-DD` or timestamp),
        the agent generates malformed arguments, triggering syntax errors and failed tool executions.
        """
    )

st.markdown("### 4 Production Traps & Antipatterns")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.error(
        """
        🚨 **Trap 1: Deploying ReAct Agents Without Recursion Limits**  
        **Antipattern**: Letting an autonomous agent execute while loops without a hard hop ceiling.  
        **Consequence**: When an answer does not exist, the agent loops continuously through web search and vector tools, consuming $50+ in API credits in minutes.  
        **Production Fix**: Set hard recursion limits (`max_hops = 3`) and strict timeout ceilings in LangGraph.
        """
    )
    st.error(
        """
        🚨 **Trap 2: Context Window Accumulation Bloat**  
        **Antipattern**: Appending the complete raw observations of every tool call into the working prompt memory across 5 hops.  
        **Consequence**: Context ballooning causes prompt lengths to exceed 20,000 tokens, degrading TTFT latency to 8+ seconds.  
        **Production Fix**: Apply summarization or extractive compression to tool observations before appending them to the agent state.
        """
    )
with t_col2:
    st.warning(
        """
        ⚠️ **Trap 3: Using Agentic Loops for Simple Factoid Queries**  
        **Antipattern**: Running a 3-hop ReAct agent for simple lookups like *"What is the HR email?"*.  
        **Consequence**: Multiplies latency by 4x and cost by 5x for a query that basic BM25 could resolve in 2ms.  
        **Production Fix**: Place a lightweight Query Router in front of the agent to route simple queries to Basic or Modular RAG.
        """
    )
    st.warning(
        """
        ⚠️ **Trap 4: Hallucinated Tool Argument Generation**  
        **Antipattern**: Defining tool parameter schemas with vague descriptions like `{"query": "string"}`.  
        **Consequence**: The agent passes malformed SQL syntax, inverted dates, or unescaped strings that crash database tools.  
        **Production Fix**: Validate all tool inputs with Pydantic schemas and provide unambiguous few-shot parameter examples.
        """
    )

st.markdown("### 4 Essential Concept Flashcards")

f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    with st.container(border=True):
        st.markdown("**Front**: What defines Agentic RAG?")
        st.markdown("**Back**: An architecture where retrieval is an autonomous, cyclic action governed by an agent that uses ReAct loops, tool calling, and self-reflection to iteratively gather evidence.")
with f_col2:
    with st.container(border=True):
        st.markdown("**Front**: What is the ReAct loop?")
        st.markdown("**Back**: Thought -> Action -> Observation. The agent reasons about missing facts, calls a specialized tool, observes results, and iterates until the goal is achieved.")
with f_col3:
    with st.container(border=True):
        st.markdown("**Front**: What is Self-RAG?")
        st.markdown("**Back**: A framework where the model self-evaluates whether retrieved passages are relevant (IsREL) and whether its generated answers are fully supported (IsSUP) by context.")
with f_col4:
    with st.container(border=True):
        st.markdown("**Front**: What is Corrective RAG (CRAG)?")
        st.markdown("**Back**: A framework that evaluates retrieval confidence; if internal context is poor, it triggers corrective query rewriting and falls back to external web search.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your technical understanding of the enterprise Agentic RAG architecture and cyclic state machines:")

chk1 = st.checkbox("I understand the architectural shift from linear DAGs to cyclic multi-turn state machines in Agentic RAG.", value=True)
chk2 = st.checkbox("I can implement the ReAct pattern (Thought -> Action -> Observation) for autonomous tool calling.", value=True)
chk3 = st.checkbox("I know how to integrate heterogeneous tools including vector search, text-to-SQL, and live web APIs.", value=True)
chk4 = st.checkbox("I understand how Self-RAG reflection tokens audit groundedness and prevent unanchored hallucinations.", value=True)
chk5 = st.checkbox("I can implement Corrective RAG (CRAG) loops that trigger query reformulation and fallback search when confidence is low.", value=True)
chk6 = st.checkbox("I know how to enforce hard recursion limits, state cycle detection, and context compression to prevent runaway agent costs.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎉 Congratulations! You have mastered the Enterprise Agentic RAG Architecture.")
