import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="10.3 LLM Workflow Design - LangGraph Curriculum",
    page_icon="🗺️",
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
    .badge-node {
        background-color: #1E293B;
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-route {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-fallback {
        background-color: #3B1A1A;
        color: #F87171;
        border: 1px solid #DC2626;
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
class MultiModelWorkflowEngine:
    """
    Simulates multi-model architectural workflows:
    Single-node loops, multi-specialist division of labor, classifier routing,
    and automatic outage fallback cascades.
    """
    def __init__(self):
        self.catalog = {
            "Small Router (gpt-4o-mini)": {
                "role": "Classification & Triage",
                "cost_input_per_m": 0.15,
                "cost_output_per_m": 0.60,
                "latency_base_ms": 190,
                "reasoning_index": 62
            },
            "Frontier Specialist (claude-3-5-sonnet)": {
                "role": "Deep Synthesis & Coding",
                "cost_input_per_m": 3.00,
                "cost_output_per_m": 15.00,
                "latency_base_ms": 820,
                "reasoning_index": 94
            },
            "High-Throughput Generalist (gemini-1.5-flash)": {
                "role": "Large Context & Extraction",
                "cost_input_per_m": 0.075,
                "cost_output_per_m": 0.30,
                "latency_base_ms": 240,
                "reasoning_index": 78
            },
            "Private Backup (ollama-llama-3.3-70b)": {
                "role": "Self-Hosted Air-Gapped Fallback",
                "cost_input_per_m": 0.00,
                "cost_output_per_m": 0.00,
                "latency_base_ms": 480,
                "reasoning_index": 82
            }
        }

    def simulate_workflow(
        self,
        query: str,
        simulate_outage: bool,
        routing_mode: str
    ) -> Dict[str, Any]:
        # Classifier triage
        is_complex = any(k in query.lower() for k in ["proof", "optimize", "architect", "vulnerability", "algorithm", "complex"])
        
        if routing_mode == "Single Model (Always Frontier)":
            selected_tier = "Frontier Specialist (claude-3-5-sonnet)"
            routing_reason = "Static topology: Single node passes all requests to Frontier model regardless of complexity."
        elif routing_mode == "Dynamic Routing by Complexity":
            if is_complex:
                selected_tier = "Frontier Specialist (claude-3-5-sonnet)"
                routing_reason = "Dynamic router classified query as High Complexity -> Dispatched to Frontier reasoning tier."
            else:
                selected_tier = "Small Router (gpt-4o-mini)"
                routing_reason = "Dynamic router classified query as Low/Medium Complexity -> Dispatched to Small Fast tier to save 90 percent cost."
        else:
            selected_tier = "High-Throughput Generalist (gemini-1.5-flash)"
            routing_reason = "High-throughput mode selected: prioritizing low latency."

        # Outage and fallback handling
        fallback_triggered = False
        final_model = selected_tier
        execution_trace = []
        
        execution_trace.append(f"Step 1: Router analyzed query ('{query[:35]}...') -> Selected [{selected_tier}]")

        if simulate_outage and "Frontier" in selected_tier:
            fallback_triggered = True
            execution_trace.append(f"Step 2 (Alert): Primary provider [{selected_tier}] timed out with HTTP 504 Gateway Timeout.")
            final_model = "Private Backup (ollama-llama-3.3-70b)"
            execution_trace.append(f"Step 3 (Fallback): Graph caught outage and cascaded to local fallback [{final_model}].")
        else:
            execution_trace.append(f"Step 2: [{selected_tier}] executed successfully in single attempt.")

        model_meta = self.catalog[final_model]
        tokens_est = int(len(query.split()) * 4 + 180)
        cost = (tokens_est / 1_000_000) * model_meta["cost_input_per_m"] + (150 / 1_000_000) * model_meta["cost_output_per_m"]
        latency = model_meta["latency_base_ms"] + int(np.random.randint(-40, 60))
        if fallback_triggered:
            latency += 1200 # timeout penalty

        return {
            "initial_model": selected_tier,
            "final_model": final_model,
            "fallback_triggered": fallback_triggered,
            "routing_reason": routing_reason,
            "execution_trace": execution_trace,
            "cost_usd": round(cost, 6),
            "latency_ms": latency,
            "reasoning_index": model_meta["reasoning_index"]
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Enterprise Multi-Model LLM Workflow Topologies in LangGraph", divider="rainbow")

st.markdown("""
Designing production LLM applications requires choosing between **single-node**, **multi-node**, **semantic routing**,
and **fallback cascade** architectures. 
Rather than binding the entire system to a single expensive frontier model, LangGraph empowers developers to construct
**heterogeneous topological graphs** where lightweight models classify and route queries, specialist models solve domain sub-tasks,
and automated fallback circuits guarantee enterprise zero-downtime availability.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph MultiModelWorkflows {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        USER_IN [label="User Request Ingestion\\n(MessagesState)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_routing {
            label = "1. Semantic Model Router (Fast / Low Cost)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            ROUTER_NODE [label="Router Node: Fast LLM (gpt-4o-mini)\\nClassifies Intent, Task Complexity, and SLA", fillcolor="#1E1E38", color="#8B5CF6"];
            ROUTE_DECISION [label="Conditional Routing Edge\\nbranch_by_complexity()", shape=diamond, fillcolor="#0F172A", color="#38BDF8"];

            ROUTER_NODE -> ROUTE_DECISION;
        }

        subgraph cluster_specialists {
            label = "2. Specialist Execution Tier";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            CHEAP_SPECIALIST [label="Fast Node: Small Model\\nSimple QA, Summarization, Extraction\\n(~190ms, $0.15/1M tokens)", fillcolor="#064E3B", color="#34D399"];
            DEEP_SPECIALIST [label="Frontier Node: Advanced Reasoner\\nComplex Logic, Architecture, Code Synthesis\\n(~850ms, $3.00/1M tokens)", fillcolor="#0F172A", color="#38BDF8"];
            FALLBACK_RECOVERY [label="Fallback Recovery Node\\nLocal Ollama (Llama-3.3-70b) or Backup Provider\\nActivates on HTTP 429/500/504", fillcolor="#450A0A", color="#F87171"];

            DEEP_SPECIALIST -> FALLBACK_RECOVERY [label="On Exception / Timeout\\nwith_fallbacks()", style=dashed, color="#F87171"];
        }

        subgraph cluster_merge {
            label = "3. State Reconciliation and Final Synthesis";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#475569";
            fontcolor = "#38BDF8";

            SYNTHESIS_NODE [label="Reconciliation Node\\nFormats Final Output and Updates Checkpointer", fillcolor="#0F172A", color="#F59E0B"];
        }

        USER_IN -> ROUTER_NODE [label="START"];
        ROUTE_DECISION -> CHEAP_SPECIALIST [label="Simple / Standard"];
        ROUTE_DECISION -> DEEP_SPECIALIST [label="Complex / Critical"];
        CHEAP_SPECIALIST -> SYNTHESIS_NODE [label="Channel merge"];
        DEEP_SPECIALIST -> SYNTHESIS_NODE [label="Channel merge"];
        FALLBACK_RECOVERY -> SYNTHESIS_NODE [label="Recovery merge"];
        SYNTHESIS_NODE -> END_NODE [label="Committed to Checkpointer"];

        END_NODE [label="END", fillcolor="#0F172A", color="#64748B"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
<b>Architectural Rule:</b> Never send 100 percent of enterprise queries to a frontier model.
A dynamic routing graph achieves identical benchmark quality to pure-frontier architectures
while reducing overall inference expenditure by <b>70 to 85 percent</b> and slashing p50 user latency by half.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: SINGLE LLM NODE ARCHITECTURES
# ==============================================================================
st.header("2. Single LLM Node Architectures: Minimalist Reasoning Loops", divider="rainbow")

st.markdown("""
A **Single LLM Node Architecture** is the foundational building block for conversational assistants and standard ReAct loops.
Here, one model function handles all user turns, system prompting, and optional tool invocation loops.
""")

col_s1, col_s2 = st.columns([1, 1])
with col_s1:
    st.markdown("#### Single Node ReAct Cycle")
    st.code("""
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Single model handles both conversational reasoning and tool calls
builder = StateGraph(MessagesState)

builder.add_node("agent", call_model_node)
builder.add_node("tools", ToolNode(tools=[search_tool, sql_tool]))

builder.add_edge(START, "agent")
# Conditional loop: agent invokes tools until completed
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

react_graph = builder.compile()
""", language="python")

with col_s2:
    st.markdown("#### Trade-Off Analysis: Single Node")
    st.markdown("""
    - **Pros**:
      - Minimal state schema complexity.
      - Seamless conversation continuity with zero handoff serialization.
      - Trivial debugging via single trace path.
    - **Cons**:
      - One size fits all: Simple greetings cost as much as complex multi-step reasoning.
      - Vulnerable to vendor outages (single point of failure).
      - Prompt interference: General conversational guidelines degrade strict tool-calling adherence.
    """)

# ==============================================================================
# SECTION 3: MULTIPLE LLM NODES
# ==============================================================================
st.header("3. Multiple LLM Nodes: Multi-Specialist Decomposition and Handoffs", divider="rainbow")

st.markdown("""
In **Multiple LLM Node Architectures**, the graph breaks a monolithic prompt into distinct specialist nodes,
each equipped with a focused persona, restricted tool access, and separate prompt boundaries.
""")

multi_df = pd.DataFrame([
    {
        "Node Identifier": "triage_classifier",
        "Assigned Model": "gpt-4o-mini",
        "Role": "Scans input, extracts intent, routes traffic",
        "Tools Bound": "None (zero latency overhead)"
    },
    {
        "Node Identifier": "sql_data_analyst",
        "Assigned Model": "claude-3-5-sonnet",
        "Role": "Generates and validates complex analytical SQL queries",
        "Tools Bound": "[run_sql_query, inspect_schema]"
    },
    {
        "Node Identifier": "compliance_auditor",
        "Assigned Model": "gemini-1.5-pro",
        "Role": "Verifies data privacy (PII) against 2M token regulatory docs",
        "Tools Bound": "[check_compliance_policies]"
    },
    {
        "Node Identifier": "editor_synthesizer",
        "Assigned Model": "gpt-4o-mini",
        "Role": "Formats findings into polite, executive-ready Markdown",
        "Tools Bound": "None"
    }
])
st.table(multi_df)

# ==============================================================================
# SECTION 4: MODEL ROUTING
# ==============================================================================
st.header("4. Model Routing: Semantic Classifier-Driven Model Dispatch", divider="rainbow")

st.markdown("""
**Model Routing** uses a conditional edge function to dynamically direct execution to the optimal node
based on state variables or a rapid initial classification pass.
""")

with st.expander("🔍 View Model Routing Decision Logic", expanded=True):
    route_dot = """
    digraph RoutingLogic {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        QUERY [label="Incoming User Query", fillcolor="#0F172A", color="#38BDF8"];
        ROUTER [label="Router Evaluator: def route_model(state)", shape=diamond, fillcolor="#1E1E38", color="#8B5CF6"];

        FAST_NODE [label="route -> 'fast_worker'\\nFor FAQs, text extraction, basic lookup", fillcolor="#064E3B", color="#34D399"];
        CODE_NODE [label="route -> 'code_specialist'\\nFor Python/SQL generation and AST refactoring", fillcolor="#0F172A", color="#38BDF8"];
        REASON_NODE [label="route -> 'deep_reasoner'\\nFor mathematical proofs, logic puzzles", fillcolor="#1E293B", color="#F59E0B"];

        QUERY -> ROUTER;
        ROUTER -> FAST_NODE [label="intent == 'simple'"];
        ROUTER -> CODE_NODE [label="intent == 'coding'"];
        ROUTER -> REASON_NODE [label="intent == 'reasoning'"];
    }
    """
    st.graphviz_chart(route_dot, use_container_width=True)

st.code("""
from typing import Literal

def route_model(state: MessagesState) -> Literal["fast_worker", "code_specialist", "deep_reasoner"]:
    \"\"\"Conditional routing function inspecting latest user query.\"\"\"
    last_msg = state["messages"][-1].content.lower()
    
    if any(k in last_msg for k in ["def ", "class ", "sql", "bug", "traceback"]):
        return "code_specialist"
    elif any(k in last_msg for k in ["derive", "proof", "evaluate", "compare trade-offs"]):
        return "deep_reasoner"
    else:
        return "fast_worker"

# Wire into graph
builder.add_conditional_edges("triage", route_model)
""", language="python")

# ==============================================================================
# SECTION 5: MODEL FALLBACK
# ==============================================================================
st.header("5. Model Fallback: Cascade Recovery and Outage Resilience", divider="rainbow")

st.markdown("""
Cloud APIs experience transient errors: HTTP 429 (Rate Limit Exceeded), HTTP 500 (Internal Server Error),
and HTTP 504 (Gateway Timeout). In LangGraph, fallbacks can be handled at **two distinct layers**:
1. **Model-Level Fallback** (`model.with_fallbacks()`): Completely transparent inside a single node.
2. **Graph-Level Fallback**: Routed through dedicated exception edges into fallback nodes.
""")

with st.expander("🔍 View Dual-Layer Fallback Cascade Flow", expanded=True):
    fallback_dot = """
    digraph FallbackFlow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        INVOKE [label="Node Invocation: primary_model.invoke()\\n(e.g., Primary Cloud Provider API)", fillcolor="#0F172A", color="#38BDF8"];

        subgraph cluster_fallback_engine {
            label = "Automatic Resilient Cascade via with_fallbacks()";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#F87171";
            fontcolor = "#F87171";

            TRY_PRIMARY [label="Primary Execution Attempt", fillcolor="#0F172A", color="#38BDF8"];
            CHECK_ERROR [label="Did Provider Return 429/500/Timeout?", shape=diamond, fillcolor="#1E1E38", color="#F87171"];
            TRY_SECONDARY [label="Fallback 1: Secondary Cloud Vendor (Claude / Gemini)", fillcolor="#1E293B", color="#F59E0B"];
            TRY_TERTIARY [label="Fallback 2: On-Premise Air-Gapped Ollama (Llama 3.3)", fillcolor="#047857", color="#34D399"];

            TRY_PRIMARY -> CHECK_ERROR;
            CHECK_ERROR -> TRY_SECONDARY [label="Yes (API Error)"];
            TRY_SECONDARY -> TRY_TERTIARY [label="Secondary fails"];
        }

        INVOKE -> TRY_PRIMARY;
        CHECK_ERROR -> SUCCESS_RETURN [label="No (200 OK)"];
        TRY_TERTIARY -> SUCCESS_RETURN [label="Local recovery success"];

        SUCCESS_RETURN [label="Valid AIMessage committed to State Channel", fillcolor="#0F172A", color="#34D399"];
    }
    """
    st.graphviz_chart(fallback_dot, use_container_width=True)

st.code("""
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# Define primary and backup model chain
primary_model = ChatOpenAI(model="gpt-4o", max_retries=1, timeout=5.0)
backup_cloud = ChatAnthropic(model="claude-3-5-sonnet-20241022", max_retries=1, timeout=6.0)
backup_local = ChatOllama(model="llama3.3:70b")

# Seamless combined fallback runnable
resilient_model = primary_model.with_fallbacks([backup_cloud, backup_local])

def safe_reasoning_node(state: MessagesState) -> dict:
    # If OpenAI fails or times out, Anthropic executes; if that fails, Ollama executes
    response = resilient_model.invoke(state["messages"])
    return {"messages": [response]}
""", language="python")

# ==============================================================================
# SECTION 6: MODEL SELECTION
# ==============================================================================
st.header("6. Model Selection: Balancing Cost, Latency, and Reasoning Tiers", divider="rainbow")

st.markdown("""
Enterprises maintain a **Model Tiering Matrix** to assign specific models to graph nodes
based on measurable performance trade-offs.
""")

matrix_df = pd.DataFrame([
    {
        "Tier": "Tier 1: Triage & Fast Filter",
        "Representative Models": "gpt-4o-mini, gemini-1.5-flash",
        "Latency Profile": "150 - 250 ms",
        "Blended Cost / 1M Tokens": "approx. $0.35",
        "Best Used For": "Classification, PII redaction, keyword extraction, routing"
    },
    {
        "Tier": "Tier 2: General Reasoning",
        "Representative Models": "gpt-4o, claude-3-5-haiku",
        "Latency Profile": "400 - 650 ms",
        "Blended Cost / 1M Tokens": "approx. $3.50",
        "Best Used For": "Multi-turn dialogues, tool-calling agents, summarization"
    },
    {
        "Tier": "Tier 3: Frontier Synthesis",
        "Representative Models": "claude-3-5-sonnet, o1-preview",
        "Latency Profile": "900 - 2500 ms",
        "Blended Cost / 1M Tokens": "approx. $12.00",
        "Best Used For": "Code generation, architectural proofs, formal policy synthesis"
    },
    {
        "Tier": "Tier 4: Private & Regulated",
        "Representative Models": "Ollama / vLLM Llama 3.3 70B",
        "Latency Profile": "300 - 600 ms (local GPU)",
        "Blended Cost / 1M Tokens": "$0.00 direct API cost",
        "Best Used For": "Air-gapped enterprise, HIPAA/GDPR strict compliance, zero data egress"
    }
])
st.table(matrix_df)

# ==============================================================================
# SECTION 7: MODEL-SPECIFIC WORKFLOWS
# ==============================================================================
st.header("7. Model-Specific Workflows: Tailoring Graph Nodes to Provider Paradigms", divider="rainbow")

st.markdown("""
Different model providers excel under distinct prompting patterns. A mature LangGraph architecture
customizes node inputs to match each model's native strengths:
""")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.info("""
    **Anthropic Claude Paradigms (XML Tagging)**:
    - Claude models exhibit state-of-the-art adherence when complex contextual data is enclosed in explicit XML tags:
    ```xml
    <context>
      <schema>{db_schema}</schema>
      <user_query>{query}</user_query>
    </context>
    ```
    - Anthropic prompt caching activates when static system messages exceed 1024 tokens and remain unchanged across calls.
    """)

with col_m2:
    st.info("""
    **OpenAI & Local Models (JSON Schema & Strict Mode)**:
    - OpenAI models natively optimize for structured tool calls via `tools=[{"type": "function", ...}]`.
    - Activating `strict=True` in structured output guarantees 100 percent adherence to JSON schemas by compiling a context-free grammar at generation time.
    """)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Resilient Heterogeneous Router and Fallback Pipeline", divider="rainbow")

st.markdown("""
Here is the complete production implementation of a multi-model LangGraph workflow featuring
semantic triage routing, model-level fallbacks, and heterogeneous worker dispatch:
""")

st.code("""
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, add_messages

# 1. State Definition
class ResilientEnterpriseState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    route_choice: Literal["fast_lane", "deep_lane"]
    execution_metadata: Dict[str, Any]

# 2. Heterogeneous Model Providers
# Fast triage router
router_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Deep worker with automatic cloud-to-cloud fallback
frontier_primary = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=6.0)
frontier_backup = ChatOpenAI(model="gpt-4o", timeout=6.0)
resilient_deep_model = frontier_primary.with_fallbacks([frontier_backup])

# 3. Graph Nodes
def semantic_router_node(state: ResilientEnterpriseState) -> dict:
    \"\"\"Node 1: Evaluates user intent to determine optimal model lane.\"\"\"
    last_user_query = state["messages"][-1].content
    is_complex = any(term in last_user_query.lower() for term in ["architect", "derive", "refactor", "security"])
    
    route = "deep_lane" if is_complex else "fast_lane"
    return {
        "route_choice": route,
        "execution_metadata": {"router_model": "gpt-4o-mini", "selected_lane": route}
    }

def fast_worker_node(state: ResilientEnterpriseState) -> dict:
    \"\"\"Node 2: Lightweight, ultra-fast model execution.\"\"\"
    system_prompt = SystemMessage(content="You are a quick and concise enterprise support assistant.")
    response = router_model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def deep_worker_node(state: ResilientEnterpriseState) -> dict:
    \"\"\"Node 3: Frontier reasoning model with automatic backup fallback.\"\"\"
    system_prompt = SystemMessage(content="You are a principal staff architect. Provide rigorous, exhaustive solutions.")
    response = resilient_deep_model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# 4. Routing Decision Function
def determine_lane(state: ResilientEnterpriseState) -> str:
    return state["route_choice"]

# 5. Graph Assembly
builder = StateGraph(ResilientEnterpriseState)
builder.add_node("router", semantic_router_node)
builder.add_node("fast_lane", fast_worker_node)
builder.add_node("deep_lane", deep_worker_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    determine_lane,
    {
        "fast_lane": "fast_lane",
        "deep_lane": "deep_lane"
    }
)
builder.add_edge("fast_lane", END)
builder.add_edge("deep_lane", END)

resilient_graph = builder.compile()
""", language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Multi-Model Routing, Cost, and Fallback Simulator", divider="rainbow")

st.markdown("""
Simulate enterprise multi-model routing in real time. Compare the financial cost, latency,
and resilience of single-model vs. dynamically-routed vs. fallback-cascade topologies.
""")

engine = MultiModelWorkflowEngine()

wb_c1, wb_c2 = st.columns([1, 1])

with wb_c1:
    st.subheader("⚙️ Simulation Controls")
    test_query = st.selectbox(
        "Select Test Ingestion Query:",
        [
            "How do I reset my account password on the customer portal?",
            "Derive the computational complexity of flash attention and refactor this CUDA kernel.",
            "Summarize the key takeaway of this 1-paragraph status email.",
            "Architect a zero-trust multi-region active-active database failover pipeline."
        ]
    )
    routing_mode = st.radio(
        "Select Routing Topology Strategy:",
        [
            "Dynamic Routing by Complexity",
            "Single Model (Always Frontier)",
            "High-Throughput Generalist Only"
        ]
    )
    simulate_outage = st.checkbox(
        "🔥 Simulate Primary Cloud Provider HTTP 504 Outage (Test Fallback Cascade)",
        value=False
    )
    run_sim = st.button("🚀 Run Multi-Model Simulation", type="primary")

with wb_c2:
    st.subheader("📊 Execution & Telemetry Results")
    if run_sim:
        sim_res = engine.simulate_workflow(
            query=test_query,
            simulate_outage=simulate_outage,
            routing_mode=routing_mode
        )
        
        if sim_res["fallback_triggered"]:
            st.error("🚨 Outage Encountered! Fallback cascade successfully recovered execution.")
        else:
            st.success("✅ Workflow Dispatched and Executed Cleanly!")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Latency", f"{sim_res['latency_ms']} ms")
        with m2:
            st.metric("Estimated Cost", f"${sim_res['cost_usd']:.6f}")
        with m3:
            st.metric("Reasoning Index", f"{sim_res['reasoning_index']} / 100")

        st.markdown(f"**Initial Dispatch**: <span class='badge-node'>{sim_res['initial_model']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Final Executing Model**: <span class='badge-route'>{sim_res['final_model']}</span>", unsafe_allow_html=True)
        st.info(f"**Routing Rationale**: {sim_res['routing_reason']}")

        st.markdown("#### Execution Trace Audit")
        for step in sim_res["execution_trace"]:
            st.write(f"- {step}")
    else:
        st.info("Select a test query and routing strategy, then click 'Run Multi-Model Simulation' to view the routing trace.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Answers")

with st.expander("Q1: What are the primary trade-offs between node-level fallbacks (with_fallbacks) vs graph-level fallback edges?", expanded=False):
    st.markdown("""
    **Answer**:
    - **`model.with_fallbacks([backup])` (Node-Level)**: Encapsulated and seamless. Occurs synchronously within a single node invocation. The graph topology remains clean and uncomplicated.
    - **Graph-Level Fallback Edges**: Explicitly models business logic branching in the state diagram. Enables switching entire prompts, modifying system instructions, altering tool lists, or notifying downstream alerting channels when a tier-1 model fails.
    """)

with st.expander("Q2: Why does dynamic model routing dramatically outperform static single-model architectures in production?", expanded=False):
    st.markdown("""
    **Answer**:
    In production applications, approximately 70 to 80 percent of incoming requests are low-to-medium complexity (FAQs, simple extractions, formatting).
    Using a frontier model ($15/1M tokens, 800ms) for trivial requests inflates cloud bills and increases user latency.
    Routing 80 percent of traffic to small models ($0.15/1M tokens, 190ms) achieves parity in benchmark accuracy while cutting overall API costs by up to 85 percent.
    """)

with st.expander("Q3: How does Anthropic prompt caching interact with dynamic model routing in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    Anthropic prompt caching caches static prefixes (system instructions, documents) that exceed 1024 tokens.
    In multi-model graphs, if the router directs requests to Claude, the prompt prefix must remain byte-for-byte identical across turns to achieve cache hits.
    Injecting variable timestamps or dynamic IDs into the system prompt breaks the cache key and incurs full prompt costs on every call.
    """)

with st.expander("Q4: What is the risk of cascading fallbacks across models with differing context window sizes?", expanded=False):
    st.markdown("""
    **Answer**:
    If your primary model has a 2M token context window (e.g. Gemini 1.5 Pro) and the graph accumulates 80k tokens of conversation history,
    cascading to a fallback model with an 8k or 32k context limit (e.g. certain local Ollama models) will trigger a fatal `ContextWindowExceeded` exception.
    Fallbacks must either share comparable context limits or the node must apply `trim_messages()` prior to calling the fallback.
    """)

with st.expander("Q5: How should state channels be designed when different model nodes return incompatible metadata formats?", expanded=False):
    st.markdown("""
    **Answer**:
    Normalize the output schema! All specialist nodes should return standard LangChain `AIMessage` objects for the message channel,
    and map provider-specific telemetry into a normalized TypedDict channel (e.g., `model_metadata: Dict[str, Any]` storing normalized fields like `model_family`, `latency_ms`, and `tokens_used`).
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Infinite Fallback Retry Storms")
    st.markdown("""
    *Mechanism*: Configuring high `max_retries` on both primary and fallback models during upstream cloud outages.  
    *Consequence*: Nodes block for over 60 seconds waiting for timeouts, exhausting thread pools and causing downstream cascading service failures.  
    *Fix*: Set strict short timeouts (`timeout=5.0`, `max_retries=1`) on primary models when fallbacks are configured.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Schema Incompatibility in Fallback Tool Calling")
    st.markdown("""
    *Mechanism*: Falling back from an OpenAI model using JSON schema tools to an open-source model that expects raw prompt-based tool definitions.  
    *Consequence*: The fallback model hallucinates free-form tool calls that fail parser validation.  
    *Fix*: Ensure all fallback targets support LangChain's standardized `bind_tools` protocol.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Over-Complex Classifier Overhead")
    st.markdown("""
    *Mechanism*: Using a slow 500ms LLM classifier to decide whether to route to a 200ms worker model.  
    *Consequence*: The router latency negates any speed benefits of the lightweight worker model.  
    *Fix*: Use regex, fast keyword heuristics, embedding classifiers, or tiny sub-50ms models for routing.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Context Window Truncation on Fallback")
    st.markdown("""
    *Mechanism*: Passing a large conversational state from a 128k cloud model into an 8k local fallback model.  
    *Consequence*: Silent truncation or immediate out-of-memory errors on local GPU instances.  
    *Fix*: Always wrap inputs with `trim_messages(max_tokens=...)` before invoking fallback runnables.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**with_fallbacks()**\\nLangChain method binding ordered list of fallback models that trigger automatically on exceptions.")
with col_fc2:
    st.info("**Model Routing**\\nDirecting graph execution to specialized models based on intent, cost, SLA, or complexity classification.")
with col_fc3:
    st.info("**Heterogeneous Graph**\\nA topology combining models from diverse providers (OpenAI, Claude, Gemini, Ollama) in one pipeline.")
with col_fc4:
    st.info("**Model Tiering**\\nCataloging models into tiers (Triage, Generalist, Frontier, Air-Gapped) to optimize cost-performance balance.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural differences between single-node and multi-node LLM workflows.", value=True)
c2 = st.checkbox("I know how to implement semantic model routing using conditional edges in LangGraph.", value=True)
c3 = st.checkbox("I can configure automatic multi-provider fallback cascades using with_fallbacks().", value=True)
c4 = st.checkbox("I understand how model tiering reduces enterprise API costs by 70 to 85 percent.", value=True)
c5 = st.checkbox("I know how to tailor prompt structures (XML vs JSON strict mode) to provider paradigms.", value=True)
c6 = st.checkbox("I can prevent retry storms and context window mismatches across fallback models.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 10.3 'LLM Workflow Design' with full architectural and practical comprehension!")
