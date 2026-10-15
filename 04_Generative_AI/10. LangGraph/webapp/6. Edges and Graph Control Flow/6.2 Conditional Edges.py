import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="6.2 Conditional Edges - LangGraph Curriculum",
    page_icon="🔀",
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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE CONDITIONAL ROUTING SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class ConditionalRoutingSimulator:
    """
    Simulates dynamic path routing, router predicate evaluation,
    conditional entry points, and path mapping resolution.
    """
    def __init__(self):
        self.routes = {
            "technical_specialist": "Handles code generation, API schemas, and architecture queries.",
            "billing_agent": "Handles refunds, invoices, subscription tiers, and payment receipts.",
            "general_faq": "Handles standard knowledge questions and company policies.",
            "escalate_human": "Triggers human-in-the-loop review for ambiguous or sensitive intents."
        }

    def evaluate_route(self, query: str, sentiment: str, confidence: float) -> Tuple[str, str, Dict[str, Any]]:
        """
        Predicate function evaluating routing logic.
        Returns: (selected_route_key, explanation, route_diagnostics)
        """
        q = query.lower()
        if confidence < 0.60:
            return "escalate_human", "Low classification confidence (< 0.60). Escalating to human supervisor.", {"confidence": confidence}

        if any(w in q for w in ["refund", "invoice", "charge", "credit", "subscription", "pricing"]):
            return "billing_agent", "Detected billing keywords and payment intent.", {"matched_keywords": ["billing", "refund"]}
        elif any(w in q for w in ["code", "python", "api", "bug", "error", "stacktrace", "function"]):
            return "technical_specialist", "Detected software engineering and technical vocabulary.", {"matched_keywords": ["code", "api"]}
        else:
            return "general_faq", "Standard inquiry. Routing to general knowledge base.", {"matched_keywords": ["faq"]}


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Dynamic Routing and Conditional Edge Topology", divider="rainbow")

st.markdown("""
In LangGraph, **Conditional Edges** allow graphs to branch dynamically based on the current state.
Instead of hardcoding transitions, a conditional edge invokes a **router function** that inspects state
and returns a branch key or destination node name, enabling adaptive workflows, triage agents, and self-correction loops.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    conditional_blueprint_dot = """
    digraph ConditionalRoutingArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress {
            label="1. Ingress & Classification";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START", fillcolor="#1E293B", shape=ellipse];
            Node_Classifier [label="Intent Classifier Node\\nEvaluates user prompt & intent", fillcolor="#0284C7"];
        }

        subgraph cluster_decision_router {
            label="2. Conditional Edge Router (Predicate)";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Router_Predicate [label="route_intent(state)\\nReturns: 'billing' | 'tech' | 'faq' | 'human'", shape=diamond, fillcolor="#4338CA"];
        }

        subgraph cluster_specialist_branches {
            label="3. Targeted Execution Branches";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Branch_Tech [label="Technical Specialist Node\\nExecutes Python sandbox", fillcolor="#0D9488"];
            Branch_Billing [label="Billing Agent Node\\nQueries Stripe / NetSuite", fillcolor="#0D9488"];
            Branch_FAQ [label="General FAQ Node\\nRAG over knowledge base", fillcolor="#0D9488"];
            Branch_Human [label="Human Review Node\\nTriggers interrupt() breakpoint", fillcolor="#D97706"];
        }

        subgraph cluster_egress {
            label="4. Termination";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Node_Classifier;
        Node_Classifier -> Router_Predicate [label="Evaluate state"];
        Router_Predicate -> Branch_Tech [label="'tech' path"];
        Router_Predicate -> Branch_Billing [label="'billing' path"];
        Router_Predicate -> Branch_FAQ [label="'faq' path"];
        Router_Predicate -> Branch_Human [label="'human' (conf < 0.6)"];
        Branch_Tech -> END;
        Branch_Billing -> END;
        Branch_FAQ -> END;
        Branch_Human -> END;
    }
    """
    st.graphviz_chart(conditional_blueprint_dot, use_container_width=True)

st.markdown("""
### Syntax of Conditional Edges
Conditional edges are registered with `builder.add_conditional_edges(source, path_function, path_map)`:
- `source`: The originating node name.
- `path_function`: A callable taking `state -> str | List[str]`.
- `path_map`: (Optional) A dictionary mapping return strings to destination node names.
""")

# ==============================================================================
# SECTION 2: CONDITIONAL ROUTING
# ==============================================================================
st.header("2. Conditional Routing: Dynamic Path Determination Based on State", divider="rainbow")

st.markdown("""
Unlike static pipelines where every node executes on every turn, **Conditional Routing** selects
only the relevant execution path for a given user request.
""")

col_cr1, col_cr2 = st.columns(2)
with col_cr1:
    st.markdown("""
    #### Why Conditional Routing is Essential:
    - **Token Efficiency**: Avoid calling heavyweight reasoning models when a simple retrieval node suffices.
    - **Safety & Guardrails**: Divert prompt injections or hazardous queries to containment nodes.
    - **Autonomous Cycles**: Evaluate if generated output is satisfactory or requires self-correction.
    """)
with col_cr2:
    st.markdown("""
    #### Common Conditional Scenarios:
    - Intent triage (Customer Support vs Billing vs Technical)
    - Tool call vs Final Answer (Does the LLM require external tools?)
    - Quality threshold check (Passes evaluation vs Regenerate)
    - Escalation to human supervisor
    """)

# ==============================================================================
# SECTION 3: ROUTER FUNCTIONS
# ==============================================================================
st.header("3. Router Functions: Predicate Logic and Path Mapping Dictionaries", divider="rainbow")

st.markdown("""
A router function is a pure Python callable that inspects state and returns a path key.
Using a **Path Mapping Dictionary** makes the relationship explicit and self-documenting.
""")

code_router_demo = """def route_after_evaluation(state: AgentState) -> str:
    # 1. Inspect state variables
    score = state.get("quality_score", 0.0)
    retry_count = state.get("retry_count", 0)
    
    # 2. Decision predicate
    if score >= 0.85:
        return "approved"
    elif retry_count < 3:
        return "retry"
    else:
        return "escalate"

# Path Mapping Dictionary: maps return value -> destination node!
path_map = {
    "approved": "finalize_answer",
    "retry": "regenerate_draft",
    "escalate": "human_operator"
}

builder.add_conditional_edges("evaluate_draft", route_after_evaluation, path_map)
"""
st.code(code_router_demo, language="python")

# ==============================================================================
# SECTION 4: BRANCHING WORKFLOWS
# ==============================================================================
st.header("4. Branching Workflows: Fan-Out and Selective Path Execution", divider="rainbow")

st.markdown("""
Branching workflows divide incoming requests into distinct specialized domains.
LangGraph guarantees that only the selected branch is activated; other branches consume zero compute resources.
""")

st.markdown("""
#### Prebuilt Routing: Tools Condition
LangGraph provides a prebuilt conditional edge `tools_condition` designed specifically for tool-calling agents:
```python
from langgraph.prebuilt import tools_condition

# Routes to "tools" if the last message contains tool_calls, else routes to END!
builder.add_conditional_edges("agent", tools_condition)
```
""")

# ==============================================================================
# SECTION 5: DYNAMIC ROUTING
# ==============================================================================
st.header("5. Dynamic Routing: Runtime Context and LLM-Guided Decisions", divider="rainbow")

st.markdown("""
In advanced workflows, routing decisions are not based on rigid if/else statements, but on structured LLM outputs.
""")

code_dynamic_llm_route = """from pydantic import BaseModel, Field
from typing import Literal

class RouterDecision(BaseModel):
    destination: Literal["sql_expert", "web_search", "direct_response"]
    reasoning: str = Field(description="Why this destination was chosen")

def llm_guided_router(state: AgentState) -> str:
    # Ask LLM to produce structured decision
    structured_llm = llm.with_structured_output(RouterDecision)
    decision = structured_llm.invoke(state["messages"])
    return decision.destination
"""
st.code(code_dynamic_llm_route, language="python")

# ==============================================================================
# SECTION 6: CONDITIONAL ENTRY POINTS
# ==============================================================================
st.header("6. Conditional Entry Points: Routing Directly from START", divider="rainbow")

st.markdown("""
You can attach a conditional edge directly to `START`!
This allows routing incoming traffic to completely different entry pipelines based on initial payload parameters
(e.g. user tier, authorization token, or input modality).
""")

code_start_routing = """def route_entry_point(state: IngressState) -> str:
    if state.get("is_authenticated", False):
        return "dashboard_agent"
    return "login_prompt"

# Directly route from START!
builder.add_conditional_edges(
    START,
    route_entry_point,
    {"dashboard_agent": "dashboard_agent", "login_prompt": "login_prompt"}
)
"""
st.code(code_start_routing, language="python")

# ==============================================================================
# SECTION 7: MULTIPLE BRANCHES
# ==============================================================================
st.header("7. Multiple Branches: Multi-Path Dispatch and Parallel Conditional Routing", divider="rainbow")

st.markdown("""
Did you know a router function can return a **list of strings**?
When a router returns `["branch_a", "branch_b"]`, LangGraph dispatches **both** destination nodes in parallel!
""")

col_mb1, col_mb2 = st.columns(2)
with col_mb1:
    st.markdown("""
    #### Dynamic Multi-Path Dispatch:
    - **Single Path**: Return a single string `"node_name"`.
    - **Parallel Multi-Path**: Return a list `["node_a", "node_b"]`.
    - **Skip Execution**: Return an empty list `[]` to terminate the path.
    """)
with col_mb2:
    st.code("""def multi_worker_router(state: ComplexState) -> list[str]:
    tasks = []
    if state["needs_code_review"]:
        tasks.append("code_reviewer")
    if state["needs_security_scan"]:
        tasks.append("security_scanner")
    return tasks # Dispatches active tasks in parallel!
""", language="python")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Multi-Branch Conditional Routing Engine", divider="rainbow")

st.markdown("""
Below is a complete, runnable script demonstrating conditional routing from `START`,
triage branching, and fallback recovery.
""")

prod_conditional_code = '''from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: State Contract
class TriageState(TypedDict):
    query: str
    category: Optional[str]
    confidence: float
    response: Optional[str]

# Node Functions
def triage_classifier_node(state: TriageState) -> Dict[str, Any]:
    q = state["query"].lower()
    if "billing" in q or "invoice" in q:
        cat, conf = "billing", 0.95
    elif "api" in q or "code" in q:
        cat, conf = "technical", 0.92
    else:
        cat, conf = "general", 0.70
    return {"category": cat, "confidence": conf}

def billing_agent_node(state: TriageState) -> Dict[str, Any]:
    return {"response": "Billing Agent: Invoices and subscription details loaded."}

def technical_agent_node(state: TriageState) -> Dict[str, Any]:
    return {"response": "Technical Specialist: API schema and documentation loaded."}

def general_faq_node(state: TriageState) -> Dict[str, Any]:
    return {"response": "General FAQ: Support policies and help center links."}

def human_escalation_node(state: TriageState) -> Dict[str, Any]:
    return {"response": "Supervisor: Ambiguous request flagged for manual agent review."}

# Router Function
def route_by_category(state: TriageState) -> str:
    if state["confidence"] < 0.75:
        return "escalate"
    cat = state.get("category", "general")
    if cat == "billing":
        return "billing"
    elif cat == "technical":
        return "technical"
    return "general"

# Step 2: Build Graph
def create_conditional_graph():
    builder = StateGraph(TriageState)
    
    builder.add_node("classifier", triage_classifier_node)
    builder.add_node("billing_agent", billing_agent_node)
    builder.add_node("technical_agent", technical_agent_node)
    builder.add_node("general_faq", general_faq_node)
    builder.add_node("human_escalation", human_escalation_node)
    
    # Static edge into classifier
    builder.add_edge(START, "classifier")
    
    # Conditional edge out of classifier
    builder.add_conditional_edges(
        "classifier",
        route_by_category,
        {
            "billing": "billing_agent",
            "technical": "technical_agent",
            "general": "general_faq",
            "escalate": "human_escalation"
        }
    )
    
    builder.add_edge("billing_agent", END)
    builder.add_edge("technical_agent", END)
    builder.add_edge("general_faq", END)
    builder.add_edge("human_escalation", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = create_conditional_graph()
    test_q = {"query": "How do I fix the API authentication error?", "category": None, "confidence": 0.0, "response": None}
    res = app.invoke(test_q)
    print("Resolved Category:", res["category"])
    print("Agent Response:", res["response"])
'''
st.code(prod_conditional_code, language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Conditional Branch Tester and Dynamic Router Simulator", divider="rainbow")

st.markdown("""
Test how conditional router functions evaluate state inputs. Enter custom query texts,
adjust classification confidence, and observe the dynamically selected path.
""")

router_sim = ConditionalRoutingSimulator()

col_cw_l, col_cw_r = st.columns([1, 1.2])

with col_cw_l:
    st.subheader("Simulate Ingress Query")
    query_input = st.text_input("Customer Query", value="I need an updated invoice for last month's subscription")
    sentiment_input = st.selectbox("Customer Sentiment", ["Neutral", "Frustrated", "Urgent"])
    confidence_input = st.slider("Classification Confidence Score", 0.0, 1.0, 0.90)

    btn_eval = st.button("🔀 Evaluate Conditional Router", use_container_width=True)

with col_cw_r:
    st.subheader("Router Evaluation Results")
    route_key, reasoning, diag = router_sim.evaluate_route(query_input, sentiment_input, confidence_input)
    
    st.metric("Selected Route Target", route_key.upper())
    st.info(f"**Rationale**: {reasoning}")
    st.markdown("**Route Description:**")
    st.write(router_sim.routes.get(route_key, "N/A"))
    st.markdown("**Router Diagnostics:**")
    st.json(diag)

st.markdown("### Registered Branch Destinations")
df_routes = pd.DataFrame([
    {"Branch Key": k, "Description": v} for k, v in router_sim.routes.items()
])
st.dataframe(df_routes, use_container_width=True)

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: What happens if a conditional router function returns a string not present in the path_map dictionary?", expanded=False):
    st.markdown("""
    **Answer**:
    If a `path_map` dictionary is provided (e.g. `{"approved": "node_a", "rejected": "node_b"}`) and the router callable returns `"unknown"`, LangGraph raises a runtime `KeyError: 'unknown'`.
    To prevent this, ensure your router callable handles unexpected cases with a fallback default or omit `path_map` if your router returns destination node names directly.
    """)

with st.expander("Q2: Can a conditional edge function be asynchronous (async def)?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. In LangGraph, router functions can be asynchronous `async def route(state) -> str`.
    This is especially useful if the routing decision itself requires calling an LLM, querying an external feature store, or verifying an authentication API before picking the path.
    """)

with st.expander("Q3: How do you achieve dynamic multi-branch parallel execution using conditional edges?", expanded=False):
    st.markdown("""
    **Answer**:
    Have the router callable return a `List[str]` of keys rather than a single string:
    ```python
    def parallel_router(state) -> List[str]:
        return ["worker_a", "worker_b"]
    ```
    LangGraph treats the list as a parallel fan-out, dispatching both destination nodes in the same superstep.
    """)

with st.expander("Q4: What is the prebuilt tools_condition function in LangGraph?", expanded=False):
    st.markdown("""
    **Answer**:
    `tools_condition` is a built-in router function from `langgraph.prebuilt`.
    It inspects the last message in `state["messages"]`. If that message has `tool_calls` populated, it returns `"tools"`; otherwise, it returns `END`. It is the standard conditional edge for tool-calling agents.
    """)

with st.expander("Q5: How does returning Command(goto=...) in modern LangGraph replace conditional edges?", expanded=False):
    st.markdown("""
    **Answer**:
    In modern LangGraph (v0.2+), nodes can return `Command(update={...}, goto="target_node")`.
    This co-locates the state update and the routing decision inside the node itself, eliminating the need for a separate `add_conditional_edges` call for many simple branching decisions.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Unhandled Return Value in Router Predicates")
    st.markdown("""
    *Mechanism*: Using `if / elif` in a router without a final `else:` fallback, causing the function to return `None`.  
    *Consequence*: Crash with `KeyError: None` during runtime routing.  
    *Fix*: Always provide a defensive fallback: `return "default_node"`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Infinite Loops Caused by Flaky Exit Predicates")
    st.markdown("""
    *Mechanism*: Routing back to a generator node until `quality >= 0.9`, but the LLM never hits 0.9.  
    *Consequence*: Graph exceeds `recursion_limit` (25 steps) and crashes with `GraphRecursionError`.  
    *Fix*: Always combine quality thresholds with a loop guard: `if score >= 0.9 or loop_count >= 3: return END`.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Expensive LLM Invocations Inside Router Functions")
    st.markdown("""
    *Mechanism*: Calling a 70B parameter model inside the router function for every trivial transition.  
    *Consequence*: Doubles latency and cost on every branch.  
    *Fix*: Use fast deterministic regex/heuristics where possible, or use lightweight specialized 8B router models.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Hardcoding Node Strings Without Path Mapping")
    st.markdown("""
    *Mechanism*: Returning `"dest_node_name"` directly without a `path_map` dictionary in complex enterprise graphs.  
    *Consequence*: Renaming a destination node breaks the router silently without compile-time static validation.  
    *Fix*: Use explicit `path_map = {"category": "node_name"}` to maintain a clear decoupled contract.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Conditional Edge**\\nA graph transition evaluated dynamically at runtime via a router predicate function.")
with col_fc2:
    st.info("**path_map**\\nA dictionary mapping router string return values to destination node identifiers.")
with col_fc3:
    st.info("**tools_condition**\\nA prebuilt LangGraph router that checks for tool calls on the last message.")
with col_fc4:
    st.info("**Multi-Branch Routing**\\nA conditional edge returning a list of strings to dispatch parallel destination nodes.")

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know how to register conditional edges using add_conditional_edges().", value=True)
c2 = st.checkbox("I understand the role of path_map in mapping router keys to node names.", value=True)
c3 = st.checkbox("I know how to route directly from START to create dynamic entry pipelines.", value=True)
c4 = st.checkbox("I can implement multi-branch parallel dispatch by returning a list of strings from a router.", value=True)
c5 = st.checkbox("I know how to prevent GraphRecursionError by combining quality checks with loop guards.", value=True)
c6 = st.checkbox("I understand when to use prebuilt tools_condition versus custom router predicates.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 6.2 'Conditional Edges' with full architectural and practical comprehension!")
