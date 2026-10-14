import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="19.2 Subgraph Design - LangGraph Curriculum",
    page_icon="📐",
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
    .badge-shared {
        background-color: #142E1F;
        color: #34D399;
        border: 1px solid #059669;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-isolated {
        background-color: #2E1065;
        color: #C084FC;
        border: 1px solid #7E22CE;
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
class SubgraphDesignSimulator:
    """
    Simulates subgraph state interaction patterns:
    - Shared state: Direct channel mutation
    - Separate state: Explicit input/output translation mapping
    - Subgraph fan-out composition
    """
    def __init__(self):
        pass

    def simulate_shared_state(self, prompt: str) -> Dict[str, Any]:
        # Both parent and child mutate the same state channels
        return {
            "initial_messages": [prompt],
            "child_intermediate": ["Child added web search results", "Child added synthesis draft"],
            "final_messages": [prompt, "Child added web search results", "Child added synthesis draft", "Parent finalized response"]
        }

    def simulate_separate_state(self, parent_query: str, filters: str) -> Dict[str, Any]:
        # 1. Parent state
        parent_state = {"user_query": parent_query, "filters": filters, "final_report": None}
        
        # 2. Input mapping
        child_input = {"search_string": f"{parent_query} with {filters}", "max_hits": 5}
        
        # 3. Child execution with private state
        child_internal_state = {
            "raw_html": "<payload 42kb>",
            "extracted_urls": ["url1", "url2"],
            "cleaned_summary": f"Targeted analysis on '{parent_query}' under constraints '{filters}'"
        }
        
        # 4. Output mapping
        parent_output = {
            "user_query": parent_query,
            "filters": filters,
            "final_report": child_internal_state["cleaned_summary"]
        }
        
        return {
            "parent_initial": parent_state,
            "child_input_mapped": child_input,
            "child_private_data": child_internal_state,
            "parent_final": parent_output
        }


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Subgraph State Topologies, Schema Binding, and Composition Patterns", divider="rainbow")

st.markdown("""
When designing subgraphs in LangGraph, the most critical architectural decision is **how state is shared between the parent and child graphs**.
There are two primary paradigms:
1. **Shared State Topology**: Parent and child share the exact same state class (or common keys). Updates flow directly through shared channels.
2. **Separate (Isolated) State Topology**: Parent and child define completely independent state schemas. An explicit mapping function acts as an anti-corruption layer, translating parent keys into child inputs and extracting child results.

The blueprint below contrasts these two architectural patterns.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph SubgraphDesignTopology {
        bgcolor="#0B1120";
        rankdir=TB;
        nodesep=0.4;
        ranksep=0.45;

        node [shape=box, style="filled,rounded", fontname="Segoe UI, Roboto, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, Roboto, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1", penwidth=1.3];

        subgraph cluster_shared_pattern {
            label="Pattern A: Shared State Topology (Direct Channel Pass-Through)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            shared_parent [label="Parent Graph\\nState: SharedAgentState\\nChannels: [messages, context]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            shared_child [label="Child Graph\\nState: SharedAgentState\\nDirect Node Registration", fillcolor="#142E1F", fontcolor="#34D399", color="#059669"];
            shared_parent -> shared_child [label="builder.add_node('child', child_app)\\nZero Transformation Needed", color="#34D399"];
        }

        subgraph cluster_isolated_pattern {
            label="Pattern B: Separate State Topology (Encapsulated Schema Mapping)";
            color="#334155";
            fillcolor="#1E293B";
            style="filled,rounded";
            iso_parent [label="Parent Graph\\nState: EnterpriseOrderState\\nKeys: [order_id, user_email]", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
            input_adapter [label="Input Adapter Function\\nTransforms order_id -> SKU query", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            iso_child [label="Child Subgraph\\nState: InventoryState (Private)\\nKeys: [sku_list, stock_count, warehouse]", fillcolor="#2E1065", fontcolor="#C084FC", color="#A855F7"];
            output_adapter [label="Output Adapter Function\\nExtracts status -> order_status", fillcolor="#713F12", fontcolor="#FEF08A", color="#EAB308"];
            iso_parent_final [label="Parent State Updated\\nZero Warehouse Data Leakage", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];

            iso_parent -> input_adapter [label="Parent Calls Wrapper"];
            input_adapter -> iso_child [label="child_app.invoke()"];
            iso_child -> output_adapter [label="Child Returns Result"];
            output_adapter -> iso_parent_final [label="Updates Enterprise State"];
        }
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: SHARED STATE
# ==============================================================================
st.header("2. Shared State: Common Schema Inheritance and Direct Channel Pass-Through", divider="rainbow")

st.markdown("""
In the **Shared State** pattern, the parent and child graphs utilize the identical TypedDict schema or share a common subset of channel keys.
Because channel keys match, you can add the compiled child graph **directly as a node** to the parent builder without writing a wrapper function!
""")

st.code('''from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Both parent and child share MessagesState
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    shared_flag: bool

# Child definition
child_builder = StateGraph(SharedState)
child_builder.add_node("child_reasoner", child_logic)
child_builder.add_edge(START, "child_reasoner")
child_builder.add_edge("child_reasoner", END)
child_app = child_builder.compile()

# Parent directly registers child graph as a node!
parent_builder = StateGraph(SharedState)
parent_builder.add_node("agent_child", child_app) # Direct pass-through!
parent_builder.add_edge(START, "agent_child")
parent_builder.add_edge("agent_child", END)''', language="python")

st.markdown("""
**When to Use Shared State**:
- Conversational chat agents where all agents simply read and append messages to a common history channel.
- Lightweight pipelines where parent and child share the same domain context.
""")

# ==============================================================================
# SECTION 3: SEPARATE STATE
# ==============================================================================
st.header("3. Separate State: Isolated Schemas and Encapsulated Domain Boundaries", divider="rainbow")

st.markdown("""
In the **Separate State** pattern, the child graph defines its own private schema completely decoupled from the parent.
This is the recommended standard for enterprise systems:
""")

st.markdown("""
- **Information Hiding**: The child's internal scratchpad variables (e.g., raw HTML, retry counters, debug logs) never leak into parent state.
- **Independent Testing**: The child can be unit-tested using a standalone pytest suite with mock inputs without needing parent context.
- **Contract Stability**: Changes to the child's internal state schema do not break the parent graph as long as the public interface is maintained.
""")

# ==============================================================================
# SECTION 4: INPUT/OUTPUT MAPPING
# ==============================================================================
st.header("4. Input/Output Mapping: Transform Functions and Channel Adapters", divider="rainbow")

st.markdown("""
When state schemas are separate, the parent graph cannot directly add the child graph as a node.
Instead, you write an **adapter node** (wrapper function) that:
1. Extracts required fields from the parent state.
2. Formats and validates the input dictionary for the child.
3. Invokes the child graph.
4. Extracts only the final required fields from the child output and returns them to the parent.
""")

st.code('''# Adapter Node Pattern
def call_inventory_subgraph_node(parent_state: OrderState) -> dict:
    # 1. Input Mapping: Translate parent order to child inventory request
    child_input = {
        "items": [item["sku"] for item in parent_state["cart"]],
        "postal_code": parent_state["shipping_address"]["zip"]
    }
    
    # 2. Invoke Subgraph
    child_output = inventory_graph.invoke(child_input)
    
    # 3. Output Mapping: Translate child result back into parent channel
    return {
        "is_available": child_output["all_in_stock"],
        "estimated_delivery_days": child_output["max_lead_time_days"]
    }''', language="python")

# ==============================================================================
# SECTION 5: SUBGRAPH INVOCATION
# ==============================================================================
st.header("5. Subgraph Invocation: Direct Node Compilation vs Wrapper Node Execution", divider="rainbow")

st.markdown("""
There are two distinct ways to invoke subgraphs in LangGraph:
""")

col_inv1, col_inv2 = st.columns(2)
with col_inv1:
    st.markdown("#### Approach 1: Direct Node Registration")
    st.code('''# Direct Registration
builder.add_node("specialist", compiled_subgraph)''', language="python")
    st.markdown("""
    - **Pros**: Clean code, automatic checkpoint namespace propagation.
    - **Cons**: Requires shared channel keys; no opportunity to filter or transform data.
    """)

with col_inv2:
    st.markdown("#### Approach 2: Wrapper Node Function")
    st.code('''# Wrapper Function
def wrapper_node(state: ParentState, config: RunnableConfig) -> dict:
    child_res = compiled_subgraph.invoke(
        transform(state),
        config=config # Forward config for tracing!
    )
    return extract(child_res)

builder.add_node("specialist", wrapper_node)''', language="python")
    st.markdown("""
    - **Pros**: Total encapsulation, arbitrary state mapping, error handling.
    - **Cons**: Slightly more boilerplate code.
    """)

# ==============================================================================
# SECTION 6: SUBGRAPH COMPOSITION
# ==============================================================================
st.header("6. Subgraph Composition: Parallel Subgraph Fan-Out, Cascades, and Pipelines", divider="rainbow")

st.markdown("""
Subgraphs can be composed into powerful structural patterns:
1. **Parallel Fan-Out (Map-Reduce)**: The parent broadcasts tasks to multiple specialized subgraphs concurrently (e.g., Code Reviewer + Security Auditor + Documentation Writer).
2. **Sequential Cascades**: Subgraph A finishes and passes its output to Subgraph B.
3. **Supervisor-Worker Hub**: A centralized supervisor dynamically selects which subgraph to invoke next based on user intent.
""")

with st.expander("🔍 View Parallel Subgraph Composition Diagram", expanded=False):
    comp_dot = """
    digraph CompositionDiagram {
        bgcolor="#0B1120";
        rankdir=LR;
        node [shape=box, style="filled,rounded", fontname="Segoe UI, sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="Segoe UI, sans-serif", fontsize=9, color="#94A3B8", fontcolor="#CBD5E1"];

        supervisor [label="Supervisor Node", fillcolor="#0F172A", fontcolor="#F8FAFC", color="#38BDF8"];
        sub_legal [label="Legal Subgraph\\n(Contract Checks)", fillcolor="#1E3A8A", fontcolor="#BFDBFE", color="#60A5FA"];
        sub_finance [label="Finance Subgraph\\n(Budget & ROI)", fillcolor="#312E81", fontcolor="#E0E7FF", color="#818CF8"];
        sub_tech [label="Technical Subgraph\\n(Architecture Review)", fillcolor="#431407", fontcolor="#FED7AA", color="#F97316"];
        consolidate [label="Executive Decision\\nConsolidator", fillcolor="#064E3B", fontcolor="#A7F3D0", color="#34D399"];

        supervisor -> sub_legal;
        supervisor -> sub_finance;
        supervisor -> sub_tech;
        
        sub_legal -> consolidate;
        sub_finance -> consolidate;
        sub_tech -> consolidate;
    }
    """
    st.graphviz_chart(comp_dot, use_container_width=True)

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Complete Multi-Subgraph Architecture with Schema Mapping", divider="rainbow")

st.markdown("""
Below is a complete, copy-pasteable production pattern demonstrating clean schema separation and input/output mapping adapters.
""")

st.code('''from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Specialized SQL Subgraph with Private State
class SqlSubgraphState(TypedDict):
    raw_query: str
    db_schema: str
    generated_sql: str
    row_count: int

def generate_sql_node(state: SqlSubgraphState) -> dict:
    return {"generated_sql": f"SELECT * FROM sales WHERE desc LIKE '%{state['raw_query']}%'"}

def execute_query_node(state: SqlSubgraphState) -> dict:
    return {"row_count": 42}

sql_builder = StateGraph(SqlSubgraphState)
sql_builder.add_node("generate_sql", generate_sql_node)
sql_builder.add_node("execute_query", execute_query_node)
sql_builder.add_edge(START, "generate_sql")
sql_builder.add_edge("generate_sql", "execute_query")
sql_builder.add_edge("execute_query", END)
sql_subgraph = sql_builder.compile()

# 2. Parent Enterprise Graph
class ParentEnterpriseState(TypedDict):
    user_request: str
    data_points_found: int
    executive_summary: str

def sql_adapter_node(state: ParentEnterpriseState) -> dict:
    # Input mapping
    child_input = {"raw_query": state["user_request"], "db_schema": "sales_v2"}
    # Invoke child
    child_res = sql_subgraph.invoke(child_input)
    # Output mapping
    return {"data_points_found": child_res["row_count"]}

def summarize_node(state: ParentEnterpriseState) -> dict:
    return {
        "executive_summary": f"Query '{state['user_request']}' returned {state['data_points_found']} records."
    }

parent_builder = StateGraph(ParentEnterpriseState)
parent_builder.add_node("sql_adapter", sql_adapter_node)
parent_builder.add_node("summarize", summarize_node)
parent_builder.add_edge(START, "sql_adapter")
parent_builder.add_edge("sql_adapter", "summarize")
parent_builder.add_edge("summarize", END)

parent_app = parent_builder.compile(checkpointer=MemorySaver())

result = parent_app.invoke(
    {"user_request": "Q3 Enterprise License Sales"},
    config={"configurable": {"thread_id": "subgraph_mapping_01"}}
)
print("Result:", result["executive_summary"])''', language="python")

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: State Topology Simulator and Schema Mapping Laboratory", divider="rainbow")

st.markdown("""
Experiment with how state boundaries and adapter functions isolate data between parent and child graphs.
""")

sim = SubgraphDesignSimulator()

col_sim1, col_sim2 = st.columns(2)

with col_sim1:
    st.subheader("🧪 Simulation 1: Shared vs Separate State")
    sim_mode = st.radio("Select Design Paradigm", ["Shared State (Direct Channels)", "Separate State (Adapter Mapping)"])
    query_in = st.text_input("User Prompt", value="Find top 3 cloud security vendors")
    filter_in = st.text_input("Category Filter", value="FedRAMP Certified")

    if st.button("🚀 Run Architecture Simulation"):
        if sim_mode.startswith("Shared"):
            res = sim.simulate_shared_state(query_in)
            st.session_state["design_res"] = {"mode": "shared", "data": res}
        else:
            res = sim.simulate_separate_state(query_in, filter_in)
            st.session_state["design_res"] = {"mode": "separate", "data": res}

with col_sim2:
    st.subheader("📋 State Channel Audit")
    if "design_res" in st.session_state:
        d = st.session_state["design_res"]
        if d["mode"] == "shared":
            st.info("Direct Shared State: Both graphs appended to the messages channel directly.")
            st.json(d["data"])
        else:
            st.success("Clean Encapsulation: Input and output adapters isolated private scratchpad data!")
            st.markdown("#### 1. Input Mapped into Child:")
            st.json(d["data"]["child_input_mapped"])
            st.markdown("#### 2. Child Private Data (Hidden from Parent):")
            st.json(d["data"]["child_private_data"])
            st.markdown("#### 3. Output Mapped Back to Parent:")
            st.json(d["data"]["parent_final"])

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("9. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Top 5 Technical Interview Questions & Answers")

with st.expander("Q1: What are the advantages of using separate state schemas for subgraphs?", expanded=False):
    st.markdown("""
    **Answer**:
    Separate state schemas enforce strict encapsulation.
    Internal scratchpad variables and raw payloads are confined to the child graph, preventing parent state pollution.
    It also allows child graphs to be independently tested and reused across different parent applications.
    """)

with st.expander("Q2: When is direct node registration of a subgraph preferred over a wrapper function?", expanded=False):
    st.markdown("""
    **Answer**:
    Direct registration (`builder.add_node('sub', child_app)`) is preferred when parent and child share a standard schema (e.g. `MessagesState`) and no data transformation or key mapping is required.
    """)

with st.expander("Q3: How do you pass execution configuration (like thread_id or callbacks) into a child subgraph?", expanded=False):
    st.markdown("""
    **Answer**:
    In a wrapper node function, include the `config: RunnableConfig` parameter in the node signature and pass it directly to the child call:
    `child_graph.invoke(child_input, config=config)`.
    """)

with st.expander("Q4: Can a child subgraph invoke another child subgraph (multi-tier nesting)?", expanded=False):
    st.markdown("""
    **Answer**:
    Yes! Because compiled subgraphs are standard Runnables, nesting can continue to arbitrary depths, creating multi-tiered organizational agent hierarchies.
    """)

with st.expander("Q5: What happens if a child subgraph throws an unhandled exception inside a wrapper node?", expanded=False):
    st.markdown("""
    **Answer**:
    Unless caught in a `try/except` block inside the wrapper node, the exception bubbles up and aborts the parent graph's active superstep, preserving the last valid checkpoint in the parent thread.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Forgetting to Forward RunnableConfig to Child Subgraphs")
    st.markdown("""
    *Mechanism*: Calling `child_graph.invoke(sub_input)` inside a wrapper node without passing `config=config`.  
    *Consequence*: Child execution loses tracing context in LangSmith and cannot access parent checkpointer configurations.  
    *Fix*: Always pass `child_graph.invoke(sub_input, config=config)`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Returning Full Child State Dictionaries into Parent State")
    st.markdown("""
    *Mechanism*: Writing `return child_output` where `child_output` contains keys not recognized by the parent TypedDict.  
    *Consequence*: Raises schema validation errors or pollutes the parent state channel with unintended keys.  
    *Fix*: Explicitly map only the required fields into a clean output dictionary.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Architectural Warning): Overusing Shared State Leading to Accidental Overwrites")
    st.markdown("""
    *Mechanism*: Parent and child sharing a generic key like `status` or `data` without custom channel reducers.  
    *Consequence*: The child's internal steps overwrite the parent's value unexpectedly.  
    *Fix*: Use domain-prefixed keys or isolated state schemas.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Latency Warning): Synchronous Blocking Fan-Out Across Multiple Subgraphs")
    st.markdown("""
    *Mechanism*: Sequentially invoking 3 subgraphs in a single node function using blocking `.invoke()`.  
    *Consequence*: Total execution time equals the sum of all three subgraphs instead of the slowest one.  
    *Fix*: Use LangGraph parallel nodes or the Send API for concurrent subgraph execution.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Shared State**\\nTopology where parent and child share common channel keys for direct pass-through.")
with col_fc2:
    st.info("**Separate State**\\nTopology where parent and child define distinct schemas decoupled by adapters.")
with col_fc3:
    st.info("**Adapter Node**\\nA wrapper function that translates state schemas across parent and child boundaries.")
with col_fc4:
    st.info("**RunnableConfig**\\nThe execution context that must be forwarded to child subgraphs for tracing.")

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I know when to use shared state vs separate state schemas for subgraphs.", value=True)
c2 = st.checkbox("I know how to write an adapter node function for input and output mapping.", value=True)
c3 = st.checkbox("I understand the difference between direct node registration and wrapper functions.", value=True)
c4 = st.checkbox("I know why forwarding config to child_graph.invoke() is mandatory for LangSmith tracing.", value=True)
c5 = st.checkbox("I can design a parallel subgraph composition using LangGraph nodes.", value=True)
c6 = st.checkbox("I understand how separate state schemas prevent state pollution in large projects.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 19.2 'Subgraph Design' with full architectural and practical comprehension!")
