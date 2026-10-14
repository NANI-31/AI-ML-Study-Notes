import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="5.1 Node Basics - LangGraph Curriculum",
    page_icon="⚙️",
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
# MODULE-SCOPE NODE EXECUTION SIMULATOR FOR WORKBENCH & PIPELINE
# ==============================================================================
class NodeBasicsEngine:
    """
    Simulates node invocation mechanics: callable resolution, state inspection,
    config injection, delta calculation, and channel reducer updates.
    """
    def __init__(self):
        self.registered_nodes = {
            "ingest_query": {
                "desc": "Validates and normalizes incoming query string",
                "reads": ["query"],
                "writes": ["normalized_query", "audit_log"]
            },
            "extract_keywords": {
                "desc": "Parses domain entities and keywords",
                "reads": ["normalized_query"],
                "writes": ["keywords", "audit_log"]
            },
            "evaluate_urgency": {
                "desc": "Assigns numeric urgency score and operational priority",
                "reads": ["normalized_query", "keywords"],
                "writes": ["urgency_score", "audit_log"]
            }
        }

    def execute_node(self, node_name: str, current_state: Dict[str, Any], config_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes a simulated node function cleanly without side-effects.
        Returns: (delta_emitted, updated_state)
        """
        delta = {}
        log_entry = f"Node [{node_name}] executed with caller: {config_meta.get('user_id', 'anon')}"

        if node_name == "ingest_query":
            raw_q = current_state.get("query", "")
            norm_q = raw_q.strip().lower()
            delta = {
                "normalized_query": norm_q,
                "audit_log": [log_entry + f" -> Normalized to '{norm_q}'"]
            }
        elif node_name == "extract_keywords":
            norm_q = current_state.get("normalized_query", current_state.get("query", "").lower())
            words = [w for w in norm_q.replace("?", "").replace("!", "").split() if len(w) > 3]
            delta = {
                "keywords": words,
                "audit_log": [log_entry + f" -> Extracted {len(words)} keywords: {words}"]
            }
        elif node_name == "evaluate_urgency":
            norm_q = current_state.get("normalized_query", "")
            is_urgent = any(k in norm_q for k in ["urgent", "error", "emergency", "fail", "critical"])
            score = 0.95 if is_urgent else 0.25
            delta = {
                "urgency_score": score,
                "audit_log": [log_entry + f" -> Urgency score calculated: {score}"]
            }

        # Apply updates
        new_state = dict(current_state)
        for k, v in delta.items():
            if k == "audit_log":
                new_state["audit_log"] = current_state.get("audit_log", []) + v
            else:
                new_state[k] = v

        return delta, new_state


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The Node Runtime Pipeline and Execution Model", divider="rainbow")

st.markdown("""
In LangGraph, a **Node** is a standard Python callable (synchronous function, async coroutine, or LangChain `Runnable`)
that represents an atomic unit of computation. Nodes receive the frozen graph state snapshot at superstep `t`,
execute domain or AI logic, and return a **partial state update dictionary** or an orchestration `Command`.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    node_blueprint_dot = """
    digraph NodeArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_dispatch {
            label="1. Superstep Dispatcher";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            
            State_Snapshot [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" bgcolor="#0F172A">
                <tr><td colspan="2" bgcolor="#1E293B"><b>Frozen State S_t</b></td></tr>
                <tr><td>messages</td><td>[HumanMessage('Run pipeline')]</td></tr>
                <tr><td>iteration</td><td>1</td></tr>
            </table>>, shape=plaintext];
            
            Runtime_Config [label="RunnableConfig\\n{'configurable': {'user_id': 'u101', 'thread_id': 't9'}}", fillcolor="#334155"];
        }

        subgraph cluster_node_runtime {
            label="2. Node Computational Boundary";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            
            Node_Function [label="Node Callable: my_node_fn(state, config)\\n1. Reads state attributes (Defensive read)\\n2. Calls LLM, Tool, or Business Logic\\n3. Constructs delta dictionary", fillcolor="#0284C7"];
            
            Local_Computation [label="Local Computation (Isolated)\\nNo mutation of input S_t!\\nPure functional or external API call", fillcolor="#1E293B"];
        }

        subgraph cluster_egress_merge {
            label="3. Barrier Egress and Reducer Merge";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            
            Delta_Output [label="Partial State Delta\\n{'messages': [AIMessage('Done')], 'iteration': 2}", fillcolor="#059669"];
            
            State_Consolidator [label="StateGraph Reducer Engine\\nMerges delta into S_{t+1}\\nTriggers Checkpointer Persist", fillcolor="#1E3A8A"];
        }

        State_Snapshot -> Node_Function [label="Injected as 1st arg: state"];
        Runtime_Config -> Node_Function [label="Injected as 2nd arg: config (optional)"];
        Node_Function -> Local_Computation [label="Executes logic"];
        Local_Computation -> Delta_Output [label="Emits dictionary"];
        Delta_Output -> State_Consolidator [label="Barrier synchronization"];
    }
    """
    st.graphviz_chart(node_blueprint_dot, use_container_width=True)

st.markdown("""
### Core Rule of Node Execution
Nodes are isolated workers. A node cannot see updates made by sibling nodes running in the same superstep.
Every node evaluates strictly against the state snapshot passed into it at the start of the turn.
""")

# ==============================================================================
# SECTION 2: WHAT IS A NODE?
# ==============================================================================
st.header("2. What is a Node? Discrete Compute Units in LangGraph", divider="rainbow")

st.markdown("""
A node in LangGraph represents a single vertex in the directed execution graph.
Unlike traditional LangChain components where chains wrap entire end-to-end applications,
LangGraph encourages splitting your AI workflows into **fine-grained, specialized nodes**.
""")

col_n1, col_n2 = st.columns(2)
with col_n1:
    st.markdown("""
    #### Characteristics of a LangGraph Node
    - **Single Responsibility**: Each node should perform one logical task (e.g. classify intent, invoke a specific tool, format prompt).
    - **Any Python Callable**: Can be a normal function `def f(state)`, an async coroutine `async def f(state)`, or a runnable object.
    - **Zero Graph Coupling**: A node function does not need to know what node preceded it or what node follows it. Routing is handled by graph edges.
    """)
with col_n2:
    st.markdown("""
    #### What Can Be a Node?
    - Standard Python Functions
    - Async Functions (`async def`)
    - Python Classes with `__call__`
    - LangChain `Runnable` instances (`prompt | model`)
    - Pre-compiled Subgraphs (`CompiledStateGraph`)
    """)

# ==============================================================================
# SECTION 3: CREATING NODE FUNCTIONS
# ==============================================================================
st.header("3. Creating Node Functions: Standard Signatures and Callable Contracts", divider="rainbow")

st.markdown("""
LangGraph accepts two primary functional signatures for node callables:
""")

col_sig1, col_sig2 = st.columns(2)
with col_sig1:
    st.markdown("""
    #### 1. State-Only Signature (Most Common)
    ```python
    def analyze_node(state: MyState) -> dict:
        query = state["query"]
        # compute...
        return {"analysis": "complete"}
    ```
    """)
with col_sig2:
    st.markdown("""
    #### 2. State + Config Signature
    ```python
    from langchain_core.runnables import RunnableConfig

    def user_aware_node(state: MyState, config: RunnableConfig) -> dict:
        thread_id = config.get("configurable", {}).get("thread_id")
        return {"processed_by_thread": thread_id}
    ```
    """)

code_register_demo = """from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class PipelineState(TypedDict):
    input_text: str
    output_text: str

# 1. Define Callable
def format_text_node(state: PipelineState) -> dict:
    return {"output_text": state["input_text"].strip().capitalize()}

# 2. Register to Graph
builder = StateGraph(PipelineState)
builder.add_node("format_text", format_text_node) # "format_text" is the node key!
"""
st.code(code_register_demo, language="python")

# ==============================================================================
# SECTION 4: NODE INPUTS
# ==============================================================================
st.header("4. Node Inputs: State Snapshots and Runtime Configuration", divider="rainbow")

st.markdown("""
When LangGraph calls a node, it injects the execution state and optional context into the arguments.
""")

st.markdown("""
#### The Two Core Injected Arguments:
1. **`state: StateType`**: A shallow-copied dictionary matching your graph schema.
   - All top-level keys declared in your schema are available.
   - If an optional key was never written, `.get("key", default)` prevents `KeyError`.
2. **`config: RunnableConfig` (Optional)**: Contains metadata injected by the caller during `.invoke(inputs, config=...)`:
   - `config["configurable"]["thread_id"]`: Active persistence thread identifier.
   - `config["configurable"]["user_id"]`: Multi-tenant user identity.
   - `config["callbacks"]`: LangSmith tracing and observability handlers.
""")

# ==============================================================================
# SECTION 5: NODE OUTPUTS
# ==============================================================================
st.header("5. Node Outputs: Partial Dictionaries and Orchestration Commands", divider="rainbow")

st.markdown("""
What can a node function return? LangGraph supports two main return types:
""")

col_o1, col_o2 = st.columns(2)
with col_o1:
    st.markdown("""
    #### 1. Partial State Dictionary (`dict`)
    The most frequent return type. Contains only the fields you wish to mutate or append.
    ```python
    def update_status(state: AgentState) -> dict:
        return {"status": "SUCCESS", "step_count": state["step_count"] + 1}
    ```
    An empty dictionary `{}` signals zero state modifications.
    """)
with col_o2:
    st.markdown("""
    #### 2. Orchestration Command (`Command`)
    Introduced in LangGraph v0.2+, `Command` combines state updates with dynamic edge routing:
    ```python
    from langgraph.types import Command

    def router_worker(state: AgentState) -> Command:
        return Command(
            update={"score": 0.95},
            goto="approved_node"
        )
    ```
    """)

# ==============================================================================
# SECTION 6: STATE ACCESS
# ==============================================================================
st.header("6. State Access: Defensive Reads, Typing, and Channel Unpacking", divider="rainbow")

st.markdown("""
Writing defensive, robust node code requires disciplined state access patterns.
Never assume every optional channel is present or populated with non-None data.
""")

code_defensive = """from typing import TypedDict, Optional, List

class ResilientState(TypedDict):
    query: str
    tokens: Optional[int]
    tags: Optional[List[str]]

def production_node(state: ResilientState) -> dict:
    # Pattern 1: Mandatory key access
    query = state["query"]
    
    # Pattern 2: Defensive optional access with sensible defaults
    tokens = state.get("tokens") or 0
    tags = state.get("tags") or []
    
    # Pattern 3: Type guard
    if not isinstance(tags, list):
        tags = [str(tags)]
        
    return {"tokens": tokens + 15, "tags": tags + ["processed"]}
"""
st.code(code_defensive, language="python")

# ==============================================================================
# SECTION 7: STATE UPDATES
# ==============================================================================
st.header("7. State Updates: Reducer Triggers and Atomic Delta Emissions", divider="rainbow")

st.markdown("""
When a node returns a dictionary, LangGraph matches each returned key against the schema channels:
- If the channel is an **Overwrite channel**, LangGraph replaces the previous value with the emitted value.
- If the channel is an **Annotated Reducer channel**, LangGraph invokes `reducer(existing_value, emitted_value)`.
""")

# Contextual diagram for update routing
update_flow_dot = """
digraph StateUpdateFlow {
    graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.4];
    node [shape=box, style="filled,rounded", fontname="Arial", fontsize=10, fontcolor="#F8FAFC", margin="0.15,0.08"];
    edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

    Node_Return [label="Node returns: {'counter': 5, 'log': ['Step completed']}", fillcolor="#1E293B"];
    Inspect [label="LangGraph Channel Router\\nInspects schema definitions", fillcolor="#0284C7"];
    Branch_Counter [label="counter channel: Overwrite\\nold_val = 4 -> new_val = 5", fillcolor="#334155"];
    Branch_Log [label="log channel: Annotated[list, operator.add]\\n['Initial'] + ['Step completed'] = 2 items", fillcolor="#334155"];
    Merged_Snapshot [label="Consolidated S_{t+1} State Snapshot", fillcolor="#059669"];

    Node_Return -> Inspect;
    Inspect -> Branch_Counter [label="Direct Overwrite"];
    Inspect -> Branch_Log [label="Invoke Reducer"];
    Branch_Counter -> Merged_Snapshot;
    Branch_Log -> Merged_Snapshot;
}
"""
with st.expander("🔀 View State Update Channel Dispatch Flow", expanded=True):
    st.graphviz_chart(update_flow_dot, use_container_width=True)

# ==============================================================================
# SECTION 8: NODE NAMING
# ==============================================================================
st.header("8. Node Naming: Semantic Identity and Namespace Best Practices", divider="rainbow")

st.markdown("""
When registering nodes with `builder.add_node("name", fn)`, the name string becomes the **node identifier** in:
- Graphviz blueprints and visualization diagrams
- Edge routing definitions (`builder.add_edge("name", "next")`)
- Checkpointer history traces and LangSmith spans
""")

st.markdown("""
#### Recommended Node Naming Rules:
1. **Verb-Noun Format**: Use clear action phrases (e.g. `classify_intent`, `retrieve_documents`, `generate_critique`).
2. **Snake Case**: Keep names lowercase with underscores (`summarize_findings`, not `SummarizeFindings` or `summarize-findings`).
3. **Avoid Generic Labels**: Do not name nodes `node_1`, `step`, or `process`. Generic names obscure debugging in complex multi-agent graphs.
4. **Reserved Names**: Never name a node `__start__` or `__end__`, as these are internal identifiers used by LangGraph for `START` and `END`.
""")

# ==============================================================================
# SECTION 9: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("9. Production Implementation: Fully Typed Production Node Pipeline", divider="rainbow")

st.markdown("""
Below is a complete, runnable script illustrating best-practice node construction,
defensive state reading, config access, and typed delta returns.
""")

prod_node_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# Step 1: Define Strongly Typed State
class DocumentProcessingState(TypedDict):
    document_title: str
    raw_content: str
    cleaned_content: Optional[str]
    word_count: int
    processing_logs: Annotated[List[str], operator.add]
    status: str

# Step 2: Implement Single-Responsibility Nodes
def sanitize_node(state: DocumentProcessingState, config: RunnableConfig) -> Dict[str, Any]:
    user = config.get("configurable", {}).get("user_id", "system")
    raw = state["raw_content"]
    cleaned = " ".join(raw.split()) # Strip redundant whitespace
    
    return {
        "cleaned_content": cleaned,
        "processing_logs": [f"Sanitized by user '{user}': length {len(cleaned)} chars"],
        "status": "SANITIZED"
    }

def analyze_metrics_node(state: DocumentProcessingState) -> Dict[str, Any]:
    cleaned = state.get("cleaned_content") or state["raw_content"]
    words = len(cleaned.split())
    
    return {
        "word_count": words,
        "processing_logs": [f"Calculated word count: {words} words"],
        "status": "ANALYZED"
    }

def finalize_node(state: DocumentProcessingState) -> Dict[str, Any]:
    return {
        "status": "COMPLETED",
        "processing_logs": ["Document processing successfully completed."]
    }

# Step 3: Build and Compile Graph
def build_node_pipeline():
    builder = StateGraph(DocumentProcessingState)
    
    builder.add_node("sanitize", sanitize_node)
    builder.add_node("analyze_metrics", analyze_metrics_node)
    builder.add_node("finalize", finalize_node)
    
    builder.add_edge(START, "sanitize")
    builder.add_edge("sanitize", "analyze_metrics")
    builder.add_edge("analyze_metrics", "finalize")
    builder.add_edge("finalize", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = build_node_pipeline()
    init_state = {
        "document_title": "AI Roadmap 2026",
        "raw_content": "  LangGraph enables    resilient multi-agent state machines!  ",
        "cleaned_content": None,
        "word_count": 0,
        "processing_logs": ["Ingestion initiated."],
        "status": "INITIALIZED"
    }
    config = {"configurable": {"user_id": "eng_lead_01"}}
    res = app.invoke(init_state, config=config)
    print("Final Status:", res["status"])
    print("Cleaned:", res["cleaned_content"])
    print("Logs:", res["processing_logs"])
'''
st.code(prod_node_code, language="python")

# ==============================================================================
# SECTION 10: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("10. Interactive Workbench: Node Invocation and State Delta Inspector", divider="rainbow")

st.markdown("""
Test node mechanics in real-time. Configure a query, inject runtime config parameters,
select a node function to execute, and inspect the emitted delta versus the consolidated state.
""")

engine = NodeBasicsEngine()

if "node_state" not in st.session_state:
    st.session_state.node_state = {
        "query": "Critical emergency: Database replica failure detected!",
        "normalized_query": "",
        "keywords": [],
        "urgency_score": 0.0,
        "audit_log": ["Pipeline state initialized."]
    }

col_ctrl, col_view = st.columns([1, 1.2])

with col_ctrl:
    st.subheader("Node Dispatch Controller")
    selected_node = st.selectbox("Select Node to Execute", list(engine.registered_nodes.keys()))
    node_meta = engine.registered_nodes[selected_node]
    st.caption(f"**Description**: {node_meta['desc']}")
    st.caption(f"**Reads**: `{node_meta['reads']}` | **Writes**: `{node_meta['writes']}`")

    cfg_user = st.text_input("Config 'user_id'", value="admin_worker_42")
    cfg_thread = st.text_input("Config 'thread_id'", value="thread_tx_99")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⚡ Dispatch Node Callable", use_container_width=True):
            config = {"user_id": cfg_user, "thread_id": cfg_thread}
            delta, updated = engine.execute_node(selected_node, st.session_state.node_state, config)
            st.session_state.node_state = updated
            st.session_state.last_delta = delta
            st.success(f"Executed node '{selected_node}'!")
            
    with col_b2:
        if st.button("🔄 Reset Node State", use_container_width=True):
            st.session_state.node_state = {
                "query": "Critical emergency: Database replica failure detected!",
                "normalized_query": "",
                "keywords": [],
                "urgency_score": 0.0,
                "audit_log": ["Pipeline state initialized."]
            }
            st.session_state.last_delta = {}
            st.info("State reset to initial values.")

with col_view:
    st.subheader("Execution Delta & Consolidated Snapshot")
    if "last_delta" in st.session_state and st.session_state.last_delta:
        st.markdown("**Last Emitted Partial Delta:**")
        st.json(st.session_state.last_delta)
    else:
        st.info("No node dispatched yet. Click 'Dispatch Node Callable' to run.")

    st.markdown("**Active Consolidated State:**")
    st.json(st.session_state.node_state)

st.markdown("### Audit Log History (Channel Reducer Append)")
for idx, entry in enumerate(st.session_state.node_state.get("audit_log", [])):
    st.markdown(f"- `{idx}`: {entry}")

# ==============================================================================
# SECTION 11: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("11. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: Can a node in LangGraph return values that are not declared in the state schema?", expanded=False):
    st.markdown("""
    **Answer**:
    By default, if you use a `TypedDict` or Pydantic model for your state schema, returning undeclared keys will either:
    1. Be ignored or stripped if strict schema validation is enabled.
    2. Raise an `InvalidUpdateError` if the graph compiler cannot find a matching channel for the returned key.
    Always declare all channels in your schema to maintain predictable state management.
    """)

with st.expander("Q2: How does LangGraph pass runtime arguments to nodes when using RunnableConfig?", expanded=False):
    st.markdown("""
    **Answer**:
    LangGraph inspects the function signature of each node callable via Python's `inspect.signature`.
    If the function accepts a second argument (e.g. `def my_node(state, config)`), LangGraph automatically injects the active `RunnableConfig` containing tracing tags, metadata, and configurable parameters. If the node only accepts one argument (`def my_node(state)`), LangGraph passes only `state`.
    """)

with st.expander("Q3: What happens if two sibling nodes in a parallel superstep return conflicting updates for the same overwrite channel?", expanded=False):
    st.markdown("""
    **Answer**:
    If two parallel nodes write to an unannotated (overwrite) channel in the same superstep, LangGraph detects an ambiguous concurrent write and raises an `InvalidUpdateError`. To resolve this, you must either:
    1. Make the channel an append reducer (`Annotated[list, operator.add]`).
    2. Provide a custom binary resolution reducer function.
    3. Route the nodes sequentially instead of in parallel.
    """)

with st.expander("Q4: Can a pre-compiled LangGraph subgraph be registered as a node inside another parent graph?", expanded=False):
    st.markdown("""
    **Answer**:
    **Yes**. A compiled subgraph (`CompiledStateGraph`) implements the standard `Runnable` protocol. You can register it directly:
    ```python
    parent_builder.add_node("sub_agent", child_graph)
    ```
    LangGraph treats the entire subgraph as an atomic node, invoking it and merging its outputs back into the parent graph state.
    """)

with st.expander("Q5: What is the recommended way to handle external API failures inside a node function?", expanded=False):
    st.markdown("""
    **Answer**:
    Wrap the external API call in a `try/except` block and emit an explicit error state:
    ```python
    except Exception as e:
        return {"has_error": True, "error_message": str(e), "retry_count": state["retry_count"] + 1}
    ```
    Then, use a conditional edge in the graph to route to a retry node, fallback LLM, or human intervention node.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Mutating Input State Instead of Returning a Delta")
    st.markdown("""
    *Mechanism*: Executing `state['status'] = 'DONE'` inside a node function instead of returning `{'status': 'DONE'}`.  
    *Consequence*: Bypasses channel reducers, creates dirty reads in parallel branches, and corrupts checkpointer state.  
    *Fix*: Always treat `state` as read-only and return a new dictionary of updates.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Naming Collision with Reserved Identifiers")
    st.markdown("""
    *Mechanism*: Registering a node with `builder.add_node('__start__', my_fn)` or `builder.add_node('__end__', my_fn)`.  
    *Consequence*: Overwrites internal virtual entry/exit markers, resulting in broken compilation or infinite loops.  
    *Fix*: Use semantic domain names like `start_pipeline` or `entry_router` instead of reserved strings.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Synchronous Blocking I/O Inside Async Nodes")
    st.markdown("""
    *Mechanism*: Calling `requests.get()` or `time.sleep()` inside an `async def` node callable.  
    *Consequence*: Blocks the asyncio event loop, starving parallel coroutines and spiking API latency.  
    *Fix*: Use `httpx.AsyncClient` or `asyncio.sleep()` inside async node callables.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Monolithic God Nodes Violating Single Responsibility")
    st.markdown("""
    *Mechanism*: Writing a 400-line node function that performs ingestion, validation, vector search, LLM calls, and formatting in one step.  
    *Consequence*: Destroys graph modularity, eliminates granular retry capabilities, and makes checkpoint inspection useless.  
    *Fix*: Decompose into discrete single-purpose nodes connected by clear graph edges.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**Node Callable**\\nA pure or side-effect function accepting `(state, config)` and emitting a partial state update.")
with col_fc2:
    st.info("**Partial Delta**\\nA subset dictionary containing only modified keys merged into state at superstep boundaries.")
with col_fc3:
    st.info("**RunnableConfig**\\nExecution context containing thread ID, user metadata, and callbacks passed to nodes.")
with col_fc4:
    st.info("**Command**\\nUnified return object combining state updates with explicit dynamic graph routing.")

# ==============================================================================
# SECTION 12: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("12. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the function signature requirements for LangGraph nodes.", value=True)
c2 = st.checkbox("I know how to access runtime thread and user configuration via RunnableConfig.", value=True)
c3 = st.checkbox("I understand that nodes must return partial delta dictionaries rather than mutating state.", value=True)
c4 = st.checkbox("I can differentiate between overwrite channels and append reducer channels.", value=True)
c5 = st.checkbox("I adhere to standard semantic naming conventions (verb-noun) for all registered nodes.", value=True)
c6 = st.checkbox("I know how to embed subgraphs as modular atomic nodes within a parent graph.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 5.1 'Node Basics' with full architectural and practical comprehension!")
