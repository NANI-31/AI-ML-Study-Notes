import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="5.2 Node Types - LangGraph Curriculum",
    page_icon="🧩",
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
# MODULE-SCOPE NODE TAXONOMY DISPATCHER FOR WORKBENCH & PIPELINE
# ==============================================================================
class NodeTaxonomyEngine:
    """
    Simulates execution behaviors across 8 distinct LangGraph node archetypes:
    LLM, Tool, Data Processing, Validation, Router, Retrieval, Human Approval, Summarization.
    """
    def __init__(self):
        self.node_catalog = {
            "LLM Node": {"role": "Language generation and reasoning", "io": "messages -> AIMessage"},
            "Tool Node": {"role": "External API / Python environment execution", "io": "tool_calls -> ToolMessage"},
            "Data-Processing Node": {"role": "Deterministic transformations", "io": "raw_data -> cleaned_data"},
            "Validation Node": {"role": "Schema & safety guardrails", "io": "payload -> is_valid flag"},
            "Router Node": {"role": "Conditional branch path calculation", "io": "state -> branch_key string"},
            "Retrieval Node": {"role": "VectorDB / BM25 document query", "io": "query -> context_docs list"},
            "Human Approval Node": {"role": "Breakpoint interrupt & human review", "io": "action -> is_approved bool"},
            "Summarization Node": {"role": "Context window compression", "io": "messages -> summary string"}
        }

    def execute_node_type(self, node_type: str, state: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the simulated node type logic and produces a partial delta.
        """
        delta = {}
        if node_type == "LLM Node":
            prompt = state.get("query", "Default query")
            delta = {
                "messages": [f"AIMessage(content='Reasoned response for: {prompt}')"],
                "last_active_node": "llm_node"
            }
        elif node_type == "Tool Node":
            tool_name = params.get("tool_name", "web_search")
            delta = {
                "tool_results": [{ "tool": tool_name, "result": "Found 3 verified records." }],
                "last_active_node": "tool_node"
            }
        elif node_type == "Data-Processing Node":
            raw = state.get("raw_text", "   messy input text   ")
            delta = {
                "cleaned_text": raw.strip().upper(),
                "last_active_node": "data_processing_node"
            }
        elif node_type == "Validation Node":
            confidence = params.get("confidence", 0.85)
            is_valid = confidence >= 0.70
            delta = {
                "is_valid": is_valid,
                "validation_error": None if is_valid else "Confidence below threshold",
                "last_active_node": "validation_node"
            }
        elif node_type == "Router Node":
            intent = params.get("intent", "technical")
            route_target = "specialist_agent" if intent == "technical" else "general_faq"
            delta = {
                "next_route": route_target,
                "last_active_node": "router_node"
            }
        elif node_type == "Retrieval Node":
            q = state.get("query", "LangGraph")
            delta = {
                "retrieved_docs": [f"Doc 1: Best practices for {q}", f"Doc 2: Architecture for {q}"],
                "last_active_node": "retrieval_node"
            }
        elif node_type == "Human Approval Node":
            approved = params.get("human_decision", True)
            delta = {
                "human_approved": approved,
                "status": "APPROVED" if approved else "REJECTED_BY_HUMAN",
                "last_active_node": "human_approval_node"
            }
        elif node_type == "Summarization Node":
            msg_count = len(state.get("messages", []))
            delta = {
                "summary": f"Compressed {msg_count} prior messages into 1 memory token buffer.",
                "last_active_node": "summarization_node"
            }
        return delta


# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Taxonomy of Specialized Node Types in AI Agent Workflows", divider="rainbow")

st.markdown("""
In real-world enterprise agent architectures, graphs are composed of specialized node archetypes.
Each archetype fulfills a precise role: cognitive generation (**LLM**), grounded actuation (**Tool**),
data hygiene (**Data-Processing**), safety checking (**Validation**), topological branching (**Router**),
external knowledge injection (**Retrieval**), governance (**Human Approval**), and memory preservation (**Summarization**).
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    taxonomy_blueprint_dot = """
    digraph NodeTaxonomy {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.5, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8", penwidth=1.5];

        subgraph cluster_ingress_nodes {
            label="1. Ingestion & Retrieval Layer";
            style=dashed;
            color="#334155";
            fontcolor="#94A3B8";
            START [label="START", fillcolor="#1E293B", shape=ellipse];
            Node_Retrieval [label="Retrieval Node\\nFetches VectorDB & BM25 chunks", fillcolor="#0284C7"];
            Node_DataProc [label="Data-Processing Node\\nCleans & normalizes context", fillcolor="#0284C7"];
        }

        subgraph cluster_reasoning_nodes {
            label="2. Cognitive & Actuation Core";
            style="filled";
            color="#1E293B";
            fontcolor="#38BDF8";
            Node_LLM [label="LLM Node\\nGenerates reasoning or tool calls", fillcolor="#4338CA"];
            Node_Router [label="Router Node\\nEvaluates intent & branch logic", fillcolor="#6D28D9"];
            Node_Tool [label="Tool Node (Actuator)\\nExecutes Python/APIs (ToolNode)", fillcolor="#0D9488"];
        }

        subgraph cluster_governance_nodes {
            label="3. Governance, Validation & Memory Layer";
            style="filled";
            color="#1E293B";
            fontcolor="#10B981";
            Node_Human [label="Human Approval Node\\nTriggers interrupt() breakpoint", fillcolor="#D97706"];
            Node_Validation [label="Validation Node\\nEvaluates schema & PII safety", fillcolor="#059669"];
            Node_Summarize [label="Summarization Node\\nPrunes token history before END", fillcolor="#059669"];
            END [label="END", fillcolor="#1E293B", shape=doublecircle];
        }

        START -> Node_Retrieval;
        Node_Retrieval -> Node_DataProc;
        Node_DataProc -> Node_LLM;
        Node_LLM -> Node_Router;
        Node_Router -> Node_Tool [label="Has Tool Calls"];
        Node_Tool -> Node_LLM [label="Tool Result Loop"];
        Node_Router -> Node_Human [label="High Risk Action"];
        Node_Human -> Node_Validation [label="Approved"];
        Node_Validation -> Node_Summarize [label="Safe"];
        Node_Summarize -> END;
    }
    """
    st.graphviz_chart(taxonomy_blueprint_dot, use_container_width=True)

st.markdown("""
### Modern Agent Design: The Separation of Cognition and Actuation
LangGraph formalizes a clean boundary between the **reasoner** (LLM Node) and the **actuator** (Tool Node).
The LLM node only decides *what* tool to call; the `ToolNode` executes the tool call safely in the environment.
""")

# ==============================================================================
# SECTION 2: LLM NODE
# ==============================================================================
st.header("2. LLM Node: Prompt Ingestion, Model Invocation, and Message Emission", divider="rainbow")

st.markdown("""
An **LLM Node** encapsulates calls to language models (e.g. ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI).
It typically formats messages from graph state into a prompt, invokes the model, and appends the resulting `AIMessage` to state.
""")

code_llm_node = """from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]

llm = ChatOpenAI(model="gpt-4o")

def call_model_node(state: AgentState) -> dict:
    # 1. Read messages from state
    messages = state["messages"]
    
    # 2. Invoke model
    response = llm.invoke(messages)
    
    # 3. Return delta (AIMessage will be appended by operator.add)
    return {"messages": [response]}
"""
st.code(code_llm_node, language="python")

# ==============================================================================
# SECTION 3: TOOL NODE
# ==============================================================================
st.header("3. Tool Node: Tool Call Execution and Environment Grounding", divider="rainbow")

st.markdown("""
LangGraph provides a prebuilt `ToolNode` from `langgraph.prebuilt` specifically engineered to inspect the last `AIMessage`
in state, extract `tool_calls`, execute each tool in parallel or sequence, and emit `ToolMessage` updates.
""")

code_tool_node = """from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> float:
    \"\"\"Fetches live stock ticker price.\"\"\"
    return 145.20

tools = [get_stock_price]
# Prebuilt ToolNode handles state unpacking, execution, and ToolMessage construction!
tool_node = ToolNode(tools)

# Register into graph
builder.add_node("tools", tool_node)
"""
st.code(code_tool_node, language="python")

# ==============================================================================
# SECTION 4: DATA-PROCESSING NODE
# ==============================================================================
st.header("4. Data-Processing Node: Feature Engineering and Structured Transformation", divider="rainbow")

st.markdown("""
A **Data-Processing Node** executes pure deterministic Python transformations.
It contains zero LLM calls and zero external API requests. Common uses include:
- Parsing raw JSON strings into structured dictionaries
- Normalizing timestamps, currency values, or document text
- Filtering arrays or aggregating summary statistics
""")

code_dataproc = """def parse_tabular_data_node(state: Dict[str, Any]) -> dict:
    raw_csv = state["raw_csv_payload"]
    lines = [line.strip().split(",") for line in raw_csv.strip().split("\\n")]
    headers, rows = lines[0], lines[1:]
    
    parsed_records = [dict(zip(headers, r)) for r in rows]
    return {
        "structured_records": parsed_records,
        "record_count": len(parsed_records)
    }
"""
st.code(code_dataproc, language="python")

# ==============================================================================
# SECTION 5: VALIDATION NODE
# ==============================================================================
st.header("5. Validation Node: Guardrails, Schema Assertions, and Output Parsing", divider="rainbow")

st.markdown("""
A **Validation Node** acts as a quality gatekeeper. It evaluates the output of previous nodes against business invariants,
safety policies, or Pydantic schemas. If validation passes, execution continues; otherwise, it flags an error or loops back.
""")

code_val_node = """from pydantic import BaseModel, Field, ValidationError

class OutputContract(BaseModel):
    confidence: float = Field(ge=0.7)
    answer: str = Field(min_length=10)

def validate_agent_output_node(state: Dict[str, Any]) -> dict:
    candidate = state.get("draft_answer", {})
    try:
        validated = OutputContract.model_validate(candidate)
        return {"is_valid": True, "error_message": None}
    except ValidationError as err:
        return {
            "is_valid": False,
            "error_message": str(err),
            "retry_count": state.get("retry_count", 0) + 1
        }
"""
st.code(code_val_node, language="python")

# ==============================================================================
# SECTION 6: ROUTER NODE
# ==============================================================================
st.header("6. Router Node: Dynamic Intent Branching and Conditional Evaluation", divider="rainbow")

st.markdown("""
A **Router Node** inspects state and writes a routing decision into state (or acts as the predicate function in conditional edges).
Routers decide whether to invoke a specialist agent, escalate to a human, or terminate at `END`.
""")

code_router = """def route_intent_node(state: Dict[str, Any]) -> dict:
    last_msg = state["messages"][-1].content.lower()
    
    if "refund" in last_msg or "billing" in last_msg:
        route = "billing_agent"
    elif "api" in last_msg or "code" in last_msg:
        route = "technical_agent"
    else:
        route = "general_support"
        
    return {"assigned_specialist": route}
"""
st.code(code_router, language="python")

# ==============================================================================
# SECTION 7: RETRIEVAL NODE
# ==============================================================================
st.header("7. Retrieval Node: VectorDB Queries and Context Augmentation", divider="rainbow")

st.markdown("""
A **Retrieval Node** connects the graph to vector databases (Chroma, Pinecone, Qdrant) or search engines (BM25, Tavily).
It takes a query from state, runs similarity search, and injects retrieved context chunks into state.
""")

code_retrieval = """def retrieve_chunks_node(state: Dict[str, Any]) -> dict:
    query = state["user_query"]
    # Simulated vector similarity search
    chunks = [
        {"id": "doc_41", "text": "LangGraph state channels support custom reducers.", "score": 0.92},
        {"id": "doc_88", "text": "ToolNode executes tools and returns ToolMessage.", "score": 0.88}
    ]
    return {
        "retrieved_context": chunks,
        "citation_count": len(chunks)
    }
"""
st.code(code_retrieval, language="python")

# ==============================================================================
# SECTION 8: HUMAN APPROVAL NODE
# ==============================================================================
st.header("8. Human Approval Node: Human-in-the-Loop Interrupts and Approval Workflows", divider="rainbow")

st.markdown("""
A **Human Approval Node** pauses graph execution when an action requires human review (e.g. executing financial transactions,
sending external emails, or executing shell commands). In LangGraph, this is achieved using `interrupt()`.
""")

code_human_node = """from langgraph.types import interrupt

def human_approval_node(state: Dict[str, Any]) -> dict:
    action_details = state["pending_action"]
    
    # 1. Interrupt pauses execution and surfaces prompt to client/UI
    human_response = interrupt({
        "question": "Do you approve this sensitive transaction?",
        "details": action_details
    })
    
    # 2. When graph is resumed with Command(resume=True/False):
    return {
        "is_approved": human_response.get("approved", False),
        "approver_notes": human_response.get("notes", "")
    }
"""
st.code(code_human_node, language="python")

# ==============================================================================
# SECTION 9: SUMMARIZATION NODE
# ==============================================================================
st.header("9. Summarization Node: Context Window Management and Token Compression", divider="rainbow")

st.markdown("""
In long-running conversational graphs, accumulating raw messages leads to context overflow and prohibitive token costs.
A **Summarization Node** periodically compresses older messages into a rolling summary string, pruning the message list.
""")

code_summarize = """from langchain_core.messages import RemoveMessage

def summarize_conversation_node(state: Dict[str, Any]) -> dict:
    messages = state["messages"]
    
    # If messages exceed 10, summarize older messages
    if len(messages) > 10:
        old_msgs = messages[:-4] # Keep 4 most recent messages
        summary = f"Summary of {len(old_msgs)} prior turns..."
        
        # Prune old messages via RemoveMessage reducer
        delete_deltas = [RemoveMessage(id=m.id) for m in old_msgs]
        return {
            "summary": summary,
            "messages": delete_deltas
        }
    return {}
"""
st.code(code_summarize, language="python")

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: Full Multi-Node Architecture with Human Approval and Tool Calls", divider="rainbow")

st.markdown("""
Below is an end-to-end production script demonstrating a multi-node architecture combining
an LLM reasoning node, Tool execution node, and validation quality gate.
""")

prod_multi_node_code = '''import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Step 1: Define Multi-Node State
class EnterpriseWorkflowState(TypedDict):
    user_query: str
    messages: Annotated[List[str], operator.add]
    retrieved_docs: List[str]
    tool_action: Optional[str]
    is_approved: bool
    final_output: Optional[str]

# Node 1: Retrieval Node
def retrieval_node(state: EnterpriseWorkflowState) -> Dict[str, Any]:
    q = state["user_query"]
    docs = [f"Policy Doc: Access requirements for {q}"]
    return {
        "retrieved_docs": docs,
        "messages": [f"Retrieved {len(docs)} knowledge documents."]
    }

# Node 2: Reasoning Node (Simulated LLM)
def reasoning_node(state: EnterpriseWorkflowState) -> Dict[str, Any]:
    return {
        "tool_action": "DATABASE_MIGRATE_V2",
        "messages": ["Reasoning complete: Scheduled database migration."]
    }

# Node 3: Router Node
def should_seek_approval(state: EnterpriseWorkflowState) -> str:
    action = state.get("tool_action")
    if action and "MIGRATE" in action:
        return "human_approval"
    return "execute_tool"

# Node 4: Human Approval Node
def human_approval_node(state: EnterpriseWorkflowState) -> Dict[str, Any]:
    # In production with checkpointer: uses interrupt()
    return {
        "is_approved": True,
        "messages": ["Action reviewed and approved by administrator."]
    }

# Node 5: Tool Execution Node
def tool_execution_node(state: EnterpriseWorkflowState) -> Dict[str, Any]:
    action = state["tool_action"]
    return {
        "final_output": f"Executed action: {action} successfully.",
        "messages": [f"Tool execution complete for {action}"]
    }

# Build Graph
def create_enterprise_graph():
    builder = StateGraph(EnterpriseWorkflowState)
    
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_tool", tool_execution_node)
    
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "reasoning")
    builder.add_conditional_edges(
        "reasoning",
        should_seek_approval,
        {"human_approval": "human_approval", "execute_tool": "execute_tool"}
    )
    builder.add_edge("human_approval", "execute_tool")
    builder.add_edge("execute_tool", END)
    
    return builder.compile()

if __name__ == "__main__":
    app = create_enterprise_graph()
    initial_payload = {
        "user_query": "Apply database migration script",
        "messages": ["Starting pipeline."],
        "retrieved_docs": [],
        "tool_action": None,
        "is_approved": False,
        "final_output": None
    }
    result = app.invoke(initial_payload)
    print("Final Output:", result["final_output"])
    print("Messages Log:", result["messages"])
'''
st.code(prod_multi_node_code, language="python")

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Multi-Node Type Dispatcher and Pipeline Simulator", divider="rainbow")

st.markdown("""
Test the behavior of all 8 node types interactively. Select a node archetype, configure its parameters,
and inspect the delta emitted into state.
""")

taxonomy = NodeTaxonomyEngine()

if "type_state" not in st.session_state:
    st.session_state.type_state = {
        "query": "How to optimize LangGraph state memory?",
        "messages": ["User: How to optimize LangGraph state memory?"],
        "raw_text": "   untrimmed raw context   ",
        "last_active_node": "START"
    }

col_tw_l, col_tw_r = st.columns([1, 1.2])

with col_tw_l:
    st.subheader("Select Node Archetype")
    chosen_type = st.selectbox("Node Type Archetype", list(taxonomy.node_catalog.keys()))
    st.caption(f"**Role**: {taxonomy.node_catalog[chosen_type]['role']}")
    st.caption(f"**I/O Signature**: `{taxonomy.node_catalog[chosen_type]['io']}`")

    # Dynamic parameter controls based on node type
    params = {}
    if chosen_type == "Tool Node":
        params["tool_name"] = st.selectbox("Tool to Execute", ["calculator", "weather_api", "sql_executor"])
    elif chosen_type == "Validation Node":
        params["confidence"] = st.slider("Quality Confidence Score", 0.0, 1.0, 0.85)
    elif chosen_type == "Router Node":
        params["intent"] = st.radio("User Intent Category", ["technical", "billing", "general"])
    elif chosen_type == "Human Approval Node":
        params["human_decision"] = st.checkbox("Human Approves Action?", value=True)

    if st.button("🚀 Invoke Selected Node Type", use_container_width=True):
        delta = taxonomy.execute_node_type(chosen_type, st.session_state.type_state, params)
        st.session_state.last_type_delta = delta
        
        # Apply to state
        for k, v in delta.items():
            if k == "messages":
                st.session_state.type_state["messages"] = st.session_state.type_state.get("messages", []) + v
            else:
                st.session_state.type_state[k] = v
        st.success(f"Executed archetype '{chosen_type}' successfully!")

with col_tw_r:
    st.subheader("Simulated Node Delta & Resulting State")
    if "last_type_delta" in st.session_state and st.session_state.last_type_delta:
        st.markdown("**Emitted Partial Delta:**")
        st.json(st.session_state.last_type_delta)
    else:
        st.info("Select an archetype and click Invoke to observe emitted deltas.")

    st.markdown("**Active Consolidated State:**")
    st.json(st.session_state.type_state)

st.markdown("### Node Archetype Reference Matrix")
df_catalog = pd.DataFrame([
    {"Archetype": k, **v} for k, v in taxonomy.node_catalog.items()
])
st.dataframe(df_catalog, use_container_width=True)

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREP & PRODUCTION TRAPS
# ==============================================================================
st.header("12. Technical Interview Prep & Production Traps", divider="rainbow")

st.subheader("🎯 Technical Interview Questions & Architectural Solutions")

with st.expander("Q1: How does LangGraph's prebuilt ToolNode handle concurrent tool execution?", expanded=False):
    st.markdown("""
    **Answer**:
    When an LLM emits multiple `tool_calls` in a single `AIMessage`, the prebuilt `ToolNode`:
    1. Inspects the `tool_calls` list on the last message.
    2. Dispatches tool executions concurrently using `asyncio.gather` (in async graphs) or thread pools.
    3. Packages the returned outputs into individual `ToolMessage` instances keyed by `tool_call_id`.
    4. Emits `{"messages": [ToolMessage, ToolMessage, ...]}` which are appended into state via `add_messages`.
    """)

with st.expander("Q2: What is the architectural role of interrupt() in Human Approval nodes?", expanded=False):
    st.markdown("""
    **Answer**:
    In LangGraph, `interrupt(value)`:
    1. Halts execution of the node immediately.
    2. Triggers the checkpointer to save the graph state snapshot at the current superstep.
    3. Surfaces `value` back to the client/caller.
    When the human provides feedback via `app.invoke(Command(resume=decision), config=...)`, LangGraph restores state from the checkpoint and resumes execution right where it was paused.
    """)

with st.expander("Q3: Why should Data-Processing nodes never make external network calls?", expanded=False):
    st.markdown("""
    **Answer**:
    Data-processing nodes should remain **pure deterministic functions**.
    Mixing external network calls (HTTP requests, database reads) into data-processing nodes destroys testability, makes replay debugging non-deterministic, and eliminates the ability to retry flaky network calls independently of data parsing logic. External interactions belong in dedicated Tool or Retrieval nodes.
    """)

with st.expander("Q4: How does a Summarization node prevent unbounded token growth in conversational graphs?", expanded=False):
    st.markdown("""
    **Answer**:
    A Summarization node monitors the length of the `messages` list. When it exceeds a threshold (e.g. 10 messages), the node:
    1. Passes older messages to an LLM to generate a concise summary.
    2. Emits `RemoveMessage(id=msg.id)` deltas to prune older messages from state.
    3. Updates a persistent `summary: str` channel with the consolidated narrative.
    This keeps the active prompt size constant regardless of session duration.
    """)

with st.expander("Q5: What is the difference between a Router Node and a Conditional Edge?", expanded=False):
    st.markdown("""
    **Answer**:
    - **Router Node**: A registered node in the graph that reads state, computes routing analysis, and writes a routing decision into state (e.g. `{"selected_route": "support"}`).
    - **Conditional Edge**: A graph connection (`add_conditional_edges`) that evaluates a predicate function to decide which downstream node to transition to without mutating state.
    In modern LangGraph with the `Command` API, a node can combine both roles by returning `Command(update={...}, goto="target_node")`.
    """)

st.subheader("⚠️ Production Traps & Critical Failure Modes")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("🚨 Trap 1 (Fatal Error): Missing tool_call_id Matching in Tool Nodes")
    st.markdown("""
    *Mechanism*: Returning custom `ToolMessage` objects where `tool_call_id` does not match the ID emitted by the LLM's `tool_calls`.  
    *Consequence*: OpenAI and Anthropic APIs reject subsequent invocations with 400 Bad Request error.  
    *Fix*: Always use prebuilt `ToolNode` or strictly propagate `tool_call_id=call["id"]`.
    """)

with col_tr2:
    st.error("🚨 Trap 2 (Fatal Error): Forgetting Checkpointer Support for interrupt() Nodes")
    st.markdown("""
    *Mechanism*: Calling `interrupt()` in a graph compiled without a persistence checkpointer (`builder.compile()`).  
    *Consequence*: LangGraph raises a runtime `GraphInterruptError` or crashes because there is no storage adapter to persist the paused state.  
    *Fix*: Always pass a checkpointer (e.g. `checkpointer=MemorySaver()`) when using human approval nodes.
    """)

col_tr3, col_tr4 = st.columns(2)
with col_tr3:
    st.warning("⚠️ Trap 3 (Performance Warning): Sequential Execution of Independent Tool Calls")
    st.markdown("""
    *Mechanism*: Writing a custom tool execution loop with sequential `for call in tool_calls: call()` instead of parallel dispatch.  
    *Consequence*: If an agent triggers 5 web searches taking 1.5s each, total latency is 7.5s instead of 1.5s.  
    *Fix*: Execute independent tool calls concurrently using `asyncio.gather` or ThreadPoolExecutor.
    """)

with col_tr4:
    st.warning("⚠️ Trap 4 (Architectural Warning): Loss of Conversation Context During Naive Summarization")
    st.markdown("""
    *Mechanism*: Truncating messages without providing an LLM-generated rolling summary.  
    *Consequence*: The agent suffers sudden amnesia, losing user preferences, constraints, or previous instructions.  
    *Fix*: Always distill key entities, constraints, and unresolved questions into the rolling summary channel.
    """)

st.subheader("💡 Core Knowledge Flashcards")

col_fc1, col_fc2, col_fc3, col_fc4 = st.columns(4)
with col_fc1:
    st.info("**LLM Node**\\nA cognitive node that formats prompts, invokes a language model, and emits AIMessage responses.")
with col_fc2:
    st.info("**ToolNode**\\nA specialized prebuilt actuator node that executes tool calls and returns ToolMessages.")
with col_fc3:
    st.info("**Validation Node**\\nA quality gatekeeper node that enforces schema compliance, safety checks, and business invariants.")
with col_fc4:
    st.info("**interrupt()**\\nA LangGraph breakpoint function that halts execution for human review and persists thread state.")

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

c1 = st.checkbox("I understand the architectural separation between LLM reasoning nodes and Tool actuator nodes.", value=True)
c2 = st.checkbox("I know how to implement prebuilt ToolNode to handle single and parallel tool calls.", value=True)
c3 = st.checkbox("I understand the role of Validation nodes in enforcing Pydantic guardrails.", value=True)
c4 = st.checkbox("I know how to implement human-in-the-loop approvals using interrupt() and checkpointers.", value=True)
c5 = st.checkbox("I understand how Summarization nodes manage long context windows and token budgets.", value=True)
c6 = st.checkbox("I can design complex multi-node workflows combining cognition, validation, and actuation.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎉 Mastery Confirmed: You have completed Lesson 5.2 'Node Types' with full architectural and practical comprehension!")
