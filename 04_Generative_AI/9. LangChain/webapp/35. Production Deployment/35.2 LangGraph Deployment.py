import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="35.2 LangGraph Deployment - LangChain Curriculum",
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
    .lg-badge {
        background-color: #1E1B4B;
        color: #A78BFA;
        border: 1px solid #6366F1;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .badge-stream {
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
class LangGraphDeploymentEngine:
    """
    Simulates LangGraph Cloud / Self-Hosted Agent Server,
    streaming modes (values, updates, events), and thread state transitions.
    """
    def __init__(self):
        self.stream_modes = {
            "values": {
                "desc": "Emits full state representation at every step.",
                "payload_example": '{"messages": [HumanMessage, AIMessage], "current_step": "synthesized"}',
                "best_for": "UI components that require complete state redraws."
            },
            "updates": {
                "desc": "Emits only state deltas produced by each node execution.",
                "payload_example": '{"agent_node": {"messages": [AIMessage(content="Search completed.")]}}',
                "best_for": "Efficient frontend updates and step-by-step progress bars."
            },
            "events": {
                "desc": "Emits fine-grained tokens and intermediate tool call chunks in real time.",
                "payload_example": '{"event": "on_chat_model_stream", "data": {"chunk": "LangGraph"}}',
                "best_for": "Real-time character-by-character typewriter chat streaming."
            }
        }

    def simulate_server_run(self, stream_mode: str, query: str) -> Dict[str, Any]:
        info = self.stream_modes.get(stream_mode, self.stream_modes["updates"])
        return {
            "mode": stream_mode,
            "query": query,
            "sample_emission": info["payload_example"],
            "thread_id": "th_prod_9981a",
            "checkpointer_persisted": True,
            "status": "COMPLETED"
        }

# Instantiate engine
lg_deploy_engine = LangGraphDeploymentEngine()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The LangGraph Server Architecture, State Checkpointers, & Task Queues", divider="rainbow")

st.markdown("""
Unlike stateless REST endpoints, long-running agent workflows require a **stateful execution runtime**. 
**LangGraph Server** provides a scalable, production-grade deployment infrastructure engineered specifically 
for cyclic agents, background task queues, human-in-the-loop interrupts, and durable thread checkpointers.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph LangGraph_Server_Blueprint {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
        node [fontname="Helvetica", shape=box, style="filled,rounded", color="#38BDF8", penwidth=1.5];
        edge [fontname="Helvetica", color="#94A3B8", fontcolor="#CBD5E1", fontsize=10, penwidth=1.2];

        subgraph cluster_client {
            label = "Client & Ingress Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            client_app [label="Frontend WebApp / SDK Client\\n(Sends thread_id, inputs, stream_mode)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
            http_sse [label="HTTP / SSE Streaming Gateway\\n(Bi-directional Server-Sent Events)", fillcolor="#0F172A", fontcolor="#38BDF8"];
        }

        subgraph cluster_server {
            label = "LangGraph Server Runtime (FastAPI + Task Workers)";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            api_gateway [label="LangGraph API Gateway\\n(Routes runs, threads, assistants, cron jobs)", fillcolor="#1E1B4B", fontcolor="#A78BFA"];
            task_worker [label="Asynchronous Task Worker Pool\\n(Executes StateGraph nodes, tool calls, interrupts)", fillcolor="#064E3B", fontcolor="#34D399", color="#10B981"];
            checkpointer [label="PostgreSQL / Redis Checkpointer\\n(Serializes thread state after every node step)", fillcolor="#0F172A", fontcolor="#F59E0B", color="#F59E0B"];
        }

        subgraph cluster_state {
            label = "State & Thread Persistence Layer";
            style = "filled,rounded";
            fillcolor = "#1E293B";
            color = "#64748B";
            fontcolor = "#F8FAFC";
            fontsize = 12;

            postgres_db [label="PostgreSQL Database\\n(Thread checkpoints, run metadata, time-travel history)", fillcolor="#0F172A", fontcolor="#F8FAFC"];
        }

        client_app -> http_sse;
        http_sse -> api_gateway;
        api_gateway -> task_worker [label="dispatches run"];

        task_worker -> checkpointer [label="saves checkpoint"];
        checkpointer -> postgres_db;

        task_worker -> http_sse [label="streams tokens & state deltas"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
<div class="modern-callout">
    <strong>Stateful Server Architecture:</strong> Standard web frameworks lose in-flight memory if a container restarts. 
    LangGraph Server persists state to PostgreSQL at every step boundary. If a pod crashes mid-execution, 
    the workflow resumes seamlessly from its last checkpoint without losing thread context.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 2: APPLICATION STRUCTURE
# ==============================================================================
st.header("2. LangGraph Application Structure: Directory Layout and Graph Definitions", divider="rainbow")

st.markdown("""
A production-ready LangGraph application adheres to a standardized project layout 
recognized by the LangGraph CLI and deployment orchestrator.
""")

st.markdown("""
```
my-agent-service/
├── langgraph.json          # Core deployment specification file
├── pyproject.toml          # Pinned dependency definitions
├── .env                    # Local development secrets (ignored by Git)
└── src/
    ├── __init__.py
    ├── state.py            # TypedDict / Pydantic agent state schemas
    ├── tools.py            # Independent @tool definitions
    └── agent.py            # StateGraph builder and compiled graph export
```
""")

# ==============================================================================
# SECTION 3: LANGGRAPH.JSON
# ==============================================================================
st.header("3. langgraph.json: Configuration Schema and Graph Pointers", divider="rainbow")

st.markdown("""
The `langgraph.json` manifest instructs the deployment server which Python files contain compiled graphs, 
what dependencies to bundle, and how environment variables are wired.
""")

st.code('''{
  "dependencies": ["."],
  "graphs": {
    "customer_support_agent": "./src/agent.py:graph"
  },
  "env": ".env",
  "python_version": "3.12"
}
''', language="json")

st.markdown("""
### Key Manifest Properties
- **`graphs`**: A mapping of public API endpoint names to their Python entrypoints (`file_path:variable_name`).
- **`dependencies`**: Specifies local paths or wheel packages required to execute the graph.
- **`python_version`**: Ensures consistent runtime container compilation.
""")

# ==============================================================================
# SECTION 4: LANGGRAPH CLI
# ==============================================================================
st.header("4. LangGraph CLI: Installation, Docker Orchestration, and Command Lifecycle", divider="rainbow")

st.markdown("""
The **LangGraph CLI** (`langgraph-cli`) automates local Docker-based development, 
providing containerized infrastructure management and local compilation tools.
""")

st.markdown("""
### Core CLI Lifecycle Commands
- `langgraph up`: Spins up the full Dockerized LangGraph server (FastAPI, Redis, PostgreSQL checkpointer).
- `langgraph down`: Tears down local containers and releases network ports.
- `langgraph build`: Compiles the application into a standalone production container.
""")

# ==============================================================================
# SECTION 5: LOCAL DEVELOPMENT
# ==============================================================================
st.header("5. Local Development: LangGraph Studio UI, Hot-Reloading, and Interactive Debugging", divider="rainbow")

st.markdown("""
Local development requires rapid iteration. Running `langgraph dev` launches a lightweight local 
development server with **automatic code reloading** and an integrated visual debugging studio.
""")

st.markdown("""
### LangGraph Studio Visual IDE
When running `langgraph dev`, the CLI launches **LangGraph Studio** at `http://localhost:2024`. 
Engineers can visualize graph topology, trigger test runs, step through node executions, 
and inspect thread state snapshots interactively.
""")

# ==============================================================================
# SECTION 6: LOCAL TESTING
# ==============================================================================
st.header("6. Local Testing: Headless Graph Execution and Thread Checkpointer Assertions", divider="rainbow")

st.markdown("""
Before deploying agents to production, graphs must be validated through automated headless unit tests 
verifying state transitions and interrupt behaviors.
""")

st.code('''import pytest
from langgraph.checkpoint.memory import MemorySaver
# from src.agent import build_agent_graph

def test_agent_graph_state_transition():
    # 1. Initialize graph with in-memory checkpointer
    checkpointer = MemorySaver()
    # graph = build_agent_graph(checkpointer=checkpointer)
    
    # 2. Invoke graph on test thread
    config = {"configurable": {"thread_id": "test_thread_001"}}
    # result = graph.invoke({"messages": [("user", "Hello!")]}, config=config)
    
    # 3. Assert state invariants
    # assert len(result["messages"]) >= 2
    # assert result["messages"][-1].type == "ai"
''', language="python")

# ==============================================================================
# SECTION 7: AGENT SERVER
# ==============================================================================
st.header("7. Agent Server: Production ASGI Serving and Streaming Protocols", divider="rainbow")

st.markdown("""
The production LangGraph Agent Server exposes REST and SSE endpoints supporting 
three distinct streaming protocols:
""")

stream_df = pd.DataFrame([
    {"Stream Mode": "stream_mode='values'", "Payload Emitted": "Full state dictionary at every step", "Network Overhead": "Higher (emits entire state)", "Ideal Use Case": "Stateless UI redraws and final summary screens"},
    {"Stream Mode": "stream_mode='updates'", "Payload Emitted": "Node-specific state delta dictionaries", "Network Overhead": "Low (only emits changed keys)", "Ideal Use Case": "Step-by-step progress steppers and workflow logs"},
    {"Stream Mode": "stream_mode='events'", "Payload Emitted": "Fine-grained token chunks and tool call events", "Network Overhead": "Minimal chunked bytes", "Ideal Use Case": "Typewriter-style LLM token streaming in chat interfaces"}
])
st.dataframe(stream_df, use_container_width=True)

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full LangGraph Standalone Service with Durable Checkpointing", divider="rainbow")

st.markdown("""
Below is the self-contained production Python script demonstrating a complete 
**standalone LangGraph agent service featuring thread state persistence and step-by-step updates**.
""")

st.code('''from typing import TypedDict, List, Dict, Any

# 1. State Definition
class ProductionAgentState(TypedDict):
    thread_id: str
    messages: List[str]
    current_node: str

# 2. Standalone Agent Server Simulator
class StandaloneAgentServer:
    def __init__(self):
        # Simulated durable storage (PostgreSQL / SQLite)
        self.checkpoint_store: Dict[str, ProductionAgentState] = {}

    def get_or_create_state(self, thread_id: str) -> ProductionAgentState:
        if thread_id not in self.checkpoint_store:
            self.checkpoint_store[thread_id] = {
                "thread_id": thread_id,
                "messages": [],
                "current_node": "START"
            }
        return self.checkpoint_store[thread_id]

    def execute_turn(self, thread_id: str, user_message: str, stream_mode: str = "updates") -> List[Dict[str, Any]]:
        state = self.get_or_create_state(thread_id)
        emissions = []

        # Step 1: User message ingestion
        state["messages"].append(f"User: {user_message}")
        if stream_mode == "updates":
            emissions.append({"node": "ingress", "delta": {"user_message_received": True}})

        # Step 2: Agent reasoning node
        state["current_node"] = "agent_reasoning"
        ai_reply = f"AI Response to: {user_message}"
        state["messages"].append(f"AI: {ai_reply}")
        
        if stream_mode == "updates":
            emissions.append({"node": "agent_reasoning", "delta": {"new_ai_message": ai_reply}})
        elif stream_mode == "values":
            emissions.append({"node": "agent_reasoning", "full_state": state.copy()})

        # Step 3: Commit state to persistent checkpointer
        self.checkpoint_store[thread_id] = state
        return emissions

server = StandaloneAgentServer()
# Execute turn with step-by-step updates streaming
updates = server.execute_turn("th_user_101", "How do I deploy LangGraph?", stream_mode="updates")
print("Emitted Stream Updates:", updates)
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: langgraph.json Config Builder & Stream Mode Protocol Tester", divider="rainbow")

st.markdown("""
Generate valid `langgraph.json` configuration manifests and simulate 
different streaming protocols across agent execution turns.
""")

wb_view = st.selectbox(
    "Select Workbench Tool:",
    ["1. Streaming Protocol Mode Simulator", "2. Interactive langgraph.json Generator", "3. Thread Checkpoint Inspector"]
)

if wb_view == "1. Streaming Protocol Mode Simulator":
    st.subheader("Streaming Mode Emission Inspector")
    mode_sel = st.selectbox("Select stream_mode:", ["updates", "values", "events"])
    test_q = st.text_input("Test Prompt:", "What are the core advantages of checkpointers?")

    if st.button("Simulate Server Streaming"):
        run_data = lg_deploy_engine.simulate_server_run(mode_sel, test_q)
        st.markdown(f"**Stream Mode Active:** `{run_data['mode']}`")
        st.markdown(f"**Thread ID:** `{run_data['thread_id']}` | **Checkpointer Status:** `PERSISTED`")

        st.markdown("#### Sample Stream Payload Emitted to Client:")
        st.code(run_data["sample_emission"], language="json")
        st.success(f"Mode '{mode_sel}' successfully simulated!")

elif wb_view == "2. Interactive langgraph.json Generator":
    st.subheader("langgraph.json Manifest Generator")
    graph_name = st.text_input("Graph Identifier Name:", "support_agent")
    entry_file = st.text_input("Entrypoint Python File:", "./src/agent.py")
    graph_var = st.text_input("Compiled Graph Variable:", "graph")
    py_ver = st.selectbox("Target Python Version:", ["3.12", "3.11"])

    generated_json = f"""{{
  "dependencies": ["."],
  "graphs": {{
    "{graph_name}": "{entry_file}:{graph_var}"
  }},
  "env": ".env",
  "python_version": "{py_ver}"
}}"""
    st.code(generated_json, language="json")
    st.caption("Save this manifest in your project root to deploy via LangGraph CLI or Cloud.")

else:
    st.subheader("Thread Checkpoint Persistence Inspector")
    st.markdown("Simulating durable state storage across multi-turn thread interactions:")
    st.info("""
    - **Thread**: `th_prod_9981a`
    - **Checkpoint 1 (Step 1)**: User submitted prompt. State: `messages=[HumanMessage]`
    - **Checkpoint 2 (Step 2)**: Agent planned tool call. State: `tool_calls=[search_docs]`
    - **Checkpoint 3 (Step 3)**: Tool executed. State: `messages=[..., ToolMessage]`
    - **Checkpoint 4 (Step 4)**: Agent synthesized final response.
    """)
    st.success("All checkpoints durable. If server experiences a restart at Step 2, execution replays from Checkpoint 2.")

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### High-Yield Technical Interview Questions")

with st.expander("Q1: Why is a durable checkpointer (PostgreSQL/Redis) mandatory for production LangGraph servers?"):
    st.markdown("""
    **Answer:**  
    Unlike simple linear chains, agents can execute multi-turn workflows, pause for human approval (`interrupt()`), 
    or run for minutes. An in-memory checkpointer (`MemorySaver`) loses all state if a worker pod is restarted or rescheduled. 
    A durable checkpointer (e.g., `PostgresSaver`) serializes graph state after every node transition to database storage, 
    ensuring fault tolerance and thread persistence.
    """)

with st.expander("Q2: How does stream_mode='updates' differ from stream_mode='values'?"):
    st.markdown("""
    **Answer:**  
    - `stream_mode='values'` emits the complete agent state dictionary at the completion of every node. 
    - `stream_mode='updates'` emits only the state delta dictionary returned by that specific node. 
    In large agent states (containing megabytes of conversation history or tool outputs), `updates` consumes vastly less network bandwidth.
    """)

with st.expander("Q3: What is the purpose of the langgraph.json configuration file?"):
    st.markdown("""
    **Answer:**  
    `langgraph.json` is the deployment contract for LangGraph CLI and Cloud. It maps public API endpoint names 
    to specific Python graph variables (`path/to/file.py:graph`), declares dependencies, specifies Python runtime versions, 
    and defines environment variable file paths.
    """)

with st.expander("Q4: How does LangGraph handle human-in-the-loop approvals during production deployment?"):
    st.markdown("""
    **Answer:**  
    When a graph node invokes `interrupt()`, the execution halts, and the current state is saved to the persistent checkpointer. 
    The server marks the run status as `INTERRUPTED`. A human reviewer inspects the state via UI or API, updates state variables 
    if needed, and resumes execution by dispatching a new run with `None` or feedback inputs.
    """)

with st.expander("Q5: What is the role of LangGraph Studio during local development?"):
    st.markdown("""
    **Answer:**  
    Launched via `langgraph dev`, LangGraph Studio provides a specialized visual IDE that automatically renders 
    the graph's nodes, edges, and branches. Developers can trigger test runs, inject custom state overrides, 
    step through node executions, and visualize time-travel checkpoints interactively.
    """)

st.markdown("### Production Traps & Antipatterns")

st.error("""
**Trap 1: Using In-Memory MemorySaver in Production Deployments**  
Deploying an agent with `MemorySaver` means any Kubernetes pod restart wipes out all active user threads. 
Always use `PostgresSaver` in production.
""")

st.error("""
**Trap 2: Forgetting to Pin Compiled Graph Entrypoints in langgraph.json**  
Referencing an uncompiled `StateGraph` object rather than a compiled graph (`workflow.compile()`) 
causes the deployment server to fail at startup.
""")

st.warning("""
**Trap 3: Streaming stream_mode='values' Over High-Volume Mobile Connections**  
Sending full state representations (containing dozens of messages) at every step introduces severe 
network latency on mobile clients. Use `stream_mode='updates'` or `stream_mode='events'`.
""")

st.warning("""
**Trap 4: Modifying State Keys Directly Inside Nodes Without Returning Deltas**  
Mutating state objects in-place instead of returning new key-value dictionary deltas breaks channel reducers 
and prevents checkpointers from detecting state changes.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>PostgresSaver</h4>
        <p>A production checkpointer persisting agent graph state, thread snapshots, and execution history into PostgreSQL.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>stream_mode='updates'</h4>
        <p>A streaming protocol emitting only node output deltas at each execution step, minimizing network bandwidth.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>langgraph.json</h4>
        <p>The core deployment manifest defining graph pointers, environment configurations, and Python dependencies.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>interrupt()</h4>
        <p>A LangGraph control primitive halting graph execution to persist state and await external human authorization.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the production directory structure and graph entrypoint conventions for LangGraph.", value=True)
chk_2 = st.checkbox("I know how to author a valid langgraph.json manifest mapping graph endpoints to Python variables.", value=True)
chk_3 = st.checkbox("I can use the LangGraph CLI (langgraph dev, langgraph up) for local hot-reloading development.", value=True)
chk_4 = st.checkbox("I know how to author headless unit tests verifying state transitions against checkpointers.", value=True)
chk_5 = st.checkbox("I understand the architectural differences between stream_mode values, updates, and events.", value=True)
chk_6 = st.checkbox("I know why durable checkpointers (PostgresSaver) are required to support human interrupts and restarts.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangGraph Deployment competencies verified! Ready for LangSmith Deployment.")
