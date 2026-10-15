import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.4 LangSmith Studio - LangChain Architecture",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling Studio UI simulator cards
st.markdown("""
<style>
    .studio-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-studio {
        background-color: #4338CA;
        color: #E0E7FF;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-checkpoint {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockLangGraphStudioSimulator:
    """
    Simulation engine showcasing LangGraph Studio & LangSmith local desktop/web UI:
    interactive graph canvas, step-by-step state inspection, thread forking,
    and hot-reload breakpoint simulation.
    """
    def __init__(self):
        self.threads = {
            "thread-studio-01": {
                "id": "thread-studio-01",
                "name": "CustomerSupportAgent_v3",
                "current_node": "tool_node",
                "checkpoint_step": 3,
                "state_channels": {
                    "messages": [
                        {"role": "user", "content": "I need to dispute transaction TX-8891."},
                        {"role": "assistant", "content": "I will check the transaction record.", "tool_calls": ["lookup_transaction"]}
                    ],
                    "customer_tier": "VIP_PLATINUM",
                    "dispute_flag": True,
                    "auth_token": "VALID_JWT_SESSION"
                },
                "breakpoints": ["tool_node"],
                "status": "PAUSED_AT_BREAKPOINT"
            },
            "thread-studio-02": {
                "id": "thread-studio-02",
                "name": "CodeReviewAgent",
                "current_node": "security_scanner",
                "checkpoint_step": 2,
                "state_channels": {
                    "messages": [
                        {"role": "user", "content": "Review PR-402 for injection risks."}
                    ],
                    "files_scanned": ["auth.py", "db_client.py"],
                    "vulnerabilities_detected": 0
                },
                "breakpoints": [],
                "status": "COMPLETED"
            }
        }

    def fork_and_edit_state(self, thread_id: str, channel_key: str, new_val: Any) -> Dict[str, Any]:
        source = self.threads.get(thread_id)
        if not source:
            return {"status": "ERROR", "message": "Thread not found."}
        
        forked_id = f"{thread_id}-forked-{source['checkpoint_step']}"
        new_channels = dict(source["state_channels"])
        new_channels[channel_key] = new_val

        self.threads[forked_id] = {
            "id": forked_id,
            "name": f"{source['name']} (Forked)",
            "current_node": source["current_node"],
            "checkpoint_step": source["checkpoint_step"],
            "state_channels": new_channels,
            "breakpoints": source["breakpoints"],
            "status": "FORKED_READY"
        }
        return {
            "status": "SUCCESS",
            "forked_thread_id": forked_id,
            "message": f"Successfully forked thread from step {source['checkpoint_step']}.",
            "updated_state": new_channels
        }

# Main Application Title & Objective
st.title("🖥️ 21.4 LangSmith & LangGraph Studio")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
**LangGraph Studio** is the specialized graphical IDE and visual debugger built for developing, testing, and debugging stateful AI agents. 
Integrated natively with LangSmith, Studio provides a local desktop canvas to **visualize graph topologies**, **run agents interactively**, 
**inspect state channel memory**, **fork threads to replay edge cases**, and **iterate rapidly with hot-reloading**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: LangGraph Studio Local Debugger & Hot-Reload Architecture", divider="rainbow")
st.markdown("""
The architecture below delineates how LangGraph Studio pairs a local development server with an interactive web UI. 
The Studio daemon watches Python source files, hot-reloads graph topologies, provides local memory checkpointing, 
and synchronizes traces with cloud LangSmith workspaces.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    std_dot = graphviz.Digraph(comment="LangGraph Studio Architecture")
    std_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    std_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    std_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    std_dot.node("local_code", "Local Codebase\n[agent.py / langgraph.json]\nPython Files on Disk", fillcolor="#0F172A", shape="ellipse")
    std_dot.node("hot_reload", "Studio Hot-Reload Watcher\n[Detects AST Changes & Recompiles Graph]", fillcolor="#1E3A8A")
    std_dot.node("local_server", "LangGraph Local Server\n[Runs Graph API on localhost:8123]", fillcolor="#0E7490")
    
    # Studio Client UI
    std_dot.node("studio_ui", "LangGraph Studio UI\n[Interactive Visual Canvas & State Inspector]", fillcolor="#701A75")
    std_dot.node("thread_store", "Local SQLite Checkpointer\n[Time-Travel Checkpoint Store]", fillcolor="#334155")
    std_dot.node("smith_cloud", "LangSmith Cloud Ingestion\n[Automated Remote Trace Sync]", fillcolor="#047857", shape="ellipse")

    std_dot.edge("local_code", "hot_reload", label=" File Save (Ctrl+S)")
    std_dot.edge("hot_reload", "local_server", label=" Rebuild StateGraph")
    std_dot.edge("local_server", "studio_ui", label=" WebSocket Stream")
    std_dot.edge("studio_ui", "thread_store", label=" Branch / Fork Checkpoint")
    std_dot.edge("local_server", "smith_cloud", label=" Telemetry Sync")

    st.graphviz_chart(std_dot, use_container_width=True)

# --- SECTION 2: STUDIO BASICS ---
st.header("2. Studio Basics & Workspace Setup", divider="rainbow")
st.markdown("""
LangGraph Studio is available as a native desktop application (macOS, Windows, Linux) and as a Docker-backed local CLI service. 
It operates by reading a configuration file named **`langgraph.json`** located in the project root.
""")

col_b1, col_b2 = st.columns(2)
with col_b1:
    st.subheader("`langgraph.json` Configuration Schema")
    st.markdown("""
    * `python_version`: Target Python interpreter version (e.g. `"3.11"`).
    * `dependencies`: Local paths or `pyproject.toml` references.
    * `graphs`: Dictionary mapping user-friendly names to module entry points.
    * `env`: Environment variables or path to `.env` file containing API keys.
    """)
with col_b2:
    st.subheader("Minimal `langgraph.json` Manifest")
    st.code("""
{
  "dependencies": ["."],
  "graphs": {
    "customer_support": "./agent.py:app",
    "financial_analyst": "./finance_graph.py:graph"
  },
  "env": ".env"
}
    """, language="json")

# --- SECTION 3: RUNNING AGENTS ---
st.header("3. Running Agents Interactively in Local Studio", divider="rainbow")
st.markdown("""
Instead of writing temporary test scripts or mocking CLI inputs, Studio allows developers to execute graphs 
through an interactive visual chat interface.
""")

col_run1, col_run2 = st.columns(2)
with col_run1:
    st.subheader("Execution Features in Studio")
    st.markdown("""
    * **Visual Node Highlighting**: Watch nodes illuminate in real time as the Pregel engine traverses edges.
    * **Parallel Step Rendering**: Graph branches that execute in parallel display simultaneous active node indicators.
    * **Streaming Output Bubbles**: View model tokens and tool observations as they stream across the WebSocket connection.
    * **Thread Management**: Create multiple conversational threads to test distinct user personas side by side.
    """)
with col_run2:
    st.subheader("Starting LangGraph Studio Server")
    st.code("""
# Launch local server using LangGraph CLI:
pip install langgraph-cli
langgraph dev

# Studio launches on http://localhost:8123
# Open in browser or connect LangGraph Desktop App
    """, language="bash")

# --- SECTION 4: INSPECTING STATE ---
st.header("4. Inspecting State & Memory Channels", divider="rainbow")
st.markdown("""
The core superpower of Studio is **Total State Visibility**. At every step of graph execution, Studio renders the 
exact JSON state dictionary, including all message channels, custom reducers, and private node variables.
""")

col_st1, col_st2 = st.columns(2)
with col_st1:
    st.subheader("State Channel Inspector")
    st.markdown("""
    * **Reducers Diffing**: Shows which keys were updated by the most recent node execution (highlighted in green).
    * **Message Buffer Expansion**: Click any message in the `messages` channel to view tool calls, system prompts, or raw metadata.
    * **Checkpoint History**: Scroll backward through the timeline to see what the state looked like 5 steps ago.
    """)
with col_st2:
    st.subheader("State Inspection Panel View")
    st.code("""
{
  "messages": [
    {"type": "human", "content": "Refund order #4401"},
    {"type": "ai", "tool_calls": [{"name": "process_refund"}]}
  ],
  "customer_status": "AUTHENTICATED",
  "refund_amount": 49.99,
  "retry_counter": 0
}
    """, language="json")

# --- SECTION 5: TESTING INPUTS ---
st.header("5. Testing Inputs & Edge-Case Replays", divider="rainbow")
st.markdown("""
Debugging non-deterministic agents requires testing adversarial inputs, malformed JSON, and unusual edge cases. 
Studio allows engineers to edit input payloads directly in a structured JSON editor before triggering execution.
""")

col_in1, col_in2 = st.columns(2)
with col_in1:
    st.subheader("Input Testing Strategies")
    st.markdown("""
    * **Adversarial Injections**: Test jailbreak prompts against your system guardrails in real time.
    * **Context Edge Cases**: Inject 10,000-word documents to test token window truncation and retriever behavior.
    * **Role Impersonation**: Inject mock tool messages to evaluate how the agent behaves when external APIs return errors.
    """)
with col_in2:
    st.subheader("Direct State Forking")
    st.markdown("""
    * In the Studio canvas, click **Fork Thread** on any historical checkpoint.
    * Modify a variable (e.g. change `user_tier` from `BASIC` to `VIP`).
    * Re-run execution from that exact step forward without re-executing previous steps!
    """)

# --- SECTION 6: DEBUGGING WORKFLOWS ---
st.header("6. Debugging Workflows with Visual Breakpoints", divider="rainbow")
st.markdown("""
Studio brings traditional IDE debugging to agentic graphs. Engineers can click on any node in the canvas 
to toggle **Visual Breakpoints** (`interrupt_before` / `interrupt_after`).
""")

col_bp1, col_bp2 = st.columns(2)
with col_bp1:
    st.subheader("Visual Breakpoint Workflow")
    st.markdown("""
    1. Click the red breakpoint circle on `execute_database_mutation`.
    2. Submit a user query that triggers the database tool.
    3. The graph executes until it reaches the node, then halts cleanly.
    4. Review the planned SQL statement in the state inspector.
    5. Click **Step Over** to execute, or **Edit State** to correct syntax before running!
    """)
with col_bp2:
    st.subheader("Breakpoint Interaction Pattern")
    st.code("""
# Breakpoint triggers in Studio UI:
# [Status: PAUSED AT NODE: execute_database_mutation]

# Action options available in Studio toolbar:
1. [▶ Continue]    -> Resume execution without modification
2. [⏭ Step Over]   -> Run this single node and pause again
3. [✏️ Edit State]  -> Patch state values before running
4. [✖ Cancel]       -> Abort current thread execution
    """, language="text")

# --- SECTION 7: ITERATING ON AGENTS ---
st.header("7. Iterating on Agents via Hot-Reload & Prompt Tweaks", divider="rainbow")
st.markdown("""
The feedback loop in AI engineering is often bottlenecked by server restarts. 
LangGraph Studio features **Instant Hot-Reload**: when you modify a prompt template or add a new tool in Python code, 
Studio reloads the graph definition in milliseconds while **preserving existing conversational threads and checkpoints**.
""")

col_it1, col_it2 = st.columns(2)
with col_it1:
    st.subheader("The Rapid Iteration Loop")
    st.markdown("""
    1. Notice model hallucination at Step 3 in Studio canvas.
    2. Edit system prompt instructions in `agent.py` in your code editor.
    3. Press `Ctrl+S` (Save).
    4. Studio instantly hot-reloads the graph.
    5. Click **Replay from Step 3** in Studio.
    6. Verify that the updated prompt fixes the hallucination immediately!
    """)
with col_it2:
    st.subheader("Developer Velocity Gains")
    st.markdown("""
    | Task | Traditional CLI Loop | LangGraph Studio Loop |
    | :--- | :--- | :--- |
    | **Prompt Tweak & Verify** | 45 - 90 seconds (Full restart) | 2 - 5 seconds (Replay step) |
    | **State Inspection** | Multiple `print()` statements | Interactive JSON tree |
    | **Edge-Case Reproduction** | Manually re-typing 5 turns | One-click Thread Forking |
    | **Graph Visualization** | Static Graphviz PNG export | Live interactive WebGL canvas |
    """)

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: LangGraph Studio Local Manifest & Hot-Reload Debug Harness", divider="rainbow")
st.markdown("""
The following complete configuration demonstrates how to set up a production-ready LangGraph project with a 
`langgraph.json` manifest, environment isolation, and stateful debug harnesses.
""")

st.code('''
# === FILE 1: langgraph.json ===
{
  "dependencies": ["."],
  "graphs": {
    "enterprise_support_agent": "./support_graph.py:app"
  },
  "env": ".env"
}

# === FILE 2: support_graph.py ===
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    customer_id: str
    is_escalated: bool

def triage_node(state: SupportState) -> dict:
    last_text = state["messages"][-1].content.lower()
    if "lawyer" in last_text or "dispute" in last_text:
        return {"is_escalated": True}
    return {"is_escalated": False}

def resolution_node(state: SupportState) -> dict:
    if state.get("is_escalated"):
        return {"messages": [AIMessage(content="Escalating case to Priority Legal Team.")]}
    return {"messages": [AIMessage(content="Standard support resolution provided.")]}

# Assemble StateGraph
builder = StateGraph(SupportState)
builder.add_node("triage", triage_node)
builder.add_node("resolution", resolution_node)

builder.add_edge(START, "triage")
builder.add_edge("triage", "resolution")
builder.add_edge("resolution", END)

# In local Studio development, compiling with MemorySaver enables instant time travel
app = builder.compile(checkpointer=MemorySaver())
''', language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Local Studio State Inspector & Thread Forking Simulator", divider="rainbow")
st.markdown("""
Experience the visual debugging power of LangGraph Studio. Select a thread paused at a breakpoint, 
inspect its internal state channels, and simulate **Forking the Thread** with modified parameters.
""")

# Instantiate Module Scope Simulator
if "studio_sim" not in st.session_state:
    st.session_state.studio_sim = MockLangGraphStudioSimulator()

sim_std = st.session_state.studio_sim

col_stb1, col_stb2 = st.columns([1, 1])

with col_stb1:
    st.subheader("1. Studio Thread & Breakpoint Canvas")
    active_thread_id = st.selectbox(
        "Select Active Thread:",
        options=list(sim_std.threads.keys()),
        format_func=lambda x: f"{x} - {sim_std.threads[x]['name']} ({sim_std.threads[x]['status']})"
    )
    th_info = sim_std.threads[active_thread_id]

    st.markdown(f"**Thread ID**: `{th_info['id']}`")
    st.markdown(f"**Current Node**: `{th_info['current_node']}`")
    st.markdown(f"**Status**: `{th_info['status']}`")
    st.markdown(f"**Step Index**: `{th_info['checkpoint_step']}`")
    st.markdown(f"**Breakpoints Configured**: `{th_info['breakpoints'] or 'None'}`")

    st.markdown("#### Live State Channels Inspector:")
    st.json(th_info["state_channels"])

with col_stb2:
    st.subheader("2. Time-Travel Forking & State Patching")
    st.markdown("""
    Simulate forking this thread to test an alternative execution path. 
    Select a state key to patch and provide a modified value:
    """)

    edit_key = st.selectbox("State Key to Override:", options=["customer_tier", "dispute_flag", "auth_token"])
    edit_val = st.text_input("New Overridden Value:", value="VIP_EXECUTIVE")

    if st.button("🍴 Fork Thread & Replay from Breakpoint", use_container_width=True):
        fork_res = sim_std.fork_and_edit_state(active_thread_id, edit_key, edit_val)
        st.success(fork_res["message"])
        st.markdown(f"**New Forked Thread ID**: `{fork_res['forked_thread_id']}`")
        st.json(fork_res["updated_state"])

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How does LangGraph Studio integrate with local codebases without requiring code modifications?"):
    st.markdown("""
    LangGraph Studio reads a `langgraph.json` configuration manifest that points to the compiled `StateGraph` object 
    in Python (e.g., `./agent.py:app`). The local Studio daemon dynamically imports the graph, wraps it in local 
    HTTP/WebSocket servers, and watches the underlying files for changes to trigger instant AST hot-reloading.
    """)

with st.expander("Q2: What is 'Thread Forking' in LangGraph Studio, and why is it valuable for debugging?"):
    st.markdown("""
    Thread Forking creates a new branch from an existing checkpoint timestamp. It duplicates the exact state dictionary 
    at Step N, applies optional human parameter edits, and launches a new thread from that step forward. 
    This allows developers to test hypotheses and fix hallucinations without re-executing expensive upstream steps.
    """)

with st.expander("Q3: How do visual breakpoints in Studio differ from Python's standard `breakpoint()` / pdb?"):
    st.markdown("""
    * `pdb.set_trace()` freezes the entire Python process and blocks OS threads, which crashes async WebSocket streams.
    * Studio visual breakpoints leverage LangGraph's native Pregel interrupts (`interrupt_before`). The graph serializes 
      its state to the checkpointer and exits cleanly, keeping the server responsive and allowing interactive state inspection in the web browser.
    """)

with st.expander("Q4: Can LangGraph Studio connect to remote cloud agent deployments?"):
    st.markdown("""
    Yes. In addition to local development via `langgraph dev`, LangGraph Studio can connect directly to remote 
    LangGraph Cloud or LangSmith deployments. Developers can inspect production threads, replay customer sessions, 
    and debug remote state in the visual Studio canvas.
    """)

with st.expander("Q5: What is the role of reducers in state inspection within Studio?"):
    st.markdown("""
    Studio uses the state schema's channel reducers (e.g. `add_messages`) to calculate the exact diff between 
    what a node returned and what the accumulated state looks like. In the Studio UI, keys updated by reducers 
    are highlighted in real time, making state accumulation transparent.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_std_t1, col_std_t2 = st.columns(2)
with col_std_t1:
    st.error("""
    **Production Trap 1 (Fatal Error): Non-Deterministic Nodes Breaking Time-Travel Replays**  
    Writing nodes that rely on unseeded random numbers or live mutable system state (e.g. modifying an external database during replay). 
    When forking threads in Studio, non-deterministic side effects will corrupt external databases and invalidate replay results.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Missing `langgraph.json` Dependencies Declaration**  
    Failing to declare all local dependencies in `langgraph.json`. 
    When Studio compiles the graph, missing imports will cause hot-reload failures with silent background daemon errors.
    """)
with col_std_t2:
    st.error("""
    **Production Trap 3 (Fatal Error): Deploying Studio Debug Endpoints to Public Networks**  
    Leaving the local `langgraph dev` port (8123) bound to `0.0.0.0` on an unauthenticated public server. 
    Studio allows arbitrary state editing, input injection, and code reloading, creating a severe remote code execution vulnerability.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): State Schema Type Inconsistencies**  
    Editing state values in Studio to types that violate the underlying `TypedDict` or `Pydantic` schema (e.g. passing a string for an integer field). 
    Subsequent downstream nodes will fail with runtime `TypeError` exceptions.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc_std1, col_fc_std2 = st.columns(2)
with col_fc_std1:
    st.markdown("""
    <div class="studio-card">
        <h4>Flashcard 1: langgraph.json</h4>
        <p><strong>Config:</strong> Studio Manifest</p>
        <p><strong>Definition:</strong> The project manifest file defining graph entry points, dependencies, and environment files for LangGraph Studio.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <h4>Flashcard 2: Thread Forking</h4>
        <p><strong>Feature:</strong> Time-Travel Branching</p>
        <p><strong>Definition:</strong> Branching execution from any historical checkpoint to test alternative inputs or modified state parameters.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc_std2:
    st.markdown("""
    <div class="studio-card">
        <h4>Flashcard 3: Visual Breakpoint</h4>
        <p><strong>Tool:</strong> Graphical Interruption</p>
        <p><strong>Definition:</strong> Halting graph execution at a specific node in Studio to inspect or modify channel variables before resumption.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="studio-card">
        <h4>Flashcard 4: Hot-Reload</h4>
        <p><strong>Developer Velocity:</strong> Instant Recompilation</p>
        <p><strong>Definition:</strong> Automatic re-import of updated Python code on file save without destroying active conversational threads.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of LangGraph Studio and visual agent debugging workflows:")

chk_std1 = st.checkbox("1. I understand how to configure a `langgraph.json` manifest to register graphs for Studio.")
chk_std2 = st.checkbox("2. I know how to launch local Studio development servers using the `langgraph dev` CLI command.")
chk_std3 = st.checkbox("3. I understand how to inspect state channels and reducer diffs step-by-step in Studio.")
chk_std4 = st.checkbox("4. I know how to fork historical threads to replay edge cases and test prompt adjustments.")
chk_std5 = st.checkbox("5. I understand how visual breakpoints halt graph execution without blocking async event loops.")
chk_std6 = st.checkbox("6. I understand how hot-reloading accelerates the developer feedback loop during agent iteration.")

if chk_std1 and chk_std2 and chk_std3 and chk_std4 and chk_std5 and chk_std6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of LangSmith & LangGraph Studio.")
