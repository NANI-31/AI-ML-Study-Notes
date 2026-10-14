import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="26.2 Real-Time Interfaces - LangGraph Curriculum",
    page_icon="⚡",
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
        background-color: #1E1B4B;
        color: #C084FC;
        border: 1px solid #7E22CE;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
    }
    .tool-box {
        background-color: #0F172A;
        border: 1px dashed #334155;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODULE-SCOPE SIMULATOR & WORKBENCH ENGINE
# ==============================================================================
class RealTimeInterfaceSimulator:
    """
    Simulates real-time interactive UI states:
    - Streaming message chunks
    - Node progress indicators
    - Collapsible tool execution cards
    - HITL interrupt modal and resume payloads
    """
    def __init__(self):
        self.interrupt_active = False
        self.interrupt_payload = {
            "action": "execute_database_migration",
            "table": "customer_accounts",
            "rows_affected": 14200,
            "risk_level": "HIGH"
        }
        self.execution_timeline = [
            {"step": 1, "node": "query_parser", "status": "COMPLETED", "duration_ms": 140},
            {"step": 2, "node": "schema_validator", "status": "COMPLETED", "duration_ms": 220},
            {"step": 3, "node": "human_approval_gate", "status": "PAUSED_INTERRUPT", "duration_ms": 0},
            {"step": 4, "node": "migration_executor", "status": "PENDING", "duration_ms": 0}
        ]

    def trigger_interrupt(self) -> None:
        self.interrupt_active = True

    def resolve_interrupt(self, approved: bool) -> str:
        self.interrupt_active = False
        if approved:
            self.execution_timeline[2]["status"] = "COMPLETED"
            self.execution_timeline[3]["status"] = "COMPLETED"
            self.execution_timeline[3]["duration_ms"] = 480
            return "Migration approved and executed successfully."
        else:
            self.execution_timeline[2]["status"] = "REJECTED"
            self.execution_timeline[3]["status"] = "SKIPPED"
            return "Migration rejected by supervisor. Graph aborted."

    def reset_state(self) -> None:
        self.interrupt_active = False
        self.execution_timeline[2]["status"] = "PAUSED_INTERRUPT"
        self.execution_timeline[3]["status"] = "PENDING"
        self.execution_timeline[3]["duration_ms"] = 0

ui_sim = RealTimeInterfaceSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Reactive UI State Machine, Event Dispatchers, and Real-Time Agent Interfaces", divider="rainbow")

st.markdown("""
A state-of-the-art AI agent interface is fundamentally a **reactive UI state machine**. Unlike passive chatbots that 
display only final messages, real-time agent frontends reflect the internal reasoning state of the graph: 
they stream tokens with smooth typing effects, display active node progress indicators, reveal expandable tool execution pills, 
and render interactive approval cards when the graph triggers `interrupt()`.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph RealTimeUIBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_backend_events {
            label="1. LangGraph Streaming Event Dispatcher";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            evt_node [label="Node Start/End Events\\n(Superstep Indicators)", fillcolor="#0F172A"];
            evt_token [label="Token Chunk Events\\n(Text Stream Segments)", fillcolor="#0F172A"];
            evt_tool [label="Tool Invocation Events\\n(Tool Name, Inputs & Results)", fillcolor="#0F172A"];
            evt_interrupt [label="Interrupt Barrier Event\\n(Action Risk, Form Schema)", fillcolor="#312E81"];
        }

        subgraph cluster_ui_components {
            label="2. Reactive Frontend Components";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            comp_progress [label="Progress & Step Tracker\\nHorizontal / Vertical Node Stepper", fillcolor="#1E293B"];
            comp_chat [label="Streaming Message Bubble\\nMarkdown Parser with Cursor Blinking", fillcolor="#1E293B"];
            comp_tool [label="Expandable Tool Card\\nSyntax-Highlighted Arguments & Outputs", fillcolor="#064E3B", fontcolor="#A7F3D0"];
            comp_modal [label="Human Approval Banner / Modal\\nApprove / Reject Action Buttons", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        subgraph cluster_user_actions {
            label="3. User Interaction & State Resumption";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            user_action [label="Supervisor Decision\\nClick 'Approve' or 'Reject'", fillcolor="#0F172A"];
            resume_cmd [label="POST /threads/{id}/runs/resume\\nCommand(resume={'approved': True})", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        evt_node -> comp_progress;
        evt_token -> comp_chat;
        evt_tool -> comp_tool;
        evt_interrupt -> comp_modal;

        comp_modal -> user_action;
        user_action -> resume_cmd;
        resume_cmd -> evt_node [label="Resumes Graph Execution"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: Streaming Agent Responses
# ==============================================================================
st.header("2. Streaming Agent Responses: Token Accumulation, Smooth Typing Effects, and Markdown Parsing", divider="rainbow")

st.markdown("""
Streaming agent responses requires managing partial token chunks without flickering or breaking markdown formatting:

1. **Token Buffer Accumulation**: Incoming token chunks are appended into an in-memory string buffer.
2. **Incremental Markdown Rendering**: Streaming incomplete markdown (such as unclosed code fences or bold asterisks) 
   can break layout. Frontend libraries (such as `react-markdown` with syntax highlighters) auto-close unfinished syntax 
   tokens during rendering.
3. **Cursor Typing Indicator**: A pulsating cursor block (`▌`) indicates active streaming until the node emits an `end` event.
""")

st.code('''// React Component for Streaming Message Bubble
import React from 'react';
import ReactMarkdown from 'react-markdown';

export const StreamingMessage = ({ content, isStreaming }: { content: string, isStreaming: boolean }) => {
  return (
    <div className="message-bubble assistant">
      <ReactMarkdown>{content}</ReactMarkdown>
      {isStreaming && <span className="animate-pulse inline-block w-2 h-4 bg-blue-500 ml-1" />}
    </div>
  );
};
''', language="typescript")

# ==============================================================================
# SECTION 3: Node Execution Status
# ==============================================================================
st.header("3. Node Execution Status: Visualizing Active Supersteps, Node Banners, and Lifecycle Indicators", divider="rainbow")

st.markdown("""
In complex multi-agent graphs, user confidence improves when the interface communicates which node is active 
(e.g., `"Querying Financial Records"`, `"Validating Schema"`, `"Synthesizing Summary"`).

### Node Lifecycle Indicators:
- **Pending (`⚪`)**: Scheduled to run in a future superstep.
- **Active (`🔵` - Spinning)**: Currently executing coroutine or awaiting model response.
- **Completed (`🟢`)**: Superstep committed; state delta merged into channels.
- **Failed (`🔴`)**: Node encountered an unhandled exception or retry exhaustion.
""")

# ==============================================================================
# SECTION 4: Tool Execution Status
# ==============================================================================
st.header("4. Tool Execution Status: Collapsible Tool Call Cards, Argument Payloads, and Execution Results", divider="rainbow")

st.markdown("""
When an agent calls an external tool (such as an SQL query or API request), displaying the invocation explicitly 
builds transparency and trust.

### Tool Card Architecture:
- **Collapsed by Default**: Shows a clean badge: `🔧 Executed tool: search_customers (120ms)`.
- **Expandable Details**: Expanding the card reveals syntax-highlighted JSON parameters and raw outputs.
- **Error Visualization**: If the tool failed, highlights the stack trace and shows how the agent recovered.
""")

with st.expander("🔍 View Tool Card UI State Breakdown Flow", expanded=True):
    tool_ui_dot = """
    digraph ToolUIFlow {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.4, ranksep=0.5];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        call_event [label="tool_start Event Received\\nTool: execute_sql_query\\nArgs: {'query': 'SELECT *...'}", fillcolor="#0F172A"];
        card_loading [label="Render Card (Loading State)\\nBadge: 'Executing Database Query...'\\nSpinner Active", fillcolor="#312E81"];
        card_done [label="Render Card (Completed State)\\nBadge: 'Database Query Finished (180ms)'\\nCollapsible Accordion with Results", fillcolor="#064E3B", fontcolor="#A7F3D0"];

        call_event -> card_loading;
        card_loading -> card_done [label="tool_end Event"];
    }
    """
    st.graphviz_chart(tool_ui_dot, use_container_width=True)

# ==============================================================================
# SECTION 5: Human Approval UI
# ==============================================================================
st.header("5. Human Approval UI: Interactive Confirmation Modals, Diff Editors, and Decision Banners", divider="rainbow")

st.markdown("""
When an agent workflow pauses at an approval gate, the UI transitions to an interactive human-in-the-loop modal. 
The approval interface must provide complete context:
- **Proposed Action**: What action the agent intends to take.
- **Impact Diff**: Highlighting changes (e.g., records to be deleted, email body draft).
- **Explicit Action Triggers**: Primary **Approve** and destructive **Reject** buttons that dispatch resumption payloads.
""")

# ==============================================================================
# SECTION 6: Interrupt UI
# ==============================================================================
st.header("6. Interrupt UI: Presenting interrupt() Payloads, Form Fields, and Command(resume) Triggers", divider="rainbow")

st.markdown("""
LangGraph's `interrupt()` primitive allows arbitrary Python dictionaries to be passed to the frontend. 
The frontend parses the payload and dynamically renders appropriate input forms:

- **Boolean Confirmation**: Simple Yes/No binary choice.
- **Form Fields**: Text inputs for human corrections (e.g., editing a draft email or adjusting an account ID).
- **Resumption Dispatch**: Submitting the form posts `Command(resume=payload)` back to the server.
""")

st.code('''// React Form Handler for LangGraph interrupt()
const handleResume = async (threadId: string, runId: string, approved: boolean, feedback: string) => {
  await fetch("/api/threads/" + threadId + "/runs/" + runId + "/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      resume: {
        approved: approved,
        feedback: feedback
      }
    })
  });
};
''', language="typescript")

# ==============================================================================
# SECTION 7: Progress Indicators
# ==============================================================================
st.header("7. Progress Indicators: Step Counters, Spinner Indicators, and Progress Bars for Long Workflows", divider="rainbow")

st.markdown("""
For multi-stage autonomous workflows (such as web research, data extraction, or code analysis spanning minutes), 
progress indicators reassure the user that the background process remains healthy:

- **Determinate Steppers**: Used when total supersteps are predetermined (e.g., `Step 3 of 5: Document Indexing`).
- **Indeterminate Spinners**: Used during cyclic ReAct loops where final iteration count is dynamic.
- **Elapsed Time Counters**: Live timers indicating total active execution duration.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full React / Streamlit Real-Time Agent Dashboard Component Suite", divider="rainbow")

st.markdown("""
The following complete module implements a real-time reactive agent monitor rendering node steppers, 
collapsible tool pills, and an interactive HITL approval banner.
""")

st.code('''from typing import Dict, Any, List
import streamlit as st

class AgentDashboardRenderer:
    """
    Renders real-time node progress, tool execution status,
    and human-in-the-loop approval banners.
    """
    def render_timeline(self, steps: List[Dict[str, Any]]):
        cols = st.columns(len(steps))
        for idx, (col, step) in enumerate(zip(cols, steps)):
            with col:
                status = step["status"]
                badge_color = "🟢" if status == "COMPLETED" else "🔵" if status == "RUNNING" else "⏸️" if status == "PAUSED_INTERRUPT" else "⚪"
                st.markdown(f"**Step {step['step']}**")
                st.caption(f"{badge_color} `{step['node']}`")
                if step["duration_ms"] > 0:
                    st.caption(f"{step['duration_ms']} ms")

    def render_tool_card(self, tool_name: str, args: dict, result: str):
        with st.expander(f"🔧 Tool Executed: {tool_name}", expanded=False):
            st.markdown("**Arguments:**")
            st.json(args)
            st.markdown("**Execution Result:**")
            st.code(result)

    def render_approval_banner(self, payload: dict, on_approve, on_reject):
        st.warning("⚠️ **Human Approval Required** — Action Suspended")
        st.markdown(f"**Action:** `{payload.get('action')}` | **Risk:** `{payload.get('risk_level')}`")
        st.json(payload)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve Action", type="primary"):
                on_approve()
        with c2:
            if st.button("Reject Action"):
                on_reject()
''', language="python")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: Live Agent Interaction & HITL Approval Interface Sandbox", divider="rainbow")

st.markdown("""
Experiment with live node steppers, expandable tool cards, and interactive human-in-the-loop interruption approvals.
""")

st.subheader("Live Workflow Execution Timeline")
cols = st.columns(4)
for idx, step in enumerate(ui_sim.execution_timeline):
    with cols[idx]:
        st.markdown(f"**Step {step['step']}: {step['node']}**")
        if step["status"] == "COMPLETED":
            st.success(f"COMPLETED ({step['duration_ms']} ms)")
        elif step["status"] == "PAUSED_INTERRUPT":
            st.warning("PAUSED (Awaiting Approval)")
        elif step["status"] == "REJECTED":
            st.error("REJECTED")
        else:
            st.info("PENDING")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Tool Execution Output")
    with st.expander("🔧 Tool: sql_schema_validator (220 ms)", expanded=True):
        st.json({"target_table": "customer_accounts", "ddl": "ALTER TABLE customer_accounts ADD COLUMN credit_score INT;"})
        st.success("Result: Schema syntax verified. 0 foreign key constraint conflicts detected.")

with col_right:
    st.subheader("Human-in-the-Loop Interrupt Gate")
    if ui_sim.execution_timeline[2]["status"] == "PAUSED_INTERRUPT":
        st.warning("⚠️ High-Risk Action Detected: `execute_database_migration`")
        st.json(ui_sim.interrupt_payload)
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("Approve & Resume", type="primary"):
                msg = ui_sim.resolve_interrupt(approved=True)
                st.success(msg)
                st.rerun()
        with c_btn2:
            if st.button("Reject & Abort"):
                msg = ui_sim.resolve_interrupt(approved=False)
                st.error(msg)
                st.rerun()
    elif ui_sim.execution_timeline[2]["status"] == "COMPLETED":
        st.success("Action was approved and migration committed successfully.")
        if st.button("Reset Simulation"):
            ui_sim.reset_state()
            st.rerun()
    else:
        st.error("Action was rejected and execution aborted.")
        if st.button("Reset Simulation"):
            ui_sim.reset_state()
            st.rerun()

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: How do you prevent layout shifts and flickering when streaming markdown responses to the browser?"):
    st.markdown("""
    **Answer:**  
    1. **Debounced Rendering**: Batch token updates using requestAnimationFrame or short debounce intervals (e.g., 30-50ms) rather than re-rendering React trees on every single token.  
    2. **Auto-Closing Parsers**: Use robust markdown renderers with plugins that automatically balance incomplete tags, bold indicators, or code fences during generation.  
    3. **Container Size Constraints**: Pre-allocate minimum heights or CSS containment to prevent vertical page jumps.
    """)

with st.expander("Q2: How does the frontend know whether an incoming event is a token stream or a tool call?"):
    st.markdown("""
    **Answer:**  
    LangGraph `astream_events(v2)` tags every emitted event with an explicit `event` discriminator (e.g., `on_chat_model_stream` vs `on_tool_start`). 
    The frontend client event dispatcher parses this discriminator and routes token strings to the message bubble and tool payloads to the tool card store.
    """)

with st.expander("Q3: What happens to the frontend UI when an interrupt() call is triggered on the backend?"):
    st.markdown("""
    **Answer:**  
    The backend stream halts and emits an `on_interrupt` event carrying the interruption payload. The frontend updates the step progress indicator 
    to `PAUSED`, opens the human approval modal, and awaits supervisor input. The browser connection can safely close; when the supervisor submits the form, 
    a new HTTP POST dispatches `Command(resume=...)` to resume the thread.
    """)

with st.expander("Q4: How should errors in intermediate tool nodes be displayed to end users?"):
    st.markdown("""
    **Answer:**  
    Display a collapsible warning card showing that the tool encountered an error, along with an indicator that the agent is attempting recovery. 
    Never crash the entire chat interface or hide the failure; transparency allows users to understand why the agent might be retrying or taking longer.
    """)

with st.expander("Q5: What is the benefit of rendering intermediate thinking/reasoning blocks in the UI?"):
    st.markdown("""
    **Answer:**  
    Revealing reasoning steps provides users with immediate feedback during multi-second latency gaps, clarifies the agent's problem-solving strategy, 
    and allows humans to catch faulty reasoning early before the agent commits destructive actions.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Re-Rendering Full Message Trees on Every Streamed Token**  
Binding raw SSE token events directly to top-level React state causes hundreds of full-tree re-renders per second, 
causing browser thread lockups and mobile device lag. Always isolate streaming updates in dedicated leaf components.
""")

st.error("""
**Trap 2: Losing Interrupt Context on Page Refresh**  
If a user refreshes the page while a graph is paused on `interrupt()`, a naive UI might reset to empty chat. 
Always query `GET /threads/{id}/state` on page load to detect pending interrupts and restore the approval banner.
""")

st.warning("""
**Trap 3: Cluttering UI with Verbose Raw Tool Outputs**  
Dumping thousands of lines of raw SQL results or JSON into chat bubbles degrades readability. 
Always wrap tool outputs in collapsible accordions with summarized status previews.
""")

st.warning("""
**Trap 4: Missing Loading States on Human Approval Submission**  
Failing to show a loading spinner when the supervisor clicks "Approve" can prompt the user to double-click, 
triggering duplicate resume requests that fail with 409 Conflict errors.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Reactive UI State Machine</h4>
        <p>A frontend architecture transitioning through Loading, Streaming, Tool Executing, Interrupted, and Completed states based on backend events.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Token Typing Indicator</h4>
        <p>A visual pulsating cursor indicating real-time text generation, replaced by committed text upon node completion.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>Collapsible Tool Cards</h4>
        <p>UI components encapsulating tool names, parameters, execution latencies, and raw outputs in compact expandable views.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Command(resume=...) Action</h4>
        <p>The client API call delivering human approval or feedback data to unblock and resume an interrupted graph thread.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I know how to stream agent token responses with markdown formatting and typing indicators.", value=True)
chk_2 = st.checkbox("I understand how to render active node progress indicators across superstep lifecycles.", value=True)
chk_3 = st.checkbox("I can implement expandable tool cards displaying arguments, execution times, and results.", value=True)
chk_4 = st.checkbox("I know how to build interactive human approval interfaces for gating high-risk operations.", value=True)
chk_5 = st.checkbox("I understand how to handle interrupt() payloads and resume threads via Command(resume).", value=True)
chk_6 = st.checkbox("I can design responsive progress bars and steppers for multi-stage autonomous workflows.", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("Real-time agent interface engineering mastery confirmed! Ready for LangGraph Frontend SDK patterns.")
