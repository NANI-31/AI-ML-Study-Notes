import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="32.3 LangChain + React - Decoupled Frontend Architecture",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for React UI Component emulation
st.markdown("""
<style>
    .react-component {
        background-color: #0F172A;
        border: 1px solid #38BDF8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .tool-badge {
        background-color: #312E81;
        color: #C7D2FE;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class ReactAgentStateSimulator:
    """
    Simulates a client-side React frontend managing streaming agent messages,
    tool call visualization cards, interrupt approval modals, and structured component rendering.
    """
    def __init__(self):
        self.preset_agent_runs = {
            "Run 1: Database Query & Chart": {
                "user_prompt": "Show top 3 products by revenue this quarter.",
                "tool_calls": [
                    {"tool": "sql_query_generator", "args": {"schema": "sales_db", "limit": 3}, "status": "completed", "output": "[{'name': 'Enterprise AI', 'rev': '$2.4M'}, {'name': 'Cloud Seats', 'rev': '$1.8M'}]"},
                    {"tool": "chart_renderer", "args": {"type": "bar_chart", "metric": "revenue"}, "status": "completed", "output": "<BarChartData points=2 />"}
                ],
                "requires_human_approval": False,
                "structured_output": {"component": "RevenueBarChart", "data": [{"label": "Enterprise AI", "value": 2400000}, {"label": "Cloud Seats", "value": 1800000}]},
                "final_message": "Retrieved quarterly sales metrics. Enterprise AI leads revenue generation at $2.4M."
            },
            "Run 2: Sensitive Financial Transfer (HITL)": {
                "user_prompt": "Transfer $50,000 from treasury account to AWS Vendor billing.",
                "tool_calls": [
                    {"tool": "wire_transfer_tool", "args": {"recipient": "Amazon Web Services", "amount": 50000, "currency": "USD"}, "status": "interrupted", "output": "Awaiting Human Review"}
                ],
                "requires_human_approval": True,
                "structured_output": {"component": "ApprovalCard", "action": "Wire Transfer", "risk_level": "CRITICAL", "amount": "$50,000.00"},
                "final_message": "Agent execution halted at checkpoint 'human_approval'. Confirm transfer to proceed."
            }
        }

    def simulate_client_state(self, run_key: str, approved: bool = False) -> Dict[str, Any]:
        data = self.preset_agent_runs.get(run_key, self.preset_agent_runs["Run 1: Database Query & Chart"])
        state = dict(data)
        if state["requires_human_approval"] and approved:
            state["tool_calls"][0]["status"] = "approved_and_executed"
            state["tool_calls"][0]["output"] = "Transaction Confirmed: Wire #TX-99812 dispatched."
            state["requires_human_approval"] = False
            state["final_message"] = "Wire transfer for $50,000 confirmed and successfully dispatched."
        return state


# --- LESSON HEADER ---
st.title("32.3 LangChain + React: Decoupled Full-Stack Architecture")
st.markdown("""
Modern enterprise AI products decouple the backend LangChain/LangGraph agent runtime from the user interface
using an asynchronous REST/SSE backend (FastAPI) paired with a rich, dynamic frontend (React, Next.js).
Building production-grade client experiences requires managing Server-Sent Events (SSE) token streaming,
visualizing real-time tool execution trajectories, handling Human-in-the-Loop (HITL) interrupt dialogs,
and transforming structured JSON schema payloads into interactive UI components (Generative UI).
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Decoupled Full-Stack Architecture (FastAPI + React)", divider="rainbow")
st.markdown("""
The decoupled architecture establishes a clean separation of concerns. The React frontend interacts with FastAPI via
Server-Sent Events for streaming and standard REST for session state. State changes on the backend emit events consumed
by React hooks (`useChat`, `EventSource`, or TanStack Query), triggering component re-renders without refreshing the page.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph React_LangChain_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_react_frontend {
            label = "React / Next.js Client Application";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ReactApp [label="React App Root (Next.js App Router)\\nState Machine: Zustand / Redux / TanStack", fillcolor="#0F172A", color="#3B82F6"];
            SSEHook [label="SSE Streaming Hook\\n(useChat / fetchEventSource / custom parser)", fillcolor="#1E293B", color="#60A5FA"];
            ToolVisualizer [label="Tool Call Component\\n(Status Badges, Accordion, Inputs/Outputs)", fillcolor="#1E293B", color="#60A5FA"];
            HITLModal [label="Human Approval Modal Dialog\\n(Review Tool Parameters, Approve/Reject)", fillcolor="#1E293B", color="#60A5FA"];
            GenUI [label="Generative UI Registry\\n(Maps JSON Schema -> <Chart />, <Table />)", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_transport_layer {
            label = "Network Protocol & Ingress";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            HTTPGateway [label="API Ingress / Reverse Proxy (Nginx / Cloudflare)\\nHTTP/2 Keep-Alive + Server-Sent Events", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_fastapi_backend {
            label = "FastAPI Asynchronous Gateway & LangGraph";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            FastAPIRoutes [label="FastAPI Endpoints\\n(/api/chat/stream, /api/agent/resume)", fillcolor="#065F46", color="#10B981"];
            LangGraphRuntime [label="LangGraph Compiled Workflow\\n(Checkpointer, Tools, HITL Interrupts)", fillcolor="#065F46", color="#10B981"];
        }

        ReactApp -> SSEHook [label="Initialize Stream"];
        SSEHook -> HTTPGateway [label="POST /api/chat/stream"];
        HTTPGateway -> FastAPIRoutes [label="Forward Request"];
        FastAPIRoutes -> LangGraphRuntime [label="astream_events(v2)"];
        LangGraphRuntime -> FastAPIRoutes [label="Yield SSE Events"];
        FastAPIRoutes -> SSEHook [label="SSE Event Stream (text/event-stream)"];
        SSEHook -> ToolVisualizer [label="event: tool_call"];
        SSEHook -> HITLModal [label="event: interrupt"];
        SSEHook -> GenUI [label="event: structured_output"];
        HITLModal -> FastAPIRoutes [label="POST /api/agent/resume (Approval)"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: STREAMING CHAT ---
st.header("2. Real-Time Streaming Chat in React: SSE Consumption and useChat Hooks", divider="rainbow")
st.markdown("""
Consuming token streams in React requires handling Server-Sent Events without leaking memory.
Using either `@microsoft/fetch-event-source` or Vercel AI SDK's `useChat`, React components maintain an append-only
message list, updating the assistant message as each token arrives.
""")

col_code1, col_code2 = st.columns(2)
with col_code1:
    st.markdown("### Client-Side Streaming Principles")
    st.markdown("""
    - **Buffer Flushes**: Flush incoming chunks into React state with debouncing or `requestAnimationFrame` for high-frequency token streams.
    - **Error Recovery**: Automatically reconnect on network drops with `fetchEventSource`, passing the last message ID.
    - **AbortController**: Bind a cancel button to `controller.abort()` to allow users to halt generation immediately.
    """)
with col_code2:
    st.code("""
// React useChat Streaming Hook Pattern
import React, { useState } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';

export function useAgentChat(threadId) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = async (prompt) => {
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: 'user', content: prompt }]);

    let assistantText = '';
    await fetchEventSource('/api/v1/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: prompt, thread_id: threadId }),
      onmessage(event) {
        if (event.data === '[DONE]') return;
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'token') {
          assistantText += parsed.content;
          setMessages((prev) => [
            ...prev.slice(0, -1),
            { role: 'assistant', content: assistantText }
          ]);
        }
      },
      onclose() { setIsStreaming(false); }
    });
  };

  return { messages, sendMessage, isStreaming };
}
    """, language="javascript")

# --- SECTION 3: AGENT UI ---
st.header("3. Modern Agent UI Architecture: Thread Management, Markdown, and Generative UI", divider="rainbow")
st.markdown("""
Agent interfaces differ from standard chatbots because responses contain mixed content: raw text, source citations,
code snippets with syntax highlighting, and interactive data widgets.
A robust Agent UI parses markdown and renders domain-specific components directly into the chat flow.
""")

st.code("""
// Message Item Component with Rich Markdown & Code Blocks
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';

export const AgentMessageItem = ({ message }) => {
  const isUser = message.role === 'user';
  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} my-3`}>
      <div className={`p-4 rounded-xl max-w-2xl ${isUser ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-100'}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter language={match[1]} PreTag="div" {...props}>
                  {String(children).replace(/\\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code className="bg-slate-900 px-1 py-0.5 rounded text-amber-300" {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {message.content}
        </ReactMarkdown>
      </div>
    </div>
  );
};
""", language="javascript")

# --- SECTION 4: TOOL-CALL VISUALIZATION ---
st.header("4. Tool-Call Visualization & Execution Badging: Status, Arguments, and Trace Trees", divider="rainbow")
st.markdown("""
When an agent invokes tools (such as searching a database or running code), users must not be left staring at a blank screen.
Real-time tool-call cards inform users about ongoing actions, parameter arguments, and results.
""")

st.code("""
// Tool Call Status Card Component
import React, { useState } from 'react';

export const ToolCallCard = ({ toolName, inputArgs, output, status }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="border border-slate-700 bg-slate-900 rounded-lg p-3 my-2 text-sm">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex items-center gap-2">
          <span className="animate-spin text-blue-400">{status === 'running' ? '⚙️' : '✅'}</span>
          <span className="font-semibold text-slate-200">Tool: {toolName}</span>
        </div>
        <span className={`text-xs px-2 py-1 rounded font-mono ${status === 'completed' ? 'bg-green-900 text-green-300' : 'bg-amber-900 text-amber-300'}`}>
          {status.toUpperCase()}
        </span>
      </div>
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-slate-800 space-y-2">
          <div>
            <span className="text-xs text-slate-400">Arguments:</span>
            <pre className="bg-slate-950 p-2 rounded text-xs text-blue-300 overflow-x-auto">{JSON.stringify(inputArgs, null, 2)}</pre>
          </div>
          {output && (
            <div>
              <span className="text-xs text-slate-400">Output:</span>
              <pre className="bg-slate-950 p-2 rounded text-xs text-emerald-300 overflow-x-auto">{JSON.stringify(output, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
""", language="javascript")

# --- SECTION 5: HUMAN APPROVAL UI ---
st.header("5. Human-in-the-Loop Approval UI: Interrupt Interception, Review Modals, and Resuming", divider="rainbow")
st.markdown("""
When LangGraph raises an `interrupt()`, the execution state is safely serialized to the database checkpointer,
and the FastAPI backend pushes an interrupt event to React. The React client intercepts this event and renders an
approval modal displaying the sensitive action details. Once the human confirms or edits parameters, React posts
the resumption payload back to `/api/agent/resume`.
""")

st.code("""
// Human Approval Modal Dialog in React
import React from 'react';

export const HumanApprovalModal = ({ interruptData, onApprove, onReject }) => {
  if (!interruptData) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-900 border border-amber-500/50 rounded-xl max-w-lg w-full p-6 shadow-2xl">
        <h3 className="text-lg font-bold text-amber-400 flex items-center gap-2">
          ⚠️ Action Requires Human Authorization
        </h3>
        <p className="text-sm text-slate-300 mt-2">
          The agent has halted execution awaiting explicit user sign-off for the following tool call:
        </p>
        <div className="bg-slate-950 p-3 rounded my-4 border border-slate-800 text-xs font-mono text-slate-200">
          <div><b>Tool:</b> {interruptData.tool}</div>
          <div className="mt-1"><b>Parameters:</b> {JSON.stringify(interruptData.args, null, 2)}</div>
        </div>
        <div className="flex justify-end gap-3 mt-4">
          <button onClick={onReject} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-sm">
            Reject & Abort
          </button>
          <button onClick={onApprove} className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded text-sm">
            Confirm & Execute
          </button>
        </div>
      </div>
    </div>
  );
};
""", language="javascript")

# --- SECTION 6: STRUCTURED OUTPUT UI ---
st.header("6. Dynamic Structured Output UI: Rendering Pydantic JSON Schemas into Native React Components", divider="rainbow")
st.markdown("""
Instead of forcing the LLM to return plain text, LangChain's `.with_structured_output()` emits typed JSON data.
In React, a component registry maps the emitted schema `type` to rich visual components (e.g., interactive data tables,
metric cards, or charts). This architecture is known as **Generative UI**.
""")

st.code("""
// Generative UI Dynamic Component Registry
import React from 'react';

const ComponentRegistry = {
  RevenueBarChart: ({ data }) => (
    <div className="bg-slate-950 p-4 rounded border border-blue-500/30">
      <h4 className="text-xs uppercase text-blue-400 font-bold mb-2">Quarterly Breakdown</h4>
      {data.map((item) => (
        <div key={item.label} className="flex justify-between py-1 text-sm border-b border-slate-800">
          <span>{item.label}</span>
          <span className="font-mono text-emerald-400">${item.value.toLocaleString()}</span>
        </div>
      ))}
    </div>
  ),
  AlertBadge: ({ level, message }) => (
    <div className={`p-3 rounded text-sm ${level === 'CRITICAL' ? 'bg-red-950 text-red-200 border border-red-800' : 'bg-blue-950 text-blue-200'}`}>
      <b>[{level}]</b> {message}
    </div>
  )
};

export const StructuredOutputRenderer = ({ payload }) => {
  const Component = ComponentRegistry[payload.component];
  if (!Component) return <pre>{JSON.stringify(payload, null, 2)}</pre>;
  return <Component {...payload} />;
};
""", language="javascript")

# --- SECTION 7: PRODUCTION IMPLEMENTATION ---
st.header("7. Production Implementation: Full-Stack React + LangChain Contract and Component Engine", divider="rainbow")
st.markdown("""
Below is the full architecture contract between a FastAPI backend and a Next.js / React client, establishing
typed SSE event contracts for tokens, tool calls, interrupts, and generative UI widgets.
""")

st.code("""
/* Full-Stack Communication Contract
 * Backend emits events formatted as:
 * event: [token | tool_start | tool_end | interrupt | structured_ui]
 * data: { payload_object }
 */

// Typescript Contract Definitions
export interface AgentTokenEvent {
  type: 'token';
  content: string;
}

export interface AgentToolEvent {
  type: 'tool_start' | 'tool_end';
  tool: string;
  args?: Record<string, any>;
  output?: string;
}

export interface AgentInterruptEvent {
  type: 'interrupt';
  thread_id: string;
  checkpoint_id: string;
  tool: string;
  args: Record<string, any>;
}

export interface AgentStructuredUIEvent {
  type: 'structured_ui';
  component: string;
  props: Record<string, any>;
}
""", language="typescript")

# --- SECTION 8: INTERACTIVE WORKBENCH ---
st.header("8. Interactive Workbench: React Agent State Machine & Tool-Call Visualizer", divider="rainbow")
st.markdown("""
Simulate how a React frontend renders multi-step agent trajectories, tool execution cards,
and Human-in-the-Loop approval dialogs based on backend SSE event streams.
""")

react_sim = ReactAgentStateSimulator()

col_config, col_preview = st.columns([1, 1])

with col_config:
    st.subheader("⚙️ Scenario Selection")
    chosen_run = st.selectbox(
        "Choose Agent Execution Scenario:",
        list(react_sim.preset_agent_runs.keys())
    )
    
    st.caption("Human Approval Controls (For Interrupted Workflows):")
    approve_action = st.checkbox("Simulate Human Authorization ('Approve & Execute')", value=False)
    render_sim_btn = st.button("🚀 Render React Component State", type="primary")

with col_preview:
    st.subheader("⚛️ Emulated React Client View")
    state = react_sim.simulate_client_state(chosen_run, approved=approve_action)
    
    # User message card
    st.markdown(f"**User Prompt**: `{state['user_prompt']}`")
    
    # Tool call visualizations
    st.caption("Active Tool Call Components:")
    for tool in state["tool_calls"]:
        status_color = "#10B981" if "approved" in tool["status"] or tool["status"] == "completed" else "#F59E0B"
        st.markdown(
            f"<div class='react-component'>"
            f"<span class='tool-badge'>{tool['tool']}</span> — <b>Status:</b> <span style='color:{status_color}'>{tool['status'].upper()}</span>"
            f"<br><small>Arguments: <code>{json.dumps(tool['args'])}</code></small>"
            f"<br><small>Output: <code>{tool['output']}</code></small>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    # Human in the Loop Modal Banner
    if state["requires_human_approval"]:
        st.error("⚠️ Modal Alert: Action suspended at checkpointer. Awaiting Human Review.")
    else:
        st.success(f"Assistant Completion: {state['final_message']}")
        
    # Generative UI Structured Component
    if "structured_output" in state:
        with st.expander("🖼️ Generative UI Widget Rendered", expanded=True):
            st.json(state["structured_output"])

# --- SECTION 9: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("9. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: Why is `@microsoft/fetch-event-source` preferred over native `EventSource` in React applications?"):
    st.markdown("""
    **Answer**: The browser's native `EventSource` API only supports `GET` requests and cannot send custom request headers (such as `Authorization: Bearer <JWT>`).
    LangChain agent endpoints require `POST` requests with complex JSON request bodies (prompt, thread ID, configuration parameters) and custom auth headers.
    `@microsoft/fetch-event-source` supports `POST` bodies, custom headers, and sophisticated retry policies.
    """)

with st.expander("Q2: What is 'Generative UI' and how does LangChain enable it with React?"):
    st.markdown("""
    **Answer**: Generative UI refers to the LLM returning structured JSON data (enforced via Pydantic or JSON schema) instead of raw text,
    which the React frontend maps to client components (charts, forms, data tables, booking widgets). LangChain's `.with_structured_output()`
    emits typed objects that the frontend can inspect by type and render natively.
    """)

with st.expander("Q3: How does React resume execution after a LangGraph `interrupt()`?"):
    st.markdown("""
    **Answer**: When LangGraph encounters an `interrupt()`, it halts and checkpoints state, returning an interrupt payload to the client.
    The React frontend renders an approval UI. When the user clicks 'Approve', React dispatches a `POST /api/resume` request with the `thread_id`
    and the human decision. The backend loads the checkpoint via checkpointer, passes the input to `.ainvoke(Command(resume=...))`, and continues the graph.
    """)

with st.expander("Q4: How do you prevent React UI re-render lag during high-frequency token streaming?"):
    st.markdown("""
    **Answer**: High-frequency LLMs can emit 50 to 100 token chunks per second. Updating React state on every single token causes excessive DOM reconciliation
    and visual lag. Production apps use ref-based token accumulation, batching state updates with `requestAnimationFrame` or debouncing state dispatches
    every 50-100 milliseconds.
    """)

with st.expander("Q5: How do you handle client network disconnections during long-running agent tool execution?"):
    st.markdown("""
    **Answer**: Decouple agent execution from the HTTP connection. The backend continues executing the LangGraph checkpoint in the background.
    When the React client reconnects using the `thread_id`, the frontend queries the checkpointer state endpoint (`GET /api/thread/{id}/state`)
    to retrieve completed tool results and resumes streaming from the current node.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Using GET EventSource with Secrets in URLs)**: Using standard `EventSource` and appending sensitive user queries and tokens to query params. Query strings are logged in plaintext across proxies, access logs, and browser history.")
    st.warning("**Trap 2 (Uncontrolled Re-render Freezes)**: Updating a complex React component tree on every single LLM token chunk without memoization or state batching, freezing the browser thread.")
with col_trap2:
    st.error("**Trap 3 (Stateless Client-Side Approval Resumption)**: Sending an approval resumption request without passing the server-verified `checkpoint_id`. If the server state moved or timed out, resuming executes an outdated state.")
    st.warning("**Trap 4 (Missing Stream Abort Handlers)**: Failing to trigger `AbortController.abort()` when users navigate away or close tabs, leaving orphan backend agent loops running and wasting LLM tokens.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: Why can't native `EventSource` be used for authenticated LangChain endpoints?")
        st.caption("Flip for answer:")
        st.info("**Back**: Native `EventSource` only supports `GET` requests and cannot attach `Authorization: Bearer` headers.")
    with st.container(border=True):
        st.markdown("**Front**: What is the core mechanism of Generative UI?")
        st.caption("Flip for answer:")
        st.info("**Back**: Returning typed JSON schemas from the LLM and dynamically mapping schema types to native React components.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: How does React notify LangGraph to proceed past an interrupt?")
        st.caption("Flip for answer:")
        st.info("**Back**: By sending a resume command with the thread ID and user input to `Command(resume=payload)`.")
    with st.container(border=True):
        st.markdown("**Front**: How do production React frontends prevent token render lag?")
        st.caption("Flip for answer:")
        st.info("**Back**: By buffering tokens and batching state updates with `requestAnimationFrame` instead of updating state per token.")

# --- SECTION 10: SELF-ASSESSMENT CHECKLIST ---
st.header("10. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I understand why @microsoft/fetch-event-source is preferred over native EventSource for AI streaming.", value=True)
c2 = st.checkbox("I can implement React streaming hooks that update message state incrementally.", value=True)
c3 = st.checkbox("I know how to design tool-call visualization components showing status, inputs, and outputs.", value=True)
c4 = st.checkbox("I understand the client-server contract for handling LangGraph interrupt() dialogs.", value=True)
c5 = st.checkbox("I can implement Generative UI by mapping structured JSON schema payloads to React widgets.", value=True)
c6 = st.checkbox("I know how to prevent UI lag by buffering incoming token streams with requestAnimationFrame.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 32.4: LangChain + Data Science.")
