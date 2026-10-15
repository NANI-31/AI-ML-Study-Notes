import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union, Annotated, TypedDict

# ==============================================================================
# PAGE CONFIGURATION & ROOT STYLING
# ==============================================================================
st.set_page_config(
    page_title="26.3 LangGraph Frontend - LangGraph Curriculum",
    page_icon="⚛️",
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
    .badge-sdk {
        background-color: #022C22;
        color: #34D399;
        border: 1px solid #059669;
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
class UseStreamLifecycleSimulator:
    """
    Simulates the official useStream React hook lifecycle:
    - connection initialization
    - stream event chunk processing
    - optimistic state binding
    - interrupt detection and resolution
    """
    def __init__(self):
        self.hook_state = {
            "status": "idle",  # idle, streaming, paused_interrupt, error
            "threadId": "thread_demo_882",
            "messages": [
                {"id": "m1", "type": "human", "content": "Generate a sales forecast chart for Q4."}
            ],
            "values": {
                "forecast_data": [120, 145, 180, 210],
                "confidence_score": 0.94
            },
            "interrupt": None
        }

    def simulate_user_send(self, new_query: str) -> None:
        self.hook_state["status"] = "streaming"
        self.hook_state["messages"].append({
            "id": f"m{len(self.hook_state['messages'])+1}",
            "type": "human",
            "content": new_query
        })
        self.hook_state["messages"].append({
            "id": f"m{len(self.hook_state['messages'])+1}",
            "type": "ai",
            "content": "Analyzing Q4 performance trends and synthesizing projections..."
        })
        self.hook_state["status"] = "idle"

    def simulate_interrupt(self) -> None:
        self.hook_state["status"] = "paused_interrupt"
        self.hook_state["interrupt"] = {
            "action": "export_financial_pdf",
            "target_format": "A4",
            "requires_signature": True
        }

    def resolve_interrupt(self) -> None:
        self.hook_state["status"] = "idle"
        self.hook_state["interrupt"] = None
        self.hook_state["messages"].append({
            "id": f"m{len(self.hook_state['messages'])+1}",
            "type": "ai",
            "content": "Financial report PDF exported successfully."
        })

us_sim = UseStreamLifecycleSimulator()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: LangGraph Frontend SDK Architecture, useStream Primitives, and State Binding", divider="rainbow")

st.markdown("""
The official `@langchain/langgraph-sdk` provides purpose-built abstractions for integrating LangGraph deployments 
into React and Next.js applications. At the core is the `useStream` hook, which eliminates boilerplate code 
by managing WebSocket/SSE connections, accumulating token deltas, synchronizing graph channel values, 
and exposing reactive triggers for human-in-the-loop approvals.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph FrontendSDKBlueprint {
        graph [bgcolor="#0B1120", rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6];
        node [fontname="Helvetica", shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="#F8FAFC", color="#334155", margin="0.2,0.1"];
        edge [fontname="Helvetica", color="#38BDF8", fontcolor="#94A3B8", fontsize=10];

        subgraph cluster_hook {
            label="1. React Application Layer (useStream Hook)";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            use_stream [label="const stream = useStream({\\n  apiUrl: '/api',\\n  assistantId: 'analyst_agent'\\n})", fillcolor="#0F172A"];
            hook_state [label="Reactive Hook State Returns:\\n- stream.messages (Message[])\\n- stream.values (GraphState)\\n- stream.interrupt (InterruptPayload)\\n- stream.isLoading (boolean)", fillcolor="#022C22", fontcolor="#6EE7B7"];
            hook_methods [label="Exposed Actions:\\n- stream.submit(input)\\n- stream.resume(value)\\n- stream.stop()", fillcolor="#1E293B"];
        }

        subgraph cluster_ui {
            label="2. Generative UI & View Components";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            chat_view [label="ChatFeed & Message Renderer\\nMarkdown, Thought Blocks, Source Citations", fillcolor="#1E293B"];
            state_view [label="Graph State Inspector / Canvas\\nLive Charts, Form Inputs, JSON Delta Tree", fillcolor="#1E293B"];
            interrupt_view [label="Interrupt Modal / Action Banner\\nCalls stream.resume() on Submission", fillcolor="#831843", fontcolor="#FBCFE8"];
        }

        subgraph cluster_server {
            label="3. LangGraph Server / Cloud Backend";
            style=dashed;
            color="#475569";
            fontcolor="#94A3B8";

            lg_server [label="LangGraph Server Ingress\\nPOST /threads/{id}/runs/stream", fillcolor="#312E81"];
            checkpointer [label="PostgresSaver Checkpoint Store\\nState Snapshots & Interrupt Metadata", fillcolor="#064E3B", fontcolor="#A7F3D0"];
        }

        use_stream -> hook_state;
        use_stream -> hook_methods;

        hook_state -> chat_view;
        hook_state -> state_view;
        hook_state -> interrupt_view;

        hook_methods -> lg_server [label="HTTP/2 SSE Pipeline"];
        lg_server -> checkpointer;
        lg_server -> use_stream [label="SSE Event Multiplex"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# ==============================================================================
# SECTION 2: useStream
# ==============================================================================
st.header("2. useStream: The Official React Hook for Connecting to LangGraph Server Deployments", divider="rainbow")

st.markdown("""
The `useStream` hook connects a React component to a remote LangGraph assistant. 
It encapsulates the full lifecycle of an agent execution:

### Core Properties & Capabilities:
- **`messages`**: Array of conversation messages, automatically merging streamed token chunks into the active message.
- **`values`**: The current snapshot of the graph's state channels (such as `plan`, `documents`, `chart_data`).
- **`interrupt`**: The active interruption payload if the graph paused on `interrupt()`; `null` otherwise.
- **`submit(inputs)`**: Dispatches a new execution run to the server.
- **`resume(payload)`**: Submits the resolution value to unblock an interrupted run.
""")

st.code('''import { useStream } from "@langchain/langgraph-sdk/react";

export function AnalystDashboard({ threadId }: { threadId: string }) {
  const { messages, values, interrupt, submit, resume, isLoading } = useStream({
    apiUrl: "https://api.my-agent.com",
    assistantId: "financial_analyst",
    threadId: threadId,
  });

  return (
    <div>
      <ChatFeed messages={messages} />
      {interrupt && <ApprovalModal interrupt={interrupt} onResolve={resume} />}
      <InputBar onSubmit={(text) => submit({ messages: [{ role: "user", content: text }] })} disabled={isLoading} />
    </div>
  );
}
''', language="typescript")

# ==============================================================================
# SECTION 3: Graph State Rendering
# ==============================================================================
st.header("3. Graph State Rendering: Real-Time State Reflection, Channel Visualization, and Custom Reducer Tracking", divider="rainbow")

st.markdown("""
In many enterprise agent applications, displaying chat text alone is insufficient. 
Agents produce structured state: research briefs, data tables, generated diagrams, or multi-step plans.

Because `useStream` exposes `stream.values`, frontends can bind UI components directly to graph state channels:
- When a `planner` node updates `values.plan`, a side-panel checklist re-renders immediately.
- When a `retriever` node populates `values.sources`, a citation drawer updates in real time.
- When an `analyst` node produces `values.kpis`, interactive metric cards refresh dynamically.
""")

# ==============================================================================
# SECTION 4: Message Rendering
# ==============================================================================
st.header("4. Message Rendering: Rich Chat Bubbles, Artifact Cards, Generative UI, and Message Deltas", divider="rainbow")

st.markdown("""
Modern agent interfaces utilize **Generative UI**—transforming structured model outputs and tool responses into 
rich, interactive components rather than raw markdown blocks.

### Message Types:
- **HumanMessage**: Standard user prompt bubble.
- **AIMessage**: Assistant response supporting token streaming, reasoning disclosure accordions, and markdown.
- **ToolMessage**: Structured cards showing external tool executions, interactive maps, or data tables.
""")

st.code('''// Dynamic Component Rendering Based on Message Types
function MessageItem({ message }: { message: any }) {
  if (message.type === "human") {
    return <UserBubble content={message.content} />;
  }
  if (message.type === "tool") {
    return <ToolCard name={message.name} output={message.content} />;
  }
  return (
    <AssistantBubble content={message.content}>
      {message.additional_kwargs?.generative_ui && (
        <GenerativeChart data={message.additional_kwargs.generative_ui} />
      )}
    </AssistantBubble>
  );
}
''', language="typescript")

# ==============================================================================
# SECTION 5: Node Metadata
# ==============================================================================
st.header("5. Node Metadata: Displaying Execution Timestamps, Model Names, Token Counts, and Run IDs", divider="rainbow")

st.markdown("""
Every event emitted by LangGraph Server contains rich metadata. Exposing this metadata in the UI 
provides observability and debugging visibility for developers and enterprise power users:

- **Run ID & Thread ID**: Click-to-copy identifiers for customer support and LangSmith audit lookups.
- **Model Attribution**: Badges indicating which model generated each turn (e.g., `gpt-4o` vs `claude-3-5-sonnet`).
- **Token Usage**: Granular breakdown of prompt and completion tokens for the message.
- **Step Latency**: Precise milliseconds elapsed during the superstep.
""")

# ==============================================================================
# SECTION 6: Streaming UI
# ==============================================================================
st.header("6. Streaming UI: Composing Stream-Aware Components, Skeleton Loaders, and Auto-Scroll Containers", divider="rainbow")

st.markdown("""
Building smooth streaming interfaces requires addressing several frontend UX challenges:

1. **Auto-Scroll Management**: Chat containers must automatically stick to the bottom during active token generation, 
   but allow the user to scroll up without annoying scroll-snapping.
2. **Skeleton Loaders**: Show shimmering skeleton placeholders while awaiting initial token response (TTFT).
3. **Optimistic Message Merging**: Display optimistic human input instantly while preserving message ordering.
""")

# ==============================================================================
# SECTION 7: Interactive Agent Interfaces
# ==============================================================================
st.header("7. Interactive Agent Interfaces: Canvas Interfaces, Human Feedback Forms, and Collaborative Co-Pilot Workflows", divider="rainbow")

st.markdown("""
Advanced agents move beyond linear chat windows toward **Canvas / Artifact interfaces** (similar to Claude Artifacts 
or OpenAI Canvas). In a canvas interface:
- The left pane contains conversational chat history.
- The right pane renders an editable workspace artifact (code file, document draft, interactive spreadsheet).
- The agent updates the artifact state via graph channels, and human edits sync back to the agent via `update_state()`.
""")

# ==============================================================================
# SECTION 8: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("8. Production Implementation: Full Next.js / React Client using @langchain/langgraph-sdk and useStream", divider="rainbow")

st.markdown("""
Below is a complete, production-grade Next.js client component demonstrating the use of `useStream` 
with generative UI, state inspection, and interrupt handling.
""")

st.code('''// app/agent/page.tsx - Complete Production Next.js Agent UI
"use client";

import React, { useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";

export default function AgentPage() {
  const [input, setInput] = useState("");
  
  const { messages, values, interrupt, submit, resume, isLoading } = useStream({
    apiUrl: process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://localhost:8123",
    assistantId: "enterprise_agent",
    threadId: "session_user_401"
  });

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    submit({ messages: [{ role: "user", content: input }] });
    setInput("");
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      {/* Main Chat Feed */}
      <div className="flex-1 flex flex-col p-6 border-r border-slate-800">
        <div className="flex-1 overflow-y-auto space-y-4">
          {messages.map((m: any) => (
            <div key={m.id} className={`p-4 rounded-lg ${m.type === "human" ? "bg-slate-800 ml-auto max-w-lg" : "bg-slate-900 mr-auto max-w-2xl"}`}>
              <div className="text-xs text-slate-500 mb-1">{m.type.toUpperCase()}</div>
              <div>{m.content}</div>
            </div>
          ))}
          {isLoading && <div className="text-sm text-cyan-400 animate-pulse">Agent reasoning in progress...</div>}
        </div>

        {/* Interrupt Banner */}
        {interrupt && (
          <div className="p-4 bg-purple-950 border border-purple-600 rounded-lg my-4">
            <h4 className="font-semibold text-purple-200">Human Approval Required</h4>
            <p className="text-sm text-purple-300">Action: {interrupt.action}</p>
            <div className="mt-3 flex gap-3">
              <button onClick={() => resume({ approved: true })} className="px-4 py-2 bg-emerald-600 rounded text-white font-medium">Approve</button>
              <button onClick={() => resume({ approved: false })} className="px-4 py-2 bg-rose-600 rounded text-white font-medium">Reject</button>
            </div>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSend} className="mt-4 flex gap-2">
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask the agent..." className="flex-1 bg-slate-900 border border-slate-700 rounded px-4 py-2" />
          <button type="submit" disabled={isLoading} className="px-6 py-2 bg-cyan-600 rounded font-semibold disabled:opacity-50">Send</button>
        </form>
      </div>

      {/* Side Panel: Live Graph State */}
      <div className="w-96 p-6 bg-slate-900 overflow-y-auto">
        <h3 className="font-bold text-lg mb-4 text-slate-200">Live Graph State (values)</h3>
        <pre className="bg-slate-950 p-4 rounded text-xs text-cyan-300 overflow-x-auto">
          {JSON.stringify(values, null, 2)}
        </pre>
      </div>
    </div>
  );
}
''', language="typescript")

# ==============================================================================
# SECTION 9: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("9. Interactive Workbench: useStream Lifecycle Emulator & Generative UI Renderer", divider="rainbow")

st.markdown("""
Simulate the official `useStream` hook state, test user message submissions, and observe real-time channel updates.
""")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("React useStream Client Simulator")
    st.markdown(f"**Connected Thread ID:** `{us_sim.hook_state['threadId']}`")
    st.markdown(f"**Hook Status:** `{us_sim.hook_state['status'].upper()}`")
    
    user_query = st.text_input("Simulate User Submission via submit()", value="Calculate revised Q4 forecast.")
    if st.button("Dispatch stream.submit()", type="primary"):
        us_sim.simulate_user_send(user_query)
        st.success("Dispatched run to LangGraph Server!")
        st.rerun()

    st.markdown("---")
    st.markdown("#### Interrupt Simulation")
    if us_sim.hook_state["status"] == "paused_interrupt":
        st.warning("⚠️ interrupt detected in hook state!")
        st.json(us_sim.hook_state["interrupt"])
        if st.button("Dispatch stream.resume({ approved: true })"):
            us_sim.resolve_interrupt()
            st.success("Resumed interrupted graph execution.")
            st.rerun()
    else:
        if st.button("Trigger Interrupt"):
            us_sim.simulate_interrupt()
            st.rerun()

with col_right:
    st.subheader("Reactive UI Viewport")
    st.markdown("#### Message Feed (`stream.messages`)")
    for msg in us_sim.hook_state["messages"]:
        with st.chat_message(msg["type"]):
            st.write(msg["content"])
            
    st.markdown("#### Graph State Channels (`stream.values`)")
    st.json(us_sim.hook_state["values"])

# ==============================================================================
# SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS
# ==============================================================================
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

st.markdown("### Technical Interview Q&A")

with st.expander("Q1: What are the primary advantages of useStream over manual fetch or EventSource implementations?"):
    st.markdown("""
    **Answer:**  
    `useStream` automatically handles:  
    1. **Reconnection & State Resumption**: Tracks stream progress and manages connection drops without losing local state.  
    2. **State Channel Synchronization**: Synchronizes both `messages` and arbitrary state channels (`values`) simultaneously.  
    3. **Interrupt Lifecycle**: Automatically extracts `interrupt` payloads and provides the `resume()` handler to unblock runs.
    """)

with st.expander("Q2: How does useStream maintain optimistic updates while token streaming is in progress?"):
    st.markdown("""
    **Answer:**  
    When `submit()` is invoked, `useStream` immediately appends an optimistic message to the local `messages` state with a temporary client ID. 
    As the server begins streaming, it appends a streaming assistant placeholder and reconciles incoming tokens into it. Upon completion, 
    the server's checkpointed message IDs overwrite the temporary records.
    """)

with st.expander("Q3: How do you implement Generative UI with LangGraph tool messages?"):
    st.markdown("""
    **Answer:**  
    Tools return structured JSON payloads rather than raw strings. In the frontend component registry, map each tool name 
    (e.g., `generate_chart`, `book_flight`) to a corresponding React component (`<FlightPickerCard />`, `<InteractiveChart />`). 
    When a message with `type: "tool"` is rendered, the registry renders the rich component hydrated with tool arguments and results.
    """)

with st.expander("Q4: What is the difference between assistantId and threadId in the LangGraph SDK client?"):
    st.markdown("""
    **Answer:**  
    - `assistantId`: Identifies the compiled graph definition or deployment template (e.g., `"customer_support"`, `"code_reviewer"`).  
    - `threadId`: Identifies an individual conversation session instance, scoping memory and checkpoints to a specific user context.
    """)

with st.expander("Q5: How can a user cancel an active LangGraph execution stream in the browser?"):
    st.markdown("""
    **Answer:**  
    By invoking `stream.stop()`. This aborts the underlying browser `AbortController` signal, closing the HTTP connection, 
    and sends a cancellation request to `POST /threads/{id}/runs/{run_id}/cancel` to terminate backend worker processing.
    """)

st.markdown("### Production Traps")

st.error("""
**Trap 1: Hardcoding Assistant IDs in Client Bundles**  
Hardcoding production assistant UUIDs into frontend code causes breaking deployments when updating graph versions. 
Always look up assistant IDs via environment variables or dynamic deployment tags.
""")

st.error("""
**Trap 2: Double Submission Race Conditions**  
Failing to disable the user input button while `isLoading` is true allows users to trigger concurrent runs on the same thread, 
resulting in state collision or 409 Conflict errors from the server.
""")

st.warning("""
**Trap 3: Not Handling stream.error in UI**  
Ignoring `stream.error` leaves the user interface in a permanent spinning state if the backend suffers a rate limit or timeout. 
Always render an explicit retry banner when errors occur.
""")

st.warning("""
**Trap 4: Forgetting Thread Persistence Across Route Navigations**  
Failing to persist `threadId` in local storage or URL query parameters resets conversation state when the user refreshes or navigates pages.
""")

st.markdown("### Concept Flashcards")

f_col1, f_col2 = st.columns(2)
with f_col1:
    st.markdown("""
    <div class="metric-card">
        <h4>useStream Hook</h4>
        <p>The official React hook connecting frontend components to LangGraph Server streaming runs and state channels.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>Generative UI</h4>
        <p>Dynamic rendering of interactive React components hydrated by structured model tools and state channels.</p>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="metric-card">
        <h4>stream.values</h4>
        <p>The reactive state property exposing non-message graph channels (such as plans, charts, or document arrays) to components.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <h4>stream.resume()</h4>
        <p>The hook method used to resolve paused interrupt() states by submitting human approval data back to the server.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SECTION 11: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("11. Self-Assessment Checklist", divider="rainbow")

chk_1 = st.checkbox("I understand the architecture and properties of the official useStream React hook.", value=True)
chk_2 = st.checkbox("I know how to render real-time graph state channels (values) alongside conversational messages.", value=True)
chk_3 = st.checkbox("I can build Generative UI components hydrated by tool execution payloads.", value=True)
chk_4 = st.checkbox("I understand how to display execution metadata (timestamps, models, tokens, run IDs).", value=True)
chk_5 = st.checkbox("I know how to implement streaming UI patterns (auto-scrolling, skeleton loaders, typing indicators).", value=True)
chk_6 = st.checkbox("I can handle human-in-the-loop approval workflows using interrupt and stream.resume().", value=True)

if chk_1 and chk_2 and chk_3 and chk_4 and chk_5 and chk_6:
    st.success("LangGraph Frontend SDK and useStream mastery confirmed! Ready for Chapter 27 Databases and RAG.")
