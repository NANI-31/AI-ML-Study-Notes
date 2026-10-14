import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="23.1 Callbacks - LangChain Master Curriculum",
    page_icon="📡",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LCELCallbackDispatcherEngine:
    """
    Enterprise LCEL Callback Dispatcher & Event Lifecycle Engine.
    Simulates real-time event streaming across all 5 core LangChain callback interfaces:
    1. BaseCallbackHandler event emission (Lifecycle orchestration)
    2. Model Callbacks (on_llm_start, on_llm_new_token, on_llm_end)
    3. Chain Callbacks (on_chain_start, on_chain_end, on_chain_error)
    4. Tool Callbacks (on_tool_start, on_tool_end, on_tool_error)
    5. Agent Callbacks (on_agent_action, on_agent_finish)
    """
    def __init__(self, handler_name: str = "production-telemetry-dispatcher"):
        self.handler_name = handler_name
        self.dispatched_events: List[Dict[str, Any]] = []

    def dispatch_execution_pipeline(
        self,
        query: str,
        include_tool_execution: bool = True,
        simulate_error: bool = False,
        streaming_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Dispatches an LCEL execution flow and records the sequential callback lifecycle events.
        """
        events = []

        # 1. Chain Start
        events.append({
            "event_type": "on_chain_start",
            "component": "CustomerSupportPipeline",
            "phase": "Chain Ingress",
            "payload": {"query": query},
            "timestamp_offset_ms": 0.0
        })

        # 2. Tool Invocation (if applicable)
        if include_tool_execution:
            events.append({
                "event_type": "on_tool_start",
                "component": "crm_database_tool",
                "phase": "Tool Dispatch",
                "payload": {"action": "lookup", "account_id": "acc_881"},
                "timestamp_offset_ms": 45.0
            })

            if simulate_error:
                events.append({
                    "event_type": "on_tool_error",
                    "component": "crm_database_tool",
                    "phase": "Tool Crash",
                    "payload": {"error": "HTTP 504 Gateway Timeout (CRM connection pool exhausted)"},
                    "timestamp_offset_ms": 120.0
                })
                events.append({
                    "event_type": "on_chain_error",
                    "component": "CustomerSupportPipeline",
                    "phase": "Chain Failure",
                    "payload": {"error": "Pipeline terminated due to unhandled tool timeout"},
                    "timestamp_offset_ms": 125.0
                })
                return {
                    "status": "TERMINATED_WITH_ERROR",
                    "total_events": len(events),
                    "events": events,
                    "final_output": None
                }
            else:
                events.append({
                    "event_type": "on_tool_end",
                    "component": "crm_database_tool",
                    "phase": "Tool Success",
                    "payload": {"status": "active", "tier": "enterprise", "balance": "0 USD"},
                    "timestamp_offset_ms": 95.0
                })

        # 3. Model Start
        events.append({
            "event_type": "on_llm_start",
            "component": "ChatOpenAI (gpt-4o)",
            "phase": "Model Prefill",
            "payload": {"model": "gpt-4o", "temperature": 0.0},
            "timestamp_offset_ms": 110.0
        })

        # 4. Streaming Token Callback Events
        sample_tokens = ["Your ", "enterprise ", "account ", "is ", "active ", "and ", "in ", "good ", "standing."]
        if streaming_enabled:
            for idx, token in enumerate(sample_tokens):
                events.append({
                    "event_type": "on_llm_new_token",
                    "component": "ChatOpenAI",
                    "phase": "Token Stream",
                    "payload": {"token": token, "index": idx},
                    "timestamp_offset_ms": round(150.0 + (idx * 25.0), 1)
                })

        # 5. Model End
        full_text = "".join(sample_tokens)
        events.append({
            "event_type": "on_llm_end",
            "component": "ChatOpenAI",
            "phase": "Model Complete",
            "payload": {"output_text": full_text, "total_tokens": 142},
            "timestamp_offset_ms": 380.0
        })

        # 6. Chain End
        events.append({
            "event_type": "on_chain_end",
            "component": "CustomerSupportPipeline",
            "phase": "Chain Complete",
            "payload": {"final_response": full_text},
            "timestamp_offset_ms": 385.0
        })

        report = {
            "status": "SUCCESSFUL_EXECUTION",
            "total_events": len(events),
            "events": events,
            "final_output": full_text
        }
        self.dispatched_events.extend(events)
        return report

# Initialize engine in session state
if "lcel_callback_engine" not in st.session_state:
    st.session_state.lcel_callback_engine = LCELCallbackDispatcherEngine()

# --- TOP NAVIGATION & HEADER ---
st.title("23.1 Callbacks in LangChain")
st.caption("Chapter 23: Callbacks, Logging, and Monitoring | Domain: 9. LangChain Orchestration & LCEL")

st.markdown("""
In LangChain, **Callbacks** represent the primary nervous system for observability, streaming, token tracking, 
and external telemetry integration. Rather than hardcoding logging and metrics directly into business logic, 
LangChain's callback system allows engineers to subscribe to lifecycle hooks (`on_llm_start`, `on_chain_end`, `on_tool_error`) 
cleanly and non-invasively across complex LCEL pipelines.

In this lesson, you will master:
1. Fundamental callback concepts: `BaseCallbackHandler`, `AsyncCallbackHandler`, and constructor vs runtime injection
2. Model callbacks: capturing TTFT, streaming token events via `on_llm_new_token`, and completion metrics
3. Chain callbacks: tracking nested LCEL execution trees, input dictionaries, and unhandled errors
4. Tool callbacks: monitoring external API execution timings, payload arguments, and crash events
5. Agent callbacks: observing intermediate reasoning thoughts, action decisions, and goal completions
""")

st.divider()

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: The LangChain Callback Handler Lifecycle Architecture", divider="rainbow")

st.markdown("""
The LangChain callback architecture operates via the **Observer Pattern**. 
When an LCEL pipeline executes, the `CallbackManager` automatically dispatches strongly typed lifecycle events 
to all registered handlers without modifying or interrupting the core runnable data flow.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph CallbackArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, pad="0.4", nodesep="0.6", ranksep="0.7"];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", penwidth=2];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_lcel_run {
            label = "1. Active LCEL Execution Pipeline";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            C_START [label="Pipeline Ingress\\n(prompt | model | parser)", fillcolor="#1E3A8A", color="#3B82F6"];
            T_RUN [label="Tool Invocation\\ntool.invoke({'id': 881})", fillcolor="#1E3A8A", color="#3B82F6"];
            M_RUN [label="Model Invocation\\nchat_model.stream(...)", fillcolor="#047857", color="#10B981"];
            C_END [label="Pipeline Egress\\nReturns Final Typed Response", fillcolor="#064E3B", color="#10B981"];
        }

        subgraph cluster_manager {
            label = "2. LangChain CallbackManager & Event Bus";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#F59E0B";
            fontsize = 12;

            CB_MGR [label="CallbackManager\\n(Propagates via RunnableConfig)", fillcolor="#7C2D12", color="#EA580C"];
        }

        subgraph cluster_handlers {
            label = "3. Registered Callback Handlers";
            style = "filled,rounded";
            color = "#1E293B";
            fillcolor = "#0F172A";
            fontcolor = "#38BDF8";
            fontsize = 12;

            H_STREAM [label="StreamingHandler\\non_llm_new_token -> Web UI Stream", fillcolor="#0E7490", color="#06B6D4"];
            H_TELEMETRY [label="LangSmithHandler / OpenTelemetry\\non_llm_end -> Records Latency & Cost", fillcolor="#0E7490", color="#06B6D4"];
            H_ALERT [label="ErrorAlertHandler\\non_tool_error -> Sends Slack Alert", fillcolor="#881337", color="#F43F5E"];
        }

        C_START -> CB_MGR [label="Fires on_chain_start"];
        T_RUN -> CB_MGR [label="Fires on_tool_start / on_tool_end"];
        M_RUN -> CB_MGR [label="Fires on_llm_new_token / on_llm_end"];
        C_END -> CB_MGR [label="Fires on_chain_end"];

        CB_MGR -> H_STREAM [label="Streams Chunks"];
        CB_MGR -> H_TELEMETRY [label="Dispatches Spans"];
        CB_MGR -> H_ALERT [label="Routes Exceptions"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.markdown("""
### Enterprise Engineering Core Axioms: Callback Discipline
1. **Never Hardcode Side Effects in Business Logic**: Logging, token counting, user streaming, and metrics collection belong strictly in callback handlers. Keeping LCEL chains pure ensures testability and modularity.
2. **Propagate Callbacks via `RunnableConfig` in LCEL**: In modern LangChain (v0.3+), callbacks are injected at runtime via the `config={"callbacks": [...]}` parameter during `.invoke()` or `.stream()`. This allows different callers (e.g., API vs Batch Worker) to attach different handlers to the same chain.
3. **Use `AsyncCallbackHandler` for Non-Blocking I/O**: If a callback handler performs external network requests (e.g., sending metrics to Datadog or Slack), it must inherit from `AsyncCallbackHandler` and implement `async def` methods to avoid blocking the main inference event loop.
""")

st.divider()

# ==============================================================================
# SECTION 2: CALLBACK CONCEPTS
# ==============================================================================
st.header("2. Callback Concepts (BaseCallbackHandler, AsyncCallbackHandler, Constructor vs Runtime Injection)", divider="rainbow")

st.markdown("""
Understanding callback architecture requires mastering the distinction between handler base classes 
and the two mechanisms of callback injection: constructor-level vs runtime-level.
""")

st.markdown("""
### Core Concepts Breakdown
* **`BaseCallbackHandler`**: Synchronous base class implementing default no-op methods for all lifecycle hooks (`on_llm_start`, `on_chain_end`, etc.). Subclasses override only the hooks they need.
* **`AsyncCallbackHandler`**: Asynchronous base class designed for non-blocking I/O operations (`async def on_llm_start(...)`).
* **Constructor Injection (Static)**: Passing callbacks when instantiating the model or chain (`ChatOpenAI(callbacks=[my_handler])`). The handler is bound permanently to that specific object.
* **Runtime Injection (Dynamic - Best Practice)**: Passing callbacks when invoking the chain (`chain.invoke(input, config={"callbacks": [my_handler]})`). The handler automatically propagates down to all children in the LCEL graph.
""")

st.code('''
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Custom Callback Handler Definition
class ConsoleMetricsCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        print(f"[CALLBACK] LLM Invocation Started on model: {serialized.get('name')}")

    def on_llm_end(self, response, **kwargs):
        tokens = response.llm_output.get("token_usage", {})
        print(f"[CALLBACK] LLM Completed. Total Tokens: {tokens.get('total_tokens')}")

# 2. Modern LCEL Runtime Injection Pattern
prompt = ChatPromptTemplate.from_template("Summarize: {topic}")
model = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | model

# Injected dynamically at runtime via RunnableConfig!
response = chain.invoke(
    {"topic": "Quantum Computing"},
    config={"callbacks": [ConsoleMetricsCallbackHandler()]}
)
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production web services (e.g., FastAPI), request-specific telemetry handlers (capturing `tenant_id` and `request_id`) 
are created per HTTP request and injected dynamically at runtime via `config={"callbacks": [...]}`.
""")

st.divider()

# ==============================================================================
# SECTION 3: MODEL CALLBACKS
# ==============================================================================
st.header("3. Model Callbacks (on_llm_start, on_llm_new_token, on_llm_end, Streaming Token Capture)", divider="rainbow")

st.markdown("""
Model callbacks hook directly into the token generation lifecycle of language models. 
They allow applications to measure prompt token usage, track time-to-first-token (TTFT), and stream output characters to web frontends.
""")

st.markdown("""
### The 3 Core Model Lifecycle Hooks
* **`on_llm_start`**: Fired immediately before the HTTP request is dispatched to the LLM provider. Captures prompt strings, model parameters, and starts latency stopwatches.
* **`on_llm_new_token`**: Fired on *every individual token* as it arrives from the streaming socket. Used to pipe tokens directly to WebSockets or frontend UIs.
* **`on_llm_end`**: Fired when generation is complete. Captures total token usage (`prompt_tokens`, `completion_tokens`), final output text, and total model execution latency.
""")

model_callbacks_df = pd.DataFrame([
    {"Lifecycle Hook": "on_llm_start", "Parameters Passed": "serialized, prompts, run_id, parent_run_id", "Primary Use Case": "Start timing, log prompt inputs, prefill token auditing"},
    {"Lifecycle Hook": "on_llm_new_token", "Parameters Passed": "token, chunk, run_id, parent_run_id", "Primary Use Case": "Real-time user streaming, TTFT measurement"},
    {"Lifecycle Hook": "on_llm_end", "Parameters Passed": "response (LLMResult), run_id, parent_run_id", "Primary Use Case": "Cost calculation, total latency, token accounting"},
    {"Lifecycle Hook": "on_llm_error", "Parameters Passed": "error (Exception), run_id, parent_run_id", "Primary Use Case": "Provider rate-limit alerts (HTTP 429), failover triggers"}
])
st.dataframe(model_callbacks_df, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
While `on_llm_new_token` is widely used for UI streaming, modern LCEL applications also leverage `chain.astream_events(version="v2")`, 
which emits an asynchronous event stream under the hood powered by LangChain's callback engine.
""")

st.divider()

# ==============================================================================
# SECTION 4: CHAIN CALLBACKS
# ==============================================================================
st.header("4. Chain Callbacks (on_chain_start, on_chain_end, on_chain_error, Nested LCEL State Spans)", divider="rainbow")

st.markdown("""
Chain callbacks govern the lifecycle of composite execution pipelines: prompts, output parsers, retrievers, 
and sequential LCEL expressions (`prompt | model | parser`). 
They enable hierarchical distributed tracing by establishing parent-child relationships between spans.
""")

st.markdown("""
### Managing Nested LCEL Execution Spans
* **`on_chain_start`**: Fired when a chain begins execution. Receives the input dictionary and creates the root span.
* **Hierarchical Nesting**: When an outer chain executes inner runnables (e.g., `RunnableParallel`), each child component fires its own `on_chain_start` containing the `parent_run_id`, allowing tracing backends to assemble an interactive Gantt waterfall.
* **`on_chain_error`**: Fired when any child component raises an unhandled exception, capturing the exact stack trace and failing component.
""")

st.code('''
class ChainTracingCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict, inputs: dict, run_id, parent_run_id=None, **kwargs):
        name = serialized.get("name", "Runnable")
        print(f"[SPAN START] Chain '{name}' | Run ID: {run_id} | Parent: {parent_run_id}")

    def on_chain_end(self, outputs: dict, run_id, **kwargs):
        print(f"[SPAN END] Run ID: {run_id} completed successfully.")

    def on_chain_error(self, error: Exception, run_id, **kwargs):
        print(f"[SPAN ERROR] Run ID: {run_id} crashed with: {error}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangSmith's automatic tracing is built entirely on top of LangChain's chain callback hooks. 
The `LangChainTracer` handler listens to `on_chain_start` and `on_chain_end` to construct trace trees automatically.
""")

st.divider()

# ==============================================================================
# SECTION 5: TOOL CALLBACKS
# ==============================================================================
st.header("5. Tool Callbacks (on_tool_start, on_tool_end, on_tool_error, Execution Timing and Payload Inspection)", divider="rainbow")

st.markdown("""
Tools represent the external boundary of an LLM application: database queries, REST API calls, code execution, 
and file lookups. Tool callbacks monitor the reliability, execution latency, and security of these external calls.
""")

st.markdown("""
### What Tool Callbacks Provide
* **Payload Auditing (`on_tool_start`)**: Inspects the exact arguments passed to the tool before execution. Can be used as a security gatekeeper to block dangerous parameters.
* **Latency Profiling (`on_tool_end`)**: Measures how long external databases or APIs took to respond, identifying slow third-party bottlenecks.
* **Exception Logging (`on_tool_error`)**: Captures HTTP status codes (404, 500, 504) and database connection exceptions without crashing the main application thread.
""")

st.code('''
class ToolAuditCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name")
        print(f"[TOOL AUDIT] Invoking tool '{tool_name}' with arguments: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        print(f"[TOOL AUDIT] Tool completed with response: {output[:100]}...")

    def on_tool_error(self, error: Exception, **kwargs):
        print(f"[TOOL ALERT] External tool crashed: {error}")
''', language="python")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise teams use tool callbacks to feed Prometheus or Datadog metrics, tracking external API success rates 
and alerting on-call engineers when internal database tool failure rates exceed 2%.
""")

st.divider()

# ==============================================================================
# SECTION 6: AGENT CALLBACKS
# ==============================================================================
st.header("6. Agent Callbacks (on_agent_action, on_agent_finish, Decision Thought Streaming and Action Auditing)", divider="rainbow")

st.markdown("""
Agents differ from simple chains because they make autonomous runtime decisions. 
Agent callbacks hook into the agent's thought-action loop, surfacing internal reasoning steps to users 
and creating audit trails for regulatory compliance.
""")

st.markdown("""
### The 2 Core Agent Lifecycle Hooks
* **`on_agent_action`**: Fired whenever the agent selects a tool to execute (`AgentAction`). Contains the agent's internal thought string (`log`), the chosen tool name (`tool`), and the parameter payload (`tool_input`).
* **`on_agent_finish`**: Fired when the agent decides that the task is complete and generates its final user-facing response (`AgentFinish`).
""")

agent_callback_matrix = pd.DataFrame([
    {"Agent Hook": "on_agent_action", "Captured Information": "Thought rationale, tool name, tool parameters", "Use Case": "Streaming 'Agent is searching CRM...' status updates to frontend UI"},
    {"Agent Hook": "on_agent_finish", "Captured Information": "Final synthesized answer, stopping criteria", "Use Case": "Logging completed task resolution and audit archiving"}
])
st.dataframe(agent_callback_matrix, use_container_width=True)

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In modern user interfaces (ChatGPT-style web apps), `on_agent_action` callbacks power the collapsible "Thinking..." accordions, 
showing users which tools were called in real time while the agent works toward task completion.
""")

st.divider()

# ==============================================================================
# SECTION 7: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("7. Production Implementation: Full LCEL Callback Dispatcher Engine (LCELCallbackDispatcherEngine)", divider="rainbow")

st.markdown("""
Below is the full production-grade Python implementation of the callback dispatcher engine. 
It captures sequential lifecycle events across chains, tools, model prefill, streaming token callbacks, 
and error terminations.
""")

st.code('''
class LCELCallbackDispatcherEngine:
    """
    Enterprise LCEL Callback Dispatcher & Event Lifecycle Engine.
    Simulates real-time event streaming across all 5 core LangChain callback interfaces:
    1. BaseCallbackHandler event emission (Lifecycle orchestration)
    2. Model Callbacks (on_llm_start, on_llm_new_token, on_llm_end)
    3. Chain Callbacks (on_chain_start, on_chain_end, on_chain_error)
    4. Tool Callbacks (on_tool_start, on_tool_end, on_tool_error)
    5. Agent Callbacks (on_agent_action, on_agent_finish)
    """
    def __init__(self, handler_name: str = "production-telemetry-dispatcher"):
        self.handler_name = handler_name
        self.dispatched_events: List[Dict[str, Any]] = []

    def dispatch_execution_pipeline(
        self,
        query: str,
        include_tool_execution: bool = True,
        simulate_error: bool = False,
        streaming_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Dispatches an LCEL execution flow and records the sequential callback lifecycle events.
        """
        events = []

        # 1. Chain Start
        events.append({
            "event_type": "on_chain_start",
            "component": "CustomerSupportPipeline",
            "phase": "Chain Ingress",
            "payload": {"query": query},
            "timestamp_offset_ms": 0.0
        })

        # 2. Tool Invocation
        if include_tool_execution:
            events.append({
                "event_type": "on_tool_start",
                "component": "crm_database_tool",
                "phase": "Tool Dispatch",
                "payload": {"action": "lookup", "account_id": "acc_881"},
                "timestamp_offset_ms": 45.0
            })

            if simulate_error:
                events.append({
                    "event_type": "on_tool_error",
                    "component": "crm_database_tool",
                    "phase": "Tool Crash",
                    "payload": {"error": "HTTP 504 Gateway Timeout"},
                    "timestamp_offset_ms": 120.0
                })
                events.append({
                    "event_type": "on_chain_error",
                    "component": "CustomerSupportPipeline",
                    "phase": "Chain Failure",
                    "payload": {"error": "Pipeline terminated due to tool error"},
                    "timestamp_offset_ms": 125.0
                })
                return {"status": "TERMINATED_WITH_ERROR", "total_events": len(events), "events": events, "final_output": None}
            else:
                events.append({
                    "event_type": "on_tool_end",
                    "component": "crm_database_tool",
                    "phase": "Tool Success",
                    "payload": {"status": "active", "tier": "enterprise"},
                    "timestamp_offset_ms": 95.0
                })

        # 3. Model Start
        events.append({
            "event_type": "on_llm_start",
            "component": "ChatOpenAI (gpt-4o)",
            "phase": "Model Prefill",
            "payload": {"model": "gpt-4o", "temperature": 0.0},
            "timestamp_offset_ms": 110.0
        })

        # 4. Streaming Token Callbacks
        sample_tokens = ["Your ", "enterprise ", "account ", "is ", "active ", "and ", "in ", "good ", "standing."]
        if streaming_enabled:
            for idx, token in enumerate(sample_tokens):
                events.append({
                    "event_type": "on_llm_new_token",
                    "component": "ChatOpenAI",
                    "phase": "Token Stream",
                    "payload": {"token": token, "index": idx},
                    "timestamp_offset_ms": round(150.0 + (idx * 25.0), 1)
                })

        # 5. Model End
        full_text = "".join(sample_tokens)
        events.append({
            "event_type": "on_llm_end",
            "component": "ChatOpenAI",
            "phase": "Model Complete",
            "payload": {"output_text": full_text, "total_tokens": 142},
            "timestamp_offset_ms": 380.0
        })

        # 6. Chain End
        events.append({
            "event_type": "on_chain_end",
            "component": "CustomerSupportPipeline",
            "phase": "Chain Complete",
            "payload": {"final_response": full_text},
            "timestamp_offset_ms": 385.0
        })

        return {"status": "SUCCESSFUL_EXECUTION", "total_events": len(events), "events": events, "final_output": full_text}
''', language="python")

st.divider()

# ==============================================================================
# SECTION 8: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("8. Interactive Workbench: Real-Time Callback Event Stream and Token Lifecycle Inspector", divider="rainbow")

st.markdown("""
Simulate an LCEL pipeline execution and inspect the sequential callback events emitted across the lifecycle. 
Toggle tool invocations, token streaming, and simulated tool crashes to observe callback event dispatching.
""")

wb_col1, wb_col2 = st.columns([1, 2])

with wb_col1:
    st.markdown("### Execution Pipeline Controls")
    u_query = st.text_input("User Ingress Query", value="Check account status for customer 881.")
    include_tool = st.toggle("Include External Tool Execution (CRM Lookup)", value=True)
    stream_tokens = st.toggle("Enable on_llm_new_token Streaming Events", value=True)
    sim_error = st.toggle("Simulate Tool Crash (HTTP 504 Timeout)", value=False, help="Simulates an unhandled external API crash.")

with wb_col2:
    st.markdown("### Live Callback Event Stream Waterfall")
    run_report = st.session_state.lcel_callback_engine.dispatch_execution_pipeline(
        query=u_query,
        include_tool_execution=include_tool,
        simulate_error=sim_error,
        streaming_enabled=stream_tokens
    )

    if run_report["status"] == "SUCCESSFUL_EXECUTION":
        st.success(f"**STATUS: {run_report['status']}** — Dispatched {run_report['total_events']} callback events successfully.")
        st.markdown(f"**Final Output Delivered:** `{run_report['final_output']}`")
    else:
        st.error(f"**STATUS: {run_report['status']}** — Pipeline aborted via `on_tool_error` and `on_chain_error`.")

    # Event Stream Table
    st.markdown("#### Sequential Callback Event Log")
    events_df = pd.DataFrame(run_report["events"])
    st.dataframe(events_df[["timestamp_offset_ms", "event_type", "component", "phase"]], use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 9: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("9. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### High-Frequency Architectural Interview Questions")

with st.expander("Q1: What is the architectural difference between Constructor and Runtime callback injection in LangChain?"):
    st.markdown("""
    **Answer**: Constructor injection binds handlers at object instantiation (`ChatOpenAI(callbacks=[h])`). The handler is static 
    and shared across all calls to that instance. Runtime injection passes handlers during execution (`chain.invoke(..., config={"callbacks": [h]})`). 
    Runtime injection is the enterprise best practice because it automatically propagates across all child runnables in the LCEL graph 
    and allows request-scoped handlers (e.g., binding unique `request_id` or `tenant_id` per web request) without recreating model objects.
    """)

with st.expander("Q2: Why should external logging callbacks inherit from AsyncCallbackHandler rather than BaseCallbackHandler?"):
    st.markdown("""
    **Answer**: `BaseCallbackHandler` methods are synchronous (`def on_llm_end(...)`). If a callback makes a blocking HTTP call to an 
    external service (like Datadog or Slack), it freezes the Python thread, delaying token generation and blocking the entire application. 
    `AsyncCallbackHandler` implements non-blocking coroutines (`async def on_llm_end(...)`), allowing telemetry dispatches to execute 
    concurrently on the asyncio event loop without inflating user latency.
    """)

with st.expander("Q3: What parameters are passed to on_llm_new_token during streaming?"):
    st.markdown("""
    **Answer**: `on_llm_new_token` receives: 
    1) `token` (the string representation of the newly generated token chunk); 
    2) `chunk` (the `ChatGenerationChunk` object containing metadata and logprobs); 
    3) `run_id` (the unique UUID of the active model execution); and 
    4) `parent_run_id` (the UUID of the parent chain run, enabling hierarchy tracking).
    """)

with st.expander("Q4: How does LangSmith use LangChain callbacks under the hood?"):
    st.markdown("""
    **Answer**: LangSmith's distributed tracing is implemented as an internal LangChain callback handler (`LangChainTracer`). 
    When `LANGCHAIN_TRACING_V2=true` is set, LangChain automatically attaches this handler to all LCEL runs. The tracer listens to 
    `on_chain_start`, `on_llm_start`, `on_tool_end`, etc., and dispatches background telemetry spans over a non-blocking queue.
    """)

with st.expander("Q5: What is the difference between on_tool_error and on_chain_error?"):
    st.markdown("""
    **Answer**: `on_tool_error` is fired specifically when an external tool execution crashes (e.g., database timeout, SQL syntax error). 
    It provides an opportunity for an agent to capture the exception and self-correct. `on_chain_error` is fired when an entire LCEL chain 
    or runnable pipeline fails, signaling that the top-level workflow has aborted.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_cb1, col_cb2 = st.columns(2)
with col_cb1:
    st.error("**Trap 1 (Blocking Callbacks in High-Throughput Services)**: Making synchronous database writes inside `on_llm_end`. In high-throughput APIs, synchronous I/O inside callbacks adds 100-300ms of artificial latency to every request. Always use asynchronous or queue-based handlers.")
    st.warning("**Trap 2 (State Mutation Inside Callbacks)**: Modifying input dictionaries or model responses inside callback methods. Callbacks are strictly observational observers; mutating data inside handlers causes unpredictable bugs in downstream LCEL runnables.")
with col_cb2:
    st.error("**Trap 3 (Forgotten Parent Context in Multi-Threading)**: Running child tasks in a custom `ThreadPoolExecutor` without passing the `RunnableConfig`. The thread loses the parent callback manager, dropping all downstream spans from tracing.")
    st.warning("**Trap 4 (Leaking PII in Logging Callbacks)**: Emitting raw prompt arguments to external logging services inside `on_llm_start`. If customer prompts contain credit cards or SSNs, unscrubbed callbacks create severe GDPR/HIPAA compliance violations.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: How do you inject callbacks dynamically at runtime in LCEL?")
        st.caption("Flip for answer:")
        st.info("**Back**: Via the `config` argument in invoke or stream: `chain.invoke(input, config={'callbacks': [my_handler]})`.")

    with st.container(border=True):
        st.markdown("**Front**: What lifecycle hook captures real-time streaming tokens?")
        st.caption("Flip for answer:")
        st.info("**Back**: `on_llm_new_token(token, chunk, run_id, parent_run_id, **kwargs)`.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What design pattern powers LangChain's callback system?")
        st.caption("Flip for answer:")
        st.info("**Back**: The Observer Pattern: the `CallbackManager` broadcasts typed lifecycle events to registered subscribers without coupling to business logic.")

    with st.container(border=True):
        st.markdown("**Front**: What two hooks govern agent reasoning loops?")
        st.caption("Flip for answer:")
        st.info("**Back**: `on_agent_action` (emitted before tool invocation) and `on_agent_finish` (emitted upon task completion).")

st.divider()

# ==============================================================================
# SECTION 10: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("10. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Callbacks in LangChain:")

c1 = st.checkbox("I understand the Observer Pattern architecture underlying LangChain Callbacks.", value=True)
c2 = st.checkbox("I can differentiate between BaseCallbackHandler (sync) and AsyncCallbackHandler (async).", value=True)
c3 = st.checkbox("I know how to inject callbacks at runtime using the LCEL RunnableConfig interface.", value=True)
c4 = st.checkbox("I can capture streaming tokens in real time using the on_llm_new_token lifecycle hook.", value=True)
c5 = st.checkbox("I understand how to monitor tool execution timings and exceptions using tool callbacks.", value=True)
c6 = st.checkbox("I know how to track multi-step agent decisions using on_agent_action and on_agent_finish.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Callbacks, asynchronous event streaming, and lifecycle observability in LangChain.")
