import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional, Callable

# ==============================================================================
# MODULE-SCOPE SIMULATOR CLASSES (INTERACTIVE WORKBENCH ENGINE)
# ==============================================================================

class HookTraceEvent:
    """Represents a discrete lifecycle hook interception event."""
    def __init__(self, hook_type: str, component: str, description: str, latency_ms: float, payload_summary: str):
        self.hook_type = hook_type
        self.component = component
        self.description = description
        self.latency_ms = latency_ms
        self.payload_summary = payload_summary

class MockCustomHookTracerStudio:
    """Executes a simulated ReAct agent wrapped with granular user-defined custom hooks."""
    def __init__(self,
                 enable_before_model: bool = True,
                 enable_after_model: bool = True,
                 enable_before_tool: bool = True,
                 enable_after_tool: bool = True,
                 enable_wrap_model: bool = True,
                 enable_wrap_tool: bool = True):
        self.enable_before_model = enable_before_model
        self.enable_after_model = enable_after_model
        self.enable_before_tool = enable_before_tool
        self.enable_after_tool = enable_after_tool
        self.enable_wrap_model = enable_wrap_model
        self.enable_wrap_tool = enable_wrap_tool

    def run_agent_turn(self, raw_prompt: str, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        events: List[HookTraceEvent] = []
        current_prompt = raw_prompt

        # 1. Wrap-Model Context Enter
        if self.enable_wrap_model:
            events.append(HookTraceEvent(
                "Wrap-Model (Enter)", "TelemetryWrapper",
                "Started OpenTelemetry span 'llm.invocation' and initialized circuit breaker.",
                1.5, "Span ID: span-84920-alpha"
            ))

        # 2. Before-Model Hook
        if self.enable_before_model:
            # Inject dynamic tenant policy
            current_prompt = f"[SYSTEM_DIRECTIVE: Enforce SOC2 Confidentiality]\\n{current_prompt}"
            events.append(HookTraceEvent(
                "Before-Model", "PromptEnricher",
                "Appended dynamic security header and validated token bounds.",
                3.2, f"Prompt mutated (+38 chars)"
            ))

        # 3. Model Simulation
        simulated_llm_response = f"Plan: Must invoke tool '{tool_name}' with verified parameters."
        events.append(HookTraceEvent(
            "Core Model", "LLM-Engine",
            f"Model executed inference and planned tool call '{tool_name}'.",
            120.0, f"Generated: {simulated_llm_response}"
        ))

        # 4. After-Model Hook
        if self.enable_after_model:
            events.append(HookTraceEvent(
                "After-Model", "ResponseSanitizer",
                "Scanned model output for unintended system prompt leakage.",
                4.8, "Output verified safe (0 leaks detected)"
            ))

        # 5. Wrap-Model Context Exit
        if self.enable_wrap_model:
            events.append(HookTraceEvent(
                "Wrap-Model (Exit)", "TelemetryWrapper",
                "Closed span 'llm.invocation' and published token metrics.",
                1.1, "Duration: 124.6ms"
            ))

        # 6. Wrap-Tool Context Enter
        if self.enable_wrap_tool:
            events.append(HookTraceEvent(
                "Wrap-Tool (Enter)", "SandboxGovernor",
                "Enforced 3.0s execution timeout and initialized memory cgroup.",
                2.0, f"Sandbox assigned to tool '{tool_name}'"
            ))

        # 7. Before-Tool Hook
        clean_args = tool_args.copy()
        if self.enable_before_tool:
            if "query" in clean_args and "DROP" in str(clean_args["query"]).upper():
                clean_args["query"] = "[BLOCKED_SQL_INJECTION]"
            events.append(HookTraceEvent(
                "Before-Tool", "ArgumentValidator",
                "Sanitized SQL and argument types against Pydantic contract.",
                2.5, f"Arguments validated: {clean_args}"
            ))

        # 8. Tool Execution Simulation
        raw_tool_result = f"DB Result: {{'rows_returned': 4, 'status': 'OK', 'payload': 'Sensitive Internal Records'}}"
        events.append(HookTraceEvent(
            "Core Tool", tool_name,
            "Dispatched native Python tool execution.",
            45.0, raw_tool_result
        ))

        # 9. After-Tool Hook
        if self.enable_after_tool:
            sanitized_result = raw_tool_result.replace("Sensitive Internal Records", "[REDACTED_BY_AFTER_TOOL_HOOK]")
            events.append(HookTraceEvent(
                "After-Tool", "DataTruncator",
                "Truncated heavy database payload and redacted confidential columns.",
                3.4, f"Truncated Result: {sanitized_result}"
            ))
        else:
            sanitized_result = raw_tool_result

        # 10. Wrap-Tool Context Exit
        if self.enable_wrap_tool:
            events.append(HookTraceEvent(
                "Wrap-Tool (Exit)", "SandboxGovernor",
                "Teardown sandbox container and verified zero memory leaks.",
                1.8, "Tool exit code: 0"
            ))

        total_latency = sum(e.latency_ms for e in events)
        return {
            "events": events,
            "final_prompt": current_prompt,
            "final_tool_output": sanitized_result,
            "total_latency_ms": total_latency
        }


# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="11.3 Custom Middleware",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0EA5E9; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">11.3 Custom Middleware</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Granular lifecycle interceptors: Before-Model, After-Model, Before-Tool, After-Tool, Wrap-Model, Wrap-Tool, and object-oriented middleware classes.</div>', unsafe_allow_html=True)

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Custom Hook Hierarchy & Interceptor Contracts", divider="rainbow")

st.markdown("""
While pre-built middleware covers generalized governance, enterprise systems require domain-specific interception hooks. 
LangChain and LangGraph provide granular, composable interceptors operating at two core primitives:
1. **Model Boundary Hooks**: Intercepting prompt generation, model inputs, completions, and model-level latency.
2. **Tool Boundary Hooks**: Intercepting argument serialization, input validation, execution sandboxes, and returned observation payloads.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph CustomHookArchitecture {
        graph [bgcolor="#0B1120", rankdir=TB, nodesep=0.4, ranksep=0.5, fontname="Segoe UI"];
        node [shape=box, style="filled,rounded", fontname="Segoe UI", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Segoe UI", fontsize=9, color="#64748B", fontcolor="#94A3B8"];

        subgraph cluster_model_boundary {
            label="Model Interception Boundary";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#38BDF8";
            fontsize=12;

            WrapModelEnter [label="Wrap-Model Hook (Enter)\\n(Start OpenTelemetry span, allocate connection)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            BeforeModel [label="Before-Model Hook\\n(Mutate prompt, dynamic context injection)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            CoreLLM [label="LLM Invocation\\n(chat_model.invoke / astream)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
            AfterModel [label="After-Model Hook\\n(Inspect AIMessage, scrub leaks, check safety)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
            WrapModelExit [label="Wrap-Model Hook (Exit)\\n(Close trace span, calculate TTFT & cost)", fillcolor="#1E3A8A", fontcolor="#FFFFFF", color="#60A5FA"];
        }

        subgraph cluster_tool_boundary {
            label="Tool Interception Boundary";
            style="dashed,rounded";
            bgcolor="#1E293B";
            fontcolor="#34D399";
            fontsize=12;

            WrapToolEnter [label="Wrap-Tool Hook (Enter)\\n(Enforce sandbox limits, timeout context)", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
            BeforeTool [label="Before-Tool Hook\\n(Validate arguments, SQL injection scanner)", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
            CoreTool [label="Tool Function Execution\\n(Execute Python/SQL/API call)", fillcolor="#581C87", fontcolor="#FAF5FF", color="#C084FC", penwidth=2];
            AfterTool [label="After-Tool Hook\\n(Truncate heavy outputs, redact API keys)", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
            WrapToolExit [label="Wrap-Tool Hook (Exit)\\n(Clean up sandbox, publish telemetry)", fillcolor="#064E3B", fontcolor="#FFFFFF", color="#34D399"];
        }

        WrapModelEnter -> BeforeModel [label="1. Span Active"];
        BeforeModel -> CoreLLM [label="2. Clean Prompt"];
        CoreLLM -> AfterModel [label="3. Model Response"];
        AfterModel -> WrapModelExit [label="4. Output Verified"];
        WrapModelExit -> WrapToolEnter [label="5. Planned Tool Dispatch"];
        WrapToolEnter -> BeforeTool [label="6. Sandbox Armed"];
        BeforeTool -> CoreTool [label="7. Validated Args"];
        CoreTool -> AfterTool [label="8. Raw Output"];
        AfterTool -> WrapToolExit [label="9. Truncated Result"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# --- SECTION 2: BEFORE-MODEL HOOKS ---
st.header("2. Before-Model Hooks", divider="rainbow")

st.markdown("""
**Before-Model Hooks** execute immediately before an LLM invocation. 
They accept the current messages list, state schema, and runtime parameters, returning a mutated state before tokenization begins.

### Primary Use Cases:
- **Dynamic Context Injection**: Injecting real-time variables (e.g. current UTC timestamp, user permission scopes, or geolocation).
- **Prompt Specialization**: Adding tenant-specific directives or formatting guidelines.
- **Token Cap Enforcement**: Trimming oversized message histories before causing model context window overflow errors.
""")

st.code("""def before_model_hook(messages: list, config: dict) -> list:
    \"\"\"Intercepts prompt messages to inject dynamic tenant permissions.\"\"\"
    user_tenant = config.get("configurable", {}).get("tenant_id", "default")
    
    # Inject dynamic system message at the front
    injected_header = ("system", f"Runtime Tenant Context: {user_tenant}. Strictly isolate data.")
    return [injected_header] + messages""", language="python")

st.divider()

# --- SECTION 3: AFTER-MODEL HOOKS ---
st.header("3. After-Model Hooks", divider="rainbow")

st.markdown("""
**After-Model Hooks** execute immediately after the LLM completes its generation, intercepting the raw `AIMessage` 
before it is returned to the agent loop or parsed by output parsers.

### Primary Use Cases:
- **System Prompt Extraction Detection**: Detecting if the model inadvertently parroted proprietary system instructions.
- **Output Masking & Redaction**: Sanitizing responses that contain confidential database identifiers.
- **Token Consumption Accounting**: Extracting `usage_metadata` from the message for billing ledgers.
""")

st.code("""def after_model_hook(ai_message, config: dict):
    \"\"\"Scans generated model content for security leaks.\"\"\"
    content = ai_message.content
    if "INTERNAL_SECRET_KEY" in content:
        ai_message.content = "[REDACTED: Output contained proprietary configuration key]"
    return ai_message""", language="python")

st.divider()

# --- SECTION 4: BEFORE-TOOL HOOKS ---
st.header("4. Before-Tool Hooks", divider="rainbow")

st.markdown("""
**Before-Tool Hooks** intercept tool invocations before the underlying Python function executes. 
They inspect the tool's identifier and its argument dictionary.

### Primary Use Cases:
- **Argument Sanitization**: Preventing SQL injection in database tools or command injection in terminal execution tools.
- **Permission Checking**: Confirming whether the authenticated user has clearance for the requested arguments (e.g. checking whether user has write permissions on the targeted table).
- **Default Argument Injection**: Automatically filling missing tenant or session identifiers into tool arguments.
""")

st.code("""def before_tool_hook(tool_name: str, tool_args: dict, config: dict) -> dict:
    \"\"\"Intercepts tool calls to prevent SQL injection in queries.\"\"\"
    if tool_name == "sql_database_query":
        query = tool_args.get("query", "")
        # Enforce read-only constraint
        for forbidden in ["DROP", "DELETE", "ALTER", "UPDATE", "INSERT"]:
            if forbidden in query.upper():
                raise PermissionError(f"Mutating SQL operations ({forbidden}) are blocked by Before-Tool hook.")
    return tool_args""", language="python")

st.divider()

# --- SECTION 5: AFTER-TOOL HOOKS ---
st.header("5. After-Tool Hooks", divider="rainbow")

st.markdown("""
**After-Tool Hooks** intercept the raw return payload from tools before the agent sees it as an observation.

### Primary Use Cases:
- **Output Truncation**: When a tool returns a 10MB JSON dump or 50,000-row SQL table, returning it directly into the prompt blows the context window. After-tool hooks truncate or summarize data before prompt inclusion.
- **Error Formatting**: Transforming Python exceptions into polite, actionable feedback for the model.
- **Confidential Field Scrubbing**: Stripping hashed passwords or internal keys returned by third-party APIs.
""")

st.divider()

# --- SECTION 6: WRAP-MODEL HOOKS ---
st.header("6. Wrap-Model Hooks", divider="rainbow")

st.markdown("""
**Wrap-Model Hooks** surround the entire LLM call with a continuous execution context (using the decorator or context manager pattern).
Unlike simple before/after hooks, wrap hooks maintain persistent local variables across the invocation.

### Primary Use Cases:
- **Distributed Tracing**: Spanning OpenTelemetry / Datadog traces from request start to finish.
- **Accurate Latency & TTFT Tracking**: Measuring Time-To-First-Token in streaming calls and total socket latency.
- **Circuit Breakers**: Tracking consecutive model failures and temporarily opening the circuit to prevent cascading outages.
""")

st.code("""from contextlib import contextmanager
import time

@contextmanager
def wrap_model_context(model_name: str):
    start_time = time.perf_counter()
    print(f"📡 [WRAP-MODEL] Starting trace span for {model_name}...")
    try:
        yield
    except Exception as exc:
        print(f"❌ [WRAP-MODEL] Caught model exception: {exc}")
        raise
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        print(f"⏱️ [WRAP-MODEL] Invocation completed in {duration_ms:.2f}ms")""", language="python")

st.divider()

# --- SECTION 7: WRAP-TOOL HOOKS ---
st.header("7. Wrap-Tool Hooks", divider="rainbow")

st.markdown("""
**Wrap-Tool Hooks** surround tool execution with operational sandboxing:
- **Timeout Protection**: Enforcing hard deadlines (e.g. killing long-running subprocess tools after 5 seconds).
- **Resource Sandboxing**: Allocating memory quotas or temporary directory scratchpads that auto-delete upon completion.
- **Fallback Execution**: Automatically falling back to an alternate data source if the primary tool times out.
""")

st.divider()

# --- SECTION 8: CUSTOM MIDDLEWARE CLASSES ---
st.header("8. Custom Middleware Classes", divider="rainbow")

st.markdown("""
In enterprise production applications, grouping loose hook functions into cohesive, reusable **Middleware Classes** 
improves testability, parameterization, and clean dependency injection.
""")

st.code("""class EnterpriseAgentMiddleware:
    \"\"\"Standard base class defining the complete custom hook contract.\"\"\"
    def __init__(self, name: str):
        self.name = name

    def before_model(self, messages: list, config: dict) -> list:
        return messages

    def after_model(self, ai_message, config: dict):
        return ai_message

    def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> dict:
        return tool_args

    def after_tool(self, tool_name: str, tool_output: str, config: dict) -> str:
        return tool_output

    def wrap_model(self, model_callable):
        def wrapped(*args, **kwargs):
            return model_callable(*args, **kwargs)
        return wrapped

    def wrap_tool(self, tool_callable):
        def wrapped(*args, **kwargs):
            return tool_callable(*args, **kwargs)
        return wrapped""", language="python")

st.divider()

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Composable Enterprise Middleware Engine", divider="rainbow")

st.markdown("""
The following complete Python implementation demonstrates how custom hook classes are composed and executed 
within an autonomous agent harness.
""")

st.code("""import time
from typing import Dict, List, Any, Optional

class SecurityAuditMiddleware:
    \"\"\"Custom middleware enforcing SQL read-only guardrails and payload truncation.\"\"\"
    def __init__(self, max_output_chars: int = 500):
        self.max_output_chars = max_output_chars

    def before_model(self, messages: list, config: dict) -> list:
        # Stamp every prompt with immutable audit tags
        audit_tag = f"[AUDIT_TIMESTAMP: {time.time():.0f}]"
        return [("system", audit_tag)] + messages

    def after_model(self, ai_message, config: dict):
        # Scan for output compliance
        return ai_message

    def before_tool(self, tool_name: str, tool_args: dict, config: dict) -> dict:
        # Block destructive SQL queries
        if tool_name == "database_query":
            q = tool_args.get("query", "").upper()
            if any(kw in q for kw in ["DROP", "DELETE", "TRUNCATE"]):
                raise ValueError("Destructive operations blocked by SecurityAuditMiddleware.")
        return tool_args

    def after_tool(self, tool_name: str, tool_output: str, config: dict) -> str:
        # Truncate massive tool results
        if len(tool_output) > self.max_output_chars:
            return tool_output[:self.max_output_chars] + f"... [Truncated {len(tool_output) - self.max_output_chars} chars]"
        return tool_output

class GovernedAgentRunner:
    \"\"\"Executes an agent pipeline while honoring registered custom middleware hooks.\"\"\"
    def __init__(self, middlewares: List[SecurityAuditMiddleware]):
        self.middlewares = middlewares

    def execute_turn(self, prompt: str, tool_fn, tool_args: dict) -> Dict[str, Any]:
        messages = [("human", prompt)]
        config = {"configurable": {"tenant": "tenant-corp"}}

        # 1. Run all before_model hooks
        for mw in self.middlewares:
            messages = mw.before_model(messages, config)

        # 2. Simulated Model Run
        ai_response_content = f"I will query the database with parameters: {tool_args}"

        # 3. Run all after_model hooks
        for mw in reversed(self.middlewares):
            ai_response_content = mw.after_model(ai_response_content, config)

        # 4. Run all before_tool hooks
        active_args = tool_args
        for mw in self.middlewares:
            active_args = mw.before_tool(tool_fn.__name__, active_args, config)

        # 5. Execute Tool
        raw_output = tool_fn(**active_args)

        # 6. Run all after_tool hooks
        governed_output = raw_output
        for mw in reversed(self.middlewares):
            governed_output = mw.after_tool(tool_fn.__name__, governed_output, config)

        return {"plan": ai_response_content, "observation": governed_output}""", language="python")

st.divider()

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Custom Hook Interceptor & Execution Tracer", divider="rainbow")

st.markdown("""
Experiment with the custom hook lifecycle. Enable or disable individual interceptors, trigger destructive tool calls 
to test argument sanitization, and view the precise chronological flamechart of hook execution.
""")

col_wb1, col_wb2 = st.columns([1, 1])

with col_wb1:
    st.subheader("⚙️ Interceptor Hook Configuration")
    wb_prompt = st.text_input("User Prompt", value="Analyze quarterly corporate ledger records.")
    wb_tool_name = st.selectbox("Simulated Tool Target", options=["sql_query_runner", "file_reader", "system_terminal"])
    wb_sql_query = st.text_input("Simulated Tool Argument ('query')", value="SELECT id, name FROM accounts WHERE balance > 1000;")

    st.markdown("#### Active Hook Toggles:")
    h_before_m = st.toggle("Enable Before-Model Hook", value=True)
    h_after_m = st.toggle("Enable After-Model Hook", value=True)
    h_before_t = st.toggle("Enable Before-Tool Hook", value=True)
    h_after_t = st.toggle("Enable After-Tool Hook", value=True)
    h_wrap_m = st.toggle("Enable Wrap-Model Hook", value=True)
    h_wrap_t = st.toggle("Enable Wrap-Tool Hook", value=True)

    st.markdown("#### Adversarial Injections:")
    inject_sql = st.toggle("Inject Malicious 'DROP TABLE accounts;' into SQL", value=False)

    btn_trace = st.button("🚀 Execute Custom Hook Trace", type="primary", use_container_width=True)

with col_wb2:
    st.subheader("📊 Execution Trace & Flamechart")

    if btn_trace:
        target_query = "DROP TABLE accounts;" if inject_sql else wb_sql_query
        studio = MockCustomHookTracerStudio(
            enable_before_model=h_before_m,
            enable_after_model=h_after_m,
            enable_before_tool=h_before_t,
            enable_after_tool=h_after_t,
            enable_wrap_model=h_wrap_m,
            enable_wrap_tool=h_wrap_t
        )

        trace_data = studio.run_agent_turn(
            wb_prompt,
            wb_tool_name,
            {"query": target_query}
        )

        m1, m2 = st.columns(2)
        m1.metric("Total Turn Latency", f"{trace_data['total_latency_ms']:.1f} ms")
        m2.metric("Total Hook Events", len(trace_data["events"]))

        st.markdown("#### 💬 Final Intercepted Observation:")
        st.success(trace_data["final_tool_output"])

        st.markdown("#### 📜 Chronological Hook Execution Timeline:")
        for idx, ev in enumerate(trace_data["events"], 1):
            st.markdown(f"**Step {idx}: `{ev.hook_type}`** ({ev.component}) — *{ev.latency_ms:.1f}ms*")
            st.caption(f"{ev.description} | Payload: `{ev.payload_summary}`")
    else:
        st.info("Configure your hook preferences and click **Execute Custom Hook Trace** to observe lifecycle interception.")

st.divider()

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 High-Frequency Technical Interview Questions")
interview_qas = [
    ("What is the difference between Before-Model hooks and Wrap-Model hooks?",
     "A Before-Model hook is a stateless function that receives prompt messages, modifies them, and returns them prior to model dispatch. In contrast, a Wrap-Model hook surrounds the entire execution lifecycle (e.g. using a context manager). It maintains persistent state from before the call until after completion, making it ideal for distributed tracing (OpenTelemetry spans), socket latency timing, and circuit breaker exception trapping."),
    ("Why should argument validation happen in Before-Tool hooks rather than inside the tool function itself?",
     "Placing validation in Before-Tool hooks decouples business capabilities from governance policies. Standard tools can remain pure and reusable across multiple applications, while company-specific constraints (such as preventing DROP TABLE queries, verifying tenant ownership, or checking spending limits) are enforced consistently across all tools via centralized middleware."),
    ("How do After-Tool hooks prevent context window exhaustion in agents?",
     "When an agent calls an external API or database, the raw return payload can be hundreds of kilobytes or megabytes in size. Ingesting this directly into message history can exceed the model's context window or cost tens of dollars per turn. An After-Tool hook inspects the output size, truncates excessive records, or triggers an auxiliary summarizer to generate a bounded observation."),
    ("How do you propagate distributed trace contexts across custom middleware hooks?",
     "Use standard telemetry context carriers like OpenTelemetry's context.attach() or Python's contextvars module. The Wrap-Model or Ingress hook initializes a trace span and stores the SpanID in contextvars. Downstream Before-Tool and After-Tool hooks read from this context to correlate all child spans under the same root trace ID without polluting function argument signatures."),
    ("What happens if a custom middleware hook throws an unhandled exception?",
     "An unhandled exception inside a Before-Model or Before-Tool hook crashes the agent turn immediately. In enterprise systems, hooks must be wrapped in defensive exception handlers that either gracefully log and bypass non-critical hooks (fail-open for telemetry) or cleanly translate fatal errors into informative agent observations (fail-closed for security).")
]

for i, (q, a) in enumerate(interview_qas, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(a)

st.subheader("⚠️ Production Traps & Failure Modes")
trap_c1, trap_c2 = st.columns(2)
with trap_c1:
    st.error("**Trap 1 (Breaking Error)**: Breaking Streaming with Monolithic After-Model Hooks. If an After-Model hook expects a complete AIMessage string, it forces the entire response to buffer, completely breaking token streaming to the UI. Streaming requires custom token generator wrappers.")
    st.error("**Trap 2 (Breaking Error)**: Mutating Tool Arguments In-Place. Modifying tool argument dictionaries directly in a Before-Tool hook without copying can cause subtle bugs if other callbacks or retry handlers inspect the original payload. Always return modified copies.")
with trap_c2:
    st.warning("**Trap 3 (Latency Warning)**: Network I/O in Synchronous Before-Tool Hooks. Making remote HTTP calls to authorization servers inside synchronous Before-Tool hooks adds 200ms of latency per tool call. Use local caching for authorization policies.")
    st.warning("**Trap 4 (Quality Warning)**: Over-Truncation in After-Tool Hooks. Truncating tool responses too aggressively can remove the exact data point the LLM needed to answer the question, causing infinite tool retry loops. Truncate intelligently by preserving schema headers.")

st.subheader("🗂️ Core Knowledge Flashcards")
fc_c1, fc_c2 = st.columns(2)
with fc_c1:
    with st.expander("🎴 Flashcard 1: Wrap vs Before/After"):
        st.markdown("**Wrap hooks maintain persistent context managers across the invocation; Before/After are discrete state mutators.**")
    with st.expander("🎴 Flashcard 2: Before-Tool Validation"):
        st.markdown("**Centralizes argument policy enforcement (e.g. SQL read-only) outside the tool's core logic.**")
with fc_c2:
    with st.expander("🎴 Flashcard 3: After-Tool Truncation"):
        st.markdown("**Caps large data observations to protect context windows and control token costs.**")
    with st.expander("🎴 Flashcard 4: Distributed Tracing"):
        st.markdown("**Uses Wrap-Model hooks with contextvars to link OpenTelemetry child spans to the root trace.**")

st.divider()

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
chk1 = st.checkbox("I understand the architectural differences between Before/After hooks and Wrap hooks.", value=True)
chk2 = st.checkbox("I can implement Before-Model hooks for dynamic context injection and prompt governance.", value=True)
chk3 = st.checkbox("I know how to sanitize arguments using Before-Tool hooks to prevent injection attacks.", value=True)
chk4 = st.checkbox("I can use After-Tool hooks to truncate large payloads and prevent context overflow.", value=True)
chk5 = st.checkbox("I understand how to structure reusable custom middleware classes with clean lifecycle contracts.", value=True)
chk6 = st.checkbox("I am prepared to answer all 5 custom middleware interview questions without reference notes.", value=True)

if chk1 and chk2 and chk3 and chk4 and chk5 and chk6:
    st.success("🎯 Mastered! You have achieved 100% completion on 11.3 Custom Middleware and completed all of Chapter 11: Middleware!")
