import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import Dict, List, Any, Optional

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="21.3 Debugging - LangChain Architecture",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling diagnostic cards and waterfalls
st.markdown("""
<style>
    .debug-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-cost {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-latency {
        background-color: #78350F;
        color: #FDE68A;
        padding: 3px 8px;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class MockDebuggingDiagnosticEngine:
    """
    Simulates LangSmith trace diagnostics, failure isolation,
    latency waterfall breakdown, token consumption accounting,
    and cost calculation across heterogeneous models.
    """
    def __init__(self):
        # Pricing reference in USD per 1K tokens
        self.pricing = {
            "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015},
            "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
            "claude-3-5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015}
        }
        self.traces = {
            "TR-FAIL-401": {
                "id": "TR-FAIL-401",
                "name": "SQL_Analyst_Pipeline",
                "root_status": "FAILED",
                "failure_node": "sql_execution_tool",
                "error_type": "DatabaseSyntaxError",
                "error_message": "relation 'users_v2' does not exist at character 19",
                "total_latency_ms": 2420,
                "latency_waterfall": [
                    {"step": "Prompt Hydration", "latency_ms": 40, "share_pct": 1.6},
                    {"step": "LLM SQL Generation (GPT-4o)", "latency_ms": 1680, "share_pct": 69.4},
                    {"step": "Tool DB Execution", "latency_ms": 700, "share_pct": 29.0}
                ],
                "tokens": {"input": 1420, "output": 180, "reasoning": 0},
                "model_used": "gpt-4o"
            },
            "TR-SLOW-802": {
                "id": "TR-SLOW-802",
                "name": "Financial_Report_RAG",
                "root_status": "SUCCESS",
                "failure_node": None,
                "error_type": None,
                "error_message": None,
                "total_latency_ms": 6840,
                "latency_waterfall": [
                    {"step": "Dense Vector Search", "latency_ms": 4200, "share_pct": 61.4},
                    {"step": "Cross-Encoder Rerank", "latency_ms": 1100, "share_pct": 16.1},
                    {"step": "LLM Synthesis (Sonnet)", "latency_ms": 1540, "share_pct": 22.5}
                ],
                "tokens": {"input": 4500, "output": 620, "reasoning": 0},
                "model_used": "claude-3-5-sonnet"
            }
        }

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rates = self.pricing.get(model, {"input_per_1k": 0.005, "output_per_1k": 0.015})
        cost = ((input_tokens / 1000.0) * rates["input_per_1k"]) + ((output_tokens / 1000.0) * rates["output_per_1k"])
        return cost

# Main Application Title & Objective
st.title("🔍 21.3 Debugging in LangSmith")
st.markdown("""
**LangChain & LangGraph Enterprise Architecture Guide**  
Debugging production AI applications requires specialized analytical tools capable of untangling non-deterministic reasoning chains. 
LangSmith transforms opaque black-box models into transparent execution pipelines by enabling **system prompt inspection**, 
**model response audits**, **tool argument validation**, **automated root-cause failure detection**, **latency waterfall profiling**, 
and **granular cost/token attribution**.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: The Root-Cause Diagnostic & Cost Profiling Architecture", divider="rainbow")
st.markdown("""
The architecture below delineates the diagnostic flow when a trace is analyzed in LangSmith. 
Telemetry data is continuously parsed by analytical engines that calculate latency waterfalls, verify schema conformance, 
attribute errors to specific components, and compute exact monetary costs.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    diag_dot = graphviz.Digraph(comment="Debugging Architecture")
    diag_dot.attr(bgcolor="#0B1120", rankdir="TB", size="9,7")
    diag_dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", color="#334155", fontcolor="#F8FAFC", fillcolor="#1E293B")
    diag_dot.attr("edge", color="#64748B", fontname="Helvetica", fontcolor="#94A3B8", fontsize="10")

    diag_dot.node("trace_in", "Captured Trace Run Tree\n[Root & Nested Child Runs]", fillcolor="#0F172A", shape="ellipse")
    diag_dot.node("triage", "Automated Triage Analyzer\n[Status: SUCCESS vs ERROR]", shape="diamond", fillcolor="#334155")
    
    # Analysis Engines
    diag_dot.node("error_isolator", "1. Root-Cause Error Isolator\n[Locates First Failing Node & Exception]", fillcolor="#7F1D1D")
    diag_dot.node("latency_engine", "2. Latency Waterfall Engine\n[P50/P99 Profiler & Bottleneck Detector]", fillcolor="#78350F")
    diag_dot.node("token_accountant", "3. Token Accountant & Profiler\n[Input, Output & Reasoning Allocation]", fillcolor="#0E7490")
    diag_dot.node("cost_calculator", "4. Financial Cost Calculator\n[Dynamic Pricing Model Table]", fillcolor="#065F46")

    diag_dot.node("output_dashboard", "Diagnostic Workbench\n[Actionable Bug Fixes & Optimization Steps]", fillcolor="#4338CA", shape="ellipse")

    diag_dot.edge("trace_in", "triage")
    diag_dot.edge("triage", "error_isolator", label=" Status: ERROR")
    diag_dot.edge("triage", "latency_engine", label=" Status: SUCCESS / SLOW")
    diag_dot.edge("triage", "token_accountant")
    diag_dot.edge("token_accountant", "cost_calculator")

    diag_dot.edge("error_isolator", "output_dashboard")
    diag_dot.edge("latency_engine", "output_dashboard")
    diag_dot.edge("cost_calculator", "output_dashboard")

    st.graphviz_chart(diag_dot, use_container_width=True)

# --- SECTION 2: INSPECTING PROMPTS ---
st.header("2. Inspecting Prompts & System Message Hydration", divider="rainbow")
st.markdown("""
In production, prompts are rarely static strings; they are assembled dynamically through **few-shot retrievers**, 
**system message templates**, and **user session history**. LangSmith captures the exact byte-for-byte prompt delivered to the model.
""")

col_p1, col_p2 = st.columns(2)
with col_p1:
    st.subheader("What Prompt Inspection Reveals")
    st.markdown("""
    * **Template Hydration Failures**: Unresolved variable placeholders (e.g. `{customer_name}` rendered verbatim).
    * **Prompt Bloat**: Injected RAG context exceeding necessary length, driving up costs and degrading attention.
    * **System Prompt Drift**: Verification that the intended system guardrails were not accidentally overwritten by user message formatting.
    """)
with col_p2:
    st.subheader("Inspecting Raw Prompts in Code")
    st.code("""
# Fetch trace run directly via LangSmith SDK:
from langsmith import Client
client = Client()

run = client.read_run("run-uuid-101")
print("Rendered Messages:")
for msg in run.inputs.get("messages", []):
    print(f"[{msg['role'].upper()}]: {msg['content'][:120]}...")
    """, language="python")

# --- SECTION 3: INSPECTING MODEL RESPONSES ---
st.header("3. Inspecting Model Responses & Stop Reasons", divider="rainbow")
st.markdown("""
Inspecting model outputs involves evaluating not only the generated text, but the accompanying **Generation Metadata** 
emitted by foundational models.
""")

col_resp1, col_resp2 = st.columns(2)
with col_resp1:
    st.subheader("Key Generation Metadata Indicators")
    st.markdown("""
    * **`finish_reason: "stop"`**: Model completed its generation naturally.
    * **`finish_reason: "length"`**: Generation was abruptly cut off because output hit `max_tokens`. (Indicates truncated responses!)
    * **`finish_reason: "tool_calls"`**: Model yielded control to invoke external tools.
    * **`system_fingerprint`**: Identifies specific underlying model weights used by the provider, tracking silent vendor updates.
    """)
with col_resp2:
    st.subheader("Inspecting Output Payload")
    st.code("""
run = client.read_run("run-uuid-101")
generation = run.outputs["generations"][0]

print("Text Content:", generation["text"])
print("Finish Reason:", generation["generation_info"].get("finish_reason"))
print("Model Fingerprint:", generation["generation_info"].get("system_fingerprint"))
    """, language="python")

# --- SECTION 4: INSPECTING TOOL CALLS ---
st.header("4. Inspecting Tool Calls & Schema Conformance", divider="rainbow")
st.markdown("""
A primary source of runtime failures in tool-calling agents is **argument hallucination** or **schema violation**. 
LangSmith captures the exact parsed tool call dictionary, allowing developers to detect discrepancies between Pydantic schemas and model generation.
""")

col_tc1, col_tc2 = st.columns(2)
with col_tc1:
    st.subheader("Common Tool Call Failure Modes")
    st.markdown("""
    * **Type Mismatch**: Model outputs `"123"` (string) when the tool expects `123` (integer).
    * **Missing Required Fields**: Model omits mandatory parameters specified in Pydantic schema.
    * **Hallucinated Parameters**: Model invents non-existent arguments (e.g. `order_by` on a tool that lacks SQL sorting).
    * **Invalid Enumerations**: Model selects an invalid enum choice (e.g. `"EXPEDITE"` instead of `"EXPEDITED"`).
    """)
with col_tc2:
    st.subheader("Tool Trace Argument Verification")
    st.code("""
# Inspect tool execution run
tool_run = client.read_run("tool-run-uuid-202")
print("Tool Invoked:", tool_run.name)
print("Arguments Passed:", tool_run.inputs)
print("Tool Execution Return:", tool_run.outputs)
    """, language="python")

# --- SECTION 5: FINDING FAILURES ---
st.header("5. Finding Failures & Automated Error Attribution", divider="rainbow")
st.markdown("""
In a deeply nested graph containing dozens of nodes, finding the exact point of failure manually is time-consuming. 
LangSmith automatically traverses the run tree to locate the **First Failing Node** and surfaces the exact Python exception.
""")

col_fail1, col_fail2 = st.columns(2)
with col_fail1:
    st.subheader("Automated Failure Queries")
    st.markdown("""
    In the LangSmith console or via the SDK, developers can query:
    * `status = "ERROR"`: Filters for all failed traces.
    * `error ~ "RateLimitError"`: Locates all provider quota breaches.
    * `error ~ "ValidationError"`: Identifies Pydantic parsing failures.
    * `run_type = "tool" and status = "ERROR"`: Isolates third-party API breakages.
    """)
with col_fail2:
    st.subheader("SDK Error Search")
    st.code("""
# Query runs with error status across past 24 hours:
error_runs = client.list_runs(
    project_name="production-agents",
    error=True,
    limit=10
)

for r in error_runs:
    print(f"Failed Run {r.id}: {r.name} -> {r.error}")
    """, language="python")

# --- SECTION 6: LATENCY ANALYSIS ---
st.header("6. Latency Analysis & Component Bottleneck Profiling", divider="rainbow")
st.markdown("""
When an end-to-end agent invocation takes 6,000 milliseconds, naive timing cannot tell you which component caused the delay. 
LangSmith constructs an exact **Latency Waterfall**, breaking down execution into discrete component percentages.
""")

col_lat1, col_lat2 = st.columns(2)
with col_lat1:
    st.subheader("Latency Anatomy in AI Pipelines")
    st.markdown("""
    * **LLM Prefill & Inference (40% to 70%)**: The primary contributor in long generations.
    * **Vector Search & Embedding (15% to 30%)**: Unindexed dense vector stores or unoptimized hybrid rerankers.
    * **External API Tools (10% to 40%)**: Slow third-party web services or relational database queries.
    * **Local Middleware (< 2%)**: Python JSON parsing, schema validation, and template formatting.
    """)
with col_lat2:
    st.subheader("Component Latency Profiling Example")
    st.code("""
# Programmatically inspect child run durations:
root = client.read_run("root-run-uuid")
child_runs = client.list_runs(parent_run_id=root.id)

for child in child_runs:
    duration_ms = (child.end_time - child.start_time).total_seconds() * 1000
    print(f"{child.name} ({child.run_type}): {duration_ms:.1f} ms")
    """, language="python")

# --- SECTION 7: TOKEN USAGE ---
st.header("7. Token Usage Breakdown (Input vs Output vs Reasoning)", divider="rainbow")
st.markdown("""
Token usage directly dictates both operational latency and financial cost. 
Modern observability requires granular accounting across three distinct token classes: **Input Tokens**, **Output Tokens**, and **Reasoning Tokens**.
""")

col_tok1, col_tok2 = st.columns(2)
with col_tok1:
    st.subheader("The Three Token Classes")
    st.markdown("""
    * **Prompt / Input Tokens**: Tokens consumed by system instructions, conversational history, and retrieved RAG context.
    * **Completion / Output Tokens**: Tokens generated by the model returned to the user or downstream agent.
    * **Reasoning Tokens**: Internal tokens generated by reasoning models (e.g. o1/o3, R1) during thinking before emitting the final answer.
    """)
with col_tok2:
    st.subheader("Token Optimization Strategies")
    st.markdown("""
    * **Context Pruning**: Truncate conversational history using sliding window token buffers (`trim_messages`).
    * **Prompt Compression**: Remove boilerplate instructions from system prompts.
    * **Top-K Tuning**: Reduce vector retriever chunk limits from K=10 to K=3 if precision allows.
    """)

# --- SECTION 8: COST TRACKING ---
st.header("8. Cost Tracking & Financial Attribution Models", divider="rainbow")
st.markdown("""
LangSmith automatically maintains an up-to-date **Model Pricing Index**, mapping token usage to exact monetary expenditures in real time. 
Costs can be grouped by project, customer tier, or individual developer.
""")

st.markdown("""
### Pricing Formula
```text
Total_Cost_USD = (Input_Tokens / 1000 * Input_Rate) + (Output_Tokens / 1000 * Output_Rate)
```
* **GPT-4o Standard Rates**: 0.005 USD per 1K Input Tokens | 0.015 USD per 1K Output Tokens.
* **GPT-4o-Mini Rates**: 0.00015 USD per 1K Input Tokens | 0.0006 USD per 1K Output Tokens.
* **Claude 3.5 Sonnet Rates**: 0.003 USD per 1K Input Tokens | 0.015 USD per 1K Output Tokens.
""")

# --- SECTION 9: PRODUCTION IMPLEMENTATION ---
st.header("9. Production Implementation: Automated Failure Diagnostic & Cost/Token Attribution Engine", divider="rainbow")
st.markdown("""
The following complete production pattern implements an automated trace analyzer that evaluates completed runs, 
detects root-cause errors, benchmarks latency waterfalls, and calculates precise dollar costs.
""")

st.code('''
from typing import Dict, Any, List
from langsmith import Client

client = Client()

class ProductionTraceDiagnosticEngine:
    """
    Automated diagnostic engine that analyzes LangSmith runs,
    pinpoints failure nodes, and computes financial metrics.
    """
    PRICING_CATALOG = {
        "gpt-4o": {"in": 0.005 / 1000, "out": 0.015 / 1000},
        "gpt-4o-mini": {"in": 0.00015 / 1000, "out": 0.0006 / 1000},
        "claude-3-5-sonnet": {"in": 0.003 / 1000, "out": 0.015 / 1000}
    }

    def analyze_trace(self, root_run_id: str) -> Dict[str, Any]:
        root_run = client.read_run(root_run_id)
        children = list(client.list_runs(parent_run_id=root_run_id))

        total_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        waterfall = []
        failure_culprit = None

        for child in children:
            # Latency attribution
            if child.start_time and child.end_time:
                duration_ms = (child.end_time - child.start_time).total_seconds() * 1000
            else:
                duration_ms = 0.0

            waterfall.append({
                "component": child.name,
                "type": child.run_type,
                "duration_ms": round(duration_ms, 2)
            })

            # Check for failures
            if child.error and not failure_culprit:
                failure_culprit = {
                    "failing_node": child.name,
                    "error_text": child.error,
                    "inputs_passed": child.inputs
                }

            # Cost & token calculation
            if child.run_type == "llm":
                prompt_tok = child.prompt_tokens or 0
                comp_tok = child.completion_tokens or 0
                total_prompt_tokens += prompt_tok
                total_completion_tokens += comp_tok

                model_name = child.extra.get("metadata", {}).get("ls_model_name", "gpt-4o")
                rates = self.PRICING_CATALOG.get(model_name, self.PRICING_CATALOG["gpt-4o"])
                step_cost = (prompt_tok * rates["in"]) + (comp_tok * rates["out"])
                total_cost += step_cost

        return {
            "root_run_id": root_run_id,
            "status": root_run.status,
            "failure_details": failure_culprit,
            "latency_waterfall": waterfall,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_cost_usd": round(total_cost, 6)
        }
''', language="python")

# --- SECTION 10: INTERACTIVE WORKBENCH ---
st.header("10. Interactive Workbench: Trace Debugger, Latency Waterfall & Cost Profiler", divider="rainbow")
st.markdown("""
Explore simulated production traces. Select a trace below to inspect **Root-Cause Error Attribution**, 
view the **Latency Waterfall Breakdown**, and analyze **Token Consumption and Dollar Costs**.
""")

# Instantiate Module Scope Simulator
if "dbg_sim" not in st.session_state:
    st.session_state.dbg_sim = MockDebuggingDiagnosticEngine()

dbg_engine = st.session_state.dbg_sim

col_trg1, col_trg2 = st.columns([1, 1])

with col_trg1:
    st.subheader("1. Select Diagnostic Trace")
    selected_trace_id = st.selectbox(
        "Trace Identifier:",
        options=list(dbg_engine.traces.keys()),
        format_func=lambda x: f"{x} - {dbg_engine.traces[x]['name']} ({dbg_engine.traces[x]['root_status']})"
    )
    trace = dbg_engine.traces[selected_trace_id]

    st.markdown(f"**Trace ID**: `{trace['id']}`")
    st.markdown(f"**Pipeline**: `{trace['name']}`")
    st.markdown(f"**Status**: `{'FAILED' if trace['root_status'] == 'FAILED' else 'SUCCESS'}`")
    st.markdown(f"**Total Latency**: `{trace['total_latency_ms']} ms`")

    if trace["root_status"] == "FAILED":
        st.error(f"**Failing Component**: `{trace['failure_node']}`")
        st.code(f"{trace['error_type']}: {trace['error_message']}", language="text")
    else:
        st.success("All components completed successfully with 0 exceptions.")

with col_trg2:
    st.subheader("2. Financial Cost & Token Accounting")
    toks = trace["tokens"]
    calc_cost = dbg_engine.compute_cost(trace["model_used"], toks["input"], toks["output"])

    col_tk1, col_tk2, col_tk3 = st.columns(3)
    col_tk1.metric("Input Tokens", toks["input"])
    col_tk2.metric("Output Tokens", toks["output"])
    col_tk3.metric("Computed Cost", f"{calc_cost:.5f} USD")

    st.markdown("#### Latency Waterfall Breakdown:")
    wf_df = pd.DataFrame(trace["latency_waterfall"])
    st.dataframe(wf_df, use_container_width=True)

# --- SECTION 11: TECHNICAL INTERVIEW PREPARATION ---
st.header("11. Technical Interview Preparation & Production Traps", divider="rainbow")

st.subheader("💡 5 High-Frequency Technical Interview Questions")
with st.expander("Q1: How do you isolate the root-cause failure in an agent with 15 nested sub-runs?"):
    st.markdown("""
    In LangSmith, query the trace tree for runs where `status == "ERROR"`. 
    Sort the matching nodes chronologically by `start_time`. The earliest child run with an error status 
    is the **Root-Cause Failure**. Inspecting its input parameters, API status code, and exception stack trace 
    reveals whether the error originated from bad model generation, third-party API timeout, or schema parsing.
    """)

with st.expander("Q2: What is the diagnostic significance of `finish_reason == 'length'` in model traces?"):
    st.markdown("""
    When `finish_reason == "length"`, the LLM was terminated prematurely because the generated token count reached `max_tokens`. 
    In JSON structured output, this causes corrupt, unclosed JSON (e.g. `{"summary": "text without closing quote`). 
    Fixing this requires either increasing `max_tokens` or optimizing the prompt to generate more concise outputs.
    """)

with st.expander("Q3: How do you identify whether vector search or LLM generation is the primary latency bottleneck?"):
    st.markdown("""
    By examining the Latency Waterfall in the trace. Compare the duration of the `retriever` run against the `llm` run. 
    If the retriever duration accounts for more than 40% of total latency, the vector database needs indexing (HNSW/IVF), 
    metadata partition filtering, or embedding cache optimization.
    """)

with st.expander("Q4: Why does token usage tracking need to separate input tokens from output tokens?"):
    st.markdown("""
    Foundational model providers charge significantly higher prices for output tokens compared to input tokens 
    (typically 3x to 5x more expensive per token). Furthermore, output tokens require autoregressive decode generation 
    (20-50ms per token), making output token volume the primary driver of both cost and end-to-end latency.
    """)

with st.expander("Q5: How can LangSmith traces be used to create regression test datasets?"):
    st.markdown("""
    LangSmith provides an **Add to Dataset** button on any trace run. 
    When an unexpected edge-case bug occurs in production, developers can convert the exact input prompt and desired corrected 
    output into a test case in an evaluation dataset with a single click, preventing future regressions in CI/CD.
    """)

st.subheader("⚠️ Production Traps & Anti-Patterns")
col_dbg_t1, col_dbg_t2 = st.columns(2)
with col_dbg_t1:
    st.error("""
    **Production Trap 1 (Fatal Error): Unmonitored Recursive Agent Loops**  
    Deploying an agent without strict `max_iterations` recursion limits. 
    If a tool fails and returns ambiguous feedback, the agent can loop 50 times attempting to re-query, consuming hundreds of thousands of tokens in minutes.
    """)
    st.warning("""
    **Production Trap 2 (Subtle Warning): Silent Truncation from `max_tokens` Limits**  
    Setting `max_tokens` too low on structured output chains. 
    The model generates valid JSON until hitting the ceiling, resulting in un-parseable JSON errors that appear as random model hallucinations.
    """)
with col_dbg_t2:
    st.error("""
    **Production Trap 3 (Fatal Error): Ignoring Token Price Asymmetry**  
    Optimizing input prompts while allowing agents to generate verbose, unconstrained responses. 
    Because output tokens cost up to 5x more than input tokens, unconstrained verbosity is the fastest driver of cloud budget overruns.
    """)
    st.warning("""
    **Production Trap 4 (Subtle Warning): Masking Exceptions in Try/Except Blocks**  
    Catching exceptions inside node code and returning empty strings without re-raising or logging an error callback. 
    This hides the failure from LangSmith, marking the trace as `SUCCESS` and concealing critical production bugs from monitoring dashboards.
    """)

st.subheader("🃏 Core Concept Flashcards")
col_fc_d1, col_fc_d2 = st.columns(2)
with col_fc_d1:
    st.markdown("""
    <div class="debug-card">
        <h4>Flashcard 1: Latency Waterfall</h4>
        <p><strong>Metric:</strong> Component Duration Profiler</p>
        <p><strong>Definition:</strong> Visual timeline breaking down end-to-end latency into retriever, model inference, and external tool percentages.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="debug-card">
        <h4>Flashcard 2: finish_reason='length'</h4>
        <p><strong>Status:</strong> Generation Truncation Flag</p>
        <p><strong>Definition:</strong> Indicator that the model stopped generating because it hit <code>max_tokens</code>, frequently causing malformed JSON.</p>
    </div>
    """, unsafe_allow_html=True)
with col_fc_d2:
    st.markdown("""
    <div class="debug-card">
        <h4>Flashcard 3: Token Pricing Asymmetry</h4>
        <p><strong>Economics:</strong> Input vs Output Cost</p>
        <p><strong>Definition:</strong> The pricing structure where output tokens are 3x to 5x more costly than input tokens due to autoregressive compute costs.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="debug-card">
        <h4>Flashcard 4: Root-Cause Culprit</h4>
        <p><strong>Diagnostics:</strong> First Failing Node</p>
        <p><strong>Definition:</strong> The earliest child node in a trace hierarchy that raised an unhandled exception, identifying the origin of failure.</p>
    </div>
    """, unsafe_allow_html=True)

# --- SECTION 12: SELF-ASSESSMENT CHECKLIST ---
st.header("12. Self-Assessment Checklist", divider="rainbow")
st.markdown("Verify your engineering mastery of LangSmith debugging, latency profiling, and cost attribution:")

chk_dbg1 = st.checkbox("1. I understand how to inspect rendered prompts to identify unresolved variables and context bloat.")
chk_dbg2 = st.checkbox("2. I know how to interpret model generation metadata, including `finish_reason` and system fingerprints.")
chk_dbg3 = st.checkbox("3. I understand how to inspect tool call traces to identify argument hallucinations and schema violations.")
chk_dbg4 = st.checkbox("4. I know how to isolate the root-cause failing node in complex, deeply nested agent traces.")
chk_dbg5 = st.checkbox("5. I understand how to construct and interpret a component latency waterfall to eliminate bottlenecks.")
chk_dbg6 = st.checkbox("6. I understand how to calculate exact monetary costs across models based on input and output token volumes.")

if chk_dbg1 and chk_dbg2 and chk_dbg3 and chk_dbg4 and chk_dbg5 and chk_dbg6:
    st.success("🎯 Section Mastery Complete! You have demonstrated comprehensive understanding of Debugging in LangSmith.")
