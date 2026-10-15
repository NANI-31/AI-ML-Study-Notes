import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="39.3 Reliability Best Practices - LangChain",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for reliability scorecard
st.markdown("""
<style>
    .resilience-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-recovered {
        background-color: #065F46;
        color: #A7F3D0;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .badge-failed {
        background-color: #7F1D1D;
        color: #FECACA;
        padding: 3px 8px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MODULE SCOPE SIMULATORS FOR INTERACTIVE WORKBENCH ---
class ChaosResilienceSimulator:
    """
    Simulates production failure conditions and evaluates recovery mechanisms:
    HTTP 429 rate limits, socket timeouts, tool execution errors,
    and automatic failover to secondary fallback models.
    """
    def __init__(self):
        self.failure_modes = {
            "Rate Limit Burst (HTTP 429)": {
                "injected_fault": "Provider Rate Limit Exceeded (HTTP 429 Too Many Requests)",
                "with_retries": True,
                "retry_delay_s": [0.5, 1.0, 2.0],
                "fallback_triggered": False,
                "final_status": "RECOVERED",
                "recovery_action": "Exponential backoff succeeded on Attempt 3. Total latency 3.5s."
            },
            "Primary Provider Outage (HTTP 503)": {
                "injected_fault": "Service Unavailable (HTTP 503 Cloud Outage)",
                "with_retries": True,
                "retry_delay_s": [0.5, 1.0],
                "fallback_triggered": True,
                "final_status": "RECOVERED",
                "recovery_action": "with_fallbacks([AnthropicClaude]) activated. Rerouted successfully."
            },
            "External Tool Socket Timeout": {
                "injected_fault": "Tool Execution Exceeded 5.0s SLA Threshold",
                "with_retries": False,
                "retry_delay_s": [],
                "fallback_triggered": False,
                "final_status": "RECOVERED",
                "recovery_action": "Tool timeout trapped gracefully. Error returned to LLM for self-correction."
            },
            "Malformed JSON Output from Model": {
                "injected_fault": "Pydantic ValidationError on Missing Field 'summary'",
                "with_retries": True,
                "retry_delay_s": [0.2],
                "fallback_triggered": False,
                "final_status": "RECOVERED",
                "recovery_action": "OutputFixingParser prompted LLM with error snippet. Valid JSON produced."
            }
        }

    def simulate_fault(self, mode_key: str, enable_guards: bool = True) -> Dict[str, Any]:
        data = self.failure_modes.get(mode_key, self.failure_modes["Rate Limit Burst (HTTP 429)"])
        res = dict(data)
        if not enable_guards:
            res["final_status"] = "CRASHED"
            res["recovery_action"] = "Unhandled exception raised. Application process terminated."
        return res


# --- LESSON HEADER ---
st.title("39.3 LangChain Reliability Best Practices")
st.markdown("""
In distributed AI architectures, failures are guaranteed to happen: model providers return HTTP 429 rate limits,
network connections time out, external database tools throw exceptions, and models periodically emit malformed output.
Engineering reliable systems requires defense-in-depth: combining exponential backoff retries, strict per-step timeouts,
automatic provider fallbacks, self-correcting parsers, and continuous production monitoring.
""")

# --- SECTION 1: ARCHITECTURAL BLUEPRINT ---
st.header("1. Architectural Blueprint: Fault-Tolerant & Self-Healing Agent Architecture", divider="rainbow")
st.markdown("""
A reliable LangChain architecture wraps every external dependency in resilience boundaries.
Calls to LLM providers are protected by retry loops and multi-provider fallbacks; tool executions are bounded
by timeouts and structured error traps; and completions pass through validating parsers that re-prompt the model
if validation fails.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph Reliability_Best_Practices_Arch {
        graph [bgcolor="#0B1120", rankdir=TB, splines=polyline, nodesep=0.4, ranksep=0.6];
        node [shape=box, style="filled,rounded", fontname="Arial", fontsize=11, fontcolor="#F8FAFC", margin="0.2,0.1"];
        edge [color="#38BDF8", fontname="Arial", fontsize=9, fontcolor="#94A3B8"];

        subgraph cluster_ingress {
            label = "Request Execution Boundary";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";
            UserRequest [label="Inbound Request\\n(Bounded by Overall SLA Timeout: 30s)", fillcolor="#0F172A", color="#3B82F6"];
        }

        subgraph cluster_model_resilience {
            label = "Model Resilience Tier (Retries & Multi-Provider Fallbacks)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            PrimaryModel [label="Primary Model (e.g. OpenAI GPT-4o)\\n(max_retries=3 + Exponential Backoff)", fillcolor="#1E293B", color="#60A5FA"];
            FallbackModel [label="Secondary Fallback (e.g. Anthropic Claude)\\n(with_fallbacks([backup_model]))", fillcolor="#1E293B", color="#60A5FA"];
        }

        subgraph cluster_tool_safety {
            label = "Tool Execution Guardrails & Timeout Boundaries";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            ToolExecutor [label="Tool Invocation Wrapper\\n(timeout=5s + try/except Exception)", fillcolor="#0369A1", color="#38BDF8"];
            SafeErrorReturn [label="Error Formatter\\n(Returns 'Tool error: [reason]' to LLM)", fillcolor="#0369A1", color="#38BDF8"];
        }

        subgraph cluster_output_healing {
            label = "Validation & Self-Correction Tier";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            PydanticValidator [label="Pydantic Schema Validator\\n(Validates typed structured response)", fillcolor="#065F46", color="#10B981"];
            FixingParser [label="OutputFixingParser / Self-Correction\\n(Re-prompts model with validation error)", fillcolor="#065F46", color="#10B981"];
        }

        subgraph cluster_observability {
            label = "Production Telemetry (LangSmith / Datadog)";
            color = "#1E293B";
            style = "filled";
            fontcolor = "#93C5FD";

            TelemetryCollector [label="Distributed Traces & Error Budgets\\n(Latency, failure rate, token anomalies)", fillcolor="#065F46", color="#10B981"];
        }

        UserRequest -> PrimaryModel [label="Invoke with Timeout"];
        PrimaryModel -> PrimaryModel [label="429: Exponential Retry"];
        PrimaryModel -> FallbackModel [label="503 Outage: Switch Fallback"];
        PrimaryModel -> ToolExecutor [label="Tool Call Request"];
        FallbackModel -> ToolExecutor [label="Tool Call Request"];
        ToolExecutor -> SafeErrorReturn [label="Timeout / Exception Trapped"];
        SafeErrorReturn -> PrimaryModel [label="Self-Correction Prompt"];
        ToolExecutor -> PydanticValidator [label="Generate Output"];
        PydanticValidator -> FixingParser [label="Schema ValidationError"];
        FixingParser -> PrimaryModel [label="Re-prompt with Error"];
        PydanticValidator -> TelemetryCollector [label="Log Trace & Metrics"];
        TelemetryCollector -> UserRequest [label="Safe 200 OK Delivery"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

# --- SECTION 2: ADD RETRIES ---
st.header("2. Exponential Backoff & Retry Strategies: Handling 429 Rate Limits and Transmit Errors", divider="rainbow")
st.markdown("""
Network hiccups and transient HTTP 429 (Rate Limit) errors are common in high-concurrency applications.
Hardcoded immediate retries exacerbate provider rate limits.
Reliable systems employ **exponential backoff with jitter**, doubling wait times between retry attempts
to allow provider rate-limiting token buckets to replenish.
""")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("### Retry Configuration Rules")
    st.markdown("""
    - **Native Retries**: Configure `max_retries=3` directly on the model instance (e.g. `ChatOpenAI(max_retries=3)`).
    - **Decorators (`tenacity`)**: Wrap custom API calls or tools with `@retry(wait=wait_exponential(multiplier=1, min=2, max=10))` to ensure resilient network calls.
    - **Jitter**: Always enable randomized jitter to prevent the 'thundering herd' problem where thousands of clients retry at the exact same millisecond.
    """)
with col_r2:
    st.code("""
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. Native provider retry configuration
model = ChatOpenAI(
    model="gpt-4o",
    max_retries=4,  # Automatically handles 429 and 500 errors with backoff
    temperature=0.0
)

# 2. Resilient tool wrapper using Tenacity
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=False
)
def call_external_crm(customer_id: str) -> dict:
    # Retries transient network connection errors
    return api_client.get(f"/customers/{customer_id}")
    """, language="python")

# --- SECTION 3: ADD TIMEOUTS ---
st.header("3. Granular Timeout Configurations: Per-Step SLAs and Tool-Level Guards", divider="rainbow")
st.markdown("""
Without explicit timeouts, a hanging network socket or slow external database query will hold server threads indefinitely,
exhausting connection pools and crashing the microservice.
Every model invocation and tool execution must enforce strict timeouts aligned with user SLAs.
""")

st.code("""
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

# 1. Model-level connection timeout (seconds)
model = ChatOpenAI(
    model="gpt-4o",
    timeout=10.0,  # Fails fast if provider takes > 10s
    max_retries=2
)

# 2. Per-invocation runtime timeout via RunnableConfig
config = RunnableConfig(
    timeout=15.0,  # Entire chain execution hard timeout
    max_concurrency=5
)

# If execution exceeds 15 seconds, asyncio.TimeoutError is raised cleanly
try:
    response = chain.invoke({"query": "Generate report"}, config=config)
except TimeoutError:
    # Graceful degradation response
    response = "The operation timed out. Please narrow your query parameters."
""", language="python")

# --- SECTION 4: VALIDATE OUTPUTS ---
st.header("4. Strict Output Validation: Pydantic Parsing Guards and Self-Correction", divider="rainbow")
st.markdown("""
Never trust that an LLM's raw text response contains valid formatting.
Completions must be parsed against Pydantic models. If validation fails, use LangChain's `OutputFixingParser`
to feed the broken JSON and the validation error back into the model for automated self-correction.
""")

st.code("""
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI

class AuditReport(BaseModel):
    compliance_score: int = Field(..., ge=1, le=100)
    flagged_violations: list[str]

base_parser = PydanticOutputParser(pydantic_object=AuditReport)

# OutputFixingParser catches ValidationError and prompts the LLM to fix the syntax
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
)

# Example: Broken string missing required integer type
malformed_output = '{"compliance_score": "HIGH", "flagged_violations": []}'
# Automatically self-corrects: returns AuditReport(compliance_score=90, flagged_violations=[])
recovered_object = fixing_parser.parse(malformed_output)
""", language="python")

# --- SECTION 5: HANDLE TOOL FAILURES ---
st.header("5. Resilient Tool Failure Handling: Safe Error Bubbling and Informative Feedback", divider="rainbow")
st.markdown("""
When an external tool throws an unhandled exception (e.g. database down, 404 not found), the agent loop crashes.
Tools must catch exceptions internally and return informative error strings.
This allows the agent's LLM reasoning loop to read the error message, realize what failed, and attempt an alternative strategy.
""")

col_tf1, col_tf2 = st.columns(2)
with col_tf1:
    st.markdown("### Best Practices for Tool Resilience")
    st.markdown("""
    - **Never Let Exceptions Escape**: Wrap all tool internals in `try/except Exception as e:`.
    - **Descriptive Error Messages**: Return strings explaining what went wrong so the LLM can self-heal (e.g. 'Error: Table orders does not exist. Available tables: order_items, customer_orders').
    - **Fallback Alternatives**: Provide secondary fallback search paths if primary external tools are unreachable.
    """)
with col_tf2:
    st.code("""
from langchain_core.tools import tool

@tool
def query_production_database(query_str: str) -> str:
    \"\"\"Executes read-only SQL queries against database.\"\"\"
    try:
        results = db_engine.execute(query_str)
        return str(results.fetchall())
    except Exception as exc:
        # DO NOT crash agent loop with raise exc
        # Return informative error text to LLM
        return (
            f"Database Execution Error: {str(exc)}. "
            "Please verify table names and query syntax, then retry."
        )
    """, language="python")

# --- SECTION 6: ADD FALLBACKS ---
st.header("6. Graceful Multi-Model Fallbacks: with_fallbacks() and Provider Redundancy", divider="rainbow")
st.markdown("""
Relying on a single AI provider introduces a single point of failure. When OpenAI, Anthropic, or Azure experiences
an unexpected outage, production traffic must failover automatically.
LangChain provides native `.with_fallbacks()` support, creating seamless failover cascades across independent providers.
""")

st.code("""
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Primary frontier model
primary_llm = ChatOpenAI(model="gpt-4o", timeout=10.0, max_retries=1)

# Secondary failover model (Independent provider infrastructure)
secondary_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=10.0)

# Tertiary emergency local / cloud fallback
emergency_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", timeout=10.0)

# Composite resilient model: Tries Primary -> Secondary -> Tertiary
resilient_chat_model = primary_llm.with_fallbacks([secondary_llm, emergency_llm])

# If OpenAI fails or returns 500/503, Anthropic takes over in milliseconds
response = resilient_chat_model.invoke("Synthesize incident summary.")
""", language="python")

# --- SECTION 7: MONITOR PRODUCTION ---
st.header("7. Production Observability & Monitoring: Distributed Tracing and Token Telemetry", divider="rainbow")
st.markdown("""
You cannot fix what you cannot measure. Production LangChain applications must report distributed traces,
latency distributions, error budgets, and token usage into observability platforms like LangSmith, Datadog, or OpenTelemetry.
""")

col_mon1, col_mon2 = st.columns(2)
with col_mon1:
    st.markdown("### Critical Production Telemetry")
    st.markdown("""
    - **Step-Level Latency**: Track time spent in retrieval vs model generation vs tool execution.
    - **Token Usage Trends**: Monitor sudden prompt token inflation indicating un-pruned context loops.
    - **Error Classification**: Distinguish between tool timeouts, model rate limits, and parsing validation failures.
    - **Feedback Loops**: Collect explicit thumbs-up/down ratings and correlate them with traces.
    """)
with col_mon2:
    st.code("""
# LangSmith Automated Tracing Configuration
# Configured via environment variables:
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_API_KEY="ls__..."
# LANGCHAIN_PROJECT="enterprise-production"

from langsmith import traceable

# Custom business function tracing
@traceable(run_type="chain", name="enterprise_compliance_gate")
def verify_compliance(user_input: str, model_output: str) -> bool:
    # Execution automatically logged with latency, inputs, and outputs
    return "APPROVED" in model_output
    """, language="python")

# --- SECTION 8: PRODUCTION IMPLEMENTATION ---
st.header("8. Production Implementation: Production-Grade Fault-Tolerant Agent Pipeline", divider="rainbow")
st.markdown("""
Below is a full production-ready implementation uniting exponential retries, strict timeouts,
tool error trapping, and automatic provider fallbacks into a single resilient pipeline.
""")

st.code("""
# resilient_production_agent.py
import asyncio
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Resilient Model Cascade
primary_model = ChatOpenAI(model="gpt-4o", timeout=8.0, max_retries=2)
fallback_model = ChatAnthropic(model="claude-3-5-sonnet-20241022", timeout=8.0)
resilient_llm = primary_model.with_fallbacks([fallback_model])

# 2. Resilient Tool with Safe Error Trap
@tool
def lookup_customer_record(customer_id: str) -> str:
    \"\"\"Fetches customer account details safely.\"\"\"
    try:
        # Simulate external network call
        if not customer_id.startswith("CUST-"):
            raise ValueError("Customer ID must start with prefix 'CUST-'")
        return f"Customer Record: {customer_id} (Tier: Enterprise)"
    except Exception as e:
        return f"TOOL_ERROR: {str(e)}. Please correct customer ID format and retry."

# 3. Composed Resilient Chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an enterprise support assistant. If tools report errors, assist user in correcting input."),
    ("human", "{query}")
])

resilient_pipeline = prompt | resilient_llm.bind_tools([lookup_customer_record]) | StrOutputParser()

# Safe top-level invocation with error boundary
def safe_execute(user_query: str) -> str:
    try:
        return resilient_pipeline.invoke({"query": user_query})
    except Exception as fatal_error:
        # Ultimate fallback response preventing 500 error leaks to users
        return "Our intelligence services are currently experiencing elevated latency. Our engineering team has been alerted."
""", language="python")

# --- SECTION 9: INTERACTIVE WORKBENCH ---
st.header("9. Interactive Workbench: Resilience & Chaos Simulation Harness", divider="rainbow")
st.markdown("""
Inject simulated production failures (rate limits, cloud outages, tool timeouts, malformed JSON)
and observe how resilience guardrails intercept errors, trigger backoffs, and execute fallbacks.
""")

chaos_sim = ChaosResilienceSimulator()

col_fault, col_outcome = st.columns([1, 1])

with col_fault:
    st.subheader("💥 Chaos Fault Injection")
    selected_fault = st.selectbox(
        "Select Production Failure Scenario:",
        list(chaos_sim.failure_modes.keys())
    )
    guardrails_toggle = st.checkbox("Enable Resilience Guardrails (Retries, Fallbacks, Traps)", value=True)
    trigger_fault_btn = st.button("🚀 Inject Failure into Pipeline", type="primary")

with col_outcome:
    st.subheader("🛡️ Self-Healing & Recovery Telemetry")
    if trigger_fault_btn:
        res = chaos_sim.simulate_fault(selected_fault, enable_guards=guardrails_toggle)
        badge = "<span class='badge-recovered'>RECOVERED</span>" if res["final_status"] == "RECOVERED" else "<span class='badge-failed'>CRASHED (500)</span>"
        
        st.markdown(
            f"<div class='resilience-card'>"
            f"<b>Injected Fault:</b> <code>{res['injected_fault']}</code><br><br>"
            f"<b>Pipeline Status:</b> {badge}<br>"
            f"<b>Retries Applied:</b> {res['with_retries']} (Backoff delays: {res['retry_delay_s']})<br>"
            f"<b>Fallback Triggered:</b> {res['fallback_triggered']}<br><br>"
            f"<b>Action Taken:</b> {res['recovery_action']}"
            f"</div>",
            unsafe_allow_html=True
        )
        if res["final_status"] == "RECOVERED":
            st.success("Self-healing successful: Zero user disruption.")
        else:
            st.error("System crash: Unhandled exception propagated to end user.")
    else:
        st.info("Select a fault scenario and trigger to inspect self-healing behavior.")

# --- SECTION 10: TECHNICAL INTERVIEW PREPARATION & PRODUCTION TRAPS ---
st.header("10. Technical Interview Preparation & Production Traps", divider="rainbow")

with st.expander("Q1: How does `with_fallbacks()` work in LangChain and what exceptions trigger it?"):
    st.markdown("""
    **Answer**: `with_fallbacks([secondary_model, tertiary_model])` wraps a Runnable in an internal try/except block.
    If the primary model raises an exception (such as `RateLimitError`, `APIConnectionError`, `APITimeoutError`, or HTTP 500/503),
    LangChain immediately invokes the first fallback in the list without raising an error to the caller. You can configure
    `exceptions_to_handle=(APITimeoutError, RateLimitError)` to selectively catch only transient network errors while letting code bugs propagate.
    """)

with st.expander("Q2: Why must custom tools catch exceptions and return error strings instead of raising exceptions?"):
    st.markdown("""
    **Answer**: When an external tool raises an unhandled exception, Python halts execution and terminates the agent loop.
    By catching the exception inside the tool and returning a descriptive string (e.g. `'Error: Database table not found'`),
    the LLM receives the error as an observation. This allows the model to reason about the failure, adjust its parameters,
    and retry or explain the issue to the user.
    """)

with st.expander("Q3: What is the difference between client-side timeouts and server-side timeouts in LLM calls?"):
    st.markdown("""
    **Answer**: **Server-side timeouts** depend on the cloud provider's internal limits (often 60-120 seconds).
    **Client-side timeouts** (set via `timeout=8.0` in LangChain) terminate the client's HTTP connection after a strict deadline,
    preventing application worker threads from hanging indefinitely when providers experience network stalls.
    """)

with st.expander("Q4: How does `OutputFixingParser` recover from malformed JSON completions?"):
    st.markdown("""
    **Answer**: When a standard parser fails with a `ValidationError` or `JSONDecodeError`, `OutputFixingParser` catches the exception,
    constructs a correction prompt containing the broken completion and the exact error traceback, and sends it to a fast model (e.g. `gpt-4o-mini`)
    with instructions to fix the formatting. The corrected string is then successfully parsed into the target Pydantic object.
    """)

with st.expander("Q5: Why should exponential backoff always include randomized jitter?"):
    st.markdown("""
    **Answer**: When a service outage recovers or a shared rate limit resets, hundreds of client instances waiting on identical retry timers
    will simultaneously send requests at the exact same millisecond ('thundering herd' problem), immediately crashing the service again.
    Adding randomized jitter desynchronizes client retries, smoothing traffic spikes.
    """)

st.subheader("⚠️ Critical Production Traps & Antipatterns")
col_trap1, col_trap2 = st.columns(2)
with col_trap1:
    st.error("**Trap 1 (Immediate Infinite Retries)**: Retrying failed calls in a tight while loop without backoff. This consumes rate-limit quotas instantly and leads to permanent IP bans.")
    st.warning("**Trap 2 (Single-Provider Fragility)**: Depending entirely on a single model provider for business-critical features without configured `with_fallbacks()` redundancy.")
with col_trap2:
    st.error("**Trap 3 (Letting Tool Exceptions Crash Agents)**: Raising raw `psycopg2` or `requests` exceptions out of tools, crashing the user's session over transient network blips.")
    st.warning("**Trap 4 (Missing Global Hard Timeouts)**: Setting per-step timeouts but forgetting an overall timeout on the multi-step agent loop, allowing an agent to loop for 10 minutes.")

st.subheader("🧠 Concept Flashcards")
col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What LangChain method creates multi-model provider failovers?")
        st.caption("Flip for answer:")
        st.info("**Back**: `.with_fallbacks([secondary_model, tertiary_model])` on any Runnable.")
    with st.container(border=True):
        st.markdown("**Front**: What should a custom tool do when an external API fails?")
        st.caption("Flip for answer:")
        st.info("**Back**: Catch the exception and return a descriptive error string to allow LLM self-correction.")
with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What parser automatically fixes broken JSON output from models?")
        st.caption("Flip for answer:")
        st.info("**Back**: `OutputFixingParser` from `langchain.output_parsers`.")
    with st.container(border=True):
        st.markdown("**Front**: Why is jitter required in exponential retry loops?")
        st.caption("Flip for answer:")
        st.info("**Back**: Jitter randomizes retry intervals to prevent synchronized 'thundering herd' traffic spikes.")

# --- SECTION 11: SELF-ASSESSMENT CHECKLIST ---
st.header("11. Self-Assessment Checklist", divider="rainbow")
c1 = st.checkbox("I know how to configure exponential backoff retries with max_retries and tenacity.", value=True)
c2 = st.checkbox("I can enforce per-step and chain-level timeouts using RunnableConfig.", value=True)
c3 = st.checkbox("I know how to implement OutputFixingParser to recover from malformed JSON completions.", value=True)
c4 = st.checkbox("I understand how to handle tool errors safely to empower LLM self-correction.", value=True)
c5 = st.checkbox("I can configure multi-provider failover cascades using .with_fallbacks().", value=True)
c6 = st.checkbox("I know how to track latency, token usage, and error budgets with LangSmith distributed tracing.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("🎯 All checkpoints verified! You are ready to advance to Lesson 39.4: Security.")
