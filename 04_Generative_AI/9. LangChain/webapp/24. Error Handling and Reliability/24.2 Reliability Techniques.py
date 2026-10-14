import streamlit as st
import pandas as pd
import numpy as np
import graphviz
from typing import List, Dict, Any, Tuple, Optional, Union

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="24.2 Reliability Techniques - LangChain Master Curriculum",
    page_icon="🛡️",
    layout="wide"
)

# --- MODULE LEVEL PRODUCTION ENGINE ---
class LCELReliabilityOrchestratorEngine:
    """
    Enterprise Reliability Orchestrator for resilient LCEL pipelines.
    Implements and evaluates the 8 core reliability techniques:
    1. Retries (Bounded retry loops with with_retry())
    2. Exponential Backoff (Full jitter interval calculations)
    3. Timeouts (Per-runnable deadline enforcement)
    4. Fallback Models (Cross-provider failover via with_fallbacks())
    5. Fallback Tools (Secondary search indices and cached data)
    6. Error Recovery (Self-healing feedback loops and exception transforms)
    7. Validation (Pydantic contract enforcement with auto-repair prompts)
    8. Guardrails (Safety classifiers and PII regex tripwires)
    """
    def __init__(self, primary_provider: str = "OpenAI", fallback_provider: str = "Anthropic"):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

    def simulate_resilience_execution(
        self,
        primary_outage_prob_pct: float = 60.0,
        backoff_strategy: str = "Exponential with Full Jitter",
        max_retries: int = 3,
        enable_fallback_model: bool = True,
        enable_guardrails: bool = True,
        random_seed: int = 101
    ) -> Dict[str, Any]:
        """
        Simulates an LCEL execution under chaos fault injection conditions.
        """
        rng = np.random.default_rng(random_seed)
        log_steps = []
        total_latency_ms = 0.0
        success = False
        final_model_used = None
        recovered_by = "None"

        # Step 0: Input Guardrails Check
        if enable_guardrails:
            log_steps.append({
                "Phase": "Input Guardrail",
                "Action": "PII Masking & Toxic Prompt Classifier Scan",
                "Status": "PASSED",
                "Duration (ms)": 15.0,
                "Details": "No injection patterns or raw credit cards detected. Passed to LCEL chain."
            })
            total_latency_ms += 15.0

        # Step 1: Primary Model Invocation with Retries
        primary_succeeded = False
        for attempt in range(1, max_retries + 1):
            total_latency_ms += 350.0  # Network roundtrip
            is_failure = (rng.uniform(0, 100) < primary_outage_prob_pct)

            if not is_failure:
                primary_succeeded = True
                success = True
                final_model_used = f"{self.primary_provider} (Primary)"
                recovered_by = "Primary Model" if attempt == 1 else f"Retry Attempt #{attempt}"
                log_steps.append({
                    "Phase": f"Primary Attempt #{attempt}",
                    "Action": f"Invoke {self.primary_provider} GPT-4o",
                    "Status": "SUCCESS",
                    "Duration (ms)": 350.0,
                    "Details": "Model returned valid generation with status HTTP 200 OK."
                })
                break
            else:
                # Calculate sleep backoff
                if backoff_strategy == "Fixed Delay":
                    sleep_ms = 1000.0
                elif backoff_strategy == "Exponential Backoff":
                    sleep_ms = 500.0 * (2 ** (attempt - 1))
                else:  # Full Jitter
                    max_sleep = 500.0 * (2 ** (attempt - 1))
                    sleep_ms = float(rng.uniform(100.0, max_sleep))

                total_latency_ms += sleep_ms
                log_steps.append({
                    "Phase": f"Primary Attempt #{attempt}",
                    "Action": f"Invoke {self.primary_provider} GPT-4o",
                    "Status": "FAILED (HTTP 503 Service Unavailable)",
                    "Duration (ms)": round(350.0 + sleep_ms, 1),
                    "Details": f"Transient error caught. Backoff sleep: {round(sleep_ms, 1)}ms before next attempt."
                })

        # Step 2: Fallback Model Activation
        if not primary_succeeded:
            if enable_fallback_model:
                log_steps.append({
                    "Phase": "Fallback Router",
                    "Action": f"Tripping with_fallbacks() to {self.fallback_provider} Claude 3.5 Sonnet",
                    "Status": "TRIGGERED",
                    "Duration (ms)": 20.0,
                    "Details": "Primary retry ceiling exhausted. Switching runnable provider dynamically."
                })
                total_latency_ms += 20.0

                # Fallback invocation
                fallback_latency = 480.0
                total_latency_ms += fallback_latency
                success = True
                final_model_used = f"{self.fallback_provider} (Fallback)"
                recovered_by = "Fallback Model Failover"
                log_steps.append({
                    "Phase": "Fallback Execution",
                    "Action": f"Invoke {self.fallback_provider} Claude 3.5 Sonnet",
                    "Status": "SUCCESS",
                    "Duration (ms)": fallback_latency,
                    "Details": "Secondary model responded cleanly with valid schema."
                })
            else:
                success = False
                final_model_used = "None (Pipeline Crashed)"
                recovered_by = "Failed (No Fallback Configured)"
                log_steps.append({
                    "Phase": "Pipeline Failure",
                    "Action": "Exception Propagated to Client",
                    "Status": "FATAL (HTTP 503)",
                    "Duration (ms)": 0.0,
                    "Details": "All retries exhausted with no fallback runnable configured. Application error thrown."
                })

        # Step 3: Output Validation & Guardrails
        if success and enable_guardrails:
            log_steps.append({
                "Phase": "Output Guardrail",
                "Action": "Pydantic Schema Validation & Hallucination Tripwire",
                "Status": "PASSED",
                "Duration (ms)": 25.0,
                "Details": "Pydantic model fields validated without exceptions."
            })
            total_latency_ms += 25.0

        return {
            "success": success,
            "final_model_used": final_model_used,
            "recovered_by": recovered_by,
            "total_latency_ms": round(total_latency_ms, 1),
            "step_count": len(log_steps),
            "execution_log": log_steps
        }


# ==============================================================================
# HEADER BLOCK
# ==============================================================================
st.title("24.2 Reliability Techniques: Resilient Pipelines, Retries, and Fallbacks")
st.caption("LangChain v0.3 Enterprise Resilience • with_retry(), with_fallbacks(), Exponential Backoff with Jitter, Dynamic Guardrails, and Self-Healing Chains")

# ==============================================================================
# SECTION 1: ARCHITECTURAL BLUEPRINT
# ==============================================================================
st.header("1. Architectural Blueprint: Resilient LCEL Reliability and Self-Healing Pipeline Architecture", divider="rainbow")

st.markdown("""
In distributed AI architectures, achieving high availability (99.9% uptime) requires assuming that 
every single component—the model provider, the vector database, the parsing layer, and external tools—will fail. 
High reliability is achieved not by hoping for flawless infrastructure, but by engineering multi-layered defense lines.

Modern LangChain v0.3 LCEL provides native primitives for building self-healing pipelines:
- **`with_retry()`**: Automatically retries transient errors with exponential backoff and jitter.
- **`with_fallbacks()`**: Dynamically routes execution to alternative models or cached tools when primary runnables crash.
- **Output Repair Loops**: Self-correcting chains that feed parsing errors back into LLM context for real-time syntax repair.
""")

with st.expander("🔍 View Complete Architectural Blueprint (Top-to-Bottom Flow)", expanded=True):
    blueprint_dot = """
    digraph ReliabilityBlueprint {
        rankdir=TB;
        bgcolor="#0B1120";
        node [fontname="Arial", fontsize=11, fontcolor="#F8FAFC", style="filled", penwidth=1.5];
        edge [fontname="Arial", fontsize=10, fontcolor="#94A3B8", color="#38BDF8", penwidth=1.5];

        subgraph cluster_defense_layer1 {
            label = "Defense Line 1: Pre-Execution Validation & Guardrails";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            raw_input [label="User Input Prompt", shape=box, fillcolor="#1E3A8A", color="#3B82F6"];
            input_guardrail [label="Input Guardrail / PII Redactor\\n(NeMo Guardrails / Regex / Presidio)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
        }

        subgraph cluster_defense_layer2 {
            label = "Defense Line 2: LCEL Runnable Retry & Timeout Wrapper";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            retry_wrapper [label="Primary Model with_retry()\\n(Max Attempts=3, Full Jitter Backoff)", shape=box, fillcolor="#B45309", color="#F59E0B"];
            step_timeout [label="Step Timeout Guard\\n(with_config(timeout=10.0))", shape=octagon, fillcolor="#B45309", color="#F59E0B"];
            primary_llm [label="Primary Provider LLM\\n(ChatOpenAI: gpt-4o)", shape=box, fillcolor="#4C1D95", color="#8B5CF6"];
        }

        subgraph cluster_defense_layer3 {
            label = "Defense Line 3: Multi-Provider Fallback Routing";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            fallback_router [label="with_fallbacks([backup_model])\\n(Circuit Router)", shape=diamond, fillcolor="#991B1B", color="#EF4444"];
            fallback_llm [label="Secondary Provider LLM\\n(ChatAnthropic: claude-3-5-sonnet)", shape=box, fillcolor="#0E7490", color="#06B6D4"];
            cache_fallback [label="Tertiary Fallback\\n(Semantic Cache / Mock Response)", shape=cylinder, fillcolor="#1E3A8A", color="#3B82F6"];
        }

        subgraph cluster_defense_layer4 {
            label = "Defense Line 4: Self-Healing Parsing & Output Validation";
            color = "#1E293B";
            style = "filled";
            fillcolor = "#0F172A";

            pydantic_validator [label="Pydantic Schema Validator\\n(PydanticOutputParser)", shape=box, fillcolor="#047857", color="#10B981"];
            repair_chain [label="Self-Correction Repair Chain\\n(OutputFixingParser / Reflection Loop)", shape=box, fillcolor="#047857", color="#10B981"];
            delivery_ok [label="Validated Reliable Output\\n(Delivery to Client)", shape=box, fillcolor="#047857", color="#10B981"];
        }

        raw_input -> input_guardrail;
        input_guardrail -> retry_wrapper [label="Sanitized Prompt"];
        retry_wrapper -> step_timeout;
        step_timeout -> primary_llm [label="Invoke"];
        primary_llm -> pydantic_validator [label="Success (HTTP 200)"];

        primary_llm -> retry_wrapper [label="Transient 429/503 (Attempt < 3)"];
        primary_llm -> fallback_router [label="Outage / Retries Exhausted"];
        fallback_router -> fallback_llm [label="Route to Secondary"];
        fallback_llm -> pydantic_validator [label="Success"];
        fallback_llm -> cache_fallback [label="Secondary Outage"];
        cache_fallback -> delivery_ok [label="Degraded Output"];

        pydantic_validator -> delivery_ok [label="Schema Valid"];
        pydantic_validator -> repair_chain [label="ValidationError"];
        repair_chain -> pydantic_validator [label="Repaired Output"];
    }
    """
    st.graphviz_chart(blueprint_dot, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 2: RETRIES
# ==============================================================================
st.header("2. Retries (Automatic Retries with with_retry(), Attempt Ceilings and Safe Exception Scopes)", divider="rainbow")

st.markdown("""
Network requests over the public internet fail regularly due to dropped packets, gateway restarts, and temporary 
provider congestion. In LangChain v0.3, any LCEL runnable can be decorated with `.with_retry()`:

```python
from langchain_openai import ChatOpenAI

# Native LCEL retry binding
resilient_model = model.with_retry(
    stop_after_attempt=4,
    retry_if_exception_type=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError
    )
)
```

**Essential Retry Rules**:
1. **Never retry infinitely**: Always enforce a hard ceiling (`stop_after_attempt <= 5`) to prevent frozen worker threads.
2. **Restrict Exception Scope**: Only retry transient network/capacity errors (429, 502, 503, 504). Never retry 
   non-transient errors like `AuthenticationError` (401) or `BadRequestError` (400).
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production services wrap external API calls in a retry policy that records every retry attempt as an OpenTelemetry 
event. If a runnable exhausts its retry budget, the system transitions to a fallback runnable rather than crashing 
the client session.
""")

st.divider()

# ==============================================================================
# SECTION 3: EXPONENTIAL BACKOFF
# ==============================================================================
st.header("3. Exponential Backoff (Full Jitter, Base Backoff Intervals and Rate Limit Absorption)", divider="rainbow")

st.markdown("""
When a server responds with HTTP 429 (Rate Limit), retrying immediately ensures another failure. 
**Exponential Backoff** increases the wait interval multiplicatively between successive attempts:
`interval = base_backoff * (2 ** attempt_index)`

However, pure exponential backoff suffers from the **Thundering Herd** problem: multiple synchronized threads 
wake up at the exact same millisecond and re-saturate the provider.
""")

st.markdown("""
### The Solution: Full Jitter Algorithm
Full jitter randomizes the sleep duration uniformly across the entire exponential window:
`sleep_duration = uniform_random(0, min(max_backoff, base_backoff * (2 ** attempt)))`

This spreads retries evenly over time, dramatically reducing peak request concurrency during provider recovery.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangChain's `.with_retry()` implements full jitter internally when configuring `wait_exponential_jitter=True`. 
This allows enterprise applications to absorb brief token spikes without dropping client queries.
""")

st.divider()

# ==============================================================================
# SECTION 4: TIMEOUTS
# ==============================================================================
st.header("4. Timeouts (Step-Level Deadlines, Request Timeouts and Non-Blocking Worker Guarantees)", divider="rainbow")

st.markdown("""
Without explicit timeouts, a stuck network socket or an infinite database lock will hang a worker thread indefinitely. 
In asynchronous web servers (FastAPI, Uvicorn, Tornado), a handful of hung threads can exhaust worker pool capacity, 
bringing down the entire application.

In LangChain, timeouts can be enforced at two levels:
1. **Model-Level Timeout**: `ChatOpenAI(model="gpt-4o", timeout=12.0)`
2. **Runnable-Level Timeout via Config**: `chain.with_config(timeout=15.0)`
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Production platforms implement hierarchical timeout budgets:
- Total request timeout: 20 seconds.
- Vector retrieval timeout: 2 seconds.
- Tool execution timeout: 4 seconds.
- LLM generation timeout: 14 seconds.
If any step exceeds its dedicated budget, the step aborts cleanly, allowing fallback logic to take over.
""")

st.divider()

# ==============================================================================
# SECTION 5: FALLBACK MODELS
# ==============================================================================
st.header("5. Fallback Models (with_fallbacks() Routing, Provider Diversity: GPT to Claude, Cost Degradation)", divider="rainbow")

st.markdown("""
No single LLM vendor provides 100% uptime. Building high-availability AI systems requires cross-provider redundancy. 
LangChain's `.with_fallbacks()` primitive allows chaining fallback runnables that activate when the primary fails:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o", max_retries=2)
backup = ChatAnthropic(model="claude-3-5-sonnet", max_retries=2)

# Seamless multi-provider failover
fault_tolerant_model = primary.with_fallbacks([backup])
```

**Common Fallback Strategies**:
1. **Provider Diversity**: OpenAI -> Anthropic -> Azure OpenAI.
2. **Cost and Tier Degradation**: GPT-4o (Premier) -> GPT-4o-mini (Lightweight) on provider capacity issues.
3. **Local Self-Hosted Fallback**: Cloud SaaS -> Local Ollama / vLLM instance inside the corporate VPC.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
To maintain prompt compatibility across providers, enterprise stacks use standard chat message formats 
(`SystemMessage`, `HumanMessage`) that LCEL maps cleanly to either OpenAI or Anthropic payload specifications.
""")

st.divider()

# ==============================================================================
# SECTION 6: FALLBACK TOOLS
# ==============================================================================
st.header("6. Fallback Tools (Alternative Retrieval Sources, Graceful Tool Degradation and Cached Mocks)", divider="rainbow")

st.markdown("""
Agents frequently depend on external APIs (weather, financial data, internal ERPs). When a tool goes down, 
an agent should degrade gracefully rather than crash:

1. **Secondary Search Indices**: If a hybrid vector search times out, fail over to a simple BM25 keyword index.
2. **Semantic Caching**: If live stock price API calls fail, return the most recent cached quote with a warning: 
   `"External financial API unavailable; displaying cached price as of 14:00 UTC."`
3. **Mock Tool Fallbacks**: Returning synthetic or baseline schema-compliant data so downstream LLM steps can proceed.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
LangChain tool wrappers define fallback handlers using `Tool.from_function(..., handle_tool_error=custom_fallback_func)`. 
The fallback function catches the network timeout and returns cached data directly to the agent.
""")

st.divider()

# ==============================================================================
# SECTION 7: ERROR RECOVERY
# ==============================================================================
st.header("7. Error Recovery (Prompt-Based Self-Correction, Output Repair Chains and Exception Handlers)", divider="rainbow")

st.markdown("""
When an LLM generates invalid output or a tool returns an error, traditional software crashes. 
In GenAI systems, we can leverage the model's reasoning capabilities to perform **Self-Healing Error Recovery**:

- **Reflection Loops**: Capturing the raw exception and injecting it into a recovery prompt: 
  `"Your previous SQL query failed with 'table users not found'. Inspect the schema: [...] and correct your query."`
- **Output Repair Chains**: Using `OutputFixingParser` to automatically repair broken JSON without changing downstream logic.
- **Exception Transformers**: Catching low-level database connection strings and sanitizing them before they reach the user.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
In production agent loops, self-correction is strictly bounded to 2 attempts. Unbounded self-correction loops 
risk runaway token usage if the underlying issue is fundamental (e.g., missing database permissions).
""")

st.divider()

# ==============================================================================
# SECTION 8: VALIDATION
# ==============================================================================
st.header("8. Validation (Pydantic Output Enforcement, Retry on Schema Failure and Auto-Repair Parsing)", divider="rainbow")

st.markdown("""
Reliable systems enforce type-safe contracts at every boundary. In LangChain, **Pydantic** is the primary 
standard for output validation:

```python
from pydantic import BaseModel, Field

class OrderSummary(BaseModel):
    order_id: str = Field(description="Unique order alphanumeric identifier")
    item_count: int = Field(ge=1, description="Number of items purchased")
    total_amount_usd: float = Field(ge=0.0, description="Total amount paid in USD")
```

If the LLM returns an `order_id` as `None` or a negative `total_amount_usd`, Pydantic raises a `ValidationError`. 
The system catches this error and triggers an immediate auto-repair prompt containing the validation report.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Modern models support native tool calling and JSON schema constraints (`response_format={"type": "json_object"}`), 
drastically reducing syntax errors. However, semantic validation (e.g., ensuring `start_date < end_date`) 
must still be verified by Pydantic validators after generation.
""")

st.divider()

# ==============================================================================
# SECTION 9: GUARDRAILS
# ==============================================================================
st.header("9. Guardrails (Input/Output Safety Tripwires, Hallucination Suppression and PII Masking)", divider="rainbow")

st.markdown("""
Reliability encompasses safety and compliance. A system that reliably outputs toxic text or leaks credit card numbers 
is fundamentally broken. **Guardrails** act as pre-flight and post-flight tripwires:

1. **Input Guardrails**:
   - Prompt Injection Detectors (blocking jailbreak phrases like `"Ignore all previous instructions"`).
   - PII Scrubbers (masking SSNs, credit cards, and emails before payloads leave the enterprise perimeter).
2. **Output Guardrails**:
   - Hallucination suppression (cross-referencing generated claims against retrieved context).
   - Brand safety and toxicity filters.
""")

st.markdown("""
### 🚀 Modern Production Reality: How Current GenAI Stacks Implement This
Enterprise stacks deploy guardrail frameworks (such as NeMo Guardrails, Llama Guard, or Microsoft Presidio) 
as lightweight middleware runnables chained directly into LCEL before and after LLM invocations.
""")

st.divider()

# ==============================================================================
# SECTION 10: PRODUCTION IMPLEMENTATION
# ==============================================================================
st.header("10. Production Implementation: LCEL Reliability Orchestrator Engine (LCELReliabilityOrchestratorEngine)", divider="rainbow")

st.markdown("""
The following complete Python implementation demonstrates `LCELReliabilityOrchestratorEngine`, an enterprise 
reliability orchestrator implementing multi-provider failover, jittered backoff, and guardrail verification.
""")

st.code(r'''
import numpy as np
from typing import List, Dict, Any

class LCELReliabilityOrchestratorEngine:
    """
    Enterprise Reliability Orchestrator for resilient LCEL pipelines.
    """
    def __init__(self, primary_provider: str = "OpenAI", fallback_provider: str = "Anthropic"):
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

    def simulate_resilience_execution(
        self,
        primary_outage_prob_pct: float = 60.0,
        backoff_strategy: str = "Exponential with Full Jitter",
        max_retries: int = 3,
        enable_fallback_model: bool = True,
        enable_guardrails: bool = True,
        random_seed: int = 101
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(random_seed)
        log_steps = []
        total_latency_ms = 0.0

        if enable_guardrails:
            log_steps.append({"Phase": "Input Guardrail", "Status": "PASSED", "Duration (ms)": 15.0})
            total_latency_ms += 15.0

        primary_succeeded = False
        for attempt in range(1, max_retries + 1):
            total_latency_ms += 350.0
            if rng.uniform(0, 100) >= primary_outage_prob_pct:
                primary_succeeded = True
                log_steps.append({"Phase": f"Primary Attempt #{attempt}", "Status": "SUCCESS"})
                break
            else:
                sleep_ms = float(rng.uniform(100.0, 500.0 * (2 ** (attempt - 1))))
                total_latency_ms += sleep_ms
                log_steps.append({"Phase": f"Primary Attempt #{attempt}", "Status": "FAILED (HTTP 503)"})

        if not primary_succeeded and enable_fallback_model:
            log_steps.append({"Phase": "Fallback Router", "Status": "TRIGGERED"})
            total_latency_ms += 480.0
            log_steps.append({"Phase": "Fallback Execution", "Status": "SUCCESS"})
            success = True
        else:
            success = primary_succeeded

        return {
            "success": success,
            "total_latency_ms": round(total_latency_ms, 1),
            "execution_log": log_steps
        }
''', language="python")

st.divider()

# ==============================================================================
# SECTION 11: INTERACTIVE WORKBENCH
# ==============================================================================
st.header("11. Interactive Workbench: Fallback Pipeline Simulator and Chaos Fault Injection Sandbox", divider="rainbow")

st.markdown("""
Inject simulated outages, modify retry policies, and test fallback models below to observe how 
resilience mechanisms absorb infrastructure chaos in real time.
""")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    outage_prob = st.slider("Simulated Primary Provider Outage Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=5.0)
    backoff_mode = st.selectbox("Retry Backoff Policy", options=["Fixed Delay", "Exponential Backoff", "Exponential with Full Jitter"], index=2)
    retry_ceiling = st.slider("Max Retry Attempts (stop_after_attempt)", min_value=1, max_value=5, value=3)

with col_sim2:
    use_fallback = st.toggle("Enable Secondary Model Fallback (Claude 3.5 Sonnet)", value=True)
    use_guardrails = st.toggle("Enable Input/Output Guardrails & Pydantic Validation", value=True)

# Run simulation
orchestrator = LCELReliabilityOrchestratorEngine()
sim_results = orchestrator.simulate_resilience_execution(
    primary_outage_prob_pct=outage_prob,
    backoff_strategy=backoff_mode,
    max_retries=retry_ceiling,
    enable_fallback_model=use_fallback,
    enable_guardrails=use_guardrails
)

st.subheader("⚡ Chaos Execution Telemetry")

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Pipeline Outcome", "✅ SUCCESS" if sim_results["success"] else "❌ FATAL CRASH")
col_kpi2.metric("Final Model Used", sim_results["final_model_used"])
col_kpi3.metric("Total Latency", f"{sim_results['total_latency_ms']} ms")

st.markdown(f"**Recovery Mechanism**: `{sim_results['recovered_by']}`")

st.subheader("📜 Step-by-Step Execution Trace")
trace_table = pd.DataFrame(sim_results["execution_log"])
st.dataframe(trace_table, use_container_width=True)

st.divider()

# ==============================================================================
# SECTION 12: TECHNICAL INTERVIEW PREPARATION
# ==============================================================================
st.header("12. Technical Interview Preparation and Production Traps", divider="rainbow")

st.markdown("### Top 5 Technical Interview Questions on Reliability")

with st.expander("Q1: How does LangChain's with_fallbacks() differ from a traditional try/except block in Python?"):
    st.markdown("""
    **Answer**: In traditional Python, wrapping calls in try/except requires manual state reconstruction, duplicated prompt formatting, 
    and custom error routing logic. In LCEL, `.with_fallbacks()` operates at the **Runnable** abstraction level. It automatically 
    forwards the identical input dictionary, streams tokens seamlessly, shares callback telemetry listeners, and preserves type contracts 
    across both primary and fallback runnables with zero boilerplate code.
    """)

with st.expander("Q2: Why should you never use fixed-interval retries in high-volume production LLM services?"):
    st.markdown("""
    **Answer**: Fixed-interval retries lead directly to the **Thundering Herd** problem. During provider rate limit episodes, 
    hundreds of concurrent client threads wait the exact same duration (e.g., 2000ms) and retry at the identical instant. 
    This creates synchronized periodic request spikes that immediately exhaust provider quotas again, preventing recovery.
    """)

with st.expander("Q3: What is the risk of unbounded self-healing reflection loops in autonomous agents?"):
    st.markdown("""
    **Answer**: If an agent is given an impossible task or lacks underlying database permissions, an unbounded reflection loop 
    will repeatedly pass the error back to the LLM. The agent will attempt dozens of alternative queries, consuming thousands of tokens 
    and running up cloud bills before eventually timing out. Production self-correction loops must always enforce a strict ceiling (typically max 2 iterations).
    """)

with st.expander("Q4: How do step-level timeouts protect asynchronous microservices running LangChain?"):
    st.markdown("""
    **Answer**: Without step-level timeouts, a hanging vector database or hung upstream socket will tie up an event loop worker thread 
    indefinitely. In asynchronous frameworks like FastAPI, a few hung worker threads can exhaust connection pools, causing the entire 
    API gateway to stop accepting new client traffic. Step timeouts ensure deadlocks abort fast, enabling fallbacks to take over.
    """)

with st.expander("Q5: What is the primary operational trade-off when using multi-provider model fallbacks?"):
    st.markdown("""
    **Answer**: The trade-off is **prompt drift and behavioral divergence**. Different model architectures (e.g., GPT-4o vs Claude 3.5 Sonnet) 
    exhibit distinct adherence to system prompts, token formatting quirks, and temperature sensitivities. When falling back to a secondary vendor, 
    output phrasing or JSON structures may vary slightly, requiring robust Pydantic normalization parsers downstream.
    """)

st.markdown("### Critical Production Traps & Antipatterns")

col_tr1, col_tr2 = st.columns(2)
with col_tr1:
    st.error("**Trap 1 (Retrying 400 Bad Request Errors)**: Configuring retries on all exceptions. A 400 Bad Request indicates invalid syntax or schema errors that will never succeed on retry, needlessly burning compute.")
    st.warning("**Trap 2 (Zero Step Timeouts)**: Relying solely on client HTTP timeouts. If intermediate tools hang for 60 seconds, the entire request hangs, causing gateway timeouts before fallback models can even be attempted.")
with col_tr2:
    st.error("**Trap 3 (Single-Vendor Lock-in for Mission-Critical APIs)**: Deploying production systems with no fallback model configured. When the primary provider suffers an outage, the application suffers 100% downtime.")
    st.warning("**Trap 4 (Unbounded Agent Self-Correction)**: Letting an agent loop indefinitely trying to repair a broken tool query. Always cap self-correction attempts at 2 before escalating to human or fallback.")

st.markdown("### Core Technical Flashcards")

col_fc1, col_fc2 = st.columns(2)
with col_fc1:
    with st.container(border=True):
        st.markdown("**Front**: What does with_retry() do in LCEL?")
        st.caption("Flip for answer:")
        st.info("**Back**: It wraps an LCEL runnable in a configurable retry policy with exponential backoff and jitter for transient errors.")

    with st.container(border=True):
        st.markdown("**Front**: What does with_fallbacks() do in LCEL?")
        st.caption("Flip for answer:")
        st.info("**Back**: It binds a list of alternative runnables that execute sequentially if the primary runnable throws an exception.")

with col_fc2:
    with st.container(border=True):
        st.markdown("**Front**: What is the maximum recommended number of self-healing iterations?")
        st.caption("Flip for answer:")
        st.info("**Back**: 2 iterations. Beyond 2, diminishing returns drop sharply and token costs escalate rapidly.")

    with st.container(border=True):
        st.markdown("**Front**: What is the role of full jitter in exponential backoff?")
        st.caption("Flip for answer:")
        st.info("**Back**: It randomizes retry wait times across the backoff window to eliminate synchronized request waves (thundering herds).")

st.divider()

# ==============================================================================
# SECTION 13: SELF-ASSESSMENT CHECKLIST
# ==============================================================================
st.header("13. Self-Assessment Checklist", divider="rainbow")

st.markdown("Verify your engineering mastery of Reliability Techniques in LangChain:")

c1 = st.checkbox("I know how to attach with_retry() to LCEL runnables to handle transient 429 and 503 errors.", value=True)
c2 = st.checkbox("I understand the mathematics of exponential backoff with full jitter to avoid thundering herd spikes.", value=True)
c3 = st.checkbox("I can configure multi-provider fallback models using with_fallbacks() across OpenAI and Anthropic.", value=True)
c4 = st.checkbox("I know how to set explicit step-level timeouts to prevent hung worker threads and resource starvation.", value=True)
c5 = st.checkbox("I understand how to design self-healing repair loops that feed validation errors back into prompts.", value=True)
c6 = st.checkbox("I can implement input and output guardrails to enforce PII scrubbing and schema compliance.", value=True)

if c1 and c2 and c3 and c4 and c5 and c6:
    st.success("Mastery Certified: You have mastered Reliability Techniques, resilient LCEL architectures, and self-healing pipelines in LangChain.")
